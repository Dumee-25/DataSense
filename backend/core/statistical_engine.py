import pandas as pd, numpy as np, logging
from typing import Dict, Any
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

_LEAK_KW = (
    # ── Original post-event value indicators ─────────────────────────────────
    'result', 'outcome', 'final', 'actual', 'true_', 'real_',
    # ── Batch/operational metadata — causes the model to memorise production ─
    # artefacts instead of learning real signal (plate effects, instrument drift,
    # operator variance etc).  These are data-collection facts, not data features.
    'batch',          # batch_id, batch_number, batch_no, run_batch
    'plate_',         # plate_id, plate_no, plate_number  (lab / pharma)
    '_plate',         # well_plate, assay_plate
    'machine_id', 'machine_no', 'machine_num',   # manufacturing / lab
    'instrument_id', 'instrument_no',             # scientific instruments
    'operator_id', 'operator_no',                 # who ran the process
    'run_no', 'run_num', 'run_id',               # experiment / pipeline run
    'job_id', 'job_no', 'job_num',               # ETL / processing jobs
    'session_id',                                  # data-collection session
    'experiment_id', 'experiment_no',             # scientific experiment ref
    'pipeline_id',                                 # ETL pipeline reference
    'etl_',                                        # etl_batch, etl_load_dt
    'load_dt', 'ingest_dt', 'extract_dt',        # automated ETL timestamps
)
_SAMPLE_MAX_ROWS = 50_000  # cap for O(n²) operations like correlation matrix and Simpson's check
# Skewness above this threshold is almost never a natural distribution shape —
# it signals sentinel values, broken sensors, or severe data-entry errors.
_EXTREME_SKEW_THRESHOLD = 50

# Sentinel values to scrub — mirrors structural_analyzer._NUMERIC_SENTINELS.
# Defined here independently so StatisticalEngine is self-contained (no cross-module import).
# These are the classic placeholder values legacy databases and instruments use for "missing".
_SENTINEL_VALUES: frozenset = frozenset({
    -9999, -999, -99, -9, 9, 99, 999, 9999,
    -1,
    -1.0, 9999.0, -9999.0, 999.0, -999.0,
})
_SENTINEL_MIN_PCT = 0.5  # minimum % of non-null values that must match to trigger replacement


class StatisticalEngine:
    __slots__ = ()

    def analyze(self, df: pd.DataFrame, blueprint: dict) -> Dict[str, Any]:
        # ── Step 0: Scrub sentinel values BEFORE any statistics are computed ──────
        # Sentinel placeholders (-9999, 999, etc.) corrupt correlations, skewness,
        # variance, and outlier detection if left in place.  We work on a clean copy;
        # the caller's DataFrame is never mutated.
        df_clean, scrub_log = self._scrub_sentinels(df, blueprint)
        if scrub_log:
            logger.info(f"StatisticalEngine sentinel scrub: {scrub_log}")

        # ── Step 1: Downsample for O(n²) heavy operations only ───────────────────
        df_sample = (df_clean.sample(n=_SAMPLE_MAX_ROWS, random_state=42)
                     if len(df_clean) > _SAMPLE_MAX_ROWS else df_clean)
        if len(df_clean) > _SAMPLE_MAX_ROWS:
            logger.info(f"StatisticalEngine: downsampled {len(df_clean):,} → "
                        f"{_SAMPLE_MAX_ROWS:,} rows for correlation/Simpson's checks")

        nc   = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        cc   = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
        nc_s = df_sample.select_dtypes(include=[np.number]).columns.tolist()
        cc_s = df_sample.select_dtypes(include=['object', 'category']).columns.tolist()
        return {
            'red_flags': self._find_red_flags(df_clean, df_sample, blueprint, nc, cc, nc_s, cc_s),
            'warnings': self._find_warnings(df_sample, nc_s, cc_s),
            'patterns': self._detect_patterns(df_clean, nc),
            'correlations': self._compute_correlations(df_sample, nc_s),
            'outlier_summary': self._outlier_summary(df_clean, nc),
            'distribution_summary': self._distribution_summary(df_clean, nc),
            'sentinel_scrub': scrub_log,   # audit trail of what was replaced
        }

    def _scrub_sentinels(self, df: pd.DataFrame, blueprint: dict):
        """Return a sentinel-free copy of df and an audit log of replacements made.

        Two-pass strategy:
          Pass 1 — Blueprint-guided: use sentinel column→value pairs already
                   identified by StructuralAnalyzer (stored in blueprint quality_issues).
                   These are the most accurate because they already passed the
                   distribution-range check.
          Pass 2 — Safety-net scan: find any remaining numeric sentinels the blueprint
                   didn't catch (e.g. if called without a prior StructuralAnalyzer pass,
                   or if new columns appeared).  Uses the same range-outlier heuristic.

        The caller's df is never mutated — we always work on a copy.
        """
        df_clean = df.copy()
        scrub_log: Dict[str, Any] = {}   # col → {'values': [...], 'cells_replaced': int}

        # ── Pass 1: Blueprint-guided replacements ────────────────────────────────
        for issue in blueprint.get('quality_issues', []):
            if issue.get('type') != 'numeric_sentinel_values':
                continue
            col = issue.get('column', '')
            sentinels = issue.get('sentinel_values', [])
            if col not in df_clean.columns or not sentinels:
                continue
            before = df_clean[col].isnull().sum()
            df_clean[col] = df_clean[col].replace(sentinels, np.nan)
            replaced = int(df_clean[col].isnull().sum()) - int(before)
            if replaced > 0:
                scrub_log[col] = {'source': 'blueprint', 'values': sentinels,
                                  'cells_replaced': replaced}
                logger.debug(f"  Scrubbed {replaced} sentinel(s) from '{col}': {sentinels}")

        # ── Pass 2: Safety-net scan on columns not already scrubbed ─────────────
        already_scrubbed = set(scrub_log.keys())
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            if col in already_scrubbed:
                continue
            s = df_clean[col].dropna()
            if len(s) < 10:
                continue
            non_sentinel = s[~s.isin(_SENTINEL_VALUES)]
            if len(non_sentinel) < 5:
                continue
            p1, p99 = non_sentinel.quantile(0.01), non_sentinel.quantile(0.99)
            found = []
            for sv in _SENTINEL_VALUES:
                count = int((s == sv).sum())
                if count == 0:
                    continue
                if count / len(s) * 100 < _SENTINEL_MIN_PCT:
                    continue
                if sv < p1 or sv > p99:
                    found.append(sv)
            if found:
                before = df_clean[col].isnull().sum()
                df_clean[col] = df_clean[col].replace(found, np.nan)
                replaced = int(df_clean[col].isnull().sum()) - int(before)
                scrub_log[col] = {'source': 'safety_net', 'values': found,
                                  'cells_replaced': replaced}
                logger.debug(f"  Safety-net scrubbed {replaced} sentinel(s) from '{col}': {found}")

        return df_clean, scrub_log

    def _find_red_flags(self, df, df_sample, bp, nc, cc, nc_s, cc_s) -> list:
        flags = []
        for cand in bp.get('target_candidates', [])[:1]:
            col = cand['column']
            if col in df.columns and df[col].nunique() <= 20:
                maj = float(df[col].value_counts(normalize=True).iloc[0])
                if maj >= 0.85:
                    flags.append({
                        'type': 'class_imbalance', 'severity': 'critical', 'column': col,
                        'majority_pct': round(maj * 100, 1),
                        'message': f'Your target column "{col}" is severely imbalanced — {maj*100:.0f}% of rows belong to one class. A model trained on this will simply predict the majority class every time and look accurate while being completely useless.',
                        'plain_english': f'Imagine a fraud detector where 95% of transactions are legitimate. A model that always says "not fraud" is 95% accurate — but catches zero fraud. That\'s what will happen here without intervention.',
                        'recommendation': 'Use SMOTE to create synthetic minority samples, or set class_weight="balanced" in your model. Measure with F1-score or AUC-ROC, not accuracy.'})
        # Simpson's paradox uses the sampled df — it is an O(groups × n) correlation check
        if len(cc_s) >= 1 and len(nc_s) >= 2:
            p = self._check_simpsons_paradox(df_sample, nc_s, cc_s)
            if p: flags.append(p)
        flags.extend(self._check_data_leakage(df, bp))
        return flags

    def _check_simpsons_paradox(self, df, nc, cc) -> dict:
        try:
            if len(nc) < 2: return None
            a, b, cat = nc[0], nc[1], cc[0]
            oc, _ = sp_stats.pearsonr(df[a].dropna(), df[b].dropna())
            grps = df[cat].dropna().unique()
            if not 2 <= len(grps) <= 10: return None
            rev = sum(1 for g in grps
                      for sub in (df[df[cat] == g][[a, b]].dropna(),)
                      if len(sub) >= 10
                      for gc in (sp_stats.pearsonr(sub[a], sub[b])[0],)
                      if np.sign(gc) != np.sign(oc) and abs(gc) > 0.2)
            if rev >= 2:
                return {
                    'type': 'simpsons_paradox', 'severity': 'critical', 'columns': [a, b, cat],
                    'message': f'A statistical reversal was detected. The relationship between "{a}" and "{b}" flips direction when you split by "{cat}". Your overall analysis is misleading.',
                    'plain_english': f'Think of it like this: overall, taller people might earn more. But within each job type, taller people earn less. The "{cat}" variable is hiding the true story.',
                    'recommendation': f'Always analyze results separately for each group in "{cat}". Do not draw conclusions from the overall dataset without accounting for this split.'}
        except Exception as e:
            logger.warning(f"Simpson's paradox check failed for cols {nc[:2]}/{cc[:1]}: {e}")
        return None

    def _check_data_leakage(self, df, bp) -> list:
        tgt = bp.get('target_candidates', [])
        if not tgt: return []
        tc = tgt[0]['column']
        return [{'type': 'data_leakage_risk', 'severity': 'critical', 'column': c,
                 'message': f'Column "{c}" might be leaking information about your target "{tc}". Its name suggests it contains post-event data.',
                 'plain_english': 'Data leakage means your model is "cheating" — it\'s learning from information that wouldn\'t be available at prediction time. Your model will look great in testing but fail completely in production.',
                 'recommendation': f'Verify that "{c}" is genuinely available before the event you\'re predicting. If it\'s collected after, remove it immediately.'}
                for c in df.columns if c != tc and any(k in c.lower() for k in _LEAK_KW)]

    def _find_warnings(self, df, nc, cc) -> list:
        w = []
        if len(nc) >= 2:
            cm = df[nc].corr().abs()
            for i in range(len(nc)):
                for j in range(i + 1, len(nc)):
                    v = cm.iloc[i, j]
                    if v >= 0.95:
                        w.append({'type': 'multicollinearity', 'severity': 'high',
                            'var1': nc[i], 'var2': nc[j], 'correlation': round(float(v), 3),
                            'message': f'"{nc[i]}" and "{nc[j]}" are {v*100:.0f}% correlated — they are essentially measuring the same thing.',
                            'plain_english': 'Having both columns is like asking someone their height in cm and also in inches. It adds noise without adding new information, and confuses linear models.',
                            'recommendation': 'Remove one of these two columns. Keep the one that is easier to explain.'})
                    elif v >= 0.85:
                        w.append({'type': 'high_correlation', 'severity': 'medium',
                            'var1': nc[i], 'var2': nc[j], 'correlation': round(float(v), 3),
                            'message': f'"{nc[i]}" and "{nc[j]}" are strongly correlated ({v:.2f}).',
                            'plain_english': 'These two columns tend to move together. This is fine for tree-based models but will hurt linear regression and logistic regression.',
                            'recommendation': 'Consider combining them into one feature or using PCA to reduce dimensionality.'})
        for col in nc:
            s = df[col].dropna()
            if len(s) < 10: continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0: continue
            op = float(((s < q1 - 3*iqr) | (s > q3 + 3*iqr)).mean() * 100)
            if op >= 5:
                w.append({'type': 'high_outliers', 'severity': 'medium', 'column': col,
                    'percentage': round(op, 1),
                    'message': f'{op:.1f}% of values in "{col}" are extreme outliers.',
                    'plain_english': 'Outliers can drag model predictions in the wrong direction — like how one billionaire in a room raises the average salary dramatically even if everyone else earns minimum wage.',
                    'recommendation': f'Cap values in "{col}" at the 1st and 99th percentiles, or use a tree-based model which is naturally robust to outliers.'})
        if len(nc) >= 2:
            h = self._check_heteroscedasticity(df, nc)
            if h: w.append(h)
        for col in nc:
            s = df[col].dropna()
            if len(s) < 10: continue
            cv = s.std() / (abs(s.mean()) + 1e-10)
            if cv < 0.01 and s.nunique() > 1:
                w.append({'type': 'near_zero_variance', 'severity': 'medium', 'column': col,
                    'message': f'Column "{col}" has almost no variation — nearly all values are the same.',
                    'plain_english': 'A column that barely changes cannot teach a model anything. It\'s like trying to predict exam scores using everyone\'s age when everyone is 20.',
                    'recommendation': f'Consider dropping "{col}" — it is unlikely to contribute to model performance.'})
        # Fix 4: extreme skewness is not just a distribution shape — it is a data integrity alarm.
        # |skewness| > _EXTREME_SKEW_THRESHOLD almost always means sentinel values, broken sensors,
        # or severe data-entry errors have contaminated the column.
        for col in nc:
            s = df[col].dropna()
            if len(s) < 10: continue
            try:
                sk = float(s.skew())
            except Exception as e:
                logger.warning(f"Skewness computation failed for {col!r}: {e}")
                continue
            if abs(sk) < _EXTREME_SKEW_THRESHOLD: continue
            direction = 'right' if sk > 0 else 'left'
            extreme_val = float(s.max() if sk > 0 else s.min())
            w.append({
                'type': 'extreme_skewness', 'severity': 'high', 'column': col,
                'skewness': round(sk, 2),
                'message': (f'Column "{col}" has an extreme skewness of {sk:.1f} '
                            f'({direction}-skewed). This is far beyond any natural distribution — '
                            f'it almost certainly indicates data corruption, unhandled sentinel values, '
                            f'or broken instrument readings.'),
                'plain_english': (f'Normal skewed data (like salaries) has skewness between 1 and 10. '
                                  f'A skewness of {sk:.0f} in "{col}" is like someone sneaking a single '
                                  f'value of {extreme_val:,.0f} into a column where everything else '
                                  f'sits between {float(s.quantile(0.01)):,.1f} and '
                                  f'{float(s.quantile(0.99)):,.1f}. '
                                  f'That one rogue value is dragging every statistic in the column off a cliff.'),
                'recommendation': (f'First, check whether "{col}" contains sentinel/placeholder values '
                                   f'(e.g. -9999, 9999) and replace them with np.nan. '
                                   f'Then re-run the skewness check. If the extreme skewness persists, '
                                   f'investigate the raw data source for broken readings or entry errors '
                                   f'before applying any log transformation.'),
            })
        return w

    def _check_heteroscedasticity(self, df, nc) -> dict:
        try:
            if len(nc) < 2: return None
            col, tgt = nc[0], nc[-1]
            s = df[[col, tgt]].dropna()
            if len(s) < 30: return None
            x, y = s[col].values, s[tgt].values
            sl, ic, *_ = sp_stats.linregress(x, y)
            res = y - (sl * x + ic)
            med = np.median(x)
            lo, hi = res[x < med], res[x >= med]
            if len(lo) < 5 or len(hi) < 5: return None
            r = np.var(hi) / (np.var(lo) + 1e-10)
            if r > 4 or r < 0.25:
                return {'type': 'heteroscedasticity', 'severity': 'high', 'columns': [col, tgt],
                    'message': f'The spread of errors in "{tgt}" changes significantly across the range of "{col}". This violates a core assumption of linear regression.',
                    'plain_english': 'Think of predicting house prices by size. Small houses might be priced fairly consistently, but large mansions vary wildly. A linear model assumes consistent variation — this data breaks that.',
                    'recommendation': 'Apply a log transformation to the target variable, or switch to a tree-based model which does not make this assumption.'}
        except Exception as e:
            logger.warning(f"Heteroscedasticity check failed for {nc[0]!r}/{nc[-1]!r}: {e}")
        return None

    def _detect_patterns(self, df, nc) -> dict:
        skewed, extremely_skewed = [], []
        for c in nc:
            try:
                sk = abs(df[c].skew())
                if sk > _EXTREME_SKEW_THRESHOLD:
                    extremely_skewed.append(c)
                elif sk > 2:
                    skewed.append(c)
            except Exception:
                pass
        return {
            'has_numeric_features': len(nc) > 0, 'high_dimensional': len(df.columns) > 50,
            'large_dataset': len(df) > 100_000, 'small_dataset': len(df) < 500,
            'many_categoricals': len(df.select_dtypes(include='object').columns) > len(nc),
            'skewed_columns': skewed, 'has_skewed_features': len(skewed) > 0,
            'extremely_skewed_columns': extremely_skewed,
            'has_extreme_skewness': len(extremely_skewed) > 0,
        }

    def _compute_correlations(self, df, nc) -> list:
        if len(nc) < 2: return []
        try:
            cr = df[nc].corr()
            pairs = [{'col_a': nc[i], 'col_b': nc[j], 'correlation': round(float(cr.iloc[i, j]), 3),
                      'strength': self._corr_str(cr.iloc[i, j])}
                     for i in range(len(nc)) for j in range(i+1, len(nc)) if not np.isnan(cr.iloc[i, j])]
            return sorted(pairs, key=lambda x: abs(x['correlation']), reverse=True)[:20]
        except Exception as e:
            logger.warning(f"Correlation computation failed: {e}")
            return []

    @staticmethod
    def _corr_str(v) -> str:
        a = abs(v)
        return 'very strong' if a >= 0.9 else 'strong' if a >= 0.7 else 'moderate' if a >= 0.5 else 'weak' if a >= 0.3 else 'negligible'

    _correlation_strength = _corr_str  # backward compat

    def _outlier_summary(self, df, nc) -> list:
        out = []
        for col in nc[:15]:
            s = df[col].dropna()
            if len(s) < 10: continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0: continue
            n = int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum())
            if n: out.append({'column': col, 'outlier_count': n, 'outlier_pct': round(n / len(s) * 100, 2)})
        return sorted(out, key=lambda x: x['outlier_pct'], reverse=True)

    def _distribution_summary(self, df, nc) -> list:
        out = []
        for col in nc[:15]:
            s = df[col].dropna()
            if len(s) < 10: continue
            sk = float(s.skew())
            if abs(sk) > _EXTREME_SKEW_THRESHOLD:
                shape = f'critically skewed ({sk:+.0f}) — possible data corruption or sentinel values'
            elif sk > 2:   shape = 'heavily right-skewed'
            elif sk > 0.5: shape = 'right-skewed'
            elif sk < -2:  shape = 'heavily left-skewed'
            elif sk < -0.5: shape = 'left-skewed'
            else:           shape = 'roughly normal'
            out.append({'column': col, 'skewness': round(sk, 3), 'kurtosis': round(float(s.kurt()), 3),
                'shape': shape})
        return out