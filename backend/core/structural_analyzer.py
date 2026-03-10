import polars as pl, polars.selectors as cs, numpy as np, re
from datetime import datetime, date
from typing import Dict, Any, List

_DISGUISED_MISSING = {
    '', 'na', 'n/a', 'null', 'none', 'nan', 'missing', '?', '-', '--',
    'undefined', 'not available', 'not applicable', '#n/a', '#ref!', '#value!',
}
_ID_KW = ('id', 'key', 'code', 'uuid', 'guid')
_TARGET_KW = ('target', 'label', 'class', 'output', 'result', 'outcome',
              'churn', 'fraud', 'default', 'price', 'sales', 'revenue', 'score', 'rating', 'y',
              'surviv', 'diagnosis', 'status', 'approved', 'accept', 'reject',
              'spam', 'sentiment', 'category', 'species', 'quality', 'risk',
              'severity', 'grade', 'pass', 'fail', 'positive', 'negative',
              'attrition', 'readmit', 'clicked', 'purchased', 'converted')

# Fix 1: Common numeric sentinel/placeholder values used by legacy databases,
# scientific instruments, and enterprise ETL systems to represent "missing".
# These pass right through null_count() and silently corrupt correlations and variance.
_NUMERIC_SENTINELS: frozenset = frozenset({
    -9999, -999, -99, -9, 9, 99, 999, 9999,   # generic fill values
    -1,                                          # common "not applicable" marker
    -1.0, 9999.0, -9999.0, 999.0, -999.0,      # float variants
})
# Minimum share of non-null values that must equal a sentinel to trigger the flag.
# 0.5% prevents noise on rare coincidences while catching genuine sentinel usage.
_SENTINEL_MIN_PCT = 0.5


def _is_str_like(s: pl.Series) -> bool:
    """True for polars String/Utf8 columns."""
    return s.dtype in (pl.String, pl.Utf8)


class StructuralAnalyzer:
    __slots__ = ('_cache',)

    def analyze(self, df: pl.DataFrame, explicit_target: str = None) -> Dict[str, Any]:
        # Normalize NaN → null for consistent null handling
        float_cols = df.select(cs.float()).columns
        if float_cols:
            df = df.with_columns([pl.col(c).fill_nan(None) for c in float_cols])

        n = max(len(df), 1)
        self._cache = {
            col: (int(s.null_count()), int(s.drop_nulls().n_unique()),
                  round(float(s.null_count()) / n * 100, 2),
                  round(float(s.drop_nulls().n_unique()) / n * 100, 2))
            for col in df.columns for s in (df[col],)
        }

        # Build the heuristic candidate list, then promote the user-selected target if provided
        target_candidates = self._suggest_targets(df)
        if explicit_target and explicit_target in df.columns:
            # Remove it from wherever the heuristic placed it (avoid duplication)
            target_candidates = [c for c in target_candidates if c['column'] != explicit_target]
            # Build a proper candidate entry with full stats so downstream code works correctly
            col_s = df[explicit_target]
            nu = int(col_s.drop_nulls().n_unique())
            task = 'classification' if nu <= 20 else 'regression'
            ir = None
            if task == 'classification':
                vc = col_s.value_counts().sort("count", descending=True)
                total = vc["count"].sum()
                if len(vc) >= 2:
                    ir = round(float((vc["count"][0] / total) / max(vc["count"][-1] / total, 1e-10)), 2)
            explicit_entry = {
                'column': explicit_target, 'score': 999,  # always wins sorting
                'task_type': task, 'n_classes': nu,
                'imbalance_ratio': ir,
                'reasons': ['user selected via API'],
            }
            target_candidates.insert(0, explicit_entry)

        bp = {
            'basic_info': self._basic_info(df),
            'data_structure': self._detect_structure(df),
            'column_profiles': self._profile_columns(df),
            'quality_issues': self._detect_quality_issues(df),
            'target_candidates': target_candidates,
            'dtype_recommendations': self._dtype_recommendations(df),
            'column_name_issues': self._check_column_names(df),
        }
        del self._cache
        return bp

    def _basic_info(self, df: pl.DataFrame) -> dict:
        r, c = df.shape
        total = r * c
        miss = sum(df[col].null_count() for col in df.columns)
        return {
            'rows': r, 'columns': c, 'missing_cells': miss,
            'missing_percentage': round(miss / total * 100, 2) if total else 0.0,
            'duplicate_rows': int(df.is_duplicated().sum()),
            'memory_mb': round(float(df.estimated_size("b")) / 1048576, 3),
        }

    def _detect_structure(self, df: pl.DataFrame) -> dict:
        dt_cols = [c for c in df.columns if self._is_datetime_col(df[c])]
        id_cols = [c for c in df.columns if self._is_id_col(df[c], c)]
        stype = 'time-series' if dt_cols else ('panel' if len(id_cols) >= 2 else 'cross-sectional')
        return {
            'type': stype, 'datetime_columns': dt_cols, 'id_columns': id_cols,
            'numeric_count': df.select(cs.numeric()).shape[1],
            'categorical_count': df.select(cs.string() | cs.categorical()).shape[1],
            'boolean_count': df.select(cs.boolean()).shape[1],
        }

    def _is_datetime_col(self, col: pl.Series) -> bool:
        if col.dtype.is_temporal():
            return True
        if _is_str_like(col):
            try:
                sample = col.drop_nulls().head(20)
                if len(sample) == 0:
                    return False
                sample.str.to_datetime(strict=True)
                return True
            except Exception:
                pass
        return False

    def _is_id_col(self, col: pl.Series, name: str) -> bool:
        nn = col.drop_nulls()
        return any(k in name.lower() for k in _ID_KW) or (len(nn) > 0 and nn.n_unique() == len(nn))

    def _profile_columns(self, df: pl.DataFrame) -> list:
        profiles = []
        for col in df.columns:
            s = df[col]
            nm, nu, mp, up = self._cache[col]
            p = {'name': col, 'dtype': str(s.dtype), 'kind': self._classify_column(s),
                 'missing_count': nm, 'missing_pct': mp, 'unique_count': nu, 'unique_pct': up,
                 'sample_values': self._sample_values(s)}
            if s.dtype.is_numeric():
                p.update({
                    'mean': self._sf(s.mean()), 'std': self._sf(s.std()),
                    'min': self._sf(s.min()), 'max': self._sf(s.max()),
                    'median': self._sf(s.median()), 'skewness': self._sf(s.skew()),
                    'kurtosis': self._sf(s.kurtosis()),
                    'zeros_pct': round(float((s == 0).mean() * 100), 2),
                    'negative_pct': round(float((s < 0).mean() * 100), 2),
                    'inf_count': int(s.is_infinite().sum()) if s.dtype.is_float() else 0,
                })
            elif _is_str_like(s) or s.dtype == pl.Categorical:
                ss = s.drop_nulls().cast(pl.String)
                vc = s.value_counts().sort("count", descending=True).head(5)
                top_values = {}
                for row in vc.iter_rows():
                    top_values[str(row[0])] = int(row[1])
                p.update({
                    'top_values': top_values,
                    'avg_length': self._sf(ss.str.len_chars().mean()),
                    'whitespace_issues': int((ss.str.strip_chars() != ss).sum()),
                    'disguised_missing': int(ss.str.strip_chars().str.to_lowercase().is_in(list(_DISGUISED_MISSING)).sum()),
                })
            profiles.append(p)
        return profiles

    def _sample_values(self, s: pl.Series, n: int = 5) -> list:
        nn = s.drop_nulls()
        return [self._to_jsonable(v) for v in nn.sample(n=min(n, len(nn)), seed=0).to_list()] if len(nn) else []

    def _classify_column(self, s: pl.Series) -> str:
        if s.dtype == pl.Boolean: return 'boolean'
        if s.dtype.is_temporal(): return 'datetime'
        if s.dtype.is_numeric():
            nu = s.drop_nulls().n_unique()
            return 'numeric_categorical' if nu <= 10 and nu / max(len(s), 1) < 0.05 else 'numeric'
        if _is_str_like(s):
            nu = s.drop_nulls().n_unique()
            if self._is_mixed_type(s): return 'mixed_type'
            if nu <= 20: return 'categorical_low'
            if nu / max(len(s), 1) > 0.9: return 'high_cardinality'
            return 'categorical'
        if s.dtype == pl.Categorical: return 'categorical_low'
        return 'unknown'

    def _is_mixed_type(self, s: pl.Series) -> bool:
        if not _is_str_like(s): return False
        samp = s.drop_nulls().head(200)
        if not len(samp): return False
        r = samp.map_elements(lambda v: isinstance(v, str) and self._looks_numeric(v), return_dtype=pl.Boolean).sum() / len(samp)
        return 0.1 < r < 0.9

    @staticmethod
    def _looks_numeric(val: str) -> bool:
        try: float(val.replace(',', '')); return True
        except (ValueError, AttributeError): return False

    def _detect_quality_issues(self, df: pl.DataFrame) -> list:
        issues = []
        n = max(len(df), 1)
        obj_cols = df.select(cs.string()).columns
        # Missingness + constant in single pass
        for col in df.columns:
            pct = df[col].null_count() / n * 100
            if pct >= 50:
                issues.append({'type': 'high_missingness', 'severity': 'high', 'column': col,
                    'percentage': round(pct, 1),
                    'message': f'Column "{col}" is {pct:.0f}% empty — may not be useful for modeling.',
                    'recommendation': f'Consider dropping "{col}" or imputing if it carries meaningful signal.'})
            elif pct >= 20:
                issues.append({'type': 'moderate_missingness', 'severity': 'medium', 'column': col,
                    'percentage': round(pct, 1), 'message': f'Column "{col}" has {pct:.0f}% missing values.',
                    'recommendation': 'Impute with median (numeric) or mode (categorical), or use a model that handles missingness.'})
            if df[col].drop_nulls().n_unique() <= 1:
                issues.append({'type': 'constant_column', 'severity': 'high', 'column': col,
                    'message': f'Column "{col}" has only one unique value — it carries no information.',
                    'recommendation': f'Drop "{col}" — it will not help any model learn.'})
        # Fix 1: scan numeric columns for sentinel placeholder values
        issues.extend(self._detect_numeric_sentinels(df, n))
        # High cardinality + mixed types + disguised missing + whitespace (single pass over string cols)
        for col in obj_cols:
            s = df[col]; nu = s.drop_nulls().n_unique(); sv = s.drop_nulls().cast(pl.String)
            if nu / n > 0.9 and nu > 100:
                issues.append({'type': 'high_cardinality', 'severity': 'medium', 'column': col,
                    'unique_count': int(nu), 'message': f'Column "{col}" has {nu} unique values — likely a free-text or ID field.',
                    'recommendation': 'Drop this column or apply target encoding / embedding before modeling.'})
            if self._is_mixed_type(s):
                issues.append({'type': 'mixed_types', 'severity': 'high', 'column': col,
                    'message': f'Column "{col}" contains a mix of numeric and non-numeric values. This usually means dirty data — numbers stored as text alongside error codes or labels.',
                    'recommendation': f'Investigate the non-numeric values in "{col}". Clean or coerce to numeric, then handle the resulting nulls.'})
            lo = sv.str.strip_chars().str.to_lowercase()
            dc = int(lo.is_in(list(_DISGUISED_MISSING)).sum())
            if dc:
                dp = dc / n * 100
                issues.append({'type': 'disguised_missing', 'severity': 'medium' if dp < 5 else 'high',
                    'column': col, 'count': dc, 'percentage': round(dp, 1),
                    'message': f'Column "{col}" has {dc} values that look like disguised nulls (e.g. "NA", "null", "?"). These are NOT detected by null_count().',
                    'recommendation': f'Replace these sentinel values with null before any analysis.'})
            nw = int((sv.str.strip_chars() != sv).sum())
            if nw:
                issues.append({'type': 'whitespace_issues', 'severity': 'low', 'column': col, 'count': nw,
                    'message': f'{nw} values in "{col}" have leading/trailing whitespace. This causes silent join failures and inflated unique counts.',
                    'recommendation': f'Strip whitespace: df = df.with_columns(pl.col("{col}").str.strip_chars())'})
        # Duplicates
        dc = df.is_duplicated().sum()
        if dc:
            dp = dc / len(df) * 100
            issues.append({'type': 'duplicate_rows', 'severity': 'medium' if dp < 5 else 'high',
                'count': int(dc), 'percentage': round(dp, 1),
                'message': f'{dc} duplicate rows found ({dp:.1f}% of data).',
                'recommendation': 'Remove duplicates before training — they artificially inflate model performance.'})
        # Infinite values
        for col in df.select(cs.numeric()).columns:
            if df[col].dtype.is_float():
                ni = int(df[col].is_infinite().sum())
                if ni:
                    issues.append({'type': 'infinite_values', 'severity': 'high', 'column': col, 'count': ni,
                        'message': f'Column "{col}" contains {ni} infinite value(s). Most ML models cannot handle inf/-inf and will crash or produce NaN predictions.',
                        'recommendation': f'Replace infinities with null, then impute or drop.'})
        return issues

    def _detect_numeric_sentinels(self, df: pl.DataFrame, n: int) -> list:
        """Scan numeric columns for hardcoded placeholder values that masquerade as real data.

        Legacy databases and instruments often store -9999, 999, -1 etc. instead of NULL.
        These silent sentinels pass through null_count() unchecked, inflating mean/variance and
        warping correlation matrices.  We flag a sentinel when it appears in >= _SENTINEL_MIN_PCT
        of non-null values AND is a statistical outlier relative to the column's own distribution.
        """
        issues = []
        for col in df.select(cs.numeric()).columns:
            s = df[col].drop_nulls()
            if len(s) < 10:
                continue
            # Establish the column's "normal" range excluding known sentinel candidates
            non_sentinel = s.filter(~s.is_in(list(_NUMERIC_SENTINELS)))
            if len(non_sentinel) < 5:
                continue  # column is almost entirely sentinels — will be caught as constant/high-missing
            p1, p99 = non_sentinel.quantile(0.01), non_sentinel.quantile(0.99)
            found = []
            for sentinel in _NUMERIC_SENTINELS:
                count = int((s == sentinel).sum())
                if count == 0:
                    continue
                pct = count / len(s) * 100
                if pct < _SENTINEL_MIN_PCT:
                    continue
                # Only flag if the sentinel value is clearly outside the column's real range
                if sentinel < p1 or sentinel > p99:
                    found.append((sentinel, count, round(pct, 1)))
            if found:
                examples = ', '.join(f'{v} ({c} rows, {p:.1f}%)' for v, c, p in found[:3])
                issues.append({
                    'type': 'numeric_sentinel_values', 'severity': 'high', 'column': col,
                    'sentinel_values': [v for v, _, _ in found],
                    'message': (f'Column "{col}" contains suspicious placeholder values that are likely '
                                f'encoded missing data: {examples}.'),
                    'plain_english': (f'Think of it like a form where someone writes "9999" when they '
                                      f'don\'t know the answer instead of leaving it blank. '
                                      f'"{col}" has values like {[v for v,_,_ in found[:2]]} that look like '
                                      f'"we didn\'t record this" codes — they\'re not real measurements. '
                                      f'If left in, they will massively warp the average and '
                                      f'create fake correlations with other columns.'),
                    'recommendation': (f'Replace these values with null before any analysis, '
                                       f'then re-run your missing-value imputation step.'),
                })
        return issues

    def _suggest_targets(self, df: pl.DataFrame) -> list:
        cands = []
        last_col = df.columns[-1] if len(df.columns) else None
        for col in df.columns:
            sc, reasons, cl = 0, [], col.lower()
            nm, nu, _, _ = self._cache[col]
            if any(k in cl for k in _TARGET_KW): sc += 3; reasons.append('name suggests it is a target variable')
            if col == last_col: sc += 1; reasons.append('last column in dataset')
            if df[col].dtype.is_numeric():
                if nu == 2: sc += 2; reasons.append('binary numeric (likely 0/1 classification target)')
                elif nu <= 10: sc += 1; reasons.append('low cardinality numeric')
            elif _is_str_like(df[col]) and nu <= 10: sc += 2; reasons.append('low cardinality categorical')
            if sc >= 2:
                task = 'classification' if nu <= 10 else 'regression'
                ir = None
                if task == 'classification':
                    vc = df[col].value_counts().sort("count", descending=True)
                    total = vc["count"].sum()
                    if len(vc) >= 2:
                        ir = round(float((vc["count"][0] / total) / max(vc["count"][-1] / total, 1e-10)), 2)
                    elif len(vc) == 1:
                        ir = 1.0
                cands.append({'column': col, 'score': sc, 'task_type': task, 'n_classes': nu,
                              'imbalance_ratio': ir, 'reasons': reasons})
        return sorted(cands, key=lambda x: x['score'], reverse=True)[:3]

    def _dtype_recommendations(self, df: pl.DataFrame) -> List[dict]:
        recs = []
        for col in df.columns:
            s, (_, nu, _, _) = df[col], self._cache[col]
            if _is_str_like(s) and 0 < nu <= 50:
                recs.append({'column': col, 'current_dtype': str(s.dtype), 'suggested_dtype': 'category',
                             'reason': f'Only {nu} unique values — category type saves memory.'})
            if s.dtype.is_float():
                nn = s.drop_nulls()
                if len(nn) and (nn % 1 == 0).all():
                    recs.append({'column': col, 'current_dtype': str(s.dtype),
                                 'suggested_dtype': 'Int64 (nullable integer)',
                                 'reason': 'All non-null values are whole numbers stored as float.'})
            if s.dtype.is_integer():
                lo, hi = s.min(), s.max()
                for dt in ('int8', 'int16'):
                    info = np.iinfo(dt)
                    if info.min <= lo and hi <= info.max and str(s.dtype).lower() != dt:
                        recs.append({'column': col, 'current_dtype': str(s.dtype), 'suggested_dtype': dt,
                                     'reason': f'Range [{lo}, {hi}] fits in {dt} — saves memory.'}); break
        return recs

    def _check_column_names(self, df: pl.DataFrame) -> List[dict]:
        issues, seen = [], {}
        for col in df.columns:
            low = col.strip().lower()
            if low in seen:
                issues.append({'type': 'duplicate_after_lowering', 'columns': [seen[low], col],
                    'message': f'"{seen[low]}" and "{col}" differ only by case — will collide in case-insensitive systems.'})
            seen[low] = col
            if re.search(r'[^\w]', col):
                issues.append({'type': 'special_characters', 'column': col,
                    'message': f'Column "{col}" contains spaces or special characters — may cause issues in some frameworks.',
                    'suggestion': re.sub(r'[^\w]+', '_', col).strip('_').lower()})
            if col != col.strip():
                issues.append({'type': 'whitespace_in_name', 'column': col,
                    'message': f'Column name has leading/trailing whitespace: "{repr(col)}".'})
        return issues

    def _sf(self, val) -> float:
        try:
            f = float(val)
            return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
        except Exception: return None

    _safe_float = _sf  # backward compat alias

    @staticmethod
    def _to_jsonable(val):
        if isinstance(val, np.integer): return int(val)
        if isinstance(val, np.floating): return round(float(val), 4)
        if isinstance(val, np.bool_): return bool(val)
        if isinstance(val, (datetime, date)): return val.isoformat()
        return val
