import pandas as pd, numpy as np, re
from typing import Dict, Any, List

_DISGUISED_MISSING = {
    '', 'na', 'n/a', 'null', 'none', 'nan', 'missing', '?', '-', '--',
    'undefined', 'not available', 'not applicable', '#n/a', '#ref!', '#value!',
}
_ID_KW = ('id', 'key', 'code', 'uuid', 'guid')
_TARGET_KW = ('target', 'label', 'class', 'output', 'result', 'outcome',
              'churn', 'fraud', 'default', 'price', 'sales', 'revenue', 'score', 'rating', 'y')

# Fix 1: Common numeric sentinel/placeholder values used by legacy databases,
# scientific instruments, and enterprise ETL systems to represent "missing".
# These pass right through isnull() and silently corrupt correlations and variance.
_NUMERIC_SENTINELS: frozenset = frozenset({
    -9999, -999, -99, -9, 9, 99, 999, 9999,   # generic fill values
    -1,                                          # common "not applicable" marker
    -1.0, 9999.0, -9999.0, 999.0, -999.0,      # float variants
})
# Minimum share of non-null values that must equal a sentinel to trigger the flag.
# 0.5% prevents noise on rare coincidences while catching genuine sentinel usage.
_SENTINEL_MIN_PCT = 0.5


def _is_str_like(s: pd.Series) -> bool:
    """True for both legacy object columns and modern pd.StringDtype columns.

    Pandas 1.0+ introduced a dedicated StringDtype.  Checking only ``dtype == object``
    silently misclassifies those columns as 'unknown', breaking categorical profiling,
    disguised-null detection, and whitespace checks.
    """
    return s.dtype == object or isinstance(s.dtype, pd.StringDtype)


class StructuralAnalyzer:
    __slots__ = ('_cache',)

    def analyze(self, df: pd.DataFrame, explicit_target: str = None) -> Dict[str, Any]:
        n = max(len(df), 1)
        self._cache = {
            col: (int(s.isnull().sum()), int(s.nunique()),
                  round(float(s.isnull().sum()) / n * 100, 2),
                  round(float(s.nunique()) / n * 100, 2))
            for col in df.columns for s in (df[col],)
        }

        # Build the heuristic candidate list, then promote the user-selected target if provided
        target_candidates = self._suggest_targets(df)
        if explicit_target and explicit_target in df.columns:
            # Remove it from wherever the heuristic placed it (avoid duplication)
            target_candidates = [c for c in target_candidates if c['column'] != explicit_target]
            # Build a proper candidate entry with full stats so downstream code works correctly
            col_s = df[explicit_target]
            nu = int(col_s.nunique())
            task = 'classification' if nu <= 20 else 'regression'
            ir = None
            if task == 'classification':
                vc = col_s.value_counts(normalize=True)
                ir = round(float(vc.iloc[0] / max(vc.iloc[-1], 1e-10)), 2)
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

    def _basic_info(self, df: pd.DataFrame) -> dict:
        r, c = df.shape
        total = r * c
        miss = int(df.isnull().sum().sum())
        return {
            'rows': r, 'columns': c, 'missing_cells': miss,
            'missing_percentage': round(miss / total * 100, 2) if total else 0.0,
            'duplicate_rows': int(df.duplicated().sum()),
            'memory_mb': round(float(df.memory_usage(deep=True).sum()) / 1048576, 3),
        }

    def _detect_structure(self, df: pd.DataFrame) -> dict:
        dt_cols = [c for c in df.columns if self._is_datetime_col(df[c])]
        id_cols = [c for c in df.columns if self._is_id_col(df[c], c)]
        stype = 'time-series' if dt_cols else ('panel' if len(id_cols) >= 2 else 'cross-sectional')
        return {
            'type': stype, 'datetime_columns': dt_cols, 'id_columns': id_cols,
            'numeric_count': int(df.select_dtypes(include=[np.number]).shape[1]),
            'categorical_count': int(df.select_dtypes(include=['object', 'category', 'string']).shape[1]),
            'boolean_count': int(df.select_dtypes(include=['bool']).shape[1]),
        }

    def _is_datetime_col(self, col: pd.Series) -> bool:
        if pd.api.types.is_datetime64_any_dtype(col):
            return True
        if _is_str_like(col):
            try:
                pd.to_datetime(col.dropna().head(20))
                return True
            except Exception:
                pass
        return False

    def _is_id_col(self, col: pd.Series, name: str) -> bool:
        return any(k in name.lower() for k in _ID_KW) or col.nunique() == len(col.dropna())

    def _profile_columns(self, df: pd.DataFrame) -> list:
        profiles = []
        for col in df.columns:
            s = df[col]
            nm, nu, mp, up = self._cache[col]
            p = {'name': col, 'dtype': str(s.dtype), 'kind': self._classify_column(s),
                 'missing_count': nm, 'missing_pct': mp, 'unique_count': nu, 'unique_pct': up,
                 'sample_values': self._sample_values(s)}
            if pd.api.types.is_numeric_dtype(s):
                d = s.describe()
                p.update({
                    'mean': self._sf(d.get('mean')), 'std': self._sf(d.get('std')),
                    'min': self._sf(d.get('min')), 'max': self._sf(d.get('max')),
                    'median': self._sf(s.median()), 'skewness': self._sf(s.skew()),
                    'kurtosis': self._sf(s.kurt()),
                    'zeros_pct': round(float((s == 0).mean() * 100), 2),
                    'negative_pct': round(float((s < 0).mean() * 100), 2),
                    'inf_count': int(np.isinf(s).sum()) if np.issubdtype(s.dtype, np.floating) else 0,
                })
            elif _is_str_like(s) or str(s.dtype) == 'category':
                ss = s.dropna().astype(str)
                p.update({
                    'top_values': {str(k): int(v) for k, v in s.value_counts().head(5).items()},
                    'avg_length': self._sf(ss.str.len().mean()),
                    'whitespace_issues': int(ss.str.strip().ne(ss).sum()),
                    'disguised_missing': int(ss.str.strip().str.lower().isin(_DISGUISED_MISSING).sum()),
                })
            profiles.append(p)
        return profiles

    def _sample_values(self, s: pd.Series, n: int = 5) -> list:
        nn = s.dropna()
        return [self._to_jsonable(v) for v in nn.sample(min(n, len(nn)), random_state=0).tolist()] if len(nn) else []

    def _classify_column(self, s: pd.Series) -> str:
        if pd.api.types.is_bool_dtype(s): return 'boolean'
        if pd.api.types.is_datetime64_any_dtype(s): return 'datetime'
        if pd.api.types.is_numeric_dtype(s):
            nu = s.nunique()
            return 'numeric_categorical' if nu <= 10 and nu / max(len(s), 1) < 0.05 else 'numeric'
        if _is_str_like(s):
            nu = s.nunique()
            if self._is_mixed_type(s): return 'mixed_type'
            if nu <= 20: return 'categorical_low'
            if nu / max(len(s), 1) > 0.9: return 'high_cardinality'
            return 'categorical'
        if str(s.dtype) == 'category': return 'categorical_low'
        return 'unknown'

    def _is_mixed_type(self, s: pd.Series) -> bool:
        if not _is_str_like(s): return False
        samp = s.dropna().head(200)
        if not len(samp): return False
        r = samp.apply(lambda v: isinstance(v, str) and self._looks_numeric(v)).sum() / len(samp)
        return 0.1 < r < 0.9

    @staticmethod
    def _looks_numeric(val: str) -> bool:
        try: float(val.replace(',', '')); return True
        except (ValueError, AttributeError): return False

    def _detect_quality_issues(self, df: pd.DataFrame) -> list:
        issues = []
        n = max(len(df), 1)
        # Fix 2: include modern StringDtype as well as legacy object columns
        obj_cols = df.select_dtypes(include=['object', 'string']).columns
        # Missingness + constant in single pass
        for col in df.columns:
            pct = df[col].isnull().mean() * 100
            if pct >= 50:
                issues.append({'type': 'high_missingness', 'severity': 'high', 'column': col,
                    'percentage': round(pct, 1),
                    'message': f'Column "{col}" is {pct:.0f}% empty — may not be useful for modeling.',
                    'recommendation': f'Consider dropping "{col}" or imputing if it carries meaningful signal.'})
            elif pct >= 20:
                issues.append({'type': 'moderate_missingness', 'severity': 'medium', 'column': col,
                    'percentage': round(pct, 1), 'message': f'Column "{col}" has {pct:.0f}% missing values.',
                    'recommendation': 'Impute with median (numeric) or mode (categorical), or use a model that handles missingness.'})
            if df[col].nunique() <= 1:
                issues.append({'type': 'constant_column', 'severity': 'high', 'column': col,
                    'message': f'Column "{col}" has only one unique value — it carries no information.',
                    'recommendation': f'Drop "{col}" — it will not help any model learn.'})
        # Fix 1: scan numeric columns for sentinel placeholder values
        issues.extend(self._detect_numeric_sentinels(df, n))
        # High cardinality + mixed types + disguised missing + whitespace (single pass over string cols)
        for col in obj_cols:
            s = df[col]; nu = s.nunique(); sv = s.dropna().astype(str)
            if nu / n > 0.9 and nu > 100:
                issues.append({'type': 'high_cardinality', 'severity': 'medium', 'column': col,
                    'unique_count': int(nu), 'message': f'Column "{col}" has {nu} unique values — likely a free-text or ID field.',
                    'recommendation': 'Drop this column or apply target encoding / embedding before modeling.'})
            if self._is_mixed_type(s):
                issues.append({'type': 'mixed_types', 'severity': 'high', 'column': col,
                    'message': f'Column "{col}" contains a mix of numeric and non-numeric values. This usually means dirty data — numbers stored as text alongside error codes or labels.',
                    'recommendation': f'Investigate the non-numeric values in "{col}". Clean or coerce to numeric with pd.to_numeric(errors="coerce"), then handle the resulting NaNs.'})
            lo = sv.str.strip().str.lower()
            dc = int(lo.isin(_DISGUISED_MISSING).sum())
            if dc:
                dp = dc / n * 100
                issues.append({'type': 'disguised_missing', 'severity': 'medium' if dp < 5 else 'high',
                    'column': col, 'count': dc, 'percentage': round(dp, 1),
                    'message': f'Column "{col}" has {dc} values that look like disguised nulls (e.g. "NA", "null", "?"). These are NOT detected by isnull().',
                    'recommendation': f'Replace these sentinel values with np.nan before any analysis: df["{col}"].replace({list(_DISGUISED_MISSING)[:5]}, np.nan, inplace=True)'})
            nw = int(sv.str.strip().ne(sv).sum())
            if nw:
                issues.append({'type': 'whitespace_issues', 'severity': 'low', 'column': col, 'count': nw,
                    'message': f'{nw} values in "{col}" have leading/trailing whitespace. This causes silent join failures and inflated unique counts.',
                    'recommendation': f'Strip whitespace: df["{col}"] = df["{col}"].str.strip()'})
        # Duplicates
        dc = df.duplicated().sum()
        if dc:
            dp = dc / len(df) * 100
            issues.append({'type': 'duplicate_rows', 'severity': 'medium' if dp < 5 else 'high',
                'count': int(dc), 'percentage': round(dp, 1),
                'message': f'{dc} duplicate rows found ({dp:.1f}% of data).',
                'recommendation': 'Remove duplicates before training — they artificially inflate model performance.'})
        # Infinite values
        for col in df.select_dtypes(include=[np.number]).columns:
            if np.issubdtype(df[col].dtype, np.floating):
                ni = int(np.isinf(df[col]).sum())
                if ni:
                    issues.append({'type': 'infinite_values', 'severity': 'high', 'column': col, 'count': ni,
                        'message': f'Column "{col}" contains {ni} infinite value(s). Most ML models cannot handle inf/-inf and will crash or produce NaN predictions.',
                        'recommendation': f'Replace infinities: df["{col}"].replace([np.inf, -np.inf], np.nan) then impute or drop.'})
        return issues

    def _detect_numeric_sentinels(self, df: pd.DataFrame, n: int) -> list:
        """Scan numeric columns for hardcoded placeholder values that masquerade as real data.

        Legacy databases and instruments often store -9999, 999, -1 etc. instead of NULL.
        These silent sentinels pass through isnull() unchecked, inflating mean/variance and
        warping correlation matrices.  We flag a sentinel when it appears in >= _SENTINEL_MIN_PCT
        of non-null values AND is a statistical outlier relative to the column's own distribution.
        """
        issues = []
        for col in df.select_dtypes(include=[np.number]).columns:
            s = df[col].dropna()
            if len(s) < 10:
                continue
            # Establish the column's "normal" range excluding known sentinel candidates
            non_sentinel = s[~s.isin(_NUMERIC_SENTINELS)]
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
                    'recommendation': (f'Replace these values with np.nan before any analysis: '
                                       f'df["{col}"].replace({[v for v,_,_ in found]}, np.nan, inplace=True) '
                                       f'— then re-run your missing-value imputation step.'),
                })
        return issues

    def _suggest_targets(self, df: pd.DataFrame) -> list:
        cands = []
        last_col = df.columns[-1] if len(df.columns) else None
        for col in df.columns:
            sc, reasons, cl = 0, [], col.lower()
            nm, nu, _, _ = self._cache[col]
            if any(k in cl for k in _TARGET_KW): sc += 3; reasons.append('name suggests it is a target variable')
            if col == last_col: sc += 1; reasons.append('last column in dataset')
            if pd.api.types.is_numeric_dtype(df[col]):
                if nu == 2: sc += 2; reasons.append('binary numeric (likely 0/1 classification target)')
                elif nu <= 10: sc += 1; reasons.append('low cardinality numeric')
            elif _is_str_like(df[col]) and nu <= 10: sc += 2; reasons.append('low cardinality categorical')
            if sc >= 2:
                task = 'classification' if nu <= 10 else 'regression'
                ir = None
                if task == 'classification':
                    vc = df[col].value_counts(normalize=True)
                    ir = round(float(vc.iloc[0] / max(vc.iloc[-1], 1e-10)), 2)
                cands.append({'column': col, 'score': sc, 'task_type': task, 'n_classes': nu,
                              'imbalance_ratio': ir, 'reasons': reasons})
        return sorted(cands, key=lambda x: x['score'], reverse=True)[:3]

    def _dtype_recommendations(self, df: pd.DataFrame) -> List[dict]:
        recs = []
        for col in df.columns:
            s, (_, nu, _, _) = df[col], self._cache[col]
            if _is_str_like(s) and 0 < nu <= 50:
                recs.append({'column': col, 'current_dtype': str(s.dtype), 'suggested_dtype': 'category',
                             'reason': f'Only {nu} unique values — category type saves memory.'})
            if pd.api.types.is_float_dtype(s):
                nn = s.dropna()
                if len(nn) and (nn == nn.astype(int)).all():
                    recs.append({'column': col, 'current_dtype': str(s.dtype),
                                 'suggested_dtype': 'Int64 (nullable integer)',
                                 'reason': 'All non-null values are whole numbers stored as float.'})
            if pd.api.types.is_integer_dtype(s):
                lo, hi = s.min(), s.max()
                for dt in ('int8', 'int16'):
                    info = np.iinfo(dt)
                    if info.min <= lo and hi <= info.max and str(s.dtype) != dt:
                        recs.append({'column': col, 'current_dtype': str(s.dtype), 'suggested_dtype': dt,
                                     'reason': f'Range [{lo}, {hi}] fits in {dt} — saves memory.'}); break
        return recs

    def _check_column_names(self, df: pd.DataFrame) -> List[dict]:
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
        if isinstance(val, pd.Timestamp): return val.isoformat()
        return val