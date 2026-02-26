"""AggregationEngine — groups identical/similar findings before output.

Instead of 50 separate "high correlation" warnings, this produces:
  "The following 15 variable pairs are highly correlated and capture redundant
   information: [list]. Consider applying PCA or dropping them."

This drastically reduces document bloat and makes the output actionable.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

# Maximum number of individual items to list before summarizing
_MAX_INDIVIDUAL = 3
# If more than this many items share a type, aggregate
_AGGREGATION_THRESHOLD = 3


class AggregationEngine:
    """Groups duplicate/similar findings into concise aggregated blocks."""
    __slots__ = ()

    def aggregate(self, issues: list, model_name: str = '') -> list:
        """Groups identical finding types into concise summaries.

        Args:
            issues: List of raw issue dicts (from quality_issues + red_flags + warnings)
            model_name: The recommended model name, used for relevance hints

        Returns:
            Aggregated list where repetitive findings are grouped into single entries.
        """
        if not issues:
            return []

        # Group by type
        by_type: Dict[str, list] = defaultdict(list)
        for issue in issues:
            by_type[issue.get('type', 'unknown')].append(issue)

        aggregated = []
        for issue_type, group in by_type.items():
            if len(group) >= _AGGREGATION_THRESHOLD:
                merged = self._merge_group(issue_type, group)
                if merged:
                    aggregated.append(merged)
                    continue
            # Below threshold — keep individual items
            aggregated.extend(group)

        # Re-sort by severity weight
        sev_order = {'critical': 0, 'high': 1, 'medium': 2}
        return sorted(aggregated, key=lambda i: sev_order.get(i.get('severity', 'medium'), 3))

    def _merge_group(self, issue_type: str, group: list) -> Optional[dict]:
        """Merge a group of identical-type issues into one aggregated entry."""
        handler = _MERGE_HANDLERS.get(issue_type)
        if handler:
            return handler(self, group)
        # Generic fallback merging
        return self._generic_merge(issue_type, group)

    # ── Type-specific merge handlers ──────────────────────────────────────

    def _merge_high_correlation(self, group: list) -> dict:
        """Aggregate multiple high-correlation warnings into one block."""
        pairs = []
        for item in group:
            v1, v2 = item.get('var1', ''), item.get('var2', '')
            corr = item.get('correlation', 0)
            if v1 and v2:
                pairs.append((v1, v2, corr))
        # Deduplicate columns involved
        all_cols = sorted(set(c for p in pairs for c in (p[0], p[1])))
        shown_pairs = pairs[:_MAX_INDIVIDUAL]
        hidden = len(pairs) - len(shown_pairs)

        pair_text = ", ".join(f'"{a}"↔"{b}" ({c:+.2f})' for a, b, c in shown_pairs)
        if hidden > 0:
            pair_text += f" and {hidden} more pair{'s' if hidden > 1 else ''}"

        severity = group[0].get('severity', 'medium')
        return {
            'type': 'high_correlation',
            'severity': severity,
            'aggregated': True,
            'count': len(pairs),
            'columns': all_cols,
            'column': f"{len(all_cols)} columns",
            'pairs': [{'var1': a, 'var2': b, 'correlation': c} for a, b, c in pairs],
            'message': (
                f'{len(pairs)} pairs of columns are strongly correlated, involving '
                f'{len(all_cols)} variables: {", ".join(f"{c!r}" for c in all_cols[:8])}'
                f'{"..." if len(all_cols) > 8 else ""}. '
                f'These capture redundant information.'
            ),
            'plain_english': (
                f'Your dataset has {len(pairs)} pairs of columns that are essentially '
                f'measuring the same thing in different ways. Key pairs: {pair_text}. '
                f'Having all of them adds noise without adding new information.'
            ),
            'recommendation': (
                f'Consider applying PCA to reduce these {len(all_cols)} correlated variables '
                f'into uncorrelated components, or manually drop one column from each pair. '
                f'For tree-based models, this is less critical but still reduces training time.'
            ),
        }

    def _merge_multicollinearity(self, group: list) -> dict:
        """Aggregate multicollinearity warnings."""
        pairs = []
        for item in group:
            v1, v2 = item.get('var1', ''), item.get('var2', '')
            corr = item.get('correlation', 0)
            if v1 and v2:
                pairs.append((v1, v2, corr))
        all_cols = sorted(set(c for p in pairs for c in (p[0], p[1])))

        return {
            'type': 'multicollinearity',
            'severity': 'high',
            'aggregated': True,
            'count': len(pairs),
            'columns': all_cols,
            'column': f"{len(all_cols)} columns",
            'pairs': [{'var1': a, 'var2': b, 'correlation': c} for a, b, c in pairs],
            'message': (
                f'{len(pairs)} pairs of variables are nearly identical (≥95% correlated), '
                f'involving {len(all_cols)} columns: '
                f'{", ".join(f"{c!r}" for c in all_cols[:6])}'
                f'{"..." if len(all_cols) > 6 else ""}. '
                f'These are essentially measuring the same thing.'
            ),
            'plain_english': (
                f'You have {len(all_cols)} columns that are practically carbon copies of each other. '
                f'It\'s like asking someone their height in centimetres AND inches AND feet — '
                f'each extra copy just adds confusion without adding information.'
            ),
            'recommendation': (
                f'Remove one column from each highly correlated pair, keeping the one that is '
                f'easiest to explain. Alternatively, apply PCA to reduce all {len(all_cols)} '
                f'into a smaller set of independent components.'
            ),
        }

    def _merge_high_outliers(self, group: list) -> dict:
        """Aggregate outlier warnings across columns."""
        cols_with_pct = [(item.get('column', ''), item.get('percentage', 0)) for item in group]
        cols_with_pct.sort(key=lambda x: x[1], reverse=True)
        all_cols = [c for c, _ in cols_with_pct]

        shown = cols_with_pct[:_MAX_INDIVIDUAL]
        hidden = len(cols_with_pct) - len(shown)
        detail = ", ".join(f'"{c}" ({p:.1f}%)' for c, p in shown)
        if hidden > 0:
            detail += f" and {hidden} more"

        return {
            'type': 'high_outliers',
            'severity': 'medium',
            'aggregated': True,
            'count': len(group),
            'columns': all_cols,
            'column': f"{len(all_cols)} columns",
            'message': (
                f'{len(group)} columns have significant outlier problems: {detail}. '
                f'These extreme values can distort model predictions.'
            ),
            'plain_english': (
                f'{len(group)} of your columns have extreme values far outside the normal range. '
                f'Worst offenders: {detail}. '
                f'Outliers can drag model predictions in the wrong direction — like how '
                f'one billionaire raises the average salary of a room dramatically.'
            ),
            'recommendation': (
                f'Cap values in these {len(group)} columns at the 1st and 99th percentiles '
                f'(winsorization), or use a tree-based model which is naturally robust to outliers.'
            ),
        }

    def _merge_near_zero_variance(self, group: list) -> dict:
        """Aggregate near-zero variance warnings."""
        cols = [item.get('column', '') for item in group]
        return {
            'type': 'near_zero_variance',
            'severity': 'medium',
            'aggregated': True,
            'count': len(group),
            'columns': cols,
            'column': f"{len(cols)} columns",
            'message': (
                f'{len(cols)} columns have almost no variation: '
                f'{", ".join(f"{c!r}" for c in cols[:5])}'
                f'{"..." if len(cols) > 5 else ""}. '
                f'They cannot teach a model anything useful.'
            ),
            'plain_english': (
                f'{len(cols)} columns barely change — nearly all values are the same. '
                f'It\'s like trying to predict exam scores using everyone\'s age when '
                f'everyone is 20. No variation means no signal.'
            ),
            'recommendation': (
                f'Drop these {len(cols)} columns: {", ".join(f"{c!r}" for c in cols[:5])}'
                f'{"..." if len(cols) > 5 else ""}. '
                f'They are unlikely to contribute to model performance.'
            ),
        }

    def _merge_whitespace_issues(self, group: list) -> dict:
        """Aggregate whitespace issues."""
        cols = [item.get('column', '') for item in group]
        return {
            'type': 'whitespace_issues',
            'severity': 'medium',
            'aggregated': True,
            'count': len(group),
            'columns': cols,
            'column': f"{len(cols)} columns",
            'message': (
                f'{len(cols)} columns have whitespace issues: '
                f'{", ".join(f"{c!r}" for c in cols[:5])}'
                f'{"..." if len(cols) > 5 else ""}.'
            ),
            'plain_english': (
                f'{len(cols)} text columns have leading/trailing spaces or inconsistent whitespace. '
                f'This causes "Yes" and " Yes" to be treated as different categories.'
            ),
            'recommendation': (
                f'Strip whitespace from all {len(cols)} affected columns: '
                f'df[col] = df[col].str.strip()'
            ),
        }

    def _merge_disguised_missing(self, group: list) -> dict:
        """Aggregate disguised missing value warnings."""
        cols = [item.get('column', '') for item in group]
        return {
            'type': 'disguised_missing',
            'severity': 'medium',
            'aggregated': True,
            'count': len(group),
            'columns': cols,
            'column': f"{len(cols)} columns",
            'message': (
                f'{len(cols)} columns contain disguised null values (like "NA", "?", "--"): '
                f'{", ".join(f"{c!r}" for c in cols[:5])}'
                f'{"..." if len(cols) > 5 else ""}.'
            ),
            'plain_english': (
                f'{len(cols)} columns have missing values hiding behind placeholder text. '
                f'These don\'t show up as null but are meaningless data points that can '
                f'confuse your model.'
            ),
            'recommendation': (
                f'Replace disguised nulls with proper NaN in all {len(cols)} columns, '
                f'then decide on an imputation strategy.'
            ),
        }

    def _merge_extreme_skewness(self, group: list) -> dict:
        """Aggregate extreme skewness warnings."""
        cols_with_skew = [(item.get('column', ''), item.get('skewness', 0)) for item in group]
        cols_with_skew.sort(key=lambda x: abs(x[1]), reverse=True)
        cols = [c for c, _ in cols_with_skew]

        shown = cols_with_skew[:_MAX_INDIVIDUAL]
        detail = ", ".join(f'"{c}" (skew={s:.0f})' for c, s in shown)
        hidden = len(cols_with_skew) - len(shown)
        if hidden > 0:
            detail += f" and {hidden} more"

        return {
            'type': 'extreme_skewness',
            'severity': 'high',
            'aggregated': True,
            'count': len(group),
            'columns': cols,
            'column': f"{len(cols)} columns",
            'message': (
                f'{len(cols)} columns have extreme skewness indicating data corruption: '
                f'{detail}. This is far beyond any natural distribution.'
            ),
            'plain_english': (
                f'{len(cols)} columns have statistical shapes that are physically impossible '
                f'for real data. This almost certainly means sentinel values, broken sensors, '
                f'or severe data-entry errors are still in the data.'
            ),
            'recommendation': (
                f'Check these {len(cols)} columns for sentinel/placeholder values '
                f'(e.g. -9999, 9999) and replace them with np.nan before any analysis.'
            ),
        }

    def _generic_merge(self, issue_type: str, group: list) -> dict:
        """Fallback merger for types without a specific handler."""
        cols = [item.get('column', '') for item in group if item.get('column')]
        severity = group[0].get('severity', 'medium')
        type_label = issue_type.replace('_', ' ')

        if not cols:
            # Non-column-specific issues (like duplicate_rows)
            return group[0]  # keep the first one

        unique_cols = sorted(set(cols))

        return {
            'type': issue_type,
            'severity': severity,
            'aggregated': True,
            'count': len(group),
            'columns': unique_cols,
            'column': f"{len(unique_cols)} columns",
            'message': (
                f'{len(group)} {type_label} issues found across '
                f'{len(unique_cols)} columns: '
                f'{", ".join(f"{c!r}" for c in unique_cols[:6])}'
                f'{"..." if len(unique_cols) > 6 else ""}.'
            ),
            'plain_english': group[0].get('plain_english', ''),
            'recommendation': group[0].get('recommendation', ''),
        }


# Map issue types to their merge handlers
_MERGE_HANDLERS = {
    'high_correlation': AggregationEngine._merge_high_correlation,
    'multicollinearity': AggregationEngine._merge_multicollinearity,
    'high_outliers': AggregationEngine._merge_high_outliers,
    'near_zero_variance': AggregationEngine._merge_near_zero_variance,
    'whitespace_issues': AggregationEngine._merge_whitespace_issues,
    'disguised_missing': AggregationEngine._merge_disguised_missing,
    'extreme_skewness': AggregationEngine._merge_extreme_skewness,
}
