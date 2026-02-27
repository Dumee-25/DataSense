"""RelevanceFilter — filters and contextualises findings for the recommended model.

If the system recommends a tree-based model (Random Forest, XGBoost, LightGBM)
it should NOT spend pages warning about issues that only affect linear models
(multicollinearity, heteroscedasticity, etc.).

Each finding that passes through this filter is annotated with:
  1. ``model_context_note`` — a concrete, value-specific explanation of why this
     finding matters (or doesn't) for the recommended model.
  2. ``action_priority``    — one of ``must-fix``, ``should-fix``, ``nice-to-have``,
     ``informational`` so the user knows what to tackle first.
  3. Severity adjustments   — issues irrelevant to the model family are downgraded.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ── Issue relevance by model family ──────────────────────────────────────────

# Issues caused entirely by linear-model assumptions — safe to downgrade for trees/SVM.
_LINEAR_ONLY_ISSUES = frozenset({
    'multicollinearity',
    'heteroscedasticity',
})

# Issues whose impact changes *quantitatively* by model family (need contextual notes).
_MODEL_CONTEXTUAL_ISSUES = frozenset({
    'high_correlation',
    'high_outliers',
    'near_zero_variance',
    'extreme_skewness',
    'high_missingness',
    'moderate_missingness',
})

# Data-quality issues that are ALWAYS relevant regardless of model.
_UNIVERSAL_DATA_ISSUES = frozenset({
    'class_imbalance',
    'data_leakage_risk',
    'metadata_leakage',
    'simpsons_paradox',
    'constant_column',
    'mixed_types',
    'disguised_missing',
    'infinite_values',
    'numeric_sentinel_values',
})


class RelevanceFilter:
    """Filters and contextualises warnings based on the recommended model."""
    __slots__ = ()

    def filter(self, issues: list, model_name: str, task_type: str = 'classification') -> list:
        """Filter and contextualise issues based on the recommended model.

        - Issues irrelevant to the recommended model are downgraded (not removed)
        - Every finding gets a concrete ``model_context_note`` with actual values
        - Every finding gets an ``action_priority`` label
        - Critical and data-quality issues are NEVER filtered (they're always relevant)
        """
        if not issues or not model_name:
            return issues

        model_family = self._classify_model(model_name)
        filtered: list = []

        for issue in issues:
            issue = {**issue}  # shallow copy — never mutate originals
            severity = issue.get('severity', 'medium')
            issue_type = issue.get('type', '')

            # ── Critical + universal data-quality: always keep, always annotate ──
            if severity == 'critical' or issue_type in _UNIVERSAL_DATA_ISSUES:
                issue['action_priority'] = self._action_priority(issue_type, severity, model_family, task_type)
                note = self._universal_note(issue, model_name, task_type)
                if note:
                    issue['model_context_note'] = note
                filtered.append(issue)
                continue

            # ── Linear-only issues: downgrade for trees/SVM ──────────────────
            if issue_type in _LINEAR_ONLY_ISSUES and model_family in ('tree', 'svm'):
                issue['model_context_note'] = self._linear_only_note(issue, model_family, model_name)
                if severity == 'high':
                    issue['original_severity'] = severity
                    issue['severity'] = 'medium'
                issue['action_priority'] = 'nice-to-have' if model_family == 'tree' else 'should-fix'
                filtered.append(issue)
                continue

            # ── Model-contextual issues: add value-specific notes ────────────
            if issue_type in _MODEL_CONTEXTUAL_ISSUES:
                note = self._contextual_note(issue, model_family, model_name, task_type)
                if note:
                    issue['model_context_note'] = note
                # Downgrade outlier severity for tree models
                if issue_type == 'high_outliers' and model_family == 'tree' and severity == 'medium':
                    issue['original_severity'] = severity
                    issue['severity'] = 'medium'  # keep same but mark as optional
                    issue['action_priority'] = 'nice-to-have'
                else:
                    issue['action_priority'] = self._action_priority(issue_type, severity, model_family, task_type)
                filtered.append(issue)
                continue

            # ── Everything else: annotate with default priority ──────────────
            issue['action_priority'] = self._action_priority(issue_type, severity, model_family, task_type)
            filtered.append(issue)

        return filtered

    # ── Model classification ──────────────────────────────────────────────────

    @staticmethod
    def _classify_model(model_name: str) -> str:
        name_lower = model_name.lower().strip()
        if any(t in name_lower for t in ('random forest', 'xgboost', 'lightgbm',
                                          'gradient boost', 'decision tree', 'catboost')):
            return 'tree'
        if any(t in name_lower for t in ('logistic', 'ridge', 'lasso', 'linear', 'elastic')):
            return 'linear'
        if any(t in name_lower for t in ('svm', 'support vector')):
            return 'svm'
        if any(t in name_lower for t in ('knn', 'k-nearest', 'nearest neighbor')):
            return 'knn'
        if any(t in name_lower for t in ('neural', 'deep learning', 'mlp')):
            return 'neural'
        return 'unknown'

    # ── Action priority assignment ────────────────────────────────────────────

    @staticmethod
    def _action_priority(issue_type: str, severity: str, model_family: str, task_type: str) -> str:
        """Assign a concrete action priority based on issue, severity, and model."""
        # Critical data-integrity issues are always must-fix
        if severity == 'critical':
            return 'must-fix'
        # Data quality problems that corrupt any model
        if issue_type in ('constant_column', 'infinite_values', 'numeric_sentinel_values',
                          'mixed_types', 'disguised_missing'):
            return 'must-fix'
        if issue_type == 'data_leakage_risk':
            return 'must-fix'
        # Model-specific severity mapping
        if issue_type in ('multicollinearity', 'heteroscedasticity'):
            if model_family in ('tree', 'svm'):
                return 'nice-to-have'
            return 'must-fix' if severity == 'high' else 'should-fix'
        if issue_type == 'high_outliers':
            if model_family == 'tree':
                return 'nice-to-have'
            return 'should-fix'
        if issue_type in ('high_correlation',):
            if model_family == 'tree':
                return 'nice-to-have'
            return 'should-fix'
        if issue_type in ('high_missingness', 'moderate_missingness'):
            return 'must-fix' if severity == 'high' else 'should-fix'
        if issue_type == 'high_cardinality':
            if model_family == 'tree':
                return 'should-fix'
            return 'must-fix'  # linear/SVM can't handle raw high-cardinality
        if issue_type in ('extreme_skewness',):
            if model_family == 'tree':
                return 'nice-to-have'
            return 'should-fix'
        if severity == 'high':
            return 'should-fix'
        return 'informational'

    # ── Universal notes (critical / data-quality) ─────────────────────────────

    @staticmethod
    def _universal_note(issue: dict, model_name: str, task_type: str) -> str:
        """Concrete, value-specific note for data-quality issues that affect every model."""
        t = issue.get('type', '')
        col = issue.get('column', '')

        if t == 'class_imbalance':
            pct = issue.get('majority_pct', '?')
            return (f'With {pct}% of rows in one class, {model_name} will default to predicting '
                    f'the majority class unless you explicitly address the imbalance. '
                    f'Use class_weight="balanced" or SMOTE, and evaluate with F1 / AUC-ROC — '
                    f'not accuracy, which will be misleadingly high at ~{pct}%.')

        if t == 'data_leakage_risk':
            return (f'If "{col}" contains information from after the event you\'re predicting, '
                    f'{model_name} will achieve near-perfect test scores but fail completely on '
                    f'new data. Verify the column\'s temporal relationship to the target before '
                    f'proceeding — this is the single most damaging issue in the dataset.')

        if t == 'metadata_leakage':
            return (f'"{col}" appears to be a data-collection artefact (batch ID, operator code, etc.). '
                    f'{model_name} will memorise batch-specific patterns instead of learning real signal. '
                    f'Remove it from the feature set — keep it only for post-hoc auditing.')

        if t == 'simpsons_paradox':
            cols = issue.get('columns', [])
            if len(cols) >= 3:
                return (f'The correlation between "{cols[0]}" and "{cols[1]}" reverses direction '
                        f'inside subgroups of "{cols[2]}". Any model trained on the overall data — '
                        f'including {model_name} — will learn the wrong relationship. '
                        f'Always stratify by "{cols[2]}" or include it as a feature.')
            return ''

        if t == 'constant_column':
            return (f'"{col}" has a single unique value. It carries zero information for any model. '
                    f'Drop it — it wastes memory and can cause numerical issues in some solvers.')

        if t == 'mixed_types':
            return (f'"{col}" contains both numeric and text values. {model_name} cannot '
                    f'handle mixed types natively — it will either crash or silently coerce '
                    f'values incorrectly. Split the column or convert to a consistent type.')

        if t == 'disguised_missing':
            return (f'"{col}" has placeholder strings ("NA", "?", "--") pretending to be real values. '
                    f'{model_name} will treat these as valid categories, corrupting its learned patterns. '
                    f'Replace them with proper NaN before any preprocessing.')

        if t == 'infinite_values':
            return (f'"{col}" contains Inf/-Inf values. {model_name} will either crash or '
                    f'produce NaN predictions. Replace infinities with NaN or clip to the '
                    f'column\'s 1st/99th percentile.')

        if t == 'numeric_sentinel_values':
            sentinel_vals = issue.get('sentinel_values', [])
            vals_str = ', '.join(str(v) for v in sentinel_vals[:4])
            return (f'"{col}" contains sentinel placeholders ({vals_str}) masquerading as real numbers. '
                    f'These corrupt every statistic — mean, variance, correlation — and will mislead '
                    f'{model_name}. Replace them with NaN before any analysis.')

        return ''

    # ── Linear-only issue notes ───────────────────────────────────────────────

    @staticmethod
    def _linear_only_note(issue: dict, model_family: str, model_name: str) -> str:
        """Value-specific note when a linear-model issue appears with a non-linear model."""
        t = issue.get('type', '')
        family_label = 'Tree models' if model_family == 'tree' else model_name

        if t == 'multicollinearity':
            v1, v2 = issue.get('var1', '?'), issue.get('var2', '?')
            corr = issue.get('correlation', 0)
            return (f'"{v1}" and "{v2}" are {corr:.0%} correlated, which would destabilise '
                    f'a linear model\'s coefficients. {family_label} split on one feature at a '
                    f'time and are not confused by this. You can keep both columns — but '
                    f'dropping one will reduce training time and make feature-importance '
                    f'scores easier to interpret.')

        if t == 'heteroscedasticity':
            cols = issue.get('columns', [])
            col_desc = f'"{cols[0]}" → "{cols[1]}"' if len(cols) >= 2 else 'the listed columns'
            return (f'The error spread for {col_desc} varies across the range, violating '
                    f'a core assumption of linear regression. {family_label} make no assumption '
                    f'about error distribution, so this has zero impact on predictions. '
                    f'Only revisit if you switch to a linear model.')

        return ''

    # ── Contextual notes (model-dependent severity) ───────────────────────────

    @staticmethod
    def _contextual_note(issue: dict, model_family: str, model_name: str, task_type: str) -> str:
        """Value-specific note for issues whose impact depends on the model family."""
        t = issue.get('type', '')

        # ── High correlation ──────────────────────────────────────────────────
        if t == 'high_correlation':
            v1 = issue.get('var1', '?')
            v2 = issue.get('var2', '?')
            corr = issue.get('correlation', 0)
            # Aggregated variant
            if issue.get('aggregated'):
                count = issue.get('count', 0)
                n_cols = len(issue.get('columns', []))
                if model_family == 'tree':
                    return (f'{count} correlated pairs across {n_cols} columns detected. '
                            f'{model_name} handles this naturally, so predictions won\'t suffer. '
                            f'However, feature-importance scores will be split between correlated '
                            f'columns, making them harder to interpret. Dropping redundant columns '
                            f'or using PCA will clean up importance rankings and speed up training.')
                if model_family == 'linear':
                    return (f'{count} correlated pairs across {n_cols} columns detected. '
                            f'For {model_name}, each correlated pair inflates standard errors and '
                            f'makes individual coefficients unreliable — you cannot trust which '
                            f'variable is "driving" the prediction. Apply PCA or drop one column '
                            f'from each pair before fitting.')
                if model_family == 'svm':
                    return (f'{count} correlated pairs across {n_cols} columns detected. '
                            f'SVM with RBF kernel is somewhat robust to this, but correlated '
                            f'features inflate the effective dimensionality, slowing convergence. '
                            f'Apply PCA or drop redundant columns to speed up training.')
                return (f'{count} correlated pairs across {n_cols} columns. Consider reducing '
                        f'redundancy via PCA or manual feature selection.')
            # Single-pair variant
            if model_family == 'tree':
                return (f'"{v1}" and "{v2}" at r={corr:.2f} won\'t hurt {model_name}\'s accuracy, '
                        f'but feature-importance will be split between them. If interpretability '
                        f'matters, keep the more intuitive one and drop the other.')
            if model_family == 'linear':
                return (f'"{v1}" and "{v2}" at r={corr:.2f} will make {model_name}\'s coefficients '
                        f'for both variables unreliable. Drop one (keep the more explainable one) '
                        f'or combine them via PCA before fitting.')
            if model_family == 'svm':
                return (f'"{v1}" and "{v2}" at r={corr:.2f} increase effective dimensionality, '
                        f'slowing SVM convergence. Consider dropping one or applying PCA.')
            return (f'"{v1}" and "{v2}" are correlated at r={corr:.2f}. Consider whether both '
                    f'carry independent information.')

        # ── High outliers ─────────────────────────────────────────────────────
        if t == 'high_outliers':
            col = issue.get('column', '?')
            pct = issue.get('percentage', '?')
            # Aggregated
            if issue.get('aggregated'):
                count = issue.get('count', 0)
                cols = issue.get('columns', [])
                top3 = ', '.join(f'"{c}"' for c in cols[:3])
                if model_family == 'tree':
                    return (f'{count} columns have >5% extreme outliers (worst: {top3}). '
                            f'{model_name} splits on rank order, not magnitude, so outliers '
                            f'don\'t distort splits the way they warp linear coefficients. '
                            f'No action needed unless you also plan to use a linear model.')
                if model_family == 'linear':
                    return (f'{count} columns have >5% extreme outliers (worst: {top3}). '
                            f'Each outlier drags {model_name}\'s fitted line toward it, '
                            f'distorting predictions for the majority of normal observations. '
                            f'Winsorize at the 1st/99th percentiles before fitting.')
                return (f'{count} columns have significant outliers (worst: {top3}). '
                        f'Consider winsorization or robust scaling.')
            # Single-column
            if model_family == 'tree':
                return (f'{pct}% of "{col}" are extreme outliers, but {model_name} splits on '
                        f'rank order so they won\'t distort predictions. No action required.')
            if model_family == 'linear':
                return (f'{pct}% of "{col}" are extreme outliers that will pull {model_name}\'s '
                        f'regression line toward them. Cap at the 1st/99th percentile or apply '
                        f'a log transform to compress the range.')
            if model_family == 'svm':
                return (f'{pct}% of "{col}" are extreme outliers. SVM is sensitive to feature '
                        f'scale — outliers will dominate the distance metric. Apply robust '
                        f'scaling (RobustScaler) or winsorize before fitting.')
            return f'{pct}% of "{col}" are outliers. Consider capping or transforming.'

        # ── Near-zero variance ────────────────────────────────────────────────
        if t == 'near_zero_variance':
            col = issue.get('column', '?')
            if issue.get('aggregated'):
                count = issue.get('count', 0)
                cols = issue.get('columns', [])
                col_list = ', '.join(f'"{c}"' for c in cols[:5])
                return (f'{count} columns barely vary ({col_list}). They carry almost no signal '
                        f'for any model. Drop them to save memory, reduce noise, and speed up '
                        f'{model_name}\'s training without losing predictive power.')
            if model_family == 'tree':
                return (f'"{col}" barely varies. {model_name} will simply never split on it '
                        f'(no information gain). Drop it to reduce noise and training time.')
            if model_family == 'linear':
                return (f'"{col}" barely varies. {model_name} will assign it a coefficient, but '
                        f'it will explain essentially zero variance. Drop it — it adds a degree '
                        f'of freedom without contributing signal.')
            return f'"{col}" has almost no variation. Drop it.'

        # ── Extreme skewness ──────────────────────────────────────────────────
        if t == 'extreme_skewness':
            col = issue.get('column', '?')
            skew = issue.get('skewness', 0)
            if model_family == 'tree':
                return (f'"{col}" has extreme skewness ({skew:+.0f}), which usually signals '
                        f'data corruption rather than a natural distribution. {model_name} is '
                        f'robust to skewness for prediction, but the underlying data problem '
                        f'(sentinel values, sensor errors) should still be investigated — '
                        f'the corrupted rows may carry wrong target labels too.')
            if model_family == 'linear':
                return (f'"{col}" has extreme skewness ({skew:+.0f}), almost certainly caused '
                        f'by sentinel values or data corruption. For {model_name}, this will '
                        f'massively distort the coefficient and residuals. Investigate and fix '
                        f'the source, then apply a log or Box-Cox transform if skew remains.')
            return (f'"{col}" has extreme skewness ({skew:+.0f}). Investigate for sentinel '
                    f'values or data corruption before training any model.')

        # ── Missingness ───────────────────────────────────────────────────────
        if t in ('high_missingness', 'moderate_missingness'):
            col = issue.get('column', '?')
            pct = issue.get('percentage', '?')
            if t == 'high_missingness':
                if model_family == 'tree':
                    return (f'"{col}" is {pct}% missing. {model_name} (XGBoost/LightGBM) can '
                            f'handle missing values natively by learning optimal split directions '
                            f'for NaN. However, at {pct}% missing, consider whether the column '
                            f'carries enough signal to justify inclusion — if most values are '
                            f'absent, even native handling may not help.')
                return (f'"{col}" is {pct}% missing. {model_name} cannot handle NaN natively. '
                        f'At this level, imputation may introduce more noise than signal — '
                        f'consider dropping the column unless domain knowledge confirms it\'s '
                        f'critical. If you keep it, add a binary "is_missing" indicator column.')
            # moderate
            if model_family == 'tree':
                return (f'"{col}" has {pct}% missing values. {model_name} handles NaN natively, '
                        f'so no imputation is needed. The model will learn whether "missing" '
                        f'itself is predictive.')
            return (f'"{col}" has {pct}% missing values. Impute with the median (numeric) or '
                    f'mode (categorical) before fitting {model_name}. Consider adding a binary '
                    f'"was_imputed" indicator column to preserve the missingness signal.')

        return ''
