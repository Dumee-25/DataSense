"""RelevanceFilter — filters warnings based on the recommended model.

If the system recommends a tree-based model (Random Forest, XGBoost, LightGBM)
it should NOT spend pages warning about issues that only affect linear models
(multicollinearity, heteroscedasticity, etc.).

This creates internal consistency: the recommendations and warnings agree.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

# ── Model categories ──────────────────────────────────────────────────────────
_TREE_MODELS = frozenset({
    'random forest', 'xgboost', 'lightgbm',
    'random forest regressor', 'xgboost regressor', 'lightgbm regressor',
    'gradient boosting', 'decision tree',
})

_LINEAR_MODELS = frozenset({
    'logistic regression', 'ridge regression', 'lasso regression',
    'linear regression', 'elastic net',
})

_SVM_MODELS = frozenset({
    'support vector machine', 'svm', 'svr',
})

# ── Issue relevance by model family ──────────────────────────────────────────
# Issues that are ONLY relevant to specific model families.
# If the recommended model is NOT in that family, the warning is downgraded.
_LINEAR_ONLY_ISSUES = frozenset({
    'multicollinearity',
    'heteroscedasticity',
})

# Issues where the wording should change based on the model
_MODEL_CONTEXTUAL_ISSUES = frozenset({
    'high_correlation',
    'high_outliers',
    'near_zero_variance',
})


class RelevanceFilter:
    """Filters and contextualizes warnings based on the recommended model."""
    __slots__ = ()

    def filter(self, issues: list, model_name: str, task_type: str = 'classification') -> list:
        """Filter and contextualize issues based on the recommended model.

        - Issues irrelevant to the recommended model are downgraded (not removed)
        - Contextual notes are added explaining relevance to the chosen model
        - Critical and data-quality issues are NEVER filtered (they're always relevant)

        Args:
            issues: List of finding dicts
            model_name: The primary recommended model name
            task_type: 'classification' or 'regression'

        Returns:
            Filtered and annotated list of issues
        """
        if not issues or not model_name:
            return issues

        model_family = self._classify_model(model_name)
        filtered = []

        for issue in issues:
            severity = issue.get('severity', 'medium')
            issue_type = issue.get('type', '')

            # NEVER filter critical issues or data-quality warnings
            if severity == 'critical':
                filtered.append(issue)
                continue

            # Check if this warning is irrelevant to the recommended model
            if issue_type in _LINEAR_ONLY_ISSUES and model_family == 'tree':
                # Downgrade and add context note instead of removing
                issue = {**issue, 'model_context_note': self._tree_context_note(issue_type, model_name)}
                # Downgrade severity if it was high
                if severity == 'high':
                    issue['original_severity'] = severity
                    issue['severity'] = 'medium'
                filtered.append(issue)
                continue

            # Add contextual notes for model-relevant issues
            if issue_type in _MODEL_CONTEXTUAL_ISSUES:
                note = self._contextual_note(issue_type, model_family, model_name)
                if note:
                    issue = {**issue, 'model_context_note': note}

            filtered.append(issue)

        return filtered

    def _classify_model(self, model_name: str) -> str:
        """Classify a model name into a family."""
        name_lower = model_name.lower().strip()
        if any(t in name_lower for t in ('random forest', 'xgboost', 'lightgbm',
                                          'gradient boost', 'decision tree')):
            return 'tree'
        if any(t in name_lower for t in ('logistic', 'ridge', 'lasso', 'linear', 'elastic')):
            return 'linear'
        if any(t in name_lower for t in ('svm', 'support vector')):
            return 'svm'
        return 'unknown'

    def _tree_context_note(self, issue_type: str, model_name: str) -> str:
        """Context note when a linear-model warning appears with a tree-model recommendation."""
        notes = {
            'multicollinearity': (
                f'Note: Since {model_name} is recommended, multicollinearity is much less '
                f'concerning. Tree-based models split on individual features and are not '
                f'confused by correlated inputs the way linear models are. You can safely '
                f'keep both columns — though removing one may still speed up training.'
            ),
            'heteroscedasticity': (
                f'Note: Since {model_name} is recommended, heteroscedasticity is not a problem. '
                f'Tree-based models make no assumptions about error distribution. '
                f'This would only matter if you switch to linear regression.'
            ),
        }
        return notes.get(issue_type, '')

    def _contextual_note(self, issue_type: str, model_family: str, model_name: str) -> str:
        """Add model-specific context to warnings that apply differently by model type."""
        if model_family == 'tree':
            notes = {
                'high_correlation': (
                    f'{model_name} handles correlated features naturally, but removing '
                    f'redundant columns can still improve training speed and interpretability.'
                ),
                'high_outliers': (
                    f'{model_name} is robust to outliers — they won\'t distort predictions '
                    f'the way they would with linear models. Fixing outliers is optional here.'
                ),
                'near_zero_variance': (
                    f'Near-zero variance columns won\'t hurt {model_name}, but they add '
                    f'unnecessary computation. Dropping them is a free optimization.'
                ),
            }
            return notes.get(issue_type, '')

        if model_family == 'linear':
            notes = {
                'high_correlation': (
                    f'This is important for {model_name}: correlated features inflate '
                    f'standard errors and make coefficients unreliable. Consider PCA or '
                    f'dropping one column from each correlated pair.'
                ),
                'high_outliers': (
                    f'Outliers directly distort {model_name}\'s coefficients. Winsorize '
                    f'these columns at the 1st/99th percentile before fitting.'
                ),
            }
            return notes.get(issue_type, '')

        return ''
