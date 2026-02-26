from typing import Dict, Any

# Compact model defs: (name, missing, imbalance, outliers, categoricals, small, large, interpretable, why)
_CLF = [
    ('Random Forest', False, False, True, False, True, True, True,
     'Robust, handles outliers well, gives feature importance scores, and rarely overfits.'),
    ('XGBoost', True, True, True, False, False, True, True,
     'State-of-the-art on tabular data. Handles missing values natively and has built-in class weighting.'),
    ('Logistic Regression', False, True, False, False, True, True, True,
     'Fast, interpretable, and great as a baseline. Coefficients are easy to explain to stakeholders.'),
    ('LightGBM', True, True, True, True, False, True, True,
     'Extremely fast on large datasets. Handles categoricals natively without encoding.'),
    ('Support Vector Machine', False, True, False, False, True, False, False,
     'Works well when classes are clearly separable. Best on small-to-medium datasets.'),
]
_REG = [
    ('Random Forest Regressor', False, False, True, False, True, True, True,
     'Reliable out-of-the-box regressor with feature importance. Robust to outliers.'),
    ('XGBoost Regressor', True, False, True, False, False, True, True,
     'Top-performing regressor on structured data. Handles missing values natively.'),
    ('Ridge Regression', False, False, False, False, True, True, True,
     'A regularized linear model — fast, interpretable, and great baseline. Good when features are correlated.'),
    ('LightGBM Regressor', True, False, True, True, False, True, True,
     'Best choice for large datasets with mixed feature types.'),
    ('Lasso Regression', False, False, False, False, True, True, True,
     'Performs automatic feature selection by zeroing out unimportant features. Good when you have many columns.'),
]
_MODELS = {'classification': _CLF, 'regression': _REG}


class ModelRecommender:
    __slots__ = ()

    def recommend(self, blueprint: dict, stats: dict) -> Dict[str, Any]:
        task = self._determine_task(blueprint)
        chars = self._extract_characteristics(blueprint, stats)
        scored = self._score_models(task, chars)
        pri, alts = scored[0], scored[1:3]
        return {
            'task_type': task, 'primary_model': pri['name'],
            'confidence': round(pri['score'] / 10, 2), 'why_this_model': pri['why'],
            'reasoning': pri['reasons'],
            'alternatives': [{'model': m['name'], 'why': m['why']} for m in alts],
            'preprocessing_steps': self._recommend_preprocessing(chars),
            'cv_strategy': self._recommend_cv(blueprint, chars),
            'recommended_metrics': self._recommend_metrics(task, chars),
            'characteristics': chars,
        }

    def _determine_task(self, bp: dict) -> str:
        tgt = bp.get('target_candidates', [])
        return tgt[0].get('task_type', 'classification') if tgt else 'classification'

    def _extract_characteristics(self, bp: dict, stats: dict) -> dict:
        b, p = bp.get('basic_info', {}), stats.get('patterns', {})
        rf = stats.get('red_flags', [])
        w = stats.get('warnings', [])
        rows = b.get('rows', 0)
        return {
            'n_rows': rows, 'n_cols': b.get('columns', 0),
            'has_imbalance': any(f['type'] == 'class_imbalance' for f in rf),
            'has_multicollinearity': any(x['type'] == 'multicollinearity' for x in w),
            'has_outliers': any(x['type'] == 'high_outliers' for x in w),
            'has_missing': b.get('missing_percentage', 0) > 5,
            'missing_pct': b.get('missing_percentage', 0),
            'is_large': rows > 50_000, 'is_small': rows < 500,
            'is_high_dimensional': b.get('columns', 0) > 50,
            'has_categoricals': p.get('many_categoricals', False),
            'has_skewed': p.get('has_skewed_features', False),
        }

    def _score_models(self, task: str, ch: dict) -> list:
        scored = []
        for name, hmiss, himb, hout, hcat, gsm, glg, interp, why in _MODELS.get(task, _CLF):
            sc, reasons = 5.0, []

            # Missing values — strong signal; models that handle it natively earn a clear lead
            if ch['has_missing']:
                if hmiss:
                    sc += 2.0; reasons.append(f'handles the {ch["missing_pct"]:.0f}% missing values natively')
                else:
                    sc -= 1.5

            # Class imbalance — built-in support matters a lot here
            if ch['has_imbalance'] and himb:
                sc += 2.0; reasons.append('has built-in support for imbalanced classes')

            # Outliers
            if ch['has_outliers']:
                if hout: sc += 1.0; reasons.append('robust to the outliers detected in your data')
                else: sc -= 1.0

            # Categorical features — native handling avoids encoding noise
            if ch['has_categoricals'] and hcat:
                sc += 1.0; reasons.append('handles categorical variables without manual encoding')

            # Dataset size
            if ch['is_large']:
                if glg:
                    sc += 1.5; reasons.append('scales well to your large dataset size')
                else:
                    sc -= 2.5
                # Extra reward for models purpose-built for large data (LightGBM variants)
                if 'LightGBM' in name:
                    sc += 0.5; reasons.append('histogram-based algorithm is memory-efficient at scale')
            if ch['is_small'] and gsm:
                sc += 0.5
            elif not ch['is_small'] and not gsm and not ch['is_large']:
                # Don't penalize large-data specialists on large datasets —
                # being purpose-built for scale is already rewarded above
                sc -= 0.5

            # Regularized linear models
            if ch['has_multicollinearity'] and name in ('Ridge Regression', 'Lasso Regression'):
                sc += 1.0; reasons.append('regularization handles the multicollinearity in your data')
            if ch['is_high_dimensional'] and name == 'Lasso Regression':
                sc += 1.0; reasons.append('automatically selects important features from your many columns')

            # Interpretability bonus for linear models on clean, small-to-medium data
            # (encourages Logistic/Ridge as a sensible baseline over "throw RF at it")
            if name in ('Logistic Regression', 'Ridge Regression') and not ch['has_missing'] \
                    and not ch['has_imbalance'] and not ch['is_large'] and not ch['has_categoricals']:
                sc += 0.5; reasons.append('clean data makes a fast, interpretable baseline the smart first choice')

            # Generalist tax — models that work for every size but have no native handling
            # of the dataset's specific issues shouldn't edge out specialists on a technicality
            if gsm and glg and not hmiss and not himb:
                sc -= 0.4  # Random Forest is the main target here

            if not reasons:
                reasons.append('solid all-around choice for this type of data')
            scored.append({'name': name, 'score': min(sc, 10.0), 'reasons': reasons, 'why': why})
        return sorted(scored, key=lambda x: x['score'], reverse=True)

    def _recommend_preprocessing(self, ch: dict) -> list:
        s = []
        if ch['has_missing']:
            s.append('Impute missing values — use median for numeric columns, most frequent value for categorical columns.')
        if ch['has_outliers']:
            s.append('Cap extreme outliers at the 1st and 99th percentile (called "winsorization") to prevent them from distorting the model.')
        if ch['has_skewed']:
            s.append('Apply log transformation to heavily skewed numeric columns. This makes distributions more symmetric and helps linear models.')
        if ch['has_categoricals']:
            s.append('Encode categorical columns — use One-Hot Encoding for low-cardinality columns (under 10 unique values) and Target Encoding for high-cardinality ones.')
        if ch['has_multicollinearity']:
            s.append('Remove one column from each highly correlated pair, or use PCA to reduce correlated features into uncorrelated components.')
        if not ch['is_large']:
            s.append('Scale numeric features using StandardScaler before training — especially important for distance-based models and linear models.')
        s.append('Split data into 80% training and 20% test before any fitting or transformation.')
        return s

    def _recommend_cv(self, bp: dict, ch: dict) -> str:
        st = bp.get('data_structure', {}).get('type', 'cross-sectional')
        if st == 'time-series':
            return 'Time Series Split (Walk-Forward Validation) — always train on past data and validate on future data. Never shuffle time-series data before splitting.'
        if ch['has_imbalance']:
            return 'Stratified 5-Fold Cross-Validation — ensures each fold has the same class distribution as the full dataset. Critical when your target classes are imbalanced.'
        if ch['is_small']:
            return 'Leave-One-Out Cross-Validation (LOOCV) or Stratified 10-Fold — maximizes the use of your limited data.'
        return '5-Fold Cross-Validation — trains 5 separate models on different data splits and averages the results for a reliable estimate of real-world performance.'

    def _recommend_metrics(self, task: str, ch: dict) -> list:
        if task == 'regression':
            return ['RMSE (Root Mean Square Error) — penalizes large errors more heavily',
                    'MAE (Mean Absolute Error) — easy to interpret in original units',
                    'R² (R-squared) — percentage of variance explained (higher is better)']
        if ch['has_imbalance']:
            return ['F1-Score — balances precision and recall, ideal for imbalanced data',
                    'AUC-ROC — measures discrimination ability across all thresholds',
                    'Precision-Recall AUC — better than ROC when imbalance is severe',
                    '⚠ Do NOT use Accuracy — it is misleading on imbalanced datasets']
        return ['Accuracy — percentage of correct predictions',
                'F1-Score — good balance of precision and recall',
                'AUC-ROC — measures how well the model separates classes']