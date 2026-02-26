"""DeterministicSummary — builds a clean, pre-calculated JSON summary for LLM consumption.

This module sits between the deterministic analysis pipeline (StructuralAnalyzer,
StatisticalEngine, ModelRecommender) and the InsightGenerator.  Its purpose is to
ensure the LLM receives ONLY pre-computed facts — never raw data rows — so it
cannot hallucinate mathematical conclusions.

Design principle:
  Raw CSV → [Pandas/SciPy deterministic code] → DeterministicSummary (JSON) → LLM interprets
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DeterministicSummary:
    """Produces a structured, LLM-safe digest of all deterministic analysis results.

    Every value in the output dict is a pre-calculated fact.  The LLM's only job
    is to *interpret* and *explain* these numbers — never to compute them.
    """
    __slots__ = ()

    def build(self, blueprint: dict, stats: dict, recommendations: dict,
              user_context: Optional[str] = None) -> Dict[str, Any]:
        """Return a single JSON-serialisable summary for the LLM."""
        basic = blueprint.get('basic_info', {})
        structure = blueprint.get('data_structure', {})
        quality = blueprint.get('quality_issues', [])
        targets = blueprint.get('target_candidates', [])
        profiles = blueprint.get('column_profiles', [])

        red_flags = stats.get('red_flags', [])
        warnings = stats.get('warnings', [])
        correlations = stats.get('correlations', [])
        outlier_summary = stats.get('outlier_summary', [])
        distribution_summary = stats.get('distribution_summary', [])
        patterns = stats.get('patterns', {})
        sentinel_scrub = stats.get('sentinel_scrub', {})

        chars = recommendations.get('characteristics', {})

        return {
            # ── Dataset shape (pre-calculated) ───────────────────────────────
            'shape': {
                'rows': basic.get('rows', 0),
                'columns': basic.get('columns', 0),
                'missing_cells': basic.get('missing_cells', 0),
                'missing_pct': basic.get('missing_percentage', 0),
                'duplicate_rows': basic.get('duplicate_rows', 0),
                'memory_mb': basic.get('memory_mb', 0),
            },
            # ── Structure (pre-calculated) ───────────────────────────────────
            'structure': {
                'type': structure.get('type', 'cross-sectional'),
                'numeric_count': structure.get('numeric_count', 0),
                'categorical_count': structure.get('categorical_count', 0),
                'datetime_columns': structure.get('datetime_columns', []),
                'id_columns': structure.get('id_columns', []),
            },
            # ── Target candidate (pre-calculated) ────────────────────────────
            'target': self._summarize_target(targets),
            # ── Column digest (pre-calculated) ───────────────────────────────
            'column_digest': self._summarize_columns(profiles),
            # ── Quality issues (pre-calculated) ──────────────────────────────
            'quality_issue_counts': self._count_quality_issues(quality),
            'quality_issues_summary': self._summarize_quality_issues(quality),
            # ── Statistical red flags (pre-calculated) ───────────────────────
            'red_flags': [self._summarize_finding(f) for f in red_flags],
            # ── Statistical warnings (pre-calculated) ────────────────────────
            'warnings': [self._summarize_finding(w) for w in warnings],
            # ── Correlations (pre-calculated, top 10) ────────────────────────
            'top_correlations': [
                {'col_a': c['col_a'], 'col_b': c['col_b'],
                 'correlation': c['correlation'], 'strength': c['strength']}
                for c in correlations[:10]
            ],
            # ── Outliers (pre-calculated) ────────────────────────────────────
            'outlier_summary': [
                {'column': o['column'], 'outlier_pct': o['outlier_pct'],
                 'outlier_count': o['outlier_count']}
                for o in outlier_summary[:10]
            ],
            # ── Distribution shapes (pre-calculated) ─────────────────────────
            'distribution_summary': [
                {'column': d['column'], 'skewness': d['skewness'],
                 'kurtosis': d['kurtosis'], 'shape': d['shape']}
                for d in distribution_summary[:10]
            ],
            # ── Sentinel scrub log (pre-calculated) ──────────────────────────
            'sentinels_scrubbed': {
                col: {'values': info['values'], 'cells_replaced': info['cells_replaced']}
                for col, info in sentinel_scrub.items()
            } if sentinel_scrub else {},
            # ── Dataset characteristics (pre-calculated by ModelRecommender) ─
            'characteristics': {
                'has_missing': chars.get('has_missing', False),
                'has_imbalance': chars.get('has_imbalance', False),
                'has_outliers': chars.get('has_outliers', False),
                'has_multicollinearity': chars.get('has_multicollinearity', False),
                'has_categoricals': chars.get('has_categoricals', False),
                'is_large': chars.get('is_large', False),
                'is_small': chars.get('is_small', False),
                'is_high_dimensional': chars.get('is_high_dimensional', False),
            },
            # ── Model recommendation (pre-calculated) ────────────────────────
            'model': {
                'primary': recommendations.get('primary_model', 'Unknown'),
                'task_type': recommendations.get('task_type', 'classification'),
                'confidence': recommendations.get('confidence', 0),
                'reasoning': recommendations.get('reasoning', []),
                'alternatives': [a['model'] for a in recommendations.get('alternatives', [])],
                'preprocessing': recommendations.get('preprocessing_steps', []),
                'cv_strategy': recommendations.get('cv_strategy', ''),
                'metrics': recommendations.get('recommended_metrics', []),
            },
            # ── Patterns (pre-calculated) ────────────────────────────────────
            'patterns': {
                'skewed_columns': patterns.get('skewed_columns', []),
                'extremely_skewed_columns': patterns.get('extremely_skewed_columns', []),
                'high_dimensional': patterns.get('high_dimensional', False),
                'large_dataset': patterns.get('large_dataset', False),
                'small_dataset': patterns.get('small_dataset', False),
            },
            # ── User-provided context (passed through, not computed) ─────────
            'user_context': user_context or None,
        }

    def _summarize_target(self, candidates: list) -> dict:
        if not candidates:
            return {'column': None, 'task_type': 'unknown'}
        c = candidates[0]
        return {
            'column': c.get('column', ''),
            'task_type': c.get('task_type', 'classification'),
            'n_classes': c.get('n_classes'),
            'imbalance_ratio': c.get('imbalance_ratio'),
        }

    def _summarize_columns(self, profiles: list) -> list:
        """Compact column digest: name, kind, missing%, unique count, key stats."""
        digest = []
        for p in profiles[:50]:  # cap at 50 columns for LLM context window
            entry = {
                'name': p.get('name', ''),
                'kind': p.get('kind', 'unknown'),
                'missing_pct': p.get('missing_pct', 0),
                'unique_count': p.get('unique_count', 0),
            }
            # Include pre-calculated stats for numeric columns
            if p.get('kind') in ('numeric', 'numeric_categorical'):
                for k in ('mean', 'std', 'min', 'max', 'median', 'skewness', 'kurtosis'):
                    if k in p:
                        entry[k] = p[k]
            digest.append(entry)
        return digest

    def _count_quality_issues(self, issues: list) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for i in issues:
            t = i.get('type', 'unknown')
            counts[t] = counts.get(t, 0) + 1
        return counts

    def _summarize_quality_issues(self, issues: list) -> list:
        """Return a compact list of quality issues without raw data."""
        return [
            {
                'type': i.get('type', 'unknown'),
                'severity': i.get('severity', 'medium'),
                'column': i.get('column', ''),
                'message': i.get('message', ''),
            }
            for i in issues[:30]  # cap for LLM context
        ]

    def _summarize_finding(self, finding: dict) -> dict:
        """Extract only the pre-calculated facts from a finding."""
        return {
            'type': finding.get('type', 'unknown'),
            'severity': finding.get('severity', 'medium'),
            'column': finding.get('column') or finding.get('columns', ''),
            'message': finding.get('message', ''),
            # Carry forward specific pre-calculated values
            **{k: finding[k] for k in ('majority_pct', 'correlation', 'percentage',
                                        'var1', 'var2', 'skewness')
               if k in finding},
        }
