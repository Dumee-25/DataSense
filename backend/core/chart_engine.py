"""ChartEngine — generates visual analysis charts from pre-computed stats.

Design principles:
  - Works entirely from the pre-computed results dict (no raw DataFrame required).
    This means it can run after the analysis pipeline without holding the original
    data in memory, and fits cleanly into both the PDF and any future HTML renderer.
  - Adaptive to dataset size: charts automatically switch representations when
    column counts would make a naive layout unreadable (e.g. correlation heatmap
    → ranked bar chart for wide datasets).
  - Self-skipping: each chart returns None when its data is absent or trivial,
    so callers never need to check availability themselves.
  - Returns PNG bytes (io.BytesIO) for direct embedding in ReportLab or base64
    encoding for HTML.

Usage:
    engine = ChartEngine()
    charts = engine.generate_all(results_dict)
    # charts is a dict of chart_name → bytes | None
    # None means the chart was skipped (insufficient data)
"""
from __future__ import annotations

import io
import math
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Colour palette — mirrors pdf_generator constants ─────────────────────────
_BRAND      = "#4F46E5"
_ACCENT     = "#F59E0B"
_TEAL       = "#0D9488"
_CRITICAL   = "#EF4444"
_HIGH       = "#F97316"
_MEDIUM     = "#EAB308"
_SUCCESS    = "#10B981"
_MUTED      = "#6B7280"
_EDGE       = "#E5E7EB"
_DARK       = "#111827"
_SOFT_BLUE  = "#EEF2FF"
_COVER_BG   = "#1E1B4B"

# Gradient palette for multi-bar charts
_GRAD = ["#4F46E5", "#7C3AED", "#0D9488", "#F59E0B", "#EF4444",
         "#10B981", "#F97316", "#EC4899", "#6366F1", "#14B8A6"]

# Column caps — beyond these the chart switches rendering strategy
_HEATMAP_MAX_COLS = 14   # above this, use ranked-pairs bar instead of matrix
_BAR_MAX_COLS     = 25   # cap for horizontal bar charts (show top-N only)

# DPI and figure sizes (width × height in inches)
_DPI          = 130
_WIDE         = (7.2, 3.2)   # full-width chart
_WIDE_TALL    = (7.2, 4.0)
_HALF         = (3.4, 2.8)   # two charts side by side (not used yet)


def _mpl():
    """Lazy import matplotlib with non-interactive backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    return plt, mpatches


def _fig_to_bytes(fig) -> bytes:
    """Render a matplotlib figure to PNG bytes and close the figure."""
    plt, _ = _mpl()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _styled_fig(w, h, bg="#FAFAFA"):
    """Create a consistently styled figure."""
    plt, _ = _mpl()
    fig, ax = plt.subplots(figsize=(w, h), facecolor=bg)
    ax.set_facecolor(bg)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_EDGE)
    ax.spines["bottom"].set_color(_EDGE)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.grid(axis="x", color=_EDGE, linewidth=0.6, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    return fig, ax


def _styled_fig_multi(w, h, n_rows=1, n_cols=1, bg="#FAFAFA"):
    """Create a multi-panel figure."""
    plt, _ = _mpl()
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(w, h), facecolor=bg)
    for ax in (axes.flat if hasattr(axes, 'flat') else [axes]):
        ax.set_facecolor(bg)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(_EDGE)
        ax.spines["bottom"].set_color(_EDGE)
        ax.tick_params(colors=_MUTED, labelsize=8)
        ax.set_axisbelow(True)
    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════════
#  ChartEngine
# ═══════════════════════════════════════════════════════════════════════════════
class ChartEngine:
    """Generates all visual analysis charts from a results dict."""
    __slots__ = ()

    def generate_all(self, results: dict) -> Dict[str, Optional[bytes]]:
        """Generate every chart, returning a dict of name → PNG bytes (or None).

        Args:
            results: The top-level results dict produced by the analysis pipeline.
                     Expected keys: 'results' → {dataset_info, structural_analysis,
                     statistical_analysis, model_recommendations, insights}

        Returns:
            Dict with keys: missing_values, correlation, target_distribution,
            outliers, skewness, data_health_radar
            Values are PNG bytes or None if the chart was skipped.
        """
        res   = results.get('results', {})
        st    = res.get('structural_analysis', {})
        stats = res.get('statistical_analysis', {})
        rec   = res.get('model_recommendations', {})
        ins   = res.get('insights', {})

        profiles     = st.get('column_profiles', [])
        correlations = stats.get('correlations', [])
        outliers     = stats.get('outlier_summary', [])
        dist_summary = stats.get('distribution_summary', [])
        basic        = st.get('basic_info', res.get('dataset_info', {}))
        target_cands = st.get('target_candidates', [])

        charts: Dict[str, Optional[bytes]] = {}

        for name, fn, args in [
            ('missing_values',      self._chart_missing,      (profiles, basic)),
            ('correlation',         self._chart_correlation,  (correlations, profiles)),
            ('target_distribution', self._chart_target,       (target_cands, profiles)),
            ('outliers',            self._chart_outliers,     (outliers,)),
            ('skewness',            self._chart_skewness,     (dist_summary,)),
            ('data_health_radar',   self._chart_health_radar, (basic, stats, ins)),
        ]:
            try:
                charts[name] = fn(*args)
            except Exception as e:
                logger.warning(f"ChartEngine: '{name}' failed — {e}")
                charts[name] = None

        return charts

    # ── Chart 1: Missing Values ───────────────────────────────────────────
    def _chart_missing(self, profiles: list, basic: dict) -> Optional[bytes]:
        """Horizontal bar chart of missing % per column. Skips if no missing data."""
        missing = [
            (p['name'], p.get('missing_pct', 0))
            for p in profiles
            if p.get('missing_pct', 0) > 0
        ]
        if not missing:
            return None

        missing.sort(key=lambda x: x[1], reverse=True)
        missing = missing[:_BAR_MAX_COLS]
        labels, values = zip(*missing)

        fig_h = max(2.4, 0.38 * len(labels) + 0.6)
        fig, ax = _styled_fig(7.2, fig_h)

        colors = [
            _CRITICAL if v >= 50 else _HIGH if v >= 20 else _MEDIUM if v >= 5 else _TEAL
            for v in values
        ]
        bars = ax.barh(range(len(labels)), values, color=colors, height=0.65, zorder=3)

        # Value labels on bars
        for bar, v in zip(bars, values):
            x = bar.get_width()
            ax.text(x + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}%", va="center", ha="left",
                    fontsize=8, color=_DARK, fontweight="bold")

        # Threshold lines
        overall_missing = basic.get('missing_percentage', 0)
        ax.axvline(5,  color=_MEDIUM,   linestyle=":", linewidth=1.2, alpha=0.8, label="5% threshold")
        ax.axvline(20, color=_HIGH,     linestyle=":", linewidth=1.2, alpha=0.8, label="20% threshold")
        ax.axvline(50, color=_CRITICAL, linestyle=":", linewidth=1.2, alpha=0.8, label="50% threshold")

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(
            [l[:28] + "…" if len(l) > 28 else l for l in labels],
            fontsize=8.5, color=_DARK
        )
        ax.set_xlabel("Missing %", fontsize=9, color=_MUTED)
        ax.set_xlim(0, max(values) * 1.18)
        ax.invert_yaxis()

        title = f"Missing Values by Column  ·  Overall {overall_missing:.1f}% missing"
        if len(missing) < sum(1 for p in profiles if p.get('missing_pct', 0) > 0):
            title += f"  (top {len(missing)} shown)"
        ax.set_title(title, fontsize=10, fontweight="bold", color=_DARK, pad=10)

        plt, _ = _mpl()
        plt.tight_layout()
        return _fig_to_bytes(fig)

    # ── Chart 2: Correlation ──────────────────────────────────────────────
    def _chart_correlation(self, correlations: list, profiles: list) -> Optional[bytes]:
        """Correlation chart. Uses a heatmap for ≤14 cols, ranked bar for wider datasets."""
        if not correlations:
            return None

        # Gather all unique column names involved
        all_cols = list(dict.fromkeys(
            c for pair in correlations for c in (pair['col_a'], pair['col_b'])
        ))

        if len(all_cols) <= _HEATMAP_MAX_COLS:
            return self._correlation_heatmap(correlations, all_cols)
        else:
            return self._correlation_ranked_bar(correlations)

    def _correlation_heatmap(self, correlations: list, cols: list) -> bytes:
        """Full symmetric heatmap for small/medium datasets."""
        import numpy as np
        plt, _ = _mpl()

        n = len(cols)
        idx = {c: i for i, c in enumerate(cols)}
        mat = [[0.0] * n for _ in range(n)]
        for i in range(n):
            mat[i][i] = 1.0
        for pair in correlations:
            i, j = idx.get(pair['col_a']), idx.get(pair['col_b'])
            if i is not None and j is not None:
                v = pair['correlation']
                mat[i][j] = v
                mat[j][i] = v

        mat_np = [[mat[i][j] for j in range(n)] for i in range(n)]
        size = max(4.5, n * 0.48)
        fig, ax = _styled_fig_multi(size, size * 0.88, 1, 1)

        im = ax.imshow(mat_np, vmin=-1, vmax=1, cmap="RdYlBu_r", aspect="auto")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        short = [c[:14] + "…" if len(c) > 14 else c for c in cols]
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7.5)
        ax.set_yticklabels(short, fontsize=7.5)

        # Annotate cells
        for i in range(n):
            for j in range(n):
                v = mat_np[i][j]
                if abs(v) >= 0.3 or i == j:
                    col = "white" if abs(v) > 0.65 else _DARK
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6.5 if n > 10 else 8, color=col)

        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("Pearson r", fontsize=8, color=_MUTED)

        ax.set_title("Correlation Heatmap", fontsize=10, fontweight="bold",
                     color=_DARK, pad=10)
        plt.tight_layout()
        return _fig_to_bytes(fig)

    def _correlation_ranked_bar(self, correlations: list) -> bytes:
        """Ranked horizontal bar of top correlation pairs for wide datasets."""
        top = sorted(correlations, key=lambda p: abs(p['correlation']), reverse=True)[:20]
        if not top:
            return None

        labels = [f"{p['col_a'][:14]}↔{p['col_b'][:14]}" for p in top]
        values = [p['correlation'] for p in top]

        fig_h = max(3.0, 0.4 * len(top) + 0.8)
        fig, ax = _styled_fig(7.2, fig_h)

        colors = [_CRITICAL if abs(v) >= 0.9 else _HIGH if abs(v) >= 0.7 else _BRAND for v in values]
        bars = ax.barh(range(len(labels)), values, color=colors, height=0.65, zorder=3)

        for bar, v in zip(bars, values):
            xpos = v + (0.01 if v >= 0 else -0.01)
            ha = "left" if v >= 0 else "right"
            ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                    f"{v:+.2f}", va="center", ha=ha, fontsize=8,
                    color=_DARK, fontweight="bold")

        ax.axvline(0, color=_DARK, linewidth=0.8)
        ax.axvline(0.7,  color=_HIGH,     linestyle=":", linewidth=1, alpha=0.7)
        ax.axvline(-0.7, color=_HIGH,     linestyle=":", linewidth=1, alpha=0.7)
        ax.axvline(0.9,  color=_CRITICAL, linestyle=":", linewidth=1, alpha=0.7)
        ax.axvline(-0.9, color=_CRITICAL, linestyle=":", linewidth=1, alpha=0.7)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8, color=_DARK)
        ax.set_xlabel("Pearson r", fontsize=9, color=_MUTED)
        ax.set_xlim(-1.15, 1.15)
        ax.invert_yaxis()

        n_total = len(correlations)
        title = f"Top {len(top)} Column Correlations"
        if n_total > len(top):
            title += f"  ·  {n_total} pairs total"
        ax.set_title(title, fontsize=10, fontweight="bold", color=_DARK, pad=10)

        plt, _ = _mpl()
        plt.tight_layout()
        return _fig_to_bytes(fig)

    # ── Chart 3: Target Distribution ──────────────────────────────────────
    def _chart_target(self, target_cands: list, profiles: list) -> Optional[bytes]:
        """Bar chart of target class distribution. Works for classification and regression."""
        if not target_cands:
            return None

        tgt = target_cands[0]
        col_name = tgt.get('column', '')
        task = tgt.get('task_type', 'classification')

        # Find the column profile to get top_values
        profile = next((p for p in profiles if p['name'] == col_name), {})
        top_vals = profile.get('top_values', {})

        if not top_vals and task == 'regression':
            # For regression targets, show a simple stat summary instead
            return self._chart_target_regression(profile, col_name)

        if not top_vals:
            return None

        labels = list(top_vals.keys())
        counts = list(top_vals.values())
        total  = sum(counts)
        pcts   = [c / total * 100 for c in counts]

        # Colour: highlight imbalance
        if len(counts) >= 2:
            max_c = max(counts)
            bar_colors = [
                _CRITICAL if c == max_c and (max_c / total) > 0.7
                else _HIGH if c == max_c and (max_c / total) > 0.5
                else _BRAND if i == 0
                else _GRAD[i % len(_GRAD)]
                for i, c in enumerate(counts)
            ]
        else:
            bar_colors = [_BRAND]

        # Choose layout based on number of classes
        if len(labels) <= 8:
            fig, ax = _styled_fig(*_WIDE)
            ax.grid(axis="y", color=_EDGE, linewidth=0.6, linestyle="--", alpha=0.7)
            ax.grid(axis="x", visible=False)
            bars = ax.bar(range(len(labels)), counts, color=bar_colors, width=0.65, zorder=3)
            for bar, c, p in zip(bars, counts, pcts):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                        f"{c:,}\n({p:.1f}%)", ha="center", va="bottom",
                        fontsize=8, color=_DARK, fontweight="bold")
            short_labels = [str(l)[:18] + "…" if len(str(l)) > 18 else str(l) for l in labels]
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(short_labels, fontsize=8.5, rotation=20 if len(labels) > 4 else 0)
            ax.set_ylabel("Count", fontsize=9, color=_MUTED)
        else:
            # Many classes → horizontal bar
            fig_h = max(3.2, 0.35 * len(labels) + 0.8)
            fig, ax = _styled_fig(7.2, fig_h)
            bars = ax.barh(range(len(labels)), counts, color=bar_colors[:len(labels)], height=0.65, zorder=3)
            for bar, c, p in zip(bars, counts, pcts):
                ax.text(bar.get_width() + max(counts) * 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{c:,} ({p:.1f}%)", va="center", ha="left", fontsize=7.5, color=_DARK)
            short_labels = [str(l)[:22] + "…" if len(str(l)) > 22 else str(l) for l in labels]
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(short_labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Count", fontsize=9, color=_MUTED)

        imbalance_note = ""
        ir = tgt.get('imbalance_ratio')
        if ir and ir > 3:
            imbalance_note = f"  ·  ⚠ Imbalance ratio {ir:.1f}x"

        ax.set_title(
            f'Target: "{col_name}"  ·  {task.title()}{imbalance_note}',
            fontsize=10, fontweight="bold", color=_DARK, pad=10
        )

        plt, _ = _mpl()
        plt.tight_layout()
        return _fig_to_bytes(fig)

    def _chart_target_regression(self, profile: dict, col_name: str) -> Optional[bytes]:
        """Simple stat summary bar for regression targets (no top_values available)."""
        stats_map = {k: profile.get(k) for k in ('min', 'max', 'mean', 'median') if profile.get(k) is not None}
        if not stats_map:
            return None

        fig, ax = _styled_fig(*_WIDE)
        ax.grid(axis="y", color=_EDGE, linewidth=0.6, linestyle="--", alpha=0.7)
        ax.grid(axis="x", visible=False)
        labels = list(stats_map.keys())
        vals   = [float(v) for v in stats_map.values()]
        colors = [_TEAL, _BRAND, _ACCENT, _HIGH][:len(labels)]
        bars   = ax.bar(labels, vals, color=colors, width=0.5, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{v:,.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=_DARK)
        ax.set_title(f'Target Distribution: "{col_name}"  ·  Regression',
                     fontsize=10, fontweight="bold", color=_DARK, pad=10)
        plt, _ = _mpl()
        plt.tight_layout()
        return _fig_to_bytes(fig)

    # ── Chart 4: Outlier Severity ─────────────────────────────────────────
    def _chart_outliers(self, outlier_summary: list) -> Optional[bytes]:
        """Horizontal bar of outlier % per column. Skips if nothing significant."""
        significant = [o for o in outlier_summary if o.get('outlier_pct', 0) >= 1.0]
        if not significant:
            return None

        significant.sort(key=lambda o: o['outlier_pct'], reverse=True)
        significant = significant[:_BAR_MAX_COLS]
        labels = [o['column'] for o in significant]
        values = [o['outlier_pct'] for o in significant]
        counts = [o.get('outlier_count', 0) for o in significant]

        fig_h = max(2.4, 0.38 * len(labels) + 0.6)
        fig, ax = _styled_fig(7.2, fig_h)

        colors = [
            _CRITICAL if v >= 15 else _HIGH if v >= 5 else _MEDIUM
            for v in values
        ]
        bars = ax.barh(range(len(labels)), values, color=colors, height=0.65, zorder=3)

        for bar, v, c in zip(bars, values, counts):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}%  ({c:,} rows)", va="center", ha="left",
                    fontsize=7.5, color=_DARK)

        ax.axvline(5,  color=_HIGH,     linestyle=":", linewidth=1.2, alpha=0.8)
        ax.axvline(15, color=_CRITICAL, linestyle=":", linewidth=1.2, alpha=0.8)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(
            [l[:28] + "…" if len(l) > 28 else l for l in labels],
            fontsize=8.5, color=_DARK
        )
        ax.set_xlabel("Outlier %", fontsize=9, color=_MUTED)
        ax.set_xlim(0, max(values) * 1.30)
        ax.invert_yaxis()

        n_total = len(outlier_summary)
        title = "Outlier Severity by Column"
        if n_total > len(significant):
            title += f"  (top {len(significant)} of {n_total} affected columns)"
        ax.set_title(title, fontsize=10, fontweight="bold", color=_DARK, pad=10)

        plt, _ = _mpl()
        plt.tight_layout()
        return _fig_to_bytes(fig)

    # ── Chart 5: Skewness Overview ────────────────────────────────────────
    def _chart_skewness(self, distribution_summary: list) -> Optional[bytes]:
        """Horizontal bar of skewness per column. Only shows notable skew (|skew| ≥ 1)."""
        notable = [
            d for d in distribution_summary
            if d.get('skewness') is not None and abs(d['skewness']) >= 1.0
        ]
        if not notable:
            return None

        notable.sort(key=lambda d: abs(d['skewness']), reverse=True)
        notable = notable[:_BAR_MAX_COLS]
        labels = [d['column'] for d in notable]
        values = [d['skewness'] for d in notable]

        fig_h = max(2.4, 0.38 * len(labels) + 0.6)
        fig, ax = _styled_fig(7.2, fig_h)

        colors = []
        for v in values:
            av = abs(v)
            if av >= 50:   colors.append(_CRITICAL)
            elif av >= 10: colors.append(_HIGH)
            elif av >= 3:  colors.append(_MEDIUM)
            else:          colors.append(_BRAND)

        bars = ax.barh(range(len(labels)), values, color=colors, height=0.65, zorder=3)

        for bar, v in zip(bars, values):
            xpos = v + (0.3 if v >= 0 else -0.3)
            ha   = "left" if v >= 0 else "right"
            ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}", va="center", ha=ha, fontsize=8, color=_DARK, fontweight="bold")

        ax.axvline(0,   color=_DARK,     linewidth=0.8)
        ax.axvline(3,   color=_MEDIUM,   linestyle=":", linewidth=1.1, alpha=0.7, label="moderate skew")
        ax.axvline(-3,  color=_MEDIUM,   linestyle=":", linewidth=1.1, alpha=0.7)
        ax.axvline(10,  color=_HIGH,     linestyle=":", linewidth=1.1, alpha=0.7, label="high skew")
        ax.axvline(-10, color=_HIGH,     linestyle=":", linewidth=1.1, alpha=0.7)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(
            [l[:28] + "…" if len(l) > 28 else l for l in labels],
            fontsize=8.5, color=_DARK
        )
        ax.set_xlabel("Skewness", fontsize=9, color=_MUTED)
        ax.invert_yaxis()

        title = "Distribution Skewness  ·  Positive = right tail, Negative = left tail"
        ax.set_title(title, fontsize=9.5, fontweight="bold", color=_DARK, pad=10)

        plt, _ = _mpl()
        plt.tight_layout()
        return _fig_to_bytes(fig)

    # ── Chart 6: Data Health Radar ────────────────────────────────────────
    def _chart_health_radar(self, basic: dict, stats: dict, ins: dict) -> Optional[bytes]:
        """Radar / spider chart showing 6 data health dimensions at a glance."""
        import numpy as np
        plt, _ = _mpl()

        # ── Compute dimension scores (0–100, higher = healthier) ──────────
        missing_pct = basic.get('missing_percentage', 0)
        dup_rows    = basic.get('duplicate_rows', 0)
        total_rows  = max(basic.get('rows', 1), 1)
        dup_pct     = dup_rows / total_rows * 100

        red_flags   = stats.get('red_flags', [])
        warnings    = stats.get('warnings', [])
        outliers    = stats.get('outlier_summary', [])
        dist_sum    = stats.get('distribution_summary', [])

        # Avg outlier pct across affected columns
        avg_outlier = (sum(o.get('outlier_pct', 0) for o in outliers) / max(len(outliers), 1)) if outliers else 0
        # Count extreme skew columns
        extreme_skew = sum(1 for d in dist_sum if abs(d.get('skewness', 0)) >= 10)
        # Count high/critical findings
        crit_count = ins.get('severity_breakdown', {}).get('critical', 0)
        high_count = ins.get('severity_breakdown', {}).get('high', 0)

        def _score_completeness():
            if missing_pct == 0:   return 100
            if missing_pct < 2:    return 90
            if missing_pct < 5:    return 75
            if missing_pct < 15:   return 55
            if missing_pct < 30:   return 35
            return 15

        def _score_uniqueness():
            if dup_pct == 0:       return 100
            if dup_pct < 1:        return 85
            if dup_pct < 5:        return 65
            if dup_pct < 10:       return 45
            return 25

        def _score_consistency():
            # Based on quality issue types (whitespace, disguised missing, mixed types)
            all_issues = list(red_flags) + list(warnings)
            bad = sum(1 for i in all_issues if i.get('type') in
                      ('whitespace_issues', 'disguised_missing', 'mixed_types', 'constant_column'))
            if bad == 0:   return 100
            if bad <= 2:   return 80
            if bad <= 5:   return 60
            if bad <= 10:  return 40
            return 20

        def _score_distribution():
            if extreme_skew == 0:  return 100
            if extreme_skew <= 2:  return 75
            if extreme_skew <= 5:  return 50
            return 25

        def _score_outliers():
            if avg_outlier == 0:   return 100
            if avg_outlier < 2:    return 85
            if avg_outlier < 5:    return 65
            if avg_outlier < 10:   return 45
            if avg_outlier < 20:   return 25
            return 10

        def _score_reliability():
            penalty = crit_count * 20 + high_count * 8
            return max(0, 100 - penalty)

        dimensions = [
            ("Completeness",  _score_completeness()),
            ("Uniqueness",    _score_uniqueness()),
            ("Consistency",   _score_consistency()),
            ("Distribution",  _score_distribution()),
            ("Outliers",      _score_outliers()),
            ("Reliability",   _score_reliability()),
        ]

        labels  = [d[0] for d in dimensions]
        scores  = [d[1] for d in dimensions]
        N       = len(labels)
        angles  = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]   # close the polygon
        vals    = [s / 100 for s in scores] + [scores[0] / 100]

        fig = plt.figure(figsize=(3.8, 3.2), facecolor="#FAFAFA")
        ax  = fig.add_subplot(111, polar=True)
        ax.set_facecolor("#FAFAFA")

        # Grid rings
        for r in [0.25, 0.5, 0.75, 1.0]:
            ax.plot(angles, [r] * (N + 1), color=_EDGE, linewidth=0.6, linestyle="--")

        # Fill area
        ax.fill(angles, vals, color=_BRAND, alpha=0.18)
        ax.plot(angles, vals, color=_BRAND, linewidth=2.0, linestyle="-")

        # Data points
        for ang, val, score in zip(angles[:-1], vals[:-1], scores):
            color = (_SUCCESS if score >= 80 else _MEDIUM if score >= 55
                     else _HIGH if score >= 35 else _CRITICAL)
            ax.plot(ang, val, "o", color=color, markersize=7, zorder=5)

        # Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            [f"{l}\n{s:.0f}" for l, s in zip(labels, scores)],
            fontsize=7.5, color=_DARK
        )
        ax.set_yticks([])
        ax.spines["polar"].set_color(_EDGE)
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)

        overall = sum(scores) / len(scores)
        health_label = (
            "Excellent" if overall >= 85 else
            "Good"      if overall >= 70 else
            "Fair"      if overall >= 50 else
            "Poor"
        )
        ax.set_title(f"Data Health  ·  {health_label} ({overall:.0f}/100)",
                     fontsize=9.5, fontweight="bold", color=_DARK, pad=14)

        plt.tight_layout()
        return _fig_to_bytes(fig)
