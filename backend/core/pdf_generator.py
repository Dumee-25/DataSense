"""PDFReportGenerator — clean, modern PDF report with consistent layout.

Key design decisions:
  - Slate/emerald colour scheme (professional, easy on the eyes).
  - Every chart + its title sits in a KeepTogether block so they never split
    across pages; charts that are too tall get their own page.
  - Generous whitespace, soft dividers, and a clear type hierarchy
    (Helvetica-Bold for headings, Helvetica for body).
  - Stat cards, callout boxes, and finding cards share a unified card style.
  - Colombo timezone stamps preserved.
"""
import io
import math
from datetime import datetime, timezone, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io as _io
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether, Flowable, Image,
)
from reportlab.graphics.shapes import (
    Drawing, Circle, Rect, Line, Polygon, String,
)
from reportlab.graphics import renderPDF

# ── Colombo Time (UTC+5:30) ──────────────────────────────────────────────────
COLOMBO_TZ = timezone(timedelta(hours=5, minutes=30))

def _now_colombo():
    return datetime.now(COLOMBO_TZ).strftime("%B %d, %Y at %H:%M IST (Colombo)")


# ═══════════════════════════════════════════════════════════════════════════════
# COLOUR PALETTE — Slate + Emerald
# ═══════════════════════════════════════════════════════════════════════════════
PRIMARY     = colors.HexColor("#0F172A")   # Slate-900  – cover & headings
PRIMARY_MID = colors.HexColor("#334155")   # Slate-700
ACCENT      = colors.HexColor("#059669")   # Emerald-600
ACCENT_LIGHT = colors.HexColor("#D1FAE5")  # Emerald-100
ACCENT_DARK = colors.HexColor("#065F46")   # Emerald-800

CRITICAL    = colors.HexColor("#DC2626")   # Red-600
HIGH        = colors.HexColor("#EA580C")   # Orange-600
MEDIUM      = colors.HexColor("#CA8A04")   # Yellow-600
SUCCESS     = colors.HexColor("#16A34A")   # Green-600

CARD_BG     = colors.HexColor("#F8FAFC")   # Slate-50
SOFT_GREEN  = colors.HexColor("#ECFDF5")   # Emerald-50
SOFT_AMBER  = colors.HexColor("#FFFBEB")   # Amber-50
SOFT_RED    = colors.HexColor("#FEF2F2")   # Red-50
SOFT_BLUE   = colors.HexColor("#F0F9FF")   # Sky-50
LAVENDER    = colors.HexColor("#F5F3FF")   # Violet-50

DARK_TEXT   = colors.HexColor("#1E293B")   # Slate-800
BODY_TEXT   = colors.HexColor("#334155")   # Slate-700
MUTED       = colors.HexColor("#94A3B8")   # Slate-400
BORDER      = colors.HexColor("#E2E8F0")   # Slate-200
WHITE       = colors.white

PAGE_W, PAGE_H = letter
MARGIN = 0.75 * inch
CW = PAGE_W - 2 * MARGIN           # content width
FRAME_H = PAGE_H - 2 * MARGIN - 0.35 * inch  # usable content height


# ═══════════════════════════════════════════════════════════════════════════════
# ICON FLOWABLES  (small inline drawings)
# ═══════════════════════════════════════════════════════════════════════════════
class _IconFlowable(Flowable):
    def __init__(self, drawing, width=36, height=36):
        Flowable.__init__(self)
        self._drawing = drawing
        self.width = width
        self.height = height

    def draw(self):
        renderPDF.draw(self._drawing, self.canv, 0, 0)


def _make_icon(bg_color, fg_cb, size=36):
    """Generic rounded-square icon builder."""
    d = Drawing(size, size)
    d.add(Rect(0, 0, size, size, rx=7, ry=7,
               fillColor=bg_color, strokeColor=None))
    fg_cb(d, size)
    return _IconFlowable(d, size, size)


def _icon_dataset(size=36):
    def _fg(d, s):
        for y_off in [0, 9, 18]:
            y = 5 + y_off
            d.add(Rect(7, y, s - 14, 6, fillColor=WHITE if y_off != 18 else ACCENT,
                        strokeColor=None, rx=1, ry=1))
        d.add(Line(s // 2, 5, s // 2, 29, strokeColor=colors.Color(1, 1, 1, 0.5), strokeWidth=0.7))
    return _make_icon(ACCENT, _fg, size)


def _icon_warning(size=36, color=None):
    color = color or HIGH
    def _fg(d, s):
        cx = s / 2
        d.add(Polygon([cx, s - 6, 5, 6, s - 5, 6],
                       fillColor=WHITE, strokeColor=None))
        d.add(String(cx - 3.5, 9, '!', fontSize=14, fillColor=color, fontName='Helvetica-Bold'))
    return _make_icon(color, _fg, size)


def _icon_critical(size=36):
    def _fg(d, s):
        c = s / 2
        d.add(Circle(c, c, 10, fillColor=WHITE, strokeColor=None))
        m = 12
        d.add(Line(m, s - m, s - m, m, strokeColor=CRITICAL, strokeWidth=2.5))
        d.add(Line(m, m, s - m, s - m, strokeColor=CRITICAL, strokeWidth=2.5))
    return _make_icon(CRITICAL, _fg, size)


def _icon_model(size=36):
    def _fg(d, s):
        nodes_l = [(10, 26), (10, 18), (10, 10)]
        nodes_r = [(26, 22), (26, 14)]
        for ax, ay in nodes_l:
            for bx, by in nodes_r:
                d.add(Line(ax, ay, bx, by, strokeColor=colors.Color(1, 1, 1, 0.5), strokeWidth=0.7))
        for x, y in nodes_l + nodes_r:
            d.add(Circle(x, y, 3.5, fillColor=WHITE, strokeColor=None))
    return _make_icon(PRIMARY_MID, _fg, size)


def _icon_lightning(size=36):
    def _fg(d, s):
        pts = [20, 30, 14, 19, 18, 19, 12, 6, 20, 17, 16, 17, 20, 30]
        d.add(Polygon(pts, fillColor=WHITE, strokeColor=None))
    return _make_icon(MEDIUM, _fg, size)


def _icon_link(size=36):
    def _fg(d, s):
        mid = s // 2
        d.add(Circle(12, mid, 6, fillColor=WHITE, strokeColor=None))
        d.add(Circle(24, mid, 6, fillColor=WHITE, strokeColor=None))
        d.add(Line(18, mid, 18, mid, strokeColor=colors.Color(1, 1, 1, 0.5), strokeWidth=2))
    return _make_icon(ACCENT, _fg, size)


def _icon_balance(size=36):
    def _fg(d, s):
        cx = s / 2
        d.add(Line(cx, 28, cx, 14, strokeColor=WHITE, strokeWidth=2))
        d.add(Line(8, 20, s - 8, 14, strokeColor=WHITE, strokeWidth=1.5))
        d.add(Rect(5, 16, 10, 6, rx=2, ry=2, fillColor=WHITE, strokeColor=None))
        d.add(Rect(s - 15, 8, 10, 6, rx=2, ry=2, fillColor=WHITE, strokeColor=None))
    return _make_icon(CRITICAL, _fg, size)


def _icon_checklist(size=36):
    def _fg(d, s):
        for i, y in enumerate([24, 16, 8]):
            d.add(Rect(7, y, 7, 5, rx=1, ry=1,
                        fillColor=WHITE, strokeColor=None))
            d.add(Line(18, y + 2.5, s - 7, y + 2.5, strokeColor=colors.Color(1, 1, 1, 0.6), strokeWidth=1.2))
    return _make_icon(ACCENT, _fg, size)


def _icon_chart(size=36):
    def _fg(d, s):
        bw = 5
        bars = [(8, 10), (16, 18), (24, 14)]
        for x, h in bars:
            d.add(Rect(x, 6, bw, h, fillColor=WHITE, strokeColor=None, rx=1, ry=1))
        d.add(Line(6, 6, s - 6, 6, strokeColor=colors.Color(1, 1, 1, 0.6), strokeWidth=1))
    return _make_icon(PRIMARY_MID, _fg, size)


# ═══════════════════════════════════════════════════════════════════════════════
# PARAGRAPH STYLES
# ═══════════════════════════════════════════════════════════════════════════════
_S = None

def _s():
    global _S
    if _S:
        return _S
    ps = lambda n, **kw: ParagraphStyle(n, **kw)
    _S = {
        # Cover page
        'ct':   ps('ct', fontName='Helvetica-Bold', fontSize=36, textColor=WHITE, leading=42, spaceAfter=4),
        'cs':   ps('cs', fontName='Helvetica',      fontSize=14, textColor=colors.HexColor("#94A3B8"), spaceAfter=4),
        'cm':   ps('cm', fontName='Helvetica',      fontSize=10, textColor=colors.HexColor("#CBD5E1"), leading=15),

        # Section heading (used with icons)
        'sh':   ps('sh', fontName='Helvetica-Bold', fontSize=16, textColor=DARK_TEXT,
                    spaceBefore=18, spaceAfter=6, leading=20),

        # Sub-heading
        'sub':  ps('sub', fontName='Helvetica-Bold', fontSize=11, textColor=PRIMARY_MID,
                    spaceBefore=10, spaceAfter=4, leading=15),

        # Body text
        'b':    ps('b',   fontName='Helvetica',      fontSize=10, textColor=BODY_TEXT, leading=16, spaceAfter=4),
        'bb':   ps('bb',  fontName='Helvetica-Bold', fontSize=10, textColor=DARK_TEXT, leading=15),
        'sb':   ps('sb',  fontName='Helvetica',      fontSize=11, textColor=DARK_TEXT, leading=18),

        # Captions & labels
        'cap':  ps('cap', fontName='Helvetica',      fontSize=8,  textColor=MUTED, leading=11),

        # Contextual styles
        'act':  ps('act', fontName='Helvetica', fontSize=10, textColor=ACCENT_DARK,   leading=15, leftIndent=12),
        'imp':  ps('imp', fontName='Helvetica', fontSize=10, textColor=colors.HexColor("#9A3412"), leading=15),
        'dd':   ps('dd',  fontName='Helvetica', fontSize=9,  textColor=colors.HexColor("#4C1D95"), leading=14, leftIndent=8),

        # Stat card numbers
        'num':      ps('num',      fontName='Helvetica-Bold', fontSize=22, textColor=ACCENT, leading=26),
        'numlabel': ps('numlabel', fontName='Helvetica',      fontSize=8,  textColor=MUTED,  leading=10),
    }
    return _S


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE / LAYOUT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _ts(*cmds):
    return TableStyle(list(cmds))

_GRID_BASE = [
    ('FONTNAME',      (0, 0), (-1, -1), 'Helvetica'),
    ('FONTSIZE',      (0, 0), (-1, -1), 9),
    ('GRID',          (0, 0), (-1, -1), 0.5, BORDER),
    ('LEFTPADDING',   (0, 0), (-1, -1), 10),
    ('RIGHTPADDING',  (0, 0), (-1, -1), 10),
    ('TOPPADDING',    (0, 0), (-1, -1), 8),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ('TEXTCOLOR',     (0, 0), (-1, -1), BODY_TEXT),
]

_HEADER_ROW = [
    ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
    ('TEXTCOLOR',  (0, 0), (-1, 0), WHITE),
    ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE',   (0, 0), (-1, 0), 9),
]


def _data_table(rows, widths, extra=None):
    t = Table(rows, colWidths=widths)
    cmds = _HEADER_ROW + _GRID_BASE + [
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, CARD_BG]),
    ]
    if extra:
        cmds += extra
    t.setStyle(_ts(*cmds))
    return t


def _section_header(icon_fn, title, s):
    """Icon + title row for each section."""
    icon = icon_fn()
    row = Table([[icon, Paragraph(title, s['sh'])]], colWidths=[48, CW - 48])
    row.setStyle(_ts(
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (0, -1), 0),
    ))
    return row


def _divider(color=None):
    """Thin coloured rule under section headers."""
    return HRFlowable(width="100%", thickness=1.2, color=color or ACCENT, spaceAfter=12)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════
def _on_first(c, _doc):
    """Cover page: dark slate background with a subtle emerald accent bar."""
    c.saveState()
    # Full-page dark background
    c.setFillColor(PRIMARY)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    # Accent bar at top
    c.setFillColor(ACCENT)
    c.rect(0, PAGE_H - 6, PAGE_W, 6, fill=1, stroke=0)
    # Subtle circle decorations (low-alpha)
    c.setFillColor(colors.HexColor("#1E293B"))
    c.setFillAlpha(0.35)
    c.circle(PAGE_W - 40, PAGE_H - 40, 130, fill=1, stroke=0)
    c.setFillAlpha(0.25)
    c.circle(50, 60, 100, fill=1, stroke=0)
    c.restoreState()


def _on_later(c, d):
    """Subsequent pages: thin top accent line, header bar, page number footer."""
    c.saveState()
    # Top accent line
    c.setFillColor(ACCENT)
    c.rect(0, PAGE_H - 3, PAGE_W, 3, fill=1, stroke=0)
    # Header strip
    c.setFillColor(CARD_BG)
    c.rect(0, PAGE_H - 22, PAGE_W, 19, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 7.5)
    c.setFillColor(PRIMARY_MID)
    c.drawString(MARGIN, PAGE_H - 16, "DataSense — Analysis Report")
    c.setFont("Helvetica", 7.5)
    c.setFillColor(MUTED)
    c.drawRightString(PAGE_W - MARGIN, PAGE_H - 16, d.filename_label)

    # Footer
    c.setStrokeColor(BORDER)
    c.setLineWidth(0.4)
    c.line(MARGIN, 34, PAGE_W - MARGIN, 34)
    c.setFont("Helvetica", 7)
    c.setFillColor(MUTED)
    c.drawString(MARGIN, 22, f"Generated {d.generated_at}")
    c.drawRightString(PAGE_W - MARGIN, 22, f"Page {d.page}")
    c.restoreState()


# ═══════════════════════════════════════════════════════════════════════════════
# PDF REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════
class PDFReportGenerator:

    # ── Parse helper ───────────────────────────────────────────────────────
    def _parse(self, results):
        res = results.get('results', {})
        return {
            'job':   {
                'filename': results.get('filename', 'Unknown'),
                'completed_at': results.get('completed_at', ''),
                'processing_time': results.get('processing_time_seconds', 0),
            },
            'di':    res.get('dataset_info', {}),
            'st':    res.get('structural_analysis', {}),
            'stats': res.get('statistical_analysis', {}),
            'rec':   res.get('model_recommendations', {}),
            'ins':   res.get('insights', {}),
            'at':    _now_colombo(),
            '_raw_results': results,
        }

    # ── Build story (page flow) ───────────────────────────────────────────
    def _build_story(self, d):
        s = _s()
        ins, st, rec = d['ins'], d['st'], d['rec']

        story  = self._cover(s, d['job'], d['di'], d['at'], ins) + [PageBreak()]
        story += self._summary(s, ins, d['di'])
        story += self._dataset_overview(s, d['di'], st)
        story += self._findings(s, ins)
        story += self._visual_analysis(s, d.get('_raw_results', {}))
        story += self._model_section(s, ins, rec)
        story += self._quick_wins(s, ins)
        story += self._column_relationships(s, ins)
        story += self._imbalance_guidance(s, ins)
        story += self._column_profiles(s, st)
        story += [PageBreak()] + self._closing(s, d['at'])
        return story

    # ── Document factory ──────────────────────────────────────────────────
    def _make_doc(self, target, label, at):
        doc = SimpleDocTemplate(
            target, pagesize=letter,
            leftMargin=MARGIN, rightMargin=MARGIN,
            topMargin=MARGIN + 0.35 * inch,
            bottomMargin=MARGIN + 0.15 * inch,
        )
        doc.filename_label = label
        doc.generated_at   = at
        return doc

    # ── Public API ─────────────────────────────────────────────────────────
    def generate(self, results: dict, output_path: str) -> str:
        d = self._parse(results)
        doc = self._make_doc(output_path, d['job']['filename'], d['at'])
        doc.build(self._build_story(d), onFirstPage=_on_first, onLaterPages=_on_later)
        return output_path

    def generate_bytes(self, results: dict) -> bytes:
        buf = io.BytesIO()
        self.generate_to_buffer(results, buf)
        return buf.getvalue()

    def generate_to_buffer(self, results: dict, buffer: io.BytesIO):
        d = self._parse(results)
        doc = self._make_doc(buffer, d['job']['filename'], d['at'])
        doc.build(self._build_story(d), onFirstPage=_on_first, onLaterPages=_on_later)

    # ══════════════════════════════════════════════════════════════════════
    #  COVER PAGE
    # ══════════════════════════════════════════════════════════════════════
    def _cover(self, s, job, di, at, ins=None):
        items = [
            Spacer(1, 2.0 * inch),
            Paragraph("DataSense", s['ct']),
            Spacer(1, 2),
            Paragraph("Automated Data Analysis Report", s['cs']),
            Spacer(1, 0.3 * inch),
            HRFlowable(width="50%", thickness=2, color=ACCENT, spaceAfter=20),
        ]
        r, c = di.get('rows', '—'), di.get('columns', '—')
        r_fmt = f"{r:,}" if isinstance(r, (int, float)) else str(r)
        items += [
            Paragraph(f"<b>File:</b>  {job['filename']}", s['cm']),
            Spacer(1, 4),
            Paragraph(f"<b>Dataset:</b>  {r_fmt} rows  ×  {c} columns", s['cm']),
            Spacer(1, 4),
            Paragraph(f"<b>Generated:</b>  {at}", s['cm']),
        ]
        if job.get('processing_time'):
            items += [Spacer(1, 4),
                      Paragraph(f"<b>Analysis time:</b>  {round(job['processing_time'], 1)} seconds", s['cm'])]
        if ins and ins.get('llm_enhanced'):
            items += [Spacer(1, 4),
                      Paragraph(f"<b>AI-enhanced</b> via {ins.get('llm_provider', 'LLM')}", s['cm'])]
        return items

    # ══════════════════════════════════════════════════════════════════════
    #  REUSABLE COMPONENTS — box / badge / card wrapper
    # ══════════════════════════════════════════════════════════════════════
    def _box(self, text, s, bg=None, border=None):
        """Callout box with left accent border."""
        bg = bg or SOFT_GREEN
        border = border or ACCENT
        t = Table([[Paragraph(text, s['sb'])]], colWidths=[CW - 24])
        t.setStyle(_ts(
            ('BACKGROUND',   (0, 0), (-1, -1), bg),
            ('BOX',          (0, 0), (-1, -1), 0.6, BORDER),
            ('LINEBEFORE',   (0, 0), (0, -1),  4, border),
            ('LEFTPADDING',  (0, 0), (-1, -1), 16),
            ('RIGHTPADDING', (0, 0), (-1, -1), 14),
            ('TOPPADDING',   (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING',(0, 0), (-1, -1), 12),
        ))
        return t

    def _badge(self, text, bg, s):
        t = Table([[Paragraph(f"<font color='white'><b>{text}</b></font>", s['cap'])]])
        t.setStyle(_ts(
            ('BACKGROUND',   (0, 0), (-1, -1), bg),
            ('ROUNDEDCORNERS', [4]),
            ('LEFTPADDING',  (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING',   (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING',(0, 0), (-1, -1), 5),
        ))
        return t

    # ══════════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    def _summary(self, s, ins, di):
        items = [
            _section_header(_icon_dataset, "Executive Summary", s),
            _divider(),
            self._box(ins.get('executive_summary', 'No summary available.'), s),
            Spacer(1, 16),
        ]

        # Stat cards — 4 across
        bd = ins.get('severity_breakdown', {})
        stat_items = [
            (str(bd.get('critical', 0)), "Critical",     CRITICAL),
            (str(bd.get('high', 0)),     "High",          HIGH),
            (str(bd.get('medium', 0)),   "Medium",        MEDIUM),
            (str(ins.get('total_insights', 0)), "Total Findings", ACCENT),
        ]
        cells = []
        card_w = CW / 4 - 6
        for num, label, col in stat_items:
            inner = Table(
                [[Paragraph(f"<font color='{col.hexval()}'><b>{num}</b></font>", s['num'])],
                 [Paragraph(label, s['numlabel'])]],
                colWidths=[card_w],
            )
            inner.setStyle(_ts(
                ('BACKGROUND',   (0, 0), (-1, -1), CARD_BG),
                ('BOX',          (0, 0), (-1, -1), 0.6, BORDER),
                ('LINEBEFORE',   (0, 0), (0, -1),  3, col),
                ('ALIGN',        (0, 0), (-1, -1), 'CENTER'),
                ('TOPPADDING',   (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING',(0, 0), (-1, -1), 10),
                ('LEFTPADDING',  (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ))
            cells.append(inner)
        stat_row = Table([cells], colWidths=[CW / 4] * 4)
        stat_row.setStyle(_ts(
            ('ALIGN',  (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ))
        items += [stat_row, Spacer(1, 12)]

        # Data story
        story_text = ins.get('data_story', '')
        if story_text:
            items += [
                Paragraph("The Full Picture", s['sub']),
                self._box(story_text, s, bg=SOFT_AMBER, border=MEDIUM),
                Spacer(1, 8),
            ]
        return items

    # ══════════════════════════════════════════════════════════════════════
    #  DATASET OVERVIEW
    # ══════════════════════════════════════════════════════════════════════
    def _dataset_overview(self, s, di, st):
        basic  = st.get('basic_info', di) or di
        struct = st.get('data_structure', {}) or {}
        _r = basic.get('rows', '—')
        rows = [
            ["Metric",              "Value"],
            ["Total Rows",          f"{_r:,}" if isinstance(_r, (int, float)) else str(_r)],
            ["Total Columns",       str(basic.get('columns', '—'))],
            ["Missing Values",      f"{basic.get('missing_percentage', 0):.1f}%"],
            ["Duplicate Rows",      str(basic.get('duplicate_rows', 0))],
            ["Memory Usage",        f"{basic.get('memory_mb', 0):.2f} MB"],
            ["Structure Type",      struct.get('type', '—').replace('-', ' ').title()],
            ["Numeric Columns",     str(struct.get('numeric_count', '—'))],
            ["Categorical Columns", str(struct.get('categorical_count', '—'))],
        ]
        hw = CW / 2
        return [
            _section_header(_icon_checklist, "Dataset Overview", s),
            _divider(),
            _data_table(rows, [hw, hw], [
                ('FONTNAME',  (0, 1), (0, -1), 'Helvetica-Bold'),
                ('TEXTCOLOR', (0, 1), (0, -1), ACCENT),
                ('ALIGN',     (1, 0), (1, -1), 'RIGHT'),
            ]),
            Spacer(1, 12),
        ]

    # ══════════════════════════════════════════════════════════════════════
    #  FINDINGS
    # ══════════════════════════════════════════════════════════════════════
    def _findings(self, s, ins):
        all_items = (
            [(i, CRITICAL, "CRITICAL") for i in ins.get('critical_insights', [])] +
            [(i, HIGH,     "HIGH")     for i in ins.get('high_priority_insights', [])] +
            [(i, MEDIUM,   "MEDIUM")   for i in ins.get('medium_priority_insights', [])]
        )
        if not all_items:
            return [
                _section_header(_icon_warning, "Findings", s),
                _divider(SUCCESS),
                self._box(
                    "No significant issues found — your dataset looks clean and ready to use!", s,
                    bg=SOFT_GREEN, border=SUCCESS,
                ),
                Spacer(1, 8),
            ]
        items = [
            _section_header(_icon_warning, "Findings", s),
            _divider(),
        ]
        for ins_item, col, lab in all_items:
            items += [KeepTogether(self._finding_card(ins_item, col, lab, s)), Spacer(1, 14)]
        return items

    def _finding_card(self, ins_data, color, label, s):
        """A single finding rendered as a compact card."""
        cw = CW - 16  # inner content width

        # Headline
        hl = f"<b>{ins_data.get('headline', '')}</b>"
        ci = ins_data.get('column', '')
        if ci:
            if isinstance(ci, list):
                ci = ', '.join(ci)
            hl += f"<br/><font size='8' color='#94A3B8'>Column: {ci}</font>"

        header = Table(
            [[Paragraph(f"<font color='white'><b> {label} </b></font>", s['cap']),
              Paragraph(hl, s['bb'])]],
            colWidths=[0.8 * inch, cw - 0.8 * inch],
        )
        header.setStyle(_ts(
            ('BACKGROUND', (0, 0), (0, 0), color),
            ('BACKGROUND', (1, 0), (1, 0), CARD_BG),
            ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING',   (0, 0), (-1, -1), 10),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 10),
            ('TOPPADDING',    (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('BOX',        (0, 0), (-1, -1), 0.6, color),
        ))
        parts = [header]

        def _detail_row(txt, style, bg, lc=None):
            t = Table([[Paragraph(txt, style)]], colWidths=[cw])
            cmds = [
                ('BACKGROUND',   (0, 0), (-1, -1), bg),
                ('LEFTPADDING',  (0, 0), (-1, -1), 14),
                ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                ('TOPPADDING',   (0, 0), (-1, -1), 7),
                ('BOTTOMPADDING',(0, 0), (-1, -1), 7),
                ('LINEBELOW',    (0, -1), (-1, -1), 0.4, BORDER),
            ]
            if lc:
                cmds.append(('LINEBEFORE', (0, 0), (0, -1), 2.5, lc))
            t.setStyle(_ts(*cmds))
            return t

        w = ins_data.get('what_it_means', '')
        if w:
            parts.append(_detail_row(f"<b>What it means:</b> {w}", s['b'], WHITE, color))

        bi = ins_data.get('business_impact', '')
        if bi:
            parts.append(_detail_row(f"<b>Why it matters:</b> {bi}", s['imp'], SOFT_AMBER,
                                     CRITICAL if label == "CRITICAL" else HIGH))

        act = ins_data.get('what_to_do', '')
        if act:
            parts.append(_detail_row(f"<b>What to do:</b> {act}", s['act'], SOFT_GREEN, SUCCESS))

        dd = ins_data.get('deep_dive', '')
        if dd:
            t = Table([[Paragraph(f"<font color='#6D28D9'>&#9733; Deeper look:</font> {dd}", s['dd'])]],
                      colWidths=[cw])
            t.setStyle(_ts(
                ('BACKGROUND',   (0, 0), (-1, -1), LAVENDER),
                ('LEFTPADDING',  (0, 0), (-1, -1), 14),
                ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                ('TOPPADDING',   (0, 0), (-1, -1), 7),
                ('BOTTOMPADDING',(0, 0), (-1, -1), 7),
                ('LINEBEFORE',   (0, 0), (0, -1), 2.5, colors.HexColor("#7C3AED")),
            ))
            parts.append(t)

        return parts

    # ══════════════════════════════════════════════════════════════════════
    #  MODEL RECOMMENDATION
    # ══════════════════════════════════════════════════════════════════════
    def _model_section(self, s, ins, rec):
        items = [
            _section_header(_icon_model, "Recommended Approach", s),
            _divider(PRIMARY_MID),
        ]
        g = ins.get('model_guidance', {}) or {}
        if not g:
            g = {
                'recommended_model': rec.get('primary_model', '—'),
                'task_type': rec.get('task_type', '—'),
                'why_this_model': rec.get('why_this_model', ''),
                'key_reasons': rec.get('reasoning', []),
                'alternatives': rec.get('alternatives', []),
                'before_you_train': rec.get('preprocessing_steps', []),
                'how_to_validate': rec.get('cv_strategy', ''),
                'how_to_measure_success': rec.get('recommended_metrics', []),
                'confidence_label': '',
                'confidence_score': rec.get('confidence', 0),
            }

        mn   = g.get('recommended_model', '—')
        task = g.get('task_type', '—').title()
        conf = g.get('confidence_label', '')
        why  = g.get('why_this_model', '')
        reasons = g.get('key_reasons', [])
        alts    = g.get('alternatives', [])
        prep    = g.get('before_you_train', [])
        cv      = g.get('how_to_validate_narrative', '') or g.get('how_to_validate', '')
        metrics = g.get('success_metrics_narrative', '') or g.get('how_to_measure_success', [])

        # Model highlight card
        pt = Table(
            [[Paragraph(f"<font color='white'><b>{mn}</b></font>", s['sh']),
              Paragraph(f"<font color='#94A3B8'>Task: {task}</font><br/>"
                        f"<font color='#34D399'>{conf}</font>", s['cm'])]],
            colWidths=[3 * inch, CW - 3 * inch - 16],
        )
        pt.setStyle(_ts(
            ('BACKGROUND',   (0, 0), (-1, -1), PRIMARY),
            ('LEFTPADDING',  (0, 0), (-1, -1), 16),
            ('RIGHTPADDING', (0, 0), (-1, -1), 14),
            ('TOPPADDING',   (0, 0), (-1, -1), 14),
            ('BOTTOMPADDING',(0, 0), (-1, -1), 14),
            ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
            ('LINEBEFORE',   (0, 0), (0, -1),  4, ACCENT),
        ))
        items += [pt, Spacer(1, 10)]

        if why:
            items.append(Paragraph(f"<b>Why this model:</b> {why}", s['b']))
        for r in reasons:
            items.append(Paragraph(f"&#8226; {r}", s['b']))
        items.append(Spacer(1, 8))

        if alts:
            items.append(Paragraph("Other options worth considering", s['sub']))
            for a in alts:
                items.append(Paragraph(f"<b>{a.get('model', '')}</b> — {a.get('why', '')}", s['b']))
            items.append(Spacer(1, 8))

        if prep:
            items.append(Paragraph("Before you start — data preparation steps", s['sub']))
            narrative = g.get('before_you_train_narrative', '')
            if narrative:
                items.append(Paragraph(narrative, s['b']))
            else:
                for i, step in enumerate(prep, 1):
                    items.append(Paragraph(f"{i}. {step}", s['b']))
            items.append(Spacer(1, 8))

        if cv:
            items += [Paragraph("How to check if it's working", s['sub']),
                      Paragraph(cv, s['b']), Spacer(1, 8)]

        if metrics:
            items.append(Paragraph("What a good result looks like", s['sub']))
            if isinstance(metrics, str):
                items.append(Paragraph(metrics, s['b']))
            else:
                for m in metrics:
                    items.append(Paragraph(f"&#8226; {m}", s['b']))

        items.append(Spacer(1, 10))
        return items

    # ══════════════════════════════════════════════════════════════════════
    #  QUICK WINS
    # ══════════════════════════════════════════════════════════════════════
    def _quick_wins(self, s, ins):
        wins = ins.get('quick_wins', [])
        if not wins:
            return []
        items = [
            _section_header(_icon_lightning, "Quick Wins — Start Here", s),
            _divider(MEDIUM),
            Paragraph("These are the easiest, highest-impact steps to take right now.", s['b']),
            Spacer(1, 10),
        ]
        rows = [["#", "Action"]] + [[str(i), w] for i, w in enumerate(wins, 1)]
        t = _data_table(rows, [0.35 * inch, CW - 0.35 * inch], [
            ('ALIGN',    (0, 0), (0, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('TEXTCOLOR',(0, 1), (0, -1), ACCENT),
            ('VALIGN',   (0, 0), (-1, -1), 'TOP'),
        ])
        return items + [t, Spacer(1, 12)]

    # ══════════════════════════════════════════════════════════════════════
    #  COLUMN RELATIONSHIPS
    # ══════════════════════════════════════════════════════════════════════
    def _column_relationships(self, s, ins):
        notable = [r for r in ins.get('column_relationships', [])
                   if r.get('severity') in ('critical', 'high', 'medium')]
        if not notable:
            return []
        items = [
            PageBreak(),
            _section_header(_icon_link, "Column Relationships", s),
            _divider(),
            Paragraph("These columns move together in ways that could affect your analysis.", s['b']),
            Spacer(1, 10),
        ]
        for r in notable:
            sev = r.get('severity', 'medium')
            col = CRITICAL if sev == 'critical' else HIGH if sev == 'high' else MEDIUM
            corr = r.get('correlation')
            cs   = f"{corr:+.2f}" if corr is not None else "reversal"
            pair = f"{r.get('col_a', '')}  \u2194  {r.get('col_b', '')}"
            cl   = (f"{cs}  {r.get('direction', '')}" if corr is not None
                    else f"Reverses by: {r.get('split_by', '')}")

            ht = Table(
                [[Paragraph(f"<b>{pair}</b>", s['bb']),
                  Paragraph(f"<font color='#94A3B8'>{cl}</font>", s['cap'])]],
                colWidths=[CW * 0.65, CW * 0.35],
            )
            ht.setStyle(_ts(
                ('BACKGROUND',   (0, 0), (-1, -1), CARD_BG),
                ('LEFTPADDING',  (0, 0), (-1, -1), 12),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING',   (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING',(0, 0), (-1, -1), 8),
                ('LINEBEFORE',   (0, 0), (0, -1), 3, col),
                ('ALIGN',        (1, 0), (1, -1), 'RIGHT'),
                ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
            ))
            card = [ht]
            for txt, style, bg, lc in [
                (r.get('explanation', ''), s['b'], WHITE, None),
                (f"<b>What to do:</b> {r.get('action', '')}", s['act'], SOFT_GREEN, col),
            ]:
                if not txt:
                    continue
                rt = Table([[Paragraph(txt, style)]], colWidths=[CW])
                cmds = [
                    ('BACKGROUND',   (0, 0), (-1, -1), bg),
                    ('LEFTPADDING',  (0, 0), (-1, -1), 14),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                    ('TOPPADDING',   (0, 0), (-1, -1), 7),
                    ('BOTTOMPADDING',(0, 0), (-1, -1), 7),
                    ('LINEBELOW',    (0, -1), (-1, -1), 0.4, BORDER),
                ]
                if lc:
                    cmds.append(('LINEBEFORE', (0, 0), (0, -1), 1.5, lc))
                rt.setStyle(_ts(*cmds))
                card.append(rt)
            items += [KeepTogether(card), Spacer(1, 10)]
        return items

    # ══════════════════════════════════════════════════════════════════════
    #  IMBALANCE GUIDANCE
    # ══════════════════════════════════════════════════════════════════════
    def _imbalance_guidance(self, s, ins):
        g = ins.get('class_imbalance_guidance')
        if not g:
            return []
        items = [
            _section_header(_icon_balance, "Class Imbalance — Important!", s),
            _divider(CRITICAL),
        ]
        tgt, maj = g.get('target_column', '—'), g.get('majority_pct', 0)
        items += [
            Paragraph(f'The column <b>"{tgt}"</b> has {maj:.0f}% of its values in a single group '
                      f'— this is called class imbalance.', s['bb']),
            Spacer(1, 8),
        ]
        why = g.get('why_it_matters', '')
        if why:
            items += [self._box(why, s, bg=SOFT_RED, border=CRITICAL), Spacer(1, 12)]
        tg = g.get('technique_guidance', '')
        if tg:
            items += [Paragraph("Which fix to try first", s['sub']),
                      Paragraph(tg, s['b']), Spacer(1, 6)]
        fs = g.get('first_step', '')
        if fs:
            items += [self._box(f"<b>Start here:</b> {fs}", s, bg=SOFT_GREEN, border=SUCCESS),
                      Spacer(1, 10)]
        techs = g.get('techniques', [])
        if techs:
            items.append(Paragraph("Available techniques", s['sub']))
            rows = [["Technique", "Difficulty", "What it does"]] + [
                [Paragraph(f"<b>{t.get('name', '')}</b>", s['b']),
                 t.get('difficulty', '').title(),
                 Paragraph(t.get('description', ''), s['b'])]
                for t in techs
            ]
            items += [
                _data_table(rows, [CW * 0.27, CW * 0.13, CW * 0.60],
                            [('FONTSIZE', (0, 0), (-1, -1), 9),
                             ('VALIGN', (0, 0), (-1, -1), 'TOP')]),
                Spacer(1, 10),
            ]

        items.append(Paragraph("How to measure success (not the usual way!)", s['sub']))
        mr = g.get('metric_reasoning', '') or g.get('metric_explanation', '')
        if mr:
            items.append(Paragraph(mr, s['b']))
        wm = g.get('wrong_metrics', [])
        if wm:
            items.append(Paragraph(f"<font color='#DC2626'>&#10007; Avoid using:</font> {', '.join(wm)}", s['b']))
        rm = g.get('right_metrics', [])
        if rm:
            items.append(Paragraph(f"<font color='#16A34A'>&#10003; Use these instead:</font> {', '.join(rm)}", s['b']))
        items.append(Spacer(1, 10))
        return items

    # ══════════════════════════════════════════════════════════════════════
    #  COLUMN PROFILES
    # ══════════════════════════════════════════════════════════════════════
    def _column_profiles(self, s, st):
        profs = st.get('column_profiles', [])
        if not profs:
            return []
        items = [
            PageBreak(),
            _section_header(_icon_checklist, "Column Profiles", s),
            _divider(),
            Paragraph("A snapshot of every column — useful for spotting patterns at a glance.", s['b']),
            Spacer(1, 10),
        ]
        rows = [["Column", "Type", "Missing", "Unique", "Notes"]]
        for p in profs:
            notes = []
            if p.get('missing_pct', 0) >= 50:
                notes.append("high missingness")
            if p.get('skewness') and abs(p['skewness']) > 2:
                notes.append(f"skewed ({p['skewness']:.1f})")
            if p.get('disguised_missing', 0) > 0:
                notes.append(f"{p['disguised_missing']} disguised nulls")
            if p.get('whitespace_issues', 0) > 0:
                notes.append("whitespace issues")
            if p.get('inf_count', 0) > 0:
                notes.append(f"{p['inf_count']} infinities")
            rows.append([
                p.get('name', ''),
                p.get('kind', '').replace('_', ' '),
                f"{p.get('missing_pct', 0):.1f}%",
                str(p.get('unique_count', '')),
                ', '.join(notes) or '—',
            ])
        items.append(_data_table(rows, [CW * 0.25, CW * 0.18, CW * 0.12, CW * 0.10, CW * 0.35], [
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('VALIGN',   (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ]))
        items.append(Spacer(1, 10))
        return items

    # ══════════════════════════════════════════════════════════════════════
    #  VISUAL ANALYSIS  (charts from ChartEngine)
    # ══════════════════════════════════════════════════════════════════════
    def _visual_analysis(self, s, raw_results: dict) -> list:
        """Embed charts — each chart is kept together with its title on the
        same page.  Tall charts get their own dedicated page."""
        if not raw_results:
            return []

        try:
            from core.chart_engine import ChartEngine
        except ImportError:
            try:
                import sys, os
                sys.path.insert(0, os.path.dirname(__file__))
                from chart_engine import ChartEngine
            except ImportError:
                return []

        try:
            charts = ChartEngine().generate_all(raw_results)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"ChartEngine failed: {e}")
            return []

        available = {k: v for k, v in charts.items() if v is not None}
        if not available:
            return []

        # ── Chart metadata ─────────────────────────────────────────────
        _titles = {
            'missing_values':      'Missing Values',
            'correlation':         'Correlations',
            'target_distribution': 'Target Distribution',
            'outliers':            'Outlier Severity',
            'skewness':            'Distribution Skewness',
            'data_health_radar':   'Data Health Overview',
            'column_types':        'Column Types',
            'duplicates':          'Duplicate Rows',
            'cardinality':         'Cardinality',
            'feature_importance':  'Feature Importance',
            'missing_pattern':     'Missing Data Pattern',
            'box_plots':           'Box Plots',
            'pca_variance':        'PCA Variance',
            'cluster_preview':     'Cluster Preview',
        }
        _captions = {
            'missing_values':      'Columns sorted by missing % \u2014 red \u226550 %, orange \u226520 %, yellow \u22655 %.',
            'correlation':         'Strength of linear relationships between numeric columns.',
            'target_distribution': 'How the target variable is distributed across categories.',
            'outliers':            'Share of rows with extreme values per column (IQR method).',
            'skewness':            'How far each column\'s distribution deviates from symmetry.',
            'data_health_radar':   'Six-dimension data quality score \u2014 higher is healthier.',
            'column_types':        'Breakdown of column data types in the dataset.',
            'duplicates':          'Proportion of unique vs duplicate rows.',
            'cardinality':         'Number of unique values per column \u2014 high values may indicate IDs.',
            'feature_importance':  'Estimated feature importance based on correlation with the target.',
            'missing_pattern':     'Overview of which columns have missing data and how much.',
            'box_plots':           'Spread and outliers for the top numeric columns.',
            'pca_variance':        'How much variance each principal component explains.',
            'cluster_preview':     'Data projected onto first two principal components, coloured by cluster.',
        }

        def _png_to_image(png_bytes: bytes, max_width: float) -> Image:
            from PIL import Image as PilImage
            pil = PilImage.open(_io.BytesIO(png_bytes))
            w_px, h_px = pil.size
            pil.close()
            aspect = h_px / w_px
            w = min(max_width, w_px / 130 * 72)
            h = w * aspect
            # Leave room for title + caption + section header (~90 pt)
            max_h = PAGE_H - 2 * MARGIN - 90
            if h > max_h:
                h = max_h
                w = h / aspect
            return Image(_io.BytesIO(png_bytes), width=w, height=h)

        def _chart_block(key: str, png_bytes: bytes) -> list:
            """Build a self-contained block: title + chart + caption.
            Wrapped in KeepTogether so title and plot are never split."""
            title = _titles.get(key, key.replace('_', ' ').title())
            caption = _captions.get(key, '')
            img = _png_to_image(png_bytes, CW)

            header = Paragraph(f"<b>{title}</b>", s['sub'])
            block_items = [header, Spacer(1, 4), img]
            if caption:
                block_items.append(Spacer(1, 4))
                block_items.append(Paragraph(caption, s['cap']))
            block_items.append(Spacer(1, 18))
            return [KeepTogether(block_items)]

        # ── Section intro ──────────────────────────────────────────────
        items = [
            PageBreak(),
            _section_header(_icon_chart, "Visual Analysis", s),
            _divider(PRIMARY_MID),
            Paragraph(
                "Charts generated from the pre-computed analysis. Each chart adapts to "
                "your dataset size and is skipped automatically when the underlying data "
                "is absent or trivial.", s['b'],
            ),
            Spacer(1, 14),
        ]

        # Preferred display order
        order = [
            'data_health_radar', 'missing_values', 'target_distribution',
            'correlation', 'outliers', 'skewness', 'column_types',
            'duplicates', 'cardinality', 'feature_importance',
            'missing_pattern', 'box_plots', 'pca_variance', 'cluster_preview',
        ]

        # Radar + first companion chart side-by-side if both exist
        companion_keys = [k for k in order if k != 'data_health_radar' and k in available]
        radar_bytes    = available.get('data_health_radar')
        rendered: set  = set()

        if radar_bytes and companion_keys:
            companion_key   = companion_keys[0]
            companion_bytes = available[companion_key]
            try:
                radar_img     = _png_to_image(radar_bytes, CW * 0.38)
                companion_img = _png_to_image(companion_bytes, CW * 0.56)
                side = Table(
                    [[companion_img, radar_img]],
                    colWidths=[CW * 0.58, CW * 0.42],
                )
                side.setStyle(TableStyle([
                    ('VALIGN',       (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING',  (0, 0), (-1, -1), 0),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                    ('TOPPADDING',   (0, 0), (-1, -1), 0),
                    ('BOTTOMPADDING',(0, 0), (-1, -1), 0),
                ]))
                cap = Table(
                    [[Paragraph(_captions.get(companion_key, ''), s['cap']),
                      Paragraph(_captions.get('data_health_radar', ''), s['cap'])]],
                    colWidths=[CW * 0.58, CW * 0.42],
                )
                cap.setStyle(TableStyle([
                    ('VALIGN',       (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING',  (0, 0), (-1, -1), 0),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                    ('TOPPADDING',   (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING',(0, 0), (-1, -1), 0),
                ]))
                items += [KeepTogether([side, cap, Spacer(1, 18)])]
                rendered = {companion_key, 'data_health_radar'}
            except Exception:
                pass

        # ── Remaining charts — each in its own KeepTogether block ──────
        for key in order:
            if key in rendered or key not in available:
                continue
            try:
                items += _chart_block(key, available[key])
            except Exception:
                continue

        return items

    # ══════════════════════════════════════════════════════════════════════
    #  CLOSING PAGE
    # ══════════════════════════════════════════════════════════════════════
    def _closing(self, s, at):
        # Checkmark illustration
        d = Drawing(CW, 60)
        cx, cy = CW / 2, 30
        d.add(Circle(cx, cy, 22, fillColor=SOFT_GREEN, strokeColor=SUCCESS, strokeWidth=2))
        d.add(Line(cx - 10, cy, cx - 3, cy - 8, strokeColor=SUCCESS, strokeWidth=2.5))
        d.add(Line(cx - 3, cy - 8, cx + 11, cy + 10, strokeColor=SUCCESS, strokeWidth=2.5))
        check_art = _IconFlowable(d, CW, 60)

        return [
            Spacer(1, 1.5 * inch),
            check_art,
            Spacer(1, 18),
            Paragraph("You're all set!", s['sh']),
            _divider(SUCCESS),
            Paragraph(
                "This report was generated automatically by DataSense. All findings are "
                "based on a statistical analysis of your uploaded dataset. It is always a "
                "good idea to double-check important findings with someone familiar with "
                "your data before making major decisions.", s['b'],
            ),
            Spacer(1, 14),
            Paragraph(f"Generated on {at}", s['cap']),
        ]
