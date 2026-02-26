"""PDFReportGenerator — reader-friendly redesign with Colombo time & illustrations."""
import io
import math
from datetime import datetime, timezone, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                HRFlowable, PageBreak, KeepTogether, Flowable)
from reportlab.graphics.shapes import (Drawing, Circle, Rect, Line, Polygon,
                                       String, Path, Ellipse)
from reportlab.graphics import renderPDF

# ── Colombo Time (UTC+5:30) ───────────────────────────────────────────────────
COLOMBO_TZ = timezone(timedelta(hours=5, minutes=30))

def _now_colombo():
    return datetime.now(COLOMBO_TZ).strftime("%B %d, %Y at %H:%M IST (Colombo)")

# ── Colour Palette — warm, approachable, reader-friendly ─────────────────────
BRAND      = colors.HexColor("#4F46E5")   # Indigo — main brand
ACCENT     = colors.HexColor("#F59E0B")   # Amber — warm highlight
TEAL       = colors.HexColor("#0D9488")   # Teal — secondary accent
COVER_BG   = colors.HexColor("#1E1B4B")   # Deep indigo for cover
COVER_LINE = colors.HexColor("#818CF8")   # Soft lavender accent line

CRITICAL   = colors.HexColor("#EF4444")
HIGH       = colors.HexColor("#F97316")
MEDIUM     = colors.HexColor("#EAB308")
SUCCESS    = colors.HexColor("#10B981")

CARD_BG    = colors.HexColor("#F8FAFC")   # Near-white card background
SOFT_BLUE  = colors.HexColor("#EEF2FF")   # Light indigo tint
SOFT_AMBER = colors.HexColor("#FFFBEB")   # Warm amber tint
SOFT_TEAL  = colors.HexColor("#F0FDFA")   # Mint tint
SOFT_RED   = colors.HexColor("#FEF2F2")   # Light red tint
DEEPDIVE   = colors.HexColor("#F5F3FF")   # Lavender tint

DARK       = colors.HexColor("#111827")   # Near-black text
MUTED      = colors.HexColor("#6B7280")   # Grey muted text
EDGE       = colors.HexColor("#E5E7EB")   # Border colour
WHITE      = colors.white

PAGE_W, PAGE_H = letter
MARGIN = 0.75 * inch
CW = PAGE_W - 2 * MARGIN


# ── Simple Illustrations ──────────────────────────────────────────────────────
class _IconFlowable(Flowable):
    """Small inline drawing rendered as a Flowable."""
    def __init__(self, drawing, width=40, height=40):
        Flowable.__init__(self)
        self._drawing = drawing
        self.width = width
        self.height = height

    def draw(self):
        renderPDF.draw(self._drawing, self.canv, 0, 0)


def _icon_dataset(size=48):
    """Table/grid icon representing a dataset."""
    d = Drawing(size, size)
    s = size
    d.add(Rect(2, 2, s-4, s-4, rx=6, ry=6, fillColor=SOFT_BLUE, strokeColor=BRAND, strokeWidth=1.5))
    d.add(Rect(8, s-18, s-16, 10, fillColor=BRAND, strokeColor=None))
    for y in [s-30, s-41, s-52]:
        if y >= 6:
            d.add(Rect(8, y, s-16, 8, fillColor=EDGE, strokeColor=None))
    d.add(Line(s//2, s-18, s//2, 8, strokeColor=MUTED, strokeWidth=0.8))
    return _IconFlowable(d, size, size)


def _icon_warning(size=48, color=None):
    """Triangle warning icon."""
    color = color or HIGH
    d = Drawing(size, size)
    cx = size / 2
    pts = [cx, size-4, 4, 4, size-4, 4]
    d.add(Polygon(pts, fillColor=SOFT_AMBER, strokeColor=color, strokeWidth=1.5))
    d.add(String(cx-3.5, 10, '!', fontSize=16, fillColor=color, fontName='Helvetica-Bold'))
    return _IconFlowable(d, size, size)


def _icon_critical(size=48):
    """Red circle with X for critical issues."""
    d = Drawing(size, size)
    c = size / 2
    d.add(Circle(c, c, c-2, fillColor=SOFT_RED, strokeColor=CRITICAL, strokeWidth=1.5))
    m = 14
    d.add(Line(m, size-m, size-m, m, strokeColor=CRITICAL, strokeWidth=2.5))
    d.add(Line(m, m, size-m, size-m, strokeColor=CRITICAL, strokeWidth=2.5))
    return _IconFlowable(d, size, size)


def _icon_model(size=48):
    """Neural-net style icon for model recommendations."""
    d = Drawing(size, size)
    d.add(Rect(2, 2, size-4, size-4, rx=6, ry=6, fillColor=SOFT_BLUE, strokeColor=BRAND, strokeWidth=1.2))
    nodes = [(12, 36), (12, 24), (12, 12), (36, 30), (36, 18)]
    for ax, ay in [(12,36),(12,24),(12,12)]:
        for bx, by in [(36,30),(36,18)]:
            d.add(Line(ax+4, ay, bx-4, by, strokeColor=COVER_LINE, strokeWidth=0.8))
    for (x, y) in nodes:
        d.add(Circle(x, y, 4, fillColor=BRAND, strokeColor=WHITE, strokeWidth=1))
    return _IconFlowable(d, size, size)


def _icon_lightning(size=48):
    """Lightning bolt icon for quick wins."""
    d = Drawing(size, size)
    d.add(Rect(2, 2, size-4, size-4, rx=6, ry=6, fillColor=SOFT_AMBER, strokeColor=ACCENT, strokeWidth=1.2))
    pts = [28, 42, 20, 26, 26, 26, 18, 6, 28, 22, 22, 22, 28, 42]
    d.add(Polygon(pts, fillColor=ACCENT, strokeColor=None))
    return _IconFlowable(d, size, size)


def _icon_link(size=48):
    """Two connected circles for relationships."""
    d = Drawing(size, size)
    d.add(Rect(2, 2, size-4, size-4, rx=6, ry=6, fillColor=SOFT_TEAL, strokeColor=TEAL, strokeWidth=1.2))
    mid = size // 2
    d.add(Line(22, mid, 26, mid, strokeColor=MUTED, strokeWidth=2))
    d.add(Circle(14, mid, 8, fillColor=TEAL, strokeColor=WHITE, strokeWidth=1.5))
    d.add(Circle(34, mid, 8, fillColor=BRAND, strokeColor=WHITE, strokeWidth=1.5))
    return _IconFlowable(d, size, size)


def _icon_balance(size=48):
    """Tilted balance scale icon for imbalance section."""
    d = Drawing(size, size)
    d.add(Rect(2, 2, size-4, size-4, rx=6, ry=6, fillColor=SOFT_RED, strokeColor=CRITICAL, strokeWidth=1.2))
    cx = size / 2
    d.add(Line(cx, 36, cx, 20, strokeColor=DARK, strokeWidth=2))
    d.add(Line(8, 24, size-8, 16, strokeColor=DARK, strokeWidth=1.8))
    d.add(Rect(4, 18, 12, 8, rx=2, ry=2, fillColor=HIGH, strokeColor=None))
    d.add(Rect(size-16, 8, 12, 8, rx=2, ry=2, fillColor=CRITICAL, strokeColor=None))
    return _IconFlowable(d, size, size)


def _icon_checklist(size=48):
    """Checklist icon for column profiles."""
    d = Drawing(size, size)
    d.add(Rect(2, 2, size-4, size-4, rx=6, ry=6, fillColor=SOFT_TEAL, strokeColor=TEAL, strokeWidth=1.2))
    for i, y in enumerate([34, 24, 14]):
        d.add(Rect(8, y-3, 10, 10, rx=2, ry=2,
                   fillColor=TEAL if i < 2 else EDGE,
                   strokeColor=TEAL, strokeWidth=1))
        if i < 2:
            d.add(String(9.5, y, 'v', fontSize=7, fillColor=WHITE, fontName='Helvetica-Bold'))
        d.add(Line(24, y+3, size-8, y+3, strokeColor=MUTED, strokeWidth=1.5))
    return _IconFlowable(d, size, size)


def _cover_decoration(canvas, page_w, page_h):
    """Draw subtle geometric decoration on the cover page."""
    canvas.saveState()
    canvas.setFillColor(colors.HexColor("#312E81"))
    for r, alpha in [(120, 0.15), (80, 0.20), (45, 0.25)]:
        canvas.setFillAlpha(alpha)
        canvas.circle(page_w - 10, page_h - 10, r, fill=1, stroke=0)
    canvas.setFillAlpha(0.12)
    canvas.circle(30, 30, 90, fill=1, stroke=0)
    canvas.restoreState()


# ── Styles ────────────────────────────────────────────────────────────────────
_S = None

def _s():
    global _S
    if _S:
        return _S
    def ps(n, **kw):
        return ParagraphStyle(n, **kw)
    _S = {
        'ct':       ps('ct',       fontName='Helvetica-Bold', fontSize=34, textColor=WHITE,  spaceAfter=6,  leading=40),
        'cs':       ps('cs',       fontName='Helvetica',      fontSize=14, textColor=colors.HexColor("#A5B4FC"), spaceAfter=4),
        'cm':       ps('cm',       fontName='Helvetica',      fontSize=10, textColor=colors.HexColor("#94A3B8")),
        'sh':       ps('sh',       fontName='Helvetica-Bold', fontSize=15, textColor=DARK,   spaceBefore=20, spaceAfter=6),
        'sub':      ps('sub',      fontName='Helvetica-Bold', fontSize=11, textColor=colors.HexColor("#374151"), spaceBefore=10, spaceAfter=4),
        'b':        ps('b',        fontName='Helvetica',      fontSize=10, textColor=colors.HexColor("#374151"), leading=16, spaceAfter=4),
        'bb':       ps('bb',       fontName='Helvetica-Bold', fontSize=10, textColor=DARK,   leading=15),
        'cap':      ps('cap',      fontName='Helvetica',      fontSize=8,  textColor=MUTED,  leading=11),
        'sb':       ps('sb',       fontName='Helvetica',      fontSize=11, textColor=DARK,   leading=18),
        'act':      ps('act',      fontName='Helvetica',      fontSize=10, textColor=colors.HexColor("#065F46"), leading=15, leftIndent=12),
        'imp':      ps('imp',      fontName='Helvetica',      fontSize=10, textColor=colors.HexColor("#92400E"), leading=15),
        'dd':       ps('dd',       fontName='Helvetica',      fontSize=9,  textColor=colors.HexColor("#4C1D95"), leading=14, leftIndent=8),
        'num':      ps('num',      fontName='Helvetica-Bold', fontSize=20, textColor=BRAND,  leading=24),
        'numlabel': ps('numlabel', fontName='Helvetica',      fontSize=8,  textColor=MUTED,  leading=10),
    }
    return _S


# ── Table helpers ─────────────────────────────────────────────────────────────
def _ts(*cmds):
    return TableStyle(list(cmds))

_GRID_BASE = [
    ('FONTSIZE',      (0,0), (-1,-1), 10),
    ('GRID',          (0,0), (-1,-1), 0.5, EDGE),
    ('LEFTPADDING',   (0,0), (-1,-1), 10),
    ('RIGHTPADDING',  (0,0), (-1,-1), 10),
    ('TOPPADDING',    (0,0), (-1,-1), 9),
    ('BOTTOMPADDING', (0,0), (-1,-1), 9),
]
_HEADER_ROW = [
    ('BACKGROUND', (0,0), (-1,0), BRAND),
    ('TEXTCOLOR',  (0,0), (-1,0), WHITE),
    ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
]

def _data_table(rows, widths, extra=None):
    t = Table(rows, colWidths=widths)
    cmds = _HEADER_ROW + _GRID_BASE + [('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, SOFT_BLUE])]
    if extra:
        cmds += extra
    t.setStyle(_ts(*cmds))
    return t

def _section_header(icon_fn, title, s):
    """Helper: create an icon + title row."""
    icon = icon_fn()
    row = Table([[icon, Paragraph(title, s['sh'])]], colWidths=[56, CW - 56])
    row.setStyle(_ts(('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('LEFTPADDING', (0,0), (0,-1), 0)))
    return row


# ── Page callbacks ────────────────────────────────────────────────────────────
def _on_first(c, d):
    c.saveState()
    c.setFillColor(COVER_BG)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    _cover_decoration(c, PAGE_W, PAGE_H)
    c.setFillColor(BRAND)
    c.rect(0, PAGE_H - 8, PAGE_W, 8, fill=1, stroke=0)
    c.setFillColor(ACCENT)
    c.rect(0, PAGE_H - 12, PAGE_W * 0.4, 4, fill=1, stroke=0)
    c.restoreState()


def _on_later(c, d):
    c.saveState()
    c.setFillColor(BRAND)
    c.rect(0, PAGE_H - 4, PAGE_W, 4, fill=1, stroke=0)
    c.setFillColor(SOFT_BLUE)
    c.rect(0, PAGE_H - 22, PAGE_W, 18, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(BRAND)
    c.drawString(MARGIN, PAGE_H - 16, "DataSense Analysis Report")
    c.setFillColor(MUTED)
    c.drawRightString(PAGE_W - MARGIN, PAGE_H - 16, d.filename_label)
    c.setStrokeColor(EDGE)
    c.setLineWidth(0.5)
    c.line(MARGIN, 36, PAGE_W - MARGIN, 36)
    c.setFont("Helvetica", 7)
    c.setFillColor(MUTED)
    c.drawString(MARGIN, 24, f"Generated {d.generated_at}")
    c.drawRightString(PAGE_W - MARGIN, 24, f"Page {d.page}")
    c.restoreState()


# ── Main Generator ────────────────────────────────────────────────────────────
class PDFReportGenerator:

    def _parse(self, results):
        res = results.get('results', {})
        return {
            'job':   {'filename': results.get('filename', 'Unknown'),
                      'completed_at': results.get('completed_at', ''),
                      'processing_time': results.get('processing_time_seconds', 0)},
            'di':    res.get('dataset_info', {}),
            'st':    res.get('structural_analysis', {}),
            'stats': res.get('statistical_analysis', {}),
            'rec':   res.get('model_recommendations', {}),
            'ins':   res.get('insights', {}),
            'at':    _now_colombo(),
        }

    def _build_story(self, d):
        s = _s()
        ins, st, rec = d['ins'], d['st'], d['rec']
        story = self._cover(s, d['job'], d['di'], d['at'], ins) + [PageBreak()]
        story += self._summary(s, ins, d['di'])
        story += self._dataset_overview(s, d['di'], st)
        story += self._findings(s, ins)
        story += self._model_section(s, ins, rec)
        story += self._quick_wins(s, ins)
        story += self._column_relationships(s, ins)
        story += self._imbalance_guidance(s, ins)
        story += self._column_profiles(s, st)
        story += [PageBreak()] + self._closing(s, d['at'])
        return story

    def _make_doc(self, target, label, at):
        doc = SimpleDocTemplate(
            target, pagesize=letter,
            leftMargin=MARGIN, rightMargin=MARGIN,
            topMargin=MARGIN + 0.35 * inch,
            bottomMargin=MARGIN,
        )
        doc.filename_label = label
        doc.generated_at   = at
        return doc

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

    # ── Cover ──────────────────────────────────────────────────────────────
    def _cover(self, s, job, di, at, ins=None):
        it = [
            Spacer(1, 1.6 * inch),
            Paragraph("DataSense", s['ct']),
            Paragraph("Data Analysis Report", s['cs']),
            Spacer(1, 0.25 * inch),
            HRFlowable(width="60%", thickness=1, color=COVER_LINE, spaceAfter=18),
            Paragraph(f"File: {job['filename']}", s['cm']),
            Spacer(1, 4),
        ]
        r, c = di.get('rows', '—'), di.get('columns', '—')
        r_fmt = f"{r:,}" if isinstance(r, (int, float)) else str(r)
        it += [
            Paragraph(f"Dataset: {r_fmt} rows x {c} columns", s['cm']),
            Spacer(1, 4),
            Paragraph(f"Generated: {at}", s['cm']),
        ]
        if job.get('processing_time'):
            it += [Spacer(1, 4), Paragraph(f"Analysis completed in {round(job['processing_time'], 1)}s", s['cm'])]
        if ins and ins.get('llm_enhanced'):
            it += [Spacer(1, 4), Paragraph(f"AI-enhanced insights via {ins.get('llm_provider', 'LLM')}", s['cm'])]
        return it

    # ── Box / callout helper ───────────────────────────────────────────────
    def _box(self, text, s, bg=None, border=None):
        bg = bg or SOFT_BLUE
        border = border or BRAND
        t = Table([[Paragraph(text, s['sb'])]], colWidths=[CW - 24])
        t.setStyle(_ts(
            ('BACKGROUND',    (0,0), (-1,-1), bg),
            ('BOX',           (0,0), (-1,-1), 1.5, border),
            ('LINEBEFORE',    (0,0), (0,-1),  4,   border),
            ('LEFTPADDING',   (0,0), (-1,-1), 16),
            ('RIGHTPADDING',  (0,0), (-1,-1), 14),
            ('TOPPADDING',    (0,0), (-1,-1), 12),
            ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ))
        return t

    def _badge(self, text, bg, s):
        t = Table([[Paragraph(f"<font color='white'><b>{text}</b></font>", s['cap'])]])
        t.setStyle(_ts(
            ('BACKGROUND',    (0,0), (-1,-1), bg),
            ('ROUNDEDCORNERS',[5]),
            ('LEFTPADDING',   (0,0), (-1,-1), 10),
            ('RIGHTPADDING',  (0,0), (-1,-1), 10),
            ('TOPPADDING',    (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ))
        return t

    # ── Summary ────────────────────────────────────────────────────────────
    def _summary(self, s, ins, di):
        it = [_section_header(_icon_dataset, "Summary", s),
              HRFlowable(width="100%", thickness=1.5, color=BRAND, spaceAfter=12),
              self._box(ins.get('executive_summary', 'No summary available.'), s),
              Spacer(1, 14)]

        # Stat cards
        bd = ins.get('severity_breakdown', {})
        stat_items = [
            (str(bd.get('critical', 0)), "Critical Issues",  CRITICAL),
            (str(bd.get('high', 0)),     "High Priority",    HIGH),
            (str(bd.get('medium', 0)),   "Medium Priority",  MEDIUM),
            (str(ins.get('total_insights', 0)), "Total Findings", BRAND),
        ]
        cells = []
        for num, label, col in stat_items:
            inner = Table(
                [[Paragraph(f"<font color='{col.hexval()}'><b>{num}</b></font>", s['num'])],
                 [Paragraph(label, s['numlabel'])]],
                colWidths=[CW / 4 - 14],
            )
            inner.setStyle(_ts(
                ('BACKGROUND',    (0,0), (-1,-1), CARD_BG),
                ('BOX',           (0,0), (-1,-1), 1,   EDGE),
                ('LINEBEFORE',    (0,0), (0,-1),  3,   col),
                ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
                ('LEFTPADDING',   (0,0), (-1,-1), 8),
                ('RIGHTPADDING',  (0,0), (-1,-1), 8),
                ('TOPPADDING',    (0,0), (-1,-1), 10),
                ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ))
            cells.append(inner)
        stat_table = Table([cells], colWidths=[CW / 4] * 4)
        stat_table.setStyle(_ts(('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'TOP')))
        it += [stat_table, Spacer(1, 8)]

        # Data story callout
        story = ins.get('data_story', '')
        if story:
            it += [Spacer(1, 6),
                   Paragraph("The Full Picture", s['sub']),
                   self._box(story, s, bg=SOFT_AMBER, border=ACCENT),
                   Spacer(1, 6)]
        return it

    # ── Dataset Overview ───────────────────────────────────────────────────
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
            HRFlowable(width="100%", thickness=1.5, color=BRAND, spaceAfter=10),
            _data_table(rows, [hw, hw], [
                ('FONTNAME',  (0,1), (0,-1), 'Helvetica-Bold'),
                ('TEXTCOLOR', (0,1), (0,-1), BRAND),
                ('ALIGN',     (1,0), (1,-1), 'RIGHT'),
            ]),
            Spacer(1, 8),
        ]

    # ── Findings ───────────────────────────────────────────────────────────
    def _findings(self, s, ins):
        all_i = (
            [(i, CRITICAL, "CRITICAL") for i in ins.get('critical_insights', [])] +
            [(i, HIGH,     "HIGH")     for i in ins.get('high_priority_insights', [])] +
            [(i, MEDIUM,   "MEDIUM")   for i in ins.get('medium_priority_insights', [])]
        )
        if not all_i:
            return [_section_header(_icon_warning, "Findings", s),
                    HRFlowable(width="100%", thickness=1.5, color=SUCCESS, spaceAfter=10),
                    self._box("No significant issues found — your dataset looks clean and ready to use!", s,
                              bg=SOFT_TEAL, border=SUCCESS)]
        it = [_section_header(_icon_warning, "Findings", s),
              HRFlowable(width="100%", thickness=1.5, color=BRAND, spaceAfter=10)]
        for ins_item, col, lab in all_i:
            it += [KeepTogether(self._card(ins_item, col, lab, s)), Spacer(1, 12)]
        return it

    def _card(self, ins, color, label, s):
        cw = CW - 24
        hl = f"<b>{ins.get('headline', '')}</b>"
        ci = ins.get('column', '')
        if ci:
            if isinstance(ci, list):
                ci = ', '.join(ci)
            hl += f"<br/><font size='8' color='#6B7280'>Column: {ci}</font>"

        hd = Table(
            [[Paragraph(f"<font color='white'><b> {label} </b></font>", s['cap']),
              Paragraph(hl, s['bb'])]],
            colWidths=[0.85 * inch, cw - 0.85 * inch],
        )
        hd.setStyle(_ts(
            ('BACKGROUND', (0,0), (0,0), color),
            ('BACKGROUND', (1,0), (1,0), CARD_BG),
            ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING',   (0,0), (-1,-1), 10),
            ('RIGHTPADDING',  (0,0), (-1,-1), 10),
            ('TOPPADDING',    (0,0), (-1,-1), 9),
            ('BOTTOMPADDING', (0,0), (-1,-1), 9),
            ('BOX',        (0,0), (-1,-1), 0.75, color),
        ))
        items = [hd]

        def _row(txt, style, bg, lc=None):
            t = Table([[Paragraph(txt, style)]], colWidths=[cw])
            cmds = [
                ('BACKGROUND',    (0,0), (-1,-1), bg),
                ('LEFTPADDING',   (0,0), (-1,-1), 14),
                ('RIGHTPADDING',  (0,0), (-1,-1), 12),
                ('TOPPADDING',    (0,0), (-1,-1), 7),
                ('BOTTOMPADDING', (0,0), (-1,-1), 7),
                ('LINEBELOW',     (0,-1), (-1,-1), 0.5, EDGE),
            ]
            if lc:
                cmds.append(('LINEBEFORE', (0,0), (0,-1), 2.5, lc))
            t.setStyle(_ts(*cmds))
            return t

        w = ins.get('what_it_means', '')
        if w:
            items.append(_row(f"<b>What it means:</b> {w}", s['b'], WHITE, color))
        bi = ins.get('business_impact', '')
        if bi:
            items.append(_row(f"<b>Why it matters:</b> {bi}", s['imp'], SOFT_AMBER,
                              CRITICAL if label == "CRITICAL" else HIGH))
        act = ins.get('what_to_do', '')
        if act:
            items.append(_row(f"<b>What to do:</b> {act}", s['act'], SOFT_TEAL, SUCCESS))
        dd = ins.get('deep_dive', '')
        if dd:
            t = Table([[Paragraph(f"<font color='#6D28D9'>&#9733; Deeper look:</font> {dd}", s['dd'])]],
                      colWidths=[cw])
            t.setStyle(_ts(
                ('BACKGROUND',    (0,0), (-1,-1), DEEPDIVE),
                ('LEFTPADDING',   (0,0), (-1,-1), 14),
                ('RIGHTPADDING',  (0,0), (-1,-1), 12),
                ('TOPPADDING',    (0,0), (-1,-1), 7),
                ('BOTTOMPADDING', (0,0), (-1,-1), 7),
                ('LINEBEFORE',    (0,0), (0,-1), 2.5, colors.HexColor("#7C3AED")),
            ))
            items.append(t)
        return items

    # ── Model Recommendation ───────────────────────────────────────────────
    def _model_section(self, s, ins, rec):
        it = [_section_header(_icon_model, "Recommended Approach", s),
              HRFlowable(width="100%", thickness=1.5, color=BRAND, spaceAfter=10)]
        g = ins.get('model_guidance', {}) or {}
        if not g:
            g = {'recommended_model': rec.get('primary_model', '—'), 'task_type': rec.get('task_type', '—'),
                 'why_this_model': rec.get('why_this_model', ''), 'key_reasons': rec.get('reasoning', []),
                 'alternatives': rec.get('alternatives', []), 'before_you_train': rec.get('preprocessing_steps', []),
                 'how_to_validate': rec.get('cv_strategy', ''),
                 'how_to_measure_success': rec.get('recommended_metrics', []),
                 'confidence_label': '', 'confidence_score': rec.get('confidence', 0)}
        mn   = g.get('recommended_model', '—')
        task = g.get('task_type', '—').title()
        conf = g.get('confidence_label', '')
        why  = g.get('why_this_model', '')
        reasons = g.get('key_reasons', [])
        alts    = g.get('alternatives', [])
        prep    = g.get('before_you_train', [])
        cv      = g.get('how_to_validate_narrative', '') or g.get('how_to_validate', '')
        metrics = g.get('success_metrics_narrative', '') or g.get('how_to_measure_success', [])

        # Primary model highlight box
        pt = Table(
            [[Paragraph(f"<font color='white'><b>{mn}</b></font>", s['sh']),
              Paragraph(f"<font color='#A5B4FC'>Task: {task}</font><br/><font color='#FCD34D'>{conf}</font>", s['cm'])]],
            colWidths=[3 * inch, CW - 3 * inch - 24],
        )
        pt.setStyle(_ts(
            ('BACKGROUND',    (0,0), (-1,-1), COVER_BG),
            ('LEFTPADDING',   (0,0), (-1,-1), 16),
            ('RIGHTPADDING',  (0,0), (-1,-1), 14),
            ('TOPPADDING',    (0,0), (-1,-1), 14),
            ('BOTTOMPADDING', (0,0), (-1,-1), 14),
            ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
            ('LINEBEFORE',    (0,0), (0,-1),  5, ACCENT),
        ))
        it += [pt, Spacer(1, 10)]
        if why:
            it.append(Paragraph(f"<b>Why this model:</b> {why}", s['b']))
        for r in reasons:
            it.append(Paragraph(f"&#8226; {r}", s['b']))
        it.append(Spacer(1, 8))
        if alts:
            it.append(Paragraph("Other options worth considering", s['sub']))
            for a in alts:
                it.append(Paragraph(f"<b>{a.get('model', '')}</b> — {a.get('why', '')}", s['b']))
            it.append(Spacer(1, 8))
        if prep:
            it.append(Paragraph("Before you start — data preparation steps", s['sub']))
            narrative = g.get('before_you_train_narrative', '')
            if narrative:
                it.append(Paragraph(narrative, s['b']))
            else:
                for i, step in enumerate(prep, 1):
                    it.append(Paragraph(f"{i}. {step}", s['b']))
            it.append(Spacer(1, 8))
        if cv:
            it += [Paragraph("How to check if it's working", s['sub']), Paragraph(cv, s['b']), Spacer(1, 8)]
        if metrics:
            it.append(Paragraph("What a good result looks like", s['sub']))
            if isinstance(metrics, str):
                it.append(Paragraph(metrics, s['b']))
            else:
                for m in metrics:
                    it.append(Paragraph(f"&#8226; {m}", s['b']))
        it.append(Spacer(1, 6))
        return it

    # ── Quick Wins ─────────────────────────────────────────────────────────
    def _quick_wins(self, s, ins):
        wins = ins.get('quick_wins', [])
        if not wins:
            return []
        it = [_section_header(_icon_lightning, "Quick Wins — Start Here", s),
              HRFlowable(width="100%", thickness=1.5, color=ACCENT, spaceAfter=10),
              Paragraph("These are the easiest, highest-impact steps to take right now — tackle them in order.", s['b']),
              Spacer(1, 10)]
        rows = [["#", "Action"]] + [[str(i), w] for i, w in enumerate(wins, 1)]
        t = _data_table(rows, [0.4 * inch, CW - 0.4 * inch], [
            ('ALIGN',    (0,0), (0,-1), 'CENTER'),
            ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
            ('TEXTCOLOR',(0,1), (0,-1), ACCENT),
            ('VALIGN',   (0,0), (-1,-1), 'TOP'),
        ])
        return it + [t, Spacer(1, 8)]

    # ── Column Relationships ───────────────────────────────────────────────
    def _column_relationships(self, s, ins):
        notable = [r for r in ins.get('column_relationships', [])
                   if r.get('severity') in ('critical', 'high', 'medium')]
        if not notable:
            return []
        it = [PageBreak(),
              _section_header(_icon_link, "Column Relationships", s),
              HRFlowable(width="100%", thickness=1.5, color=TEAL, spaceAfter=10),
              Paragraph("These columns move together in ways that could affect your analysis — good to know about before building a model.", s['b']),
              Spacer(1, 10)]
        for r in notable:
            sev = r.get('severity', 'medium')
            col = CRITICAL if sev == 'critical' else HIGH if sev == 'high' else MEDIUM
            corr = r.get('correlation')
            cs   = f"{corr:+.2f}" if corr is not None else "reversal"
            pair = f"{r.get('col_a', '')}  <->  {r.get('col_b', '')}"
            cl   = f"{cs}  {r.get('direction', '')}" if corr is not None else f"Reverses by: {r.get('split_by', '')}"
            ht = Table(
                [[Paragraph(f"<b>{pair}</b>", s['bb']),
                  Paragraph(f"<font color='#6B7280'>{cl}</font>", s['cap'])]],
                colWidths=[CW * 0.65, CW * 0.35],
            )
            ht.setStyle(_ts(
                ('BACKGROUND',    (0,0), (-1,-1), CARD_BG),
                ('LEFTPADDING',   (0,0), (-1,-1), 12),
                ('RIGHTPADDING',  (0,0), (-1,-1), 10),
                ('TOPPADDING',    (0,0), (-1,-1), 8),
                ('BOTTOMPADDING', (0,0), (-1,-1), 8),
                ('LINEBEFORE',    (0,0), (0,-1),  3.5, col),
                ('ALIGN',         (1,0), (1,-1), 'RIGHT'),
                ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
            ))
            card = [ht]
            for txt, style, bg, lc in [
                (r.get('explanation', ''), s['b'],   WHITE,     None),
                (f"<b>What to do:</b> {r.get('action', '')}", s['act'], SOFT_TEAL, col),
            ]:
                if not txt:
                    continue
                rt = Table([[Paragraph(txt, style)]], colWidths=[CW])
                cmds = [
                    ('BACKGROUND',    (0,0), (-1,-1), bg),
                    ('LEFTPADDING',   (0,0), (-1,-1), 14),
                    ('RIGHTPADDING',  (0,0), (-1,-1), 12),
                    ('TOPPADDING',    (0,0), (-1,-1), 7),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 7),
                    ('LINEBELOW',     (0,-1), (-1,-1), 0.5, EDGE),
                ]
                if lc:
                    cmds.append(('LINEBEFORE', (0,0), (0,-1), 1.5, lc))
                rt.setStyle(_ts(*cmds))
                card.append(rt)
            it += [KeepTogether(card), Spacer(1, 10)]
        return it

    # ── Imbalance Guidance ─────────────────────────────────────────────────
    def _imbalance_guidance(self, s, ins):
        g = ins.get('class_imbalance_guidance')
        if not g:
            return []
        it = [_section_header(_icon_balance, "Class Imbalance — Important!", s),
              HRFlowable(width="100%", thickness=1.5, color=CRITICAL, spaceAfter=10)]
        tgt, maj = g.get('target_column', '—'), g.get('majority_pct', 0)
        it += [Paragraph(f'The column <b>"{tgt}"</b> has {maj:.0f}% of its values in a single group — this is called class imbalance.', s['bb']),
               Spacer(1, 8)]
        why = g.get('why_it_matters', '')
        if why:
            it += [self._box(why, s, bg=SOFT_RED, border=CRITICAL), Spacer(1, 12)]
        tg = g.get('technique_guidance', '')
        if tg:
            it += [Paragraph("Which fix to try first", s['sub']), Paragraph(tg, s['b']), Spacer(1, 6)]
        fs = g.get('first_step', '')
        if fs:
            it += [self._box(f"<b>Start here:</b> {fs}", s, bg=SOFT_TEAL, border=SUCCESS), Spacer(1, 10)]
        techs = g.get('techniques', [])
        if techs:
            it.append(Paragraph("Available techniques", s['sub']))
            rows = [["Technique", "Difficulty", "What it does"]] + [
                [Paragraph(f"<b>{t.get('name', '')}</b>", s['b']),
                 t.get('difficulty', '').title(),
                 Paragraph(t.get('description', ''), s['b'])]
                for t in techs
            ]
            it += [_data_table(rows, [CW * 0.27, CW * 0.13, CW * 0.60],
                               [('FONTSIZE', (0,0), (-1,-1), 9), ('VALIGN', (0,0), (-1,-1), 'TOP')]),
                   Spacer(1, 10)]
        it.append(Paragraph("How to measure success (not the usual way!)", s['sub']))
        mr = g.get('metric_reasoning', '') or g.get('metric_explanation', '')
        if mr:
            it.append(Paragraph(mr, s['b']))
        wm = g.get('wrong_metrics', [])
        if wm:
            it.append(Paragraph(f"<font color='#EF4444'>&#10007; Avoid using:</font> {', '.join(wm)}", s['b']))
        rm = g.get('right_metrics', [])
        if rm:
            it.append(Paragraph(f"<font color='#10B981'>&#10003; Use these instead:</font> {', '.join(rm)}", s['b']))
        it.append(Spacer(1, 10))
        return it

    # ── Column Profiles ────────────────────────────────────────────────────
    def _column_profiles(self, s, st):
        profs = st.get('column_profiles', [])
        if not profs:
            return []
        it = [PageBreak(),
              _section_header(_icon_checklist, "Column Profiles", s),
              HRFlowable(width="100%", thickness=1.5, color=BRAND, spaceAfter=10),
              Paragraph("A snapshot of every column — useful for spotting patterns at a glance.", s['b']),
              Spacer(1, 10)]
        rows = [["Column", "Type", "Missing", "Unique", "Notes"]]
        for p in profs:
            notes = []
            if p.get('missing_pct', 0) >= 50:      notes.append("high missingness")
            if p.get('skewness') and abs(p['skewness']) > 2: notes.append(f"skewed ({p['skewness']:.1f})")
            if p.get('disguised_missing', 0) > 0:   notes.append(f"{p['disguised_missing']} disguised nulls")
            if p.get('whitespace_issues', 0) > 0:   notes.append("whitespace issues")
            if p.get('inf_count', 0) > 0:           notes.append(f"{p['inf_count']} infinities")
            rows.append([p.get('name', ''), p.get('kind', '').replace('_', ' '),
                         f"{p.get('missing_pct', 0):.1f}%", str(p.get('unique_count', '')),
                         ', '.join(notes) or '—'])
        it.append(_data_table(rows, [CW * 0.25, CW * 0.18, CW * 0.12, CW * 0.10, CW * 0.35], [
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('VALIGN',   (0,0), (-1,-1), 'TOP'),
            ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
        ]))
        it.append(Spacer(1, 8))
        return it

    # ── Closing ────────────────────────────────────────────────────────────
    def _closing(self, s, at):
        # Simple checkmark illustration
        d = Drawing(CW, 70)
        cx, cy = CW / 2, 35
        d.add(Circle(cx, cy, 26, fillColor=SOFT_TEAL, strokeColor=SUCCESS, strokeWidth=2.5))
        # Checkmark legs
        d.add(Line(cx - 12, cy, cx - 4, cy - 10, strokeColor=SUCCESS, strokeWidth=3))
        d.add(Line(cx - 4, cy - 10, cx + 14, cy + 12, strokeColor=SUCCESS, strokeWidth=3))
        closing_art = _IconFlowable(d, CW, 70)
        return [
            Spacer(1, 1.2 * inch),
            closing_art,
            Spacer(1, 16),
            Paragraph("You're all set!", s['sh']),
            HRFlowable(width="100%", thickness=1.5, color=SUCCESS, spaceAfter=14),
            Paragraph(
                "This report was generated automatically by DataSense. All findings are based on a "
                "statistical analysis of your uploaded dataset. It is always a good idea to double-check "
                "important findings with someone familiar with your data before making major decisions.",
                s['b'],
            ),
            Spacer(1, 12),
            Paragraph(f"Generated on {at}", s['cap']),
        ]