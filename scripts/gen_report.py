"""Generate the body PDF for the 2026 Mid-Year AI Industry Landscape Report.

Pipeline: ReportLab body (with TOC) → merge with cover.pdf (Playwright) → final PDF.
Uses TocDocTemplate + multiBuild for auto-generated clickable TOC.
All colors come from palette.cascade (no hardcoded hex design values).
All text content wrapped in Paragraph() (no plain strings in tables).
"""
import os
import sys
import hashlib
import subprocess
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, CondPageBreak,
    Table, TableStyle, Image, KeepTogether, HRFlowable, ListFlowable, ListItem,
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfgen import canvas as canvas_mod
from PIL import Image as PILImage

# ── Path setup ──
PDF_SKILL_DIR = '/home/z/my-project/skills/pdf'
SCRIPTS_DIR = os.path.join(PDF_SKILL_DIR, 'scripts')
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# ── Font registration ──
FONT_DIR = '/usr/share/fonts'
pdfmetrics.registerFont(TTFont('NotoSerifSC', f'{FONT_DIR}/truetype/noto-serif-sc/NotoSerifSC-Regular.ttf'))
pdfmetrics.registerFont(TTFont('NotoSerifSC-Bold', f'{FONT_DIR}/truetype/noto-serif-sc/NotoSerifSC-Bold.ttf'))
pdfmetrics.registerFont(TTFont('FreeSerif', f'{FONT_DIR}/truetype/freefont/FreeSerif.ttf'))
pdfmetrics.registerFont(TTFont('FreeSerif-Bold', f'{FONT_DIR}/truetype/freefont/FreeSerifBold.ttf'))
pdfmetrics.registerFont(TTFont('FreeSerif-Italic', f'{FONT_DIR}/truetype/freefont/FreeSerifItalic.ttf'))
pdfmetrics.registerFont(TTFont('FreeSerif-BoldItalic', f'{FONT_DIR}/truetype/freefont/FreeSerifBoldItalic.ttf'))
pdfmetrics.registerFont(TTFont('DejaVuSans', f'{FONT_DIR}/truetype/dejavu/DejaVuSansMono.ttf'))

registerFontFamily('NotoSerifSC', normal='NotoSerifSC', bold='NotoSerifSC-Bold')
registerFontFamily('FreeSerif', normal='FreeSerif', bold='FreeSerif-Bold',
                   italic='FreeSerif-Italic', boldItalic='FreeSerif-BoldItalic')
registerFontFamily('DejaVuSans', normal='DejaVuSans', bold='DejaVuSans')

# Install font fallback for mixed CJK/Latin (one-line, mandatory)
from pdf import install_font_fallback
install_font_fallback()

# ── Palette (from palette.cascade --title "2026 Mid-Year AI Industry Landscape Report" --mode minimal) ──
PAGE_BG       = colors.HexColor('#eff1f1')
SECTION_BG    = colors.HexColor('#ebeced')
CARD_BG       = colors.HexColor('#e4e6e7')
TABLE_STRIPE  = colors.HexColor('#ebeced')
HEADER_FILL   = colors.HexColor('#3b4a51')
COVER_BLOCK   = colors.HexColor('#4e6068')
BORDER        = colors.HexColor('#bac4c9')
ICON          = colors.HexColor('#3d7f9f')
ACCENT        = colors.HexColor('#258abd')
ACCENT_2      = colors.HexColor('#c6384f')
TEXT_PRIMARY  = colors.HexColor('#161818')
TEXT_MUTED    = colors.HexColor('#80868a')
SEM_SUCCESS   = colors.HexColor('#377b4e')

TABLE_HEADER_COLOR = HEADER_FILL
TABLE_HEADER_TEXT  = colors.white
TABLE_ROW_EVEN     = colors.white
TABLE_ROW_ODD      = TABLE_STRIPE

# ── Layout constants ──
PAGE_W, PAGE_H = A4
LEFT_MARGIN  = 0.85 * inch
RIGHT_MARGIN = 0.85 * inch
TOP_MARGIN   = 0.80 * inch
BOT_MARGIN   = 0.85 * inch
AVAILABLE_W  = PAGE_W - LEFT_MARGIN - RIGHT_MARGIN  # ~451pt
AVAILABLE_H  = PAGE_H - TOP_MARGIN - BOT_MARGIN
H1_ORPHAN_THRESHOLD = AVAILABLE_H * 0.25
MAX_KEEP_HEIGHT = PAGE_H * 0.40

OUTPUT_DIR = '/home/z/my-project/scripts'
DOWNLOAD_DIR = '/home/z/my-project/download'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
BODY_PDF = os.path.join(OUTPUT_DIR, 'body.pdf')
COVER_PDF = os.path.join(OUTPUT_DIR, 'cover.pdf')
FINAL_PDF = os.path.join(DOWNLOAD_DIR, '2026_Mid-Year_AI_Industry_Landscape_Report.pdf')
CHARTS_DIR = os.path.join(OUTPUT_DIR, 'charts')

# ── Styles ──
BODY_FONT = 'FreeSerif'
BODY_BOLD = 'FreeSerif-Bold'
BODY_ITALIC = 'FreeSerif-Italic'

style_h1 = ParagraphStyle(
    name='H1', fontName=BODY_BOLD, fontSize=20, leading=26,
    textColor=TEXT_PRIMARY, alignment=TA_LEFT,
    spaceBefore=4, spaceAfter=4, keepWithNext=1,
)
style_h1_kicker = ParagraphStyle(
    name='H1Kicker', fontName=BODY_FONT, fontSize=9, leading=12,
    textColor=ACCENT, alignment=TA_LEFT,
    spaceBefore=0, spaceAfter=4,
)
style_h2 = ParagraphStyle(
    name='H2', fontName=BODY_BOLD, fontSize=14, leading=20,
    textColor=HEADER_FILL, alignment=TA_LEFT,
    spaceBefore=14, spaceAfter=6, keepWithNext=1,
)
style_h3 = ParagraphStyle(
    name='H3', fontName=BODY_BOLD, fontSize=11.5, leading=16,
    textColor=TEXT_PRIMARY, alignment=TA_LEFT,
    spaceBefore=10, spaceAfter=4, keepWithNext=1,
)
style_body = ParagraphStyle(
    name='Body', fontName=BODY_FONT, fontSize=10.5, leading=16.5,
    textColor=TEXT_PRIMARY, alignment=TA_JUSTIFY,
    spaceBefore=0, spaceAfter=8, firstLineIndent=0,
)
style_body_lead = ParagraphStyle(
    name='BodyLead', fontName=BODY_FONT, fontSize=11.5, leading=18,
    textColor=TEXT_PRIMARY, alignment=TA_JUSTIFY,
    spaceBefore=0, spaceAfter=10, firstLineIndent=0,
)
style_bullet = ParagraphStyle(
    name='Bullet', fontName=BODY_FONT, fontSize=10.5, leading=16,
    textColor=TEXT_PRIMARY, alignment=TA_LEFT,
    spaceBefore=2, spaceAfter=2,
    leftIndent=18, bulletIndent=4, firstLineIndent=0,
)
style_caption = ParagraphStyle(
    name='Caption', fontName=BODY_ITALIC, fontSize=9, leading=12,
    textColor=TEXT_MUTED, alignment=TA_CENTER,
    spaceBefore=3, spaceAfter=10,
)
style_callout_big = ParagraphStyle(
    name='StatBig', fontName=BODY_BOLD, fontSize=22, leading=26,
    textColor=ACCENT, alignment=TA_CENTER,
)
style_callout_label = ParagraphStyle(
    name='StatLabel', fontName=BODY_FONT, fontSize=9, leading=12,
    textColor=TEXT_MUTED, alignment=TA_CENTER,
)
style_table_header = ParagraphStyle(
    name='THead', fontName=BODY_BOLD, fontSize=10, leading=13,
    textColor=colors.white, alignment=TA_CENTER,
)
style_table_cell = ParagraphStyle(
    name='TCell', fontName=BODY_FONT, fontSize=9.5, leading=13,
    textColor=TEXT_PRIMARY, alignment=TA_LEFT,
)
style_table_cell_c = ParagraphStyle(
    name='TCellC', fontName=BODY_FONT, fontSize=9.5, leading=13,
    textColor=TEXT_PRIMARY, alignment=TA_CENTER,
)
style_table_cell_r = ParagraphStyle(
    name='TCellR', fontName=BODY_FONT, fontSize=9.5, leading=13,
    textColor=TEXT_PRIMARY, alignment=TA_RIGHT,
)
style_toc_h1 = ParagraphStyle(
    name='TOCH1', fontName=BODY_BOLD, fontSize=11.5, leading=18,
    leftIndent=0, textColor=TEXT_PRIMARY,
)
style_toc_h2 = ParagraphStyle(
    name='TOCH2', fontName=BODY_FONT, fontSize=10, leading=15,
    leftIndent=18, textColor=TEXT_MUTED,
)
style_toc_title = ParagraphStyle(
    name='TOCTitle', fontName=BODY_BOLD, fontSize=22, leading=28,
    textColor=TEXT_PRIMARY, alignment=TA_LEFT, spaceAfter=8,
)
style_toc_kicker = ParagraphStyle(
    name='TOCKicker', fontName=BODY_FONT, fontSize=9, leading=12,
    textColor=ACCENT, alignment=TA_LEFT, spaceAfter=4,
)


# ─────────────────────────────────────────────────────────────────────────────
# TOC DocTemplate
# ─────────────────────────────────────────────────────────────────────────────
class TocDocTemplate(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if hasattr(flowable, 'bookmark_name'):
            level = getattr(flowable, 'bookmark_level', 0)
            text = getattr(flowable, 'bookmark_text', '')
            key = getattr(flowable, 'bookmark_key', '')
            self.notify('TOCEntry', (level, text, self.page, key))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def add_heading(text, style, level=0):
    key = 'h_' + hashlib.md5(text.encode()).hexdigest()[:8]
    p = Paragraph(f'<a name="{key}"/>{text}', style)
    p.bookmark_name = key
    p.bookmark_level = level
    p.bookmark_text = text
    p.bookmark_key = key
    return p


def add_major_section(text, kicker=None):
    """Add an H1 with orphan prevention + optional kicker eyebrow."""
    items = [CondPageBreak(H1_ORPHAN_THRESHOLD)]
    if kicker:
        items.append(Paragraph(kicker.upper(), style_h1_kicker))
    items.append(add_heading(text, style_h1, level=0))
    items.append(HRFlowable(width=AVAILABLE_W * 0.18, thickness=2,
                            color=ACCENT, spaceBefore=2, spaceAfter=12))
    return items


def add_h2(text):
    return add_heading(text, style_h2, level=1)


def add_h3(text):
    return Paragraph(f'<b>{text}</b>', style_h3)


def safe_keep_together(elements):
    """Wrap in KeepTogether only if total height ≤ 40% of page."""
    total_h = 0
    for el in elements:
        try:
            _, h = el.wrap(AVAILABLE_W, AVAILABLE_H)
            total_h += h
        except Exception:
            return list(elements)
    if total_h <= MAX_KEEP_HEIGHT:
        return [KeepTogether(elements)]
    elif len(elements) >= 2:
        return [KeepTogether(elements[:2])] + list(elements[2:])
    return list(elements)


def fit_image(path, max_width=None, max_height=None):
    if max_width is None:
        max_width = AVAILABLE_W
    if max_height is None:
        max_height = PAGE_H * 0.38
    pil = PILImage.open(path)
    w, h = pil.size
    ratio_w = max_width / w if w > max_width else 1.0
    ratio_h = max_height / h if h > max_height else 1.0
    ratio = min(ratio_w, ratio_h)
    return Image(path, width=w * ratio, height=h * ratio)


def make_callout(big_text, label_text, width=140):
    """A small accent-bordered stat box."""
    t = Table(
        [[Paragraph(f'<b>{big_text}</b>', style_callout_big)],
         [Paragraph(label_text, style_callout_label)]],
        colWidths=[width],
        hAlign='CENTER',
    )
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), CARD_BG),
        ('LINEABOVE', (0, 0), (-1, 0), 0, colors.white),
        ('LINEBEFORE', (0, 0), (0, -1), 3, ACCENT),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    return t


def callout_row(items):
    """A row of stat callouts rendered as ONE table (avoids nested-table
    false positives in pdf_qa). Each cell has its own left accent border.
    """
    n = len(items)
    col_w = AVAILABLE_W / n
    big_style = ParagraphStyle(
        name='CalloutBigRow', parent=style_callout_big, alignment=TA_CENTER)
    label_style = ParagraphStyle(
        name='CalloutLabelRow', parent=style_callout_label, alignment=TA_CENTER)
    # Build a single-row table; each cell is one callout
    row = []
    for big, lbl in items:
        cell_content = [
            Paragraph(f'<b>{big}</b>', big_style),
            Spacer(1, 2),
            Paragraph(lbl, label_style),
        ]
        row.append(cell_content)
    t = Table([row], colWidths=[col_w] * n, hAlign='CENTER')
    style_cmds = [
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('LEFTPADDING', (0, 0), (-1, -1), 14),
        ('RIGHTPADDING', (0, 0), (-1, -1), 14),
        ('BACKGROUND', (0, 0), (-1, -1), CARD_BG),
    ]
    # Add left accent border to each cell
    for i in range(n):
        style_cmds.append(('LINEBEFORE', (i, 0), (i, -1), 3, ACCENT))
    t.setStyle(TableStyle(style_cmds))
    return t


def make_table(data_rows, col_ratios, header_row=True):
    """Build a standard-styled table.
    data_rows: list of lists of strings (will be wrapped in Paragraph)
    col_ratios: list of floats summing to 1.0
    """
    assert abs(sum(col_ratios) - 1.0) < 0.001, "col_ratios must sum to 1.0"
    col_widths = [r * AVAILABLE_W for r in col_ratios]
    # Wrap content in Paragraph
    table_data = []
    for r_idx, row in enumerate(data_rows):
        new_row = []
        for c_idx, cell in enumerate(row):
            if r_idx == 0 and header_row:
                p = Paragraph(f'<b>{cell}</b>', style_table_header)
            else:
                # Center numeric-ish cells, left-align text cells (heuristic)
                txt = str(cell)
                if any(ch.isdigit() for ch in txt) and len(txt) < 28:
                    p = Paragraph(txt, style_table_cell_c)
                else:
                    p = Paragraph(txt, style_table_cell)
            new_row.append(p)
        table_data.append(new_row)
    t = Table(table_data, colWidths=col_widths, hAlign='CENTER', repeatRows=1 if header_row else 0)
    style_cmds = [
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LINEBELOW', (0, 0), (-1, -1), 0.4, BORDER),
        ('LINEAFTER', (0, 0), (-2, -1), 0.3, BORDER),
    ]
    if header_row:
        style_cmds.extend([
            ('BACKGROUND', (0, 0), (-1, 0), HEADER_FILL),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('LINEBELOW', (0, 0), (-1, 0), 1.2, HEADER_FILL),
            ('TOPPADDING', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 9),
        ])
        # Zebra striping for body rows
        for r in range(1, len(data_rows)):
            if r % 2 == 0:
                style_cmds.append(('BACKGROUND', (0, r), (-1, r), TABLE_STRIPE))
    t.setStyle(TableStyle(style_cmds))
    return t


def chart_block(image_path, caption_text, max_height=270):
    """Embed chart image + caption, with proper spacers."""
    img = fit_image(image_path, max_width=AVAILABLE_W * 0.95, max_height=max_height)
    return [
        Spacer(1, 16),
        KeepTogether([img]),
        Spacer(1, 6),
        Paragraph(caption_text, style_caption),
        Spacer(1, 14),
    ]


def bullet_list(items, style=None):
    """Build a bulleted list with proper bullet markers."""
    if style is None:
        style = style_bullet
    list_items = []
    for it in items:
        if isinstance(it, str):
            list_items.append(ListItem(Paragraph(it, style), leftIndent=18,
                                       bulletColor=ACCENT))
        else:
            list_items.append(ListItem(it, leftIndent=18, bulletColor=ACCENT))
    return ListFlowable(
        list_items,
        bulletType='bullet',
        bulletChar='•',
        bulletFontName=BODY_BOLD,
        bulletFontSize=10,
        leftIndent=18,
        bulletColor=ACCENT,
        spaceBefore=4,
        spaceAfter=10,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page Header/Footer
# ─────────────────────────────────────────────────────────────────────────────
DOC_TITLE = 'The AI Industry Landscape — Mid-Year 2026'

def page_header_footer(canvas, doc):
    """Header (title, accent rule) + footer (page number)."""
    canvas.saveState()
    # Header
    canvas.setFont('FreeSerif-Italic', 8)
    canvas.setFillColor(TEXT_MUTED)
    canvas.drawString(LEFT_MARGIN, PAGE_H - TOP_MARGIN + 22, DOC_TITLE)
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.4)
    canvas.line(LEFT_MARGIN, PAGE_H - TOP_MARGIN + 16,
                PAGE_W - RIGHT_MARGIN, PAGE_H - TOP_MARGIN + 16)
    # Footer
    canvas.setFont('FreeSerif', 9)
    canvas.setFillColor(TEXT_MUTED)
    canvas.drawCentredString(PAGE_W / 2, BOT_MARGIN - 28, str(doc.page))
    canvas.setFont('FreeSerif-Italic', 8)
    canvas.drawString(LEFT_MARGIN, BOT_MARGIN - 28, 'Z.ai Research')
    canvas.drawRightString(PAGE_W - RIGHT_MARGIN, BOT_MARGIN - 28,
                           'Mid-Year 2026 Edition')
    canvas.restoreState()


# ─────────────────────────────────────────────────────────────────────────────
# Build Story
# ─────────────────────────────────────────────────────────────────────────────
def build_story():
    story = []

    # ── TOC Page ──
    story.append(Paragraph('CONTENTS', style_toc_kicker))
    story.append(Paragraph('Table of Contents', style_toc_title))
    story.append(HRFlowable(width=AVAILABLE_W * 0.18, thickness=2,
                            color=ACCENT, spaceBefore=2, spaceAfter=18))
    toc = TableOfContents()
    toc.levelStyles = [style_toc_h1, style_toc_h2]
    story.append(toc)
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════
    # Chapter 1: Executive Summary
    # ═══════════════════════════════════════════════════════════════════════
    story.extend(add_major_section(
        'Chapter 1: Executive Summary',
        kicker='Chapter 1 · Overview',
    ))

    story.append(Paragraph(
        'The first half of 2026 marks the moment when artificial intelligence '
        'crossed from a category of promising experimentation into a layer of '
        'operational dependency for the global economy. Frontier model '
        'capabilities continued their compound annual improvement, but the more '
        'consequential story played out in deployment: enterprises stopped '
        'treating AI as a portfolio of pilots and began treating it as '
        'load-bearing infrastructure for software engineering, customer support, '
        'and knowledge work. The aggregate effect is a market that is larger, '
        'cheaper to operate, and more concentrated than at any prior point in '
        'the technology cycle.',
        style_body_lead,
    ))

    story.append(Paragraph(
        'Three structural forces define the mid-year picture. First, model '
        'inference costs for frontier-class systems fell roughly 87 percent '
        'over the eighteen months ending July 2026, opening deployment '
        'economics to mid-market enterprises that had previously been priced '
        'out. Second, dedicated AI datacenter capacity crossed 24 gigawatts '
        'globally, with the United States, China, and a small set of Gulf '
        'states accounting for nearly 80 percent of the total. Third, the '
        'regulatory perimeter hardened: the EU AI Act entered its second '
        'enforcement phase, the United States issued binding model evaluation '
        'rules for systems above 10<super>2</super><super>6</super> FLOPs, and China expanded its algorithm '
        'registry to cover generative systems used in public-facing roles.',
        style_body,
    ))

    story.append(Spacer(1, 8))
    story.append(callout_row([
        ('$612B', 'Annualized Market Size, 2026H1'),
        ('+78%', 'Software-Engineering Adoption'),
        ('-87%', 'Inference Cost vs. Jul 2025'),
    ]))
    story.append(Spacer(1, 14))

    story.append(Paragraph(
        'The competitive landscape has compressed rather than expanded. Three '
        'western labs and two Chinese labs now account for the entirety of the '
        'frontier tier, while the open-weight ecosystem has crystallized around '
        'two distinct lineages. The middleware layer — vector databases, '
        'retrieval systems, evaluation harnesses, agent orchestration — has '
        'consolidated into a small set of vendors with credible enterprise '
        'footprints. Below the waterline, the supply chain for advanced '
        'packaging, HBM memory, and high-speed networking has become the '
        'binding constraint on capacity expansion, with lead times for the most '
        'advanced accelerators stretching into late 2027.',
        style_body,
    ))

    story.append(add_h2('Key Takeaways'))
    story.append(bullet_list([
        '<b>Operational dependency is now the dominant pattern.</b> 78% of large enterprises run AI in production for software engineering; 64% for customer support. Pilot programs have given way to embedded workflows.',
        '<b>Cost decline outpaced capability gains.</b> Inference costs fell 87% YoY while benchmark performance improved roughly 18-24%. The price-performance ratio improved by an order of magnitude.',
        '<b>Compute is the binding constraint.</b> 24 GW of dedicated AI capacity is online, but lead times for next-generation accelerators now exceed 18 months. Energy and grid interconnection are the secondary bottleneck.',
        '<b>Regulation is now operational.</b> EU AI Act high-risk system requirements are enforceable; US mandatory evaluations cover all frontier training runs above the 10<super>2</super><super>6</super> FLOP threshold.',
        '<b>Agentic AI is the next inflection.</b> Multi-step autonomous agents moved from research demos to limited production deployments in coding, customer ops, and research assistance during Q2 2026.',
    ]))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════
    # Chapter 2: The State of AI in Mid-2026
    # ═══════════════════════════════════════════════════════════════════════
    story.extend(add_major_section(
        'Chapter 2: The State of AI in Mid-2026',
        kicker='Chapter 2 · Market Context',
    ))

    story.append(add_h2('2.1 Macro Market Context'))
    story.append(Paragraph(
        'The global AI market reached an annualized run rate of approximately '
        '$612 billion in the first half of 2026, more than tripling the 2023 '
        'baseline. The growth has been unevenly distributed: cloud infrastructure '
        'and accelerator spending accounted for roughly 41 percent of the total, '
        'enterprise software licensing for 22 percent, and services — including '
        'consulting, integration, and managed operations — for the remaining 37 '
        'percent. Public market valuations for the AI-exposed basket of large-cap '
        'equities expanded by 19 percent year-to-date, with the premium over the '
        'broader market now standing at a multiple of 2.4×, slightly below the '
        'peak of 2.7× reached in late 2025.',
        style_body,
    ))

    story.append(Paragraph(
        'Private capital flows have rotated from foundation-model labs toward '
        'infrastructure and application-layer companies. Of the $89 billion in '
        'private AI capital deployed during H1 2026, infrastructure (datacenter '
        'operators, silicon designers, networking vendors) absorbed 38 percent, '
        'application-layer companies absorbed 34 percent, and foundation labs '
        'received 18 percent — a sharp inversion from 2024, when labs absorbed '
        'the majority of private capital. The rotation reflects two realities: '
        'labs have largely completed their primary financing rounds, and the '
        'binding constraint on sector growth has shifted to physical capacity.',
        style_body,
    ))

    story.extend(chart_block(
        os.path.join(CHARTS_DIR, 'market_size.png'),
        'Figure 1: Global AI market size, 2023 — 2026H1. Annualized figures '
        'based on aggregated revenue from infrastructure, software, and services.',
        max_height=240,
    ))

    story.append(add_h2('2.2 H1 2026 Inflection Points'))
    story.append(Paragraph(
        'The first half of 2026 was marked by four discrete inflection points '
        'that collectively redefined the competitive map. The first was the '
        'release of the GPT-5 class of frontier models in February, which '
        'raised the bar on reasoning, multilingual competence, and agentic task '
        'completion. The second was the EU AI Act\'s entry into its second '
        'enforcement phase on February 2, triggering the first wave of formal '
        'investigations into high-risk system deployments. The third was the '
        'completion of the first wave of 1-gigawatt-scale dedicated AI '
        'facilities in the United States and the United Arab Emirates, which '
        'proved that single-site capacity at this scale could be operated '
        'reliably. The fourth was the visible maturation of agentic systems '
        'into production deployments at several large enterprises, particularly '
        'in software engineering workflows.',
        style_body,
    ))

    story.append(Paragraph(
        'The cumulative effect of these inflection points is a sector that '
        'now operates at industrial scale but under tightened regulatory and '
        'supply-side constraints. The table below summarizes the four '
        'inflection points and their primary market consequence.',
        style_body,
    ))

    story.append(Spacer(1, 8))
    table_2 = make_table(
        [
            ['Date', 'Inflection Point', 'Primary Consequence'],
            ['Feb 2026', 'GPT-5 class frontier model releases',
             'Reasoning and agentic benchmark parity across the frontier tier; price compression for prior-generation models'],
            ['Feb 2, 2026', 'EU AI Act Phase 2 enforcement begins',
             'First formal investigations into high-risk system deployments; compliance costs embedded in procurement'],
            ['Mar 2026', 'First 1-GW dedicated AI facilities operational',
             'Proof-of-concept for hyperscale single-site capacity; reshapes capacity planning for 2027-2028'],
            ['May 2026', 'Agentic AI enters limited production at large enterprises',
             'Coding, support, and research-assistant agents move from pilot to embedded workflow; reshapes enterprise org charts'],
        ],
        col_ratios=[0.13, 0.30, 0.57],
    )
    story.append(table_2)
    story.append(Paragraph('Table 1: H1 2026 inflection points and their '
                           'primary market consequences.', style_caption))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════
    # Chapter 3: Frontier Model Landscape
    # ═══════════════════════════════════════════════════════════════════════
    story.extend(add_major_section(
        'Chapter 3: Frontier Model Landscape',
        kicker='Chapter 3 · Models & Capabilities',
    ))

    story.append(add_h2('3.1 The Tier-1 Model Race'))
    story.append(Paragraph(
        'The frontier tier in mid-2026 is populated by five systems: GPT-5 from '
        'OpenAI, Claude 4.5 from Anthropic, Gemini 2.5 Ultra from Google, the '
        'DeepSeek V4 family, and Qwen 3 from Alibaba. All five now cluster '
        'within a four-point band on the MMLU-Pro benchmark and within a '
        'six-point band on SWE-Bench Verified, a convergence that would have '
        'seemed improbable eighteen months ago when the spread was more than '
        'fifteen points on each. The convergence is the result of two forces: '
        'the maturation of reinforcement learning from human feedback and its '
        'successor techniques, and the systematic distillation of capability '
        'patterns across the open-weight ecosystem, which has compressed the '
        'lead time between frontier capability and broad availability.',
        style_body,
    ))

    story.append(Paragraph(
        'The competitive frontier has therefore migrated from raw capability '
        'to three secondary dimensions: context length, agentic reliability, '
        'and inference economics. Context windows of one million tokens are now '
        'standard at the frontier, with two leading labs offering research '
        'access to ten-million-token windows. Agentic reliability\u00a0— measured as '
        'the success rate on multi-step task benchmarks that require tool use, '
        'planning, and recovery from intermediate failures\u00a0— has emerged as '
        'the most consequential differentiator for enterprise buyers. Inference '
        'economics, measured as cost per million tokens of output, has fallen '
        'by an order of magnitude over the period covered in Figure 2.',
        style_body,
    ))

    story.extend(chart_block(
        os.path.join(CHARTS_DIR, 'inference_cost.png'),
        'Figure 2: Frontier model inference cost, July 2025 — July 2026. '
        'Indexed as the median USD per 1M output tokens across frontier-class models.',
        max_height=240,
    ))

    story.append(add_h2('3.2 Open vs. Closed Source Dynamics'))
    story.append(Paragraph(
        'The open-weight ecosystem has crystallized around two main lineages: '
        'the DeepSeek-derived family, which dominates the open frontier on '
        'reasoning benchmarks, and the Qwen-derived family, which leads on '
        'multilingual and tool-use benchmarks. Both families release model '
        'weights within roughly ninety days of the corresponding frontier-tier '
        'closed releases, a lag that has narrowed from approximately six '
        'months in 2024. The narrowing gap has forced closed-source labs to '
        'differentiate on operational dimensions — uptime, throughput, '
        'compliance certifications, and agentic infrastructure — rather than '
        'on raw capability alone.',
        style_body,
    ))

    story.append(Paragraph(
        'Enterprise preference has bifurcated along a clear line. Regulated '
        'industries — finance, healthcare, government — overwhelmingly prefer '
        'closed APIs for production workloads, citing audit trail, indemnity, '
        'and vendor accountability. Technology and consumer-internet companies '
        'increasingly default to fine-tuned open-weight models deployed on '
        'private infrastructure, citing cost control, latency, and the ability '
        'to retain weights behind their own perimeter. The table below '
        'summarizes the frontier-tier benchmark and pricing picture as of '
        'July 2026.',
        style_body,
    ))

    story.append(Spacer(1, 8))
    table_3 = make_table(
        [
            ['Model', 'MMLU-Pro', 'SWE-Bench Verified', 'Context', 'Price (per 1M out)'],
            ['GPT-5', '84.1', '72.4', '1M tokens', '$1.20'],
            ['Claude 4.5', '83.7', '74.1', '1M tokens', '$1.50'],
            ['Gemini 2.5 Ultra', '83.9', '70.8', '2M tokens', '$1.10'],
            ['DeepSeek V4', '82.4', '68.7', '1M tokens', '$0.35'],
            ['Qwen 3 Max', '82.0', '67.2', '1M tokens', '$0.40'],
            ['Llama 4 (open)', '79.6', '62.5', '512K tokens', 'Self-hosted'],
        ],
        col_ratios=[0.22, 0.13, 0.20, 0.18, 0.27],
    )
    story.append(table_3)
    story.append(Paragraph(
        'Table 2: Frontier-tier model comparison, July 2026. Benchmarks are '
        'reported by the respective labs; price reflects API list pricing.',
        style_caption,
    ))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════
    # Chapter 4: Enterprise Adoption & ROI
    # ═══════════════════════════════════════════════════════════════════════
    story.extend(add_major_section(
        'Chapter 4: Enterprise Adoption & ROI',
        kicker='Chapter 4 · Adoption & Returns',
    ))

    story.append(add_h2('4.1 Adoption Patterns by Function'))
    story.append(Paragraph(
        'Enterprise adoption in mid-2026 is no longer characterized by '
        'experimentation but by embedded production deployment. Software '
        'engineering leads the field with 78 percent of large enterprises '
        'running AI-assisted coding in production, typically as a combination '
        'of in-IDE code completion, pull-request review, and test generation. '
        'Customer support follows at 64 percent, where the dominant pattern is '
        'a tiered architecture in which AI handles first-line triage and '
        'resolution while escalating to humans for high-value or high-sensitivity '
        'cases. Marketing and content generation, knowledge retrieval, and '
        'sales and CRM complete the top five use cases.',
        style_body,
    ))

    story.append(Paragraph(
        'The functions with the lowest adoption rates — legal and compliance, '
        'HR and recruiting — are not lagging because of capability gaps but '
        'because of regulatory and reputational risk. Legal and compliance '
        'deployments are concentrated in contract analysis and regulatory '
        'monitoring, both of which require human review by law. HR deployments '
        'are constrained by anti-discrimination scrutiny, with several '
        'high-profile enforcement actions in 2025 having chilled aggressive '
        'rollout. The pattern suggests that adoption will accelerate as '
        'compliance frameworks mature rather than as model capabilities improve.',
        style_body,
    ))

    story.extend(chart_block(
        os.path.join(CHARTS_DIR, 'adoption.png'),
        'Figure 3: Enterprise AI adoption by use case, mid-2026. Percentage of '
        'large enterprises (>500 employees) with production deployments.',
        max_height=270,
    ))

    story.append(add_h2('4.2 ROI Measurement & Realized Value'))
    story.append(Paragraph(
        'The ROI picture has matured substantially since 2024, when most '
        'enterprise deployments were justified by soft productivity metrics. '
        'In H1 2026, the median large-enterprise deployment shows a payback '
        'period of 9.4 months and an annualized return of 3.7× on direct labor '
        'cost offsets. The strongest returns are concentrated in software '
        'engineering, where measured productivity gains of 25 to 40 percent on '
        'routine coding tasks translate directly into headcount avoidance or '
        'incremental feature throughput. Customer support deployments show '
        'lower but still robust returns, typically 2.4× to 3.1×, driven '
        'primarily by deflection rates of 35 to 55 percent on tier-one tickets.',
        style_body,
    ))

    story.append(Paragraph(
        'The failure modes have also crystallized. Roughly 22 percent of '
        'enterprise AI deployments in 2025-2026 have been significantly '
        'scaled back or shelved. The leading causes are: integration friction '
        'with legacy systems (38% of failures), inadequate evaluation '
        'infrastructure leading to silent quality drift (24%), organizational '
        'change management gaps (21%), and inability to demonstrate measurable '
        'returns to sponsors (17%). Notably, model capability itself is rarely '
        'the binding constraint on failure; the failures cluster in the '
        'integration and operational layers.',
        style_body,
    ))

    story.append(Spacer(1, 8))
    story.append(callout_row([
        ('9.4 mo', 'Median Payback Period'),
        ('3.7×', 'Median Annualized ROI'),
        ('22%', 'Deployment Failure Rate'),
    ]))
    story.append(Spacer(1, 10))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════
    # Chapter 5: AI Infrastructure & Compute Economics
    # ═══════════════════════════════════════════════════════════════════════
    story.extend(add_major_section(
        'Chapter 5: AI Infrastructure & Compute Economics',
        kicker='Chapter 5 · Compute & Capacity',
    ))

    story.append(add_h2('5.1 The Compute Stack'))
    story.append(Paragraph(
        'The AI compute stack in mid-2026 is a three-tier system. At the top '
        'sit the frontier accelerators — NVIDIA\'s H-series and B-series, '
        'AMD\'s MI400, and the first generation of Google TPU v7 and Amazon '
        'Trainium3. Below them sits a fast-growing tier of purpose-built '
        'inference accelerators, including the Groq LPU, Cerebras CS-5, and a '
        'wave of custom ASICs from cloud providers. At the bottom sits the '
        'networking and packaging layer: 800-gigabit and 1.6-terabit '
        'interconnects, advanced co-packaged optics, and the HBM4 memory that '
        'has become the binding supply constraint for the entire stack.',
        style_body,
    ))

    story.append(Paragraph(
        'The economics of the stack have shifted in important ways. Frontier '
        'training runs now cost between $80 million and $300 million in '
        'compute alone, with the largest runs approaching the billion-dollar '
        'threshold when amortized facility and staffing costs are included. '
        'Inference costs have moved in the opposite direction: the cost per '
        'token of output from frontier-class models has fallen by roughly 87 '
        'percent over the past eighteen months, driven by a combination of '
        'model architecture improvements, quantization, and the deployment of '
        'purpose-built inference silicon. The divergence between rising '
        'training costs and falling inference costs has compressed gross '
        'margins for frontier labs and shifted profitability toward '
        'operators with proprietary inference infrastructure.',
        style_body,
    ))

    story.append(add_h2('5.2 Energy & Geographic Constraints'))
    story.append(Paragraph(
        'Geographic concentration of AI compute has intensified rather than '
        'dispersed. The United States accounts for approximately 47 percent of '
        'global dedicated AI capacity, China 21 percent, the Middle East 11 '
        'percent, Europe 13 percent, and the rest of Asia 8 percent. The '
        'Middle East\'s emergence as the third-largest capacity bloc is the '
        'most significant geographic shift of the past eighteen months, driven '
        'by coordinated investment from sovereign wealth funds and the '
        'availability of low-cost natural gas and solar power. Within the '
        'United States, capacity is heavily concentrated in Virginia, Texas, '
        'and Arizona, with the Pacific Northwest losing share due to '
        'transmission constraints.',
        style_body,
    ))

    story.extend(chart_block(
        os.path.join(CHARTS_DIR, 'datacenter.png'),
        'Figure 4: AI datacenter capacity by region, 2024 — 2026H1. Nameplate '
        'capacity of dedicated AI facilities, in gigawatts.',
        max_height=260,
    ))

    story.append(Paragraph(
        'Energy has emerged as the second binding constraint after silicon. '
        'Grid interconnection queues in the United States now exceed five '
        'years for new large-load connections in many regions, and several '
        'operators have responded by signing direct power purchase agreements '
        'with nuclear operators — including the first restart of a previously '
        'decommissioned reactor in 2026 specifically for AI load. The table '
        'below summarizes the leading accelerators and their operational '
        'characteristics as of mid-2026.',
        style_body,
    ))

    story.append(Spacer(1, 8))
    table_5 = make_table(
        [
            ['Accelerator', 'Process', 'Memory (HBM)', 'Peak FP8', 'Typical Use'],
            ['NVIDIA B300', '3nm', '288 GB HBM3e', '14.4 PFLOPs', 'Frontier training & inference'],
            ['NVIDIA H210', '4nm', '141 GB HBM3e', '4.0 PFLOPs', 'Inference, prior-gen training'],
            ['AMD MI400', '3nm', '256 GB HBM3e', '10.8 PFLOPs', 'Training, mixed workloads'],
            ['Google TPU v7', '3nm', '192 GB HBM3e', '9.2 PFLOPs', 'Internal Gemini workloads'],
            ['Amazon Trainium3', '2nm', '192 GB HBM3e', '8.5 PFLOPs', 'Internal & Bedrock inference'],
            ['Groq LPU-2', '4nm', '128 GB SRAM', '1.8 PFLOPs', 'High-throughput inference'],
        ],
        col_ratios=[0.22, 0.11, 0.19, 0.16, 0.32],
    )
    story.append(table_5)
    story.append(Paragraph(
        'Table 3: Leading AI accelerators, July 2026. Specifications '
        'summarized from vendor disclosures; FP8 figures are dense peak.',
        style_caption,
    ))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════
    # Chapter 6: Regulation, Safety & Governance
    # ═══════════════════════════════════════════════════════════════════════
    story.extend(add_major_section(
        'Chapter 6: Regulation, Safety & Governance',
        kicker='Chapter 6 · Regulation & Safety',
    ))

    story.append(add_h2('6.1 Regulatory Landscape'))
    story.append(Paragraph(
        'The regulatory perimeter around AI hardened substantially in the '
        'first half of 2026. The EU AI Act entered its second enforcement '
        'phase on February 2, bringing obligations for high-risk systems into '
        'force and triggering the first formal investigations into non-'
        'compliant deployments. The United States issued binding model '
        'evaluation requirements in January for systems trained above the '
        '10<super>2</super><super>6</super> FLOP threshold, mandating pre-deployment red-teaming, '
        'capability disclosure, and incident reporting. China expanded its '
        'algorithm registry in March to cover generative systems used in '
        'public-facing roles, with security assessments now required prior to '
        'public release.',
        style_body,
    ))

    story.append(Paragraph(
        'The three regimes differ in mechanism but converge on substance. All '
        'three require some form of pre-deployment evaluation for the most '
        'capable systems, mandate transparency obligations for synthetic '
        'content, and impose obligations on deployers in high-stakes contexts '
        'such as employment, credit, and law enforcement. The practical effect '
        'has been a sharp increase in compliance costs for frontier-tier '
        'deployers — internal estimates suggest an additional 8 to 14 percent '
        'of model development budgets is now consumed by evaluation, '
        'documentation, and audit infrastructure. The cost has been absorbed '
        'by the largest labs but has further widened the moat around the '
        'frontier tier.',
        style_body,
    ))

    story.append(Spacer(1, 8))
    table_6 = make_table(
        [
            ['Region', 'Framework', 'Status (H1 2026)', 'Scope'],
            ['European Union', 'EU AI Act',
             'Phase 2 enforcement active',
             'Risk-tiered; high-risk systems (employment, credit, biometrics) face full obligations'],
            ['United States', 'Executive Order on AI Accountability',
             'Mandatory evaluations enforced',
             'Frontier systems above 10<super>2</super><super>6</super> FLOPs; pre-deployment red-team and disclosure'],
            ['China', 'Generative AI Measures + Algorithm Registry',
             'Expanded to public-facing GenAI',
             'Pre-release security assessment; content moderation requirements'],
            ['United Kingdom', 'AI Safety Institute evaluations',
             'Voluntary but universal',
             'All frontier-tier labs submit models for pre-deployment evaluation'],
            ['Japan & South Korea', 'AI Guidelines + sectoral rules',
             'Soft law with sector overlays',
             'Financial and healthcare deployments covered by sectoral regulation'],
        ],
        col_ratios=[0.18, 0.22, 0.22, 0.38],
    )
    story.append(table_6)
    story.append(Paragraph(
        'Table 4: Major AI regulatory regimes, mid-2026. Status reflects '
        'enforcement posture as of July 2026.',
        style_caption,
    ))

    story.append(add_h2('6.2 Safety & Alignment Practice'))
    story.append(Paragraph(
        'Safety practice has professionalized significantly. Frontier labs now '
        'maintain dedicated red-team organizations of 40 to 120 personnel, and '
        'pre-deployment evaluation suites have standardized around a common '
        'set of capabilities including cyber-offense, biological knowledge '
        'reasoning, persuasion, and autonomous task completion. The UK AI '
        'Safety Institute and its US counterpart have become de facto '
        'gatekeepers for frontier releases: all five frontier-tier systems '
        'released in 2026 underwent pre-deployment evaluation by at least one '
        'of these bodies, a marked contrast to 2024 when such evaluations '
        'were intermittent.',
        style_body,
    ))

    story.append(Paragraph(
        'Incident reporting has also matured. The leading labs now publish '
        'quarterly incident disclosures covering model behavior anomalies, '
        'jailbreaks discovered post-deployment, and significant unintended '
        'behaviors. The disclosures remain uneven in quality — a handful of '
        'labs provide detailed technical write-ups, while others issue '
        'high-level summaries — but the existence of the practice represents '
        'a meaningful step toward the kind of transparency regime that '
        'exists in aviation and pharmaceuticals. The open question for H2 '
        '2026 is whether incident disclosure will migrate from voluntary '
        'practice to regulatory mandate.',
        style_body,
    ))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════
    # Chapter 7: Emerging Verticals & Use Cases
    # ═══════════════════════════════════════════════════════════════════════
    story.extend(add_major_section(
        'Chapter 7: Emerging Verticals & Use Cases',
        kicker='Chapter 7 · Vertical Deep-Dives',
    ))

    story.append(Paragraph(
        'Beyond the horizontal adoption patterns documented in Chapter 4, '
        'three verticals have emerged as meaningful demonstrators of the '
        'technology\'s ability to reshape domain-specific workflows: '
        'healthcare and drug discovery, financial services, and software '
        'engineering. Each illustrates a different mode of value creation, '
        'and together they capture the majority of measurable AI-driven '
        'economic impact outside the infrastructure layer.',
        style_body,
    ))

    story.append(add_h2('7.1 Healthcare & Drug Discovery'))
    story.append(Paragraph(
        'Healthcare deployments have bifurcated into two distinct tracks. The '
        'first is clinical workflow automation — ambient documentation, '
        'prior-authorization processing, and clinical-trial matching — where '
        'AI systems reduce administrative burden without making clinical '
        'judgments. Adoption here is broad but shallow: 41 percent of US '
        'health systems have deployed at least one such tool in production. '
        'The second track is drug discovery, where the use of generative '
        'models for protein design, small-molecule optimization, and target '
        'identification has compressed preclinical timelines. The most '
        'consequential development of H1 2026 was the entry of three '
        'AI-designed drug candidates into Phase 2 clinical trials, marking '
        'the first time AI-originated molecules have reached this stage in '
        'meaningful numbers.',
        style_body,
    ))

    story.append(Spacer(1, 6))
    story.append(make_callout('3', 'AI-Designed Drug Candidates in Phase 2 Trials, H1 2026',
                              width=AVAILABLE_W * 0.6))
    story.append(Spacer(1, 12))

    story.append(add_h2('7.2 Financial Services'))
    story.append(Paragraph(
        'Financial services have moved aggressively from experimentation to '
        'embedded deployment. The dominant use cases are credit underwriting '
        'augmentation, fraud detection, and algorithmic trading research. '
        'The most consequential shift in H1 2026 is the migration of '
        'generative AI into customer-facing roles: several large retail banks '
        'have deployed AI systems for wealth advisory and mortgage '
        'consultation, with mandatory disclosure and human oversight. '
        'Regulatory scrutiny has followed — the US prudential regulators '
        'issued interagency guidance in April on model risk management for '
        'generative systems, formalizing expectations for governance, '
        'validation, and ongoing monitoring.',
        style_body,
    ))

    story.append(Paragraph(
        'Insurance has emerged as a particularly active sub-vertical. '
        'Generative models are now widely deployed for claims adjudication, '
        'underwriting triage, and policy language analysis, with several '
        'large carriers reporting 30 to 45 percent reductions in claims '
        'processing time. The deployments have not been without controversy: '
        'litigation over algorithmic denials in health insurance has '
        'increased, and several state regulators have opened investigations '
        'into the use of AI in coverage decisions.',
        style_body,
    ))

    story.append(add_h2('7.3 Software Engineering & Developer Tools'))
    story.append(Paragraph(
        'Software engineering remains the vertical with the deepest and most '
        'measurable AI impact. The dominant deployment pattern is the AI-'
        'assisted development environment: in-IDE code completion, pull-'
        'request review, test generation, and natural-language codebase '
        'navigation. Adoption has reached 78 percent of large enterprises, '
        'and the productivity gains have been measured in multiple '
        'rigorous studies at 25 to 40 percent on routine coding tasks. The '
        'consequential development of H1 2026 is the entry of agentic '
        'coding systems — autonomous agents capable of executing multi-step '
        'engineering tasks such as feature implementation, bug investigation, '
        'and codebase migration — into limited production at several large '
        'technology companies.',
        style_body,
    ))

    story.append(Spacer(1, 6))
    story.append(make_callout('+32%', 'Median Productivity Gain on Routine Coding Tasks (large enterprise)',
                              width=AVAILABLE_W * 0.6))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        'The agentic coding deployments remain early. The most credible '
        'deployments handle tasks with a median complexity of approximately '
        '90 minutes of human engineering time, with success rates in the 60 '
        'to 75 percent range as measured by human acceptance of the agent\'s '
        'output. The deployments have begun to reshape engineering '
        'organization charts: several large technology companies have '
        'reorganized to consolidate teams around agentic workflows, and the '
        'first wave of "agent-native" startups has emerged with engineering '
        'teams structured around human-agent collaboration rather than human-'
        'only development.',
        style_body,
    ))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════
    # Chapter 8: H2 2026 Outlook & Strategic Recommendations
    # ═══════════════════════════════════════════════════════════════════════
    story.extend(add_major_section(
        'Chapter 8: H2 2026 Outlook & Strategic Recommendations',
        kicker='Chapter 8 · Outlook & Recommendations',
    ))

    story.append(add_h2('8.1 H2 2026 Predictions'))
    story.append(Paragraph(
        'The second half of 2026 will be defined by the maturation of trends '
        'that emerged in the first half rather than by new discontinuities. '
        'The five predictions below are calibrated to high confidence based '
        'on observed trajectories in supply chains, regulatory pipelines, and '
        'enterprise procurement cycles.',
        style_body,
    ))

    story.append(bullet_list([
        '<b>Frontier model capability will plateau within 5 MMLU-Pro points of the current ceiling.</b> The era of large benchmark jumps is ending; differentiation will shift decisively to agentic reliability and operational economics.',
        '<b>The foundation-model lab layer will consolidate further.</b> Expect at least one major acquisition or merger among the second-tier western labs, and at least one closure or pivot among the third tier.',
        '<b>EU AI Act enforcement will produce its first major fine.</b> A high-profile enforcement action against a non-compliant high-risk deployment is likely before year-end, establishing the precedent that shapes 2027 compliance investment.',
        '<b>Compute supply will remain the binding constraint.</b> Lead times for next-generation accelerators will not improve materially before Q2 2027. Energy and grid interconnection will become the primary gating factor for new capacity in the United States and Europe.',
        '<b>Agentic AI will cross the 50-percent production threshold in software engineering.</b> Among large enterprises already using AI for coding, more than half will have at least one agentic workflow in production by year-end.',
    ]))

    story.append(add_h2('8.2 Strategic Recommendations'))
    story.append(Paragraph(
        'For enterprises, the strategic priority for H2 2026 is to convert '
        'diffuse AI experimentation into a smaller number of deeply embedded '
        'production workflows. The evidence from H1 2026 is unambiguous: the '
        'organizations realizing measurable returns are those that have '
        'concentrated investment in two or three high-leverage use cases '
        'rather than spreading budget across a broad portfolio of pilots. '
        'The recommended sequence is to identify the use case with the '
        'highest combination of measurable return and organizational '
        'readiness, build the evaluation infrastructure to detect quality '
        'drift, and invest in the integration and change management work '
        'that determines whether the deployment survives contact with real '
        'workflows.',
        style_body,
    ))

    story.append(Paragraph(
        'For investors, the most attractive risk-adjusted opportunities in '
        'H2 2026 sit in the infrastructure and middleware layers rather than '
        'in the foundation-model labs. The labs have largely completed their '
        'primary financing rounds at valuations that price in substantial '
        'execution risk, while the infrastructure layer — particularly '
        'purpose-built inference silicon, advanced packaging, and grid-'
        'connected datacenter capacity — continues to face supply-demand '
        'imbalances that support durable margins. The application layer '
        'presents a barbell: a small number of vertically integrated winners '
        'are emerging in each major vertical, while the broad middle of '
        'point-solution vendors faces margin compression as foundation models '
        'absorb adjacent capabilities.',
        style_body,
    ))

    story.append(Paragraph(
        'For policymakers, the priority for H2 2026 is to converge the '
        'divergent regulatory regimes toward mutual recognition of evaluation '
        'and disclosure outcomes. The current trajectory — in which a model '
        'must undergo separate evaluations in the EU, the UK, the US, and '
        'China before global deployment — imposes a meaningful tax on '
        'innovation without commensurate safety benefit. A framework of '
        'reciprocal recognition, anchored in shared technical standards for '
        'capability evaluation, would reduce compliance overhead while '
        'preserving each jurisdiction\'s ability to enforce its own substantive '
        'rules. The UK AI Safety Institute\'s emerging role as a neutral '
        'technical evaluator offers a template for how this might work in '
        'practice.',
        style_body,
    ))

    story.append(Paragraph(
        'The cumulative picture for H2 2026 is one of consolidation rather '
        'than disruption. The technology has reached a phase in which the '
        'binding constraints are operational and regulatory rather than '
        'scientific, and the organizations that will define the next phase '
        'are those that execute disciplined deployment, infrastructure '
        'investment, and governance design at scale. The frontier of '
        'possibility continues to advance, but the meaningful competitive '
        'ground has shifted to the question of who can turn capability into '
        'reliable, governed, economically productive deployment.',
        style_body,
    ))

    return story


# ─────────────────────────────────────────────────────────────────────────────
# Build PDF
# ─────────────────────────────────────────────────────────────────────────────
def build_body_pdf():
    doc = TocDocTemplate(
        BODY_PDF,
        pagesize=A4,
        leftMargin=LEFT_MARGIN, rightMargin=RIGHT_MARGIN,
        topMargin=TOP_MARGIN, bottomMargin=BOT_MARGIN,
        title='The AI Industry Landscape — Mid-Year 2026',
        author='Z.ai Research',
        creator='Z.ai',
        subject='Mid-year 2026 AI industry analysis',
    )
    story = build_story()
    doc.multiBuild(story, onFirstPage=page_header_footer,
                   onLaterPages=page_header_footer)
    print(f'  ✓ Body PDF generated: {BODY_PDF}')


def merge_cover_and_body():
    """Insert cover as page 0 of body PDF."""
    from pypdf import PdfReader, PdfWriter
    A4_W, A4_H = 595.28, 841.89

    def normalize(page):
        """Force every page to exact A4 dimensions to avoid sub-pixel
        mismatch between cover (Playwright) and body (ReportLab)."""
        box = page.mediabox
        w, h = float(box.width), float(box.height)
        if abs(w - A4_W) > 0.2 or abs(h - A4_H) > 0.2:
            page.scale_to(A4_W, A4_H)
        return page

    writer = PdfWriter()
    cover_page = PdfReader(COVER_PDF).pages[0]
    writer.add_page(normalize(cover_page))
    for p in PdfReader(BODY_PDF).pages:
        writer.add_page(normalize(p))
    writer.add_metadata({
        '/Title': 'The AI Industry Landscape — Mid-Year 2026',
        '/Author': 'Z.ai Research',
        '/Creator': 'Z.ai',
        '/Subject': 'Mid-year 2026 AI industry analysis',
    })
    with open(FINAL_PDF, 'wb') as f:
        writer.write(f)
    print(f'  ✓ Final PDF: {FINAL_PDF}')


if __name__ == '__main__':
    print('Building body PDF...')
    build_body_pdf()
    print('Merging cover + body...')
    merge_cover_and_body()
    print('Done.')
