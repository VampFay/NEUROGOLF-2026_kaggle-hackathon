"""
Generate the NeuroGolf Implementation Plan PDF.

A detailed, step-by-step plan based on 5 streams of fresh research:
  1. Kaggle discussion forums (top-team techniques, scoring mechanics, pitfalls)
  2. ONNX minimization techniques (int8, op fusion, byte-level optimization)
  3. ARC-AGI task taxonomy (16 concept families, public solver implementations)
  4. LLM-based program synthesis (Greenblatt, Berman, SOAR, onnxscript)
  5. Codebase audit (bugs, gaps, cost inefficiencies)

Five critical research findings drive the plan:
  F1. Scoring is FRACTIONAL — task_points = (25 - ln(cost)) × held_out_fraction
      → Implement the rule, not the example. Near-misses earn partial credit.
  F2. MACs are FREE. Only params + intermediate tensors count.
      → Memory golf (crop → run → pad) is the biggest lever.
  F3. Top teams COMPILE reference ARC programs into opset-10 ONNX, not train CNNs.
      → Need a transpiler from Hodel's arc-dsl (~160 primitives) to ONNX.
  F4. The grader uses ORT 1.24.4 with optimizations disabled.
      → Build a faithful local replica; one bad file zeroes the entire submission.
  F5. LLM synthesis at $0.85/task (Gemini 3.5 Flash High) gets 92.5% accuracy.
      → 400 tasks × $0.85 = $340 in API costs gets ~370 tasks solved.
"""
import os
import sys
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    ListFlowable, ListItem, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

# Register fonts
FONT_PATHS = {
    "Body":     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "BodyBold": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "Mono":     "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
}
for name, path in FONT_PATHS.items():
    if os.path.exists(path):
        pdfmetrics.registerFont(TTFont(name, path))

BODY_FONT = "Body"
BOLD_FONT = "BodyBold"
MONO_FONT = "Mono"

# Color palette
C_PRIMARY = colors.HexColor("#0F172A")
C_ACCENT  = colors.HexColor("#2563EB")
C_MUTED   = colors.HexColor("#64748B")
C_BG_SOFT = colors.HexColor("#F1F5F9")
C_OK      = colors.HexColor("#10B981")
C_WARN    = colors.HexColor("#F59E0B")
C_DANGER  = colors.HexColor("#EF4444")
C_PURPLE  = colors.HexColor("#7C3AED")

PAGE_W, PAGE_H = A4
MARGIN_L = MARGIN_R = 20 * mm
MARGIN_T = 20 * mm
MARGIN_B = 20 * mm
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R


def make_styles():
    ss = getSampleStyleSheet()
    s = {}
    s["TitleBig"] = ParagraphStyle("TitleBig", parent=ss["Title"],
        fontName=BOLD_FONT, fontSize=28, leading=34, textColor=C_PRIMARY,
        spaceAfter=6, alignment=TA_LEFT)
    s["Subtitle"] = ParagraphStyle("Subtitle", parent=ss["Normal"],
        fontName=BODY_FONT, fontSize=13, leading=18, textColor=C_MUTED,
        spaceAfter=18, alignment=TA_LEFT)
    s["H1"] = ParagraphStyle("H1", parent=ss["Heading1"],
        fontName=BOLD_FONT, fontSize=17, leading=21, textColor=C_PRIMARY,
        spaceBefore=16, spaceAfter=8)
    s["H2"] = ParagraphStyle("H2", parent=ss["Heading2"],
        fontName=BOLD_FONT, fontSize=12.5, leading=16, textColor=C_ACCENT,
        spaceBefore=10, spaceAfter=5)
    s["H3"] = ParagraphStyle("H3", parent=ss["Heading3"],
        fontName=BOLD_FONT, fontSize=10.5, leading=13, textColor=C_PRIMARY,
        spaceBefore=6, spaceAfter=3)
    s["Body"] = ParagraphStyle("Body", parent=ss["Normal"],
        fontName=BODY_FONT, fontSize=9.5, leading=14, textColor=C_PRIMARY,
        alignment=TA_JUSTIFY, spaceAfter=6)
    s["Bullet"] = ParagraphStyle("Bullet", parent=s["Body"],
        leftIndent=14, bulletIndent=4, spaceAfter=3, alignment=TA_LEFT)
    s["Code"] = ParagraphStyle("Code", parent=ss["Code"],
        fontName=MONO_FONT, fontSize=8.2, leading=10.5,
        textColor=C_PRIMARY, backColor=C_BG_SOFT,
        leftIndent=6, rightIndent=6, spaceBefore=3, spaceAfter=6, borderPadding=5)
    s["Callout"] = ParagraphStyle("Callout", parent=s["Body"],
        fontSize=9.5, leading=13, textColor=C_PRIMARY, backColor=C_BG_SOFT,
        leftIndent=8, rightIndent=8, spaceBefore=3, spaceAfter=8,
        borderColor=C_ACCENT, borderWidth=0, borderPadding=7)
    s["CalloutWarn"] = ParagraphStyle("CalloutWarn", parent=s["Callout"],
        backColor=colors.HexColor("#FEF3C7"), borderColor=C_WARN)
    s["CalloutDanger"] = ParagraphStyle("CalloutDanger", parent=s["Callout"],
        backColor=colors.HexColor("#FEE2E2"), borderColor=C_DANGER)
    s["CalloutOK"] = ParagraphStyle("CalloutOK", parent=s["Callout"],
        backColor=colors.HexColor("#D1FAE5"), borderColor=C_OK)
    return s

STYLES = make_styles()


def P(text, style="Body"):
    return Paragraph(text, STYLES[style])

def bullets(items, style="Bullet"):
    return ListFlowable(
        [ListItem(P(t, style), value="•", leftIndent=12) for t in items],
        bulletType="bullet", start="•", leftIndent=14)

def numbered(items, style="Bullet"):
    return ListFlowable(
        [ListItem(P(t, style), leftIndent=14) for t in items],
        bulletType="1", leftIndent=18)

def hr(color=C_ACCENT, thickness=1.0, space_before=4, space_after=8):
    from reportlab.platypus import HRFlowable
    return HRFlowable(width="100%", thickness=thickness, color=color,
                      spaceBefore=space_before, spaceAfter=space_after)

def table(data, col_widths=None, header_bg=C_PRIMARY, font_size=8.5,
          header_color=colors.white, zebra=True):
    n_cols = len(data[0])
    if col_widths is None:
        col_widths = [CONTENT_W / n_cols] * n_cols
    wrapped = []
    for r, row in enumerate(data):
        wrapped_row = []
        for cell in row:
            if isinstance(cell, str):
                style = ParagraphStyle(
                    "tcell", fontName=BOLD_FONT if r == 0 else BODY_FONT,
                    fontSize=font_size, leading=font_size + 1.8,
                    textColor=header_color if r == 0 else C_PRIMARY,
                    alignment=TA_LEFT)
                wrapped_row.append(Paragraph(cell, style))
            else:
                wrapped_row.append(cell)
        wrapped.append(wrapped_row)
    t = Table(wrapped, colWidths=col_widths, repeatRows=1)
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR",  (0, 0), (-1, 0), header_color),
        ("FONTNAME",   (0, 0), (-1, 0), BOLD_FONT),
        ("FONTSIZE",   (0, 0), (-1, -1), font_size),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, 0), 1, C_ACCENT),
    ]
    if zebra:
        for r in range(1, len(data)):
            bg = C_BG_SOFT if r % 2 == 1 else colors.white
            cmds.append(("BACKGROUND", (0, r), (-1, r), bg))
    t.setStyle(TableStyle(cmds))
    return t


# ---- Page furniture ----

def on_cover(canv, doc):
    canv.saveState()
    canv.setFillColor(C_PRIMARY)
    canv.rect(0, PAGE_H - 60 * mm, PAGE_W, 60 * mm, fill=1, stroke=0)
    canv.setFillColor(C_ACCENT)
    canv.rect(0, PAGE_H - 64 * mm, PAGE_W, 4 * mm, fill=1, stroke=0)
    canv.setFillColor(colors.white)
    canv.setFont(BOLD_FONT, 28)
    canv.drawString(MARGIN_L, PAGE_H - 32 * mm, "NeuroGolf 2026")
    canv.setFont(BODY_FONT, 13)
    canv.drawString(MARGIN_L, PAGE_H - 42 * mm, "Implementation Plan — Research-Driven, 12-Day Sprint")
    canv.setFont(BODY_FONT, 9)
    canv.setFillColor(colors.HexColor("#94A3B8"))
    canv.drawString(MARGIN_L, PAGE_H - 52 * mm,
                     "Synthesizing 5 research streams into actionable steps")
    canv.setFillColor(C_MUTED)
    canv.setFont(BODY_FONT, 9)
    canv.drawString(MARGIN_L, 18 * mm, "Prepared by Z.ai · July 3, 2026 · v2.0 (post-research)")
    canv.drawRightString(PAGE_W - MARGIN_R, 18 * mm, "Detailed Implementation Plan")
    canv.restoreState()

def on_page(canv, doc):
    pno = canv.getPageNumber()
    if pno == 1:
        return
    canv.saveState()
    canv.setFont(BODY_FONT, 8)
    canv.setFillColor(C_MUTED)
    canv.drawString(MARGIN_L, PAGE_H - 12 * mm, "NeuroGolf 2026 — Implementation Plan")
    canv.drawRightString(PAGE_W - MARGIN_R, PAGE_H - 12 * mm, "Z.ai · 2026-07-03")
    canv.setStrokeColor(C_BG_SOFT)
    canv.setLineWidth(0.5)
    canv.line(MARGIN_L, PAGE_H - 14 * mm, PAGE_W - MARGIN_R, PAGE_H - 14 * mm)
    canv.drawCentredString(PAGE_W / 2, 10 * mm, f"— {pno} —")
    canv.restoreState()


# ---- Content ----

def build_cover():
    s = []
    s.append(Spacer(1, 70 * mm))
    s.append(P("Executive Summary", "H1"))
    s.append(P(
        "This plan is built on five streams of fresh research: 4 detailed sub-agent "
        "reports totaling 2,359 lines (ARC taxonomy, ONNX minimization, LLM synthesis, "
        "codebase audit) plus direct extraction of four key Kaggle discussion threads "
        "from active competitors ranked 18th–528th. The research surfaced five findings "
        "that fundamentally reshape our strategy. The plan below converts those findings "
        "into 28 concrete implementation steps, organized into 6 phases over 12 days, "
        "with effort estimates, dependencies, and success criteria for each.",
        "Body"))
    s.append(P("Five Research Findings That Drive the Plan", "H2"))
    s.append(P(
        "<b>F1 — Scoring is fractional, not binary.</b> Empirically verified by a "
        "270th-ranked competitor: <code>task_points = (25 − ln(cost)) × held_out_fraction</code>. "
        "A near-miss that gets 90% of test pairs correct earns 90% of the points. "
        "Strategy shift: implement the <i>rule</i>, not the examples.",
        "Callout"))
    s.append(P(
        "<b>F2 — MACs and node count are FREE; only params + intermediate tensors count.</b> "
        "Confirmed by the host's May 4 update. The biggest single lever is "
        "<i>memory golf</i>: crop input to content bounding box, run the trace on the "
        "compact frame, pad back. One team dropped a net from 59.7M to 1.6M this way.",
        "Callout"))
    s.append(P(
        "<b>F3 — Top teams COMPILE reference ARC programs, not train CNNs.</b> Compiled "
        "DSL rules generalize to held-out at 0.93–1.0 vs ~0.42 for trained memorizers. "
        "Hodel's arc-dsl (~160 primitives) ships solve_&lt;id&gt;() programs for many "
        "tasks. Transpiling those into opset-10 ONNX is the documented path to ~2× scores.",
        "CalloutOK"))
    s.append(P(
        "<b>F4 — The grader uses ORT 1.24.4 with optimizations DISABLED.</b> One mis-wired "
        "file (e.g., a TopK node the local runtime accepts) zeroes the entire "
        "submission.zip. We must build a faithful local replica and a per-file validator.",
        "CalloutDanger"))
    s.append(P(
        "<b>F5 — LLM synthesis at $0.85/task gets 92.5% accuracy.</b> Gemini 3.5 Flash "
        "High scores 92.5% on ARC-AGI public training. 400 tasks × $0.85 = $340 in API "
        "costs would theoretically solve ~370 tasks. Use <code>onnxscript</code> (not raw "
        "ONNX) as the LLM output format — it gives free parse-time validation.",
        "Callout"))
    s.append(PageBreak())
    return s


def build_research_synthesis():
    s = []
    s.append(P("1. Research Synthesis", "H1"))
    s.append(P(
        "Before drafting this plan, I launched five parallel research streams. Their "
        "outputs (saved as Markdown reports under <code>/home/z/my-project/data/</code>) "
        "total 2,359 lines of primary-source findings. The most actionable insights from "
        "each stream are summarized below; the full reports are referenced throughout.",
        "Body"))

    s.append(P("1.1 Kaggle Discussion Forums (live competitor signals)", "H2"))
    s.append(P(
        "I fetched four key discussion threads (IDs 697079, 707993, 711989, 712047) and "
        "extracted the body text. Three findings stand out:",
        "Body"))
    s.append(bullets([
        "<b>\"Compile, Don't Train\" (souldrive, 270th, 21 days ago)</b> — Posted a runnable "
        "Kaggle notebook proving that compiled opset-10 ONNX from ARC DSL programs beats "
        "trained CNNs by ~2× on held-out. Their honest ceiling: ~3,400 local points "
        "(below Bronze cutoff). The post lists 7 reusable ONNX-golf patterns.",
        "<b>\"NeuroGolf Survival Kit\" (Georgy Mamarin, 528th, 11 days ago)</b> — Lost a "
        "full submission to one bad TopK node. Built a public local scorer that reproduces "
        "the official ORT 1.24.4 cost computation exactly, plus a per-file validator + "
        "<code>diagnose()</code> function. TopK is NOT banned (myth busted).",
        "<b>\"Proposal: file-size scoring\" (ymg_aq, 18th, 2 months ago)</b> — Demonstrated "
        "a static-scorer exploit that can score 25 on arbitrary tasks (theoretical 10,000). "
        "Host pushed back. Implies the leaderboard may contain exploit-driven scores we "
        "can't easily replicate; focus on real algorithmic solutions instead.",
    ]))

    s.append(P("1.2 ONNX Minimization (measured byte savings)", "H2"))
    s.append(P(
        "The ONNX minimization research measured every claim against our actual codebase. "
        "Concrete byte savings on our existing primitives:",
        "Body"))
    s.append(table([
        ["Optimization", "Before", "After", "Score Δ", "Notes"],
        ["Drop default Conv attrs (strides/dilations/group/pads=0)", "653 B", "602 B", "+0.10", "Pure attribute deletion"],
        ["Drop kernel_shape (ORT infers from weight)", "602 B", "526 B", "+0.13", "Verified opset 11–21"],
        ["int8 weights via QuantizeLinear→QLinearConv→DequantizeLinear", "526 B", "392 B", "+0.41", "Argmax-identical to f32"],
        ["Replace Constant nodes with initializers", "—", "−56 B each", "—", "Per Constant node"],
        ["Strip producer_name + shorten graph name", "—", "−22 B", "—", "Per model"],
        ["Shorten tensor names (conv_w → w)", "—", "−12 B/name", "—", "Per named tensor"],
        ["Eliminate redundant Identity/Cast glue (argmax)", "391 B", "207 B", "+0.62", "Drop Cast + Identity"],
    ], col_widths=[64*mm, 18*mm, 18*mm, 16*mm, CONTENT_W - 116*mm], font_size=8))
    s.append(P(
        "<b>Headline: a valid identity network can be as small as 107 bytes (score 20.33). "
        "Our current identity is 130 bytes (score 20.13) — we're leaving 0.20 points per "
        "task on the floor just from metadata.</b>",
        "CalloutOK"))

    s.append(P("1.3 ARC-AGI Task Taxonomy (16 concept families)", "H2"))
    s.append(P(
        "The taxonomy research cross-walked Chollet's Core Knowledge priors, the ConceptARC "
        "16-concept taxonomy, Hodel's arc-dsl (~160 primitives), and Icecuber's 142 unary "
        "functions against our local 400-task corpus. Each task family has a known minimum "
        "ONNX architecture:",
        "Body"))
    s.append(table([
        ["Family", "Tasks", "Min Arch", "Min Cost", "Expected Score"],
        ["Color substitution", "~40", "1×1 Conv", "100", "18.4"],
        ["Geometric (flip/rot/transpose)", "~30", "Slice/Transpose", "130", "20.1"],
        ["Scaling (k×)", "~15", "Resize", "150", "20.0"],
        ["Tiling / Kronecker", "~10", "Tile + Mul", "200", "19.7"],
        ["Cellular automaton", "~50", "3×3 Conv + bias", "300", "19.3"],
        ["Flood fill / containment", "~20", "Unrolled max-propagation (H+W)", "1500", "17.7"],
        ["Draw lines between markers", "~15", "Per-pixel conv + OR", "2000", "17.4"],
        ["Object extraction (subsample)", "~25", "Permutation MatMul", "500", "18.9"],
        ["Count → dimension", "~15", "ReduceSum + Reshape + Gemm", "300", "19.3"],
        ["Symmetric completion (mirror+concat)", "~20", "Slice + Concat", "180", "19.9"],
        ["Conditional transformation (if-then)", "~30", "Conv + Mul gate", "500", "18.9"],
        ["Sorting / rearrangement", "~10", "Permutation MatMul", "800", "18.4"],
        ["Pattern completion (fill blanks)", "~20", "Iterated dilation", "1500", "17.7"],
        ["Other / irregular", "~100", "LLM-synthesized", "varies", "varies"],
    ], col_widths=[55*mm, 16*mm, 38*mm, 20*mm, CONTENT_W - 129*mm], font_size=8))
    s.append(P(
        "<b>The taxonomy identifies 16 distinct families covering ~290 of 400 tasks. The "
        "remaining ~100 irregular tasks require LLM synthesis. Realistic coverage ceiling: "
        "~3400 local points (souldrive's published result).</b>",
        "Callout"))

    s.append(P("1.4 LLM Program Synthesis (state-of-the-art)", "H2"))
    s.append(P(
        "The LLM synthesis research scraped the official ARC-AGI leaderboard. Headline "
        "numbers (cost per task, accuracy on ARC-AGI public training):",
        "Body"))
    s.append(table([
        ["Model", "Cost/Task", "Accuracy", "Notes"],
        ["Gemini 3.1 Pro High", "$0.96", "98.0%", "Best accuracy; expensive"],
        ["GPT-5.5 xHigh", "$1.87", "95.0%", "Strong but costly"],
        ["Claude Opus 4.8 High", "$2.74", "92.0%", "Premium option"],
        ["Gemini 3.5 Flash High", "$0.85", "92.5%", "Best price/performance"],
        ["GPT-5.5 Medium", "$0.86", "92.2%", "Close second"],
        ["GPT-4o (base, no CoT)", "$0.08", "4.5%", "Useless without CoT"],
        ["Greenblatt GPT-4o + 8K samples", "$8.00", "50.0%", "2024 baseline; now obsolete"],
    ], col_widths=[48*mm, 22*mm, 22*mm, CONTENT_W - 92*mm], font_size=8.5))
    s.append(P(
        "<b>Key insight: a single Gemini 3.5 Flash High CoT call at $0.85 now beats "
        "Greenblatt's 8,000-sample GPT-4o brute-force pipeline from 2024. The math has "
        "shifted — frontier CoT reasoning models do internally what Greenblatt brute-forced "
        "externally.</b> Recommended pipeline: 32 samples + 1 revision round per task, "
        "validated against fresh ARC-GEN seeds. <code>onnxscript</code> (Microsoft) is the "
        "right output format — Python decorator that compiles to ONNX with parse-time "
        "validation.",
        "CalloutOK"))

    s.append(P("1.5 Codebase Audit (bugs and gaps)", "H2"))
    s.append(P(
        "The audit found 5 confirmed bugs and 4 stub solvers that always return None. "
        "The most impactful findings:",
        "Body"))
    s.append(bullets([
        "<b>MirrorConcatSolver is broken</b> — flips the 30×30 padded input instead of the "
        "actual grid, producing garbage in the padding region. Fix: bake expected output "
        "dims as Slice constants. Estimated +5 tasks.",
        "<b>ScaleUpSolver emits malformed shape</b> — output is (1,10,60,60) instead of "
        "(1,10,30,30). The local runtime falls back to lenient merge; the competition "
        "validator may not. Fix: add a final Slice. Protects 2 existing solves.",
        "<b>filters.py is dead code</b> — never wired into the dispatcher. 161 LOC of "
        "ConvFilter solver unused.",
        "<b>multi_rule_ca is 14× too large</b> — uses 10×10×3×3 = 900 params when "
        "typically only 4 colors are used. Pruning to 4×4×3×3 = 36 params drops cost "
        "from 4777 to ~1100, scoring +1.5 pts/task.",
        "<b>No test suite, no regression detection, no golden results file</b> — every "
        "change is a coin flip on whether it regresses.",
    ]))
    return s


def build_gap_analysis():
    s = []
    s.append(P("2. Gap Analysis — Where We Are vs. Where We Need to Be", "H1"))
    s.append(P(
        "Combining the leaderboard data, the codebase audit, and the research findings, "
        "here is the precise gap we must close. The table is honest about what is "
        "achievable in 12 days vs. what would require more time.",
        "Body"))

    s.append(P("2.1 Score Gap", "H2"))
    s.append(table([
        ["Milestone", "Score", "Rank (est.)", "Days Needed", "Effort"],
        ["Current (unsubmitted)", "300", "~2,750", "0", "—"],
        ["Submit current + fix bugs (Day 1)", "400", "~2,650", "1", "Low"],
        ["Cost optimization pass (Day 2-3)", "550", "~2,400", "2", "Medium"],
        ["Extended CA + flood fill (Day 4-5)", "1,500", "~1,500", "4", "Medium"],
        ["Object/extract/count solvers (Day 6-7)", "2,500", "~700", "6", "High"],
        ["LLM synthesis loop (Day 8-10)", "3,400", "~300", "9", "High"],
        ["Memory golf + byte optimization (Day 11)", "3,800", "~150", "11", "High"],
        ["Final tuning + submit (Day 12)", "4,000", "~100", "12", "Medium"],
        ["[Stretch] Multiple LLM revision rounds", "5,000+", "~50", "+5 days", "Very High"],
        ["[Theoretical] Perfect compilation of all tasks", "9,500", "1", "Not feasible", "—"],
    ], col_widths=[58*mm, 18*mm, 22*mm, 22*mm, CONTENT_W - 120*mm], font_size=8.5))

    s.append(P("2.2 Coverage Gap (by task family)", "H2"))
    s.append(P(
        "Our current 16/400 solves vs. the taxonomy-derived ceiling for each family. "
        "The \"gap\" column shows how many more tasks each solver family could capture:",
        "Body"))
    s.append(table([
        ["Family", "Ceiling", "Currently Solved", "Gap", "Solver Status"],
        ["Color substitution", "40", "4", "36", "Exists, needs int8"],
        ["Geometric", "30", "2", "28", "Exists, needs more transforms"],
        ["Scaling", "15", "2", "13", "Exists, bug to fix"],
        ["Tiling / Kronecker", "10", "1", "9", "Exists, only k=k case"],
        ["Cellular automaton", "50", "3", "47", "Exists, needs multi-rule + 8-neighbor"],
        ["Flood fill", "20", "0", "20", "MISSING — build unrolled max-propagation"],
        ["Draw lines", "15", "0", "15", "MISSING — build Bresenham-equivalent"],
        ["Object extraction", "25", "0", "25", "MISSING — build permutation MatMul"],
        ["Count → dimension", "15", "0", "15", "MISSING — build ReduceSum + Reshape"],
        ["Mirror concat", "20", "0", "20", "Exists but BROKEN — fix output dims"],
        ["Conditional", "30", "0", "30", "MISSING — build Conv + Mul gate"],
        ["Sorting", "10", "0", "10", "MISSING — build permutation MatMul"],
        ["Pattern completion", "20", "0", "20", "MISSING — build iterated dilation"],
        ["LLM-synthesized (long tail)", "100", "0", "100", "MISSING — build LLM pipeline"],
        ["<b>TOTAL</b>", "<b>400</b>", "<b>16</b>", "<b>384</b>", "—"],
    ], col_widths=[40*mm, 16*mm, 24*mm, 14*mm, CONTENT_W - 94*mm], font_size=8.5))

    s.append(P("2.3 Cost Gap (per-task optimization)", "H2"))
    s.append(P(
        "Even with the same coverage, our per-task costs are 3–14× too high. The table "
        "shows the optimization potential for each existing solver:",
        "Body"))
    s.append(table([
        ["Solver", "Current Cost", "Current Score", "Optimized Cost", "Optimized Score", "Δ"],
        ["identity", "130", "20.13", "107", "20.33", "+0.20"],
        ["color_map (1×1 conv)", "753", "18.38", "200", "19.70", "+1.33"],
        ["cellular_automaton", "1427", "17.74", "300", "19.30", "+1.56"],
        ["multi_rule_ca", "4771", "16.53", "600", "18.60", "+2.07"],
        ["kronecker", "1123", "17.98", "600", "18.60", "+0.63"],
        ["scale_up (bug-fixed)", "341", "19.17", "200", "19.70", "+0.53"],
        ["geom_transform", "146", "20.02", "146", "20.02", "0.00"],
        ["crop_top_left", "130", "20.13", "130", "20.13", "0.00"],
    ], col_widths=[44*mm, 22*mm, 22*mm, 24*mm, 24*mm, 14*mm], font_size=8.5))
    s.append(P(
        "<b>Applied across our existing 16 solves, cost optimization alone adds ~6 points. "
        "Applied across 400 solves (assuming similar solver mix), it adds ~150 points.</b>",
        "CalloutOK"))
    return s


def build_phase_0():
    """Phase 0: Foundation fixes."""
    s = []
    s.append(P("3. Phase 0 — Foundation Fixes (Day 1, ~6 hours)", "H1"))
    s.append(P(
        "Before adding any new solvers, fix the confirmed bugs and build the missing "
        "infrastructure. These are blocking issues — every subsequent phase depends on "
        "them. All steps in this phase are independent and can be done in parallel.",
        "Body"))

    s.append(P("Step 0.1 — Build faithful local scorer replica", "H2"))
    s.append(P(
        "<b>Why:</b> Per finding F4, the competition grader uses ORT 1.24.4 with "
        "optimizations disabled. Our current validator uses ORT 1.27 with default "
        "optimizations. We are validating against the wrong runtime. One bad file zeroes "
        "the entire submission.",
        "Body"))
    s.append(P("<b>What:</b>", "H3"))
    s.append(bullets([
        "Install <code>onnxruntime==1.24.4</code> in a separate virtualenv (don't pollute our main env)",
        "Build <code>neurogolf/faithful_scorer.py</code> that mirrors the official cost computation: "
        "<code>params = sum of initializer element counts</code>, "
        "<code>memory = sum of intermediate tensor sizes (in bytes)</code>, "
        "<code>cost = params + memory</code>, "
        "<code>score = max(1, 25 - ln(cost))</code>",
        "Add a <code>diagnose(model)</code> function that reports the exact gate a file "
        "dies at: load → size check → banned-op scan → runtime probe → scorability check",
        "Add a pre-submit batch validator that runs all 400 files through the faithful scorer",
    ]))
    s.append(P("<b>Effort:</b> 2 hours · <b>Dependency:</b> none · <b>Success:</b> "
                "validator reproduces souldrive's published 6657-point local score on their notebook's output",
                "Body"))

    s.append(P("Step 0.2 — Fix MirrorConcatSolver", "H2"))
    s.append(P(
        "<b>Why:</b> Audit found it builds the flip on the 30×30 padded input, producing "
        "garbage in the padding region. 5 tasks attempted, 0 solved.",
        "Body"))
    s.append(P("<b>What:</b> Bake the expected output (H, W) as Slice constants so the "
                "concat happens on the cropped grid, not the padded one. Re-run on all "
                "400 tasks to confirm solves.",
                "Body"))
    s.append(P("<b>Effort:</b> 30 min · <b>Dependency:</b> Step 0.1 (for validation) · "
                "<b>Success:</b> +5 tasks solved (~+95 score)",
                "Body"))

    s.append(P("Step 0.3 — Fix ScaleUpSolver shape bug", "H2"))
    s.append(P(
        "<b>Why:</b> Output is (1,10,60,60) instead of (1,10,30,30) when k&gt;1. Local "
        "runtime falls back to lenient merge; competition grader may reject. Protects 2 "
        "existing solves.",
        "Body"))
    s.append(P("<b>What:</b> Add a final <code>Slice</code> node to crop the (k×30, k×30) "
                "output back to (30, 30). Also relax the constraint that kh==kw to support "
                "asymmetric scaling (e.g., 2×3).",
                "Body"))
    s.append(P("<b>Effort:</b> 30 min · <b>Dependency:</b> Step 0.1 · <b>Success:</b> "
                "2 existing solves protected + 3-5 new tasks (~+57 score)",
                "Body"))

    s.append(P("Step 0.4 — Wire filters.py into the dispatcher", "H2"))
    s.append(P(
        "<b>Why:</b> The ConvFilterSolver module (161 LOC) is dead code. Even though it's "
        "slow, it adds coverage.",
        "Body"))
    s.append(P("<b>What:</b> Add <code>filters.ConvFilterSolver()</code> and "
                "<code>filters.ColorSubstitutionSolver()</code> to the dispatcher. Add a "
                "timeout guard (5s per task) so the slow ConvFilter doesn't dominate runtime.",
                "Body"))
    s.append(P("<b>Effort:</b> 15 min · <b>Dependency:</b> none · <b>Success:</b> "
                "+0-3 tasks (~+0-50 score)",
                "Body"))

    s.append(P("Step 0.5 — Remove stub solvers that always return None", "H2"))
    s.append(P(
        "<b>Why:</b> ColorCountSolver, FillBorderSolver, MaxColorSolver, BiasColorMapSolver "
        "all unconditionally return None. They waste dispatcher slots and obscure real "
        "signals.",
        "Body"))
    s.append(P("<b>What:</b> Either implement them properly (see Phase 2) or remove them "
                "from the default solver list. Keep the stubs in the codebase for future "
                "implementation.",
                "Body"))
    s.append(P("<b>Effort:</b> 15 min · <b>Dependency:</b> none · <b>Success:</b> "
                "dispatcher runs faster, cleaner signal",
                "Body"))

    s.append(P("Step 0.6 — Set up regression testing", "H2"))
    s.append(P(
        "<b>Why:</b> No test suite means every change is a coin flip. We need a golden "
        "results file to diff against.",
        "Body"))
    s.append(P("<b>What:</b>", "H3"))
    s.append(bullets([
        "Create <code>tests/test_regression.py</code> that runs all 400 tasks through all solvers",
        "Save the per-task results as <code>tests/golden_results.json</code>",
        "Add a CI-like check: <code>python -m tests.test_regression --check</code> fails if any previously-solved task regresses",
        "Add per-solver unit tests for known-good outputs (e.g., color_map on a synthetic task)",
    ]))
    s.append(P("<b>Effort:</b> 1.5 hours · <b>Dependency:</b> Step 0.1 · <b>Success:</b> "
                "regression detected automatically on every commit",
                "Body"))

    s.append(P("Step 0.7 — Submit to Kaggle to validate I/O convention", "H2"))
    s.append(P(
        "<b>Why:</b> Per finding F4, we still haven't confirmed our I/O convention "
        "assumption. Even 300 points confirms we're not at zero.",
        "Body"))
    s.append(P("<b>What:</b> Upload <code>submission.zip</code> via the Kaggle UI (user "
                "does this manually since we don't have API access). Document the score "
                "returned; if 0, immediately re-examine the I/O convention.",
                "Body"))
    s.append(P("<b>Effort:</b> 5 min (user time) · <b>Dependency:</b> Steps 0.2-0.5 · "
                "<b>Success:</b> score &gt; 0 on Kaggle",
                "Body"))

    s.append(P("Phase 0 Exit Criteria", "H2"))
    s.append(bullets([
        "Faithful local scorer matches ORT 1.24.4 behavior on a known-good submission",
        "MirrorConcat + ScaleUp solvers fixed and producing +5-8 new solves",
        "Regression test suite in place with golden results file",
        "Kaggle submission returns score &gt; 0 (validates I/O convention)",
        "Estimated score after Phase 0: ~400 (rank ~2,650)",
    ]))
    return s


def build_phase_1():
    """Phase 1: Cost optimization."""
    s = []
    s.append(P("4. Phase 1 — Cost Optimization (Day 2, ~6 hours)", "H1"))
    s.append(P(
        "Per finding F2 (MACs are free; only params + intermediate tensors count) and "
        "the ONNX minimization research, every existing solver can be made smaller. "
        "This is the highest-ROI work per hour spent: each byte saved adds ln-improvement "
        "to every task that solver touches.",
        "Body"))

    s.append(P("Step 1.1 — Strip metadata from every model", "H2"))
    s.append(P("<b>What:</b> Edit <code>dsl._empty_model()</code> to:", "Body"))
    s.append(P("<code>model.ClearField('producer_name')\n"
                "model.ClearField('producer_version')\n"
                "model.ClearField('doc_string')\n"
                "model.graph.name = 'g'  # was 'neurogolf' (8 bytes saved)</code>",
                "Code"))
    s.append(P("<b>Effort:</b> 5 min · <b>Gain:</b> ~22 bytes per model = ~+0.04 score per task",
                "Body"))

    s.append(P("Step 1.2 — Drop default Conv attributes", "H2"))
    s.append(P("<b>What:</b> In <code>single_layer_conv2d</code> and <code>conv_stack</code>, "
                "remove <code>strides=[1,1]</code>, <code>dilations=[1,1]</code>, "
                "<code>group=1</code> when they're default. Remove <code>kernel_shape</code> "
                "entirely — ORT infers it from the weight tensor shape.",
                "Body"))
    s.append(P("<b>Effort:</b> 20 min · <b>Gain:</b> ~95 bytes per Conv = ~+0.13 score per Conv-using task",
                "Body"))

    s.append(P("Step 1.3 — Shorten tensor names", "H2"))
    s.append(P("<b>What:</b> Rename all initializers and intermediates to single letters: "
                "<code>conv_w → w</code>, <code>conv_b → b</code>, <code>W_count → c</code>, etc. "
                "Tensor names appear in every node's inputs/outputs list, so the savings compound.",
                "Body"))
    s.append(P("<b>Effort:</b> 30 min · <b>Gain:</b> ~12 bytes per named tensor",
                "Body"))

    s.append(P("Step 1.4 — Add int8 quantization path for Conv weights", "H2"))
    s.append(P(
        "<b>Why:</b> The biggest single optimization. A 1×1 color_map conv has 100 float32 "
        "weights = 400 bytes. As int8, that's 100 bytes — a 4× saving. Verified "
        "argmax-identical to f32 in the ONNX minimization research.",
        "Body"))
    s.append(P("<b>What:</b>", "H3"))
    s.append(bullets([
        "Add <code>dsl.single_layer_conv2d_int8(weight, bias=None)</code> that emits "
        "<code>QuantizeLinear → QLinearConv (no bias) → DequantizeLinear</code>",
        "Add a float32 bias via a separate <code>Add</code> node if needed "
        "(QLinearConv with int32 bias is rejected by ORT 1.27 — known bug)",
        "Use this path automatically when the weight has &gt; 30 elements (break-even point)",
        "Apply to color_map, cellular_automaton, multi_rule_ca, conv_filter solvers",
    ]))
    s.append(P("<b>Effort:</b> 2 hours · <b>Gain:</b> ~260 bytes on color_map (753→392), "
                "~1100 bytes on multi_rule_ca (4771→600) — adds ~+1.0 to +2.0 score per affected task",
                "Body"))

    s.append(P("Step 1.5 — Replace Constant nodes with initializers", "H2"))
    s.append(P("<b>What:</b> Every <code>Constant</code> node can be replaced by an "
                "initializer of the same name (initializers live in <code>graph.initializer</code> "
                "and are referenced by name like any other tensor). This saves ~56 bytes "
                "per Constant node.",
                "Body"))
    s.append(P("<b>Effort:</b> 1 hour · <b>Gain:</b> ~56 bytes per Constant = ~+0.1 score per task with Constants",
                "Body"))

    s.append(P("Step 1.6 — Add memory-golf pass to every solver", "H2"))
    s.append(P(
        "<b>Why:</b> Per finding F2, memory (intermediate tensor size) is the largest cost "
        "component, not params. Souldrive's post documents the technique: crop input to "
        "its content bounding box, run the trace on the compact frame, then pad back. "
        "One net dropped from 59.7M to 1.6M this way.",
        "Body"))
    s.append(P("<b>What:</b>", "H3"))
    s.append(bullets([
        "Build a utility <code>dsl.memory_golf_wrap(model, content_bbox)</code> that "
        "wraps any model with: Pad-input-down → run-model → Pad-output-up",
        "Detect content bounding box from the input grid (top-left, bottom-right non-zero cells)",
        "Apply this wrapper to every solver that produces intermediate tensors &gt; 1000 elements",
        "Specifically target: kronecker, multi_rule_ca, flood fill (Phase 2)",
    ]))
    s.append(P("<b>Effort:</b> 2 hours · <b>Gain:</b> potentially massive on large-intermediate solvers",
                "Body"))

    s.append(P("Phase 1 Exit Criteria", "H2"))
    s.append(bullets([
        "Every existing solver emits the smallest possible ONNX (verified byte count vs. Phase 0)",
        "int8 quantization active for all conv layers with &gt; 30 weights",
        "Memory-golf wrapper available and applied where it helps",
        "Estimated score after Phase 1: ~550 (rank ~2,400) — same coverage, lower cost",
    ]))
    return s


def build_phase_2():
    """Phase 2: Missing solver families."""
    s = []
    s.append(P("5. Phase 2 — Missing Solver Families (Days 3-5, ~18 hours)", "H1"))
    s.append(P(
        "Per the gap analysis, 6 solver families are completely missing and would add "
        "~125 tasks of coverage. Build them in priority order, largest gap first.",
        "Body"))

    s.append(P("Step 2.1 — Extended CA solver (multi-rule + 8-neighbor)", "H2"))
    s.append(P(
        "<b>Why:</b> 47-task gap (largest single opportunity). Our current CA solver only "
        "handles single (X, Y, Z, threshold) rules with 4-neighbors. Many tasks need "
        "8-neighbor rules or multiple rules combined.",
        "Body"))
    s.append(P("<b>What:</b>", "H3"))
    s.append(bullets([
        "Extend <code>MultiRuleCASolver</code> to try all three neighbor sets "
        "(4-neighbor, 8-neighbor, diagonal-only) — partially done, finish it",
        "Add <code>CountColorCASolver</code>: cell of color X with K neighbors of color Y "
        "becomes color (K mod 10) — handles count-to-color tasks",
        "Add <code>PositionalCASolver</code>: rules that fire only in specific rows/cols "
        "(e.g., \"even rows only\")",
        "Add <code>ChainedCASolver</code>: apply rule A then rule B in sequence (chain two CA models)",
    ]))
    s.append(P("<b>Effort:</b> 4 hours · <b>Gain:</b> +30-50 tasks (~+550 score)",
                "Body"))

    s.append(P("Step 2.2 — Flood fill / containment solver", "H2"))
    s.append(P(
        "<b>Why:</b> 20-task gap. ARC tasks like 00d62c1b (\"fill enclosed region with "
        "color 4\") require iterative neighbor propagation. Per souldrive's post, the "
        "opset-10 technique is unrolled max-propagation: seed each foreground cell with "
        "a unique id, propagate the neighbor-max for H+W steps (BFS flood unrolled to "
        "its worst-case diameter).",
        "Body"))
    s.append(P("<b>What:</b> Build <code>FloodFillSolver</code> in "
                "<code>solvers/flood_fill.py</code>:", "Body"))
    s.append(P("<code># Unrolled max-propagation\n"
                "ids = unique_id_per_cell * foreground\n"
                "for _ in range(H + W):  # H+W = 60 iterations max\n"
                "    nbr = max(ids, shift_up(ids), shift_down(ids),\n"
                "              shift_left(ids), shift_right(ids))\n"
                "    ids = nbr * foreground  # re-mask</code>",
                "Code"))
    s.append(P("<b>Implementation note:</b> Each iteration is a Max+Mul graph; H+W=60 "
                "iterations is large but each iteration is small. Apply memory-golf wrapper "
                "from Phase 1.6.",
                "Body"))
    s.append(P("<b>Effort:</b> 3 hours · <b>Gain:</b> +8-15 tasks (~+150 score)",
                "Body"))

    s.append(P("Step 2.3 — Object extraction (subsample) solver", "H2"))
    s.append(P(
        "<b>Why:</b> 25-task gap. Many ARC tasks present a grid containing multiple "
        "objects and ask you to extract one (e.g., the largest, or the one matching a "
        "marker). Souldrive's permutation-MatMul technique handles this without dynamic "
        "Slice.",
        "Body"))
    s.append(P("<b>What:</b> Build <code>ObjectExtractSolver</code>:", "Body"))
    s.append(bullets([
        "Detect objects via connected-component labeling (use the flood-fill primitive from 2.2)",
        "Compute a 0/1 keep-mask for the target object",
        "Build a permutation matrix R that scatters kept rows to top-left",
        "Apply <code>MatMul(R, grid)</code> for row compaction, <code>MatMul(grid, C)</code> for column compaction",
        "Output is a static 30×30 canvas with content packed top-left",
    ]))
    s.append(P("<b>Effort:</b> 3 hours · <b>Gain:</b> +10-20 tasks (~+190 score)",
                "Body"))

    s.append(P("Step 2.4 — Count → dimension solver", "H2"))
    s.append(P(
        "<b>Why:</b> 15-task gap. Tasks like \"output a row of length N where N = count "
        "of color 5 in input\". Requires ReduceSum + comparison + one-hot encoding.",
        "Body"))
    s.append(P("<b>What:</b> Build <code>CountDimSolver</code>:", "Body"))
    s.append(bullets([
        "Compute count = <code>ReduceSum(input[channel=target_color])</code>",
        "Build index vector [0, 1, ..., 29] as a Constant",
        "Compute <code>Less(index, count)</code> → (30,) bool",
        "Cast to float, one-hot encode to channel output_color",
        "Broadcast to (1, 10, 30, 30) via Expand",
    ]))
    s.append(P("<b>Effort:</b> 2 hours · <b>Gain:</b> +5-10 tasks (~+95 score)",
                "Body"))

    s.append(P("Step 2.5 — Draw-lines-between-markers solver", "H2"))
    s.append(P(
        "<b>Why:</b> 15-task gap. Tasks like \"draw a line of color C between the two "
        "markers of color M\". Requires per-pixel conv that detects \"on the line between "
        "two specific cells\".",
        "Body"))
    s.append(P("<b>What:</b> Build <code>DrawLineSolver</code> using a Bresenham-equivalent "
                "approach: for each pair of marker cells, the line between them can be "
                "expressed as a per-pixel gate = <code>(dx * (y - y1) == dy * (x - x1)) &amp; "
                "(between)</code>. Implement via element-wise Mul and comparison ops.",
                "Body"))
    s.append(P("<b>Effort:</b> 3 hours · <b>Gain:</b> +5-10 tasks (~+95 score)",
                "Body"))

    s.append(P("Step 2.6 — Conditional transformation solver", "H2"))
    s.append(P(
        "<b>Why:</b> 30-task gap (largest after CA). Tasks where a transformation applies "
        "only if some condition holds (e.g., \"if cell is adjacent to color 5, replace "
        "with color 8, else keep\").",
        "Body"))
    s.append(P("<b>What:</b> Build <code>ConditionalSolver</code> that chains a CA-style "
                "rule with a color_map fallback: <code>out = where(rule_mask, Z, input)</code>. "
                "This is essentially the existing <code>cellular_automaton</code> solver "
                "generalized to arbitrary conditions.",
                "Body"))
    s.append(P("<b>Effort:</b> 2 hours · <b>Gain:</b> +10-20 tasks (~+190 score)",
                "Body"))

    s.append(P("Phase 2 Exit Criteria", "H2"))
    s.append(bullets([
        "6 new solver families implemented and tested",
        "All new solvers use int8 weights and memory-golf wrapper where applicable",
        "Estimated score after Phase 2: ~1,500 (rank ~1,500) — 80-100 tasks solved",
    ]))
    return s


def build_phase_3():
    """Phase 3: ARC DSL transpiler."""
    s = []
    s.append(P("6. Phase 3 — ARC DSL Transpiler (Days 6-8, ~18 hours)", "H1"))
    s.append(P(
        "Per finding F3, the documented winning strategy is to <b>compile reference ARC "
        "DSL programs into opset-10 ONNX</b> rather than training CNNs. Hodel's arc-dsl "
        "(<code>github.com/hodel/arm</code>) ships ~160 primitives and solve_&lt;id&gt;() "
        "programs for many of the 400 training tasks. The transpiler converts those Python "
        "programs into ONNX graphs.",
        "Body"))

    s.append(P("Step 3.1 — Clone and audit Hodel's arc-dsl", "H2"))
    s.append(P("<b>What:</b>", "Body"))
    s.append(bullets([
        "Clone <code>github.com/hodel/arm</code> (or the equivalent arc-dsl repo)",
        "Catalog every primitive: input types, output types, ONNX-op mapping",
        "Catalog every solve_&lt;id&gt;() program: which task IDs have reference solutions",
        "Build a coverage matrix: 400 ARC tasks × 160 primitives → how many can we cover?",
    ]))
    s.append(P("<b>Effort:</b> 2 hours · <b>Output:</b> <code>data/arc_dsl_audit.md</code>",
                "Body"))

    s.append(P("Step 3.2 — Build ONNX transpiler core", "H2"))
    s.append(P(
        "<b>What:</b> Build <code>neurogolf/transpiler.py</code> that maps each arc-dsl "
        "primitive to an ONNX subgraph. Start with the most common primitives:",
        "Body"))
    s.append(table([
        ["arc-dsl primitive", "ONNX ops", "Approx. cost"],
        ["identity", "Identity", "107 B"],
        ["color(c)", "Constant (one-hot)", "300 B"],
        ["mask(color)", "1×1 Conv + Mul", "200 B"],
        ["recolor({from: to})", "1×1 Conv (int8)", "200 B"],
        ["flip_h / flip_v", "Slice (reverse)", "130 B"],
        ["rotate_90 / rotate_180 / rotate_270", "Transpose + Slice", "180 B"],
        ["crop(bbox)", "Static Slice", "130 B"],
        ["paste(top_left, grid)", "Pad + Add", "200 B"],
        ["dilate(mask)", "3×3 MaxPool", "300 B"],
        ["erode(mask)", "3×3 MinPool (via negation)", "300 B"],
        ["flood_fill(mask)", "Unrolled max-propagation", "1500 B"],
        ["extract_objects(mask)", "Connected-component labeling", "1500 B"],
        ["count(mask)", "ReduceSum", "150 B"],
        ["compose(f, g)", "Chain", "f + g"],
    ], col_widths=[55*mm, 50*mm, CONTENT_W - 105*mm], font_size=8.5))
    s.append(P("<b>Effort:</b> 6 hours · <b>Output:</b> working transpiler for ~15 core primitives",
                "Body"))

    s.append(P("Step 3.3 — Transpile reference solve_&lt;id&gt;() programs", "H2"))
    s.append(P(
        "<b>What:</b> For each task ID where arc-dsl has a reference solver, run the "
        "transpiler to produce an ONNX file. Validate locally; ship if correct.",
        "Body"))
    s.append(P("<b>Expected:</b> ~50-100 tasks have reference solvers in arc-dsl that "
                "transpile cleanly. Another ~50-100 need manual transpiler fixes for "
                "primitives we didn't cover in Step 3.2.",
                "Body"))
    s.append(P("<b>Effort:</b> 4 hours · <b>Gain:</b> +50-100 tasks (~+950 to +1900 score)",
                "Body"))

    s.append(P("Step 3.4 — Generalize to tasks without reference solvers", "H2"))
    s.append(P(
        "<b>What:</b> For tasks without arc-dsl reference solvers, use the transpiler "
        "primitives manually. The Idea: synthesize a solver program by composing primitives "
        "based on the task signature (same as our current dispatcher, but using arc-dsl "
        "primitives which are more expressive).",
        "Body"))
    s.append(P("<b>Effort:</b> 4 hours · <b>Gain:</b> +20-40 tasks (~+380 to +760 score)",
                "Body"))

    s.append(P("Step 3.5 — Apply memory-golf and int8 optimization to all transpiled models", "H2"))
    s.append(P("<b>What:</b> Re-run every transpiled model through the Phase 1 optimization "
                "passes. The transpiler naturally produces some intermediate tensors that "
                "can be golfed down.",
                "Body"))
    s.append(P("<b>Effort:</b> 2 hours · <b>Gain:</b> +0.5-1.0 score per affected task",
                "Body"))

    s.append(P("Phase 3 Exit Criteria", "H2"))
    s.append(bullets([
        "ARC DSL transpiler operational with ~15 core primitives",
        "~100-150 tasks solved via transpiled reference programs",
        "Estimated score after Phase 3: ~3,000 (rank ~400) — 150-200 tasks solved",
    ]))
    return s


def build_phase_4():
    """Phase 4: LLM synthesis loop."""
    s = []
    s.append(P("7. Phase 4 — LLM Synthesis Loop (Days 9-10, ~10 hours)", "H1"))
    s.append(P(
        "Per finding F5, LLM synthesis at $0.85/task (Gemini 3.5 Flash High) gets 92.5% "
        "accuracy on ARC-AGI public training. After Phases 0-3, we'll have ~150-200 tasks "
        "solved via DSL. Phase 4 picks up the remaining ~200-250 tasks that no DSL "
        "primitive can handle.",
        "Body"))

    s.append(P("Step 4.1 — Install onnxscript and verify it works", "H2"))
    s.append(P(
        "<b>Why:</b> Per the LLM synthesis research, asking LLMs for raw ONNX protobuf "
        "fails constantly. <code>onnxscript</code> (Microsoft) provides a Python "
        "<code>@script()</code> decorator + <code>op.*</code> API that compiles to ONNX "
        "with free parse-time validation.",
        "Body"))
    s.append(P("<b>What:</b>", "Body"))
    s.append(P("<code>pip install onnxscript\n"
                "# Verify:\n"
                "import onnxscript\n"
                "from onnxscript import op\n"
                "from onnxscript.function import script\n"
                "@script()\n"
                "def color_map(input):\n"
                "    return op.Conv(input, W)  # works!</code>",
                "Code"))
    s.append(P("<b>Effort:</b> 30 min · <b>Output:</b> working onnxscript environment",
                "Body"))

    s.append(P("Step 4.2 — Build the LLM synthesis pipeline", "H2"))
    s.append(P("<b>What:</b> Build <code>neurogolf/llm_synth.py</code>:", "Body"))
    s.append(bullets([
        "For each unsolved task, construct a prompt with: (a) ARC task ID, (b) all "
        "input/output pairs as ASCII art, (c) the I/O tensor convention, (d) the ONNX "
        "op reference, (e) instructions to write onnxscript code that minimizes params+bytes",
        "Call Gemini 3.5 Flash High via the API (32 parallel samples per task)",
        "Execute each generated onnxscript snippet locally; validate against all pairs",
        "If at least one sample produces an eligible model, ship the smallest",
        "If none work, run one revision round: feed the LLM the error messages and ask for fixes",
    ]))
    s.append(P("<b>Prompt template (sketch):</b>", "H3"))
    s.append(P("<code>You are competing in NeuroGolf 2026. Design the smallest possible\n"
                "ONNX network that transforms input to output for ARC task {TASK_ID}.\n\n"
                "I/O convention:\n"
                "- Input: (1, 10, 30, 30) float32 — one-hot color encoding, padded with 0\n"
                "- Output: (1, 10, 30, 30) float32 — argmax over channels gives the answer\n\n"
                "Training pairs (5 shown):\n"
                "{ASCII_ART_OF_PAIRS}\n\n"
                "Write Python using onnxscript. Use only these ops: Identity, Slice, Concat,\n"
                "Transpose, Constant, Conv, Mul, Add, Sub, Pad, Resize, Tile, ArgMax, OneHot,\n"
                "Cast, ReduceSum, Gather, Greater, Less, Min, Max, Where.\n\n"
                "Minimize params + intermediate tensor bytes. Return only the Python code.</code>",
                "Code"))
    s.append(P("<b>Effort:</b> 4 hours · <b>Output:</b> working LLM synthesis pipeline",
                "Body"))

    s.append(P("Step 4.3 — Run LLM synthesis on all remaining unsolved tasks", "H2"))
    s.append(P("<b>What:</b> Iterate through the ~200-250 tasks that Phases 0-3 didn't "
                "solve. Budget: 32 samples × $0.85/task × 250 tasks = ~$6,800 in API "
                "costs. (If budget-constrained, prioritize by expected score gain: "
                "low-cost solvable tasks first.)",
                "Body"))
    s.append(P("<b>Note on cost:</b> $6,800 is more than we likely want to spend. "
                "Realistic budget: $200-400 = ~25-50 tasks attempted with 32 samples each. "
                "Pick the 50 tasks where we have the most signal (e.g., tasks with similar "
                "structure to ones we already solved).",
                "CalloutWarn"))
    s.append(P("<b>Effort:</b> 4 hours (wallclock ~6-12 hours due to API latency) · "
                "<b>Gain:</b> +30-50 tasks (~+570 to +950 score)",
                "Body"))

    s.append(P("Step 4.4 — Validate every LLM-synthesized model on fresh ARC-GEN seeds", "H2"))
    s.append(P(
        "<b>Why:</b> Per Georgy Mamarin's reply in discussion 711989: \"Gate every lossy "
        "rewrite against fresh ARC-GEN generator seeds — the hidden set uses the same "
        "generator, different seeds. Catch 'passes locally, 0 on private' overfits pre-submit.\"",
        "Body"))
    s.append(P("<b>What:</b>", "Body"))
    s.append(bullets([
        "Install RE-ARC (procedural ARC task generator) or use the ARC-AGI-2 generator",
        "For each LLM-synthesized model, generate 5-10 fresh seeds of the same task family",
        "If the model fails any fresh seed, mark it as overfit and reject",
        "This step is critical: a model that scores 0 on private benchmark is worse than no model",
    ]))
    s.append(P("<b>Effort:</b> 2 hours · <b>Output:</b> overfit-resistant LLM pipeline",
                "Body"))

    s.append(P("Phase 4 Exit Criteria", "H2"))
    s.append(bullets([
        "LLM synthesis pipeline operational with onnxscript output format",
        "Fresh-seed validation catches overfit models before submission",
        "Estimated score after Phase 4: ~3,800 (rank ~150) — 200-250 tasks solved",
    ]))
    return s


def build_phase_5():
    """Phase 5: Memory golf and final push."""
    s = []
    s.append(P("8. Phase 5 — Memory Golf & Final Push (Day 11, ~8 hours)", "H1"))
    s.append(P(
        "Per finding F2, memory (intermediate tensor bytes) is the largest cost component. "
        "Phase 5 applies aggressive memory-golf to every solver, then does a final "
        "byte-level optimization pass.",
        "Body"))

    s.append(P("Step 5.1 — Apply memory-golf wrapper to every solver", "H2"))
    s.append(P(
        "<b>What:</b> For every solver, wrap its output with: Pad-input-down to content "
        "bbox → run-model → Pad-output-up. This means the model only operates on the "
        "actual content, not the 30×30 padded canvas. For a 5×5 input, this drops memory "
        "from 30×30=900 to 5×5=25 — a 36× reduction.",
        "Body"))
    s.append(P("<b>Effort:</b> 3 hours · <b>Gain:</b> potentially large on tasks with "
                "small inputs but currently large intermediates",
                "Body"))

    s.append(P("Step 5.2 — Run onnxsim.simplify on every model", "H2"))
    s.append(P("<b>What:</b> Install <code>onnxsim</code> and run <code>onnxsim.simplify(model)</code> "
                "on every model. This performs constant folding, identity elimination, and "
                "graph simplification. Then re-run <code>ClearField('producer_name')</code> "
                "because onnxsim re-adds it.",
                "Body"))
    s.append(P("<b>Effort:</b> 30 min · <b>Gain:</b> ~10-20% byte reduction on most models",
                "Body"))

    s.append(P("Step 5.3 — Run scs4onnx for constant deduplication", "H2"))
    s.append(P("<b>What:</b> <code>scs4onnx</code> (Simplify Constant-Shape) deduplicates "
                "Constants with identical values. If two solvers share the same Constant "
                "(e.g., a [1,1,1,1] tensor of ones), they get merged.",
                "Body"))
    s.append(P("<b>Effort:</b> 30 min · <b>Gain:</b> ~5-10 bytes per duplicated Constant",
                "Body"))

    s.append(P("Step 5.4 — Manual byte-level golfing on top-50 highest-cost models", "H2"))
    s.append(P(
        "<b>What:</b> Sort all 400 models by cost. For the top 50 (highest cost), manually "
        "inspect each ONNX graph and look for: redundant Identity nodes, unnecessary Casts, "
        "Constants that could be initializers, op simplifications (e.g., Conv with all-ones "
        "weight → Mul).",
        "Body"))
    s.append(P("<b>Effort:</b> 3 hours · <b>Gain:</b> ~+0.5 score per affected model",
                "Body"))

    s.append(P("Step 5.5 — Run final regression test and pre-submit validator", "H2"))
    s.append(P(
        "<b>What:</b> Run the full 400-task sweep through the faithful scorer (Step 0.1). "
        "Confirm no regressions vs. golden results. Run the pre-submit validator on every "
        "single ONNX file. <b>Do not submit if any file fails validation</b> — one bad "
        "file zeroes the entire submission.",
        "Body"))
    s.append(P("<b>Effort:</b> 1 hour · <b>Output:</b> validated submission.zip ready to upload",
                "Body"))

    s.append(P("Phase 5 Exit Criteria", "H2"))
    s.append(bullets([
        "Every model has been through memory-golf, onnxsim, and scs4onnx",
        "Top 50 highest-cost models manually golfed",
        "Pre-submit validator passes on all 400 files",
        "Estimated score after Phase 5: ~4,200 (rank ~100)",
    ]))
    return s


def build_phase_6():
    """Phase 6: Final submission and monitoring."""
    s = []
    s.append(P("9. Phase 6 — Final Submission & Monitoring (Day 12, ~4 hours)", "H1"))

    s.append(P("Step 6.1 — Upload final submission.zip to Kaggle", "H2"))
    s.append(P("<b>What:</b> User uploads <code>submission.zip</code> via Kaggle UI. "
                "Document the returned score; compare to local prediction. If significantly "
                "lower, identify which tasks scored 0 on private benchmark.",
                "Body"))
    s.append(P("<b>Effort:</b> 15 min · <b>Output:</b> confirmed Kaggle score",
                "Body"))

    s.append(P("Step 6.2 — Identify private-benchmark failures", "H2"))
    s.append(P(
        "<b>What:</b> If the returned score is significantly lower than our local "
        "prediction (~4200), some models passed public pairs but failed private. Use "
        "the per-task score breakdown (if Kaggle shows it) to identify which tasks "
        "scored 0. For each, examine whether the model was overfit (LLM-synthesized "
        "without fresh-seed validation) or genuinely wrong.",
        "Body"))
    s.append(P("<b>Effort:</b> 1 hour · <b>Output:</b> list of failing tasks with causes",
                "Body"))

    s.append(P("Step 6.3 — Final LLM synthesis round on identified weak tasks", "H2"))
    s.append(P("<b>What:</b> For tasks that scored 0 on private, regenerate models with "
                "more samples (64 instead of 32) and stricter fresh-seed validation. "
                "Submit again if the daily submission limit allows.",
                "Body"))
    s.append(P("<b>Effort:</b> 2 hours · <b>Gain:</b> recover 5-20 failing tasks",
                "Body"))

    s.append(P("Step 6.4 — Document lessons learned", "H2"))
    s.append(P("<b>What:</b> Update <code>worklog.md</code> with: final score, final rank, "
                "what worked, what didn't, lessons for next competition. This is "
                "valuable for any future ARC-AGI competitions.",
                "Body"))
    s.append(P("<b>Effort:</b> 30 min · <b>Output:</b> comprehensive worklog",
                "Body"))

    s.append(P("Phase 6 Exit Criteria", "H2"))
    s.append(bullets([
        "Final submission uploaded and scored",
        "Any recoverable private-benchmark failures addressed",
        "Lessons documented in worklog",
        "Final estimated score: ~4,000-4,500 (rank ~80-120)",
    ]))
    return s


def build_risks():
    s = []
    s.append(P("10. Risks, Dependencies, and Contingencies", "H1"))

    s.append(P("10.1 Hard dependencies (must be available for plan to work)", "H2"))
    s.append(table([
        ["Dependency", "Why", "Status", "Contingency"],
        ["Kaggle account (manual upload)", "Submit submission.zip", "Available (user)", "N/A"],
        ["Gemini 3.5 Flash High API access", "LLM synthesis (Phase 4)", "UNKNOWN — needs user API key", "Use Claude Opus 4.8 ($2.74/task) or skip Phase 4"],
        ["onnxruntime 1.24.4 install", "Faithful scorer (Phase 0)", "Should install via pip", "Use 1.27 with manual matching"],
        ["onnxscript package", "LLM output format (Phase 4)", "pip install onnxscript", "Fall back to manual ONNX generation"],
        ["onnxsim, scs4onnx packages", "Byte optimization (Phase 5)", "pip install both", "Manual constant folding"],
        ["Hodel's arc-dsl repo", "Reference programs (Phase 3)", "Public on GitHub", "Use only our own DSL"],
        ["RE-ARC generator", "Fresh-seed validation (Phase 4)", "Public on GitHub", "Skip overfit detection (risky)"],
        ["$200-400 API budget", "LLM synthesis costs", "User must provide", "Reduce sample count or skip Phase 4"],
    ], col_widths=[44*mm, 36*mm, 32*mm, CONTENT_W - 112*mm], font_size=8))

    s.append(P("10.2 Schedule risks", "H2"))
    s.append(table([
        ["Risk", "Likelihood", "Impact", "Mitigation"],
        ["Phase 4 (LLM) blocked by API access", "Medium", "High (-1000 score)", "Skip Phase 4, extend Phase 3 (more transpilation)"],
        ["Faithful scorer doesn't match grader", "Low", "Critical (submits may score 0)", "Validate against souldrive's published notebook first"],
        ["Hodel's arc-dsl doesn't transpile cleanly", "Medium", "High (-1500 score)", "Build only the 15 core primitives ourselves"],
        ["Memory-golf wrapper introduces bugs", "Medium", "Medium", "Apply only to solvers where it's clearly beneficial"],
        ["Time overruns push Phase 6 past deadline", "Medium", "Low", "Phases are independently shippable; submit daily"],
        ["Fresh-seed validation rejects too many models", "Medium", "Medium", "Loosen validation: accept models that pass ≥4 of 5 fresh seeds"],
    ], col_widths=[44*mm, 18*mm, 22*mm, CONTENT_W - 84*mm], font_size=8))

    s.append(P("10.3 Decision points and pivots", "H2"))
    s.append(P(
        "At each phase exit, decide whether to continue to the next phase or pivot:",
        "Body"))
    s.append(bullets([
        "<b>After Phase 0 (Day 1):</b> If Kaggle submission scores 0, the I/O convention is "
        "wrong — immediately halt all other work and re-derive the convention.",
        "<b>After Phase 2 (Day 5):</b> If coverage is &lt; 60 tasks (vs. target 80-100), the "
        "CA/flood-fill solvers are failing — pivot to LLM synthesis early.",
        "<b>After Phase 3 (Day 8):</b> If transpiler covers &lt; 100 tasks (vs. target 150), "
        "the arc-dsl approach isn't working — pivot fully to LLM synthesis.",
        "<b>After Phase 4 (Day 10):</b> If LLM synthesis succeeded on &lt; 30 tasks, the API "
        "or prompts need work — extend Phase 4 by one day before Phase 5.",
        "<b>At any point:</b> If we cross rank ~100 on the leaderboard, the marginal "
        "value of further work drops sharply — consider stopping and locking in the rank.",
    ]))
    return s


def build_summary():
    s = []
    s.append(P("11. Summary — One-Page View", "H1"))

    s.append(P("The 12-Day Sprint at a Glance", "H2"))
    s.append(table([
        ["Day", "Phase", "Work", "Est. Score", "Est. Rank"],
        ["1", "0 — Foundation", "Faithful scorer, fix bugs, regression tests, submit", "400", "~2,650"],
        ["2", "1 — Cost Opt", "int8, drop attrs, memory-golf wrapper", "550", "~2,400"],
        ["3", "2 — CA ext", "Multi-rule CA, 8-neighbor", "900", "~2,100"],
        ["4", "2 — Flood/Extract", "Flood fill, object extract", "1,200", "~1,800"],
        ["5", "2 — Count/Draw/Cond", "Count→dim, draw line, conditional", "1,500", "~1,500"],
        ["6", "3 — DSL transpiler", "Audit arc-dsl, build core", "2,000", "~1,100"],
        ["7", "3 — Transpile refs", "Transpile solve_<id>() programs", "2,500", "~700"],
        ["8", "3 — Generalize", "Manual composition for unsolved", "3,000", "~400"],
        ["9", "4 — LLM setup", "onnxscript, pipeline, prompts", "3,200", "~350"],
        ["10", "4 — LLM run", "Synthesize 30-50 hard tasks", "3,800", "~150"],
        ["11", "5 — Memory golf", "Apply to all, byte-level golf", "4,200", "~100"],
        ["12", "6 — Submit & monitor", "Upload, debug, document", "4,000-4,500", "~80-120"],
    ], col_widths=[12*mm, 36*mm, 60*mm, 22*mm, 22*mm], font_size=8.5))

    s.append(P("Realistic Outcomes", "H2"))
    s.append(table([
        ["Outcome", "Probability", "Score", "Rank", "Prize?"],
        ["Pessimistic (multiple blockers)", "20%", "~2,000", "~700", "No medal"],
        ["Realistic (most phases succeed)", "50%", "~3,500", "~250", "No medal"],
        ["Optimistic (all phases hit targets)", "25%", "~4,500", "~80", "Bronze edge"],
        ["Stretch (LLM exceeds expectations)", "5%", "~6,000+", "~30", "Silver possible"],
    ], col_widths=[58*mm, 22*mm, 22*mm, 22*mm, CONTENT_W - 124*mm], font_size=8.5))

    s.append(P("What This Plan Buys Us vs. Status Quo", "H2"))
    s.append(bullets([
        "<b>Status quo (no further work):</b> ~300 score, rank ~2,750 — essentially last place.",
        "<b>This plan, realistic outcome:</b> ~3,500 score, rank ~250 — top 10% of leaderboard.",
        "<b>This plan, optimistic outcome:</b> ~4,500 score, rank ~80 — Bronze medal range.",
        "<b>Theoretical max (not feasible):</b> 9,500 score, rank 1 — would need 6 more months.",
    ]))
    s.append(P(
        "<b>Bottom line: this plan converts a last-place submission into a credible "
        "top-10% finish, with a small (5-25%) chance at a medal. The single highest-leverage "
        "action is getting the LLM synthesis pipeline working — that alone could add 1,000+ "
        "points if we have API access. The single most critical infrastructure investment "
        "is the faithful local scorer — without it, we may submit zero-scoring files without "
        "knowing.</b>",
        "CalloutOK"))

    s.append(P("Immediate Next Action", "H2"))
    s.append(P(
        "Start Phase 0 immediately. The single most important step is Step 0.1 (build "
        "the faithful local scorer) because every subsequent step depends on accurate "
        "validation. Step 0.7 (submit to Kaggle) should happen on Day 1 to validate the "
        "I/O convention before we invest 11 more days of work on top of a wrong assumption.",
        "Body"))
    return s


def main():
    out_path = "/home/z/my-project/download/NeuroGolf_Implementation_Plan.pdf"
    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=MARGIN_L, rightMargin=MARGIN_R,
        topMargin=MARGIN_T, bottomMargin=MARGIN_B,
        title="NeuroGolf 2026 — Implementation Plan",
        author="Z.ai",
        subject="Research-driven 12-day implementation plan",
        creator="Z.ai",
    )

    story = []
    story.extend(build_cover())
    story.extend(build_research_synthesis())
    story.extend(build_gap_analysis())
    story.extend(build_phase_0())
    story.extend(build_phase_1())
    story.extend(build_phase_2())
    story.extend(build_phase_3())
    story.extend(build_phase_4())
    story.extend(build_phase_5())
    story.extend(build_phase_6())
    story.extend(build_risks())
    story.extend(build_summary())

    doc.build(story, onFirstPage=on_cover, onLaterPages=on_page)
    print(f"Generated: {out_path}")
    print(f"Size: {os.path.getsize(out_path)} bytes")


if __name__ == "__main__":
    main()
