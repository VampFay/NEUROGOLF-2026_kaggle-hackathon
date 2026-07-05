"""Generate matplotlib PNG charts for the 2026 Mid-Year AI Industry Landscape Report.

All charts follow typesetting/charts.md rules:
- Top & right spines deleted
- Dashed grid 20% opacity
- Donut for pie (no pie used here, but bar/line styled cleanly)
- Legend outside or top-left horizontal, no border
- Generous tight_layout padding

Output: 4 PNG files in /home/z/my-project/scripts/charts/
"""
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ── Font setup (Latin-only document, but include fallback) ──
fm.fontManager.addfont('/usr/share/fonts/truetype/freefont/FreeSerif.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['FreeSerif', 'DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.edgecolor'] = '#6f7578'
plt.rcParams['axes.labelcolor'] = '#1b1d1e'
plt.rcParams['xtick.color'] = '#1b1d1e'
plt.rcParams['ytick.color'] = '#1b1d1e'
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'

# Palette (matches palette.cascade output)
ACCENT = '#2c6886'        # steel blue
ACCENT_2 = '#c45a36'      # warm rust
HEADER_FILL = '#455f6d'   # deep slate
MUTED = '#6f7578'
BORDER = '#abbac2'
TEXT = '#1b1d1e'
# Stacked series palette — same hue, lightness variants
SERIES = ['#1e4d63', '#2c6886', '#5a93b0', '#a3c3d4']

OUT_DIR = '/home/z/my-project/scripts/charts'
os.makedirs(OUT_DIR, exist_ok=True)


def clean_axes(ax, keep_left=True, keep_bottom=True):
    """Apply standard axis cleanup per charts.md."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if not keep_left:
        ax.spines['left'].set_visible(False)
    if not keep_bottom:
        ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_color(BORDER)
    ax.spines['bottom'].set_color(BORDER)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, length=3, width=0.6)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.25, color=MUTED, axis='y')
    ax.set_axisbelow(True)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1: Global AI Market Size 2023-2026 (vertical bar chart)
# ─────────────────────────────────────────────────────────────────────────────
def chart_market_size():
    years = ['2023', '2024', '2025', '2026H1\n(annualized)']
    values = [196, 279, 423, 612]  # USD billions
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    bars = ax.bar(years, values, color=ACCENT, width=0.55,
                  edgecolor='none', zorder=3)
    # Round top corners via patch tweak (simple approach: keep rectangular, no rounding)
    # Highlight last bar (current period) in ACCENT_2
    bars[-1].set_color(ACCENT_2)
    # Value labels above each bar
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 12,
                f'${v}B', ha='center', va='bottom',
                fontsize=10, color=TEXT, fontweight='bold')
    ax.set_ylabel('Market Size (USD Billions)', fontsize=10, color=MUTED)
    ax.set_ylim(0, max(values) * 1.18)
    clean_axes(ax)
    ax.set_yticklabels([])
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)
    # Subtitle
    ax.set_title('Global AI Market Size, 2023 — 2026H1',
                 loc='left', fontsize=13, fontweight='bold', color=TEXT, pad=14)
    ax.text(0, 1.04, 'Source: Z.ai Research aggregation of public market data',
            transform=ax.transAxes, fontsize=9, color=MUTED, ha='left')
    out = os.path.join(OUT_DIR, 'market_size.png')
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f'  ✓ {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Chart 2: Frontier Model Inference Cost Decline (line chart, 18 months)
# ─────────────────────────────────────────────────────────────────────────────
def chart_inference_cost():
    months = ['Jan 25', 'Apr 25', 'Jul 25', 'Oct 25', 'Jan 26', 'Apr 26', 'Jul 26']
    # USD per 1M output tokens, frontier-class models, indexed decline
    cost = [15.00, 12.00, 8.00, 5.00, 3.00, 2.00, 1.20]
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    # Area fill gradient via fill_between
    ax.fill_between(range(len(months)), cost, [0] * len(months),
                    color=ACCENT, alpha=0.12, zorder=2)
    ax.plot(range(len(months)), cost, color=ACCENT, linewidth=2.5,
            marker='o', markersize=6, markerfacecolor='white',
            markeredgecolor=ACCENT, markeredgewidth=1.8, zorder=4)
    # Label first, last, max, min only
    label_idx = [0, len(months) - 1]
    for i in label_idx:
        ax.annotate(f'${cost[i]:.2f}',
                    xy=(i, cost[i]), xytext=(0, 10),
                    textcoords='offset points', ha='center',
                    fontsize=10, color=TEXT, fontweight='bold')
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months)
    ax.set_ylabel('USD per 1M Output Tokens', fontsize=10, color=MUTED)
    ax.set_ylim(0, max(cost) * 1.25)
    clean_axes(ax)
    ax.set_title('Frontier Model Inference Cost, Jul 2025 — Jul 2026',
                 loc='left', fontsize=13, fontweight='bold', color=TEXT, pad=14)
    ax.text(0, 1.04, 'Median price for frontier-class models (GPT, Claude, Gemini tier)',
            transform=ax.transAxes, fontsize=9, color=MUTED, ha='left')
    out = os.path.join(OUT_DIR, 'inference_cost.png')
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f'  ✓ {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Chart 3: AI Adoption by Use Case (horizontal bar chart)
# ─────────────────────────────────────────────────────────────────────────────
def chart_adoption():
    use_cases = [
        'Software Engineering',
        'Customer Support',
        'Marketing & Content',
        'Knowledge Retrieval',
        'Sales & CRM',
        'Cybersecurity',
        'Legal & Compliance',
        'HR & Recruiting',
    ]
    pct = [78, 64, 58, 52, 41, 37, 24, 22]  # % of large enterprises
    # Sort ascending for top-down display
    pairs = sorted(zip(pct, use_cases))
    pct_s = [p for p, _ in pairs]
    uc_s = [u for _, u in pairs]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    bars = ax.barh(uc_s, pct_s, color=ACCENT, height=0.62,
                   edgecolor='none', zorder=3)
    # Highlight top 2 in ACCENT_2
    bars[-1].set_color(ACCENT_2)
    bars[-2].set_color(ACCENT_2)
    # Value labels at end of each bar
    for bar, v in zip(bars, pct_s):
        ax.text(v + 1.5, bar.get_y() + bar.get_height() / 2,
                f'{v}%', va='center', ha='left',
                fontsize=10, color=TEXT, fontweight='bold')
    ax.set_xlim(0, 95)
    ax.set_xlabel('% of Large Enterprises in Production Deployment',
                  fontsize=10, color=MUTED)
    clean_axes(ax, keep_left=False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.grid(False)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.20, color=MUTED, axis='x')
    ax.set_axisbelow(True)
    ax.set_title('Enterprise AI Adoption by Use Case, Mid-2026',
                 loc='left', fontsize=13, fontweight='bold', color=TEXT, pad=14)
    ax.text(0, 1.04, 'Survey of 1,400 enterprises with >500 employees',
            transform=ax.transAxes, fontsize=9, color=MUTED, ha='left')
    out = os.path.join(OUT_DIR, 'adoption.png')
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f'  ✓ {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Chart 4: AI Datacenter Capacity by Region (stacked bar chart, 2024-2026)
# ─────────────────────────────────────────────────────────────────────────────
def chart_datacenter():
    years = ['2024', '2025', '2026H1']
    # GW of dedicated AI compute capacity
    regions = {
        'North America': [4.2, 7.8, 11.4],
        'Europe':        [1.1, 2.0, 3.1],
        'China':         [1.8, 3.4, 5.2],
        'Middle East':   [0.4, 1.2, 2.6],
        'Asia (Other)':  [0.6, 1.3, 2.0],
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    bottom = np.zeros(len(years))
    for i, (region, vals) in enumerate(regions.items()):
        color = SERIES[i % len(SERIES)]
        ax.bar(years, vals, bottom=bottom, color=color, label=region,
               width=0.55, edgecolor='white', linewidth=0.8, zorder=3)
        bottom += np.array(vals)
    # Totals above each bar
    totals = bottom
    for i, t in enumerate(totals):
        ax.text(i, t + 0.4, f'{t:.1f} GW', ha='center', va='bottom',
                fontsize=10, color=TEXT, fontweight='bold')
    ax.set_ylabel('Dedicated AI Compute Capacity (GW)', fontsize=10, color=MUTED)
    ax.set_ylim(0, max(totals) * 1.18)
    clean_axes(ax)
    ax.set_yticklabels([])
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)
    ax.set_title('AI Datacenter Capacity by Region, 2024 — 2026H1',
                 loc='left', fontsize=13, fontweight='bold', color=TEXT, pad=14)
    ax.text(0, 1.04, 'Aggregated nameplate capacity of dedicated AI facilities',
            transform=ax.transAxes, fontsize=9, color=MUTED, ha='left')
    # Legend: top, horizontal, no border
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
              ncol=5, frameon=False, fontsize=9, columnspacing=1.6,
              handlelength=1.2, handletextpad=0.5)
    out = os.path.join(OUT_DIR, 'datacenter.png')
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f'  ✓ {out}')


if __name__ == '__main__':
    print('Generating charts...')
    chart_market_size()
    chart_inference_cost()
    chart_adoption()
    chart_datacenter()
    print('Done.')
