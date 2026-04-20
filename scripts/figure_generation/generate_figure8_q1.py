#!/usr/bin/env python3
"""
Figure 8 – Normalized Physics Constraints Illustration
Q1 journal quality (double-column width, Arial 12 pt, shared x-axis).

Design choices
--------------
* Double-column width: 7.0 in (178 mm)
* Font: Arial 12 pt (ticks, axis labels); 10 pt for legend and in-panel notes
* Three stacked panels share one x-axis → tick labels and x-label only on the
  bottom panel; upper panels have no x-axis tick labels or label.
* Panel tags "(a)", "(b)", "(c)" are regular-weight Arial inside each panel (top-left),
  not as figure titles, so they never collide with the neighbour's content.
* hspace kept tight (0.10) because the inter-panel space is clean.
* Line weight 1.5 pt; legend box with thin black border.
* 600 dpi PNG + PDF output.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FONT_MAIN = 12   # ticks, axis labels, panel tags
FONT_SMALL = 10  # legends, annotations

# ---------------------------------------------------------------------------
# Global font setup (Arial throughout)
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "text.usetex": False,
    "axes.formatter.use_mathtext": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "font.size": FONT_MAIN,
    "axes.labelsize": FONT_MAIN,
    "axes.titlesize": FONT_MAIN,
    "xtick.labelsize": FONT_MAIN,
    "ytick.labelsize": FONT_MAIN,
    "legend.fontsize": FONT_SMALL,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "pdf.fonttype": 42,   # embed TrueType fonts in PDF
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
SUCTION_MIN = 0.1    # kPa
SUCTION_MAX = 1e6    # kPa
psi = np.logspace(np.log10(SUCTION_MIN), np.log10(SUCTION_MAX), 200)
s_norm = (np.log10(psi) - np.log10(SUCTION_MIN)) / (
    np.log10(SUCTION_MAX) - np.log10(SUCTION_MIN)
)
theta_norm = 1.0 / (1.0 + (s_norm / 0.25) ** 1.5)
dtheta_dlogpsi = np.gradient(theta_norm, np.log10(psi))

# x-axis ticks (plain numbers, no sci notation)
XTICKS  = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
XLABELS = ["0.1", "1", "10", "100", "1 000", "10 000", "100 000", "1 000 000"]

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
FIG_W = 7.0          # inches (double-column)
PANEL_H = 1.9        # inches per panel
HSPACE = 0.24        # larger inter-panel spacing for cleaner separation

fig, axes = plt.subplots(
    3, 1,
    figsize=(FIG_W, PANEL_H * 3 + 0.85),   # extra height to preserve panel area with larger hspace
    gridspec_kw=dict(hspace=HSPACE),
)
ax_a, ax_b, ax_c = axes

# Colors
C_BLUE   = "#2166AC"
C_RED    = "#D6604D"
C_GREEN  = "#4DAC26"
LW = 1.5

# ---------------------------------------------------------------------------
# Shared x-axis style helper
# ---------------------------------------------------------------------------
def style_ax(ax, *, bottom_panel: bool):
    ax.set_xlim([SUCTION_MIN, SUCTION_MAX])
    ax.set_xticks(XTICKS)
    if bottom_panel:
        ax.set_xticklabels(XLABELS)
        ax.set_xlabel("Matric suction \u03c8 (kPa)", labelpad=4)
    else:
        ax.set_xticklabels([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("black")
    ax.tick_params(which="both", top=True, right=True)


def panel_tag(ax, tag, x=0.0, y=1.03):
    """Place panel tag outside top-left of axes (above the frame)."""
    ax.text(
        x, y, tag,
        transform=ax.transAxes,
        fontsize=FONT_MAIN, fontweight="normal", fontfamily="Arial",
        va="bottom", ha="left", clip_on=False,
    )


def small_legend(ax, handles, **kw):
    leg = ax.legend(
        handles=handles,
        frameon=False, borderpad=0.5, handlelength=1.6,
        **kw,
    )
    for t in leg.get_texts():
        t.set_fontfamily("Arial")
        t.set_fontsize(FONT_SMALL)
    return leg


# ---------------------------------------------------------------------------
# Panel (a) – Ideal θ_norm(ψ)
# ---------------------------------------------------------------------------
ax_a.semilogx(psi, theta_norm, color=C_BLUE, lw=LW)
ax_a.set_ylim([0, 1.08])
ax_a.set_ylabel(
    r"$\theta_{\mathrm{norm}}(\psi)$  [–]",
    labelpad=4,
)
style_ax(ax_a, bottom_panel=False)
panel_tag(ax_a, "(a)  Normalized water content")

# Annotations
ax_a.annotate(
    r"$\theta_{\mathrm{norm}} \approx 1$ (saturated)",
    xy=(0.4, theta_norm[np.searchsorted(psi, 0.4)]),
    xytext=(1.0, 0.95),
    fontsize=FONT_SMALL, fontfamily="Arial",
    arrowprops=dict(arrowstyle="-", color="gray", lw=0.7),
    va="center",
)
ax_a.text(
    2e5,
    0.25,
    r"$\theta_{\mathrm{norm}} \approx 0$ (residual)",
    ha="center",
    va="center",
    fontsize=FONT_SMALL,
    fontfamily="Arial",
)
small_legend(
    ax_a,
    [Line2D([0], [0], color=C_BLUE, lw=LW,
            label=r"$\theta_{\mathrm{norm}}(\psi)$ — example curve")],
    loc="upper right",
)

# ---------------------------------------------------------------------------
# Panel (b) – Derivative ≤ 0 (monotonicity)
# ---------------------------------------------------------------------------
ax_b.semilogx(psi, dtheta_dlogpsi, color=C_RED, lw=LW)
ax_b.axhline(0.0, color="black", lw=0.8, linestyle="--", dashes=(4, 3))
ax_b.set_ylabel(
    r"$d\theta_{\mathrm{norm}}/d\log_{10}(\psi)$  [–]",
    labelpad=4,
)
style_ax(ax_b, bottom_panel=False)
panel_tag(ax_b, "(b)  Monotonicity constraint")

ax_b.text(
    0.50, 0.28, "All slopes \u2264 0",
    transform=ax_b.transAxes,
    ha="center", va="center", fontsize=FONT_SMALL, fontfamily="Arial",
    color="#555555",
)
small_legend(
    ax_b,
    [
        Line2D([0], [0], color=C_RED, lw=LW,
               label=r"$d\theta_{\mathrm{norm}}/d\log_{10}(\psi)$"),
        Line2D([0], [0], color="black", lw=0.8, linestyle="--", dashes=(4, 3),
               label="Zero reference"),
    ],
    loc="lower right",
)

# ---------------------------------------------------------------------------
# Panel (c) – Physically plausible band
# ---------------------------------------------------------------------------
mid = (psi >= 1e2) & (psi <= 1e4)
ax_c.semilogx(psi, theta_norm, color=C_BLUE, lw=LW)
ax_c.fill_between(
    psi, theta_norm - 0.10, theta_norm + 0.10,
    where=mid, color=C_GREEN, alpha=0.20,
)
# Outline the band with a dashed green border for clarity
ax_c.semilogx(
    psi[mid], theta_norm[mid] + 0.10,
    color=C_GREEN, lw=0.6, linestyle="--", dashes=(4, 3),
)
ax_c.semilogx(
    psi[mid], theta_norm[mid] - 0.10,
    color=C_GREEN, lw=0.6, linestyle="--", dashes=(4, 3),
)
ax_c.set_ylim([0, 1.08])
ax_c.set_ylabel(
    r"$\theta_{\mathrm{norm}}(\psi)$  [–]",
    labelpad=4,
)
style_ax(ax_c, bottom_panel=True)
panel_tag(ax_c, "(c)  Physically plausible band")

small_legend(
    ax_c,
    [
        Line2D([0], [0], color=C_BLUE, lw=LW,
               label=r"$\theta_{\mathrm{norm}}(\psi)$ — example curve"),
        Patch(facecolor=C_GREEN, alpha=0.35, edgecolor="none",
              label="Illustrative plausible band"),
    ],
    loc="upper right",
)

# ---------------------------------------------------------------------------
# Final adjustments
# ---------------------------------------------------------------------------
fig.align_ylabels(axes)
fig.subplots_adjust(left=0.11, right=0.98, top=0.98, bottom=0.09)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
stem = OUTPUT_DIR / "Figure8_Normalized_Physics_Constraints"
fig.savefig(str(stem) + ".png", dpi=600, bbox_inches="tight", pad_inches=0.05)
fig.savefig(str(stem) + ".pdf", dpi=600, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print(f"Saved:\n  {stem}.png\n  {stem}.pdf")
