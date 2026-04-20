#!/usr/bin/env python3
"""
Figure 9 - MonotonicPINN Training Curves
Q1 journal quality (double-column width, Arial 12 pt).

Layout: 1 row × 3 columns
  (a) Train data loss (left y-axis) + Val data loss (right twin y-axis)
      — twin axes because train [0.04, 0.07] and val [0.112, 0.140] differ by ~3×
  (b) Validation data loss alone with best-epoch marker
  (c) Boundary loss (the only non-zero physics penalty)

Design
------
* 7.0 in wide (178 mm, double-column), ~2.8 in tall
* Arial 12 pt labels/ticks; 10 pt legends and annotations (same as Figure 10 Q1)
* Panel tags (a)–(c) regular weight (not bold)
* Standard rotated y-axis labels (no horizontal rotation that bleeds across panels)
* Ticks inward, mirrored top/right; no grid; clean box
* 600 dpi PNG + PDF, pdf.fonttype=42 (embedded TrueType)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_PATH = ROOT / "results_pinn_fixed" / "training_history.json"

FONT_MAIN  = 12
FONT_SMALL = 10

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
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

LW = 1.5

# Colorblind-friendly palette
C_TRAIN = "#2166AC"   # blue   - train data loss
C_TTOT  = "#D6604D"   # red    - train total loss
C_VAL   = "#4DAC26"   # green  - val data loss
C_BOUND = "#E08214"   # orange - boundary loss


def _style(ax, *, twin=False):
    """Polish spines and tick direction."""
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("black")
    ax.tick_params(which="both", top=(not twin), right=True,
                   direction="in", labelsize=FONT_MAIN)
    ax.grid(False)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("Arial")


def _panel_tag(ax, tag, x=0.04, y=0.96, ha="left"):
    t = ax.text(x, y, tag, transform=ax.transAxes,
                fontsize=FONT_MAIN, fontweight="normal",
                fontfamily="Arial", va="top", ha=ha,
                zorder=10)
    t.set_clip_on(False)
    return t


def _legend(ax, handles, **kw):
    leg = ax.legend(handles=handles, frameon=True,
                    edgecolor="#555555", facecolor="white",
                    framealpha=1.0, borderpad=0.4, handlelength=1.5,
                    fontsize=FONT_SMALL, **kw)
    leg.set_zorder(3)   # below panel tags (zorder=10)
    for t in leg.get_texts():
        t.set_fontfamily("Arial")
        t.set_fontsize(FONT_SMALL)
    return leg


def _ylabel(ax, text, **kw):
    ax.set_ylabel(text, fontsize=FONT_MAIN, fontfamily="Arial",
                  labelpad=5, **kw)


def main() -> int:
    if not HISTORY_PATH.is_file():
        print(f"Missing: {HISTORY_PATH}", file=sys.stderr)
        return 1

    with open(HISTORY_PATH) as f:
        hist = json.load(f)

    epochs      = np.array(hist.get("epoch", []))
    train_total = np.array(hist.get("train_total", []))
    train_data  = np.array(hist.get("train_data",  []))
    val_data    = np.array(hist.get("val_data",    []))
    train_mono  = np.array(hist.get("train_mono",  []))
    train_bound = np.array(hist.get("train_bound", []))
    train_phys  = np.array(hist.get("train_physics", []))

    if len(epochs) == 0:
        print("No epochs in history; skipping.", file=sys.stderr)
        return 1

    # best-epoch index
    best_idx   = int(np.argmin(val_data))
    best_epoch = float(epochs[best_idx])
    best_val   = float(val_data[best_idx])

    # -----------------------------------------------------------------------
    # Figure layout: 3 rows stacked vertically, shared x-axis (Epoch)
    # Mirrors Figure 8 layout: wide panels, tick labels only on bottom panel.
    # -----------------------------------------------------------------------
    FIG_W    = 7.0
    PANEL_H  = 1.85   # inches per panel (wide-but-not-tall = good readability)
    HSPACE   = 0.10   # tight gap — no x-labels between panels

    fig, axes = plt.subplots(
        3, 1,
        figsize=(FIG_W, PANEL_H * 3 + 0.6),
        gridspec_kw=dict(hspace=HSPACE),
    )
    ax_a, ax_b, ax_c = axes

    def _smooth(y, w=12):
        """Centered moving-average; nearest-edge boundary avoids artifacts."""
        out = np.empty_like(y, dtype=float)
        half = w // 2
        for i in range(len(y)):
            lo = max(0, i - half)
            hi = min(len(y), i + half + 1)
            out[i] = y[lo:hi].mean()
        return out

    def _style_shared(ax, *, bottom_panel: bool):
        """Common axis style; x-tick labels only on bottom panel."""
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)
            sp.set_color("black")
        ax.tick_params(which="both", top=True, right=True,
                       direction="in", labelsize=FONT_MAIN)
        if not bottom_panel:
            ax.tick_params(axis="x", labelbottom=False)
        ax.grid(False)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily("Arial")

    epoch_max = float(epochs[-1])

    # -----------------------------------------------------------------------
    # Panel (a) - Training losses
    # -----------------------------------------------------------------------
    l1, = ax_a.plot(epochs, train_data,  color=C_TRAIN, lw=LW,
                    label="Train data loss")
    l2, = ax_a.plot(epochs, train_total, color=C_TTOT,  lw=LW,
                    linestyle="--", dashes=(5, 3), label="Train total loss")
    _ylabel(ax_a, "Train loss  [-]")
    _style_shared(ax_a, bottom_panel=False)
    _panel_tag(ax_a, "(a)")
    _legend(ax_a, [l1, l2], loc="upper right",
            bbox_to_anchor=(1.0, 1.0), bbox_transform=ax_a.transAxes)
    ax_a.set_xlim([0, epoch_max])

    # -----------------------------------------------------------------------
    # Panel (b) - Validation data loss with best-epoch marker
    # Raw faint + smoothed solid; dedicated panel because val [0.112-0.140]
    # is ~3x higher than train [0.04-0.07].
    # -----------------------------------------------------------------------
    val_smooth = _smooth(val_data, w=12)
    ax_b.plot(epochs, val_data,   color=C_VAL, lw=0.6, alpha=0.35)
    ax_b.plot(epochs, val_smooth, color=C_VAL, lw=LW)
    ax_b.axvline(best_epoch, color=C_TTOT, lw=0.9,
                 linestyle="--", dashes=(5, 3))
    # Place annotation to the right of the vertical marker, in mid-panel
    ax_b.text(
        best_epoch + 20, float(np.median(val_data)),
        f"Best: epoch {int(best_epoch)}",
        fontsize=FONT_SMALL, fontfamily="Arial", color=C_TTOT,
        va="center", ha="left",
    )
    _ylabel(ax_b, "Val data loss  [-]")
    _style_shared(ax_b, bottom_panel=False)
    _panel_tag(ax_b, "(b)")
    _legend(ax_b,
            [Line2D([0], [0], color=C_VAL, lw=LW, label="Val data loss"),
             Line2D([0], [0], color=C_TTOT, lw=0.9, linestyle="--",
                    dashes=(5, 3), label=f"Best epoch ({int(best_epoch)})")],
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0), bbox_transform=ax_b.transAxes)
    ax_b.set_xlim([0, epoch_max])

    # -----------------------------------------------------------------------
    # Panel (c) - Boundary loss (only non-zero physics penalty)
    # -----------------------------------------------------------------------
    handles_c = []
    if len(train_bound) == len(epochs):
        ax_c.plot(epochs, train_bound, color=C_BOUND, lw=LW)
        handles_c.append(Line2D([0], [0], color=C_BOUND, lw=LW,
                                label="Boundary loss"))
    if len(train_phys) == len(epochs) and np.any(train_phys > 0):
        ax_c.plot(epochs, train_phys, color="#4DAC26", lw=LW)
        handles_c.append(Line2D([0], [0], color="#4DAC26", lw=LW,
                                label="Physics loss"))
    if len(train_mono) == len(epochs) and np.any(train_mono > 0):
        ax_c.plot(epochs, train_mono, color="#8E44AD", lw=LW)
        handles_c.append(Line2D([0], [0], color="#8E44AD", lw=LW,
                                label="Monotonicity loss"))

    _ylabel(ax_c, "Boundary loss  [-]")
    ax_c.set_xlabel("Epoch", labelpad=4)
    _style_shared(ax_c, bottom_panel=True)
    _panel_tag(ax_c, "(c)")
    ax_c.set_xlim([0, epoch_max])
    if handles_c:
        _legend(
            ax_c,
            handles_c,
            loc="lower right",
            bbox_to_anchor=(0.98, 0.06),
            bbox_transform=ax_c.transAxes,
        )

    # -----------------------------------------------------------------------
    # Align y-labels and save
    # -----------------------------------------------------------------------
    fig.align_ylabels(axes)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.10)
    stem = OUTPUT_DIR / "Figure9_MonotonicPINN_Training_Curves"
    fig.savefig(str(stem) + ".png", dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(str(stem) + ".pdf", dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved:\n  {stem}.png\n  {stem}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
