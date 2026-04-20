#!/usr/bin/env python3
"""
Figure S1 — WGAN-GP Training Diagnostics
Q1 journal quality (7.0 in wide × 6.0 in tall, 2 stacked panels).

Layout: 2 rows × 1 column
  (a) Critic loss and Generator loss vs epoch
      Raw (faint) + 20-epoch rolling-mean overlay (solid, same colour)
  (b) Wasserstein distance estimate and Gradient penalty vs epoch
      Wasserstein distance (left y-axis, solid) and
      Gradient penalty (right y-axis, dashed) — both with 20-epoch rolling mean

Data source: results_gan/training_history.json
  Fields used: epoch | d_loss | g_loss | wasserstein_dist | gradient_penalty

Design
------
* 7.0 in wide × 6.0 in tall; Arial 12 pt labels/tags; 10 pt legend/annotations
* Colours: Critic = #2E86AB (blue) | Generator = #C73E1D (red) |
           Wasserstein = #2CA02C (green) | GP = #9467BD (purple)
* Raw curves: lw=0.6, alpha=0.35; rolling-mean curves: lw=2.0, alpha=1.0
* Inward ticks mirrored (top/right); no grid; clean white box
* 600 dpi PNG (→ paper_figures/supplementary/) + PDF (same)
* pdf.fonttype = 42
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT    = Path(__file__).resolve().parents[2]
SUPP_DIR = ROOT / "paper_figures" / "supplementary"
SUPP_DIR.mkdir(parents=True, exist_ok=True)

FONT_MAIN  = 12
FONT_SMALL = 10
LW_RAW     = 0.7
LW_MEAN    = 2.0
ALPHA_RAW  = 0.30
ROLL_W     = 20        # rolling-mean window (epochs)

matplotlib.rcParams.update({
    "text.usetex": False,
    "axes.formatter.use_mathtext": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "font.size": FONT_MAIN,
    "axes.labelsize": FONT_MAIN,
    "xtick.labelsize": FONT_MAIN,
    "ytick.labelsize": FONT_MAIN,
    "legend.fontsize": FONT_SMALL,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

C_CRITIC = "#2E86AB"   # blue
C_GEN    = "#C73E1D"   # red
C_WASS   = "#2CA02C"   # green
C_GP     = "#9467BD"   # purple


def _rolling(x: np.ndarray, w: int) -> np.ndarray:
    """Centred rolling mean; edges padded with edge values."""
    kernel = np.ones(w) / w
    pad = w // 2
    x_pad = np.pad(x, (pad, w - 1 - pad), mode="edge")
    return np.convolve(x_pad, kernel, mode="valid")


def _style(ax):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_linewidth(0.8); sp.set_color("black")
    ax.tick_params(which="both", top=True, right=True, direction="in")
    ax.grid(False)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("Arial"); lbl.set_fontsize(FONT_MAIN)


def _style_twin(ax):
    """Style for the right-hand twin axis (no top/left ticks)."""
    ax.set_facecolor("none")
    for sp in ax.spines.values():
        sp.set_linewidth(0.8); sp.set_color("black")
    ax.tick_params(which="both", direction="in")
    ax.grid(False)
    for lbl in ax.get_yticklabels():
        lbl.set_fontfamily("Arial"); lbl.set_fontsize(FONT_MAIN)


def _panel_tag(ax, tag):
    ax.text(0.0, 1.03, tag, transform=ax.transAxes,
            fontsize=FONT_MAIN, fontweight="normal", fontfamily="Arial",
            va="bottom", ha="left", clip_on=False)


def _plot_panel_a(ax, epochs, d_loss, g_loss):
    """Critic and Generator training loss curves."""
    # Raw (faint)
    ax.plot(epochs, d_loss, color=C_CRITIC, lw=LW_RAW, alpha=ALPHA_RAW, zorder=2)
    ax.plot(epochs, g_loss, color=C_GEN,    lw=LW_RAW, alpha=ALPHA_RAW, zorder=2)

    # Rolling mean (solid)
    ax.plot(epochs, _rolling(d_loss, ROLL_W), color=C_CRITIC, lw=LW_MEAN, zorder=4,
            label="Critic loss (W-distance \u2212 \u03bbGP)")
    ax.plot(epochs, _rolling(g_loss, ROLL_W), color=C_GEN,    lw=LW_MEAN, zorder=4,
            label="Generator loss (\u2212 fake score)")

    ax.set_ylabel("Loss", fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)
    ax.set_xlim(epochs[0], epochs[-1])

    # Annotation: convergence note
    ax.axhline(0, color="#AAAAAA", lw=0.6, ls="--", dashes=(4, 4), zorder=1)
    ax.text(0.02, 0.35,
            f"Solid lines = {ROLL_W}-epoch rolling mean",
            transform=ax.transAxes,
            fontsize=FONT_SMALL - 1, fontfamily="Arial", color="#555555",
            va="bottom", ha="left")

    leg = ax.legend(fontsize=FONT_SMALL, frameon=False, loc="upper right",
                    borderpad=0.3, handlelength=1.8)
    for t in leg.get_texts():
        t.set_fontfamily("Arial"); t.set_fontsize(FONT_SMALL)

    _style(ax)
    _panel_tag(ax, "(a) WGAN-GP training losses")


def _plot_panel_b(ax, epochs, wass_dist, grad_pen):
    """Wasserstein distance (left) and Gradient penalty (right twin axis)."""
    # Left axis: Wasserstein distance
    ax.plot(epochs, wass_dist, color=C_WASS, lw=LW_RAW, alpha=ALPHA_RAW, zorder=2)
    ax.plot(epochs, _rolling(wass_dist, ROLL_W), color=C_WASS, lw=LW_MEAN, zorder=4,
            label="Wasserstein distance  (left axis)")

    ax.set_ylabel("Wasserstein distance  (kPa)",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6, color=C_WASS)
    ax.tick_params(axis="y", labelcolor=C_WASS)
    ax.set_xlim(epochs[0], epochs[-1])
    ax.set_ylim(bottom=0)

    # Right twin axis: Gradient penalty
    ax2 = ax.twinx()
    ax2.plot(epochs, grad_pen, color=C_GP, lw=LW_RAW, alpha=ALPHA_RAW, zorder=2)
    ax2.plot(epochs, _rolling(grad_pen, ROLL_W), color=C_GP, lw=LW_MEAN, ls="--",
             dashes=(6, 3), zorder=4, label="Gradient penalty  (right axis)")

    ax2.set_ylabel("Gradient penalty  [\u2212]",
                   fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6, color=C_GP)
    ax2.tick_params(axis="y", labelcolor=C_GP)
    ax2.set_ylim(bottom=0)

    # Combined legend
    lines_a, labels_a = ax.get_legend_handles_labels()
    lines_b, labels_b = ax2.get_legend_handles_labels()
    leg = ax.legend(lines_a + lines_b, labels_a + labels_b,
                    fontsize=FONT_SMALL, frameon=False, loc="lower right",
                    borderpad=0.3, handlelength=1.8)
    for t in leg.get_texts():
        t.set_fontfamily("Arial"); t.set_fontsize(FONT_SMALL)

    ax.set_xlabel("Epoch", fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)

    _style(ax)
    _style_twin(ax2)
    # Re-apply tick label fonts on twin axis
    for lbl in ax2.get_yticklabels():
        lbl.set_fontfamily("Arial"); lbl.set_fontsize(FONT_MAIN)
    # Disable top ticks on twin (already handled by primary)
    ax2.tick_params(top=False)

    _panel_tag(ax, "(b) Wasserstein distance and gradient penalty")


def main() -> int:
    hist_path = ROOT / "results_gan" / "training_history.json"
    if not hist_path.exists():
        print(f"ERROR: training_history.json not found:\n  {hist_path}")
        return 1

    hist = json.load(open(hist_path))
    epochs    = np.array(hist["epoch"],            dtype=float)
    d_loss    = np.array(hist["d_loss"],            dtype=float)
    g_loss    = np.array(hist["g_loss"],            dtype=float)
    wass_dist = np.array(hist["wasserstein_dist"],  dtype=float)
    grad_pen  = np.array(hist["gradient_penalty"],  dtype=float)

    print(f"Loaded: {len(epochs)} epochs, "
          f"d_loss [{d_loss.min():.3f}, {d_loss.max():.3f}], "
          f"g_loss [{g_loss.min():.3f}, {g_loss.max():.3f}]")
    print(f"  Wasserstein dist: [{wass_dist.min():.3f}, {wass_dist.max():.3f}]")
    print(f"  Gradient penalty: [{grad_pen.min():.4f}, {grad_pen.max():.4f}]")

    # ── Build figure ──────────────────────────────────────────────────────────
    FIG_W, FIG_H = 7.0, 6.0
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, FIG_H),
                             gridspec_kw=dict(hspace=0.54))

    _plot_panel_a(axes[0], epochs, d_loss, g_loss)
    _plot_panel_b(axes[1], epochs, wass_dist, grad_pen)

    fig.subplots_adjust(left=0.12, right=0.88, top=0.95, bottom=0.10)

    stem    = "Figure_S1_WGAN_Training_Loss_q1"
    pdf_out = SUPP_DIR / f"{stem}.pdf"
    png_out = SUPP_DIR / f"{stem}.png"
    fig.savefig(str(pdf_out), dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(str(png_out), dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"\nSaved:\n  {pdf_out}\n  {png_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
