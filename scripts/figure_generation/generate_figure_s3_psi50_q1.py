#!/usr/bin/env python3
"""
Figure S3 — ψ50 Distribution: Real UNSODA vs GAN Synthetic
Q1 journal quality (7.0 in wide × 5.0 in tall, single panel).

Key design choices
------------------
* Y-axis: RELATIVE FREQUENCY (fraction of samples per bin), not raw count.
  Real N=556 and Synthetic N≤1000 are different; normalising makes the
  comparison fair and makes the GAN's narrow spike visually obvious.
* Real (UNSODA): ψ50 computed from the 556 original SWCC curves
  (y_train 389 + y_val 83 + y_test 84) — NOT from psi50_train.npy which
  mixes real and GAN-augmented samples.
* Synthetic (GAN): ψ50 computed from up to 1000 GAN-generated curves
  (filtered set if available, else raw).
* Median dashed lines added for each distribution.
* GAN range band annotated to highlight mode collapse.

Data sources
------------
  data_processed/y_train.npy, y_val.npy, y_test.npy  [N, 100]
  data_processed/suction_grid.npy                    [100]
  results_gan/generated_data_filtered/synthetic_swcc_curves_filtered.npy
    or results_gan/generated_data/synthetic_swcc_curves.npy

Design
------
* 7.0 in wide × 5.0 in tall; Arial 12 pt labels/tag; 10 pt legend/annotations
* Real = #2E86AB (blue) | Synthetic = #C73E1D (red)
* Inward ticks mirrored; no grid; clean white box
* PDF only (vector); pdf.fonttype = 42
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
# Default: headless batch rendering. Use --show for interactive display.
SHOW_FIG = "--show" in sys.argv
if not SHOW_FIG:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT     = Path(__file__).resolve().parents[2]
SUPP_DIR = ROOT / "paper_figures" / "supplementary"
SUPP_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

FONT_MAIN  = 12
FONT_SMALL = 10

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

C_REAL = "#2E86AB"   # blue  — Real UNSODA
C_SYN  = "#C73E1D"   # red   — Synthetic GAN

N_SYN_MAX = 1000     # cap on synthetic samples used


# ── ψ50 computation ──────────────────────────────────────────────────────────
def _psi50_from_curves(curves: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """
    Compute ψ50 (kPa) for each curve via linear interpolation of Se = 0.5.
    Uses per-curve θs = max, θr = min to compute normalised effective saturation.
    """
    log_psi = np.log10(psi + 1e-10)
    n = len(curves)
    psi50 = np.full(n, np.nan, dtype=np.float64)

    theta_s = curves.max(axis=1, keepdims=True)
    theta_r = curves.min(axis=1, keepdims=True)
    Se = np.clip((curves - theta_r) / np.maximum(theta_s - theta_r, 1e-6),
                 1e-6, 1 - 1e-6)

    target = 0.5
    for i in range(n):
        idx = np.where(Se[i] <= target)[0]
        if len(idx) > 0 and idx[0] > 0:
            k = idx[0]
            x0, x1 = log_psi[k - 1], log_psi[k]
            y0, y1 = Se[i, k - 1], Se[i, k]
            if y1 != y0:
                t = (target - y0) / (y1 - y0)
                psi50[i] = 10 ** (x0 + t * (x1 - x0))
            else:
                psi50[i] = 10 ** x1
    return psi50


# ── Style helpers ─────────────────────────────────────────────────────────────
def _style(ax):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("black")
    ax.tick_params(which="both", top=True, right=True, direction="in",
                   labelsize=FONT_MAIN)
    ax.grid(False)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("Arial")
        lbl.set_fontsize(FONT_MAIN)


def _panel_tag(ax, tag):
    ax.text(0.0, 1.03, tag, transform=ax.transAxes,
            fontsize=FONT_MAIN, fontweight="normal", fontfamily="Arial",
            va="bottom", ha="left", clip_on=False)


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    dp = ROOT / "data_processed"
    psi = np.load(dp / "suction_grid.npy").astype(np.float64)

    # ── Real UNSODA: 556 original curves (train 389 + val 83 + test 84) ──────
    y_train = np.load(dp / "y_train.npy").astype(np.float64)
    y_val   = np.load(dp / "y_val.npy").astype(np.float64)
    y_test  = np.load(dp / "y_test.npy").astype(np.float64)
    y_real  = np.concatenate([y_train, y_val, y_test], axis=0)

    psi50_real = _psi50_from_curves(y_real, psi)
    psi50_real = psi50_real[np.isfinite(psi50_real) & (psi50_real > 0)]
    print(f"Real UNSODA: N={len(psi50_real)}, "
          f"median={np.median(psi50_real):.1f} kPa, "
          f"range=[{psi50_real.min():.2f}, {psi50_real.max():.1f}] kPa")

    # ── Synthetic GAN ─────────────────────────────────────────────────────────
    filt_curves = ROOT / "results_gan" / "generated_data_filtered" / "synthetic_swcc_curves_filtered.npy"
    filt_psi    = ROOT / "results_gan" / "generated_data_filtered" / "suction_grid.npy"
    raw_curves  = ROOT / "results_gan" / "generated_data" / "synthetic_swcc_curves.npy"
    raw_psi     = ROOT / "results_gan" / "generated_data" / "suction_grid.npy"

    if filt_curves.exists():
        y_syn   = np.load(filt_curves).astype(np.float64)
        psi_syn = np.load(filt_psi).astype(np.float64)
    else:
        y_syn   = np.load(raw_curves).astype(np.float64)
        psi_syn = np.load(raw_psi).astype(np.float64)

    rng = np.random.RandomState(42)
    if len(y_syn) > N_SYN_MAX:
        y_syn = y_syn[rng.choice(len(y_syn), N_SYN_MAX, replace=False)]

    psi50_syn = _psi50_from_curves(y_syn, psi_syn)
    psi50_syn = psi50_syn[np.isfinite(psi50_syn) & (psi50_syn > 0)]
    print(f"Synthetic GAN: N={len(psi50_syn)}, "
          f"median={np.median(psi50_syn):.1f} kPa, "
          f"range=[{psi50_syn.min():.1f}, {psi50_syn.max():.1f}] kPa")
    print(f"  GAN IQR: [{np.percentile(psi50_syn,25):.1f}, "
          f"{np.percentile(psi50_syn,75):.1f}] kPa")

    # ── Bins: 40 log-spaced over combined range ───────────────────────────────
    lo = min(psi50_real.min(), psi50_syn.min())
    hi = max(psi50_real.max(), psi50_syn.max())
    bins = np.logspace(np.log10(lo), np.log10(hi), 40)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    # Normalised to relative frequency (weights sum to 1.0 per dataset)
    w_real = np.ones(len(psi50_real)) / len(psi50_real)
    w_syn  = np.ones(len(psi50_syn))  / len(psi50_syn)

    ax.hist(psi50_real, bins=bins, weights=w_real,
            color=C_REAL, alpha=0.55, edgecolor="white", linewidth=0.3,
            label=f"Real UNSODA  (N = {len(psi50_real):,})", zorder=2)
    ax.hist(psi50_syn, bins=bins, weights=w_syn,
            color=C_SYN,  alpha=0.60, edgecolor="white", linewidth=0.3,
            label=f"Synthetic GAN  (N = {len(psi50_syn):,})", zorder=3)

    # Median lines
    med_real = np.median(psi50_real)
    med_syn  = np.median(psi50_syn)
    ax.axvline(med_real, color=C_REAL, lw=1.4, ls="--", dashes=(5, 3), zorder=4)
    ax.axvline(med_syn,  color=C_SYN,  lw=1.4, ls="--", dashes=(5, 3), zorder=4)

    # GAN range shaded band
    gan_lo, gan_hi = psi50_syn.min(), psi50_syn.max()
    ax.axvspan(gan_lo, gan_hi, color=C_SYN, alpha=0.08, zorder=1)

    # Annotation: GAN confined range — use blended transform (data x, axes-fraction y)
    # Place label to the right of the GAN band to avoid overlap with the GAN median line
    gan_mid = 10 ** ((np.log10(gan_lo) + np.log10(gan_hi)) / 2)
    ax.text(gan_hi * 1.5, 0.60,
            f"GAN range:\n{gan_lo:.0f}\u2013{gan_hi:.0f} kPa",
            transform=ax.get_xaxis_transform(),
            fontsize=FONT_SMALL - 1, fontfamily="Arial", color=C_SYN,
            ha="left", va="bottom")

    # Median annotations
    ymax_f = 0.96   # axes fraction
    ax.text(med_real * 1.25, ymax_f,
            f"Median\n{med_real:.0f} kPa",
            transform=ax.get_xaxis_transform(),
            fontsize=FONT_SMALL - 1, fontfamily="Arial", color=C_REAL,
            va="top", ha="left")
    ax.text(med_syn * 0.75, ymax_f,
            f"Median\n{med_syn:.0f} kPa",
            transform=ax.get_xaxis_transform(),
            fontsize=FONT_SMALL - 1, fontfamily="Arial", color=C_SYN,
            va="top", ha="right")

    # Axes
    ax.set_xscale("log")
    ax.set_xticks([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    ax.set_xticklabels(
        ["0.1", "1", "10", "100", "1 000", "10 000", "100 000", "1 000 000"],
        fontsize=8, fontfamily="Arial", rotation=0, ha="center")
    ax.set_xlabel("Matric suction \u03c850  (kPa)",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)
    ax.set_ylabel("Relative frequency  [\u2212]",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)
    ax.set_xlim(0.07, float(hi) * 2.0)

    # Legend
    leg = ax.legend(frameon=False, borderpad=0.4, handlelength=1.6,
                    fontsize=FONT_SMALL, loc="upper left")
    for t in leg.get_texts():
        t.set_fontfamily("Arial"); t.set_fontsize(FONT_SMALL)

    _style(ax)
    _panel_tag(ax, "(a)  \u03c850 distribution: real UNSODA vs GAN synthetic")

    fig.subplots_adjust(left=0.12, right=0.97, top=0.91, bottom=0.13)

    # ── Save PDF only ─────────────────────────────────────────────────────────
    pdf_out = SUPP_DIR / "Figure_S3_Psi50_Distribution_Real_vs_Synthetic_q1.pdf"
    fig.savefig(str(pdf_out), bbox_inches="tight", pad_inches=0.05)
    if SHOW_FIG:
        plt.show()
    plt.close(fig)
    print(f"\nSaved:\n  {pdf_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
