#!/usr/bin/env python3
"""
Figure S2 — GSHP External Validation (Detailed Metrics)
Q1 journal quality (7.0 in wide × 5.0 in tall, 2 side-by-side panels).

Layout: 1 row × 2 columns  (GridSpec: left 42 %, right 58 %)
  (a) Knee fidelity metrics — grouped boxplots (log y-axis)
      Groups: ψ₅₀ (kPa) and max |dθ/dlog ψ|  [-]
      Within each group: GSHP (blue, left) | VGParamNet (purple, right)
  (b) Worst-case prediction — single SWCC (top 1 % RMSE)
      GSHP observed  → black solid
      VGParamNet pred → purple dashed

Data source:
  results_gshp_validation/gshp_predicted_curves.npy   [7072, 100]
  results_gshp_validation/gshp_observed_curves.npy    [7072, 100]
  data_pinn_normalized/suction_grid.npy               [100]

Design
------
* 7.0 in wide × 5.0 in tall; Arial 12 pt labels/tags; 10 pt legend/annotations
* Colors: GSHP = #1F77B4 (blue) | VGParamNet = #9B59B6 (purple, matches Figs 16–17)
* Inward ticks mirrored (top/right); no grid; clean white box
* PDF only (vector) — no PNG
* pdf.fonttype = 42
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.ticker as mticker
import numpy as np

ROOT     = Path(__file__).resolve().parents[2]
SUPP_DIR = ROOT / "paper_figures" / "supplementary"
SUPP_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

FONT_MAIN  = 12
FONT_SMALL = 10
LW         = 1.8

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

C_GSHP = "#1F77B4"   # blue   — GSHP observed
C_VG   = "#9B59B6"   # purple — VGParamNet (matches Figs 16–17)

# 6 ticks — fits the narrow panel (b) without label overlap
# (drop 10 000 and 100 000; curve is flat in that range anyway)
XTICKS_PSI  = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e6]
XLABELS_PSI = ["0.1", "1", "10", "100", "1 000", "1 000 000"]


# ── Style helpers ────────────────────────────────────────────────────────────
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


def _legend(ax, **kw):
    leg = ax.legend(frameon=False, borderpad=0.4, handlelength=1.8,
                    fontsize=FONT_SMALL, **kw)
    for t in leg.get_texts():
        t.set_fontfamily("Arial")
        t.set_fontsize(FONT_SMALL)
    return leg


def _boxplot_style(bp, color):
    """Uniform colour for a single-colour patch_artist boxplot group."""
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_linewidth(0.8)
    for elem in ["whiskers", "caps"]:
        for line in bp[elem]:
            line.set_linewidth(0.8)
            line.set_color("#444444")
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.5)
    for fl in bp.get("fliers", []):
        fl.set_marker("o")
        fl.set_markersize(3)
        fl.set_markerfacecolor("none")
        fl.set_markeredgewidth(0.6)
        fl.set_markeredgecolor("#888888")


# ── Knee metric computation ──────────────────────────────────────────────────
def _compute_knee_metrics(curves, theta_s, theta_r, psi):
    """Return (psi50 [N], max_slope [N]) arrays."""
    log_psi = np.log10(psi + 1e-10)
    n = len(curves)
    psi50     = np.full(n, np.nan, dtype=np.float32)
    max_slope = np.full(n, np.nan, dtype=np.float32)

    theta_s = theta_s.reshape(-1, 1)
    theta_r = theta_r.reshape(-1, 1)
    Se = np.clip((curves - theta_r) / np.maximum(theta_s - theta_r, 1e-6), 1e-6, 1 - 1e-6)

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

    dtheta  = np.diff(curves, axis=1)
    dlogpsi = np.diff(log_psi)[None, :]
    max_slope = np.max(np.abs(dtheta / (dlogpsi + 1e-10)), axis=1).astype(np.float32)

    return psi50, max_slope


# ── Panel plotters ───────────────────────────────────────────────────────────
def _plot_panel_a(ax, curves_pred, curves_gshp, psi, theta_s, theta_r):
    """(a) Grouped boxplots: ψ₅₀ and max slope, GSHP vs VGParamNet."""
    psi50_vg,  slope_vg  = _compute_knee_metrics(curves_pred, theta_s, theta_r, psi)
    psi50_gs,  slope_gs  = _compute_knee_metrics(curves_gshp, theta_s, theta_r, psi)

    def _clean(arr):
        return arr[np.isfinite(arr) & (arr > 0)]

    p50_gs  = _clean(psi50_gs);  p50_vg  = _clean(psi50_vg)
    slp_gs  = _clean(slope_gs);  slp_vg  = _clean(slope_vg)

    print(f"  psi50  — GSHP median={np.median(p50_gs):.2f} kPa,  "
          f"VGParamNet median={np.median(p50_vg):.2f} kPa")
    print(f"  slope  — GSHP median={np.median(slp_gs):.3f},  "
          f"VGParamNet median={np.median(slp_vg):.3f}")

    bp_kw = dict(patch_artist=True, showfliers=False,
                 medianprops=dict(color="black", linewidth=1.5),
                 whiskerprops=dict(linewidth=0.8),
                 capprops=dict(linewidth=0.8),
                 flierprops=dict(marker="o", markersize=3,
                                 markerfacecolor="none", markeredgewidth=0.6))

    # Group 1: ψ₅₀ (positions 1=GSHP, 2=VGParamNet)
    bp1g = ax.boxplot([p50_gs], positions=[1], widths=0.5, **bp_kw)
    bp1v = ax.boxplot([p50_vg], positions=[2], widths=0.5, **bp_kw)
    _boxplot_style(bp1g, C_GSHP)
    _boxplot_style(bp1v, C_VG)

    # Group 2: max slope (positions 4=GSHP, 5=VGParamNet)
    bp2g = ax.boxplot([slp_gs], positions=[4], widths=0.5, **bp_kw)
    bp2v = ax.boxplot([slp_vg], positions=[5], widths=0.5, **bp_kw)
    _boxplot_style(bp2g, C_GSHP)
    _boxplot_style(bp2v, C_VG)

    ax.set_yscale("log")
    ax.set_ylim(0.03, 200)   # covers both metrics' whisker ranges; clips extreme psi50 fliers
    ax.set_xlim(0.2, 5.8)
    ax.set_xticks([1.5, 4.5])
    ax.set_xticklabels(["\u03c850\n(kPa)", "Max |d\u03b8/dlog\u03c8|\n[\u2212]"],
                       fontsize=FONT_MAIN, fontfamily="Arial", linespacing=1.3)
    ax.set_ylabel("Metric value", fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)

    # Minor log ticks on y
    ax.yaxis.set_minor_locator(mticker.LogLocator(
        base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20))

    # Separator between groups
    ax.axvline(3.0, color="#CCCCCC", lw=0.8, ls="--", dashes=(4, 4), zorder=0)

    # Legend — GSHP and VGParamNet by colour (consistent across both groups)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=C_GSHP, alpha=0.55, edgecolor="none", label="GSHP"),
               Patch(facecolor=C_VG,   alpha=0.55, edgecolor="none", label="VGParamNet")]
    leg = ax.legend(handles=handles, frameon=False, borderpad=0.4,
                    handlelength=1.4, fontsize=FONT_SMALL, loc="upper right")
    for t in leg.get_texts():
        t.set_fontfamily("Arial"); t.set_fontsize(FONT_SMALL)

    _style(ax)
    _panel_tag(ax, "(a)  Knee fidelity: GSHP vs VGParamNet")


def _plot_panel_b(ax, curves_pred, curves_gshp, psi, rmse):
    """(b) Worst-case SWCC (top 1 % RMSE)."""
    outlier_idx = int(np.argmax(rmse))
    print(f"  Outlier: sample {outlier_idx}, RMSE = {rmse[outlier_idx]:.4f}")

    ax.semilogx(psi, curves_gshp[outlier_idx],
                color=C_GSHP, lw=LW, ls="-", label="GSHP  (observed)", zorder=3)
    ax.semilogx(psi, curves_pred[outlier_idx],
                color=C_VG, lw=LW, ls="--", dashes=(6, 3),
                label="VGParamNet  (predicted)", zorder=2)

    ax.set_xscale("log")
    ax.set_xticks(XTICKS_PSI)
    ax.set_xticklabels(XLABELS_PSI, fontsize=8, fontfamily="Arial",
                       rotation=0, ha="center")
    ax.set_xlabel("Matric suction \u03c8  (kPa)", fontsize=FONT_MAIN,
                  fontfamily="Arial", labelpad=6)
    ax.set_ylabel("Water content \u03b8  [\u2212]", fontsize=FONT_MAIN,
                  fontfamily="Arial", labelpad=6)
    ax.set_xlim(0.07, float(psi.max()))   # 0.07 gives room for "0.1" label at left edge
    y_max = max(float(curves_gshp[outlier_idx].max()),
                float(curves_pred[outlier_idx].max())) * 1.08
    ax.set_ylim(0, y_max)

    # RMSE annotation — plain text, no bbox
    ax.text(0.04, 0.95,
            f"RMSE = {rmse[outlier_idx]:.4f}",
            transform=ax.transAxes,
            fontsize=FONT_SMALL, fontfamily="Arial", color="#333333",
            va="top", ha="left")

    _legend(ax, loc="upper right")
    _style(ax)
    _panel_tag(ax, "(b)  Worst-case prediction (top 1 % error)")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    data_dir  = ROOT / "results_gshp_validation"
    psi_path  = ROOT / "data_pinn_normalized" / "suction_grid.npy"

    for p in [data_dir / "gshp_predicted_curves.npy",
              data_dir / "gshp_observed_curves.npy",
              psi_path]:
        if not p.exists():
            print(f"ERROR: missing file:\n  {p}")
            return 1

    curves_pred = np.load(data_dir / "gshp_predicted_curves.npy")
    curves_gshp = np.load(data_dir / "gshp_observed_curves.npy")
    psi         = np.load(psi_path).astype(np.float32)

    theta_s = np.max(curves_gshp,  axis=1).astype(np.float32)
    theta_r = np.min(curves_gshp,  axis=1).astype(np.float32)
    rmse    = np.sqrt(np.mean((curves_pred - curves_gshp) ** 2, axis=1))

    print(f"Loaded: {len(curves_pred)} GSHP samples, {len(psi)} suction points")
    print(f"RMSE: mean={rmse.mean():.4f}, max={rmse.max():.4f}")

    # ── Figure layout ────────────────────────────────────────────────────────
    FIG_W, FIG_H = 7.0, 5.0
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs  = mgridspec.GridSpec(1, 2, figure=fig,
                             width_ratios=[42, 58],
                             wspace=0.42,
                             left=0.11, right=0.97,
                             top=0.92, bottom=0.13)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    _plot_panel_a(ax_a, curves_pred, curves_gshp, psi, theta_s, theta_r)
    _plot_panel_b(ax_b, curves_pred, curves_gshp, psi, rmse)

    # ── Save PDF only (vector) ───────────────────────────────────────────────
    stem    = "Figure_S2_GSHP_External_Validation_Detailed_q1"
    pdf_out = SUPP_DIR / f"{stem}.pdf"
    fig.savefig(str(pdf_out), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"\nSaved:\n  {pdf_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
