#!/usr/bin/env python3
"""
Figure S7 — Suction Grid Sensitivity Analysis
Q1 journal quality (14.0 in wide × 9.0 in tall, 5 panels).

Layout: 2 rows × 3 columns (GridSpec)
  Row 0: (a) col 0 — dry extrapolation histogram
         (b) col 1 — measured-max suction histogram
         (c) col 2 — RMSE vs grid extent (line)
  Row 1: (d) col 0 — RMSE vs grid resolution (line)
         (e) cols 1-2 — representative SWCC with extrapolation zones

Key messages
------------
  1. 65 % of test curves extend beyond their measured suction range (dry extrap)
  2. The grid must reach 10⁶ kPa to cover all samples
  3. RMSE decreases slightly as grid extends further (dry tail → θ ≈ θr, easy)
  4. RMSE is nearly flat from 50 to 200 grid points — 100 pts is sufficient

Data sources
------------
  results_pinn_fixed/vgparamnet/run_B/theta_vgparamnet_test.npy   [84, 100]
  data_pinn_normalized/y_test_original.npy                        [84, 100]
  data_pinn_normalized/suction_grid.npy                           [100]
  data_pinn_normalized/X_test.csv                                 [84, feats]
  results_analysis/grid_sensitivity/grid_sensitivity_results.json  (extrap stats)

Note: extrapolation stats in the JSON are estimated heuristically from curve
      derivatives (slope-based active-region detection), not from raw UNSODA
      measurement records. They provide approximate ranges for visualization.

Design
------
* 14.0 in wide × 9.0 in tall; Arial 12 pt labels/tags; 10 pt legend/annotations
* Inward ticks, mirrored top/right; no grid; white background
* Panel tags at (0.0, 1.03) va='bottom'; frameon=False legends
* PDF + 600 dpi PNG; pdf.fonttype = 42
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

ROOT     = Path(__file__).resolve().parents[2]
SUPP_DIR = ROOT / "paper_figures" / "supplementary"
SUPP_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

FONT_MAIN  = 12
FONT_SMALL = 10
FONT_TICK  = 8   # log-axis x-tick labels

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
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

C_BLUE   = "#2E86AB"
C_AMBER  = "#F18F01"
C_RED    = "#E74C3C"
C_WET    = "#2E86AB"   # wet extrapolation shading
C_DRY    = "#C73E1D"   # dry extrapolation shading
C_OBS    = "#000000"   # observed curve


# ── Style helpers ──────────────────────────────────────────────────────────────
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


def _panel_tag(ax, tag: str) -> None:
    ax.text(0.0, 1.03, tag, transform=ax.transAxes,
            fontsize=FONT_MAIN, fontweight="normal", fontfamily="Arial",
            va="bottom", ha="left", clip_on=False)


def _legend(ax, **kw):
    kw.setdefault("frameon", False)
    kw.setdefault("borderpad", 0.4)
    kw.setdefault("handlelength", 1.8)
    kw.setdefault("fontsize", FONT_SMALL)
    leg = ax.legend(**kw)
    for t in leg.get_texts():
        t.set_fontfamily("Arial")
        t.set_fontsize(FONT_SMALL)
    return leg


# ── Metric computation ─────────────────────────────────────────────────────────
def _rmse_by_extent(y_pred: np.ndarray, y_obs: np.ndarray,
                    psi: np.ndarray, max_extent: float) -> np.ndarray:
    mask = psi <= max_extent
    out = []
    for i in range(len(y_pred)):
        m = mask & np.isfinite(y_obs[i])
        if m.sum() > 0:
            out.append(float(np.sqrt(np.mean((y_pred[i][m] - y_obs[i][m]) ** 2))))
    return np.array(out)


def _rmse_at_resolution(y_pred: np.ndarray, y_obs: np.ndarray,
                        psi: np.ndarray, n_pts: int) -> tuple[float, float]:
    log_psi = np.log10(np.maximum(psi, 1e-30))
    psi_new = np.power(10.0, np.linspace(log_psi[0], log_psi[-1], n_pts))
    out = []
    for i in range(len(y_pred)):
        fp = interp1d(log_psi, y_pred[i], kind="linear",
                      bounds_error=False, fill_value="extrapolate")
        fo = interp1d(log_psi, y_obs[i],  kind="linear",
                      bounds_error=False, fill_value="extrapolate")
        lnew = np.log10(psi_new)
        yp = fp(lnew)
        yo = fo(lnew)
        m = np.isfinite(yo)
        if m.sum() > 0:
            out.append(float(np.sqrt(np.mean((yp[m] - yo[m]) ** 2))))
    return float(np.mean(out)), float(np.median(out))


# ── Panel (a): dry extrapolation histogram ────────────────────────────────────
def _plot_panel_a(ax, dry_extrap: np.ndarray):
    n_max = int(dry_extrap.max())
    ax.hist(dry_extrap, bins=min(30, n_max + 1),
            color=C_RED, alpha=0.70, edgecolor="white", linewidth=0.4)
    mean_v = float(np.mean(dry_extrap))
    ax.axvline(mean_v, color=C_BLUE, lw=1.4, ls="--", dashes=(5, 3),
               label=f"Mean: {mean_v:.1f} pts")
    ax.set_xlabel("Dry-end extrapolated points per curve",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)
    ax.set_ylabel("Count  [-]",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)
    n_extrap = int(np.sum(dry_extrap > 0))
    n_total  = len(dry_extrap)
    # Annotation: fraction requiring extrapolation
    ax.text(0.97, 0.96,
            f"{n_extrap}/{n_total} curves\nrequire dry extrap.",
            transform=ax.transAxes, fontsize=FONT_SMALL - 1,
            fontfamily="Arial", color="#333333",
            va="top", ha="right")
    _legend(ax, loc="upper right")
    _style(ax)
    _panel_tag(ax, f"(a)  Dry-end extrapolation  "
                   f"({n_extrap}/{n_total} curves, {100*n_extrap/n_total:.0f}%)")


# ── Panel (b): measured max suction histogram ─────────────────────────────────
def _plot_panel_b(ax, measured_max: np.ndarray):
    bins = np.logspace(np.log10(measured_max.min() * 0.9),
                       np.log10(measured_max.max() * 1.1), 40)
    ax.hist(measured_max, bins=bins,
            color=C_AMBER, alpha=0.70, edgecolor="white", linewidth=0.4)
    med_v = float(np.median(measured_max))
    ax.axvline(1e6, color=C_RED, lw=1.4, ls="--", dashes=(5, 3),
               label="Grid max (10^6 kPa)")
    ax.axvline(med_v, color=C_BLUE, lw=1.4, ls="--", dashes=(3, 3),
               label=f"Median: {med_v:.1e} kPa")
    ax.set_xscale("log")
    ax.set_xlabel("Estimated measured max suction (kPa)",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)
    ax.set_ylabel("Count  [-]",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    for lbl in ax.get_xticklabels():
        lbl.set_fontfamily("Arial")
        lbl.set_fontsize(FONT_TICK)
    _legend(ax, loc="upper left")
    _style(ax)
    _panel_tag(ax, "(b)  Estimated maximum measured suction per curve")


# ── Panel (c): RMSE vs grid extent ────────────────────────────────────────────
def _plot_panel_c(ax, extents, rmse_means, rmse_medians):
    ax.plot(extents, rmse_means,   "o-", color=C_BLUE,  lw=1.8, ms=6,
            label="Mean RMSE")
    ax.plot(extents, rmse_medians, "s--", color=C_AMBER, lw=1.8, ms=6,
            dashes=(5, 3), label="Median RMSE")

    # Annotate chosen grid extent
    idx_chosen = extents.index(1e6) if 1e6 in extents else -1
    if idx_chosen >= 0:
        ax.axvline(1e6, color="#CCCCCC", lw=0.8, ls=":", zorder=0)
        ax.text(1e6 * 1.15, rmse_means[idx_chosen],
                "Chosen\n(10^6 kPa)",
                fontsize=FONT_SMALL - 1, fontfamily="Arial", color="#555555",
                va="center", ha="left")

    ax.set_xscale("log")
    ax.set_xlabel("Grid maximum extent (kPa)",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)
    ax.set_ylabel("RMSE  (m\u00b3/m\u00b3)",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    for lbl in ax.get_xticklabels():
        lbl.set_fontfamily("Arial")
        lbl.set_fontsize(FONT_TICK)
    _legend(ax, loc="upper right")
    _style(ax)
    _panel_tag(ax, "(c)  RMSE vs grid extent  (VGParamNet Run B)")


# ── Panel (d): RMSE vs grid resolution ───────────────────────────────────────
def _plot_panel_d(ax, n_pts_list, rmse_means, rmse_medians):
    ax.plot(n_pts_list, rmse_means,   "o-", color=C_BLUE,  lw=1.8, ms=6,
            label="Mean RMSE")
    ax.plot(n_pts_list, rmse_medians, "s--", color=C_AMBER, lw=1.8, ms=6,
            dashes=(5, 3), label="Median RMSE")

    ax.axvline(100, color="#CCCCCC", lw=0.8, ls=":", zorder=0)
    ax.text(102, float(np.mean(rmse_means)),
            "Chosen\n(100 pts)",
            fontsize=FONT_SMALL - 1, fontfamily="Arial", color="#555555",
            va="center", ha="left")

    # Show y-axis range explicitly to make near-flat trend visible
    y_all = rmse_means + rmse_medians
    y_lo  = min(y_all) * 0.996
    y_hi  = max(y_all) * 1.004
    ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel("Number of grid points",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)
    ax.set_ylabel("RMSE  (m\u00b3/m\u00b3)",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)
    # Move legend down by ~30 mm from panel top-left (physical-distance based).
    fig = ax.figure
    pos = ax.get_position()
    ax_h_mm = pos.height * fig.get_figheight() * 25.4
    dy_frac = 30.0 / max(ax_h_mm, 1e-9)
    _legend(ax, loc="upper left", bbox_to_anchor=(0.0, 1.0 - dy_frac))
    _style(ax)
    _panel_tag(ax, "(d)  RMSE vs grid resolution  (Run B)")


# ── Panel (e): representative SWCC with extrapolation zones ──────────────────
def _plot_panel_e(ax, psi, y_test, measured_min_arr, measured_max_arr):
    # Pick sample with largest dry extrapolation for clearest illustration
    dry_extrap = np.array([np.sum(psi > m) for m in measured_max_arr])
    sample_idx = int(np.argsort(dry_extrap)[-2])   # 2nd largest (avoid extreme outlier)
    theta = y_test[sample_idx]
    mmin  = float(measured_min_arr[sample_idx])
    mmax  = float(measured_max_arr[sample_idx])

    psi_min, psi_max = float(psi.min()), float(psi.max())

    # Shaded extrapolation regions
    ax.axvspan(psi_min, mmin, color=C_WET, alpha=0.12, zorder=0,
               label="Wet extrapolation")
    ax.axvspan(mmax,   psi_max, color=C_DRY, alpha=0.12, zorder=0,
               label="Dry extrapolation")
    ax.axvspan(mmin,   mmax,    color="#AAAAAA", alpha=0.08, zorder=0,
               label="Interpolation region")

    # Boundary lines
    ax.axvline(mmin, color=C_WET, lw=1.2, ls="--", dashes=(5, 3), zorder=2)
    ax.axvline(mmax, color=C_DRY, lw=1.2, ls="--", dashes=(5, 3), zorder=2)

    # Curve
    ax.semilogx(psi, theta, color=C_OBS, lw=2.0, ls="-",
                label="Interpolated SWCC", zorder=3)

    # Annotations for boundaries
    y_anno = float(theta.max()) * 0.92
    ax.text(mmin * 1.3, y_anno, f"Est. meas.\nmin\n({mmin:.0f} kPa)",
            fontsize=FONT_SMALL - 2, fontfamily="Arial", color=C_WET,
            va="top", ha="left")
    ax.text(mmax * 0.7, y_anno, f"Est. meas.\nmax\n({mmax:.0f} kPa)",
            fontsize=FONT_SMALL - 2, fontfamily="Arial", color=C_DRY,
            va="top", ha="right")

    # Axes
    XTICKS  = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
    XLABELS = ["0.1", "1", "10", "100", "1 000", "10 000", "100 000", "1 000 000"]
    ax.set_xscale("log")
    ax.set_xticks(XTICKS)
    ax.set_xticklabels(XLABELS, fontsize=FONT_TICK, fontfamily="Arial",
                       rotation=0, ha="center")
    ax.set_xlim(psi_min, psi_max)
    ax.set_xlabel("Matric suction \u03c8  (kPa)",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)
    ax.set_ylabel("Water content \u03b8  (m\u00b3/m\u00b3)",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)
    _legend(ax, loc="lower left", ncol=1)
    _style(ax)
    _panel_tag(ax, f"(e)  Representative curve — interpolation vs extrapolation regions  "
                   f"(sample {sample_idx + 1})")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    print("Loading data...")
    y_test  = np.load(ROOT / "data_pinn_normalized" /
                      "y_test_original.npy").astype(np.float64)
    psi     = np.load(ROOT / "data_pinn_normalized" /
                      "suction_grid.npy").astype(np.float64)

    y_vg = np.load(ROOT / "results_pinn_fixed" / "vgparamnet" /
                   "run_B" / "theta_vgparamnet_test.npy").astype(np.float64)

    # Load extrapolation stats from saved JSON (heuristic estimates)
    sens_json = json.load(open(ROOT / "results_analysis" / "grid_sensitivity" /
                               "grid_sensitivity_results.json"))
    extrap     = {k: np.array(v) for k, v in
                  sens_json["extrapolation_stats"].items()}
    dry_extrap    = extrap["dry_extrapolation"]
    measured_min  = extrap["measured_min"]
    measured_max  = extrap["measured_max"]

    n = min(len(y_vg), len(y_test))
    y_vg, y_test = y_vg[:n], y_test[:n]
    print(f"  {n} test samples")

    # ── RMSE by extent (Run B) ────────────────────────────────────────────────
    extents = [1e3, 1e4, 1e5, 1e6]
    rmse_means_ext   = []
    rmse_medians_ext = []
    for ext in extents:
        arr = _rmse_by_extent(y_vg, y_test, psi, ext)
        rmse_means_ext.append(float(np.mean(arr)))
        rmse_medians_ext.append(float(np.median(arr)))
        print(f"  Extent {ext:.0e} kPa: mean={rmse_means_ext[-1]:.4f} "
              f"median={rmse_medians_ext[-1]:.4f}")

    # ── RMSE by resolution (Run B) ────────────────────────────────────────────
    n_pts_list = [50, 75, 100, 150, 200]
    rmse_means_res   = []
    rmse_medians_res = []
    for n_pts in n_pts_list:
        m, med = _rmse_at_resolution(y_vg, y_test, psi, n_pts)
        rmse_means_res.append(m)
        rmse_medians_res.append(med)
        print(f"  Resolution {n_pts} pts: mean={m:.4f} median={med:.4f}")

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14.0, 9.0))
    gs  = mgridspec.GridSpec(2, 3, figure=fig,
                             height_ratios=[1.0, 1.0],
                             width_ratios=[1.0, 1.0, 1.1],
                             hspace=0.54, wspace=0.40,
                             left=0.08, right=0.98,
                             top=0.93, bottom=0.10)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1:3])   # spans cols 1-2

    _plot_panel_a(ax_a, dry_extrap[:n])
    _plot_panel_b(ax_b, measured_max[:n])
    _plot_panel_c(ax_c, extents, rmse_means_ext, rmse_medians_ext)
    _plot_panel_d(ax_d, n_pts_list, rmse_means_res, rmse_medians_res)
    _plot_panel_e(ax_e, psi, y_test, measured_min[:n], measured_max[:n])

    # ── Save ──────────────────────────────────────────────────────────────────
    stem    = "Figure_S7_Grid_Sensitivity_Analysis_q1"
    pdf_out = SUPP_DIR / f"{stem}.pdf"
    png_out = SUPP_DIR / "png" / f"{stem}.png"
    png_out.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(str(pdf_out), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(str(png_out), dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"\nSaved:\n  {pdf_out}\n  {png_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
