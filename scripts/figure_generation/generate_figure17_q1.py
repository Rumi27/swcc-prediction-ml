#!/usr/bin/env python3
"""
Figure 17 — External Validation on GSHP Database
Q1 journal quality (wide double-column canvas, 2 panels side-by-side).

Layout: 1 row × 2 columns
  (a) Distribution of per-sample RMSE across 7,072 GSHP samples
      — histogram with Q10 / Q50 / Q90 lines + cumulative % annotations
  (b) Representative SWCCs — VGParamNet vs GSHP database
      — samples near RMSE Q50 and Q90; line styles match Figure 11 (observed = solid black,
        VGParamNet = green dashed); legend (Q50) / (Q90)

Data source (no retraining needed):
  results_gshp_validation/gshp_observed_curves.npy
  results_gshp_validation/gshp_predicted_curves.npy
  data_pinn_normalized/suction_grid.npy

Design (typography & canvas match Figure 3 Q1: generate_figure3_q1.py)
------
* 8.5 in wide × 3.4 in tall (1 row × 2 cols; same width as Fig. 3; height ≈ half of Fig. 3)
* Arial 10 pt (FONT_MAIN): axis titles, panel tags
* Arial 8 pt (FONT_TICK_Y): y-axis tick numerals on both panels
* Arial 8 pt (FONT_TICK): panel (a) x-axis tick numerals
* Arial 8 pt (FONT_TICK_XLOG): ψ x-tick numerals on panel (b), horizontal
* Arial 8 pt (FONT_LEGEND): panel (a) Q-line callouts + N/% summary; panel (b) curve legend
* Arial 10 pt (FONT_SMALL): RMSE arrow callouts only
* Panel (b) — Q50: GSHP solid black, VGParamNet green dashed (3,3) like Fig. 11
*          — Q90: GSHP red solid line + open circles (Fig. 11 markers); VGParamNet red dashed (3,3)
* Panel (a) — Q10 blue solid, Q50 green dashed, Q90 red dashed; Q90 numeric matches panel (b) RMSE text
* Panel tags outside top-left; no grid; inward ticks mirrored; clean box
* 600 dpi PNG (→ paper_figures/png/) + PDF (→ paper_figures/)
* pdf.fonttype = 42
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT    = Path(__file__).resolve().parents[2]
PNG_DIR = ROOT / "paper_figures" / "png"
PDF_DIR = ROOT / "paper_figures"
PNG_DIR.mkdir(parents=True, exist_ok=True)

FONT_MAIN  = 10   # axis titles, panel tags (match Figure 3 Q1)
FONT_TICK_Y = 8   # y-axis tick numerals — both panels
FONT_TICK  = 8    # panel (a) x-axis tick numerals
FONT_TICK_XLOG = 8   # ψ log-axis x ticks on panel (b)
FONT_SMALL = 10   # RMSE annotations on panel (b)
FONT_LEGEND = 8   # legend & legend-like keys — both panels (smaller than FONT_SMALL)
LW         = 1.8

matplotlib.rcParams.update({
    "text.usetex": False,
    "axes.formatter.use_mathtext": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "font.size": FONT_MAIN,
    "axes.labelsize": FONT_MAIN,
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK_Y,
    "legend.fontsize": FONT_LEGEND,
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

# ── Colour palette (panel b aligns with generate_figure11abc_q1.py) ────────────
C_OBS        = "#000000"   # GSHP observed — solid black (Fig. 11)
C_VG         = "#2CA02C"   # VGParamNet — green dashed dashes=(3, 3) (Fig. 11)
C_Q10        = "#1F77B4"   # panel (a) Q10 — blue (distinct from Q50 green)
C_Q50        = "#2CA02C"   # panel (a) Q50 — green dashed (ties to VGParamNet)
C_Q90        = "#C73E1D"   # panel (a) Q90 line + panel (b) Q90 series — red (high-error tail)
C_HIST       = "#5B9EBF"   # muted blue — histogram fill

# GSHP observed (Q90): markevery indices along ψ (line + open circles, Fig. 11 marker style)
N_OBS_MARKERS_Q90 = 26

XTICKS_PSI   = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
XLABELS_PSI  = ["0.1", "1", "10", "100", "1000", "10000", "100000", "1000000"]


# ── Style helpers ─────────────────────────────────────────────────────────────
def _style(ax, *, xticks_fs=FONT_TICK, yticks_fs=FONT_TICK_Y):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("black")
    ax.tick_params(which="both", top=True, right=True, direction="in")
    ax.tick_params(axis="x", labelsize=xticks_fs)
    ax.tick_params(axis="y", labelsize=yticks_fs)
    ax.grid(False)
    for lbl in ax.get_xticklabels():
        lbl.set_fontfamily("Arial")
        lbl.set_fontsize(xticks_fs)
    for lbl in ax.get_yticklabels():
        lbl.set_fontfamily("Arial")
        lbl.set_fontsize(yticks_fs)


def _panel_tag(ax, tag):
    ax.text(0.0, 1.03, tag, transform=ax.transAxes,
            fontsize=FONT_MAIN, fontweight="normal", fontfamily="Arial",
            va="bottom", ha="left", clip_on=False)


def _legend(ax, **kw):
    leg = ax.legend(
        frameon=False,
        borderpad=0.28,
        handlelength=1.9,
        fontsize=FONT_LEGEND,
        **kw,
    )
    for t in leg.get_texts():
        t.set_fontfamily("Arial")
        t.set_fontsize(FONT_LEGEND)
    return leg


# ── Panel (a): RMSE histogram ─────────────────────────────────────────────────
def _plot_panel_a(ax, rmse):
    """
    Histogram of per-sample RMSE with Q10/Q50/Q90 lines.
    Cumulative percentage annotations give the reader immediate context.
    """
    q10 = np.percentile(rmse, 10)
    q50 = np.percentile(rmse, 50)
    q90 = np.percentile(rmse, 90)

    # ---- histogram ----
    ax.hist(rmse, bins=50, density=True,
            color=C_HIST, alpha=0.80, edgecolor="white", linewidth=0.4, zorder=2)

    # ---- shade region RMSE < 0.05 ----
    ax.axvspan(0, 0.05, color=C_Q10, alpha=0.08, zorder=1, lw=0)

    # ---- Q10 (blue solid), Q50 (green dashed), Q90 (red dashed — longer dashes than Q50) ----
    ax.axvline(q10, color=C_Q10, lw=1.6, ls="-", alpha=0.95, zorder=5)
    ax.axvline(q50, color=C_Q50, lw=1.6, ls="--", dashes=(3, 3), alpha=0.95, zorder=5)
    ax.axvline(q90, color=C_Q90, lw=1.6, ls="--", dashes=(6, 3), alpha=0.95, zorder=5)

    # ---- re-get ylim after hist ----
    ax.set_ylim(bottom=0)
    y_top = ax.get_ylim()[1]

    # ---- annotated value boxes: staggered heights to avoid overlap ----
    # Q10 and Q50 are close together (0.019 vs 0.061) so we offset their y positions
    y_positions = [y_top * 0.97, y_top * 0.78, y_top * 0.97]
    for (val, col, lbl), y_pos in zip(
            [(q10, C_Q10, "Q10"), (q50, C_Q50, "Q50"), (q90, C_Q90, "Q90")],
            y_positions):
        ax.text(val, y_pos, f"{lbl} = {val:.3f}",
                ha="center", va="top", fontsize=FONT_LEGEND, fontfamily="Arial",
                color=col, fontweight="bold", zorder=7,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=col, linewidth=1.0, alpha=0.92))

    # ---- cumulative % summary (key reader-friendly statistics) ----
    pct_05  = 100 * (rmse < 0.05).mean()
    pct_10  = 100 * (rmse < 0.10).mean()
    pct_20  = 100 * (rmse < 0.20).mean()
    note = (
        f"N = {len(rmse):,}\n"
        f"RMSE < 0.05:  {pct_05:.0f}% of samples\n"
        f"RMSE < 0.10:  {pct_10:.0f}% of samples\n"
        f"RMSE < 0.20:  {pct_20:.0f}% of samples\n"
        f"\u03b1 unit: 1/m \u2192 1/kPa (\u00d70.102)"
    )
    ax.text(0.98, 0.97, note,
            transform=ax.transAxes, fontsize=FONT_LEGEND, fontfamily="Arial",
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#888888", linewidth=0.8, alpha=0.95))

    ax.set_xlabel("Per-sample RMSE  [\u2212]", fontsize=FONT_MAIN,
                  fontfamily="Arial", labelpad=5)
    ax.set_ylabel("Probability density", fontsize=FONT_MAIN,
                  fontfamily="Arial", labelpad=5)
    ax.set_xlim(left=0)

    _style(ax, xticks_fs=FONT_TICK, yticks_fs=FONT_TICK_Y)
    # histogram panel: no top/right mirrored ticks (they look odd on histogram)
    ax.tick_params(top=False, right=False)
    _panel_tag(ax, "(a) RMSE distribution  (N = 7,072)")


# ── Panel (b): representative SWCCs ──────────────────────────────────────────
def _plot_panel_b(ax, curves_pred, curves_gshp, psi, rmse):
    """
    Two SWCC pairs illustrating model fidelity.
    Y-axis: degree of saturation S = (θ − θr) / (θs − θr), per sample.
    RMSE (in θ units) annotated with arrows directly on the observed curves.
    Legend: GSHP observed vs VGParamNet predicted for RMSE near Q50 and Q90 (Fig. 11 line styles).
    """
    q50 = np.percentile(rmse, 50)
    q90 = np.percentile(rmse, 90)
    idx_med  = int(np.argmin(np.abs(rmse - q50)))
    idx_high = int(np.argmin(np.abs(rmse - q90)))
    rmse_med = float(rmse[idx_med])
    # Display same RMSE value as panel (a) Q90 label (population percentile), not nearest-sample RMSE
    rmse_q90_display = float(q90)

    # ---- normalise to degree of saturation S (per sample) ----
    # Use each sample's own observed θ range: θr ≈ min, θs ≈ max
    def _to_S(obs, pred):
        theta_r = float(obs.min())
        theta_s = float(obs.max())
        span    = max(theta_s - theta_r, 1e-6)
        return (obs - theta_r) / span, (pred - theta_r) / span

    S_obs_med,  S_pred_med  = _to_S(curves_gshp[idx_med],  curves_pred[idx_med])
    S_obs_high, S_pred_high = _to_S(curves_gshp[idx_high], curves_pred[idx_high])

    # ---- Fig. 11: solid black = GSHP observed (Q50); green dashes (3,3) = VGParamNet (Q50) ----
    ax.semilogx(psi, S_obs_med,
                color=C_OBS, lw=2.2, ls="-", zorder=4,
                label="GSHP observed (Q50)")
    ax.semilogx(psi, S_pred_med,
                color=C_VG, lw=LW, ls="--", dashes=(3, 3), zorder=3,
                label="VGParamNet predicted (Q50)")

    n_psi = len(psi)
    k_m = min(N_OBS_MARKERS_Q90, n_psi)
    _idx_m = np.unique(np.round(np.linspace(0, n_psi - 1, k_m)).astype(int))
    # Continuous observed curve + subsampled open circles (same ψ indices as before)
    ax.semilogx(
        psi, S_obs_high,
        color=C_Q90, lw=1.8, ls="-",
        marker="o", markersize=4.5, markevery=_idx_m,
        markerfacecolor="none", markeredgewidth=1.2, markeredgecolor=C_Q90,
        label="GSHP observed (Q90)", zorder=5, clip_on=True,
    )
    ax.semilogx(
        psi, S_pred_high,
        color=C_Q90, lw=LW, ls="--", dashes=(3, 3), zorder=3,
        label="VGParamNet predicted (Q90)",
    )

    # ---- RMSE annotations: arrow pointing to observed curve at S ≈ 0.5 ----
    # Median case — text offset up-left from the inflection point
    ann_idx_med = int(np.argmin(np.abs(S_obs_med - 0.5)))
    ann_psi_med = float(psi[ann_idx_med])
    ax.annotate(
        f"RMSE = {rmse_med:.3f}",
        xy     =(ann_psi_med, 0.5),
        xytext =(ann_psi_med * 0.10, 0.72),
        fontsize=FONT_SMALL, fontfamily="Arial", color=C_OBS,
        arrowprops=dict(arrowstyle="->", color=C_OBS, lw=1.0,
                        connectionstyle="arc3,rad=0.15"),
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                  edgecolor=C_OBS, linewidth=0.9, alpha=0.95),
        zorder=8,
    )

    # High-error case — text offset down-right to avoid overlap with median annotation
    ann_idx_high = int(np.argmin(np.abs(S_obs_high - 0.5)))
    ann_psi_high = float(psi[ann_idx_high])
    ax.annotate(
        f"RMSE = {rmse_q90_display:.3f}",
        xy     =(ann_psi_high, 0.5),
        xytext =(ann_psi_high * 5.0, 0.30),
        fontsize=FONT_SMALL, fontfamily="Arial", color=C_Q90,
        arrowprops=dict(arrowstyle="->", color=C_Q90, lw=1.0,
                        connectionstyle="arc3,rad=-0.15"),
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                  edgecolor=C_Q90, linewidth=0.9, alpha=0.95),
        zorder=8,
    )

    # ---- axes ----
    ax.set_xscale("log")
    ax.set_xticks(XTICKS_PSI)
    ax.set_xticklabels(
        XLABELS_PSI, fontsize=FONT_TICK_XLOG,
        fontfamily="Arial", rotation=0, ha="center",
    )
    ax.set_xlim(float(psi.min()) * 0.9, float(psi.max()) * 1.1)
    ax.set_ylim(-0.04, 1.14)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", labelsize=FONT_TICK_Y)

    ax.set_xlabel("Matric suction \u03c8  (kPa)", fontsize=FONT_MAIN,
                  fontfamily="Arial", labelpad=9)
    ax.set_ylabel("Degree of saturation  S  [\u2212]", fontsize=FONT_MAIN,
                  fontfamily="Arial", labelpad=5)

    _style(ax, xticks_fs=FONT_TICK_XLOG, yticks_fs=FONT_TICK_Y)
    _legend(ax, loc="upper right")
    _panel_tag(ax, "(b) Representative SWCCs: VGParamNet vs GSHP")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    # ---- load data ----
    obs_path  = ROOT / "results_gshp_validation" / "gshp_observed_curves.npy"
    pred_path = ROOT / "results_gshp_validation" / "gshp_predicted_curves.npy"
    psi_path  = ROOT / "data_pinn_normalized" / "suction_grid.npy"

    for p in [obs_path, pred_path, psi_path]:
        if not p.exists():
            print(f"ERROR: file not found: {p}")
            return 1

    curves_gshp = np.load(obs_path).astype(np.float32)
    curves_pred = np.load(pred_path).astype(np.float32)
    psi         = np.load(psi_path).astype(np.float32)

    rmse = np.sqrt(np.mean((curves_pred - curves_gshp) ** 2, axis=1))

    print(f"Loaded {len(rmse):,} GSHP samples")
    print(f"RMSE — Q10={np.percentile(rmse,10):.4f}  Q50={np.median(rmse):.4f}"
          f"  Q90={np.percentile(rmse,90):.4f}  max={rmse.max():.4f}")
    print(f"  RMSE < 0.05:  {100*(rmse<0.05).mean():.1f}%")
    print(f"  RMSE < 0.10:  {100*(rmse<0.10).mean():.1f}%")
    print(f"  RMSE < 0.20:  {100*(rmse<0.20).mean():.1f}%")

    # ---- build figure: 1 row × 2 columns (width / margins aligned with Figure 3 Q1) ----
    FIG_W, FIG_H = 8.5, 3.4
    fig, axes = plt.subplots(
        1, 2,
        figsize=(FIG_W, FIG_H),
        gridspec_kw=dict(wspace=0.34),
    )

    _plot_panel_a(axes[0], rmse)
    _plot_panel_b(axes[1], curves_pred, curves_gshp, psi, rmse)

    fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.14)

    stem = "Figure17_GSHP_External_Validation_q1"
    pdf_out = PDF_DIR / f"{stem}.pdf"
    png_out = PNG_DIR / f"{stem}.png"
    fig.savefig(str(pdf_out), dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(str(png_out), dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"\nSaved:\n  {pdf_out}\n  {png_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
