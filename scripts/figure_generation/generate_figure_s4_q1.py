#!/usr/bin/env python3
"""
Figure S4 — Effect of Slope Penalty: Oversmoothing of SWCC Knee
Q1 journal quality (single panel, 6.5 × 5.0 in).

Compares for a sharp-knee sand sample:
  Run B  (λ_ψ50=0.2, no slope loss) — canonical best VGParamNet model
  Run C  (λ_ψ50=0.1, λ_slope=0.05) — slope-penalty variant (expected oversmoothing)
  Observed (UNSODA test set)

Design
------
* 6.5 in × 5.0 in, single panel
* Colors consistent with Figures 11–13:
    Observed        → black open circle markers
    VGParamNet Run B → green  (#2CA02C)  solid line
    VGParamNet Run C → red    (#D62728)  dashed line
* Arial 12 pt labels / ticks; Arial 10 pt legend
* Panel tag outside top-left; no grid; inward ticks mirrored; frameless legend
* 600 dpi PNG (→ paper_figures/png/) + PDF (→ paper_figures/supplementary/)
* pdf.fonttype=42
"""

from __future__ import annotations
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parents[2]
PNG_DIR  = ROOT / "paper_figures" / "png"
PDF_DIR  = ROOT / "paper_figures" / "supplementary"
PNG_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)
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

# Consistent colors with Figures 11–13
C_OBS  = "#000000"   # black  — Observed (open circle markers)
C_RUNB = "#2CA02C"   # green  — VGParamNet Run B (ψ50-only, best)
C_RUNC = "#D62728"   # red    — VGParamNet Run C (slope penalty, oversmoothed)

XTICKS  = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
XLABELS = ["0.1", "1", "10", "100", "1000", "10000", "100000", "1000000"]

N_MARKERS = 20   # observed marker count across log range


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
    leg = ax.legend(frameon=False, borderpad=0.4, handlelength=2.2,
                    fontsize=FONT_SMALL, **kw)
    leg.set_zorder(3)
    for t in leg.get_texts():
        t.set_fontfamily("Arial")
        t.set_fontsize(FONT_SMALL)
    return leg


def compute_max_slope(psi, theta):
    log_psi = np.log(np.maximum(psi, 1e-6))
    dlog    = np.diff(log_psi)
    dtheta  = np.diff(theta)
    slopes  = np.abs(dtheta / (dlog + 1e-8))
    return float(np.max(slopes)) if slopes.size else 0.0


def main() -> int:
    from training_pinn.config_pinn_fixed import DATA_CONFIG

    # ----------------------------------------------------------------
    # Load test data
    # ----------------------------------------------------------------
    X_test = pd.read_csv(DATA_CONFIG["test_file"])
    y_true = np.load(DATA_CONFIG["y_test_original_file"]).astype(np.float32)
    psi    = np.load(DATA_CONFIG["suction_grid_file"]).astype(np.float32)

    vg_dir = ROOT / "results_pinn_fixed" / "vgparamnet"

    # ----------------------------------------------------------------
    # Load Run B (canonical best: λ_ψ50=0.2, no slope loss)
    # ----------------------------------------------------------------
    cfg_b = json.load(open(vg_dir / "run_B" / "training_config.json"))
    assert float(cfg_b.get("lambda_psi50", 0)) == 0.2, "Run B lambda_psi50 mismatch"
    assert float(cfg_b.get("lambda_slope", 0)) == 0.0, "Run B lambda_slope mismatch"
    theta_B = np.load(vg_dir / "run_B" / "theta_vgparamnet_test.npy").astype(np.float32)
    rmse_B  = float(np.sqrt(np.mean((y_true - theta_B) ** 2)))
    print(f"Run B  RMSE={rmse_B:.4f}  "
          f"(lambda_psi50={cfg_b['lambda_psi50']}, lambda_slope={cfg_b['lambda_slope']})")

    # ----------------------------------------------------------------
    # Load Run C (slope-penalty comparison: λ_slope=0.05)
    # ----------------------------------------------------------------
    cfg_c = json.load(open(vg_dir / "run_C" / "training_config.json"))
    assert float(cfg_c.get("lambda_slope", 0)) > 0, "Run C should have slope loss"
    theta_C = np.load(vg_dir / "run_C" / "theta_vgparamnet_test.npy").astype(np.float32)
    rmse_C  = float(np.sqrt(np.mean((y_true - theta_C) ** 2)))
    print(f"Run C  RMSE={rmse_C:.4f}  "
          f"(lambda_psi50={cfg_c['lambda_psi50']}, lambda_slope={cfg_c['lambda_slope']})")

    # ----------------------------------------------------------------
    # Select the sand sample with the sharpest observed knee
    # ----------------------------------------------------------------
    sand_pct = X_test["sand_pct"].values if "sand_pct" in X_test.columns else None

    # Compute max log-slope for each test sample
    max_slopes = np.array([compute_max_slope(psi, y_true[i])
                           for i in range(len(y_true))])

    # Prefer sandy soils (sand > 60 %) for a clear sharp knee
    if sand_pct is not None:
        sand_mask = sand_pct > 60
        if sand_mask.any():
            candidates = np.where(sand_mask)[0]
        else:
            candidates = np.arange(len(y_true))
    else:
        candidates = np.arange(len(y_true))

    idx = int(candidates[np.argmax(max_slopes[candidates])])
    print(f"\nSelected sample idx={idx}  "
          f"sand={sand_pct[idx]:.0f}%  max_slope={max_slopes[idx]:.3f}")
    print(f"  Observed  θ range: [{y_true[idx].min():.3f}, {y_true[idx].max():.3f}]")
    print(f"  Run B     θ range: [{theta_B[idx].min():.3f}, {theta_B[idx].max():.3f}]")
    print(f"  Run C     θ range: [{theta_C[idx].min():.3f}, {theta_C[idx].max():.3f}]")

    # Subsample observed markers evenly on log scale
    n     = len(psi)
    k     = min(N_MARKERS, n)
    m_idx = np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))

    # ----------------------------------------------------------------
    # Plot
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.0))
    fig.patch.set_facecolor("white")

    xmin = float(psi.min()) * 0.9
    xmax = float(psi.max()) * 1.1

    # Run B — green solid (best model)
    ax.semilogx(psi, theta_B[idx], color=C_RUNB, lw=LW + 0.4, ls="-",
                label="VGParamNet Run B  (\u03bb=0.2, no slope loss)",
                zorder=2)

    # Run C — red dashed (slope penalty oversmoothed)
    ax.semilogx(psi, theta_C[idx], color=C_RUNC, lw=LW, ls="--",
                dashes=(7, 3),
                label="VGParamNet Run C  (\u03bb=0.1, \u03bb_slope=0.05)",
                zorder=2)

    # Observed — black solid line + open circle markers at subsampled points
    ax.semilogx(psi, y_true[idx],
                color=C_OBS, lw=2.0, ls="-",
                zorder=4, label="_nolegend_")
    ax.semilogx(psi[m_idx], y_true[idx][m_idx],
                color=C_OBS, marker="o", markersize=5,
                markerfacecolor="none", markeredgewidth=1.2,
                linestyle="none",
                label="Observed (UNSODA)",
                zorder=5)

    # Axes
    ax.set_xlim([xmin, xmax])
    ax.set_xscale("log")
    ax.set_xticks(XTICKS)
    ax.set_xticklabels(XLABELS, fontsize=FONT_MAIN - 1, fontfamily="Arial",
                       rotation=0, ha="center")
    ax.set_xlabel("Matric suction \u03c8  (kPa)", fontsize=FONT_MAIN,
                  fontfamily="Arial", labelpad=6)
    ax.set_ylabel("Water content \u03b8  [-]", fontsize=FONT_MAIN,
                  fontfamily="Arial", labelpad=6)

    _style(ax)
    _panel_tag(ax, "(a) Effect of slope penalty on SWCC knee")
    _legend(ax, loc="upper right")

    # Annotation: arrow pointing to oversmoothed knee region
    # Find the knee location of observed curve (max slope point)
    log_psi = np.log(np.maximum(psi, 1e-6))
    slopes  = np.abs(np.diff(y_true[idx]) / (np.diff(log_psi) + 1e-8))
    knee_i  = int(np.argmax(slopes))
    knee_psi = float(psi[knee_i])
    knee_y   = float(y_true[idx][knee_i])
    ax.annotate("Sharp knee\n(observed)",
                xy=(knee_psi, knee_y),
                xytext=(knee_psi * 5, knee_y + 0.05),
                fontsize=FONT_SMALL - 1, fontfamily="Arial",
                arrowprops=dict(arrowstyle="-|>", color="black",
                                lw=0.8, mutation_scale=10),
                ha="left", va="bottom", color="black")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    stem = "Figure_S4_Slope_Penalty_Oversmoothing_q1"
    fig.savefig(str(PDF_DIR / stem) + ".pdf", dpi=600, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(str(PNG_DIR / stem) + ".png", dpi=600, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"\nSaved:\n  {PDF_DIR / stem}.pdf\n  {PNG_DIR / stem}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
