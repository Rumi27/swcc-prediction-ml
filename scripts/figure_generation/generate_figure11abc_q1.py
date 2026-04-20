#!/usr/bin/env python3
"""
Figure 11 — Representative SWCC Predictions (six separate single-panel figures)

Q1 journal quality: each SWCC is its own figure (no multi-panel layout).
Naming: Figure11_final_separate_01 … Figure11_final_separate_06 (.png / .pdf)

Order: (a) Sand, (b) Sandy loam, (c) Silt loam, (d) Clay, (e) Silty clay, (f) Outlier

Design (matches journal reference style — screenshot 2026-04-10):
  Observed        → open black circles (markers only, no line)
  MonotonicPINN   → solid red line     (physics reference)
  Gradient Boosting → dashed blue line
  VGParamNet (Run B) → dash-dot green line

* 7.0 in × 5.25 in per figure (extra width/height reduces x-tick overlap)
* Arial 12 pt axis labels and tick labels (x and y match); 10 pt legend (no frame)
* Descriptive title centered above the axes (outside plot), e.g. Sand (sand = 88%)
* Panel letter (a)–(f) below axes, outside plot, bottom-left aligned with axes
* Log x-axis 0.1 – 1 000 000 kPa; horizontal x tick labels
* Inward ticks mirrored top/right; no grid; clean box
* 600 dpi PNG + PDF, pdf.fonttype=42
"""

from __future__ import annotations
import json as _json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT       = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

FONT_MAIN  = 12   # axis labels, x/y tick labels (same size), title, panel letter
FONT_SMALL = 10   # legend only
LW         = 1.8

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

# Colors — priority: black, red, blue, green
C_OBS  = "#000000"   # black — Observed (open circle markers)
C_PINN = "#D62728"   # red   — MonotonicPINN solid line
C_GB   = "#1F77B4"   # blue  — Gradient Boosting dashed line
C_VG   = "#2CA02C"   # green — VGParamNet (Run B) dashed line

XTICKS  = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
XLABELS = ["0.1", "1", "10", "100", "1 000", "10 000", "100 000", "1 000 000"]

# Observed markers: subsample along index (log-uniform ψ grid → even spacing on semilog axis)
N_OBS_MARKERS = 26


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------
def _panel_tag(ax, tag: str) -> None:
    """Panel tag outside top-left of axes (axes coordinates, y=1.03)."""
    ax.text(0.0, 1.03, tag,
            transform=ax.transAxes,
            fontsize=FONT_MAIN, fontweight="normal", fontfamily="Arial",
            va="bottom", ha="left", clip_on=False)


def _legend(ax, **kw):
    leg = ax.legend(
        frameon=False,
        borderpad=0.4,
        handlelength=2.0,
        fontsize=FONT_SMALL,
        **kw,
    )
    leg.set_zorder(3)
    for t in leg.get_texts():
        t.set_fontfamily("Arial")
        t.set_fontsize(FONT_SMALL)
    return leg


def _plot_panel(ax, psi, y_obs, y_gb, y_vg, y_pinn, *, xmin, xmax):
    """
    Draw all four SWCC series:
      - Observed       : open black circles (markers only, like lab measurements)
      - MonotonicPINN  : solid red line   (physics-based reference)
      - Gradient Boost : blue dashed line
      - VGParamNet     : green dash-dot line
    """
    # --- Model curves first (so observed markers stay visible on top) ---
    # --- MonotonicPINN: red solid line ---
    ax.semilogx(psi, y_pinn, color=C_PINN, lw=LW + 0.4, ls="-",
                label="MonotonicPINN", zorder=2)

    # --- Gradient Boosting: blue dashed line ---
    ax.semilogx(psi, y_gb, color=C_GB, lw=LW, ls="--",
                dashes=(7, 3), label="Gradient Boosting", zorder=2)

    # --- VGParamNet (Run B): green dashed line (shorter dash) ---
    ax.semilogx(psi, y_vg, color=C_VG, lw=LW, ls="--",
                dashes=(3, 3), label="VGParamNet (Run B)", zorder=2)

    # --- Observed: open circles at ~N_OBS_MARKERS points (even along log ψ; not all 100) ---
    n = len(psi)
    k = min(N_OBS_MARKERS, n)
    _idx = np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))
    ax.semilogx(
        psi[_idx], y_obs[_idx],
        color=C_OBS, marker="o", markersize=4.5,
        markerfacecolor="none", markeredgewidth=1.2, markeredgecolor=C_OBS,
        linestyle="none",
        label="Observed", zorder=5, clip_on=True,
    )

    # --- Axes ---
    ax.set_ylabel("Water content \u03b8  [-]", fontsize=FONT_MAIN,
                  fontfamily="Arial", labelpad=6)
    ax.set_xlim([xmin, xmax])
    ax.set_xscale("log")
    ax.set_xticks(XTICKS)
    ax.set_xticklabels(XLABELS, fontsize=8, fontfamily="Arial",
                       rotation=0, ha="center")
    ax.set_xlabel("Matric suction \u03c8  (kPa)", fontsize=FONT_MAIN,
                  fontfamily="Arial", labelpad=10)

    # --- Spines / ticks ---
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("black")
    ax.tick_params(
        which="both", top=True, right=True, direction="in",
        labelsize=FONT_MAIN,
        axis="both",
    )
    ax.grid(False)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("Arial")
        lbl.set_fontsize(FONT_MAIN)

    _legend(ax, loc="upper right")


def _make_single_figure(panel_letter: str, title_text: str, idx: int, stem: str) -> None:
    """Produce one figure for a single soil sample; margins sized to limit label overlap."""
    FIG_W = 7.0
    FIG_H = 5.25

    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")

    xmin = float(PSI.min())
    xmax = float(PSI.max())
    _plot_panel(ax, PSI, Y_TRUE[idx], Y_GB[idx], Y_VG[idx], Y_PINN[idx],
                xmin=xmin, xmax=xmax)

    fig.subplots_adjust(left=0.12, right=0.97, top=0.91, bottom=0.10)
    _panel_tag(ax, f"{panel_letter}  {title_text}")

    out = OUTPUT_DIR / stem
    fig.savefig(str(out) + ".png", dpi=600, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(str(out) + ".pdf", dpi=600, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {out}.png / .pdf")


# ---------------------------------------------------------------------------
# Globals filled by load_all()
# ---------------------------------------------------------------------------
PSI    = None
Y_TRUE = None
Y_GB   = None
Y_VG   = None
Y_PINN = None


def load_all():
    global PSI, Y_TRUE, Y_GB, Y_VG, Y_PINN

    from training_pinn.config_pinn_fixed import DATA_CONFIG

    X_test  = pd.read_csv(DATA_CONFIG["test_file"])
    Y_TRUE  = np.load(DATA_CONFIG["y_test_original_file"]).astype(np.float32)
    PSI     = np.load(DATA_CONFIG["suction_grid_file"]).astype(np.float32)

    # VGParamNet Run B
    Y_VG = np.load(
        ROOT / "results_pinn_fixed/vgparamnet/run_B/theta_vgparamnet_test.npy"
    ).astype(np.float32)
    print(f"VGParamNet RMSE (Run B): {float(np.sqrt(np.mean((Y_TRUE - Y_VG)**2))):.4f}")

    # Gradient Boosting (fast retrain)
    print("Training Gradient Boosting...")
    from baseline_models import BaselineModels
    bm = BaselineModels(data_dir="data_processed", output_dir="results_baseline")
    (Xtr, Xva, Xte), (ytr, yva, yte), _ = bm.load_data()
    Xtr_f, Xva_f, Xte_f, _ = bm.prepare_features(Xtr, Xva, Xte)
    gb_models = bm.train_gradient_boosting(Xtr_f, ytr, Xva_f, yva)
    Y_GB = bm.predict_swcc(gb_models, Xte_f, model_type="gradient_boosting",
                           n_points=Y_TRUE.shape[1]).astype(np.float32)

    # MonotonicPINN
    print("Loading MonotonicPINN...")
    import tensorflow as tf
    from models.pinn_monotonic import MonotonicPINN
    from models.pinn import PhysicsEncodingLayer
    meta      = _json.load(open(DATA_CONFIG["metadata_file"]))
    feat_cols = meta["feature_cols"]

    pinn_model = MonotonicPINN(
        soil_prop_dim=meta["n_features"],
        suction_points=meta["n_swcc_points"],
        physics_units=128, hidden_dims=[128, 256, 128, 64])
    pinn_model({"soil_props": np.random.randn(1, meta["n_features"]).astype(np.float32),
                "suction":    np.random.randn(1, meta["n_swcc_points"]).astype(np.float32)})
    saved = tf.keras.models.load_model(
        str(ROOT / "results_pinn_fixed/checkpoints/pinn_best_model_fixed.keras"),
        custom_objects={"MonotonicPINN": MonotonicPINN,
                        "PhysicsEncodingLayer": PhysicsEncodingLayer},
        compile=False)
    pinn_model.set_weights(saved.get_weights())

    y_norm = []
    for i in range(0, len(X_test), 32):
        j = min(i + 32, len(X_test))
        inp = {"soil_props": X_test.iloc[i:j][feat_cols].values.astype(np.float32),
               "suction":    np.tile(PSI, (j - i, 1)).astype(np.float32)}
        y_norm.extend(pinn_model(inp, training=False).numpy())
    y_norm = np.array(y_norm, dtype=np.float32)
    ts = X_test["theta_s"].values.astype(np.float32)
    tr = X_test["theta_r"].values.astype(np.float32)
    Y_PINN = np.zeros_like(y_norm)
    for i in range(len(X_test)):
        Y_PINN[i] = tr[i] + y_norm[i] * (ts[i] - tr[i])

    return X_test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    X_test = load_all()

    sand = X_test["sand_pct"].values
    silt = X_test["silt_pct"].values
    clay = X_test["clay_pct"].values
    ps_rmse_vg = np.sqrt(np.mean((Y_TRUE - Y_VG) ** 2, axis=1))

    def _first(mask):
        idx = np.where(mask)[0]
        return int(idx[0]) if len(idx) else None

    idx_sand       = _first((sand > 70) & (clay < 15))
    idx_sandy_loam = _first((sand > 50) & (sand < 70) & (clay < 20))
    idx_silt_loam  = _first((silt > 50) & (clay < 27))
    idx_clay       = _first(clay > 60)
    idx_silty_clay = _first((clay > 35) & (silt > 40) &
                            (np.arange(len(X_test)) != idx_clay))
    idx_outlier    = int(np.argmax(ps_rmse_vg))

    print(f"Sand:       idx={idx_sand}  sand={sand[idx_sand]:.0f}%")
    print(f"Sandy loam: idx={idx_sandy_loam}  sand={sand[idx_sandy_loam]:.0f}%  "
          f"silt={silt[idx_sandy_loam]:.0f}%")
    print(f"Silt loam:  idx={idx_silt_loam}  silt={silt[idx_silt_loam]:.0f}%")
    print(f"Clay:       idx={idx_clay}  clay={clay[idx_clay]:.0f}%")
    print(f"Silty clay: idx={idx_silty_clay}  clay={clay[idx_silty_clay]:.0f}%  "
          f"silt={silt[idx_silty_clay]:.0f}%")
    print(f"Outlier:    idx={idx_outlier}  RMSE={ps_rmse_vg[idx_outlier]:.4f}")

    separate = [
        (
            "(a)",
            f"Sand (sand = {sand[idx_sand]:.0f}%)",
            idx_sand,
            "Figure11_final_separate_01",
        ),
        (
            "(b)",
            f"Sandy loam (sand = {sand[idx_sandy_loam]:.0f}%, "
            f"silt = {silt[idx_sandy_loam]:.0f}%)",
            idx_sandy_loam,
            "Figure11_final_separate_02",
        ),
        (
            "(c)",
            f"Silt loam (silt = {silt[idx_silt_loam]:.0f}%)",
            idx_silt_loam,
            "Figure11_final_separate_03",
        ),
        (
            "(d)",
            f"Clay (clay = {clay[idx_clay]:.0f}%)",
            idx_clay,
            "Figure11_final_separate_04",
        ),
        (
            "(e)",
            f"Silty clay (clay = {clay[idx_silty_clay]:.0f}%, "
            f"silt = {silt[idx_silty_clay]:.0f}%)",
            idx_silty_clay,
            "Figure11_final_separate_05",
        ),
        (
            "(f)",
            f"Outlier (highest VGParamNet error, RMSE = {ps_rmse_vg[idx_outlier]:.3f})",
            idx_outlier,
            "Figure11_final_separate_06",
        ),
    ]

    for panel_letter, title_text, idx, stem in separate:
        _make_single_figure(panel_letter, title_text, idx, stem)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
