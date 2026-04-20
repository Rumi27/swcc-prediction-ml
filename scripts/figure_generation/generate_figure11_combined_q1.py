#!/usr/bin/env python3
"""
Figure 11 — Representative SWCC Predictions (Set 1 + Set 2 combined)
Q1 journal quality (double-column width, Arial 14 pt labels/ticks, Arial 11 pt legend).

Layout: 3 rows × 2 columns
  (a) Sand         (sand ≈ 88 %)     | (b) Clay           (clay ≈ 76 %)
  (c) Sandy loam   (sand ≈ 65 %)     | (d) Silty clay      (clay ≈ 48 %, silt ≈ 51 %)
  (e) Silt loam    (silt ≈ 76 %)     | (f) Outlier — highest VGParamNet error

Left column (a, c, e)  = coarse → fine texture progression  (Set 1)
Right column (b, d, f) = fine-textured soils + outlier        (Set 2)

Shared log x-axis within each column; x-label + tick labels only on bottom row.
Legend only in panel (a); all other panels share the same line styles/colors.

Design
------
* 7.0 in wide (178 mm, double-column)
* Row height 2.2 in, 3 rows → total ≈ 7.2 in
* Arial 14 pt: axis labels, tick labels, panel tags
* Arial 11 pt: legend entries (panel a only)
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

FONT        = 14
FONT_LEGEND = 11
LW          = 1.8

matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "font.size": FONT,
    "axes.labelsize": FONT,
    "xtick.labelsize": FONT,
    "ytick.labelsize": FONT,
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

C_OBS  = "#000000"
C_GB   = "#2166AC"
C_VG   = "#9B59B6"
C_PINN = "#D6604D"

XTICKS  = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
XLABELS = ["0.1", "1", "10", "100", "1000", "10000", "100000", "1000000"]


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------
def _style(ax, *, bottom: bool):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("black")
    ax.tick_params(which="both", top=True, right=True, direction="in", labelsize=FONT)
    if not bottom:
        ax.tick_params(axis="x", labelbottom=False)
    ax.grid(False)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("Arial")
        lbl.set_fontsize(FONT)


def _x_setup(ax, *, bottom: bool, xmin, xmax):
    ax.set_xlim([xmin, xmax])
    ax.set_xscale("log")
    ax.set_xticks(XTICKS)
    if bottom:
        ax.set_xticklabels(XLABELS, fontsize=FONT - 2, fontfamily="Arial",
                           rotation=40, ha="right", rotation_mode="anchor")
        ax.set_xlabel("Matric suction \u03c8  (kPa)", fontsize=FONT,
                      fontfamily="Arial", labelpad=22)
    else:
        ax.set_xticklabels([])


def _panel_tag(ax, tag):
    t = ax.text(0.03, 0.97, tag, transform=ax.transAxes,
                fontsize=FONT, fontweight="normal", fontfamily="Arial",
                va="top", ha="left", zorder=10)
    t.set_clip_on(False)


def _plot_curves(ax, psi, y_obs, y_gb, y_vg, y_pinn):
    """Draw the four SWCC lines — no legend."""
    ax.semilogx(psi, y_obs,  color=C_OBS,  lw=2.0,  ls="-",
                label="Observed")
    ax.semilogx(psi, y_gb,   color=C_GB,   lw=LW,   ls="--",
                dashes=(5, 3), label="Gradient Boosting")
    ax.semilogx(psi, y_vg,   color=C_VG,   lw=LW,   ls=":",
                label="VGParamNet (Run B)")
    ax.semilogx(psi, y_pinn, color=C_PINN, lw=LW,   ls="-.",
                dashes=(5, 2, 1, 2), label="MonotonicPINN")


def _add_legend(ax):
    leg = ax.legend(frameon=True, edgecolor="#555555", facecolor="white",
                    framealpha=1.0, borderpad=0.4, handlelength=1.8,
                    fontsize=FONT_LEGEND, loc="upper right")
    leg.set_zorder(3)
    for t in leg.get_texts():
        t.set_fontfamily("Arial")
        t.set_fontsize(FONT_LEGEND)
    return leg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    from training_pinn.config_pinn_fixed import DATA_CONFIG

    X_test  = pd.read_csv(DATA_CONFIG["test_file"])
    y_true  = np.load(DATA_CONFIG["y_test_original_file"]).astype(np.float32)
    psi     = np.load(DATA_CONFIG["suction_grid_file"]).astype(np.float32)

    # VGParamNet Run B
    y_vg = np.load(
        ROOT / "results_pinn_fixed/vgparamnet/run_B/theta_vgparamnet_test.npy"
    ).astype(np.float32)

    # Gradient Boosting
    from baseline_models import BaselineModels
    bm = BaselineModels(data_dir="data_processed", output_dir="results_baseline")
    (Xtr, Xva, Xte), (ytr, yva, yte), _ = bm.load_data()
    Xtr_f, Xva_f, Xte_f, _ = bm.prepare_features(Xtr, Xva, Xte)
    gb_models = bm.train_gradient_boosting(Xtr_f, ytr, Xva_f, yva)
    y_gb = bm.predict_swcc(gb_models, Xte_f, model_type="gradient_boosting",
                           n_points=y_true.shape[1]).astype(np.float32)

    # MonotonicPINN
    import tensorflow as tf
    from models.pinn_monotonic import MonotonicPINN
    from models.pinn import PhysicsEncodingLayer
    meta      = _json.load(open(DATA_CONFIG["metadata_file"]))
    feat_cols = meta["feature_cols"]

    pinn_model = MonotonicPINN(
        soil_prop_dim=meta["n_features"], suction_points=meta["n_swcc_points"],
        physics_units=128, hidden_dims=[128, 256, 128, 64])
    pinn_model({"soil_props": np.random.randn(1, meta["n_features"]).astype(np.float32),
                "suction":    np.random.randn(1, meta["n_swcc_points"]).astype(np.float32)})
    saved = tf.keras.models.load_model(
        str(ROOT / "results_pinn_fixed/checkpoints/pinn_best_model_fixed.keras"),
        custom_objects={"MonotonicPINN": MonotonicPINN,
                        "PhysicsEncodingLayer": PhysicsEncodingLayer}, compile=False)
    pinn_model.set_weights(saved.get_weights())

    y_norm = []
    for i in range(0, len(X_test), 32):
        j = min(i + 32, len(X_test))
        inp = {"soil_props": X_test.iloc[i:j][feat_cols].values.astype(np.float32),
               "suction":    np.tile(psi, (j - i, 1)).astype(np.float32)}
        y_norm.extend(pinn_model(inp, training=False).numpy())
    y_norm  = np.array(y_norm, dtype=np.float32)
    ts = X_test["theta_s"].values.astype(np.float32)
    tr = X_test["theta_r"].values.astype(np.float32)
    y_pinn = np.zeros_like(y_norm)
    for i in range(len(X_test)):
        y_pinn[i] = tr[i] + y_norm[i] * (ts[i] - tr[i])

    # ------------------------------------------------------------------
    # Sample indices  (verified distinct)
    # ------------------------------------------------------------------
    sand = X_test["sand_pct"].values
    silt = X_test["silt_pct"].values
    clay = X_test["clay_pct"].values
    ps_rmse_vg = np.sqrt(np.mean((y_true - y_vg) ** 2, axis=1))

    def _first(mask):
        idx = np.where(mask)[0]
        return int(idx[0]) if len(idx) else None

    # Set 1 — left column
    idx_a = _first((sand > 70) & (clay < 15))                       # Sand
    idx_c = _first((sand > 50) & (sand < 70) & (clay < 20))         # Sandy loam
    idx_e = _first((silt > 50) & (clay < 27))                       # Silt loam

    # Set 2 — right column
    idx_b = _first(clay > 60)                                        # Clay (pure)
    idx_d = _first((clay > 35) & (silt > 40) &                      # Silty clay (distinct)
                   np.arange(len(X_test)) != idx_b)
    idx_f = int(np.argmax(ps_rmse_vg))                               # Outlier

    panels = [
        (idx_a, f"(a) Sand  (sand={sand[idx_a]:.0f}%)"),
        (idx_b, f"(b) Clay  (clay={clay[idx_b]:.0f}%)"),
        (idx_c, f"(c) Sandy loam  (sand={sand[idx_c]:.0f}%)"),
        (idx_d, f"(d) Silty clay  (clay={clay[idx_d]:.0f}%, silt={silt[idx_d]:.0f}%)"),
        (idx_e, f"(e) Silt loam  (silt={silt[idx_e]:.0f}%)"),
        (idx_f, f"(f) Outlier  (RMSE={ps_rmse_vg[idx_f]:.3f})"),
    ]

    print(f"VGParamNet global RMSE (Run B): {float(np.sqrt(np.mean((y_true-y_vg)**2))):.4f}")
    for _, (idx, lbl) in enumerate(panels):
        print(f"  {lbl}  idx={idx}")

    xmin, xmax = float(psi.min()), float(psi.max())

    # ------------------------------------------------------------------
    # Figure: 3 rows × 2 columns
    # ------------------------------------------------------------------
    FIG_W   = 7.0
    ROW_H   = 2.3
    HSPACE  = 0.12
    WSPACE  = 0.36   # room for y-axis labels on right column

    fig, axes = plt.subplots(
        3, 2,
        figsize=(FIG_W, ROW_H * 3 + 0.9),
        gridspec_kw=dict(hspace=HSPACE, wspace=WSPACE),
    )

    for row in range(3):
        for col in range(2):
            panel_idx = row * 2 + col
            idx, tag  = panels[panel_idx]
            ax        = axes[row, col]
            is_bottom = (row == 2)

            _plot_curves(ax, psi,
                         y_true[idx], y_gb[idx], y_vg[idx], y_pinn[idx])

            ax.set_ylabel("Water content \u03b8  [-]",
                          fontsize=FONT, fontfamily="Arial", labelpad=4)
            _x_setup(ax, bottom=is_bottom, xmin=xmin, xmax=xmax)
            _style(ax, bottom=is_bottom)
            _panel_tag(ax, tag)

            # Legend only in panel (a) — top-left
            if row == 0 and col == 0:
                _add_legend(ax)

    fig.align_ylabels(axes[:, 0])
    fig.align_ylabels(axes[:, 1])
    fig.subplots_adjust(left=0.11, right=0.98, top=0.98, bottom=0.11)

    stem = OUTPUT_DIR / "Figure11_Representative_SWCCs_Combined"
    fig.savefig(str(stem) + ".png", dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(str(stem) + ".pdf", dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved:\n  {stem}.png\n  {stem}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
