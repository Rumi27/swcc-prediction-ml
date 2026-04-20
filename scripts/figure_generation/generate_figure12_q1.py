#!/usr/bin/env python3
"""
Figure 12 — Representative SWCC Predictions (Set 2: fine-textured + outlier)
Q1 journal quality (double-column width, Arial 14 pt labels/ticks, Arial 11 pt legend).

Layout: 2 rows × 1 column (vertical stack, shared x-axis)
  (a) Clay  (idx=1, clay ≈ 41 %)
  (b) Outlier — sample with highest VGParamNet per-sample RMSE

Each panel shows: Observed | GB | VGParamNet (Run B) | MonotonicPINN

Design
------
* 7.0 in wide, 2 panels × 2.5 in each ≈ 5.8 in tall
* Arial 14 pt: axis labels, tick labels, panel tags
* Arial 11 pt: legend entries
* Shared log x-axis; x-label and tick labels only on bottom panel
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
XLABELS = ["0.1", "1", "10", "100", "1 000", "10 000", "100 000", "1 000 000"]


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


def _x_axis(ax, *, bottom: bool, xmin, xmax):
    ax.set_xlim([xmin, xmax])
    ax.set_xscale("log")
    ax.set_xticks(XTICKS)
    if bottom:
        ax.set_xticklabels(XLABELS, fontsize=FONT, fontfamily="Arial")
        ax.set_xlabel("Matric suction \u03c8  (kPa)", fontsize=FONT,
                      fontfamily="Arial", labelpad=4)
    else:
        ax.set_xticklabels([])


def _panel_tag(ax, tag):
    t = ax.text(0.03, 0.97, tag, transform=ax.transAxes,
                fontsize=FONT, fontweight="normal", fontfamily="Arial",
                va="top", ha="left", zorder=10)
    t.set_clip_on(False)


def _legend(ax, **kw):
    leg = ax.legend(frameon=True, edgecolor="#555555", facecolor="white",
                    framealpha=1.0, borderpad=0.4, handlelength=1.8,
                    fontsize=FONT_LEGEND, **kw)
    leg.set_zorder(3)
    for t in leg.get_texts():
        t.set_fontfamily("Arial")
        t.set_fontsize(FONT_LEGEND)
    return leg


def _plot_swcc(ax, psi, y_obs, y_gb, y_vg, y_pinn, title, *, bottom, xmin, xmax):
    ax.semilogx(psi, y_obs,  color=C_OBS,  lw=2.0, ls="-",  label="Observed")
    ax.semilogx(psi, y_gb,   color=C_GB,   lw=1.8, ls="--", dashes=(5, 3), label="Gradient Boosting")
    ax.semilogx(psi, y_vg,   color=C_VG,   lw=1.8, ls=":",  label="VGParamNet (Run B)")
    ax.semilogx(psi, y_pinn, color=C_PINN, lw=1.8, ls="-.", dashes=(5, 2, 1, 2), label="MonotonicPINN")
    ax.set_ylabel("Water content \u03b8  [-]", fontsize=FONT, fontfamily="Arial", labelpad=4)
    _x_axis(ax, bottom=bottom, xmin=xmin, xmax=xmax)
    _style(ax, bottom=bottom)
    _panel_tag(ax, title)
    _legend(ax, loc="upper right")


def main() -> int:
    from training_pinn.config_pinn_fixed import DATA_CONFIG

    X_test  = pd.read_csv(DATA_CONFIG["test_file"])
    y_true  = np.load(DATA_CONFIG["y_test_original_file"]).astype(np.float32)
    psi     = np.load(DATA_CONFIG["suction_grid_file"]).astype(np.float32)

    y_vg   = np.load(ROOT / "results_pinn_fixed/vgparamnet/run_B/theta_vgparamnet_test.npy").astype(np.float32)

    # GB
    from baseline_models import BaselineModels
    bm = BaselineModels(data_dir="data_processed", output_dir="results_baseline")
    (Xtr, Xva, Xte), (ytr, yva, yte), _ = bm.load_data()
    Xtr_f, Xva_f, Xte_f, _ = bm.prepare_features(Xtr, Xva, Xte)
    gb_models = bm.train_gradient_boosting(Xtr_f, ytr, Xva_f, yva)
    y_gb = bm.predict_swcc(gb_models, Xte_f, model_type="gradient_boosting",
                           n_points=y_true.shape[1]).astype(np.float32)

    # PINN
    import tensorflow as tf
    from models.pinn_monotonic import MonotonicPINN
    from models.pinn import PhysicsEncodingLayer
    meta      = _json.load(open(DATA_CONFIG["metadata_file"]))
    feat_cols = meta["feature_cols"]
    model = MonotonicPINN(soil_prop_dim=meta["n_features"],
                          suction_points=meta["n_swcc_points"],
                          physics_units=128, hidden_dims=[128, 256, 128, 64])
    model({"soil_props": np.random.randn(1, meta["n_features"]).astype(np.float32),
           "suction":    np.random.randn(1, meta["n_swcc_points"]).astype(np.float32)})
    saved = tf.keras.models.load_model(
        str(ROOT / "results_pinn_fixed/checkpoints/pinn_best_model_fixed.keras"),
        custom_objects={"MonotonicPINN": MonotonicPINN,
                        "PhysicsEncodingLayer": PhysicsEncodingLayer}, compile=False)
    model.set_weights(saved.get_weights())
    y_norm = []
    for i in range(0, len(X_test), 32):
        j = min(i + 32, len(X_test))
        inp = {"soil_props": X_test.iloc[i:j][feat_cols].values.astype(np.float32),
               "suction":    np.tile(psi, (j - i, 1)).astype(np.float32)}
        y_norm.extend(model(inp, training=False).numpy())
    y_norm  = np.array(y_norm, dtype=np.float32)
    theta_s = X_test["theta_s"].values.astype(np.float32)
    theta_r = X_test["theta_r"].values.astype(np.float32)
    y_pinn  = np.zeros_like(y_norm)
    for i in range(len(X_test)):
        y_pinn[i] = theta_r[i] + y_norm[i] * (theta_s[i] - theta_r[i])

    # Sample selection
    clay = X_test["clay_pct"].values
    ps_rmse_vg = np.sqrt(np.mean((y_true - y_vg) ** 2, axis=1))
    idx_clay   = int(np.where(clay > 40)[0][0])
    idx_out    = int(np.argmax(ps_rmse_vg))

    xmin = float(psi.min())
    xmax = float(psi.max())

    print(f"VGParamNet RMSE (should be ~0.083): {float(np.sqrt(np.mean((y_true - y_vg)**2))):.4f}")
    print(f"Panel (a): idx={idx_clay}  clay={clay[idx_clay]:.0f}%")
    print(f"Panel (b): idx={idx_out}   RMSE={ps_rmse_vg[idx_out]:.4f}  clay={clay[idx_out]:.0f}%")

    # Figure
    FIG_W   = 7.0
    PANEL_H = 2.5
    HSPACE  = 0.08

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1,
        figsize=(FIG_W, PANEL_H * 2 + 0.5),
        gridspec_kw=dict(hspace=HSPACE),
    )

    _plot_swcc(ax_a, psi,
               y_true[idx_clay], y_gb[idx_clay], y_vg[idx_clay], y_pinn[idx_clay],
               f"(a) Clay  (clay = {clay[idx_clay]:.0f}%)",
               bottom=False, xmin=xmin, xmax=xmax)

    _plot_swcc(ax_b, psi,
               y_true[idx_out], y_gb[idx_out], y_vg[idx_out], y_pinn[idx_out],
               f"(b) Outlier — highest VGParamNet error  (RMSE = {ps_rmse_vg[idx_out]:.3f})",
               bottom=True, xmin=xmin, xmax=xmax)

    fig.align_ylabels([ax_a, ax_b])
    fig.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.10)

    stem = OUTPUT_DIR / "Figure12_Representative_SWCCs_Set2"
    fig.savefig(str(stem) + ".png", dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(str(stem) + ".pdf", dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved:\n  {stem}.png\n  {stem}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
