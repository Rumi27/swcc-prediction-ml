#!/usr/bin/env python3
"""
Figure 10 — Global Error Metrics and Distributions
Q1 journal quality (double-column width, Arial — same sizes as Figure 9).

Layout: 3 rows × 1 column (vertical stack, shared nothing)
  (a) Grouped bar chart — Global RMSE / MAE / R² for GB, VGParamNet, MonotonicPINN
  (b) Overlapping histograms — per-sample RMSE distribution (all 3 models)
  (c) Boxplots — per-sample MAE distribution (all 3 models)

Design
------
* 7.0 in wide (178 mm, double-column), ~7.5 in tall
* Arial 12 pt labels, ticks, panel tags; 10 pt legends and bar value labels (matches Figure 9)
* Panel tags (a)–(c) inside top-left, regular weight
* Inward ticks mirrored top/right; no grid; clean box
* Colorblind-friendly palette:
    GB           #2166AC  blue
    VGParamNet   #9B59B6  purple
    MonotonicPINN #D6604D  red/orange
* 600 dpi PNG + PDF, pdf.fonttype=42
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Global font / style
# ---------------------------------------------------------------------------
FONT_MAIN = 12   # axis labels, ticks, panel tags (match generate_figure9_q1.py)
FONT_SMALL = 10  # legends, numeric labels on bars

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

# Colors
C_GB   = "#2166AC"   # blue   — Gradient Boosting
C_VG   = "#9B59B6"   # purple — VGParamNet (Run B)
C_PINN = "#D6604D"   # red    — MonotonicPINN

LW = 1.5


# ---------------------------------------------------------------------------
# Helper: set box spines and tick mirroring
# ---------------------------------------------------------------------------
def _style(ax):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("black")
        sp.set_visible(True)
    ax.tick_params(which="both", top=True, right=True, direction="in", labelsize=FONT_MAIN)
    ax.grid(False)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("Arial")
        lbl.set_fontsize(FONT_MAIN)


def _panel_tag(ax, tag, x=0.03, y=0.97):
    t = ax.text(x, y, tag, transform=ax.transAxes,
                fontsize=FONT_MAIN, fontweight="normal", fontfamily="Arial",
                va="top", ha="left", zorder=10)
    t.set_clip_on(False)
    return t


def _legend(ax, **kw):
    leg = ax.legend(
        frameon=True,
        edgecolor="#555555",
        facecolor="white",
        framealpha=1.0,
        borderpad=0.4,
        handlelength=1.5,
        fontsize=FONT_SMALL,
        **kw,
    )
    leg.set_zorder(3)
    for t in leg.get_texts():
        t.set_fontfamily("Arial")
        t.set_fontsize(FONT_SMALL)
    return leg


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data():
    from training_pinn.config_pinn_fixed import DATA_CONFIG

    metadata = json.load(open(DATA_CONFIG["metadata_file"]))
    feature_cols = metadata["feature_cols"]

    X_test  = pd.read_csv(DATA_CONFIG["test_file"])
    y_test  = np.load(DATA_CONFIG["y_test_original_file"]).astype(np.float32)
    suction = np.load(DATA_CONFIG["suction_grid_file"]).astype(np.float32)

    theta_s = X_test["theta_s"].values.astype(np.float32)
    theta_r = X_test["theta_r"].values.astype(np.float32)

    return X_test, y_test, suction, theta_s, theta_r, feature_cols, metadata


def load_pinn(X_test, suction, theta_s, theta_r, feature_cols, metadata):
    import tensorflow as tf
    from models.pinn_monotonic import MonotonicPINN
    from models.pinn import PhysicsEncodingLayer

    model = MonotonicPINN(
        soil_prop_dim=metadata["n_features"],
        suction_points=metadata["n_swcc_points"],
        physics_units=128,
        hidden_dims=[128, 256, 128, 64],
    )
    dummy_soil    = np.random.randn(1, metadata["n_features"]).astype(np.float32)
    dummy_suction = np.random.randn(1, metadata["n_swcc_points"]).astype(np.float32)
    model({"soil_props": dummy_soil, "suction": dummy_suction})

    saved = tf.keras.models.load_model(
        str(ROOT / "results_pinn_fixed/checkpoints/pinn_best_model_fixed.keras"),
        custom_objects={"MonotonicPINN": MonotonicPINN,
                        "PhysicsEncodingLayer": PhysicsEncodingLayer},
        compile=False,
    )
    model.set_weights(saved.get_weights())

    y_norm = []
    for i in range(0, len(X_test), 32):
        j = min(i + 32, len(X_test))
        inp = {
            "soil_props": X_test.iloc[i:j][feature_cols].values.astype(np.float32),
            "suction":    np.tile(suction, (j - i, 1)).astype(np.float32),
        }
        y_norm.extend(model(inp, training=False).numpy())

    y_norm = np.array(y_norm, dtype=np.float32)
    y_pinn = np.zeros_like(y_norm)
    for i in range(len(X_test)):
        y_pinn[i] = theta_r[i] + y_norm[i] * (theta_s[i] - theta_r[i])
    return y_pinn


def load_gb(y_test_shape):
    from baseline_models import BaselineModels
    bm = BaselineModels(data_dir="data_processed", output_dir="results_baseline")
    (X_tr, X_va, X_te), (y_tr, y_va, y_te), _ = bm.load_data()
    X_tr_f, X_va_f, X_te_f, _ = bm.prepare_features(X_tr, X_va, X_te)
    gb_models = bm.train_gradient_boosting(X_tr_f, y_tr, X_va_f, y_va)
    y_gb = bm.predict_swcc(gb_models, X_te_f, model_type="gradient_boosting",
                           n_points=y_test_shape[1])
    return y_gb.astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def global_metrics(y_true, y_pred):
    yt, yp = y_true.ravel(), y_pred.ravel()
    mask   = ~np.isnan(yt)
    yt, yp = yt[mask], yp[mask]
    rmse   = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae    = float(np.mean(np.abs(yt - yp)))
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return rmse, mae, r2


def per_sample_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))
    mae  = np.mean(np.abs(y_true - y_pred), axis=1)
    return rmse.astype(np.float32), mae.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("Loading data...")
    X_test, y_test, suction, theta_s, theta_r, feature_cols, metadata = load_data()

    print("Loading MonotonicPINN predictions...")
    y_pinn = load_pinn(X_test, suction, theta_s, theta_r, feature_cols, metadata)

    print("Training Gradient Boosting...")
    y_gb = load_gb(y_test.shape)

    print("Loading VGParamNet Run B predictions...")
    y_vg = np.load(ROOT / "results_pinn_fixed/vgparamnet/run_B/theta_vgparamnet_test.npy").astype(np.float32)

    # Sanity check shapes
    assert y_pinn.shape == y_test.shape == y_gb.shape == y_vg.shape, \
        f"Shape mismatch: PINN={y_pinn.shape} GB={y_gb.shape} VG={y_vg.shape} true={y_test.shape}"
    print(f"All shapes OK: {y_test.shape}")

    # Compute metrics
    rmse_gb,   mae_gb,   r2_gb   = global_metrics(y_test, y_gb)
    rmse_vg,   mae_vg,   r2_vg   = global_metrics(y_test, y_vg)
    rmse_pinn, mae_pinn, r2_pinn = global_metrics(y_test, y_pinn)

    ps_rmse_gb,  ps_mae_gb   = per_sample_metrics(y_test, y_gb)
    ps_rmse_vg,  ps_mae_vg   = per_sample_metrics(y_test, y_vg)
    ps_rmse_pinn, ps_mae_pinn = per_sample_metrics(y_test, y_pinn)

    print(f"GB:             RMSE={rmse_gb:.4f}  MAE={mae_gb:.4f}  R2={r2_gb:.4f}")
    print(f"VGParamNet (B): RMSE={rmse_vg:.4f}  MAE={mae_vg:.4f}  R2={r2_vg:.4f}")
    print(f"MonotonicPINN:  RMSE={rmse_pinn:.4f}  MAE={mae_pinn:.4f}  R2={r2_pinn:.4f}")

    # -----------------------------------------------------------------------
    # Figure: 3 stacked panels
    # -----------------------------------------------------------------------
    FIG_W   = 7.0
    PANEL_H = 2.4
    HSPACE  = 0.35   # slightly more room than Figure 8/9 because no shared x-axis

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        3, 1,
        figsize=(FIG_W, PANEL_H * 3 + 0.5),
        gridspec_kw=dict(hspace=HSPACE),
    )

    # -------------------------------------------------------------------
    # Panel (a) — Global metrics bar chart
    # -------------------------------------------------------------------
    metric_labels = ["RMSE", "MAE", "R\u00b2"]
    gb_vals   = [rmse_gb,   mae_gb,   r2_gb]
    vg_vals   = [rmse_vg,   mae_vg,   r2_vg]
    pinn_vals = [rmse_pinn, mae_pinn, r2_pinn]

    x     = np.arange(len(metric_labels))
    w     = 0.22
    gap   = 0.02

    bars_gb   = ax_a.bar(x - w - gap, gb_vals,   w, label="Gradient Boosting", color=C_GB,
                         edgecolor="white", linewidth=0.5)
    bars_vg   = ax_a.bar(x,            vg_vals,   w, label="VGParamNet (Run B)", color=C_VG,
                         edgecolor="white", linewidth=0.5)
    bars_pinn = ax_a.bar(x + w + gap,  pinn_vals, w, label="MonotonicPINN",     color=C_PINN,
                         edgecolor="white", linewidth=0.5)

    # Value labels on bars
    for bars, vals in [(bars_gb, gb_vals), (bars_vg, vg_vals), (bars_pinn, pinn_vals)]:
        for bar, val in zip(bars, vals):
            ypos = bar.get_height() + 0.005
            ax_a.text(bar.get_x() + bar.get_width() / 2, ypos,
                      f"{val:.3f}", ha="center", va="bottom",
                      fontsize=FONT_SMALL, fontfamily="Arial")

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(metric_labels, fontsize=FONT_MAIN, fontfamily="Arial")
    ax_a.set_ylabel("Value  [-]", fontsize=FONT_MAIN, fontfamily="Arial", labelpad=5)
    ax_a.set_ylim([0, max(max(gb_vals), max(vg_vals), max(pinn_vals)) * 1.25])
    _style(ax_a)
    _panel_tag(ax_a, "(a)")
    # Legend upper-left: data x = GB RMSE bar centre (0.035), higher in axes = nearer top
    _gb_rmse_x = float(x[0] - w - gap)
    _tfm_a = blended_transform_factory(ax_a.transData, ax_a.transAxes)
    _leg_a = _legend(
        ax_a,
        loc="upper left",
        bbox_to_anchor=(_gb_rmse_x, 0.995),
        bbox_transform=_tfm_a,
    )
    _leg_a.set_clip_on(False)

    # -------------------------------------------------------------------
    # Panel (b) — Per-sample RMSE histogram
    # -------------------------------------------------------------------
    all_rmse = np.concatenate([ps_rmse_gb, ps_rmse_vg, ps_rmse_pinn])
    bins = np.linspace(0, np.percentile(all_rmse, 98), 22)

    ax_b.hist(ps_rmse_gb,   bins=bins, alpha=0.55, color=C_GB,   edgecolor="none",
              label="Gradient Boosting")
    ax_b.hist(ps_rmse_vg,   bins=bins, alpha=0.55, color=C_VG,   edgecolor="none",
              label="VGParamNet (Run B)")
    ax_b.hist(ps_rmse_pinn, bins=bins, alpha=0.55, color=C_PINN, edgecolor="none",
              label="MonotonicPINN")

    # Median lines
    for arr, color in [(ps_rmse_gb, C_GB), (ps_rmse_vg, C_VG), (ps_rmse_pinn, C_PINN)]:
        ax_b.axvline(np.median(arr), color=color, lw=1.2, linestyle="--", dashes=(4, 3))

    ax_b.set_xlabel("Per-sample RMSE  [-]", fontsize=FONT_MAIN, fontfamily="Arial", labelpad=4)
    ax_b.set_ylabel("Count  [-]", fontsize=FONT_MAIN, fontfamily="Arial", labelpad=5)
    _style(ax_b)
    _panel_tag(ax_b, "(b)")
    _legend(ax_b, loc="upper right")

    # -------------------------------------------------------------------
    # Panel (c) — Per-sample MAE boxplot
    # -------------------------------------------------------------------
    bp = ax_c.boxplot(
        [ps_mae_gb, ps_mae_vg, ps_mae_pinn],
        labels=["GB", "VGParamNet\n(Run B)", "MonotonicPINN"],
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        flierprops=dict(marker="o", markersize=3, linestyle="none", alpha=0.5),
        boxprops=dict(linewidth=0.8),
    )
    for patch, color in zip(bp["boxes"], [C_GB, C_VG, C_PINN]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    for flier, color in zip(bp["fliers"], [C_GB, C_VG, C_PINN]):
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)

    ax_c.set_ylabel("Per-sample MAE  [-]", fontsize=FONT_MAIN, fontfamily="Arial", labelpad=5)
    _style(ax_c)
    # boxplot x-tick labels need resetting after _style resets fontsize
    ax_c.set_xticklabels(["GB", "VGParamNet\n(Run B)", "MonotonicPINN"],
                         fontsize=FONT_MAIN, fontfamily="Arial")
    _panel_tag(ax_c, "(c)")

    # -----------------------------------------------------------------------
    # Align y-labels and save
    # -----------------------------------------------------------------------
    fig.align_ylabels([ax_a, ax_b, ax_c])
    fig.subplots_adjust(left=0.14, right=0.97, top=0.97, bottom=0.08)

    stem = OUTPUT_DIR / "Figure10_Global_Metrics_and_Distributions"
    fig.savefig(str(stem) + ".png", dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(str(stem) + ".pdf", dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved:\n  {stem}.png\n  {stem}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
