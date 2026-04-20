#!/usr/bin/env python3
"""
Figure 15 — Sparse-Data Performance
Q1 journal quality (7.0 in wide × 9.0 in tall, 3 panels vertically stacked).

Layout: 3 rows × 1 column  (same format as Figures 13 and 14)
  (a) Example SWCC with 8 sparse ψ measurement points (line plot, log x-axis)
  (b) Per-sample RMSE evaluated at 8 sparse ψ points — boxplots for GB and PINN
  (c) RMSE ratio (sparse / full curve) — boxplots for GB and PINN

Models: Gradient Boosting | MonotonicPINN
Data:   latest trained models (best PINN keras checkpoint, GB retrained from data_processed)

Design (matches Figures 13–14)
------
* 7.0 in wide × 9.0 in tall (3 × 3.0 in rows)
* Arial 12 pt: axis labels, tick labels, panel tags
* Arial 10 pt: legend
* Colors: GB = #1F77B4 (blue), PINN = #D62728 (red), Observed = #000000 (black)
* Panel tags outside top-left; no grid; inward ticks mirrored; clean box
* 600 dpi PNG (→ paper_figures/png/) + PDF (→ paper_figures/)
* pdf.fonttype = 42
"""

from __future__ import annotations
import json as _json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parents[2]
PNG_DIR = ROOT / "paper_figures" / "png"
PDF_DIR = ROOT / "paper_figures"
PNG_DIR.mkdir(parents=True, exist_ok=True)
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

C_OBS  = "#000000"   # black  — Observed
C_GB   = "#1F77B4"   # blue   — Gradient Boosting
C_PINN = "#D62728"   # red    — MonotonicPINN
C_VG   = "#9B59B6"   # purple — VGParamNet (Run B)  [matches Fig 16/17]

XTICKS  = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
XLABELS = ["0.1", "1", "10", "100", "1 000", "10 000", "100 000", "1 000 000"]


# ---------------------------------------------------------------------------
# Style helpers (identical to Figures 13–14)
# ---------------------------------------------------------------------------
def _style(ax, *, log_x: bool = False, bar: bool = False):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("black")
    mirror = not bar
    ax.tick_params(which="both", top=mirror, right=mirror,
                   direction="in", labelsize=FONT_MAIN)
    ax.grid(False)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("Arial")
        lbl.set_fontsize(FONT_MAIN)


def _panel_tag(ax, tag):
    ax.text(0.0, 1.03, tag, transform=ax.transAxes,
            fontsize=FONT_MAIN, fontweight="normal", fontfamily="Arial",
            va="bottom", ha="left", clip_on=False)


def _legend(ax, **kw):
    leg = ax.legend(frameon=False, borderpad=0.4, handlelength=2.0,
                    fontsize=FONT_SMALL, **kw)
    leg.set_zorder(3)
    for t in leg.get_texts():
        t.set_fontfamily("Arial")
        t.set_fontsize(FONT_SMALL)
    return leg


def _boxplot_style(bp, colors):
    """Apply consistent colour/style to a patch_artist boxplot."""
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.55)
        patch.set_linewidth(0.8)
    for elem in ["whiskers", "caps"]:
        for line in bp[elem]:
            line.set_linewidth(0.8)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.5)
    for fl in bp["fliers"]:
        fl.set_marker("o")
        fl.set_markersize(3)
        fl.set_markerfacecolor("none")
        fl.set_markeredgewidth(0.6)


# ---------------------------------------------------------------------------
# Panel plotters
# ---------------------------------------------------------------------------
def _plot_panel_a(ax, psi, y_true, y_gb, y_pinn, y_vg, idx_example, idx_sparse):
    """(a) Example SWCC showing observed, GB, PINN, VGParamNet and 8 sparse points."""
    # Observed — black solid line
    ax.semilogx(psi, y_true[idx_example],
                color=C_OBS, lw=2.0, ls="-",
                label="Observed", zorder=3)

    # GB — blue dashed
    ax.semilogx(psi, y_gb[idx_example],
                color=C_GB, lw=LW, ls="--", dashes=(7, 3),
                label="Gradient Boosting", zorder=2)

    # PINN — red solid
    ax.semilogx(psi, y_pinn[idx_example],
                color=C_PINN, lw=LW, ls="-",
                label="MonotonicPINN", zorder=2)

    # VGParamNet — purple dash-dot
    ax.semilogx(psi, y_vg[idx_example],
                color=C_VG, lw=LW, ls="--", dashes=(3, 3),
                label="VGParamNet (Run B)", zorder=2)

    # 8 sparse measurement points — black open circles
    ax.scatter(psi[idx_sparse], y_true[idx_example, idx_sparse],
               color=C_OBS, s=35, zorder=5, marker="o",
               label="Sparse measurements (8 pts)")

    ax.set_xlim([float(psi.min()) * 0.9, float(psi.max()) * 1.1])
    ax.set_xscale("log")
    ax.set_xticks(XTICKS)
    ax.set_xticklabels(XLABELS, fontsize=FONT_MAIN - 1, fontfamily="Arial",
                       rotation=0, ha="center")
    ax.set_xlabel("Matric suction \u03c8  (kPa)", fontsize=FONT_MAIN,
                  fontfamily="Arial", labelpad=6)
    ax.set_ylabel("Water content \u03b8  [-]", fontsize=FONT_MAIN,
                  fontfamily="Arial", labelpad=6)

    _style(ax, log_x=True)
    _panel_tag(ax, "(a) Example SWCC with 8 sparse \u03c8 measurement points")
    _legend(ax, loc="upper right")


def _plot_panel_b(ax, rmse_gb_sparse, rmse_pinn_sparse, rmse_vg_sparse):
    """(b) Per-sample RMSE at 8 sparse points — boxplots (GB, PINN, VGParamNet)."""
    bp = ax.boxplot([rmse_gb_sparse, rmse_pinn_sparse, rmse_vg_sparse],
                    positions=[1, 2, 3], widths=0.55,
                    patch_artist=True, showfliers=True,
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8),
                    flierprops=dict(marker="o", markersize=3,
                                   markerfacecolor="none", markeredgewidth=0.6))
    _boxplot_style(bp, [C_GB, C_PINN, C_VG])

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Gradient\nBoosting", "Monotonic\nPINN", "VGParamNet"],
                       fontsize=FONT_MAIN, fontfamily="Arial")
    ax.set_xlim(0.3, 3.7)
    ax.set_ylabel("RMSE at 8 sparse points  [-]", fontsize=FONT_MAIN,
                  fontfamily="Arial", labelpad=6)

    _style(ax, bar=True)
    _panel_tag(ax, "(b) Sparse-data error (RMSE at 8 \u03c8 points)")


def _plot_panel_c(ax, ratio_gb, ratio_pinn, ratio_vg):
    """(c) RMSE ratio sparse/full — boxplots (GB, PINN, VGParamNet)."""
    bp = ax.boxplot([ratio_gb, ratio_pinn, ratio_vg],
                    positions=[1, 2, 3], widths=0.55,
                    patch_artist=True, showfliers=True,
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8),
                    flierprops=dict(marker="o", markersize=3,
                                   markerfacecolor="none", markeredgewidth=0.6))
    _boxplot_style(bp, [C_GB, C_PINN, C_VG])

    # Reference line at ratio = 1 (sparse == full performance)
    ax.axhline(1.0, color="#888888", linewidth=0.8, linestyle="--",
               dashes=(5, 3), zorder=0)
    ax.text(3.6, 1.01, "ratio = 1", fontsize=FONT_SMALL - 1,
            fontfamily="Arial", color="#888888", va="bottom", ha="right")

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Gradient\nBoosting", "Monotonic\nPINN", "VGParamNet"],
                       fontsize=FONT_MAIN, fontfamily="Arial")
    ax.set_xlim(0.3, 3.7)
    ax.set_ylabel("RMSE ratio  (sparse / full)  [-]", fontsize=FONT_MAIN,
                  fontfamily="Arial", labelpad=6)

    _style(ax, bar=True)
    _panel_tag(ax, "(c) Sparse vs full-curve RMSE ratio")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    from training_pinn.config_pinn_fixed import DATA_CONFIG

    # ---- Load test data ----
    X_test = pd.read_csv(DATA_CONFIG["test_file"])
    y_true = np.load(DATA_CONFIG["y_test_original_file"]).astype(np.float32)
    psi    = np.load(DATA_CONFIG["suction_grid_file"]).astype(np.float32)
    theta_s = X_test["theta_s"].values.astype(np.float32)
    theta_r = X_test["theta_r"].values.astype(np.float32)

    # ---- Gradient Boosting ----
    print("Training Gradient Boosting...")
    from baseline_models import BaselineModels
    bm = BaselineModels(data_dir="data_processed", output_dir="results_baseline")
    (Xtr, Xva, Xte), (ytr, yva, yte), _ = bm.load_data()
    Xtr_f, Xva_f, Xte_f, _ = bm.prepare_features(Xtr, Xva, Xte)
    gb_models = bm.train_gradient_boosting(Xtr_f, ytr, Xva_f, yva)
    y_gb = bm.predict_swcc(gb_models, Xte_f, model_type="gradient_boosting",
                           n_points=y_true.shape[1]).astype(np.float32)
    print(f"  GB RMSE = {float(np.sqrt(np.mean((y_true - y_gb)**2))):.4f}")

    # ---- MonotonicPINN ----
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
               "suction":    np.tile(psi, (j - i, 1)).astype(np.float32)}
        y_norm.extend(pinn_model(inp, training=False).numpy())
    y_norm = np.array(y_norm, dtype=np.float32)
    y_pinn = np.zeros_like(y_norm)
    for i in range(len(X_test)):
        y_pinn[i] = theta_r[i] + y_norm[i] * (theta_s[i] - theta_r[i])
    print(f"  PINN RMSE = {float(np.sqrt(np.mean((y_true - y_pinn)**2))):.4f}")

    # ---- VGParamNet Run B ----
    vg_path = ROOT / "results_pinn_fixed" / "vgparamnet" / "run_B" / "theta_vgparamnet_test.npy"
    y_vg = np.load(vg_path).astype(np.float32)
    print(f"  VGParamNet RMSE = {float(np.sqrt(np.mean((y_true - y_vg)**2))):.4f}")

    # ---- Sparse-data metrics (8 evenly spaced points on log ψ grid) ----
    n_pts = y_true.shape[1]
    idx_sparse = np.linspace(0, n_pts - 1, 8, dtype=int)

    rmse_gb_sparse   = np.sqrt(np.mean((y_true[:, idx_sparse] -
                                        y_gb[:, idx_sparse])**2, axis=1))
    rmse_pinn_sparse = np.sqrt(np.mean((y_true[:, idx_sparse] -
                                        y_pinn[:, idx_sparse])**2, axis=1))
    rmse_vg_sparse   = np.sqrt(np.mean((y_true[:, idx_sparse] -
                                        y_vg[:, idx_sparse])**2, axis=1))
    rmse_gb_full     = np.sqrt(np.mean((y_true - y_gb)**2, axis=1))
    rmse_pinn_full   = np.sqrt(np.mean((y_true - y_pinn)**2, axis=1))
    rmse_vg_full     = np.sqrt(np.mean((y_true - y_vg)**2, axis=1))
    ratio_gb         = rmse_gb_sparse   / np.maximum(rmse_gb_full,   1e-8)
    ratio_pinn       = rmse_pinn_sparse / np.maximum(rmse_pinn_full, 1e-8)
    ratio_vg         = rmse_vg_sparse   / np.maximum(rmse_vg_full,   1e-8)

    # Example sample: median PINN sparse RMSE
    idx_example = int(np.argsort(rmse_pinn_sparse)[len(rmse_pinn_sparse) // 2])
    print(f"\nSparse stats (8 pts):")
    print(f"  GB         sparse RMSE median = {np.median(rmse_gb_sparse):.4f}  "
          f"ratio median = {np.median(ratio_gb):.3f}")
    print(f"  PINN       sparse RMSE median = {np.median(rmse_pinn_sparse):.4f}  "
          f"ratio median = {np.median(ratio_pinn):.3f}")
    print(f"  VGParamNet sparse RMSE median = {np.median(rmse_vg_sparse):.4f}  "
          f"ratio median = {np.median(ratio_vg):.3f}")
    print(f"  Example sample idx = {idx_example}")

    # ---- Build figure: 3 rows × 1 column ----
    FIG_W  = 7.0
    ROW_H  = 3.0
    HSPACE = 0.55

    fig, axes = plt.subplots(3, 1,
                             figsize=(FIG_W, ROW_H * 3),
                             gridspec_kw=dict(hspace=HSPACE))

    _plot_panel_a(axes[0], psi, y_true, y_gb, y_pinn, y_vg, idx_example, idx_sparse)
    _plot_panel_b(axes[1], rmse_gb_sparse, rmse_pinn_sparse, rmse_vg_sparse)
    _plot_panel_c(axes[2], ratio_gb, ratio_pinn, ratio_vg)

    fig.align_ylabels(axes)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.97, bottom=0.06)

    stem = "Figure15_Sparse_Data_Performance_q1"
    fig.savefig(str(PDF_DIR / stem) + ".pdf", dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(str(PNG_DIR / stem) + ".png", dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"\nSaved:\n  {PDF_DIR / stem}.pdf\n  {PNG_DIR / stem}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
