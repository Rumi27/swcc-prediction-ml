#!/usr/bin/env python3
"""
Figure 14 — Physics Violation Statistics
Q1 journal quality (7.0 in wide × 9.0 in tall, 3 panels vertically stacked).

Layout: 3 rows × 1 column  (same format as Figure 13)
  (a) Monotonicity: fraction of non-monotone predicted curves (bar chart)
  (b) Endpoint errors: |θ_pred − θ_s| (wet end) and |θ_pred − θ_r| (dry end) — boxplots
  (c) Maximum positive slope dθ/d log10(ψ) — histogram for GB; PINN = 0 by design

Models: Gradient Boosting | MonotonicPINN
Data:   latest trained models (best PINN keras checkpoint, GB retrained from data_processed)

Design (matches Figure 13)
------
* 7.0 in wide × 9.0 in tall (3 × 3.0 in rows)
* Arial 12 pt: axis labels, tick labels, panel tags
* Arial 10 pt: legend
* Colors: GB = #1F77B4 (blue), PINN = #D62728 (red)
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

ROOT     = Path(__file__).resolve().parents[2]
PNG_DIR  = ROOT / "paper_figures" / "png"
PDF_DIR  = ROOT / "paper_figures"
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

C_GB   = "#1F77B4"   # blue  — Gradient Boosting
C_PINN = "#D62728"   # red   — MonotonicPINN


# ---------------------------------------------------------------------------
# Style helpers (identical to Figure 13)
# ---------------------------------------------------------------------------
def _style(ax, *, bar: bool = False):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("black")
    if bar:
        ax.tick_params(which="both", top=False, right=False, direction="in",
                       labelsize=FONT_MAIN)
    else:
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
    leg = ax.legend(frameon=True, edgecolor="#555555", facecolor="white",
                    framealpha=1.0, borderpad=0.4, handlelength=2.0,
                    fontsize=FONT_SMALL, **kw)
    leg.set_zorder(3)
    for t in leg.get_texts():
        t.set_fontfamily("Arial")
        t.set_fontsize(FONT_SMALL)
    return leg


# ---------------------------------------------------------------------------
# Physics metric helpers
# ---------------------------------------------------------------------------
def non_monotone_rate(y):
    """Fraction of curves with at least one positive-slope segment."""
    violations = (np.diff(y, axis=1) > 0).any(axis=1)
    return float(violations.mean()), violations


def boundary_errors(y, theta_s, theta_r):
    err_wet = np.abs(y[:, 0]  - theta_s)   # wet end vs θ_s
    err_dry = np.abs(y[:, -1] - theta_r)   # dry end vs θ_r
    return err_wet, err_dry


def max_positive_slope(y, psi):
    log_psi = np.log10(np.maximum(psi, 1e-6))
    dy  = np.diff(y, axis=1)
    dlog = np.diff(log_psi)[None, :]
    return np.maximum(dy / (dlog + 1e-12), 0.0).max(axis=1)


# ---------------------------------------------------------------------------
# Panel plotters
# ---------------------------------------------------------------------------
def _plot_panel_a(ax, rate_gb, rate_pinn):
    """(a) Non-monotone fraction bar chart."""
    labels = ["Gradient Boosting", "MonotonicPINN"]
    values = [rate_gb * 100.0, rate_pinn * 100.0]
    colors = [C_GB, C_PINN]
    x = np.arange(len(labels))

    bars = ax.bar(x, values, color=colors, width=0.45,
                  edgecolor="black", linewidth=0.8)

    # Value labels above bars
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + max(values) * 0.02 + 0.5,
                f"{v:.1f}%",
                ha="center", va="bottom",
                fontsize=FONT_MAIN, fontfamily="Arial", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_MAIN, fontfamily="Arial")
    ax.set_ylabel("Non-monotone curves (%)", fontsize=FONT_MAIN,
                  fontfamily="Arial", labelpad=6)
    ax.set_ylim(0, max(values) * 1.25 + 5)

    # Annotation box if GB has significant violations
    if rate_gb > 0.5:
        ax.text(0.97, 0.95,
                "Non-monotone \u03b8(\u03c8) implies\ninvalid K(\u03c8) via VG-K relation",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=FONT_SMALL - 1, fontfamily="Arial",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3CD",
                          edgecolor="#999999", linewidth=0.6, alpha=0.9))

    _style(ax, bar=True)
    _panel_tag(ax, "(a) Monotonicity of predicted \u03b8(\u03c8)")


def _plot_panel_b(ax, err_wet_gb, err_wet_pinn, err_dry_gb, err_dry_pinn):
    """(b) Endpoint boundary error boxplots."""
    data      = [err_wet_gb, err_wet_pinn, err_dry_gb, err_dry_pinn]
    positions = [1, 2, 4, 5]
    colors_bp = [C_GB, C_PINN, C_GB, C_PINN]
    tick_lbls = ["GB\n(wet)", "PINN\n(wet)", "GB\n(dry)", "PINN\n(dry)"]

    bp = ax.boxplot(data, positions=positions, widths=0.55,
                    patch_artist=True, showfliers=True,
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8),
                    flierprops=dict(marker="o", markersize=3,
                                   markerfacecolor="none", markeredgewidth=0.6))
    for patch, col in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(col)
        patch.set_alpha(0.5)
        patch.set_linewidth(0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_lbls, fontsize=FONT_MAIN, fontfamily="Arial")
    ax.set_xlim(0, 6)
    ax.set_ylabel("Absolute error in \u03b8  [-]", fontsize=FONT_MAIN,
                  fontfamily="Arial", labelpad=6)
    ax.set_xlabel("Wet end = min \u03c8;   dry end = max \u03c8 on grid",
                  fontsize=FONT_SMALL, fontfamily="Arial", labelpad=6)

    # Vertical separator between wet and dry groups
    ax.axvline(3.0, color="#AAAAAA", linewidth=0.8, linestyle="--", zorder=0)
    ax.text(1.5 / 6.0, 0.97, "Wet end (\u03b8 vs \u03b8_s)",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=FONT_SMALL, fontfamily="Arial", color="#555555")
    ax.text(4.5 / 6.0, 0.97, "Dry end (\u03b8 vs \u03b8_r)",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=FONT_SMALL, fontfamily="Arial", color="#555555")

    _style(ax, bar=True)
    _panel_tag(ax, "(b) SWCC endpoint errors vs \u03b8_s and \u03b8_r")


def _plot_panel_c(ax, max_pos_gb):
    """(c) Histogram of max positive slope; PINN = 0 by design."""
    bins = np.linspace(0, max(float(max_pos_gb.max()), 1e-6), 26)
    ax.hist(max_pos_gb, bins=bins, color=C_GB, alpha=0.75,
            edgecolor="white", linewidth=0.4, label="Gradient Boosting")
    ax.axvline(0.0, color=C_PINN, linewidth=2.0, linestyle="--",
               dashes=(6, 3), label="MonotonicPINN (0 by design)")

    ax.set_xlabel("Max positive slope  d\u03b8 / d log10(\u03c8)",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)
    ax.set_ylabel("Frequency", fontsize=FONT_MAIN,
                  fontfamily="Arial", labelpad=6)

    _style(ax, bar=False)
    _panel_tag(ax, "(c) Maximum positive slope (monotonicity violations)")
    _legend(ax, loc="upper right")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    from training_pinn.config_pinn_fixed import DATA_CONFIG

    # ---- Load test data ----
    X_test  = pd.read_csv(DATA_CONFIG["test_file"])
    y_true  = np.load(DATA_CONFIG["y_test_original_file"]).astype(np.float32)
    psi     = np.load(DATA_CONFIG["suction_grid_file"]).astype(np.float32)
    theta_s = X_test["theta_s"].values.astype(np.float32)
    theta_r = X_test["theta_r"].values.astype(np.float32)

    # ---- Gradient Boosting (retrain) ----
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

    # ---- Physics metrics ----
    rate_gb,   viol_gb   = non_monotone_rate(y_gb)
    rate_pinn, viol_pinn = non_monotone_rate(y_pinn)
    err_wet_gb,   err_dry_gb   = boundary_errors(y_gb,   theta_s, theta_r)
    err_wet_pinn, err_dry_pinn = boundary_errors(y_pinn, theta_s, theta_r)
    max_pos_gb   = max_positive_slope(y_gb,   psi)
    max_pos_pinn = max_positive_slope(y_pinn, psi)

    print(f"\nPhysics stats:")
    print(f"  GB   non-monotone: {rate_gb*100:.1f}%  |  "
          f"wet err: {err_wet_gb.mean():.4f}  dry err: {err_dry_gb.mean():.4f}  "
          f"max slope: {max_pos_gb.mean():.4f}")
    print(f"  PINN non-monotone: {rate_pinn*100:.1f}%  |  "
          f"wet err: {err_wet_pinn.mean():.4f}  dry err: {err_dry_pinn.mean():.4f}  "
          f"max slope: {max_pos_pinn.mean():.4f}")

    # ---- Build figure: 3 rows × 1 column ----
    FIG_W  = 7.0
    ROW_H  = 3.0
    HSPACE = 0.55

    fig, axes = plt.subplots(3, 1,
                             figsize=(FIG_W, ROW_H * 3),
                             gridspec_kw=dict(hspace=HSPACE))

    _plot_panel_a(axes[0], rate_gb, rate_pinn)
    _plot_panel_b(axes[1], err_wet_gb, err_wet_pinn, err_dry_gb, err_dry_pinn)
    _plot_panel_c(axes[2], max_pos_gb)

    fig.align_ylabels(axes)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.97, bottom=0.06)

    stem = "Figure14_Physics_Violation_Statistics_q1"
    fig.savefig(str(PDF_DIR / stem) + ".pdf", dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(str(PNG_DIR / stem) + ".png", dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"\nSaved:\n  {PDF_DIR / stem}.pdf\n  {PNG_DIR / stem}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
