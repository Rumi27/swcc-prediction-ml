#!/usr/bin/env python3
"""
Generate Physics Compliance and Sparse-Data Robustness Figures:
- Figure 14: Physics violation statistics (composite + Figure14a–c panels)
- Figure 15: Sparse-data scenario performance (composite + Figure15a–c panels)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

from sklearn.metrics import mean_squared_error, mean_absolute_error

from baseline_models import BaselineModels
from models.pinn_monotonic import MonotonicPINN
from models.pinn import PhysicsEncodingLayer
from training_pinn.config_pinn_fixed import DATA_CONFIG

plt.style.use("seaborn-v0_8-paper")
# Match larger, paper-ready fonts used in performance figures
plt.rcParams["font.size"] = 14
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 14

output_dir = Path("paper_figures")
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("Generating Physics Compliance & Sparse-Data Figures (Figures 14–15)")
print("=" * 80)


def load_pinn_predictions():
    """Load best PINN model and compute test predictions."""
    print("\nLoading PINN main model and test data...")
    metadata = json.load(open(DATA_CONFIG["metadata_file"]))
    X_test = pd.read_csv(DATA_CONFIG["test_file"])
    y_test_original = np.load(DATA_CONFIG["y_test_original_file"])
    suction_grid = np.load(DATA_CONFIG["suction_grid_file"])

    theta_s_test = X_test["theta_s"].values
    theta_r_test = X_test["theta_r"].values
    feature_cols = metadata["feature_cols"]

    best_model_path = Path("results_pinn_fixed/checkpoints/pinn_best_model_fixed.keras")

    model = MonotonicPINN(
        soil_prop_dim=metadata["n_features"],
        suction_points=metadata["n_swcc_points"],
        physics_units=128,
        hidden_dims=[128, 256, 128, 64],
    )
    dummy_soil = np.random.randn(1, metadata["n_features"]).astype(np.float32)
    dummy_suction = np.random.randn(1, metadata["n_swcc_points"]).astype(np.float32)
    _ = model({"soil_props": dummy_soil, "suction": dummy_suction})

    saved_model = tf.keras.models.load_model(
        str(best_model_path),
        custom_objects={"MonotonicPINN": MonotonicPINN, "PhysicsEncodingLayer": PhysicsEncodingLayer},
        compile=False,
    )
    model.set_weights(saved_model.get_weights())

    print("  Making PINN predictions on test set...")
    y_pred_norm = []
    batch_size = 32
    for i in range(0, len(X_test), batch_size):
        batch_end = min(i + batch_size, len(X_test))
        batch_soil = X_test.iloc[i:batch_end][feature_cols].values.astype(np.float32)
        batch_suction = np.tile(suction_grid, (batch_end - i, 1)).astype(np.float32)
        inputs = {"soil_props": batch_soil, "suction": batch_suction}
        theta_pred_norm_batch = model(inputs, training=False)
        y_pred_norm.extend(theta_pred_norm_batch.numpy())

    y_pred_norm = np.array(y_pred_norm)

    y_pinn = np.zeros_like(y_pred_norm)
    for i in range(len(X_test)):
        theta_range = theta_s_test[i] - theta_r_test[i]
        y_pinn[i] = theta_r_test[i] + y_pred_norm[i] * theta_range

    return X_test, y_test_original, y_pinn, suction_grid


def load_baseline_predictions():
    """Train Gradient Boosting baseline on processed data and predict test."""
    print("\nTraining Gradient Boosting baseline and predicting on test set...")
    bm = BaselineModels(data_dir="data_processed", output_dir="results_baseline")
    (X_train, X_val, X_test), (y_train, y_val, y_test), suction_grid = bm.load_data()
    X_train_feat, X_val_feat, X_test_feat, feature_cols = bm.prepare_features(X_train, X_val, X_test)

    gb_models = bm.train_gradient_boosting(X_train_feat, y_train, X_val_feat, y_val)
    y_gb = bm.predict_swcc(gb_models, X_test_feat, model_type="gradient_boosting", n_points=y_test.shape[1])

    return X_test, y_test, y_gb, suction_grid, feature_cols


import tensorflow as tf  # noqa: E402


# Load predictions
X_test_pinn, y_true, y_pinn, suction_grid = load_pinn_predictions()
X_test_gb, y_true_gb, y_gb, suction_grid_gb, feature_cols_gb = load_baseline_predictions()

assert y_true.shape == y_true_gb.shape, "Mismatch in test shapes between PINN and GB"
assert np.allclose(suction_grid, suction_grid_gb), "Suction grids differ between PINN and GB"


def non_monotone_rate(y):
    """Compute fraction of curves with any positive slope segment."""
    diff = np.diff(y, axis=1)
    violations = (diff > 0).any(axis=1)
    return violations.mean(), violations


def boundary_errors(y, theta_s, theta_r):
    """Compute absolute errors at first and last points vs θ_s and θ_r."""
    err0 = np.abs(y[:, 0] - theta_s)
    errmax = np.abs(y[:, -1] - theta_r)
    return err0, errmax


def max_positive_slope(y, suction):
    """Maximum positive slope ∂θ/∂s per curve."""
    # Approximate derivative wrt log10(s) for better scaling
    s = suction
    log_s = np.log10(s + 1e-6)
    dy = np.diff(y, axis=1)
    ds = np.diff(log_s)[None, :]
    slopes = dy / ds
    max_pos = np.maximum(slopes, 0).max(axis=1)
    return max_pos


theta_s_test = X_test_pinn["theta_s"].values
theta_r_test = X_test_pinn["theta_r"].values

# Physics stats
rate_gb, violations_gb = non_monotone_rate(y_gb)
rate_pinn, violations_pinn = non_monotone_rate(y_pinn)

err0_gb, errmax_gb = boundary_errors(y_gb, theta_s_test, theta_r_test)
err0_pinn, errmax_pinn = boundary_errors(y_pinn, theta_s_test, theta_r_test)

max_pos_gb = max_positive_slope(y_gb, suction_grid)
max_pos_pinn = max_positive_slope(y_pinn, suction_grid)

# =============================================================================
# Figure 14 – Physics violation statistics
# Same typography as Figures 11–13: Arial 11 pt, no grid; composite + standalone panels
# =============================================================================
print("\nGenerating Figure 14: Physics violation statistics...")

_F14 = 11


def _f14_spines_ticks_arial(ax):
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Arial")


def _plot_f14_panel_a(ax, rate_gb, rate_pinn):
    models = ["GB", "PINN"]
    rates = [rate_gb * 100.0, rate_pinn * 100.0]
    colors = ["#2E86AB", "#FF6B6B"]
    ax.bar(models, rates, color=colors, edgecolor="black", linewidth=1.5)
    # Single-line y-label (avoids awkward wrapping that starts a line with “% …”)
    ax.set_ylabel("Non-monotone curves (%)", fontsize=_F14, fontfamily="Arial", labelpad=10)
    ax.set_title("(a) Monotonicity of predicted θ(ψ)", fontsize=_F14, fontfamily="Arial", pad=10)
    ax.tick_params(labelsize=_F14)
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    # Headroom for value labels; note box upper-right (PINN bar short) avoids covering GB bar
    _ymax = max(rates) * 1.18 + 4
    ax.set_ylim(0, _ymax)
    for i, v in enumerate(rates):
        ax.text(
            i,
            v + max(rates) * 0.02,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=_F14,
            fontfamily="Arial",
            fontweight="bold",
        )
    if rate_gb > 0.5:
        ax.text(
            0.98,
            0.97,
            "Non-monotone θ(ψ) implies invalid K(ψ)\n(VG–K relationship; see analysis)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=_F14 - 1,
            fontfamily="Arial",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.55, edgecolor="0.5", linewidth=0.6),
        )


def _plot_f14_panel_b(ax, err0_gb, err0_pinn, errmax_gb, errmax_pinn):
    """Boxplots: |θ_pred − θ_s| at minimum ψ (wet end) and |θ_pred − θ_r| at maximum ψ (dry end)."""
    data = [err0_gb, err0_pinn, errmax_gb, errmax_pinn]
    positions = [1, 2, 4, 5]
    # Clear labels: wet = low ψ end vs θ_s; dry = high ψ end vs θ_r (same suction grid for all models)
    labels = [
        f"GB\n(wet vs \u03b8_s)",
        f"PINN\n(wet vs \u03b8_s)",
        f"GB\n(dry vs \u03b8_r)",
        f"PINN\n(dry vs \u03b8_r)",
    ]
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        labels=labels,
    )
    for patch, color in zip(box["boxes"], ["#2E86AB", "#FF6B6B", "#2E86AB", "#FF6B6B"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel("Absolute error in \u03b8 (\u2212)", fontsize=_F14, fontfamily="Arial", labelpad=10)
    ax.set_title("(b) SWCC endpoints vs \u03b8_s and \u03b8_r", fontsize=_F14, fontfamily="Arial", pad=10)
    ax.set_xlabel(
        "Wet end = minimum \u03c8 on grid; dry end = maximum \u03c8 on grid",
        fontsize=_F14 - 1,
        fontfamily="Arial",
        labelpad=12,
    )
    ax.tick_params(labelsize=_F14)
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    for tick in ax.get_xticklabels():
        tick.set_rotation(22)
        tick.set_fontsize(_F14 - 1)
        tick.set_fontfamily("Arial")


def _plot_f14_panel_c(ax, max_pos_gb):
    bins = np.linspace(0, max(max_pos_gb.max(), 1e-6), 25)
    ax.hist(max_pos_gb, bins=bins, color="#2E86AB", alpha=0.7, edgecolor="black", label="GB")
    ax.axvline(0.0, color="#FF6B6B", linestyle="--", linewidth=2, label="PINN (0 by design)")
    ax.set_xlabel("Max positive slope dθ/d log10(s)", fontsize=_F14, fontfamily="Arial", labelpad=10)
    ax.set_ylabel("Frequency", fontsize=_F14, fontfamily="Arial", labelpad=10)
    ax.set_title("(c) Maximum positive slope", fontsize=_F14, fontfamily="Arial", pad=10)
    ax.tick_params(labelsize=_F14)
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    leg = ax.legend(fontsize=_F14)
    for t in leg.get_texts():
        t.set_fontfamily("Arial")


# --- Composite figure ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
_plot_f14_panel_a(axes[0], rate_gb, rate_pinn)
_plot_f14_panel_b(axes[1], err0_gb, err0_pinn, errmax_gb, errmax_pinn)
_plot_f14_panel_c(axes[2], max_pos_gb)

for ax in axes:
    _f14_spines_ticks_arial(ax)

plt.tight_layout()
plt.savefig(output_dir / "Figure14_Physics_Violation_Statistics.png", dpi=300, bbox_inches="tight")
plt.savefig(output_dir / "Figure14_Physics_Violation_Statistics.pdf", dpi=300, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved: {output_dir / 'Figure14_Physics_Violation_Statistics.png'}")

# --- Standalone panels (a)–(c) ---
print("\nGenerating Figure 14 panels (a)–(c) separately (Arial 11 pt, no grid)...")

_fig, _ax = plt.subplots(figsize=(6.5, 5.0))
_fig.patch.set_facecolor("white")
_plot_f14_panel_a(_ax, rate_gb, rate_pinn)
_f14_spines_ticks_arial(_ax)
plt.tight_layout()
for _stem in ("Figure14a_PanelA",):
    plt.savefig(output_dir / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.savefig(output_dir / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
plt.close()

_fig, _ax = plt.subplots(figsize=(6.5, 5.0))
_fig.patch.set_facecolor("white")
_plot_f14_panel_b(_ax, err0_gb, err0_pinn, errmax_gb, errmax_pinn)
_f14_spines_ticks_arial(_ax)
plt.tight_layout()
for _stem in ("Figure14b_PanelB",):
    plt.savefig(output_dir / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.savefig(output_dir / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
plt.close()

_fig, _ax = plt.subplots(figsize=(6.5, 5.0))
_fig.patch.set_facecolor("white")
_plot_f14_panel_c(_ax, max_pos_gb)
_f14_spines_ticks_arial(_ax)
plt.tight_layout()
for _stem in ("Figure14c_PanelC",):
    plt.savefig(output_dir / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.savefig(output_dir / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
plt.close()

print(f"  ✓ Saved: Figure14a_PanelA … Figure14c_PanelC (Arial {_F14} pt) → {output_dir}")

# =============================================================================
# Figure 15 – Sparse-data scenario performance
# Same typography as Figures 11–14: Arial 11 pt, no grid; (a) Matric suction ψ + log ticks
# =============================================================================
print("\nGenerating Figure 15: Sparse-data scenario performance...")

_F15 = 11
# Match Figure 3a / Figure 11 log ψ axis
_F15_XMIN = float(np.min(suction_grid))
_F15_XMAX = float(np.max(suction_grid))
_F15_XTICKS = np.array([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
_F15_XTICKLABELS = [
    "0.1",
    "1.0",
    "10",
    "100",
    "1000",
    "10000",
    "100000",
    "1000000",
]


def _f15_log_psi_axes_swcc(ax):
    """Matric suction ψ (kPa), water content θ, decade ticks — same as Figure 11 / Figure3a."""
    ax.set_xlabel("Matric suction \u03c8 (kPa)", fontsize=_F15, fontfamily="Arial", labelpad=10)
    ax.set_ylabel("Water content (\u03b8)", fontsize=_F15, fontfamily="Arial", labelpad=10)
    ax.set_xlim([_F15_XMIN, _F15_XMAX])
    ax.set_xscale("log")
    ax.set_xticks(_F15_XTICKS)
    ax.set_xticklabels(_F15_XTICKLABELS)
    for _lbl in ax.get_xticklabels():
        _lbl.set_fontsize(10)
        _lbl.set_fontfamily("Arial")


def _plot_f15_panel_a(ax, idx_example, idx_sparse):
    ax.semilogx(suction_grid, y_true[idx_example], "k-", linewidth=2.5, label="Observed")
    ax.semilogx(suction_grid, y_gb[idx_example], color="#2E86AB", linestyle="--", linewidth=2, label="GB")
    ax.semilogx(suction_grid, y_pinn[idx_example], color="#FF6B6B", linestyle="-.", linewidth=2, label="PINN")
    ax.scatter(
        suction_grid[idx_sparse],
        y_true[idx_example, idx_sparse],
        color="black",
        s=30,
        zorder=5,
        label="Measured points (8)",
    )
    ax.set_title("(a) Example SWCC with 8 sparse \u03c8 points", fontsize=_F15, fontfamily="Arial", pad=10)
    ax.tick_params(labelsize=_F15)
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    leg = ax.legend(fontsize=_F15)
    for t in leg.get_texts():
        t.set_fontfamily("Arial")
    _f15_log_psi_axes_swcc(ax)


def _plot_f15_panel_b(ax, rmse_gb_sparse, rmse_pinn_sparse):
    box = ax.boxplot(
        [rmse_gb_sparse, rmse_pinn_sparse],
        labels=["GB", "PINN"],
        patch_artist=True,
    )
    for patch, color in zip(box["boxes"], ["#2E86AB", "#FF6B6B"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel("RMSE on 8 measured points (\u2212)", fontsize=_F15, fontfamily="Arial", labelpad=10)
    ax.set_title("(b) Sparse-data error (8 \u03c8 points)", fontsize=_F15, fontfamily="Arial", pad=10)
    ax.set_xlabel("Model", fontsize=_F15, fontfamily="Arial", labelpad=10)
    ax.tick_params(labelsize=_F15)
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(_F15)
        tick.set_fontfamily("Arial")


def _plot_f15_panel_c(ax, ratio_gb, ratio_pinn):
    box = ax.boxplot(
        [ratio_gb, ratio_pinn],
        labels=["GB", "PINN"],
        patch_artist=True,
    )
    for patch, color in zip(box["boxes"], ["#2E86AB", "#FF6B6B"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel("RMSE ratio (sparse / full) (\u2212)", fontsize=_F15, fontfamily="Arial", labelpad=10)
    ax.set_title("(c) Sparse vs full-curve RMSE", fontsize=_F15, fontfamily="Arial", pad=10)
    ax.set_xlabel("Model", fontsize=_F15, fontfamily="Arial", labelpad=10)
    ax.tick_params(labelsize=_F15)
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(_F15)
        tick.set_fontfamily("Arial")


# Choose 8 suction points (sparse measurements)
n_points = y_true.shape[1]
idx_sparse = np.linspace(0, n_points - 1, 8, dtype=int)


def per_sample_rmse_at_indices(y_true, y_pred, idxs):
    diff = y_true[:, idxs] - y_pred[:, idxs]
    return np.sqrt(np.mean(diff ** 2, axis=1))


rmse_gb_sparse = per_sample_rmse_at_indices(y_true, y_gb, idx_sparse)
rmse_pinn_sparse = per_sample_rmse_at_indices(y_true, y_pinn, idx_sparse)

rmse_gb_full = np.sqrt(np.mean((y_true - y_gb) ** 2, axis=1))
rmse_pinn_full = np.sqrt(np.mean((y_true - y_pinn) ** 2, axis=1))
ratio_gb = rmse_gb_sparse / np.maximum(rmse_gb_full, 1e-8)
ratio_pinn = rmse_pinn_sparse / np.maximum(rmse_pinn_full, 1e-8)

idx_example = int(np.argsort(rmse_pinn_sparse)[len(rmse_pinn_sparse) // 2])

# --- Composite figure ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
_plot_f15_panel_a(axes[0], idx_example, idx_sparse)
_plot_f15_panel_b(axes[1], rmse_gb_sparse, rmse_pinn_sparse)
_plot_f15_panel_c(axes[2], ratio_gb, ratio_pinn)

for ax in axes:
    _f14_spines_ticks_arial(ax)

plt.tight_layout()
plt.savefig(output_dir / "Figure15_Sparse_Data_Performance.png", dpi=300, bbox_inches="tight")
plt.savefig(output_dir / "Figure15_Sparse_Data_Performance.pdf", dpi=300, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved: {output_dir / 'Figure15_Sparse_Data_Performance.png'}")

# --- Standalone panels (a)–(c) ---
print("\nGenerating Figure 15 panels (a)–(c) separately (Arial 11 pt, no grid)...")

_fig, _ax = plt.subplots(figsize=(6.5, 5.0))
_fig.patch.set_facecolor("white")
_plot_f15_panel_a(_ax, idx_example, idx_sparse)
_f14_spines_ticks_arial(_ax)
plt.tight_layout()
for _stem in ("Figure15a_PanelA",):
    plt.savefig(output_dir / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.savefig(output_dir / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
plt.close()

_fig, _ax = plt.subplots(figsize=(6.5, 5.0))
_fig.patch.set_facecolor("white")
_plot_f15_panel_b(_ax, rmse_gb_sparse, rmse_pinn_sparse)
_f14_spines_ticks_arial(_ax)
plt.tight_layout()
for _stem in ("Figure15b_PanelB",):
    plt.savefig(output_dir / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.savefig(output_dir / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
plt.close()

_fig, _ax = plt.subplots(figsize=(6.5, 5.0))
_fig.patch.set_facecolor("white")
_plot_f15_panel_c(_ax, ratio_gb, ratio_pinn)
_f14_spines_ticks_arial(_ax)
plt.tight_layout()
for _stem in ("Figure15c_PanelC",):
    plt.savefig(output_dir / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.savefig(output_dir / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
plt.close()

print(f"  ✓ Saved: Figure15a_PanelA … Figure15c_PanelC (Arial {_F15} pt) → {output_dir}")

print("\nDone. Figures 14–15 generated in:", output_dir)

