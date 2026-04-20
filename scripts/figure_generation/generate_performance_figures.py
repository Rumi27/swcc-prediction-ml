#!/usr/bin/env python3
"""
Generate Performance Comparison Figures for Paper:
- Figure 10: Global error metrics and distributions
- Figure 11: Representative SWCC predictions (set 1)
- Figure 12: Representative SWCC predictions (set 2, including outlier)
- Figure 13: Error vs suction and dry-end comparison
"""

import sys
from pathlib import Path

# Ensure project root (containing the 'models' package) is on PYTHONPATH
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

from models.pinn_monotonic import MonotonicPINN
from models.pinn import PhysicsEncodingLayer
from baseline_models import BaselineModels
from training_pinn.config_pinn_fixed import DATA_CONFIG


plt.style.use("seaborn-v0_8-paper")
# Larger, paper-ready defaults for performance figures
plt.rcParams["font.size"] = 14
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 14

output_dir = Path("paper_figures")
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("Generating Performance Comparison Figures (Figures 10–13)")
print("=" * 80)


def load_pinn_predictions():
    """Load best PINN model and compute test predictions + per-sample metrics."""
    print("\nLoading PINN main model and test data...")
    # Load metadata and test data (normalized pipeline)
    metadata = json.load(open(DATA_CONFIG["metadata_file"]))
    X_test = pd.read_csv(DATA_CONFIG["test_file"])
    y_test_original = np.load(DATA_CONFIG["y_test_original_file"])
    suction_grid = np.load(DATA_CONFIG["suction_grid_file"])

    theta_s_test = X_test["theta_s"].values
    theta_r_test = X_test["theta_r"].values
    feature_cols = metadata["feature_cols"]

    # Load best model (real-only, normalized)
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

    # Predict
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

    # Denormalize
    y_pinn = np.zeros_like(y_pred_norm)
    for i in range(len(X_test)):
        theta_range = theta_s_test[i] - theta_r_test[i]
        y_pinn[i] = theta_r_test[i] + y_pred_norm[i] * theta_range

    return X_test, y_test_original, y_pinn, suction_grid


def load_baseline_predictions():
    """Train Gradient Boosting on processed data and get test predictions."""
    print("\nTraining Gradient Boosting baseline and predicting on test set...")
    bm = BaselineModels(data_dir="data_processed", output_dir="results_baseline")
    (X_train, X_val, X_test), (y_train, y_val, y_test), suction_grid = bm.load_data()
    X_train_feat, X_val_feat, X_test_feat, feature_cols = bm.prepare_features(X_train, X_val, X_test)

    # Train only Gradient Boosting (best baseline)
    gb_models = bm.train_gradient_boosting(X_train_feat, y_train, X_val_feat, y_val)
    # Predict on test
    y_gb = bm.predict_swcc(gb_models, X_test_feat, model_type="gradient_boosting", n_points=y_test.shape[1])

    return X_test, y_test, y_gb, suction_grid, feature_cols


def load_vgparamnet_predictions():
    """Load VGParamNet Run B predictions (final model)."""
    print("\nLoading VGParamNet Run B predictions...")
    vgparamnet_path = Path("results_pinn_fixed/vgparamnet/theta_vgparamnet_test.npy")
    if vgparamnet_path.exists():
        y_vgparamnet = np.load(vgparamnet_path).astype(np.float32)
        print(f"  ✓ Loaded VGParamNet predictions: {y_vgparamnet.shape}")
        return y_vgparamnet
    else:
        print(f"  ⚠ VGParamNet predictions not found at {vgparamnet_path}")
        return None


# Import TensorFlow only after function definitions to avoid cluttered logs at import time
import tensorflow as tf  # noqa: E402


# Load predictions
X_test_pinn, y_test_pinn_ref, y_pinn, suction_grid = load_pinn_predictions()
X_test_gb, y_test_gb_ref, y_gb, suction_grid_gb, feature_cols_gb = load_baseline_predictions()
y_vgparamnet = load_vgparamnet_predictions()

# Sanity: ensure same test ordering and suction grid
assert y_test_pinn_ref.shape == y_test_gb_ref.shape, "PINN and GB test shapes differ"
assert np.allclose(suction_grid, suction_grid_gb), "Suction grids differ between PINN and GB"
if y_vgparamnet is not None:
    assert y_vgparamnet.shape == y_test_pinn_ref.shape, "VGParamNet shape mismatch"
y_true = y_test_pinn_ref

# Per-sample metrics
def per_sample_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))
    mae = np.mean(np.abs(y_true - y_pred), axis=1)
    return rmse, mae


rmse_pinn, mae_pinn = per_sample_metrics(y_true, y_pinn)
rmse_gb, mae_gb = per_sample_metrics(y_true, y_gb)

# Global metrics
def global_metrics(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    mask = ~np.isnan(y_true_flat)
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
    ss_tot = np.sum((y_true_clean - y_true_clean.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return rmse, mae, r2


rmse_gb_global, mae_gb_global, r2_gb_global = global_metrics(y_true, y_gb)
rmse_pinn_global, mae_pinn_global, r2_pinn_global = global_metrics(y_true, y_pinn)

# =============================================================================
# Figure 10 – Global error metrics and distributions
# =============================================================================
print("\nGenerating Figure 10: Global error metrics and distributions...")

_F10 = 11
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) Bar chart of global RMSE, MAE, R²
ax = axes[0]
metrics = ["RMSE", "MAE", "R²"]
gb_vals = [rmse_gb_global, mae_gb_global, r2_gb_global]
pinn_vals = [rmse_pinn_global, mae_pinn_global, r2_pinn_global]
x = np.arange(len(metrics))
width = 0.35

ax.bar(x - width / 2, gb_vals, width, label="Gradient Boosting", color="#2E86AB")
ax.bar(x + width / 2, pinn_vals, width, label="MonotonicPINN", color="#FF6B6B")
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=_F10, fontfamily="Arial")
ax.set_ylabel("Value", fontsize=_F10, fontfamily="Arial", labelpad=10)
ax.set_title("(a) Global metrics", fontsize=_F10, fontfamily="Arial", pad=10)
ax.tick_params(labelsize=_F10)
ax.grid(False)
leg10a = ax.legend(fontsize=_F10)
for text in leg10a.get_texts():
    text.set_fontfamily("Arial")

# (b) Histogram of per-sample RMSE
ax = axes[1]
bins = np.linspace(0, max(rmse_gb.max(), rmse_pinn.max()), 20)
ax.hist(rmse_gb, bins=bins, alpha=0.6, label="GB", color="#2E86AB", edgecolor="black")
ax.hist(rmse_pinn, bins=bins, alpha=0.6, label="PINN", color="#FF6B6B", edgecolor="black")
ax.set_xlabel("Per-sample RMSE", fontsize=_F10, fontfamily="Arial", labelpad=10)
ax.set_ylabel("Frequency", fontsize=_F10, fontfamily="Arial", labelpad=10)
ax.set_title("(b) Per-sample RMSE distribution", fontsize=_F10, fontfamily="Arial", pad=10)
ax.tick_params(labelsize=_F10)
ax.grid(False)
leg10b = ax.legend(fontsize=_F10)
for text in leg10b.get_texts():
    text.set_fontfamily("Arial")

# (c) Boxplot of per-sample MAE
ax = axes[2]
ax.boxplot(
    [mae_gb, mae_pinn],
    labels=["GB", "PINN"],
    patch_artist=True,
    boxprops=dict(facecolor="#2E86AB", alpha=0.4),
)
for patch, color in zip(ax.artists, ["#2E86AB", "#FF6B6B"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
ax.set_ylabel("Per-sample MAE", fontsize=_F10, fontfamily="Arial", labelpad=10)
ax.set_title("(c) Per-sample MAE distribution", fontsize=_F10, fontfamily="Arial", pad=10)
ax.tick_params(labelsize=_F10)
# Update x-axis tick labels for boxplot
ax.set_xticklabels(["GB", "PINN"], fontsize=_F10, fontfamily="Arial")
ax.grid(False)

for ax in axes:
    # Match clean white background and visible perimeter box
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)
    # Ensure tick label fonts are Arial
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Arial")

plt.tight_layout()
plt.savefig(output_dir / "Figure10_Global_Metrics_and_Distributions.png", dpi=300, bbox_inches="tight")
plt.savefig(output_dir / "Figure10_Global_Metrics_and_Distributions.pdf", dpi=300, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved: {output_dir / 'Figure10_Global_Metrics_and_Distributions.png'}")

# Figure 10 panels (a)–(c): standalone, Arial 11 pt, no grid
print("\nGenerating Figure 10 panels (a)–(c) separately (Arial 11 pt, no grid)...")


def _f10_arial(ax):
    for _lbl in ax.get_xticklabels() + ax.get_yticklabels():
        _lbl.set_fontfamily("Arial")


def _f10_spines_white(ax):
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)


# (a)
_fig, _ax = plt.subplots(figsize=(6.5, 5.0))
_fig.patch.set_facecolor("white")
_ax.bar(x - width / 2, gb_vals, width, label="Gradient Boosting", color="#2E86AB")
_ax.bar(x + width / 2, pinn_vals, width, label="MonotonicPINN", color="#FF6B6B")
_ax.set_xticks(x)
_ax.set_xticklabels(metrics, fontsize=_F10, fontfamily="Arial")
_ax.set_ylabel("Value", fontsize=_F10, fontfamily="Arial", labelpad=10)
_ax.set_title("(a) Global metrics", fontsize=_F10, fontfamily="Arial", pad=10)
_ax.tick_params(labelsize=_F10)
_ax.grid(False)
_leg = _ax.legend(fontsize=_F10)
for _t in _leg.get_texts():
    _t.set_fontfamily("Arial")
_f10_arial(_ax)
_f10_spines_white(_ax)
plt.tight_layout()
for _stem in ("Figure10a_PanelA",):
    plt.savefig(output_dir / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.savefig(output_dir / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
plt.close()

# (b)
_fig, _ax = plt.subplots(figsize=(6.5, 5.0))
_fig.patch.set_facecolor("white")
_ax.hist(rmse_gb, bins=bins, alpha=0.6, label="GB", color="#2E86AB", edgecolor="black")
_ax.hist(rmse_pinn, bins=bins, alpha=0.6, label="PINN", color="#FF6B6B", edgecolor="black")
_ax.set_xlabel("Per-sample RMSE", fontsize=_F10, fontfamily="Arial", labelpad=10)
_ax.set_ylabel("Frequency", fontsize=_F10, fontfamily="Arial", labelpad=10)
_ax.set_title("(b) Per-sample RMSE distribution", fontsize=_F10, fontfamily="Arial", pad=10)
_ax.tick_params(labelsize=_F10)
_ax.grid(False)
_leg = _ax.legend(fontsize=_F10)
for _t in _leg.get_texts():
    _t.set_fontfamily("Arial")
_f10_arial(_ax)
_f10_spines_white(_ax)
plt.tight_layout()
for _stem in ("Figure10b_PanelB",):
    plt.savefig(output_dir / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.savefig(output_dir / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
plt.close()

# (c)
_fig, _ax = plt.subplots(figsize=(6.5, 5.0))
_fig.patch.set_facecolor("white")
_ax.boxplot(
    [mae_gb, mae_pinn],
    labels=["GB", "PINN"],
    patch_artist=True,
    boxprops=dict(facecolor="#2E86AB", alpha=0.4),
)
for patch, color in zip(_ax.artists, ["#2E86AB", "#FF6B6B"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
_ax.set_ylabel("Per-sample MAE", fontsize=_F10, fontfamily="Arial", labelpad=10)
_ax.set_title("(c) Per-sample MAE distribution", fontsize=_F10, fontfamily="Arial", pad=10)
_ax.tick_params(labelsize=_F10)
_ax.set_xticklabels(["GB", "PINN"], fontsize=_F10, fontfamily="Arial")
_ax.grid(False)
_f10_arial(_ax)
_f10_spines_white(_ax)
plt.tight_layout()
for _stem in ("Figure10c_PanelC",):
    plt.savefig(output_dir / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.savefig(output_dir / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
plt.close()

print(f"  ✓ Saved: Figure10a_PanelA … Figure10c_PanelC (Arial {_F10} pt) → {output_dir}")

# =============================================================================
# Figure 11 – Representative SWCC predictions (set 1)
# =============================================================================
print("\nGenerating Figure 11: Representative SWCC predictions (set 1)...")

_F11 = 11
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Use texture information for selection
clay = X_test_gb["clay_pct"].values
silt = X_test_gb["silt_pct"].values
sand = X_test_gb["sand_pct"].values

def pick_index(mask):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    return int(idx[0])

indices_set1 = []
# Sand
idx_sand = pick_index((sand > 70) & (clay < 15))
if idx_sand is not None:
    indices_set1.append(idx_sand)
# Sandy loam
idx_sandy_loam = pick_index((sand > 50) & (sand < 70) & (clay < 20))
if idx_sandy_loam is not None:
    indices_set1.append(idx_sandy_loam)
# Silt loam
idx_silt_loam = pick_index((silt > 50) & (clay < 27))
if idx_silt_loam is not None:
    indices_set1.append(idx_silt_loam)

indices_set1 = indices_set1[:3]
# Panel titles aligned with selection order: sand → sandy loam → silt loam
F11_PANEL_TITLES = [
    "(a) Sand (typical case)",
    "(b) Sandy loam (typical case)",
    "(c) Silt loam (typical case)",
]

# Match Figure3a_PanelA: matric suction ψ (kPa), water content θ, log x + plain decade ticks
_F11_XMIN = float(np.min(suction_grid))
_F11_XMAX = float(np.max(suction_grid))
_F11_XTICKS = np.array([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
_F11_XTICKLABELS = [
    "0.1",
    "1.0",
    "10",
    "100",
    "1000",
    "10000",
    "100000",
    "1000000",
]


def _f11_swcc_axes_match_fig3a(ax):
    ax.set_xlabel("Matric suction \u03c8 (kPa)", fontsize=_F11, fontfamily="Arial", labelpad=10)
    ax.set_ylabel("Water content (\u03b8)", fontsize=_F11, fontfamily="Arial", labelpad=10)
    ax.set_xlim([_F11_XMIN, _F11_XMAX])
    ax.set_xscale("log")
    ax.set_xticks(_F11_XTICKS)
    ax.set_xticklabels(_F11_XTICKLABELS)
    for _lbl in ax.get_xticklabels():
        _lbl.set_fontsize(10)
        _lbl.set_fontfamily("Arial")


for ax, idx, title in zip(axes, indices_set1, F11_PANEL_TITLES):
    ax.semilogx(suction_grid, y_true[idx], "k-", linewidth=2.5, label="Observed")
    ax.semilogx(suction_grid, y_gb[idx], color="#2E86AB", linestyle="--", linewidth=2, label="GB")
    ax.semilogx(suction_grid, y_pinn[idx], color="#FF6B6B", linestyle="-.", linewidth=2, label="PINN")
    if y_vgparamnet is not None:
        ax.semilogx(suction_grid, y_vgparamnet[idx], color="#9B59B6", linestyle=":", linewidth=2, label="VGParamNet")
    ax.set_title(title, fontsize=_F11, fontfamily="Arial", pad=10)
    ax.tick_params(labelsize=_F11)
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    leg11 = ax.legend(fontsize=_F11)
    for text in leg11.get_texts():
        text.set_fontfamily("Arial")
    _f11_swcc_axes_match_fig3a(ax)

for ax in axes:
    # Match white background and perimeter box style
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Arial")

plt.tight_layout()
plt.savefig(output_dir / "Figure11_Representative_SWCCs_Set1.png", dpi=300, bbox_inches="tight")
plt.savefig(output_dir / "Figure11_Representative_SWCCs_Set1.pdf", dpi=300, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved: {output_dir / 'Figure11_Representative_SWCCs_Set1.png'}")

# Figure 11 panels (a)–(c): standalone, Arial 11 pt, no grid
print("\nGenerating Figure 11 panels (a)–(c) separately (Arial 11 pt, no grid)...")


def _plot_f11_panel(ax, idx, title):
    ax.semilogx(suction_grid, y_true[idx], "k-", linewidth=2.5, label="Observed")
    ax.semilogx(suction_grid, y_gb[idx], color="#2E86AB", linestyle="--", linewidth=2, label="GB")
    ax.semilogx(suction_grid, y_pinn[idx], color="#FF6B6B", linestyle="-.", linewidth=2, label="PINN")
    if y_vgparamnet is not None:
        ax.semilogx(suction_grid, y_vgparamnet[idx], color="#9B59B6", linestyle=":", linewidth=2, label="VGParamNet")
    ax.set_title(title, fontsize=_F11, fontfamily="Arial", pad=10)
    ax.tick_params(labelsize=_F11)
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    leg = ax.legend(fontsize=_F11)
    for t in leg.get_texts():
        t.set_fontfamily("Arial")
    _f11_swcc_axes_match_fig3a(ax)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("Arial")


for _stem, (idx, title) in zip(
    ["Figure11a_PanelA", "Figure11b_PanelB", "Figure11c_PanelC"],
    zip(indices_set1, F11_PANEL_TITLES),
):
    _fig, _ax = plt.subplots(figsize=(6.5, 5.0))
    _fig.patch.set_facecolor("white")
    _plot_f11_panel(_ax, idx, title)
    plt.tight_layout()
    plt.savefig(output_dir / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.savefig(output_dir / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close()

print(f"  ✓ Saved: Figure11a_PanelA … Figure11c_PanelC (Arial {_F11} pt) → {output_dir}")

# =============================================================================
# Figure 12 – Representative SWCC predictions (set 2, including outlier)
# Same axis format as Figure 11 / Figure3a_PanelA (Matric suction ψ, Water content θ, log ticks)
# =============================================================================
print("\nGenerating Figure 12: Representative SWCC predictions (set 2)...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

indices_set2 = []
# Clay
idx_clay = pick_index(clay > 40)
if idx_clay is not None:
    indices_set2.append(idx_clay)
# Silty clay
idx_silty_clay = pick_index((clay > 35) & (silt > 40))
if idx_silty_clay is not None:
    indices_set2.append(idx_silty_clay)
# Outlier: highest GB per-sample RMSE
idx_outlier = int(np.argmax(rmse_gb))
indices_set2.append(idx_outlier)

F12_PANEL_TITLES = [
    "(a) Clay",
    "(b) Silty clay",
    "(c) Outlier (high GB error)",
]

for ax, idx, title in zip(axes, indices_set2, F12_PANEL_TITLES):
    _plot_f11_panel(ax, idx, title)

for ax in axes:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Arial")

plt.tight_layout()
plt.savefig(output_dir / "Figure12_Representative_SWCCs_Set2.png", dpi=300, bbox_inches="tight")
plt.savefig(output_dir / "Figure12_Representative_SWCCs_Set2.pdf", dpi=300, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved: {output_dir / 'Figure12_Representative_SWCCs_Set2.png'}")

# Figure 12 panels (a)–(c): standalone (same style as Figure 11)
print("\nGenerating Figure 12 panels (a)–(c) separately (Arial 11 pt, no grid)...")
for _stem, (idx, title) in zip(
    ["Figure12a_PanelA", "Figure12b_PanelB", "Figure12c_PanelC"],
    zip(indices_set2, F12_PANEL_TITLES),
):
    _fig, _ax = plt.subplots(figsize=(6.5, 5.0))
    _fig.patch.set_facecolor("white")
    _plot_f11_panel(_ax, idx, title)
    plt.tight_layout()
    plt.savefig(output_dir / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.savefig(output_dir / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close()

print(f"  ✓ Saved: Figure12a_PanelA … Figure12c_PanelC (Arial {_F11} pt) → {output_dir}")

# =============================================================================
# Figure 13 – Error vs suction and dry-end comparison
# Same typography as Figures 11–12: Arial 11 pt, no grid; (a)(b) use Matric suction ψ (kPa) + log ticks
# =============================================================================
print("\nGenerating Figure 13: Error vs suction and dry-end comparison...")

_F13 = 11

def _f13_log_psi_x_full(ax):
    """Full-range log ψ axis matching Figure3a / Figure 11 (uses _F11_* from Figure 11 block)."""
    ax.set_xlabel("Matric suction \u03c8 (kPa)", fontsize=_F13, fontfamily="Arial", labelpad=10)
    ax.set_xlim([_F11_XMIN, _F11_XMAX])
    ax.set_xscale("log")
    ax.set_xticks(_F11_XTICKS)
    ax.set_xticklabels(_F11_XTICKLABELS)
    for _lbl in ax.get_xticklabels():
        _lbl.set_fontsize(10)
        _lbl.set_fontfamily("Arial")


def _f13_spines_ticks_arial(ax):
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Arial")


def _plot_f13_panel_a(ax, mean_err_gb, mean_err_pinn):
    ax.semilogx(suction_grid, mean_err_gb, color="#2E86AB", linewidth=2, label="GB")
    ax.semilogx(suction_grid, mean_err_pinn, color="#FF6B6B", linewidth=2, label="PINN")
    ax.set_ylabel("Mean |error|", fontsize=_F13, fontfamily="Arial", labelpad=10)
    ax.set_title("(a) Mean |error| vs suction", fontsize=_F13, fontfamily="Arial", pad=10)
    ax.tick_params(labelsize=_F13)
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    leg = ax.legend(fontsize=_F13)
    for t in leg.get_texts():
        t.set_fontfamily("Arial")
    _f13_log_psi_x_full(ax)


def _plot_f13_panel_b(ax, s_dry, mean_err_gb_dry, mean_err_pinn_dry):
    ax.semilogx(s_dry, mean_err_gb_dry, color="#2E86AB", linewidth=2, label="GB")
    ax.semilogx(s_dry, mean_err_pinn_dry, color="#FF6B6B", linewidth=2, label="PINN")
    ax.set_ylabel("Mean |error|", fontsize=_F13, fontfamily="Arial", labelpad=10)
    ax.set_title("(b) Dry-end (s > 10^4 kPa)", fontsize=_F13, fontfamily="Arial", pad=10)
    ax.tick_params(labelsize=_F13)
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    leg = ax.legend(fontsize=_F13)
    for t in leg.get_texts():
        t.set_fontfamily("Arial")
    ax.set_xlabel("Matric suction \u03c8 (kPa)", fontsize=_F13, fontfamily="Arial", labelpad=10)
    ax.set_xscale("log")
    ax.set_xlim(s_dry.min() * 0.85, s_dry.max() * 1.15)


def _plot_f13_panel_c(ax, gb_contrib, pinn_contrib, labels):
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, gb_contrib, width, label="GB", color="#2E86AB")
    ax.bar(x + width / 2, pinn_contrib, width, label="PINN", color="#FF6B6B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=_F13, fontfamily="Arial")
    ax.set_xlabel("Suction range (kPa)", fontsize=_F13, fontfamily="Arial", labelpad=10)
    ax.set_ylabel("Fraction of total |error|", fontsize=_F13, fontfamily="Arial", labelpad=10)
    ax.set_title("(c) Cumulative error by suction range", fontsize=_F13, fontfamily="Arial", pad=10)
    ax.tick_params(labelsize=_F13)
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    leg = ax.legend(fontsize=_F13)
    for t in leg.get_texts():
        t.set_fontfamily("Arial")


err_gb = np.abs(y_true - y_gb)
err_pinn = np.abs(y_true - y_pinn)
mean_err_gb = np.nanmean(err_gb, axis=0)
mean_err_pinn = np.nanmean(err_pinn, axis=0)

bins = [0, 1e2, 1e3, 1e4, suction_grid.max() * 1.01]
labels_13c = ["<10²", "10²–10³", "10³–10⁴", ">10⁴"]

gb_abs = np.abs(y_true - y_gb)
pinn_abs = np.abs(y_true - y_pinn)

gb_contrib = []
pinn_contrib = []

for b0, b1 in zip(bins[:-1], bins[1:]):
    mask = (suction_grid >= b0) & (suction_grid < b1)
    if mask.any():
        gb_contrib.append(np.nansum(gb_abs[:, mask]))
        pinn_contrib.append(np.nansum(pinn_abs[:, mask]))
    else:
        gb_contrib.append(0.0)
        pinn_contrib.append(0.0)

gb_contrib = np.array(gb_contrib)
pinn_contrib = np.array(pinn_contrib)

if gb_contrib.sum() > 0:
    gb_contrib /= gb_contrib.sum()
if pinn_contrib.sum() > 0:
    pinn_contrib /= pinn_contrib.sum()

dry_mask = suction_grid > 1e4
if dry_mask.any():
    s_dry = suction_grid[dry_mask]
    mean_err_gb_dry = mean_err_gb[dry_mask]
    mean_err_pinn_dry = mean_err_pinn[dry_mask]
else:
    s_dry = None
    mean_err_gb_dry = None
    mean_err_pinn_dry = None

# --- Composite figure ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
_plot_f13_panel_a(axes[0], mean_err_gb, mean_err_pinn)
if s_dry is not None:
    _plot_f13_panel_b(axes[1], s_dry, mean_err_gb_dry, mean_err_pinn_dry)
else:
    axes[1].set_title("(b) Dry-end (s > 10^4 kPa)", fontsize=_F13, fontfamily="Arial", pad=10)
    axes[1].text(
        0.5,
        0.5,
        "No suction points > 10^4 kPa",
        ha="center",
        va="center",
        transform=axes[1].transAxes,
        fontsize=_F13,
        fontfamily="Arial",
    )
    axes[1].set_axis_off()
_plot_f13_panel_c(axes[2], gb_contrib, pinn_contrib, labels_13c)

for ax in axes:
    # Skip placeholder middle panel when dry-end data are absent (axis_off)
    if ax.get_frame_on():
        _f13_spines_ticks_arial(ax)

plt.tight_layout()
plt.savefig(output_dir / "Figure13_Error_vs_Suction_and_DryEnd.png", dpi=300, bbox_inches="tight")
plt.savefig(output_dir / "Figure13_Error_vs_Suction_and_DryEnd.pdf", dpi=300, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved: {output_dir / 'Figure13_Error_vs_Suction_and_DryEnd.png'}")

# --- Standalone panels (a)–(c) ---
print("\nGenerating Figure 13 panels (a)–(c) separately (Arial 11 pt, no grid)...")

_fig, _ax = plt.subplots(figsize=(6.5, 5.0))
_fig.patch.set_facecolor("white")
_plot_f13_panel_a(_ax, mean_err_gb, mean_err_pinn)
_f13_spines_ticks_arial(_ax)
plt.tight_layout()
for _stem in ("Figure13a_PanelA",):
    plt.savefig(output_dir / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.savefig(output_dir / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
plt.close()

if s_dry is not None:
    _fig, _ax = plt.subplots(figsize=(6.5, 5.0))
    _fig.patch.set_facecolor("white")
    _plot_f13_panel_b(_ax, s_dry, mean_err_gb_dry, mean_err_pinn_dry)
    _f13_spines_ticks_arial(_ax)
    plt.tight_layout()
    for _stem in ("Figure13b_PanelB",):
        plt.savefig(output_dir / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
        plt.savefig(output_dir / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close()

_fig, _ax = plt.subplots(figsize=(6.5, 5.0))
_fig.patch.set_facecolor("white")
_plot_f13_panel_c(_ax, gb_contrib, pinn_contrib, labels_13c)
_f13_spines_ticks_arial(_ax)
plt.tight_layout()
for _stem in ("Figure13c_PanelC",):
    plt.savefig(output_dir / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.savefig(output_dir / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
plt.close()

print(f"  ✓ Saved: Figure13a_PanelA … Figure13c_PanelC (Arial {_F13} pt) → {output_dir}")

# =============================================================================
# Figure 16 – Knee Fidelity Metrics
# Same typography as Figures 11–15: Arial 11 pt, no grid; composite + Figure16a–b panels
# =============================================================================
print("\nGenerating Figure 16: Knee fidelity metrics...")

if y_vgparamnet is not None:
    from analysis.knee_fidelity_analysis import find_psi_50, compute_max_slope

    _F16 = 11
    _F16_COLORS = ["black", "#2E86AB", "#FF6B6B", "#9B59B6"]
    _F16_LABELS = ["Observed", "GB", "PINN", "VGParamNet"]

    # Compute knee metrics for all models
    X_test = pd.read_csv(DATA_CONFIG["test_file"])
    theta_s = X_test["theta_s"].values
    theta_r = X_test["theta_r"].values

    psi50_obs = np.array([find_psi_50(suction_grid, y_true[i], theta_s[i], theta_r[i]) for i in range(len(y_true))])
    psi50_gb = np.array([find_psi_50(suction_grid, y_gb[i], theta_s[i], theta_r[i]) for i in range(len(y_gb))])
    psi50_pinn = np.array([find_psi_50(suction_grid, y_pinn[i], theta_s[i], theta_r[i]) for i in range(len(y_pinn))])
    psi50_vgparamnet = np.array([find_psi_50(suction_grid, y_vgparamnet[i], theta_s[i], theta_r[i]) for i in range(len(y_vgparamnet))])

    max_slope_obs, _ = zip(*[compute_max_slope(suction_grid, y_true[i]) for i in range(len(y_true))])
    max_slope_gb, _ = zip(*[compute_max_slope(suction_grid, y_gb[i]) for i in range(len(y_gb))])
    max_slope_pinn, _ = zip(*[compute_max_slope(suction_grid, y_pinn[i]) for i in range(len(y_pinn))])
    max_slope_vgparamnet, _ = zip(*[compute_max_slope(suction_grid, y_vgparamnet[i]) for i in range(len(y_vgparamnet))])

    max_slope_obs = np.array(max_slope_obs)
    max_slope_gb = np.array(max_slope_gb)
    max_slope_pinn = np.array(max_slope_pinn)
    max_slope_vgparamnet = np.array(max_slope_vgparamnet)

    data_psi50 = [psi50_obs, psi50_gb, psi50_pinn, psi50_vgparamnet]
    data_slope = [max_slope_obs, max_slope_gb, max_slope_pinn, max_slope_vgparamnet]

    def _plot_f16_panel_a(ax):
        bp = ax.boxplot(data_psi50, labels=_F16_LABELS, patch_artist=True, showmeans=True)
        for patch, color in zip(bp["boxes"], _F16_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_yscale("log")
        # ψ₅₀ = matric suction where S_e = (θ−θ_r)/(θ_s−θ_r) = 0.5 (midpoint of drying curve)
        ax.set_ylabel("Suction \u03c8 at S_e = 0.5 (kPa)", fontsize=_F16, fontfamily="Arial", labelpad=10)
        ax.set_title("(a) Where the curve is halfway (wet to dry)", fontsize=_F16, fontfamily="Arial", pad=10)
        ax.set_xlabel("Observed vs model", fontsize=_F16, fontfamily="Arial", labelpad=10)
        ax.tick_params(labelsize=_F16)
        ax.grid(False)
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)
        for tick in ax.get_xticklabels():
            tick.set_fontsize(_F16 - 1)
            tick.set_fontfamily("Arial")
            tick.set_rotation(18)

    def _plot_f16_panel_b(ax):
        bp = ax.boxplot(data_slope, labels=_F16_LABELS, patch_artist=True, showmeans=True)
        for patch, color in zip(bp["boxes"], _F16_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        # Largest |Δθ/Δlog₁₀ψ| along the curve (steepest segment on a semilog SWCC plot)
        ax.set_ylabel("Largest slope of \u03b8 vs log\u2081\u2080 \u03c8 (\u2212)", fontsize=_F16, fontfamily="Arial", labelpad=10)
        ax.set_title("(b) How sharp is the steepest part of the curve?", fontsize=_F16, fontfamily="Arial", pad=10)
        ax.set_xlabel("Observed vs model", fontsize=_F16, fontfamily="Arial", labelpad=10)
        ax.tick_params(labelsize=_F16)
        ax.grid(False)
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)
        for tick in ax.get_xticklabels():
            tick.set_fontsize(_F16 - 1)
            tick.set_fontfamily("Arial")
            tick.set_rotation(18)

    # --- Composite figure ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_f16_panel_a(axes[0])
    _plot_f16_panel_b(axes[1])

    for ax in axes:
        _f13_spines_ticks_arial(ax)

    plt.tight_layout()
    plt.savefig(output_dir / "Figure16_Knee_Fidelity_Metrics.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "Figure16_Knee_Fidelity_Metrics.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {output_dir / 'Figure16_Knee_Fidelity_Metrics.png'}")

    # --- Standalone panels (a)–(b) ---
    print("\nGenerating Figure 16 panels (a)–(b) separately (Arial 11 pt, no grid)...")

    _fig, _ax = plt.subplots(figsize=(6.5, 5.0))
    _fig.patch.set_facecolor("white")
    _plot_f16_panel_a(_ax)
    _f13_spines_ticks_arial(_ax)
    plt.tight_layout()
    for _stem in ("Figure16a_PanelA",):
        plt.savefig(output_dir / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
        plt.savefig(output_dir / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close()

    _fig, _ax = plt.subplots(figsize=(6.5, 5.0))
    _fig.patch.set_facecolor("white")
    _plot_f16_panel_b(_ax)
    _f13_spines_ticks_arial(_ax)
    plt.tight_layout()
    for _stem in ("Figure16b_PanelB",):
        plt.savefig(output_dir / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
        plt.savefig(output_dir / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close()

    print(f"  ✓ Saved: Figure16a_PanelA … Figure16b_PanelB (Arial {_F16} pt) → {output_dir}")
else:
    print("  ⚠ Skipping knee fidelity figure (VGParamNet not available)")

print("\nDone. Figures 10–13, 16 generated in:", output_dir)

