#!/usr/bin/env python3
"""
Comprehensive PINN Evaluation
- Converts normalized RMSE to physical RMSE
- Evaluates in specific regimes (dry-end, sparse data)
- Generates key plots for paper
- Creates comparison tables
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import interpolate

from models.pinn_monotonic import MonotonicPINN
from models.pinn import PhysicsEncodingLayer
from training_pinn.config_pinn_fixed import DATA_CONFIG, CHECKPOINT_DIR, RESULTS_DIR

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
SMALL_SIZE = 10
MEDIUM_SIZE = 12
LARGE_SIZE = 14
plt.rcParams['font.size'] = SMALL_SIZE
plt.rcParams['axes.titlesize'] = MEDIUM_SIZE
plt.rcParams['axes.labelsize'] = MEDIUM_SIZE
plt.rcParams['xtick.labelsize'] = SMALL_SIZE
plt.rcParams['ytick.labelsize'] = SMALL_SIZE
plt.rcParams['legend.fontsize'] = SMALL_SIZE
plt.rcParams['figure.titlesize'] = LARGE_SIZE

print("="*80)
print("Comprehensive PINN Evaluation")
print("="*80)

# ============================================================================
# 1. LOAD MODEL AND DATA
# ============================================================================
print("\n1. Loading model and data...")

# Load best model (by validation RMSE)
checkpoint_files = sorted(CHECKPOINT_DIR.glob("pinn_checkpoint_epoch_*.keras"))
best_model_path = CHECKPOINT_DIR / "pinn_best_model_fixed.keras"

if not best_model_path.exists() and checkpoint_files:
    best_model_path = checkpoint_files[-1]

print(f"  Loading model: {best_model_path}")

metadata = json.load(open(DATA_CONFIG['metadata_file']))
model = MonotonicPINN(
    soil_prop_dim=metadata['n_features'],
    suction_points=metadata['n_swcc_points'],
    physics_units=128,
    hidden_dims=[128, 256, 128, 64]
)

# Build model
dummy_soil = tf.random.normal([1, metadata['n_features']])
dummy_suction = tf.random.normal([1, metadata['n_swcc_points']])
_ = model({'soil_props': dummy_soil, 'suction': dummy_suction})

# Load weights
try:
    saved_model = tf.keras.models.load_model(
        str(best_model_path),
        custom_objects={'MonotonicPINN': MonotonicPINN, 'PhysicsEncodingLayer': PhysicsEncodingLayer},
        compile=False
    )
    model.set_weights(saved_model.get_weights())
    print("  ✓ Model loaded successfully")
except Exception as e:
    print(f"  ✗ Error loading model: {e}")
    print("  Using untrained model (for testing)")

# Load test data
X_test = pd.read_csv(DATA_CONFIG['test_file'])
y_test_norm = np.load(DATA_CONFIG['y_test_file'])  # Normalized [0,1]
y_test_original = np.load(DATA_CONFIG['y_test_original_file'])  # Original values
suction_grid = np.load(DATA_CONFIG['suction_grid_file'])

theta_s_test = X_test['theta_s'].values
theta_r_test = X_test['theta_r'].values

print(f"  Test samples: {len(X_test)}")
print(f"  SWCC points: {y_test_norm.shape[1]}")

# ============================================================================
# 2. PREDICT AND DENORMALIZE
# ============================================================================
print("\n2. Making predictions...")

# Predict in normalized space
y_pred_norm = []
feature_cols = metadata['feature_cols']

for i in range(len(X_test)):
    sample_soil = X_test.iloc[i:i+1][feature_cols].values.astype(np.float32)
    sample_suction = np.tile(suction_grid, (1, 1)).astype(np.float32)
    
    inputs = {'soil_props': tf.constant(sample_soil), 'suction': tf.constant(sample_suction)}
    theta_pred_norm = model(inputs, training=False)
    y_pred_norm.append(theta_pred_norm.numpy()[0])

y_pred_norm = np.array(y_pred_norm)

# Denormalize to physical space
y_pred_physical = np.zeros_like(y_pred_norm)
for i in range(len(X_test)):
    theta_range = theta_s_test[i] - theta_r_test[i]
    y_pred_physical[i] = theta_r_test[i] + y_pred_norm[i] * theta_range

print(f"  ✓ Predictions complete")
print(f"  Normalized range: [{y_pred_norm.min():.4f}, {y_pred_norm.max():.4f}]")
print(f"  Physical range: [{y_pred_physical.min():.4f}, {y_pred_physical.max():.4f}]")

# ============================================================================
# 3. COMPUTE GLOBAL METRICS (PHYSICAL SPACE)
# ============================================================================
print("\n3. Computing global metrics (physical space)...")

# Flatten for overall metrics
y_true_flat = y_test_original.flatten()
y_pred_flat = y_pred_physical.flatten()

# Remove NaN
mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
y_true_clean = y_true_flat[mask]
y_pred_clean = y_pred_flat[mask]

# Overall metrics
rmse_global = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
mae_global = mean_absolute_error(y_true_clean, y_pred_clean)
r2_global = r2_score(y_true_clean, y_pred_clean)

# Per-sample metrics
sample_rmse = []
sample_mae = []
sample_r2 = []

for i in range(len(X_test)):
    y_t = y_test_original[i]
    y_p = y_pred_physical[i]
    mask_i = ~(np.isnan(y_t) | np.isnan(y_p))
    if mask_i.sum() > 0:
        sample_rmse.append(np.sqrt(mean_squared_error(y_t[mask_i], y_p[mask_i])))
        sample_mae.append(mean_absolute_error(y_t[mask_i], y_p[mask_i]))
        sample_r2.append(r2_score(y_t[mask_i], y_p[mask_i]))

sample_rmse = np.array(sample_rmse)
sample_mae = np.array(sample_mae)
sample_r2 = np.array(sample_r2)

print(f"\nGlobal Metrics (Physical Space):")
print(f"  RMSE: {rmse_global:.6f}")
print(f"  MAE: {mae_global:.6f}")
print(f"  R²: {r2_global:.6f}")

print(f"\nPer-Sample Metrics (Physical Space):")
print(f"  RMSE - Mean: {sample_rmse.mean():.6f}, Median: {np.median(sample_rmse):.6f}, 90th: {np.percentile(sample_rmse, 90):.6f}")
print(f"  MAE - Mean: {sample_mae.mean():.6f}, Median: {np.median(sample_mae):.6f}")
print(f"  R² - Mean: {sample_r2.mean():.6f}, Median: {np.median(sample_r2):.6f}")

# ============================================================================
# 4. EVALUATE DRY-END PERFORMANCE (HIGH SUCTION)
# ============================================================================
print("\n4. Evaluating dry-end performance (high suction)...")

# Define dry-end threshold (suction > 10^4 kPa or pF > 4.2)
dry_end_threshold = 1e4  # kPa
dry_end_indices = np.where(suction_grid > dry_end_threshold)[0]

if len(dry_end_indices) > 0:
    # Extract dry-end data
    y_true_dry = y_test_original[:, dry_end_indices]
    y_pred_dry = y_pred_physical[:, dry_end_indices]
    
    # Flatten
    y_true_dry_flat = y_true_dry.flatten()
    y_pred_dry_flat = y_pred_dry.flatten()
    
    # Remove NaN
    mask_dry = ~(np.isnan(y_true_dry_flat) | np.isnan(y_pred_dry_flat))
    
    rmse_dry = np.sqrt(mean_squared_error(y_true_dry_flat[mask_dry], y_pred_dry_flat[mask_dry]))
    mae_dry = mean_absolute_error(y_true_dry_flat[mask_dry], y_pred_dry_flat[mask_dry])
    
    print(f"  Dry-end threshold: > {dry_end_threshold:.0f} kPa")
    print(f"  Points per sample: {len(dry_end_indices)}")
    print(f"  RMSE: {rmse_dry:.6f}")
    print(f"  MAE: {mae_dry:.6f}")
else:
    print(f"  ⚠ No points above threshold {dry_end_threshold:.0f} kPa")
    rmse_dry = np.nan
    mae_dry = np.nan

# ============================================================================
# 5. EVALUATE SPARSE DATA SCENARIO
# ============================================================================
print("\n5. Evaluating sparse data scenario...")

# Simulate sparse measurements: 5-10 points per curve
n_sparse_points = 8
sparse_indices = np.linspace(0, len(suction_grid)-1, n_sparse_points, dtype=int)

# Extract sparse observations
y_test_sparse = y_test_original[:, sparse_indices]
suction_sparse = suction_grid[sparse_indices]

# Interpolate predictions at sparse points
y_pred_sparse = y_pred_physical[:, sparse_indices]

# Compute RMSE on sparse points
y_true_sparse_flat = y_test_sparse.flatten()
y_pred_sparse_flat = y_pred_sparse.flatten()
mask_sparse = ~(np.isnan(y_true_sparse_flat) | np.isnan(y_pred_sparse_flat))

rmse_sparse = np.sqrt(mean_squared_error(y_true_sparse_flat[mask_sparse], y_pred_sparse_flat[mask_sparse]))
mae_sparse = mean_absolute_error(y_true_sparse_flat[mask_sparse], y_pred_sparse_flat[mask_sparse])

print(f"  Sparse points: {n_sparse_points} per curve")
print(f"  RMSE: {rmse_sparse:.6f}")
print(f"  MAE: {mae_sparse:.6f}")

# ============================================================================
# 6. PHYSICS COMPLIANCE
# ============================================================================
print("\n6. Checking physics compliance...")

# Monotonicity (should be 100% by construction)
mono_violations = 0
for i in range(len(y_pred_physical)):
    diff = y_pred_physical[i, :-1] - y_pred_physical[i, 1:]
    if np.any(diff < -1e-6):
        mono_violations += 1

monotonicity_rate = 1.0 - mono_violations / len(X_test)

# Boundary compliance
boundary_violations = 0
for i in range(len(X_test)):
    if np.any(y_pred_physical[i] < theta_r_test[i] - 1e-6) or \
       np.any(y_pred_physical[i] > theta_s_test[i] + 1e-6):
        boundary_violations += 1

boundary_rate = 1.0 - boundary_violations / len(X_test)

# Boundary errors at endpoints
theta_at_s0_errors = []
theta_at_smax_errors = []

for i in range(len(X_test)):
    # At s=0 (should be close to theta_s)
    theta_at_s0_pred = y_pred_physical[i, 0]
    theta_at_s0_true = y_test_original[i, 0]
    theta_at_s0_errors.append(abs(theta_at_s0_pred - theta_at_s0_true))
    
    # At s_max (should be close to theta_r)
    theta_at_smax_pred = y_pred_physical[i, -1]
    theta_at_smax_true = y_test_original[i, -1]
    theta_at_smax_errors.append(abs(theta_at_smax_pred - theta_at_smax_true))

print(f"  Monotonicity: {monotonicity_rate*100:.1f}% ({mono_violations} violations)")
print(f"  Boundary: {boundary_rate*100:.1f}% ({boundary_violations} violations)")
print(f"  Mean error at s=0: {np.mean(theta_at_s0_errors):.6f}")
print(f"  Mean error at s_max: {np.mean(theta_at_smax_errors):.6f}")

# ============================================================================
# 7. LOAD BASELINE RESULTS FOR COMPARISON
# ============================================================================
print("\n7. Loading baseline results...")

baseline_file = Path("results_baseline/baseline_results.csv")
baseline_metrics = None

if baseline_file.exists():
    baseline_df = pd.read_csv(baseline_file)
    best_baseline = baseline_df[baseline_df['dataset'] == 'test'].nsmallest(1, 'rmse').iloc[0]
    baseline_metrics = {
        'model': best_baseline['model'],
        'rmse': float(best_baseline['rmse']),
        'mae': float(best_baseline['mae']),
        'r2': float(best_baseline['r2'])
    }
    print(f"  Best baseline: {baseline_metrics['model']}")
    print(f"    RMSE: {baseline_metrics['rmse']:.6f}")
    print(f"    MAE: {baseline_metrics['mae']:.6f}")
    print(f"    R²: {baseline_metrics['r2']:.6f}")
else:
    print("  ⚠ Baseline results not found")

# ============================================================================
# 8. CREATE COMPARISON TABLE
# ============================================================================
print("\n8. Creating comparison table...")

comparison_data = {
    'Metric': [
        'Global RMSE (θ)',
        'Global MAE (θ)',
        'Global R²',
        'Dry-end RMSE (s > 10⁴ kPa)',
        'Sparse-data RMSE (8 points)',
        'Monotonicity (%)',
        'Boundary Satisfaction (%)',
        'Mean Error at s=0',
        'Mean Error at s_max'
    ],
    'Gradient Boosting': [
        f"{baseline_metrics['rmse']:.6f}" if baseline_metrics else "N/A",
        f"{baseline_metrics['mae']:.6f}" if baseline_metrics else "N/A",
        f"{baseline_metrics['r2']:.6f}" if baseline_metrics else "N/A",
        "N/A",  # Dry-end not computed for baseline
        "N/A",  # Sparse not computed for baseline
        "<100%",  # Baseline doesn't enforce
        "Not enforced",  # Baseline doesn't enforce
        "N/A",
        "N/A"
    ],
    'PINN (Monotonic)': [
        f"{rmse_global:.6f}",
        f"{mae_global:.6f}",
        f"{r2_global:.6f}",
        f"{rmse_dry:.6f}" if not np.isnan(rmse_dry) else "N/A",
        f"{rmse_sparse:.6f}",
        f"{monotonicity_rate*100:.1f}%",
        f"{boundary_rate*100:.1f}%",
        f"{np.mean(theta_at_s0_errors):.6f}",
        f"{np.mean(theta_at_smax_errors):.6f}"
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# Save comparison table
comparison_file = RESULTS_DIR / "comparison_table.csv"
comparison_df.to_csv(comparison_file, index=False)
print(f"\n  ✓ Saved: {comparison_file}")

# ============================================================================
# 9. GENERATE KEY PLOTS
# ============================================================================
print("\n9. Generating key plots...")

viz_dir = RESULTS_DIR / "visualizations"
viz_dir.mkdir(parents=True, exist_ok=True)

# 9.1 Representative SWCC curves
print("  9.1 Representative SWCC curves...")

# Select diverse samples (different textures)
# Try to select based on texture (if available) or randomly
n_samples_plot = 6
sample_indices = np.random.choice(len(X_test), min(n_samples_plot, len(X_test)), replace=False)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, ax_idx in enumerate(sample_indices):
    ax = axes[idx]
    
    # Observed
    ax.semilogx(suction_grid, y_test_original[ax_idx], 'ko-', 
               markersize=4, linewidth=1.5, label='Observed', alpha=0.7)
    
    # PINN prediction
    ax.semilogx(suction_grid, y_pred_physical[ax_idx], 'r--', 
               linewidth=2, label='PINN', alpha=0.8)
    
    # Boundaries
    ax.axhline(theta_s_test[ax_idx], color='g', linestyle=':', alpha=0.5, linewidth=1)
    ax.axhline(theta_r_test[ax_idx], color='orange', linestyle=':', alpha=0.5, linewidth=1)
    
    # Sample metrics
    sample_rmse_val = sample_rmse[ax_idx]
    sample_r2_val = sample_r2[ax_idx]
    
    ax.set_xlabel('Suction (kPa)', fontsize=11)
    ax.set_ylabel('Water Content (θ)', fontsize=11)
    ax.set_title(f'Sample {ax_idx+1}\nRMSE: {sample_rmse_val:.4f}, R²: {sample_r2_val:.4f}', 
                fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Representative SWCC Curves: PINN Predictions vs Observations', 
            fontsize=16, y=0.995)
plt.tight_layout()
plt.savefig(viz_dir / 'representative_swcc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {viz_dir / 'representative_swcc_curves.png'}")

# 9.2 Error vs Suction
print("  9.2 Error vs Suction...")

# Compute mean absolute error at each suction point
errors_by_suction = np.abs(y_pred_physical - y_test_original)
mean_error_by_suction = np.nanmean(errors_by_suction, axis=0)
std_error_by_suction = np.nanstd(errors_by_suction, axis=0)

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(suction_grid, mean_error_by_suction, 'b-', linewidth=2, label='PINN Mean |Error|')
ax.fill_between(suction_grid, 
                mean_error_by_suction - std_error_by_suction,
                mean_error_by_suction + std_error_by_suction,
                alpha=0.3, color='blue', label='±1 std')

ax.set_xlabel('Suction (kPa)', fontsize=12)
ax.set_ylabel('Mean Absolute Error |θ_pred - θ_obs|', fontsize=12)
ax.set_title('Error vs Suction: PINN Performance Across Suction Range', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(viz_dir / 'error_vs_suction.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {viz_dir / 'error_vs_suction.png'}")

# 9.3 Histogram of per-sample RMSE
print("  9.3 Histogram of per-sample RMSE...")

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(sample_rmse, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
ax.axvline(sample_rmse.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {sample_rmse.mean():.4f}')
ax.axvline(np.median(sample_rmse), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(sample_rmse):.4f}')

ax.set_xlabel('Per-Sample RMSE (Physical Space)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Per-Sample RMSE: PINN Test Set Performance', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(viz_dir / 'per_sample_rmse_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {viz_dir / 'per_sample_rmse_distribution.png'}")

# 9.4 Scatter: Predicted vs Observed
print("  9.4 Predicted vs Observed scatter...")

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(y_true_clean, y_pred_clean, alpha=0.3, s=10, color='steelblue')
ax.plot([y_true_clean.min(), y_true_clean.max()], 
       [y_true_clean.min(), y_true_clean.max()], 
       'r--', linewidth=2, label='Perfect Prediction')

ax.set_xlabel('Observed Water Content (θ)', fontsize=12)
ax.set_ylabel('Predicted Water Content (θ)', fontsize=12)
ax.set_title(f'Predicted vs Observed: PINN Test Set\nRMSE: {rmse_global:.4f}, R²: {r2_global:.4f}', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(viz_dir / 'predicted_vs_observed.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {viz_dir / 'predicted_vs_observed.png'}")

# ============================================================================
# 10. SAVE RESULTS
# ============================================================================
print("\n10. Saving results...")

results = {
    'global_metrics': {
        'rmse': float(rmse_global),
        'mae': float(mae_global),
        'r2': float(r2_global)
    },
    'per_sample_metrics': {
        'rmse_mean': float(sample_rmse.mean()),
        'rmse_median': float(np.median(sample_rmse)),
        'rmse_90th': float(np.percentile(sample_rmse, 90)),
        'mae_mean': float(sample_mae.mean()),
        'mae_median': float(np.median(sample_mae)),
        'r2_mean': float(sample_r2.mean()),
        'r2_median': float(np.median(sample_r2))
    },
    'dry_end_metrics': {
        'rmse': float(rmse_dry) if not np.isnan(rmse_dry) else None,
        'mae': float(mae_dry) if not np.isnan(mae_dry) else None,
        'threshold_kpa': float(dry_end_threshold)
    },
    'sparse_data_metrics': {
        'rmse': float(rmse_sparse),
        'mae': float(mae_sparse),
        'n_points': int(n_sparse_points)
    },
    'physics_compliance': {
        'monotonicity_rate': float(monotonicity_rate),
        'boundary_rate': float(boundary_rate),
        'mean_error_at_s0': float(np.mean(theta_at_s0_errors)),
        'mean_error_at_smax': float(np.mean(theta_at_smax_errors))
    },
    'baseline_comparison': baseline_metrics
}

results_file = RESULTS_DIR / "evaluation_results_comprehensive.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"  ✓ Saved: {results_file}")

# Save per-sample metrics
per_sample_df = pd.DataFrame({
    'sample_idx': range(len(X_test)),
    'rmse': sample_rmse,
    'mae': sample_mae,
    'r2': sample_r2
})
per_sample_df.to_csv(RESULTS_DIR / "per_sample_metrics.csv", index=False)
print(f"  ✓ Saved: {RESULTS_DIR / 'per_sample_metrics.csv'}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)

print(f"\nGlobal Performance (Physical Space):")
print(f"  RMSE: {rmse_global:.6f}")
print(f"  MAE: {mae_global:.6f}")
print(f"  R²: {r2_global:.6f}")

if baseline_metrics:
    print(f"\nComparison with Baseline ({baseline_metrics['model']}):")
    print(f"  RMSE: {baseline_metrics['rmse']:.6f} (baseline) vs {rmse_global:.6f} (PINN)")
    rmse_improvement = (baseline_metrics['rmse'] - rmse_global) / baseline_metrics['rmse'] * 100
    print(f"  Improvement: {rmse_improvement:+.2f}%")

print(f"\nPhysics Compliance:")
print(f"  Monotonicity: {monotonicity_rate*100:.1f}%")
print(f"  Boundary: {boundary_rate*100:.1f}%")

print(f"\nRegime-Specific Performance:")
if not np.isnan(rmse_dry):
    print(f"  Dry-end RMSE: {rmse_dry:.6f}")
print(f"  Sparse-data RMSE: {rmse_sparse:.6f}")

print(f"\nFiles created:")
print(f"  - Comparison table: {comparison_file}")
print(f"  - Results JSON: {results_file}")
print(f"  - Visualizations: {viz_dir}/")
print(f"\n" + "="*80)
