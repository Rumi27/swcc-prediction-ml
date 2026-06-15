#!/usr/bin/env python3
"""
Comprehensive Final PINN Evaluation
- Evaluates PINN on test set (real data only)
- Compares with baseline models
- Generates comprehensive metrics (global, per-sample, regime-specific)
- Creates comparison plots
- Prepares paper-ready results
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
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 16

print("="*80)
print("Comprehensive PINN Evaluation (Final Model)")
print("="*80)

# ============================================================================
# 1. LOAD MODEL AND DATA
# ============================================================================
print("\n1. Loading model and data...")

# Load final model
final_model_path = CHECKPOINT_DIR / "pinn_final_model_fixed.keras"
if not final_model_path.exists():
    # Try to find best checkpoint
    checkpoint_files = sorted(CHECKPOINT_DIR.glob("pinn_checkpoint_epoch_*.keras"))
    if checkpoint_files:
        final_model_path = checkpoint_files[-1]
        print(f"  Using latest checkpoint: {final_model_path}")
    else:
        raise FileNotFoundError(f"Model not found: {final_model_path}")

print(f"  Loading model: {final_model_path}")

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
        str(final_model_path),
        custom_objects={'MonotonicPINN': MonotonicPINN, 'PhysicsEncodingLayer': PhysicsEncodingLayer},
        compile=False
    )
    model.set_weights(saved_model.get_weights())
    print("  ✓ Model loaded successfully")
except Exception as e:
    print(f"  ✗ Error loading model: {e}")
    raise

# Load test data (real only)
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

batch_size = 32
for i in range(0, len(X_test), batch_size):
    batch_end = min(i + batch_size, len(X_test))
    batch_soil = X_test.iloc[i:batch_end][feature_cols].values.astype(np.float32)
    batch_suction = np.tile(suction_grid, (batch_end - i, 1)).astype(np.float32)
    
    inputs = {
        'soil_props': tf.constant(batch_soil),
        'suction': tf.constant(batch_suction)
    }
    theta_pred_norm_batch = model(inputs, training=False)
    y_pred_norm.extend(theta_pred_norm_batch.numpy())

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
print(f"  RMSE - Mean: {sample_rmse.mean():.6f}, Median: {np.median(sample_rmse):.6f}, Std: {sample_rmse.std():.6f}")
print(f"  MAE  - Mean: {sample_mae.mean():.6f}, Median: {np.median(sample_mae):.6f}, Std: {sample_mae.std():.6f}")
print(f"  R²   - Mean: {sample_r2.mean():.6f}, Median: {np.median(sample_r2):.6f}, Std: {sample_r2.std():.6f}")

# ============================================================================
# 4. REGIME-SPECIFIC METRICS
# ============================================================================
print("\n4. Computing regime-specific metrics...")

# Dry-end (high suction: s > 10^4 kPa)
dry_end_threshold = 1e4  # 10^4 kPa
dry_end_mask = suction_grid > dry_end_threshold
dry_end_indices = np.where(dry_end_mask)[0]

if len(dry_end_indices) > 0:
    y_true_dry = y_test_original[:, dry_end_indices]
    y_pred_dry = y_pred_physical[:, dry_end_indices]
    
    mask_dry = ~(np.isnan(y_true_dry) | np.isnan(y_pred_dry))
    y_true_dry_clean = y_true_dry[mask_dry]
    y_pred_dry_clean = y_pred_dry[mask_dry]
    
    rmse_dry = np.sqrt(mean_squared_error(y_true_dry_clean, y_pred_dry_clean))
    mae_dry = mean_absolute_error(y_true_dry_clean, y_pred_dry_clean)
    
    print(f"\nDry-end metrics (s > {dry_end_threshold:.0e} kPa):")
    print(f"  RMSE: {rmse_dry:.6f}")
    print(f"  MAE: {mae_dry:.6f}")
else:
    rmse_dry = None
    mae_dry = None
    print(f"  No dry-end data available")

# Wet-end (low suction: s < 10^2 kPa)
wet_end_threshold = 1e2  # 10^2 kPa
wet_end_mask = suction_grid < wet_end_threshold
wet_end_indices = np.where(wet_end_mask)[0]

if len(wet_end_indices) > 0:
    y_true_wet = y_test_original[:, wet_end_indices]
    y_pred_wet = y_pred_physical[:, wet_end_indices]
    
    mask_wet = ~(np.isnan(y_true_wet) | np.isnan(y_pred_wet))
    y_true_wet_clean = y_true_wet[mask_wet]
    y_pred_wet_clean = y_pred_wet[mask_wet]
    
    rmse_wet = np.sqrt(mean_squared_error(y_true_wet_clean, y_pred_wet_clean))
    mae_wet = mean_absolute_error(y_true_wet_clean, y_pred_wet_clean)
    
    print(f"\nWet-end metrics (s < {wet_end_threshold:.0e} kPa):")
    print(f"  RMSE: {rmse_wet:.6f}")
    print(f"  MAE: {mae_wet:.6f}")
else:
    rmse_wet = None
    mae_wet = None

# ============================================================================
# 5. PHYSICS COMPLIANCE
# ============================================================================
print("\n5. Checking physics compliance...")

# Monotonicity check
monotonic_count = 0
for i in range(len(y_pred_physical)):
    diff = np.diff(y_pred_physical[i])
    if np.all(diff <= 0):  # Monotonic decreasing
        monotonic_count += 1

monotonicity_rate = monotonic_count / len(y_pred_physical)

# Boundary check
boundary_satisfied = 0
for i in range(len(y_pred_physical)):
    theta_min = y_pred_physical[i].min()
    theta_max = y_pred_physical[i].max()
    if theta_min >= theta_r_test[i] - 0.01 and theta_max <= theta_s_test[i] + 0.01:
        boundary_satisfied += 1

boundary_rate = boundary_satisfied / len(y_pred_physical)

print(f"  Monotonicity: {monotonic_count}/{len(y_pred_physical)} ({monotonicity_rate*100:.2f}%)")
print(f"  Boundary satisfaction: {boundary_satisfied}/{len(y_pred_physical)} ({boundary_rate*100:.2f}%)")

# ============================================================================
# 6. LOAD BASELINE RESULTS
# ============================================================================
print("\n6. Loading baseline model results...")

baseline_results_file = Path("results_baseline/baseline_results.csv")
baseline_report_file = Path("results_baseline/baseline_report.json")
baseline_metrics = None

if baseline_report_file.exists():
    # Try JSON first
    with open(baseline_report_file, 'r') as f:
        baseline_data = json.load(f)
    
    # Extract best model from report
    if 'best_model' in baseline_data:
        best_model_name = baseline_data['best_model']
        baseline_metrics = {
            'model': best_model_name,
            'rmse': float(baseline_data.get('best_rmse', 0)),
            'mae': float(baseline_data.get('best_mae', 0)),
            'r2': float(baseline_data.get('best_r2', 0))
        }
        
        print(f"  Best baseline: {baseline_metrics['model']}")
        print(f"    RMSE: {baseline_metrics['rmse']:.6f}")
        print(f"    MAE: {baseline_metrics['mae']:.6f}")
        print(f"    R²: {baseline_metrics['r2']:.6f}")
    else:
        print("  ⚠ Could not parse baseline results")
elif baseline_results_file.exists():
    # Try CSV
    baseline_df = pd.read_csv(baseline_results_file)
    # Check column names
    print(f"  Baseline CSV columns: {baseline_df.columns.tolist()}")
    # Try different possible column names
    rmse_col = None
    for col in baseline_df.columns:
        if 'rmse' in col.lower() or 'RMSE' in col:
            rmse_col = col
            break
    
    if rmse_col:
        best_baseline_idx = baseline_df[rmse_col].idxmin()
        best_baseline = baseline_df.iloc[best_baseline_idx]
        
        baseline_metrics = {
            'model': str(best_baseline.get('Model', best_baseline.get('model', 'Unknown'))),
            'rmse': float(best_baseline[rmse_col]),
            'mae': float(best_baseline.get('MAE', best_baseline.get('mae', 0))),
            'r2': float(best_baseline.get('R2', best_baseline.get('r2', 0)))
        }
        
        print(f"  Best baseline: {baseline_metrics['model']}")
        print(f"    RMSE: {baseline_metrics['rmse']:.6f}")
        print(f"    MAE: {baseline_metrics['mae']:.6f}")
        print(f"    R²: {baseline_metrics['r2']:.6f}")
    else:
        print("  ⚠ Could not find RMSE column in baseline results")
else:
    print("  ⚠ Baseline results not found. Run baseline_models.py first.")

# ============================================================================
# 7. CREATE COMPARISON PLOTS
# ============================================================================
print("\n7. Creating comparison plots...")

viz_dir = RESULTS_DIR / "visualizations"
viz_dir.mkdir(parents=True, exist_ok=True)

# Plot 1: Representative SWCC curves
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

n_samples = min(6, len(X_test))
indices = np.linspace(0, len(X_test)-1, n_samples, dtype=int)

for idx, ax in enumerate(axes[:n_samples]):
    i = indices[idx]
    ax.semilogx(suction_grid, y_test_original[i], 'b-', linewidth=2, label='Observed', alpha=0.8)
    ax.semilogx(suction_grid, y_pred_physical[i], 'r--', linewidth=2, label='PINN Prediction', alpha=0.8)
    ax.axhline(y=theta_s_test[i], color='g', linestyle=':', alpha=0.5, label='θ_s')
    ax.axhline(y=theta_r_test[i], color='orange', linestyle=':', alpha=0.5, label='θ_r')
    ax.set_xlabel('Suction (kPa)', fontsize=12)
    ax.set_ylabel('Water Content (θ)', fontsize=12)
    ax.set_title(f'Sample {i+1}', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.legend(loc='best', fontsize=9)

# Remove extra subplots
for ax in axes[n_samples:]:
    ax.remove()

plt.tight_layout()
plt.savefig(viz_dir / 'pinn_representative_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {viz_dir / 'pinn_representative_curves.png'}")

# Plot 2: Error vs Suction
fig, ax = plt.subplots(figsize=(10, 6))
errors = np.abs(y_test_original - y_pred_physical)
mean_error = np.nanmean(errors, axis=0)
std_error = np.nanstd(errors, axis=0)

ax.semilogx(suction_grid, mean_error, 'b-', linewidth=2, label='Mean |Error|')
ax.fill_between(suction_grid, mean_error - std_error, mean_error + std_error, 
                alpha=0.3, color='blue', label='±1 Std')
ax.set_xlabel('Suction (kPa)', fontsize=13)
ax.set_ylabel('Absolute Error |θ_true - θ_pred|', fontsize=13)
ax.set_title('Error vs Suction', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(viz_dir / 'pinn_error_vs_suction.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {viz_dir / 'pinn_error_vs_suction.png'}")

# Plot 3: Per-sample RMSE distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(sample_rmse, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(sample_rmse.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {sample_rmse.mean():.4f}')
ax.axvline(np.median(sample_rmse), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(sample_rmse):.4f}')
ax.set_xlabel('Per-Sample RMSE', fontsize=13)
ax.set_ylabel('Frequency', fontsize=13)
ax.set_title('Distribution of Per-Sample RMSE', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(viz_dir / 'pinn_rmse_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {viz_dir / 'pinn_rmse_distribution.png'}")

# Plot 4: Comparison with baseline (if available)
if baseline_metrics:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # RMSE comparison
    ax = axes[0]
    models = ['PINN', baseline_metrics['model']]
    rmse_vals = [rmse_global, baseline_metrics['rmse']]
    colors = ['#2E86AB', '#F18F01']
    bars = ax.bar(models, rmse_vals, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('RMSE', fontsize=13)
    ax.set_title('RMSE Comparison', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rmse_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11)
    
    # MAE comparison
    ax = axes[1]
    mae_vals = [mae_global, baseline_metrics['mae']]
    bars = ax.bar(models, mae_vals, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('MAE', fontsize=13)
    ax.set_title('MAE Comparison', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, mae_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11)
    
    # R² comparison
    ax = axes[2]
    r2_vals = [r2_global, baseline_metrics['r2']]
    bars = ax.bar(models, r2_vals, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('R²', fontsize=13)
    ax.set_title('R² Comparison', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, r2_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'pinn_vs_baseline_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {viz_dir / 'pinn_vs_baseline_comparison.png'}")

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================
print("\n8. Saving evaluation results...")

results = {
    'global_metrics': {
        'rmse': float(rmse_global),
        'mae': float(mae_global),
        'r2': float(r2_global)
    },
    'per_sample_metrics': {
        'rmse_mean': float(sample_rmse.mean()),
        'rmse_median': float(np.median(sample_rmse)),
        'rmse_std': float(sample_rmse.std()),
        'mae_mean': float(sample_mae.mean()),
        'mae_median': float(np.median(sample_mae)),
        'mae_std': float(sample_mae.std()),
        'r2_mean': float(sample_r2.mean()),
        'r2_median': float(np.median(sample_r2)),
        'r2_std': float(sample_r2.std())
    },
    'regime_metrics': {
        'dry_end': {
            'rmse': float(rmse_dry) if rmse_dry is not None else None,
            'mae': float(mae_dry) if mae_dry is not None else None,
            'threshold_kpa': float(dry_end_threshold)
        },
        'wet_end': {
            'rmSE': float(rmse_wet) if rmse_wet is not None else None,
            'mae': float(mae_wet) if mae_wet is not None else None,
            'threshold_kpa': float(wet_end_threshold)
        }
    },
    'physics_compliance': {
        'monotonicity_rate': float(monotonicity_rate),
        'boundary_satisfaction_rate': float(boundary_rate)
    },
    'baseline_comparison': baseline_metrics,
    'n_test_samples': int(len(X_test))
}

results_file = RESULTS_DIR / "evaluation_results_final.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"  ✓ Saved: {results_file}")

# Create summary text file
summary_file = RESULTS_DIR / "evaluation_summary_final.txt"
with open(summary_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("PINN Evaluation Summary (Final Model)\n")
    f.write("="*80 + "\n\n")
    
    f.write("GLOBAL METRICS (Physical Space):\n")
    f.write(f"  RMSE: {rmse_global:.6f}\n")
    f.write(f"  MAE:  {mae_global:.6f}\n")
    f.write(f"  R²:   {r2_global:.6f}\n\n")
    
    f.write("PER-SAMPLE METRICS:\n")
    f.write(f"  RMSE - Mean: {sample_rmse.mean():.6f}, Median: {np.median(sample_rmse):.6f}\n")
    f.write(f"  MAE  - Mean: {sample_mae.mean():.6f}, Median: {np.median(sample_mae):.6f}\n")
    f.write(f"  R²   - Mean: {sample_r2.mean():.6f}, Median: {np.median(sample_r2):.6f}\n\n")
    
    if rmse_dry is not None:
        f.write(f"DRY-END METRICS (s > {dry_end_threshold:.0e} kPa):\n")
        f.write(f"  RMSE: {rmse_dry:.6f}\n")
        f.write(f"  MAE:  {mae_dry:.6f}\n\n")
    
    f.write("PHYSICS COMPLIANCE:\n")
    f.write(f"  Monotonicity: {monotonicity_rate*100:.2f}%\n")
    f.write(f"  Boundary satisfaction: {boundary_rate*100:.2f}%\n\n")
    
    if baseline_metrics:
        f.write("BASELINE COMPARISON:\n")
        f.write(f"  Best baseline: {baseline_metrics['model']}\n")
        f.write(f"    RMSE: {baseline_metrics['rmse']:.6f} (PINN: {rmse_global:.6f})\n")
        f.write(f"    MAE:  {baseline_metrics['mae']:.6f} (PINN: {mae_global:.6f})\n")
        f.write(f"    R²:   {baseline_metrics['r2']:.6f} (PINN: {r2_global:.6f})\n")

print(f"  ✓ Saved: {summary_file}")

print("\n" + "="*80)
print("Evaluation Complete!")
print("="*80)
print(f"\nResults saved to: {RESULTS_DIR}")
print(f"Visualizations saved to: {viz_dir}")
print(f"\nNext: Review results and prepare paper sections")
