#!/usr/bin/env python3
"""
Final Comprehensive Evaluation of Best PINN Model
This is the model trained on real data only (normalized) - RMSE 0.077, MAE 0.061, R² 0.771
This will be the MAIN model for the Q1 paper.
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

from models.pinn_monotonic import MonotonicPINN
from models.pinn import PhysicsEncodingLayer
from training_pinn.config_pinn_fixed import DATA_CONFIG, RESULTS_DIR

plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 13

print("="*80)
print("Final Comprehensive Evaluation: Best PINN Model (Real-Only Training)")
print("="*80)

# Load best model
CHECKPOINT_DIR = Path("results_pinn_fixed/checkpoints")
best_model_path = CHECKPOINT_DIR / "pinn_best_model_fixed.keras"

print(f"\n1. Loading model: {best_model_path}")

metadata = json.load(open(DATA_CONFIG['metadata_file']))
model = MonotonicPINN(
    soil_prop_dim=metadata['n_features'],
    suction_points=metadata['n_swcc_points'],
    physics_units=128,
    hidden_dims=[128, 256, 128, 64]
)

dummy_soil = tf.random.normal([1, metadata['n_features']])
dummy_suction = tf.random.normal([1, metadata['n_swcc_points']])
_ = model({'soil_props': dummy_soil, 'suction': dummy_suction})

saved_model = tf.keras.models.load_model(
    str(best_model_path),
    custom_objects={'MonotonicPINN': MonotonicPINN, 'PhysicsEncodingLayer': PhysicsEncodingLayer},
    compile=False
)
model.set_weights(saved_model.get_weights())
print("  ✓ Model loaded")

# Load test data
X_test = pd.read_csv(DATA_CONFIG['test_file'])
y_test_original = np.load(DATA_CONFIG['y_test_original_file'])
suction_grid = np.load(DATA_CONFIG['suction_grid_file'])
theta_s_test = X_test['theta_s'].values
theta_r_test = X_test['theta_r'].values
feature_cols = metadata['feature_cols']

print(f"  Test samples: {len(X_test)}")

# Predict
print("\n2. Making predictions...")
y_pred_norm = []
batch_size = 32
for i in range(0, len(X_test), batch_size):
    batch_end = min(i + batch_size, len(X_test))
    batch_soil = X_test.iloc[i:batch_end][feature_cols].values.astype(np.float32)
    batch_suction = np.tile(suction_grid, (batch_end - i, 1)).astype(np.float32)
    inputs = {'soil_props': tf.constant(batch_soil), 'suction': tf.constant(batch_suction)}
    theta_pred_norm_batch = model(inputs, training=False)
    y_pred_norm.extend(theta_pred_norm_batch.numpy())

y_pred_norm = np.array(y_pred_norm)
y_pred_physical = np.zeros_like(y_pred_norm)
for i in range(len(X_test)):
    theta_range = theta_s_test[i] - theta_r_test[i]
    y_pred_physical[i] = theta_r_test[i] + y_pred_norm[i] * theta_range

print("  ✓ Predictions complete")

# Compute metrics
print("\n3. Computing metrics...")

# Global
y_true_flat = y_test_original.flatten()
y_pred_flat = y_pred_physical.flatten()
mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
rmse_global = np.sqrt(mean_squared_error(y_true_flat[mask], y_pred_flat[mask]))
mae_global = mean_absolute_error(y_true_flat[mask], y_pred_flat[mask])
r2_global = r2_score(y_true_flat[mask], y_pred_flat[mask])

# Per-sample
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

# Dry-end
dry_end_threshold = 1e4
dry_end_mask = suction_grid > dry_end_threshold
dry_end_indices = np.where(dry_end_mask)[0]
if len(dry_end_indices) > 0:
    y_true_dry = y_test_original[:, dry_end_indices]
    y_pred_dry = y_pred_physical[:, dry_end_indices]
    mask_dry = ~(np.isnan(y_true_dry) | np.isnan(y_pred_dry))
    rmse_dry = np.sqrt(mean_squared_error(y_true_dry[mask_dry], y_pred_dry[mask_dry]))
    mae_dry = mean_absolute_error(y_true_dry[mask_dry], y_pred_dry[mask_dry])
else:
    rmse_dry = None
    mae_dry = None

# Wet-end
wet_end_threshold = 1e2
wet_end_mask = suction_grid < wet_end_threshold
wet_end_indices = np.where(wet_end_mask)[0]
if len(wet_end_indices) > 0:
    y_true_wet = y_test_original[:, wet_end_indices]
    y_pred_wet = y_pred_physical[:, wet_end_indices]
    mask_wet = ~(np.isnan(y_true_wet) | np.isnan(y_pred_wet))
    rmse_wet = np.sqrt(mean_squared_error(y_true_wet[mask_wet], y_pred_wet[mask_wet]))
    mae_wet = mean_absolute_error(y_true_wet[mask_wet], y_pred_wet[mask_wet])
else:
    rmse_wet = None
    mae_wet = None

# Physics
monotonic_count = sum(1 for i in range(len(y_pred_physical)) 
                     if np.all(np.diff(y_pred_physical[i]) <= 0))
monotonicity_rate = monotonic_count / len(y_pred_physical)

boundary_count = sum(1 for i in range(len(y_pred_physical))
                    if (y_pred_physical[i].min() >= theta_r_test[i] - 0.01 and
                        y_pred_physical[i].max() <= theta_s_test[i] + 0.01))
boundary_rate = boundary_count / len(y_pred_physical)

print(f"\n{'='*80}")
print("PERFORMANCE METRICS (BEST MODEL - MAIN RESULT)")
print(f"{'='*80}")

print(f"\nGlobal Metrics:")
print(f"  RMSE: {rmse_global:.6f}")
print(f"  MAE:  {mae_global:.6f}")
print(f"  R²:   {r2_global:.6f}")

print(f"\nPer-Sample Metrics:")
print(f"  RMSE - Mean: {sample_rmse.mean():.6f}, Median: {np.median(sample_rmse):.6f}, Std: {sample_rmse.std():.6f}")
print(f"  MAE  - Mean: {sample_mae.mean():.6f}, Median: {np.median(sample_mae):.6f}, Std: {sample_mae.std():.6f}")
print(f"  R²   - Mean: {sample_r2.mean():.6f}, Median: {np.median(sample_r2):.6f}, Std: {sample_r2.std():.6f}")

if rmse_dry:
    print(f"\nDry-end (s > {dry_end_threshold:.0e} kPa):")
    print(f"  RMSE: {rmse_dry:.6f}")
    print(f"  MAE:  {mae_dry:.6f}")

if rmse_wet:
    print(f"\nWet-end (s < {wet_end_threshold:.0e} kPa):")
    print(f"  RMSE: {rmse_wet:.6f}")
    print(f"  MAE:  {mae_wet:.6f}")

print(f"\nPhysics Compliance:")
print(f"  Monotonicity: {monotonic_count}/{len(y_pred_physical)} ({monotonicity_rate*100:.2f}%)")
print(f"  Boundary: {boundary_count}/{len(y_pred_physical)} ({boundary_rate*100:.2f}%)")

# Load baseline for comparison
print("\n4. Loading baseline comparison...")
baseline_report_file = Path("results_baseline/baseline_report.json")
baseline_metrics = None
if baseline_report_file.exists():
    with open(baseline_report_file, 'r') as f:
        baseline_data = json.load(f)
    baseline_metrics = {
        'model': baseline_data.get('best_model', 'gradient_boosting'),
        'rmse': float(baseline_data.get('best_rmse', 0)),
        'mae': float(baseline_data.get('best_mae', 0)),
        'r2': float(baseline_data.get('best_r2', 0))
    }
    print(f"  Best baseline: {baseline_metrics['model']}")
    print(f"    RMSE: {baseline_metrics['rmse']:.6f}")
    print(f"    MAE:  {baseline_metrics['mae']:.6f}")
    print(f"    R²:   {baseline_metrics['r2']:.6f}")

# Create comparison table
print("\n5. Creating comparison table...")
comparison_data = {
    'Model': ['PINN (Best - Real-Only)', 'Gradient Boosting (Baseline)'],
    'Global RMSE': [f"{rmse_global:.6f}", f"{baseline_metrics['rmse']:.6f}" if baseline_metrics else "N/A"],
    'Global MAE': [f"{mae_global:.6f}", f"{baseline_metrics['mae']:.6f}" if baseline_metrics else "N/A"],
    'Global R²': [f"{r2_global:.6f}", f"{baseline_metrics['r2']:.6f}" if baseline_metrics else "N/A"],
    'Median Per-Sample RMSE': [f"{np.median(sample_rmse):.6f}", "N/A"],
    'Dry-end RMSE (s > 10⁴ kPa)': [f"{rmse_dry:.6f}" if rmse_dry else "N/A", "N/A"],
    'Monotonicity (%)': [f"{monotonicity_rate*100:.2f}%", "Not enforced"],
    'Boundary Satisfaction (%)': [f"{boundary_rate*100:.2f}%", "Not enforced"]
}

df = pd.DataFrame(comparison_data)
output_file = Path("results_pinn_fixed/comparison_table_main_model.csv")
df.to_csv(output_file, index=False)
print(f"  ✓ Saved: {output_file}")

# Save comprehensive results
results = {
    'model_path': str(best_model_path),
    'training_data': 'real only (normalized)',
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
            'rmse': float(rmse_dry) if rmse_dry else None,
            'mae': float(mae_dry) if mae_dry else None,
            'threshold_kpa': float(dry_end_threshold)
        },
        'wet_end': {
            'rmse': float(rmse_wet) if rmse_wet else None,
            'mae': float(mae_wet) if mae_wet else None,
            'threshold_kpa': float(wet_end_threshold)
        }
    },
    'physics_compliance': {
        'monotonicity_rate': float(monotonicity_rate),
        'boundary_rate': float(boundary_rate)
    },
    'baseline_comparison': baseline_metrics,
    'n_test_samples': int(len(X_test))
}

results_file = RESULTS_DIR / "evaluation_results_main_model.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"  ✓ Saved: {results_file}")

# Create summary
summary_file = RESULTS_DIR / "evaluation_summary_main_model.txt"
with open(summary_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("PINN Evaluation Summary - MAIN MODEL (Best - Real-Only Training)\n")
    f.write("="*80 + "\n\n")
    f.write("MODEL: pinn_best_model_fixed.keras\n")
    f.write("TRAINING DATA: Real data only (normalized to [0,1] per sample)\n")
    f.write("TEST SAMPLES: 84 (real data only)\n\n")
    
    f.write("GLOBAL METRICS:\n")
    f.write(f"  RMSE: {rmse_global:.6f}\n")
    f.write(f"  MAE:  {mae_global:.6f}\n")
    f.write(f"  R²:   {r2_global:.6f}\n\n")
    
    f.write("PER-SAMPLE METRICS:\n")
    f.write(f"  RMSE - Mean: {sample_rmse.mean():.6f}, Median: {np.median(sample_rmse):.6f}\n")
    f.write(f"  MAE  - Mean: {sample_mae.mean():.6f}, Median: {np.median(sample_mae):.6f}\n")
    f.write(f"  R²   - Mean: {sample_r2.mean():.6f}, Median: {np.median(sample_r2):.6f}\n\n")
    
    if rmse_dry:
        f.write(f"DRY-END (s > {dry_end_threshold:.0e} kPa):\n")
        f.write(f"  RMSE: {rmse_dry:.6f}\n")
        f.write(f"  MAE:  {mae_dry:.6f}\n\n")
    
    if rmse_wet:
        f.write(f"WET-END (s < {wet_end_threshold:.0e} kPa):\n")
        f.write(f"  RMSE: {rmse_wet:.6f}\n")
        f.write(f"  MAE:  {mae_wet:.6f}\n\n")
    
    f.write("PHYSICS COMPLIANCE:\n")
    f.write(f"  Monotonicity: {monotonicity_rate*100:.2f}%\n")
    f.write(f"  Boundary: {boundary_rate*100:.2f}%\n\n")
    
    if baseline_metrics:
        f.write("BASELINE COMPARISON:\n")
        f.write(f"  Best baseline: {baseline_metrics['model']}\n")
        f.write(f"    RMSE: {baseline_metrics['rmse']:.6f} (PINN: {rmse_global:.6f})\n")
        f.write(f"    MAE:  {baseline_metrics['mae']:.6f} (PINN: {mae_global:.6f})\n")
        f.write(f"    R²:   {baseline_metrics['r2']:.6f} (PINN: {r2_global:.6f})\n")

print(f"  ✓ Saved: {summary_file}")

# Generate plots
print("\n6. Generating visualizations...")
viz_dir = RESULTS_DIR / "visualizations"
viz_dir.mkdir(parents=True, exist_ok=True)

# Representative curves
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
n_samples = min(6, len(X_test))
indices = np.linspace(0, len(X_test)-1, n_samples, dtype=int)

for idx, ax in enumerate(axes[:n_samples]):
    i = int(indices[idx])
    ax.semilogx(suction_grid, y_test_original[i], 'b-', linewidth=2, label='Observed', alpha=0.8)
    ax.semilogx(suction_grid, y_pred_physical[i], 'r--', linewidth=2, label='PINN Prediction', alpha=0.8)
    ax.axhline(y=theta_s_test[i], color='g', linestyle=':', alpha=0.5, label='θ_s')
    ax.axhline(y=theta_r_test[i], color='orange', linestyle=':', alpha=0.5, label='θ_r')
    ax.set_xlabel('Suction (kPa)', fontsize=12)
    ax.set_ylabel('Water Content (θ)', fontsize=12)
    ax.set_title(f'Sample {i+1}', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')

for ax in axes[n_samples:]:
    ax.remove()

plt.tight_layout()
plt.savefig(viz_dir / 'pinn_main_representative_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {viz_dir / 'pinn_main_representative_curves.png'}")

# Error vs suction
fig, ax = plt.subplots(figsize=(10, 6))
errors = np.abs(y_test_original - y_pred_physical)
mean_error = np.nanmean(errors, axis=0)
std_error = np.nanstd(errors, axis=0)

ax.semilogx(suction_grid, mean_error, 'b-', linewidth=2, label='Mean |Error|')
ax.fill_between(suction_grid, mean_error - std_error, mean_error + std_error, 
                alpha=0.3, color='blue', label='±1 Std')
ax.set_xlabel('Suction (kPa)', fontsize=13)
ax.set_ylabel('Absolute Error |θ_true - θ_pred|', fontsize=13)
ax.set_title('Error vs Suction (Main Model)', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(viz_dir / 'pinn_main_error_vs_suction.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {viz_dir / 'pinn_main_error_vs_suction.png'}")

# RMSE distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(sample_rmse, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(sample_rmse.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {sample_rmse.mean():.4f}')
ax.axvline(np.median(sample_rmse), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(sample_rmse):.4f}')
ax.set_xlabel('Per-Sample RMSE', fontsize=13)
ax.set_ylabel('Frequency', fontsize=13)
ax.set_title('Distribution of Per-Sample RMSE (Main Model)', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(viz_dir / 'pinn_main_rmse_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {viz_dir / 'pinn_main_rmse_distribution.png'}")

# Comparison with baseline
if baseline_metrics:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    models = ['PINN (Best)', baseline_metrics['model']]
    colors = ['#2E86AB', '#F18F01']
    
    # RMSE
    ax = axes[0]
    rmse_vals = [rmse_global, baseline_metrics['rmse']]
    bars = ax.bar(models, rmse_vals, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('RMSE', fontsize=13)
    ax.set_title('RMSE Comparison', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rmse_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11)
    
    # MAE
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
    
    # R²
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
    plt.savefig(viz_dir / 'pinn_main_vs_baseline_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {viz_dir / 'pinn_main_vs_baseline_comparison.png'}")

print("\n" + "="*80)
print("Evaluation Complete!")
print("="*80)
print(f"\n✓ Main model evaluation saved to: {RESULTS_DIR}")
print(f"✓ This is the model to use for the Q1 paper")
print(f"\nKey Results:")
if baseline_metrics:
    print(f"  RMSE: {rmse_global:.6f} (vs baseline {baseline_metrics['rmse']:.6f})")
    print(f"  MAE:  {mae_global:.6f} (vs baseline {baseline_metrics['mae']:.6f})")
    print(f"  R²:   {r2_global:.6f} (vs baseline {baseline_metrics['r2']:.6f})")
else:
    print(f"  RMSE: {rmse_global:.6f}")
    print(f"  MAE:  {mae_global:.6f}")
    print(f"  R²:   {r2_global:.6f}")
print(f"  Monotonicity: {monotonicity_rate*100:.2f}%")
print(f"  Boundary: {boundary_rate*100:.2f}%")
