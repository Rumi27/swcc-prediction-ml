#!/usr/bin/env python3
"""
Compute Baseline Metrics for Regime-Specific Comparison
Evaluates Gradient Boosting on dry-end and sparse-data scenarios
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import joblib

print("="*80)
print("Computing Baseline Regime-Specific Metrics")
print("="*80)

# Load baseline model
baseline_model_path = Path("results_baseline/baseline_models/gradient_boosting_model.pkl")
if not baseline_model_path.exists():
    print("⚠ Baseline model not found. Skipping regime-specific evaluation.")
    exit(0)

print(f"\nLoading baseline model: {baseline_model_path}")
model = joblib.load(baseline_model_path)
scaler = joblib.load(Path("results_baseline/baseline_models/scaler.pkl"))

# Load test data
X_test = pd.read_csv("data_processed/X_test.csv")
y_test = np.load("data_processed/y_test.npy")
suction_grid = np.load("data_processed/suction_grid.npy")

print(f"  Test samples: {len(X_test)}")

# Prepare features (same as baseline training)
feature_cols = ['D10', 'D30', 'D50', 'D60', 'D90', 'Cu', 'Cc', 
                'bulk_density', 'porosity', 'clay_pct', 'silt_pct', 'sand_pct',
                'OM_content', 'pH', 'theta_s', 'theta_r']

X_test_features = X_test[feature_cols].values
X_test_scaled = scaler.transform(X_test_features)

# Predict
print("\nMaking baseline predictions...")
y_pred_baseline = model.predict(X_test_scaled)

print(f"  Predictions shape: {y_pred_baseline.shape}")

# ============================================================================
# 1. DRY-END METRICS
# ============================================================================
print("\n1. Computing dry-end metrics...")

dry_end_threshold = 1e4  # kPa
dry_end_indices = np.where(suction_grid > dry_end_threshold)[0]

if len(dry_end_indices) > 0:
    y_test_dry = y_test[:, dry_end_indices]
    y_pred_dry = y_pred_baseline[:, dry_end_indices]
    
    y_test_dry_flat = y_test_dry.flatten()
    y_pred_dry_flat = y_pred_dry.flatten()
    
    mask_dry = ~(np.isnan(y_test_dry_flat) | np.isnan(y_pred_dry_flat))
    
    rmse_dry = np.sqrt(mean_squared_error(y_test_dry_flat[mask_dry], y_pred_dry_flat[mask_dry]))
    mae_dry = mean_absolute_error(y_test_dry_flat[mask_dry], y_pred_dry_flat[mask_dry])
    
    print(f"  Dry-end threshold: > {dry_end_threshold:.0f} kPa")
    print(f"  Points per sample: {len(dry_end_indices)}")
    print(f"  RMSE: {rmse_dry:.6f}")
    print(f"  MAE: {mae_dry:.6f}")
else:
    rmse_dry = np.nan
    mae_dry = np.nan

# ============================================================================
# 2. SPARSE DATA METRICS
# ============================================================================
print("\n2. Computing sparse-data metrics...")

n_sparse_points = 8
sparse_indices = np.linspace(0, len(suction_grid)-1, n_sparse_points, dtype=int)

y_test_sparse = y_test[:, sparse_indices]
y_pred_sparse = y_pred_baseline[:, sparse_indices]

y_test_sparse_flat = y_test_sparse.flatten()
y_pred_sparse_flat = y_pred_sparse.flatten()

mask_sparse = ~(np.isnan(y_test_sparse_flat) | np.isnan(y_pred_sparse_flat))

rmse_sparse = np.sqrt(mean_squared_error(y_test_sparse_flat[mask_sparse], y_pred_sparse_flat[mask_sparse]))
mae_sparse = mean_absolute_error(y_test_sparse_flat[mask_sparse], y_pred_sparse_flat[mask_sparse])

print(f"  Sparse points: {n_sparse_points} per curve")
print(f"  RMSE: {rmse_sparse:.6f}")
print(f"  MAE: {mae_sparse:.6f}")

# ============================================================================
# 3. MONOTONICITY CHECK
# ============================================================================
print("\n3. Checking monotonicity...")

mono_violations = 0
for i in range(len(y_pred_baseline)):
    diff = y_pred_baseline[i, :-1] - y_pred_baseline[i, 1:]
    if np.any(diff < -1e-6):
        mono_violations += 1

monotonicity_rate = 1.0 - mono_violations / len(X_test)
print(f"  Monotonicity rate: {monotonicity_rate*100:.1f}%")
print(f"  Violations: {mono_violations}/{len(X_test)}")

# ============================================================================
# 4. BOUNDARY CHECK
# ============================================================================
print("\n4. Checking boundary conditions...")

theta_s_test = X_test['theta_s'].values
theta_r_test = X_test['theta_r'].values

boundary_violations = 0
for i in range(len(X_test)):
    if np.any(y_pred_baseline[i] < theta_r_test[i] - 1e-6) or \
       np.any(y_pred_baseline[i] > theta_s_test[i] + 1e-6):
        boundary_violations += 1

boundary_rate = 1.0 - boundary_violations / len(X_test)
print(f"  Boundary rate: {boundary_rate*100:.1f}%")
print(f"  Violations: {boundary_violations}/{len(X_test)}")

# ============================================================================
# 5. SAVE RESULTS
# ============================================================================
results = {
    'dry_end': {
        'rmse': float(rmse_dry) if not np.isnan(rmse_dry) else None,
        'mae': float(mae_dry) if not np.isnan(mae_dry) else None,
        'threshold_kpa': float(dry_end_threshold)
    },
    'sparse_data': {
        'rmse': float(rmse_sparse),
        'mae': float(mae_sparse),
        'n_points': int(n_sparse_points)
    },
    'physics_compliance': {
        'monotonicity_rate': float(monotonicity_rate),
        'boundary_rate': float(boundary_rate)
    }
}

import json
results_file = Path("results_baseline/baseline_regime_metrics.json")
results_file.parent.mkdir(parents=True, exist_ok=True)
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Saved: {results_file}")

# Update comparison table
comparison_file = Path("results_pinn_fixed/comparison_table.csv")
if comparison_file.exists():
    comparison_df = pd.read_csv(comparison_file)
    
    # Update dry-end
    if not np.isnan(rmse_dry):
        comparison_df.loc[comparison_df['Metric'] == 'Dry-end RMSE (s > 10⁴ kPa)', 
                         'Gradient Boosting'] = f"{rmse_dry:.6f}"
    
    # Update sparse-data
    comparison_df.loc[comparison_df['Metric'] == 'Sparse-data RMSE (8 points)', 
                     'Gradient Boosting'] = f"{rmse_sparse:.6f}"
    
    # Update monotonicity
    comparison_df.loc[comparison_df['Metric'] == 'Monotonicity (%)', 
                     'Gradient Boosting'] = f"{monotonicity_rate*100:.1f}%"
    
    # Update boundary
    comparison_df.loc[comparison_df['Metric'] == 'Boundary Satisfaction (%)', 
                     'Gradient Boosting'] = f"{boundary_rate*100:.1f}%"
    
    comparison_df.to_csv(comparison_file, index=False)
    print(f"✓ Updated: {comparison_file}")

print("\n" + "="*80)
print("Baseline Regime Metrics Complete")
print("="*80)
