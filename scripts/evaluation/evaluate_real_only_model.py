#!/usr/bin/env python3
"""
Evaluate PINN Model Trained on Real Data Only (Previous Better Model)
This should be the model that achieved RMSE ~0.07, MAE ~0.018
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import tensorflow as tf
import numpy as np
import pandas as pd
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from models.pinn_monotonic import MonotonicPINN
from models.pinn import PhysicsEncodingLayer

print("="*80)
print("Evaluating PINN Model (Real Data Only - Previous Better)")
print("="*80)

# Try to find model trained on real data only
# The previous better model was likely in results_pinn_final or results_pinn_optimized
possible_model_paths = [
    Path("results_pinn_final/checkpoints/pinn_best_model_final.keras"),
    Path("results_pinn_final/checkpoints/pinn_final_model_final.keras"),
    Path("results_pinn_optimized/checkpoints/pinn_best_model_optimized.keras"),
]

# Use ORIGINAL test data (data_processed, not data_pinn_normalized)
# This is the same test set used in the previous better evaluation
DATA_DIR = Path("data_processed")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = np.load(DATA_DIR / "y_test.npy")
suction_grid = np.load(DATA_DIR / "suction_grid.npy")

# Also try data_pinn (if it exists)
if not (DATA_DIR / "X_test.csv").exists():
    DATA_DIR = Path("data_pinn")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = np.load(DATA_DIR / "y_test.npy")
    suction_grid = np.load(DATA_DIR / "suction_grid.npy")
    metadata = json.load(open(DATA_DIR / "metadata.json"))
else:
    # Load metadata - need to determine feature columns
    metadata_file = Path("data_pinn_normalized/metadata.json")
    if metadata_file.exists():
        metadata = json.load(open(metadata_file))
    else:
        # Fallback: use all columns except target columns
        metadata = {'feature_cols': [col for col in X_test.columns if col not in ['theta_s', 'theta_r']]}

theta_s_test = X_test['theta_s'].values
theta_r_test = X_test['theta_r'].values

print(f"\nTest data: {DATA_DIR}")
print(f"  Test samples: {len(X_test)}")
print(f"  SWCC points: {y_test.shape[1]}")

# Find model
model_path = None
for path in possible_model_paths:
    if path.exists():
        model_path = path
        print(f"\nFound model: {model_path}")
        break

if not model_path:
    raise FileNotFoundError("No previous model found!")

# Load model
print(f"\nLoading model: {model_path}")
model = MonotonicPINN(
    soil_prop_dim=len(metadata.get('feature_cols', X_test.columns)),
    suction_points=y_test.shape[1],
    physics_units=128,
    hidden_dims=[128, 256, 128, 64]
)

# Build model
feature_cols = metadata.get('feature_cols', [col for col in X_test.columns if col not in ['theta_s', 'theta_r']])
dummy_soil = tf.random.normal([1, len(feature_cols)])
dummy_suction = tf.random.normal([1, y_test.shape[1]])
_ = model({'soil_props': dummy_soil, 'suction': dummy_suction})

# Load weights
try:
    saved_model = tf.keras.models.load_model(
        str(model_path),
        custom_objects={'MonotonicPINN': MonotonicPINN, 'PhysicsEncodingLayer': PhysicsEncodingLayer},
        compile=False
    )
    model.set_weights(saved_model.get_weights())
    print("  ✓ Model loaded successfully")
except Exception as e:
    print(f"  ✗ Error loading model: {e}")
    raise

# Predict
print("\nMaking predictions...")
y_pred_norm = []

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

# Denormalize
y_pred_physical = np.zeros_like(y_pred_norm)
for i in range(len(X_test)):
    theta_range = theta_s_test[i] - theta_r_test[i]
    y_pred_physical[i] = theta_r_test[i] + y_pred_norm[i] * theta_range

print(f"  ✓ Predictions complete")

# Compute metrics
print("\n" + "="*80)
print("Performance Metrics (Real-Only Trained Model)")
print("="*80)

# Global metrics
y_true_flat = y_test.flatten()
y_pred_flat = y_pred_physical.flatten()
mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
y_true_clean = y_true_flat[mask]
y_pred_clean = y_pred_flat[mask]

rmse_global = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
mae_global = mean_absolute_error(y_true_clean, y_pred_clean)
r2_global = r2_score(y_true_clean, y_pred_clean)

# Per-sample metrics
sample_rmse = []
sample_mae = []
for i in range(len(X_test)):
    y_t = y_test[i]
    y_p = y_pred_physical[i]
    mask_i = ~(np.isnan(y_t) | np.isnan(y_p))
    if mask_i.sum() > 0:
        sample_rmse.append(np.sqrt(mean_squared_error(y_t[mask_i], y_p[mask_i])))
        sample_mae.append(mean_absolute_error(y_t[mask_i], y_p[mask_i]))

sample_rmse = np.array(sample_rmse)
sample_mae = np.array(sample_mae)

# Dry-end metrics
dry_end_threshold = 1e4
dry_end_mask = suction_grid > dry_end_threshold
dry_end_indices = np.where(dry_end_mask)[0]

if len(dry_end_indices) > 0:
    y_true_dry = y_test[:, dry_end_indices]
    y_pred_dry = y_pred_physical[:, dry_end_indices]
    mask_dry = ~(np.isnan(y_true_dry) | np.isnan(y_pred_dry))
    y_true_dry_clean = y_true_dry[mask_dry]
    y_pred_dry_clean = y_pred_dry[mask_dry]
    rmse_dry = np.sqrt(mean_squared_error(y_true_dry_clean, y_pred_dry_clean))
    mae_dry = mean_absolute_error(y_true_dry_clean, y_pred_dry_clean)
else:
    rmse_dry = None
    mae_dry = None

# Monotonicity check
monotonic_count = 0
for i in range(len(y_pred_physical)):
    diff = np.diff(y_pred_physical[i])
    if np.all(diff <= 0):
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

print(f"\nGlobal Metrics:")
print(f"  RMSE: {rmse_global:.6f}")
print(f"  MAE:  {mae_global:.6f}")
print(f"  R²:   {r2_global:.6f}")

print(f"\nPer-Sample Metrics:")
print(f"  RMSE - Mean: {sample_rmse.mean():.6f}, Median: {np.median(sample_rmse):.6f}")
print(f"  MAE  - Mean: {sample_mae.mean():.6f}, Median: {np.median(sample_mae):.6f}")

if rmse_dry is not None:
    print(f"\nDry-end (s > {dry_end_threshold:.0e} kPa):")
    print(f"  RMSE: {rmse_dry:.6f}")
    print(f"  MAE:  {mae_dry:.6f}")

print(f"\nPhysics Compliance:")
print(f"  Monotonicity: {monotonic_count}/{len(y_pred_physical)} ({monotonicity_rate*100:.2f}%)")
print(f"  Boundary: {boundary_satisfied}/{len(y_pred_physical)} ({boundary_rate*100:.2f}%)")

# Save results
results = {
    'model_path': str(model_path),
    'data_source': str(DATA_DIR),
    'global_metrics': {
        'rmse': float(rmse_global),
        'mae': float(mae_global),
        'r2': float(r2_global)
    },
    'per_sample_metrics': {
        'rmse_mean': float(sample_rmse.mean()),
        'rmse_median': float(np.median(sample_rmse)),
        'mae_mean': float(sample_mae.mean()),
        'mae_median': float(np.median(sample_mae))
    },
    'dry_end': {
        'rmse': float(rmse_dry) if rmse_dry is not None else None,
        'mae': float(mae_dry) if mae_dry is not None else None
    },
    'physics_compliance': {
        'monotonicity_rate': float(monotonicity_rate),
        'boundary_rate': float(boundary_rate)
    }
}

output_file = Path("results_pinn_fixed/evaluation_real_only_model.json")
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Saved results: {output_file}")

print("\n" + "="*80)
print("Evaluation Complete!")
print("="*80)
