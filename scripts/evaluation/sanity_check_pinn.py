#!/usr/bin/env python3
"""
Sanity Check for PINN: Data Scaling, Denormalization, and Checkpoint Verification
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
from sklearn.metrics import mean_squared_error, r2_score

from models.pinn import PINN, PhysicsEncodingLayer
from training_pinn.config_pinn_optimized import DATA_CONFIG, CHECKPOINT_DIR

print("="*80)
print("PINN Sanity Check: Scaling, Denormalization, and Checkpoint Verification")
print("="*80)

# ============================================================================
# 1. CHECK DATA SCALING
# ============================================================================
print("\n" + "="*80)
print("1. CHECKING DATA SCALING")
print("="*80)

# Load training data
X_train = pd.read_csv(DATA_CONFIG['train_file'])
y_train = np.load(DATA_CONFIG['y_train_file'])
suction_grid = np.load(DATA_CONFIG['suction_grid_file'])
metadata = json.load(open(DATA_CONFIG['metadata_file']))

print(f"\nTraining Data Info:")
print(f"  Samples: {len(X_train)}")
print(f"  Features: {X_train.shape[1]}")
print(f"  SWCC points: {y_train.shape[1]}")

# Check theta ranges
theta_s = X_train['theta_s'].values
theta_r = X_train['theta_r'].values

print(f"\nTheta Ranges (from features):")
print(f"  theta_s: [{theta_s.min():.4f}, {theta_s.max():.4f}]")
print(f"  theta_r: [{theta_r.min():.4f}, {theta_r.max():.4f}]")

print(f"\nSWCC Curves (y_train) Ranges:")
print(f"  y_train: [{y_train.min():.4f}, {y_train.max():.4f}]")
print(f"  y_train mean: {y_train.mean():.4f}")
print(f"  y_train std: {y_train.std():.4f}")

# Check if y_train is normalized
# If normalized to [0,1], it should be in range [0, 1]
# If normalized to [-1,1], it should be in range [-1, 1]
# If not normalized, it should match theta_s/theta_r ranges

is_normalized_01 = (y_train.min() >= 0 and y_train.max() <= 1.0)
is_normalized_neg11 = (y_train.min() >= -1.0 and y_train.max() <= 1.0)
is_in_theta_range = (y_train.min() >= theta_r.min() and y_train.max() <= theta_s.max())

print(f"\nScaling Check:")
print(f"  In [0, 1] range: {is_normalized_01}")
print(f"  In [-1, 1] range: {is_normalized_neg11}")
print(f"  In [theta_r, theta_s] range: {is_in_theta_range}")

# Check a few samples
print(f"\nSample Check (first 3 samples):")
for i in range(min(3, len(X_train))):
    print(f"\n  Sample {i+1}:")
    print(f"    theta_s: {theta_s[i]:.4f}")
    print(f"    theta_r: {theta_r[i]:.4f}")
    print(f"    y_train range: [{y_train[i].min():.4f}, {y_train[i].max():.4f}]")
    print(f"    y_train[0] (first point): {y_train[i, 0]:.4f}")
    print(f"    y_train[-1] (last point): {y_train[i, -1]:.4f}")
    print(f"    Expected: y[0] ≈ theta_s, y[-1] ≈ theta_r")
    print(f"    Actual: y[0] = {y_train[i, 0]:.4f} (expected ~{theta_s[i]:.4f})")
    print(f"    Actual: y[-1] = {y_train[i, -1]:.4f} (expected ~{theta_r[i]:.4f})")

# ============================================================================
# 2. CHECK MODEL OUTPUT SCALING
# ============================================================================
print("\n" + "="*80)
print("2. CHECKING MODEL OUTPUT SCALING")
print("="*80)

# Load model architecture
model = PINN(
    soil_prop_dim=metadata['n_features'],
    suction_points=metadata['n_swcc_points'],
    physics_units=128,
    hidden_dims=[128, 256, 128, 64]
)

# Build model
dummy_soil = tf.random.normal([1, metadata['n_features']])
dummy_suction = tf.random.normal([1, metadata['n_swcc_points']])
_ = model({'soil_props': dummy_soil, 'suction': dummy_suction})

print("\nModel Architecture:")
print(f"  Output layer activation: sigmoid (outputs [0, 1])")
print(f"  Denormalization in call(): theta = theta_norm * (theta_s - theta_r) + theta_r")

# Check what the model outputs without denormalization
# We need to check the output_layer output before denormalization
print("\nModel Output Check (before loading weights):")
sample_soil = X_train.iloc[:1].values.astype(np.float32)
sample_suction = np.tile(suction_grid, (1, 1)).astype(np.float32)

inputs = {'soil_props': tf.constant(sample_soil), 'suction': tf.constant(sample_suction)}
theta_pred = model(inputs, training=False)

print(f"  Predicted theta range: [{theta_pred.numpy().min():.4f}, {theta_pred.numpy().max():.4f}]")
print(f"  Expected range: [{theta_r[0]:.4f}, {theta_s[0]:.4f}]")
print(f"  Match: {np.allclose(theta_pred.numpy().flatten(), y_train[0], atol=0.1)} (before training)")

# ============================================================================
# 3. LOAD EPOCH 39 CHECKPOINT AND VERIFY
# ============================================================================
print("\n" + "="*80)
print("3. LOADING EPOCH 39 CHECKPOINT (BEST MODEL)")
print("="*80)

# Find epoch 39 checkpoint
checkpoint_files = sorted(CHECKPOINT_DIR.glob("pinn_checkpoint_epoch_*.keras"))
best_checkpoint = None

# Try to find epoch 39 or closest
for cp in checkpoint_files:
    epoch_num = int(cp.stem.split('_')[-1])
    if epoch_num == 39:
        best_checkpoint = cp
        break

if not best_checkpoint:
    # Try best_model
    best_model_path = CHECKPOINT_DIR / "pinn_best_model_optimized.keras"
    if best_model_path.exists():
        best_checkpoint = best_model_path
        print(f"  Using best_model_optimized.keras (should be epoch 39)")
    else:
        # Use earliest checkpoint
        if checkpoint_files:
            best_checkpoint = checkpoint_files[0]
            print(f"  Epoch 39 not found, using earliest: {best_checkpoint.name}")

if best_checkpoint and best_checkpoint.exists():
    print(f"\n  Loading: {best_checkpoint}")
    
    # Save weights before loading
    weights_before = [np.array(w).copy() for w in model.get_weights()]
    first_layer_norm_before = np.linalg.norm(weights_before[0])
    
    try:
        saved_model = tf.keras.models.load_model(
            str(best_checkpoint),
            custom_objects={'PINN': PINN, 'PhysicsEncodingLayer': PhysicsEncodingLayer},
            compile=False
        )
        model.set_weights(saved_model.get_weights())
        
        weights_after = [np.array(w).copy() for w in model.get_weights()]
        first_layer_norm_after = np.linalg.norm(weights_after[0])
        
        print(f"  ✓ Model loaded successfully")
        print(f"  First layer norm before: {first_layer_norm_before:.6f}")
        print(f"  First layer norm after: {first_layer_norm_after:.6f}")
        print(f"  Weights match: {np.allclose(weights_before[0], weights_after[0], atol=1e-6)} (should be False - weights changed)")
        
        # Check a specific weight
        print(f"\n  Sample weight check:")
        print(f"    Before: {weights_before[0][0, 0]:.6f}")
        print(f"    After: {weights_after[0][0, 0]:.6f}")
        print(f"    Difference: {abs(weights_before[0][0, 0] - weights_after[0][0, 0]):.6e}")
        
    except Exception as e:
        print(f"  ✗ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"  ✗ No checkpoint found!")

# ============================================================================
# 4. SANITY CHECK: PREDICT ON TRAINING SAMPLES
# ============================================================================
print("\n" + "="*80)
print("4. SANITY CHECK: PREDICT ON 10 TRAINING SAMPLES")
print("="*80)

# Take 10 training samples
n_samples = min(10, len(X_train))
sample_indices = np.random.choice(len(X_train), n_samples, replace=False)

print(f"\nTesting on {n_samples} training samples...")

predictions = []
observations = []
theta_s_list = []
theta_r_list = []

for idx in sample_indices:
    sample_soil = X_train.iloc[idx:idx+1].values.astype(np.float32)
    sample_suction = np.tile(suction_grid, (1, 1)).astype(np.float32)
    
    inputs = {'soil_props': tf.constant(sample_soil), 'suction': tf.constant(sample_suction)}
    theta_pred = model(inputs, training=False)
    
    predictions.append(theta_pred.numpy()[0])
    observations.append(y_train[idx])
    theta_s_list.append(theta_s[idx])
    theta_r_list.append(theta_r[idx])

predictions = np.array(predictions)
observations = np.array(observations)

# Compute metrics
rmse = np.sqrt(mean_squared_error(observations.flatten(), predictions.flatten()))
r2 = r2_score(observations.flatten(), predictions.flatten())

print(f"\nTraining Set Performance (should be excellent):")
print(f"  RMSE: {rmse:.6f}")
print(f"  R²: {r2:.6f}")

# Check per-sample
print(f"\nPer-Sample RMSE:")
for i, idx in enumerate(sample_indices):
    sample_rmse = np.sqrt(mean_squared_error(observations[i], predictions[i]))
    print(f"  Sample {idx+1}: RMSE = {sample_rmse:.6f}")

# Visual check
print(f"\nCreating visualization...")
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for i in range(n_samples):
    ax = axes[i]
    
    # Observed
    ax.semilogx(suction_grid, observations[i], 'b-', linewidth=2, label='Observed', alpha=0.7)
    
    # Predicted
    ax.semilogx(suction_grid, predictions[i], 'r--', linewidth=2, label='Predicted', alpha=0.7)
    
    # Boundaries
    ax.axhline(theta_s_list[i], color='g', linestyle=':', alpha=0.5, label='θ_s')
    ax.axhline(theta_r_list[i], color='orange', linestyle=':', alpha=0.5, label='θ_r')
    
    sample_rmse = np.sqrt(mean_squared_error(observations[i], predictions[i]))
    ax.set_title(f'Sample {sample_indices[i]+1}\nRMSE: {sample_rmse:.4f}', fontsize=9)
    ax.set_xlabel('Suction (kPa)', fontsize=8)
    ax.set_ylabel('Water Content', fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
viz_path = Path('results_pinn_optimized') / 'visualizations' / 'sanity_check_training_samples.png'
viz_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(viz_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {viz_path}")

# ============================================================================
# 5. CHECK VALIDATION SET PERFORMANCE
# ============================================================================
print("\n" + "="*80)
print("5. VALIDATION SET PERFORMANCE (EPOCH 39 MODEL)")
print("="*80)

X_val = pd.read_csv(DATA_CONFIG['val_file'])
y_val = np.load(DATA_CONFIG['y_val_file'])

print(f"\nValidation Set: {len(X_val)} samples")

val_predictions = []
val_observations = []

for i in range(len(X_val)):
    sample_soil = X_val.iloc[i:i+1].values.astype(np.float32)
    sample_suction = np.tile(suction_grid, (1, 1)).astype(np.float32)
    
    inputs = {'soil_props': tf.constant(sample_soil), 'suction': tf.constant(sample_suction)}
    theta_pred = model(inputs, training=False)
    
    val_predictions.append(theta_pred.numpy()[0])
    val_observations.append(y_val[i])

val_predictions = np.array(val_predictions)
val_observations = np.array(val_observations)

val_rmse = np.sqrt(mean_squared_error(val_observations.flatten(), val_predictions.flatten()))
val_r2 = r2_score(val_observations.flatten(), val_predictions.flatten())

print(f"\nValidation Set Performance:")
print(f"  RMSE: {val_rmse:.6f}")
print(f"  R²: {val_r2:.6f}")

# Compare with training history
history_file = Path('results_pinn_optimized') / 'training_history.json'
if history_file.exists():
    with open(history_file) as f:
        history = json.load(f)
    
    # Find epoch 39
    if 39 in history['epoch']:
        epoch_39_idx = history['epoch'].index(39)
        print(f"\nTraining History (Epoch 39):")
        print(f"  Train Loss: {history['train_total'][epoch_39_idx]:.6f}")
        print(f"  Val Loss: {history['val_total'][epoch_39_idx]:.6f}")
        print(f"  Data Loss: {history['train_data'][epoch_39_idx]:.6f}")
        print(f"  Monotonicity Loss: {history['train_mono'][epoch_39_idx]:.6e}")

# ============================================================================
# 6. CHECK DENORMALIZATION
# ============================================================================
print("\n" + "="*80)
print("6. VERIFYING DENORMALIZATION")
print("="*80)

# Check if predictions are in correct range
print(f"\nPrediction Range Check:")
print(f"  Training predictions: [{predictions.min():.4f}, {predictions.max():.4f}]")
print(f"  Validation predictions: [{val_predictions.min():.4f}, {val_predictions.max():.4f}]")
print(f"  Training observations: [{observations.min():.4f}, {observations.max():.4f}]")
print(f"  Validation observations: [{val_observations.min():.4f}, {val_observations.max():.4f}]")

# Check if predictions respect boundaries
train_boundary_violations = 0
val_boundary_violations = 0

for i in range(n_samples):
    if np.any(predictions[i] < theta_r_list[i] - 1e-6) or np.any(predictions[i] > theta_s_list[i] + 1e-6):
        train_boundary_violations += 1

val_theta_s = X_val['theta_s'].values
val_theta_r = X_val['theta_r'].values

for i in range(len(X_val)):
    if np.any(val_predictions[i] < val_theta_r[i] - 1e-6) or np.any(val_predictions[i] > val_theta_s[i] + 1e-6):
        val_boundary_violations += 1

print(f"\nBoundary Constraint Check:")
print(f"  Training violations: {train_boundary_violations}/{n_samples}")
print(f"  Validation violations: {val_boundary_violations}/{len(X_val)}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SANITY CHECK SUMMARY")
print("="*80)

print(f"\n✅ Data Scaling:")
print(f"   y_train range: [{y_train.min():.4f}, {y_train.max():.4f}]")
print(f"   Normalized to [0,1]: {is_normalized_01}")
print(f"   In theta range: {is_in_theta_range}")

print(f"\n✅ Model Output:")
print(f"   Model denormalizes: theta = theta_norm * (theta_s - theta_r) + theta_r")
print(f"   Predictions in range: [{predictions.min():.4f}, {predictions.max():.4f}]")

print(f"\n✅ Training Performance:")
print(f"   RMSE: {rmse:.6f}")
print(f"   R²: {r2:.6f}")

print(f"\n✅ Validation Performance:")
print(f"   RMSE: {val_rmse:.6f}")
print(f"   R²: {val_r2:.6f}")

print(f"\n✅ Boundary Constraints:")
print(f"   Training: {n_samples - train_boundary_violations}/{n_samples} satisfied")
print(f"   Validation: {len(X_val) - val_boundary_violations}/{len(X_val)} satisfied")

if val_rmse < 0.05 and val_r2 > 0.9:
    print(f"\n🎉 EXCELLENT: Model performance is good!")
    print(f"   Validation RMSE < 0.05 and R² > 0.9")
elif val_rmse < 0.10 and val_r2 > 0.5:
    print(f"\n⚠️  WARNING: Model performance needs improvement")
    print(f"   Validation RMSE = {val_rmse:.4f}, R² = {val_r2:.4f}")
else:
    print(f"\n❌ PROBLEM: Model performance is poor")
    print(f"   Validation RMSE = {val_rmse:.4f}, R² = {val_r2:.4f}")
    print(f"   Check scaling and denormalization!")

print("\n" + "="*80)
