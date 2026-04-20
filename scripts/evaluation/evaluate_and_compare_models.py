#!/usr/bin/env python3
"""
Evaluate and Compare: Best Model (Before GAN) vs Final Model (With GAN)
- Best model: pinn_best_model_fixed.keras (trained on real data only, normalized)
- Final model: pinn_final_model_fixed.keras (trained on real + 3,000 synthetic)
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
from training_pinn.config_pinn_fixed import DATA_CONFIG

plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 13

print("="*80)
print("Model Comparison: Best (Real-Only) vs Final (GAN-Augmented)")
print("="*80)

# Load test data (same for both)
X_test = pd.read_csv(DATA_CONFIG['test_file'])
y_test_original = np.load(DATA_CONFIG['y_test_original_file'])
suction_grid = np.load(DATA_CONFIG['suction_grid_file'])
metadata = json.load(open(DATA_CONFIG['metadata_file']))

theta_s_test = X_test['theta_s'].values
theta_r_test = X_test['theta_r'].values
feature_cols = metadata['feature_cols']

def evaluate_model_predictions(model_path):
    """Quick prediction function for plotting"""
    try:
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
            str(model_path),
            custom_objects={'MonotonicPINN': MonotonicPINN, 'PhysicsEncodingLayer': PhysicsEncodingLayer},
            compile=False
        )
        model.set_weights(saved_model.get_weights())
        
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
        
        return y_pred_physical
    except Exception as e:
        print(f"  Error in predictions: {e}")
        return None

def load_and_evaluate_model(model_path, model_name):
    """Load model and evaluate"""
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"Model: {model_path}")
    print(f"{'='*80}")
    
    if not model_path.exists():
        print(f"  ⚠ Model not found: {model_path}")
        return None
    
    # Load model
    model = MonotonicPINN(
        soil_prop_dim=metadata['n_features'],
        suction_points=metadata['n_swcc_points'],
        physics_units=128,
        hidden_dims=[128, 256, 128, 64]
    )
    
    # Build
    dummy_soil = tf.random.normal([1, metadata['n_features']])
    dummy_suction = tf.random.normal([1, metadata['n_swcc_points']])
    _ = model({'soil_props': dummy_soil, 'suction': dummy_suction})
    
    # Load weights
    try:
        saved_model = tf.keras.models.load_model(
            str(model_path),
            custom_objects={'MonotonicPINN': MonotonicPINN, 'PhysicsEncodingLayer': PhysicsEncodingLayer},
            compile=False
        )
        model.set_weights(saved_model.get_weights())
        print("  ✓ Model loaded")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None
    
    # Predict
    print("  Making predictions...")
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
    
    # Compute metrics
    y_true_flat = y_test_original.flatten()
    y_pred_flat = y_pred_physical.flatten()
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    
    rmse = np.sqrt(mean_squared_error(y_true_flat[mask], y_pred_flat[mask]))
    mae = mean_absolute_error(y_true_flat[mask], y_pred_flat[mask])
    r2 = r2_score(y_true_flat[mask], y_pred_flat[mask])
    
    # Per-sample
    sample_rmse = []
    for i in range(len(X_test)):
        y_t = y_test_original[i]
        y_p = y_pred_physical[i]
        mask_i = ~(np.isnan(y_t) | np.isnan(y_p))
        if mask_i.sum() > 0:
            sample_rmse.append(np.sqrt(mean_squared_error(y_t[mask_i], y_p[mask_i])))
    sample_rmse = np.array(sample_rmse)
    
    # Dry-end
    dry_end_threshold = 1e4
    dry_end_mask = suction_grid > dry_end_threshold
    dry_end_indices = np.where(dry_end_mask)[0]
    if len(dry_end_indices) > 0:
        y_true_dry = y_test_original[:, dry_end_indices]
        y_pred_dry = y_pred_physical[:, dry_end_indices]
        mask_dry = ~(np.isnan(y_true_dry) | np.isnan(y_pred_dry))
        rmse_dry = np.sqrt(mean_squared_error(y_true_dry[mask_dry], y_pred_dry[mask_dry]))
    else:
        rmse_dry = None
    
    # Monotonicity
    monotonic_count = sum(1 for i in range(len(y_pred_physical)) 
                         if np.all(np.diff(y_pred_physical[i]) <= 0))
    monotonicity_rate = monotonic_count / len(y_pred_physical)
    
    # Boundary
    boundary_count = sum(1 for i in range(len(y_pred_physical))
                        if (y_pred_physical[i].min() >= theta_r_test[i] - 0.01 and
                            y_pred_physical[i].max() <= theta_s_test[i] + 0.01))
    boundary_rate = boundary_count / len(y_pred_physical)
    
    print(f"\n  Global RMSE: {rmse:.6f}")
    print(f"  Global MAE:  {mae:.6f}")
    print(f"  Global R²:   {r2:.6f}")
    print(f"  Median RMSE: {np.median(sample_rmse):.6f}")
    if rmse_dry:
        print(f"  Dry-end RMSE: {rmse_dry:.6f}")
    print(f"  Monotonicity: {monotonicity_rate*100:.2f}%")
    print(f"  Boundary: {boundary_rate*100:.2f}%")
    
    return {
        'model_name': model_name,
        'model_path': str(model_path),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'rmse_median': float(np.median(sample_rmse)),
        'rmse_dry': float(rmse_dry) if rmse_dry else None,
        'monotonicity': float(monotonicity_rate),
        'boundary': float(boundary_rate),
        'predictions': y_pred_physical
    }

# Evaluate both models
CHECKPOINT_DIR = Path("results_pinn_fixed/checkpoints")
best_model = load_and_evaluate_model(
    CHECKPOINT_DIR / "pinn_best_model_fixed.keras",
    "Best Model (Real-Only, Normalized)"
)

final_model = load_and_evaluate_model(
    CHECKPOINT_DIR / "pinn_final_model_fixed.keras",
    "Final Model (Real + 3,000 Synthetic)"
)

# Comparison
if best_model and final_model:
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n{'Metric':<25} {'Best (Real-Only)':<20} {'Final (GAN-Aug)':<20} {'Better':<10}")
    print("-" * 80)
    
    metrics = [
        ('Global RMSE', 'rmse', 'lower'),
        ('Global MAE', 'mae', 'lower'),
        ('Global R²', 'r2', 'higher'),
        ('Median RMSE', 'rmse_median', 'lower'),
        ('Dry-end RMSE', 'rmse_dry', 'lower'),
        ('Monotonicity %', 'monotonicity', 'higher'),
        ('Boundary %', 'boundary', 'higher'),
    ]
    
    for name, key, direction in metrics:
        best_val = best_model.get(key)
        final_val = final_model.get(key)
        if best_val is not None and final_val is not None:
            if direction == 'lower':
                better = "Best" if best_val < final_val else "Final"
            else:
                better = "Best" if best_val > final_val else "Final"
            
            if isinstance(best_val, float):
                print(f"{name:<25} {best_val:<20.6f} {final_val:<20.6f} {better:<10}")
            else:
                print(f"{name:<25} {str(best_val):<20} {str(final_val):<20} {better:<10}")
    
    # Save comparison
    comparison = {
        'best_model': best_model,
        'final_model': final_model,
        'test_samples': int(len(X_test))
    }
    # Remove predictions from saved data (too large)
    comparison['best_model'].pop('predictions', None)
    comparison['final_model'].pop('predictions', None)
    
    output_file = Path("results_pinn_fixed/model_comparison.json")
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n✓ Saved comparison: {output_file}")
    
    # Re-predict for plotting (since we removed from dict)
    print("\nGenerating comparison plot...")
    
    # Re-evaluate best model for predictions
    best_pred = evaluate_model_predictions(CHECKPOINT_DIR / "pinn_best_model_fixed.keras")
    final_pred = evaluate_model_predictions(CHECKPOINT_DIR / "pinn_final_model_fixed.keras")
    
    if best_pred is not None and final_pred is not None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        n_samples = min(6, len(X_test))
        indices = np.linspace(0, len(X_test)-1, n_samples, dtype=int)
        
        for idx, ax in enumerate(axes[:n_samples]):
            i = int(indices[idx])
            ax.semilogx(suction_grid, y_test_original[i], 'k-', linewidth=2.5, label='Observed', alpha=0.9)
            ax.semilogx(suction_grid, best_pred[i], 'b--', linewidth=2, label='Best (Real-Only)', alpha=0.8)
            ax.semilogx(suction_grid, final_pred[i], 'r:', linewidth=2, label='Final (GAN-Aug)', alpha=0.8)
            ax.set_xlabel('Suction (kPa)', fontsize=12)
            ax.set_ylabel('Water Content (θ)', fontsize=12)
            ax.set_title(f'Sample {i+1}', fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='best')
        
        for ax in axes[n_samples:]:
            ax.remove()
        
        plt.tight_layout()
        viz_dir = Path("results_pinn_fixed/visualizations")
        viz_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(viz_dir / 'model_comparison_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved comparison plot: {viz_dir / 'model_comparison_curves.png'}")

def evaluate_model_predictions(model_path):
    """Quick prediction function for plotting"""
    try:
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
            str(model_path),
            custom_objects={'MonotonicPINN': MonotonicPINN, 'PhysicsEncodingLayer': PhysicsEncodingLayer},
            compile=False
        )
        model.set_weights(saved_model.get_weights())
        
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
        
        return y_pred_physical
    except:
        return None

print("\n" + "="*80)
print("Comparison Complete!")
print("="*80)
