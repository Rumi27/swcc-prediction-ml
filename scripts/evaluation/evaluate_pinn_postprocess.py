#!/usr/bin/env python3
"""
Evaluate PINN with Post-Processing
Enforces monotonicity on predictions
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

from models.pinn import PINN, PhysicsEncodingLayer
from training_pinn.config_pinn import DATA_CONFIG, RESULTS_DIR


def enforce_monotonicity(predictions):
    """Post-process predictions to enforce monotonicity"""
    predictions_processed = predictions.copy()
    
    for i in range(len(predictions_processed)):
        # Ensure decreasing: θ[i] >= θ[i+1]
        for j in range(1, len(predictions_processed[i])):
            if predictions_processed[i, j] > predictions_processed[i, j-1]:
                predictions_processed[i, j] = predictions_processed[i, j-1]
    
    return predictions_processed


def load_model_and_predict():
    """Load model and make predictions"""
    print("Loading model...")
    metadata = json.load(open(DATA_CONFIG['metadata_file']))
    
    model = PINN(
        soil_prop_dim=metadata['n_features'],
        suction_points=metadata['n_swcc_points']
    )
    
    # Build
    dummy_soil = tf.random.normal([1, metadata['n_features']])
    dummy_suction = tf.random.normal([1, metadata['n_swcc_points']])
    _ = model({'soil_props': dummy_soil, 'suction': dummy_suction})
    
    # Load weights from best checkpoint
    import glob
    checkpoints = sorted(glob.glob('results_pinn/checkpoints/pinn_checkpoint_epoch_*.keras'))
    if checkpoints:
        print(f"Loading from: {checkpoints[0]}")
        saved = tf.keras.models.load_model(
            checkpoints[0],
            custom_objects={'PINN': PINN, 'PhysicsEncodingLayer': PhysicsEncodingLayer},
            compile=False
        )
        model.set_weights(saved.get_weights())
        print("✓ Model loaded")
    
    # Load test data
    X_test = pd.read_csv(DATA_CONFIG['test_file']).dropna()
    y_test = np.load(DATA_CONFIG['y_test_file'])
    suction_grid = np.load(DATA_CONFIG['suction_grid_file'])
    
    mask = ~X_test.isna().any(axis=1) & ~np.isnan(y_test).any(axis=1)
    X_test = X_test[mask].reset_index(drop=True)
    y_test = y_test[mask]
    
    # Predict
    print("Making predictions...")
    predictions = []
    for i in range(0, len(X_test), 32):
        batch_X = X_test.iloc[i:i+32].values.astype(np.float32)
        batch_suction = np.tile(suction_grid, (len(batch_X), 1)).astype(np.float32)
        pred = model({'soil_props': tf.constant(batch_X), 
                     'suction': tf.constant(batch_suction)}, training=False)
        predictions.append(pred.numpy())
    
    predictions = np.vstack(predictions)
    
    return predictions, y_test, suction_grid, X_test


def main():
    """Main evaluation with post-processing"""
    print("="*80)
    print("PINN Evaluation with Post-Processing")
    print("="*80)
    
    # Load and predict
    y_pred_raw, y_test, suction_grid, X_test = load_model_and_predict()
    
    # Post-process for monotonicity
    print("\nPost-processing predictions (enforcing monotonicity)...")
    y_pred_processed = enforce_monotonicity(y_pred_raw)
    
    # Compute metrics for both
    print("\n" + "="*80)
    print("Performance Comparison")
    print("="*80)
    
    # Raw predictions
    y_true_flat = y_test.flatten()
    y_pred_raw_flat = y_pred_raw.flatten()
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_raw_flat))
    
    rmse_raw = np.sqrt(mean_squared_error(y_true_flat[mask], y_pred_raw_flat[mask]))
    mae_raw = mean_absolute_error(y_true_flat[mask], y_pred_raw_flat[mask])
    r2_raw = r2_score(y_true_flat[mask], y_pred_raw_flat[mask])
    
    # Processed predictions
    y_pred_proc_flat = y_pred_processed.flatten()
    mask_proc = ~(np.isnan(y_true_flat) | np.isnan(y_pred_proc_flat))
    
    rmse_proc = np.sqrt(mean_squared_error(y_true_flat[mask_proc], y_pred_proc_flat[mask_proc]))
    mae_proc = mean_absolute_error(y_true_flat[mask_proc], y_pred_proc_flat[mask_proc])
    r2_proc = r2_score(y_true_flat[mask_proc], y_pred_proc_flat[mask_proc])
    
    print("\nRaw Predictions:")
    print(f"  RMSE: {rmse_raw:.6f}")
    print(f"  MAE: {mae_raw:.6f}")
    print(f"  R²: {r2_raw:.6f}")
    
    print("\nPost-Processed Predictions:")
    print(f"  RMSE: {rmse_proc:.6f}")
    print(f"  MAE: {mae_proc:.6f}")
    print(f"  R²: {r2_proc:.6f}")
    
    # Check monotonicity
    print("\n" + "="*80)
    print("Physics Consistency")
    print("="*80)
    
    mono_violations_raw = 0
    mono_violations_proc = 0
    
    for i in range(len(y_test)):
        diff_raw = y_pred_raw[i, :-1] - y_pred_raw[i, 1:]
        diff_proc = y_pred_processed[i, :-1] - y_pred_processed[i, 1:]
        
        if np.any(diff_raw < -1e-6):
            mono_violations_raw += 1
        if np.any(diff_proc < -1e-6):
            mono_violations_proc += 1
    
    print(f"\nMonotonicity:")
    print(f"  Raw: {mono_violations_raw}/{len(y_test)} violations ({100*(1-mono_violations_raw/len(y_test)):.1f}% valid)")
    print(f"  Processed: {mono_violations_proc}/{len(y_test)} violations ({100*(1-mono_violations_proc/len(y_test)):.1f}% valid)")
    
    # Compare with baseline
    baseline_df = pd.read_csv('results_baseline/baseline_results.csv')
    best_baseline = baseline_df[baseline_df['dataset'] == 'test'].nsmallest(1, 'rmse').iloc[0]
    
    print("\n" + "="*80)
    print("Comparison with Baseline")
    print("="*80)
    print(f"Best Baseline ({best_baseline['model']}):")
    print(f"  RMSE: {best_baseline['rmse']:.6f}")
    print(f"  MAE: {best_baseline['mae']:.6f}")
    print(f"  R²: {best_baseline['r2']:.6f}")
    
    print(f"\nPINN (Post-Processed):")
    print(f"  RMSE: {rmse_proc:.6f} ({((best_baseline['rmse'] - rmse_proc) / best_baseline['rmse'] * 100):+.1f}%)")
    print(f"  MAE: {mae_proc:.6f} ({((best_baseline['mae'] - mae_proc) / best_baseline['mae'] * 100):+.1f}%)")
    print(f"  R²: {r2_proc:.6f} ({((r2_proc - best_baseline['r2']) / abs(best_baseline['r2']) * 100):+.1f}%)")
    
    # Save results
    results = {
        'raw': {'rmse': float(rmse_raw), 'mae': float(mae_raw), 'r2': float(r2_raw)},
        'processed': {'rmse': float(rmse_proc), 'mae': float(mae_proc), 'r2': float(r2_proc)},
        'monotonicity': {
            'raw_violations': int(mono_violations_raw),
            'processed_violations': int(mono_violations_proc),
            'raw_rate': float(1 - mono_violations_raw / len(y_test)),
            'processed_rate': float(1 - mono_violations_proc / len(y_test))
        },
        'baseline_comparison': {
            'baseline_rmse': float(best_baseline['rmse']),
            'pinn_rmse': float(rmse_proc),
            'improvement_pct': float((best_baseline['rmse'] - rmse_proc) / best_baseline['rmse'] * 100)
        }
    }
    
    with open(RESULTS_DIR / 'evaluation_postprocessed.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {RESULTS_DIR / 'evaluation_postprocessed.json'}")


if __name__ == "__main__":
    main()
