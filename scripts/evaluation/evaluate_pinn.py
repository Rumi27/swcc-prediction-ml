#!/usr/bin/env python3
"""
Evaluate PINN Model on Test Set
Compare with baseline models and analyze performance
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
from models.pinn_physics import monotonicity_loss, boundary_loss
from training_pinn.config_pinn import DATA_CONFIG, RESULTS_DIR


def load_trained_model(model_path=None):
    """Load trained PINN model"""
    if model_path is None:
        model_path = RESULTS_DIR / "checkpoints" / "pinn_best_model.keras"
    
    print(f"Loading model from: {model_path}")
    
    # Load metadata first
    metadata = json.load(open(DATA_CONFIG['metadata_file']))
    
    # Create model architecture
    model = PINN(
        soil_prop_dim=metadata['n_features'],
        suction_points=metadata['n_swcc_points'],
        hidden_dims=[128, 256, 128, 64],
        physics_units=128
    )
    
    # Build model (dummy forward pass)
    dummy_soil = tf.random.normal([1, metadata['n_features']])
    dummy_suction = tf.random.normal([1, metadata['n_swcc_points']])
    _ = model({'soil_props': dummy_soil, 'suction': dummy_suction})
    
    # Try loading weights from .keras file
    try:
        # Load the saved model to extract weights
        saved_model = tf.keras.models.load_model(
            str(model_path),
            custom_objects={
                'PINN': PINN,
                'PhysicsEncodingLayer': PhysicsEncodingLayer
            },
            compile=False
        )
        # Copy weights
        model.set_weights(saved_model.get_weights())
        print("✓ Model loaded successfully (from .keras file)")
        return model
    except Exception as e:
        print(f"  Error loading .keras: {e}")
        print("  Trying to load from checkpoint weights...")
        
        # Try loading from checkpoint directory
        checkpoint_dir = RESULTS_DIR / "checkpoints"
        checkpoint_files = sorted(checkpoint_dir.glob("pinn_checkpoint_epoch_*.keras"))
        
        if checkpoint_files:
            latest_checkpoint = checkpoint_files[-1]
            print(f"  Trying: {latest_checkpoint}")
            try:
                saved_model = tf.keras.models.load_model(
                    str(latest_checkpoint),
                    custom_objects={
                        'PINN': PINN,
                        'PhysicsEncodingLayer': PhysicsEncodingLayer
                    },
                    compile=False
                )
                model.set_weights(saved_model.get_weights())
                print(f"✓ Model loaded from checkpoint: {latest_checkpoint.name}")
                return model
            except Exception as e2:
                print(f"  Error: {e2}")
        
        print("⚠ Could not load trained weights. Using newly initialized model.")
        print("  This will result in poor performance!")
        return model


def load_test_data():
    """Load test data"""
    print("\nLoading test data...")
    X_test = pd.read_csv(DATA_CONFIG['test_file'])
    y_test = np.load(DATA_CONFIG['y_test_file'])
    suction_grid = np.load(DATA_CONFIG['suction_grid_file'])
    
    # Remove NaN rows
    mask = ~X_test.isna().any(axis=1) & ~np.isnan(y_test).any(axis=1)
    X_test = X_test[mask].reset_index(drop=True)
    y_test = y_test[mask]
    
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {X_test.shape[1]}")
    print(f"  SWCC points: {y_test.shape[1]}")
    
    return X_test, y_test, suction_grid


def predict_swcc(model, X_test, suction_grid, batch_size=32):
    """Predict SWCC curves for test data"""
    print("\nPredicting SWCC curves...")
    
    predictions = []
    n_samples = len(X_test)
    
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        batch_X = X_test.iloc[i:batch_end].values.astype(np.float32)
        batch_suction = np.tile(suction_grid, (len(batch_X), 1)).astype(np.float32)
        
        inputs = {
            'soil_props': tf.constant(batch_X),
            'suction': tf.constant(batch_suction)
        }
        
        batch_pred = model(inputs, training=False)
        predictions.append(batch_pred.numpy())
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Progress: {batch_end}/{n_samples} samples")
    
    predictions = np.vstack(predictions)
    print(f"✓ Predictions complete: {predictions.shape}")
    
    return predictions


def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics"""
    # Flatten for overall metrics
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]
    
    # Overall metrics
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    # Per-sample metrics
    sample_rmse = []
    sample_mae = []
    sample_r2 = []
    
    for i in range(len(y_true)):
        y_t = y_true[i]
        y_p = y_pred[i]
        mask_i = ~(np.isnan(y_t) | np.isnan(y_p))
        if mask_i.sum() > 0:
            sample_rmse.append(np.sqrt(mean_squared_error(y_t[mask_i], y_p[mask_i])))
            sample_mae.append(mean_absolute_error(y_t[mask_i], y_p[mask_i]))
            sample_r2.append(r2_score(y_t[mask_i], y_p[mask_i]))
    
    return {
        'overall': {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        },
        'per_sample': {
            'rmse_mean': float(np.mean(sample_rmse)),
            'rmse_std': float(np.std(sample_rmse)),
            'mae_mean': float(np.mean(sample_mae)),
            'mae_std': float(np.std(sample_mae)),
            'r2_mean': float(np.mean(sample_r2)),
            'r2_std': float(np.std(sample_r2))
        }
    }


def check_physics_consistency(predictions, X_test, suction_grid):
    """Check physics constraints on predictions"""
    print("\nChecking physics consistency...")
    
    theta_s = X_test['theta_s'].values
    theta_r = X_test['theta_r'].values
    
    # Convert to tensors
    pred_tensor = tf.constant(predictions.astype(np.float32))
    suction_tensor = tf.constant(np.tile(suction_grid, (len(predictions), 1)).astype(np.float32))
    theta_s_tensor = tf.constant(theta_s.astype(np.float32))
    theta_r_tensor = tf.constant(theta_r.astype(np.float32))
    
    # Check monotonicity
    mono_loss = monotonicity_loss(pred_tensor, suction_tensor)
    
    # Check boundaries
    bound_loss = boundary_loss(pred_tensor, theta_s_tensor, theta_r_tensor)
    
    # Per-sample checks
    n_samples = len(predictions)
    mono_violations = 0
    bound_violations = 0
    
    for i in range(n_samples):
        # Monotonicity
        diff = predictions[i, :-1] - predictions[i, 1:]
        if np.any(diff < -1e-6):  # Allow small numerical errors
            mono_violations += 1
        
        # Boundaries
        if np.any(predictions[i] < theta_r[i] - 1e-6) or np.any(predictions[i] > theta_s[i] + 1e-6):
            bound_violations += 1
    
    results = {
        'monotonicity_loss': float(mono_loss.numpy()),
        'boundary_loss': float(bound_loss.numpy()),
        'monotonicity_rate': float(1.0 - mono_violations / n_samples),
        'boundary_rate': float(1.0 - bound_violations / n_samples),
        'monotonicity_violations': int(mono_violations),
        'boundary_violations': int(bound_violations)
    }
    
    print(f"  Monotonicity loss: {results['monotonicity_loss']:.6f}")
    print(f"  Boundary loss: {results['boundary_loss']:.6f}")
    print(f"  Monotonicity rate: {results['monotonicity_rate']*100:.2f}%")
    print(f"  Boundary rate: {results['boundary_rate']*100:.2f}%")
    
    return results


def compare_with_baseline(pinn_metrics):
    """Compare PINN results with baseline models"""
    print("\nComparing with baseline models...")
    
    baseline_file = Path("results_baseline/baseline_results.csv")
    if not baseline_file.exists():
        print("  ⚠ Baseline results not found")
        return None
    
    baseline_df = pd.read_csv(baseline_file)
    
    comparison = {
        'pinn': {
            'rmse': pinn_metrics['overall']['rmse'],
            'mae': pinn_metrics['overall']['mae'],
            'r2': pinn_metrics['overall']['r2']
        },
        'baseline_best': None
    }
    
    # Find best baseline model
    if 'RMSE' in baseline_df.columns:
        best_idx = baseline_df['RMSE'].idxmin()
        best_model = baseline_df.iloc[best_idx]
        comparison['baseline_best'] = {
            'model': best_model.get('Model', 'Unknown'),
            'rmse': float(best_model.get('RMSE', np.nan)),
            'mae': float(best_model.get('MAE', np.nan)),
            'r2': float(best_model.get('R²', np.nan))
        }
        
        print(f"  Best baseline: {comparison['baseline_best']['model']}")
        print(f"    RMSE: {comparison['baseline_best']['rmse']:.6f}")
        print(f"    MAE: {comparison['baseline_best']['mae']:.6f}")
        print(f"    R²: {comparison['baseline_best']['r2']:.6f}")
        
        print(f"\n  PINN vs Baseline:")
        rmse_improvement = (comparison['baseline_best']['rmse'] - comparison['pinn']['rmse']) / comparison['baseline_best']['rmse'] * 100
        r2_improvement = (comparison['pinn']['r2'] - comparison['baseline_best']['r2']) / abs(comparison['baseline_best']['r2']) * 100
        print(f"    RMSE improvement: {rmse_improvement:+.2f}%")
        print(f"    R² improvement: {r2_improvement:+.2f}%")
    
    return comparison


def visualize_predictions(y_true, y_pred, suction_grid, n_samples=12, output_dir=None):
    """Visualize predictions vs observations"""
    if output_dir is None:
        output_dir = RESULTS_DIR / "visualizations"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating visualizations...")
    
    # Sample random indices
    n_available = min(n_samples, len(y_true))
    indices = np.random.choice(len(y_true), n_available, replace=False)
    
    # Create figure
    n_cols = 4
    n_rows = (n_available + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, ax_idx in enumerate(indices):
        ax = axes[idx]
        
        # Observed
        ax.semilogx(suction_grid, y_true[ax_idx], 'b-', linewidth=2, 
                   label='Observed', alpha=0.7, marker='o', markersize=3)
        
        # Predicted
        ax.semilogx(suction_grid, y_pred[ax_idx], 'r--', linewidth=2, 
                   label='Predicted', alpha=0.7)
        
        # Compute sample metrics
        sample_rmse = np.sqrt(mean_squared_error(y_true[ax_idx], y_pred[ax_idx]))
        sample_r2 = r2_score(y_true[ax_idx], y_pred[ax_idx])
        
        ax.set_xlabel('Suction (kPa)', fontsize=10)
        ax.set_ylabel('Water Content (θ)', fontsize=10)
        ax.set_title(f'Sample {ax_idx+1}\nRMSE: {sample_rmse:.4f}, R²: {sample_r2:.4f}', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_available, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pinn_predictions_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_dir / 'pinn_predictions_test.png'}")


def plot_error_analysis(y_true, y_pred, output_dir=None):
    """Plot error analysis"""
    if output_dir is None:
        output_dir = RESULTS_DIR / "visualizations"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Flatten
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Remove NaN
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]
    
    errors = y_pred_clean - y_true_clean
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Scatter plot: Predicted vs Observed
    ax = axes[0, 0]
    ax.scatter(y_true_clean, y_pred_clean, alpha=0.5, s=10)
    ax.plot([y_true_clean.min(), y_true_clean.max()], 
           [y_true_clean.min(), y_true_clean.max()], 'r--', linewidth=2)
    ax.set_xlabel('Observed Water Content')
    ax.set_ylabel('Predicted Water Content')
    ax.set_title('Predicted vs Observed')
    ax.grid(True, alpha=0.3)
    
    # Error distribution
    ax = axes[0, 1]
    ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.grid(True, alpha=0.3)
    
    # Residuals vs Observed
    ax = axes[1, 0]
    ax.scatter(y_true_clean, errors, alpha=0.5, s=10)
    ax.axhline(0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Observed Water Content')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Observed')
    ax.grid(True, alpha=0.3)
    
    # Q-Q plot
    ax = axes[1, 1]
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Error Normality)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pinn_error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_dir / 'pinn_error_analysis.png'}")


def main():
    """Main evaluation function"""
    print("="*80)
    print("PINN Model Evaluation on Test Set")
    print("="*80)
    
    # Load model
    model = load_trained_model()
    
    # Load test data
    X_test, y_test, suction_grid = load_test_data()
    
    # Predict
    y_pred = predict_swcc(model, X_test, suction_grid)
    
    # Compute metrics
    print("\n" + "="*80)
    print("Performance Metrics")
    print("="*80)
    metrics = compute_metrics(y_test, y_pred)
    
    print("\nOverall Metrics:")
    print(f"  RMSE: {metrics['overall']['rmse']:.6f}")
    print(f"  MAE: {metrics['overall']['mae']:.6f}")
    print(f"  R²: {metrics['overall']['r2']:.6f}")
    
    print("\nPer-Sample Metrics (Mean ± Std):")
    print(f"  RMSE: {metrics['per_sample']['rmse_mean']:.6f} ± {metrics['per_sample']['rmse_std']:.6f}")
    print(f"  MAE: {metrics['per_sample']['mae_mean']:.6f} ± {metrics['per_sample']['mae_std']:.6f}")
    print(f"  R²: {metrics['per_sample']['r2_mean']:.6f} ± {metrics['per_sample']['r2_std']:.6f}")
    
    # Check physics
    print("\n" + "="*80)
    print("Physics Consistency")
    print("="*80)
    physics_results = check_physics_consistency(y_pred, X_test, suction_grid)
    
    # Compare with baseline
    print("\n" + "="*80)
    print("Comparison with Baseline Models")
    print("="*80)
    comparison = compare_with_baseline(metrics)
    
    # Visualizations
    print("\n" + "="*80)
    print("Creating Visualizations")
    print("="*80)
    visualize_predictions(y_test, y_pred, suction_grid, n_samples=12)
    plot_error_analysis(y_test, y_pred)
    
    # Save results
    results = {
        'metrics': metrics,
        'physics': physics_results,
        'comparison': comparison,
        'n_test_samples': int(len(X_test))
    }
    
    results_file = RESULTS_DIR / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved evaluation results: {results_file}")
    
    # Save summary
    summary_file = RESULTS_DIR / "evaluation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("PINN Model Evaluation Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Test Samples: {len(X_test)}\n\n")
        f.write("Overall Metrics:\n")
        f.write(f"  RMSE: {metrics['overall']['rmse']:.6f}\n")
        f.write(f"  MAE: {metrics['overall']['mae']:.6f}\n")
        f.write(f"  R²: {metrics['overall']['r2']:.6f}\n\n")
        f.write("Physics Consistency:\n")
        f.write(f"  Monotonicity Rate: {physics_results['monotonicity_rate']*100:.2f}%\n")
        f.write(f"  Boundary Rate: {physics_results['boundary_rate']*100:.2f}%\n")
        if comparison and comparison.get('baseline_best'):
            f.write("\nComparison with Baseline:\n")
            f.write(f"  Baseline RMSE: {comparison['baseline_best']['rmse']:.6f}\n")
            f.write(f"  PINN RMSE: {metrics['overall']['rmse']:.6f}\n")
            rmse_imp = (comparison['baseline_best']['rmse'] - metrics['overall']['rmse']) / comparison['baseline_best']['rmse'] * 100
            f.write(f"  Improvement: {rmse_imp:+.2f}%\n")
    
    print(f"✓ Saved summary: {summary_file}")
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"  - Metrics: evaluation_results.json")
    print(f"  - Visualizations: visualizations/")
    print(f"  - Summary: evaluation_summary.txt")


if __name__ == "__main__":
    main()
