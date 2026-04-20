#!/usr/bin/env python3
"""
Final PINN Training Script
- Weighted data loss at s=0
- Fine-tuning phase for s=0 accuracy
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf
import numpy as np
import json
from sklearn.metrics import mean_squared_error

from models.pinn_monotonic import MonotonicPINN
from models.pinn_physics_normalized import compute_total_loss
from training_pinn.config_pinn_final import (
    DATA_CONFIG, MODEL_CONFIG, PHASE1_CONFIG, PHASE2_CONFIG, 
    PHASE3_CONFIG, PHASE4_CONFIG, MONITORING_CONFIG, 
    RESULTS_DIR, CHECKPOINT_DIR, VIZ_DIR
)
from training_pinn.train_utils_pinn import DataLoader as PINNDataLoader, TrainingMonitor as PINNTrainingMonitor

# Set random seed
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

@tf.function
def train_step(model, batch, lambda_data, lambda_mono, lambda_bound, lambda_physics, s0_weight):
    """Training step with s=0 weighting"""
    with tf.GradientTape() as tape:
        theta_pred_norm = model({
            'soil_props': batch['soil_props'],
            'suction': batch['suction_grid']
        }, training=True)
        
        losses = compute_total_loss(
            theta_pred_norm=theta_pred_norm,
            theta_obs_norm=batch['swcc_curve'],
            suction=batch['suction_grid'],
            soil_props=batch['soil_props'],
            lambda_data=lambda_data,
            lambda_mono=lambda_mono,
            lambda_bound=lambda_bound,
            lambda_physics=lambda_physics,
            s0_weight=s0_weight
        )
    
    gradients = tape.gradient(losses['total'], model.trainable_variables)
    gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in gradients]
    
    return losses, gradients

@tf.function
def validate_step(model, batch, lambda_data, lambda_mono, lambda_bound, lambda_physics, s0_weight):
    """Validation step"""
    theta_pred_norm = model({
        'soil_props': batch['soil_props'],
        'suction': batch['suction_grid']
    }, training=False)
    
    losses = compute_total_loss(
        theta_pred_norm=theta_pred_norm,
        theta_obs_norm=batch['swcc_curve'],
        suction=batch['suction_grid'],
        soil_props=batch['soil_props'],
        lambda_data=lambda_data,
        lambda_mono=lambda_mono,
        lambda_bound=lambda_bound,
        lambda_physics=lambda_physics,
        s0_weight=s0_weight
    )
    
    return losses

def compute_rmse_normalized(y_pred_norm, y_true_norm):
    """Compute RMSE in normalized space"""
    return np.sqrt(mean_squared_error(y_true_norm.flatten(), y_pred_norm.flatten()))

def train_phase(model, train_dataset, val_dataset, optimizer, monitor, phase_config, 
                phase_name, start_epoch, end_epoch, lambda_data, lambda_mono, 
                lambda_bound, lambda_physics, s0_weight):
    """Train for one phase"""
    print(f"\n{'='*80}")
    print(f"Phase: {phase_name}")
    print(f"Epochs: {start_epoch} to {end_epoch}")
    print(f"Learning rate: {phase_config['learning_rate']:.2e}")
    print(f"Loss weights: λ_data={lambda_data}, λ_mono={lambda_mono}, "
          f"λ_bound={lambda_bound}, λ_physics={lambda_physics}")
    print(f"s=0 weight: {s0_weight:.1f}x")
    print(f"{'='*80}\n")
    
    best_val_rmse = float('inf')
    patience_counter = 0
    min_delta = MONITORING_CONFIG.get('min_delta', 1e-6)
    
    for epoch in range(start_epoch, end_epoch + 1):
        # Training
        epoch_train_losses = {
            'total': [], 'data': [], 'monotonicity': [], 'boundary': [], 'physics': []
        }
        
        for batch in train_dataset:
            losses, gradients = train_step(
                model, batch, lambda_data, lambda_mono, lambda_bound, lambda_physics, s0_weight
            )
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            for key, value in losses.items():
                if not (tf.math.is_nan(value) or tf.math.is_inf(value)):
                    epoch_train_losses[key].append(value.numpy())
        
        # Validation
        epoch_val_losses = {
            'total': [], 'data': [], 'monotonicity': [], 'boundary': [], 'physics': []
        }
        val_predictions = []
        val_observations = []
        
        for batch in val_dataset:
            losses = validate_step(
                model, batch, lambda_data, lambda_mono, lambda_bound, lambda_physics, s0_weight
            )
            for key, value in losses.items():
                if not (tf.math.is_nan(value) or tf.math.is_inf(value)):
                    epoch_val_losses[key].append(value.numpy())
            
            theta_pred_norm = model({
                'soil_props': batch['soil_props'],
                'suction': batch['suction_grid']
            }, training=False)
            val_predictions.append(theta_pred_norm.numpy())
            val_observations.append(batch['swcc_curve'].numpy())
        
        # Aggregate metrics
        train_metrics = {
            key: np.mean(vals) if vals else np.nan 
            for key, vals in epoch_train_losses.items()
        }
        val_metrics = {
            key: np.mean(vals) if vals else np.nan 
            for key, vals in epoch_val_losses.items()
        }
        
        # Compute validation RMSE
        val_pred_flat = np.vstack(val_predictions)
        val_obs_flat = np.vstack(val_observations)
        val_rmse = compute_rmse_normalized(val_pred_flat, val_obs_flat)
        
        # Log metrics
        monitor.log_metrics(epoch, train_metrics, val_metrics)
        
        # Print progress
        if epoch % MONITORING_CONFIG['log_freq'] == 0 or epoch == start_epoch:
            print(f"Epoch {epoch:4d} | Train Loss: {train_metrics['total']:.6f} | "
                  f"Val Loss: {val_metrics['total']:.6f} | Val RMSE: {val_rmse:.6f} | "
                  f"Data: {train_metrics['data']:.6f} | Mono: {train_metrics['monotonicity']:.6e}")
        
        # Save best model
        if val_rmse < (best_val_rmse - min_delta):
            improvement = best_val_rmse - val_rmse
            best_val_rmse = val_rmse
            patience_counter = 0
            best_model_path = CHECKPOINT_DIR / "pinn_best_model_final.keras"
            model.save(str(best_model_path))
            if epoch % MONITORING_CONFIG['log_freq'] == 0:
                print(f"  ✓ New best model (RMSE improvement: {improvement:.6f})")
        else:
            patience_counter += 1
        
        # Save checkpoint
        if epoch % MONITORING_CONFIG['checkpoint_freq'] == 0:
            checkpoint_path = CHECKPOINT_DIR / f"pinn_checkpoint_epoch_{epoch:04d}.keras"
            model.save(str(checkpoint_path))
        
        # Early stopping
        if patience_counter >= MONITORING_CONFIG['early_stopping_patience']:
            print(f"\n⚠ Early stopping at epoch {epoch} "
                  f"(no improvement for {MONITORING_CONFIG['early_stopping_patience']} epochs)")
            print(f"   Best validation RMSE: {best_val_rmse:.6f}")
            break

def main():
    """Main training function"""
    # GPU setup
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ Using {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU available, training on CPU.")
    
    print("="*80)
    print("Final PINN Training (with s=0 weighting)")
    print("="*80)
    print("\nKey Features:")
    print("  ✓ Weighted data loss at s=0 (saturated end)")
    print("  ✓ Fine-tuning phase for s=0 accuracy")
    print("  ✓ Structural monotonicity")
    print("  ✓ Early stopping on validation RMSE")
    print("="*80)
    
    # Load data
    data_loader = PINNDataLoader(DATA_CONFIG)
    metadata = data_loader.load_metadata()
    
    train_data = data_loader.load_train_data()
    val_data = data_loader.load_val_data()
    
    suction_grid = np.load(DATA_CONFIG['suction_grid_file'])
    
    # Create datasets
    def create_dataset(X, y, suction_grid, batch_size=32, shuffle=True):
        """Create TensorFlow dataset"""
        n_samples = len(X)
        suction_expanded = np.tile(suction_grid, (n_samples, 1))
        
        theta_s = X['theta_s'].values
        theta_r = X['theta_r'].values
        feature_cols = metadata['feature_cols']
        X_features = X[feature_cols].values
        
        dataset = tf.data.Dataset.from_tensor_slices({
            'soil_props': X_features.astype(np.float32),
            'suction_grid': suction_expanded.astype(np.float32),
            'swcc_curve': y.astype(np.float32),
            'theta_s': theta_s.astype(np.float32),
            'theta_r': theta_r.astype(np.float32)
        })
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(1000, n_samples))
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    train_dataset = create_dataset(
        train_data['X'], train_data['y'], suction_grid,
        batch_size=DATA_CONFIG['batch_size'], shuffle=True
    )
    val_dataset = create_dataset(
        val_data['X'], val_data['y'], suction_grid,
        batch_size=DATA_CONFIG['batch_size'], shuffle=False
    )
    
    print(f"\nLoading data...")
    print(f"  Train: {metadata['n_train']} samples")
    print(f"  Val: {metadata['n_val']} samples")
    print(f"  Features: {metadata['n_features']}")
    print(f"  SWCC points: {metadata['n_swcc_points']}")
    print(f"  Normalized: {metadata.get('normalized', False)}")
    
    # Try to load best model from previous training
    previous_best = Path("results_pinn_fixed/checkpoints/pinn_best_model_fixed.keras")
    model = MonotonicPINN(
        soil_prop_dim=metadata['n_features'],
        suction_points=metadata['n_swcc_points'],
        physics_units=MODEL_CONFIG['physics_units'],
        hidden_dims=MODEL_CONFIG['hidden_dims']
    )
    
    # Build model
    dummy_soil = tf.random.normal([DATA_CONFIG['batch_size'], metadata['n_features']])
    dummy_suction = tf.random.normal([DATA_CONFIG['batch_size'], metadata['n_swcc_points']])
    _ = model({'soil_props': dummy_soil, 'suction': dummy_suction})
    
    if previous_best.exists():
        print(f"\nLoading previous best model: {previous_best}")
        try:
            from models.pinn import PhysicsEncodingLayer
            saved_model = tf.keras.models.load_model(
                str(previous_best),
                custom_objects={'MonotonicPINN': MonotonicPINN, 'PhysicsEncodingLayer': PhysicsEncodingLayer},
                compile=False
            )
            model.set_weights(saved_model.get_weights())
            print("  ✓ Previous model loaded (starting from Phase 4)")
        except Exception as e:
            print(f"  ⚠ Could not load previous model: {e}")
            print("  Starting from scratch")
    else:
        print("\nNo previous model found, starting from scratch")
    
    print("✓ Model initialized")
    print(f"  Total parameters: {model.count_params():,}")
    print(f"  Architecture: Structural monotonicity (cumulative sum)")
    
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=PHASE4_CONFIG['learning_rate'])
    
    # Training monitor
    monitor = PINNTrainingMonitor(RESULTS_DIR, MONITORING_CONFIG)
    
    print("\n" + "="*80)
    print("Starting Fine-Tuning (Phase 4: s=0 Accuracy)")
    print("="*80)
    
    # Phase 4: Fine-tuning for s=0 accuracy
    train_phase(
        model, train_dataset, val_dataset, optimizer, monitor,
        PHASE4_CONFIG, "Phase 4: Fine-tuning (s=0 Accuracy)",
        1, PHASE4_CONFIG['epochs'],
        PHASE4_CONFIG['lambda_data'],
        PHASE4_CONFIG['lambda_mono'],
        PHASE4_CONFIG['lambda_bound'],
        PHASE4_CONFIG['lambda_physics'],
        PHASE4_CONFIG['s0_weight']
    )
    
    # Save final model
    final_model_path = CHECKPOINT_DIR / "pinn_final_model_final.keras"
    model.save(str(final_model_path))
    print(f"\n✓ Saved final model: {final_model_path}")
    
    # Save training history
    monitor.save_history()
    monitor.plot_training_curves()
    
    print("\n" + "="*80)
    print("Fine-Tuning Complete!")
    print("="*80)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Next: Re-evaluate with improved s=0 accuracy")

if __name__ == "__main__":
    main()
