#!/usr/bin/env python3
"""
Optimized PINN Training Script
Uses optimized hyperparameters for better physics consistency
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf
import numpy as np
import json
from datetime import datetime

from models.pinn import PINN, PhysicsEncodingLayer
from models.pinn_physics import data_loss, monotonicity_loss, boundary_loss, arya_paris_physics_loss
from training_pinn.config_pinn_optimized import (
    DATA_CONFIG, MODEL_CONFIG, PHASE1_CONFIG, PHASE2_CONFIG, PHASE3_CONFIG,
    MONITORING_CONFIG, RESULTS_DIR, CHECKPOINT_DIR, VIZ_DIR
)
from training_pinn.train_utils_pinn import DataLoader as PINNDataLoader, TrainingMonitor as PINNTrainingMonitor

# Set random seed for reproducibility
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

@tf.function
def train_step(model, batch, lambda_data, lambda_mono, lambda_bound, lambda_physics):
    """Performs one training step for the PINN model."""
    with tf.GradientTape() as tape:
        theta_pred = model({
            'soil_props': batch['soil_props'], 
            'suction': batch['suction_grid']
        }, training=True)
        
        # Compute individual losses
        d_loss = data_loss(theta_pred, batch['swcc_curve'])
        m_loss = monotonicity_loss(theta_pred, batch['suction_grid'])
        b_loss = boundary_loss(theta_pred, batch['theta_s'], batch['theta_r'])
        p_loss = arya_paris_physics_loss(theta_pred, batch['suction_grid'], batch['soil_props'])
        
        # Total loss
        total_loss = (lambda_data * d_loss +
                     lambda_mono * m_loss +
                     lambda_bound * b_loss +
                     lambda_physics * p_loss)
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    
    # Clip gradients to prevent explosion
    gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in gradients]
    
    return {
        'total': total_loss,
        'data': d_loss,
        'monotonicity': m_loss,
        'boundary': b_loss,
        'physics': p_loss
    }, gradients

@tf.function
def validate_step(model, batch, lambda_data, lambda_mono, lambda_bound, lambda_physics):
    """Performs one validation step for the PINN model."""
    theta_pred = model({
        'soil_props': batch['soil_props'],
        'suction': batch['suction_grid']
    }, training=False)
    
    d_loss = data_loss(theta_pred, batch['swcc_curve'])
    m_loss = monotonicity_loss(theta_pred, batch['suction_grid'])
    b_loss = boundary_loss(theta_pred, batch['theta_s'], batch['theta_r'])
    p_loss = arya_paris_physics_loss(theta_pred, batch['suction_grid'], batch['soil_props'])
    
    total_loss = (lambda_data * d_loss +
                 lambda_mono * m_loss +
                 lambda_bound * b_loss +
                 lambda_physics * p_loss)
    
    return {
        'total': total_loss,
        'data': d_loss,
        'monotonicity': m_loss,
        'boundary': b_loss,
        'physics': p_loss
    }

def train_phase(model, train_dataset, val_dataset, optimizer, monitor, phase_config, phase_name, 
                start_epoch, end_epoch, lambda_data, lambda_mono, lambda_bound, lambda_physics):
    """Train for one phase."""
    print(f"\n{'='*80}")
    print(f"Phase: {phase_name}")
    print(f"Epochs: {start_epoch} to {end_epoch}")
    print(f"Learning rate: {phase_config['learning_rate']:.2e}")
    print(f"Loss weights: λ_data={lambda_data}, λ_mono={lambda_mono}, "
          f"λ_bound={lambda_bound}, λ_physics={lambda_physics}")
    print(f"{'='*80}\n")
    
    best_val_loss = float('inf')
    patience_counter = 0
    min_delta = MONITORING_CONFIG.get('min_delta', 1e-6)
    
    for epoch in range(start_epoch, end_epoch + 1):
        # Training
        epoch_train_losses = {
            'total': [], 'data': [], 'monotonicity': [], 'boundary': [], 'physics': []
        }
        
        for batch in train_dataset:
            losses, gradients = train_step(
                model, batch, lambda_data, lambda_mono, lambda_bound, lambda_physics
            )
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            for key, value in losses.items():
                if not (tf.math.is_nan(value) or tf.math.is_inf(value)):
                    epoch_train_losses[key].append(value.numpy())
        
        # Validation
        epoch_val_losses = {
            'total': [], 'data': [], 'monotonicity': [], 'boundary': [], 'physics': []
        }
        for batch in val_dataset:
            losses = validate_step(
                model, batch, lambda_data, lambda_mono, lambda_bound, lambda_physics
            )
            for key, value in losses.items():
                if not (tf.math.is_nan(value) or tf.math.is_inf(value)):
                    epoch_val_losses[key].append(value.numpy())
        
        # Aggregate metrics
        train_metrics = {
            key: np.mean(vals) if vals else np.nan 
            for key, vals in epoch_train_losses.items()
        }
        val_metrics = {
            key: np.mean(vals) if vals else np.nan 
            for key, vals in epoch_val_losses.items()
        }
        
        # Log metrics
        monitor.log_metrics(epoch, train_metrics, val_metrics)
        
        # Print progress
        if epoch % MONITORING_CONFIG['log_freq'] == 0 or epoch == start_epoch:
            print(f"Epoch {epoch:4d} | Train Loss: {train_metrics['total']:.6f} | "
                  f"Val Loss: {val_metrics['total']:.6f} | "
                  f"Data: {train_metrics['data']:.6f} | "
                  f"Mono: {train_metrics['monotonicity']:.6e} | "
                  f"Bound: {train_metrics['boundary']:.6e}")
        
        # Save best model (with minimum delta check)
        if val_metrics['total'] < (best_val_loss - min_delta):
            improvement = best_val_loss - val_metrics['total']
            best_val_loss = val_metrics['total']
            patience_counter = 0
            best_model_path = CHECKPOINT_DIR / "pinn_best_model_optimized.keras"
            model.save(str(best_model_path))
            if epoch % MONITORING_CONFIG['log_freq'] == 0:
                print(f"  ✓ New best model (improvement: {improvement:.6f})")
        else:
            patience_counter += 1
        
        # Save checkpoint
        if epoch % MONITORING_CONFIG['checkpoint_freq'] == 0:
            checkpoint_path = CHECKPOINT_DIR / f"pinn_checkpoint_epoch_{epoch:04d}.keras"
            model.save(str(checkpoint_path))
            if epoch % MONITORING_CONFIG['log_freq'] == 0:
                print(f"  ✓ Saved checkpoint: {checkpoint_path.name}")
        
        # Generate sample curves
        if epoch % MONITORING_CONFIG['plot_freq'] == 0 and epoch > start_epoch:
            sample_batch = next(iter(val_dataset.take(1)))
            monitor.plot_sample_predictions(
                model,
                sample_batch['soil_props'][:4].numpy(),
                sample_batch['suction_grid'][:4].numpy(),
                sample_batch['swcc_curve'][:4].numpy(),
                sample_batch['theta_s'][:4].numpy(),
                sample_batch['theta_r'][:4].numpy(),
                epoch
            )
        
        # Early stopping
        if patience_counter >= MONITORING_CONFIG['early_stopping_patience']:
            print(f"\n⚠ Early stopping at epoch {epoch} "
                  f"(no improvement for {MONITORING_CONFIG['early_stopping_patience']} epochs)")
            print(f"   Best validation loss: {best_val_loss:.6f}")
            break

def main():
    """Main training function"""
    # Ensure GPU is available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"✓ Using {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU available, training on CPU.")
    
    print("="*80)
    print("Optimized PINN Training for SWCC Prediction")
    print("="*80)
    print("\nKey Optimizations:")
    print("  - Increased monotonicity weights (Phase 2: 2.0, Phase 3: 5.0)")
    print("  - Increased boundary weights (Phase 2: 1.0, Phase 3: 2.0)")
    print("  - Longer training (900 total epochs)")
    print("  - Better early stopping (patience: 50)")
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
        
        # Extract theta_s and theta_r
        theta_s = X['theta_s'].values
        theta_r = X['theta_r'].values
        
        # Select feature columns
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
    
    train_dataset = create_dataset(train_data['X'], train_data['y'], suction_grid, 
                                   batch_size=DATA_CONFIG['batch_size'], shuffle=True)
    val_dataset = create_dataset(val_data['X'], val_data['y'], suction_grid,
                                 batch_size=DATA_CONFIG['batch_size'], shuffle=False)
    
    print("\nLoading data...")
    print(f"  Train: {metadata['n_train']} samples")
    print(f"  Val: {metadata['n_val']} samples")
    print(f"  Features: {metadata['n_features']}")
    print(f"  SWCC points: {metadata['n_swcc_points']}")
    
    # Initialize PINN model
    print("\nInitializing PINN model...")
    model = PINN(
        soil_prop_dim=metadata['n_features'],
        suction_points=metadata['n_swcc_points'],
        physics_units=MODEL_CONFIG['physics_units'],
        hidden_dims=MODEL_CONFIG['hidden_dims']
    )
    
    # Build model (dummy forward pass)
    dummy_soil_props = tf.random.normal([DATA_CONFIG['batch_size'], metadata['n_features']])
    dummy_suction = tf.random.normal([DATA_CONFIG['batch_size'], metadata['n_swcc_points']])
    _ = model({'soil_props': dummy_soil_props, 'suction': dummy_suction})
    
    print("✓ Model initialized")
    print(f"  Total parameters: {model.count_params():,}")
    
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=PHASE1_CONFIG['learning_rate'])
    
    # Training monitor
    monitor = PINNTrainingMonitor(RESULTS_DIR, MONITORING_CONFIG)
    
    # Ensure monitor has required methods
    if not hasattr(monitor, 'plot_sample_predictions'):
        # Add simple plot method if missing
        def plot_sample_predictions(model, soil_props, suction_grid, swcc_true, theta_s, theta_r, epoch):
            """Plot sample predictions"""
            try:
                import matplotlib.pyplot as plt
                inputs = {'soil_props': tf.constant(soil_props), 'suction': tf.constant(suction_grid)}
                theta_pred = model(inputs, training=False)
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                axes = axes.flatten()
                
                for i in range(min(4, len(soil_props))):
                    ax = axes[i]
                    ax.semilogx(suction_grid[i], swcc_true[i], 'b-', label='True', linewidth=2)
                    ax.semilogx(suction_grid[i], theta_pred[i].numpy(), 'r--', label='Pred', linewidth=2)
                    ax.set_xlabel('Suction (kPa)')
                    ax.set_ylabel('Water Content')
                    ax.set_title(f'Sample {i+1} - Epoch {epoch}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(VIZ_DIR / f'predictions_epoch_{epoch:04d}.png', dpi=150)
                plt.close()
            except Exception as e:
                print(f"  Warning: Could not plot predictions: {e}")
        
        monitor.plot_sample_predictions = plot_sample_predictions
    
    print("\n" + "="*80)
    print("Starting Optimized Training")
    print("="*80)
    
    # Phase 1: Pre-training (Data-driven)
    train_phase(
        model, train_dataset, val_dataset, optimizer, monitor,
        PHASE1_CONFIG, "Pre-training (Data-driven)",
        1, PHASE1_CONFIG['epochs'],
        PHASE1_CONFIG['lambda_data'],
        PHASE1_CONFIG['lambda_mono'],
        PHASE1_CONFIG['lambda_bound'],
        PHASE1_CONFIG['lambda_physics']
    )
    
    # Update learning rate for Phase 2
    optimizer.learning_rate.assign(PHASE2_CONFIG['learning_rate'])
    
    # Phase 2: Physics-Informed training
    train_phase(
        model, train_dataset, val_dataset, optimizer, monitor,
        PHASE2_CONFIG, "Physics-Informed Training",
        PHASE1_CONFIG['epochs'] + 1,
        PHASE1_CONFIG['epochs'] + PHASE2_CONFIG['epochs'],
        PHASE2_CONFIG['lambda_data'],
        PHASE2_CONFIG['lambda_mono'],
        PHASE2_CONFIG['lambda_bound'],
        PHASE2_CONFIG['lambda_physics']
    )
    
    # Update learning rate for Phase 3
    optimizer.learning_rate.assign(PHASE3_CONFIG['learning_rate'])
    
    # Phase 3: Fine-tuning (Strong Physics)
    train_phase(
        model, train_dataset, val_dataset, optimizer, monitor,
        PHASE3_CONFIG, "Fine-tuning (Strong Physics)",
        PHASE1_CONFIG['epochs'] + PHASE2_CONFIG['epochs'] + 1,
        PHASE1_CONFIG['epochs'] + PHASE2_CONFIG['epochs'] + PHASE3_CONFIG['epochs'],
        PHASE3_CONFIG['lambda_data'],
        PHASE3_CONFIG['lambda_mono'],
        PHASE3_CONFIG['lambda_bound'],
        PHASE3_CONFIG['lambda_physics']
    )
    
    # Save final model
    final_model_path = CHECKPOINT_DIR / "pinn_final_model_optimized.keras"
    model.save(str(final_model_path))
    print(f"\n✓ Saved final model: {final_model_path}")
    
    # Save training history and plots
    monitor.save_history()
    monitor.plot_training_curves()
    
    print("\n" + "="*80)
    print("Optimized Training Complete!")
    print("="*80)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Next: Evaluate optimized model")

if __name__ == "__main__":
    main()
