#!/usr/bin/env python3
"""
PINN Training Script
3-phase training: Pre-training, Physics-Informed, Fine-tuning
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from datetime import datetime

from models.pinn import PINN
from models.pinn_physics import compute_total_loss
from training_pinn.config_pinn import *
from training_pinn.train_utils_pinn import DataLoader, TrainingMonitor


def create_dataset(X, y, suction_grid, batch_size=32, shuffle=True):
    """Create TensorFlow dataset"""
    # Expand suction grid for each sample
    n_samples = len(X)
    suction_expanded = np.tile(suction_grid, (n_samples, 1))
    
    dataset = tf.data.Dataset.from_tensor_slices({
        'soil_props': X.values.astype(np.float32),
        'suction': suction_expanded.astype(np.float32),
        'theta_obs': y.astype(np.float32),
        'theta_s': X['theta_s'].values.astype(np.float32),
        'theta_r': X['theta_r'].values.astype(np.float32)
    })
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(1000, n_samples))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


@tf.function
def train_step(model, batch, lambda_data, lambda_mono, lambda_bound, lambda_physics):
    """Single training step"""
    with tf.GradientTape() as tape:
        # Forward pass
        inputs = {
            'soil_props': batch['soil_props'],
            'suction': batch['suction']
        }
        theta_pred = model(inputs, training=True)
        
        # Compute losses
        losses = compute_total_loss(
            theta_pred=theta_pred,
            theta_obs=batch['theta_obs'],
            suction=batch['suction'],
            soil_props=batch['soil_props'],
            theta_s=batch['theta_s'],
            theta_r=batch['theta_r'],
            lambda_data=lambda_data,
            lambda_mono=lambda_mono,
            lambda_bound=lambda_bound,
            lambda_physics=lambda_physics
        )
    
    # Compute gradients
    gradients = tape.gradient(losses['total'], model.trainable_variables)
    
    # Clip gradients
    gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in gradients]
    
    return losses, gradients


def train_phase(model, train_dataset, val_dataset, optimizer, monitor, monitoring_config, 
                phase_config, phase_name, start_epoch, end_epoch, 
                lambda_data, lambda_mono, lambda_bound, lambda_physics):
    """Train for one phase"""
    print(f"\n{'='*60}")
    print(f"Phase: {phase_name}")
    print(f"Epochs: {start_epoch} to {end_epoch}")
    print(f"Learning rate: {phase_config['learning_rate']:.2e}")
    print(f"Loss weights: λ_data={lambda_data}, λ_mono={lambda_mono}, "
          f"λ_bound={lambda_bound}, λ_physics={lambda_physics}")
    print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(start_epoch, end_epoch + 1):
        # Training
        epoch_losses = {'total': [], 'data': [], 'monotonicity': [], 'boundary': [], 'physics': []}
        
        for batch in train_dataset:
            losses, gradients = train_step(
                model, batch, lambda_data, lambda_mono, lambda_bound, lambda_physics
            )
            
            # Apply gradients
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Collect losses
            for key in epoch_losses:
                if not np.isnan(losses[key].numpy()) and not np.isinf(losses[key].numpy()):
                    epoch_losses[key].append(losses[key].numpy())
        
        # Validation
        val_losses = {'total': [], 'data': [], 'monotonicity': [], 'boundary': [], 'physics': []}
        
        for batch in val_dataset:
            inputs = {
                'soil_props': batch['soil_props'],
                'suction': batch['suction']
            }
            theta_pred = model(inputs, training=False)
            
            losses = compute_total_loss(
                theta_pred=theta_pred,
                theta_obs=batch['theta_obs'],
                suction=batch['suction'],
                soil_props=batch['soil_props'],
                theta_s=batch['theta_s'],
                theta_r=batch['theta_r'],
                lambda_data=lambda_data,
                lambda_mono=lambda_mono,
                lambda_bound=lambda_bound,
                lambda_physics=lambda_physics
            )
            
            for key in val_losses:
                if not np.isnan(losses[key].numpy()) and not np.isinf(losses[key].numpy()):
                    val_losses[key].append(losses[key].numpy())
        
        # Average losses (filter NaN/Inf)
        train_metrics = {}
        val_metrics = {}
        
        for key in epoch_losses:
            train_values = [v for v in epoch_losses[key] if not (np.isnan(v) or np.isinf(v))]
            train_metrics[key] = np.mean(train_values) if train_values else np.nan
        
        for key in val_losses:
            val_values = [v for v in val_losses[key] if not (np.isnan(v) or np.isinf(v))]
            val_metrics[key] = np.mean(val_values) if val_values else np.nan
        
        # Check for NaN in total loss
        if np.isnan(train_metrics['total']) or np.isnan(val_metrics['total']):
            print(f"\n⚠ WARNING: NaN losses detected at epoch {epoch}")
            print("  This may indicate numerical instability. Consider:")
            print("  1. Reducing learning rate")
            print("  2. Checking input data for NaN values")
            print("  3. Adjusting physics loss weights")
            # Don't break, but log the warning
        
        # Log metrics
        monitor.log_metrics(epoch, train_metrics, val_metrics)
        
        # Print progress
        if epoch % monitoring_config['log_freq'] == 0:
            print(f"Epoch {epoch:4d} | "
                  f"Train Loss: {train_metrics['total']:.4f} | "
                  f"Val Loss: {val_metrics['total']:.4f} | "
                  f"Data: {train_metrics['data']:.4f} | "
                  f"Mono: {train_metrics['monotonicity']:.4f} | "
                  f"Bound: {train_metrics['boundary']:.4f}")
        
        # Save checkpoint
        if epoch % monitoring_config['checkpoint_freq'] == 0:
            checkpoint_path = CHECKPOINT_DIR / f"pinn_checkpoint_epoch_{epoch:04d}.keras"
            model.save(str(checkpoint_path))
            print(f"✓ Saved checkpoint: {checkpoint_path}")
        
        # Early stopping
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            patience_counter = 0
            # Save best model
            best_model_path = CHECKPOINT_DIR / "pinn_best_model.keras"
            model.save(str(best_model_path))
        else:
            patience_counter += 1
            if patience_counter >= monitoring_config['early_stopping_patience']:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience_counter} epochs)")
                break
        
        # Generate plots
        if epoch % monitoring_config['plot_freq'] == 0 and epoch > 0:
            monitor.plot_predictions(model, val_dataset, epoch)


def main():
    """Main training function"""
    print("="*80)
    print("PINN Training for SWCC Prediction")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    loader = DataLoader(DATA_CONFIG)
    train_data = loader.load_train_data()
    val_data = loader.load_val_data()
    metadata = loader.load_metadata()
    
    print(f"  Train: {len(train_data['X'])} samples")
    print(f"  Val: {len(val_data['X'])} samples")
    print(f"  Features: {metadata['n_features']}")
    print(f"  SWCC points: {metadata['n_swcc_points']}")
    
    # Create datasets
    suction_grid = np.load(DATA_CONFIG['suction_grid_file'])
    train_dataset = create_dataset(
        train_data['X'], train_data['y'], suction_grid,
        batch_size=DATA_CONFIG['batch_size'], shuffle=True
    )
    val_dataset = create_dataset(
        val_data['X'], val_data['y'], suction_grid,
        batch_size=DATA_CONFIG['batch_size'], shuffle=False
    )
    
    # Initialize model
    print("\nInitializing PINN model...")
    model = PINN(
        soil_prop_dim=metadata['n_features'],
        suction_points=metadata['n_swcc_points'],
        hidden_dims=MODEL_CONFIG['hidden_dims'],
        physics_units=MODEL_CONFIG['physics_units']
    )
    
    # Build model (dummy forward pass)
    dummy_soil = tf.random.normal([1, metadata['n_features']])
    dummy_suction = tf.random.normal([1, metadata['n_swcc_points']])
    _ = model({'soil_props': dummy_soil, 'suction': dummy_suction})
    
    print("✓ Model initialized")
    print(f"  Total parameters: {model.count_params():,}")
    
    # Training monitor
    monitor = TrainingMonitor(RESULTS_DIR, MONITORING_CONFIG)
    
    # Training phases
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    
    # Phase 1: Pre-training
    optimizer1 = tf.keras.optimizers.Adam(learning_rate=PHASE1_CONFIG['learning_rate'])
    train_phase(
        model, train_dataset, val_dataset, optimizer1, monitor, MONITORING_CONFIG, PHASE1_CONFIG,
        "Pre-training",
        1, PHASE1_CONFIG['epochs'],
        PHASE1_CONFIG['lambda_data'],
        PHASE1_CONFIG['lambda_mono'],
        PHASE1_CONFIG['lambda_bound'],
        PHASE1_CONFIG['lambda_physics']
    )
    
    # Phase 2: Physics-Informed Training
    optimizer2 = tf.keras.optimizers.Adam(learning_rate=PHASE2_CONFIG['learning_rate'])
    train_phase(
        model, train_dataset, val_dataset, optimizer2, monitor, MONITORING_CONFIG, PHASE2_CONFIG,
        "Physics-Informed Training",
        PHASE1_CONFIG['epochs'] + 1,
        PHASE1_CONFIG['epochs'] + PHASE2_CONFIG['epochs'],
        PHASE2_CONFIG['lambda_data'],
        PHASE2_CONFIG['lambda_mono'],
        PHASE2_CONFIG['lambda_bound'],
        PHASE2_CONFIG['lambda_physics']
    )
    
    # Phase 3: Fine-tuning
    optimizer3 = tf.keras.optimizers.Adam(learning_rate=PHASE3_CONFIG['learning_rate'])
    train_phase(
        model, train_dataset, val_dataset, optimizer3, monitor, MONITORING_CONFIG, PHASE3_CONFIG,
        "Fine-tuning",
        PHASE1_CONFIG['epochs'] + PHASE2_CONFIG['epochs'] + 1,
        PHASE1_CONFIG['epochs'] + PHASE2_CONFIG['epochs'] + PHASE3_CONFIG['epochs'],
        PHASE3_CONFIG['lambda_data'],
        PHASE3_CONFIG['lambda_mono'],
        PHASE3_CONFIG['lambda_bound'],
        PHASE3_CONFIG['lambda_physics']
    )
    
    # Save final model
    final_model_path = CHECKPOINT_DIR / "pinn_final_model.keras"
    model.save(str(final_model_path))
    print(f"\n✓ Saved final model: {final_model_path}")
    
    # Save training history and plots
    monitor.save_history()
    monitor.plot_training_curves()
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Next: Evaluate model on test set")


if __name__ == "__main__":
    main()
