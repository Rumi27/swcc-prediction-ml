#!/usr/bin/env python3
"""
Main Training Script for WGAN-GP
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import glob

from models.wgan_gp import WGAN_GP
from training.config import *
from training.train_utils import DataLoader, TrainingMonitor


@tf.function
def train_discriminator_step(model, real_batch, noise, lambda_gp):
    """Train discriminator (critic) for one step."""
    with tf.GradientTape() as tape:
        theta_s = real_batch['theta_s']
        theta_r = real_batch['theta_r']
        # Generate fake curves with structural monotonicity and bounds
        fake_swcc = model.generator(
            [noise, real_batch['soil_props'], theta_s, theta_r],
            training=True,
        )

        # Numerics checks (fail fast during development; can relax later)
        tf.debugging.check_numerics(fake_swcc, "fake_swcc NaN/Inf")
        tf.debugging.check_numerics(real_batch['swcc_curve'], "real_swcc NaN/Inf")
        tf.debugging.check_numerics(real_batch['soil_props'], "soil_props NaN/Inf")

        # Compute discriminator loss
        d_loss, wasserstein_dist, gp = model.discriminator_loss(
            real_batch['swcc_curve'],
            fake_swcc,
            real_batch['soil_props']
        )

    # Compute gradients and apply global-norm clipping
    d_gradients = tape.gradient(d_loss, model.discriminator.trainable_variables)
    d_gradients, _ = tf.clip_by_global_norm(d_gradients, 10.0)

    return d_loss, wasserstein_dist, gp, d_gradients


@tf.function
def train_generator_step(model, real_batch, noise, lambda_mono, lambda_bound):
    """Train generator for one step."""
    with tf.GradientTape() as tape:
        theta_s = real_batch['theta_s']
        theta_r = real_batch['theta_r']

        # Generate fake curves with structural monotonicity and bounds
        fake_swcc = model.generator(
            [noise, real_batch['soil_props'], theta_s, theta_r],
            training=True,
        )

        tf.debugging.check_numerics(fake_swcc, "fake_swcc NaN/Inf")

        # Compute generator loss (physics now acts as light regularizer)
        g_loss, g_adv, mono_loss, bound_loss = model.generator_loss(
            fake_swcc,
            real_batch['soil_props'],
            theta_s,
            theta_r
        )

    # Compute gradients and apply global-norm clipping
    g_gradients = tape.gradient(g_loss, model.generator.trainable_variables)
    g_gradients, _ = tf.clip_by_global_norm(g_gradients, 10.0)

    return g_loss, g_adv, mono_loss, bound_loss, g_gradients


def train_phase(model, dataset, optimizer_g, optimizer_d, monitor, config, 
               phase_name, start_epoch, end_epoch, lambda_mono, lambda_bound):
    """Train for one phase"""
    print(f"\n{'='*60}")
    print(f"Phase: {phase_name}")
    print(f"Epochs: {start_epoch} to {end_epoch}")
    print(f"Physics weights: λ_mono={lambda_mono}, λ_bound={lambda_bound}")
    print(f"{'='*60}\n")
    
    # Update physics weights
    model.physics.lambda_mono = lambda_mono
    model.physics.lambda_bound = lambda_bound
    
    for epoch in range(start_epoch, end_epoch + 1):
        epoch_d_losses = []
        epoch_g_losses = []
        epoch_wd = []
        epoch_gp = []
        epoch_g_adv = []
        epoch_mono = []
        epoch_bound = []
        
        # Repeat dataset for this epoch (needed for multiple epochs)
        epoch_dataset = dataset.repeat(1)
        
        for batch in epoch_dataset:
            batch_size = tf.shape(batch['soil_props'])[0]
            
            # Update discriminator (5 times per generator update)
            for _ in range(config['d_updates_per_g']):
                noise = tf.random.normal([batch_size, model.noise_dim])
                
                d_loss, wd, gp, d_grads = train_discriminator_step(
                    model, batch, noise, config['lambda_gp']
                )
                
                # Apply gradients (already clipped in train_discriminator_step)
                optimizer_d.apply_gradients(
                    zip(d_grads, model.discriminator.trainable_variables)
                )
                
                # Check for NaN and Inf
                d_loss_val = d_loss.numpy()
                wd_val = wd.numpy()
                gp_val = gp.numpy()
                
                if not (np.isnan(d_loss_val) or np.isinf(d_loss_val) or 
                        np.isnan(wd_val) or np.isinf(wd_val) or
                        np.isnan(gp_val) or np.isinf(gp_val)):
                    epoch_d_losses.append(d_loss_val)
                    epoch_wd.append(wd_val)
                    epoch_gp.append(gp_val)
            
            # Update generator (once per 5 discriminator updates)
            noise = tf.random.normal([batch_size, model.noise_dim])
            
            g_loss, g_adv, mono_loss, bound_loss, g_grads = train_generator_step(
                model, batch, noise, lambda_mono, lambda_bound
            )
            
            # Apply gradients (already clipped in train_generator_step)
            optimizer_g.apply_gradients(
                zip(g_grads, model.generator.trainable_variables)
            )
            
            # Check for NaN and Inf
            g_loss_val = g_loss.numpy()
            g_adv_val = g_adv.numpy()
            mono_val = mono_loss.numpy()
            bound_val = bound_loss.numpy()
            
            if not (np.isnan(g_loss_val) or np.isinf(g_loss_val) or
                    np.isnan(g_adv_val) or np.isinf(g_adv_val) or
                    np.isnan(mono_val) or np.isinf(mono_val) or
                    np.isnan(bound_val) or np.isinf(bound_val)):
                epoch_g_losses.append(g_loss_val)
                epoch_g_adv.append(g_adv_val)
                epoch_mono.append(mono_val)
                epoch_bound.append(bound_val)
        
        # Log metrics (handle empty lists)
        metrics = {
            'd_loss': np.mean(epoch_d_losses) if epoch_d_losses else np.nan,
            'g_loss': np.mean(epoch_g_losses) if epoch_g_losses else np.nan,
            'wasserstein_dist': np.mean(epoch_wd) if epoch_wd else np.nan,
            'gradient_penalty': np.mean(epoch_gp) if epoch_gp else np.nan,
            'g_adversarial': np.mean(epoch_g_adv) if epoch_g_adv else np.nan,
            'mono_loss': np.mean(epoch_mono) if epoch_mono else np.nan,
            'bound_loss': np.mean(epoch_bound) if epoch_bound else np.nan
        }
        
        # Check for NaN losses
        has_nan = (np.isnan(metrics['d_loss']) or np.isnan(metrics['g_loss']))
        
        if has_nan:
            print(f"\n⚠ WARNING: NaN losses detected at epoch {epoch}.")
            
            # Try to save current checkpoint before stopping
            try:
                checkpoint_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch:04d}_nan.weights.h5"
                # Ensure model is built
                if not model.built:
                    dummy_noise = tf.random.normal([1, model.noise_dim])
                    dummy_props = tf.random.normal([1, model.soil_prop_dim])
                    dummy_swcc = tf.random.normal([1, model.swcc_points])
                    _ = model.generator([dummy_noise, dummy_props])
                    _ = model.discriminator([dummy_swcc, dummy_props])
                
                model.save_weights(str(checkpoint_path))
                print(f"✓ Saved checkpoint before stopping: {checkpoint_path}")
            except Exception as e:
                print(f"⚠ Could not save checkpoint: {e}")
            
            # Check if we have a recent valid checkpoint to continue from
            checkpoints = sorted(glob.glob(str(CHECKPOINT_DIR / "checkpoint_epoch_*.weights.h5")))
            if checkpoints and epoch > 50:
                last_checkpoint = checkpoints[-1]
                print(f"⚠ Attempting to recover from: {last_checkpoint}")
                try:
                    model.load_weights(last_checkpoint)
                    print("✓ Model weights reloaded. Continuing with reduced learning rate...")
                    # Reduce learning rates
                    optimizer_g.learning_rate.assign(optimizer_g.learning_rate * 0.5)
                    optimizer_d.learning_rate.assign(optimizer_d.learning_rate * 0.5)
                    print(f"  New learning rates: G={optimizer_g.learning_rate.numpy():.2e}, D={optimizer_d.learning_rate.numpy():.2e}")
                    continue  # Continue training instead of breaking
                except Exception as e:
                    print(f"⚠ Could not recover: {e}")
            
            print("Stopping training.")
            print("This may indicate:")
            print("  1. Learning rate too high")
            print("  2. Numerical instability in loss computation")
            print("  3. Invalid input data")
            break
        
        monitor.log_metrics(epoch, metrics)
        
        # Print progress
        if epoch % config['log_freq'] == 0:
            print(f"Epoch {epoch:4d} | "
                  f"D_loss: {metrics['d_loss']:.4f} | "
                  f"G_loss: {metrics['g_loss']:.4f} | "
                  f"WD: {metrics['wasserstein_dist']:.4f} | "
                  f"Mono: {metrics['mono_loss']:.4f} | "
                  f"Bound: {metrics['bound_loss']:.4f}")
        
        # Save checkpoint
        if epoch % config['checkpoint_freq'] == 0:
            try:
                checkpoint_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch:04d}.weights.h5"
                # Ensure model is built before saving
                if not model.built:
                    # Build model with dummy input
                    dummy_noise = tf.random.normal([1, model.noise_dim])
                    dummy_props = tf.random.normal([1, model.soil_prop_dim])
                    dummy_swcc = tf.random.normal([1, model.swcc_points])
                    dummy_theta_s = tf.constant([[0.45]], dtype=tf.float32)
                    dummy_theta_r = tf.constant([[0.05]], dtype=tf.float32)
                    _ = model.generator([dummy_noise, dummy_props, dummy_theta_s, dummy_theta_r], training=False)
                    _ = model.discriminator([dummy_swcc, dummy_props])
                
                model.save_weights(str(checkpoint_path))
                print(f"✓ Saved checkpoint: {checkpoint_path}")
            except Exception as e:
                print(f"⚠ Could not save checkpoint at epoch {epoch}: {e}")
        
        # Generate sample curves
        if epoch % config['sample_freq'] == 0 and epoch > 0:
            # Get sample batch
            sample_batch = next(iter(dataset.take(1)))
            monitor.plot_sample_generations(
                model,
                sample_batch['soil_props'][:4].numpy(),
                sample_batch['swcc_curve'][:4].numpy(),
                sample_batch['theta_s'][:4].numpy(),
                sample_batch['theta_r'][:4].numpy(),
                np.load(DATA_DIR / 'suction_grid.npy'),
                epoch
            )


def main():
    """Main training function"""
    print("="*80)
    print("WGAN-GP Training for SWCC Data Augmentation")
    print("="*80)
    
    # Load data
    loader = DataLoader({
        'train_file': DATA_CONFIG['train_file'],
        'swcc_file': DATA_CONFIG['swcc_file'],
        'suction_grid_file': DATA_CONFIG['suction_grid_file'],
        'feature_cols': FEATURE_COLS
    })
    
    data = loader.load_data()
    
    # Update config with actual feature dimension
    MODEL_CONFIG['soil_prop_dim'] = len(data['feature_cols'])
    
    # Create dataset
    dataset = loader.create_dataset(
        data['features'],
        data['swcc_curves'],
        data['theta_s'],
        data['theta_r'],
        batch_size=DATA_CONFIG['batch_size'],
        shuffle=True
    )
    
    # Initialize model
    print("\nInitializing WGAN-GP model...")
    model = WGAN_GP(
        noise_dim=MODEL_CONFIG['noise_dim'],
        soil_prop_dim=MODEL_CONFIG['soil_prop_dim'],
        swcc_points=MODEL_CONFIG['swcc_points'],
        lambda_gp=TRAINING_CONFIG['lambda_gp'],
        lambda_mono=TRAINING_CONFIG['lambda_mono_phase1'],
        lambda_bound=TRAINING_CONFIG['lambda_bound_phase1'],
        generator_hidden=MODEL_CONFIG['generator_hidden'],
        discriminator_hidden=MODEL_CONFIG['discriminator_hidden']
    )
    
    # Build models (dummy forward pass)
    dummy_noise = tf.random.normal([1, MODEL_CONFIG['noise_dim']])
    dummy_props = tf.random.normal([1, MODEL_CONFIG['soil_prop_dim']])
    dummy_swcc = tf.random.normal([1, MODEL_CONFIG['swcc_points']])
    # Dummy θ values (must satisfy θs >= θr)
    dummy_theta_s = tf.constant([[0.45]], dtype=tf.float32)
    dummy_theta_r = tf.constant([[0.05]], dtype=tf.float32)

    _ = model.generator([dummy_noise, dummy_props, dummy_theta_s, dummy_theta_r], training=False)
    _ = model.discriminator([dummy_swcc, dummy_props])
    
    print("✓ Model initialized")
    print(f"  Generator parameters: {model.generator.count_params():,}")
    print(f"  Discriminator parameters: {model.discriminator.count_params():,}")
    
    # Optimizers
    optimizer_g = tf.keras.optimizers.Adam(
        learning_rate=TRAINING_CONFIG['learning_rate_g'],
        beta_1=TRAINING_CONFIG['beta1'],
        beta_2=TRAINING_CONFIG['beta2']
    )
    
    optimizer_d = tf.keras.optimizers.Adam(
        learning_rate=TRAINING_CONFIG['learning_rate_d'],
        beta_1=TRAINING_CONFIG['beta1'],
        beta_2=TRAINING_CONFIG['beta2']
    )
    
    # Training monitor
    monitor = TrainingMonitor(RESULTS_DIR, TRAINING_CONFIG)
    
    # Training phases
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    
    # Phase 1: Pre-training
    train_phase(
        model, dataset, optimizer_g, optimizer_d, monitor, TRAINING_CONFIG,
        "Pre-training",
        1, TRAINING_CONFIG['phase1_epochs'],
        TRAINING_CONFIG['lambda_mono_phase1'],
        TRAINING_CONFIG['lambda_bound_phase1']
    )
    
    # Phase 2: Physics-informed training
    train_phase(
        model, dataset, optimizer_g, optimizer_d, monitor, TRAINING_CONFIG,
        "Physics-Informed Training",
        TRAINING_CONFIG['phase1_epochs'] + 1,
        TRAINING_CONFIG['phase1_epochs'] + TRAINING_CONFIG['phase2_epochs'],
        TRAINING_CONFIG['lambda_mono_phase2'],
        TRAINING_CONFIG['lambda_bound_phase2']
    )
    
    # Phase 3: Fine-tuning
    train_phase(
        model, dataset, optimizer_g, optimizer_d, monitor, TRAINING_CONFIG,
        "Fine-tuning",
        TRAINING_CONFIG['phase1_epochs'] + TRAINING_CONFIG['phase2_epochs'] + 1,
        TRAINING_CONFIG['total_epochs'],
        TRAINING_CONFIG['lambda_mono_phase3'],
        TRAINING_CONFIG['lambda_bound_phase3']
    )
    
    # Save final model
    final_checkpoint = CHECKPOINT_DIR / "final_model.weights.h5"
    try:
        # Ensure model is built before saving
        if not model.built:
            print("Building model before saving...")
            dummy_noise = tf.random.normal([1, model.noise_dim])
            dummy_props = tf.random.normal([1, model.soil_prop_dim])
            dummy_swcc = tf.random.normal([1, model.swcc_points])
            dummy_theta_s = tf.constant([[0.45]], dtype=tf.float32)
            dummy_theta_r = tf.constant([[0.05]], dtype=tf.float32)
            _ = model.generator([dummy_noise, dummy_props, dummy_theta_s, dummy_theta_r], training=False)
            _ = model.discriminator([dummy_swcc, dummy_props])
            print("✓ Model built")
        
        model.save_weights(str(final_checkpoint))
        print(f"\n✓ Saved final model: {final_checkpoint}")
    except Exception as e:
        print(f"\n⚠ Could not save final model as weights: {e}")
        # Try saving as Keras format instead
        try:
            final_checkpoint_keras = CHECKPOINT_DIR / "final_model.keras"
            model.save(str(final_checkpoint_keras))
            print(f"✓ Saved final model as Keras format: {final_checkpoint_keras}")
        except Exception as e2:
            print(f"⚠ Could not save model in Keras format: {e2}")
            # Try SavedModel format
            try:
                final_checkpoint_dir = CHECKPOINT_DIR / "final_model"
                model.export(str(final_checkpoint_dir))
                print(f"✓ Saved final model as SavedModel: {final_checkpoint_dir}")
            except Exception as e3:
                print(f"⚠ Could not save model in any format: {e3}")
                print("  Model weights are in memory but not saved to disk.")
    
    # Save training history and plots
    monitor.save_history()
    monitor.plot_training_curves()
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Next: Generate synthetic data using trained model")


if __name__ == "__main__":
    main()
