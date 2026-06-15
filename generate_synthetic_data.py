#!/usr/bin/env python3
"""
Generate Synthetic SWCC Data using Trained GAN
Phase 2: Data Augmentation
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
from datetime import datetime

from models.wgan_gp import WGAN_GP
from training.config import *
from training.train_utils import DataLoader


def load_trained_model(checkpoint_path=None):
    """
    Load trained WGAN-GP model
    
    Args:
        checkpoint_path: Path to checkpoint file. If None, uses latest checkpoint.
        
    Returns:
        Trained model
    """
    print("Loading trained model...")
    
    # Find latest checkpoint if not specified
    if checkpoint_path is None:
        # Try multiple checkpoint patterns
        checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint_epoch_*.weights.h5"))
        
        # Try final model in different formats
        if not checkpoints:
            # Try .weights.h5 format
            final_model_h5 = CHECKPOINT_DIR / "final_model.weights.h5"
            # Try .keras format
            final_model_keras = CHECKPOINT_DIR / "final_model.keras"
            # Try SavedModel format
            saved_model_dir = CHECKPOINT_DIR / "final_model"
            
            if final_model_h5.exists():
                checkpoint_path = final_model_h5
                model_format = 'weights'
            elif final_model_keras.exists():
                checkpoint_path = final_model_keras
                model_format = 'keras'
            elif saved_model_dir.exists() and (saved_model_dir / "saved_model.pb").exists():
                checkpoint_path = saved_model_dir
                model_format = 'savedmodel'
            else:
                raise FileNotFoundError(
                    f"No checkpoint found in {CHECKPOINT_DIR}\n"
                    f"Please train the model first:\n"
                    f"  python3 training/train_gan.py"
                )
        else:
            checkpoint_path = checkpoints[-1]
            model_format = 'weights'
    
    print(f"  Loading from: {checkpoint_path}")
    print(f"  Format: {model_format if 'model_format' in locals() else 'auto-detect'}")
    
    # Handle different model formats
    if str(checkpoint_path).endswith('.keras'):
        # Load Keras format (complete model)
        print("  Loading as Keras format...")
        # Need to provide custom objects for custom classes
        from models.wgan_gp import WGAN_GP
        from models.generator import Generator
        from models.discriminator import Discriminator
        from models.physics_constraints import PhysicsConstraints
        
        custom_objects = {
            'WGAN_GP': WGAN_GP,
            'Generator': Generator,
            'Discriminator': Discriminator,
            'PhysicsConstraints': PhysicsConstraints
        }
        
        model = tf.keras.models.load_model(
            str(checkpoint_path), 
            compile=False,
            custom_objects=custom_objects
        )
        print("✓ Model loaded successfully (Keras format)")
        
        # Load data for return
        loader = DataLoader({
            'train_file': DATA_CONFIG['train_file'],
            'swcc_file': DATA_CONFIG['swcc_file'],
            'suction_grid_file': DATA_CONFIG['suction_grid_file'],
            'feature_cols': FEATURE_COLS
        })
        data = loader.load_data()
        
        return model, data
    
    elif str(checkpoint_path).endswith('.weights.h5') or str(checkpoint_path).endswith('.h5'):
        # Load weights format (need to build model first)
        print("  Loading as weights format...")
        
        # Load data to get dimensions
        loader = DataLoader({
            'train_file': DATA_CONFIG['train_file'],
            'swcc_file': DATA_CONFIG['swcc_file'],
            'suction_grid_file': DATA_CONFIG['suction_grid_file'],
            'feature_cols': FEATURE_COLS
        })
        
        data = loader.load_data()
        num_features = len(data['feature_cols'])
        
        # Initialize model with same architecture
        model = WGAN_GP(
            noise_dim=MODEL_CONFIG['noise_dim'],
            soil_prop_dim=num_features,
            swcc_points=MODEL_CONFIG['swcc_points'],
            lambda_gp=TRAINING_CONFIG['lambda_gp'],
            lambda_mono=TRAINING_CONFIG['lambda_mono_phase1'],
            lambda_bound=TRAINING_CONFIG['lambda_bound_phase1'],
            generator_hidden=MODEL_CONFIG['generator_hidden'],
            discriminator_hidden=MODEL_CONFIG['discriminator_hidden']
        )
        
        # Build model (dummy forward pass)
        dummy_noise = tf.random.normal([1, MODEL_CONFIG['noise_dim']])
        dummy_props = tf.random.normal([1, num_features])
        dummy_swcc = tf.random.normal([1, MODEL_CONFIG['swcc_points']])
        
        _ = model.generator([dummy_noise, dummy_props])
        _ = model.discriminator([dummy_swcc, dummy_props])
        
        # Load weights
        model.load_weights(str(checkpoint_path))
        print("✓ Model loaded successfully (weights format)")
        
        return model, data
    
    else:
        # SavedModel format (not implemented yet)
        raise NotImplementedError(
            f"SavedModel format loading not yet implemented.\n"
            f"Please use .keras or .weights.h5 format."
        )
    
    return model, data


def validate_generated_curves(curves, theta_s, theta_r, suction_grid):
    """
    Validate generated curves against physics constraints
    
    Args:
        curves: Generated SWCC curves [N, n_points]
        theta_s: Saturated water content [N]
        theta_r: Residual water content [N]
        suction_grid: Suction values [n_points]
        
    Returns:
        Dictionary with validation results
    """
    print("\nValidating generated curves...")
    
    n_samples, n_points = curves.shape
    
    # Check boundaries
    theta_s_2d = theta_s.reshape(-1, 1) if len(theta_s.shape) == 1 else theta_s
    theta_r_2d = theta_r.reshape(-1, 1) if len(theta_r.shape) == 1 else theta_r
    
    within_bounds = np.all((curves >= theta_r_2d) & (curves <= theta_s_2d), axis=1)
    bound_rate = np.mean(within_bounds)
    
    # Check monotonicity (decreasing)
    diff = curves[:, :-1] - curves[:, 1:]
    is_monotonic = np.all(diff >= -1e-6, axis=1)  # Allow small numerical errors
    mono_rate = np.mean(is_monotonic)
    
    # Check for NaN/Inf
    has_nan = np.any(np.isnan(curves), axis=1)
    has_inf = np.any(np.isinf(curves), axis=1)
    nan_rate = np.mean(has_nan)
    inf_rate = np.mean(has_inf)
    
    # Statistics
    mean_curve = np.mean(curves, axis=0)
    std_curve = np.std(curves, axis=0)
    
    results = {
        'n_samples': n_samples,
        'n_points': n_points,
        'boundary_satisfaction_rate': float(bound_rate),
        'monotonicity_rate': float(mono_rate),
        'nan_rate': float(nan_rate),
        'inf_rate': float(inf_rate),
        'mean_curve': mean_curve.tolist(),
        'std_curve': std_curve.tolist(),
        'all_valid': bool(bound_rate > 0.95 and mono_rate > 0.95 and nan_rate == 0 and inf_rate == 0)
    }
    
    print(f"  Boundary satisfaction: {bound_rate*100:.2f}%")
    print(f"  Monotonicity: {mono_rate*100:.2f}%")
    print(f"  NaN rate: {nan_rate*100:.2f}%")
    print(f"  Inf rate: {inf_rate*100:.2f}%")
    
    return results


def generate_synthetic_data(model, data, num_samples_per_soil=20, seed=42):
    """
    Generate synthetic SWCC curves
    
    Args:
        model: Trained WGAN-GP model
        data: Data dictionary from DataLoader
        num_samples_per_soil: Number of synthetic curves per real sample
        seed: Random seed
        
    Returns:
        Dictionary with synthetic data
    """
    print(f"\nGenerating {num_samples_per_soil} synthetic curves per real sample...")
    
    features = data['features']
    theta_s = data['theta_s']
    theta_r = data['theta_r']
    suction_grid = data['suction_grid']
    
    n_real = len(features)
    print(f"  Real samples: {n_real}")
    print(f"  Total synthetic samples: {n_real * num_samples_per_soil}")
    
    # Convert to tensors
    soil_props_tensor = tf.constant(features.astype(np.float32))
    theta_s_tensor = tf.constant(theta_s.astype(np.float32))
    theta_r_tensor = tf.constant(theta_r.astype(np.float32))
    
    # Generate in batches to avoid memory issues
    batch_size = 32
    all_generated = []
    all_soil_props = []
    all_theta_s = []
    all_theta_r = []
    
    for i in range(0, n_real, batch_size):
        batch_end = min(i + batch_size, n_real)
        batch_props = soil_props_tensor[i:batch_end]
        batch_theta_s = theta_s_tensor[i:batch_end]
        batch_theta_r = theta_r_tensor[i:batch_end]
        
        # Generate samples
        generated = model.generate_samples(
            batch_props,
            num_samples=num_samples_per_soil,
            theta_s=batch_theta_s,
            theta_r=batch_theta_r,
            seed=seed + i if seed is not None else None
        )
        
        # Convert to numpy
        generated_np = generated.numpy()
        
        # Expand soil properties for multiple samples
        batch_props_expanded = tf.repeat(batch_props, num_samples_per_soil, axis=0).numpy()
        batch_theta_s_expanded = tf.repeat(batch_theta_s, num_samples_per_soil, axis=0).numpy()
        batch_theta_r_expanded = tf.repeat(batch_theta_r, num_samples_per_soil, axis=0).numpy()
        
        all_generated.append(generated_np)
        all_soil_props.append(batch_props_expanded)
        all_theta_s.append(batch_theta_s_expanded)
        all_theta_r.append(batch_theta_r_expanded)
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Progress: {batch_end}/{n_real} samples processed")
    
    # Concatenate all batches
    synthetic_curves = np.vstack(all_generated)
    synthetic_soil_props = np.vstack(all_soil_props)
    synthetic_theta_s = np.hstack(all_theta_s)
    synthetic_theta_r = np.hstack(all_theta_r)
    
    print(f"✓ Generated {len(synthetic_curves)} synthetic curves")
    
    return {
        'swcc_curves': synthetic_curves,
        'soil_properties': synthetic_soil_props,
        'theta_s': synthetic_theta_s,
        'theta_r': synthetic_theta_r,
        'suction_grid': suction_grid,
        'feature_cols': data['feature_cols']
    }


def save_synthetic_data(synthetic_data, output_dir):
    """
    Save synthetic data to files
    
    Args:
        synthetic_data: Dictionary with synthetic data
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving synthetic data to {output_dir}...")
    
    # Save SWCC curves
    np.save(output_dir / 'synthetic_swcc_curves.npy', synthetic_data['swcc_curves'])
    print(f"  ✓ Saved SWCC curves: {synthetic_data['swcc_curves'].shape}")
    
    # Save soil properties as CSV
    df_props = pd.DataFrame(
        synthetic_data['soil_properties'],
        columns=synthetic_data['feature_cols']
    )
    df_props['theta_s'] = synthetic_data['theta_s']
    df_props['theta_r'] = synthetic_data['theta_r']
    df_props.to_csv(output_dir / 'synthetic_soil_properties.csv', index=False)
    print(f"  ✓ Saved soil properties: {df_props.shape}")
    
    # Save suction grid
    np.save(output_dir / 'suction_grid.npy', synthetic_data['suction_grid'])
    print(f"  ✓ Saved suction grid")
    
    # Save metadata
    metadata = {
        'n_samples': int(len(synthetic_data['swcc_curves'])),
        'n_points': int(synthetic_data['swcc_curves'].shape[1]),
        'n_features': int(len(synthetic_data['feature_cols'])),
        'feature_cols': synthetic_data['feature_cols'],
        'generation_date': datetime.now().isoformat(),
        'num_samples_per_soil': GENERATION_CONFIG.get('num_samples_per_soil', 20)
    }
    
    with open(output_dir / 'synthetic_data_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata")


def visualize_synthetic_data(synthetic_data, real_data, output_dir, n_samples=20):
    """
    Visualize synthetic vs real data
    
    Args:
        synthetic_data: Dictionary with synthetic data
        real_data: Dictionary with real data
        output_dir: Output directory
        n_samples: Number of samples to visualize
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating visualizations...")
    
    suction_grid = synthetic_data['suction_grid']
    
    # Sample random indices
    n_synthetic = len(synthetic_data['swcc_curves'])
    n_real = len(real_data['swcc_curves'])
    
    synthetic_indices = np.random.choice(n_synthetic, min(n_samples, n_synthetic), replace=False)
    real_indices = np.random.choice(n_real, min(n_samples, n_real), replace=False)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Sample synthetic curves
    ax = axes[0, 0]
    for idx in synthetic_indices[:10]:
        ax.semilogx(suction_grid, synthetic_data['swcc_curves'][idx], 
                   'b-', alpha=0.3, linewidth=0.5)
    ax.set_xlabel('Suction (kPa)')
    ax.set_ylabel('Water Content (θ)')
    ax.set_title('Sample Synthetic SWCC Curves')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Sample real curves
    ax = axes[0, 1]
    for idx in real_indices[:10]:
        ax.semilogx(suction_grid, real_data['swcc_curves'][idx], 
                   'r-', alpha=0.3, linewidth=0.5)
    ax.set_xlabel('Suction (kPa)')
    ax.set_ylabel('Water Content (θ)')
    ax.set_title('Sample Real SWCC Curves')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Mean and std comparison
    ax = axes[1, 0]
    synthetic_mean = np.mean(synthetic_data['swcc_curves'], axis=0)
    synthetic_std = np.std(synthetic_data['swcc_curves'], axis=0)
    real_mean = np.mean(real_data['swcc_curves'], axis=0)
    real_std = np.std(real_data['swcc_curves'], axis=0)
    
    ax.semilogx(suction_grid, synthetic_mean, 'b-', label='Synthetic Mean', linewidth=2)
    ax.fill_between(suction_grid, synthetic_mean - synthetic_std, 
                    synthetic_mean + synthetic_std, alpha=0.3, color='blue', label='Synthetic ±1σ')
    ax.semilogx(suction_grid, real_mean, 'r--', label='Real Mean', linewidth=2)
    ax.fill_between(suction_grid, real_mean - real_std, 
                    real_mean + real_std, alpha=0.3, color='red', label='Real ±1σ')
    ax.set_xlabel('Suction (kPa)')
    ax.set_ylabel('Water Content (θ)')
    ax.set_title('Mean and Standard Deviation Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Distribution at specific suction values
    ax = axes[1, 1]
    suction_indices = [10, 30, 50, 70, 90]  # Sample points
    for idx in suction_indices:
        synthetic_vals = synthetic_data['swcc_curves'][:, idx]
        real_vals = real_data['swcc_curves'][:, idx]
        ax.hist(synthetic_vals, bins=30, alpha=0.5, label=f'Synthetic (ψ={suction_grid[idx]:.1f})', density=True)
        ax.hist(real_vals, bins=30, alpha=0.5, label=f'Real (ψ={suction_grid[idx]:.1f})', density=True, histtype='step')
    ax.set_xlabel('Water Content (θ)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution Comparison at Selected Suction Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'synthetic_vs_real_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved visualization: {output_dir / 'synthetic_vs_real_comparison.png'}")


def main():
    """Main function"""
    print("="*80)
    print("Synthetic SWCC Data Generation")
    print("="*80)
    
    # Load trained model
    try:
        model, data = load_trained_model()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease train the model first:")
        print("  python3 training/train_gan.py")
        return
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(
        model,
        data,
        num_samples_per_soil=GENERATION_CONFIG['num_samples_per_soil'],
        seed=GENERATION_CONFIG['seed']
    )
    
    # Validate generated curves
    validation_results = validate_generated_curves(
        synthetic_data['swcc_curves'],
        synthetic_data['theta_s'],
        synthetic_data['theta_r'],
        synthetic_data['suction_grid']
    )
    
    # Save validation results
    validation_file = GENERATED_DIR / 'validation_results.json'
    with open(validation_file, 'w') as f:
        json.dump(validation_results, f, indent=2)
    print(f"\n✓ Saved validation results: {validation_file}")
    
    # Save synthetic data
    save_synthetic_data(synthetic_data, GENERATED_DIR)
    
    # Load real data for comparison
    real_data = {
        'swcc_curves': np.load(DATA_CONFIG['swcc_file']),
        'suction_grid': np.load(DATA_CONFIG['suction_grid_file'])
    }
    
    # Create visualizations
    visualize_synthetic_data(synthetic_data, real_data, GENERATED_DIR)
    
    print("\n" + "="*80)
    print("Synthetic Data Generation Complete!")
    print("="*80)
    print(f"\nResults saved to: {GENERATED_DIR}")
    print(f"  - Synthetic SWCC curves: {len(synthetic_data['swcc_curves'])} samples")
    print(f"  - Validation: {validation_results['boundary_satisfaction_rate']*100:.1f}% boundaries, "
          f"{validation_results['monotonicity_rate']*100:.1f}% monotonic")
    print(f"\nNext: Use synthetic data for PINN training (Phase 3)")


if __name__ == "__main__":
    main()
