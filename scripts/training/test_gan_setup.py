#!/usr/bin/env python3
"""
Test GAN Setup
Verify that all components are working correctly before training
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

import tensorflow as tf
from models.wgan_gp import WGAN_GP
from training.config import *
from training.train_utils import DataLoader

def test_model_initialization():
    """Test model initialization"""
    print("="*60)
    print("Testing Model Initialization")
    print("="*60)
    
    model = WGAN_GP(
        noise_dim=MODEL_CONFIG['noise_dim'],
        soil_prop_dim=16,  # Test dimension
        swcc_points=MODEL_CONFIG['swcc_points'],
        lambda_gp=TRAINING_CONFIG['lambda_gp'],
        lambda_mono=TRAINING_CONFIG['lambda_mono_phase1'],
        lambda_bound=TRAINING_CONFIG['lambda_bound_phase1']
    )
    
    # Test forward pass
    batch_size = 4
    noise = tf.random.normal([batch_size, MODEL_CONFIG['noise_dim']])
    soil_props = tf.random.normal([batch_size, 16])
    real_swcc = tf.random.normal([batch_size, MODEL_CONFIG['swcc_points']])
    
    # Generator
    fake_swcc = model.generator([noise, soil_props])
    print(f"✓ Generator output shape: {fake_swcc.shape}")
    assert fake_swcc.shape == (batch_size, MODEL_CONFIG['swcc_points']), "Generator shape mismatch"
    
    # Discriminator
    real_score = model.discriminator([real_swcc, soil_props])
    fake_score = model.discriminator([fake_swcc, soil_props])
    print(f"✓ Discriminator real score shape: {real_score.shape}")
    print(f"✓ Discriminator fake score shape: {fake_score.shape}")
    assert real_score.shape == (batch_size, 1), "Discriminator shape mismatch"
    
    # Losses
    theta_s = tf.ones([batch_size]) * 0.5
    theta_r = tf.ones([batch_size]) * 0.1
    
    d_loss, wd, gp = model.discriminator_loss(real_swcc, fake_swcc, soil_props)
    g_loss, g_adv, mono, bound = model.generator_loss(fake_swcc, soil_props, theta_s, theta_r)
    
    print(f"✓ Discriminator loss: {d_loss.numpy():.4f}")
    print(f"✓ Generator loss: {g_loss.numpy():.4f}")
    print(f"✓ Wasserstein distance: {wd.numpy():.4f}")
    print(f"✓ Gradient penalty: {gp.numpy():.4f}")
    print(f"✓ Monotonicity loss: {mono.numpy():.4f}")
    print(f"✓ Boundary loss: {bound.numpy():.4f}")
    
    print("\n✓ Model initialization test passed!")
    return model


def test_data_loading():
    """Test data loading"""
    print("\n" + "="*60)
    print("Testing Data Loading")
    print("="*60)
    
    loader = DataLoader({
        'train_file': DATA_CONFIG['train_file'],
        'swcc_file': DATA_CONFIG['swcc_file'],
        'suction_grid_file': DATA_CONFIG['suction_grid_file'],
        'feature_cols': FEATURE_COLS
    })
    
    data = loader.load_data()
    
    # Check data shapes
    print(f"\nData shapes:")
    print(f"  Features: {data['features'].shape}")
    print(f"  SWCC curves: {data['swcc_curves'].shape}")
    print(f"  Theta_s: {data['theta_s'].shape}")
    print(f"  Theta_r: {data['theta_r'].shape}")
    print(f"  Suction grid: {data['suction_grid'].shape}")
    
    # Create dataset
    dataset = loader.create_dataset(
        data['features'],
        data['swcc_curves'],
        data['theta_s'],
        data['theta_r'],
        batch_size=8,
        shuffle=False
    )
    
    # Get one batch
    batch = next(iter(dataset))
    print(f"\nBatch shapes:")
    print(f"  Soil props: {batch['soil_props'].shape}")
    print(f"  SWCC curve: {batch['swcc_curve'].shape}")
    print(f"  Theta_s: {batch['theta_s'].shape}")
    print(f"  Theta_r: {batch['theta_r'].shape}")
    
    print("\n✓ Data loading test passed!")
    return data, dataset


def test_physics_constraints():
    """Test physics constraints"""
    print("\n" + "="*60)
    print("Testing Physics Constraints")
    print("="*60)
    
    from models.physics_constraints import PhysicsConstraints
    
    physics = PhysicsConstraints(lambda_mono=0.5, lambda_bound=0.3)
    
    # Test monotonicity
    batch_size = 4
    n_points = 100
    
    # Good curve (decreasing)
    good_curve = tf.linspace(0.5, 0.1, n_points)
    good_curve = tf.tile(tf.expand_dims(good_curve, 0), [batch_size, 1])
    
    # Bad curve (increasing in some parts)
    bad_curve = tf.random.uniform([batch_size, n_points], 0.1, 0.5)
    
    mono_good = physics.monotonicity_loss(good_curve)
    mono_bad = physics.monotonicity_loss(bad_curve)
    
    print(f"Monotonicity loss (good curve): {mono_good.numpy():.6f}")
    print(f"Monotonicity loss (bad curve): {mono_bad.numpy():.6f}")
    assert mono_good < mono_bad, "Monotonicity loss should be lower for good curves"
    
    # Test boundaries
    theta_s = tf.ones([batch_size]) * 0.5
    theta_r = tf.ones([batch_size]) * 0.1
    
    # Good curve (within bounds)
    good_curve_bounds = tf.linspace(0.5, 0.1, n_points)
    good_curve_bounds = tf.tile(tf.expand_dims(good_curve_bounds, 0), [batch_size, 1])
    
    # Bad curve (outside bounds)
    bad_curve_bounds = tf.random.uniform([batch_size, n_points], 0.0, 1.0)
    
    bound_good = physics.boundary_loss(good_curve_bounds, theta_s, theta_r)
    bound_bad = physics.boundary_loss(bad_curve_bounds, theta_s, theta_r)
    
    print(f"Boundary loss (good curve): {bound_good.numpy():.6f}")
    print(f"Boundary loss (bad curve): {bound_bad.numpy():.6f}")
    assert bound_good < bound_bad, "Boundary loss should be lower for good curves"
    
    print("\n✓ Physics constraints test passed!")


def test_generation():
    """Test sample generation"""
    print("\n" + "="*60)
    print("Testing Sample Generation")
    print("="*60)
    
    model = WGAN_GP(
        noise_dim=MODEL_CONFIG['noise_dim'],
        soil_prop_dim=16,
        swcc_points=MODEL_CONFIG['swcc_points']
    )
    
    # Generate samples
    batch_size = 4
    soil_props = tf.random.normal([batch_size, 16])
    theta_s = tf.ones([batch_size]) * 0.5
    theta_r = tf.ones([batch_size]) * 0.1
    
    generated = model.generate_samples(
        soil_props,
        num_samples=5,
        theta_s=theta_s,
        theta_r=theta_r
    )
    
    print(f"✓ Generated shape: {generated.shape}")
    print(f"  Expected: ({batch_size * 5}, {MODEL_CONFIG['swcc_points']})")
    assert generated.shape == (batch_size * 5, MODEL_CONFIG['swcc_points']), "Generation shape mismatch"
    
    # Check boundaries
    assert tf.reduce_all(generated >= theta_r[0] - 0.01), "Generated curves below θr"
    assert tf.reduce_all(generated <= theta_s[0] + 0.01), "Generated curves above θs"
    
    print("\n✓ Sample generation test passed!")


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("GAN Setup Test Suite")
    print("="*80)
    
    try:
        # Test 1: Model initialization
        model = test_model_initialization()
        
        # Test 2: Data loading
        data, dataset = test_data_loading()
        
        # Test 3: Physics constraints
        test_physics_constraints()
        
        # Test 4: Generation
        test_generation()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        print("\nReady to start training:")
        print("  python3 training/train_gan.py")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
