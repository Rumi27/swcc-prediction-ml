"""
Configuration for GAN Training
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data_processed"
RESULTS_DIR = BASE_DIR / "results_gan"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
GENERATED_DIR = RESULTS_DIR / "generated_data"
VIZ_DIR = RESULTS_DIR / "visualizations"

# Create directories
for dir_path in [RESULTS_DIR, CHECKPOINT_DIR, GENERATED_DIR, VIZ_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data configuration
DATA_CONFIG = {
    'train_file': DATA_DIR / 'X_train.csv',
    'val_file': DATA_DIR / 'y_train.npy',
    'swcc_file': DATA_DIR / 'y_train.npy',
    'suction_grid_file': DATA_DIR / 'suction_grid.npy',
    'batch_size': 32,
    'shuffle_buffer': 1000,
}

# Model configuration
MODEL_CONFIG = {
    'noise_dim': 100,
    'soil_prop_dim': 16,  # Will be determined from data
    'swcc_points': 100,
    'generator_hidden': [256, 512, 256],
    'discriminator_hidden': [256, 128, 64],
    'dropout_rate': 0.3,
}

# Training configuration
TRAINING_CONFIG = {
    'total_epochs': 350,
    'phase1_epochs': 100,  # Pre-training
    'phase2_epochs': 200,  # Physics-informed
    'phase3_epochs': 50,   # Fine-tuning
    
    'd_updates_per_g': 5,  # Discriminator updates per generator update
    
    # Recommended WGAN-GP settings
    'learning_rate_g': 1e-4,
    'learning_rate_d': 1e-4,
    'beta1': 0.0,
    'beta2': 0.9,
    
    'lambda_gp': 10.0,  # Gradient penalty weight
    
    # Physics weights (will change by phase)
    # Physics weights (reduced after structural constraints)
    'lambda_mono_phase1': 0.0,
    'lambda_bound_phase1': 0.0,
    'lambda_mono_phase2': 0.0,
    'lambda_bound_phase2': 0.0,
    'lambda_mono_phase3': 0.0,
    'lambda_bound_phase3': 0.0,
    
    'checkpoint_freq': 10,  # Save checkpoint every N epochs (reduced for more frequent saves)
    'log_freq': 10,  # Log metrics every N epochs
    'sample_freq': 25,  # Generate sample curves every N epochs
}

# Generation configuration
GENERATION_CONFIG = {
    'num_samples_per_soil': 20,  # Generate 20 synthetic curves per real sample
    'seed': 42,
}

# Feature selection (which soil properties to use)
FEATURE_COLS = [
    'D10', 'D30', 'D50', 'D60', 'D90',
    'Cu', 'Cc',
    'clay_pct', 'silt_pct', 'sand_pct',
    'bulk_density', 'porosity',
    'OM_content', 'pH',
    'theta_s', 'theta_r'
]

# Device configuration
DEVICE_CONFIG = {
    'use_gpu': True,
    'gpu_memory_growth': True,
}

# Set GPU memory growth if using TensorFlow
if DEVICE_CONFIG['use_gpu']:
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, DEVICE_CONFIG['gpu_memory_growth'])
            print(f"✓ Using {len(gpus)} GPU(s)")
        else:
            print("⚠ No GPU found, using CPU")
            DEVICE_CONFIG['use_gpu'] = False
    except Exception as e:
        print(f"⚠ GPU configuration error: {e}")
        DEVICE_CONFIG['use_gpu'] = False
