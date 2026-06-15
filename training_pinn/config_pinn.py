"""
Configuration for PINN Training
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data_pinn"
RESULTS_DIR = BASE_DIR / "results_pinn"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
VIZ_DIR = RESULTS_DIR / "visualizations"

# Create directories
for dir_path in [RESULTS_DIR, CHECKPOINT_DIR, VIZ_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data configuration
DATA_CONFIG = {
    'train_file': DATA_DIR / 'X_train.csv',
    'val_file': DATA_DIR / 'X_val.csv',
    'test_file': DATA_DIR / 'X_test.csv',
    'y_train_file': DATA_DIR / 'y_train.npy',
    'y_val_file': DATA_DIR / 'y_val.npy',
    'y_test_file': DATA_DIR / 'y_test.npy',
    'suction_grid_file': DATA_DIR / 'suction_grid.npy',
    'metadata_file': DATA_DIR / 'metadata.json',
    'batch_size': 32,
}

# Model configuration
MODEL_CONFIG = {
    'soil_prop_dim': 16,  # Will be determined from data
    'suction_points': 100,
    'hidden_dims': [128, 256, 128, 64],
    'physics_units': 128,
    'alpha': 1.38,  # Arya-Paris parameter
}

# Training configuration - Phase 1: Pre-training
PHASE1_CONFIG = {
    'epochs': 200,
    'learning_rate': 5e-4,  # Reduced from 1e-3 for stability
    'lambda_data': 1.0,
    'lambda_mono': 0.1,
    'lambda_bound': 0.1,
    'lambda_physics': 0.01,  # Reduced from 0.05 (physics loss is still large)
}

# Training configuration - Phase 2: Physics-Informed
PHASE2_CONFIG = {
    'epochs': 300,
    'learning_rate': 2e-4,  # Reduced from 5e-4 for stability
    'lambda_data': 1.0,
    'lambda_mono': 0.5,
    'lambda_bound': 0.3,
    'lambda_physics': 0.05,  # Reduced from 0.2
}

# Training configuration - Phase 3: Fine-tuning
PHASE3_CONFIG = {
    'epochs': 100,
    'learning_rate': 1e-4,
    'lambda_data': 1.0,
    'lambda_mono': 1.0,
    'lambda_bound': 0.5,
    'lambda_physics': 0.1,  # Reduced from 0.3
}

# Training monitoring
MONITORING_CONFIG = {
    'checkpoint_freq': 25,  # Save checkpoint every N epochs
    'log_freq': 10,  # Log metrics every N epochs
    'plot_freq': 25,  # Generate plots every N epochs
    'early_stopping_patience': 30,  # Stop if no improvement for N epochs
}

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
