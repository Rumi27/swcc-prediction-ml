"""
Fixed Configuration for PINN Training
- Normalized targets [0,1]
- Rebalanced loss weights
- Structural monotonicity
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data_pinn_normalized"  # Use normalized data
RESULTS_DIR = BASE_DIR / "results_pinn_fixed"
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
    'y_train_file': DATA_DIR / 'y_train.npy',  # Normalized [0,1]
    'y_val_file': DATA_DIR / 'y_val.npy',  # Normalized [0,1]
    'y_test_file': DATA_DIR / 'y_test.npy',  # Normalized [0,1]
    'y_train_original_file': DATA_DIR / 'y_train_original.npy',  # For evaluation
    'y_val_original_file': DATA_DIR / 'y_val_original.npy',
    'y_test_original_file': DATA_DIR / 'y_test_original.npy',
    'suction_grid_file': DATA_DIR / 'suction_grid.npy',
    'metadata_file': DATA_DIR / 'metadata.json',
    'batch_size': 32,
}

# Model configuration
MODEL_CONFIG = {
    'soil_prop_dim': 16,
    'suction_points': 100,
    'hidden_dims': [128, 256, 128, 64],
    'physics_units': 128,
    'alpha': 1.38,
}

# REBALANCED Training configuration - Phase 1: Data-driven pretraining
PHASE1_CONFIG = {
    'epochs': 300,
    'learning_rate': 1e-3,
    'lambda_data': 1.0,
    'lambda_mono': 0.01,  # Very small (structural monotonicity handles it)
    'lambda_bound': 0.1,
    'lambda_physics': 0.05,
}

# REBALANCED Training configuration - Phase 2: Joint tuning
PHASE2_CONFIG = {
    'epochs': 300,
    'learning_rate': 3e-4,
    'lambda_data': 1.0,
    'lambda_mono': 0.1,  # Moderate
    'lambda_bound': 0.3,
    'lambda_physics': 0.1,
}

# REBALANCED Training configuration - Phase 3: Physics refinement
PHASE3_CONFIG = {
    'epochs': 300,
    'learning_rate': 5e-5,
    'lambda_data': 1.0,
    'lambda_mono': 0.5,  # Higher but not extreme
    'lambda_bound': 0.5,
    'lambda_physics': 0.2,
}

# Training monitoring
MONITORING_CONFIG = {
    'checkpoint_freq': 25,
    'log_freq': 10,
    'plot_freq': 25,
    'early_stopping_patience': 50,
    'early_stopping_metric': 'val_rmse',  # Use RMSE, not total loss
    'min_delta': 1e-6,
}

# Device configuration
DEVICE_CONFIG = {
    'use_gpu': True,
    'gpu_memory_growth': True,
}

# Set GPU memory growth
if DEVICE_CONFIG['use_gpu']:
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ Using {len(gpus)} GPU(s)")
        else:
            print("⚠ No GPU found, using CPU")
            DEVICE_CONFIG['use_gpu'] = False
    except Exception as e:
        print(f"⚠ GPU configuration error: {e}")
        DEVICE_CONFIG['use_gpu'] = False

print("="*80)
print("Fixed PINN Configuration")
print("="*80)
print("\nKey Fixes:")
print("  1. Normalized targets [0,1] per sample")
print("  2. Structural monotonicity (cumulative sum)")
print("  3. Rebalanced loss weights (lower physics)")
print("  4. Early stopping on validation RMSE")
print("="*80)
