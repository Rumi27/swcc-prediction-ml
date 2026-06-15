"""
Optimized Configuration for PINN Training
Based on evaluation results and best practices
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data_pinn"
RESULTS_DIR = BASE_DIR / "results_pinn_optimized"  # New directory for optimized training
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
    'hidden_dims': [128, 256, 128, 64],  # Keep same architecture
    'physics_units': 128,
    'alpha': 1.38,  # Arya-Paris parameter
}

# OPTIMIZED Training configuration - Phase 1: Pre-training (Data-driven)
PHASE1_CONFIG = {
    'epochs': 300,  # Increased from 200 for better data fitting
    'learning_rate': 1e-3,  # Higher initial LR for faster learning
    'lambda_data': 1.0,
    'lambda_mono': 0.5,  # INCREASED from 0.1 - enforce monotonicity from start
    'lambda_bound': 0.3,  # INCREASED from 0.1 - enforce boundaries from start
    'lambda_physics': 0.01,  # Keep low initially
}

# OPTIMIZED Training configuration - Phase 2: Physics-Informed
PHASE2_CONFIG = {
    'epochs': 400,  # Increased from 300 for better convergence
    'learning_rate': 3e-4,  # Slightly higher than before
    'lambda_data': 1.0,
    'lambda_mono': 2.0,  # SIGNIFICANTLY INCREASED from 0.5 - strong monotonicity enforcement
    'lambda_bound': 1.0,  # INCREASED from 0.3 - stronger boundary enforcement
    'lambda_physics': 0.1,  # INCREASED from 0.05 - more physics guidance
}

# OPTIMIZED Training configuration - Phase 3: Fine-tuning (Strong Physics)
PHASE3_CONFIG = {
    'epochs': 200,  # Increased from 100 for better fine-tuning
    'learning_rate': 5e-5,  # Lower for fine-tuning
    'lambda_data': 1.0,
    'lambda_mono': 5.0,  # VERY HIGH - maximum monotonicity enforcement
    'lambda_bound': 2.0,  # INCREASED from 0.5 - very strong boundaries
    'lambda_physics': 0.2,  # INCREASED from 0.1 - strong physics
}

# Training monitoring
MONITORING_CONFIG = {
    'checkpoint_freq': 25,  # Save checkpoint every N epochs
    'log_freq': 10,  # Log metrics every N epochs
    'plot_freq': 25,  # Generate plots every N epochs
    'early_stopping_patience': 50,  # INCREASED from 30 - allow more time for improvement
    'min_delta': 1e-6,  # Minimum change to qualify as improvement
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

# Summary of optimizations
OPTIMIZATION_SUMMARY = """
Optimized Hyperparameters Summary:
===================================

Key Changes from Previous Training:
1. Monotonicity weights SIGNIFICANTLY increased:
   - Phase 1: 0.1 → 0.5 (5x increase)
   - Phase 2: 0.5 → 2.0 (4x increase)
   - Phase 3: 1.0 → 5.0 (5x increase)

2. Boundary weights increased:
   - Phase 1: 0.1 → 0.3 (3x increase)
   - Phase 2: 0.3 → 1.0 (3.3x increase)
   - Phase 3: 0.5 → 2.0 (4x increase)

3. Training epochs increased:
   - Phase 1: 200 → 300 (+50%)
   - Phase 2: 300 → 400 (+33%)
   - Phase 3: 100 → 200 (+100%)
   - Total: 600 → 900 epochs

4. Early stopping patience increased:
   - 30 → 50 epochs (more time for improvement)

5. Learning rates adjusted:
   - Phase 1: 5e-4 → 1e-3 (higher for faster initial learning)
   - Phase 2: 2e-4 → 3e-4 (slightly higher)
   - Phase 3: 1e-4 → 5e-5 (lower for fine-tuning)

Expected Improvements:
- Higher monotonicity compliance (target: >95% without post-processing)
- Better physics consistency
- Comparable or better RMSE than baseline
- Positive R² values
"""

print(OPTIMIZATION_SUMMARY)
