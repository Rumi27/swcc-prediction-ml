#!/usr/bin/env python3
"""
Prepare PINN Data with Normalized Targets
Normalizes y_train, y_val, y_test to [0,1] per sample
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

# Paths
DATA_DIR = Path("data_processed")
SYNTHETIC_DIR = Path("results_gan/generated_data_filtered")
OUTPUT_DIR = Path("data_pinn_normalized")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_swcc(y, theta_s, theta_r):
    """
    Normalize SWCC curves to [0,1] per sample
    
    Args:
        y: Water content [N, n_points]
        theta_s: Saturated water content [N]
        theta_r: Residual water content [N]
    
    Returns:
        y_norm: Normalized water content [N, n_points] in [0,1]
    """
    theta_s_2d = theta_s.reshape(-1, 1) if len(theta_s.shape) == 1 else theta_s
    theta_r_2d = theta_r.reshape(-1, 1) if len(theta_r.shape) == 1 else theta_r
    theta_range = theta_s_2d - theta_r_2d
    
    # Avoid division by zero
    theta_range = np.maximum(theta_range, 1e-6)
    
    # Normalize: (θ - θr) / (θs - θr)
    y_norm = (y - theta_r_2d) / theta_range
    
    # Clip to [0, 1]
    y_norm = np.clip(y_norm, 0.0, 1.0)
    
    return y_norm


def load_real_data():
    """Load all real data splits (train, val, test)"""
    print("Loading real data...")
    
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = np.load(DATA_DIR / "y_train.npy")
    
    X_val = pd.read_csv(DATA_DIR / "X_val.csv")
    y_val = np.load(DATA_DIR / "y_val.npy")
    
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = np.load(DATA_DIR / "y_test.npy")
    
    suction_grid = np.load(DATA_DIR / "suction_grid.npy")
    
    print(f"  Real training: {len(X_train)} samples")
    print(f"  Real validation: {len(X_val)} samples")
    print(f"  Real test: {len(X_test)} samples")
    print(f"  Total real: {len(X_train) + len(X_val) + len(X_test)} samples")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  SWCC points: {y_train.shape[1]}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), suction_grid


def load_synthetic_data():
    """Load filtered synthetic data"""
    print("\nLoading filtered synthetic data...")
    
    synthetic_curves = np.load(SYNTHETIC_DIR / "synthetic_swcc_curves_filtered.npy")
    synthetic_props = pd.read_csv(SYNTHETIC_DIR / "synthetic_soil_properties_filtered.csv")
    suction_grid = np.load(DATA_DIR / "suction_grid.npy")  # Use same grid
    
    print(f"  Synthetic samples: {len(synthetic_curves)}")
    print(f"  Features: {synthetic_props.shape[1]}")
    print(f"  SWCC points: {synthetic_curves.shape[1]}")
    
    return synthetic_props, synthetic_curves, suction_grid


def normalize_and_combine_train(X_real_train, y_real_train, X_synthetic, y_synthetic):
    """Combine real training data with synthetic data and normalize"""
    print("\nCombining training data (real + synthetic)...")
    
    # Ensure feature columns match
    real_cols = set(X_real_train.columns)
    synth_cols = set(X_synthetic.columns)
    
    if real_cols != synth_cols:
        common_cols = sorted(list(real_cols & synth_cols))
        X_real_train = X_real_train[common_cols]
        X_synthetic = X_synthetic[common_cols]
        print(f"  Using common columns: {len(common_cols)}")
    
    # Impute NaN values in features (use column means)
    print("  Imputing NaN values in features...")
    X_real_train_imputed = X_real_train.copy()
    for col in X_real_train_imputed.columns:
        if X_real_train_imputed[col].isna().any():
            col_mean = X_real_train_imputed[col].mean()
            if pd.isna(col_mean):
                col_mean = 0.0  # Fallback if all values are NaN
            X_real_train_imputed[col] = X_real_train_imputed[col].fillna(col_mean)
            print(f"    Imputed {X_real_train[col].isna().sum()} NaNs in {col} with mean={col_mean:.4f}")
    
    # Remove rows with NaN in y (should be none, but check)
    y_mask = ~np.isnan(y_real_train).any(axis=1)
    X_real_clean = X_real_train_imputed[y_mask].reset_index(drop=True)
    y_real_clean = y_real_train[y_mask]
    
    if not y_mask.all():
        print(f"  Removed {np.sum(~y_mask)} rows with NaN in y")
    
    # Impute NaN in synthetic features
    X_synthetic_imputed = X_synthetic.copy()
    for col in X_synthetic_imputed.columns:
        if X_synthetic_imputed[col].isna().any():
            col_mean = X_synthetic_imputed[col].mean()
            if pd.isna(col_mean):
                col_mean = 0.0
            X_synthetic_imputed[col] = X_synthetic_imputed[col].fillna(col_mean)
    
    # Remove rows with NaN in y
    y_synth_mask = ~np.isnan(y_synthetic).any(axis=1)
    X_synthetic_clean = X_synthetic_imputed[y_synth_mask].reset_index(drop=True)
    y_synthetic_clean = y_synthetic[y_synth_mask]
    
    # Combine training data
    X_train_combined = pd.concat([X_real_clean, X_synthetic_clean], ignore_index=True)
    y_train_combined = np.vstack([y_real_clean, y_synthetic_clean])
    
    print(f"  Real training: {len(X_real_clean)} samples")
    print(f"  Synthetic: {len(X_synthetic_clean)} samples")
    print(f"  Combined training: {len(X_train_combined)} samples")
    print(f"  Augmentation ratio: {len(X_train_combined)/len(X_real_clean):.2f}x")
    
    # Normalize SWCC curves
    print("\nNormalizing training SWCC curves to [0,1]...")
    theta_s = X_train_combined['theta_s'].values
    theta_r = X_train_combined['theta_r'].values
    
    y_train_norm = normalize_swcc(y_train_combined, theta_s, theta_r)
    
    print(f"  Original y range: [{y_train_combined.min():.4f}, {y_train_combined.max():.4f}]")
    print(f"  Normalized y range: [{y_train_norm.min():.4f}, {y_train_norm.max():.4f}]")
    
    return X_train_combined, y_train_combined, y_train_norm


def normalize_val_test(X_val, y_val, X_test, y_test):
    """Normalize validation and test data (real only)"""
    print("\nNormalizing validation and test data (real only)...")
    
    # Impute NaN in validation and test features
    X_val_imputed = X_val.copy()
    for col in X_val_imputed.columns:
        if X_val_imputed[col].isna().any():
            col_mean = X_val_imputed[col].mean()
            if pd.isna(col_mean):
                col_mean = 0.0
            X_val_imputed[col] = X_val_imputed[col].fillna(col_mean)
    
    X_test_imputed = X_test.copy()
    for col in X_test_imputed.columns:
        if X_test_imputed[col].isna().any():
            col_mean = X_test_imputed[col].mean()
            if pd.isna(col_mean):
                col_mean = 0.0
            X_test_imputed[col] = X_test_imputed[col].fillna(col_mean)
    
    # Normalize validation
    theta_s_val = X_val_imputed['theta_s'].values
    theta_r_val = X_val_imputed['theta_r'].values
    y_val_norm = normalize_swcc(y_val, theta_s_val, theta_r_val)
    
    # Normalize test
    theta_s_test = X_test_imputed['theta_s'].values
    theta_r_test = X_test_imputed['theta_r'].values
    y_test_norm = normalize_swcc(y_test, theta_s_test, theta_r_test)
    
    print(f"  Validation: {len(X_val_imputed)} samples")
    print(f"  Test: {len(X_test_imputed)} samples")
    
    return X_val_imputed, y_val_norm, X_test_imputed, y_test_norm


def save_data(X_train, y_train_orig, y_train_norm,
              X_val, y_val_orig, y_val_norm,
              X_test, y_test_orig, y_test_norm,
              suction_grid):
    """Save all data splits"""
    print("\nSaving data...")
    
    # Save normalized data (for training)
    print("  Saving normalized data (for training)...")
    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    X_val.to_csv(OUTPUT_DIR / "X_val.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    
    np.save(OUTPUT_DIR / "y_train.npy", y_train_norm)
    np.save(OUTPUT_DIR / "y_val.npy", y_val_norm)
    np.save(OUTPUT_DIR / "y_test.npy", y_test_norm)
    
    # Save original data (for evaluation)
    print("  Saving original data (for evaluation)...")
    np.save(OUTPUT_DIR / "y_train_original.npy", y_train_orig)
    np.save(OUTPUT_DIR / "y_val_original.npy", y_val_orig)
    np.save(OUTPUT_DIR / "y_test_original.npy", y_test_orig)
    
    np.save(OUTPUT_DIR / "suction_grid.npy", suction_grid)
    
    # Metadata
    metadata = {
        'n_train': int(len(X_train)),
        'n_val': int(len(X_val)),
        'n_test': int(len(X_test)),
        'n_features': int(X_train.shape[1]),
        'n_swcc_points': int(y_train_norm.shape[1]),
        'feature_cols': X_train.columns.tolist(),
        'normalized': True,
        'normalization_method': 'per_sample_theta_range',
        'training_data': 'real + synthetic',
        'validation_data': 'real only',
        'test_data': 'real only'
    }
    
    with open(OUTPUT_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("  ✓ Data saved")
    
    return metadata


def main():
    """Main function"""
    print("="*80)
    print("Preparing PINN Data with Normalized Targets")
    print("Training: Real + Synthetic | Validation/Test: Real only")
    print("="*80)
    
    # Load all real data splits
    (X_train_real, y_train_real), (X_val_real, y_val_real), (X_test_real, y_test_real), suction_grid = load_real_data()
    
    # Load synthetic data
    X_synthetic, y_synthetic, _ = load_synthetic_data()
    
    # Combine training data (real + synthetic) and normalize
    X_train_combined, y_train_orig, y_train_norm = normalize_and_combine_train(
        X_train_real, y_train_real, X_synthetic, y_synthetic
    )
    
    # Normalize validation and test (real only)
    X_val_clean, y_val_norm, X_test_clean, y_test_norm = normalize_val_test(
        X_val_real, y_val_real, X_test_real, y_test_real
    )
    
    # Save all data
    metadata = save_data(
        X_train_combined, y_train_orig, y_train_norm,
        X_val_clean, y_val_real, y_val_norm,
        X_test_clean, y_test_real, y_test_norm,
        suction_grid
    )
    
    print("\n" + "="*80)
    print("Data Preparation Complete!")
    print("="*80)
    print(f"\nData saved to: {OUTPUT_DIR}")
    print(f"  Train: {metadata['n_train']} samples (real + synthetic)")
    print(f"  Val: {metadata['n_val']} samples (real only)")
    print(f"  Test: {metadata['n_test']} samples (real only)")
    print(f"  Normalized: {metadata['normalized']}")
    print(f"\nNext: Train PINN with normalized targets")


if __name__ == "__main__":
    main()
