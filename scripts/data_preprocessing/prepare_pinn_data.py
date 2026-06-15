#!/usr/bin/env python3
"""
Prepare Combined Dataset for PINN Training
Combines real data + filtered synthetic data
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

# Paths
DATA_DIR = Path("data_processed")
SYNTHETIC_DIR = Path("results_gan/generated_data_filtered")
OUTPUT_DIR = Path("data_pinn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_real_data():
    """Load real training data"""
    print("Loading real data...")
    
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = np.load(DATA_DIR / "y_train.npy")
    suction_grid = np.load(DATA_DIR / "suction_grid.npy")
    
    print(f"  Real samples: {len(X_train)}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  SWCC points: {y_train.shape[1]}")
    
    return X_train, y_train, suction_grid


def load_synthetic_data():
    """Load filtered synthetic data"""
    print("\nLoading filtered synthetic data...")
    
    synthetic_curves = np.load(SYNTHETIC_DIR / "synthetic_swcc_curves_filtered.npy")
    synthetic_props = pd.read_csv(SYNTHETIC_DIR / "synthetic_soil_properties_filtered.csv")
    suction_grid = np.load(SYNTHETIC_DIR / "suction_grid.npy")
    
    print(f"  Synthetic samples: {len(synthetic_curves)}")
    print(f"  Features: {synthetic_props.shape[1]}")
    print(f"  SWCC points: {synthetic_curves.shape[1]}")
    
    return synthetic_props, synthetic_curves, suction_grid


def combine_datasets(X_real, y_real, X_synthetic, y_synthetic, suction_grid):
    """Combine real and synthetic datasets"""
    print("\nCombining datasets...")
    
    # Ensure feature columns match
    real_cols = set(X_real.columns)
    synth_cols = set(X_synthetic.columns)
    
    if real_cols != synth_cols:
        print(f"  Warning: Column mismatch")
        print(f"  Real columns: {real_cols - synth_cols}")
        print(f"  Synthetic columns: {synth_cols - real_cols}")
        
        # Use intersection
        common_cols = sorted(list(real_cols & synth_cols))
        X_real = X_real[common_cols]
        X_synthetic = X_synthetic[common_cols]
        print(f"  Using common columns: {len(common_cols)}")
    
    # Remove NaN rows from real data
    print("\nRemoving NaN values...")
    real_mask = ~X_real.isna().any(axis=1) & ~np.isnan(y_real).any(axis=1)
    X_real_clean = X_real[real_mask].reset_index(drop=True)
    y_real_clean = y_real[real_mask]
    
    synth_mask = ~X_synthetic.isna().any(axis=1) & ~np.isnan(y_synthetic).any(axis=1)
    X_synthetic_clean = X_synthetic[synth_mask].reset_index(drop=True)
    y_synthetic_clean = y_synthetic[synth_mask]
    
    print(f"  Real: {len(X_real)} → {len(X_real_clean)} (removed {len(X_real) - len(X_real_clean)} NaN rows)")
    print(f"  Synthetic: {len(X_synthetic)} → {len(X_synthetic_clean)} (removed {len(X_synthetic) - len(X_synthetic_clean)} NaN rows)")
    
    # Combine
    X_combined = pd.concat([X_real_clean, X_synthetic_clean], ignore_index=True)
    y_combined = np.vstack([y_real_clean, y_synthetic_clean])
    
    print(f"  Combined samples: {len(X_combined)}")
    print(f"    Real: {len(X_real)}")
    print(f"    Synthetic: {len(X_synthetic)}")
    print(f"    Augmentation ratio: {len(X_combined)/len(X_real):.2f}x")
    
    return X_combined, y_combined


def split_data(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split data into train/val/test sets"""
    print("\nSplitting data...")
    
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_val = X.iloc[val_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]
    
    print(f"  Train: {len(X_train)} ({len(X_train)/n_samples*100:.1f}%)")
    print(f"  Val: {len(X_val)} ({len(X_val)/n_samples*100:.1f}%)")
    print(f"  Test: {len(X_test)} ({len(X_test)/n_samples*100:.1f}%)")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def save_data(train_data, val_data, test_data, suction_grid):
    """Save prepared data"""
    print("\nSaving prepared data...")
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_data, val_data, test_data
    
    # Save features
    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    X_val.to_csv(OUTPUT_DIR / "X_val.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    print("  ✓ Saved feature files")
    
    # Save SWCC curves
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "y_val.npy", y_val)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)
    print("  ✓ Saved SWCC curves")
    
    # Save suction grid
    np.save(OUTPUT_DIR / "suction_grid.npy", suction_grid)
    print("  ✓ Saved suction grid")
    
    # Save metadata
    metadata = {
        'n_train': int(len(X_train)),
        'n_val': int(len(X_val)),
        'n_test': int(len(X_test)),
        'n_features': int(X_train.shape[1]),
        'n_swcc_points': int(y_train.shape[1]),
        'feature_cols': X_train.columns.tolist(),
        'augmentation_ratio': float(len(X_train) / 389)  # Original real data size
    }
    
    with open(OUTPUT_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print("  ✓ Saved metadata")
    
    return metadata


def main():
    """Main function"""
    print("="*80)
    print("Preparing PINN Training Data")
    print("="*80)
    
    # Set random seed
    np.random.seed(42)
    
    # Load data
    X_real, y_real, suction_grid = load_real_data()
    X_synthetic, y_synthetic, _ = load_synthetic_data()
    
    # Combine
    X_combined, y_combined = combine_datasets(
        X_real, y_real, X_synthetic, y_synthetic, suction_grid
    )
    
    # Split
    train_data, val_data, test_data = split_data(
        X_combined, y_combined,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # Save
    metadata = save_data(train_data, val_data, test_data, suction_grid)
    
    print("\n" + "="*80)
    print("Data Preparation Complete!")
    print("="*80)
    print(f"\nData saved to: {OUTPUT_DIR}")
    print(f"  Train: {metadata['n_train']} samples")
    print(f"  Val: {metadata['n_val']} samples")
    print(f"  Test: {metadata['n_test']} samples")
    print(f"  Augmentation ratio: {metadata['augmentation_ratio']:.2f}x")
    print(f"\nNext: Train PINN model")


if __name__ == "__main__":
    main()
