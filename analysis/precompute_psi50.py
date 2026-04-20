#!/usr/bin/env python3
"""
Precompute ψ50 (suction at Se=0.5) for all training/validation/test curves.
This is used as a target for the ψ50 loss in VGParamNet training.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_pinn.config_pinn_fixed import DATA_CONFIG, RESULTS_DIR


def compute_effective_saturation(theta, theta_s, theta_r):
    """Compute effective saturation Se = (θ - θr)/(θs - θr)"""
    denom = np.maximum(theta_s - theta_r, 1e-6)
    Se = (theta - theta_r) / denom
    return np.clip(Se, 0.0, 1.0)


def find_psi50(psi, theta, theta_s, theta_r):
    """
    Find suction at which Se = 0.5 (knee location).
    
    Returns:
        psi_50: suction at Se=0.5 (interpolated), or NaN if not found
    """
    Se = compute_effective_saturation(theta, theta_s, theta_r)
    
    # Find where Se crosses 0.5
    if np.any(Se >= 0.5) and np.any(Se <= 0.5):
        # Interpolate to find exact crossing
        idx_above = np.where(Se >= 0.5)[0]
        idx_below = np.where(Se <= 0.5)[0]
        
        if len(idx_above) > 0 and len(idx_below) > 0:
            # Find closest pair
            i_above = idx_above[-1]  # Last point >= 0.5
            i_below = idx_below[0] if idx_below[0] > i_above else (i_above + 1)
            
            if i_below < len(Se):
                # Linear interpolation
                Se_above = Se[i_above]
                Se_below = Se[i_below]
                if Se_above != Se_below:
                    w = (0.5 - Se_above) / (Se_below - Se_above)
                    psi_50 = psi[i_above] * (1 - w) + psi[i_below] * w
                    return psi_50
    
    # Fallback: find closest point to Se=0.5
    idx_closest = np.argmin(np.abs(Se - 0.5))
    return psi[idx_closest]


def main():
    print("=" * 80)
    print("Precomputing ψ50 for all datasets")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data...")
    X_train = pd.read_csv(DATA_CONFIG["train_file"])
    X_val = pd.read_csv(DATA_CONFIG["val_file"])
    X_test = pd.read_csv(DATA_CONFIG["test_file"])
    
    y_train = np.load(DATA_CONFIG["y_train_original_file"]).astype(np.float32)
    y_val = np.load(DATA_CONFIG["y_val_original_file"]).astype(np.float32)
    y_test = np.load(DATA_CONFIG["y_test_original_file"]).astype(np.float32)
    
    psi = np.load(DATA_CONFIG["suction_grid_file"]).astype(np.float32)
    
    theta_s_train = X_train["theta_s"].values.astype(np.float32)
    theta_r_train = X_train["theta_r"].values.astype(np.float32)
    theta_s_val = X_val["theta_s"].values.astype(np.float32)
    theta_r_val = X_val["theta_r"].values.astype(np.float32)
    theta_s_test = X_test["theta_s"].values.astype(np.float32)
    theta_r_test = X_test["theta_r"].values.astype(np.float32)
    
    print(f"   Training: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Compute ψ50 for each split
    print("\n2. Computing ψ50...")
    
    psi50_train = []
    for i in range(len(X_train)):
        psi50 = find_psi50(psi, y_train[i], theta_s_train[i], theta_r_train[i])
        psi50_train.append(psi50)
    
    psi50_val = []
    for i in range(len(X_val)):
        psi50 = find_psi50(psi, y_val[i], theta_s_val[i], theta_r_val[i])
        psi50_val.append(psi50)
    
    psi50_test = []
    for i in range(len(X_test)):
        psi50 = find_psi50(psi, y_test[i], theta_s_test[i], theta_r_test[i])
        psi50_test.append(psi50)
    
    psi50_train = np.array(psi50_train, dtype=np.float32)
    psi50_val = np.array(psi50_val, dtype=np.float32)
    psi50_test = np.array(psi50_test, dtype=np.float32)
    
    print(f"   Training ψ50: median={np.median(psi50_train):.2f} kPa, "
          f"range=[{np.min(psi50_train):.2f}, {np.max(psi50_train):.2f}] kPa")
    print(f"   Validation ψ50: median={np.median(psi50_val):.2f} kPa, "
          f"range=[{np.min(psi50_val):.2f}, {np.max(psi50_val):.2f}] kPa")
    print(f"   Test ψ50: median={np.median(psi50_test):.2f} kPa, "
          f"range=[{np.min(psi50_test):.2f}, {np.max(psi50_test):.2f}] kPa")
    
    # Check for NaN
    nan_train = np.isnan(psi50_train).sum()
    nan_val = np.isnan(psi50_val).sum()
    nan_test = np.isnan(psi50_test).sum()
    
    if nan_train > 0 or nan_val > 0 or nan_test > 0:
        print(f"\n⚠ Warning: Found NaN values - Train: {nan_train}, Val: {nan_val}, Test: {nan_test}")
        # Replace NaN with median
        median_psi50 = np.nanmedian(np.concatenate([psi50_train, psi50_val, psi50_test]))
        psi50_train = np.where(np.isnan(psi50_train), median_psi50, psi50_train)
        psi50_val = np.where(np.isnan(psi50_val), median_psi50, psi50_val)
        psi50_test = np.where(np.isnan(psi50_test), median_psi50, psi50_test)
        print(f"   Replaced NaN with median: {median_psi50:.2f} kPa")
    
    # Save results
    out_dir = RESULTS_DIR / "vgparamnet"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(out_dir / "psi50_train.npy", psi50_train)
    np.save(out_dir / "psi50_val.npy", psi50_val)
    np.save(out_dir / "psi50_test.npy", psi50_test)
    
    print(f"\n✓ Saved ψ50 arrays to: {out_dir}")
    print(f"  - psi50_train.npy: {len(psi50_train)} values")
    print(f"  - psi50_val.npy: {len(psi50_val)} values")
    print(f"  - psi50_test.npy: {len(psi50_test)} values")


if __name__ == "__main__":
    main()
