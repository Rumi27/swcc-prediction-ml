#!/usr/bin/env python3
"""
Filter Valid Synthetic Data
Removes NaN/Inf values and enforces physics constraints
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Paths
GENERATED_DIR = Path("results_gan/generated_data")
OUTPUT_DIR = Path("results_gan/generated_data_filtered")

def filter_valid_curves():
    """Filter valid synthetic curves"""
    print("="*80)
    print("Filtering Valid Synthetic Data")
    print("="*80)
    
    # Load synthetic data
    print("\nLoading synthetic data...")
    curves = np.load(GENERATED_DIR / "synthetic_swcc_curves.npy")
    props_df = pd.read_csv(GENERATED_DIR / "synthetic_soil_properties.csv")
    suction_grid = np.load(GENERATED_DIR / "suction_grid.npy")
    
    print(f"  Original: {len(curves)} curves")
    
    # Filter 1: Remove NaN/Inf
    print("\nFilter 1: Removing NaN/Inf values...")
    has_nan = np.any(np.isnan(curves), axis=1)
    has_inf = np.any(np.isinf(curves), axis=1)
    valid_mask = ~(has_nan | has_inf)
    
    curves_valid = curves[valid_mask]
    props_valid = props_df.iloc[valid_mask].reset_index(drop=True)
    
    print(f"  After NaN/Inf filter: {len(curves_valid)} curves ({valid_mask.sum()/len(curves)*100:.1f}%)")
    
    if len(curves_valid) == 0:
        print("\n❌ ERROR: No valid curves after NaN/Inf filtering!")
        return
    
    # Filter 2: Boundary constraints
    print("\nFilter 2: Applying boundary constraints...")
    theta_s = props_valid['theta_s'].values
    theta_r = props_valid['theta_r'].values
    
    # Check boundaries
    theta_s_2d = theta_s.reshape(-1, 1)
    theta_r_2d = theta_r.reshape(-1, 1)
    
    within_bounds = np.all(
        (curves_valid >= theta_r_2d) & (curves_valid <= theta_s_2d),
        axis=1
    )
    
    curves_bounded = curves_valid[within_bounds]
    props_bounded = props_valid[within_bounds].reset_index(drop=True)
    theta_s_bounded = theta_s[within_bounds]
    theta_r_bounded = theta_r[within_bounds]
    
    print(f"  After boundary filter: {len(curves_bounded)} curves ({within_bounds.sum()/len(curves_valid)*100:.1f}%)")
    
    if len(curves_bounded) == 0:
        print("\n⚠ WARNING: No curves within boundaries. Using NaN-filtered curves.")
        curves_bounded = curves_valid
        props_bounded = props_valid
        theta_s_bounded = theta_s
        theta_r_bounded = theta_r
    
    # Filter 3: Monotonicity (enforce post-processing)
    print("\nFilter 3: Enforcing monotonicity...")
    curves_mono = enforce_monotonicity(curves_bounded.copy())
    
    # Final validation
    print("\nFinal validation...")
    final_valid = validate_curves(curves_mono, theta_s_bounded, theta_r_bounded)
    
    print(f"\n✓ Final valid curves: {len(curves_mono)} ({len(curves_mono)/len(curves)*100:.1f}% of original)")
    print(f"  Boundary satisfaction: {final_valid['boundary_rate']*100:.1f}%")
    print(f"  Monotonicity: {final_valid['monotonicity_rate']*100:.1f}%")
    print(f"  NaN rate: {final_valid['nan_rate']*100:.1f}%")
    
    # Save filtered data
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving filtered data to {OUTPUT_DIR}...")
    np.save(OUTPUT_DIR / "synthetic_swcc_curves_filtered.npy", curves_mono)
    props_bounded.to_csv(OUTPUT_DIR / "synthetic_soil_properties_filtered.csv", index=False)
    np.save(OUTPUT_DIR / "suction_grid.npy", suction_grid)
    
    # Save metadata
    metadata = {
        'original_count': int(len(curves)),
        'filtered_count': int(len(curves_mono)),
        'retention_rate': float(len(curves_mono) / len(curves)),
        'boundary_satisfaction': float(final_valid['boundary_rate']),
        'monotonicity_rate': float(final_valid['monotonicity_rate']),
        'nan_rate': float(final_valid['nan_rate'])
    }
    
    import json
    with open(OUTPUT_DIR / "filtering_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✓ Filtered data saved")
    
    return curves_mono, props_bounded


def enforce_monotonicity(curves):
    """Enforce monotonicity (decreasing) on curves"""
    # For each curve, ensure decreasing: θ[i] >= θ[i+1]
    for i in range(len(curves)):
        for j in range(1, len(curves[i])):
            if curves[i, j] > curves[i, j-1]:
                curves[i, j] = curves[i, j-1]
    return curves


def validate_curves(curves, theta_s, theta_r):
    """Validate curves"""
    # Check NaN
    has_nan = np.any(np.isnan(curves), axis=1)
    nan_rate = np.mean(has_nan)
    
    # Check boundaries
    theta_s_2d = theta_s.reshape(-1, 1) if len(theta_s.shape) == 1 else theta_s
    theta_r_2d = theta_r.reshape(-1, 1) if len(theta_r.shape) == 1 else theta_r
    within_bounds = np.all(
        (curves >= theta_r_2d) & (curves <= theta_s_2d),
        axis=1
    )
    bound_rate = np.mean(within_bounds)
    
    # Check monotonicity
    diff = curves[:, :-1] - curves[:, 1:]
    is_monotonic = np.all(diff >= -1e-6, axis=1)
    mono_rate = np.mean(is_monotonic)
    
    return {
        'nan_rate': nan_rate,
        'boundary_rate': bound_rate,
        'monotonicity_rate': mono_rate
    }


if __name__ == "__main__":
    filter_valid_curves()
