#!/usr/bin/env python3
"""
Analyze VGParamNet Training Results
Compares current results with targets and shows improvement opportunities
"""

import sys
from pathlib import Path
import numpy as np
import json

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results_pinn_fixed" / "vgparamnet"

def analyze_results():
    """Analyze current VGParamNet results"""
    print("=" * 80)
    print("VGParamNet Training Results Analysis")
    print("=" * 80)
    
    # Load predictions
    alpha_file = RESULTS_DIR / "alpha_test.npy"
    n_file = RESULTS_DIR / "n_test.npy"
    theta_file = RESULTS_DIR / "theta_vgparamnet_test.npy"
    
    if not all(f.exists() for f in [alpha_file, n_file, theta_file]):
        print("\n⚠ Prediction files not found. Training may not be complete.")
        return
    
    alpha = np.load(alpha_file)
    n = np.load(n_file)
    theta = np.load(theta_file)
    
    print("\n📊 Current Results:")
    print(f"   Alpha (1/kPa):")
    print(f"      Median: {np.median(alpha):.4f}")
    print(f"      Mean: {np.mean(alpha):.4f}")
    print(f"      Range: [{np.min(alpha):.4f}, {np.max(alpha):.4f}]")
    
    print(f"\n   n parameter:")
    print(f"      Median: {np.median(n):.4f}")
    print(f"      Mean: {np.mean(n):.4f}")
    print(f"      Range: [{np.min(n):.4f}, {np.max(n):.4f}]")
    
    print(f"\n   Theta predictions:")
    print(f"      Shape: {theta.shape}")
    print(f"      NaN count: {np.isnan(theta).sum()}")
    print(f"      Range: [{np.nanmin(theta):.4f}, {np.nanmax(theta):.4f}]")
    
    # Compare with targets
    print("\n" + "=" * 80)
    print("Comparison with Observed/Target Values")
    print("=" * 80)
    
    # Observed values (from previous analysis)
    obs_alpha_median = 0.114
    obs_n_median = 1.665
    
    print(f"\n   Alpha comparison:")
    print(f"      Observed median: {obs_alpha_median:.4f} 1/kPa")
    print(f"      VGParamNet median: {np.median(alpha):.4f} 1/kPa")
    alpha_diff = abs(np.median(alpha) - obs_alpha_median) / obs_alpha_median * 100
    if alpha_diff < 10:
        print(f"      ✓ Good match (within {alpha_diff:.1f}%)")
    elif alpha_diff < 50:
        print(f"      ⚠ Moderate difference ({alpha_diff:.1f}% off)")
    else:
        print(f"      ❌ Large difference ({alpha_diff:.1f}% off)")
    
    print(f"\n   n comparison:")
    print(f"      Observed median: {obs_n_median:.4f}")
    print(f"      VGParamNet median: {np.median(n):.4f}")
    n_diff = abs(np.median(n) - obs_n_median) / obs_n_median * 100
    print(f"      Difference: {n_diff:.1f}%")
    
    if np.median(n) < 1.3:
        print(f"      ❌ n is too low (target: ~1.665)")
        print(f"      → ψ50 loss should help improve this")
    elif np.median(n) < 1.5:
        print(f"      ⚠ n is improving but still below target")
        print(f"      → Consider increasing λ_ψ50 to 0.2 or enabling slope loss")
    elif np.median(n) < 1.6:
        print(f"      ✓ n is close to target")
    else:
        print(f"      ✓✓ n matches or exceeds target")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("Recommendations")
    print("=" * 80)
    
    if np.median(n) < 1.4:
        print("""
1. ⚠ n is still significantly below observed (1.215 vs 1.665)
   
   Actions:
   a) Increase ψ50 loss weight: λ_ψ50 = 0.2 (currently 0.1)
   b) Enable slope loss: set use_slope_loss = True
   c) Retrain and monitor n distribution
   
2. Monitor training output for:
   - ψ50 loss decreasing (target: <1.0)
   - n values increasing during training
   - Validation loss improving
""")
    elif np.median(n) < 1.5:
        print("""
1. ✓ n is improving but could be better
   
   Actions:
   a) Slightly increase ψ50 loss: λ_ψ50 = 0.15
   b) Or enable slope loss with λ_slope = 0.05
   
2. Current results are acceptable for initial training
""")
    else:
        print("""
1. ✓ n is in good range
   
   Current training appears successful!
   Consider evaluating on test set and comparing with baseline.
""")
    
    # Training status
    print("\n" + "=" * 80)
    print("Next Steps")
    print("=" * 80)
    print("""
1. If training just completed with ψ50 loss:
   - Check if n improved compared to previous run
   - Run VG fit stability analysis to compare
   - Run knee fidelity analysis to check ψ50 accuracy

2. If n is still low:
   - Retrain with higher λ_ψ50 (0.2) or enable slope loss
   - Monitor training to see n values increase

3. Evaluate final model:
   - Run: python3 analysis/run_vg_fit_stability.py
   - Run: python3 analysis/knee_fidelity_analysis.py
   - Compare with baseline and PINN results
""")

if __name__ == "__main__":
    analyze_results()
