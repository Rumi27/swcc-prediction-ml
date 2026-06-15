#!/usr/bin/env python3
"""
Compute correlation between VGParamNet predicted parameters and observed-fitted VG parameters.
This provides evidence that the model learns meaningful structure from PSD/density.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_pinn.config_pinn_fixed import DATA_CONFIG, RESULTS_DIR
from analysis.vg_fit import fit_vg_alpha_n

def main():
    print("=" * 80)
    print("VG Parameter Correlation Analysis")
    print("=" * 80)
    
    # Load test data
    X_test = pd.read_csv(DATA_CONFIG["test_file"])
    y_test_obs = np.load(DATA_CONFIG["y_test_original_file"]).astype(np.float32)
    psi = np.load(DATA_CONFIG["suction_grid_file"]).astype(np.float32)
    theta_s = X_test["theta_s"].values.astype(np.float32)
    theta_r = X_test["theta_r"].values.astype(np.float32)
    
    # Load VGParamNet Run B predictions
    vgparamnet_dir = RESULTS_DIR / "vgparamnet"
    alpha_pred = np.load(vgparamnet_dir / "alpha_test.npy").astype(np.float32)
    n_pred = np.load(vgparamnet_dir / "n_test.npy").astype(np.float32)
    
    print(f"\nLoaded {len(alpha_pred)} test samples")
    
    # Try to load existing VG fit results first
    vg_fit_results_path = RESULTS_DIR / "vg_fit" / "vg_fit_results.csv"
    
    if vg_fit_results_path.exists():
        print("\nLoading existing VG fit results...")
        df_vg = pd.read_csv(vg_fit_results_path)
        df_obs = df_vg[df_vg["curve_type"] == "Observed"].copy()
        df_obs = df_obs.sort_values("sample_id")
        
        alpha_obs_fitted = df_obs["alpha"].values
        n_obs_fitted = df_obs["n"].values
        fit_success = df_obs["fit_success"].values
        
        print(f"  ✓ Loaded {fit_success.sum()}/{len(fit_success)} successful fits")
    else:
        # Fit VG parameters to observed curves
        print("\nFitting VG parameters to observed curves...")
        print("  (This may take a few minutes...)")
        alpha_obs_fitted = []
        n_obs_fitted = []
        fit_success = []
        
        for i in range(len(y_test_obs)):
            theta_obs = y_test_obs[i]
            ts = theta_s[i]
            tr = theta_r[i]
            
            try:
                alpha_fit, n_fit, rmse = fit_vg_alpha_n(psi, theta_obs, ts, tr)
                if np.isfinite(alpha_fit) and np.isfinite(n_fit) and rmse < 0.1:
                    alpha_obs_fitted.append(alpha_fit)
                    n_obs_fitted.append(n_fit)
                    fit_success.append(True)
                else:
                    alpha_obs_fitted.append(np.nan)
                    n_obs_fitted.append(np.nan)
                    fit_success.append(False)
            except:
                alpha_obs_fitted.append(np.nan)
                n_obs_fitted.append(np.nan)
                fit_success.append(False)
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{len(y_test_obs)} samples...")
        
        alpha_obs_fitted = np.array(alpha_obs_fitted)
        n_obs_fitted = np.array(n_obs_fitted)
        fit_success = np.array(fit_success)
        
        print(f"\n✓ Successfully fitted {fit_success.sum()}/{len(fit_success)} observed curves")
    
    # Filter to successful fits
    mask = fit_success & np.isfinite(alpha_pred) & np.isfinite(n_pred)
    
    if mask.sum() == 0:
        print("\n⚠ No valid pairs found for correlation analysis.")
        print("  This may indicate:")
        print("  1. VG fitting failed for observed curves")
        print("  2. VGParamNet predictions contain NaN values")
        print("  3. Sample ordering mismatch")
        print("\n  Skipping correlation analysis.")
        return
    
    alpha_pred_valid = alpha_pred[mask]
    n_pred_valid = n_pred[mask]
    alpha_obs_valid = alpha_obs_fitted[mask]
    n_obs_valid = n_obs_fitted[mask]
    
    print(f"  Valid pairs for correlation: {mask.sum()}")
    
    # Compute correlations
    print("\n" + "=" * 80)
    print("Correlation Analysis")
    print("=" * 80)
    
    # Alpha correlation
    if len(alpha_pred_valid) > 0:
        r_alpha_pearson, p_alpha_pearson = pearsonr(alpha_pred_valid, alpha_obs_valid)
        r_alpha_spearman, p_alpha_spearman = spearmanr(alpha_pred_valid, alpha_obs_valid)
        r2_alpha = r2_score(alpha_obs_valid, alpha_pred_valid)
        
        print(f"\nα parameter:")
        print(f"  Pearson r: {r_alpha_pearson:.4f} (p={p_alpha_pearson:.2e})")
        print(f"  Spearman ρ: {r_alpha_spearman:.4f} (p={p_alpha_spearman:.2e})")
        print(f"  R²: {r2_alpha:.4f}")
        print(f"  Predicted median: {np.median(alpha_pred_valid):.4f} 1/kPa")
        print(f"  Observed-fitted median: {np.median(alpha_obs_valid):.4f} 1/kPa")
    
    # n correlation
    if len(n_pred_valid) > 0:
        r_n_pearson, p_n_pearson = pearsonr(n_pred_valid, n_obs_valid)
        r_n_spearman, p_n_spearman = spearmanr(n_pred_valid, n_obs_valid)
        r2_n = r2_score(n_obs_valid, n_pred_valid)
        
        print(f"\nn parameter:")
        print(f"  Pearson r: {r_n_pearson:.4f} (p={p_n_pearson:.2e})")
        print(f"  Spearman ρ: {r_n_spearman:.4f} (p={p_n_spearman:.2e})")
        print(f"  R²: {r2_n:.4f}")
        print(f"  Predicted median: {np.median(n_pred_valid):.4f}")
        print(f"  Observed-fitted median: {np.median(n_obs_valid):.4f}")
    
    # Generate scatter plots
    print("\nGenerating correlation plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Alpha scatter
    ax = axes[0]
    ax.scatter(alpha_obs_valid, alpha_pred_valid, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # 1:1 line
    min_val = min(alpha_obs_valid.min(), alpha_pred_valid.min())
    max_val = max(alpha_obs_valid.max(), alpha_pred_valid.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 line')
    
    ax.set_xlabel('Observed-fitted α (1/kPa)', fontsize=12)
    ax.set_ylabel('VGParamNet predicted α (1/kPa)', fontsize=12)
    ax.set_title(f'α Parameter Correlation\nr={r_alpha_pearson:.3f}, R²={r2_alpha:.3f}', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # n scatter
    ax = axes[1]
    ax.scatter(n_obs_valid, n_pred_valid, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # 1:1 line
    min_val = min(n_obs_valid.min(), n_pred_valid.min())
    max_val = max(n_obs_valid.max(), n_pred_valid.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 line')
    
    ax.set_xlabel('Observed-fitted n [-]', fontsize=12)
    ax.set_ylabel('VGParamNet predicted n [-]', fontsize=12)
    ax.set_title(f'n Parameter Correlation\nr={r_n_pearson:.3f}, R²={r2_n:.3f}', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    out_dir = RESULTS_DIR / "vgparamnet" / "correlation_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig_path = out_dir / "vg_parameter_correlation.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {fig_path}")
    
    # Save results
    results = {
        'alpha': {
            'pearson_r': float(r_alpha_pearson),
            'pearson_p': float(p_alpha_pearson),
            'spearman_rho': float(r_alpha_spearman),
            'spearman_p': float(p_alpha_spearman),
            'r2': float(r2_alpha),
            'predicted_median': float(np.median(alpha_pred_valid)),
            'observed_median': float(np.median(alpha_obs_valid)),
            'n_valid': int(mask.sum())
        },
        'n': {
            'pearson_r': float(r_n_pearson),
            'pearson_p': float(p_n_pearson),
            'spearman_rho': float(r_n_spearman),
            'spearman_p': float(p_n_spearman),
            'r2': float(r2_n),
            'predicted_median': float(np.median(n_pred_valid)),
            'observed_median': float(np.median(n_obs_valid)),
            'n_valid': int(mask.sum())
        }
    }
    
    results_path = out_dir / "correlation_results.json"
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved: {results_path}")
    
    # Paper-ready interpretation
    print("\n" + "=" * 80)
    print("Paper-Ready Interpretation")
    print("=" * 80)
    
    if r_alpha_pearson > 0.5:
        print(f"\n✅ Strong α correlation (r={r_alpha_pearson:.3f}):")
        print("   VGParamNet successfully learns air-entry scaling from soil properties.")
    elif r_alpha_pearson > 0.3:
        print(f"\n⚠ Moderate α correlation (r={r_alpha_pearson:.3f}):")
        print("   VGParamNet learns some structure but may need refinement.")
    else:
        print(f"\n❌ Weak α correlation (r={r_alpha_pearson:.3f}):")
        print("   VGParamNet may not be learning meaningful α structure.")
    
    if r_n_pearson > 0.5:
        print(f"\n✅ Strong n correlation (r={r_n_pearson:.3f}):")
        print("   VGParamNet successfully learns knee sharpness from soil properties.")
    elif r_n_pearson > 0.3:
        print(f"\n⚠ Moderate n correlation (r={r_n_pearson:.3f}):")
        print("   VGParamNet learns some structure; n may be harder to infer due to")
        print("   sparse knee sampling and measurement noise.")
    else:
        print(f"\n❌ Weak n correlation (r={r_n_pearson:.3f}):")
        print("   n is harder to infer due to sparse knee sampling and noise.")
        print("   This is expected and can be addressed in future work.")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
