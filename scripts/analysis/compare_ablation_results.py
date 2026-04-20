#!/usr/bin/env python3
"""
Compare results from VGParamNet ablation study (2x2 grid)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results_pinn_fixed" / "vgparamnet"

def load_run_results(run_id):
    """Load results for a specific run"""
    run_dir = RESULTS_DIR / f"run_{run_id}"
    
    if not run_dir.exists():
        return None
    
    results = {'run_id': run_id}
    
    # Load config
    config_file = run_dir / "training_config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        results.update(config)
    else:
        # Fallback: try main directory
        alpha_file = RESULTS_DIR / "alpha_test.npy"
        n_file = RESULTS_DIR / "n_test.npy"
        if alpha_file.exists() and n_file.exists():
            results['lambda_psi50'] = 0.1  # Default
            results['lambda_slope'] = 0.0
            results['use_huber'] = False
            results['use_curriculum'] = False
        else:
            return None
    
    # Load predictions
    alpha_file = run_dir / "alpha_test.npy"
    n_file = run_dir / "n_test.npy"
    theta_file = run_dir / "theta_vgparamnet_test.npy"
    
    if not all(f.exists() for f in [alpha_file, n_file, theta_file]):
        # Fallback to main directory
        alpha_file = RESULTS_DIR / "alpha_test.npy"
        n_file = RESULTS_DIR / "n_test.npy"
        theta_file = RESULTS_DIR / "theta_vgparamnet_test.npy"
    
    if all(f.exists() for f in [alpha_file, n_file, theta_file]):
        results['alpha'] = np.load(alpha_file)
        results['n'] = np.load(n_file)
        results['theta'] = np.load(theta_file)
        results['exists'] = True
    else:
        results['exists'] = False
    
    return results

def compute_knee_metrics(psi, theta, theta_s, theta_r):
    """Compute knee fidelity metrics"""
    from scipy.interpolate import interp1d
    
    def find_psi_50(psi_grid, theta_curve, theta_s, theta_r):
        """Find ψ where Se = 0.5"""
        Se = (theta_curve - theta_r) / (theta_s - theta_r + 1e-8)
        Se = np.clip(Se, 0.0, 1.0)
        
        # Find where Se crosses 0.5
        target_se = 0.5
        if np.any(Se >= target_se) and np.any(Se <= target_se):
            # Interpolate to find exact psi
            try:
                f = interp1d(Se, psi_grid, kind='linear', bounds_error=False, fill_value='extrapolate')
                psi50 = float(f(target_se))
                return max(psi_grid[0], min(psi_grid[-1], psi50))
            except:
                # Fallback: find closest point
                idx = np.argmin(np.abs(Se - target_se))
                return psi_grid[idx]
        else:
            # Extrapolate
            if Se[0] < target_se:
                # Extrapolate to lower psi
                return psi_grid[0] * (Se[0] / target_se) if Se[0] > 0 else psi_grid[0]
            else:
                # Extrapolate to higher psi
                return psi_grid[-1] * (target_se / Se[-1]) if Se[-1] < 1.0 else psi_grid[-1]
    
    def compute_max_slope(psi_grid, theta_curve):
        """Compute maximum slope |dθ/dlog(ψ)|"""
        log_psi = np.log(np.maximum(psi_grid, 1e-6))
        dlog = np.diff(log_psi)
        dtheta = np.diff(theta_curve)
        dtheta_dlog = np.abs(dtheta / (dlog + 1e-8))
        max_slope = np.max(dtheta_dlog) if len(dtheta_dlog) > 0 else 0.0
        max_idx = np.argmax(dtheta_dlog) if len(dtheta_dlog) > 0 else 0
        return max_slope, max_idx
    
    psi50_list = []
    max_slope_list = []
    
    for i in range(len(theta)):
        psi50 = find_psi_50(psi, theta[i], theta_s[i], theta_r[i])
        max_slope, _ = compute_max_slope(psi, theta[i])
        psi50_list.append(psi50)
        max_slope_list.append(max_slope)
    
    return np.array(psi50_list), np.array(max_slope_list)

def main():
    print("=" * 80)
    print("VGParamNet Ablation Study Comparison")
    print("=" * 80)
    
    # Load observed data for comparison
    from training_pinn.config_pinn_fixed import DATA_CONFIG
    import pandas as pd
    
    X_test = pd.read_csv(DATA_CONFIG["test_file"])
    y_test_obs = np.load(DATA_CONFIG["y_test_original_file"]).astype(np.float32)
    psi = np.load(DATA_CONFIG["suction_grid_file"]).astype(np.float32)
    theta_s = X_test["theta_s"].values.astype(np.float32)
    theta_r = X_test["theta_r"].values.astype(np.float32)
    
    # Observed knee metrics
    psi50_obs, max_slope_obs = compute_knee_metrics(psi, y_test_obs, theta_s, theta_r)
    
    # Load all runs
    runs = {}
    for run_id in ['A', 'B', 'C', 'D']:
        results = load_run_results(run_id)
        if results and results.get('exists'):
            runs[run_id] = results
    
    if not runs:
        print("\n⚠ No run results found. Run the ablation study first:")
        print("   bash run_ablation_study.sh")
        return
    
    print(f"\n📊 Found {len(runs)} runs to compare")
    
    # Compare metrics
    comparison = []
    
    for run_id, results in runs.items():
        alpha = results['alpha']
        n = results['n']
        theta = results['theta']
        
        # Compute knee metrics
        psi50_pred, max_slope_pred = compute_knee_metrics(psi, theta, theta_s, theta_r)
        
        # Compute RMSE
        from sklearn.metrics import mean_squared_error
        rmse = np.sqrt(mean_squared_error(y_test_obs.flatten(), theta.flatten()))
        
        comparison.append({
            'run_id': run_id,
            'lambda_psi50': results.get('lambda_psi50', 0.1),
            'lambda_slope': results.get('lambda_slope', 0.0),
            'use_huber': results.get('use_huber', False),
            'use_curriculum': results.get('use_curriculum', False),
            'alpha_median': float(np.median(alpha)),
            'n_median': float(np.median(n)),
            'n_mean': float(np.mean(n)),
            'psi50_median': float(np.median(psi50_pred)),
            'psi50_rmse': float(np.sqrt(np.mean((psi50_pred - psi50_obs)**2))),
            'max_slope_median': float(np.median(max_slope_pred)),
            'rmse': float(rmse),
        })
    
    df = pd.DataFrame(comparison)
    
    # Add observed reference
    print("\n" + "=" * 80)
    print("Comparison Table")
    print("=" * 80)
    print(f"\nObserved (reference):")
    print(f"  ψ₅₀ median: {np.median(psi50_obs):.2f} kPa")
    print(f"  max_slope median: {np.median(max_slope_obs):.4f}")
    print(f"  n (from VG fit): ~1.665")
    print(f"  α (from VG fit): ~0.114 1/kPa")
    
    print(f"\n{'Run':<6} {'λ_ψ50':<8} {'λ_slope':<10} {'n_median':<10} {'ψ₅₀_med':<10} {'max_slope':<12} {'RMSE':<8}")
    print("-" * 80)
    for _, row in df.iterrows():
        print(f"{row['run_id']:<6} {row['lambda_psi50']:<8.2f} {row['lambda_slope']:<10.2f} "
              f"{row['n_median']:<10.4f} {row['psi50_median']:<10.2f} "
              f"{row['max_slope_median']:<12.4f} {row['rmse']:<8.4f}")
    
    # Save comparison
    out_dir = RESULTS_DIR / "ablation_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(out_dir / "comparison_table.csv", index=False)
    print(f"\n✓ Saved comparison to: {out_dir / 'comparison_table.csv'}")
    
    # Generate comparison plots
    print("\n📈 Generating comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: n distribution
    ax = axes[0, 0]
    for run_id, results in runs.items():
        n = results['n']
        ax.hist(n, bins=30, alpha=0.6, label=f"Run {run_id}", edgecolor='black', linewidth=0.5)
    ax.axvline(1.665, color='red', linestyle='--', linewidth=2, label='Observed (1.665)')
    ax.set_xlabel('n parameter', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('n Parameter Distribution', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: ψ₅₀ distribution
    ax = axes[0, 1]
    for run_id, results in runs.items():
        theta = results['theta']
        psi50_pred, _ = compute_knee_metrics(psi, theta, theta_s, theta_r)
        ax.hist(psi50_pred, bins=30, alpha=0.6, label=f"Run {run_id}", edgecolor='black', linewidth=0.5)
    ax.axvline(np.median(psi50_obs), color='red', linestyle='--', linewidth=2, label=f'Observed ({np.median(psi50_obs):.1f})')
    ax.set_xlabel('ψ₅₀ (kPa)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Knee Location (ψ₅₀) Distribution', fontsize=14)
    ax.set_xscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: max_slope distribution
    ax = axes[1, 0]
    for run_id, results in runs.items():
        theta = results['theta']
        _, max_slope_pred = compute_knee_metrics(psi, theta, theta_s, theta_r)
        ax.hist(max_slope_pred, bins=30, alpha=0.6, label=f"Run {run_id}", edgecolor='black', linewidth=0.5)
    ax.axvline(np.median(max_slope_obs), color='red', linestyle='--', linewidth=2, label=f'Observed ({np.median(max_slope_obs):.4f})')
    ax.set_xlabel('Maximum Slope |dθ/dlog(ψ)|', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Knee Sharpness (max slope) Distribution', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary bar chart
    ax = axes[1, 1]
    x = np.arange(len(df))
    width = 0.35
    
    # Normalize metrics for comparison
    n_norm = (df['n_median'] - 1.0) / (2.0 - 1.0)  # Normalize to [0, 1] range
    slope_norm = df['max_slope_median'] / df['max_slope_median'].max()
    
    ax.bar(x - width/2, n_norm, width, label='n (normalized)', alpha=0.7)
    ax.bar(x + width/2, slope_norm, width, label='max_slope (normalized)', alpha=0.7)
    ax.set_xlabel('Run', fontsize=12)
    ax.set_ylabel('Normalized Value', fontsize=12)
    ax.set_title('Normalized Metrics Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(df['run_id'])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = out_dir / "ablation_comparison.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved: {fig_path}")
    
    # Best run recommendation
    print("\n" + "=" * 80)
    print("Best Run Recommendation")
    print("=" * 80)
    
    # Score each run (lower is better for most metrics, but higher n is better)
    df['score'] = (
        abs(df['n_median'] - 1.665) / 1.665 +  # n closer to 1.665
        abs(df['psi50_median'] - np.median(psi50_obs)) / np.median(psi50_obs) +  # ψ₅₀ closer to observed
        (np.median(max_slope_obs) - df['max_slope_median']) / np.median(max_slope_obs) +  # max_slope closer to observed
        df['rmse'] / df['rmse'].max()  # Lower RMSE
    )
    
    best_run = df.loc[df['score'].idxmin()]
    print(f"\n🏆 Best run: {best_run['run_id']}")
    print(f"   Configuration: λ_ψ50={best_run['lambda_psi50']}, λ_slope={best_run['lambda_slope']}")
    print(f"   n median: {best_run['n_median']:.4f} (target: 1.665)")
    print(f"   ψ₅₀ median: {best_run['psi50_median']:.2f} kPa (observed: {np.median(psi50_obs):.2f} kPa)")
    print(f"   max_slope median: {best_run['max_slope_median']:.4f} (observed: {np.median(max_slope_obs):.4f})")
    print(f"   RMSE: {best_run['rmse']:.4f}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
