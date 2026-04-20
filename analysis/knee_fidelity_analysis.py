#!/usr/bin/env python3
"""
Knee Fidelity Analysis for SWCC Curves
Computes knee-related metrics to quantify sand knee problem:
- ψ_50: suction at which Se = 0.5
- max_slope: maximum |dθ/dlog(ψ)| (knee sharpness)
- Compares across Observed, GB, PINN, VGParamNet, and GB+monotone repair
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_pinn.config_pinn_fixed import DATA_CONFIG, RESULTS_DIR
from baseline_models import BaselineModels


def compute_effective_saturation(theta, theta_s, theta_r):
    """Compute effective saturation Se = (θ - θr)/(θs - θr)"""
    denom = np.maximum(theta_s - theta_r, 1e-6)
    Se = (theta - theta_r) / denom
    return np.clip(Se, 0.0, 1.0)


def find_psi_50(psi, theta, theta_s, theta_r):
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


def compute_max_slope(psi, theta):
    """
    Compute maximum |dθ/dlog(ψ)| (knee sharpness).
    
    Returns:
        max_slope: maximum absolute slope in log-suction space
        max_slope_idx: index where max slope occurs
    """
    log_psi = np.log10(np.maximum(psi, 1e-8))
    dtheta = np.diff(theta)
    dlog_psi = np.diff(log_psi)
    
    # Avoid division by zero
    dlog_psi = np.maximum(np.abs(dlog_psi), 1e-10)
    slope = np.abs(dtheta / dlog_psi)
    
    max_slope_idx = np.argmax(slope)
    max_slope = slope[max_slope_idx]
    
    return max_slope, max_slope_idx


def apply_monotone_repair(theta):
    """
    Apply monotone repair using cumulative minimum (isotonic regression).
    Ensures θ is strictly decreasing.
    
    Args:
        theta: [n_points] water content array
    
    Returns:
        theta_repaired: monotone decreasing version
    """
    theta_repaired = theta.copy()
    
    # Forward pass: ensure no increases
    for i in range(1, len(theta_repaired)):
        if theta_repaired[i] > theta_repaired[i-1]:
            theta_repaired[i] = theta_repaired[i-1]
    
    return theta_repaired


def load_predictions():
    """Load all predictions: Observed, GB, PINN, VGParamNet"""
    # Load test data
    X_test = pd.read_csv(DATA_CONFIG["test_file"])
    y_test_obs = np.load(DATA_CONFIG["y_test_original_file"]).astype(np.float32)
    psi = np.load(DATA_CONFIG["suction_grid_file"]).astype(np.float32)
    
    theta_s = X_test["theta_s"].values.astype(np.float32)
    theta_r = X_test["theta_r"].values.astype(np.float32)
    
    # Load GB predictions (train GB if needed)
    print("   Training/loading GB baseline...")
    baseline = BaselineModels(data_dir="data_processed", output_dir="results_baseline")
    (X_train, X_val, X_test_gb), (y_train, y_val, y_test_gb), suction_grid = baseline.load_data()
    X_train_feat, X_val_feat, X_test_feat, feature_cols = baseline.prepare_features(X_train, X_val, X_test_gb)
    
    gb_models = baseline.train_gradient_boosting(X_train_feat, y_train, X_val_feat, y_val)
    y_gb = baseline.predict_swcc(gb_models, X_test_feat, model_type="gradient_boosting", n_points=y_test_obs.shape[1])
    
    # Load PINN predictions
    print("   Loading PINN model...")
    import tensorflow as tf
    from models.pinn_monotonic import MonotonicPINN
    from models.pinn import PhysicsEncodingLayer
    
    metadata = json.load(open(DATA_CONFIG['metadata_file']))
    pinn_model = MonotonicPINN(
        soil_prop_dim=metadata['n_features'],
        suction_points=metadata['n_swcc_points'],
        physics_units=128,
        hidden_dims=[128, 256, 128, 64]
    )
    
    # Build model
    dummy_soil = tf.random.normal([1, metadata['n_features']])
    dummy_suction = tf.random.normal([1, metadata['n_swcc_points']])
    _ = pinn_model({'soil_props': dummy_soil, 'suction': dummy_suction})
    
    checkpoint_path = RESULTS_DIR / "checkpoints" / "pinn_best_model_fixed.keras"
    saved_model = tf.keras.models.load_model(
        str(checkpoint_path),
        custom_objects={"MonotonicPINN": MonotonicPINN, "PhysicsEncodingLayer": PhysicsEncodingLayer},
        compile=False
    )
    pinn_model.set_weights(saved_model.get_weights())
    
    X_test_features = X_test[metadata['feature_cols']].values.astype(np.float32)
    psi_batch = np.tile(psi[None, :], (len(X_test_features), 1)).astype(np.float32)
    
    y_pinn_norm = pinn_model.predict({
        'soil_props': X_test_features,
        'suction': psi_batch
    }, verbose=0)
    
    # Denormalize PINN predictions
    theta_range = theta_s - theta_r
    y_pinn = theta_r[:, None] + y_pinn_norm * theta_range[:, None]
    
    # Load VGParamNet predictions
    vgparamnet_path = RESULTS_DIR / "vgparamnet" / "theta_vgparamnet_test.npy"
    if vgparamnet_path.exists():
        y_vgnet = np.load(vgparamnet_path).astype(np.float32)
    else:
        y_vgnet = None
    
    # Apply monotone repair to GB
    y_gb_repaired = np.array([apply_monotone_repair(y_gb[i]) for i in range(len(y_gb))])
    
    return {
        'psi': psi,
        'theta_s': theta_s,
        'theta_r': theta_r,
        'Observed': y_test_obs,
        'GB': y_gb,
        'GB_repaired': y_gb_repaired,
        'PINN': y_pinn,
        'VGParamNet': y_vgnet,
    }


def main():
    print("=" * 80)
    print("Knee Fidelity Analysis")
    print("=" * 80)
    
    # Load predictions
    print("\n1. Loading predictions...")
    data = load_predictions()
    psi = data['psi']
    theta_s = data['theta_s']
    theta_r = data['theta_r']
    
    N = len(theta_s)
    print(f"   Test samples: {N}")
    
    # Compute knee metrics for each method
    print("\n2. Computing knee fidelity metrics...")
    
    methods = ['Observed', 'GB', 'GB_repaired', 'PINN']
    if data['VGParamNet'] is not None:
        methods.append('VGParamNet')
    
    results = []
    
    for method_name in methods:
        theta = data[method_name]
        print(f"   Processing {method_name}...")
        
        psi_50_list = []
        max_slope_list = []
        
        for i in range(N):
            # Compute psi_50
            psi_50 = find_psi_50(psi, theta[i], theta_s[i], theta_r[i])
            psi_50_list.append(psi_50)
            
            # Compute max slope
            max_slope, _ = compute_max_slope(psi, theta[i])
            max_slope_list.append(max_slope)
            
            results.append({
                'method': method_name,
                'sample_id': i,
                'psi_50': psi_50,
                'max_slope': max_slope,
            })
    
    df = pd.DataFrame(results)
    
    # Save results
    out_dir = RESULTS_DIR / "knee_fidelity"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_dir / "knee_fidelity_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved results to: {csv_path}")
    
    # Compute summary statistics
    print("\n3. Summary statistics:")
    summary = {}
    for method in methods:
        sub = df[df['method'] == method]
        summary[method] = {
            'psi_50_median': float(sub['psi_50'].median()),
            'psi_50_mean': float(sub['psi_50'].mean()),
            'psi_50_std': float(sub['psi_50'].std()),
            'max_slope_median': float(sub['max_slope'].median()),
            'max_slope_mean': float(sub['max_slope'].mean()),
            'max_slope_std': float(sub['max_slope'].std()),
        }
        print(f"\n   {method}:")
        print(f"     ψ_50: median={summary[method]['psi_50_median']:.2f} kPa, "
              f"mean={summary[method]['psi_50_mean']:.2f} ± {summary[method]['psi_50_std']:.2f} kPa")
        print(f"     max_slope: median={summary[method]['max_slope_median']:.4f}, "
              f"mean={summary[method]['max_slope_mean']:.4f} ± {summary[method]['max_slope_std']:.4f}")
    
    # Save summary
    json_path = out_dir / "knee_fidelity_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved summary to: {json_path}")
    
    # Generate figures
    print("\n4. Generating figures...")
    
    # Figure 1: ψ_50 distribution
    plt.figure(figsize=(10, 6))
    for method in methods:
        sub = df[df['method'] == method]
        plt.hist(sub['psi_50'], bins=30, alpha=0.6, label=method, edgecolor='black', linewidth=0.5)
    plt.xlabel('ψ₅₀ (kPa)', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.legend(fontsize=12)
    plt.title('Distribution of Knee Location (ψ₅₀)', fontsize=16)
    plt.tight_layout()
    fig1_path = out_dir / "psi50_distribution.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"   ✓ Saved: {fig1_path}")
    
    # Figure 2: max_slope distribution
    plt.figure(figsize=(10, 6))
    for method in methods:
        sub = df[df['method'] == method]
        plt.hist(sub['max_slope'], bins=30, alpha=0.6, label=method, edgecolor='black', linewidth=0.5)
    plt.xlabel('Maximum Slope |dθ/dlog(ψ)|', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.legend(fontsize=12)
    plt.title('Distribution of Knee Sharpness (max slope)', fontsize=16)
    plt.tight_layout()
    fig2_path = out_dir / "max_slope_distribution.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"   ✓ Saved: {fig2_path}")
    
    # Figure 3: Boxplot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ψ_50 boxplot
    data_psi50 = [df[df['method'] == m]['psi_50'].values for m in methods]
    ax1.boxplot(data_psi50, labels=methods)
    ax1.set_ylabel('ψ₅₀ (kPa)', fontsize=14)
    ax1.set_title('Knee Location (ψ₅₀)', fontsize=16)
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_yscale('log')
    
    # max_slope boxplot
    data_slope = [df[df['method'] == m]['max_slope'].values for m in methods]
    ax2.boxplot(data_slope, labels=methods)
    ax2.set_ylabel('Maximum Slope', fontsize=14)
    ax2.set_title('Knee Sharpness (max |dθ/dlog(ψ)|)', fontsize=16)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    fig3_path = out_dir / "knee_metrics_comparison.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"   ✓ Saved: {fig3_path}")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"Results directory: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
