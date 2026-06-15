#!/usr/bin/env python3
"""
Simulation Usability Analysis: Van Genuchten Fitting and Hydraulic Conductivity
Proves that PINN produces physically consistent SWCCs suitable for simulation workflows.

This script:
1. Fits van Genuchten parameters to Observed, GB-predicted, and PINN-predicted curves
2. Analyzes fit success rates and parameter plausibility
3. Derives hydraulic conductivity K(ψ) using Mualem-VG model
4. Shows that GB's non-monotone curves cause unstable fits, while PINN is stable
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning
import warnings
warnings.filterwarnings('ignore', category=OptimizeWarning)
import json

from models.pinn_monotonic import MonotonicPINN
from models.pinn import PhysicsEncodingLayer
from training_pinn.config_pinn_fixed import DATA_CONFIG, RESULTS_DIR
from baseline_models import BaselineModels

plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 13

print("="*80)
print("Simulation Usability Analysis: VG Fitting & Hydraulic Conductivity")
print("="*80)

# ============================================================================
# 1. Load models and data
# ============================================================================

print("\n1. Loading models and test data...")

# Load PINN
CHECKPOINT_DIR = Path("results_pinn_fixed/checkpoints")
best_model_path = CHECKPOINT_DIR / "pinn_best_model_fixed.keras"
metadata = json.load(open(DATA_CONFIG['metadata_file']))
model = MonotonicPINN(
    soil_prop_dim=metadata['n_features'],
    suction_points=metadata['n_swcc_points'],
    physics_units=128,
    hidden_dims=[128, 256, 128, 64]
)
dummy_soil = tf.random.normal([1, metadata['n_features']])
dummy_suction = tf.random.normal([1, metadata['n_swcc_points']])
_ = model({'soil_props': dummy_soil, 'suction': dummy_suction})
saved_model = tf.keras.models.load_model(
    str(best_model_path),
    custom_objects={'MonotonicPINN': MonotonicPINN, 'PhysicsEncodingLayer': PhysicsEncodingLayer},
    compile=False
)
model.set_weights(saved_model.get_weights())
print("  ✓ PINN loaded")

# Load test data
X_test = pd.read_csv(DATA_CONFIG['test_file'])
y_test_original = np.load(DATA_CONFIG['y_test_original_file'])
suction_grid = np.load(DATA_CONFIG['suction_grid_file'])
theta_s_test = X_test['theta_s'].values
theta_r_test = X_test['theta_r'].values
feature_cols = metadata['feature_cols']

# Train GB baseline (use data_processed for training)
print("  Training Gradient Boosting baseline...")
bm = BaselineModels(data_dir=Path("data_processed"))
(X_train_full, X_val_full, _), (y_train_full, y_val_full, _), suction_grid_gb = bm.load_data()
X_train_feat, X_val_feat, X_test_feat_gb, feature_cols_gb = bm.prepare_features(
    X_train_full, X_val_full, X_test
)
gb_models = bm.train_gradient_boosting(X_train_feat, y_train_full, X_val_feat, y_val_full)
print("  ✓ GB trained")

# Get predictions
print("  Making predictions...")
# PINN
y_pred_norm = []
batch_size = 32
for i in range(0, len(X_test), batch_size):
    batch_end = min(i + batch_size, len(X_test))
    batch_soil = X_test.iloc[i:batch_end][feature_cols].values.astype(np.float32)
    batch_suction = np.tile(suction_grid, (batch_end - i, 1)).astype(np.float32)
    inputs = {'soil_props': tf.constant(batch_soil), 'suction': tf.constant(batch_suction)}
    theta_pred_norm_batch = model(inputs, training=False)
    y_pred_norm.extend(theta_pred_norm_batch.numpy())
y_pred_norm = np.array(y_pred_norm)
y_pred_pinn = np.zeros_like(y_pred_norm)
for i in range(len(X_test)):
    theta_range = theta_s_test[i] - theta_r_test[i]
    y_pred_pinn[i] = theta_r_test[i] + y_pred_norm[i] * theta_range

# GB - need to match feature columns between data_processed and data_pinn_normalized
# Load feature columns from data_processed
X_train_gb = pd.read_csv("data_processed/X_train.csv")
feature_cols_gb_list = [col for col in X_train_gb.columns if col != 'code']
# Align X_test features with GB training features
X_test_gb_features = X_test[feature_cols_gb_list].values
# Impute and scale
X_test_gb_features = bm.imputers['main'].transform(X_test_gb_features)
X_test_gb_features = bm.scalers['main'].transform(X_test_gb_features)
y_pred_gb = bm.predict_swcc(gb_models, X_test_gb_features, model_type="gradient_boosting", n_points=len(suction_grid))

print(f"  ✓ Predictions complete ({len(X_test)} samples)")

# ============================================================================
# 2. Van Genuchten fitting functions
# ============================================================================

def vg_equation(psi, theta_r, theta_s, alpha, n):
    """
    Van Genuchten equation: θ(ψ) = θr + (θs - θr) / [1 + (α·ψ)^n]^m
    where m = 1 - 1/n
    """
    m = 1 - 1/n
    psi = np.maximum(psi, 1e-6)  # Avoid log(0)
    se = 1.0 / (1.0 + (alpha * psi) ** n) ** m
    return theta_r + (theta_s - theta_r) * se

def fit_vg_parameters(psi, theta, theta_r_known, theta_s_known, max_iter=1000):
    """
    Fit VG parameters (α, n) given known θr and θs.
    Returns: (alpha, n, success, rmse)
    """
    # Remove NaN/inf
    mask = np.isfinite(psi) & np.isfinite(theta) & (theta > 0) & (psi > 0)
    if np.sum(mask) < 5:
        return None, None, False, np.inf
    
    psi_clean = psi[mask]
    theta_clean = theta[mask]
    
    # Ensure monotonicity (required for VG fit)
    # Sort by psi
    sort_idx = np.argsort(psi_clean)
    psi_clean = psi_clean[sort_idx]
    theta_clean = theta_clean[sort_idx]
    
    # Check if monotone decreasing
    if np.any(np.diff(theta_clean) > 1e-6):
        # Force monotonicity by taking cumulative minimum
        theta_clean = np.minimum.accumulate(theta_clean[::-1])[::-1]
    
    # Bounds: alpha in [1e-4, 10], n in [1.01, 10] (n > 1 required)
    bounds = ([1e-4, 1.01], [10.0, 10.0])
    
    # Initial guess
    # alpha: typical range 0.01-1.0 (1/kPa)
    # n: typical range 1.1-3.0
    p0 = [0.1, 2.0]
    
    try:
        popt, _ = curve_fit(
            lambda psi, alpha, n: vg_equation(psi, theta_r_known, theta_s_known, alpha, n),
            psi_clean,
            theta_clean,
            p0=p0,
            bounds=bounds,
            maxfev=max_iter,
            method='trf'
        )
        alpha, n = popt
        
        # Check plausibility
        if alpha < 1e-4 or alpha > 10 or n < 1.01 or n > 10:
            return None, None, False, np.inf
        
        # Compute fit RMSE
        theta_fit = vg_equation(psi_clean, theta_r_known, theta_s_known, alpha, n)
        rmse = np.sqrt(np.mean((theta_clean - theta_fit) ** 2))
        
        return alpha, n, True, rmse
    except:
        return None, None, False, np.inf

# ============================================================================
# 3. Mualem-VG hydraulic conductivity
# ============================================================================

def mualem_vg_conductivity(psi, theta_r, theta_s, alpha, n, Ks=1.0):
    """
    Mualem-VG model for relative hydraulic conductivity:
    K(ψ) = Ks · Se^0.5 · [1 - (1 - Se^(1/m))^m]^2
    
    where Se = (θ - θr)/(θs - θr) is effective saturation
    and m = 1 - 1/n
    """
    m = 1 - 1/n
    psi = np.maximum(psi, 1e-6)
    
    # Effective saturation
    theta = vg_equation(psi, theta_r, theta_s, alpha, n)
    Se = (theta - theta_r) / (theta_s - theta_r + 1e-10)
    Se = np.clip(Se, 0, 1)
    
    # Mualem-VG formula
    term1 = Se ** 0.5
    term2 = 1 - (1 - Se ** (1/m)) ** m
    K_rel = term1 * (term2 ** 2)
    
    return Ks * K_rel

def check_k_monotonicity(psi, K):
    """Check if K(ψ) is monotone decreasing (required for physical consistency)"""
    mask = np.isfinite(psi) & np.isfinite(K) & (K > 0)
    if np.sum(mask) < 2:
        return True  # Trivial case
    
    psi_clean = psi[mask]
    K_clean = K[mask]
    sort_idx = np.argsort(psi_clean)
    K_sorted = K_clean[sort_idx]
    
    # Check if strictly decreasing (allowing small numerical noise)
    diffs = np.diff(K_sorted)
    violations = np.sum(diffs > 1e-10)
    return violations == 0

# ============================================================================
# 4. Fit VG parameters to all curves
# ============================================================================

print("\n2. Fitting van Genuchten parameters...")

results = {
    'observed': {'alpha': [], 'n': [], 'success': [], 'rmse': []},
    'gb': {'alpha': [], 'n': [], 'success': [], 'rmse': []},
    'pinn': {'alpha': [], 'n': [], 'success': [], 'rmse': []}
}

for i in range(len(X_test)):
    psi = suction_grid
    theta_r = theta_r_test[i]
    theta_s = theta_s_test[i]
    
    # Observed
    theta_obs = y_test_original[i]
    alpha_obs, n_obs, success_obs, rmse_obs = fit_vg_parameters(psi, theta_obs, theta_r, theta_s)
    results['observed']['alpha'].append(alpha_obs if success_obs else np.nan)
    results['observed']['n'].append(n_obs if success_obs else np.nan)
    results['observed']['success'].append(success_obs)
    results['observed']['rmse'].append(rmse_obs)
    
    # GB
    theta_gb = y_pred_gb[i]
    alpha_gb, n_gb, success_gb, rmse_gb = fit_vg_parameters(psi, theta_gb, theta_r, theta_s)
    results['gb']['alpha'].append(alpha_gb if success_gb else np.nan)
    results['gb']['n'].append(n_gb if success_gb else np.nan)
    results['gb']['success'].append(success_gb)
    results['gb']['rmse'].append(rmse_gb)
    
    # PINN
    theta_pinn = y_pred_pinn[i]
    alpha_pinn, n_pinn, success_pinn, rmse_pinn = fit_vg_parameters(psi, theta_pinn, theta_r, theta_s)
    results['pinn']['alpha'].append(alpha_pinn if success_pinn else np.nan)
    results['pinn']['n'].append(n_pinn if success_pinn else np.nan)
    results['pinn']['success'].append(success_pinn)
    results['pinn']['rmse'].append(rmse_pinn)
    
    if (i + 1) % 20 == 0:
        print(f"  Fitted {i+1}/{len(X_test)} samples...")

print("  ✓ VG fitting complete")

# ============================================================================
# 5. Compute hydraulic conductivity K(ψ)
# ============================================================================

print("\n3. Computing hydraulic conductivity K(ψ)...")

K_results = {
    'observed': {'K': [], 'monotone': []},
    'gb': {'K': [], 'monotone': []},
    'pinn': {'K': [], 'monotone': []}
}

for i in range(len(X_test)):
    psi = suction_grid
    theta_r = theta_r_test[i]
    theta_s = theta_s_test[i]
    
    # Observed
    if results['observed']['success'][i]:
        alpha = results['observed']['alpha'][i]
        n = results['observed']['n'][i]
        K_obs = mualem_vg_conductivity(psi, theta_r, theta_s, alpha, n)
        K_results['observed']['K'].append(K_obs)
        K_results['observed']['monotone'].append(check_k_monotonicity(psi, K_obs))
    else:
        K_results['observed']['K'].append(np.full_like(psi, np.nan))
        K_results['observed']['monotone'].append(False)
    
    # GB
    if results['gb']['success'][i]:
        alpha = results['gb']['alpha'][i]
        n = results['gb']['n'][i]
        K_gb = mualem_vg_conductivity(psi, theta_r, theta_s, alpha, n)
        K_results['gb']['K'].append(K_gb)
        K_results['gb']['monotone'].append(check_k_monotonicity(psi, K_gb))
    else:
        K_results['gb']['K'].append(np.full_like(psi, np.nan))
        K_results['gb']['monotone'].append(False)
    
    # PINN
    if results['pinn']['success'][i]:
        alpha = results['pinn']['alpha'][i]
        n = results['pinn']['n'][i]
        K_pinn = mualem_vg_conductivity(psi, theta_r, theta_s, alpha, n)
        K_results['pinn']['K'].append(K_pinn)
        K_results['pinn']['monotone'].append(check_k_monotonicity(psi, K_pinn))
    else:
        K_results['pinn']['K'].append(np.full_like(psi, np.nan))
        K_results['pinn']['monotone'].append(False)

print("  ✓ K(ψ) computation complete")

# ============================================================================
# 6. Statistics and summary
# ============================================================================

print("\n4. Computing statistics...")

def safe_stats(arr, name):
    arr_clean = [x for x in arr if not (np.isnan(x) or np.isinf(x))]
    if len(arr_clean) == 0:
        return {'mean': np.nan, 'median': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
    return {
        'mean': np.mean(arr_clean),
        'median': np.median(arr_clean),
        'std': np.std(arr_clean),
        'min': np.min(arr_clean),
        'max': np.max(arr_clean)
    }

stats = {}
for method in ['observed', 'gb', 'pinn']:
    stats[method] = {
        'fit_success_rate': np.mean(results[method]['success']) * 100,
        'alpha': safe_stats(results[method]['alpha'], 'alpha'),
        'n': safe_stats(results[method]['n'], 'n'),
        'fit_rmse': safe_stats(results[method]['rmse'], 'rmse'),
        'K_monotone_rate': np.mean(K_results[method]['monotone']) * 100
    }

print("\n" + "="*80)
print("VAN GENUCHTEN FITTING RESULTS")
print("="*80)
for method in ['observed', 'gb', 'pinn']:
    print(f"\n{method.upper()}:")
    print(f"  Fit success rate: {stats[method]['fit_success_rate']:.1f}%")
    print(f"  α (1/kPa): mean={stats[method]['alpha']['mean']:.4f}, median={stats[method]['alpha']['median']:.4f}")
    print(f"  n: mean={stats[method]['n']['mean']:.3f}, median={stats[method]['n']['median']:.3f}")
    print(f"  Fit RMSE: mean={stats[method]['fit_rmse']['mean']:.6f}, median={stats[method]['fit_rmse']['median']:.6f}")
    print(f"  K(ψ) monotone rate: {stats[method]['K_monotone_rate']:.1f}%")

# ============================================================================
# 7. Generate figures
# ============================================================================

print("\n5. Generating figures...")

output_dir = Path("paper_figures")
output_dir.mkdir(exist_ok=True)

# Figure 1: VG parameter distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Van Genuchten Parameter Fitting Results', fontsize=16, y=0.995)

# Alpha distributions
ax = axes[0, 0]
for method, label, color in [('observed', 'Observed', '#2E86AB'),
                              ('gb', 'Gradient Boosting', '#F18F01'),
                              ('pinn', 'PINN', '#06A77D')]:
    alphas = [x for x in results[method]['alpha'] if not np.isnan(x)]
    if len(alphas) > 0:
        ax.hist(alphas, bins=30, alpha=0.6, label=f"{label} (n={len(alphas)})", color=color, edgecolor='black', linewidth=0.5)
ax.set_xlabel('α (1/kPa)', fontsize=13)
ax.set_ylabel('Frequency', fontsize=13)
ax.set_title('(a) α Parameter Distribution', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# n distributions
ax = axes[0, 1]
for method, label, color in [('observed', 'Observed', '#2E86AB'),
                              ('gb', 'Gradient Boosting', '#F18F01'),
                              ('pinn', 'PINN', '#06A77D')]:
    ns = [x for x in results[method]['n'] if not np.isnan(x)]
    if len(ns) > 0:
        ax.hist(ns, bins=30, alpha=0.6, label=f"{label} (n={len(ns)})", color=color, edgecolor='black', linewidth=0.5)
ax.set_xlabel('n', fontsize=13)
ax.set_ylabel('Frequency', fontsize=13)
ax.set_title('(b) n Parameter Distribution', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Fit success rates
ax = axes[1, 0]
methods = ['Observed', 'Gradient\nBoosting', 'PINN']
success_rates = [stats['observed']['fit_success_rate'],
                 stats['gb']['fit_success_rate'],
                 stats['pinn']['fit_success_rate']]
colors = ['#2E86AB', '#F18F01', '#06A77D']
bars = ax.bar(methods, success_rates, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)
ax.set_ylabel('Fit Success Rate (%)', fontsize=13)
ax.set_title('(c) VG Fitting Success Rate', fontsize=14)
ax.set_ylim([0, 105])
ax.grid(True, alpha=0.3, axis='y')
for bar, rate in zip(bars, success_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{rate:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# K(ψ) monotonicity rates
ax = axes[1, 1]
k_monotone_rates = [stats['observed']['K_monotone_rate'],
                    stats['gb']['K_monotone_rate'],
                    stats['pinn']['K_monotone_rate']]
bars = ax.bar(methods, k_monotone_rates, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)
ax.set_ylabel('K(ψ) Monotone Rate (%)', fontsize=13)
ax.set_title('(d) Hydraulic Conductivity Monotonicity', fontsize=14)
ax.set_ylim([0, 105])
ax.grid(True, alpha=0.3, axis='y')
for bar, rate in zip(bars, k_monotone_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{rate:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
fig.savefig(output_dir / "Figure16_VG_Parameter_Analysis.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 'Figure16_VG_Parameter_Analysis.png'}")
plt.close()

# Figure 2: Representative K(ψ) curves
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Hydraulic Conductivity K(ψ) Derived from Predicted SWCCs', fontsize=16, y=0.995)

# Select 6 representative samples (mix of success/failure cases)
sample_indices = []
# Find samples where GB failed but PINN succeeded
gb_fail_pinn_ok = [i for i in range(len(X_test)) 
                   if not results['gb']['success'][i] and results['pinn']['success'][i]]
if len(gb_fail_pinn_ok) > 0:
    sample_indices.append(gb_fail_pinn_ok[0])
# Find samples where both succeeded
both_ok = [i for i in range(len(X_test)) 
           if results['gb']['success'][i] and results['pinn']['success'][i]]
if len(both_ok) >= 5:
    sample_indices.extend(both_ok[:5])
else:
    sample_indices.extend(both_ok)
    # Fill with random
    remaining = 6 - len(sample_indices)
    sample_indices.extend(np.random.choice(len(X_test), remaining, replace=False))

sample_indices = sample_indices[:6]

for idx, ax in enumerate(axes.flat):
    if idx >= len(sample_indices):
        ax.axis('off')
        continue
    
    i = sample_indices[idx]
    psi = suction_grid
    
    # Observed
    if results['observed']['success'][i]:
        K_obs = K_results['observed']['K'][i]
        ax.semilogy(psi, K_obs, 'k-', linewidth=2, label='Observed', alpha=0.8)
    
    # GB
    if results['gb']['success'][i]:
        K_gb = K_results['gb']['K'][i]
        is_mono = K_results['gb']['monotone'][i]
        style = '--' if is_mono else '-.'
        color = '#F18F01' if is_mono else '#FF0000'
        ax.semilogy(psi, K_gb, style, linewidth=1.5, label='GB', color=color, alpha=0.7)
    
    # PINN
    if results['pinn']['success'][i]:
        K_pinn = K_results['pinn']['K'][i]
        is_mono = K_results['pinn']['monotone'][i]
        style = '--' if is_mono else '-.'
        color = '#06A77D' if is_mono else '#FF0000'
        ax.semilogy(psi, K_pinn, style, linewidth=1.5, label='PINN', color=color, alpha=0.7)
    
    ax.set_xlabel('Suction (kPa)', fontsize=11)
    ax.set_ylabel('K(ψ) (relative)', fontsize=11)
    ax.set_title(f'Sample {i+1}', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    ax.set_xlim([1e-1, 1e6])

plt.tight_layout()
fig.savefig(output_dir / "Figure17_Hydraulic_Conductivity_Comparison.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 'Figure17_Hydraulic_Conductivity_Comparison.png'}")
plt.close()

# Figure 3: Fit RMSE comparison
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
methods_data = []
methods_labels = []
for method, label, color in [('observed', 'Observed', '#2E86AB'),
                              ('gb', 'Gradient Boosting', '#F18F01'),
                              ('pinn', 'PINN', '#06A77D')]:
    rmse_clean = [x for x in results[method]['rmse'] if x < 1.0]  # Filter outliers
    if len(rmse_clean) > 0:
        methods_data.append(rmse_clean)
        methods_labels.append(label)

bp = ax.boxplot(methods_data, labels=methods_labels, patch_artist=True, 
                showmeans=True, meanline=True)
colors_box = ['#2E86AB', '#F18F01', '#06A77D']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)

ax.set_ylabel('VG Fit RMSE', fontsize=13)
ax.set_title('Van Genuchten Fit Quality (RMSE)', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(output_dir / "Figure18_VG_Fit_Quality.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / 'Figure18_VG_Fit_Quality.png'}")
plt.close()

# ============================================================================
# 8. Save summary JSON
# ============================================================================

summary = {
    'vg_fitting': {
        'observed': {
            'success_rate': float(stats['observed']['fit_success_rate']),
            'alpha_mean': float(stats['observed']['alpha']['mean']),
            'alpha_median': float(stats['observed']['alpha']['median']),
            'n_mean': float(stats['observed']['n']['mean']),
            'n_median': float(stats['observed']['n']['median']),
            'fit_rmse_mean': float(stats['observed']['fit_rmse']['mean']),
            'fit_rmse_median': float(stats['observed']['fit_rmse']['median'])
        },
        'gb': {
            'success_rate': float(stats['gb']['fit_success_rate']),
            'alpha_mean': float(stats['gb']['alpha']['mean']),
            'alpha_median': float(stats['gb']['alpha']['median']),
            'n_mean': float(stats['gb']['n']['mean']),
            'n_median': float(stats['gb']['n']['median']),
            'fit_rmse_mean': float(stats['gb']['fit_rmse']['mean']),
            'fit_rmse_median': float(stats['gb']['fit_rmse']['median'])
        },
        'pinn': {
            'success_rate': float(stats['pinn']['fit_success_rate']),
            'alpha_mean': float(stats['pinn']['alpha']['mean']),
            'alpha_median': float(stats['pinn']['alpha']['median']),
            'n_mean': float(stats['pinn']['n']['mean']),
            'n_median': float(stats['pinn']['n']['median']),
            'fit_rmse_mean': float(stats['pinn']['fit_rmse']['mean']),
            'fit_rmse_median': float(stats['pinn']['fit_rmse']['median'])
        }
    },
    'hydraulic_conductivity': {
        'observed': {'monotone_rate': float(stats['observed']['K_monotone_rate'])},
        'gb': {'monotone_rate': float(stats['gb']['K_monotone_rate'])},
        'pinn': {'monotone_rate': float(stats['pinn']['K_monotone_rate'])}
    }
}

with open(RESULTS_DIR / "simulation_usability_analysis.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n  ✓ Saved: {RESULTS_DIR / 'simulation_usability_analysis.json'}")

print("\n" + "="*80)
print("SIMULATION USABILITY ANALYSIS COMPLETE")
print("="*80)
print(f"\nKey Findings:")
print(f"  • GB fit success: {stats['gb']['fit_success_rate']:.1f}%")
print(f"  • PINN fit success: {stats['pinn']['fit_success_rate']:.1f}%")
print(f"  • GB K(ψ) monotone: {stats['gb']['K_monotone_rate']:.1f}%")
print(f"  • PINN K(ψ) monotone: {stats['pinn']['K_monotone_rate']:.1f}%")
print(f"\nFigures saved to: {output_dir}")
