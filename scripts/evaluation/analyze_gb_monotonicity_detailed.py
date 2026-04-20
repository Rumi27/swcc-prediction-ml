#!/usr/bin/env python3
"""
Comprehensive Gradient Boosting Monotonicity Analysis
- Pass rates under strict and practical tolerances
- Distribution of bump counts per curve
- Maximum bump amplitudes
- Representative failure plots
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from scipy.signal import find_peaks

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_pinn.config_pinn_fixed import DATA_CONFIG
from baseline_models import BaselineModels

# Output directory
OUTPUT_DIR = ROOT_DIR / "results_baseline" / "monotonicity_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("Gradient Boosting Monotonicity Analysis")
print("="*80)

# ============================================================================
# 1. Load Data and Generate GB Predictions
# ============================================================================
print("\n1. Loading data and generating GB predictions...")

# Load test data
X_test = pd.read_csv(DATA_CONFIG["test_file"])
y_test = np.load(DATA_CONFIG["y_test_original_file"]).astype(np.float32)
psi = np.load(DATA_CONFIG["suction_grid_file"]).astype(np.float32)

print(f"   Test samples: {len(X_test)}")
print(f"   Suction grid points: {len(psi)}")

# Train GB and get predictions
print("   Training Gradient Boosting...")
bm = BaselineModels(data_dir=ROOT_DIR / "data_processed", 
                    output_dir=ROOT_DIR / "results_baseline")
(X_train, X_val, X_test_gb), (y_train, y_val, y_test_gb), suction_grid_gb = bm.load_data()
X_train_feat, X_val_feat, X_test_feat, feature_cols = bm.prepare_features(
    X_train, X_val, X_test_gb)

# Train GB
gb_models = bm.train_gradient_boosting(X_train_feat, y_train, X_val_feat, y_val)

# Predict on test set (matching indices)
# Need to align X_test_gb with X_test from DATA_CONFIG
# For simplicity, predict on all test samples
y_gb = bm.predict_swcc(gb_models, X_test_feat, model_type="gradient_boosting", 
                       n_points=len(psi))

# Ensure same length
if len(y_gb) != len(y_test):
    print(f"   ⚠ Warning: GB predictions ({len(y_gb)}) != test data ({len(y_test)})")
    min_len = min(len(y_gb), len(y_test))
    y_gb = y_gb[:min_len]
    y_test = y_test[:min_len]
    X_test = X_test.iloc[:min_len].reset_index(drop=True)

print(f"   ✓ GB predictions generated: {y_gb.shape}")

# ============================================================================
# 2. Monotonicity Analysis with Multiple Tolerances
# ============================================================================
print("\n2. Analyzing monotonicity violations...")

def analyze_monotonicity(theta_curves, psi_grid, tolerance_strict=1e-8, tolerance_practical=1e-4):
    """
    Analyze monotonicity with strict and practical tolerances.
    
    Returns:
        - pass_strict: boolean array (True if monotone under strict tolerance)
        - pass_practical: boolean array (True if monotone under practical tolerance)
        - bump_counts: number of non-monotonic segments per curve
        - bump_amplitudes: maximum amplitude of each bump (increase in theta)
        - bump_locations: indices where bumps occur
    """
    n_samples = len(theta_curves)
    pass_strict = np.ones(n_samples, dtype=bool)
    pass_practical = np.ones(n_samples, dtype=bool)
    bump_counts = np.zeros(n_samples, dtype=int)
    bump_amplitudes = []
    bump_locations = []
    
    for i in range(n_samples):
        theta = theta_curves[i]
        
        # Compute differences: theta[i] - theta[i+1] should be >= 0 for monotone
        diff = theta[:-1] - theta[1:]
        
        # Strict tolerance: any negative difference (even tiny)
        violations_strict = diff < -tolerance_strict
        if np.any(violations_strict):
            pass_strict[i] = False
        
        # Practical tolerance: only "significant" violations
        violations_practical = diff < -tolerance_practical
        if np.any(violations_practical):
            pass_practical[i] = False
        
        # Count bumps (consecutive violations form one bump)
        if np.any(violations_strict):
            # Find groups of consecutive violations
            violation_indices = np.where(violations_strict)[0]
            if len(violation_indices) > 0:
                # Count distinct bump regions
                gaps = np.diff(violation_indices) > 1
                n_bumps = 1 + np.sum(gaps)
                bump_counts[i] = n_bumps
                
                # Compute amplitudes (maximum increase in theta within each bump)
                bump_amps = []
                bump_locs = []
                start_idx = violation_indices[0]
                
                for j in range(len(violation_indices)):
                    if j == len(violation_indices) - 1 or violation_indices[j+1] - violation_indices[j] > 1:
                        # End of current bump
                        end_idx = violation_indices[j]
                        # Find max increase in this region
                        region_diff = diff[start_idx:end_idx+1]
                        max_amp = np.max(-region_diff[region_diff < 0]) if np.any(region_diff < 0) else 0
                        bump_amps.append(max_amp)
                        bump_locs.append((start_idx, end_idx))
                        
                        if j < len(violation_indices) - 1:
                            start_idx = violation_indices[j+1]
                
                bump_amplitudes.append(bump_amps)
                bump_locations.append(bump_locs)
            else:
                bump_amplitudes.append([])
                bump_locations.append([])
        else:
            bump_amplitudes.append([])
            bump_locations.append([])
    
    return pass_strict, pass_practical, bump_counts, bump_amplitudes, bump_locations

# Analyze with multiple tolerance levels
tolerance_levels = {
    'strict': 1e-8,      # Machine precision level
    'practical_1e6': 1e-6,  # Very small practical
    'practical_1e4': 1e-4,  # Moderate practical (typical measurement noise)
    'practical_1e2': 1e-2,  # Large practical (coarse tolerance)
}

results = {}
# First get strict analysis
pass_strict_all, _, bump_counts_strict, bump_amps_strict, bump_locs_strict = analyze_monotonicity(
    y_gb, psi, tolerance_strict=1e-8, tolerance_practical=1e-8)

results['strict'] = {
    'pass_rate': float(np.mean(pass_strict_all)),
    'n_violations': int(np.sum(~pass_strict_all)),
    'n_samples': int(len(y_gb)),
    'mean_bumps': float(np.mean(bump_counts_strict)),
    'median_bumps': float(np.median(bump_counts_strict)),
    'max_bumps': int(np.max(bump_counts_strict)),
    'bump_counts': bump_counts_strict.tolist(),
    'pass_strict': pass_strict_all.tolist(),
}

# Then analyze with practical tolerances
for tol_name, tol_val in tolerance_levels.items():
    if tol_name == 'strict':
        continue  # Already done
    _, pass_practical, _, _, _ = analyze_monotonicity(
        y_gb, psi, tolerance_strict=1e-8, tolerance_practical=tol_val)
    
    results[tol_name] = {
        'pass_rate': float(np.mean(pass_practical)),
        'n_violations': int(np.sum(~pass_practical)),
        'n_samples': int(len(y_gb)),
        'mean_bumps': float(np.mean(bump_counts_strict)),
        'median_bumps': float(np.median(bump_counts_strict)),
        'max_bumps': int(np.max(bump_counts_strict)),
    }

# Also compute maximum amplitudes across all curves
all_max_amplitudes = []
for i in range(len(y_gb)):
    theta = y_gb[i]
    diff = theta[:-1] - theta[1:]
    violations = diff < -1e-8
    if np.any(violations):
        max_amp = np.max(-diff[violations])
        all_max_amplitudes.append(max_amp)

results['amplitude_stats'] = {
    'mean_max_amplitude': float(np.mean(all_max_amplitudes)) if all_max_amplitudes else 0.0,
    'median_max_amplitude': float(np.median(all_max_amplitudes)) if all_max_amplitudes else 0.0,
    'max_amplitude': float(np.max(all_max_amplitudes)) if all_max_amplitudes else 0.0,
    'q90_amplitude': float(np.percentile(all_max_amplitudes, 90)) if all_max_amplitudes else 0.0,
    'q95_amplitude': float(np.percentile(all_max_amplitudes, 95)) if all_max_amplitudes else 0.0,
    'n_curves_with_bumps': len(all_max_amplitudes),
}

# Print summary
print("\n" + "="*80)
print("MONOTONICITY ANALYSIS RESULTS")
print("="*80)
print(f"\nTotal samples analyzed: {len(y_gb)}")
print(f"\nPass rates by tolerance:")
for tol_name, tol_val in tolerance_levels.items():
    r = results[tol_name]
    print(f"  {tol_name:20s} (tol={tol_val:.0e}): {r['pass_rate']*100:6.2f}% "
          f"({r['n_violations']}/{r['n_samples']} violations)")

print(f"\nBump statistics (strict tolerance, 1e-8):")
r_strict = results['strict']
print(f"  Mean bumps per curve:     {r_strict['mean_bumps']:.2f}")
print(f"  Median bumps per curve:   {r_strict['median_bumps']:.1f}")
print(f"  Maximum bumps in curve:  {r_strict['max_bumps']}")

print(f"\nBump amplitude statistics:")
amp_stats = results['amplitude_stats']
print(f"  Mean max amplitude:       {amp_stats['mean_max_amplitude']:.6f}")
print(f"  Median max amplitude:     {amp_stats['median_max_amplitude']:.6f}")
print(f"  Maximum amplitude:        {amp_stats['max_amplitude']:.6f}")
print(f"  90th percentile:         {amp_stats['q90_amplitude']:.6f}")
print(f"  95th percentile:         {amp_stats['q95_amplitude']:.6f}")
print(f"  Curves with bumps:        {amp_stats['n_curves_with_bumps']}/{len(y_gb)}")

# ============================================================================
# 3. Distribution Plots
# ============================================================================
print("\n3. Creating distribution plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16

# (a) Bump count distribution
ax = axes[0, 0]
bump_counts_all = np.array(results['strict']['bump_counts'])
bump_counts_nonzero = bump_counts_all[bump_counts_all > 0]
ax.hist(bump_counts_all, bins=min(50, int(np.max(bump_counts_all)) + 1), 
        edgecolor='black', linewidth=1, alpha=0.7, color='#e74c3c')
ax.axvline(np.mean(bump_counts_all), color='blue', linestyle='--', linewidth=2, 
           label=f'Mean: {np.mean(bump_counts_all):.2f}')
ax.axvline(np.median(bump_counts_all), color='green', linestyle='--', linewidth=2, 
           label=f'Median: {np.median(bump_counts_all):.1f}')
ax.set_xlabel('Number of Bumps per Curve', fontsize=18, labelpad=10)
ax.set_ylabel('Count', fontsize=18, labelpad=10)
ax.set_title('(a) Distribution of Bump Counts', fontsize=18, pad=10)
ax.legend(fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, min(50, np.max(bump_counts_all) + 1))

# (b) Maximum amplitude distribution
ax = axes[0, 1]
if all_max_amplitudes:
    ax.hist(all_max_amplitudes, bins=50, edgecolor='black', linewidth=1, 
            alpha=0.7, color='#f39c12')
    ax.axvline(np.mean(all_max_amplitudes), color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(all_max_amplitudes):.6f}')
    ax.axvline(np.median(all_max_amplitudes), color='green', linestyle='--', linewidth=2,
               label=f'Median: {np.median(all_max_amplitudes):.6f}')
    ax.set_xlabel('Maximum Bump Amplitude Δθ', fontsize=18, labelpad=10)
    ax.set_ylabel('Count', fontsize=18, labelpad=10)
    ax.set_title('(b) Distribution of Maximum Bump Amplitudes', fontsize=18, pad=10)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
else:
    ax.text(0.5, 0.5, 'No bumps detected', ha='center', va='center', fontsize=16)
    ax.set_title('(b) Distribution of Maximum Bump Amplitudes', fontsize=18, pad=10)

# (c) Pass rate by tolerance
ax = axes[1, 0]
tol_names = list(tolerance_levels.keys())
tol_values = [tolerance_levels[k] for k in tol_names]
pass_rates = [results[k]['pass_rate'] * 100 for k in tol_names]
colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
bars = ax.bar(range(len(tol_names)), pass_rates, color=colors, alpha=0.7, 
              edgecolor='black', linewidth=1.5)
ax.set_xticks(range(len(tol_names)))
ax.set_xticklabels([f'{k}\n({v:.0e})' for k, v in zip(tol_names, tol_values)], 
                   fontsize=14)
ax.set_ylabel('Pass Rate (%)', fontsize=18, labelpad=10)
ax.set_title('(c) Monotonicity Pass Rate by Tolerance', fontsize=18, pad=10)
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3, axis='y')
# Add value labels on bars
for i, (bar, rate) in enumerate(zip(bars, pass_rates)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{rate:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

# (d) Cumulative distribution of bump counts
ax = axes[1, 1]
sorted_bumps = np.sort(bump_counts_all)
cumulative = np.arange(1, len(sorted_bumps) + 1) / len(sorted_bumps) * 100
ax.plot(sorted_bumps, cumulative, linewidth=2.5, color='#9b59b6')
ax.axvline(np.median(bump_counts_all), color='green', linestyle='--', linewidth=2,
           label=f'Median: {np.median(bump_counts_all):.1f}')
ax.axvline(np.percentile(bump_counts_all, 90), color='orange', linestyle='--', linewidth=2,
           label=f'90th percentile: {np.percentile(bump_counts_all, 90):.0f}')
ax.set_xlabel('Number of Bumps', fontsize=18, labelpad=10)
ax.set_ylabel('Cumulative Percentage (%)', fontsize=18, labelpad=10)
ax.set_title('(d) Cumulative Distribution of Bump Counts', fontsize=18, pad=10)
ax.legend(fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, min(50, np.max(bump_counts_all) + 1))

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "gb_monotonicity_statistics.png", dpi=300, bbox_inches='tight')
fig.savefig(OUTPUT_DIR / "gb_monotonicity_statistics.pdf", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR / 'gb_monotonicity_statistics.png'}")

# ============================================================================
# 4. Representative Failure Plots
# ============================================================================
print("\n4. Creating representative failure plots...")

# Select representative examples
pass_strict_array = np.array(results['strict']['pass_strict'])
violation_indices = np.where(~pass_strict_array)[0]
bump_counts_array = np.array(results['strict']['bump_counts'])

# Get all_max_amplitudes properly
all_max_amplitudes = []
for i in range(len(y_gb)):
    theta = y_gb[i]
    diff = theta[:-1] - theta[1:]
    violations = diff < -1e-8
    if np.any(violations):
        max_amp = np.max(-diff[violations])
        all_max_amplitudes.append(max_amp)

# Select examples:
# - One with few bumps (1-3)
# - One with moderate bumps (5-10)
# - One with many bumps (>10)
# - One with largest amplitude
examples = []

# Few bumps
few_bumps = violation_indices[bump_counts_array[violation_indices] <= 3]
if len(few_bumps) > 0:
    examples.append(('Few bumps (1-3)', few_bumps[0]))

# Moderate bumps
mod_bumps = violation_indices[(bump_counts_array[violation_indices] > 3) & 
                              (bump_counts_array[violation_indices] <= 10)]
if len(mod_bumps) > 0:
    examples.append(('Moderate bumps (4-10)', mod_bumps[0]))

# Many bumps
many_bumps = violation_indices[bump_counts_array[violation_indices] > 10]
if len(many_bumps) > 0:
    examples.append(('Many bumps (>10)', many_bumps[0]))

# Largest amplitude
if all_max_amplitudes:
    max_amp_idx = violation_indices[np.argmax([all_max_amplitudes[i] 
                                                for i in range(len(violation_indices)) 
                                                if violation_indices[i] < len(all_max_amplitudes)])]
    examples.append(('Largest amplitude', max_amp_idx))

# Fill to 6 examples if needed
while len(examples) < 6 and len(violation_indices) > len(examples):
    remaining = [idx for idx in violation_indices if idx not in [e[1] for e in examples]]
    if remaining:
        examples.append(('Typical violation', remaining[0]))

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (label, sample_idx) in enumerate(examples[:6]):
    ax = axes[idx]
    
    theta_gb = y_gb[sample_idx]
    theta_true = y_test[sample_idx]
    
    # Plot observed
    ax.semilogx(psi, theta_true, 'k-', linewidth=2.5, label='Observed', alpha=0.8)
    
    # Plot GB prediction
    ax.semilogx(psi, theta_gb, 'r--', linewidth=2.5, label='Gradient Boosting', alpha=0.8)
    
    # Highlight non-monotonic regions
    diff = theta_gb[:-1] - theta_gb[1:]
    violations = diff < -1e-8
    if np.any(violations):
        violation_indices_psi = np.where(violations)[0]
        for v_idx in violation_indices_psi:
            ax.axvspan(psi[v_idx], psi[v_idx+1], alpha=0.2, color='red', 
                      label='Non-monotonic' if v_idx == violation_indices_psi[0] else '')
    
    # Add boundaries
    theta_s = X_test.iloc[sample_idx]['theta_s']
    theta_r = X_test.iloc[sample_idx]['theta_r']
    ax.axhline(theta_s, color='g', linestyle=':', alpha=0.5, linewidth=1, label='θ_s')
    ax.axhline(theta_r, color='orange', linestyle=':', alpha=0.5, linewidth=1, label='θ_r')
    
    # Title with statistics
    n_bumps = bump_counts_array[sample_idx]
    max_amp = all_max_amplitudes[sample_idx] if sample_idx < len(all_max_amplitudes) else 0
    title = f"{label}\nSample {sample_idx}: {n_bumps} bumps, max Δθ={max_amp:.4f}"
    ax.set_title(title, fontsize=14, pad=8)
    ax.set_xlabel('Suction ψ (kPa)', fontsize=16, labelpad=8)
    ax.set_ylabel('Water content θ (m³/m³)', fontsize=16, labelpad=8)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=14)

plt.suptitle('Representative Gradient Boosting Monotonicity Violations', 
             fontsize=20, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(OUTPUT_DIR / "gb_representative_failures.png", dpi=300, bbox_inches='tight')
fig.savefig(OUTPUT_DIR / "gb_representative_failures.pdf", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR / 'gb_representative_failures.png'}")

# ============================================================================
# 5. Save Results
# ============================================================================
print("\n5. Saving results...")

# Save detailed JSON
with open(OUTPUT_DIR / "gb_monotonicity_analysis.json", 'w') as f:
    json.dump(results, f, indent=2)

# Create summary table
summary_data = {
    'Tolerance': [],
    'Tolerance Value': [],
    'Pass Rate (%)': [],
    'Violations': [],
    'Mean Bumps': [],
    'Median Bumps': [],
    'Max Bumps': [],
}

for tol_name, tol_val in tolerance_levels.items():
    r = results[tol_name]
    summary_data['Tolerance'].append(tol_name)
    summary_data['Tolerance Value'].append(f'{tol_val:.0e}')
    summary_data['Pass Rate (%)'].append(f"{r['pass_rate']*100:.2f}")
    summary_data['Violations'].append(f"{r['n_violations']}/{r['n_samples']}")
    summary_data['Mean Bumps'].append(f"{r['mean_bumps']:.2f}")
    summary_data['Median Bumps'].append(f"{r['median_bumps']:.1f}")
    summary_data['Max Bumps'].append(f"{r['max_bumps']}")

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(OUTPUT_DIR / "gb_monotonicity_summary.csv", index=False)

# Amplitude summary
amp_summary = pd.DataFrame([results['amplitude_stats']])
amp_summary.to_csv(OUTPUT_DIR / "gb_amplitude_statistics.csv", index=False)

print(f"  ✓ Saved: {OUTPUT_DIR / 'gb_monotonicity_analysis.json'}")
print(f"  ✓ Saved: {OUTPUT_DIR / 'gb_monotonicity_summary.csv'}")
print(f"  ✓ Saved: {OUTPUT_DIR / 'gb_amplitude_statistics.csv'}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nKey Findings:")
print(f"  - Strict tolerance (1e-8): {results['strict']['pass_rate']*100:.2f}% pass rate")
print(f"  - Practical tolerance (1e-4): {results['practical_1e4']['pass_rate']*100:.2f}% pass rate")
print(f"  - Mean bumps per violating curve: {results['strict']['mean_bumps']:.2f}")
print(f"  - Maximum bump amplitude: {results['amplitude_stats']['max_amplitude']:.6f}")
print(f"\nFiles created in: {OUTPUT_DIR}")
