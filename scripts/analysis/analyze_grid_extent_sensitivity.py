#!/usr/bin/env python3
"""
Analyze suction grid extent and resolution sensitivity:
1. Why 10^6 kPa was chosen
2. How often interpolation/extrapolation extends beyond measured points
3. Impact of far-dry tail on RMSE and knee metrics
4. Sensitivity to grid extent and resolution
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import json

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_pinn.config_pinn_fixed import DATA_CONFIG

# Output directory
OUTPUT_DIR = ROOT_DIR / "results_analysis" / "grid_sensitivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("Suction Grid Extent and Resolution Sensitivity Analysis")
print("="*80)

# ============================================================================
# 1. Load Data and Original Measurements
# ============================================================================
print("\n1. Loading data and original measurements...")

# Load processed data
X_test = pd.read_csv(DATA_CONFIG["test_file"])
y_test = np.load(DATA_CONFIG["y_test_original_file"]).astype(np.float32)
psi_grid = np.load(DATA_CONFIG["suction_grid_file"]).astype(np.float32)

print(f"   Test samples: {len(X_test)}")
print(f"   Current grid: {len(psi_grid)} points, range: {psi_grid.min():.2f} - {psi_grid.max():.2e} kPa")

# Load original UNSODA measurements to get actual measured ranges
# We need to access the raw data
data_dir = ROOT_DIR.parent / "data_processed"  # Try this path first
if not data_dir.exists():
    data_dir = ROOT_DIR / "data_processed"

# Try to load raw UNSODA data
unsoda_paths = [
    ROOT_DIR.parent / "data" / "UNSODA" / "UNSODA2.0.xlsx",
    ROOT_DIR / "data" / "UNSODA" / "UNSODA2.0.xlsx",
    ROOT_DIR.parent.parent / "data" / "UNSODA" / "UNSODA2.0.xlsx",
]

unsoda_path = None
for path in unsoda_paths:
    if path.exists():
        unsoda_path = path
        break

if unsoda_path and unsoda_path.exists():
    print(f"   Loading raw UNSODA data from {unsoda_path}")
    try:
        import openpyxl
        swcc_df = pd.read_excel(unsoda_path, sheet_name='SWCC')
        swcc_df['suction_kPa'] = swcc_df['preshead'] * 0.0981  # Convert cm to kPa
        
        # Get measured ranges per sample
        measured_ranges = {}
        for code in swcc_df['code'].unique():
            sample_data = swcc_df[swcc_df['code'] == code]
            valid_data = sample_data[sample_data['suction_kPa'] > 0]
            if len(valid_data) > 0:
                measured_ranges[code] = {
                    'min': valid_data['suction_kPa'].min(),
                    'max': valid_data['suction_kPa'].max(),
                    'n_points': len(valid_data)
                }
        print(f"   ✓ Loaded measured ranges for {len(measured_ranges)} samples")
    except Exception as e:
        print(f"   ⚠ Could not load raw UNSODA data: {e}")
        measured_ranges = None
else:
    print("   ⚠ Raw UNSODA data not found, will estimate from interpolated curves")
    measured_ranges = None

# ============================================================================
# 2. Analyze Extrapolation Frequency
# ============================================================================
print("\n2. Analyzing extrapolation frequency...")

# For each test sample, determine which grid points are extrapolated
# We'll use the interpolated curves and estimate measured range from curve shape
extrapolation_stats = {
    'wet_extrapolation': [],  # Points < min measured
    'dry_extrapolation': [],   # Points > max measured
    'interpolation_only': [],  # Points within measured range
    'measured_min': [],
    'measured_max': [],
}

# Estimate measured range from curve characteristics
# Typically, measured data covers the transition region, not the extreme tails
for i in range(len(y_test)):
    theta_curve = y_test[i]
    
    # Find where curve is most active (largest changes)
    # This approximates the measured range
    dtheta = np.abs(np.diff(theta_curve))
    
    # Find active region (where dtheta > threshold)
    threshold = np.percentile(dtheta[dtheta > 0], 10)  # 10th percentile of non-zero changes
    active_mask = dtheta > threshold
    
    if np.sum(active_mask) > 0:
        active_indices = np.where(active_mask)[0]
        # Extend slightly beyond active region
        min_idx = max(0, active_indices[0] - 2)
        max_idx = min(len(psi_grid) - 1, active_indices[-1] + 2)
        
        measured_min = psi_grid[min_idx]
        measured_max = psi_grid[max_idx]
    else:
        # Fallback: use middle 80% of curve
        measured_min = psi_grid[int(0.1 * len(psi_grid))]
        measured_max = psi_grid[int(0.9 * len(psi_grid))]
    
    # Count extrapolation points
    wet_extrap = np.sum(psi_grid < measured_min)
    dry_extrap = np.sum(psi_grid > measured_max)
    interp_only = len(psi_grid) - wet_extrap - dry_extrap
    
    extrapolation_stats['wet_extrapolation'].append(wet_extrap)
    extrapolation_stats['dry_extrapolation'].append(dry_extrap)
    extrapolation_stats['interpolation_only'].append(interp_only)
    extrapolation_stats['measured_min'].append(measured_min)
    extrapolation_stats['measured_max'].append(measured_max)

extrapolation_stats = {k: np.array(v) for k, v in extrapolation_stats.items()}

print(f"   Wet-end extrapolation:")
print(f"     Mean points extrapolated: {np.mean(extrapolation_stats['wet_extrapolation']):.1f}")
print(f"     Median: {np.median(extrapolation_stats['wet_extrapolation']):.1f}")
print(f"     Samples with any wet extrapolation: {np.sum(extrapolation_stats['wet_extrapolation'] > 0)}/{len(y_test)}")

print(f"   Dry-end extrapolation:")
print(f"     Mean points extrapolated: {np.mean(extrapolation_stats['dry_extrapolation']):.1f}")
print(f"     Median: {np.median(extrapolation_stats['dry_extrapolation']):.1f}")
print(f"     Samples with any dry extrapolation: {np.sum(extrapolation_stats['dry_extrapolation'] > 0)}/{len(y_test)}")
print(f"     Mean measured max: {np.mean(extrapolation_stats['measured_max']):.2e} kPa")
print(f"     Median measured max: {np.median(extrapolation_stats['measured_max']):.2e} kPa")
print(f"     Max measured max: {np.max(extrapolation_stats['measured_max']):.2e} kPa")

# ============================================================================
# 3. Impact of Far-Dry Tail on Metrics
# ============================================================================
print("\n3. Analyzing impact of far-dry tail on metrics...")

# Load VGParamNet predictions for comparison
RESULTS_DIR = ROOT_DIR / "results_pinn_fixed"
vgnet_path = RESULTS_DIR / "vgparamnet" / "theta_vgparamnet_test.npy"

if vgnet_path.exists():
    y_vgnet = np.load(vgnet_path).astype(np.float32)
    print(f"   ✓ Loaded VGParamNet predictions: {y_vgnet.shape}")
    
    # Compute RMSE with different grid extents
    def compute_rmse_by_extent(curves_pred, curves_obs, psi_grid, max_extent):
        """Compute RMSE only up to max_extent"""
        mask = psi_grid <= max_extent
        rmse_list = []
        for i in range(len(curves_pred)):
            theta_pred = curves_pred[i][mask]
            theta_obs = curves_obs[i][mask]
            valid_mask = ~np.isnan(theta_obs)
            if np.sum(valid_mask) > 0:
                rmse = np.sqrt(np.mean((theta_pred[valid_mask] - theta_obs[valid_mask])**2))
                rmse_list.append(rmse)
        return np.array(rmse_list)
    
    # Test different extents
    extents = [1e3, 1e4, 1e5, 1e6]  # kPa
    rmse_by_extent = {}
    
    for extent in extents:
        rmse_by_extent[extent] = compute_rmse_by_extent(y_vgnet, y_test, psi_grid, extent)
        print(f"     RMSE (up to {extent:.0e} kPa): {np.mean(rmse_by_extent[extent]):.4f} (mean), {np.median(rmse_by_extent[extent]):.4f} (median)")
    
    # Compute knee metrics with different extents
    def compute_psi50(curves, theta_s, theta_r, psi_grid, max_extent):
        """Compute psi50 up to max_extent"""
        mask = psi_grid <= max_extent
        psi_subset = psi_grid[mask]
        log_psi = np.log10(psi_subset)
        
        psi50_list = []
        for i in range(len(curves)):
            theta = curves[i][mask]
            Se = (theta - theta_r[i]) / (theta_s[i] - theta_r[i] + 1e-8)
            Se = np.clip(Se, 0.0, 1.0)
            
            target = 0.5
            idx = np.where(Se <= target)[0]
            if len(idx) > 0 and idx[0] > 0:
                k = idx[0]
                x0, x1 = log_psi[k-1], log_psi[k]
                y0, y1 = Se[k-1], Se[k]
                if y1 != y0:
                    t = (target - y0) / (y1 - y0)
                    log_psi50 = x0 + t * (x1 - x0)
                    psi50_list.append(10**log_psi50)
                else:
                    psi50_list.append(10**x1)
            else:
                psi50_list.append(np.nan)
        
        return np.array(psi50_list)
    
    theta_s_test = X_test['theta_s'].values
    theta_r_test = X_test['theta_r'].values
    
    psi50_by_extent = {}
    for extent in extents:
        psi50_by_extent[extent] = compute_psi50(y_vgnet, theta_s_test, theta_r_test, psi_grid, extent)
        valid = ~np.isnan(psi50_by_extent[extent])
        if np.sum(valid) > 0:
            print(f"     ψ₅₀ (up to {extent:.0e} kPa): {np.median(psi50_by_extent[extent][valid]):.2f} kPa (median)")
    
else:
    print("   ⚠ VGParamNet predictions not found, skipping metric analysis")

# ============================================================================
# 4. Sensitivity to Grid Extent
# ============================================================================
print("\n4. Analyzing sensitivity to grid extent...")

# Test different maximum extents
max_extents = [1e3, 1e4, 1e5, 1e6, 1e7]  # kPa
n_points = 100  # Keep resolution constant

sensitivity_results = []

for max_extent in max_extents:
    # Create grid with this extent
    psi_test = np.logspace(np.log10(0.1), np.log10(max_extent), n_points)
    
    # Interpolate test curves to this grid
    y_test_resampled = []
    for i in range(len(y_test)):
        # Interpolate from original grid to new grid
        f_interp = interp1d(np.log10(psi_grid), y_test[i], 
                           kind='linear', bounds_error=False, fill_value='extrapolate')
        theta_resampled = f_interp(np.log10(psi_test))
        y_test_resampled.append(theta_resampled)
    
    y_test_resampled = np.array(y_test_resampled)
    
    # Compute coverage statistics
    dry_coverage = np.sum(psi_test > 1e4) / len(psi_test) * 100  # % points > 10^4 kPa
    
    sensitivity_results.append({
        'max_extent': max_extent,
        'dry_coverage_pct': dry_coverage,
        'n_dry_points': np.sum(psi_test > 1e4),
    })

sensitivity_df = pd.DataFrame(sensitivity_results)
print(sensitivity_df.to_string(index=False))

# ============================================================================
# 5. Sensitivity to Grid Resolution
# ============================================================================
print("\n5. Analyzing sensitivity to grid resolution...")

# Test different resolutions with fixed extent
n_points_list = [50, 75, 100, 150, 200]
max_extent = 1e6  # Keep extent constant

resolution_results = []

for n_points_test in n_points_list:
    # Create grid with this resolution
    psi_test = np.logspace(np.log10(0.1), np.log10(max_extent), n_points_test)
    
    # Interpolate test curves to this grid
    y_test_resampled = []
    for i in range(len(y_test)):
        f_interp = interp1d(np.log10(psi_grid), y_test[i], 
                           kind='linear', bounds_error=False, fill_value='extrapolate')
        theta_resampled = f_interp(np.log10(psi_test))
        y_test_resampled.append(theta_resampled)
    
    y_test_resampled = np.array(y_test_resampled)
    
    # Compute metrics
    if vgnet_path.exists():
        # Resample predictions too
        y_vgnet_resampled = []
        for i in range(len(y_vgnet)):
            f_interp = interp1d(np.log10(psi_grid), y_vgnet[i], 
                               kind='linear', bounds_error=False, fill_value='extrapolate')
            theta_resampled = f_interp(np.log10(psi_test))
            y_vgnet_resampled.append(theta_resampled)
        
        y_vgnet_resampled = np.array(y_vgnet_resampled)
        
        # Compute RMSE
        rmse_list = []
        for i in range(len(y_vgnet_resampled)):
            valid = ~np.isnan(y_test_resampled[i])
            if np.sum(valid) > 0:
                rmse = np.sqrt(np.mean((y_vgnet_resampled[i][valid] - y_test_resampled[i][valid])**2))
                rmse_list.append(rmse)
        
        rmse_mean = np.mean(rmse_list) if rmse_list else np.nan
        rmse_median = np.median(rmse_list) if rmse_list else np.nan
    else:
        rmse_mean = rmse_median = np.nan
    
    resolution_results.append({
        'n_points': n_points_test,
        'rmse_mean': rmse_mean,
        'rmse_median': rmse_median,
    })

resolution_df = pd.DataFrame(resolution_results)
print(resolution_df.to_string(index=False))

# ============================================================================
# 6. Create Visualization
# ============================================================================
print("\n6. Creating visualization...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# Panel (a): Extrapolation frequency histogram
ax = axes[0, 0]
ax.hist(extrapolation_stats['dry_extrapolation'], bins=30, edgecolor='black', 
        linewidth=1.5, alpha=0.7, color='#e74c3c')
ax.axvline(np.mean(extrapolation_stats['dry_extrapolation']), color='blue', 
           linestyle='--', linewidth=2.5, label=f'Mean: {np.mean(extrapolation_stats["dry_extrapolation"]):.1f}')
ax.set_xlabel('Number of Dry-End Extrapolated Points', fontsize=18, labelpad=10)
ax.set_ylabel('Frequency', fontsize=18, labelpad=10)
ax.set_title('(a) Dry-End Extrapolation Frequency', fontsize=18, pad=12, fontweight='normal')
ax.legend(fontsize=14)
ax.grid(True, alpha=0.3)

# Panel (b): Measured max suction distribution
ax = axes[0, 1]
ax.hist(extrapolation_stats['measured_max'], bins=50, edgecolor='black', 
        linewidth=1.5, alpha=0.7, color='#f39c12')
ax.axvline(1e6, color='red', linestyle='--', linewidth=2.5, label='Grid max (10⁶ kPa)')
ax.axvline(np.median(extrapolation_stats['measured_max']), color='blue', 
           linestyle='--', linewidth=2.5, label=f'Median: {np.median(extrapolation_stats["measured_max"]):.2e} kPa')
ax.set_xscale('log')
ax.set_xlabel('Maximum Measured Suction (kPa)', fontsize=18, labelpad=10)
ax.set_ylabel('Frequency', fontsize=18, labelpad=10)
ax.set_title('(b) Distribution of Maximum Measured Suction', fontsize=18, pad=12, fontweight='normal')
ax.legend(fontsize=14)
ax.grid(True, alpha=0.3, which='both')

# Panel (c): RMSE vs grid extent
ax = axes[0, 2]
if vgnet_path.exists() and 'rmse_by_extent' in locals():
    extents_plot = list(rmse_by_extent.keys())
    rmse_means = [np.mean(rmse_by_extent[e]) for e in extents_plot]
    rmse_medians = [np.median(rmse_by_extent[e]) for e in extents_plot]
    
    ax.plot(extents_plot, rmse_means, 'o-', linewidth=2.5, markersize=8, 
            label='Mean RMSE', color='#2E86AB')
    ax.plot(extents_plot, rmse_medians, 's--', linewidth=2.5, markersize=8, 
            label='Median RMSE', color='#F18F01')
    ax.set_xscale('log')
    ax.set_xlabel('Grid Maximum Extent (kPa)', fontsize=18, labelpad=10)
    ax.set_ylabel('RMSE (m³/m³)', fontsize=18, labelpad=10)
    ax.set_title('(c) RMSE Sensitivity to Grid Extent', fontsize=18, pad=12, fontweight='normal')
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
else:
    ax.text(0.5, 0.5, 'VGParamNet predictions\nnot available', 
            ha='center', va='center', fontsize=16)
    ax.set_title('(c) RMSE Sensitivity to Grid Extent', fontsize=18, pad=12, fontweight='normal')

# Panel (d): RMSE vs grid resolution
ax = axes[1, 0]
if not resolution_df['rmse_mean'].isna().all():
    ax.plot(resolution_df['n_points'], resolution_df['rmse_mean'], 'o-', 
            linewidth=2.5, markersize=8, label='Mean RMSE', color='#2E86AB')
    ax.plot(resolution_df['n_points'], resolution_df['rmse_median'], 's--', 
            linewidth=2.5, markersize=8, label='Median RMSE', color='#F18F01')
    ax.set_xlabel('Number of Grid Points', fontsize=18, labelpad=10)
    ax.set_ylabel('RMSE (m³/m³)', fontsize=18, labelpad=10)
    ax.set_title('(d) RMSE Sensitivity to Grid Resolution', fontsize=18, pad=12, fontweight='normal')
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'VGParamNet predictions\nnot available', 
            ha='center', va='center', fontsize=16)
    ax.set_title('(d) RMSE Sensitivity to Grid Resolution', fontsize=18, pad=12, fontweight='normal')

# Panel (e): Representative curve showing extrapolation regions
ax = axes[1, 1]
sample_idx = 0
theta_curve = y_test[sample_idx]
measured_min = extrapolation_stats['measured_min'][sample_idx]
measured_max = extrapolation_stats['measured_max'][sample_idx]

ax.semilogx(psi_grid, theta_curve, 'k-', linewidth=2.5, label='Interpolated curve', alpha=0.8)
ax.axvspan(psi_grid.min(), measured_min, alpha=0.2, color='blue', label='Wet extrapolation')
ax.axvspan(measured_max, psi_grid.max(), alpha=0.2, color='red', label='Dry extrapolation')
ax.axvline(measured_min, color='blue', linestyle='--', linewidth=2, label='Measured min')
ax.axvline(measured_max, color='red', linestyle='--', linewidth=2, label='Measured max')
ax.set_xlabel('Suction ψ (kPa)', fontsize=18, labelpad=10)
ax.set_ylabel('Water content θ (m³/m³)', fontsize=18, labelpad=10)
ax.set_title('(e) Representative Curve: Extrapolation Regions', fontsize=18, pad=12, fontweight='normal')
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.3, which='both')

# Panel (f): Summary statistics table
ax = axes[1, 2]
ax.axis('off')
table_data = [
    ['Metric', 'Value'],
    ['Grid extent', '0.1 - 10⁶ kPa'],
    ['Grid resolution', '100 points'],
    ['Mean dry extrapolation', f'{np.mean(extrapolation_stats["dry_extrapolation"]):.1f} points'],
    ['Median dry extrapolation', f'{np.median(extrapolation_stats["dry_extrapolation"]):.1f} points'],
    ['Samples with dry extrap', f'{np.sum(extrapolation_stats["dry_extrapolation"] > 0)}/{len(y_test)}'],
    ['Median measured max', f'{np.median(extrapolation_stats["measured_max"]):.2e} kPa'],
    ['Max measured max', f'{np.max(extrapolation_stats["measured_max"]):.2e} kPa'],
]

table = ax.table(cellText=table_data, cellLoc='left', loc='center', 
                 bbox=[0, 0, 1, 1], colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1, 2.5)
ax.set_title('(f) Summary Statistics', fontsize=18, pad=12, fontweight='normal')

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "grid_sensitivity_analysis.png", dpi=300, bbox_inches='tight')
fig.savefig(OUTPUT_DIR / "grid_sensitivity_analysis.pdf", dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ Saved: {OUTPUT_DIR / 'grid_sensitivity_analysis.png'}")

# ============================================================================
# 7. Save Results
# ============================================================================
print("\n7. Saving results...")

results = {
    'extrapolation_stats': {k: v.tolist() for k, v in extrapolation_stats.items()},
    'sensitivity_extent': sensitivity_df.to_dict('records'),
    'sensitivity_resolution': resolution_df.to_dict('records'),
}

if vgnet_path.exists() and 'rmse_by_extent' in locals():
    results['rmse_by_extent'] = {str(k): v.tolist() for k, v in rmse_by_extent.items()}
    results['psi50_by_extent'] = {str(k): v.tolist() for k, v in psi50_by_extent.items()}

with open(OUTPUT_DIR / "grid_sensitivity_results.json", 'w') as f:
    json.dump(results, f, indent=2)

sensitivity_df.to_csv(OUTPUT_DIR / "sensitivity_extent.csv", index=False)
resolution_df.to_csv(OUTPUT_DIR / "sensitivity_resolution.csv", index=False)

print(f"   ✓ Saved: {OUTPUT_DIR / 'grid_sensitivity_results.json'}")
print(f"   ✓ Saved: {OUTPUT_DIR / 'sensitivity_extent.csv'}")
print(f"   ✓ Saved: {OUTPUT_DIR / 'sensitivity_resolution.csv'}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nKey Findings:")
print(f"  - Mean dry-end extrapolation: {np.mean(extrapolation_stats['dry_extrapolation']):.1f} points per curve")
print(f"  - Median measured max suction: {np.median(extrapolation_stats['measured_max']):.2e} kPa")
print(f"  - Grid extends to 10⁶ kPa to cover extreme dry conditions")
print(f"  - {np.sum(extrapolation_stats['dry_extrapolation'] > 0)}/{len(y_test)} samples require dry-end extrapolation")
