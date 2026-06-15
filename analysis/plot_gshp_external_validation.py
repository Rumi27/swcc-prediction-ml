#!/usr/bin/env python3
"""
Generate Figure 17: External Validation on GSHP (Curve Space)

This script creates a 3-panel figure showing:
(a) Distribution of per-sample RMSE (VGParamNet vs GSHP reconstructed curves)
(b) Distributions of ψ₅₀ and max_slope (GSHP vs VGParamNet)
(c) 4 representative curve overlays (low, median, high error, outlier)

Author: Generated for external validation figure
Date: 2026-02-24
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Configuration
RESULTS_DIR = ROOT_DIR / "results_gshp_validation"
OUTPUT_DIR = ROOT_DIR / "paper_figures"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Style settings
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300

def load_data():
    """Load all GSHP validation data."""
    print("Loading GSHP validation data...")
    
    # Load results JSON
    with open(RESULTS_DIR / "gshp_validation_results.json", 'r') as f:
        results = json.load(f)
    
    # Load curves
    curves_pred = np.load(RESULTS_DIR / "gshp_predicted_curves.npy")
    curves_gshp = np.load(RESULTS_DIR / "gshp_observed_curves.npy")
    
    # Load suction grid
    DATA_DIR = ROOT_DIR / "data_pinn_normalized"
    suction_grid = np.load(DATA_DIR / "suction_grid.npy")
    
    # Load theta_s and theta_r from GSHP data (needed for knee metrics)
    # We need to reload the GSHP data to get theta_s/theta_r
    GSHP_DATA_PATH = ROOT_DIR.parent / "data" / "GSHP_downloaded" / "WRC_dataset_surya_et_al_2021_final.csv"
    df_gshp = pd.read_csv(GSHP_DATA_PATH, encoding='latin-1', low_memory=False)
    
    # Apply same QC as evaluation script
    needed_cols = ['sand_tot_psa','silt_tot_psa','clay_tot_psa','db_od','alpha','n','thetar','thetas']
    df_clean = df_gshp.dropna(subset=needed_cols).copy()
    for c in needed_cols:
        df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce')
    df_clean = df_clean.dropna(subset=needed_cols)
    
    # Build X_input with same QC
    X_input = pd.DataFrame(index=df_clean.index)
    X_input['sand_pct'] = df_clean['sand_tot_psa']
    X_input['silt_pct'] = df_clean['silt_tot_psa']
    X_input['clay_pct'] = df_clean['clay_tot_psa']
    X_input['bulk_density'] = df_clean['db_od']
    if 'porosity' in df_clean.columns:
        X_input['porosity'] = pd.to_numeric(df_clean['porosity'], errors='coerce')
        mask = X_input['porosity'].isna()
        X_input.loc[mask, 'porosity'] = 1 - X_input.loc[mask, 'bulk_density']/2.65
    else:
        X_input['porosity'] = 1 - X_input['bulk_density']/2.65
    X_input['theta_s'] = df_clean['thetas']
    X_input['theta_r'] = df_clean['thetar']
    
    # Apply QC filters
    req = ['sand_pct','silt_pct','clay_pct','bulk_density','porosity','theta_s','theta_r']
    X_input = X_input.dropna(subset=req)
    mask = (
        (X_input['sand_pct']>=0)&(X_input['sand_pct']<=100)&
        (X_input['silt_pct']>=0)&(X_input['silt_pct']<=100)&
        (X_input['clay_pct']>=0)&(X_input['clay_pct']<=100)&
        (X_input['bulk_density']>0)&(X_input['bulk_density']<3.0)&
        (X_input['porosity']>0)&(X_input['porosity']<1)&
        (X_input['theta_s']>0)&(X_input['theta_s']<=1)&
        (X_input['theta_r']>=0)&(X_input['theta_r']<X_input['theta_s'])
    )
    X_input = X_input[mask]
    
    # Extract theta_s and theta_r (aligned with curves)
    # Note: curves were generated from a subset after QC, so we need to match
    # For now, extract from curves (theta_s = max, theta_r = min approximately)
    # Actually better: save theta_s/theta_r in evaluation script, but for now use curve endpoints
    theta_s = np.max(curves_gshp, axis=1)  # Use GSHP curves to get theta_s
    theta_r = np.min(curves_gshp, axis=1)  # Use GSHP curves to get theta_r
    
    # Alternatively, we could load from saved arrays if they exist
    # For now, using curve endpoints is reasonable approximation
    
    print(f"  Loaded {len(curves_pred)} samples")
    print(f"  Curve shape: {curves_pred.shape}")
    print(f"  Suction grid: {len(suction_grid)} points")
    print(f"  Theta_s/theta_r shape: {theta_s.shape}")
    print(f"  Theta_s range: {theta_s.min():.3f} - {theta_s.max():.3f}")
    print(f"  Theta_r range: {theta_r.min():.3f} - {theta_r.max():.3f}")
    
    return results, curves_pred, curves_gshp, suction_grid, theta_s, theta_r


def compute_per_sample_rmse(curves_pred, curves_gshp):
    """Compute per-sample RMSE for each GSHP sample."""
    print("\nComputing per-sample RMSE...")
    
    n_samples = len(curves_pred)
    rmse_per_sample = np.zeros(n_samples)
    
    for i in range(n_samples):
        diff = curves_pred[i] - curves_gshp[i]
        rmse_per_sample[i] = np.sqrt(np.mean(diff**2))
    
    print(f"  RMSE range: {rmse_per_sample.min():.4f} - {rmse_per_sample.max():.4f}")
    print(f"  RMSE median: {np.median(rmse_per_sample):.4f}")
    
    return rmse_per_sample


def select_representative_samples(rmse_per_sample):
    """Select 4 representative samples based on RMSE quantiles."""
    print("\nSelecting representative samples...")
    
    # Quantiles
    q10_idx = np.argmin(np.abs(rmse_per_sample - np.percentile(rmse_per_sample, 10)))
    q50_idx = np.argmin(np.abs(rmse_per_sample - np.percentile(rmse_per_sample, 50)))
    q90_idx = np.argmin(np.abs(rmse_per_sample - np.percentile(rmse_per_sample, 90)))
    
    # Top 1% outlier
    top1_threshold = np.percentile(rmse_per_sample, 99)
    outlier_candidates = np.where(rmse_per_sample >= top1_threshold)[0]
    outlier_idx = outlier_candidates[np.argmax(rmse_per_sample[outlier_candidates])]
    
    indices = {
        'low_error': q10_idx,
        'median': q50_idx,
        'high_error': q90_idx,
        'outlier': outlier_idx
    }
    
    print(f"  Low error (q10): sample {q10_idx}, RMSE = {rmse_per_sample[q10_idx]:.4f}")
    print(f"  Median (q50): sample {q50_idx}, RMSE = {rmse_per_sample[q50_idx]:.4f}")
    print(f"  High error (q90): sample {q90_idx}, RMSE = {rmse_per_sample[q90_idx]:.4f}")
    print(f"  Outlier (top 1%): sample {outlier_idx}, RMSE = {rmse_per_sample[outlier_idx]:.4f}")
    
    return indices


def plot_panel_a(ax, rmse_per_sample):
    """Panel (a): Distribution of per-sample RMSE."""
    print("\nPlotting panel (a): RMSE distribution...")
    
    # Histogram with KDE
    ax.hist(rmse_per_sample, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Add KDE
    from scipy import stats
    kde = stats.gaussian_kde(rmse_per_sample)
    x_kde = np.linspace(rmse_per_sample.min(), rmse_per_sample.max(), 200)
    ax.plot(x_kde, kde(x_kde), 'r-', linewidth=2, label='KDE')
    
    # Add vertical lines for quantiles
    q10 = np.percentile(rmse_per_sample, 10)
    q50 = np.percentile(rmse_per_sample, 50)
    q90 = np.percentile(rmse_per_sample, 90)
    q99 = np.percentile(rmse_per_sample, 99)
    
    ax.axvline(q10, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'q10 = {q10:.3f}')
    ax.axvline(q50, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label=f'q50 = {q50:.3f}')
    ax.axvline(q90, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'q90 = {q90:.3f}')
    ax.axvline(q99, color='purple', linestyle='--', linewidth=1.5, alpha=0.7, label=f'q99 = {q99:.3f}')
    
    ax.set_xlabel('Per-sample RMSE', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('(a) Distribution of per-sample RMSE', fontweight='bold', pad=10)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add text box with summary stats
    textstr = f'Median: {q50:.3f}\nMean: {rmse_per_sample.mean():.3f}'
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_panel_b(ax, results):
    """Panel (b): Distributions of ψ₅₀ and max_slope."""
    print("\nPlotting panel (b): Knee metrics distributions...")
    
    knee_data = results['knee_metrics_summary']
    
    # Prepare data for boxplot
    psi50_data = {
        'GSHP': [knee_data['psi50_gshp']['q10'], knee_data['psi50_gshp']['median'], knee_data['psi50_gshp']['q90']],
        'VGParamNet': [knee_data['psi50_pred']['q10'], knee_data['psi50_pred']['median'], knee_data['psi50_pred']['q90']]
    }
    
    maxslope_data = {
        'GSHP': [knee_data['maxslope_gshp']['q10'], knee_data['maxslope_gshp']['median'], knee_data['maxslope_gshp']['q90']],
        'VGParamNet': [knee_data['maxslope_pred']['q10'], knee_data['maxslope_pred']['median'], knee_data['maxslope_pred']['q90']]
    }
    
    # Create two subplots within this panel
    ax1 = ax[0] if hasattr(ax, '__len__') else ax
    ax2 = ax[1] if hasattr(ax, '__len__') else None
    
    # If ax is not a list, create a subplot layout
    if not hasattr(ax, '__len__'):
        # This shouldn't happen if called correctly, but handle it
        ax1 = ax
        ax2 = None
    
    # Plot ψ₅₀
    positions = [1, 2]
    bp1 = ax1.boxplot([
        [knee_data['psi50_gshp']['q10'], knee_data['psi50_gshp']['median'], knee_data['psi50_gshp']['q90']],
        [knee_data['psi50_pred']['q10'], knee_data['psi50_pred']['median'], knee_data['psi50_pred']['q90']]
    ], positions=positions, widths=0.6, patch_artist=True,
    boxprops=dict(facecolor='lightblue', alpha=0.7),
    medianprops=dict(color='red', linewidth=2),
    whiskerprops=dict(color='black', linewidth=1.5),
    capprops=dict(color='black', linewidth=1.5))
    
    ax1.set_xticks(positions)
    ax1.set_xticklabels(['GSHP', 'VGParamNet'])
    ax1.set_ylabel('ψ₅₀ (kPa)', fontweight='bold')
    ax1.set_title('(b) Knee location (ψ₅₀)', fontweight='bold', pad=10)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Plot max_slope
    bp2 = ax2.boxplot([
        [knee_data['maxslope_gshp']['q10'], knee_data['maxslope_gshp']['median'], knee_data['maxslope_gshp']['q90']],
        [knee_data['maxslope_pred']['q10'], knee_data['maxslope_pred']['median'], knee_data['maxslope_pred']['q90']]
    ], positions=positions, widths=0.6, patch_artist=True,
    boxprops=dict(facecolor='lightcoral', alpha=0.7),
    medianprops=dict(color='red', linewidth=2),
    whiskerprops=dict(color='black', linewidth=1.5),
    capprops=dict(color='black', linewidth=1.5))
    
    ax2.set_xticks(positions)
    ax2.set_xticklabels(['GSHP', 'VGParamNet'])
    ax2.set_ylabel('max |dθ/dlog(ψ)|', fontweight='bold')
    ax2.set_title('(b) Knee sharpness (max slope)', fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')


def find_psi50(psi, theta, theta_s, theta_r):
    """Find suction at Se = 0.5."""
    Se = (theta - theta_r) / (theta_s - theta_r + 1e-10)
    Se = np.clip(Se, 0, 1)
    
    # Find where Se crosses 0.5
    idx = np.argmin(np.abs(Se - 0.5))
    return psi[idx]


def compute_max_slope(psi, theta):
    """Compute maximum absolute slope in log-suction space."""
    log_psi = np.log10(psi + 1e-10)
    dtheta = np.diff(theta)
    dlog_psi = np.diff(log_psi)
    slopes = np.abs(dtheta / (dlog_psi + 1e-10))
    max_slope = np.max(slopes)
    max_idx = np.argmax(slopes)
    return max_slope, max_idx


def plot_panel_b_simple(ax, curves_pred, curves_gshp, suction_grid, theta_s, theta_r):
    """Panel (b): Boxplots of knee metrics computed from full distributions."""
    print("\nPlotting panel (b): Knee metrics distributions...")
    print("  Computing full distributions from curves...")
    
    # Compute full distributions
    psi50_pred, maxslope_pred = compute_knee_metrics_full(curves_pred, theta_s, theta_r, suction_grid)
    psi50_gshp, maxslope_gshp = compute_knee_metrics_full(curves_gshp, theta_s, theta_r, suction_grid)
    
    print(f"    ψ₅₀: GSHP median={np.median(psi50_gshp):.2f}, VGParamNet median={np.median(psi50_pred):.2f}")
    print(f"    max_slope: GSHP median={np.median(maxslope_gshp):.3f}, VGParamNet median={np.median(maxslope_pred):.3f}")
    
    # Create side-by-side boxplots
    positions = [1, 2, 4, 5]
    
    # Filter out NaN/inf values
    psi50_pred_clean = psi50_pred[np.isfinite(psi50_pred) & (psi50_pred > 0)]
    psi50_gshp_clean = psi50_gshp[np.isfinite(psi50_gshp) & (psi50_gshp > 0)]
    maxslope_pred_clean = maxslope_pred[np.isfinite(maxslope_pred) & (maxslope_pred > 0)]
    maxslope_gshp_clean = maxslope_gshp[np.isfinite(maxslope_gshp) & (maxslope_gshp > 0)]
    
    # Create boxplots
    bp1 = ax.boxplot([psi50_gshp_clean, psi50_pred_clean], 
                     positions=[1, 2], widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor='steelblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(color='black', linewidth=1.5),
                     capprops=dict(color='black', linewidth=1.5),
                     showfliers=False)
    
    bp2 = ax.boxplot([maxslope_gshp_clean, maxslope_pred_clean],
                     positions=[4, 5], widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor='coral', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(color='black', linewidth=1.5),
                     capprops=dict(color='black', linewidth=1.5),
                     showfliers=False)
    
    # Add labels
    ax.set_xticks([1.5, 4.5])
    ax.set_xticklabels(['ψ₅₀ (kPa)', 'max |dθ/dlog(ψ)|'], fontweight='bold')
    ax.set_ylabel('Value', fontweight='bold')
    ax.set_title('(b) Knee metrics: GSHP vs VGParamNet', fontweight='bold', pad=10)
    ax.set_yscale('log')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.7, label='GSHP'),
        Patch(facecolor='coral', alpha=0.7, label='VGParamNet')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')


def compute_knee_metrics_full(curves, theta_s, theta_r, suction_grid):
    """
    Compute full arrays of psi50 and max_slope from curves.
    
    Args:
        curves: θ(ψ) [N, P]
        theta_s: [N]
        theta_r: [N]
        suction_grid: ψ grid [P]
    
    Returns:
        psi50: [N] knee locations (kPa)
        max_slope: [N] max |dθ/dlogψ|
    """
    psi = suction_grid
    log_psi = np.log10(psi + 1e-10)
    
    n_samples = len(curves)
    psi50 = np.full(n_samples, np.nan, dtype=np.float32)
    max_slope = np.full(n_samples, np.nan, dtype=np.float32)
    
    theta_s = theta_s.reshape(-1, 1)
    theta_r = theta_r.reshape(-1, 1)
    theta_range = np.maximum(theta_s - theta_r, 1e-6)
    
    Se = (curves - theta_r) / theta_range
    Se = np.clip(Se, 1e-6, 1 - 1e-6)
    
    # Compute psi50 for each sample
    target = 0.5
    for i in range(n_samples):
        Se_i = Se[i]
        idx = np.where(Se_i <= target)[0]
        if len(idx) > 0 and idx[0] > 0:
            k = idx[0]
            x0, x1 = log_psi[k-1], log_psi[k]
            y0, y1 = Se_i[k-1], Se_i[k]
            if y1 != y0:
                t = (target - y0) / (y1 - y0)
                log_psi50 = x0 + t * (x1 - x0)
                psi50[i] = 10**log_psi50
            else:
                psi50[i] = 10**x1
    
    # Compute max_slope for each sample
    dtheta = np.diff(curves, axis=1)
    dlogpsi = np.diff(log_psi)[None, :]
    slopes = np.abs(dtheta / (dlogpsi + 1e-10))
    max_slope = np.max(slopes, axis=1)
    
    return psi50, max_slope


def plot_panel_c(fig, gs_c, curves_pred, curves_gshp, suction_grid, indices, rmse_per_sample):
    """Panel (c): 4 representative curve overlays."""
    print("\nPlotting panel (c): Representative curves...")
    
    # Create 2x2 subplot grid within the allocated space
    labels = ['Low error (q10)', 'Median (q50)', 'High error (q90)', 'Outlier (top 1%)']
    idx_list = [indices['low_error'], indices['median'], indices['high_error'], indices['outlier']]
    
    # Create 2x2 subplots
    axes = []
    for i in range(2):
        for j in range(2):
            ax = fig.add_subplot(gs_c[i, j])
            axes.append(ax)
    
    for i, (sub_ax, label, idx) in enumerate(zip(axes, labels, idx_list)):
        # Plot curves
        sub_ax.plot(suction_grid, curves_gshp[idx], 'b-', linewidth=2, label='GSHP', alpha=0.8)
        sub_ax.plot(suction_grid, curves_pred[idx], 'r--', linewidth=2, label='VGParamNet', alpha=0.8)
        
        # Formatting
        sub_ax.set_xscale('log')
        sub_ax.set_xlabel('Suction ψ (kPa)', fontweight='bold')
        sub_ax.set_ylabel('Water content θ', fontweight='bold')
        sub_ax.set_title(f'{label}\nRMSE = {rmse_per_sample[idx]:.4f}', fontweight='bold', fontsize=10)
        sub_ax.legend(loc='best', framealpha=0.9, fontsize=8)
        sub_ax.grid(True, alpha=0.3, linestyle='--')
        sub_ax.set_xlim(suction_grid.min(), suction_grid.max())
        sub_ax.set_ylim(0, max(curves_gshp[idx].max(), curves_pred[idx].max()) * 1.1)


def main():
    """Main function to generate Figure 17."""
    print("="*80)
    print("Generating Figure 17: External Validation on GSHP (Curve Space)")
    print("="*80)
    
    # Load data
    results, curves_pred, curves_gshp, suction_grid, theta_s, theta_r = load_data()
    
    # Compute per-sample RMSE
    rmse_per_sample = compute_per_sample_rmse(curves_pred, curves_gshp)
    
    # Select representative samples
    indices = select_representative_samples(rmse_per_sample)
    
    # Create figure
    print("\nCreating figure...")
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Panel (a): RMSE distribution
    ax_a = fig.add_subplot(gs[0, 0])
    plot_panel_a(ax_a, rmse_per_sample)
    
    # Panel (b): Knee metrics
    ax_b = fig.add_subplot(gs[0, 1])
    plot_panel_b_simple(ax_b, curves_pred, curves_gshp, suction_grid, theta_s, theta_r)
    
    # Panel (c): Representative curves (2x2 grid)
    gs_c = gs[1, :].subgridspec(2, 2, hspace=0.3, wspace=0.3)
    plot_panel_c(fig, gs_c, curves_pred, curves_gshp, suction_grid, indices, rmse_per_sample)
    
    # Add overall title
    fig.suptitle('Figure 17: External Validation on GSHP (Curve Space)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Add inset text with sample info
    textstr = f'N = {results["n_samples"]:,} samples\nα conversion: 1/m → 1/kPa (factor 0.10197)'
    fig.text(0.99, 0.01, textstr, fontsize=8, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save figure
    output_path = OUTPUT_DIR / "Figure17_GSHP_External_Validation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved figure to: {output_path}")
    
    # Also save as PDF
    output_path_pdf = OUTPUT_DIR / "Figure17_GSHP_External_Validation.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved figure (PDF) to: {output_path_pdf}")
    
    print("\n" + "="*80)
    print("Figure generation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
