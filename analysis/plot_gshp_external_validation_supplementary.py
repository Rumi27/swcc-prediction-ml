#!/usr/bin/env python3
"""
Generate Supplementary Figure: GSHP External Validation (Detailed Metrics)

This script creates a supplementary figure with:
(a) Knee metrics boxplots (ψ₅₀ and max_slope)
(b) Outlier curve (top 1% error case)

Author: Generated for supplementary material
Date: 2026-02-25
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
OUTPUT_DIR = ROOT_DIR / "paper_figures" / "supplementary"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Style settings
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
# Match paper Figures 11–17 style request
_F2 = 11
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = _F2
plt.rcParams['axes.labelsize'] = _F2
plt.rcParams['axes.titlesize'] = _F2
plt.rcParams['xtick.labelsize'] = _F2
plt.rcParams['ytick.labelsize'] = _F2
plt.rcParams['legend.fontsize'] = _F2
plt.rcParams['figure.dpi'] = 300


def _f2_set_arial_ticks(ax):
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("Arial")
        lbl.set_fontsize(_F2)

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
    
    # Extract theta_s and theta_r from curves
    theta_s = np.max(curves_gshp, axis=1)
    theta_r = np.min(curves_gshp, axis=1)
    
    print(f"  Loaded {len(curves_pred)} samples")
    
    return results, curves_pred, curves_gshp, suction_grid, theta_s, theta_r


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


def plot_panel_a(ax, curves_pred, curves_gshp, suction_grid, theta_s, theta_r):
    """Panel (a): Knee metrics boxplots."""
    print("\nPlotting panel (a): Knee metrics boxplots...")
    
    # Compute full distributions
    psi50_pred, maxslope_pred = compute_knee_metrics_full(curves_pred, theta_s, theta_r, suction_grid)
    psi50_gshp, maxslope_gshp = compute_knee_metrics_full(curves_gshp, theta_s, theta_r, suction_grid)
    
    print(f"    ψ₅₀: GSHP median={np.median(psi50_gshp):.2f}, VGParamNet median={np.median(psi50_pred):.2f}")
    print(f"    max_slope: GSHP median={np.median(maxslope_gshp):.3f}, VGParamNet median={np.median(maxslope_pred):.3f}")
    
    # Filter out NaN/inf values
    psi50_pred_clean = psi50_pred[np.isfinite(psi50_pred) & (psi50_pred > 0)]
    psi50_gshp_clean = psi50_gshp[np.isfinite(psi50_gshp) & (psi50_gshp > 0)]
    maxslope_pred_clean = maxslope_pred[np.isfinite(maxslope_pred) & (maxslope_pred > 0)]
    maxslope_gshp_clean = maxslope_gshp[np.isfinite(maxslope_gshp) & (maxslope_gshp > 0)]
    
    # Create side-by-side boxplots
    positions = [1, 2, 4, 5]
    
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
    # Use mathtext to avoid Unicode subscript glyph issues in Arial
    ax.set_xticklabels([r'$\psi_{50}$ (kPa)', r'max $|d\theta/d\log(\psi)|$'])
    ax.set_ylabel('Value')
    ax.set_title('(a) Knee metrics: GSHP vs VGParamNet', pad=10)
    ax.set_yscale('log')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.7, label='GSHP'),
        Patch(facecolor='coral', alpha=0.7, label='VGParamNet')
    ]
    # Move legend slightly downward to avoid covering plot content
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.68),
        framealpha=0.9,
        fontsize=_F2,
    )
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    _f2_set_arial_ticks(ax)


def plot_panel_b(ax, curves_pred, curves_gshp, suction_grid, rmse_per_sample):
    """Panel (b): Outlier curve (top 1% error)."""
    print("\nPlotting panel (b): Outlier curve...")
    
    # Select top 1% outlier
    top1_threshold = np.percentile(rmse_per_sample, 99)
    outlier_candidates = np.where(rmse_per_sample >= top1_threshold)[0]
    outlier_idx = outlier_candidates[np.argmax(rmse_per_sample[outlier_candidates])]
    
    print(f"  Outlier (top 1%): sample {outlier_idx}, RMSE = {rmse_per_sample[outlier_idx]:.4f}")
    
    # Plot curves
    ax.plot(suction_grid, curves_gshp[outlier_idx], 'b-', linewidth=2.5, 
            label='GSHP', alpha=0.8)
    ax.plot(suction_grid, curves_pred[outlier_idx], 'r--', linewidth=2.5, 
            label='VGParamNet', alpha=0.8)
    
    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel('Matric suction ψ (kPa)', fontsize=_F2, fontfamily="Arial", labelpad=10)
    ax.set_ylabel('Water content θ', fontsize=_F2, fontfamily="Arial", labelpad=10)
    ax.set_title('(b) Outlier case (top 1% error)', fontsize=_F2, fontfamily="Arial", pad=10)

    # Match SWCC axis styling (Figure 3 / Figure 11): log ψ with decade ticks
    _ticks = np.array([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    _ticklabels = ["0.1", "1.0", "10", "100", "1000", "10000", "100000", "1000000"]
    ax.set_xticks(_ticks)
    ax.set_xticklabels(_ticklabels)
    for _lbl in ax.get_xticklabels():
        _lbl.set_fontfamily("Arial")
        _lbl.set_fontsize(_F2 - 1)

    ax.set_xlim(suction_grid.min(), suction_grid.max())
    ax.tick_params(labelsize=_F2 - 1)
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    # Move legend to the top-right corner
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        framealpha=0.9,
        fontsize=_F2 - 1,
    )
    ax.set_ylim(0, max(curves_gshp[outlier_idx].max(), curves_pred[outlier_idx].max()) * 1.1)
    
    # Add RMSE annotation
    ax.text(0.05, 0.95, f'RMSE = {rmse_per_sample[outlier_idx]:.4f}', 
            transform=ax.transAxes, fontsize=_F2, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    _f2_set_arial_ticks(ax)


def compute_per_sample_rmse(curves_pred, curves_gshp):
    """Compute per-sample RMSE for each GSHP sample."""
    n_samples = len(curves_pred)
    rmse_per_sample = np.zeros(n_samples)
    
    for i in range(n_samples):
        diff = curves_pred[i] - curves_gshp[i]
        rmse_per_sample[i] = np.sqrt(np.mean(diff**2))
    
    return rmse_per_sample


def main():
    """Main function to generate supplementary figure."""
    print("="*80)
    print("Generating Supplementary Figure: GSHP External Validation (Detailed Metrics)")
    print("="*80)
    
    # Load data
    results, curves_pred, curves_gshp, suction_grid, theta_s, theta_r = load_data()
    
    # Compute per-sample RMSE
    rmse_per_sample = compute_per_sample_rmse(curves_pred, curves_gshp)
    
    # Create figure (2 panels, side by side)
    print("\nCreating figure...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel (a): Knee metrics
    plot_panel_a(axes[0], curves_pred, curves_gshp, suction_grid, theta_s, theta_r)
    
    # Panel (b): Outlier curve
    plot_panel_b(axes[1], curves_pred, curves_gshp, suction_grid, rmse_per_sample)
    
    # (intentionally no overall suptitle for Figure S2)
    
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / "Figure_S2_GSHP_External_Validation_Detailed.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved figure to: {output_path}")
    
    # Also save as PDF
    output_path_pdf = OUTPUT_DIR / "Figure_S2_GSHP_External_Validation_Detailed.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved figure (PDF) to: {output_path_pdf}")
    
    print("\n" + "="*80)
    print("Figure generation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
