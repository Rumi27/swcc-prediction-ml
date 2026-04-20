#!/usr/bin/env python3
"""
Generate Figure S5: Gradient Boosting Monotonicity Violations
Combined statistics and representative failure curves for Supplementary Material
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_pinn.config_pinn_fixed import DATA_CONFIG
from baseline_models import BaselineModels

# Output directory
OUTPUT_DIR = ROOT_DIR / "paper_figures" / "supplementary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Analysis results directory
ANALYSIS_DIR = ROOT_DIR / "results_baseline" / "monotonicity_analysis"

print("="*80)
print("Generating Figure S5: GB Monotonicity Violations")
print("="*80)

# ============================================================================
# 1. Load Data and Analysis Results
# ============================================================================
print("\n1. Loading data and analysis results...")

# Load analysis results
with open(ANALYSIS_DIR / "gb_monotonicity_analysis.json", 'r') as f:
    results = json.load(f)

# Load test data
X_test = pd.read_csv(DATA_CONFIG["test_file"])
y_test = np.load(DATA_CONFIG["y_test_original_file"]).astype(np.float32)
psi = np.load(DATA_CONFIG["suction_grid_file"]).astype(np.float32)

# Generate GB predictions (or load if available)
print("   Generating GB predictions...")
bm = BaselineModels(data_dir=ROOT_DIR / "data_processed", 
                    output_dir=ROOT_DIR / "results_baseline")
(X_train, X_val, X_test_gb), (y_train, y_val, y_test_gb), suction_grid_gb = bm.load_data()
X_train_feat, X_val_feat, X_test_feat, feature_cols = bm.prepare_features(
    X_train, X_val, X_test_gb)

# Train GB
gb_models = bm.train_gradient_boosting(X_train_feat, y_train, X_val_feat, y_val)

# Predict on test set
y_gb = bm.predict_swcc(gb_models, X_test_feat, model_type="gradient_boosting", 
                       n_points=len(psi))

# Ensure same length
if len(y_gb) != len(y_test):
    min_len = min(len(y_gb), len(y_test))
    y_gb = y_gb[:min_len]
    y_test = y_test[:min_len]
    X_test = X_test.iloc[:min_len].reset_index(drop=True)

print(f"   ✓ Data loaded: {len(y_gb)} samples")

# ============================================================================
# 2. Extract Statistics
# ============================================================================
print("\n2. Extracting statistics...")

bump_counts = np.array(results['strict']['bump_counts'])
pass_strict = np.array(results['strict']['pass_strict'])
violation_indices = np.where(~pass_strict)[0]

# Compute maximum amplitudes for each curve
all_max_amplitudes = []
for i in range(len(y_gb)):
    theta = y_gb[i]
    diff = theta[:-1] - theta[1:]
    violations = diff < -1e-8
    if np.any(violations):
        max_amp = np.max(-diff[violations])
        all_max_amplitudes.append(max_amp)
    else:
        all_max_amplitudes.append(0.0)

all_max_amplitudes = np.array(all_max_amplitudes)

# Find representative samples
median_bumps = int(np.median(bump_counts))
max_amp_idx = np.argmax(all_max_amplitudes)
max_bumps_idx = np.argmax(bump_counts)

# Find sample with median number of bumps
median_bumps_indices = np.where(bump_counts == median_bumps)[0]
if len(median_bumps_indices) > 0:
    typical_idx = median_bumps_indices[len(median_bumps_indices) // 2]
else:
    typical_idx = violation_indices[0]

print(f"   ✓ Representative samples selected:")
print(f"     - Typical (median bumps): sample {typical_idx}, {bump_counts[typical_idx]} bumps")
print(f"     - Max amplitude: sample {max_amp_idx}, amplitude {all_max_amplitudes[max_amp_idx]:.6f}")
print(f"     - Max bumps: sample {max_bumps_idx}, {bump_counts[max_bumps_idx]} bumps")

# ============================================================================
# 3. Create Figure S5
# ============================================================================
print("\n3. Creating Figure S5...")

# Set global font sizes and font family (Arial 17 style)
plt.rcParams['font.size'] = 17
plt.rcParams['axes.labelsize'] = 17
plt.rcParams['axes.titlesize'] = 17
plt.rcParams['xtick.labelsize'] = 17
plt.rcParams['ytick.labelsize'] = 17
plt.rcParams['legend.fontsize'] = 17
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# Slightly less wide, a bit taller
fig = plt.figure(figsize=(18, 11))
# 2 rows: top row has 2 panels (a, b), bottom row has 3 panels (c, d, e)
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3, 
                      left=0.08, right=0.98, top=0.95, bottom=0.08)

# ============================================================================
# Row 1: Statistics (Panels a and b)
# ============================================================================

# Panel (a): Distribution of Bump Counts
ax_a = fig.add_subplot(gs[0, 0])
bump_counts_nonzero = bump_counts[bump_counts > 0]
ax_a.hist(bump_counts, bins=min(35, int(np.max(bump_counts)) + 1), 
          range=(0, 35), edgecolor='black', linewidth=1.5, alpha=0.7, color='#e74c3c')
ax_a.axvline(np.mean(bump_counts), color='blue', linestyle='--', linewidth=2.5, 
             label=f'Mean: {np.mean(bump_counts):.1f}')
ax_a.axvline(np.median(bump_counts), color='green', linestyle='--', linewidth=2.5, 
             label=f'Median: {np.median(bump_counts):.0f}')
ax_a.set_xlabel('Number of bumps per curve', fontsize=17, fontfamily='Arial', labelpad=10)
ax_a.set_ylabel('Frequency', fontsize=17, fontfamily='Arial', labelpad=10)
ax_a.set_title('(a) Distribution of bump counts', fontsize=17, fontfamily='Arial', pad=12, fontweight='normal')
ax_a.legend(fontsize=17, loc='upper right')
ax_a.grid(False)
ax_a.set_xlim(-0.5, 35.5)

# Panel (b): Distribution of Maximum Bump Amplitudes
ax_b = fig.add_subplot(gs[0, 1])
amplitudes_nonzero = all_max_amplitudes[all_max_amplitudes > 0]
if len(amplitudes_nonzero) > 0:
    ax_b.hist(amplitudes_nonzero, bins=50, edgecolor='black', linewidth=1.5, 
              alpha=0.7, color='#f39c12')
    ax_b.axvline(np.mean(amplitudes_nonzero), color='blue', linestyle='--', linewidth=2.5,
                 label=f'Mean: {np.mean(amplitudes_nonzero):.4f}')
    ax_b.axvline(np.median(amplitudes_nonzero), color='green', linestyle='--', linewidth=2.5,
                 label=f'Median: {np.median(amplitudes_nonzero):.4f}')
    ax_b.axvline(np.max(amplitudes_nonzero), color='red', linestyle='--', linewidth=2.5,
                 label=f'Max: {np.max(amplitudes_nonzero):.4f}')
    ax_b.set_xlabel('Maximum bump amplitude Δθ (m³/m³)', fontsize=17, fontfamily='Arial', labelpad=10)
    ax_b.set_ylabel('Frequency', fontsize=17, fontfamily='Arial', labelpad=10)
    ax_b.set_title('(b) Distribution of maximum bump amplitudes', fontsize=17, fontfamily='Arial', pad=12, fontweight='normal')
    ax_b.legend(fontsize=17, loc='upper right')
    ax_b.grid(False)
    ax_b.set_xscale('log')
    ax_b.set_xlim(amplitudes_nonzero.min() * 0.8, amplitudes_nonzero.max() * 1.2)
else:
    ax_b.text(0.5, 0.5, 'No bumps detected', ha='center', va='center', fontsize=18)
    ax_b.set_title('(b) Distribution of Maximum Bump Amplitudes', fontsize=20, pad=12, fontweight='normal')

# Leave third column of top row empty (or we can span panel b across 2 columns)
# Actually, let's make panel b span 2 columns for better visibility
ax_b.remove()
ax_b = fig.add_subplot(gs[0, 1:3])  # Span columns 1-2
amplitudes_nonzero = all_max_amplitudes[all_max_amplitudes > 0]
if len(amplitudes_nonzero) > 0:
    ax_b.hist(amplitudes_nonzero, bins=50, edgecolor='black', linewidth=1.5, 
              alpha=0.7, color='#f39c12')
    ax_b.axvline(np.mean(amplitudes_nonzero), color='blue', linestyle='--', linewidth=2.5,
                 label=f'Mean: {np.mean(amplitudes_nonzero):.4f}')
    ax_b.axvline(np.median(amplitudes_nonzero), color='green', linestyle='--', linewidth=2.5,
                 label=f'Median: {np.median(amplitudes_nonzero):.4f}')
    ax_b.axvline(np.max(amplitudes_nonzero), color='red', linestyle='--', linewidth=2.5,
                 label=f'Max: {np.max(amplitudes_nonzero):.4f}')
    ax_b.set_xlabel('Maximum bump amplitude Δθ (m³/m³)', fontsize=17, fontfamily='Arial', labelpad=10)
    ax_b.set_ylabel('Frequency', fontsize=17, fontfamily='Arial', labelpad=10)
    ax_b.set_title('(b) Distribution of maximum bump amplitudes', fontsize=17, fontfamily='Arial', pad=12, fontweight='normal')
    ax_b.legend(fontsize=17, loc='upper right')
    ax_b.grid(False)
    ax_b.set_xscale('log')
    ax_b.set_xlim(amplitudes_nonzero.min() * 0.8, amplitudes_nonzero.max() * 1.2)

# ============================================================================
# Row 2: Representative Failure Curves (Panels c, d, e)
# ============================================================================

def plot_failure_curve(ax, sample_idx, panel_label, title_suffix, highlight_violations=True):
    """Plot a single GB failure curve with clean violation highlighting.

    Violation approach:
    - The GB prediction curve is drawn in two visual layers:
        • Normal (monotone) segments: dashed grey line
        • Non-monotone segments: solid bold orange-red line (immediately visible)
    - Up to 5 filled triangle markers label the worst violations on the curve.
    - No axvspan shading (eliminates the wall-of-red-lines problem).
    """
    from matplotlib.lines import Line2D

    theta_gb = y_gb[sample_idx]
    theta_true = y_test[sample_idx]

    # ── Violation detection ──────────────────────────────────────────────────
    diff = theta_gb[:-1] - theta_gb[1:]          # positive = good (θ decreases)
    violations = np.append(diff < -1e-8, False)  # length == len(psi)
    n_viol = int(np.sum(violations[:-1]))         # number of violating intervals
    viol_amplitudes = np.where(violations[:-1], -diff, 0.0)

    # ── Observed curve ───────────────────────────────────────────────────────
    ax.semilogx(psi, theta_true, color='black', linestyle='-',
                linewidth=2.0, zorder=4)

    # ── GB prediction: segment into monotone vs violating stretches ──────────
    # Build a mask array for the END of each segment (index i covers psi[i]→psi[i+1])
    # We colour each point by whether the *preceding* interval was a violation.
    # Strategy: iterate over contiguous same-type runs and plot each run.
    seg_mask = np.append(violations[:-1], violations[-2] if len(violations) > 1 else False)
    # Plot monotone parts first (all at once won't work for segmented colouring;
    # use a simple loop over contiguous runs).
    i = 0
    n = len(psi)
    first_ok_seg = True
    first_bad_seg = True
    while i < n - 1:
        is_viol = bool(violations[i])
        j = i + 1
        while j < n - 1 and bool(violations[j]) == is_viol:
            j += 1
        # Segment covers psi[i:j+1], theta_gb[i:j+1]
        seg_psi = psi[i:j + 1]
        seg_theta = theta_gb[i:j + 1]
        if is_viol:
            lbl = '_nolegend_' if not first_bad_seg else '_nolegend_'
            ax.semilogx(seg_psi, seg_theta, color='#E74C3C', linestyle='-',
                        linewidth=2.5, zorder=3, label=lbl)
            first_bad_seg = False
        else:
            lbl = '_nolegend_'
            ax.semilogx(seg_psi, seg_theta, color='#7F8C8D', linestyle='--',
                        linewidth=1.5, zorder=2, label=lbl)
            first_ok_seg = False
        i = j

    # ── Top-5 triangle markers at worst violations ───────────────────────────
    if highlight_violations and n_viol > 0:
        top_k = min(5, n_viol)
        top_indices = np.argsort(viol_amplitudes)[-top_k:]
        for v_idx in top_indices:
            mid_psi = np.sqrt(psi[v_idx] * psi[v_idx + 1])
            mid_theta = (theta_gb[v_idx] + theta_gb[v_idx + 1]) / 2
            ax.plot(mid_psi, mid_theta, '^', color='#C0392B',
                    markersize=8, zorder=6, markeredgecolor='white',
                    markeredgewidth=0.5, label='_nolegend_')

    # ── Boundary lines ───────────────────────────────────────────────────────
    theta_s = X_test.iloc[sample_idx]['theta_s']
    theta_r = X_test.iloc[sample_idx]['theta_r']
    ax.axhline(theta_s, color='#27AE60', linestyle=':', linewidth=1.5,
               alpha=0.7, zorder=1)
    ax.axhline(theta_r, color='#E67E22', linestyle=':', linewidth=1.5,
               alpha=0.7, zorder=1)

    # ── Custom legend ─────────────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color='black',  linestyle='-',  linewidth=2.0, label='Observed'),
        Line2D([0], [0], color='#7F8C8D', linestyle='--', linewidth=1.5, label='GB prediction (monotone)'),
        Line2D([0], [0], color='#E74C3C', linestyle='-',  linewidth=2.5, label=f'GB violation ({n_viol} segments)'),
        Line2D([0], [0], color='#27AE60', linestyle=':', linewidth=1.5, label=r'$\theta_s$ (saturated)'),
        Line2D([0], [0], color='#E67E22', linestyle=':', linewidth=1.5, label=r'$\theta_r$ (residual)'),
    ]
    leg = ax.legend(handles=legend_handles, fontsize=9.5, loc='upper right',
                    framealpha=0.95, facecolor='white', edgecolor='#AAAAAA',
                    handlelength=2.0)
    for t in leg.get_texts():
        t.set_fontfamily('Arial')

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xlabel('Matric suction ψ (kPa)', fontsize=11, fontfamily='Arial', labelpad=8)
    ax.set_ylabel('Water content θ (m³/m³)', fontsize=11, fontfamily='Arial', labelpad=8)
    ax.set_title(f'({panel_label}) {title_suffix}',
                 fontsize=11, fontfamily='Arial', pad=10, fontweight='normal')
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)

    # Decade ticks — match Figure 3 / Figure 11 (plain numeric)
    _psi_ticks = np.array([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    _psi_ticklabels = ["0.1", "1.0", "10", "100", "1000", "10000", "100000", "1000000"]
    ax.set_xticks(_psi_ticks)
    ax.set_xticklabels(_psi_ticklabels, fontsize=10, fontfamily='Arial',
                       rotation=35, ha='right')
    ax.tick_params(axis='y', labelsize=10)
    for tick in ax.get_yticklabels():
        tick.set_fontfamily('Arial')

    # ── Stats annotation ──────────────────────────────────────────────────────
    max_amp = all_max_amplitudes[sample_idx]
    ax.text(0.02, 0.04,
            f'{n_viol} violation segments\nmax Δθ = {max_amp:.4f}',
            transform=ax.transAxes, fontsize=9, fontfamily='Arial',
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.90,
                      edgecolor='#E74C3C', linewidth=1.0))

# Panel (c): Typical Failure (median bumps)
ax_c = fig.add_subplot(gs[1, 0])
plot_failure_curve(ax_c, typical_idx, 'c', 'Typical violation case', highlight_violations=True)

# Panel (d): Maximum Amplitude Failure
ax_d = fig.add_subplot(gs[1, 1])
plot_failure_curve(ax_d, max_amp_idx, 'd', 'Largest violation amplitude', highlight_violations=True)

# Panel (e): Highest Frequency Failure (max bumps)
ax_e = fig.add_subplot(gs[1, 2])
plot_failure_curve(ax_e, max_bumps_idx, 'e', 'Most frequent violations', highlight_violations=True)

# Save figure
plt.savefig(OUTPUT_DIR / "Figure_S5_GB_Monotonicity_Violations.png", dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "Figure_S5_GB_Monotonicity_Violations.pdf", dpi=300, bbox_inches='tight')

# Standalone exports for readability (panels c, d, e)
def _save_s5_panel_failure_curve(sample_idx, panel_label, title_suffix, out_stem):
    fig_p, ax_p = plt.subplots(figsize=(6.5, 5.0))
    plot_failure_curve(ax_p, sample_idx, panel_label, title_suffix, highlight_violations=True)
    # Slightly less rotation on the larger standalone panel
    for lbl in ax_p.get_xticklabels():
        lbl.set_rotation(0)
        lbl.set_ha('center')
    fig_p.tight_layout(pad=0.6)
    fig_p.savefig(OUTPUT_DIR / f"{out_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig_p.savefig(OUTPUT_DIR / f"{out_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig_p)

_save_s5_panel_failure_curve(typical_idx, "c", "Typical violation case", "Figure_S5c_PanelC")
_save_s5_panel_failure_curve(max_amp_idx, "d", "Largest violation amplitude", "Figure_S5d_PanelD")
_save_s5_panel_failure_curve(max_bumps_idx, "e", "Most frequent violations", "Figure_S5e_PanelE")
plt.close()

print(f"  ✓ Saved: {OUTPUT_DIR / 'Figure_S5_GB_Monotonicity_Violations.png'}")
print(f"  ✓ Saved: {OUTPUT_DIR / 'Figure_S5_GB_Monotonicity_Violations.pdf'}")

print("\n" + "="*80)
print("FIGURE S5 GENERATION COMPLETE")
print("="*80)
print(f"\nFigure S5 includes:")
print(f"  - Panel (a): Distribution of bump counts (0-35 bumps)")
print(f"  - Panel (b): Distribution of maximum bump amplitudes")
print(f"  - Panel (c): Summary statistics text box")
print(f"  - Panel (d): Typical failure curve (median bumps)")
print(f"  - Panel (e): Maximum amplitude failure curve")
print(f"  - Panel (f): Highest frequency failure curve (max bumps)")
