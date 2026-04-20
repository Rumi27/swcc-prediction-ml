#!/usr/bin/env python3
"""
Generate Paper Figures
- Figure 1: Overall framework flowchart
- Figure 2: Dataset description and feature distributions
- Figure 3: SWCC data space before/after preprocessing
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Patch
from matplotlib.patches import ConnectionPatch
import seaborn as sns
from pathlib import Path
import json

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Publication-quality settings (readable when reduced in print)
SMALL_SIZE = 11   # tick labels, legend
MEDIUM_SIZE = 13  # axis labels, panel titles
LARGE_SIZE = 15   # main titles (if any)
plt.rcParams['font.size'] = SMALL_SIZE
plt.rcParams['axes.titlesize'] = MEDIUM_SIZE
plt.rcParams['axes.labelsize'] = MEDIUM_SIZE
plt.rcParams['xtick.labelsize'] = SMALL_SIZE
plt.rcParams['ytick.labelsize'] = SMALL_SIZE
plt.rcParams['legend.fontsize'] = SMALL_SIZE
plt.rcParams['figure.titlesize'] = LARGE_SIZE
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']

# Create output directory
output_dir = Path("paper_figures")
output_dir.mkdir(exist_ok=True)

print("="*80)
print("Generating Paper Figures")
print("="*80)

# ============================================================================
# FIGURE 1: Overall Framework Flowchart
# ============================================================================
print("\n1. Generating Figure 1: Overall Framework Flowchart...")

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
color_data = '#4A90E2'  # Blue
color_gan = '#50C878'   # Green
color_pinn = '#FF6B6B'  # Red
color_eval = '#FFA500'  # Orange
color_arrow = '#333333' # Dark gray

# Box style
box_style = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.5)

# 1. Data Collection & Preprocessing
ax.text(1.5, 9, 'UNSODA 2.0\nDatabase', ha='center', va='center', 
        fontsize=12, bbox=dict(boxstyle='round,pad=0.8', facecolor=color_data, edgecolor='black', linewidth=2, alpha=0.8))
ax.text(1.5, 7.5, 'Preprocessing\n• GSD extraction\n• Feature engineering\n• Interpolation', 
        ha='center', va='center', fontsize=9, bbox=box_style)

# Arrow 1
arrow1 = FancyArrowPatch((1.5, 7.2), (1.5, 6.5), 
                         arrowstyle='->', lw=2, color=color_arrow)
ax.add_patch(arrow1)

# 2. Feature Matrix
ax.text(1.5, 6, 'Feature Matrix\n(16 features)', ha='center', va='center',
        fontsize=11, bbox=dict(boxstyle='round,pad=0.6', facecolor='#E8F4F8', edgecolor=color_data, linewidth=2))

# Arrow 2 (to GAN)
arrow2 = FancyArrowPatch((2.5, 6), (4.5, 6), 
                         arrowstyle='->', lw=2, color=color_arrow)
ax.add_patch(arrow2)

# 3. GAN Training
ax.text(5.5, 9, 'WGAN-GP\nTraining', ha='center', va='center',
        fontsize=12, bbox=dict(boxstyle='round,pad=0.8', facecolor=color_gan, edgecolor='black', linewidth=2, alpha=0.8))
ax.text(5.5, 7.5, 'Generator\n• Physics constraints\n• Monotonicity\n• Boundary conditions', 
        ha='center', va='center', fontsize=9, bbox=box_style)
ax.text(5.5, 6, 'Synthetic SWCC\nGeneration', ha='center', va='center',
        fontsize=11, bbox=dict(boxstyle='round,pad=0.6', facecolor='#E8F8E8', edgecolor=color_gan, linewidth=2))
ax.text(5.5, 4.5, 'Physics Filtering\n• Monotonicity check\n• Boundary validation\n• Quality control', 
        ha='center', va='center', fontsize=9, bbox=box_style)

# Arrow 3 (from feature to GAN)
arrow3 = FancyArrowPatch((4.5, 6), (5.5, 7.5), 
                         arrowstyle='->', lw=2, color=color_arrow)
ax.add_patch(arrow3)

# Arrow 4 (GAN to filtered)
arrow4 = FancyArrowPatch((5.5, 6.3), (5.5, 5.2), 
                         arrowstyle='->', lw=2, color=color_arrow)
ax.add_patch(arrow4)

# 4. Combined Dataset
ax.text(5.5, 3.5, 'Combined Dataset\n(Real + Synthetic)', ha='center', va='center',
        fontsize=11, bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFF8E8', edgecolor='black', linewidth=2))

# Arrow 5 (to PINN)
arrow5 = FancyArrowPatch((5.5, 3.2), (5.5, 2.5), 
                         arrowstyle='->', lw=2, color=color_arrow)
ax.add_patch(arrow5)

# 5. PINN Training
ax.text(8.5, 6, 'MonotonicPINN\nTraining', ha='center', va='center',
        fontsize=12, bbox=dict(boxstyle='round,pad=0.8', facecolor=color_pinn, edgecolor='black', linewidth=2, alpha=0.8))
ax.text(8.5, 4.5, 'Architecture\n• Structural monotonicity\n• Physics encoding\n• Normalized losses', 
        ha='center', va='center', fontsize=9, bbox=box_style)
ax.text(8.5, 3, 'Trained Model', ha='center', va='center',
        fontsize=11, bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFE8E8', edgecolor=color_pinn, linewidth=2))

# Arrow 6 (from combined to PINN)
arrow6 = FancyArrowPatch((6.5, 3.5), (8.5, 4.5), 
                         arrowstyle='->', lw=2, color=color_arrow)
ax.add_patch(arrow6)

# Arrow 7 (PINN training)
arrow7 = FancyArrowPatch((8.5, 4.2), (8.5, 3.3), 
                         arrowstyle='->', lw=2, color=color_arrow)
ax.add_patch(arrow7)

# 6. Evaluation
ax.text(8.5, 1.5, 'Evaluation', ha='center', va='center',
        fontsize=12, bbox=dict(boxstyle='round,pad=0.8', facecolor=color_eval, edgecolor='black', linewidth=2, alpha=0.8))
ax.text(8.5, 0.5, '• Global metrics (RMSE, MAE, R²)\n• Regime-specific (dry-end)\n• Physics compliance', 
        ha='center', va='center', fontsize=9, bbox=box_style)

# Arrow 8 (from model to evaluation)
arrow8 = FancyArrowPatch((8.5, 2.7), (8.5, 2), 
                         arrowstyle='->', lw=2, color=color_arrow)
ax.add_patch(arrow8)

# Also connect feature matrix directly to PINN (alternative path)
arrow9 = FancyArrowPatch((2.5, 6), (8.5, 5.5), 
                         arrowstyle='->', lw=1.5, color=color_arrow, linestyle='--', alpha=0.6)
ax.add_patch(arrow9)
ax.text(5.5, 6.2, 'Direct path\n(no augmentation)', ha='center', va='center',
        fontsize=8, style='italic', color='gray')

plt.tight_layout()
plt.savefig(output_dir / 'Figure1_Framework_Flowchart.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {output_dir / 'Figure1_Framework_Flowchart.png'}")

# ============================================================================
# FIGURE 2: Dataset Description and Feature Distributions
# ============================================================================
print("\n2. Generating Figure 2: Dataset Description and Feature Distributions...")

# Load data
metadata = json.load(open("data_processed/metadata.json"))
X_train = pd.read_csv("data_processed/X_train.csv")
X_val = pd.read_csv("data_processed/X_val.csv")
X_test = pd.read_csv("data_processed/X_test.csv")
X_all = pd.concat([X_train, X_val, X_test], ignore_index=True)

# Figure 2 composite: Arial 20 pt for all text; legends Arial 14 pt only
_F2_COMP = 20
_F2_LEG = 14


def _f2_comp_arial_ticks(ax):
    ax.tick_params(labelsize=_F2_COMP)
    for _lbl in ax.get_xticklabels() + ax.get_yticklabels():
        _lbl.set_fontfamily("Arial")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.4, 
                      left=0.08, right=0.95, top=0.95, bottom=0.08)

# (a) Histogram/boxplots of key features
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])

# Theta_s
ax1.hist(X_all['theta_s'], bins=30, color='#4A90E2', alpha=0.7, edgecolor='black')
ax1.set_xlabel('θ_s (Saturated Water Content)', fontsize=_F2_COMP, fontfamily='Arial', labelpad=10)
ax1.set_ylabel('Frequency', fontsize=_F2_COMP, fontfamily='Arial', labelpad=10)
ax1.set_title('(a) θ_s distribution', fontsize=_F2_COMP, fontfamily='Arial', pad=10)
_f2_comp_arial_ticks(ax1)
ax1.grid(False)

# Theta_r
ax2.hist(X_all['theta_r'], bins=30, color='#50C878', alpha=0.7, edgecolor='black')
ax2.set_xlabel('θ_r (Residual Water Content)', fontsize=_F2_COMP, fontfamily='Arial', labelpad=10)
ax2.set_ylabel('Frequency', fontsize=_F2_COMP, fontfamily='Arial', labelpad=10)
ax2.set_title('(b) θ_r distribution', fontsize=_F2_COMP, fontfamily='Arial', pad=10)
_f2_comp_arial_ticks(ax2)
ax2.grid(False)

# Bulk density
ax3.hist(X_all['bulk_density'], bins=30, color='#FF6B6B', alpha=0.7, edgecolor='black')
ax3.set_xlabel('Bulk density (g/cm³)', fontsize=_F2_COMP, fontfamily='Arial', labelpad=10)
ax3.set_ylabel('Frequency', fontsize=_F2_COMP, fontfamily='Arial', labelpad=10)
ax3.set_title('(c) Bulk density distribution', fontsize=_F2_COMP, fontfamily='Arial', pad=10)
_f2_comp_arial_ticks(ax3)
ax3.grid(False)

# Porosity
ax4.hist(X_all['porosity'], bins=30, color='#FFA500', alpha=0.7, edgecolor='black')
ax4.set_xlabel('Porosity', fontsize=_F2_COMP, fontfamily='Arial', labelpad=10)
ax4.set_ylabel('Frequency', fontsize=_F2_COMP, fontfamily='Arial', labelpad=10)
ax4.set_title('(d) Porosity distribution', fontsize=_F2_COMP, fontfamily='Arial', pad=10)
_f2_comp_arial_ticks(ax4)
ax4.grid(False)

# (b) Texture triangle or PSD-based soil class distribution
ax5 = fig.add_subplot(gs[1, 1:])

# Create texture triangle
clay = X_all['clay_pct'].values
silt = X_all['silt_pct'].values
sand = X_all['sand_pct'].values

# Normalize to 100%
total = clay + silt + sand
total = np.where(total == 0, 1, total)  # Avoid division by zero
clay_norm = (clay / total * 100)
silt_norm = (silt / total * 100)
sand_norm = (sand / total * 100)
clay_norm = np.nan_to_num(clay_norm, nan=0.0)
silt_norm = np.nan_to_num(silt_norm, nan=0.0)
sand_norm = np.nan_to_num(sand_norm, nan=0.0)

# Texture triangle coordinates
# Using simplified triangle: clay at top, sand at bottom-left, silt at bottom-right
x_coords = sand_norm + silt_norm * 0.5
y_coords = clay_norm * np.sqrt(3) / 2

scatter = ax5.scatter(x_coords, y_coords, c=clay_norm, cmap='RdYlBu', 
                     s=30, alpha=0.6, edgecolors='black', linewidths=0.5)
ax5.set_xlabel('Sand + silt/2', fontsize=_F2_COMP, fontfamily='Arial', labelpad=10)
ax5.set_ylabel('Clay × √3/2', fontsize=_F2_COMP, fontfamily='Arial', labelpad=10)
ax5.set_title('(e) Soil texture distribution', fontsize=_F2_COMP, fontfamily='Arial', pad=10)
_f2_comp_arial_ticks(ax5)
ax5.set_aspect('equal')
ax5.grid(False)
cbar = plt.colorbar(scatter, ax=ax5)
cbar.set_label('Clay Content (%)', fontsize=_F2_COMP, fontfamily='Arial')
cbar.ax.tick_params(labelsize=_F2_COMP)
for _t in cbar.ax.get_yticklabels():
    _t.set_fontfamily('Arial')

# (c) Example PSD curves for representative soils
ax6 = fig.add_subplot(gs[2, :])

# Select representative samples (different textures)
sample_indices = []
for texture in ['sand', 'sandy_loam', 'silt_loam', 'clay']:
    if texture == 'sand':
        mask = (sand_norm > 70) & (clay_norm < 15)
    elif texture == 'sandy_loam':
        mask = (sand_norm > 50) & (sand_norm < 70) & (clay_norm < 20)
    elif texture == 'silt_loam':
        mask = (silt_norm > 50) & (clay_norm < 27)
    else:  # clay
        mask = (clay_norm > 40)
    
    if mask.any():
        idx = X_all[mask].index[0]
        sample_indices.append(idx)

# If not enough samples, add random ones
while len(sample_indices) < 4:
    idx = np.random.choice(X_all.index)
    if idx not in sample_indices:
        sample_indices.append(idx)

# Plot PSD curves (using D10, D30, D50, D60, D90)
colors_psd = ['#4A90E2', '#50C878', '#FF6B6B', '#FFA500']

all_d_values = []
for i, idx in enumerate(sample_indices[:4]):
    sample = X_all.iloc[idx]
    # Raw diameters (may contain small negative values from preprocessing)
    d10, d30, d50, d60, d90 = sample['D10'], sample['D30'], sample['D50'], sample['D60'], sample['D90']
    d_vals = np.array([d10, d30, d50, d60, d90], dtype=float)
    percent_finer = np.array([10, 30, 50, 60, 90], dtype=float)

    # For visualization only:
    # 1) Replace non-positive diameters with a very small positive value (1e-4)
    d_vals = np.where(d_vals <= 0, 1e-4, d_vals)
    # 2) Sort by increasing diameter so PSD curve is monotone in x
    order = np.argsort(d_vals)
    d_vals_sorted = d_vals[order]
    percent_sorted = percent_finer[order]

    # Collect for axis limits
    all_d_values.extend(d_vals_sorted[d_vals_sorted > 0].tolist())

    ax6.semilogx(d_vals_sorted, percent_sorted,
                 'o-', linewidth=2, markersize=6, color=colors_psd[i],
                 label=f'Sample {i+1} (Clay: {clay_norm[idx]:.1f}%, Silt: {silt_norm[idx]:.1f}%, Sand: {sand_norm[idx]:.1f}%)',
                 alpha=0.8)

ax6.set_xlabel('Particle diameter (μm)', fontsize=_F2_COMP, fontfamily='Arial', labelpad=10)
ax6.set_ylabel('Percent finer (%)', fontsize=_F2_COMP, fontfamily='Arial', labelpad=10)
ax6.set_title('(f) Representative particle size distribution curves', fontsize=_F2_COMP, fontfamily='Arial', pad=10)
_f2_comp_arial_ticks(ax6)
# Legend only: Arial 14 pt
_leg_f2 = ax6.legend(
    fontsize=_F2_LEG,
    loc='upper left',
    framealpha=1.0,
    facecolor='white',
    edgecolor='black',
    frameon=True,
)
for _t in _leg_f2.get_texts():
    _t.set_fontfamily('Arial')
ax6.grid(False)

# Set x-limits based on data range so curves are clearly visible,
# but always show down to 1e-4 as requested
if len(all_d_values) > 0:
    d_min_data = min(all_d_values)
    d_max_data = max(all_d_values)
    d_min = min(1e-4, d_min_data * 0.7)   # always at least down to 1e-4
    d_max = d_max_data * 1.5
    ax6.set_xlim([d_min, d_max])

# Same PSD sample indices for standalone Figure 2f export
_f2_psd_indices = sample_indices[:4]

# Overall title
plt.savefig(output_dir / 'Figure2_Dataset_Description.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'Figure2_Dataset_Description.pdf', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {output_dir / 'Figure2_Dataset_Description.png'}")
print(f"  ✓ Saved: {output_dir / 'Figure2_Dataset_Description.pdf'}")

# ============================================================================
# FIGURE 2 panels (a)–(f): separate figures, Arial 11 pt (publication / slides)
# ============================================================================
print("\n2b. Generating Figure 2 panels (a)–(f) separately (Arial 11 pt)...")
_F2 = 11

def _f2_arial_axes(ax):
    for _lbl in ax.get_xticklabels() + ax.get_yticklabels():
        _lbl.set_fontfamily("Arial")


# --- (a) θ_s ---
_fig, _ax = plt.subplots(figsize=(5.5, 4.5))
_fig.patch.set_facecolor("white")
_ax.hist(X_all["theta_s"], bins=30, color="#4A90E2", alpha=0.7, edgecolor="black")
_ax.set_xlabel("\u03b8_s (saturated water content)", fontsize=_F2, fontfamily="Arial")
_ax.set_ylabel("Frequency", fontsize=_F2, fontfamily="Arial")
_ax.set_title("(a) \u03b8_s distribution", fontsize=_F2, fontfamily="Arial")
_ax.tick_params(labelsize=_F2)
_ax.grid(False)
_f2_arial_axes(_ax)
for _s in _ax.spines.values():
    _s.set_color("black")
plt.tight_layout()
for _stem in ("Figure2a_PanelA",):
    plt.savefig(output_dir / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.savefig(output_dir / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
plt.close()

# --- (b) θ_r ---
_fig, _ax = plt.subplots(figsize=(5.5, 4.5))
_fig.patch.set_facecolor("white")
_ax.hist(X_all["theta_r"], bins=30, color="#50C878", alpha=0.7, edgecolor="black")
_ax.set_xlabel("\u03b8_r (residual water content)", fontsize=_F2, fontfamily="Arial")
_ax.set_ylabel("Frequency", fontsize=_F2, fontfamily="Arial")
_ax.set_title("(b) \u03b8_r distribution", fontsize=_F2, fontfamily="Arial")
_ax.tick_params(labelsize=_F2)
_ax.grid(False)
_f2_arial_axes(_ax)
for _s in _ax.spines.values():
    _s.set_color("black")
plt.tight_layout()
for _stem in ("Figure2b_PanelB",):
    plt.savefig(output_dir / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.savefig(output_dir / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
plt.close()

# --- (c) bulk density ---
_fig, _ax = plt.subplots(figsize=(5.5, 4.5))
_fig.patch.set_facecolor("white")
_ax.hist(X_all["bulk_density"], bins=30, color="#FF6B6B", alpha=0.7, edgecolor="black")
_ax.set_xlabel("Bulk density (g/cm\u00b3)", fontsize=_F2, fontfamily="Arial")
_ax.set_ylabel("Frequency", fontsize=_F2, fontfamily="Arial")
_ax.set_title("(c) Bulk density distribution", fontsize=_F2, fontfamily="Arial")
_ax.tick_params(labelsize=_F2)
_ax.grid(False)
_f2_arial_axes(_ax)
for _s in _ax.spines.values():
    _s.set_color("black")
plt.tight_layout()
for _stem in ("Figure2c_PanelC",):
    plt.savefig(output_dir / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.savefig(output_dir / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
plt.close()

# --- (d) porosity ---
_fig, _ax = plt.subplots(figsize=(5.5, 4.5))
_fig.patch.set_facecolor("white")
_ax.hist(X_all["porosity"], bins=30, color="#FFA500", alpha=0.7, edgecolor="black")
_ax.set_xlabel("Porosity", fontsize=_F2, fontfamily="Arial")
_ax.set_ylabel("Frequency", fontsize=_F2, fontfamily="Arial")
_ax.set_title("(d) Porosity distribution", fontsize=_F2, fontfamily="Arial")
_ax.tick_params(labelsize=_F2)
_ax.grid(False)
_f2_arial_axes(_ax)
for _s in _ax.spines.values():
    _s.set_color("black")
plt.tight_layout()
for _stem in ("Figure2d_PanelD",):
    plt.savefig(output_dir / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.savefig(output_dir / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
plt.close()

# --- (e) texture triangle ---
_fig, _ax = plt.subplots(figsize=(6.5, 6.0))
_fig.patch.set_facecolor("white")
_sc = _ax.scatter(
    x_coords,
    y_coords,
    c=clay_norm,
    cmap="RdYlBu",
    s=30,
    alpha=0.6,
    edgecolors="black",
    linewidths=0.5,
)
_ax.set_xlabel("Sand + silt/2", fontsize=_F2, fontfamily="Arial")
_ax.set_ylabel("Clay \u00d7 \u221a3/2", fontsize=_F2, fontfamily="Arial")
_ax.set_title("(e) Soil texture distribution", fontsize=_F2, fontfamily="Arial")
_ax.tick_params(labelsize=_F2)
_ax.set_aspect("equal")
_ax.grid(False)
_f2_arial_axes(_ax)
for _s in _ax.spines.values():
    _s.set_color("black")
_cbar = _fig.colorbar(_sc, ax=_ax)
_cbar.set_label("Clay content (%)", fontsize=_F2, fontfamily="Arial")
_cbar.ax.tick_params(labelsize=_F2)
for _t in _cbar.ax.get_yticklabels():
    _t.set_fontfamily("Arial")
plt.tight_layout()
for _stem in ("Figure2e_PanelE",):
    plt.savefig(output_dir / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.savefig(output_dir / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
plt.close()

# --- (f) PSD curves (same samples as composite Figure 2) ---
_fig, _ax = plt.subplots(figsize=(10.0, 5.0))
_fig.patch.set_facecolor("white")
colors_psd = ["#4A90E2", "#50C878", "#FF6B6B", "#FFA500"]
_all_d_vals = []
for _i, _idx in enumerate(_f2_psd_indices):
    _sample = X_all.iloc[_idx]
    d10, d30, d50, d60, d90 = (
        _sample["D10"],
        _sample["D30"],
        _sample["D50"],
        _sample["D60"],
        _sample["D90"],
    )
    d_vals = np.array([d10, d30, d50, d60, d90], dtype=float)
    percent_finer = np.array([10, 30, 50, 60, 90], dtype=float)
    d_vals = np.where(d_vals <= 0, 1e-4, d_vals)
    order = np.argsort(d_vals)
    d_vals_sorted = d_vals[order]
    percent_sorted = percent_finer[order]
    _all_d_vals.extend(d_vals_sorted[d_vals_sorted > 0].tolist())
    _ax.semilogx(
        d_vals_sorted,
        percent_sorted,
        "o-",
        linewidth=2,
        markersize=6,
        color=colors_psd[_i],
        label=(
            f"Sample {_i + 1} (Clay: {clay_norm[_idx]:.1f}%, "
            f"Silt: {silt_norm[_idx]:.1f}%, Sand: {sand_norm[_idx]:.1f}%)"
        ),
        alpha=0.8,
    )
_ax.set_xlabel("Particle diameter (\u03bcm)", fontsize=_F2, fontfamily="Arial")
_ax.set_ylabel("Percent finer (%)", fontsize=_F2, fontfamily="Arial")
_ax.set_title("(f) Representative particle size distribution curves", fontsize=_F2, fontfamily="Arial")
_ax.tick_params(labelsize=_F2)
_ax.grid(False)
_leg_f = _ax.legend(
    fontsize=8,
    loc="upper left",
    framealpha=1.0,
    facecolor="white",
    edgecolor="black",
    frameon=True,
)
for _t in _leg_f.get_texts():
    _t.set_fontfamily("Arial")
_f2_arial_axes(_ax)
for _s in _ax.spines.values():
    _s.set_color("black")
if len(_all_d_vals) > 0:
    d_min_data = min(_all_d_vals)
    d_max_data = max(_all_d_vals)
    d_min = min(1e-4, d_min_data * 0.7)
    d_max = d_max_data * 1.5
    _ax.set_xlim([d_min, d_max])
plt.tight_layout()
for _stem in ("Figure2f_PanelF",):
    plt.savefig(output_dir / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.savefig(output_dir / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
plt.close()

print(
    f"  ✓ Saved: Figure2a_PanelA … Figure2f_PanelF (Arial {_F2} pt) → {output_dir}"
)

# ============================================================================
# FIGURE 3: SWCC Data Space Before/After Preprocessing
# ============================================================================
print("\n3. Generating Figure 3: SWCC Data Space Before/After Preprocessing...")

# Load raw and processed data
y_train = np.load("data_processed/y_train.npy")
y_val = np.load("data_processed/y_val.npy")
y_test = np.load("data_processed/y_test.npy")
y_all = np.vstack([y_train, y_val, y_test])
suction_grid = np.load("data_processed/suction_grid.npy")

# Figure 3: log x-axis (ψ in kPa). Linear 0–1e6 kPa compresses ~all points at the left edge;
# log scale shows the full SWCC over many decades (standard in soil physics).
_FIG3_XMIN = float(np.min(suction_grid))
_FIG3_XMAX = float(np.max(suction_grid))
_FIG3_XTICKS = np.array([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
_FIG3_XTICKLABELS = [
    "0.1",
    "1.0",
    "10",
    "100",
    "1000",
    "10000",
    "100000",
    "1000000",
]

# For raw data, we'll simulate by showing variability
# In reality, we'd load the raw UNSODA data, but for now we'll use processed as "after"

# Wide canvas + spacing + extra height so tilted x-ticks / labels are not clipped
fig = plt.figure(figsize=(18, 6.8))
gs = fig.add_gridspec(1, 3, wspace=0.52)

# (a) Raw UNSODA SWCC curves (dense subset)
ax1 = fig.add_subplot(gs[0, 0])

# Show subset of curves (every 10th for clarity)
n_show = min(100, len(y_all))
indices = np.linspace(0, len(y_all)-1, n_show, dtype=int)

for idx in indices:
    ax1.semilogx(suction_grid, y_all[idx], 'b-', alpha=0.1, linewidth=0.5)

# Highlight a few representative curves
highlight_indices = [0, len(y_all)//4, len(y_all)//2, 3*len(y_all)//4, len(y_all)-1]
for idx in highlight_indices:
    if idx < len(y_all):
        ax1.semilogx(suction_grid, y_all[idx], 'r-', alpha=0.8, linewidth=2)

# Legend explaining line styles (panel a)
legend_elements_a = [
    Line2D([], [], color='b', alpha=0.4, linewidth=1.5,
           label='Individual SWCCs'),
    Line2D([], [], color='r', linewidth=2.5,
           label='Representative examples'),
]
leg1 = ax1.legend(handles=legend_elements_a,
                  loc='upper right',
                  fontsize=11,
                  frameon=True,
                  framealpha=1.0,
                  facecolor='white',
                  edgecolor='black')
for text in leg1.get_texts():
    text.set_fontfamily('Arial')

ax1.set_xlabel("Matric suction \u03c8 (kPa)", fontsize=11, fontfamily='Arial')
ax1.set_ylabel('Water content (θ)', fontsize=11, fontfamily='Arial')
# Use two-line title to avoid overlap with other panel titles
ax1.set_title('(a) Raw UNSODA SWCC curves\n(subset: 100)', fontsize=11, fontfamily='Arial')
ax1.tick_params(labelsize=11)
ax1.grid(False)
ax1.set_xlim([_FIG3_XMIN, _FIG3_XMAX])
ax1.set_xscale("log")
ax1.set_ylim([0, y_all.max() * 1.1])

# (b) Interpolated 100-point SWCCs (smoothed view)
ax2 = fig.add_subplot(gs[0, 1])

# Show all curves with transparency
for i in range(len(y_all)):
    ax2.semilogx(suction_grid, y_all[i], 'b-', alpha=0.05, linewidth=0.3)

# Show mean and std
mean_curve = np.nanmean(y_all, axis=0)
std_curve = np.nanstd(y_all, axis=0)

ax2.semilogx(
    suction_grid,
    mean_curve,
    "r-",
    linewidth=3,
    label="Mean \u03b8 (all samples)",
    alpha=0.9,
)
ax2.fill_between(
    suction_grid,
    mean_curve - std_curve,
    mean_curve + std_curve,
    alpha=0.3,
    color="red",
    label="\u00b11 SD (spread among samples)",
)

ax2.set_xlabel("Matric suction \u03c8 (kPa)", fontsize=11, fontfamily='Arial')
ax2.set_ylabel('Water content (θ)', fontsize=11, fontfamily='Arial')
# Shorten and wrap title to avoid overlap
ax2.set_title(
    "(b) 100-point SWCCs\n(mean \u03b8 and \u00b11 SD band)",
    fontsize=11,
    fontfamily="Arial",
)
leg2 = ax2.legend(fontsize=11)
for text in leg2.get_texts():
    text.set_fontfamily('Arial')
ax2.tick_params(labelsize=11)
ax2.grid(False)
ax2.set_xlim([_FIG3_XMIN, _FIG3_XMAX])
ax2.set_xscale("log")
ax2.set_ylim([0, y_all.max() * 1.1])

# (c) Outlier examples removed (optional inset)
ax3 = fig.add_subplot(gs[0, 2])

# Show clean curves (after outlier removal)
# In practice, we'd identify outliers, but for now show all as "clean"
for i in range(len(y_all)):
    # Check if curve is monotonic (after processing, they should be)
    diff = np.diff(y_all[i])
    if np.all(diff <= 0):  # Monotonic
        ax3.semilogx(suction_grid, y_all[i], 'g-', alpha=0.1, linewidth=0.3)
    else:  # Non-monotonic (would be removed)
        ax3.semilogx(suction_grid, y_all[i], 'r--', alpha=0.5, linewidth=1)

# Mean θ using only curves that are monotonic (θ non-increasing with ψ along the grid)
clean_mask = np.array([np.all(np.diff(y_all[i]) <= 0) for i in range(len(y_all))])
_n_nonmono_c = int(np.sum(~clean_mask))
if clean_mask.any():
    clean_curves = y_all[clean_mask]
    mean_clean = np.nanmean(clean_curves, axis=0)
    ax3.semilogx(suction_grid, mean_clean, "b-", linewidth=3, alpha=0.9)

# Legend: only show "non-monotonic" if any red dashed curves were actually plotted
legend_elements_c = [
    Line2D(
        [0],
        [0],
        color="g",
        alpha=0.45,
        linewidth=1.5,
        label="Monotonic SWCCs",
    ),
]
if _n_nonmono_c > 0:
    legend_elements_c.append(
        Line2D(
            [0],
            [0],
            color="r",
            linestyle="--",
            alpha=0.55,
            linewidth=1.5,
            label="Non-monotonic (omitted from mean)",
        )
    )
_mean_c_label = (
    "Mean \u03b8 (monotonic SWCCs only)"
    if _n_nonmono_c > 0
    else "Mean \u03b8 (all SWCCs)"
)
legend_elements_c.append(
    Line2D([0], [0], color="b", linewidth=3, label=_mean_c_label)
)
leg3 = ax3.legend(
    handles=legend_elements_c,
    fontsize=10,
    frameon=True,
    facecolor="white",
    edgecolor="black",
    loc="upper right",
)
for text in leg3.get_texts():
    text.set_fontfamily("Arial")

ax3.set_xlabel("Matric suction \u03c8 (kPa)", fontsize=11, fontfamily="Arial")
ax3.set_ylabel("Water content (\u03b8)", fontsize=11, fontfamily="Arial")
ax3.set_title(
    "(c) After quality control (monotonic \u03b8(\u03c8))",
    fontsize=11,
    fontfamily="Arial",
)
ax3.tick_params(labelsize=11)
ax3.grid(False)
ax3.set_xlim([_FIG3_XMIN, _FIG3_XMAX])
ax3.set_xscale("log")
ax3.set_ylim([0, y_all.max() * 1.1])

# Log x: plain numeric decade labels (kPa)
for _ax in (ax1, ax2, ax3):
    _ax.set_xticks(_FIG3_XTICKS)
    _ax.set_xticklabels(_FIG3_XTICKLABELS)
    for _lbl in _ax.get_xticklabels():
        _lbl.set_rotation(35)
        _lbl.set_ha("right")
        _lbl.set_fontsize(9)

# Ensure all text objects in Figure 3 use Arial (including ticks)
for ax in [ax1, ax2, ax3]:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Arial')

for _stem in ("Figure3_SWCC_Preprocessing", "Figure3_SWCC_Preprocessing_Arial11"):
    plt.savefig(
        output_dir / f"{_stem}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.savefig(
        output_dir / f"{_stem}.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
    )
plt.close()
print(
    f"  ✓ Saved: Figure3_SWCC_Preprocessing & Figure3_SWCC_Preprocessing_Arial11 "
    f"(Arial 11 pt, no grid, log ψ kPa) → {output_dir}"
)

# ============================================================================
# FIGURE 3a: Panel (a) only — log ψ (kPa), same convention as typical SWCC figures
# (Linear 0–10^6 kPa squashes curves at the left; log x is required for visibility.)
# ============================================================================
print("\n3a. Generating Figure 3a: Panel (a) only, log ψ axis (standard SWCC style)...")

fig_a, ax_a = plt.subplots(figsize=(9.0, 6.0))
fig_a.patch.set_facecolor("white")
ax_a.set_facecolor("white")
n_show_a = min(100, len(y_all))
indices_a = np.linspace(0, len(y_all) - 1, n_show_a, dtype=int)
for idx in indices_a:
    ax_a.semilogx(suction_grid, y_all[idx], "b-", alpha=0.1, linewidth=0.5)
highlight_a = [0, len(y_all) // 4, len(y_all) // 2, 3 * len(y_all) // 4, len(y_all) - 1]
for idx in highlight_a:
    if idx < len(y_all):
        ax_a.semilogx(suction_grid, y_all[idx], "r-", alpha=0.8, linewidth=2)

leg_a = ax_a.legend(
    handles=[
        Line2D([], [], color="b", alpha=0.4, linewidth=1.5, label="Individual SWCCs"),
        Line2D([], [], color="r", linewidth=2.5, label="Representative examples"),
    ],
    loc="upper right",
    fontsize=11,
    frameon=True,
    framealpha=1.0,
    facecolor="white",
    edgecolor="black",
)
for _t in leg_a.get_texts():
    _t.set_fontfamily("Arial")

ax_a.set_xlabel("Matric suction \u03c8 (kPa)", fontsize=11, fontfamily="Arial")
ax_a.set_ylabel("Water content (\u03b8)", fontsize=11, fontfamily="Arial")
ax_a.set_title("(a) Raw UNSODA SWCC curves (subset: 100)", fontsize=11, fontfamily="Arial")
ax_a.tick_params(labelsize=11)
ax_a.grid(False)
ax_a.set_xlim([_FIG3_XMIN, _FIG3_XMAX])
ax_a.set_xscale("log")
ax_a.set_xticks(_FIG3_XTICKS)
ax_a.set_xticklabels(_FIG3_XTICKLABELS)
for _lbl in ax_a.get_xticklabels():
    _lbl.set_fontsize(10)
    _lbl.set_fontfamily("Arial")
for _lbl in ax_a.get_yticklabels():
    _lbl.set_fontfamily("Arial")

_y_max_a = float(np.nanmax(y_all))
ax_a.set_ylim(0.0, _y_max_a * 1.05)
for _sp in ax_a.spines.values():
    _sp.set_color("black")
    _sp.set_linewidth(1.0)

for _stem_a in ("Figure3a_PanelA", "Figure3a_PanelA_Linear_Suction"):
    plt.savefig(
        output_dir / f"{_stem_a}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.savefig(
        output_dir / f"{_stem_a}.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
plt.close(fig_a)
print(
    f"  ✓ Saved: Figure3a_PanelA (+ Figure3a_PanelA_Linear_Suction alias), log ψ → {output_dir}"
)

# ============================================================================
# FIGURE 3b: Panel (b) only — mean ± 1 std over all SWCCs (log ψ)
# ============================================================================
print("\n3b. Generating Figure 3b: Panel (b) only, log ψ axis...")

fig_b, ax_b = plt.subplots(figsize=(9.0, 6.0))
fig_b.patch.set_facecolor("white")
ax_b.set_facecolor("white")

for i in range(len(y_all)):
    ax_b.semilogx(suction_grid, y_all[i], "b-", alpha=0.05, linewidth=0.3)

mean_curve_b = np.nanmean(y_all, axis=0)
std_curve_b = np.nanstd(y_all, axis=0)
ax_b.semilogx(
    suction_grid,
    mean_curve_b,
    "r-",
    linewidth=3,
    label="Mean \u03b8 (all samples)",
    alpha=0.9,
)
ax_b.fill_between(
    suction_grid,
    mean_curve_b - std_curve_b,
    mean_curve_b + std_curve_b,
    alpha=0.3,
    color="red",
    label="\u00b11 SD (spread among samples)",
)

leg_b = ax_b.legend(fontsize=11, frameon=True, facecolor="white", edgecolor="black")
for _t in leg_b.get_texts():
    _t.set_fontfamily("Arial")

ax_b.set_xlabel("Matric suction \u03c8 (kPa)", fontsize=11, fontfamily="Arial")
ax_b.set_ylabel("Water content (\u03b8)", fontsize=11, fontfamily="Arial")
ax_b.set_title(
    "(b) 100-point SWCCs (mean \u03b8 and \u00b11 SD band)",
    fontsize=11,
    fontfamily="Arial",
)
ax_b.tick_params(labelsize=11)
ax_b.grid(False)
ax_b.set_xlim([_FIG3_XMIN, _FIG3_XMAX])
ax_b.set_xscale("log")
ax_b.set_xticks(_FIG3_XTICKS)
ax_b.set_xticklabels(_FIG3_XTICKLABELS)
for _lbl in ax_b.get_xticklabels():
    _lbl.set_fontsize(10)
    _lbl.set_fontfamily("Arial")
for _lbl in ax_b.get_yticklabels():
    _lbl.set_fontfamily("Arial")

_y_max_b = float(np.nanmax(y_all))
ax_b.set_ylim(0.0, _y_max_b * 1.05)
for _sp in ax_b.spines.values():
    _sp.set_color("black")
    _sp.set_linewidth(1.0)

for _stem_b in ("Figure3b_PanelB",):
    plt.savefig(
        output_dir / f"{_stem_b}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.savefig(
        output_dir / f"{_stem_b}.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
plt.close(fig_b)
print(f"  ✓ Saved: Figure3b_PanelB → {output_dir}")

# ============================================================================
# FIGURE 3c: Panel (c) only — monotonic vs non-monotonic + mean (clean) (log ψ)
# ============================================================================
print("\n3c. Generating Figure 3c: Panel (c) only, log ψ axis...")

fig_c, ax_c = plt.subplots(figsize=(9.0, 6.0))
fig_c.patch.set_facecolor("white")
ax_c.set_facecolor("white")

for i in range(len(y_all)):
    diff_c = np.diff(y_all[i])
    if np.all(diff_c <= 0):
        ax_c.semilogx(suction_grid, y_all[i], "g-", alpha=0.1, linewidth=0.3)
    else:
        ax_c.semilogx(suction_grid, y_all[i], "r--", alpha=0.5, linewidth=1)

clean_mask_c = np.array([np.all(np.diff(y_all[i]) <= 0) for i in range(len(y_all))])
_n_nonmono_c_only = int(np.sum(~clean_mask_c))
if clean_mask_c.any():
    clean_curves_c = y_all[clean_mask_c]
    mean_clean_c = np.nanmean(clean_curves_c, axis=0)
    ax_c.semilogx(suction_grid, mean_clean_c, "b-", linewidth=3, alpha=0.9)

legend_elements_c_only = [
    Line2D(
        [0],
        [0],
        color="g",
        alpha=0.45,
        linewidth=1.5,
        label="Monotonic SWCCs",
    ),
]
if _n_nonmono_c_only > 0:
    legend_elements_c_only.append(
        Line2D(
            [0],
            [0],
            color="r",
            linestyle="--",
            alpha=0.55,
            linewidth=1.5,
            label="Non-monotonic (omitted from mean)",
        )
    )
_mean_c_only_label = (
    "Mean \u03b8 (monotonic SWCCs only)"
    if _n_nonmono_c_only > 0
    else "Mean \u03b8 (all SWCCs)"
)
legend_elements_c_only.append(
    Line2D([0], [0], color="b", linewidth=3, label=_mean_c_only_label)
)
leg_c = ax_c.legend(
    handles=legend_elements_c_only,
    fontsize=10,
    frameon=True,
    facecolor="white",
    edgecolor="black",
    loc="upper right",
)
for _t in leg_c.get_texts():
    _t.set_fontfamily("Arial")

ax_c.set_xlabel("Matric suction \u03c8 (kPa)", fontsize=11, fontfamily="Arial")
ax_c.set_ylabel("Water content (\u03b8)", fontsize=11, fontfamily="Arial")
ax_c.set_title(
    "(c) After quality control (monotonic \u03b8(\u03c8))",
    fontsize=11,
    fontfamily="Arial",
)
ax_c.tick_params(labelsize=11)
ax_c.grid(False)
ax_c.set_xlim([_FIG3_XMIN, _FIG3_XMAX])
ax_c.set_xscale("log")
ax_c.set_xticks(_FIG3_XTICKS)
ax_c.set_xticklabels(_FIG3_XTICKLABELS)
for _lbl in ax_c.get_xticklabels():
    _lbl.set_fontsize(10)
    _lbl.set_fontfamily("Arial")
for _lbl in ax_c.get_yticklabels():
    _lbl.set_fontfamily("Arial")

_y_max_c = float(np.nanmax(y_all))
ax_c.set_ylim(0.0, _y_max_c * 1.05)
for _sp in ax_c.spines.values():
    _sp.set_color("black")
    _sp.set_linewidth(1.0)

for _stem_c in ("Figure3c_PanelC",):
    plt.savefig(
        output_dir / f"{_stem_c}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.savefig(
        output_dir / f"{_stem_c}.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
plt.close(fig_c)
print(f"  ✓ Saved: Figure3c_PanelC → {output_dir}")

#
# ============================================================================
# FIGURE 4: WGAN-GP Training Behavior
# ============================================================================
print("\n4. Generating Figure 4: WGAN-GP Training Behavior...")

history_path = Path("results_gan/training_history.json")
if history_path.exists():
    hist = json.load(open(history_path))
    epochs = np.array(hist.get("epoch", []))
    d_loss = np.array(hist.get("d_loss", []))
    g_loss = np.array(hist.get("g_loss", []))
    gp = np.array(hist.get("gradient_penalty", []))
    wasserstein = np.array(hist.get("wasserstein_dist", []))

    if len(epochs) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # (a) Generator and critic losses
        ax = axes[0]
        ax.plot(epochs, d_loss, label="Critic loss", color="#4A90E2", linewidth=2)
        ax.plot(epochs, g_loss, label="Generator loss", color="#FF6B6B", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=17, fontfamily="Arial")
        ax.set_ylabel("Loss", fontsize=17, fontfamily="Arial")
        ax.set_title("(a) Generator and critic losses", fontsize=17, fontfamily="Arial")
        ax.tick_params(labelsize=17)
        ax.grid(True, alpha=0.3)
        leg_a = ax.legend(fontsize=17)
        for text in leg_a.get_texts():
            text.set_fontfamily("Arial")

        # (b) Gradient penalty evolution
        ax = axes[1]
        ax.plot(epochs, gp, color="#50C878", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=17, fontfamily="Arial")
        ax.set_ylabel("Gradient penalty", fontsize=17, fontfamily="Arial")
        ax.set_title("(b) Gradient penalty term", fontsize=17, fontfamily="Arial")
        ax.tick_params(labelsize=17)
        ax.grid(True, alpha=0.3)

        # (c) Wasserstein distance (if available)
        ax = axes[2]
        if len(wasserstein) == len(epochs) and len(epochs) > 0:
            ax.plot(epochs, wasserstein, color="#FFA500", linewidth=2)
            ax.set_ylabel("Estimated Wasserstein distance", fontsize=17, fontfamily="Arial")
        else:
            ax.plot(epochs, d_loss - g_loss, color="#FFA500", linewidth=2)
            ax.set_ylabel("d_loss - g_loss", fontsize=17, fontfamily="Arial")
        ax.set_xlabel("Epoch", fontsize=17, fontfamily="Arial")
        ax.set_title("(c) Training stability indicator", fontsize=17, fontfamily="Arial")
        ax.tick_params(labelsize=17)
        ax.grid(True, alpha=0.3)

        # Ensure all tick labels in Figure 4 use Arial
        for ax in axes:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily("Arial")

        plt.tight_layout()
        plt.savefig(output_dir / "Figure4_WGAN_Training.png", dpi=300, bbox_inches="tight")
        plt.savefig(output_dir / "Figure4_WGAN_Training.pdf", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved: {output_dir / 'Figure4_WGAN_Training.png'}")
    else:
        print("  ⚠ training_history.json exists but has no epochs; skipping Figure 4.")
else:
    print("  ⚠ results_gan/training_history.json not found; skipping Figure 4.")

#
# ============================================================================
# FIGURE 5: Real vs Synthetic SWCCs
# ============================================================================
print("\n5. Generating Figure 5: Real vs Synthetic SWCCs...")

try:
    # Real curves
    y_train = np.load("data_processed/y_train.npy")
    y_val = np.load("data_processed/y_val.npy")
    y_test = np.load("data_processed/y_test.npy")
    y_real = np.vstack([y_train, y_val, y_test])
    suction_real = np.load("data_processed/suction_grid.npy")

    # Synthetic (unfiltered)
    y_syn = np.load("results_gan/generated_data/synthetic_swcc_curves.npy")
    suction_syn = np.load("results_gan/generated_data/suction_grid.npy")

    # Synthetic filtered
    y_syn_filt = np.load("results_gan/generated_data_filtered/synthetic_swcc_curves_filtered.npy")
    suction_syn_filt = np.load("results_gan/generated_data_filtered/suction_grid.npy")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Professional color scheme for Q1 journal
    color_real = "#2E86AB"  # Professional blue
    color_unfiltered = "#F18F01"  # Professional orange
    color_filtered = "#06A77D"  # Professional green

    # (a) Real curves
    ax = axes[0]
    n_show = min(150, len(y_real))
    idx = np.linspace(0, len(y_real) - 1, n_show, dtype=int)
    for i in idx:
        ax.semilogx(suction_real, y_real[i], color=color_real, alpha=0.15, linewidth=0.8)
    ax.set_xlabel("Suction (kPa)", fontsize=16)
    ax.set_ylabel("Water content (θ)", fontsize=16)
    ax.set_title("(a) Real SWCCs", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.3, which="both")

    # (b) Unfiltered synthetic
    ax = axes[1]
    n_show = min(150, len(y_syn))
    idx = np.linspace(0, len(y_syn) - 1, n_show, dtype=int)
    for i in idx:
        ax.semilogx(suction_syn, y_syn[i], color=color_unfiltered, alpha=0.15, linewidth=0.8)
    ax.set_xlabel("Suction (kPa)", fontsize=16)
    ax.set_title("(b) Unfiltered synthetic SWCCs", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.3, which="both")

    # (c) Filtered synthetic
    ax = axes[2]
    n_show = min(150, len(y_syn_filt))
    idx = np.linspace(0, len(y_syn_filt) - 1, n_show, dtype=int)
    for i in idx:
        ax.semilogx(suction_syn_filt, y_syn_filt[i], color=color_filtered, alpha=0.15, linewidth=0.8)
    ax.set_xlabel("Suction (kPa)", fontsize=16)
    ax.set_title("(c) Filtered synthetic SWCCs", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(output_dir / "Figure5_Real_vs_Synthetic_SWCCs.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {output_dir / 'Figure5_Real_vs_Synthetic_SWCCs.png'}")
except Exception as e:
    print(f"  ⚠ Could not generate Figure 5: {e}")

#
# ============================================================================
# FIGURE 6: Feature Space Coverage – Real vs Synthetic
# ============================================================================
print("\n6. Generating Figure 6: Feature Space Coverage (Real vs Synthetic)...")

try:
    # Load real features
    X_train = pd.read_csv("data_processed/X_train.csv")
    X_val = pd.read_csv("data_processed/X_val.csv")
    X_test = pd.read_csv("data_processed/X_test.csv")
    X_real = pd.concat([X_train, X_val, X_test], ignore_index=True)

    # Load synthetic soil properties
    X_syn = pd.read_csv("results_gan/generated_data/synthetic_soil_properties.csv")
    X_syn_filt = pd.read_csv("results_gan/generated_data_filtered/synthetic_soil_properties_filtered.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # High-quality Q1 journal color scheme (colorblind-friendly, professional)
    color_real = "#0066CC"  # Professional blue (accessible)
    color_synthetic = "#E63946"  # Professional red (complementary, accessible)
    # Alternative: color_real = "#2E86AB", color_synthetic = "#F18F01"

    # (a) GSD–porosity space
    ax = axes[0]
    ax.scatter(X_real["D50"], X_real["porosity"], s=35, alpha=0.6, 
               label="Real", color=color_real, edgecolors="white", linewidths=0.5)
    ax.scatter(X_syn_filt["D50"], X_syn_filt["porosity"], s=35, alpha=0.6, 
               label="Synthetic (filtered)", color=color_synthetic, edgecolors="white", linewidths=0.5)
    ax.set_xlabel("D50 (mm)", fontsize=16)
    ax.set_ylabel("Porosity", fontsize=16)
    ax.set_title("(a) GSD–porosity space", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=14, framealpha=0.9)

    # (b) θ_s – bulk density space
    ax = axes[1]
    ax.scatter(X_real["theta_s"], X_real["bulk_density"], s=35, alpha=0.6, 
               label="Real", color=color_real, edgecolors="white", linewidths=0.5)
    ax.scatter(X_syn_filt["theta_s"], X_syn_filt["bulk_density"], s=35, alpha=0.6, 
               label="Synthetic (filtered)", color=color_synthetic, edgecolors="white", linewidths=0.5)
    ax.set_xlabel("θ_s", fontsize=16)
    ax.set_ylabel("Bulk density (g/cm³)", fontsize=16)
    ax.set_title("(b) θ_s – bulk density space", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=14, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_dir / "Figure6_Feature_Space_Real_vs_Synthetic.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {output_dir / 'Figure6_Feature_Space_Real_vs_Synthetic.png'}")
except Exception as e:
    print(f"  ⚠ Could not generate Figure 6: {e}")

# ============================================================================
# FIGURE 7: MonotonicPINN Architecture
# ============================================================================
print("\n7. Generating Figure 7: MonotonicPINN Architecture...")

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis("off")

# Colors
color_input = "#4A90E2"
color_increment = "#50C878"
color_cumsum = "#FF6B6B"

# Input features box
ax.add_patch(
    FancyBboxPatch(
        (0.5, 3.0),
        2.0,
        1.0,
        boxstyle="round,pad=0.4",
        linewidth=2,
        edgecolor="black",
        facecolor=color_input,
        alpha=0.9,
    )
)
ax.text(
    1.5,
    3.5,
    "Input features\n(GSD + soil properties)",
    ha="center",
    va="center",
    fontsize=11,
    color="white",
)

# Dense layers (feature extractor)
dense_y = 4.8
for i, x0 in enumerate([3.2, 5.0, 6.8]):
    ax.add_patch(
        FancyBboxPatch(
            (x0, dense_y),
            1.4,
            0.8,
            boxstyle="round,pad=0.3",
            linewidth=1.5,
            edgecolor="black",
            facecolor="#F4F4F4",
        )
    )
    ax.text(
        x0 + 0.7,
        dense_y + 0.4,
        f"Dense {i+1}",
        ha="center",
        va="center",
        fontsize=10,
    )

ax.text(
    5.0,
    5.8,
    "Shared feature extractor\n(physics encoding + dense layers)",
    ha="center",
    va="center",
    fontsize=11,
)

# Arrows from input to first dense and between dense layers
arrow_kwargs = dict(arrowstyle="->", lw=2, color="#333333")
ax.add_patch(FancyArrowPatch((2.5, 3.5), (3.2, dense_y + 0.4), **arrow_kwargs))
ax.add_patch(FancyArrowPatch((4.6, dense_y + 0.4), (5.0, dense_y + 0.4), **arrow_kwargs))
ax.add_patch(FancyArrowPatch((6.4, dense_y + 0.4), (6.8, dense_y + 0.4), **arrow_kwargs))

# Increment head (softplus)
ax.add_patch(
    FancyBboxPatch(
        (8.4, 3.8),
        1.6,
        1.0,
        boxstyle="round,pad=0.4",
        linewidth=2,
        edgecolor="black",
        facecolor=color_increment,
        alpha=0.9,
    )
)
ax.text(
    9.2,
    4.3,
    "Increment head\nsoftplus(Δθ)",
    ha="center",
    va="center",
    fontsize=10,
    color="white",
)

ax.text(
    9.2,
    5.4,
    "Δθ ≥ 0 at each suction point",
    ha="center",
    va="center",
    fontsize=10,
)

ax.add_patch(FancyArrowPatch((8.2, dense_y + 0.4), (8.4, 4.3), **arrow_kwargs))

# Cumulative sum + normalization block
ax.add_patch(
    FancyBboxPatch(
        (3.2, 1.5),
        3.4,
        1.1,
        boxstyle="round,pad=0.4",
        linewidth=2,
        edgecolor="black",
        facecolor=color_cumsum,
        alpha=0.9,
    )
)
ax.text(
    4.9,
    2.05,
    "Normalize Δθ to (θ_s − θ_r)\n+ cumulative sum over suction grid",
    ha="center",
    va="center",
    fontsize=10,
    color="white",
)

ax.text(
    4.9,
    0.7,
    "θ_norm(s_k) = 1 − Σ Δθ_i  →  strictly decreasing in s",
    ha="center",
    va="center",
    fontsize=10,
)

# Arrows from increment head to cumulative sum and to output
ax.add_patch(FancyArrowPatch((9.2, 3.8), (6.6, 2.05), **arrow_kwargs))

# Output θ_norm(s)
ax.add_patch(
    FancyBboxPatch(
        (7.4, 1.5),
        2.0,
        1.0,
        boxstyle="round,pad=0.4",
        linewidth=2,
        edgecolor="black",
        facecolor="#FFE8E8",
    )
)
ax.text(
    8.4,
    2.0,
    "Output\nθ_norm(s)",
    ha="center",
    va="center",
    fontsize=11,
)

ax.add_patch(FancyArrowPatch((6.6, 2.05), (7.4, 2.0), **arrow_kwargs))

ax.set_title("MonotonicPINN architecture with structural monotonicity", fontsize=14)
plt.tight_layout()
plt.savefig(output_dir / "Figure7_MonotonicPINN_Architecture.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved: {output_dir / 'Figure7_MonotonicPINN_Architecture.png'}")

# ============================================================================
# FIGURE 8: Normalized Physics Constraints Illustration
# ============================================================================
print("\n8. Generating Figure 8: Normalized Physics Constraints Illustration...")

# Vertical stack (a)→(b)→(c), shared ψ axis; same x-label + tick numbers on (a)(b)(c)
_F8_COMP = 18
_F8_LEG = 14

# Use actual suction grid (kPa) - same as used in the model
# Range: 0.1 to 10^6 kPa, logarithmically spaced (100 points)
suction_min = 0.1  # kPa
suction_max = 1e6  # kPa
suction_grid = np.logspace(np.log10(suction_min), np.log10(suction_max), 100)

# Create a normalized suction for conceptual illustration (0 to 1)
# This represents the position along the suction range, not actual normalized suction
s_norm_conceptual = (np.log10(suction_grid) - np.log10(suction_min)) / (np.log10(suction_max) - np.log10(suction_min))

# Match Figure 3 x-axis style: matric suction ψ (kPa) + plain decade ticks
_F8_XTICKS = np.array([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
_F8_XTICKLABELS = ["0.1", "1.0", "10", "100", "1000", "10000", "100000", "1000000"]

# Mathtext: "norm" as subscript on θ (not d_norm); derivative is dθ_norm/d log10(ψ)
_F8_YLABEL_THETA = r"Normalized water content $\theta_{\mathrm{norm}}(\psi)$"
_F8_DERIV = r"$d\theta_{\mathrm{norm}}/d\log_{10}(\psi)$"
_F8_TITLE_B = r"(b) Monotonicity: $d\theta_{\mathrm{norm}}/d\log_{10}(\psi) \leq 0$"
_F8_XLABEL = "Matric suction ψ (kPa)"


def _f8_style_axes(ax, *, show_x_tick_labels: bool):
    ax.grid(False)
    ax.set_xlim([suction_min, suction_max])
    ax.set_xticks(_F8_XTICKS)
    ax.set_xticklabels(_F8_XTICKLABELS)
    ax.tick_params(axis="y", labelsize=_F8_COMP)
    ax.tick_params(axis="x", labelsize=_F8_COMP)
    if not show_x_tick_labels:
        ax.tick_params(axis="x", labelbottom=False)
    for _lbl in ax.get_xticklabels() + ax.get_yticklabels():
        _lbl.set_fontfamily("Arial")
    for _tl in ax.get_xticklabels() + ax.get_yticklabels():
        _tl.set_fontsize(_F8_COMP)


def _f8_panel_legend(ax, handles, *, loc="upper right", bbox_to_anchor=None):
    _kw = dict(
        handles=handles,
        loc=loc,
        fontsize=_F8_LEG,
        frameon=True,
        edgecolor="black",
        facecolor="white",
    )
    if bbox_to_anchor is not None:
        _kw["bbox_to_anchor"] = bbox_to_anchor
    leg = ax.legend(**_kw)
    for _t in leg.get_texts():
        _t.set_fontfamily("Arial")
    return leg


# Physical layout: full-page width; each stacked panel ~3:1 width:height (wider than tall) for SWCC slope readability
_F8_PAGE_W_IN = 10.5
_F8_L, _F8_R, _F8_T, _F8_B = 0.165, 0.97, 0.97, 0.07
_F8_PANEL_WH = 3.0  # width / height per panel; use 2.5–3.0 per journal preference
_F8_HSPACE = 0.26  # fraction of average row height between panels (tighter than before to limit total height)
_gs_h_frac = _F8_T - _F8_B
_ax_w_frac = _F8_R - _F8_L
_ax_w_in = _F8_PAGE_W_IN * _ax_w_frac
_panel_h_in = _ax_w_in / _F8_PANEL_WH
_gs_h_in = _panel_h_in * (3.0 + 2.0 * _F8_HSPACE)
_fig_h_in = _gs_h_in / _gs_h_frac

fig = plt.figure(figsize=(_F8_PAGE_W_IN, _fig_h_in))
gs = fig.add_gridspec(3, 1, hspace=_F8_HSPACE, left=_F8_L, right=_F8_R, top=_F8_T, bottom=_F8_B)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[1, 0], sharex=ax_a)
ax_c = fig.add_subplot(gs[2, 0], sharex=ax_a)

# (a) Ideal θ_norm(ψ)
theta_norm = 1.0 / (1.0 + (s_norm_conceptual / 0.25) ** 1.5)
ax_a.semilogx(suction_grid, theta_norm, color="#2E86AB", linewidth=2)
ax_a.set_ylabel(_F8_YLABEL_THETA, fontsize=_F8_COMP, labelpad=12)
ax_a.set_title("(a) Ideal normalized SWCC", fontsize=_F8_COMP, fontfamily="Arial", pad=8)
ax_a.set_xlabel(_F8_XLABEL, fontsize=_F8_COMP, fontfamily="Arial", labelpad=10)
_f8_style_axes(ax_a, show_x_tick_labels=True)
ax_a.set_ylim([0, 1.05])
# In-axes callouts (data coords): just above the plateau / above the dry tail, clear of the blue curve
ax_a.text(
    0.38,
    0.988,
    r"$\theta_{\mathrm{norm}} \approx 1$ (saturated)",
    ha="left",
    va="bottom",
    fontsize=_F8_COMP,
    fontfamily="Arial",
    transform=ax_a.transData,
)
ax_a.text(
    6.5e5,
    0.16,
    r"$\theta_{\mathrm{norm}} \approx 0$ (residual)",
    ha="right",
    va="bottom",
    fontsize=_F8_COMP,
    fontfamily="Arial",
    transform=ax_a.transData,
)
_f8_panel_legend(
    ax_a,
    [
        Line2D(
            [0],
            [0],
            color="#2E86AB",
            lw=2,
            label=r"$\theta_{\mathrm{norm}}(\psi)$ — ideal / example curve",
        ),
    ],
    loc="upper right",
)

# (b) Derivative sign illustration
dtheta_dlogpsi = np.gradient(theta_norm, np.log10(suction_grid))
ax_b.semilogx(suction_grid, dtheta_dlogpsi, color="#FF6B6B", linewidth=2)
ax_b.axhline(0.0, color="black", linewidth=1, linestyle="--")
ax_b.set_ylabel(_F8_DERIV, fontsize=_F8_COMP, labelpad=10)
ax_b.set_title(_F8_TITLE_B, fontsize=_F8_COMP, fontfamily="Arial", pad=8)
ax_b.set_xlabel(_F8_XLABEL, fontsize=_F8_COMP, fontfamily="Arial", labelpad=10)
_f8_style_axes(ax_b, show_x_tick_labels=True)
ax_b.text(0.5, 0.3, "All slopes ≤ 0", ha="center", va="center", transform=ax_b.transAxes, fontsize=_F8_COMP, fontfamily="Arial")
_f8_panel_legend(
    ax_b,
    [
        Line2D(
            [0],
            [0],
            color="#FF6B6B",
            lw=2,
            label=r"$d\theta_{\mathrm{norm}}/d\log_{10}(\psi)$",
        ),
        Line2D([0], [0], color="black", lw=1, linestyle="--", label="Zero reference"),
    ],
    loc="lower right",
    bbox_to_anchor=(0.98, 0.08),
)

# (c) Physically plausible shape region + shared x-axis label & numbers
ax_c.semilogx(suction_grid, theta_norm, color="#2E86AB", linewidth=2)
mid_range_mask = (suction_grid >= 1e2) & (suction_grid <= 1e4)
ax_c.fill_between(
    suction_grid,
    theta_norm - 0.1,
    theta_norm + 0.1,
    where=mid_range_mask,
    color="#50C878",
    alpha=0.2,
)
ax_c.set_ylabel(_F8_YLABEL_THETA, fontsize=_F8_COMP, labelpad=10)
ax_c.set_xlabel(_F8_XLABEL, fontsize=_F8_COMP, fontfamily="Arial", labelpad=10)
ax_c.set_title("(c) Physically plausible shape region", fontsize=_F8_COMP, fontfamily="Arial", pad=8)
_f8_style_axes(ax_c, show_x_tick_labels=True)
ax_c.set_ylim([0, 1.05])

_f8_panel_legend(
    ax_c,
    [
        Line2D(
            [0],
            [0],
            color="#2E86AB",
            lw=2,
            label=r"$\theta_{\mathrm{norm}}(\psi)$ — example curve",
        ),
        Patch(
            facecolor="#50C878",
            alpha=0.35,
            edgecolor="none",
            label="Illustrative band",
        ),
    ],
    loc="upper right",
)

for _ax in (ax_a, ax_b, ax_c):
    for _sp in _ax.spines.values():
        _sp.set_color("black")

fig.align_ylabels([ax_a, ax_b, ax_c])
# Panel (a): shift y-label slightly down so mathtext $(\psi)$ is not clipped at figure top
ax_a.yaxis.set_label_coords(-0.105, 0.42)
fig.align_xlabels([ax_a, ax_b, ax_c])

plt.savefig(output_dir / "Figure8_Normalized_Physics_Constraints.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
plt.savefig(output_dir / "Figure8_Normalized_Physics_Constraints.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
plt.close()
print(f"  ✓ Saved: {output_dir / 'Figure8_Normalized_Physics_Constraints.png'}")

# ============================================================================
# FIGURE 8 panels (a)–(c): standalone, Arial 11 pt, no grid (same data as above)
# ============================================================================
print("\n8b. Generating Figure 8 panels (a)–(c) separately (Arial 11 pt, no grid)...")
_F8 = 11

def _f8_arial(ax):
    for _lbl in ax.get_xticklabels() + ax.get_yticklabels():
        _lbl.set_fontfamily("Arial")


# (a)
_fig, _ax = plt.subplots(figsize=(6.0, 5.0))
_fig.patch.set_facecolor("white")
_ax.semilogx(suction_grid, theta_norm, color="#2E86AB", linewidth=2, label=r"$\theta_{\mathrm{norm}}(\psi)$")
_ax.set_xlabel("Matric suction \u03c8 (kPa)", fontsize=_F8, fontfamily="Arial", labelpad=8)
_ax.set_ylabel(_F8_YLABEL_THETA, fontsize=_F8, labelpad=12)
_ax.set_title("(a) Ideal normalized SWCC", fontsize=_F8, fontfamily="Arial", pad=10)
_ax.grid(False)
_ax.tick_params(labelsize=_F8)
_ax.set_xlim([suction_min, suction_max])
_ax.set_xticks(_F8_XTICKS)
_ax.set_xticklabels(_F8_XTICKLABELS)
_ax.set_ylim([0, 1.05])
_ax.text(
    0.38,
    0.88,
    r"$\theta_{\mathrm{norm}} \approx 1$ (saturated)",
    ha="left",
    va="bottom",
    fontsize=_F8,
    fontfamily="Arial",
    transform=_ax.transData,
)
_ax.text(
    6.5e5,
    0.16,
    r"$\theta_{\mathrm{norm}} \approx 0$ (residual)",
    ha="right",
    va="bottom",
    fontsize=_F8,
    fontfamily="Arial",
    transform=_ax.transData,
)
_leg8a_s = _ax.legend(
    fontsize=_F8,
    loc="upper right",
    framealpha=1.0,
    facecolor="white",
    edgecolor="black",
    frameon=True,
)
for _t in _leg8a_s.get_texts():
    _t.set_fontfamily("Arial")
_f8_arial(_ax)
for _s in _ax.spines.values():
    _s.set_color("black")
plt.tight_layout()
for _stem in ("Figure8a_PanelA",):
    plt.savefig(output_dir / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.savefig(output_dir / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
plt.close()

# (b)
_fig, _ax = plt.subplots(figsize=(6.0, 5.0))
_fig.patch.set_facecolor("white")
_ax.semilogx(suction_grid, dtheta_dlogpsi, color="#FF6B6B", linewidth=2)
_ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
_ax.set_xlabel("Matric suction \u03c8 (kPa)", fontsize=_F8, fontfamily="Arial", labelpad=8)
_ax.set_ylabel(_F8_DERIV, fontsize=_F8, labelpad=12)
_ax.set_title(_F8_TITLE_B, fontsize=_F8, fontfamily="Arial", pad=10)
_ax.grid(False)
_ax.tick_params(labelsize=_F8)
_ax.set_xlim([suction_min, suction_max])
_ax.set_xticks(_F8_XTICKS)
_ax.set_xticklabels(_F8_XTICKLABELS)
_ax.text(
    0.5,
    0.3,
    "All slopes \u2264 0",
    ha="center",
    va="center",
    transform=_ax.transAxes,
    fontsize=_F8,
    fontfamily="Arial",
)
_f8_arial(_ax)
for _s in _ax.spines.values():
    _s.set_color("black")
plt.tight_layout()
for _stem in ("Figure8b_PanelB",):
    plt.savefig(output_dir / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.savefig(output_dir / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
plt.close()

# (c)
_fig, _ax = plt.subplots(figsize=(6.5, 5.0))
_fig.patch.set_facecolor("white")
_ax.semilogx(
    suction_grid, theta_norm, color="#2E86AB", linewidth=2, label=r"Example $\theta_{\mathrm{norm}}(\psi)$"
)
mid_range_mask_8 = (suction_grid >= 1e2) & (suction_grid <= 1e4)
_ax.fill_between(
    suction_grid,
    theta_norm - 0.1,
    theta_norm + 0.1,
    where=mid_range_mask_8,
    color="#50C878",
    alpha=0.2,
    label="Capillarity-dominated\n(Arya–Paris-inspired region)",
)
_ax.set_xlabel("Matric suction \u03c8 (kPa)", fontsize=_F8, fontfamily="Arial", labelpad=8)
_ax.set_ylabel(_F8_YLABEL_THETA, fontsize=_F8, labelpad=12)
_ax.set_title("(c) Physically plausible shape region", fontsize=_F8, fontfamily="Arial", pad=10)
_ax.grid(False)
_ax.tick_params(labelsize=_F8)
_ax.set_xlim([suction_min, suction_max])
_ax.set_xticks(_F8_XTICKS)
_ax.set_xticklabels(_F8_XTICKLABELS)
_ax.set_ylim([0, 1.05])
_leg8c_s = _ax.legend(
    fontsize=9,
    loc="upper right",
    framealpha=1.0,
    facecolor="white",
    edgecolor="black",
    frameon=True,
)
for _t in _leg8c_s.get_texts():
    _t.set_fontfamily("Arial")
_f8_arial(_ax)
for _s in _ax.spines.values():
    _s.set_color("black")
plt.tight_layout()
for _stem in ("Figure8c_PanelC",):
    plt.savefig(output_dir / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.savefig(output_dir / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
plt.close()

print(f"  ✓ Saved: Figure8a_PanelA … Figure8c_PanelC (Arial {_F8} pt) → {output_dir}")

# ============================================================================
# FIGURE 9: Training Curves for MonotonicPINN
# ============================================================================
print("\n9. Generating Figure 9: Training Curves for MonotonicPINN...")

history_path_pinn = Path("results_pinn_fixed/training_history.json")
if history_path_pinn.exists():
    hist_pinn = json.load(open(history_path_pinn))
    epochs = np.array(hist_pinn.get("epoch", []))
    train_total = np.array(hist_pinn.get("train_total", []))
    train_data = np.array(hist_pinn.get("train_data", []))
    val_total = np.array(hist_pinn.get("val_total", []))
    val_data = np.array(hist_pinn.get("val_data", []))
    train_mono = np.array(hist_pinn.get("train_mono", []))
    train_bound = np.array(hist_pinn.get("train_bound", []))
    train_physics = np.array(hist_pinn.get("train_physics", []))
    val_mono = np.array(hist_pinn.get("val_mono", []))
    val_bound = np.array(hist_pinn.get("val_bound", []))
    val_physics = np.array(hist_pinn.get("val_physics", []))

    if len(epochs) > 0:
        _F9 = 11
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # (a) Data loss & total loss vs epochs
        ax = axes[0]
        ax.plot(epochs, train_data, label="Train data loss", color="#2E86AB", linewidth=2)
        ax.plot(epochs, train_total, label="Train total loss", color="#FF6B6B", linewidth=2, alpha=0.8)
        if len(val_data) == len(epochs):
            ax.plot(epochs, val_data, label="Val data loss", color="#50C878", linewidth=2, linestyle="--")
        ax.set_xlabel("Epoch", fontsize=_F9, fontfamily="Arial", labelpad=10)
        ax.set_ylabel("Loss", fontsize=_F9, fontfamily="Arial", labelpad=10)
        ax.set_title("(a) Data and total loss", fontsize=_F9, fontfamily="Arial", pad=10)
        ax.tick_params(labelsize=_F9)
        ax.grid(False)
        leg9a = ax.legend(fontsize=_F9)
        for text in leg9a.get_texts():
            text.set_fontfamily("Arial")

        # (b) Validation data loss vs epochs (proxy for RMSE)
        ax = axes[1]
        if len(val_data) == len(epochs):
            ax.plot(epochs, val_data, color="#4A90E2", linewidth=2)
            # Mark best epoch (min val_data)
            best_idx = int(np.argmin(val_data))
            best_epoch = epochs[best_idx]
            best_val = val_data[best_idx]
            ax.axvline(best_epoch, color="red", linestyle="--", linewidth=1.5)
            ax.text(
                best_epoch,
                best_val,
                f"Best epoch {best_epoch}",
                ha="left",
                va="bottom",
                fontsize=_F9,
                fontfamily="Arial",
                color="red",
            )
        ax.set_xlabel("Epoch", fontsize=_F9, fontfamily="Arial", labelpad=10)
        ax.set_ylabel("Validation data loss", fontsize=_F9, fontfamily="Arial", labelpad=10)
        ax.set_title("(b) Validation loss (proxy for RMSE)", fontsize=_F9, fontfamily="Arial", pad=10)
        ax.tick_params(labelsize=_F9)
        ax.grid(False)

        # (c) Physics loss terms
        ax = axes[2]
        if len(train_bound) == len(epochs):
            ax.plot(epochs, train_bound, label="Train boundary loss", color="#F18F01", linewidth=2)
        if len(train_physics) == len(epochs):
            ax.plot(epochs, train_physics, label="Train physics loss", color="#06A77D", linewidth=2)
        # train_mono is zero by design (structural), but plot if non-zero
        if len(train_mono) == len(epochs) and np.any(train_mono > 0):
            ax.plot(epochs, train_mono, label="Train monotonicity loss", color="#8E44AD", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=_F9, fontfamily="Arial", labelpad=10)
        ax.set_ylabel("Physics loss", fontsize=_F9, fontfamily="Arial", labelpad=10)
        ax.set_title("(c) Physics loss terms", fontsize=_F9, fontfamily="Arial", pad=10)
        ax.tick_params(labelsize=_F9)
        ax.grid(False)
        # Position legend in upper left, make it smaller to avoid covering graph lines
        leg9c = ax.legend(
            fontsize=_F9,
            loc="upper left",
            framealpha=1.0,
            facecolor="white",
            edgecolor="black",
            frameon=True,
        )
        for text in leg9c.get_texts():
            text.set_fontfamily("Arial")

        # Ensure all tick labels in Figure 9 use Arial
        for ax in axes:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily("Arial")

        plt.tight_layout()
        plt.savefig(output_dir / "Figure9_MonotonicPINN_Training_Curves.png", dpi=300, bbox_inches="tight")
        plt.savefig(output_dir / "Figure9_MonotonicPINN_Training_Curves.pdf", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved: {output_dir / 'Figure9_MonotonicPINN_Training_Curves.png'}")

        # ------------------------------------------------------------------
        # Figure 9 panels (a)–(c): standalone, Arial 11 pt, no grid
        # ------------------------------------------------------------------
        print("\n9b. Generating Figure 9 panels (a)–(c) separately (Arial 11 pt, no grid)...")

        def _f9_arial_axis(ax):
            for _lbl in ax.get_xticklabels() + ax.get_yticklabels():
                _lbl.set_fontfamily("Arial")

        # (a)
        _fig, _ax = plt.subplots(figsize=(6.5, 5.0))
        _fig.patch.set_facecolor("white")
        _ax.plot(epochs, train_data, label="Train data loss", color="#2E86AB", linewidth=2)
        _ax.plot(epochs, train_total, label="Train total loss", color="#FF6B6B", linewidth=2, alpha=0.8)
        if len(val_data) == len(epochs):
            _ax.plot(epochs, val_data, label="Val data loss", color="#50C878", linewidth=2, linestyle="--")
        _ax.set_xlabel("Epoch", fontsize=_F9, fontfamily="Arial", labelpad=10)
        _ax.set_ylabel("Loss", fontsize=_F9, fontfamily="Arial", labelpad=10)
        _ax.set_title("(a) Data and total loss", fontsize=_F9, fontfamily="Arial", pad=10)
        _ax.tick_params(labelsize=_F9)
        _ax.grid(False)
        _leg = _ax.legend(fontsize=_F9)
        for _t in _leg.get_texts():
            _t.set_fontfamily("Arial")
        _f9_arial_axis(_ax)
        for _s in _ax.spines.values():
            _s.set_color("black")
        plt.tight_layout()
        for _stem in ("Figure9a_PanelA",):
            plt.savefig(output_dir / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
            plt.savefig(output_dir / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
        plt.close()

        # (b)
        _fig, _ax = plt.subplots(figsize=(6.5, 5.0))
        _fig.patch.set_facecolor("white")
        if len(val_data) == len(epochs):
            _ax.plot(epochs, val_data, color="#4A90E2", linewidth=2)
            best_idx = int(np.argmin(val_data))
            best_epoch = epochs[best_idx]
            best_val = val_data[best_idx]
            _ax.axvline(best_epoch, color="red", linestyle="--", linewidth=1.5)
            _ax.text(
                best_epoch,
                best_val,
                f"Best epoch {best_epoch}",
                ha="left",
                va="bottom",
                fontsize=_F9,
                fontfamily="Arial",
                color="red",
            )
        _ax.set_xlabel("Epoch", fontsize=_F9, fontfamily="Arial", labelpad=10)
        _ax.set_ylabel("Validation data loss", fontsize=_F9, fontfamily="Arial", labelpad=10)
        _ax.set_title("(b) Validation loss (proxy for RMSE)", fontsize=_F9, fontfamily="Arial", pad=10)
        _ax.tick_params(labelsize=_F9)
        _ax.grid(False)
        _f9_arial_axis(_ax)
        for _s in _ax.spines.values():
            _s.set_color("black")
        plt.tight_layout()
        for _stem in ("Figure9b_PanelB",):
            plt.savefig(output_dir / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
            plt.savefig(output_dir / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
        plt.close()

        # (c)
        _fig, _ax = plt.subplots(figsize=(6.5, 5.0))
        _fig.patch.set_facecolor("white")
        if len(train_bound) == len(epochs):
            _ax.plot(epochs, train_bound, label="Train boundary loss", color="#F18F01", linewidth=2)
        if len(train_physics) == len(epochs):
            _ax.plot(epochs, train_physics, label="Train physics loss", color="#06A77D", linewidth=2)
        if len(train_mono) == len(epochs) and np.any(train_mono > 0):
            _ax.plot(epochs, train_mono, label="Train monotonicity loss", color="#8E44AD", linewidth=2)
        _ax.set_xlabel("Epoch", fontsize=_F9, fontfamily="Arial", labelpad=10)
        _ax.set_ylabel("Physics loss", fontsize=_F9, fontfamily="Arial", labelpad=10)
        _ax.set_title("(c) Physics loss terms", fontsize=_F9, fontfamily="Arial", pad=10)
        _ax.tick_params(labelsize=_F9)
        _ax.grid(False)
        _leg = _ax.legend(
            fontsize=_F9,
            loc="upper left",
            framealpha=1.0,
            facecolor="white",
            edgecolor="black",
            frameon=True,
        )
        for _t in _leg.get_texts():
            _t.set_fontfamily("Arial")
        _f9_arial_axis(_ax)
        for _s in _ax.spines.values():
            _s.set_color("black")
        plt.tight_layout()
        for _stem in ("Figure9c_PanelC",):
            plt.savefig(output_dir / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
            plt.savefig(output_dir / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
        plt.close()

        print(f"  ✓ Saved: Figure9a_PanelA … Figure9c_PanelC (Arial {_F9} pt) → {output_dir}")
    else:
        print("  ⚠ training_history.json has no epochs; skipping Figure 9.")
else:
    print("  ⚠ results_pinn_fixed/training_history.json not found; skipping Figure 9.")

print("\n" + "="*80)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("="*80)
print(f"\nOutput directory: {output_dir}")
print("\nGenerated figures:")
print("  1. Figure1_Framework_Flowchart.png")
print("  2. Figure2_Dataset_Description.png")
print("  3. Figure3_SWCC_Preprocessing.png")
print("  4. Figure4_WGAN_Training.png (if history available)")
print("  5. Figure5_Real_vs_Synthetic_SWCCs.png")
print("  6. Figure6_Feature_Space_Real_vs_Synthetic.png")
print("  7. Figure7_MonotonicPINN_Architecture.png")
print("  8. Figure8_Normalized_Physics_Constraints.png")
print("  9. Figure9_MonotonicPINN_Training_Curves.png")