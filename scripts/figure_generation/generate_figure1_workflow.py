#!/usr/bin/env python3
"""
Generate Figure 1: Overall Workflow Schematic
Shows the complete pipeline from data to evaluation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np
from pathlib import Path

# Set style
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
color_data = '#4A90E2'      # Blue
color_preprocess = '#50C878'  # Green
color_models = '#FF6B6B'     # Red
color_eval = '#FFA500'       # Orange
color_bg = '#F5F5F5'         # Light gray

# Define box positions and sizes
box_height = 1.2
box_width = 1.8
spacing_x = 1.5
spacing_y = 2.0

# Row 1: Data
y1 = 8.5
x_data = 1.0
data_box = FancyBboxPatch((x_data - box_width/2, y1 - box_height/2), 
                          box_width, box_height,
                          boxstyle="round,pad=0.1", 
                          facecolor=color_data, edgecolor='black', linewidth=1.5)
ax.add_patch(data_box)
ax.text(x_data, y1, 'UNSODA 2.0\nDatabase', ha='center', va='center', 
        fontsize=11, fontweight='normal', color='white')

# Row 2: Preprocessing
y2 = 6.5
x_preprocess = 1.0
preprocess_box = FancyBboxPatch((x_preprocess - box_width/2, y2 - box_height/2),
                                box_width, box_height,
                                boxstyle="round,pad=0.1",
                                facecolor=color_preprocess, edgecolor='black', linewidth=1.5)
ax.add_patch(preprocess_box)
ax.text(x_preprocess, y2, 'Data\nPreprocessing', ha='center', va='center',
        fontsize=11, fontweight='normal', color='white')

# Arrow from data to preprocessing
arrow1 = FancyArrowPatch((x_data, y1 - box_height/2), (x_preprocess, y2 + box_height/2),
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax.add_patch(arrow1)

# Row 3: Models (three boxes side by side)
y3 = 4.0
x_gb = 2.0
x_pinn = 5.0
x_vg = 8.0

# Gradient Boosting
gb_box = FancyBboxPatch((x_gb - box_width/2, y3 - box_height/2),
                        box_width, box_height,
                        boxstyle="round,pad=0.1",
                        facecolor=color_models, edgecolor='black', linewidth=1.5)
ax.add_patch(gb_box)
ax.text(x_gb, y3, 'Gradient\nBoosting', ha='center', va='center',
        fontsize=10, fontweight='normal', color='white')

# Monotone PINN
pinn_box = FancyBboxPatch((x_pinn - box_width/2, y3 - box_height/2),
                          box_width, box_height,
                          boxstyle="round,pad=0.1",
                          facecolor=color_models, edgecolor='black', linewidth=1.5)
ax.add_patch(pinn_box)
ax.text(x_pinn, y3, 'Monotone\nPINN', ha='center', va='center',
        fontsize=10, fontweight='normal', color='white')

# VGParamNet
vg_box = FancyBboxPatch((x_vg - box_width/2, y3 - box_height/2),
                        box_width, box_height,
                        boxstyle="round,pad=0.1",
                        facecolor=color_models, edgecolor='black', linewidth=1.5)
ax.add_patch(vg_box)
ax.text(x_vg, y3, 'VGParamNet', ha='center', va='center',
        fontsize=10, fontweight='normal', color='white')

# Arrows from preprocessing to models
arrow2a = FancyArrowPatch((x_preprocess, y2 - box_height/2), (x_gb, y3 + box_height/2),
                          arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax.add_patch(arrow2a)
arrow2b = FancyArrowPatch((x_preprocess, y2 - box_height/2), (x_pinn, y3 + box_height/2),
                          arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax.add_patch(arrow2b)
arrow2c = FancyArrowPatch((x_preprocess, y2 - box_height/2), (x_vg, y3 + box_height/2),
                          arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax.add_patch(arrow2c)

# Row 4: Evaluation/Diagnostics (four boxes)
y4 = 1.5
x_eval1 = 1.5
x_eval2 = 3.5
x_eval3 = 5.5
x_eval4 = 7.5
eval_width = 1.4

# Admissibility
eval1_box = FancyBboxPatch((x_eval1 - eval_width/2, y4 - box_height/2),
                           eval_width, box_height,
                           boxstyle="round,pad=0.1",
                           facecolor=color_eval, edgecolor='black', linewidth=1.5)
ax.add_patch(eval1_box)
ax.text(x_eval1, y4, 'Admissibility\n(θ, K)', ha='center', va='center',
        fontsize=9, fontweight='normal', color='white')

# VG-fit
eval2_box = FancyBboxPatch((x_eval2 - eval_width/2, y4 - box_height/2),
                           eval_width, box_height,
                           boxstyle="round,pad=0.1",
                           facecolor=color_eval, edgecolor='black', linewidth=1.5)
ax.add_patch(eval2_box)
ax.text(x_eval2, y4, 'VG-fit\nStability', ha='center', va='center',
        fontsize=9, fontweight='normal', color='white')

# Knee fidelity
eval3_box = FancyBboxPatch((x_eval3 - eval_width/2, y4 - box_height/2),
                           eval_width, box_height,
                           boxstyle="round,pad=0.1",
                           facecolor=color_eval, edgecolor='black', linewidth=1.5)
ax.add_patch(eval3_box)
ax.text(x_eval3, y4, 'Knee\nFidelity', ha='center', va='center',
        fontsize=9, fontweight='normal', color='white')

# Kr(ψ)
eval4_box = FancyBboxPatch((x_eval4 - eval_width/2, y4 - box_height/2),
                           eval_width, box_height,
                           boxstyle="round,pad=0.1",
                           facecolor=color_eval, edgecolor='black', linewidth=1.5)
ax.add_patch(eval4_box)
ax.text(x_eval4, y4, 'K_r(ψ)\nConsistency', ha='center', va='center',
        fontsize=9, fontweight='normal', color='white')

# Arrows from models to evaluation
for x_model in [x_gb, x_pinn, x_vg]:
    for x_eval in [x_eval1, x_eval2, x_eval3, x_eval4]:
        arrow = FancyArrowPatch((x_model, y3 - box_height/2), (x_eval, y4 + box_height/2),
                               arrowstyle='->', mutation_scale=15, linewidth=1.2, 
                               color='gray', alpha=0.4)
        ax.add_patch(arrow)

# Add labels for preprocessing details
preprocess_details = [
    '• SWCC interpolation\n• PSD extraction\n• Feature engineering\n• Quality filtering'
]
ax.text(x_preprocess + 1.5, y2, '\n'.join(preprocess_details), 
        ha='left', va='center', fontsize=9, style='italic',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8))

# Add labels for evaluation details
eval_details = [
    '• Monotonicity\n• Boundary conditions\n• Bump detection'
]
ax.text(x_eval1 - 0.3, y4 - 1.2, '\n'.join(eval_details),
        ha='center', va='top', fontsize=8, style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

# Title
ax.text(5, 9.5, 'Figure 1: Overall Workflow and Evaluation Framework', 
        ha='center', va='top', fontsize=14, fontweight='bold')

# Save
output_dir = Path(__file__).parent.parent.parent / 'paper_figures'
output_dir.mkdir(exist_ok=True)
output_path = output_dir / 'Figure1_Workflow_Schematic.png'
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Figure 1 saved to: {output_path}")
plt.close()
