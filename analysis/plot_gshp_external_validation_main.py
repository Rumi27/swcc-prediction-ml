#!/usr/bin/env python3
"""
Generate Figure 17: External Validation on GSHP (Main Paper - Simplified)

This script creates a 2-panel figure for the main paper (Arial 11 pt, no grid; panel (b) matches Figure 11 ψ-axis):
(a) Distribution of per-sample RMSE (histogram with q10/q50/q90 lines)
(b) Representative curve overlays (median and high-error cases)
Outputs composite Figure17 plus standalone Figure17a_PanelA / Figure17b_PanelB.

Author: Generated for external validation figure
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
OUTPUT_DIR = ROOT_DIR / "paper_figures"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Style settings (match paper Figures 11–16: Arial 11 pt, no grid)
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
_F17 = 11
plt.rcParams['font.size'] = _F17
plt.rcParams['axes.labelsize'] = _F17
plt.rcParams['axes.titlesize'] = _F17
plt.rcParams['xtick.labelsize'] = _F17
plt.rcParams['ytick.labelsize'] = _F17
plt.rcParams['legend.fontsize'] = _F17
plt.rcParams['figure.dpi'] = 300


def _f17_spines_ticks_arial(ax):
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Arial")


def _f17_log_psi_axes_swcc(ax, suction_grid):
    """Matric suction ψ (kPa), water content θ, decade ticks — same as Figure 11 / Figure3a."""
    ax.set_xlabel("Matric suction \u03c8 (kPa)", fontsize=_F17, fontfamily="Arial", labelpad=10)
    ax.set_ylabel("Water content (\u03b8)", fontsize=_F17, fontfamily="Arial", labelpad=10)
    ax.set_xlim([float(np.min(suction_grid)), float(np.max(suction_grid))])
    ax.set_xscale("log")
    _ticks = np.array([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    _ticklabels = ["0.1", "1.0", "10", "100", "1000", "10000", "100000", "1000000"]
    ax.set_xticks(_ticks)
    ax.set_xticklabels(_ticklabels)
    for _lbl in ax.get_xticklabels():
        _lbl.set_fontsize(10)
        _lbl.set_fontfamily("Arial")


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
    print(f"  Curve shape: {curves_pred.shape}")
    print(f"  Suction grid: {len(suction_grid)} points")
    
    return results, curves_pred, curves_gshp, suction_grid


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


def plot_panel_a(ax, rmse_per_sample):
    """Panel (a): Clean RMSE distribution with 3 annotations."""
    print("\nPlotting panel (a): RMSE distribution...")

    ax.hist(
        rmse_per_sample,
        bins=50,
        density=True,
        alpha=0.75,
        color="#2E86AB",
        edgecolor="white",
        linewidth=0.5,
    )

    q10 = np.percentile(rmse_per_sample, 10)
    q50 = np.percentile(rmse_per_sample, 50)
    q90 = np.percentile(rmse_per_sample, 90)

    ax.axvline(q10, color="#06A77D", linestyle="--", linewidth=2.0, alpha=0.9, zorder=5)
    ax.axvline(q50, color="#F18F01", linestyle="--", linewidth=2.0, alpha=0.9, zorder=5)
    ax.axvline(q90, color="#C73E1D", linestyle="--", linewidth=2.0, alpha=0.9, zorder=5)

    ax.set_xlabel("Per-sample RMSE", fontsize=_F17, fontfamily="Arial", labelpad=10)
    ax.set_ylabel("Density", fontsize=_F17, fontfamily="Arial", labelpad=10)
    ax.set_title("(a) Distribution of per-sample RMSE", fontsize=_F17, fontfamily="Arial", pad=10)
    ax.tick_params(labelsize=_F17)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily("Arial")
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    y_min, y_max = ax.get_ylim()
    y_text = y_max * 0.92

    ax.text(
        q10,
        y_text,
        f"{q10:.3f}",
        ha="center",
        va="bottom",
        fontsize=_F17,
        color="#06A77D",
        fontweight="bold",
        fontfamily="Arial",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#06A77D", linewidth=1.2),
        zorder=6,
    )
    x_min, x_max = ax.get_xlim()
    x_shift = (x_max - x_min) * 0.07
    ax.text(
        q50 + x_shift,
        y_text,
        f"{q50:.3f}",
        ha="center",
        va="bottom",
        fontsize=_F17,
        color="#F18F01",
        fontweight="bold",
        fontfamily="Arial",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#F18F01", linewidth=1.2),
        zorder=6,
    )
    ax.text(
        q90,
        y_text,
        f"{q90:.3f}",
        ha="center",
        va="bottom",
        fontsize=_F17,
        color="#C73E1D",
        fontweight="bold",
        fontfamily="Arial",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#C73E1D", linewidth=1.2),
        zorder=6,
    )

    textstr = "N = 7,072\n\u03b1: 1/m \u2192 1/kPa (\u00d70.10197)"
    ax.text(
        0.98,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=_F17 - 1,
        verticalalignment="top",
        horizontalalignment="right",
        fontfamily="Arial",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.95, edgecolor="gray", linewidth=1),
    )


def plot_panel_b(ax, curves_pred, curves_gshp, suction_grid, rmse_per_sample):
    """Panel (b): 2 representative curves (median and high-error)."""
    print("\nPlotting panel (b): Representative curves...")

    q50_idx = np.argmin(np.abs(rmse_per_sample - np.percentile(rmse_per_sample, 50)))
    q90_idx = np.argmin(np.abs(rmse_per_sample - np.percentile(rmse_per_sample, 90)))

    print(f"  Median case (q50): sample {q50_idx}, RMSE = {rmse_per_sample[q50_idx]:.4f}")
    print(f"  High-error case (q90): sample {q90_idx}, RMSE = {rmse_per_sample[q90_idx]:.4f}")

    color_median = "#2E86AB"
    color_high = "#C73E1D"

    ax.plot(suction_grid, curves_gshp[q50_idx], color=color_median, linewidth=2.5, label="GSHP (median)", alpha=0.9, zorder=3)
    ax.plot(suction_grid, curves_pred[q50_idx], color=color_median, linewidth=2.0, linestyle="--", label="VGParamNet (median)", alpha=0.9, zorder=3)
    ax.plot(suction_grid, curves_gshp[q90_idx], color=color_high, linewidth=2.5, label="GSHP (high-error)", alpha=0.9, zorder=3)
    ax.plot(suction_grid, curves_pred[q90_idx], color=color_high, linewidth=2.0, linestyle="--", label="VGParamNet (high-error)", alpha=0.9, zorder=3)

    y_max = max(
        curves_gshp[q50_idx].max(),
        curves_pred[q50_idx].max(),
        curves_gshp[q90_idx].max(),
        curves_pred[q90_idx].max(),
    )
    ax.set_ylim(0, y_max * 1.12)

    legend = ax.legend(loc="upper right", framealpha=0.95, fontsize=_F17, ncol=1, facecolor="white", edgecolor="gray", frameon=True)
    for t in legend.get_texts():
        t.set_fontfamily("Arial")

    ax.text(
        0.98,
        0.88,
        f"High-error RMSE = {rmse_per_sample[q90_idx]:.4f}",
        transform=ax.transAxes,
        fontsize=_F17 - 1,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontfamily="Arial",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.95, edgecolor=color_high, linewidth=1.2),
    )
    ax.text(
        0.98,
        0.08,
        f"Median RMSE = {rmse_per_sample[q50_idx]:.4f}",
        transform=ax.transAxes,
        fontsize=_F17 - 1,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontfamily="Arial",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.95, edgecolor=color_median, linewidth=1.2),
    )

    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    _f17_log_psi_axes_swcc(ax, suction_grid)
    ax.set_title("(b) Representative curve overlays", fontsize=_F17, fontfamily="Arial", pad=10)
    ax.tick_params(labelsize=_F17)
    for tick in ax.get_yticklabels():
        tick.set_fontfamily("Arial")


def main():
    """Main function to generate Figure 17 (simplified)."""
    print("="*80)
    print("Generating Figure 17: External Validation on GSHP (Main Paper - Simplified)")
    print("="*80)
    
    # Load data
    results, curves_pred, curves_gshp, suction_grid = load_data()
    
    # Compute per-sample RMSE
    rmse_per_sample = compute_per_sample_rmse(curves_pred, curves_gshp)
    
    print("\nCreating figure...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_panel_a(axes[0], rmse_per_sample)
    plot_panel_b(axes[1], curves_pred, curves_gshp, suction_grid, rmse_per_sample)

    for ax in axes:
        _f17_spines_ticks_arial(ax)

    fig.suptitle(
        "External validation on GSHP (curve space)",
        fontsize=_F17,
        fontweight="normal",
        fontfamily="Arial",
        y=1.02,
    )

    plt.tight_layout(rect=[0, 0, 1.0, 0.96], pad=2.0)

    output_path = OUTPUT_DIR / "Figure17_GSHP_External_Validation.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\n✓ Saved figure to: {output_path}")

    output_path_pdf = OUTPUT_DIR / "Figure17_GSHP_External_Validation.pdf"
    plt.savefig(output_path_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved figure (PDF) to: {output_path_pdf}")
    plt.close()

    print("\nGenerating Figure 17 panels (a)–(b) separately (Arial 11 pt, no grid)...")
    _fig, _ax = plt.subplots(figsize=(6.5, 5.0))
    _fig.patch.set_facecolor("white")
    plot_panel_a(_ax, rmse_per_sample)
    _f17_spines_ticks_arial(_ax)
    plt.tight_layout()
    for _stem in ("Figure17a_PanelA",):
        plt.savefig(OUTPUT_DIR / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
        plt.savefig(OUTPUT_DIR / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close()

    _fig, _ax = plt.subplots(figsize=(6.5, 5.0))
    _fig.patch.set_facecolor("white")
    plot_panel_b(_ax, curves_pred, curves_gshp, suction_grid, rmse_per_sample)
    _f17_spines_ticks_arial(_ax)
    plt.tight_layout()
    for _stem in ("Figure17b_PanelB",):
        plt.savefig(OUTPUT_DIR / f"{_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
        plt.savefig(OUTPUT_DIR / f"{_stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close()

    print(f"  ✓ Saved: Figure17a_PanelA … Figure17b_PanelB (Arial {_F17} pt) → {OUTPUT_DIR}")

    print("\n" + "=" * 80)
    print("Figure generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
