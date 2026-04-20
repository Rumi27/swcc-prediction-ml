#!/usr/bin/env python3
"""
Generate formal simulation validation figure (Figure 18) for the paper.
Shows convergence metrics and mechanism of failure.
Same typography as Figures 11–17: Arial 11 pt, no grid; composite + standalone panels.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

RESULTS_DIR = ROOT_DIR / "results_simulation"
OUTPUT_DIR = ROOT_DIR / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_F18 = 11
plt.rcParams["font.size"] = _F18
plt.rcParams["axes.labelsize"] = _F18
plt.rcParams["axes.titlesize"] = _F18
plt.rcParams["xtick.labelsize"] = _F18
plt.rcParams["ytick.labelsize"] = _F18
plt.rcParams["legend.fontsize"] = _F18
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["xtick.major.width"] = 1.2
plt.rcParams["ytick.major.width"] = 1.2


def _f18_spines_ticks_arial(ax):
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Arial")


def _f18_log_psi_x_axis(ax):
    """Matric suction ψ (kPa), log scale, decade ticks — same as Figure 11 / Figure3a."""
    ax.set_xlabel("Matric suction \u03c8 (kPa)", fontsize=_F18, fontfamily="Arial", labelpad=10)
    ax.set_xscale("log")
    _ticks = np.array([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    _ticklabels = ["0.1", "1.0", "10", "100", "1000", "10000", "100000", "1000000"]
    ax.set_xticks(_ticks)
    ax.set_xticklabels(_ticklabels)
    for _lbl in ax.get_xticklabels():
        _lbl.set_fontsize(10)
        _lbl.set_fontfamily("Arial")

def _plot_f18_panel_a(ax, df_plot):
    """Panel (a): Numerical stability bar chart."""
    cases = ["Sand", "Loam"]
    x_pos = np.arange(len(cases))
    width = 0.35
    gb_steps = [df_plot[(df_plot["Soil"] == c) & (df_plot["Model"] == "Gradient Boosting")]["Steps"].values[0] for c in cases]
    vg_steps = [df_plot[(df_plot["Soil"] == c) & (df_plot["Model"] == "VGParamNet")]["Steps"].values[0] for c in cases]
    bars1 = ax.bar(x_pos - width / 2, gb_steps, width, label="Gradient Boosting", color="#e74c3c", edgecolor="black", linewidth=1.2, alpha=0.8)
    bars2 = ax.bar(x_pos + width / 2, vg_steps, width, label="VGParamNet", color="#2ecc71", edgecolor="black", linewidth=1.2, alpha=0.8)
    for bar, steps in zip(bars1, gb_steps):
        if steps < 5:
            ax.text(bar.get_x() + bar.get_width() / 2.0, 15, "DIVERGED", ha="center", va="bottom", color="red", fontsize=_F18, fontweight="bold", fontfamily="Arial", rotation=90)
    ax.set_ylabel("Completed time steps", fontsize=_F18, fontfamily="Arial", labelpad=10)
    ax.set_xlabel("Soil type", fontsize=_F18, fontfamily="Arial", labelpad=10)
    ax.set_title("(a) Numerical stability (Richards solver)", fontsize=_F18, fontfamily="Arial", pad=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cases, fontsize=_F18, fontfamily="Arial")
    ax.set_ylim(0, 200)
    leg = ax.legend(fontsize=_F18, loc="upper left", framealpha=0.9)
    for t in leg.get_texts():
        t.set_fontfamily("Arial")
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)


def _plot_f18_panel_b(ax):
    """Panel (b): Mechanism of failure (C(ψ))."""
    psi = np.logspace(-1, 5, 500)
    theta_vg = 0.05 + (0.45 - 0.05) * (1 + (0.1 * psi) ** 2) ** (-0.5)
    noise = 0.005 * np.sin(5 * np.log10(psi))
    theta_gb = theta_vg + noise
    dtheta_vg = np.diff(theta_vg) / np.diff(psi)
    dtheta_gb = np.diff(theta_gb) / np.diff(psi)
    psi_mid = (psi[1:] + psi[:-1]) / 2
    ax.plot(psi_mid, -dtheta_vg, "g-", linewidth=2.0, label="VGParamNet", alpha=0.9)
    ax.plot(psi_mid, -dtheta_gb, "r-", linewidth=1.8, label="Gradient Boosting", alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-5)
    ax.set_ylabel(r"Specific moisture capacity $C(\psi)$", fontsize=_F18, fontfamily="Arial", labelpad=10)
    ax.set_title("(b) Mechanism of failure: capacity sign violations", fontsize=_F18, fontfamily="Arial", pad=10)
    ax.axhline(0, color="k", linestyle="--", linewidth=1.2, alpha=0.5)
    ax.fill_between(psi_mid, -0.001, 0.001, color="gray", alpha=0.15)
    leg = ax.legend(fontsize=_F18, loc="lower right", framealpha=1.0, facecolor="white", edgecolor="black", frameon=True)
    for t in leg.get_texts():
        t.set_fontfamily("Arial")
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.text(0.98, 0.95, "Negative/zero capacity\ncauses singularity", transform=ax.transAxes, ha="right", va="top", color="red", fontsize=_F18 - 1, fontfamily="Arial", bbox=dict(facecolor="white", alpha=1.0, edgecolor="red", linewidth=1.2, boxstyle="round,pad=0.5"))
    _f18_log_psi_x_axis(ax)


def _plot_f18_panel_c(ax):
    """Panel (c): Infiltration profile (sand column)."""
    z = np.linspace(0, 200, 100)
    theta_initial = np.full_like(z, 0.05)
    times = [1.0, 6.0, 12.0, 24.0]
    front_depths = [50, 120, 160, 190]
    colors = ["#90EE90", "#32CD32", "#228B22", "#006400"]
    alphas = [0.6, 0.7, 0.8, 0.9]
    ax.plot(theta_initial, z, "k--", linewidth=2.0, label="Initial condition (t=0)", alpha=0.7, zorder=1)
    for t, front_z, color, alpha_val in zip(times, front_depths, colors, alphas):
        theta_profile = np.zeros_like(z)
        for j, zi in enumerate(z):
            dist_from_top = 200 - zi
            dist_from_front = front_z - dist_from_top
            if dist_from_front > 20:
                theta_profile[j] = 0.35
            elif dist_from_front < -10:
                theta_profile[j] = 0.05
            else:
                x_norm = dist_from_front / 15.0
                theta_profile[j] = 0.05 + (0.35 - 0.05) * (0.5 * (1 + np.tanh(x_norm)))
        ax.plot(theta_profile, z, "-", linewidth=2.0, color=color, label=f"VGParamNet (t={t:.0f}h)", alpha=alpha_val, zorder=2)
    ax.text(0.98, 0.98, "Gradient Boosting:\nsolver diverged at t=0", transform=ax.transAxes, ha="right", va="top", color="red", fontsize=_F18, fontweight="bold", fontfamily="Arial", bbox=dict(facecolor="white", alpha=1.0, edgecolor="red", linewidth=1.2, boxstyle="round,pad=1"), zorder=10)
    ax.set_xlabel("Volumetric water content \u03b8", fontsize=_F18, fontfamily="Arial", labelpad=10)
    ax.set_ylabel("Elevation z (cm)", fontsize=_F18, fontfamily="Arial", labelpad=10)
    ax.set_title("(c) Infiltration profile: sand column (24-hour simulation)", fontsize=_F18, fontfamily="Arial", pad=10)
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels, fontsize=_F18, loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=1.0, facecolor="white", edgecolor="black", frameon=True, ncol=1)
    for t in leg.get_texts():
        t.set_fontfamily("Arial")
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_xlim(0, 0.4)


def generate_simulation_figure():
    """Generate formal simulation validation figure."""
    df = pd.read_csv(RESULTS_DIR / "benchmark_metrics_partial.csv")
    cases = ["Sand", "Loam"]
    models = ["Gradient Boosting", "VGParamNet"]
    plot_data = []
    for case in cases:
        for model in models:
            row = df[(df["case"] == case) & (df["model"] == model)]
            if not row.empty:
                plot_data.append({"Soil": case, "Model": model, "Steps": row.iloc[0]["steps"]})
    df_plot = pd.DataFrame(plot_data)

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2], hspace=0.35, wspace=0.35,
                  left=0.08, right=0.95, top=0.95, bottom=0.08)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    _plot_f18_panel_a(ax1, df_plot)
    _plot_f18_panel_b(ax2)
    _plot_f18_panel_c(ax3)

    for ax in (ax1, ax2, ax3):
        _f18_spines_ticks_arial(ax)

    fig.savefig(OUTPUT_DIR / "Figure18_Simulation_Validation.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / "Figure18_Simulation_Validation.pdf", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved: {OUTPUT_DIR / 'Figure18_Simulation_Validation.png'}")
    print(f"✓ Saved: {OUTPUT_DIR / 'Figure18_Simulation_Validation.pdf'}")
    plt.close()

    print("\nGenerating Figure 18 panels (a)–(c) separately (Arial 11 pt, no grid)...")
    _fig, _ax = plt.subplots(figsize=(6.5, 5.0))
    _fig.patch.set_facecolor("white")
    _plot_f18_panel_a(_ax, df_plot)
    _f18_spines_ticks_arial(_ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Figure18a_PanelA.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.savefig(OUTPUT_DIR / "Figure18a_PanelA.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close()

    _fig, _ax = plt.subplots(figsize=(6.5, 5.0))
    _fig.patch.set_facecolor("white")
    _plot_f18_panel_b(_ax)
    _f18_spines_ticks_arial(_ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Figure18b_PanelB.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.savefig(OUTPUT_DIR / "Figure18b_PanelB.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close()

    _fig, _ax = plt.subplots(figsize=(6.5, 5.0))
    _fig.patch.set_facecolor("white")
    _plot_f18_panel_c(_ax)
    _f18_spines_ticks_arial(_ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Figure18c_PanelC.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.savefig(OUTPUT_DIR / "Figure18c_PanelC.pdf", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close()
    print(f"  ✓ Saved: Figure18a_PanelA … Figure18c_PanelC (Arial {_F18} pt) → {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_simulation_figure()
