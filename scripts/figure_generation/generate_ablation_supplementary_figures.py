#!/usr/bin/env python3
"""
Generate supplementary ablation figure:

Figure S4: Effect of explicit slope penalty (oversmoothing of SWCC knee).

Uses VGParamNet ablation runs (run_A, run_B, run_C, run_D) stored in:
results_pinn_fixed/vgparamnet/run_*/theta_vgparamnet_test.npy and training_config.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is on sys.path
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def compute_max_slope(psi_grid, theta_curve):
    """Compute maximum |dθ/dlog(ψ)| for one curve."""
    log_psi = np.log(np.maximum(psi_grid, 1e-6))
    dlog = np.diff(log_psi)
    dtheta = np.diff(theta_curve)
    dtheta_dlog = np.abs(dtheta / (dlog + 1e-8))
    if dtheta_dlog.size == 0:
        return 0.0
    return float(np.max(dtheta_dlog))


def main():
    root_dir = ROOT_DIR
    results_dir = root_dir / "results_pinn_fixed" / "vgparamnet"
    supp_dir = root_dir / "paper_figures" / "supplementary"
    supp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Generating supplementary ablation figure (Figure S4)")
    print("=" * 80)

    # Global style (match main paper figures)
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["axes.titlesize"] = 20
    plt.rcParams["xtick.labelsize"] = 18
    plt.rcParams["ytick.labelsize"] = 18
    plt.rcParams["legend.fontsize"] = 18
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["xtick.major.width"] = 1.5
    plt.rcParams["ytick.major.width"] = 1.5

    # ----------------------------------------------------------------------
    # 1. Load observed test curves and metadata
    # ----------------------------------------------------------------------
    from training_pinn.config_pinn_fixed import DATA_CONFIG

    print("\n1. Loading observed test data...")
    X_test = pd.read_csv(DATA_CONFIG["test_file"])
    y_test = np.load(DATA_CONFIG["y_test_original_file"]).astype(np.float32)
    psi = np.load(DATA_CONFIG["suction_grid_file"]).astype(np.float32)

    theta_s = X_test["theta_s"].values.astype(np.float32)
    theta_r = X_test["theta_r"].values.astype(np.float32)

    # Optional texture information
    texture_col = None
    for col in ["texture_class", "USDA_class", "soil_texture"]:
        if col in X_test.columns:
            texture_col = col
            break

    # ----------------------------------------------------------------------
    # 2. Identify final (ψ50-only) and slope-penalty runs
    # ----------------------------------------------------------------------
    print("\n2. Loading ablation runs...")
    runs = {}
    for run_id in ["A", "B", "C", "D"]:
        run_dir = results_dir / f"run_{run_id}"
        cfg_path = run_dir / "training_config.json"
        theta_path = run_dir / "theta_vgparamnet_test.npy"
        if cfg_path.exists() and theta_path.exists():
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            theta_pred = np.load(theta_path).astype(np.float32)
            runs[run_id] = {
                "config": cfg,
                "theta": theta_pred,
            }

    if not runs:
        print("  ⚠ No ablation runs found in results_pinn_fixed/vgparamnet/run_*/")
        return

    # Choose final ψ50-only run (lambda_psi50 > 0, lambda_slope == 0)
    final_run_id = None
    slope_run_id = None
    for run_id, info in runs.items():
        lam_psi = float(info["config"].get("lambda_psi50", 0.0))
        lam_slope = float(info["config"].get("lambda_slope", 0.0))
        if lam_psi > 0 and lam_slope == 0 and final_run_id is None:
            final_run_id = run_id
        if lam_slope > 0 and slope_run_id is None:
            slope_run_id = run_id

    if final_run_id is None or slope_run_id is None:
        print("  ⚠ Could not uniquely identify ψ50-only and slope-penalty runs;")
        print("    please check training_config.json in run_A..run_D.")
        return

    print(f"  • Final (ψ50-only) run:   {final_run_id} "
          f"(λ_ψ50={runs[final_run_id]['config'].get('lambda_psi50')}, "
          f"λ_slope={runs[final_run_id]['config'].get('lambda_slope')})")
    print(f"  • Slope-penalty run:      {slope_run_id} "
          f"(λ_ψ50={runs[slope_run_id]['config'].get('lambda_psi50')}, "
          f"λ_slope={runs[slope_run_id]['config'].get('lambda_slope')})")

    theta_final = runs[final_run_id]["theta"]
    theta_slope = runs[slope_run_id]["theta"]

    # Sanity check shapes
    if theta_final.shape != theta_slope.shape or theta_final.shape != y_test.shape:
        print("  ⚠ Shape mismatch between observed and predicted curves; "
              "cannot generate Figure S4.")
        print(f"    y_test: {y_test.shape}, theta_final: {theta_final.shape}, "
              f"theta_slope: {theta_slope.shape}")
        return

    # ----------------------------------------------------------------------
    # 3. Select a sharp-knee sand sample
    # ----------------------------------------------------------------------
    print("\n3. Selecting a sharp-knee sand sample...")

    n_samples = y_test.shape[0]
    max_slope_obs = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        max_slope_obs[i] = compute_max_slope(psi, y_test[i])

    # Restrict to sands if texture info exists
    candidate_indices = np.arange(n_samples)
    if texture_col is not None:
        texture_series = X_test[texture_col].astype(str)
        sand_mask = texture_series.str.contains("Sand", case=False, na=False)
        if sand_mask.any():
            candidate_indices = np.where(sand_mask.values)[0]

    # Among candidates, pick sample with largest max_slope_obs
    if candidate_indices.size == 0:
        print("  ⚠ No sand samples found; using global sharpest knee instead.")
        idx_best = int(np.argmax(max_slope_obs))
    else:
        idx_best = int(candidate_indices[np.argmax(max_slope_obs[candidate_indices])])

    print(f"  • Selected sample index: {idx_best}")
    if texture_col is not None:
        print(f"    Texture class: {X_test.loc[idx_best, texture_col]}")
    print(f"    Observed max slope: {max_slope_obs[idx_best]:.3e}")

    # Extract curves
    theta_true = y_test[idx_best]
    theta_final_sample = theta_final[idx_best]
    theta_slope_sample = theta_slope[idx_best]

    # ----------------------------------------------------------------------
    # 4. Plot Figure S4: Oversmoothing effect
    # ----------------------------------------------------------------------
    print("\n4. Plotting Figure S4 (slope penalty oversmoothing)...")

    # Increase height for a more balanced aspect ratio
    fig, ax = plt.subplots(figsize=(8, 7.5))

    # Colors
    color_true = "black"
    color_final = "#2E86AB"   # blue
    color_slope = "#F18F01"   # orange

    # Plot observed
    ax.semilogx(psi, theta_true, color=color_true, linewidth=2.5,
                label="Observed (UNSODA)")
    # Final psi50-only model (avoid subscript digits that Arial lacks)
    ax.semilogx(psi, theta_final_sample, color=color_final, linewidth=2.5,
                linestyle="-", label=f"VGParamNet (psi50 loss only, run {final_run_id})")
    # Slope-penalty model (expected oversmoothing)
    ax.semilogx(psi, theta_slope_sample, color=color_slope, linewidth=2.5,
                linestyle="--", label=f"VGParamNet (slope loss, run {slope_run_id})")

    ax.set_xlabel("Suction ψ (kPa)", fontsize=17, fontfamily="Arial", labelpad=10)
    ax.set_ylabel("Water content θ (m³/m³)", fontsize=17, fontfamily="Arial", labelpad=10)
    ax.set_title("Effect of explicit slope penalty on SWCC knee", fontsize=17, fontfamily="Arial", pad=12)

    ax.grid(True, alpha=0.3, which="both", linestyle="--", color="#CCCCCC", linewidth=0.7)
    ax.tick_params(labelsize=17)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Arial")

    # Legend at bottom to avoid squeezing x-axis
    # Place legend below plot, centered horizontally
    legend = ax.legend(loc="upper center",
                       bbox_to_anchor=(0.5, -0.25),
                       ncol=1,
                       framealpha=0.95,
                       facecolor="white",
                       edgecolor="gray")
    for text in legend.get_texts():
        text.set_fontfamily("Arial")

    fig.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])

    out_png = supp_dir / "Figure_S4_Slope_Penalty_Oversmoothing.png"
    out_pdf = supp_dir / "Figure_S4_Slope_Penalty_Oversmoothing.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✓ Saved: {out_png}")
    print(f"  ✓ Saved: {out_pdf}")
    print("\nDone.")


if __name__ == "__main__":
    main()

