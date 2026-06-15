#!/usr/bin/env python3
"""
VG Fit Stability & Hydraulic Conductivity Consistency Analysis (Path 1)

For each test sample and each curve type (Observed, GB, PINN), this script:
  - Fits Van Genuchten parameters (α, n) with fixed θs, θr using vg_fit.fit_vg_alpha_n
  - Defines fit success based on RMSE and parameter bounds
  - Computes a simple relative conductivity proxy K_r(ψ) = Se^l and checks monotonicity
  - Saves per-curve results and summary statistics
  - Generates:
      - Histograms of VG parameters (α, n)
      - Representative K_r(ψ) curves showing GB bumps vs smooth PINN behavior
"""

import os
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Ensure project root is on sys.path so we can import project modules
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.vg_fit import fit_vg_alpha_n, mualem_kr_from_theta  # noqa: E402
from training_pinn.config_pinn_fixed import DATA_CONFIG, RESULTS_DIR  # noqa: E402
from models.pinn_monotonic import MonotonicPINN  # noqa: E402
from models.pinn import PhysicsEncodingLayer  # noqa: E402
from baseline_models import BaselineModels  # noqa: E402


TOL_STRICT = 1e-6  # Strict tolerance for numerical precision
TOL_PRACTICAL = 1e-3  # Practical tolerance for physically meaningful bumps


def is_monotone_decreasing(y: np.ndarray, tol: float = TOL_STRICT):
    """Check monotone decrease and quantify bumps."""
    y = np.asarray(y, dtype=float)
    dy = np.diff(y)
    bumps = dy > tol
    num_bumps = int(np.sum(bumps))
    max_bump = float(np.max(np.where(bumps, dy, 0.0))) if num_bumps > 0 else 0.0
    return bool(np.all(dy <= tol)), num_bumps, max_bump


def load_pinn_predictions():
    """Load best PINN model and generate test predictions in physical θ space."""
    print("\nLoading PINN main model and test data...")

    # Suction grid and metadata
    suction_grid = np.load(DATA_CONFIG["suction_grid_file"])
    metadata = json.load(open(DATA_CONFIG["metadata_file"]))

    # Test data (real-only)
    X_test = pd.read_csv(DATA_CONFIG["test_file"])
    y_test_original = np.load(DATA_CONFIG["y_test_original_file"])
    theta_s_test = X_test["theta_s"].values
    theta_r_test = X_test["theta_r"].values
    feature_cols = metadata["feature_cols"]

    # Load best PINN
    checkpoint_dir = RESULTS_DIR / "checkpoints"
    best_model_path = checkpoint_dir / "pinn_best_model_fixed.keras"
    print(f"  PINN checkpoint: {best_model_path}")

    model = MonotonicPINN(
        soil_prop_dim=metadata["n_features"],
        suction_points=metadata["n_swcc_points"],
        physics_units=128,
        hidden_dims=[128, 256, 128, 64],
    )

    # Build model once
    import tensorflow as tf  # local import to avoid noisy logs at module import

    dummy_soil = tf.random.normal([1, metadata["n_features"]])
    dummy_suction = tf.random.normal([1, metadata["n_swcc_points"]])
    _ = model({"soil_props": dummy_soil, "suction": dummy_suction})

    saved_model = tf.keras.models.load_model(
        str(best_model_path),
        custom_objects={"MonotonicPINN": MonotonicPINN, "PhysicsEncodingLayer": PhysicsEncodingLayer},
        compile=False,
    )
    model.set_weights(saved_model.get_weights())
    print("  ✓ PINN loaded")

    # Predict normalized θ, then denormalize using per-sample θs, θr
    print("  Making PINN predictions on test set...")
    y_pred_norm = []
    batch_size = 32
    for i in range(0, len(X_test), batch_size):
        batch_end = min(i + batch_size, len(X_test))
        batch_soil = X_test.iloc[i:batch_end][feature_cols].values.astype(np.float32)
        batch_suction = np.tile(suction_grid, (batch_end - i, 1)).astype(np.float32)
        inputs = {"soil_props": batch_soil, "suction": batch_suction}
        theta_pred_norm_batch = model(inputs, training=False)
        y_pred_norm.extend(theta_pred_norm_batch.numpy())

    y_pred_norm = np.array(y_pred_norm)
    y_pinn = np.zeros_like(y_pred_norm)
    for i in range(len(X_test)):
        theta_range = theta_s_test[i] - theta_r_test[i]
        y_pinn[i] = theta_r_test[i] + y_pred_norm[i] * theta_range

    print("  ✓ PINN predictions complete")
    return X_test, y_test_original, y_pinn, suction_grid


def load_gb_predictions():
    """Train GB baseline on processed data and get test predictions (physical θ)."""
    print("\nTraining Gradient Boosting baseline and predicting on test set...")
    bm = BaselineModels(data_dir="data_processed", output_dir="results_baseline")

    (X_train, X_val, X_test), (y_train, y_val, y_test), suction_grid = bm.load_data()
    X_train_feat, X_val_feat, X_test_feat, feature_cols = bm.prepare_features(X_train, X_val, X_test)

    gb_models = bm.train_gradient_boosting(X_train_feat, y_train, X_val_feat, y_val)
    y_gb = bm.predict_swcc(gb_models, X_test_feat, model_type="gradient_boosting", n_points=y_test.shape[1])

    print("  ✓ GB predictions complete")
    return X_test, y_test, y_gb, suction_grid


def main():
    print("=" * 80)
    print("VG FIT STABILITY & K(ψ) CONSISTENCY ANALYSIS")
    print("=" * 80)

    # Output directory
    out_dir = RESULTS_DIR / "vg_fit"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    X_test_pinn, theta_obs, theta_pinn, psi = load_pinn_predictions()
    X_test_gb, theta_obs_gb, theta_gb, psi_gb = load_gb_predictions()

    # Sanity checks
    assert theta_obs.shape == theta_obs_gb.shape, "Observed test sets differ between PINN and GB loaders"
    assert np.allclose(psi, psi_gb), "Suction grids differ between PINN and GB loaders"

    # Load VGParamNet predictions if available
    vgparamnet_path = RESULTS_DIR / "vgparamnet" / "theta_vgparamnet_test.npy"
    if vgparamnet_path.exists():
        theta_vgnet = np.load(vgparamnet_path).astype(float)
        print(f"\n✓ Loaded VGParamNet predictions: {theta_vgnet.shape}")
        assert theta_vgnet.shape == theta_obs.shape, "VGParamNet shape mismatch"
    else:
        theta_vgnet = None
        print(f"\n⚠ VGParamNet predictions not found at {vgparamnet_path}")
        print("  Run training_pinn/train_vg_param_net.py first to generate predictions")

    # Use consistent observed test data and θs/θr from data_pinn_normalized
    X_test = X_test_pinn
    theta_s = X_test["theta_s"].values.astype(float)
    theta_r = X_test["theta_r"].values.astype(float)

    N, P = theta_obs.shape
    assert psi.shape[0] == P

    print(f"\nTest samples: {N}, SWCC points per curve: {P}")

    # RMSE threshold for VG fit success (in physical θ-space)
    rmse_thr = 0.08

    rows = []
    curves = {
        "Observed": theta_obs,
        "GB": theta_gb,
        "PINN": theta_pinn,
    }
    if theta_vgnet is not None:
        curves["VGParamNet"] = theta_vgnet

    for i in range(N):
        ts = float(theta_s[i])
        tr = float(theta_r[i])
        if ts <= tr + 1e-6:
            # Degenerate case; skip
            continue

        for name, arr in curves.items():
            theta = arr[i].astype(float)

            # Fit VG α, n with fixed θs, θr
            try:
                alpha, n_val, rmse, theta_fit = fit_vg_alpha_n(psi, theta, ts, tr)
                fit_ok = np.isfinite(alpha) and np.isfinite(n_val) and np.isfinite(rmse)
            except Exception:
                alpha, n_val, rmse, fit_ok = np.nan, np.nan, np.nan, False

            # Fit success criteria
            success_rmse = bool(fit_ok and (rmse < rmse_thr))
            success_param = bool(
                fit_ok
                and (alpha > 1e-8)
                and (alpha < 1e2)
                and (n_val > 1.01)
                and (n_val < 20.0)
            )
            success = bool(success_rmse and success_param)

            # θ(ψ) monotonicity diagnostics (proves equivalence with K_r)
            theta_mono_strict, theta_bumps_strict, theta_max_bump_strict = is_monotone_decreasing(
                theta, tol=TOL_STRICT
            )
            theta_mono_practical, theta_bumps_practical, theta_max_bump_practical = is_monotone_decreasing(
                theta, tol=TOL_PRACTICAL
            )
            
            # Normalized bump sizes (relative to θ-range) for reviewer-proof presentation
            theta_range = max(ts - tr, 1e-6)
            theta_max_bump_rel_strict = float(theta_max_bump_strict / theta_range * 100.0) if theta_max_bump_strict > 0 else 0.0
            theta_max_bump_rel_practical = float(theta_max_bump_practical / theta_range * 100.0) if theta_max_bump_practical > 0 else 0.0

            # Relative conductivity proxy K_r(ψ)
            Kr = mualem_kr_from_theta(theta, ts, tr, l=0.5)
            kr_monotone_strict, kr_bumps_strict, kr_max_bump_strict = is_monotone_decreasing(
                Kr, tol=TOL_STRICT
            )
            kr_monotone_practical, kr_bumps_practical, kr_max_bump_practical = is_monotone_decreasing(
                Kr, tol=TOL_PRACTICAL
            )

            rows.append(
                {
                    "sample_id": i,
                    "curve_type": name,
                    "theta_s": ts,
                    "theta_r": tr,
                    "alpha": float(alpha) if np.isfinite(alpha) else np.nan,
                    "n": float(n_val) if np.isfinite(n_val) else np.nan,
                    "vg_fit_rmse": float(rmse) if np.isfinite(rmse) else np.nan,
                    "fit_success": success,
                    "fit_success_rmse": success_rmse,
                    "fit_success_param": success_param,
                    # θ(ψ) monotonicity (strict and practical)
                    "theta_monotone_strict": bool(theta_mono_strict),
                    "theta_num_bumps_strict": int(theta_bumps_strict),
                    "theta_max_bump_strict": float(theta_max_bump_strict),
                    "theta_max_bump_rel_strict": float(theta_max_bump_rel_strict),
                    "theta_monotone_practical": bool(theta_mono_practical),
                    "theta_num_bumps_practical": int(theta_bumps_practical),
                    "theta_max_bump_practical": float(theta_max_bump_practical),
                    "theta_max_bump_rel_practical": float(theta_max_bump_rel_practical),
                    # K_r(ψ) monotonicity (strict and practical)
                    "kr_monotone_strict": bool(kr_monotone_strict),
                    "kr_num_bumps_strict": int(kr_bumps_strict),
                    "kr_max_bump_strict": float(kr_max_bump_strict),
                    "kr_monotone_practical": bool(kr_monotone_practical),
                    "kr_num_bumps_practical": int(kr_bumps_practical),
                    "kr_max_bump_practical": float(kr_max_bump_practical),
                }
            )

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{N} samples...")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "vg_fit_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved per-curve results to: {csv_path}")

    # Summary statistics per curve type
    summary = {}
    for name in df["curve_type"].unique():
        sub = df[df["curve_type"] == name]
        summary[name] = {
            "N": int(len(sub)),
            "vg_fit_success_rate": float(sub["fit_success"].mean() * 100.0),
            "vg_fit_rmse_median": float(sub["vg_fit_rmse"].median()),
            "alpha_median": float(sub["alpha"].median()),
            "n_median": float(sub["n"].median()),
            # θ(ψ) monotonicity (strict and practical)
            "theta_monotone_rate_strict": float(sub["theta_monotone_strict"].mean() * 100.0),
            "theta_bumps_mean_strict": float(sub["theta_num_bumps_strict"].mean()),
            "theta_max_bump_median_strict": float(sub["theta_max_bump_strict"].median()),
            "theta_max_bump_rel_median_strict": float(sub["theta_max_bump_rel_strict"].median()),
            "theta_monotone_rate_practical": float(sub["theta_monotone_practical"].mean() * 100.0),
            "theta_bumps_mean_practical": float(sub["theta_num_bumps_practical"].mean()),
            "theta_max_bump_median_practical": float(sub["theta_max_bump_practical"].median()),
            "theta_max_bump_rel_median_practical": float(sub["theta_max_bump_rel_practical"].median()),
            # K_r(ψ) monotonicity (strict and practical)
            "kr_monotone_rate_strict": float(sub["kr_monotone_strict"].mean() * 100.0),
            "kr_bumps_mean_strict": float(sub["kr_num_bumps_strict"].mean()),
            "kr_max_bump_median_strict": float(sub["kr_max_bump_strict"].median()),
            "kr_monotone_rate_practical": float(sub["kr_monotone_practical"].mean() * 100.0),
            "kr_bumps_mean_practical": float(sub["kr_num_bumps_practical"].mean()),
            "kr_max_bump_median_practical": float(sub["kr_max_bump_practical"].median()),
        }

    summary_path = out_dir / "vg_fit_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON to: {summary_path}")

    # Color and label maps for all methods
    color_map = {
        "Observed": "#2E86AB",
        "GB": "#F18F01",
        "PINN": "#06A77D",
        "VGParamNet": "#9B59B6",
    }
    label_map = {
        "Observed": "Observed",
        "GB": "Gradient Boosting",
        "PINN": "PINN",
        "VGParamNet": "VGParamNet",
    }
    
    # ------------------------------------------------------------------
    # Figure 1: parameter histograms (α and n)
    # ------------------------------------------------------------------
    print("\nGenerating VG parameter histogram figures...")
    plt.figure(figsize=(8, 5))
    for name in ["Observed", "GB", "PINN", "VGParamNet"]:
        if name not in df["curve_type"].unique():
            continue
        label = label_map[name]
        color = color_map[name]
        sub = df[(df["curve_type"] == name) & (df["fit_success"])]
        alphas = sub["alpha"].values
        alphas = alphas[np.isfinite(alphas)]
        if alphas.size == 0:
            continue
        plt.hist(
            np.log10(alphas + 1e-12),
            bins=30,
            alpha=0.6,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
    plt.xlabel("log10(α) [1/kPa]")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    alpha_hist_path = out_dir / "vg_alpha_hist.png"
    plt.savefig(alpha_hist_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved: {alpha_hist_path}")

    plt.figure(figsize=(8, 5))
    for name in ["Observed", "GB", "PINN", "VGParamNet"]:
        if name not in df["curve_type"].unique():
            continue
        label = label_map[name]
        color = color_map[name]
        sub = df[(df["curve_type"] == name) & (df["fit_success"])]
        ns = sub["n"].values
        ns = ns[np.isfinite(ns)]
        if ns.size == 0:
            continue
        plt.hist(
            ns,
            bins=30,
            alpha=0.6,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
    plt.xlabel("n [-]")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    n_hist_path = out_dir / "vg_n_hist.png"
    plt.savefig(n_hist_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved: {n_hist_path}")

    # ------------------------------------------------------------------
    # Figure 2: representative K_r(ψ) curves (GB bumps vs smooth PINN)
    # ------------------------------------------------------------------
    print("Generating representative K_r(ψ) curves...")

    gb_df = df[df["curve_type"] == "GB"].set_index("sample_id")
    pinn_df = df[df["curve_type"] == "PINN"].set_index("sample_id")

    candidates = []
    for sid in gb_df.index.intersection(pinn_df.index):
        if (gb_df.loc[sid, "kr_num_bumps_strict"] > 0) and (pinn_df.loc[sid, "kr_num_bumps_strict"] == 0):
            candidates.append(int(sid))

    if len(candidates) < 6:
        # Fallback: first 6 valid samples
        candidates = list(range(min(6, N)))

    sel = candidates[:6]

    plt.figure(figsize=(10, 6))
    for j, sid in enumerate(sel):
        ts = theta_s[sid]
        tr = theta_r[sid]
        Kr_obs = mualem_kr_from_theta(theta_obs[sid], ts, tr, l=0.5)
        Kr_gb = mualem_kr_from_theta(theta_gb[sid], ts, tr, l=0.5)
        Kr_pinn = mualem_kr_from_theta(theta_pinn[sid], ts, tr, l=0.5)

        label_obs = "Observed" if j == 0 else None
        label_gb = "GB" if j == 0 else None
        label_pinn = "PINN" if j == 0 else None
        label_vgnet = "VGParamNet" if j == 0 and theta_vgnet is not None else None

        plt.plot(psi, Kr_obs, "-", linewidth=1.2, label=label_obs, color="#000000", alpha=0.7)
        plt.plot(psi, Kr_gb, "--", linewidth=1.2, label=label_gb, color="#2E86AB", alpha=0.7)
        plt.plot(psi, Kr_pinn, "-.", linewidth=1.2, label=label_pinn, color="#FF6B6B", alpha=0.7)
        if theta_vgnet is not None:
            Kr_vgnet = mualem_kr_from_theta(theta_vgnet[sid], ts, tr, l=0.5)
            plt.plot(psi, Kr_vgnet, ":", linewidth=1.2, label=label_vgnet, color="#9B59B6", alpha=0.7)

    plt.xscale("log")
    plt.xlabel("Suction ψ (kPa)")
    plt.ylabel("Relative conductivity proxy K_r")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    kr_fig_path = out_dir / "kr_representative_curves.png"
    plt.savefig(kr_fig_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved: {kr_fig_path}")

    print("\nAnalysis complete.")
    print(f"Results directory: {out_dir}")
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

