#!/usr/bin/env python3
"""
Training script for VGParamNet Ablation (Task 3)
Trains model to predict α, n, θs, θr from 14 inputs (excluding θs, θr).
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from training_pinn.config_pinn_fixed import DATA_CONFIG, RESULTS_DIR
from models.vg_param_net_ablation import VGParamNetAblation
from models.vg_param_net import vg_theta


def to_se(theta, theta_s, theta_r):
    """Convert θ to effective saturation Se."""
    denom = np.maximum(theta_s - theta_r, 1e-6)
    Se = (theta - theta_r[:, None]) / denom[:, None]
    return np.clip(Se, 1e-6, 1.0 - 1e-6)


def main():
    print("=" * 80)
    print("Training VGParamNet Ablation (Task 3)")
    print("14 inputs → 4 outputs (α, n, θs, θr)")
    print("=" * 80)
    
    out_dir = RESULTS_DIR / "vgparamnet_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load suction grid and metadata
    psi = np.load(DATA_CONFIG["suction_grid_file"]).astype(np.float32)
    metadata = json.load(open(DATA_CONFIG["metadata_file"]))
    feature_cols = metadata["feature_cols"]
    
    # Exclude theta_s and theta_r from features
    feature_cols_ablation = [f for f in feature_cols if f not in ['theta_s', 'theta_r']]
    
    print(f"\n1. Loading data...")
    print(f"   Suction grid: {len(psi)} points")
    print(f"   Original features: {len(feature_cols)}")
    print(f"   Ablation features: {len(feature_cols_ablation)} (excluding θs, θr)")
    
    # Load train/val/test tables
    X_train = pd.read_csv(DATA_CONFIG["train_file"])
    X_val = pd.read_csv(DATA_CONFIG["val_file"])
    X_test = pd.read_csv(DATA_CONFIG["test_file"])
    
    y_train = np.load(DATA_CONFIG["y_train_original_file"]).astype(np.float32)
    y_val = np.load(DATA_CONFIG["y_val_original_file"]).astype(np.float32)
    y_test = np.load(DATA_CONFIG["y_test_original_file"]).astype(np.float32)
    
    # Extract theta_s/theta_r as targets (not inputs)
    theta_s_train = X_train["theta_s"].values.astype(np.float32)
    theta_r_train = X_train["theta_r"].values.astype(np.float32)
    theta_s_val = X_val["theta_s"].values.astype(np.float32)
    theta_r_val = X_val["theta_r"].values.astype(np.float32)
    theta_s_test = X_test["theta_s"].values.astype(np.float32)
    theta_r_test = X_test["theta_r"].values.astype(np.float32)
    
    # Inputs (14 features, excluding θs, θr)
    Xtr = X_train[feature_cols_ablation].values.astype(np.float32)
    Xva = X_val[feature_cols_ablation].values.astype(np.float32)
    Xte = X_test[feature_cols_ablation].values.astype(np.float32)
    
    print(f"   Training: {len(Xtr)} samples")
    print(f"   Validation: {len(Xva)} samples")
    print(f"   Test: {len(Xte)} samples")
    
    # Training target in Se-space
    Se_tr = to_se(y_train, theta_s_train, theta_r_train).astype(np.float32)
    Se_va = to_se(y_val, theta_s_val, theta_r_val).astype(np.float32)
    
    # Build model
    print("\n2. Building VGParamNet Ablation model...")
    model = VGParamNetAblation(soil_prop_dim=len(feature_cols_ablation), hidden=(128, 128))
    
    # Build with dummy input
    dummy_input = tf.random.normal([1, len(feature_cols_ablation)])
    _ = model(dummy_input)
    print(f"   Model built: {model.count_params()} parameters")
    
    # Optimizer (lower LR than default for stability; clipping handles large gradients)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    # Training step
    @tf.function
    def train_step(x, se_target, ts_target, tr_target, psi_grid):
        with tf.GradientTape() as tape:
            alpha, n, theta_s_pred, theta_r_pred = model(x, training=True)

            # Reconstruct curves using predicted parameters
            psi_tf = tf.tile(tf.reshape(psi_grid, [1, -1]), [tf.shape(alpha)[0], 1])
            theta_pred = vg_theta(psi_tf, theta_s_pred, theta_r_pred, alpha, n)

            # Convert to Se (use predicted theta_s/theta_r — consistent with model outputs)
            theta_range = theta_s_pred[:, None] - theta_r_pred[:, None]
            se_pred = (theta_pred - theta_r_pred[:, None]) / (theta_range + 1e-6)
            se_pred = tf.clip_by_value(se_pred, 1e-6, 1.0 - 1e-6)

            # Loss: curve-space MSE + parameter supervision
            curve_loss = tf.reduce_mean((se_pred - se_target) ** 2)
            param_loss = (tf.reduce_mean((theta_s_pred - ts_target) ** 2) +
                          tf.reduce_mean((theta_r_pred - tr_target) ** 2))

            total_loss = curve_loss + 0.1 * param_loss
            # Guard against non-finite loss before computing gradients
            total_loss = tf.where(tf.math.is_finite(total_loss),
                                  total_loss, tf.zeros_like(total_loss))

        gradients = tape.gradient(total_loss, model.trainable_variables)
        # Replace None gradients with zeros; clip by global norm to preserve direction
        gradients = [g if g is not None else tf.zeros_like(v)
                     for g, v in zip(gradients, model.trainable_variables)]
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return total_loss, curve_loss, param_loss

    # Validation step — identical transforms to train_step
    @tf.function
    def val_step(x, se_target, ts_target, tr_target, psi_grid):
        alpha, n, theta_s_pred, theta_r_pred = model(x, training=False)

        psi_tf = tf.tile(tf.reshape(psi_grid, [1, -1]), [tf.shape(alpha)[0], 1])
        theta_pred = vg_theta(psi_tf, theta_s_pred, theta_r_pred, alpha, n)

        theta_range = theta_s_pred[:, None] - theta_r_pred[:, None]
        se_pred = (theta_pred - theta_r_pred[:, None]) / (theta_range + 1e-6)
        se_pred = tf.clip_by_value(se_pred, 1e-6, 1.0 - 1e-6)

        curve_loss = tf.reduce_mean((se_pred - se_target) ** 2)
        param_loss = (tf.reduce_mean((theta_s_pred - ts_target) ** 2) +
                      tf.reduce_mean((theta_r_pred - tr_target) ** 2))

        total_loss = curve_loss + 0.1 * param_loss
        total_loss = tf.where(tf.math.is_finite(total_loss),
                              total_loss, tf.zeros_like(total_loss))
        return total_loss, curve_loss, param_loss
    
    # Create datasets (shuffle train set each epoch)
    train_ds = tf.data.Dataset.from_tensor_slices(
        (Xtr, Se_tr, theta_s_train, theta_r_train)
    ).shuffle(len(Xtr)).batch(64).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(
        (Xva, Se_va, theta_s_val, theta_r_val)
    ).batch(64).prefetch(tf.data.AUTOTUNE)

    # Train loop
    print("\n3. Training...")
    print("   Max epochs: 200, early stopping patience: 15")

    best = 1e9
    patience, bad = 15, 0

    for epoch in range(1, 201):
        tr_losses, tr_curve_losses, tr_param_losses = [], [], []

        for batch in train_ds:
            x, se, ts, tr = batch
            loss, loss_curve, loss_param = train_step(x, se, ts, tr, psi)
            tr_losses.append(loss)
            tr_curve_losses.append(loss_curve)
            tr_param_losses.append(loss_param)

        va_losses, va_curve_losses = [], []
        for batch in val_ds:
            x, se, ts, tr = batch
            loss, loss_curve, loss_param = val_step(x, se, ts, tr, psi)
            va_losses.append(loss)
            va_curve_losses.append(loss_curve)

        tr_loss  = float(tf.reduce_mean(tr_losses))
        va_loss  = float(tf.reduce_mean(va_losses))
        tr_curve = float(tf.reduce_mean(tr_curve_losses))
        va_curve = float(tf.reduce_mean(va_curve_losses))

        # Stop if weights contain NaN
        if any(np.any(np.isnan(w)) for w in model.get_weights()):
            print(f"   Epoch {epoch:03d} | NaN detected in weights! Stopping training.")
            break

        print(f"   Epoch {epoch:03d} | train {tr_loss:.6f} (curve: {tr_curve:.6f}) | val {va_loss:.6f} (curve: {va_curve:.6f})")

        if va_loss < best - 1e-6 and not np.isnan(va_loss):
            best = va_loss
            bad = 0
            model.save(out_dir / "vgparamnet_ablation_best.keras")
            print(f"     -> New best validation loss: {best:.6f}")
        else:
            bad += 1
            if bad >= patience:
                print(f"   Early stopping at epoch {epoch}")
                break
    
    # Evaluate on test set
    print("\n4. Evaluating on test set...")
    best_model = tf.keras.models.load_model(
        out_dir / "vgparamnet_ablation_best.keras",
        custom_objects={"VGParamNetAblation": VGParamNetAblation},
        compile=False
    )
    
    alpha, n, theta_s_pred, theta_r_pred = best_model(Xte, training=False)

    # Parameter range sanity check
    print(f"   alpha:   min={alpha.numpy().min():.4f}  max={alpha.numpy().max():.4f}  median={float(np.median(alpha.numpy())):.4f}")
    print(f"   n:       min={n.numpy().min():.4f}  max={n.numpy().max():.4f}  median={float(np.median(n.numpy())):.4f}")
    print(f"   theta_s: min={theta_s_pred.numpy().min():.4f}  max={theta_s_pred.numpy().max():.4f}  median={float(np.median(theta_s_pred.numpy())):.4f}")
    print(f"   theta_r: min={theta_r_pred.numpy().min():.4f}  max={theta_r_pred.numpy().max():.4f}  median={float(np.median(theta_r_pred.numpy())):.4f}")

    # Reconstruct curves
    psi_tf = tf.tile(tf.reshape(psi, [1, -1]), [tf.shape(alpha)[0], 1])
    theta_pred = vg_theta(psi_tf, theta_s_pred, theta_r_pred, alpha, n).numpy()

    # Compute global RMSE
    rmse_global = np.sqrt(np.mean((y_test - theta_pred) ** 2))

    print(f"\n  Training complete!")
    print(f"  Global RMSE (test set): {rmse_global:.4f}")
    print(f"  Model saved to: {out_dir / 'vgparamnet_ablation_best.keras'}")
    
    # Save results
    np.save(out_dir / "theta_pred_test.npy", theta_pred)
    np.save(out_dir / "alpha_test.npy", alpha.numpy())
    np.save(out_dir / "n_test.npy", n.numpy())
    np.save(out_dir / "theta_s_pred_test.npy", theta_s_pred.numpy())
    np.save(out_dir / "theta_r_pred_test.npy", theta_r_pred.numpy())
    
    # Save metrics
    results = {
        'global_rmse': float(rmse_global),
        'n_test_samples': int(len(Xte)),
        'input_features': feature_cols_ablation,
        'outputs': ['alpha', 'n', 'theta_s', 'theta_r']
    }
    with open(out_dir / "task3_ablation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {out_dir}")
    print(f"  - vgparamnet_ablation_best.keras: trained model")
    print(f"  - task3_ablation_results.json: metrics and configuration")


if __name__ == "__main__":
    main()
