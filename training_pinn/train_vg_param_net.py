#!/usr/bin/env python3
"""
Training script for VGParamNet (Path 2 Variant A)
Predicts van Genuchten parameters (α, n) and uses analytical VG formula to compute θ(ψ).
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
from models.vg_param_net import VGParamNet, vg_theta, psi_at_Se, PSI50_MIN, PSI50_MAX


def to_se(theta, theta_s, theta_r):
    """Convert θ to effective saturation Se."""
    denom = np.maximum(theta_s - theta_r, 1e-6)
    Se = (theta - theta_r[:, None]) / denom[:, None]
    return np.clip(Se, 1e-6, 1.0 - 1e-6)


def knee_weights_from_observed(theta_obs, psi, beta=2.0, wmin=1.0, wmax=8.0):
    """
    Compute knee-aware weights based on |dθ/dlog10(psi)| per sample.
    
    Args:
        theta_obs: [B, P] physical θ values
        psi: [P] suction in kPa
        beta: strength of knee emphasis
        wmin, wmax: weight bounds
    
    Returns:
        weights: [B, P] per-point weights
    """
    logpsi = np.log10(np.maximum(psi, 1e-6))
    dlog = np.diff(logpsi)  # [P-1]
    dth = np.diff(theta_obs, axis=1)  # [B, P-1]
    slope = np.abs(dth / dlog[None, :])  # [B, P-1]
    
    # Pad to [B, P]
    slope = np.concatenate([slope[:, :1], slope], axis=1)
    denom = np.mean(slope, axis=1, keepdims=True) + 1e-6
    w = 1.0 + beta * (slope / denom)
    return np.clip(w, wmin, wmax).astype(np.float32)


def main():
    # Parse command-line arguments first
    import argparse
    parser = argparse.ArgumentParser(description='VGParamNet Training with Ablation Study')
    parser.add_argument('--lambda_psi50', type=float, default=0.1, help='ψ50 loss weight (default: 0.1)')
    parser.add_argument('--lambda_slope', type=float, default=0.0, help='Slope loss weight (default: 0.0)')
    parser.add_argument('--use_huber', action='store_true', help='Use Huber loss for ψ50 (default: False)')
    parser.add_argument('--use_curriculum', action='store_true', help='Use curriculum scheduling for λ_ψ50 (default: False)')
    parser.add_argument('--run_id', type=str, default='A', help='Run ID for ablation study (A, B, C, D)')
    args = parser.parse_args()
    
    lambda_psi50 = args.lambda_psi50
    lambda_slope = args.lambda_slope
    use_slope_loss = lambda_slope > 0.0
    use_huber = args.use_huber
    use_curriculum = args.use_curriculum
    run_id = args.run_id
    
    print("=" * 80)
    print("Training VGParamNet (Path 2 Variant A)")
    print("=" * 80)
    print(f"\n📋 Training Configuration (Run {run_id}):")
    print(f"   λ_ψ50: {lambda_psi50}")
    print(f"   λ_slope: {lambda_slope}")
    print(f"   Use Huber loss: {use_huber}")
    print(f"   Use curriculum: {use_curriculum}")
    
    out_dir = RESULTS_DIR / "vgparamnet"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load suction grid and metadata
    psi = np.load(DATA_CONFIG["suction_grid_file"]).astype(np.float32)  # [P]
    metadata = json.load(open(DATA_CONFIG["metadata_file"]))
    feature_cols = metadata["feature_cols"]
    P = metadata["n_swcc_points"]
    
    print(f"\n1. Loading data...")
    print(f"   Suction grid: {len(psi)} points")
    print(f"   Features: {len(feature_cols)}")
    
    # Load train/val/test tables and observed curves (physical space)
    X_train = pd.read_csv(DATA_CONFIG["train_file"])
    X_val = pd.read_csv(DATA_CONFIG["val_file"])
    X_test = pd.read_csv(DATA_CONFIG["test_file"])
    
    y_train = np.load(DATA_CONFIG["y_train_original_file"]).astype(np.float32)  # [N,P] physical θ
    y_val = np.load(DATA_CONFIG["y_val_original_file"]).astype(np.float32)
    y_test = np.load(DATA_CONFIG["y_test_original_file"]).astype(np.float32)
    
    # Pull theta_s/theta_r from the same split tables (must align with y_* arrays)
    theta_s_train = X_train["theta_s"].values.astype(np.float32)
    theta_r_train = X_train["theta_r"].values.astype(np.float32)
    theta_s_val = X_val["theta_s"].values.astype(np.float32)
    theta_r_val = X_val["theta_r"].values.astype(np.float32)
    theta_s_test = X_test["theta_s"].values.astype(np.float32)
    theta_r_test = X_test["theta_r"].values.astype(np.float32)
    
    # Inputs
    Xtr = X_train[feature_cols].values.astype(np.float32)
    Xva = X_val[feature_cols].values.astype(np.float32)
    Xte = X_test[feature_cols].values.astype(np.float32)
    
    print(f"   Training: {len(Xtr)} samples")
    print(f"   Validation: {len(Xva)} samples")
    print(f"   Test: {len(Xte)} samples")
    
    # Training target in Se-space (more stable)
    Se_tr = to_se(y_train, theta_s_train, theta_r_train).astype(np.float32)
    Se_va = to_se(y_val, theta_s_val, theta_r_val).astype(np.float32)
    
    # Load precomputed ψ50 values (Tweak 1)
    psi50_train_path = out_dir / "psi50_train.npy"
    psi50_val_path = out_dir / "psi50_val.npy"
    
    if psi50_train_path.exists() and psi50_val_path.exists():
        psi50_train = np.load(psi50_train_path).astype(np.float32)
        psi50_val = np.load(psi50_val_path).astype(np.float32)
        print(f"\n2. Loaded precomputed ψ50 values")
        print(f"   Training ψ50: median={np.median(psi50_train):.2f} kPa")
        print(f"   Validation ψ50: median={np.median(psi50_val):.2f} kPa")
    else:
        print(f"\n⚠ Warning: ψ50 files not found. Run analysis/precompute_psi50.py first.")
        print("   Training without ψ50 loss...")
        psi50_train = None
        psi50_val = None
    
    # Knee-aware weights from observed θ (optional but recommended for sands)
    print("\n3. Computing knee-aware weights...")
    W_tr = knee_weights_from_observed(y_train, psi, beta=2.0)  # [N,P]
    W_va = knee_weights_from_observed(y_val, psi, beta=2.0)
    print("   ✓ Weights computed")
    
    # Build model with better initialization
    print("\n3. Building VGParamNet model...")
    model = VGParamNet(soil_prop_dim=Xtr.shape[1], hidden=(128, 128))
    
    # Dummy build
    dummy_input = tf.random.normal([1, Xtr.shape[1]])
    _ = model(dummy_input, training=False)
    
    # Re-initialize weights with small random values to avoid NaN
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            if hasattr(layer, 'kernel') and layer.kernel is not None:
                new_kernel = tf.keras.initializers.GlorotUniform()(layer.kernel.shape)
                layer.kernel.assign(new_kernel)
        if hasattr(layer, 'bias_initializer'):
            if hasattr(layer, 'bias') and layer.bias is not None:
                new_bias = tf.keras.initializers.Zeros()(layer.bias.shape)
                layer.bias.assign(new_bias)
    
    print("   ✓ Model built and re-initialized")
    
    # Optimizer with very conservative learning rate
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    
    @tf.function
    def train_step(x, se_obs, ts, tr, w, psi50_obs=None, lambda_psi50_current=None):
        with tf.GradientTape() as tape:
            alpha, n = model(x, training=True)
            # No extra clipping here: the model's own sigmoid parameterisation
            # already guarantees alpha in [ALPHA_MIN, ALPHA_MAX] and
            # n in [N_MIN, N_MAX] with well-conditioned gradients at all points.

            # psi tiled for batch
            psi_tf = tf.tile(tf.reshape(psi, [1, -1]), [tf.shape(x)[0], 1])
            theta_pred = vg_theta(psi_tf, ts, tr, alpha, n)

            # Convert to Se; guard against near-zero (ts - tr) range
            denom   = tf.maximum(ts - tr, 1e-6)
            se_pred = tf.clip_by_value(
                (theta_pred - tf.reshape(tr, [-1, 1])) / tf.reshape(denom, [-1, 1]),
                1e-6, 1.0 - 1e-6,
            )

            # Main loss: weighted MSE in Se-space
            loss_curve = tf.reduce_mean(tf.square(se_pred - se_obs) * w)

            # ψ50 loss in log-space
            # psi_at_Se output is already clipped to [PSI50_MIN, PSI50_MAX] in the
            # model, so log() is safe and the loss cannot explode.
            loss_psi50 = tf.constant(0.0, dtype=tf.float32)
            if psi50_obs is not None:
                psi50_pred = psi_at_Se(alpha, n, Se_target=0.5)
                # Clip observed psi50 to the same range for consistency
                psi50_obs_safe = tf.clip_by_value(psi50_obs, PSI50_MIN, PSI50_MAX)
                err = tf.math.log(psi50_pred) - tf.math.log(psi50_obs_safe)

                if use_huber:
                    # Huber loss: huber(y_true=0, y_pred=err, delta=0.5)
                    huber_fn   = tf.keras.losses.Huber(delta=0.5, reduction="none")
                    loss_psi50 = tf.reduce_mean(
                        huber_fn(tf.zeros_like(err), tf.expand_dims(err, -1))
                    )
                else:
                    loss_psi50 = tf.reduce_mean(tf.square(err))

            # Knee slope loss (optional) in log(ψ) space, transition band only
            loss_slope = tf.constant(0.0, dtype=tf.float32)
            if use_slope_loss:
                log_psi  = tf.math.log(tf.maximum(psi_tf, 1e-6))
                dlog     = tf.maximum(log_psi[:, 1:] - log_psi[:, :-1], 1e-8)
                dse_pred = (se_pred[:, 1:] - se_pred[:, :-1]) / dlog
                dse_obs  = (se_obs[:, 1:]  - se_obs[:, :-1])  / dlog
                mid_mask = tf.cast(
                    (se_obs[:, :-1] > 0.2) & (se_obs[:, :-1] < 0.8), tf.float32
                )
                loss_slope = tf.reduce_mean(tf.square(dse_pred - dse_obs) * mid_mask)

            if lambda_psi50_current is None:
                lambda_psi50_current = lambda_psi50
            loss = loss_curve + lambda_psi50_current * loss_psi50 + lambda_slope * loss_slope

            # Mask any non-finite loss terms rather than silently replacing with a
            # constant (which would zero the gradient and hide real numerical issues).
            loss = tf.where(tf.math.is_finite(loss), loss, tf.zeros_like(loss))

        grads = tape.gradient(loss, model.trainable_variables)
        # Replace None gradients (unused variables) with zeros; clip by global norm
        grads = [g if g is not None else tf.zeros_like(v)
                 for g, v in zip(grads, model.trainable_variables)]
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss, loss_curve, loss_psi50, loss_slope
    
    @tf.function
    def val_step(x, se_obs, ts, tr, w, psi50_obs=None, lambda_psi50_current=None):
        # Identical transforms to train_step (no extra clipping) so that
        # validation metrics match the training regime exactly.
        alpha, n = model(x, training=False)

        psi_tf = tf.tile(tf.reshape(psi, [1, -1]), [tf.shape(x)[0], 1])
        theta_pred = vg_theta(psi_tf, ts, tr, alpha, n)

        denom   = tf.maximum(ts - tr, 1e-6)
        se_pred = tf.clip_by_value(
            (theta_pred - tf.reshape(tr, [-1, 1])) / tf.reshape(denom, [-1, 1]),
            1e-6, 1.0 - 1e-6,
        )

        loss_curve = tf.reduce_mean(tf.square(se_pred - se_obs) * w)

        loss_psi50 = tf.constant(0.0, dtype=tf.float32)
        if psi50_obs is not None:
            psi50_pred     = psi_at_Se(alpha, n, Se_target=0.5)
            psi50_obs_safe = tf.clip_by_value(psi50_obs, PSI50_MIN, PSI50_MAX)
            err            = tf.math.log(psi50_pred) - tf.math.log(psi50_obs_safe)
            if use_huber:
                huber_fn   = tf.keras.losses.Huber(delta=0.5, reduction="none")
                loss_psi50 = tf.reduce_mean(
                    huber_fn(tf.zeros_like(err), tf.expand_dims(err, -1))
                )
            else:
                loss_psi50 = tf.reduce_mean(tf.square(err))

        loss_slope = tf.constant(0.0, dtype=tf.float32)
        if use_slope_loss:
            log_psi  = tf.math.log(tf.maximum(psi_tf, 1e-6))
            dlog     = tf.maximum(log_psi[:, 1:] - log_psi[:, :-1], 1e-8)
            dse_pred = (se_pred[:, 1:] - se_pred[:, :-1]) / dlog
            dse_obs  = (se_obs[:, 1:]  - se_obs[:, :-1])  / dlog
            mid_mask = tf.cast(
                (se_obs[:, :-1] > 0.2) & (se_obs[:, :-1] < 0.8), tf.float32
            )
            loss_slope = tf.reduce_mean(tf.square(dse_pred - dse_obs) * mid_mask)

        if lambda_psi50_current is None:
            lambda_psi50_current = lambda_psi50
        loss = loss_curve + lambda_psi50_current * loss_psi50 + lambda_slope * loss_slope
        loss = tf.where(tf.math.is_finite(loss), loss, tf.zeros_like(loss))
        return loss, loss_curve, loss_psi50, loss_slope
    
    # Dataset
    print("\n5. Creating datasets...")
    batch = 64
    
    if psi50_train is not None:
        train_ds = tf.data.Dataset.from_tensor_slices(
            (Xtr, Se_tr, theta_s_train, theta_r_train, W_tr, psi50_train)
        ).shuffle(len(Xtr)).batch(batch).prefetch(tf.data.AUTOTUNE)
        
        val_ds = tf.data.Dataset.from_tensor_slices(
            (Xva, Se_va, theta_s_val, theta_r_val, W_va, psi50_val)
        ).batch(batch).prefetch(tf.data.AUTOTUNE)
    else:
        train_ds = tf.data.Dataset.from_tensor_slices(
            (Xtr, Se_tr, theta_s_train, theta_r_train, W_tr)
        ).shuffle(len(Xtr)).batch(batch).prefetch(tf.data.AUTOTUNE)
        
        val_ds = tf.data.Dataset.from_tensor_slices(
            (Xva, Se_va, theta_s_val, theta_r_val, W_va)
        ).batch(batch).prefetch(tf.data.AUTOTUNE)
    
    print("   ✓ Datasets created")
    
    # Train loop
    print("\n6. Training...")
    if psi50_train is not None:
        print(f"   Using ψ50 loss (λ={lambda_psi50})")
    if use_slope_loss:
        print(f"   Using slope loss (λ={lambda_slope})")
    if use_huber:
        print(f"   Using Huber loss for ψ50")
    if use_curriculum:
        print(f"   Using curriculum scheduling for λ_ψ50")
    
    best = 1e9
    patience, bad = 15, 0
    for epoch in range(1, 201):
        # Curriculum scheduling: ramp up λ_ψ50 after epoch 10
        if use_curriculum:
            current_lambda_psi50 = 0.05 if epoch < 10 else lambda_psi50
        else:
            current_lambda_psi50 = lambda_psi50
        
        tr_losses = []
        tr_curve_losses = []
        tr_psi50_losses = []
        
        for batch in train_ds:
            if psi50_train is not None:
                x, se, ts, tr, w, psi50 = batch
                loss, loss_curve, loss_psi50, loss_slope = train_step(x, se, ts, tr, w, psi50, current_lambda_psi50)
            else:
                x, se, ts, tr, w = batch
                loss, loss_curve, loss_psi50, loss_slope = train_step(x, se, ts, tr, w, None, current_lambda_psi50)
            tr_losses.append(loss)
            tr_curve_losses.append(loss_curve)
            tr_psi50_losses.append(loss_psi50)
        
        va_losses = []
        va_curve_losses = []
        va_psi50_losses = []
        for batch in val_ds:
            if psi50_val is not None:
                x, se, ts, tr, w, psi50 = batch
                loss, loss_curve, loss_psi50, loss_slope = val_step(x, se, ts, tr, w, psi50, current_lambda_psi50)
            else:
                x, se, ts, tr, w = batch
                loss, loss_curve, loss_psi50, loss_slope = val_step(x, se, ts, tr, w, None, current_lambda_psi50)
            va_losses.append(loss)
            va_curve_losses.append(loss_curve)
            va_psi50_losses.append(loss_psi50)
        
        tr_loss = float(tf.reduce_mean(tr_losses))
        va_loss = float(tf.reduce_mean(va_losses))
        tr_curve = float(tf.reduce_mean(tr_curve_losses))
        tr_psi50 = float(tf.reduce_mean(tr_psi50_losses)) if psi50_train is not None else 0.0
        
        # Check for NaN in weights
        has_nan = False
        for w in model.get_weights():
            if np.any(np.isnan(w)):
                has_nan = True
                break
        
        if has_nan:
            print(f"   Epoch {epoch:03d} | NaN detected in weights! Stopping training.")
            break
        
        if psi50_train is not None:
            print(f"   Epoch {epoch:03d} | train {tr_loss:.6f} (curve: {tr_curve:.6f}, ψ50: {tr_psi50:.6f}) | val {va_loss:.6f}")
        else:
            print(f"   Epoch {epoch:03d} | train {tr_loss:.6f} | val {va_loss:.6f}")
        
        if va_loss < best - 1e-6 and not np.isnan(va_loss):
            best = va_loss
            bad = 0
            model.save(out_dir / "vgparamnet_best.keras")
            print(f"     → New best validation loss: {best:.6f}")
        else:
            bad += 1
            if bad >= patience:
                print(f"   Early stopping at epoch {epoch}")
                break
    
    # Load best, predict test θ curves
    # Identical transforms to train/val steps — no extra clipping.
    print("\n6. Evaluating on test set...")
    best_model = tf.keras.models.load_model(
        out_dir / "vgparamnet_best.keras",
        custom_objects={"VGParamNet": VGParamNet},
        compile=False,
    )
    alpha, n   = best_model(Xte, training=False)
    psi_tf     = tf.tile(tf.reshape(psi, [1, -1]), [tf.shape(alpha)[0], 1])
    theta_pred = vg_theta(psi_tf, theta_s_test, theta_r_test, alpha, n).numpy()

    # Report predicted parameter ranges for sanity check
    print(f"   alpha: min={alpha.numpy().min():.4f}  max={alpha.numpy().max():.4f}  "
          f"median={float(np.median(alpha.numpy())):.4f}")
    print(f"   n:     min={n.numpy().min():.4f}  max={n.numpy().max():.4f}  "
          f"median={float(np.median(n.numpy())):.4f}")
    
    # Save results with run ID
    results_subdir = out_dir / f"run_{run_id}"
    results_subdir.mkdir(parents=True, exist_ok=True)
    
    np.save(results_subdir / "theta_vgparamnet_test.npy", theta_pred)
    np.save(results_subdir / "alpha_test.npy", alpha.numpy())
    np.save(results_subdir / "n_test.npy", n.numpy())
    
    # Also save to main directory (for compatibility)
    np.save(out_dir / "theta_vgparamnet_test.npy", theta_pred)
    np.save(out_dir / "alpha_test.npy", alpha.numpy())
    np.save(out_dir / "n_test.npy", n.numpy())
    
    # Save training config
    config = {
        'run_id': run_id,
        'lambda_psi50': float(lambda_psi50),
        'lambda_slope': float(lambda_slope),
        'use_huber': use_huber,
        'use_curriculum': use_curriculum,
    }
    with open(results_subdir / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Saved results to: {out_dir}")
    print(f"  - Run {run_id} results in: {results_subdir}")
    print(f"  - theta_vgparamnet_test.npy: predicted θ(ψ) curves")
    print(f"  - alpha_test.npy: predicted α parameters")
    print(f"  - n_test.npy: predicted n parameters")
    print(f"  - vgparamnet_best.keras: trained model")
    print(f"  - training_config.json: training configuration")


if __name__ == "__main__":
    main()
