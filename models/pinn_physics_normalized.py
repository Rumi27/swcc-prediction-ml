"""
Physics Loss Functions in Normalized Space
All losses operate on normalized [0,1] water content
"""

import tensorflow as tf
import numpy as np


def knee_weights_from_observed(theta_obs_norm,
                               suction,
                               beta=3.0,
                               eps=1e-6,
                               clip_min=1.0,
                               clip_max=8.0):
    """
    Per-point weights that emphasize the knee region using
    |dθ_obs/dlog10(s)| in normalized space.
    
    Args:
        theta_obs_norm: [B, P] observed θ_norm in [0,1]
        suction:        [B, P] or [P] suction (kPa)
        beta:           strength of knee emphasis
        clip_min/max:   bounds for numerical stability
    Returns:
        weights: [B, P] >= 1
    """
    # Ensure suction is [B, P]
    if len(suction.shape) == 1:
        suction = tf.expand_dims(suction, 0)
        batch_size = tf.shape(theta_obs_norm)[0]
        suction = tf.tile(suction, [batch_size, 1])
    
    # log10(s) with stability
    log_s = tf.math.log(suction + eps) / tf.math.log(tf.constant(10.0, dtype=suction.dtype))
    
    # finite differences in log space
    dtheta = theta_obs_norm[:, 1:] - theta_obs_norm[:, :-1]          # [B, P-1]
    dlog = tf.maximum(log_s[:, 1:] - log_s[:, :-1], eps)             # [B, P-1]
    slope = tf.abs(dtheta / dlog)                                    # [B, P-1]
    
    # pad to [B, P]
    slope_padded = tf.concat([slope, slope[:, -1:]], axis=1)         # [B, P]
    
    # normalize per sample
    mean_slope = tf.reduce_mean(slope_padded, axis=1, keepdims=True) + eps
    slope_rel = slope_padded / mean_slope                            # [B, P]
    
    # baseline 1 + knee emphasis
    weights = 1.0 + beta * slope_rel
    weights = tf.clip_by_value(weights, clip_min, clip_max)
    
    return weights


def data_loss(theta_pred_norm,
              theta_obs_norm,
              suction,
              s0_weight=2.0,
              beta_knee=3.0,
              eps=1e-6):
    """
    Weighted MSE emphasizing:
      - saturated end (first point, via s0_weight)
      - knee/transition via |dθ_obs/dlog10(s)| per sample
    
    Args:
        theta_pred_norm: Predicted normalized water content [batch, n_points] in [0,1]
        theta_obs_norm: Observed normalized water content [batch, n_points] in [0,1]
        suction: Suction values [batch, n_points] or [n_points]
        s0_weight: Weight for error at first suction point (s=0, saturated)
        beta_knee: Strength of knee emphasis
    
    Returns:
        Weighted MSE loss (scalar)
    """
    # Squared error
    se = tf.square(theta_pred_norm - theta_obs_norm)
    
    # Knee weights from observed curve
    w_knee = knee_weights_from_observed(theta_obs_norm, suction, beta=beta_knee, eps=eps)
    
    # Wet-end anchor for first point
    B = tf.shape(se)[0]
    P = tf.shape(se)[1]
    first = tf.ones([B, 1], dtype=se.dtype) * s0_weight
    rest = tf.ones([B, P - 1], dtype=se.dtype)
    w_s0 = tf.concat([first, rest], axis=1)
    
    weights = w_knee * w_s0
    return tf.reduce_mean(se * weights)


def monotonicity_loss_normalized(theta_pred_norm, suction):
    """
    Enforce monotonicity in normalized space: ∂θ_norm/∂s ≤ 0
    
    Args:
        theta_pred_norm: Normalized water content [batch, n_points] in [0,1]
        suction: Suction values [batch, n_points]
    
    Returns:
        Monotonicity loss (scalar)
    """
    # Work in log(s) space to reflect SWCC behavior on log-suction axis
    if len(suction.shape) == 1:
        suction = tf.expand_dims(suction, 0)
        batch_size = tf.shape(theta_pred_norm)[0]
        suction = tf.tile(suction, [batch_size, 1])

    log_s = tf.math.log(suction + 1e-6)  # natural log

    theta_diff = theta_pred_norm[:, 1:] - theta_pred_norm[:, :-1]  # θ[i+1] - θ[i]
    log_s_diff = tf.maximum(log_s[:, 1:] - log_s[:, :-1], 1e-8)    # Δ log(s)

    gradient = theta_diff / log_s_diff  # dθ/d(log s)
    
    # Violations: positive gradients (increasing water content with suction)
    violations = tf.maximum(0.0, gradient)
    
    # Average violation
    loss = tf.reduce_mean(violations)
    
    return loss


def boundary_loss_normalized(theta_pred_norm, suction_grid):
    """
    Enforce boundary conditions in normalized space:
    - At s=0 (first point): θ_norm ≈ 1.0 (saturated)
    - At s_max (last point): θ_norm ≈ 0.0 (residual)
    
    Args:
        theta_pred_norm: Normalized water content [batch, n_points] in [0,1]
        suction_grid: Suction grid [n_points] or [batch, n_points]
    
    Returns:
        Boundary loss (scalar)
    """
    # Ensure suction_grid is 2D
    if len(suction_grid.shape) == 1:
        suction_grid = tf.expand_dims(suction_grid, 0)
        batch_size = tf.shape(theta_pred_norm)[0]
        suction_grid = tf.tile(suction_grid, [batch_size, 1])
    
    # First point (lowest suction, should be saturated)
    theta_at_s0 = theta_pred_norm[:, 0]  # [batch]
    loss_s0 = tf.reduce_mean(tf.square(theta_at_s0 - 1.0))
    
    # Last point (highest suction, should be residual)
    theta_at_smax = theta_pred_norm[:, -1]  # [batch]
    loss_smax = tf.reduce_mean(tf.square(theta_at_smax - 0.0))
    
    # Also check that values stay in [0, 1]
    lower_violations = tf.maximum(0.0, -theta_pred_norm)  # θ < 0
    upper_violations = tf.maximum(0.0, theta_pred_norm - 1.0)  # θ > 1
    loss_bounds = tf.reduce_mean(lower_violations + upper_violations)
    
    total_loss = loss_s0 + loss_smax + 0.1 * loss_bounds
    
    return total_loss


def arya_paris_physics_loss_normalized(theta_pred_norm, suction, soil_props, alpha=1.38):
    """
    Enforce Arya-Paris physics in normalized space.
    Compares predicted air-entry value with expected from GSD.
    
    Args:
        theta_pred_norm: Normalized water content [batch, n_points] in [0,1]
        suction: Suction values [batch, n_points]
        soil_props: Soil properties [batch, soil_prop_dim]
        alpha: Arya-Paris parameter
    
    Returns:
        Physics loss (scalar, normalized)
    """
    # Extract GSD features
    feature_cols = ['Cc', 'Cu', 'D10', 'D30', 'D50', 'D60', 'D90', 'OM_content', 
                   'bulk_density', 'clay_pct', 'pH', 'porosity', 'sand_pct', 
                   'silt_pct', 'theta_r', 'theta_s']
    
    d50_idx = feature_cols.index('D50')
    bulk_density_idx = feature_cols.index('bulk_density')
    porosity_idx = feature_cols.index('porosity')
    
    d50 = tf.abs(tf.gather(soil_props, d50_idx, axis=1))  # [batch]
    bulk_density = tf.gather(soil_props, bulk_density_idx, axis=1)  # [batch]
    porosity = tf.clip_by_value(tf.gather(soil_props, porosity_idx, axis=1), 0.01, 0.99)  # [batch]
    
    # Convert D50 to meters (assuming it's in micrometers, may need adjustment)
    # For normalized inputs, we'll use a simplified approach
    d50_m = tf.abs(d50) * 1e-6  # Convert to meters (adjust if needed)
    d50_m = tf.maximum(d50_m, 1e-9)  # Avoid zero
    
    # Arya-Paris pore radius
    porosity_ratio = porosity / (1.0 - porosity + 1e-8)
    porosity_ratio = tf.maximum(porosity_ratio, 1e-8)
    r_pore_expected = alpha * d50_m * tf.sqrt(porosity_ratio)
    r_pore_expected = tf.maximum(r_pore_expected, 1e-9)
    
    # Expected capillary pressure
    surface_tension = 0.072  # N/m
    psi_expected = 2.0 * surface_tension / r_pore_expected  # Pa
    psi_expected_kpa = psi_expected / 1000.0  # kPa [batch]
    
    # Compare with first suction point (air-entry value)
    suction_first = suction[:, 0]  # [batch]
    
    # Normalized loss in log space
    psi_expected_log = tf.math.log(psi_expected_kpa + 1e-3)
    suction_first_log = tf.math.log(suction_first + 1e-3)
    
    loss = tf.reduce_mean(tf.abs(psi_expected_log - suction_first_log))
    
    return loss


def compute_total_loss(theta_pred_norm, theta_obs_norm, suction, soil_props, 
                       lambda_data=1.0, lambda_mono=0.1, lambda_bound=0.3, 
                       lambda_physics=0.0, s0_weight=2.0):
    """
    Compute total loss in normalized space.
    
    Args:
        theta_pred_norm: Predicted normalized water content [batch, n_points]
        theta_obs_norm: Observed normalized water content [batch, n_points]
        suction: Suction values [batch, n_points]
        soil_props: Soil properties [batch, soil_prop_dim]
        lambda_*: Loss weights
        s0_weight: Weight for data loss at s=0 (saturated end)
    
    Returns:
        Dictionary of losses
    """
    d_loss = data_loss(
        theta_pred_norm,
        theta_obs_norm,
        suction,
        s0_weight=s0_weight,
    )
    m_loss = monotonicity_loss_normalized(theta_pred_norm, suction)
    b_loss = boundary_loss_normalized(theta_pred_norm, suction)
    # For now, disable Arya–Paris term in the main loss to avoid potential bias;
    # it can be re-enabled once calibrated.
    # p_loss = arya_paris_physics_loss_normalized(theta_pred_norm, suction, soil_props)
    p_loss = tf.constant(0.0, dtype=theta_pred_norm.dtype)
    
    total_loss = (lambda_data * d_loss +
                 lambda_mono * m_loss +
                 lambda_bound * b_loss +
                 lambda_physics * p_loss)
    
    return {
        'total': total_loss,
        'data': d_loss,
        'monotonicity': m_loss,
        'boundary': b_loss,
        'physics': p_loss
    }
