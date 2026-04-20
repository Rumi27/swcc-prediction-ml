"""
Physics Loss Functions for PINN
Implements monotonicity, boundary, and Arya-Paris physics constraints
"""

import tensorflow as tf
import numpy as np


def monotonicity_loss(theta_pred, suction):
    """
    Enforce monotonicity: ∂θ/∂ψ ≤ 0
    Water content must decrease with increasing suction
    
    Args:
        theta_pred: Predicted water content [batch, n_points]
        suction: Suction values [batch, n_points] or [n_points]
    
    Returns:
        Monotonicity loss (scalar)
    """
    # Ensure 2D
    if len(suction.shape) == 1:
        suction = tf.expand_dims(suction, 0)
        batch_size = tf.shape(theta_pred)[0]
        suction = tf.tile(suction, [batch_size, 1])
    
    # Compute gradient: ∂θ/∂ψ
    # For discrete points: (θ[i+1] - θ[i]) / (ψ[i+1] - ψ[i])
    theta_diff = theta_pred[:, 1:] - theta_pred[:, :-1]  # θ[i+1] - θ[i]
    suction_diff = suction[:, 1:] - suction[:, :-1]  # ψ[i+1] - ψ[i]
    
    # Avoid division by zero
    suction_diff = tf.maximum(suction_diff, 1e-8)
    gradient = theta_diff / suction_diff
    
    # Violations: positive gradients (increasing water content with suction)
    violations = tf.maximum(0.0, gradient)
    
    # Average violation
    loss = tf.reduce_mean(violations)
    
    return loss


def boundary_loss(theta_pred, theta_s, theta_r):
    """
    Enforce boundary conditions: θ_r ≤ θ ≤ θ_s
    
    Args:
        theta_pred: Predicted water content [batch, n_points]
        theta_s: Saturated water content [batch] or [batch, 1]
        theta_r: Residual water content [batch] or [batch, 1]
    
    Returns:
        Boundary loss (scalar)
    """
    # Ensure 2D
    if len(theta_s.shape) == 1:
        theta_s = tf.expand_dims(theta_s, axis=1)
    if len(theta_r.shape) == 1:
        theta_r = tf.expand_dims(theta_r, axis=1)
    
    # Lower bound violations: θ < θ_r
    lower_violations = tf.maximum(0.0, theta_r - theta_pred)
    
    # Upper bound violations: θ > θ_s
    upper_violations = tf.maximum(0.0, theta_pred - theta_s)
    
    # Total violations
    total_violations = lower_violations + upper_violations
    
    # Average violation
    loss = tf.reduce_mean(total_violations)
    
    return loss


def arya_paris_physics_loss(theta_pred, suction, soil_props, alpha=1.38):
    """
    Enforce Arya-Paris physics: relationship between GSD and pore size
    
    Args:
        theta_pred: Predicted water content [batch, n_points]
        suction: Suction values [batch, n_points]
        soil_props: Soil properties [batch, soil_prop_dim]
        alpha: Arya-Paris parameter
    
    Returns:
        Physics loss (scalar, normalized)
    """
    # Extract GSD features (based on metadata order: D10=2, D30=3, D50=4, D60=5, D90=6)
    gsd = soil_props[:, 2:7]  # D10, D30, D50, D60, D90
    bulk_density = soil_props[:, 8]  # Bulk density (index 8)
    porosity = soil_props[:, 11]  # Porosity (index 11)
    
    # Clip porosity to valid range
    porosity = tf.clip_by_value(porosity, 0.01, 0.99)
    
    # Compute expected pore radius from Arya-Paris
    # Note: GSD might be normalized, so use absolute value
    d50 = tf.abs(gsd[:, 2]) + 1e-6  # D50, ensure positive
    d50_m = d50 * 1e-6  # Convert from μm to meters (if not normalized)
    
    porosity_ratio = porosity / (1.0 - porosity + 1e-8)
    porosity_ratio = tf.maximum(porosity_ratio, 1e-8)  # Ensure positive
    r_pore_expected = alpha * d50_m * tf.sqrt(porosity_ratio)
    r_pore_expected = tf.maximum(r_pore_expected, 1e-9)
    
    # Expected capillary pressure from Young-Laplace
    surface_tension = 0.072  # N/m
    psi_expected = 2.0 * surface_tension / r_pore_expected  # Pa
    psi_expected_kpa = psi_expected / 1000.0  # kPa
    
    # Compare with actual suction values
    # Use the first point (highest water content, lowest suction) as air-entry value
    suction_first = suction[:, 0]  # First suction point [batch]
    
    # Normalized loss: relative difference (to avoid scale issues)
    # Use log scale for better numerical stability
    psi_expected_log = tf.math.log(psi_expected_kpa + 1e-3)  # Add small value to avoid log(0)
    suction_first_log = tf.math.log(suction_first + 1e-3)
    
    # Loss: normalized difference in log space
    loss = tf.reduce_mean(tf.abs(psi_expected_log - suction_first_log))
    
    return loss


def data_loss(theta_pred, theta_obs):
    """
    Data fitting loss: MSE between predicted and observed
    
    Args:
        theta_pred: Predicted water content [batch, n_points]
        theta_obs: Observed water content [batch, n_points]
    
    Returns:
        MSE loss (scalar)
    """
    return tf.reduce_mean(tf.square(theta_pred - theta_obs))


def compute_total_loss(theta_pred, theta_obs, suction, soil_props, theta_s, theta_r,
                       lambda_data=1.0, lambda_mono=0.5, lambda_bound=0.3, lambda_physics=0.2):
    """
    Compute total PINN loss
    
    Args:
        theta_pred: Predicted water content [batch, n_points]
        theta_obs: Observed water content [batch, n_points]
        suction: Suction values [batch, n_points]
        soil_props: Soil properties [batch, soil_prop_dim]
        theta_s: Saturated water content [batch]
        theta_r: Residual water content [batch]
        lambda_data: Weight for data loss
        lambda_mono: Weight for monotonicity loss
        lambda_bound: Weight for boundary loss
        lambda_physics: Weight for physics loss
    
    Returns:
        Dictionary with all losses
    """
    # Data loss
    L_data = data_loss(theta_pred, theta_obs)
    
    # Physics losses
    L_mono = monotonicity_loss(theta_pred, suction)
    L_bound = boundary_loss(theta_pred, theta_s, theta_r)
    L_physics = arya_paris_physics_loss(theta_pred, suction, soil_props)
    
    # Total loss
    L_total = (lambda_data * L_data + 
               lambda_mono * L_mono + 
               lambda_bound * L_bound + 
               lambda_physics * L_physics)
    
    return {
        'total': L_total,
        'data': L_data,
        'monotonicity': L_mono,
        'boundary': L_bound,
        'physics': L_physics
    }
