"""
Physics Constraints for GAN Training
Enforces physical laws on generated SWCC curves
"""

import tensorflow as tf
import numpy as np


class PhysicsConstraints:
    """Physics constraints for SWCC curves"""
    
    def __init__(self, lambda_mono=0.5, lambda_bound=0.3):
        """
        Initialize physics constraints
        
        Args:
            lambda_mono: Weight for monotonicity loss
            lambda_bound: Weight for boundary loss
        """
        self.lambda_mono = lambda_mono
        self.lambda_bound = lambda_bound
    
    def monotonicity_loss(self, swcc_curve):
        """
        Enforce monotonicity: ∂θ/∂ψ ≤ 0
        SWCC must be decreasing (water content decreases with increasing suction)
        
        Args:
            swcc_curve: Generated SWCC curve [batch_size, n_points]
            
        Returns:
            Loss value (scalar)
        """
        # Compute differences: θ[i] - θ[i+1]
        # For decreasing curve: θ[i] - θ[i+1] ≥ 0
        diff = swcc_curve[:, :-1] - swcc_curve[:, 1:]
        
        # Violations: negative differences (increasing parts)
        violations = tf.maximum(0.0, -diff)
        
        # Average violation per curve
        loss = tf.reduce_mean(violations)
        
        return loss
    
    def boundary_loss(self, swcc_curve, theta_s, theta_r):
        """
        Enforce boundary conditions: θ_r ≤ θ ≤ θ_s
        
        Args:
            swcc_curve: Generated SWCC curve [batch_size, n_points]
            theta_s: Saturated water content [batch_size, 1]
            theta_r: Residual water content [batch_size, 1]
            
        Returns:
            Loss value (scalar)
        """
        # Ensure column vectors for broadcasting
        theta_s = tf.reshape(theta_s, [-1, 1])  # [batch_size, 1]
        theta_r = tf.reshape(theta_r, [-1, 1])  # [batch_size, 1]
        
        # Lower bound violations: θ < θ_r
        lower_violations = tf.maximum(0.0, theta_r - swcc_curve)
        
        # Upper bound violations: θ > θ_s
        upper_violations = tf.maximum(0.0, swcc_curve - theta_s)
        
        # Total violations
        total_violations = lower_violations + upper_violations
        
        # Average violation
        loss = tf.reduce_mean(total_violations)
        
        return loss
    
    def compute_physics_loss(self, swcc_curve, theta_s, theta_r):
        """
        Compute total physics loss
        
        Args:
            swcc_curve: Generated SWCC curve [batch_size, n_points]
            theta_s: Saturated water content [batch_size, 1] or [batch_size]
            theta_r: Residual water content [batch_size, 1] or [batch_size]
            
        Returns:
            Total physics loss (scalar)
        """
        # Ensure theta_s and theta_r are 1D
        if len(theta_s.shape) > 1:
            theta_s = tf.squeeze(theta_s)
        if len(theta_r.shape) > 1:
            theta_r = tf.squeeze(theta_r)
        
        # Compute individual losses
        mono_loss = self.monotonicity_loss(swcc_curve)
        bound_loss = self.boundary_loss(swcc_curve, theta_s, theta_r)
        
        # Weighted sum
        total_loss = (self.lambda_mono * mono_loss + 
                     self.lambda_bound * bound_loss)
        
        return total_loss, mono_loss, bound_loss
    
    def enforce_monotonicity(self, swcc_curve):
        """
        Post-process to enforce monotonicity (non-trainable)
        
        Args:
            swcc_curve: SWCC curve [batch_size, n_points] or [n_points]
            
        Returns:
            Monotonic curve
        """
        # Ensure decreasing: θ[i] ≥ θ[i+1]
        curve = tf.identity(swcc_curve)
        
        # For each sample, ensure decreasing
        for i in range(1, curve.shape[-1]):
            curve = tf.concat([
                curve[..., :i],
                tf.minimum(curve[..., i-1:i], curve[..., i:i+1]),
                curve[..., i+1:]
            ], axis=-1)
        
        return curve
    
    def enforce_boundaries(self, swcc_curve, theta_s, theta_r):
        """
        Post-process to enforce boundaries (non-trainable)
        
        Args:
            swcc_curve: SWCC curve [batch_size, n_points] or [n_points]
            theta_s: Saturated water content
            theta_r: Residual water content
            
        Returns:
            Bounded curve
        """
        # Clip to boundaries
        curve = tf.clip_by_value(swcc_curve, theta_r, theta_s)
        
        return curve
    
    def validate_curve(self, swcc_curve, theta_s, theta_r, suction_grid=None):
        """
        Validate if curve satisfies physics constraints
        
        Args:
            swcc_curve: SWCC curve [n_points] or [batch_size, n_points]
            theta_s: Saturated water content
            theta_r: Residual water content
            suction_grid: Optional suction values for detailed analysis
            
        Returns:
            Dictionary with validation results
        """
        # Ensure 2D
        if len(swcc_curve.shape) == 1:
            swcc_curve = tf.expand_dims(swcc_curve, 0)
        
        # Check monotonicity
        diff = swcc_curve[:, :-1] - swcc_curve[:, 1:]
        is_monotonic = tf.reduce_all(diff >= -1e-6, axis=1)  # Allow small numerical errors
        
        # Check boundaries
        within_bounds = tf.reduce_all(
            (swcc_curve >= theta_r - 1e-6) & (swcc_curve <= theta_s + 1e-6),
            axis=1
        )
        
        # Compute statistics
        mono_rate = tf.reduce_mean(tf.cast(is_monotonic, tf.float32))
        bound_rate = tf.reduce_mean(tf.cast(within_bounds, tf.float32))
        
        results = {
            'is_monotonic': is_monotonic.numpy() if hasattr(is_monotonic, 'numpy') else is_monotonic,
            'within_bounds': within_bounds.numpy() if hasattr(within_bounds, 'numpy') else within_bounds,
            'monotonicity_rate': float(mono_rate),
            'boundary_satisfaction_rate': float(bound_rate),
            'all_valid': tf.reduce_all(is_monotonic & within_bounds)
        }
        
        return results
