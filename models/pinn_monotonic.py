"""
PINN with Structural Monotonicity
Uses cumulative-sum architecture to guarantee monotonicity
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from models.pinn import PhysicsEncodingLayer


class MonotonicPINN(keras.Model):
    """
    Physics-Informed Neural Network with Structural Monotonicity
    
    Architecture:
    - Physics encoding layer (Arya-Paris)
    - Hidden layers for feature extraction
    - Monotonic output head using cumulative sum:
      θ(s_k) = θ_s - Σ_{i=1}^k Δθ_i
      where Δθ_i ≥ 0 (enforced by softplus/ReLU)
    """
    
    def __init__(self, soil_prop_dim, suction_points, physics_units=128, 
                 hidden_dims=[128, 256, 128, 64], **kwargs):
        super(MonotonicPINN, self).__init__(**kwargs)
        self.soil_prop_dim = soil_prop_dim
        self.suction_points = suction_points
        self.physics_units = physics_units
        self.hidden_dims = hidden_dims
        
        # Physics encoding layer
        self.physics_encoding = PhysicsEncodingLayer(units=physics_units, name='physics_encoding')
        
        # Feature extraction layers (before monotonic head)
        self.feature_layers = []
        for i, dim in enumerate(hidden_dims[:-1]):  # All except last
            self.feature_layers.append(
                layers.Dense(dim, activation='relu', name=f'feature_{i+1}')
            )
            self.feature_layers.append(
                layers.BatchNormalization(name=f'feature_bn_{i+1}')
            )
        
        # Monotonic head: outputs one non-negative increment per suction point
        # Input: [batch*n_suction, last_hidden_dim]
        # Output: [batch*n_suction, 1] - one increment per point
        self.increment_head = layers.Dense(
            1,  # One increment value per input point
            activation='softplus',  # Ensures Δθ_i ≥ 0
            name='increment_head'
        )
        
        # Optional: residual connection for flexibility
        self.residual_head = layers.Dense(
            1,  # One residual value per input point
            activation='linear',
            name='residual_head'
        )
        
    def call(self, inputs, training=None):
        """
        Forward pass with structural monotonicity
        
        Args:
            inputs: Dict with 'soil_props' and 'suction'
            training: Training flag
        
        Returns:
            theta_norm: Normalized water content [batch, n_suction] in [0, 1]
        """
        soil_props = inputs['soil_props']
        suction = inputs['suction']
        
        # Ensure suction is 2D
        if len(suction.shape) == 1:
            suction = tf.expand_dims(suction, 0)
            batch_size = tf.shape(soil_props)[0]
            suction = tf.tile(suction, [batch_size, 1])
        
        batch_size = tf.shape(soil_props)[0]
        n_suction = tf.shape(suction)[1]
        
        # Extract features for physics encoding
        feature_cols = ['Cc', 'Cu', 'D10', 'D30', 'D50', 'D60', 'D90', 'OM_content', 
                       'bulk_density', 'clay_pct', 'pH', 'porosity', 'sand_pct', 
                       'silt_pct', 'theta_r', 'theta_s']
        
        gsd_indices = [feature_cols.index(f) for f in ['D10', 'D30', 'D50', 'D60', 'D90']]
        bulk_density_idx = feature_cols.index('bulk_density')
        porosity_idx = feature_cols.index('porosity')
        
        gsd = tf.gather(soil_props, gsd_indices, axis=1)  # [batch, 5]
        bulk_density = tf.expand_dims(tf.gather(soil_props, bulk_density_idx, axis=1), axis=1)  # [batch, 1]
        porosity = tf.expand_dims(tf.gather(soil_props, porosity_idx, axis=1), axis=1)  # [batch, 1]
        
        other_feature_indices = sorted(list(set(range(self.soil_prop_dim)) - 
                                            set(gsd_indices + [bulk_density_idx, porosity_idx])))
        other_features = tf.gather(soil_props, other_feature_indices, axis=1)  # [batch, other_dim]
        
        physics_input = {
            'gsd': gsd,
            'bulk_density': bulk_density,
            'porosity': porosity,
            'other_features': other_features
        }
        physics_features = self.physics_encoding(physics_input)  # [batch, physics_units]
        
        # For each suction point, extract features
        # We'll use the physics features + suction value
        physics_expanded = tf.tile(tf.expand_dims(physics_features, 1), [1, n_suction, 1])  # [batch, n_suction, physics_units]
        suction_expanded = tf.expand_dims(suction, 2)  # [batch, n_suction, 1]
        combined = tf.concat([physics_expanded, suction_expanded], axis=2)  # [batch, n_suction, physics_units+1]
        
        # Flatten for feature extraction
        combined_flat = tf.reshape(combined, [-1, self.physics_units + 1])  # [batch*n_suction, physics_units+1]
        
        # Feature extraction
        x = combined_flat
        for layer in self.feature_layers:
            x = layer(x, training=training)
        
        # Monotonic head: output one increment per suction point
        # Each point gets one increment value
        delta_theta_base = self.increment_head(x)  # [batch*n_suction, 1]
        # Reshape to [batch, n_suction]
        delta_theta_base = tf.reshape(delta_theta_base, [batch_size, n_suction])
        
        # Optional residual for flexibility (small contribution)
        residual = self.residual_head(x)  # [batch*n_suction, 1]
        residual = tf.reshape(residual, [batch_size, n_suction])
        # Apply softplus to keep non-negative
        residual = tf.nn.softplus(residual) * 0.1
        
        # Combine base and residual
        delta_theta = delta_theta_base + residual  # [batch, n_suction], all ≥ 0
        
        # Normalize increments so they sum to at most 1 (to keep output in [0,1])
        # Sum of increments should be ≤ 1 to ensure θ_norm stays in [0,1]
        delta_sum = tf.reduce_sum(delta_theta, axis=1, keepdims=True)  # [batch, 1]
        delta_theta_normalized = delta_theta / (delta_sum + 1e-8)  # Normalize to sum to 1
        
        # Cumulative sum: θ(s_k) = 1 - Σ_{i=0}^{k-1} Δθ_i
        # Start at 1 (saturated) and decrease
        theta_norm = 1.0 - tf.cumsum(delta_theta_normalized, axis=1)  # [batch, n_suction]
        
        # Ensure it stays in [0, 1]
        theta_norm = tf.clip_by_value(theta_norm, 0.0, 1.0)
        
        return theta_norm
    
    def get_config(self):
        """Get model configuration"""
        config = super().get_config()
        config.update({
            'soil_prop_dim': self.soil_prop_dim,
            'suction_points': self.suction_points,
            'physics_units': self.physics_units,
            'hidden_dims': self.hidden_dims
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create model from config"""
        soil_prop_dim = config.pop('soil_prop_dim', 16)
        suction_points = config.pop('suction_points', 100)
        physics_units = config.pop('physics_units', 128)
        hidden_dims = config.pop('hidden_dims', [128, 256, 128, 64])
        
        model = cls(
            soil_prop_dim=soil_prop_dim,
            suction_points=suction_points,
            physics_units=physics_units,
            hidden_dims=hidden_dims,
            name=config.get('name', 'MonotonicPINN')
        )
        
        return model
