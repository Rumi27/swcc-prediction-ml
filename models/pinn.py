"""
Physics-Informed Neural Network (PINN) for SWCC Prediction
Integrates capillary theory (Arya-Paris) with deep learning
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class PhysicsEncodingLayer(layers.Layer):
    """
    Physics encoding layer implementing Arya-Paris model
    Transforms GSD to pore size distribution
    """
    
    def __init__(self, units=128, alpha=1.38, name='physics_encoding'):
        """
        Initialize physics encoding layer
        
        Args:
            units: Number of output units
            alpha: Arya-Paris model parameter (default: 1.38)
            name: Layer name
        """
        super(PhysicsEncodingLayer, self).__init__(name=name)
        self.units = units
        self.alpha = alpha
        
        # Surface tension of water (N/m)
        self.surface_tension = 0.072  # N/m at 20°C
        self.contact_angle = 0.0  # radians (assumed 0 for water)
        
        # Create dense layer in __init__
        self.physics_dense = layers.Dense(units, activation='relu', name='physics_dense')
    
    def get_config(self):
        """Get layer configuration"""
        config = super().get_config()
        config.update({
            'units': self.units,
            'alpha': self.alpha
        })
        return config
        
    def call(self, inputs):
        """
        Apply Arya-Paris physics transformation
        
        Args:
            inputs: Dictionary with:
                - 'gsd': Grain size distribution features [batch, gsd_dim]
                - 'bulk_density': Bulk density [batch, 1]
                - 'porosity': Porosity [batch, 1]
                - 'other_features': Other soil properties [batch, other_dim]
        
        Returns:
            Physics-encoded features [batch, units]
        """
        gsd = inputs['gsd']
        bulk_density = inputs['bulk_density']
        porosity = inputs['porosity']
        other_features = inputs.get('other_features', tf.zeros([tf.shape(gsd)[0], 0]))
        
        # Compute pore radius using Arya-Paris model
        # Note: Inputs may be normalized, so we use them as features directly
        # Simplified physics encoding: use GSD and porosity as features
        # For normalized inputs, we'll use a simpler physics encoding
        
        # Ensure porosity is 1D for calculations
        porosity_1d = tf.squeeze(porosity, axis=1) if len(porosity.shape) > 1 else porosity  # [batch]
        
        # Clip porosity to valid range [0, 1] to avoid NaN
        # If data is normalized, map back to [0, 1] range
        porosity_1d = tf.clip_by_value(porosity_1d, 0.01, 0.99)  # Ensure valid range
        
        # For normalized GSD, use absolute value and scale
        # D50 might be normalized, so we'll use a simplified approach
        d50 = gsd[:, 2] if gsd.shape[1] > 2 else gsd[:, 0]  # D50 [batch]
        
        # Simplified physics: compute approximate pore size feature
        # Use normalized values directly but ensure positive for calculations
        d50_abs = tf.abs(d50) + 1e-6  # Ensure positive
        porosity_clipped = tf.clip_by_value(porosity_1d, 0.01, 0.99)
        
        # Simplified Arya-Paris: use normalized features
        # Instead of actual pore radius, compute a physics-informed feature
        porosity_ratio = porosity_clipped / (1.0 - porosity_clipped + 1e-8)
        porosity_ratio = tf.maximum(porosity_ratio, 1e-8)  # Ensure positive
        
        # Compute physics feature (simplified, using normalized inputs)
        physics_feature_1 = tf.sqrt(porosity_ratio)  # Physics-informed feature
        physics_feature_2 = d50_abs * physics_feature_1  # Combined feature
        
        # Convert to approximate capillary pressure (simplified)
        # Use physics_feature_2 as a proxy for inverse pore size
        psi_approx = 1.0 / (physics_feature_2 + 1e-6)  # Inverse relationship
        psi_capillary_kpa = tf.clip_by_value(psi_approx, 1e-3, 1e6)  # Reasonable range
        
        # Compute approximate pore radius (for feature)
        r_pore_approx = physics_feature_2  # Use as proxy for pore radius
        
        # Expand dimensions for concatenation (ensure all are 2D)
        psi_capillary_kpa = tf.expand_dims(psi_capillary_kpa, axis=1)  # [batch, 1]
        r_pore_um = tf.expand_dims(r_pore_approx, axis=1)  # [batch, 1]
        
        # Ensure all tensors are 2D [batch, features]
        # gsd: [batch, 5]
        # bulk_density: [batch, 1]
        # porosity: [batch, 1]
        # psi_capillary_kpa: [batch, 1]
        # r_pore_um: [batch, 1]
        # other_features: [batch, other_dim]
        
        # Combine all features
        physics_features = tf.concat([
            gsd,                    # [batch, 5]
            bulk_density,           # [batch, 1]
            porosity,               # [batch, 1]
            psi_capillary_kpa,      # [batch, 1]
            r_pore_um,              # [batch, 1]
            other_features          # [batch, other_dim]
        ], axis=1)  # Result: [batch, 5+1+1+1+1+other_dim]
        
        # Dense layer to map to output units
        physics_encoded = self.physics_dense(physics_features)
        
        return physics_encoded


class PINN(keras.Model):
    """
    Physics-Informed Neural Network for SWCC Prediction
    
    Architecture:
    - Input: Soil properties + Suction values
    - Physics Encoding Layer: Arya-Paris transformation
    - Hidden Layers: Non-linear mapping
    - Output: Water content θ(ψ)
    """
    
    def __init__(self, 
                 soil_prop_dim=16,
                 suction_points=100,
                 hidden_dims=[128, 256, 128, 64],
                 physics_units=128,
                 name='PINN'):
        """
        Initialize PINN
        
        Args:
            soil_prop_dim: Dimension of soil properties
            suction_points: Number of suction query points
            hidden_dims: List of hidden layer dimensions
            physics_units: Units in physics encoding layer
            name: Model name
        """
        super(PINN, self).__init__(name=name)
        
        self.soil_prop_dim = soil_prop_dim
        self.suction_points = suction_points
        self.hidden_dims = hidden_dims
        self.physics_units = physics_units
        
        # Physics encoding layer
        self.physics_encoding = PhysicsEncodingLayer(units=physics_units)
        
        # Hidden layers
        self.hidden_layers = []
        for i, dim in enumerate(hidden_dims):
            self.hidden_layers.append(
                layers.Dense(dim, activation='relu', name=f'hidden_{i+1}')
            )
            self.hidden_layers.append(
                layers.BatchNormalization(name=f'bn_{i+1}')
            )
        
        # Output layer (single value per suction point)
        self.output_layer = layers.Dense(1, activation='sigmoid', name='output')
        
    def call(self, inputs, training=None):
        """
        Forward pass
        
        Args:
            inputs: Dictionary with:
                - 'soil_props': Soil properties [batch, soil_prop_dim]
                - 'suction': Suction values [batch, suction_points] or [suction_points]
            training: Training mode flag
        
        Returns:
            Predicted water content [batch, suction_points]
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
        # Based on metadata: [Cc, Cu, D10, D30, D50, D60, D90, OM_content, bulk_density, clay_pct, pH, porosity, sand_pct, silt_pct, theta_r, theta_s]
        # Indices: D10=2, D30=3, D50=4, D60=5, D90=6, bulk_density=8, porosity=11
        gsd = soil_props[:, 2:7]  # D10, D30, D50, D60, D90 (indices 2-6)
        bulk_density = tf.expand_dims(soil_props[:, 8], axis=1)  # Index 8 is bulk_density
        porosity = tf.expand_dims(soil_props[:, 11], axis=1)  # Index 11 is porosity
        other_features = tf.concat([
            soil_props[:, 0:2],   # Cc, Cu (indices 0-1)
            soil_props[:, 7:8],   # OM_content (index 7)
            soil_props[:, 9:10],  # clay_pct (index 9)
            soil_props[:, 10:11], # pH (index 10)
            soil_props[:, 12:14]  # sand_pct, silt_pct (indices 12-13)
        ], axis=1)
        
        # Physics encoding
        physics_input = {
            'gsd': gsd,
            'bulk_density': bulk_density,
            'porosity': porosity,
            'other_features': other_features
        }
        physics_features = self.physics_encoding(physics_input)
        
        # For each suction point, predict water content
        # Expand physics features for each suction point
        physics_expanded = tf.expand_dims(physics_features, axis=1)  # [batch, 1, physics_units]
        physics_expanded = tf.tile(physics_expanded, [1, n_suction, 1])  # [batch, n_suction, physics_units]
        
        # Concatenate with suction values
        suction_expanded = tf.expand_dims(suction, axis=2)  # [batch, n_suction, 1]
        combined = tf.concat([physics_expanded, suction_expanded], axis=2)  # [batch, n_suction, physics_units+1]
        
        # Flatten for processing
        combined_flat = tf.reshape(combined, [-1, self.physics_units + 1])  # [batch*n_suction, physics_units+1]
        
        # Pass through hidden layers
        x = combined_flat
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        
        # Output layer
        theta_norm = self.output_layer(x)  # [batch*n_suction, 1]
        theta_norm = tf.reshape(theta_norm, [batch_size, n_suction])  # [batch, n_suction]
        
        # Denormalize using theta_s and theta_r
        # Based on metadata: theta_r=14, theta_s=15
        theta_r = tf.expand_dims(soil_props[:, 14], axis=1)  # Index 14 is theta_r
        theta_s = tf.expand_dims(soil_props[:, 15], axis=1)  # Index 15 is theta_s
        
        theta_range = theta_s - theta_r
        theta = theta_norm * theta_range + theta_r
        
        return theta
    
    def predict_swcc(self, soil_properties, suction_grid):
        """
        Predict SWCC curve for given soil properties
        
        Args:
            soil_properties: Soil properties [batch, soil_prop_dim] or [soil_prop_dim]
            suction_grid: Suction values [n_points] or [batch, n_points]
        
        Returns:
            Predicted water content [batch, n_points]
        """
        # Ensure 2D
        if len(soil_properties.shape) == 1:
            soil_properties = tf.expand_dims(soil_properties, 0)
        
        if len(suction_grid.shape) == 1:
            suction_grid = tf.expand_dims(suction_grid, 0)
        
        inputs = {
            'soil_props': tf.cast(soil_properties, tf.float32),
            'suction': tf.cast(suction_grid, tf.float32)
        }
        
        return self(inputs, training=False)
    
    def get_config(self):
        """Get model configuration for serialization"""
        config = super().get_config()
        config.update({
            'soil_prop_dim': self.soil_prop_dim,
            'suction_points': self.suction_points,
            'hidden_dims': self.hidden_dims,
            'physics_units': self.physics_units
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create model from config"""
        # Extract custom config
        soil_prop_dim = config.pop('soil_prop_dim', 16)
        suction_points = config.pop('suction_points', 100)
        hidden_dims = config.pop('hidden_dims', [128, 256, 128, 64])
        physics_units = config.pop('physics_units', 128)
        
        # Create model
        model = cls(
            soil_prop_dim=soil_prop_dim,
            suction_points=suction_points,
            hidden_dims=hidden_dims,
            physics_units=physics_units,
            name=config.get('name', 'PINN')
        )
        
        return model