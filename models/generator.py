"""
Generator for Conditional WGAN-GP
Generates SWCC curves conditioned on soil properties
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Generator(keras.Model):
    """
    Conditional Generator for SWCC curves.
    
    Input:  [noise (100) + soil_properties (16)]
    Output: SWCC curve θ(s) with:
             - structural monotonicity (non-increasing)
             - values bounded in [θ_r, θ_s]
    """
    
    def __init__(self, noise_dim=100, soil_prop_dim=16, swcc_points=100, 
                 hidden_dims=[256, 512, 256], name='Generator'):
        """
        Initialize Generator
        
        Args:
            noise_dim: Dimension of noise vector
            soil_prop_dim: Dimension of soil properties
            swcc_points: Number of points in SWCC curve
            hidden_dims: List of hidden layer dimensions
            name: Model name
        """
        super(Generator, self).__init__(name=name)
        
        self.noise_dim = noise_dim
        self.soil_prop_dim = soil_prop_dim
        self.swcc_points = swcc_points
        self.input_dim = noise_dim + soil_prop_dim
        
        # Build hidden layers
        self.hidden_layers = []
        
        # First layer
        self.hidden_layers.append(
            layers.Dense(hidden_dims[0], use_bias=False, name='dense_1')
        )
        self.hidden_layers.append(layers.BatchNormalization(name='bn_1'))
        self.hidden_layers.append(layers.ReLU(name='relu_1'))
        
        # Hidden layers
        for i, dim in enumerate(hidden_dims[1:], start=2):
            self.hidden_layers.append(
                layers.Dense(dim, use_bias=False, name=f'dense_{i}')
            )
            self.hidden_layers.append(layers.BatchNormalization(name=f'bn_{i}'))
            self.hidden_layers.append(layers.ReLU(name=f'relu_{i}'))
        
        # Output layer: raw increments (no activation)
        self.out_layer = layers.Dense(swcc_points, activation=None, name='output_logits')
    
    def call(self, inputs, training=None):
        """
        Forward pass
        
        Args:
            inputs: 
                - Tuple/List: (noise, soil_properties, theta_s, theta_r)
                  or (noise, soil_properties) for normalized output
                - Or concatenated input [noise, soil_properties]
            training: Training mode flag
            
        Returns:
            Generated SWCC curve [batch_size, swcc_points]
        """
        theta_s = None
        theta_r = None
        
        # Handle input format
        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 4:
                noise, soil_props, theta_s, theta_r = inputs
            elif len(inputs) == 3:
                noise, soil_props, theta_s = inputs
            else:
                noise, soil_props = inputs
            x = tf.concat([noise, soil_props], axis=1)
        else:
            x = inputs
        
        # Forward through hidden layers
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        
        # Raw increments (any real values)
        raw = self.out_layer(x)
        
        # theta_s/theta_r are required for physical SWCC output
        if theta_s is None or theta_r is None:
            raise ValueError(
                "Generator requires theta_s and theta_r to produce bounded "
                "monotone SWCC curves."
            )
        
        # Ensure proper shapes for theta_s, theta_r
        theta_s = tf.reshape(theta_s, [-1, 1])
        theta_r = tf.reshape(theta_r, [-1, 1])
        
        eps = 1e-6
        # Positive increments (clip logits to avoid overflow / NaNs)
        raw_clipped = tf.clip_by_value(raw, -20.0, 20.0)
        delta = tf.nn.softplus(raw_clipped) + eps              # [B, P] > 0
        delta_sum = tf.reduce_sum(delta, axis=1, keepdims=True)
        # Normalize to (theta_s - theta_r)
        theta_range = tf.maximum(theta_s - theta_r, eps)
        delta = delta / delta_sum * theta_range                # sums to theta_range
        
        # Cumulative sum → decreasing curve from theta_s toward theta_r
        cum = tf.cumsum(delta, axis=1)
        theta = theta_s - cum
        
        # Force last point exactly θ_r for numerical cleanliness
        theta_last = tf.concat([theta[:, :-1], theta_r], axis=1)
        
        return theta_last
    
    def generate(self, soil_properties, theta_s, theta_r, num_samples=1, seed=None):
        """
        Generate SWCC curves for given soil properties
        
        Args:
            soil_properties: Soil properties [batch_size, soil_prop_dim] or [soil_prop_dim]
            theta_s: Saturated water content [batch_size] or [batch_size, 1]
            theta_r: Residual water content [batch_size] or [batch_size, 1]
            num_samples: Number of samples to generate per soil
            seed: Random seed
            
        Returns:
            Generated SWCC curves [batch_size * num_samples, swcc_points]
        """
        # Ensure 2D for soil properties and theta bounds
        if len(soil_properties.shape) == 1:
            soil_properties = tf.expand_dims(soil_properties, 0)
        if len(theta_s.shape) == 1:
            theta_s = tf.expand_dims(theta_s, 0)
        if len(theta_r.shape) == 1:
            theta_r = tf.expand_dims(theta_r, 0)

        batch_size = soil_properties.shape[0]
        
        # Expand soil properties for multiple samples
        if num_samples > 1:
            soil_properties = tf.repeat(soil_properties, num_samples, axis=0)
            theta_s = tf.repeat(theta_s, num_samples, axis=0)
            theta_r = tf.repeat(theta_r, num_samples, axis=0)
        
        # Generate noise
        if seed is not None:
            tf.random.set_seed(seed)
        noise = tf.random.normal([batch_size * num_samples, self.noise_dim])
        
        # Generate physical SWCC curves
        swcc_curves = self([noise, soil_properties, theta_s, theta_r], training=False)
        
        return swcc_curves
    
    def denormalize_output(self, normalized_curve, theta_s, theta_r):
        """
        Denormalize generated curve from [-1, 1] to [θr, θs]
        
        Args:
            normalized_curve: Normalized curve [-1, 1] [batch_size, n_points]
            theta_s: Saturated water content [batch_size] or [batch_size, 1]
            theta_r: Residual water content [batch_size] or [batch_size, 1]
            
        Returns:
            Denormalized curve [θr, θs] [batch_size, n_points]
        """
        # Ensure theta_s and theta_r are 1D
        if len(theta_s.shape) > 1:
            theta_s = tf.squeeze(theta_s)
        if len(theta_r.shape) > 1:
            theta_r = tf.squeeze(theta_r)
        
        # Expand for broadcasting: [batch_size, 1]
        theta_s = tf.expand_dims(theta_s, axis=1)
        theta_r = tf.expand_dims(theta_r, axis=1)
        
        # Normalize from [-1, 1] to [0, 1]
        curve_01 = (normalized_curve + 1.0) / 2.0
        
        # Scale to [θr, θs]
        theta_range = theta_s - theta_r
        curve = curve_01 * theta_range + theta_r
        
        return curve
