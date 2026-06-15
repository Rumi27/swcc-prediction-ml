"""
Discriminator (Critic) for WGAN-GP
Distinguishes real from fake SWCC curves
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Discriminator(keras.Model):
    """
    Discriminator (Critic) for WGAN-GP
    
    Input: [SWCC_curve (100) + soil_properties (16)] = 116 dim
    Output: Real/Fake score (scalar, no activation)
    """
    
    def __init__(self, swcc_points=100, soil_prop_dim=16, 
                 hidden_dims=[256, 128, 64], dropout_rate=0.3, name='Discriminator'):
        """
        Initialize Discriminator
        
        Args:
            swcc_points: Number of points in SWCC curve
            soil_prop_dim: Dimension of soil properties
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate
            name: Model name
        """
        super(Discriminator, self).__init__(name=name)
        
        self.swcc_points = swcc_points
        self.soil_prop_dim = soil_prop_dim
        self.input_dim = swcc_points + soil_prop_dim
        self.dropout_rate = dropout_rate
        
        # Build layers
        self.layers_list = []
        
        # First layer
        self.layers_list.append(
            layers.Dense(hidden_dims[0], name='dense_1')
        )
        self.layers_list.append(layers.LeakyReLU(alpha=0.2, name='leaky_relu_1'))
        self.layers_list.append(layers.Dropout(dropout_rate, name='dropout_1'))
        
        # Hidden layers
        for i, dim in enumerate(hidden_dims[1:], start=2):
            self.layers_list.append(
                layers.Dense(dim, name=f'dense_{i}')
            )
            self.layers_list.append(layers.LeakyReLU(alpha=0.2, name=f'leaky_relu_{i}'))
            if i < len(hidden_dims):  # No dropout on last hidden layer
                self.layers_list.append(layers.Dropout(dropout_rate, name=f'dropout_{i}'))
        
        # Output layer (no activation for WGAN)
        self.layers_list.append(
            layers.Dense(1, name='output')
        )
    
    def call(self, inputs, training=None):
        """
        Forward pass
        
        Args:
            inputs: Tuple of (swcc_curve, soil_properties) or concatenated input
            training: Training mode flag
            
        Returns:
            Real/Fake score [batch_size, 1]
        """
        # Handle input format
        if isinstance(inputs, (list, tuple)):
            swcc_curve, soil_props = inputs
            x = tf.concat([swcc_curve, soil_props], axis=1)
        else:
            x = inputs
        
        # Forward through layers
        for layer in self.layers_list:
            x = layer(x, training=training)
        
        return x
