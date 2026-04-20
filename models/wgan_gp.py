"""
Wasserstein GAN with Gradient Penalty (WGAN-GP)
Complete WGAN-GP implementation with physics constraints
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

from .generator import Generator
from .discriminator import Discriminator
from .physics_constraints import PhysicsConstraints


class WGAN_GP(keras.Model):
    """
    Conditional Wasserstein GAN with Gradient Penalty
    
    Combines Generator, Discriminator, and Physics Constraints
    """
    
    def __init__(self, noise_dim=100, soil_prop_dim=16, swcc_points=100,
                 lambda_gp=10.0, lambda_mono=0.5, lambda_bound=0.3,
                 generator_hidden=[256, 512, 256],
                 discriminator_hidden=[256, 128, 64],
                 name='WGAN_GP'):
        """
        Initialize WGAN-GP
        
        Args:
            noise_dim: Dimension of noise vector
            soil_prop_dim: Dimension of soil properties
            swcc_points: Number of points in SWCC curve
            lambda_gp: Gradient penalty weight
            lambda_mono: Monotonicity loss weight
            lambda_bound: Boundary loss weight
            generator_hidden: Generator hidden layer dimensions
            discriminator_hidden: Discriminator hidden layer dimensions
            name: Model name
        """
        super(WGAN_GP, self).__init__(name=name)
        
        # Models
        self.generator = Generator(
            noise_dim=noise_dim,
            soil_prop_dim=soil_prop_dim,
            swcc_points=swcc_points,
            hidden_dims=generator_hidden
        )
        
        self.discriminator = Discriminator(
            swcc_points=swcc_points,
            soil_prop_dim=soil_prop_dim,
            hidden_dims=discriminator_hidden
        )
        
        # Physics constraints
        self.physics = PhysicsConstraints(
            lambda_mono=lambda_mono,
            lambda_bound=lambda_bound
        )
        
        # Hyperparameters
        self.noise_dim = noise_dim
        self.soil_prop_dim = soil_prop_dim
        self.swcc_points = swcc_points
        self.lambda_gp = lambda_gp
        self.lambda_mono = lambda_mono
        self.lambda_bound = lambda_bound
    
    def gradient_penalty(self, real_swcc, fake_swcc, soil_props):
        """
        Compute gradient penalty for WGAN-GP
        
        Args:
            real_swcc: Real SWCC curves [batch_size, swcc_points]
            fake_swcc: Fake SWCC curves [batch_size, swcc_points]
            soil_props: Soil properties [batch_size, soil_prop_dim]
            
        Returns:
            Gradient penalty loss
        """
        batch_size = tf.shape(real_swcc)[0]
        
        # Random interpolation coefficient
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        
        # Interpolate between real and fake
        interpolated = alpha * real_swcc + (1 - alpha) * fake_swcc
        
        # Compute gradient
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            d_interpolated = self.discriminator([interpolated, soil_props])
        
        gradients = tape.gradient(d_interpolated, interpolated)
        
        # Compute gradient norm over all non-batch axes
        axes = tf.range(1, tf.rank(gradients))
        gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=axes) + 1e-12)
        
        # Gradient penalty: (||gradient|| - 1)^2
        penalty = tf.reduce_mean(tf.square(gradient_norm - 1.0))
        
        return penalty
    
    def discriminator_loss(self, real_swcc, fake_swcc, soil_props):
        """
        Compute discriminator (critic) loss
        
        Args:
            real_swcc: Real SWCC curves
            fake_swcc: Fake SWCC curves
            soil_props: Soil properties
            
        Returns:
            Discriminator loss and gradient penalty
        """
        # Real and fake scores
        real_score = self.discriminator([real_swcc, soil_props])
        fake_score = self.discriminator([fake_swcc, soil_props])
        
        # Wasserstein distance: E[D(real)] - E[D(fake)]
        wasserstein_distance = tf.reduce_mean(real_score) - tf.reduce_mean(fake_score)
        
        # Gradient penalty
        gp = self.gradient_penalty(real_swcc, fake_swcc, soil_props)
        
        # Total loss: maximize Wasserstein distance, minimize gradient penalty
        d_loss = -wasserstein_distance + self.lambda_gp * gp
        
        return d_loss, wasserstein_distance, gp
    
    def generator_loss(self, fake_swcc, soil_props, theta_s, theta_r):
        """
        Compute generator loss with physics constraints
        
        Args:
            fake_swcc: Generated SWCC curves
            soil_props: Soil properties
            theta_s: Saturated water content
            theta_r: Residual water content
            
        Returns:
            Generator loss and component losses
        """
        # Adversarial loss: maximize discriminator score on fake data
        fake_score = self.discriminator([fake_swcc, soil_props])
        g_adversarial = -tf.reduce_mean(fake_score)
        
        # Physics losses
        physics_loss, mono_loss, bound_loss = self.physics.compute_physics_loss(
            fake_swcc, theta_s, theta_r
        )
        
        # Total generator loss
        g_loss = g_adversarial + physics_loss
        
        return g_loss, g_adversarial, mono_loss, bound_loss
    
    def generate_samples(self, soil_properties, num_samples=1, theta_s=None, theta_r=None, seed=None):
        """
        Generate SWCC curves for given soil properties
        
        Args:
            soil_properties: Soil properties [batch_size, soil_prop_dim]
            num_samples: Number of samples per soil
            theta_s: Saturated water content (for denormalization)
            theta_r: Residual water content (for denormalization)
            seed: Random seed
            
        Returns:
            Generated SWCC curves [batch_size * num_samples, swcc_points]
        """
        if theta_s is None or theta_r is None:
            raise ValueError(
                "generate_samples requires theta_s and theta_r for monotone "
                "bounded SWCC generation."
            )

        curves = self.generator.generate(
            soil_properties, theta_s, theta_r, num_samples=num_samples, seed=seed
        )

        return curves
    
    def get_config(self):
        """Get model configuration"""
        return {
            'noise_dim': self.noise_dim,
            'soil_prop_dim': self.soil_prop_dim,
            'swcc_points': self.swcc_points,
            'lambda_gp': self.lambda_gp,
            'lambda_mono': self.lambda_mono,
            'lambda_bound': self.lambda_bound
        }
