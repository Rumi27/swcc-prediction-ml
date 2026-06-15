"""
Training Utilities for GAN
Data loading, monitoring, visualization
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt/X11 issues
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime


class DataLoader:
    """Data loader for GAN training"""
    
    def __init__(self, config):
        """
        Initialize data loader
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.feature_cols = config.get('feature_cols', [])
        
    def load_data(self):
        """Load training data"""
        print("Loading training data...")
        
        # Load features
        X_train = pd.read_csv(self.config['train_file'])
        
        # Load SWCC curves
        y_train = np.load(self.config['swcc_file'])
        
        # Load suction grid
        suction_grid = np.load(self.config['suction_grid_file'])
        
        # Extract features
        feature_cols = [col for col in self.feature_cols if col in X_train.columns]
        X_features_df = X_train[feature_cols]

        # Handle missing values in features (impute with column means)
        if X_features_df.isna().values.any():
            col_means = X_features_df.mean()
            X_features_df = X_features_df.fillna(col_means)
        X_features = X_features_df.values
        
        # Extract theta_s and theta_r for denormalization
        theta_s = X_train['theta_s'].values if 'theta_s' in X_train.columns else None
        theta_r = X_train['theta_r'].values if 'theta_r' in X_train.columns else None
        
        print(f"✓ Loaded {len(X_features)} samples")
        print(f"✓ Features: {len(feature_cols)}")
        print(f"✓ SWCC points: {y_train.shape[1]}")
        
        return {
            'features': X_features,
            'swcc_curves': y_train,
            'theta_s': theta_s,
            'theta_r': theta_r,
            'suction_grid': suction_grid,
            'feature_cols': feature_cols
        }
    
    def create_dataset(self, features, swcc_curves, theta_s, theta_r, batch_size=32, shuffle=True):
        """
        Create TensorFlow dataset
        
        Args:
            features: Soil properties [N, feature_dim]
            swcc_curves: SWCC curves [N, swcc_points]
            theta_s: Saturated water content [N]
            theta_r: Residual water content [N]
            batch_size: Batch size
            shuffle: Whether to shuffle
            
        Returns:
            TensorFlow dataset
        """
        # Normalize SWCC curves to [-1, 1]
        # θ_norm = 2 * (θ - θr) / (θs - θr) - 1
        theta_s_2d = theta_s.reshape(-1, 1) if len(theta_s.shape) == 1 else theta_s
        theta_r_2d = theta_r.reshape(-1, 1) if len(theta_r.shape) == 1 else theta_r
        theta_range = theta_s_2d - theta_r_2d
        
        # Avoid division by zero - use larger epsilon and ensure minimum range
        theta_range = np.maximum(theta_range, 1e-4)  # Minimum range of 0.0001
        
        swcc_normalized = 2.0 * (swcc_curves - theta_r_2d) / theta_range - 1.0
        
        # Clip to [-1, 1]
        swcc_normalized = np.clip(swcc_normalized, -1.0, 1.0)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices({
            'soil_props': features.astype(np.float32),
            'swcc_curve': swcc_normalized.astype(np.float32),
            'theta_s': theta_s.astype(np.float32),
            'theta_r': theta_r.astype(np.float32)
        })
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(1000, len(features)))
        
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        # Note: Dataset will be repeated in training loop for multiple epochs
        
        return dataset


class TrainingMonitor:
    """Monitor and log training progress"""
    
    def __init__(self, output_dir, config):
        """
        Initialize training monitor
        
        Args:
            output_dir: Output directory for logs and plots
            config: Configuration dictionary
        """
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        
        self.config = config
        self.history = {
            'epoch': [],
            'd_loss': [],
            'g_loss': [],
            'wasserstein_dist': [],
            'gradient_penalty': [],
            'g_adversarial': [],
            'mono_loss': [],
            'bound_loss': []
        }
        
    def log_metrics(self, epoch, metrics):
        """Log training metrics"""
        self.history['epoch'].append(epoch)
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(float(value))
    
    def save_history(self):
        """Save training history"""
        history_file = self.output_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Saved training history: {history_file}")
    
    def plot_training_curves(self):
        """Plot training curves"""
        if len(self.history['epoch']) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Discriminator and Generator losses
        ax = axes[0, 0]
        ax.plot(self.history['epoch'], self.history['d_loss'], label='Discriminator', linewidth=2)
        ax.plot(self.history['epoch'], self.history['g_loss'], label='Generator', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Wasserstein distance
        ax = axes[0, 1]
        ax.plot(self.history['epoch'], self.history['wasserstein_dist'], 'g-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Wasserstein Distance')
        ax.set_title('Wasserstein Distance (should decrease)')
        ax.grid(True, alpha=0.3)
        
        # Gradient penalty
        ax = axes[1, 0]
        ax.plot(self.history['epoch'], self.history['gradient_penalty'], 'r-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Penalty')
        ax.set_title('Gradient Penalty')
        ax.grid(True, alpha=0.3)
        
        # Physics losses
        ax = axes[1, 1]
        ax.plot(self.history['epoch'], self.history['mono_loss'], label='Monotonicity', linewidth=2)
        ax.plot(self.history['epoch'], self.history['bound_loss'], label='Boundary', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Physics Loss')
        ax.set_title('Physics Constraint Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved training curves: {self.viz_dir / 'training_curves.png'}")
    
    def plot_sample_generations(self, model, sample_soil_props, sample_swcc, 
                                sample_theta_s, sample_theta_r, suction_grid, epoch):
        """Plot sample generated curves"""
        # Generate curves
        generated = model.generate_samples(
            sample_soil_props.astype(np.float32),
            num_samples=5,
            theta_s=sample_theta_s.astype(np.float32),
            theta_r=sample_theta_r.astype(np.float32)
        )
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i in range(min(4, len(sample_soil_props))):
            ax = axes[i]
            
            # Real curve
            real_curve = sample_swcc[i]
            ax.semilogx(suction_grid, real_curve, 'b-', linewidth=2, label='Real', alpha=0.7)
            
            # Generated curves
            for j in range(5):
                gen_curve = generated[i*5 + j].numpy()
                ax.semilogx(suction_grid, gen_curve, 'r--', linewidth=1, alpha=0.5)
            
            ax.set_xlabel('Suction (kPa)')
            ax.set_ylabel('Water Content (θ)')
            ax.set_title(f'Sample {i+1} - Epoch {epoch}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'samples_epoch_{epoch:04d}.png', dpi=300, bbox_inches='tight')
        plt.close()
