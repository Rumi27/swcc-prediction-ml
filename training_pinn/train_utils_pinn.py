"""
Training Utilities for PINN
Data loading, monitoring, visualization
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime


class DataLoader:
    """Data loader for PINN training"""
    
    def __init__(self, config):
        self.config = config
    
    def load_train_data(self):
        """Load training data"""
        X = pd.read_csv(self.config['train_file'])
        y = np.load(self.config['y_train_file'])
        return {'X': X, 'y': y}
    
    def load_val_data(self):
        """Load validation data"""
        X = pd.read_csv(self.config['val_file'])
        y = np.load(self.config['y_val_file'])
        return {'X': X, 'y': y}
    
    def load_test_data(self):
        """Load test data"""
        X = pd.read_csv(self.config['test_file'])
        y = np.load(self.config['y_test_file'])
        return {'X': X, 'y': y}
    
    def load_metadata(self):
        """Load metadata"""
        with open(self.config['metadata_file'], 'r') as f:
            return json.load(f)


class TrainingMonitor:
    """Monitor and log training progress"""
    
    def __init__(self, output_dir, config):
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        
        self.config = config
        self.history = {
            'epoch': [],
            'train_total': [], 'train_data': [], 'train_mono': [], 'train_bound': [], 'train_physics': [],
            'val_total': [], 'val_data': [], 'val_mono': [], 'val_bound': [], 'val_physics': []
        }
    
    def log_metrics(self, epoch, train_metrics, val_metrics):
        """Log training metrics"""
        self.history['epoch'].append(epoch)
        
        # Map loss keys to history keys
        key_mapping = {
            'total': ('train_total', 'val_total'),
            'data': ('train_data', 'val_data'),
            'monotonicity': ('train_mono', 'val_mono'),
            'boundary': ('train_bound', 'val_bound'),
            'physics': ('train_physics', 'val_physics')
        }
        
        for key, (train_key, val_key) in key_mapping.items():
            self.history[train_key].append(float(train_metrics.get(key, np.nan)))
            self.history[val_key].append(float(val_metrics.get(key, np.nan)))
    
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
        
        # Total loss
        ax = axes[0, 0]
        ax.plot(self.history['epoch'], self.history['train_total'], 'b-', label='Train', linewidth=2)
        ax.plot(self.history['epoch'], self.history['val_total'], 'r--', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Data loss
        ax = axes[0, 1]
        ax.plot(self.history['epoch'], self.history['train_data'], 'b-', label='Train', linewidth=2)
        ax.plot(self.history['epoch'], self.history['val_data'], 'r--', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Data Loss')
        ax.set_title('Data Loss (MSE)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Physics losses
        ax = axes[1, 0]
        ax.plot(self.history['epoch'], self.history['train_mono'], 'g-', label='Monotonicity', linewidth=2)
        ax.plot(self.history['epoch'], self.history['train_bound'], 'm-', label='Boundary', linewidth=2)
        ax.plot(self.history['epoch'], self.history['train_physics'], 'c-', label='Physics', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Physics Loss')
        ax.set_title('Physics Constraint Losses (Train)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Validation physics losses
        ax = axes[1, 1]
        ax.plot(self.history['epoch'], self.history['val_mono'], 'g--', label='Monotonicity', linewidth=2)
        ax.plot(self.history['epoch'], self.history['val_bound'], 'm--', label='Boundary', linewidth=2)
        ax.plot(self.history['epoch'], self.history['val_physics'], 'c--', label='Physics', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Physics Loss')
        ax.set_title('Physics Constraint Losses (Val)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved training curves: {self.viz_dir / 'training_curves.png'}")
    
    def plot_predictions(self, model, val_dataset, epoch, n_samples=4):
        """Plot sample predictions"""
        # Get sample batch
        sample_batch = next(iter(val_dataset.take(1)))
        
        # Predict
        inputs = {
            'soil_props': sample_batch['soil_props'][:n_samples],
            'suction': sample_batch['suction'][:n_samples]
        }
        theta_pred = model(inputs, training=False)
        
        # Load suction grid
        from training_pinn.config_pinn import DATA_CONFIG
        suction_grid = np.load(DATA_CONFIG['suction_grid_file'])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i in range(min(n_samples, len(sample_batch['soil_props']))):
            ax = axes[i]
            
            # Observed
            ax.semilogx(suction_grid, sample_batch['theta_obs'][i].numpy(), 
                       'b-', linewidth=2, label='Observed', alpha=0.7)
            
            # Predicted
            ax.semilogx(suction_grid, theta_pred[i].numpy(), 
                       'r--', linewidth=2, label='Predicted', alpha=0.7)
            
            ax.set_xlabel('Suction (kPa)')
            ax.set_ylabel('Water Content (θ)')
            ax.set_title(f'Sample {i+1} - Epoch {epoch}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'predictions_epoch_{epoch:04d}.png', dpi=300, bbox_inches='tight')
        plt.close()
