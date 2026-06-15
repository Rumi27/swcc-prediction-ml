#!/usr/bin/env python3
"""
Baseline Models for SWCC Prediction
Implements traditional ML models to establish performance benchmarks
Models: Random Forest, Gradient Boosting, SVM, Neural Network
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class BaselineModels:
    """Baseline ML models for SWCC prediction"""
    
    def __init__(self, data_dir="data_processed", output_dir="results_baseline"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.results = {}
        self.scalers = {}
        self.imputers = {}
        
    def load_data(self):
        """Load processed data"""
        print("="*60)
        print("Loading Processed Data")
        print("="*60)
        
        # Load feature matrices
        X_train = pd.read_csv(self.data_dir / 'X_train.csv')
        X_val = pd.read_csv(self.data_dir / 'X_val.csv')
        X_test = pd.read_csv(self.data_dir / 'X_test.csv')
        
        # Load SWCC matrices
        y_train = np.load(self.data_dir / 'y_train.npy')
        y_val = np.load(self.data_dir / 'y_val.npy')
        y_test = np.load(self.data_dir / 'y_test.npy')
        
        # Load suction grid
        suction_grid = np.load(self.data_dir / 'suction_grid.npy')
        
        print(f"✓ Training set: {X_train.shape[0]} samples, {X_train.shape[1]-1} features")
        print(f"✓ Validation set: {X_val.shape[0]} samples")
        print(f"✓ Test set: {X_test.shape[0]} samples")
        print(f"✓ SWCC curves: {y_train.shape[1]} points per curve")
        
        return (X_train, X_val, X_test), (y_train, y_val, y_test), suction_grid
    
    def prepare_features(self, X_train, X_val, X_test):
        """Prepare features: handle missing values and scale"""
        print("\n" + "="*60)
        print("Preparing Features")
        print("="*60)
        
        # Separate code column
        codes_train = X_train['code'].values
        codes_val = X_val['code'].values
        codes_test = X_test['code'].values
        
        # Get feature columns (exclude 'code')
        feature_cols = [col for col in X_train.columns if col != 'code']
        
        # Extract features
        X_train_feat = X_train[feature_cols].values
        X_val_feat = X_val[feature_cols].values
        X_test_feat = X_test[feature_cols].values
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_train_feat = imputer.fit_transform(X_train_feat)
        X_val_feat = imputer.transform(X_val_feat)
        X_test_feat = imputer.transform(X_test_feat)
        self.imputers['main'] = imputer
        
        # Scale features
        scaler = StandardScaler()
        X_train_feat = scaler.fit_transform(X_train_feat)
        X_val_feat = scaler.transform(X_val_feat)
        X_test_feat = scaler.transform(X_test_feat)
        self.scalers['main'] = scaler
        
        print(f"✓ Features prepared: {X_train_feat.shape[1]} features")
        print(f"  Missing values imputed: {np.isnan(X_train_feat).sum()} remaining")
        
        return X_train_feat, X_val_feat, X_test_feat, feature_cols
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train Random Forest model"""
        print("\n" + "="*60)
        print("Training Random Forest")
        print("="*60)
        
        # Random Forest for each SWCC point
        n_points = y_train.shape[1]
        models = []
        
        for i in range(n_points):
            if (i + 1) % 20 == 0:
                print(f"  Training point {i+1}/{n_points}...")
            
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train[:, i])
            models.append(rf)
        
        print("✓ Random Forest training complete")
        return models
    
    def train_gradient_boosting(self, X_train, y_train, X_val, y_val):
        """Train Gradient Boosting model (best performer in literature)"""
        print("\n" + "="*60)
        print("Training Gradient Boosting")
        print("="*60)
        
        # Gradient Boosting for each SWCC point
        n_points = y_train.shape[1]
        models = []
        
        for i in range(n_points):
            if (i + 1) % 20 == 0:
                print(f"  Training point {i+1}/{n_points}...")
            
            gb = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            gb.fit(X_train, y_train[:, i])
            models.append(gb)
        
        print("✓ Gradient Boosting training complete")
        return models
    
    def train_svm(self, X_train, y_train, X_val, y_val):
        """Train Support Vector Machine model"""
        print("\n" + "="*60)
        print("Training Support Vector Machine")
        print("="*60)
        
        # SVM for each SWCC point (sample subset for speed)
        n_points = y_train.shape[1]
        models = []
        
        # Use subset for SVM (it's slower)
        sample_size = min(200, len(X_train))
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_train_subset = X_train[indices]
        y_train_subset = y_train[indices]
        
        for i in range(0, n_points, 5):  # Train every 5th point for speed
            if (i + 1) % 20 == 0:
                print(f"  Training point {i+1}/{n_points}...")
            
            svm = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.01)
            svm.fit(X_train_subset, y_train_subset[:, i])
            models.append((i, svm))
        
        print("✓ SVM training complete (trained on subset of points)")
        return models
    
    def train_neural_network(self, X_train, y_train, X_val, y_val):
        """Train Neural Network model"""
        print("\n" + "="*60)
        print("Training Neural Network")
        print("="*60)
        
        # Simple MLP for each SWCC point
        n_points = y_train.shape[1]
        models = []
        
        for i in range(n_points):
            if (i + 1) % 20 == 0:
                print(f"  Training point {i+1}/{n_points}...")
            
            nn = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            nn.fit(X_train, y_train[:, i])
            models.append(nn)
        
        print("✓ Neural Network training complete")
        return models
    
    def predict_swcc(self, models, X, model_type='rf', n_points=100):
        """Predict SWCC curves using trained models"""
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, n_points))
        
        if model_type == 'svm':
            # SVM has sparse predictions (trained on subset of points)
            trained_indices = []
            for idx, model in models:
                if idx < n_points:
                    predictions[:, idx] = model.predict(X)
                    trained_indices.append(idx)
            
            # Interpolate missing points
            for i in range(n_points):
                if i not in trained_indices:
                    # Find nearest trained point
                    if trained_indices:
                        nearest_idx = min(trained_indices, key=lambda x: abs(x - i))
                        predictions[:, i] = predictions[:, nearest_idx]
        else:
            for i, model in enumerate(models):
                if i < n_points:
                    predictions[:, i] = model.predict(X)
        
        return predictions
    
    def evaluate_model(self, y_true, y_pred, model_name, dataset_name):
        """Evaluate model performance"""
        # Flatten for overall metrics
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        r2 = r2_score(y_true_flat, y_pred_flat)
        
        # Per-curve metrics
        curve_rmse = np.sqrt(np.mean((y_true - y_pred)**2, axis=1))
        curve_mae = np.mean(np.abs(y_true - y_pred), axis=1)
        
        results = {
            'model': model_name,
            'dataset': dataset_name,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'curve_rmse_mean': np.mean(curve_rmse),
            'curve_rmse_std': np.std(curve_rmse),
            'curve_mae_mean': np.mean(curve_mae),
            'curve_mae_std': np.std(curve_mae)
        }
        
        return results
    
    def train_all_models(self, X_train, y_train, X_val, y_val):
        """Train all baseline models"""
        print("\n" + "="*80)
        print("Training All Baseline Models")
        print("="*80)
        
        # Train models
        self.models['random_forest'] = self.train_random_forest(X_train, y_train, X_val, y_val)
        self.models['gradient_boosting'] = self.train_gradient_boosting(X_train, y_train, X_val, y_val)
        self.models['svm'] = self.train_svm(X_train, y_train, X_val, y_val)
        self.models['neural_network'] = self.train_neural_network(X_train, y_train, X_val, y_val)
        
        print("\n✓ All models trained successfully!")
        return self.models
    
    def evaluate_all_models(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Evaluate all models on all datasets"""
        print("\n" + "="*80)
        print("Evaluating All Models")
        print("="*80)
        
        all_results = []
        
        for model_name, models in self.models.items():
            print(f"\nEvaluating {model_name}...")
            
            # Predictions
            n_points = y_train.shape[1]
            y_train_pred = self.predict_swcc(models, X_train, model_type=model_name, n_points=n_points)
            y_val_pred = self.predict_swcc(models, X_val, model_type=model_name, n_points=n_points)
            y_test_pred = self.predict_swcc(models, X_test, model_type=model_name, n_points=n_points)
            
            # Evaluate
            results_train = self.evaluate_model(y_train, y_train_pred, model_name, 'train')
            results_val = self.evaluate_model(y_val, y_val_pred, model_name, 'val')
            results_test = self.evaluate_model(y_test, y_test_pred, model_name, 'test')
            
            all_results.extend([results_train, results_val, results_test])
            
            print(f"  Test RMSE: {results_test['rmse']:.4f}")
            print(f"  Test MAE:  {results_test['mae']:.4f}")
            print(f"  Test R²:   {results_test['r2']:.4f}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(self.output_dir / 'baseline_results.csv', index=False)
        
        print(f"\n✓ Results saved to: {self.output_dir / 'baseline_results.csv'}")
        
        return results_df
    
    def visualize_results(self, results_df, X_test, y_test, suction_grid):
        """Visualize model performance"""
        print("\n" + "="*60)
        print("Generating Visualizations")
        print("="*60)
        
        # 1. Performance comparison
        print("  1. Performance comparison...")
        test_results = results_df[results_df['dataset'] == 'test']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        metrics = ['rmse', 'mae', 'r2']
        metric_names = ['RMSE', 'MAE', 'R²']
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx]
            test_results.plot(x='model', y=metric, kind='bar', ax=ax, legend=False)
            ax.set_title(f'{name} Comparison', fontsize=12, fontweight='bold')
            ax.set_xlabel('Model')
            ax.set_ylabel(name)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'baseline_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("     ✓ Saved: baseline_performance.png")
        
        # 2. Sample predictions
        print("  2. Sample predictions...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # Select diverse test samples
        sample_indices = np.linspace(0, len(X_test)-1, 4, dtype=int)
        
        for model_name in ['random_forest', 'gradient_boosting']:
            if model_name in self.models:
                models = self.models[model_name]
                for idx, ax_idx in enumerate([0, 1] if model_name == 'random_forest' else [2, 3]):
                    sample_idx = sample_indices[idx]
                    
                    y_pred = self.predict_swcc(models, X_test[sample_idx:sample_idx+1], 
                                              model_type=model_name, n_points=len(suction_grid))
                    
                    ax = axes[ax_idx]
                    ax.semilogx(suction_grid, y_test[sample_idx], 'b-', linewidth=2, label='Observed')
                    ax.semilogx(suction_grid, y_pred[0], 'r--', linewidth=2, label='Predicted')
                    ax.set_xlabel('Suction (kPa)', fontsize=11)
                    ax.set_ylabel('Water Content (θ)', fontsize=11)
                    ax.set_title(f'{model_name.replace("_", " ").title()}\nSample {sample_idx}', fontsize=11)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'baseline_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("     ✓ Saved: baseline_predictions.png")
        
        # 3. Error distribution
        print("  3. Error distribution...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, model_name in enumerate(['random_forest', 'gradient_boosting', 'neural_network', 'svm']):
            if model_name in self.models:
                models = self.models[model_name]
                y_pred = self.predict_swcc(models, X_test, model_type=model_name, n_points=y_test.shape[1])
                errors = (y_test - y_pred).flatten()
                
                ax = axes[idx]
                ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
                ax.set_xlabel('Prediction Error (θ)', fontsize=11)
                ax.set_ylabel('Frequency', fontsize=11)
                ax.set_title(f'{model_name.replace("_", " ").title()}\nMean: {np.mean(errors):.4f}, Std: {np.std(errors):.4f}', fontsize=11)
                ax.axvline(0, color='r', linestyle='--', linewidth=2)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'baseline_errors.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("     ✓ Saved: baseline_errors.png")
        
        print(f"\n✓ All visualizations saved to: {self.viz_dir}")
    
    def generate_report(self, results_df):
        """Generate summary report"""
        print("\n" + "="*60)
        print("Generating Summary Report")
        print("="*60)
        
        # Test set results
        test_results = results_df[results_df['dataset'] == 'test'].copy()
        test_results = test_results.sort_values('rmse')
        
        print("\n" + "="*60)
        print("BASELINE MODEL PERFORMANCE (Test Set)")
        print("="*60)
        print(f"\n{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
        print("-" * 60)
        
        for _, row in test_results.iterrows():
            print(f"{row['model']:<20} {row['rmse']:<10.4f} {row['mae']:<10.4f} {row['r2']:<10.4f}")
        
        # Best model
        best_model = test_results.iloc[0]
        print("\n" + "="*60)
        print("BEST MODEL:")
        print("="*60)
        print(f"Model: {best_model['model']}")
        print(f"RMSE:  {best_model['rmse']:.4f}")
        print(f"MAE:   {best_model['mae']:.4f}")
        print(f"R²:    {best_model['r2']:.4f}")
        
        # Comparison with literature
        print("\n" + "="*60)
        print("COMPARISON WITH LITERATURE:")
        print("="*60)
        print("Bakhshi et al. (2023) - Gradient Boosting: RMSE = 0.016")
        print(f"Our Gradient Boosting: RMSE = {test_results[test_results['model']=='gradient_boosting']['rmse'].values[0]:.4f}")
        
        # Save report
        report = {
            'best_model': best_model['model'],
            'best_rmse': float(best_model['rmse']),
            'best_mae': float(best_model['mae']),
            'best_r2': float(best_model['r2']),
            'all_results': test_results.to_dict('records')
        }
        
        with open(self.output_dir / 'baseline_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Report saved to: {self.output_dir / 'baseline_report.json'}")
        
        return report
    
    def run_full_pipeline(self):
        """Run complete baseline model pipeline"""
        print("\n" + "="*80)
        print("BASELINE MODELS PIPELINE")
        print("="*80)
        
        # Load data
        (X_train, X_val, X_test), (y_train, y_val, y_test), suction_grid = self.load_data()
        
        # Prepare features
        X_train_feat, X_val_feat, X_test_feat, feature_cols = self.prepare_features(
            X_train, X_val, X_test
        )
        
        # Train all models
        self.train_all_models(X_train_feat, y_train, X_val_feat, y_val)
        
        # Evaluate all models
        results_df = self.evaluate_all_models(
            X_train_feat, y_train,
            X_val_feat, y_val,
            X_test_feat, y_test
        )
        
        # Visualize results
        self.visualize_results(results_df, X_test_feat, y_test, suction_grid)
        
        # Generate report
        report = self.generate_report(results_df)
        
        print("\n" + "="*80)
        print("BASELINE MODELS COMPLETE!")
        print("="*80)
        print(f"\nBest Model: {report['best_model']}")
        print(f"Best RMSE: {report['best_rmse']:.4f}")
        print(f"\nReady to proceed to Phase 2: GAN Development!")
        
        return results_df, report

if __name__ == "__main__":
    baseline = BaselineModels()
    results, report = baseline.run_full_pipeline()
