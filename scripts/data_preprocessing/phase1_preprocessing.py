#!/usr/bin/env python3
"""
Phase 1: Data Preprocessing for SWCC Prediction
- Extract GSD percentiles
- Create feature matrix
- Process SWCC data
- Generate visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import interpolate
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class UNSODAPreprocessor:
    """Preprocess UNSODA data for ML training"""
    
    def __init__(self, data_dir="data_UNSODA_extracted", output_dir="data_processed"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create visualization directory
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        
        self.tables = {}
        self.features_df = None
        self.swcc_data = None
        
    def load_data(self):
        """Load all required tables"""
        print("="*60)
        print("Loading UNSODA Data")
        print("="*60)
        
        # Load key tables
        tables_to_load = {
            'swcc': 'lab_drying_h-t.csv',  # Primary SWCC data
            'particle_size': 'particle_size.csv',
            'soil_properties': 'soil_properties.csv',
            'general': 'general.csv'
        }
        
        for key, filename in tables_to_load.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                self.tables[key] = pd.read_csv(filepath)
                print(f"✓ Loaded {key}: {len(self.tables[key])} rows")
            else:
                print(f"✗ File not found: {filename}")
        
        print(f"\nTotal samples: {self.tables['swcc']['code'].nunique()}")
        return self.tables
    
    def extract_gsd_percentiles(self):
        """Extract GSD percentiles (D10, D30, D50, D60, D90) from particle size data"""
        print("\n" + "="*60)
        print("Extracting GSD Percentiles")
        print("="*60)
        
        particle_df = self.tables['particle_size'].copy()
        
        # Remove rows with missing values
        particle_df = particle_df.dropna(subset=['particle_size', 'particle_fraction'])
        
        # Convert particle_size from μm to mm
        particle_df['particle_size_mm'] = particle_df['particle_size'] / 1000.0
        
        # Group by code and extract percentiles
        percentiles = []
        
        for code in particle_df['code'].unique():
            sample_data = particle_df[particle_df['code'] == code].sort_values('particle_size_mm')
            
            if len(sample_data) < 2:
                continue
            
            # Get cumulative fractions
            sizes = sample_data['particle_size_mm'].values
            fractions = sample_data['particle_fraction'].values
            
            # Ensure fractions are in [0, 1] and sorted
            fractions = np.clip(fractions, 0, 1)
            
            # Interpolate to get percentiles
            try:
                # Use interpolation to find sizes at specific percentiles
                # Handle duplicate sizes
                unique_sizes, unique_indices = np.unique(sizes, return_index=True)
                unique_fractions = fractions[unique_indices]
                
                if len(unique_sizes) < 2:
                    continue
                
                # Interpolate
                interp_func = interpolate.interp1d(
                    unique_fractions, unique_sizes, 
                    kind='linear', 
                    bounds_error=False, 
                    fill_value='extrapolate'
                )
                
                # Extract percentiles
                D10 = float(interp_func(0.10))
                D30 = float(interp_func(0.30))
                D50 = float(interp_func(0.50))
                D60 = float(interp_func(0.60))
                D90 = float(interp_func(0.90))
                
                # Calculate derived parameters
                Cu = D60 / D10 if D10 > 0 else np.nan  # Uniformity coefficient
                Cc = (D30**2) / (D10 * D60) if (D10 > 0 and D60 > 0) else np.nan  # Curvature coefficient
                
                # Extract clay, silt, sand fractions (USDA classification)
                # Clay: < 0.002 mm, Silt: 0.002-0.05 mm, Sand: 0.05-2 mm
                clay_fraction = float(interp_func(0.002)) if 0.002 <= max(unique_fractions) else 0.0
                silt_fraction = float(interp_func(0.05)) if 0.05 <= max(unique_fractions) else 0.0
                
                # Get actual fractions at boundaries
                clay_idx = np.searchsorted(unique_sizes, 0.002, side='right')
                silt_idx = np.searchsorted(unique_sizes, 0.05, side='right')
                sand_idx = np.searchsorted(unique_sizes, 2.0, side='right')
                
                clay_pct = unique_fractions[min(clay_idx, len(unique_fractions)-1)] * 100
                silt_pct = (unique_fractions[min(silt_idx, len(unique_fractions)-1)] - unique_fractions[min(clay_idx, len(unique_fractions)-1)]) * 100
                sand_pct = (1.0 - unique_fractions[min(silt_idx, len(unique_fractions)-1)]) * 100
                
                percentiles.append({
                    'code': code,
                    'D10': D10,
                    'D30': D30,
                    'D50': D50,
                    'D60': D60,
                    'D90': D90,
                    'Cu': Cu,
                    'Cc': Cc,
                    'clay_pct': clay_pct,
                    'silt_pct': silt_pct,
                    'sand_pct': sand_pct
                })
            except Exception as e:
                # Skip samples with interpolation issues
                continue
        
        gsd_df = pd.DataFrame(percentiles)
        print(f"✓ Extracted GSD percentiles for {len(gsd_df)} samples")
        print(f"  D10 range: {gsd_df['D10'].min():.4f} - {gsd_df['D10'].max():.4f} mm")
        print(f"  D50 range: {gsd_df['D50'].min():.4f} - {gsd_df['D50'].max():.4f} mm")
        
        return gsd_df
    
    def process_soil_properties(self):
        """Process and clean soil properties"""
        print("\n" + "="*60)
        print("Processing Soil Properties")
        print("="*60)
        
        props_df = self.tables['soil_properties'].copy()
        
        # Calculate porosity if missing but bulk and particle density available
        mask = props_df['porosity'].isna() & props_df['bulk_density'].notna() & props_df['particle_density'].notna()
        props_df.loc[mask, 'porosity'] = 1 - (props_df.loc[mask, 'bulk_density'] / props_df.loc[mask, 'particle_density'])
        
        # Calculate void ratio
        props_df['void_ratio'] = props_df['porosity'] / (1 - props_df['porosity'])
        
        # Calculate specific gravity
        props_df['Gs'] = props_df['particle_density'] / 1.0  # Assuming water density = 1.0 g/cm³
        
        print(f"✓ Processed {len(props_df)} samples")
        print(f"  Bulk density: {props_df['bulk_density'].notna().sum()} available ({props_df['bulk_density'].notna().sum()/len(props_df)*100:.1f}%)")
        print(f"  Porosity: {props_df['porosity'].notna().sum()} available ({props_df['porosity'].notna().sum()/len(props_df)*100:.1f}%)")
        print(f"  OM content: {props_df['OM_content'].notna().sum()} available ({props_df['OM_content'].notna().sum()/len(props_df)*100:.1f}%)")
        print(f"  pH: {props_df['pH'].notna().sum()} available ({props_df['pH'].notna().sum()/len(props_df)*100:.1f}%)")
        
        return props_df
    
    def process_swcc_data(self, n_points=100):
        """Process SWCC data and interpolate to fixed grid"""
        print("\n" + "="*60)
        print("Processing SWCC Data")
        print("="*60)
        
        swcc_df = self.tables['swcc'].copy()
        
        # Remove rows with missing values
        swcc_df = swcc_df.dropna(subset=['preshead', 'theta'])
        
        # Convert suction from cm to kPa (1 cm = 0.0981 kPa, approximately 0.1 kPa)
        swcc_df['suction_kPa'] = swcc_df['preshead'] * 0.0981
        
        # Create log-scaled suction grid for interpolation
        # Range: 0.1 to 10^6 kPa
        suction_min = 0.1
        suction_max = 1e6
        suction_grid = np.logspace(np.log10(suction_min), np.log10(suction_max), n_points)
        
        # Process each sample
        swcc_curves = []
        swcc_stats = []
        
        for code in swcc_df['code'].unique():
            sample_data = swcc_df[swcc_df['code'] == code].sort_values('suction_kPa')
            
            if len(sample_data) < 3:  # Need at least 3 points for interpolation
                continue
            
            suction = sample_data['suction_kPa'].values
            theta = sample_data['theta'].values
            
            # Remove duplicates
            unique_indices = np.unique(suction, return_index=True)[1]
            suction = suction[unique_indices]
            theta = theta[unique_indices]
            
            if len(suction) < 3:
                continue
            
            # Interpolate to fixed grid
            try:
                # Use log interpolation for better accuracy
                log_suction = np.log10(suction + 0.1)  # Add small value to avoid log(0)
                log_grid = np.log10(suction_grid + 0.1)
                
                interp_func = interpolate.interp1d(
                    log_suction, theta,
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                
                theta_interp = interp_func(log_grid)
                
                # Ensure monotonicity (theta should decrease with suction)
                for i in range(1, len(theta_interp)):
                    if theta_interp[i] > theta_interp[i-1]:
                        theta_interp[i] = theta_interp[i-1]
                
                # Clip to reasonable bounds
                theta_interp = np.clip(theta_interp, 0, 1)
                
                # Calculate statistics
                theta_s = theta_interp[0]  # Saturated water content (at lowest suction)
                theta_r = theta_interp[-1]  # Residual water content (at highest suction)
                
                swcc_curves.append({
                    'code': code,
                    'suction_grid': suction_grid,
                    'theta_interp': theta_interp,
                    'theta_s': theta_s,
                    'theta_r': theta_r,
                    'n_points_original': len(suction)
                })
                
                swcc_stats.append({
                    'code': code,
                    'theta_s': theta_s,
                    'theta_r': theta_r,
                    'n_points': len(suction),
                    'suction_min': suction.min(),
                    'suction_max': suction.max()
                })
            except Exception as e:
                continue
        
        swcc_processed = pd.DataFrame(swcc_stats)
        print(f"✓ Processed SWCC data for {len(swcc_curves)} samples")
        print(f"  Average points per curve: {swcc_processed['n_points'].mean():.1f}")
        print(f"  Theta_s range: {swcc_processed['theta_s'].min():.3f} - {swcc_processed['theta_s'].max():.3f}")
        print(f"  Theta_r range: {swcc_processed['theta_r'].min():.3f} - {swcc_processed['theta_r'].max():.3f}")
        
        return swcc_curves, swcc_processed
    
    def create_feature_matrix(self, gsd_df, props_df, swcc_stats):
        """Create feature matrix for ML training"""
        print("\n" + "="*60)
        print("Creating Feature Matrix")
        print("="*60)
        
        # Merge all data
        features = gsd_df.merge(props_df, on='code', how='inner')
        features = features.merge(swcc_stats[['code', 'theta_s', 'theta_r']], on='code', how='inner')
        
        # Select features
        feature_columns = [
            # GSD percentiles
            'D10', 'D30', 'D50', 'D60', 'D90',
            'Cu', 'Cc',
            'clay_pct', 'silt_pct', 'sand_pct',
            # Physical properties
            'bulk_density', 'particle_density', 'porosity', 'void_ratio', 'Gs',
            # Chemical properties (optional)
            'OM_content', 'pH', 'EC',
            # SWCC parameters
            'theta_s', 'theta_r'
        ]
        
        # Create feature matrix
        feature_matrix = features[['code'] + [col for col in feature_columns if col in features.columns]].copy()
        
        # Remove samples with too many missing essential features
        essential_features = ['D10', 'D50', 'D60', 'bulk_density', 'porosity', 'theta_s', 'theta_r']
        missing_threshold = len(essential_features) // 2  # Allow up to half missing
        
        mask = feature_matrix[essential_features].isna().sum(axis=1) <= missing_threshold
        feature_matrix = feature_matrix[mask]
        
        print(f"✓ Created feature matrix: {len(feature_matrix)} samples × {len(feature_matrix.columns)-1} features")
        print(f"  Features: {list(feature_matrix.columns[1:])}")
        
        # Feature completeness
        print("\nFeature Completeness:")
        for col in feature_matrix.columns[1:]:
            completeness = feature_matrix[col].notna().sum() / len(feature_matrix) * 100
            print(f"  {col:20s}: {completeness:5.1f}%")
        
        return feature_matrix
    
    def create_swcc_matrix(self, swcc_curves, feature_matrix):
        """Create SWCC matrix (samples × interpolated points)"""
        print("\n" + "="*60)
        print("Creating SWCC Matrix")
        print("="*60)
        
        # Create matrix: samples × suction points
        codes_in_features = set(feature_matrix['code'].values)
        
        swcc_matrix = []
        swcc_codes = []
        
        for curve in swcc_curves:
            if curve['code'] in codes_in_features:
                swcc_matrix.append(curve['theta_interp'])
                swcc_codes.append(curve['code'])
        
        swcc_matrix = np.array(swcc_matrix)
        swcc_codes = np.array(swcc_codes)
        
        # Align with feature matrix
        feature_codes = feature_matrix['code'].values
        common_codes = np.intersect1d(feature_codes, swcc_codes)
        
        # Filter both matrices
        feature_matrix_aligned = feature_matrix[feature_matrix['code'].isin(common_codes)].sort_values('code')
        swcc_indices = [np.where(swcc_codes == code)[0][0] for code in common_codes]
        swcc_matrix_aligned = swcc_matrix[swcc_indices]
        
        print(f"✓ Created SWCC matrix: {len(swcc_matrix_aligned)} samples × {swcc_matrix_aligned.shape[1]} points")
        
        return swcc_matrix_aligned, feature_matrix_aligned
    
    def remove_outliers(self, feature_matrix, swcc_matrix, z_threshold=3):
        """Remove outliers using Z-score"""
        print("\n" + "="*60)
        print("Removing Outliers")
        print("="*60)
        
        # Calculate Z-scores for numeric features
        numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'code']
        
        z_scores = feature_matrix[numeric_cols].apply(zscore, nan_policy='omit')
        outlier_mask = (z_scores.abs() > z_threshold).any(axis=1)
        
        n_outliers = outlier_mask.sum()
        print(f"  Identified {n_outliers} outliers (Z-score > {z_threshold})")
        
        # Remove outliers
        feature_matrix_clean = feature_matrix[~outlier_mask].copy()
        swcc_matrix_clean = swcc_matrix[~outlier_mask]
        
        print(f"✓ Removed outliers: {len(feature_matrix_clean)} samples remaining")
        
        return feature_matrix_clean, swcc_matrix_clean
    
    def split_data(self, feature_matrix, swcc_matrix, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        """Split data into train/validation/test sets"""
        print("\n" + "="*60)
        print("Splitting Data")
        print("="*60)
        
        np.random.seed(random_state)
        n_samples = len(feature_matrix)
        indices = np.random.permutation(n_samples)
        
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]
        
        # Split feature matrices
        X_train = feature_matrix.iloc[train_idx].reset_index(drop=True)
        X_val = feature_matrix.iloc[val_idx].reset_index(drop=True)
        X_test = feature_matrix.iloc[test_idx].reset_index(drop=True)
        
        # Split SWCC matrices
        y_train = swcc_matrix[train_idx]
        y_val = swcc_matrix[val_idx]
        y_test = swcc_matrix[test_idx]
        
        print(f"✓ Data split:")
        print(f"  Training:   {len(X_train)} samples ({len(X_train)/n_samples*100:.1f}%)")
        print(f"  Validation: {len(X_val)} samples ({len(X_val)/n_samples*100:.1f}%)")
        print(f"  Test:       {len(X_test)} samples ({len(X_test)/n_samples*100:.1f}%)")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def generate_visualizations(self, feature_matrix, swcc_matrix, swcc_curves):
        """Generate data distribution visualizations"""
        print("\n" + "="*60)
        print("Generating Visualizations")
        print("="*60)
        
        # 1. Feature distributions
        print("  1. Feature distributions...")
        numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'code']
        
        n_cols = 4
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(numeric_cols[:len(axes)]):
            ax = axes[idx]
            data = feature_matrix[col].dropna()
            ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
            ax.set_title(f'{col}\n(n={len(data)})', fontsize=10)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("     ✓ Saved: feature_distributions.png")
        
        # 2. SWCC curves sample
        print("  2. Sample SWCC curves...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # Select diverse samples
        sample_indices = np.linspace(0, len(swcc_curves)-1, 4, dtype=int)
        suction_grid = swcc_curves[0]['suction_grid']
        
        for idx, ax in enumerate(axes):
            curve_idx = sample_indices[idx]
            curve = swcc_curves[curve_idx]
            code = curve['code']
            
            ax.semilogx(curve['suction_grid'], curve['theta_interp'], 'b-', linewidth=2, label='Interpolated')
            ax.set_xlabel('Suction (kPa)', fontsize=12)
            ax.set_ylabel('Water Content (θ)', fontsize=12)
            ax.set_title(f'Sample {code}', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'swcc_samples.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("     ✓ Saved: swcc_samples.png")
        
        # 3. Feature correlation matrix
        print("  3. Feature correlation matrix...")
        corr_features = ['D10', 'D50', 'D60', 'Cu', 'Cc', 'clay_pct', 'bulk_density', 'porosity', 'OM_content', 'theta_s', 'theta_r']
        corr_features = [f for f in corr_features if f in feature_matrix.columns]
        
        corr_matrix = feature_matrix[corr_features].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'feature_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("     ✓ Saved: feature_correlation.png")
        
        # 4. SWCC curve statistics
        print("  4. SWCC statistics...")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Theta_s distribution
        theta_s_values = [c['theta_s'] for c in swcc_curves]
        axes[0].hist(theta_s_values, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0].set_xlabel('θs (Saturated Water Content)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of θs')
        axes[0].grid(True, alpha=0.3)
        
        # Theta_r distribution
        theta_r_values = [c['theta_r'] for c in swcc_curves]
        axes[1].hist(theta_r_values, bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
        axes[1].set_xlabel('θr (Residual Water Content)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of θr')
        axes[1].grid(True, alpha=0.3)
        
        # Number of points per curve
        n_points = [c['n_points_original'] for c in swcc_curves]
        axes[2].hist(n_points, bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
        axes[2].set_xlabel('Number of Data Points')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('SWCC Data Points per Sample')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'swcc_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("     ✓ Saved: swcc_statistics.png")
        
        # 5. All SWCC curves overlay
        print("  5. All SWCC curves overlay...")
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for curve in swcc_curves[:100]:  # Plot first 100 to avoid clutter
            ax.semilogx(curve['suction_grid'], curve['theta_interp'], 'b-', alpha=0.1, linewidth=0.5)
        
        # Add mean curve
        all_curves = np.array([c['theta_interp'] for c in swcc_curves])
        mean_curve = np.mean(all_curves, axis=0)
        std_curve = np.std(all_curves, axis=0)
        
        ax.semilogx(suction_grid, mean_curve, 'r-', linewidth=3, label='Mean')
        ax.fill_between(suction_grid, mean_curve - std_curve, mean_curve + std_curve, 
                       alpha=0.3, color='red', label='±1 Std Dev')
        
        ax.set_xlabel('Suction (kPa)', fontsize=12)
        ax.set_ylabel('Water Content (θ)', fontsize=12)
        ax.set_title('All SWCC Curves (n={})'.format(len(swcc_curves)), fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'swcc_all_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("     ✓ Saved: swcc_all_curves.png")
        
        print(f"\n✓ All visualizations saved to: {self.viz_dir}")
    
    def save_processed_data(self, splits, feature_matrix, swcc_matrix, swcc_curves=None):
        """Save processed data"""
        print("\n" + "="*60)
        print("Saving Processed Data")
        print("="*60)
        
        # Save feature matrices
        splits['train'][0].to_csv(self.output_dir / 'X_train.csv', index=False)
        splits['val'][0].to_csv(self.output_dir / 'X_val.csv', index=False)
        splits['test'][0].to_csv(self.output_dir / 'X_test.csv', index=False)
        print("✓ Saved feature matrices: X_train.csv, X_val.csv, X_test.csv")
        
        # Save SWCC matrices (numpy arrays)
        np.save(self.output_dir / 'y_train.npy', splits['train'][1])
        np.save(self.output_dir / 'y_val.npy', splits['val'][1])
        np.save(self.output_dir / 'y_test.npy', splits['test'][1])
        print("✓ Saved SWCC matrices: y_train.npy, y_val.npy, y_test.npy")
        
        # Save full feature matrix
        feature_matrix.to_csv(self.output_dir / 'features_full.csv', index=False)
        print("✓ Saved full feature matrix: features_full.csv")
        
        # Save suction grid
        if swcc_curves and len(swcc_curves) > 0:
            suction_grid = swcc_curves[0]['suction_grid']
            np.save(self.output_dir / 'suction_grid.npy', suction_grid)
            print("✓ Saved suction grid: suction_grid.npy")
        
        # Save metadata
        metadata = {
            'n_samples': len(feature_matrix),
            'n_features': len(feature_matrix.columns) - 1,  # Exclude 'code'
            'n_swcc_points': swcc_matrix.shape[1],
            'feature_columns': list(feature_matrix.columns[1:]),
            'train_size': len(splits['train'][0]),
            'val_size': len(splits['val'][0]),
            'test_size': len(splits['test'][0])
        }
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print("✓ Saved metadata: metadata.json")
        
        print(f"\n✓ All data saved to: {self.output_dir}")
    
    def run_full_pipeline(self):
        """Run complete preprocessing pipeline"""
        print("\n" + "="*80)
        print("UNSODA Data Preprocessing Pipeline - Phase 1")
        print("="*80)
        
        # Load data
        self.load_data()
        
        # Extract GSD percentiles
        gsd_df = self.extract_gsd_percentiles()
        
        # Process soil properties
        props_df = self.process_soil_properties()
        
        # Process SWCC data
        swcc_curves, swcc_stats = self.process_swcc_data()
        
        # Create feature matrix
        feature_matrix = self.create_feature_matrix(gsd_df, props_df, swcc_stats)
        
        # Create SWCC matrix
        swcc_matrix, feature_matrix_aligned = self.create_swcc_matrix(swcc_curves, feature_matrix)
        
        # Remove outliers
        feature_matrix_clean, swcc_matrix_clean = self.remove_outliers(
            feature_matrix_aligned, swcc_matrix
        )
        
        # Split data
        splits = self.split_data(feature_matrix_clean, swcc_matrix_clean)
        
        # Generate visualizations
        self.generate_visualizations(feature_matrix_clean, swcc_matrix_clean, swcc_curves)
        
        # Save processed data
        self.save_processed_data(splits, feature_matrix_clean, swcc_matrix_clean, swcc_curves)
        
        # Store for return
        self.features_df = feature_matrix_clean
        self.swcc_data = swcc_matrix_clean
        
        print("\n" + "="*80)
        print("Preprocessing Complete!")
        print("="*80)
        print(f"\nFinal Dataset:")
        print(f"  Samples: {len(feature_matrix_clean)}")
        print(f"  Features: {len(feature_matrix_clean.columns) - 1}")
        print(f"  SWCC points: {swcc_matrix_clean.shape[1]}")
        print(f"\nReady for Phase 2: GAN Development!")
        
        return splits, feature_matrix_clean, swcc_matrix_clean

if __name__ == "__main__":
    preprocessor = UNSODAPreprocessor()
    splits, features, swcc = preprocessor.run_full_pipeline()
