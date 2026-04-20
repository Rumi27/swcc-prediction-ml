#!/usr/bin/env python3
"""
Comprehensive GSHP (Global database of soil hydraulic properties) Validation Script
for VGParamNet model trained on UNSODA 2.0.

This script:
1. Loads GSHP dataset
2. Maps GSHP columns to UNSODA feature structure (16 features)
3. Handles missing features via imputation from UNSODA training data
4. Applies proper standardization using UNSODA training statistics
5. Evaluates VGParamNet predictions on GSHP
6. Handles unit conversions (especially α parameter)
7. Generates comprehensive evaluation metrics and figures
8. Compares GSHP performance to UNSODA test set

Author: Generated for cross-database validation
Date: 2026-02-16
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU for inference

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Add project root to path
# Script is at: paper_1_swcc_ml/paper_1_swcc_ml/scripts/evaluation/evaluate_gshp_comprehensive.py
# Need to go up to: paper_1_swcc_ml/paper_1_swcc_ml/
SCRIPT_DIR = Path(__file__).resolve()
ROOT_DIR = SCRIPT_DIR.parent.parent.parent  # paper_1_swcc_ml/paper_1_swcc_ml/

# Data is at paper_1_swcc_ml/data/, so check both possible locations
# Try ROOT_DIR.parent first (paper_1_swcc_ml/data/) since that's where the actual data is
# Check for GSHP file specifically to determine correct path
potential_data_roots = [ROOT_DIR.parent, ROOT_DIR]  # Try parent first
DATA_ROOT = None
for root in potential_data_roots:
    test_path = root / "data" / "GSHP_downloaded" / "WRC_dataset_surya_et_al_2021_final.csv"
    if test_path.exists():
        DATA_ROOT = root
        break

# If still not found, default to parent
if DATA_ROOT is None:
    DATA_ROOT = ROOT_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.vg_param_net import VGParamNet, vg_theta

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
DATA_DIR = ROOT_DIR / "data_pinn_normalized"

# Try multiple possible GSHP paths (handle colon in directory name)
GSHP_PATHS = [
    DATA_ROOT / "data" / "GSHP_downloaded" / "WRC_dataset_surya_et_al_2021_final.csv",
    DATA_ROOT / "data" / "GSHP: Global database of soil hydraulic properties" / "WRC_dataset_surya_et_al_2021_final.csv",
    ROOT_DIR / "data" / "GSHP_downloaded" / "WRC_dataset_surya_et_al_2021_final.csv",
    ROOT_DIR / "data" / "GSHP: Global database of soil hydraulic properties" / "WRC_dataset_surya_et_al_2021_final.csv",
]

# Find the first existing path
GSHP_DATA_PATH = None
for path in GSHP_PATHS:
    if path.exists():
        GSHP_DATA_PATH = path
        break

if GSHP_DATA_PATH is None:
    # If neither exists, use the first one and let it fail with a clearer error
    GSHP_DATA_PATH = GSHP_PATHS[0]

MODEL_PATH = ROOT_DIR / "results_pinn_fixed" / "vgparamnet" / "vgparamnet_best.keras"
OUTPUT_DIR = ROOT_DIR / "results_gshp_validation"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Unit conversion factors (to be determined from data analysis)
# Model was trained on UNSODA with α in 1/kPa
# GSHP may have different units - we'll detect and convert
ALPHA_CONVERSION_FACTORS = {
    '1/kPa': 1.0,      # No conversion needed
    '1/cm': 0.1,       # 1/cm to 1/kPa (approx, assuming 1 kPa ≈ 10 cm)
    '1/m': 0.001,      # 1/m to 1/kPa
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_metadata():
    """Load metadata from training data directory."""
    metadata_path = DATA_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
    
    return meta


def compute_psd_percentiles(psd_data, percentiles=[10, 30, 50, 60, 90]):
    """
    Compute PSD percentiles from particle size distribution data.
    
    Args:
        psd_data: DataFrame or dict with particle size columns
        percentiles: List of percentiles to compute (e.g., [10, 30, 50, 60, 90])
    
    Returns:
        Dictionary with D10, D30, D50, D60, D90 values (in mm)
    """
    # This is a placeholder - actual implementation depends on GSHP PSD format
    # GSHP may have PSD as cumulative percentages or individual size classes
    # For now, return None to indicate missing data
    return None


def detect_alpha_units(alpha_obs, alpha_pred):
    """
    Detect the units of GSHP alpha values by comparing distributions.
    
    Args:
        alpha_obs: Observed alpha from GSHP
        alpha_pred: Predicted alpha from model (in 1/kPa)
    
    Returns:
        Conversion factor to convert GSHP alpha to 1/kPa
    """
    # Compare median values
    median_obs = np.median(alpha_obs[alpha_obs > 0])
    median_pred = np.median(alpha_pred[alpha_pred > 0])
    
    if median_obs > 0 and median_pred > 0:
        ratio = median_obs / median_pred
        
        # Heuristic: if ratio is ~10, likely 1/cm → 1/kPa
        # if ratio is ~1000, likely 1/m → 1/kPa
        if 5 < ratio < 20:
            print(f"  Detected alpha units: likely 1/cm (ratio={ratio:.2f})")
            return 0.1  # Convert 1/cm to 1/kPa
        elif 100 < ratio < 10000:
            print(f"  Detected alpha units: likely 1/m (ratio={ratio:.2f})")
            return 0.001  # Convert 1/m to 1/kPa
        elif 0.1 < ratio < 2:
            print(f"  Detected alpha units: likely 1/kPa (ratio={ratio:.2f})")
            return 1.0  # Already in correct units
        else:
            print(f"  Warning: Unusual ratio {ratio:.2f}, assuming 1/kPa")
            return 1.0
    
    return 1.0  # Default: assume already in 1/kPa


def prepare_gshp_features(df_gshp, X_train, feature_cols, match_cols):
    """
    Prepare GSHP features for model input.
    
    Args:
        df_gshp: GSHP dataframe
        X_train: UNSODA training data (for imputation and standardization)
        feature_cols: List of feature column names expected by model
        match_cols: Columns to use for nearest neighbor matching
    
    Returns:
        X_gshp: Prepared feature matrix [N, D]
        df_clean: Cleaned GSHP dataframe aligned with X_gshp
    """
    print("\n" + "="*80)
    print("STEP 1: Loading and Cleaning GSHP Data")
    print("="*80)
    
    print(f"Total GSHP samples: {len(df_gshp)}")
    print(f"GSHP columns: {list(df_gshp.columns)[:20]}...")  # Show first 20
    
    # ========================================================================
    # Map GSHP columns to UNSODA feature names
    # ========================================================================
    
    # Essential columns that must exist
    gshp_to_unsoda = {
        # Texture (required)
        'sand_tot_psa': 'sand_pct',
        'silt_tot_psa': 'silt_pct',
        'clay_tot_psa': 'clay_pct',
        
        # Bulk density (required)
        'db_od': 'bulk_density',
        
        # VG parameters (for evaluation)
        'alpha': 'alpha',
        'n': 'n',
        'thetas': 'theta_s',
        'thetar': 'theta_r',
    }
    
    # Optional columns
    optional_mappings = {
        'ph_h2o': 'pH',
        'oc': 'OM_content',  # Will convert OC to OM
        'porosity': 'porosity',
    }
    
    # Check which columns exist
    print("\nChecking GSHP column availability...")
    available_cols = set(df_gshp.columns)
    
    X_input = pd.DataFrame(index=df_gshp.index)
    
    # Map required columns
    missing_required = []
    for gshp_col, unsoda_col in gshp_to_unsoda.items():
        if gshp_col in available_cols:
            X_input[unsoda_col] = pd.to_numeric(df_gshp[gshp_col], errors='coerce')
            print(f"  ✓ {gshp_col} → {unsoda_col}")
        else:
            if unsoda_col in ['alpha', 'n']:
                # These are for evaluation, not input
                continue
            missing_required.append(gshp_col)
            print(f"  ✗ {gshp_col} → {unsoda_col} (MISSING)")
    
    if missing_required:
        print(f"\nERROR: Missing required columns: {missing_required}")
        return None, None
    
    # Map optional columns
    for gshp_col, unsoda_col in optional_mappings.items():
        if gshp_col in available_cols:
            if gshp_col == 'oc':
                # Convert OC to OM: OM ≈ 1.724 * OC
                X_input[unsoda_col] = pd.to_numeric(df_gshp[gshp_col], errors='coerce') * 1.724
            else:
                X_input[unsoda_col] = pd.to_numeric(df_gshp[gshp_col], errors='coerce')
            print(f"  ✓ {gshp_col} → {unsoda_col}")
        else:
            print(f"  - {gshp_col} → {unsoda_col} (not available, will impute)")
    
    # ========================================================================
    # Compute derived features
    # ========================================================================
    
    # Porosity: compute if not available
    if 'porosity' not in X_input.columns or X_input['porosity'].isna().all():
        print("\nComputing porosity from bulk density...")
        # Assume particle density ≈ 2.65 g/cm³ if not available
        particle_density = 2.65
        X_input['porosity'] = 1 - (X_input['bulk_density'] / particle_density)
    
    # PSD percentiles: Try to compute from available data
    # Note: GSHP may not have D10-D90 directly
    # For now, we'll impute these from UNSODA neighbors
    print("\nPSD percentiles (D10, D30, D50, D60, D90): Will impute from UNSODA")
    
    # ========================================================================
    # Quality Control: Filter invalid samples
    # ========================================================================
    
    print("\nApplying quality control filters...")
    initial_count = len(X_input)
    
    # Filter 1: Required features must be non-null
    required_features = ['sand_pct', 'silt_pct', 'clay_pct', 'bulk_density', 
                         'porosity', 'theta_s', 'theta_r']
    X_input = X_input.dropna(subset=required_features)
    print(f"  After removing NaN in required features: {len(X_input)}/{initial_count}")
    
    # Filter 2: Physical bounds
    valid_mask = (
        (X_input['sand_pct'] >= 0) & (X_input['sand_pct'] <= 100) &
        (X_input['silt_pct'] >= 0) & (X_input['silt_pct'] <= 100) &
        (X_input['clay_pct'] >= 0) & (X_input['clay_pct'] <= 100) &
        (X_input['bulk_density'] > 0) & (X_input['bulk_density'] < 3.0) &
        (X_input['porosity'] > 0) & (X_input['porosity'] < 1) &
        (X_input['theta_s'] > 0) & (X_input['theta_s'] <= 1) &
        (X_input['theta_r'] >= 0) & (X_input['theta_r'] < X_input['theta_s'])
    )
    X_input = X_input[valid_mask]
    print(f"  After physical bounds check: {len(X_input)}/{initial_count}")
    
    # Align df_gshp with X_input
    df_clean = df_gshp.loc[X_input.index].copy()
    
    # ========================================================================
    # Impute missing features from UNSODA training data
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 2: Imputing Missing Features from UNSODA Training Data")
    print("="*80)
    
    # Features to match on for nearest neighbor
    match_features = ['sand_pct', 'silt_pct', 'clay_pct', 'bulk_density']
    
    # Check which features are missing
    missing_features = [col for col in feature_cols if col not in X_input.columns]
    if missing_features:
        print(f"Missing features to impute: {missing_features}")
    else:
        print("All features available in GSHP!")
    
    # Build nearest neighbor matcher
    print("\nBuilding nearest neighbor matcher on UNSODA training data...")
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    
    # Prepare UNSODA match features (fill NaN with 0 for matching)
    X_train_match = X_train[match_features].fillna(0).values
    
    # Standardize for matching (use UNSODA statistics)
    scaler_match = StandardScaler()
    X_train_match_scaled = scaler_match.fit_transform(X_train_match)
    nn.fit(X_train_match_scaled)
    
    # Prepare GSHP match features
    X_gshp_match = X_input[match_features].fillna(0).values
    X_gshp_match_scaled = scaler_match.transform(X_gshp_match)
    
    # Find nearest neighbors (in batches for large datasets)
    print("Finding nearest UNSODA neighbors for GSHP samples...")
    batch_size = 5000
    n_samples = len(X_gshp_match_scaled)
    n_batches = int(np.ceil(n_samples / batch_size))
    
    neighbor_indices = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        
        if i % 10 == 0:
            print(f"  Processing batch {i+1}/{n_batches}...")
        
        batch = X_gshp_match_scaled[start_idx:end_idx]
        _, indices = nn.kneighbors(batch)
        neighbor_indices.append(indices.flatten())
    
    neighbor_indices = np.concatenate(neighbor_indices)
    
    # Construct imputed feature matrix
    print("\nConstructing imputed feature matrix...")
    X_imputed = X_train.iloc[neighbor_indices][feature_cols].copy()
    X_imputed.index = X_input.index
    
    # Overwrite with observed GSHP values where available
    for col in feature_cols:
        if col in X_input.columns:
            # Only overwrite where GSHP value is not NaN
            valid_mask = X_input[col].notna()
            X_imputed.loc[valid_mask, col] = X_input.loc[valid_mask, col]
    
    # Final check: ensure no NaN
    if X_imputed.isna().any().any():
        print("Warning: Some features still have NaN after imputation. Filling with median...")
        X_imputed = X_imputed.fillna(X_imputed.median())
    
    print(f"\nFinal GSHP dataset size: {len(X_imputed)} samples")
    print(f"Feature matrix shape: {X_imputed.shape}")
    
    return X_imputed, df_clean


def standardize_features(X_gshp, X_train, feature_cols):
    """
    Standardize GSHP features using UNSODA training statistics.
    
    Args:
        X_gshp: GSHP feature matrix
        X_train: UNSODA training data
        feature_cols: Feature column names
    
    Returns:
        X_gshp_scaled: Standardized feature matrix
        scaler: Fitted StandardScaler
    """
    print("\n" + "="*80)
    print("STEP 3: Standardizing Features Using UNSODA Training Statistics")
    print("="*80)
    
    # Fit scaler on UNSODA training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[feature_cols])
    
    # Transform GSHP data
    X_gshp_scaled = scaler.transform(X_gshp[feature_cols])
    
    print("✓ Features standardized")
    
    return X_gshp_scaled, scaler


def evaluate_parameters(alpha_pred, n_pred, alpha_obs, n_obs, output_dir):
    """
    Evaluate parameter-level predictions.
    
    Args:
        alpha_pred: Predicted α values
        n_pred: Predicted n values
        alpha_obs: Observed α values (may need unit conversion)
        n_obs: Observed n values
        output_dir: Output directory for figures
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    print("\n" + "="*80)
    print("STEP 5: Parameter-Level Evaluation")
    print("="*80)
    
    # Hard-code GSHP alpha units: GSHP alpha is in 1/m head → convert to 1/kPa
    # Using 1 kPa ≈ 0.10197 m H2O → α[1/kPa] = α[1/m] * 0.10197
    print("\nApplying hard-coded alpha unit conversion: GSHP alpha (1/m) → 1/kPa (×0.10197)")
    conversion_factor = 0.10197
    alpha_obs_converted = alpha_obs * conversion_factor
    
    # Compute metrics
    metrics = {
        'alpha': {
            'r2': r2_score(alpha_obs_converted, alpha_pred),
            'rmse': np.sqrt(mean_squared_error(alpha_obs_converted, alpha_pred)),
            'mae': mean_absolute_error(alpha_obs_converted, alpha_pred),
            'conversion_factor': conversion_factor,
        },
        'n': {
            'r2': r2_score(n_obs, n_pred),
            'rmse': np.sqrt(mean_squared_error(n_obs, n_pred)),
            'mae': mean_absolute_error(n_obs, n_pred),
        }
    }
    
    print(f"\nAlpha (after unit conversion):")
    print(f"  R² = {metrics['alpha']['r2']:.4f}")
    print(f"  RMSE = {metrics['alpha']['rmse']:.6f}")
    print(f"  MAE = {metrics['alpha']['mae']:.6f}")
    print(f"  Conversion factor = {conversion_factor}")
    
    print(f"\nn:")
    print(f"  R² = {metrics['n']['r2']:.4f}")
    print(f"  RMSE = {metrics['n']['rmse']:.4f}")
    print(f"  MAE = {metrics['n']['mae']:.4f}")
    
    # Generate scatter plots
    print("\nGenerating scatter plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Alpha scatter
    ax = axes[0]
    ax.scatter(alpha_obs_converted, alpha_pred, alpha=0.3, s=2, edgecolors='none')
    ax.plot([alpha_obs_converted.min(), alpha_obs_converted.max()], 
            [alpha_obs_converted.min(), alpha_obs_converted.max()], 
            'r--', lw=2, label='1:1 line')
    ax.set_xlabel('Observed α (GSHP, converted to 1/kPa)', fontsize=12)
    ax.set_ylabel('Predicted α (VGParamNet, 1/kPa)', fontsize=12)
    ax.set_title(f'Alpha Comparison\nR² = {metrics["alpha"]["r2"]:.3f}, RMSE = {metrics["alpha"]["rmse"]:.4f}', 
                 fontsize=13)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # n scatter
    ax = axes[1]
    ax.scatter(n_obs, n_pred, alpha=0.3, s=2, edgecolors='none')
    ax.plot([n_obs.min(), n_obs.max()], 
            [n_obs.min(), n_obs.max()], 
            'r--', lw=2, label='1:1 line')
    ax.set_xlabel('Observed n (GSHP)', fontsize=12)
    ax.set_ylabel('Predicted n (VGParamNet)', fontsize=12)
    ax.set_title(f'n Comparison\nR² = {metrics["n"]["r2"]:.3f}, RMSE = {metrics["n"]["rmse"]:.4f}', 
                 fontsize=13)
    ax.set_xlim(1, max(n_obs.max(), n_pred.max()) * 1.1)
    ax.set_ylim(1, max(n_obs.max(), n_pred.max()) * 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "gshp_parameter_comparison.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir / 'gshp_parameter_comparison.png'}")
    
    return metrics, alpha_obs_converted


def evaluate_curves(alpha_pred, n_pred, theta_s, theta_r, suction_grid, output_dir):
    """
    Generate and evaluate SWCC curves from predicted parameters.
    
    Args:
        alpha_pred: Predicted α values
        n_pred: Predicted n values
        theta_s: Saturated water content
        theta_r: Residual water content
        suction_grid: Suction grid for curve generation
        output_dir: Output directory
    
    Returns:
        curves_pred: Predicted SWCC curves [N, P]
    """
    print("\n" + "="*80)
    print("STEP 6: Generating SWCC Curves from Predicted Parameters")
    print("="*80)
    
    # Convert to tensors
    psi_tf = tf.constant(suction_grid, dtype=tf.float32)
    psi_tf = tf.tile(psi_tf[None, :], [len(alpha_pred), 1])  # [N, P]
    
    theta_s_tf = tf.constant(theta_s, dtype=tf.float32)
    theta_r_tf = tf.constant(theta_r, dtype=tf.float32)
    alpha_tf = tf.constant(alpha_pred, dtype=tf.float32)
    n_tf = tf.constant(n_pred, dtype=tf.float32)
    
    # Generate curves
    print("Generating curves...")
    curves_pred = vg_theta(psi_tf, theta_s_tf, theta_r_tf, alpha_tf, n_tf)
    curves_pred = curves_pred.numpy()
    
    print(f"  ✓ Generated {len(curves_pred)} curves")
    print(f"  Curve shape: {curves_pred.shape}")
    
    # Save curves
    np.save(output_dir / "gshp_predicted_curves.npy", curves_pred)
    print(f"  ✓ Saved: {output_dir / 'gshp_predicted_curves.npy'}")
    
    return curves_pred


def reconstruct_gshp_curves(alpha_obs_kpa, n_obs, theta_s, theta_r, suction_grid, output_dir):
    """
    Reconstruct GSHP θ(ψ) curves from GSHP parameters on the common suction grid.

    Args:
        alpha_obs_kpa: Observed α values converted to 1/kPa
        n_obs: Observed n values
        theta_s: Saturated water content (GSHP)
        theta_r: Residual water content (GSHP)
        suction_grid: Suction grid in kPa
        output_dir: Output directory

    Returns:
        curves_gshp: Reconstructed GSHP SWCC curves [N, P]
    """
    print("\n" + "="*80)
    print("STEP 6b: Reconstructing GSHP SWCC Curves from Reported Parameters")
    print("="*80)

    psi_tf = tf.constant(suction_grid, dtype=tf.float32)
    psi_tf = tf.tile(psi_tf[None, :], [len(alpha_obs_kpa), 1])  # [N, P]

    theta_s_tf = tf.constant(theta_s, dtype=tf.float32)
    theta_r_tf = tf.constant(theta_r, dtype=tf.float32)
    alpha_tf = tf.constant(alpha_obs_kpa, dtype=tf.float32)
    n_tf = tf.constant(n_obs, dtype=tf.float32)

    print("Generating GSHP curves...")
    curves_gshp = vg_theta(psi_tf, theta_s_tf, theta_r_tf, alpha_tf, n_tf)
    curves_gshp = curves_gshp.numpy()

    print(f"  ✓ Generated {len(curves_gshp)} GSHP curves")
    print(f"  Curve shape: {curves_gshp.shape}")

    np.save(output_dir / "gshp_observed_curves.npy", curves_gshp)
    print(f"  ✓ Saved: {output_dir / 'gshp_observed_curves.npy'}")

    return curves_gshp


def compute_curve_metrics(curves_pred, curves_gshp, suction_grid):
    """
    Compute global and regime-specific RMSE/MAE between predicted and GSHP curves.

    Args:
        curves_pred: Predicted θ(ψ) [N, P]
        curves_gshp: GSHP θ(ψ) [N, P]
        suction_grid: Suction grid [P]

    Returns:
        metrics: dict with global / wet / dry RMSE & MAE
    """
    print("\n" + "="*80)
    print("STEP 7: Curve-Space Metrics on GSHP")
    print("="*80)

    assert curves_pred.shape == curves_gshp.shape, "Predicted and GSHP curves must have same shape"

    diff = curves_pred - curves_gshp
    mse_global = np.mean(diff**2)
    rmse_global = float(np.sqrt(mse_global))
    mae_global = float(np.mean(np.abs(diff)))

    # Regime masks
    psi = suction_grid
    wet_mask = psi < 1e2
    dry_mask = psi > 1e4

    def regime_metrics(mask, name):
        if not np.any(mask):
            return {"rmse": float("nan"), "mae": float("nan")}
        d = diff[:, mask]
        mse = np.mean(d**2)
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(d)))
        print(f"  {name}: RMSE={rmse:.4f}, MAE={mae:.4f}")
        return {"rmse": rmse, "mae": mae}

    print(f"  Global: RMSE={rmse_global:.4f}, MAE={mae_global:.4f}")
    wet_metrics = regime_metrics(wet_mask, "Wet-end (ψ < 10^2 kPa)")
    dry_metrics = regime_metrics(dry_mask, "Dry-end (ψ > 10^4 kPa)")

    metrics = {
        "global": {"rmse": rmse_global, "mae": mae_global},
        "wet_end": wet_metrics,
        "dry_end": dry_metrics,
    }

    return metrics


def compute_knee_metrics(curves, theta_s, theta_r, suction_grid):
    """
    Compute ψ50 (Se=0.5) and max |dθ/dlogψ| for a set of curves.

    Args:
        curves: θ(ψ) [N, P]
        theta_s: [N]
        theta_r: [N]
        suction_grid: ψ grid [P]

    Returns:
        psi50: [N] knee locations (kPa)
        max_slope: [N] max |dθ/dlogψ|
    """
    psi = suction_grid
    log_psi = np.log10(psi)

    theta_s = theta_s.reshape(-1, 1)
    theta_r = theta_r.reshape(-1, 1)
    theta_range = np.maximum(theta_s - theta_r, 1e-6)

    Se = (curves - theta_r) / theta_range
    Se = np.clip(Se, 1e-6, 1 - 1e-6)

    # ψ50: Se crosses 0.5
    target = 0.5
    psi50 = np.full(curves.shape[0], np.nan, dtype=np.float32)

    for i in range(curves.shape[0]):
        Se_i = Se[i]
        # Find first index where Se <= 0.5 going from wet to dry
        idx = np.where(Se_i <= target)[0]
        if len(idx) == 0 or idx[0] == 0:
            continue
        k = idx[0]
        # Linear interpolation in log-ψ space between k-1 and k
        x0, x1 = log_psi[k-1], log_psi[k]
        y0, y1 = Se_i[k-1], Se_i[k]
        if y1 == y0:
            psi50[i] = 10**x1
        else:
            t = (target - y0) / (y1 - y0)
            log_psi50 = x0 + t * (x1 - x0)
            psi50[i] = 10**log_psi50

    # max_slope: max |dθ/dlogψ|
    dtheta = np.diff(curves, axis=1)
    dlogpsi = np.diff(log_psi)[None, :]
    slope = dtheta / dlogpsi
    max_slope = np.max(np.abs(slope), axis=1)

    return psi50, max_slope


def main():
    """Main evaluation function."""
    print("="*80)
    print("GSHP External Validation for VGParamNet")
    print("="*80)
    print(f"Model: {MODEL_PATH}")
    print(f"GSHP Data: {GSHP_DATA_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*80)
    
    # Load metadata
    print("\nLoading metadata...")
    metadata = load_metadata()
    feature_cols = metadata.get('feature_cols', [])
    print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")
    
    # Load UNSODA training data
    print("\nLoading UNSODA training data...")
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    print(f"  UNSODA training samples: {len(X_train)}")
    print(f"  UNSODA features: {list(X_train.columns)[:10]}...")
    
    # Load GSHP data
    print("\nLoading GSHP data...")
    print(f"  Looking for GSHP file at: {GSHP_DATA_PATH}")
    
    if not GSHP_DATA_PATH.exists():
        print(f"  ✗ GSHP file not found at: {GSHP_DATA_PATH}")
        print("\n  Trying alternative paths...")
        for alt_path in GSHP_PATHS:
            print(f"    - {alt_path} {'✓ EXISTS' if alt_path.exists() else '✗ NOT FOUND'}")
        print("\n  Please ensure GSHP dataset is downloaded and placed in one of these locations:")
        for path in GSHP_PATHS:
            print(f"    {path.parent}")
        return
    
    try:
        df_gshp = pd.read_csv(GSHP_DATA_PATH, encoding='latin-1', low_memory=False)
        print(f"  ✓ Loaded {len(df_gshp)} GSHP samples from: {GSHP_DATA_PATH.name}")
    except Exception as e:
        print(f"  ✗ Error loading GSHP data: {e}")
        print(f"  File path: {GSHP_DATA_PATH}")
        return
    
    # Prepare GSHP features
    X_gshp, df_clean = prepare_gshp_features(
        df_gshp, X_train, feature_cols, 
        match_cols=['sand_pct', 'silt_pct', 'clay_pct', 'bulk_density']
    )
    
    if X_gshp is None:
        print("ERROR: Failed to prepare GSHP features")
        return
    
    # Standardize features
    X_gshp_scaled, scaler = standardize_features(X_gshp, X_train, feature_cols)
    
    # Load model
    print("\n" + "="*80)
    print("STEP 4: Loading VGParamNet Model")
    print("="*80)
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = keras.models.load_model(
            MODEL_PATH, 
            custom_objects={"VGParamNet": VGParamNet},
            compile=False
        )
        print("  ✓ Model loaded successfully")
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        return
    
    # Make predictions
    print("\nMaking predictions...")
    X_input_tf = tf.constant(X_gshp_scaled, dtype=tf.float32)
    alpha_pred, n_pred = model(X_input_tf, training=False)
    alpha_pred = alpha_pred.numpy().flatten()
    n_pred = n_pred.numpy().flatten()
    
    print(f"  ✓ Predicted {len(alpha_pred)} samples")
    print(f"  Alpha range: {alpha_pred.min():.4f} - {alpha_pred.max():.4f}")
    print(f"  n range: {n_pred.min():.4f} - {n_pred.max():.4f}")
    
    # Get observed values from GSHP
    alpha_obs = df_clean['alpha'].values
    n_obs = df_clean['n'].values
    theta_s = X_gshp['theta_s'].values
    theta_r = X_gshp['theta_r'].values
    
    # Evaluate parameters (and get unit-harmonized alpha)
    metrics_params, alpha_obs_kpa = evaluate_parameters(alpha_pred, n_pred, alpha_obs, n_obs, OUTPUT_DIR)
    
    # Generate predicted curves from VGParamNet
    suction_grid = np.load(DATA_DIR / "suction_grid.npy")
    curves_pred = evaluate_curves(alpha_pred, n_pred, theta_s, theta_r, suction_grid, OUTPUT_DIR)
    
    # Reconstruct GSHP curves from reported parameters (after unit harmonization)
    curves_gshp = reconstruct_gshp_curves(alpha_obs_kpa, n_obs, theta_s, theta_r, suction_grid, OUTPUT_DIR)
    
    # Curve-space metrics
    metrics_curves = compute_curve_metrics(curves_pred, curves_gshp, suction_grid)
    
    # Knee metrics
    print("\n" + "="*80)
    print("STEP 8: Knee Metrics on GSHP (ψ50, max_slope)")
    print("="*80)
    psi50_pred, maxslope_pred = compute_knee_metrics(curves_pred, theta_s, theta_r, suction_grid)
    psi50_gshp, maxslope_gshp = compute_knee_metrics(curves_gshp, theta_s, theta_r, suction_grid)

    # Dimensionless sanity check: u50 = alpha * psi50 (should be O(1) and match VG theory)
    print("\n" + "="*80)
    print("STEP 8b: Dimensionless Knee Check (u50 = alpha * psi50)")
    print("="*80)
    m_vec = 1.0 - 1.0 / np.maximum(n_obs, 1.0001)
    u50_theory = (np.power(2.0, 1.0 / m_vec) - 1.0) ** (1.0 / n_obs)
    u50_empirical = alpha_obs_kpa * psi50_gshp

    # Filter finite, positive values
    mask_u = np.isfinite(u50_empirical) & (u50_empirical > 0) & np.isfinite(u50_theory) & (u50_theory > 0)
    u50_emp = u50_empirical[mask_u]
    u50_th = u50_theory[mask_u]

    u50_log_err = np.abs(np.log10(u50_emp) - np.log10(u50_th))
    frac_close = float(np.mean(u50_log_err < 0.05))  # within ~12% in u-space

    def summarize_u(arr, name):
        if len(arr) == 0:
            return {}
        return {
            "median": float(np.median(arr)),
            "q10": float(np.percentile(arr, 10)),
            "q90": float(np.percentile(arr, 90)),
        }

    u50_summary = {
        "u50_empirical": summarize_u(u50_emp, "u50_empirical"),
        "u50_theory": summarize_u(u50_th, "u50_theory"),
        "log10_error": summarize_u(u50_log_err, "log10_error"),
        "fraction_close_log10<0.05": frac_close,
    }

    print(f"  Empirical u50 median (alpha*psi50): {u50_summary.get('u50_empirical', {}).get('median', float('nan')):.3f}")
    print(f"  Theoretical u50 median: {u50_summary.get('u50_theory', {}).get('median', float('nan')):.3f}")
    print(f"  Median |log10 u50_emp - log10 u50_theory|: {u50_summary.get('log10_error', {}).get('median', float('nan')):.4f}")
    print(f"  Fraction with |log10 error| < 0.05 (~12% in u): {frac_close:.3f}")
    
    # Summaries for reporting
    def summarize(arr, name):
        arr_f = arr[np.isfinite(arr) & (arr > 0)]
        if len(arr_f) == 0:
            return {}
        return {
            "median": float(np.median(arr_f)),
            "q10": float(np.percentile(arr_f, 10)),
            "q90": float(np.percentile(arr_f, 90)),
        }
    
    knee_summary = {
        "psi50_pred": summarize(psi50_pred, "psi50_pred"),
        "psi50_gshp": summarize(psi50_gshp, "psi50_gshp"),
        "maxslope_pred": summarize(maxslope_pred, "maxslope_pred"),
        "maxslope_gshp": summarize(maxslope_gshp, "maxslope_gshp"),
    }
    
    # Save results
    print("\n" + "="*80)
    print("STEP 9: Saving Results")
    print("="*80)
    
    results = {
        'n_samples': len(X_gshp),
        'metrics_parameters': metrics_params,
        'metrics_curves': metrics_curves,
        'knee_metrics_summary': knee_summary,
        'u50_check': u50_summary,
        'alpha_pred': alpha_pred.tolist(),
        'n_pred': n_pred.tolist(),
        'alpha_obs': alpha_obs.tolist(),
        'n_obs': n_obs.tolist(),
    }
    
    with open(OUTPUT_DIR / "gshp_validation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✓ Saved: {OUTPUT_DIR / 'gshp_validation_results.json'}")
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"GSHP samples evaluated: {len(X_gshp)}")
    print(f"\nParameter-space metrics (after unit conversion):")
    print(f"  Alpha: R² = {metrics_params['alpha']['r2']:.4f}, RMSE = {metrics_params['alpha']['rmse']:.6f}, MAE = {metrics_params['alpha']['mae']:.6f}")
    print(f"  n    : R² = {metrics_params['n']['r2']:.4f}, RMSE = {metrics_params['n']['rmse']:.4f}, MAE = {metrics_params['n']['mae']:.4f}")

    print(f"\nCurve-space metrics on GSHP (VGParamNet vs reconstructed GSHP θ(ψ)):")
    print(f"  Global: RMSE = {metrics_curves['global']['rmse']:.4f}, MAE = {metrics_curves['global']['mae']:.4f}")
    print(f"  Wet-end (ψ < 10^2 kPa): RMSE = {metrics_curves['wet_end']['rmse']:.4f}, MAE = {metrics_curves['wet_end']['mae']:.4f}")
    print(f"  Dry-end (ψ > 10^4 kPa): RMSE = {metrics_curves['dry_end']['rmse']:.4f}, MAE = {metrics_curves['dry_end']['mae']:.4f}")
    print("\n" + "="*80)
    print("Validation complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
