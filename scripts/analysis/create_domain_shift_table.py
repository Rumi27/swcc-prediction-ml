#!/usr/bin/env python3
"""
Create Table S1: Domain Shift between UNSODA and GSHP

This script generates a supplementary table showing:
- Feature name (16 features)
- GSHP missing %
- UNSODA range (q5-q95)
- GSHP range (q5-q95)
- Additional rows for theta_s, theta_r, theta_s - theta_r (q5/median/q95)

Author: Generated for supplementary material
Date: 2026-02-25
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Configuration
DATA_DIR = ROOT_DIR / "data_pinn_normalized"
GSHP_DATA_PATH = ROOT_DIR.parent / "data" / "GSHP_downloaded" / "WRC_dataset_surya_et_al_2021_final.csv"
OUTPUT_DIR = ROOT_DIR / "paper_figures" / "supplementary"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def load_unsoda_data():
    """Load UNSODA training data."""
    print("Loading UNSODA training data...")
    
    # Load metadata to get feature names
    with open(DATA_DIR / "metadata.json", 'r') as f:
        metadata = json.load(f)
    feature_cols = metadata['feature_cols']
    
    # Load training data
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    
    print(f"  Loaded {len(X_train)} UNSODA training samples")
    print(f"  Features: {len(feature_cols)}")
    
    return X_train, feature_cols


def load_gshp_data():
    """Load GSHP raw data (before QC)."""
    print("\nLoading GSHP raw data...")
    
    # Try multiple encodings
    encodings = ['latin-1', 'cp1252', 'ISO-8859-1', 'utf-8']
    df_gshp = None
    
    for enc in encodings:
        try:
            df_gshp = pd.read_csv(GSHP_DATA_PATH, encoding=enc, low_memory=False)
            print(f"  Successfully loaded with encoding: {enc}")
            break
        except (UnicodeDecodeError, FileNotFoundError) as e:
            continue
    
    if df_gshp is None:
        raise FileNotFoundError(f"Could not load GSHP data from {GSHP_DATA_PATH}")
    
    print(f"  Loaded {len(df_gshp)} GSHP samples (raw, before QC)")
    
    return df_gshp


def map_gshp_to_unsoda_features(df_gshp, feature_cols):
    """Map GSHP columns to UNSODA feature names."""
    print("\nMapping GSHP columns to UNSODA features...")
    
    # Mapping dictionary
    gshp_to_unsoda = {
        'sand_tot_psa': 'sand_pct',
        'silt_tot_psa': 'silt_pct',
        'clay_tot_psa': 'clay_pct',
        'db_od': 'bulk_density',
        'ph_h2o': 'pH',
        'oc': 'OM_content',  # Will convert OC to OM
        'porosity': 'porosity',
    }
    
    # PSD percentiles - these are typically missing in GSHP
    psd_features = ['D10', 'D30', 'D50', 'D60', 'D90', 'Cu', 'Cc']
    
    # Create output dataframe
    X_gshp = pd.DataFrame(index=df_gshp.index)
    
    # Map available features
    for gshp_col, unsoda_col in gshp_to_unsoda.items():
        if gshp_col in df_gshp.columns:
            if gshp_col == 'oc':
                # Convert OC to OM: OM ≈ 1.724 * OC
                X_gshp[unsoda_col] = pd.to_numeric(df_gshp[gshp_col], errors='coerce') * 1.724
            else:
                X_gshp[unsoda_col] = pd.to_numeric(df_gshp[gshp_col], errors='coerce')
            print(f"  ✓ {gshp_col} → {unsoda_col}")
        else:
            print(f"  ✗ {gshp_col} → {unsoda_col} (not available)")
    
    # Handle porosity (compute if missing)
    if 'porosity' not in X_gshp.columns or X_gshp['porosity'].isna().all():
        if 'db_od' in df_gshp.columns:
            X_gshp['porosity'] = 1 - pd.to_numeric(df_gshp['db_od'], errors='coerce') / 2.65
            print(f"  ✓ Computed porosity from bulk density")
    
    # Handle theta_s and theta_r
    if 'thetas' in df_gshp.columns:
        X_gshp['theta_s'] = pd.to_numeric(df_gshp['thetas'], errors='coerce')
    if 'thetar' in df_gshp.columns:
        X_gshp['theta_r'] = pd.to_numeric(df_gshp['thetar'], errors='coerce')
    
    # PSD features - mark as missing (will be imputed)
    for feat in psd_features:
        if feat in feature_cols:
            X_gshp[feat] = np.nan  # Mark as missing
    
    return X_gshp


def compute_statistics(X_unsoda, X_gshp, feature_cols):
    """Compute statistics for domain shift table."""
    print("\nComputing statistics...")
    
    rows = []
    
    # Process each feature
    for feat in feature_cols:
        # UNSODA stats
        unsoda_vals = X_unsoda[feat].values
        unsoda_vals = unsoda_vals[np.isfinite(unsoda_vals)]
        
        if len(unsoda_vals) > 0:
            unsoda_q5 = np.percentile(unsoda_vals, 5)
            unsoda_q95 = np.percentile(unsoda_vals, 95)
            unsoda_range = f"{unsoda_q5:.3f} - {unsoda_q95:.3f}"
        else:
            unsoda_range = "N/A"
        
        # GSHP stats
        if feat in X_gshp.columns:
            gshp_vals = X_gshp[feat].values
            gshp_vals = gshp_vals[np.isfinite(gshp_vals)]
            
            missing_pct = (X_gshp[feat].isna().sum() / len(X_gshp)) * 100
            
            if len(gshp_vals) > 0:
                gshp_q5 = np.percentile(gshp_vals, 5)
                gshp_q95 = np.percentile(gshp_vals, 95)
                gshp_range = f"{gshp_q5:.3f} - {gshp_q95:.3f}"
            else:
                gshp_range = "N/A"
        else:
            missing_pct = 100.0
            gshp_range = "N/A"
        
        rows.append({
            'Feature': feat,
            'GSHP_missing_pct': f"{missing_pct:.1f}",
            'UNSODA_range_q5_q95': unsoda_range,
            'GSHP_range_q5_q95': gshp_range
        })
    
    return rows


def compute_theta_stats(X_unsoda, X_gshp):
    """Compute theta_s, theta_r, and theta_s - theta_r statistics."""
    print("\nComputing theta endpoint statistics...")
    
    theta_stats = []
    
    # UNSODA
    ths_u = X_unsoda['theta_s'].values
    thr_u = X_unsoda['theta_r'].values
    rng_u = ths_u - thr_u
    
    ths_u = ths_u[np.isfinite(ths_u)]
    thr_u = thr_u[np.isfinite(thr_u)]
    rng_u = rng_u[np.isfinite(rng_u)]
    
    # GSHP
    if 'theta_s' in X_gshp.columns and 'theta_r' in X_gshp.columns:
        ths_g = X_gshp['theta_s'].values
        thr_g = X_gshp['theta_r'].values
        rng_g = ths_g - thr_g
        
        ths_g = ths_g[np.isfinite(ths_g)]
        thr_g = thr_g[np.isfinite(thr_g)]
        rng_g = rng_g[np.isfinite(rng_g)]
    else:
        ths_g = np.array([])
        thr_g = np.array([])
        rng_g = np.array([])
    
    def format_stats(vals, name, dataset):
        if len(vals) > 0:
            q5 = np.percentile(vals, 5)
            med = np.median(vals)
            q95 = np.percentile(vals, 95)
            return f"q5={q5:.3f}, median={med:.3f}, q95={q95:.3f}"
        else:
            return "N/A"
    
    # Add rows for theta endpoints
    theta_stats.append({
        'Feature': 'θ_s',
        'GSHP_missing_pct': 'N/A',
        'UNSODA_range_q5_q95': format_stats(ths_u, 'theta_s', 'UNSODA'),
        'GSHP_range_q5_q95': format_stats(ths_g, 'theta_s', 'GSHP')
    })
    
    theta_stats.append({
        'Feature': 'θ_r',
        'GSHP_missing_pct': 'N/A',
        'UNSODA_range_q5_q95': format_stats(thr_u, 'theta_r', 'UNSODA'),
        'GSHP_range_q5_q95': format_stats(thr_g, 'theta_r', 'GSHP')
    })
    
    theta_stats.append({
        'Feature': 'θ_s - θ_r',
        'GSHP_missing_pct': 'N/A',
        'UNSODA_range_q5_q95': format_stats(rng_u, 'theta_range', 'UNSODA'),
        'GSHP_range_q5_q95': format_stats(rng_g, 'theta_range', 'GSHP')
    })
    
    return theta_stats


def create_table_csv(rows, theta_stats):
    """Create CSV table."""
    print("\nCreating CSV table...")
    
    # Combine feature rows and theta stats
    all_rows = rows + theta_stats
    
    df_table = pd.DataFrame(all_rows)
    
    # Save CSV
    output_path = OUTPUT_DIR / "Table_S1_Domain_shift_UNSODA_vs_GSHP.csv"
    df_table.to_csv(output_path, index=False)
    print(f"  ✓ Saved CSV to: {output_path}")
    
    return df_table


def create_table_pdf(df_table):
    """Create PDF table (optional, formatted nicely)."""
    print("\nCreating PDF table...")
    
    fig, ax = plt.subplots(figsize=(14, max(8, len(df_table) * 0.3)))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table_data = []
    for _, row in df_table.iterrows():
        table_data.append([
            row['Feature'],
            row['GSHP_missing_pct'],
            row['UNSODA_range_q5_q95'],
            row['GSHP_range_q5_q95']
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Feature', 'GSHP missing %', 'UNSODA range (q5-q95)', 'GSHP range (q5-q95)'],
        cellLoc='left',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Header row styling
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    # Highlight theta endpoint rows
    n_feature_rows = len(df_table) - 3
    for i in range(n_feature_rows, len(df_table) + 1):
        for j in range(4):
            table[(i, j)].set_facecolor('#fff9c4')
    
    plt.title('Table S1: Domain Shift between UNSODA and GSHP', 
              fontsize=14, fontweight='bold', pad=20)
    
    output_path = OUTPUT_DIR / "Table_S1_Domain_shift_UNSODA_vs_GSHP.pdf"
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"  ✓ Saved PDF to: {output_path}")
    
    plt.close()


def main():
    """Main function."""
    print("="*80)
    print("Creating Table S1: Domain Shift between UNSODA and GSHP")
    print("="*80)
    
    # Load data
    X_unsoda, feature_cols = load_unsoda_data()
    df_gshp = load_gshp_data()
    
    # Map GSHP to UNSODA features
    X_gshp = map_gshp_to_unsoda_features(df_gshp, feature_cols)
    
    # Compute statistics
    rows = compute_statistics(X_unsoda, X_gshp, feature_cols)
    theta_stats = compute_theta_stats(X_unsoda, X_gshp)
    
    # Create tables
    df_table = create_table_csv(rows, theta_stats)
    create_table_pdf(df_table)
    
    # Print summary
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    print(f"Total features analyzed: {len(rows)}")
    print(f"Theta endpoint statistics: {len(theta_stats)} rows")
    print(f"Total table rows: {len(df_table)}")
    print("\n✓ Table S1 created successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
