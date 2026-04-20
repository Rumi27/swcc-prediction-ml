
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Config
DATA_DIR = ROOT_DIR / "data/Hohenbrink_2023/DataFiles_v2"
OUTPUT_DIR = ROOT_DIR / "data/Hohenbrink_2023"
UNSODA_DIR = ROOT_DIR / "data_pinn_normalized"

def process_hohenbrink():
    print("Loading Hohenbrink data...")
    basic_df = pd.read_csv(DATA_DIR / "BasicProp.csv")
    ret_df = pd.read_csv(DATA_DIR / "RetMeas.csv")
    param_df = pd.read_csv(DATA_DIR / "Param.csv")
    
    # 1. Prepare Basic Properties (Inputs)
    # Target columns: sand_pct, silt_pct, clay_pct, bulk_density, porosity, om_pct, ph, theta_s, theta_r
    
    # Merge Basic + Param (for theta_s/theta_r inputs)
    # Param.csv has 'ths_VGM', 'thr_VGM'
    merged_df = basic_df.merge(param_df[['Sample_ID', 'ths_VGM', 'thr_VGM', 'al_VGM', 'n_VGM']], on='Sample_ID', how='inner')
    
    # Map columns
    # Texture: USDA is standard.
    merged_df['sand_pct'] = merged_df['Sand_USDA']
    merged_df['silt_pct'] = merged_df['Silt_USDA']
    merged_df['clay_pct'] = merged_df['Clay_USDA']
    merged_df['bulk_density'] = merged_df['BD']
    merged_df['porosity'] = merged_df['Porosity']
    
    # OM = Corg * 1.724 (standard conversion)
    merged_df['om_pct'] = merged_df['Corg'] * 1.724
    
    # theta_s / theta_r
    merged_df['theta_s'] = merged_df['ths_VGM']
    merged_df['theta_r'] = merged_df['thr_VGM'] # Note: thr is often 0 in simple fits, check values.
    
    # pH is missing. Impute with global median from UNSODA if possible, or just standard value (7.0 or 6.5)
    # Let's check UNSODA X_train if accessible, otherwise assume 6.5.
    try:
        X_train = pd.read_csv(UNSODA_DIR / "X_train.csv")
        median_ph = X_train['ph'].median()
        print(f"Imputing pH with UNSODA median: {median_ph}")
    except:
        median_ph = 7.0
        print("UNSODA not found, using pH = 7.0")
        
    merged_df['ph'] = median_ph
    
    # D-values (d10, d30, d60). 
    # Hohenbrink doesn't give full PSD curves, just fractions.
    # We MUST impute D-values from Textures using the same logic as our fallback in `evaluate_literature_data.py`.
    # Or, effectively, we will re-use the nearest neighbor imputation from `evaluate_literature_data.py` 
    # inside that script, so we just need to provide the base columns here.
    # We will compute columns required for the mapping script.
    
    final_cols = ['Sample_ID', 'sand_pct', 'silt_pct', 'clay_pct', 'bulk_density', 'porosity', 'om_pct', 'ph', 'theta_s', 'theta_r', 'al_VGM', 'n_VGM']
    
    processed_df = merged_df[final_cols].copy()
    
    # 2. Process Measured Curves (Ground Truth)
    # Group RetMeas by Sample_ID
    # pF to Suction (kPa)
    # h_cm = 10^pF
    # psi_kPa = h_cm * 0.0980665
    
    print("Processing measured curves...")
    # Filter only relevant samples
    valid_ids = processed_df['Sample_ID'].unique()
    ret_df = ret_df[ret_df['Sample_ID'].isin(valid_ids)]
    
    ret_df['h_cm'] = 10 ** ret_df['pF']
    ret_df['suction_kpa'] = ret_df['h_cm'] * 0.0980665
    
    # Pivot or consolidate?
    # We want to enable curve comparison plotting. 
    # The evaluation script usually takes a list of curves or generates comparison for specific ones.
    # Let's save the full measured points data as a separate CSV or JSON.
    
    ret_df[['Sample_ID', 'suction_kpa', 'theta']].to_csv(OUTPUT_DIR / "hohenbrink_measured_points.csv", index=False)
    processed_df.to_csv(OUTPUT_DIR / "hohenbrink_inputs.csv", index=False)
    
    print(f"Processed {len(processed_df)} samples.")
    print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    process_hohenbrink()
