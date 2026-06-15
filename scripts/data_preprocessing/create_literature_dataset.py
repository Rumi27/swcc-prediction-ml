
import pandas as pd
import numpy as np
from pathlib import Path

# Carsel and Parrish (1988) Parameters
# Units: alpha (1/cm), Ks (cm/h), theta (cm3/cm3)
# Note: n is dimensionless
cp_data = {
    'Texture': ['Sand', 'Loamy Sand', 'Sandy Loam', 'Loam', 'Silt', 'Silt Loam', 
                'Sandy Clay Loam', 'Clay Loam', 'Silty Clay Loam', 'Sandy Clay', 'Silty Clay', 'Clay'],
    # Centroids (Sand, Silt, Clay) - Approximated from USDA Triangle
    'Sand_pct': [92.0, 82.0, 65.0, 40.0, 5.0, 20.0, 60.0, 32.0, 10.0, 52.0, 6.0, 20.0],
    'Silt_pct': [5.0, 12.0, 25.0, 40.0, 87.0, 65.0, 10.0, 34.0, 55.0, 6.0, 47.0, 20.0],
    'Clay_pct': [3.0, 6.0, 10.0, 20.0, 8.0, 15.0, 30.0, 34.0, 35.0, 42.0, 47.0, 60.0],
    # Hydraulic Parameters (Mean values from Carsel & Parrish 1988)
    'theta_r': [0.045, 0.057, 0.065, 0.078, 0.034, 0.067, 0.100, 0.095, 0.089, 0.100, 0.070, 0.068],
    'theta_s': [0.430, 0.410, 0.410, 0.430, 0.460, 0.450, 0.390, 0.410, 0.430, 0.380, 0.360, 0.380],
    'alpha':   [0.145, 0.124, 0.075, 0.036, 0.016, 0.020, 0.059, 0.019, 0.010, 0.027, 0.005, 0.008],
    'n':       [2.68, 2.28, 1.89, 1.56, 1.37, 1.41, 1.48, 1.31, 1.23, 1.23, 1.09, 1.09],
    'Ks':      [29.70, 14.59, 4.42, 1.04, 0.25, 0.45, 1.31, 0.26, 0.07, 0.12, 0.02, 0.20], # cm/h
    # Estimated Bulk Density (g/cm3) - Not in C&P, estimated based on typical texture porosity
    # BD approx (1 - theta_s) * 2.65
    'BulkDensity': [1.51, 1.56, 1.56, 1.51, 1.43, 1.46, 1.62, 1.56, 1.51, 1.64, 1.70, 1.64] 
}

def create_dataset():
    df = pd.DataFrame(cp_data)
    
    # Calculate derived features used in our model
    # D10, D30, D60 etc are needed for the ML model.
    # Since we only have Texture %, we need to Estimate D-values or use a transfer function.
    # OR, we can impute them based on the training set's average for that texture class.
    # Strategy: Leave them NaN for now and let the preprocessing pipeline handle imputation if robust?
    # Better: Use the 'Texture' class to look up average D-values from our UNSODA training set.
    
    # For now, let's just save the raw literature data. 
    # The evaluation script will need to map these to model inputs.
    
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "external_literature_data.csv"
    
    df.to_csv(output_path, index=False)
    print(f"Created literature dataset at {output_path}")
    print(df.head())

if __name__ == "__main__":
    create_dataset()
