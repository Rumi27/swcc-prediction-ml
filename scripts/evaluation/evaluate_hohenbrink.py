
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import joblib
import sys

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Configuration
DATA_DIR = ROOT_DIR / "data_pinn_normalized"
HOHENBRINK_DIR = ROOT_DIR / "data/Hohenbrink_2023"
INPUT_DATA_PATH = HOHENBRINK_DIR / "hohenbrink_inputs.csv"
MEASURED_POINTS_PATH = HOHENBRINK_DIR / "hohenbrink_measured_points.csv"
OUTPUT_DIR = ROOT_DIR / "results_external_validation_hohenbrink"
OUTPUT_DIR.mkdir(exist_ok=True)

def get_imputation_map(X_train):
    from sklearn.neighbors import NearestNeighbors
    
    # Features to match on: Sand, Silt, Clay, BD
    match_cols = ['sand_pct', 'silt_pct', 'clay_pct', 'bulk_density']
    
    metadata = json.load(open(DATA_DIR / "metadata.json"))
    feature_cols = metadata["feature_cols"]
    
    # Prepare Reference Data
    ref_df = X_train
    
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(ref_df[match_cols].fillna(0))
    
    return nn, ref_df, feature_cols

def evaluate():
    print("Loading Hohenbrink data...")
    input_df_raw = pd.read_csv(INPUT_DATA_PATH)
    measured_points = pd.read_csv(MEASURED_POINTS_PATH)
    
    # Load Training Data for Imputation
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train_original = np.load(DATA_DIR / "y_train_original.npy")
    
    # Impute missing features (D-values etc)
    print("Imputing missing features (D-values)...")
    nn, ref_df, feature_cols = get_imputation_map(X_train)
    
    # map columns for matching
    match_df = pd.DataFrame()
    match_df['sand_pct'] = input_df_raw['sand_pct']
    match_df['silt_pct'] = input_df_raw['silt_pct']
    match_df['clay_pct'] = input_df_raw['clay_pct']
    match_df['bulk_density'] = input_df_raw['bulk_density']
    
    distances, indices = nn.kneighbors(match_df.fillna(0))
    
    # Construct final input dataframe
    input_df = pd.DataFrame(index=input_df_raw.index, columns=feature_cols)
    
    for i, ref_idx in enumerate(indices.flatten()):
        input_df.iloc[i] = ref_df.iloc[ref_idx][feature_cols]
        cols_to_overwrite = ['sand_pct', 'silt_pct', 'clay_pct', 'bulk_density', 'porosity', 'om_pct', 'ph', 'theta_s', 'theta_r']
        for col in cols_to_overwrite:
            if col in feature_cols and col in input_df_raw.columns:
                 input_df.iloc[i][col] = input_df_raw.iloc[i][col]

    print("Imputed Input Data (Head):")
    print(input_df.head())
    
    # Run Predictions (Gradient Boosting)
    print("Training Gradient Boosting Baseline on X_train...")
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor
    
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_gb = MultiOutputRegressor(gb)
    
    model_gb.fit(X_train[feature_cols].values, y_train_original)
    print("GB Training complete.")
    
    print("Predicting on Hohenbrink data...")
    input_df = input_df.fillna(input_df.mean())
    y_pred_gb = model_gb.predict(input_df.values)
    
    np.save(OUTPUT_DIR / "hohenbrink_predictions_gb.npy", y_pred_gb)
    
    # Generate Plots
    print("Generating comparison plots...")
    generate_plots(input_df_raw, measured_points, y_pred_gb)

def generate_plots(input_df_raw, measured_points, y_pred_gb):
    try:
        plt.style.use('seaborn-v0_8-paper')
    except OSError:
        plt.style.use('seaborn-paper')
        
    suction_grid = np.load(DATA_DIR / "suction_grid.npy") # kPa
    
    sample_ids = input_df_raw['Sample_ID'].unique()
    np.random.seed(42)
    selected_samples = np.random.choice(sample_ids, 9, replace=False)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, sample_id in enumerate(selected_samples):
        ax = axes[i]
        
        # Get Measured Points
        sample_points = measured_points[measured_points['Sample_ID'] == sample_id]
        
        # Get Predicted Curve
        idx = input_df_raw[input_df_raw['Sample_ID'] == sample_id].index[0]
        theta_pred = y_pred_gb[idx]
        
        # Plot
        # Measured
        ax.semilogx(sample_points['suction_kpa'], sample_points['theta'], 'ko', label='Measured (Hohenbrink)', markersize=8, alpha=0.7)
        
        # Predicted
        ax.semilogx(suction_grid, theta_pred, 'r-', label='GB Prediction', linewidth=2)
        
        # Info
        row = input_df_raw.iloc[idx]
        title = f"ID: {sample_id}\nSand: {row['sand_pct']:.1f}%, Clay: {row['clay_pct']:.1f}%"
        ax.set_title(title)
        ax.set_xlabel('Suction (kPa)')
        ax.set_ylabel('Theta')
        ax.grid(True, which="both", alpha=0.3)
        
        if i == 0:
            ax.legend()
            
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hohenbrink_gb_comparison.png")
    plt.close()
    
    # Calculate RMSE
    rmse_list = []
    from scipy.interpolate import interp1d
    
    for idx, row in input_df_raw.iterrows():
        sample_id = row['Sample_ID']
        sample_points = measured_points[measured_points['Sample_ID'] == sample_id]
        
        if len(sample_points) == 0:
            continue
            
        pred_curve = y_pred_gb[idx]
        f_interp = interp1d(np.log10(suction_grid), pred_curve, kind='linear', fill_value="extrapolate")
        
        valid_points = sample_points[sample_points['suction_kpa'] > 0]
        if len(valid_points) == 0:
            continue
            
        pred_at_points = f_interp(np.log10(valid_points['suction_kpa']))
        mse = np.mean((pred_at_points - valid_points['theta'])**2)
        rmse_list.append(np.sqrt(mse))
        
    mean_rmse = np.mean(rmse_list)
    print(f"Global RMSE on Hohenbrink Data: {mean_rmse:.4f}")
    
    with open(OUTPUT_DIR / "metrics.txt", "w") as f:
        f.write(f"Global RMSE: {mean_rmse:.4f}\n")
        f.write(f"Number of samples evaluated: {len(rmse_list)}\n")

if __name__ == "__main__":
    evaluate()
