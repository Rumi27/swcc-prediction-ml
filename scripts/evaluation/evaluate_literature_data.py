
import pandas as pd
print("Script started!")
import sys
print("Imports starting...")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import joblib

# Add project root to path
import sys
try:
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
except NameError:
    ROOT_DIR = Path.cwd() # Fallback for exec() execution
    
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import tensorflow as tf
from models.vg_param_net import VGParamNet, vg_theta

# Configuration
LIT_DATA_PATH = ROOT_DIR / "data/external_literature_data.csv"
RESULTS_DIR = ROOT_DIR / "results_pinn_fixed/vgparamnet" 
MODEL_PATH = RESULTS_DIR / "vgparamnet_best.keras"
OUTPUT_DIR = ROOT_DIR / "results_external_validation"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load metadata to get feature scaler statistics for filling missing values?
# Better: Load X_train and compute average D-values per texture class.

def get_imputation_map(X_train):
    """
    Compute average PSD metrics per texture class from UNSODA training data.
    We need to map 'Sand', 'Loam', etc to the feature columns.
    Assumes X_train has 'sand_pct', 'clay_pct' etc.
    We need to classify X_train rows into USDA textures first.
    """
    # Simple classification based on Sand/Clay/Silt (USDA logic)
    # Or just use the entire dataset average as a fallback if too complex? 
    # Let's try to be specific.
    
    # Actually, simpler approach for this script:
    # Just use the global means for D10, D30, D60, and derive Cu, Cc.
    # Texture is the main driver. D-values are correlated.
    # Even better: Use a lookup table if we can. 
    # Let's use specific "Typical" D-values for Sand, Loam, Clay from a textbook or just global mean.
    # Given the constraint, Global Mean with scalar adjustment based on Sand% is a decent heuristic.
    # D50 ~ exp( -0.05 * clay_pct - 0.02 * silt_pct ) * scale?
    
    # Alternative: KNN Imputation.
    # Find nearest neighbor in X_train based on Sand/Silt/Clay/BD 
    # and copy their D-values. This is the most robust method.
    
    from sklearn.neighbors import NearestNeighbors
    
    # Features to match on: Sand, Silt, Clay, BD
    match_cols = ['sand_pct', 'silt_pct', 'clay_pct', 'bulk_density']
    target_cols = ['d10', 'd30', 'd50', 'd60', 'd90', 'coco', 'cu', 'om_pct', 'ph', 'theta_s', 'theta_r'] 
    # Note: theta_s/theta_r are inputs to our model in some versions? 
    # Check feature_cols in metadata.json.
    
    metadata = json.load(open(DATA_DIR / "metadata.json"))
    feature_cols = metadata["feature_cols"]
    
    print("Feature columns expected:", feature_cols)
    
    # Prepare Reference Data
    ref_df = X_train
    
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(ref_df[match_cols].fillna(0))
    
    return nn, ref_df, feature_cols

def evaluate():
    print("Starting evaluate()...")
    print("Loading data...")
    lit_df = pd.read_csv(LIT_DATA_PATH)
    
    # Load Training Data for Imputation
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    
    # Normalize naming
    # Literature csv: Sand_pct, Silt_pct, Clay_pct, BulkDensity
    # X_train: sand_pct, silt_pct, clay_pct, bulk_density
    
    lit_df_mapped = pd.DataFrame()
    lit_df_mapped['sand_pct'] = lit_df['Sand_pct']
    lit_df_mapped['silt_pct'] = lit_df['Silt_pct']
    lit_df_mapped['clay_pct'] = lit_df['Clay_pct']
    lit_df_mapped['bulk_density'] = lit_df['BulkDensity']
    
    # Impute missing features
    print("Imputing missing features...")
    nn, ref_df, feature_cols = get_imputation_map(X_train)
    
    print("Running NearestNeighbors...")
    distances, indices = nn.kneighbors(lit_df_mapped)
    
    # Construct input dataframe
    input_df = pd.DataFrame(index=lit_df.index, columns=feature_cols)
    
    for i, ref_idx in enumerate(indices.flatten()):
        # Copy all features from nearest neighbor
        input_df.iloc[i] = ref_df.iloc[ref_idx][feature_cols]
        
        # Overwrite with known literature values where available
        # PSD % and BD
        input_df.iloc[i]['sand_pct'] = lit_df_mapped.iloc[i]['sand_pct']
        input_df.iloc[i]['silt_pct'] = lit_df_mapped.iloc[i]['silt_pct']
        input_df.iloc[i]['clay_pct'] = lit_df_mapped.iloc[i]['clay_pct']
        input_df.iloc[i]['bulk_density'] = lit_df_mapped.iloc[i]['bulk_density']
        
        # Calculate Porosity if it exists in feature_cols
        if 'porosity' in feature_cols:
             input_df.iloc[i]['porosity'] = 1 - (lit_df_mapped.iloc[i]['bulk_density'] / 2.65)
             
        # Also, VGParamNet uses theta_s/theta_r as inputs? 
        # Usually they are inputs if available, or predicted?
        # In this project, theta_s/theta_r were listed as INPUTS in preprocessing.
        # Literature data HAS theta_s/theta_r. Let's use them!
        if 'theta_s' in feature_cols:
            input_df.iloc[i]['theta_s'] = lit_df.iloc[i]['theta_s']
        if 'theta_r' in feature_cols:
            input_df.iloc[i]['theta_r'] = lit_df.iloc[i]['theta_r']
            
    print("Imputed Input Data (Head):")
    print(input_df.head())
    
    print("Imputed Input Data (Head):")
    print(input_df.head())
    
    # Scale Data? NO. Model trained on raw X_train.csv (confirmed via inspection).
    X_scaled = input_df.values.astype(np.float32)
    
    print("Imputed Input Data (Head):")
    print(input_df.head())
    
    # Scale Data? NO. Model trained on raw X_train.csv (confirmed via inspection).
    X_scaled = input_df.values.astype(np.float32)
    
    print("Imputed Input Data (Head):")
    print(input_df.head())
    
    # Scale Data? NO. Model trained on raw X_train.csv (confirmed via inspection).
    X_lit = input_df.values.astype(np.float32)
    print("Data loaded and imputed.")
    
    # Run Predictions
    # Load VGParamNet Model
    print("Loading VGParamNet model...")
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"VGParamNet": VGParamNet},
            compile=False
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Check input dimension
    # metadata = json.load(open(DATA_DIR / "metadata.json")) # Already loaded inside get_imputation_map if needed, but we need it here
    # Just check shape of input_df
    print(f"Input shape: {input_df.shape}")
    
    # Predict Alpha, n
    X_test = input_df.values.astype(np.float32)
    alpha_pred, n_pred = model(X_test, training=False)
    
    # Save parameters
    np.save(OUTPUT_DIR / "literature_predictions_alpha.npy", alpha_pred.numpy())
    np.save(OUTPUT_DIR / "literature_predictions_n.npy", n_pred.numpy())
    
    # Generate Plots
    generate_comparison_plots(lit_df, alpha_pred.numpy(), n_pred.numpy(), input_df)

def generate_comparison_plots(df, alpha_pred, n_pred, input_df):
    try:
        plt.style.use('seaborn-v0_8-paper')
    except OSError:
        plt.style.use('seaborn-paper')

    # Load Suction Grid
    # We need a suction grid for plotting. 
    # The training used a specific grid, but for plotting we can use a generated log-space grid.
    # Let's generate a dense grid for smooth curves.
    h_plot = np.logspace(-1, 5, 100) # 0.1 to 100,000 cm (approx 0.01 to 10,000 kPa)
    # Note: Model trained on kPa. Literature often in cm. 
    # Check units!
    # Codebase generally seems to use kPa for VGParamNet (psi in kPa).
    # Literature data: "h" usually cm.
    # Let's assume we plot in cm (since typical SWCC plots are pF or Log h cm).
    # VGParamNet inputs: Features. 
    # VGParamNet outputs: alpha (1/kPa), n.
    # We need to be careful with units in the plot.
    
    # 2. Curve Comparison for 3 key textures
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    textures = ['Sand', 'Loam', 'Clay']
    
    for i, tex in enumerate(textures):
        # Find row index
        rows = df[df['Texture'] == tex]
        if rows.empty:
            continue
        idx = rows.index[0]
        row = rows.iloc[0]
        ax = axes[i]
        
        # Lit Curve (Analytical)
        # Check units of Literature parameters.
        # Usually C&P 1988 gives alpha in 1/cm (?) or 1/kPa?
        # Standard Carsel & Parrish is 1/cm.
        alpha_lit = row['alpha'] # Assuming 1/cm
        n_lit = row['n']
        theta_r_lit = row['theta_r']
        theta_s_lit = row['theta_s']
        
        # Predicted Parameters (VGParamNet gives alpha in 1/kPa)
        alpha_pred_i = alpha_pred[idx] # 1/kPa
        n_pred_i = n_pred[idx]
        
        # For this plot, use Literature theta_s/theta_r for the prediction curve too?
        # VGParamNet assumes theta_s/theta_r are inputs or known.
        # In our input_df, we used literature theta_s/theta_r.
        theta_s_input = input_df.iloc[idx]['theta_s']
        theta_r_input = input_df.iloc[idx]['theta_r']
        
        # Plotting Domain: Suction Head h (cm)
        # h_plot is in cm.
        
        # Lit Curve (alpha in 1/cm)
        m_lit = 1 - 1/n_lit
        theta_lit = theta_r_lit + (theta_s_lit - theta_r_lit) * \
                    (1 + (alpha_lit * h_plot)**n_lit)**(-m_lit)
        
        # Predicted Curve
        # Convert h_plot (cm) to psi (kPa) for Model Parameter usage
        # 100 cm H2O approx 9.81 kPa. 1 cm approx 0.0981 kPa.
        psi_kPa = h_plot * 0.0980665
        
        # Calculate Theta using Predicted Params (alpha in 1/kPa)
        m_pred = 1 - 1/n_pred_i
        # VG Formula: Se = [1 + (alpha * psi)^n]^-m
        # (alpha * psi) is dimensionless. (1/kPa * kPa). Correct.
        theta_pred_val = theta_r_input + (theta_s_input - theta_r_input) * \
                         (1 + (alpha_pred_i * psi_kPa)**n_pred_i)**(-m_pred)
        
        ax.semilogx(h_plot, theta_lit, 'k--', label=f'Lit (C&P 1988)', linewidth=2)
        ax.semilogx(h_plot, theta_pred_val, 'b-', label='VGParamNet', linewidth=2)
        
        ax.set_xlabel('Suction Head (cm)')
        ax.set_ylabel('Theta')
        ax.set_title(tex)
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "curve_comparison_lit_vgparamnet.png")
    plt.close()
    print("Plots generated in results_external_validation/curve_comparison_lit_vgparamnet.png")
    
if __name__ == "__main__":
    evaluate()
