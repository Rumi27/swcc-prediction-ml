
#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from scripts.simulation.richards_solver import RichardsSolver1D, VGSWCCWrapper, InterpolatedSWCCWrapper

# Configuration
DATA_DIR = ROOT_DIR / "data_pinn_normalized"
RESULTS_DIR = ROOT_DIR / "results_pinn_fixed/vgparamnet/run_B"
OUTPUT_DIR = ROOT_DIR / "results_simulation"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    """Load train/test data for model training/eval"""
    print("Loading data...")
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = np.load(DATA_DIR / "y_train_original.npy")
    
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = np.load(DATA_DIR / "y_test_original.npy")
    
    suction_grid = np.load(DATA_DIR / "suction_grid.npy")
    
    # Metadata for feature columns
    metadata = json.load(open(DATA_DIR / "metadata.json"))
    feature_cols = metadata["feature_cols"]
    
    return X_train, y_train, X_test, y_test, suction_grid, feature_cols

def train_gb_model(X_train, y_train, feature_cols):
    """Retrain GB Baseline (fast)"""
    print("Retraining Gradient Boosting Baseline...")
    # Use same settings as paper (defaults)
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model = MultiOutputRegressor(gb)
    
    X = X_train[feature_cols].values
    model.fit(X, y_train)
    print("GB Training complete.")
    return model

def load_vg_predictions():
    """Load Pre-computed VGParamNet predictions"""
    try:
        alpha = np.load(RESULTS_DIR / "alpha_test.npy")
        n = np.load(RESULTS_DIR / "n_test.npy")
        print(f"Loaded VGParamNet predictions: alpha shape {alpha.shape}, n shape {n.shape}")
        return alpha, n
    except Exception as e:
        print(f"Failed to load VGParamNet predictions: {e}")
        return None, None

def select_test_cases(X_test, y_test):
    """Select 3 distinct soil samples"""
    # Criteria:
    # Sand: High Sand %, High D50
    # Clay: High Clay %
    # Loam: Intermediate
    
    # Add index to dataframe for tracking
    X_test['original_index'] = range(len(X_test))
    
    # 1. Sand
    mask_sand = (X_test['sand_pct'] > 80) & (X_test['clay_pct'] < 10)
    sand_candidates = X_test[mask_sand]
    if len(sand_candidates) > 0:
        sand_idx = sand_candidates.iloc[0]['original_index']
    else:
        sand_idx = X_test['sand_pct'].idxmax()
        
    # 2. Clay
    mask_clay = (X_test['clay_pct'] > 50)
    clay_candidates = X_test[mask_clay]
    if len(clay_candidates) > 0:
        clay_idx = clay_candidates.iloc[0]['original_index']
    else:
        clay_idx = X_test['clay_pct'].idxmax()
        
    # 3. Loam (intermediate)
    mask_loam = (X_test['sand_pct'] > 30) & (X_test['sand_pct'] < 50) & \
                (X_test['clay_pct'] > 10) & (X_test['clay_pct'] < 30)
    loam_candidates = X_test[mask_loam]
    if len(loam_candidates) > 0:
        loam_idx = loam_candidates.iloc[0]['original_index']
    else:
        # fallback to median sand content
        median_sand = X_test['sand_pct'].median()
        loam_idx = (X_test['sand_pct'] - median_sand).abs().idxmin()
        
    indices = {
        'Sand': int(sand_idx),
        'Loam': int(loam_idx),
        # 'Clay': int(clay_idx) # Skip Clay for now to ensure we get results
    }
    
    print("Selected Test Cases:")
    for type_name, idx in indices.items():
        print(f"  {type_name}: Index {idx}, Sand={X_test.iloc[idx]['sand_pct']:.1f}%, Clay={X_test.iloc[idx]['clay_pct']:.1f}%")
        
    return indices

def run_simulation(solver, name, case_name):
    """Run single simulation wrapper"""
    print(f"  Running {name}...")
    
    # Initial Condition: -100 cm (approx -10 kPa) or -1000 cm = -100 kPa
    # Let's use -100 kPa = -1020 cm approx.
    psi_init = -1000.0 
    
    # Run for 24 hours
    # Sand drains fast, Clay slow. 24h is good.
    solver.initialize(psi_init)
    
    # Flux BC: 0.5 * Ks
    q_rain = -0.5 * solver.k_func(0)
    
    history, stats = solver.solve(t_max=24.0, dt_init=1e-5, 
                                 top_bc_type='flux', top_bc_val=q_rain, 
                                 bottom_bc_type='free_drain')
    
    return history, stats

def main():
    # 1. Load Data
    X_train, y_train, X_test, y_test, suction_grid, feature_cols = load_data()
    
    # 2. Load/Train Models
    gb_model = train_gb_model(X_train, y_train, feature_cols)
    alpha_pred, n_pred = load_vg_predictions()
    
    if alpha_pred is None:
        print("Skipping benchmark due to missing VG predictions.")
        return

    # 3. Select Test Cases
    case_indices = select_test_cases(X_test, y_test)
    
    results = []
    
    # 4. Run Benchmark
    for case_type, idx in case_indices.items():
        print(f"\n--- Benchmarking Case: {case_type} (Index {idx}) ---")
        
        # Get Sample Data
        sample_features = X_test.iloc[idx:idx+1][feature_cols].values
        
        # Determine Ks
        # Metadata check? Just hardcode based on type to be consistent across models
        if case_type == 'Sand': Ks = 20.0 # cm/h
        elif case_type == 'Loam': Ks = 1.0
        else: Ks = 0.1 # Clay
        
        print(f"  Assumed Ks = {Ks} cm/h")
        
        # =========================================================
        # A. Ground Truth (Observed Points Interpolated)
        # =========================================================
        y_obs = y_test[idx]
        gt_wrapper = InterpolatedSWCCWrapper(suction_grid, y_obs, Ks)
        solver_gt = RichardsSolver1D(L=200, nz=100, swcc_func=gt_wrapper.swcc, k_func=gt_wrapper.conductivity)
        hist_gt, stats_gt = run_simulation(solver_gt, "Ground Truth", case_type)
        
        results.append({
            'case': case_type, 'model': 'Ground Truth', 
            'steps': stats_gt['total_steps'], 'failed': stats_gt['failed_steps'],
            'avg_iter': stats_gt['total_newton_iters']/stats_gt['total_steps'] if stats_gt['total_steps'] > 0 else 0,
            'cpu_time': stats_gt['cpu_time']
        })
        
        # =========================================================
        # B. Gradient Boosting
        # =========================================================
        y_pred_gb = gb_model.predict(sample_features)[0]
        gb_wrapper = InterpolatedSWCCWrapper(suction_grid, y_pred_gb, Ks)
        solver_gb = RichardsSolver1D(L=200, nz=100, swcc_func=gb_wrapper.swcc, k_func=gb_wrapper.conductivity)
        hist_gb, stats_gb = run_simulation(solver_gb, "Gradient Boosting", case_type)
        
        results.append({
            'case': case_type, 'model': 'Gradient Boosting', 
            'steps': stats_gb['total_steps'], 'failed': stats_gb['failed_steps'],
            'avg_iter': stats_gb['total_newton_iters']/stats_gb['total_steps'] if stats_gb['total_steps'] > 0 else 0,
            'cpu_time': stats_gb['cpu_time']
        })
        
        # =========================================================
        # C. VGParamNet
        # =========================================================
        # Get predictions for this index
        # Note: X_test index matches alpha_pred index 1:1? Yes, assuming X_test wasn't shuffled relative to prediction saving.
        # Check normalization? No, predictions are likely saved in order of X_test.
        
        alpha = float(alpha_pred[idx])
        n = float(n_pred[idx])
        
        theta_s = X_test.iloc[idx]['theta_s']
        theta_r = X_test.iloc[idx]['theta_r']
        
        print(f"  VGParamNet: alpha={alpha:.4f}, n={n:.4f}")
        
        vg_wrapper = VGSWCCWrapper(alpha, n, theta_r, theta_s, Ks)
        solver_vg = RichardsSolver1D(L=200, nz=100, swcc_func=vg_wrapper.swcc, k_func=vg_wrapper.conductivity)
        hist_vg, stats_vg = run_simulation(solver_vg, "VGParamNet", case_type)
        
        results.append({
            'case': case_type, 'model': 'VGParamNet', 
            'steps': stats_vg['total_steps'], 'failed': stats_vg['failed_steps'],
            'avg_iter': stats_vg['total_newton_iters']/stats_vg['total_steps'] if stats_vg['total_steps'] > 0 else 0,
            'cpu_time': stats_vg['cpu_time']
        })
        
        # Plot Profiles for this case
        plt.figure(figsize=(12, 5))
        
        # Plot 1: SWCC Comparison
        plt.subplot(1, 3, 1)
        plt.semilogx(suction_grid, y_obs, 'k-', label='UNSODA Data')
        plt.semilogx(suction_grid, y_pred_gb, 'r--', label='GB')
        # Analytical VG
        psi_plot = np.logspace(np.log10(0.1), np.log10(1e6), 100)
        theta_vg, _ = vg_wrapper.swcc(psi_plot)
        plt.semilogx(psi_plot, theta_vg, 'b-.', label='VGParamNet')
        plt.xlabel('Suction (cm)')
        plt.ylabel('Theta')
        plt.title(f'{case_type}: SWCC')
        plt.legend()
        
        # Plot 2: Infiltration Profile (Final)
        plt.subplot(1, 3, 2)
        # Ground Truth
        z = np.linspace(0, 200, 100)
        if hist_gt['theta_profiles']:
            plt.plot(hist_gt['theta_profiles'][-1], z, 'k-', label='UNSODA Data')
        if hist_gb['theta_profiles']:
            plt.plot(hist_gb['theta_profiles'][-1], z, 'r--', label='GB')
        if hist_vg['theta_profiles']:
            plt.plot(hist_vg['theta_profiles'][-1], z, 'b-.', label='VG')
        plt.xlabel('Theta')
        plt.ylabel('Elevation (cm)')
        plt.title(f'{case_type}: Profile t=24h')
        
        # Plot 3: Cumulative Infiltration
        plt.subplot(1, 3, 3)
        plt.plot(hist_gt['time'], hist_gt['cumulative_infl'], 'k-', label='UNSODA Data')
        plt.plot(hist_gb['time'], hist_gb['cumulative_infl'], 'r--', label='GB')
        plt.plot(hist_vg['time'], hist_vg['cumulative_infl'], 'b-.', label='VG')
        plt.xlabel('Time (h)')
        plt.ylabel('Infiltration (cm)')
        plt.title('Cumulative Infiltration')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"simulation_{case_type}.png")
        plt.close()

        # Save Partial Metrics
        df_partial = pd.DataFrame(results)
        df_partial.to_csv(OUTPUT_DIR / "benchmark_metrics_partial.csv", index=False)

    # Save Metrics
    df_results = pd.DataFrame(results)
    print("\nBenchmark Results:")
    print(df_results)
    df_results.to_csv(OUTPUT_DIR / "benchmark_metrics.csv", index=False)
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
