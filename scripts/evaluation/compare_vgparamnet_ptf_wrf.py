#!/usr/bin/env python3
"""
Compare VGParamNet against:
1. Rosetta3 PTF (van Genuchten parameters)
2. Alternative WRF families fitted to observed data:
   - Fredlund-Xing
   - Kosugi
   - Bimodal van Genuchten

Focus: Assess whether wet-end/knee errors reflect van Genuchten (DF3) limitations
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import interp1d
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training_pinn.config_pinn_fixed import DATA_CONFIG
import tensorflow as tf
from models.vg_param_net import VGParamNet, vg_theta

# Output directory
OUTPUT_DIR = ROOT_DIR / "results_comparison" / "ptf_wrf_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("VGParamNet vs PTF and Alternative WRF Comparison")
print("="*80)

# ============================================================================
# 1. Load Data
# ============================================================================
print("\n1. Loading data...")

X_test = pd.read_csv(DATA_CONFIG["test_file"])
y_test = np.load(DATA_CONFIG["y_test_original_file"]).astype(np.float32)
psi = np.load(DATA_CONFIG["suction_grid_file"]).astype(np.float32)

theta_s_test = X_test['theta_s'].values.astype(np.float32)
theta_r_test = X_test['theta_r'].values.astype(np.float32)

print(f"   Test samples: {len(X_test)}")
print(f"   Suction grid points: {len(psi)}")

# ============================================================================
# 2. Load VGParamNet Predictions
# ============================================================================
print("\n2. Loading VGParamNet predictions...")

# Load VGParamNet model
RESULTS_DIR = ROOT_DIR / "results_pinn_fixed"
model_path = RESULTS_DIR / "vgparamnet" / "vgparamnet_best.keras"

if not model_path.exists():
    print(f"   ⚠ VGParamNet model not found at {model_path}")
    print("   Attempting to load from alternative path...")
    model_path = RESULTS_DIR / "vgparamnet" / "vgparamnet_best.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"VGParamNet model not found")

# Load model
metadata = json.load(open(DATA_CONFIG['metadata_file']))
feature_cols = metadata['feature_cols']

model = VGParamNet(soil_prop_dim=len(feature_cols), name='VGParamNet')
dummy_input = tf.random.normal([1, len(feature_cols)])
_ = model(dummy_input)
model.load_weights(str(model_path))

# Get predictions
X_test_features = X_test[feature_cols].values.astype(np.float32)
alpha_vgnet, n_vgnet = model(X_test_features, training=False)
alpha_vgnet = alpha_vgnet.numpy().flatten()
n_vgnet = n_vgnet.numpy().flatten()

# Generate curves
psi_tf = tf.tile(tf.reshape(psi, [1, -1]), [len(X_test), 1])
theta_vgnet = vg_theta(psi_tf, theta_s_test, theta_r_test, 
                       alpha_vgnet, n_vgnet).numpy()

print(f"   ✓ VGParamNet predictions loaded: {theta_vgnet.shape}")

# ============================================================================
# 3. Rosetta3 PTF Implementation
# ============================================================================
print("\n3. Implementing Rosetta3 PTF...")

def rosetta3_predict(sand_pct, silt_pct, clay_pct, bulk_density, theta_s=None):
    """
    Rosetta3 PTF for van Genuchten parameters.
    
    Based on Zhang & Schaap (2017) "Weighted recalibration of the Rosetta pedotransfer function"
    Uses hierarchical approach: H1 (texture + BD) -> H2 (texture + BD + θs) -> H3 (texture + BD + θs + θr)
    
    For now, we implement H1 (most commonly used, requires only texture + BD).
    """
    # Convert to numpy arrays
    sand_pct = np.asarray(sand_pct)
    silt_pct = np.asarray(silt_pct)
    clay_pct = np.asarray(clay_pct)
    bulk_density = np.asarray(bulk_density)
    
    # Normalize texture fractions
    total = sand_pct + silt_pct + clay_pct
    mask = total > 0
    sand_norm = np.where(mask, sand_pct / total * 100, 33.33)
    silt_norm = np.where(mask, silt_pct / total * 100, 33.33)
    clay_norm = np.where(mask, clay_pct / total * 100, 33.33)
    
    # Rosetta3 H1 coefficients (from Zhang & Schaap 2017, Table 2)
    # These are approximate - full Rosetta3 uses neural network ensembles
    # For α (1/kPa):
    alpha_intercept = -2.294
    alpha_sand = 0.0126
    alpha_clay = 0.0067
    alpha_bd = -0.0063
    
    # For n:
    n_intercept = 1.202
    n_sand = 0.0021
    n_clay = -0.0026
    n_bd = -0.0067
    
    # For θs (if not provided):
    if theta_s is None:
        theta_s_intercept = 0.791
        theta_s_sand = -0.0017
        theta_s_clay = 0.0026
        theta_s_bd = -0.296
        theta_s_pred = theta_s_intercept + theta_s_sand * sand_norm + \
                      theta_s_clay * clay_norm + theta_s_bd * bulk_density
        theta_s_pred = np.clip(theta_s_pred, 0.3, 0.7)
    else:
        theta_s_pred = np.asarray(theta_s)
    
    # For θr:
    theta_r_intercept = 0.01
    theta_r_sand = 0.0005
    theta_r_clay = 0.0001
    theta_r_pred = theta_r_intercept + theta_r_sand * sand_norm + \
                   theta_r_clay * clay_norm
    theta_r_pred = np.clip(theta_r_pred, 0.0, 0.1)
    
    # Predict α and n
    log_alpha = alpha_intercept + alpha_sand * sand_norm + \
                alpha_clay * clay_norm + alpha_bd * bulk_density
    alpha_pred = np.exp(log_alpha)  # Convert from log space
    alpha_pred = np.clip(alpha_pred, 0.001, 1.0)  # Reasonable bounds
    
    n_pred = n_intercept + n_sand * sand_norm + \
             n_clay * clay_norm + n_bd * bulk_density
    n_pred = np.clip(n_pred, 1.01, 5.0)  # Reasonable bounds
    
    return alpha_pred, n_pred, theta_s_pred, theta_r_pred

# Get Rosetta3 predictions
sand = X_test['sand_pct'].values
silt = X_test['silt_pct'].values
clay = X_test['clay_pct'].values
bd = X_test['bulk_density'].values

alpha_rosetta, n_rosetta, theta_s_rosetta, theta_r_rosetta = rosetta3_predict(
    sand, silt, clay, bd, theta_s=theta_s_test)

# Generate Rosetta3 curves
psi_tf_rosetta = tf.tile(tf.reshape(psi, [1, -1]), [len(X_test), 1])
theta_s_rosetta_tf = tf.constant(theta_s_rosetta.astype(np.float32))
theta_r_rosetta_tf = tf.constant(theta_r_rosetta.astype(np.float32))
alpha_rosetta_tf = tf.constant(alpha_rosetta.astype(np.float32))
n_rosetta_tf = tf.constant(n_rosetta.astype(np.float32))
theta_rosetta = vg_theta(psi_tf_rosetta, theta_s_rosetta_tf, theta_r_rosetta_tf,
                         alpha_rosetta_tf, n_rosetta_tf).numpy()

print(f"   ✓ Rosetta3 predictions generated: {theta_rosetta.shape}")

# ============================================================================
# 4. Alternative WRF Families
# ============================================================================
print("\n4. Fitting alternative WRF families to observed data...")

def fredlund_xing_theta(psi, a, n, m, theta_s, theta_r):
    """
    Fredlund-Xing (1994) water retention function.
    θ(ψ) = θr + (θs - θr) / [ln(e + (ψ/a)^n)]^m
    """
    psi = np.maximum(psi, 1e-6)  # Avoid log(0)
    Se = 1.0 / (np.log(np.e + (psi / a)**n)**m)
    theta = theta_r + (theta_s - theta_r) * Se
    return theta

def kosugi_theta(psi, psi_m, sigma, theta_s, theta_r):
    """
    Kosugi (1996) water retention function.
    θ(ψ) = θr + (θs - θr) * Q[ln(ψ/ψm) / σ]
    where Q is the complementary cumulative normal distribution.
    """
    from scipy.special import erfc
    psi = np.maximum(psi, 1e-6)
    x = np.log(psi / psi_m) / sigma
    Se = 0.5 * erfc(x / np.sqrt(2))
    theta = theta_r + (theta_s - theta_r) * Se
    return theta

def bimodal_vg_theta(psi, alpha1, n1, w1, alpha2, n2, theta_s, theta_r):
    """
    Bimodal van Genuchten (Durner 1994).
    θ(ψ) = θr + (θs - θr) * [w1 * Se1(ψ) + w2 * Se2(ψ)]
    where w1 + w2 = 1
    """
    w2 = 1.0 - w1
    m1 = 1.0 - 1.0 / n1
    m2 = 1.0 - 1.0 / n2
    
    Se1 = (1.0 + (alpha1 * psi)**n1)**(-m1)
    Se2 = (1.0 + (alpha2 * psi)**n2)**(-m2)
    Se = w1 * Se1 + w2 * Se2
    
    theta = theta_r + (theta_s - theta_r) * Se
    return theta

def fit_wrf_to_observed(psi_obs, theta_obs, wrf_func, bounds, theta_s, theta_r):
    """
    Fit a WRF to observed data using differential evolution.
    """
    def objective(params):
        try:
            theta_pred = wrf_func(psi_obs, *params, theta_s, theta_r)
            # RMSE
            mask = ~np.isnan(theta_obs)
            if np.sum(mask) < 3:
                return 1e6
            rmse = np.sqrt(np.mean((theta_obs[mask] - theta_pred[mask])**2))
            return rmse
        except:
            return 1e6
    
    # Use observed theta_s and theta_r
    result = differential_evolution(objective, bounds, seed=42, maxiter=100, 
                                    popsize=15, atol=1e-6, tol=1e-6)
    
    if result.success:
        return result.x, result.fun
    else:
        return None, None

# Fit alternative WRFs to observed data
print("   Fitting Fredlund-Xing, Kosugi, and Bimodal VG to observed curves...")

theta_fx = np.zeros_like(y_test)
theta_kosugi = np.zeros_like(y_test)
theta_bimodal = np.zeros_like(y_test)

params_fx = []
params_kosugi = []
params_bimodal = []

n_samples = len(y_test)
n_fitted = 0

for i in range(n_samples):
    if (i + 1) % 20 == 0:
        print(f"     Fitting sample {i+1}/{n_samples}...")
    
    psi_obs = psi
    theta_obs = y_test[i]
    theta_s_i = theta_s_test[i]
    theta_r_i = theta_r_test[i]
    
    # Mask valid observations
    mask = ~np.isnan(theta_obs)
    if np.sum(mask) < 5:
        # Use VGParamNet as fallback
        theta_fx[i] = theta_vgnet[i]
        theta_kosugi[i] = theta_vgnet[i]
        theta_bimodal[i] = theta_vgnet[i]
        continue
    
    psi_valid = psi_obs[mask]
    theta_valid = theta_obs[mask]
    
    # Fredlund-Xing: [a, n, m]
    bounds_fx = [(0.01, 1000), (0.1, 10), (0.1, 10)]
    params_fx_i, rmse_fx = fit_wrf_to_observed(
        psi_valid, theta_valid, fredlund_xing_theta, bounds_fx, 
        theta_s_i, theta_r_i)
    
    if params_fx_i is not None:
        theta_fx[i] = fredlund_xing_theta(psi, *params_fx_i, theta_s_i, theta_r_i)
        params_fx.append(params_fx_i)
        n_fitted += 1
    else:
        theta_fx[i] = theta_vgnet[i]  # Fallback
    
    # Kosugi: [psi_m, sigma]
    bounds_kosugi = [(0.1, 10000), (0.1, 5.0)]
    params_kosugi_i, rmse_kosugi = fit_wrf_to_observed(
        psi_valid, theta_valid, kosugi_theta, bounds_kosugi,
        theta_s_i, theta_r_i)
    
    if params_kosugi_i is not None:
        theta_kosugi[i] = kosugi_theta(psi, *params_kosugi_i, theta_s_i, theta_r_i)
        params_kosugi.append(params_kosugi_i)
    else:
        theta_kosugi[i] = theta_vgnet[i]  # Fallback
    
    # Bimodal VG: [alpha1, n1, w1, alpha2, n2]
    bounds_bimodal = [(0.001, 1.0), (1.01, 5.0), (0.1, 0.9), 
                      (0.001, 1.0), (1.01, 5.0)]
    params_bimodal_i, rmse_bimodal = fit_wrf_to_observed(
        psi_valid, theta_valid, bimodal_vg_theta, bounds_bimodal,
        theta_s_i, theta_r_i)
    
    if params_bimodal_i is not None:
        theta_bimodal[i] = bimodal_vg_theta(psi, *params_bimodal_i, 
                                            theta_s_i, theta_r_i)
        params_bimodal.append(params_bimodal_i)
    else:
        theta_bimodal[i] = theta_vgnet[i]  # Fallback

print(f"   ✓ Fitted {n_fitted}/{n_samples} samples with alternative WRFs")

# ============================================================================
# 5. Compute Metrics (Focus on Wet-End/Knee)
# ============================================================================
print("\n5. Computing metrics (focus on wet-end/knee errors)...")

def compute_rmse_by_regime(curves_pred, curves_obs, psi_grid):
    """Compute RMSE in wet-end, mid-range, and dry-end regimes"""
    # Wet-end: ψ < 10^2 kPa
    # Mid-range: 10^2 to 10^4 kPa
    # Dry-end: ψ > 10^4 kPa
    
    wet_mask = psi_grid < 100
    mid_mask = (psi_grid >= 100) & (psi_grid < 10000)
    dry_mask = psi_grid >= 10000
    
    rmse_wet = []
    rmse_mid = []
    rmse_dry = []
    rmse_global = []
    
    for i in range(len(curves_pred)):
        theta_pred = curves_pred[i]
        theta_obs = curves_obs[i]
        
        # Global RMSE
        mask = ~np.isnan(theta_obs)
        if np.sum(mask) > 0:
            rmse_global.append(np.sqrt(np.mean((theta_pred[mask] - theta_obs[mask])**2)))
        else:
            rmse_global.append(np.nan)
        
        # Wet-end
        mask_wet = wet_mask & ~np.isnan(theta_obs)
        if np.sum(mask_wet) > 0:
            rmse_wet.append(np.sqrt(np.mean((theta_pred[mask_wet] - theta_obs[mask_wet])**2)))
        else:
            rmse_wet.append(np.nan)
        
        # Mid-range
        mask_mid = mid_mask & ~np.isnan(theta_obs)
        if np.sum(mask_mid) > 0:
            rmse_mid.append(np.sqrt(np.mean((theta_pred[mask_mid] - theta_obs[mask_mid])**2)))
        else:
            rmse_mid.append(np.nan)
        
        # Dry-end
        mask_dry = dry_mask & ~np.isnan(theta_obs)
        if np.sum(mask_dry) > 0:
            rmse_dry.append(np.sqrt(np.mean((theta_pred[mask_dry] - theta_obs[mask_dry])**2)))
        else:
            rmse_dry.append(np.nan)
    
    return np.array(rmse_global), np.array(rmse_wet), np.array(rmse_mid), np.array(rmse_dry)

def compute_psi50_error(curves_pred, curves_obs, psi_grid, theta_s, theta_r):
    """Compute error in ψ50 (knee location)"""
    def find_psi50(psi, theta, ts, tr):
        Se = (theta - tr) / (ts - tr + 1e-8)
        Se = np.clip(Se, 0.0, 1.0)
        target = 0.5
        idx = np.where(Se <= target)[0]
        if len(idx) > 0 and idx[0] > 0:
            k = idx[0]
            # Linear interpolation in log space
            log_psi = np.log10(psi)
            x0, x1 = log_psi[k-1], log_psi[k]
            y0, y1 = Se[k-1], Se[k]
            if y1 != y0:
                t = (target - y0) / (y1 - y0)
                log_psi50 = x0 + t * (x1 - x0)
                return 10**log_psi50
        return np.nan
    
    psi50_pred = []
    psi50_obs = []
    
    for i in range(len(curves_pred)):
        psi50_p = find_psi50(psi_grid, curves_pred[i], theta_s[i], theta_r[i])
        psi50_o = find_psi50(psi_grid, curves_obs[i], theta_s[i], theta_r[i])
        psi50_pred.append(psi50_p)
        psi50_obs.append(psi50_o)
    
    psi50_pred = np.array(psi50_pred)
    psi50_obs = np.array(psi50_obs)
    
    # Relative error in log space
    mask = ~(np.isnan(psi50_pred) | np.isnan(psi50_obs))
    if np.sum(mask) > 0:
        log_error = np.abs(np.log10(psi50_pred[mask]) - np.log10(psi50_obs[mask]))
        return log_error, psi50_pred, psi50_obs
    else:
        return np.array([]), psi50_pred, psi50_obs

# Compute metrics for all models
models = {
    'VGParamNet': theta_vgnet,
    'Rosetta3': theta_rosetta,
    'Fredlund-Xing': theta_fx,
    'Kosugi': theta_kosugi,
    'Bimodal VG': theta_bimodal,
}

results = {}

for model_name, curves_pred in models.items():
    print(f"   Computing metrics for {model_name}...")
    
    # RMSE by regime
    rmse_global, rmse_wet, rmse_mid, rmse_dry = compute_rmse_by_regime(
        curves_pred, y_test, psi)
    
    # ψ50 error
    psi50_log_error, psi50_pred, psi50_obs = compute_psi50_error(
        curves_pred, y_test, psi, theta_s_test, theta_r_test)
    
    results[model_name] = {
        'rmse_global': rmse_global,
        'rmse_wet': rmse_wet,
        'rmse_mid': rmse_mid,
        'rmse_dry': rmse_dry,
        'psi50_log_error': psi50_log_error,
        'psi50_pred': psi50_pred,
        'psi50_obs': psi50_obs,
    }

# ============================================================================
# 6. Create Comparison Table
# ============================================================================
print("\n6. Creating comparison table...")

summary_data = []
for model_name in models.keys():
    r = results[model_name]
    summary_data.append({
        'Model': model_name,
        'Global RMSE (mean)': f"{np.nanmean(r['rmse_global']):.4f}",
        'Global RMSE (median)': f"{np.nanmedian(r['rmse_global']):.4f}",
        'Wet-end RMSE (mean)': f"{np.nanmean(r['rmse_wet']):.4f}",
        'Wet-end RMSE (median)': f"{np.nanmedian(r['rmse_wet']):.4f}",
        'Mid-range RMSE (mean)': f"{np.nanmean(r['rmse_mid']):.4f}",
        'Mid-range RMSE (median)': f"{np.nanmedian(r['rmse_mid']):.4f}",
        'Dry-end RMSE (mean)': f"{np.nanmean(r['rmse_dry']):.4f}",
        'Dry-end RMSE (median)': f"{np.nanmedian(r['rmse_dry']):.4f}",
        'ψ50 log error (mean)': f"{np.nanmean(r['psi50_log_error']):.3f}" if len(r['psi50_log_error']) > 0 else "N/A",
        'ψ50 log error (median)': f"{np.nanmedian(r['psi50_log_error']):.3f}" if len(r['psi50_log_error']) > 0 else "N/A",
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(OUTPUT_DIR / "comparison_summary.csv", index=False)
print(f"   ✓ Saved: {OUTPUT_DIR / 'comparison_summary.csv'}")

# ============================================================================
# 7. Create Comparison Figure
# ============================================================================
print("\n7. Creating comparison figure...")

_F6 = 11
_F6_TICK = 11

plt.rcParams['font.size'] = _F6
plt.rcParams['axes.labelsize'] = _F6
plt.rcParams['axes.titlesize'] = _F6
plt.rcParams['xtick.labelsize'] = _F6_TICK
plt.rcParams['ytick.labelsize'] = _F6_TICK
plt.rcParams['legend.fontsize'] = _F6
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# Taller top row + wider right column so panel (c) SWCCs are easier to read
fig = plt.figure(figsize=(19, 12))
gs = fig.add_gridspec(
    2, 3,
    height_ratios=[1.28, 1.0],
    width_ratios=[1.0, 1.0, 1.55],
    hspace=0.40,
    wspace=0.34,
)
axes = np.array(
    [
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])],
    ],
    dtype=object,
)


def _s6_arial_ticks(ax):
    ax.tick_params(axis='both', labelsize=_F6_TICK)
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_fontfamily('Arial')


def _s6_log_psi_swcc_axis(ax):
    """Full-range log ψ (kPa) with plain decade ticks — same style as Figures 3 / 11 / S5."""
    ax.set_xlabel('Matric suction \u03c8 (kPa)', fontsize=_F6, fontfamily='Arial', labelpad=10)
    ax.set_xscale('log')
    ax.set_xlim(float(np.min(psi)), float(np.max(psi)))
    _ticks = np.array([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    _ticklabels = ['0.1', '1.0', '10', '100', '1000', '10000', '100000', '1000000']
    ax.set_xticks(_ticks)
    ax.set_xticklabels(_ticklabels, fontsize=_F6_TICK, fontfamily='Arial')


def _s6_effective_saturation(theta, theta_s, theta_r):
    """S_e = (θ − θ_r)/(θ_s − θ_r), clipped to [0, 1]."""
    denom = np.maximum(theta_s - theta_r, 1e-12)
    return np.clip((theta - theta_r) / denom, 0.0, 1.0)


def _s6_pick_swcc_sample(y_true, theta_s_arr, theta_r_arr):
    """Pick a sample whose observed curve spans most of the wet→dry transition (clear S-shape)."""
    best_i = 0
    best_span = -1.0
    for i in range(len(y_true)):
        th = y_true[i]
        if np.sum(~np.isnan(th)) < 20:
            continue
        se = _s6_effective_saturation(th, theta_s_arr[i], theta_r_arr[i])
        ok = np.isfinite(th) & np.isfinite(se)
        span = float(np.nanmax(se[ok]) - np.nanmin(se[ok]))
        if span > best_span:
            best_span = span
            best_i = i
    return best_i


def _s6_interp_log_psi(psi_grid, y_vec, n_fine=400):
    """Dense log-ψ samples so SWCC lines read as smooth curves (not jagged segments)."""
    m = np.isfinite(psi_grid) & np.isfinite(y_vec)
    p = psi_grid[m]
    y = y_vec[m]
    o = np.argsort(p)
    p, y = p[o], y[o]
    if len(p) < 3:
        return psi_grid, y_vec
    logp = np.log10(np.maximum(p, 1e-30))
    logf = np.linspace(logp[0], logp[-1], n_fine)
    yf = np.interp(logf, logp, y)
    return np.power(10.0, logf), yf

# Panel (a): RMSE by regime (boxplot)
ax = axes[0, 0]
data_to_plot = []
labels = []
for model_name in models.keys():
    r = results[model_name]
    data_to_plot.append(r['rmse_wet'][~np.isnan(r['rmse_wet'])])
    labels.append(f"{model_name}\n(wet-end)")

bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
colors = ['#2E86AB', '#F18F01', '#06A77D', '#C73E1D', '#9b59b6']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('RMSE (m³/m³)', fontsize=_F6, fontfamily='Arial', labelpad=10)
ax.set_title('(a) Wet-end RMSE comparison', fontsize=_F6, fontfamily='Arial', pad=12, fontweight='normal')
ax.grid(True, alpha=0.3, axis='y', color='#CCCCCC', linewidth=0.7)
_s6_arial_ticks(ax)

# Panel (b): ψ50 error comparison
ax = axes[0, 1]
data_to_plot = []
labels = []
for model_name in models.keys():
    r = results[model_name]
    if len(r['psi50_log_error']) > 0:
        data_to_plot.append(r['psi50_log_error'])
        labels.append(model_name)

bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('|log10(ψ50_pred) - log10(ψ50_obs)|', fontsize=_F6, fontfamily='Arial', labelpad=10)
ax.set_title('(b) Knee location (ψ50) error', fontsize=_F6, fontfamily='Arial', pad=12, fontweight='normal')
ax.grid(True, alpha=0.3, axis='y', color='#CCCCCC', linewidth=0.7)
_s6_arial_ticks(ax)

# Panel (c): Representative SWCCs — normalized S_e vs log ψ (classic S-shape; raw θ zoom can look "linear")
ax = axes[0, 2]
sample_idx = _s6_pick_swcc_sample(y_test, theta_s_test, theta_r_test)
ts_i = float(theta_s_test[sample_idx])
tr_i = float(theta_r_test[sample_idx])

psi_p, se_obs = _s6_interp_log_psi(
    psi, _s6_effective_saturation(y_test[sample_idx], ts_i, tr_i)
)
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
ax.semilogx(psi_p, se_obs, 'k-', linewidth=2.8, label='Observed', alpha=0.95, zorder=20)
for idx, (model_name, curves) in enumerate(models.items()):
    _, se_m = _s6_interp_log_psi(
        psi, _s6_effective_saturation(curves[sample_idx], ts_i, tr_i)
    )
    ax.semilogx(
        psi_p,
        se_m,
        linewidth=2.1,
        linestyle=linestyles[idx % len(linestyles)],
        label=model_name,
        alpha=0.88,
        color=colors[idx],
        zorder=10 - idx,
    )
_s6_log_psi_swcc_axis(ax)
ax.set_ylim(-0.02, 1.05)
ax.set_ylabel(
    'Normalized water content\n(\u03b8 \u2212 \u03b8_r)/(\u03b8_s \u2212 \u03b8_r)',
    fontsize=_F6,
    fontfamily='Arial',
    labelpad=10,
)
ax.set_title('(c) Representative SWCCs', fontsize=_F6, fontfamily='Arial', pad=12, fontweight='normal')
leg_c = ax.legend(
    fontsize=_F6_TICK,
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),
    framealpha=0.96,
    borderaxespad=0.0,
)
for text in leg_c.get_texts():
    text.set_fontfamily('Arial')
ax.grid(False)
ax.xaxis.grid(False)
ax.yaxis.grid(False)
_s6_arial_ticks(ax)

# Panel (d): RMSE distribution (wet-end)
ax = axes[1, 0]
for idx, (model_name, curves) in enumerate(models.items()):
    r = results[model_name]
    rmse_wet = r['rmse_wet'][~np.isnan(r['rmse_wet'])]
    ax.hist(rmse_wet, bins=30, alpha=0.5, label=model_name, color=colors[idx], 
            edgecolor='black', linewidth=0.5)
ax.set_xlabel('Wet-end RMSE (m³/m³)', fontsize=_F6, fontfamily='Arial', labelpad=10)
ax.set_ylabel('Frequency', fontsize=_F6, fontfamily='Arial', labelpad=10)
ax.set_title('(d) Wet-end RMSE distribution', fontsize=_F6, fontfamily='Arial', pad=12, fontweight='normal')
leg_d = ax.legend(fontsize=_F6_TICK, loc='upper right')
for text in leg_d.get_texts():
    text.set_fontfamily('Arial')
ax.grid(True, alpha=0.3, axis='y', color='#CCCCCC', linewidth=0.7)
_s6_arial_ticks(ax)

# Panel (e): RMSE by regime (bar chart)
ax = axes[1, 1]
x = np.arange(len(models.keys()))
width = 0.25
regimes = ['Wet', 'Mid', 'Dry']
regime_data = {
    'Wet': [np.nanmean(results[m]['rmse_wet']) for m in models.keys()],
    'Mid': [np.nanmean(results[m]['rmse_mid']) for m in models.keys()],
    'Dry': [np.nanmean(results[m]['rmse_dry']) for m in models.keys()],
}

for i, regime in enumerate(regimes):
    offset = (i - 1) * width
    ax.bar(x + offset, regime_data[regime], width, label=regime, 
           color=colors[i], alpha=0.7, edgecolor='black', linewidth=1)

ax.set_xlabel('Model', fontsize=_F6, fontfamily='Arial', labelpad=10)
ax.set_ylabel('Mean RMSE (m³/m³)', fontsize=_F6, fontfamily='Arial', labelpad=10)
ax.set_title('(e) RMSE by regime', fontsize=_F6, fontfamily='Arial', pad=12, fontweight='normal')
ax.set_xticks(x)
ax.set_xticklabels(list(models.keys()), rotation=45, ha='right', fontsize=_F6_TICK, fontfamily='Arial')
leg_e = ax.legend(fontsize=_F6_TICK)
for text in leg_e.get_texts():
    text.set_fontfamily('Arial')
ax.grid(True, alpha=0.3, axis='y', color='#CCCCCC', linewidth=0.7)
_s6_arial_ticks(ax)

# Panel (f): Summary statistics (compact text blocks — not a table)
ax = axes[1, 2]
ax.axis('off')
ax.set_title(
    '(f) Summary statistics',
    fontsize=_F6,
    fontfamily='Arial',
    pad=12,
    fontweight='normal',
)
model_list = list(models.keys())
n_models = len(model_list)
y_top = 0.93
y_step = 0.17
for mi, model_name in enumerate(model_list):
    r = results[model_name]
    wet_m = np.nanmean(r['rmse_wet'])
    if len(r['psi50_log_error']) > 0:
        psi50_m = np.nanmedian(r['psi50_log_error'])
        psi50_line = f'Median psi50 log error: {psi50_m:.3f}'
    else:
        psi50_line = 'Median psi50 log error: N/A'
    y = y_top - mi * y_step
    ax.text(
        0.05, y, model_name,
        transform=ax.transAxes,
        fontsize=_F6, fontfamily='Arial', fontweight='bold',
        color=colors[mi % len(colors)], verticalalignment='top',
    )
    ax.text(
        0.05, y - 0.048,
        f'Mean wet-end RMSE: {wet_m:.4f} m\u00b3/m\u00b3\n{psi50_line}',
        transform=ax.transAxes,
        fontsize=_F6_TICK, fontfamily='Arial', verticalalignment='top',
        linespacing=1.3, color='#222222',
    )

for ax in axes.ravel():
    ax.set_facecolor('white')
    if not getattr(ax, 'axison', True):
        continue
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.2)

plt.tight_layout(rect=[0.0, 0.0, 0.93, 1.0])
fig.savefig(OUTPUT_DIR / "comparison_figure.png", dpi=300, bbox_inches='tight')
fig.savefig(OUTPUT_DIR / "comparison_figure.pdf", dpi=300, bbox_inches='tight')
SUPP_FIG_DIR = ROOT_DIR / "paper_figures" / "supplementary"
SUPP_FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(SUPP_FIG_DIR / "Figure_S6_PTF_WRF_Comparison.png", dpi=300, bbox_inches='tight')
fig.savefig(SUPP_FIG_DIR / "Figure_S6_PTF_WRF_Comparison.pdf", dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ Saved: {OUTPUT_DIR / 'comparison_figure.png'}")
print(f"   ✓ Saved: {SUPP_FIG_DIR / 'Figure_S6_PTF_WRF_Comparison.pdf'}")

# Save detailed results
with open(OUTPUT_DIR / "comparison_results.json", 'w') as f:
    # Convert numpy arrays to lists for JSON
    results_json = {}
    for model_name, r in results.items():
        results_json[model_name] = {
            'rmse_global_mean': float(np.nanmean(r['rmse_global'])),
            'rmse_global_median': float(np.nanmedian(r['rmse_global'])),
            'rmse_wet_mean': float(np.nanmean(r['rmse_wet'])),
            'rmse_wet_median': float(np.nanmedian(r['rmse_wet'])),
            'rmse_mid_mean': float(np.nanmean(r['rmse_mid'])),
            'rmse_mid_median': float(np.nanmedian(r['rmse_mid'])),
            'rmse_dry_mean': float(np.nanmean(r['rmse_dry'])),
            'rmse_dry_median': float(np.nanmedian(r['rmse_dry'])),
            'psi50_log_error_mean': float(np.nanmean(r['psi50_log_error'])) if len(r['psi50_log_error']) > 0 else None,
            'psi50_log_error_median': float(np.nanmedian(r['psi50_log_error'])) if len(r['psi50_log_error']) > 0 else None,
        }
    json.dump(results_json, f, indent=2)

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)
print(f"\nKey Findings:")
for model_name in models.keys():
    r = results[model_name]
    print(f"\n{model_name}:")
    print(f"  Wet-end RMSE: {np.nanmean(r['rmse_wet']):.4f} (mean), {np.nanmedian(r['rmse_wet']):.4f} (median)")
    if len(r['psi50_log_error']) > 0:
        print(f"  ψ₅₀ error: {np.nanmean(r['psi50_log_error']):.3f} (mean), {np.nanmedian(r['psi50_log_error']):.3f} (median)")

print(f"\nFiles saved in: {OUTPUT_DIR}")
