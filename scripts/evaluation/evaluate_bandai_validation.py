#!/usr/bin/env python3
"""
Bandai 2023 External Validation — 7 independent soils
(AZ2 Sand → AZ18 Clay, Arizona, USA)

Runs GB, PGNN/MonotonicPINN, and VGParamNet PTF models and compares
predicted SWCCs against lab-measured data.  Also compares VGParamNet
predicted (α, n) against lab-fitted VG parameters.

Unit notes:
  Suction grid  : kPa     (UNSODA training convention)
  Bandai alpha  : 1/cm → multiply ×10.197 to convert to 1/kPa
  VGParamNet α  : 1/kPa  (bounds [0.001, 1.0], see models/vg_param_net.py)
  Bandai EC     : μS/cm  → divide ÷100 to convert to dS/m (UNSODA convention)

Output → results_bandai_validation/
  bandai_metrics.csv       — RMSE / MAE / admissibility per soil × model
  bandai_vg_params.csv     — VGParamNet (α,n) vs. lab-fitted (α,n)
  bandai_swcc_curves.pdf/png
  bandai_vg_scatter.pdf/png
  bandai_theta_scatter.pdf/png

Author: Avzalshoev et al. 2026
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES']    = '-1'

import sys, warnings, zipfile, io
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

import tensorflow as tf

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.vg_param_net import vg_theta as vg_theta_tf, VGParamNet
from models.pinn_monotonic import MonotonicPINN
from models.pinn import PhysicsEncodingLayer

DATA_PROC   = ROOT / "data_processed"
DATA_NORM   = ROOT / "data_pinn_normalized"
RES_FIXED   = ROOT / "results_pinn_fixed"
BANDAI_ZIP  = ROOT / "Bandai_2023" / "10622405.zip"
OUT_DIR     = ROOT / "results_bandai_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VGPARAM_MODEL = RES_FIXED / "vgparamnet"  / "vgparamnet_best.keras"
PGNN_MODEL    = RES_FIXED / "checkpoints" / "pinn_best_model_fixed.keras"

CM_PER_KPA = 10.197          # 1 kPa ≈ 10.197 cmH₂O

# ── Feature column lists ─────────────────────────────────────────────────────
FEAT20 = ['D10','D30','D50','D60','D90','Cu','Cc',
          'clay_pct','silt_pct','sand_pct',
          'bulk_density','particle_density','porosity','void_ratio','Gs',
          'OM_content','pH','EC','theta_s','theta_r']

FEAT16 = ['Cc','Cu','D10','D30','D50','D60','D90',
          'OM_content','bulk_density','clay_pct','pH','porosity',
          'sand_pct','silt_pct','theta_r','theta_s']

AVAILABLE   = ['clay_pct','silt_pct','sand_pct',
               'bulk_density','particle_density','porosity','void_ratio','Gs',
               'OM_content','EC','theta_s','theta_r']
NEED_IMPUTE = ['D10','D30','D50','D60','D90','Cu','Cc','pH']

C_GB   = '#D62728'
C_PGNN = '#F4A636'
C_VGP  = '#1F77B4'
C_MEAS = '#2E2E2E'

TEXTURE = {
    'AZ2': 'Sand', 'AZ4B': 'Loamy Sand', 'AZ7': 'Sandy Loam',
    'AZ11': 'Loam', 'AZ13': 'Sandy Clay Loam',
    'AZ15': 'Silt Loam', 'AZ18': 'Clay'}


def vg_curve_np(psi_kPa, alpha_kPa, n, theta_s, theta_r):
    """NumPy VG SWCC; psi and alpha in kPa."""
    m  = 1.0 - 1.0 / n
    Se = (1.0 + (alpha_kPa * np.abs(psi_kPa)) ** n) ** (-m)
    return theta_r + Se * (theta_s - theta_r)


# ── Step 1: Load Bandai soil data ────────────────────────────────────────────
print("=" * 60)
print("Step 1: Loading Bandai 2023 soil data")

with zipfile.ZipFile(BANDAI_ZIP, 'r') as outer:
    with outer.open('data.zip') as dz:
        inner_bytes = io.BytesIO(dz.read())

with zipfile.ZipFile(inner_bytes, 'r') as inner:
    with inner.open('data_ver2/soil_properties.xlsx') as f:
        sp = pd.read_excel(io.BytesIO(f.read()))
    with inner.open('data_ver2/WRC_fitted_summary.csv') as f:
        wrc_fit = pd.read_csv(f)
    with inner.open('data_ver2/water_retention_curve_tempe_cell.xlsx') as f:
        wrc_raw = pd.read_excel(io.BytesIO(f.read()))
    with inner.open('data_ver2/WP4_measurements.xlsx') as f:
        wp4_raw = pd.read_excel(io.BytesIO(f.read()))

# VGM-fitted parameters
vgm = wrc_fit[wrc_fit['hydraulic_model'] == 'VGM'].copy()
vgm = vgm[['dataset','theta_r','theta_s','alpha','n']].rename(
    columns={'dataset': 'Soil', 'alpha': 'alpha_lab_cm', 'n': 'n_lab',
             'theta_r': 'theta_r_lab', 'theta_s': 'theta_s_lab'})
vgm['alpha_lab_kPa'] = vgm['alpha_lab_cm'] * CM_PER_KPA

sp = sp.rename(columns={
    'Sand_%':                            'sand_pct',
    'Silt_%':                            'silt_pct',
    'Clay_%':                            'clay_pct',
    'Organic_Matter_weight_%':           'OM_content',
    'Particle_Density_g_cm-3':           'particle_density',
    'Established_Bulk_Density_g_cm-3':   'bulk_density',
    'Electrical_Conductivity_uS_cm-1':   'EC_uS',
    'fitted_theta_s_WRC':                'theta_s_wrc',
})
sp['EC']         = sp['EC_uS'] / 100.0
sp['porosity']   = 1.0 - sp['bulk_density'] / sp['particle_density']
sp['void_ratio'] = sp['porosity'] / (1.0 - sp['porosity'])
sp['Gs']         = sp['particle_density']

df = sp[['Soil','sand_pct','silt_pct','clay_pct','bulk_density',
         'particle_density','porosity','void_ratio','Gs',
         'OM_content','EC','theta_s_wrc']].merge(vgm, on='Soil')

df['theta_s'] = df['theta_s_wrc']
df['theta_r'] = df['theta_r_lab'].clip(lower=1e-4)

SOILS   = df['Soil'].tolist()
theta_s = df['theta_s'].values
theta_r = df['theta_r'].values
print(f"  Soils ({len(SOILS)}): {SOILS}")

# ── Step 2: Measured SWCC data ───────────────────────────────────────────────
print("\nStep 2: Loading measured SWCC data")

# AZ2: raw tempe-cell + WP4 measurements (3 replicates, averaged)
psi_az2   = np.concatenate([wrc_raw['suction_cm'].values,
                             wp4_raw['suction_cm'].values]) / CM_PER_KPA
theta_az2 = np.concatenate([wrc_raw['volumetric_water_content'].values,
                             wp4_raw['volumetric_water_content'].values])
az2_df = (pd.DataFrame({'psi': psi_az2, 'theta': theta_az2})
          .round({'psi': 4}).groupby('psi').mean().reset_index()
          .sort_values('psi'))

# AZ4B–AZ18: VGM-fitted reference curve on dense grid (no raw WRC points provided)
measured = {'AZ2': az2_df}
psi_dense = np.logspace(np.log10(0.001), np.log10(4e4), 300)
for _, row in vgm.iterrows():
    s = row['Soil']
    if s == 'AZ2':
        continue
    th = vg_curve_np(psi_dense, row['alpha_lab_kPa'], row['n_lab'],
                     row['theta_s_lab'], max(float(row['theta_r_lab']), 1e-4))
    measured[s] = pd.DataFrame({'psi': psi_dense, 'theta': th})

print(f"  AZ2: {len(az2_df)} raw measurement points (tempe cell + WP4)")
print(f"  AZ4B–AZ18: VGM-fitted reference curves (WRC_fitted_summary.csv)")

# ── Step 3: k-NN imputation ──────────────────────────────────────────────────
print("\nStep 3: k-NN imputation for D10–D90, Cu, Cc, pH")

X_train_raw = pd.read_csv(DATA_PROC / 'X_train.csv').drop(columns=['code'], errors='ignore')
X_norm_raw  = pd.read_csv(DATA_NORM / 'X_train.csv')

MATCH_FEAT = ['sand_pct','silt_pct','clay_pct','bulk_density',
              'porosity','OM_content','theta_s','theta_r']

scaler_knn   = StandardScaler()
X_train_knn  = scaler_knn.fit_transform(X_train_raw[MATCH_FEAT].fillna(0))
X_bandai_knn = scaler_knn.transform(df[MATCH_FEAT].fillna(0).values)
nn = NearestNeighbors(n_neighbors=5, metric='euclidean')
nn.fit(X_train_knn)
_, idx = nn.kneighbors(X_bandai_knn)   # (7, 5)

X20 = df[AVAILABLE].copy()
global_means = {feat: float(X_train_raw[feat].mean(skipna=True)) for feat in NEED_IMPUTE}
for feat in NEED_IMPUTE:
    vals = []
    for i in idx:
        v = float(X_train_raw.iloc[i][feat].mean(skipna=True))
        vals.append(v if not np.isnan(v) else global_means[feat])
    X20[feat] = vals
X20 = X20[FEAT20]
X16 = X20[FEAT16].copy()
print(f"  Done: {NEED_IMPUTE}")
print(f"  NaN remaining in X16: {X16.isna().sum().sum()} | X20: {X20.isna().sum().sum()}")

# ── Step 4: Suction grid ─────────────────────────────────────────────────────
psi_grid = np.load(DATA_PROC / 'suction_grid.npy')

# ── Step 5: PGNN predictions ─────────────────────────────────────────────────
print("\nStep 5: PGNN/MonotonicPINN predictions")

scaler_pgnn = StandardScaler()
scaler_pgnn.fit(X_norm_raw[FEAT16])
X16_sc_pgnn = scaler_pgnn.transform(X16).astype(np.float32)
psi_tiled   = np.tile(psi_grid.astype(np.float32), (len(SOILS), 1))

# Build model from scratch, warm up, then load saved weights
pgnn_model = MonotonicPINN(soil_prop_dim=16, suction_points=100,
                            physics_units=128, hidden_dims=[128, 256, 128, 64])
pgnn_model({'soil_props': np.random.randn(1, 16).astype(np.float32),
             'suction':   np.random.randn(1, 100).astype(np.float32)})
saved_pgnn = tf.keras.models.load_model(
    str(PGNN_MODEL), compile=False,
    custom_objects={'MonotonicPINN': MonotonicPINN,
                    'PhysicsEncodingLayer': PhysicsEncodingLayer})
pgnn_model.set_weights(saved_pgnn.get_weights())

y_norm    = pgnn_model({'soil_props': X16_sc_pgnn, 'suction': psi_tiled},
                        training=False).numpy()
pgnn_pred = theta_r[:, None] + y_norm * (theta_s - theta_r)[:, None]

# ── Step 6: VGParamNet predictions ───────────────────────────────────────────
print("Step 6: VGParamNet predictions")

scaler_vgp = StandardScaler()
scaler_vgp.fit(X_norm_raw[FEAT16])
vgp_model  = tf.keras.models.load_model(
    str(VGPARAM_MODEL), compile=False,
    custom_objects={'VGParamNet': VGParamNet})
X16_sc_vgp = tf.constant(scaler_vgp.transform(X16), dtype=tf.float32)
alpha_tf, n_tf = vgp_model(X16_sc_vgp, training=False)
alpha_pred = alpha_tf.numpy().flatten()
n_pred     = n_tf.numpy().flatten()
vgp_pred   = np.array([
    vg_curve_np(psi_grid, alpha_pred[i], n_pred[i], theta_s[i], theta_r[i])
    for i in range(len(SOILS))])

print(f"  α: {alpha_pred.min():.5f}–{alpha_pred.max():.5f} 1/kPa  |  "
      f"n: {n_pred.min():.3f}–{n_pred.max():.3f}")

# ── Step 7: GB predictions ────────────────────────────────────────────────────
print("Step 7: GB — retraining on UNSODA training split")

y_train    = np.load(DATA_PROC / 'y_train.npy')
X_train_gb_df = X_train_raw[FEAT20].copy()
for col in X_train_gb_df.columns:
    X_train_gb_df[col] = X_train_gb_df[col].fillna(X_train_gb_df[col].mean())
scaler_gb  = StandardScaler()
X_train_gb = scaler_gb.fit_transform(X_train_gb_df)
gb_model   = MultiOutputRegressor(
    GradientBoostingRegressor(n_estimators=300, max_depth=4,
                              learning_rate=0.05, subsample=0.8, random_state=42),
    n_jobs=-1)
gb_model.fit(X_train_gb, y_train)
gb_pred = np.clip(gb_model.predict(scaler_gb.transform(X20)), 0.0, 0.85)

# ── Step 8: Metrics ───────────────────────────────────────────────────────────
print("\nStep 8: Metrics")

def interp_rmse(curve, psi_m, th_m):
    f = interp1d(psi_grid, curve,
                 bounds_error=False, fill_value=(curve[0], curve[-1]))
    return float(np.sqrt(np.mean((f(psi_m) - th_m) ** 2)))

def admissible(curve, tol=1e-4):
    viol = int(np.sum(np.diff(curve) > tol))
    return viol == 0, viol

records = []
for i, soil in enumerate(SOILS):
    m     = measured[soil]
    psi_m = m['psi'].values
    th_m  = m['theta'].values
    th_vgp_m = vg_curve_np(psi_m, alpha_pred[i], n_pred[i], theta_s[i], theta_r[i])
    gb_a,   gb_v   = admissible(gb_pred[i])
    pgnn_a, pgnn_v = admissible(pgnn_pred[i])
    vgp_a,  vgp_v  = admissible(vgp_pred[i])
    records.append({
        'Soil': soil, 'Texture': TEXTURE[soil],
        'GB_RMSE':   interp_rmse(gb_pred[i],   psi_m, th_m),
        'PGNN_RMSE': interp_rmse(pgnn_pred[i], psi_m, th_m),
        'VGP_RMSE':  float(np.sqrt(np.mean((th_vgp_m - th_m) ** 2))),
        'GB_adm':    gb_a,   'GB_viol':   gb_v,
        'PGNN_adm':  pgnn_a, 'PGNN_viol': pgnn_v,
        'VGP_adm':   vgp_a,  'VGP_viol':  vgp_v,
        'alpha_lab_cm':  float(df.iloc[i]['alpha_lab_cm']),
        'alpha_lab_kPa': float(df.iloc[i]['alpha_lab_kPa']),
        'n_lab':         float(df.iloc[i]['n_lab']),
        'alpha_pred':    float(alpha_pred[i]),
        'n_pred':        float(n_pred[i]),
    })
    r = records[-1]
    print(f"  {soil:4s}: GB={r['GB_RMSE']:.4f}  PGNN={r['PGNN_RMSE']:.4f}  "
          f"VGP={r['VGP_RMSE']:.4f}  GB_adm={'✓' if r['GB_adm'] else '✗'}  "
          f"α_lab={r['alpha_lab_kPa']:.4f}  α_pred={r['alpha_pred']:.4f}  "
          f"n_lab={r['n_lab']:.3f}  n_pred={r['n_pred']:.3f}")

df_metrics = pd.DataFrame(records)
df_metrics.to_csv(OUT_DIR / 'bandai_metrics.csv', index=False)

df_vg = df_metrics[['Soil','Texture','alpha_lab_cm','alpha_lab_kPa',
                     'n_lab','alpha_pred','n_pred']].copy()
df_vg['alpha_err_pct'] = 100 * np.abs(df_vg['alpha_pred'] - df_vg['alpha_lab_kPa']) / df_vg['alpha_lab_kPa']
df_vg['n_err_pct']     = 100 * np.abs(df_vg['n_pred']     - df_vg['n_lab'])          / df_vg['n_lab']
df_vg.to_csv(OUT_DIR / 'bandai_vg_params.csv', index=False)

# ── Step 9: Figure — SWCC panels ─────────────────────────────────────────────
print("\nStep 9: Generating figures")

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()

for i, soil in enumerate(SOILS):
    ax   = axes[i]
    meas = measured[soil]

    if soil == 'AZ2':
        ax.scatter(meas['psi'].values, meas['theta'].values,
                   s=22, color=C_MEAS, zorder=5, label='Measured')
    else:
        ax.semilogx(meas['psi'].values, meas['theta'].values,
                    color=C_MEAS, lw=1.0, ls=':', zorder=5, label='Lab VG fit')

    ax.semilogx(psi_grid, gb_pred[i],   color=C_GB,   lw=1.5, ls='--', label='GB')
    ax.semilogx(psi_grid, pgnn_pred[i], color=C_PGNN, lw=1.5, ls='-.', label='PGNN')
    psi_line = np.logspace(-3, np.log10(5e4), 300)
    ax.semilogx(psi_line,
                vg_curve_np(psi_line, alpha_pred[i], n_pred[i], theta_s[i], theta_r[i]),
                color=C_VGP, lw=1.8, label='VGParamNet')

    r = df_metrics.iloc[i]
    ax.set_title(f"{soil} — {TEXTURE[soil]}\n"
                 f"RMSE  GB={r['GB_RMSE']:.3f}  PGNN={r['PGNN_RMSE']:.3f}  VGP={r['VGP_RMSE']:.3f}",
                 fontsize=8)
    ax.set_xlim(0.001, 5e4)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Suction (kPa)', fontsize=7)
    ax.set_ylabel('θ (m³/m³)', fontsize=7)
    ax.tick_params(labelsize=7)

axes[-1].axis('off')
handles, labels = axes[0].get_legend_handles_labels()
axes[-1].legend(handles, labels, loc='center', fontsize=9)
fig.suptitle('External validation — Bandai (2023) vs. PTF predictions\n'
             '(k-NN imputation for D-values and pH; 7 soils, Sand → Clay)',
             fontsize=10, y=1.01)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(OUT_DIR / f'bandai_swcc_curves.{ext}', dpi=200, bbox_inches='tight')
plt.close()

# ── Step 10: Figure — VG parameter scatter ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
for ax, col_lab, col_pred, label, log_scale in [
        (axes[0], 'alpha_lab_kPa', 'alpha_pred', 'α (1/kPa)', True),
        (axes[1], 'n_lab',         'n_pred',      'n (−)',     False)]:
    lab_  = df_vg[col_lab].values
    pred_ = df_vg[col_pred].values
    ax.scatter(lab_, pred_, s=70, c=C_VGP, zorder=3)
    for _, row in df_vg.iterrows():
        ax.annotate(row['Soil'], (row[col_lab], row[col_pred]),
                    textcoords='offset points', xytext=(5, 3), fontsize=8)
    lo = min(lab_.min(), pred_.min())
    hi = max(lab_.max(), pred_.max())
    lims = [lo * 0.6 if log_scale else lo - (hi-lo)*0.1,
            hi * 1.8 if log_scale else hi + (hi-lo)*0.1]
    ax.plot(lims, lims, 'k--', lw=0.8)
    if log_scale:
        ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lims); ax.set_ylim(lims)
    r_val = np.corrcoef(np.log(lab_) if log_scale else lab_,
                        np.log(pred_) if log_scale else pred_)[0, 1]
    ax.text(0.05, 0.91, f'r{"(log)" if log_scale else ""} = {r_val:.3f}',
            transform=ax.transAxes, fontsize=9)
    ax.set_xlabel(f'Lab-fitted {label}', fontsize=9)
    ax.set_ylabel(f'VGParamNet predicted {label}', fontsize=9)
    ax.set_title(label, fontsize=9)
fig.suptitle('VGParamNet: predicted vs. lab-fitted VG parameters\n'
             '(Bandai 2023 — 7 independent soils)', fontsize=10)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(OUT_DIR / f'bandai_vg_scatter.{ext}', dpi=200, bbox_inches='tight')
plt.close()

# ── Step 11: Figure — aggregate θ scatter ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
for ax, (mname, col, parr) in zip(axes, [
        ('GB',         C_GB,   gb_pred),
        ('PGNN',       C_PGNN, pgnn_pred),
        ('VGParamNet', C_VGP,  None)]):
    all_obs, all_pred = [], []
    for i, soil in enumerate(SOILS):
        psi_m = measured[soil]['psi'].values
        th_m  = measured[soil]['theta'].values
        if mname == 'VGParamNet':
            th_p = vg_curve_np(psi_m, alpha_pred[i], n_pred[i], theta_s[i], theta_r[i])
        else:
            f = interp1d(psi_grid, parr[i],
                         bounds_error=False, fill_value=(parr[i, 0], parr[i, -1]))
            th_p = f(psi_m)
        all_obs.extend(th_m); all_pred.extend(th_p)
    all_obs = np.array(all_obs); all_pred = np.array(all_pred)
    rmse = np.sqrt(np.mean((all_pred - all_obs) ** 2))
    r2   = 1 - np.sum((all_pred - all_obs)**2) / np.sum((all_obs - all_obs.mean())**2)
    ax.scatter(all_obs, all_pred, s=14, alpha=0.6, c=col)
    lim = [min(all_obs.min(), all_pred.min()) - 0.01,
           max(all_obs.max(), all_pred.max()) + 0.01]
    ax.plot(lim, lim, 'k--', lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Measured θ (m³/m³)', fontsize=9)
    ax.set_ylabel('Predicted θ (m³/m³)', fontsize=9)
    ax.set_title(f'{mname}\nRMSE = {rmse:.4f}   R² = {r2:.3f}', fontsize=9)
fig.suptitle('All 7 Bandai soils combined: predicted vs. measured θ', fontsize=10)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(OUT_DIR / f'bandai_theta_scatter.{ext}', dpi=200, bbox_inches='tight')
plt.close()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("SUMMARY — Bandai 2023 External Validation")
print("=" * 72)
print(f"{'Soil':5s} {'Texture':16s} {'GB RMSE':>9s} {'PGNN':>8s} {'VGP':>8s}  Adm GB/PGNN/VGP")
print("-" * 72)
for _, r in df_metrics.iterrows():
    a = ('✓' if r['GB_adm'] else '✗', '✓' if r['PGNN_adm'] else '✗',
         '✓' if r['VGP_adm'] else '✗')
    print(f"{r['Soil']:5s} {r['Texture']:16s} "
          f"{r['GB_RMSE']:9.4f} {r['PGNN_RMSE']:8.4f} {r['VGP_RMSE']:8.4f}  "
          f"  {a[0]}    {a[1]}    {a[2]}")
print("-" * 72)
m = df_metrics[['GB_RMSE','PGNN_RMSE','VGP_RMSE']].mean()
ar = df_metrics[['GB_adm','PGNN_adm','VGP_adm']].mean() * 100
print(f"{'Mean':5s} {'':16s} "
      f"{m['GB_RMSE']:9.4f} {m['PGNN_RMSE']:8.4f} {m['VGP_RMSE']:8.4f}  "
      f"{ar['GB_adm']:4.0f}%  {ar['PGNN_adm']:4.0f}%  {ar['VGP_adm']:4.0f}%")

print("\nVG PARAMETER COMPARISON (all α in 1/kPa):")
print(f"{'Soil':5s} {'Texture':16s} {'α_lab':>9s} {'α_pred':>9s} {'α_err%':>7s}  "
      f"{'n_lab':>7s} {'n_pred':>7s} {'n_err%':>7s}")
print("-" * 78)
for _, r in df_vg.iterrows():
    print(f"{r['Soil']:5s} {r['Texture']:16s} "
          f"{r['alpha_lab_kPa']:9.5f} {r['alpha_pred']:9.5f} {r['alpha_err_pct']:7.1f}%  "
          f"{r['n_lab']:7.3f} {r['n_pred']:7.3f} {r['n_err_pct']:7.1f}%")
print("-" * 78)
print(f"  Mean |error|: α = {df_vg['alpha_err_pct'].mean():.1f}%   "
      f"n = {df_vg['n_err_pct'].mean():.1f}%")

print(f"\n→ Results: {OUT_DIR}")
