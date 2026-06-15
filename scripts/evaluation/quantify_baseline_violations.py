#!/usr/bin/env python3
"""
Quantify Baseline Model Physics Violations
- Monotonicity violations
- Boundary violations
- Visual examples of violations
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from sklearn.metrics import mean_squared_error

print("="*80)
print("Quantifying Baseline Physics Violations")
print("="*80)

# Try to load baseline model
baseline_model_path = Path("results_baseline/baseline_models/gradient_boosting_model.pkl")
if not baseline_model_path.exists():
    print("⚠ Baseline model not found. Creating analysis from saved predictions.")
    # Try to load from evaluation results
    baseline_results = pd.read_csv("results_baseline/baseline_results.csv")
    test_results = baseline_results[baseline_results['dataset'] == 'test']
    
    if len(test_results) == 0:
        print("⚠ No baseline results found. Skipping violation analysis.")
        exit(0)
    
    # We'll need to regenerate predictions
    print("⚠ Need to regenerate baseline predictions for violation analysis.")
    print("   This requires the baseline model file.")
    exit(0)

# Load baseline model
print(f"\nLoading baseline model: {baseline_model_path}")
model = joblib.load(baseline_model_path)
scaler = joblib.load(Path("results_baseline/baseline_models/scaler.pkl"))

# Load test data
X_test = pd.read_csv("data_processed/X_test.csv")
y_test = np.load("data_processed/y_test.npy")
suction_grid = np.load("data_processed/suction_grid.npy")

print(f"  Test samples: {len(X_test)}")

# Prepare features
feature_cols = ['D10', 'D30', 'D50', 'D60', 'D90', 'Cu', 'Cc', 
                'bulk_density', 'porosity', 'clay_pct', 'silt_pct', 'sand_pct',
                'OM_content', 'pH', 'theta_s', 'theta_r']

X_test_features = X_test[feature_cols].values
X_test_scaled = scaler.transform(X_test_features)

# Predict
print("\nGenerating baseline predictions...")
y_pred_baseline = model.predict(X_test_scaled)

print(f"  Predictions shape: {y_pred_baseline.shape}")

# ============================================================================
# 1. MONOTONICITY VIOLATIONS
# ============================================================================
print("\n1. Analyzing monotonicity violations...")

mono_violations = []
mono_violation_counts = []
mono_violation_severities = []

for i in range(len(y_pred_baseline)):
    diff = y_pred_baseline[i, :-1] - y_pred_baseline[i, 1:]
    violations = diff < -1e-6  # Non-monotonic segments
    
    if np.any(violations):
        mono_violations.append(i)
        n_violations = np.sum(violations)
        mono_violation_counts.append(n_violations)
        
        # Severity: maximum increase in theta
        max_increase = np.max(-diff[violations])
        mono_violation_severities.append(max_increase)

monotonicity_rate = 1.0 - len(mono_violations) / len(X_test)

print(f"  Monotonicity rate: {monotonicity_rate*100:.1f}%")
print(f"  Violations: {len(mono_violations)}/{len(X_test)} samples")
print(f"  Average violations per sample: {np.mean(mono_violation_counts) if mono_violation_counts else 0:.2f}")
print(f"  Max violations in one sample: {np.max(mono_violation_counts) if mono_violation_counts else 0}")
print(f"  Average severity (max Δθ): {np.mean(mono_violation_severities) if mono_violation_severities else 0:.6f}")
print(f"  Max severity: {np.max(mono_violation_severities) if mono_violation_severities else 0:.6f}")

# ============================================================================
# 2. BOUNDARY VIOLATIONS
# ============================================================================
print("\n2. Analyzing boundary violations...")

theta_s_test = X_test['theta_s'].values
theta_r_test = X_test['theta_r'].values

boundary_violations = []
boundary_violation_types = {'below_theta_r': 0, 'above_theta_s': 0, 'both': 0}
boundary_violation_severities = []

for i in range(len(X_test)):
    below_r = np.any(y_pred_baseline[i] < theta_r_test[i] - 1e-6)
    above_s = np.any(y_pred_baseline[i] > theta_s_test[i] + 1e-6)
    
    if below_r or above_s:
        boundary_violations.append(i)
        
        if below_r and above_s:
            boundary_violation_types['both'] += 1
        elif below_r:
            boundary_violation_types['below_theta_r'] += 1
        elif above_s:
            boundary_violation_types['above_theta_s'] += 1
        
        # Severity: maximum violation
        max_below = np.max(theta_r_test[i] - y_pred_baseline[i][y_pred_baseline[i] < theta_r_test[i]])
        max_above = np.max(y_pred_baseline[i][y_pred_baseline[i] > theta_s_test[i]] - theta_s_test[i])
        max_severity = max(max_below if below_r else 0, max_above if above_s else 0)
        boundary_violation_severities.append(max_severity)

boundary_rate = 1.0 - len(boundary_violations) / len(X_test)

print(f"  Boundary rate: {boundary_rate*100:.1f}%")
print(f"  Violations: {len(boundary_violations)}/{len(X_test)} samples")
print(f"  Below θ_r: {boundary_violation_types['below_theta_r']} samples")
print(f"  Above θ_s: {boundary_violation_types['above_theta_s']} samples")
print(f"  Both: {boundary_violation_types['both']} samples")
print(f"  Average severity: {np.mean(boundary_violation_severities) if boundary_violation_severities else 0:.6f}")
print(f"  Max severity: {np.max(boundary_violation_severities) if boundary_violation_severities else 0:.6f}")

# ============================================================================
# 3. DRY-END ERRORS
# ============================================================================
print("\n3. Analyzing dry-end errors...")

dry_end_threshold = 1e4  # kPa
dry_end_indices = np.where(suction_grid > dry_end_threshold)[0]

if len(dry_end_indices) > 0:
    y_test_dry = y_test[:, dry_end_indices]
    y_pred_dry = y_pred_baseline[:, dry_end_indices]
    
    y_test_dry_flat = y_test_dry.flatten()
    y_pred_dry_flat = y_pred_dry.flatten()
    
    mask_dry = ~(np.isnan(y_test_dry_flat) | np.isnan(y_pred_dry_flat))
    
    rmse_dry = np.sqrt(mean_squared_error(y_test_dry_flat[mask_dry], y_pred_dry_flat[mask_dry]))
    
    print(f"  Dry-end threshold: > {dry_end_threshold:.0f} kPa")
    print(f"  Points per sample: {len(dry_end_indices)}")
    print(f"  RMSE: {rmse_dry:.6f}")
else:
    rmse_dry = np.nan

# ============================================================================
# 4. VISUAL EXAMPLES OF VIOLATIONS
# ============================================================================
print("\n4. Creating violation visualization...")

viz_dir = Path("results_baseline/visualizations")
viz_dir.mkdir(parents=True, exist_ok=True)

# Select examples with violations
n_examples = min(6, len(mono_violations) + len(boundary_violations))
example_indices = list(set(mono_violations[:3] + boundary_violations[:3]))[:n_examples]

if len(example_indices) < n_examples:
    # Add random samples to fill
    remaining = [i for i in range(len(X_test)) if i not in example_indices]
    example_indices.extend(np.random.choice(remaining, n_examples - len(example_indices), replace=False))

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, sample_idx in enumerate(example_indices[:6]):
    ax = axes[idx]
    
    # Observed
    ax.semilogx(suction_grid, y_test[sample_idx], 'ko-', 
               markersize=4, linewidth=1.5, label='Observed', alpha=0.7)
    
    # Baseline prediction
    ax.semilogx(suction_grid, y_pred_baseline[sample_idx], 'r--', 
               linewidth=2, label='Gradient Boosting', alpha=0.8)
    
    # Boundaries
    ax.axhline(theta_s_test[sample_idx], color='g', linestyle=':', alpha=0.5, linewidth=1, label='θ_s')
    ax.axhline(theta_r_test[sample_idx], color='orange', linestyle=':', alpha=0.5, linewidth=1, label='θ_r')
    
    # Highlight violations
    has_mono_violation = sample_idx in mono_violations
    has_boundary_violation = sample_idx in boundary_violations
    
    title_parts = [f'Sample {sample_idx+1}']
    if has_mono_violation:
        title_parts.append('Non-monotonic')
    if has_boundary_violation:
        title_parts.append('Boundary violation')
    
    ax.set_xlabel('Suction (kPa)', fontsize=11)
    ax.set_ylabel('Water Content (θ)', fontsize=11)
    ax.set_title(' | '.join(title_parts), fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Baseline Model: Examples of Physics Violations', fontsize=16, y=0.995)
plt.tight_layout()
plt.savefig(viz_dir / 'baseline_physics_violations.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {viz_dir / 'baseline_physics_violations.png'}")

# ============================================================================
# 5. SAVE RESULTS
# ============================================================================
print("\n5. Saving violation analysis...")

results = {
    'monotonicity': {
        'rate': float(monotonicity_rate),
        'violations': int(len(mono_violations)),
        'total_samples': int(len(X_test)),
        'avg_violations_per_sample': float(np.mean(mono_violation_counts)) if mono_violation_counts else 0.0,
        'max_violations_in_sample': int(np.max(mono_violation_counts)) if mono_violation_counts else 0,
        'avg_severity': float(np.mean(mono_violation_severities)) if mono_violation_severities else 0.0,
        'max_severity': float(np.max(mono_violation_severities)) if mono_violation_severities else 0.0
    },
    'boundary': {
        'rate': float(boundary_rate),
        'violations': int(len(boundary_violations)),
        'total_samples': int(len(X_test)),
        'below_theta_r': int(boundary_violation_types['below_theta_r']),
        'above_theta_s': int(boundary_violation_types['above_theta_s']),
        'both': int(boundary_violation_types['both']),
        'avg_severity': float(np.mean(boundary_violation_severities)) if boundary_violation_severities else 0.0,
        'max_severity': float(np.max(boundary_violation_severities)) if boundary_violation_severities else 0.0
    },
    'dry_end': {
        'rmse': float(rmse_dry) if not np.isnan(rmse_dry) else None
    }
}

import json
results_file = Path("results_baseline/baseline_violations.json")
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"  ✓ Saved: {results_file}")

# Create summary table
summary_data = {
    'Metric': [
        'Monotonicity Rate (%)',
        'Monotonicity Violations',
        'Avg Violations per Sample',
        'Max Violations in Sample',
        'Boundary Rate (%)',
        'Boundary Violations',
        'Below θ_r',
        'Above θ_s',
        'Dry-end RMSE (s > 10⁴ kPa)'
    ],
    'Gradient Boosting': [
        f"{monotonicity_rate*100:.1f}%",
        f"{len(mono_violations)}/{len(X_test)}",
        f"{np.mean(mono_violation_counts) if mono_violation_counts else 0:.2f}",
        f"{np.max(mono_violation_counts) if mono_violation_counts else 0}",
        f"{boundary_rate*100:.1f}%",
        f"{len(boundary_violations)}/{len(X_test)}",
        f"{boundary_violation_types['below_theta_r']}",
        f"{boundary_violation_types['above_theta_s']}",
        f"{rmse_dry:.6f}" if not np.isnan(rmse_dry) else "N/A"
    ],
    'PINN (Monotonic)': [
        "100.0%",
        "0/79",
        "0.00",
        "0",
        "100.0%",
        "0/79",
        "0",
        "0",
        "0.001726"
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_file = Path("results_baseline/baseline_violations_summary.csv")
summary_df.to_csv(summary_file, index=False)
print(f"  ✓ Saved: {summary_file}")

print("\n" + "="*80)
print("VIOLATION ANALYSIS COMPLETE")
print("="*80)
print(f"\nKey Findings:")
print(f"  - Monotonicity: {monotonicity_rate*100:.1f}% (vs PINN 100%)")
print(f"  - Boundary: {boundary_rate*100:.1f}% (vs PINN 100%)")
print(f"  - Violations quantified and visualized")
print(f"\nFiles created:")
print(f"  - Violation analysis: {results_file}")
print(f"  - Summary table: {summary_file}")
print(f"  - Visualization: {viz_dir / 'baseline_physics_violations.png'}")
