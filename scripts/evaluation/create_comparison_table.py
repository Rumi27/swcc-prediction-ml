#!/usr/bin/env python3
"""
Create Comparison Table: PINN vs Baseline
"""

import json
import pandas as pd
from pathlib import Path

# Load PINN results
pinn_results_file = Path("results_pinn_fixed/evaluation_results_final.json")
with open(pinn_results_file, 'r') as f:
    pinn_results = json.load(f)

# Load baseline results
baseline_report_file = Path("results_baseline/baseline_report.json")
with open(baseline_report_file, 'r') as f:
    baseline_data = json.load(f)

# Create comparison table
comparison_data = {
    'Model': ['PINN (Monotonic)', 'Gradient Boosting (Baseline)'],
    'Global RMSE': [
        f"{pinn_results['global_metrics']['rmse']:.6f}",
        f"{baseline_data['best_rmse']:.6f}"
    ],
    'Global MAE': [
        f"{pinn_results['global_metrics']['mae']:.6f}",
        f"{baseline_data['best_mae']:.6f}"
    ],
    'Global R²': [
        f"{pinn_results['global_metrics']['r2']:.6f}",
        f"{baseline_data['best_r2']:.6f}"
    ],
    'Median Per-Sample RMSE': [
        f"{pinn_results['per_sample_metrics']['rmse_median']:.6f}",
        "N/A"
    ],
    'Dry-end RMSE (s > 10⁴ kPa)': [
        f"{pinn_results['regime_metrics']['dry_end']['rmse']:.6f}" if pinn_results['regime_metrics']['dry_end']['rmse'] else "N/A",
        "N/A"
    ],
    'Monotonicity (%)': [
        f"{pinn_results['physics_compliance']['monotonicity_rate']*100:.2f}%",
        "Not enforced"
    ],
    'Boundary Satisfaction (%)': [
        f"{pinn_results['physics_compliance']['boundary_satisfaction_rate']*100:.2f}%",
        "Not enforced"
    ]
}

df = pd.DataFrame(comparison_data)

# Save as CSV
output_file = Path("results_pinn_fixed/comparison_table.csv")
df.to_csv(output_file, index=False)
print(f"✓ Saved comparison table: {output_file}")

# Print table
print("\n" + "="*100)
print("COMPARISON TABLE: PINN vs Baseline")
print("="*100)
print(df.to_string(index=False))
print("="*100)

# Save as markdown for paper
md_file = Path("results_pinn_fixed/comparison_table.md")
with open(md_file, 'w') as f:
    f.write("# Model Comparison Table\n\n")
    f.write(df.to_markdown(index=False))
    f.write("\n")
print(f"✓ Saved markdown table: {md_file}")
