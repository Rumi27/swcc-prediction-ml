#!/usr/bin/env python3
"""
Fix particle-size distribution (PSD) data in data_processed X_*.csv files.

Issues observed:
- D10, D30, D50 contain small negative values (artefacts of preprocessing)
- This leads to non-physical particle diameters and odd PSD plots

Fixes applied:
1) Clip D10, D30, D50 to a small positive minimum (d_min_mm)
2) Enforce monotone increasing PSD percentiles per row:
   D10 <= D30 <= D50 <= D60 <= D90

The script overwrites X_train.csv, X_val.csv, X_test.csv in data_processed/.
It also creates backup copies with suffix *_psd_backup.csv on first run.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data_processed"

FILES = ["X_train.csv", "X_val.csv", "X_test.csv"]
PSD_COLS = ["D10", "D30", "D50", "D60", "D90"]

# Minimum physical diameter in mm for fine particles (for safety on log scale)
D_MIN_MM = 1e-3  # 0.001 mm = 1 micron


def fix_psd_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean PSD columns in a dataframe."""
    df = df.copy()

    # 1) Clip D10, D30, D50 to be >= D_MIN_MM
    for col in ["D10", "D30", "D50"]:
        if col in df.columns:
            df[col] = df[col].astype(float).clip(lower=D_MIN_MM)

    # 2) Enforce monotone non-decreasing across D10..D90 per row
    cols_present = [c for c in PSD_COLS if c in df.columns]
    if len(cols_present) == len(PSD_COLS):
        psd_values = df[cols_present].to_numpy(dtype=float)
        # cumulative maximum along each row
        psd_fixed = np.maximum.accumulate(psd_values, axis=1)
        df[cols_present] = psd_fixed

    return df


def main():
    print("=" * 80)
    print("Fixing PSD data in data_processed/X_*.csv")
    print("=" * 80)

    for fname in FILES:
        path = DATA_DIR / fname
        if not path.exists():
            print(f"⚠ File not found, skipping: {path}")
            continue

        # Create backup if not already present
        backup_path = DATA_DIR / (path.stem + "_psd_backup.csv")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)
            print(f"  ✓ Backup created: {backup_path.name}")
        else:
            print(f"  • Backup already exists: {backup_path.name}")

        df = pd.read_csv(path)

        # Show basic stats before
        print(f"\nProcessing {fname}:")
        for col in PSD_COLS:
            if col in df.columns:
                arr = df[col].to_numpy(dtype=float)
                print(
                    f"  {col} before: min={arr.min():.5f}, "
                    f"max={arr.max():.5f}, <0 count={(arr < 0).sum()}"
                )

        df_fixed = fix_psd_df(df)

        # Show basic stats after
        for col in PSD_COLS:
            if col in df_fixed.columns:
                arr = df_fixed[col].to_numpy(dtype=float)
                print(
                    f"  {col} after : min={arr.min():.5f}, "
                    f"max={arr.max():.5f}, <0 count={(arr < 0).sum()}"
                )

        # Save back
        df_fixed.to_csv(path, index=False)
        print(f"  ✓ Cleaned file saved: {path.name}")

    print("\nDone. PSD columns are now non-negative and monotone per row.")


if __name__ == "__main__":
    main()

