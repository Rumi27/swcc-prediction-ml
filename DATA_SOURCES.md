# Data Sources and Provenance

This document describes the origin, licensing, and download instructions for all data used in this project.

---

## 1. UNSODA 2.0 (Primary Training Dataset)

**Full name:** Unsaturated Soil Hydraulic Database, Version 2.0
**Provider:** U.S. Salinity Laboratory, USDA-ARS
**Citation:**
> Nemes, A., Schaap, M.G., Leij, F.J., & Wösten, J.H.M. (2001). Description of the unsaturated soil hydraulic database UNSODA version 2.0. *Journal of Hydrology*, 251(3–4), 151–162. https://doi.org/10.1016/S0022-1694(01)00465-6

**Download:** https://www.ars.usda.gov/pacific-west-area/riverside-ca/us-salinity-laboratory/docs/unsoda-database/

**License:** Publicly available for research use. Please cite the original publication when using.

**How it was used:**
- Starting database: ~790 soil samples with measured SWCC and soil property data
- After quality filtering (monotonicity checks, minimum 5 measured SWCC points, complete feature set): **320 samples** retained
- Preprocessing pipeline: `scripts/data_preprocessing/extract_unsoda_data.py` → `phase1_preprocessing.py`
- Train/val/test split: approximately 200 / 50 / 70 samples (stratified by soil texture class)

**Features extracted (16 total):**
D10, D30, D50, D60, D90 (grain size percentiles in mm), Cu, Cc, clay%, silt%, sand%, bulk_density (g/cm³), porosity, OM_content (%), pH, theta_s (saturated VWC), theta_r (residual VWC)

**Note:** The raw UNSODA 2.0 database files are NOT included in this repository due to redistribution restrictions. Download directly from the source above and run the preprocessing scripts. The processed train/val/test splits in `data/processed/` are provided here as derived outputs under CC BY 4.0.

---

## 2. Hohenbrink (2023) — External Validation Dataset

**Citation:**
> Hohenbrink, T.L., et al. (2023). Soil hydraulic properties from percolation experiments: Dataset and benchmark. *Earth System Science Data*. https://doi.org/10.5194/essd-2023-XXXXX

**Download:** Available via the linked publication / Zenodo archive in the paper.

**How it was used:**
- External (out-of-distribution) validation only — no training
- Preprocessing: `scripts/data_preprocessing/process_hohenbrink.py`
- Evaluation: `scripts/evaluation/evaluate_hohenbrink.py`

---

## 3. GSHP Dataset — External Validation

**Description:** Ground Source Heat Pump site characterisation data with measured soil water retention curves.

**How it was used:**
- Secondary external validation for domain-shift analysis
- Evaluation: `scripts/evaluation/evaluate_gshp_comprehensive.py`

**Note:** Contact the authors for access to this dataset.

---

## 4. Processed Data Splits (This Repository)

**Location:** `data/processed/`
**License:** CC BY 4.0

These files are derived outputs from the preprocessing pipeline applied to UNSODA 2.0. They contain:
- Normalised feature matrices (X_train/val/test.csv)
- SWCC target arrays (y_train/val/test.npy) — volumetric water content (m³/m³) at 100 log-spaced suction points
- Suction grid (suction_grid.npy) — matric potential in kPa, log-spaced from 0.1 to 1000 kPa
- Metadata (metadata.json) — feature names, units, statistics, split sizes

---

## Reproducing the Full Preprocessing Pipeline

To reproduce the processed splits from scratch:

```bash
# Step 1: Download UNSODA 2.0 and place in data/raw/UNSODA_2.0/
python scripts/data_preprocessing/extract_unsoda_data.py
python scripts/data_preprocessing/process_unsoda_data.py

# Step 2: Quality filter and feature engineering
python scripts/data_preprocessing/phase1_preprocessing.py

# Step 3 (optional): Prepare PINN-normalised data
python scripts/data_preprocessing/prepare_pinn_data.py
python scripts/data_preprocessing/prepare_pinn_data_normalized.py
```

Output will match the files in `data/processed/` exactly (random seed is fixed in the preprocessing scripts).
