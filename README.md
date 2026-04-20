# SWCC Prediction using Physics-Informed Machine Learning

This repository contains the processed data, trained model weights, evaluation scripts, and figure-generation code for the paper:

> **"Physics-Informed Machine Learning for Soil Water Characteristic Curve Prediction"**
> *(manuscript under review)*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

---

## Repository Structure

```
.
├── data/processed/          # Train/val/test splits (features + SWCC curves)
├── weights/                 # Trained model weights (.keras format)
├── results/
│   ├── metrics/             # Evaluation JSON files (RMSE, MAE, R², etc.)
│   └── tables/              # CSV comparison and per-sample tables
├── models/                  # Model architecture definitions
│   ├── pinn_monotonic.py    # MonotonicPINN (Physics-Informed Neural Network)
│   ├── vg_param_net.py      # VGParamNet (Van Genuchten parameter predictor)
│   ├── wgan_gp.py           # WGAN-GP for synthetic data generation
│   └── physics_constraints.py
├── training/                # GAN training scripts and config
├── training_pinn/           # PINN and VGParamNet training scripts
├── scripts/
│   ├── data_preprocessing/  # UNSODA extraction and preprocessing pipelines
│   ├── evaluation/          # Model evaluation and comparison scripts
│   ├── figure_generation/   # Scripts to reproduce all paper figures
│   ├── simulation/          # Richards equation solver and benchmark
│   └── training/            # Training monitoring utilities
├── analysis/                # Van Genuchten fitting, knee-point analysis
├── configs/                 # Training configuration files
├── baseline_models.py       # Random Forest, GB, SVM, MLP baselines
├── generate_synthetic_data.py
├── requirements_gan.txt
├── DATA_SOURCES.md          # Data provenance, download instructions
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install tensorflow==2.12 tensorflow-probability==0.20 \
    numpy pandas scipy scikit-learn matplotlib seaborn tqdm pyyaml
```

### 2. Load processed data

```python
import numpy as np
import pandas as pd

X_train = pd.read_csv("data/processed/X_train.csv")
y_train = np.load("data/processed/y_train.npy")        # shape: (N, 100)
suction  = np.load("data/processed/suction_grid.npy")  # 100 suction points (kPa)
```

### 3. Load the trained model and predict

```python
import tensorflow as tf

model = tf.keras.models.load_model("weights/pinn_best_model_fixed.keras")
X_test = pd.read_csv("data/processed/X_test.csv").values
predictions = model.predict(X_test)   # shape: (N, 100), volumetric water content
```

### 4. Reproduce evaluation results

```bash
python scripts/evaluation/evaluate_pinn_comprehensive.py
python scripts/evaluation/evaluate_and_compare_models.py
```

### 5. Reproduce paper figures

```bash
python scripts/figure_generation/generate_figure8_q1.py   # e.g. Figure 8
# All figure scripts follow the same pattern
```

---

## Models

| Model | File | Description |
|---|---|---|
| **MonotonicPINN** (main) | `weights/pinn_best_model_fixed.keras` | Physics-informed net with structural monotonicity |
| MonotonicPINN (final epoch) | `weights/pinn_final_model_fixed.keras` | Same training run, last epoch |
| **WGAN-GP** | `weights/gan_final_model.keras` | Conditional GAN for synthetic SWCC augmentation |

---

## Data

The processed data splits in `data/processed/` were derived from the **UNSODA 2.0** database and the **Hohenbrink (2023)** dataset. See `DATA_SOURCES.md` for full provenance, licensing, and download instructions for the original raw data.

| File | Description | Shape |
|---|---|---|
| `X_train/val/test.csv` | 16 soil features | (N, 16) |
| `y_train/val/test.npy` | SWCC curves (θ at 100 suction points) | (N, 100) |
| `suction_grid.npy` | Suction values in kPa (log-spaced 0.1–1000) | (100,) |
| `metadata.json` | Feature names, units, statistics | — |

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{SWCC_ML_2026,
  title   = {Physics-Informed Machine Learning for Soil Water Characteristic Curve Prediction},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {2026},
  doi     = {[DOI]}
}
```

---

## License

Code: MIT License
Data (processed splits): CC BY 4.0 — derived from UNSODA 2.0 (see DATA_SOURCES.md for original license)
