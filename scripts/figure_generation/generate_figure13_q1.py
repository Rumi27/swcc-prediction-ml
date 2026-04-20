#!/usr/bin/env python3
"""
Figure 13 — Error vs Suction and Dry-End Analysis
Q1 journal quality (single-column width 3.5 in, vertically stacked 3 panels).

Layout: 3 rows × 1 column
  (a) Mean |error| vs matric suction ψ  (all three models, full log range)
  (b) Mean |error| in dry-end regime (ψ > 10 000 kPa)
  (c) Fraction of total |error| by suction range (grouped bar chart)

All three models included: Gradient Boosting | VGParamNet (Run B) | MonotonicPINN
Data source: latest trained models (Run B VGParamNet, best PINN keras checkpoint,
             GB retrained from data_processed).

Design
------
* 7.0 in wide × 9.0 in tall (3 × 3.0 in rows)
* Arial 12 pt: axis labels, y tick labels, panel tags, panel (c) x tick labels
* Arial 11 pt: panel (a)(b) x tick labels (slightly smaller to reduce overlap)
* Arial 10 pt: legend (no frame); panel (b) legend upper left, (a)(c) upper right
* Colours: GB=#1F77B4 (blue), VGParamNet=#2CA02C (green), PINN=#D62728 (red)
* Line styles: GB dashed (7,3), VGParamNet dashed (3,3), PINN solid
* Inward ticks mirrored top/right; no grid; clean box
* 600 dpi PNG (→ paper_figures/png/) + PDF (→ paper_figures/)
* pdf.fonttype=42
"""

from __future__ import annotations
import json as _json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT       = Path(__file__).resolve().parents[2]
PNG_DIR    = ROOT / "paper_figures" / "png"
PDF_DIR    = ROOT / "paper_figures"
PNG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

FONT_MAIN  = 12
FONT_SMALL = 10
LW         = 1.8

matplotlib.rcParams.update({
    "text.usetex": False,
    "axes.formatter.use_mathtext": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "font.size": FONT_MAIN,
    "axes.labelsize": FONT_MAIN,
    "xtick.labelsize": FONT_MAIN,
    "ytick.labelsize": FONT_MAIN,
    "legend.fontsize": FONT_SMALL,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

# Priority colors: blue, green, red
C_GB   = "#1F77B4"   # blue  — Gradient Boosting
C_VG   = "#2CA02C"   # green — VGParamNet (Run B)
C_PINN = "#D62728"   # red   — MonotonicPINN

XTICKS_FULL  = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
XLABELS_FULL = ["0.1", "1", "10", "100", "1000", "10000", "100000", "1000000"]

XTICKS_DRY  = [1e4, 1e5, 1e6]
XLABELS_DRY = ["10000", "100000", "1000000"]


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------
def _style(ax, *, bottom: bool = True):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("black")
    ax.tick_params(which="both", top=True, right=True, direction="in",
                   labelsize=FONT_MAIN)
    ax.grid(False)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("Arial")
        lbl.set_fontsize(FONT_MAIN)


def _panel_tag(ax, tag):
    """Place panel label OUTSIDE the axes, top-left corner (above the box)."""
    ax.text(0.0, 1.03, tag, transform=ax.transAxes,
            fontsize=FONT_MAIN, fontweight="normal", fontfamily="Arial",
            va="bottom", ha="left", zorder=10, clip_on=False)


def _legend(ax, **kw):
    leg = ax.legend(
        frameon=False,
        borderpad=0.4,
        handlelength=2.2,
        fontsize=FONT_SMALL,
        **kw,
    )
    leg.set_zorder(3)
    for t in leg.get_texts():
        t.set_fontfamily("Arial")
        t.set_fontsize(FONT_SMALL)
    return leg


# ---------------------------------------------------------------------------
# Panel plotters
# ---------------------------------------------------------------------------
def _plot_panel_a(ax, psi, err_gb, err_vg, err_pinn):
    """(a) Mean |error| vs matric suction — full log range."""
    ax.semilogx(psi, err_pinn, color=C_PINN, lw=LW + 0.4, ls="-",
                label="MonotonicPINN")
    ax.semilogx(psi, err_gb, color=C_GB, lw=LW, ls="--",
                dashes=(7, 3), label="Gradient Boosting")
    ax.semilogx(psi, err_vg, color=C_VG, lw=LW, ls="--",
                dashes=(3, 3), label="VGParamNet (Run B)")

    ax.set_ylabel("Mean |error|  [-]", fontsize=FONT_MAIN,
                  fontfamily="Arial", labelpad=6)
    ax.set_xlabel("Matric suction \u03c8  (kPa)", fontsize=FONT_MAIN,
                  fontfamily="Arial", labelpad=6)
    ax.set_xlim([psi.min() * 0.9, psi.max() * 1.1])
    ax.set_xscale("log")
    ax.set_xticks(XTICKS_FULL)
    ax.set_xticklabels(XLABELS_FULL, fontsize=FONT_MAIN - 1, fontfamily="Arial",
                       rotation=0, ha="center")
    _style(ax)
    _panel_tag(ax, "(a) Mean |error| vs matric suction")
    _legend(ax, loc="upper right")


def _plot_panel_b(ax, psi_dry, err_gb_dry, err_vg_dry, err_pinn_dry, xmin_full, xmax_full):
    """(b) Mean |error| dry-end (ψ > 10 000 kPa) — same x-axis scale as panel (a)."""
    ax.semilogx(psi_dry, err_pinn_dry, color=C_PINN, lw=LW + 0.4, ls="-",
                label="MonotonicPINN")
    ax.semilogx(psi_dry, err_gb_dry, color=C_GB, lw=LW, ls="--",
                dashes=(7, 3), label="Gradient Boosting")
    ax.semilogx(psi_dry, err_vg_dry, color=C_VG, lw=LW, ls="--",
                dashes=(3, 3), label="VGParamNet (Run B)")

    ax.set_ylabel("Mean |error|  [-]", fontsize=FONT_MAIN,
                  fontfamily="Arial", labelpad=6)
    ax.set_xlabel("Matric suction \u03c8  (kPa)", fontsize=FONT_MAIN,
                  fontfamily="Arial", labelpad=6)
    # same full range as panel (a)
    ax.set_xlim([xmin_full, xmax_full])
    ax.set_xscale("log")
    ax.set_xticks(XTICKS_FULL)
    ax.set_xticklabels(XLABELS_FULL, fontsize=FONT_MAIN - 1, fontfamily="Arial",
                       rotation=0, ha="center")
    _style(ax)
    _panel_tag(ax, "(b) Dry-end regime  (\u03c8 > 10 000 kPa)")
    _legend(ax, loc="upper left")


def _plot_panel_c(ax, gb_frac, vg_frac, pinn_frac, bin_labels):
    """(c) Fraction of total |error| by suction range — grouped bar chart."""
    x = np.arange(len(bin_labels))
    w = 0.25

    ax.bar(x - w, pinn_frac, w, color=C_PINN, label="MonotonicPINN",
           edgecolor="white", linewidth=0.5)
    ax.bar(x,     gb_frac,   w, color=C_GB,   label="Gradient Boosting",
           edgecolor="white", linewidth=0.5)
    ax.bar(x + w, vg_frac,   w, color=C_VG,   label="VGParamNet (Run B)",
           edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=FONT_MAIN, fontfamily="Arial")
    ax.set_xlabel("Suction range (kPa)", fontsize=FONT_MAIN,
                  fontfamily="Arial", labelpad=6)
    ax.set_ylabel("Fraction of total |error|  [-]", fontsize=FONT_MAIN,
                  fontfamily="Arial", labelpad=6)
    ax.set_ylim([0, max(gb_frac.max(), vg_frac.max(), pinn_frac.max()) * 1.20])
    _style(ax)
    # No top/right tick mirrors on bar chart (looks odd)
    ax.tick_params(top=False, right=False)
    _panel_tag(ax, "(c) Error contribution by suction range")
    _legend(ax, loc="upper right")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    from training_pinn.config_pinn_fixed import DATA_CONFIG

    # ---- Load test data ----
    X_test  = pd.read_csv(DATA_CONFIG["test_file"])
    y_true  = np.load(DATA_CONFIG["y_test_original_file"]).astype(np.float32)
    psi     = np.load(DATA_CONFIG["suction_grid_file"]).astype(np.float32)

    # ---- VGParamNet Run B ----
    y_vg = np.load(
        ROOT / "results_pinn_fixed/vgparamnet/theta_vgparamnet_test.npy"
    ).astype(np.float32)
    print(f"VGParamNet RMSE (Run B): {float(np.sqrt(np.mean((y_true - y_vg)**2))):.4f}")

    # ---- Gradient Boosting (retrain) ----
    print("Training Gradient Boosting...")
    from baseline_models import BaselineModels
    bm = BaselineModels(data_dir="data_processed", output_dir="results_baseline")
    (Xtr, Xva, Xte), (ytr, yva, yte), _ = bm.load_data()
    Xtr_f, Xva_f, Xte_f, _ = bm.prepare_features(Xtr, Xva, Xte)
    gb_models = bm.train_gradient_boosting(Xtr_f, ytr, Xva_f, yva)
    y_gb = bm.predict_swcc(gb_models, Xte_f, model_type="gradient_boosting",
                           n_points=y_true.shape[1]).astype(np.float32)
    print(f"GB RMSE: {float(np.sqrt(np.mean((y_true - y_gb)**2))):.4f}")

    # ---- MonotonicPINN ----
    print("Loading MonotonicPINN...")
    import tensorflow as tf
    from models.pinn_monotonic import MonotonicPINN
    from models.pinn import PhysicsEncodingLayer
    meta      = _json.load(open(DATA_CONFIG["metadata_file"]))
    feat_cols = meta["feature_cols"]

    pinn_model = MonotonicPINN(
        soil_prop_dim=meta["n_features"],
        suction_points=meta["n_swcc_points"],
        physics_units=128, hidden_dims=[128, 256, 128, 64])
    pinn_model({"soil_props": np.random.randn(1, meta["n_features"]).astype(np.float32),
                "suction":    np.random.randn(1, meta["n_swcc_points"]).astype(np.float32)})
    saved = tf.keras.models.load_model(
        str(ROOT / "results_pinn_fixed/checkpoints/pinn_best_model_fixed.keras"),
        custom_objects={"MonotonicPINN": MonotonicPINN,
                        "PhysicsEncodingLayer": PhysicsEncodingLayer},
        compile=False)
    pinn_model.set_weights(saved.get_weights())

    y_norm = []
    for i in range(0, len(X_test), 32):
        j = min(i + 32, len(X_test))
        inp = {"soil_props": X_test.iloc[i:j][feat_cols].values.astype(np.float32),
               "suction":    np.tile(psi, (j - i, 1)).astype(np.float32)}
        y_norm.extend(pinn_model(inp, training=False).numpy())
    y_norm = np.array(y_norm, dtype=np.float32)
    ts = X_test["theta_s"].values.astype(np.float32)
    tr = X_test["theta_r"].values.astype(np.float32)
    y_pinn = np.zeros_like(y_norm)
    for i in range(len(X_test)):
        y_pinn[i] = tr[i] + y_norm[i] * (ts[i] - tr[i])
    print(f"PINN RMSE: {float(np.sqrt(np.mean((y_true - y_pinn)**2))):.4f}")

    # ---- Per-point mean absolute errors ----
    mean_err_gb   = np.mean(np.abs(y_true - y_gb),   axis=0)
    mean_err_vg   = np.mean(np.abs(y_true - y_vg),   axis=0)
    mean_err_pinn = np.mean(np.abs(y_true - y_pinn), axis=0)

    # ---- Dry-end (ψ > 10 000 kPa) ----
    dry_mask = psi > 1e4
    if dry_mask.any():
        psi_dry       = psi[dry_mask]
        err_gb_dry    = mean_err_gb[dry_mask]
        err_vg_dry    = mean_err_vg[dry_mask]
        err_pinn_dry  = mean_err_pinn[dry_mask]
    else:
        psi_dry = err_gb_dry = err_vg_dry = err_pinn_dry = None
        print("WARNING: no suction points > 10 000 kPa — panel (b) will be empty.")

    # ---- Error fractions by suction bin ----
    bins        = [0, 1e2, 1e3, 1e4, psi.max() * 1.01]
    bin_labels  = ["<100", "100-1000", "1000-10000", ">10000"]
    gb_frac, vg_frac, pinn_frac = [], [], []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (psi >= b0) & (psi < b1)
        gb_frac.append(  np.nansum(np.abs(y_true - y_gb)[:,   m]) if m.any() else 0.)
        vg_frac.append(  np.nansum(np.abs(y_true - y_vg)[:,   m]) if m.any() else 0.)
        pinn_frac.append(np.nansum(np.abs(y_true - y_pinn)[:, m]) if m.any() else 0.)
    gb_frac   = np.array(gb_frac,   dtype=np.float32)
    vg_frac   = np.array(vg_frac,   dtype=np.float32)
    pinn_frac = np.array(pinn_frac, dtype=np.float32)
    gb_frac   /= gb_frac.sum()
    vg_frac   /= vg_frac.sum()
    pinn_frac /= pinn_frac.sum()

    print("Error fractions by bin:")
    for lbl, gv, vv, pv in zip(bin_labels, gb_frac, vg_frac, pinn_frac):
        print(f"  {lbl:12s}  GB={gv:.3f}  VG={vv:.3f}  PINN={pv:.3f}")

    # ----------------------------------------------------------------
    # Build figure: 3 rows × 1 column, vertically stacked
    # ----------------------------------------------------------------
    FIG_W  = 7.0
    ROW_H  = 3.0
    HSPACE = 0.55   # vertical gap — room for x-labels + outside panel tags above

    fig, axes = plt.subplots(3, 1,
                             figsize=(FIG_W, ROW_H * 3),
                             gridspec_kw=dict(hspace=HSPACE))

    xmin_full = float(psi.min()) * 0.9
    xmax_full = float(psi.max()) * 1.1

    _plot_panel_a(axes[0], psi, mean_err_gb, mean_err_vg, mean_err_pinn)

    if psi_dry is not None:
        _plot_panel_b(axes[1], psi_dry, err_gb_dry, err_vg_dry, err_pinn_dry,
                      xmin_full, xmax_full)
    else:
        axes[1].set_axis_off()
        axes[1].text(0.5, 0.5, "No data above 10 000 kPa",
                     ha="center", va="center", transform=axes[1].transAxes,
                     fontsize=FONT_MAIN, fontfamily="Arial")
        _panel_tag(axes[1], "(b) Dry-end regime  (\u03c8 > 10 000 kPa)")

    _plot_panel_c(axes[2], gb_frac, vg_frac, pinn_frac, bin_labels)

    fig.align_ylabels(axes)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.97, bottom=0.06)

    stem = "Figure13_Error_vs_Suction_and_DryEnd_q1"
    fig.savefig(str(PDF_DIR / stem) + ".pdf", dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(str(PNG_DIR / stem) + ".png", dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved:\n  {PDF_DIR / stem}.pdf\n  {PNG_DIR / stem}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
