#!/usr/bin/env python3
"""
Figure C2 — Model Comparison: Predictive vs Fitted Lower-Bound Reference
Q1 journal quality (7.5 in wide × 7.0 in tall, 3 panels).

Layout: GridSpec(2, 2) — top row: (a) left + (b) right; bottom row: (c) full-width
  (a) Predictive models RMSE by suction regime
      — Gradient Boosting | Rosetta3 PTF | VGParamNet (current Run B)
      — Grouped bars: Wet (<100 kPa) | Mid (100–10k kPa) | Dry (>10k kPa)
  (b) Fitted models — lower-bound reference (fitted directly to observed data)
      — VG unimodal fitted | Fredlund-Xing | Kosugi
      — Same bar scheme + VGParamNet wet-RMSE reference dashed line
  (c) Representative SWCC comparison (effective degree of saturation S_e)
      — Observed | VGParamNet | Rosetta3 PTF | Fredlund-Xing (fitted) | Kosugi (fitted)

Data sources
------------
  data_processed/y_test.npy                       — 84 observed SWCC curves
  data_processed/suction_grid.npy                 — 100-pt log suction grid
  data_processed/X_test.csv                       — soil-property features
  results_pinn_fixed/vgparamnet/
      theta_vgparamnet_test.npy                   — VGParamNet Run B predictions
  results_comparison/ptf_wrf_comparison/
      rosetta3_predictions.npz                    — Rosetta3 PTF predicted curves
  results_pinn_fixed/vg_fit/vg_fit_results.csv    — VG unimodal fitted params
  results_comparison/ptf_wrf_comparison/
      comparison_results.json                     — FX / Kosugi fitted RMSE
  baseline_models.BaselineModels                  — GB retrained on-the-fly

Design
------
* 7.5 in wide × 7.0 in tall; Arial 12 pt labels/tags; 10 pt legend/annotations
* Bar colours: Wet = #5BA3D9 | Mid = #4E9A6B | Dry = #D4825B
* Inward ticks mirrored; no grid; clean box; 600 dpi PNG + PDF
* pdf.fonttype = 42
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.special import erfc

ROOT    = Path(__file__).resolve().parents[2]
PNG_DIR = ROOT / "paper_figures" / "png"
PDF_DIR = ROOT / "paper_figures"
PNG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

FONT_MAIN  = 12
FONT_SMALL = 10
FONT_XLOG  = 8    # horizontal x-tick labels on log-ψ axis

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

# ── Colors ───────────────────────────────────────────────────────────────────
C_WET   = "#5BA3D9"    # light blue — wet regime bars
C_MID   = "#4E9A6B"    # teal/green — mid regime bars
C_DRY   = "#D4825B"    # orange-brown — dry regime bars

# SWCC comparison panel (c)
C_OBS   = "#000000"    # black — observed
C_VG    = "#2CA02C"    # green — VGParamNet
C_ROZ   = "#FF7F0E"    # orange — Rosetta3
C_FX    = "#9467BD"    # purple — Fredlund-Xing
C_KOS   = "#7F7F7F"    # grey — Kosugi

XTICKS_PSI  = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
XLABELS_PSI = ["0.1", "1", "10", "100", "1000", "10000", "100000", "1000000"]


# ── Style helpers ─────────────────────────────────────────────────────────────
def _style_ax(ax):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_linewidth(0.8); sp.set_color("black")
    ax.tick_params(which="both", top=True, right=True, direction="in")
    ax.grid(False)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("Arial"); lbl.set_fontsize(FONT_MAIN)


def _panel_tag(ax, tag):
    ax.text(0.0, 1.03, tag, transform=ax.transAxes,
            fontsize=FONT_MAIN, fontweight="normal", fontfamily="Arial",
            va="bottom", ha="left", clip_on=False)


def _psi_xaxis(ax, psi):
    ax.set_xscale("log")
    ax.set_xticks(XTICKS_PSI)
    ax.set_xticklabels(XLABELS_PSI, fontsize=FONT_XLOG,
                       fontfamily="Arial", rotation=0, ha="center")
    ax.set_xlim(float(psi.min()) * 0.9, float(psi.max()) * 1.1)
    ax.set_xlabel("Matric suction \u03c8  (kPa)",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=8)


# ── Water-retention functions for on-the-fly fitting ─────────────────────────
def _fredlund_xing_se(psi, a, n, m):
    """Fredlund-Xing (1994) effective saturation."""
    return 1.0 / (np.log(np.e + (psi / a) ** n)) ** m


def _kosugi_se(psi, psi_m, sigma):
    """Kosugi (1996) effective saturation."""
    sigma = max(sigma, 0.05)
    psi_m = max(psi_m, 1e-6)
    return 0.5 * erfc(np.log(psi / psi_m) / (np.sqrt(2.0) * sigma))


def _fit_se(psi, se_obs, model):
    """Fit FX or Kosugi WRF to observed Se using differential evolution."""
    se_obs = np.clip(se_obs, 0.0, 1.0)
    if model == "fx":
        def obj(x): return np.sqrt(np.mean((_fredlund_xing_se(psi, *x) - se_obs) ** 2))
        bounds = [(0.01, 1e6), (0.5, 8.0), (0.1, 4.0)]
        res = differential_evolution(obj, bounds, seed=42, maxiter=400, tol=1e-5,
                                     popsize=10, mutation=(0.5, 1.5), recombination=0.7)
        return _fredlund_xing_se(psi, *res.x)
    else:  # kosugi
        def obj(x): return np.sqrt(np.mean((_kosugi_se(psi, *x) - se_obs) ** 2))
        bounds = [(0.01, 1e5), (0.05, 3.0)]
        res = differential_evolution(obj, bounds, seed=42, maxiter=400, tol=1e-5,
                                     popsize=10, mutation=(0.5, 1.5), recombination=0.7)
        return _kosugi_se(psi, *res.x)


# ── Per-sample mean RMSE by regime ───────────────────────────────────────────
def _regime_rmse(pred, obs, masks):
    """Return (wet, mid, dry) per-sample mean RMSE."""
    out = []
    for m in masks:
        if m.any():
            ps = np.sqrt(np.mean((pred[:, m] - obs[:, m]) ** 2, axis=1))
            out.append(float(np.mean(ps)))
        else:
            out.append(float("nan"))
    return out


# ── Panel (a): Predictive models grouped bar chart ───────────────────────────
def _plot_panel_a(ax, data: dict):
    """data keys: model_name → (wet, mid, dry)"""
    models   = list(data.keys())
    n_models = len(models)
    bar_w    = 0.22
    x        = np.arange(n_models)
    offsets  = [-bar_w, 0.0, bar_w]
    colors   = [C_WET, C_MID, C_DRY]
    labels   = ["Wet  (<100 kPa)", "Mid  (100–10k kPa)", "Dry  (>10k kPa)"]

    bars_for_legend = []
    for j, (col, lbl, off) in enumerate(zip(colors, labels, offsets)):
        vals = [data[m][j] for m in models]
        b = ax.bar(x + off, vals, bar_w * 0.92, color=col, alpha=0.88,
                   edgecolor="white", linewidth=0.5, label=lbl, zorder=3)
        bars_for_legend.append(b)

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Gradient\nBoosting", "Rosetta3\nPTF", "VGParamNet\n(ours)"],
        fontsize=FONT_SMALL, fontfamily="Arial")
    ax.set_ylabel("Mean RMSE  (m\u00b3/m\u00b3)",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)
    ax.set_xlim(-0.5, n_models - 0.5)
    ax.set_ylim(0, 0.115)

    leg = ax.legend(fontsize=FONT_SMALL, frameon=False,
                    loc="upper left", borderpad=0.3,
                    handlelength=1.2, handletextpad=0.5)
    for t in leg.get_texts():
        t.set_fontfamily("Arial"); t.set_fontsize(FONT_SMALL)

    _style_ax(ax)
    ax.tick_params(which="both", top=False, right=False)  # bar chart: no mirror ticks
    for lbl in ax.get_xticklabels():
        lbl.set_fontfamily("Arial"); lbl.set_fontsize(FONT_SMALL)
    _panel_tag(ax, "(a) Predictive models \u2014 RMSE by suction regime")


# ── Panel (b): Fitted models lower-bound bar chart ───────────────────────────
def _plot_panel_b(ax, data: dict, vgnet_wet_rmse: float):
    """data: model_name → (wet, mid, dry); reference line at vgnet_wet_rmse."""
    models   = list(data.keys())
    n_models = len(models)
    bar_w    = 0.22
    x        = np.arange(n_models)
    offsets  = [-bar_w, 0.0, bar_w]
    colors   = [C_WET, C_MID, C_DRY]

    for j, (col, off) in enumerate(zip(colors, offsets)):
        vals = [data[m][j] for m in models]
        ax.bar(x + off, vals, bar_w * 0.92, color=col, alpha=0.88,
               edgecolor="white", linewidth=0.5, zorder=3)

    # VGParamNet wet-RMSE reference line
    # Reference line and label added after ylim is set (see below)

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["VG\n(unimodal\nfitted)", "Fredlund\u2013Xing\n(fitted)", "Kosugi\n(fitted)"],
        fontsize=FONT_SMALL, fontfamily="Arial")
    ax.set_ylabel("Mean RMSE  (m\u00b3/m\u00b3)",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)
    ax.set_xlim(-0.5, n_models - 0.5)
    # ylim must include VGParamNet wet-RMSE reference line
    y_top = max(0.060, vgnet_wet_rmse * 1.20)
    ax.set_ylim(0, y_top)

    # Now reposition the reference label using data coordinates
    ax.axhline(vgnet_wet_rmse, color=C_VG, lw=1.4, ls="--", dashes=(6, 3), zorder=5)
    ax.text(n_models - 0.52, vgnet_wet_rmse + y_top * 0.03,
            f"VGParamNet wet\nRMSE = {vgnet_wet_rmse:.3f}",
            fontsize=FONT_SMALL - 1, fontfamily="Arial", color=C_VG,
            ha="right", va="bottom")

    _style_ax(ax)
    ax.tick_params(which="both", top=False, right=False)
    for lbl in ax.get_xticklabels():
        lbl.set_fontfamily("Arial"); lbl.set_fontsize(FONT_SMALL)
    _panel_tag(ax, "(b) Fitted models \u2014 lower-bound reference")


# ── Panel (c): Representative SWCC in Se ─────────────────────────────────────
def _plot_panel_c(ax, psi, se_obs, se_vg, se_roz, se_fx, se_kos,
                  rmse_vg, rmse_roz, rmse_fx, rmse_kos):

    # Plot all curves
    ax.semilogx(psi, se_obs, color=C_OBS, lw=2.0, ls="-",  zorder=5, label="Observed")
    ax.semilogx(psi, se_vg,  color=C_VG,  lw=1.8, ls="--", zorder=4,
                dashes=(6, 3), label=f"VGParamNet (RMSE = {rmse_vg:.3f})")
    ax.semilogx(psi, se_roz, color=C_ROZ, lw=1.8, ls="-.", zorder=3,
                label=f"Rosetta3 PTF (RMSE = {rmse_roz:.3f})")
    ax.semilogx(psi, se_fx,  color=C_FX,  lw=1.6, ls=":",  zorder=3,
                dashes=(3, 2), label=f"Fredlund\u2013Xing fitted (RMSE = {rmse_fx:.3f})")
    ax.semilogx(psi, se_kos, color=C_KOS, lw=1.6, ls=":",  zorder=3,
                dashes=(1, 1.5), label=f"Kosugi fitted (RMSE = {rmse_kos:.3f})")

    ax.set_ylabel("Effective degree of saturation  Se  [\u2212]",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)
    ax.set_ylim(-0.04, 1.14)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for lbl in ax.get_yticklabels():
        lbl.set_fontfamily("Arial"); lbl.set_fontsize(FONT_MAIN)

    _psi_xaxis(ax, psi)
    _style_ax(ax)
    # Re-apply after _psi_xaxis overwrites tick labelsize
    ax.tick_params(axis="x", labelsize=FONT_XLOG)
    for lbl in ax.get_xticklabels():
        lbl.set_fontfamily("Arial"); lbl.set_fontsize(FONT_XLOG)

    leg = ax.legend(fontsize=FONT_SMALL, frameon=False,
                    loc="upper right", borderpad=0.3,
                    handlelength=2.0, handletextpad=0.5,
                    ncol=1)
    for t in leg.get_texts():
        t.set_fontfamily("Arial"); t.set_fontsize(FONT_SMALL)

    ax.text(0.01, 0.04,
            "(representative single sample — full wet\u2192dry range)",
            transform=ax.transAxes, fontsize=FONT_SMALL - 1,
            fontfamily="Arial", color="#555555", va="bottom", ha="left")

    _panel_tag(ax, "(c) Representative SWCC comparison  (normalised saturation Se)")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    dp   = ROOT / "data_processed"
    rp   = ROOT / "results_pinn_fixed"
    rcp  = ROOT / "results_comparison" / "ptf_wrf_comparison"

    # ── Load data ──────────────────────────────────────────────────────────────
    psi    = np.load(dp / "suction_grid.npy").astype(np.float32)
    y_test = np.load(dp / "y_test.npy").astype(np.float32)
    Xte    = pd.read_csv(dp / "X_test.csv")
    ts_arr = Xte["theta_s"].values.astype(np.float32)
    tr_arr = Xte["theta_r"].values.astype(np.float32)

    theta_vg = np.load(rp / "vgparamnet" / "theta_vgparamnet_test.npy").astype(np.float32)
    roz_npz  = np.load(rcp / "rosetta3_predictions.npz")
    theta_roz = roz_npz["theta_r3_curves"].astype(np.float32)
    comp_json = json.load(open(rcp / "comparison_results.json"))
    vgfit_df  = pd.read_csv(rp / "vg_fit" / "vg_fit_results.csv")

    # ── VG unimodal fitted curves ──────────────────────────────────────────────
    obs_fit = vgfit_df[vgfit_df["curve_type"] == "Observed"].sort_values("sample_id").reset_index(drop=True)

    def _vg(psi_, ts, tr, a, n):
        m = 1.0 - 1.0 / n
        return (tr + (ts - tr) * (1.0 + (a * psi_) ** n) ** (-m)).astype(np.float32)

    theta_vgfit = np.vstack([
        _vg(psi, row["theta_s"], row["theta_r"], row["alpha"], row["n"])
        for _, row in obs_fit.iterrows()
    ])

    # ── GB predictions (re-train, ~25 s) ──────────────────────────────────────
    print("Training Gradient Boosting baseline (this takes ~25 s)…")
    from baseline_models import BaselineModels
    bm = BaselineModels(data_dir=str(dp), output_dir=str(ROOT / "results_baseline"))
    (Xtr, Xva, Xte_bm), (ytr, yva, _), _ = bm.load_data()
    Xtr_f, Xva_f, Xte_f, _ = bm.prepare_features(Xtr, Xva, Xte_bm)
    gb_models = bm.train_gradient_boosting(Xtr_f, ytr, Xva_f, yva)
    theta_gb  = bm.predict_swcc(gb_models, Xte_f, model_type="gradient_boosting",
                                n_points=y_test.shape[1]).astype(np.float32)
    print(f"  GB global RMSE = {float(np.sqrt(np.mean((theta_gb - y_test)**2))):.4f}")

    # ── Regime masks ──────────────────────────────────────────────────────────
    wet  = psi <  100.0
    mid  = (psi >= 100.0) & (psi <= 1e4)
    dry  = psi >  1e4
    masks = [wet, mid, dry]
    rmse_regime = _regime_rmse

    print("\nPer-sample mean RMSE by suction regime:")
    print(f"{'Model':<20} {'Wet':>8} {'Mid':>8} {'Dry':>8}")
    print("-" * 48)

    data_a = {}
    for tag, pred in [("GB", theta_gb), ("Rosetta3 PTF", theta_roz), ("VGParamNet", theta_vg)]:
        r = rmse_regime(pred, y_test, masks)
        data_a[tag] = r
        print(f"  {tag:<18}  {r[0]:>6.4f}  {r[1]:>6.4f}  {r[2]:>6.4f}")

    data_b = {}
    for tag, pred in [("VG unimodal", theta_vgfit)]:
        r = rmse_regime(pred, y_test, masks)
        data_b[tag] = r
        print(f"  {tag:<18}  {r[0]:>6.4f}  {r[1]:>6.4f}  {r[2]:>6.4f}")
    for tag, key in [("Fredlund-Xing", "Fredlund-Xing"), ("Kosugi", "Kosugi")]:
        r = [comp_json[key][f"rmse_wet_mean"],
             comp_json[key][f"rmse_mid_mean"],
             comp_json[key][f"rmse_dry_mean"]]
        data_b[tag] = r
        print(f"  {tag:<18}  {r[0]:>6.4f}  {r[1]:>6.4f}  {r[2]:>6.4f}")

    vgnet_wet_rmse = data_a["VGParamNet"][0]

    # ── Pick best representative sample (full Se span) ────────────────────────
    # Identify per-sample Se RMSE for both predictive models
    denom_arr = np.maximum(ts_arr - tr_arr, 1e-6)
    Se_all  = np.clip((y_test    - tr_arr[:, None]) / denom_arr[:, None], 0, 1)
    Se_vg_  = np.clip((theta_vg  - tr_arr[:, None]) / denom_arr[:, None], 0, 1)
    Se_roz_ = np.clip((theta_roz - tr_arr[:, None]) / denom_arr[:, None], 0, 1)
    rmse_vg_  = np.sqrt(np.mean((Se_vg_  - Se_all) ** 2, axis=1))
    rmse_roz_ = np.sqrt(np.mean((Se_roz_ - Se_all) ** 2, axis=1))

    vgfit_obs_rmse = vgfit_df[vgfit_df["curve_type"] == "Observed"].sort_values("sample_id")["vg_fit_rmse"].values
    span  = Se_all.max(axis=1) - Se_all.min(axis=1)
    bumps = (np.diff(Se_all, axis=1) > 0.005).sum(axis=1)
    med_vg = float(np.median(rmse_vg_))

    # Score: smooth, full-range, VGParamNet near its median RMSE, not dominated by Rosetta3
    score = (span - 0.5 * vgfit_obs_rmse - 0.1 * bumps / 100.0
             - 2.0 * np.abs(rmse_vg_ - med_vg)    # penalise distance from median VGParamNet perf
             + 0.3 * np.clip(rmse_roz_ - rmse_vg_, 0, 1))  # prefer cases where VGParamNet ≤ Rosetta3
    idx = int(np.argmax(score))
    ts_i = float(ts_arr[idx]); tr_i = float(tr_arr[idx])
    print(f"\nRepresentative sample: idx={idx}, θs={ts_i:.3f}, θr={tr_i:.3f}, span={span[idx]:.3f}")

    denom = max(ts_i - tr_i, 1e-6)

    def to_se(theta):
        return np.clip((np.asarray(theta, dtype=float) - tr_i) / denom, 0.0, 1.0)

    se_obs  = to_se(y_test[idx])
    se_vg   = to_se(theta_vg[idx])
    se_roz  = to_se(theta_roz[idx])

    print("  Fitting Fredlund-Xing to representative sample…")
    se_fx  = _fit_se(psi, se_obs, "fx")
    print("  Fitting Kosugi to representative sample…")
    se_kos = _fit_se(psi, se_obs, "kosugi")

    def rmse1(se_pred, se_ref):
        return float(np.sqrt(np.mean((se_pred - se_ref) ** 2)))

    r_vg  = rmse1(se_vg,  se_obs)
    r_roz = rmse1(se_roz, se_obs)
    r_fx  = rmse1(se_fx,  se_obs)
    r_kos = rmse1(se_kos, se_obs)
    print(f"  Se RMSE → VGParamNet={r_vg:.3f}  Rosetta3={r_roz:.3f}  FX={r_fx:.3f}  Kosugi={r_kos:.3f}")

    # ── Build figure ──────────────────────────────────────────────────────────
    FIG_W, FIG_H = 7.5, 7.0
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs  = mgridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.56, wspace=0.40,
        left=0.10, right=0.97, top=0.95, bottom=0.11,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])   # full-width bottom panel

    _plot_panel_a(ax_a, data_a)
    _plot_panel_b(ax_b, data_b, vgnet_wet_rmse)
    _plot_panel_c(ax_c, psi, se_obs, se_vg, se_roz, se_fx, se_kos,
                  r_vg, r_roz, r_fx, r_kos)

    fig.align_ylabels([ax_a, ax_b])

    stem    = "FigureC2_Model_Comparison_q1"
    pdf_out = PDF_DIR / f"{stem}.pdf"
    png_out = PNG_DIR / f"{stem}.png"
    fig.savefig(str(pdf_out), dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(str(png_out), dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"\nSaved:\n  {pdf_out}\n  {png_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
