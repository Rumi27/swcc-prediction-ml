#!/usr/bin/env python3
"""
Figure S6 — VGParamNet vs PTF and Alternative WRF Models
Q1 journal quality (12.0 in wide × 8.5 in tall, 4 panels).

Layout: 2 rows × 3 columns (GridSpec)
  Row 0:  (a) col 0  — wet-end RMSE boxplots + reference lines
          (b) col 1  — ψ50 log error boxplots
          (c) col 2  — representative SWCC (Observed / VGParamNet / Rosetta3)
  Row 1:  (d) cols 0-2 — mean RMSE by regime for all 5 models

Key distinction (labelled in figure)
--------------------------------------
  Predictive models (from soil properties only, no observed curve used):
    - VGParamNet (Run B)  — this paper
    - Rosetta3            — Zhang & Schaap (2017) PTF baseline
  Descriptive models (curve-fitted directly to observed SWCC data):
    - Fredlund-Xing (1994)
    - Kosugi (1996)
    - Bimodal van Genuchten (Durner 1994)
  The descriptive models provide a lower-bound reference for prediction error
  given the observed data; they do NOT reflect prediction capability.

Data sources
------------
  results_pinn_fixed/vgparamnet/run_B/theta_vgparamnet_test.npy   [84, 100]
  results_comparison/ptf_wrf_comparison/rosetta3_predictions.npz
      key: 'theta_r3_curves'                                       [84, 100]
  results_comparison/ptf_wrf_comparison/comparison_results.json   aggregate
  data_pinn_normalized/y_test_original.npy                        [84, 100]
  data_pinn_normalized/suction_grid.npy                           [100]
  data_pinn_normalized/X_test.csv                                 [84, features]

Design
------
* 12.0 in wide × 8.5 in tall; Arial 12 pt labels; 10 pt legend/annotations
* Inward ticks, mirrored top/right; no grid; white background
* Panel tags at (0.0, 1.03), va='bottom'; frameon=False legends
* PDF + 600 dpi PNG; pdf.fonttype = 42
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT     = Path(__file__).resolve().parents[2]
SUPP_DIR = ROOT / "paper_figures" / "supplementary"
SUPP_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

FONT_MAIN  = 11
FONT_SMALL = 11
FONT_TICK  = 11     # log-axis x-tick labels
FONT_TICK_CAT = 11  # compact categorical x-tick labels for panels (a)(b)

matplotlib.rcParams.update({
    "text.usetex": False,
    "axes.formatter.use_mathtext": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "font.size": FONT_MAIN,
    "axes.labelsize": FONT_MAIN,
    "axes.titlesize": FONT_MAIN,
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

# Colors
C_VG  = "#2CA02C"   # green  — VGParamNet (Run B, matches Fig 11)
C_R3  = "#F18F01"   # amber  — Rosetta3
C_FX  = "#06A77D"   # teal   — Fredlund-Xing
C_KO  = "#C73E1D"   # red    — Kosugi
C_BV  = "#9B59B6"   # purple — Bimodal VG
C_OBS = "#000000"   # black  — Observed

COLORS_ALL = [C_VG, C_R3, C_FX, C_KO, C_BV]
MODEL_NAMES = ["VGParamNet\n(Run B)", "Rosetta3", "Fredlund-\nXing", "Kosugi", "Bimodal\nVG"]

# Log-axis ticks for SWCC panels (6 ticks — no overlap)
XTICKS_PSI  = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e6]
XLABELS_PSI = ["0.1", "1", "10", "100", "1 000", "1 000 000"]


# ── Style helpers ──────────────────────────────────────────────────────────────
def _style(ax, xticks_fs: float = FONT_MAIN, yticks_fs: float = FONT_MAIN):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("black")
    ax.tick_params(which="both", top=True, right=True, direction="in")
    ax.grid(False)
    for lbl in ax.get_xticklabels():
        lbl.set_fontfamily("Arial")
        lbl.set_fontsize(xticks_fs)
    for lbl in ax.get_yticklabels():
        lbl.set_fontfamily("Arial")
        lbl.set_fontsize(yticks_fs)


def _panel_tag(ax, tag: str) -> None:
    ax.text(0.0, 1.03, tag, transform=ax.transAxes,
            fontsize=FONT_MAIN, fontweight="normal", fontfamily="Arial",
            va="bottom", ha="left", clip_on=False)


def _legend(ax, handles=None, **kw):
    kw.setdefault("frameon", False)
    kw.setdefault("borderpad", 0.4)
    kw.setdefault("handlelength", 1.8)
    kw.setdefault("fontsize", FONT_SMALL)
    leg = ax.legend(handles=handles, **kw) if handles else ax.legend(**kw)
    for t in leg.get_texts():
        t.set_fontfamily("Arial")
        t.set_fontsize(FONT_SMALL)
    return leg


def _boxplot_style(bp, color):
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.60)
        patch.set_linewidth(0.8)
    for elem in ["whiskers", "caps"]:
        for line in bp[elem]:
            line.set_linewidth(0.8)
            line.set_color("#444444")
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.4)
    for fl in bp.get("fliers", []):
        fl.set_marker("o")
        fl.set_markersize(3)
        fl.set_markerfacecolor("none")
        fl.set_markeredgewidth(0.6)
        fl.set_markeredgecolor("#888888")


# ── Metric helpers ─────────────────────────────────────────────────────────────
def _wet_rmse(curves_pred: np.ndarray, y_obs: np.ndarray,
              wet_mask: np.ndarray) -> np.ndarray:
    out = np.full(len(curves_pred), np.nan)
    for i in range(len(curves_pred)):
        m = wet_mask & np.isfinite(y_obs[i])
        if m.sum() > 0:
            out[i] = float(np.sqrt(np.mean((curves_pred[i][m] - y_obs[i][m]) ** 2)))
    return out


def _psi50_log_error(curves_pred: np.ndarray, y_obs: np.ndarray,
                     psi: np.ndarray,
                     theta_s: np.ndarray, theta_r: np.ndarray) -> np.ndarray:
    def _p50(theta, ts, tr):
        Se = np.clip((theta - tr) / max(ts - tr, 1e-8), 0.0, 1.0)
        idx = np.where(Se <= 0.5)[0]
        if len(idx) > 0 and idx[0] > 0:
            k = idx[0]
            lp = np.log10(psi)
            t = (0.5 - Se[k - 1]) / (Se[k] - Se[k - 1] + 1e-12)
            return 10 ** (lp[k - 1] + t * (lp[k] - lp[k - 1]))
        return np.nan

    errors = []
    for i in range(len(curves_pred)):
        p50_p = _p50(curves_pred[i], theta_s[i], theta_r[i])
        p50_o = _p50(y_obs[i],       theta_s[i], theta_r[i])
        if np.isfinite(p50_p) and np.isfinite(p50_o) and p50_p > 0 and p50_o > 0:
            errors.append(abs(np.log10(p50_p) - np.log10(p50_o)))
    return np.array(errors)


def _regime_rmse(curves_pred, y_obs, psi):
    """Return (mean_wet, mean_mid, mean_dry) RMSE."""
    wet = psi < 100
    mid = (psi >= 100) & (psi < 1e4)
    dry = psi >= 1e4
    out = {}
    for label, mask in [("wet", wet), ("mid", mid), ("dry", dry)]:
        vals = []
        for i in range(len(curves_pred)):
            m = mask & np.isfinite(y_obs[i])
            if m.sum() > 0:
                vals.append(float(np.sqrt(np.mean(
                    (curves_pred[i][m] - y_obs[i][m]) ** 2))))
        out[label] = float(np.nanmean(vals))
    return out


# ── Panel (a): wet-end RMSE boxplots (VGParamNet & Rosetta3) ──────────────────
def _plot_panel_a(ax, rmse_vg, rmse_r3, agg_json):
    """
    Left pair: boxplots for the two *predictive* models (VGParamNet, Rosetta3).
    Right trio: mean markers for the *descriptive* curve-fit models as reference.
    """
    bp_kw = dict(patch_artist=True, showfliers=True,
                 medianprops=dict(color="black", linewidth=1.4),
                 whiskerprops=dict(linewidth=0.8),
                 capprops=dict(linewidth=0.8),
                 flierprops=dict(marker="o", markersize=3,
                                 markerfacecolor="none", markeredgewidth=0.6,
                                 markeredgecolor="#888888"))

    bp1 = ax.boxplot([rmse_vg], positions=[1], widths=0.55, **bp_kw)
    bp2 = ax.boxplot([rmse_r3], positions=[2], widths=0.55, **bp_kw)
    _boxplot_style(bp1, C_VG)
    _boxplot_style(bp2, C_R3)

    # Reference means for curve-fit models (from saved JSON)
    cx_means = [
        ("Fredlund-Xing\n(fit)", C_FX, agg_json["Fredlund-Xing"]["rmse_wet_mean"]),
        ("Kosugi\n(fit)",        C_KO, agg_json["Kosugi"]["rmse_wet_mean"]),
        ("Bimodal VG\n(fit)",    C_BV, agg_json["Bimodal VG"]["rmse_wet_mean"]),
    ]
    for xpos, (name, col, mean_v) in enumerate(cx_means, start=4):
        ax.plot(xpos, mean_v, "D", color=col, markersize=7,
                zorder=5, markeredgecolor="white", markeredgewidth=0.5)
        ax.axhline(mean_v, color=col, lw=0.6, ls="--",
                   dashes=(4, 4), alpha=0.5, zorder=1, xmin=0.55)

    # Separator between predictive and descriptive
    ax.axvline(3.0, color="#CCCCCC", lw=0.8, ls="--", dashes=(4, 4), zorder=0)
    ax.text(3.03, ax.get_ylim()[1] * 0.97, "curve-fit",
            fontsize=FONT_SMALL - 1, fontfamily="Arial", color="#888888",
            va="top", ha="left")
    ax.text(0.98, ax.get_ylim()[1] * 0.97, "predictive",
            fontsize=FONT_SMALL - 1, fontfamily="Arial", color="#888888",
            va="top", ha="right")

    ax.set_xlim(0.3, 6.7)
    ax.set_xticks([1, 2, 4, 5, 6])
    ax.set_xticklabels(["VGParamNet\n(Run B)", "Rosetta3",
                         "FX\n(fit)", "Kosugi\n(fit)", "Bimodal\n(fit)"],
                       fontsize=FONT_TICK_CAT, fontfamily="Arial",
                       rotation=90, ha="center", va="top")
    ax.tick_params(axis="x", pad=1)
    ax.set_ylabel("Wet-end RMSE  (m\u00b3/m\u00b3)",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)

    # Legend patches
    handles = [Patch(facecolor=C_VG,  alpha=0.60, edgecolor="none",
                     label="VGParamNet (Run B)"),
               Patch(facecolor=C_R3,  alpha=0.60, edgecolor="none",
                     label="Rosetta3 PTF"),
               Line2D([0], [0], marker="D", color="none",
                      markerfacecolor="#555555", markersize=6,
                      label="Curve-fit mean")]
    _legend(ax, handles=handles, loc="upper right")
    _style(ax, xticks_fs=FONT_TICK, yticks_fs=FONT_MAIN)
    _panel_tag(ax, "(a)  Wet-end RMSE  (\u03c8 < 100 kPa)")


# ── Panel (b): ψ50 log error boxplots ─────────────────────────────────────────
def _plot_panel_b(ax, p50_vg, p50_r3, agg_json):
    bp_kw = dict(patch_artist=True, showfliers=True,
                 medianprops=dict(color="black", linewidth=1.4),
                 whiskerprops=dict(linewidth=0.8),
                 capprops=dict(linewidth=0.8),
                 flierprops=dict(marker="o", markersize=3,
                                 markerfacecolor="none", markeredgewidth=0.6,
                                 markeredgecolor="#888888"))

    bp1 = ax.boxplot([p50_vg], positions=[1], widths=0.55, **bp_kw)
    bp2 = ax.boxplot([p50_r3], positions=[2], widths=0.55, **bp_kw)
    _boxplot_style(bp1, C_VG)
    _boxplot_style(bp2, C_R3)

    # Reference means (FX and Kosugi have psi50 error in JSON; Bimodal may not)
    for xpos, (key, name, col) in enumerate(
        [("Fredlund-Xing", "FX (fit)", C_FX),
         ("Kosugi",        "Ko (fit)", C_KO)], start=4):
        if agg_json.get(key, {}).get("psi50_log_error_median") is not None:
            v = agg_json[key]["psi50_log_error_median"]
            ax.plot(xpos, v, "D", color=col, markersize=7,
                    zorder=5, markeredgecolor="white", markeredgewidth=0.5)

    ax.axvline(3.0, color="#CCCCCC", lw=0.8, ls="--", dashes=(4, 4), zorder=0)

    ax.set_xlim(0.3, 5.7)
    ax.set_xticks([1, 2, 4, 5])
    ax.set_xticklabels(["VGParamNet\n(Run B)", "Rosetta3",
                         "FX\n(fit)", "Kosugi\n(fit)"],
                       fontsize=FONT_TICK_CAT, fontfamily="Arial",
                       rotation=90, ha="center", va="top")
    ax.tick_params(axis="x", pad=1)
    ax.set_ylabel("|log10(\u03c850 pred / \u03c850 obs)|  [\u2212]",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)

    handles = [Patch(facecolor=C_VG, alpha=0.60, edgecolor="none",
                     label=f"VGParamNet  (med = {np.median(p50_vg):.2f})"),
               Patch(facecolor=C_R3, alpha=0.60, edgecolor="none",
                     label=f"Rosetta3  (med = {np.median(p50_r3):.2f})")]
    _legend(ax, handles=handles, loc="upper right")
    _style(ax, xticks_fs=FONT_TICK, yticks_fs=FONT_MAIN)
    _panel_tag(ax, "(b)  Knee-location error  (\u03c850)")


# ── Panel (c): representative SWCC ────────────────────────────────────────────
def _pick_sample(y_obs, theta_s, theta_r):
    """Sample with broadest Se span (most informative S-curve)."""
    best_i, best_span = 0, -1.0
    for i in range(len(y_obs)):
        th = y_obs[i]
        if np.sum(np.isfinite(th)) < 20:
            continue
        denom = max(theta_s[i] - theta_r[i], 1e-8)
        Se = np.clip((th - theta_r[i]) / denom, 0.0, 1.0)
        span = float(np.nanmax(Se) - np.nanmin(Se))
        if span > best_span:
            best_span, best_i = span, i
    return best_i


def _plot_panel_c(ax, psi, y_obs, y_vg, y_r3, theta_s, theta_r, sample_idx):
    ts, tr = float(theta_s[sample_idx]), float(theta_r[sample_idx])
    denom = max(ts - tr, 1e-8)

    def _Se(th):
        return np.clip((th - tr) / denom, 0.0, 1.0)

    # Smooth interpolation for cleaner curves
    def _smooth(y, n=400):
        lp = np.log10(np.maximum(psi, 1e-30))
        lpf = np.linspace(lp[0], lp[-1], n)
        return np.power(10.0, lpf), np.interp(lpf, lp, y)

    psi_f, se_obs = _smooth(_Se(y_obs[sample_idx]))
    _,     se_vg  = _smooth(_Se(y_vg[sample_idx]))
    _,     se_r3  = _smooth(_Se(y_r3[sample_idx]))

    ax.semilogx(psi_f, se_obs, color=C_OBS, lw=2.2, ls="-",
                label="Observed", zorder=4)
    ax.semilogx(psi_f, se_vg, color=C_VG, lw=1.8, ls="--",
                dashes=(6, 3), label="VGParamNet (Run B)", zorder=3)
    ax.semilogx(psi_f, se_r3, color=C_R3, lw=1.8, ls="-.",
                label="Rosetta3", zorder=2)

    ax.set_xscale("log")
    ax.set_xticks(XTICKS_PSI)
    ax.set_xticklabels(XLABELS_PSI, fontsize=FONT_TICK,
                       fontfamily="Arial", rotation=0, ha="center")
    ax.set_xlabel("Matric suction \u03c8  (kPa)",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)
    ax.set_ylabel("Effective saturation S_e  [\u2212]",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)
    ax.set_xlim(float(psi.min()), float(psi.max()))
    ax.set_ylim(-0.03, 1.08)

    _legend(ax, loc="upper right")
    _style(ax, xticks_fs=FONT_TICK, yticks_fs=FONT_MAIN)
    _panel_tag(ax, f"(c)  Representative SWCC  (sample {sample_idx + 1})")


# ── Panel (d): RMSE by regime — all 5 models ──────────────────────────────────
def _plot_panel_d(ax, regime_vg, regime_r3, agg_json):
    """Grouped bar chart: wet/mid/dry RMSE for all 5 models.
    Predictive models (left group) separated from curve-fit models (right group)."""

    labels    = ["Wet\n(\u03c8 < 100 kPa)", "Mid\n(100\u201310 000 kPa)", "Dry\n(> 10 000 kPa)"]
    regime_keys = ["wet", "mid", "dry"]

    # Aggregate regime means for all 5 models
    model_data = {
        "VGParamNet":    [regime_vg[k] for k in regime_keys],
        "Rosetta3":      [regime_r3[k] for k in regime_keys],
        "Fredlund-Xing": [agg_json["Fredlund-Xing"][f"rmse_{k}_mean"] for k in regime_keys],
        "Kosugi":        [agg_json["Kosugi"][f"rmse_{k}_mean"]        for k in regime_keys],
        "Bimodal VG":    [agg_json["Bimodal VG"][f"rmse_{k}_mean"]    for k in regime_keys],
    }
    colors_use = [C_VG, C_R3, C_FX, C_KO, C_BV]
    n_models = len(model_data)
    n_groups = len(labels)
    w = 0.14
    group_centers = np.arange(n_groups)

    for mi, (mname, vals) in enumerate(model_data.items()):
        offset = (mi - (n_models - 1) / 2) * w
        bars = ax.bar(group_centers + offset, vals, width=w,
                      color=colors_use[mi], alpha=0.70,
                      edgecolor="white", linewidth=0.3, zorder=2)

    # Visual separator between predictive (VGParamNet, Rosetta3) and curve-fit
    ax.axvline(0.63, color="#CCCCCC", lw=0.8, ls="--",
               dashes=(4, 4), zorder=0)
    ax.text(0.68, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.1,
            "curve-fit below", fontsize=FONT_SMALL - 2,
            fontfamily="Arial", color="#888888", va="top")

    ax.set_xticks(group_centers)
    ax.set_xticklabels(labels, fontsize=FONT_MAIN, fontfamily="Arial")
    ax.set_ylabel("Mean RMSE  (m\u00b3/m\u00b3)",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)

    handles = [Patch(facecolor=c, alpha=0.70, edgecolor="none", label=n)
               for c, n in zip(colors_use,
                               ["VGParamNet (Run B)", "Rosetta3",
                                "Fredlund-Xing (fit)", "Kosugi (fit)",
                                "Bimodal VG (fit)"])]
    _legend(ax, handles=handles, loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            ncol=1, handlelength=1.2, columnspacing=0.8)
    _style(ax)
    _panel_tag(ax, "(d)  Mean RMSE by suction regime  "
                   "(predictive vs curve-fit models)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    # ── Load arrays ──────────────────────────────────────────────────────────
    print("Loading data...")
    y_test   = np.load(ROOT / "data_pinn_normalized" / "y_test_original.npy").astype(np.float64)
    psi      = np.load(ROOT / "data_pinn_normalized" / "suction_grid.npy").astype(np.float64)
    X_test   = pd.read_csv(ROOT / "data_pinn_normalized" / "X_test.csv")
    theta_s  = X_test["theta_s"].values.astype(np.float64)
    theta_r  = X_test["theta_r"].values.astype(np.float64)

    # VGParamNet Run B
    y_vg = np.load(ROOT / "results_pinn_fixed" / "vgparamnet" /
                   "run_B" / "theta_vgparamnet_test.npy").astype(np.float64)

    # Rosetta3 (actual neural-network ensemble, rosetta-soil package)
    r3_npz = np.load(ROOT / "results_comparison" / "ptf_wrf_comparison" /
                     "rosetta3_predictions.npz")
    y_r3 = r3_npz["theta_r3_curves"].astype(np.float64)

    # Aggregate stats for curve-fit models (from saved JSON)
    agg_json = json.load(open(ROOT / "results_comparison" / "ptf_wrf_comparison" /
                              "comparison_results.json"))

    n = min(len(y_vg), len(y_r3), len(y_test))
    y_vg, y_r3, y_test = y_vg[:n], y_r3[:n], y_test[:n]
    theta_s, theta_r = theta_s[:n], theta_r[:n]
    print(f"  {n} test samples")

    # ── Compute metrics from arrays ───────────────────────────────────────────
    wet_mask = psi < 100
    print("Computing metrics...")
    rmse_wet_vg = _wet_rmse(y_vg, y_test, wet_mask)
    rmse_wet_r3 = _wet_rmse(y_r3, y_test, wet_mask)
    p50_err_vg  = _psi50_log_error(y_vg, y_test, psi, theta_s, theta_r)
    p50_err_r3  = _psi50_log_error(y_r3, y_test, psi, theta_s, theta_r)
    regime_vg   = _regime_rmse(y_vg, y_test, psi)
    regime_r3   = _regime_rmse(y_r3, y_test, psi)

    print(f"  VGParamNet Run B  — wet RMSE mean={rmse_wet_vg.mean():.4f} "
          f"median={np.median(rmse_wet_vg):.4f}, "
          f"psi50 err median={np.median(p50_err_vg):.3f}")
    print(f"  Rosetta3          — wet RMSE mean={rmse_wet_r3.mean():.4f} "
          f"median={np.median(rmse_wet_r3):.4f}, "
          f"psi50 err median={np.median(p50_err_r3):.3f}")
    print(f"  Regime (VGParamNet): " +
          ", ".join(f"{k}={v:.4f}" for k, v in regime_vg.items()))
    print(f"  Regime (Rosetta3):   " +
          ", ".join(f"{k}={v:.4f}" for k, v in regime_r3.items()))

    sample_idx = _pick_sample(y_test, theta_s, theta_r)
    print(f"  Representative sample: idx={sample_idx}")

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12.0, 8.5))
    gs  = mgridspec.GridSpec(2, 3, figure=fig,
                             height_ratios=[1.0, 1.0],
                             width_ratios=[1.0, 1.0, 1.1],
                             hspace=0.55, wspace=0.42,
                             left=0.09, right=0.98,
                             top=0.93, bottom=0.10)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, :])   # full-width bottom panel

    _plot_panel_a(ax_a, rmse_wet_vg, rmse_wet_r3, agg_json)
    _plot_panel_b(ax_b, p50_err_vg,  p50_err_r3,  agg_json)
    _plot_panel_c(ax_c, psi, y_test, y_vg, y_r3, theta_s, theta_r, sample_idx)
    _plot_panel_d(ax_d, regime_vg, regime_r3, agg_json)

    # ── Save ──────────────────────────────────────────────────────────────────
    stem    = "Figure_S6_PTF_WRF_Comparison_q1"
    pdf_out = SUPP_DIR / f"{stem}.pdf"
    png_out = SUPP_DIR / "png" / f"{stem}.png"
    png_out.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(str(pdf_out), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(str(png_out), dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"\nSaved:\n  {pdf_out}\n  {png_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
