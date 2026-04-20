#!/usr/bin/env python3
"""
Figure S5 — Gradient Boosting Monotonicity Violations
Q1 journal quality (14.0 in wide × 9.5 in tall, 5 panels).

Layout: 2 rows × 3 columns (GridSpec)
  Row 0:  (a) col 0 — bump-count histogram
          (b) cols 1-2 — max-amplitude histogram (wider)
  Row 1:  (c) col 0 — typical violation SWCC
          (d) col 1 — max amplitude SWCC
          (e) col 2 — most frequent violations SWCC

Data integrity
--------------
All statistics AND representative curves are derived from the *same* GB
re-train (random_state=42 fixed in BaselineModels call below) so panels
(a)/(b) and (c)/(d)/(e) are self-consistent.

Design
------
* 14.0 in wide × 9.5 in tall; Arial 12 pt labels; 10 pt legend/annotations
* Tick direction inward, mirrored top/right; no grid; white background
* X-tick labels on log axes: 8 pt, horizontal, space-separated thousands
* Panel tags at (0.0, 1.03) outside axes, fontsize=FONT_MAIN, va='bottom'
* Legend: frameon=False
* Annotation box: plain white, no coloured border
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
from matplotlib.lines import Line2D

ROOT     = Path(__file__).resolve().parents[2]
SUPP_DIR = ROOT / "paper_figures" / "supplementary"
SUPP_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

FONT_MAIN  = 12
FONT_SMALL = 10
FONT_TICK  = 8    # for log-axis x-tick labels

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

# SWCC panel x-ticks (6 ticks — fits narrow sub-panel without overlap)
XTICKS_PSI  = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e6]
XLABELS_PSI = ["0.1", "1", "10", "100", "1 000", "1 000 000"]

# Colors
C_OBS    = "#000000"   # black  — Observed SWCC
C_MONO   = "#7F8C8D"   # grey   — GB monotone segments
C_VIOL   = "#E74C3C"   # red    — GB violation segments
C_TS     = "#27AE60"   # green  — θs boundary
C_TR     = "#E67E22"   # orange — θr boundary
C_BUMP_A = "#E74C3C"   # red    — bump-count histogram
C_BUMP_B = "#F39C12"   # amber  — amplitude histogram


# ── Style helpers ─────────────────────────────────────────────────────────────
def _style(ax):
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


def _panel_tag(ax, tag: str) -> None:
    ax.text(0.0, 1.03, tag, transform=ax.transAxes,
            fontsize=FONT_MAIN, fontweight="normal", fontfamily="Arial",
            va="bottom", ha="left", clip_on=False)


def _legend(ax, handles=None, **kw):
    if handles is not None:
        leg = ax.legend(handles=handles, frameon=False, borderpad=0.4,
                        handlelength=2.0, fontsize=FONT_SMALL, **kw)
    else:
        leg = ax.legend(frameon=False, borderpad=0.4, handlelength=2.0,
                        fontsize=FONT_SMALL, **kw)
    for t in leg.get_texts():
        t.set_fontfamily("Arial")
        t.set_fontsize(FONT_SMALL)
    return leg


# ── Monotonicity analysis ──────────────────────────────────────────────────────
def _violation_stats(y_gb: np.ndarray):
    """Per-curve bump counts and max amplitudes from fresh GB predictions."""
    bump_counts    = np.zeros(len(y_gb), dtype=int)
    max_amplitudes = np.zeros(len(y_gb), dtype=np.float64)
    for i, theta in enumerate(y_gb):
        diff = theta[:-1] - theta[1:]          # >0 means decreasing (correct)
        viol = diff < -1e-8                     # violations
        bump_counts[i] = int(np.sum(viol))
        if np.any(viol):
            max_amplitudes[i] = float(np.max(-diff[viol]))
    return bump_counts, max_amplitudes


# ── Panel (a): bump-count histogram ───────────────────────────────────────────
def _plot_panel_a(ax, bump_counts: np.ndarray):
    n_max = int(bump_counts.max())
    bins  = np.arange(-0.5, n_max + 1.5, 1)
    ax.hist(bump_counts, bins=bins, color=C_BUMP_A, alpha=0.70,
            edgecolor="white", linewidth=0.4)

    mean_v   = float(np.mean(bump_counts))
    median_v = float(np.median(bump_counts))
    ax.axvline(mean_v,   color="#2E86AB", lw=1.4, ls="--", dashes=(5, 3),
               label=f"Mean: {mean_v:.1f}")
    ax.axvline(median_v, color="#1A535C", lw=1.4, ls="--", dashes=(3, 3),
               label=f"Median: {median_v:.0f}")

    ax.set_xlabel("Number of non-monotonic segments per curve",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)
    ax.set_ylabel("Count  [\u2212]",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)
    ax.set_xlim(-0.5, n_max + 0.5)

    _legend(ax, loc="upper left")
    _style(ax)
    _panel_tag(ax, f"(a)  Non-monotonic segment count per curve  (N = {len(bump_counts)})")


# ── Panel (b): amplitude histogram ────────────────────────────────────────────
def _plot_panel_b(ax, max_amplitudes: np.ndarray):
    amps = max_amplitudes[max_amplitudes > 0]
    ax.hist(amps, bins=50, color=C_BUMP_B, alpha=0.70,
            edgecolor="white", linewidth=0.4)

    mean_a   = float(np.mean(amps))
    median_a = float(np.median(amps))
    max_a    = float(np.max(amps))

    ax.axvline(mean_a,   color="#2E86AB", lw=1.4, ls="--", dashes=(5, 3),
               label=f"Mean: {mean_a:.4f}")
    ax.axvline(median_a, color="#1A535C", lw=1.4, ls="--", dashes=(3, 3),
               label=f"Median: {median_a:.4f}")
    ax.axvline(max_a,    color=C_VIOL,    lw=1.4, ls="--", dashes=(2, 2),
               label=f"Max: {max_a:.4f}")

    ax.set_xscale("log")
    ax.set_xlabel("Maximum non-monotonic amplitude \u0394\u03b8  (m\u00b3/m\u00b3)",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)
    ax.set_ylabel("Count  [\u2212]",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)

    _legend(ax, loc="upper left")
    _style(ax)
    # Override x-tick labels for log axis
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    for lbl in ax.get_xticklabels():
        lbl.set_fontfamily("Arial")
        lbl.set_fontsize(FONT_TICK)

    _panel_tag(ax, f"(b)  Maximum violation amplitude per curve  (max = {max_a:.4f} m\u00b3/m\u00b3)")


# ── Panel (c/d/e): representative failure SWCC ────────────────────────────────
def _plot_swcc_panel(ax, psi: np.ndarray, theta_true: np.ndarray,
                     theta_gb: np.ndarray, theta_s: float, theta_r: float,
                     panel_letter: str, subtitle: str,
                     bump_count: int, max_amp: float):
    """
    Draw observed (black solid) + GB prediction split into monotone (grey dashed)
    and violation segments (red solid). Top-5 worst violations marked with triangles.
    """
    # Observed
    ax.semilogx(psi, theta_true, color=C_OBS, lw=1.8, ls="-",
                label="Observed", zorder=4)

    # Violation mask (True at index i if interval i→i+1 is non-monotone)
    diff = theta_gb[:-1] - theta_gb[1:]
    viol_seg = diff < -1e-8           # shape (99,)

    # Plot contiguous monotone / violation runs
    i, n = 0, len(psi)
    first_mono = True
    first_viol = True
    while i < n - 1:
        is_v = bool(viol_seg[i])
        j = i + 1
        while j < n - 1 and bool(viol_seg[j]) == is_v:
            j += 1
        seg_psi   = psi[i:j + 1]
        seg_theta = theta_gb[i:j + 1]
        if is_v:
            lbl = "GB  (violation)" if first_viol else "_nolegend_"
            ax.semilogx(seg_psi, seg_theta, color=C_VIOL, lw=2.0, ls="-",
                        zorder=3, label=lbl)
            first_viol = False
        else:
            lbl = "GB  (monotone)" if first_mono else "_nolegend_"
            ax.semilogx(seg_psi, seg_theta, color=C_MONO, lw=1.4, ls="--",
                        dashes=(6, 3), zorder=2, label=lbl)
            first_mono = False
        i = j

    # Top-5 worst violation triangles
    viol_amps = np.where(viol_seg, -diff, 0.0)
    n_viol = int(np.sum(viol_seg))
    if n_viol > 0:
        top_k = min(5, n_viol)
        top_idx = np.argsort(viol_amps)[-top_k:]
        for vi in top_idx:
            mid_psi   = np.sqrt(psi[vi] * psi[vi + 1])
            mid_theta = (theta_gb[vi] + theta_gb[vi + 1]) / 2
            ax.plot(mid_psi, mid_theta, "^", color=C_VIOL,
                    markersize=6, zorder=6, markeredgecolor="white",
                    markeredgewidth=0.5)

    # θs / θr boundary lines
    ax.axhline(theta_s, color=C_TS, lw=1.2, ls=":", zorder=1,
               label=f"\u03b8_s = {theta_s:.3f}")
    ax.axhline(theta_r, color=C_TR, lw=1.2, ls=":", zorder=1,
               label=f"\u03b8_r = {theta_r:.3f}")

    # Axes
    ax.set_xscale("log")
    ax.set_xticks(XTICKS_PSI)
    ax.set_xticklabels(XLABELS_PSI, fontsize=FONT_TICK,
                       fontfamily="Arial", rotation=0, ha="center")
    ax.set_xlabel("Matric suction \u03c8  (kPa)",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)
    ax.set_ylabel("Water content \u03b8  (m\u00b3/m\u00b3)",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=6)
    ax.set_xlim(0.07, float(psi.max()))

    # Stats annotation — bottom left, plain white box, no coloured border
    ax.text(0.03, 0.04,
            f"{n_viol} violation segments\n"
            f"max \u0394\u03b8 = {max_amp:.4f} m\u00b3/m\u00b3",
            transform=ax.transAxes,
            fontsize=FONT_SMALL - 1, fontfamily="Arial", color="#333333",
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.90, edgecolor="#CCCCCC", linewidth=0.6))

    # Legend — upper right, no frame
    handles = [
        Line2D([0], [0], color=C_OBS,  lw=1.8, ls="-",
               label="Observed"),
        Line2D([0], [0], color=C_MONO, lw=1.4, ls="--",
               dashes=(6, 3), label="GB (monotone)"),
        Line2D([0], [0], color=C_VIOL, lw=2.0, ls="-",
               label=f"GB (violation)"),
        Line2D([0], [0], color=C_TS,   lw=1.2, ls=":",
               label=f"\u03b8_s = {theta_s:.3f}"),
        Line2D([0], [0], color=C_TR,   lw=1.2, ls=":",
               label=f"\u03b8_r = {theta_r:.3f}"),
    ]
    _legend(ax, handles=handles, loc="upper right")

    _style(ax)
    _panel_tag(ax, f"({panel_letter})  {subtitle}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    from training_pinn.config_pinn_fixed import DATA_CONFIG
    from baseline_models import BaselineModels

    print("Loading data...")
    X_test  = pd.read_csv(DATA_CONFIG["test_file"])
    y_test  = np.load(DATA_CONFIG["y_test_original_file"]).astype(np.float64)
    psi     = np.load(DATA_CONFIG["suction_grid_file"]).astype(np.float64)

    print("Training GB (random_state=42 for reproducibility)...")
    bm = BaselineModels(data_dir=ROOT / "data_processed",
                        output_dir=ROOT / "results_baseline")
    (Xtr, Xva, Xte), (ytr, yva, yte), _ = bm.load_data()
    Xtr_f, Xva_f, Xte_f, _ = bm.prepare_features(Xtr, Xva, Xte)
    gb_models = bm.train_gradient_boosting(Xtr_f, ytr, Xva_f, yva)
    y_gb = bm.predict_swcc(gb_models, Xte_f, model_type="gradient_boosting",
                           n_points=len(psi)).astype(np.float64)

    # Trim to same length (safety)
    n = min(len(y_gb), len(y_test))
    y_gb, y_test = y_gb[:n], y_test[:n]
    X_test = X_test.iloc[:n].reset_index(drop=True)

    print(f"  {n} test samples")

    # ── Compute violation statistics consistently from this GB run ───────────
    bump_counts, max_amplitudes = _violation_stats(y_gb)
    n_pass_strict = int(np.sum(bump_counts == 0))
    print(f"  Pass (strict): {n_pass_strict}/{n}  "
          f"({100*n_pass_strict/n:.1f}%)")
    print(f"  Bump counts: mean={bump_counts.mean():.1f}, "
          f"median={np.median(bump_counts):.0f}, max={bump_counts.max()}")
    print(f"  Max amplitude: {max_amplitudes.max():.4f} m³/m³")

    # ── Representative samples ────────────────────────────────────────────────
    median_k     = int(np.median(bump_counts[bump_counts > 0]))
    cand         = np.where(bump_counts == median_k)[0]
    typical_idx  = int(cand[len(cand) // 2]) if len(cand) else int(np.argmax(bump_counts > 0))
    maxamp_idx   = int(np.argmax(max_amplitudes))
    maxbump_idx  = int(np.argmax(bump_counts))

    print(f"  (c) typical  : idx={typical_idx}, bumps={bump_counts[typical_idx]}, "
          f"amp={max_amplitudes[typical_idx]:.4f}")
    print(f"  (d) max amp  : idx={maxamp_idx},  bumps={bump_counts[maxamp_idx]}, "
          f"amp={max_amplitudes[maxamp_idx]:.4f}")
    print(f"  (e) max bump : idx={maxbump_idx}, bumps={bump_counts[maxbump_idx]}, "
          f"amp={max_amplitudes[maxbump_idx]:.4f}")

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14.0, 9.5))
    gs  = mgridspec.GridSpec(2, 3, figure=fig,
                             height_ratios=[1, 1.1],
                             hspace=0.52, wspace=0.38,
                             left=0.08, right=0.98,
                             top=0.93, bottom=0.09)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1:3])   # spans cols 1-2
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2])

    _plot_panel_a(ax_a, bump_counts)
    _plot_panel_b(ax_b, max_amplitudes)

    for ax, idx, letter, subtitle in [
        (ax_c, typical_idx,  "c", f"Typical case  ({bump_counts[typical_idx]} bumps)"),
        (ax_d, maxamp_idx,   "d", f"Largest amplitude  (\u0394\u03b8 max = {max_amplitudes[maxamp_idx]:.4f} m\u00b3/m\u00b3)"),
        (ax_e, maxbump_idx,  "e", f"Most frequent  ({bump_counts[maxbump_idx]} bumps)"),
    ]:
        _plot_swcc_panel(
            ax, psi, y_test[idx], y_gb[idx],
            float(X_test.iloc[idx]["theta_s"]),
            float(X_test.iloc[idx]["theta_r"]),
            letter, subtitle,
            int(bump_counts[idx]), float(max_amplitudes[idx]),
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    stem    = "Figure_S5_GB_Monotonicity_Violations_q1"
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
