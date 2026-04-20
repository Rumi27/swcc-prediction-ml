#!/usr/bin/env python3
"""
Figure 3 — SWCC Dataset Overview and Preprocessing (no arrow callouts)
Identical to generate_figure3_q1.py except arrow-style legend text is omitted.
Infoboxes (a–c) and the frameless Train/Validation/Test key on (d) are kept.

Outputs: Figure3_SWCC_Preprocessing_q1_no_arrows.pdf / .png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT    = Path(__file__).resolve().parents[2]
PNG_DIR = ROOT / "paper_figures" / "png"
PDF_DIR = ROOT / "paper_figures"
PNG_DIR.mkdir(parents=True, exist_ok=True)

FONT_MAIN  = 10   # axis titles, panel tags
FONT_TICK_Y = 8  # y-axis tick numerals — uniform on all panels
FONT_TICK  = 8   # panel (d) x-axis ticks; default x in _style when not ψ
FONT_TICK_XLOG = 8   # ψ log-axis x ticks (horizontal; slightly smaller to avoid overlap)
FONT_SMALL = 10   # infoboxes, panel (d) colour key

matplotlib.rcParams.update({
    "text.usetex": False,
    "axes.formatter.use_mathtext": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "font.size": FONT_MAIN,
    "axes.labelsize": FONT_MAIN,
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK_Y,
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

# ---- Palette ----------------------------------------------------------------
C_TRAIN  = "#1F77B4"   # blue   -- training set
C_VAL    = "#E07B00"   # dark orange -- validation set
C_TEST   = "#2CA02C"   # green  -- test set
C_BACK   = "#7BA7C7"   # muted blue -- background curves
C_MEAN   = "#C62828"   # dark red   -- mean curve
C_SD     = "#EF9A9A"   # light red  -- +/-1 SD fill
C_REP    = "#B71C1C"   # deep red   -- representative examples

XTICKS_PSI  = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
XLABELS_PSI = [
    "0.1", "1", "10", "100", "1000", "10000", "100000", "1000000",
]
Y_MAX = 0.85


# ---- Style helpers ----------------------------------------------------------
def _style(ax, *, xticks_fs=FONT_TICK, yticks_fs=FONT_TICK_Y):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("black")
    ax.tick_params(which="both", top=True, right=True, direction="in")
    ax.tick_params(axis="x", labelsize=xticks_fs)
    ax.tick_params(axis="y", labelsize=yticks_fs)
    ax.grid(False)
    for lbl in ax.get_xticklabels():
        lbl.set_fontfamily("Arial")
        lbl.set_fontsize(xticks_fs)
    for lbl in ax.get_yticklabels():
        lbl.set_fontfamily("Arial")
        lbl.set_fontsize(yticks_fs)


def _psi_xaxis(ax, psi):
    ax.set_xscale("log")
    ax.set_xticks(XTICKS_PSI)
    ax.set_xticklabels(
        XLABELS_PSI, fontsize=FONT_TICK_XLOG,
        fontfamily="Arial", rotation=0, ha="center",
    )
    ax.set_xlim(float(psi.min()) * 0.9, float(psi.max()) * 1.1)
    ax.set_xlabel("Matric suction \u03c8  (kPa)",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=9)


def _swcc_yaxis(ax):
    ax.set_ylabel("Volumetric water content \u03b8  [\u2212]",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=5)
    ax.set_ylim(0, Y_MAX)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.tick_params(axis="y", labelsize=FONT_TICK_Y)


def _panel_tag(ax, tag):
    ax.text(0.0, 1.03, tag, transform=ax.transAxes,
            fontsize=FONT_MAIN, fontweight="normal", fontfamily="Arial",
            va="bottom", ha="left", clip_on=False)


def _infobox(ax, text, x=0.97, y=0.97, ha="right", va="top"):
    ax.text(x, y, text,
            transform=ax.transAxes, fontsize=FONT_SMALL - 1,
            fontfamily="Arial", va=va, ha=ha,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                      edgecolor="#888888", linewidth=0.7, alpha=0.95))


# ---- Panel (a): Dataset diversity -------------------------------------------
def _plot_panel_a(ax, psi, y_all):
    rng = np.random.default_rng(42)
    idx_bg = rng.choice(len(y_all), min(100, len(y_all)), replace=False)

    for i in idx_bg:
        ax.semilogx(psi, y_all[i], color=C_BACK, alpha=0.18, lw=0.5, zorder=1)

    theta_s = y_all.max(axis=1)
    rep_indices = [int(np.argmin(np.abs(theta_s - np.percentile(theta_s, p))))
                   for p in [5, 25, 50, 75, 95]]
    for i in rep_indices:
        ax.semilogx(psi, y_all[i], color=C_REP, alpha=0.88, lw=2.0, zorder=3)

    _infobox(ax, f"N = {len(y_all)} soil samples\n"
                 f"\u03b8 range: [{y_all.min():.2f}, {y_all.max():.2f}]",
             x=0.97, y=0.97, ha="right", va="top")

    _psi_xaxis(ax, psi)
    _swcc_yaxis(ax)
    _style(ax, xticks_fs=FONT_TICK_XLOG, yticks_fs=FONT_TICK_Y)
    _panel_tag(ax, f"(a) Dataset diversity  (N = {len(y_all)})")


# ---- Panel (b): Statistical envelope ----------------------------------------
def _plot_panel_b(ax, psi, y_all):
    for i in range(len(y_all)):
        ax.semilogx(psi, y_all[i], color=C_BACK, alpha=0.04, lw=0.3, zorder=1)

    mean_c = np.mean(y_all, axis=0)
    std_c  = np.std(y_all,  axis=0)

    ax.fill_between(psi,
                    np.clip(mean_c - std_c, 0, None),
                    mean_c + std_c,
                    color=C_SD, alpha=0.55, zorder=2)
    ax.semilogx(psi, mean_c, color=C_MEAN, lw=2.5, zorder=4)

    _infobox(ax,
             f"Mean \u03b8s = {mean_c[0]:.3f}\n"
             f"SD (\u03b8s)  = {y_all[:,0].std():.3f}",
             x=0.97, y=0.97, ha="right", va="top")

    _psi_xaxis(ax, psi)
    _swcc_yaxis(ax)
    _style(ax, xticks_fs=FONT_TICK_XLOG, yticks_fs=FONT_TICK_Y)
    _panel_tag(ax, "(b) Statistical envelope  (mean \u03b8 \u00b1 1 SD)")


# ---- Panel (c): Train / Val / Test split ------------------------------------
def _plot_panel_c(ax, psi, y_train, y_val, y_test):
    subsets = [
        (y_train, C_TRAIN, 0.10),
        (y_val,   C_VAL,   0.25),
        (y_test,  C_TEST,  0.25),
    ]
    for y_sub, col, alpha in subsets:
        for i in range(len(y_sub)):
            ax.semilogx(psi, y_sub[i], color=col, alpha=alpha, lw=0.35, zorder=2)
        m = np.mean(y_sub, axis=0)
        ax.semilogx(psi, m, color=col, lw=2.4, zorder=4)

    n_total = len(y_train) + len(y_val) + len(y_test)
    _infobox(ax,
             f"All {n_total} curves:\n"
             f"  d\u03b8/d\u03c8 \u2264 0  (monotonic)\n"
             f"Split: 70 / 15 / 15 %",
             x=0.03, y=0.97, ha="left", va="top")

    _psi_xaxis(ax, psi)
    _swcc_yaxis(ax)
    _style(ax, xticks_fs=FONT_TICK_XLOG, yticks_fs=FONT_TICK_Y)
    _panel_tag(ax, "(c) Train / Validation / Test split")


# ---- Panel (d): theta_s distribution histogram ------------------------------
def _plot_panel_d(ax, y_train, y_val, y_test):
    ts_train = y_train.max(axis=1)
    ts_val   = y_val.max(axis=1)
    ts_test  = y_test.max(axis=1)
    ts_all   = np.concatenate([ts_train, ts_val, ts_test])

    bins = np.linspace(ts_all.min() * 0.95, ts_all.max() * 1.02, 22)

    counts_train, _ = np.histogram(ts_train, bins=bins)
    counts_val,   _ = np.histogram(ts_val,   bins=bins)
    counts_test,  _ = np.histogram(ts_test,  bins=bins)
    x_centers = (bins[:-1] + bins[1:]) / 2

    ax.bar(x_centers, counts_train,
           width=np.diff(bins), color=C_TRAIN, alpha=0.82,
           edgecolor="white", linewidth=0.4, label=f"Train  (N={len(ts_train)})", zorder=3)
    ax.bar(x_centers, counts_val,
           width=np.diff(bins), bottom=counts_train,
           color=C_VAL, alpha=0.82,
           edgecolor="white", linewidth=0.4, label=f"Validation  (N={len(ts_val)})", zorder=3)
    ax.bar(x_centers, counts_test,
           width=np.diff(bins), bottom=counts_train + counts_val,
           color=C_TEST, alpha=0.82,
           edgecolor="white", linewidth=0.4, label=f"Test  (N={len(ts_test)})", zorder=3)

    mean_ts = ts_all.mean()
    ax.axvline(mean_ts, color=C_MEAN, lw=1.6, ls="--", dashes=(6, 3), zorder=5)

    leg_handles = [
        mpatches.Patch(
            facecolor=C_TRAIN, edgecolor="white", linewidth=0.5, label="Train"),
        mpatches.Patch(
            facecolor=C_VAL, edgecolor="white", linewidth=0.5, label="Validation"),
        mpatches.Patch(
            facecolor=C_TEST, edgecolor="white", linewidth=0.5, label="Test"),
    ]
    leg = ax.legend(
        handles=leg_handles, loc="upper right",
        frameon=False,
        borderpad=0.35, handlelength=1.4, handleheight=0.9,
        fontsize=FONT_SMALL,
    )
    for t, col in zip(leg.get_texts(), (C_TRAIN, C_VAL, C_TEST)):
        t.set_fontfamily("Arial")
        t.set_fontsize(FONT_SMALL)
        t.set_color(col)

    ax.set_xlabel("Saturated water content \u03b8s  [\u2212]",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=5)
    ax.set_ylabel("Number of samples",
                  fontsize=FONT_MAIN, fontfamily="Arial", labelpad=5)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(bottom=0)

    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("black")
    ax.tick_params(which="both", top=False, right=False, direction="in")
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    ax.tick_params(axis="y", labelsize=FONT_TICK_Y)
    ax.grid(False)
    for lbl in ax.get_xticklabels():
        lbl.set_fontfamily("Arial")
        lbl.set_fontsize(FONT_TICK)
    for lbl in ax.get_yticklabels():
        lbl.set_fontfamily("Arial")
        lbl.set_fontsize(FONT_TICK_Y)

    _panel_tag(ax, "(d) Saturated water content  \u03b8s  distribution")


# ---- Main -------------------------------------------------------------------
def main() -> int:
    dp = ROOT / "data_processed"
    y_train = np.load(dp / "y_train.npy").astype(np.float32)
    y_val   = np.load(dp / "y_val.npy").astype(np.float32)
    y_test  = np.load(dp / "y_test.npy").astype(np.float32)
    psi     = np.load(dp / "suction_grid.npy").astype(np.float32)
    y_all   = np.vstack([y_train, y_val, y_test])

    print(f"Loaded: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}, "
          f"total={len(y_all)}")
    print(f"Suction grid : {len(psi)} pts, {psi.min():.3g} - {psi.max():.3g} kPa")
    print(f"theta range  : {y_all.min():.4f} - {y_all.max():.4f}")

    FIG_W, FIG_H = 8.5, 6.0
    fig, axes = plt.subplots(
        2, 2, figsize=(FIG_W, FIG_H),
        gridspec_kw=dict(hspace=0.52, wspace=0.34),
    )
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    _plot_panel_a(ax_a, psi, y_all)
    _plot_panel_b(ax_b, psi, y_all)
    _plot_panel_c(ax_c, psi, y_train, y_val, y_test)
    _plot_panel_d(ax_d, y_train, y_val, y_test)

    fig.align_ylabels([ax_a, ax_c])
    fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.14)

    stem    = "Figure3_SWCC_Preprocessing_q1_no_arrows"
    pdf_out = PDF_DIR / f"{stem}.pdf"
    png_out = PNG_DIR / f"{stem}.png"
    fig.savefig(str(pdf_out), dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(str(png_out), dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"\nSaved:\n  {pdf_out}\n  {png_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
