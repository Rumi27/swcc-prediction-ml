#!/usr/bin/env python3
"""
Generate supplementary GAN-related figures.

Figure S1: WGAN-GP training losses (critic vs generator).
Figure S2: Real vs synthetic SWCCs.

Data sources:
- results_gan/training_history.json
- data_processed/y_*.npy, data_processed/suction_grid.npy
- results_gan/generated_data/*.npy
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    # Paths
    root_dir = Path(__file__).resolve().parent.parent.parent
    results_gan_dir = root_dir / "results_gan"
    history_path = results_gan_dir / "training_history.json"

    supp_dir = root_dir / "paper_figures" / "supplementary"
    supp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Generating supplementary GAN figures")
    print("=" * 80)

    # Global style (match main paper figures)
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["axes.titlesize"] = 20
    plt.rcParams["xtick.labelsize"] = 18
    plt.rcParams["ytick.labelsize"] = 18
    plt.rcParams["legend.fontsize"] = 18
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["xtick.major.width"] = 1.5
    plt.rcParams["ytick.major.width"] = 1.5

    # -------------------------------------------------------------------------
    # Figure S1: WGAN-GP training loss curves
    # -------------------------------------------------------------------------
    print("\nSupplementary Figure S1: WGAN-GP training losses...")

    if not history_path.exists():
        print(f"  ⚠ training_history.json not found at {history_path}, skipping Figure S1.")
    else:
        with open(history_path, "r") as f:
            hist = json.load(f)

        epochs = np.array(hist.get("epoch", []))
        d_loss = np.array(hist.get("d_loss", []))
        g_loss = np.array(hist.get("g_loss", []))

        if len(epochs) == 0 or len(d_loss) == 0 or len(g_loss) == 0:
            print("  ⚠ training_history.json does not contain valid loss arrays; skipping Figure S1.")
        else:
            fig, ax = plt.subplots(figsize=(8, 6))

            color_critic = "#2E86AB"   # blue
            color_gen = "#C73E1D"      # red

            # Figure S1 house style: Arial 11 pt
            _F1 = 11
            for _lbl in (ax.get_xticklabels() + ax.get_yticklabels()):
                _lbl.set_fontfamily("Arial")
                _lbl.set_fontsize(_F1)

            ax.plot(epochs, d_loss, label="Critic loss", color=color_critic,
                    linewidth=2.5)
            ax.plot(epochs, g_loss, label="Generator loss", color=color_gen,
                    linewidth=2.5)

            ax.set_xlabel("Epoch", fontsize=_F1, fontfamily="Arial", labelpad=10)
            ax.set_ylabel("Loss", fontsize=_F1, fontfamily="Arial", labelpad=10)
            ax.set_title("WGAN-GP training losses", fontsize=_F1, fontfamily="Arial", pad=12)

            ax.grid(True, alpha=0.3, linestyle="--")
            # Move legend downward so it does not cover the generator loss curve
            ax.legend(
                loc="upper right",
                bbox_to_anchor=(0.98, 0.58),
                fontsize=_F1,
                framealpha=0.95,
                facecolor="white",
                edgecolor="gray",
            )

            ax.tick_params(labelsize=_F1)
            for _lbl in ax.get_xticklabels() + ax.get_yticklabels():
                _lbl.set_fontfamily("Arial")

            fig.tight_layout()

            out_png = supp_dir / "Figure_S1_WGAN_Training_Loss.png"
            out_pdf = supp_dir / "Figure_S1_WGAN_Training_Loss.pdf"
            fig.savefig(out_png, dpi=300, bbox_inches="tight")
            fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
            plt.close(fig)

            print(f"  ✓ Saved: {out_png}")
            print(f"  ✓ Saved: {out_pdf}")

    # -------------------------------------------------------------------------
    # Figure S2: Real vs Synthetic SWCCs
    # -------------------------------------------------------------------------
    print("\nSupplementary Figure S2: Real vs Synthetic SWCCs...")

    try:
        # Real curves
        data_proc_dir = root_dir / "data_processed"
        y_train = np.load(data_proc_dir / "y_train.npy")
        y_val = np.load(data_proc_dir / "y_val.npy")
        y_test = np.load(data_proc_dir / "y_test.npy")
        y_real = np.vstack([y_train, y_val, y_test])
        suction_real = np.load(data_proc_dir / "suction_grid.npy")

        # Synthetic (unfiltered)
        y_syn = np.load(results_gan_dir / "generated_data" / "synthetic_swcc_curves.npy")
        suction_syn = np.load(results_gan_dir / "generated_data" / "suction_grid.npy")

        # Synthetic filtered (if available)
        y_syn_filt = None
        suction_syn_filt = None
        filt_dir = results_gan_dir / "generated_data_filtered"
        if (filt_dir / "synthetic_swcc_curves_filtered.npy").exists():
            y_syn_filt = np.load(filt_dir / "synthetic_swcc_curves_filtered.npy")
            suction_syn_filt = np.load(filt_dir / "suction_grid.npy")

        # Use a common suction grid (assume they match; otherwise, fall back to real grid)
        suction_grid = suction_real

        fig, axes = plt.subplots(1, 3 if y_syn_filt is not None else 2,
                                 figsize=(18, 6), sharey=True)

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        # Professional color scheme
        color_real = "#2E86AB"       # blue
        color_unfiltered = "#F18F01" # orange
        color_filtered = "#06A77D"   # green

        # (a) Real curves (subset)
        ax = axes[0]
        n_show_real = min(50, len(y_real))
        idx_real = np.linspace(0, len(y_real) - 1, n_show_real, dtype=int)
        for i in idx_real:
            ax.semilogx(suction_grid, y_real[i], color=color_real,
                        alpha=0.15, linewidth=0.5)

        ax.set_xlabel("Suction ψ (kPa)", labelpad=10)
        ax.set_ylabel("Water content θ (m³/m³)", labelpad=10)
        ax.set_title("(a) Real UNSODA SWCCs", pad=10)
        ax.tick_params(labelsize=18)
        ax.grid(True, alpha=0.3, which="both")

        # (b) Synthetic (unfiltered)
        ax = axes[1]
        n_show_syn = min(50, len(y_syn))
        idx_syn = np.linspace(0, len(y_syn) - 1, n_show_syn, dtype=int)
        for i in idx_syn:
            ax.semilogx(suction_syn, y_syn[i], color=color_unfiltered,
                        alpha=0.15, linewidth=0.5)

        ax.set_xlabel("Suction ψ (kPa)", labelpad=10)
        ax.set_title("(b) Synthetic SWCCs (unfiltered)", pad=10)
        ax.tick_params(labelsize=18)
        ax.grid(True, alpha=0.3, which="both")

        # (c) Synthetic filtered (optional)
        if y_syn_filt is not None and len(axes) > 2:
            ax = axes[2]
            n_show_filt = min(50, len(y_syn_filt))
            idx_filt = np.linspace(0, len(y_syn_filt) - 1, n_show_filt, dtype=int)
            for i in idx_filt:
                ax.semilogx(suction_syn_filt, y_syn_filt[i], color=color_filtered,
                            alpha=0.15, linewidth=0.5)

            ax.set_xlabel("Suction ψ (kPa)", labelpad=10)
            ax.set_title("(c) Synthetic SWCCs (filtered)", pad=10)
            ax.tick_params(labelsize=18)
            ax.grid(True, alpha=0.3, which="both")

        fig.tight_layout()

        out_png = supp_dir / "Figure_S2_Real_vs_Synthetic_SWCCs.png"
        out_pdf = supp_dir / "Figure_S2_Real_vs_Synthetic_SWCCs.pdf"
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"  ✓ Saved: {out_png}")
        print(f"  ✓ Saved: {out_pdf}")

    except Exception as e:
        print(f"  ⚠ Could not generate Figure S2 (Real vs Synthetic SWCCs): {e}")

    # -------------------------------------------------------------------------
    # Figure S3: Distribution mismatch of ψ50 (real vs synthetic)
    # -------------------------------------------------------------------------
    print("\nSupplementary Figure S3: ψ₅₀ distribution mismatch (real vs synthetic)...")

    try:
        _F3 = 11

        # Real ψ50 from precomputed UNSODA values (train/val/test)
        vg_dir = root_dir / "results_pinn_fixed" / "vgparamnet"
        psi50_train_real = None
        psi50_val_real = None
        psi50_test_real = None

        # Try both root and nested vgparamnet directories
        candidates_root = [
            root_dir / "results_pinn_fixed" / "vgparamnet" / "psi50_train.npy",
            root_dir / "results_pinn_fixed" / "vgparamnet" / "psi50_val.npy",
            root_dir / "results_pinn_fixed" / "vgparamnet" / "psi50_test.npy",
        ]
        candidates_nested = [
            vg_dir / "psi50_train.npy",
            vg_dir / "psi50_val.npy",
            vg_dir / "psi50_test.npy",
        ]

        # Prefer nested paths under paper_1_swcc_ml if present
        if candidates_nested[0].exists():
            psi50_train_real = np.load(candidates_nested[0])
            psi50_val_real = np.load(candidates_nested[1])
            psi50_test_real = np.load(candidates_nested[2])
        elif candidates_root[0].exists():
            psi50_train_real = np.load(candidates_root[0])
            psi50_val_real = np.load(candidates_root[1])
            psi50_test_real = np.load(candidates_root[2])

        if psi50_train_real is None:
            raise FileNotFoundError("Could not find real ψ50 arrays (psi50_train/val/test.npy).")

        psi50_real_all = np.concatenate([psi50_train_real, psi50_val_real, psi50_test_real])

        # Synthetic ψ50 from GAN-generated curves
        # Load synthetic curves and suction grid
        y_syn = np.load(results_gan_dir / "generated_data" / "synthetic_swcc_curves.npy")
        psi_syn = np.load(results_gan_dir / "generated_data" / "suction_grid.npy")

        # Optionally use filtered synthetic curves if available
        filt_path = results_gan_dir / "generated_data_filtered" / "synthetic_swcc_curves_filtered.npy"
        if filt_path.exists():
            y_syn = np.load(filt_path)
            psi_syn = np.load(results_gan_dir / "generated_data_filtered" / "suction_grid.npy")

        # Subsample to ~1000 synthetic curves if more
        max_syn = 1000
        if len(y_syn) > max_syn:
            idx_syn_sample = np.random.RandomState(42).choice(len(y_syn), size=max_syn, replace=False)
            y_syn_sample = y_syn[idx_syn_sample]
        else:
            y_syn_sample = y_syn

        # Compute ψ50 for synthetic curves:
        # Define effective saturation based on per-curve θs/θr estimated from min/max
        psi50_syn = []
        for theta_curve in y_syn_sample:
            theta_s = float(np.max(theta_curve))
            theta_r = float(np.min(theta_curve))
            theta_range = theta_s - theta_r
            if theta_range < 1e-6:
                # Nearly constant curve, skip or assign NaN
                psi50_syn.append(np.nan)
                continue

            Se = (theta_curve - theta_r) / (theta_range + 1e-6)

            # Find index where Se is closest to 0.5
            idx = int(np.argmin(np.abs(Se - 0.5)))
            psi50_syn.append(float(psi_syn[idx]))

        psi50_syn = np.array(psi50_syn)
        # Remove NaNs, if any
        psi50_syn = psi50_syn[~np.isnan(psi50_syn)]

        if len(psi50_syn) == 0:
            raise RuntimeError("All synthetic ψ50 values are NaN; cannot plot distribution.")

        # Plot distributions (log x-axis for ψ50)
        fig, ax = plt.subplots(figsize=(8, 6))

        # To avoid issues with log, restrict to positive ψ50
        psi50_real_pos = psi50_real_all[psi50_real_all > 0]
        psi50_syn_pos = psi50_syn[psi50_syn > 0]

        bins = np.logspace(
            np.log10(min(psi50_real_pos.min(), psi50_syn_pos.min())),
            np.log10(max(psi50_real_pos.max(), psi50_syn_pos.max())),
            40,
        )

        ax.hist(psi50_real_pos, bins=bins, alpha=0.6, color="#2E86AB",
                edgecolor="black", linewidth=0.4, label="Real (UNSODA)")
        ax.hist(psi50_syn_pos, bins=bins, alpha=0.6, color="#C73E1D",
                edgecolor="black", linewidth=0.4, label="Synthetic (GAN)")

        ax.set_xscale("log")
        # Match Figure 3-style decade ticks (plain numeric labels)
        _s_ticks = np.array([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
        _s_ticklabels = ["0.1", "1.0", "10", "100", "1000", "10000", "100000", "1000000"]
        ax.set_xticks(_s_ticks)
        ax.set_xticklabels(_s_ticklabels)
        # Use mathtext for ψ50 to avoid Arial missing-subscript glyphs
        ax.set_xlabel(r"Matric suction $\psi_{50}$ (kPa)", fontsize=_F3, fontfamily="Arial", labelpad=10)
        ax.set_ylabel("Count", fontsize=_F3, fontfamily="Arial", labelpad=10)
        ax.set_title(
            r"Distribution of $\psi_{50}$: real vs synthetic",
            pad=10,
            fontsize=_F3,
            fontfamily="Arial",
            fontweight="normal",
        )
        ax.grid(False)
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)
        leg = ax.legend(loc="upper right", framealpha=0.95, facecolor="white", edgecolor="gray")
        if leg is not None:
            for t in leg.get_texts():
                t.set_fontfamily("Arial")
                t.set_fontsize(_F3)

        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontfamily("Arial")
            tick.set_fontsize(_F3 - 1)

        fig.tight_layout()

        out_png = supp_dir / "Figure_S3_Psi50_Distribution_Real_vs_Synthetic.png"
        out_pdf = supp_dir / "Figure_S3_Psi50_Distribution_Real_vs_Synthetic.pdf"
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"  ✓ Saved: {out_png}")
        print(f"  ✓ Saved: {out_pdf}")

    except Exception as e:
        print(f"  ⚠ Could not generate Figure S3 (ψ50 distribution): {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()

