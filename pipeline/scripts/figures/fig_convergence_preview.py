#!/usr/bin/env python3
"""
GPF Convergence Preview

Generates the convergence figure using synthetic samples that mimic
what the real DL+MCMC sequential inference produces.  No ML models
or MCMC required -- runs in seconds.

Usage:
    cd pipeline && python fig_convergence_preview.py
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path

# ============================================================
#  Shared constants (copied from 12_poster_results.py)
# ============================================================

TEAL  = "#148f77"
GOLD  = "#c8962e"
BLACK = "#1a1a1a"
GREY_LIGHT = "#e8e8e8"
GREY_MED   = "#999999"
GREY_DARK  = "#444444"
GREY_FAINT = "#d0d0d0"

PARAM_LABELS      = [r"$R_a$", r"$R_b$", r"$C_a$", r"$C_b$", r"$R_{sh}$"]
PARAM_LABELS_PLAIN = ["Ra", "Rb", "Ca", "Cb", "Rsh"]
PARAM_LO = np.array([0.0,  0.0, -7.0, -7.0, 0.0])
PARAM_HI = np.array([3.5,  3.5, -4.0, -4.0, 4.0])

CIRCUIT_A = np.array([2.48, 2.70, -5.52, -5.70, 2.90])
CIRCUIT_B = np.array([1.90, 2.18, -5.22, -5.40, 2.48])
CIRCUIT_C = np.array([2.70, 2.00, -5.70, -5.10, 2.60])
CIRCUITS = {
    "Mature RPE\n(high TER)":    CIRCUIT_A,
    "Developing RPE\n(low TER)": CIRCUIT_B,
    "Asymmetric RPE\n(disease)": CIRCUIT_C,
}

KEYFRAME_SHIFTS = np.array([
    [ 0.00,  0.00,  0.00,  0.00,  0.00],
    [-0.36, -0.12, +0.09, +0.21, -0.03],
    [-0.75, -0.24, +0.18, +0.45, -0.06],
    [-1.20, -0.39, +0.27, +0.66, -0.09],
    [-1.65, -0.54, +0.36, +0.90, -0.12],
    [-1.35, -0.42, +0.27, +0.69, -0.09],
    [-0.90, -0.27, +0.18, +0.42, -0.03],
    [-0.45, -0.12, +0.09, +0.24,  0.00],
])
N_STEPS   = 8
DISPLAY_IDX = [0, 2, 4, 7]

_TIME_SHORT = [
    "t0\nBaseline", "t1\nEarly ATP", "t2\nATP 5m", "t3\nRising",
    "t4\nPeak", "t5\nDecline", "t6\nLate rec", "t7\nRecov 30m",
]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "mathtext.default": "regular",
})


# ============================================================
#  Synthetic sample generation
# ============================================================

def _make_synthetic_samples(baseline, n_samples=600, n_chains=8):
    """
    Simulate DL+MCMC sequential inference across N_STEPS.

    At t=0 the posterior is wide (uncertain).  Each timestep the
    prior is fused with the previous posterior so the IQR shrinks
    by ~15-25% per step.  The mean tracks the true trajectory with
    small zero-mean residual noise, and bimodality from the Ra/Rb
    symmetry is baked in for realism.
    """
    np.random.seed(42)
    true_traj = baseline + KEYFRAME_SHIFTS   # (8, 5)

    # Initial std in log10 units (roughly 0.5 decades -- wide prior)
    init_std = np.array([0.48, 0.48, 0.40, 0.40, 0.55])

    all_samples = []
    current_std = init_std.copy()

    for t in range(N_STEPS):
        true_t = true_traj[t]

        # Sequential fusion shrinks std each step
        # (ATP disruption at peak t=4 causes slight widening)
        if t == 0:
            std_t = current_std
        elif t == 4:
            std_t = current_std * 1.15   # brief widening at peak
        else:
            std_t = current_std * 0.82   # ~18% tighter each step

        std_t = np.clip(std_t, 0.04, 0.60)
        current_std = std_t

        n_per_chain = n_samples // n_chains
        chain_samples = []
        for c in range(n_chains):
            # Every other chain initialises with Ra/Rb swapped
            # (reflects the symmetry exploration in the real sampler)
            if c % 2 == 1:
                mean_c = true_t[[1, 0, 3, 2, 4]]
            else:
                mean_c = true_t

            samps = np.random.multivariate_normal(
                mean_c,
                np.diag(std_t**2),
                size=n_per_chain,
            )
            samps = np.clip(samps, PARAM_LO, PARAM_HI)
            chain_samples.append(samps)

        combined = np.concatenate(chain_samples, axis=0)
        np.random.shuffle(combined)
        all_samples.append(combined[:n_samples])

    return all_samples, true_traj


# ============================================================
#  Plot
# ============================================================

def plot_mcmc_convergence(circuit_results, out_dir, n_mcmc, n_chains):
    circuit_names = list(circuit_results.keys())
    n_circ = len(circuit_names)
    t_axis = np.arange(N_STEPS)
    circ_colors = [TEAL, GOLD, "#7a4f9e"]

    fig = plt.figure(figsize=(17, 13))
    gs_top = GridSpec(
        n_circ, 5,
        figure=fig,
        left=0.06, right=0.98, top=0.91, bottom=0.30,
        hspace=0.45, wspace=0.28,
    )
    gs_bot = GridSpec(
        1, 3,
        figure=fig,
        left=0.06, right=0.98, top=0.24, bottom=0.07,
        wspace=0.35,
    )

    for ci, cname in enumerate(circuit_names):
        dl_all, true_traj = circuit_results[cname]

        med = np.array([np.median(s, axis=0) for s in dl_all])
        q25 = np.array([np.percentile(s, 25, axis=0) for s in dl_all])
        q75 = np.array([np.percentile(s, 75, axis=0) for s in dl_all])
        q05 = np.array([np.percentile(s,  5, axis=0) for s in dl_all])
        q95 = np.array([np.percentile(s, 95, axis=0) for s in dl_all])

        for p in range(5):
            ax = fig.add_subplot(gs_top[ci, p])

            ax.axvspan(1, 4, color="#b0d4f1", alpha=0.20, zorder=0)
            ax.axvspan(4, 7, color="#d4f1b0", alpha=0.15, zorder=0)
            for t_idx in DISPLAY_IDX:
                ax.axvline(t_idx, color=GREY_FAINT,
                           linewidth=0.8, linestyle=":", zorder=1)

            ax.fill_between(t_axis, q05[:, p], q95[:, p],
                            color=TEAL, alpha=0.18, linewidth=0, zorder=2)
            ax.fill_between(t_axis, q25[:, p], q75[:, p],
                            color=TEAL, alpha=0.45, linewidth=0, zorder=3)
            ax.plot(t_axis, med[:, p],
                    color=TEAL, linewidth=2.0, zorder=4)
            ax.plot(t_axis, true_traj[:, p],
                    color=BLACK, linewidth=1.8, linestyle="-", zorder=5)

            ax.set_xlim(-0.2, 7.2)
            ax.set_xticks(t_axis)
            ax.grid(True, alpha=0.18, linewidth=0.5)
            ax.tick_params(labelsize=7)

            if ci == 0:
                ax.set_title(PARAM_LABELS[p], fontsize=12,
                             fontweight="bold", pad=6)
            if p == 0:
                ax.set_ylabel(cname.replace("\n", " "), fontsize=9,
                              fontweight="bold", color=circ_colors[ci])
            if ci == n_circ - 1:
                ax.set_xticklabels(_TIME_SHORT, fontsize=6.5,
                                   rotation=30, ha="right")
            else:
                ax.set_xticklabels([])

    # Bottom row: IQR width over time
    for ci, cname in enumerate(circuit_names):
        dl_all, _ = circuit_results[cname]
        ax_s = fig.add_subplot(gs_bot[0, ci])

        q25_all = np.array([np.percentile(s, 25, axis=0) for s in dl_all])
        q75_all = np.array([np.percentile(s, 75, axis=0) for s in dl_all])
        iqr = q75_all - q25_all   # (8, 5)

        for p in range(5):
            ax_s.plot(t_axis, iqr[:, p],
                      color=GREY_MED, linewidth=0.9,
                      alpha=0.55, linestyle="--", zorder=2)
            ax_s.text(7.15, iqr[-1, p], PARAM_LABELS[p],
                      fontsize=7, va="center", color=GREY_DARK)

        mean_iqr = iqr.mean(axis=1)
        ax_s.plot(t_axis, mean_iqr,
                  color=circ_colors[ci], linewidth=2.4,
                  zorder=3, label="Mean IQR")

        ax_s.axvspan(1, 4, color="#b0d4f1", alpha=0.20, zorder=0)
        ax_s.axvspan(4, 7, color="#d4f1b0", alpha=0.15, zorder=0)
        for t_idx in DISPLAY_IDX:
            ax_s.axvline(t_idx, color=GREY_FAINT,
                         linewidth=0.8, linestyle=":", zorder=1)

        ax_s.set_xlim(-0.2, 8.0)
        ax_s.set_xticks(t_axis)
        ax_s.set_xticklabels(_TIME_SHORT, fontsize=6.5,
                             rotation=30, ha="right")
        ax_s.set_ylabel("IQR (log$_{10}$ units)", fontsize=9)
        ax_s.set_title(cname.replace("\n", " "), fontsize=10,
                       fontweight="bold", color=circ_colors[ci], pad=5)
        ax_s.grid(True, alpha=0.20, linewidth=0.5)
        ax_s.tick_params(labelsize=7)

    # Legend
    fig.legend(
        handles=[
            Patch(color=TEAL,      alpha=0.18, label="90% CI (5th-95th pct)"),
            Patch(color=TEAL,      alpha=0.55, label="IQR (25th-75th pct)"),
            Line2D([0], [0], color=TEAL,  linewidth=2.0, label="DL median"),
            Line2D([0], [0], color=BLACK, linewidth=1.8, label="Truth"),
            Patch(color="#b0d4f1", alpha=0.60, label="ATP treatment (t1-t4)"),
            Patch(color="#d4f1b0", alpha=0.60, label="Recovery (t4-t7)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.997),
        ncol=6, fontsize=11, framealpha=0.95,
        edgecolor=GREY_FAINT, handlelength=1.6,
        handletextpad=0.4, columnspacing=1.2,
    )
    fig.suptitle(
        f"MCMC Posterior Narrowing Over Time"
        f"  ({n_mcmc} samples, {n_chains} chains)  [synthetic preview]",
        fontsize=14, fontweight="bold", y=1.0,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "mcmc_convergence_preview.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
    plt.close(fig)
    return path


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-mcmc",   type=int, default=600)
    parser.add_argument("--n-chains", type=int, default=8)
    parser.add_argument("--out-dir",  default="results/poster")
    args = parser.parse_args()

    print(f"Generating synthetic MCMC convergence preview "
          f"({args.n_mcmc} samples, {args.n_chains} chains)...")

    circuit_results = {}
    for cname, baseline in CIRCUITS.items():
        label = cname.split("\n")[0]
        print(f"  Synthesising {label}...")
        samples, traj = _make_synthetic_samples(
            baseline, n_samples=args.n_mcmc, n_chains=args.n_chains,
        )
        circuit_results[cname] = (samples, traj)

    path = plot_mcmc_convergence(
        circuit_results,
        out_dir=Path(args.out_dir),
        n_mcmc=args.n_mcmc,
        n_chains=args.n_chains,
    )
    print("Done.")
