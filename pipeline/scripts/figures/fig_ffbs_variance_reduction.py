#!/usr/bin/env python3
"""
12c - Synthetic Variance Reduction Demo

Shows how sequential EKF filtering reduces parameter uncertainty compared to
independent per-timepoint ECM estimates, making a small physiological change
detectable against measurement noise.

Scenario:
  - True parameter (tau_b, log10 seconds) is stable then steps +0.20 decades
    at t=50 (a ~58% increase in actual time constant -- realistic for ATP).
  - Independent ECM-like estimates have sigma ~0.35 log10 decades (realistic).
  - EKF fuses sequential measurements: posterior sigma narrows to ~0.10 decades.
  - Step SNR rises from 0.57 (ECM, below threshold) to ~2.0 (EKF, detectable).

Usage:
    cd pipeline && python fig_ffbs_variance_reduction.py
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Style ──────────────────────────────────────────────────────────────────
TEAL       = "#148f77"
BLACK      = "#1a1a1a"
GREY_MED   = "#999999"
GREY_DARK  = "#444444"
GREY_FAINT = "#d0d0d0"
RED        = "#d62728"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "mathtext.default": "regular",
})

# ── Simulation parameters ──────────────────────────────────────────────────
T           = 100       # total time points
STEP_T      = 50        # time of step change
STEP_SIZE   = 0.20      # log10 decades increase (58% actual change in tau_b)
TRUE_INIT   = -2.00     # initial log10(tau_b) [seconds]

ECM_STD     = 0.35      # per-measurement ECM noise (log10 decades)
EKF_Q_STD   = 0.030     # process noise std (log10 decades per step)
EKF_OBS_STD = ECM_STD   # EKF uses same obs noise assumption as ECM

SEED = 42


def run_ekf_1d(observations, obs_std, q_std, init_mean, init_std):
    """
    1-D Kalman filter (exact for linear-Gaussian case).

    PREDICT: P_pred = P_post + Q
    UPDATE:  K = P_pred / (P_pred + R)
             mu_post = mu_pred + K * (obs - mu_pred)
             P_post  = (1 - K) * P_pred

    Returns:
        means: (T,) posterior means
        stds:  (T,) posterior standard deviations
    """
    T_len = len(observations)
    means = np.zeros(T_len)
    stds  = np.zeros(T_len)

    Q  = q_std ** 2
    R  = obs_std ** 2
    P  = init_std ** 2
    mu = init_mean

    for t in range(T_len):
        if t > 0:
            P = P + Q                      # predict

        K  = P / (P + R)                   # Kalman gain
        mu = mu + K * (observations[t] - mu)
        P  = (1.0 - K) * P

        means[t] = mu
        stds[t]  = np.sqrt(P)

    return means, stds


def make_figure(out_dir: Path):
    np.random.seed(SEED)
    t = np.arange(T)

    # True parameter trajectory
    true_traj = np.full(T, TRUE_INIT)
    true_traj[STEP_T:] = TRUE_INIT + STEP_SIZE

    # Independent ECM-like noisy estimates
    ecm_obs = true_traj + np.random.normal(0.0, ECM_STD, size=T)

    # EKF posterior
    ekf_means, ekf_stds = run_ekf_1d(
        ecm_obs,
        obs_std   = EKF_OBS_STD,
        q_std     = EKF_Q_STD,
        init_mean = ecm_obs[0],
        init_std  = ECM_STD,
    )

    ss_std      = float(ekf_stds[STEP_T - 1])   # EKF std just before step
    ss_std_late = float(ekf_stds[-10:].mean())   # steady-state after step
    reduction   = ECM_STD / ss_std_late

    # ── Figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[3.5, 1.5, 1.5],
        hspace=0.40,
        left=0.10, right=0.95, top=0.91, bottom=0.08,
    )
    ax_main = fig.add_subplot(gs[0])
    ax_std  = fig.add_subplot(gs[1], sharex=ax_main)
    ax_snr  = fig.add_subplot(gs[2], sharex=ax_main)

    # ── Panel 1: Trajectory ────────────────────────────────────────────────
    # ECM scatter + fixed uncertainty band
    ax_main.fill_between(
        t,
        ecm_obs - ECM_STD,
        ecm_obs + ECM_STD,
        color=GREY_MED, alpha=0.12, linewidth=0, zorder=1,
    )
    ax_main.scatter(
        t, ecm_obs,
        color=GREY_MED, s=16, alpha=0.60, zorder=2,
        label=f"ECM estimates  (\u03c3 = {ECM_STD:.2f} dec, fixed)",
    )

    # EKF posterior ± 1σ
    ax_main.fill_between(
        t,
        ekf_means - ekf_stds,
        ekf_means + ekf_stds,
        color=TEAL, alpha=0.28, linewidth=0, zorder=3,
        label="KF posterior \u00b11\u03c3",
    )
    ax_main.plot(t, ekf_means, color=TEAL, linewidth=2.4, zorder=4,
                 label="KF posterior mean")

    # True trajectory
    ax_main.plot(t, true_traj, color=BLACK, linewidth=2.0,
                 linestyle="-", zorder=5, label=r"True $\tau_b$")

    # Step marker
    ax_main.axvline(STEP_T, color=RED, linewidth=1.4,
                    linestyle="--", alpha=0.70, zorder=6)
    ax_main.annotate(
        f"ATP stimulus\n+{STEP_SIZE:.2f} log dec\n"
        f"(+{100*(10**STEP_SIZE - 1):.0f}% in $\\tau_b$)",
        xy=(STEP_T, TRUE_INIT + STEP_SIZE * 0.45),
        xytext=(STEP_T + 5, TRUE_INIT + STEP_SIZE * 0.05),
        fontsize=9, color=RED,
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
    )

    # Steady-state EKF sigma callout
    ax_main.annotate(
        f"KF \u03c3 \u2192 {ss_std_late:.3f} dec",
        xy=(t[-8], ekf_means[-8] + ekf_stds[-8]),
        xytext=(t[-45], TRUE_INIT + STEP_SIZE + 0.45),
        fontsize=9, color=TEAL,
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.0),
    )

    ax_main.set_ylabel(r"$\log_{10}(\tau_b)$  [log$_{10}$ s]", fontsize=12)
    ax_main.set_ylim(TRUE_INIT - 0.85, TRUE_INIT + STEP_SIZE + 0.75)
    ax_main.grid(True, alpha=0.20, linewidth=0.5)
    ax_main.tick_params(labelsize=10)
    ax_main.legend(
        fontsize=9.5, loc="upper left",
        framealpha=0.93, edgecolor=GREY_FAINT,
        handlelength=1.8, handletextpad=0.5,
    )
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # ── Panel 2: Posterior sigma over time ─────────────────────────────────
    ax_std.axhline(ECM_STD, color=GREY_DARK, linewidth=1.6,
                   linestyle="--", alpha=0.70,
                   label=f"ECM \u03c3 = {ECM_STD:.2f} dec (fixed)")
    ax_std.plot(t, ekf_stds, color=TEAL, linewidth=2.2,
                label="KF posterior \u03c3(t)")
    ax_std.axvline(STEP_T, color=RED, linewidth=1.2,
                   linestyle="--", alpha=0.55)

    ax_std.annotate(
        f"{reduction:.1f}× reduction",
        xy=(t[-1], ss_std_late),
        xytext=(t[-35], ECM_STD * 0.50),
        fontsize=9.5, color=TEAL, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.1),
    )

    ax_std.set_ylabel("Posterior σ\n(log$_{10}$ dec)", fontsize=10)
    ax_std.set_ylim(0.0, ECM_STD * 1.30)
    ax_std.grid(True, alpha=0.20, linewidth=0.5)
    ax_std.tick_params(labelsize=10)
    ax_std.legend(fontsize=9, loc="upper right",
                  framealpha=0.92, edgecolor=GREY_FAINT)
    plt.setp(ax_std.get_xticklabels(), visible=False)

    # ── Panel 3: Detection SNR after the step ──────────────────────────────
    # SNR = step_size / sigma(t); meaningful only after step occurs
    snr_ecm = np.full(T, np.nan)
    snr_ekf = np.full(T, np.nan)
    snr_ecm[STEP_T:] = STEP_SIZE / ECM_STD
    snr_ekf[STEP_T:] = STEP_SIZE / ekf_stds[STEP_T:]

    ax_snr.axhline(1.0, color=GREY_MED, linewidth=1.2,
                   linestyle=":", alpha=0.85,
                   label="SNR = 1  (detection limit)")
    ax_snr.axhline(STEP_SIZE / ECM_STD, color=GREY_DARK,
                   linewidth=1.6, linestyle="--", alpha=0.70,
                   label=f"ECM SNR = {STEP_SIZE/ECM_STD:.2f}  (sub-threshold)")
    ax_snr.plot(t, snr_ekf, color=TEAL, linewidth=2.2,
                label="KF SNR(t)  (rising above threshold)")
    ax_snr.axvline(STEP_T, color=RED, linewidth=1.2,
                   linestyle="--", alpha=0.55)

    # Mark when EKF SNR first exceeds 1
    above = np.where(snr_ekf > 1.0)[0]
    if len(above) > 0:
        t_detect = above[0]
        ax_snr.axvline(t_detect, color=TEAL, linewidth=1.0,
                       linestyle=":", alpha=0.70)
        ax_snr.annotate(
            f"detectable\nat t={t_detect}",
            xy=(t_detect, 1.05),
            xytext=(t_detect + 3, 1.65),
            fontsize=8.5, color=TEAL,
            arrowprops=dict(arrowstyle="->", color=TEAL, lw=0.9),
        )

    ax_snr.set_ylabel("SNR\n(step / σ)", fontsize=10)
    ax_snr.set_xlabel("Time point", fontsize=12)
    ax_snr.set_xlim(-1, T)
    ax_snr.set_ylim(0.0, max(4.0, float(np.nanmax(snr_ekf)) * 1.15))
    ax_snr.grid(True, alpha=0.20, linewidth=0.5)
    ax_snr.tick_params(labelsize=10)
    ax_snr.legend(fontsize=9, loc="upper right",
                  framealpha=0.92, edgecolor=GREY_FAINT)

    # ── Title ──────────────────────────────────────────────────────────────
    fig.suptitle(
        "Sequential Kalman Filtering: Variance Reduction Enables Change Detection\n"
        f"True step = {STEP_SIZE:.2f} log dec  |  "
        f"ECM \u03c3 = {ECM_STD:.2f} \u2192 KF \u03c3 \u2248 {ss_std_late:.3f} dec  |  "
        f"SNR: {STEP_SIZE/ECM_STD:.2f} \u2192 {STEP_SIZE/ss_std_late:.1f}",
        fontsize=12, fontweight="bold",
    )

    # ── Save ───────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "variance_reduction_demo.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
    plt.close(fig)
    return path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/poster")
    args = parser.parse_args()

    print("Generating variance reduction demo...")
    make_figure(Path(args.out_dir))
    print("Done.")
