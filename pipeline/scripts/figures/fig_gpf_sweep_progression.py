#!/usr/bin/env python3
"""
12h - BL Sweep Progression Visualization

Shows two things for Sample 1:

Left panels (sweep progression):
  Pick one time point (the ATP peak).  Run the full high-to-low-frequency
  BL sweep and capture MC samples at representative checkpoints:
  prior, 10%, 25%, 50%, 75%, 100% of frequencies incorporated.
  Plot how the R1 and TEC distributions narrow as the sweep unfolds.

Right panels (time series):
  Run the full sweep for every time point in Sample 1 and show the
  final tightened posterior (median + IQR) vs 3P-EIS ground truth
  for R1 and TEC only.

Usage:
    cd pipeline && python fig_gpf_sweep_progression.py
    python fig_gpf_sweep_progression.py --model models/fisher_v3/best_model.pt
    python fig_gpf_sweep_progression.py --time-idx 9 --n-bl-freqs 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker

sys.path.insert(0, ".")
from src.models.fisher_transformer import FisherAwareTransformer, TransformerConfig
from src.physics.eis_fisher import compute_impedance


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
})

BLACK      = "#1a1a1a"
MC_COLOR   = "#2166ac"
ATP_COLOR  = "#f4a7b9"
TRUTH_COLOR = BLACK

CSV_PATH      = "../docs/fit results.csv"
CROSS_SECTION = 0.11409

SAMPLE = {"meas_id": "20211021_183356", "interval_min": 3.5, "atp_span": (7, 11), "donor": "Donor A"}

N_FREQ      = 100
FREQ_MIN_HZ = 0.1
FREQ_MAX_HZ = 1e6

# Checkpoint fractions: prior + after this % of freqs incorporated
CHECKPOINTS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]


# ---------------------------------------------------------------
#  Shared helpers (from 12g / 12f)
# ---------------------------------------------------------------

def from_tec_ratio_space_np(samples: np.ndarray) -> np.ndarray:
    Ra_l, Rb_l = samples[:, 0], samples[:, 1]
    log_TEC, log_ratio, Rsh_l = samples[:, 2], samples[:, 3], samples[:, 4]
    TEC = 10.0 ** log_TEC
    R   = 10.0 ** log_ratio
    Ca  = TEC * (1.0 + R)
    Cb  = TEC * (1.0 + 1.0 / (R + 1e-12))
    out = np.empty_like(samples)
    out[:, 0] = Ra_l
    out[:, 1] = Rb_l
    out[:, 2] = np.log10(np.clip(Ca, 1e-30, None))
    out[:, 3] = np.log10(np.clip(Cb, 1e-30, None))
    out[:, 4] = Rsh_l
    return out


_BOUNDS_LOW_NP  = np.array([0.0,  0.0,  -7.5, -7.5,  0.0])
_BOUNDS_HIGH_NP = np.array([4.5,  4.5,  -3.5, -3.5,  4.5])


def log10_to_display(log10_arr: np.ndarray) -> np.ndarray:
    # Clip to physical bounds before exponentiation to prevent overflow
    safe = np.clip(log10_arr, _BOUNDS_LOW_NP, _BOUNDS_HIGH_NP)
    vals = 10 ** safe
    out  = np.empty_like(vals)
    out[:, 0] = vals[:, 0] / 1000   # Ra  kOhm·cm2
    out[:, 1] = vals[:, 1] / 1000   # Rb
    out[:, 2] = vals[:, 2] * 1e6    # Ca  uF/cm2
    out[:, 3] = vals[:, 3] * 1e6    # Cb
    out[:, 4] = vals[:, 4] / 1000   # Rsh
    return out


def _tec_prior_to_orig(mu_np, sigma_np):
    eps = 1e-4
    J   = np.zeros((5, 5))
    for i in range(5):
        delta    = np.zeros(5); delta[i] = eps
        J[:, i]  = (from_tec_ratio_space_np((mu_np + delta)[None])[0] -
                    from_tec_ratio_space_np((mu_np - delta)[None])[0]) / (2.0 * eps)
    mu_orig    = from_tec_ratio_space_np(mu_np[None])[0]
    sigma_orig = J @ sigma_np @ J.T
    # Symmetrise and regularise to ensure positive-definite
    sigma_orig = (sigma_orig + sigma_orig.T) / 2.0
    sigma_orig += np.eye(5) * 1e-4
    return mu_orig, sigma_orig


def load_model(model_path: str):
    ckpt   = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg    = ckpt["config"]
    config = TransformerConfig(
        n_freq=cfg["n_freq"], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"], dropout=cfg["dropout"],
        n_proposals=cfg["n_proposals"], n_params=cfg["n_params"],
        use_low_rank_cov=cfg["use_low_rank_cov"], cov_rank=cfg["cov_rank"],
        use_grad_features=cfg.get("use_grad_features", True),
    )
    model  = FisherAwareTransformer(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    is_tec = ckpt.get("param_space", "original") == "tec_ratio"
    print(f"  Loaded epoch {ckpt.get('epoch','?')}, tec_space={is_tec}")
    return model, is_tec


def load_sample(df, meas_id):
    return (df[(df["chamber"] == "Ussing") & (df["meas_ID"] == meas_id)]
            .sort_values("meas_idx").reset_index(drop=True))


def extract_3p_params(row, xs=CROSS_SECTION):
    return (float(row["pg_absZr_3"]) * xs,
            float(row["pg_absZr_5"]) * xs,
            float(row["pg_absZr_4"]) / xs,
            float(row["pg_absZr_6"]) / xs,
            float(row["pg_absZr_7"]) * xs)


def compute_spectrum(Ra, Rb, Ca, Cb, Rsh, omega_np, noise=0.01, seed=None):
    if seed is not None:
        np.random.seed(seed)
    ot       = torch.tensor(omega_np, dtype=torch.float32)
    Z_r, Z_i = compute_impedance(
        torch.tensor([Ra],  dtype=torch.float32),
        torch.tensor([Rb],  dtype=torch.float32),
        torch.tensor([Ca],  dtype=torch.float32),
        torch.tensor([Cb],  dtype=torch.float32),
        torch.tensor([Rsh], dtype=torch.float32),
        ot,
    )
    Z_r_np = Z_r[0].numpy()
    Z_i_np = Z_i[0].numpy()
    mag     = np.sqrt(Z_r_np**2 + Z_i_np**2)
    Z_r_np += noise * mag * np.random.randn(*Z_r_np.shape)
    Z_i_np += noise * mag * np.random.randn(*Z_i_np.shape)
    return Z_r_np, Z_i_np


# ---------------------------------------------------------------
#  BL sweep helpers
# ---------------------------------------------------------------

# Original log10 [Ra, Rb, Ca, Cb, Rsh] bounds
_BOUNDS_LOW    = torch.tensor([0.0,  0.0,  -7.5, -7.5,  0.0])
_BOUNDS_HIGH   = torch.tensor([4.5,  4.5,  -3.5, -3.5,  4.5])
# tau_per_param: Ca/Cb are degenerate so inflate their prior trust
_TAU_ORIG      = torch.tensor([1.0, 1.0, 3.0, 3.0, 1.0])


def _cholesky_sample(mu: torch.Tensor, sigma: torch.Tensor,
                     n_samples: int) -> np.ndarray:
    """Draw n_samples from N(mu, sigma), clip to original-space bounds."""
    n_params  = mu.shape[-1]
    sigma_reg = sigma + 1e-6 * torch.eye(n_params).unsqueeze(0)
    try:
        L = torch.linalg.cholesky(sigma_reg)
    except RuntimeError:
        diag = torch.diagonal(sigma_reg, dim1=-2, dim2=-1).clamp(min=1e-8)
        L    = torch.diag_embed(torch.sqrt(diag))
    eps     = torch.randn(1, n_samples, n_params)
    samples = mu.unsqueeze(1) + torch.einsum('bpq,bnq->bnp', L, eps)
    s_np    = samples[0].detach().numpy()
    return np.clip(s_np, _BOUNDS_LOW_NP, _BOUNDS_HIGH_NP)


def _tec_model_prior_tec(model, Z_r_np, Z_i_np, omega_np, n_samples):
    """
    For a TEC-ratio model, draw TEC directly from the transformer's mixture
    posterior without any BL refinement.

    TEC is the directly supervised quantity in TEC-ratio space (index 2 = log_TEC).
    Returns (n_samples,) TEC in µF/cm².
    """
    omega_t   = torch.tensor(omega_np, dtype=torch.float32)
    Z_r_t     = torch.tensor(Z_r_np,   dtype=torch.float32).unsqueeze(0)
    Z_i_t     = torch.tensor(Z_i_np,   dtype=torch.float32).unsqueeze(0)
    log_omega = torch.log10(omega_t).unsqueeze(0)
    with torch.no_grad():
        proposals = model(Z_r_t, Z_i_t, log_omega)
        samples   = model.sample_mixture_posterior(proposals, n_samples=n_samples)
    # samples: (1, n_samples, 5) in TEC-ratio space; index 2 = log_TEC
    log_tec = samples[0, :, 2].numpy()   # (n_samples,)
    return 10 ** log_tec * 1e6           # µF/cm²


def sweep_trajectory(model, Z_r_np, Z_i_np, omega_np, n_samples, is_tec,
                     checkpoints=None, n_bl_freqs=None,
                     alpha=0.02, epsilon_floor=0.5, tau=1.0):
    """
    Run the BL sweep for a single spectrum in original log10 [Ra,Rb,Ca,Cb,Rsh] space.

    When is_tec=True the transformer prior is converted from TEC-ratio to original
    space via a numerical Jacobian before BL iterations begin.  Snapshots are always
    in original log10 space; call to_r1(s) to get R1.

    Additionally, when is_tec=True, the dict returned includes 'tec_prior': the
    TEC distribution drawn directly from the transformer's log_TEC output (constant
    through the sweep — BL does not improve TEC because the transformer already
    supervises it directly).

    Returns dict with 'freqs_used', 'snapshots', 'checkpoint_f', 'tec_prior'.
    """
    from src.physics.black_litterman_refinement import (
        black_litterman_update, relative_noise_covariance,
    )
    from src.physics.eis_fisher import compute_impedance, compute_jacobian_autodiff

    if checkpoints is None:
        checkpoints = CHECKPOINTS

    omega_t   = torch.tensor(omega_np, dtype=torch.float32)
    Z_r_t     = torch.tensor(Z_r_np,   dtype=torch.float32).unsqueeze(0)
    Z_i_t     = torch.tensor(Z_i_np,   dtype=torch.float32).unsqueeze(0)
    log_omega = torch.log10(omega_t).unsqueeze(0)

    with torch.no_grad():
        proposals   = model(Z_r_t, Z_i_t, log_omega)
        mu0, sigma0 = model.get_mixture_posterior(proposals)

    # TEC from transformer prior (valid regardless of param space)
    if is_tec:
        tec_prior = _tec_model_prior_tec(model, Z_r_np, Z_i_np, omega_np, n_samples)
        # Convert prior to original space for BL
        mu_np, sigma_np = _tec_prior_to_orig(mu0[0].numpy(), sigma0[0].numpy())
        theta  = torch.tensor(mu_np,    dtype=torch.float32).unsqueeze(0)
        sigma  = torch.tensor(sigma_np, dtype=torch.float32).unsqueeze(0)
    else:
        tec_prior = None
        theta, sigma = mu0.clone(), sigma0.clone()

    sorted_idx = torch.argsort(omega_t, descending=True)
    if n_bl_freqs is not None and n_bl_freqs < len(sorted_idx):
        step       = max(1, len(sorted_idx) // n_bl_freqs)
        sorted_idx = sorted_idx[::step][:n_bl_freqs]
    total = len(sorted_idx)

    checkpoint_steps = {0.0: -1}
    for f in checkpoints:
        if f > 0.0:
            checkpoint_steps[f] = max(0, int(f * total) - 1)

    snapshots      = {}
    snapshots[0.0] = _cholesky_sample(theta, sigma, n_samples)

    for step_i, freq_idx in enumerate(sorted_idx):
        fi         = freq_idx.item()
        omega_step = omega_t[fi].unsqueeze(0)
        Z_re_step  = Z_r_t[:, fi].unsqueeze(-1)
        Z_im_step  = Z_i_t[:, fi].unsqueeze(-1)

        params_lin = 10 ** theta
        Z_re_pred, Z_im_pred = compute_impedance(
            params_lin[:, 0], params_lin[:, 1],
            params_lin[:, 2], params_lin[:, 3],
            params_lin[:, 4], omega_step,
        )
        residual   = torch.cat([Z_re_step - Z_re_pred, Z_im_step - Z_im_pred], dim=1)
        jacobian   = compute_jacobian_autodiff(theta, omega_step)
        omega_diag = relative_noise_covariance(
            Z_re_pred, Z_im_pred,
            alpha=alpha, epsilon_floor=epsilon_floor,
            drift_c=0.0, drift_alpha=1.0, omega=omega_step,
        )
        theta, sigma = black_litterman_update(
            theta_prior=theta, sigma_prior=sigma,
            jacobian=jacobian, residual=residual,
            omega_diag=omega_diag, tau=tau,
            tau_per_param=_TAU_ORIG,
        )
        theta = torch.clamp(theta, _BOUNDS_LOW, _BOUNDS_HIGH)

        for frac, target_step in checkpoint_steps.items():
            if frac > 0.0 and step_i == target_step and frac not in snapshots:
                snapshots[frac] = _cholesky_sample(theta, sigma, n_samples)

    snapshots[1.0] = _cholesky_sample(theta, sigma, n_samples)

    return {
        "freqs_used":   total,
        "snapshots":    [snapshots[f] for f in sorted(snapshots.keys())],
        "checkpoint_f": sorted(snapshots.keys()),
        "tec_prior":    tec_prior,   # (n_samples,) µF/cm², from transformer directly
    }


def sweep_final_sample(model, Z_r_np, Z_i_np, omega_np, n_samples, is_tec,
                       n_bl_freqs=None):
    """
    Run sweep and return (r1_samples, tec_samples) in display units.
    r1_samples: (n_samples,) kOhm·cm² from final BL posterior
    tec_samples: (n_samples,) µF/cm² from transformer prior (is_tec) or BL (not is_tec)
    """
    traj = sweep_trajectory(model, Z_r_np, Z_i_np, omega_np, n_samples, is_tec,
                            checkpoints=[1.0], n_bl_freqs=n_bl_freqs)
    final_orig = traj["snapshots"][-1]   # (n_samples, 5) log10 orig
    disp       = log10_to_display(final_orig)
    r1         = np.minimum(disp[:, 0], disp[:, 1])
    if is_tec and traj["tec_prior"] is not None:
        tec = traj["tec_prior"]
    else:
        Ca, Cb = disp[:, 2], disp[:, 3]
        tec    = Ca * Cb / (Ca + Cb + 1e-12)
    return r1, tec


# ---------------------------------------------------------------
#  Display-unit derived stats
# ---------------------------------------------------------------

def traj_r1_from_snapshot(s: np.ndarray) -> np.ndarray:
    """(n_samples, 5) log10 orig -> R1 in kOhm·cm²."""
    disp = log10_to_display(s)
    return np.minimum(disp[:, 0], disp[:, 1])


def truth_r1_tec(Ra, Rb, Ca, Cb):
    r1  = min(Ra, Rb) / 1000         # kOhm·cm2
    tec = (Ca * Cb / (Ca + Cb)) * 1e6  # uF/cm2
    return r1, tec


# ---------------------------------------------------------------
#  Figure
# ---------------------------------------------------------------

def make_figure(traj, final_r1_series, final_tec_series, truth_series,
                n_t, out_path, model_label):
    """
    Left block  (2 rows × 1 col): sweep progression for one time point
    Right block (2 rows × 1 col): time series of final posterior vs truth

    Rows: R1, TEC

    traj          : dict from sweep_trajectory (has 'snapshots' in orig space + 'tec_prior')
    final_r1_series : list[n_t] of (n_samples,) R1 kOhm·cm² from BL final posterior
    final_tec_series: list[n_t] of (n_samples,) TEC µF/cm² from transformer prior (or BL)
    """
    # === Progression data ===
    snap_f  = traj["checkpoint_f"]
    snap_r1 = [traj_r1_from_snapshot(s) for s in traj["snapshots"]]
    # TEC progression: transformer prior TEC is constant through sweep — show the
    # same distribution at every checkpoint to illustrate it doesn't change with BL
    tec_prior = traj.get("tec_prior")
    snap_tec  = [tec_prior if tec_prior is not None else np.array([np.nan])
                 for _ in snap_f]

    x_snaps  = np.array([f * traj["freqs_used"] for f in snap_f])
    x_labels = [f"{int(f*100)}%" for f in snap_f]

    def band_stats(vals_list):
        meds = np.array([np.nanmedian(v) for v in vals_list])
        q25  = np.array([np.nanpercentile(v, 25) for v in vals_list])
        q75  = np.array([np.nanpercentile(v, 75) for v in vals_list])
        q05  = np.array([np.nanpercentile(v,  5) for v in vals_list])
        q95  = np.array([np.nanpercentile(v, 95) for v in vals_list])
        return meds, q25, q75, q05, q95

    prog_r1_stats  = band_stats(snap_r1)
    prog_tec_stats = band_stats(snap_tec)

    # === Time-series data ===
    t_arr  = SAMPLE["interval_min"]
    atp_lo = SAMPLE["atp_span"][0] * t_arr
    atp_hi = SAMPLE["atp_span"][1] * t_arr
    t_min  = np.arange(n_t) * t_arr

    ts_r1_med  = np.array([np.nanmedian(s)          for s in final_r1_series])
    ts_r1_q25  = np.array([np.nanpercentile(s, 25)  for s in final_r1_series])
    ts_r1_q75  = np.array([np.nanpercentile(s, 75)  for s in final_r1_series])
    ts_tec_med = np.array([np.nanmedian(s)          for s in final_tec_series])
    ts_tec_q25 = np.array([np.nanpercentile(s, 25)  for s in final_tec_series])
    ts_tec_q75 = np.array([np.nanpercentile(s, 75)  for s in final_tec_series])

    truth_r1_arr  = truth_series[:, 0]
    truth_tec_arr = truth_series[:, 1]

    # === Layout ===
    fig = plt.figure(figsize=(18, 9))
    fig.suptitle(
        f"BL Sweep Progression — {model_label}  |  Sample 1 (Donor A)\n"
        "Left: uncertainty narrowing during one sweep   Right: time series (final posterior)",
        fontsize=14, fontweight="bold", y=0.99,
    )

    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1.0, 0.08, 1.0, 1.0],
        hspace=0.35, wspace=0.12,
        top=0.89, bottom=0.13, left=0.08, right=0.98,
    )

    ax_prog_r1  = fig.add_subplot(gs[0, 0])
    ax_prog_tec = fig.add_subplot(gs[1, 0])
    ax_ts_r1    = fig.add_subplot(gs[0, 2])
    ax_ts_tec   = fig.add_subplot(gs[1, 2])

    # Divider column (gs[:, 1]) is blank — acts as visual separator
    for ax_div in [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]:
        ax_div.set_visible(False)

    # ---- Progression panels ----
    for ax, meds, q25, q75, q05, q95, truth_val, label, unit in [
        (ax_prog_r1,  *prog_r1_stats,  truth_series[0, 0],
         r"$R_1 = \min(R_a, R_b)$", r"k$\Omega$ cm$^2$"),
        (ax_prog_tec, *prog_tec_stats, truth_series[0, 1],
         "TEC", r"$\mu$F cm$^{-2}$"),
    ]:
        ax.fill_between(x_snaps, q05, q95, color=MC_COLOR, alpha=0.12, linewidth=0)
        ax.fill_between(x_snaps, q25, q75, color=MC_COLOR, alpha=0.28, linewidth=0)
        ax.plot(x_snaps, meds, color=MC_COLOR, linewidth=2.5,
                path_effects=[pe.withStroke(linewidth=4.5, foreground="white")])
        ax.axhline(truth_val, color=TRUTH_COLOR, linewidth=1.5,
                   linestyle="--", alpha=0.80, label="truth (t=ATP peak)")
        ax.set_xticks(x_snaps)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_xlabel("Sweep fraction (frequencies incorporated)", fontsize=10)
        ax.set_ylabel(f"{label}\n({unit})", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, alpha=0.14, linewidth=0.6)
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=9, frameon=False)

    ax_prog_r1.set_title("Uncertainty narrowing during one sweep\n(prior → full posterior)",
                         fontsize=12, fontweight="bold", pad=10)

    # ---- Time-series panels ----
    for ax, med, q25, q75, truth_arr, label, unit in [
        (ax_ts_r1,  ts_r1_med,  ts_r1_q25,  ts_r1_q75,  truth_r1_arr,
         r"$R_1$", r"k$\Omega$ cm$^2$"),
        (ax_ts_tec, ts_tec_med, ts_tec_q25, ts_tec_q75, truth_tec_arr,
         "TEC", r"$\mu$F cm$^{-2}$"),
    ]:
        ax.axvspan(atp_lo, atp_hi, color=ATP_COLOR, alpha=0.38, zorder=0)
        ax.fill_between(t_min, q25, q75, color=MC_COLOR, alpha=0.25, linewidth=0, zorder=2)
        ax.plot(t_min, med, color=MC_COLOR, linewidth=3.0, zorder=6,
                path_effects=[pe.withStroke(linewidth=5, foreground="white")])
        ax.plot(t_min, truth_arr, color=TRUTH_COLOR, linewidth=1.8, zorder=4)
        ax.scatter(t_min, truth_arr, s=40, marker="s",
                   facecolors="white", edgecolors=TRUTH_COLOR, linewidths=1.6, zorder=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, alpha=0.14, linewidth=0.6)
        ax.tick_params(labelsize=10)
        ax.set_ylabel(f"{label}  ({unit})", fontsize=11)

    ax_ts_r1.set_title("Final sweep posterior vs 3P-EIS truth\n(Sample 1, all time points)",
                       fontsize=12, fontweight="bold", pad=10)
    ax_ts_r1.set_xticklabels([])
    ax_ts_tec.set_xlabel("Time (min)", fontsize=11)

    # Legend
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    leg = [
        Line2D([0], [0], color=TRUTH_COLOR, lw=1.8, marker="s",
               markerfacecolor="white", markeredgecolor=TRUTH_COLOR, label="3P-EIS ground truth"),
        Line2D([0], [0], color=MC_COLOR, lw=2.5, label="Sweep posterior median"),
        mpatches.Patch(color=MC_COLOR, alpha=0.25, label="IQR (25-75%)"),
        mpatches.Patch(color=MC_COLOR, alpha=0.12, label="90% interval (5-95%)"),
        mpatches.Patch(color=ATP_COLOR, alpha=0.38, label="ATP stimulation"),
    ]
    fig.legend(handles=leg, loc="lower center", ncol=5, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, 0.01))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------
#  Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="models/fisher_v3/best_model.pt")
    parser.add_argument("--csv",        default=CSV_PATH)
    parser.add_argument("--n-samples",  type=int, default=400)
    parser.add_argument("--n-bl-freqs", type=int, default=None,
                        help="Subsample BL steps (default: all N_FREQ).")
    parser.add_argument("--time-idx",   type=int, default=None,
                        help="Time point index for the sweep progression panel "
                             "(default: ATP peak midpoint).")
    parser.add_argument("--out",        default="results/sweep_progression_v3.png")
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    df       = pd.read_csv(args.csv)
    freqs    = np.logspace(np.log10(FREQ_MIN_HZ), np.log10(FREQ_MAX_HZ), N_FREQ)
    omega_np = 2 * np.pi * freqs

    print(f"Loading model: {args.model}")
    model, is_tec = load_model(args.model)

    rows = load_sample(df, SAMPLE["meas_id"])
    if len(rows) == 0:
        print("No rows found, exiting.")
        return

    spectra, truth_params = [], []
    for _, row in rows.iterrows():
        Ra, Rb, Ca, Cb, Rsh = extract_3p_params(row)
        Z_r, Z_i = compute_spectrum(Ra, Rb, Ca, Cb, Rsh, omega_np, noise=0.01)
        spectra.append((Z_r, Z_i))
        truth_params.append((Ra, Rb, Ca, Cb, Rsh))

    n_t = len(spectra)

    # --- Pick time point for progression panel ---
    atp_lo_idx = SAMPLE["atp_span"][0]
    atp_hi_idx = SAMPLE["atp_span"][1]
    prog_idx   = args.time_idx if args.time_idx is not None else (atp_lo_idx + atp_hi_idx) // 2
    prog_idx   = min(prog_idx, n_t - 1)
    print(f"\nSweep progression panel: time index {prog_idx} "
          f"(t = {prog_idx * SAMPLE['interval_min']:.1f} min)")

    Z_r_prog, Z_i_prog = spectra[prog_idx]
    print("  Running sweep trajectory for progression panel...")
    traj = sweep_trajectory(model, Z_r_prog, Z_i_prog, omega_np,
                            n_samples=args.n_samples, is_tec=is_tec,
                            n_bl_freqs=args.n_bl_freqs)
    print(f"  Captured {len(traj['snapshots'])} snapshots at {traj['checkpoint_f']}")

    # --- Full time series (final sweep posterior per time point) ---
    print(f"\nRunning final sweep for all {n_t} time points...")
    final_r1_series  = []
    final_tec_series = []
    for t, (Z_r, Z_i) in enumerate(spectra):
        r1, tec = sweep_final_sample(model, Z_r, Z_i, omega_np,
                                     n_samples=args.n_samples, is_tec=is_tec,
                                     n_bl_freqs=args.n_bl_freqs)
        final_r1_series.append(r1)
        final_tec_series.append(tec)
        if (t + 1) % 5 == 0 or (t + 1) == n_t:
            print(f"  time series: {t + 1}/{n_t} done")

    # Truth r1 / tec for each time point
    truth_series = np.array([
        truth_r1_tec(Ra, Rb, Ca, Cb) for Ra, Rb, Ca, Cb, _ in truth_params
    ])  # (n_t, 2) — [r1, tec]

    epoch = torch.load(args.model, map_location="cpu", weights_only=False).get("epoch", "?")
    make_figure(traj, final_r1_series, final_tec_series, truth_series, n_t,
                args.out, f"fisher_v3 epoch {epoch}")


if __name__ == "__main__":
    main()
