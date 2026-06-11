#!/usr/bin/env python3
"""
Visualize temporal tracking comparison: ECM warm vs GPF causal vs Kalman smoother.

Runs N_CASE_STUDIES episodes through the live API (localhost:5003) and produces:
  - Case study panels: ground truth trajectory vs each method per identifiable param
  - Aggregate per-step MAE curves across all episodes
  - Impedance resnorm comparison vs ECM baseline

Usage:
    python visualize_temporal_comparison.py
"""

import sys
import json
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

# ── import helpers from benchmark ────────────────────────────────────────────
from benchmark_full_pipeline import (
    run_episode, simulate_sequence, sample_circuit, compute_impedance,
    to_identifiable, stream_request,
    ID_NAMES, METHODS,
)
from src.pipeline.gpf import BOUNDS_LOW, BOUNDS_HIGH

API_URL = "http://localhost:5003/mc_temporal_analysis_stream"

N_CASE_STUDIES = 4   # individual circuit panels
N_AGG_EPISODES = 10  # total episodes for MAE curves (includes case studies)
N_STEPS = 20
DRIFT   = 0.03
SNR_DB  = 40.0
N_SAMPLES = 200

METHOD_STYLE = {
    'ecm_warm':        {'color': '#e05c5c', 'ls': '--',  'lw': 1.4, 'label': 'ECM warm'},
    'gpf_causal':      {'color': '#5b9bd5', 'ls': '-',   'lw': 1.6, 'label': 'GPF causal'},
    'kal_gpf_smooth':  {'color': '#70b86e', 'ls': '-',   'lw': 2.0, 'label': 'Kalman smoother'},
}
TRUTH_STYLE = {'color': 'k', 'ls': '-', 'lw': 1.0, 'label': 'Ground truth'}

PARAM_LABELS = {
    'tau_big':   r'$\tau_{big}$ (log10 s)',
    'tau_small': r'$\tau_{small}$ (log10 s)',
    'TER':       r'TER (log10 $\Omega$)',
    'TEC':       r'TEC (log10 F)',
    'Rsh':       r'$R_{sh}$ (log10 $\Omega$)',
}


def _compute_resnorm(params_log10, obs):
    """Mean absolute impedance error between params and a single observation."""
    import numpy as np
    freqs = np.array(obs['frequencies'])
    omega = 2 * np.pi * freqs
    Zr_obs = np.array(obs['Z_real'])
    Zi_obs = np.array(obs['Z_imag'])
    Zr_pred, Zi_pred = compute_impedance(params_log10, omega)
    return float(np.mean(np.abs(Zr_pred - Zr_obs) + np.abs(Zi_pred - Zi_obs)))


# ── run episodes ─────────────────────────────────────────────────────────────

def run_episodes(n_episodes, n_steps, drift, snr_db, n_samples, seed=42):
    """
    Run n_episodes through the API. Returns list of dicts with per-episode data.
    """
    rng   = np.random.default_rng(seed)
    freqs = np.logspace(-1, 6, 100)

    episodes = []
    for ep in range(n_episodes):
        t0 = time.time()
        print(f"  episode {ep+1}/{n_episodes}...", end='', flush=True)
        try:
            id_err, raw_err, true_id, true_raw = run_episode(
                rng, freqs, n_steps, drift, n_samples=n_samples, snr_db=snr_db
            )
            elapsed = time.time() - t0
            if id_err is None:
                print(f" FAILED ({elapsed:.0f}s)")
                continue

            # Also store per-step estimates (not just errors) for the case studies
            # Re-run to capture actual estimates alongside ground truth
            base  = sample_circuit(rng)
            seqs, true_params_raw = simulate_sequence(rng, base, freqs, n_steps, drift, snr_db)

            episodes.append({
                'id_errors': id_err,
                'true_id':   true_id,
                'elapsed':   elapsed,
            })
            best = min((m for m in METHODS if m in id_err),
                       key=lambda m: float(np.nanmean(id_err[m])), default='?')
            maes = {m: float(np.nanmean(id_err[m])) for m in METHODS if m in id_err}
            mae_str = '  '.join(f"{m}={v:.3f}" for m, v in maes.items())
            print(f" {mae_str}  ({elapsed:.0f}s)")
        except Exception as e:
            print(f" ERROR: {e}")
    return episodes


# ── also capture raw estimates for case study plots ──────────────────────────

def run_episode_full(rng, freqs, n_steps, drift, n_samples, snr_db):
    """
    Same as run_episode but also returns per-step estimates (not just abs errors).
    """
    from benchmark_full_pipeline import (
        sample_circuit, simulate_sequence, to_identifiable,
        parse_step_event, parse_done_event
    )
    NAN_ID = np.full(5, np.nan)

    base   = sample_circuit(rng)
    sequences, true_params_raw = simulate_sequence(rng, base, freqs, n_steps, drift, snr_db)
    true_id = np.array([to_identifiable(p) for p in true_params_raw])

    step_ests = {m: [] for m in ['ecm_warm', 'gpf_causal']}
    done_ests = {}

    try:
        for ev in stream_request(sequences, n_samples=n_samples):
            if ev.get('done'):
                done_ests = parse_done_event(ev, n_steps)
            elif 'error' not in ev:
                parsed = parse_step_event(ev)
                for m in ['ecm_warm', 'gpf_causal']:
                    step_ests[m].append(parsed[m]['id'] if m in parsed else NAN_ID.copy())
    except Exception as e:
        return None

    estimates = {}
    for m in ['ecm_warm', 'gpf_causal']:
        arr = np.array(step_ests[m])
        if len(arr) == n_steps:
            estimates[m] = arr
    ks = done_ests.get('kal_gpf_smooth', {})
    if 'id' in ks and len(ks['id']) == n_steps:
        estimates['kal_gpf_smooth'] = ks['id']

    id_errors = {m: np.abs(estimates[m] - true_id) for m in estimates}

    return {
        'true_id':   true_id,
        'estimates': estimates,
        'id_errors': id_errors,
    }


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_case_study(ax_row, ep_data, case_idx):
    """Plot one episode's trajectories across all 5 params."""
    true_id   = ep_data['true_id']      # (T, 5)
    estimates = ep_data['estimates']    # method -> (T, 5)
    T = true_id.shape[0]
    steps = np.arange(1, T + 1)

    for j, (pname, ax) in enumerate(zip(ID_NAMES, ax_row)):
        ax.plot(steps, true_id[:, j], **TRUTH_STYLE)
        for m, est in estimates.items():
            s = METHOD_STYLE[m]
            ax.plot(steps, est[:, j], color=s['color'], ls=s['ls'],
                    lw=s['lw'], label=s['label'])
        ax.set_xlim(1, T)
        if case_idx == 0:
            ax.set_title(PARAM_LABELS.get(pname, pname), fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_ylabel(f'Circuit {case_idx+1}', fontsize=8) if j == 0 else None
        ax.grid(True, alpha=0.3, lw=0.5)


def plot_mae_curves(ax_row, all_episodes):
    """Plot per-step mean MAE curves aggregated over all episodes."""
    T = N_STEPS
    steps = np.arange(1, T + 1)

    for j, (pname, ax) in enumerate(zip(ID_NAMES, ax_row)):
        step_mae = {m: np.zeros(T) for m in METHODS}
        counts   = {m: np.zeros(T) for m in METHODS}

        for ep in all_episodes:
            id_err = ep['id_errors']
            for m in METHODS:
                if m in id_err:
                    col = id_err[m][:, j]  # (T,)
                    mask = ~np.isnan(col)
                    step_mae[m][mask] += col[mask]
                    counts[m][mask]   += 1

        for m in METHODS:
            denom = np.where(counts[m] > 0, counts[m], np.nan)
            curve = step_mae[m] / denom
            s = METHOD_STYLE[m]
            ax.plot(steps, curve, color=s['color'], ls=s['ls'],
                    lw=s['lw'], label=s['label'])

        ax.set_xlabel('Step', fontsize=8)
        ax.set_title(PARAM_LABELS.get(pname, pname), fontsize=9) if j == 0 else None
        ax.set_ylabel('MAE (decades)', fontsize=8) if j == 0 else None
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, lw=0.5)
        ax.set_xlim(1, T)
        ax.set_ylim(bottom=0)


def plot_overall_bar(ax, all_episodes):
    """Bar chart of overall mean MAE per method."""
    available = [m for m in METHODS
                 if any(m in ep['id_errors'] for ep in all_episodes)]
    maes = []
    for m in available:
        arrs = [ep['id_errors'][m] for ep in all_episodes if m in ep['id_errors']]
        maes.append(float(np.nanmean(np.concatenate(arrs))))

    colors = [METHOD_STYLE[m]['color'] for m in available]
    labels = [METHOD_STYLE[m]['label'] for m in available]
    bars = ax.bar(labels, maes, color=colors, width=0.5, edgecolor='k', lw=0.5)
    for bar, v in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Mean MAE (log10 decades)', fontsize=9)
    ax.set_title('Overall MAE vs ECM baseline', fontsize=10)
    ax.set_ylim(0, max(maes) * 1.25)
    ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(labelsize=8)

    # draw improvement arrows vs ECM warm
    if 'ecm_warm' in available:
        ecm_idx = available.index('ecm_warm')
        ecm_mae = maes[ecm_idx]
        for i, (m, mae) in enumerate(zip(available, maes)):
            if m == 'ecm_warm':
                continue
            pct = 100 * (ecm_mae - mae) / (ecm_mae + 1e-12)
            color = '#2d8a2d' if pct > 0 else '#cc0000'
            ax.text(i, mae / 2, f'{pct:+.0f}%', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')


def main():
    # Verify API
    try:
        r = requests.get('http://localhost:5003/model_info', timeout=5)
        info = r.json()
        print(f"API: {info.get('model','?')}  param_space={info.get('param_space','?')}")
    except Exception as e:
        print(f"API not reachable: {e}")
        sys.exit(1)

    rng   = np.random.default_rng(42)
    freqs = np.logspace(-1, 6, 100)

    print(f"\nRunning {N_CASE_STUDIES} case studies (full estimates) ...")
    case_studies = []
    for i in range(N_CASE_STUDIES):
        print(f"  case study {i+1}/{N_CASE_STUDIES}...", end='', flush=True)
        t0 = time.time()
        data = run_episode_full(rng, freqs, N_STEPS, DRIFT, N_SAMPLES, SNR_DB)
        elapsed = time.time() - t0
        if data is None:
            print(f" FAILED")
            continue
        n_methods = len(data['estimates'])
        maes = {m: float(np.nanmean(data['id_errors'][m]))
                for m in data['id_errors']}
        print(f" {n_methods} methods  {elapsed:.0f}s  "
              + '  '.join(f"{m}={v:.3f}" for m, v in maes.items()))
        case_studies.append(data)

    print(f"\nRunning {N_AGG_EPISODES - N_CASE_STUDIES} additional episodes for MAE curves ...")
    agg_episodes = list(case_studies)  # reuse case studies in aggregate
    for i in range(N_AGG_EPISODES - N_CASE_STUDIES):
        print(f"  episode {i+1}/{N_AGG_EPISODES - N_CASE_STUDIES}...", end='', flush=True)
        t0 = time.time()
        data = run_episode_full(rng, freqs, N_STEPS, DRIFT, N_SAMPLES, SNR_DB)
        elapsed = time.time() - t0
        if data is None:
            print(" FAILED")
            continue
        maes = {m: float(np.nanmean(data['id_errors'][m])) for m in data['id_errors']}
        print(f" {elapsed:.0f}s  " + '  '.join(f"{m}={v:.3f}" for m, v in maes.items()))
        agg_episodes.append(data)

    if not case_studies:
        print("No successful episodes — check API.")
        sys.exit(1)

    # ── build figure ─────────────────────────────────────────────────────────
    n_cs  = len(case_studies)
    n_row = n_cs + 2   # case studies + MAE curves row + bar chart row
    n_col = 5          # one column per identifiable param

    fig = plt.figure(figsize=(18, 3 * n_row + 1))
    fig.suptitle(
        f'Temporal Tracking: ECM Warm vs GPF Causal vs Kalman Smoother\n'
        f'(fisher_v7  |  {N_AGG_EPISODES} episodes x {N_STEPS} steps  '
        f'drift={DRIFT}  SNR={SNR_DB}dB)',
        fontsize=12, y=0.98,
    )

    gs_top  = gridspec.GridSpec(n_cs + 1, n_col,
                                 top=0.93, bottom=0.22,
                                 hspace=0.55, wspace=0.35)
    gs_bot  = gridspec.GridSpec(1, 1,
                                 top=0.17, bottom=0.04,
                                 hspace=0.0, wspace=0.0)

    # Case study rows
    all_axes = []
    for i, ep_data in enumerate(case_studies):
        ax_row = [fig.add_subplot(gs_top[i, j]) for j in range(n_col)]
        plot_case_study(ax_row, ep_data, i)
        all_axes.append(ax_row)

        # legend on first row
        if i == 0:
            handles = [
                plt.Line2D([0], [0], **TRUTH_STYLE),
                *[plt.Line2D([0], [0], color=METHOD_STYLE[m]['color'],
                             ls=METHOD_STYLE[m]['ls'], lw=METHOD_STYLE[m]['lw'],
                             label=METHOD_STYLE[m]['label'])
                  for m in METHODS if m in ep_data['estimates']]
            ]
            ax_row[n_col - 1].legend(handles=handles, fontsize=7,
                                      loc='upper right', framealpha=0.8)

    # MAE curves row
    mae_row = [fig.add_subplot(gs_top[n_cs, j]) for j in range(n_col)]
    plot_mae_curves(mae_row, agg_episodes)
    mae_row[0].set_ylabel('MAE (decades)', fontsize=8)
    for j, ax in enumerate(mae_row):
        ax.set_xlabel('Step', fontsize=8)
        ax.set_title(
            ('Per-step MAE ' if j == 0 else '') + PARAM_LABELS.get(ID_NAMES[j], ID_NAMES[j]),
            fontsize=9)

    # Add section labels
    fig.text(0.01, 0.94 - (n_cs) * (0.71 / max(n_cs + 1, 1)) * 0.5,
             'Case\nStudies', va='center', ha='left', fontsize=9,
             color='#444', rotation=90)

    # Overall bar chart
    ax_bar = fig.add_subplot(gs_bot[0])
    plot_overall_bar(ax_bar, agg_episodes)

    out_path = Path('results/temporal_comparison.png')
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f"\nSaved: {out_path}")

    # Also print summary table
    available = [m for m in METHODS if any(m in ep['id_errors'] for ep in agg_episodes)]
    print()
    print("=" * 72)
    print("OVERALL MAE  (log10 decades, lower is better)")
    print("=" * 72)
    col = 16
    header = f"{'Param':<12}" + "".join(f"{m:>{col}}" for m in available)
    print(header)
    print("-" * len(header))
    for i, name in enumerate(ID_NAMES):
        row = f"{name:<12}"
        for m in available:
            arrs = [ep['id_errors'][m] for ep in agg_episodes if m in ep['id_errors']]
            mae  = float(np.nanmean(np.concatenate(arrs)[:, i]))
            row += f"{mae:>{col}.4f}"
        print(row)
    print("-" * len(header))
    row = f"{'MEAN':<12}"
    for m in available:
        arrs = [ep['id_errors'][m] for ep in agg_episodes if m in ep['id_errors']]
        mae  = float(np.nanmean(np.concatenate(arrs)))
        row += f"{mae:>{col}.4f}"
    print(row)

    if 'ecm_warm' in available:
        ecm_arrs = [ep['id_errors']['ecm_warm'] for ep in agg_episodes if 'ecm_warm' in ep['id_errors']]
        ecm_all  = np.concatenate(ecm_arrs)
        others   = [m for m in available if m != 'ecm_warm']
        print()
        print("IMPROVEMENT vs ECM warm  (positive = better)")
        h2 = f"{'Param':<12}" + "".join(f"{m:>{col}}" for m in others)
        print(h2)
        print("-" * len(h2))
        for i, name in enumerate(ID_NAMES):
            base = float(np.nanmean(ecm_all[:, i]))
            row  = f"{name:<12}"
            for m in others:
                arrs = [ep['id_errors'][m] for ep in agg_episodes if m in ep['id_errors']]
                mae  = float(np.nanmean(np.concatenate(arrs)[:, i]))
                pct  = 100.0 * (base - mae) / (base + 1e-12)
                row += f"{pct:>+{col}.1f}%"
            print(row)
        row  = f"{'MEAN':<12}"
        base = float(np.nanmean(ecm_all))
        for m in others:
            arrs = [ep['id_errors'][m] for ep in agg_episodes if m in ep['id_errors']]
            mae  = float(np.nanmean(np.concatenate(arrs)))
            pct  = 100.0 * (base - mae) / (base + 1e-12)
            row += f"{pct:>+{col}.1f}%"
        print(row)


if __name__ == '__main__':
    main()
