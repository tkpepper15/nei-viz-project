#!/usr/bin/env python3
"""
Visualize model performance on raw parameters Ra, Rb, Ca, Cb.

Three panels:
  1. Scatter: true vs predicted for each raw param, ECM vs GPF vs Kalman
  2. Case study trajectories: Ra/Rb/Ca/Cb tracking over time
  3. MAE bar chart — both naive (ordered) and symmetric (min-assignment)

Symmetric MAE accounts for Ra/Rb and Ca/Cb degeneracy: the circuit model is
invariant to swapping (Ra,Ca) <-> (Rb,Cb), so we score each prediction using
the assignment that minimises error.

Usage:
    python visualize_raw_params.py
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

from benchmark_full_pipeline import (
    sample_circuit, simulate_sequence, to_raw,
    stream_request, parse_step_event, parse_done_event,
    _pred_to_raw7, _ecm_to_raw7,
    compute_impedance,
)
from src.pipeline.gpf import BOUNDS_LOW, BOUNDS_HIGH

N_CASE_STUDIES = 3
N_TOTAL_EPS    = 15      # includes case studies; used for scatter + MAE
N_STEPS        = 20
DRIFT          = 0.03
SNR_DB         = 40.0
N_SAMPLES      = 200

# raw7 indices
I_Ra, I_Rb, I_Ca, I_Cb, I_Rsh, I_tau_a, I_tau_b = range(7)

FOCUS_PARAMS = [
    (I_Ra,    r'$R_a$ (log10 $\Omega$)'),
    (I_Rb,    r'$R_b$ (log10 $\Omega$)'),
    (I_Ca,    r'$C_a$ (log10 F)'),
    (I_Cb,    r'$C_b$ (log10 F)'),
]

METHOD_STYLE = {
    'ecm_warm':        {'color': '#e05c5c', 'marker': 'o', 'ms': 3.5, 'lw': 1.4, 'ls': '--', 'label': 'ECM warm',        'alpha': 0.55},
    'gpf_causal':      {'color': '#5b9bd5', 'marker': 's', 'ms': 3.5, 'lw': 1.6, 'ls': '-',  'label': 'GPF causal',      'alpha': 0.55},
    'kal_gpf_smooth':  {'color': '#70b86e', 'marker': '^', 'ms': 4.0, 'lw': 2.0, 'ls': '-',  'label': 'Kalman smoother', 'alpha': 0.65},
}
TRUTH_STYLE = {'color': 'k', 'ls': '-', 'lw': 1.2, 'label': 'Ground truth'}

METHODS = ['ecm_warm', 'gpf_causal', 'kal_gpf_smooth']


# ── symmetric MAE ────────────────────────────────────────────────────────────

def symmetric_raw_mae(pred_log10: np.ndarray, true_log10: np.ndarray) -> dict:
    """
    Compute symmetric MAE for raw params, accounting for Ra/Rb and Ca/Cb degeneracy.

    pred_log10, true_log10: (N, 7) arrays in log10 SI
      cols: [Ra, Rb, Ca, Cb, Rsh, tau_a, tau_b]

    Returns dict with:
      'naive'     : (7,) mean abs error per param, assuming Ra>Rb ordering
      'symmetric' : (7,) mean abs error using optimal (Ra,Ca)<->(Rb,Cb) assignment
      'swap_frac' : fraction of samples where swapping improves the score
    """
    err_naive = np.abs(pred_log10 - true_log10)

    # build swapped prediction: swap (Ra,Ca,tau_a) <-> (Rb,Cb,tau_b)
    pred_swap = pred_log10.copy()
    pred_swap[:, I_Ra],    pred_swap[:, I_Rb]    = pred_log10[:, I_Rb].copy(),    pred_log10[:, I_Ra].copy()
    pred_swap[:, I_Ca],    pred_swap[:, I_Cb]    = pred_log10[:, I_Cb].copy(),    pred_log10[:, I_Ca].copy()
    pred_swap[:, I_tau_a], pred_swap[:, I_tau_b] = pred_log10[:, I_tau_b].copy(), pred_log10[:, I_tau_a].copy()

    err_swap = np.abs(pred_swap - true_log10)

    # per-sample: choose whichever assignment has lower total error on {Ra,Rb,Ca,Cb}
    rc_cols = [I_Ra, I_Rb, I_Ca, I_Cb]
    score_naive = err_naive[:, rc_cols].sum(axis=1)
    score_swap  = err_swap[:, rc_cols].sum(axis=1)
    use_swap    = score_swap < score_naive   # (N,)

    err_sym = np.where(use_swap[:, None], err_swap, err_naive)

    return {
        'naive':     np.nanmean(err_naive, axis=0),   # (7,)
        'symmetric': np.nanmean(err_sym,   axis=0),   # (7,)
        'swap_frac': float(use_swap.mean()),
    }


# ── episode runner ────────────────────────────────────────────────────────────

def run_episode_raw(rng, freqs, n_steps, drift, n_samples, snr_db):
    """
    Run one episode and return raw-param trajectories alongside ground truth.
    """
    NAN_RAW = np.full(7, np.nan)

    base   = sample_circuit(rng)
    sequences, true_params_raw = simulate_sequence(rng, base, freqs, n_steps, drift, snr_db)
    true_raw = np.array([to_raw(p) for p in true_params_raw])   # (T, 7)

    step_ests = {m: [] for m in ['ecm_warm', 'gpf_causal']}
    done_ests = {}

    try:
        for ev in stream_request(sequences, n_samples=n_samples):
            if ev.get('done'):
                done_ests = parse_done_event(ev, n_steps)
            elif 'error' not in ev:
                parsed = parse_step_event(ev)
                for m in ['ecm_warm', 'gpf_causal']:
                    step_ests[m].append(
                        parsed[m]['raw'] if (m in parsed and 'raw' in parsed[m])
                        else NAN_RAW.copy()
                    )
    except Exception as e:
        return None

    estimates = {}
    for m in ['ecm_warm', 'gpf_causal']:
        arr = np.array(step_ests[m])
        if len(arr) == n_steps:
            estimates[m] = arr   # (T, 7)

    ks = done_ests.get('kal_gpf_smooth', {})
    if 'raw' in ks and len(ks['raw']) == n_steps:
        estimates['kal_gpf_smooth'] = ks['raw']   # (T, 7)

    return {'true_raw': true_raw, 'estimates': estimates}


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_scatter_panel(axes, all_true, all_pred, param_configs):
    """
    axes: (n_params, n_methods)  each cell is a scatter of true vs predicted.
    """
    methods = [m for m in METHODS if m in all_pred]

    for row, (idx, plabel) in enumerate(param_configs):
        true_col = all_true[:, idx]

        for col, m in enumerate(methods):
            ax  = axes[row][col]
            s   = METHOD_STYLE[m]
            pred_col = all_pred[m][:, idx]

            mask = ~(np.isnan(true_col) | np.isnan(pred_col))
            x, y = true_col[mask], pred_col[mask]

            ax.scatter(x, y, c=s['color'], s=12, alpha=0.4, linewidths=0)

            # identity line
            lim = [min(x.min(), y.min()) - 0.1, max(x.max(), y.max()) + 0.1]
            ax.plot(lim, lim, 'k-', lw=0.8, alpha=0.5)
            ax.set_xlim(lim); ax.set_ylim(lim)

            mae_naive = float(np.nanmean(np.abs(y - x)))
            ax.text(0.05, 0.92, f'MAE={mae_naive:.3f}', transform=ax.transAxes,
                    fontsize=7, color=s['color'], va='top')

            ax.tick_params(labelsize=7)
            ax.set_aspect('equal', 'box')
            ax.grid(True, alpha=0.3, lw=0.5)

            if row == 0:
                ax.set_title(s['label'], fontsize=9, color=s['color'])
            if col == 0:
                ax.set_ylabel(plabel + '\npredicted', fontsize=8)
            if row == len(param_configs) - 1:
                ax.set_xlabel('true', fontsize=8)


def plot_trajectory_panel(fig, gs_row, ep_data, case_idx, param_configs):
    """One row of subplots showing trajectory for each focus param."""
    true_raw  = ep_data['true_raw']    # (T, 7)
    estimates = ep_data['estimates']
    T = true_raw.shape[0]
    steps = np.arange(1, T + 1)

    for j, (idx, plabel) in enumerate(param_configs):
        ax = fig.add_subplot(gs_row[j])
        ax.plot(steps, true_raw[:, idx], **TRUTH_STYLE)
        for m, est in estimates.items():
            s = METHOD_STYLE[m]
            ax.plot(steps, est[:, idx],
                    color=s['color'], ls=s['ls'], lw=s['lw'])
        ax.set_xlim(1, T)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, lw=0.5)
        if case_idx == 0:
            ax.set_title(plabel, fontsize=9)
        if j == 0:
            ax.set_ylabel(f'Circuit {case_idx+1}', fontsize=8)
        if case_idx == N_CASE_STUDIES - 1:
            ax.set_xlabel('Step', fontsize=8)


def plot_mae_bars(ax_naive, ax_sym, all_true, all_pred):
    """Grouped bar chart: naive MAE and symmetric MAE per param per method."""
    methods   = [m for m in METHODS if m in all_pred]
    n_methods = len(methods)
    x = np.arange(len(FOCUS_PARAMS))
    width = 0.8 / n_methods

    for ax, metric_key, title in [
        (ax_naive, 'naive',     'Naive MAE  (Ra>Rb, Ca>Cb ordering)'),
        (ax_sym,   'symmetric', 'Symmetric MAE  (optimal Ra/Ca \u2194 Rb/Cb assignment)'),
    ]:
        for i, m in enumerate(methods):
            pred = all_pred[m]
            true = all_true
            mask = ~np.any(np.isnan(pred) | np.isnan(true), axis=1)
            result = symmetric_raw_mae(pred[mask], true[mask])
            maes   = [result[metric_key][idx] for idx, _ in FOCUS_PARAMS]
            s = METHOD_STYLE[m]
            bars = ax.bar(x + (i - n_methods/2 + 0.5) * width, maes,
                          width * 0.9, color=s['color'], label=s['label'],
                          edgecolor='k', lw=0.4, alpha=0.85)
            for bar, v in zip(bars, maes):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.002,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=6,
                        rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in FOCUS_PARAMS], fontsize=8)
        ax.set_ylabel('MAE (log10 decades)', fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
        ax.tick_params(labelsize=8)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    try:
        r = requests.get('http://localhost:5003/model_info', timeout=5)
        info = r.json()
        print(f"API: {info.get('model','?')}  param_space={info.get('param_space','?')}")
    except Exception as e:
        print(f"API not reachable: {e}"); sys.exit(1)

    rng   = np.random.default_rng(7)
    freqs = np.logspace(-1, 6, 100)

    case_studies = []
    all_episodes = []

    print(f"\nRunning {N_TOTAL_EPS} episodes (raw param tracking) ...")
    for ep in range(N_TOTAL_EPS):
        t0 = time.time()
        print(f"  ep {ep+1}/{N_TOTAL_EPS}...", end='', flush=True)
        data = run_episode_raw(rng, freqs, N_STEPS, DRIFT, N_SAMPLES, SNR_DB)
        elapsed = time.time() - t0
        if data is None or not data['estimates']:
            print(f" FAILED"); continue

        n_methods = len(data['estimates'])
        # quick per-method mean naive MAE on Ra only
        maes = {}
        for m, est in data['estimates'].items():
            maes[m] = float(np.nanmean(np.abs(est[:, I_Ra] - data['true_raw'][:, I_Ra])))
        print(f" {n_methods}m  {elapsed:.0f}s  "
              + '  '.join(f"{m.split('_')[0]}={v:.3f}" for m, v in maes.items()))

        all_episodes.append(data)
        if len(case_studies) < N_CASE_STUDIES:
            case_studies.append(data)

    if not all_episodes:
        print("No episodes succeeded."); sys.exit(1)

    # Flatten all timesteps for scatter + MAE bars
    avail = [m for m in METHODS if any(m in ep['estimates'] for ep in all_episodes)]
    all_true = np.concatenate([ep['true_raw'] for ep in all_episodes], axis=0)  # (N*T, 7)
    all_pred = {}
    for m in avail:
        arrs = [ep['estimates'][m] for ep in all_episodes if m in ep['estimates']]
        all_pred[m] = np.concatenate(arrs, axis=0)  # (N*T, 7) — may be shorter

    # Trim all_true to shortest length if methods have different episode counts
    min_len = min(len(v) for v in all_pred.values())
    all_true_trim = all_true[:min_len]
    all_pred_trim = {m: v[:min_len] for m, v in all_pred.items()}

    # ── figure layout ────────────────────────────────────────────────────────
    n_methods = len(avail)
    n_params  = len(FOCUS_PARAMS)

    fig = plt.figure(figsize=(5 * n_methods, 3 * (n_params + N_CASE_STUDIES + 2.5)))
    fig.suptitle(
        f'Raw Parameter Recovery: Ra, Rb, Ca, Cb\n'
        f'(fisher_v7  |  {len(all_episodes)} episodes x {N_STEPS} steps  '
        f'drift={DRIFT}  SNR={SNR_DB}dB)',
        fontsize=12, y=0.99,
    )

    total_rows = n_params + N_CASE_STUDIES + 2   # scatter + trajectories + 2 bar rows
    gs = gridspec.GridSpec(
        total_rows, max(n_methods, n_params),
        hspace=0.55, wspace=0.35,
        top=0.96, bottom=0.04,
    )

    # ── scatter panel ─────────────────────────────────────────────────────────
    scatter_axes = [[fig.add_subplot(gs[row, col]) for col in range(n_methods)]
                    for row in range(n_params)]

    # Add section label
    fig.text(0.005, 0.96 - (n_params / total_rows) * 0.9,
             'Scatter\n(true vs pred)', va='center', ha='left',
             fontsize=8, color='#555', rotation=90)

    plot_scatter_panel(scatter_axes, all_true_trim, all_pred_trim, FOCUS_PARAMS)

    # ── trajectory panels ─────────────────────────────────────────────────────
    fig.text(0.005, 0.96 - ((n_params + N_CASE_STUDIES / 2) / total_rows) * 0.9,
             'Trajectories', va='center', ha='left',
             fontsize=8, color='#555', rotation=90)

    for i, ep_data in enumerate(case_studies):
        row_idx = n_params + i
        gs_row = [gs[row_idx, j] for j in range(n_params)]
        plot_trajectory_panel(fig, gs_row, ep_data, i, FOCUS_PARAMS)

    # legend in the last trajectory row's last subplot (or a floating legend)
    handles = [plt.Line2D([0], [0], **TRUTH_STYLE)]
    for m in avail:
        s = METHOD_STYLE[m]
        handles.append(plt.Line2D([0], [0], color=s['color'],
                                   ls=s['ls'], lw=s['lw'], label=s['label']))
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.99, 0.97),
               fontsize=9, framealpha=0.9, ncol=1)

    # ── MAE bar charts ────────────────────────────────────────────────────────
    ax_naive = fig.add_subplot(gs[n_params + N_CASE_STUDIES, :n_params])
    ax_sym   = fig.add_subplot(gs[n_params + N_CASE_STUDIES + 1, :n_params])
    plot_mae_bars(ax_naive, ax_sym, all_true_trim, all_pred_trim)

    out_path = Path('results/raw_param_comparison.png')
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f"\nSaved: {out_path}")

    # ── print symmetric MAE table ─────────────────────────────────────────────
    raw_names_focus = ['Ra', 'Rb', 'Ca', 'Cb']
    print()
    print("=" * 68)
    print("SYMMETRIC MAE  (log10 decades) — optimal Ra/Ca <-> Rb/Cb assignment")
    print("=" * 68)
    col = 18
    header = f"{'Param':<8}" + "".join(f"{m:>{col}}" for m in avail)
    print(header); print("-" * len(header))
    for idx, name in zip([I_Ra, I_Rb, I_Ca, I_Cb, I_Rsh], ['Ra', 'Rb', 'Ca', 'Cb', 'Rsh']):
        row = f"{name:<8}"
        for m in avail:
            mask = ~np.any(np.isnan(all_pred_trim[m]) | np.isnan(all_true_trim), axis=1)
            res  = symmetric_raw_mae(all_pred_trim[m][mask], all_true_trim[mask])
            row += f"{res['symmetric'][idx]:>{col}.4f}"
        print(row)
    print("-" * len(header))

    print()
    print("SWAP FRACTION  (how often optimal assignment differs from Ra>Rb order)")
    for m in avail:
        mask = ~np.any(np.isnan(all_pred_trim[m]) | np.isnan(all_true_trim), axis=1)
        res  = symmetric_raw_mae(all_pred_trim[m][mask], all_true_trim[mask])
        print(f"  {m:<22} {res['swap_frac']*100:.1f}%")


if __name__ == '__main__':
    main()
