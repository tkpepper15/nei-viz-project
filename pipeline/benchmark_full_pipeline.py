#!/usr/bin/env python3
"""
Full-pipeline temporal benchmark.

Generates synthetic drifting-circuit sequences, sends them to the live API
at localhost:5003/mc_temporal_analysis_stream, and compares three outputs
against ground truth in identifiable log10 space:

  ecm_warm      : L-BFGS-B warm-started from previous step (per-step payload 'ecm')
  gpf_causal    : GPF current-step cloud median (per-step payload 'predictions')
  kal_gpf_smooth: Kalman RTS smoother over GPF history (done payload 'kal_gpf')

Identifiable params: [tau_big, tau_small, TER, TEC, Rsh]  (log10 SI units)
"""

import sys
import json
import time
import warnings
import numpy as np
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from src.pipeline.gpf import BOUNDS_LOW, BOUNDS_HIGH

API_URL = "http://localhost:5003/mc_temporal_analysis_stream"

ID_NAMES  = ['tau_big', 'tau_small', 'TER', 'TEC', 'Rsh']
RAW_NAMES = ['Ra', 'Rb', 'Ca', 'Cb', 'Rsh', 'tau_a', 'tau_b']

METHODS = [
    'ecm_warm',
    'gpf_causal',
    'kal_gpf_smooth',
]


# ── circuit model ─────────────────────────────────────────────────────────────

def compute_impedance(params_log10, omega):
    Ra, Rb, Ca, Cb, Rsh = 10.0 ** params_log10
    Za = Ra / (1 + 1j * omega * Ra * Ca)
    Zb = Rb / (1 + 1j * omega * Rb * Cb)
    Zs = Za + Zb
    Z  = (Rsh * Zs) / (Rsh + Zs)
    return Z.real, Z.imag


def to_identifiable(log10_params):
    Ra, Rb, Ca, Cb, Rsh = 10.0 ** log10_params
    tau_a = Ra * Ca
    tau_b = Rb * Cb
    TER  = Rsh * (Ra + Rb) / (Rsh + Ra + Rb + 1e-30)
    TEC  = Ca * Cb / (Ca + Cb + 1e-30)
    return np.array([
        np.log10(max(tau_a, tau_b)),
        np.log10(min(tau_a, tau_b)),
        np.log10(TER),
        np.log10(TEC),
        np.log10(Rsh),
    ])


def to_raw(log10_params):
    """
    Return log10 [Ra, Rb, Ca, Cb, Rsh, tau_a, tau_b] for ground-truth evaluation.
    Ra > Rb and Ca > Cb are enforced by the sampling convention.
    tau_a = Ra*Ca (apical), tau_b = Rb*Cb (basolateral) — direct assignment, not max/min.
    """
    Ra_l, Rb_l, Ca_l, Cb_l, Rsh_l = log10_params
    Ra, Rb, Ca, Cb = 10.0 ** Ra_l, 10.0 ** Rb_l, 10.0 ** Ca_l, 10.0 ** Cb_l
    return np.array([
        Ra_l, Rb_l, Ca_l, Cb_l, Rsh_l,
        np.log10(Ra * Ca),   # tau_a
        np.log10(Rb * Cb),   # tau_b
    ])


def sample_circuit(rng):
    """
    Sample a random circuit with Ra > Rb and Ca > Cb (biological prior).
    Ensures tau ratio >= 2 to maintain well-separated time constants.
    """
    for _ in range(10000):
        p = rng.uniform(BOUNDS_LOW, BOUNDS_HIGH)
        # Enforce Ra > Rb and Ca > Cb by sorting
        if p[0] < p[1]:
            p[0], p[1] = p[1], p[0]
        if p[2] < p[3]:
            p[2], p[3] = p[3], p[2]
        Ra, Rb, Ca, Cb = [10.0 ** p[i] for i in range(4)]
        ratio = max(Ra * Ca, Rb * Cb) / (min(Ra * Ca, Rb * Cb) + 1e-30)
        if ratio >= 2.0:
            return p
    raise RuntimeError("sample_circuit failed")


def simulate_sequence(rng, base_log10, freqs, n_steps, drift_std, snr_db=40.0):
    omega = 2 * np.pi * freqs
    drift_per_param = np.array([1.0, 1.0, 0.5, 0.5, 1.0]) * drift_std

    log10_true = base_log10.copy()
    obs, true_params = [], []

    for t in range(n_steps):
        if t > 0:
            log10_true = np.clip(log10_true + rng.normal(0, drift_per_param), BOUNDS_LOW, BOUNDS_HIGH)

        Zr, Zi = compute_impedance(log10_true, omega)
        Z_amp = np.sqrt(Zr ** 2 + Zi ** 2)
        sigma = Z_amp * 10 ** (-snr_db / 20.0)
        Zr_n = Zr + rng.normal(0, sigma)
        Zi_n = Zi + rng.normal(0, sigma)

        obs.append({'Z_real': Zr_n.tolist(), 'Z_imag': Zi_n.tolist(),
                    'frequencies': freqs.tolist(), 'time_min': float(t)})
        true_params.append(log10_true.copy())

    return obs, true_params


# ── SSE stream parser ─────────────────────────────────────────────────────────

def stream_request(sequences, n_samples=200):
    """
    POST to the API and yield parsed JSON events from the SSE stream.
    """
    payload = {
        'sequences':     sequences,
        'n_samples':     n_samples,
        'include_ecm':   True,
        'use_sequential': True,
    }
    with requests.post(API_URL, json=payload, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        buf = ''
        for chunk in resp.iter_content(chunk_size=None):
            buf += chunk.decode('utf-8', errors='replace')
            while '\n\n' in buf:
                event, buf = buf.split('\n\n', 1)
                for line in event.split('\n'):
                    if line.startswith('data:'):
                        try:
                            yield json.loads(line[5:].strip())
                        except json.JSONDecodeError:
                            pass


# ── extract estimates from event ─────────────────────────────────────────────

def _ecm_to_raw7(ecm: dict) -> np.ndarray:
    """
    Parse ECM dict and return log10 [Ra, Rb, Ca, Cb, Rsh, tau_a, tau_b].
    ECM units: Ra/Rb/R2/TER in kOhm, Ca/Cb/TEC in uF.
    Enforce Ra > Rb ordering for consistency with the biological prior.
    """
    Ra  = float(ecm['Ra']) * 1000
    Rb  = float(ecm['Rb']) * 1000
    Ca  = float(ecm['Ca']) / 1e6
    Cb  = float(ecm['Cb']) / 1e6
    Rsh = float(ecm['R2']) * 1000
    if Ra < Rb:   # ECM may return either ordering; enforce Ra > Rb
        Ra, Rb, Ca, Cb = Rb, Ra, Cb, Ca
    return np.array([
        np.log10(Ra), np.log10(Rb), np.log10(Ca), np.log10(Cb), np.log10(Rsh),
        np.log10(Ra * Ca), np.log10(Rb * Cb),
    ])


def _ecm_to_id5(ecm: dict) -> np.ndarray:
    """Parse ECM dict → log10 identifiable (5,)."""
    Ra  = float(ecm['Ra']) * 1000
    Rb  = float(ecm['Rb']) * 1000
    Ca  = float(ecm['Ca']) / 1e6
    Cb  = float(ecm['Cb']) / 1e6
    Rsh = float(ecm['R2']) * 1000
    TER = float(ecm['TER']) * 1000
    TEC = float(ecm['TEC']) / 1e6
    tau_a = Ra * Ca; tau_b = Rb * Cb
    return np.array([
        np.log10(max(tau_a, tau_b)),
        np.log10(min(tau_a, tau_b)),
        np.log10(TER), np.log10(TEC), np.log10(Rsh),
    ])


def _pred_to_raw7(pred: dict) -> np.ndarray:
    """
    Parse GPF predictions dict → log10 [Ra, Rb, Ca, Cb, Rsh, tau_a, tau_b].
    With enforce_ordering=True in GPF, Ra > Rb is already guaranteed.
    """
    Ra  = float(pred['Ra']['mean']) * 1000
    Rb  = float(pred['Rb']['mean']) * 1000
    Ca  = float(pred['Ca']['mean']) / 1e6
    Cb  = float(pred['Cb']['mean']) / 1e6
    Rsh = float(pred['R2']['mean']) * 1000
    return np.array([
        np.log10(Ra), np.log10(Rb), np.log10(Ca), np.log10(Cb), np.log10(Rsh),
        np.log10(Ra * Ca), np.log10(Rb * Cb),
    ])


def _pred_to_id5(pred: dict) -> np.ndarray:
    """Parse GPF predictions dict → log10 identifiable (5,)."""
    Ra  = float(pred['Ra']['mean']) * 1000
    Rb  = float(pred['Rb']['mean']) * 1000
    Ca  = float(pred['Ca']['mean']) / 1e6
    Cb  = float(pred['Cb']['mean']) / 1e6
    TER = float(pred['TER']['mean']) * 1000
    TEC = float(pred['TEC']['mean']) / 1e6
    Rsh = float(pred['R2']['mean']) * 1000
    tau_a = Ra * Ca; tau_b = Rb * Cb
    return np.array([
        np.log10(max(tau_a, tau_b)),
        np.log10(min(tau_a, tau_b)),
        np.log10(TER), np.log10(TEC), np.log10(Rsh),
    ])


def parse_step_event(ev):
    """
    Extract identifiable-space and raw-space log10 estimates from a per-step SSE event.
    Returns dict method -> {'id': (5,), 'raw': (7,)} arrays.
    """
    out = {}

    ecm = ev.get('ecm')
    if ecm and ecm.get('TER') is not None:
        try:
            out['ecm_warm'] = {'id': _ecm_to_id5(ecm), 'raw': _ecm_to_raw7(ecm)}
        except (TypeError, ValueError, KeyError):
            pass

    pred = ev.get('predictions')
    if pred:
        try:
            out['gpf_causal'] = {'id': _pred_to_id5(pred), 'raw': _pred_to_raw7(pred)}
        except (TypeError, ValueError, KeyError):
            pass

    return out


def parse_done_event(ev, T):
    """
    Extract Kalman smoother estimates from the done event.
    Returns dict method -> {'id': (T, 5), 'raw': (T, 7)} arrays.
    """
    out = {}

    krb = ev.get('kal_gpf')
    if krb and krb.get('mu_smoothed') is not None:
        try:
            mu_id = np.array(krb['mu_smoothed'])      # (T, 5) log10 identifiable
            result: dict = {'id': mu_id}

            if krb.get('mu_smoothed_raw') is not None:
                mu_raw5 = np.array(krb['mu_smoothed_raw'])  # (T, 5) log10 [Ra,Rb,Ca,Cb,Rsh]
                # Append tau_a, tau_b as derived columns
                Ra_l = mu_raw5[:, 0]; Rb_l = mu_raw5[:, 1]
                Ca_l = mu_raw5[:, 2]; Cb_l = mu_raw5[:, 3]
                tau_a_l = Ra_l + Ca_l  # log10(Ra*Ca) = log10(Ra)+log10(Ca)
                tau_b_l = Rb_l + Cb_l
                mu_raw7 = np.column_stack([mu_raw5, tau_a_l, tau_b_l])  # (T, 7)
                result['raw'] = mu_raw7

            out['kal_gpf_smooth'] = result
        except (TypeError, ValueError):
            pass

    return out


# ── per-episode runner ────────────────────────────────────────────────────────

def run_episode(rng, freqs, n_steps, drift_std, n_samples=200, snr_db=40.0):
    base = sample_circuit(rng)
    sequences, true_params = simulate_sequence(rng, base, freqs, n_steps, drift_std, snr_db)
    true_id  = np.array([to_identifiable(p) for p in true_params])  # (T, 5)
    true_raw = np.array([to_raw(p)          for p in true_params])  # (T, 7)

    NAN_ID  = np.full(5, np.nan)
    NAN_RAW = np.full(7, np.nan)

    step_ests_id  = {m: [] for m in ['ecm_warm', 'gpf_causal']}
    step_ests_raw = {m: [] for m in ['ecm_warm', 'gpf_causal']}
    done_ests = {}
    T_seen = 0

    try:
        for ev in stream_request(sequences, n_samples=n_samples):
            if ev.get('done'):
                done_ests = parse_done_event(ev, n_steps)
            elif 'error' in ev:
                pass
            else:
                parsed = parse_step_event(ev)
                for m in ['ecm_warm', 'gpf_causal']:
                    if m in parsed:
                        step_ests_id[m].append(parsed[m]['id'])
                        step_ests_raw[m].append(parsed[m]['raw'])
                    else:
                        step_ests_id[m].append(NAN_ID.copy())
                        step_ests_raw[m].append(NAN_RAW.copy())
                T_seen += 1
    except Exception as e:
        print(f"    stream error: {e}")
        return None, None, true_id, true_raw

    # id_errors: (T, 5),  raw_errors: (T, 7)
    id_errors  = {}
    raw_errors = {}

    for m in ['ecm_warm', 'gpf_causal']:
        arr_id  = np.array(step_ests_id[m])   # (T, 5)
        arr_raw = np.array(step_ests_raw[m])  # (T, 7)
        if len(arr_id) == T_seen and T_seen > 0:
            pad_id  = np.full((n_steps - T_seen, 5), np.nan)
            pad_raw = np.full((n_steps - T_seen, 7), np.nan)
            if len(pad_id):
                arr_id  = np.vstack([arr_id,  pad_id])
                arr_raw = np.vstack([arr_raw, pad_raw])
            id_errors[m]  = np.abs(arr_id  - true_id)
            raw_errors[m] = np.abs(arr_raw - true_raw)

    ks = done_ests.get('kal_gpf_smooth', {})
    if 'id' in ks and len(ks['id']) == n_steps:
        id_errors['kal_gpf_smooth']  = np.abs(ks['id'] - true_id)
    if 'raw' in ks and len(ks['raw']) == n_steps:
        raw_errors['kal_gpf_smooth'] = np.abs(ks['raw'] - true_raw)

    return id_errors, raw_errors, true_id, true_raw


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', type=int, default=20)
    parser.add_argument('--n-steps',    type=int, default=25)
    parser.add_argument('--drift',      type=float, default=0.03)
    parser.add_argument('--snr-db',     type=float, default=40.0)
    parser.add_argument('--n-samples',  type=int, default=200)
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    # Verify API up
    try:
        r = requests.get('http://localhost:5003/model_info', timeout=5)
        info = r.json()
        print(f"API: {info.get('model','?')} loaded")
    except Exception as e:
        print(f"API not reachable: {e}")
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    freqs = np.logspace(-1, 6, 100)

    all_id_errors  = {m: [] for m in METHODS}
    all_raw_errors = {m: [] for m in METHODS}
    step_errors    = {m: np.zeros((args.n_steps, 5)) for m in METHODS}
    step_counts    = np.zeros(args.n_steps)
    ep_times       = []

    print(f"\n{args.n_episodes} episodes x {args.n_steps} steps  "
          f"drift={args.drift}  SNR={args.snr_db}dB  n_samples={args.n_samples}")
    print()

    for ep in range(args.n_episodes):
        t0 = time.time()
        id_errors, raw_errors, true_id, true_raw = run_episode(
            rng, freqs, args.n_steps, args.drift,
            n_samples=args.n_samples, snr_db=args.snr_db,
        )
        ep_time = time.time() - t0
        ep_times.append(ep_time)

        if id_errors is None:
            print(f"  ep {ep+1}: FAILED")
            continue

        for m in METHODS:
            if m in id_errors:
                all_id_errors[m].append(id_errors[m])
                step_errors[m] += id_errors[m]
            if m in raw_errors:
                all_raw_errors[m].append(raw_errors[m])
        step_counts += 1

        ep_maes = {m: float(np.nanmean(id_errors[m])) for m in METHODS if m in id_errors}
        best    = min(ep_maes, key=ep_maes.get) if ep_maes else '?'
        mae_str = '  '.join(f"{m}={v:.3f}" for m, v in ep_maes.items())
        print(f"  ep {ep+1:>2}/{args.n_episodes}  ({ep_time:.0f}s)  {mae_str}  <- best: {best}")

    print()
    available_id  = [m for m in METHODS if all_id_errors[m]]
    available_raw = [m for m in METHODS if all_raw_errors[m]]
    col_w = 16

    def _print_table(title, names, all_err, available):
        print("=" * 80)
        print(title)
        print("=" * 80)
        header = f"{'Param':<12}" + "".join(f"{m:>{col_w}}" for m in available)
        print(header)
        print("-" * len(header))
        for i, name in enumerate(names):
            row = f"{name:<12}"
            for m in available:
                arr = np.concatenate(all_err[m], axis=0)
                mae = float(np.nanmean(arr[:, i]))
                row += f"{mae:>{col_w}.4f}"
            print(row)
        print("-" * len(header))
        row = f"{'MEAN':<12}"
        for m in available:
            arr = np.concatenate(all_err[m], axis=0)
            row += f"{float(np.nanmean(arr)):>{col_w}.4f}"
        print(row)

    # ── identifiable space table ───────────────────────────────────────────
    _print_table("IDENTIFIABLE SPACE MAE  (log10 decades)", ID_NAMES,
                 all_id_errors, available_id)

    # ── raw parameter space table ──────────────────────────────────────────
    if available_raw:
        print()
        _print_table("RAW PARAMETER MAE  (log10 decades, Ra>Rb and Ca>Cb enforced)",
                     RAW_NAMES, all_raw_errors, available_raw)

    # ── improvement vs ECM warm (identifiable) ─────────────────────────────
    if 'ecm_warm' in available_id:
        ecm_arr = np.concatenate(all_id_errors['ecm_warm'], axis=0)
        print()
        print("IMPROVEMENT vs ECM warm  (positive = better, identifiable space)")
        others  = [m for m in available_id if m != 'ecm_warm']
        header2 = f"{'Param':<12}" + "".join(f"{m:>{col_w}}" for m in others)
        print(header2)
        print("-" * len(header2))
        for i, name in enumerate(ID_NAMES):
            base = float(np.nanmean(ecm_arr[:, i]))
            row  = f"{name:<12}"
            for m in others:
                arr = np.concatenate(all_id_errors[m], axis=0)
                mae = float(np.nanmean(arr[:, i]))
                pct = 100.0 * (base - mae) / (base + 1e-12)
                row += f"{pct:>+{col_w}.1f}%"
            print(row)
        row  = f"{'MEAN':<12}"
        base = float(np.nanmean(ecm_arr))
        for m in others:
            arr = np.concatenate(all_id_errors[m], axis=0)
            mae = float(np.nanmean(arr))
            pct = 100.0 * (base - mae) / (base + 1e-12)
            row += f"{pct:>+{col_w}.1f}%"
        print(row)

    # ── per-step convergence ───────────────────────────────────────────────
    print()
    print("PER-STEP MEAN MAE  (identifiable space, mean over params and episodes)")
    header3 = f"{'Step':>5}" + "".join(f"{m:>{col_w}}" for m in available_id)
    print(header3)
    print("-" * len(header3))
    for t in range(args.n_steps):
        row = f"{t+1:>5}"
        for m in available_id:
            mae = float(np.nanmean(step_errors[m][t] / step_counts[t])) if step_counts[t] > 0 else float('nan')
            row += f"{mae:>{col_w}.4f}"
        print(row)

    # ── timing ────────────────────────────────────────────────────────────
    total   = sum(ep_times)
    n_total = len(ep_times) * args.n_steps
    print()
    print(f"Wall time: {total:.0f}s  "
          f"({1000*total/max(n_total,1):.0f}ms/step  "
          f"{total/max(len(ep_times),1):.0f}s/episode)")


if __name__ == '__main__':
    main()
