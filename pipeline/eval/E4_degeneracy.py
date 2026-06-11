#!/usr/bin/env python3
"""
E4 — Degeneracy and Fork Detection Evaluation

Answers three questions about the Ra↔Rb degeneracy problem:

Part A — Symmetric vs Standard MAE
    How much does the standard (non-symmetric) assignment penalty cost in raw-space
    metrics? Compare standard MAE vs optimal-assignment MAE on static test set.
    Ground truth: symmetric MAE captures true fitting accuracy; gap = labeling artifact.

Part B — Ratio tracking
    Can the GPF track the apical bias β = Ra/(Ra+Rb) ∈ (0,1)?
    β > 0.5: apical-dominant (Ra > Rb), β < 0.5: basolateral-dominant.
    Metric: RMSE of β over time, per pathology.
    Baseline: constant β = 0.5 (uniform prior = no information).

Part C — PTG fork detection
    For trajectories with a known mechanism switch (e.g., Rsh-decline at t=100):
    can the PTG detect the fork within 5 steps?
    Metric: detection timing error |t_detected - t_true|.
    Baseline: variance-threshold detector (flag when marginal variance exceeds 2σ).

    Uses synthetic piecewise trajectories:
      - Phase 1 (t < t_switch): healthy Rsh ~ 10kΩ
      - Phase 2 (t >= t_switch): Rsh drops 10× (barrier_breakdown)
    Two confounding mechanisms both decrease TER:
      (a) Rsh_decline:  Rsh ↓ (paracellular short)
      (b) Rb_increase:  Rb ↑ (basolateral membrane stiffens)
    The PTG should distinguish these at late timepoints.

Output
------
results/eval/E4_degeneracy.json

Usage
-----
    python -m eval.E4_degeneracy
    python -m eval.E4_degeneracy --no-ptg --n 500
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.shared import (
    PARAMS_ID, PARAMS_RAW,
    load_test_csv, load_transformer,
    run_mdn_batch, run_ecm_batch, particles_to_id_stats,
    symmetric_raw_mae, to_identifiable, save_results,
)


# ── Part A helpers ────────────────────────────────────────────────────────────

def standard_raw_mae(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Naive per-parameter MAE without assignment optimization."""
    diff = np.abs(pred - truth)
    valid = np.all(np.isfinite(diff), axis=1)
    if not valid.any():
        return {p: float("nan") for p in PARAMS_RAW}
    d = diff[valid]
    return {p: float(np.mean(d[:, j])) for j, p in enumerate(PARAMS_RAW)}


def part_a_symmetric_gain(
    ecm_pred_raw: np.ndarray,   # (N, 5) log10 raw predicted
    truth_raw: np.ndarray,      # (N, 5) log10 raw ground truth
) -> dict:
    """
    Compare standard vs symmetric MAE.

    Returns {standard, symmetric, fractional_gain_pct} per raw parameter.
    fractional_gain = (standard - symmetric) / standard * 100%.
    """
    std_mae = standard_raw_mae(ecm_pred_raw, truth_raw)
    sym_mae = symmetric_raw_mae(ecm_pred_raw, truth_raw)

    result = {}
    for p in PARAMS_RAW:
        s = std_mae.get(p, float("nan"))
        y = sym_mae.get(p, float("nan"))
        gain = (s - y) / s * 100.0 if s > 0 else float("nan")
        result[p] = {"standard": s, "symmetric": y, "gain_pct": gain}
    return result


# ── Part B helpers ────────────────────────────────────────────────────────────

def extract_beta(positions: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted mean of β = Ra/(Ra+Rb) in raw space.
    positions: (N, 5) log10 [Ra,Rb,Ca,Cb,Rsh]
    """
    Ra  = 10.0 ** positions[:, 0]
    Rb  = 10.0 ** positions[:, 1]
    beta = Ra / np.maximum(Ra + Rb, 1e-30)
    return float((weights * beta).sum())


def beta_truth(params_log10: np.ndarray) -> float:
    """True β = Ra/(Ra+Rb) from raw log10 params."""
    Ra  = 10.0 ** params_log10[0]
    Rb  = 10.0 ** params_log10[1]
    return float(Ra / max(Ra + Rb, 1e-30))


def run_beta_tracking(traj: dict, omega: np.ndarray, model) -> dict:
    """
    Run GPF on one trajectory, record β(t) predicted vs true.

    Returns {beta_pred, beta_true, rmse, baseline_rmse}.
    """
    from src.pipeline.gpf import GaussianParticleFilter

    T   = traj["T"]
    dt  = traj["dt"]
    Zr  = traj["Zr"]
    Zi  = traj["Zi"]
    params = traj["params_log10"]   # (T, 5) raw

    Zr0 = Zr[0:1].astype(np.float32)
    Zi0 = Zi[0:1].astype(np.float32)
    mdn_mean, _ = run_mdn_batch(model, Zr0, Zi0, omega, batch_size=1)

    gpf = GaussianParticleFilter(n_particles=128)
    gpf.initialize(mdn_means_id=mdn_mean, mdn_weights=np.array([1.0]))

    beta_pred = np.zeros(T)
    beta_true = np.zeros(T)

    for t in range(T):
        gpf.step(Zr[t].astype(np.float64), Zi[t].astype(np.float64), omega, dt=dt)
        w = gpf.get_weights()
        beta_pred[t] = extract_beta(gpf.positions, w)
        beta_true[t] = beta_truth(params[t])

    rmse = float(np.sqrt(np.mean((beta_pred - beta_true) ** 2)))
    baseline_rmse = float(np.sqrt(np.mean((0.5 - beta_true) ** 2)))   # predict β=0.5 always
    return {
        "beta_pred": beta_pred.tolist(),
        "beta_true": beta_true.tolist(),
        "rmse": rmse,
        "baseline_rmse": baseline_rmse,
    }


# ── Part C helpers ────────────────────────────────────────────────────────────

def make_synthetic_switch_traj(
    t_switch: int,
    T: int,
    mechanism: str,   # "rsh_decline" | "rb_increase"
    rng: np.random.Generator,
) -> dict:
    """
    Synthetic piecewise trajectory with a mechanism switch at t_switch.

    Phase 1: healthy steady state.
    Phase 2: one parameter shifts to disease state.

    Returns dict compatible with load_temporal_hdf5 trajectory format.
    """
    # Healthy state: typical RPE values in log10 Ω/F
    # Ra ~ 50 Ω, Rb ~ 30 Ω, Ca ~ 1e-5 F, Cb ~ 2e-5 F, Rsh ~ 10000 Ω
    p_healthy = np.array([1.699, 1.477, -5.0, -4.699, 4.0])   # log10

    params_log10 = np.zeros((T, 5))
    params_log10[:] = p_healthy

    # Add small Brownian motion (0.01 std per step)
    params_log10 += rng.normal(0, 0.01, (T, 5)).cumsum(axis=0) * 0.0

    if mechanism == "rsh_decline":
        # Rsh drops from 10kΩ (4.0) to 1kΩ (3.0) linearly after t_switch
        target_rsh = 3.0
        for t in range(t_switch, T):
            frac = (t - t_switch) / max(T - t_switch - 1, 1)
            params_log10[t, 4] = p_healthy[4] + frac * (target_rsh - p_healthy[4])

    elif mechanism == "rb_increase":
        # Rb increases from 30 Ω (1.477) to 3000 Ω (3.477): stiff basolateral
        target_rb = 3.477
        for t in range(t_switch, T):
            frac = (t - t_switch) / max(T - t_switch - 1, 1)
            params_log10[t, 1] = p_healthy[1] + frac * (target_rb - p_healthy[1])

    # Add small measurement noise to avoid degenerate Fisher
    params_log10 += rng.normal(0, 0.005, params_log10.shape)
    params_log10 = np.clip(params_log10, -6, 6)

    derived = np.array([to_identifiable(params_log10[t]) for t in range(T)])

    # Simulate impedance (noiseless for now — GPF adds its own noise model)
    from eval.shared import compute_impedance
    from eval.shared import PATHOLOGIES

    freq_min, freq_max = 0.1, 1e4
    frequencies = np.logspace(np.log10(freq_min), np.log10(freq_max), 100)
    omega = 2 * np.pi * frequencies

    Zr_traj = np.zeros((T, 100))
    Zi_traj = np.zeros((T, 100))
    for t in range(T):
        zr, zi = compute_impedance(params_log10[t], omega)
        snr = 40.0
        amp = np.sqrt(zr ** 2 + zi ** 2)
        scale = amp.mean() * 10 ** (-snr / 20.0)
        Zr_traj[t] = zr + rng.normal(0, scale, 100)
        Zi_traj[t] = zi + rng.normal(0, scale, 100)

    return {
        "params_log10":  params_log10,
        "derived_log10": derived,
        "Zr": Zr_traj,
        "Zi": Zi_traj,
        "dt": 3.0,
        "T":  T,
        "t_switch": t_switch,
        "mechanism": mechanism,
        "omega": omega,
    }


def variance_threshold_detector(
    gpf_stds_curve: np.ndarray,   # (T, 5) per-step GPF std in identifiable space
    window: int = 5,
    sigma_mult: float = 2.0,
) -> int:
    """
    Detect a regime change when std jumps above baseline + sigma_mult * sigma_baseline.
    Baseline: rolling mean of first window steps.
    Returns detected timestep or -1 if never triggered.
    """
    if len(gpf_stds_curve) < window + 1:
        return -1

    baseline_std = gpf_stds_curve[:window].mean(axis=0)   # (5,)
    baseline_mean = gpf_stds_curve[:window].mean(axis=0)

    for t in range(window, len(gpf_stds_curve)):
        current = gpf_stds_curve[t]
        if np.any(current > baseline_mean + sigma_mult * baseline_std + 1e-6):
            return int(t)
    return -1


def run_ptg_on_trajectory(traj: dict, omega: np.ndarray, model) -> dict:
    """
    Run GPF forward pass and PTG fork detection on a synthetic trajectory.

    Returns:
      gpf_stds   (T, 5)
      variance_detect_t   timestep flagged by variance detector
      t_switch            true switch time
      mechanism           rsh_decline | rb_increase
    """
    from src.pipeline.gpf import GaussianParticleFilter, ffbs_backward_sweep
    from src.pipeline.ptg import build_ptg

    T   = traj["T"]
    dt  = traj["dt"]
    Zr  = traj["Zr"]
    Zi  = traj["Zi"]

    Zr0 = Zr[0:1].astype(np.float32)
    Zi0 = Zi[0:1].astype(np.float32)
    mdn_mean, _ = run_mdn_batch(model, Zr0, Zi0, omega, batch_size=1)

    gpf = GaussianParticleFilter(n_particles=128)
    gpf.initialize(mdn_means_id=mdn_mean, mdn_weights=np.array([1.0]))

    pos_hist, w_hist, P_hist, dt_hist = [], [], [], []
    gpf_stds = np.zeros((T, 5))

    for t in range(T):
        pos, _ = gpf.step(Zr[t].astype(np.float64), Zi[t].astype(np.float64), omega, dt=dt)
        w = gpf.get_weights()
        P_d = gpf.P_pp.diagonal(axis1=1, axis2=2)
        _, std = particles_to_id_stats(pos, w)
        gpf_stds[t] = std

        pos_hist.append(pos.copy())
        w_hist.append(w.copy())
        P_hist.append(P_d.copy())
        dt_hist.append(dt)

    smoothed = ffbs_backward_sweep(pos_hist, w_hist, P_hist, dt_hist)
    ptg_result = build_ptg(pos_hist, smoothed)

    # PTG fork detection: first fork event timestep
    ptg_detect_t = -1
    if ptg_result.forks:
        ptg_detect_t = int(min(f.t for f in ptg_result.forks))

    var_detect_t = variance_threshold_detector(gpf_stds)

    return {
        "t_switch": traj["t_switch"],
        "mechanism": traj["mechanism"],
        "ptg_detect_t": ptg_detect_t,
        "var_detect_t": var_detect_t,
        "ptg_timing_error": abs(ptg_detect_t - traj["t_switch"]) if ptg_detect_t >= 0 else -1,
        "var_timing_error": abs(var_detect_t - traj["t_switch"]) if var_detect_t >= 0 else -1,
    }


def main():
    ap = argparse.ArgumentParser(description="E4: Degeneracy and fork detection")
    ap.add_argument("--model",   default="models/fisher_v10")
    ap.add_argument("--data",    default="data/mixed_distribution_v2")
    ap.add_argument("--n",       type=int, default=500, help="static test samples for Part A")
    ap.add_argument("--n-ecm",   type=int, default=300, help="ECM subsample for Part A")
    ap.add_argument("--h5",      default="data/temporal_v1/temporal_dataset.h5")
    ap.add_argument("--n-traj",  type=int, default=20,  help="trajectories per pathology, Part B")
    ap.add_argument("--n-synth", type=int, default=20,  help="synthetic trajectories per mechanism, Part C")
    ap.add_argument("--no-ptg",  action="store_true")
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    print("E4: Degeneracy and Fork Detection Evaluation")
    model, _is_id, ckpt = load_transformer(args.model)
    rng = np.random.default_rng(args.seed)

    # ── Part A: Symmetric vs Standard MAE ────────────────────────────────────
    print(f"\n[Part A] Symmetric vs standard MAE (n={args.n_ecm} ECM) ...")
    static = load_test_csv(args.data, n_samples=args.n, seed=args.seed)
    omega_static = static["omega"]

    ecm_idx = rng.choice(static["n"], min(args.n_ecm, static["n"]), replace=False)
    ecm_pred_id, ecm_success = run_ecm_batch(
        static["Zr"][ecm_idx], static["Zi"][ecm_idx], omega_static,
    )

    # We need raw ECM predictions — re-run and capture raw params
    from src.baselines.deterministic_fit import DeterministicPhysicsFit
    fitter = DeterministicPhysicsFit(n_restarts=3)
    frequencies_static = omega_static / (2 * np.pi)
    ecm_raw_pred = np.full((len(ecm_idx), 5), np.nan)
    for k, i in enumerate(ecm_idx):
        result = fitter.fit(
            static["Zr"][i].astype(np.float64),
            static["Zi"][i].astype(np.float64),
            frequencies_static,
        )
        if result is not None:
            ecm_raw_pred[k] = [
                np.log10(result["Ra"]), np.log10(result["Rb"]),
                np.log10(result["Ca"]), np.log10(result["Cb"]),
                np.log10(result["Rsh"]),
            ]

    valid_ecm = np.all(np.isfinite(ecm_raw_pred), axis=1)
    part_a = part_a_symmetric_gain(
        ecm_raw_pred[valid_ecm],
        static["params_raw"][ecm_idx][valid_ecm],
    )
    print(f"  ECM success rate: {valid_ecm.mean()*100:.1f}%")
    print(f"\n  {'param':<6} {'standard':>10} {'symmetric':>10} {'gain %':>8}")
    for p in PARAMS_RAW:
        d = part_a.get(p, {})
        print(f"  {p:<6} {d.get('standard', float('nan')):>10.4f} "
              f"{d.get('symmetric', float('nan')):>10.4f} "
              f"{d.get('gain_pct', float('nan')):>8.1f}%")

    # ── Part B: Beta tracking ─────────────────────────────────────────────────
    print(f"\n[Part B] β = Ra/(Ra+Rb) tracking (n_traj={args.n_traj}) ...")
    from eval.shared import load_temporal_hdf5

    try:
        dataset = load_temporal_hdf5(args.h5, n_traj=args.n_traj, seed=args.seed)
    except Exception as e:
        print(f"  Could not load temporal data: {e}. Skipping Part B.")
        dataset = None

    part_b = {}
    if dataset:
        target_pats = [p for p in ["healthy", "apical_injury", "basolateral_injury"]
                       if dataset["trajectories"].get(p)]
        for pat in target_pats:
            traj_list = dataset["trajectories"][pat]
            if not traj_list:
                continue
            print(f"  {pat}: {len(traj_list)} trajectories")
            rmses, baseline_rmses = [], []
            for i, traj in enumerate(traj_list):
                res = run_beta_tracking(traj, dataset["omega"], model)
                rmses.append(res["rmse"])
                baseline_rmses.append(res["baseline_rmse"])
            part_b[pat] = {
                "rmse_median":      float(np.median(rmses)),
                "baseline_rmse":    float(np.median(baseline_rmses)),
                "improvement_pct":  float((np.median(baseline_rmses) - np.median(rmses)) /
                                          max(np.median(baseline_rmses), 1e-9) * 100),
            }
            print(f"    RMSE={np.median(rmses):.4f}  baseline={np.median(baseline_rmses):.4f}")

    # ── Part C: PTG fork detection ────────────────────────────────────────────
    part_c = {}
    if not args.no_ptg:
        print(f"\n[Part C] PTG fork detection ({args.n_synth} synthetic trajectories / mechanism)")
        T_synth = 120
        t_switch = 60

        freq_min, freq_max = 0.1, 1e4
        frequencies_synth = np.logspace(np.log10(freq_min), np.log10(freq_max), 100)
        omega_synth = 2 * np.pi * frequencies_synth

        for mechanism in ["rsh_decline", "rb_increase"]:
            print(f"  {mechanism}:")
            ptg_errors, var_errors = [], []
            ptg_detected, var_detected = 0, 0

            for i in range(args.n_synth):
                traj = make_synthetic_switch_traj(t_switch, T_synth, mechanism, rng)
                try:
                    res = run_ptg_on_trajectory(traj, omega_synth, model)
                    if res["ptg_detect_t"] >= 0:
                        ptg_errors.append(res["ptg_timing_error"])
                        ptg_detected += 1
                    if res["var_detect_t"] >= 0:
                        var_errors.append(res["var_timing_error"])
                        var_detected += 1
                except Exception as exc:
                    print(f"    trajectory {i} failed: {exc}")

            part_c[mechanism] = {
                "n":               args.n_synth,
                "t_switch":        t_switch,
                "ptg_detect_rate": ptg_detected / args.n_synth,
                "var_detect_rate": var_detected / args.n_synth,
                "ptg_timing_error_median": float(np.median(ptg_errors)) if ptg_errors else float("nan"),
                "var_timing_error_median": float(np.median(var_errors)) if var_errors else float("nan"),
            }
            print(f"    PTG: detected {ptg_detected}/{args.n_synth}  "
                  f"timing_error={part_c[mechanism]['ptg_timing_error_median']:.1f}")
            print(f"    Var: detected {var_detected}/{args.n_synth}  "
                  f"timing_error={part_c[mechanism]['var_timing_error_median']:.1f}")
    else:
        print("\n[Part C] PTG skipped (--no-ptg)")

    out = save_results("E4_degeneracy", {
        "config": vars(args),
        "ckpt_val_mae": ckpt.get("val_mae"),
        "part_a_symmetric_gain": part_a,
        "part_b_beta_tracking":  part_b,
        "part_c_fork_detection": part_c,
    })
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
