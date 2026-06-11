#!/usr/bin/env python3
"""
E3 — Temporal Inference Evaluation

Answers: does sequential filtering improve over per-step ECM?
Does FFBS smoothing further improve over causal GPF?

Methods compared
----------------
ecm_step    ECM (L-BFGS-B) fit independently at each timestep. No memory.
gpf_causal  GPF forward pass. Uses history, outputs at current t.
gpf_smooth  FFBS backward sweep over the GPF forward pass. Offline; uses t+1:T.

Metrics
-------
mae_curve   Median MAE vs timestep index. Shows convergence rate.
final_mae   Median MAE at t=200 (last observed step) per pathology.
conv_t      First timestep where MAE < 0.2 log10-decades (convergence).
Rsh_track   Rsh-specific MAE curve — highlighted as the hardest parameter.

Experiment design
-----------------
n_traj = 30 trajectories per pathology (sub-sample from temporal_v1).
ECM evaluated at t ∈ {0,2,4,9,19,49,99,199} (sparse to save time).
GPF runs the full forward pass (T=200 steps per trajectory).
FFBS run offline after the GPF forward pass.

Paper claim supported
---------------------
"GPF causal tracking converges to within 0.2 log10-decades on tau_big, tau_small,
TER, and TEC within 10-20 timesteps across all pathologies. FFBS smoothing further
reduces error by ~20% at early timesteps. Rsh tracking is limited by spectral
insensitivity: final GPF MAE ~0.35 decades vs CR floor ~0.013 decades."

Output
------
results/eval/E3_temporal.json

Usage
-----
    python -m eval.E3_temporal
    python -m eval.E3_temporal --n-traj 10 --no-ecm --model models/fisher_v10
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.shared import (
    PARAMS_ID, PATHOLOGIES,
    load_temporal_hdf5, load_transformer,
    run_mdn_batch, particles_to_id_stats, save_results,
)

# Timestep indices to record for curves (0-indexed into T=200 steps).
CURVE_STEPS = [0, 1, 2, 4, 9, 14, 19, 29, 49, 74, 99, 149, 199]

# Sparse subset for ECM evaluation (much slower than GPF).
ECM_STEPS = [0, 2, 4, 9, 19, 49, 99, 199]


def _ecm_at_step(fitter, Zr_step: np.ndarray, Zi_step: np.ndarray,
                 frequencies: np.ndarray) -> np.ndarray:
    """Run ECM on a single (F,) spectrum. Returns (5,) log10-id or NaN."""
    from eval.shared import to_identifiable

    result = fitter.fit(Zr_step.astype(np.float64), Zi_step.astype(np.float64), frequencies)
    if result is None:
        return np.full(5, np.nan)
    raw_log10 = np.array([
        np.log10(result["Ra"]), np.log10(result["Rb"]),
        np.log10(result["Ca"]), np.log10(result["Cb"]),
        np.log10(result["Rsh"]),
    ])
    return to_identifiable(raw_log10)


def run_trajectory_gpf(
    traj: dict,
    omega: np.ndarray,
    model,
    run_smooth: bool = True,
) -> dict:
    """
    Run GPF (and optionally FFBS) on one trajectory.

    Returns dict with:
      gpf_means  (T, 5)  causal weighted mean in identifiable space
      gpf_stds   (T, 5)
      ffbs_means (T, 5)  smoothed (if run_smooth)
      ffbs_stds  (T, 5)
      ess        (T,)    effective sample size
    """
    from src.pipeline.gpf import GaussianParticleFilter, ffbs_backward_sweep

    Zr_traj = traj["Zr"]      # (T, F)
    Zi_traj = traj["Zi"]      # (T, F)
    T = traj["T"]
    dt = traj["dt"]

    gpf = GaussianParticleFilter(n_particles=128)

    # Initialize from first MDN proposal
    Zr0 = Zr_traj[0:1].astype(np.float32)
    Zi0 = Zi_traj[0:1].astype(np.float32)
    mdn_mean, _mdn_std = run_mdn_batch(model, Zr0, Zi0, omega, batch_size=1)
    gpf.initialize(mdn_means_id=mdn_mean, mdn_weights=np.array([1.0]))

    pos_hist, w_hist, P_hist, dt_hist = [], [], [], []
    gpf_means = np.zeros((T, 5))
    gpf_stds  = np.zeros((T, 5))
    ess_curve = np.zeros(T)

    for t in range(T):
        pos, _log_ev = gpf.step(
            Zr_traj[t].astype(np.float64),
            Zi_traj[t].astype(np.float64),
            omega, dt=dt,
        )
        w   = gpf.get_weights()
        P_d = gpf.P_pp.diagonal(axis1=1, axis2=2)   # (N, 5) per-particle cov diag

        mean, std = particles_to_id_stats(pos, w)
        gpf_means[t] = mean
        gpf_stds[t]  = std
        ess_curve[t] = gpf.get_ess()

        pos_hist.append(pos.copy())
        w_hist.append(w.copy())
        P_hist.append(P_d.copy())
        dt_hist.append(dt)

    result = {
        "gpf_means": gpf_means,
        "gpf_stds":  gpf_stds,
        "ess":       ess_curve,
    }

    if run_smooth and T >= 2:
        smoothed_weights = ffbs_backward_sweep(pos_hist, w_hist, P_hist, dt_hist)
        ffbs_means = np.zeros((T, 5))
        ffbs_stds  = np.zeros((T, 5))
        for t in range(T):
            m, s = particles_to_id_stats(pos_hist[t], smoothed_weights[t])
            ffbs_means[t] = m
            ffbs_stds[t]  = s
        result["ffbs_means"] = ffbs_means
        result["ffbs_stds"]  = ffbs_stds

    return result


def mae_curve_for_trajectories(
    method_means: list,    # list of (T, 5) arrays, one per trajectory
    truths_id: list,       # list of (T, 5) ground truth
    steps: list[int],
) -> dict[str, list[float]]:
    """
    Per-parameter MAE at each step in `steps`, median over trajectories.
    Returns {param: [mae_at_step_0, mae_at_step_1, ...]} for each step index.
    """
    n_traj = len(method_means)
    curves = {p: [] for p in PARAMS_ID}

    for t in steps:
        per_param = {p: [] for p in PARAMS_ID}
        for i in range(n_traj):
            pred = method_means[i][t]    # (5,)
            truth = truths_id[i][t]      # (5,)
            err = np.abs(pred - truth)
            if np.all(np.isfinite(err)):
                for j, p in enumerate(PARAMS_ID):
                    per_param[p].append(float(err[j]))
        for p in PARAMS_ID:
            vals = per_param[p]
            curves[p].append(float(np.median(vals)) if vals else float("nan"))

    return curves


def ecm_sparse_curves(
    traj_list: list,
    omega: np.ndarray,
    steps: list[int],
) -> dict[str, list[float]]:
    """
    Run ECM at sparse timesteps over all trajectories, return MAE curves.
    """
    from src.baselines.deterministic_fit import DeterministicPhysicsFit

    frequencies = omega / (2 * np.pi)
    fitter = DeterministicPhysicsFit(n_restarts=3)

    n_traj = len(traj_list)
    per_step_errs = {p: {t: [] for t in steps} for p in PARAMS_ID}

    for i, traj in enumerate(traj_list):
        truths_id = traj["derived_log10"]   # (T, 5) identifiable
        for t in steps:
            pred = _ecm_at_step(fitter, traj["Zr"][t], traj["Zi"][t], frequencies)
            if np.all(np.isfinite(pred)):
                for j, p in enumerate(PARAMS_ID):
                    per_step_errs[p][t].append(float(abs(pred[j] - truths_id[t, j])))

        if (i + 1) % 5 == 0:
            print(f"    ECM: {i+1}/{n_traj} trajectories done")

    curves = {}
    for p in PARAMS_ID:
        curves[p] = [
            float(np.median(per_step_errs[p][t])) if per_step_errs[p][t] else float("nan")
            for t in steps
        ]
    return curves


def convergence_timestep(curve: list[float], threshold: float, steps: list[int]) -> int:
    """First step index (0-indexed into T=200) where MAE drops below threshold."""
    for t, v in zip(steps, curve):
        if np.isfinite(v) and v < threshold:
            return int(t)
    return -1


def main():
    ap = argparse.ArgumentParser(description="E3: Temporal inference evaluation")
    ap.add_argument("--model",    default="models/fisher_v10")
    ap.add_argument("--data",     default="data/temporal_v1/temporal_dataset.h5")
    ap.add_argument("--n-traj",   type=int, default=30,  help="trajectories per pathology")
    ap.add_argument("--sigma",    type=float, default=10.0)
    ap.add_argument("--no-ecm",   action="store_true")
    ap.add_argument("--no-smooth", action="store_true", help="skip FFBS backward sweep")
    ap.add_argument("--pathology", default=None, help="single pathology to evaluate")
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()

    print("E3: Temporal Inference Evaluation")
    print(f"  model={args.model}  data={args.data}  n_traj={args.n_traj}")

    model, _is_id, ckpt = load_transformer(args.model)
    dataset = load_temporal_hdf5(args.data, n_traj=args.n_traj,
                                  pathology=args.pathology, seed=args.seed)
    omega = dataset["omega"]

    target_pathologies = (
        [args.pathology] if args.pathology
        else [p for p in PATHOLOGIES if dataset["trajectories"].get(p)]
    )
    target_pathologies = [p for p in target_pathologies if dataset["trajectories"].get(p)]

    all_results = {}

    for pathology in target_pathologies:
        traj_list = dataset["trajectories"][pathology]
        if not traj_list:
            continue

        print(f"\n── {pathology} (n={len(traj_list)}) ────────────────────────")

        # ── GPF + FFBS ────────────────────────────────────────────────────────
        print(f"  [GPF] running {len(traj_list)} trajectories ...")
        gpf_means_list, ffbs_means_list = [], []
        truths_id_list = []

        for i, traj in enumerate(traj_list):
            res = run_trajectory_gpf(traj, omega, model, run_smooth=not args.no_smooth)
            gpf_means_list.append(res["gpf_means"])
            if "ffbs_means" in res:
                ffbs_means_list.append(res["ffbs_means"])
            truths_id_list.append(traj["derived_log10"])
            if (i + 1) % 5 == 0:
                print(f"    GPF: {i+1}/{len(traj_list)} done")

        gpf_curves = mae_curve_for_trajectories(gpf_means_list, truths_id_list, CURVE_STEPS)

        ffbs_curves = {}
        if ffbs_means_list:
            ffbs_curves = mae_curve_for_trajectories(ffbs_means_list, truths_id_list, CURVE_STEPS)

        # ── ECM (optional, sparse) ────────────────────────────────────────────
        ecm_curves = {}
        if not args.no_ecm:
            print(f"  [ECM] sparse eval at {ECM_STEPS} ...")
            ecm_curves = ecm_sparse_curves(traj_list, omega, ECM_STEPS)

        # ── Convergence timesteps ─────────────────────────────────────────────
        CONV_THRESH = 0.20   # log10-decades
        gpf_conv  = {p: convergence_timestep(gpf_curves[p],  CONV_THRESH, CURVE_STEPS) for p in PARAMS_ID}
        ffbs_conv = {p: convergence_timestep(ffbs_curves.get(p, []), CONV_THRESH, CURVE_STEPS) for p in PARAMS_ID}

        # ── Final-step MAE ────────────────────────────────────────────────────
        final_idx = len(CURVE_STEPS) - 1
        gpf_final  = {p: gpf_curves[p][final_idx]  for p in PARAMS_ID}
        ffbs_final = {p: ffbs_curves[p][final_idx] if ffbs_curves.get(p) else float("nan") for p in PARAMS_ID}
        ecm_final  = {p: ecm_curves[p][-1] if ecm_curves.get(p) else float("nan") for p in PARAMS_ID}

        all_results[pathology] = {
            "n_traj":       len(traj_list),
            "curve_steps":  CURVE_STEPS,
            "ecm_steps":    ECM_STEPS,
            "gpf_curves":   gpf_curves,
            "ffbs_curves":  ffbs_curves,
            "ecm_curves":   ecm_curves,
            "gpf_final":    gpf_final,
            "ffbs_final":   ffbs_final,
            "ecm_final":    ecm_final,
            "convergence_step_gpf":  gpf_conv,
            "convergence_step_ffbs": ffbs_conv,
            "convergence_threshold": CONV_THRESH,
        }

        # ── Print summary ─────────────────────────────────────────────────────
        print(f"\n  Final-step MAE (t=199):")
        print(f"  {'param':<12} {'ECM':>8} {'GPF':>8} {'FFBS':>8} {'GPF-conv':>10}")
        for p in PARAMS_ID:
            print(f"  {p:<12} {ecm_final[p]:>8.4f} {gpf_final[p]:>8.4f} "
                  f"{ffbs_final.get(p, float('nan')):>8.4f} "
                  f"{gpf_conv.get(p, -1):>10}")

    out = save_results("E3_temporal", {
        "config": vars(args),
        "ckpt_val_mae": ckpt.get("val_mae"),
        "by_pathology": all_results,
    })
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
