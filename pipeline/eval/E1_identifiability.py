#!/usr/bin/env python3
"""
E1 — Identifiability Analysis

Answers: what information is theoretically available in an EIS spectrum?

Experiments
-----------
1. CR-bound distribution per parameter across the test set.
   Shows which parameters are identifiable (small bounds) vs degenerate (large bounds).

2. Effective Fisher rank per circuit.
   Confirms the structural rank deficiency (5 params, effective rank ≈ 3).

3. Identifiability phase diagram: CR bounds stratified by
   tau_ratio (arc separation) × rsh_ratio (Rsh/TER coupling).
   Shows when each parameter becomes unidentifiable and why.

Paper claim supported
---------------------
"The RCRC EIS inverse problem is structurally non-identifiable: the Fisher matrix
has effective rank 3, with Ra and Rb (and Ca and Cb) forming a degenerate pair.
The identifiable subspace spans [tau_big, tau_small, TER, TEC, Rsh], but even
within this space identifiability is conditional on circuit operating point."

Output
------
results/eval/E1_identifiability.json

Usage
-----
    python -m eval.E1_identifiability
    python -m eval.E1_identifiability --n 2000 --sigma 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.shared import (
    PARAMS_ID, PATHOLOGIES,
    load_test_csv, compute_cr_bounds_subsample,
    identifiability_strata, save_results,
)


def compute_fisher_rank_distribution(params_log10: np.ndarray, omega: np.ndarray,
                                     sigma: float, n: int) -> dict:
    """
    Compute effective rank distribution over n sampled circuits.

    Effective rank = number of eigenvalues > 1% of max eigenvalue.
    For the RCRC circuit we expect mode = 3 (structural null dimension = 2).
    """
    from src.physics.eis_fisher import compute_fisher_information
    import torch

    rng = np.random.default_rng(0)
    idx = rng.choice(len(params_log10), min(n, len(params_log10)), replace=False)
    ranks = []

    omega_t = torch.tensor(omega, dtype=torch.float32)
    for i in idx:
        try:
            p_t = torch.tensor(params_log10[i:i+1], dtype=torch.float32)
            F = compute_fisher_information(p_t, omega_t, noise_std=sigma)   # (1, 5, 5)
            evals = np.linalg.eigvalsh(F[0].numpy())
            thresh = max(float(evals[-1]) * 1e-2, 1e-8)
            ranks.append(int((evals > thresh).sum()))
        except Exception:
            pass

    ranks = np.array(ranks)
    counts = {str(r): int((ranks == r).sum()) for r in range(1, 6)}
    return {
        "mode": int(np.bincount(ranks).argmax()) if len(ranks) > 0 else -1,
        "mean": float(ranks.mean()) if len(ranks) > 0 else float("nan"),
        "distribution": counts,
        "n": len(ranks),
    }


def cr_bound_distribution(cr_std: np.ndarray, idx: np.ndarray) -> dict:
    """Summarize CR bound distribution over sampled circuits."""
    cr_valid = cr_std[idx]  # (n_sampled, 5)
    result = {}
    for j, p in enumerate(PARAMS_ID):
        vals = cr_valid[:, j]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            result[p] = {}
            continue
        result[p] = {
            "p10":    float(np.percentile(vals, 10)),
            "median": float(np.median(vals)),
            "p90":    float(np.percentile(vals, 90)),
            "p99":    float(np.percentile(vals, 99)),
            "mean":   float(np.mean(vals)),
            "n":      int(len(vals)),
        }
    return result


def cr_by_stratum(params_log10: np.ndarray, cr_std: np.ndarray,
                  sampled_idx: np.ndarray) -> dict:
    """
    CR bounds stratified by identifiability region (easy vs hard).

    Stratification features:
      - tau_ratio: arc separation (low = merged arcs, hard to separate time constants)
      - rsh_ratio: Rsh/TER coupling (near 1 = hardest for Rsh identification)
    """
    strata = identifiability_strata(params_log10)
    result = {}

    for region in ("easy", "hard"):
        mask_region = (strata["region"] == region)
        in_sample = np.zeros(len(params_log10), dtype=bool)
        in_sample[sampled_idx] = True
        mask = mask_region & in_sample & np.all(np.isfinite(cr_std), axis=1)

        if not mask.any():
            result[region] = {}
            continue

        cr_sub = cr_std[mask]
        result[region] = {
            p: {"median": float(np.median(cr_sub[:, j])),
                "p90":    float(np.percentile(cr_sub[:, j], 90))}
            for j, p in enumerate(PARAMS_ID)
        }
        result[region]["n"] = int(mask.sum())

    # Also report fraction of circuits in each region
    result["easy_fraction"] = float((strata["region"] == "easy").mean())
    result["tau_ratio_median"] = float(np.median(strata["tau_ratio"]))
    result["rsh_ratio_median"] = float(np.median(strata["rsh_ratio"]))

    return result


def main():
    ap = argparse.ArgumentParser(description="E1: Identifiability analysis")
    ap.add_argument("--data",    default="data/mixed_distribution_v2")
    ap.add_argument("--n",       type=int, default=1000, help="test set samples to load")
    ap.add_argument("--n-cr",    type=int, default=300,  help="circuits for CR computation")
    ap.add_argument("--n-rank",  type=int, default=200,  help="circuits for rank computation")
    ap.add_argument("--sigma",   type=float, default=10.0, help="noise std (Ω)")
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    print("E1: Identifiability Analysis")
    print(f"  data={args.data}  n={args.n}  sigma={args.sigma}Ω")

    data = load_test_csv(args.data, n_samples=args.n, seed=args.seed)
    omega = data["omega"]
    params_log10 = data["params_raw"]  # (N, 5) log10 raw

    # ── 1. CR bound distribution ──────────────────────────────────────────────
    print(f"\n[1/3] CR bounds on {args.n_cr} circuits ...")
    cr_std, sampled_idx = compute_cr_bounds_subsample(
        params_log10, omega, sigma_noise=args.sigma,
        n_samples=args.n_cr, seed=args.seed,
    )
    cr_dist = cr_bound_distribution(cr_std, sampled_idx)

    # ── 2. Effective Fisher rank ──────────────────────────────────────────────
    print(f"\n[2/3] Fisher rank on {args.n_rank} circuits ...")
    rank_dist = compute_fisher_rank_distribution(params_log10, omega, args.sigma, args.n_rank)

    # ── 3. Stratified CR bounds ───────────────────────────────────────────────
    print("\n[3/3] Stratified analysis ...")
    strat = cr_by_stratum(params_log10, cr_std, sampled_idx)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n── CR Bound Summary (log10 std) ─────────────────────────────────────")
    print(f"  {'param':<12} {'p10':>8} {'median':>8} {'p90':>8} {'p99':>8}")
    for p in PARAMS_ID:
        d = cr_dist.get(p, {})
        print(f"  {p:<12} {d.get('p10', float('nan')):>8.4f} "
              f"{d.get('median', float('nan')):>8.4f} "
              f"{d.get('p90', float('nan')):>8.4f} "
              f"{d.get('p99', float('nan')):>8.4f}")

    print(f"\n── Fisher Rank Distribution ─────────────────────────────────────────")
    print(f"  Mode rank: {rank_dist['mode']}  Mean: {rank_dist['mean']:.2f}")
    print(f"  Distribution: {rank_dist['distribution']}")

    print(f"\n── Identifiability Strata ───────────────────────────────────────────")
    print(f"  Easy region fraction: {strat['easy_fraction']:.1%}")
    for region in ("easy", "hard"):
        if not strat.get(region):
            continue
        print(f"\n  {region.upper()} region (n={strat[region].get('n', '?')}):")
        for p in PARAMS_ID:
            d = strat[region].get(p, {})
            print(f"    {p:<12} median={d.get('median', float('nan')):.4f}  "
                  f"p90={d.get('p90', float('nan')):.4f}")

    out = save_results("E1_identifiability", {
        "config": vars(args),
        "cr_bounds": cr_dist,
        "fisher_rank": rank_dist,
        "by_stratum": strat,
    })
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
