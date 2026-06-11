#!/usr/bin/env python3
"""
E2 — Static Inference Evaluation

Answers: how well do static methods recover identifiable parameters from
a single spectrum? How close are we to the Cramér-Rao lower bound?

Methods compared
----------------
ecm      Classical ECM: L-BFGS-B with random restarts. No uncertainty output.
mdn      FisherAwareTransformer: MDN over identifiable space. Outputs K-component
         mixture with means and covariances (uncertainty estimate).

Metrics
-------
mae_id      Median |pred - truth| in log10 identifiable space [tau_big, tau_small, TER, TEC, Rsh]
cr_eff      CR efficiency = CR_bound_std / method_MAE_median
            > 1 super-efficient (artifact), ~ 1 near-optimal, << 1 wasted information
calib       Calibration: coverage of |error| < z·σ_pred at 68% and 90% levels
            Only for MDN (ECM produces no uncertainty).
strat       Repeat MAE broken down by easy/hard identifiability region.

Paper claim supported
---------------------
"The FisherAwareTransformer operating in identifiable coordinates approaches the
CR lower bound on TER and TEC, and substantially reduces error on tau_big and
tau_small relative to ECM. Rsh remains the hardest parameter due to spectral
insensitivity in the healthy regime."

Output
------
results/eval/E2_static.json

Usage
-----
    python -m eval.E2_static
    python -m eval.E2_static --n 1000 --no-ecm --model models/fisher_v10
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.shared import (
    PARAMS_ID, load_test_csv, load_transformer,
    run_mdn_batch, run_ecm_batch,
    mae_stats, cr_efficiency, calibration_coverage,
    compute_cr_bounds_subsample, identifiability_strata, save_results,
)


def efficiency_table(method_mae: dict, cr_dist: dict) -> dict:
    """Per-parameter CR efficiency for a given method's MAE."""
    return {
        p: cr_efficiency(cr_dist.get(p, {}).get("median", float("nan")),
                         method_mae.get(p, {}).get("median", float("nan")))
        for p in PARAMS_ID
    }


def stratified_mae(pred: np.ndarray, truth: np.ndarray,
                   params_log10: np.ndarray) -> dict:
    """MAE broken down by identifiable region (easy vs hard)."""
    strata = identifiability_strata(params_log10)
    result = {}
    for region in ("easy", "hard"):
        mask = (strata["region"] == region) & np.all(np.isfinite(pred), axis=1)
        if not mask.any():
            result[region] = {}
            continue
        stats = mae_stats(pred[mask], truth[mask])
        result[region] = {p: stats[p].get("median", float("nan")) for p in PARAMS_ID}
        result[region]["n"] = int(mask.sum())
    return result


def main():
    ap = argparse.ArgumentParser(description="E2: Static inference evaluation")
    ap.add_argument("--model",   default="models/fisher_v10")
    ap.add_argument("--data",    default="data/mixed_distribution_v2")
    ap.add_argument("--n",       type=int, default=1000, help="total test samples")
    ap.add_argument("--n-ecm",   type=int, default=500,  help="ECM subsample (slower)")
    ap.add_argument("--n-cr",    type=int, default=300,  help="CR subsample")
    ap.add_argument("--sigma",   type=float, default=10.0)
    ap.add_argument("--no-ecm",  action="store_true")
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    print("E2: Static Inference Evaluation")
    print(f"  model={args.model}  data={args.data}  n={args.n}")

    data  = load_test_csv(args.data, n_samples=args.n, seed=args.seed)
    model, is_id, ckpt = load_transformer(args.model)
    omega = data["omega"]
    truth = data["params_id"]   # (N, 5) log10 identifiable

    # ── MDN inference ─────────────────────────────────────────────────────────
    print(f"\n[1/3] MDN inference (n={data['n']}) ...")
    mdn_pred, mdn_std = run_mdn_batch(model, data["Zr"], data["Zi"], omega)
    mdn_mae  = mae_stats(mdn_pred, truth)

    # ── ECM baseline ──────────────────────────────────────────────────────────
    ecm_pred = ecm_success = None
    ecm_mae  = {}
    if not args.no_ecm:
        # subsample for ECM (it's slow)
        rng  = np.random.default_rng(args.seed)
        ecm_idx = rng.choice(data["n"], min(args.n_ecm, data["n"]), replace=False)
        print(f"\n[2/3] ECM baseline (n={len(ecm_idx)}, {args.n_ecm} max) ...")
        ecm_pred_sub, ecm_succ_sub = run_ecm_batch(
            data["Zr"][ecm_idx], data["Zi"][ecm_idx], omega,
        )
        ecm_pred = np.full((data["n"], 5), np.nan)
        ecm_pred[ecm_idx] = ecm_pred_sub
        ecm_success = np.zeros(data["n"], dtype=bool)
        ecm_success[ecm_idx] = ecm_succ_sub
        ecm_mae = mae_stats(ecm_pred, truth)
        print(f"  ECM success rate: {ecm_succ_sub.mean()*100:.1f}%")
    else:
        print("\n[2/3] ECM skipped (--no-ecm)")

    # ── CR bounds ─────────────────────────────────────────────────────────────
    print(f"\n[3/3] CR bounds (n={args.n_cr}) ...")
    cr_std, cr_idx = compute_cr_bounds_subsample(
        data["params_raw"], omega, sigma_noise=args.sigma,
        n_samples=args.n_cr, seed=args.seed,
    )
    cr_dist = {}
    for j, p in enumerate(PARAMS_ID):
        vals = cr_std[cr_idx, j]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            cr_dist[p] = {"median": float(np.median(vals)), "p90": float(np.percentile(vals, 90))}

    # ── Calibration (MDN only) ────────────────────────────────────────────────
    mdn_calib = {}
    for j, p in enumerate(PARAMS_ID):
        errors = np.abs(mdn_pred[:, j] - truth[:, j])
        pred_s = mdn_std[:, j]
        valid  = np.isfinite(errors) & np.isfinite(pred_s) & (pred_s > 0)
        if valid.any():
            mdn_calib[p] = calibration_coverage(pred_s[valid], errors[valid])

    # ── Stratified MAE ────────────────────────────────────────────────────────
    mdn_strat = stratified_mae(mdn_pred, truth, data["params_raw"])
    ecm_strat = stratified_mae(ecm_pred, truth, data["params_raw"]) if ecm_pred is not None else {}

    # ── Efficiencies ──────────────────────────────────────────────────────────
    mdn_eff = efficiency_table(mdn_mae, cr_dist)
    ecm_eff = efficiency_table(ecm_mae, cr_dist) if ecm_mae else {}

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n── Results (log10 median MAE) ───────────────────────────────────────")
    print(f"  {'param':<12} {'CR-std':>8} {'MDN':>8} {'ECM':>8} {'MDN-eff':>9} {'ECM-eff':>9}")
    for p in PARAMS_ID:
        cr_m  = cr_dist.get(p, {}).get("median", float("nan"))
        m_mae = mdn_mae.get(p, {}).get("median", float("nan"))
        e_mae = ecm_mae.get(p, {}).get("median", float("nan"))
        m_eff = mdn_eff.get(p, float("nan"))
        e_eff = ecm_eff.get(p, float("nan"))
        print(f"  {p:<12} {cr_m:>8.4f} {m_mae:>8.4f} {e_mae:>8.4f} {m_eff:>9.3f} {e_eff:>9.3f}")

    print("\n── MDN Calibration (target = nominal level) ─────────────────────────")
    for p in ["TER", "Rsh", "tau_big"]:
        c = mdn_calib.get(p, {})
        print(f"  {p:<12}  68%→{c.get('0.68', float('nan')):.2f}  "
              f"90%→{c.get('0.90', float('nan')):.2f}  "
              f"95%→{c.get('0.95', float('nan')):.2f}")

    out = save_results("E2_static", {
        "config": vars(args),
        "ckpt_val_mae": ckpt.get("val_mae"),
        "n_total": data["n"],
        "ecm_success_rate": float(ecm_success.mean()) if ecm_success is not None else None,
        "cr_bounds": cr_dist,
        "mdn_mae": mdn_mae,
        "mdn_cr_efficiency": mdn_eff,
        "mdn_calibration": mdn_calib,
        "mdn_stratified": mdn_strat,
        "ecm_mae": ecm_mae,
        "ecm_cr_efficiency": ecm_eff,
        "ecm_stratified": ecm_strat,
    })
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
