#!/usr/bin/env python3
"""
eval/report.py — Read all E1-E4 result JSONs and emit a paper-ready summary.

Outputs:
  1. ASCII table summary (always)
  2. LaTeX table fragments (--latex flag)

Usage
-----
    python -m eval.report
    python -m eval.report --latex > tables.tex
    python -m eval.report --results results/eval
"""

import argparse
import json
import math
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results" / "eval"

PARAMS_ID  = ["tau_big", "tau_small", "TER", "TEC", "Rsh"]
PARAMS_RAW = ["Ra", "Rb", "Ca", "Cb", "Rsh"]


def load(name: str, results_dir: Path) -> dict:
    path = results_dir / f"{name}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def fmt(v, width: int = 8, decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return " " * (width - 3) + "n/a"
    return f"{v:{width}.{decimals}f}"


def fmt_pct(v, width: int = 7) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return " " * (width - 3) + "n/a"
    return f"{v:{width}.1f}%"


# ── E1: Identifiability ───────────────────────────────────────────────────────

def report_e1(data: dict, latex: bool = False) -> None:
    if not data:
        print("E1: no results found.\n")
        return

    cr = data.get("cr_bounds", {})
    rank = data.get("fisher_rank", {})
    strat = data.get("by_stratum", {})

    print("═" * 60)
    print("E1 — IDENTIFIABILITY ANALYSIS")
    print("═" * 60)
    print(f"\nFisher rank: mode={rank.get('mode', '?')}  mean={rank.get('mean', float('nan')):.2f}"
          f"  (expected mode=3 for RCRC degeneracy)")
    print(f"Distribution: {rank.get('distribution', {})}")

    print(f"\nCR Bound Distribution (log10-std, n={list(cr.values())[0].get('n', '?') if cr else '?'})")
    print(f"  {'param':<12} {'p10':>8} {'median':>8} {'p90':>8} {'p99':>8}")
    for p in PARAMS_ID:
        d = cr.get(p, {})
        print(f"  {p:<12} {fmt(d.get('p10'))}{fmt(d.get('median'))}{fmt(d.get('p90'))}{fmt(d.get('p99'))}")

    if strat:
        easy_f = strat.get("easy_fraction", float("nan"))
        print(f"\nIdentifiability strata: easy={easy_f:.1%}  hard={1-easy_f:.1%}")
        for region in ("easy", "hard"):
            sub = strat.get(region, {})
            if not sub:
                continue
            print(f"\n  {region.upper()} (n={sub.get('n', '?')}):")
            for p in PARAMS_ID:
                d = sub.get(p, {})
                print(f"    {p:<12} median={fmt(d.get('median'))}  p90={fmt(d.get('p90'))}")

    if latex:
        _latex_cr_table(cr)

    print()


def _latex_cr_table(cr: dict) -> None:
    print("\n% --- LaTeX: CR bound table (E1) ---")
    print(r"\begin{tabular}{lrrrr}")
    print(r"  \toprule")
    print(r"  Parameter & $p_{10}$ & Median & $p_{90}$ & $p_{99}$ \\")
    print(r"  \midrule")
    for p in PARAMS_ID:
        d = cr.get(p, {})
        p10  = f"{d.get('p10',  float('nan')):.4f}"
        med  = f"{d.get('median',float('nan')):.4f}"
        p90  = f"{d.get('p90',  float('nan')):.4f}"
        p99  = f"{d.get('p99',  float('nan')):.4f}"
        print(f"  {p} & {p10} & {med} & {p90} & {p99} \\\\")
    print(r"  \bottomrule")
    print(r"\end{tabular}")


# ── E2: Static inference ──────────────────────────────────────────────────────

def report_e2(data: dict, latex: bool = False) -> None:
    if not data:
        print("E2: no results found.\n")
        return

    cr_bounds = data.get("cr_bounds", {})
    mdn_mae   = data.get("mdn_mae", {})
    ecm_mae   = data.get("ecm_mae", {})
    mdn_eff   = data.get("mdn_cr_efficiency", {})
    ecm_eff   = data.get("ecm_cr_efficiency", {})
    mdn_calib = data.get("mdn_calibration", {})
    mdn_strat = data.get("mdn_stratified", {})

    print("═" * 60)
    print("E2 — STATIC INFERENCE (MDN vs ECM vs CR bound)")
    print("═" * 60)

    ecm_rate = data.get("ecm_success_rate")
    print(f"\n  n={data.get('n_total','?')}  ECM success={ecm_rate*100:.1f}%" if ecm_rate is not None
          else f"\n  n={data.get('n_total','?')}")

    print(f"\n  {'param':<12} {'CR-std':>8} {'MDN-MAE':>8} {'ECM-MAE':>8} "
          f"{'MDN-eff':>9} {'ECM-eff':>9}")
    print("  " + "-" * 62)
    for p in PARAMS_ID:
        cr_m  = cr_bounds.get(p, {}).get("median", float("nan"))
        m_mae = mdn_mae.get(p, {}).get("median", float("nan"))
        e_mae = ecm_mae.get(p, {}).get("median", float("nan")) if ecm_mae else float("nan")
        m_eff = mdn_eff.get(p, float("nan"))
        e_eff = ecm_eff.get(p, float("nan")) if ecm_eff else float("nan")
        print(f"  {p:<12}{fmt(cr_m)}{fmt(m_mae)}{fmt(e_mae)}{fmt(m_eff, 9, 3)}{fmt(e_eff, 9, 3)}")

    if mdn_calib:
        print(f"\n  MDN Calibration (target = nominal level):")
        print(f"  {'param':<12} {'68%':>8} {'90%':>8} {'95%':>8}")
        for p in ["TER", "Rsh", "tau_big", "TEC", "tau_small"]:
            c = mdn_calib.get(p, {})
            print(f"  {p:<12}"
                  f"{fmt(c.get('0.68'), 8, 3)}"
                  f"{fmt(c.get('0.90'), 8, 3)}"
                  f"{fmt(c.get('0.95'), 8, 3)}")

    if mdn_strat:
        print(f"\n  Stratified MDN MAE (easy vs hard identifiability region):")
        for region in ("easy", "hard"):
            sub = mdn_strat.get(region, {})
            if not sub:
                continue
            print(f"  {region.upper()} (n={sub.get('n', '?')}):")
            for p in PARAMS_ID:
                print(f"    {p:<12} {fmt(sub.get(p, float('nan')))}")

    if latex:
        _latex_static_table(cr_bounds, mdn_mae, ecm_mae, mdn_eff, ecm_eff)

    print()


def _latex_static_table(cr, mdn_mae, ecm_mae, mdn_eff, ecm_eff) -> None:
    print("\n% --- LaTeX: static inference table (E2) ---")
    print(r"\begin{tabular}{lrrrrr}")
    print(r"  \toprule")
    print(r"  Parameter & CR std & MDN MAE & ECM MAE & MDN eff. & ECM eff. \\")
    print(r"  \midrule")
    for p in PARAMS_ID:
        cr_m  = f"{cr.get(p, {}).get('median', float('nan')):.4f}"
        m_mae = f"{mdn_mae.get(p, {}).get('median', float('nan')):.4f}"
        e_mae = f"{ecm_mae.get(p, {}).get('median', float('nan')):.4f}" if ecm_mae else "---"
        m_eff = f"{mdn_eff.get(p, float('nan')):.3f}"
        e_eff = f"{ecm_eff.get(p, float('nan')):.3f}" if ecm_eff else "---"
        print(f"  {p} & {cr_m} & {m_mae} & {e_mae} & {m_eff} & {e_eff} \\\\")
    print(r"  \bottomrule")
    print(r"\end{tabular}")


# ── E3: Temporal inference ────────────────────────────────────────────────────

def report_e3(data: dict, latex: bool = False) -> None:
    if not data:
        print("E3: no results found.\n")
        return

    by_path = data.get("by_pathology", {})

    print("═" * 60)
    print("E3 — TEMPORAL INFERENCE (GPF vs FFBS vs ECM-per-step)")
    print("═" * 60)

    for pathology, res in by_path.items():
        gpf_f  = res.get("gpf_final", {})
        ffbs_f = res.get("ffbs_final", {})
        ecm_f  = res.get("ecm_final", {})
        gpf_conv = res.get("convergence_step_gpf", {})

        print(f"\n  {pathology.upper()} (n={res.get('n_traj','?')}):")
        print(f"  {'param':<12} {'ECM':>8} {'GPF':>8} {'FFBS':>8} {'conv-t':>8}")
        for p in PARAMS_ID:
            print(f"  {p:<12}"
                  f"{fmt(ecm_f.get(p))}"
                  f"{fmt(gpf_f.get(p))}"
                  f"{fmt(ffbs_f.get(p))}"
                  f"  {gpf_conv.get(p, -1):>6}")

    if latex:
        _latex_temporal_table(by_path)

    print()


def _latex_temporal_table(by_path: dict) -> None:
    pathologies = list(by_path.keys())[:4]   # cap at 4 for space
    print("\n% --- LaTeX: temporal inference final-step MAE (E3) ---")
    header = " & ".join(f"\\multicolumn{{2}}{{c}}{{{p}}}" for p in pathologies[:2])
    print(r"\begin{tabular}{l" + "rr" * len(pathologies[:2]) + "}")
    print(r"  \toprule")
    print(f"  Param & {header} \\\\")
    print(r"  \cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    print(r"  & GPF & FFBS" * len(pathologies[:2]) + r" \\")
    print(r"  \midrule")
    for p in PARAMS_ID:
        row = [p]
        for pat in pathologies[:2]:
            res  = by_path.get(pat, {})
            row.append(f"{res.get('gpf_final', {}).get(p, float('nan')):.4f}")
            row.append(f"{res.get('ffbs_final', {}).get(p, float('nan')):.4f}")
        print("  " + " & ".join(row) + r" \\")
    print(r"  \bottomrule")
    print(r"\end{tabular}")


# ── E4: Degeneracy ────────────────────────────────────────────────────────────

def report_e4(data: dict, latex: bool = False) -> None:
    if not data:
        print("E4: no results found.\n")
        return

    part_a = data.get("part_a_symmetric_gain", {})
    part_b = data.get("part_b_beta_tracking",  {})
    part_c = data.get("part_c_fork_detection", {})

    print("═" * 60)
    print("E4 — DEGENERACY: SYMMETRIC METRICS / β-TRACKING / FORK DETECTION")
    print("═" * 60)

    if part_a:
        print(f"\n  Part A — Symmetric vs Standard MAE (ECM raw params):")
        print(f"  {'param':<6} {'standard':>10} {'symmetric':>10} {'gain':>8}")
        for p in PARAMS_RAW:
            d = part_a.get(p, {})
            print(f"  {p:<6}{fmt(d.get('standard'), 10, 4)}"
                  f"{fmt(d.get('symmetric'), 10, 4)}"
                  f"{fmt_pct(d.get('gain_pct'), 8)}")

    if part_b:
        print(f"\n  Part B — β = Ra/(Ra+Rb) tracking RMSE:")
        print(f"  {'pathology':<22} {'GPF-RMSE':>10} {'baseline':>10} {'improvement':>12}")
        for pat, res in part_b.items():
            print(f"  {pat:<22}"
                  f"{fmt(res.get('rmse_median'), 10, 4)}"
                  f"{fmt(res.get('baseline_rmse'), 10, 4)}"
                  f"{fmt_pct(res.get('improvement_pct'), 12)}")

    if part_c:
        print(f"\n  Part C — PTG fork detection (synthetic trajectories):")
        print(f"  {'mechanism':<16} {'PTG-rate':>10} {'PTG-err':>8} {'Var-rate':>10} {'Var-err':>8}")
        for mech, res in part_c.items():
            print(f"  {mech:<16}"
                  f"{fmt_pct(res.get('ptg_detect_rate', float('nan'))*100, 10)}"
                  f"{fmt(res.get('ptg_timing_error_median'), 8, 1)}"
                  f"{fmt_pct(res.get('var_detect_rate', float('nan'))*100, 10)}"
                  f"{fmt(res.get('var_timing_error_median'), 8, 1)}")

    print()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Report results from E1-E4 evaluation")
    ap.add_argument("--results", default=str(RESULTS_DIR))
    ap.add_argument("--latex",   action="store_true", help="Also emit LaTeX table fragments")
    ap.add_argument("--only",    nargs="+", default=None, choices=["E1", "E2", "E3", "E4"])
    args = ap.parse_args()

    results_dir = Path(args.results)
    to_report = args.only or ["E1", "E2", "E3", "E4"]

    print(f"\nResults directory: {results_dir}")
    print()

    if "E1" in to_report:
        report_e1(load("E1_identifiability", results_dir), args.latex)
    if "E2" in to_report:
        report_e2(load("E2_static", results_dir), args.latex)
    if "E3" in to_report:
        report_e3(load("E3_temporal", results_dir), args.latex)
    if "E4" in to_report:
        report_e4(load("E4_degeneracy", results_dir), args.latex)


if __name__ == "__main__":
    main()
