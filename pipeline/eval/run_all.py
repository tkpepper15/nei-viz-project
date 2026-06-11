#!/usr/bin/env python3
"""
eval/run_all.py — Orchestrator for the full evaluation suite.

Runs E1-E4 as subprocesses in dependency order, collects exit codes,
and reports timing. All results land in results/eval/*.json.

Usage
-----
    python -m eval.run_all                          # full suite
    python -m eval.run_all --only E1 E2             # subset
    python -m eval.run_all --no-ecm --n 200         # fast development pass
    python -m eval.run_all --parallel               # E1+E2 in parallel (E3/E4 after)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent

EXPERIMENTS = {
    "E1": {
        "module": "eval.E1_identifiability",
        "desc":   "Identifiability analysis (CR bounds, Fisher rank)",
        "flags":  [],
    },
    "E2": {
        "module": "eval.E2_static",
        "desc":   "Static inference: MDN vs ECM vs CR bound",
        "flags":  [],
    },
    "E3": {
        "module": "eval.E3_temporal",
        "desc":   "Temporal inference: GPF vs FFBS vs ECM-per-step",
        "flags":  [],
    },
    "E4": {
        "module": "eval.E4_degeneracy",
        "desc":   "Degeneracy: symmetric metrics, β-tracking, PTG forks",
        "flags":  [],
    },
}

# Flags forwarded from run_all to specific experiments.
FORWARDED_FLAGS = {
    "--no-ecm":  ["E2", "E3", "E4"],
    "--no-ptg":  ["E4"],
    "--n":       ["E1", "E2"],
    "--n-traj":  ["E3", "E4"],
    "--model":   ["E2", "E3", "E4"],
    "--data":    ["E1", "E2"],
    "--sigma":   ["E1", "E2", "E3"],
    "--seed":    ["E1", "E2", "E3", "E4"],
}


def run_experiment(name: str, extra_flags: list[str]) -> tuple[int, float]:
    """Run a single experiment module, return (exit_code, elapsed_seconds)."""
    exp = EXPERIMENTS[name]
    cmd = [sys.executable, "-m", exp["module"]] + exp["flags"] + extra_flags
    print(f"\n{'='*60}")
    print(f"  {name}: {exp['desc']}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n  {name} {status}  [{elapsed:.1f}s]")
    return result.returncode, elapsed


def run_parallel(names: list[str], extra_flags_map: dict[str, list[str]]) -> dict[str, tuple[int, float]]:
    """Run multiple experiments in parallel via subprocess."""
    import concurrent.futures

    outcomes = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(names)) as pool:
        futures = {
            pool.submit(run_experiment, name, extra_flags_map[name]): name
            for name in names
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            outcomes[name] = future.result()
    return outcomes


def main():
    ap = argparse.ArgumentParser(description="Run full evaluation suite E1-E4")
    ap.add_argument("--only",     nargs="+", default=None, choices=list(EXPERIMENTS.keys()),
                    help="Run only these experiments")
    ap.add_argument("--parallel", action="store_true",
                    help="Run E1 and E2 in parallel (independent); E3/E4 after")
    ap.add_argument("--no-ecm",   action="store_true")
    ap.add_argument("--no-ptg",   action="store_true")
    ap.add_argument("--model",    default="models/fisher_v10")
    ap.add_argument("--data",     default="data/mixed_distribution_v2")
    ap.add_argument("--n",        type=int, default=None)
    ap.add_argument("--n-traj",   type=int, default=None)
    ap.add_argument("--sigma",    type=float, default=None)
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()

    to_run = args.only or list(EXPERIMENTS.keys())

    # Build per-experiment flag lists from forwarded flags
    extra_flags_map: dict[str, list[str]] = {name: [] for name in to_run}
    for flag, targets in FORWARDED_FLAGS.items():
        flag_attr = flag.lstrip("-").replace("-", "_")
        val = getattr(args, flag_attr, None)
        if val is None:
            continue
        for name in targets:
            if name not in extra_flags_map:
                continue
            if isinstance(val, bool):
                if val:
                    extra_flags_map[name].append(flag)
            else:
                extra_flags_map[name] += [flag, str(val)]

    print(f"\nEvaluation suite: {to_run}")
    suite_start = time.time()
    outcomes: dict[str, tuple[int, float]] = {}

    if args.parallel and "E1" in to_run and "E2" in to_run:
        # E1 and E2 are independent — run in parallel
        parallel_batch = [n for n in ["E1", "E2"] if n in to_run]
        sequential_batch = [n for n in to_run if n not in parallel_batch]
        print(f"\nParallel batch: {parallel_batch}")
        outcomes.update(run_parallel(parallel_batch, extra_flags_map))
        for name in sequential_batch:
            code, elapsed = run_experiment(name, extra_flags_map[name])
            outcomes[name] = (code, elapsed)
    else:
        for name in to_run:
            code, elapsed = run_experiment(name, extra_flags_map[name])
            outcomes[name] = (code, elapsed)

    suite_elapsed = time.time() - suite_start

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  EVALUATION SUITE COMPLETE  [{suite_elapsed:.1f}s total]")
    print(f"{'='*60}")
    all_ok = True
    for name in to_run:
        code, elapsed = outcomes.get(name, (-1, 0.0))
        status = "PASS" if code == 0 else "FAIL"
        print(f"  {name:<4} {status}  [{elapsed:.1f}s]")
        if code != 0:
            all_ok = False

    result_dir = ROOT / "results" / "eval"
    if result_dir.exists():
        jsons = sorted(result_dir.glob("*.json"))
        print(f"\n  Results ({len(jsons)} files in {result_dir}):")
        for j in jsons:
            print(f"    {j.name}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
