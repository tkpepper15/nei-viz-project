#!/usr/bin/env python3
"""
EIS Degeneracy Experiments
Tests various interventions to resolve Ra/Rb and Ca/Cb degeneracy in SMC-based EIS fitting.
"""

import json
import requests
import sys
import time
import numpy as np
from typing import Optional

API_BASE = "http://localhost:5003"

def parse_sse_stream(response):
    """Parse Server-Sent Events stream, return list of parsed JSON events."""
    events = []
    buffer = ""
    for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
        buffer += chunk
        while "\n\n" in buffer:
            event_str, buffer = buffer.split("\n\n", 1)
            lines = event_str.strip().split("\n")
            for line in lines:
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        event = json.loads(data_str)
                        events.append(event)
                    except json.JSONDecodeError:
                        pass
    return events


def get_sample_data():
    """Load sample data from API."""
    resp = requests.get(f"{API_BASE}/real_sample_data", timeout=30)
    resp.raise_for_status()
    return resp.json()


def run_smc_stream(sequences, n_samples=500, n_display=15, include_ecm=True,
                   use_sequential=True, canonical_mode=None, ground_truth_seed=None,
                   timeout=120):
    """Run SMC temporal analysis stream, return all events."""
    payload = {
        "sequences": sequences,
        "n_samples": n_samples,
        "n_display": n_display,
        "include_ecm": include_ecm,
        "use_sequential": use_sequential,
        "canonical_mode": canonical_mode,
    }
    if ground_truth_seed is not None:
        payload["ground_truth_seed"] = ground_truth_seed

    resp = requests.post(
        f"{API_BASE}/mc_temporal_analysis_stream",
        json=payload,
        stream=True,
        timeout=timeout
    )
    resp.raise_for_status()
    return parse_sse_stream(resp)


def extract_final_metrics(events):
    """Extract metrics from the last non-done event."""
    last_event = None
    done_event = None
    for e in events:
        if e.get("done"):
            done_event = e
        else:
            last_event = e
    return last_event, done_event


def predict_single(Z_real, Z_imag, frequencies):
    """Run single-spectrum inference."""
    resp = requests.post(
        f"{API_BASE}/predict_single",
        json={"Z_real": Z_real, "Z_imag": Z_imag, "frequencies": frequencies},
        timeout=60
    )
    resp.raise_for_status()
    return resp.json()


def explore_constraints(Z_real, Z_imag, frequencies, n_samples=1000, constraints=None):
    """Run constraint-based uncertainty reduction."""
    payload = {
        "Z_real": Z_real,
        "Z_imag": Z_imag,
        "frequencies": frequencies,
        "n_samples": n_samples,
        "constraints": constraints or {}
    }
    resp = requests.post(f"{API_BASE}/explore_constraints", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def summarize_event(event, label=""):
    """Print key metrics from an SMC event."""
    if event is None:
        print(f"  {label}: No event data")
        return
    smc = event.get("smc_diag", {})
    print(f"  {label}:")
    def fmt(v, fmt_spec=".4f"):
        return f"{v:{fmt_spec}}" if isinstance(v, (int, float)) else str(v)
    print(f"    deg_entropy_R={fmt(event.get('deg_entropy_R'))}  deg_entropy_C={fmt(event.get('deg_entropy_C'))}")
    print(f"    p_Ra_lt_Rb={fmt(smc.get('p_Ra_lt_Rb'), '.3f')}  p_Ca_gt_Cb={fmt(smc.get('p_Ca_gt_Cb'), '.3f')}")
    print(f"    ess_frac={fmt(smc.get('ess_frac'), '.3f')}")
    hw = smc.get("hyp_weights", [])
    if hw:
        print(f"    hyp_weights={[f'{w:.3f}' for w in hw]}")
    dl_r = event.get('dl_resnorm', None)
    ecm_r = event.get('ecm_resnorm', None)
    dl_str = f"{dl_r:.4f}" if isinstance(dl_r, (int, float)) else str(dl_r)
    ecm_str = f"{ecm_r:.4f}" if isinstance(ecm_r, (int, float)) else str(ecm_r)
    print(f"    dl_resnorm={dl_str}  ecm_resnorm={ecm_str}")


# ============================================================
# Main Experiment Runner
# ============================================================

def main():
    print("=" * 70)
    print("EIS DEGENERACY EXPERIMENTS")
    print("=" * 70)

    # Load data
    print("\nLoading sample data...")
    data = get_sample_data()
    sample0 = data["samples"][0]
    timepoints = sample0["timepoints"]
    print(f"Sample: {sample0['name']}, {len(timepoints)} timepoints")

    # Use first 8 timepoints for speed
    N_TP = 8
    tps = timepoints[:N_TP]
    sequences = [{"Z_real": tp["Z_real"], "Z_imag": tp["Z_imag"],
                  "frequencies": tp["frequencies"], "time_min": tp["time_min"]}
                 for tp in tps]

    first_tp = tps[0]
    gt = first_tp["ground_truth"]
    print(f"Ground truth: Ra={gt['Ra']:.1f} Rb={gt['Rb']:.1f} Rsh={gt['Rsh']:.1f} "
          f"Ca={gt['Ca']:.2e} Cb={gt['Cb']:.2e}")
    print(f"  TER={gt['TER']:.3f} kOhm  (in display units)")

    # ============================================================
    # Experiment 1: Baseline degeneracy measurement
    # ============================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Baseline Degeneracy (all defaults)")
    print("=" * 70)
    t0 = time.time()
    events1 = run_smc_stream(sequences, n_samples=500, use_sequential=True,
                              canonical_mode=None, ground_truth_seed=None)
    elapsed = time.time() - t0
    print(f"  Stream received {len(events1)} events in {elapsed:.1f}s")

    first_evt1 = events1[0] if events1 else None
    last_evt1, done_evt1 = extract_final_metrics(events1)

    print("\n  FIRST timepoint:")
    summarize_event(first_evt1, "t=0")
    print("\n  LAST timepoint:")
    summarize_event(last_evt1, f"t={N_TP-1}")

    # ============================================================
    # Experiment 2: GT seed vs no GT seed
    # ============================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: GT Seed vs No GT Seed")
    print("=" * 70)

    # GT seed (SI units: Ohms, Farads)
    gt_seed = {
        "Ra": gt["Ra"],
        "Rb": gt["Rb"],
        "Ca": gt["Ca"],
        "Cb": gt["Cb"],
        "Rsh": gt["Rsh"],
    }
    print(f"  GT seed: {gt_seed}")

    t0 = time.time()
    events2_gt = run_smc_stream(sequences[:5], n_samples=400, use_sequential=True,
                                 canonical_mode=None, ground_truth_seed=gt_seed)
    elapsed = time.time() - t0
    print(f"  GT-seeded stream: {len(events2_gt)} events in {elapsed:.1f}s")

    t0 = time.time()
    events2_no = run_smc_stream(sequences[:5], n_samples=400, use_sequential=True,
                                 canonical_mode=None, ground_truth_seed=None)
    elapsed = time.time() - t0
    print(f"  No-seed stream: {len(events2_no)} events in {elapsed:.1f}s")

    last_gt, _ = extract_final_metrics(events2_gt)
    last_no, _ = extract_final_metrics(events2_no)

    print("\n  With GT seed:")
    summarize_event(last_gt, "GT seeded final")
    print("\n  Without GT seed:")
    summarize_event(last_no, "No seed final")

    if last_gt and last_no:
        dr_gt = last_gt.get("deg_entropy_R", float("nan"))
        dr_no = last_no.get("deg_entropy_R", float("nan"))
        dc_gt = last_gt.get("deg_entropy_C", float("nan"))
        dc_no = last_no.get("deg_entropy_C", float("nan"))
        print(f"\n  DELTA deg_entropy_R (no_seed - gt_seed): {dr_no - dr_gt:.4f}")
        print(f"  DELTA deg_entropy_C (no_seed - gt_seed): {dc_no - dc_gt:.4f}")
        if abs(dr_no - dr_gt) < 0.05:
            print("  CONCLUSION: GT seeding has NEGLIGIBLE effect — degeneracy is FUNDAMENTAL")
        else:
            print("  CONCLUSION: GT seeding HELPS — degeneracy is partly a cold-start problem")

    # ============================================================
    # Experiment 3: Constraint-based reduction
    # ============================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Constraint-Based Uncertainty Reduction")
    print("=" * 70)

    Z_real = first_tp["Z_real"]
    Z_imag = first_tp["Z_imag"]
    frequencies = first_tp["frequencies"]

    constraint_configs = [
        ("No constraints", {}),
        ("TER [0.3, 0.8] kOhm", {"TER_min": 0.3, "TER_max": 0.8}),
        ("Ra_Rb_ratio_max=3", {"Ra_Rb_ratio_max": 3.0}),
        ("TER + Ra_Rb_ratio", {"TER_min": 0.3, "TER_max": 0.8, "Ra_Rb_ratio_max": 3.0}),
        ("Tight TER [0.35, 0.5]", {"TER_min": 0.35, "TER_max": 0.5}),
        ("Ca_Cb_ratio_max=3", {"Ca_Cb_ratio_max": 3.0}),
        ("All constraints", {"TER_min": 0.3, "TER_max": 0.8, "Ra_Rb_ratio_max": 3.0,
                             "Ca_Cb_ratio_max": 3.0}),
    ]

    exp3_results = []
    for label, constraints in constraint_configs:
        try:
            result = explore_constraints(Z_real, Z_imag, frequencies,
                                         n_samples=1000, constraints=constraints)
            exp3_results.append((label, constraints, result))
            params = result.get("predictions", result.get("params", {}))
            print(f"\n  [{label}]")
            for pname, pdata in params.items():
                mean = pdata.get("mean", 0)
                std = pdata.get("std", 0)
                cv = abs(std / mean) if mean != 0 else float("inf")
                print(f"    {pname}: mean={mean:.4f}  std={std:.4f}  CV={cv:.3f}")
            n_accepted = result.get("n_accepted", result.get("n_samples", "?"))
            print(f"    n_accepted={n_accepted}")
        except Exception as ex:
            print(f"  [{label}] ERROR: {ex}")

    # ============================================================
    # Experiment 4: Frequency subset experiments
    # ============================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Frequency Subset Experiments")
    print("=" * 70)

    freqs = np.array(frequencies)
    Z_real_arr = np.array(Z_real)
    Z_imag_arr = np.array(Z_imag)

    freq_configs = [
        ("All frequencies", freqs < 1e10),
        ("Low freq (<100 Hz)", freqs < 100),
        ("High freq (>1000 Hz)", freqs > 1000),
        ("Mid-band (100-1000 Hz)", (freqs >= 100) & (freqs <= 1000)),
    ]

    exp4_results = []
    for label, mask in freq_configs:
        idx = np.where(mask)[0]
        if len(idx) < 5:
            print(f"  [{label}] SKIP — only {len(idx)} frequencies")
            continue
        try:
            result = predict_single(
                Z_real_arr[idx].tolist(),
                Z_imag_arr[idx].tolist(),
                freqs[idx].tolist()
            )
            exp4_results.append((label, result))
            print(f"\n  [{label}] — {len(idx)} frequencies ({freqs[idx].min():.1f}-{freqs[idx].max():.1f} Hz)")
            preds = result.get("predictions", result)
            for pname, pdata in preds.items():
                if isinstance(pdata, dict):
                    mean = pdata.get("mean", 0)
                    std = pdata.get("std", pdata.get("uncertainty", 0))
                    cv = abs(std / mean) if mean != 0 else float("inf")
                    print(f"    {pname}: mean={mean:.4f}  std={std:.4f}  CV={cv:.3f}")
        except Exception as ex:
            print(f"  [{label}] ERROR: {ex}")

    # ============================================================
    # Experiment 5: Sequential vs non-sequential
    # ============================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Sequential vs Non-Sequential")
    print("=" * 70)

    t0 = time.time()
    events5_seq = run_smc_stream(sequences[:5], n_samples=400, use_sequential=True)
    print(f"  Sequential: {len(events5_seq)} events in {time.time()-t0:.1f}s")

    t0 = time.time()
    events5_nseq = run_smc_stream(sequences[:5], n_samples=400, use_sequential=False)
    print(f"  Non-sequential: {len(events5_nseq)} events in {time.time()-t0:.1f}s")

    last_seq, _ = extract_final_metrics(events5_seq)
    last_nseq, _ = extract_final_metrics(events5_nseq)

    print("\n  Sequential final:")
    summarize_event(last_seq, "sequential")
    print("\n  Non-sequential final:")
    summarize_event(last_nseq, "non-sequential")

    if last_seq and last_nseq:
        dr_seq = last_seq.get("deg_entropy_R", float("nan"))
        dr_nseq = last_nseq.get("deg_entropy_R", float("nan"))
        dc_seq = last_seq.get("deg_entropy_C", float("nan"))
        dc_nseq = last_nseq.get("deg_entropy_C", float("nan"))
        print(f"\n  DELTA deg_entropy_R (seq - nseq): {dr_seq - dr_nseq:.4f}")
        print(f"  DELTA deg_entropy_C (seq - nseq): {dc_seq - dc_nseq:.4f}")

    # ============================================================
    # Experiment 6: Canonical mode impact
    # ============================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Canonical Mode (Ra_gt_Rb) vs None")
    print("=" * 70)

    t0 = time.time()
    events6_canon = run_smc_stream(sequences[:5], n_samples=400,
                                    canonical_mode="Ra_gt_Rb")
    print(f"  Canonical mode: {len(events6_canon)} events in {time.time()-t0:.1f}s")

    t0 = time.time()
    events6_none = run_smc_stream(sequences[:5], n_samples=400,
                                   canonical_mode=None)
    print(f"  No canonical mode: {len(events6_none)} events in {time.time()-t0:.1f}s")

    last_canon, _ = extract_final_metrics(events6_canon)
    last_none6, _ = extract_final_metrics(events6_none)

    print("\n  With Ra_gt_Rb canonical mode:")
    summarize_event(last_canon, "canonical")
    print("\n  No canonical mode:")
    summarize_event(last_none6, "no canonical")

    if last_canon and last_none6:
        dr_c = last_canon.get("deg_entropy_R", float("nan"))
        dr_n = last_none6.get("deg_entropy_R", float("nan"))
        dc_c = last_canon.get("deg_entropy_C", float("nan"))
        dc_n = last_none6.get("deg_entropy_C", float("nan"))
        print(f"\n  DELTA deg_entropy_R (canonical - none): {dr_c - dr_n:.4f}")
        print(f"  DELTA deg_entropy_C (canonical - none): {dc_c - dc_n:.4f}")
        print(f"  p_Ra_lt_Rb canonical: {last_canon.get('smc_diag',{}).get('p_Ra_lt_Rb','N/A')}")
        print(f"  p_Ra_lt_Rb no-canon:  {last_none6.get('smc_diag',{}).get('p_Ra_lt_Rb','N/A')}")

    # ============================================================
    # Final Summary / Rankings
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY: INTERVENTION EFFECTIVENESS RANKING")
    print("=" * 70)

    # Collect baseline entropy
    baseline_R = last_evt1.get("deg_entropy_R", float("nan")) if last_evt1 else float("nan")
    baseline_C = last_evt1.get("deg_entropy_C", float("nan")) if last_evt1 else float("nan")
    print(f"\nBaseline (Exp 1) deg_entropy_R={baseline_R:.4f}  deg_entropy_C={baseline_C:.4f}")

    rankings = []

    # GT seed
    if last_gt:
        dr = baseline_R - last_gt.get("deg_entropy_R", baseline_R)
        dc = baseline_C - last_gt.get("deg_entropy_C", baseline_C)
        rankings.append(("GT seeding", dr, dc))

    # Sequential
    if last_seq and last_nseq:
        dr = last_nseq.get("deg_entropy_R", baseline_R) - last_seq.get("deg_entropy_R", baseline_R)
        dc = last_nseq.get("deg_entropy_C", baseline_C) - last_seq.get("deg_entropy_C", baseline_C)
        rankings.append(("Sequential tracking", dr, dc))

    # Canonical
    if last_canon and last_none6:
        dr = last_none6.get("deg_entropy_R", baseline_R) - last_canon.get("deg_entropy_R", baseline_R)
        dc = last_none6.get("deg_entropy_C", baseline_C) - last_canon.get("deg_entropy_C", baseline_C)
        rankings.append(("Canonical mode (Ra>Rb)", dr, dc))

    rankings.sort(key=lambda x: -(x[1] + x[2]))
    print("\nRankings (delta entropy reduced = more effective):")
    for rank, (name, dr, dc) in enumerate(rankings, 1):
        print(f"  {rank}. {name}: ΔR={dr:+.4f}  ΔC={dc:+.4f}  total={dr+dc:+.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
