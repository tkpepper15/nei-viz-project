#!/usr/bin/env python3
"""
Generate training data sampled uniformly in identifiable parameter space.

Problem with the existing dataset (physics_constrained_corrected):
  Sampling [Ra, Rb, Ca, Cb, Rsh] log-uniformly and rejecting tau_ratio < 2
  produces a heavily right-skewed tau_ratio distribution (median ~22x, p90 ~700x).
  The model learns this prior and over-predicts tau separation, causing high
  tau_big error on circuits with ratio 2-5x.

Fix:
  Sample [tau_big, tau_small, TER, TEC, Rsh] directly in identifiable space,
  with log10(tau_ratio) uniform in [log10(1.5), log10(10000)].
  Back-solve for [Ra, Rb, Ca, Cb] using the algebraic inversion.
  Mode assignment (which branch carries tau_big) is random 50/50.

This gives equal model exposure to every decade of tau separation,
which is what the loss landscape needs to avoid the tau_big bias.

Output format matches mixed_distribution_v2: one row per circuit,
Z_real_0..99 / Z_imag_0..99 columns.
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.data.randles_circuit_simulator import RandlesCircuitSimulator


# ── algebraic inversion: identifiable → degenerate ────────────────────────

def _invert(tau_big, tau_small, TER, TEC, Rsh, big_tau_is_Ra):
    """
    Return (Ra, Rb, Ca, Cb) or None if the configuration is singular.

    Derivation:
        S_R = Ra+Rb = TER*Rsh/(Rsh-TER)          (parallel combination inverted)
        C1  = TEC*(tau_big-tau_small)/(S_R*TEC-tau_small)
        C2  = C1*TEC/(C1-TEC)
        R1  = tau_big/C1,  R2 = tau_small/C2
    """
    if Rsh <= TER or (Rsh - TER) < TER * 1e-4:
        return None
    S_R = TER * Rsh / (Rsh - TER)
    if S_R <= 0:
        return None

    if tau_big <= tau_small * 1.0001:
        return None

    denom_c = S_R * TEC - tau_small
    if abs(denom_c) < abs(tau_small) * 1e-4:
        return None

    C1 = TEC * (tau_big - tau_small) / denom_c
    if C1 <= TEC * 1.0001 or C1 <= 0:
        return None

    C2 = C1 * TEC / (C1 - TEC)
    if C2 <= 0:
        return None

    R1 = tau_big / C1
    R2 = tau_small / C2
    if R1 <= 0 or R2 <= 0:
        return None

    if big_tau_is_Ra:
        return R1, R2, C1, C2   # Ra, Rb, Ca, Cb
    else:
        return R2, R1, C2, C1


# ── sampling ────────────────────────────────────────────────────────────────

def sample_identifiable(rng):
    """
    Sample one set of identifiable parameters with guaranteed inversion success.

    The inversion (_invert) requires:
        tau_small  <  S_R * TEC  <  tau_big          (so C1 > TEC > 0)

    We satisfy this exactly by sampling tau_c = S_R * TEC directly in
    (tau_small, tau_big), then deriving TEC = tau_c / S_R.
    No rejection needed for the inversion step.

    Distributions:
        log10(tau_big)   ~ U(-5, 0)           →  tau_big in [10μs, 1s]
        log10(tau_ratio) ~ U(log10(1.5), 4)   →  ratio in [1.5, 10000], flat in log
        log10(tau_c)     ~ U(log10(tau_small) + eps, log10(tau_big) - eps)
                                               →  tau_c ∈ (tau_small, tau_big)
        log10(S_R)       ~ U(1, 4)            →  Ra+Rb in [10, 10000] Ω
        log10(Rsh/S_R)   ~ U(0.1, 1.5)       →  Rsh ∈ (S_R, 30·S_R]

    Returns None only if TEC falls outside the physical capacitance range
    [1nF, 1mF] or if derived Ra/Rb/Ca/Cb fall outside sanity bounds.
    """
    log_tau_big   = rng.uniform(-5.0, 0.0)
    log_tau_ratio = rng.uniform(np.log10(1.5), 4.0)

    tau_big   = 10.0 ** log_tau_big
    tau_ratio = 10.0 ** log_tau_ratio
    tau_small = tau_big / tau_ratio

    # tau_c = S_R * TEC must lie strictly between tau_small and tau_big
    log_tc_lo = np.log10(tau_small) + 0.05    # slight margin
    log_tc_hi = np.log10(tau_big)   - 0.05
    if log_tc_lo >= log_tc_hi:
        return None
    log_tau_c = rng.uniform(log_tc_lo, log_tc_hi)
    tau_c = 10.0 ** log_tau_c

    log_SR = rng.uniform(1.0, 4.0)            # Ra+Rb in [10, 10000] Ω
    S_R    = 10.0 ** log_SR

    TEC = tau_c / S_R
    if not (1e-9 <= TEC <= 1e-3):
        return None

    log_rsh_ratio = rng.uniform(0.1, 1.5)    # Rsh = S_R * 10^x
    Rsh = S_R * (10.0 ** log_rsh_ratio)
    TER = Rsh * S_R / (Rsh + S_R)

    big_tau_is_Ra = rng.random() < 0.5
    result = _invert(tau_big, tau_small, TER, TEC, Rsh, big_tau_is_Ra)
    if result is None:
        return None

    Ra, Rb, Ca, Cb = result

    # Sanity bounds on physical parameters
    for v in (Ra, Rb):
        if not (1.0 <= v <= 1e5):
            return None
    for v in (Ca, Cb):
        if not (1e-9 <= v <= 1e-2):
            return None

    return Ra, Rb, Ca, Cb, Rsh, tau_big, tau_small, TER, TEC, tau_ratio


# ── main ─────────────────────────────────────────────────────────────────────

def generate(
    n_samples: int = 100_000,
    n_freq:    int = 100,
    snr_db:    float = 40.0,
    output_dir: str = "data/identifiable_uniform_v1",
    seed:      int = 42,
):
    rng = np.random.default_rng(seed)
    sim = RandlesCircuitSimulator()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    freqs = np.logspace(-1, 6, n_freq)

    Z_real_cols = [f"Z_real_{i}" for i in range(n_freq)]
    Z_imag_cols = [f"Z_imag_{i}" for i in range(n_freq)]

    # Stratified sampling: 4 equal strata across log10(tau_ratio) decades.
    # Each stratum gets n_samples/4 circuits so the model sees equal density
    # at tau_ratio 2-5x (hardest) and tau_ratio 1000-10000x (easy/degenerate).
    strata = [
        (np.log10(1.5), np.log10(5.0)),    # [1.5, 5)    — nearly overlapping arcs
        (np.log10(5.0), np.log10(20.0)),   # [5, 20)     — mild separation
        (np.log10(20.0), np.log10(200.0)), # [20, 200)   — clear separation
        (np.log10(200.0), 4.0),            # [200, 10000]— strongly separated
    ]
    per_stratum = n_samples // len(strata)

    rows = []
    total_accepted = 0

    print(f"Generating {n_samples:,} circuits sampled in identifiable space...")
    print(f"  Strategy:   stratified uniform in log10(tau_ratio) — {len(strata)} strata")
    print(f"  tau_big:    log10-uniform in [10μs, 1s]")
    print(f"  Output:     {out}")

    for si, (log_ratio_lo, log_ratio_hi) in enumerate(strata):
        stratum_label = f"[{10**log_ratio_lo:.1f}x, {10**log_ratio_hi:.1f}x)"
        accepted = 0
        attempts = 0

        while accepted < per_stratum:
            attempts += 1

            log_tau_big   = rng.uniform(-5.0, 0.0)
            log_tau_ratio = rng.uniform(log_ratio_lo, log_ratio_hi)
            tau_big   = 10.0 ** log_tau_big
            tau_ratio = 10.0 ** log_tau_ratio
            tau_small = tau_big / tau_ratio

            log_tc_lo = np.log10(tau_small) + 0.05
            log_tc_hi = np.log10(tau_big)   - 0.05
            if log_tc_lo >= log_tc_hi:
                continue
            tau_c = 10.0 ** rng.uniform(log_tc_lo, log_tc_hi)

            S_R = 10.0 ** rng.uniform(1.0, 4.0)
            TEC = tau_c / S_R
            if not (1e-9 <= TEC <= 1e-3):
                continue

            Rsh = S_R * (10.0 ** rng.uniform(0.1, 1.5))
            TER = Rsh * S_R / (Rsh + S_R)

            result = _invert(tau_big, tau_small, TER, TEC, Rsh,
                             big_tau_is_Ra=(rng.random() < 0.5))
            if result is None:
                continue

            Ra, Rb, Ca, Cb = result
            if not all(1.0 <= v <= 1e5 for v in (Ra, Rb)):
                continue
            if not all(1e-9 <= v <= 1e-2 for v in (Ca, Cb)):
                continue

            Zr, Zi = sim.compute_impedance(freqs, Ra, Rb, Ca, Cb, Rsh)
            noise_amp = np.power(10.0, -snr_db / 20.0) * np.abs(Zr).mean()
            Zr = Zr + rng.normal(0, noise_amp, n_freq)
            Zi = Zi + rng.normal(0, noise_amp, n_freq)

            row = {
                "Ra": Ra, "Rb": Rb, "Ca": Ca, "Cb": Cb, "Rsh": Rsh,
                "TER": TER, "TEC": TEC,
                "tau_big": tau_big, "tau_small": tau_small, "tau_ratio": tau_ratio,
            }
            for i in range(n_freq):
                row[Z_real_cols[i]] = float(Zr[i])
                row[Z_imag_cols[i]] = float(Zi[i])

            rows.append(row)
            accepted += 1
            total_accepted += 1

        print(f"  Stratum {si+1}/4  tau_ratio {stratum_label:<20}  "
              f"{accepted:,} circuits  (accept {accepted/attempts*100:.1f}%)")

    df = pd.DataFrame(rows)

    # Verify tau ratio distribution
    tau_ratios = df["tau_ratio"].values
    print(f"\nTau ratio distribution (target: equal density per stratum):")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  p{p:2d}: {np.percentile(tau_ratios, p):.1f}x")

    # Split 80/10/10
    idx = rng.permutation(len(df))
    n_train = int(0.80 * n_samples)
    n_val   = int(0.10 * n_samples)

    train_df = df.iloc[idx[:n_train]]
    val_df   = df.iloc[idx[n_train:n_train + n_val]]
    test_df  = df.iloc[idx[n_train + n_val:]]

    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out  / "val.csv",   index=False)
    test_df.to_csv(out / "test.csv",  index=False)

    print(f"\nSplit: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    metadata = {
        "version": "identifiable_uniform_v1",
        "n_samples": n_samples,
        "n_train": len(train_df),
        "n_val":   len(val_df),
        "n_test":  len(test_df),
        "sampling": "identifiable_space",
        "tau_ratio_range": [1.5, 10000.0],
        "tau_ratio_log_uniform": True,
        "frequencies": {
            "min": float(freqs[0]),
            "max": float(freqs[-1]),
            "count": n_freq,
        },
        "snr_db": snr_db,
        "seed": seed,
        "description": (
            "Sampled uniformly in log10(tau_ratio) to eliminate the large-tau-ratio "
            "bias present in datasets generated by rejection sampling from raw parameters."
        ),
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved to {out}/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=100_000)
    p.add_argument("--snr-db",    type=float, default=40.0)
    p.add_argument("--output-dir", type=str, default="data/identifiable_uniform_v1")
    p.add_argument("--seed",      type=int, default=42)
    args = p.parse_args()

    generate(
        n_samples=args.n_samples,
        snr_db=args.snr_db,
        output_dir=args.output_dir,
        seed=args.seed,
    )
