#!/usr/bin/env python3
"""
Training dataset generator for the FisherAwareTransformer.

Generates mixed_distribution_v2, a 100k-circuit dataset with four components
designed to cover the real RPE measurement regime:
  - realistic_broad  35%  log-uniform Ra/Rb up to 10^4.5 Ohm, symmetric Ca/Cb
  - rpe_biorealistic 25%  high TER (300-700 Ohm), Ca ~ 2*Cb (log-normal ratio)
  - random_ood       30%  very wide ranges for OOD robustness
  - edge_cases       10%  extreme asymmetries and degenerate regimes

Usage:
    cd pipeline
    python scripts/analysis/generate_dataset.py
    python scripts/analysis/generate_dataset.py --n-samples 50000 --output-dir data/custom
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.randles_circuit_simulator import RandlesCircuitSimulator


# ---------------------------------------------------------------------------
# Parameter generation helpers
# ---------------------------------------------------------------------------

def _ter_from_params(Ra, Rb, Rsh):
    return (Rsh * (Ra + Rb)) / (Rsh + Ra + Rb)


def _tec_from_params(Ca, Cb):
    return (Ca * Cb) / (Ca + Cb)


def _make_circuit(Ra, Rb, Ca, Cb, Rsh, tag):
    return {
        'Ra': Ra, 'Rb': Rb, 'Ca': Ca, 'Cb': Cb, 'Rsh': Rsh,
        'TER': _ter_from_params(Ra, Rb, Rsh),
        'TEC': _tec_from_params(Ca, Cb),
        'tau_a': Ra * Ca,
        'tau_b': Rb * Cb,
        'distribution': tag,
    }


# ---------------------------------------------------------------------------
# Component 1: broad log-uniform (replaces old "realistic")
# Ra/Rb: 10 - 31,623 Ohm (10^1 to 10^4.5)
# Ca/Cb: symmetric log-uniform
# Rsh:   10 - 10,000 Ohm
# ---------------------------------------------------------------------------

def generate_realistic_broad(n, seed=42):
    rng = np.random.default_rng(seed)
    circuits = []
    for _ in range(n):
        Ra  = 10 ** rng.uniform(1.0, 4.5)
        Rb  = 10 ** rng.uniform(1.0, 4.5)
        Ca  = 10 ** rng.uniform(-7.0, -4.5)
        Cb  = 10 ** rng.uniform(-7.0, -4.5)
        Rsh = 10 ** rng.uniform(1.0, 4.0)
        circuits.append(_make_circuit(Ra, Rb, Ca, Cb, Rsh, 'realistic_broad'))
    return circuits


# ---------------------------------------------------------------------------
# Component 2: RPE bio-realistic
# TER constrained to 300-700 Ohm (typical healthy RPE monolayer).
# Ca/Cb ratio ~ LogNormal(log(2), 0.4) reflecting larger apical surface area.
# Strategy:
#   - Sample total resistance R_total = Ra + Rb, and TER directly
#   - Back-calculate Rsh = TER * R_total / (R_total - TER)
#   - Split R_total into Ra, Rb with a log-uniform ratio
# ---------------------------------------------------------------------------

def generate_rpe_biorealistic(n, seed=43):
    rng = np.random.default_rng(seed)
    circuits = []
    attempts = 0
    max_attempts = n * 20

    while len(circuits) < n and attempts < max_attempts:
        attempts += 1

        # Target TER in 300-700 Ohm
        TER_target = rng.uniform(300.0, 700.0)

        # Total membrane resistance R_total = Ra + Rb must exceed TER
        # Sample R_total in a range that makes Rsh realistic (100-5000 Ohm)
        # From: Rsh = TER * R_total / (R_total - TER)
        # We want Rsh in [100, 5000] => R_total in [TER*(1 + 100/5000), TER*(1 + 1)]
        R_total_min = TER_target * 1.05   # Rsh ~= 20 * R_total (very leaky shunt)
        R_total_max = TER_target * 10.0   # Rsh ~= R_total * TER / (9*TER) ~ R_total/9
        if R_total_max > 40000:
            R_total_max = 40000.0
        if R_total_min >= R_total_max:
            continue

        R_total = 10 ** rng.uniform(np.log10(R_total_min), np.log10(R_total_max))
        Rsh = TER_target * R_total / (R_total - TER_target)

        if Rsh < 50 or Rsh > 20000:
            continue

        # Split R_total into Ra, Rb via a log-uniform ratio
        log_ratio_R = rng.uniform(-1.0, 1.0)   # Ra/Rb in [0.1, 10]
        R_ratio = 10 ** log_ratio_R
        Rb = R_total / (1.0 + R_ratio)
        Ra = R_total - Rb

        if Ra < 5 or Rb < 5 or Ra > 32000 or Rb > 32000:
            continue

        # Ca/Cb ratio: log-normal centered at log10(2) ~ 0.301, std 0.4
        # Positive side: Ca > Cb (apical larger due to microvilli)
        log_ratio_C = rng.normal(np.log10(2.0), 0.4)   # log10(Ca/Cb)
        log_TEC = rng.uniform(np.log10(5e-8), np.log10(5e-5))
        TEC = 10 ** log_TEC
        # Recover Ca, Cb from TEC and ratio
        ratio_C = 10 ** log_ratio_C   # Ca/Cb (linear)
        # TEC = Ca*Cb/(Ca+Cb), Ca = ratio_C * Cb => TEC = ratio_C * Cb^2 / ((ratio_C+1)*Cb)
        # => Cb = TEC * (ratio_C + 1) / ratio_C
        Cb = TEC * (ratio_C + 1.0) / ratio_C
        Ca = ratio_C * Cb

        if Ca < 5e-9 or Cb < 5e-9 or Ca > 2e-4 or Cb > 2e-4:
            continue

        circuits.append(_make_circuit(Ra, Rb, Ca, Cb, Rsh, 'rpe_biorealistic'))

    if len(circuits) < n:
        print(f"  WARNING: rpe_biorealistic only generated {len(circuits)}/{n} circuits "
              f"(acceptance {len(circuits)/attempts*100:.1f}%)")
    return circuits


# ---------------------------------------------------------------------------
# Component 3: random OOD coverage
# Very wide ranges to prevent confident wrong predictions on anything unusual.
# ---------------------------------------------------------------------------

def generate_random_ood(n, seed=44):
    rng = np.random.default_rng(seed)
    circuits = []
    for _ in range(n):
        Ra  = 10 ** rng.uniform(0.0, 4.5)   # 1 - 31,623 Ohm
        Rb  = 10 ** rng.uniform(0.0, 4.5)
        Ca  = 10 ** rng.uniform(-8.0, -4.0)  # 10 nF - 100 uF
        Cb  = 10 ** rng.uniform(-8.0, -4.0)
        Rsh = 10 ** rng.uniform(0.5, 4.5)   # 3 - 31,623 Ohm
        circuits.append(_make_circuit(Ra, Rb, Ca, Cb, Rsh, 'random_ood'))
    return circuits


# ---------------------------------------------------------------------------
# Component 4: edge cases
# Three sub-scenarios: high-Ra asymmetry, high-TER tight shunt, extreme Ca/Cb
# ---------------------------------------------------------------------------

def generate_edge_cases(n, seed=45):
    rng = np.random.default_rng(seed)
    circuits = []
    n_per = n // 3

    # Scenario A: strongly asymmetric Ra >> Rb (tests apical dominance)
    for _ in range(n_per):
        Ra  = 10 ** rng.uniform(3.0, 4.5)   # 1000 - 31,623 Ohm
        Rb  = 10 ** rng.uniform(1.0, 2.5)   # 10 - 316 Ohm
        Ca  = 10 ** rng.uniform(-7.0, -5.0)
        Cb  = 10 ** rng.uniform(-7.0, -5.0)
        Rsh = 10 ** rng.uniform(2.0, 4.0)
        circuits.append(_make_circuit(Ra, Rb, Ca, Cb, Rsh, 'edge_Ra_dominant'))

    # Scenario B: very low Rsh (leaky tight junction)
    for _ in range(n_per):
        Ra  = 10 ** rng.uniform(1.5, 3.5)
        Rb  = 10 ** rng.uniform(1.5, 3.5)
        Ca  = 10 ** rng.uniform(-7.0, -5.0)
        Cb  = 10 ** rng.uniform(-7.0, -5.0)
        Rsh = 10 ** rng.uniform(0.5, 1.5)   # 3 - 32 Ohm (very leaky)
        circuits.append(_make_circuit(Ra, Rb, Ca, Cb, Rsh, 'edge_leaky_shunt'))

    # Scenario C: extreme Ca/Cb asymmetry (Ca >> Cb or Ca << Cb)
    for _ in range(n - 2 * n_per):
        Ra  = 10 ** rng.uniform(1.5, 3.5)
        Rb  = 10 ** rng.uniform(1.5, 3.5)
        if rng.uniform() > 0.5:
            Ca  = 10 ** rng.uniform(-5.5, -4.5)  # large Ca
            Cb  = 10 ** rng.uniform(-7.5, -6.5)  # small Cb
        else:
            Ca  = 10 ** rng.uniform(-7.5, -6.5)
            Cb  = 10 ** rng.uniform(-5.5, -4.5)
        Rsh = 10 ** rng.uniform(1.5, 3.5)
        circuits.append(_make_circuit(Ra, Rb, Ca, Cb, Rsh, 'edge_asymmetric_C'))

    return circuits


# ---------------------------------------------------------------------------
# Impedance simulation
# ---------------------------------------------------------------------------

def simulate_spectra(circuits, frequencies):
    sim = RandlesCircuitSimulator()
    n_freq = len(frequencies)
    print(f"\nSimulating impedance spectra for {len(circuits):,} circuits...")

    for i, circuit in enumerate(tqdm(circuits)):
        Z_real, Z_imag = sim.compute_impedance(
            frequencies,
            circuit['Ra'], circuit['Rb'],
            circuit['Ca'], circuit['Cb'],
            circuit['Rsh']
        )
        for j, freq in enumerate(frequencies):
            circuit[f'Z_real_{j}'] = float(Z_real[j])
            circuit[f'Z_imag_{j}'] = float(Z_imag[j])

    return circuits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=100000)
    parser.add_argument('--output-dir', type=str, default='data/mixed_distribution_v2')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    n = args.n_samples
    n_broad    = int(n * 0.35)
    n_rpe      = int(n * 0.25)
    n_ood      = int(n * 0.30)
    n_edge     = n - n_broad - n_rpe - n_ood

    print("=" * 70)
    print("MIXED DISTRIBUTION DATASET v2")
    print("=" * 70)
    print(f"  realistic_broad  : {n_broad:,}  (35%)")
    print(f"  rpe_biorealistic : {n_rpe:,}  (25%)")
    print(f"  random_ood       : {n_ood:,}  (30%)")
    print(f"  edge_cases       : {n_edge:,}  (10%)")
    print(f"  total            : {n:,}")
    print(f"  output           : {output_dir}")

    print("\n[1/4] Generating realistic_broad circuits...")
    broad = generate_realistic_broad(n_broad, seed=args.seed)

    print("[2/4] Generating rpe_biorealistic circuits...")
    rpe = generate_rpe_biorealistic(n_rpe, seed=args.seed + 1)

    print("[3/4] Generating random_ood circuits...")
    ood = generate_random_ood(n_ood, seed=args.seed + 2)

    print("[4/4] Generating edge_cases circuits...")
    edge = generate_edge_cases(n_edge, seed=args.seed + 3)

    all_circuits = broad + rpe + ood + edge
    print(f"\nTotal circuits: {len(all_circuits):,}")

    # Frequencies: 0.1 Hz to 1 MHz, 100 log-spaced points
    frequencies = np.logspace(-1, 6, 100)

    all_circuits = simulate_spectra(all_circuits, frequencies)

    df = pd.DataFrame(all_circuits)

    # Print coverage stats
    print("\n--- Coverage check ---")
    for param in ['Ra', 'Rb', 'Ca', 'Cb', 'Rsh', 'TER']:
        vals = df[param].values
        print(f"  {param:5s}: [{vals.min():.2e}, {vals.max():.2e}]  "
              f"median={np.median(vals):.2e}  log-width={np.log10(vals.max()/vals.min()):.1f}d")

    ter_vals = df['TER'].values
    high_ter_pct = np.mean((ter_vals >= 300) & (ter_vals <= 700)) * 100
    print(f"\n  TER in 300-700 Ohm: {high_ter_pct:.1f}% of all circuits")

    ca_cb_ratio = df['Ca'].values / df['Cb'].values
    print(f"  Ca/Cb ratio: median={np.median(ca_cb_ratio):.2f}  "
          f"p25={np.percentile(ca_cb_ratio, 25):.2f}  "
          f"p75={np.percentile(ca_cb_ratio, 75):.2f}")

    # Shuffle and split
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(df))
    n_train = int(len(df) * 0.80)
    n_val   = int(len(df) * 0.10)

    df_train = df.iloc[idx[:n_train]].reset_index(drop=True)
    df_val   = df.iloc[idx[n_train:n_train + n_val]].reset_index(drop=True)
    df_test  = df.iloc[idx[n_train + n_val:]].reset_index(drop=True)

    print(f"\nSaving splits to {output_dir} ...")
    df_train.to_csv(output_dir / 'train.csv', index=False)
    df_val.to_csv(output_dir / 'val.csv', index=False)
    df_test.to_csv(output_dir / 'test.csv', index=False)
    print(f"  train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}")

    metadata = {
        'version': 'v2',
        'n_samples': len(df),
        'n_train': len(df_train),
        'n_val': len(df_val),
        'n_test': len(df_test),
        'distributions': {
            'realistic_broad': n_broad,
            'rpe_biorealistic': len(rpe),
            'random_ood': n_ood,
            'edge_cases': n_edge,
        },
        'improvements_over_v1': [
            'Ra/Rb extended to 10^4.5 = 31,623 Ohm (was 10^3.5 = 3,162 Ohm)',
            'rpe_biorealistic component: TER=300-700 Ohm, Ca/Cb~LogNormal(log(2),0.4)',
            'random_ood also extended to 10^4.5 upper bound',
            'edge_cases include strongly asymmetric Ra >> Rb scenarios',
        ],
        'frequencies': {
            'min': float(frequencies[0]),
            'max': float(frequencies[-1]),
            'count': int(len(frequencies)),
        },
        'parameter_ranges': {
            param: {
                'min': float(df[param].min()),
                'max': float(df[param].max()),
                'median': float(df[param].median()),
                'p10': float(np.percentile(df[param], 10)),
                'p90': float(np.percentile(df[param], 90)),
            }
            for param in ['Ra', 'Rb', 'Ca', 'Cb', 'Rsh', 'TER', 'TEC', 'tau_a', 'tau_b']
        },
        'coverage': {
            'TER_300_700_pct': float(high_ter_pct),
            'CaCb_ratio_median': float(np.median(ca_cb_ratio)),
        },
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\nDone. Next step:")
    print(f"  python 02_train_transformer.py --data {args.output_dir} --epochs 100 --augment --derived-weight 0.3")


if __name__ == '__main__':
    main()
