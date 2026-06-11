#!/usr/bin/env python3
"""
Estimate process noise covariance Σ_Q and ATP mean shift from Ussing chamber data.

Reads ECM fit results for the 3 Ussing samples, computes inter-timepoint log10 deltas
in [Ra, Rb, Ca, Cb, Rsh] space, and estimates:
  - Σ_Q : (5,5) full covariance from non-ATP biological drift steps
  - mu_atp : (5,) mean shift vector for ATP-stimulated steps

Output saved to data/process_noise.npz.

Column mapping in 'fit results.csv' (pg_absZr_* from ECM best fit):
  pg_absZr_3 = Ra   (Ω·cm²)
  pg_absZr_4 = Ca   (F/cm²)
  pg_absZr_5 = Rb   (Ω·cm²)
  pg_absZr_6 = Cb   (F/cm²)
  pg_absZr_7 = Rsh  (Ω·cm²)
"""

import numpy as np
import pandas as pd
from pathlib import Path

CSV_PATH = Path(__file__).parent.parent / "docs" / "fit results.csv"
OUT_PATH = Path(__file__).parent / "data" / "process_noise.npz"

USSING_SAMPLES = {
    "Sample 1": {"meas_id": "20211021_183356", "atp_span": (7, 11)},
    "Sample 2": {"meas_id": "20211022_171622", "atp_span": (10, 15)},
    "Sample 3": {"meas_id": "20211014_165514", "atp_span": (9, 12)},
}

# pg_absZr_* column indices (1-based suffix) mapping to [Ra, Rb, Ca, Cb, Rsh]
COL_RA  = "pg_absZr_3"
COL_CA  = "pg_absZr_4"
COL_RB  = "pg_absZr_5"
COL_CB  = "pg_absZr_6"
COL_RSH = "pg_absZr_7"
PARAM_NAMES = ["Ra", "Rb", "Ca", "Cb", "Rsh"]


def load_sample_log10(df: pd.DataFrame, meas_id: str) -> np.ndarray:
    """
    Extract log10 parameter matrix for one sample, sorted by meas_idx.
    Returns (T, 5) in order [Ra, Rb, Ca, Cb, Rsh].
    """
    sub = df[df["meas_ID"] == meas_id].sort_values("meas_idx")
    Ra  = sub[COL_RA].values.astype(float)
    Rb  = sub[COL_RB].values.astype(float)
    Ca  = sub[COL_CA].values.astype(float)
    Cb  = sub[COL_CB].values.astype(float)
    Rsh = sub[COL_RSH].values.astype(float)
    params = np.stack([Ra, Rb, Ca, Cb, Rsh], axis=1)
    return np.log10(np.clip(params, 1e-30, None))  # (T, 5)


def classify_steps(n_timepoints: int, atp_span: tuple) -> np.ndarray:
    """
    Boolean mask of length (n_timepoints - 1).
    True  = delta step t->t+1 is an ATP-affected step
           (either t or t+1 is within the ATP window).
    atp_span is a 0-based [start, end] inclusive range.
    """
    atp_start, atp_end = atp_span
    atp_set = set(range(atp_start, atp_end + 1))
    is_atp = np.zeros(n_timepoints - 1, dtype=bool)
    for t in range(n_timepoints - 1):
        if t in atp_set or (t + 1) in atp_set:
            is_atp[t] = True
    return is_atp


def main():
    print(f"Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    all_drift_deltas = []
    all_atp_deltas   = []

    for name, meta in USSING_SAMPLES.items():
        meas_id  = meta["meas_id"]
        atp_span = meta["atp_span"]

        log10_params = load_sample_log10(df, meas_id)
        T = log10_params.shape[0]
        deltas = np.diff(log10_params, axis=0)   # (T-1, 5)
        is_atp = classify_steps(T, atp_span)

        n_drift = int(np.sum(~is_atp))
        n_atp   = int(np.sum(is_atp))
        print(f"\n{name} ({meas_id}): T={T}, drift_steps={n_drift}, atp_steps={n_atp}")

        drift_d = deltas[~is_atp]
        atp_d   = deltas[is_atp]

        print(f"  Drift delta mean  : {drift_d.mean(axis=0).round(4)}")
        print(f"  Drift delta std   : {drift_d.std(axis=0).round(4)}")
        if len(atp_d):
            print(f"  ATP delta mean    : {atp_d.mean(axis=0).round(4)}")

        all_drift_deltas.append(drift_d)
        all_atp_deltas.append(atp_d)

    drift_deltas = np.vstack(all_drift_deltas)   # (N_drift, 5)
    atp_deltas   = np.vstack(all_atp_deltas)     # (N_atp, 5)

    # Full 5x5 process noise covariance from drift steps
    # Regularise with a small diagonal floor so Σ_Q is always PD
    sigma_q = np.cov(drift_deltas.T)             # (5, 5)
    diag_floor = (np.diag(sigma_q) * 0.01).clip(min=1e-6)
    sigma_q += np.diag(diag_floor)
    sigma_q = (sigma_q + sigma_q.T) / 2.0        # enforce symmetry

    # ATP mean shift (log10 per step)
    mu_atp = atp_deltas.mean(axis=0) if len(atp_deltas) else np.zeros(5)

    print("\n--- Estimated Σ_Q (5x5, log10/step) ---")
    print(np.array2string(sigma_q, precision=5, suppress_small=True))
    print(f"\nSigma_Q diagonal sqrt (std/step): {np.sqrt(np.diag(sigma_q)).round(4)}")
    print(f"\nmu_atp (mean log10 shift at ATP): {mu_atp.round(4)}")
    print(f"\n  Ra:  {mu_atp[0]:+.4f}  (10^{mu_atp[0]:.2f}x factor)")
    print(f"  Rb:  {mu_atp[1]:+.4f}  (10^{mu_atp[1]:.2f}x factor)")
    print(f"  Ca:  {mu_atp[2]:+.4f}  (10^{mu_atp[2]:.2f}x factor)")
    print(f"  Cb:  {mu_atp[3]:+.4f}  (10^{mu_atp[3]:.2f}x factor)")
    print(f"  Rsh: {mu_atp[4]:+.4f}  (10^{mu_atp[4]:.2f}x factor = {10**mu_atp[4]:.2f}x)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_PATH,
        sigma_q=sigma_q,
        mu_atp=mu_atp,
        param_names=np.array(PARAM_NAMES),
        n_drift_steps=np.array(len(drift_deltas)),
        n_atp_steps=np.array(len(atp_deltas)),
    )
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
