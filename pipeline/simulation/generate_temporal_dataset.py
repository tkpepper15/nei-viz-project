#!/usr/bin/env python3
"""
Generate the temporal RPE EIS dataset.

Each trajectory contains:
    - n_timepoints consecutive EIS spectra
    - Ground-truth [Ra, Rb, Ca, Cb, Rsh] at every timepoint (log10 space)
    - Derived parameters: TER, TEC, tau_big, tau_small, Rsh
    - Measurement noise at a target SNR

Output: HDF5 file with chunked arrays, plus a CSV sample for inspection.

Usage
-----
    python generate_temporal_dataset.py \\
        --output      data/temporal_v1 \\
        --n-traj      10000 \\
        --n-tp        200 \\
        --dt-min      3.0 \\
        --n-freq      100 \\
        --snr-db      40 \\
        --seed        42

HDF5 layout
-----------
    /impedance_real   (N, T, F) float32   real component
    /impedance_imag   (N, T, F) float32   imaginary component
    /params_log10     (N, T, 5) float32   [Ra, Rb, Ca, Cb, Rsh]
    /derived_log10    (N, T, 5) float32   [tau_big, tau_small, TER, TEC, Rsh] log10
    /time_minutes     (N, T)    float32
    /pathology        (N,)      h5py special dtype string
    /frequencies      (F,)      float64   Hz
    /metadata         attrs     JSON string
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

# Allow running as a standalone script from any working directory
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation.trajectory_generator import TrajectoryGenerator
from simulation.pathology_models import PathologyType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FREQ_MIN_HZ = 0.1
FREQ_MAX_HZ = 1e5     # 100 kHz upper bound (avoids inductive artifacts)
N_FREQ      = 100


def _logspace_freqs(n: int = N_FREQ) -> np.ndarray:
    return np.logspace(np.log10(FREQ_MIN_HZ), np.log10(FREQ_MAX_HZ), n)


# ---------------------------------------------------------------------------
# RPE circuit impedance (self-contained, no external dependencies)
# ---------------------------------------------------------------------------

def _impedance_batch(params_log10: np.ndarray, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized impedance for B parameter sets at F frequencies.

    Parameters
    ----------
    params_log10 : (B, 5)  log10[Ra, Rb, Ca, Cb, Rsh]
    omega        : (F,)    angular frequencies [rad/s]

    Returns
    -------
    Z_real, Z_imag : (B, F) each
    """
    p = 10.0 ** params_log10                    # (B, 5) linear
    Ra  = p[:, 0:1]                             # (B, 1)
    Rb  = p[:, 1:2]
    Ca  = p[:, 2:3]
    Cb  = p[:, 3:4]
    Rsh = p[:, 4:5]
    w   = omega[np.newaxis, :]                  # (1, F)

    tau_a = Ra * Ca                             # (B, 1)
    tau_b = Rb * Cb

    Za_r = Ra    / (1.0 + (w * tau_a) ** 2)
    Za_i = -(w * Ra ** 2 * Ca) / (1.0 + (w * tau_a) ** 2)
    Zb_r = Rb    / (1.0 + (w * tau_b) ** 2)
    Zb_i = -(w * Rb ** 2 * Cb) / (1.0 + (w * tau_b) ** 2)

    Zs_r = Za_r + Zb_r                         # series RC
    Zs_i = Za_i + Zb_i
    denom = (Rsh + Zs_r) ** 2 + Zs_i ** 2
    Z_r   = (Rsh * Zs_r * (Rsh + Zs_r) + Rsh * Zs_i ** 2) / denom
    Z_i   = (Rsh ** 2 * Zs_i) / denom
    return Z_r, Z_i


def _add_noise(Z_r: np.ndarray, Z_i: np.ndarray, snr_db: float,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Add Gaussian noise to achieve target SNR on impedance magnitude."""
    mag      = np.sqrt(Z_r ** 2 + Z_i ** 2)
    sigma    = mag / (10.0 ** (snr_db / 20.0))
    Z_r_n    = Z_r + rng.standard_normal(Z_r.shape) * sigma
    Z_i_n    = Z_i + rng.standard_normal(Z_i.shape) * sigma
    return Z_r_n, Z_i_n


# ---------------------------------------------------------------------------
# Derived parameters
# ---------------------------------------------------------------------------

def _derive_identifiable(params_log10: np.ndarray) -> np.ndarray:
    """
    (T, 5) log10[Ra, Rb, Ca, Cb, Rsh] → (T, 5) log10[tau_big, tau_small, TER, TEC, Rsh]
    """
    p   = 10.0 ** params_log10
    Ra  = p[:, 0]; Rb  = p[:, 1]
    Ca  = p[:, 2]; Cb  = p[:, 3]
    Rsh = p[:, 4]

    tau_a = Ra * Ca
    tau_b = Rb * Cb
    tau_big   = np.maximum(tau_a, tau_b)
    tau_small = np.minimum(tau_a, tau_b)

    TER = Rsh * (Ra + Rb) / np.clip(Rsh + Ra + Rb, 1e-12, None)
    TEC = Ca * Cb / np.clip(Ca + Cb, 1e-30, None)

    out = np.column_stack([
        np.log10(np.clip(tau_big,   1e-10, None)),
        np.log10(np.clip(tau_small, 1e-10, None)),
        np.log10(np.clip(TER,       1e-6,  None)),
        np.log10(np.clip(TEC,       1e-30, None)),
        params_log10[:, 4],   # log10(Rsh) unchanged
    ])
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_dataset(
    output_dir:     str,
    n_trajectories: int   = 10_000,
    n_timepoints:   int   = 200,
    dt_minutes:     float = 3.0,
    n_freq:         int   = N_FREQ,
    snr_db:         float = 40.0,
    seed:           int   = 42,
    chunk_size:     int   = 128,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    freqs = _logspace_freqs(n_freq)
    omega = 2.0 * np.pi * freqs

    gen  = TrajectoryGenerator(seed=seed)
    rng  = np.random.default_rng(seed + 1)   # separate seed for noise

    h5_path = out_path / "temporal_dataset.h5"
    print(f"Writing {n_trajectories} trajectories × {n_timepoints} timepoints "
          f"× {n_freq} freq to {h5_path}", flush=True)

    start = time.time()

    # Chunk shape aligned with write pattern: one full trajectory per chunk.
    # Writing ds_zr[i] (shape T×F) touches exactly 1 chunk, avoiding the
    # (128, 1, F) misalignment that caused 200 gzip cycles per trajectory.
    iz_chunk  = (1, n_timepoints, n_freq)   # ~80 KB per chunk at float32
    ipar_chunk = (1, n_timepoints, 5)       # ~4 KB per chunk

    with h5py.File(h5_path, "w") as h5:
        ds_zr = h5.create_dataset(
            "impedance_real", shape=(n_trajectories, n_timepoints, n_freq),
            dtype="float32", chunks=iz_chunk,
            compression="gzip", compression_opts=4,
        )
        ds_zi = h5.create_dataset(
            "impedance_imag", shape=(n_trajectories, n_timepoints, n_freq),
            dtype="float32", chunks=iz_chunk,
            compression="gzip", compression_opts=4,
        )
        ds_params = h5.create_dataset(
            "params_log10", shape=(n_trajectories, n_timepoints, 5),
            dtype="float32", chunks=ipar_chunk,
            compression="gzip", compression_opts=4,
        )
        ds_derived = h5.create_dataset(
            "derived_log10", shape=(n_trajectories, n_timepoints, 5),
            dtype="float32", chunks=ipar_chunk,
            compression="gzip", compression_opts=4,
        )
        ds_time = h5.create_dataset(
            "time_minutes", shape=(n_trajectories, n_timepoints),
            dtype="float32",
        )
        dt_path = h5py.special_dtype(vlen=str)
        ds_path = h5.create_dataset(
            "pathology", shape=(n_trajectories,), dtype=dt_path,
        )
        h5.create_dataset("frequencies", data=freqs)

        meta = {
            "n_trajectories": n_trajectories,
            "n_timepoints":   n_timepoints,
            "dt_minutes":     dt_minutes,
            "n_freq":         n_freq,
            "freq_min_hz":    FREQ_MIN_HZ,
            "freq_max_hz":    FREQ_MAX_HZ,
            "snr_db":         snr_db,
            "seed":           seed,
            "params":         ["Ra", "Rb", "Ca", "Cb", "Rsh"],
            "derived":        ["tau_big", "tau_small", "TER", "TEC", "Rsh"],
            "units":          "log10(Ω) / log10(F) / log10(s)",
            "cross_section_cm2": gen.area,
        }
        h5.attrs["metadata"] = json.dumps(meta)

        pathology_counts: dict[str, int] = {}
        log_interval = max(1, n_trajectories // 50)   # ~2% increments

        for i in range(n_trajectories):
            traj = gen.generate(n_timepoints=n_timepoints, dt_minutes=dt_minutes)
            p_name = traj.pathology.value
            pathology_counts[p_name] = pathology_counts.get(p_name, 0) + 1

            Z_r, Z_i = _impedance_batch(traj.params_log10, omega)  # (T, F)
            Z_r_n, Z_i_n = _add_noise(Z_r, Z_i, snr_db, rng)

            ds_zr[i]      = Z_r_n.astype(np.float32)
            ds_zi[i]      = Z_i_n.astype(np.float32)
            ds_params[i]  = traj.params_log10.astype(np.float32)
            ds_derived[i] = _derive_identifiable(traj.params_log10)
            ds_time[i]    = traj.time_minutes.astype(np.float32)
            ds_path[i]    = p_name

            if (i + 1) % log_interval == 0:
                elapsed  = time.time() - start
                rate     = (i + 1) / elapsed
                eta      = (n_trajectories - i - 1) / rate
                print(f"  {i + 1:6d}/{n_trajectories}  "
                      f"[{elapsed:5.0f}s elapsed, ETA {eta:5.0f}s]",
                      flush=True)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s", flush=True)
    print("Pathology distribution:")
    for name, count in sorted(pathology_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / n_trajectories
        print(f"  {name:<22s}: {count:6d}  ({pct:.1f}%)", flush=True)

    # ------------------------------------------------------------------
    # Save a small CSV sample (first 5 trajectories, first timepoint)
    # for quick inspection without opening the HDF5 file.
    # ------------------------------------------------------------------
    import csv
    sample_csv = out_path / "sample_trajectories.csv"
    n_sample = min(5, n_trajectories)
    with h5py.File(h5_path, "r") as h5:
        params  = h5["params_log10"][:n_sample, :, :]   # (5, T, 5)
        derived = h5["derived_log10"][:n_sample, :, :]
        times   = h5["time_minutes"][:n_sample, :]
        paths   = [h5["pathology"][k].decode() if isinstance(h5["pathology"][k], bytes)
                   else h5["pathology"][k] for k in range(n_sample)]

    rows = []
    for i in range(n_sample):
        for t in range(n_timepoints):
            row = {
                "traj_id":    i,
                "pathology":  paths[i],
                "time_min":   float(times[i, t]),
                "Ra_log10":   float(params[i, t, 0]),
                "Rb_log10":   float(params[i, t, 1]),
                "Ca_log10":   float(params[i, t, 2]),
                "Cb_log10":   float(params[i, t, 3]),
                "Rsh_log10":  float(params[i, t, 4]),
                "tau_big_log10":   float(derived[i, t, 0]),
                "tau_small_log10": float(derived[i, t, 1]),
                "TER_log10":       float(derived[i, t, 2]),
                "TEC_log10":       float(derived[i, t, 3]),
            }
            rows.append(row)

    with open(sample_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Sample CSV written to {sample_csv}")

    # ------------------------------------------------------------------
    # Save metadata JSON
    # ------------------------------------------------------------------
    with open(out_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    file_size_mb = h5_path.stat().st_size / 1e6
    print(f"\nHDF5 file size: {file_size_mb:.0f} MB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate temporal RPE EIS dataset")
    p.add_argument("--output",   default="data/temporal_v1", help="Output directory")
    p.add_argument("--n-traj",   type=int,   default=10_000,  help="Number of trajectories")
    p.add_argument("--n-tp",     type=int,   default=200,     help="Timepoints per trajectory")
    p.add_argument("--dt-min",   type=float, default=3.0,     help="Interval [minutes]")
    p.add_argument("--n-freq",   type=int,   default=N_FREQ,  help="Frequency points")
    p.add_argument("--snr-db",   type=float, default=40.0,    help="Measurement SNR [dB]")
    p.add_argument("--seed",     type=int,   default=42,      help="Random seed")
    p.add_argument("--chunk",    type=int,   default=128,     help="HDF5 chunk size")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_dataset(
        output_dir     = args.output,
        n_trajectories = args.n_traj,
        n_timepoints   = args.n_tp,
        dt_minutes     = args.dt_min,
        n_freq         = args.n_freq,
        snr_db         = args.snr_db,
        seed           = args.seed,
        chunk_size     = args.chunk,
    )
