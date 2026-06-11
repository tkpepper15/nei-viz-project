"""
eval/shared.py  —  Shared utilities for all evaluation experiments.

All experiments import from here. No duplication of impedance models,
metric helpers, data loaders, or CR-bound computations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch

# ── parameter name constants ──────────────────────────────────────────────────

PARAMS_RAW = ["Ra", "Rb", "Ca", "Cb", "Rsh"]
PARAMS_ID  = ["tau_big", "tau_small", "TER", "TEC", "Rsh"]

PATHOLOGIES = [
    "healthy", "maturation", "barrier_breakdown", "apical_injury",
    "basolateral_injury", "oxidative_stress", "recovery", "mixed_pathology", "unknown",
]

# ── physics ───────────────────────────────────────────────────────────────────

def compute_impedance(params_log10: np.ndarray, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    (5,) log10[Ra,Rb,Ca,Cb,Rsh] → (F,) Zr, Zi.
    Also accepts (N, 5) for batch computation → (N, F) each.
    """
    scalar = params_log10.ndim == 1
    p = np.atleast_2d(params_log10)
    Ra  = 10.0 ** p[:, 0:1]; Rb  = 10.0 ** p[:, 1:2]
    Ca  = 10.0 ** p[:, 2:3]; Cb  = 10.0 ** p[:, 3:4]
    Rsh = 10.0 ** p[:, 4:5]
    w = omega[np.newaxis, :]

    Za_r = Ra / (1 + (w * Ra * Ca) ** 2)
    Za_i = -(w * Ra ** 2 * Ca) / (1 + (w * Ra * Ca) ** 2)
    Zb_r = Rb / (1 + (w * Rb * Cb) ** 2)
    Zb_i = -(w * Rb ** 2 * Cb) / (1 + (w * Rb * Cb) ** 2)
    Zs_r = Za_r + Zb_r
    Zs_i = Za_i + Zb_i
    denom = (Rsh + Zs_r) ** 2 + Zs_i ** 2
    Zr = (Rsh * Zs_r * (Rsh + Zs_r) + Rsh * Zs_i ** 2) / denom
    Zi = (Rsh ** 2 * Zs_i) / denom

    return (Zr[0], Zi[0]) if scalar else (Zr, Zi)


def add_noise(Zr: np.ndarray, Zi: np.ndarray, snr_db: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    sigma = 10 ** (-snr_db / 20.0)
    amp = np.sqrt(Zr ** 2 + Zi ** 2)
    scale = amp.mean() * sigma
    return Zr + np.random.randn(*Zr.shape) * scale, Zi + np.random.randn(*Zi.shape) * scale


# ── identifiable-space transform ──────────────────────────────────────────────

def to_identifiable(params_log10: np.ndarray) -> np.ndarray:
    """
    (T, 5) or (5,) log10[Ra,Rb,Ca,Cb,Rsh] → log10[tau_big, tau_small, TER, TEC, Rsh]

    This collapses the Ra↔Rb permutation symmetry: tau_big = max(Ra·Ca, Rb·Cb).
    """
    scalar = params_log10.ndim == 1
    p = np.atleast_2d(params_log10)
    Ra, Rb, Ca, Cb, Rsh = 10.0 ** p[:, 0], 10.0 ** p[:, 1], 10.0 ** p[:, 2], 10.0 ** p[:, 3], 10.0 ** p[:, 4]
    tau_a     = Ra * Ca
    tau_b     = Rb * Cb
    tau_big   = np.maximum(tau_a, tau_b)
    tau_small = np.minimum(tau_a, tau_b)
    TER = Rsh * (Ra + Rb) / np.maximum(Rsh + Ra + Rb, 1e-30)
    TEC = Ca * Cb / np.maximum(Ca + Cb, 1e-30)
    out = np.column_stack([
        np.log10(np.maximum(tau_big,   1e-12)),
        np.log10(np.maximum(tau_small, 1e-12)),
        np.log10(np.maximum(TER,       1e-6)),
        np.log10(np.maximum(TEC,       1e-30)),
        np.log10(np.maximum(Rsh,       1e-6)),
    ])
    return out[0] if scalar else out


def particles_to_id_stats(
    positions: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    (N, 5) particle positions, (N,) weights → (5,) weighted mean, (5,) weighted std
    in log10 identifiable space.
    """
    Ra, Rb, Ca, Cb, Rsh = [10.0 ** positions[:, i] for i in range(5)]
    tau_a     = Ra * Ca
    tau_b     = Rb * Cb
    tau_big   = np.maximum(tau_a, tau_b)
    tau_small = np.minimum(tau_a, tau_b)
    TER = Rsh * (Ra + Rb) / np.maximum(Rsh + Ra + Rb, 1e-30)
    TEC = Ca * Cb / np.maximum(Ca + Cb, 1e-30)

    id_vals = np.column_stack([
        np.log10(np.maximum(tau_big,   1e-12)),
        np.log10(np.maximum(tau_small, 1e-12)),
        np.log10(np.maximum(TER,       1e-6)),
        np.log10(np.maximum(TEC,       1e-30)),
        np.log10(np.maximum(Rsh,       1e-6)),
    ])
    w = weights / (weights.sum() + 1e-300)
    mean = (w[:, None] * id_vals).sum(axis=0)
    var  = (w[:, None] * (id_vals - mean) ** 2).sum(axis=0)
    return mean, np.sqrt(np.maximum(var, 0.0))


# ── metrics ───────────────────────────────────────────────────────────────────

def mae_stats(pred: np.ndarray, truth: np.ndarray) -> dict[str, dict]:
    """
    Compute per-parameter MAE summary over valid rows.

    pred, truth: (N, 5) in log10 identifiable space.
    Returns dict[param_name → {median, mean, p10, p90, n}].
    """
    diff  = np.abs(pred - truth)
    valid = np.all(np.isfinite(diff), axis=1)
    if not valid.any():
        return {p: {} for p in PARAMS_ID}
    d = diff[valid]
    return {
        p: {
            "median": float(np.median(d[:, j])),
            "mean":   float(np.mean(d[:, j])),
            "p10":    float(np.percentile(d[:, j], 10)),
            "p90":    float(np.percentile(d[:, j], 90)),
            "n":      int(valid.sum()),
        }
        for j, p in enumerate(PARAMS_ID)
    }


def symmetric_raw_mae(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """
    Per-parameter mean MAE with optimal Ra↔Rb and Ca↔Cb assignment.

    pred, truth: (T, 5) log10[Ra, Rb, Ca, Cb, Rsh]
    Returns {Ra, Rb, Ca, Cb, Rsh} → mean MAE across T.
    """
    e_rr_d = np.abs(pred[:, 0] - truth[:, 0]) + np.abs(pred[:, 1] - truth[:, 1])
    e_rr_s = np.abs(pred[:, 0] - truth[:, 1]) + np.abs(pred[:, 1] - truth[:, 0])
    swap_r  = e_rr_s < e_rr_d
    e_Ra = np.where(swap_r, np.abs(pred[:, 0] - truth[:, 1]), np.abs(pred[:, 0] - truth[:, 0]))
    e_Rb = np.where(swap_r, np.abs(pred[:, 1] - truth[:, 0]), np.abs(pred[:, 1] - truth[:, 1]))

    e_cc_d = np.abs(pred[:, 2] - truth[:, 2]) + np.abs(pred[:, 3] - truth[:, 3])
    e_cc_s = np.abs(pred[:, 2] - truth[:, 3]) + np.abs(pred[:, 3] - truth[:, 2])
    swap_c  = e_cc_s < e_cc_d
    e_Ca = np.where(swap_c, np.abs(pred[:, 2] - truth[:, 3]), np.abs(pred[:, 2] - truth[:, 2]))
    e_Cb = np.where(swap_c, np.abs(pred[:, 3] - truth[:, 2]), np.abs(pred[:, 3] - truth[:, 3]))

    return {
        "Ra":  float(np.mean(e_Ra)),
        "Rb":  float(np.mean(e_Rb)),
        "Ca":  float(np.mean(e_Ca)),
        "Cb":  float(np.mean(e_Cb)),
        "Rsh": float(np.mean(np.abs(pred[:, 4] - truth[:, 4]))),
    }


def cr_efficiency(cr_std_median: float, method_mae_median: float) -> float:
    """
    CR efficiency = CR_bound_std / method_MAE.

    Interpretation:
      > 1.0  super-efficient (artifact of nonlinear transform; impossible strictly)
      ~ 0.8  near-optimal — 80% of Fisher information extracted
      < 0.3  large gap — most information unused
    """
    if method_mae_median <= 0:
        return float("nan")
    return cr_std_median / method_mae_median


def calibration_coverage(
    pred_std: np.ndarray,
    errors: np.ndarray,
    levels: tuple[float, ...] = (0.68, 0.90, 0.95),
) -> dict[str, float]:
    """
    Calibration: what fraction of |errors| < z_level * pred_std?

    Perfectly calibrated: coverage[level] ≈ level.
    Over-confident: coverage < level (errors larger than predicted).
    Under-confident: coverage > level.

    pred_std: (N,)  predicted posterior std
    errors:   (N,)  |pred - truth| per sample
    """
    import scipy.stats as stats
    result = {}
    for level in levels:
        z = float(stats.norm.ppf((1 + level) / 2))
        result[str(level)] = float(np.mean(errors < z * pred_std))
    return result


# ── data loaders ─────────────────────────────────────────────────────────────

def load_test_csv(
    data_dir: str,
    n_freq: int = 100,
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """
    Load static test set from mixed_distribution_v2.

    Returns dict with keys: Zr, Zi, omega, params_raw (N,5), params_id (N,5),
    frequencies (F,), n.
    """
    data_path = Path(data_dir)
    with open(data_path / "metadata.json") as f:
        meta = json.load(f)

    df = pd.read_csv(data_path / "test.csv")
    if n_samples and n_samples < len(df):
        df = df.sample(n_samples, random_state=seed)

    freq_min = meta["frequencies"]["min"]
    freq_max = meta["frequencies"]["max"]
    frequencies = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freq)
    omega = 2 * np.pi * frequencies

    Zr = df[[f"Z_real_{i}" for i in range(n_freq)]].values.astype(np.float32)
    Zi = df[[f"Z_imag_{i}" for i in range(n_freq)]].values.astype(np.float32)

    params_raw_lin = df[PARAMS_RAW].values.astype(np.float64)
    params_log10   = np.log10(np.maximum(params_raw_lin, 1e-30))
    params_id      = to_identifiable(params_log10)

    return {
        "Zr": Zr, "Zi": Zi, "omega": omega, "frequencies": frequencies,
        "params_raw": params_log10, "params_id": params_id, "n": len(df),
    }


def load_temporal_hdf5(
    h5_path: str,
    n_traj: int = 50,
    pathology: Optional[str] = None,
    seed: int = 42,
) -> dict:
    """
    Load trajectories from the temporal HDF5 dataset.

    Returns dict with per-pathology lists of:
      params_log10 (T, 5), derived_log10 (T, 5), Zr (T, F), Zi (T, F), dt_minutes.
    """
    rng = np.random.default_rng(seed)

    with h5py.File(h5_path, "r") as f:
        pathologies_raw = f["pathology"][:]
        frequencies = f["frequencies"][:]

        if pathology:
            targets = [pathology.encode() if isinstance(pathology, str) else pathology]
        else:
            targets = [p.encode() for p in PATHOLOGIES]

        data = {p: [] for p in PATHOLOGIES}

        for tgt in targets:
            idxs = np.where(pathologies_raw == tgt)[0]
            chosen = rng.choice(idxs, min(n_traj, len(idxs)), replace=False)

            pname = tgt.decode() if isinstance(tgt, bytes) else tgt
            for idx in chosen:
                params = f["params_log10"][idx]         # (T, 5)
                derived = f["derived_log10"][idx]       # (T, 5)
                Zr = f["impedance_real"][idx]           # (T, F)
                Zi = f["impedance_imag"][idx]           # (T, F)
                t_min = f["time_minutes"][idx]          # (T,)
                dt = float(np.median(np.diff(t_min)))  # minutes per step

                data[pname].append({
                    "params_log10": params.astype(np.float64),
                    "derived_log10": derived.astype(np.float64),
                    "Zr": Zr.astype(np.float64),
                    "Zi": Zi.astype(np.float64),
                    "dt": dt,
                    "T": params.shape[0],
                })

    omega = 2 * np.pi * frequencies
    return {"trajectories": data, "omega": omega, "frequencies": frequencies}


# ── model loading & inference ────────────────────────────────────────────────

def load_transformer(model_dir: str):
    """Load FisherAwareTransformer from checkpoint directory."""
    from src.models.fisher_transformer import FisherAwareTransformer, TransformerConfig

    ckpt = torch.load(Path(model_dir) / "best_model.pt", map_location="cpu", weights_only=False)
    cfg  = dict(ckpt["config"])
    cfg.setdefault("use_drt", False)
    model = FisherAwareTransformer(TransformerConfig(**cfg))
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    model.eval()
    is_id = ckpt.get("param_space", "degenerate") == "identifiable"
    return model, is_id, ckpt


def run_mdn_batch(
    model,
    Zr: np.ndarray,
    Zi: np.ndarray,
    omega: np.ndarray,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run MDN forward pass in batches.

    Returns:
        pred_mean (N, 5) mixture mean in log10 identifiable space
        pred_std  (N, 5) mixture std (sqrt of mixture variance)
    """
    from tqdm import tqdm

    N = len(Zr)
    log_omega = torch.tensor(np.log10(omega), dtype=torch.float32)
    means_all, stds_all = [], []

    for start in tqdm(range(0, N, batch_size), desc="MDN inference", leave=False):
        Zr_b = torch.tensor(Zr[start:start+batch_size], dtype=torch.float32)
        Zi_b = torch.tensor(Zi[start:start+batch_size], dtype=torch.float32)
        lo_b = log_omega.unsqueeze(0).expand(len(Zr_b), -1)

        with torch.no_grad():
            out = model(Zr_b, Zi_b, lo_b)

        means_b   = out["means"].cpu().numpy()    # (B, K, 5)
        weights_b = out["weights"].cpu().numpy()  # (B, K)
        weights_b = np.clip(weights_b, 0, None)
        weights_b /= weights_b.sum(axis=1, keepdims=True) + 1e-12

        mu_mix = (means_b * weights_b[:, :, None]).sum(axis=1)  # (B, 5)

        # Mixture variance = E[var] + Var[mean]
        var_mix = np.zeros_like(mu_mix)
        for k in range(means_b.shape[1]):
            # per-component variance from covariance diagonal
            if "covs" in out:
                cov_k = out["covs"][:, k].cpu().numpy()  # (B, 5, 5)
                var_k = np.diagonal(cov_k, axis1=1, axis2=2)  # (B, 5)
            else:
                var_k = np.full_like(mu_mix, 0.01)
            d_k = means_b[:, k] - mu_mix
            var_mix += weights_b[:, k:k+1] * (var_k + d_k ** 2)

        means_all.append(mu_mix)
        stds_all.append(np.sqrt(np.maximum(var_mix, 1e-10)))

    return np.concatenate(means_all, axis=0), np.concatenate(stds_all, axis=0)


# ── ECM baseline ──────────────────────────────────────────────────────────────

def run_ecm_batch(
    Zr: np.ndarray,
    Zi: np.ndarray,
    omega: np.ndarray,
    n_restarts: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run ECM (L-BFGS-B) on a batch of spectra.

    Returns:
        pred_id (N, 5) predictions in log10 identifiable space (NaN on failure)
        success  (N,)  bool mask
    """
    from src.baselines.deterministic_fit import DeterministicPhysicsFit
    from tqdm import tqdm

    N = len(Zr)
    frequencies = omega / (2 * np.pi)
    fitter = DeterministicPhysicsFit(n_restarts=n_restarts)
    pred_id = np.full((N, 5), np.nan)
    success = np.zeros(N, dtype=bool)

    for i in tqdm(range(N), desc="ECM", leave=False):
        result = fitter.fit(Zr[i], Zi[i], frequencies)
        if result is not None:
            raw_log10 = np.array([
                np.log10(result["Ra"]), np.log10(result["Rb"]),
                np.log10(result["Ca"]), np.log10(result["Cb"]),
                np.log10(result["Rsh"]),
            ])
            pred_id[i] = to_identifiable(raw_log10)
            success[i] = True

    return pred_id, success


# ── CR bounds ────────────────────────────────────────────────────────────────

def compute_cr_bounds_subsample(
    params_log10: np.ndarray,
    omega: np.ndarray,
    sigma_noise: float = 10.0,
    n_samples: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute CR bounds (std, log10 identifiable) on a random subsample.

    Returns:
        cr_std   (N, 5) — NaN for unsampled rows
        sampled_idx (n_samples,) — indices that were computed
    """
    from src.physics.eis_fisher import compute_cr_bounds_identifiable
    from tqdm import tqdm

    N = len(params_log10)
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, min(N, n_samples), replace=False)

    cr_var = np.full((N, 5), np.nan)
    for i in tqdm(idx, desc="CR bounds", leave=False):
        try:
            var = compute_cr_bounds_identifiable(
                params_log10[i:i+1], omega, sigma_noise,
            )
            cr_var[i] = var[0]
        except Exception:
            pass

    return np.sqrt(np.maximum(cr_var, 0.0)), idx


# ── identifiability stratifiers ───────────────────────────────────────────────

def identifiability_strata(params_log10: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute per-sample identifiability features for stratification.

    Returns:
        tau_ratio  = (tau_big - tau_small) / (tau_big + tau_small) ∈ [0, 1]
                     near 0: both arcs merge → both τ unreadable
                     near 1: arcs well-separated → both τ readable
        rsh_ratio  = Rsh / (Ra + Rb)
                     >> 1: Rsh dominates → TER ≈ Ra+Rb, Rsh unreadable
                     << 1: RC branches dominate → TER ≈ Rsh, easily read
                     near 1: TER hardest to decompose
        identifiable_region: "easy" (tau_ratio > 0.5 AND rsh_ratio > 2)
                              "hard" otherwise
    """
    p = np.atleast_2d(params_log10)
    Ra, Rb, Ca, Cb, Rsh = [10.0 ** p[:, i] for i in range(5)]
    tau_a = Ra * Ca; tau_b = Rb * Cb
    tau_big   = np.maximum(tau_a, tau_b)
    tau_small = np.minimum(tau_a, tau_b)

    tau_ratio = (tau_big - tau_small) / np.maximum(tau_big + tau_small, 1e-30)
    rsh_ratio = Rsh / np.maximum(Ra + Rb, 1e-30)

    easy = (tau_ratio > 0.5) & (rsh_ratio > 2.0)
    region = np.where(easy, "easy", "hard")

    return {"tau_ratio": tau_ratio, "rsh_ratio": rsh_ratio, "region": region}


# ── result IO ─────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent.parent / "results" / "eval"


def save_results(name: str, data: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / f"{name}.json"
    with open(p, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    return p


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")
