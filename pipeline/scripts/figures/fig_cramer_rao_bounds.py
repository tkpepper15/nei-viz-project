#!/usr/bin/env python3
"""
12l - Cramér-Rao Bound Analysis: Theoretical Limits of EIS Parameter Estimation

Answers the question: how close is each pipeline stage to the statistical optimum?

MATHEMATICAL FOUNDATION
-----------------------
For an unbiased estimator of θ, the Cramér-Rao Lower Bound (CRLB) is:

    Var(θ̂_i) ≥ [F(θ)⁻¹]_{ii}

where the Fisher Information Matrix under the relative noise model is:

    F(θ) = J^T Ω(θ)⁻¹ J

and J = ∂Z/∂log₁₀(θ) is the Jacobian in log10-parameter space
(already computed this way by compute_jacobian_autodiff), and

    Ω(θ)_{kk} = (α|Z_k(θ)|)² + ε²     [relative noise: 2% of |Z| + 0.5Ω floor]

This gives CRLB_std_i = sqrt([F⁻¹]_{ii}) in log10-decade units.

KEY INSIGHT ON DEGENERACY
--------------------------
The circuit has near-degenerate directions (Ra↔Rb, Ca↔Cb swaps). The FIM will
have two near-zero eigenvalues corresponding to these directions. The CRLB
diverges in the degenerate subspace but is finite for identifiable combinations:

    Highly identifiable  →  small CRLB  →  top FIM eigenvalues
    Nearly degenerate    →  large CRLB  →  near-zero FIM eigenvalues

Delta-method CRLB for derived quantities g(θ) = log₁₀(f(θ)):
    Var(ĝ) ≥ (∂g/∂θ_log10)^T F(θ)⁻¹ (∂g/∂θ_log10)

This gives tight bounds for TER, τ_b, τ_a, TEC — the physically observable
quantities — even when individual Ra, Rb have large CRLB.

EFFICIENCY METRIC
-----------------
Fisher efficiency (for each stage, parameter i, timepoint t):

    η_i(t) = CRLB_std_i(θ_t) / σ̂_i(t)

    η → 1:  near CR-optimal (best possible for unbiased estimator)
    η → 0:  severely inefficient
    η > 1:  impossible for unbiased estimator → indicates BIAS

OUTPUTS
-------
1. crlb_tracking.png     — tracking figure with theoretical CRLB band overlaid
2. crlb_efficiency.png   — efficiency η(t) per stage per parameter
3. crlb_scenario.png     — FIM eigenspectrum + CRLB across physiological scenarios

Usage:
    cd pipeline && python fig_cramer_rao_bounds.py
    python fig_cramer_rao_bounds.py --recompute   # re-run CRLB computation
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm

sys.path.insert(0, ".")
from src.physics.eis_fisher import (
    compute_impedance,
    compute_jacobian_autodiff,
    compute_derived_from_log10,
)


# ============================================================
#  Style
# ============================================================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "mathtext.default": "regular",
})

BLACK           = "#1a1a1a"
GOLD            = "#c8962e"
DL_COLOR        = "#2166ac"
STAGE_A_COLOR   = "#7b2d8b"
STAGE_ABL_COLOR = "#15803d"
CRLB_COLOR      = "#6b7280"   # gray for theoretical bound
ATP_COLOR       = "#f4a7b9"

PARAM_NAMES   = ["Ra", "Rb", "Ca", "Cb", "Rsh"]
PARAM_LABELS  = [r"$R_a$", r"$R_b$", r"$C_a$", r"$C_b$", r"$R_{sh}$"]
DERIVED_NAMES = ["TER", "TEC", "tau_a", "tau_b", "Ra_plus_Rb"]
DERIVED_LABELS = [r"TER", r"TEC", r"$\tau_a$", r"$\tau_b$", r"$R_a+R_b$"]

# ============================================================
#  Sample configuration — mirrors 12e/12k exactly
# ============================================================
SAMPLES = {
    "Sample 1": {
        "meas_id":      "20211021_183356",
        "interval_min": 3.5,
        "atp_span":     (7, 11),
        "donor":        "Donor A",
    },
    "Sample 2": {
        "meas_id":      "20211022_171622",
        "interval_min": 2.4,
        "atp_span":     (10, 15),
        "donor":        "Donor A",
    },
    "Sample 3": {
        "meas_id":      "20211014_165514",
        "interval_min": 1.7,
        "atp_span":     (9, 12),
        "donor":        "Donor B",
    },
}

CSV_PATH      = "../docs/fit results.csv"
CROSS_SECTION = 0.11409
N_FREQ        = 100
FREQ_MIN_HZ   = 0.1
FREQ_MAX_HZ   = 1e6
RSH_MODEL_MAX = 50000.0

# Relative noise model parameters (must match BL refinement settings)
BL_ALPHA         = 0.02    # 2% relative noise
BL_EPSILON_FLOOR = 0.5     # 0.5 Ohm absolute floor

ROWS = [
    {"param": 2, "label": r"$C_a$  ($\mu$F cm$^{-2}$)",    "buffer": 1.00},
    {"param": 3, "label": r"$C_b$  ($\mu$F cm$^{-2}$)",    "buffer": 1.20},
    {"param": 0, "label": r"$R_a$  (k$\Omega$ cm$^2$)",    "buffer": 0.70},
    {"param": 1, "label": r"$R_b$  (k$\Omega$ cm$^2$)",    "buffer": 0.70},
    {"param": 4, "label": r"$R_{sh}$  (k$\Omega$ cm$^2$)", "buffer": 0.70},
]


# ============================================================
#  Unit conversions
# ============================================================

def log10_to_display_nd(log10_arr):
    vals = 10 ** log10_arr
    out = np.empty_like(vals)
    out[:, 0] = vals[:, 0] / 1000
    out[:, 1] = vals[:, 1] / 1000
    out[:, 2] = vals[:, 2] * 1e6
    out[:, 3] = vals[:, 3] * 1e6
    out[:, 4] = vals[:, 4] / 1000
    return out


def display_1d_to_log10(disp_val, param_idx):
    """Convert single display-unit value to log10 model units."""
    if param_idx in (0, 1, 4):   # R values: kOhm*cm2 -> Ohm -> log10
        return np.log10(disp_val * 1000)
    else:                         # C values: uF/cm2 -> F -> log10
        return np.log10(disp_val / 1e6)


def log10_std_from_band(lo_disp, hi_disp):
    """
    Recover log10-space standard deviation from ±1σ display-unit bands.

    Since lo = 10^(mean - std) * conv and hi = 10^(mean + std) * conv:
        std_log10 = 0.5 * log10(hi / lo)

    The conversion factor cancels exactly.
    """
    ratio = np.maximum(hi_disp / np.maximum(lo_disp, 1e-30), 1.0)
    return 0.5 * np.log10(ratio)   # (n_t, 5)


def log10_std_from_samples(samples_disp_list, param_idx):
    """
    Compute log10-space std from list of display-unit sample arrays.
    Conversion constant drops out of std(log10(x * const)) = std(log10(x)).
    """
    stds = []
    for samples in samples_disp_list:
        if samples is None or len(samples) < 2:
            stds.append(np.nan)
            continue
        arr = np.asarray(samples, dtype=np.float64)
        vals = arr[:, param_idx]
        valid = vals[vals > 0]
        if len(valid) < 2:
            stds.append(np.nan)
        else:
            stds.append(np.std(np.log10(valid)))
    return np.array(stds)


# ============================================================
#  CRLB computation
# ============================================================

def compute_crlb_at_params(
    params_log10: np.ndarray,
    omega_np: np.ndarray,
    alpha: float = BL_ALPHA,
    epsilon_floor: float = BL_EPSILON_FLOOR,
) -> dict:
    """
    Compute the Cramér-Rao Lower Bound at given log10 parameters.

    Uses the SAME relative noise model as Black-Litterman refinement:
        Ω_{kk} = (α|Z_k|)² + ε²

    The Jacobian J = ∂Z/∂log₁₀(θ) is already in log10 space
    (compute_jacobian_autodiff differentiates through 10**params_log10),
    so FIM = J^T Ω⁻¹ J gives variance directly in log10-decade units.

    Returns dict with:
        crlb_std    : (5,) CRLB std dev in log10 decades for base params
        fim         : (5, 5) Fisher information matrix
        fim_inv     : (5, 5) CRLB matrix
        eigenvalues : (5,) FIM eigenvalues descending (identifiability spectrum)
        eigenvectors: (5, 5) FIM eigenvectors (columns = principal directions)
        condition_number: max_eigval / max(min_eigval, 1e-12)
        n_identifiable: number of eigenvalues > 1e-4 relative threshold
        crlb_derived: dict of CRLB std for TER, TEC, tau_a, tau_b, Ra+Rb
    """
    params_t = torch.tensor(params_log10, dtype=torch.float32).unsqueeze(0)  # (1, 5)
    omega_t  = torch.tensor(omega_np, dtype=torch.float32)

    # Step 1: predicted impedance at true params (for noise model)
    with torch.no_grad():
        Ra, Rb, Ca, Cb, Rsh = [float(10 ** v) for v in params_log10]
        Z_r, Z_i = compute_impedance(
            torch.tensor([Ra]), torch.tensor([Rb]), torch.tensor([Ca]),
            torch.tensor([Cb]), torch.tensor([Rsh]), omega_t,
        )

    # Step 2: relative noise covariance Ω (diagonal, 2n_freq entries)
    Z_mag_sq = Z_r ** 2 + Z_i ** 2                          # (1, n_freq)
    noise_var = (alpha ** 2) * Z_mag_sq + epsilon_floor ** 2  # (1, n_freq)
    omega_diag = torch.cat([noise_var, noise_var], dim=1)[0].numpy()  # (2n_freq,)

    # Step 3: Jacobian in log10 space: ∂[Z_real, Z_imag] / ∂log10(θ)
    with torch.enable_grad():
        J_t = compute_jacobian_autodiff(params_t, omega_t)   # (1, 2n_freq, 5)
    J = J_t[0].detach().numpy()   # (2n_freq, 5)

    # Step 4: FIM = J^T Ω⁻¹ J
    inv_omega = 1.0 / (omega_diag + 1e-30)    # (2n_freq,)
    FIM = J.T @ (inv_omega[:, np.newaxis] * J)  # (5, 5)

    # Symmetrize (numerical safety)
    FIM = 0.5 * (FIM + FIM.T)

    # Step 5: Invert via eigendecomposition (handles near-degeneracy gracefully)
    eigvals_raw, eigvecs = np.linalg.eigh(FIM)   # ascending order
    eigvals_desc = eigvals_raw[::-1].copy()
    eigvecs_desc  = eigvecs[:, ::-1].copy()

    # Relative threshold: eigenvalue / max_eigenvalue < threshold → degenerate
    max_eig = max(eigvals_desc[0], 1e-30)
    rel_threshold = 1e-6
    n_identifiable = int(np.sum(eigvals_desc / max_eig > rel_threshold))

    # Pseudo-inverse: only invert eigenvalues above threshold
    eigvals_inv = np.where(eigvals_raw / max_eig > rel_threshold, 1.0 / np.maximum(eigvals_raw, 1e-30), 0.0)
    FIM_inv = eigvecs @ np.diag(eigvals_inv) @ eigvecs.T

    # CRLB std in log10 decades
    crlb_var = np.maximum(np.diag(FIM_inv), 0.0)
    crlb_std = np.sqrt(crlb_var)

    condition_number = eigvals_desc[0] / max(np.abs(eigvals_desc[-1]), 1e-30)

    # Step 6: delta-method CRLB for derived quantities
    crlb_derived = _compute_crlb_derived(params_t, FIM_inv, omega_t)

    return {
        "crlb_std":        crlb_std,
        "fim":             FIM,
        "fim_inv":         FIM_inv,
        "eigenvalues":     eigvals_desc,
        "eigenvectors":    eigvecs_desc,
        "condition_number": condition_number,
        "n_identifiable":  n_identifiable,
        "crlb_derived":    crlb_derived,
    }


def _compute_crlb_derived(params_log10_t: torch.Tensor, FIM_inv: np.ndarray, omega_t: torch.Tensor) -> dict:
    """
    Delta-method CRLB for derived quantities.

    For g(θ) = log₁₀(f(θ)):
        CRLB_g = sqrt( ∇g^T F⁻¹ ∇g )
    where ∇g = ∂log₁₀(g)/∂log₁₀(θ)

    This correctly propagates log10-space uncertainty into derived quantities.
    """
    p = params_log10_t.clone().detach().requires_grad_(True)
    derived = compute_derived_from_log10(p)   # dict of (1,) log10 tensors

    crlb_derived = {}
    for name, val in derived.items():
        if p.grad is not None:
            p.grad.zero_()
        val.sum().backward(retain_graph=True)
        g = p.grad[0].detach().numpy().copy()   # (5,) = ∂log10(quantity)/∂log10(θ)
        crlb_var_g = float(g @ FIM_inv @ g)
        crlb_derived[name] = float(np.sqrt(max(crlb_var_g, 0.0)))

    if p.grad is not None:
        p.grad.zero_()
    return crlb_derived


def compute_crlb_trajectory(truth_log10_arr: np.ndarray, omega_np: np.ndarray) -> list:
    """
    Compute CRLB at each timepoint along a parameter trajectory.

    Returns list[n_t] of result dicts from compute_crlb_at_params.
    """
    results = []
    for t, params_log10 in enumerate(truth_log10_arr):
        res = compute_crlb_at_params(params_log10, omega_np)
        results.append(res)
        if (t + 1) % 5 == 0:
            print(f"    CRLB: {t + 1}/{len(truth_log10_arr)} done  "
                  f"(cond={res['condition_number']:.1e}, "
                  f"ident={res['n_identifiable']}/5)")
    return results


# ============================================================
#  Efficiency computation
# ============================================================

def compute_efficiency(crlb_std_arr: np.ndarray, sigma_arr: np.ndarray) -> np.ndarray:
    """
    η(t) = CRLB_std(t) / σ̂(t)    in [0, ∞)

    η = 1 → CR-optimal (best possible for unbiased estimator)
    η < 1 → below CRLB: either inefficient (normal) or biased
    η > 1 → tighter than CRLB: BIASED estimator (flag this)
    """
    return np.where(sigma_arr > 1e-10, crlb_std_arr / sigma_arr, np.nan)


# ============================================================
#  Data helpers (mirrors 12e/12k)
# ============================================================

def load_sample(df, meas_id):
    sub = (df[(df["chamber"] == "Ussing") & (df["meas_ID"] == meas_id)]
           .sort_values("meas_idx").reset_index(drop=True))
    return sub


def extract_3p_params(row, xs=CROSS_SECTION):
    return (float(row["pg_absZr_3"]) * xs, float(row["pg_absZr_5"]) * xs,
            float(row["pg_absZr_4"]) / xs, float(row["pg_absZr_6"]) / xs,
            float(row["pg_absZr_7"]) * xs)


def params_to_log10_arr(rows_params):
    return np.array([[np.log10(max(p, 1e-12)) for p in row] for row in rows_params])


# ============================================================
#  Cache I/O
# ============================================================

def _12e_cache_path(sname, cache_dir="results/poster/cache_12e"):
    return Path(cache_dir) / f"cache_{sname.replace(' ', '_')}.npz"


def _12k_cache_path(sname, cache_dir="results/poster/cache_12k"):
    return Path(cache_dir) / f"cache_{sname.replace(' ', '_')}.npz"


def _crlb_cache_path(sname, cache_dir):
    return Path(cache_dir) / f"cache_{sname.replace(' ', '_')}.npz"


def load_12e_cache(sname, cache_dir="results/poster/cache_12e"):
    p = _12e_cache_path(sname, cache_dir)
    if not p.exists():
        raise FileNotFoundError(f"12e cache missing: {p}\nRun 12e_real_data_tracking.py first.")
    d = np.load(p, allow_pickle=True)
    return {
        "n_t":        int(d["n_t"][0]),
        "truth_disp": d["truth_disp"],
        "dl_disp":    list(d["dl_disp"]),
        "ecm_disp":   list(d["ecm_disp"]),
    }


def load_12k_cache(sname, cache_dir="results/poster/cache_12k"):
    p = _12k_cache_path(sname, cache_dir)
    if not p.exists():
        raise FileNotFoundError(f"12k cache missing: {p}\nRun 12k_stage_ablation.py first.")
    d = np.load(p, allow_pickle=True)
    return {
        "n_t":           int(d["n_t"][0]),
        "truth_disp":    d["truth_disp"],
        "stageA_mean":   d["stageA_mean"],
        "stageA_lo":     d["stageA_lo"],
        "stageA_hi":     d["stageA_hi"],
        "stageABL_mean": d["stageABL_mean"],
        "stageABL_lo":   d["stageABL_lo"],
        "stageABL_hi":   d["stageABL_hi"],
    }


def save_crlb_cache(sname, data, cache_dir):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez(
        _crlb_cache_path(sname, cache_dir),
        crlb_std=data["crlb_std"],             # (n_t, 5)
        eigenvalues=data["eigenvalues"],        # (n_t, 5)
        condition_numbers=data["condition_numbers"],   # (n_t,)
        n_identifiable=data["n_identifiable"],  # (n_t,)
        crlb_derived_ter=data["crlb_derived_ter"],     # (n_t,)
        crlb_derived_tau_b=data["crlb_derived_tau_b"],
        crlb_derived_tau_a=data["crlb_derived_tau_a"],
        crlb_derived_tec=data["crlb_derived_tec"],
        truth_log10=data["truth_log10"],        # (n_t, 5)
    )


def load_crlb_cache(sname, cache_dir):
    p = _crlb_cache_path(sname, cache_dir)
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return {
        "crlb_std":          d["crlb_std"],
        "eigenvalues":       d["eigenvalues"],
        "condition_numbers": d["condition_numbers"],
        "n_identifiable":    d["n_identifiable"],
        "crlb_derived_ter":  d["crlb_derived_ter"],
        "crlb_derived_tau_b": d["crlb_derived_tau_b"],
        "crlb_derived_tau_a": d["crlb_derived_tau_a"],
        "crlb_derived_tec":  d["crlb_derived_tec"],
        "truth_log10":       d["truth_log10"],
    }


# ============================================================
#  Figure 1: CRLB tracking
# ============================================================

def _smart_tick_formatter(ax, y_lo, y_hi):
    span = y_hi - y_lo
    fmt = "%.2f" if span < 0.5 else ("%.1f" if span < 5 else "%.0f")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(fmt))


def make_crlb_tracking_figure(all_results, out_dir):
    """
    Same layout as stage_ablation.png but adds CRLB band as theoretical floor.

    CRLB band (gray hatched) shows ±1σ_CRLB around ground truth — the minimum
    width any unbiased estimator's uncertainty band can achieve.
    """
    n_rows = len(SAMPLES)
    n_cols = len(ROWS)
    sample_names = list(SAMPLES.keys())

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(34, 16),
        gridspec_kw={"hspace": 0.18, "wspace": 0.32,
                     "top": 0.86, "bottom": 0.22,
                     "left": 0.08, "right": 0.99},
    )

    last_dl_n = 0

    for row_idx, sname in enumerate(sample_names):
        res = all_results[sname]
        cfg = SAMPLES[sname]
        n_t   = res["n_t"]
        t_min = np.arange(n_t) * cfg["interval_min"]
        atp_lo = cfg["atp_span"][0] * cfg["interval_min"]
        atp_hi = cfg["atp_span"][1] * cfg["interval_min"]

        truth_disp     = res["truth_disp"]
        stageA_mean    = res["stageA_mean"]
        stageA_lo      = res["stageA_lo"]
        stageA_hi      = res["stageA_hi"]
        stageABL_mean  = res["stageABL_mean"]
        stageABL_lo    = res["stageABL_lo"]
        stageABL_hi    = res["stageABL_hi"]
        dl_disp        = res["dl_disp"]
        ecm_disp       = res["ecm_disp"]
        crlb_std_log10 = res["crlb_std"]    # (n_t, 5) in log10 decades

        if dl_disp and len(dl_disp[0]) > 0:
            last_dl_n = dl_disp[0].shape[0]

        for col_idx, row_def in enumerate(ROWS):
            ax  = axes[row_idx, col_idx]
            p   = row_def["param"]
            buf = row_def["buffer"]

            if row_idx == 0:
                ax.set_title(row_def["label"], fontsize=24, fontweight="bold", pad=16)

            ax.axvspan(atp_lo, atp_hi, color=ATP_COLOR, alpha=0.38, zorder=0)

            truth_col = truth_disp[:, p]
            t_lo   = truth_col.min()
            t_hi   = truth_col.max()
            t_span = max(t_hi - t_lo, t_hi * 0.10)
            y_lo   = max(0.0, t_lo - buf * t_span)
            y_hi   = t_hi + buf * t_span

            def clip(arr):
                return np.clip(arr, y_lo, y_hi)

            # CRLB band: ±1σ_CRLB around ground truth in display units
            truth_log10_col = np.log10(truth_col * (1000 if p in (0, 1, 4) else 1e-6))
            crlb_s = crlb_std_log10[:, p]   # (n_t,) in log10 decades

            # lo/hi in log10 space, then convert to display
            # Same conversion as truth: log10 -> absolute -> display
            log10_lo = truth_log10_col - crlb_s
            log10_hi = truth_log10_col + crlb_s
            # Convert log10 model units to display
            if p in (0, 1, 4):   # resistance: Ohm -> kOhm
                crlb_lo_disp = 10 ** log10_lo / 1000
                crlb_hi_disp = 10 ** log10_hi / 1000
            else:                 # capacitance: F -> uF
                crlb_lo_disp = 10 ** log10_lo * 1e6
                crlb_hi_disp = 10 ** log10_hi * 1e6

            ax.fill_between(
                t_min, clip(crlb_lo_disp), clip(crlb_hi_disp),
                color=CRLB_COLOR, alpha=0.35, linewidth=0,
                hatch="////", edgecolor=CRLB_COLOR, facecolor="none",
                zorder=1, label="CRLB" if (row_idx == 0 and col_idx == 0) else "",
            )
            # Solid CRLB boundary lines
            ax.plot(t_min, clip(crlb_lo_disp),
                    color=CRLB_COLOR, linewidth=1.0, linestyle="-", alpha=0.6, zorder=2)
            ax.plot(t_min, clip(crlb_hi_disp),
                    color=CRLB_COLOR, linewidth=1.0, linestyle="-", alpha=0.6, zorder=2)

            # ECM
            ecm_vals = [e[:, p] if len(e) > 0 else np.empty(0) for e in ecm_disp]
            ecm_med  = np.array([np.nanmedian(v) if len(v) > 0 else np.nan for v in ecm_vals])
            ecm_q25  = np.array([np.nanpercentile(v, 25) if len(v) > 0 else np.nan for v in ecm_vals])
            ecm_q75  = np.array([np.nanpercentile(v, 75) if len(v) > 0 else np.nan for v in ecm_vals])
            valid_e  = ~np.isnan(ecm_med)
            if valid_e.any():
                ax.fill_between(
                    t_min[valid_e], clip(ecm_q25[valid_e]), clip(ecm_q75[valid_e]),
                    color=GOLD, alpha=0.15, linewidth=0, zorder=3)
                ax.plot(t_min[valid_e], clip(ecm_med[valid_e]),
                        color=GOLD, linewidth=1.8, linestyle="--", alpha=0.6, zorder=4)

            # Stage A
            ax.fill_between(
                t_min, clip(stageA_lo[:, p]), clip(stageA_hi[:, p]),
                color=STAGE_A_COLOR, alpha=0.15, linewidth=0, zorder=5)
            ax.plot(t_min, clip(stageA_mean[:, p]),
                    color=STAGE_A_COLOR, linewidth=2.0, linestyle=":", zorder=6)

            # Stage A+BL
            ax.fill_between(
                t_min, clip(stageABL_lo[:, p]), clip(stageABL_hi[:, p]),
                color=STAGE_ABL_COLOR, alpha=0.15, linewidth=0, zorder=7)
            ax.plot(t_min, clip(stageABL_mean[:, p]),
                    color=STAGE_ABL_COLOR, linewidth=2.2, linestyle="--", zorder=8)

            # Full DL+KF
            dl_vals = [d[:, p] for d in dl_disp]
            dl_med  = np.array([np.nanmedian(v) for v in dl_vals])
            dl_q25  = np.array([np.nanpercentile(v, 25) for v in dl_vals])
            dl_q75  = np.array([np.nanpercentile(v, 75) for v in dl_vals])
            ax.fill_between(t_min, clip(dl_q25), clip(dl_q75),
                            color=DL_COLOR, alpha=0.20, linewidth=0, zorder=9)
            ax.plot(t_min, clip(dl_med),
                    color=DL_COLOR, linewidth=3.0, linestyle="-", zorder=10,
                    path_effects=[pe.withStroke(linewidth=5, foreground="white")])

            # Ground truth
            ax.plot(t_min, truth_col,
                    color=BLACK, linewidth=2.0, linestyle="-", zorder=11)
            ax.scatter(t_min, truth_col, s=45, marker="s",
                       facecolors="white", edgecolors=BLACK, linewidths=1.8, zorder=12)

            ax.set_xlim(-1, t_min[-1] + cfg["interval_min"])
            ax.set_ylim(y_lo, y_hi)
            _smart_tick_formatter(ax, y_lo, y_hi)
            ax.grid(True, alpha=0.14, linewidth=0.6, color="#bbbbbb")
            ax.tick_params(labelsize=16, which="both", length=5)
            ax.spines[["top", "right"]].set_visible(False)
            for spine in ax.spines.values():
                spine.set_linewidth(0.9)

            if col_idx == 0:
                ax.set_ylabel(f"{sname}  ({cfg['donor']})\n"
                              f"n={n_t},  {cfg['interval_min']:.1f} min/meas",
                              fontsize=22, fontweight="bold", labelpad=12)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Time (min)", fontsize=19)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)

    legend_handles = [
        Line2D([0], [0], color=BLACK, linewidth=2.0, marker="s", markersize=7,
               markerfacecolor="white", markeredgewidth=1.6, label="3P-EIS (ground truth)"),
        mpatches.Patch(facecolor="none", hatch="////", edgecolor=CRLB_COLOR, linewidth=1.2,
                       label=r"CRLB $\pm 1\sigma$  (theoretical minimum width)"),
        Line2D([0], [0], color=STAGE_A_COLOR, linewidth=2.0, linestyle=":",
               label="Stage A: Transformer only"),
        Line2D([0], [0], color=STAGE_ABL_COLOR, linewidth=2.2, linestyle="--",
               label="Stage A+C: Transformer + BL"),
        Line2D([0], [0], color=DL_COLOR, linewidth=3.0,
               path_effects=[pe.withStroke(linewidth=5, foreground="white")],
               label=f"Full A+C+KF  (n={last_dl_n} MCMC samples)"),
        Line2D([0], [0], color=GOLD, linewidth=1.8, linestyle="--", alpha=0.7,
               label="ECM  (reference)"),
        mpatches.Patch(facecolor=ATP_COLOR, alpha=0.65, label="100 µM ATP"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=14,
               framealpha=0.97, edgecolor="#cccccc", handlelength=3.0,
               handletextpad=0.9, labelspacing=0.8, columnspacing=2.0,
               bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        "CRLB-Bounded Tracking: Theoretical Minimum vs Pipeline Stages\n"
        r"Gray hatching = $\pm 1\sigma$ Cramér-Rao Bound (any unbiased estimator cannot do better)",
        fontsize=23, fontweight="bold", y=0.975)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "crlb_tracking.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {path}")
    plt.close(fig)


# ============================================================
#  Figure 2: Efficiency over time
# ============================================================

def make_efficiency_figure(all_results, out_dir):
    """
    η(t) = CRLB_std(t) / σ̂_stage(t) for each stage, parameter, sample.

    η = 1 (dashed) = CR-optimal.
    η < 1 = inefficient (normal — estimator variance > CRLB).
    η > 1 = tighter than CRLB → BIASED (highlight in red).

    Displayed as percentages (η × 100%) on log scale for clarity.
    """
    n_rows = len(SAMPLES)
    n_cols = len(ROWS)
    sample_names = list(SAMPLES.keys())

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(34, 14),
        gridspec_kw={"hspace": 0.22, "wspace": 0.32,
                     "top": 0.88, "bottom": 0.20,
                     "left": 0.08, "right": 0.99},
    )

    for row_idx, sname in enumerate(sample_names):
        res = all_results[sname]
        cfg = SAMPLES[sname]
        n_t   = res["n_t"]
        t_min = np.arange(n_t) * cfg["interval_min"]
        atp_lo = cfg["atp_span"][0] * cfg["interval_min"]
        atp_hi = cfg["atp_span"][1] * cfg["interval_min"]

        crlb_std_log10 = res["crlb_std"]          # (n_t, 5)
        stageA_lo  = res["stageA_lo"]
        stageA_hi  = res["stageA_hi"]
        stageABL_lo = res["stageABL_lo"]
        stageABL_hi = res["stageABL_hi"]
        dl_disp    = res["dl_disp"]
        ecm_disp   = res["ecm_disp"]

        # σ in log10 space for each stage
        sigma_stageA   = log10_std_from_band(stageA_lo, stageA_hi)      # (n_t, 5)
        sigma_stageABL = log10_std_from_band(stageABL_lo, stageABL_hi)  # (n_t, 5)

        for col_idx, row_def in enumerate(ROWS):
            ax = axes[row_idx, col_idx]
            p  = row_def["param"]

            if row_idx == 0:
                ax.set_title(row_def["label"], fontsize=22, fontweight="bold", pad=12)

            ax.axvspan(atp_lo, atp_hi, color=ATP_COLOR, alpha=0.25, zorder=0)
            ax.axhline(1.0, color=BLACK, linewidth=1.5, linestyle="--", zorder=1,
                       label="CR-optimal (η=1)")
            ax.axhspan(1.0, 10.0, color="#fee2e2", alpha=0.3, zorder=0)  # bias zone

            crlb_s = crlb_std_log10[:, p]   # (n_t,)

            # Stage A efficiency
            eta_A = compute_efficiency(crlb_s, sigma_stageA[:, p])
            ax.plot(t_min, eta_A, color=STAGE_A_COLOR, linewidth=2.0,
                    linestyle=":", zorder=4, alpha=0.9)

            # Stage A+BL efficiency
            eta_ABL = compute_efficiency(crlb_s, sigma_stageABL[:, p])
            ax.plot(t_min, eta_ABL, color=STAGE_ABL_COLOR, linewidth=2.2,
                    linestyle="--", zorder=5)

            # Full DL+KF efficiency (std from MCMC samples)
            sigma_dl = log10_std_from_samples(dl_disp, p)    # (n_t,)
            eta_dl   = compute_efficiency(crlb_s, sigma_dl)
            ax.plot(t_min, eta_dl, color=DL_COLOR, linewidth=2.8,
                    linestyle="-", zorder=6,
                    path_effects=[pe.withStroke(linewidth=4, foreground="white")])

            # ECM efficiency (std from multi-start fits, IQR/1.35 ≈ σ)
            sigma_ecm_pts = []
            for ecm_t in ecm_disp:
                if len(ecm_t) < 2:
                    sigma_ecm_pts.append(np.nan)
                else:
                    log10_ecm = np.log10(np.maximum(np.asarray(ecm_t, dtype=np.float64)[:, p], 1e-30))
                    q75, q25 = np.nanpercentile(log10_ecm, [75, 25])
                    sigma_ecm_pts.append((q75 - q25) / 1.35)  # IQR / 1.35 ≈ σ
            sigma_ecm = np.array(sigma_ecm_pts)
            eta_ecm   = compute_efficiency(crlb_s, sigma_ecm)
            valid_e   = ~np.isnan(eta_ecm)
            if valid_e.any():
                ax.plot(t_min[valid_e], eta_ecm[valid_e],
                        color=GOLD, linewidth=1.8, linestyle="--", alpha=0.7, zorder=3)

            ax.set_xlim(-1, t_min[-1] + cfg["interval_min"])
            ax.set_ylim(0.0, min(1.5, np.nanmax([
                np.nanmax(eta_A[np.isfinite(eta_A)]) if np.any(np.isfinite(eta_A)) else 1.5,
                np.nanmax(eta_ABL[np.isfinite(eta_ABL)]) if np.any(np.isfinite(eta_ABL)) else 1.5,
                1.5
            ]) * 1.15))
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.grid(True, alpha=0.14, linewidth=0.6, color="#bbbbbb")
            ax.tick_params(labelsize=14, which="both", length=5)
            ax.spines[["top", "right"]].set_visible(False)
            for spine in ax.spines.values():
                spine.set_linewidth(0.9)

            if col_idx == 0:
                ax.set_ylabel(f"{sname}  ({cfg['donor']})\n"
                              r"Efficiency  $\eta = \sigma_{CRLB}/\hat{\sigma}$",
                              fontsize=18, fontweight="bold", labelpad=12)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Time (min)", fontsize=17)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)

    legend_handles = [
        Line2D([0], [0], color=BLACK, linewidth=1.5, linestyle="--",
               label=r"CR-optimal  ($\eta = 1$)"),
        mpatches.Patch(facecolor="#fee2e2", alpha=0.6,
                       label=r"Bias zone  ($\eta > 1$, impossible for unbiased estimator)"),
        Line2D([0], [0], color=STAGE_A_COLOR, linewidth=2.0, linestyle=":",
               label="Stage A: Transformer only"),
        Line2D([0], [0], color=STAGE_ABL_COLOR, linewidth=2.2, linestyle="--",
               label="Stage A+C: Transformer + BL"),
        Line2D([0], [0], color=DL_COLOR, linewidth=2.8,
               path_effects=[pe.withStroke(linewidth=4, foreground="white")],
               label="Full A+C+KF"),
        Line2D([0], [0], color=GOLD, linewidth=1.8, linestyle="--", alpha=0.7,
               label="ECM (reference)"),
        mpatches.Patch(facecolor=ATP_COLOR, alpha=0.5, label="100 µM ATP"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=14,
               framealpha=0.97, edgecolor="#cccccc", handlelength=3.0,
               handletextpad=0.9, labelspacing=0.8, columnspacing=2.0,
               bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        r"Fisher Efficiency per Pipeline Stage: $\eta = \sigma_{\mathrm{CRLB}}(t) / \hat{\sigma}(t)$"
        "\n"
        r"$\eta \to 1$: CR-optimal. $\eta < 1$: room to improve. $\eta > 1$: bias detected.",
        fontsize=22, fontweight="bold", y=0.975)

    out_path = Path(out_dir) / "crlb_efficiency.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {out_path}")
    plt.close(fig)


# ============================================================
#  Figure 3: Scenario analysis + FIM eigenspectrum
# ============================================================

def make_scenario_figure(all_results, out_dir):
    """
    Two panels:
    A) FIM eigenspectrum over time (heatmap): shows WHEN parameters are identifiable
    B) CRLB for base params vs derived quantities across physiological phases
    """
    sample_names = list(SAMPLES.keys())
    fig = plt.figure(figsize=(34, 18))

    # Layout: 2 rows, each row has 3 columns (one per sample)
    # Row 1: FIM eigenspectrum heatmap per sample
    # Row 2: CRLB comparison (base vs derived) per phase per sample
    gs = fig.add_gridspec(
        2, 3, hspace=0.40, wspace=0.30,
        top=0.92, bottom=0.10, left=0.07, right=0.98,
    )

    for col_idx, sname in enumerate(sample_names):
        res  = all_results[sname]
        cfg  = SAMPLES[sname]
        n_t  = res["n_t"]
        t_min = np.arange(n_t) * cfg["interval_min"]
        atp_span = cfg["atp_span"]

        eigenvalues     = res["eigenvalues"]        # (n_t, 5) descending
        condition_nums  = res["condition_numbers"]   # (n_t,)
        crlb_std        = res["crlb_std"]            # (n_t, 5)
        crlb_ter        = res["crlb_derived_ter"]    # (n_t,)
        crlb_tau_b      = res["crlb_derived_tau_b"]  # (n_t,)
        crlb_tau_a      = res["crlb_derived_tau_a"]  # (n_t,)
        crlb_tec        = res["crlb_derived_tec"]    # (n_t,)

        # ---- Row 1: FIM eigenspectrum heatmap ----
        ax1 = fig.add_subplot(gs[0, col_idx])

        eig_matrix = eigenvalues.T   # (5, n_t)  - rows = eigenvalue index (1=largest)
        eig_positive = np.maximum(eig_matrix, 1e-12)

        im = ax1.imshow(
            np.log10(eig_positive),
            aspect="auto",
            origin="lower",
            extent=[t_min[0], t_min[-1], 0.5, 5.5],
            cmap="RdYlGn",
            vmin=-4, vmax=6,
        )
        ax1.axvspan(
            atp_span[0] * cfg["interval_min"],
            atp_span[1] * cfg["interval_min"],
            color=ATP_COLOR, alpha=0.5, zorder=3,
        )
        ax1.set_xlabel("Time (min)", fontsize=14)
        ax1.set_ylabel("FIM eigenvalue rank\n(1 = largest)", fontsize=13)
        ax1.set_yticks([1, 2, 3, 4, 5])
        ax1.set_title(f"{sname}  ({cfg['donor']})\nFIM Eigenspectrum log₁₀(λ)",
                      fontsize=16, fontweight="bold")

        plt.colorbar(im, ax=ax1, fraction=0.04, pad=0.02,
                     label="log₁₀(eigenvalue)")

        # Overlay condition number as twin axis
        ax1b = ax1.twinx()
        ax1b.plot(t_min, np.log10(np.maximum(condition_nums, 1.0)),
                  color=BLACK, linewidth=1.8, linestyle="-", alpha=0.8, zorder=5)
        ax1b.set_ylabel("log₁₀(condition number)", fontsize=12, color=BLACK)
        ax1b.tick_params(axis="y", colors=BLACK, labelsize=11)
        ax1b.spines["right"].set_color(BLACK)

        # ---- Row 2: CRLB base params vs derived per phase ----
        ax2 = fig.add_subplot(gs[1, col_idx])

        # Define physiological phases
        pre_mask  = np.arange(n_t) < atp_span[0]
        atp_mask  = (np.arange(n_t) >= atp_span[0]) & (np.arange(n_t) < atp_span[1])
        post_mask = np.arange(n_t) >= atp_span[1]
        phase_masks = [pre_mask, atp_mask, post_mask]
        phase_labels = ["Pre-ATP", "ATP", "Post-ATP"]
        phase_colors = ["#3b82f6", "#ef4444", "#10b981"]

        # Items: 5 base params + 4 derived
        item_labels = PARAM_LABELS + [r"TER", r"$\tau_b$", r"$\tau_a$", r"TEC"]
        item_colors = ["#6366f1"] * 5 + ["#f59e0b"] * 4
        n_items = len(item_labels)
        x_base  = np.arange(n_items)
        width   = 0.25

        for ph_idx, (mask, ph_label, ph_color) in enumerate(
            zip(phase_masks, phase_labels, phase_colors)
        ):
            if not mask.any():
                continue
            # Median CRLB std over timepoints in this phase
            crlb_base_phase   = np.nanmedian(crlb_std[mask], axis=0)      # (5,)
            crlb_ter_phase    = float(np.nanmedian(crlb_ter[mask]))
            crlb_taub_phase   = float(np.nanmedian(crlb_tau_b[mask]))
            crlb_taua_phase   = float(np.nanmedian(crlb_tau_a[mask]))
            crlb_tec_phase    = float(np.nanmedian(crlb_tec[mask]))

            vals = np.concatenate([crlb_base_phase,
                                   [crlb_ter_phase, crlb_taub_phase,
                                    crlb_taua_phase, crlb_tec_phase]])

            bars = ax2.bar(
                x_base + ph_idx * width, vals, width,
                label=ph_label, color=ph_color, alpha=0.75,
                edgecolor="white", linewidth=0.8,
            )

        ax2.axvline(4.5, color="#9ca3af", linewidth=1.2, linestyle="--")
        ax2.text(4.6, ax2.get_ylim()[1] * 0.01, "derived →",
                 fontsize=11, color="#9ca3af", va="bottom")
        ax2.set_xticks(x_base + width)
        ax2.set_xticklabels(item_labels, fontsize=13)
        ax2.set_ylabel("Median CRLB std  (log₁₀ decades)", fontsize=13)
        ax2.set_title(f"{sname}: CRLB by Physiological Phase",
                      fontsize=16, fontweight="bold")
        ax2.legend(fontsize=12, loc="upper right")
        ax2.grid(True, alpha=0.18, axis="y")
        ax2.spines[["top", "right"]].set_visible(False)

        # Annotate: identifiability note
        for bar_idx, label in enumerate(item_labels):
            yval = ax2.patches[bar_idx].get_height()  # first phase bar height

    # Shared annotation
    fig.text(
        0.5, 0.505,
        "Top: Green = well-identified (high FIM eigenvalue), Red = near-degenerate. "
        "Black line = log₁₀(condition number). "
        "Bottom: derived quantities (TER, τ_b) have tighter CRLB than individual Ra, Rb despite same spectrum.",
        ha="center", va="center", fontsize=12, color="#374151",
        style="italic",
    )

    fig.suptitle(
        "Identifiability Landscape: FIM Eigenspectrum and Scenario-Specific CRLB\n"
        "Derived quantities (TER, τ_b) are fundamentally more identifiable than base parameters",
        fontsize=22, fontweight="bold", y=0.975,
    )

    out_path = Path(out_dir) / "crlb_scenario.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {out_path}")
    plt.close(fig)


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CRLB analysis (12l)")
    parser.add_argument("--csv",         default=CSV_PATH)
    parser.add_argument("--out-dir",     default="results/poster")
    parser.add_argument("--cache-dir",   default="results/poster/cache_12l",
                        help="Cache directory for CRLB results")
    parser.add_argument("--e-cache-dir", default="results/poster/cache_12e")
    parser.add_argument("--k-cache-dir", default="results/poster/cache_12k")
    parser.add_argument("--recompute",   action="store_true",
                        help="Recompute CRLB even if cache exists (slow: ~2 min/sample)")
    args = parser.parse_args()

    import pandas as pd
    np.random.seed(42)
    torch.manual_seed(42)

    freqs    = np.logspace(np.log10(FREQ_MIN_HZ), np.log10(FREQ_MAX_HZ), N_FREQ)
    omega_np = 2 * np.pi * freqs

    df = pd.read_csv(args.csv)

    all_results = {}

    for sname, cfg in SAMPLES.items():
        print(f"{'=' * 60}")
        print(f"  {sname}  ({cfg['meas_id']})")
        print(f"{'=' * 60}")

        cached_12e = load_12e_cache(sname, args.e_cache_dir)
        cached_12k = load_12k_cache(sname, args.k_cache_dir)

        # Try CRLB cache
        crlb_cached = None if args.recompute else load_crlb_cache(sname, args.cache_dir)

        if crlb_cached is not None:
            print(f"  CRLB loaded from cache ({args.cache_dir})")
            crlb_std          = crlb_cached["crlb_std"]
            eigenvalues       = crlb_cached["eigenvalues"]
            condition_numbers = crlb_cached["condition_numbers"]
            n_identifiable    = crlb_cached["n_identifiable"]
            crlb_derived_ter  = crlb_cached["crlb_derived_ter"]
            crlb_derived_tau_b = crlb_cached["crlb_derived_tau_b"]
            crlb_derived_tau_a = crlb_cached["crlb_derived_tau_a"]
            crlb_derived_tec  = crlb_cached["crlb_derived_tec"]
            truth_log10       = crlb_cached["truth_log10"]
        else:
            # Reconstruct ground truth log10 parameters from CSV
            sub = load_sample(df, cfg["meas_id"])
            raw_params = []
            for _, row in sub.iterrows():
                Ra, Rb, Ca, Cb, Rsh = extract_3p_params(row)
                if Rsh > RSH_MODEL_MAX:
                    continue
                raw_params.append((Ra, Rb, Ca, Cb, Rsh))

            truth_log10 = params_to_log10_arr(raw_params)   # (n_t, 5)
            n_t = len(truth_log10)
            print(f"  {n_t} timepoints, computing CRLB at each...")

            crlb_results = compute_crlb_trajectory(truth_log10, omega_np)

            crlb_std          = np.array([r["crlb_std"]        for r in crlb_results])  # (n_t, 5)
            eigenvalues       = np.array([r["eigenvalues"]      for r in crlb_results])  # (n_t, 5)
            condition_numbers = np.array([r["condition_number"] for r in crlb_results])  # (n_t,)
            n_identifiable    = np.array([r["n_identifiable"]   for r in crlb_results])  # (n_t,)
            crlb_derived_ter  = np.array([r["crlb_derived"]["TER"]       for r in crlb_results])
            crlb_derived_tau_b = np.array([r["crlb_derived"]["tau_b"]    for r in crlb_results])
            crlb_derived_tau_a = np.array([r["crlb_derived"]["tau_a"]    for r in crlb_results])
            crlb_derived_tec  = np.array([r["crlb_derived"]["TEC"]       for r in crlb_results])

            save_crlb_cache(sname, {
                "crlb_std": crlb_std,
                "eigenvalues": eigenvalues,
                "condition_numbers": condition_numbers,
                "n_identifiable": n_identifiable,
                "crlb_derived_ter":  crlb_derived_ter,
                "crlb_derived_tau_b": crlb_derived_tau_b,
                "crlb_derived_tau_a": crlb_derived_tau_a,
                "crlb_derived_tec":  crlb_derived_tec,
                "truth_log10": truth_log10,
            }, args.cache_dir)

        # Print summary
        print(f"  CRLB summary (median across time, in log10 decades):")
        for i, name in enumerate(PARAM_NAMES):
            med = float(np.nanmedian(crlb_std[:, i]))
            print(f"    {name:>5s}:  {med:.4f} decades")
        print(f"  Derived:")
        print(f"    TER:   {float(np.nanmedian(crlb_derived_ter)):.4f} decades")
        print(f"    tau_b: {float(np.nanmedian(crlb_derived_tau_b)):.4f} decades")
        print(f"    tau_a: {float(np.nanmedian(crlb_derived_tau_a)):.4f} decades")
        print(f"    TEC:   {float(np.nanmedian(crlb_derived_tec)):.4f} decades")
        print(f"  Median identifiable dims: {float(np.nanmedian(n_identifiable)):.1f}/5")
        print(f"  Median condition number:  {float(np.nanmedian(condition_numbers)):.2e}\n")

        all_results[sname] = {
            "n_t":           cached_12e["n_t"],
            "truth_disp":    cached_12e["truth_disp"],
            "dl_disp":       cached_12e["dl_disp"],
            "ecm_disp":      cached_12e["ecm_disp"],
            "stageA_mean":   cached_12k["stageA_mean"],
            "stageA_lo":     cached_12k["stageA_lo"],
            "stageA_hi":     cached_12k["stageA_hi"],
            "stageABL_mean": cached_12k["stageABL_mean"],
            "stageABL_lo":   cached_12k["stageABL_lo"],
            "stageABL_hi":   cached_12k["stageABL_hi"],
            "crlb_std":          crlb_std,
            "eigenvalues":       eigenvalues,
            "condition_numbers": condition_numbers,
            "n_identifiable":    n_identifiable,
            "crlb_derived_ter":  crlb_derived_ter,
            "crlb_derived_tau_b": crlb_derived_tau_b,
            "crlb_derived_tau_a": crlb_derived_tau_a,
            "crlb_derived_tec":  crlb_derived_tec,
            "truth_log10":       truth_log10 if not isinstance(truth_log10, type(None)) else None,
        }

    print("Generating figures...")
    make_crlb_tracking_figure(all_results, args.out_dir)
    make_efficiency_figure(all_results, args.out_dir)
    make_scenario_figure(all_results, args.out_dir)
    print("Done. Outputs: crlb_tracking.png, crlb_efficiency.png, crlb_scenario.png")


if __name__ == "__main__":
    main()
