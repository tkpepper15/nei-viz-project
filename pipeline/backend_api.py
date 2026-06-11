#!/usr/bin/env python3
"""
Flask API for FisherAwareTransformer (auto-selects best fisher_v* checkpoint by val_mae)

Endpoints:
- GET  /model_info                    : model metadata
- POST /predict_single                : single spectrum -> parameters + uncertainty
- GET  /real_sample_data              : load the 3 Ussing chamber samples from CSV
- POST /mc_temporal_analysis          : batch temporal analysis (returns all at once)
- POST /mc_temporal_analysis_stream   : streaming temporal analysis (SSE, one event per timepoint)

DL inference pipeline (stream endpoint):
  1. Transformer -> K proposals (means + covariances)
  2. Black-Litterman refinement (iterative re-linearization, 3 passes)
  3. Sequential prior carry-forward with process noise between timepoints
  4. Sample from refined posterior -> median + IQR

ECM baseline:
  - L-BFGS-B resnorm minimization, warm-started from previous timepoint
  - Same objective as the 3P-EIS ground truth fitting
"""

import csv
import json
import math
import time
from datetime import datetime as _datetime


class _SafeEncoder(json.JSONEncoder):
    """Converts NaN/Inf floats to null so SSE payloads are always valid JSON."""
    def iterencode(self, o, _one_shot=False):
        return super().iterencode(self._sanitize(o), _one_shot)

    def _sanitize(self, obj):
        if isinstance(obj, float):
            return None if (math.isnan(obj) or math.isinf(obj)) else obj
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._sanitize(v) for v in obj]
        return obj
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from models.fisher_transformer import FisherAwareTransformer, TransformerConfig

try:
    from scipy.optimize import minimize as scipy_minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from src.pipeline.gpf import GaussianParticleFilter, BOUNDS_LOW, BOUNDS_HIGH, BOUNDS_MID, GPF_VERSION, ffbs_backward_sweep

app = Flask(__name__)
CORS(app)

MODEL = None
MDN_TEMPERATURE_SCALE = None   # (5,) per-parameter sigma multipliers, or None
IS_TEC_RATIO = False
IS_IDENTIFIABLE = False
MODEL_NAME = None
MODEL_EPOCH: int | None = None
_MODEL_CACHE: dict = {}  # name -> {model, is_identifiable, is_tec_ratio, mdn_temperature_scale, val_mae, epoch}
MODEL_VAL_MAE: float | None = None
MODEL_PARAM_MAE: dict | None = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CROSS_SECTION = 0.11409   # cm^2
USSING_SAMPLES = {
    "Sample 1": {"meas_id": "20211021_183356", "interval_min": 3.5, "atp_span": (7, 11),  "donor": "Donor A"},
    "Sample 2": {"meas_id": "20211022_171622", "interval_min": 2.4, "atp_span": (10, 15), "donor": "Donor A"},
    "Sample 3": {"meas_id": "20211014_165514", "interval_min": 1.7, "atp_span": (9, 12),  "donor": "Donor B"},
}
CSV_PATH = Path(__file__).parent.parent / "docs" / "fit results.csv"
N_FREQ = 100
FREQ_MIN_HZ = 0.1
FREQ_MAX_HZ = 1e6

# Process noise (log10/step) for temporal carry-forward.
# Loaded from data/process_noise.npz; falls back to diagonal if absent.
_PROCESS_NOISE_NPZ = Path(__file__).parent / 'data' / 'process_noise.npz'

def _load_process_noise():
    """Load Σ_Q (5x5) and mu_atp (5,) from the estimated noise file."""
    if _PROCESS_NOISE_NPZ.exists():
        data    = np.load(str(_PROCESS_NOISE_NPZ))
        sigma_q = data['sigma_q'].astype(np.float64)
        mu_atp  = data['mu_atp'].astype(np.float64)
        print(f"Loaded Σ_Q from {_PROCESS_NOISE_NPZ}")
        print(f"  diag sqrt (std/step): {np.sqrt(np.diag(sigma_q)).round(4)}")
        print(f"  mu_atp: {mu_atp.round(4)}")
        return sigma_q, mu_atp
    else:
        std     = np.array([0.04, 0.04, 0.02, 0.02, 0.04])
        print("process_noise.npz not found, using diagonal fallback Σ_Q")
        return np.diag(std ** 2), np.zeros(5)

_SIGMA_Q, _MU_ATP = _load_process_noise()


def _find_best_model() -> tuple[Path, str] | tuple[None, None]:
    """
    Scan all fisher_v* directories and return the path + name of the checkpoint
    with the lowest val_mae. Reads val_mae directly from each checkpoint — the
    training_log.csv cannot be used because logs accumulate across resumed runs
    and their minimum does not correspond to the saved checkpoint.
    """
    base = Path(__file__).parent / 'models'
    candidates = sorted(base.glob('fisher_v*/best_model.pt'))
    if not candidates:
        return None, None

    best_path: Path | None = None
    best_mae = float('inf')

    for pt_path in candidates:
        try:
            ckpt = torch.load(str(pt_path), map_location='cpu', weights_only=False)
            # Prefer canonical MAE (re-evaluated on mixed_distribution_v2) so all
            # models are compared on the same distribution regardless of which dataset
            # each was originally trained on.
            mae = float(ckpt.get(
                'val_mae_canonical',
                ckpt.get('val_mae', ckpt.get('val_mae_derived', float('inf')))
            ))
            name = pt_path.parent.name
            canon = 'canonical' if 'val_mae_canonical' in ckpt else 'stored'
            print(f"  {name}: val_mae={mae:.4f} ({canon}, epoch {ckpt.get('epoch', '?')})")
            if mae < best_mae:
                best_mae = mae
                best_path = pt_path
        except Exception as e:
            print(f"  Warning: could not read {pt_path.parent.name}: {e}")

    if best_path is None:
        return None, None
    return best_path, best_path.parent.name


def load_model():
    global MODEL, IS_TEC_RATIO, IS_IDENTIFIABLE, MODEL_NAME, MDN_TEMPERATURE_SCALE
    global MODEL_EPOCH, MODEL_VAL_MAE, MODEL_PARAM_MAE

    model_path, model_name = _find_best_model()
    if model_path is None:
        raise FileNotFoundError(
            f"No fisher_v* model found in {Path(__file__).parent / 'models'}"
        )

    ckpt = torch.load(str(model_path), map_location='cpu', weights_only=False)
    cfg_dict = ckpt['config']
    config = TransformerConfig(
        n_freq=cfg_dict['n_freq'],
        d_model=cfg_dict['d_model'],
        n_heads=cfg_dict['n_heads'],
        n_layers=cfg_dict['n_layers'],
        d_ff=cfg_dict['d_ff'],
        dropout=cfg_dict['dropout'],
        n_proposals=cfg_dict['n_proposals'],
        n_params=cfg_dict['n_params'],
        use_low_rank_cov=cfg_dict['use_low_rank_cov'],
        cov_rank=cfg_dict['cov_rank'],
        use_grad_features=cfg_dict.get('use_grad_features', True),
        use_drt=cfg_dict.get('use_drt', False),
    )
    MODEL = FisherAwareTransformer(config)
    MODEL.load_state_dict(ckpt['model_state_dict'])
    MODEL.to(DEVICE)
    MODEL.eval()

    param_space = ckpt.get('param_space', 'original')
    IS_TEC_RATIO    = param_space == 'tec_ratio'
    IS_IDENTIFIABLE = param_space == 'identifiable'
    MODEL_NAME      = model_name
    MODEL_EPOCH   = ckpt.get('epoch')
    MODEL_VAL_MAE = float(ckpt.get(
        'val_mae_canonical',
        ckpt.get('val_mae', ckpt.get('val_mae_derived', float('nan')))
    ))
    _param_src = ckpt.get('val_param_mae_canonical') or ckpt.get('val_param_mae') or {}
    MODEL_PARAM_MAE = {k: round(float(v), 4) for k, v in _param_src.items()}

    print(f"Loaded {MODEL_NAME} epoch {MODEL_EPOCH}, "
          f"val_mae={MODEL_VAL_MAE:.4f}, param_space={param_space}")
    if MODEL_PARAM_MAE:
        print(f"  per-param MAE: { {k: f'{v:.3f}' for k, v in MODEL_PARAM_MAE.items()} }")

    # Load per-parameter temperature calibration if present.
    # Scales MDN covariance diagonals so EKF observations are correctly weighted.
    MDN_TEMPERATURE_SCALE = None
    _cal_path = model_path.parent / 'temperature_calibration.json'
    if _cal_path.exists():
        with open(_cal_path) as _f:
            _cal = json.load(_f)
        MDN_TEMPERATURE_SCALE = np.array(_cal['temperatures'], dtype=np.float64)
        print(f"  Temperature calibration loaded: T={[f'{t:.3f}' for t in MDN_TEMPERATURE_SCALE]}")


def _load_model_ctx(model_name: str) -> dict:
    """Load (or return cached) model context for the given model name.

    Returns dict with keys: model, is_identifiable, is_tec_ratio, mdn_temperature_scale,
    val_mae, epoch, param_space.
    Falls back to the global model if model_name matches MODEL_NAME or is not found.
    """
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    pt_path = Path(__file__).parent / 'models' / model_name / 'best_model.pt'
    if not pt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {pt_path}")

    ckpt = torch.load(str(pt_path), map_location='cpu', weights_only=False)
    cfg_dict = ckpt['config']
    config = TransformerConfig(
        n_freq=cfg_dict['n_freq'],
        d_model=cfg_dict['d_model'],
        n_heads=cfg_dict['n_heads'],
        n_layers=cfg_dict/['n_layers'],
        d_ff=cfg_dict['d_ff'],
        dropout=cfg_dict['dropout'],
        n_proposals=cfg_dict['n_proposals'],
        n_params=cfg_dict['n_params'],
        use_low_rank_cov=cfg_dict['use_low_rank_cov'],
        cov_rank=cfg_dict['cov_rank'],
        use_grad_features=cfg_dict.get('use_grad_features', True),
        use_drt=cfg_dict.get('use_drt', False),
    )
    m = FisherAwareTransformer(config)
    m.load_state_dict(ckpt['model_state_dict'])
    m.to(DEVICE)
    m.eval()

    param_space = ckpt.get('param_space', 'original')
    val_mae = float(ckpt.get('val_mae_canonical', ckpt.get('val_mae', ckpt.get('val_mae_derived', float('nan')))))
    epoch = ckpt.get('epoch')

    temp_scale = None
    cal_path = pt_path.parent / 'temperature_calibration.json'
    if cal_path.exists():
        with open(cal_path) as f:
            cal = json.load(f)
        temp_scale = np.array(cal['temperatures'], dtype=np.float64)

    ctx = {
        'model':                 m,
        'is_identifiable':       param_space == 'identifiable',
        'is_tec_ratio':          param_space == 'tec_ratio',
        'mdn_temperature_scale': temp_scale,
        'val_mae':               val_mae,
        'epoch':                 epoch,
        'param_space':           param_space,
    }
    _MODEL_CACHE[model_name] = ctx
    print(f"Cached {model_name} epoch {epoch}, val_mae={val_mae:.4f}, param_space={param_space}")
    return ctx


def _prepare_input(Z_real_list, Z_imag_list, frequencies_list):
    """Convert raw arrays to (Z_r, Z_i, log_omega) tensors, all (1, n_freq) float32."""
    Z_r = torch.tensor(Z_real_list, dtype=torch.float32).unsqueeze(0)
    Z_i = torch.tensor(Z_imag_list, dtype=torch.float32).unsqueeze(0)
    omega = 2 * np.pi * np.array(frequencies_list)
    log_omega = torch.tensor(np.log10(omega), dtype=torch.float32).unsqueeze(0)
    return Z_r.to(DEVICE), Z_i.to(DEVICE), log_omega.to(DEVICE)


def _tec_ratio_to_original(log10_samples: np.ndarray) -> np.ndarray:
    """
    (N, 5) [Ra_log10, Rb_log10, log10(TEC), log10(Ca/Cb), Rsh_log10]
        -> (N, 5) [Ra_log10, Rb_log10, Ca_log10, Cb_log10, Rsh_log10]
    """
    Ra_l      = log10_samples[:, 0]
    Rb_l      = log10_samples[:, 1]
    log_TEC   = log10_samples[:, 2]
    log_ratio = log10_samples[:, 3]
    Rsh_l     = log10_samples[:, 4]
    TEC = 10.0 ** log_TEC
    R   = 10.0 ** log_ratio
    Ca  = TEC * (1.0 + R)
    Cb  = TEC * (1.0 + 1.0 / (R + 1e-12))
    out = np.empty_like(log10_samples)
    out[:, 0] = Ra_l
    out[:, 1] = Rb_l
    out[:, 2] = np.log10(np.clip(Ca, 1e-30, None))
    out[:, 3] = np.log10(np.clip(Cb, 1e-30, None))
    out[:, 4] = Rsh_l
    return out



def _identifiable_to_original(log10_samples: np.ndarray) -> np.ndarray:
    """
    (N, 5) [log10(tau_big), log10(tau_small), log10(TER), log10(TEC), log10(Rsh)]
         -> (N, 5) [log10(Ra), log10(Rb), log10(Ca), log10(Cb), log10(Rsh)]

    Ra has tau_small (smaller time constant), Rb has tau_big.
    Blends toward Ra=Rb=S/2 when tau_big ~ tau_small for numerical stability.
    """
    tau_big   = 10.0 ** log10_samples[:, 0]
    tau_small = 10.0 ** log10_samples[:, 1]
    TER       = 10.0 ** log10_samples[:, 2]
    TEC       = 10.0 ** log10_samples[:, 3]
    Rsh       = 10.0 ** log10_samples[:, 4]

    # Ra + Rb from TER: TER = Rsh*S/(Rsh+S) => S = TER*Rsh/(Rsh-TER)
    S = TER * Rsh / np.clip(Rsh - TER, 1e-6, None)
    S = np.clip(S, 1e-3, None)

    # Ra = tau_small*(tau_big/TEC - S) / (tau_big - tau_small)
    denom     = tau_big - tau_small
    Ra_exact  = tau_small * (tau_big / np.clip(TEC, 1e-30, None) - S) / np.clip(denom, 1e-6, None)
    Ra_sym    = S * 0.5
    alpha     = 1.0 / (1.0 + np.exp(-(denom / (tau_big * 0.1 + 1e-12)) * 5.0))
    Ra        = alpha * Ra_exact + (1.0 - alpha) * Ra_sym
    Ra        = np.clip(Ra, 1e-3, S - 1e-3)
    Rb        = np.clip(S - Ra, 1e-3, None)

    Ca = np.clip(tau_small / (Ra + 1e-12), 1e-30, None)
    Cb = np.clip(tau_big   / (Rb + 1e-12), 1e-30, None)

    out = np.empty_like(log10_samples)
    out[:, 0] = np.log10(Ra)
    out[:, 1] = np.log10(Rb)
    out[:, 2] = np.log10(Ca)
    out[:, 3] = np.log10(Cb)
    out[:, 4] = log10_samples[:, 4]   # Rsh unchanged
    return out



def _run_inference(Z_real_list, Z_imag_list, frequencies_list, n_samples=200, model_ctx=None):
    """
    Raw transformer inference (no BL refinement).

    Returns:
        samples_orig: (n_samples, 5) log10 [Ra, Rb, Ca, Cb, Rsh]
        mu_orig:      (5,) log10 mixture mean
        std_orig:     (5,) log10 mixture std
    """
    _model          = model_ctx['model']          if model_ctx else MODEL
    _is_identifiable = model_ctx['is_identifiable'] if model_ctx else IS_IDENTIFIABLE
    _is_tec_ratio   = model_ctx['is_tec_ratio']   if model_ctx else IS_TEC_RATIO

    Z_r, Z_i, log_omega = _prepare_input(Z_real_list, Z_imag_list, frequencies_list)
    with torch.no_grad():
        proposals      = _model(Z_r, Z_i, log_omega)
        samples_tec    = _model.sample_mixture_posterior(proposals, n_samples=n_samples)
        mu_tec, cov_tec = _model.get_mixture_posterior(proposals)

    samples_tec_np = samples_tec[0].cpu().numpy()
    mu_tec_np      = mu_tec[0].cpu().numpy()

    if _is_identifiable:
        samples_orig = _identifiable_to_original(samples_tec_np)
        mu_orig      = _identifiable_to_original(mu_tec_np[None])[0]
    elif _is_tec_ratio:
        samples_orig = _tec_ratio_to_original(samples_tec_np)
        mu_orig      = _tec_ratio_to_original(mu_tec_np[None])[0]
    else:
        samples_orig = samples_tec_np
        mu_orig      = mu_tec_np

    std_orig = np.std(samples_orig, axis=0)
    return samples_orig, mu_orig, std_orig



def _init_particles_from_mdn(proposals: dict, n_particles: int, model_ctx=None) -> np.ndarray:
    """
    Stratified initialization: sample N/K particles from each MDN component.

    Samples in the model's native parameter space, then converts to original
    log10 [Ra, Rb, Ca, Cb, Rsh]. This guarantees both Ra/Rb swap modes are
    represented regardless of MDN mixture weights.

    Args:
        proposals:    dict from MODEL forward pass (means, covs, weights)
        n_particles:  total number of particles
        model_ctx:    optional model context dict (from _load_model_ctx); uses globals if None

    Returns:
        particles: (n_particles, 5) log10 in original space, clipped to bounds
    """
    _is_identifiable = model_ctx['is_identifiable'] if model_ctx else IS_IDENTIFIABLE
    _is_tec_ratio    = model_ctx['is_tec_ratio']    if model_ctx else IS_TEC_RATIO

    means_k   = proposals['means'][0].cpu().numpy()    # (K, 5) model space
    covs_k    = proposals['covs'][0].cpu().numpy()     # (K, 5, 5)
    K         = means_k.shape[0]
    per_k     = n_particles // K
    extra     = n_particles - per_k * K
    all_parts = []

    for k in range(K):
        n_k = per_k + (1 if k < extra else 0)
        cov = covs_k[k] + 1e-6 * np.eye(5)
        try:
            L_k = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            L_k = np.diag(np.sqrt(np.diag(cov).clip(1e-8)))
        eps = np.random.randn(n_k, 5)
        pts = means_k[k] + eps @ L_k.T    # (n_k, 5) in model space

        if _is_identifiable:
            pts = _identifiable_to_original(pts)
        elif _is_tec_ratio:
            pts = _tec_ratio_to_original(pts)
        all_parts.append(pts)

    return np.clip(np.vstack(all_parts), BOUNDS_LOW, BOUNDS_HIGH)












def _enforce_canonical_mode(particles_log10: np.ndarray, mode: str) -> np.ndarray:
    """
    Project particles to a canonical labeling by swapping (Ra,Ca)<->(Rb,Cb) for
    particles that violate the constraint. Impedance is invariant under this swap
    so no fit information is lost — only the label assignment changes.

    mode='Ra_gt_Rb': enforce Ra >= Rb (apical resistance higher due to greater surface area).
    Ca > Cb follows automatically since the swap is always joint: (Ra,Ca)<->(Rb,Cb).

    Returns a new array; does not modify the input in-place.
    """
    p = particles_log10.copy()
    if mode == 'Ra_gt_Rb':
        needs_swap = p[:, 0] < p[:, 1]   # Ra < Rb -> relabel
        p[needs_swap, 0], p[needs_swap, 1] = p[needs_swap, 1].copy(), p[needs_swap, 0].copy()
        p[needs_swap, 2], p[needs_swap, 3] = p[needs_swap, 3].copy(), p[needs_swap, 2].copy()
    return p




def _compute_circuit_impedance(Ra, Rb, Ca, Cb, Rsh, frequencies):
    """Compute RPE circuit impedance at given frequencies (numpy)."""
    omega    = 2 * np.pi * np.array(frequencies)
    Za_real  = Ra / (1 + (omega * Ra * Ca) ** 2)
    Za_imag  = -(omega * Ra ** 2 * Ca) / (1 + (omega * Ra * Ca) ** 2)
    Zb_real  = Rb / (1 + (omega * Rb * Cb) ** 2)
    Zb_imag  = -(omega * Rb ** 2 * Cb) / (1 + (omega * Rb * Cb) ** 2)
    Zser_real = Za_real + Zb_real
    Zser_imag = Za_imag + Zb_imag
    denom    = (Rsh + Zser_real) ** 2 + Zser_imag ** 2
    Z_real   = (Rsh * Zser_real * (Rsh + Zser_real) + Rsh * Zser_imag ** 2) / denom
    Z_imag   = (Rsh ** 2 * Zser_imag) / denom
    return Z_real.tolist(), Z_imag.tolist()


def _compute_dl_resnorm(theta_log10_np, Z_real_list, Z_imag_list, frequencies_list):
    """
    Compute normalized resnorm for a parameter vector using the same cost function as ECM.
    Uses channel-max normalization: sum((Z_r - Z_pred_r)^2 / max|Z_r|^2 + ...).
    theta_log10_np: (5,) numpy array in log10 space [Ra, Rb, Ca, Cb, Rsh].
    Returns scalar float.
    """
    params = 10.0 ** theta_log10_np
    Ra, Rb, Ca, Cb, Rsh = params[0], params[1], params[2], params[3], params[4]
    Z_r_pred, Z_i_pred = _compute_circuit_impedance(Ra, Rb, Ca, Cb, Rsh, frequencies_list)
    Z_r_pred = np.array(Z_r_pred)
    Z_i_pred = np.array(Z_i_pred)
    Z_r_data = np.array(Z_real_list)
    Z_i_data = np.array(Z_imag_list)
    max_r = max(float(np.max(np.abs(Z_r_data))), 1e-10)
    max_i = max(float(np.max(np.abs(Z_i_data))), 1e-10)
    return float(np.sum(
        (Z_r_data - Z_r_pred) ** 2 / max_r ** 2 +
        (Z_i_data - Z_i_pred) ** 2 / max_i ** 2
    ))


def _compute_dl_resnorm_batch(particles_log10, Z_real_list, Z_imag_list, frequencies_list):
    """
    Vectorized resnorm for N particles. Returns (N,) array using the same channel-max
    normalization as ECM. Used to find the best particle (minimum resnorm) rather than
    evaluating the posterior median, making DL resnorm comparable to ECM's optimized cost.
    """
    from src.pipeline.smc_filter import _impedance_batch
    omega = np.array([2.0 * np.pi * f for f in frequencies_list])
    Z_r_pred, Z_i_pred = _impedance_batch(particles_log10, omega)
    Z_r_data = np.array(Z_real_list)
    Z_i_data = np.array(Z_imag_list)
    max_r = max(float(np.max(np.abs(Z_r_data))), 1e-10)
    max_i = max(float(np.max(np.abs(Z_i_data))), 1e-10)
    return np.sum(
        (Z_r_data[np.newaxis, :] - Z_r_pred) ** 2 / max_r ** 2 +
        (Z_i_data[np.newaxis, :] - Z_i_pred) ** 2 / max_i ** 2,
        axis=1,
    )


def _ecm_fit(Z_real_list, Z_imag_list, frequencies_list, x0_warm=None, n_random_starts=24):
    """
    Classical ECM fitting via L-BFGS-B resnorm minimisation.

    Cost function: channel-max normalised squared residuals (following 3P-EIS methodology).
        Each channel is divided by its own max absolute value before squaring, so all
        frequencies contribute equally within each channel and the two channels are
        balanced regardless of their absolute magnitudes:
        C(θ) = Σ_ω [(Z_r(ω) - Z_r_data(ω))² / max|Z_r_data|²
                   + (Z_i(ω) - Z_i_data(ω))² / max|Z_i_data|²]

    Starts:
      - x0_warm (previous timepoint optimum, if available)
      - n_random_starts log-uniform draws within bounds
    All starts are run independently; the lowest-cost solution is kept.

    Returns: (derived_dict, best_log10_params) or (None, None) if scipy unavailable.
    """
    if not HAS_SCIPY:
        return None, None

    Z_r_data = np.array(Z_real_list, dtype=np.float64)
    Z_i_data = np.array(Z_imag_list, dtype=np.float64)
    norm_r   = np.max(np.abs(Z_r_data)) + 1e-20  # per-channel max, not per-point
    norm_i   = np.max(np.abs(Z_i_data)) + 1e-20
    freqs    = frequencies_list

    def cost(log_p):
        Ra, Rb, Ca, Cb, Rsh = 10.0 ** log_p
        try:
            Z_r_m, Z_i_m = _compute_circuit_impedance(Ra, Rb, Ca, Cb, Rsh, freqs)
            dr = np.array(Z_r_m) - Z_r_data
            di = np.array(Z_i_m) - Z_i_data
            return float(np.sum((dr / norm_r) ** 2 + (di / norm_i) ** 2))
        except Exception:
            return 1e10

    # log10 bounds calibrated to real iPSC-RPE measurements (matches 12f reference)
    # Ra/Rb: 50–25000 Ω·cm²,  Ca/Cb: 2e-6–5e-4 F/cm²,  Rsh: 50–25000 Ω·cm²
    import math
    bounds = [
        (math.log10(50), math.log10(25000)),   # Ra
        (math.log10(50), math.log10(25000)),   # Rb
        (math.log10(2e-6), math.log10(5e-4)),  # Ca
        (math.log10(2e-6), math.log10(5e-4)),  # Cb
        (math.log10(50), math.log10(25000)),   # Rsh
    ]

    rng    = np.random.default_rng()
    starts = []
    if x0_warm is not None:
        starts.append(np.clip(x0_warm, [lo for lo, _ in bounds], [hi for _, hi in bounds]))
    for _ in range(n_random_starts):
        starts.append(np.array([rng.uniform(lo, hi) for lo, hi in bounds]))

    best_x, best_cost = None, float('inf')
    for x0 in starts:
        try:
            res = scipy_minimize(cost, x0, method='L-BFGS-B', bounds=bounds,
                                 options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_cost:
                best_cost = res.fun
                best_x    = res.x
        except Exception:
            pass

    if best_x is None:
        return None, None

    Ra, Rb, Ca, Cb, Rsh = 10.0 ** best_x
    TER = (Rsh * (Ra + Rb)) / (Rsh + Ra + Rb + 1e-20)
    TEC = (Ca * Cb) / (Ca + Cb + 1e-20)

    result = {
        'Ra': float(Ra), 'Rb': float(Rb), 'Ca': float(Ca), 'Cb': float(Cb), 'Rsh': float(Rsh),
        'R1':  float(min(Ra, Rb) / 1000),
        'C1':  float(min(Ca, Cb) * 1e6),
        'R2':  float(Rsh / 1000),
        'TER': float(TER / 1000),
        'TEC': float(TEC * 1e6),
        'resnorm': float(best_cost),
    }
    return result, best_x


def _get_analytics(Z_real_list, Z_imag_list, frequencies_list, proposals=None):
    """
    Extract MDN proposals + attention weights for display.

    Accepts an already-computed proposals dict to avoid a redundant forward pass
    when called inside generate() where proposals are computed every step.
    """
    if MODEL is None:
        return None
    try:
        if proposals is None:
            Z_r, Z_i, log_omega = _prepare_input(Z_real_list, Z_imag_list, frequencies_list)
            with torch.no_grad():
                proposals = MODEL(Z_r, Z_i, log_omega)

        means_tec = proposals['means'][0].cpu().numpy()   # (K, 5) model space
        weights   = proposals['weights'][0].cpu().numpy() # (K,)

        if IS_IDENTIFIABLE:
            means_orig = _identifiable_to_original(means_tec)
        elif IS_TEC_RATIO:
            means_orig = _tec_ratio_to_original(means_tec)
        else:
            means_orig = means_tec

        proposal_display = []
        for k in range(means_orig.shape[0]):
            Ra  = 10 ** means_orig[k, 0]
            Rb  = 10 ** means_orig[k, 1]
            Ca  = 10 ** means_orig[k, 2]
            Cb  = 10 ** means_orig[k, 3]
            Rsh = 10 ** means_orig[k, 4]
            TER = (Rsh * (Ra + Rb)) / (Rsh + Ra + Rb + 1e-20)
            TEC_val = (Ca * Cb) / (Ca + Cb + 1e-20)
            proposal_display.append({
                'R1': float(min(Ra, Rb) / 1000),
                'C1': float(min(Ca, Cb) * 1e6),
                'R2': float(Rsh / 1000),
                'TER': float(TER / 1000),
                'TEC': float(TEC_val * 1e6),
                'Ra': float(Ra / 1000),
                'Rb': float(Rb / 1000),
                'Ca': float(Ca * 1e6),
                'Cb': float(Cb * 1e6),
            })

        attn_data = MODEL.get_last_attention_and_pooling()

        return {
            'proposals': proposal_display,
            'weights':   weights.tolist(),
            'attention': attn_data['layers'],
            'pooling':   attn_data['pooling'],
        }
    except Exception:
        return None


def _raw_samples_display(phys, n_display=15):
    """
    Return a stratified subset of MC samples in display units for spaghetti line rendering.
    Uses evenly-spaced indices so the subset spans the full distribution.
    Returns dict: param_key -> list of n_display values.
    """
    n = len(phys)
    idx = np.round(np.linspace(0, n - 1, min(n_display, n))).astype(int)
    s = phys[idx]  # (n_display, 5): Ra, Rb, Ca, Cb, Rsh in linear Ω/F
    Ra, Rb, Ca, Cb, Rsh = s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4]
    TER = (Rsh * (Ra + Rb)) / (Rsh + Ra + Rb + 1e-20)
    TEC = (Ca * Cb) / (Ca + Cb + 1e-20)
    return {
        'Ra': (Ra / 1000).tolist(),
        'Rb': (Rb / 1000).tolist(),
        'Ca': (Ca * 1e6).tolist(),
        'Cb': (Cb * 1e6).tolist(),
        'R1': (np.minimum(Ra, Rb) / 1000).tolist(),
        'C1': (np.minimum(Ca, Cb) * 1e6).tolist(),
        'R2': (Rsh / 1000).tolist(),
        'TER': (TER / 1000).tolist(),
        'TEC': (TEC * 1e6).tolist(),
    }


def _derived_stats(phys):
    """Compute derived-parameter percentile stats from (n_samples, 5) physical array."""
    Ra, Rb, Ca, Cb, Rsh = phys[:, 0], phys[:, 1], phys[:, 2], phys[:, 3], phys[:, 4]
    R1  = np.minimum(Ra, Rb) / 1000
    C1  = np.minimum(Ca, Cb) * 1e6
    R2  = Rsh / 1000
    TER = (Rsh * (Ra + Rb)) / (Rsh + Ra + Rb + 1e-20) / 1000
    TEC = (Ca * Cb) / (Ca + Cb + 1e-20) * 1e6
    result = {}
    for key, arr in [
        ('R1', R1), ('C1', C1), ('R2', R2), ('TER', TER), ('TEC', TEC),
        ('Ra', Ra / 1000), ('Rb', Rb / 1000), ('Ca', Ca * 1e6), ('Cb', Cb * 1e6),
    ]:
        result[key] = {
            'mean': float(np.nanmedian(arr)),
            'q25':  float(np.nanpercentile(arr, 25)),
            'q75':  float(np.nanpercentile(arr, 75)),
        }
    return result


def _cohens_d(prev_phys, curr_phys):
    """Cohen's d attribution between consecutive timepoint physical samples."""
    def cd(col_a, col_b=None):
        a = prev_phys[:, col_a] + (prev_phys[:, col_b] if col_b is not None else 0)
        b = curr_phys[:, col_a] + (curr_phys[:, col_b] if col_b is not None else 0)
        pooled = np.sqrt((np.std(a) ** 2 + np.std(b) ** 2) / 2.0 + 1e-12)
        return float((np.mean(b) - np.mean(a)) / pooled)
    return {
        'd_Ra': cd(0), 'd_Rb': cd(1), 'd_Ca': cd(2),
        'd_Cb': cd(3), 'd_Rsh': cd(4), 'd_RaRb': cd(0, 1),
    }


# ---- Endpoints ----

@app.route('/model_info', methods=['GET'])
def model_info():
    if MODEL is None:
        return jsonify({'error': 'Model not loaded'}), 500
    if IS_IDENTIFIABLE:
        _param_space = 'identifiable'
    elif IS_TEC_RATIO:
        _param_space = 'tec_ratio'
    else:
        _param_space = 'original'
    return jsonify({
        'model':        MODEL_NAME,
        'epoch':        MODEL_EPOCH,
        'val_mae':      MODEL_VAL_MAE,
        'param_mae':    MODEL_PARAM_MAE,
        'param_space':  _param_space,
        'device':       str(DEVICE),
        'has_ecm':      HAS_SCIPY,
        'inference':   'gpf',
        'gpf_version': GPF_VERSION,
        'gpf_bounds': {
            'low':  BOUNDS_LOW.tolist(),
            'high': BOUNDS_HIGH.tolist(),
            'params': ['Ra', 'Rb', 'Ca', 'Cb', 'Rsh'],
        },
    })


@app.route('/available_models', methods=['GET'])
def available_models():
    base = Path(__file__).parent / 'models'
    result = []
    for pt_path in sorted(base.glob('fisher_v*/best_model.pt')):
        name = pt_path.parent.name
        try:
            ckpt = torch.load(str(pt_path), map_location='cpu', weights_only=False)
            mae = float(ckpt.get('val_mae_canonical', ckpt.get('val_mae', ckpt.get('val_mae_derived', float('nan')))))
            epoch = ckpt.get('epoch')
            param_space = ckpt.get('param_space', 'original')
            mae_src = 'canonical' if 'val_mae_canonical' in ckpt else 'stored'
            result.append({'name': name, 'val_mae': round(mae, 4), 'epoch': epoch, 'param_space': param_space, 'mae_source': mae_src, 'active': name == MODEL_NAME})
        except Exception as e:
            result.append({'name': name, 'val_mae': None, 'epoch': None, 'param_space': None, 'mae_source': None, 'active': name == MODEL_NAME, 'error': str(e)})
    result.sort(key=lambda x: x['val_mae'] if x['val_mae'] is not None else float('inf'))
    return jsonify({'models': result})


@app.route('/predict_single', methods=['POST'])
def predict_single():
    """Single spectrum -> parameter posterior summary."""
    if MODEL is None:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        data = request.json
        for f in ['Z_real', 'Z_imag', 'frequencies']:
            if f not in data:
                return jsonify({'error': f'Missing field: {f}'}), 400

        samples, mu, std = _run_inference(data['Z_real'], data['Z_imag'], data['frequencies'])
        names    = ['Ra', 'Rb', 'Ca', 'Cb', 'Rsh']
        phys_mu  = 10 ** mu
        phys_std = 10 ** (mu + std) - 10 ** mu

        return jsonify({
            'parameters': {n: float(phys_mu[i])  for i, n in enumerate(names)},
            'sigma':      {n: float(phys_std[i]) for i, n in enumerate(names)},
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/real_sample_data', methods=['GET'])
def real_sample_data():
    """
    Load 3 Ussing chamber samples from CSV.
    Returns synthetic spectra computed from 3P-EIS ground truth + 1% noise.
    """
    try:
        import pandas as pd
        if not CSV_PATH.exists():
            return jsonify({'error': f'CSV not found at {CSV_PATH}'}), 404

        df    = pd.read_csv(str(CSV_PATH))
        freqs = np.logspace(np.log10(FREQ_MIN_HZ), np.log10(FREQ_MAX_HZ), N_FREQ).tolist()

        samples_out = []
        rng = np.random.default_rng(42)
        for sample_name, cfg in USSING_SAMPLES.items():
            rows = (df[(df['chamber'] == 'Ussing') & (df['meas_ID'] == cfg['meas_id'])]
                    .sort_values('meas_idx').reset_index(drop=True))
            if len(rows) == 0:
                continue

            timepoints = []
            for idx, row in rows.iterrows():
                Ra  = float(row['pg_absZr_3']) * CROSS_SECTION
                Rb  = float(row['pg_absZr_5']) * CROSS_SECTION
                Ca  = float(row['pg_absZr_4']) / CROSS_SECTION
                Cb  = float(row['pg_absZr_6']) / CROSS_SECTION
                Rsh = float(row['pg_absZr_7']) * CROSS_SECTION

                Z_r, Z_i = _compute_circuit_impedance(Ra, Rb, Ca, Cb, Rsh, freqs)
                mag  = np.sqrt(np.array(Z_r) ** 2 + np.array(Z_i) ** 2)
                Z_r  = (np.array(Z_r) + 0.01 * mag * rng.standard_normal(len(Z_r))).tolist()
                Z_i  = (np.array(Z_i) + 0.01 * mag * rng.standard_normal(len(Z_i))).tolist()

                TER = (Rsh * (Ra + Rb)) / (Rsh + Ra + Rb + 1e-20)
                TEC = (Ca * Cb) / (Ca + Cb + 1e-20)
                timepoints.append({
                    'time_min':    float(idx * cfg['interval_min']),
                    'Z_real':      Z_r,
                    'Z_imag':      Z_i,
                    'frequencies': freqs,
                    'ground_truth': {
                        'Ra': Ra, 'Rb': Rb, 'Ca': Ca, 'Cb': Cb, 'Rsh': Rsh,
                        'R1':  float(min(Ra, Rb) / 1000),
                        'C1':  float(min(Ca, Cb) * 1e6),
                        'R2':  float(Rsh / 1000),
                        'TER': float(TER / 1000),
                        'TEC': float(TEC * 1e6),
                    },
                })

            samples_out.append({
                'name':         sample_name,
                'donor':        cfg['donor'],
                'interval_min': cfg['interval_min'],
                'atp_lo':       cfg['atp_span'][0] * cfg['interval_min'],
                'atp_hi':       cfg['atp_span'][1] * cfg['interval_min'],
                'timepoints':   timepoints,
            })

        return jsonify({'samples': samples_out, 'frequencies': freqs})

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/mc_temporal_analysis', methods=['POST'])
def mc_temporal_analysis():
    """
    Batch temporal analysis (returns all results at once).
    Kept for compatibility; prefer /mc_temporal_analysis_stream for real-time updates.
    Uses raw transformer sampling (no BL refinement, no sequential prior).
    """
    if MODEL is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data      = request.json
        sequences = data.get('sequences', [])
        n_samples = int(data.get('n_samples', 200))
        if len(sequences) == 0:
            return jsonify({'error': 'Empty sequences'}), 400

        time_min         = []
        all_phys_samples = []

        for seq in sequences:
            time_min.append(float(seq.get('time_min', len(time_min))))
            samples_log10, _, _ = _run_inference(
                seq['Z_real'], seq['Z_imag'], seq['frequencies'], n_samples=n_samples
            )
            all_phys_samples.append(10 ** samples_log10)

        per_derived = {k: {'mean': [], 'q25': [], 'q75': []} for k in ['R1', 'C1', 'R2', 'TER', 'TEC']}
        for phys in all_phys_samples:
            stats = _derived_stats(phys)
            for key in per_derived:
                per_derived[key]['mean'].append(stats[key]['mean'])
                per_derived[key]['q25'].append(stats[key]['q25'])
                per_derived[key]['q75'].append(stats[key]['q75'])

        n_t = len(time_min)

        def cohens_d_series(col_idx):
            return [
                float(
                    (np.mean(all_phys_samples[t + 1][:, col_idx]) - np.mean(all_phys_samples[t][:, col_idx])) /
                    np.sqrt((np.std(all_phys_samples[t][:, col_idx]) ** 2 +
                             np.std(all_phys_samples[t + 1][:, col_idx]) ** 2) / 2.0 + 1e-12)
                )
                for t in range(n_t - 1)
            ]

        def cohens_d_sum(col_a, col_b):
            return [
                float(
                    (np.mean(all_phys_samples[t + 1][:, col_a] + all_phys_samples[t + 1][:, col_b]) -
                     np.mean(all_phys_samples[t][:, col_a] + all_phys_samples[t][:, col_b])) /
                    np.sqrt((np.std(all_phys_samples[t][:, col_a] + all_phys_samples[t][:, col_b]) ** 2 +
                             np.std(all_phys_samples[t + 1][:, col_a] + all_phys_samples[t + 1][:, col_b]) ** 2) / 2.0 + 1e-12)
                )
                for t in range(n_t - 1)
            ]

        return jsonify({
            'time_min':    time_min,
            'predictions': per_derived,
            'attribution': {
                'd_Ra': cohens_d_series(0), 'd_Rb': cohens_d_series(1),
                'd_Ca': cohens_d_series(2), 'd_Cb': cohens_d_series(3),
                'd_Rsh': cohens_d_series(4), 'd_RaRb': cohens_d_sum(0, 1),
            },
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


def _derived_from_log10(p: np.ndarray) -> dict:
    """Convert a log10 [Ra, Rb, Ca, Cb, Rsh] particle to derived display quantities."""
    Ra  = float(10.0 ** p[0])
    Rb  = float(10.0 ** p[1])
    Ca  = float(10.0 ** p[2])
    Cb  = float(10.0 ** p[3])
    Rsh = float(10.0 ** p[4])
    TER = Rsh * (Ra + Rb) / (Rsh + Ra + Rb + 1e-20)
    TEC = Ca * Cb / (Ca + Cb + 1e-20)
    return {
        'TER':       float(TER),
        'TEC':       float(TEC),
        'tau_big':   float(max(Ra * Ca, Rb * Cb)),
        'tau_small': float(min(Ra * Ca, Rb * Cb)),
        'Rsh':       float(Rsh),
        'Ra':        float(Ra),
        'Rb':        float(Rb),
        'Ca':        float(Ca),
        'Cb':        float(Cb),
    }


def _to_identifiable_log10(particles: np.ndarray) -> np.ndarray:
    """(N,5) log10[Ra,Rb,Ca,Cb,Rsh] -> log10[tau_big,tau_small,TER,TEC,Rsh].

    Canonical ordering tau_big >= tau_small breaks the Ra/Rb swap degeneracy.
    The hard max/min is replaced with a smooth softmax approximation near the
    crossing point (τ_a ≈ τ_b), eliminating the kink in the identifiable
    manifold that breaks the Gaussian Kalman approximation:

        tau_big   ≈ log-sum-exp(log τ_a, log τ_b) / β
        tau_small ≈ -log-sum-exp(-log τ_a, -log τ_b) / β

    For β=20 the approximation is within 0.01 log10 of hard max/min across
    the entire parameter space, but is differentiable everywhere.
    """
    Ra  = 10.0 ** particles[:, 0]
    Rb  = 10.0 ** particles[:, 1]
    Ca  = 10.0 ** particles[:, 2]
    Cb  = 10.0 ** particles[:, 3]
    Rsh = 10.0 ** particles[:, 4]
    tau_a  = Ra * Ca
    tau_b  = Rb * Cb
    # Smooth max/min via log-sum-exp in log space (β controls sharpness)
    _BETA = 20.0
    log_a = np.log(tau_a + 1e-30)
    log_b = np.log(tau_b + 1e-30)
    log_tau_big   = (np.logaddexp(_BETA * log_a, _BETA * log_b) / _BETA) / np.log(10)
    log_tau_small = -(np.logaddexp(-_BETA * log_a, -_BETA * log_b) / _BETA) / np.log(10)
    S   = Ra + Rb
    TER = Rsh * S / (Rsh + S + 1e-30)
    TEC = Ca * Cb / (Ca + Cb + 1e-30)
    return np.stack([
        log_tau_big,
        log_tau_small,
        np.log10(TER + 1e-30),
        np.log10(TEC + 1e-30),
        particles[:, 4],
    ], axis=1)


def _rts_smooth_1d(series: np.ndarray, q: float = 0.02, r: float = 0.05) -> np.ndarray:
    """Fixed-gain RTS (Rauch-Tung-Striebel) smoother for a scalar time series.

    Forward Kalman filter pass followed by a backward smoothing pass.
    Removes step-to-step jitter while preserving real transitions (e.g. ATP events).

    Args:
        q: process noise variance in log10 units (smaller = assume slower change)
        r: measurement noise variance (larger = smooth more aggressively)
    """
    T = len(series)
    if T <= 2:
        return np.asarray(series, dtype=float).copy()
    s = np.asarray(series, dtype=float)
    xf = np.empty(T)
    pf = np.empty(T)
    xf[0], pf[0] = s[0], r
    for t in range(1, T):
        pp    = pf[t - 1] + q
        k     = pp / (pp + r)
        xf[t] = xf[t - 1] + k * (s[t] - xf[t - 1])
        pf[t] = (1.0 - k) * pp
    xs = xf.copy()
    for t in range(T - 2, -1, -1):
        g     = pf[t] / (pf[t] + q)
        xs[t] = xf[t] + g * (xs[t + 1] - xf[t])
    return xs


def _structured_Q_VEL(use_raw_space: bool) -> np.ndarray:
    """
    Velocity process noise matrix Q_VEL (5×5, log10/min² spectral density).

    Identifiable space [tau_big, tau_small, TER, TEC, Rsh]:
        Parameters are approximately independent in this space.
        Q_VEL set so 3σ velocity matches observed RPE drift rates.
        Derived from raw-space rates propagated through the Jacobian of the
        identifiable transform:
            log τ_big ≈ log Ra + log Ca  →  q_τ ≈ q_Ra + q_Ca = 1.6e-5 + 1.0e-6
            log TER   ≈ log Rsh (weak approx)  →  q_TER = q_Rsh * (TER/Rsh)² ≈ 4.0e-6

    Raw space [Ra, Rb, Ca, Cb, Rsh]:
        Ra and Ca are anti-correlated in velocity because τ_a = Ra·Ca is biologically
        constrained (membrane time constant doesn't drift freely). Similarly Rb/Cb.
        In log10 space: d(log Ra)/dt = -d(log Ca)/dt when τ_a is constant.
        This is encoded as off-diagonal terms: Q[Ra,Ca] = Q[Rb,Cb] = -q_R.
        The anti-correlation is partial (scaled by 0.7) since τ does change somewhat.
    """
    if use_raw_space:
        q_R   = 1.6e-5   # Ra, Rb velocity spectral density
        q_C   = 1.0e-6   # Ca, Cb velocity spectral density
        q_Rsh = 4.9e-5   # Rsh velocity spectral density
        # Off-diagonal: Ra-Ca anti-correlation encodes τ_a = Ra·Ca constraint.
        # Sign: if log Ra increases, log Ca must decrease to keep log τ_a fixed.
        # Magnitude scaled by geometric mean and 0.7 correlation coefficient.
        q_RC  = -0.7 * np.sqrt(q_R * q_C)
        Q = np.array([
            [ q_R,    0,    q_RC,  0,    0     ],   # Ra
            [ 0,      q_R,  0,     q_RC, 0     ],   # Rb
            [ q_RC,   0,    q_C,   0,    0     ],   # Ca
            [ 0,      q_RC, 0,     q_C,  0     ],   # Cb
            [ 0,      0,    0,     0,    q_Rsh ],   # Rsh
        ])
    else:
        # Identifiable space — approximately independent evolution.
        # q_τ ≈ q_Ra + q_Ca = 1.6e-5 + 1.0e-6 (from Jacobian of τ = R·C transform)
        q_tau = 1.7e-5
        q_TER = 4.0e-6
        q_TEC = 1.0e-6
        q_Rsh = 4.9e-5
        Q = np.diag([q_tau, q_tau, q_TER, q_TEC, q_Rsh])
    # Ensure positive definite (numerical safety)
    Q += np.eye(5) * 1e-12
    return Q


def _kalman_smooth_identifiable(
    particles_hist: list[np.ndarray],
    time_min: list[float] | None = None,
    n_predict_steps: int = 0,
    predict_dt_min: float = 1.0,
    use_raw_space: bool = False,
    changepoint_times: list[float] | None = None,
) -> dict:
    """
    Constant-velocity (CV) Kalman smoother in identifiable log10 space.

    Fixes applied vs naive implementation:

    1. R_t = cov_meas[t] / N_eff  (not cov_meas[t]).
       cov_meas is the posterior covariance — the spread of the particle cloud.
       The Kalman observation noise R_t should be the precision of our ESTIMATE
       of the mean, not the width of the distribution. Using cov_meas directly
       conflates posterior uncertainty with measurement noise, causing the Kalman
       to ignore degenerate directions entirely and accumulate large innovations.
       Standard error: R_t = cov_meas[t] / N_eff where N_eff = ESS of the cloud.

    2. Structured Q_VEL with physical coupling (raw space only).
       In raw space, τ_a = Ra·Ca is constrained, so Ra and Ca are anti-correlated
       in velocity. Diagonal Q_VEL ignores this; the structured version encodes
       Q[Ra,Ca] < 0 (and Q[Rb,Cb] < 0), reflecting the RC product constraint.

    3. Segmented smoother at change-points.
       The RTS smoother is run separately within each segment between detected
       change-points. This prevents post-event data from distorting pre-event
       estimates via retroactive smoothing.

    4. Downweight bimodal timepoints.
       When the Ra/Rb distribution is bimodal (p_Ra_lt_Rb ≈ 0.5), the Gaussian
       approximation is poor — the covariance is inflated. We detect this via the
       bimodality coefficient of the particle cloud and scale up R_t accordingly,
       effectively reducing the Kalman gain at those steps.

    State: x = [theta (5D), theta_dot (5D)]
    Transition: F(Δt), Q(Δt) — CWNA model
    Measurement: H = [I, 0] — observe position only
    """
    T = len(particles_hist)
    D = 5
    D2 = 10

    _Q_VEL = _structured_Q_VEL(use_raw_space)

    if time_min is None or len(time_min) != T:
        dt_arr = np.ones(T)
        t_arr  = np.arange(T, dtype=float)
    else:
        dt_arr = np.diff([0.0] + list(time_min))
        dt_arr = np.clip(dt_arr, 1e-3, None)
        dt_arr[0] = dt_arr[1] if T > 1 else 1.0
        t_arr = np.array(time_min, dtype=float)

    if use_raw_space:
        particles_id = particles_hist
    else:
        particles_id = [_to_identifiable_log10(p) for p in particles_hist]

    mu_meas  = np.stack([np.mean(p, axis=0)             for p in particles_id])
    cov_meas = np.stack([np.cov(p.T) + np.eye(D) * 1e-6 for p in particles_id])

    # Fix 1: R_t = cov_meas / N_eff — standard error of the mean, not posterior std.
    # N_eff approximated as N_particles (all particles have equal weight after resampling).
    # This separates measurement precision from posterior uncertainty and prevents the
    # Kalman from ignoring degenerate directions due to inflated R_t.
    N_eff_arr = np.array([float(p.shape[0]) for p in particles_id])

    # Fix 4: bimodality detection — scale up R_t when cloud is bimodal.
    # Bimodality coefficient: BC = (skew² + 1) / kurtosis. BC > 5/9 suggests bimodal.
    # We use a simpler proxy: variance of the Ra-Rb sign (fraction near 0.5 = bimodal).
    bimodal_scale = np.ones(T)
    if not use_raw_space:
        for t in range(T):
            p = particles_hist[t]
            p_Ra_lt_Rb = float(np.mean(p[:, 0] < p[:, 1]))
            # Entropy of the Ra/Rb assignment: 0 = unimodal, 1 = maximally bimodal
            p_clamp = float(np.clip(p_Ra_lt_Rb, 1e-6, 1 - 1e-6))
            entropy = -p_clamp * np.log2(p_clamp) - (1 - p_clamp) * np.log2(1 - p_clamp)
            # Inflate R_t when bimodal: at entropy=1 → scale=10x, at entropy=0 → scale=1x
            bimodal_scale[t] = 1.0 + 9.0 * entropy

    H = np.hstack([np.eye(D), np.zeros((D, D))])

    # ── Cramér-Rao bounds from Fisher Information Matrix ──────────────────────
    # cr_fim[t] gives the minimum achievable variance (log10)² per parameter at
    # operating point t. The CR bound is a hard physical floor: no estimator —
    # including the Kalman — can be more precise than this, regardless of how
    # many particles or time steps it uses.
    #
    # Two uses:
    #   R_t floor  — prevents the filter from over-trusting observations in
    #                directions the spectrum cannot resolve (Ra/Rb individually).
    #   Q_vel aug  — unobservable directions (large CR) are allowed to drift
    #                more freely; the filter cannot detect such drift anyway.
    #
    # Only applied in identifiable space (use_raw_space=False). The raw-space
    # smoother is diagnostic; it does not need the same guarantees.
    _CR_Q_WEIGHT = 1e-6     # (log10/min)² per (log10)² of CR — small, conservative
    cr_fim = np.zeros((T, D))
    if not use_raw_space:
        try:
            from src.physics.eis_fisher import compute_cr_bounds_identifiable as _cr_fn
            mu_raw_meas = np.stack([np.mean(p, axis=0) for p in particles_hist])
            # Estimate per-step noise: SNR=40dB relative to TER (DC impedance limit)
            Ra_m  = 10.0 ** mu_raw_meas[:, 0]
            Rb_m  = 10.0 ** mu_raw_meas[:, 1]
            Rsh_m = 10.0 ** mu_raw_meas[:, 4]
            S_m   = Ra_m + Rb_m
            ter_m = Rsh_m * S_m / (Rsh_m + S_m + 1e-6)    # TER in Ω
            sigma_m = np.maximum(ter_m * 0.01, 0.1)         # 1% of TER, min 0.1 Ω
            # Build the global omega grid (matches training and inference)
            _freqs_hz = np.logspace(
                np.log10(FREQ_MIN_HZ), np.log10(FREQ_MAX_HZ), N_FREQ
            )
            _omega_cr = 2.0 * np.pi * _freqs_hz
            # Single batched FIM call for all T timepoints, then scale per-step
            sigma_ref  = float(np.median(sigma_m))
            cr_batch   = _cr_fn(mu_raw_meas, _omega_cr, noise_std=sigma_ref)  # (T, 5)
            # Scale each timepoint by (σ_t / σ_ref)² since CR ∝ σ²
            cr_fim = cr_batch * (sigma_m / sigma_ref).reshape(-1, 1) ** 2
        except Exception:
            pass  # Fall back to empirical-only R_t if FIM computation fails

    def _make_F_Q(dt: float, t: int = -1) -> tuple[np.ndarray, np.ndarray]:
        F = np.eye(D2)
        F[:D, D:] = dt * np.eye(D)
        # FIM-informed Q: augment velocity noise in directions the spectrum cannot
        # constrain. Unobservable parameters (large CR) can drift more freely —
        # the filter cannot detect such drift from the spectrum alone so there is
        # no reason to penalise it via a tight prior.
        Q_vel_t = _Q_VEL.copy()
        if t >= 0 and np.any(cr_fim[t] > 0):
            Q_vel_t += _CR_Q_WEIGHT * np.diag(cr_fim[t])
        Q = np.zeros((D2, D2))
        Q[:D, :D] = (dt**3 / 3.0) * Q_vel_t
        Q[:D, D:] = (dt**2 / 2.0) * Q_vel_t
        Q[D:, :D] = (dt**2 / 2.0) * Q_vel_t
        Q[D:, D:] =  dt            * Q_vel_t
        return F, Q

    def _R_t(t: int) -> np.ndarray:
        """Observation noise: empirical SE of the particle mean, floored by the
        Cramér-Rao bound and inflated for bimodal timepoints.

        The CR floor ensures the Kalman cannot claim more precision than the EIS
        spectrum physically supports at the current operating point. Without it,
        the filter can become over-confident in unobservable parameters (Ra, Rb
        individually) when the particle cloud collapses — not because the data
        supports it but because the cloud lost variance through resampling.
        """
        R_empirical = cov_meas[t] / N_eff_arr[t] * bimodal_scale[t]
        if np.any(cr_fim[t] > 0):
            R_out = R_empirical.copy()
            for d in range(D):
                R_out[d, d] = max(R_empirical[d, d], cr_fim[t, d])
            return R_out
        return R_empirical

    def _fresh_state(t: int) -> tuple[np.ndarray, np.ndarray]:
        """Initial augmented state at timepoint t: position from measurement, velocity = 0."""
        x = np.zeros(D2)
        x[:D] = mu_meas[t]
        P = np.zeros((D2, D2))
        P[:D, :D] = cov_meas[t]   # initial position uncertainty = full posterior
        P[D:, D:] = _Q_VEL * 100.0
        return x, P

    # Fix 3: segment the smoother at detected change-points.
    # Build segment boundaries: indices in [0..T-1] where the smoother reinitializes.
    seg_starts = [0]
    if changepoint_times:
        for cp_t in sorted(changepoint_times):
            # Find the first timepoint >= cp_t
            idx = int(np.searchsorted(t_arr, cp_t, side='left'))
            if 0 < idx < T and idx not in seg_starts:
                seg_starts.append(idx)
    seg_starts.sort()
    # Segment boundaries: [(start0, end0), (start1, end1), ...]
    segments = [(seg_starts[i], seg_starts[i + 1] if i + 1 < len(seg_starts) else T)
                for i in range(len(seg_starts))]

    # Allocate full arrays (filled segment by segment)
    x_f = np.zeros((T, D2))
    P_f = np.zeros((T, D2, D2))
    innov     = np.zeros((T, D))
    innov_cov = np.zeros((T, D, D))

    # Forward filter — run independently in each segment
    for seg_lo, seg_hi in segments:
        x_f[seg_lo], P_f[seg_lo] = _fresh_state(seg_lo)
        innov_cov[seg_lo] = _R_t(seg_lo)
        for t in range(seg_lo + 1, seg_hi):
            F, Q = _make_F_Q(float(dt_arr[t]), t)
            x_pred = F @ x_f[t - 1]
            P_pred = F @ P_f[t - 1] @ F.T + Q
            R_t    = _R_t(t)
            S_t    = H @ P_pred @ H.T + R_t
            try:
                K_t = np.linalg.solve(S_t.T, (P_pred @ H.T).T).T
            except np.linalg.LinAlgError:
                K_t = np.zeros((D2, D))
            innov[t]     = mu_meas[t] - H @ x_pred
            innov_cov[t] = S_t
            x_f[t]       = x_pred + K_t @ innov[t]
            P_f[t]       = (np.eye(D2) - K_t @ H) @ P_pred

    # Backward RTS smoother — run independently within each segment so post-event
    # data cannot retroactively distort pre-event estimates.
    x_s = x_f.copy()
    P_s = P_f.copy()
    for seg_lo, seg_hi in segments:
        for t in range(seg_hi - 2, seg_lo - 1, -1):
            F, Q   = _make_F_Q(float(dt_arr[t + 1]), t + 1)
            P_pred = F @ P_f[t] @ F.T + Q
            try:
                G_t = np.linalg.solve(P_pred.T, (P_f[t] @ F.T).T).T
            except np.linalg.LinAlgError:
                G_t = np.zeros((D2, D2))
            x_s[t] = x_f[t] + G_t @ (x_s[t + 1] - F @ x_f[t])
            P_s[t] = P_f[t] + G_t @ (P_s[t + 1] - P_pred) @ G_t.T

    def _diag_std(P_arr: np.ndarray, lo: int, hi: int) -> np.ndarray:
        return np.sqrt(np.maximum(
            np.stack([P_arr[t, lo:hi, lo:hi].diagonal() for t in range(T)]), 0
        ))

    # Dense temporal prediction grid: interpolate between measurements + extrapolate forward.
    # Resolution: 0.5-min intervals. Extrapolation uses final smoothed velocity state —
    # uncertainty grows as Δt² beyond the last measurement (correct for CV model).
    t_lo   = float(t_arr[0])
    t_hi   = float(t_arr[-1])
    t_pred_hi = t_hi + n_predict_steps * predict_dt_min
    t_grid = np.arange(t_lo, t_pred_hi + 1e-9, 0.5)

    mu_grid  = np.zeros((len(t_grid), D))
    std_grid = np.zeros((len(t_grid), D))

    for i, tq in enumerate(t_grid):
        if tq <= t_arr[0]:
            idx, dt = 0, tq - t_arr[0]
        elif tq >= t_arr[-1]:
            idx, dt = T - 1, tq - t_arr[-1]
        else:
            idx = int(np.searchsorted(t_arr, tq, side='right')) - 1
            dt  = tq - t_arr[idx]
        F, Q = _make_F_Q(abs(float(dt)))
        x_q  = F @ x_s[idx]
        P_q  = F @ P_s[idx] @ F.T + Q
        mu_grid[i]  = (H @ x_q)
        std_grid[i] = np.sqrt(np.maximum((H @ P_q @ H.T).diagonal(), 0))

    # Overall trend estimation via linear regression on smoothed positions.
    # slope[d] = best-fit log10/min rate over the entire experiment duration.
    # Captures the dominant direction even through transient events (ATP etc).
    t_c = t_arr - t_arr.mean()
    denom = float(np.dot(t_c, t_c)) + 1e-10
    pos_s = x_s[:, :D]
    vel_s = x_s[:, D:]
    trends: list[dict] = []
    for d in range(D):
        slope      = float(np.dot(t_c, pos_s[:, d]) / denom)   # log10/min, linear fit
        net_change = float(pos_s[-1, d] - pos_s[0, d])         # total log10 displacement
        mean_vel   = float(np.mean(vel_s[:, d]))                # mean instantaneous velocity
        # Classify by net change (≥ 0.02 log10 ≈ 5% change threshold)
        if abs(net_change) < 0.02:
            direction = 'stable'
        elif net_change > 0:
            direction = 'rising'
        else:
            direction = 'falling'
        trends.append({
            'slope':      round(slope,      6),
            'net_change': round(net_change, 4),
            'mean_vel':   round(mean_vel,   6),
            'direction':  direction,
        })

    # Posterior geometry: eigendecompose measurement covariance at each timepoint.
    # Eigenvectors reveal degenerate directions (large eigenvalue = spread = uncertain).
    # Identifiability index = λ_min / λ_max: 0 = fully degenerate, 1 = isotropic.
    # Innovations projected onto eigenvectors show which directions receive signal.
    evals  = np.zeros((T, D))
    evecs  = np.zeros((T, D, D))   # rows = eigenvectors (descending eigenvalue)
    ident  = np.zeros(T)
    innov_proj = np.zeros((T, D))
    for t in range(T):
        vals, vecs = np.linalg.eigh(cov_meas[t])   # ascending
        order = np.argsort(vals)[::-1]              # descending
        evals[t]  = vals[order]
        evecs[t]  = vecs[:, order].T                # rows = eigenvectors
        ident[t]  = float(vals[order[-1]] / (vals[order[0]] + 1e-12))
        innov_proj[t] = evecs[t] @ innov[t]

    geometry = {
        'eigen_vals':      evals.tolist(),       # (T, D) descending — spread per direction
        'eigen_vecs':      evecs.tolist(),        # (T, D, D) — rows are eigenvectors in param space
        'identifiability': ident.tolist(),        # (T,) scalar 0..1
        'innov_proj':      innov_proj.tolist(),   # (T, D) — signed projection of innovation
    }

    return {
        # Filtered (causal)
        'mu_filtered':       x_f[:, :D].tolist(),
        'std_filtered':      _diag_std(P_f, 0, D).tolist(),
        'mu_vel_filtered':   x_f[:, D:].tolist(),
        'std_vel_filtered':  _diag_std(P_f, D, D2).tolist(),
        # Smoothed (retrospective)
        'mu_smoothed':       x_s[:, :D].tolist(),
        'std_smoothed':      _diag_std(P_s, 0, D).tolist(),
        'mu_vel_smoothed':   x_s[:, D:].tolist(),
        'std_vel_smoothed':  _diag_std(P_s, D, D2).tolist(),
        # Dense temporal grid
        't_grid':    t_grid.tolist(),
        'mu_grid':   mu_grid.tolist(),
        'std_grid':  std_grid.tolist(),
        # Overall trend: linear slope, net displacement, mean velocity, direction per param
        'trends': trends,
        # Posterior geometry (eigendecomposition of measurement covariance)
        'geometry': geometry,
        # For change-point detection (unchanged interface)
        'innovations': innov,
        'innov_cov':   innov_cov,
    }


def _detect_changepoints(innovations: np.ndarray, innov_covs: np.ndarray, threshold_chi2: float = 20.5) -> list[dict]:
    """
    Change-point detection on the Kalman innovation sequence.

    Under the null hypothesis (no structural break at t):
        e_t @ S_t^{-1} @ e_t ~ chi-squared(5)
    Critical value at p=0.001 with 5 df ≈ 20.5.

    A spike above threshold indicates a parameter change larger than expected from
    process noise alone — likely an ATP event or other biological perturbation.

    Returns a list of detected events with the dominant identifiable parameter,
    its direction of change, and a physiological interpretation.
    """
    _PARAM_NAMES = ['tau_big', 'tau_small', 'TER', 'TEC', 'Rsh']
    _INTERPRETATIONS: dict[tuple[str, str], str] = {
        ('TER',       'decreased'): 'Transepithelial resistance drop — barrier weakened',
        ('TER',       'increased'): 'Transepithelial resistance rise — barrier strengthened',
        ('Rsh',       'decreased'): 'Shunt resistance drop — paracellular leak opened',
        ('Rsh',       'increased'): 'Shunt resistance rise — paracellular seal improved',
        ('tau_big',   'decreased'): 'Dominant time constant shortened — membrane RC changed',
        ('tau_big',   'increased'): 'Dominant time constant lengthened — membrane RC changed',
        ('tau_small', 'decreased'): 'Secondary time constant shortened',
        ('tau_small', 'increased'): 'Secondary time constant lengthened',
        ('TEC',       'decreased'): 'Series capacitance decreased — membrane geometry changed',
        ('TEC',       'increased'): 'Series capacitance increased — membrane geometry changed',
    }

    events = []
    for t in range(1, len(innovations)):
        e = innovations[t]
        S = innov_covs[t]
        if not (np.all(np.isfinite(e)) and np.all(np.isfinite(S))):
            continue
        try:
            S_inv = np.linalg.solve(S + np.eye(len(e)) * 1e-8, np.eye(len(e)))
            mahal = float(e @ S_inv @ e)
        except np.linalg.LinAlgError:
            continue
        if mahal > threshold_chi2:
            S_diag = np.sqrt(np.maximum(np.diag(S), 1e-20))
            norm_delta = np.abs(e) / S_diag
            dom_idx   = int(np.argmax(norm_delta))
            dom_param = _PARAM_NAMES[dom_idx]
            direction = 'increased' if e[dom_idx] > 0 else 'decreased'
            interp    = _INTERPRETATIONS.get((dom_param, direction), f'{dom_param} {direction}')
            events.append({
                'timepoint':      t,
                'mahal':          float(mahal),
                'dominant_param': dom_param,
                'direction':      direction,
                'delta_log':      e.tolist(),
                'interpretation': interp,
            })
    return events








@app.route('/mc_temporal_analysis_stream', methods=['POST'])
def mc_temporal_analysis_stream():
    """
    Streaming temporal analysis via Server-Sent Events.

    DL path:
      - First timepoint: stratified SMC particles from MDN (N/K per component)
      - Subsequent timepoints: jitter particles with process noise, re-weight by likelihood
      - Full-spectrum impedance likelihood (vectorized, exact — no linearization)

    ECM path: L-BFGS-B warm-started from previous timepoint (always on if scipy available).

    Yields one JSON event per timepoint as it completes.

    Request body:
        sequences:    [{Z_real, Z_imag, frequencies, time_min}, ...]
        n_samples:    int   (default 500)
        include_ecm:  bool  (default true)
        use_sequential: bool (default true) — carry particles across timepoints
    """
    if MODEL is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data           = request.json
        sequences      = data.get('sequences', [])
        n_samples      = int(data.get('n_samples', 500))
        include_ecm    = bool(data.get('include_ecm', True)) and HAS_SCIPY
        n_display      = int(data.get('n_display', 15))
        use_sequential = bool(data.get('use_sequential', True))
        sample_id      = str(data.get('sample_id', 'unknown'))
        canonical_mode      = data.get('canonical_mode', None)   # e.g. 'Ra_gt_Rb' or None
        ground_truth_seed   = data.get('ground_truth_seed', None)  # {Ra,Rb,Ca,Cb,Rsh} in SI units
        model_override      = data.get('model_override', None)
        total               = len(sequences)

        # Resolve model context: use override if specified, else fall back to global model.
        if model_override and model_override != MODEL_NAME:
            _mctx = _load_model_ctx(model_override)
        else:
            _mctx = None
        _active_model    = _mctx['model']          if _mctx else MODEL
        _is_identifiable = _mctx['is_identifiable'] if _mctx else IS_IDENTIFIABLE

        # Validate and resample spectra to the model's expected frequency count.
        # The transformer has fixed positional structure (n_freq tokens); passing a
        # different length silently produces garbage output or crashes the attention layer.
        expected_n_freq = getattr(_active_model, 'config', None)
        expected_n_freq = expected_n_freq.n_freq if expected_n_freq is not None else N_FREQ
        _target_log_freqs = np.linspace(
            np.log10(FREQ_MIN_HZ), np.log10(FREQ_MAX_HZ), expected_n_freq
        )
        _target_freqs = 10.0 ** _target_log_freqs

        def _resample_spectrum(seq: dict) -> dict:
            """Resample Z_real/Z_imag to the model's frequency grid via log-linear interpolation."""
            f_in = np.array(seq['frequencies'])
            if len(f_in) == expected_n_freq and np.allclose(f_in, _target_freqs, rtol=1e-3):
                return seq
            log_f_in = np.log10(np.clip(f_in, 1e-30, None))
            Z_r_rs = np.interp(_target_log_freqs, log_f_in, np.array(seq['Z_real']))
            Z_i_rs = np.interp(_target_log_freqs, log_f_in, np.array(seq['Z_imag']))
            return {**seq, 'Z_real': Z_r_rs.tolist(), 'Z_imag': Z_i_rs.tolist(),
                    'frequencies': _target_freqs.tolist()}

        sequences = [_resample_spectrum(s) for s in sequences]

        # Resolve ATP span for mean-shift during temporal jitter
        _atp_span = None
        for _cfg in USSING_SAMPLES.values():
            if _cfg['meas_id'] in sample_id or sample_id in _cfg['meas_id']:
                _atp_span = _cfg['atp_span']
                break

        if total == 0:
            return jsonify({'error': 'Empty sequences'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 400

    def generate():
        ecm_warm_x0: np.ndarray | None = None
        ecm_warm_history: list[np.ndarray | None] = []
        time_history: list[float] = []

        _gpf = GaussianParticleFilter(n_particles=max(64, min(n_samples, 256)), noise_snr_db=40.0, n_iekf_iter=2)
        _gpf_initialized = False
        _gpf_raw_particles_history: list[np.ndarray] = []
        _gpf_weights_history:       list[np.ndarray] = []   # (N,) normalized weights per step
        _gpf_P_diag_history:        list[np.ndarray] = []   # (N, 5) diag(P_pp) per step
        _gpf_dt_history:            list[float]       = []  # Δt in minutes per step
        _prev_t_min: float = 0.0
        _ffbs_smoothed_w: list[np.ndarray] | None = None

        # Tracked across iterations for post-sequence admissibility computation
        _last_analytics: dict | None = None
        _last_frequencies: list | None = None

        try:
            for i, seq in enumerate(sequences):
                try:
                    t_min       = float(seq.get('time_min', i))
                    Z_real      = seq['Z_real']
                    Z_imag      = seq['Z_imag']
                    frequencies = seq['frequencies']
                    omega       = 2 * np.pi * np.array(frequencies)

                    t_dl = time.perf_counter()

                    # MDN inference — one forward pass per timepoint
                    Z_r, Z_i, log_omega = _prepare_input(Z_real, Z_imag, frequencies)
                    with torch.no_grad():
                        proposals = _active_model(Z_r, Z_i, log_omega)

                    mdn_log10        = _init_particles_from_mdn(proposals, 500, _mctx)
                    mdn_phys         = 10 ** mdn_log10
                    mdn_predictions  = _derived_stats(mdn_phys)
                    mdn_entropy      = float(np.mean(np.std(mdn_log10, axis=0)))

                    dl_ms = (time.perf_counter() - t_dl) * 1000.0

                    # ECM fit: cold at t=0, warm-started at subsequent steps
                    ecm_result = None
                    ecm_ms     = None
                    if include_ecm:
                        t_ecm = time.perf_counter()
                        if i == 0:
                            ecm_result, ecm_warm_x0 = _ecm_fit(Z_real, Z_imag, frequencies)
                            if ecm_warm_x0 is not None:
                                Ra, Rb, Ca, Cb, Rsh = 10.0 ** ecm_warm_x0
                                TER = (Rsh * (Ra + Rb)) / (Rsh + Ra + Rb + 1e-20)
                                TEC = (Ca * Cb) / (Ca + Cb + 1e-20)
                                _rn = float(_compute_dl_resnorm(ecm_warm_x0, Z_real, Z_imag, frequencies))
                                ecm_result = {
                                    'Ra': float(Ra / 1000), 'Rb': float(Rb / 1000),
                                    'Ca': float(Ca * 1e6),  'Cb': float(Cb * 1e6),
                                    'R1': float(min(Ra, Rb) / 1000),
                                    'C1': float(min(Ca, Cb) * 1e6),
                                    'R2': float(Rsh / 1000),
                                    'TER': float(TER / 1000),
                                    'TEC': float(TEC * 1e6),
                                    'resnorm': _rn,
                                }
                        else:
                            ecm_result, ecm_warm_x0 = _ecm_fit(
                                Z_real, Z_imag, frequencies, x0_warm=ecm_warm_x0
                            )
                            if ecm_result is not None and ecm_warm_x0 is not None:
                                Ra, Rb, Ca, Cb, Rsh = 10.0 ** ecm_warm_x0
                                TER = (Rsh * (Ra + Rb)) / (Rsh + Ra + Rb + 1e-20)
                                TEC = (Ca * Cb) / (Ca + Cb + 1e-20)
                                ecm_result['Ra']  = float(Ra / 1000)
                                ecm_result['Rb']  = float(Rb / 1000)
                                ecm_result['Ca']  = float(Ca * 1e6)
                                ecm_result['Cb']  = float(Cb * 1e6)
                                ecm_result['R1']  = float(min(Ra, Rb) / 1000)
                                ecm_result['C1']  = float(min(Ca, Cb) * 1e6)
                                ecm_result['R2']  = float(Rsh / 1000)
                                ecm_result['TER'] = float(TER / 1000)
                                ecm_result['TEC'] = float(TEC * 1e6)
                        ecm_ms = (time.perf_counter() - t_ecm) * 1000.0

                    ecm_warm_history.append(ecm_warm_x0.copy() if ecm_warm_x0 is not None else None)
                    time_history.append(t_min)

                    # GPF step
                    _gpf_diag: dict | None = None
                    _dt_gpf = max(float(t_min - _prev_t_min), 1e-3) if i > 0 else 1.0
                    try:
                        if not _gpf_initialized:
                            _mdn_means_id = None
                            _mdn_weights  = None
                            if _is_identifiable and proposals is not None:
                                try:
                                    _mdn_means_id = proposals['means'][0].cpu().numpy()
                                    _mdn_weights  = proposals['weights'][0].cpu().numpy()
                                except Exception:
                                    pass
                            _gpf.initialize(
                                ecm_log10=ecm_warm_x0,
                                mdn_samples=mdn_log10,
                                mdn_means_id=_mdn_means_id,
                                mdn_weights=_mdn_weights,
                            )
                            _gpf_initialized = True
                            _gpf_raw_particles_history.append(_gpf.get_particles())
                            _gpf_weights_history.append(_gpf.get_weights())
                            _gpf_P_diag_history.append(np.diagonal(_gpf.P_pp, axis1=1, axis2=2).copy())
                            _gpf_dt_history.append(_dt_gpf)
                        else:
                            _gpf_pts, _ = _gpf.step(
                                np.array(Z_real, dtype=np.float64),
                                np.array(Z_imag, dtype=np.float64),
                                omega, dt=_dt_gpf,
                            )
                            _gpf_raw_particles_history.append(_gpf_pts)
                            _gpf_weights_history.append(_gpf.get_weights())
                            _gpf_P_diag_history.append(np.diagonal(_gpf.P_pp, axis1=1, axis2=2).copy())
                            _gpf_dt_history.append(_dt_gpf)
                        _gpf_diag = _gpf.get_diagnostics()
                        _prev_t_min = t_min
                    except Exception:
                        pass

                    # Per-step predictions from GPF cloud, falling back to MDN
                    if _gpf_raw_particles_history:
                        _display_phys = 10 ** _gpf_raw_particles_history[-1]
                    else:
                        _display_phys = mdn_phys
                    predictions = _derived_stats(_display_phys)

                    # DL resnorm: best GPF particle
                    dl_resnorm = None
                    try:
                        _pts = _gpf_raw_particles_history[-1] if _gpf_raw_particles_history else mdn_log10
                        valid_mask = np.all(np.isfinite(_pts), axis=1)
                        if valid_mask.any():
                            dl_resnorm = float(np.min(
                                _compute_dl_resnorm_batch(_pts[valid_mask], Z_real, Z_imag, frequencies)
                            ))
                    except Exception:
                        pass

                    attribution = _cohens_d(10 ** _gpf_raw_particles_history[-2], _display_phys) \
                        if len(_gpf_raw_particles_history) >= 2 else None
                    analytics   = _get_analytics(Z_real, Z_imag, frequencies, proposals=proposals)
                    if analytics is not None:
                        _last_analytics   = analytics
                        _last_frequencies = frequencies

                    payload = {
                        'index':           i,
                        'total':           total,
                        'time_min':        t_min,
                        'predictions':     predictions,
                        'mdn_predictions': mdn_predictions,
                        'mdn_entropy':     mdn_entropy,
                        'ecm':             ecm_result,
                        'analytics':       analytics,
                        'dl_resnorm':      dl_resnorm,
                        'gpf_diag':       _gpf_diag,
                        'attribution':     attribution,
                        'timing': {
                            'dl_ms':  round(dl_ms, 1),
                            'ecm_ms': round(ecm_ms, 1) if ecm_ms is not None else None,
                        },
                        'done': False,
                    }
                    yield f"data: {json.dumps(payload, cls=_SafeEncoder)}\n\n"

                except GeneratorExit:
                    return
                except Exception as exc:
                    import traceback
                    yield f"data: {json.dumps({'index': i, 'total': total, 'error': str(exc), 'traceback': traceback.format_exc(), 'done': False})}\n\n"
                    continue

        except GeneratorExit:
            return

        # --- Post-sequence: GPF Kalman smoother ---
        kal_gpf_result = None
        _gpf_full_changepoints: list[dict] = []
        if len(_gpf_raw_particles_history) >= 3:
            try:
                _gpf_kal_pass1 = _kalman_smooth_identifiable(
                    _gpf_raw_particles_history,
                    time_min=time_history,
                    n_predict_steps=0,
                    use_raw_space=False,
                )
                _raw_cps = [
                    cp for cp in _detect_changepoints(
                        _gpf_kal_pass1['innovations'], _gpf_kal_pass1['innov_cov']
                    )
                    if cp['timepoint'] < len(time_history)
                ]
                _gpf_cp_times = [float(time_history[cp['timepoint']]) for cp in _raw_cps]
                _gpf_full_changepoints = [
                    {**cp, 'time_min': float(time_history[cp['timepoint']])}
                    for cp in _raw_cps
                ]
                _gpf_kal = _kalman_smooth_identifiable(
                    _gpf_raw_particles_history,
                    time_min=time_history,
                    n_predict_steps=15,
                    predict_dt_min=1.0,
                    use_raw_space=False,
                    changepoint_times=_gpf_cp_times,
                )
                # Raw-space smoother: operates directly on log10 [Ra, Rb, Ca, Cb, Rsh].
                # With ordering enforced, particles are unimodal — raw smoother gives
                # individual Ra, Rb, Ca, Cb, tau_a, tau_b estimates.
                _gpf_kal_raw = _kalman_smooth_identifiable(
                    _gpf_raw_particles_history,
                    time_min=time_history,
                    n_predict_steps=0,
                    use_raw_space=True,
                    changepoint_times=_gpf_cp_times,
                )

                kal_gpf_result = {
                    'time_min':         time_history,
                    'param_names':      ['tau_big', 'tau_small', 'TER', 'TEC', 'Rsh'],
                    'mu_filtered':      _gpf_kal['mu_filtered'],
                    'std_filtered':     _gpf_kal['std_filtered'],
                    'mu_smoothed':      _gpf_kal['mu_smoothed'],
                    'std_smoothed':     _gpf_kal['std_smoothed'],
                    'mu_vel_smoothed':  _gpf_kal['mu_vel_smoothed'],
                    'std_vel_smoothed': _gpf_kal['std_vel_smoothed'],
                    't_grid':           _gpf_kal['t_grid'],
                    'mu_grid':          _gpf_kal['mu_grid'],
                    'std_grid':         _gpf_kal['std_grid'],
                    'trends':           _gpf_kal['trends'],
                    'geometry':         _gpf_kal['geometry'],
                    # Raw-space smooth: (T, 5) log10 [Ra, Rb, Ca, Cb, Rsh]
                    'param_names_raw':  ['Ra', 'Rb', 'Ca', 'Cb', 'Rsh'],
                    'mu_smoothed_raw':  _gpf_kal_raw['mu_smoothed'],
                    'std_smoothed_raw': _gpf_kal_raw['std_smoothed'],
                }
            except Exception:
                pass

        # --- ECM warm trajectory (unsmoothed + 1D-RTS smoothed) ---
        ecm_raw_path = []
        cluster_paths = []
        valid_warm = [(idx, x) for idx, x in enumerate(ecm_warm_history) if x is not None]
        if len(valid_warm) >= 3:
            vidx     = [idx for idx, _ in valid_warm]
            warm_arr = np.stack([x for _, x in valid_warm])
            # Project to Ra > Rb before smoothing to avoid mode-flip discontinuities
            for k in range(len(warm_arr)):
                if warm_arr[k, 0] < warm_arr[k, 1]:
                    warm_arr[k, 0], warm_arr[k, 1] = warm_arr[k, 1], warm_arr[k, 0]
                    warm_arr[k, 2], warm_arr[k, 3] = warm_arr[k, 3], warm_arr[k, 2]

            for k, t_idx in enumerate(vidx):
                dm = _derived_from_log10(warm_arr[k])
                ecm_raw_path.append({
                    'time_min':  float(time_history[t_idx]),
                    'TER':       float(dm['TER']),   'TEC':       float(dm['TEC']),
                    'tau_big':   float(dm['tau_big']), 'tau_small': float(dm['tau_small']),
                    'Rsh':       float(dm['Rsh']),
                    'Ra':        float(dm['Ra']),    'Rb':        float(dm['Rb']),
                    'Ca':        float(dm['Ca']),    'Cb':        float(dm['Cb']),
                })

            warm_arr_s = np.stack([_rts_smooth_1d(warm_arr[:, d]) for d in range(5)], axis=1)
            warm_pts = []
            for k, t_idx in enumerate(vidx):
                dm = _derived_from_log10(warm_arr_s[k])
                warm_pts.append({
                    't': t_idx, 'time_min': float(time_history[t_idx]),
                    'TER':       dm['TER'],   'TER_q25':       dm['TER'],   'TER_q75':       dm['TER'],
                    'TEC':       dm['TEC'],   'TEC_q25':       dm['TEC'],   'TEC_q75':       dm['TEC'],
                    'tau_big':   dm['tau_big'], 'tau_big_q25': dm['tau_big'], 'tau_big_q75': dm['tau_big'],
                    'tau_small': dm['tau_small'], 'tau_small_q25': dm['tau_small'], 'tau_small_q75': dm['tau_small'],
                    'Rsh':       dm['Rsh'],   'Rsh_q25':       dm['Rsh'],   'Rsh_q75':       dm['Rsh'],
                    'Ra':        dm['Ra'],    'Ra_q25':        dm['Ra'],    'Ra_q75':        dm['Ra'],
                    'Rb':        dm['Rb'],    'Rb_q25':        dm['Rb'],    'Rb_q75':        dm['Rb'],
                    'Ca':        dm['Ca'],    'Ca_q25':        dm['Ca'],    'Ca_q75':        dm['Ca'],
                    'Cb':        dm['Cb'],    'Cb_q25':        dm['Cb'],    'Cb_q75':        dm['Cb'],
                })
            cluster_paths = [{
                'rank': 0, 'probability': 1.0, 'hypothesis': 0,
                'label': 'ECM (smoothed)', 'path': warm_pts,
            }]

        # --- GPF (smoothed) trajectory via FFBS backward sweep ---
        # Replaces the causal GPF path: same particles, backward-informed weights.
        if len(_gpf_raw_particles_history) >= 2 and len(_gpf_weights_history) == len(_gpf_raw_particles_history):
            try:
                smoothed_w = ffbs_backward_sweep(
                    _gpf_raw_particles_history,
                    _gpf_weights_history,
                    _gpf_P_diag_history,
                    _gpf_dt_history,
                )
                _ffbs_smoothed_w = smoothed_w

                def _wm(arr, w):
                    return float(np.average(arr, weights=w))

                def _wq(arr, w, q):
                    idx     = np.argsort(arr)
                    cum_w   = np.cumsum(w[idx])
                    total_w = cum_w[-1]
                    if total_w <= 0:
                        return float(np.percentile(arr, q))
                    pos = np.searchsorted(cum_w, q / 100.0 * total_w)
                    return float(arr[idx[min(pos, len(arr) - 1)]])

                gpf_pts = []
                for step_idx, pts_log10 in enumerate(_gpf_raw_particles_history):
                    finite_mask = np.all(np.isfinite(pts_log10), axis=1)
                    pts_log10   = pts_log10[finite_mask]
                    s_w         = smoothed_w[step_idx][finite_mask]
                    if len(pts_log10) == 0 or s_w.sum() <= 0:
                        continue
                    s_w = s_w / s_w.sum()

                    phys = 10.0 ** pts_log10
                    Ra_s, Rb_s, Ca_s, Cb_s, Rsh_s = phys[:,0], phys[:,1], phys[:,2], phys[:,3], phys[:,4]
                    TER_s       = Rsh_s * (Ra_s + Rb_s) / (Rsh_s + Ra_s + Rb_s + 1e-20)
                    TEC_s       = Ca_s * Cb_s / (Ca_s + Cb_s + 1e-20)
                    tau_big_s   = np.maximum(Ra_s * Ca_s, Rb_s * Cb_s)
                    tau_small_s = np.minimum(Ra_s * Ca_s, Rb_s * Cb_s)

                    t_min_val = float(time_history[step_idx]) if step_idx < len(time_history) else float(step_idx)
                    gpf_pts.append({
                        't': step_idx, 'time_min': t_min_val,
                        'TER':       _wm(TER_s, s_w),       'TER_q25':       _wq(TER_s, s_w, 25),       'TER_q75':       _wq(TER_s, s_w, 75),       'TER_q05':       _wq(TER_s, s_w, 5),       'TER_q95':       _wq(TER_s, s_w, 95),
                        'TEC':       _wm(TEC_s, s_w),       'TEC_q25':       _wq(TEC_s, s_w, 25),       'TEC_q75':       _wq(TEC_s, s_w, 75),       'TEC_q05':       _wq(TEC_s, s_w, 5),       'TEC_q95':       _wq(TEC_s, s_w, 95),
                        'tau_big':   _wm(tau_big_s, s_w),   'tau_big_q25':   _wq(tau_big_s, s_w, 25),   'tau_big_q75':   _wq(tau_big_s, s_w, 75),   'tau_big_q05':   _wq(tau_big_s, s_w, 5),   'tau_big_q95':   _wq(tau_big_s, s_w, 95),
                        'tau_small': _wm(tau_small_s, s_w), 'tau_small_q25': _wq(tau_small_s, s_w, 25), 'tau_small_q75': _wq(tau_small_s, s_w, 75), 'tau_small_q05': _wq(tau_small_s, s_w, 5), 'tau_small_q95': _wq(tau_small_s, s_w, 95),
                        'Rsh':       _wm(Rsh_s, s_w),       'Rsh_q25':       _wq(Rsh_s, s_w, 25),       'Rsh_q75':       _wq(Rsh_s, s_w, 75),       'Rsh_q05':       _wq(Rsh_s, s_w, 5),       'Rsh_q95':       _wq(Rsh_s, s_w, 95),
                        'Ra':        _wm(Ra_s, s_w),        'Ra_q25':        _wq(Ra_s, s_w, 25),        'Ra_q75':        _wq(Ra_s, s_w, 75),        'Ra_q05':        _wq(Ra_s, s_w, 5),        'Ra_q95':        _wq(Ra_s, s_w, 95),
                        'Rb':        _wm(Rb_s, s_w),        'Rb_q25':        _wq(Rb_s, s_w, 25),        'Rb_q75':        _wq(Rb_s, s_w, 75),        'Rb_q05':        _wq(Rb_s, s_w, 5),        'Rb_q95':        _wq(Rb_s, s_w, 95),
                        'Ca':        _wm(Ca_s, s_w),        'Ca_q25':        _wq(Ca_s, s_w, 25),        'Ca_q75':        _wq(Ca_s, s_w, 75),        'Ca_q05':        _wq(Ca_s, s_w, 5),        'Ca_q95':        _wq(Ca_s, s_w, 95),
                        'Cb':        _wm(Cb_s, s_w),        'Cb_q25':        _wq(Cb_s, s_w, 25),        'Cb_q75':        _wq(Cb_s, s_w, 75),        'Cb_q05':        _wq(Cb_s, s_w, 5),        'Cb_q95':        _wq(Cb_s, s_w, 95),
                    })
                if gpf_pts:
                    cluster_paths.append({
                        'rank': 1, 'probability': 1.0, 'hypothesis': 1,
                        'label': 'GPF (smoothed)', 'path': gpf_pts,
                    })
            except Exception:
                pass

        # --- Mechanistic Admissibility ---
        admissibility_result = None
        if kal_gpf_result is not None:
            try:
                from mechanistic_admissibility import compute_admissibility
                last_particles_log10 = (
                    np.mean(_gpf_raw_particles_history[-1], axis=0)
                    if _gpf_raw_particles_history else None
                )
                admissibility_result = compute_admissibility(
                    kal_gpf_result,
                    _gpf_full_changepoints,
                    _last_analytics,
                    last_particles_log10,
                    _last_frequencies,
                )
            except Exception:
                pass

        # --- Posterior summary from final FFBS timepoint ---
        posterior_summary = None
        if _ffbs_smoothed_w is not None and _gpf_raw_particles_history:
            try:
                def _posterior_summary(arr, w):
                    return {
                        'median': _wq(arr, w, 50),
                        'q05':    _wq(arr, w, 5),
                        'q95':    _wq(arr, w, 95),
                        'identifiability': float(np.clip(
                            1.0 - np.sqrt(np.average((arr - np.average(arr, weights=w))**2, weights=w)) / (np.std(arr) + 1e-30),
                            0.0, 1.0
                        )),
                    }

                pts_log10_last = _gpf_raw_particles_history[-1]
                s_w_last = _ffbs_smoothed_w[-1]
                finite_mask = np.all(np.isfinite(pts_log10_last), axis=1)
                pts_log10_last = pts_log10_last[finite_mask]
                s_w_last = s_w_last[finite_mask]
                if s_w_last.sum() > 0:
                    s_w_last = s_w_last / s_w_last.sum()
                    phys_last = 10.0 ** pts_log10_last
                    Ra_l, Rb_l, Ca_l, Cb_l, Rsh_l = phys_last[:,0], phys_last[:,1], phys_last[:,2], phys_last[:,3], phys_last[:,4]
                    TER_l = Rsh_l * (Ra_l + Rb_l) / (Rsh_l + Ra_l + Rb_l + 1e-20)
                    TEC_l = Ca_l * Cb_l / (Ca_l + Cb_l + 1e-20)
                    posterior_summary = {
                        'TER': _posterior_summary(TER_l / 1000, s_w_last),
                        'TEC': _posterior_summary(TEC_l * 1e6, s_w_last),
                        'Rsh': _posterior_summary(Rsh_l / 1000, s_w_last),
                        'Ra':  _posterior_summary(Ra_l / 1000, s_w_last),
                        'Rb':  _posterior_summary(Rb_l / 1000, s_w_last),
                    }
            except Exception:
                pass

        # --- Mechanism / hypothesis analysis ---
        mechanism = None
        if _ffbs_smoothed_w is not None and len(_gpf_raw_particles_history) >= 3:
            try:
                def _ffbs_weighted_mean_traj(param_key):
                    means = []
                    for step_idx in range(len(_gpf_raw_particles_history)):
                        pts = _gpf_raw_particles_history[step_idx]
                        sw = _ffbs_smoothed_w[step_idx]
                        finite = np.all(np.isfinite(pts), axis=1)
                        pts = pts[finite]; sw = sw[finite]
                        if sw.sum() <= 0:
                            means.append(np.nan)
                            continue
                        sw = sw / sw.sum()
                        phys = 10.0 ** pts
                        if param_key == 'Ra':
                            arr = phys[:, 0]
                        elif param_key == 'Rb':
                            arr = phys[:, 1]
                        else:
                            arr = phys[:, 4]
                        means.append(float(np.average(np.log10(np.maximum(arr, 1e-30)), weights=sw)))
                    return np.array(means)

                T_mech = len(_gpf_raw_particles_history)
                cutoff = max(1, int(T_mech * 0.7))
                tail_idx = np.arange(cutoff, T_mech)

                t_tail = np.array(time_history[cutoff:]) if len(time_history) > cutoff else np.arange(len(tail_idx), dtype=float)
                if len(t_tail) < 2:
                    raise ValueError('not enough timepoints for slope')

                def _slope_log10_per_hr(traj):
                    y = traj[cutoff:]
                    valid = np.isfinite(y)
                    if valid.sum() < 2:
                        return 0.0
                    x = t_tail[valid]
                    y = y[valid]
                    xc = x - x.mean()
                    denom = float(np.dot(xc, xc)) + 1e-10
                    slope_per_min = float(np.dot(xc, y) / denom)
                    return slope_per_min * 60.0

                traj_Ra  = _ffbs_weighted_mean_traj('Ra')
                traj_Rb  = _ffbs_weighted_mean_traj('Rb')
                traj_Rsh = _ffbs_weighted_mean_traj('Rsh')

                slope_Ra  = _slope_log10_per_hr(traj_Ra)
                slope_Rb  = _slope_log10_per_hr(traj_Rb)
                slope_Rsh = _slope_log10_per_hr(traj_Rsh)

                def _gauss_score(x, mu, sigma):
                    return float(np.exp(-0.5 * ((x - mu) / sigma) ** 2))

                score_barrier    = _gauss_score(slope_Rb, -0.1, 0.05) * _gauss_score(slope_Ra, 0.0, 0.05) * _gauss_score(slope_Rsh, 0.0, 0.05)
                score_paracell   = _gauss_score(slope_Rsh, -0.15, 0.05) * _gauss_score(slope_Ra, 0.0, 0.05) * _gauss_score(slope_Rb, 0.0, 0.05)
                score_apical     = _gauss_score(slope_Ra, -0.1, 0.05) * _gauss_score(slope_Rb, 0.0, 0.05)

                total_score = score_barrier + score_paracell + score_apical + 1e-30
                hypotheses = [
                    {'name': 'barrier_breakdown',  'label': 'Basolateral barrier breakdown',  'probability': float(score_barrier  / total_score)},
                    {'name': 'paracellular_leak',   'label': 'Paracellular leak (tight junction)', 'probability': float(score_paracell / total_score)},
                    {'name': 'apical_failure',      'label': 'Apical membrane failure',        'probability': float(score_apical   / total_score)},
                ]
                hypotheses.sort(key=lambda h: h['probability'], reverse=True)

                n_windows = 5
                win_size = max(1, T_mech // n_windows)
                t_arr_full = np.array(time_history) if time_history else np.arange(T_mech, dtype=float)

                low_freq_series = []
                for step_idx in range(T_mech):
                    pts = _gpf_raw_particles_history[step_idx]
                    sw = _ffbs_smoothed_w[step_idx]
                    finite = np.all(np.isfinite(pts), axis=1)
                    pts = pts[finite]; sw = sw[finite]
                    if sw.sum() <= 0:
                        low_freq_series.append(0.0)
                        continue
                    sw = sw / sw.sum()
                    phys = 10.0 ** pts
                    Rsh_w = phys[:, 4]
                    Ra_w = phys[:, 0]; Rb_w = phys[:, 1]
                    TER_w = Rsh_w * (Ra_w + Rb_w) / (Rsh_w + Ra_w + Rb_w + 1e-20)
                    low_freq_series.append(float(np.average(np.log10(np.maximum(TER_w, 1e-30)), weights=sw)))

                low_freq_arr = np.array(low_freq_series)
                evidence_windows = []
                for wi in range(n_windows):
                    lo = wi * win_size
                    hi = min(lo + win_size, T_mech)
                    seg = low_freq_arr[lo:hi]
                    contribution = float(np.std(seg)) if len(seg) > 1 else 0.0
                    t_start = float(t_arr_full[lo]) if lo < len(t_arr_full) else float(lo)
                    t_end   = float(t_arr_full[hi - 1]) if hi - 1 < len(t_arr_full) else float(hi - 1)
                    evidence_windows.append({'t_start': t_start, 't_end': t_end, 'label': f'window_{wi}', 'contribution': contribution})

                evidence_windows.sort(key=lambda w: w['contribution'], reverse=True)
                evidence_windows = evidence_windows[:3]

                established = []
                ambiguous = []
                if posterior_summary is not None:
                    for pname, psumm in posterior_summary.items():
                        ident = psumm.get('identifiability', 0.0)
                        if ident > 0.7:
                            established.append(pname)
                        elif ident < 0.4:
                            ambiguous.append(pname)

                mechanism = {
                    'hypotheses':    hypotheses,
                    'evidence_time': evidence_windows,
                    'established':   established,
                    'ambiguous':     ambiguous,
                }
            except Exception:
                pass

        done_payload = {
            'done':              True,
            'ecm_cold_path':     ecm_raw_path,
            'cluster_paths':     cluster_paths,
            'kal_gpf':          kal_gpf_result,
            'kalman':            kal_gpf_result,   # frontend parses 'kalman' key
            'changepoints':      _gpf_full_changepoints,
            'admissibility':     admissibility_result,
            'posterior_summary': posterior_summary,
            'mechanism':         mechanism,
        }
        yield f"data: {json.dumps(done_payload, cls=_SafeEncoder)}\n\n"

    return Response(
        stream_with_context(generate()),
        content_type='text/event-stream',
        headers={
            'X-Accel-Buffering': 'no',
            'Cache-Control':     'no-cache',
            'Access-Control-Allow-Origin': '*',
        },
    )




@app.route('/foveated_explore', methods=['POST'])
def foveated_explore():
    """
    MDN-guided parameter exploration.

    Accepts the same spectrum as predict_single plus optional foveation_config
    and mdn_predictions hints. Runs ECM fitting for the best-fit parameters
    and MDN inference for the posterior, returning the shape expected by
    FoveatedExplorationTab.

    Request body (lowercase z keys accepted):
        frequencies, z_real / Z_real, z_imag / Z_imag
        foveation_config: ignored (for UI compatibility)
        mdn_predictions:  optional hints, echoed back if inference unavailable
        resnorm_method:   ignored (ECM always uses MAE internally)
    """
    if MODEL is None:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        import time as _time
        t0 = _time.time()

        data  = request.json
        # Accept both lower-case (frontend) and upper-case (internal) key names
        Z_real = data.get('Z_real', data.get('z_real', []))
        Z_imag = data.get('Z_imag', data.get('z_imag', []))
        freqs  = data.get('frequencies', [])
        n_samples = int(data.get('n_samples', 1000))

        # MDN posterior
        samples_log10, mu_log10, std_log10 = _run_inference(
            Z_real, Z_imag, freqs, n_samples=n_samples
        )
        phys = 10.0 ** samples_log10   # (N, 5)

        # ECM best fit, warm-started from MDN mean
        ecm_x0 = mu_log10.copy()
        _, ecm_log10 = _ecm_fit(Z_real, Z_imag, freqs, x0_warm=ecm_x0, n_random_starts=8)

        if ecm_log10 is not None:
            Ra, Rb, Ca, Cb, Rsh = 10.0 ** ecm_log10
            best_resnorm = float(_compute_dl_resnorm(ecm_log10, Z_real, Z_imag, freqs))
        else:
            Ra, Rb, Ca, Cb, Rsh = 10.0 ** mu_log10
            best_resnorm = float('inf')

        # Per-param posterior stats from MDN samples
        param_names = ['Ra', 'Rb', 'Ca', 'Cb', 'Rsh']
        mdn_predictions = {}
        for i, name in enumerate(param_names):
            col = phys[:, i]
            mdn_predictions[name] = {
                'mean': float(np.mean(col)),
                'std':  float(np.std(col)),
            }

        elapsed = _time.time() - t0

        return jsonify({
            'best_parameters': {
                'Ra': float(Ra), 'Rb': float(Rb),
                'Ca': float(Ca), 'Cb': float(Cb), 'Rsh': float(Rsh),
            },
            'best_resnorm': best_resnorm,
            'mdn_predictions': mdn_predictions,
            'exploration_stages': {},
            'performance_metrics': {
                'total_time_seconds':    elapsed,
                'total_evaluations':     n_samples,
                'equivalent_uniform_grid': int(n_samples ** (1/5)),
                'speedup_factor':        1.0,
            },
            'all_results': [],
            'n_total_results': n_samples,
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/explore_constraints', methods=['POST'])
def explore_constraints():
    """
    Constraint-based parameter exploration.
    Samples from the MDN posterior, applies physical constraints, and returns
    uncertainty statistics in identifiable space before and after filtering.

    Request body:
        Z_real, Z_imag, frequencies: spectrum
        n_samples:    int (default 3000)
        constraints:  dict with optional keys:
            ra_rb_ratio_min/max  (float)
            ca_cb_ratio_min/max  (float)
            ter_min/max          (float, Ohm)
            canonical_mode       (str | null) — 'Ra_gt_Rb' or null
    """
    if MODEL is None:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        data        = request.json
        n_samples   = int(data.get('n_samples', 3000))
        constraints = data.get('constraints', {})
        canon       = constraints.get('canonical_mode', None)

        samples_log10, _, _ = _run_inference(
            data['Z_real'], data['Z_imag'], data['frequencies'], n_samples=n_samples
        )
        if canon is not None:
            samples_log10 = _enforce_canonical_mode(samples_log10, canon)

        phys = 10.0 ** samples_log10
        Ra, Rb, Ca, Cb, Rsh = phys[:,0], phys[:,1], phys[:,2], phys[:,3], phys[:,4]
        TER = Rsh * (Ra + Rb) / (Rsh + Ra + Rb + 1e-20)
        TEC = Ca * Cb / (Ca + Cb + 1e-20)

        mask = np.ones(len(phys), dtype=bool)
        ra_rb_min = constraints.get('ra_rb_ratio_min', 0.0)
        ra_rb_max = constraints.get('ra_rb_ratio_max', 1e9)
        ca_cb_min = constraints.get('ca_cb_ratio_min', 0.0)
        ca_cb_max = constraints.get('ca_cb_ratio_max', 1e9)
        ter_min   = constraints.get('ter_min', 0.0)
        ter_max   = constraints.get('ter_max', 1e9)

        ratio_RaRb = Ra / (Rb + 1e-20)
        ratio_CaCb = Ca / (Cb + 1e-20)
        mask &= (ratio_RaRb >= ra_rb_min) & (ratio_RaRb <= ra_rb_max)
        mask &= (ratio_CaCb >= ca_cb_min) & (ratio_CaCb <= ca_cb_max)
        mask &= (TER >= ter_min) & (TER <= ter_max)

        samples_id = _to_identifiable_log10(samples_log10)   # (N, 5)

        def _stats(arr: np.ndarray) -> dict:
            return {
                'mean': float(np.mean(arr)),
                'std':  float(np.std(arr)),
                'q05':  float(np.percentile(arr, 5)),
                'q95':  float(np.percentile(arr, 95)),
            }

        id_names = ['tau_big', 'tau_small', 'TER', 'TEC', 'Rsh']
        unc_id = {id_names[d]: _stats(10.0 ** samples_id[:, d]) for d in range(5)}

        constrained = None
        corr_constrained = None
        if mask.sum() >= 10:
            samples_id_c = samples_id[mask]
            constrained = {
                'n_samples':     int(mask.sum()),
                'retention_pct': float(mask.mean() * 100),
                'identifiable':  {id_names[d]: _stats(10.0 ** samples_id_c[:, d]) for d in range(5)},
            }
            corr_constrained = np.corrcoef(samples_id_c.T).tolist()

        return jsonify({
            'unconstrained': {
                'n_samples':    int(len(phys)),
                'identifiable': unc_id,
            },
            'constrained': constrained,
            'correlation_matrix': {
                'labels':        id_names,
                'unconstrained': np.corrcoef(samples_id.T).tolist(),
                'constrained':   corr_constrained,
            },
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/predict_pipeline', methods=['POST'])
def predict_pipeline():
    """
    Run the full EIS preprocessing pipeline and return each stage for visualization.

    Stages returned:
      raw       : original Z_real, Z_imag + derived log_mag, phase
      smoothed  : Gaussian-smoothed (σ=0.5 decade) spectrum
      gradient_features : d(log|Z|)/d(logω), dφ/d(logω) and 2nd derivatives
      drt       : Distribution of Relaxation Times γ(τ) via Tikhonov ridge
      proposals : MDN component means/stds in model parameter space
      prediction: mixture mean ± std (log10)
      pooling_weights: attention pooling weights over the frequency grid

    Request body: { frequencies, Z_real, Z_imag }
    """
    if MODEL is None:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        import scipy.ndimage as _ndimage

        data = request.json
        Z_real_raw = np.array(data.get('Z_real', data.get('z_real', [])), dtype=np.float64)
        Z_imag_raw = np.array(data.get('Z_imag', data.get('z_imag', [])), dtype=np.float64)
        freqs = np.array(data.get('frequencies', []), dtype=np.float64)

        if len(freqs) == 0:
            return jsonify({'error': 'No frequency data provided'}), 400

        omega = 2.0 * np.pi * freqs
        log_omega = np.log10(omega)
        n_freq = len(freqs)
        spacing = (log_omega[-1] - log_omega[0]) / max(n_freq - 1, 1)

        # ── Stage 1: Gaussian smoothing in log-frequency space ──────────────
        smooth_sigma_decades = 0.5
        sigma_idx = smooth_sigma_decades / spacing if abs(spacing) > 1e-10 else 7.0
        Z_real_sm = _ndimage.gaussian_filter1d(Z_real_raw, sigma=sigma_idx, mode='reflect')
        Z_imag_sm = _ndimage.gaussian_filter1d(Z_imag_raw, sigma=sigma_idx, mode='reflect')

        log_Z_mag_raw = np.log10(np.sqrt(Z_real_raw ** 2 + Z_imag_raw ** 2).clip(1e-10))
        Z_phase_raw   = np.degrees(np.arctan2(Z_imag_raw, Z_real_raw))
        log_Z_mag_sm  = np.log10(np.sqrt(Z_real_sm ** 2 + Z_imag_sm ** 2).clip(1e-10))
        Z_phase_sm    = np.degrees(np.arctan2(Z_imag_sm, Z_real_sm))

        # ── Stage 2: Gradient features on smoothed spectrum ─────────────────
        def _central_diff(x: np.ndarray) -> np.ndarray:
            d = np.empty_like(x)
            d[1:-1] = (x[2:] - x[:-2]) / 2.0
            d[0]    = x[1]  - x[0]
            d[-1]   = x[-1] - x[-2]
            return d

        d_log_mag  = _central_diff(log_Z_mag_sm)
        d_phase    = _central_diff(Z_phase_sm)
        d2_log_mag = _central_diff(d_log_mag)
        d2_phase   = _central_diff(d_phase)

        # ── Stage 3: DRT via Tikhonov ridge ─────────────────────────────────
        tau_m = 1.0 / omega                              # τ_m aligned to ω_k
        log_tau = np.log10(tau_m)

        ot     = np.outer(omega, tau_m)                  # (N, N)
        A_real = 1.0 / (1.0 + ot ** 2)
        A_imag = -ot / (1.0 + ot ** 2)
        A      = np.vstack([A_real, A_imag])             # (2N, N)

        idx_d = np.arange(n_freq - 1)
        L = np.zeros((n_freq - 1, n_freq))
        L[idx_d, idx_d]     = -1.0
        L[idx_d, idx_d + 1] =  1.0

        drt_lambda = 1e-3
        system  = A.T @ A + drt_lambda * (L.T @ L)
        M_solve = np.linalg.solve(system, A.T)           # (N, 2N)

        Z_inf   = float(Z_real_sm[-1])
        delta_R = max(float(Z_real_sm[0]) - Z_inf, 1.0)
        b = np.concatenate([(Z_real_sm - Z_inf) / delta_R,
                            Z_imag_sm / delta_R])

        gamma = M_solve @ b
        gamma = np.clip(gamma, 0.0, None)
        g_max = float(np.max(gamma))
        if g_max > 0:
            gamma = gamma / g_max

        # ── Stage 4: MDN inference ───────────────────────────────────────────
        Z_r_t       = torch.tensor(Z_real_raw, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        Z_i_t       = torch.tensor(Z_imag_raw, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        log_omega_t = torch.tensor(log_omega,  dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            proposals_raw = MODEL(Z_r_t, Z_i_t, log_omega_t)
            analytics     = MODEL.get_last_attention_and_pooling()
            mu_t, cov_t   = MODEL.get_mixture_posterior(proposals_raw)

        weights_np = proposals_raw['weights'][0].cpu().numpy()       # (K,)
        means_np   = proposals_raw['means'][0].cpu().numpy()         # (K, 5)
        covs_np    = proposals_raw['covs'][0].cpu().numpy()          # (K, 5, 5)
        mu_np      = mu_t[0].cpu().numpy()                           # (5,)
        cov_np     = cov_t[0].cpu().numpy()                          # (5, 5)
        std_np     = np.sqrt(np.diag(cov_np))

        param_names = (
            ['tau_big', 'tau_small', 'TER', 'TEC', 'Rsh'] if IS_IDENTIFIABLE
            else ['Ra', 'Rb', 'Ca', 'Cb', 'Rsh']
        )

        proposal_list = []
        for k in range(len(weights_np)):
            stds_k = np.sqrt(np.maximum(np.diag(covs_np[k]), 0.0))
            proposal_list.append({
                'weight': float(weights_np[k]),
                'means':  {param_names[i]: float(means_np[k, i]) for i in range(len(param_names))},
                'stds':   {param_names[i]: float(stds_k[i])      for i in range(len(param_names))},
            })

        prediction = {param_names[i]: float(mu_np[i])  for i in range(len(param_names))}
        prediction.update({f'{param_names[i]}_std': float(std_np[i]) for i in range(len(param_names))})

        return jsonify({
            'frequencies':    freqs.tolist(),
            'log_omega':      log_omega.tolist(),
            'raw': {
                'Z_real':  Z_real_raw.tolist(),
                'Z_imag':  Z_imag_raw.tolist(),
                'log_mag': log_Z_mag_raw.tolist(),
                'phase':   Z_phase_raw.tolist(),
            },
            'smoothed': {
                'Z_real':  Z_real_sm.tolist(),
                'Z_imag':  Z_imag_sm.tolist(),
                'log_mag': log_Z_mag_sm.tolist(),
                'phase':   Z_phase_sm.tolist(),
            },
            'gradient_features': {
                'd_log_mag':  d_log_mag.tolist(),
                'd_phase':    d_phase.tolist(),
                'd2_log_mag': d2_log_mag.tolist(),
                'd2_phase':   d2_phase.tolist(),
            },
            'drt': {
                'log_tau': log_tau.tolist(),
                'gamma':   gamma.tolist(),
            },
            'proposals':       proposal_list,
            'prediction':      prediction,
            'pooling_weights': analytics.get('pooling') or [],
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/eval_trajectory', methods=['POST'])
def eval_trajectory():
    """
    Run GPF on a single ground-truth trajectory from the temporal dataset.

    Request body (all optional except data_path):
        data_path  : str   — path to dataset directory (default: data/temporal_v1)
        traj_idx   : int   — specific trajectory index (default: random)
        pathology  : str   — pick a random trajectory of this pathology type
        particles  : int   — GPF particle count (default: 64)
        seed       : int   — RNG seed for random trajectory selection

    Returns full per-step tracking data including ground truth, GPF posterior
    mean/std, ESS, and summary metrics. Typically takes 5-30s on CPU.
    """
    try:
        from evaluate_rbpf import eval_single_trajectory
        body = request.json or {}
        data_path  = body.get('data_path', 'data/temporal_v1')
        traj_idx   = body.get('traj_idx',  None)
        pathology  = body.get('pathology', None)
        n_particles = int(body.get('particles', 64))
        seed       = int(body.get('seed', 0))

        result = eval_single_trajectory(
            data_path   = data_path,
            traj_idx    = traj_idx,
            pathology   = pathology,
            n_particles = n_particles,
            seed        = seed,
        )
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({'error': str(e), 'hint': 'Run generate_temporal_dataset.py first'}), 404
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/synthetic_trajectory', methods=['POST'])
def synthetic_trajectory():
    """
    Load raw synthetic trajectory data (impedance + ground truth) without inference.

    Request body (all optional):
        data_path    : str  — path to dataset directory (default: data/temporal_v1)
        traj_idx     : int  — specific trajectory index (default: random)
        pathology    : str  — pick a random trajectory of this pathology type
        seed         : int  — RNG seed (default: 0)
        n_timepoints : int  — max timepoints to return, subsampled evenly (default: 60)
    """
    try:
        import h5py, json as _json
        from pathlib import Path as _Path

        body       = request.json or {}
        data_path  = body.get('data_path', 'data/temporal_v1')
        traj_idx   = body.get('traj_idx', None)
        pathology  = body.get('pathology', None)
        seed       = int(body.get('seed', 0))
        n_tp       = int(body.get('n_timepoints', 60))

        h5_path = _Path(data_path) / 'temporal_dataset.h5'
        if not h5_path.exists():
            return jsonify({'error': f'Dataset not found: {h5_path}', 'hint': 'Run generate_temporal_dataset.py first'}), 404

        rng = np.random.default_rng(seed)

        with h5py.File(str(h5_path), 'r') as h5:
            freqs      = h5['frequencies'][:].tolist()
            meta       = _json.loads(h5.attrs['metadata'])
            n_total    = int(meta['n_trajectories'])

            pathologies_all = np.array([
                p.decode() if isinstance(p, bytes) else p
                for p in h5['pathology'][:]
            ])

            if traj_idx is None:
                if pathology and pathology != 'random':
                    candidates = np.where(pathologies_all == pathology)[0]
                    if len(candidates) == 0:
                        return jsonify({'error': f"No trajectories for pathology '{pathology}'"}), 404
                    traj_idx = int(rng.choice(candidates))
                else:
                    traj_idx = int(rng.integers(0, n_total))

            traj_idx = int(np.clip(traj_idx, 0, n_total - 1))
            pathology_label = str(pathologies_all[traj_idx])

            Z_r     = h5['impedance_real'][traj_idx].astype(np.float64)
            Z_i     = h5['impedance_imag'][traj_idx].astype(np.float64)
            derived = h5['derived_log10'][traj_idx].astype(np.float64)
            times   = h5['time_minutes'][traj_idx].astype(np.float64)
            raw_params = h5['params_log10'][traj_idx].astype(np.float64) if 'params_log10' in h5 else None

        T = len(times)
        step = max(1, T // n_tp)
        idx = list(range(0, T, step))

        dt_min = float(times[1] - times[0]) if len(times) > 1 else 0.0
        raw_gt = raw_params[idx].tolist() if raw_params is not None else None
        return jsonify({
            'pathology':   pathology_label,
            'traj_idx':    traj_idx,
            'time_min':    times[idx].tolist(),
            'dt_minutes':  dt_min,
            'frequencies': freqs,
            'Z_real':      Z_r[idx].tolist(),
            'Z_imag':      Z_i[idx].tolist(),
            'id_gt':       derived[idx].tolist(),
            'id_names':    ['tau_big', 'tau_small', 'TER', 'TEC', 'Rsh'],
            'raw_gt':      raw_gt,
            'raw_names':   ['Ra', 'Rb', 'Ca', 'Cb', 'Rsh'],
        })

    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/observability_analysis', methods=['POST'])
def observability_analysis():
    """
    Analyze observability and data quality of a sequence of EIS sweeps.

    Input: { sequences: [{Z_real, Z_imag, frequencies, time_min}, ...] }
    """
    try:
        data = request.json or {}
        sequences = data.get('sequences', [])
        if len(sequences) == 0:
            return jsonify({'error': 'Empty sequences'}), 400

        # 1. SNR score: log-linear regression residuals on log|Z| vs log(f)
        residual_stds = []
        for seq in sequences:
            try:
                f_arr = np.array(seq['frequencies'], dtype=np.float64)
                Zr = np.array(seq['Z_real'], dtype=np.float64)
                Zi = np.array(seq['Z_imag'], dtype=np.float64)
                log_f = np.log10(np.clip(f_arr, 1e-30, None))
                log_mag = np.log10(np.sqrt(Zr**2 + Zi**2).clip(1e-30))
                valid = np.isfinite(log_f) & np.isfinite(log_mag)
                if valid.sum() < 3:
                    continue
                x = log_f[valid]; y = log_mag[valid]
                xc = x - x.mean()
                slope = float(np.dot(xc, y) / (np.dot(xc, xc) + 1e-10))
                intercept = float(y.mean() - slope * x.mean())
                resid = y - (slope * x + intercept)
                residual_stds.append(float(np.std(resid)))
            except Exception:
                pass

        mean_resid_std = float(np.mean(residual_stds)) if residual_stds else 0.3
        snr_score = float(np.clip(1.0 - mean_resid_std / 0.3, 0.0, 1.0))

        # 2. Frequency coverage
        all_freqs = []
        for seq in sequences:
            try:
                f_arr = np.array(seq['frequencies'], dtype=np.float64)
                all_freqs.extend(f_arr.tolist())
            except Exception:
                pass

        if all_freqs:
            fmin = float(np.min(all_freqs))
            fmax = float(np.max(all_freqs))
        else:
            fmin, fmax = 0.1, 1e6

        span = float(np.log10(max(fmax / max(fmin, 1e-30), 1.0)))
        freq_coverage_score = float(
            np.clip(span / 4.0, 0.0, 1.0)
            * (1.0 if fmin < 1.0 else 0.7)
            * (1.0 if fmax > 1000.0 else 0.8)
        )

        # 3. Drift stability: std of Z_real at lowest frequency across time
        low_freq_Zr = []
        for seq in sequences:
            try:
                f_arr = np.array(seq['frequencies'], dtype=np.float64)
                Zr = np.array(seq['Z_real'], dtype=np.float64)
                low_idx = int(np.argmin(f_arr))
                low_freq_Zr.append(float(Zr[low_idx]))
            except Exception:
                pass

        if len(low_freq_Zr) >= 2:
            lf_arr = np.array(low_freq_Zr)
            drift_raw = float(np.std(lf_arr) / (np.mean(np.abs(lf_arr)) + 1e-10))
        else:
            drift_raw = 0.0

        if drift_raw < 0.05:
            drift_label = 'stable'
        elif drift_raw < 0.3:
            drift_label = 'event-driven'
        else:
            drift_label = 'high drift'

        # 4. Identifiability from snapshot physics at t=0
        identifiability_result = {
            'TER': {'score': 0.5, 'status': 'moderate', 'note': 'Could not compute'},
            'TEC': {'score': 0.5, 'status': 'moderate', 'note': 'Could not compute'},
            'Rsh': {'score': 0.5, 'status': 'moderate', 'note': 'Could not compute'},
            'Ra':  {'score': 0.3, 'status': 'ambiguous', 'note': 'Could not compute', 'structurally_ambiguous': True},
            'Rb':  {'score': 0.3, 'status': 'ambiguous', 'note': 'Could not compute', 'structurally_ambiguous': True},
        }
        try:
            seq0 = sequences[0]
            Zr0 = seq0['Z_real']; Zi0 = seq0['Z_imag']; f0 = seq0['frequencies']
            _, seed_log10 = _ecm_fit(Zr0, Zi0, f0, n_random_starts=8)
            if seed_log10 is not None:
                from src.pipeline.gpf import _jacobian_numerical
                omega0 = 2.0 * np.pi * np.array(f0, dtype=np.float64)
                J = _jacobian_numerical(seed_log10, omega0)
                sigma2 = 0.01
                FtF = J.T @ J / sigma2
                try:
                    FtF_inv = np.linalg.inv(FtF + np.eye(5) * 1e-8)
                    prior_var = (BOUNDS_HIGH - BOUNDS_LOW) ** 2
                    param_scores_raw = np.clip(1.0 - np.diag(FtF_inv) / (prior_var + 1e-30), 0.0, 1.0)
                except np.linalg.LinAlgError:
                    param_scores_raw = np.full(5, 0.3)

                dZ_dRa = J[:, 0]
                dZ_dRb = J[:, 1]
                norm_Ra = np.linalg.norm(dZ_dRa) + 1e-30
                struct_ambig = float(np.mean(np.abs(dZ_dRa - dZ_dRb) / norm_Ra)) < 0.1

                def _status(s):
                    if s > 0.7:
                        return 'strong'
                    elif s > 0.4:
                        return 'moderate'
                    return 'ambiguous'

                phys0 = 10.0 ** seed_log10
                Ra0, Rb0, Ca0, Cb0, Rsh0 = phys0
                TER0 = Rsh0 * (Ra0 + Rb0) / (Rsh0 + Ra0 + Rb0 + 1e-20)
                TEC0 = Ca0 * Cb0 / (Ca0 + Cb0 + 1e-20)

                s_TER = float(param_scores_raw[0] * 0.5 + param_scores_raw[4] * 0.5)
                s_TEC = float(param_scores_raw[2] * 0.5 + param_scores_raw[3] * 0.5)
                s_Rsh = float(param_scores_raw[4])
                s_Ra  = float(param_scores_raw[0])
                s_Rb  = float(param_scores_raw[1])

                identifiability_result = {
                    'TER': {'score': s_TER, 'status': _status(s_TER), 'note': f'TER={TER0/1000:.2f} kOhm'},
                    'TEC': {'score': s_TEC, 'status': _status(s_TEC), 'note': f'TEC={TEC0*1e6:.3f} uF'},
                    'Rsh': {'score': s_Rsh, 'status': _status(s_Rsh), 'note': f'Rsh={Rsh0/1000:.2f} kOhm'},
                    'Ra':  {'score': s_Ra, 'status': _status(s_Ra), 'note': f'Ra={Ra0/1000:.2f} kOhm', 'structurally_ambiguous': struct_ambig},
                    'Rb':  {'score': s_Rb, 'status': _status(s_Rb), 'note': f'Rb={Rb0/1000:.2f} kOhm', 'structurally_ambiguous': struct_ambig},
                }
        except Exception:
            pass

        # 5. Event windows: rolling std of Z_real at lowest frequency
        event_windows = []
        if len(low_freq_Zr) >= 10:
            try:
                lf_arr = np.array(low_freq_Zr)
                t_arr_obs = np.array([float(s.get('time_min', idx)) for idx, s in enumerate(sequences)])
                win = 10
                rolling_std = np.array([
                    float(np.std(lf_arr[max(0, k - win):k + 1]))
                    for k in range(len(lf_arr))
                ])
                med_std = float(np.median(rolling_std)) + 1e-10
                for k in range(len(rolling_std)):
                    if rolling_std[k] > 3.0 * med_std:
                        t_lo = float(t_arr_obs[max(0, k - win)])
                        t_hi = float(t_arr_obs[k])
                        conf = float(np.clip(rolling_std[k] / (3.0 * med_std) - 1.0, 0.0, 1.0))
                        event_windows.append({'t_start': t_lo, 't_end': t_hi, 'confidence': conf})
            except Exception:
                pass

        return jsonify({
            'snr_score':           snr_score,
            'freq_coverage_score': freq_coverage_score,
            'drift_score':         drift_raw,
            'drift_label':         drift_label,
            'identifiability':     identifiability_result,
            'event_windows':       event_windows,
            'n_sweeps':            len(sequences),
            'freq_range':          [fmin, fmax],
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


if __name__ == '__main__':
    print("Loading transformer model...")
    load_model()
    print(f"ECM fitting available: {HAS_SCIPY}")
    print("Starting API on port 5003...")
    app.run(host='0.0.0.0', port=5003, debug=False, threaded=True)
