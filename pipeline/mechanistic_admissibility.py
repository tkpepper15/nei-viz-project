"""
Mechanistic Admissibility Framework.

Determines whether mechanistic claims about circuit parameters are scientifically
admissible given the measurement information geometry.

Confidence = O * A * T * C
- O: Observability — per-parameter Fisher info reconstructed from eigendecomposition
- A: Attention alignment — model feature usage vs theoretical sensitivity (global)
- T: Temporal stability — Kalman smoother uncertainty reduction
- C: Changepoint corroboration — co-occurrence with high-confidence anchor events

The system reasons in eigenmodes (biophysical modes), not raw parameter space.
"""

import numpy as np

_PARAM_NAMES = ['tau_big', 'tau_small', 'TER', 'TEC', 'Rsh']

_PARAM_DISPLAY = {
    'tau_big':   'τ_b',
    'tau_small': 'τ_a',
    'TER':       'TER',
    'TEC':       'TEC',
    'Rsh':       'Rsh',
}

_BIOPHYSICAL_MODES = {
    'global_membrane':           'Global Membrane Mode',
    'polarized_decoupling':      'Polarized Decoupling Mode',
    'barrier_pathway_ambiguity': 'Barrier Pathway Ambiguity',
    'pure_capacitive':           'Pure Capacitive Mode',
    'trans_para_divergence':     'Trans/Para Divergence Mode',
    'unclassified':              'Unclassified Mode',
}

# Language degrades as confidence collapses — semantic resolution drops with observability
_CLAIM_LANGUAGE = {
    'supported': '{param} {direction}',
    'hedged':    'data supports a likely {direction} in {param}',
    'weak':      'a {mode}-related mode changed (direction uncertain)',
    'refused':   'measurement insufficient for mechanistic attribution',
}


def _score_observability(
    eigenvalues: np.ndarray,   # (n_pc,) descending
    eigenvectors: np.ndarray,  # (n_pc, n_params) — rows are PC vectors
) -> dict:
    """
    Per-parameter Fisher info = diagonal of V.T @ diag(λ) @ V.
    Each diagonal entry is the sum of squared loadings weighted by eigenvalues,
    which is the marginal information the measurement carries about that parameter.
    Normalized on log scale to [0, 1].
    """
    lam = np.maximum(eigenvalues, 1e-15)
    # V: (n_pc, n_params) — diagonal of V.T @ diag(λ) @ V = sum_i λ_i * v_i[p]^2
    F_diag = ((eigenvectors ** 2) * lam[:, None]).sum(axis=0)  # (n_params,)
    log_F = np.log10(np.maximum(F_diag, 1e-15))
    log_max = log_F.max()
    log_floor = np.log10(1e-6)
    O = np.clip((log_F - log_floor) / (log_max - log_floor + 1e-15), 0.0, 1.0)
    return dict(zip(_PARAM_NAMES, O.tolist()))


def _score_attention_alignment(
    attention_layers: list,         # nested list: (n_layers, n_heads, n_freq)
    last_params_log10: np.ndarray,  # (5,) log10 [Ra, Rb, Ca, Cb, Rsh]
    frequencies: list,              # (n_freq,) Hz
) -> float:
    """
    Global Pearson r between mean attention weights and theoretical sensitivity.
    Uses autodiff Jacobian for sensitivity (matches frontend's finite-difference logic).
    Maps r in [-1, 1] to [0, 1]. Returns 0.5 (maximum uncertainty) on failure.
    """
    try:
        import torch
        from src.physics.eis_fisher import compute_jacobian_autodiff

        attn_arr = np.array(attention_layers, dtype=np.float64)  # (n_layers, n_heads, n_freq)
        attn_mean = attn_arr.mean(axis=(0, 1))                   # (n_freq,)
        n_freq = len(frequencies)

        params_t = torch.tensor(last_params_log10, dtype=torch.float64).unsqueeze(0)
        omega_t  = torch.tensor(
            [2.0 * np.pi * f for f in frequencies], dtype=torch.float64
        ).unsqueeze(0)
        J = compute_jacobian_autodiff(params_t, omega_t)  # (1, 2*n_freq, 5)
        J_np = J[0].detach().numpy()                       # (2*n_freq, 5)

        # Per-frequency max sensitivity over params (mirrors frontend theoreticalSensitivity)
        sens = np.abs(J_np[:n_freq, :]).max(axis=1)        # (n_freq,)

        attn_n = attn_mean / (attn_mean.max() + 1e-12)
        sens_n = sens / (sens.max() + 1e-12)
        r = float(np.corrcoef(attn_n, sens_n)[0, 1])
        return float(np.clip((r + 1) / 2, 0.0, 1.0)) if np.isfinite(r) else 0.5
    except Exception:
        return 0.5


def _score_temporal_stability(
    std_smoothed: np.ndarray,  # (T, n_params)
    std_filtered: np.ndarray,  # (T, n_params)
) -> dict:
    """
    T = 1 - mean(std_smoothed / std_filtered) per parameter.
    RTS smoother only reduces uncertainty when the trajectory is internally consistent.
    High ratio means the smoother barely helps — trajectory is unstable.
    """
    if std_smoothed.shape[0] == 0:
        return {p: 0.5 for p in _PARAM_NAMES}
    ratio = std_smoothed / (std_filtered + 1e-12)
    mean_ratio = ratio.mean(axis=0)  # (n_params,)
    T = np.clip(1.0 - mean_ratio, 0.0, 1.0)
    return dict(zip(_PARAM_NAMES, T.tolist()))


def _score_changepoint_corroboration(
    changepoints: list,
    tolerance_min: float = 3.5,
) -> dict:
    """
    Per-parameter corroboration: fraction of its changepoints that co-occur with
    a changepoint in tau_big or tau_small (highest-identifiability anchors).
    Returns 1.0 for parameters with no changepoints — nothing to corroborate.
    """
    anchor_times = {
        cp['time_min'] for cp in changepoints
        if cp.get('dominant_param') in ('tau_big', 'tau_small')
        and cp.get('time_min') is not None
    }
    scores = {}
    for param in _PARAM_NAMES:
        relevant = [
            cp for cp in changepoints
            if cp.get('dominant_param') == param and cp.get('time_min') is not None
        ]
        if not relevant:
            scores[param] = 1.0
        else:
            corroborated = sum(
                any(abs(cp['time_min'] - t) < tolerance_min for t in anchor_times)
                for cp in relevant
            )
            scores[param] = corroborated / len(relevant)
    return scores


def _assign_biophysical_mode(eigenvector: list) -> str:
    """
    Classify a PC eigenvector into a named biophysical mode.
    Parameter order: [tau_big, tau_small, TER, TEC, Rsh]
    """
    v = np.array(eigenvector)
    abs_v = np.abs(v)

    if abs_v[0] > 0.45 and abs_v[1] > 0.45:
        return 'global_membrane' if np.sign(v[0]) == np.sign(v[1]) else 'polarized_decoupling'
    if abs_v[3] > 0.65:
        return 'pure_capacitive'
    if abs_v[2] > 0.45 and abs_v[4] > 0.45:
        return 'trans_para_divergence' if np.sign(v[2]) != np.sign(v[4]) else 'barrier_pathway_ambiguity'
    return 'unclassified'


def _get_claim_tier(product: float) -> str:
    if product >= 0.6: return 'supported'
    if product >= 0.3: return 'hedged'
    if product >= 0.1: return 'weak'
    return 'refused'


def _format_claim(tier: str, direction: str, param: str, mode_name: str) -> str:
    tmpl = _CLAIM_LANGUAGE[tier]
    return tmpl.format(
        param=_PARAM_DISPLAY.get(param, param),
        direction=direction,
        mode=_BIOPHYSICAL_MODES.get(mode_name, mode_name),
    )


def compute_admissibility(
    kalman_result: dict,
    changepoints: list,
    analytics_last: dict | None,
    last_params_log10: np.ndarray | None,
    last_frequencies: list | None,
) -> dict:
    """
    Compute mechanistic admissibility for all identifiable parameters.

    Args:
        kalman_result:      Output of _kalman_smooth_identifiable (identifiable space)
        changepoints:       Full changepoint list with time_min populated
        analytics_last:     Last per-timepoint analytics dict (attention, pooling)
        last_params_log10:  Particle mean at last timepoint, log10 [Ra, Rb, Ca, Cb, Rsh]
        last_frequencies:   Measurement frequencies (Hz) at last timepoint

    Returns:
        Dict with scores, claim_language, biophysical_modes, mechanistic_inconsistency
    """
    geo = kalman_result.get('geometry', {})
    eigen_vals_history = geo.get('eigen_vals', [])
    eigen_vecs_history = geo.get('eigen_vecs', [])

    if not eigen_vals_history or not eigen_vecs_history:
        return {
            'scores': {}, 'claim_language': {}, 'biophysical_modes': [],
            'mechanistic_inconsistency': False,
        }

    last_ev   = np.array(eigen_vals_history[-1])   # (5,) descending eigenvalues
    last_evec = np.array(eigen_vecs_history[-1])    # (5, 5) rows = PC vectors

    std_smoothed = np.array(kalman_result.get('std_smoothed', []))
    std_filtered = np.array(kalman_result.get('std_filtered', []))
    trends       = kalman_result.get('trends', [])

    O = _score_observability(last_ev, last_evec)

    if analytics_last and last_params_log10 is not None and last_frequencies:
        A = _score_attention_alignment(
            analytics_last.get('attention', []),
            last_params_log10,
            last_frequencies,
        )
    else:
        A = 0.5

    T = _score_temporal_stability(std_smoothed, std_filtered)
    C = _score_changepoint_corroboration(changepoints)

    biophysical_modes = []
    for ei, (ev, evec) in enumerate(zip(last_ev.tolist(), last_evec.tolist())):
        mode_name = _assign_biophysical_mode(evec)
        biophysical_modes.append({
            'pc':         ei,
            'name':       mode_name,
            'label':      _BIOPHYSICAL_MODES.get(mode_name, mode_name),
            'eigenvalue': float(ev),
            'loadings':   dict(zip(_PARAM_NAMES, evec)),
        })

    scores         = {}
    claim_language = {}
    for pi, param in enumerate(_PARAM_NAMES):
        o = O[param]; t = T[param]; c = C[param]
        product = o * A * t * c

        direction = trends[pi].get('direction', 'stable') if pi < len(trends) else 'stable'

        dom_mode = 'unclassified'
        for m in biophysical_modes:
            if abs(m['loadings'].get(param, 0)) > 0.4:
                dom_mode = m['name']
                break

        tier = _get_claim_tier(product)
        scores[param] = {
            'O': round(o, 3), 'A': round(A, 3),
            'T': round(t, 3), 'C': round(c, 3),
            'product': round(product, 3),
        }
        claim_language[param] = {
            'tier': tier,
            'text': _format_claim(tier, direction, param, dom_mode),
        }

    high_O_params = [p for p in _PARAM_NAMES if O[p] > 0.7]
    mechanistic_inconsistency = A < 0.3 and len(high_O_params) > 0

    return {
        'scores':                    scores,
        'claim_language':            claim_language,
        'biophysical_modes':         biophysical_modes,
        'mechanistic_inconsistency': mechanistic_inconsistency,
    }
