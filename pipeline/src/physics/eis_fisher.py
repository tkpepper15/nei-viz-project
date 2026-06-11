#!/usr/bin/env python3
"""
Fisher Information and Derived Quantities for EIS Circuit Model

This module provides:
1. Derived quantity computation (TER, TEC, tau_a, tau_b, Ra+Rb)
2. Analytic Jacobian of impedance w.r.t. parameters
3. Fisher information matrix computation
4. Identifiability analysis via eigendecomposition
5. Cramér-Rao bounds in identifiable parameter space (for Kalman filter R_t floor)

The RPE equivalent circuit model:
       Rsh (Shunt/Paracellular Resistance)
   ----[Rsh]----+----------+------
                |          |
            [Ra]|      [Rb]|
                |          |
            [Ca]|      [Cb]|
                |          |
                +----------+
    Apical RC       Basolateral RC

Rsh is in PARALLEL with the series combination of the two RC branches.

Identifiability Hierarchy (from CLAUDE.md):
- Highly identifiable: tau_b, Rsh
- Moderately identifiable: TER, tau_a, TEC
- Weakly identifiable (degenerate): Ra, Rb, Ca, Cb individually
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DerivedQuantities:
    """Container for derived circuit quantities."""
    TER: torch.Tensor      # Transepithelial resistance
    TEC: torch.Tensor      # Transepithelial capacitance
    tau_a: torch.Tensor    # Apical time constant
    tau_b: torch.Tensor    # Basolateral time constant
    Ra_plus_Rb: torch.Tensor  # Sum of membrane resistances


def compute_derived_quantities(
    Ra: torch.Tensor,
    Rb: torch.Tensor,
    Ca: torch.Tensor,
    Cb: torch.Tensor,
    Rsh: torch.Tensor
) -> DerivedQuantities:
    """
    Compute derived quantities from base parameters.

    Args:
        Ra, Rb: Membrane resistances (Ohm)
        Ca, Cb: Membrane capacitances (F)
        Rsh: Shunt resistance (Ohm)

    Returns:
        DerivedQuantities dataclass

    Physics:
        TER = Rsh || (Ra + Rb) = (Rsh * (Ra + Rb)) / (Rsh + Ra + Rb)
        TEC = (Ca * Cb) / (Ca + Cb)  [series capacitance]
        tau_a = Ra * Ca
        tau_b = Rb * Cb
    """
    Ra_plus_Rb = Ra + Rb

    # TER: parallel combination of Rsh and (Ra + Rb)
    TER = (Rsh * Ra_plus_Rb) / (Rsh + Ra_plus_Rb + 1e-12)

    # TEC: series capacitance
    TEC = (Ca * Cb) / (Ca + Cb + 1e-20)

    # Time constants
    tau_a = Ra * Ca
    tau_b = Rb * Cb

    return DerivedQuantities(
        TER=TER,
        TEC=TEC,
        tau_a=tau_a,
        tau_b=tau_b,
        Ra_plus_Rb=Ra_plus_Rb
    )


def compute_derived_from_log10(params_log10: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute derived quantities from log10 parameters.

    Args:
        params_log10: (batch, 5) tensor with [log10(Ra), log10(Rb), log10(Ca), log10(Cb), log10(Rsh)]

    Returns:
        Dict with log10 of derived quantities for consistent MAE computation
    """
    # Convert from log10 to linear
    Ra = 10 ** params_log10[:, 0]
    Rb = 10 ** params_log10[:, 1]
    Ca = 10 ** params_log10[:, 2]
    Cb = 10 ** params_log10[:, 3]
    Rsh = 10 ** params_log10[:, 4]

    derived = compute_derived_quantities(Ra, Rb, Ca, Cb, Rsh)

    # Return log10 of derived quantities (for consistent MAE in decades)
    return {
        'TER': torch.log10(derived.TER + 1e-12),
        'TEC': torch.log10(derived.TEC + 1e-20),
        'tau_a': torch.log10(derived.tau_a + 1e-12),
        'tau_b': torch.log10(derived.tau_b + 1e-12),
        'Ra_plus_Rb': torch.log10(derived.Ra_plus_Rb + 1e-12)
    }


def compute_impedance(
    Ra: torch.Tensor,
    Rb: torch.Tensor,
    Ca: torch.Tensor,
    Cb: torch.Tensor,
    Rsh: torch.Tensor,
    omega: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute complex impedance Z(omega) for the RPE circuit model.

    Args:
        Ra, Rb, Ca, Cb, Rsh: Circuit parameters (can be batched)
        omega: Angular frequencies (rad/s), shape (n_freq,) or (batch, n_freq)

    Returns:
        Z_real, Z_imag: Real and imaginary parts of impedance

    Circuit model:
        Z_a = Ra / (1 + j*omega*Ra*Ca)  [apical RC]
        Z_b = Rb / (1 + j*omega*Rb*Cb)  [basolateral RC]
        Z_series = Z_a + Z_b
        Z_total = Rsh || Z_series = (Rsh * Z_series) / (Rsh + Z_series)
    """
    # Ensure proper broadcasting
    if omega.dim() == 1:
        omega = omega.unsqueeze(0)  # (1, n_freq)

    # Expand parameters for broadcasting with frequencies
    if Ra.dim() == 0:
        Ra = Ra.unsqueeze(0)
        Rb = Rb.unsqueeze(0)
        Ca = Ca.unsqueeze(0)
        Cb = Cb.unsqueeze(0)
        Rsh = Rsh.unsqueeze(0)

    Ra = Ra.unsqueeze(-1)   # (batch, 1)
    Rb = Rb.unsqueeze(-1)
    Ca = Ca.unsqueeze(-1)
    Cb = Cb.unsqueeze(-1)
    Rsh = Rsh.unsqueeze(-1)

    # Time constants
    tau_a = Ra * Ca
    tau_b = Rb * Cb

    # Apical impedance: Z_a = Ra / (1 + j*omega*tau_a)
    denom_a = 1 + 1j * omega * tau_a
    Z_a = Ra / denom_a

    # Basolateral impedance: Z_b = Rb / (1 + j*omega*tau_b)
    denom_b = 1 + 1j * omega * tau_b
    Z_b = Rb / denom_b

    # Series combination
    Z_series = Z_a + Z_b

    # Parallel with Rsh: Z_total = Rsh * Z_series / (Rsh + Z_series)
    Z_total = (Rsh * Z_series) / (Rsh + Z_series + 1e-12)

    return Z_total.real, Z_total.imag


def compute_jacobian_analytic(
    Ra: torch.Tensor,
    Rb: torch.Tensor,
    Ca: torch.Tensor,
    Cb: torch.Tensor,
    Rsh: torch.Tensor,
    omega: torch.Tensor
) -> torch.Tensor:
    """
    Compute analytic Jacobian dZ/d(theta) for Fisher information.

    Args:
        Ra, Rb, Ca, Cb, Rsh: Circuit parameters (batch,)
        omega: Angular frequencies (n_freq,)

    Returns:
        J: Jacobian tensor (batch, 2*n_freq, 5)
           First n_freq rows are dZ_real/d(theta)
           Last n_freq rows are dZ_imag/d(theta)
           Columns are [Ra, Rb, Ca, Cb, Rsh]

    Note: Parameters and derivatives are in LINEAR space.
    For log-space Fisher, multiply by theta * ln(10).
    """
    batch_size = Ra.shape[0]
    n_freq = omega.shape[0]
    device = Ra.device
    dtype = Ra.dtype

    # Use autograd for robustness
    Ra_req = Ra.clone().requires_grad_(True)
    Rb_req = Rb.clone().requires_grad_(True)
    Ca_req = Ca.clone().requires_grad_(True)
    Cb_req = Cb.clone().requires_grad_(True)
    Rsh_req = Rsh.clone().requires_grad_(True)

    Z_real, Z_imag = compute_impedance(Ra_req, Rb_req, Ca_req, Cb_req, Rsh_req, omega)

    # Jacobian: (batch, 2*n_freq, 5)
    J = torch.zeros(batch_size, 2 * n_freq, 5, device=device, dtype=dtype)

    params = [Ra_req, Rb_req, Ca_req, Cb_req, Rsh_req]

    for i in range(n_freq):
        for b in range(batch_size):
            # Gradient of Z_real[b, i] w.r.t. parameters
            if Ra_req.grad is not None:
                Ra_req.grad.zero_()
                Rb_req.grad.zero_()
                Ca_req.grad.zero_()
                Cb_req.grad.zero_()
                Rsh_req.grad.zero_()

            Z_real[b, i].backward(retain_graph=True)

            for p_idx, param in enumerate(params):
                if param.grad is not None:
                    J[b, i, p_idx] = param.grad[b].clone()

            # Reset gradients
            for param in params:
                if param.grad is not None:
                    param.grad.zero_()

            # Gradient of Z_imag[b, i] w.r.t. parameters
            Z_imag[b, i].backward(retain_graph=True)

            for p_idx, param in enumerate(params):
                if param.grad is not None:
                    J[b, n_freq + i, p_idx] = param.grad[b].clone()

    return J


def compute_jacobian_autodiff(
    params_log10: torch.Tensor,
    omega: torch.Tensor
) -> torch.Tensor:
    """
    Compute Jacobian using PyTorch autograd (more efficient batched version).

    Args:
        params_log10: (batch, 5) log10 parameters
        omega: (n_freq,) angular frequencies

    Returns:
        J: (batch, 2*n_freq, 5) Jacobian in log10 parameter space
    """
    batch_size = params_log10.shape[0]
    n_freq = omega.shape[0]

    params_log10 = params_log10.clone().requires_grad_(True)

    # Convert to linear
    params_lin = 10 ** params_log10
    Ra, Rb, Ca, Cb, Rsh = params_lin[:, 0], params_lin[:, 1], params_lin[:, 2], params_lin[:, 3], params_lin[:, 4]

    Z_real, Z_imag = compute_impedance(Ra, Rb, Ca, Cb, Rsh, omega)

    # Stack outputs
    Z_stack = torch.cat([Z_real, Z_imag], dim=1)  # (batch, 2*n_freq)

    # Compute Jacobian using vmap if available, else loop
    J = torch.zeros(batch_size, 2 * n_freq, 5, device=params_log10.device, dtype=params_log10.dtype)

    for i in range(2 * n_freq):
        grad_outputs = torch.zeros_like(Z_stack)
        grad_outputs[:, i] = 1.0

        grads = torch.autograd.grad(
            Z_stack, params_log10,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=False
        )[0]

        J[:, i, :] = grads

    return J


def compute_fisher_information(
    params_log10: torch.Tensor,
    omega: torch.Tensor,
    noise_std: float = 1.0
) -> torch.Tensor:
    """
    Compute Fisher information matrix for EIS parameters.

    Args:
        params_log10: (batch, 5) log10 parameters
        omega: (n_freq,) angular frequencies
        noise_std: Standard deviation of measurement noise (Ohm)

    Returns:
        F: (batch, 5, 5) Fisher information matrix

    Fisher information:
        F = J^T * Sigma_Z^{-1} * J

    where J is the Jacobian and Sigma_Z is the noise covariance.
    For i.i.d. noise with variance sigma^2:
        F = (1/sigma^2) * J^T * J
    """
    J = compute_jacobian_autodiff(params_log10, omega)  # (batch, 2*n_freq, 5)

    # Fisher information: F = J^T * J / sigma^2
    F = torch.bmm(J.transpose(1, 2), J) / (noise_std ** 2)

    return F


def _jacobian_raw_to_identifiable(
    params_raw_log10: np.ndarray,
    beta: float = 20.0,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Numerical Jacobian d(θ_id)/d(θ_raw), shape (5, 5).

    Uses the same smooth log-sum-exp approximation (β=20) as _to_identifiable_log10
    in backend_api.py so the Jacobian matches the actual transform used in the
    Kalman filter.

    θ_raw: [log10 Ra, log10 Rb, log10 Ca, log10 Cb, log10 Rsh]
    θ_id:  [log10 τ_big, log10 τ_small, log10 TER, log10 TEC, log10 Rsh]
    """
    def _fwd(p: np.ndarray) -> np.ndarray:
        Ra, Rb, Ca, Cb, Rsh = 10.0 ** p
        tau_a = Ra * Ca
        tau_b = Rb * Cb
        log_a = np.log(tau_a + 1e-30)
        log_b = np.log(tau_b + 1e-30)
        log_tau_big   =  np.logaddexp(beta * log_a, beta * log_b) / (beta * np.log(10))
        log_tau_small = -np.logaddexp(-beta * log_a, -beta * log_b) / (beta * np.log(10))
        S   = Ra + Rb
        TER = Rsh * S / (Rsh + S + 1e-30)
        TEC = Ca * Cb / (Ca + Cb + 1e-30)
        return np.array([
            log_tau_big,
            log_tau_small,
            np.log10(TER + 1e-30),
            np.log10(TEC + 1e-30),
            np.log10(Rsh + 1e-30),
        ])

    J = np.zeros((5, 5))
    for i in range(5):
        h = np.zeros(5)
        h[i] = eps
        J[:, i] = (_fwd(params_raw_log10 + h) - _fwd(params_raw_log10 - h)) / (2.0 * eps)
    return J


def compute_cr_bounds_identifiable(
    params_raw_log10: np.ndarray,
    omega: np.ndarray,
    noise_std: float = 1.0,
) -> np.ndarray:
    """
    Cramér-Rao lower bound diagonal in identifiable log10 space.

    Args:
        params_raw_log10: (T, 5) batch of raw log10 operating points
                          [log10 Ra, log10 Rb, log10 Ca, log10 Cb, log10 Rsh]
        omega:            (n_freq,) angular frequencies (rad/s)
        noise_std:        measurement noise std (Ω). CR bound scales as σ².

    Returns:
        cr_diag: (T, 5) diagonal of CR bound matrix in identifiable space.
                 Units: (log10)². Entry i gives the minimum achievable variance
                 on log10(θ_id[i]) given the measurement noise.

    Method:
        J_raw  = dZ/d(log10 θ_raw)        autograd     (T, 2n_freq, 5)
        J_fwd  = d(θ_id)/d(θ_raw)         numerical    (5, 5) per timepoint
        J_id   = J_raw @ pinv(J_fwd)      chain rule   (T, 2n_freq, 5)
        F_id   = J_id^T J_id / σ²         FIM          (5, 5) per timepoint
        CR     = pinv(F_id)                             (5, 5) per timepoint

    The CR bound is the theoretical minimum variance floor — no filter can be
    more precise than this, regardless of how many samples it uses. Using it as
    a floor for the Kalman R_t prevents the filter from claiming identifiability
    it cannot have given the spectrum's information content.
    """
    T = params_raw_log10.shape[0]
    params_t = torch.tensor(params_raw_log10, dtype=torch.float32)
    omega_t  = torch.tensor(omega,            dtype=torch.float32)
    # Batch Jacobian: (T, 2*n_freq, 5) — dZ/d(log10 θ_raw) for all timepoints at once
    J_raw_np = compute_jacobian_autodiff(params_t, omega_t).detach().numpy()

    # Maximum CR bound per parameter (log10)².
    # Directions where CR exceeds this are effectively unobservable from a single
    # spectrum (e.g. the residual {tau_big, tau_small, Rsh} degeneracy). The cap
    # prevents infinite-CR directions from zeroing out the Kalman gain for those
    # parameters — temporal continuity resolves this degeneracy across timepoints.
    CR_CAP = 0.25   # (0.5 log10 std) — generous cap, only kicks in for truly null dirs

    cr_diag = np.zeros((T, 5))
    for t in range(T):
        # Jacobian of the raw→identifiable transform at this operating point
        J_fwd     = _jacobian_raw_to_identifiable(params_raw_log10[t])
        # dθ_raw/dθ_id via pseudo-inverse (handles near-singular at τ_a ≈ τ_b)
        J_fwd_inv = np.linalg.pinv(J_fwd)
        # dZ/dθ_id = dZ/dθ_raw @ dθ_raw/dθ_id
        J_id      = J_raw_np[t] @ J_fwd_inv                    # (2*n_freq, 5)
        # FIM in identifiable space: F = J^T J / σ²
        F_id      = J_id.T @ J_id / (noise_std ** 2)           # (5, 5)
        # CR bound via eigendecomposition — only sum over identifiable directions.
        # Directions with near-zero eigenvalue (spectral degeneracies, e.g. the
        # residual {τ_big, τ_small, Rsh} null mode) would give infinite CR;
        # excluding them gives the CR bound within the observable subspace.
        evals, evecs = np.linalg.eigh(F_id)                    # ascending
        # Threshold: direction is identifiable if eigenvalue > 1% of max eigenvalue
        threshold = max(evals[-1] * 1e-2, 1e-8)
        cr_per_param = np.zeros(5)
        for k in range(5):
            if evals[k] > threshold:
                cr_per_param += (evecs[:, k] ** 2) / evals[k]
        # Apply cap: null directions default to CR_CAP (not infinite)
        cr_diag[t] = np.minimum(cr_per_param, CR_CAP)

    return cr_diag


def analyze_identifiability(
    fisher_matrix: torch.Tensor,
    threshold: float = 1e-6
) -> Dict[str, torch.Tensor]:
    """
    Analyze parameter identifiability via Fisher eigendecomposition.

    Args:
        fisher_matrix: (batch, 5, 5) Fisher information matrix
        threshold: Eigenvalue threshold for identifiability

    Returns:
        Dict containing:
            eigenvalues: (batch, 5) sorted eigenvalues
            eigenvectors: (batch, 5, 5) corresponding eigenvectors
            identifiable_dims: (batch,) number of identifiable dimensions
            cr_bounds: (batch, 5) Cramer-Rao bounds (diagonal of F^{-1})
    """
    batch_size = fisher_matrix.shape[0]
    device = fisher_matrix.device
    dtype = fisher_matrix.dtype

    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(fisher_matrix)

    # Sort by eigenvalue (descending)
    idx = torch.argsort(eigenvalues, dim=-1, descending=True)
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, 5)

    eigenvalues_sorted = eigenvalues[batch_idx, idx]
    eigenvectors_sorted = torch.zeros_like(eigenvectors)
    for b in range(batch_size):
        eigenvectors_sorted[b] = eigenvectors[b, :, idx[b]]

    # Count identifiable dimensions (clip negative eigenvalues first)
    eigenvalues_clipped = torch.clamp(eigenvalues_sorted, min=0)
    identifiable_dims = (eigenvalues_clipped > threshold).sum(dim=-1)

    # Cramer-Rao bounds via proper PSD inverse
    cr_bounds = torch.zeros(batch_size, 5, device=device, dtype=dtype)
    for b in range(batch_size):
        # Clip negative eigenvalues for numerical stability
        eigvals = torch.clamp(eigenvalues[b], min=1e-10)
        eigvecs = eigenvectors[b]

        # Reconstruct PSD Fisher and invert
        F_psd = eigvecs @ torch.diag(eigvals) @ eigvecs.T
        try:
            F_inv = torch.linalg.inv(F_psd)
            cr_bounds[b] = torch.clamp(torch.diag(F_inv), min=0)  # CR bounds must be non-negative
        except RuntimeError:
            cr_bounds[b] = torch.full((5,), float('inf'), device=device, dtype=dtype)

    return {
        'eigenvalues': eigenvalues_sorted,
        'eigenvectors': eigenvectors_sorted,
        'identifiable_dims': identifiable_dims,
        'cr_bounds': cr_bounds
    }


def gauss_newton_refinement(
    params_log10_init: torch.Tensor,
    Z_real_obs: torch.Tensor,
    Z_imag_obs: torch.Tensor,
    omega: torch.Tensor,
    n_steps: int = 5,
    damping: float = 0.1,
    noise_std: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gauss-Newton refinement with Fisher-based update.

    This performs local maximum likelihood estimation starting from
    the neural network's proposal.

    Args:
        params_log10_init: (batch, 5) initial log10 parameters from neural net
        Z_real_obs, Z_imag_obs: (batch, n_freq) observed impedance
        omega: (n_freq,) angular frequencies
        n_steps: Number of refinement iterations
        damping: Levenberg-Marquardt damping factor
        noise_std: Assumed measurement noise

    Returns:
        params_log10_refined: (batch, 5) refined parameters
        cov_refined: (batch, 5, 5) posterior covariance (approx F^{-1})

    Update rule:
        theta <- theta + (J^T J + lambda*I)^{-1} * J^T * residual
    """
    batch_size = params_log10_init.shape[0]
    n_freq = omega.shape[0]
    device = params_log10_init.device
    dtype = params_log10_init.dtype

    params = params_log10_init.clone()

    for step in range(n_steps):
        # Compute predicted impedance
        params_lin = 10 ** params
        Ra, Rb, Ca, Cb, Rsh = params_lin[:, 0], params_lin[:, 1], params_lin[:, 2], params_lin[:, 3], params_lin[:, 4]

        Z_real_pred, Z_imag_pred = compute_impedance(Ra, Rb, Ca, Cb, Rsh, omega)

        # Residual
        residual = torch.cat([
            Z_real_obs - Z_real_pred,
            Z_imag_obs - Z_imag_pred
        ], dim=1)  # (batch, 2*n_freq)

        # Jacobian
        J = compute_jacobian_autodiff(params, omega)  # (batch, 2*n_freq, 5)

        # Gauss-Newton update: (J^T J + lambda*I)^{-1} J^T r
        JtJ = torch.bmm(J.transpose(1, 2), J)  # (batch, 5, 5)
        Jtr = torch.bmm(J.transpose(1, 2), residual.unsqueeze(-1)).squeeze(-1)  # (batch, 5)

        # Damped normal equations
        damping_matrix = damping * torch.eye(5, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        A = JtJ + damping_matrix

        # Solve for update
        try:
            delta = torch.linalg.solve(A, Jtr)
        except RuntimeError:
            # Fallback to pseudoinverse
            delta = torch.bmm(torch.linalg.pinv(A), Jtr.unsqueeze(-1)).squeeze(-1)

        # Apply update with line search factor
        params = params + 0.5 * delta

        # Clamp to reasonable bounds (log10 space)
        # Ra, Rb: 1 to 1e6 Ohm -> log10: 0 to 6
        # Ca, Cb: 1e-9 to 1e-3 F -> log10: -9 to -3
        # Rsh: 10 to 1e7 Ohm -> log10: 1 to 7
        bounds_low = torch.tensor([0.0, 0.0, -9.0, -9.0, 1.0], device=device, dtype=dtype)
        bounds_high = torch.tensor([6.0, 6.0, -3.0, -3.0, 7.0], device=device, dtype=dtype)
        params = torch.clamp(params, bounds_low, bounds_high)

    # Final Fisher information for covariance estimate
    F = compute_fisher_information(params, omega, noise_std)

    # Posterior covariance approximation: F^{-1}
    cov_refined = torch.zeros(batch_size, 5, 5, device=device, dtype=dtype)
    for b in range(batch_size):
        try:
            cov_refined[b] = torch.linalg.pinv(F[b])
        except RuntimeError:
            cov_refined[b] = torch.eye(5, device=device, dtype=dtype) * 1e6

    return params, cov_refined


def nullspace_projected_refinement(
    params_log10_init: torch.Tensor,
    Z_real_obs: torch.Tensor,
    Z_imag_obs: torch.Tensor,
    omega: torch.Tensor,
    frequency_weights: Optional[torch.Tensor] = None,
    n_steps: int = 5,
    damping: float = 0.1,
    noise_std: float = 1.0,
    eigenvalue_threshold: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Gauss-Newton refinement with nullspace projection.

    Critical innovation: Only update along identifiable directions.
    This prevents artificial breaking of degeneracies while achieving
    CR efficiency in identifiable subspaces.

    Update rule:
        theta <- theta + U_I @ Lambda_I^{-1} @ U_I^T @ gradient

    where U_I spans the identifiable subspace (large eigenvalues).

    Args:
        params_log10_init: (batch, 5) initial log10 parameters
        Z_real_obs, Z_imag_obs: (batch, n_freq) observed impedance
        omega: (n_freq,) angular frequencies
        frequency_weights: (n_freq,) optional RF-derived weights, must sum to 1
        n_steps: Number of refinement iterations
        damping: Levenberg-Marquardt damping factor
        noise_std: Assumed measurement noise
        eigenvalue_threshold: Threshold for identifiable eigenvalues

    Returns:
        params_log10_refined: (batch, 5) refined parameters
        cov_refined: (batch, 5, 5) posterior covariance
        diagnostics: Dict with eigenvalues, identifiable_dims, etc.
    """
    batch_size = params_log10_init.shape[0]
    n_freq = omega.shape[0]
    device = params_log10_init.device
    dtype = params_log10_init.dtype

    # Default: uniform frequency weights
    if frequency_weights is None:
        frequency_weights = torch.ones(n_freq, device=device, dtype=dtype) / n_freq
    else:
        frequency_weights = frequency_weights.to(device=device, dtype=dtype)
        # Ensure weights sum to 1
        frequency_weights = frequency_weights / (frequency_weights.sum() + 1e-12)

    params = params_log10_init.clone()

    # Track diagnostics
    eigenvalue_history = []
    identifiable_dims_history = []

    for step in range(n_steps):
        # Compute predicted impedance
        params_lin = 10 ** params
        Ra = params_lin[:, 0]
        Rb = params_lin[:, 1]
        Ca = params_lin[:, 2]
        Cb = params_lin[:, 3]
        Rsh = params_lin[:, 4]

        Z_real_pred, Z_imag_pred = compute_impedance(Ra, Rb, Ca, Cb, Rsh, omega)

        # Weighted residual: sqrt(w) * (Z_obs - Z_pred)
        # This is equivalent to using weighted least squares
        w_sqrt = torch.sqrt(frequency_weights).unsqueeze(0)  # (1, n_freq)

        residual_real = w_sqrt * (Z_real_obs - Z_real_pred)
        residual_imag = w_sqrt * (Z_imag_obs - Z_imag_pred)
        residual = torch.cat([residual_real, residual_imag], dim=1)  # (batch, 2*n_freq)

        # Jacobian (weighted)
        J = compute_jacobian_autodiff(params, omega)  # (batch, 2*n_freq, 5)

        # Weight the Jacobian rows
        w_sqrt_full = torch.cat([w_sqrt, w_sqrt], dim=1).unsqueeze(-1)  # (1, 2*n_freq, 1)
        J_weighted = J * w_sqrt_full  # (batch, 2*n_freq, 5)

        # Fisher information: J^T @ J
        JtJ = torch.bmm(J_weighted.transpose(1, 2), J_weighted)  # (batch, 5, 5)

        # Eigendecomposition for nullspace projection
        eigenvalues, eigenvectors = torch.linalg.eigh(JtJ)  # eigenvalues ascending

        # Identify identifiable subspace
        identifiable_mask = eigenvalues > eigenvalue_threshold  # (batch, 5)
        n_identifiable = identifiable_mask.sum(dim=-1)  # (batch,)

        eigenvalue_history.append(eigenvalues.detach().clone())
        identifiable_dims_history.append(n_identifiable.detach().clone())

        # Gradient: J^T @ residual
        Jtr = torch.bmm(J_weighted.transpose(1, 2), residual.unsqueeze(-1)).squeeze(-1)  # (batch, 5)

        # Project gradient onto identifiable subspace and solve
        # delta = U_I @ Lambda_I^{-1} @ U_I^T @ Jtr
        delta = torch.zeros(batch_size, 5, device=device, dtype=dtype)

        for b in range(batch_size):
            # Get identifiable eigenvectors and eigenvalues
            mask = identifiable_mask[b]
            if mask.sum() == 0:
                continue  # No identifiable directions, skip update

            U_I = eigenvectors[b, :, mask]  # (5, n_ident)
            Lambda_I = eigenvalues[b, mask]  # (n_ident,)

            # Project gradient to identifiable subspace
            grad_proj = U_I.T @ Jtr[b]  # (n_ident,)

            # Solve with damping
            Lambda_damped = Lambda_I + damping
            update_proj = grad_proj / Lambda_damped  # (n_ident,)

            # Project back to full space
            delta[b] = U_I @ update_proj

        # Apply update with line search factor
        params = params + 0.5 * delta

        # Clamp to reasonable bounds
        bounds_low = torch.tensor([0.0, 0.0, -9.0, -9.0, 1.0], device=device, dtype=dtype)
        bounds_high = torch.tensor([6.0, 6.0, -3.0, -3.0, 7.0], device=device, dtype=dtype)
        params = torch.clamp(params, bounds_low, bounds_high)

    # Final Fisher information for covariance estimate
    # Recompute with uniform weights for stability
    J_final = compute_jacobian_autodiff(params, omega)
    F_final = torch.bmm(J_final.transpose(1, 2), J_final) / (noise_std ** 2)

    # Regularize Fisher for numerical stability
    reg = 1e-4 * torch.eye(5, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
    F_reg = F_final + reg

    # Compute covariance as inverse of regularized Fisher
    cov_refined = torch.zeros(batch_size, 5, 5, device=device, dtype=dtype)
    for b in range(batch_size):
        try:
            cov_refined[b] = torch.linalg.inv(F_reg[b])
        except RuntimeError:
            # Fallback: diagonal approximation
            cov_refined[b] = torch.diag(1.0 / (torch.diag(F_reg[b]) + 1e-8))

    # Ensure PSD via eigenvalue clipping (project onto PSD cone)
    for b in range(batch_size):
        eigvals, eigvecs = torch.linalg.eigh(cov_refined[b])
        eigvals_clipped = torch.clamp(eigvals, min=1e-10)
        cov_refined[b] = eigvecs @ torch.diag(eigvals_clipped) @ eigvecs.T

    diagnostics = {
        'eigenvalue_history': torch.stack(eigenvalue_history, dim=0),  # (n_steps, batch, 5)
        'identifiable_dims_history': torch.stack(identifiable_dims_history, dim=0),  # (n_steps, batch)
        'final_eigenvalues': eigenvalue_history[-1],
        'final_identifiable_dims': identifiable_dims_history[-1]
    }

    return params, cov_refined, diagnostics


class HybridInference:
    """
    Hybrid inference pipeline: Neural proposals + Fisher refinement.

    This implements Strategy A from the theoretical framework:
    1. Neural network provides K initial proposals
    2. Gauss-Newton refinement moves each to local optimum
    3. Fisher information provides posterior covariance
    4. Final estimate achieves near-CR bound where identifiable

    Enhanced with:
    - Nullspace projection (only update identifiable directions)
    - RF-derived frequency weights (redistribute sensitivity)
    """

    def __init__(
        self,
        n_refinement_steps: int = 5,
        damping: float = 0.1,
        noise_std: float = 1.0,
        frequency_weights: Optional[torch.Tensor] = None,
        use_nullspace_projection: bool = True,
        eigenvalue_threshold: float = 1e-6
    ):
        self.n_refinement_steps = n_refinement_steps
        self.damping = damping
        self.noise_std = noise_std
        self.frequency_weights = frequency_weights
        self.use_nullspace_projection = use_nullspace_projection
        self.eigenvalue_threshold = eigenvalue_threshold

    def refine_proposals(
        self,
        proposals: Dict[str, torch.Tensor],
        Z_real_obs: torch.Tensor,
        Z_imag_obs: torch.Tensor,
        omega: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Refine neural network proposals using nullspace-projected Gauss-Newton.

        Args:
            proposals: Dict from FisherAwareTransformer.forward()
            Z_real_obs, Z_imag_obs: (batch, n_freq) observed impedance
            omega: (n_freq,) angular frequencies

        Returns:
            refined: Dict with refined means and Fisher-based covariances
        """
        means = proposals['means']      # (batch, K, n_params)
        weights = proposals['weights']  # (batch, K)

        batch_size, K, n_params = means.shape

        refined_means = torch.zeros_like(means)
        refined_covs = torch.zeros(batch_size, K, n_params, n_params, device=means.device)
        refined_likelihoods = torch.zeros(batch_size, K, device=means.device)
        all_diagnostics = []

        for k in range(K):
            # Refine k-th proposal
            mean_k = means[:, k, :]  # (batch, n_params)

            if self.use_nullspace_projection:
                # Use nullspace-projected refinement
                refined_mean, refined_cov, diagnostics = nullspace_projected_refinement(
                    mean_k,
                    Z_real_obs, Z_imag_obs,
                    omega,
                    frequency_weights=self.frequency_weights,
                    n_steps=self.n_refinement_steps,
                    damping=self.damping,
                    noise_std=self.noise_std,
                    eigenvalue_threshold=self.eigenvalue_threshold
                )
                all_diagnostics.append(diagnostics)
            else:
                # Legacy: standard Gauss-Newton
                refined_mean, refined_cov = gauss_newton_refinement(
                    mean_k,
                    Z_real_obs, Z_imag_obs,
                    omega,
                    n_steps=self.n_refinement_steps,
                    damping=self.damping,
                    noise_std=self.noise_std
                )

            refined_means[:, k, :] = refined_mean
            refined_covs[:, k, :, :] = refined_cov

            # Compute likelihood for model selection (weighted if weights provided)
            params_lin = 10 ** refined_mean
            Ra = params_lin[:, 0]
            Rb = params_lin[:, 1]
            Ca = params_lin[:, 2]
            Cb = params_lin[:, 3]
            Rsh = params_lin[:, 4]
            Z_real_pred, Z_imag_pred = compute_impedance(Ra, Rb, Ca, Cb, Rsh, omega)

            residual_sq = (Z_real_obs - Z_real_pred)**2 + (Z_imag_obs - Z_imag_pred)**2

            if self.frequency_weights is not None:
                w = self.frequency_weights.to(residual_sq.device)
                residual_sq = residual_sq * w.unsqueeze(0)

            log_likelihood = -0.5 * residual_sq.sum(dim=-1) / (self.noise_std ** 2)
            refined_likelihoods[:, k] = log_likelihood

        # Update weights based on refined likelihoods
        refined_weights = torch.softmax(refined_likelihoods, dim=-1)

        result = {
            'means': refined_means,
            'covs': refined_covs,
            'weights': refined_weights,
            'likelihoods': refined_likelihoods
        }

        if all_diagnostics:
            result['diagnostics'] = all_diagnostics

        return result

    def get_best_refined(
        self,
        refined: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the best refined proposal based on likelihood.

        Returns:
            best_mean: (batch, n_params)
            best_cov: (batch, n_params, n_params)
        """
        weights = refined['weights']
        means = refined['means']
        covs = refined['covs']

        best_idx = torch.argmax(weights, dim=-1)
        batch_indices = torch.arange(means.shape[0], device=means.device)

        best_mean = means[batch_indices, best_idx]
        best_cov = covs[batch_indices, best_idx]

        return best_mean, best_cov
