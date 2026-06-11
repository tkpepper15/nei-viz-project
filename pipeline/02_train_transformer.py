#!/usr/bin/env python3
"""
02 - Train Fisher-Aware Transformer (Stage A)

Trains the proposal network for the hybrid inference pipeline.
The transformer learns global frequency-parameter structure via NLL loss,
with an optional auxiliary loss and canonical ordering constraint.

Reparameterization: fully identifiable coordinates
  [Ra, Rb, Ca, Cb, Rsh] -> [tau_big, tau_small, TER, TEC, Rsh]

  tau_big  = max(Ra*Ca, Rb*Cb)  -- dominant time constant
  tau_small= min(Ra*Ca, Rb*Cb)  -- secondary time constant
  TER      = Rsh*(Ra+Rb)/(Rsh+Ra+Rb)  -- DC transepithelial resistance
  TEC      = Ca*Cb/(Ca+Cb)             -- series capacitance
  Rsh      = shunt resistance (paracellular pathway)

  Canonical ordering tau_big >= tau_small breaks the apical/basolateral
  swap symmetry. The model no longer predicts Ra/Rb/Ca/Cb individually
  — those are non-identifiable from 2-electrode EIS. The inverse mapping
  recovers one consistent (Ra,Rb,Ca,Cb) tuple for the reconstruction loss.

Features:
  - Noise augmentation: relative Gaussian noise on Z spectra
  - Identifiable MAE loss: weighted supervision on all 5 identifiable params
  - Canonical ordering loss: soft penalty tau_small <= tau_big

Usage:
    python 02_train_transformer.py --data data/mixed_distribution_v2 --epochs 100 --augment
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.models.fisher_transformer import (
    FisherAwareTransformer,
    TransformerConfig,
)
from src.physics.eis_fisher import compute_impedance


# ---------------------------------------------------------------------------
# Fully identifiable reparameterization
#
# [log10 Ra, Rb, Ca, Cb, Rsh] <-> [log10 tau_big, tau_small, TER, TEC, Rsh]
#
# All five output quantities are identifiable from a broadband EIS spectrum:
#   tau_big   = max(Ra*Ca, Rb*Cb)  dominant arc time constant
#   tau_small = min(Ra*Ca, Rb*Cb)  secondary arc time constant
#   TER       = Rsh*(Ra+Rb)/(Rsh+Ra+Rb)  DC resistance limit
#   TEC       = Ca*Cb/(Ca+Cb)  series capacitance (high-freq limit)
#   Rsh       = shunt/paracellular resistance
#
# Canonical ordering tau_big >= tau_small breaks the discrete apical/
# basolateral swap symmetry without any loss of information.
#
# The degenerate directions (individual Ra, Rb, Ca, Cb) are not represented
# in the model output at all — they are recovered analytically in
# from_identifiable_space() only when needed for the reconstruction loss.
# ---------------------------------------------------------------------------

def _np_to_identifiable(p: np.ndarray) -> np.ndarray:
    """
    [log10 Ra, Rb, Ca, Cb, Rsh] -> [log10 tau_big, tau_small, TER, TEC, Rsh]

    Canonical: tau_big >= tau_small (sorts time constants; symmetric under
    apical/basolateral swap so the label is unique).
    """
    Ra_l, Rb_l, Ca_l, Cb_l, Rsh_l = p
    Ra  = 10.0 ** Ra_l;  Rb  = 10.0 ** Rb_l
    Ca  = 10.0 ** Ca_l;  Cb  = 10.0 ** Cb_l
    Rsh = 10.0 ** Rsh_l

    tau_a = Ra * Ca
    tau_b = Rb * Cb
    tau_big   = max(tau_a, tau_b)
    tau_small = min(tau_a, tau_b)

    S   = Ra + Rb
    TER = Rsh * S / (Rsh + S + 1e-30)
    TEC = Ca * Cb / (Ca + Cb + 1e-30)

    return np.array([
        np.log10(tau_big   + 1e-30),
        np.log10(tau_small + 1e-30),
        np.log10(TER + 1e-30),
        np.log10(TEC + 1e-30),
        Rsh_l,
    ], dtype=np.float32)


def from_identifiable_space(params_id: torch.Tensor) -> torch.Tensor:
    """
    [log10 tau_big, tau_small, TER, TEC, Rsh] -> [log10 Ra, Rb, Ca, Cb, Rsh]

    Analytic inverse used only in the reconstruction loss.

    Derivation:
      S = Ra+Rb = TER*Rsh / (Rsh - TER)
      TEC = tau_small*tau_big / (Ra*(tau_big - tau_small) + S*tau_small)
        => Ra = tau_small*(tau_big/TEC - S) / (tau_big - tau_small)
      Rb = S - Ra,  Ca = tau_small/Ra,  Cb = tau_big/Rb

    Near tau_big == tau_small the formula is ill-conditioned; we blend toward
    the symmetric solution Ra = Rb = S/2 via a sigmoid gate.  The impedance
    spectrum is insensitive to the Ra/Rb split when tau_big ~ tau_small, so
    gradient noise in that direction is harmless.
    """
    log_tb, log_ts, log_TER, log_TEC, log_Rsh = params_id.unbind(-1)
    tau_big   = 10.0 ** log_tb
    tau_small = 10.0 ** log_ts
    TER       = 10.0 ** log_TER
    TEC       = 10.0 ** log_TEC
    Rsh       = 10.0 ** log_Rsh

    # Ra + Rb from TER: TER = Rsh*S/(Rsh+S)
    S = (TER * Rsh / (Rsh - TER + 1e-6)).clamp(min=1e-3)

    # Ra (membrane with tau_small)
    denom    = tau_big - tau_small
    Ra_exact = tau_small * (tau_big / (TEC + 1e-30) - S) / (denom + 1e-6)
    Ra_sym   = S * 0.5

    # Sigmoid gate: 1 when tau_big >> tau_small, 0.5 when equal
    alpha = torch.sigmoid(denom / (tau_big * 0.1 + 1e-12) * 5.0)
    Ra    = (alpha * Ra_exact + (1.0 - alpha) * Ra_sym).clamp(min=1e-3)
    Ra    = Ra.clamp(max=S - 1e-3)
    Rb    = (S - Ra).clamp(min=1e-3)

    Ca = (tau_small / (Ra + 1e-12)).clamp(min=1e-30)
    Cb = (tau_big   / (Rb + 1e-12)).clamp(min=1e-30)

    return torch.stack([
        torch.log10(Ra + 1e-30),
        torch.log10(Rb + 1e-30),
        torch.log10(Ca + 1e-30),
        torch.log10(Cb + 1e-30),
        log_Rsh,
    ], dim=-1)


class EISDataset(Dataset):
    """Dataset for EIS spectra -> parameter mapping with optional augmentation."""

    def __init__(
        self,
        csv_path: Path,
        metadata_path: Path,
        n_freq: int = 100,
        augment: bool = False,
        noise_a: float = 0.02,
        noise_b: float = 0.5,
        noise_c: float = 0.0,
        noise_alpha: float = 1.0,
    ):
        self.df = pd.read_csv(csv_path)
        self.n_freq = n_freq
        self.params = ["Ra", "Rb", "Ca", "Cb", "Rsh"]
        self.augment = augment
        self.noise_a = noise_a
        self.noise_b = noise_b
        self.noise_c = noise_c
        self.noise_alpha = noise_alpha

        with open(metadata_path) as f:
            metadata = json.load(f)

        freq_min = metadata["frequencies"]["min"]
        freq_max = metadata["frequencies"]["max"]
        frequencies = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freq)
        omega = 2 * np.pi * frequencies
        self.log_omega = np.log10(omega).astype(np.float32)
        self.omega_np = omega.astype(np.float32)           # (n_freq,) for noise augmentation
        self.omega = torch.from_numpy(self.omega_np)       # (n_freq,) tensor for BL

        self.Z_real_cols = [f"Z_real_{i}" for i in range(n_freq)]
        self.Z_imag_cols = [f"Z_imag_{i}" for i in range(n_freq)]

        print(f"Loaded {len(self.df)} samples from {csv_path.name}"
              f"{' [augment=ON]' if augment else ''}")

    def __len__(self):
        return len(self.df)

    def _noise_sigma(self, Z_mag: np.ndarray) -> np.ndarray:
        """
        Frequency-dependent noise standard deviation.

            sigma(omega) = sqrt(a^2 * |Z|^2 + b^2 + c / omega^alpha)

        - a * |Z|: relative component (instrument gain noise)
        - b:       absolute floor (quantisation / amplifier noise)
        - c / omega^alpha: low-frequency drift (1/f or random-walk type)

        Defaults (c=0, alpha=1) reproduce the original 2%+0.5Ohm model.
        Once real measurements are characterised, plug in fitted a, b, c, alpha.
        """
        return np.sqrt(
            self.noise_a ** 2 * Z_mag ** 2
            + self.noise_b ** 2
            + self.noise_c / (self.omega_np ** self.noise_alpha + 1e-12)
        ).astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        Z_real = row[self.Z_real_cols].values.astype(np.float32).copy()
        Z_imag = row[self.Z_imag_cols].values.astype(np.float32).copy()
        log_omega = self.log_omega.copy()
        params_log10 = np.array(
            [np.log10(row[p]) for p in self.params], dtype=np.float32
        )
        # Reparameterise: [Ra,Rb,Ca,Cb,Rsh] -> [tau_big, tau_small, TER, TEC, Rsh]
        # Canonical ordering (tau_big >= tau_small) is enforced inside _np_to_identifiable,
        # so no label augmentation is needed — the swap is already absorbed.
        params_log10 = _np_to_identifiable(params_log10)

        if self.augment:
            # Frequency-dependent noise augmentation only.
            # The apical/basolateral swap augmentation is no longer needed:
            # _np_to_identifiable canonicalises tau_big >= tau_small, so
            # swapping (Ra,Ca)<->(Rb,Cb) produces the same label.
            noise_scale = np.random.uniform(0.0, 1.0)
            Z_mag = np.sqrt(Z_real ** 2 + Z_imag ** 2 + 1e-12)
            sigma = self._noise_sigma(Z_mag)
            Z_real = Z_real + noise_scale * sigma * np.random.randn(self.n_freq).astype(np.float32)
            Z_imag = Z_imag + noise_scale * sigma * np.random.randn(self.n_freq).astype(np.float32)

        return {
            "Z_real": torch.from_numpy(Z_real),
            "Z_imag": torch.from_numpy(Z_imag),
            "log_omega": torch.from_numpy(log_omega),
            "omega": self.omega,
            "params_log10": torch.from_numpy(params_log10),
        }


def compute_identifiable_mae_loss(
    proposals: dict,
    targets: torch.Tensor,
    model: FisherAwareTransformer,
    sep_weight: float = 2.0,
    tau_big_weight: float = 2.0,
    ter_weight: float = 1.5,
    rsh_weight: float = 1.5,
) -> torch.Tensor:
    """
    Separation-aware weighted MAE on the 5 identifiable params.

    Params: [tau_big, tau_small, TER, TEC, Rsh] — all in log10 space.

    Base weights reflect identifiability from EIS spectra:
      tau_big   2.0  -- dominant arc, direct readout from phase peak
      tau_small 1.0  -- secondary arc
      TER       ter_weight  -- configurable; CRB=0.008 decades means TER is highly constrained
      TEC       2.0  -- high-freq; only identifiable capacitance quantity
      Rsh       rsh_weight  -- relatively independent (low-freq plateau)

    For the tau pair specifically, we apply a separation-aware multiplier:
      delta_tau = (tau_big - tau_small) / (tau_big + tau_small + eps)
      w_tau = 1 + sep_weight * delta_tau

    When the two time constants are far apart (delta_tau near 1), the tau
    predictions are directly readable from distinct arc peaks — push harder.
    When they are near-equal (delta_tau near 0), the problem is genuinely
    ill-posed (merged poles) — don't over-penalize, let the NLL handle it.

    This matches the Fisher information structure: large FIM eigenvalue in the
    tau_big/tau_small directions when they are well-separated, near-zero when merged.
    """
    mix_mean, _ = model.get_mixture_posterior(proposals)   # (batch, 5)

    # Separation measure on the TARGET tau values (ground truth, not prediction)
    tau_big_t   = 10.0 ** targets[:, 0]
    tau_small_t = 10.0 ** targets[:, 1]
    delta_tau   = (tau_big_t - tau_small_t) / (tau_big_t + tau_small_t + 1e-8)   # (batch,) in [0,1]
    w_tau = (1.0 + sep_weight * delta_tau).detach()   # no gradient through the weight

    # Base static weights — tau_big, TER, and Rsh are configurable via CLI.
    # TEC kept at 2.0: it is the only identifiable capacitance quantity and
    # was previously under-weighted (0.5), causing the backbone to ignore the
    # high-frequency band where Ca/Cb information lives.
    base_weights = torch.tensor([tau_big_weight, 1.0, ter_weight, 2.0, rsh_weight], device=targets.device)
    total_static = base_weights.sum()

    abs_err = torch.abs(mix_mean - targets)   # (batch, 5)

    # Apply separation-aware weight to tau_big (idx 0) and tau_small (idx 1)
    abs_err[:, 0] = abs_err[:, 0] * w_tau
    abs_err[:, 1] = abs_err[:, 1] * w_tau

    loss = (abs_err * base_weights.unsqueeze(0)).sum(dim=1).mean()
    return loss / total_static


def compute_ter_anchor_loss(
    proposals: dict,
    model: FisherAwareTransformer,
    Z_real_obs: torch.Tensor,
    n_low_freq: int = 5,
) -> torch.Tensor:
    """
    Tie TER prediction to the low-frequency Z_real measurements.

    At low ω, Re[Z(ω)] → TER (the DC resistance limit of the RPE circuit).
    The 5 lowest-frequency Z_real samples are therefore a direct TER proxy.
    This loss closes the gradient path from the low-frequency band to the TER
    output that is otherwise starved when the reconstruction loss uses high-freq
    emphasis (omega_w = omega / omega.mean()) and the derived MAE weight is low.

    Loss: MAE in log10 space between predicted TER and the low-freq Z_real proxy.
    """
    mix_mean, _ = model.get_mixture_posterior(proposals)   # (batch, 5) identifiable
    TER_pred_log10 = mix_mean[:, 2]                        # log10(TER), (batch,)

    # Low-frequency Z_real average as TER proxy (Z_real is in Ohm, linear)
    # Z_real_obs shape: (batch, n_freq); first columns are lowest frequencies.
    ter_proxy_lin = Z_real_obs[:, :n_low_freq].mean(dim=1).clamp(min=1e-3)  # (batch,)
    ter_proxy_log10 = torch.log10(ter_proxy_lin)

    return torch.abs(TER_pred_log10 - ter_proxy_log10).mean()


def canonical_ordering_loss(proposals: dict) -> torch.Tensor:
    """
    Soft penalty to encourage tau_big >= tau_small across all K proposals.

    In the identifiable space, index 0 = log(tau_big) and index 1 = log(tau_small).
    Training labels satisfy tau_big >= tau_small by construction, so this loss
    prevents the model from inverting the ordering at inference time.
    """
    means = proposals["means"]   # (batch, K, 5)
    log_tau_big   = means[:, :, 0]
    log_tau_small = means[:, :, 1]
    return F.relu(log_tau_small - log_tau_big).mean()


def compute_reconstruction_loss(
    proposals: dict,
    model: FisherAwareTransformer,
    Z_real_obs: torch.Tensor,
    Z_imag_obs: torch.Tensor,
    omega: torch.Tensor,
    low_freq_blend: float = 0.0,
) -> torch.Tensor:
    """
    Forward reconstruction loss in polar log space.

    Measures error as:
        (log10|Z_pred| - log10|Z_obs|)^2  +  (phase_pred - phase_obs)^2

    Both terms are dimensionless and O(1) at initialization (log-magnitude error
    of ~1-2 decades, phase error of ~0.5-1 rad), making this compatible with the
    NLL loss scale of ~7 nats without needing tiny weights.

    At convergence, a good fit has log-magnitude error < 0.01 and phase error
    < 0.01 rad, giving a loss contribution < 0.001.

    Gradients flow through compute_impedance back to the mixture mean, forcing
    the predicted parameters to actually reproduce the observed spectrum rather
    than just match training labels statistically.

    low_freq_blend controls the frequency weighting strategy:
      0.0  -- high-freq only (original behavior, omega / mean(omega))
      0.5  -- bimodal: equal mix of high-freq and low-freq emphasis
      1.0  -- low-freq only (1/omega, normalized)
    At blend > 0, the low-frequency end (where TER dominates) receives explicit
    gradient pressure rather than being down-weighted by the high-freq emphasis.
    """
    mix_mean, _ = model.get_mixture_posterior(proposals)   # (batch, 5) identifiable space
    mix_mean_orig = from_identifiable_space(mix_mean)      # back to [Ra,Rb,Ca,Cb,Rsh]

    params_lin = 10 ** mix_mean_orig
    Z_real_pred, Z_imag_pred = compute_impedance(
        params_lin[:, 0],   # Ra
        params_lin[:, 1],   # Rb
        params_lin[:, 2],   # Ca
        params_lin[:, 3],   # Cb
        params_lin[:, 4],   # Rsh
        omega,
    )

    # Polar log space — same coordinates as FrequencyTokenizer
    log_mag_pred  = torch.log10(torch.sqrt(Z_real_pred**2 + Z_imag_pred**2 + 1e-12))
    phase_pred    = torch.atan2(Z_imag_pred, Z_real_pred)

    log_mag_obs   = torch.log10(torch.sqrt(Z_real_obs**2 + Z_imag_obs**2 + 1e-12))
    phase_obs     = torch.atan2(Z_imag_obs, Z_real_obs)

    # Frequency weighting:
    # High-freq component: omega / mean(omega) — emphasises Ca/Cb band
    # Low-freq component:  mean(omega) / omega — emphasises TER/Rsh band (DC limit)
    # Bimodal blend (low_freq_blend in [0,1]) mixes both components.
    omega_f = omega.float()                                      # (n_freq,)
    omega_mean = omega_f.mean() + 1e-12
    w_high = omega_f / omega_mean                                # normalised to mean=1
    w_low  = omega_mean / (omega_f + 1e-12)
    w_low  = w_low / (w_low.mean() + 1e-12)                     # also normalised to mean=1
    omega_w = ((1.0 - low_freq_blend) * w_high
               + low_freq_blend       * w_low).unsqueeze(0)     # (1, n_freq) for broadcast

    recon = (log_mag_pred - log_mag_obs) ** 2 + (phase_pred - phase_obs) ** 2
    return (recon * omega_w).mean()


def compute_sigma_regularization(proposals: dict) -> torch.Tensor:
    """
    Penalize large predicted variances to combat σ inflation.

    The NLL objective has a well-known failure mode: when the model is
    uncertain about a prediction, it can minimise NLL by inflating σ
    rather than improving μ.  This penalty directly counteracts that by
    adding mean(log σ) — which is monotone in σ — to the loss.

    Only the diagonal variances are penalized (the low-rank factors are
    indirectly constrained through Cholesky).  The weighted mixture mean
    log-σ is used so collapsed components don't dominate.

    λ ≈ 0.05 is a good starting point.  Increase if σ/MAE > 2; decrease
    if the model becomes overconfident (σ/MAE < 0.8).
    """
    covs    = proposals['covs']     # (batch, K, 5, 5)
    weights = proposals['weights']  # (batch, K)

    # Diagonal variances per component: (batch, K, 5)
    diag_vars = torch.diagonal(covs, dim1=-2, dim2=-1)
    log_sigma = 0.5 * torch.log(diag_vars.clamp(min=1e-8))   # log(σ), (batch, K, 5)

    # Weight by mixture weights so dead components don't inflate the loss
    w = weights.unsqueeze(-1)                          # (batch, K, 1)
    weighted_log_sigma = (w * log_sigma).sum(dim=1)    # (batch, 5)

    return weighted_log_sigma.mean()                   # scalar; minimising pushes σ down


def compute_mdn_diversity_loss(proposals: dict) -> torch.Tensor:
    """
    Two-term MDN diversity loss to prevent component collapse.

    Term 1 — repulsion between component means:
        L_rep = -mean_{j≠k} ||μ_j - μ_k||_2
        Pushes component means apart so the MDN explores distinct hypotheses.

    Term 2 — best-proposal auxiliary supervision (hard-EM style):
        For each training example, find the component k* whose mean is closest
        to the NLL-weighted mixture mean, then add a direct MAE term on that
        component.  This ensures at least one component stays near a good solution
        even when the NLL allows all components to cluster at the same point.

    Without this, NLL is minimised by concentrating all K components on the
    empirical mean of the training distribution — which is exactly the collapse
    seen in the Ca/Cb visualisation (all proposals ≈ 0.75).
    """
    means   = proposals["means"]    # (batch, K, n_params)
    weights = proposals["weights"]  # (batch, K)

    # Term 1: margin-hinge repulsion between component means.
    # Pushes components apart until they exceed `margin` decades of separation,
    # then the loss goes to zero — preventing unbounded gradient growth.
    margin = 0.5  # target minimum separation in log10 space (0.5 decades)
    K = means.shape[1]
    rep_loss = torch.tensor(0.0, device=means.device)
    n_pairs = 0
    for j in range(K):
        for k in range(j + 1, K):
            dist = torch.norm(means[:, j, :] - means[:, k, :], dim=-1)  # (batch,)
            rep_loss = rep_loss + F.relu(margin - dist).mean()  # 0 once dist > margin
            n_pairs += 1
    if n_pairs > 0:
        rep_loss = rep_loss / n_pairs

    # Term 2: weight entropy — penalise weight collapse toward one dominant component
    # H = -sum_k w_k log(w_k); maximising H keeps weights spread across components
    eps = 1e-8
    entropy = -(weights * torch.log(weights + eps)).sum(dim=-1).mean()
    entropy_loss = -entropy   # negative: we want to maximise entropy

    return rep_loss + 0.5 * entropy_loss


def train_epoch(
    model, dataloader, optimizer, device, epoch, total_epochs,
    derived_weight, manifold_weight, recon_weight, diversity_weight,
    tau_big_weight=2.0, ter_weight=1.5, rsh_weight=1.5,
    sigma_reg_weight=0.0, ter_anchor_weight=0.0, recon_low_freq_blend=0.0,
):
    """Train for one epoch with NLL, derived, manifold, reconstruction, diversity, sigma-reg, and TER-anchor losses."""
    model.train()
    total_loss       = 0.0
    total_nll        = 0.0
    total_derived    = 0.0
    total_manifold   = 0.0
    total_recon      = 0.0
    total_diversity  = 0.0
    total_sigma_reg  = 0.0
    total_ter_anchor = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for batch in pbar:
        Z_real = batch["Z_real"].to(device)
        Z_imag = batch["Z_imag"].to(device)
        log_omega = batch["log_omega"].to(device)
        omega = batch["omega"].to(device)
        targets = batch["params_log10"].to(device)

        proposals = model(Z_real, Z_imag, log_omega)
        loss_nll = model.compute_nll_loss(proposals, targets)

        loss_derived    = torch.tensor(0.0, device=device)
        loss_manifold   = torch.tensor(0.0, device=device)
        loss_recon      = torch.tensor(0.0, device=device)
        loss_diversity  = torch.tensor(0.0, device=device)
        loss_sigma_reg  = torch.tensor(0.0, device=device)
        loss_ter_anchor = torch.tensor(0.0, device=device)

        if derived_weight > 0:
            loss_derived = compute_identifiable_mae_loss(
                proposals, targets, model,
                tau_big_weight=tau_big_weight, ter_weight=ter_weight, rsh_weight=rsh_weight,
            )

        if manifold_weight > 0:
            loss_manifold = canonical_ordering_loss(proposals)

        if recon_weight > 0:
            # omega is (batch, n_freq) after DataLoader stacking; all rows are
            # identical (same frequency grid), so pass just the first row.
            loss_recon = compute_reconstruction_loss(
                proposals, model, Z_real, Z_imag, omega[0],
                low_freq_blend=recon_low_freq_blend,
            )

        if diversity_weight > 0:
            loss_diversity = compute_mdn_diversity_loss(proposals)

        if ter_anchor_weight > 0:
            loss_ter_anchor = compute_ter_anchor_loss(proposals, model, Z_real)

        if sigma_reg_weight > 0:
            loss_sigma_reg = compute_sigma_regularization(proposals)

        loss = (loss_nll
                + derived_weight    * loss_derived
                + manifold_weight   * loss_manifold
                + recon_weight      * loss_recon
                + diversity_weight  * loss_diversity
                + sigma_reg_weight  * loss_sigma_reg
                + ter_anchor_weight * loss_ter_anchor)

        if torch.isnan(loss):
            print("WARNING: NaN loss detected, skipping batch")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss       += loss.item()
        total_nll        += loss_nll.item()
        total_derived    += loss_derived.item()
        total_manifold   += loss_manifold.item()
        total_recon      += loss_recon.item()
        total_diversity  += loss_diversity.item()
        total_sigma_reg  += loss_sigma_reg.item()
        total_ter_anchor += loss_ter_anchor.item()
        n_batches += 1
        pbar.set_postfix({
            "loss": f"{total_loss/n_batches:.4f}",
            "nll":  f"{total_nll/n_batches:.4f}",
            "rec":  f"{total_recon/n_batches:.4f}",
            "ter":  f"{total_ter_anchor/n_batches:.4f}",
            "sig":  f"{total_sigma_reg/n_batches:.4f}",
        })

    n = max(n_batches, 1)
    return (total_loss / n, total_nll / n, total_derived / n,
            total_manifold / n, total_recon / n, total_diversity / n,
            total_sigma_reg / n, total_ter_anchor / n)


def evaluate(model, dataloader, device, desc="Eval"):
    """
    Evaluate NLL loss + per-parameter MAE + delta_tau diagnostics.

    Targets are in identifiable space [tau_big, tau_small, TER, TEC, Rsh].

    Also computes:
      delta_tau = (tau_big - tau_small) / (tau_big + tau_small)
    for both targets (data distribution) and predictions, to detect if the
    model is collapsing toward symmetric (delta_tau -> 0) solutions.
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    param_names = ["tau_big", "tau_small", "TER", "TEC", "Rsh"]
    param_errors = {p: [] for p in param_names}
    delta_tau_true_all = []
    delta_tau_pred_all = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"[{desc}]", leave=False):
            Z_real    = batch["Z_real"].to(device)
            Z_imag    = batch["Z_imag"].to(device)
            log_omega = batch["log_omega"].to(device)
            targets   = batch["params_log10"].to(device)

            proposals = model(Z_real, Z_imag, log_omega)
            loss = model.compute_nll_loss(proposals, targets)

            if not torch.isnan(loss):
                total_loss += loss.item()
                n_batches  += 1

            mix_mean, _ = model.get_mixture_posterior(proposals)

            for i, p in enumerate(param_names):
                errors = torch.abs(mix_mean[:, i] - targets[:, i])
                param_errors[p].extend(errors.cpu().numpy().tolist())

            # delta_tau on targets (ground truth distribution)
            tb_t = 10.0 ** targets[:, 0];  ts_t = 10.0 ** targets[:, 1]
            dt_true = ((tb_t - ts_t) / (tb_t + ts_t + 1e-8)).cpu().numpy()
            delta_tau_true_all.extend(dt_true.tolist())

            # delta_tau on predictions (collapse diagnostic)
            tb_p = 10.0 ** mix_mean[:, 0];  ts_p = 10.0 ** mix_mean[:, 1]
            dt_pred = ((tb_p - ts_p) / (tb_p + ts_p + 1e-8)).cpu().numpy()
            delta_tau_pred_all.extend(dt_pred.tolist())

    avg_loss    = total_loss / max(n_batches, 1)
    avg_mae     = {p: np.mean(e) for p, e in param_errors.items()}
    overall_mae = float(np.mean(list(avg_mae.values())))

    diag = {
        "delta_tau_true_mean": float(np.mean(delta_tau_true_all)),
        "delta_tau_pred_mean": float(np.mean(delta_tau_pred_all)),
        "delta_tau_pred_p10":  float(np.percentile(delta_tau_pred_all, 10)),
        "delta_tau_pred_p50":  float(np.percentile(delta_tau_pred_all, 50)),
        "delta_tau_pred_p90":  float(np.percentile(delta_tau_pred_all, 90)),
    }
    return avg_loss, avg_mae, overall_mae, diag


def main():
    parser = argparse.ArgumentParser(description="Train Fisher-Aware Transformer (Stage A)")
    parser.add_argument("--data", type=str, default="data/mixed_distribution_v2")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-proposals", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", type=str, default="models/fisher_transformer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--derived-weight", type=float, default=0.8,
                        help="Weight for derived quantity MAE loss (0 to disable). "
                             "Raised from 0.3: tau_b and Ra+Rb supervision matters most.")
    parser.add_argument("--manifold-weight", type=float, default=0.3,
                        help="Weight for Ca-Cb degeneracy variance penalty (0 to disable).")
    parser.add_argument("--cov-rank", type=int, default=4,
                        help="Rank of low-rank covariance factor. 4 captures all key "
                             "parameter correlations (Ra-Ca, Rb-Cb, Ca-Cb, Rsh-TER).")
    parser.add_argument("--augment", action="store_true", default=False,
                        help="Enable symmetry + noise augmentation during training")
    parser.add_argument("--no-augment", action="store_false", dest="augment")
    parser.add_argument("--grad-features", action="store_true", default=True,
                        help="Append spectral gradient features (d/dlogf, d²/dlogf²) to each token")
    parser.add_argument("--no-grad-features", action="store_false", dest="grad_features")
    parser.add_argument("--smooth-sigma", type=float, default=0.5,
                        help="Gaussian smoothing width in decades applied to Z_real/Z_imag "
                             "before gradient features are computed. Prevents ~350x noise "
                             "amplification from finite differences. 0 to disable.")
    parser.add_argument("--drt", action="store_true", default=True,
                        help="Append DRT γ(τ) spectrum as an additional per-frequency feature. "
                             "Ridge-regression solve with tau_k = 1/omega_k alignment.")
    parser.add_argument("--no-drt", action="store_false", dest="drt")
    parser.add_argument("--drt-lambda", type=float, default=1e-3,
                        help="Tikhonov regularization strength for the DRT ridge solve.")
    parser.add_argument("--recon-weight", type=float, default=0.5,
                        help="Weight for physics reconstruction loss (0 to disable). "
                             "Forces predicted parameters to reproduce the observed spectrum.")
    parser.add_argument("--diversity-weight", type=float, default=0.2,
                        help="Weight for MDN diversity loss (0 to disable). "
                             "Prevents component collapse via inter-component repulsion "
                             "and mixture-weight entropy regularisation.")
    # Noise model parameters — shared by training augmentation and BL refinement.
    # Defaults reproduce the original 2%+0.5Ohm model (c=0 disables drift term).
    # Once real EIS measurements are characterised (repeated sweeps on a stable
    # reference), fit sigma(omega)=sqrt(a^2*|Z|^2 + b^2 + c/omega^alpha) and
    # plug in the fitted values here.
    parser.add_argument("--noise-a", type=float, default=0.02,
                        help="Relative noise fraction (instrument gain noise).")
    parser.add_argument("--noise-b", type=float, default=0.5,
                        help="Absolute noise floor in Ohm (quantisation / amplifier noise).")
    parser.add_argument("--noise-c", type=float, default=0.0,
                        help="Low-frequency drift amplitude. 0 disables drift term.")
    parser.add_argument("--noise-alpha", type=float, default=1.0,
                        help="Drift frequency exponent in c/omega^alpha.")
    parser.add_argument("--tau-big-weight", type=float, default=2.0,
                        help="Loss weight for tau_big in identifiable MAE loss. "
                             "Raise above 2.0 to push harder on dominant time constant.")
    parser.add_argument("--ter-weight", type=float, default=1.5,
                        help="Loss weight for TER in identifiable MAE loss. "
                             "Default 1.5 preserves v7 behavior. "
                             "CRB for TER is 0.008 decades — raise to 5-6 for v8 to reflect "
                             "that TER is information-theoretically highly constrained.")
    parser.add_argument("--rsh-weight", type=float, default=1.5,
                        help="Loss weight for Rsh in identifiable MAE loss.")
    parser.add_argument("--ter-anchor-weight", type=float, default=0.0,
                        help="Weight for TER anchor loss. Ties TER prediction to the "
                             "5 lowest-frequency Z_real values (direct DC resistance proxy). "
                             "Provides gradient from measured data, not just parameter labels. "
                             "Start at 1.0-2.0 when tuning TER.")
    parser.add_argument("--recon-low-freq-blend", type=float, default=0.0,
                        help="Blend fraction for low-frequency reconstruction weighting. "
                             "0.0 = high-freq only (original). "
                             "0.5 = bimodal (equal high- and low-freq emphasis). "
                             "1.0 = low-freq only. "
                             "Values > 0 give the TER / Rsh band explicit gradient pressure "
                             "in the reconstruction loss.")
    parser.add_argument("--sigma-reg-weight", type=float, default=0.0,
                        help="Weight for sigma regularization loss (0 to disable). "
                             "Penalizes large predicted variances (mean log-sigma across "
                             "mixture components) to combat the NLL inflation failure mode "
                             "where the model inflates sigma instead of improving mu. "
                             "Start at 0.05; increase if sigma/MAE > 2 after training.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a best_model.pt checkpoint to warm-start from. "
                             "Loads model weights (and optimizer state if present). "
                             "The LR schedule and epoch counter restart from 1.")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze tokenizer + encoder_layers + final_norm. Only "
                             "pooling and proposal_head are trained. Use when resuming "
                             "from a checkpoint with good tau_big to prevent catastrophic "
                             "forgetting while fine-tuning TER/Rsh via the output heads.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("Stage A: Train Fisher-Aware Transformer")
    print("=" * 70)
    print(f"  Data:           {args.data}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  LR:             {args.lr}")
    print(f"  d_model:        {args.d_model}")
    print(f"  Layers:         {args.n_layers}")
    print(f"  Proposals:      {args.n_proposals}")
    print(f"  Derived weight:   {args.derived_weight}")
    print(f"  Manifold weight:  {args.manifold_weight}")
    print(f"  Recon weight:     {args.recon_weight}")
    print(f"  Diversity weight: {args.diversity_weight}")
    print(f"  Sigma reg weight: {args.sigma_reg_weight}")
    print(f"  Cov rank:         {args.cov_rank}")
    print(f"  Augmentation:     {args.augment}")
    print(f"  Grad features:    {args.grad_features}")
    print(f"  Smooth sigma:     {args.smooth_sigma} decades")
    print(f"  DRT features:     {args.drt} (lambda={args.drt_lambda})")
    print(f"  Noise model:    a={args.noise_a} b={args.noise_b} "
          f"c={args.noise_c} alpha={args.noise_alpha}")
    print(f"  tau_big weight: {args.tau_big_weight}")
    print(f"  TER weight:     {args.ter_weight}")
    print(f"  Rsh weight:     {args.rsh_weight}")
    print(f"  TER anchor weight: {args.ter_anchor_weight}")
    print(f"  Recon low-freq blend: {args.recon_low_freq_blend}")
    print(f"  Resume from:    {args.resume or 'none (fresh start)'}")

    # Device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  Device:         {device}")

    # Data
    data_dir = Path(args.data)
    metadata_path = data_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    if "frequencies" in metadata:
        n_freq = metadata["frequencies"]["count"]
    else:
        # Legacy format (physics_constrained_corrected): fixed 100-point grid
        n_freq = 100

    noise_kwargs = dict(
        noise_a=args.noise_a,
        noise_b=args.noise_b,
        noise_c=args.noise_c,
        noise_alpha=args.noise_alpha,
    )
    train_dataset = EISDataset(
        data_dir / "train.csv", metadata_path, n_freq,
        augment=args.augment, **noise_kwargs,
    )
    val_dataset = EISDataset(
        data_dir / "val.csv", metadata_path, n_freq,
        augment=False, **noise_kwargs,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    config = TransformerConfig(
        n_freq=n_freq,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_model * 4,
        dropout=0.1,
        n_proposals=args.n_proposals,
        n_params=5,
        use_low_rank_cov=True,
        cov_rank=args.cov_rank,
        use_grad_features=args.grad_features,
        smooth_sigma_decades=args.smooth_sigma,
        use_drt=args.drt,
        drt_lambda=args.drt_lambda,
    )
    model = FisherAwareTransformer(config).to(device)
    n_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params:         {n_model_params:,}")

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")
        ckpt = torch.load(str(resume_path), map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        print(f"  Resumed from:   {resume_path} (epoch {ckpt.get('epoch', '?')}, "
              f"val_mae={ckpt.get('val_mae', float('nan')):.4f})")
    else:
        # Override ProposalHead initialization for identifiable space.
        # Defaults in the model assume [Ra,Rb,Ca,Cb,Rsh] (log10 ~ [2,2,-6,-6,2.7]).
        # Identifiable space [tau_big, tau_small, TER, TEC, Rsh] has very different scales.
        init_means = torch.tensor([-4.0, -4.5, 2.5, -6.5, 2.7])
        model.proposal_head.mean_net.bias.data = init_means.repeat(config.n_proposals).to(device)

    if args.freeze_encoder:
        frozen_modules = [model.tokenizer, *model.encoder_layers, model.final_norm]
        for mod in frozen_modules:
            for param in mod.parameters():
                param.requires_grad = False
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        n_frozen = sum(p.numel() for mod in frozen_modules for p in mod.parameters())
        n_trainable = sum(p.numel() for p in trainable_params)
        print(f"  Encoder frozen: {n_frozen:,} params frozen, {n_trainable:,} trainable "
              f"(pooling + proposal_head only)")
    else:
        trainable_params = list(model.parameters())

    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training
    print("\n" + "=" * 70)
    best_val_mae = float("inf")
    training_log = []

    for epoch in range(1, args.epochs + 1):
        (train_loss, train_nll, train_derived, train_manifold,
         train_recon, train_diversity, train_sigma_reg, train_ter_anchor) = train_epoch(
            model, train_loader, optimizer, device, epoch, args.epochs,
            derived_weight=args.derived_weight,
            manifold_weight=args.manifold_weight,
            recon_weight=args.recon_weight,
            diversity_weight=args.diversity_weight,
            tau_big_weight=args.tau_big_weight,
            ter_weight=args.ter_weight,
            rsh_weight=args.rsh_weight,
            sigma_reg_weight=args.sigma_reg_weight,
            ter_anchor_weight=args.ter_anchor_weight,
            recon_low_freq_blend=args.recon_low_freq_blend,
        )
        val_loss, val_mae, overall_mae, diag = evaluate(model, val_loader, device, desc="Val")
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_nll": train_nll,
            "train_derived": train_derived,
            "train_manifold": train_manifold,
            "train_recon": train_recon,
            "train_diversity": train_diversity,
            "train_sigma_reg": train_sigma_reg,
            "train_ter_anchor": train_ter_anchor,
            "val_loss": val_loss,
            "val_mae": overall_mae,
            "lr": lr,
            **{f"mae_{p}": v for p, v in val_mae.items()},
            **diag,
        }
        training_log.append(log_entry)
        pd.DataFrame(training_log).to_csv(output_dir / "training_log.csv", index=False)

        print(f"\nEpoch {epoch:3d}/{args.epochs}")
        print(f"  Loss: total={train_loss:.4f}  nll={train_nll:.4f}  "
              f"id_mae={train_derived:.4f}  canonical={train_manifold:.4f}  "
              f"recon={train_recon:.4f}  diversity={train_diversity:.4f}  "
              f"ter_anchor={train_ter_anchor:.4f}  sigma_reg={train_sigma_reg:.4f}")
        print(f"  Val NLL: {val_loss:.4f}  Val MAE: {overall_mae:.4f} decades")
        print(f"    tau_big={val_mae['tau_big']:.3f}  tau_small={val_mae['tau_small']:.3f}  "
              f"TER={val_mae['TER']:.3f}  TEC={val_mae['TEC']:.3f}  Rsh={val_mae['Rsh']:.3f}")
        print(f"  delta_tau: data={diag['delta_tau_true_mean']:.3f}  "
              f"pred p10/p50/p90={diag['delta_tau_pred_p10']:.3f}/"
              f"{diag['delta_tau_pred_p50']:.3f}/{diag['delta_tau_pred_p90']:.3f}")

        if overall_mae < best_val_mae:
            best_val_mae = overall_mae
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "param_space": "identifiable",   # signals downstream code to convert
                    "val_mae": overall_mae,
                    "val_param_mae": val_mae,
                    "augment": args.augment,
                    "derived_weight": args.derived_weight,
                },
                output_dir / "best_model.pt",
            )
            print(f"  >> Best model saved (val MAE: {overall_mae:.4f})")

        if epoch % 20 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.__dict__,
                },
                output_dir / f"checkpoint_epoch_{epoch}.pt",
            )

    # Final save
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
        },
        output_dir / "final_model.pt",
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"  Best val MAE:     {best_val_mae:.4f} decades (identifiable space)")
    print(f"  Model saved to:   {output_dir}")
    print(f"  Augmentation:     {args.augment}")
    print(f"  Derived weight:   {args.derived_weight}")
    print("\nNext: python 03_train_rf_weights.py")


if __name__ == "__main__":
    main()
