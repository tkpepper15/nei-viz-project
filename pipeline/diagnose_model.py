#!/usr/bin/env python3
"""
Three diagnostic checks for the fisher transformer:

1. Dead neurons / MDN component collapse
   - Fraction of activation neurons near zero (GELU can suppress but not kill)
   - MDN component weight distribution — are any components always ignored?
   - Effective number of mixture components (entropy of weight distribution)

2. Calibration
   - Are the predicted sigma values consistent with actual errors?
   - For each parameter: fraction of test samples where |err| < 1σ, 2σ, 3σ
   - Expected calibration error (ECE) across parameters
   - Well-calibrated Gaussian: 68% within 1σ, 95% within 2σ, 99.7% within 3σ

3. Covariate shift
   - Compare test CSV distribution to training CSV on key statistics
   - Compare parameter ranges seen during training vs test
   - Check if model predictions shift significantly across tau_ratio strata
"""

import sys
import csv
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from src.models.fisher_transformer import FisherAwareTransformer, TransformerConfig


# ── Load model ────────────────────────────────────────────────────────────────

def load_model(model_dir="models/fisher_v10"):
    ckpt = torch.load(
        Path(model_dir) / "best_model.pt",
        map_location="cpu", weights_only=False
    )
    cfg_data = ckpt.get("config", {})
    config = TransformerConfig(**cfg_data)
    model = FisherAwareTransformer(config)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, ckpt


# ── Load test data ─────────────────────────────────────────────────────────

def load_test_data(data_dir="data/mixed_distribution_v2", max_samples=2000):
    path = Path(data_dir) / "test.csv"
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if len(rows) >= max_samples:
                break

    n_freq = 100
    Z_real = np.array([[float(r[f"Z_real_{i}"]) for i in range(n_freq)] for r in rows])
    Z_imag = np.array([[float(r[f"Z_imag_{i}"]) for i in range(n_freq)] for r in rows])

    targets = np.array([
        [np.log10(float(r["tau_big"])),
         np.log10(float(r["tau_small"])),
         np.log10(float(r["TER"])),
         np.log10(float(r["TEC"])),
         np.log10(float(r["Rsh"]))]
        for r in rows
    ])

    tau_ratios = np.array([float(r["tau_ratio"]) for r in rows])
    return Z_real, Z_imag, targets, tau_ratios, rows


def load_train_data(data_dir="data/mixed_distribution_v2", max_samples=5000):
    path = Path(data_dir) / "train.csv"
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if len(rows) >= max_samples:
                break

    targets = np.array([
        [np.log10(float(r["tau_big"])),
         np.log10(float(r["tau_small"])),
         np.log10(float(r["TER"])),
         np.log10(float(r["TEC"])),
         np.log10(float(r["Rsh"]))]
        for r in rows
    ])
    return targets


# ── Inference ──────────────────────────────────────────────────────────────

def run_inference(model, Z_real, Z_imag, batch_size=128):
    freqs = np.logspace(-1, 6, 100)
    log_omega = np.log10(2 * np.pi * freqs)

    all_means = []
    all_covs  = []
    all_weights = []
    all_mix_means = []
    all_mix_stds  = []
    all_hook_acts = []

    # Hook to capture FFN activations from last encoder layer
    ffn_acts = []
    def hook_fn(module, inp, out):
        ffn_acts.append(out.detach().cpu())
    hook = model.encoder_layers[-1].ffn[1].register_forward_hook(hook_fn)  # after GELU

    n = len(Z_real)
    for i in range(0, n, batch_size):
        Zr = torch.tensor(Z_real[i:i+batch_size], dtype=torch.float32)
        Zi = torch.tensor(Z_imag[i:i+batch_size], dtype=torch.float32)
        lo = torch.tensor(log_omega, dtype=torch.float32).unsqueeze(0).expand(len(Zr), -1)

        with torch.no_grad():
            props = model(Zr, Zi, lo)

        means   = props['means'].cpu().numpy()    # (B, K, 5)
        covs    = props['covs'].cpu().numpy()     # (B, K, 5, 5)
        weights = props['weights'].cpu().numpy()  # (B, K)

        all_means.append(means)
        all_covs.append(covs)
        all_weights.append(weights)

        # Mixture mean and std
        w = weights[:, :, None]
        mix_mean = (w * means).sum(axis=1)   # (B, 5)
        all_mix_means.append(mix_mean)

        mix_var = np.zeros_like(mix_mean)
        for k in range(means.shape[1]):
            d = means[:, k, :] - mix_mean
            mix_var += weights[:, k:k+1] * (np.diagonal(covs[:, k], axis1=-2, axis2=-1) + d**2)
        all_mix_stds.append(np.sqrt(mix_var))

        if ffn_acts:
            all_hook_acts.append(ffn_acts[-1].numpy())

    hook.remove()

    return (
        np.concatenate(all_means),
        np.concatenate(all_covs),
        np.concatenate(all_weights),
        np.concatenate(all_mix_means),
        np.concatenate(all_mix_stds),
        np.concatenate(all_hook_acts) if all_hook_acts else None,
    )


# ── Diagnostic 1: Dead neurons / MDN collapse ─────────────────────────────

def check_dead_neurons(ffn_acts, weights):
    print("=" * 65)
    print("1. DEAD NEURONS / MDN COMPONENT COLLAPSE")
    print("=" * 65)

    if ffn_acts is not None:
        # ffn_acts: (N, n_freq, d_ff) — activations after GELU in last FFN layer
        # GELU doesn't hard-zero but values < 0.01 are effectively suppressed
        flat = ffn_acts.reshape(-1, ffn_acts.shape[-1])  # (N*n_freq, d_ff)
        frac_near_zero = (np.abs(flat) < 0.01).mean(axis=0)  # per neuron
        dead_threshold = 0.95  # "dead" if near-zero >95% of time
        n_dead = (frac_near_zero > dead_threshold).sum()
        d_ff = flat.shape[-1]

        print(f"\nFFN activations (last encoder layer, post-GELU):")
        print(f"  Neurons: {d_ff}")
        print(f"  Dead (>95% near-zero): {n_dead} / {d_ff}  ({100*n_dead/d_ff:.1f}%)")
        print(f"  Mean near-zero fraction per neuron: {frac_near_zero.mean():.3f}")
        print(f"  p50 near-zero fraction: {np.percentile(frac_near_zero, 50):.3f}")
        print(f"  p90 near-zero fraction: {np.percentile(frac_near_zero, 90):.3f}")
        print(f"  p99 near-zero fraction: {np.percentile(frac_near_zero, 99):.3f}")

    # MDN component weights
    print(f"\nMDN component weights (K={weights.shape[1]} components):")
    mean_w = weights.mean(axis=0)
    std_w  = weights.std(axis=0)
    for k in range(weights.shape[1]):
        near_zero = (weights[:, k] < 0.05).mean()
        dominant  = (weights[:, k] > 0.9).mean()
        print(f"  Component {k}: mean={mean_w[k]:.3f} ± {std_w[k]:.3f}  "
              f"[<5%: {100*near_zero:.1f}%  >90%: {100*dominant:.1f}%]")

    # Effective number of components (entropy)
    eps = 1e-10
    entropy = -(weights * np.log(weights + eps)).sum(axis=1)
    max_entropy = np.log(weights.shape[1])
    eff_k = np.exp(entropy)  # effective K
    print(f"\n  Mean entropy: {entropy.mean():.3f} / max {max_entropy:.3f}")
    print(f"  Mean effective K: {eff_k.mean():.2f} / {weights.shape[1]}")
    print(f"  Samples with effective K < 1.1 (collapsed): {(eff_k < 1.1).mean()*100:.1f}%")


# ── Diagnostic 2: Calibration ─────────────────────────────────────────────

def check_calibration(mix_means, mix_stds, targets, tau_ratios):
    param_names = ['tau_big', 'tau_small', 'TER', 'TEC', 'Rsh']
    errors = np.abs(mix_means - targets)  # (N, 5)

    print("\n" + "=" * 65)
    print("2. CALIBRATION  (predicted σ vs actual error)")
    print("=" * 65)

    print(f"\n{'Param':<12}  {'MAE':>6}  {'σ_pred':>7}  {'|1σ%':>6}  {'|2σ%':>6}  {'|3σ%':>6}  {'ECE':>6}")
    print("  " + "-" * 58)

    ece_total = 0.0
    for i, name in enumerate(param_names):
        err   = errors[:, i]
        sigma = mix_stds[:, i]
        mae   = err.mean()
        mean_sigma = sigma.mean()

        frac_1s = (err < 1.0 * sigma).mean()
        frac_2s = (err < 2.0 * sigma).mean()
        frac_3s = (err < 3.0 * sigma).mean()

        # ECE: mean |empirical_frac - expected_frac| at 1σ, 2σ, 3σ
        ece = (abs(frac_1s - 0.683) + abs(frac_2s - 0.954) + abs(frac_3s - 0.997)) / 3.0
        ece_total += ece

        print(f"  {name:<12}  {mae:>6.3f}  {mean_sigma:>7.3f}  "
              f"{100*frac_1s:>5.1f}%  {100*frac_2s:>5.1f}%  {100*frac_3s:>5.1f}%  {ece:>6.3f}")

    print(f"\n  Well-calibrated: 68.3% within 1σ, 95.4% within 2σ, 99.7% within 3σ")
    print(f"  Mean ECE (lower=better): {ece_total/len(param_names):.4f}")

    # Overconfident vs underconfident
    print(f"\nConfidence diagnosis:")
    for i, name in enumerate(param_names):
        ratio = mix_stds[:, i].mean() / errors[:, i].mean()
        if ratio < 0.5:
            diag = "OVERCONFIDENT (σ << actual error)"
        elif ratio > 2.0:
            diag = "UNDERCONFIDENT (σ >> actual error)"
        else:
            diag = "reasonable"
        print(f"  {name:<12}  σ/MAE = {ratio:.2f}  → {diag}")


# ── Diagnostic 3: Covariate shift ─────────────────────────────────────────

def check_covariate_shift(train_targets, test_targets, mix_means, tau_ratios):
    param_names = ['tau_big', 'tau_small', 'TER', 'TEC', 'Rsh']

    print("\n" + "=" * 65)
    print("3. COVARIATE SHIFT  (train vs test distribution)")
    print("=" * 65)

    print(f"\n{'Param':<12}  {'Train mean':>11}  {'Test mean':>10}  {'Train std':>10}  {'Test std':>9}  {'Shift':>6}")
    print("  " + "-" * 64)
    for i, name in enumerate(param_names):
        tr_m, tr_s = train_targets[:, i].mean(), train_targets[:, i].std()
        te_m, te_s = test_targets[:, i].mean(),  test_targets[:, i].std()
        shift = abs(tr_m - te_m) / (tr_s + 1e-9)
        flag = " ** HIGH" if shift > 0.2 else ""
        print(f"  {name:<12}  {tr_m:>11.3f}  {te_m:>10.3f}  {tr_s:>10.3f}  {te_s:>9.3f}  {shift:>6.3f}{flag}")

    # Per-stratum error to detect distributional gaps
    strata = [
        ("1.5-5x",    1.5,    5.0),
        ("5-20x",     5.0,   20.0),
        ("20-200x",  20.0,  200.0),
        ("200-10Kx", 200.0, 10001.0),
    ]
    errors = np.abs(mix_means - test_targets)

    print(f"\nError by tau_ratio stratum (tests whether model generalizes uniformly):")
    print(f"  {'Stratum':<12}  {'n':>5}  {'tau_big':>8}  {'tau_sml':>8}  {'TER':>8}  {'TEC':>8}  {'Rsh':>8}")
    print("  " + "-" * 62)
    for label, lo, hi in strata:
        mask = (tau_ratios >= lo) & (tau_ratios < hi)
        if mask.sum() == 0:
            continue
        e = errors[mask]
        print(f"  {label:<12}  {mask.sum():>5}  "
              f"{e[:,0].mean():>8.3f}  {e[:,1].mean():>8.3f}  "
              f"{e[:,2].mean():>8.3f}  {e[:,3].mean():>8.3f}  {e[:,4].mean():>8.3f}")

    # Check noise model shift: real EIS has frequency-dependent noise,
    # training used additive Gaussian at SNR=40dB
    print(f"\nNoise model notes:")
    print(f"  Training: additive Gaussian, SNR=40dB, amplitude = mean(|Z_real|) * 10^(-40/20)")
    print(f"  Real EIS: typically frequency-dependent, correlated, non-Gaussian")
    print(f"  This is the primary expected source of covariate shift on real data.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading model (fisher_v10 best checkpoint)...")
    model, ckpt = load_model("models/fisher_v10")
    epoch = ckpt.get("epoch", "?")
    val_mae = ckpt.get("val_mae", "?")
    print(f"  Epoch {epoch}  val_mae={val_mae:.4f}" if isinstance(val_mae, float) else f"  Epoch {epoch}")

    print("Loading test data...")
    Z_real, Z_imag, targets, tau_ratios, rows = load_test_data(max_samples=2000)
    print(f"  {len(Z_real)} test samples")

    print("Loading train data (for distribution comparison)...")
    train_targets = load_train_data(max_samples=5000)

    print("Running inference...")
    means, covs, weights, mix_means, mix_stds, ffn_acts = run_inference(
        model, Z_real, Z_imag
    )
    print(f"  Done. Shapes: means={means.shape}, weights={weights.shape}")

    check_dead_neurons(ffn_acts, weights)
    check_calibration(mix_means, mix_stds, targets, tau_ratios)
    check_covariate_shift(train_targets, targets, mix_means, tau_ratios)

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)


if __name__ == "__main__":
    main()
