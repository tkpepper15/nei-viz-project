#!/usr/bin/env python3
"""
Post-hoc temperature calibration for the MDN uncertainty estimates.

Method
------
For a well-calibrated Gaussian predictor, 68% of |err_i| should fall within
1 * sigma_i.  Currently sigma/MAE ≈ 3-7x, so coverage at 1-sigma is 92-100%.

Temperature scaling finds a scalar T_i per parameter such that:
    sigma_calibrated_i = T_i * sigma_pred_i
    => Pr(|err_i| < sigma_calibrated_i) ≈ 0.683

T_i is the 68th percentile of |err_i| / sigma_pred_i on the validation set.

The calibration is saved to a JSON file and applied in backend_api.py by
multiplying the MDN covariance diagonal by T_i^2 before passing to the EKF.

Global temperature
------------------
A single scalar T_global = median(T_i) is also saved.  For the EKF, per-
parameter scaling is preferred because TER/Rsh have much larger inflation than
tau_big/tau_small.

Usage
-----
    python calibrate_temperature.py --model models/fisher_v10 --data data/mixed_distribution_v2
    python calibrate_temperature.py --model models/fisher_v10 --data data/mixed_distribution_v2 --save
"""

import sys
import csv
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.models.fisher_transformer import FisherAwareTransformer, TransformerConfig


PARAM_NAMES = ['tau_big', 'tau_small', 'TER', 'TEC', 'Rsh']


def load_model(model_dir):
    ckpt = torch.load(
        Path(model_dir) / "best_model.pt",
        map_location="cpu", weights_only=False
    )
    config = TransformerConfig(**ckpt.get("config", {}))
    model = FisherAwareTransformer(config)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()
    epoch   = ckpt.get("epoch", "?")
    val_mae = ckpt.get("val_mae", float("nan"))
    print(f"Loaded {model_dir}  epoch={epoch}  val_mae={val_mae:.4f}")
    return model


def load_val_data(data_dir, split="val", max_samples=5000):
    path = Path(data_dir) / f"{split}.csv"
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
            if len(rows) >= max_samples:
                break

    n_freq = 100
    Z_real = np.array([[float(r[f"Z_real_{i}"]) for i in range(n_freq)] for r in rows], dtype=np.float32)
    Z_imag = np.array([[float(r[f"Z_imag_{i}"]) for i in range(n_freq)] for r in rows], dtype=np.float32)
    targets = np.array([
        [np.log10(float(r["tau_big"])),
         np.log10(float(r["tau_small"])),
         np.log10(float(r["TER"])),
         np.log10(float(r["TEC"])),
         np.log10(float(r["Rsh"]))]
        for r in rows
    ], dtype=np.float32)
    return Z_real, Z_imag, targets


def run_inference(model, Z_real, Z_imag, batch_size=256):
    freqs     = np.logspace(-1, 6, 100)
    log_omega = np.log10(2 * np.pi * freqs).astype(np.float32)

    all_mix_means = []
    all_mix_stds  = []

    n = len(Z_real)
    for i in range(0, n, batch_size):
        Zr = torch.tensor(Z_real[i:i+batch_size])
        Zi = torch.tensor(Z_imag[i:i+batch_size])
        lo = torch.tensor(log_omega).unsqueeze(0).expand(len(Zr), -1)

        with torch.no_grad():
            props = model(Zr, Zi, lo)

        means   = props['means'].cpu().numpy()    # (B, K, 5)
        covs    = props['covs'].cpu().numpy()     # (B, K, 5, 5)
        weights = props['weights'].cpu().numpy()  # (B, K)

        # Mixture mean
        w = weights[:, :, None]
        mix_mean = (w * means).sum(axis=1)        # (B, 5)
        all_mix_means.append(mix_mean)

        # Mixture std (within + between component variance)
        mix_var = np.zeros_like(mix_mean)
        for k in range(means.shape[1]):
            d = means[:, k, :] - mix_mean
            mix_var += weights[:, k:k+1] * (
                np.diagonal(covs[:, k], axis1=-2, axis2=-1) + d**2
            )
        all_mix_stds.append(np.sqrt(np.clip(mix_var, 1e-10, None)))

    return np.concatenate(all_mix_means), np.concatenate(all_mix_stds)


def compute_temperatures(mix_means, mix_stds, targets, coverage_target=0.683):
    """
    Find T_i = 68th percentile of |err_i| / sigma_i for each parameter.
    After scaling sigma by T_i, exactly coverage_target fraction of samples
    fall within 1 standard deviation.
    """
    errors = np.abs(mix_means - targets)          # (N, 5)
    norm_residuals = errors / (mix_stds + 1e-10)  # (N, 5)  i.e. |err| / sigma

    # T_i = quantile of norm_residuals at coverage_target
    temperatures = np.quantile(norm_residuals, coverage_target, axis=0)  # (5,)
    return temperatures, norm_residuals


def coverage_table(errors, mix_stds, temperatures):
    """
    Print coverage before and after calibration at 1σ, 2σ, 3σ.
    """
    norm_r = errors / (mix_stds + 1e-10)

    print(f"\n{'Param':<12}  {'T_scale':>8}  {'σ/MAE':>7}  "
          f"{'Pre-1σ%':>8}  {'Post-1σ%':>9}  {'Pre-2σ%':>8}  {'Post-2σ%':>9}")
    print("  " + "-" * 75)

    for i, name in enumerate(PARAM_NAMES):
        T = temperatures[i]
        mae = errors[:, i].mean()
        pre_1s  = (norm_r[:, i] < 1.0).mean()
        post_1s = (norm_r[:, i] < T).mean()       # by definition ≈ 0.683
        pre_2s  = (norm_r[:, i] < 2.0).mean()
        post_2s = (norm_r[:, i] < 2 * T).mean()
        sigma_over_mae = mix_stds[:, i].mean() / (mae + 1e-10)

        print(f"  {name:<12}  {T:>8.3f}  {sigma_over_mae:>7.2f}x  "
              f"{100*pre_1s:>7.1f}%  {100*post_1s:>8.1f}%  "
              f"{100*pre_2s:>7.1f}%  {100*post_2s:>8.1f}%")

    T_global = float(np.median(temperatures))
    print(f"\n  Global temperature (median T_i): {T_global:.4f}")
    print(f"  Effective sigma multiplier applied to covariance: T_i^2 each diagonal")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="models/fisher_v10")
    parser.add_argument("--data",   default="data/mixed_distribution_v2")
    parser.add_argument("--split",  default="val", choices=["val", "test"])
    parser.add_argument("--save",   action="store_true",
                        help="Write temperature_calibration.json to model directory")
    args = parser.parse_args()

    model = load_model(args.model)
    Z_real, Z_imag, targets = load_val_data(args.data, args.split)
    print(f"Calibrating on {len(Z_real)} {args.split} samples from {args.data}")

    mix_means, mix_stds = run_inference(model, Z_real, Z_imag)
    temperatures, norm_residuals = compute_temperatures(mix_means, mix_stds, targets)

    errors = np.abs(mix_means - targets)

    print("\n" + "=" * 70)
    print("TEMPERATURE CALIBRATION RESULTS")
    print("=" * 70)
    coverage_table(errors, mix_stds, temperatures)

    # EKF impact estimate
    print("\nIMPACT ON EKF OBSERVATIONS:")
    print("  Before calibration: EKF weights prior vs observation ~ Sigma_obs/(Sigma_obs + Q)")
    print("  After calibration:  Sigma_obs shrinks by T_i^2, observation weight increases")
    print(f"  tau_big  Sigma reduction: {temperatures[0]**2:.3f}x  "
          f"(observation weight increases ~{1/temperatures[0]**2:.1f}x)")
    print(f"  tau_small Sigma reduction: {temperatures[1]**2:.3f}x")
    print(f"  TER       Sigma reduction: {temperatures[2]**2:.3f}x")
    print(f"  TEC       Sigma reduction: {temperatures[3]**2:.3f}x")
    print(f"  Rsh       Sigma reduction: {temperatures[4]**2:.3f}x")

    calibration = {
        "param_names": PARAM_NAMES,
        "temperatures": temperatures.tolist(),         # multiply sigma by T_i
        "variance_scales": (temperatures**2).tolist(), # multiply variance by T_i^2
        "global_temperature": float(np.median(temperatures)),
        "coverage_target": 0.683,
        "method": "empirical_quantile_val_set",
        "model_dir": str(args.model),
        "n_samples": len(Z_real),
        "split": args.split,
    }

    if args.save:
        out = Path(args.model) / "temperature_calibration.json"
        with open(out, "w") as f:
            json.dump(calibration, f, indent=2)
        print(f"\nSaved: {out}")
        print("Apply in backend_api.py by loading this file and multiplying")
        print("MDN covariance diagonals by variance_scales before EKF update.")
    else:
        print(f"\nRun with --save to write calibration to {args.model}/temperature_calibration.json")

    return calibration


if __name__ == "__main__":
    main()
