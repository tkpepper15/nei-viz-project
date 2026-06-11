#!/usr/bin/env python3
"""
Test suite for Ra/Rb and Ca/Cb parameter assignment accuracy.

Because Ra/Rb and Ca/Cb form degenerate pairs under the apical/basolateral
swap symmetry, all metrics use symmetric (best-assignment) evaluation.
Stratifies by difficulty regime:

  Regime              What it tests
  ----------------    -------------------------------------------------------
  all                 Full test set — overall view
  separated_taus      |log(tau_a/tau_b)| > 1.0 decade: arcs are well resolved
  degenerate_taus     |log(tau_a/tau_b)| < 0.3 decades: arcs merge, hard case
  asymmetric_R        |log(Ra/Rb)| > 1.0 decade: resistances differ strongly
  asymmetric_C        |log(Ca/Cb)| > 1.0 decade: capacitances differ strongly
  balanced_params     |log(Ra/Rb)| < 0.3 AND |log(Ca/Cb)| < 0.3: near-symmetric

Usage:
    cd pipeline
    python scripts/analysis/test_parameter_assignment.py
    python scripts/analysis/test_parameter_assignment.py --data data/mixed_distribution_v2 --batch-size 256
    python scripts/analysis/test_parameter_assignment.py --no-version-compare
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.fisher_transformer import FisherAwareTransformer, TransformerConfig
from src.evaluation.symmetric_metrics import compute_symmetric_mae


# ---------------------------------------------------------------------------
# Model loading — handles all param_space variants
# ---------------------------------------------------------------------------

def _infer_param_space_for_dir(ckpt_dir: Path) -> str:
    """
    Read param_space from best_model.pt in the same directory.

    Epoch checkpoints in fisher_v4 were saved before param_space was
    added to the checkpoint dict. We fall back to the best_model.pt in
    the same directory which does carry the key.
    """
    best = ckpt_dir / "best_model.pt"
    if best.exists():
        ref = torch.load(str(best), map_location="cpu", weights_only=False)
        return ref.get("param_space", "original")
    return "original"


def load_checkpoint(path: Path, param_space_override: str | None = None):
    """
    Load any fisher_v* checkpoint. Returns (model, param_space, epoch).

    param_space_override: if provided, use this instead of reading from the
    checkpoint (needed for epoch checkpoints that pre-date the key).
    """
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    config = TransformerConfig(
        n_freq=cfg["n_freq"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        dropout=0.0,  # eval mode — no dropout
        n_proposals=cfg["n_proposals"],
        n_params=cfg["n_params"],
        use_low_rank_cov=cfg.get("use_low_rank_cov", True),
        cov_rank=cfg.get("cov_rank", 2),
        use_grad_features=cfg.get("use_grad_features", True),
    )
    model = FisherAwareTransformer(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if param_space_override is not None:
        param_space = param_space_override
    elif "param_space" in ckpt:
        param_space = ckpt["param_space"]
    else:
        # Not stored — infer from best_model.pt in the same directory
        param_space = _infer_param_space_for_dir(path.parent)

    epoch = ckpt.get("epoch", "?")
    return model, param_space, epoch


def _tec_ratio_to_ra_rb_ca_cb_rsh(means: np.ndarray) -> np.ndarray:
    """
    Convert tec_ratio model output to [log10 Ra, Rb, Ca, Cb, Rsh].

    tec_ratio space: [log10 Ra, log10 Rb, log10 TEC, log10(Ca/Cb), log10 Rsh]
    Ca = TEC * (1 + Ca/Cb)
    Cb = TEC * (1 + Cb/Ca) = TEC * (1 + 1/(Ca/Cb))
    """
    log_Ra, log_Rb, log_TEC, log_ratio, log_Rsh = means.T
    TEC = 10.0 ** log_TEC
    R   = 10.0 ** log_ratio          # Ca/Cb
    Ca  = TEC * (1.0 + R)
    Cb  = TEC * (1.0 + 1.0 / (R + 1e-12))
    log_Ca = np.log10(np.clip(Ca, 1e-30, None))
    log_Cb = np.log10(np.clip(Cb, 1e-30, None))
    return np.stack([log_Ra, log_Rb, log_Ca, log_Cb, log_Rsh], axis=-1)


def _identifiable_to_ra_rb_ca_cb_rsh(means_t: torch.Tensor) -> np.ndarray:
    """
    Convert identifiable-space output to [log10 Ra, Rb, Ca, Cb, Rsh].

    identifiable space: [log10 tau_big, tau_small, TER, TEC, Rsh]
    Uses the analytic inversion from 02_train_transformer.py.
    """
    log_tb, log_ts, log_TER, log_TEC, log_Rsh = means_t.unbind(-1)
    tau_big   = 10.0 ** log_tb
    tau_small = 10.0 ** log_ts
    TER       = 10.0 ** log_TER
    TEC       = 10.0 ** log_TEC
    Rsh       = 10.0 ** log_Rsh

    S = (TER * Rsh / (Rsh - TER + 1e-6)).clamp(min=1e-3)

    denom    = tau_big - tau_small
    Ra_exact = tau_small * (tau_big / (TEC + 1e-30) - S) / (denom + 1e-6)
    Ra_sym   = S * 0.5
    alpha    = torch.sigmoid(denom / (tau_big * 0.1 + 1e-12) * 5.0)
    Ra       = (alpha * Ra_exact + (1.0 - alpha) * Ra_sym).clamp(min=1e-3)
    Ra       = Ra.clamp(max=S - 1e-3)
    Rb       = (S - Ra).clamp(min=1e-3)
    Ca       = (tau_small / (Ra + 1e-12)).clamp(min=1e-30)
    Cb       = (tau_big   / (Rb + 1e-12)).clamp(min=1e-30)

    return np.stack([
        torch.log10(Ra + 1e-30).numpy(),
        torch.log10(Rb + 1e-30).numpy(),
        torch.log10(Ca + 1e-30).numpy(),
        torch.log10(Cb + 1e-30).numpy(),
        log_Rsh.numpy(),
    ], axis=-1)


@torch.no_grad()
def infer_to_original(
    model: FisherAwareTransformer,
    param_space: str,
    Z_real: torch.Tensor,
    Z_imag: torch.Tensor,
    log_omega: torch.Tensor,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Run batched inference, returning predictions in original
    [log10 Ra, log10 Rb, log10 Ca, log10 Cb, log10 Rsh] space for any model.
    """
    n = Z_real.shape[0]
    all_preds = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bZ_r = Z_real[start:end]
        bZ_i = Z_imag[start:end]
        b_lo = log_omega.unsqueeze(0).expand(end - start, -1)

        proposals = model(bZ_r, bZ_i, b_lo)
        means    = proposals["means"].cpu()     # (batch, K, 5)
        weights  = proposals["weights"].cpu()   # (batch, K)

        # Weighted mean across proposals
        mix_mean = (means * weights.unsqueeze(-1)).sum(dim=1)  # (batch, 5)

        if param_space == "tec_ratio":
            preds_orig = _tec_ratio_to_ra_rb_ca_cb_rsh(mix_mean.numpy())
        elif param_space == "identifiable":
            preds_orig = _identifiable_to_ra_rb_ca_cb_rsh(mix_mean)
        else:
            # original: direct [Ra, Rb, Ca, Cb, Rsh] log10 output
            preds_orig = mix_mean.numpy()

        all_preds.append(preds_orig)

    return np.concatenate(all_preds, axis=0)  # (N, 5)


# ---------------------------------------------------------------------------
# Test data loading and regime masks
# ---------------------------------------------------------------------------

def load_test_data(data_dir: str, n_freq: int = 100):
    """Load test.csv and return tensors + raw DataFrame for regime slicing."""
    data_path = Path(data_dir)
    df = pd.read_csv(data_path / "test.csv")

    Z_real_cols = [f"Z_real_{i}" for i in range(n_freq)]
    Z_imag_cols = [f"Z_imag_{i}" for i in range(n_freq)]

    Z_real = torch.tensor(df[Z_real_cols].values, dtype=torch.float32)
    Z_imag = torch.tensor(df[Z_imag_cols].values, dtype=torch.float32)

    with open(data_path / "metadata.json") as f:
        meta = json.load(f)
    freq_min = meta["frequencies"]["min"]
    freq_max = meta["frequencies"]["max"]
    freqs    = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freq)
    log_omega = torch.tensor(np.log10(2 * np.pi * freqs), dtype=torch.float32)

    # True parameters in log10 space — (N, 5): Ra Rb Ca Cb Rsh
    params_log10 = np.log10(df[["Ra", "Rb", "Ca", "Cb", "Rsh"]].values.astype(float))

    return Z_real, Z_imag, log_omega, params_log10, df


def build_regime_masks(df: pd.DataFrame) -> dict:
    """
    Return boolean index arrays for different difficulty regimes.

    All computed in log10 space so ratios are in decades.
    """
    log_Ra  = np.log10(df["Ra"].values)
    log_Rb  = np.log10(df["Rb"].values)
    log_Ca  = np.log10(df["Ca"].values)
    log_Cb  = np.log10(df["Cb"].values)

    tau_a = df["Ra"].values * df["Ca"].values
    tau_b = df["Rb"].values * df["Cb"].values
    log_tau_sep = np.abs(np.log10(tau_a + 1e-30) - np.log10(tau_b + 1e-30))
    log_R_sep   = np.abs(log_Ra - log_Rb)
    log_C_sep   = np.abs(log_Ca - log_Cb)

    n = len(df)
    return {
        "all":             np.ones(n, dtype=bool),
        "separated_taus":  log_tau_sep > 1.0,
        "degenerate_taus": log_tau_sep < 0.3,
        "asymmetric_R":    log_R_sep > 1.0,
        "asymmetric_C":    log_C_sep > 1.0,
        "balanced_params": (log_R_sep < 0.3) & (log_C_sep < 0.3),
    }


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(preds_log10: np.ndarray, targets_log10: np.ndarray, mask: np.ndarray) -> dict:
    """
    Compute all Ra/Rb/Ca/Cb/Rsh metrics over the masked subset.

    Returns dict with:
        sym_R_mae   — symmetric Ra/Rb MAE (best assignment, decades)
        sym_C_mae   — symmetric Ca/Cb MAE (best assignment, decades)
        Rsh_mae     — direct Rsh MAE (decades)
        std_Ra_mae  — fixed Ra→Ra MAE (for reference, shows degeneracy cost)
        std_Rb_mae  — fixed Rb→Rb MAE
        std_Ca_mae  — fixed Ca→Ca MAE
        std_Cb_mae  — fixed Cb→Cb MAE
        n_samples   — number of samples in regime
    """
    p = preds_log10[mask]
    t = targets_log10[mask]
    n = int(mask.sum())

    if n == 0:
        return {k: float("nan") for k in
                ["sym_R_mae", "sym_C_mae", "Rsh_mae",
                 "std_Ra_mae", "std_Rb_mae", "std_Ca_mae", "std_Cb_mae",
                 "n_samples"]}

    pred_Ra_lin  = 10.0 ** p[:, 0]
    pred_Ca_lin  = 10.0 ** p[:, 2]
    pred_Rsh_lin = 10.0 ** p[:, 4]
    true_Ra_lin  = 10.0 ** t[:, 0]
    true_Rb_lin  = 10.0 ** t[:, 1]
    true_Ca_lin  = 10.0 ** t[:, 2]
    true_Cb_lin  = 10.0 ** t[:, 3]
    true_Rsh_lin = 10.0 ** t[:, 4]

    sym = compute_symmetric_mae(
        pred_R1=pred_Ra_lin,
        pred_C1=pred_Ca_lin,
        pred_R2=pred_Rsh_lin,
        true_Ra=true_Ra_lin,
        true_Rb=true_Rb_lin,
        true_Ca=true_Ca_lin,
        true_Cb=true_Cb_lin,
        true_Rsh=true_Rsh_lin,
        log_space=True,
    )

    return {
        "sym_R_mae":  sym["R1_mae"],
        "sym_C_mae":  sym["C1_mae"],
        "Rsh_mae":    float(np.mean(np.abs(p[:, 4] - t[:, 4]))),
        "std_Ra_mae": float(np.mean(np.abs(p[:, 0] - t[:, 0]))),
        "std_Rb_mae": float(np.mean(np.abs(p[:, 1] - t[:, 1]))),
        "std_Ca_mae": float(np.mean(np.abs(p[:, 2] - t[:, 2]))),
        "std_Cb_mae": float(np.mean(np.abs(p[:, 3] - t[:, 3]))),
        "n_samples":  n,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

REGIME_ORDER = [
    "all", "separated_taus", "degenerate_taus",
    "asymmetric_R", "asymmetric_C", "balanced_params",
]

REGIME_LABELS = {
    "all":             "All samples",
    "separated_taus":  "|log(τ_a/τ_b)| > 1 dec",
    "degenerate_taus": "|log(τ_a/τ_b)| < 0.3 dec",
    "asymmetric_R":    "|log(Ra/Rb)| > 1 dec",
    "asymmetric_C":    "|log(Ca/Cb)| > 1 dec",
    "balanced_params": "Ra≈Rb and Ca≈Cb",
}

def print_table(title: str, rows: list[dict], col_width: int = 10):
    """Print a formatted metrics table."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

    headers = ["Model/Epoch", "Regime", "N", "sym_R_mae", "sym_C_mae", "Rsh_mae",
               "std_Ra_mae", "std_Rb_mae"]
    fmt = f"{{:<28}} {{:<28}} {{:>7}} {{:>10}} {{:>10}} {{:>10}} {{:>10}} {{:>10}}"
    print(fmt.format(*headers))
    print("-" * 110)

    for row in rows:
        print(fmt.format(
            row["label"][:28],
            REGIME_LABELS.get(row["regime"], row["regime"])[:28],
            str(row["n_samples"]),
            f"{row['sym_R_mae']:.4f}",
            f"{row['sym_C_mae']:.4f}",
            f"{row['Rsh_mae']:.4f}",
            f"{row['std_Ra_mae']:.4f}",
            f"{row['std_Rb_mae']:.4f}",
        ))

    print(f"{'='*80}")
    print("  sym_R_mae = symmetric Ra/Rb MAE (best-assignment, decades)")
    print("  sym_C_mae = symmetric Ca/Cb MAE (best-assignment, decades)")
    print("  std_Ra_mae/Rb_mae = fixed-label MAE (higher = degeneracy is hurting)")


def save_csv(rows: list[dict], path: Path):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"\nSaved: {path}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(
    model_entries: list[dict],  # each: {label, path, checkpoint_key}
    Z_real, Z_imag, log_omega,
    targets_log10: np.ndarray,
    masks: dict,
    batch_size: int,
) -> list[dict]:
    """Run inference for each model entry and compute metrics across all regimes."""
    all_rows = []

    for entry in model_entries:
        label    = entry["label"]
        ckpt_path = entry["path"]

        if not ckpt_path.exists():
            print(f"  [skip] {label}: {ckpt_path} not found")
            continue

        print(f"  Loading {label} from {ckpt_path.name} ...")
        override = entry.get("param_space_override", None)
        model, param_space, epoch = load_checkpoint(ckpt_path, param_space_override=override)
        print(f"    param_space={param_space}  epoch={epoch}")

        preds = infer_to_original(
            model, param_space, Z_real, Z_imag, log_omega, batch_size=batch_size
        )

        for regime in REGIME_ORDER:
            mask = masks[regime]
            metrics = compute_metrics(preds, targets_log10, mask)
            row = {"label": label, "regime": regime}
            row.update(metrics)
            all_rows.append(row)

    return all_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/mixed_distribution_v1",
                        help="Path to data directory with test.csv and metadata.json")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-version-compare", action="store_true",
                        help="Skip the v2/v3/v4 version comparison")
    parser.add_argument("--out-dir", default="results/ra_rb_ca_cb_suite",
                        help="Directory for CSV outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load test data
    # -----------------------------------------------------------------------
    print(f"\nLoading test data from {args.data} ...")
    Z_real, Z_imag, log_omega, targets_log10, df = load_test_data(args.data)
    masks = build_regime_masks(df)
    print(f"  Test set: {len(df)} samples")
    for name, m in masks.items():
        print(f"    {REGIME_LABELS[name]:35s}: {m.sum():6d} samples ({100*m.mean():.1f}%)")

    # -----------------------------------------------------------------------
    # Analysis 1: Epoch progression within fisher_v4
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("ANALYSIS 1: Epoch progression (fisher_v4)")
    print("="*60)

    epoch_entries = []
    models_dir = Path("models/fisher_v4")
    epoch_checkpoints = [20, 40, 60, 80, 100]
    for ep in epoch_checkpoints:
        p = models_dir / f"checkpoint_epoch_{ep}.pt"
        epoch_entries.append({"label": f"v4 epoch {ep:3d}", "path": p})
    epoch_entries.append({"label": "v4 best", "path": models_dir / "best_model.pt"})

    epoch_rows = run_analysis(
        epoch_entries, Z_real, Z_imag, log_omega, targets_log10, masks, args.batch_size
    )

    # Print focused table: just "all" and "separated_taus" to show progression clearly
    epoch_all = [r for r in epoch_rows if r["regime"] == "all"]
    print_table("Epoch Progression — All samples (sym Ra/Rb and Ca/Cb MAE)", epoch_all)

    epoch_sep = [r for r in epoch_rows if r["regime"] == "separated_taus"]
    print_table("Epoch Progression — Separated τ regime (arcs resolved)", epoch_sep)

    epoch_deg = [r for r in epoch_rows if r["regime"] == "degenerate_taus"]
    print_table("Epoch Progression — Degenerate τ regime (arcs merged, hard)", epoch_deg)

    save_csv(epoch_rows, out_dir / "epoch_progression.csv")

    # -----------------------------------------------------------------------
    # Analysis 2: Version comparison (v2, v3, v4 best)
    # -----------------------------------------------------------------------
    if not args.no_version_compare:
        print("\n" + "="*60)
        print("ANALYSIS 2: Version comparison (v2, v3, v4 best models)")
        print("="*60)

        version_entries = [
            {"label": "v2 (original space)", "path": Path("models/fisher_v2/best_model.pt")},
            {"label": "v3 (tec_ratio)",       "path": Path("models/fisher_v3/best_model.pt")},
            {"label": "v4 (tec_ratio+OOD)",   "path": Path("models/fisher_v4/best_model.pt")},
        ]

        version_rows = run_analysis(
            version_entries, Z_real, Z_imag, log_omega, targets_log10, masks, args.batch_size
        )

        # Full table across all regimes
        print_table("Version Comparison — All regimes", version_rows)
        save_csv(version_rows, out_dir / "version_comparison.csv")

    # -----------------------------------------------------------------------
    # Analysis 3: Regime breakdown for best model
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("ANALYSIS 3: Best model (v4) regime breakdown")
    print("="*60)

    best_path = Path("models/fisher_v4/best_model.pt")
    if best_path.exists():
        model, param_space, epoch = load_checkpoint(best_path)
        preds_best = infer_to_original(
            model, param_space, Z_real, Z_imag, log_omega, batch_size=args.batch_size
        )

        regime_rows = []
        for regime in REGIME_ORDER:
            mask = masks[regime]
            metrics = compute_metrics(preds_best, targets_log10, mask)
            row = {"label": f"v4 (ep {epoch})", "regime": regime}
            row.update(metrics)
            regime_rows.append(row)

        print_table("v4 Best Model — Symmetric Ra/Rb and Ca/Cb by regime", regime_rows)
        save_csv(regime_rows, out_dir / "regime_breakdown.csv")

        # Print interpretation summary
        print("\nInterpretation:")
        m_all  = next(r for r in regime_rows if r["regime"] == "all")
        m_sep  = next(r for r in regime_rows if r["regime"] == "separated_taus")
        m_deg  = next(r for r in regime_rows if r["regime"] == "degenerate_taus")
        print(f"  Overall sym Ra/Rb MAE:  {m_all['sym_R_mae']:.4f} dec  |  "
              f"sym Ca/Cb MAE: {m_all['sym_C_mae']:.4f} dec")
        print(f"  Separated τ regime:     R={m_sep['sym_R_mae']:.4f} dec  |  "
              f"C={m_sep['sym_C_mae']:.4f} dec  (easier, arcs resolved)")
        print(f"  Degenerate τ regime:    R={m_deg['sym_R_mae']:.4f} dec  |  "
              f"C={m_deg['sym_C_mae']:.4f} dec  (harder, arcs merged)")
        sym_gain_R = m_all["std_Ra_mae"] - m_all["sym_R_mae"]
        sym_gain_C = m_all["std_Ca_mae"] - m_all["sym_C_mae"]
        print(f"\n  Symmetric vs fixed-label improvement:")
        print(f"    Ra/Rb: fixed={m_all['std_Ra_mae']:.4f}  sym={m_all['sym_R_mae']:.4f}  "
              f"gain={sym_gain_R:+.4f} dec")
        print(f"    Ca/Cb: fixed={m_all['std_Ca_mae']:.4f}  sym={m_all['sym_C_mae']:.4f}  "
              f"gain={sym_gain_C:+.4f} dec")
        if sym_gain_R > 0.05:
            print("    (Large gain = the model often swaps the Ra/Rb label, which is correct "
                  "behaviour — the spectrum doesn't identify apical vs basolateral)")

    print(f"\nAll results saved to {out_dir}/\n")


if __name__ == "__main__":
    main()
