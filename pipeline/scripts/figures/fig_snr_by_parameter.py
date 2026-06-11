#!/usr/bin/env python3
"""
SNR by Parameter

Signal-to-noise ratio per parameter:
  SNR = |ATP-induced change in 3P-EIS ground truth| / posterior IQR

Compares DL framework (narrow posteriors) vs ECM fitting
(wide posteriors) across all 3 iPSC-RPE samples.

SNR > 1 means the change is larger than the method's uncertainty —
i.e., it is detectable.  The DL framework achieves this; ECM often does not.

Usage:
    cd pipeline && python fig_snr_by_parameter.py
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_DIR = Path("results/poster/cache_12e")
OUT_DIR   = Path("results/poster")

BLACK      = "#1a1a1a"
GOLD       = "#c8962e"
DL_COLOR   = "#2166ac"
GREY_FAINT = "#d8d8d8"
RED_LINE   = "#c0392b"

SAMPLES = {
    "Sample 1": {"interval_min": 3.5, "atp_span": (7,  11)},
    "Sample 2": {"interval_min": 2.4, "atp_span": (10, 15)},
    "Sample 3": {"interval_min": 1.7, "atp_span": (9,  12)},
}

# Display vector columns: [Ra, Rb, Ca, Cb, Rsh]
# We show in order: Ca, Cb, Ra, Rb, Rsh
DISPLAY_ORDER  = [2, 3, 0, 1, 4]
PARAM_LABELS   = [
    r"$C_a$",
    r"$C_b$",
    r"$R_a$",
    r"$R_b$",
    r"$R_{sh}$",
]

mpl.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.default": "regular",
})


# ── Cache helpers ─────────────────────────────────────────────────────────────
def load_cache(sname):
    p = CACHE_DIR / f"cache_{sname.replace(' ', '_')}.npz"
    d = np.load(p, allow_pickle=True)
    return {
        "n_t":        int(d["n_t"][0]),
        "truth_disp": d["truth_disp"],
        "dl_disp":    list(d["dl_disp"]),
        "ecm_disp":   list(d["ecm_disp"]),
    }


# ── SNR computation ───────────────────────────────────────────────────────────
def compute_snr(res, cfg):
    """
    For each of the 5 parameters:
      signal    = |median(truth_ATP) - median(truth_baseline)|
      dl_noise  = median IQR across all time points
      ecm_noise = median IQR across all time points

    Returns snr_dl (5,), snr_ecm (5,), signal (5,), dl_iqr (5,), ecm_iqr (5,)
    all in display units.
    """
    truth = res["truth_disp"]   # (n_t, 5)
    dl    = res["dl_disp"]      # list[n_t] of (N, 5)
    ecm   = res["ecm_disp"]     # list[n_t] of (M, 5)
    n_t   = res["n_t"]

    atp_lo, atp_hi = cfg["atp_span"]
    base_mask = np.arange(n_t) < atp_lo
    atp_mask  = (np.arange(n_t) >= atp_lo) & (np.arange(n_t) < atp_hi)

    signals  = np.zeros(5)
    dl_iqrs  = np.zeros(5)
    ecm_iqrs = np.zeros(5)

    for p in range(5):
        if base_mask.sum() > 0 and atp_mask.sum() > 0:
            signals[p] = abs(
                np.median(truth[atp_mask, p]) - np.median(truth[base_mask, p])
            )
        else:
            signals[p] = abs(truth[:, p].max() - truth[:, p].min())

        dl_iqrs[p] = np.median([
            np.nanpercentile(d[:, p], 75) - np.nanpercentile(d[:, p], 25)
            for d in dl
        ])

        valid_ecm = [e for e in ecm if len(e) > 0]
        ecm_iqrs[p] = np.median([
            np.nanpercentile(e[:, p], 75) - np.nanpercentile(e[:, p], 25)
            for e in valid_ecm
        ]) if valid_ecm else np.nan

    snr_dl  = np.where(dl_iqrs  > 0, signals / dl_iqrs,  0.0)
    snr_ecm = np.where(ecm_iqrs > 0, signals / ecm_iqrs, 0.0)

    return snr_dl, snr_ecm, signals, dl_iqrs, ecm_iqrs


# ── Figure ────────────────────────────────────────────────────────────────────
def make_figure():
    per_sample_dl  = []
    per_sample_ecm = []

    for sname, cfg in SAMPLES.items():
        res = load_cache(sname)
        snr_dl, snr_ecm, _, _, _ = compute_snr(res, cfg)
        per_sample_dl.append(snr_dl[DISPLAY_ORDER])
        per_sample_ecm.append(snr_ecm[DISPLAY_ORDER])

    # Shape: (3 samples, 5 params)
    per_sample_dl  = np.array(per_sample_dl)
    per_sample_ecm = np.array(per_sample_ecm)

    mean_dl  = per_sample_dl.mean(axis=0)
    std_dl   = per_sample_dl.std(axis=0)
    mean_ecm = per_sample_ecm.mean(axis=0)
    std_ecm  = per_sample_ecm.std(axis=0)

    n_params = len(PARAM_LABELS)
    x     = np.arange(n_params)
    width = 0.32

    fig, ax = plt.subplots(figsize=(11, 9))

    # ── Bars ──
    bars_dl = ax.bar(
        x - width / 2, mean_dl, width,
        color=DL_COLOR, alpha=0.88,
        yerr=std_dl, capsize=7,
        error_kw={"linewidth": 2.0, "color": BLACK, "capthick": 2.0},
        zorder=3, label="DL framework",
    )
    bars_ecm = ax.bar(
        x + width / 2, mean_ecm, width,
        color=GOLD, alpha=0.88,
        yerr=std_ecm, capsize=7,
        error_kw={"linewidth": 2.0, "color": BLACK, "capthick": 2.0},
        zorder=3, label="ECM",
    )

    # ── Individual sample dots ──
    jitter = 0.06
    for i, (s_dl, s_ecm) in enumerate(zip(per_sample_dl, per_sample_ecm)):
        ax.scatter(
            x - width / 2 + (i - 1) * jitter, s_dl,
            color="white", edgecolors=DL_COLOR, s=55,
            linewidths=1.8, zorder=5,
        )
        ax.scatter(
            x + width / 2 + (i - 1) * jitter, s_ecm,
            color="white", edgecolors=GOLD, s=55,
            linewidths=1.8, zorder=5,
        )

    # ── Detection threshold ──
    ax.axhline(
        1.0, color=RED_LINE, linewidth=2.2,
        linestyle="--", alpha=0.85, zorder=4,
        label="Detection threshold  (SNR = 1)",
    )

    # ── SNR value labels on bars ──
    for bar, val in zip(bars_dl, mean_dl):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std_dl[list(mean_dl).index(val)] + 0.08,
            f"{val:.1f}×",
            ha="center", va="bottom",
            fontsize=13, fontweight="bold", color=DL_COLOR,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )
    for bar, val in zip(bars_ecm, mean_ecm):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std_ecm[list(mean_ecm).index(val)] + 0.08,
            f"{val:.1f}×",
            ha="center", va="bottom",
            fontsize=13, fontweight="bold", color=GOLD,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )

    # ── Axes ──
    ax.set_xticks(x)
    ax.set_xticklabels(PARAM_LABELS, fontsize=22)
    ax.set_ylabel(
        "SNR  =  ATP change  /  posterior IQR",
        fontsize=18, labelpad=10,
    )
    ax.tick_params(labelsize=16, length=5)
    ax.set_ylim(0, max(mean_dl.max(), mean_ecm.max()) * 1.35)
    ax.grid(True, axis="y", alpha=0.18, linewidth=0.7, color="#aaaaaa")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(1.0)

    # ── Legend ──
    ax.legend(
        fontsize=16, framealpha=0.96,
        edgecolor=GREY_FAINT, handlelength=2.2,
        loc="upper right",
    )

    # ── Title ──
    ax.set_title(
        "DL Framework Extracts More Signal from Impedance Measurements\n"
        r"Higher SNR $\Rightarrow$ ATP-induced changes are detectable above noise",
        fontsize=18, fontweight="bold", pad=18, linespacing=1.6,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "snr_bar.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
    plt.close(fig)


def make_iqr_reduction_figure():
    per_sample_reduction = []

    for sname, cfg in SAMPLES.items():
        res = load_cache(sname)
        _, _, _, dl_iqrs, ecm_iqrs = compute_snr(res, cfg)
        reduction = ecm_iqrs[DISPLAY_ORDER] / dl_iqrs[DISPLAY_ORDER]
        per_sample_reduction.append(reduction)

    per_sample_reduction = np.array(per_sample_reduction)  # (3, 5)
    mean_red = per_sample_reduction.mean(axis=0)
    std_red  = per_sample_reduction.std(axis=0)

    n_params = len(PARAM_LABELS)
    x = np.arange(n_params)

    fig, ax = plt.subplots(figsize=(11, 9))

    bars = ax.bar(
        x, mean_red, 0.55,
        color=DL_COLOR, alpha=0.88,
        yerr=std_red, capsize=8,
        error_kw={"linewidth": 2.0, "color": BLACK, "capthick": 2.0},
        zorder=3,
    )

    # Individual sample dots
    jitter_offsets = [-0.10, 0.0, 0.10]
    for i, row in enumerate(per_sample_reduction):
        ax.scatter(
            x + jitter_offsets[i], row,
            color="white", edgecolors=DL_COLOR,
            s=70, linewidths=2.0, zorder=5,
        )

    # 1× baseline — no improvement
    ax.axhline(
        1.0, color=RED_LINE, linewidth=2.0,
        linestyle="--", alpha=0.80, zorder=4,
        label="No improvement  (1×)",
    )

    # Value labels
    for bar, val, sd in zip(bars, mean_red, std_red):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + sd + 1.5,
            f"{val:.0f}×",
            ha="center", va="bottom",
            fontsize=15, fontweight="bold", color=DL_COLOR,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(PARAM_LABELS, fontsize=24)
    ax.set_ylabel("IQR Reduction  (ECM IQR / DL IQR)", fontsize=19, labelpad=10)
    ax.tick_params(labelsize=17, length=5)
    ax.set_ylim(0, (mean_red + std_red).max() * 1.30)
    ax.grid(True, axis="y", alpha=0.18, linewidth=0.7, color="#aaaaaa")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(1.0)

    ax.legend(fontsize=17, framealpha=0.96, edgecolor=GREY_FAINT,
              handlelength=2.2, loc="upper left")

    ax.set_title(
        "DL Framework Reduces Posterior Uncertainty vs ECM\n"
        "Across All Parameters  (n = 3 iPSC-RPE samples)",
        fontsize=20, fontweight="bold", pad=18, linespacing=1.6,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "iqr_reduction_bar.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
    make_iqr_reduction_figure()
    print("Done.")
