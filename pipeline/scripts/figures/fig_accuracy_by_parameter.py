#!/usr/bin/env python3
"""
12g - Accuracy Bar Chart

Mean Absolute Error (MAE) per parameter:
  MAE = mean over time of |method_median - 3P-EIS_truth|

Compares DL framework vs ECM fitting across all 3 iPSC-RPE
samples. Shows whether each method is tracking the correct values, not
just whether its posteriors are narrow.

Usage:
    cd pipeline && python fig_accuracy_by_parameter.py
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

SAMPLES = {
    "Sample 1": {"interval_min": 3.5, "atp_span": (7,  11)},
    "Sample 2": {"interval_min": 2.4, "atp_span": (10, 15)},
    "Sample 3": {"interval_min": 1.7, "atp_span": (9,  12)},
}

DISPLAY_ORDER = [2, 3, 0, 1, 4]   # Ca, Cb, Ra, Rb, Rsh
PARAM_LABELS  = [r"$C_a$", r"$C_b$", r"$R_a$", r"$R_b$", r"$R_{sh}$"]
PARAM_UNITS   = [r"$\mu$F cm$^{-2}$", r"$\mu$F cm$^{-2}$",
                 r"k$\Omega$ cm$^{2}$", r"k$\Omega$ cm$^{2}$",
                 r"k$\Omega$ cm$^{2}$"]

mpl.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.default": "regular",
})


# ── Cache ─────────────────────────────────────────────────────────────────────
def load_cache(sname):
    p = CACHE_DIR / f"cache_{sname.replace(' ', '_')}.npz"
    d = np.load(p, allow_pickle=True)
    return {
        "n_t":        int(d["n_t"][0]),
        "truth_disp": d["truth_disp"],
        "dl_disp":    list(d["dl_disp"]),
        "ecm_disp":   list(d["ecm_disp"]),
    }


# ── MAE computation ───────────────────────────────────────────────────────────
def compute_mae(res):
    """
    For each parameter, compute time-averaged MAE:
      MAE_DL[p]  = mean_t( |median(DL_t[:,p])  - truth_t[p]| )
      MAE_ECM[p] = mean_t( |median(ECM_t[:,p]) - truth_t[p]| )

    Returns mae_dl (5,), mae_ecm (5,) in display units.
    """
    truth = res["truth_disp"]   # (n_t, 5)
    dl    = res["dl_disp"]
    ecm   = res["ecm_disp"]

    mae_dl  = np.zeros(5)
    mae_ecm = np.zeros(5)

    for p in range(5):
        dl_med  = np.array([np.nanmedian(d[:, p]) for d in dl])
        ecm_med = np.array([
            np.nanmedian(e[:, p]) if len(e) > 0 else np.nan for e in ecm
        ])
        mae_dl[p]  = np.nanmean(np.abs(dl_med  - truth[:, p]))
        mae_ecm[p] = np.nanmean(np.abs(ecm_med - truth[:, p]))

    return mae_dl, mae_ecm


# ── Figure ────────────────────────────────────────────────────────────────────
def make_figure():
    per_sample_dl  = []
    per_sample_ecm = []

    for sname in SAMPLES:
        res = load_cache(sname)
        mae_dl, mae_ecm = compute_mae(res)
        per_sample_dl.append(mae_dl[DISPLAY_ORDER])
        per_sample_ecm.append(mae_ecm[DISPLAY_ORDER])

    per_sample_dl  = np.array(per_sample_dl)   # (3, 5)
    per_sample_ecm = np.array(per_sample_ecm)

    mean_dl  = per_sample_dl.mean(axis=0)
    std_dl   = per_sample_dl.std(axis=0)
    mean_ecm = per_sample_ecm.mean(axis=0)
    std_ecm  = per_sample_ecm.std(axis=0)

    n_params = len(PARAM_LABELS)
    x     = np.arange(n_params)
    width = 0.32

    fig, ax = plt.subplots(figsize=(12, 10))

    bars_dl = ax.bar(
        x - width / 2, mean_dl, width,
        color=DL_COLOR, alpha=0.88,
        yerr=std_dl, capsize=8,
        error_kw={"linewidth": 2.0, "color": BLACK, "capthick": 2.0},
        zorder=3, label="DL framework",
    )
    bars_ecm = ax.bar(
        x + width / 2, mean_ecm, width,
        color=GOLD, alpha=0.88,
        yerr=std_ecm, capsize=8,
        error_kw={"linewidth": 2.0, "color": BLACK, "capthick": 2.0},
        zorder=3, label="ECM",
    )

    # Individual sample dots
    jitter = [-0.07, 0.0, 0.07]
    for i, (s_dl, s_ecm) in enumerate(zip(per_sample_dl, per_sample_ecm)):
        ax.scatter(
            x - width / 2 + jitter[i], s_dl,
            color="white", edgecolors=DL_COLOR,
            s=65, linewidths=2.0, zorder=5,
        )
        ax.scatter(
            x + width / 2 + jitter[i], s_ecm,
            color="white", edgecolors=GOLD,
            s=65, linewidths=2.0, zorder=5,
        )

    # MAE value labels
    for bar, val, sd in zip(bars_dl, mean_dl, std_dl):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + sd + max(mean_ecm) * 0.02,
            f"{val:.2f}",
            ha="center", va="bottom",
            fontsize=15, fontweight="bold", color=DL_COLOR,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )
    for bar, val, sd in zip(bars_ecm, mean_ecm, std_ecm):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + sd + max(mean_ecm) * 0.02,
            f"{val:.2f}",
            ha="center", va="bottom",
            fontsize=15, fontweight="bold", color=GOLD,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )

    combined_labels = [f"{p}\n{u}" for p, u in zip(PARAM_LABELS, PARAM_UNITS)]
    ax.set_xticks(x)
    ax.set_xticklabels(combined_labels, fontsize=20, linespacing=1.5)
    ax.set_ylabel("Mean Absolute Error  (display units)", fontsize=21, labelpad=12)
    ax.tick_params(axis="y", labelsize=18, length=5)
    ax.tick_params(axis="x", length=5)
    ax.set_ylim(0, (mean_ecm + std_ecm).max() * 1.28)
    ax.grid(True, axis="y", alpha=0.18, linewidth=0.7, color="#aaaaaa")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(1.0)

    ax.legend(
        fontsize=19, framealpha=0.96, edgecolor=GREY_FAINT,
        handlelength=2.2, loc="upper right",
    )

    ax.set_title(
        "DL Framework Tracks Ground Truth More Accurately than ECM\n"
        r"MAE = mean$_t$  |method median $-$ 3P-EIS truth|  "
        "(n = 3 iPSC-RPE samples)",
        fontsize=21, fontweight="bold", pad=20, linespacing=1.6,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "accuracy_bar.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
    print("Done.")
