#!/usr/bin/env python3
"""
Nyquist plot comparing ECM best-fit impedance vs true spectrum,
plus bar chart of parameter errors (5 base + TER + TEC).

Style matched to the overall diagram layout.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.baselines.deterministic_fit import DeterministicPhysicsFit

# Global style to match diagram
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
})

# Shared palette
GOLD = "#c8962e"
BLUE_DERIVED = "#2d5fa1"
RED_MEDIAN = "#c0392b"
BLACK = "#1a1a1a"
GRAY_REF = "#999999"


def load_and_fit(n_fits=50):
    """Load sample 5, run ECM fit many times, return everything for both plots."""
    test_df = pd.read_csv("data/mixed_distribution_v1/test.csv")
    row = test_df.iloc[5]

    true_params = {
        "Ra": row["Ra"], "Rb": row["Rb"],
        "Ca": row["Ca"], "Cb": row["Cb"], "Rsh": row["Rsh"],
    }
    true_TER = row["TER"]
    true_TEC = row["TEC"]

    Z_real_cols = [c for c in test_df.columns if c.startswith("Z_real_")]
    Z_imag_cols = [c for c in test_df.columns if c.startswith("Z_imag_")]
    Z_real_data = row[Z_real_cols].values.astype(float)
    Z_imag_data = row[Z_imag_cols].values.astype(float)
    n_freq = len(Z_real_data)

    frequencies = np.logspace(np.log10(0.1), np.log10(1_000_000.0), n_freq)

    print(f"Sample 5 -- distribution: {row['distribution']}")
    print(f"  True:  Ra={true_params['Ra']:.2f}, Rb={true_params['Rb']:.2f}, "
          f"Ca={true_params['Ca']:.2e}, Cb={true_params['Cb']:.2e}, "
          f"Rsh={true_params['Rsh']:.2f}")
    print(f"         TER={true_TER:.2f}, TEC={true_TEC:.2e}")

    print(f"\nRunning {n_fits} independent ECM fits...")
    all_results = []
    for i in range(n_fits):
        fitter = DeterministicPhysicsFit(
            method="L-BFGS-B", use_weights=True,
            n_restarts=1, use_relative_error=True,
        )
        result = fitter.fit(frequencies, Z_real_data, Z_imag_data, verbose=False)
        if result["relative_error"] < 0.25:
            all_results.append(result)

    print(f"  {len(all_results)}/{n_fits} fits converged (rel err < 25%)")

    best = min(all_results, key=lambda r: r["relative_error"])

    print(f"\n  Best ECM: Ra={best['params_dict']['Ra']:.2f}, "
          f"Rb={best['params_dict']['Rb']:.2f}, "
          f"Ca={best['params_dict']['Ca']:.2e}, "
          f"Cb={best['params_dict']['Cb']:.2e}, "
          f"Rsh={best['params_dict']['Rsh']:.2f}")
    print(f"            TER={best['TER']:.2f}, TEC={best['TEC']:.2e}")
    print(f"  Relative impedance error: {best['relative_error']*100:.2f}%")

    return {
        "Z_real_data": Z_real_data, "Z_imag_data": Z_imag_data,
        "Z_real_fit": best["Z_real_fit"], "Z_imag_fit": best["Z_imag_fit"],
        "true_params": true_params, "true_TER": true_TER, "true_TEC": true_TEC,
        "fit_params": best["params_dict"],
        "fit_TER": best["TER"], "fit_TEC": best["TEC"],
        "rel_err": best["relative_error"],
        "all_results": all_results,
        "frequencies": frequencies,
    }


def plot_nyquist(d, out_dir):
    """Compact Nyquist plot matching diagram style."""
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    Z_real_data = d["Z_real_data"]
    Z_imag_data = d["Z_imag_data"]
    Z_real_fit = d["Z_real_fit"]
    Z_imag_fit = d["Z_imag_fit"]
    fit_TER = d["fit_TER"]
    fit_TEC = d["fit_TEC"]
    rel_err = d["rel_err"]

    # Shade inside the ECM fit curve
    sort_idx = np.argsort(Z_real_fit)
    ax.fill_between(
        Z_real_fit[sort_idx], 0, Z_imag_fit[sort_idx],
        alpha=0.15, color=GOLD, zorder=1,
    )

    # ECM fit line
    ax.plot(
        Z_real_fit, Z_imag_fit,
        "-", color=GOLD, linewidth=1.8,
        label=f"ECM fit (err = {rel_err*100:.1f}%)", zorder=2,
    )

    # True data points
    ax.plot(
        Z_real_data, Z_imag_data,
        "o", color=BLACK, markersize=2.5, markeredgewidth=0,
        label="True", zorder=3,
    )

    depth = abs(min(Z_imag_fit))

    # TER arrow
    y_arrow = -depth * 0.04
    ax.annotate(
        "", xy=(fit_TER, y_arrow), xytext=(0, y_arrow),
        arrowprops=dict(arrowstyle="<->", color=BLUE_DERIVED, lw=1.5),
        zorder=4,
    )
    ax.text(
        fit_TER / 2, y_arrow - depth * 0.07,
        f"TER = {fit_TER:.1f} $\\Omega$",
        fontsize=8, fontweight="bold", color=BLUE_DERIVED,
        ha="center", va="top",
    )

    # TEC label inside shaded region
    peak_idx = np.argmin(Z_imag_fit)
    ax.text(
        Z_real_fit[peak_idx], Z_imag_fit[peak_idx] * 0.38,
        f"TEC = {fit_TEC:.2e} F",
        fontsize=7.5, color="#7a6a4f", ha="center", va="center", alpha=0.9,
    )

    # Axis styling
    ax.set_xlabel(r"Real ($\Omega$)")
    ax.set_ylabel(r"Imaginary ($\Omega$)")
    ax.axhline(0, color="k", linewidth=0.4)
    ax.tick_params(axis="both", which="major", direction="in", length=3)
    ax.tick_params(axis="both", which="minor", direction="in", length=1.5)
    ax.minorticks_on()

    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9,
              edgecolor="#cccccc", fancybox=False, handlelength=1.5,
              borderpad=0.4, labelspacing=0.3)

    ax.set_aspect("equal")

    x_max = max(Z_real_data.max(), Z_real_fit.max()) * 1.1
    y_min = min(Z_imag_data.min(), Z_imag_fit.min()) * 1.15
    y_max = abs(y_min) * 0.15
    ax.set_xlim(-x_max * 0.03, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_title("Nyquist Plot  |  ECM Fit vs True Spectrum",
                 fontsize=10, fontweight="bold", pad=6)

    plt.tight_layout(pad=0.5)
    out_path = out_dir / "nyquist_sample5_ecm_vs_actual.png"
    fig.savefig(out_path, dpi=250, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_parameter_errors(d, out_dir):
    """Compact bar chart matching diagram style."""
    true = d["true_params"]
    fit = d["fit_params"]
    all_results = d["all_results"]

    names = ["Ra", "Rb", "Ca", "Cb", "Rsh", "TER", "TEC"]
    # Use subscript-style labels
    labels = [r"$R_a$", r"$R_b$", r"$C_a$", r"$C_b$", r"$R_{sh}$", "TER", "TEC"]

    true_log = {
        "Ra": np.log10(true["Ra"]), "Rb": np.log10(true["Rb"]),
        "Ca": np.log10(true["Ca"]), "Cb": np.log10(true["Cb"]),
        "Rsh": np.log10(true["Rsh"]),
        "TER": np.log10(d["true_TER"]), "TEC": np.log10(d["true_TEC"]),
    }

    best_fit_log = {
        "Ra": np.log10(fit["Ra"]), "Rb": np.log10(fit["Rb"]),
        "Ca": np.log10(fit["Ca"]), "Cb": np.log10(fit["Cb"]),
        "Rsh": np.log10(fit["Rsh"]),
        "TER": np.log10(d["fit_TER"]), "TEC": np.log10(d["fit_TEC"]),
    }
    best_err = [abs(true_log[n] - best_fit_log[n]) for n in names]

    all_errors = {name: [] for name in names}
    for r in all_results:
        fp = r["params_dict"]
        fl = {
            "Ra": np.log10(fp["Ra"]), "Rb": np.log10(fp["Rb"]),
            "Ca": np.log10(fp["Ca"]), "Cb": np.log10(fp["Cb"]),
            "Rsh": np.log10(fp["Rsh"]),
            "TER": np.log10(r["TER"]), "TEC": np.log10(r["TEC"]),
        }
        for name in names:
            all_errors[name].append(abs(true_log[name] - fl[name]))

    median_err = [np.median(all_errors[n]) for n in names]
    min_err = [np.min(all_errors[n]) for n in names]
    max_err = [np.max(all_errors[n]) for n in names]
    n_fits = len(all_results)

    # Print table
    print(f"\n{'='*62}")
    print(f"  True vs Best ECM Fit Parameters")
    print(f"{'='*62}")
    print(f"  {'Param':>5s}  {'True':>12s}  {'Best Fit':>12s}  {'Error (dec)':>12s}")
    print(f"  {'-'*5}  {'-'*12}  {'-'*12}  {'-'*12}")
    for n in ["Ra", "Rb", "Rsh"]:
        print(f"  {n:>5s}  {true[n]:>10.2f} O  {fit[n]:>10.2f} O  "
              f"{abs(true_log[n] - best_fit_log[n]):>10.2f}")
    for n in ["Ca", "Cb"]:
        print(f"  {n:>5s}  {true[n]:>10.2e} F  {fit[n]:>10.2e} F  "
              f"{abs(true_log[n] - best_fit_log[n]):>10.2f}")
    print(f"  {'TER':>5s}  {d['true_TER']:>10.2f} O  {d['fit_TER']:>10.2f} O  "
          f"{abs(true_log['TER'] - best_fit_log['TER']):>10.2f}")
    print(f"  {'TEC':>5s}  {d['true_TEC']:>10.2e} F  {d['fit_TEC']:>10.2e} F  "
          f"{abs(true_log['TEC'] - best_fit_log['TEC']):>10.2f}")
    print(f"{'='*62}")

    # Colors
    colors = [GOLD] * 5 + [BLUE_DERIVED] * 2
    bar_width = 0.55

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    x = np.arange(len(names))

    # Bars at best-fit error
    bars = ax.bar(x, best_err, width=bar_width, color=colors, edgecolor="#444444",
                  linewidth=0.5, zorder=3, alpha=0.9)

    # Whiskers: full range
    err_lower = [be - lo for be, lo in zip(best_err, min_err)]
    err_upper = [hi - be for be, hi in zip(best_err, max_err)]
    ax.errorbar(x, best_err, yerr=[err_lower, err_upper],
                fmt="none", ecolor="#444444", elinewidth=1.2,
                capsize=4, capthick=1.0, zorder=4)

    # Median line inside each bar
    for i, med in enumerate(median_err):
        left = x[i] - bar_width / 2 + 0.03
        right = x[i] + bar_width / 2 - 0.03
        ax.plot([left, right], [med, med],
                color="white", linewidth=2.0, zorder=5)
        ax.plot([left, right], [med, med],
                color=RED_MEDIAN, linewidth=1.2, zorder=6)

    # Value labels
    for i, (bar, be) in enumerate(zip(bars, best_err)):
        label_y = max_err[i] + 0.04
        ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                f"{be:.2f}", ha="center", va="bottom", fontsize=7.5,
                fontweight="bold", color="#333333")

    # Reference line
    ax.axhline(0.3, color=GRAY_REF, linewidth=0.7, linestyle="--", zorder=2)
    ax.text(len(names) - 0.5, 0.32, "0.3 dec", fontsize=6.5,
            color=GRAY_REF, ha="right", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Absolute Error (decades)")
    ax.set_title(
        f"ECM Fit Parameter Errors  |  {n_fits} Independent Fits\n"
        "Bars = best fit, red line = median, whiskers = full range",
        fontsize=9, fontweight="bold", pad=6,
    )

    ax.tick_params(axis="y", which="major", direction="in", length=3)
    ax.tick_params(axis="y", which="minor", direction="in", length=1.5)
    ax.tick_params(axis="x", which="major", length=0)
    ax.minorticks_on()
    ax.set_xlim(-0.5, len(names) - 0.5)
    ax.set_ylim(0, max(max_err) * 1.25)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    legend_elements = [
        Patch(facecolor=GOLD, edgecolor="#444", label="Base parameters"),
        Patch(facecolor=BLUE_DERIVED, edgecolor="#444", label="Derived quantities"),
        Line2D([0], [0], color=RED_MEDIAN, linewidth=1.2, label="Median error"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              framealpha=0.9, edgecolor="#cccccc", fancybox=False,
              borderpad=0.4, labelspacing=0.3)

    ax.grid(axis="y", alpha=0.15, zorder=1)

    plt.tight_layout(pad=0.5)
    out_path = out_dir / "parameter_errors_sample5.png"
    fig.savefig(out_path, dpi=250, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    out_dir = Path("figures/nyquist")
    out_dir.mkdir(parents=True, exist_ok=True)

    d = load_and_fit()
    plot_nyquist(d, out_dir)
    plot_parameter_errors(d, out_dir)


if __name__ == "__main__":
    main()
