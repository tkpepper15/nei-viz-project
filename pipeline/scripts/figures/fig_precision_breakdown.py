#!/usr/bin/env python3
"""
12m - Precision Breakdown: What the Spectrum Knows vs What We Extract

Three-panel figure designed to make the CRLB immediately interpretable.

WHAT THIS SHOWS
---------------
Panel A  "Information directions" — The FIM has 5 eigenvectors, each a
         linear combination of parameters. The eigenvalue of each tells you
         how precisely that combination can be measured. Direction 5 is
         essentially Ra alone and has eigenvalue ~0.003 → CRLB = 19 decades:
         the spectrum fundamentally cannot determine Ra independently.

Panel B  "Precision budget" — For every parameter and identifiable derived
         quantity, show the CRLB floor alongside what each pipeline stage
         actually achieves. The gap from CRLB to pipeline = room to improve.
         Shown in log10-decade units with % equivalent on the right axis.

Panel C  "How biology changes difficulty" — Condition number κ(F) over time
         per sample. κ spikes during ATP (Ra drops toward Rb, the two time
         constants merge, the degenerate direction gets worse).

Usage:
    cd pipeline && python fig_precision_breakdown.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

sys.path.insert(0, ".")
from src.physics.eis_fisher import (
    compute_impedance,
    compute_jacobian_autodiff,
    compute_derived_from_log10,
)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "mathtext.default": "regular",
})

# ── colours ──────────────────────────────────────────────────────────────
BLACK           = "#1a1a1a"
GOLD            = "#c8962e"
DL_COLOR        = "#2166ac"
STAGE_A_COLOR   = "#7b2d8b"
STAGE_ABL_COLOR = "#15803d"
ATP_COLOR       = "#f4a7b9"

IDENT_GREEN  = "#16a34a"
DEGEN_RED    = "#dc2626"
NEUTRAL_GREY = "#6b7280"

PARAM_NAMES  = ["Ra", "Rb", "Ca", "Cb", "Rsh"]
PARAM_LABELS = [r"$R_a$", r"$R_b$", r"$C_a$", r"$C_b$", r"$R_{sh}$"]

SAMPLES = {
    "Sample 1": {"interval_min": 3.5, "atp_span": (7,  11), "donor": "Donor A"},
    "Sample 2": {"interval_min": 2.4, "atp_span": (10, 15), "donor": "Donor A"},
    "Sample 3": {"interval_min": 1.7, "atp_span": (9,  12), "donor": "Donor B"},
}

N_FREQ      = 100
FREQ_MIN_HZ = 0.1
FREQ_MAX_HZ = 1e6
BL_ALPHA    = 0.02
BL_EPS      = 0.5


# ── helpers ───────────────────────────────────────────────────────────────

def decades_to_pct(d):
    """log10-decade std → approximate % factor uncertainty."""
    return (10.0 ** d - 1.0) * 100.0


def log10_std_from_band(lo, hi):
    """(n_t,5) display ±1σ bands → (n_t,5) log10 std. Conversion cancels."""
    return 0.5 * np.log10(np.maximum(hi / np.maximum(lo, 1e-30), 1.0))


def log10_std_from_samples(samples_list, p):
    """list[n_t] of (n_samples,5) display arrays → (n_t,) log10 std."""
    out = []
    for s in samples_list:
        arr = np.asarray(s, dtype=np.float64)
        v   = arr[:, p]
        v   = v[v > 0]
        out.append(np.std(np.log10(v)) if len(v) >= 2 else np.nan)
    return np.array(out)


def compute_fim_eigenvectors(params_log10, omega_np):
    """FIM at given log10 params, returns eigenvalues (desc) and eigenvectors."""
    params_t = torch.tensor(params_log10, dtype=torch.float32).unsqueeze(0)
    omega_t  = torch.tensor(omega_np, dtype=torch.float32)
    Ra, Rb, Ca, Cb, Rsh = [float(10 ** v) for v in params_log10]
    Z_r, Z_i = compute_impedance(
        torch.tensor([Ra]), torch.tensor([Rb]), torch.tensor([Ca]),
        torch.tensor([Cb]), torch.tensor([Rsh]), omega_t)
    noise_var  = (BL_ALPHA**2) * (Z_r**2 + Z_i**2) + BL_EPS**2
    omega_diag = torch.cat([noise_var, noise_var], dim=1)[0].numpy()
    with torch.enable_grad():
        J = compute_jacobian_autodiff(params_t, omega_t)[0].detach().numpy()
    inv_w = 1.0 / (omega_diag + 1e-30)
    FIM   = 0.5 * (J.T @ (inv_w[:, None] * J))
    FIM   = 0.5 * (FIM + FIM.T)
    ev, evec = np.linalg.eigh(FIM)
    return ev[::-1].copy(), evec[:, ::-1].copy(), FIM


# ── cache loading ─────────────────────────────────────────────────────────

def load_cache(tag, sname, cache_root="results/poster"):
    safe = sname.replace(" ", "_")
    p = Path(cache_root) / f"cache_{tag}" / f"cache_{safe}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing cache: {p}")
    return np.load(p, allow_pickle=True)


def aggregate_pipeline_stds(sample_names, cache_root="results/poster"):
    """
    For each parameter, compute median log10-std across all timepoints
    and all 3 samples, for each pipeline stage.
    Returns dict keyed by stage name → (5,) array.
    """
    stageA_stds   = {i: [] for i in range(5)}
    stageABL_stds = {i: [] for i in range(5)}
    dl_stds       = {i: [] for i in range(5)}
    ecm_stds      = {i: [] for i in range(5)}

    for sname in sample_names:
        d12e = load_cache("12e", sname, cache_root)
        d12k = load_cache("12k", sname, cache_root)

        sigA   = log10_std_from_band(d12k["stageA_lo"],   d12k["stageA_hi"])   # (n_t,5)
        sigABL = log10_std_from_band(d12k["stageABL_lo"], d12k["stageABL_hi"]) # (n_t,5)

        dl_list  = list(d12e["dl_disp"])
        ecm_list = list(d12e["ecm_disp"])

        for p in range(5):
            stageA_stds[p].extend(sigA[:, p].tolist())
            stageABL_stds[p].extend(sigABL[:, p].tolist())
            dl_stds[p].extend(log10_std_from_samples(dl_list,  p).tolist())
            for ecm_t in ecm_list:
                arr = np.asarray(ecm_t, dtype=np.float64)
                if len(arr) >= 2:
                    v = arr[:, p]; v = v[v > 0]
                    if len(v) >= 2:
                        q75, q25 = np.nanpercentile(np.log10(v), [75, 25])
                        ecm_stds[p].append((q75 - q25) / 1.35)

    def _med(d):
        return np.array([np.nanmedian(d[p]) for p in range(5)])

    return {
        "Stage A":       _med(stageA_stds),
        "Stage A+C(BL)": _med(stageABL_stds),
        "Full A+C+KF":   _med(dl_stds),
        "ECM":           _med(ecm_stds),
    }


def aggregate_crlb_stds(sample_names, cache_root="results/poster"):
    """Median CRLB std (base params + derived) across all samples/timepoints."""
    base = {i: [] for i in range(5)}
    ter, tau_b, tau_a, tec = [], [], [], []

    for sname in sample_names:
        d = load_cache("12l", sname, cache_root)
        for p in range(5):
            base[p].extend(d["crlb_std"][:, p].tolist())
        ter.extend(d["crlb_derived_ter"].tolist())
        tau_b.extend(d["crlb_derived_tau_b"].tolist())
        tau_a.extend(d["crlb_derived_tau_a"].tolist())
        tec.extend(d["crlb_derived_tec"].tolist())

    base_med = np.array([np.nanmedian(base[p]) for p in range(5)])
    return base_med, {
        "TER":    float(np.nanmedian(ter)),
        "tau_b":  float(np.nanmedian(tau_b)),
        "tau_a":  float(np.nanmedian(tau_a)),
        "TEC":    float(np.nanmedian(tec)),
    }


# ── Panel A: FIM eigenvector decomposition ────────────────────────────────

def draw_fim_directions(ax_list, eigvals, eigvecs):
    """
    Draw 5 eigenvector bar charts into ax_list[0..4].
    Each chart shows the loading of each parameter on that information direction.
    Background color: green (identifiable) → red (degenerate) by eigenvalue.
    """
    max_ev = max(eigvals[0], 1e-30)
    # Map eigenvalue to colour: interpolate green→yellow→red on log scale
    def ev_color(ev):
        t = np.clip(np.log10(max(ev, 1e-30) / max_ev) / 8.0 + 1.0, 0.0, 1.0)
        # t=1 → identifiable (green), t=0 → degenerate (red)
        r = 1.0 - t * 0.7
        g = 0.2 + t * 0.6
        b = 0.2
        return (r, g, b)

    bar_colors_pos = "#2563eb"   # blue for positive loading
    bar_colors_neg = "#f97316"   # orange for negative loading

    for i, (ax, ev, evec) in enumerate(zip(ax_list, eigvals, eigvecs.T)):
        crlb_d = 1.0 / np.sqrt(max(abs(ev), 1e-30))
        crlb_pct = decades_to_pct(crlb_d)
        bg = ev_color(ev)

        ax.set_facecolor((*bg, 0.10))
        ax.axhline(0, color=BLACK, linewidth=0.8, alpha=0.5)

        colors = [bar_colors_pos if v >= 0 else bar_colors_neg for v in evec]
        bars = ax.bar(range(5), evec, color=colors, edgecolor="white",
                      linewidth=0.8, width=0.65, zorder=3)

        ax.set_xticks(range(5))
        ax.set_xticklabels(PARAM_LABELS, fontsize=12)
        ax.set_ylim(-1.1, 1.1)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.tick_params(labelsize=10)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="y", alpha=0.20, linewidth=0.6)

        # Title with eigenvalue and CRLB
        label_color = IDENT_GREEN if ev > 100 else (NEUTRAL_GREY if ev > 1 else DEGEN_RED)
        if crlb_d < 50:
            crlb_str = f"±{crlb_d:.3f} dec\n({crlb_pct:.1f}%)"
        else:
            crlb_str = f"CRLB = {crlb_d:.0f} dec\n(unidentifiable)"

        ax.set_title(
            f"Direction {i+1}\nλ = {ev:.2e}",
            fontsize=12, fontweight="bold", color=label_color,
        )
        ax.text(0.5, 0.02, crlb_str, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=9.5,
                color=label_color, fontweight="bold")

        if i == 0:
            ax.set_ylabel("Parameter loading", fontsize=11)


# ── Panel B: precision budget ─────────────────────────────────────────────

def draw_precision_budget(ax, crlb_base, crlb_derived, pipeline_stds):
    """
    Horizontal dot chart: each row = one parameter or derived quantity.
    X-axis: uncertainty std in log10 decades (log scale).
    One marker per pipeline stage + CRLB floor.
    """
    # Ordered from most to least identifiable by CRLB
    items = [
        # (label,          crlb_val,              derived?)
        (r"$R_{sh}$",      crlb_base[4],           False),
        ("TER",            crlb_derived["TER"],     True),
        ("TEC",            crlb_derived["TEC"],     True),
        (r"$C_b$",         crlb_base[3],            False),
        (r"$\tau_b$",      crlb_derived["tau_b"],   True),
        (r"$C_a$",         crlb_base[2],            False),
        (r"$\tau_a$",      crlb_derived["tau_a"],   True),
        (r"$R_a$",         crlb_base[0],            False),
        (r"$R_b$",         crlb_base[1],            False),
    ]
    # pipeline stds mapped by parameter index or derived key
    param_idx_map = {r"$R_{sh}$": 4, r"$C_b$": 3, r"$C_a$": 2,
                     r"$R_a$": 0, r"$R_b$": 1}
    pipeline_derived_map = {
        # derived quantities: use CRLB as proxy lower bound
        # (we don't compute pipeline std for derived separately here)
        "TER": "TER", "TEC": "TEC", r"$\tau_b$": "tau_b", r"$\tau_a$": "tau_a",
    }

    stages = [
        ("Stage A",       STAGE_A_COLOR,   "o",  6.0),
        ("Stage A+C(BL)", STAGE_ABL_COLOR, "s",  6.0),
        ("Full A+C+KF",   DL_COLOR,        "D",  6.5),
        ("ECM",           GOLD,            "^",  6.0),
    ]

    n = len(items)
    y_positions = np.arange(n)[::-1]  # top = most identifiable

    ax.set_xscale("log")

    for yi, (label, crlb_val, is_derived) in zip(y_positions, items):
        # CRLB floor (downward triangle)
        ax.scatter([crlb_val], [yi], marker="v", s=110, color=BLACK,
                   zorder=6, label="CRLB" if yi == y_positions[0] else "")
        ax.axhline(yi, color="#e5e7eb", linewidth=0.8, zorder=0)

        # Pipeline stages
        if label in param_idx_map:
            pidx = param_idx_map[label]
            for sname, color, marker, ms in stages:
                val = pipeline_stds[sname][pidx]
                if np.isfinite(val) and val > 0:
                    ax.scatter([val], [yi], marker=marker, s=ms**2,
                               color=color, edgecolors="white", linewidths=0.8,
                               zorder=5)
                    # connect CRLB to this stage
            # draw connecting line from CRLB to max pipeline
            vals = [pipeline_stds[s][pidx] for s, _, _, _ in stages
                    if np.isfinite(pipeline_stds[s][pidx])]
            if vals:
                ax.plot([crlb_val, max(vals)], [yi, yi],
                        color="#9ca3af", linewidth=1.2, zorder=2)

        # % annotation beside CRLB marker
        pct = decades_to_pct(crlb_val)
        pct_str = f"{pct:.1f}%" if pct < 200 else f"{pct:.0f}%"
        ax.text(crlb_val * 0.82, yi, pct_str, ha="right", va="center",
                fontsize=8.5, color=BLACK, fontstyle="italic")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([item[0] for item in items], fontsize=13)
    ax.set_xlabel("Precision floor  (std in log₁₀ decades)", fontsize=12)
    ax.set_xlim(5e-4, 5.0)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))
    ax.tick_params(axis="x", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="x", which="both", alpha=0.15, linewidth=0.6)

    # Divider between base and derived
    # Mark derived items differently
    for yi, (label, _, is_derived) in zip(y_positions, items):
        if is_derived:
            ax.get_yticklabels()[list(y_positions).index(yi)].set_color("#6d28d9")

    # Vertical reference lines
    for x, lbl in [(0.01, "1%"), (0.1, "26%"), (1.0, "10×")]:
        ax.axvline(x, color="#e5e7eb", linewidth=1.0, linestyle="--", zorder=0)

    # Secondary axis: % equivalent
    ax2 = ax.twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(ax.get_xlim())
    pct_ticks = [0.001, 0.01, 0.1, 1.0]
    pct_labels = [f"{decades_to_pct(v):.1g}%" for v in pct_ticks]
    ax2.set_xticks(pct_ticks)
    ax2.set_xticklabels(pct_labels, fontsize=9, color=NEUTRAL_GREY)
    ax2.set_xlabel("Equivalent % parameter uncertainty", fontsize=10,
                   color=NEUTRAL_GREY)

    # Legend
    handles = [
        Line2D([0], [0], marker="v", color="w", markerfacecolor=BLACK,
               markersize=9, label="CRLB (theoretical minimum)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=STAGE_A_COLOR,
               markersize=8, label="Stage A: Transformer only"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=STAGE_ABL_COLOR,
               markersize=8, label="Stage A+C: + BL refinement"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=DL_COLOR,
               markersize=8, label="Full A+C+KF: + Kalman filter"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=GOLD,
               markersize=8, label="ECM: classical fitting"),
    ]
    ax.legend(handles=handles, fontsize=10, loc="lower right",
              framealpha=0.95, edgecolor="#d1d5db")

    ax.set_title("Precision budget: CRLB floor vs pipeline stages\n"
                 r"(purple = derived quantity, black $\blacktriangledown$ = theoretical minimum)",
                 fontsize=13, fontweight="bold", pad=10)


# ── Panel C: condition number over time ───────────────────────────────────

def draw_condition_number(ax):
    sample_colors = [DL_COLOR, STAGE_ABL_COLOR, STAGE_A_COLOR]
    sample_names  = list(SAMPLES.keys())

    for col, sname in zip(sample_colors, sample_names):
        d   = load_cache("12l", sname)
        cfg = SAMPLES[sname]
        n_t = d["condition_numbers"].shape[0]
        t   = np.arange(n_t) * cfg["interval_min"]
        kappa = np.log10(np.maximum(d["condition_numbers"], 1.0))

        ax.plot(t, kappa, color=col, linewidth=2.0, label=sname)
        # ATP shading
        a0 = cfg["atp_span"][0] * cfg["interval_min"]
        a1 = cfg["atp_span"][1] * cfg["interval_min"]
        ax.axvspan(a0, a1, color=ATP_COLOR, alpha=0.4, zorder=0)

    ax.axhline(8, color=DEGEN_RED, linewidth=1.2, linestyle="--", alpha=0.7)
    ax.text(0.5, 8.05, "κ = 10⁸ threshold", color=DEGEN_RED,
            fontsize=9, ha="left", va="bottom")

    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("log₁₀(condition number κ)", fontsize=12)
    ax.set_title("Estimation difficulty over time\n"
                 "κ ↑ = more degenerate (harder)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.15)

    # Add ATP label
    ax.text(0.72, 0.92, "ATP\nstimulation", transform=ax.transAxes,
            fontsize=9, ha="center", color="#9f1239",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=ATP_COLOR, alpha=0.6))


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/poster")
    args = parser.parse_args()

    sample_names = list(SAMPLES.keys())

    # Compute FIM at median parameters across all samples
    print("Computing representative FIM (median parameters)...")
    all_truth = []
    for sname in sample_names:
        d = load_cache("12l", sname)
        all_truth.append(d["truth_log10"])
    median_params = np.median(np.concatenate(all_truth, axis=0), axis=0)
    print(f"  Median log10 params: Ra={median_params[0]:.3f} Rb={median_params[1]:.3f} "
          f"Ca={median_params[2]:.3f} Cb={median_params[3]:.3f} Rsh={median_params[4]:.3f}")

    freqs    = np.logspace(np.log10(0.1), np.log10(1e6), 100)
    omega_np = 2 * np.pi * freqs
    eigvals, eigvecs, FIM = compute_fim_eigenvectors(median_params, omega_np)

    print("  FIM eigenvalues:", [f"{v:.2e}" for v in eigvals])
    for i in range(5):
        crlb = 1.0 / np.sqrt(max(abs(eigvals[i]), 1e-30))
        loadings = "  ".join(f"{n}:{eigvecs[:,i][j]:+.3f}"
                             for j, n in enumerate(PARAM_NAMES))
        print(f"  Dir {i+1}: λ={eigvals[i]:.2e}  CRLB={crlb:.3f}  [{loadings}]")

    # Aggregate precision stats
    print("\nAggregating pipeline precision across samples...")
    pipeline_stds = aggregate_pipeline_stds(sample_names)
    crlb_base, crlb_derived = aggregate_crlb_stds(sample_names)

    print("\nMedian CRLB (log10 decades):")
    for i, n in enumerate(PARAM_NAMES):
        print(f"  {n}: {crlb_base[i]:.4f} dec  ({decades_to_pct(crlb_base[i]):.1f}%)")
    print("  Derived:")
    for k, v in crlb_derived.items():
        print(f"    {k}: {v:.4f} dec  ({decades_to_pct(v):.1f}%)")

    print("\nMedian pipeline stds:")
    for stage, stds in pipeline_stds.items():
        vals = "  ".join(f"{n}:{stds[i]:.3f}" for i, n in enumerate(PARAM_NAMES))
        print(f"  {stage}: {vals}")

    # ── Build figure ──────────────────────────────────────────────────────
    print("\nBuilding figure...")

    fig = plt.figure(figsize=(28, 18))
    gs  = fig.add_gridspec(
        2, 6,
        hspace=0.50, wspace=0.55,
        top=0.91, bottom=0.08,
        left=0.06, right=0.98,
    )

    # Row 0: 5 eigenvector subplots (cols 0-4) + spacer
    ax_dirs = [fig.add_subplot(gs[0, i]) for i in range(5)]

    # Row 1: precision budget (cols 0-3), condition number (cols 4-5)
    ax_budget = fig.add_subplot(gs[1, 0:4])
    ax_kappa  = fig.add_subplot(gs[1, 4:6])

    # ── Panel A ──
    draw_fim_directions(ax_dirs, eigvals, eigvecs)

    # Shared row title
    fig.text(
        0.50, 0.955,
        "A   What does the impedance spectrum contain? — Fisher Information Matrix eigenvectors\n"
        "Each direction = a linear combination of parameters. Eigenvalue = how precisely it can be measured.",
        ha="center", va="top", fontsize=13, fontweight="bold", color=BLACK,
    )

    # Eigenvalue colorbar legend (manual)
    ax_cb = fig.add_axes([0.88, 0.525, 0.01, 0.42])
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "ident", [DEGEN_RED, "#facc15", IDENT_GREEN], N=256)
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=8), cmap=cmap),
        cax=ax_cb,
    )
    cb.set_label("log₁₀(eigenvalue)", fontsize=9)
    cb.set_ticks([0, 2, 4, 6, 8])
    cb.ax.tick_params(labelsize=8)

    # ── Panel B ──
    draw_precision_budget(ax_budget, crlb_base, crlb_derived, pipeline_stds)

    fig.text(0.03, 0.47, "B", fontsize=16, fontweight="bold", color=BLACK)
    fig.text(0.67, 0.47, "C", fontsize=16, fontweight="bold", color=BLACK)

    # ── Panel C ──
    draw_condition_number(ax_kappa)

    fig.suptitle(
        "Cramér-Rao Bound Analysis: Information Limits of EIS Parameter Estimation",
        fontsize=20, fontweight="bold", y=0.995,
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "precision_breakdown.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
