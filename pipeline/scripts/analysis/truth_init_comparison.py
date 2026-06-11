"""
truth_init_comparison.py

Tests the core question: does initialization determine the outcome, or does the
degenerate manifold attract all solutions regardless of starting point?

Three conditions on the same N_TEST spectra (ground truth known):

  A  ECM cold start   -- L-BFGS-B with 8 random restarts. Current baseline.
  B  SMC truth-init   -- Particle cloud started AT the true parameters (sigma=0.05
                         log10). SVGD updates find the posterior. If initialization
                         mattered, this should win convincingly.
  C  SMC blind        -- Particle cloud uniform random in bounds. Shows raw SMC
                         without any privileged starting information.

Figure panels (3 x 4):

  Row 0  RESNORM LANDSCAPE
    [0,0]  Violin: resnorm distribution for A / B / C
    [0,1]  Scatter: ECM resnorm vs Truth-SMC resnorm per spectrum
    [0,2]  Histogram: fractional improvement (ECM - TruthSMC) / ECM
    [0,3]  Cumulative: % of spectra where truth-init beats ECM by threshold

  Row 1  DEGENERATE PARAMETER MANIFOLD  (Ra/Rb, Ca/Cb — the problematic pairs)
    [1,0]  Ra vs Rb cloud: ECM solutions, colored by log-resnorm
    [1,1]  Ra vs Rb cloud: truth-SMC posterior medians, same coloring
    [1,2]  (Ca/Cb)_pred vs (Ca/Cb)_true — ratio comparison, both conditions
    [1,3]  Symmetric error distribution: how often does each condition swap Ra/Rb?

  Row 2  IDENTIFIABLE PARAMETER RECOVERY + PHASE PORTRAIT
    [2,0]  TER_pred vs TER_true: ECM and truth-SMC overlaid
    [2,1]  tau_big_pred vs tau_big_true: both conditions
    [2,2]  Phase portrait: sample solutions in (TER, tau_big) space for 30 spectra
    [2,3]  Summary statistics table

Output: comparison_truth_init.png
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

warnings.filterwarnings('ignore')

# ---- Path setup ----
ROOT = Path(__file__).parent.parent.parent  # pipeline/
sys.path.insert(0, str(ROOT / 'src'))

from pipeline.smc_filter import (
    smc_step, _impedance_batch,
    BOUNDS_LOW, BOUNDS_HIGH, BOUNDS_MID,
)

try:
    from scipy.optimize import minimize as _scipy_min
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("scipy not found — ECM condition will be skipped")

# ---- Configuration ----
DATA_PATH  = ROOT / 'data' / 'mixed_distribution_v2' / 'test.csv'
N_TEST     = 200       # spectra to evaluate (subset for speed)
N_FREQ     = 100       # frequency points in the spectra
FREQ_MIN   = 0.1       # Hz
FREQ_MAX   = 1e6       # Hz
N_PARTICLES = 200      # particles per SMC run
N_ECM_STARTS = 8      # random restarts for ECM (speed/quality trade-off)
SMC_BETAS  = [0.25, 0.5, 0.75, 1.0]  # annealing schedule
SMC_SVGD_STEPS = 5    # SVGD rejuvenation steps per beta
SMC_STEP_SIZE  = 0.025
TRUTH_SIGMA    = 0.05  # log10 noise around truth for condition B
SEED           = 42
RNG            = np.random.default_rng(SEED)

FREQS_HZ = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N_FREQ)
OMEGA     = 2 * np.pi * FREQS_HZ      # rad/s

# ---- Impedance resnorm (channel-max, matches ECM and SMC objectives) ----

def compute_resnorm(log10_params: np.ndarray, Z_r: np.ndarray, Z_i: np.ndarray) -> float:
    p = log10_params[np.newaxis, :]
    Zr_pred, Zi_pred = _impedance_batch(p, OMEGA)
    nr = np.max(np.abs(Z_r)) + 1e-20
    ni = np.max(np.abs(Z_i)) + 1e-20
    return float(np.sum(((Z_r - Zr_pred[0]) / nr)**2 + ((Z_i - Zi_pred[0]) / ni)**2))


def compute_resnorm_batch(particles: np.ndarray, Z_r: np.ndarray, Z_i: np.ndarray) -> np.ndarray:
    Zr_pred, Zi_pred = _impedance_batch(particles, OMEGA)
    nr = np.max(np.abs(Z_r)) + 1e-20
    ni = np.max(np.abs(Z_i)) + 1e-20
    return np.sum(((Z_r[np.newaxis, :] - Zr_pred) / nr)**2 +
                  ((Z_i[np.newaxis, :] - Zi_pred) / ni)**2, axis=1)


# ---- ECM cold start ----

def ecm_cold(Z_r: np.ndarray, Z_i: np.ndarray, n_starts: int = N_ECM_STARTS):
    """L-BFGS-B with n_starts random restarts. Returns (best_log10_params, resnorm)."""
    if not HAS_SCIPY:
        return None, None

    bounds = list(zip(BOUNDS_LOW.tolist(), BOUNDS_HIGH.tolist()))

    def cost(x: np.ndarray) -> float:
        return compute_resnorm(x, Z_r, Z_i)

    best_x = None
    best_c = np.inf
    for _ in range(n_starts):
        x0 = RNG.uniform(BOUNDS_LOW, BOUNDS_HIGH)
        try:
            res = _scipy_min(cost, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 300, 'ftol': 1e-12})
            if res.fun < best_c:
                best_c = float(res.fun)
                best_x = np.array(res.x)
        except Exception:
            pass

    return best_x, best_c


# ---- SMC helpers ----

def run_smc(init_particles: np.ndarray, Z_r: np.ndarray, Z_i: np.ndarray):
    """Annealed SMC, returns (posterior_particles, log_evidence, best_resnorm)."""
    resampled, log_ev, diag = smc_step(
        init_particles, Z_r, Z_i, OMEGA,
        betas=SMC_BETAS,
        n_mcmc_steps=SMC_SVGD_STEPS,
        step_size=SMC_STEP_SIZE,
        prior_weight=0.0,
    )
    resnorms = compute_resnorm_batch(resampled, Z_r, Z_i)
    best_resnorm = float(np.min(resnorms))
    return resampled, log_ev, best_resnorm


def truth_init_particles(log10_true: np.ndarray, n: int = N_PARTICLES) -> np.ndarray:
    """Tight cloud around true parameters (σ = TRUTH_SIGMA in log10)."""
    noise = RNG.standard_normal((n, 5)) * TRUTH_SIGMA
    return np.clip(log10_true[np.newaxis, :] + noise, BOUNDS_LOW, BOUNDS_HIGH)


def blind_init_particles(n: int = N_PARTICLES) -> np.ndarray:
    """Uniform random in parameter bounds — no privileged initialization."""
    return RNG.uniform(BOUNDS_LOW, BOUNDS_HIGH, size=(n, 5))


# ---- Derived quantities ----

def derived(log10_params: np.ndarray):
    """TER (Ω), TEC (F), tau_big (s), tau_small (s) from log10 [Ra,Rb,Ca,Cb,Rsh]."""
    p = 10.0 ** log10_params
    Ra, Rb, Ca, Cb, Rsh = p
    TER      = (Rsh * (Ra + Rb)) / (Rsh + Ra + Rb + 1e-20)
    TEC      = (Ca * Cb) / (Ca + Cb + 1e-20)
    tau_a    = Ra * Ca
    tau_b    = Rb * Cb
    tau_big  = max(tau_a, tau_b)
    tau_small = min(tau_a, tau_b)
    return {'TER': TER, 'TEC': TEC, 'tau_big': tau_big, 'tau_small': tau_small,
            'Ra': Ra, 'Rb': Rb, 'Ca': Ca, 'Cb': Cb, 'Rsh': Rsh}


def cloud_median_derived(particles: np.ndarray):
    """Derived quantities from cloud median in log10 space."""
    med = np.median(particles, axis=0)
    return derived(med)


# ---- Load and filter data ----

print("Loading data…")
df = pd.read_csv(DATA_PATH)

# Filter to samples whose true parameters fall inside the SMC/ECM search bounds
log10_true_all = np.column_stack([
    np.log10(df['Ra'].values), np.log10(df['Rb'].values),
    np.log10(df['Ca'].values), np.log10(df['Cb'].values),
    np.log10(df['Rsh'].values),
])
in_bounds = np.all(
    (log10_true_all >= BOUNDS_LOW) & (log10_true_all <= BOUNDS_HIGH), axis=1
)
df_valid = df[in_bounds].reset_index(drop=True)
log10_true_valid = log10_true_all[in_bounds]

print(f"  {in_bounds.sum()} / {len(df)} spectra within bounds")
print(f"  Distributions: {df_valid['distribution'].value_counts().to_dict()}")

# Stratified sample: draw evenly from each distribution type
n_per_dist = max(1, N_TEST // df_valid['distribution'].nunique())
selected = (
    df_valid
    .groupby('distribution', group_keys=False)
    .apply(lambda g: g.sample(min(len(g), n_per_dist), random_state=SEED))
)
selected = selected.sample(min(N_TEST, len(selected)), random_state=SEED).reset_index(drop=True)
log10_true = log10_true_valid[selected.index.values]  # (N_TEST, 5) in log10

# Extract impedance arrays
Z_real_cols = [f'Z_real_{i}' for i in range(N_FREQ)]
Z_imag_cols = [f'Z_imag_{i}' for i in range(N_FREQ)]
Z_r_all = selected[Z_real_cols].values   # (N_TEST, N_FREQ)
Z_i_all = selected[Z_imag_cols].values

print(f"  Running on {len(selected)} spectra")

# ---- Main evaluation loop ----

results = {
    'ecm_resnorm': [], 'smc_truth_resnorm': [], 'smc_blind_resnorm': [],
    'ecm_log10': [], 'smc_truth_med_log10': [], 'smc_blind_med_log10': [],
    'true_log10': [],
    'distribution': [],
}

t0 = time.perf_counter()
for idx in range(len(selected)):
    Z_r = Z_r_all[idx]
    Z_i = Z_i_all[idx]
    true_x = log10_true[idx]

    # -- A: ECM cold start --
    ecm_x, ecm_rn = ecm_cold(Z_r, Z_i)
    if ecm_x is None:
        ecm_x  = BOUNDS_MID.copy()
        ecm_rn = compute_resnorm(ecm_x, Z_r, Z_i)

    # -- B: SMC truth-initialized --
    init_truth = truth_init_particles(true_x)
    post_truth, _, truth_rn = run_smc(init_truth, Z_r, Z_i)

    # -- C: SMC blind --
    init_blind = blind_init_particles()
    post_blind, _, blind_rn = run_smc(init_blind, Z_r, Z_i)

    results['ecm_resnorm'].append(ecm_rn)
    results['smc_truth_resnorm'].append(truth_rn)
    results['smc_blind_resnorm'].append(blind_rn)
    results['ecm_log10'].append(ecm_x)
    results['smc_truth_med_log10'].append(np.median(post_truth, axis=0))
    results['smc_blind_med_log10'].append(np.median(post_blind, axis=0))
    results['true_log10'].append(true_x)
    results['distribution'].append(selected['distribution'].iloc[idx])

    if (idx + 1) % 20 == 0 or idx == len(selected) - 1:
        elapsed = time.perf_counter() - t0
        rate    = (idx + 1) / elapsed
        remain  = (len(selected) - idx - 1) / max(rate, 1e-6)
        print(f"  {idx+1}/{len(selected)}  "
              f"ECM={ecm_rn:.4f}  TruthSMC={truth_rn:.4f}  BlindSMC={blind_rn:.4f}  "
              f"[{elapsed:.0f}s elapsed, ~{remain:.0f}s remain]")

print(f"Done in {time.perf_counter() - t0:.1f}s")

# ---- Convert to arrays ----
ecm_rn   = np.array(results['ecm_resnorm'])
truth_rn = np.array(results['smc_truth_resnorm'])
blind_rn = np.array(results['smc_blind_resnorm'])

ecm_x   = np.array(results['ecm_log10'])          # (N, 5) log10
truth_x  = np.array(results['smc_truth_med_log10']) # (N, 5) log10
blind_x  = np.array(results['smc_blind_med_log10']) # (N, 5) log10
true_x   = np.array(results['true_log10'])          # (N, 5) log10
dists    = np.array(results['distribution'])

# Physical parameter arrays
ecm_phys   = 10.0 ** ecm_x
truth_phys = 10.0 ** truth_x
blind_phys = 10.0 ** blind_x
true_phys  = 10.0 ** true_x

# Derived identifiable quantities from each estimate vs truth
def batch_derived(log10_arr):
    """(N, 5) log10 → (N,) TER, tau_big, TEC in SI."""
    p = 10.0 ** log10_arr
    Ra, Rb, Ca, Cb, Rsh = p[:,0], p[:,1], p[:,2], p[:,3], p[:,4]
    TER = (Rsh*(Ra+Rb)) / (Rsh+Ra+Rb+1e-20)
    TEC = (Ca*Cb) / (Ca+Cb+1e-20)
    tau_a = Ra*Ca; tau_b = Rb*Cb
    tau_big  = np.maximum(tau_a, tau_b)
    tau_small = np.minimum(tau_a, tau_b)
    return TER, tau_big, tau_small, TEC

TER_true,  tb_true,  ts_true,  TEC_true  = batch_derived(true_x)
TER_ecm,   tb_ecm,   ts_ecm,   TEC_ecm   = batch_derived(ecm_x)
TER_truth, tb_truth, ts_truth, TEC_truth  = batch_derived(truth_x)
TER_blind, tb_blind, ts_blind, TEC_blind  = batch_derived(blind_x)

# Symmetric log10 error (uses min assignment to handle Ra/Rb, Ca/Cb swaps)
def sym_log10_err(pred: np.ndarray, true: np.ndarray, idx_a: int, idx_b: int) -> np.ndarray:
    """Min-distance assignment: which label assignment gives lower |log10(pred/true)|?"""
    err_direct = np.abs(pred[:, idx_a] - true[:, idx_a]) + np.abs(pred[:, idx_b] - true[:, idx_b])
    err_swap   = np.abs(pred[:, idx_a] - true[:, idx_b]) + np.abs(pred[:, idx_b] - true[:, idx_a])
    return np.minimum(err_direct, err_swap) / 2  # per-parameter average

Ra_err_ecm   = sym_log10_err(ecm_x,   true_x, 0, 1)
Ra_err_truth = sym_log10_err(truth_x, true_x, 0, 1)
Ra_err_blind = sym_log10_err(blind_x, true_x, 0, 1)

Ca_err_ecm   = sym_log10_err(ecm_x,   true_x, 2, 3)
Ca_err_truth = sym_log10_err(truth_x, true_x, 2, 3)
Ca_err_blind = sym_log10_err(blind_x, true_x, 2, 3)

Rsh_err_ecm   = np.abs(ecm_x[:,4]   - true_x[:,4])
Rsh_err_truth = np.abs(truth_x[:,4] - true_x[:,4])
Rsh_err_blind = np.abs(blind_x[:,4] - true_x[:,4])

# Swap detection: does the assigned Ra > Rb match truth Ra > Rb?
true_Ra_bigger  = true_phys[:,0]  > true_phys[:,1]
ecm_Ra_bigger   = ecm_phys[:,0]   > ecm_phys[:,1]
truth_Ra_bigger = truth_phys[:,0] > truth_phys[:,1]
blind_Ra_bigger = blind_phys[:,0] > blind_phys[:,1]
swap_rate_ecm   = np.mean(ecm_Ra_bigger   != true_Ra_bigger)
swap_rate_truth = np.mean(truth_Ra_bigger != true_Ra_bigger)
swap_rate_blind = np.mean(blind_Ra_bigger != true_Ra_bigger)

# ---- Figure ----

FIG_W, FIG_H = 20, 15
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor='#0d1117')
gs  = gridspec.GridSpec(3, 4, figure=fig,
                        hspace=0.48, wspace=0.38,
                        left=0.06, right=0.97, top=0.93, bottom=0.07)

COND_COLORS = {
    'ECM':       '#94a3b8',  # slate
    'TruthSMC':  '#34d399',  # emerald
    'BlindSMC':  '#f59e0b',  # amber
    'Truth':     '#f87171',  # red/truth
}
AX_BG    = '#0f1117'
GRID_COL = '#1e2430'
TEXT_COL = '#cbd5e1'
LABEL_KW = dict(color=TEXT_COL, fontsize=9)
TITLE_KW = dict(color='#94a3b8', fontsize=9, fontweight='bold', pad=6)


def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(AX_BG)
    for spine in ax.spines.values():
        spine.set_color('#2d3748')
    ax.tick_params(colors='#64748b', labelsize=7.5)
    ax.xaxis.label.set_color('#94a3b8')
    ax.yaxis.label.set_color('#94a3b8')
    if title:  ax.set_title(title, **TITLE_KW)
    if xlabel: ax.set_xlabel(xlabel, **LABEL_KW)
    if ylabel: ax.set_ylabel(ylabel, **LABEL_KW)
    ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.6)


# ══════════════════════════════════════════════
# ROW 0: RESNORM LANDSCAPE
# ══════════════════════════════════════════════

# [0,0] Violin: resnorm distributions
ax00 = fig.add_subplot(gs[0, 0])
style_ax(ax00, title='Resnorm distributions', xlabel='Condition', ylabel='Resnorm (log₁₀)')
vdata = [np.log10(ecm_rn + 1e-12), np.log10(truth_rn + 1e-12), np.log10(blind_rn + 1e-12)]
vlabels = ['ECM\ncold', 'SMC\ntruth-init', 'SMC\nblind']
vp = ax00.violinplot(vdata, positions=[1, 2, 3], showmedians=True, showextrema=True)
colors_v = [COND_COLORS['ECM'], COND_COLORS['TruthSMC'], COND_COLORS['BlindSMC']]
for body, col in zip(vp['bodies'], colors_v):
    body.set_facecolor(col)
    body.set_alpha(0.55)
for part in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
    if part in vp:
        vp[part].set_color('#e2e8f0')
        vp[part].set_linewidth(1.2)
ax00.set_xticks([1, 2, 3])
ax00.set_xticklabels(vlabels, fontsize=7.5, color='#94a3b8')
# Medians annotation
for pos, data, col in zip([1,2,3], vdata, colors_v):
    med = np.median(data)
    ax00.text(pos, med, f' {10**med:.4f}', va='center', ha='left', fontsize=6.5, color=col)

# [0,1] Scatter: ECM vs truth-SMC resnorm
ax01 = fig.add_subplot(gs[0, 1])
style_ax(ax01, title='ECM vs truth-init SMC resnorm', xlabel='ECM resnorm', ylabel='Truth-SMC resnorm')
lims = [min(ecm_rn.min(), truth_rn.min()) * 0.8, max(ecm_rn.max(), truth_rn.max()) * 1.2]
ax01.set_xscale('log'); ax01.set_yscale('log')
ax01.plot(lims, lims, '--', color='#475569', linewidth=1, label='equal', zorder=1)
# Region coloring
ax01.fill_between(lims, lims, [lims[1]]*2, alpha=0.04, color=COND_COLORS['TruthSMC'])
ax01.fill_between(lims, [lims[0]]*2, lims, alpha=0.04, color=COND_COLORS['ECM'])
sc = ax01.scatter(ecm_rn, truth_rn, c=np.log10(TER_true / 1000),
                  cmap='plasma', alpha=0.55, s=12, zorder=2, rasterized=True)
plt.colorbar(sc, ax=ax01, label='log₁₀(TER/kΩ)', pad=0.01).ax.tick_params(labelsize=7)
ax01.set_xlim(lims); ax01.set_ylim(lims)
n_truth_wins = np.sum(truth_rn < ecm_rn)
ax01.text(0.05, 0.93, f'Truth-SMC better: {n_truth_wins}/{len(ecm_rn)} ({100*n_truth_wins/len(ecm_rn):.0f}%)',
          transform=ax01.transAxes, fontsize=7.5, color=COND_COLORS['TruthSMC'])
ax01.text(0.05, 0.06, f'ECM better: {len(ecm_rn)-n_truth_wins}/{len(ecm_rn)} ({100*(1-n_truth_wins/len(ecm_rn)):.0f}%)',
          transform=ax01.transAxes, fontsize=7.5, color=COND_COLORS['ECM'])

# [0,2] Histogram: fractional improvement (ECM - TruthSMC) / ECM
ax02 = fig.add_subplot(gs[0, 2])
style_ax(ax02, title='Resnorm improvement: truth-init vs ECM',
         xlabel='(ECM − TruthSMC) / ECM', ylabel='Count')
improvement = (ecm_rn - truth_rn) / (ecm_rn + 1e-12)
ax02.hist(improvement, bins=40, color=COND_COLORS['TruthSMC'], alpha=0.7, edgecolor='none')
ax02.axvline(0, color='#475569', linewidth=1, linestyle='--')
ax02.axvline(np.median(improvement), color=COND_COLORS['TruthSMC'], linewidth=1.5,
             linestyle=':', label=f'median {np.median(improvement):.3f}')
ax02.legend(fontsize=7, labelcolor=TEXT_COL, facecolor='#1a2030', edgecolor='none')
ax02.text(0.05, 0.88,
          f'mean  {improvement.mean():+.3f}\n'
          f'std   {improvement.std():.3f}\n'
          f'>0 (truth better): {100*np.mean(improvement>0):.0f}%',
          transform=ax02.transAxes, fontsize=7.5, color=TEXT_COL,
          va='top', family='monospace')

# [0,3] Cumulative: % of spectra where truth-init beats ECM by threshold
ax03 = fig.add_subplot(gs[0, 3])
style_ax(ax03, title='Cumulative: truth-init win rate by threshold',
         xlabel='Min improvement required (× ECM resnorm)', ylabel='% spectra where truth-init wins')
thresholds = np.linspace(0, 0.5, 200)
win_pct_truth = [100 * np.mean(improvement > t) for t in thresholds]
win_pct_blind = [100 * np.mean((ecm_rn - blind_rn) / (ecm_rn + 1e-12) > t) for t in thresholds]
ax03.plot(thresholds, win_pct_truth, color=COND_COLORS['TruthSMC'], linewidth=1.8, label='SMC truth-init')
ax03.plot(thresholds, win_pct_blind, color=COND_COLORS['BlindSMC'],  linewidth=1.8, label='SMC blind')
ax03.axhline(50, color='#475569', linewidth=0.8, linestyle='--')
ax03.set_ylim(0, 105)
ax03.legend(fontsize=7.5, labelcolor=TEXT_COL, facecolor='#1a2030', edgecolor='none')

# ══════════════════════════════════════════════
# ROW 1: DEGENERATE PARAMETER MANIFOLD
# ══════════════════════════════════════════════

rn_color_ecm   = np.log10(np.clip(ecm_rn, 1e-6, None))
rn_color_truth = np.log10(np.clip(truth_rn, 1e-6, None))
rn_vmin = min(rn_color_ecm.min(), rn_color_truth.min())
rn_vmax = max(rn_color_ecm.max(), rn_color_truth.max())

# [1,0] Ra vs Rb manifold — ECM solutions
ax10 = fig.add_subplot(gs[1, 0])
style_ax(ax10, title='Ra/Rb manifold — ECM solutions',
         xlabel='Ra (kΩ)', ylabel='Rb (kΩ)')
ax10.set_xscale('log'); ax10.set_yscale('log')
# True values as tiny red dots
ax10.scatter(true_phys[:,0]/1000, true_phys[:,1]/1000, c='#f87171', s=5, alpha=0.4, zorder=3, label='truth')
# ECM solutions
sc10 = ax10.scatter(ecm_phys[:,0]/1000, ecm_phys[:,1]/1000,
                    c=rn_color_ecm, cmap='magma', vmin=rn_vmin, vmax=rn_vmax,
                    s=14, alpha=0.65, zorder=2, label='ECM')
# Diagonal — same Ra=Rb (full degeneracy)
xy_lim = [0.04, 30]
ax10.plot(xy_lim, xy_lim, '--', color='#334155', linewidth=1, label='Ra=Rb (sym)')
ax10.set_xlim(xy_lim); ax10.set_ylim(xy_lim)
ax10.legend(fontsize=6.5, labelcolor=TEXT_COL, facecolor='#1a2030', edgecolor='none')
ax10.text(0.04, 0.93, f'Ra/Rb swap rate: {swap_rate_ecm:.0%}',
          transform=ax10.transAxes, fontsize=7.5, color=COND_COLORS['ECM'])

# [1,1] Ra vs Rb manifold — truth-init SMC posterior medians
ax11 = fig.add_subplot(gs[1, 1])
style_ax(ax11, title='Ra/Rb manifold — truth-init SMC',
         xlabel='Ra (kΩ)', ylabel='Rb (kΩ)')
ax11.set_xscale('log'); ax11.set_yscale('log')
ax11.scatter(true_phys[:,0]/1000, true_phys[:,1]/1000, c='#f87171', s=5, alpha=0.4, zorder=3, label='truth')
sc11 = ax11.scatter(truth_phys[:,0]/1000, truth_phys[:,1]/1000,
                    c=rn_color_truth, cmap='magma', vmin=rn_vmin, vmax=rn_vmax,
                    s=14, alpha=0.65, zorder=2, label='SMC truth-init')
ax11.plot(xy_lim, xy_lim, '--', color='#334155', linewidth=1)
ax11.set_xlim(xy_lim); ax11.set_ylim(xy_lim)
cbar11 = plt.colorbar(sc11, ax=ax11, label='log₁₀(resnorm)', pad=0.01)
cbar11.ax.tick_params(labelsize=7)
ax11.legend(fontsize=6.5, labelcolor=TEXT_COL, facecolor='#1a2030', edgecolor='none')
ax11.text(0.04, 0.93, f'Ra/Rb swap rate: {swap_rate_truth:.0%}',
          transform=ax11.transAxes, fontsize=7.5, color=COND_COLORS['TruthSMC'])

# [1,2] Ca/Cb ratio: pred vs truth — both conditions
ax12 = fig.add_subplot(gs[1, 2])
style_ax(ax12, title='Ca/Cb ratio: predicted vs truth',
         xlabel='Ca_true / Cb_true', ylabel='Ca_pred / Cb_pred')
ratio_true = true_phys[:,2] / (true_phys[:,3] + 1e-30)
ratio_ecm  = ecm_phys[:,2]  / (ecm_phys[:,3]  + 1e-30)
ratio_truth_smc = truth_phys[:,2] / (truth_phys[:,3] + 1e-30)
ax12.set_xscale('log'); ax12.set_yscale('log')
r_lim = [0.08, 12]
ax12.plot(r_lim, r_lim, '--', color='#475569', linewidth=1, label='perfect')
ax12.scatter(ratio_true, ratio_ecm,        c=COND_COLORS['ECM'],      alpha=0.4, s=10, label='ECM',        rasterized=True)
ax12.scatter(ratio_true, ratio_truth_smc,  c=COND_COLORS['TruthSMC'], alpha=0.4, s=10, label='Truth-SMC',  rasterized=True)
ax12.set_xlim(r_lim); ax12.set_ylim(r_lim)
ax12.legend(fontsize=7, labelcolor=TEXT_COL, facecolor='#1a2030', edgecolor='none')

# [1,3] Per-parameter symmetric log10 error comparison
ax13 = fig.add_subplot(gs[1, 3])
style_ax(ax13, title='Symmetric log₁₀ error by parameter',
         xlabel='Parameter', ylabel='Median |log₁₀(pred/true)|')
param_labels = ['R (Ra/Rb)', 'C (Ca/Cb)', 'Rsh']
errs_ecm   = [np.median(Ra_err_ecm),   np.median(Ca_err_ecm),   np.median(Rsh_err_ecm)]
errs_truth = [np.median(Ra_err_truth), np.median(Ca_err_truth), np.median(Rsh_err_truth)]
errs_blind = [np.median(Ra_err_blind), np.median(Ca_err_blind), np.median(Rsh_err_blind)]

x_pos = np.arange(len(param_labels))
bw = 0.25
ax13.bar(x_pos - bw,  errs_ecm,   bw, color=COND_COLORS['ECM'],      alpha=0.8, label='ECM')
ax13.bar(x_pos,       errs_truth, bw, color=COND_COLORS['TruthSMC'], alpha=0.8, label='Truth-SMC')
ax13.bar(x_pos + bw,  errs_blind, bw, color=COND_COLORS['BlindSMC'], alpha=0.8, label='Blind-SMC')
ax13.set_xticks(x_pos)
ax13.set_xticklabels(param_labels, fontsize=7.5, color='#94a3b8')
ax13.legend(fontsize=7, labelcolor=TEXT_COL, facecolor='#1a2030', edgecolor='none')
# Value labels
for xi, (a, b, c) in enumerate(zip(errs_ecm, errs_truth, errs_blind)):
    for offset, val, col in zip([-bw, 0, bw], [a, b, c],
                                 [COND_COLORS['ECM'], COND_COLORS['TruthSMC'], COND_COLORS['BlindSMC']]):
        ax13.text(xi + offset, val + 0.002, f'{val:.3f}', ha='center', va='bottom', fontsize=6, color=col)

# ══════════════════════════════════════════════
# ROW 2: IDENTIFIABLE PARAMETERS + PHASE PORTRAIT
# ══════════════════════════════════════════════

def scatter_pred_true(ax, true_v, ecm_v, truth_v, title, xlabel, ylabel, scale=1.0):
    style_ax(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    ax.set_xscale('log'); ax.set_yscale('log')
    vals = np.concatenate([true_v, ecm_v, truth_v]) * scale
    lim_lo = np.percentile(vals[vals > 0], 2) * 0.8
    lim_hi = np.percentile(vals[vals > 0], 98) * 1.2
    lims_v = [lim_lo, lim_hi]
    ax.plot(lims_v, lims_v, '--', color='#475569', linewidth=1)
    ax.scatter(true_v * scale, ecm_v * scale,   c=COND_COLORS['ECM'],      alpha=0.4, s=10, label='ECM',       rasterized=True)
    ax.scatter(true_v * scale, truth_v * scale, c=COND_COLORS['TruthSMC'], alpha=0.4, s=10, label='Truth-SMC', rasterized=True)
    ax.set_xlim(lims_v); ax.set_ylim(lims_v)
    ax.legend(fontsize=6.5, labelcolor=TEXT_COL, facecolor='#1a2030', edgecolor='none')

# [2,0] TER recovery
ax20 = fig.add_subplot(gs[2, 0])
scatter_pred_true(ax20, TER_true, TER_ecm, TER_truth,
                  title='TER recovery (identifiable)',
                  xlabel='TER_true (kΩ)', ylabel='TER_pred (kΩ)', scale=1e-3)

# [2,1] tau_big recovery
ax21 = fig.add_subplot(gs[2, 1])
scatter_pred_true(ax21, tb_true, tb_ecm, tb_truth,
                  title='τ_big recovery (identifiable)',
                  xlabel='τ_big_true (s)', ylabel='τ_big_pred (s)', scale=1.0)

# [2,2] Phase portrait: where solutions cluster in (TER, tau_big) space
ax22 = fig.add_subplot(gs[2, 2])
style_ax(ax22, title='Phase portrait: solution clusters (TER vs τ_big)',
         xlabel='TER (kΩ)', ylabel='τ_big (s)')
ax22.set_xscale('log'); ax22.set_yscale('log')

# Draw lines connecting truth → ECM and truth → TruthSMC for a sample of spectra
sample_idx = RNG.choice(len(TER_true), size=min(40, len(TER_true)), replace=False)
for i in sample_idx:
    # truth → ECM line
    ax22.plot([TER_true[i]/1000, TER_ecm[i]/1000],
              [tb_true[i],       tb_ecm[i]],
              color=COND_COLORS['ECM'], alpha=0.2, linewidth=0.7)
    # truth → TruthSMC line
    ax22.plot([TER_true[i]/1000, TER_truth[i]/1000],
              [tb_true[i],       tb_truth[i]],
              color=COND_COLORS['TruthSMC'], alpha=0.2, linewidth=0.7)

ax22.scatter(TER_ecm[sample_idx]/1000,   tb_ecm[sample_idx],
             c=COND_COLORS['ECM'],      s=18, alpha=0.7, zorder=4, label='ECM', rasterized=True)
ax22.scatter(TER_truth[sample_idx]/1000, tb_truth[sample_idx],
             c=COND_COLORS['TruthSMC'], s=18, alpha=0.7, zorder=4, label='Truth-SMC', rasterized=True)
ax22.scatter(TER_true[sample_idx]/1000,  tb_true[sample_idx],
             c=COND_COLORS['Truth'], s=24, marker='*', alpha=0.9, zorder=5, label='Truth')
ax22.legend(fontsize=7, labelcolor=TEXT_COL, facecolor='#1a2030', edgecolor='none')

# [2,3] Summary statistics table
ax23 = fig.add_subplot(gs[2, 3])
ax23.set_facecolor('#0f1117')
ax23.axis('off')

TER_err_ecm   = np.abs(np.log10(TER_ecm   + 1e-20) - np.log10(TER_true + 1e-20))
TER_err_truth = np.abs(np.log10(TER_truth  + 1e-20) - np.log10(TER_true + 1e-20))
TER_err_blind = np.abs(np.log10(TER_blind  + 1e-20) - np.log10(TER_true + 1e-20))
tb_err_ecm    = np.abs(np.log10(tb_ecm    + 1e-20) - np.log10(tb_true  + 1e-20))
tb_err_truth  = np.abs(np.log10(tb_truth   + 1e-20) - np.log10(tb_true  + 1e-20))

table_rows = [
    ['Metric',           'ECM',
     'SMC\ntruth-init',  'SMC\nblind'],
    ['resnorm median',
     f'{np.median(ecm_rn):.4f}',
     f'{np.median(truth_rn):.4f}',
     f'{np.median(blind_rn):.4f}'],
    ['resnorm mean',
     f'{ecm_rn.mean():.4f}',
     f'{truth_rn.mean():.4f}',
     f'{blind_rn.mean():.4f}'],
    ['R err (log10)',
     f'{np.median(Ra_err_ecm):.3f}',
     f'{np.median(Ra_err_truth):.3f}',
     f'{np.median(Ra_err_blind):.3f}'],
    ['C err (log10)',
     f'{np.median(Ca_err_ecm):.3f}',
     f'{np.median(Ca_err_truth):.3f}',
     f'{np.median(Ca_err_blind):.3f}'],
    ['Rsh err (log10)',
     f'{np.median(Rsh_err_ecm):.3f}',
     f'{np.median(Rsh_err_truth):.3f}',
     f'{np.median(Rsh_err_blind):.3f}'],
    ['TER err (log10)',
     f'{np.median(TER_err_ecm):.3f}',
     f'{np.median(TER_err_truth):.3f}',
     f'{np.median(TER_err_blind):.3f}'],
    ['τ_big err (log10)',
     f'{np.median(tb_err_ecm):.3f}',
     f'{np.median(tb_err_truth):.3f}',
     '—'],
    ['Ra/Rb swap rate',
     f'{swap_rate_ecm:.1%}',
     f'{swap_rate_truth:.1%}',
     f'{swap_rate_blind:.1%}'],
    ['N spectra', f'{len(ecm_rn)}', f'{len(truth_rn)}', f'{len(blind_rn)}'],
]

col_colors_header = [COND_COLORS['ECM'], COND_COLORS['ECM'],
                     COND_COLORS['TruthSMC'], COND_COLORS['BlindSMC']]
y_start = 0.97
line_h  = 0.09
for ri, row in enumerate(table_rows):
    is_header = (ri == 0)
    for ci, cell in enumerate(row):
        col = '#e2e8f0' if is_header else (
            col_colors_header[ci] if ci > 0 else '#94a3b8'
        )
        weight = 'bold' if is_header else 'normal'
        x_pos_cell = 0.01 + ci * 0.25
        ax23.text(x_pos_cell, y_start - ri * line_h, cell,
                  transform=ax23.transAxes,
                  fontsize=7.5, color=col, fontweight=weight,
                  va='top', family='monospace')

ax23.set_title('Summary statistics', **TITLE_KW)

# ---- Main title ----
fig.suptitle(
    f'Initialization vs manifold: does starting at truth help?   '
    f'[N={len(selected)} spectra, ECM={N_ECM_STARTS} starts, '
    f'SMC N={N_PARTICLES} particles]',
    color='#e2e8f0', fontsize=11, fontweight='bold', y=0.97
)

# ---- Save ----
out_path = ROOT / 'comparison_truth_init.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nSaved: {out_path}")

# ---- Terminal summary ----
print("\n" + "="*60)
print("KEY RESULT: Does initialization determine the outcome?")
print("="*60)
print(f"  Resnorm  median  ECM={np.median(ecm_rn):.4f}  TruthSMC={np.median(truth_rn):.4f}  BlindSMC={np.median(blind_rn):.4f}")
print(f"  Truth-init wins on resnorm: {100*np.mean(truth_rn < ecm_rn):.0f}% of spectra")
print(f"  R (Ra/Rb) sym log10 err: ECM={np.median(Ra_err_ecm):.3f}  TruthSMC={np.median(Ra_err_truth):.3f}")
print(f"  C (Ca/Cb) sym log10 err: ECM={np.median(Ca_err_ecm):.3f}  TruthSMC={np.median(Ca_err_truth):.3f}")
print(f"  Ra/Rb swap:              ECM={swap_rate_ecm:.0%}  TruthSMC={swap_rate_truth:.0%}")
print(f"  TER log10 err:           ECM={np.median(TER_err_ecm):.3f}  TruthSMC={np.median(TER_err_truth):.3f}")
print()
if np.median(truth_rn) < np.median(ecm_rn) * 0.9:
    print("CONCLUSION: Truth-init gives meaningfully lower resnorm -> ECM is stuck in local minima.")
    print("  Implication: better initialization strategy (phase portrait context) would help.")
elif abs(np.median(truth_rn) - np.median(ecm_rn)) < np.median(ecm_rn) * 0.05:
    print("CONCLUSION: Resnorms are equivalent -> the manifold is the problem, not initialization.")
    print("  Implication: the degenerate manifold has the same objective value as truth.")
    print("  The full trajectory (phase portrait) is needed to resolve the degeneracy.")
else:
    print("CONCLUSION: Mixed — truth-init helps somewhat but not decisively.")
print("="*60)
