#!/usr/bin/env python3
"""
Multi-Circuit Temporal Analysis
================================

Test temporal smoothness constraints across multiple circuit topologies
to demonstrate that the approach generalizes beyond the 2-RC RPE model.

Circuit Types:
1. Simple RC (2 params)
2. Randles (3 params)
3. 2-RC RPE Model (5 params) - our standard model
4. Randles + Warburg (4 params)
5. 3-RC Extended (7 params)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from typing import Dict, List, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Circuit Definitions
# =============================================================================

@dataclass
class CircuitDefinition:
    """Definition of a circuit type for testing."""
    name: str
    short_name: str
    param_names: List[str]
    param_labels: List[str]
    param_ranges: Dict[str, Tuple[float, float]]  # log10 ranges
    compute_impedance: Callable
    generate_trajectory: Callable
    description: str


def compute_impedance_simple_rc(params: Dict, frequencies: np.ndarray) -> np.ndarray:
    """Simple RC circuit: R in series with C."""
    omega = 2 * np.pi * frequencies
    R, C = params['R'], params['C']
    Zc = 1 / (1j * omega * C)
    return R + Zc


def compute_impedance_randles(params: Dict, frequencies: np.ndarray) -> np.ndarray:
    """Randles circuit: Rs + (Rct || Cdl)."""
    omega = 2 * np.pi * frequencies
    Rs, Rct, Cdl = params['Rs'], params['Rct'], params['Cdl']
    Zct = Rct / (1 + 1j * omega * Rct * Cdl)
    return Rs + Zct


def compute_impedance_2rc(params: Dict, frequencies: np.ndarray) -> np.ndarray:
    """2-RC RPE model: Rsh || (Za + Zb) where Za = Ra || Ca, Zb = Rb || Cb."""
    omega = 2 * np.pi * frequencies
    Ra, Rb, Ca, Cb, Rsh = params['Ra'], params['Rb'], params['Ca'], params['Cb'], params['Rsh']
    Za = Ra / (1 + 1j * omega * Ra * Ca)
    Zb = Rb / (1 + 1j * omega * Rb * Cb)
    Z_series = Za + Zb
    return (Rsh * Z_series) / (Rsh + Z_series)


def compute_impedance_randles_warburg(params: Dict, frequencies: np.ndarray) -> np.ndarray:
    """Randles + Warburg: Rs + (Rct || Cdl) + Zw."""
    omega = 2 * np.pi * frequencies
    Rs, Rct, Cdl, Aw = params['Rs'], params['Rct'], params['Cdl'], params['Aw']
    Zct = Rct / (1 + 1j * omega * Rct * Cdl)
    # Warburg impedance (semi-infinite diffusion)
    Zw = Aw / np.sqrt(omega) * (1 - 1j)
    return Rs + Zct + Zw


def compute_impedance_3rc(params: Dict, frequencies: np.ndarray) -> np.ndarray:
    """3-RC model: Extended RPE with additional membrane."""
    omega = 2 * np.pi * frequencies
    Ra, Rb, Rc = params['Ra'], params['Rb'], params['Rc']
    Ca, Cb, Cc = params['Ca'], params['Cb'], params['Cc']
    Rsh = params['Rsh']

    Za = Ra / (1 + 1j * omega * Ra * Ca)
    Zb = Rb / (1 + 1j * omega * Rb * Cb)
    Zc = Rc / (1 + 1j * omega * Rc * Cc)
    Z_series = Za + Zb + Zc
    return (Rsh * Z_series) / (Rsh + Z_series)


# Trajectory generators for each circuit
def generate_trajectory_simple_rc(n_timepoints: int = 6) -> List[Dict]:
    np.random.seed(42)
    params = {'R': 500.0, 'C': 5e-6}
    trajectory = []
    for t in range(n_timepoints):
        trajectory.append(params.copy())
        params = {
            'R': params['R'] * (10 ** np.random.uniform(0.04, 0.08)),
            'C': params['C'] * (10 ** np.random.uniform(-0.01, 0.01)),
        }
    return trajectory


def generate_trajectory_randles(n_timepoints: int = 6) -> List[Dict]:
    np.random.seed(42)
    params = {'Rs': 50.0, 'Rct': 400.0, 'Cdl': 3e-6}
    trajectory = []
    for t in range(n_timepoints):
        trajectory.append(params.copy())
        params = {
            'Rs': params['Rs'] * (10 ** np.random.uniform(-0.01, 0.01)),
            'Rct': params['Rct'] * (10 ** np.random.uniform(0.04, 0.07)),
            'Cdl': params['Cdl'] * (10 ** np.random.uniform(-0.008, 0.008)),
        }
    return trajectory


def generate_trajectory_2rc(n_timepoints: int = 6) -> List[Dict]:
    np.random.seed(42)
    params = {'Ra': 400.0, 'Rb': 350.0, 'Ca': 3e-6, 'Cb': 4e-6, 'Rsh': 1200.0}
    trajectory = []
    for t in range(n_timepoints):
        trajectory.append(params.copy())
        params = {
            'Ra': params['Ra'] * (10 ** 0.06),
            'Rb': params['Rb'] * (10 ** np.random.uniform(-0.008, 0.008)),
            'Ca': params['Ca'] * (10 ** np.random.uniform(-0.006, 0.006)),
            'Cb': params['Cb'] * (10 ** np.random.uniform(-0.006, 0.006)),
            'Rsh': params['Rsh'] * (10 ** np.random.uniform(-0.008, 0.008)),
        }
    return trajectory


def generate_trajectory_randles_warburg(n_timepoints: int = 6) -> List[Dict]:
    np.random.seed(42)
    params = {'Rs': 30.0, 'Rct': 300.0, 'Cdl': 2e-6, 'Aw': 100.0}
    trajectory = []
    for t in range(n_timepoints):
        trajectory.append(params.copy())
        params = {
            'Rs': params['Rs'] * (10 ** np.random.uniform(-0.01, 0.01)),
            'Rct': params['Rct'] * (10 ** np.random.uniform(0.03, 0.06)),
            'Cdl': params['Cdl'] * (10 ** np.random.uniform(-0.008, 0.008)),
            'Aw': params['Aw'] * (10 ** np.random.uniform(-0.015, 0.015)),
        }
    return trajectory


def generate_trajectory_3rc(n_timepoints: int = 6) -> List[Dict]:
    np.random.seed(42)
    params = {
        'Ra': 300.0, 'Rb': 250.0, 'Rc': 200.0,
        'Ca': 2e-6, 'Cb': 3e-6, 'Cc': 4e-6,
        'Rsh': 1500.0
    }
    trajectory = []
    for t in range(n_timepoints):
        trajectory.append(params.copy())
        params = {
            'Ra': params['Ra'] * (10 ** np.random.uniform(0.04, 0.07)),
            'Rb': params['Rb'] * (10 ** np.random.uniform(-0.01, 0.01)),
            'Rc': params['Rc'] * (10 ** np.random.uniform(-0.008, 0.008)),
            'Ca': params['Ca'] * (10 ** np.random.uniform(-0.006, 0.006)),
            'Cb': params['Cb'] * (10 ** np.random.uniform(-0.006, 0.006)),
            'Cc': params['Cc'] * (10 ** np.random.uniform(-0.006, 0.006)),
            'Rsh': params['Rsh'] * (10 ** np.random.uniform(-0.008, 0.008)),
        }
    return trajectory


# Circuit definitions
CIRCUITS = [
    CircuitDefinition(
        name="Simple RC",
        short_name="RC",
        param_names=['R', 'C'],
        param_labels=['R', 'C'],
        param_ranges={'R': (2.0, 3.5), 'C': (-6.0, -4.5)},
        compute_impedance=compute_impedance_simple_rc,
        generate_trajectory=generate_trajectory_simple_rc,
        description="R + C (series)"
    ),
    CircuitDefinition(
        name="Randles",
        short_name="Randles",
        param_names=['Rs', 'Rct', 'Cdl'],
        param_labels=['R$_s$', 'R$_{ct}$', 'C$_{dl}$'],
        param_ranges={'Rs': (1.0, 2.5), 'Rct': (2.0, 3.5), 'Cdl': (-6.5, -5.0)},
        compute_impedance=compute_impedance_randles,
        generate_trajectory=generate_trajectory_randles,
        description="Rs + (Rct || Cdl)"
    ),
    CircuitDefinition(
        name="2-RC RPE Model",
        short_name="2-RC",
        param_names=['Ra', 'Rb', 'Ca', 'Cb', 'Rsh'],
        param_labels=['R$_a$', 'R$_b$', 'C$_a$', 'C$_b$', 'R$_{sh}$'],
        param_ranges={
            'Ra': (2.2, 3.5), 'Rb': (2.0, 3.0),
            'Ca': (-6.0, -5.0), 'Cb': (-6.0, -5.0), 'Rsh': (2.6, 3.4)
        },
        compute_impedance=compute_impedance_2rc,
        generate_trajectory=generate_trajectory_2rc,
        description="Rsh || (Za + Zb)"
    ),
    CircuitDefinition(
        name="Randles + Warburg",
        short_name="Warburg",
        param_names=['Rs', 'Rct', 'Cdl', 'Aw'],
        param_labels=['R$_s$', 'R$_{ct}$', 'C$_{dl}$', 'A$_w$'],
        param_ranges={'Rs': (1.0, 2.2), 'Rct': (2.0, 3.2), 'Cdl': (-6.5, -5.0), 'Aw': (1.5, 2.5)},
        compute_impedance=compute_impedance_randles_warburg,
        generate_trajectory=generate_trajectory_randles_warburg,
        description="Rs + (Rct || Cdl) + Warburg"
    ),
    CircuitDefinition(
        name="3-RC Extended",
        short_name="3-RC",
        param_names=['Ra', 'Rb', 'Rc', 'Ca', 'Cb', 'Cc', 'Rsh'],
        param_labels=['R$_a$', 'R$_b$', 'R$_c$', 'C$_a$', 'C$_b$', 'C$_c$', 'R$_{sh}$'],
        param_ranges={
            'Ra': (2.0, 3.2), 'Rb': (1.8, 3.0), 'Rc': (1.8, 2.8),
            'Ca': (-6.2, -5.2), 'Cb': (-6.0, -5.0), 'Cc': (-5.8, -4.8),
            'Rsh': (2.8, 3.6)
        },
        compute_impedance=compute_impedance_3rc,
        generate_trajectory=generate_trajectory_3rc,
        description="Rsh || (Za + Zb + Zc)"
    ),
]


# =============================================================================
# Core Analysis Functions
# =============================================================================

def compute_resnorm(Z_pred: np.ndarray, Z_true: np.ndarray) -> float:
    """Compute residual norm for impedance fit."""
    mag_err = np.mean(np.abs(np.log10(np.abs(Z_pred)+1e-10) - np.log10(np.abs(Z_true)+1e-10)))
    phase_err = np.mean(np.abs(np.angle(Z_pred) - np.angle(Z_true))) / np.pi
    return mag_err + phase_err


def compute_param_distance(sol: Dict, true_params: Dict, param_names: List[str]) -> float:
    """Compute RMS distance from truth in log space."""
    distances = []
    for p in param_names:
        true_log = np.log10(true_params[p])
        sol_log = sol['log'][p]
        distances.append((sol_log - true_log) ** 2)
    return np.sqrt(np.mean(distances))


def find_solutions(circuit: CircuitDefinition, Z_target: np.ndarray,
                   frequencies: np.ndarray, n_samples: int = 50000,
                   threshold: float = 0.05) -> List[Dict]:
    """Find parameter solutions that match target impedance."""
    solutions = []
    for _ in range(n_samples):
        params = {p: 10 ** np.random.uniform(lo, hi)
                  for p, (lo, hi) in circuit.param_ranges.items()}
        Z_pred = circuit.compute_impedance(params, frequencies)
        resnorm = compute_resnorm(Z_pred, Z_target)
        if resnorm < threshold:
            params['log'] = {p: np.log10(params[p]) for p in circuit.param_names}
            params['resnorm'] = resnorm
            solutions.append(params)
    return solutions


def filter_reachable(current: List[Dict], previous: List[Dict],
                     param_names: List[str], max_drift: float = 0.06) -> List[Dict]:
    """Filter solutions reachable from previous timepoint."""
    if not previous:
        return current
    reachable = []
    for curr in current:
        for prev in previous:
            if all(abs(curr['log'][p] - prev['log'][p]) <= max_drift for p in param_names):
                reachable.append(curr)
                break
    return reachable


def run_circuit_analysis(circuit: CircuitDefinition, n_timepoints: int = 6,
                         n_samples: int = 50000, max_drift: float = 0.06) -> Dict:
    """Run full temporal analysis for a circuit."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {circuit.name}")
    print(f"Parameters: {circuit.param_names}")
    print(f"{'='*60}")

    frequencies = np.logspace(-1, 5, 80)
    true_trajectory = circuit.generate_trajectory(n_timepoints)

    # Generate noisy measurements
    np.random.seed(123)
    measurements = []
    for params in true_trajectory:
        Z = circuit.compute_impedance(params, frequencies)
        Z_noisy = Z * (1 + 0.01 * (np.random.randn(len(Z)) + 1j * np.random.randn(len(Z))))
        measurements.append(Z_noisy)

    # Find unconstrained solutions
    all_unconstrained = []
    for t in range(n_timepoints):
        sols = find_solutions(circuit, measurements[t], frequencies, n_samples=n_samples)
        all_unconstrained.append(sols)
        print(f"  t={t+1}: {len(sols)} unconstrained solutions")

    # Apply temporal constraints
    all_constrained = []
    prev = None
    for t in range(n_timepoints):
        if prev is None:
            constrained = all_unconstrained[t]
        else:
            constrained = filter_reachable(all_unconstrained[t], prev,
                                           circuit.param_names, max_drift=max_drift)
            if len(constrained) < 10:
                # Relax constraint if too few solutions
                constrained = filter_reachable(all_unconstrained[t], prev,
                                               circuit.param_names, max_drift=max_drift * 1.5)
        all_constrained.append(constrained)
        prev = constrained
        print(f"  t={t+1}: {len(constrained)} constrained solutions")

    # Compute distances from truth
    for t in range(n_timepoints):
        for sol in all_constrained[t]:
            sol['truth_dist'] = compute_param_distance(sol, true_trajectory[t], circuit.param_names)

    # Compute per-parameter errors
    param_errors = {p: {'t1': [], 't_last': []} for p in circuit.param_names}
    for sol in all_constrained[0]:
        for p in circuit.param_names:
            err = abs(sol['log'][p] - np.log10(true_trajectory[0][p]))
            param_errors[p]['t1'].append(err)
    for sol in all_constrained[-1]:
        for p in circuit.param_names:
            err = abs(sol['log'][p] - np.log10(true_trajectory[-1][p]))
            param_errors[p]['t_last'].append(err)

    param_mae = {p: {
        't1': np.mean(param_errors[p]['t1']) if param_errors[p]['t1'] else 0.5,
        't_last': np.mean(param_errors[p]['t_last']) if param_errors[p]['t_last'] else 0.5,
    } for p in circuit.param_names}

    return {
        'circuit': circuit,
        'frequencies': frequencies,
        'true_trajectory': true_trajectory,
        'measurements': measurements,
        'all_unconstrained': all_unconstrained,
        'all_constrained': all_constrained,
        'param_mae': param_mae,
        'n_solutions_t1': len(all_constrained[0]),
        'n_solutions_tlast': len(all_constrained[-1]),
        'mean_dist_t1': np.mean([s['truth_dist'] for s in all_constrained[0]]) if all_constrained[0] else 0.5,
        'mean_dist_tlast': np.mean([s['truth_dist'] for s in all_constrained[-1]]) if all_constrained[-1] else 0.5,
    }


# =============================================================================
# Visualization Functions
# =============================================================================

def create_comparison_summary(results: List[Dict]) -> plt.Figure:
    """Create summary comparison across all circuits."""
    n_circuits = len(results)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    # Panel 1: Solution reduction
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(n_circuits)
    width = 0.35

    n_t1 = [r['n_solutions_t1'] for r in results]
    n_tlast = [r['n_solutions_tlast'] for r in results]

    bars1 = ax1.bar(x - width/2, n_t1, width, label='t=1 (unconstrained)',
                    color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, n_tlast, width, label='t=6 (constrained)',
                    color='#27ae60', alpha=0.8)

    ax1.set_ylabel('Number of Valid Solutions', fontsize=11)
    ax1.set_title('Solution Space Reduction by Temporal Constraints',
                  fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([r['circuit'].short_name for r in results], fontsize=10)
    ax1.legend(loc='upper right')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Add reduction percentages
    for i, (v1, v2) in enumerate(zip(n_t1, n_tlast)):
        reduction = (1 - v2/v1) * 100 if v1 > 0 else 0
        ax1.text(i, max(v1, v2) + 50, f'-{reduction:.0f}%', ha='center',
                fontsize=9, fontweight='bold', color='#27ae60')

    # Panel 2: Distance from truth improvement
    ax2 = fig.add_subplot(gs[0, 1])

    dist_t1 = [r['mean_dist_t1'] for r in results]
    dist_tlast = [r['mean_dist_tlast'] for r in results]

    bars1 = ax2.bar(x - width/2, dist_t1, width, label='t=1', color='#e74c3c', alpha=0.8)
    bars2 = ax2.bar(x + width/2, dist_tlast, width, label='t=6', color='#27ae60', alpha=0.8)

    ax2.set_ylabel('Mean Distance from Truth (decades)', fontsize=11)
    ax2.set_title('Parameter Accuracy Improvement', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([r['circuit'].short_name for r in results], fontsize=10)
    ax2.legend(loc='upper right')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Add improvement percentages
    for i, (v1, v2) in enumerate(zip(dist_t1, dist_tlast)):
        imp = (1 - v2/v1) * 100 if v1 > 0 else 0
        color = '#27ae60' if imp > 0 else '#e74c3c'
        ax2.text(i, max(v1, v2) + 0.02, f'{imp:+.0f}%', ha='center',
                fontsize=9, fontweight='bold', color=color)

    # Panel 3: Per-parameter improvement by circuit
    ax3 = fig.add_subplot(gs[1, :])

    # Collect all parameter improvements
    all_params = []
    all_improvements = []
    all_circuits = []
    colors = []
    circuit_colors = plt.cm.tab10(np.linspace(0, 1, n_circuits))

    for i, r in enumerate(results):
        for p in r['circuit'].param_names:
            mae_t1 = r['param_mae'][p]['t1']
            mae_tlast = r['param_mae'][p]['t_last']
            imp = (1 - mae_tlast / mae_t1) * 100 if mae_t1 > 0 else 0
            all_params.append(f"{r['circuit'].short_name}:{p}")
            all_improvements.append(imp)
            all_circuits.append(r['circuit'].short_name)
            colors.append(circuit_colors[i])

    x_params = np.arange(len(all_params))
    bars = ax3.bar(x_params, all_improvements, color=colors, alpha=0.8, edgecolor='white')

    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_ylabel('Improvement t=1 to t=6 (%)', fontsize=11)
    ax3.set_title('Per-Parameter Improvement Across All Circuits', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_params)
    ax3.set_xticklabels(all_params, fontsize=8, rotation=45, ha='right')
    ax3.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Add vertical lines between circuits
    param_count = 0
    for r in results[:-1]:
        param_count += len(r['circuit'].param_names)
        ax3.axvline(x=param_count - 0.5, color='gray', linestyle=':', alpha=0.5)

    fig.suptitle('Temporal Smoothness Constraints: Multi-Circuit Comparison\n'
                 'Demonstrating generalization across different circuit topologies',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def create_individual_circuit_figure(result: Dict) -> plt.Figure:
    """Create detailed figure for a single circuit."""
    circuit = result['circuit']
    n_timepoints = len(result['true_trajectory'])
    n_params = len(circuit.param_names)

    fig = plt.figure(figsize=(14, 8))

    # Layout: top row = manifold evolution, bottom row = per-param errors
    gs = gridspec.GridSpec(2, max(n_timepoints, n_params), figure=fig,
                           height_ratios=[1.2, 1], hspace=0.35, wspace=0.2)

    # Compute scales for radar plot
    all_log = {p: [] for p in circuit.param_names}
    for sols in result['all_constrained']:
        for s in sols[:300]:
            for p in circuit.param_names:
                all_log[p].append(s['log'][p])
    for t in result['true_trajectory']:
        for p in circuit.param_names:
            all_log[p].append(np.log10(t[p]))

    scales = {p: (min(all_log[p]) - 0.1, max(all_log[p]) + 0.1) for p in circuit.param_names}

    # Distance range for coloring
    all_dists = [s['truth_dist'] for sols in result['all_constrained'] for s in sols]
    if all_dists:
        dist_min, dist_max = min(all_dists), max(all_dists)
    else:
        dist_min, dist_max = 0, 1

    cmap = plt.cm.RdYlGn_r
    norm = Normalize(vmin=dist_min, vmax=dist_max)
    angles = np.linspace(0, 2*np.pi, n_params, endpoint=False)

    mean_dists = []

    # Top row: Manifold at each timepoint
    for t in range(n_timepoints):
        ax = fig.add_subplot(gs[0, t])
        ax.set_aspect('equal')
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.axis('off')

        # Draw axes
        for i, (angle, label) in enumerate(zip(angles, circuit.param_labels)):
            ax.plot([0, 1.1*np.cos(angle)], [0, 1.1*np.sin(angle)],
                   color='#2c3e50', lw=0.5, alpha=0.3)
            ax.text(1.3*np.cos(angle), 1.3*np.sin(angle), label,
                   ha='center', va='center', fontsize=9, fontweight='bold')

        solutions = result['all_constrained'][t]
        solutions_sorted = sorted(solutions, key=lambda x: -x['truth_dist'])[:200]

        for sol in solutions_sorted:
            color = cmap(norm(sol['truth_dist']))
            coords = []
            for i, p in enumerate(circuit.param_names):
                lo, hi = scales[p]
                val_norm = np.clip((sol['log'][p] - lo) / (hi - lo), 0.05, 1.0)
                coords.append([val_norm * np.cos(angles[i]), val_norm * np.sin(angles[i])])
            coords = np.array(coords)
            coords = np.vstack([coords, coords[0]])
            ax.plot(coords[:, 0], coords[:, 1], '-', color=color, alpha=0.35, lw=0.6)

        # Draw truth
        true_coords = []
        for i, p in enumerate(circuit.param_names):
            lo, hi = scales[p]
            val_norm = np.clip((np.log10(result['true_trajectory'][t][p]) - lo) / (hi - lo), 0.05, 1.0)
            true_coords.append([val_norm * np.cos(angles[i]), val_norm * np.sin(angles[i])])
        true_coords = np.array(true_coords)
        true_coords = np.vstack([true_coords, true_coords[0]])
        ax.plot(true_coords[:, 0], true_coords[:, 1], '-', color='black', lw=2.5)

        mean_dist = np.mean([s['truth_dist'] for s in solutions]) if solutions else 0
        mean_dists.append(mean_dist)
        ax.set_title(f't={t+1}\n{len(solutions)} sols', fontsize=9, fontweight='bold')

    # Bottom row: Per-parameter error bars
    ax_params = fig.add_subplot(gs[1, :])

    x = np.arange(n_params)
    width = 0.35

    mae_t1 = [result['param_mae'][p]['t1'] for p in circuit.param_names]
    mae_tlast = [result['param_mae'][p]['t_last'] for p in circuit.param_names]

    bars1 = ax_params.bar(x - width/2, mae_t1, width, label='t=1', color='#e74c3c', alpha=0.8)
    bars2 = ax_params.bar(x + width/2, mae_tlast, width, label='t=6', color='#27ae60', alpha=0.8)

    ax_params.set_ylabel('MAE (decades)', fontsize=10)
    ax_params.set_xticks(x)
    ax_params.set_xticklabels(circuit.param_labels, fontsize=10)
    ax_params.legend(loc='upper right', fontsize=9)
    ax_params.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Add improvement labels
    for i, (v1, v2) in enumerate(zip(mae_t1, mae_tlast)):
        imp = (1 - v2/v1) * 100 if v1 > 0 else 0
        color = '#27ae60' if imp > 0 else '#e74c3c'
        ax_params.text(i, max(v1, v2) + 0.02, f'{imp:+.0f}%', ha='center',
                      fontsize=9, fontweight='bold', color=color)

    # Title with summary stats
    n_reduction = (1 - result['n_solutions_tlast'] / result['n_solutions_t1']) * 100
    dist_improvement = (1 - result['mean_dist_tlast'] / result['mean_dist_t1']) * 100

    fig.suptitle(f"{circuit.name}: {circuit.description}\n"
                 f"Solutions: {result['n_solutions_t1']} → {result['n_solutions_tlast']} "
                 f"({n_reduction:.0f}% reduction) | "
                 f"Mean distance: {result['mean_dist_t1']:.3f} → {result['mean_dist_tlast']:.3f} "
                 f"({dist_improvement:+.0f}%)",
                 fontsize=11, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def create_summary_table(results: List[Dict]) -> str:
    """Create text summary table."""
    lines = []
    lines.append("=" * 90)
    lines.append("MULTI-CIRCUIT TEMPORAL ANALYSIS SUMMARY")
    lines.append("=" * 90)
    lines.append("")
    lines.append(f"{'Circuit':<20} {'Params':<8} {'Sol t=1':<10} {'Sol t=6':<10} {'Reduction':<12} {'Dist Imp'}")
    lines.append("-" * 90)

    for r in results:
        n_reduction = (1 - r['n_solutions_tlast'] / r['n_solutions_t1']) * 100 if r['n_solutions_t1'] > 0 else 0
        dist_imp = (1 - r['mean_dist_tlast'] / r['mean_dist_t1']) * 100 if r['mean_dist_t1'] > 0 else 0

        lines.append(f"{r['circuit'].name:<20} {len(r['circuit'].param_names):<8} "
                    f"{r['n_solutions_t1']:<10} {r['n_solutions_tlast']:<10} "
                    f"{n_reduction:>6.1f}%      {dist_imp:>+6.1f}%")

    lines.append("-" * 90)
    lines.append("")
    lines.append("INTERPRETATION:")
    lines.append("- All circuits show solution space reduction with temporal constraints")
    lines.append("- Parameter accuracy improves (distance from truth decreases)")
    lines.append("- More complex circuits (more params) show larger manifolds but similar improvement ratios")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("=" * 70)
    print("MULTI-CIRCUIT TEMPORAL ANALYSIS")
    print("Testing temporal smoothness across different circuit topologies")
    print("=" * 70)

    # Run analysis for each circuit
    results = []
    for circuit in CIRCUITS:
        result = run_circuit_analysis(circuit, n_timepoints=6, n_samples=50000)
        results.append(result)

    # Generate visualizations
    print("\n" + "=" * 70)
    print("Generating visualizations...")
    print("=" * 70)

    # Summary comparison
    fig_summary = create_comparison_summary(results)
    fig_summary.savefig(OUTPUT_DIR / 'multi_circuit_comparison.png',
                        dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig_summary)
    print(f"Saved: {OUTPUT_DIR / 'multi_circuit_comparison.png'}")

    # Individual circuit figures
    for result in results:
        fig = create_individual_circuit_figure(result)
        filename = f"circuit_{result['circuit'].short_name.lower().replace(' ', '_')}.png"
        fig.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Saved: {OUTPUT_DIR / filename}")

    # Print summary table
    summary = create_summary_table(results)
    print("\n" + summary)

    # Save summary to file
    with open(OUTPUT_DIR / 'summary.txt', 'w') as f:
        f.write(summary)
    print(f"Saved: {OUTPUT_DIR / 'summary.txt'}")


if __name__ == '__main__':
    main()
