#!/usr/bin/env python3
"""
Randles Circuit Simulator for EIS

Forward simulation: Circuit parameters → Impedance spectrum

Used for:
1. Generating synthetic training data
2. Validating CNN predictions
3. Computing residuals between predicted and real spectra
4. Testing parameter identifiability
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


class RandlesCircuitSimulator:
    """
    Simulate impedance spectra from RCRC equivalent circuit.

    ╔════════════════════════════════════════════════════════════════╗
    ║  CIRCUIT TOPOLOGY: PARALLEL (per Srinivasan et al.)           ║
    ║                                                                 ║
    ║    Rsh || [(Ra||Ca) ── (Rb||Cb)]                              ║
    ║                                                                 ║
    ║  Where:                                                         ║
    ║    Z_a(ω) = Ra / (1 + jωRaCa)                                 ║
    ║    Z_b(ω) = Rb / (1 + jωRbCb)                                 ║
    ║    Z_series = Z_a(ω) + Z_b(ω)                                 ║
    ║                                                                 ║
    ║  Impedance (parallel combination):                             ║
    ║    Z(ω) = [Rsh × Z_series] / [Rsh + Z_series]                ║
    ║         = [Rsh × (Z_a + Z_b)] / [Rsh + Z_a + Z_b]            ║
    ║                                                                 ║
    ║  At DC (ω→0): Z_a(0)=Ra, Z_b(0)=Rb                            ║
    ║    TER = [Rsh × (Ra + Rb)] / [Rsh + Ra + Rb]                 ║
    ║                                                                 ║
    ║  At high frequency (ω→∞): Z_a(∞)=0, Z_b(∞)=0                 ║
    ║    Z(∞) = 0  (short circuit)                                   ║
    ╚════════════════════════════════════════════════════════════════╝

    Parameters:
    - Rsh: Solution/electrode resistance (Ω)
    - Ra: Apical membrane resistance (Ω)
    - Ca: Apical membrane capacitance (F)
    - Rb: Basolateral membrane resistance (Ω)
    - Cb: Basolateral membrane capacitance (F)

    Reference: Figure 1D from Srinivasan et al. (epithelial transport model)
    """

    def __init__(self):
        pass

    def compute_impedance(self, frequencies, Ra, Rb, Ca, Cb, Rsh):
        """
        Compute complex impedance spectrum.

        Args:
            frequencies: Array of frequencies (Hz)
            Ra: Apical resistance (Ω)
            Rb: Basolateral resistance (Ω)
            Ca: Apical capacitance (F)
            Cb: Basolateral capacitance (F)
            Rsh: Shunt resistance (Ω)

        Returns:
            Z_real: Real impedance (Z', Ω)
            Z_imag: Imaginary impedance (Z'', Ω)
        """
        omega = 2 * np.pi * frequencies

        # Apical branch impedance (parallel RC)
        # Z_a(ω) = Ra / (1 + jωRaCa)
        denominator_a = 1 + 1j * omega * Ra * Ca
        Z_apical = Ra / denominator_a

        # Basolateral branch impedance (parallel RC)
        # Z_b(ω) = Rb / (1 + jωRbCb)
        denominator_b = 1 + 1j * omega * Rb * Cb
        Z_basal = Rb / denominator_b

        # Series combination of apical and basolateral branches
        # Z_series = Z_a(ω) + Z_b(ω)
        Z_series = Z_apical + Z_basal

        # Total impedance: PARALLEL topology per Srinivasan et al.
        # Rsh || Z_series = [Rsh × Z_series] / [Rsh + Z_series]
        # Where:
        #   Z_series = Z_a + Z_b
        #   Z_a = Ra / (1 + jωRaCa)
        #   Z_b = Rb / (1 + jωRbCb)
        # Therefore:
        #   Z_total = [Rsh × (Z_a + Z_b)] / [Rsh + Z_a + Z_b]
        Z_total = (Rsh * Z_series) / (Rsh + Z_series)

        return Z_total.real, Z_total.imag

    def compute_ter(self, Ra, Rb, Rsh):
        """
        Compute TER (Transepithelial Resistance) from circuit parameters.

        Topology: PARALLEL per Srinivasan et al.
        Circuit: Rsh || [(Ra||Ca) ── (Rb||Cb)]

        At DC (ω → 0), capacitors are OPEN circuits:
          Z_a(ω→0) = Ra
          Z_b(ω→0) = Rb
          Z_series(ω→0) = Ra + Rb

        Parallel combination:
          TER = Z_total(ω→0) = [Rsh × (Ra + Rb)] / [Rsh + Ra + Rb]

        Units: Ω (ohms)
        """
        return (Rsh * (Ra + Rb)) / (Rsh + Ra + Rb)

    def compute_spectrum_dict(self, frequencies, Ra, Rb, Ca, Cb, Rsh):
        """
        Compute impedance spectrum and return as dictionary.

        Returns:
            dict with frequencies, z_real, z_imag, parameters
        """
        z_real, z_imag = self.compute_impedance(frequencies, Ra, Rb, Ca, Cb, Rsh)
        ter = self.compute_ter(Ra, Rb, Rsh)

        return {
            'frequencies': frequencies,
            'z_real': z_real,
            'z_imag': z_imag,
            'parameters': {
                'Ra': Ra,
                'Rb': Rb,
                'Ca': Ca,
                'Cb': Cb,
                'Rsh': Rsh,
                'TER': ter
            }
        }

    def generate_synthetic_dataset(self, n_samples=1000, frequency_range=(5, 10000),
                                   n_frequencies=25, add_noise=True, noise_level=0.02):
        """
        Generate synthetic dataset by sampling parameter space.

        Args:
            n_samples: Number of synthetic spectra to generate
            frequency_range: (min, max) frequencies in Hz
            n_frequencies: Number of frequency points
            add_noise: Whether to add measurement noise
            noise_level: Fraction of signal for noise (default 2%)

        Returns:
            DataFrame with synthetic spectra and ground-truth parameters
        """
        # Frequency grid (log-spaced)
        frequencies = np.logspace(
            np.log10(frequency_range[0]),
            np.log10(frequency_range[1]),
            n_frequencies
        )

        # Parameter ranges (based on typical RPE cell barriers)
        param_ranges = {
            'Ra': (5, 30),      # Ω (apical resistance)
            'Rb': (5, 30),      # Ω (basolateral resistance)
            'Ca': (1e-6, 10e-6),  # F (apical capacitance, 1-10 μF)
            'Cb': (1e-6, 10e-6),  # F (basolateral capacitance, 1-10 μF)
            'Rsh': (10, 50)     # Ω (shunt resistance)
        }

        # Generate random parameter samples
        samples = []

        for i in range(n_samples):
            # Sample parameters uniformly
            Ra = np.random.uniform(*param_ranges['Ra'])
            Rb = np.random.uniform(*param_ranges['Rb'])
            Ca = np.random.uniform(*param_ranges['Ca'])
            Cb = np.random.uniform(*param_ranges['Cb'])
            Rsh = np.random.uniform(*param_ranges['Rsh'])

            # Compute spectrum
            z_real, z_imag = self.compute_impedance(frequencies, Ra, Rb, Ca, Cb, Rsh)

            # Add noise if requested
            if add_noise:
                noise_real = np.random.normal(0, noise_level * np.abs(z_real).mean(), len(z_real))
                noise_imag = np.random.normal(0, noise_level * np.abs(z_imag).mean(), len(z_imag))
                z_real += noise_real
                z_imag += noise_imag

            # Compute TER
            ter = self.compute_ter(Ra, Rb, Rsh)

            # Store sample
            for j, freq in enumerate(frequencies):
                samples.append({
                    'sample_id': i,
                    'frequency_hz': freq,
                    'z_real_ohm': z_real[j],
                    'z_imag_ohm': z_imag[j],
                    'Ra': Ra,
                    'Rb': Rb,
                    'Ca': Ca,
                    'Cb': Cb,
                    'Rsh': Rsh,
                    'TER': ter
                })

        return pd.DataFrame(samples)

    def compute_residual(self, real_spectrum, predicted_params, frequencies):
        """
        Compute residual between real and simulated spectrum.

        Args:
            real_spectrum: Dict with 'z_real' and 'z_imag' arrays
            predicted_params: Dict with 'Ra', 'Rb', 'Ca', 'Cb', 'Rsh'
            frequencies: Frequency array

        Returns:
            residual: MAE across all frequencies
        """
        # Forward simulate with predicted parameters
        z_real_sim, z_imag_sim = self.compute_impedance(
            frequencies,
            predicted_params['Ra'],
            predicted_params['Rb'],
            predicted_params['Ca'],
            predicted_params['Cb'],
            predicted_params['Rsh']
        )

        # Compute point-wise residuals
        residual_real = np.abs(z_real_sim - real_spectrum['z_real'])
        residual_imag = np.abs(z_imag_sim - real_spectrum['z_imag'])

        # Combined residual (MAE)
        residual = (residual_real + residual_imag).mean()

        return residual


def plot_example_spectra(simulator, n_examples=6):
    """Plot example synthetic spectra."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    frequencies = np.logspace(np.log10(5), np.log10(10000), 25)

    for i in range(n_examples):
        # Random parameters
        Ra = np.random.uniform(5, 30)
        Rb = np.random.uniform(5, 30)
        Ca = np.random.uniform(1e-6, 10e-6)
        Cb = np.random.uniform(1e-6, 10e-6)
        Rsh = np.random.uniform(10, 50)

        # Simulate
        z_real, z_imag = simulator.compute_impedance(frequencies, Ra, Rb, Ca, Cb, Rsh)
        ter = simulator.compute_ter(Ra, Rb, Rsh)

        # Plot Nyquist
        ax = axes[i]
        ax.plot(z_real, z_imag, 'o-', linewidth=2, markersize=4)
        ax.plot(z_real[0], z_imag[0], 'ro', markersize=8, label='High freq')
        ax.plot(z_real[-1], z_imag[-1], 'go', markersize=8, label='Low freq')

        ax.set_xlabel("Z' (Ω)")
        ax.set_ylabel("Z'' (Ω)")
        ax.set_title(f'TER={ter:.1f}Ω, Ra={Ra:.1f}, Rb={Rb:.1f}', fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)

        if i == 0:
            ax.legend(fontsize=8)

    plt.suptitle('Synthetic Nyquist Plots from Randles Circuit', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('synthetic_spectra_examples.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: synthetic_spectra_examples.png")


def main():
    """Generate synthetic dataset for training."""
    print("="*70)
    print("RANDLES CIRCUIT SIMULATOR")
    print("="*70)

    simulator = RandlesCircuitSimulator()

    # Generate synthetic dataset
    print("\n🔬 Generating synthetic dataset...")
    print("   Sampling parameter space uniformly")

    df = simulator.generate_synthetic_dataset(
        n_samples=5000,
        frequency_range=(5, 10000),
        n_frequencies=25,
        add_noise=True,
        noise_level=0.02
    )

    print(f"✓ Generated {df['sample_id'].nunique()} synthetic spectra")
    print(f"✓ Total data points: {len(df):,}")

    # Summary statistics
    print("\n📊 Parameter Ranges:")
    for param in ['Ra', 'Rb', 'Ca', 'Cb', 'Rsh', 'TER']:
        param_values = df.groupby('sample_id')[param].first()
        print(f"   {param:5s}: {param_values.min():.2e} - {param_values.max():.2e}")

    # Save dataset
    output_dir = Path("data/synthetic_eis")
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "synthetic_spectra.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved: {csv_path}")

    # Generate example plots
    print("\n🎨 Generating example plots...")
    plot_example_spectra(simulator)

    # Save simulator for later use
    import pickle
    simulator_path = output_dir / "randles_simulator.pkl"
    with open(simulator_path, 'wb') as f:
        pickle.dump(simulator, f)
    print(f"✓ Saved simulator: {simulator_path}")

    print("\n" + "="*70)
    print("SYNTHETIC DATA GENERATION COMPLETE")
    print("="*70)
    print(f"\nDataset: {csv_path}")
    print(f"Use this for:")
    print("  1. Training parameter inference CNNs")
    print("  2. Validating predictions")
    print("  3. Testing identifiability")
    print("="*70)


if __name__ == "__main__":
    main()
