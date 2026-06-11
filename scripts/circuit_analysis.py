#!/usr/bin/env python3

"""
Circuit Parameter Analysis Script

Interactive script for generating circuit parameter variations,
computing PCA/directional analysis, and creating visualizations.

Usage: python scripts/circuit_analysis.py
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any
import argparse
from dataclasses import dataclass
from pathlib import Path

@dataclass
class CircuitParameters:
    """Circuit parameters for the equivalent circuit model"""
    Rsh: float  # Shunt resistance (Ω)
    Ra: float   # Apical resistance (Ω)
    Ca: float   # Apical capacitance (F)
    Rb: float   # Basal resistance (Ω)
    Cb: float   # Basal capacitance (F)
    frequency_range: Tuple[float, float] = (1.0, 10000.0)

@dataclass
class AnalysisConfig:
    """Configuration for the analysis run"""
    ground_truth_circuit: CircuitParameters
    variation_ranges: Dict[str, Tuple[float, float]]
    num_variations: int
    frequency_range: Tuple[float, float]
    num_frequencies: int
    output_dir: str
    run_name: str
    grid_mode: bool = False  # Enable grid-based parameter variations
    grid_size: int = 5  # Grid points per parameter dimension

class CircuitAnalyzer:
    """Main class for circuit parameter analysis"""

    def __init__(self):
        self.frequencies = None
        self.ground_truth_spectrum = None

    def calculate_impedance(self, params: CircuitParameters, frequency: float) -> complex:
        """
        Calculate impedance at a given frequency using the equivalent circuit model

        Z_total = Rsh || (Za + Zb) where:
        - Za = Ra/(1+jωRaCa)
        - Zb = Rb/(1+jωRbCb)
        """
        omega = 2 * np.pi * frequency

        # Calculate impedance of apical membrane (Ra || Ca)
        Za_denom = 1 + (omega * params.Ra * params.Ca)**2
        Za_real = params.Ra / Za_denom
        Za_imag = -omega * params.Ra**2 * params.Ca / Za_denom

        # Calculate impedance of basal membrane (Rb || Cb)
        Zb_denom = 1 + (omega * params.Rb * params.Cb)**2
        Zb_real = params.Rb / Zb_denom
        Zb_imag = -omega * params.Rb**2 * params.Cb / Zb_denom

        # Series combination of membranes
        Zab_real = Za_real + Zb_real
        Zab_imag = Za_imag + Zb_imag
        Zab = complex(Zab_real, Zab_imag)

        # Parallel combination with shunt resistance
        # Z_total = (Rsh * Zab) / (Rsh + Zab)
        numerator = params.Rsh * Zab
        denominator = params.Rsh + Zab

        return numerator / denominator

    def calculate_spectrum(self, params: CircuitParameters, frequencies: np.ndarray) -> pd.DataFrame:
        """Calculate impedance spectrum across frequency range"""
        spectrum_data = []

        for freq in frequencies:
            Z = self.calculate_impedance(params, freq)
            spectrum_data.append({
                'frequency': freq,
                'real': Z.real,
                'imaginary': Z.imag,
                'magnitude': abs(Z),
                'phase': np.angle(Z, deg=True)
            })

        return pd.DataFrame(spectrum_data)

    def calculate_resnorm(self, test_spectrum: pd.DataFrame, ref_spectrum: pd.DataFrame) -> float:
        """Calculate residual norm (MAE) between test and reference spectra"""
        if len(test_spectrum) != len(ref_spectrum):
            raise ValueError("Spectra must have same length")

        # Calculate complex magnitude error at each frequency
        real_diff = test_spectrum['real'].values - ref_spectrum['real'].values
        imag_diff = test_spectrum['imaginary'].values - ref_spectrum['imaginary'].values
        magnitude_errors = np.sqrt(real_diff**2 + imag_diff**2)

        # Return mean absolute error
        return np.mean(magnitude_errors)

    def compute_jacobian(self, params: CircuitParameters, frequencies: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute analytic Jacobian of complex impedance with respect to circuit parameters
        Returns partial derivatives ∂Z/∂param for each frequency and parameter
        """
        num_freq = len(frequencies)
        param_names = ['Rsh', 'Ra', 'Ca', 'Rb', 'Cb']
        jacobian_real = np.zeros((num_freq, len(param_names)))
        jacobian_imag = np.zeros((num_freq, len(param_names)))

        for i, freq in enumerate(frequencies):
            omega = 2 * np.pi * freq

            # Pre-compute common terms
            omega_Ra_Ca = omega * params.Ra * params.Ca
            omega_Rb_Cb = omega * params.Rb * params.Cb
            Da = 1 + omega_Ra_Ca**2
            Db = 1 + omega_Rb_Cb**2

            # Branch impedances
            Za_real = params.Ra / Da
            Za_imag = -omega_Ra_Ca * params.Ra / Da
            Zb_real = params.Rb / Db
            Zb_imag = -omega_Rb_Cb * params.Rb / Db

            # Series combination
            Zab_real = Za_real + Zb_real
            Zab_imag = Za_imag + Zb_imag

            # Parallel combination denominators
            denom_real = params.Rsh + Zab_real
            denom_imag = Zab_imag
            denom_mag_sq = denom_real**2 + denom_imag**2

            # ∂Z/∂Rsh
            num_real = Zab_real
            num_imag = Zab_imag
            jacobian_real[i, 0] = (num_real * denom_real + num_imag * denom_imag) / denom_mag_sq
            jacobian_imag[i, 0] = (num_imag * denom_real - num_real * denom_imag) / denom_mag_sq

            # ∂Z/∂Ra
            dZa_real_dRa = 1/Da - 2*params.Ra*omega_Ra_Ca**2/(Da**2)
            dZa_imag_dRa = -omega*params.Ca/Da + 2*params.Ra*omega*params.Ca*omega_Ra_Ca**2/(Da**2)

            # Chain rule through parallel combination
            dZ_dZab_real = (params.Rsh * denom_real - params.Rsh * Zab_real) / denom_mag_sq
            dZ_dZab_imag = (params.Rsh * denom_imag - params.Rsh * Zab_imag) / denom_mag_sq

            jacobian_real[i, 1] = dZ_dZab_real * dZa_real_dRa - dZ_dZab_imag * dZa_imag_dRa
            jacobian_imag[i, 1] = dZ_dZab_imag * dZa_real_dRa + dZ_dZab_real * dZa_imag_dRa

            # ∂Z/∂Ca
            dZa_real_dCa = 2*params.Ra**3*omega**2*params.Ca/(Da**2)
            dZa_imag_dCa = -omega*params.Ra**2/Da + 2*params.Ra**3*omega**2*params.Ca*omega_Ra_Ca/(Da**2)

            jacobian_real[i, 2] = dZ_dZab_real * dZa_real_dCa - dZ_dZab_imag * dZa_imag_dCa
            jacobian_imag[i, 2] = dZ_dZab_imag * dZa_real_dCa + dZ_dZab_real * dZa_imag_dCa

            # ∂Z/∂Rb (symmetric to Ra)
            dZb_real_dRb = 1/Db - 2*params.Rb*omega_Rb_Cb**2/(Db**2)
            dZb_imag_dRb = -omega*params.Cb/Db + 2*params.Rb*omega*params.Cb*omega_Rb_Cb**2/(Db**2)

            jacobian_real[i, 3] = dZ_dZab_real * dZb_real_dRb - dZ_dZab_imag * dZb_imag_dRb
            jacobian_imag[i, 3] = dZ_dZab_imag * dZb_real_dRb + dZ_dZab_real * dZb_imag_dRb

            # ∂Z/∂Cb (symmetric to Ca)
            dZb_real_dCb = 2*params.Rb**3*omega**2*params.Cb/(Db**2)
            dZb_imag_dCb = -omega*params.Rb**2/Db + 2*params.Rb**3*omega**2*params.Cb*omega_Rb_Cb/(Db**2)

            jacobian_real[i, 4] = dZ_dZab_real * dZb_real_dCb - dZ_dZab_imag * dZb_imag_dCb
            jacobian_imag[i, 4] = dZ_dZab_imag * dZb_real_dCb + dZ_dZab_real * dZb_imag_dCb

        return {
            'real': jacobian_real,
            'imag': jacobian_imag,
            'param_names': param_names
        }

    def compute_pca_directions(self, jacobian: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Compute principal directions of parameter sensitivity using SVD"""
        # Stack real and imaginary parts
        J_real = jacobian['real']
        J_imag = jacobian['imag']
        J_stacked = np.vstack([J_real, J_imag])

        # Perform SVD
        U, s, Vt = np.linalg.svd(J_stacked, full_matrices=False)

        # Principal directions are rows of Vt
        principal_directions = []
        param_names = jacobian['param_names']

        for i in range(min(3, len(s))):  # Top 3 directions
            direction = Vt[i, :]
            sensitivity = s[i]

            # Find dominant parameter
            dominant_idx = np.argmax(np.abs(direction))
            dominant_param = param_names[dominant_idx]
            dominant_contrib = np.abs(direction[dominant_idx])

            spectral_effects = ["Low-frequency magnitude", "Mid-frequency phase", "High-frequency behavior"]

            principal_directions.append({
                'rank': i + 1,
                'direction': direction,
                'sensitivity': sensitivity,
                'dominant_parameter': dominant_param,
                'dominant_contribution': dominant_contrib,
                'interpretation': f"Primary: {dominant_param} ({dominant_contrib*100:.1f}%)",
                'spectral_effect': spectral_effects[i] if i < len(spectral_effects) else f"Component {i+1}"
            })

        # Condition number
        condition_number = s[0] / s[-1] if len(s) > 1 and s[-1] > 1e-10 else np.inf

        return {
            'directions': principal_directions,
            'singular_values': s,
            'condition_number': condition_number
        }

    def generate_parameter_variations(self, config: AnalysisConfig) -> List[CircuitParameters]:
        """Generate systematic parameter variations around ground truth"""
        variations = []

        # Add ground truth as first variation
        variations.append(config.ground_truth_circuit)

        if config.grid_mode:
            # Generate grid-based variations
            print(f"   Using grid mode with {config.grid_size} points per parameter")
            grid_variations = self._generate_grid_variations(config)
            variations.extend(grid_variations)
            print(f"   Generated {len(grid_variations)} grid variations (total: {len(variations)})")
        else:
            # Generate random variations within specified ranges
            for _ in range(config.num_variations - 1):
                variation = CircuitParameters(
                    Rsh=np.random.uniform(*config.variation_ranges['Rsh']),
                    Ra=np.random.uniform(*config.variation_ranges['Ra']),
                    Ca=np.random.uniform(*config.variation_ranges['Ca']),
                    Rb=np.random.uniform(*config.variation_ranges['Rb']),
                    Cb=np.random.uniform(*config.variation_ranges['Cb']),
                    frequency_range=config.frequency_range
                )
                variations.append(variation)

        return variations

    def _generate_grid_variations(self, config: AnalysisConfig) -> List[CircuitParameters]:
        """Generate grid-based parameter variations"""
        from itertools import product

        # Create parameter grids
        param_grids = {}
        for param_name, (min_val, max_val) in config.variation_ranges.items():
            param_grids[param_name] = np.linspace(min_val, max_val, config.grid_size)

        # Generate all combinations
        grid_variations = []
        param_names = ['Rsh', 'Ra', 'Ca', 'Rb', 'Cb']

        # Limit total combinations to prevent explosion
        total_combinations = config.grid_size ** len(param_names)
        max_combinations = min(total_combinations, config.num_variations - 1)

        if total_combinations > config.num_variations - 1:
            print(f"   Warning: Grid would generate {total_combinations} combinations")
            print(f"   Sampling {max_combinations} combinations randomly")

            # Generate all grid points and sample
            all_combinations = list(product(*[param_grids[param] for param in param_names]))
            selected_combinations = np.random.choice(
                len(all_combinations),
                size=max_combinations,
                replace=False
            )

            for idx in selected_combinations:
                combo = all_combinations[idx]
                variation = CircuitParameters(
                    Rsh=combo[0],
                    Ra=combo[1],
                    Ca=combo[2],
                    Rb=combo[3],
                    Cb=combo[4],
                    frequency_range=config.frequency_range
                )
                grid_variations.append(variation)
        else:
            # Generate all combinations
            for combo in product(*[param_grids[param] for param in param_names]):
                variation = CircuitParameters(
                    Rsh=combo[0],
                    Ra=combo[1],
                    Ca=combo[2],
                    Rb=combo[3],
                    Cb=combo[4],
                    frequency_range=config.frequency_range
                )
                grid_variations.append(variation)

        return grid_variations

    def analyze_variations(self, variations: List[CircuitParameters], config: AnalysisConfig) -> pd.DataFrame:
        """Perform complete analysis on parameter variations"""
        # Generate frequency array
        frequencies = np.logspace(
            np.log10(config.frequency_range[0]),
            np.log10(config.frequency_range[1]),
            config.num_frequencies
        )

        # Calculate ground truth spectrum
        ground_truth_spectrum = self.calculate_spectrum(config.ground_truth_circuit, frequencies)

        results = []
        print(f"\n🔍 Analyzing {len(variations)} parameter variations...")

        for i, params in enumerate(variations):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(variations)} ({100*i/len(variations):.1f}%)")

            try:
                # Calculate spectrum
                spectrum = self.calculate_spectrum(params, frequencies)

                # Calculate resnorm
                resnorm = self.calculate_resnorm(spectrum, ground_truth_spectrum)

                # Compute Jacobian and PCA
                jacobian = self.compute_jacobian(params, frequencies)
                pca_result = self.compute_pca_directions(jacobian)

                # Store results
                result = {
                    'variation_id': f"variation_{i:03d}",
                    'Rsh': params.Rsh,
                    'Ra': params.Ra,
                    'Ca': params.Ca,
                    'Rb': params.Rb,
                    'Cb': params.Cb,
                    'resnorm': resnorm,
                    'condition_number': pca_result['condition_number'],
                    'pc1_sensitivity': pca_result['singular_values'][0] if len(pca_result['singular_values']) > 0 else 0,
                    'pc2_sensitivity': pca_result['singular_values'][1] if len(pca_result['singular_values']) > 1 else 0,
                    'pc3_sensitivity': pca_result['singular_values'][2] if len(pca_result['singular_values']) > 2 else 0
                }

                # Add PCA direction components
                for j, direction_data in enumerate(pca_result['directions']):
                    for k, param_name in enumerate(['Rsh', 'Ra', 'Ca', 'Rb', 'Cb']):
                        result[f'pc{j+1}_{param_name.lower()}_component'] = direction_data['direction'][k]

                results.append(result)

                # Store detailed spectrum and PCA data for later export
                if not hasattr(self, 'detailed_results'):
                    self.detailed_results = []

                self.detailed_results.append({
                    'variation_id': result['variation_id'],
                    'parameters': params,
                    'spectrum': spectrum,
                    'pca_directions': pca_result['directions'],
                    'jacobian': jacobian,
                    'frequencies': frequencies
                })

            except Exception as e:
                print(f"  Warning: Failed to analyze variation {i}: {e}")
                continue

        print(f"✅ Analysis complete: {len(results)} successful variations")
        return pd.DataFrame(results)

def get_user_config() -> AnalysisConfig:
    """Interactive configuration setup"""
    print("\n🔬 Circuit Parameter Analysis Configuration\n")

    # Ground truth circuit parameters
    print("📋 Ground Truth Circuit Parameters:")
    Rsh = float(input("  Rsh (Ω) [default: 25.88]: ") or "25.88")
    Ra = float(input("  Ra (Ω) [default: 5011.87]: ") or "5011.87")
    Ca = float(input("  Ca (F) [default: 2.69e-5]: ") or "2.69e-5")
    Rb = float(input("  Rb (Ω) [default: 5011.87]: ") or "5011.87")
    Cb = float(input("  Cb (F) [default: 2.69e-5]: ") or "2.69e-5")

    # Variation ranges
    print("\n📊 Parameter Variation Ranges (±percentage):")
    rsh_var = float(input("  Rsh variation (%) [default: 20]: ") or "20") / 100
    ra_var = float(input("  Ra variation (%) [default: 15]: ") or "15") / 100
    ca_var = float(input("  Ca variation (%) [default: 25]: ") or "25") / 100
    rb_var = float(input("  Rb variation (%) [default: 15]: ") or "15") / 100
    cb_var = float(input("  Cb variation (%) [default: 25]: ") or "25") / 100

    # Analysis parameters
    print("\n⚙️ Analysis Parameters:")
    num_variations = int(input("  Number of variations [default: 100]: ") or "100")
    min_freq = float(input("  Min frequency (Hz) [default: 1]: ") or "1")
    max_freq = float(input("  Max frequency (Hz) [default: 10000]: ") or "10000")
    num_freq = int(input("  Number of frequency points [default: 50]: ") or "50")

    # Grid mode parameters
    print("\n🔳 Parameter Space Sampling:")
    grid_mode = input("  Use grid sampling instead of random? (y/N) [default: N]: ").lower().startswith('y')
    grid_size = 5
    if grid_mode:
        grid_size = int(input("  Grid points per parameter [default: 5]: ") or "5")
        total_grid_points = grid_size ** 5
        print(f"  Warning: Grid mode will generate up to {total_grid_points} parameter combinations")
        if total_grid_points > num_variations:
            print(f"  Will sample {num_variations} random combinations from the grid")

    # Output configuration
    print("\n📁 Output Configuration:")
    run_name = input("  Run name [default: circuit_analysis]: ") or "circuit_analysis"
    output_dir = input("  Output directory [default: ./analysis_output]: ") or "./analysis_output"

    ground_truth = CircuitParameters(Rsh, Ra, Ca, Rb, Cb, (min_freq, max_freq))

    return AnalysisConfig(
        ground_truth_circuit=ground_truth,
        variation_ranges={
            'Rsh': (Rsh * (1 - rsh_var), Rsh * (1 + rsh_var)),
            'Ra': (Ra * (1 - ra_var), Ra * (1 + ra_var)),
            'Ca': (Ca * (1 - ca_var), Ca * (1 + ca_var)),
            'Rb': (Rb * (1 - rb_var), Rb * (1 + rb_var)),
            'Cb': (Cb * (1 - cb_var), Cb * (1 + cb_var))
        },
        num_variations=num_variations,
        frequency_range=(min_freq, max_freq),
        num_frequencies=num_freq,
        output_dir=output_dir,
        run_name=run_name,
        grid_mode=grid_mode,
        grid_size=grid_size
    )

def setup_output_directory(config: AnalysisConfig) -> Path:
    """Create organized output directory structure"""
    timestamp = datetime.now().strftime("%Y-%m-%d")
    output_path = Path(config.output_dir) / f"{config.run_name}_{timestamp}"

    # Create directories
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "csv").mkdir(exist_ok=True)
    (output_path / "visualizations").mkdir(exist_ok=True)
    (output_path / "analysis").mkdir(exist_ok=True)

    return output_path

def export_data(analyzer: CircuitAnalyzer, results_df: pd.DataFrame, output_path: Path, config: AnalysisConfig):
    """Export analysis results to CSV files"""
    print("\n💾 Exporting data files...")

    # Export main parameters and results
    results_df.to_csv(output_path / "csv" / "parameters.csv", index=False)
    print("   ✅ parameters.csv")

    # Export detailed spectra
    spectra_data = []
    pca_directions_data = []

    if hasattr(analyzer, 'detailed_results'):
        for detail in analyzer.detailed_results:
            # Spectrum data
            spectrum_df = detail['spectrum'].copy()
            spectrum_df['variation_id'] = detail['variation_id']
            spectra_data.append(spectrum_df)

            # PCA directions data
            for direction in detail['pca_directions']:
                pca_row = {
                    'variation_id': detail['variation_id'],
                    'direction_rank': direction['rank'],
                    'sensitivity': direction['sensitivity'],
                    'interpretation': direction['interpretation'],
                    'spectral_effect': direction['spectral_effect']
                }
                # Add component values
                for i, param in enumerate(['Rsh', 'Ra', 'Ca', 'Rb', 'Cb']):
                    pca_row[f'{param}_component'] = direction['direction'][i]

                pca_directions_data.append(pca_row)

    if spectra_data:
        spectra_df = pd.concat(spectra_data, ignore_index=True)
        spectra_df.to_csv(output_path / "csv" / "spectra.csv", index=False)
        print("   ✅ spectra.csv")

    if pca_directions_data:
        pca_df = pd.DataFrame(pca_directions_data)
        pca_df.to_csv(output_path / "csv" / "pca_directions.csv", index=False)
        print("   ✅ pca_directions.csv")

    # Save configuration
    config_dict = {
        'ground_truth_circuit': {
            'Rsh': config.ground_truth_circuit.Rsh,
            'Ra': config.ground_truth_circuit.Ra,
            'Ca': config.ground_truth_circuit.Ca,
            'Rb': config.ground_truth_circuit.Rb,
            'Cb': config.ground_truth_circuit.Cb,
            'frequency_range': config.ground_truth_circuit.frequency_range
        },
        'variation_ranges': config.variation_ranges,
        'num_variations': config.num_variations,
        'frequency_range': config.frequency_range,
        'num_frequencies': config.num_frequencies,
        'run_name': config.run_name,
        'grid_mode': config.grid_mode,
        'grid_size': config.grid_size,
        'timestamp': datetime.now().isoformat()
    }

    with open(output_path / "analysis_config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Circuit Parameter Analysis with PCA")
    parser.add_argument("--config", help="JSON config file (optional)")
    parser.add_argument("--batch", action="store_true", help="Run in batch mode with default settings")
    args = parser.parse_args()

    print("🧪 Circuit Parameter Analysis Tool")
    print("=" * 50)

    try:
        # Get configuration
        if args.config and os.path.exists(args.config):
            with open(args.config, 'r') as f:
                config_dict = json.load(f)
            # Convert back to AnalysisConfig object (implementation depends on structure)
            config = get_user_config()  # Fallback for now
        elif args.batch:
            # Use default configuration for batch processing
            ground_truth = CircuitParameters(25.88, 5011.87, 2.69e-5, 5011.87, 2.69e-5)
            config = AnalysisConfig(
                ground_truth_circuit=ground_truth,
                variation_ranges={
                    'Rsh': (20.7, 31.06), 'Ra': (4260.09, 5763.65),
                    'Ca': (2.02e-5, 3.36e-5), 'Rb': (4260.09, 5763.65),
                    'Cb': (2.02e-5, 3.36e-5)
                },
                num_variations=100,
                frequency_range=(1.0, 10000.0),
                num_frequencies=50,
                output_dir="./analysis_output",
                run_name="circuit_analysis"
            )
        else:
            config = get_user_config()

        # Setup output directory
        print("\n📁 Setting up output directory...")
        output_path = setup_output_directory(config)
        print(f"   Output directory: {output_path}")

        # Initialize analyzer
        analyzer = CircuitAnalyzer()

        # Generate parameter variations
        print("\n🎲 Generating parameter variations...")
        variations = analyzer.generate_parameter_variations(config)
        print(f"   Generated {len(variations)} parameter sets")

        # Perform analysis
        results_df = analyzer.analyze_variations(variations, config)

        # Export results
        export_data(analyzer, results_df, output_path, config)

        # Save detailed results for Jacobian visualization
        try:
            import pickle
            detailed_results_file = output_path / "analysis" / "detailed_results.pkl"
            with open(detailed_results_file, 'wb') as f:
                pickle.dump(analyzer.detailed_results, f)
            print("   ✅ detailed_results.pkl (for Jacobian analysis)")
        except Exception as e:
            print(f"   ⚠️ Could not save detailed results: {e}")

        print(f"\n🎉 Analysis complete! Results saved to: {output_path}")
        print("\nNext steps:")
        print("  1. Run visualization script: python scripts/create_visualizations.py")
        print(f"  2. Check results in: {output_path}")

        # Print summary statistics
        print(f"\n📊 Summary Statistics:")
        print(f"   Resnorm range: {results_df['resnorm'].min():.6f} - {results_df['resnorm'].max():.6f}")
        print(f"   Mean condition number: {results_df['condition_number'].mean():.2f}")
        print(f"   PC1 sensitivity range: {results_df['pc1_sensitivity'].min():.2f} - {results_df['pc1_sensitivity'].max():.2f}")

    except KeyboardInterrupt:
        print("\n\n⏹️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()