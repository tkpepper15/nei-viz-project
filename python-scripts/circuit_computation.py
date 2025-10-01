#!/usr/bin/env python3
"""
EIS Circuit Computation - Python Implementation
Implements the same Randles circuit model as the JavaScript version
for debugging and validation purposes.

Run in PyCharm terminal: python circuit_computation.py
"""

import numpy as np
import itertools
import time
import os
from typing import List, Dict, Tuple, NamedTuple
from dataclasses import dataclass

@dataclass
class CircuitParameters:
    """Circuit parameters for Randles model"""
    Rsh: float  # Shunt resistance (Ω)
    Ra: float   # Apical resistance (Ω)
    Ca: float   # Apical capacitance (F)
    Rb: float   # Basal resistance (Ω)
    Cb: float   # Basal capacitance (F)

@dataclass
class ImpedancePoint:
    """Single impedance measurement point"""
    frequency: float
    real: float
    imaginary: float
    magnitude: float
    phase: float

class CircuitSimulator:
    """Main circuit simulation class"""
    
    def __init__(self):
        self.results = []
        
    def calculate_impedance(self, params: CircuitParameters, frequency: float) -> complex:
        """
        Calculate impedance for Randles circuit at given frequency.
        Uses parallel combination: Z_total = (Rsh * (Za + Zb)) / (Rsh + Za + Zb)
        """
        omega = 2 * np.pi * frequency
        
        # Za = Ra/(1+jωRaCa)  
        za_denom = 1 + 1j * omega * params.Ra * params.Ca
        za = params.Ra / za_denom
        
        # Zb = Rb/(1+jωRbCb)  
        zb_denom = 1 + 1j * omega * params.Rb * params.Cb
        zb = params.Rb / zb_denom
        
        # Calculate sum of membrane impedances (Za + Zb)
        zab = za + zb
        
        # Calculate parallel combination: Z_total = (Rsh * (Za + Zb)) / (Rsh + Za + Zb)
        numerator = params.Rsh * zab
        denominator = params.Rsh + zab
        
        z_total = numerator / denominator
        return z_total
    
    def calculate_spectrum(self, params: CircuitParameters, frequencies: np.ndarray) -> List[ImpedancePoint]:
        """Calculate impedance spectrum across frequency range"""
        spectrum = []
        
        for freq in frequencies:
            z = self.calculate_impedance(params, freq)
            magnitude = abs(z)
            phase = np.angle(z) * (180 / np.pi)
            
            point = ImpedancePoint(
                frequency=freq,
                real=z.real,
                imaginary=z.imag,
                magnitude=magnitude,
                phase=phase
            )
            spectrum.append(point)
            
        return spectrum
    
    def calculate_resnorm_ssr(self, test_spectrum: List[ImpedancePoint],
                             ref_spectrum: List[ImpedancePoint]) -> float:
        """
        Calculate residual norm using Sum of Squared Residuals (SSR) method.

        SSR = (1/N) * Σ√[(Z_real,test - Z_real,ref)² + (Z_imag,test - Z_imag,ref)²]

        This cost function:
        - Uses real and imaginary component differences directly
        - Penalizes large errors more heavily (quadratic distance in complex plane)
        - Better for ML optimization (smooth, differentiable)
        - Standard approach in EIS parameter extraction research
        """
        if not test_spectrum or not ref_spectrum:
            return float('inf')

        ssr = 0.0
        valid_points = 0

        min_len = min(len(test_spectrum), len(ref_spectrum))

        for i in range(min_len):
            ref_point = ref_spectrum[i]
            test_point = test_spectrum[i]

            # Check frequency alignment (within 0.1% tolerance)
            if abs(ref_point.frequency - test_point.frequency) / ref_point.frequency > 0.001:
                continue

            # Calculate reference magnitude
            ref_mag = abs(complex(ref_point.real, ref_point.imaginary))

            # Skip points with zero reference magnitude
            if ref_mag == 0:
                continue

            # Calculate real and imaginary component differences
            real_diff = test_point.real - ref_point.real
            imag_diff = test_point.imaginary - ref_point.imaginary

            # Complex magnitude error: sqrt((real_diff)² + (imag_diff)²)
            complex_magnitude_error = np.sqrt(real_diff ** 2 + imag_diff ** 2)

            # Sum of squared residuals
            ssr += complex_magnitude_error
            valid_points += 1

        if valid_points == 0:
            return float('inf')

        # Return mean (normalized by number of points)
        return ssr / valid_points
    
    def generate_parameter_grid(self, grid_size: int) -> List[CircuitParameters]:
        """Generate parameter grid matching JavaScript implementation"""
        print(f"🔧 Generating parameter grid (size: {grid_size})")
        
        # Use canonical parameter ranges (matching parameterRanges.ts)
        rsh_values = np.logspace(np.log10(10), np.log10(10000), grid_size)
        ra_values = np.logspace(np.log10(10), np.log10(10000), grid_size) 
        ca_values = np.logspace(np.log10(0.1e-6), np.log10(50e-6), grid_size)
        rb_values = np.logspace(np.log10(10), np.log10(10000), grid_size)
        cb_values = np.logspace(np.log10(0.1e-6), np.log10(50e-6), grid_size)
        
        grid_points = []
        total_combinations = grid_size ** 5
        
        print(f"📊 Total parameter combinations: {total_combinations:,}")
        
        count = 0
        start_time = time.time()
        
        for rsh, ra, ca, rb, cb in itertools.product(rsh_values, ra_values, ca_values, rb_values, cb_values):
            params = CircuitParameters(
                Rsh=float(rsh),
                Ra=float(ra),
                Ca=float(ca),
                Rb=float(rb),
                Cb=float(cb)
            )
            grid_points.append(params)
            count += 1
            
            # Progress reporting
            if count % 5000 == 0 or count == total_combinations:
                elapsed = time.time() - start_time
                percent = (count / total_combinations) * 100
                rate = count / elapsed if elapsed > 0 else 0
                print(f"⏳ Generated {count:,}/{total_combinations:,} ({percent:.1f}%) - {rate:.0f} params/sec")
        
        print(f"✅ Parameter grid complete: {len(grid_points):,} combinations")
        return grid_points
    
    def run_computation(self, grid_size: int = 5,
                       min_freq: float = 1.0, max_freq: float = 10000.0, num_points: int = 20,
                       ref_params: CircuitParameters = None):
        """Run the complete circuit computation"""
        print("🚀 Starting EIS Circuit Computation")
        print("=" * 50)
        
        # Use provided reference parameters or default
        if ref_params is None:
            ref_params = CircuitParameters(
                Rsh=5005.0,      # Shunt resistance (Ω) - from user config
                Ra=5005.0,       # Apical resistance (Ω) - from user config  
                Ca=0.00002505,   # Apical capacitance (F) - from user config
                Rb=5005.0,       # Basal resistance (Ω) - from user config
                Cb=0.00002505    # Basal capacitance (F) - from user config
            )
        
        # Generate frequency range (matching user's JavaScript config)
        frequencies = np.logspace(np.log10(min_freq), np.log10(max_freq), num_points)
        
        print(f"📈 Frequency range: {frequencies[0]:.1f} - {frequencies[-1]:.1f} Hz ({len(frequencies)} points)")
        print(f"🎯 Reference parameters:")
        print(f"   Rsh: {ref_params.Rsh} Ω")
        print(f"   Ra: {ref_params.Ra} Ω, Ca: {ref_params.Ca*1e6:.3f} µF")
        print(f"   Rb: {ref_params.Rb} Ω, Cb: {ref_params.Cb*1e6:.3f} µF")
        
        # Calculate reference spectrum
        print("\n🔬 Calculating reference spectrum...")
        ref_spectrum = self.calculate_spectrum(ref_params, frequencies)
        
        # Generate parameter grid
        print("\n⚙️ Generating parameter combinations...")
        param_grid = self.generate_parameter_grid(grid_size)
        
        # Run computation on parameter grid
        print(f"\n🧮 Computing impedance spectra and resnorms...")
        results = []
        start_time = time.time()
        
        for i, params in enumerate(param_grid):
            # Calculate spectrum for this parameter set
            test_spectrum = self.calculate_spectrum(params, frequencies)

            # Calculate resnorm vs reference using SSR (Sum of Squared Residuals)
            resnorm = self.calculate_resnorm_ssr(test_spectrum, ref_spectrum)
            
            result = {
                'parameters': params,
                'spectrum': test_spectrum,
                'resnorm': resnorm
            }
            results.append(result)
            
            # Progress reporting
            if (i + 1) % 1000 == 0 or i == len(param_grid) - 1:
                elapsed = time.time() - start_time
                percent = ((i + 1) / len(param_grid)) * 100
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"⚡ Processed {i + 1:,}/{len(param_grid):,} ({percent:.1f}%) - {rate:.0f} params/sec")
        
        # Sort results by resnorm
        print("\n📊 Sorting results by resnorm...")
        results.sort(key=lambda x: x['resnorm'])
        
        # Display top results
        print(f"\n🏆 Top {min(10, len(results))} Results:")
        print("-" * 80)
        print(f"{'Rank':<5} {'Resnorm':<12} {'Rsh':<8} {'Ra':<8} {'Ca(µF)':<8} {'Rb':<8} {'Cb(µF)':<8}")
        print("-" * 80)
        
        for i, result in enumerate(results[:min(10, len(results))]):
            params = result['parameters']
            resnorm = result['resnorm']
            print(f"{i+1:<5} {resnorm:<12.6f} {params.Rsh:<8.1f} {params.Ra:<8.1f} "
                  f"{params.Ca*1e6:<8.2f} {params.Rb:<8.1f} {params.Cb*1e6:<8.2f}")
        
        # Summary statistics
        valid_results = [r for r in results if np.isfinite(r['resnorm'])]
        print(f"\n📈 Computation Summary:")
        print(f"   Total parameter sets: {len(results):,}")
        print(f"   Valid results: {len(valid_results):,}")
        print(f"   Best resnorm: {valid_results[0]['resnorm']:.6f}")
        print(f"   Worst resnorm: {valid_results[-1]['resnorm']:.6f}")
        print(f"   Computation time: {time.time() - start_time:.2f} seconds")
        
        # Export to ultra-efficient NPZ format following Supabase-compatible naming convention
        # Format: grid_{size}_freq_{min}_{max}.npz for automatic discovery
        npz_filename = f"grid_{grid_size}_freq_{min_freq}_{max_freq}.npz"
        npz_path = f"data/npz/user_generated/{npz_filename}"
        
        # Ensure directory exists with proper structure
        os.makedirs("data/npz/user_generated", exist_ok=True)
        
        self.export_to_npz(results, npz_path, frequencies)
        
        return results
    
    def export_to_npz(self, results: List[Dict], filename: str, frequencies: np.ndarray):
        """Ultra-efficient NPZ export - perfect for web backend integration"""
        print(f"\n⚡ Exporting to NPZ: {filename}")
        start_time = time.time()
        
        n_results = len(results)
        n_freqs = len(frequencies)
        
        print(f"📊 Dataset: {n_results:,} parameter sets × {n_freqs} frequencies")
        
        # Pre-allocate arrays for maximum efficiency
        parameters = np.zeros((n_results, 5), dtype=np.float32)  # Rsh, Ra, Ca, Rb, Cb
        resnorms = np.zeros(n_results, dtype=np.float32)
        spectrum = np.zeros((n_results, n_freqs, 4), dtype=np.float32)  # real, imag, mag, phase
        
        # Vectorized data extraction
        for i, result in enumerate(results):
            params = result['parameters']
            parameters[i] = [params.Rsh, params.Ra, params.Ca, params.Rb, params.Cb]
            resnorms[i] = result['resnorm']
            
            # Extract spectrum data efficiently
            for j, point in enumerate(result['spectrum']):
                spectrum[i, j] = [point.real, point.imaginary, point.magnitude, point.phase]
        
        # Create comprehensive metadata for Supabase compatibility
        grid_size_calculated = int(round(n_results**(1/5)))  # More accurate grid size calculation
        metadata = {
            # Core dataset info
            'grid_size': grid_size_calculated,
            'n_parameter_sets': n_results,
            'n_frequencies': n_freqs,
            'freq_min': float(frequencies[0]),
            'freq_max': float(frequencies[-1]),
            
            # Schema compatibility
            'param_names': ['Rsh', 'Ra', 'Ca', 'Rb', 'Cb'],
            'param_units': ['ohm', 'ohm', 'farad', 'ohm', 'farad'],
            'spectrum_columns': ['real', 'imag', 'magnitude', 'phase'],
            
            # Computation metadata
            'export_timestamp': time.time(),
            'best_resnorm': float(np.min(resnorms)),
            'computation_complete': True,
            'user_generated': True,
            
            # Auto-registration hints for Supabase sync
            'storage_location': 'user_generated',
            'filename': filename.split('/')[-1],
            'ready_for_registration': True
        }
        
        # Save with maximum compression
        np.savez_compressed(
            filename,
            parameters=parameters,
            resnorms=resnorms,
            spectrum=spectrum,
            frequencies=frequencies.astype(np.float32),
            metadata=np.array([metadata], dtype=object)[0]  # Pickle metadata
        )
        
        # File stats
        file_size = os.path.getsize(filename) / (1024**2)  # MB
        compression_ratio = (n_results * n_freqs * 8 * 8) / (1024**2) / file_size  # Rough estimate
        
        print(f"✅ NPZ export complete!")
        print(f"   File size: {file_size:.1f} MB")
        print(f"   Compression ratio: ~{compression_ratio:.1f}:1")
        print(f"   Export time: {time.time() - start_time:.2f}s")
        print(f"   Web-ready: parameters[{parameters.shape}], spectrum[{spectrum.shape}]")
        
        return filename

def main():
    """Main execution function"""
    simulator = CircuitSimulator()
    
    # USER'S EXACT CONFIGURATION from JavaScript app
    config = {
        "profileName": "17 Grid Size High Storage Test",
        "gridSettings": {
            "gridSize": 5,     # 17^5 = 1.4M combinations - good balance for testing
            "minFreq": 0.1,     # 0.1 Hz minimum frequency
            "maxFreq": 100000,  # 100 kHz maximum frequency  
            "numPoints": 100    # 100 frequency points
        },
        "circuitParameters": {
            "Rsh": 5005,
            "Ra": 5005,
            "Ca": 0.00002505,   # 25.05 µF
            "Rb": 5005, 
            "Cb": 0.00002505,   # 25.05 µF
            "frequency_range": [0.1, 100000]
        }
    }
    
    print("🔧 Using exact configuration from JavaScript application:")
    print(f"   Profile: {config['profileName']}")
    print(f"   Grid size: {config['gridSettings']['gridSize']} (will generate {config['gridSettings']['gridSize']**5:,} combinations)")
    print(f"   Frequency range: {config['gridSettings']['minFreq']} - {config['gridSettings']['maxFreq']} Hz")
    print(f"   Frequency points: {config['gridSettings']['numPoints']}")
    print()
    
    # Create reference parameters from config
    ref_params = CircuitParameters(
        Rsh=config['circuitParameters']['Rsh'],
        Ra=config['circuitParameters']['Ra'], 
        Ca=config['circuitParameters']['Ca'],
        Rb=config['circuitParameters']['Rb'],
        Cb=config['circuitParameters']['Cb']
    )
    
    # Run computation with exact user configuration
    results = simulator.run_computation(
        grid_size=config['gridSettings']['gridSize'],
        min_freq=config['gridSettings']['minFreq'],
        max_freq=config['gridSettings']['maxFreq'],
        num_points=config['gridSettings']['numPoints'],
        ref_params=ref_params
    )
    
    print(f"\n✅ Computation complete! Generated {len(results)} results.")
    npz_filename = f"grid_{config['gridSettings']['gridSize']}_freq_{config['gridSettings']['minFreq']}_{config['gridSettings']['maxFreq']}.npz"
    print(f"\nNPZ File created:")
    print(f"  🗜️ data/npz/user_generated/{npz_filename} - Complete dataset with all {len(results):,} results and {config['gridSettings']['numPoints']} frequency points")
    
    # Show what JavaScript should be computing
    total_combinations = config['gridSettings']['gridSize'] ** 5
    print(f"\n🎯 JavaScript Application Should Show:")
    print(f"   Grid points generated: {total_combinations:,} parameters")
    print(f"   Final computation results: {total_combinations:,} models (or limited by maxComputationResults)")
    print(f"   Frequency spectrum: {config['gridSettings']['numPoints']} points from {config['gridSettings']['minFreq']:.1f} to {config['gridSettings']['maxFreq']:,} Hz")

if __name__ == "__main__":
    main()