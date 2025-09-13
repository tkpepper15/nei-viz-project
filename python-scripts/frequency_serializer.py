"""
Frequency Measurement Configuration Serializer
==============================================

Handles frequency sweep settings with compact serialization using scientific notation.
Designed for efficient storage and easy reconstruction of frequency arrays.

Usage:
    freq_config = FrequencyConfig(min_freq=1e-1, max_freq=1e5, n_points=100, spacing="log")
    freq_id = freq_config.to_id()  # "L_1.0E-01_1.0E+05_100"
    reconstructed = FrequencyConfig.from_id(freq_id)
"""

import numpy as np
import json
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import re


@dataclass
class FrequencyConfig:
    """Compact frequency configuration with scientific notation serialization"""
    
    min_freq: float       # Minimum frequency (Hz) 
    max_freq: float       # Maximum frequency (Hz)
    n_points: int         # Number of frequency points
    spacing: str = "log"  # "log" or "linear" spacing
    
    def __post_init__(self):
        """Validate frequency configuration"""
        if self.min_freq <= 0:
            raise ValueError("Minimum frequency must be positive")
        if self.max_freq <= self.min_freq:
            raise ValueError("Maximum frequency must be greater than minimum")
        if self.n_points < 2:
            raise ValueError("Number of points must be at least 2")
        if self.spacing not in ["log", "linear"]:
            raise ValueError("Spacing must be 'log' or 'linear'")
    
    def to_id(self) -> str:
        """Convert to compact ID string with scientific notation"""
        spacing_code = "L" if self.spacing == "log" else "N"  # L=log, N=linear
        min_sci = f"{self.min_freq:.1E}"
        max_sci = f"{self.max_freq:.1E}"
        return f"{spacing_code}_{min_sci}_{max_sci}_{self.n_points:03d}"
    
    @classmethod
    def from_id(cls, freq_id: str) -> 'FrequencyConfig':
        """Parse from compact ID string"""
        parts = freq_id.split('_')
        if len(parts) != 4:
            raise ValueError(f"Invalid frequency ID format: {freq_id}")
        
        spacing_code, min_sci, max_sci, n_points_str = parts
        
        # Parse spacing
        spacing = "log" if spacing_code == "L" else "linear"
        
        # Parse scientific notation frequencies
        min_freq = float(min_sci)
        max_freq = float(max_sci)
        n_points = int(n_points_str)
        
        return cls(min_freq, max_freq, n_points, spacing)
    
    def generate_frequencies(self) -> np.ndarray:
        """Generate frequency array based on configuration"""
        if self.spacing == "log":
            return np.logspace(np.log10(self.min_freq), np.log10(self.max_freq), self.n_points)
        else:
            return np.linspace(self.min_freq, self.max_freq, self.n_points)
    
    def get_frequency_info(self) -> Dict:
        """Get detailed frequency information"""
        frequencies = self.generate_frequencies()
        
        return {
            "id": self.to_id(),
            "config": asdict(self),
            "frequencies": {
                "first": f"{frequencies[0]:.2E} Hz",
                "last": f"{frequencies[-1]:.2E} Hz", 
                "center": f"{frequencies[len(frequencies)//2]:.2E} Hz",
                "step_size": f"{frequencies[1] - frequencies[0]:.2E} Hz" if self.spacing == "linear" else None,
                "decade_span": f"{np.log10(self.max_freq / self.min_freq):.1f} decades" if self.spacing == "log" else None,
                "points_per_decade": f"{self.n_points / np.log10(self.max_freq / self.min_freq):.1f}" if self.spacing == "log" else None
            },
            "array_preview": {
                "first_5": [f"{f:.2E}" for f in frequencies[:5]],
                "last_5": [f"{f:.2E}" for f in frequencies[-5:]]
            }
        }


class FrequencySerializer:
    """Manages frequency configuration presets and serialization"""
    
    def __init__(self):
        self.presets = self._create_standard_presets()
    
    def _create_standard_presets(self) -> Dict[str, FrequencyConfig]:
        """Create standard frequency measurement presets"""
        return {
            "standard": FrequencyConfig(1e-1, 1e5, 100, "log"),           # L_1.0E-01_1.0E+05_100
            "high_res": FrequencyConfig(1e-2, 1e6, 500, "log"),           # L_1.0E-02_1.0E+06_500  
            "fast": FrequencyConfig(1e0, 1e4, 50, "log"),                 # L_1.0E+00_1.0E+04_050
            "low_freq": FrequencyConfig(1e-3, 1e2, 200, "log"),           # L_1.0E-03_1.0E+02_200
            "linear": FrequencyConfig(1e1, 1e3, 100, "linear"),           # N_1.0E+01_1.0E+03_100
            "ultra_wide": FrequencyConfig(1e-4, 1e7, 1000, "log"),        # L_1.0E-04_1.0E+07_1000
            "mid_range": FrequencyConfig(1e1, 1e5, 250, "log"),           # L_1.0E+01_1.0E+05_250
        }
    
    def get_preset(self, name: str) -> Optional[FrequencyConfig]:
        """Get a preset frequency configuration"""
        return self.presets.get(name)
    
    def list_presets(self) -> List[str]:
        """List all available preset names"""
        return list(self.presets.keys())
    
    def get_preset_info(self, name: str) -> Optional[Dict]:
        """Get detailed information about a preset"""
        preset = self.get_preset(name)
        if preset is None:
            return None
        return preset.get_frequency_info()
    
    def create_custom_config(self, min_freq: float, max_freq: float, 
                           n_points: int, spacing: str = "log") -> FrequencyConfig:
        """Create a custom frequency configuration"""
        return FrequencyConfig(min_freq, max_freq, n_points, spacing)
    
    def register_preset(self, name: str, config: FrequencyConfig):
        """Register a new preset"""
        self.presets[name] = config
    
    def save_presets(self, filepath: str):
        """Save all presets to JSON file"""
        data = {
            name: {
                "id": config.to_id(),
                "config": asdict(config),
                "info": config.get_frequency_info()
            }
            for name, config in self.presets.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


@dataclass
class ComputationResult:
    """Complete computation result linking circuit config, frequency config, and resnorm"""
    
    circuit_config_id: str      # e.g., "15_01_02_04_03_05" 
    frequency_config_id: str    # e.g., "L_1.0E-01_1.0E+05_100"
    resnorm: float              # Computed residual norm
    computation_time: Optional[float] = None
    timestamp: Optional[str] = None
    
    def to_compact_string(self) -> str:
        """Create ultra-compact representation"""
        return f"{self.circuit_config_id}|{self.frequency_config_id}|{self.resnorm:.6E}"
    
    @classmethod
    def from_compact_string(cls, compact_str: str) -> 'ComputationResult':
        """Parse from compact string"""
        parts = compact_str.split('|')
        if len(parts) != 3:
            raise ValueError(f"Invalid compact string format: {compact_str}")
        
        circuit_id, freq_id, resnorm_str = parts
        return cls(circuit_id, freq_id, float(resnorm_str))


def demonstrate_frequency_serialization():
    """Demonstrate the frequency serialization system"""
    
    print("🔬 Frequency Configuration Serialization Demo")
    print("=" * 60)
    
    # Create serializer
    serializer = FrequencySerializer()
    
    print("\n📊 Standard Frequency Presets:")
    print("-" * 40)
    
    for name in serializer.list_presets():
        info = serializer.get_preset_info(name)
        print(f"{name.upper()}:")
        print(f"  ID: {info['id']}")
        print(f"  Range: {info['frequencies']['first']} to {info['frequencies']['last']}")
        print(f"  Points: {info['config']['n_points']} ({info['config']['spacing']} spacing)")
        if info['frequencies']['decade_span']:
            print(f"  Span: {info['frequencies']['decade_span']}")
        print()
    
    print("🎯 Sample Circuit with All Associated Values:")
    print("=" * 60)
    
    # Import circuit serializer
    from config_serializer import ConfigSerializer, ConfigId
    
    # Create sample circuit
    circuit_serializer = ConfigSerializer(grid_size=15)
    circuit_config = circuit_serializer.serialize_config(2, 3, 1, 4, 5)
    circuit_params = circuit_serializer.deserialize_config(circuit_config)
    
    # Create sample frequency config
    freq_config = serializer.get_preset("standard")
    frequencies = freq_config.generate_frequencies()
    
    # Simulate resnorm computation
    np.random.seed(42)
    sample_resnorm = np.random.exponential(1.0)
    
    # Create computation result
    result = ComputationResult(
        circuit_config_id=circuit_config.to_string(),
        frequency_config_id=freq_config.to_id(),
        resnorm=sample_resnorm
    )
    
    print("SAMPLE CIRCUIT CONFIGURATION:")
    print(f"  Circuit ID: {circuit_config.to_string()}")
    print(f"  Circuit Parameters:")
    print(f"    Rsh: {circuit_params.rsh:.0f} Ω")
    print(f"    Ra:  {circuit_params.ra:.0f} Ω") 
    print(f"    Ca:  {circuit_params.ca*1e6:.1f} μF")
    print(f"    Rb:  {circuit_params.rb:.0f} Ω")
    print(f"    Cb:  {circuit_params.cb*1e6:.1f} μF")
    
    print(f"\nFREQUENCY MEASUREMENT CONFIG:")
    print(f"  Frequency ID: {freq_config.to_id()}")
    print(f"  Range: {freq_config.min_freq:.1E} to {freq_config.max_freq:.1E} Hz")
    print(f"  Points: {freq_config.n_points} ({freq_config.spacing} spacing)")
    print(f"  Decade Span: {np.log10(freq_config.max_freq / freq_config.min_freq):.1f}")
    
    print(f"\nCOMPUTATION RESULT:")
    print(f"  Resnorm: {result.resnorm:.6E}")
    print(f"  Compact String: {result.to_compact_string()}")
    
    print(f"\nFREQUENCY ARRAY SAMPLE:")
    print(f"  First 5: {[f'{f:.2E}' for f in frequencies[:5]]}")
    print(f"  Last 5:  {[f'{f:.2E}' for f in frequencies[-5:]]}")
    print(f"  Total Array Size: {len(frequencies)} points")
    
    print("\n🚀 Rendering Applications:")
    print("-" * 40)
    print("NYQUIST PLOT:")
    print(f"  • Uses circuit config: {circuit_config.to_string()}")
    print(f"  • Regenerates frequencies from: {freq_config.to_id()}")
    print(f"  • Calculates impedance points on-demand")
    print(f"  • Only stores config IDs + resnorm, not full arrays")
    
    print(f"\nSPIDER PLOT:")
    print(f"  • Uses circuit config: {circuit_config.to_string()}")
    print(f"  • Associated resnorm: {result.resnorm:.6E}")
    print(f"  • No frequency data needed for visualization")
    print(f"  • Enables filtering (e.g., lowest Ra circuits)")
    
    print(f"\nSTORAGE EFFICIENCY:")
    freq_array_size = len(frequencies) * 8  # 8 bytes per float64
    impedance_array_size = len(frequencies) * 2 * 8  # Complex numbers
    total_traditional = freq_array_size + impedance_array_size
    serialized_size = len(result.to_compact_string())
    
    print(f"  Traditional: {total_traditional:,} bytes (freq + impedance arrays)")
    print(f"  Serialized:  {serialized_size} bytes (config IDs + resnorm)")
    print(f"  Reduction:   {total_traditional/serialized_size:.0f}x smaller")
    
    return result, circuit_config, freq_config


def demo_filtering_scenarios():
    """Demonstrate filtering scenarios for spider plots"""
    
    print("\n" + "=" * 60)
    print("🎯 Filtering Scenarios for Spider Plots")
    print("=" * 60)
    
    from config_serializer import ConfigSerializer
    
    # Create test data
    circuit_serializer = ConfigSerializer(grid_size=10)
    freq_serializer = FrequencySerializer()
    freq_config = freq_serializer.get_preset("standard")
    
    # Generate sample results
    np.random.seed(123)
    sample_results = []
    
    for _ in range(20):
        # Random circuit config
        indices = np.random.randint(1, 11, size=5)
        circuit_config = circuit_serializer.serialize_config(*indices)
        circuit_params = circuit_serializer.deserialize_config(circuit_config)
        
        # Simulate resnorm based on parameters (lower Ra = better performance)
        synthetic_resnorm = 1.0 / (circuit_params.ra * circuit_params.ca * 1e6)
        resnorm = synthetic_resnorm + np.random.normal(0, synthetic_resnorm * 0.1)
        
        result = ComputationResult(
            circuit_config_id=circuit_config.to_string(),
            frequency_config_id=freq_config.to_id(),
            resnorm=resnorm
        )
        
        sample_results.append((result, circuit_config, circuit_params))
    
    # Sort by performance
    sample_results.sort(key=lambda x: x[0].resnorm)
    
    print("SAMPLE FILTERING RESULTS:")
    print("Top 10 circuits (lowest resnorm):")
    print("-" * 50)
    
    for i, (result, circuit_config, params) in enumerate(sample_results[:10]):
        print(f"{i+1:2d}. Config: {circuit_config.to_string()}")
        print(f"    Resnorm: {result.resnorm:.6E}")
        print(f"    Ra: {params.ra:.0f}Ω, Ca: {params.ca*1e6:.1f}μF")
        print(f"    Compact: {result.to_compact_string()}")
        print()
    
    print("🔍 Filter Examples:")
    print("-" * 30)
    
    # Filter by Ra range
    low_ra_circuits = [r for r in sample_results if r[2].ra < 100]
    print(f"Low Ra circuits (< 100Ω): {len(low_ra_circuits)}")
    
    # Filter by performance
    good_performers = [r for r in sample_results if r[0].resnorm < 0.001]
    print(f"Good performers (resnorm < 0.001): {len(good_performers)}")
    
    # Filter by capacitance
    high_cap_circuits = [r for r in sample_results if r[2].ca > 5e-6]
    print(f"High capacitance (> 5μF): {len(high_cap_circuits)}")


if __name__ == "__main__":
    result, circuit_config, freq_config = demonstrate_frequency_serialization()
    demo_filtering_scenarios()