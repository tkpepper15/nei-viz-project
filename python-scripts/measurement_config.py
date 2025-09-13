"""
Measurement Configuration System
===============================

Handles frequency sweep settings and measurement parameters separately
from circuit configurations. This allows for flexible measurement
setups that can be applied to any circuit configuration.

Usage:
    config = MeasurementConfig(min_freq=0.1, max_freq=100000, n_points=100)
    frequencies = config.generate_frequencies()
    config.save('measurement_setup.json')
"""

import numpy as np
import json
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class MeasurementConfig:
    """Configuration for frequency sweep measurements"""
    
    # Frequency range
    min_freq: float = 0.1        # Minimum frequency (Hz)
    max_freq: float = 100000.0   # Maximum frequency (Hz)
    n_points: int = 100          # Number of frequency points
    spacing: str = "log"         # "log" or "linear" spacing
    
    # Measurement settings
    measurement_type: str = "impedance"  # "impedance", "admittance", etc.
    complex_format: str = "rectangular"  # "rectangular", "polar"
    
    # Computational settings
    precision: str = "double"    # "single", "double"
    units: Dict[str, str] = None # Unit specifications
    
    # Metadata
    name: str = "Default Measurement"
    description: str = "Standard EIS measurement configuration"
    created_by: str = "system"
    version: str = "1.0"
    
    def __post_init__(self):
        """Initialize default units if not provided"""
        if self.units is None:
            self.units = {
                "frequency": "Hz",
                "impedance": "ohm", 
                "phase": "degrees",
                "time": "seconds"
            }
    
    def generate_frequencies(self) -> np.ndarray:
        """Generate frequency array based on configuration"""
        if self.spacing == "log":
            return np.logspace(np.log10(self.min_freq), np.log10(self.max_freq), self.n_points)
        elif self.spacing == "linear":
            return np.linspace(self.min_freq, self.max_freq, self.n_points)
        else:
            raise ValueError(f"Unknown spacing type: {self.spacing}")
    
    def get_frequency_info(self) -> Dict:
        """Get detailed frequency configuration information"""
        frequencies = self.generate_frequencies()
        
        return {
            "range": {
                "min_freq": self.min_freq,
                "max_freq": self.max_freq,
                "span_decades": np.log10(self.max_freq / self.min_freq),
                "units": self.units["frequency"]
            },
            "sampling": {
                "n_points": self.n_points,
                "spacing": self.spacing,
                "points_per_decade": self.n_points / np.log10(self.max_freq / self.min_freq) if self.spacing == "log" else None
            },
            "frequencies": {
                "first": float(frequencies[0]),
                "last": float(frequencies[-1]),
                "center": float(frequencies[len(frequencies)//2]),
                "step_size": float(frequencies[1] - frequencies[0]) if self.spacing == "linear" else None,
                "decade_steps": np.diff(np.log10(frequencies[:10])).mean() if self.spacing == "log" else None
            }
        }
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        if self.min_freq <= 0:
            issues.append("Minimum frequency must be positive")
        
        if self.max_freq <= self.min_freq:
            issues.append("Maximum frequency must be greater than minimum frequency")
        
        if self.n_points < 2:
            issues.append("Number of points must be at least 2")
        
        if self.spacing not in ["log", "linear"]:
            issues.append("Spacing must be 'log' or 'linear'")
        
        if self.spacing == "log" and self.min_freq <= 0:
            issues.append("Logarithmic spacing requires positive minimum frequency")
        
        if self.n_points > 10000:
            issues.append("Warning: Large number of points may impact performance")
        
        return issues
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MeasurementConfig':
        """Create from dictionary"""
        return cls(**data)
    
    def save(self, filepath: Union[str, Path]):
        """Save configuration to JSON file"""
        filepath = Path(filepath)
        data = self.to_dict()
        data['_metadata'] = {
            'file_version': '1.0',
            'config_type': 'measurement',
            'generated_by': 'measurement_config.py'
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'MeasurementConfig':
        """Load configuration from JSON file"""
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Remove metadata if present
        data.pop('_metadata', None)
        
        return cls.from_dict(data)
    
    def copy(self) -> 'MeasurementConfig':
        """Create a copy of this configuration"""
        return MeasurementConfig(**self.to_dict())
    
    def update(self, **kwargs) -> 'MeasurementConfig':
        """Create a new configuration with updated parameters"""
        data = self.to_dict()
        data.update(kwargs)
        return MeasurementConfig(**data)


class MeasurementConfigManager:
    """Manages multiple measurement configurations"""
    
    def __init__(self):
        self.configs: Dict[str, MeasurementConfig] = {}
        self._load_default_configs()
    
    def _load_default_configs(self):
        """Load standard measurement configurations"""
        
        # Standard EIS sweep
        self.configs["standard_eis"] = MeasurementConfig(
            name="Standard EIS",
            description="Standard electrochemical impedance spectroscopy sweep",
            min_freq=0.1,
            max_freq=100000.0,
            n_points=100,
            spacing="log"
        )
        
        # High resolution sweep
        self.configs["high_resolution"] = MeasurementConfig(
            name="High Resolution EIS", 
            description="High resolution sweep for detailed analysis",
            min_freq=0.01,
            max_freq=1000000.0,
            n_points=500,
            spacing="log"
        )
        
        # Fast sweep for real-time monitoring
        self.configs["fast_sweep"] = MeasurementConfig(
            name="Fast Sweep",
            description="Quick sweep for real-time monitoring",
            min_freq=1.0,
            max_freq=10000.0,
            n_points=50,
            spacing="log"
        )
        
        # Low frequency focus
        self.configs["low_freq_focus"] = MeasurementConfig(
            name="Low Frequency Focus",
            description="Detailed low frequency characterization",
            min_freq=0.001,
            max_freq=100.0,
            n_points=200,
            spacing="log"
        )
        
        # Linear spacing example
        self.configs["linear_sweep"] = MeasurementConfig(
            name="Linear Frequency Sweep",
            description="Linear spacing for specific applications",
            min_freq=10.0,
            max_freq=1000.0,
            n_points=100,
            spacing="linear"
        )
    
    def add_config(self, name: str, config: MeasurementConfig):
        """Add a new measurement configuration"""
        self.configs[name] = config
    
    def get_config(self, name: str) -> Optional[MeasurementConfig]:
        """Get a measurement configuration by name"""
        return self.configs.get(name)
    
    def list_configs(self) -> List[str]:
        """List all available configuration names"""
        return list(self.configs.keys())
    
    def get_config_info(self, name: str) -> Optional[Dict]:
        """Get detailed information about a configuration"""
        config = self.get_config(name)
        if config is None:
            return None
        
        info = config.to_dict()
        info['frequency_info'] = config.get_frequency_info()
        info['validation_issues'] = config.validate()
        
        return info
    
    def save_all(self, directory: Union[str, Path]):
        """Save all configurations to a directory"""
        directory = Path(directory)
        directory.mkdir(exist_ok=True)
        
        for name, config in self.configs.items():
            filepath = directory / f"{name}.json"
            config.save(filepath)
    
    def load_from_directory(self, directory: Union[str, Path]):
        """Load all configurations from a directory"""
        directory = Path(directory)
        
        for filepath in directory.glob("*.json"):
            try:
                config = MeasurementConfig.load(filepath)
                name = filepath.stem
                self.configs[name] = config
            except Exception as e:
                print(f"Failed to load {filepath}: {e}")


def create_measurement_presets():
    """Create and save measurement configuration presets"""
    
    # Create measurement configs directory
    presets_dir = Path("measurement_presets")
    presets_dir.mkdir(exist_ok=True)
    
    manager = MeasurementConfigManager()
    
    # Save all default configurations
    manager.save_all(presets_dir)
    
    # Create a summary file
    summary = {
        "presets": {},
        "metadata": {
            "created_by": "measurement_config.py",
            "version": "1.0",
            "description": "Standard measurement configuration presets"
        }
    }
    
    for name in manager.list_configs():
        info = manager.get_config_info(name)
        summary["presets"][name] = {
            "name": info["name"],
            "description": info["description"],
            "frequency_range": f"{info['min_freq']:.3f} - {info['max_freq']:.0f} Hz",
            "points": info["n_points"],
            "spacing": info["spacing"],
            "file": f"{name}.json"
        }
    
    with open(presets_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Created {len(manager.configs)} measurement presets in {presets_dir}/")
    return presets_dir


def demo_measurement_config():
    """Demonstrate measurement configuration system"""
    print("=== Measurement Configuration Demo ===\n")
    
    # Create a custom measurement config
    config = MeasurementConfig(
        name="Custom RPE Measurement",
        description="Optimized for retinal pigment epithelium impedance",
        min_freq=0.1,
        max_freq=100000.0,
        n_points=150,
        spacing="log"
    )
    
    print(f"Configuration: {config.name}")
    print(f"Description: {config.description}")
    print(f"Frequency Range: {config.min_freq} - {config.max_freq} Hz")
    print(f"Points: {config.n_points} ({config.spacing} spacing)")
    
    # Generate frequencies
    frequencies = config.generate_frequencies()
    print(f"\nGenerated {len(frequencies)} frequency points:")
    print(f"  First 5: {frequencies[:5]}")
    print(f"  Last 5: {frequencies[-5:]}")
    
    # Get frequency info
    freq_info = config.get_frequency_info()
    print(f"\nFrequency Analysis:")
    print(f"  Span: {freq_info['range']['span_decades']:.1f} decades")
    print(f"  Points per decade: {freq_info['sampling']['points_per_decade']:.1f}")
    print(f"  Center frequency: {freq_info['frequencies']['center']:.1f} Hz")
    
    # Validation
    issues = config.validate()
    print(f"\nValidation: {'✓ Passed' if not issues else '⚠ Issues found'}")
    for issue in issues:
        print(f"  - {issue}")
    
    # Demonstrate manager
    print(f"\n=== Configuration Manager ===")
    manager = MeasurementConfigManager()
    
    print(f"Available presets: {len(manager.list_configs())}")
    for name in manager.list_configs():
        cfg = manager.get_config(name)
        print(f"  {name}: {cfg.min_freq}-{cfg.max_freq} Hz, {cfg.n_points} pts")


if __name__ == "__main__":
    demo_measurement_config()
    print("\n" + "="*50)
    create_measurement_presets()