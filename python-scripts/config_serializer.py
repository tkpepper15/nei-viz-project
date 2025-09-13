"""
Configuration Serialization System
==================================

This system serializes circuit parameter configurations using compact IDs
instead of storing full parameter values. Configurations can be procedurally
regenerated from their serialized form.

Usage:
    serializer = ConfigSerializer(grid_size=15)
    config_id = serializer.serialize_config(ra_idx=5, rb_idx=10, ...)
    config = serializer.deserialize_config(config_id)
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ParameterRange:
    """Defines a parameter's range and grid configuration"""
    name: str
    min_val: float
    max_val: float
    unit: str
    scale: str = "log"  # "log" or "linear"
    
    def generate_values(self, n_points: int) -> np.ndarray:
        """Generate parameter values for the grid"""
        if self.scale == "log":
            return np.logspace(np.log10(self.min_val), np.log10(self.max_val), n_points)
        else:
            return np.linspace(self.min_val, self.max_val, n_points)


@dataclass
class ConfigId:
    """Compact representation of a configuration using parameter indices (1-based)"""
    ra_idx: int    # Ra parameter index (1 to grid_size)
    rb_idx: int    # Rb parameter index (1 to grid_size)
    rsh_idx: int   # Rsh parameter index (1 to grid_size)
    ca_idx: int    # Ca parameter index (1 to grid_size)
    cb_idx: int    # Cb parameter index (1 to grid_size)
    grid_size: int # Grid size used for generation
    
    def __post_init__(self):
        """Validate that indices are in valid 1-based range"""
        for idx_name, idx_val in [('ra_idx', self.ra_idx), ('rb_idx', self.rb_idx), 
                                  ('rsh_idx', self.rsh_idx), ('ca_idx', self.ca_idx), ('cb_idx', self.cb_idx)]:
            if not (1 <= idx_val <= self.grid_size):
                raise ValueError(f"{idx_name} must be between 1 and {self.grid_size}, got {idx_val}")
    
    def to_string(self) -> str:
        """Convert to compact string representation (1-based indices)"""
        return f"{self.grid_size:02d}_{self.rsh_idx:02d}_{self.ra_idx:02d}_{self.ca_idx:02d}_{self.rb_idx:02d}_{self.cb_idx:02d}"
    
    @classmethod
    def from_string(cls, config_str: str) -> 'ConfigId':
        """Parse from string representation"""
        parts = config_str.split('_')
        if len(parts) != 6:
            raise ValueError(f"Invalid config string format: {config_str}")
        
        grid_size, rsh_idx, ra_idx, ca_idx, rb_idx, cb_idx = map(int, parts)
        return cls(ra_idx, rb_idx, rsh_idx, ca_idx, cb_idx, grid_size)
    
    def to_linear_index(self) -> int:
        """Convert to single linear index for the parameter space (0-based internally)"""
        # Convert to 0-based for calculation
        rsh_0 = self.rsh_idx - 1
        ra_0 = self.ra_idx - 1
        ca_0 = self.ca_idx - 1
        rb_0 = self.rb_idx - 1
        cb_0 = self.cb_idx - 1
        
        return (rsh_0 * (self.grid_size ** 4) + 
                ra_0 * (self.grid_size ** 3) +
                ca_0 * (self.grid_size ** 2) +
                rb_0 * self.grid_size +
                cb_0)
    
    @classmethod
    def from_linear_index(cls, linear_idx: int, grid_size: int) -> 'ConfigId':
        """Convert from linear index back to parameter indices (1-based)"""
        # Ensure linear_idx is an integer
        linear_idx = int(linear_idx)
        
        cb_0 = linear_idx % grid_size
        linear_idx //= grid_size
        
        rb_0 = linear_idx % grid_size
        linear_idx //= grid_size
        
        ca_0 = linear_idx % grid_size
        linear_idx //= grid_size
        
        ra_0 = linear_idx % grid_size
        linear_idx //= grid_size
        
        rsh_0 = linear_idx % grid_size
        
        # Convert back to 1-based indices
        return cls(ra_0 + 1, rb_0 + 1, rsh_0 + 1, ca_0 + 1, cb_0 + 1, grid_size)


@dataclass 
class CircuitParameters:
    """Actual circuit parameter values"""
    rsh: float  # Shunt resistance (ohm)
    ra: float   # Ra resistance (ohm)
    ca: float   # Ca capacitance (farad)
    rb: float   # Rb resistance (ohm)  
    cb: float   # Cb capacitance (farad)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class ConfigSerializer:
    """Handles serialization/deserialization of circuit configurations"""
    
    def __init__(self, grid_size: int = 15):
        self.grid_size = grid_size
        
        # Default parameter ranges (matching your existing ranges)
        self.parameter_ranges = {
            'rsh': ParameterRange('rsh', 10, 10000, 'ohm', 'log'),
            'ra': ParameterRange('ra', 10, 10000, 'ohm', 'log'), 
            'ca': ParameterRange('ca', 0.1e-6, 50e-6, 'farad', 'log'),
            'rb': ParameterRange('rb', 10, 10000, 'ohm', 'log'),
            'cb': ParameterRange('cb', 0.1e-6, 50e-6, 'farad', 'log')
        }
        
        # Pre-generate parameter grids for efficiency
        self._generate_parameter_grids()
    
    def _generate_parameter_grids(self):
        """Pre-generate all parameter value grids"""
        self.parameter_grids = {}
        for param_name, param_range in self.parameter_ranges.items():
            self.parameter_grids[param_name] = param_range.generate_values(self.grid_size)
    
    def serialize_config(self, ra_idx: int, rb_idx: int, rsh_idx: int, 
                        ca_idx: int, cb_idx: int) -> ConfigId:
        """Create a ConfigId from parameter indices (1-based: 1 to grid_size)"""
        return ConfigId(ra_idx, rb_idx, rsh_idx, ca_idx, cb_idx, self.grid_size)
    
    def deserialize_config(self, config_id: ConfigId) -> CircuitParameters:
        """Generate actual parameter values from ConfigId"""
        if config_id.grid_size != self.grid_size:
            raise ValueError(f"ConfigId grid size {config_id.grid_size} doesn't match serializer grid size {self.grid_size}")
        
        # Convert 1-based indices to 0-based for array access
        return CircuitParameters(
            rsh=self.parameter_grids['rsh'][config_id.rsh_idx - 1],
            ra=self.parameter_grids['ra'][config_id.ra_idx - 1], 
            ca=self.parameter_grids['ca'][config_id.ca_idx - 1],
            rb=self.parameter_grids['rb'][config_id.rb_idx - 1],
            cb=self.parameter_grids['cb'][config_id.cb_idx - 1]
        )
    
    def deserialize_from_string(self, config_str: str) -> CircuitParameters:
        """Deserialize from string representation"""
        config_id = ConfigId.from_string(config_str)
        return self.deserialize_config(config_id)
    
    def generate_all_configs(self) -> List[Tuple[ConfigId, CircuitParameters]]:
        """Generate all possible configurations for the grid"""
        configs = []
        
        # Use 1-based indexing (1 to grid_size inclusive)
        for rsh_idx in range(1, self.grid_size + 1):
            for ra_idx in range(1, self.grid_size + 1):
                for ca_idx in range(1, self.grid_size + 1):
                    for rb_idx in range(1, self.grid_size + 1):
                        for cb_idx in range(1, self.grid_size + 1):
                            config_id = self.serialize_config(ra_idx, rb_idx, rsh_idx, ca_idx, cb_idx)
                            params = self.deserialize_config(config_id)
                            configs.append((config_id, params))
        
        return configs
    
    def get_parameter_info(self) -> Dict:
        """Get information about parameter ranges and grids"""
        return {
            'grid_size': self.grid_size,
            'total_configurations': self.grid_size ** 5,
            'parameter_ranges': {name: asdict(range_def) for name, range_def in self.parameter_ranges.items()},
            'sample_values': {
                name: {
                    'min': float(grid[0]),
                    'max': float(grid[-1]), 
                    'sample_indices': [1, self.grid_size//2 + 1, self.grid_size],  # 1-based indices
                    'sample_values': [float(grid[0]), float(grid[self.grid_size//2]), float(grid[-1])]
                }
                for name, grid in self.parameter_grids.items()
            }
        }
    
    def save_parameter_info(self, filepath: str):
        """Save parameter configuration to JSON file"""
        info = self.get_parameter_info()
        with open(filepath, 'w') as f:
            json.dump(info, f, indent=2)
    
    def load_parameter_ranges(self, filepath: str):
        """Load parameter ranges from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for param_name, range_data in data['parameter_ranges'].items():
            self.parameter_ranges[param_name] = ParameterRange(**range_data)
        
        self.grid_size = data['grid_size']
        self._generate_parameter_grids()


# Demonstration functions
def demo_serialization():
    """Demonstrate the serialization system"""
    print("=== Configuration Serialization Demo ===\n")
    
    # Create serializer
    serializer = ConfigSerializer(grid_size=15)  # Small grid for demo
    
    # Create a configuration using 1-based indexing
    config_id = serializer.serialize_config(ra_idx=2, rb_idx=3, rsh_idx=1, ca_idx=4, cb_idx=5)
    print(f"Config ID: {config_id}")
    print(f"Config String: {config_id.to_string()}")
    print(f"Linear Index: {config_id.to_linear_index()}")
    
    # Deserialize back to parameters
    params = serializer.deserialize_config(config_id)
    print(f"\nDeserialized Parameters:")
    print(f"  Rsh: {params.rsh:.2e} Ω")
    print(f"  Ra:  {params.ra:.2e} Ω") 
    print(f"  Ca:  {params.ca:.2e} F ({params.ca*1e6:.1f} μF)")
    print(f"  Rb:  {params.rb:.2e} Ω")
    print(f"  Cb:  {params.cb:.2e} F ({params.cb*1e6:.1f} μF)")
    
    # Test round-trip serialization
    config_str = config_id.to_string()
    recovered_params = serializer.deserialize_from_string(config_str)
    print(f"\nRound-trip test: {params == recovered_params}")
    
    # Show parameter space info
    print(f"\n=== Parameter Space Info ===")
    info = serializer.get_parameter_info()
    print(f"Grid Size: {info['grid_size']}")
    print(f"Total Configurations: {info['total_configurations']:,}")
    
    print(f"\n=== All Parameter Values (Grid Size {info['grid_size']}) ===")
    for param_name, grid in serializer.parameter_grids.items():
        print(f"{param_name.upper()}:")
        unit = "Ω" if param_name in ['rsh', 'ra', 'rb'] else "F"
        for i, val in enumerate(grid):
            idx = i + 1  # Convert to 1-based index
            if unit == "F":
                print(f"  Index {idx:02d}: {val:.2e} F ({val*1e6:.1f} μF)")
            else:
                print(f"  Index {idx:02d}: {val:.2e} {unit}")
        print()  # Add spacing between parameters


if __name__ == "__main__":
    demo_serialization()