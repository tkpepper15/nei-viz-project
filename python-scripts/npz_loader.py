#!/usr/bin/env python3
"""
NPZ Circuit Data Loader - Web Backend API
Ultra-fast NPZ file serving for circuit simulation results
"""

import numpy as np
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import time

@dataclass
class NPZDataset:
    """Loaded NPZ dataset with fast access methods"""
    parameters: np.ndarray    # (N, 5) - Rsh, Ra, Ca, Rb, Cb
    resnorms: np.ndarray      # (N,) - Residual norms
    spectrum: np.ndarray      # (N, F, 4) - Real, imag, mag, phase
    frequencies: np.ndarray   # (F,) - Frequency points
    metadata: dict           # Dataset info
    
    def __post_init__(self):
        """Setup fast indexing"""
        self.n_params = len(self.parameters)
        self.n_freqs = len(self.frequencies)
        
        # Create sorted indices for fast queries
        self.resnorm_indices = np.argsort(self.resnorms)
        self.best_params = self.parameters[self.resnorm_indices[:1000]]  # Cache top 1000

class CircuitNPZLoader:
    """Lightning-fast NPZ data loader for web applications"""
    
    def __init__(self):
        self.datasets: Dict[str, NPZDataset] = {}
        self.load_cache = {}
        
    def load_dataset(self, npz_file: str) -> NPZDataset:
        """Load NPZ dataset with caching"""
        if npz_file in self.datasets:
            return self.datasets[npz_file]
            
        print(f"⚡ Loading NPZ dataset: {npz_file}")
        start_time = time.time()
        
        # Load NPZ file
        data = np.load(npz_file, allow_pickle=True)
        
        # Extract arrays
        dataset = NPZDataset(
            parameters=data['parameters'],
            resnorms=data['resnorms'], 
            spectrum=data['spectrum'],
            frequencies=data['frequencies'],
            metadata=data['metadata'].item()  # Unpickle metadata
        )
        
        print(f"✅ Dataset loaded: {dataset.n_params:,} params, {dataset.n_freqs} freqs ({time.time()-start_time:.2f}s)")
        
        self.datasets[npz_file] = dataset
        return dataset
    
    def get_top_results(self, dataset: NPZDataset, n: int = 1000) -> dict:
        """Get top N results for web display"""
        top_indices = dataset.resnorm_indices[:n]
        
        return {
            'parameters': dataset.parameters[top_indices].tolist(),
            'resnorms': dataset.resnorms[top_indices].tolist(),
            'param_names': dataset.metadata['param_names'],
            'param_units': dataset.metadata['param_units'],
            'n_total': dataset.n_params,
            'best_resnorm': float(dataset.resnorms[top_indices[0]])
        }
    
    def get_spectrum_data(self, dataset: NPZDataset, param_indices: List[int]) -> dict:
        """Get spectrum data for specific parameter sets"""
        spectra = dataset.spectrum[param_indices]  # (n, freqs, 4)
        
        return {
            'frequencies': dataset.frequencies.tolist(),
            'spectra': spectra.tolist(),  # [param][freq][real,imag,mag,phase]
            'spectrum_columns': dataset.metadata['spectrum_columns'],
            'freq_range': [float(dataset.frequencies[0]), float(dataset.frequencies[-1])]
        }
    
    def get_parameter_ranges(self, dataset: NPZDataset) -> dict:
        """Get parameter min/max for web UI"""
        params = dataset.parameters
        
        ranges = {}
        for i, name in enumerate(dataset.metadata['param_names']):
            ranges[name.lower()] = {
                'min': float(params[:, i].min()),
                'max': float(params[:, i].max()),
                'unit': dataset.metadata['param_units'][i]
            }
        
        return ranges
    
    def search_by_resnorm(self, dataset: NPZDataset, max_resnorm: float, limit: int = 1000) -> dict:
        """Find all parameters below resnorm threshold"""
        mask = dataset.resnorms <= max_resnorm
        indices = np.where(mask)[0]
        
        if len(indices) > limit:
            # Sort by resnorm and take best N
            sorted_indices = indices[np.argsort(dataset.resnorms[indices])][:limit]
            indices = sorted_indices
            
        return {
            'indices': indices.tolist(),
            'parameters': dataset.parameters[indices].tolist(),
            'resnorms': dataset.resnorms[indices].tolist(),
            'count': len(indices),
            'total_searched': dataset.n_params
        }

# Web API Functions (FastAPI/Flask ready)
def create_web_api_functions(loader: CircuitNPZLoader):
    """Create web-ready API functions"""
    
    def load_circuit_dataset(npz_file: str):
        """API: Load circuit dataset"""
        try:
            dataset = loader.load_dataset(npz_file)
            return {
                'status': 'success',
                'metadata': dataset.metadata,
                'dataset_info': {
                    'n_parameters': dataset.n_params,
                    'n_frequencies': dataset.n_freqs,
                    'parameter_ranges': loader.get_parameter_ranges(dataset)
                }
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def get_best_results(npz_file: str, n: int = 1000):
        """API: Get top N results"""
        try:
            dataset = loader.datasets[npz_file]
            results = loader.get_top_results(dataset, n)
            return {'status': 'success', 'data': results}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def get_spectrum_for_params(npz_file: str, indices: List[int]):
        """API: Get full spectrum for specific parameter indices"""
        try:
            dataset = loader.datasets[npz_file]
            spectra = loader.get_spectrum_data(dataset, indices)
            return {'status': 'success', 'data': spectra}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
            
    def search_parameters(npz_file: str, max_resnorm: float, limit: int = 1000):
        """API: Search parameters by resnorm threshold"""
        try:
            dataset = loader.datasets[npz_file]
            results = loader.search_by_resnorm(dataset, max_resnorm, limit)
            return {'status': 'success', 'data': results}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    return {
        'load_dataset': load_circuit_dataset,
        'get_best_results': get_best_results, 
        'get_spectrum': get_spectrum_for_params,
        'search_parameters': search_parameters
    }

# Example usage / testing
if __name__ == "__main__":
    loader = CircuitNPZLoader()
    
    # Test with your existing NPZ file (if it exists)
    npz_files = list(Path('.').glob('*.npz'))
    if npz_files:
        test_file = str(npz_files[0])
        print(f"🧪 Testing with: {test_file}")
        
        # Load dataset
        dataset = loader.load_dataset(test_file)
        print(f"📊 Dataset: {dataset.n_params:,} parameters")
        
        # Get top 10 results
        top_results = loader.get_top_results(dataset, 10)
        print(f"🏆 Best resnorm: {top_results['best_resnorm']:.6f}")
        
        # Get spectrum for best result
        spectrum_data = loader.get_spectrum_data(dataset, [0])
        print(f"📈 Spectrum: {len(spectrum_data['frequencies'])} frequency points")
        
        print("✅ NPZ loader ready for web integration!")
    else:
        print("⚠️  No NPZ files found. Run circuit_computation.py first.")