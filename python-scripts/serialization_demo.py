"""
Configuration Serialization Integration Demo
===========================================

This demonstrates how the serialization system integrates with your
existing circuit computation workflow, showing the dramatic reduction
in storage requirements.

Key Benefits:
- 95%+ reduction in storage size
- Procedural parameter generation
- Fast querying and filtering
- Separation of computation config from measurement config
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

from config_serializer import ConfigSerializer, ConfigId, CircuitParameters
from measurement_config import MeasurementConfig, MeasurementConfigManager
from simple_lightweight_storage import SimpleLightweightStorage


def simulate_traditional_storage(config_ids: List[ConfigId], resnorms: List[float], 
                                serializer: ConfigSerializer) -> Dict:
    """Simulate traditional full-parameter storage for comparison"""
    
    # Generate full parameter data like traditional approach
    full_data = []
    for config_id, resnorm in zip(config_ids, resnorms):
        params = serializer.deserialize_config(config_id)
        full_data.append({
            'rsh': params.rsh,
            'ra': params.ra,
            'ca': params.ca,
            'rb': params.rb,
            'cb': params.cb,
            'resnorm': resnorm,
            'frequencies': list(np.logspace(-1, 5, 100)),  # Example frequency data
            'impedance_real': list(np.random.normal(0, 100, 100)),  # Simulated data
            'impedance_imag': list(np.random.normal(0, 100, 100))
        })
    
    # Calculate storage requirements
    json_str = json.dumps(full_data)
    storage_size_mb = len(json_str.encode('utf-8')) / 1024**2
    
    return {
        'approach': 'Traditional Full Storage',
        'data_points': len(full_data),
        'storage_size_mb': storage_size_mb,
        'fields_per_record': len(full_data[0]) if full_data else 0
    }


def demonstrate_storage_comparison():
    """Compare traditional vs serialized storage approaches"""
    
    print("=== Storage Approach Comparison ===\n")
    
    # Create test data
    grid_sizes = [5, 10, 15]
    results = {}
    
    for grid_size in grid_sizes:
        print(f"Testing Grid Size: {grid_size}")
        print(f"Total Configurations: {grid_size**5:,}")
        
        # Create serializer
        serializer = ConfigSerializer(grid_size=grid_size)
        
        # Generate subset of configurations (simulate realistic computation)
        # In practice, you might only compute/store the best 10-50% of results
        n_computed = min(10000, grid_size**5)  # Limit for demo
        
        # Generate random sample of configurations
        np.random.seed(42)
        all_configs = serializer.generate_all_configs()
        sampled_configs = np.random.choice(len(all_configs), size=n_computed, replace=False)
        
        config_ids = [all_configs[i][0] for i in sampled_configs]
        resnorms = np.random.exponential(1.0, n_computed)
        
        # Traditional approach
        traditional = simulate_traditional_storage(config_ids, resnorms, serializer)
        
        # Serialized approach
        storage = SimpleLightweightStorage(f"demo_grid_{grid_size}")
        start_time = time.time()
        dataset_id = storage.store_results(config_ids, resnorms, "standard_eis")
        serialized_time = time.time() - start_time
        
        metadata = storage.get_metadata(dataset_id)
        serialized = {
            'approach': 'Serialized Storage',
            'data_points': metadata['stored_results'],
            'storage_size_mb': metadata['storage_size_mb'],
            'storage_time_seconds': serialized_time
        }
        
        # Calculate savings
        size_reduction = (traditional['storage_size_mb'] - serialized['storage_size_mb']) / traditional['storage_size_mb'] * 100
        
        results[grid_size] = {
            'traditional': traditional,
            'serialized': serialized,
            'size_reduction_percent': size_reduction,
            'space_savings_mb': traditional['storage_size_mb'] - serialized['storage_size_mb']
        }
        
        print(f"  Traditional: {traditional['storage_size_mb']:.2f} MB")
        print(f"  Serialized:  {serialized['storage_size_mb']:.2f} MB")
        print(f"  Reduction:   {size_reduction:.1f}%")
        print(f"  Savings:     {results[grid_size]['space_savings_mb']:.2f} MB")
        print()
    
    return results


def demonstrate_procedural_generation():
    """Show how configurations are generated procedurally"""
    
    print("=== Procedural Configuration Generation ===\n")
    
    # Create serializer
    serializer = ConfigSerializer(grid_size=7)
    
    print("Parameter Ranges:")
    info = serializer.get_parameter_info()
    for param_name, param_info in info['sample_values'].items():
        unit = "Ω" if param_name in ['rsh', 'ra', 'rb'] else "F"
        print(f"  {param_name.upper()}: {param_info['min']:.2e} - {param_info['max']:.2e} {unit}")
    
    print(f"\nGrid Configuration:")
    print(f"  Grid Size: {info['grid_size']} points per parameter")
    print(f"  Total Combinations: {info['total_configurations']:,}")
    
    # Demonstrate config ID encoding/decoding
    print(f"\n=== Configuration ID Examples ===")
    
    sample_indices = [
        (1, 1, 1, 1, 1),    # Minimum values (1-based)
        (4, 4, 4, 4, 4),    # Middle values (1-based)
        (7, 7, 7, 7, 7),    # Maximum values (1-based)
        (2, 3, 4, 5, 6),    # Mixed values (1-based)
    ]
    
    for i, (ra_idx, rb_idx, rsh_idx, ca_idx, cb_idx) in enumerate(sample_indices):
        config_id = serializer.serialize_config(ra_idx, rb_idx, rsh_idx, ca_idx, cb_idx)
        params = serializer.deserialize_config(config_id)
        
        print(f"\nExample {i+1}:")
        print(f"  Indices: Ra={ra_idx}, Rb={rb_idx}, Rsh={rsh_idx}, Ca={ca_idx}, Cb={cb_idx}")
        print(f"  Config ID: {config_id.to_string()}")
        print(f"  Linear Index: {config_id.to_linear_index()}")
        print(f"  Parameters:")
        print(f"    Rsh: {params.rsh:.0f} Ω")
        print(f"    Ra:  {params.ra:.0f} Ω")
        print(f"    Ca:  {params.ca*1e6:.1f} μF")
        print(f"    Rb:  {params.rb:.0f} Ω") 
        print(f"    Cb:  {params.cb*1e6:.1f} μF")


def demonstrate_measurement_separation():
    """Show separation of measurement configuration from circuit parameters"""
    
    print("=== Measurement Configuration Separation ===\n")
    
    # Load measurement configs
    manager = MeasurementConfigManager()
    
    print("Available Measurement Configurations:")
    for name in manager.list_configs():
        config = manager.get_config(name)
        print(f"  {name}:")
        print(f"    Frequency: {config.min_freq} - {config.max_freq} Hz")
        print(f"    Points: {config.n_points} ({config.spacing} spacing)")
        print(f"    Description: {config.description}")
        print()
    
    # Show how same circuit config can use different measurements
    print("=== Same Circuit, Different Measurements ===")
    
    serializer = ConfigSerializer(grid_size=5)
    sample_config = serializer.serialize_config(2, 2, 2, 2, 2)  # Middle values
    params = serializer.deserialize_config(sample_config)
    
    print(f"Circuit Configuration: {sample_config.to_string()}")
    print(f"Circuit Parameters:")
    print(f"  Rsh: {params.rsh:.0f} Ω, Ra: {params.ra:.0f} Ω, Ca: {params.ca*1e6:.1f} μF")
    print(f"  Rb: {params.rb:.0f} Ω, Cb: {params.cb*1e6:.1f} μF")
    
    print(f"\nDifferent Measurement Setups:")
    
    measurement_configs = ['standard_eis', 'high_resolution', 'fast_sweep']
    for measurement_name in measurement_configs:
        config = manager.get_config(measurement_name)
        frequencies = config.generate_frequencies()
        
        print(f"\n  {measurement_name}:")
        print(f"    Frequencies: {len(frequencies)} points from {frequencies[0]:.3f} to {frequencies[-1]:.0f} Hz")
        print(f"    Use case: {config.description}")
        
        # Simulate storage with this measurement
        # In practice, you'd store: (config_id, measurement_config_name, resnorm)
        storage_entry = {
            'config_id': sample_config.to_string(),
            'measurement_config': measurement_name,
            'resnorm': np.random.exponential(1.0),  # Simulated result
            'computation_time': np.random.uniform(0.5, 5.0)
        }
        print(f"    Storage: {storage_entry}")


def demonstrate_query_capabilities():
    """Show advanced querying capabilities"""
    
    print("=== Advanced Query Capabilities ===\n")
    
    # Create test dataset
    serializer = ConfigSerializer(grid_size=8)
    storage = SimpleLightweightStorage("query_demo")
    
    # Generate sample data
    np.random.seed(123)
    n_samples = 1000
    config_ids = []
    resnorms = []
    
    for _ in range(n_samples):
        # Random configuration using 1-based indexing
        indices = np.random.randint(1, 9, size=5)  # 1 to 8 inclusive for grid_size=8
        config_id = serializer.serialize_config(*indices)
        config_ids.append(config_id)
        
        # Simulate resnorm based on some pattern
        params = serializer.deserialize_config(config_id)
        # Better performance for lower resistance, higher capacitance
        synthetic_resnorm = 1.0 / (params.rsh * params.ca * 1e6)  # Rough heuristic
        resnorms.append(synthetic_resnorm + np.random.normal(0, synthetic_resnorm * 0.1))
    
    # Store data
    dataset_id = storage.store_results(config_ids, resnorms, "high_resolution")
    
    print(f"Created test dataset with {n_samples} configurations")
    
    # Demonstrate queries
    queries = [
        ("Top 50 performers", {"max_results": 50}),
        ("Good performers (resnorm < 0.001)", {"max_resnorm": 0.001}),
        ("Moderate performers", {"min_resnorm": 0.001, "max_resnorm": 0.01, "max_results": 100}),
    ]
    
    for query_name, params in queries:
        results = storage.load_results(dataset_id, **params)
        
        print(f"\n{query_name}: {len(results)} results")
        if results:
            resnorm_range = [r.resnorm for r in results]
            print(f"  Resnorm range: {min(resnorm_range):.6f} - {max(resnorm_range):.6f}")
            
            # Show parameter diversity
            expanded_results = [(r.config_id, serializer.deserialize_config(r.config_id), r.resnorm) 
                              for r in results[:5]]
            
            print(f"  Sample results:")
            for i, (config_id, params, resnorm) in enumerate(expanded_results):
                print(f"    {i+1}. {config_id.to_string()} -> resnorm: {resnorm:.6f}")
                print(f"       Rsh: {params.rsh:.0f}Ω, Ca: {params.ca*1e6:.1f}μF")


def main():
    """Run all demonstrations"""
    
    print("🔬 Circuit Configuration Serialization System Demo")
    print("=" * 60)
    
    # 1. Storage comparison
    storage_results = demonstrate_storage_comparison()
    
    print("\n" + "=" * 60)
    
    # 2. Procedural generation
    demonstrate_procedural_generation()
    
    print("\n" + "=" * 60)
    
    # 3. Measurement separation
    demonstrate_measurement_separation()
    
    print("\n" + "=" * 60)
    
    # 4. Query capabilities
    demonstrate_query_capabilities()
    
    print("\n" + "=" * 60)
    print("✅ Serialization System Benefits Summary:")
    print()
    
    # Calculate average savings
    avg_reduction = np.mean([r['size_reduction_percent'] for r in storage_results.values()])
    total_savings = sum([r['space_savings_mb'] for r in storage_results.values()])
    
    print(f"📊 Storage Efficiency:")
    print(f"   Average size reduction: {avg_reduction:.1f}%")
    print(f"   Total space saved in demo: {total_savings:.1f} MB")
    
    print(f"\n🚀 Key Features:")
    print(f"   ✓ Procedural parameter generation")
    print(f"   ✓ Compact configuration IDs (e.g., '15_05_10_03_07_12')")
    print(f"   ✓ Separate measurement configurations")
    print(f"   ✓ Fast querying and filtering")
    print(f"   ✓ Only essential data stored (config ID + resnorm)")
    print(f"   ✓ Full parameter reconstruction on demand")
    
    print(f"\n💡 Integration with your workflow:")
    print(f"   1. Generate parameter grid using ConfigSerializer")
    print(f"   2. Run computations, store only (ConfigId, resnorm)")
    print(f"   3. Query results by performance thresholds")
    print(f"   4. Regenerate full parameters only when needed")
    print(f"   5. Apply different measurement configs to same circuit configs")


if __name__ == "__main__":
    main()