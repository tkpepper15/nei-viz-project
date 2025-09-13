"""
Complete Serialization System Example
=====================================

This demonstrates the full workflow from your original vision to a complete
procedural system with circuit configs, frequency configs, and resnorm results.
"""

from config_serializer import ConfigSerializer, ConfigId
from frequency_serializer import FrequencySerializer, FrequencyConfig, ComputationResult
import numpy as np

def demonstrate_complete_workflow():
    """Show the complete workflow with all serializations"""
    
    print("🎯 COMPLETE SERIALIZATION SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    # Step 1: Initialize serializers
    circuit_serializer = ConfigSerializer(grid_size=15)
    freq_serializer = FrequencySerializer()
    
    print("📋 STEP 1: System Initialization")
    print(f"Circuit Grid: 15x15x15x15x15 = {15**5:,} possible configurations")
    print(f"Frequency Presets: {len(freq_serializer.list_presets())} standard configurations")
    print()
    
    # Step 2: Create circuit configurations (your original vision)
    print("🔧 STEP 2: Circuit Configuration Creation")
    print("Your original ConfigId('01', '01', '01', '01', '01', '01') concept:")
    
    sample_configs = [
        (1, 1, 1, 1, 1),    # Minimum configuration
        (2, 3, 1, 4, 5),    # Mixed configuration  
        (15, 15, 15, 15, 15), # Maximum configuration
        (5, 8, 3, 12, 7),   # Random configuration
    ]
    
    circuit_results = []
    for i, (ra, rb, rsh, ca, cb) in enumerate(sample_configs):
        config = circuit_serializer.serialize_config(ra, rb, rsh, ca, cb)
        params = circuit_serializer.deserialize_config(config)
        
        print(f"  Config {i+1}: Indices Ra={ra:02d}, Rb={rb:02d}, Rsh={rsh:02d}, Ca={ca:02d}, Cb={cb:02d}")
        print(f"    Serial ID: {config.to_string()}")
        print(f"    Parameters: Rsh={params.rsh:.0f}Ω, Ra={params.ra:.0f}Ω, Ca={params.ca*1e6:.1f}μF")
        
        circuit_results.append((config, params))
    print()
    
    # Step 3: Create frequency configurations
    print("📡 STEP 3: Frequency Configuration Creation")
    
    freq_presets = ["standard", "high_res", "fast", "low_freq"]
    freq_configs = []
    
    for preset_name in freq_presets:
        freq_config = freq_serializer.get_preset(preset_name)
        freq_configs.append((preset_name, freq_config))
        
        print(f"  {preset_name.upper()}:")
        print(f"    Serial ID: {freq_config.to_id()}")
        print(f"    Range: {freq_config.min_freq:.1E} to {freq_config.max_freq:.1E} Hz")
        print(f"    Points: {freq_config.n_points} ({freq_config.spacing})")
    print()
    
    # Step 4: Generate computation results
    print("⚡ STEP 4: Computation Results Generation")
    
    # Simulate computation for different circuit/frequency combinations
    np.random.seed(42)
    all_results = []
    
    for circuit_config, circuit_params in circuit_results[:2]:  # Use first 2 circuits
        for preset_name, freq_config in freq_configs[:2]:       # Use first 2 freq configs
            
            # Simulate resnorm computation (better performance for lower Ra)
            base_resnorm = 1.0 / (circuit_params.ra * circuit_params.ca * 1e6)
            resnorm = base_resnorm + np.random.normal(0, base_resnorm * 0.1)
            
            result = ComputationResult(
                circuit_config_id=circuit_config.to_string(),
                frequency_config_id=freq_config.to_id(),
                resnorm=resnorm,
                computation_time=np.random.uniform(0.5, 3.0)
            )
            
            all_results.append((result, circuit_config, circuit_params, freq_config))
            
            print(f"  Circuit {circuit_config.to_string()} + Freq {preset_name}:")
            print(f"    Resnorm: {resnorm:.6E}")
            print(f"    Compact: {result.to_compact_string()}")
    print()
    
    # Step 5: Demonstrate filtering and analysis
    print("🔍 STEP 5: Filtering and Analysis")
    
    # Sort by performance
    all_results.sort(key=lambda x: x[0].resnorm)
    
    print("Best performing configurations:")
    for i, (result, circuit_config, circuit_params, freq_config) in enumerate(all_results):
        print(f"  {i+1}. Circuit: {circuit_config.to_string()}")
        print(f"     Frequency: {freq_config.to_id()}")
        print(f"     Resnorm: {result.resnorm:.6E}")
        print(f"     Ra: {circuit_params.ra:.0f}Ω, Ca: {circuit_params.ca*1e6:.1f}μF")
        print(f"     Storage: {len(result.to_compact_string())} bytes")
    print()
    
    # Step 6: Show visualization scenarios
    print("🎨 STEP 6: Visualization Scenarios")
    
    best_result = all_results[0]
    result, circuit_config, circuit_params, freq_config = best_result
    
    print("NYQUIST PLOT (for detailed impedance analysis):")
    print(f"  1. Load circuit config: {circuit_config.to_string()}")
    print(f"  2. Regenerate parameters: Rsh={circuit_params.rsh:.0f}Ω, etc.")
    print(f"  3. Load frequency config: {freq_config.to_id()}")
    print(f"  4. Generate frequencies: {freq_config.n_points} points")
    frequencies = freq_config.generate_frequencies()
    print(f"     Range: {frequencies[0]:.2E} to {frequencies[-1]:.2E} Hz")
    print(f"  5. Calculate impedance array on-demand")
    print(f"  6. Plot real vs imaginary components")
    print()
    
    print("SPIDER PLOT (for parameter space exploration):")
    print(f"  1. Use circuit configs directly: {[r[1].to_string() for r in all_results]}")
    print(f"  2. Extract parameter indices for plotting:")
    for result, circuit_config, _, _ in all_results:
        indices = f"Ra={circuit_config.ra_idx:02d}, Rb={circuit_config.rb_idx:02d}, etc."
        print(f"     {circuit_config.to_string()}: {indices}, resnorm={result.resnorm:.3E}")
    print(f"  3. Color/size by resnorm values")
    print(f"  4. Filter by parameter ranges (e.g., low Ra)")
    print(f"  5. No frequency data needed!")
    print()
    
    # Step 7: Storage efficiency analysis
    print("💾 STEP 7: Storage Efficiency Analysis")
    
    # Traditional storage calculation
    traditional_size_per_result = (
        5 * 8 +           # Circuit parameters (5 floats)
        freq_config.n_points * 8 +  # Frequency array
        freq_config.n_points * 16   # Complex impedance array
    )
    
    # Serialized storage
    serialized_size_per_result = len(result.to_compact_string())
    
    reduction_factor = traditional_size_per_result / serialized_size_per_result
    
    print(f"Traditional storage per result:")
    print(f"  Circuit params: 5 × 8 = 40 bytes")
    print(f"  Frequency array: {freq_config.n_points} × 8 = {freq_config.n_points * 8} bytes")
    print(f"  Impedance array: {freq_config.n_points} × 16 = {freq_config.n_points * 16} bytes")
    print(f"  Total: {traditional_size_per_result:,} bytes")
    print()
    
    print(f"Serialized storage per result:")
    print(f"  Compact string: '{result.to_compact_string()}'")
    print(f"  Total: {serialized_size_per_result} bytes")
    print()
    
    print(f"Efficiency gain: {reduction_factor:.0f}x smaller storage")
    print(f"For 1M results: {traditional_size_per_result * 1e6 / 1e9:.1f} GB → {serialized_size_per_result * 1e6 / 1e6:.0f} MB")
    print()
    
    # Step 8: Summary
    print("✅ STEP 8: System Summary")
    print("Your original ConfigId vision has been fully realized:")
    print(f"  ✅ 1-based indexing (01-15 for grid size 15)")
    print(f"  ✅ Compact string representation")
    print(f"  ✅ Procedural parameter generation")
    print(f"  ✅ Scientific notation for frequencies")
    print(f"  ✅ Associated resnorm storage")
    print(f"  ✅ Easy filtering and analysis")
    print(f"  ✅ Optimized for both Nyquist and Spider plots")
    print(f"  ✅ Massive storage efficiency ({reduction_factor:.0f}x)")
    
    return all_results


def show_all_parameter_values():
    """Display all possible parameter values for reference"""
    
    print("\n" + "=" * 80)
    print("📊 COMPLETE PARAMETER VALUE REFERENCE (Grid Size 15)")
    print("=" * 80)
    
    serializer = ConfigSerializer(grid_size=15)
    
    param_names = ['rsh', 'ra', 'ca', 'rb', 'cb']
    param_labels = ['Rsh (Shunt Resistance)', 'Ra (Apical Resistance)', 
                   'Ca (Apical Capacitance)', 'Rb (Basal Resistance)', 
                   'Cb (Basal Capacitance)']
    
    for param_name, label in zip(param_names, param_labels):
        print(f"\n{label.upper()}:")
        print("-" * 50)
        
        grid = serializer.parameter_grids[param_name]
        unit = "Ω" if param_name in ['rsh', 'ra', 'rb'] else "F"
        
        # Display in columns for better readability
        for i in range(0, 15, 3):
            row_items = []
            for j in range(3):
                if i + j < 15:
                    idx = i + j + 1  # 1-based index
                    val = grid[i + j]
                    if unit == "F":
                        row_items.append(f"{idx:02d}: {val:.2E} F ({val*1e6:.1f}μF)")
                    else:
                        row_items.append(f"{idx:02d}: {val:.0f} {unit}")
            
            # Print row with proper spacing
            print("  " + "    ".join(f"{item:<25}" for item in row_items))


def show_frequency_presets():
    """Display all frequency preset configurations"""
    
    print("\n" + "=" * 80)
    print("📡 FREQUENCY CONFIGURATION PRESETS")
    print("=" * 80)
    
    serializer = FrequencySerializer()
    
    for preset_name in serializer.list_presets():
        config = serializer.get_preset(preset_name)
        info = config.get_frequency_info()
        
        print(f"\n{preset_name.upper()}:")
        print(f"  Serial ID: {config.to_id()}")
        print(f"  Range: {config.min_freq:.1E} to {config.max_freq:.1E} Hz")
        print(f"  Points: {config.n_points} ({config.spacing} spacing)")
        if config.spacing == "log":
            decades = np.log10(config.max_freq / config.min_freq)
            print(f"  Span: {decades:.1f} decades")
            print(f"  Points/decade: {config.n_points / decades:.1f}")
        
        # Show frequency array samples
        frequencies = config.generate_frequencies()
        print(f"  First 3: {[f'{f:.2E}' for f in frequencies[:3]]}")
        print(f"  Last 3:  {[f'{f:.2E}' for f in frequencies[-3:]]}")


if __name__ == "__main__":
    results = demonstrate_complete_workflow()
    show_all_parameter_values()
    show_frequency_presets()
    
    print("\n" + "🎉" * 20)
    print("COMPLETE SERIALIZATION SYSTEM READY!")
    print("Your original ConfigId vision → Production-ready system")
    print("🎉" * 20)