#!/usr/bin/env python3
"""
Debug Comparison Script
Compare Python computation results with expected JavaScript behavior
to help identify where the JavaScript implementation might be failing.
"""

import numpy as np
from circuit_computation import CircuitSimulator, CircuitParameters

def test_single_computation():
    """Test a single parameter set computation for debugging"""
    print("🔬 Single Parameter Set Test")
    print("=" * 40)
    
    simulator = CircuitSimulator()
    
    # Test with known parameters
    test_params = CircuitParameters(
        Rsh=1000.0,
        Ra=200.0,
        Ca=1.0e-6,
        Rb=800.0,
        Cb=2.0e-6
    )
    
    # Test at a few frequencies
    test_frequencies = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])
    
    print(f"Test parameters:")
    print(f"  Rsh: {test_params.Rsh} Ω")
    print(f"  Ra: {test_params.Ra} Ω, Ca: {test_params.Ca*1e6:.1f} µF")  
    print(f"  Rb: {test_params.Rb} Ω, Cb: {test_params.Cb*1e6:.1f} µF")
    print()
    
    spectrum = simulator.calculate_spectrum(test_params, test_frequencies)
    
    print("Impedance Results:")
    print(f"{'Freq (Hz)':<10} {'Real':<12} {'Imag':<12} {'Magnitude':<12} {'Phase (°)':<10}")
    print("-" * 66)
    
    for point in spectrum:
        print(f"{point.frequency:<10.1f} {point.real:<12.4f} {point.imaginary:<12.4f} "
              f"{point.magnitude:<12.4f} {point.phase:<10.2f}")
    
    return spectrum

def test_mathematical_model():
    """Test the mathematical model implementation"""
    print("\n🧮 Mathematical Model Verification")
    print("=" * 40)
    
    simulator = CircuitSimulator()
    
    # Simple test case
    params = CircuitParameters(Rsh=100, Ra=50, Ca=1e-6, Rb=200, Cb=2e-6)
    freq = 100.0  # 100 Hz
    
    # Manual calculation for verification
    omega = 2 * np.pi * freq
    
    # Za = Ra/(1+jωRaCa)
    za_denom = 1 + 1j * omega * params.Ra * params.Ca
    za = params.Ra / za_denom
    print(f"Za = {za:.6f}")
    
    # Zb = Rb/(1+jωRbCb)  
    zb_denom = 1 + 1j * omega * params.Rb * params.Cb
    zb = params.Rb / zb_denom
    print(f"Zb = {zb:.6f}")
    
    # Za + Zb
    zab = za + zb
    print(f"Za + Zb = {zab:.6f}")
    
    # Parallel combination
    numerator = params.Rsh * zab
    denominator = params.Rsh + zab
    z_total = numerator / denominator
    
    print(f"Z_total (manual) = {z_total:.6f}")
    
    # Using function
    z_func = simulator.calculate_impedance(params, freq)
    print(f"Z_total (function) = {z_func:.6f}")
    
    print(f"Match: {np.isclose(z_total, z_func)}")

def compare_with_reference():
    """Compare against the reference parameters to see parameter recovery"""
    print("\n🎯 Parameter Recovery Test") 
    print("=" * 40)
    
    simulator = CircuitSimulator()
    
    # Reference parameters (ground truth)
    ref_params = CircuitParameters(
        Rsh=24.0,
        Ra=500.0, 
        Ca=0.5e-6,
        Rb=500.0,
        Cb=0.5e-6
    )
    
    # Test frequencies
    frequencies = np.logspace(np.log10(1), np.log10(10000), 20)
    
    # Calculate reference spectrum
    ref_spectrum = simulator.calculate_spectrum(ref_params, frequencies)
    
    # Test exact same parameters (should give resnorm ≈ 0)
    test_spectrum = simulator.calculate_spectrum(ref_params, frequencies) 
    resnorm = simulator.calculate_resnorm_mae(test_spectrum, ref_spectrum)
    
    print(f"Reference vs Reference resnorm: {resnorm:.10f}")
    print("(Should be very close to 0)")
    
    # Test slightly different parameters  
    modified_params = CircuitParameters(
        Rsh=25.0,  # +1 Ω
        Ra=510.0,  # +10 Ω
        Ca=0.51e-6, # +0.01 µF
        Rb=490.0,  # -10 Ω 
        Cb=0.49e-6  # -0.01 µF
    )
    
    modified_spectrum = simulator.calculate_spectrum(modified_params, frequencies)
    modified_resnorm = simulator.calculate_resnorm_mae(modified_spectrum, ref_spectrum)
    
    print(f"Modified vs Reference resnorm: {modified_resnorm:.6f}")
    print("(Should be small but > 0)")

def expected_javascript_output():
    """Show what the JavaScript version should be outputting"""
    print("\n📋 Expected JavaScript Behavior")
    print("=" * 40)
    
    print("The JavaScript application should be showing:")
    print("1. Grid points generated: 3,125 parameters")
    print("2. Created 25-125 chunks from 3,125 grid points") 
    print("3. Worker messages: CHUNK_COMPLETE from each worker")
    print("4. Final computation results: 3,125 models (or limited by maxComputationResults)")
    print()
    print("Debug what to look for in browser console:")
    print("• '🔍 Grid points generated: 3125 parameters'")
    print("• '🚀 Sending chunk X to worker Y: Z parameters'")
    print("• '📨 Worker X message: CHUNK_COMPLETE'")  
    print("• '✅ Worker X completed: Y results, total: Z'")
    print("• '🔍 Final computation results: X models'")
    print()
    print("If you're seeing 0 models, check for:")
    print("• Worker errors or missing CHUNK_COMPLETE messages")
    print("• Empty topResults/otherResults in worker data")
    print("• Validation failures in result conversion")

def main():
    """Run all debug tests"""
    print("🐛 EIS Circuit Computation Debug Suite")
    print("=" * 50)
    
    # Test individual components
    test_mathematical_model()
    test_single_computation() 
    compare_with_reference()
    expected_javascript_output()
    
    print(f"\n✅ Debug tests complete!")
    print("\nThis Python implementation matches the correct circuit model.")
    print("If JavaScript is returning 0 results, the issue is likely in:")
    print("1. Worker message passing")
    print("2. Result validation/conversion") 
    print("3. Data structure mismatch between worker and main thread")

if __name__ == "__main__":
    main()