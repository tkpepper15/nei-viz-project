#!/usr/bin/env python3

"""
Example script demonstrating grid mode functionality
"""

from circuit_analysis import CircuitParameters, AnalysisConfig, CircuitAnalyzer
from pathlib import Path
from datetime import datetime

def run_grid_example():
    """Run an example analysis with grid mode"""

    print("🔬 Running Grid Mode Example")
    print("=" * 40)

    # Define ground truth circuit
    ground_truth = CircuitParameters(
        Rsh=25.88,
        Ra=5011.87,
        Ca=2.69e-5,
        Rb=5011.87,
        Cb=2.69e-5,
        frequency_range=(1.0, 10000.0)
    )

    # Create configuration with grid mode
    config = AnalysisConfig(
        ground_truth_circuit=ground_truth,
        variation_ranges={
            'Rsh': (20.7, 31.06),     # ±20%
            'Ra': (4260.09, 5763.65), # ±15%
            'Ca': (2.02e-5, 3.36e-5), # ±25%
            'Rb': (4260.09, 5763.65), # ±15%
            'Cb': (2.02e-5, 3.36e-5)  # ±25%
        },
        num_variations=100,  # Maximum variations to analyze
        frequency_range=(1.0, 10000.0),
        num_frequencies=30,
        output_dir="./analysis_output",
        run_name="grid_example",
        grid_mode=True,
        grid_size=4  # 4^5 = 1024 total grid points, will sample 99 + ground truth
    )

    # Setup output directory
    timestamp = datetime.now().strftime("%Y-%m-%d")
    output_path = Path(config.output_dir) / f"{config.run_name}_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "csv").mkdir(exist_ok=True)
    (output_path / "visualizations").mkdir(exist_ok=True)
    (output_path / "analysis").mkdir(exist_ok=True)

    print(f"Output directory: {output_path}")

    # Initialize analyzer and run analysis
    analyzer = CircuitAnalyzer()

    # Generate parameter variations
    print("\n🎲 Generating parameter variations...")
    variations = analyzer.generate_parameter_variations(config)

    # Perform analysis
    print(f"\n🔍 Analyzing {len(variations)} parameter variations...")
    results_df = analyzer.analyze_variations(variations, config)

    # Export results
    print("\n💾 Exporting results...")
    from circuit_analysis import export_data
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

    print(f"\n🎉 Grid analysis complete! Results saved to: {output_path}")
    print("\nNext steps:")
    print(f"  1. Run: python scripts/create_visualizations.py {output_path}")
    print(f"  2. Open: {output_path}/visualizations/summary_report.html")

    # Print summary statistics
    print(f"\n📊 Summary Statistics:")
    print(f"   Resnorm range: {results_df['resnorm'].min():.6f} - {results_df['resnorm'].max():.6f}")
    finite_condition_numbers = results_df['condition_number'][results_df['condition_number'] != float('inf')]
    if len(finite_condition_numbers) > 0:
        print(f"   Finite condition numbers: {len(finite_condition_numbers)}/{len(results_df)}")
        print(f"   Condition number range: {finite_condition_numbers.min():.2f} - {finite_condition_numbers.max():.2f}")
    else:
        print(f"   All condition numbers are infinite (singular matrices)")
    print(f"   PC1 sensitivity range: {results_df['pc1_sensitivity'].min():.2f} - {results_df['pc1_sensitivity'].max():.2f}")

if __name__ == "__main__":
    run_grid_example()