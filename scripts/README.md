# Circuit Parameter Analysis Scripts

This directory contains Python scripts for systematic circuit parameter analysis with PCA and visualization generation.

## 🚀 Quick Start

### 0. Install Dependencies
```bash
pip install -r scripts/requirements.txt
```

### 1. Run Circuit Analysis
```bash
python scripts/circuit_analysis.py
```

Interactive prompts will guide you through:
- **Ground Truth Circuit**: Base parameters (Rsh, Ra, Ca, Rb, Cb)
- **Variation Ranges**: ±percentage for each parameter
- **Analysis Settings**: Number of variations, frequency range
- **Output Configuration**: Run name and directory

**Batch mode (with defaults):**
```bash
python scripts/circuit_analysis.py --batch
```

**Quick grid example:**
```bash
python scripts/run_grid_example.py
```

### 2. Generate Visualizations
```bash
python scripts/create_visualizations.py
```

Or specify a specific analysis directory:
```bash
python scripts/create_visualizations.py ./analysis_output/circuit_analysis_2025-01-15
```

**List available analysis directories:**
```bash
python scripts/create_visualizations.py --list
```

## 📊 Output Structure

```
analysis_output/
└── {run_name}_{date}/
    ├── analysis_config.json          # Configuration used
    ├── csv/
    │   ├── parameters.csv            # Circuit parameters + resnorm + condition numbers
    │   ├── spectra.csv              # Complete impedance spectra for each variation
    │   └── pca_directions.csv       # Principal component directions and sensitivities
    └── visualizations/
        ├── correlation_heatmap.html  # Parameter correlation matrix
        ├── pca_analysis.html        # PCA loadings and score plots
        ├── pairs_plots.html         # Parameter relationship grid
        └── summary_report.html      # Overview with key findings
```

## 🔬 Analysis Features

### Circuit Parameter Variations
- **Systematic sampling** around ground truth circuit
- **Configurable ranges** for each parameter (±percentage)
- **Frequency response** analysis across specified range

### PCA & Directional Analysis
- **Analytic Jacobian** computation for exact sensitivities
- **Principal components** identify parameter combinations with biggest impact
- **Condition number** analysis for parameter identifiability

### Visualizations
- **Correlation heatmaps** with color-coded correlation coefficients
- **PCA score plots** colored by resnorm values
- **Component loadings** showing parameter contributions
- **Pairs plots** for detailed parameter relationships

## 🧪 Example Use Cases

### 1. Parameter Sensitivity Study
```bash
# Study how ±20% variations affect impedance
Ground Truth: Ra=5000Ω, Ca=25μF, Rb=5000Ω, Cb=25μF, Rsh=25Ω
Variations: ±20% each parameter
Number: 200 variations
```

### 2. Frequency Range Impact
```bash
# Compare analysis at different frequency ranges
Low: 1-1000 Hz (membrane-dominated)
High: 100-10000 Hz (resistance-dominated)
```

### 3. Parameter Coupling Analysis
```bash
# Investigate Ra-Ca vs Rb-Cb coupling
Asymmetric: Ra≠Rb, Ca≠Cb
Symmetric: Ra=Rb, Ca=Cb
```

## 📈 Interpretation Guide

### Correlation Coefficients
- **|r| > 0.7**: Strong correlation
- **0.3 < |r| < 0.7**: Moderate correlation
- **|r| < 0.3**: Weak correlation

### PCA Components
- **PC1**: Usually dominated by resistance values (Rsh, Ra, Rb)
- **PC2**: Often capacitance-related (Ca, Cb)
- **PC3+**: Higher-order parameter interactions

### Condition Numbers
- **< 10**: Well-conditioned (parameters easily distinguishable)
- **10-100**: Moderately conditioned
- **> 100**: Ill-conditioned (parameter estimation difficult)

## 🛠 Customization

### Adding New Parameters
1. Modify `CircuitParameters` interface in `types/parameters.ts`
2. Update parameter ranges in analysis scripts
3. Extend visualization functions for new dimensions

### Custom Visualizations
The visualization script generates HTML files with Plotly.js. You can:
- Modify color schemes in `generateCorrelationHeatmapHTML()`
- Add new plot types in `generatePCAVisualizationHTML()`
- Customize layouts and styling

### Export Formats
Current exports include CSV and HTML. To add other formats:
- **JSON**: Add `exportJSON()` functions
- **MATLAB**: Export `.mat` files with parameter matrices
- **Python**: Generate `.npz` files for NumPy/SciPy

## 🔧 Dependencies

Required packages (already in project):
- **TypeScript**: Script execution
- **Node.js fs/path**: File operations
- **readline**: Interactive CLI
- **Plotly.js**: Web-based visualizations (CDN)

## 📝 Script Details

### `circuit-analysis.ts`
- Interactive parameter configuration
- Systematic parameter variation generation
- Directional analysis and PCA computation
- CSV data export with organized directory structure

### `create-visualizations.ts`
- CSV data parsing and validation
- Correlation matrix computation
- PCA eigenvalue decomposition
- HTML visualization generation with embedded JavaScript

Both scripts include comprehensive error handling and progress reporting for large parameter sets.