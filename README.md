# Circuit Simulator & Spider Plot Visualization for EIS

This project provides an interactive circuit simulator with visualization tools for electrochemical impedance spectroscopy (EIS), including spider plots, nyquist plots, and residual norm analytics.

## Project Structure

The project follows modern Next.js application architecture with a clear separation of concerns:

```
src/
├── app/                  # Next.js app router
│   ├── components/       # App-specific components
│   ├── globals.css       # Global styles
│   ├── layout.tsx        # Root layout
│   └── page.tsx          # Homepage
├── components/           # Shared React components
│   └── circuit-simulator/ # Circuit simulator components
│       ├── controls/     # UI controls
│       ├── visualizations/ # Visualization components
│       └── CircuitSimulator.tsx # Main component
├── lib/                  # Shared libraries 
│   └── python/           # Python backend code
├── types/                # TypeScript type definitions
│   └── circuit-simulator.ts # Circuit simulator types
├── utils/                # Utility functions
│   └── circuit-simulator/ # Circuit-specific utilities
├── hooks/                # Custom React hooks
├── services/             # API services
└── styles/               # Shared styles
```

## Core Features

- Interactive circuit simulation with RC-RC circuit model
- Parameter visualization via spider plots
- Nyquist plot for impedance visualization
- Residual norm calculations for model comparison with frequency weighting
- Mathematical insights into impedance calculations
- Grid parameter space exploration

## Recent Improvements

### Enhanced Frequency Controls
- Extended minimum frequency from 0.1Hz down to 0.01Hz for better low-frequency response analysis
- Added more precise tick marks on logarithmic sliders for improved usability
- Added a control for number of frequency points (10-200) to adjust computational detail

### Complete Grid Point Computation
- Modified to display all computed parameter combinations instead of just a subset
- Enhanced `generateGridPoints` function to compute all possible parameter combinations
- Updated UI to reflect the total points being displayed

### Improved Resnorm Calculations
- Enhanced the residual norm calculation to follow industry-standard practices for EIS
- Added frequency-weighting to give more importance to lower frequencies
- Improved normalization using reference magnitude
- Added detailed debugging information showing calculation steps
- Created better data structures with proper type definitions

## Getting Started

### Prerequisites

- Node.js 18+ 
- Python 3.8+

### Installation

1. Clone the repository
2. Install dependencies:

```bash
yarn install
pip install -r requirements.txt
```

### Development

Start the development server:

```bash
yarn dev
```

This will start both the Next.js frontend and the Python backend for circuit calculations.

### Building for Production

```bash
yarn build
```

## Technologies

- Next.js 14
- React 18
- TypeScript
- Python for backend calculations
- Material UI components
- D3.js and Recharts for visualizations

# RPE Impedance Simulator

[![Live Demo](https://img.shields.io/badge/demo-online-green.svg)](https://nei-viz-project.vercel.app/)
[![Next.js](https://img.shields.io/badge/built%20with-Next.js-black)](https://nextjs.org)
[![TailwindCSS](https://img.shields.io/badge/styled%20with-TailwindCSS-06B6D4)](https://tailwindcss.com)
[![Version](https://img.shields.io/badge/version-1.1.0-blue.svg)](https://github.com/tkpepper15/nei-viz-project/releases/tag/v1.1.0)

> 🎉 RPE Impedance Simulator - Advanced circuit modeling and visualization for electrochemical impedance spectroscopy

An interactive web application for simulating and visualizing retinal pigment epithelium (RPE) impedance characteristics using equivalent circuit models.

<div align="center">
  <img src="public/screenshot.png" alt="RPE Impedance Simulator Screenshot" width="800"/>
</div>

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/tkpepper15/nei-viz-project.git

# Install dependencies
npm install

# Start the development server
npm run dev
```

Then open [http://localhost:3000](http://localhost:3000) to see the application.

## 🎯 Overview

This simulator empowers researchers and students to:
- 🔬 Explore RPE electrical properties through an equivalent circuit model
- 📊 Visualize impedance characteristics using Nyquist plots
- 🕸️ Analyze parameter relationships through spider plots
- 📈 Compare multiple states and track changes
- 💾 Save and load simulation states locally
- 🧮 Explore the full parameter space with enhanced computational capabilities

## ✨ Features

### 🔧 Circuit Model
- Complete equivalent circuit model for RPE cells
- Interactive parameter controls with real-time updates
- Parameters include:
  - Blank resistance (Rblank): 10-50 Ω
  - Shunt resistance (Rs): 0.1-10 kΩ
  - Apical resistance (Ra): 0.1-10 kΩ
  - Apical capacitance (Ca): 0.1-10 µF
  - Basal resistance (Rb): 0.1-10 kΩ
  - Basal capacitance (Cb): 0.1-10 µF

### 📊 Visualizations

#### Nyquist Plot
- Real vs. imaginary impedance visualization
- Frequency response from 0.01 Hz to 10 kHz
- Interactive tooltips with detailed measurements:
  - Frequency (Hz)
  - Real impedance (Ω)
  - Imaginary impedance (Ω)
  - Magnitude (Ω)
  - Phase (degrees)

#### Spider Plot
- Parameter space visualization
- Normalized parameter comparison (0-1 scale)
- Multi-state overlay support
- Real-time parameter mapping

### 💾 State Management
- Save and load multiple simulation states
- Compare parameter changes across states
- Visual state tracking with customizable colors
- Toggle state visibility in plots
- Real-time parameter updates
- Total Epithelial Resistance (TER) calculation

## 📐 Mathematical Model

### Equivalent Circuit Model

The RPE cellular layer is modeled as an equivalent circuit with the following components:

```
Rs ────┬────────────┬──────
       │            │
       Ra           Rb
       │            │
       Ca           Cb
       │            │
       └────────────┘
```

Where:
- Rs: Shunt resistance (paracellular pathway)
- Ra: Apical membrane resistance
- Ca: Apical membrane capacitance
- Rb: Basal membrane resistance
- Cb: Basal membrane capacitance

### Impedance Calculation

The impedance at a given frequency is calculated as:

```
Zeq(ω) = Rs + Za(ω) + Zb(ω)
```

Where Za and Zb are the impedances of the apical and basal membranes:

```
Za(ω) = Ra/(1 + jωRaCa)
Zb(ω) = Rb/(1 + jωRbCb)
```

### Residual Norm (Resnorm) Calculation

The resnorm quantifies the difference between a test impedance spectrum and a reference spectrum. Our enhanced calculation includes:

1. **Frequency Weighting**: Lower frequencies are weighted more heavily to emphasize capacitive effects:
   ```
   frequencyWeight = 1 / max(0.1, log10(frequency))
   ```

2. **Magnitude Normalization**: Residuals are normalized by the reference impedance magnitude:
   ```
   normalizedResidual = (Z_test - Z_ref) / |Z_ref|
   ```

3. **Component Weighting**: Different weights for real and imaginary components:
   ```
   For low frequencies (<100 Hz):
     realWeight = 1.0, imagWeight = 1.5
   For high frequencies (≥100 Hz):
     realWeight = 1.5, imagWeight = 1.0
   ```

4. **Final Calculation**:
   ```
   resnorm = sqrt(sum(weighted squared residuals) / sum(weights)) * rangeAmplifier
   ```

   Where rangeAmplifier adjusts based on the frequency range ratio:
   ```
   rangeAmplifier = 
     3.0 if ratio < 100 (narrow range)
     2.5 if ratio < 1000 (moderate range)
     2.0 otherwise (wide range)
   ```

5. **Frequency Range Impact**: The simulator allows adjustment of the frequency range from 0.01 Hz to 10 kHz:
   - Very low frequencies (0.01-1 Hz) are crucial for resolving capacitive elements (Ca, Cb)
   - Mid-range frequencies (1-100 Hz) highlight the RC time constants and membrane properties
   - High frequencies (>1 kHz) emphasize series resistance (Rs)
   - The number of frequency points (10-200) controls the resolution of the analysis

This approach ensures:
- Low-frequency capacitive behavior is emphasized
- Impedance differences are properly normalized
- The calculation is robust across different frequency ranges
- The results align with industry-standard EIS analysis methods

### Parameter Space Exploration

The simulator explores the full parameter space by:

1. Generating all possible combinations of the circuit parameters
2. Computing the impedance spectrum for each parameter set
3. Calculating the resnorm between each generated spectrum and the reference
4. Grouping results by resnorm quality into:
   - Very Good Fits (lowest resnorm)
   - Good Fits
   - Moderate Fits
   - Poor Fits (highest resnorm)

This comprehensive approach allows users to visualize the entire solution space and understand which parameter combinations produce similar impedance responses.

## 🛠️ Technical Stack

- **Framework**: [Next.js](https://nextjs.org/)
- **Styling**: [TailwindCSS](https://tailwindcss.com/)
- **Visualization**: [Recharts](https://recharts.org/)
- **Math Typesetting**: [KaTeX](https://katex.org/)

## 🧑‍💻 Development

The main component is located at `/app/components/CircuitSimulator.tsx`. The application features hot reloading, so the page will auto-update as you edit the file.

### Project Structure
```
nei-viz-project/
├── app/
│   ├── components/
│   │   └── CircuitSimulator.tsx    # Main simulator component
│   ├── layout.tsx                  # App layout
│   └── page.tsx                    # Main page
├── public/                         # Static assets
└── styles/                         # Global styles
```

## 📄 License

MIT License - feel free to use this project for research, education, or any other purpose.

## 🙏 Acknowledgments

This project was created to support research in retinal physiology and provide an interactive tool for understanding RPE electrical properties. Special thanks to:

- The NEI Visual Function Core for supporting this work
- The research community for valuable feedback and suggestions

## 📚 References

1. [New technique enhances quality control of lab-grown cells for AMD treatment](https://www.nei.nih.gov/about/news-and-events/news/new-technique-enhances-quality-control-lab-grown-cells-amd-treatment)
2. [NEI Ocular and Stem Cell Translational Research Section](https://www.nei.nih.gov/research/research-labs-and-branches/ocular-and-stem-cell-translational-research-section)
3. [Basics of Electrochemical Impedance Spectroscopy](https://www.gamry.com/application-notes/EIS/basics-of-electrochemical-impedance-spectroscopy/)
4. [Impedance Spectroscopy: Theory, Experiment, and Applications](https://onlinelibrary.wiley.com/doi/book/10.1002/9781119333623)

---

<div align="center">
  <sub>Built with ❤️ to accelerate Vision Research</sub>
</div>
