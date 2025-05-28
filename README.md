# Circuit Simulator & Spider Plot Visualization

This project provides an interactive circuit simulator with visualization tools including spider plots, nyquist plots, and residual norm analytics.

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
- Residual norm calculations for model comparison
- Mathematical insights into impedance calculations

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
[![Version](https://img.shields.io/badge/version-1.0.0_alpha-blue.svg)](https://github.com/tkpepper15/nei-viz-project/releases/tag/v1.0.0-alpha)

> 🎉 RPE Impedance Simulator Alpha Release - Foundation for interactive RPE electrical property visualization

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
- Frequency response from 1 Hz to 10 kHz
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

The equivalent circuit model is described by the following equations:

### Total Equivalent Impedance
```
Zeq(ω) = Rblank + [Rs(Za(ω) + Zb(ω))] / [Rs + Za(ω) + Zb(ω)]
```

### Apical Impedance
```
Za(ω) = Ra + 1/(jωCa)
```

### Basal Impedance
```
Zb(ω) = Rb + 1/(jωCb)
```

Where:
- ω is the angular frequency (2πf)
- j is the imaginary unit
- All resistances are in Ohms (Ω)
- All capacitances are in Farads (F)

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

---

<div align="center">
  <sub>Built with ❤️ to accelerate Vision Research</sub>
</div>
