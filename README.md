# RPE Impedance Simulator

An interactive web application for simulating and visualizing retinal pigment epithelium (RPE) impedance characteristics using equivalent circuit models.

## Overview

This simulator allows researchers and students to:
- Explore RPE electrical properties through an equivalent circuit model
- Visualize impedance characteristics using Nyquist plots
- Analyze parameter relationships through spider plots
- Compare multiple states and track changes
- Export and save simulation states

## Features

### Circuit Model
- Complete equivalent circuit model for RPE cells
- Interactive parameter controls with real-time updates
- Parameters include:
  - Blank resistance (Rblank): 10-50 Ω
  - Shunt resistance (Rs): 0.1-10 kΩ
  - Apical resistance (Ra): 0.1-10 kΩ
  - Apical capacitance (Ca): 0.1-10 µF
  - Basal resistance (Rb): 0.1-10 kΩ
  - Basal capacitance (Cb): 0.1-10 µF

### Visualizations
1. **Nyquist Plot**
   - Real vs. imaginary impedance visualization
   - Frequency response from 1 Hz to 10 kHz
   - Interactive tooltips with detailed measurements:
     - Frequency (Hz)
     - Real impedance (Ω)
     - Imaginary impedance (Ω)
     - Magnitude (Ω)
     - Phase (degrees)

2. **Spider Plot**
   - Parameter space visualization
   - Normalized parameter comparison (0-1 scale)
   - Multi-state overlay support
   - Real-time parameter mapping

### State Management
- Save and load multiple simulation states
- Compare parameter changes across states
- Visual state tracking with customizable colors
- Toggle state visibility in plots
- Real-time parameter updates
- Total Epithelial Resistance (TER) calculation

## Mathematical Model

The equivalent circuit model is described by the following equations:

1. **Total Equivalent Impedance:**
   ```
   Zeq(ω) = Rblank + [Rs(Za(ω) + Zb(ω))] / [Rs + Za(ω) + Zb(ω)]
   ```

2. **Apical Impedance:**
   ```
   Za(ω) = Ra + 1/(jωCa)
   ```

3. **Basal Impedance:**
   ```
   Zb(ω) = Rb + 1/(jωCb)
   ```

Where:
- ω is the angular frequency (2πf)
- j is the imaginary unit
- All resistances are in Ohms (Ω)
- All capacitances are in Farads (F)

## Technical Details

Built with:
- [Next.js](https://nextjs.org/) - React framework
- [Recharts](https://recharts.org/) - Charting library
- [KaTeX](https://katex.org/) - Math typesetting
- [TailwindCSS](https://tailwindcss.com/) - Styling

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rpe-impedance-simulator.git
   ```

2. Install dependencies:
   ```bash
   npm install
   # or
   yarn install
   ```

3. Run the development server:
   ```bash
   npm run dev
   # or
   yarn dev
   ```

4. Open [http://localhost:3000](http://localhost:3000) with your browser to see the application.

## Development

The main component is located at `/app/components/CircuitSimulator.tsx`. The page auto-updates as you edit the file.

## License

MIT License - feel free to use this project for research, education, or any other purpose.

## Acknowledgments

This project was created to support research in retinal physiology and provide an interactive tool for understanding RPE electrical properties.
