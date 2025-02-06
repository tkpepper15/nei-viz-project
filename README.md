[Web Access](https://nei-viz-project.vercel.app/)

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
- Parameters include:
  - Blank resistance (Rblank): 10-50 Ω
  - Shunt resistance (Rs): 0.1-10 kΩ
  - Apical resistance (Ra): 0.1-10 kΩ
  - Apical capacitance (Ca): 0.1-10 µF
  - Basal resistance (Rb): 0.1-10 kΩ
  - Basal capacitance (Cb): 0.1-10 µF

### Visualizations
1. **Nyquist Plot**
   - Real vs. imaginary impedance
   - Frequency response visualization
   - Interactive tooltips with detailed measurements

2. **Spider Plot**
   - Parameter space visualization
   - Normalized parameter comparison
   - Multi-state overlay support

### State Management
- Save and load multiple states
- Compare parameter changes
- Visual state tracking
- Customizable state colors and visibility

## Mathematical Model

The equivalent circuit model is described by the following equations:

----

Built on a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `/app/components/CircuitSimulator.tsx`. The page auto-updates as you edit the file.
