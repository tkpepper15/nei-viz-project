# SpideyPlot v3.0

**Ultra-high-performance electrochemical impedance spectroscopy (EIS) simulation platform for retinal pigment epithelium (RPE) research**

---

## Quick Start

### Development Server (frontend + backend together)

```bash
./dev.sh
# Frontend: http://localhost:3000
# ML API:   http://localhost:5003
```

### Frontend only

```bash
npm run dev
```

### ML API only

```bash
cd pipeline
./start_api.sh
# Starts http://localhost:5003
```

---

## Project Structure

```
nei-viz-project/
├── app/                          # Next.js 15 application
│   ├── components/               # React components
│   ├── lib/                      # Utilities and hooks
│   └── page.tsx                  # Main entry point
│
├── pipeline/                     # ML pipeline and Flask API
│   ├── backend_api.py            # Flask API (port 5003): MDN inference + GPF tracking
│   ├── src/                      # Core library (models, physics, pipeline)
│   ├── eval/                     # Evaluation suite (E1-E4 experiments)
│   ├── simulation/               # Temporal dataset generation
│   ├── scripts/                  # Figure and analysis scripts
│   ├── docs/                     # Background documentation
│   └── README.md                 # ML pipeline overview and key results
│
├── lib/                          # Shared TypeScript utilities
├── public/                       # Static assets and Web Workers
├── supabase/                     # Supabase configuration
├── sql-scripts/                  # Database schemas
├── docs/                         # Project-level documentation
└── data/                         # Measurement presets
```

---

## Technology Stack

### Frontend
- **Framework:** Next.js 15 with App Router
- **Language:** TypeScript
- **Styling:** TailwindCSS
- **UI:** Material-UI + custom components
- **Math:** KaTeX for LaTeX rendering
- **Computation:** Web Workers + WebGPU + optimized algorithms

### Backend
- **API:** Flask (Python 3.11+) on port 5003
- **Database:** Supabase PostgreSQL
- **Auth:** Supabase Authentication

### ML Pipeline
- **Framework:** PyTorch
- **Model:** FisherAwareTransformer (MDN, K=3 components) — fisher_v10, val_mae=0.2166
- **Filter:** Gaussian Particle Filter (128 particles, IEKF update) with FFBS smoother
- **Dataset:** mixed_distribution_v2 (50k/5k/5k train/val/test, 100 frequencies)

---

## Features

### 3-Tier Computation Pipeline
1. **Web Workers:** Multi-core parallel processing
2. **WebGPU:** GPU-accelerated computation
3. **Optimized:** Advanced algorithms for massive parameter spaces

### ML-Powered Predictions
- **Inverse solver:** FisherAwareTransformer predicts identifiable parameters [tau_big, tau_small, TER, TEC, Rsh] from impedance spectra
- **Temporal tracking:** Gaussian Particle Filter tracks parameter evolution over time with FFBS smoothing
- **Physics-informed:** Enforces circuit equations and biological priors

### Visualization
- **Spider Plots 3D:** Advanced parameter space exploration
- **Nyquist Plots:** Complex impedance visualization
- **Circuit Diagrams:** Interactive equivalent circuits

---

## Development

### Prerequisites

```bash
Node.js 18+
Python 3.11+
npm install
pip install -r pipeline/requirements.txt
```

### Environment Variables

```bash
cp .env.example .env.local
# Add Supabase credentials and NEXT_PUBLIC_ML_API_URL=http://localhost:5003
```

### Build

```bash
npm run build      # Production build
npm run lint       # Lint check
```

---

## Documentation

- **ML Pipeline:** `pipeline/README.md`
- **Directory structure:** `pipeline/STRUCTURE.md`
- **Background docs:** `pipeline/docs/`
- **Architecture:** `docs/`

---

## Circuit Model

Modified Randles circuit for RPE impedance:

```
       Rsh (Shunt/Paracellular)
   ────[Rsh]────┬──────────┬──────
                │          │
            [Ra]│      [Rb]│
                │          │
            [Ca]│      [Cb]│
                │          │
                └──────────┘
    Apical RC       Basolateral RC
```

**TER:** `TER = Rsh * (Ra + Rb) / (Ra + Rb + Rsh)`

---

## License

Research project - NEI/RPE studies
