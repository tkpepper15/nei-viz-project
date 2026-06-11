# Project Directory Structure

Visual representation of the organized nei-viz-project structure.

```
nei-viz-project/
│
├── 📱 APPLICATION CODE
│   ├── app/                           # Next.js application
│   │   ├── components/               # React components
│   │   │   ├── CircuitSimulator.tsx # Main orchestrator
│   │   │   ├── auth/                # Authentication
│   │   │   ├── settings/            # Settings UI
│   │   │   ├── npz/                 # NPZ management
│   │   │   └── circuit-simulator/   # Core circuit sim
│   │   │       ├── controls/        # UI controls
│   │   │       ├── visualizations/  # Charts & plots
│   │   │       ├── utils/           # Computation engine
│   │   │       ├── types/           # TypeScript types
│   │   │       ├── hooks/           # React hooks
│   │   │       └── npz-manager/     # NPZ tools
│   │   ├── hooks/                   # Global hooks
│   │   └── (landing)/               # Landing page
│   ├── lib/                          # Services
│   │   ├── supabase.ts              # Supabase client
│   │   └── *Service.ts              # Service layers
│   └── public/                       # Static assets
│       └── grid-worker.js           # Web worker
│
├── 🐍 BACKEND CODE
│   ├── python-scripts/               # Backend services
│   │   ├── circuit_api.py           # Flask API (port 5001)
│   │   ├── npz_loader.py            # Data loading
│   │   ├── circuit_computation.py   # Computation
│   │   └── *_serializer.py          # Serialization
│   ├── ml_ideation/                  # ML research
│   │   ├── complete_pipeline*.py    # Training pipelines
│   │   ├── eis_predictor*.py        # ML models
│   │   └── *.md                     # ML docs
│   └── scripts/                      # Analysis scripts
│       ├── circuit_analysis.py      # Parameter analysis
│       └── create_visualizations.py # Viz generation
│
├── 📚 DOCUMENTATION
│   ├── docs/
│   │   ├── INDEX.md                 # Documentation index
│   │   ├── architecture/            # System design
│   │   │   ├── APP_STRUCTURE.md
│   │   │   ├── ROUTING_ARCHITECTURE.md
│   │   │   ├── COMPUTATION_PIPELINE_DOCUMENTATION.md
│   │   │   └── OPTIMIZATION_MATHEMATICS.md
│   │   ├── setup/                   # Setup guides
│   │   │   ├── SETUP_INSTRUCTIONS.md
│   │   │   ├── SUPABASE_IMPLEMENTATION_PLAN.md
│   │   │   ├── UI_SETTINGS_DATABASE_SCHEMA.md
│   │   │   └── database-schemas/   # SQL schemas
│   │   ├── ml/                      # ML documentation
│   │   │   ├── ML_WEB_INTEGRATION_GUIDE.md
│   │   │   ├── QUICK_START_MLX.md
│   │   │   ├── GEOMETRIC_PARAMETER_COMPLETION.md
│   │   │   └── ML_IMPLEMENTATION_PLAN.md
│   │   ├── deployment/              # Deployment
│   │   │   ├── README-Docker.md
│   │   │   ├── deploy-github-registry.md
│   │   │   └── deploy-to-unraid.md
│   │   ├── reference/               # Technical refs
│   │   │   ├── CURRENT_STATUS.md
│   │   │   ├── PERFORMANCE_ANALYSIS.md
│   │   │   └── python_scripts.md
│   │   └── archive/                 # Historical docs
│   │       ├── CLAUDE_inspiration.md
│   │       ├── designrevisions.md
│   │       └── planning/
│   ├── CLAUDE.md                    # Development guidelines
│   ├── README.md                    # Project overview
│   └── ORGANIZATION.md              # This structure doc
│
├── 🛠️ TOOLS & UTILITIES
│   ├── tools/
│   │   └── database/                # DB testing tools
│   │       ├── check-database.js
│   │       ├── test-*-workflow.js
│   │       └── *.sh
│   └── reference/                   # Reference code
│       └── checkResnorm.m          # MATLAB reference
│
├── 🗄️ DATABASE
│   ├── supabase/                    # Active migrations
│   │   └── migrations/             # Timestamped migrations
│   ├── supabase-sql-simple/         # Setup scripts
│   │   └── 00-10*.sql              # Numbered setup
│   └── sql-scripts/                 # Ad-hoc scripts
│
├── 📦 DATA
│   └── data/
│       ├── measurement_presets/    # EIS configs
│       └── npz/                    # Dataset storage
│
├── ⚙️ CONFIGURATION
│   ├── next.config.ts              # Next.js config
│   ├── tailwind.config.ts          # Tailwind config
│   ├── tsconfig.json               # TypeScript config
│   ├── package.json                # Dependencies
│   ├── vercel.json                 # Deployment config
│   └── start_backend.sh            # Backend startup
│
└── 📊 OUTPUT
    └── analysis_output/            # Analysis results
        └── *_2025-*/               # Timestamped outputs
```

## Key Directories

### Application (`/app`)
**Purpose**: Next.js 15 application with React components
**Entry Point**: `app/components/CircuitSimulator.tsx`
**Key Areas**: 
- Components: UI and visualization
- Hooks: State management
- Types: TypeScript definitions

### Backend (`/python-scripts`, `/ml_ideation`)
**Purpose**: Backend computation and ML services
**Key Files**:
- `circuit_api.py` - Flask API server
- `complete_pipeline_mlx.py` - ML training
- Circuit computation engine

### Documentation (`/docs`)
**Purpose**: All project documentation
**Navigation**: Start with `docs/INDEX.md`
**Categories**: Architecture, Setup, ML, Deployment, Reference, Archive

### Tools (`/tools`)
**Purpose**: Development and debugging utilities
**Key Tools**: Database testing, workflow validation

### Data (`/data`)
**Purpose**: Application data and presets
**Contents**: Measurement configs, NPZ datasets

### Database (`/supabase`, `/sql-scripts`)
**Purpose**: Database schemas and migrations
**Active**: `supabase/migrations/` for production changes

## Navigation Guide

**For Developers**:
1. Start with `CLAUDE.md` for guidelines
2. Check `docs/architecture/` for system design
3. Look in `app/components/` for UI code
4. Backend is in `python-scripts/`

**For Documentation**:
1. Go to `docs/INDEX.md` for complete navigation
2. Browse by category in `docs/*/`
3. Check `ORGANIZATION.md` for structure overview

**For Setup**:
1. Read `README.md` for overview
2. Follow `docs/setup/SETUP_INSTRUCTIONS.md`
3. Run `start_backend.sh` for backend
4. Check `docs/deployment/` for production

---

**Last Updated**: October 7, 2025
**Maintained**: Automatically updated on structural changes
