# Project Organization Summary

**Last Updated**: October 7, 2025
**Project**: SpideyPlot v3.0

This document describes the organizational structure of the nei-viz-project after the cleanup and reorganization performed on October 7, 2025.

---

## 📁 Top-Level Directory Structure

```
nei-viz-project/
├── app/                        # Next.js application (React components)
├── docs/                       # All documentation (organized by category)
├── lib/                        # External service integrations (Supabase)
├── public/                     # Static assets and web workers
├── python-scripts/             # Backend Python code (Flask API)
├── ml_ideation/                # ML research and experimental code
├── scripts/                    # Analysis and utility scripts
├── tools/                      # Development tools
├── reference/                  # Reference implementations (MATLAB)
├── data/                       # Data files and presets
├── supabase/                   # Database migrations
├── supabase-sql-simple/        # Simplified SQL setup scripts
├── sql-scripts/                # Ad-hoc SQL scripts
├── analysis_output/            # Analysis results
├── CLAUDE.md                   # Development guidelines (ESSENTIAL)
├── README.md                   # Project overview
└── start_backend.sh            # Backend startup script
```

---

## 📚 Documentation Organization

### `/docs` - All Documentation

Documentation is now organized into clear categories:

```
docs/
├── INDEX.md                    # Complete documentation index
├── architecture/               # System architecture
│   ├── APP_STRUCTURE.md
│   ├── ROUTING_ARCHITECTURE.md
│   ├── COMPUTATION_PIPELINE_DOCUMENTATION.md
│   └── OPTIMIZATION_MATHEMATICS.md
├── setup/                      # Setup and configuration
│   ├── SETUP_INSTRUCTIONS.md
│   ├── SUPABASE_IMPLEMENTATION_PLAN.md
│   ├── UI_SETTINGS_DATABASE_SCHEMA.md
│   └── database-schemas/       # SQL schema definitions
├── ml/                         # Machine learning
│   ├── ML_WEB_INTEGRATION_GUIDE.md
│   ├── QUICK_START_MLX.md
│   ├── GEOMETRIC_PARAMETER_COMPLETION.md
│   └── ML_IMPLEMENTATION_PLAN.md
├── deployment/                 # Deployment guides
│   ├── README-Docker.md
│   ├── deploy-github-registry.md
│   └── deploy-to-unraid.md
├── reference/                  # Technical references
│   ├── CURRENT_STATUS.md
│   ├── PERFORMANCE_ANALYSIS.md
│   ├── performance_fixes.md
│   ├── python_scripts.md
│   └── test_enhanced_functionality.md
└── archive/                    # Historical docs
    ├── CLAUDE_inspiration.md
    ├── designrevisions.md
    ├── specs.md
    └── planning/               # Planning documents
```

**Key Documentation Files:**
- **Start Here**: [`docs/INDEX.md`](docs/INDEX.md) - Navigation guide to all docs
- **Development**: [`CLAUDE.md`](CLAUDE.md) - Essential development guidelines
- **Overview**: [`README.md`](README.md) - Project overview and quick start

---

## 🛠️ Tools Organization

### `/tools` - Development Tools

```
tools/
├── database/                   # Database utilities
│   ├── check-database.js
│   ├── cleanup-duplicates.js
│   ├── debug-circuit-config.js
│   ├── test-supabase-profiles.js
│   ├── test-full-workflow.js
│   ├── test-final-workflow.js
│   └── apply-ui-settings-migrations.sh
└── testing/                    # Test utilities (future)
```

**Purpose**: Development and debugging tools that aren't part of the main application.

---

## 🐍 Python Code Organization

### `/python-scripts` - Backend Services

**Core Backend**:
- `circuit_api.py` - Flask API server (port 5001)
- `npz_loader.py` - NPZ dataset loading
- `circuit_computation.py` - Circuit computation engine

**Utilities**:
- `config_serializer.py` - Configuration serialization
- `frequency_serializer.py` - Frequency serialization
- `npz_supabase_sync.py` - Database sync
- `measurement_config.py` - Measurement presets
- `lightweight_storage.py` - Experimental storage

### `/ml_ideation` - ML Research

**ML Training**:
- `complete_pipeline.py` - PyTorch pipeline
- `complete_pipeline_mlx.py` - Apple MLX pipeline
- `dataset_generation_system.py` - Training data generation
- `eis_predictor_implementation.py` - PyTorch model
- `eis_predictor_mlx.py` - MLX model
- `ml_api.py` - Standalone ML API (optional)

**Documentation**:
- `README_MLX.md` - MLX setup guide
- `TRAINING_GUIDE.md` - PyTorch training
- `MLX_TRAINING_GUIDE.md` - MLX training
- `COMPARISON_PYTORCH_VS_MLX.md` - Framework comparison
- `eis_ml_strategy.md` - ML strategy
- `implementation_guide.md` - Implementation details
- `strategy_summary.md` - Strategy overview

### `/scripts` - Analysis Scripts

**Analysis Tools**:
- `circuit_analysis.py` - Parameter variation analysis
- `create_visualizations.py` - Visualization generator
- `run_grid_example.py` - Grid mode example

---

## 📦 Data Organization

### `/data` - Data Files

```
data/
├── measurement_presets/        # EIS measurement configurations
│   ├── standard_eis.json
│   ├── fast_sweep.json
│   ├── high_resolution.json
│   ├── linear_sweep.json
│   ├── low_freq_focus.json
│   └── summary.json
└── npz/                        # NPZ datasets
    └── README.md               # NPZ format documentation
```

---

## 🗄️ Database Organization

### Active Database Files

**Supabase Migrations** (`/supabase/migrations/`):
- Official migration files with timestamps
- Complete migration history
- **Location**: Keep in place (actively used)

**Simple Setup** (`/supabase-sql-simple/`):
- Simplified setup scripts (00-10 numbered)
- Fresh installation setup
- **Location**: Keep in place (setup tool)

**Ad-hoc Scripts** (`/sql-scripts/`):
- Various SQL utilities
- NPZ integration scripts
- **Location**: Keep in place (reference)

**Schema Documentation** (`/docs/setup/database-schemas/`):
- Table definitions
- Migration plans
- Implementation guides
- **Location**: Moved to docs for organization

---

## 📖 Reference Materials

### `/reference` - Reference Implementations

**Contents**:
- `checkResnorm.m` - MATLAB resnorm calculation (algorithm reference)

**Purpose**: Reference implementations for algorithm validation.

---

## 🗑️ Removed Files

### Cleanup Summary (October 7, 2025)

**Total Files Removed**: 57 files

**Categories**:
- **React Components**: 15 files (unused UI components)
- **Utilities**: 8 files (unused computation utilities)
- **Workers**: 3 files (unused web workers)
- **Python**: 2 files (legacy backend code)
- **Documentation**: 9 files (obsolete serialization docs)
- **Config**: 13 files (duplicates, old data)
- **Demo Code**: 2 directories (`demos/`, `examples/`)

**Storage Freed**: ~14MB (primarily 11MB JSON snapshot)

**Backup Location**: `/tmp/deleted_files_backup_20251007_164520`

---

## 🎯 Quick Reference

### For Developers

**Essential Files to Know**:
1. [`CLAUDE.md`](CLAUDE.md) - Development guidelines
2. [`docs/INDEX.md`](docs/INDEX.md) - Documentation index
3. [`docs/architecture/APP_STRUCTURE.md`](docs/architecture/APP_STRUCTURE.md) - App architecture
4. [`docs/architecture/COMPUTATION_PIPELINE_DOCUMENTATION.md`](docs/architecture/COMPUTATION_PIPELINE_DOCUMENTATION.md) - Computation system

**Code Locations**:
- React components: `app/components/`
- Backend API: `python-scripts/circuit_api.py`
- Computation engine: `app/components/circuit-simulator/utils/`
- Database services: `lib/`

### For DevOps

**Deployment Files**:
- Docker: `docs/deployment/README-Docker.md`
- Migrations: `supabase/migrations/`
- Setup: `docs/setup/SETUP_INSTRUCTIONS.md`
- Backend start: `start_backend.sh`

### For ML Engineers

**ML Code**:
- Research: `ml_ideation/`
- Integration: `docs/ml/ML_WEB_INTEGRATION_GUIDE.md`
- Training: `docs/ml/QUICK_START_MLX.md`
- API endpoint: `python-scripts/circuit_api.py` (ML prediction integrated)

---

## 📊 Statistics

**Before Cleanup**:
- Total source files: ~287
- Documentation files: 52 markdown files (scattered)
- Root directory: Cluttered with 20+ files

**After Cleanup & Organization**:
- Source files: ~230 (19% reduction)
- Documentation: Organized into 6 categories
- Root directory: Clean with essential files only
- All docs indexed in `docs/INDEX.md`

**Impact**:
- ✅ Cleaner project structure
- ✅ Easier to navigate
- ✅ Better documentation discoverability
- ✅ Faster builds (fewer files)
- ✅ Maintained all functionality

---

## 🔄 Migration Notes

If you need to find a file that was moved:

**Documentation Moved**:
- Root-level `.md` files → `docs/` subdirectories
- Check `docs/INDEX.md` for new locations

**Tools Moved**:
- Database test scripts → `tools/database/`
- Reference code → `reference/`

**Schema Docs Moved**:
- `database-schemas-updated/` → `docs/setup/database-schemas/`

**Files Removed**:
- Check backup at `/tmp/deleted_files_backup_20251007_164520`
- See cleanup report in git commit message

---

**Maintained By**: Development Team
**Next Review**: As needed when adding major features
