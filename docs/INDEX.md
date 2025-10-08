# Documentation Index

Welcome to the SpideyPlot v3.0 documentation. This index will help you find the information you need.

## 📚 Quick Start

- **Main README**: [`../README.md`](../README.md) - Project overview and getting started
- **Development Guide**: [`../CLAUDE.md`](../CLAUDE.md) - Development guidelines and architecture decisions
- **Setup Instructions**: [`setup/SETUP_INSTRUCTIONS.md`](setup/SETUP_INSTRUCTIONS.md) - Installation and configuration

---

## 🏗️ Architecture Documentation

Understanding the system design and structure:

- **[App Structure](architecture/APP_STRUCTURE.md)** - Overview of the 3-tab interface and component hierarchy
- **[Routing Architecture](architecture/ROUTING_ARCHITECTURE.md)** - Next.js 15 routing with pagination system
- **[Computation Pipeline](architecture/COMPUTATION_PIPELINE_DOCUMENTATION.md)** - 3-tier computation architecture (Web Workers → WebGPU → Optimized)
- **[Optimization Mathematics](architecture/OPTIMIZATION_MATHEMATICS.md)** - Mathematical foundations of the optimization system

---

## 🔧 Setup & Configuration

Getting the project running:

- **[Setup Instructions](setup/SETUP_INSTRUCTIONS.md)** - Complete installation guide
- **[Supabase Implementation](setup/SUPABASE_IMPLEMENTATION_PLAN.md)** - Database setup and authentication
- **[UI Settings Schema](setup/UI_SETTINGS_DATABASE_SCHEMA.md)** - Settings persistence system
- **[Database Schemas](setup/database-schemas/)** - SQL table definitions and migrations

---

## 🤖 Machine Learning

ML features and training:

- **[ML Web Integration Guide](ml/ML_WEB_INTEGRATION_GUIDE.md)** - Integrating ML models into the web app
- **[MLX Quick Start](ml/QUICK_START_MLX.md)** - Apple Silicon MLX training setup
- **[Geometric Parameter Completion](ml/GEOMETRIC_PARAMETER_COMPLETION.md)** - ML parameter prediction system
- **[ML Implementation Plan](ml/ML_IMPLEMENTATION_PLAN.md)** - ML architecture and strategy
- **[ML Research Directory](../ml_ideation/)** - Experimental ML code and comparisons

---

## 🚀 Deployment

Production deployment guides:

- **[Docker Deployment](deployment/README-Docker.md)** - Containerization and Docker setup
- **[GitHub Registry](deployment/deploy-github-registry.md)** - Publishing to GitHub Container Registry
- **[Unraid Deployment](deployment/deploy-to-unraid.md)** - Deploying to Unraid server

---

## 📖 Reference

Technical references and historical context:

- **[Current Status](reference/CURRENT_STATUS.md)** - System status snapshot (Aug 2025)
- **[Performance Analysis](reference/PERFORMANCE_ANALYSIS.md)** - Performance testing and optimization
- **[Performance Fixes](reference/performance_fixes.md)** - Historical performance improvements
- **[Python Scripts Reference](reference/python_scripts.md)** - Backend Python modules documentation
- **[Test Enhanced Functionality](reference/test_enhanced_functionality.md)** - Testing documentation

---

## 🗃️ Archive

Historical documentation (for reference only):

- **[CLAUDE Inspiration](archive/CLAUDE_inspiration.md)** - Original development philosophy
- **[Design Revisions](archive/designrevisions.md)** - UI/UX evolution history
- **[Original Specs](archive/specs.md)** - Initial project specifications
- **[Planning Documents](archive/planning/)** - Historical planning and design docs

---

## 🛠️ Tools & Utilities

Development and database tools:

- **Database Tools**: `../tools/database/` - Database testing and migration scripts
- **Analysis Scripts**: `../scripts/` - Circuit analysis and visualization generators
- **Python Backend**: `../python-scripts/` - Flask API and computation engine
- **ML Pipeline**: `../ml_ideation/` - ML training and prediction code

---

## 📂 Directory Structure

```
nei-viz-project/
├── docs/                       # All documentation
│   ├── INDEX.md               # This file
│   ├── architecture/          # System architecture
│   ├── setup/                 # Setup guides
│   ├── ml/                    # ML documentation
│   ├── deployment/            # Deployment guides
│   ├── reference/             # Technical references
│   └── archive/               # Historical docs
├── python-scripts/            # Backend Python code
├── ml_ideation/               # ML research code
├── scripts/                   # Analysis utilities
├── tools/                     # Development tools
│   ├── database/             # DB testing scripts
│   └── testing/              # Test utilities
├── reference/                 # Reference implementations
├── data/                      # Data files
│   ├── measurement_presets/  # EIS measurement configs
│   └── npz/                  # Compressed datasets
├── supabase/                  # Database migrations
└── app/                       # Next.js application
    └── components/           # React components
```

---

## 🔍 Finding What You Need

**For Developers:**
- Start with [`../CLAUDE.md`](../CLAUDE.md) - Essential development guidelines
- Architecture docs for understanding the system
- Reference docs for specific implementation details

**For DevOps:**
- Setup guides for installation
- Deployment docs for production
- Database schemas for infrastructure

**For ML Engineers:**
- ML documentation directory
- `ml_ideation/` for research code
- Backend integration guides

**For Users:**
- Main README for overview
- Setup instructions for getting started

---

## 📝 Documentation Standards

When contributing documentation:

1. **Current docs** go in appropriate subdirectories (architecture, setup, ml, deployment, reference)
2. **Historical docs** go in `archive/`
3. **Update this index** when adding new documentation
4. **Use relative links** for cross-references
5. **Include dates** when documenting status or snapshots

---

**Last Updated**: October 2025
**Project Version**: SpideyPlot v3.0
