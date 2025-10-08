# SpideyPlot - Electrochemical Impedance Spectroscopy Simulation

SpideyPlot is an advanced web-based tool for electrochemical impedance spectroscopy (EIS) simulation and visualization, specifically designed for retinal pigment epithelium (RPE) research. The application provides comprehensive circuit modeling, parameter optimization, and interactive visualization capabilities.

## Core Technology Stack

- **Framework**: Next.js 15 with App Router and TypeScript
- **Styling**: TailwindCSS with custom dark theme
- **Visualization**: Custom React components with D3.js mathematics, Plotly.js, and Recharts
- **Math Rendering**: KaTeX for LaTeX equations
- **Parallel Computation**: Web Workers for multi-core processing
- **UI Components**: Material-UI (@mui/material) and Heroicons
- **Database**: Supabase for profile storage and data persistence

## Mathematical Model

The application implements a modified Randles equivalent circuit model:

```
       Rs (Shunt Resistance)
   ────[Rs]────┬──────────┬──────
               │          │
           [Ra]│      [Rb]│
               │          │
           [Ca]│      [Cb]│
               │          │
               └──────────┘
```

Complex impedance calculation: `Z(ω) = Rs + Ra/(1+jωRaCa) + Rb/(1+jωRbCb)`

## Project Structure

### Core Application Files

#### `/app` - Next.js Application Directory
- `layout.tsx` - Root application layout with theme and provider setup
- `page.tsx` - Main application entry point
- `globals.css` - Global styles and TailwindCSS imports

#### `/app/components` - React Components
- `CircuitSimulator.tsx` - Main orchestrator component managing entire application state
- `TabSelector.tsx` - Main navigation component for switching between application views

#### `/app/components/circuit-simulator` - Circuit Simulation Engine
- `types/` - TypeScript definitions
  - `index.ts` - Core data structures (ModelSnapshot, ResnormGroup, BackendMeshPoint)
  - `parameters.ts` - Circuit parameter definitions (CircuitParameters interface)
  - `gpuSettings.ts` - WebGPU acceleration settings
  - `savedProfiles.ts` - User profile persistence types
- `hooks/` - React Hooks
  - `useComputationState.ts` - Main state management hook for computation results and progress
  - `useSerializedComputation.ts` - Hook for compressed serialized computation workflow
- `utils/` - Core Computation Utilities
  - `workerManager.ts` - Web Worker orchestration for parallel computation
  - `hybridComputeManager.ts` - GPU/CPU hybrid computation pipeline
  - `webgpuManager.ts` - WebGPU acceleration for large-scale computations
  - `serializedComputationManager.ts` - Memory-efficient serialized storage system (67x compression)
  - `resnorm.ts` - Residual norm calculation with multiple methods (MAE, MSE, RMSE)
  - `configSerializer.ts` - Parameter configuration compression and decompression
  - `frequencySerializer.ts` - Frequency range serialization for efficient storage
- `controls/` - UI Control Components
  - `ParameterControls.tsx` - Circuit parameter input controls with validation
  - `ProfileCard.tsx` - User profile management and switching interface
  - `SavedProfiles.tsx` - Profile persistence and organization system
  - `OptimizationControls.tsx` - Advanced computation optimization settings
  - `StaticRenderControls.tsx` - Visualization rendering configuration
- `visualizations/` - Data Visualization Components
  - `SpiderPlot3D.tsx` - Primary 3D spider plot visualization with interactive parameter exploration
  - `TiledSpiderPlot.tsx` - Grid-based visualization for large datasets
- `insights/` - Analysis and Insights
  - `ResnormDisplay.tsx` - Residual norm analysis with percentile grouping
- `notifications/` - User Feedback
  - `ComputationNotification.tsx` - Real-time computation progress and results
- `config/` - Configuration Management
  - Circuit parameter presets and default configurations
- `examples/` - Example Implementations
  - `SerializedSpiderPlotDemo.tsx` - Demonstration of serialized computation workflow
- `test/` - Testing Utilities
  - Component testing helpers and mock data generation
- `npz-manager/` - NPZ File Management
  - `NPZManager.tsx` - Main NPZ dataset management interface
  - `NPZExportPanel.tsx` - Export computation results to NPZ format
  - `NPZImportPanel.tsx` - Import existing NPZ datasets
  - `NPZLibraryManager.tsx` - Dataset library organization and management

#### `/app/components/settings` - Application Settings
- `SettingsModal.tsx` - User preferences and application configuration
- `SettingsModal.module.css` - Settings modal styling

#### `/app/components/npz` - NPZ Data Management
- `NPZDatasetManager.tsx` - External NPZ dataset integration and management

#### `/app/hooks` - Global React Hooks
- `useUserSettings.ts` - User preference persistence and synchronization
- `useNPZData.ts` - NPZ dataset loading and caching

#### `/lib` - External Service Integration
- `supabase.ts` - Supabase client configuration and authentication
- `circuitConfigService.ts` - Circuit configuration persistence service
- `profilesService.ts` - User profile storage and retrieval service

#### `/public` - Static Assets
- `grid-worker.js` - Web Worker for grid computation
- `enhanced-tile-worker.js` - Optimized tile-based computation worker
- `tile-worker.js` - Basic tile computation worker
- `webgpu-compute.wgsl` - WebGPU compute shader for acceleration
- `spiderweb.png` - Application icon and branding assets
- `screenshot.png` - Application screenshot for documentation

### Configuration Files

- `next.config.ts` - Next.js configuration with performance optimizations
- `tailwind.config.ts` - TailwindCSS configuration with custom theme
- `tsconfig.json` - TypeScript compiler configuration
- `package.json` - Node.js dependencies and scripts
- `vercel.json` - Vercel deployment configuration
- `webgpu.d.ts` - WebGPU TypeScript definitions
- `next-env.d.ts` - Next.js TypeScript environment definitions
- `CLAUDE.md` - Development guidelines and project instructions for AI assistance

### Documentation

All project documentation is organized in `/docs` with the following structure:

- **📖 [Documentation Index](docs/INDEX.md)** - Complete guide to all documentation
- **🏗️ [Architecture](docs/architecture/)** - System design and structure
- **🔧 [Setup](docs/setup/)** - Installation and configuration guides
- **🤖 [Machine Learning](docs/ml/)** - ML features and training
- **🚀 [Deployment](docs/deployment/)** - Production deployment guides
- **📚 [Reference](docs/reference/)** - Technical references and status
- **🗃️ [Archive](docs/archive/)** - Historical documentation

See [`docs/INDEX.md`](docs/INDEX.md) for a complete documentation map.

#### `/data` - Sample Data and Testing
- `sample-outputs/` - Example computation results
  - `detailed_spectrum_data_grid_5.csv` - 5x5 grid computation results
  - `user_config_grid_9_detailed.csv` - 9x9 grid detailed analysis
  - `test_1.json`, `test_1_small.json` - JSON test datasets
- `demo_grid_5/`, `demo_grid_10/`, `demo_grid_15/` - Demonstration datasets for different grid sizes
- `test_1_based/` - Unit test data and validation sets
- `measurement_presets/` - Predefined measurement configurations
- `query_demo/` - Database query examples and test cases

#### `/python-scripts` - Backend Processing Scripts
- `circuit_computation.py` - Core circuit simulation engine
- `circuit_api.py` - Flask API for Python integration
- `config_serializer.py` - Python implementation of configuration serialization
- `frequency_serializer.py` - Frequency range serialization utilities
- `npz_loader.py` - NPZ file loading and processing
- `npz_supabase_sync.py` - Database synchronization for NPZ datasets
- `measurement_config.py` - Measurement parameter configuration
- `serialization_demo.py` - Demonstration of serialization capabilities
- `lightweight_storage.py` - Optimized storage implementations
- Testing and debugging scripts: `debug_comparison.py`, `test_full_1_based_system.py`

#### `/sql-scripts` - Database Schema and Setup
- `create_missing_tables.sql` - Initial database table creation
- `supabase_npz_table.sql` - NPZ dataset table schema
- `create-user-profiles-table.sql` - User profile storage schema
- `ui_settings_schema_changes.sql` - Settings persistence schema
- `npz_integration_final.sql` - Complete NPZ integration schema
- `auto_register_npz_datasets.sql` - Automated dataset registration
- Setup and migration scripts for database initialization

#### `/external-modules` - External Dependencies
- `wasm-impedance/` - WebAssembly module for high-performance impedance calculations
  - `Cargo.toml` - Rust dependencies
  - `build.sh` - Build script for WASM compilation
  - `src/` - Rust source code for impedance computation

## Key Features

### Computation Engine
- **Parallel Processing**: Multi-core Web Worker computation for parameter space exploration
- **Memory Optimization**: SerializedComputationManager provides 67x memory reduction
- **GPU Acceleration**: WebGPU support for large-scale computations
- **Streaming Results**: Real-time computation progress and result streaming

### Visualization
- **3D Spider Plots**: Interactive multi-dimensional parameter visualization
- **Resnorm Analysis**: Dynamic percentile-based categorization (25%, 50%, 75%, 90%)
- **Reference Models**: Ground truth parameter comparison and overlay
- **Performance Scaling**: Adaptive rendering limits for datasets up to 1M+ models

### Data Management
- **Profile System**: Save and restore parameter configurations
- **NPZ Integration**: Import/export NumPy ZIP format datasets
- **Supabase Storage**: Cloud-based profile and data persistence
- **Serialized Storage**: Ultra-compressed in-memory result storage

### Mathematical Accuracy
- **Resnorm Methods**: Mean Absolute Error (MAE) following battery EIS research standards
- **Frequency Weighting**: Low-frequency emphasis for biological systems
- **Parameter Validation**: Automatic range capping and scientific notation support
- **Spectrum Generation**: Real-time impedance spectrum calculation

## Development Commands

```bash
npm run dev         # Start development server (8GB memory allocated)
npm run build       # Build production application
npm run start       # Start production server
npm run lint        # Run ESLint checks
```

## Performance Characteristics

- **Computation Complexity**: O(n^5) scaling where n is grid size per parameter
- **Memory Usage**: ~60 bytes per model (serialized) vs ~4KB per model (traditional)
- **Maximum Scale**: Supports up to 9.7M+ parameter combinations (25^5 grid)
- **Browser Requirements**: Modern browsers with Web Worker and optional WebGPU support

## Architecture Highlights

### Serialized Storage System
The application uses an innovative serialized storage approach that:
- Stores only configuration IDs and resnorm values (60 bytes per result)
- Generates full ModelSnapshot objects procedurally on-demand
- Provides 67x memory reduction compared to traditional storage
- Enables browser-based analysis of massive parameter spaces

### Hybrid Computation Pipeline
- **Web Workers**: CPU-based parallel computation for standard workloads
- **WebGPU**: GPU acceleration for compute-intensive operations
- **Streaming**: Real-time result processing and visualization updates
- **Cancellation**: User-controlled computation termination

### Component Architecture
- **Modular Design**: Self-contained components with clear responsibilities
- **Type Safety**: Comprehensive TypeScript coverage with strict typing
- **State Management**: Centralized computation state with React hooks
- **Error Boundaries**: Graceful degradation and error recovery

This project represents a comprehensive solution for electrochemical impedance spectroscopy research with modern web technologies, providing researchers with powerful tools for circuit parameter exploration and analysis.