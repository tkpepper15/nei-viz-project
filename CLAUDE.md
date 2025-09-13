# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚨 DEVELOPMENT PARTNERSHIP

We're building production-quality code together. Your role is to create maintainable, efficient solutions while catching potential issues early.

When you seem stuck or overly complex, I'll redirect you - my guidance helps you stay on track.

### CRITICAL WORKFLOW - ALWAYS FOLLOW THIS!

#### Research → Plan → Implement
**NEVER JUMP STRAIGHT TO CODING!** Always follow this sequence:
1. **Research**: Explore the codebase, understand existing patterns
2. **Plan**: Create a detailed implementation plan and verify it with me  
3. **Implement**: Execute the plan with validation checkpoints

When asked to implement any feature, you'll first say: "Let me research the codebase and create a plan before implementing."

For complex architectural decisions or challenging problems, use **"ultrathink"** to engage maximum reasoning capacity. Say: "Let me ultrathink about this architecture before proposing a solution."

#### USE MULTIPLE AGENTS!
*Leverage subagents aggressively* for better results:

* Spawn agents to explore different parts of the codebase in parallel
* Use one agent to write tests while another implements features
* Delegate research tasks: "I'll have an agent investigate the Web Worker architecture while I analyze the React component structure"
* For complex refactors: One agent identifies changes, another implements them

Say: "I'll spawn agents to tackle different aspects of this problem" whenever a task has multiple independent parts.

#### Reality Checkpoints
**Stop and validate** at these moments:
- After implementing a complete feature
- Before starting a new major component  
- When something feels wrong
- Before declaring "done"
- **WHEN LINTING/TYPE CHECKS FAIL** ❌

Run: `npm run lint && npm run build` to validate your changes.

#### Working Memory Management
When context gets long:
- Re-read this CLAUDE.md file
- Use the TodoWrite tool to track progress
- Document current state before major changes

### Quality Standards for This Project

#### FORBIDDEN - NEVER DO THESE:
- **NO any types** - use proper TypeScript types!
- **NO setTimeout() for synchronization** - use proper async/await patterns!
- **NO** keeping old and new code together
- **NO** migration functions or compatibility layers
- **NO** versioned function names (computeV2, handleNew)
- **NO** TODOs in final code
- **NO** mutations of readonly data structures

#### Required Standards:
- **Delete** old code when replacing it
- **Meaningful names**: `circuitParameters` not `params`
- **Early returns** to reduce nesting
- **Proper TypeScript types**: interfaces and type guards
- **Error boundaries** for React components
- **Proper async/await** for Web Worker communication
- **Immutable state updates** with proper React patterns

## Common Development Commands

### Development Server
```bash
npm run dev         # Start development server with increased memory (8GB)
npm run build       # Build the production application  
npm run start       # Start the production server
npm run lint        # Run ESLint checks
```

### Memory Configuration
The project uses `NODE_OPTIONS='--max-old-space-size=8192'` for development due to large computational datasets and visualization requirements.

## Project Architecture

### Core Application Structure
This is **SpideyPlot v3.0** - an ultra-high-performance electrochemical impedance spectroscopy (EIS) simulation platform for retinal pigment epithelium (RPE) research. The application features a revolutionary 3-tier computation architecture with advanced optimization capabilities scaling to millions of parameter configurations.

### Technology Stack
- **Framework**: Next.js 15 with App Router and TypeScript
- **Styling**: TailwindCSS with professional dark theme
- **Visualization**: Custom React components with advanced spider plot mathematics
- **Math Rendering**: KaTeX for LaTeX equations
- **Computation**: 3-tier pipeline (Web Workers → WebGPU → Optimized algorithms)
- **Backend**: Python Flask API serving NPZ datasets
- **Database**: Supabase PostgreSQL for user profiles and configurations
- **UI Components**: Material-UI integration with custom design system

### Key Architectural Components

#### 1. Main Application Orchestrator (`app/components/CircuitSimulator.tsx`)
- Central component managing the complete application architecture
- Integrates 3-tier computation pipeline: `useWorkerManager` → `useHybridComputeManager` → `useOptimizedComputeManager`
- Manages user authentication, settings, and profile system
- Implements sophisticated Randles equivalent circuit model for RPE impedance analysis
- Supports massive parameter space exploration (up to 25^5 = 9.7M+ parameter combinations)
- Coordinates between visualization tabs: `VisualizerTab`, `MathDetailsTab`, `NPZManager`

#### 2. Revolutionary 3-Tier Computation Pipeline

**Tier 1: Web Workers Foundation** (`utils/workerManager.ts`)
- Multi-core parallel processing across all available CPU cores
- Grid generation, impedance calculations, and resnorm analysis
- Streaming computation for datasets >100K parameter combinations
- Dynamic chunk size optimization based on hardware capabilities

**Tier 2: WebGPU Hybrid Layer** (`utils/hybridComputeManager.ts`)  
- Hardware GPU acceleration with automatic CPU fallback
- WebGPU compute shaders for massive parallel processing
- Intelligent hardware detection and performance optimization
- Memory management for GPU-accelerated computations

**Tier 3: Optimized Pipeline** (`utils/optimizedComputeManager.ts`)
- Advanced optimization algorithms for massive parameter spaces
- Threshold-based activation (automatic for >10K parameter combinations)
- Integration layer providing backward compatibility
- Intelligent fallback chain: Optimized → Hybrid → Web Workers

#### 3. Modern 3-Tab Interface Architecture
- **Visualizer Tab** (`VisualizerTab.tsx`): Advanced spider plot visualization with parameter space exploration
- **Math Details Tab** (`MathDetailsTab.tsx`): Comprehensive mathematical documentation with LaTeX rendering
- **NPZ Data Manager** (`NPZManager.tsx`): Compressed data management, import/export, and dataset library

#### 4. Advanced Visualization System
- **Spider Plots 3D** (`visualizations/SpiderPlot3D.tsx`): Primary 3D spider plot visualization engine
- **Nyquist Plots** (`visualizations/NyquistPlot.tsx`): Complex impedance plane visualization
- **Circuit Diagrams** (`visualizations/CircuitDiagram.tsx`): Interactive equivalent circuit representation
- **Resnorm Grouping**: Dynamic percentile-based categorization with intelligent color coding
- **Performance Optimization**: Adaptive rendering limits and memory management for massive datasets

#### 5. Backend Integration & Data Management
- **Python Flask API** (`circuit_api.py`): Ultra-fast NPZ dataset serving on port 5001
- **NPZ Data Pipeline** (`npz_loader.py`): Compressed data storage with 60-80% memory reduction
- **User Authentication**: Supabase-powered secure user profiles and session management
- **Settings System** (`settings/SettingsModal.tsx`): Advanced optimization controls and GPU settings
- **Profile Management** (`controls/SavedProfiles.tsx`): Persistent computation configurations

#### 6. Performance Features
- **Load Indicators**: Visual feedback for computational complexity (Lo/Med/Hi load)
- **Memory Optimization**: Intelligent sampling for large datasets
- **Adaptive Limits**: Dynamic performance optimization based on dataset size
- **Progress Tracking**: Real-time computation progress with cancellation support

### Mathematical Model Implementation

#### Equivalent Circuit
The application implements a modified Randles circuit model:
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

#### Resnorm Calculation
Advanced residual norm calculation using **Mean Absolute Error (MAE)** methodology:
- **MAE Cost Function**: (1/n) * sum(|Z_test - Z_reference|) following battery EIS research
- Frequency weighting for low-frequency emphasis (optional)
- Component-specific weighting (resistive vs capacitive)
- Magnitude normalization for scale independence
- Range amplification based on frequency span (optional)
- **Default**: Uses MAE method for consistency with published EIS parameter extraction approaches

## Current Architecture Status (Updated 2025-09)

### ✅ **COMPRESSED PIPELINE IS FUNCTIONAL**
The 3-tier computation architecture is working correctly:
1. **Web Workers** → **WebGPU Hybrid** → **Optimized Pipeline** 
2. Development server starts successfully with all compilation errors resolved
3. All core functionality validated through comprehensive codebase audit

### 🧹 **Codebase Cleanup Recommendations**

**SAFE TO REMOVE (~39 files, 28% reduction):**

#### Unused React Components (12 files)
```
app/components/DataTableTab.tsx                    # Replaced by NPZManager
app/components/ParamSlider.tsx                     # No usage found
app/components/PerformanceDashboard.tsx            # No imports
app/components/demos/WebGLSpiderDemo.tsx           # Demo only
app/components/examples/ResnormMethodDemo.tsx      # Demo only  
app/components/auth/LoginPage.tsx                  # Standalone unused
app/components/auth/UserSelector.tsx               # No usage
app/components/controls/EditProfileModal.tsx       # No usage
app/components/controls/ExportModal.tsx            # No imports  
app/components/controls/ProfileCard.tsx            # No usage
app/components/controls/ResizableSplitPane.tsx     # No usage
app/components/controls/SaveProfileModal.tsx       # No usage
```

#### Deprecated Utility Files (13 files)  
```
utils/configSerializer.ts               # Unused serialization system
utils/frequencySerializer.ts            # Unused serialization system  
utils/serializedComputationManager.ts   # Experimental feature not integrated
utils/serializedComputeIntegration.ts   # No usage
utils/optimizedComputePipeline.ts       # Complex unused pipeline
utils/spectralFingerprinting.ts         # Part of unused pipeline  
utils/streamingParameterIterator.ts     # Over-engineered approach
utils/topKHeap.ts                       # Unused optimization
utils/orchestratorWorkerManager.ts      # No usage
utils/sharedWorkerStrategy.ts           # Alternative approach unused
utils/tileRenderer.ts                   # No usage
utils/smartGridFiltering.ts             # No usage
utils/webgpu-compute.wgsl               # Unused shader
```

#### Obsolete Documentation/Config (14 files)
```
EASY_SERIALIZATION_INTEGRATION.md       # Superseded approach
SERIALIZATION_REFERENCE.md              # Old documentation  
SETUP_NPZ_INTEGRATION.md                # Outdated setup
config_code.py                          # 17 lines, incomplete
render_jobs.db                          # SQLite cache
commands.txt                            # Command history
[Multiple outdated *.md files]          # Various superseded approaches
```

### 🏗️ **Current Production Architecture**

#### File Organization Patterns

**Essential Components Structure:**
- **Main Orchestrator**: `CircuitSimulator.tsx` (CORE)
- **Primary Tabs**: `VisualizerTab.tsx`, `MathDetailsTab.tsx`, `NPZManager.tsx`  
- **Authentication**: `auth/AuthProvider.tsx`, `auth/AuthModal.tsx` (ESSENTIAL)
- **Settings**: `settings/SettingsModal.tsx`, `settings/SettingsButton.tsx` (CORE)
- **Controls**: Active UI controls in `circuit-simulator/controls/`
- **Visualizations**: `SpiderPlot3D.tsx`, `NyquistPlot.tsx`, `CircuitDiagram.tsx`
- **Utilities**: Core math and computation utilities (12 files after consolidation)

**Backend Infrastructure:**
- **Python API**: `circuit_api.py`, `npz_loader.py`, `circuit_computation.py` (ESSENTIAL)
- **Serialization**: `config_serializer.py`, `frequency_serializer.py` (ESSENTIAL)
- **Database**: SQL schema files for Supabase integration (ESSENTIAL)

#### State Management
- **Local State**: React hooks for component-specific state
- **Shared State**: Props drilling for cross-component communication
- **Persistence**: localStorage for saved profiles and user preferences
- **Worker Communication**: Message passing for parallel computation

### Development Guidelines

#### Working with Circuit Parameters
- All capacitance values are stored in Farads (not microfarads)
- Frequency ranges use [min, max] array format
- Parameter validation occurs at multiple levels (UI, computation, worker)
- Grid sizes are capped at 20 points per dimension for performance

#### Performance Considerations
- Computation complexity scales as O(n^5) where n is grid size
- Use Web Workers for any computation involving >1000 parameter combinations
- Memory usage estimation: ~500 bytes per model + spectrum data
- Adaptive rendering limits prevent UI blocking on large datasets

#### Web Worker Integration
- Worker scripts are pre-built JavaScript files in `/public/`
- Communication uses structured message passing with type safety
- Progress reporting is mandatory for long-running computations
- Always implement cancellation tokens for user experience

#### Error Handling
- Comprehensive error boundaries for computation failures
- Worker error propagation with meaningful messages
- User-friendly error notifications with recovery suggestions
- Graceful degradation for unsupported browser features

### Testing Approach
The project does not include explicit test files or frameworks. Testing should focus on:
- Circuit model mathematical accuracy
- Web Worker computation correctness
- UI responsiveness under computational load
- Memory usage optimization
- Cross-browser compatibility for Web Workers

### Browser Compatibility
- Requires modern browsers with Web Worker support
- Hardware concurrency detection for optimal performance
- Memory API usage for performance monitoring (Chrome/Edge)
- File API for data export functionality

## Problem-Solving Protocol

When you're stuck or confused:
1. **Stop** - Don't spiral into complex solutions
2. **Delegate** - Consider spawning agents for parallel investigation
3. **Ultrathink** - For complex problems, say "I need to ultrathink through this challenge" to engage deeper reasoning
4. **Step back** - Re-read the requirements
5. **Simplify** - The simple solution is usually correct
6. **Ask** - "I see two approaches: [A] vs [B]. Which do you prefer?"

My insights on better approaches are valued - please ask for them!

## Performance & Security Guidelines

### **Measure First**:
- No premature optimization
- Test Web Worker performance with actual datasets
- Use browser dev tools for real bottlenecks
- Monitor memory usage during large computations

### **Security Always**:
- Validate all user inputs (especially circuit parameters)
- Sanitize data before Web Worker communication
- Use proper TypeScript types to prevent runtime errors
- Handle computation errors gracefully

## Communication Protocol

### Progress Updates:
```
✓ Implemented spider plot rendering (all tests passing)
✓ Added Web Worker computation  
✗ Found issue with memory overflow on large grids - investigating
```

### Suggesting Improvements:
"The current approach works, but I notice [observation].
Would you like me to [specific improvement]?"

## Implementation Checklist

### Our code is complete when:
- ✅ All TypeScript compilation passes with zero errors
- ✅ npm run lint passes with zero issues
- ✅ npm run build succeeds
- ✅ Feature works end-to-end
- ✅ Old code is deleted
- ✅ Proper error handling implemented
- ✅ Web Workers properly terminate on cleanup

### Testing Strategy for This Project
- Complex mathematical logic → Write tests first
- UI components → Write tests after implementation
- Web Worker communication → Add integration tests
- Performance critical paths → Add benchmarks
- Skip tests for simple utility functions

## Working Together

- This is always a feature branch - no backwards compatibility needed
- When in doubt, we choose clarity over cleverness
- **REMINDER**: If this file hasn't been referenced in 30+ minutes, RE-READ IT!

Avoid complex abstractions or "clever" code. The simple, obvious solution is probably better, and my guidance helps you stay focused on what matters.