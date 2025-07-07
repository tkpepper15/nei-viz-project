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
This is **SpideyPlot** - an advanced electrochemical impedance spectroscopy (EIS) simulation and visualization tool for retinal pigment epithelium (RPE) research. The application is built with Next.js 15 and features a sophisticated circuit simulation engine with parallel computation capabilities.

### Technology Stack
- **Framework**: Next.js 15 with App Router and TypeScript
- **Styling**: TailwindCSS with custom dark theme
- **Visualization**: Custom React components with D3.js mathematics, Plotly.js, and Recharts
- **Math Rendering**: KaTeX for LaTeX equations
- **Parallel Computation**: Web Workers for multi-core processing
- **UI Components**: Material-UI (@mui/material) and Heroicons

### Key Architectural Components

#### 1. Circuit Simulation Engine (`app/components/CircuitSimulator.tsx`)
- Main orchestrator component managing the entire application state
- Handles parallel computation using Web Workers via `useWorkerManager`
- Implements Randles equivalent circuit model for RPE cell impedance
- Complex impedance calculation: `Z(ω) = Rs + Ra/(1+jωRaCa) + Rb/(1+jωRbCb)`
- Supports parameter space exploration with up to 25^5 (9.7M+) parameter combinations

#### 2. Web Worker System (`app/components/circuit-simulator/utils/workerManager.ts`)
- Manages parallel computation across multiple CPU cores
- Handles grid generation, impedance calculations, and resnorm analysis
- Implements streaming computation for large datasets (>100k points)
- Optimizes chunk sizes based on available hardware and dataset size
- Web worker scripts located in `/public/` directory (grid-worker.js, enhanced-tile-worker.js, tile-worker.js)

#### 3. Tabbed Interface Architecture
- **Playground Tab** (`VisualizerTab.tsx`): Interactive spider plot visualization with parameter exploration
- **Model Tab** (`MathDetailsTab.tsx`): Mathematical documentation with LaTeX rendering
- **Data Table Tab** (`DataTableTab.tsx`): Sortable data analysis interface
- **Activity Log Tab**: Real-time computation logging and status updates
- **Orchestrator Tab** (`OrchestratorTab.tsx`): Advanced visualization management

#### 4. Visualization System
- **Spider Plots** (`visualizations/SpiderPlot.tsx`, `TiledSpiderPlot.tsx`): Multi-dimensional parameter visualization
- **Resnorm Grouping**: Dynamic percentile-based categorization (25%, 50%, 75%, 90%)
- **Reference Model Overlay**: Ground truth parameter comparison
- **Performance Optimization**: Adaptive rendering limits and memory management

#### 5. Parameter Management
- **Circuit Parameters** (`types/parameters.ts`): Rs, Ra, Ca, Rb, Cb with frequency ranges
- **Saved Profiles** (`controls/SavedProfiles.tsx`): Profile persistence using localStorage
- **Parameter Validation**: Automatic range capping and validation
- **Grid Generation**: Logarithmic or linear parameter space sampling

#### 6. Performance Features
- **Load Indicators**: Visual feedback for computational complexity (Lo/Med/Hi load)
- **Memory Optimization**: Intelligent sampling for large datasets
- **Adaptive Limits**: Dynamic performance optimization based on dataset size
- **Progress Tracking**: Real-time computation progress with cancellation support

### Mathematical Model Implementation

#### Equivalent Circuit
The application implements a modified Randles circuit model:
```
       Rs (Series Resistance)
   ────[Rs]────┬──────────┬──────
               │          │
           [Ra]│      [Rb]│
               │          │
           [Ca]│      [Cb]│
               │          │
               └──────────┘
```

#### Resnorm Calculation
Advanced residual norm calculation with:
- Frequency weighting for low-frequency emphasis
- Component-specific weighting (resistive vs capacitive)
- Magnitude normalization for scale independence
- Range amplification based on frequency span

### File Organization Patterns

#### Component Structure
- **Main Components**: Top-level feature components in `circuit-simulator/`
- **Controls**: UI control components in `circuit-simulator/controls/`
- **Utilities**: Mathematical and computational utilities in `circuit-simulator/utils/`
- **Types**: TypeScript definitions organized by domain in `circuit-simulator/types/`
- **Visualizations**: Chart and plot components in `circuit-simulator/visualizations/`

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