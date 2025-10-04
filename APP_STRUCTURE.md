# SpideyPlot v3.0 - Application Structure & Routing Documentation

**Last Updated**: 2025-10-02
**Version**: 3.0
**Status**: Production-Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Application Architecture](#application-architecture)
3. [Routing & Navigation](#routing--navigation)
4. [Component Hierarchy](#component-hierarchy)
5. [Data Flow](#data-flow)
6. [Tab System](#tab-system)
7. [Filter & Analysis Pipeline](#filter--analysis-pipeline)
8. [Backend Integration](#backend-integration)
9. [State Management](#state-management)
10. [File Organization](#file-organization)

---

## Overview

**SpideyPlot v3.0** is an ultra-high-performance electrochemical impedance spectroscopy (EIS) simulation platform for retinal pigment epithelium (RPE) research. The application uses a single-page architecture with tab-based navigation and no traditional routing.

### Key Features
- **3-Tier Computation Pipeline**: Web Workers → WebGPU → Optimized algorithms
- **Advanced Visualizations**: 3D Spider Plots, Pentagon Quartiles, Nyquist Plots, t-SNE
- **TER/TEC Analysis**: Filtering by transepithelial resistance/capacitance
- **Real-time Filtering**: Dynamic model filtering with multiple criteria
- **User Authentication**: Supabase-powered profiles and saved configurations

---

## Application Architecture

### Framework & Technology Stack
```
Next.js 15 (App Router)
├── TypeScript (strict mode)
├── React 18 (with hooks)
├── TailwindCSS (dark theme)
├── Supabase (authentication + database)
└── Python Flask API (port 5001) for NPZ data serving
```

### Application Entry Points

#### Root Layout (`app/layout.tsx`)
```typescript
<html lang="en" className="dark">
  <body>
    <AuthProvider>
      {children}
    </AuthProvider>
  </body>
</html>
```
- Sets dark theme globally
- Wraps app in authentication context
- Loads KaTeX for math rendering

#### Home Page (`app/page.tsx`)
```typescript
export default function Home() {
  return (
    <main className="min-h-screen bg-background">
      <CircuitSimulator />
    </main>
  );
}
```
- Single entry point to main application
- Mounts `CircuitSimulator` component

#### Additional Routes
- `/sweeper` - Utility page (not main application)
- `/test-computation` - Development testing page

---

## Routing & Navigation

### No Traditional Routing
SpideyPlot uses a **tab-based navigation system** with local state management instead of URL-based routing. This design choice was made because:

1. **Single-Page Application**: All functionality is accessible from one view
2. **State Preservation**: Tab switches preserve computation results
3. **Performance**: No page reloads or route transitions
4. **User Experience**: Instant transitions between views

### Tab State Management

```typescript
// In CircuitSimulator.tsx (line 424-425)
const visualizationTab = uiSettings.activeTab;
const setVisualizationTab = setActiveTab;
```

**Available Tabs**:
- `'visualizer'` - Main visualization playground (default)
- `'math'` - Mathematical model documentation
- `'activity'` - Activity log and performance metrics

### Tab Navigation UI

Located in left sidebar (lines 3106-3154):

```typescript
<button onClick={() => setVisualizationTab('visualizer')}>
  Playground
</button>
<button onClick={() => setVisualizationTab('math')}>
  Model
</button>
<button onClick={() => setVisualizationTab('activity')}>
  Activity Log
</button>
```

### Auto-Navigation Events

The application automatically switches tabs in response to certain events:

1. **After Grid Computation** (line 723):
   ```typescript
   setActiveTab('visualizer'); // Switch to visualizer after computation
   ```

2. **After NPZ Upload** (line 811):
   ```typescript
   setActiveTab('visualizer'); // Show uploaded results
   ```

---

## Component Hierarchy

```
CircuitSimulator (Root)
├── SlimNavbar
│   ├── Status Indicator
│   ├── Settings Button
│   └── UserProfile
│
├── Left Sidebar
│   ├── Tab Navigation (Playground, Model, Activity)
│   ├── SavedProfiles
│   │   ├── Profile List
│   │   └── Multi-Select Actions
│   └── SettingsButton
│
├── Main Content Area (Tab-Based)
│   │
│   ├── [Visualizer Tab]
│   │   ├── VisualizerTab
│   │   │   ├── Left Sidebar Controls
│   │   │   │   ├── Visualization Selector Dropdown
│   │   │   │   ├── Resnorm Filtering
│   │   │   │   ├── TER/TEC Filtering
│   │   │   │   ├── Opacity Controls
│   │   │   │   └── Reference Model Selector
│   │   │   │
│   │   │   └── Visualization Area
│   │   │       ├── SpiderPlot3D
│   │   │       ├── PentagonBoxWhisker
│   │   │       ├── TSNEPlot
│   │   │       ├── NyquistPlot
│   │   │       └── CircuitDiagram
│   │   │
│   │   └── CollapsibleBottomPanel
│   │       ├── Tab Icons (Data, Config, Analysis, etc.)
│   │       └── Tab Content
│   │           ├── ImpedanceComparisonTab
│   │           ├── CircuitConfigurationTab
│   │           ├── AnalysisTab
│   │           ├── QuartileAnalysisTab
│   │           ├── TERTECAnalysisTab (with distribution plot)
│   │           ├── ResnormRangeTab
│   │           └── ExportTab
│   │
│   ├── [Math Tab]
│   │   └── MathDetailsTab
│   │       ├── Circuit Topology
│   │       ├── Impedance Equations (LaTeX)
│   │       ├── Parameter Definitions
│   │       └── Resnorm Calculation
│   │
│   └── [Activity Tab]
│       ├── Performance Report
│       └── Activity Log Messages
│
└── Modal Overlays
    ├── SettingsModal
    ├── AuthModal
    └── Various Action Modals
```

---

## Data Flow

### Computation Pipeline

```
User Input (Parameters)
    ↓
CircuitSimulator State
    ↓
┌─────────────────────────────────────┐
│  3-Tier Computation Pipeline        │
│  ├── useOptimizedComputeManager     │
│  ├── useHybridComputeManager        │
│  └── useWorkerManager               │
└─────────────────────────────────────┘
    ↓
ModelSnapshot[] (Grid Results)
    ↓
Resnorm Grouping & Sorting
    ↓
Filter Pipeline
    ↓
Visible Models (with opacity)
    ↓
Visualizations
```

### Filter Pipeline Architecture

```typescript
// 1. Base Models
allFilteredModels (from grid results)
    ↓
// 2. Resnorm Range Filter (left sidebar)
resnormRangeFilteredModels
    ↓
// 3. TER/TEC Filter (priority: bottom panel > left sidebar)
terTecFilteredModels
    ↓
// 4. Opacity Calculation
visibleModelsWithOpacity
    ↓
// 5. Visualization Rendering
SpiderPlot3D / PentagonBoxWhisker / etc.
```

**Filter Priority Order**:
1. **Bottom Panel TER/TEC Filter** (highest priority) - Applied via "Apply to Visualizations" button
2. **Left Sidebar TER/TEC Filter** - Manual toggle with dropdown
3. **Left Sidebar Resnorm Range** - Slider-based filtering
4. **Opacity Settings** - Applied after all filters

### State Flow Diagram

```
CircuitSimulator (Root State)
│
├─> gridResults: ModelSnapshot[]
├─> resnormGroups: ResnormGroup[]
├─> parameters: CircuitParameters
├─> terTecFilteredModelIds: string[]  ← From bottom panel
│
└─> Props to VisualizerTab
    │
    ├─> Local state: terTecFilterEnabled, terTecTargetValue, etc.
    │
    ├─> Computed: terTecFilteredModels (useMemo)
    │
    └─> Props to CollapsibleBottomPanel
        │
        └─> onTERTECFilterChange callback
            │
            └─> Updates terTecFilteredModelIds in CircuitSimulator
```

---

## Tab System

### Main Tabs (Top-Level Navigation)

| Tab ID | Label | Component | Purpose |
|--------|-------|-----------|---------|
| `visualizer` | Playground | `VisualizerTab` | Main visualization and analysis area |
| `math` | Model | `MathDetailsTab` | Mathematical documentation with LaTeX |
| `activity` | Activity Log | Activity Log UI | Performance metrics and logs |

### Bottom Panel Tabs (Within Visualizer)

| Tab ID | Label | Icon | Component | Purpose |
|--------|-------|------|-----------|---------|
| `impedance` | Data | `TableCellsIcon` | `ImpedanceComparisonTab` | View impedance data tables |
| `configuration` | Config | `CpuChipIcon` | `CircuitConfigurationTab` | Circuit parameter details |
| `analysis` | Analysis | `BeakerIcon` | `AnalysisTab` | Statistical analysis tools |
| `quartile` | Quartile | `ChartPieIcon` | `QuartileAnalysisTab` | Quartile distribution |
| `tertec` | TER/TEC | `BoltIcon` | `TERTECAnalysisTab` | TER/TEC filtering & IQR analysis |
| `resnorm` | Resnorm | `ChartBarIcon` | `ResnormRangeTab` | Resnorm range navigation |
| `export` | Export | `ArrowDownTrayIcon` | `ExportTab` | Data export functionality |

### Tab Integration Points

**File**: `app/components/circuit-simulator/panels/CollapsibleBottomPanel.tsx`

```typescript
const DEFAULT_TABS: BottomPanelTab[] = [
  { id: 'impedance', label: 'Data', icon: TableCellsIcon, component: ImpedanceComparisonTab },
  { id: 'configuration', label: 'Config', icon: CpuChipIcon, component: CircuitConfigurationTab },
  // ... other tabs
  { id: 'tertec', label: 'TER/TEC', icon: BoltIcon, component: TERTECAnalysisTab },
];
```

---

## Filter & Analysis Pipeline

### TER/TEC Analysis System

#### What is TER/TEC?

- **TER (Transepithelial Resistance)**: DC resistance of the circuit
  - Formula: `TER = Rsh × (Ra + Rb) / (Rsh + Ra + Rb)`
  - Units: Ohms (Ω)

- **TEC (Transepithelial Capacitance)**: Series capacitance
  - Formula: `TEC = (Ca × Cb) / (Ca + Cb)`
  - Units: Farads (F), typically displayed in µF, nF, or pF

#### TER/TEC Filtering Flow

```
1. User selects TER or TEC in bottom panel
   ↓
2. Unique values calculated from all models
   ↓
3. User selects target value from dropdown
   ↓
4. Tolerance applied (default ±5%)
   ↓
5. Matching models filtered
   ↓
6. IQR analysis calculated for 5 parameters
   ↓
7. User clicks "Apply to Visualizations"
   ↓
8. filteredModelIds sent to CircuitSimulator
   ↓
9. VisualizerTab receives updated filter
   ↓
10. All visualizations show only filtered models
```

#### TER/TEC Components

**File**: `app/components/circuit-simulator/panels/tabs/TERTECAnalysisTab.tsx`

Features:
- **Filter Type Toggle**: TER vs TEC
- **Value Dropdown**: Populated with unique values from dataset
- **Tolerance Slider**: Adjust matching range (1-20%)
- **IQR Analysis Table**: Shows parameter distribution statistics
- **Distribution Plot**: Canvas-based scatter plot of TER/TEC vs Resnorm
- **Apply Button**: Activates filter across all visualizations

**Calculation Utilities**: `app/components/circuit-simulator/utils/terTecCalculations.ts`

```typescript
export function calculateTER(params: CircuitParameters): number;
export function calculateTEC(params: CircuitParameters): number;
export function formatTER(ter: number): string;
export function formatTEC(tec: number): string;
export function getTERRange(ter: number, tolerance: number): { min: number; max: number };
export function getTECRange(tec: number, tolerance: number): { min: number; max: number };
```

### Quartile Analysis System

**File**: `app/components/circuit-simulator/panels/tabs/QuartileAnalysisTab.tsx`

- Displays parameter distributions as box-whisker plots
- Shows Q1, median, Q3, and outliers
- Visualized on **Pentagon Box-Whisker** plot

**Visualization**: `app/components/circuit-simulator/visualizations/PentagonBoxWhisker.tsx`

- 5-axis pentagon (one per parameter: Rsh, Ra, Rb, Ca, Cb)
- Box-whisker overlays on each axis
- Dynamic grid lines based on `gridSize` prop

---

## Backend Integration

### Python Flask API

**File**: `circuit_api.py` (port 5001)

#### Endpoints

1. **POST `/api/load/<filename>`**
   - Loads NPZ compressed data
   - Returns circuit models with pre-computed impedance spectra
   - Used by NPZ Data Manager

2. **GET `/api/health`**
   - Health check endpoint
   - Returns server status

### Data Format

**NPZ Structure**:
```python
{
  'parameters': np.array([[Rsh, Ra, Rb, Ca, Cb], ...]),
  'impedance_real': np.array([...]),
  'impedance_imag': np.array([...]),
  'frequencies': np.array([...]),
  'resnorm': np.array([...])
}
```

**ModelSnapshot Interface**:
```typescript
interface ModelSnapshot {
  id: string;
  parameters: CircuitParameters;
  data: ImpedancePoint[];
  resnorm: number;
  metadata?: {
    computationTime?: number;
    method?: string;
  };
}
```

---

## State Management

### Global State (CircuitSimulator)

```typescript
// Core computation state
const [gridResults, setGridResults] = useState<ModelSnapshot[]>([]);
const [resnormGroups, setResnormGroups] = useState<ResnormGroup[]>([]);
const [parameters, setParameters] = useState<CircuitParameters>({...});

// UI state
const [activeTab, setActiveTab] = useState<'visualizer' | 'math' | 'activity'>('visualizer');
const [isComputingGrid, setIsComputingGrid] = useState(false);

// Filter state
const [hiddenGroups, setHiddenGroups] = useState<Set<string>>(new Set());
const [opacityLevel, setOpacityLevel] = useState(0.5);
const [terTecFilteredModelIds, setTerTecFilteredModelIds] = useState<string[]>([]);

// User state
const [circuitConfigurations, setCircuitConfigurations] = useState<CircuitConfiguration[]>([]);
const [activeConfigId, setActiveConfigId] = useState<string | null>(null);
```

### Local State (VisualizerTab)

```typescript
// Visualization state
const [selectedVisualization, setSelectedVisualization] = useState<string>('spider3d');

// Left sidebar filter state
const [terTecFilterEnabled, setTerTecFilterEnabled] = useState(false);
const [terTecFilterType, setTerTecFilterType] = useState<'TER' | 'TEC'>('TER');
const [terTecTargetValue, setTerTecTargetValue] = useState<number>(0);
const [terTecTolerance, setTerTecTolerance] = useState<number>(5);

// Resnorm range state
const [currentResnorm, setCurrentResnorm] = useState<number | null>(null);
const [selectedResnormRange, setSelectedResnormRange] = useState<{min: number; max: number} | null>(null);
```

### Props Threading Pattern

```
CircuitSimulator
├─ terTecFilteredModelIds: string[]
└─ setTerTecFilteredModelIds: (ids: string[]) => void
    ↓ (props)
VisualizerTab
├─ terTecFilteredModelIds (received)
├─ onTERTECFilterChange (received)
└─ Passes to: CollapsibleBottomPanel
    ↓ (props)
CollapsibleBottomPanel
├─ terTecFilteredModelIds (received)
├─ onTERTECFilterChange (received)
└─ Passes to: TERTECAnalysisTab
    ↓ (props)
TERTECAnalysisTab
├─ Computes filtered models
├─ User clicks "Apply to Visualizations"
└─ Calls: onTERTECFilterChange(filteredIds)
    ↓ (callback chain)
CircuitSimulator updates terTecFilteredModelIds
    ↓
VisualizerTab re-renders with new filter
    ↓
All visualizations show filtered models
```

---

## File Organization

### Core Application Files

```
app/
├── layout.tsx                          # Root layout with AuthProvider
├── page.tsx                            # Home page (mounts CircuitSimulator)
├── globals.css                         # Global styles
│
├── components/
│   ├── CircuitSimulator.tsx           # 🎯 Main orchestrator (3700+ lines)
│   │
│   ├── auth/
│   │   ├── AuthProvider.tsx           # Supabase authentication context
│   │   ├── AuthModal.tsx              # Login/signup modal
│   │   └── UserProfile.tsx            # User profile dropdown
│   │
│   ├── settings/
│   │   ├── SettingsModal.tsx          # Global settings modal
│   │   └── SettingsButton.tsx         # Settings button component
│   │
│   └── circuit-simulator/
│       ├── index.ts                   # Re-exports
│       ├── VisualizerTab.tsx          # 🎯 Main visualization tab (1800+ lines)
│       ├── MathDetailsTab.tsx         # Mathematical documentation
│       ├── SlimNavbar.tsx             # Top navigation bar
│       │
│       ├── types/
│       │   ├── index.ts               # Core types
│       │   └── parameters.ts          # CircuitParameters interface
│       │
│       ├── utils/
│       │   ├── impedance.ts           # Impedance calculations
│       │   ├── terTecCalculations.ts  # TER/TEC utilities
│       │   ├── workerManager.ts       # Web Worker management
│       │   ├── hybridComputeManager.ts # WebGPU compute
│       │   └── optimizedComputeManager.ts # Optimization layer
│       │
│       ├── visualizations/
│       │   ├── SpiderPlot3D.tsx       # 🎯 Primary 3D spider plot
│       │   ├── PentagonBoxWhisker.tsx # Pentagon quartile plot
│       │   ├── TSNEPlot.tsx           # t-SNE dimensionality reduction
│       │   ├── NyquistPlot.tsx        # Nyquist impedance plot
│       │   └── CircuitDiagram.tsx     # Circuit schematic
│       │
│       ├── panels/
│       │   ├── CollapsibleBottomPanel.tsx  # 🎯 Bottom panel container
│       │   └── tabs/
│       │       ├── ImpedanceComparisonTab.tsx
│       │       ├── CircuitConfigurationTab.tsx
│       │       ├── AnalysisTab.tsx
│       │       ├── QuartileAnalysisTab.tsx
│       │       ├── TERTECAnalysisTab.tsx   # 🎯 TER/TEC analysis
│       │       ├── ResnormRangeTab.tsx
│       │       └── ExportTab.tsx
│       │
│       └── controls/
│           ├── SavedProfiles.tsx      # Profile list sidebar
│           ├── CenterParameterPanel.tsx # Parameter input panel
│           └── StaticRenderControls.tsx # Visualization controls
│
└── hooks/
    ├── useUserSettings.ts             # User settings hook
    └── useEnhancedUserSettings.ts     # Enhanced settings with sync
```

### Backend Files

```
/
├── circuit_api.py                     # Flask API server (port 5001)
├── npz_loader.py                      # NPZ data loading utilities
├── circuit_computation.py             # Circuit impedance calculations
├── config_serializer.py               # Configuration serialization
└── frequency_serializer.py            # Frequency array serialization
```

---

## Design Patterns & Best Practices

### Component Design

1. **Separation of Concerns**
   - Visualization components are pure (no state mutations)
   - State management centralized in CircuitSimulator
   - UI controls separated from data logic

2. **Props Threading**
   - Callbacks passed down for state updates
   - Data flows down, events flow up
   - Clear prop interfaces with TypeScript

3. **Performance Optimization**
   - `useMemo` for expensive calculations
   - `useCallback` for stable function references
   - Web Workers for heavy computation
   - Canvas-based rendering for visualizations

### State Management Principles

1. **Single Source of Truth**
   - Grid results stored once in CircuitSimulator
   - Derived state computed in child components

2. **Filter Composition**
   - Filters applied in sequence
   - Each filter is independently toggleable
   - Priority system for conflicting filters

3. **Persistence**
   - User settings saved to localStorage
   - Circuit configurations saved to Supabase
   - Computation results cached in session

---

## Common Development Tasks

### Adding a New Visualization

1. Create component in `app/components/circuit-simulator/visualizations/`
2. Add to visualization selector in `VisualizerTab.tsx`
3. Pass filtered models as props
4. Use `visibleModelsWithOpacity` for rendering

### Adding a New Bottom Panel Tab

1. Create tab component in `app/components/circuit-simulator/panels/tabs/`
2. Implement `BottomPanelTabProps` interface
3. Add to `DEFAULT_TABS` array in `CollapsibleBottomPanel.tsx`
4. Import appropriate icon from `@heroicons/react`

### Adding a New Filter Type

1. Add state to `VisualizerTab.tsx` or `CircuitSimulator.tsx`
2. Create filter logic with `useMemo`
3. Add UI controls in left sidebar or bottom panel
4. Update `visibleModels` computation to include new filter
5. Thread callback props if needed

### Modifying Computation Pipeline

1. Check tier priority: Optimized → Hybrid → Web Workers
2. Update relevant manager in `utils/`
3. Ensure backward compatibility
4. Test with various dataset sizes

---

## Troubleshooting

### Issue: TEC values not showing in dropdown

**Root Cause**: TEC values are extremely small (10^-7 to 10^-12 F), so rounding to 3 decimals resulted in zeros.

**Solution**: Use significant figures instead of decimal places for TEC:
```typescript
if (filterType === 'TEC') {
  const magnitude = Math.floor(Math.log10(Math.abs(value)));
  const scale = Math.pow(10, magnitude - 2);
  return Math.round(value / scale) * scale;
}
```

### Issue: Filter not hiding non-matching models

**Root Cause**: Pentagon visualization used wrong model array.

**Solution**: Pass `visibleModelsWithOpacity` instead of `modelsWithUpdatedResnorm`.

### Issue: Page "hops" between views

**Root Cause**: This is intentional tab-based navigation, not a bug.

**Understanding**: SpideyPlot uses local state tabs, not URL routing. Transitions are instant and preserve state.

---

## Future Enhancements

### Planned Features

1. **URL Query Parameters**
   - Shareable links with filter state
   - Deep linking to specific configurations

2. **Enhanced Routing**
   - Browser back/forward for tab navigation
   - Bookmarkable visualization states

3. **Multi-Window Support**
   - Detachable visualization panels
   - Side-by-side comparison mode

4. **Real-time Collaboration**
   - Share sessions with other users
   - Live cursor tracking

---

## Version History

### v3.0 (Current)
- ✅ TER/TEC filtering and analysis
- ✅ Quartile analysis with pentagon visualization
- ✅ Distribution plots (TER/TEC vs Resnorm)
- ✅ Improved precision handling for small values
- ✅ 3-tier computation pipeline
- ✅ Comprehensive documentation

### v2.x
- Spider plot 3D visualization
- Web Worker parallelization
- NPZ data compression
- User authentication

### v1.x
- Initial impedance simulation
- Basic circuit model
- Single-threaded computation

---

## Contact & Support

For questions about the application structure or routing:
1. Review this documentation
2. Check `CLAUDE.md` for development guidelines
3. Examine component source code with inline comments
4. Consult TypeScript type definitions

**Key Files to Reference**:
- `app/components/CircuitSimulator.tsx` - Main application logic
- `app/components/circuit-simulator/VisualizerTab.tsx` - Visualization orchestration
- `app/components/circuit-simulator/panels/CollapsibleBottomPanel.tsx` - Bottom panel system

---

**End of Documentation**
