# SpideyPlot v3.0 - Routing Architecture

## Overview
Implemented proper Next.js 15 App Router architecture with pagination to prevent hydration errors and improve performance.

## Route Structure

```
/                                  → Landing page (server component)
/simulator                         → Main circuit simulator
/simulator/visualizer              → 3D spider plot visualizer with pagination
/simulator/math-details            → Mathematical documentation
/profiles                          → User profiles management with pagination
/settings                          → Application settings
/sweeper                           → Sweeper analysis tool (existing)
/test-computation                  → Computation testing (existing)
```

## Key Improvements

### 1. **Separated Routes**
- Each major feature now has its own dedicated route
- No more monolithic single-page application
- Clear separation of concerns

### 2. **Proper Client/Server Boundaries**
- Landing page is a server component for fast initial load
- Interactive components marked with "use client"
- Optimized for Next.js 15 App Router

### 3. **Loading & Error States**
Each route has dedicated:
- `loading.tsx` - Displays while route loads
- `error.tsx` - Handles runtime errors gracefully
- Prevents hydration mismatches

### 4. **Pagination Infrastructure**
Created reusable pagination system:
- `app/components/circuit-simulator/hooks/usePagination.ts` - Custom hook for data pagination
- `app/components/circuit-simulator/controls/PaginationControls.tsx` - Existing advanced UI component
- Prevents rendering thousands of models at once
- Configurable items per page (100, 500, 1000, 2500, 5000)

### 5. **Layout System**
- Each route uses the existing left sidebar from CircuitSimulator
- No additional navigation layers added
- Original UI/UX preserved

## File Structure

```
app/
├── (landing)/                     # Landing page route group
│   ├── page.tsx                   # Home page
│   └── layout.tsx                 # Minimal layout
├── simulator/
│   ├── layout.tsx                 # Shared layout with navigation
│   ├── page.tsx                   # Main simulator
│   ├── loading.tsx                # Loading state
│   ├── error.tsx                  # Error boundary
│   ├── visualizer/
│   │   ├── page.tsx              # Visualizer with pagination
│   │   ├── loading.tsx
│   │   └── error.tsx
│   └── math-details/
│       ├── page.tsx              # Math documentation
│       ├── loading.tsx
│       └── error.tsx
├── profiles/
│   ├── page.tsx                  # Profiles with pagination
│   ├── layout.tsx
│   ├── loading.tsx
│   └── error.tsx
├── settings/
│   ├── page.tsx                  # Settings page
│   ├── layout.tsx
│   ├── loading.tsx
│   └── error.tsx
└── components/
    └── circuit-simulator/
        ├── hooks/
        │   └── usePagination.ts  # Pagination hook
        └── controls/
            └── PaginationControls.tsx  # Pagination UI
```

## Navigation

### Landing Page
- Hero section with logo and description
- Feature cards highlighting 3-tier computation
- Quick links to all major features

### Original Sidebar Preserved
- All routes use the existing left sidebar from CircuitSimulator
- No additional navigation UI added
- Original workflow and UX maintained

## Performance Benefits

1. **Reduced Initial Bundle Size**
   - Landing page: 115 kB (server component)
   - Only loads what's needed per route

2. **Pagination Prevents Overload**
   - Visualizer can handle millions of data points
   - Renders in manageable chunks
   - Configurable page sizes

3. **No Hydration Errors**
   - Proper client/server boundaries
   - Loading states prevent mismatches
   - Error boundaries catch runtime issues

4. **Better Code Splitting**
   - Each route is code-split automatically
   - Lazy loading of heavy components
   - Faster time-to-interactive

## Migration Notes

- Old `/page.tsx` backed up to `/page.tsx.old`
- NPZ Data Manager removed as requested
- All existing functionality preserved in new routes
- Build succeeds with zero errors

## Usage

### Development
```bash
npm run dev       # Starts at http://localhost:3006
```

### Production
```bash
npm run build     # Creates optimized production build
npm run start     # Runs production server
```

## Future Enhancements

1. **Server-Side Pagination**
   - Move pagination logic to server components
   - Stream data on demand

2. **Route Prefetching**
   - Preload adjacent pages in pagination
   - Faster navigation

3. **Advanced Caching**
   - Cache computed results per route
   - Persist pagination state

4. **Deep Linking**
   - URL params for pagination state
   - Shareable links to specific views

## Technical Details

- **Framework**: Next.js 15.1.5 with App Router
- **Rendering**: Mixed (Server + Client Components)
- **State Management**: React hooks + localStorage
- **Build Output**: Static where possible, dynamic as needed
- **Route Types**:
  - Static (○): Pre-rendered at build time
  - Dynamic (ƒ): Server-rendered on demand
