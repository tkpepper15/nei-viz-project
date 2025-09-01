# UI Settings Auto-Save Database Schema

## Overview

This document describes the database schema changes required to support the comprehensive UI settings auto-save system implemented for the NEI Viz Project (SpideyPlot).

## 🗄️ Database Tables

### 1. `circuit_configurations` (Primary Table)

**Purpose**: Stores circuit simulation configurations with full UI state persistence.

```sql
CREATE TABLE circuit_configurations (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  
  -- Configuration metadata
  name TEXT NOT NULL,
  description TEXT,
  is_public BOOLEAN DEFAULT FALSE,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Circuit simulation parameters
  grid_size INTEGER NOT NULL CHECK (grid_size BETWEEN 2 AND 25),
  min_freq REAL NOT NULL CHECK (min_freq > 0),
  max_freq REAL NOT NULL CHECK (max_freq > min_freq),
  num_points INTEGER NOT NULL CHECK (num_points BETWEEN 10 AND 1000),
  circuit_parameters JSONB NOT NULL,
  
  -- Computation status and results
  is_computed BOOLEAN DEFAULT FALSE,
  computation_time REAL,
  total_points INTEGER,
  valid_points INTEGER,
  computation_results JSONB,
  
  -- 🆕 UI settings persistence
  ui_settings JSONB, -- ⭐ NEW: Auto-save all UI state
  
  -- Constraints
  CONSTRAINT valid_frequency_range CHECK (max_freq > min_freq),
  CONSTRAINT valid_circuit_params CHECK (
    circuit_parameters ? 'Rsh' AND
    circuit_parameters ? 'Ra' AND  
    circuit_parameters ? 'Ca' AND
    circuit_parameters ? 'Rb' AND
    circuit_parameters ? 'Cb' AND
    circuit_parameters ? 'frequency_range'
  )
);
```

### 2. `user_sessions` (Enhanced)

**Purpose**: Tracks user sessions with active configuration persistence.

```sql
ALTER TABLE user_sessions 
ADD COLUMN current_circuit_config_id UUID REFERENCES circuit_configurations(id) ON DELETE SET NULL;
```

**New Field**:
- `current_circuit_config_id`: Tracks which configuration is active for session persistence across page reloads

### 3. `tagged_models` (Enhanced)

**Purpose**: Links tagged models to the new circuit configurations table.

```sql
ALTER TABLE tagged_models 
ADD COLUMN circuit_config_id UUID REFERENCES circuit_configurations(id) ON DELETE CASCADE;
```

**New Field**:
- `circuit_config_id`: References the new `circuit_configurations` table instead of the old `saved_configurations`

---

## 🔧 UI Settings JSON Structure

The `ui_settings` JSONB column stores a comprehensive UI state object:

```typescript
interface UISettings {
  // Tab and panel states
  activeTab: 'visualizer' | 'math' | 'data' | 'activity';
  splitPaneHeight: number; // Bottom panel height percentage
  
  // Visualization settings
  opacityLevel: number;
  opacityExponent: number;
  logScalar: number;
  visualizationMode: 'color' | 'opacity';
  backgroundColor: 'transparent' | 'white' | 'black';
  showGroundTruth: boolean;
  includeLabels: boolean;
  maxPolygons: number;
  
  // Performance settings
  useSymmetricGrid: boolean;
  adaptiveLimit: boolean;
  maxMemoryUsage: number;
  
  // Reference model settings
  referenceModelVisible: boolean;
  manuallyHidden: boolean;
  
  // Multi-select and interaction states
  isMultiSelectMode: boolean;
  selectedCircuits: string[];
  
  // Window and panel positions (for future modal/window support)
  windowPositions?: {
    [key: string]: {
      x: number;
      y: number;
      width: number;
      height: number;
    };
  };
  
  // Sidebar and toolbox states
  sidebarCollapsed?: boolean;
  toolboxPositions?: {
    [key: string]: { x: number; y: number };
  };
}
```

### Example JSON:

```json
{
  "activeTab": "visualizer",
  "splitPaneHeight": 35,
  "opacityLevel": 0.7,
  "opacityExponent": 5,
  "logScalar": 1.0,
  "visualizationMode": "color",
  "backgroundColor": "white",
  "showGroundTruth": true,
  "includeLabels": true,
  "maxPolygons": 10000,
  "useSymmetricGrid": false,
  "adaptiveLimit": true,
  "maxMemoryUsage": 8192,
  "referenceModelVisible": true,
  "manuallyHidden": false,
  "isMultiSelectMode": false,
  "selectedCircuits": [],
  "windowPositions": {},
  "sidebarCollapsed": false,
  "toolboxPositions": {}
}
```

---

## 📊 Database Indexes

Performance indexes for efficient queries:

```sql
-- Circuit configurations indexes
CREATE INDEX idx_circuit_configurations_user ON circuit_configurations(user_id, created_at DESC);
CREATE INDEX idx_circuit_configurations_public ON circuit_configurations(is_public, created_at DESC) WHERE is_public = TRUE;
CREATE INDEX idx_circuit_configurations_name ON circuit_configurations(user_id, name);
CREATE INDEX idx_circuit_configurations_computed ON circuit_configurations(user_id, is_computed, updated_at DESC);
CREATE INDEX idx_circuit_configurations_updated ON circuit_configurations(user_id, updated_at DESC);

-- Session management indexes
CREATE INDEX idx_user_sessions_config ON user_sessions(user_id, current_circuit_config_id);

-- Tagged models indexes
CREATE INDEX idx_tagged_models_circuit_config ON tagged_models(circuit_config_id);
```

---

## 🔒 Row Level Security (RLS)

All tables have RLS policies ensuring users can only access their own data:

```sql
-- Circuit configurations policies
CREATE POLICY "Users can view their own circuit configurations"
  ON circuit_configurations FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can view public circuit configurations"
  ON circuit_configurations FOR SELECT
  USING (is_public = true);

CREATE POLICY "Users can create their own circuit configurations"
  ON circuit_configurations FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own circuit configurations"
  ON circuit_configurations FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own circuit configurations"
  ON circuit_configurations FOR DELETE
  USING (auth.uid() = user_id);
```

---

## 🛠️ Helper Functions

### Default UI Settings Function:

```sql
CREATE OR REPLACE FUNCTION get_default_ui_settings()
RETURNS JSONB AS $$
BEGIN
  RETURN jsonb_build_object(
    'activeTab', 'visualizer',
    'splitPaneHeight', 35,
    'opacityLevel', 0.7,
    'opacityExponent', 5,
    'logScalar', 1.0,
    'visualizationMode', 'color',
    'backgroundColor', 'white',
    'showGroundTruth', true,
    'includeLabels', true,
    'maxPolygons', 10000,
    'useSymmetricGrid', false,
    'adaptiveLimit', true,
    'maxMemoryUsage', 8192,
    'referenceModelVisible', true,
    'manuallyHidden', false,
    'isMultiSelectMode', false,
    'selectedCircuits', '[]'::jsonb,
    'windowPositions', '{}'::jsonb,
    'sidebarCollapsed', false,
    'toolboxPositions', '{}'::jsonb
  );
END;
$$ LANGUAGE plpgsql;
```

### Session Management View:

```sql
CREATE OR REPLACE VIEW session_with_config AS
SELECT 
  s.id as session_id,
  s.user_id,
  s.session_name,
  s.current_circuit_config_id,
  c.name as config_name,
  c.updated_at as config_last_modified,
  s.last_accessed,
  s.is_active
FROM user_sessions s
LEFT JOIN circuit_configurations c ON s.current_circuit_config_id = c.id;
```

---

## 📝 Migration Files Created

1. **`20250109000001_create_circuit_configurations.sql`**
   - Creates the main `circuit_configurations` table
   - Sets up indexes and RLS policies
   - Includes `ui_settings` JSONB column

2. **`20250109000002_add_ui_settings_and_session_config.sql`**
   - Adds `current_circuit_config_id` to `user_sessions`
   - Adds `circuit_config_id` to `tagged_models` 
   - Creates performance indexes

3. **`20250109000003_verify_ui_settings_schema.sql`**
   - Verifies all required tables and columns exist
   - Creates helper functions
   - Provides example data structures

---

## 🚀 Application Integration

### Service Layer:
- `CircuitConfigService.updateUISettings()` - Lightweight UI settings updates
- Auto-save with 1-second debouncing
- Emergency save on page unload

### React Hooks:
- `useUISettingsManager()` - Comprehensive UI state management
- `useAutoSaveUISettings()` - Debounced auto-save functionality
- Session persistence with `useSessionManagement()`

### API Endpoints:
- `POST /api/save-ui-settings` - Emergency beacon saves

---

## 🎯 Key Benefits

1. **Seamless User Experience** - All UI preferences automatically saved and restored
2. **No Data Loss** - Settings preserved even during unexpected page exits  
3. **Session Persistence** - Exact workspace state maintained across sessions
4. **Performance Optimized** - Debounced saves prevent excessive database writes
5. **Scalable Architecture** - JSONB allows flexible UI state storage

---

## 🔧 Setup Instructions

1. **Run Migrations**:
   ```bash
   ./apply-ui-settings-migrations.sh
   ```

2. **Verify Database**:
   ```sql
   SELECT * FROM circuit_configurations LIMIT 1;
   SELECT current_circuit_config_id FROM user_sessions LIMIT 1;
   ```

3. **Test Auto-Save**:
   ```bash
   npm run dev
   # Change UI settings and observe auto-save status indicator
   ```

The database schema is now fully configured to support the comprehensive UI settings auto-save system!