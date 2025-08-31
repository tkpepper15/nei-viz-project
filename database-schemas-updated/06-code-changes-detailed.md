# Detailed Code Changes Required

## Files to Create (New)

### 1. `lib/circuitConfigService.ts`
**Purpose**: Replace ProfilesService with proper circuit configuration management
**Key Features**:
- CRUD operations for circuit configurations
- User-scoped queries (cross-session persistence)
- Proper TypeScript interfaces
- Error handling and validation

### 2. `lib/taggedModelsService.ts`
**Purpose**: Manage tagged models linked to specific circuit configurations
**Key Features**:
- Get tagged models for specific circuit configuration
- Create/update/delete tagged models with circuit context
- Proper foreign key relationships

### 3. `app/hooks/useCircuitConfigurations.ts`
**Purpose**: React hook for circuit configuration management
**Key Features**:
- Load user's circuit configurations
- Handle active configuration state
- CRUD operations with state updates
- Error handling and loading states

### 4. `app/components/circuit-simulator/controls/CircuitConfigSelector.tsx`
**Purpose**: UI for selecting and managing circuit configurations
**Key Features**:
- Dropdown/list of saved configurations
- Create new configuration button
- Delete/rename configurations
- Clear indication of active configuration

## Files to Modify (Major Changes)

### 1. `lib/profilesService.ts` → RENAME/REPLACE
**Changes**:
- Either rename to `circuitConfigService.ts` OR delete and replace
- Update all table references from `user_profiles` to `circuit_configurations`
- Update field mappings to match new schema
- Add proper TypeScript interfaces for circuit configurations

### 2. `app/hooks/useSessionManagement.ts`
**Changes**:
```typescript
// Add to SessionState interface:
currentCircuitConfigId: string | null

// Update SessionActions interface:
tagModel: (modelData: {
  circuitConfigId: string // NEW: Required field
  modelId: string
  tagName: string
  // ... existing fields
}) => Promise<boolean>

// Add new action:
setActiveCircuitConfig: (configId: string) => Promise<void>

// Update tagModel implementation:
const tagModel = async (modelData) => {
  const insertData = {
    user_id: sessionState.userId,
    session_id: sessionState.sessionId,
    circuit_config_id: modelData.circuitConfigId, // NEW: Required
    model_id: modelData.modelId,
    // ... other fields
  };
  // ... rest of implementation
};
```

### 3. `app/components/CircuitSimulator.tsx`
**Major Changes**:
```typescript
// Replace profile management:
// OLD:
const { savedProfilesState, setSavedProfilesState, handleSaveProfile } = useProfileManagement(/* ... */);

// NEW:
const { 
  circuitConfigs, 
  activeCircuitConfigId,
  setActiveCircuitConfig,
  createCircuitConfig,
  updateCircuitConfig,
  deleteCircuitConfig 
} = useCircuitConfigurations(user?.id);

// Update tagged model creation:
// OLD:
const handleTagModel = async (model, tagName) => {
  const success = await sessionManagement.actions.tagModel({
    modelId: model.id,
    tagName,
    // ... other fields
  });
};

// NEW:
const handleTagModel = async (model, tagName) => {
  if (!activeCircuitConfigId) {
    setStatusMessage('Please select a circuit configuration first');
    return;
  }
  
  const success = await sessionManagement.actions.tagModel({
    circuitConfigId: activeCircuitConfigId, // Required
    modelId: model.id,
    tagName,
    // ... other fields
  });
};

// Add circuit configuration selector to UI:
<CircuitConfigSelector 
  configs={circuitConfigs}
  activeConfigId={activeCircuitConfigId}
  onConfigChange={setActiveCircuitConfig}
  onCreateNew={() => {
    // Create new config with current parameters
    createCircuitConfig({
      name: `Configuration ${Date.now()}`,
      circuitParameters: groundTruthParams,
      gridSize,
      minFreq,
      maxFreq,
      numPoints
    });
  }}
/>

// Filter tagged models by active configuration:
const visibleTaggedModels = useMemo(() => 
  taggedModels.filter(tm => tm.circuitConfigId === activeCircuitConfigId),
  [taggedModels, activeCircuitConfigId]
);
```

### 4. `app/components/circuit-simulator/controls/SavedProfiles.tsx`
**Option A: Rename and Update**
- Rename to `CircuitConfigManager.tsx`
- Update all references from profiles to circuit configurations
- Add circuit configuration selector
- Update CRUD operations

**Option B: Replace Completely**
- Delete existing file
- Create new `CircuitConfigSelector.tsx` and `CircuitConfigManager.tsx`

### 5. `app/hooks/useUserProfiles.tsx`
**Changes**:
- Either update to work with new circuit configurations
- Or replace with `useCircuitConfigurations.ts`

## Files to Modify (Minor Changes)

### 1. `lib/supabase.ts`
**Changes**:
- Update any hardcoded references to `user_profiles` table
- Add helper functions for circuit configurations if needed
- Update tagged models functions to include `circuit_config_id`

### 2. Database Types File
**Changes**:
- Regenerate `lib/database.types.ts` after schema migration
- Or manually add interfaces for new tables

### 3. Component Files Using SavedProfiles
Search for and update imports:
```bash
grep -r "SavedProfiles" app/components/
```

## TypeScript Interfaces to Add

### 1. Circuit Configuration Types
```typescript
// In lib/circuitConfigService.ts or types/
export interface CircuitConfiguration {
  id: string;
  userId: string;
  name: string;
  description?: string;
  isPublic: boolean;
  circuitParameters: CircuitParameters;
  gridSize: number;
  minFreq: number;
  maxFreq: number;
  numPoints: number;
  isComputed: boolean;
  computationTime?: number;
  totalPoints?: number;
  validPoints?: number;
  computationResults?: unknown;
  createdAt: string;
  updatedAt: string;
}

export interface CreateCircuitConfigRequest {
  name: string;
  description?: string;
  circuitParameters: CircuitParameters;
  gridSize: number;
  minFreq: number;
  maxFreq: number;
  numPoints: number;
}
```

### 2. Tagged Model Types
```typescript
// In lib/taggedModelsService.ts or types/
export interface TaggedModel {
  id: string;
  userId: string;
  sessionId?: string;
  circuitConfigId: string; // NEW: Links to circuit configuration
  modelId: string;
  tagName: string;
  tagCategory: string;
  circuitParameters: CircuitParameters;
  resnormValue?: number;
  taggedAt: string;
  notes?: string;
  isInteresting: boolean;
}

export interface CreateTaggedModelRequest {
  modelId: string;
  tagName: string;
  tagCategory?: string;
  circuitParameters: CircuitParameters;
  resnormValue?: number;
  notes?: string;
  isInteresting?: boolean;
}
```

## Database Migration Script

### Critical Migration Steps
```sql
-- 1. Backup existing data
CREATE TABLE user_profiles_backup AS SELECT * FROM public.user_profiles;
CREATE TABLE tagged_models_backup AS SELECT * FROM public.tagged_models;

-- 2. Create new circuit_configurations table
-- (Run 02-circuit-configurations-table.sql)

-- 3. Migrate circuit data
INSERT INTO public.circuit_configurations (
  user_id, name, description, circuit_parameters,
  grid_size, min_freq, max_freq, num_points,
  is_computed, computation_time, total_points, valid_points,
  computation_results, created_at, updated_at
)
SELECT 
  user_id,
  COALESCE(name, 'Imported Configuration') as name,
  description,
  parameters as circuit_parameters,
  COALESCE(grid_size, 5) as grid_size,
  COALESCE(min_freq, 0.1) as min_freq,
  COALESCE(max_freq, 100000) as max_freq,
  COALESCE(num_points, 100) as num_points,
  COALESCE(is_computed, false) as is_computed,
  computation_time,
  total_points,
  valid_points,
  computation_results,
  COALESCE(created_at, NOW()) as created_at,
  COALESCE(updated_at, NOW()) as updated_at
FROM public.user_profiles 
WHERE parameters IS NOT NULL;

-- 4. Update tagged_models table structure
-- (Run 03-tagged-models-table.sql - this will add circuit_config_id column)

-- 5. Link existing tagged models to circuit configurations
-- This is complex and may require custom logic based on circuit parameters matching
UPDATE public.tagged_models tm
SET circuit_config_id = (
  SELECT cc.id 
  FROM public.circuit_configurations cc 
  WHERE cc.user_id = tm.user_id 
  ORDER BY cc.created_at ASC 
  LIMIT 1
);

-- 6. Clean up user_profiles table
-- Remove circuit-related columns, keep only user metadata
ALTER TABLE public.user_profiles 
DROP COLUMN IF EXISTS parameters,
DROP COLUMN IF EXISTS grid_size,
DROP COLUMN IF EXISTS min_freq,
DROP COLUMN IF EXISTS max_freq,
DROP COLUMN IF EXISTS num_points,
DROP COLUMN IF EXISTS is_computed,
DROP COLUMN IF EXISTS computation_time,
DROP COLUMN IF EXISTS total_points,
DROP COLUMN IF EXISTS valid_points,
DROP COLUMN IF EXISTS computation_results;

-- 7. Add user metadata columns if they don't exist
ALTER TABLE public.user_profiles 
ADD COLUMN IF NOT EXISTS username TEXT,
ADD COLUMN IF NOT EXISTS full_name TEXT,
ADD COLUMN IF NOT EXISTS avatar_url TEXT;
```

## Testing Strategy

### 1. Before Migration
- Export all existing user data
- Document current functionality
- Create test cases for existing features

### 2. After Migration
- Verify all circuit configurations were migrated
- Verify all tagged models are properly linked
- Test cross-session persistence
- Test multi-configuration switching
- Verify no data loss

### 3. User Acceptance Testing
- Test with actual users
- Verify improved workflow
- Check for any missing features

## Rollback Plan

If migration fails:
1. Restore tables from backup
2. Revert application code changes
3. Deploy previous version
4. Investigate issues and retry migration

## Performance Considerations

### Database Indexes
All necessary indexes are included in schema files:
- `idx_circuit_configs_user_id` - for user's configurations
- `idx_tagged_models_circuit_config` - for circuit-specific tagged models
- `idx_tagged_models_user_circuit` - for user + circuit queries

### Query Optimization
- Use JOIN queries instead of multiple round trips
- Implement proper pagination for large datasets
- Cache frequently accessed circuit configurations

### Memory Usage
- Load only active circuit configuration data
- Lazy-load tagged models when switching configurations
- Implement proper cleanup when switching circuits