# Implementation Plan: Circuit Profiles & Tagged Models Fix

## Phase 1: Database Schema Updates (High Priority)

### 1.1 Apply New Schema Files
```bash
# Apply in order:
1. 01-user-profiles-table.sql
2. 02-circuit-configurations-table.sql  
3. 03-tagged-models-table.sql
4. 04-user-sessions-table.sql
```

### 1.2 Data Migration Script (CRITICAL)
Create migration script to move existing data:

```sql
-- Migrate circuit data from user_profiles to circuit_configurations
INSERT INTO public.circuit_configurations (
  user_id, name, description, circuit_parameters, 
  grid_size, min_freq, max_freq, num_points,
  is_computed, computation_time, total_points, valid_points,
  computation_results, created_at, updated_at
)
SELECT 
  user_id,
  name,
  description,
  parameters as circuit_parameters,
  COALESCE(grid_size, 5),
  COALESCE(min_freq, 0.1),
  COALESCE(max_freq, 100000),
  COALESCE(num_points, 100),
  COALESCE(is_computed, false),
  computation_time,
  total_points,
  valid_points,
  computation_results,
  created_at,
  updated_at
FROM public.user_profiles 
WHERE parameters IS NOT NULL;

-- Update tagged_models to reference circuit_configurations
-- This requires mapping logic based on circuit parameters
```

## Phase 2: Application Code Changes (High Priority)

### 2.1 Create New Service: CircuitConfigService
**File**: `lib/circuitConfigService.ts`

Replace `ProfilesService` with proper circuit configuration management:

```typescript
export class CircuitConfigService {
  // Get all circuit configurations for a user (persist across sessions)
  static async getUserCircuitConfigurations(userId: string): Promise<CircuitConfiguration[]>
  
  // Create new circuit configuration
  static async createCircuitConfiguration(userId: string, config: CreateCircuitConfigRequest): Promise<CircuitConfiguration>
  
  // Update existing circuit configuration
  static async updateCircuitConfiguration(configId: string, updates: Partial<CircuitConfiguration>): Promise<CircuitConfiguration>
  
  // Delete circuit configuration (cascades to tagged models)
  static async deleteCircuitConfiguration(configId: string): Promise<void>
  
  // Set active circuit configuration for current session
  static async setActiveCircuitConfiguration(sessionId: string, configId: string): Promise<void>
}
```

### 2.2 Update SessionManagement
**File**: `app/hooks/useSessionManagement.ts`

Add circuit context tracking:

```typescript
export interface SessionState {
  userId: string | null
  sessionId: string | null
  sessionName: string | null
  currentCircuitConfigId: string | null // NEW: Track active circuit
  isLoading: boolean
  error: string | null
}

export interface SessionActions {
  // Updated to require circuit config ID
  tagModel: (modelData: {
    circuitConfigId: string // NEW: Required field
    modelId: string
    tagName: string
    tagCategory?: string
    circuitParameters: Record<string, unknown>
    resnormValue?: number
    notes?: string
    isInteresting?: boolean
  }) => Promise<boolean>
  
  // NEW: Set active circuit configuration
  setActiveCircuitConfig: (configId: string) => Promise<void>
  
  // NEW: Get tagged models for current circuit
  getTaggedModelsForCurrentCircuit: () => Promise<TaggedModel[]>
}
```

### 2.3 Update Tagged Models Service
**File**: `lib/taggedModelsService.ts`

Create circuit-specific tagged models service:

```typescript
export class TaggedModelsService {
  // Get tagged models for specific circuit configuration
  static async getTaggedModelsForCircuit(
    circuitConfigId: string, 
    userId: string
  ): Promise<TaggedModel[]>
  
  // Create tagged model linked to circuit configuration
  static async createTaggedModel(
    circuitConfigId: string,
    modelData: CreateTaggedModelRequest
  ): Promise<TaggedModel>
  
  // Delete tagged model
  static async deleteTaggedModel(taggedModelId: string): Promise<void>
  
  // Update tagged model (notes, category, etc.)
  static async updateTaggedModel(
    taggedModelId: string, 
    updates: Partial<TaggedModel>
  ): Promise<TaggedModel>
}
```

### 2.4 Update Main Circuit Simulator
**File**: `app/components/CircuitSimulator.tsx`

Key changes needed:

```typescript
// 1. Replace profile management with circuit configuration management
const {
  circuitConfigs,
  activeCircuitConfigId,
  setActiveCircuitConfig,
  createCircuitConfig,
  // ... other circuit config methods
} = useCircuitConfigurations(userId);

// 2. Update tagged model creation to include circuit config ID
const handleTagModel = async (model: ModelSnapshot, tagName: string) => {
  if (!activeCircuitConfigId) {
    setStatusMessage('Please select or create a circuit configuration first');
    return;
  }
  
  const success = await sessionManagement.actions.tagModel({
    circuitConfigId: activeCircuitConfigId, // Required
    modelId: model.id,
    tagName,
    // ... other fields
  });
};

// 3. Filter tagged models by active circuit configuration
const visibleTaggedModels = taggedModels.filter(
  tm => tm.circuitConfigId === activeCircuitConfigId
);
```

## Phase 3: UI/UX Updates (Medium Priority)

### 3.1 Circuit Configuration Selector
Add UI component to switch between saved circuit configurations:

```tsx
<CircuitConfigSelector 
  configs={circuitConfigs}
  activeConfigId={activeCircuitConfigId}
  onConfigChange={setActiveCircuitConfig}
  onCreateNew={createNewCircuitConfig}
/>
```

### 3.2 Update Saved Profiles Component
**File**: `app/components/circuit-simulator/controls/SavedProfiles.tsx`

Rename and update to handle circuit configurations instead of mixed profiles.

### 3.3 Tagged Models Display
Update tagged models display to show:
- "Tagged Models for: [Circuit Config Name]"
- Clear indication when no circuit is selected
- Option to view tagged models from other configurations

## Phase 4: Testing & Validation (High Priority)

### 4.1 Cross-Session Persistence Test
1. Create circuit configuration in Session A
2. Close browser / clear session
3. Login again → verify configuration persists
4. Tag some models in configuration
5. Switch to different configuration → verify tagged models are hidden
6. Switch back → verify tagged models reappear

### 4.2 Multi-Configuration Test
1. Create multiple circuit configurations
2. Tag models in each configuration
3. Verify tagged models only show for active configuration
4. Test switching between configurations

### 4.3 Data Migration Validation
1. Backup existing data before migration
2. Run migration scripts
3. Verify all existing circuit data is preserved
4. Verify all existing tagged models are properly linked

## Implementation Checklist

### Database Changes
- [ ] Apply schema migration files
- [ ] Create data migration script
- [ ] Test migration on copy of production data
- [ ] Backup existing data before migration

### Service Layer Changes  
- [ ] Create `CircuitConfigService`
- [ ] Create `TaggedModelsService`
- [ ] Update `useSessionManagement` hook
- [ ] Create `useCircuitConfigurations` hook

### Component Updates
- [ ] Update `CircuitSimulator.tsx`
- [ ] Update `SavedProfiles.tsx` → `CircuitConfigSelector.tsx`
- [ ] Update tagged models display components
- [ ] Add circuit configuration management UI

### Testing
- [ ] Unit tests for new services
- [ ] Integration tests for cross-session persistence
- [ ] UI tests for configuration switching
- [ ] Data migration validation tests

## Risk Mitigation

### Data Loss Prevention
- Complete database backup before migration
- Test migration on staging environment first
- Implement rollback plan if migration fails

### User Experience
- Show migration progress to users
- Provide clear messaging about new features
- Maintain backward compatibility during transition

### Performance
- Add proper database indexes (included in schema files)
- Implement efficient queries for large datasets
- Monitor query performance after migration

## Timeline Estimate

- **Phase 1 (Database)**: 1-2 days
- **Phase 2 (Services)**: 3-4 days  
- **Phase 3 (UI)**: 2-3 days
- **Phase 4 (Testing)**: 2-3 days
- **Total**: ~8-12 days

## Success Criteria

✅ Circuit configurations persist across all sessions and browser restarts  
✅ Tagged models only appear for their associated circuit configuration  
✅ Users can switch between multiple saved circuit configurations  
✅ No data loss during migration  
✅ Performance is maintained or improved  
✅ All existing functionality continues to work