# Database Schema Migration Plan

## Overview
This plan fixes the tagged models and circuit profiles integration by properly separating concerns and establishing correct relationships between entities.

## Current Issues Being Fixed

### 1. **Tagged Models Not Circuit-Specific**
- **Problem**: Tagged models show globally instead of per-circuit
- **Solution**: Add `circuit_config_id` foreign key to link tagged models to specific circuit configurations

### 2. **Circuit Profiles Don't Persist Across Sessions**
- **Problem**: Circuit configurations are tied to sessions instead of users
- **Solution**: Create dedicated `circuit_configurations` table linked directly to users

### 3. **Mixed Table Usage**
- **Problem**: `user_profiles` table stores circuit data instead of user metadata
- **Solution**: Separate user metadata from circuit configurations

## New Schema Architecture

```
auth.users (Supabase Auth)
    ↓
user_profiles (User metadata & preferences)
    ↓
circuit_configurations (Circuit configs - persist across sessions)
    ↓
tagged_models (Tagged models linked to specific circuits)
    ↑
user_sessions (Session tracking - temporary)
```

## Key Relationships

### Circuit Configurations → Users
- **One-to-Many**: Users can have multiple circuit configurations
- **Persistence**: Circuit configurations persist across ALL sessions
- **Visibility**: User sees all their configurations regardless of current session

### Tagged Models → Circuit Configurations  
- **Many-to-One**: Multiple tagged models per circuit configuration
- **Isolation**: Tagged models only visible when their circuit configuration is active
- **Cascade Delete**: When circuit configuration is deleted, its tagged models are deleted

### Sessions → Users
- **Many-to-One**: Users can have multiple sessions (but typically one active)
- **Temporary**: Sessions are for activity tracking, not data persistence
- **Current Context**: Session tracks which circuit configuration is currently active

## Migration Steps

### Phase 1: Create New Tables
1. Create `circuit_configurations` table (replaces circuit data in `user_profiles`)
2. Update `user_profiles` to only store user metadata
3. Update `tagged_models` to include `circuit_config_id` foreign key
4. Update `user_sessions` to track current circuit configuration

### Phase 2: Data Migration
1. Migrate existing circuit data from `user_profiles` to `circuit_configurations`
2. Update existing `tagged_models` to link to appropriate circuit configurations
3. Clean up old schema remnants

### Phase 3: Application Updates
1. Update `ProfilesService` → `CircuitConfigService`
2. Update session management to track current circuit context
3. Update tagged models service to filter by circuit configuration
4. Update UI to show circuit-specific tagged models

## Benefits

### For Users
- ✅ Circuit configurations persist across browser sessions, devices, logins
- ✅ Tagged models only show for relevant circuits (no clutter)
- ✅ Clear separation between user preferences and circuit data
- ✅ Can switch between different circuit configurations easily

### For Developers
- ✅ Clean separation of concerns
- ✅ Proper foreign key relationships and data integrity
- ✅ Efficient queries with proper indexing
- ✅ Clear data ownership and access patterns

## Implementation Priority

1. **High Priority**: Circuit configuration persistence (users need their configs saved)
2. **High Priority**: Tagged models linking (prevents confusion)
3. **Medium Priority**: User profiles cleanup (can be done gradually)
4. **Low Priority**: Session optimization (already working, just improving)