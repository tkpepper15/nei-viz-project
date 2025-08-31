# Updated Database Schemas & Implementation Plan

This folder contains the complete plan to fix tagged models and circuit profiles integration issues.

## 📁 Files Overview

### Database Schema Files
- **`01-user-profiles-table.sql`** - User metadata table (username, preferences, etc.)
- **`02-circuit-configurations-table.sql`** - Circuit configurations that persist across sessions
- **`03-tagged-models-table.sql`** - Tagged models linked to specific circuit configurations
- **`04-user-sessions-table.sql`** - Session tracking (temporary, for activity monitoring)

### Documentation Files
- **`00-migration-plan.md`** - High-level overview of schema changes and benefits
- **`05-implementation-plan.md`** - Step-by-step implementation guide with timeline
- **`06-code-changes-detailed.md`** - Detailed code changes required for each file

## 🎯 Key Problems Being Solved

### 1. Tagged Models Not Circuit-Specific
**Current Issue**: Tagged models show globally instead of per-circuit
**Solution**: Add `circuit_config_id` foreign key to link tagged models to specific circuits

### 2. Circuit Profiles Don't Persist Across Sessions
**Current Issue**: Circuit configurations lost when browser restarts or user logs out/in
**Solution**: Store circuit configurations at user level, not session level

### 3. Mixed Data in user_profiles Table
**Current Issue**: User metadata mixed with circuit configuration data
**Solution**: Separate concerns with dedicated tables

## 🏗️ New Architecture

```
auth.users (Supabase)
    ↓
user_profiles (User metadata ONLY)
    ↓
circuit_configurations (Persist across ALL sessions)
    ↓
tagged_models (Linked to specific circuits)
    ↑
user_sessions (Temporary session tracking)
```

## ✅ Expected Outcomes

After implementing these changes:

- **Circuit configurations persist forever** (until user deletes them)
- **Tagged models only appear for their circuit** (no confusion)
- **Clean separation of user vs circuit data**
- **Better performance** with proper indexes and relationships
- **Cross-session/device persistence** - login anywhere, see your configs

## 🚀 Quick Start

### For Database Admin
1. Review `00-migration-plan.md` for overview
2. Apply schema files in numerical order (01, 02, 03, 04)
3. Run data migration script (in `06-code-changes-detailed.md`)

### For Developers
1. Read `05-implementation-plan.md` for step-by-step guide
2. Review `06-code-changes-detailed.md` for specific code changes
3. Create new services: `CircuitConfigService`, `TaggedModelsService`
4. Update hooks: `useCircuitConfigurations`, modify `useSessionManagement`
5. Update UI components for circuit configuration management

### For Testing
1. Backup existing data before migration
2. Test on staging environment first
3. Verify circuit configurations persist after browser restart
4. Verify tagged models show only for active circuit
5. Test switching between multiple circuit configurations

## ⚠️ Important Notes

- **Data migration is required** - existing circuit data needs to be moved
- **Breaking changes** - some application code needs updates
- **Backup critical** - always backup before migration
- **Staged rollout recommended** - test thoroughly before production

## 📊 Timeline Estimate

- **Database Migration**: 1-2 days
- **Service Layer Changes**: 3-4 days  
- **UI Updates**: 2-3 days
- **Testing & Validation**: 2-3 days
- **Total**: ~8-12 days

## 🆘 Support

If you encounter issues during implementation:
1. Check the detailed documentation in each file
2. Verify database migration completed successfully
3. Test individual components before full integration
4. Maintain rollback capability until fully validated