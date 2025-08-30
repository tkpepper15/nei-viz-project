# Simplified Supabase SQL - Session Storage Only

These SQL files create a lightweight database schema that replaces localStorage with Supabase storage, without heavy ML data storage.

## What Gets Stored in Supabase:
- **User Sessions**: Environment variables, visualization settings, performance settings
- **Saved Profiles**: Circuit parameter profiles (what was in localStorage)
- **Tagged Models**: Just the essential info (parameters, resnorm, tag name) - no full computation data
- **User Preferences**: Session management and UI state

## What Stays Local:
- **Full Computation Data**: All impedance spectra, large datasets
- **ML Training Data**: Feature matrices, model files, training datasets
- **Heavy Visualizations**: Full spider plot data, computation results

## Deployment Instructions

1. **Drop existing tables** (if you ran the full ML version):
   ```sql
   DROP TABLE IF EXISTS ml_models CASCADE;
   DROP TABLE IF EXISTS ml_training_datasets CASCADE;
   DROP TABLE IF EXISTS parameter_optimization_jobs CASCADE;
   DROP TABLE IF EXISTS visualization_snapshots CASCADE;
   ```

2. **Execute these files in order**:
   1. `00_cleanup_existing.sql` (Optional - only if you ran the full ML version first)
   2. `01_create_user_sessions.sql`
   3. `02_create_tagged_models.sql` 
   4. `03_enable_rls.sql`
   5. `04_create_rls_policies.sql`
   6. `05_create_indexes.sql`
   7. `06_create_helper_functions.sql`
   8. `07_create_triggers.sql`
   9. `08_enable_anonymous_auth.sql` (Creates demo user and sample data)
   10. `09_create_user_session.sql` (Creates session for tejjas15@gmail.com)

## Storage Size:
- **Supabase**: ~1-5KB per session (lightweight)
- **Local**: All heavy data remains client-side
- **Perfect for**: Session persistence, profile sharing, basic tagging

## Benefits:
- ✅ Persistent sessions across devices
- ✅ Saved profiles in the cloud  
- ✅ Tagged model references for collaboration
- ✅ Minimal storage costs
- ✅ Fast queries
- ✅ All heavy ML data stays local