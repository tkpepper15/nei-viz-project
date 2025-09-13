# Complete NPZ Integration Setup

## 🎯 **Fully Integrated with Your Existing Schema**

This NPZ system now works seamlessly with your existing:
- ✅ `circuit_configurations` table
- ✅ `user_profiles` table  
- ✅ `user_sessions` table
- ✅ Existing tagged models system
- ✅ RLS policies and authentication

## 🚀 **Setup Steps**

### 1. **Run the Complete SQL Schema**
Execute `npz_integration_final.sql` in your Supabase SQL editor. This creates:

```sql
-- Main NPZ datasets table (links to circuit_configurations)
circuit_npz_datasets

-- Tagged NPZ models (integrates with your tagging system)  
tagged_npz_models

-- Convenient views for queries
npz_datasets_with_config
tagged_npz_summary
```

### 2. **Start the NPZ API Server**
```bash
python circuit_api.py
# Server starts at http://localhost:5000
```

### 3. **Test with Sample Data** 
```bash
python generate_sample_npz.py
# Creates: data/npz/precomputed/sample_grid_5_test.npz
```

## 🔗 **How It Integrates**

### **Circuit Configuration → NPZ Dataset Flow**
```typescript
// 1. User saves circuit configuration (existing)
const config = await saveConfiguration({
  name: "My Circuit",
  circuit_parameters: {...},
  grid_size: 15,
  // ... other settings
});

// 2. User runs Python computation (generates NPZ file)
// python circuit_computation.py

// 3. Link NPZ dataset to configuration
const { registerNPZWithConfiguration } = useNPZData(sessionId);
await registerNPZWithConfiguration(
  config.id,
  'grid_15_freq_0.1_100000.npz',
  metadata
);
```

### **Session-Aware NPZ Management**
```typescript
// NPZ data is now session-aware
const { 
  datasets, 
  taggedModels, 
  datasetStats,
  tagModel 
} = useNPZData(currentSessionId);

// Tag interesting models from NPZ results
await tagModel(
  datasetId,
  parameterRank,    // e.g., rank 1 (best result)
  resnorm,          // residual norm value
  circuitParameters, // {Rsh: 5000, Ra: 3000, ...}
  {
    label: "Best fit",
    category: "optimal",
    qualityScore: 0.95,
    isReference: true
  }
);
```

### **Enhanced Data Relationships**
```
user_profiles ──┐
                ├── user_sessions ──┐
auth.users ─────┘                   ├── circuit_npz_datasets ──┐
                                    │                           ├── tagged_npz_models
circuit_configurations ─────────────┘                           │
                                                                 │
                                    NPZ Files ←─────────────────┘
                                    (data/npz/)
```

## 📊 **New Database Schema Overview**

### **circuit_npz_datasets**
- Links to `circuit_configurations(id)` 
- References `user_sessions(id)` for context
- Stores NPZ file metadata and performance stats
- Tracks availability and file locations

### **tagged_npz_models**  
- Individual parameter sets tagged from NPZ results
- Links to `circuit_npz_datasets(id)`
- Session-aware tagging system
- Integrates with your existing model tagging workflow

## 🎛️ **UI Integration**

The NPZ Dataset Manager is available in:
**Settings Modal → Datasets Tab**

Features:
- 🔍 **Real-time API status** monitoring
- 📊 **Dataset overview** with file sizes and parameters
- 🏷️ **Session filtering** for current work context  
- 🔗 **Profile linking** shows circuit configuration names
- 👥 **Public datasets** browser
- ⚡ **One-click loading** for instant visualization

## 🎯 **Usage Examples**

### **Background Computation Workflow**
```bash
# 1. Generate large dataset overnight
python circuit_computation.py
# → Creates: data/npz/user_generated/grid_15_freq_0.1_100000.npz

# 2. Start API server
python circuit_api.py

# 3. In your app: Link to circuit configuration
# Settings → Datasets → Link to Profile
```

### **Model Tagging Workflow**
```typescript
// Load NPZ dataset with 759K parameter results
const { loadDataset, getBestResults, tagModel } = useNPZData(sessionId);

await loadDataset('grid_15_freq_0.1_100000.npz');
const topResults = await getBestResults(100);

// Tag the best result as reference
await tagModel(
  datasetId,
  1, // Best rank
  topResults[0].resnorm,
  topResults[0].parameters,
  { 
    label: "Reference Model",
    isReference: true 
  }
);
```

## 🔒 **Security & Permissions**

- **RLS Enabled**: Users only see their own datasets
- **Session Context**: NPZ data filtered by user session
- **Public Sharing**: Datasets linked to public circuit configs are discoverable
- **Safe References**: All foreign keys with proper cascade behavior

## ⚡ **Performance Features**

- **Smart Indexing**: Optimized queries for user/session/availability
- **Views**: Pre-joined data for common UI patterns
- **Stats Tracking**: Load times, memory usage, access patterns
- **Compressed Storage**: 600MB NPZ files vs 6GB CSV

## ✅ **Ready to Use!**

Your NPZ integration is now **production-ready** with:

1. ✅ **Complete schema integration** with existing tables
2. ✅ **Session-aware data management** 
3. ✅ **Tagged model system** for research workflows
4. ✅ **Real-time API** for instant dataset access
5. ✅ **Sleek UI** integrated in settings modal
6. ✅ **Performance analytics** and monitoring

Run `npz_integration_final.sql` → Start `python circuit_api.py` → Test in your app!