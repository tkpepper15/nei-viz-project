# SQL Strategy Implementation

This document provides a comprehensive overview of the SQL strategy implementation for the NEI Visualization Project, designed to streamline parameter experimentation and machine learning integration through optimal database storage.

## 🎯 Overview

The SQL strategy transforms the application from client-side data storage to a comprehensive database-backed system that supports:

- **Session Management**: Persistent user sessions with environment variables
- **Parameter Storage**: All circuit parameters and computation results
- **Tagged Model System**: User-tagged models from 3D spider plot interactions
- **ML-Ready Data Interface**: Direct integration with machine learning workflows
- **Real-time Collaboration**: Shared configurations and visualization snapshots
- **Performance Optimization**: Efficient data retrieval and caching

## 🗄️ Database Schema

### Core Tables

#### 1. **user_sessions**
Manages user sessions with environment variables and settings.
```sql
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id),
    session_name VARCHAR(255) NOT NULL,
    environment_variables JSONB DEFAULT '{}',
    visualization_settings JSONB DEFAULT '{}',
    performance_settings JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    total_computations INTEGER DEFAULT 0,
    total_models_generated BIGINT DEFAULT 0
);
```

#### 2. **tagged_models**
Stores models tagged by users in the 3D spider plot.
```sql
CREATE TABLE tagged_models (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id),
    session_id UUID REFERENCES user_sessions(id),
    model_id VARCHAR(255) NOT NULL,
    tag_name VARCHAR(100) NOT NULL,
    tag_category VARCHAR(50) DEFAULT 'user',
    circuit_parameters JSONB NOT NULL,
    resnorm_value DOUBLE PRECISION,
    impedance_spectrum JSONB,
    tagging_context JSONB,
    ml_relevance_score DOUBLE PRECISION,
    is_ml_training_data BOOLEAN DEFAULT false
);
```

#### 3. **parameter_exploration_sessions**
Tracks detailed parameter space exploration sessions.
```sql
CREATE TABLE parameter_exploration_sessions (
    id UUID PRIMARY KEY,
    exploration_name VARCHAR(255) NOT NULL,
    exploration_type VARCHAR(50) DEFAULT 'manual',
    parameter_ranges JSONB NOT NULL,
    sampling_strategy JSONB NOT NULL,
    ml_objectives JSONB,
    ml_recommendations JSONB,
    status VARCHAR(50) DEFAULT 'active'
);
```

#### 4. **ml_training_datasets**
Preprocessed datasets ready for machine learning.
```sql
CREATE TABLE ml_training_datasets (
    id UUID PRIMARY KEY,
    dataset_name VARCHAR(255) NOT NULL,
    source_configurations UUID[] NOT NULL,
    source_sessions UUID[] NOT NULL,
    total_samples INTEGER NOT NULL,
    feature_count INTEGER NOT NULL,
    dataset_type VARCHAR(50) NOT NULL,
    data_quality_score DOUBLE PRECISION,
    feature_matrix_path TEXT,
    target_vector_path TEXT
);
```

#### 5. **ml_models**
Trained machine learning models with performance metrics.
```sql
CREATE TABLE ml_models (
    id UUID PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    training_config JSONB NOT NULL,
    hyperparameters JSONB NOT NULL,
    training_metrics JSONB,
    validation_metrics JSONB,
    model_path TEXT NOT NULL,
    is_deployed BOOLEAN DEFAULT false
);
```

#### 6. **visualization_snapshots**
Complete visualization states for reproducibility.
```sql
CREATE TABLE visualization_snapshots (
    id UUID PRIMARY KEY,
    snapshot_name VARCHAR(255) NOT NULL,
    visualization_type VARCHAR(50) NOT NULL,
    camera_position JSONB,
    filter_settings JSONB,
    display_settings JSONB,
    selected_models UUID[],
    is_public BOOLEAN DEFAULT false
);
```

#### 7. **parameter_optimization_jobs**
Automated parameter optimization runs.
```sql
CREATE TABLE parameter_optimization_jobs (
    id UUID PRIMARY KEY,
    job_name VARCHAR(255) NOT NULL,
    optimization_algorithm VARCHAR(100) NOT NULL,
    objective_function VARCHAR(100) NOT NULL,
    parameter_bounds JSONB NOT NULL,
    best_parameters JSONB,
    best_objective_value DOUBLE PRECISION,
    status VARCHAR(50) DEFAULT 'queued'
);
```

## 🔧 Service Layer Architecture

### 1. **DatabaseService** (`lib/database-service.ts`)
Main orchestrator that provides clean interfaces for all database operations.

```typescript
// Session Management
await DatabaseService.session.createSession(userId, sessionName, description)
await DatabaseService.session.getActiveSession(userId)
await DatabaseService.session.updateEnvironment(sessionId, environmentVars)

// Tagged Models
await DatabaseService.taggedModels.tagModel(userId, sessionId, configId, modelData)
await DatabaseService.taggedModels.getMLTrainingCandidates(userId, minScore)

// ML Datasets
await DatabaseService.datasets.createDataset(userId, datasetData)
await DatabaseService.datasets.generateDatasetFromTags(userId, datasetName)

// ML Models
await DatabaseService.models.saveModel(userId, modelData)
await DatabaseService.models.deployModel(modelId, endpoint)
```

### 2. **MLDataInterface** (`lib/ml-data-interface.ts`)
Comprehensive ML data processing pipeline.

```typescript
// Extract data from tagged models
const extractedData = await MLDataInterface.extraction.extractFromTaggedModels(
  userId,
  { tagCategories: ['optimal', 'interesting'], includeImpedanceSpectrum: true }
)

// Feature engineering
const engineeredData = MLDataInterface.extraction.engineerFeatures(
  features,
  featureNames,
  { includeTimeConstants: true, includeRatios: true }
)

// Export in multiple formats
const jsonExport = MLDataInterface.export.exportToJSON(dataset)
const csvExport = MLDataInterface.export.exportToCSV(dataset)
const numpyExport = MLDataInterface.export.exportToNumpyFormat(dataset)
const tfExport = MLDataInterface.export.exportToTensorFlowJS(dataset)
```

### 3. **useSessionManagement** (`app/hooks/useSessionManagement.ts`)
React hook for centralized session management.

```typescript
const { sessionState, actions, isReady } = useSessionManagement()

// Create new session
await actions.createSession('My Experiment', 'Testing new parameters')

// Update visualization settings
await actions.updateVisualizationSettings({ groupPortion: 0.3 })

// Tag a model
await actions.tagModel({
  modelId: 'model-123',
  tagName: 'optimal',
  circuitParameters: params,
  resnormValue: 0.05
})
```

## 🚀 Integration Points

### 1. **3D Spider Plot Integration**
- **Automatic tagging**: Models tagged in the 3D visualization are automatically saved to the database
- **Session context**: Camera position, zoom level, and filter settings are captured with each tag
- **Real-time updates**: Tagged models are immediately available for ML training

```typescript
// In SpiderPlot3D.tsx
await DatabaseService.taggedModels.tagModel(userId, sessionId, configId, {
  modelId: clickedModel.id,
  tagName: userTagName,
  circuitParameters: clickedModel.parameters,
  resnormValue: clickedModel.resnorm,
  taggingContext: {
    cameraPosition: camera.position,
    currentZoom: camera.distance,
    filterSettings: { resnormRange }
  }
})
```

### 2. **Parameter Panel Integration**
- **Environment variables**: All UI settings are persisted in the session
- **Configuration tracking**: Parameter changes are automatically logged
- **Profile management**: Saved profiles include full session context

### 3. **ML Workflow Integration**
- **Automatic dataset generation**: Tagged models are automatically processed into ML-ready datasets
- **Feature engineering**: Circuit parameters are enhanced with time constants, ratios, and transformations
- **Model management**: Trained models are versioned and tracked with performance metrics

## 📊 ML Features

### Data Extraction
- **Tagged models**: Extract circuit parameters and resnorm values from user-tagged models
- **Computation results**: Process grid computation results for comprehensive datasets
- **Impedance spectra**: Include full impedance data for advanced feature engineering

### Feature Engineering
- **Time constants**: τ = RC calculations for both apical and basal circuits
- **Parameter ratios**: Ra/Rb and Ca/Cb ratios for relative analysis
- **Log transformations**: Apply logarithmic scaling to appropriate parameters
- **Frequency-dependent features**: Extract features from impedance spectra

### Export Formats
- **JSON**: For JavaScript ML libraries (TensorFlow.js, ml-matrix)
- **CSV**: For R, Python pandas, Excel analysis
- **NumPy**: Direct integration with scikit-learn, PyTorch
- **TensorFlow.js**: Ready for browser-based ML training

### Model Management
- **Versioning**: Automatic model versioning with performance tracking
- **Deployment**: Integration with deployment endpoints
- **Metrics tracking**: Training, validation, and test metrics storage
- **Hyperparameter optimization**: Integration with optimization algorithms

## 🔄 Real-time Features

### Live Updates
- **Session synchronization**: Real-time updates across multiple browser tabs
- **Collaboration**: Shared sessions with live parameter updates
- **Progress tracking**: Real-time optimization job progress

### Subscriptions
```typescript
// Subscribe to ML training progress
const subscription = supabase
  .channel('ml_training_progress')
  .on('postgres_changes', { 
    event: 'UPDATE', 
    schema: 'public', 
    table: 'parameter_optimization_jobs' 
  }, handleProgressUpdate)
  .subscribe()

// Subscribe to tagged model updates
const tagSubscription = subscribeToTaggedModels(userId, sessionId, handleTagUpdate)
```

## 🧪 Testing Strategy

### Comprehensive Test Suite
- **Unit tests**: Individual service methods
- **Integration tests**: End-to-end workflows
- **Performance tests**: Large dataset handling
- **ML pipeline tests**: Feature engineering and export validation

```typescript
// Example test
describe('ML Data Extraction', () => {
  test('should extract features from tagged models', async () => {
    const extractedData = await MLDataInterface.extraction.extractFromTaggedModels(
      testUserId,
      { tagCategories: ['optimal'], includeImpedanceSpectrum: true }
    )
    
    expect(extractedData.features.featureNames).toEqual(['Rsh', 'Ra', 'Ca', 'Rb', 'Cb'])
    expect(extractedData.targets.targetName).toBe('resnorm_value')
  })
})
```

## 📈 Performance Optimizations

### Database Optimizations
- **Indexes**: Strategic indexes on frequently queried columns
- **Partitioning**: Large tables partitioned by user_id and date
- **Connection pooling**: Efficient database connection management
- **Query optimization**: Optimized queries with proper joins and filters

### Caching Strategy
- **Session caching**: Active session data cached in memory
- **Query result caching**: Frequently accessed data cached with Redis
- **ML model caching**: Deployed models cached for fast inference

### Batch Operations
```typescript
// Batch insert tagged models
await DatabaseService.taggedModels.batchInsertTaggedModels(taggedModels)

// Batch update session statistics
await DatabaseService.session.batchUpdateSessionStats(sessionId, stats)
```

## 🔐 Security Features

### Row Level Security (RLS)
- **User isolation**: Users can only access their own data
- **Session-based access**: Access control based on active session
- **Shared resource permissions**: Granular permissions for shared configurations

### Data Privacy
- **Encrypted storage**: Sensitive data encrypted at rest
- **Audit logging**: All data modifications logged for compliance
- **Access controls**: Fine-grained access controls for sensitive operations

## 🚀 Getting Started

### 1. Database Setup
```bash
# Run migrations
supabase migration up

# Set up RLS policies
supabase db push
```

### 2. Environment Variables
```env
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

### 3. Initialize Session
```typescript
import { useSessionManagement } from './hooks/useSessionManagement'

function MyComponent() {
  const { sessionState, actions, isReady } = useSessionManagement()
  
  useEffect(() => {
    if (isReady) {
      // Session is ready, start using the application
      console.log('Active session:', sessionState.sessionName)
    }
  }, [isReady])
}
```

### 4. Start ML Workflow
```typescript
// Process data for ML training
const result = await MLDataInterface.processForTraining(userId, {
  dataSource: 'tagged_models',
  featureEngineering: {
    includeTimeConstants: true,
    includeRatios: true,
    logTransform: ['Rsh', 'Ra', 'Rb']
  },
  exportFormat: 'numpy'
})

console.log('Dataset ready:', result.dataset)
console.log('Exported data:', result.exportedData)
```

## 📋 Migration Guide

### From LocalStorage to Database
1. **Export existing profiles**: Use the existing profile export functionality
2. **Create user session**: Initialize database session for the user
3. **Import profiles**: Convert and import existing profiles to database
4. **Update components**: Replace localStorage calls with database service calls

### Gradual Migration Strategy
- **Phase 1**: Add database integration alongside existing localStorage
- **Phase 2**: Migrate tagged models and session data
- **Phase 3**: Migrate saved profiles and computation results
- **Phase 4**: Remove localStorage dependencies

## 🔮 Future Enhancements

### Advanced ML Features
- **Automated hyperparameter optimization**: Integration with Optuna, Ray Tune
- **Model serving**: RESTful API endpoints for model inference
- **A/B testing**: Framework for testing different parameter strategies
- **Federated learning**: Collaborative model training across users

### Advanced Analytics
- **Usage analytics**: Track parameter exploration patterns
- **Performance analytics**: Monitor computation efficiency
- **Model performance tracking**: Long-term model performance analysis
- **Anomaly detection**: Automatic detection of unusual parameter combinations

### Collaboration Features
- **Team workspaces**: Shared workspaces for research groups
- **Real-time editing**: Collaborative parameter exploration
- **Discussion threads**: Comments and discussions on tagged models
- **Version control**: Git-like versioning for experiment tracking

## 🎯 Benefits

### For Researchers
- **Reproducibility**: Complete experiment tracking and versioning
- **Collaboration**: Easy sharing of configurations and findings
- **ML Integration**: Seamless transition from visualization to machine learning
- **Data Management**: Centralized, organized data storage

### For Developers
- **Clean Architecture**: Well-structured, maintainable codebase
- **Type Safety**: Full TypeScript support with proper types
- **Testing**: Comprehensive test coverage for reliability
- **Scalability**: Database-backed architecture scales with usage

### For the Application
- **Performance**: Optimized data retrieval and caching
- **Reliability**: Persistent data storage with backup capabilities
- **Features**: Rich functionality powered by structured data
- **Insights**: Advanced analytics and reporting capabilities

---

This SQL strategy implementation transforms the NEI Visualization Project into a comprehensive, database-backed platform optimized for parameter experimentation and machine learning integration. The clean architecture, comprehensive testing, and ML-ready interfaces provide a solid foundation for advanced electrochemical impedance spectroscopy research and analysis.