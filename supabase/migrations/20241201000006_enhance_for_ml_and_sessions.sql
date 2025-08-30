-- Enhanced database schema for ML integration, session management, and comprehensive parameter tracking

-- Session Management Table
-- Stores user session data with environment variables and settings
CREATE TABLE user_sessions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    session_name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Environment variables and settings
    environment_variables JSONB DEFAULT '{}'::jsonb,
    visualization_settings JSONB DEFAULT '{}'::jsonb,
    performance_settings JSONB DEFAULT '{}'::jsonb,
    
    -- Session metadata
    is_active BOOLEAN DEFAULT true,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Session statistics
    total_computations INTEGER DEFAULT 0,
    total_models_generated BIGINT DEFAULT 0,
    total_computation_time INTERVAL DEFAULT '0 seconds',
    
    UNIQUE(user_id, session_name)
);

-- Tagged Models Table
-- Stores models that users have tagged in the 3D spider plot
CREATE TABLE tagged_models (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id UUID REFERENCES user_sessions(id) ON DELETE CASCADE,
    configuration_id UUID REFERENCES saved_configurations(id) ON DELETE CASCADE,
    
    -- Model identification
    model_id VARCHAR(255) NOT NULL, -- From ModelSnapshot.id
    tag_name VARCHAR(100) NOT NULL,
    tag_category VARCHAR(50) DEFAULT 'user', -- 'user', 'ml_generated', 'optimal', 'interesting'
    
    -- Model parameters at time of tagging
    circuit_parameters JSONB NOT NULL,
    resnorm_value DOUBLE PRECISION,
    impedance_spectrum JSONB, -- Store the full impedance data
    
    -- Tagging context
    tagged_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tagging_context JSONB, -- Stores context like current view settings, zoom level, etc.
    notes TEXT,
    
    -- ML relevance scoring
    ml_relevance_score DOUBLE PRECISION, -- For ML training prioritization
    is_ml_training_data BOOLEAN DEFAULT false,
    
    UNIQUE(user_id, session_id, model_id, tag_name)
);

-- Parameter Exploration Sessions Table
-- Tracks detailed parameter space exploration sessions
CREATE TABLE parameter_exploration_sessions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id UUID REFERENCES user_sessions(id) ON DELETE CASCADE,
    
    -- Exploration metadata
    exploration_name VARCHAR(255) NOT NULL,
    exploration_type VARCHAR(50) DEFAULT 'manual', -- 'manual', 'automated', 'ml_guided'
    
    -- Parameter space definition
    parameter_ranges JSONB NOT NULL, -- Full parameter ranges being explored
    sampling_strategy JSONB NOT NULL, -- Grid vs random vs latin hypercube etc.
    
    -- Exploration results
    total_parameter_combinations INTEGER,
    successful_computations INTEGER,
    failed_computations INTEGER,
    
    -- Performance tracking
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    total_duration INTERVAL,
    average_computation_time INTERVAL,
    
    -- ML integration
    ml_objectives JSONB, -- What we're optimizing for
    ml_constraints JSONB, -- Constraints to respect
    ml_recommendations JSONB, -- AI-generated parameter recommendations
    
    -- Status tracking
    status VARCHAR(50) DEFAULT 'active', -- 'active', 'completed', 'paused', 'failed'
    progress_percentage DOUBLE PRECISION DEFAULT 0.0
);

-- ML Training Datasets Table
-- Preprocessed datasets ready for machine learning
CREATE TABLE ml_training_datasets (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    
    -- Dataset metadata
    dataset_name VARCHAR(255) NOT NULL,
    dataset_version INTEGER DEFAULT 1,
    description TEXT,
    
    -- Dataset composition
    source_configurations UUID[] NOT NULL, -- Array of configuration IDs used
    source_sessions UUID[] NOT NULL, -- Array of session IDs used
    total_samples INTEGER NOT NULL,
    feature_count INTEGER NOT NULL,
    
    -- ML-specific metadata
    dataset_type VARCHAR(50) NOT NULL, -- 'regression', 'classification', 'optimization'
    target_variable VARCHAR(100), -- What we're predicting/optimizing
    feature_engineering_config JSONB,
    
    -- Data storage
    feature_matrix_path TEXT, -- Path to stored feature matrix (HDF5/Parquet)
    target_vector_path TEXT, -- Path to stored target vector
    metadata_path TEXT, -- Path to additional metadata
    
    -- Quality metrics
    data_quality_score DOUBLE PRECISION,
    completeness_percentage DOUBLE PRECISION,
    outlier_percentage DOUBLE PRECISION,
    
    -- Versioning and tracking
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true,
    
    UNIQUE(user_id, dataset_name, dataset_version)
);

-- ML Models Table
-- Store trained machine learning models
CREATE TABLE ml_models (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    training_dataset_id UUID REFERENCES ml_training_datasets(id) ON DELETE SET NULL,
    
    -- Model metadata
    model_name VARCHAR(255) NOT NULL,
    model_version INTEGER DEFAULT 1,
    model_type VARCHAR(100) NOT NULL, -- 'neural_network', 'random_forest', 'gaussian_process', etc.
    description TEXT,
    
    -- Training configuration
    training_config JSONB NOT NULL,
    hyperparameters JSONB NOT NULL,
    
    -- Model performance
    training_metrics JSONB, -- Accuracy, loss, etc. during training
    validation_metrics JSONB, -- Performance on validation set
    test_metrics JSONB, -- Performance on test set
    
    -- Model storage
    model_path TEXT NOT NULL, -- Path to serialized model
    model_size_bytes BIGINT,
    
    -- Deployment status
    is_deployed BOOLEAN DEFAULT false,
    deployment_endpoint TEXT,
    
    -- Metadata
    trained_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    training_duration INTERVAL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(user_id, model_name, model_version)
);

-- Visualization State Snapshots Table
-- Store complete visualization states for reproducibility
CREATE TABLE visualization_snapshots (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id UUID REFERENCES user_sessions(id) ON DELETE CASCADE,
    configuration_id UUID REFERENCES saved_configurations(id) ON DELETE CASCADE,
    
    -- Snapshot metadata
    snapshot_name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Complete visualization state
    visualization_type VARCHAR(50) NOT NULL, -- 'spider2d', 'spider3d', 'nyquist'
    camera_position JSONB, -- 3D camera state
    filter_settings JSONB, -- Resnorm ranges, group selections
    display_settings JSONB, -- Colors, opacity, labels
    
    -- UI state
    panel_states JSONB, -- Which panels are open/closed
    slider_values JSONB, -- All slider positions
    selected_models UUID[], -- Currently selected/highlighted models
    
    -- Annotations
    user_annotations JSONB, -- User-added notes, arrows, etc.
    
    -- Sharing and collaboration
    is_public BOOLEAN DEFAULT false,
    shared_with UUID[], -- Array of user IDs
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Parameter Optimization Jobs Table
-- Track automated parameter optimization runs
CREATE TABLE parameter_optimization_jobs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id UUID REFERENCES user_sessions(id) ON DELETE CASCADE,
    ml_model_id UUID REFERENCES ml_models(id) ON DELETE SET NULL,
    
    -- Job configuration
    job_name VARCHAR(255) NOT NULL,
    optimization_algorithm VARCHAR(100) NOT NULL, -- 'genetic', 'bayesian', 'grid_search', etc.
    objective_function VARCHAR(100) NOT NULL, -- What we're optimizing
    
    -- Parameter space constraints
    parameter_bounds JSONB NOT NULL,
    constraints JSONB,
    
    -- Optimization settings
    max_iterations INTEGER DEFAULT 100,
    convergence_tolerance DOUBLE PRECISION DEFAULT 1e-6,
    population_size INTEGER DEFAULT 50,
    
    -- Progress tracking
    current_iteration INTEGER DEFAULT 0,
    best_parameters JSONB,
    best_objective_value DOUBLE PRECISION,
    convergence_history JSONB,
    
    -- Resource usage
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    estimated_completion TIMESTAMP WITH TIME ZONE,
    cpu_hours_used DOUBLE PRECISION DEFAULT 0,
    
    -- Status
    status VARCHAR(50) DEFAULT 'queued', -- 'queued', 'running', 'completed', 'failed', 'cancelled'
    error_message TEXT,
    progress_percentage DOUBLE PRECISION DEFAULT 0.0
);

-- Add indexes for performance
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_last_accessed ON user_sessions(last_accessed);
CREATE INDEX idx_tagged_models_user_session ON tagged_models(user_id, session_id);
CREATE INDEX idx_tagged_models_resnorm ON tagged_models(resnorm_value);
CREATE INDEX idx_tagged_models_ml_training ON tagged_models(is_ml_training_data) WHERE is_ml_training_data = true;
CREATE INDEX idx_parameter_exploration_status ON parameter_exploration_sessions(status);
CREATE INDEX idx_ml_datasets_active ON ml_training_datasets(user_id) WHERE is_active = true;
CREATE INDEX idx_ml_models_deployed ON ml_models(user_id) WHERE is_deployed = true;
CREATE INDEX idx_optimization_jobs_status ON parameter_optimization_jobs(status);

-- Add RLS (Row Level Security) policies
ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE tagged_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE parameter_exploration_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml_training_datasets ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE visualization_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE parameter_optimization_jobs ENABLE ROW LEVEL SECURITY;

-- RLS Policies for user_sessions
CREATE POLICY "Users can view their own sessions" ON user_sessions
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own sessions" ON user_sessions
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own sessions" ON user_sessions
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own sessions" ON user_sessions
    FOR DELETE USING (auth.uid() = user_id);

-- RLS Policies for tagged_models
CREATE POLICY "Users can view their own tagged models" ON tagged_models
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own tagged models" ON tagged_models
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own tagged models" ON tagged_models
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own tagged models" ON tagged_models
    FOR DELETE USING (auth.uid() = user_id);

-- RLS Policies for parameter_exploration_sessions
CREATE POLICY "Users can view their own exploration sessions" ON parameter_exploration_sessions
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own exploration sessions" ON parameter_exploration_sessions
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own exploration sessions" ON parameter_exploration_sessions
    FOR UPDATE USING (auth.uid() = user_id);

-- RLS Policies for ml_training_datasets
CREATE POLICY "Users can view their own datasets" ON ml_training_datasets
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own datasets" ON ml_training_datasets
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own datasets" ON ml_training_datasets
    FOR UPDATE USING (auth.uid() = user_id);

-- RLS Policies for ml_models
CREATE POLICY "Users can view their own ML models" ON ml_models
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own ML models" ON ml_models
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own ML models" ON ml_models
    FOR UPDATE USING (auth.uid() = user_id);

-- RLS Policies for visualization_snapshots
CREATE POLICY "Users can view their own snapshots" ON visualization_snapshots
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can view public snapshots" ON visualization_snapshots
    FOR SELECT USING (is_public = true);

CREATE POLICY "Users can view snapshots shared with them" ON visualization_snapshots
    FOR SELECT USING (auth.uid() = ANY(shared_with));

CREATE POLICY "Users can insert their own snapshots" ON visualization_snapshots
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own snapshots" ON visualization_snapshots
    FOR UPDATE USING (auth.uid() = user_id);

-- RLS Policies for parameter_optimization_jobs
CREATE POLICY "Users can view their own optimization jobs" ON parameter_optimization_jobs
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own optimization jobs" ON parameter_optimization_jobs
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own optimization jobs" ON parameter_optimization_jobs
    FOR UPDATE USING (auth.uid() = user_id);

-- Create functions for common operations

-- Function to get active session for user
CREATE OR REPLACE FUNCTION get_active_user_session(user_uuid UUID)
RETURNS TABLE (
    session_id UUID,
    session_name VARCHAR,
    environment_variables JSONB,
    visualization_settings JSONB,
    performance_settings JSONB,
    last_accessed TIMESTAMP WITH TIME ZONE
) 
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        us.id,
        us.session_name,
        us.environment_variables,
        us.visualization_settings,
        us.performance_settings,
        us.last_accessed
    FROM user_sessions us
    WHERE us.user_id = user_uuid 
    AND us.is_active = true
    ORDER BY us.last_accessed DESC
    LIMIT 1;
END;
$$;

-- Function to update session activity
CREATE OR REPLACE FUNCTION update_session_activity(session_uuid UUID)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    UPDATE user_sessions 
    SET last_accessed = NOW(), updated_at = NOW()
    WHERE id = session_uuid;
END;
$$;

-- Function to get ML-ready dataset
CREATE OR REPLACE FUNCTION get_ml_training_data(
    user_uuid UUID,
    dataset_type VARCHAR DEFAULT NULL,
    min_samples INTEGER DEFAULT 100
)
RETURNS TABLE (
    dataset_id UUID,
    dataset_name VARCHAR,
    total_samples INTEGER,
    feature_count INTEGER,
    data_quality_score DOUBLE PRECISION,
    created_at TIMESTAMP WITH TIME ZONE
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ds.id,
        ds.dataset_name,
        ds.total_samples,
        ds.feature_count,
        ds.data_quality_score,
        ds.created_at
    FROM ml_training_datasets ds
    WHERE ds.user_id = user_uuid
    AND ds.is_active = true
    AND ds.total_samples >= min_samples
    AND (dataset_type IS NULL OR ds.dataset_type = dataset_type)
    ORDER BY ds.data_quality_score DESC, ds.created_at DESC;
END;
$$;

-- Function to automatically create ML training data from tagged models
CREATE OR REPLACE FUNCTION generate_ml_dataset_from_tags(
    user_uuid UUID,
    dataset_name_param VARCHAR,
    target_tag_categories VARCHAR[] DEFAULT ARRAY['optimal', 'interesting']
)
RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    dataset_id UUID;
    sample_count INTEGER;
BEGIN
    -- Count eligible tagged models
    SELECT COUNT(*) INTO sample_count
    FROM tagged_models tm
    WHERE tm.user_id = user_uuid
    AND tm.tag_category = ANY(target_tag_categories)
    AND tm.impedance_spectrum IS NOT NULL;
    
    -- Create dataset entry
    INSERT INTO ml_training_datasets (
        user_id,
        dataset_name,
        dataset_type,
        total_samples,
        feature_count,
        source_configurations,
        source_sessions,
        target_variable,
        data_quality_score
    ) VALUES (
        user_uuid,
        dataset_name_param,
        'optimization',
        sample_count,
        5, -- Circuit parameters: Rsh, Ra, Ca, Rb, Cb
        ARRAY(SELECT DISTINCT configuration_id FROM tagged_models WHERE user_id = user_uuid),
        ARRAY(SELECT DISTINCT session_id FROM tagged_models WHERE user_id = user_uuid),
        'resnorm_value',
        CASE 
            WHEN sample_count >= 1000 THEN 0.95
            WHEN sample_count >= 500 THEN 0.85
            WHEN sample_count >= 100 THEN 0.75
            ELSE 0.65
        END
    ) RETURNING id INTO dataset_id;
    
    RETURN dataset_id;
END;
$$;

-- Grant necessary permissions
GRANT USAGE ON SCHEMA public TO authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA public TO authenticated;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO authenticated;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO authenticated;