-- Manual Supabase Database Setup for SpideyPlot
-- Run this SQL in your Supabase dashboard: Project > SQL Editor
-- Copy and paste each section individually

-- 1. Session Management Table
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

-- 2. Tagged Models Table
CREATE TABLE tagged_models (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id UUID REFERENCES user_sessions(id) ON DELETE CASCADE,
    
    -- Model identification
    model_id VARCHAR(255) NOT NULL,
    tag_name VARCHAR(100) NOT NULL,
    tag_category VARCHAR(50) DEFAULT 'user',
    
    -- Model parameters at time of tagging
    circuit_parameters JSONB NOT NULL,
    resnorm_value DOUBLE PRECISION,
    impedance_spectrum JSONB,
    
    -- Tagging context
    tagged_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tagging_context JSONB,
    notes TEXT,
    
    -- ML features
    ml_relevance_score DOUBLE PRECISION DEFAULT 0,
    is_ml_training_data BOOLEAN DEFAULT false
);

-- 3. ML Training Datasets Table  
CREATE TABLE ml_training_datasets (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    dataset_name VARCHAR(255) NOT NULL,
    
    -- Data sources
    source_sessions UUID[] NOT NULL DEFAULT '{}',
    total_samples INTEGER NOT NULL DEFAULT 0,
    feature_count INTEGER NOT NULL DEFAULT 0,
    
    -- Dataset metadata
    dataset_type VARCHAR(50) NOT NULL DEFAULT 'regression',
    data_quality_score DOUBLE PRECISION DEFAULT 0,
    feature_matrix_path TEXT,
    target_vector_path TEXT,
    
    -- Processing configuration
    feature_config JSONB DEFAULT '{}',
    preprocessing_config JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 4. ML Models Table
CREATE TABLE ml_models (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    training_dataset_id UUID REFERENCES ml_training_datasets(id) ON DELETE CASCADE,
    
    model_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    
    -- Model configuration
    training_config JSONB NOT NULL DEFAULT '{}',
    hyperparameters JSONB NOT NULL DEFAULT '{}',
    
    -- Performance metrics
    training_metrics JSONB DEFAULT '{}',
    validation_metrics JSONB DEFAULT '{}',
    test_metrics JSONB DEFAULT '{}',
    
    -- Model storage
    model_path TEXT NOT NULL,
    model_size_bytes BIGINT DEFAULT 0,
    
    -- Deployment
    is_deployed BOOLEAN DEFAULT false,
    deployment_endpoint TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 5. Enable Row Level Security
ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE tagged_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml_training_datasets ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml_models ENABLE ROW LEVEL SECURITY;

-- 6. Create RLS Policies
CREATE POLICY "Users can manage their own sessions" ON user_sessions
    FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can manage their own tagged models" ON tagged_models
    FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can manage their own datasets" ON ml_training_datasets
    FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can manage their own ML models" ON ml_models
    FOR ALL USING (auth.uid() = user_id);

-- 7. Create indexes for performance
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_active ON user_sessions(user_id, is_active);
CREATE INDEX idx_tagged_models_user_id ON tagged_models(user_id);
CREATE INDEX idx_tagged_models_session_id ON tagged_models(session_id);
CREATE INDEX idx_tagged_models_ml_training ON tagged_models(user_id, is_ml_training_data);
CREATE INDEX idx_ml_datasets_user_id ON ml_training_datasets(user_id);
CREATE INDEX idx_ml_models_user_id ON ml_models(user_id);

-- 8. Create helper functions
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 9. Add update triggers
CREATE TRIGGER update_user_sessions_updated_at BEFORE UPDATE ON user_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ml_training_datasets_updated_at BEFORE UPDATE ON ml_training_datasets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ml_models_updated_at BEFORE UPDATE ON ml_models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();