-- NPZ Dataset Integration - Works with your existing schema
-- Integrates with circuit_configurations, user_profiles, and user_sessions
-- Run this in your Supabase SQL editor

-- Create NPZ datasets table
CREATE TABLE public.circuit_npz_datasets (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  
  -- Link to your existing tables
  circuit_config_id UUID NOT NULL REFERENCES circuit_configurations(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  session_id UUID REFERENCES user_sessions(id) ON DELETE SET NULL, -- Optional session context
  
  -- NPZ file information
  npz_filename VARCHAR(255) NOT NULL,
  dataset_name VARCHAR(255), -- Human readable name
  file_path VARCHAR(500), -- Full path: data/npz/precomputed/filename.npz
  
  -- Dataset metadata (mirrors circuit_configurations structure)
  grid_size INTEGER NOT NULL,
  min_freq NUMERIC NOT NULL,
  max_freq NUMERIC NOT NULL, 
  num_points INTEGER NOT NULL,
  n_parameters BIGINT NOT NULL,
  file_size_mb DECIMAL(10,2) NOT NULL DEFAULT 0,
  
  -- Storage and availability
  storage_location VARCHAR(20) NOT NULL DEFAULT 'local' CHECK (storage_location IN ('local', 'cloud', 'hybrid')),
  is_available BOOLEAN NOT NULL DEFAULT true,
  
  -- Computation metadata
  computation_metadata JSONB DEFAULT '{}'::jsonb,
  best_resnorm DECIMAL(15,10), -- Best parameter result
  total_valid_results INTEGER, -- Number of valid results
  computation_time INTERVAL, -- Time taken to compute
  
  -- Performance stats
  load_time_ms INTEGER, -- Time to load into memory
  memory_usage_mb DECIMAL(8,2), -- Memory usage when loaded
  
  -- Timestamps
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  last_checked TIMESTAMPTZ,
  computed_at TIMESTAMPTZ,
  last_accessed TIMESTAMPTZ,
  
  -- Constraints
  UNIQUE(circuit_config_id), -- One NPZ per circuit config
  UNIQUE(user_id, npz_filename), -- No duplicate filenames per user
  
  -- Validation constraints (match your existing patterns)
  CONSTRAINT npz_datasets_min_freq_check CHECK (min_freq > 0),
  CONSTRAINT npz_datasets_freq_range_check CHECK (max_freq > min_freq),
  CONSTRAINT npz_datasets_grid_size_check CHECK (grid_size >= 2 AND grid_size <= 25),
  CONSTRAINT npz_datasets_num_points_check CHECK (num_points >= 10 AND num_points <= 1000)
);

-- Create tagged NPZ models table (integrates with your tagged_models system)
CREATE TABLE public.tagged_npz_models (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  
  -- Links
  npz_dataset_id UUID NOT NULL REFERENCES circuit_npz_datasets(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  session_id UUID REFERENCES user_sessions(id) ON DELETE CASCADE,
  
  -- Model identification (from NPZ results)
  parameter_rank INTEGER NOT NULL, -- Rank in sorted results (1 = best)
  resnorm DECIMAL(15,10) NOT NULL, -- Residual norm value
  
  -- Circuit parameters (extracted from NPZ)
  circuit_parameters JSONB NOT NULL, -- {Rsh, Ra, Ca, Rb, Cb}
  
  -- Tagging info
  tag_label VARCHAR(100), -- User-defined label
  tag_notes TEXT, -- User notes
  tag_category VARCHAR(50), -- e.g., 'best', 'interesting', 'outlier'
  
  -- Spectrum data reference
  frequency_index_start INTEGER, -- Where in spectrum array this model starts
  frequency_index_end INTEGER, -- Where it ends
  
  -- Performance metrics
  model_quality_score DECIMAL(5,3), -- 0-1 score
  is_reference_model BOOLEAN DEFAULT false, -- User-marked as reference
  
  -- Timestamps
  tagged_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  
  -- Constraints
  UNIQUE(npz_dataset_id, parameter_rank), -- One tag per rank per dataset
  CONSTRAINT tagged_npz_models_rank_check CHECK (parameter_rank > 0),
  CONSTRAINT tagged_npz_models_quality_check CHECK (model_quality_score >= 0 AND model_quality_score <= 1)
);

-- Indexes for performance
CREATE INDEX idx_npz_datasets_user_id ON circuit_npz_datasets(user_id, created_at DESC);
CREATE INDEX idx_npz_datasets_config_id ON circuit_npz_datasets(circuit_config_id);
CREATE INDEX idx_npz_datasets_session_id ON circuit_npz_datasets(session_id) WHERE session_id IS NOT NULL;
CREATE INDEX idx_npz_datasets_filename ON circuit_npz_datasets(npz_filename);
CREATE INDEX idx_npz_datasets_available ON circuit_npz_datasets(is_available, storage_location);
CREATE INDEX idx_npz_datasets_performance ON circuit_npz_datasets(user_id, best_resnorm, file_size_mb);

CREATE INDEX idx_tagged_npz_user_session ON tagged_npz_models(user_id, session_id, tagged_at DESC);
CREATE INDEX idx_tagged_npz_dataset ON tagged_npz_models(npz_dataset_id, parameter_rank);
CREATE INDEX idx_tagged_npz_resnorm ON tagged_npz_models(resnorm, model_quality_score);
CREATE INDEX idx_tagged_npz_category ON tagged_npz_models(tag_category, tagged_at DESC);
CREATE INDEX idx_tagged_npz_reference ON tagged_npz_models(is_reference_model) WHERE is_reference_model = true;

-- Enable RLS
ALTER TABLE circuit_npz_datasets ENABLE ROW LEVEL SECURITY;
ALTER TABLE tagged_npz_models ENABLE ROW LEVEL SECURITY;

-- RLS Policies for NPZ datasets
CREATE POLICY "Users can view own NPZ datasets"
  ON circuit_npz_datasets FOR SELECT
  USING (user_id = auth.uid());

CREATE POLICY "Users can create own NPZ datasets"
  ON circuit_npz_datasets FOR INSERT
  WITH CHECK (user_id = auth.uid());

CREATE POLICY "Users can update own NPZ datasets"
  ON circuit_npz_datasets FOR UPDATE
  USING (user_id = auth.uid());

CREATE POLICY "Users can delete own NPZ datasets"
  ON circuit_npz_datasets FOR DELETE
  USING (user_id = auth.uid());

-- Public NPZ datasets (linked to public circuit configs)
CREATE POLICY "Public NPZ datasets viewable"
  ON circuit_npz_datasets FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM circuit_configurations cc
      WHERE cc.id = circuit_config_id 
      AND cc.is_public = true
    )
  );

-- RLS Policies for tagged NPZ models
CREATE POLICY "Users can view own tagged NPZ models"
  ON tagged_npz_models FOR SELECT
  USING (user_id = auth.uid());

CREATE POLICY "Users can create own tagged NPZ models"
  ON tagged_npz_models FOR INSERT
  WITH CHECK (user_id = auth.uid());

CREATE POLICY "Users can update own tagged NPZ models"
  ON tagged_npz_models FOR UPDATE
  USING (user_id = auth.uid());

CREATE POLICY "Users can delete own tagged NPZ models"
  ON tagged_npz_models FOR DELETE
  USING (user_id = auth.uid());

-- Public tagged models (from public datasets)
CREATE POLICY "Public tagged NPZ models viewable"
  ON tagged_npz_models FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM circuit_npz_datasets npz
      JOIN circuit_configurations cc ON cc.id = npz.circuit_config_id
      WHERE npz.id = npz_dataset_id 
      AND cc.is_public = true
    )
  );

-- Triggers for updated_at
CREATE OR REPLACE FUNCTION update_npz_datasets_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_npz_datasets_updated_at
  BEFORE UPDATE ON circuit_npz_datasets
  FOR EACH ROW
  EXECUTE FUNCTION update_npz_datasets_updated_at();

CREATE TRIGGER trigger_update_tagged_npz_updated_at
  BEFORE UPDATE ON tagged_npz_models
  FOR EACH ROW
  EXECUTE FUNCTION update_npz_datasets_updated_at();

-- Trigger to update last_accessed when NPZ dataset is queried
CREATE OR REPLACE FUNCTION update_npz_last_accessed()
RETURNS TRIGGER AS $$
BEGIN
  UPDATE circuit_npz_datasets 
  SET last_accessed = NOW() 
  WHERE id = NEW.npz_dataset_id;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_npz_access
  AFTER INSERT ON tagged_npz_models
  FOR EACH ROW
  EXECUTE FUNCTION update_npz_last_accessed();

-- Grant permissions
GRANT ALL ON circuit_npz_datasets TO authenticated;
GRANT ALL ON tagged_npz_models TO authenticated;
GRANT SELECT ON circuit_npz_datasets TO anon;
GRANT SELECT ON tagged_npz_models TO anon;

-- Sample views for common queries
CREATE VIEW npz_datasets_with_config AS
SELECT 
  npz.*,
  cc.name as config_name,
  cc.description as config_description,
  cc.is_public,
  cc.circuit_parameters,
  up.username,
  up.full_name,
  us.session_name
FROM circuit_npz_datasets npz
JOIN circuit_configurations cc ON cc.id = npz.circuit_config_id
JOIN user_profiles up ON up.user_id = npz.user_id
LEFT JOIN user_sessions us ON us.id = npz.session_id;

CREATE VIEW tagged_npz_summary AS
SELECT 
  tnm.*,
  npz.npz_filename,
  npz.dataset_name,
  cc.name as config_name,
  up.username
FROM tagged_npz_models tnm
JOIN circuit_npz_datasets npz ON npz.id = tnm.npz_dataset_id
JOIN circuit_configurations cc ON cc.id = npz.circuit_config_id
JOIN user_profiles up ON up.user_id = tnm.user_id;

-- Grant view permissions
GRANT SELECT ON npz_datasets_with_config TO authenticated;
GRANT SELECT ON tagged_npz_summary TO authenticated;
GRANT SELECT ON npz_datasets_with_config TO anon;
GRANT SELECT ON tagged_npz_summary TO anon;