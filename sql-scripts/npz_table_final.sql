-- NPZ Dataset References Table
-- Links to your existing circuit_configurations table
-- Run this in your Supabase SQL editor

CREATE TABLE public.circuit_npz_datasets (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  
  -- Link to your existing circuit_configurations table
  circuit_config_id UUID REFERENCES circuit_configurations(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  
  -- NPZ file information
  npz_filename VARCHAR(255) NOT NULL,
  dataset_name VARCHAR(255), -- Human readable name
  grid_size INTEGER NOT NULL,
  n_parameters BIGINT NOT NULL,
  n_frequencies INTEGER NOT NULL,
  file_size_mb DECIMAL(10,2) NOT NULL DEFAULT 0,
  
  -- Storage and availability
  storage_location VARCHAR(20) NOT NULL DEFAULT 'local' CHECK (storage_location IN ('local', 'cloud', 'hybrid')),
  is_available BOOLEAN NOT NULL DEFAULT true,
  
  -- Computation metadata
  computation_metadata JSONB,
  frequency_range JSONB, -- [min_freq, max_freq]
  
  -- Timestamps
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  last_checked TIMESTAMPTZ,
  computed_at TIMESTAMPTZ,
  
  -- Constraints
  UNIQUE(circuit_config_id), -- One NPZ per circuit config
  UNIQUE(user_id, npz_filename) -- No duplicate filenames per user
);

-- Indexes for performance
CREATE INDEX idx_npz_datasets_user_id ON circuit_npz_datasets(user_id);
CREATE INDEX idx_npz_datasets_config_id ON circuit_npz_datasets(circuit_config_id);
CREATE INDEX idx_npz_datasets_filename ON circuit_npz_datasets(npz_filename);
CREATE INDEX idx_npz_datasets_available ON circuit_npz_datasets(is_available);
CREATE INDEX idx_npz_datasets_storage ON circuit_npz_datasets(storage_location);

-- Enable RLS
ALTER TABLE circuit_npz_datasets ENABLE ROW LEVEL SECURITY;

-- RLS Policies
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

-- Allow viewing public NPZ datasets (linked to public circuit configs)
CREATE POLICY "Public NPZ datasets viewable"
  ON circuit_npz_datasets FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM circuit_configurations cc
      WHERE cc.id = circuit_config_id 
      AND cc.is_public = true
    )
  );

-- Trigger for updated_at
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

-- Grant permissions
GRANT ALL ON circuit_npz_datasets TO authenticated;
GRANT SELECT ON circuit_npz_datasets TO anon;