-- Safe NPZ Dataset References Table
-- This only creates the NPZ table without touching existing schema
-- Run this in your Supabase SQL editor

-- First, let's see what tables you already have
-- Run this query to see your existing tables:
-- SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE';

-- Create NPZ table with flexible references
CREATE TABLE IF NOT EXISTS public.circuit_npz_datasets (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  
  -- Flexible references - will link to whatever tables you have
  config_reference UUID, -- Link to your existing config table
  user_reference UUID,   -- Link to your existing user/profile table
  
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
  
  -- Circuit parameters (store directly if no config table link)
  circuit_parameters JSONB,
  
  -- Computation metadata
  computation_metadata JSONB,
  frequency_range JSONB, -- [min_freq, max_freq] 
  
  -- Timestamps
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  last_checked TIMESTAMPTZ,
  computed_at TIMESTAMPTZ,
  
  -- Constraints
  UNIQUE(user_reference, npz_filename) -- No duplicate filenames per user
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_circuit_npz_user_ref ON circuit_npz_datasets(user_reference);
CREATE INDEX IF NOT EXISTS idx_circuit_npz_config_ref ON circuit_npz_datasets(config_reference);
CREATE INDEX IF NOT EXISTS idx_circuit_npz_filename ON circuit_npz_datasets(npz_filename);
CREATE INDEX IF NOT EXISTS idx_circuit_npz_available ON circuit_npz_datasets(is_available);
CREATE INDEX IF NOT EXISTS idx_circuit_npz_storage ON circuit_npz_datasets(storage_location);

-- Enable RLS
ALTER TABLE circuit_npz_datasets ENABLE ROW LEVEL SECURITY;

-- Basic RLS Policies (safe defaults)
CREATE POLICY "Users can view own NPZ datasets"
  ON circuit_npz_datasets FOR SELECT
  USING (user_reference = auth.uid());

CREATE POLICY "Users can create own NPZ datasets"
  ON circuit_npz_datasets FOR INSERT
  WITH CHECK (user_reference = auth.uid());

CREATE POLICY "Users can update own NPZ datasets"
  ON circuit_npz_datasets FOR UPDATE
  USING (user_reference = auth.uid());

CREATE POLICY "Users can delete own NPZ datasets"
  ON circuit_npz_datasets FOR DELETE
  USING (user_reference = auth.uid());

-- Optional: Allow public datasets (uncomment if needed)
-- CREATE POLICY "Public NPZ datasets viewable"
--   ON circuit_npz_datasets FOR SELECT
--   USING (circuit_parameters->>'is_public' = 'true');

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION update_circuit_npz_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_circuit_npz_updated_at
  BEFORE UPDATE ON circuit_npz_datasets
  FOR EACH ROW
  EXECUTE FUNCTION update_circuit_npz_updated_at();

-- Grant permissions
GRANT ALL ON circuit_npz_datasets TO authenticated;
GRANT SELECT ON circuit_npz_datasets TO anon;

-- Sample insert (shows how to use the table)
-- INSERT INTO circuit_npz_datasets (
--   user_reference,
--   npz_filename,
--   dataset_name,
--   grid_size,
--   n_parameters,
--   n_frequencies,
--   file_size_mb,
--   circuit_parameters,
--   computation_metadata
-- ) VALUES (
--   auth.uid(),
--   'sample_grid_5_test.npz',
--   'Sample Test Dataset',
--   5,
--   125,
--   50,
--   0.1,
--   '{"Rsh": 5000, "Ra": 3000, "Ca": 25e-6, "Rb": 4000, "Cb": 30e-6}'::jsonb,
--   '{"grid_size": 5, "freq_min": 0.1, "freq_max": 100000}'::jsonb
-- );