-- NPZ Dataset References Table
-- Links circuit configurations to their pre-computed NPZ files
-- Run this in your Supabase SQL editor

CREATE TABLE public.npz_dataset_references (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  configuration_id UUID NOT NULL REFERENCES saved_configurations(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES user_profiles(id) ON DELETE CASCADE,
  
  -- NPZ file information
  npz_filename VARCHAR(255) NOT NULL,
  grid_size INTEGER NOT NULL,
  n_parameters BIGINT NOT NULL,
  n_frequencies INTEGER NOT NULL,
  file_size_mb DECIMAL(10,2) NOT NULL DEFAULT 0,
  
  -- Storage and availability
  storage_type VARCHAR(10) NOT NULL DEFAULT 'local' CHECK (storage_type IN ('local', 'cloud', 'hybrid')),
  is_available BOOLEAN NOT NULL DEFAULT true,
  
  -- Metadata
  computation_metadata JSONB,
  
  -- Timestamps
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  last_checked TIMESTAMPTZ,
  
  -- Constraints
  UNIQUE(configuration_id), -- One NPZ per configuration
  UNIQUE(user_id, npz_filename) -- No duplicate filenames per user
);

-- Indexes for performance
CREATE INDEX idx_npz_references_user_id ON npz_dataset_references(user_id);
CREATE INDEX idx_npz_references_filename ON npz_dataset_references(npz_filename);
CREATE INDEX idx_npz_references_available ON npz_dataset_references(is_available);
CREATE INDEX idx_npz_references_storage_type ON npz_dataset_references(storage_type);

-- RLS Policies
ALTER TABLE npz_dataset_references ENABLE ROW LEVEL SECURITY;

-- Users can see their own NPZ references
CREATE POLICY "Users can view own NPZ references"
  ON npz_dataset_references FOR SELECT
  USING (user_id = auth.uid());

-- Users can insert their own NPZ references  
CREATE POLICY "Users can create own NPZ references"
  ON npz_dataset_references FOR INSERT
  WITH CHECK (user_id = auth.uid());

-- Users can update their own NPZ references
CREATE POLICY "Users can update own NPZ references"
  ON npz_dataset_references FOR UPDATE
  USING (user_id = auth.uid());

-- Users can delete their own NPZ references
CREATE POLICY "Users can delete own NPZ references"
  ON npz_dataset_references FOR DELETE
  USING (user_id = auth.uid());

-- Public NPZ references (linked to public configurations)
CREATE POLICY "Public NPZ references are viewable"
  ON npz_dataset_references FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM saved_configurations sc
      WHERE sc.id = configuration_id 
      AND sc.is_public = true
    )
  );

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_npz_references_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_npz_references_updated_at
  BEFORE UPDATE ON npz_dataset_references
  FOR EACH ROW
  EXECUTE FUNCTION update_npz_references_updated_at();

-- Grant permissions
GRANT ALL ON npz_dataset_references TO authenticated;
GRANT SELECT ON npz_dataset_references TO anon;