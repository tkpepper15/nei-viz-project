-- Create missing tables for NPZ integration
-- Run this in your Supabase SQL editor

-- 1. Create saved_configurations table (if missing)
CREATE TABLE IF NOT EXISTS public.saved_configurations (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  
  -- Configuration details
  name VARCHAR(255) NOT NULL,
  description TEXT,
  
  -- Circuit parameters
  circuit_parameters JSONB NOT NULL,
  
  -- Settings
  is_public BOOLEAN NOT NULL DEFAULT false,
  is_computed BOOLEAN NOT NULL DEFAULT false,
  
  -- Timestamps
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create user_profiles table (if missing)
CREATE TABLE IF NOT EXISTS public.user_profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  
  -- Profile info
  username VARCHAR(50) UNIQUE,
  full_name VARCHAR(255),
  avatar_url TEXT,
  
  -- Timestamps
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_saved_configurations_user_id ON saved_configurations(user_id);
CREATE INDEX IF NOT EXISTS idx_saved_configurations_public ON saved_configurations(is_public) WHERE is_public = true;
CREATE INDEX IF NOT EXISTS idx_user_profiles_username ON user_profiles(username);

-- Enable RLS
ALTER TABLE saved_configurations ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;

-- RLS Policies for saved_configurations
CREATE POLICY "Users can view own configurations"
  ON saved_configurations FOR SELECT
  USING (user_id = auth.uid());

CREATE POLICY "Users can create own configurations"
  ON saved_configurations FOR INSERT
  WITH CHECK (user_id = auth.uid());

CREATE POLICY "Users can update own configurations"
  ON saved_configurations FOR UPDATE
  USING (user_id = auth.uid());

CREATE POLICY "Users can delete own configurations"
  ON saved_configurations FOR DELETE
  USING (user_id = auth.uid());

CREATE POLICY "Public configurations are viewable"
  ON saved_configurations FOR SELECT
  USING (is_public = true);

-- RLS Policies for user_profiles  
CREATE POLICY "Users can view own profile"
  ON user_profiles FOR SELECT
  USING (id = auth.uid());

CREATE POLICY "Users can update own profile"
  ON user_profiles FOR UPDATE
  USING (id = auth.uid());

CREATE POLICY "Users can insert own profile"
  ON user_profiles FOR INSERT
  WITH CHECK (id = auth.uid());

CREATE POLICY "Public profiles are viewable"
  ON user_profiles FOR SELECT
  USING (true); -- All profiles visible for usernames, etc.

-- Triggers for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_saved_configurations_updated_at
  BEFORE UPDATE ON saved_configurations
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_profiles_updated_at
  BEFORE UPDATE ON user_profiles
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- Now create the NPZ dataset references table
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

-- Indexes for NPZ references
CREATE INDEX idx_npz_references_user_id ON npz_dataset_references(user_id);
CREATE INDEX idx_npz_references_filename ON npz_dataset_references(npz_filename);
CREATE INDEX idx_npz_references_available ON npz_dataset_references(is_available);
CREATE INDEX idx_npz_references_storage_type ON npz_dataset_references(storage_type);

-- Enable RLS for NPZ references
ALTER TABLE npz_dataset_references ENABLE ROW LEVEL SECURITY;

-- NPZ RLS Policies
CREATE POLICY "Users can view own NPZ references"
  ON npz_dataset_references FOR SELECT
  USING (user_id = auth.uid());

CREATE POLICY "Users can create own NPZ references"
  ON npz_dataset_references FOR INSERT
  WITH CHECK (user_id = auth.uid());

CREATE POLICY "Users can update own NPZ references"
  ON npz_dataset_references FOR UPDATE
  USING (user_id = auth.uid());

CREATE POLICY "Users can delete own NPZ references"
  ON npz_dataset_references FOR DELETE
  USING (user_id = auth.uid());

CREATE POLICY "Public NPZ references are viewable"
  ON npz_dataset_references FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM saved_configurations sc
      WHERE sc.id = configuration_id 
      AND sc.is_public = true
    )
  );

-- NPZ updated_at trigger
CREATE TRIGGER update_npz_references_updated_at
  BEFORE UPDATE ON npz_dataset_references
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL ON saved_configurations TO authenticated;
GRANT ALL ON user_profiles TO authenticated;
GRANT ALL ON npz_dataset_references TO authenticated;
GRANT SELECT ON saved_configurations TO anon;
GRANT SELECT ON user_profiles TO anon;
GRANT SELECT ON npz_dataset_references TO anon;