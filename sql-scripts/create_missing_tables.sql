-- Create missing tables for SpideyPlot
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

-- Grant permissions
GRANT ALL ON saved_configurations TO authenticated;
GRANT ALL ON user_profiles TO authenticated;
GRANT SELECT ON saved_configurations TO anon;
GRANT SELECT ON user_profiles TO anon;