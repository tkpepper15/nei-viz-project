-- USER PROFILES TABLE (User metadata only)
-- This table should store actual user profile information, not circuit configurations

CREATE TABLE IF NOT EXISTS public.user_profiles (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  
  -- User metadata
  username TEXT,
  full_name TEXT,
  avatar_url TEXT,
  
  -- User preferences for circuit simulator
  default_grid_size INTEGER DEFAULT 5 CHECK (default_grid_size BETWEEN 2 AND 25),
  default_min_freq NUMERIC DEFAULT 0.1 CHECK (default_min_freq > 0),
  default_max_freq NUMERIC DEFAULT 100000 CHECK (default_max_freq > default_min_freq),
  default_num_points INTEGER DEFAULT 100 CHECK (default_num_points BETWEEN 10 AND 1000),
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Constraints
  UNIQUE(user_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON public.user_profiles(user_id);

-- RLS Policies
ALTER TABLE public.user_profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view their own profile" ON public.user_profiles
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own profile" ON public.user_profiles
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own profile" ON public.user_profiles
  FOR UPDATE USING (auth.uid() = user_id);

-- Trigger for updated_at
CREATE TRIGGER update_user_profiles_updated_at
  BEFORE UPDATE ON public.user_profiles
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Comments
COMMENT ON TABLE public.user_profiles IS 'User profile metadata and circuit simulator preferences';
COMMENT ON COLUMN public.user_profiles.user_id IS 'References auth.users(id) - one profile per user';
COMMENT ON COLUMN public.user_profiles.username IS 'Display name for the user';
COMMENT ON COLUMN public.user_profiles.default_grid_size IS 'Default grid size for new circuit configurations';