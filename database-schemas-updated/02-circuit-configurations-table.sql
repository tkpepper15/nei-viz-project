-- CIRCUIT CONFIGURATIONS TABLE 
-- This table stores saved circuit simulation configurations
-- These should persist across all sessions for the same user

CREATE TABLE IF NOT EXISTS public.circuit_configurations (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  
  -- Configuration metadata
  name TEXT NOT NULL,
  description TEXT,
  is_public BOOLEAN DEFAULT FALSE,
  
  -- Circuit parameters (the actual circuit configuration)
  circuit_parameters JSONB NOT NULL,
  
  -- Grid computation settings
  grid_size INTEGER NOT NULL DEFAULT 5 CHECK (grid_size BETWEEN 2 AND 25),
  min_freq NUMERIC NOT NULL DEFAULT 0.1 CHECK (min_freq > 0),
  max_freq NUMERIC NOT NULL DEFAULT 100000 CHECK (max_freq > min_freq),
  num_points INTEGER NOT NULL DEFAULT 100 CHECK (num_points BETWEEN 10 AND 1000),
  
  -- Computation status and results
  is_computed BOOLEAN DEFAULT FALSE,
  computation_time NUMERIC, -- seconds
  total_points INTEGER,
  valid_points INTEGER,
  computation_results JSONB, -- Store grid computation results
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Constraints
  CONSTRAINT valid_frequency_range CHECK (max_freq > min_freq),
  CONSTRAINT valid_circuit_params CHECK (
    circuit_parameters ? 'Rsh' AND
    circuit_parameters ? 'Ra' AND  
    circuit_parameters ? 'Ca' AND
    circuit_parameters ? 'Rb' AND
    circuit_parameters ? 'Cb' AND
    circuit_parameters ? 'frequency_range'
  )
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_circuit_configs_user_id ON public.circuit_configurations(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_circuit_configs_name ON public.circuit_configurations(user_id, name);
CREATE INDEX IF NOT EXISTS idx_circuit_configs_computed ON public.circuit_configurations(user_id, is_computed, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_circuit_configs_public ON public.circuit_configurations(is_public, created_at DESC) WHERE is_public = TRUE;

-- Enable RLS
ALTER TABLE public.circuit_configurations ENABLE ROW LEVEL SECURITY;

-- RLS Policies
CREATE POLICY "Users can view their own configurations" ON public.circuit_configurations
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own configurations" ON public.circuit_configurations
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own configurations" ON public.circuit_configurations
  FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own configurations" ON public.circuit_configurations
  FOR DELETE USING (auth.uid() = user_id);

CREATE POLICY "Public configurations are viewable" ON public.circuit_configurations
  FOR SELECT USING (is_public = TRUE AND auth.uid() IS NOT NULL);

-- Trigger for updated_at
CREATE TRIGGER update_circuit_configurations_updated_at
  BEFORE UPDATE ON public.circuit_configurations
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Comments
COMMENT ON TABLE public.circuit_configurations IS 'Saved circuit simulation configurations that persist across sessions';
COMMENT ON COLUMN public.circuit_configurations.circuit_parameters IS 'JSONB containing Rsh, Ra, Ca, Rb, Cb values and frequency_range';
COMMENT ON COLUMN public.circuit_configurations.is_public IS 'Whether this configuration can be viewed by other users';
COMMENT ON COLUMN public.circuit_configurations.computation_results IS 'Stored grid computation results for performance';
COMMENT ON COLUMN public.circuit_configurations.is_computed IS 'Whether the grid computation has been completed for this configuration';