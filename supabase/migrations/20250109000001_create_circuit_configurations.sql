-- Create circuit_configurations table (replaces saved_configurations)
-- This table stores circuit simulation configurations with UI settings persistence

CREATE TABLE IF NOT EXISTS circuit_configurations (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  
  -- Configuration metadata
  name TEXT NOT NULL,
  description TEXT,
  is_public BOOLEAN DEFAULT FALSE,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Circuit simulation parameters
  grid_size INTEGER NOT NULL CHECK (grid_size BETWEEN 2 AND 25),
  min_freq REAL NOT NULL CHECK (min_freq > 0),
  max_freq REAL NOT NULL CHECK (max_freq > min_freq),
  num_points INTEGER NOT NULL CHECK (num_points BETWEEN 10 AND 1000),
  
  -- Circuit parameters (JSONB for flexibility)
  circuit_parameters JSONB NOT NULL,
  
  -- Computation status and results
  is_computed BOOLEAN DEFAULT FALSE,
  computation_time REAL, -- seconds
  total_points INTEGER,
  valid_points INTEGER,
  computation_results JSONB, -- Store computation results if needed
  
  -- UI settings persistence (NEW)
  ui_settings JSONB, -- Store all UI state for auto-save/restore
  
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

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_circuit_configurations_user ON circuit_configurations(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_circuit_configurations_public ON circuit_configurations(is_public, created_at DESC) WHERE is_public = TRUE;
CREATE INDEX IF NOT EXISTS idx_circuit_configurations_name ON circuit_configurations(user_id, name);
CREATE INDEX IF NOT EXISTS idx_circuit_configurations_computed ON circuit_configurations(user_id, is_computed, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_circuit_configurations_updated ON circuit_configurations(user_id, updated_at DESC);

-- Enable RLS
ALTER TABLE circuit_configurations ENABLE ROW LEVEL SECURITY;

-- Create RLS policies
DROP POLICY IF EXISTS "Users can view their own circuit configurations" ON circuit_configurations;
CREATE POLICY "Users can view their own circuit configurations"
  ON circuit_configurations FOR SELECT
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can view public circuit configurations" ON circuit_configurations;
CREATE POLICY "Users can view public circuit configurations"
  ON circuit_configurations FOR SELECT
  USING (is_public = true);

DROP POLICY IF EXISTS "Users can create their own circuit configurations" ON circuit_configurations;
CREATE POLICY "Users can create their own circuit configurations"
  ON circuit_configurations FOR INSERT
  WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update their own circuit configurations" ON circuit_configurations;
CREATE POLICY "Users can update their own circuit configurations"
  ON circuit_configurations FOR UPDATE
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete their own circuit configurations" ON circuit_configurations;
CREATE POLICY "Users can delete their own circuit configurations"
  ON circuit_configurations FOR DELETE
  USING (auth.uid() = user_id);

-- Create trigger for updated_at (ensure function exists)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_proc WHERE proname = 'update_updated_at_column'
  ) THEN
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS '
    BEGIN
      NEW.updated_at = NOW();
      RETURN NEW;
    END;
    ' LANGUAGE plpgsql;
  END IF;
END $$;

DROP TRIGGER IF EXISTS update_circuit_configurations_updated_at ON circuit_configurations;
CREATE TRIGGER update_circuit_configurations_updated_at
  BEFORE UPDATE ON circuit_configurations
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Add helpful comments
COMMENT ON TABLE circuit_configurations IS 'Circuit simulator configurations with computation settings, parameters, and UI state persistence';
COMMENT ON COLUMN circuit_configurations.circuit_parameters IS 'JSONB containing Rsh, Ra, Ca, Rb, Cb values and frequency_range';
COMMENT ON COLUMN circuit_configurations.ui_settings IS 'JSONB containing UI state for auto-save/restore: tabs, opacity, panels, etc.';
COMMENT ON COLUMN circuit_configurations.is_public IS 'Whether this configuration can be viewed by other users';
COMMENT ON COLUMN circuit_configurations.computation_time IS 'Time taken to compute grid in seconds';
COMMENT ON COLUMN circuit_configurations.total_points IS 'Total number of grid points computed';
COMMENT ON COLUMN circuit_configurations.valid_points IS 'Number of valid computation results';
COMMENT ON COLUMN circuit_configurations.computation_results IS 'Cached computation results for performance';

-- Migration note: This table replaces the old saved_configurations table
-- The ui_settings column enables full UI state persistence for seamless user experience