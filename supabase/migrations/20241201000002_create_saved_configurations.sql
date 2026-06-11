-- Create saved_configurations table for circuit simulator profiles
CREATE TABLE saved_configurations (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  
  -- Profile metadata
  name TEXT NOT NULL,
  description TEXT,
  is_public BOOLEAN DEFAULT FALSE,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Grid computation settings
  grid_size INTEGER NOT NULL CHECK (grid_size BETWEEN 2 AND 25),
  min_frequency REAL NOT NULL CHECK (min_frequency > 0),
  max_frequency REAL NOT NULL CHECK (max_frequency > min_frequency),
  num_points INTEGER NOT NULL CHECK (num_points BETWEEN 10 AND 1000),
  
  -- Circuit parameters (stored as JSONB for flexibility)
  circuit_parameters JSONB NOT NULL,
  
  -- Computation status
  is_computed BOOLEAN DEFAULT FALSE,
  computation_time REAL, -- seconds
  total_points INTEGER,
  valid_points INTEGER,
  
  -- Constraints
  CONSTRAINT valid_frequency_range CHECK (max_frequency > min_frequency),
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
CREATE INDEX idx_user_configurations ON saved_configurations(user_id, created_at DESC);
CREATE INDEX idx_public_configurations ON saved_configurations(is_public, created_at DESC) WHERE is_public = TRUE;
CREATE INDEX idx_configuration_name ON saved_configurations(user_id, name);
CREATE INDEX idx_computed_configurations ON saved_configurations(user_id, is_computed, updated_at DESC);

-- Enable RLS
ALTER TABLE saved_configurations ENABLE ROW LEVEL SECURITY;

-- Create trigger for updated_at
CREATE TRIGGER update_saved_configurations_updated_at
  BEFORE UPDATE ON saved_configurations
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Add helpful comments
COMMENT ON TABLE saved_configurations IS 'Circuit simulator saved configurations with computation settings and parameters';
COMMENT ON COLUMN saved_configurations.circuit_parameters IS 'JSONB containing Rsh, Ra, Ca, Rb, Cb values and frequency_range';
COMMENT ON COLUMN saved_configurations.is_public IS 'Whether this configuration can be viewed by other users';
COMMENT ON COLUMN saved_configurations.computation_time IS 'Time taken to compute grid in seconds';
COMMENT ON COLUMN saved_configurations.total_points IS 'Total number of grid points computed';
COMMENT ON COLUMN saved_configurations.valid_points IS 'Number of valid computation results';