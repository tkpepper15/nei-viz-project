-- Create computation_results table for caching large computation datasets
-- This is optional but helps with performance for large grid computations
CREATE TABLE computation_results (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  configuration_id UUID REFERENCES saved_configurations(id) ON DELETE CASCADE NOT NULL,
  
  -- Results data (compressed JSONB for large datasets)
  grid_results JSONB NOT NULL,
  resnorm_groups JSONB,
  grid_parameter_arrays JSONB,
  performance_metrics JSONB,
  
  -- Metadata
  computed_at TIMESTAMPTZ DEFAULT NOW(),
  computation_duration REAL NOT NULL, -- seconds
  result_size_bytes INTEGER, -- for monitoring storage usage
  
  -- Optimization hints
  grid_size INTEGER NOT NULL,
  total_points INTEGER NOT NULL,
  valid_points INTEGER NOT NULL
);

-- Create indexes
CREATE INDEX idx_configuration_results ON computation_results(configuration_id);
CREATE INDEX idx_computation_date ON computation_results(computed_at DESC);
CREATE INDEX idx_computation_performance ON computation_results(grid_size, computation_duration);

-- Enable RLS
ALTER TABLE computation_results ENABLE ROW LEVEL SECURITY;

-- Add constraints
ALTER TABLE computation_results ADD CONSTRAINT valid_computation_duration 
  CHECK (computation_duration > 0);
ALTER TABLE computation_results ADD CONSTRAINT valid_points_count 
  CHECK (valid_points <= total_points);

-- Add helpful comments
COMMENT ON TABLE computation_results IS 'Cached computation results for large grid calculations';
COMMENT ON COLUMN computation_results.grid_results IS 'Complete grid computation results as JSONB';
COMMENT ON COLUMN computation_results.resnorm_groups IS 'Grouped results by resnorm percentiles';
COMMENT ON COLUMN computation_results.performance_metrics IS 'Computation performance data and timing breakdown';
COMMENT ON COLUMN computation_results.result_size_bytes IS 'Storage size for monitoring and optimization';

-- Create function to automatically update result size
CREATE OR REPLACE FUNCTION calculate_result_size()
RETURNS TRIGGER AS $$
BEGIN
  NEW.result_size_bytes = octet_length(NEW.grid_results::text) + 
                         COALESCE(octet_length(NEW.resnorm_groups::text), 0) +
                         COALESCE(octet_length(NEW.grid_parameter_arrays::text), 0) +
                         COALESCE(octet_length(NEW.performance_metrics::text), 0);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to calculate result size
CREATE TRIGGER calculate_computation_result_size
  BEFORE INSERT OR UPDATE ON computation_results
  FOR EACH ROW EXECUTE FUNCTION calculate_result_size();