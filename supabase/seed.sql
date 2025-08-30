-- Seed data for SpideyPlot Circuit Simulator
-- This file contains sample data for testing and development

-- Insert sample user profiles (these will be created automatically via triggers)
-- But we can insert additional profile data for existing test users

-- Sample public configurations for demonstration
-- Note: These will need real user_ids from auth.users after users are created

-- Sample circuit parameters for testing
INSERT INTO saved_configurations (
  user_id, name, description, is_public,
  grid_size, min_frequency, max_frequency, num_points,
  circuit_parameters, is_computed
) VALUES 
-- Example configuration 1: Basic RPE cell model
(
  '00000000-0000-0000-0000-000000000001'::UUID, -- Replace with real user_id
  'Basic RPE Cell Model',
  'Standard retinal pigment epithelium cell impedance model with typical parameter values',
  true,
  5, 0.1, 100000, 50,
  '{
    "Rsh": 1000,
    "Ra": 2000,
    "Ca": 0.000001,
    "Rb": 3000,
    "Cb": 0.000002,
    "frequency_range": [0.1, 100000]
  }',
  false
),
-- Example configuration 2: High capacitance model
(
  '00000000-0000-0000-0000-000000000001'::UUID, -- Replace with real user_id
  'High Capacitance RPE Model',
  'RPE cell model with elevated capacitance values for pathological conditions',
  true,
  4, 1, 50000, 40,
  '{
    "Rsh": 800,
    "Ra": 1500,
    "Ca": 0.000005,
    "Rb": 2500,
    "Cb": 0.000008,
    "frequency_range": [1, 50000]
  }',
  false
),
-- Example configuration 3: Low resistance model
(
  '00000000-0000-0000-0000-000000000001'::UUID, -- Replace with real user_id
  'Low Resistance RPE Model',
  'RPE cell model with reduced resistance values for increased permeability studies',
  true,
  3, 0.5, 10000, 30,
  '{
    "Rsh": 500,
    "Ra": 800,
    "Ca": 0.000003,
    "Rb": 1200,
    "Cb": 0.000004,
    "frequency_range": [0.5, 10000]
  }',
  true
),
-- Example configuration 4: Research template
(
  '00000000-0000-0000-0000-000000000002'::UUID, -- Replace with real user_id
  'Research Template - Tight Junctions',
  'Optimized for studying tight junction resistance in RPE monolayers',
  true,
  6, 0.01, 1000000, 100,
  '{
    "Rsh": 2000,
    "Ra": 4000,
    "Ca": 0.0000015,
    "Rb": 5000,
    "Cb": 0.0000025,
    "frequency_range": [0.01, 1000000]
  }',
  false
);

-- Sample computation results for the computed configuration
-- Note: This is a simplified example - real results would be much larger
INSERT INTO computation_results (
  configuration_id,
  grid_results,
  resnorm_groups,
  performance_metrics,
  computation_duration,
  grid_size,
  total_points,
  valid_points
) VALUES (
  (SELECT id FROM saved_configurations WHERE name = 'Low Resistance RPE Model' LIMIT 1),
  '[
    {
      "id": "result_001",
      "parameters": {"Rsh": 500, "Ra": 800, "Ca": 0.000003, "Rb": 1200, "Cb": 0.000004},
      "resnorm": 0.125,
      "impedanceSpectrum": [
        {"frequency": 0.5, "real": 1200, "imaginary": -800},
        {"frequency": 5, "real": 1100, "imaginary": -600},
        {"frequency": 50, "real": 950, "imaginary": -400}
      ]
    }
  ]',
  '[
    {
      "group": "excellent",
      "percentile": 25,
      "count": 7,
      "resnormRange": [0.1, 0.25]
    },
    {
      "group": "good", 
      "percentile": 50,
      "count": 12,
      "resnormRange": [0.25, 0.5]
    }
  ]',
  '{
    "totalComputeTime": 2.5,
    "averagePointTime": 0.093,
    "memoryUsage": "45MB",
    "workerCount": 4
  }',
  2.5,
  3,
  27,
  27
);

-- Sample sharing relationships
-- User 1 shares their public config with User 2
INSERT INTO shared_configurations (
  configuration_id,
  shared_with,
  shared_by,
  permission_level,
  share_message
) VALUES (
  (SELECT id FROM saved_configurations WHERE name = 'Basic RPE Cell Model' LIMIT 1),
  '00000000-0000-0000-0000-000000000002'::UUID, -- Replace with real user_id
  '00000000-0000-0000-0000-000000000001'::UUID, -- Replace with real user_id
  'read',
  'Here is the basic model we discussed in our research meeting'
);

-- Add some helpful comments
COMMENT ON TABLE saved_configurations IS 'This seed data provides sample circuit configurations for testing';

-- Create indexes that might be helpful for development queries
-- (These should also be in migration files for production)
CREATE INDEX IF NOT EXISTS idx_config_public_demo ON saved_configurations(is_public, name) WHERE is_public = true;

-- Note: To use this seed file:
-- 1. Replace the UUID placeholders with real user IDs from your auth.users table
-- 2. Run: supabase db reset (this will run migrations + seed)
-- 3. Or run: psql -h localhost -p 54322 -d postgres -U postgres -f supabase/seed.sql