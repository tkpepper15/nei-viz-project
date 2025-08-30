-- 9. Create session for tejjas15@gmail.com user
-- This creates a session for your real user account

-- Create a session for the real user
INSERT INTO user_sessions (
  id,
  user_id,
  session_name,
  description,
  environment_variables,
  visualization_settings,
  performance_settings,
  saved_profiles,
  is_active
) VALUES (
  '44444444-4444-4444-4444-444444444444'::uuid,
  '36dfd52f-2e98-411f-adf6-e69bb619d0ae'::uuid, -- Your actual user ID
  'Tejjas Research Session',
  'Primary research session for NEI visualization',
  '{
    "nodeOptions": "--max-old-space-size=8192",
    "computeWorkers": 8,
    "debugMode": false,
    "memoryLimit": "8GB"
  }'::jsonb,
  '{
    "groupPortion": 0.25,
    "selectedOpacityGroups": [0, 1],
    "visualizationType": "spider3d",
    "chromaEnabled": true,
    "resnormSpread": 1.2,
    "useResnormCenter": false,
    "showLabels": true
  }'::jsonb,
  '{
    "maxWorkers": 8,
    "chunkSize": 1500,
    "memoryThreshold": 7000,
    "enableCaching": true,
    "adaptiveLimits": true
  }'::jsonb,
  '[
    {
      "id": "tejjas_profile_1",
      "name": "Standard RPE Model",
      "parameters": {
        "Rsh": 1200,
        "Ra": 150,
        "Ca": 1.2e-6,
        "Rb": 450,
        "Cb": 1.8e-6
      },
      "frequencyRange": [0.1, 10000],
      "description": "Standard RPE parameters for baseline comparison",
      "createdAt": "2024-12-01T00:00:00Z"
    },
    {
      "id": "tejjas_profile_2", 
      "name": "High Resistance Model",
      "parameters": {
        "Rsh": 2000,
        "Ra": 300,
        "Ca": 0.8e-6,
        "Rb": 800,
        "Cb": 1.2e-6
      },
      "frequencyRange": [0.1, 10000],
      "description": "High resistance variant for pathological conditions",
      "createdAt": "2024-12-01T00:00:00Z"
    },
    {
      "id": "tejjas_profile_3",
      "name": "Low Capacitance Model",
      "parameters": {
        "Rsh": 800,
        "Ra": 120,
        "Ca": 2.5e-6,
        "Rb": 350,
        "Cb": 3.0e-6
      },
      "frequencyRange": [0.1, 10000],
      "description": "Low capacitance model for membrane studies",
      "createdAt": "2024-12-01T00:00:00Z"
    }
  ]'::jsonb,
  TRUE
) ON CONFLICT (id) DO UPDATE SET
  user_id = EXCLUDED.user_id,
  session_name = EXCLUDED.session_name,
  description = EXCLUDED.description,
  environment_variables = EXCLUDED.environment_variables,
  visualization_settings = EXCLUDED.visualization_settings,
  performance_settings = EXCLUDED.performance_settings,
  saved_profiles = EXCLUDED.saved_profiles,
  is_active = EXCLUDED.is_active;

-- Create some sample tagged models for Tejjas
INSERT INTO tagged_models (
  id,
  user_id,
  session_id,
  model_id,
  tag_name,
  tag_category,
  circuit_parameters,
  resnorm_value,
  notes,
  is_interesting
) VALUES (
  '55555555-5555-5555-5555-555555555555'::uuid,
  '36dfd52f-2e98-411f-adf6-e69bb619d0ae'::uuid,
  '44444444-4444-4444-4444-444444444444'::uuid,
  'tejjas_model_optimal_1',
  'optimal',
  'research',
  '{
    "Rsh": 1350,
    "Ra": 180,
    "Ca": 1.1e-6,
    "Rb": 520,
    "Cb": 1.9e-6
  }'::jsonb,
  0.018,
  'Best fit from parameter sweep - excellent match to experimental data',
  TRUE
), (
  '66666666-6666-6666-6666-666666666666'::uuid,
  '36dfd52f-2e98-411f-adf6-e69bb619d0ae'::uuid,
  '44444444-4444-4444-4444-444444444444'::uuid,
  'tejjas_model_promising_1',
  'promising',
  'research',
  '{
    "Rsh": 950,
    "Ra": 220,
    "Ca": 2.1e-6,
    "Rb": 380,
    "Cb": 2.8e-6
  }'::jsonb,
  0.032,
  'Interesting parameter combination - worth exploring further',
  TRUE
), (
  '77777777-7777-7777-7777-777777777777'::uuid,
  '36dfd52f-2e98-411f-adf6-e69bb619d0ae'::uuid,
  '44444444-4444-4444-4444-444444444444'::uuid,
  'tejjas_model_baseline_1',
  'baseline',
  'reference',
  '{
    "Rsh": 1000,
    "Ra": 100,
    "Ca": 1.0e-6,
    "Rb": 500,
    "Cb": 2.0e-6
  }'::jsonb,
  0.055,
  'Standard baseline model for comparison',
  FALSE
) ON CONFLICT (id) DO UPDATE SET
  user_id = EXCLUDED.user_id,
  session_id = EXCLUDED.session_id,
  model_id = EXCLUDED.model_id,
  tag_name = EXCLUDED.tag_name,
  tag_category = EXCLUDED.tag_category,
  circuit_parameters = EXCLUDED.circuit_parameters,
  resnorm_value = EXCLUDED.resnorm_value,
  notes = EXCLUDED.notes,
  is_interesting = EXCLUDED.is_interesting;