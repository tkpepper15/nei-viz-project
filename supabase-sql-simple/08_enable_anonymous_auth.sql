-- 8. Enable anonymous authentication and create test data
-- This allows users to click "Enable Cloud Sessions" without email signup

-- Create a test user manually (run this in Supabase dashboard)
-- This simulates what happens when a user signs up
INSERT INTO auth.users (
  id,
  aud,
  role,
  email,
  encrypted_password,
  email_confirmed_at,
  created_at,
  updated_at,
  raw_app_meta_data,
  raw_user_meta_data,
  is_super_admin,
  confirmation_token,
  email_change_token_new,
  email_change,
  email_change_token_current,
  email_change_confirm_status,
  banned_until,
  phone,
  phone_confirmed_at,
  phone_change,
  phone_change_token,
  phone_change_sent_at,
  confirmed_at,
  email_change_sent_at,
  recovery_token,
  aud_claim,
  role_claim
) VALUES (
  '11111111-1111-1111-1111-111111111111'::uuid,
  'authenticated',
  'authenticated',
  'test@example.com',
  '$2a$10$example.hash.for.password.testing.only',
  NOW(),
  NOW(),
  NOW(),
  '{"provider": "email", "providers": ["email"]}',
  '{}',
  FALSE,
  '',
  '',
  '',
  '',
  0,
  NULL,
  NULL,
  NULL,
  '',
  '',
  NULL,
  NOW(),
  NULL,
  '',
  '',
  ''
) ON CONFLICT (id) DO NOTHING;

-- Create a test session for this user
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
  '22222222-2222-2222-2222-222222222222'::uuid,
  '11111111-1111-1111-1111-111111111111'::uuid,
  'Demo Session',
  'Sample session with demo data',
  '{
    "nodeOptions": "--max-old-space-size=8192",
    "computeWorkers": 4,
    "debugMode": false
  }'::jsonb,
  '{
    "groupPortion": 0.2,
    "selectedOpacityGroups": [0],
    "visualizationType": "spider3d",
    "chromaEnabled": true,
    "resnormSpread": 1.0,
    "useResnormCenter": false,
    "showLabels": true
  }'::jsonb,
  '{
    "maxWorkers": 4,
    "chunkSize": 1000,
    "memoryThreshold": 7000,
    "enableCaching": true,
    "adaptiveLimits": true
  }'::jsonb,
  '[
    {
      "id": "demo_profile_1",
      "name": "Healthy RPE",
      "parameters": {
        "Rsh": 1000,
        "Ra": 100,
        "Ca": 1e-6,
        "Rb": 500,
        "Cb": 2e-6
      },
      "frequencyRange": [0.1, 10000],
      "description": "Typical healthy RPE parameters",
      "createdAt": "2024-12-01T00:00:00Z"
    },
    {
      "id": "demo_profile_2", 
      "name": "Diseased RPE",
      "parameters": {
        "Rsh": 500,
        "Ra": 200,
        "Ca": 2e-6,
        "Rb": 800,
        "Cb": 3e-6
      },
      "frequencyRange": [0.1, 10000],
      "description": "Example diseased RPE parameters",
      "createdAt": "2024-12-01T00:00:00Z"
    }
  ]'::jsonb,
  TRUE
) ON CONFLICT (id) DO NOTHING;

-- Create sample tagged models
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
  '33333333-3333-3333-3333-333333333333'::uuid,
  '11111111-1111-1111-1111-111111111111'::uuid,
  '22222222-2222-2222-2222-222222222222'::uuid,
  'demo_model_optimal',
  'optimal',
  'user',
  '{
    "Rsh": 1200,
    "Ra": 120,
    "Ca": 1.2e-6,
    "Rb": 480,
    "Cb": 1.8e-6
  }'::jsonb,
  0.025,
  'Best fit model from parameter sweep',
  TRUE
), (
  '44444444-4444-4444-4444-444444444444'::uuid,
  '11111111-1111-1111-1111-111111111111'::uuid,
  '22222222-2222-2222-2222-222222222222'::uuid,
  'demo_model_interesting',
  'interesting',
  'user',
  '{
    "Rsh": 800,
    "Ra": 180,
    "Ca": 2.5e-6,
    "Rb": 600,
    "Cb": 2.2e-6
  }'::jsonb,
  0.045,
  'Unusual parameter combination worth investigating',
  TRUE
) ON CONFLICT (id) DO NOTHING;