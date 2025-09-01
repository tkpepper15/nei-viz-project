-- Add UI settings support and session configuration tracking
-- This migration adds the missing fields for the auto-save UI system

-- Add current_circuit_config_id to user_sessions for session persistence
ALTER TABLE user_sessions 
ADD COLUMN IF NOT EXISTS current_circuit_config_id UUID REFERENCES circuit_configurations(id) ON DELETE SET NULL;

-- Add index for performance
CREATE INDEX IF NOT EXISTS idx_user_sessions_config ON user_sessions(user_id, current_circuit_config_id);

-- If circuit_configurations table already exists, add ui_settings column
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'circuit_configurations') THEN
    -- Add ui_settings column if it doesn't exist
    IF NOT EXISTS (
      SELECT 1 FROM information_schema.columns 
      WHERE table_name = 'circuit_configurations' AND column_name = 'ui_settings'
    ) THEN
      ALTER TABLE circuit_configurations ADD COLUMN ui_settings JSONB;
      COMMENT ON COLUMN circuit_configurations.ui_settings IS 'JSONB containing UI state for auto-save/restore: tabs, opacity, panels, etc.';
    END IF;
  END IF;
END $$;

-- Update tagged_models table to reference circuit_configurations instead of saved_configurations
-- First, check if the constraint exists and handle the migration
DO $$
BEGIN
  -- Check if tagged_models exists and has old reference
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'tagged_models') THEN
    -- Drop old constraint if it exists
    IF EXISTS (
      SELECT 1 FROM information_schema.table_constraints 
      WHERE table_name = 'tagged_models' AND constraint_name LIKE '%configuration_id%'
    ) THEN
      -- Add new column for circuit_configurations
      ALTER TABLE tagged_models 
      ADD COLUMN IF NOT EXISTS circuit_config_id UUID REFERENCES circuit_configurations(id) ON DELETE CASCADE;
      
      -- Create index for performance
      CREATE INDEX IF NOT EXISTS idx_tagged_models_circuit_config ON tagged_models(circuit_config_id);
      
      -- Add comment
      COMMENT ON COLUMN tagged_models.circuit_config_id IS 'References circuit_configurations table (replaces configuration_id)';
    END IF;
  END IF;
END $$;

-- Add helpful comments for the new session management feature
COMMENT ON COLUMN user_sessions.current_circuit_config_id IS 'Tracks the active circuit configuration for session persistence across page reloads';

-- Create a helpful view for session management with configuration details
CREATE OR REPLACE VIEW session_with_config AS
SELECT 
  s.id as session_id,
  s.user_id,
  s.session_name,
  s.current_circuit_config_id,
  c.name as config_name,
  c.updated_at as config_last_modified,
  s.last_accessed,
  s.is_active
FROM user_sessions s
LEFT JOIN circuit_configurations c ON s.current_circuit_config_id = c.id;

COMMENT ON VIEW session_with_config IS 'Helper view for session management with configuration details';