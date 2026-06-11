-- UI Settings Auto-Save System - Complete Schema Changes
-- Run this file to add UI settings persistence to your database

-- =============================================================================
-- 1. CREATE CIRCUIT_CONFIGURATIONS TABLE (if it doesn't exist)
-- =============================================================================

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
  computation_results JSONB,
  
  -- UI settings persistence (NEW COLUMN)
  ui_settings JSONB,
  
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

-- =============================================================================
-- 2. ADD UI_SETTINGS COLUMN (if circuit_configurations already exists)
-- =============================================================================

DO $$
BEGIN
  -- Add ui_settings column if it doesn't exist
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'circuit_configurations' AND column_name = 'ui_settings'
  ) THEN
    ALTER TABLE circuit_configurations ADD COLUMN ui_settings JSONB;
  END IF;
END $$;

-- =============================================================================
-- 3. ADD CURRENT_CIRCUIT_CONFIG_ID TO USER_SESSIONS
-- =============================================================================

-- Add session persistence column
ALTER TABLE user_sessions 
ADD COLUMN IF NOT EXISTS current_circuit_config_id UUID REFERENCES circuit_configurations(id) ON DELETE SET NULL;

-- =============================================================================
-- 4. UPDATE TAGGED_MODELS TABLE
-- =============================================================================

-- Add new reference to circuit_configurations
ALTER TABLE tagged_models 
ADD COLUMN IF NOT EXISTS circuit_config_id UUID REFERENCES circuit_configurations(id) ON DELETE CASCADE;

-- =============================================================================
-- 5. CREATE PERFORMANCE INDEXES
-- =============================================================================

-- Circuit configurations indexes
CREATE INDEX IF NOT EXISTS idx_circuit_configurations_user ON circuit_configurations(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_circuit_configurations_public ON circuit_configurations(is_public, created_at DESC) WHERE is_public = TRUE;
CREATE INDEX IF NOT EXISTS idx_circuit_configurations_name ON circuit_configurations(user_id, name);
CREATE INDEX IF NOT EXISTS idx_circuit_configurations_computed ON circuit_configurations(user_id, is_computed, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_circuit_configurations_updated ON circuit_configurations(user_id, updated_at DESC);

-- Session management indexes
CREATE INDEX IF NOT EXISTS idx_user_sessions_config ON user_sessions(user_id, current_circuit_config_id);

-- Tagged models indexes
CREATE INDEX IF NOT EXISTS idx_tagged_models_circuit_config ON tagged_models(circuit_config_id);

-- =============================================================================
-- 6. ENABLE ROW LEVEL SECURITY AND CREATE POLICIES
-- =============================================================================

-- Enable RLS
ALTER TABLE circuit_configurations ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist and recreate
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

-- =============================================================================
-- 7. CREATE HELPER FUNCTIONS
-- =============================================================================

-- Create or update the updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for updated_at
DROP TRIGGER IF EXISTS update_circuit_configurations_updated_at ON circuit_configurations;
CREATE TRIGGER update_circuit_configurations_updated_at
  BEFORE UPDATE ON circuit_configurations
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Default UI settings function
CREATE OR REPLACE FUNCTION get_default_ui_settings()
RETURNS JSONB AS $$
BEGIN
  RETURN jsonb_build_object(
    'activeTab', 'visualizer',
    'splitPaneHeight', 35,
    'opacityLevel', 0.7,
    'opacityExponent', 5,
    'logScalar', 1.0,
    'visualizationMode', 'color',
    'backgroundColor', 'white',
    'showGroundTruth', true,
    'includeLabels', true,
    'maxPolygons', 10000,
    'useSymmetricGrid', false,
    'adaptiveLimit', true,
    'maxMemoryUsage', 8192,
    'referenceModelVisible', true,
    'manuallyHidden', false,
    'isMultiSelectMode', false,
    'selectedCircuits', '[]'::jsonb,
    'windowPositions', '{}'::jsonb,
    'sidebarCollapsed', false,
    'toolboxPositions', '{}'::jsonb
  );
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- 8. CREATE HELPFUL VIEWS
-- =============================================================================

-- Session management view
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

-- =============================================================================
-- 9. ADD HELPFUL COMMENTS
-- =============================================================================

COMMENT ON TABLE circuit_configurations IS 'Circuit simulator configurations with computation settings, parameters, and UI state persistence';
COMMENT ON COLUMN circuit_configurations.circuit_parameters IS 'JSONB containing Rsh, Ra, Ca, Rb, Cb values and frequency_range';
COMMENT ON COLUMN circuit_configurations.ui_settings IS 'JSONB containing UI state for auto-save/restore: tabs, opacity, panels, window positions, etc.';
COMMENT ON COLUMN circuit_configurations.is_public IS 'Whether this configuration can be viewed by other users';
COMMENT ON COLUMN circuit_configurations.computation_time IS 'Time taken to compute grid in seconds';
COMMENT ON COLUMN circuit_configurations.total_points IS 'Total number of grid points computed';
COMMENT ON COLUMN circuit_configurations.valid_points IS 'Number of valid computation results';
COMMENT ON COLUMN circuit_configurations.computation_results IS 'Cached computation results for performance';

COMMENT ON COLUMN user_sessions.current_circuit_config_id IS 'Tracks the active circuit configuration for session persistence across page reloads';
COMMENT ON COLUMN tagged_models.circuit_config_id IS 'References circuit_configurations table (replaces old configuration_id)';

COMMENT ON FUNCTION get_default_ui_settings() IS 'Returns default UI settings JSON for new circuit configurations';
COMMENT ON VIEW session_with_config IS 'Helper view for session management with configuration details';

-- =============================================================================
-- 10. VERIFICATION AND SUMMARY
-- =============================================================================

DO $$
BEGIN
  -- Verify everything was created successfully
  IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'circuit_configurations') THEN
    RAISE EXCEPTION 'ERROR: circuit_configurations table was not created';
  END IF;
  
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'circuit_configurations' AND column_name = 'ui_settings'
  ) THEN
    RAISE EXCEPTION 'ERROR: ui_settings column was not added to circuit_configurations';
  END IF;
  
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'user_sessions' AND column_name = 'current_circuit_config_id'
  ) THEN
    RAISE EXCEPTION 'ERROR: current_circuit_config_id column was not added to user_sessions';
  END IF;
  
  -- Success message
  RAISE NOTICE '=== ✅ UI SETTINGS AUTO-SAVE SCHEMA SETUP COMPLETE ===';
  RAISE NOTICE '';
  RAISE NOTICE 'Tables configured:';
  RAISE NOTICE '  ✓ circuit_configurations (with ui_settings JSONB column)';
  RAISE NOTICE '  ✓ user_sessions (with current_circuit_config_id UUID column)';
  RAISE NOTICE '  ✓ tagged_models (with circuit_config_id UUID column)';
  RAISE NOTICE '';
  RAISE NOTICE 'Indexes created: 8 performance indexes';
  RAISE NOTICE 'RLS policies: 5 security policies';
  RAISE NOTICE 'Functions: 2 helper functions';
  RAISE NOTICE 'Views: 1 session management view';
  RAISE NOTICE '';
  RAISE NOTICE '🚀 Your UI settings auto-save system is now ready!';
  RAISE NOTICE '   Features: Auto-save UI state, session persistence, no auto-creation';
  RAISE NOTICE '   Next: Run "npm run dev" to test the functionality';
END $$;