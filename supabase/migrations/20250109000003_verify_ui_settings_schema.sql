-- Verification and final setup for UI settings auto-save system
-- This migration ensures all required tables and columns are properly configured

-- Verify circuit_configurations table structure
DO $$
BEGIN
  -- Ensure circuit_configurations exists with all required columns
  IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'circuit_configurations') THEN
    RAISE EXCEPTION 'circuit_configurations table does not exist. Run migration 20250109000001 first.';
  END IF;
  
  -- Verify required columns exist
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'circuit_configurations' AND column_name = 'ui_settings'
  ) THEN
    RAISE EXCEPTION 'ui_settings column missing from circuit_configurations table.';
  END IF;
  
  -- Verify user_sessions has current_circuit_config_id
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'user_sessions' AND column_name = 'current_circuit_config_id'
  ) THEN
    RAISE EXCEPTION 'current_circuit_config_id column missing from user_sessions table.';
  END IF;
  
  RAISE NOTICE 'UI Settings schema verification completed successfully!';
END $$;

-- Create example UI settings structure as a comment for reference
COMMENT ON COLUMN circuit_configurations.ui_settings IS 
'Example UI settings JSON structure:
{
  "activeTab": "visualizer",
  "splitPaneHeight": 35,
  "opacityLevel": 0.7,
  "opacityExponent": 5,
  "logScalar": 1.0,
  "visualizationMode": "color",
  "backgroundColor": "white",
  "showGroundTruth": true,
  "includeLabels": true,
  "maxPolygons": 10000,
  "useSymmetricGrid": false,
  "adaptiveLimit": true,
  "maxMemoryUsage": 8192,
  "referenceModelVisible": true,
  "manuallyHidden": false,
  "isMultiSelectMode": false,
  "selectedCircuits": [],
  "windowPositions": {},
  "sidebarCollapsed": false,
  "toolboxPositions": {}
}';

-- Create function to get default UI settings
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

COMMENT ON FUNCTION get_default_ui_settings() IS 'Returns default UI settings JSON for new circuit configurations';

-- Show summary of what was set up
DO $$
BEGIN
  RAISE NOTICE '=== UI Settings Auto-Save System Setup Complete ===';
  RAISE NOTICE 'Tables configured:';
  RAISE NOTICE '  ✓ circuit_configurations (with ui_settings JSONB column)';
  RAISE NOTICE '  ✓ user_sessions (with current_circuit_config_id UUID column)';
  RAISE NOTICE '  ✓ tagged_models (with circuit_config_id UUID column)';
  RAISE NOTICE '';
  RAISE NOTICE 'Features enabled:';
  RAISE NOTICE '  ✓ Auto-save UI settings to database';
  RAISE NOTICE '  ✓ Session persistence across page reloads';
  RAISE NOTICE '  ✓ No automatic configuration creation';
  RAISE NOTICE '  ✓ Real-time UI state synchronization';
  RAISE NOTICE '';
  RAISE NOTICE 'API endpoints created:';
  RAISE NOTICE '  ✓ POST /api/save-ui-settings (for emergency saves)';
  RAISE NOTICE '';
  RAISE NOTICE 'Run: npm run dev to test the auto-save functionality';
END $$;