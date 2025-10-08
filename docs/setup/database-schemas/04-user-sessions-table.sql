-- USER SESSIONS TABLE
-- Sessions are temporary and used for activity tracking
-- Circuit configurations should persist BEYOND sessions

CREATE TABLE IF NOT EXISTS public.user_sessions (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  
  -- Session metadata
  session_name VARCHAR(255) NOT NULL,
  description TEXT,
  
  -- Current activity context (can be changed during session)
  current_circuit_config_id UUID REFERENCES public.circuit_configurations(id) ON DELETE SET NULL,
  
  -- Environment variables and settings (session-specific)
  environment_variables JSONB DEFAULT '{}'::jsonb,
  visualization_settings JSONB DEFAULT '{}'::jsonb,
  performance_settings JSONB DEFAULT '{}'::jsonb,
  
  -- Session status
  is_active BOOLEAN DEFAULT TRUE,
  last_accessed TIMESTAMPTZ DEFAULT NOW(),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Session statistics
  total_computations INTEGER DEFAULT 0,
  total_models_generated BIGINT DEFAULT 0,
  total_computation_time INTERVAL DEFAULT '0 seconds',
  
  -- Constraints
  UNIQUE(user_id, session_name)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON public.user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_active ON public.user_sessions(user_id, is_active, last_accessed DESC);
CREATE INDEX IF NOT EXISTS idx_user_sessions_current_config ON public.user_sessions(current_circuit_config_id) WHERE current_circuit_config_id IS NOT NULL;

-- Enable RLS
ALTER TABLE public.user_sessions ENABLE ROW LEVEL SECURITY;

-- RLS Policies  
CREATE POLICY "Users can view their own sessions" ON public.user_sessions
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own sessions" ON public.user_sessions
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own sessions" ON public.user_sessions
  FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own sessions" ON public.user_sessions
  FOR DELETE USING (auth.uid() = user_id);

-- Trigger for updated_at
CREATE TRIGGER update_user_sessions_updated_at
  BEFORE UPDATE ON public.user_sessions
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Helper function to get or create active session
CREATE OR REPLACE FUNCTION get_or_create_active_session(
  user_uuid UUID DEFAULT auth.uid(),
  session_name_param VARCHAR DEFAULT NULL
)
RETURNS TABLE (
  session_id UUID,
  session_name VARCHAR,
  current_circuit_config_id UUID,
  last_accessed TIMESTAMPTZ
) 
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  active_session RECORD;
  new_session_name VARCHAR;
BEGIN
  -- Try to find existing active session
  SELECT id, session_name, current_circuit_config_id, last_accessed
  INTO active_session
  FROM public.user_sessions
  WHERE user_id = user_uuid AND is_active = TRUE
  ORDER BY last_accessed DESC
  LIMIT 1;
  
  IF FOUND THEN
    -- Update last accessed time
    UPDATE public.user_sessions 
    SET last_accessed = NOW(), updated_at = NOW()
    WHERE id = active_session.id;
    
    RETURN QUERY SELECT 
      active_session.id,
      active_session.session_name,
      active_session.current_circuit_config_id,
      NOW()::TIMESTAMPTZ;
  ELSE
    -- Create new session
    new_session_name := COALESCE(session_name_param, 'Session ' || TO_CHAR(NOW(), 'YYYY-MM-DD HH24:MI'));
    
    INSERT INTO public.user_sessions (user_id, session_name)
    VALUES (user_uuid, new_session_name)
    RETURNING id, session_name, current_circuit_config_id, last_accessed
    INTO active_session;
    
    RETURN QUERY SELECT 
      active_session.id,
      active_session.session_name,
      active_session.current_circuit_config_id,
      active_session.last_accessed;
  END IF;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT EXECUTE ON FUNCTION get_or_create_active_session(UUID, VARCHAR) TO authenticated;

-- Comments
COMMENT ON TABLE public.user_sessions IS 'User sessions for activity tracking - circuit configurations persist beyond sessions';
COMMENT ON COLUMN public.user_sessions.current_circuit_config_id IS 'Currently active circuit configuration for this session';
COMMENT ON COLUMN public.user_sessions.is_active IS 'Whether this session is currently active';