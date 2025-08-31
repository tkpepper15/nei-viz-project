-- TAGGED MODELS TABLE
-- Links tagged models to specific circuit configurations
-- Tagged models should only appear for their associated circuit configuration

CREATE TABLE IF NOT EXISTS public.tagged_models (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  session_id UUID NULL REFERENCES public.user_sessions(id) ON DELETE CASCADE,
  
  -- CRITICAL: Link to specific circuit configuration
  circuit_config_id UUID NOT NULL REFERENCES public.circuit_configurations(id) ON DELETE CASCADE,
  
  -- Model identification
  model_id VARCHAR(255) NOT NULL, -- From ModelSnapshot.id
  tag_name VARCHAR(100) NOT NULL,
  tag_category VARCHAR(50) DEFAULT 'user'::varchar,
  
  -- Model parameters at time of tagging (snapshot)
  circuit_parameters JSONB NOT NULL,
  resnorm_value DOUBLE PRECISION,
  
  -- Tagging context
  tagged_at TIMESTAMPTZ DEFAULT NOW(),
  notes TEXT,
  is_interesting BOOLEAN DEFAULT FALSE,
  
  -- Ensure unique tagging per model per circuit configuration
  UNIQUE(circuit_config_id, model_id, tag_name)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_tagged_models_user_id ON public.tagged_models(user_id);
CREATE INDEX IF NOT EXISTS idx_tagged_models_session_id ON public.tagged_models(session_id);
CREATE INDEX IF NOT EXISTS idx_tagged_models_circuit_config ON public.tagged_models(circuit_config_id, tagged_at DESC);
CREATE INDEX IF NOT EXISTS idx_tagged_models_user_circuit ON public.tagged_models(user_id, circuit_config_id);
CREATE INDEX IF NOT EXISTS idx_tagged_models_resnorm ON public.tagged_models(circuit_config_id, resnorm_value);
CREATE INDEX IF NOT EXISTS idx_tagged_models_interesting ON public.tagged_models(circuit_config_id) WHERE is_interesting = TRUE;

-- Enable RLS
ALTER TABLE public.tagged_models ENABLE ROW LEVEL SECURITY;

-- RLS Policies
CREATE POLICY "Users can view tagged models for their own circuit configurations" ON public.tagged_models
  FOR SELECT USING (
    auth.uid() = user_id AND 
    EXISTS (
      SELECT 1 FROM public.circuit_configurations cc 
      WHERE cc.id = circuit_config_id AND cc.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can insert tagged models for their own circuit configurations" ON public.tagged_models
  FOR INSERT WITH CHECK (
    auth.uid() = user_id AND 
    EXISTS (
      SELECT 1 FROM public.circuit_configurations cc 
      WHERE cc.id = circuit_config_id AND cc.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can update their own tagged models" ON public.tagged_models
  FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own tagged models" ON public.tagged_models
  FOR DELETE USING (auth.uid() = user_id);

-- Comments
COMMENT ON TABLE public.tagged_models IS 'Tagged models linked to specific circuit configurations - only visible when that circuit is active';
COMMENT ON COLUMN public.tagged_models.circuit_config_id IS 'CRITICAL: Links tagged model to specific circuit configuration';
COMMENT ON COLUMN public.tagged_models.model_id IS 'Unique identifier from ModelSnapshot.id';
COMMENT ON COLUMN public.tagged_models.circuit_parameters IS 'Snapshot of circuit parameters at time of tagging';
COMMENT ON COLUMN public.tagged_models.session_id IS 'Optional: Session where model was tagged (can be null)';

-- Helper function to get tagged models for a specific circuit configuration
CREATE OR REPLACE FUNCTION get_tagged_models_for_circuit(
  config_id UUID,
  user_uuid UUID DEFAULT auth.uid()
)
RETURNS TABLE (
  id UUID,
  model_id VARCHAR,
  tag_name VARCHAR,
  tag_category VARCHAR,
  circuit_parameters JSONB,
  resnorm_value DOUBLE PRECISION,
  tagged_at TIMESTAMPTZ,
  notes TEXT,
  is_interesting BOOLEAN
) 
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  RETURN QUERY
  SELECT 
    tm.id,
    tm.model_id,
    tm.tag_name,
    tm.tag_category,
    tm.circuit_parameters,
    tm.resnorm_value,
    tm.tagged_at,
    tm.notes,
    tm.is_interesting
  FROM public.tagged_models tm
  JOIN public.circuit_configurations cc ON tm.circuit_config_id = cc.id
  WHERE tm.circuit_config_id = config_id 
    AND tm.user_id = user_uuid
    AND cc.user_id = user_uuid -- Double-check ownership
  ORDER BY tm.tagged_at DESC;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT EXECUTE ON FUNCTION get_tagged_models_for_circuit(UUID, UUID) TO authenticated;