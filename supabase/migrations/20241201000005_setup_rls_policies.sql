-- Row Level Security (RLS) Policies for saved_configurations

-- Users can view their own configurations
CREATE POLICY "Users can view own configurations" ON saved_configurations
  FOR SELECT USING (auth.uid() = user_id);

-- Users can insert their own configurations  
CREATE POLICY "Users can insert own configurations" ON saved_configurations
  FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Users can update their own configurations
CREATE POLICY "Users can update own configurations" ON saved_configurations
  FOR UPDATE USING (auth.uid() = user_id);

-- Users can delete their own configurations
CREATE POLICY "Users can delete own configurations" ON saved_configurations
  FOR DELETE USING (auth.uid() = user_id);

-- Public configurations are viewable by all authenticated users
CREATE POLICY "Public configurations are viewable" ON saved_configurations
  FOR SELECT USING (is_public = TRUE AND auth.uid() IS NOT NULL);

-- Shared configurations are viewable based on sharing permissions
CREATE POLICY "Shared configurations are viewable" ON saved_configurations
  FOR SELECT USING (
    auth.uid() IS NOT NULL AND
    EXISTS (
      SELECT 1 FROM shared_configurations sc
      WHERE sc.configuration_id = id 
        AND sc.shared_with = auth.uid()
        AND sc.is_active = TRUE
        AND (sc.expires_at IS NULL OR sc.expires_at > NOW())
    )
  );

-- Users with write permission can update shared configurations
CREATE POLICY "Shared configurations are editable" ON saved_configurations
  FOR UPDATE USING (
    auth.uid() IS NOT NULL AND
    EXISTS (
      SELECT 1 FROM shared_configurations sc
      WHERE sc.configuration_id = id 
        AND sc.shared_with = auth.uid()
        AND sc.permission_level IN ('write', 'admin')
        AND sc.is_active = TRUE
        AND (sc.expires_at IS NULL OR sc.expires_at > NOW())
    )
  );

-- RLS Policies for computation_results

-- Users can view computation results for their own configurations
CREATE POLICY "Users can view own computation results" ON computation_results
  FOR SELECT USING (
    EXISTS (
      SELECT 1 FROM saved_configurations sc
      WHERE sc.id = configuration_id AND sc.user_id = auth.uid()
    )
  );

-- Users can insert computation results for their own configurations
CREATE POLICY "Users can insert own computation results" ON computation_results
  FOR INSERT WITH CHECK (
    EXISTS (
      SELECT 1 FROM saved_configurations sc
      WHERE sc.id = configuration_id AND sc.user_id = auth.uid()
    )
  );

-- Users can update computation results for their own configurations
CREATE POLICY "Users can update own computation results" ON computation_results
  FOR UPDATE USING (
    EXISTS (
      SELECT 1 FROM saved_configurations sc
      WHERE sc.id = configuration_id AND sc.user_id = auth.uid()
    )
  );

-- Users can delete computation results for their own configurations
CREATE POLICY "Users can delete own computation results" ON computation_results
  FOR DELETE USING (
    EXISTS (
      SELECT 1 FROM saved_configurations sc
      WHERE sc.id = configuration_id AND sc.user_id = auth.uid()
    )
  );

-- Users can view computation results for shared configurations
CREATE POLICY "Shared computation results are viewable" ON computation_results
  FOR SELECT USING (
    auth.uid() IS NOT NULL AND
    EXISTS (
      SELECT 1 FROM saved_configurations sc
      JOIN shared_configurations shc ON sc.id = shc.configuration_id
      WHERE sc.id = configuration_id 
        AND shc.shared_with = auth.uid()
        AND shc.is_active = TRUE
        AND (shc.expires_at IS NULL OR shc.expires_at > NOW())
    )
  );

-- RLS Policies for shared_configurations

-- Users can view shares they created or received
CREATE POLICY "Users can view relevant shares" ON shared_configurations
  FOR SELECT USING (
    auth.uid() = shared_by OR 
    auth.uid() = shared_with OR
    EXISTS (
      SELECT 1 FROM saved_configurations sc
      WHERE sc.id = configuration_id AND sc.user_id = auth.uid()
    )
  );

-- Users can create shares for their own configurations
CREATE POLICY "Users can create shares for own configurations" ON shared_configurations
  FOR INSERT WITH CHECK (
    auth.uid() = shared_by AND
    EXISTS (
      SELECT 1 FROM saved_configurations sc
      WHERE sc.id = configuration_id AND sc.user_id = auth.uid()
    )
  );

-- Users can update shares they created or have admin permission on
CREATE POLICY "Users can update relevant shares" ON shared_configurations
  FOR UPDATE USING (
    auth.uid() = shared_by OR
    (auth.uid() = shared_with AND permission_level = 'admin') OR
    EXISTS (
      SELECT 1 FROM saved_configurations sc
      WHERE sc.id = configuration_id AND sc.user_id = auth.uid()
    )
  );

-- Users can delete shares they created or have admin permission on  
CREATE POLICY "Users can delete relevant shares" ON shared_configurations
  FOR DELETE USING (
    auth.uid() = shared_by OR
    (auth.uid() = shared_with AND permission_level = 'admin') OR
    EXISTS (
      SELECT 1 FROM saved_configurations sc
      WHERE sc.id = configuration_id AND sc.user_id = auth.uid()
    )
  );

-- Create helper functions for common queries

-- Function to get user's configurations (own + shared)
CREATE OR REPLACE FUNCTION get_user_configurations(user_uuid UUID DEFAULT auth.uid())
RETURNS TABLE (
  id UUID,
  name TEXT,
  description TEXT,
  created_at TIMESTAMPTZ,
  updated_at TIMESTAMPTZ,
  is_computed BOOLEAN,
  is_owner BOOLEAN,
  permission_level TEXT,
  owner_name TEXT
) 
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  RETURN QUERY
  -- Own configurations
  SELECT 
    sc.id,
    sc.name,
    sc.description,
    sc.created_at,
    sc.updated_at,
    sc.is_computed,
    TRUE as is_owner,
    'admin'::TEXT as permission_level,
    up.full_name as owner_name
  FROM saved_configurations sc
  JOIN user_profiles up ON sc.user_id = up.id
  WHERE sc.user_id = user_uuid
  
  UNION ALL
  
  -- Shared configurations
  SELECT 
    sc.id,
    sc.name,
    sc.description,
    sc.created_at,
    sc.updated_at,
    sc.is_computed,
    FALSE as is_owner,
    shc.permission_level,
    up.full_name as owner_name
  FROM saved_configurations sc
  JOIN shared_configurations shc ON sc.id = shc.configuration_id
  JOIN user_profiles up ON sc.user_id = up.id
  WHERE shc.shared_with = user_uuid 
    AND shc.is_active = TRUE
    AND (shc.expires_at IS NULL OR shc.expires_at > NOW())
  
  ORDER BY updated_at DESC;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission to authenticated users
GRANT EXECUTE ON FUNCTION get_user_configurations(UUID) TO authenticated;