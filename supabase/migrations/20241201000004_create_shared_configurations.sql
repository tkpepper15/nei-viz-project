-- Create shared_configurations table for collaboration features
CREATE TABLE shared_configurations (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  configuration_id UUID REFERENCES saved_configurations(id) ON DELETE CASCADE NOT NULL,
  shared_with UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  shared_by UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  
  -- Permission levels
  permission_level TEXT CHECK (permission_level IN ('read', 'write', 'admin')) DEFAULT 'read',
  
  -- Timestamps
  shared_at TIMESTAMPTZ DEFAULT NOW(),
  expires_at TIMESTAMPTZ, -- Optional expiration
  last_accessed TIMESTAMPTZ,
  
  -- Metadata
  share_message TEXT, -- Optional message from sharer
  is_active BOOLEAN DEFAULT TRUE,
  
  -- Ensure unique sharing relationships
  UNIQUE(configuration_id, shared_with)
);

-- Create indexes
CREATE INDEX idx_shared_with_user ON shared_configurations(shared_with, shared_at DESC);
CREATE INDEX idx_shared_by_user ON shared_configurations(shared_by, shared_at DESC);
CREATE INDEX idx_configuration_shares ON shared_configurations(configuration_id, is_active);
CREATE INDEX idx_active_shares ON shared_configurations(shared_with, is_active, expires_at) 
  WHERE is_active = TRUE;

-- Enable RLS  
ALTER TABLE shared_configurations ENABLE ROW LEVEL SECURITY;

-- Add helpful comments
COMMENT ON TABLE shared_configurations IS 'Collaboration and sharing relationships between users and configurations';
COMMENT ON COLUMN shared_configurations.permission_level IS 'Access level: read (view only), write (edit), admin (share/delete)';
COMMENT ON COLUMN shared_configurations.expires_at IS 'Optional expiration timestamp for temporary shares';
COMMENT ON COLUMN shared_configurations.last_accessed IS 'Track when shared configuration was last viewed';

-- Create function to handle share expiration
CREATE OR REPLACE FUNCTION check_share_expiration()
RETURNS TRIGGER AS $$
BEGIN
  -- Automatically deactivate expired shares
  IF NEW.expires_at IS NOT NULL AND NEW.expires_at < NOW() THEN
    NEW.is_active = FALSE;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for share expiration
CREATE TRIGGER handle_share_expiration
  BEFORE INSERT OR UPDATE ON shared_configurations
  FOR EACH ROW EXECUTE FUNCTION check_share_expiration();

-- Create function to update last_accessed
CREATE OR REPLACE FUNCTION update_share_access()
RETURNS TRIGGER AS $$
BEGIN
  UPDATE shared_configurations 
  SET last_accessed = NOW()
  WHERE configuration_id = NEW.id 
    AND shared_with = auth.uid()
    AND is_active = TRUE;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create view for easy access to shared configurations with owner info
CREATE VIEW shared_configurations_with_details AS
SELECT 
  sc.*,
  c.name as configuration_name,
  c.description as configuration_description,
  c.is_computed,
  c.created_at as configuration_created_at,
  owner.full_name as owner_name,
  owner.username as owner_username,
  shared_by_user.full_name as shared_by_name,
  shared_by_user.username as shared_by_username
FROM shared_configurations sc
JOIN saved_configurations c ON sc.configuration_id = c.id
JOIN user_profiles owner ON c.user_id = owner.id  
JOIN user_profiles shared_by_user ON sc.shared_by = shared_by_user.id
WHERE sc.is_active = TRUE 
  AND (sc.expires_at IS NULL OR sc.expires_at > NOW());

-- Grant access to the view
GRANT SELECT ON shared_configurations_with_details TO authenticated;