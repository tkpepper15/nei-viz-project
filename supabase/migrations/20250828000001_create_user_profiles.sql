-- NOTE: This migration was creating a conflicting user_profiles table.
-- The proper user_profiles table is created in 20241201000001_create_user_profiles.sql
-- Circuit configuration profiles are stored in saved_configurations table.
-- This migration is now disabled to prevent conflicts.

-- If you need circuit configuration profiles, use the saved_configurations table instead
-- which is properly defined in 20241201000002_create_saved_configurations.sql

-- Create updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE OR REPLACE TRIGGER update_user_profiles_updated_at
    BEFORE UPDATE ON user_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();