-- 1. Session Management Table
CREATE TABLE user_sessions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    session_name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Environment variables and settings (what was stored in localStorage)
    environment_variables JSONB DEFAULT '{}'::jsonb,
    visualization_settings JSONB DEFAULT '{}'::jsonb,
    performance_settings JSONB DEFAULT '{}'::jsonb,
    
    -- Saved profiles (what was in localStorage)
    saved_profiles JSONB DEFAULT '[]'::jsonb,
    
    -- Session metadata
    is_active BOOLEAN DEFAULT true,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(user_id, session_name)
);