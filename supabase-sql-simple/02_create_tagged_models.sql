-- 2. Tagged Models Table (lightweight - just references, not full data)
CREATE TABLE tagged_models (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id UUID REFERENCES user_sessions(id) ON DELETE CASCADE,
    
    -- Model identification (lightweight references)
    model_id VARCHAR(255) NOT NULL,
    tag_name VARCHAR(100) NOT NULL,
    tag_category VARCHAR(50) DEFAULT 'user',
    
    -- Just the essential parameters (not full computation data)
    circuit_parameters JSONB NOT NULL,
    resnorm_value DOUBLE PRECISION,
    
    -- Tagging context
    tagged_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    notes TEXT,
    
    -- Simple flag for local ML processing (no storage)
    is_interesting BOOLEAN DEFAULT false
);