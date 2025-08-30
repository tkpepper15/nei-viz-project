-- 5. Create indexes for performance
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_active ON user_sessions(user_id, is_active);
CREATE INDEX idx_tagged_models_user_id ON tagged_models(user_id);
CREATE INDEX idx_tagged_models_session_id ON tagged_models(session_id);