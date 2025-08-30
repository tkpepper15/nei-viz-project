-- 4. Create RLS Policies
CREATE POLICY "Users can manage their own sessions" ON user_sessions
    FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can manage their own tagged models" ON tagged_models
    FOR ALL USING (auth.uid() = user_id);