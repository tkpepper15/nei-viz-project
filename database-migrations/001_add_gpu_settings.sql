-- Migration: Add GPU acceleration settings to existing user_sessions table
-- This extends the performance_settings JSON field to include WebGPU options

-- Add RLS policy for user access to their GPU settings
CREATE POLICY "Users can manage their own GPU settings" ON user_sessions
  FOR ALL USING (auth.uid() = user_id::uuid);

-- Create index for efficient settings queries
CREATE INDEX IF NOT EXISTS idx_user_sessions_performance_settings 
  ON user_sessions USING gin (performance_settings);

-- Add comment explaining the extended performance_settings structure
COMMENT ON COLUMN user_sessions.performance_settings IS 
'JSON containing performance settings including: 
{
  "useSymmetricGrid": boolean,
  "maxComputationResults": number,
  "gpuAcceleration": {
    "enabled": boolean,
    "preferWebGPU": boolean,
    "fallbackToCPU": boolean,
    "maxBatchSize": number,
    "deviceType": "discrete" | "integrated" | "cpu",
    "enableProfiling": boolean,
    "memoryThreshold": number
  },
  "cpuSettings": {
    "maxWorkers": number,
    "chunkSize": number
  }
}';