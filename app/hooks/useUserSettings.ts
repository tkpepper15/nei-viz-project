'use client';

import { useState, useEffect, useCallback } from 'react';
import { supabase } from '../../lib/supabase';
import { Database } from '../../lib/database.types';
import { ExtendedPerformanceSettings } from '../components/circuit-simulator/types/gpuSettings';

export interface UserSettings {
  id: string;
  user_id: string;
  session_name?: string;
  performance_settings?: string | ExtendedPerformanceSettings;
  visualization_settings?: object;
  is_active?: boolean;
  updated_at?: string;
}

interface UseUserSettingsReturn {
  settings: UserSettings | null;
  isLoading: boolean;
  error: string | null;
  updateSettings: (newSettings: Partial<UserSettings>) => Promise<void>;
  createUserSession: (sessionName: string, settings?: Partial<UserSettings>) => Promise<string>;
  getUserSessions: () => Promise<UserSettings[]>;
  setActiveSession: (sessionId: string) => Promise<void>;
}

export function useUserSettings(userId?: string): UseUserSettingsReturn {
  const [settings, setSettings] = useState<UserSettings | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load user settings on mount
  useEffect(() => {
    if (!userId) {
      setIsLoading(false);
      setSettings(null);
      return;
    }

    loadUserSettings();
  }, [userId]);

  const loadUserSettings = useCallback(async () => {
    if (!userId) return;

    try {
      setIsLoading(true);
      setError(null);

      // Get the active session for this user
      const { data: activeSessions, error: sessionError } = await supabase
        .from('user_sessions')
        .select('*')
        .eq('user_id', userId)
        .eq('is_active', true)
        .order('last_accessed', { ascending: false })
        .limit(1);

      if (sessionError) {
        throw sessionError;
      }

      let activeSession = activeSessions?.[0];

      // If no active session exists, create a default one
      if (!activeSession) {
        const { data: newSession, error: createError } = await supabase
          .from('user_sessions')
          .insert({
            user_id: userId,
            session_name: 'Default Session',
            description: 'Auto-created default session',
            environment_variables: {},
            visualization_settings: {},
            performance_settings: {
              useSymmetricGrid: true,
              maxComputationResults: 5000,
              gpuAcceleration: {
                enabled: false,
                preferWebGPU: true,
                fallbackToCPU: true,
                maxBatchSize: 65536,
                deviceType: 'discrete',
                enableProfiling: false,
                memoryThreshold: 1024
              },
              cpuSettings: {
                maxWorkers: navigator.hardwareConcurrency || 4,
                chunkSize: 5000
              }
            },
            is_active: true,
            total_computations: 0,
            total_models_generated: 0,
            total_computation_time: '0 seconds'
          })
          .select()
          .single();

        if (createError) {
          throw createError;
        }

        activeSession = newSession;
      }

      setSettings({
        id: activeSession.id,
        user_id: activeSession.user_id,
        session_name: activeSession.session_name,
        performance_settings: activeSession.performance_settings as string | ExtendedPerformanceSettings,
        visualization_settings: activeSession.visualization_settings as object,
        is_active: activeSession.is_active,
        updated_at: activeSession.updated_at
      });

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load user settings';
      setError(errorMessage);
      console.error('Error loading user settings:', err);
    } finally {
      setIsLoading(false);
    }
  }, [userId]);

  const updateSettings = useCallback(async (newSettings: Partial<UserSettings>) => {
    if (!userId || !settings) {
      throw new Error('User not authenticated or settings not loaded');
    }

    try {
      setError(null);

      // Prepare the update data
      const updateData: Record<string, unknown> = {
        updated_at: new Date().toISOString(),
        last_accessed: new Date().toISOString()
      };

      // Handle performance_settings specially - ensure it's stored as JSON
      if (newSettings.performance_settings) {
        updateData.performance_settings = typeof newSettings.performance_settings === 'string'
          ? newSettings.performance_settings
          : JSON.stringify(newSettings.performance_settings);
      }

      // Add other settings
      if (newSettings.visualization_settings) {
        updateData.visualization_settings = newSettings.visualization_settings;
      }
      if (newSettings.session_name) {
        updateData.session_name = newSettings.session_name;
      }

      const { data, error: updateError } = await supabase
        .from('user_sessions')
        .update(updateData)
        .eq('id', settings.id)
        .eq('user_id', userId)
        .select()
        .single();

      if (updateError) {
        throw updateError;
      }

      setSettings({
        id: data.id,
        user_id: data.user_id,
        session_name: data.session_name,
        performance_settings: data.performance_settings as string | ExtendedPerformanceSettings,
        visualization_settings: data.visualization_settings as object,
        is_active: data.is_active,
        updated_at: data.updated_at
      });

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to update settings';
      setError(errorMessage);
      console.error('Error updating user settings:', err);
      throw err;
    }
  }, [userId, settings]);

  const createUserSession = useCallback(async (sessionName: string, sessionSettings?: Partial<UserSettings>): Promise<string> => {
    if (!userId) {
      throw new Error('User not authenticated');
    }

    try {
      setError(null);

      // Deactivate other sessions first
      await supabase
        .from('user_sessions')
        .update({ is_active: false })
        .eq('user_id', userId);

      const { data, error: createError } = await supabase
        .from('user_sessions')
        .insert([{
          user_id: userId,
          session_name: sessionName,
          description: sessionSettings?.session_name || `User session: ${sessionName}`,
          environment_variables: {},
          visualization_settings: sessionSettings?.visualization_settings || {},
          performance_settings: sessionSettings?.performance_settings || {
            useSymmetricGrid: true,
            maxComputationResults: 5000,
            gpuAcceleration: {
              enabled: false,
              preferWebGPU: true,
              fallbackToCPU: true,
              maxBatchSize: 65536,
              deviceType: 'discrete',
              enableProfiling: false,
              memoryThreshold: 1024
            },
            cpuSettings: {
              maxWorkers: navigator.hardwareConcurrency || 4,
              chunkSize: 5000
            }
          },
          is_active: true
        } as Database['public']['Tables']['user_sessions']['Insert']])
        .select()
        .single();

      if (createError) {
        throw createError;
      }

      setSettings({
        id: data.id,
        user_id: data.user_id,
        session_name: data.session_name,
        performance_settings: data.performance_settings as string | ExtendedPerformanceSettings,
        visualization_settings: data.visualization_settings as object,
        is_active: data.is_active,
        updated_at: data.updated_at
      });
      return data.id;

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to create session';
      setError(errorMessage);
      console.error('Error creating user session:', err);
      throw err;
    }
  }, [userId]);

  const getUserSessions = useCallback(async (): Promise<UserSettings[]> => {
    if (!userId) {
      return [];
    }

    try {
      const { data, error: fetchError } = await supabase
        .from('user_sessions')
        .select('*')
        .eq('user_id', userId)
        .order('last_accessed', { ascending: false });

      if (fetchError) {
        throw fetchError;
      }

      return (data || []) as UserSettings[];

    } catch (err) {
      console.error('Error fetching user sessions:', err);
      return [];
    }
  }, [userId]);

  const setActiveSession = useCallback(async (sessionId: string) => {
    if (!userId) {
      throw new Error('User not authenticated');
    }

    try {
      setError(null);

      // Deactivate all sessions
      await supabase
        .from('user_sessions')
        .update({ is_active: false })
        .eq('user_id', userId);

      // Activate the selected session
      const { data, error: updateError } = await supabase
        .from('user_sessions')
        .update({ 
          is_active: true,
          last_accessed: new Date().toISOString()
        })
        .eq('id', sessionId)
        .eq('user_id', userId)
        .select()
        .single();

      if (updateError) {
        throw updateError;
      }

      setSettings({
        id: data.id,
        user_id: data.user_id,
        session_name: data.session_name,
        performance_settings: data.performance_settings as string | ExtendedPerformanceSettings,
        visualization_settings: data.visualization_settings as object,
        is_active: data.is_active,
        updated_at: data.updated_at
      });

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to set active session';
      setError(errorMessage);
      console.error('Error setting active session:', err);
      throw err;
    }
  }, [userId]);

  return {
    settings,
    isLoading,
    error,
    updateSettings,
    createUserSession,
    getUserSessions,
    setActiveSession
  };
}