'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { supabase } from '../../lib/supabase';
import { Database } from '../../lib/database.types';
import { ExtendedPerformanceSettings } from '../components/circuit-simulator/types/gpuSettings';

export interface EnhancedUserSettings {
  id: string;
  user_id: string;
  session_name?: string;
  performance_settings?: ExtendedPerformanceSettings;
  visualization_settings?: {
    gridSize?: number;
    defaultGridSize?: number;
    visualizationType?: string;
    activeTab?: string;
  };
  circuit_parameters?: {
    Rsh?: number;
    Ra?: number;
    Ca?: number;
    Rb?: number;
    Cb?: number;
    frequency_range?: [number, number];
  };
  computation_settings?: {
    minFreq?: number;
    maxFreq?: number;
    numPoints?: number;
    maxComputationResults?: number;
    useSymmetricGrid?: boolean;
  };
  is_active?: boolean;
  updated_at?: string;
}

interface UseEnhancedUserSettingsReturn {
  settings: EnhancedUserSettings | null;
  isLoading: boolean;
  error: string | null;
  updateSettings: (newSettings: Partial<EnhancedUserSettings>) => Promise<void>;
  updateGridSize: (gridSize: number) => Promise<void>;
  updateComputationSettings: (settings: Partial<EnhancedUserSettings['computation_settings']>) => Promise<void>;
  createUserSession: (sessionName: string, settings?: Partial<EnhancedUserSettings>) => Promise<string>;
  getUserSessions: () => Promise<EnhancedUserSettings[]>;
  setActiveSession: (sessionId: string) => Promise<void>;
  getGridSize: () => number;
  isReady: boolean;
}

const DEFAULT_GRID_SIZE = 9;
const DEFAULT_COMPUTATION_SETTINGS = {
  minFreq: 0.1,
  maxFreq: 100000,
  numPoints: 100,
  maxComputationResults: 5000,
  useSymmetricGrid: true,
};

export function useEnhancedUserSettings(userId?: string): UseEnhancedUserSettingsReturn {
  const [settings, setSettings] = useState<EnhancedUserSettings | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isReady, setIsReady] = useState(false);

  // Debounce mechanism to prevent excessive Supabase calls
  const updateTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastUpdateRef = useRef<number>(0);

  // Load user settings on mount
  useEffect(() => {
    if (!userId) {
      setIsLoading(false);
      setSettings(null);
      setIsReady(true);
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
        const defaultPerformanceSettings: ExtendedPerformanceSettings = {
          useSymmetricGrid: true,
          maxComputationResults: 5000,
          gpuAcceleration: {
            enabled: false,
            preferWebGPU: true,
            fallbackToCPU: true,
            maxBatchSize: 65536,
            deviceType: 'discrete',
            enableProfiling: false,
            memoryThreshold: 1024,
            workgroupSize: 64,
            enableDebugMode: false,
            powerPreference: 'default',
            adaptiveThresholds: true
          },
          cpuSettings: {
            maxWorkers: navigator.hardwareConcurrency || 4,
            chunkSize: 5000
          }
        };

        const { data: newSession, error: createError } = await supabase
          .from('user_sessions')
          .insert({
            user_id: userId,
            session_name: 'Default Session',
            description: 'Auto-created default session',
            environment_variables: {},
            visualization_settings: {
              gridSize: DEFAULT_GRID_SIZE,
              defaultGridSize: DEFAULT_GRID_SIZE,
              visualizationType: 'spider3d',
              activeTab: 'visualizer'
            },
            performance_settings: defaultPerformanceSettings as unknown as Database['public']['Tables']['user_sessions']['Insert']['performance_settings'],
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

      // Parse and structure settings
      const performanceSettings = typeof activeSession.performance_settings === 'string'
        ? JSON.parse(activeSession.performance_settings)
        : (activeSession.performance_settings as unknown as ExtendedPerformanceSettings);

      const visualizationSettings = (activeSession.visualization_settings as Record<string, unknown>) || {};

      setSettings({
        id: activeSession.id,
        user_id: activeSession.user_id,
        session_name: activeSession.session_name,
        performance_settings: performanceSettings,
        visualization_settings: {
          gridSize: (visualizationSettings.gridSize as number) || DEFAULT_GRID_SIZE,
          defaultGridSize: (visualizationSettings.defaultGridSize as number) || DEFAULT_GRID_SIZE,
          visualizationType: (visualizationSettings.visualizationType as string) || 'spider3d',
          activeTab: (visualizationSettings.activeTab as string) || 'visualizer',
          ...visualizationSettings
        },
        computation_settings: {
          ...DEFAULT_COMPUTATION_SETTINGS,
          ...(visualizationSettings.computation_settings || {})
        },
        is_active: activeSession.is_active,
        updated_at: activeSession.updated_at
      });

      console.log('✅ User settings loaded:', {
        gridSize: visualizationSettings.gridSize || DEFAULT_GRID_SIZE,
        sessionName: activeSession.session_name
      });

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load user settings';
      setError(errorMessage);
      console.error('❌ Error loading user settings:', err);
    } finally {
      setIsLoading(false);
      setIsReady(true);
    }
  }, [userId]);

  // Debounced update function to prevent excessive Supabase calls
  const debouncedUpdateSupabase = useCallback(async (newSettings: Partial<EnhancedUserSettings>) => {
    if (!userId || !settings) return;

    const now = Date.now();
    const timeSinceLastUpdate = now - lastUpdateRef.current;

    // Prevent updates if less than 2 seconds since last update
    if (timeSinceLastUpdate < 2000) {
      if (updateTimeoutRef.current) {
        clearTimeout(updateTimeoutRef.current);
      }

      updateTimeoutRef.current = setTimeout(() => {
        debouncedUpdateSupabase(newSettings);
      }, 2000 - timeSinceLastUpdate);
      return;
    }

    try {
      lastUpdateRef.current = now;

      // Prepare the update data
      const updateData: Record<string, unknown> = {
        updated_at: new Date().toISOString(),
        last_accessed: new Date().toISOString()
      };

      // Handle performance_settings
      if (newSettings.performance_settings) {
        updateData.performance_settings = newSettings.performance_settings;
      }

      // Handle visualization_settings
      if (newSettings.visualization_settings || newSettings.computation_settings) {
        const currentVizSettings = settings.visualization_settings || {};
        const currentCompSettings = settings.computation_settings || {};

        updateData.visualization_settings = {
          ...currentVizSettings,
          ...newSettings.visualization_settings,
          computation_settings: {
            ...currentCompSettings,
            ...newSettings.computation_settings
          }
        };
      }

      const { error: updateError } = await supabase
        .from('user_sessions')
        .update(updateData)
        .eq('id', settings.id)
        .eq('user_id', userId);

      if (updateError) {
        throw updateError;
      }

      console.log('💾 Settings saved to Supabase');

    } catch (err) {
      console.error('❌ Error updating settings in Supabase:', err);
    }
  }, [userId, settings]);

  const updateSettings = useCallback(async (newSettings: Partial<EnhancedUserSettings>) => {
    if (!settings) return;

    try {
      setError(null);

      // Update local state immediately for responsive UI
      setSettings(prev => ({
        ...prev!,
        ...newSettings,
        visualization_settings: {
          ...prev!.visualization_settings,
          ...newSettings.visualization_settings
        },
        computation_settings: {
          ...prev!.computation_settings,
          ...newSettings.computation_settings
        }
      }));

      // Debounce Supabase update
      debouncedUpdateSupabase(newSettings);

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to update settings';
      setError(errorMessage);
      console.error('❌ Error updating user settings:', err);
      throw err;
    }
  }, [settings, debouncedUpdateSupabase]);

  const updateGridSize = useCallback(async (gridSize: number) => {
    console.log(`🔧 Updating grid size to: ${gridSize}`);

    await updateSettings({
      visualization_settings: {
        gridSize,
        defaultGridSize: gridSize
      }
    });
  }, [updateSettings]);

  const updateComputationSettings = useCallback(async (computationSettings: Partial<EnhancedUserSettings['computation_settings']>) => {
    console.log('🔧 Updating computation settings:', computationSettings);

    await updateSettings({
      computation_settings: computationSettings
    });
  }, [updateSettings]);

  const getGridSize = useCallback((): number => {
    return settings?.visualization_settings?.gridSize || DEFAULT_GRID_SIZE;
  }, [settings]);

  const createUserSession = useCallback(async (sessionName: string, sessionSettings?: Partial<EnhancedUserSettings>): Promise<string> => {
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

      const defaultPerformanceSettings: ExtendedPerformanceSettings = {
        useSymmetricGrid: true,
        maxComputationResults: 5000,
        gpuAcceleration: {
          enabled: false,
          preferWebGPU: true,
          fallbackToCPU: true,
          maxBatchSize: 65536,
          deviceType: 'discrete',
          enableProfiling: false,
          memoryThreshold: 1024,
          workgroupSize: 64,
          enableDebugMode: false,
          powerPreference: 'default',
          adaptiveThresholds: true
        },
        cpuSettings: {
          maxWorkers: navigator.hardwareConcurrency || 4,
          chunkSize: 5000
        }
      };

      const { data, error: createError } = await supabase
        .from('user_sessions')
        .insert([{
          user_id: userId,
          session_name: sessionName,
          description: sessionSettings?.session_name || `User session: ${sessionName}`,
          environment_variables: {},
          visualization_settings: {
            gridSize: sessionSettings?.visualization_settings?.gridSize || DEFAULT_GRID_SIZE,
            defaultGridSize: DEFAULT_GRID_SIZE,
            visualizationType: 'spider3d',
            activeTab: 'visualizer',
            ...sessionSettings?.visualization_settings
          },
          performance_settings: (sessionSettings?.performance_settings || defaultPerformanceSettings) as unknown as Database['public']['Tables']['user_sessions']['Insert']['performance_settings'],
          is_active: true
        } as Database['public']['Tables']['user_sessions']['Insert']])
        .select()
        .single();

      if (createError) {
        throw createError;
      }

      // Update local state
      setSettings({
        id: data.id,
        user_id: data.user_id,
        session_name: data.session_name,
        performance_settings: data.performance_settings as unknown as ExtendedPerformanceSettings,
        visualization_settings: data.visualization_settings as Record<string, unknown>,
        computation_settings: DEFAULT_COMPUTATION_SETTINGS,
        is_active: data.is_active,
        updated_at: data.updated_at
      });

      return data.id;

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to create session';
      setError(errorMessage);
      console.error('❌ Error creating user session:', err);
      throw err;
    }
  }, [userId]);

  const getUserSessions = useCallback(async (): Promise<EnhancedUserSettings[]> => {
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

      return (data || []).map(session => ({
        id: session.id,
        user_id: session.user_id,
        session_name: session.session_name,
        performance_settings: session.performance_settings as unknown as ExtendedPerformanceSettings,
        visualization_settings: session.visualization_settings as Record<string, unknown>,
        computation_settings: DEFAULT_COMPUTATION_SETTINGS,
        is_active: session.is_active,
        updated_at: session.updated_at
      }));

    } catch (err) {
      console.error('❌ Error fetching user sessions:', err);
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

      // Update local state
      setSettings({
        id: data.id,
        user_id: data.user_id,
        session_name: data.session_name,
        performance_settings: data.performance_settings as unknown as ExtendedPerformanceSettings,
        visualization_settings: data.visualization_settings as Record<string, unknown>,
        computation_settings: DEFAULT_COMPUTATION_SETTINGS,
        is_active: data.is_active,
        updated_at: data.updated_at
      });

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to set active session';
      setError(errorMessage);
      console.error('❌ Error setting active session:', err);
      throw err;
    }
  }, [userId]);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (updateTimeoutRef.current) {
        clearTimeout(updateTimeoutRef.current);
      }
    };
  }, []);

  return {
    settings,
    isLoading,
    error,
    updateSettings,
    updateGridSize,
    updateComputationSettings,
    createUserSession,
    getUserSessions,
    setActiveSession,
    getGridSize,
    isReady
  };
}