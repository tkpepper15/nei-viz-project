'use client';

import { useState, useEffect, useCallback } from 'react';
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

const SETTINGS_KEY = 'nei-viz-enhanced-settings';
const DEFAULT_GRID_SIZE = 9;

function loadSettings(): EnhancedUserSettings {
  if (typeof window === 'undefined') {
    return { id: 'local-session', user_id: 'local-user', is_active: true };
  }
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    return raw ? JSON.parse(raw) : { id: 'local-session', user_id: 'local-user', is_active: true };
  } catch {
    return { id: 'local-session', user_id: 'local-user', is_active: true };
  }
}

function persistSettings(s: EnhancedUserSettings): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(SETTINGS_KEY, JSON.stringify({ ...s, updated_at: new Date().toISOString() }));
}

export function useEnhancedUserSettings(_userId?: string) {
  const [settings, setSettings] = useState<EnhancedUserSettings | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    setSettings(loadSettings());
    setIsLoading(false);
  }, []);

  const updateSettings = useCallback(async (updates: Partial<EnhancedUserSettings>) => {
    setSettings(prev => {
      const next = { ...prev!, ...updates };
      persistSettings(next);
      return next;
    });
  }, []);

  const updateGridSize = useCallback(async (gridSize: number) => {
    await updateSettings({
      visualization_settings: { ...settings?.visualization_settings, gridSize, defaultGridSize: gridSize },
    });
  }, [updateSettings, settings]);

  const updateComputationSettings = useCallback(async (
    compSettings: Partial<EnhancedUserSettings['computation_settings']>,
  ) => {
    await updateSettings({
      computation_settings: { ...settings?.computation_settings, ...compSettings },
    });
  }, [updateSettings, settings]);

  const createUserSession = useCallback(async (
    sessionName: string,
    initialSettings?: Partial<EnhancedUserSettings>,
  ): Promise<string> => {
    await updateSettings({ session_name: sessionName, ...initialSettings });
    return 'local-session';
  }, [updateSettings]);

  const getUserSessions = useCallback(async (): Promise<EnhancedUserSettings[]> => {
    return settings ? [settings] : [];
  }, [settings]);

  const setActiveSession = useCallback(async (_sessionId: string) => {
    // Single session in local mode — nothing to do
  }, []);

  const getGridSize = useCallback((): number => {
    return settings?.visualization_settings?.defaultGridSize ?? DEFAULT_GRID_SIZE;
  }, [settings]);

  return {
    settings,
    isLoading,
    error: null,
    updateSettings,
    updateGridSize,
    updateComputationSettings,
    createUserSession,
    getUserSessions,
    setActiveSession,
    getGridSize,
    isReady: !isLoading,
  };
}