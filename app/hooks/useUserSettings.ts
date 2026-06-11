'use client';

import { useState, useEffect, useCallback } from 'react';
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

const SETTINGS_KEY = 'nei-viz-user-settings';

function loadSettings(userId: string): UserSettings {
  if (typeof window === 'undefined') {
    return { id: 'local-session', user_id: userId, is_active: true };
  }
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    return raw ? JSON.parse(raw) : { id: 'local-session', user_id: userId, is_active: true };
  } catch {
    return { id: 'local-session', user_id: userId, is_active: true };
  }
}

function persistSettings(s: UserSettings): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(SETTINGS_KEY, JSON.stringify({ ...s, updated_at: new Date().toISOString() }));
}

export function useUserSettings(userId?: string): UseUserSettingsReturn {
  const [settings, setSettings] = useState<UserSettings | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!userId) {
      setIsLoading(false);
      setSettings(null);
      return;
    }
    setSettings(loadSettings(userId));
    setIsLoading(false);
  }, [userId]);

  const updateSettings = useCallback(async (newSettings: Partial<UserSettings>) => {
    setSettings(prev => {
      const next = { ...prev!, ...newSettings };
      persistSettings(next);
      return next;
    });
  }, []);

  const createUserSession = useCallback(async (sessionName: string, sessionSettings?: Partial<UserSettings>): Promise<string> => {
    const next: UserSettings = {
      id: 'local-session',
      user_id: userId ?? 'local-user',
      session_name: sessionName,
      ...sessionSettings,
      is_active: true,
    };
    persistSettings(next);
    setSettings(next);
    return 'local-session';
  }, [userId]);

  const getUserSessions = useCallback(async (): Promise<UserSettings[]> => {
    return settings ? [settings] : [];
  }, [settings]);

  const setActiveSession = useCallback(async (_sessionId: string) => {
    // Single session in local mode — nothing to do
  }, []);

  return {
    settings,
    isLoading,
    error: null,
    updateSettings,
    createUserSession,
    getUserSessions,
    setActiveSession,
  };
}
