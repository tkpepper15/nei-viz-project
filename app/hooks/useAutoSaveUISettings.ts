import { useCallback, useEffect, useRef } from 'react';
import { CircuitConfigService, UISettings } from '../../lib/circuitConfigService';

interface UseAutoSaveUISettingsOptions {
  configId: string | null;
  enabled?: boolean;
  debounceMs?: number;
}

/**
 * Hook for auto-saving UI settings to prevent data loss and provide seamless experience
 */
export const useAutoSaveUISettings = ({
  configId,
  enabled = true,
  debounceMs = 1000 // 1 second debounce
}: UseAutoSaveUISettingsOptions) => {
  const debounceTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastSaveTimeRef = useRef<number>(0);
  const pendingSaveRef = useRef<UISettings | null>(null);

  // Auto-save UI settings with debouncing
  const autoSaveUISettings = useCallback(async (uiSettings: UISettings) => {
    if (!configId || !enabled) {
      return;
    }

    // Store the latest settings for pending save
    pendingSaveRef.current = uiSettings;

    // Clear existing timeout
    if (debounceTimeoutRef.current) {
      clearTimeout(debounceTimeoutRef.current);
    }

    // Set new debounced timeout
    debounceTimeoutRef.current = setTimeout(async () => {
      const settingsToSave = pendingSaveRef.current;
      if (!settingsToSave) return;

      try {
        const success = await CircuitConfigService.updateUISettings(configId, settingsToSave);
        if (success) {
          lastSaveTimeRef.current = Date.now();
          console.log('🔄 Auto-saved UI settings');
        } else {
          console.warn('⚠️ Failed to auto-save UI settings');
        }
      } catch (error) {
        console.error('❌ Error auto-saving UI settings:', error);
      }

      pendingSaveRef.current = null;
    }, debounceMs);
  }, [configId, enabled, debounceMs]);

  // Force save (for critical moments like page unload)
  const forceSaveUISettings = useCallback(async (uiSettings: UISettings) => {
    if (!configId || !enabled) {
      return false;
    }

    // Cancel any pending debounced save
    if (debounceTimeoutRef.current) {
      clearTimeout(debounceTimeoutRef.current);
      debounceTimeoutRef.current = null;
    }

    try {
      const success = await CircuitConfigService.updateUISettings(configId, uiSettings);
      if (success) {
        lastSaveTimeRef.current = Date.now();
        pendingSaveRef.current = null;
        console.log('💾 Force-saved UI settings');
      }
      return success;
    } catch (error) {
      console.error('❌ Error force-saving UI settings:', error);
      return false;
    }
  }, [configId, enabled]);

  // Get save status
  const getSaveStatus = useCallback(() => {
    return {
      hasPendingSave: pendingSaveRef.current !== null,
      lastSaveTime: lastSaveTimeRef.current,
      isEnabled: enabled && !!configId
    };
  }, [enabled, configId]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current);
      }
    };
  }, []);

  // Auto-save on page unload
  useEffect(() => {
    const handleBeforeUnload = (event: BeforeUnloadEvent) => {
      if (pendingSaveRef.current && configId && enabled) {
        // Try to save synchronously (though this is limited by browser)
        navigator.sendBeacon('/api/save-ui-settings', JSON.stringify({
          configId,
          uiSettings: pendingSaveRef.current
        }));
        
        // Show warning if there are unsaved changes
        event.preventDefault();
        event.returnValue = 'You have unsaved UI settings. Are you sure you want to leave?';
        return event.returnValue;
      }
    };

    const handleVisibilityChange = () => {
      if (document.hidden && pendingSaveRef.current && configId && enabled) {
        // Page is becoming hidden, force save
        forceSaveUISettings(pendingSaveRef.current);
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [configId, enabled, forceSaveUISettings]);

  return {
    autoSaveUISettings,
    forceSaveUISettings,
    getSaveStatus,
    isEnabled: enabled && !!configId
  };
};