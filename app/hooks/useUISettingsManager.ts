import { useState, useCallback, useEffect, useMemo } from 'react';
import { UISettings } from '../../lib/circuitConfigService';
import { useAutoSaveUISettings } from './useAutoSaveUISettings';

// Default UI settings
const defaultUISettings: UISettings = {
  activeTab: 'visualizer',
  splitPaneHeight: 35,
  opacityLevel: 0.7,
  opacityExponent: 5,
  logScalar: 1.0,
  visualizationMode: 'color',
  backgroundColor: 'white',
  showGroundTruth: true,
  includeLabels: true,
  maxPolygons: 10000,
  useSymmetricGrid: false,
  adaptiveLimit: true,
  maxMemoryUsage: 8192,
  referenceModelVisible: true,
  manuallyHidden: false,
  isMultiSelectMode: false,
  selectedCircuits: [],
  windowPositions: {},
  sidebarCollapsed: false,
  toolboxPositions: {}
};

interface UseUISettingsManagerOptions {
  configId: string | null;
  initialSettings?: Partial<UISettings>;
  autoSaveEnabled?: boolean;
}

/**
 * Comprehensive hook for managing all UI settings with auto-save functionality
 */
export const useUISettingsManager = ({
  configId,
  initialSettings,
  autoSaveEnabled = true
}: UseUISettingsManagerOptions) => {
  const [uiSettings, setUISettings] = useState<UISettings>(() => ({
    ...defaultUISettings,
    ...initialSettings
  }));

  const { autoSaveUISettings, forceSaveUISettings, getSaveStatus, isEnabled } = useAutoSaveUISettings({
    configId,
    enabled: autoSaveEnabled,
    debounceMs: 1000
  });

  // Update UI settings and trigger auto-save
  const updateUISettings = useCallback((updates: Partial<UISettings>) => {
    setUISettings(prev => {
      const newSettings = { ...prev, ...updates };
      
      // Auto-save the new settings
      if (isEnabled) {
        autoSaveUISettings(newSettings);
      }
      
      return newSettings;
    });
  }, [autoSaveUISettings, isEnabled]);

  // Individual setters for convenience
  const setActiveTab = useCallback((tab: UISettings['activeTab']) => {
    updateUISettings({ activeTab: tab });
  }, [updateUISettings]);

  const setSplitPaneHeight = useCallback((height: number) => {
    updateUISettings({ splitPaneHeight: height });
  }, [updateUISettings]);

  const setOpacityLevel = useCallback((opacity: number) => {
    updateUISettings({ opacityLevel: opacity });
  }, [updateUISettings]);

  const setOpacityExponent = useCallback((exponent: number) => {
    updateUISettings({ opacityExponent: exponent });
  }, [updateUISettings]);

  const setLogScalar = useCallback((scalar: number) => {
    updateUISettings({ logScalar: scalar });
  }, [updateUISettings]);

  const setVisualizationMode = useCallback((mode: UISettings['visualizationMode']) => {
    updateUISettings({ visualizationMode: mode });
  }, [updateUISettings]);

  const setBackgroundColor = useCallback((color: UISettings['backgroundColor']) => {
    updateUISettings({ backgroundColor: color });
  }, [updateUISettings]);

  const setShowGroundTruth = useCallback((show: boolean) => {
    updateUISettings({ showGroundTruth: show });
  }, [updateUISettings]);

  const setIncludeLabels = useCallback((include: boolean) => {
    updateUISettings({ includeLabels: include });
  }, [updateUISettings]);

  const setMaxPolygons = useCallback((max: number) => {
    updateUISettings({ maxPolygons: max });
  }, [updateUISettings]);

  const setReferenceModelVisible = useCallback((visible: boolean) => {
    updateUISettings({ referenceModelVisible: visible });
  }, [updateUISettings]);

  const setManuallyHidden = useCallback((hidden: boolean) => {
    updateUISettings({ manuallyHidden: hidden });
  }, [updateUISettings]);

  const setIsMultiSelectMode = useCallback((multiSelect: boolean) => {
    updateUISettings({ isMultiSelectMode: multiSelect });
  }, [updateUISettings]);

  const setSelectedCircuits = useCallback((circuits: string[]) => {
    updateUISettings({ selectedCircuits: circuits });
  }, [updateUISettings]);

  const setWindowPosition = useCallback((windowId: string, position: { x: number; y: number; width: number; height: number }) => {
    updateUISettings({
      windowPositions: {
        ...uiSettings.windowPositions,
        [windowId]: position
      }
    });
  }, [updateUISettings, uiSettings.windowPositions]);

  const setToolboxPosition = useCallback((toolboxId: string, position: { x: number; y: number }) => {
    updateUISettings({
      toolboxPositions: {
        ...uiSettings.toolboxPositions,
        [toolboxId]: position
      }
    });
  }, [updateUISettings, uiSettings.toolboxPositions]);

  const setSidebarCollapsed = useCallback((collapsed: boolean) => {
    updateUISettings({ sidebarCollapsed: collapsed });
  }, [updateUISettings]);

  // Load settings from configuration when configId or initialSettings change
  useEffect(() => {
    if (initialSettings) {
      setUISettings(prev => ({
        ...prev,
        ...initialSettings
      }));
    }
  }, [initialSettings]);

  // Reset to defaults
  const resetToDefaults = useCallback(() => {
    const resetSettings = { ...defaultUISettings };
    setUISettings(resetSettings);
    if (isEnabled) {
      autoSaveUISettings(resetSettings);
    }
  }, [autoSaveUISettings, isEnabled]);

  // Force save current settings
  const forceSave = useCallback(() => {
    if (isEnabled) {
      return forceSaveUISettings(uiSettings);
    }
    return Promise.resolve(false);
  }, [forceSaveUISettings, uiSettings, isEnabled]);

  // Export settings for backup/sharing
  const exportSettings = useCallback(() => {
    return JSON.stringify(uiSettings, null, 2);
  }, [uiSettings]);

  // Import settings from backup/share
  const importSettings = useCallback((settingsJson: string) => {
    try {
      const importedSettings = JSON.parse(settingsJson) as Partial<UISettings>;
      updateUISettings(importedSettings);
      return true;
    } catch (error) {
      console.error('Failed to import UI settings:', error);
      return false;
    }
  }, [updateUISettings]);

  // Get current save status
  const saveStatus = useMemo(() => getSaveStatus(), [getSaveStatus]);

  return {
    // Current settings
    uiSettings,
    
    // Main updater
    updateUISettings,
    
    // Individual setters
    setActiveTab,
    setSplitPaneHeight,
    setOpacityLevel,
    setOpacityExponent,
    setLogScalar,
    setVisualizationMode,
    setBackgroundColor,
    setShowGroundTruth,
    setIncludeLabels,
    setMaxPolygons,
    setReferenceModelVisible,
    setManuallyHidden,
    setIsMultiSelectMode,
    setSelectedCircuits,
    setWindowPosition,
    setToolboxPosition,
    setSidebarCollapsed,
    
    // Utilities
    resetToDefaults,
    forceSave,
    exportSettings,
    importSettings,
    
    // Status
    saveStatus,
    isAutoSaveEnabled: isEnabled
  };
};