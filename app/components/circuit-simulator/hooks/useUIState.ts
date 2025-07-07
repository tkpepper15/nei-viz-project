import { useState, useEffect } from 'react';
import { ModelSnapshot } from '../types';
import { PerformanceSettings, DEFAULT_PERFORMANCE_SETTINGS } from '../controls/PerformanceControls';

export const useUIState = () => {
  // Tab navigation state
  const [visualizationTab, setVisualizationTab] = useState<'visualizer' | 'math' | 'data' | 'activity' | 'orchestrator'>('visualizer');
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(false);

  // Visualization settings
  const [opacityLevel, setOpacityLevel] = useState<number>(0.7);
  const [logScalar, setLogScalar] = useState<number>(1.0);

  // Reference model state
  const [referenceModelId, setReferenceModelId] = useState<string | null>(null);
  const [referenceModel, setReferenceModel] = useState<ModelSnapshot | null>(null);
  const [manuallyHidden, setManuallyHidden] = useState<boolean>(false);

  // Performance settings
  const [performanceSettings, setPerformanceSettings] = useState<PerformanceSettings>(DEFAULT_PERFORMANCE_SETTINGS);

  // Auto-manage toolbox visibility based on tab
  useEffect(() => {
    if (visualizationTab === 'visualizer') {
      setSidebarCollapsed(false); // Open toolbox for playground
    } else {
      setSidebarCollapsed(true); // Close toolbox for other tabs
    }
  }, [visualizationTab]);

  // Memory usage monitoring
  const getCurrentMemoryUsage = (): number => {
    if (typeof window !== 'undefined' && 'performance' in window && 'memory' in performance) {
      const memory = (performance as typeof performance & { memory: { usedJSHeapSize: number } }).memory;
      return memory.usedJSHeapSize / (1024 * 1024);
    }
    return 0;
  };

  return {
    // Tab navigation
    visualizationTab,
    setVisualizationTab,
    sidebarCollapsed,
    setSidebarCollapsed,

    // Visualization settings
    opacityLevel,
    setOpacityLevel,
    logScalar,
    setLogScalar,

    // Reference model
    referenceModelId,
    setReferenceModelId,
    referenceModel,
    setReferenceModel,
    manuallyHidden,
    setManuallyHidden,

    // Performance settings
    performanceSettings,
    setPerformanceSettings,

    // Utility functions
    getCurrentMemoryUsage,
  };
};