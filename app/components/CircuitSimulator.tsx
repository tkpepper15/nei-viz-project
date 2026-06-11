"use client";

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useRouter, usePathname, useSearchParams } from 'next/navigation';
import 'katex/dist/katex.min.css';
import { 
  BackendMeshPoint 
} from './circuit-simulator/types';
import { ModelSnapshot, ImpedancePoint, ResnormGroup, PerformanceLog, PipelinePhase } from './circuit-simulator/types';
import { CircuitParameters } from './circuit-simulator/types/parameters';
import { useWorkerManager, WorkerProgress } from './circuit-simulator/utils/workerManager';
import { useHybridComputeManager } from './circuit-simulator/utils/hybridComputeManager';
import { useOptimizedComputeManager, OptimizedComputeManagerConfig } from './circuit-simulator/utils/optimizedComputeManager';
import { SerializedComputationManager, createSerializedComputationManager } from './circuit-simulator/utils/serializedComputationManager';
import { createEnhancedSerializedManager } from './circuit-simulator/utils/enhancedSerializedManager';
import { useEnhancedUserSettings } from '../hooks/useEnhancedUserSettings';
import { createParameterConfigManager } from './circuit-simulator/utils/parameterConfigManager';
import { ExtendedPerformanceSettings, DEFAULT_GPU_SETTINGS, DEFAULT_CPU_SETTINGS } from './circuit-simulator/types/gpuSettings';
import { useComputationState } from './circuit-simulator/hooks/useComputationState';
import { useUserProfiles } from '../hooks/useUserProfiles';
import { useSessionManagement } from '../hooks/useSessionManagement';
import { useCircuitConfigurations } from '../hooks/useCircuitConfigurations';
import { useUISettingsManager } from '../hooks/useUISettingsManager';
import { ProfilesService } from '../../lib/profilesService';
import { CentralizedLimitsManager, setGlobalLimitsManager } from './circuit-simulator/utils/centralizedLimits';

// Tab components
import { AIModelTab } from './circuit-simulator/AIModelTab';

import { PerformanceSettings, DEFAULT_PERFORMANCE_SETTINGS } from './circuit-simulator/controls/PerformanceControls';
import { ComputationNotification, ComputationSummary } from './circuit-simulator/notifications/ComputationNotification';
import { ResnormConfig, ResnormMethod } from './circuit-simulator/utils/resnorm';
import { AuthModal } from './auth/AuthModal';
import { useAuth } from './auth/AuthProvider';
import { SlimNavbar } from './circuit-simulator/SlimNavbar';
import { ModelInfoModal } from './model-info/ModelInfoModal';
import { PILL_BASE, DIVIDER, ICON_BTN, MONO_VALUE, LABEL_BTN_DISABLED, ICON } from './circuit-simulator/ui/tokens';

// Remove empty interface and replace with type
type CircuitSimulatorProps = Record<string, never>;

export const CircuitSimulator: React.FC<CircuitSimulatorProps> = () => {
  // Routing
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams(); // eslint-disable-line @typescript-eslint/no-unused-vars

  // Authentication
  const { user, loading: authLoading, signOut } = useAuth();
  const [showAuthModal, setShowAuthModal] = useState(false);

  // Enhanced User Settings with grid size persistence
  const enhancedSettings = useEnhancedUserSettings(user?.id);

  // Enhanced Serialized Data Manager for intelligent data loading
  const [enhancedSerializedManager] = useState(() => createEnhancedSerializedManager());

  // State for user's default grid size (now sourced from enhanced settings)
  const [defaultGridSize, setDefaultGridSize] = useState<number>(9);

  // Centralized limits manager - single variable to control all computation limits
  const [centralizedLimits, setCentralizedLimits] = useState<CentralizedLimitsManager>(
    CentralizedLimitsManager.fromGridSize(9, 100) // Force: 100% of grid to ensure complete computation
  );
  
  // Initialize session management
  const sessionManagement = useSessionManagement();
  
  // Circuit configurations management (replaces savedProfilesState)
  const {
    configurations: circuitConfigurations, // ENABLED: For sidebar display
    activeConfigId,
    activeConfiguration, // NOW USED: For UI settings restoration
    createConfiguration,
    updateConfiguration, // NEEDED: For handleSaveProfile
    deleteConfiguration: _deleteConfiguration, // kept for future use
    deleteMultipleConfigurations, // ENABLED: For bulk deletion
    setActiveConfiguration
  } = useCircuitConfigurations(
    sessionManagement.sessionState.userId || undefined,
    sessionManagement.sessionState.currentCircuitConfigId
  );

  // Ref mirror so the auto-save effect can read circuitConfigurations without it being a dep
  const circuitConfigurationsRef = useRef(circuitConfigurations);
  useEffect(() => { circuitConfigurationsRef.current = circuitConfigurations; }, [circuitConfigurations]);

  // UI Settings management with auto-save
  const {
    uiSettings,
    setActiveTab,
    setSplitPaneHeight, // eslint-disable-line @typescript-eslint/no-unused-vars
    setOpacityLevel, // eslint-disable-line @typescript-eslint/no-unused-vars
    setOpacityExponent, // eslint-disable-line @typescript-eslint/no-unused-vars
    setLogScalar, // eslint-disable-line @typescript-eslint/no-unused-vars
    setVisualizationMode, // eslint-disable-line @typescript-eslint/no-unused-vars
    setBackgroundColor, // eslint-disable-line @typescript-eslint/no-unused-vars
    setShowGroundTruth, // eslint-disable-line @typescript-eslint/no-unused-vars
    setIncludeLabels, // eslint-disable-line @typescript-eslint/no-unused-vars
    setMaxPolygons, // eslint-disable-line @typescript-eslint/no-unused-vars
    setReferenceModelVisible, // eslint-disable-line @typescript-eslint/no-unused-vars
    setManuallyHidden,
    setIsMultiSelectMode,
    setSelectedCircuits,
    forceSave // eslint-disable-line @typescript-eslint/no-unused-vars
  } = useUISettingsManager({
    configId: activeConfigId,
    initialSettings: activeConfiguration?.uiSettings,
    autoSaveEnabled: true
  });

  // User profiles (for backward compatibility - will be phased out)
  const { profilesState: savedProfilesState, actions: profileActions } = useUserProfiles();
  
  // Initialize worker manager for parallel computation
  const cpuWorkerManager = useWorkerManager();
  const hybridComputeManager = useHybridComputeManager(cpuWorkerManager);
  const { cancelComputation } = cpuWorkerManager;

  // Initialize optimization manager with fallback to hybrid compute
  const [optimizationConfig] = useState<OptimizedComputeManagerConfig>({
    enableOptimizedPipeline: true,
    optimizationThreshold: 10000,
    fallbackToOriginal: true,
    maxGridSizeForOptimization: 1000000,
    pipelineConfig: {
      fingerprintFrequencies: 8,
      approximationFrequencies: 5,
      coarseFrequencies: 12,
      fullFrequencies: 20,
      topMSurvivors: 5000,
      finalTopK: 1000,
      chunkSize: 15000,
      toleranceForTies: 0.01,
      enableLocalOptimization: false,
      quantizationBin: 0.05
    }
  });

  const optimizedComputeManager = useOptimizedComputeManager(
    hybridComputeManager.computeGridHybrid,
    optimizationConfig
  );

  
  // Add frequency control state - extended range for full Nyquist closure
  const [minFreq, setMinFreq] = useState<number>(0.1); // 0.1 Hz for full low-frequency closure
  const [maxFreq, setMaxFreq] = useState<number>(100000); // 100 kHz for high-frequency closure
  const [numPoints, setNumPoints] = useState<number>(100); // Default number of frequency points
  const [frequencyPoints, setFrequencyPoints] = useState<number[]>([]);
  
  // Resnorm calculation configuration - fixed to SSR method only
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [resnormConfig, setResnormConfig] = useState<ResnormConfig>({
    method: ResnormMethod.SSR, // Sum of Squared Residuals method only
    useRangeAmplification: false,
    useFrequencyWeighting: false
  });
  
  // Performance settings for computation optimization
  const [performanceSettings] = useState<PerformanceSettings>(DEFAULT_PERFORMANCE_SETTINGS);
  const [extendedSettings, _setExtendedSettings] = useState<ExtendedPerformanceSettings>({
    useSymmetricGrid: true,
    maxComputationResults: 500000, // Compute all data, limit display separately
    gpuAcceleration: DEFAULT_GPU_SETTINGS,
    cpuSettings: DEFAULT_CPU_SETTINGS
  });
  
  // Model info modal state
  const [modelInfoModalOpen, setModelInfoModalOpen] = useState(false);

  // Canvas zoom — driven by AIModelTab via callback
  const [canvasZoom, setCanvasZoom] = useState(1);
  const zoomActionsRef = useRef<{ zoomIn: () => void; zoomOut: () => void; zoomReset: () => void } | null>(null);

  // Welcome modal state — shown once on first visit
  const [showWelcome, setShowWelcome] = useState(false);
  useEffect(() => {
    if (typeof window !== 'undefined' && !localStorage.getItem('spideyplot_welcomed')) {
      setShowWelcome(true);
    }
  }, []);
  const dismissWelcome = () => {
    localStorage.setItem('spideyplot_welcomed', '1');
    setShowWelcome(false);
  };

  // Memory optimization settings
  const [visibleResultsLimit, setVisibleResultsLimit] = useState(100000); // Increased limit for comprehensive data analysis
  const [memoryOptimizationEnabled, setMemoryOptimizationEnabled] = useState(true);

  // SRD upload state
  const [, setSrdUploadMessage] = useState('');

  // Tagged models state - for model tagging and highlighting functionality
  const [taggedModels, setTaggedModels] = useState<Map<string, { tagName: string; profileId: string; resnormValue: number; taggedAt: number; notes?: string }>>(new Map());

  // Streamlined parameter configuration manager - initialized with defaults to avoid circular deps
  const [paramConfigManager] = useState(() => createParameterConfigManager({
    gridSize: 9, // Will be updated from settings in useEffect
    minFreq: 0.1,
    maxFreq: 100000,
    numPoints: 100,
    maxVisibleResults: 1000,
    memoryOptimizationEnabled: true
  }));

  // Sync grid size with enhanced user settings
  // Use the new computation state hook
  const {
    gridResults, setGridResults,
    gridSize, setGridSize,
    gridError, setGridError,
    isComputingGrid, setIsComputingGrid,
    computationProgress, setComputationProgress,
    computationSummary, setComputationSummary,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    skippedPoints, setSkippedPoints,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    totalGridPoints, setTotalGridPoints,
    actualComputedPoints, setActualComputedPoints,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    memoryLimitedPoints, setMemoryLimitedPoints,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    estimatedMemoryUsage, setEstimatedMemoryUsage,
    userVisualizationPercentage,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    maxVisualizationPoints,
    isUserControlledLimits,
    calculateEffectiveVisualizationLimit,
    // maxComputationResults, setMaxComputationResults, // Now using centralizedLimits instead
    resnormGroups: _resnormGroups, setResnormGroups, // eslint-disable-line @typescript-eslint/no-unused-vars
    hiddenGroups, setHiddenGroups,
    logMessages,
    statusMessage: _statusMessage,
    updateStatusMessage,
    resetComputationState,
    saveComputationState,
    restoreComputationState: _restoreComputationState, // eslint-disable-line @typescript-eslint/no-unused-vars
    clearAllComputationData, // eslint-disable-line @typescript-eslint/no-unused-vars
    lastComputedResults: _lastComputedResults, // eslint-disable-line @typescript-eslint/no-unused-vars
    updateDefaultGridSize
  } = useComputationState(defaultGridSize);



  // Sync grid size with enhanced user settings (only when settings change, not when gridSize changes)
  useEffect(() => {
    if (enhancedSettings.isReady && enhancedSettings.settings) {
      const persistedGridSize = enhancedSettings.getGridSize();
      if (persistedGridSize !== gridSize) {
        console.log(`Syncing grid size from user settings: ${gridSize} -> ${persistedGridSize}`);
        setGridSize(persistedGridSize);
        updateDefaultGridSize(persistedGridSize);
        paramConfigManager.updateGridSize(persistedGridSize);
      }
    }
  }, [enhancedSettings.isReady, enhancedSettings.settings]); // eslint-disable-line react-hooks/exhaustive-deps

  // Enhanced grid size update function that persists to settings
  const updateGridSizeWithPersistence = useCallback((newGridSize: number) => {
    console.log(`Updating grid size with persistence: ${gridSize} -> ${newGridSize}`);

    // Update local state
    setGridSize(newGridSize);
    updateDefaultGridSize(newGridSize);
    paramConfigManager.updateGridSize(newGridSize);

    // Persist to user settings (with debouncing handled by enhanced settings)
    if (enhancedSettings.isReady) {
      enhancedSettings.updateGridSize(newGridSize).catch(err => {
        console.error('Failed to persist grid size:', err);
      });
    }
  }, [gridSize, setGridSize, updateDefaultGridSize, paramConfigManager, enhancedSettings]);

  // Serialized computation manager for efficient memory storage
  const [serializedManager, setSerializedManager] = useState<SerializedComputationManager | null>(null);

  // SINGLE-CIRCUIT MEMORY MANAGEMENT: Only keep current circuit in memory
  // Track previous grid size to avoid unnecessary clears
  const prevGridSizeRef = React.useRef(gridSize);

  useEffect(() => {
    // Only clear if grid size actually changed
    const gridSizeChanged = prevGridSizeRef.current !== gridSize;
    prevGridSizeRef.current = gridSize;

    if (gridSizeChanged) {
      // Clear previous manager and force cleanup of all cached data
      if (serializedManager) {
        serializedManager.clearCaches();
        console.log('Cleared previous SerializedComputationManager and all caches');
      }

      // Clear UI state to prevent memory accumulation across computations
      setGridResults([]);
      setResnormGroups([]);
      setComputationSummary(null);
    }

    // Update centralized limits manager for new grid size
    const newCentralizedLimits = CentralizedLimitsManager.fromGridSize(gridSize, centralizedLimits.masterLimitPercentage);
    setCentralizedLimits(newCentralizedLimits);
    setGlobalLimitsManager(newCentralizedLimits);
    console.log(`Updated centralized limits manager for grid ${gridSize} (${newCentralizedLimits.masterLimitPercentage}% = ${newCentralizedLimits.masterLimitResults.toLocaleString()} models)`);

    // DEBUG: Check if the percentage is not 100%
    if (newCentralizedLimits.masterLimitPercentage !== 100) {
      console.warn(`LIMITS DEBUG: Master limit is ${newCentralizedLimits.masterLimitPercentage}% instead of 100%! This explains incomplete grid generation.`);
      console.log(`FIXING: Forcing master limit back to 100% to compute complete grid`);

      // Force reset to 100% for complete grid computation
      const fixedLimits = CentralizedLimitsManager.fromGridSize(gridSize, 100);
      setCentralizedLimits(fixedLimits);
      setGlobalLimitsManager(fixedLimits);
      console.log(`Fixed: Master limit reset to 100% (${fixedLimits.masterLimitResults.toLocaleString()} models)`);
    }

    // Create new optimized manager for current grid configuration
    const manager = createSerializedComputationManager(gridSize, 'standard');
    setSerializedManager(manager);

    const estimatedTraditionalMemory = Math.pow(gridSize, 5) * 4000 / (1024 * 1024);
    const estimatedOptimizedMemory = Math.pow(gridSize, 5) * 61 / (1024 * 1024);
    const reductionFactor = estimatedTraditionalMemory / estimatedOptimizedMemory;

    console.log(`SINGLE-CIRCUIT MODE: New manager for ${gridSize}^5 grid`);
    console.log(`Memory optimization: ${estimatedTraditionalMemory.toFixed(1)}MB → ${estimatedOptimizedMemory.toFixed(1)}MB (${reductionFactor.toFixed(1)}x reduction)`);
  }, [gridSize, centralizedLimits.masterLimitPercentage, setGridResults, setResnormGroups, setComputationSummary]);


  // Enhanced UI coordination refs for large dataset processing
  const progressUpdateQueueRef = useRef<WorkerProgress[]>([]);
  const isProcessingUpdatesRef = useRef(false);

  // Performance tracking helper functions
  const startPhase = useCallback((name: string, description: string) => {
    const phase: PipelinePhase = {
      name,
      description,
      startTime: Date.now(),
      status: 'running'
    };
    setCurrentPhases(prev => [...prev, phase]);
    updateStatusMessage(`[${name.toUpperCase()}] ${description}`);
    return phase;
  }, [updateStatusMessage]);

  const completePhase = useCallback((phaseName: string, details?: Record<string, unknown>) => {
    setCurrentPhases(prev => {
      const updated = prev.map(phase => {
        if (phase.name === phaseName && phase.status === 'running') {
          const endTime = Date.now();
          return {
            ...phase,
            endTime,
            duration: endTime - phase.startTime,
            status: 'completed' as const,
            details
          };
        }
        return phase;
      });
      return updated;
    });
  }, []);

  const generatePerformanceLog = useCallback((phases: PipelinePhase[], totalDuration: number, totalPoints: number): PerformanceLog => {
    const completedPhases = phases.filter(p => p.status === 'completed' && p.duration);
    
    // Find bottleneck (longest phase)
    const bottleneck = completedPhases.reduce((longest, current) => 
      (current.duration || 0) > (longest.duration || 0) ? current : longest
    , completedPhases[0]);

    // Calculate efficiency metrics
    const cpuCores = navigator.hardwareConcurrency || 4;
    const parallelization = Math.min(1, (totalPoints / totalDuration) / (cpuCores * 1000)); // rough estimate
    const throughput = totalPoints / (totalDuration / 1000); // points per second
    const memoryUsage = (totalPoints * 800) / (1024 * 1024); // estimated MB

    // Create summary breakdown
    const summary = {
      gridGeneration: completedPhases.find(p => p.name === 'grid-generation')?.duration || 0,
      impedanceComputation: completedPhases.find(p => p.name === 'impedance-computation')?.duration || 0,
      resnormAnalysis: completedPhases.find(p => p.name === 'resnorm-analysis')?.duration || 0,
      dataProcessing: completedPhases.find(p => p.name === 'data-processing')?.duration || 0,
      rendering: completedPhases.find(p => p.name === 'rendering')?.duration || 0
    };

    return {
      totalDuration,
      phases: completedPhases,
      bottleneck: bottleneck ? `${bottleneck.name} (${(bottleneck.duration! / 1000).toFixed(2)}s)` : 'Unknown',
      efficiency: {
        parallelization,
        memoryUsage,
        throughput,
        cpuCores
      },
      summary
    };
  }, []);

  // Visualization tab is now managed by UI settings
  const visualizationTab = uiSettings.activeTab;
  const setVisualizationTab = setActiveTab;
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_parameterChanged, setParameterChanged] = useState<boolean>(false);

  // Sync route with visualization tab state
  useEffect(() => {
    if (pathname === '/simulator/activity' && visualizationTab !== 'activity') {
      setVisualizationTab('activity');
    }
  }, [pathname, visualizationTab, setVisualizationTab]);

  // Load user's default grid size when user changes
  useEffect(() => {
    if (user?.id && !authLoading) {
      ProfilesService.getUserDefaultGridSize(user.id).then((loadedDefaultGridSize) => {
        // Only update the default, don't change the current grid size if a circuit is loaded
        setDefaultGridSize(loadedDefaultGridSize);
        console.log(`[PROFILE] Loaded user default grid size: ${loadedDefaultGridSize}`);
        // DO NOT call updateDefaultGridSize() here - it would override the active circuit's grid size
      });
    }
  }, [user?.id, authLoading]);
  
  
  // Multi-select state for circuits is now managed by UI settings
  const selectedCircuits = uiSettings.selectedCircuits;
  const isMultiSelectMode = uiSettings.isMultiSelectMode;
  // Track if reference model was manually hidden
  // Manually hidden state is now managed by UI settings
  const manuallyHidden = uiSettings.manuallyHidden;
  // Configuration name for saving profiles
  const [configurationName, setConfigurationName] = useState<string>('');
  

  // Saved profiles state is now handled by useUserProfiles hook

  
  // Log scalar is now managed by UI settings
  const logScalar = uiSettings.logScalar; // eslint-disable-line @typescript-eslint/no-unused-vars
  
  // Visualization settings state
  const [visualizationSettings] = useState<{
    groupPortion: number;
    selectedOpacityGroups: number[];
    visualizationType: 'spider' | 'nyquist';
    view3D: boolean;
  }>({
    groupPortion: 0.5,
    selectedOpacityGroups: [0],
    visualizationType: 'spider',
    view3D: false
  });


  // Apply visualization settings filtering to computed results
  const applyVisualizationFiltering = useCallback((
    results: ModelSnapshot[], 
    groups: ResnormGroup[], 
    settings: typeof visualizationSettings
  ): ModelSnapshot[] => {
    if (!results.length || !groups.length) return results;
    
    // Step 1: Filter by selected opacity groups
    const groupFilteredResults = results.filter(model => {
      // Find which group this model belongs to
      const groupIndex = groups.findIndex(group => 
        group.items.some(item => item.id === model.id)
      );
      
      // Only include models from selected groups
      return settings.selectedOpacityGroups.includes(groupIndex);
    });
    
    // Step 2: Apply group portion filtering
    // Group models by their resnorm groups
    const groupedModels: { [key: number]: ModelSnapshot[] } = {};
    
    groupFilteredResults.forEach(model => {
      const groupIndex = groups.findIndex(group => 
        group.items.some(item => item.id === model.id)
      );
      
      if (groupIndex !== -1) {
        if (!groupedModels[groupIndex]) groupedModels[groupIndex] = [];
        groupedModels[groupIndex].push(model);
      }
    });
    
    // Step 3: Apply portion filtering to each group
    const filteredResults: ModelSnapshot[] = [];
    
    Object.keys(groupedModels).forEach(groupKey => {
      const groupIndex = parseInt(groupKey);
      const groupModels = groupedModels[groupIndex];
      
      if (!groupModels || groupModels.length === 0) return;
      
      // Sort by resnorm (ascending - lower is better) and apply portion filtering
      const sortedGroupModels = groupModels
        .filter(m => m.resnorm !== undefined)
        .sort((a, b) => (a.resnorm || 0) - (b.resnorm || 0));
      
      // Keep only the specified portion of the group (best models)
      const keepCount = Math.max(1, Math.floor(sortedGroupModels.length * settings.groupPortion));
      const filteredGroupModels = sortedGroupModels.slice(0, keepCount);
      
      filteredResults.push(...filteredGroupModels);
    });
    
    updateStatusMessage(`[VIZ FILTERING] Applied settings: ${settings.selectedOpacityGroups.length} groups, ${(settings.groupPortion * 100).toFixed(1)}% portion`);
    updateStatusMessage(`[VIZ FILTERING] Filtered from ${results.length} to ${filteredResults.length} models (${((filteredResults.length / results.length) * 100).toFixed(1)}% rendered)`);
    
    return filteredResults;
  }, [updateStatusMessage]);
  
  const [referenceParams, setReferenceParams] = useState<CircuitParameters>({
    Rsh: 100,
    Ra: 1000,
    Ca: 1.0e-6, // 1.0 microfarads (converted to farads)
    Rb: 800,
    Cb: 0.8e-6, // 0.8 microfarads (converted to farads)
    frequency_range: [minFreq, maxFreq]
  });
  
  // Reference model state
  const [referenceModelId, setReferenceModelId] = useState<string | null>(null);


  // Performance logging state
  const [performanceLog, setPerformanceLog] = useState<PerformanceLog | null>(null);
  const [currentPhases, setCurrentPhases] = useState<PipelinePhase[]>([]);
  const [referenceModel, setReferenceModel] = useState<ModelSnapshot | null>(null);
  
  // Ground truth parameters now use the user-controlled parameters directly
  
  // Use a single parameters state object with new standard starting configuration
  const [parameters, setParameters] = useState<CircuitParameters>({
    Rsh: 100,     // Standard shunt resistance
    Ra: 1000,     // Standard apical resistance
    Ca: 1.0e-6,   // Standard apical capacitance (1.0 µF)
    Rb: 800,      // Standard basal resistance
    Cb: 0.8e-6,   // Standard basal capacitance (0.8 µF)
    frequency_range: [0.1, 100000]
  });
  
  // Subscribe to parameter changes for automatic UI synchronization
  useEffect(() => {
    const unsubscribe = paramConfigManager.subscribe((state, change) => {
      if (change) {
        console.log(`Parameter change detected: ${change.type}.${change.field}`);

        // Auto-sync UI state with parameter manager
        switch (change.type) {
          case 'parameter':
            if (JSON.stringify(state.parameters) !== JSON.stringify(parameters)) {
              setParameters(state.parameters);
            }
            break;
          case 'grid':
            if (state.gridSize !== gridSize) {
              setGridSize(state.gridSize);
            }
            break;
          case 'frequency':
            if (state.minFreq !== minFreq) setMinFreq(state.minFreq);
            if (state.maxFreq !== maxFreq) setMaxFreq(state.maxFreq);
            if (state.numPoints !== numPoints) setNumPoints(state.numPoints);
            break;
          case 'optimization':
            if (state.maxVisibleResults !== visibleResultsLimit) {
              setVisibleResultsLimit(state.maxVisibleResults);
            }
            if (state.memoryOptimizationEnabled !== memoryOptimizationEnabled) {
              setMemoryOptimizationEnabled(state.memoryOptimizationEnabled);
            }
            break;
        }
      }
    });

    return unsubscribe;
  }, [paramConfigManager, parameters, gridSize, minFreq, maxFreq, numPoints, visibleResultsLimit, memoryOptimizationEnabled]);

  // Auto-save configuration when parameters are adjusted
  useEffect(() => {
    const autoSaveTimer = setTimeout(async () => {
      // Only auto-save if we have an active config and the user is signed in
      if (activeConfigId && sessionManagement.sessionState.userId) {
        try {
          console.log('[AUTO-SAVE] Saving configuration changes...', {
            activeConfigId,
            parameters: Object.keys(parameters),
            gridSize,
            frequency: { min: minFreq, max: maxFreq, points: numPoints }
          });

          const success = await updateConfiguration(activeConfigId, {
            circuitParameters: parameters,
            gridSize,
            minFreq,
            maxFreq,
            numPoints
          });

          if (success) {
            console.log('[AUTO-SAVE] Configuration saved successfully');

            // Log activity with timestamp
            const currentConfig = circuitConfigurationsRef.current?.find(c => c.id === activeConfigId);
            // Format capacitance values in microfarads for readability
            const paramDisplay = Object.entries(parameters)
              .map(([k, v]) => {
                if ((k === 'Ca' || k === 'Cb') && typeof v === 'number') {
                  return `${k}=${(v * 1e6).toFixed(2)}µF`;
                }
                return `${k}=${typeof v === 'number' ? v.toFixed(1) : v}`;
              })
              .join(', ');
            const activityMessage = `Auto-saved "${currentConfig?.name || 'Current Profile'}" - Parameters: ${paramDisplay}, Grid: ${gridSize}^5, Freq: ${minFreq}-${maxFreq}Hz (${numPoints}pts)`;

            // Update status with detailed info but don't spam
            const timestamp = new Date().toLocaleTimeString();
            updateStatusMessage(`[${timestamp}] Auto-saved configuration changes`);

            console.log('[ACTIVITY]', activityMessage);
          } else {
            console.warn('[AUTO-SAVE] Configuration save failed');
            updateStatusMessage('Auto-save failed - changes may be lost');
          }
        } catch (error) {
          const errorMsg = error instanceof Error ? error.message : String(error);
          console.error('[AUTO-SAVE ERROR]', errorMsg, error);
          updateStatusMessage(`Auto-save failed: ${errorMsg}`);
        }
      }
    }, 5000); // 5-second delay to prevent aggressive saves during parameter adjustment

    return () => clearTimeout(autoSaveTimer);
  }, [parameters, gridSize, minFreq, maxFreq, numPoints, activeConfigId, sessionManagement.sessionState.userId, updateConfiguration]);

  // SRD Upload Handlers (kept for potential future use)
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const handleSRDUploaded = useCallback(async (manager: SerializedComputationManager, metadata: { title: string; totalResults: number; gridSize: number }) => {
    try {
      // Set the serialized manager as the active one
      setSerializedManager(manager);

      // Update grid size and frequency settings to match the SRD (with persistence)
      updateGridSizeWithPersistence(metadata.gridSize);
      // Extract frequency settings from manager config
      const configData = manager.getConfig();
      if (configData.frequencyPreset) {
        // Update frequency settings based on preset
        switch (configData.frequencyPreset) {
          case 'Standard (0.1-100K Hz)':
            setMinFreq(0.1);
            setMaxFreq(100000);
            break;
          case 'Extended (0.01-1M Hz)':
            setMinFreq(0.01);
            setMaxFreq(1000000);
            break;
          case 'High Resolution (1-10K Hz)':
            setMinFreq(1);
            setMaxFreq(10000);
            break;
        }
      }

      // Generate ModelSnapshots for visualization using intelligent limits - NO RECOMPUTATION NEEDED
      console.log('Generating model snapshots from existing serialized data (no recomputation)');
      const smartLimit = calculateEffectiveVisualizationLimit(metadata.totalResults);
      const effectiveLimit = Math.min(smartLimit, metadata.totalResults);
      const modelSnapshots = manager.generateModelSnapshots(effectiveLimit);
      setGridResults(modelSnapshots);

      // Generate resnorm groups for visualization (required for spider plot display)
      if (modelSnapshots.length > 0) {
        const resnormValues = modelSnapshots.map(m => m.resnorm).filter((val): val is number => val !== undefined);
        const minResnorm = resnormValues.length > 0 ? Math.min(...resnormValues) : 0;
        const maxResnorm = resnormValues.length > 0 ? Math.max(...resnormValues) : 1;
        const groups = createResnormGroups(modelSnapshots, minResnorm, maxResnorm);
        setResnormGroups(groups);

      }

      // Get first model parameters as reference circuit parameters
      if (modelSnapshots.length > 0) {
        const bestModel = modelSnapshots[0]; // First model should be best (lowest resnorm)
        setParameters(bestModel.parameters);
      }

      // Update statistics
      setTotalGridPoints(Math.pow(metadata.gridSize, 5));
      setActualComputedPoints(metadata.totalResults);

      // Auto-save as profile with file icon indicator
      const profileName = `📄 ${metadata.title}`;
      const profileDescription = `Loaded from SRD file: ${metadata.totalResults.toLocaleString()} precomputed results (Grid: ${metadata.gridSize}^5)`;

      // Schedule profile saving for after this callback completes (only if user is authenticated)
      setTimeout(async () => {
        if (!sessionManagement.sessionState.userId) {
          console.log('SRD upload: User not authenticated, skipping auto-save profile creation');
          updateStatusMessage('SRD loaded successfully - sign in to save configurations');
          return;
        }

        try {
          const savedProfileId = await handleSaveProfile(profileName, profileDescription, true); // Force new profile
          if (savedProfileId) {
            // Set as active config to maintain tagged models and other data
            setActiveConfiguration(savedProfileId);
            console.log(`Auto-saved SRD config as profile: ${savedProfileId}`);
          }
        } catch (saveError) {
          console.warn('Failed to auto-save SRD profile:', saveError);
        }
      }, 100); // Small delay to ensure handleSaveProfile is available

      // Update status
      const memoryReduction = manager.getStorageStats().reductionFactor;
      updateStatusMessage(`SRD uploaded: "${metadata.title}" - ${modelSnapshots.length.toLocaleString()}/${metadata.totalResults.toLocaleString()} models loaded (NO RECOMPUTATION - ${memoryReduction.toFixed(1)}x memory optimized)`);
      setSrdUploadMessage(`Successfully loaded ${metadata.totalResults.toLocaleString()} results from "${metadata.title}" without recomputation`);

      // Clear any previous errors
      setGridError('');

      console.log(`SRD Upload Success:`, {
        title: metadata.title,
        totalResults: metadata.totalResults,
        gridSize: metadata.gridSize,
        visibleModels: modelSnapshots.length,
        memoryReduction: memoryReduction.toFixed(1) + 'x',
        referenceParams: modelSnapshots[0]?.parameters
      });

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to process uploaded SRD data';
      setGridError(errorMessage);
      updateStatusMessage(`❌ SRD upload failed: ${errorMessage}`);
      setSrdUploadMessage('');
      console.error('SRD upload processing error:', error);
    }
  }, [visibleResultsLimit, setGridResults, setTotalGridPoints, setActualComputedPoints, setGridError, updateStatusMessage, setGridSize, setMinFreq, setMaxFreq, setParameters, setResnormGroups, createResnormGroups, calculateEffectiveVisualizationLimit, setActiveTab]); // handleSaveProfile used in setTimeout closure intentionally not in deps

  // Enhanced file upload handler for JSON, SRD, and other serialized formats
  const _handleEnhancedFileUpload = useCallback(async (file: File) => { // eslint-disable-line @typescript-eslint/no-unused-vars
    try {
      console.log(`📁 Processing file upload: ${file.name}`);

      const importResult = await enhancedSerializedManager.importSerializedData(file);

      if (importResult.success) {
        // Update grid size to match imported data (with persistence)
        updateGridSizeWithPersistence(importResult.gridSize);

        // Generate models from imported data - NO RECOMPUTATION
        const modelSnapshots = enhancedSerializedManager.generateModelSnapshots();
        setGridResults(modelSnapshots);

        // Generate resnorm groups for visualization
        if (modelSnapshots.length > 0) {
          const resnormValues = modelSnapshots.map(m => m.resnorm).filter((val): val is number => val !== undefined);
          const minResnorm = resnormValues.length > 0 ? Math.min(...resnormValues) : 0;
          const maxResnorm = resnormValues.length > 0 ? Math.max(...resnormValues) : 1;
          const groups = createResnormGroups(modelSnapshots, minResnorm, maxResnorm);
          setResnormGroups(groups);
        }

        // Update status with success message
        const compressionInfo = importResult.compressionInfo
          ? ` (${importResult.compressionInfo.ratio.toFixed(1)}x compression)`
          : '';

        updateStatusMessage(`✅ Imported ${importResult.dataCount.toLocaleString()} models from ${file.name} (NO RECOMPUTATION)${compressionInfo}`);

        console.log(`Enhanced import successful: ${importResult.dataCount.toLocaleString()} models from ${file.name}`);
      } else {
        throw new Error(importResult.error || 'Import failed');
      }
    } catch (error) {
      console.error('Enhanced file upload failed:', error);
      handleSRDUploadError(error instanceof Error ? error.message : 'File upload failed');
    }
  }, [enhancedSerializedManager, updateGridSizeWithPersistence, createResnormGroups, setActiveTab, updateStatusMessage]);

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const handleSRDUploadError = useCallback((error: string) => {
    setGridError(`SRD Upload Error: ${error}`);
    updateStatusMessage(`❌ SRD upload error: ${error}`);
    setSrdUploadMessage('');
    console.error('SRD upload error:', error);
  }, [setGridError, updateStatusMessage]);


  // Helper function to generate unique circuit names
  const generateUniqueCircuitName = useCallback((baseName: string): string => {
    if (!circuitConfigurations) return baseName;
    
    const existingNames = circuitConfigurations.map(config => config.name.toLowerCase());
    const baseNameLower = baseName.toLowerCase();
    
    // If the base name doesn't exist, return it
    if (!existingNames.includes(baseNameLower)) {
      return baseName;
    }
    
    // Find the next available number
    let counter = 1;
    let uniqueName = `${baseName} (${counter})`;
    
    while (existingNames.includes(uniqueName.toLowerCase())) {
      counter++;
      uniqueName = `${baseName} (${counter})`;
    }
    
    return uniqueName;
  }, [circuitConfigurations]);
  
  // Save configuration name when it makes logical sense
  const saveConfigurationNameIfNeeded = useCallback(async (nameToSave?: string) => {
    const finalName = nameToSave || configurationName;

    if (activeConfigId && activeConfiguration && finalName.trim() && finalName.trim() !== activeConfiguration.name) {
      try {
        const success = await updateConfiguration(activeConfigId, {
          name: finalName.trim(),
          description: activeConfiguration.description,
          circuitParameters: parameters,
          gridSize,
          minFreq,
          maxFreq,
          numPoints
        });

        if (success) {
          console.log('Configuration name saved successfully');
          return true;
        } else {
          console.error('Failed to save configuration name');
          return false;
        }
      } catch (error) {
        console.error('Error saving configuration name:', error);
        return false;
      }
    }
    return true;
  }, [activeConfigId, activeConfiguration, configurationName, parameters, gridSize, minFreq, maxFreq, numPoints, updateConfiguration]);

  // Moved handleNewCircuit after handleSaveProfile definition

  // Compute grid function
  const _handleCompute = useCallback(async () => {
    if (isComputingGrid) {
      updateStatusMessage('Computation already in progress');
      return;
    }

    console.log('Starting handleCompute with parameters:', {
      parameters,
      gridSize,
      minFreq,
      maxFreq,
      numPoints,
      isComputingGrid
    });

    try {
      // Save configuration name before starting computation
      await saveConfigurationNameIfNeeded();

      // Ensure there's always a profile before starting computation
      if (!savedProfilesState.selectedProfile) {
        console.log('No profile selected - creating one before computation...');
        const preComputeName = configurationName.trim() || `Pre-compute ${new Date().toLocaleString('en-US', { 
          month: 'short', 
          day: 'numeric', 
          hour: '2-digit', 
          minute: '2-digit',
          hour12: false 
        })}`;
        
        try {
          const preComputeProfileId = await handleSaveProfile(
            preComputeName,
            'Last action: Profile created for grid computation'
          );
          
          if (preComputeProfileId) {
            console.log('Pre-compute profile created:', preComputeProfileId);
            updateStatusMessage(`Profile "${preComputeName}" created - starting computation...`);
          } else {
            console.log('WARNING: Pre-compute profile creation failed - proceeding with computation anyway');
          }
        } catch (error) {
          console.error('Pre-compute profile creation error:', error);
          updateStatusMessage('Profile creation failed - proceeding with computation...');
        }
      }
      
      setIsComputingGrid(true);
      setGridError(null);
      setComputationProgress(null);
      setComputationSummary(null);
      
      updateStatusMessage(`Starting grid computation with ${gridSize}x${gridSize} grid...`);

      // AGGRESSIVE DEBUG: Check and force 100% limit before computation
      const expectedFullGrid = Math.pow(gridSize, 5);
      const masterLimitToUse = centralizedLimits.masterLimitPercentage !== 100 ?
        expectedFullGrid : // Force full grid if not 100%
        centralizedLimits.masterLimitResults;

      console.log(`COMPUTATION DEBUG: masterLimitPercentage=${centralizedLimits.masterLimitPercentage}%, masterLimitResults=${centralizedLimits.masterLimitResults.toLocaleString()}, expected=${expectedFullGrid.toLocaleString()}, using=${masterLimitToUse.toLocaleString()}`);

      if (centralizedLimits.masterLimitPercentage !== 100) {
        console.error(`🚨 FORCING 100% COMPUTATION: Detected ${centralizedLimits.masterLimitPercentage}% limit, forcing full ${expectedFullGrid.toLocaleString()} results`);
      }

      const hybridResult = await hybridComputeManager.computeGridHybrid(
        parameters,
        gridSize,
        minFreq,
        maxFreq,
        numPoints,
        performanceSettings,
        extendedSettings,
        resnormConfig,
        (progress) => {
          setComputationProgress(progress);
          if (progress.message) {
            updateStatusMessage(progress.message);
          }
        },
        (error) => {
          setGridError(error);
          updateStatusMessage(`Computation error: ${error}`);
        },
        masterLimitToUse // Force full computation if needed
      );
      
      const results = hybridResult.results;

      if (results && results.length > 0) {
        // Process results - results is directly an array, not an object with topResults
        const processedResults: BackendMeshPoint[] = results.map((result, index) => ({
          id: index,
          parameters: result.parameters,
          resnorm: result.resnorm,
          // Keep spectrum format as-is since BackendMeshPoint expects {freq, imag, mag} format
          spectrum: result.spectrum || [],
          isReference: false
        }));

        // MEMORY OPTIMIZATION: Ultra-compact serialized storage with lazy loading
        if (serializedManager && memoryOptimizationEnabled) {
          // Clear previous results to prevent memory accumulation
          serializedManager.clearCaches();

          // Store results in ultra-compact format (98% memory reduction)
          const serializedCount = serializedManager.storeResults(processedResults, 'standard');
          const stats = serializedManager.getStorageStats();

          console.log(`MEMORY OPTIMIZED: ${serializedCount} results stored`);
          console.log(`Memory: ${stats.traditionalSizeMB.toFixed(1)}MB → ${stats.serializedSizeMB.toFixed(1)}MB (${stats.reductionFactor.toFixed(1)}x reduction)`);

          // Generate results using intelligent visualization limits
          const smartLimit = calculateEffectiveVisualizationLimit(serializedCount);
          const effectiveLimit = Math.min(smartLimit, serializedCount);
          const modelSnapshots = serializedManager.generateModelSnapshots(effectiveLimit);

          setGridResults(modelSnapshots);

          // Force garbage collection of large arrays
          processedResults.length = 0;
          console.log(`🧹 Cleared ${processedResults.length} large arrays for memory cleanup`);

          updateStatusMessage(`✅ Optimized computation: ${effectiveLimit}/${serializedCount} models (${stats.reductionFactor.toFixed(1)}x memory saved)`);
        } else {
          // Legacy mode or disabled optimization
          console.warn('⚠️ Using legacy memory model - consider enabling optimization for large grids');
          const smartLimit = calculateEffectiveVisualizationLimit(processedResults.length);
          const limitedResults = processedResults.slice(0, Math.min(smartLimit, processedResults.length));
          const modelSnapshots = limitedResults.map((result, index) => mapBackendMeshToSnapshot(result, index));
          setGridResults(modelSnapshots);
          updateStatusMessage(`Legacy storage: ${modelSnapshots.length} models loaded (memory optimization disabled)`);
        }
        
        // Update computation statistics
        const expectedGridPoints = Math.pow(gridSize, 5);
        setTotalGridPoints(expectedGridPoints);
        setActualComputedPoints(results.length);

        console.log(`COMPUTATION COMPLETE: gridSize=${gridSize}, expected=${expectedGridPoints}, computed=${results.length}, previous_actualComputedPoints=${actualComputedPoints}`);

        if (results.length !== actualComputedPoints) {
          console.warn(`MISMATCH: Final results (${results.length}) != generated count (${actualComputedPoints})`);
        }
        
        // Grid parameter arrays removed - functionality moved to visualization layer

        console.log('Computation successful:', {
          processedResultsLength: processedResults.length,
          sampleResult: processedResults[0]
        });
        
        // Mark current profile as computed if one is selected
        if (savedProfilesState.selectedProfile) {
          await profileActions.updateProfile(savedProfilesState.selectedProfile, { isComputed: true });
        }

        // Automatically save grid configuration to localStorage after successful computation
        const gridConfig = {
          gridSize,
          minFreq,
          maxFreq,
          numPoints,
          parameters,
          computedAt: new Date().toISOString(),
          totalModels: processedResults.length
        };
        localStorage.setItem('lastGridConfig', JSON.stringify(gridConfig));
        console.log('Grid configuration automatically saved to localStorage');

        // Update URL to include computed config ID without causing remount
        // Use query params instead of route change to preserve state
        if (activeConfigId) {
          const newUrl = `/simulator?computed=${activeConfigId}`;
          router.push(newUrl, { scroll: false });
        }

        // Also create a backup profile entry in localStorage as failsafe
        try {
          const backupProfiles = JSON.parse(localStorage.getItem('backup-nei-profiles') || '[]');
          const backupProfile = {
            id: `backup_${Date.now()}`,
            name: `Backup ${new Date().toLocaleString()}`,
            description: `Backup from computation at ${new Date().toISOString()}`,
            groundTruthParams: parameters,
            gridSize,
            minFreq,
            maxFreq,
            numPoints,
            created: Date.now(),
            lastModified: Date.now(),
            isComputed: true,
            totalModels: processedResults.length
          };
          backupProfiles.unshift(backupProfile);
          // Keep only last 10 backups
          if (backupProfiles.length > 10) {
            backupProfiles.splice(10);
          }
          localStorage.setItem('backup-nei-profiles', JSON.stringify(backupProfiles));
          console.log('Backup profile created in localStorage');
        } catch (err) {
          console.warn('Failed to create backup profile:', err);
        }
      } else {
        throw new Error('No results returned from computation');
      }

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';

      // Enhanced error tracking with memory context
      const memoryInfo = {
        gridSize,
        totalPoints: Math.pow(gridSize, 5),
        estimatedMemory: Math.pow(gridSize, 5) * 800 / (1024 * 1024),
        memoryOptimization: memoryOptimizationEnabled,
        timestamp: new Date().toISOString()
      };

      console.error('Computation Error Details:', {
        error: errorMessage,
        memoryContext: memoryInfo,
        parameters,
        serializedManagerAvailable: !!serializedManager
      });

      // Clear any partial results to prevent memory leaks
      if (serializedManager) {
        serializedManager.clearCaches();
      }

      setGridError(errorMessage);
      updateStatusMessage(`⚠️ Computation failed: ${errorMessage} (Grid: ${gridSize}^5, Est. Memory: ${memoryInfo.estimatedMemory.toFixed(1)}MB)`);
    } finally {
      setIsComputingGrid(false);

      // Force garbage collection attempt
      if (typeof window !== 'undefined' && window.gc) {
        window.gc();
        console.log('Forced garbage collection after computation');
      }
    }
  }, [
    isComputingGrid, gridSize, parameters, minFreq, maxFreq, numPoints,
    performanceSettings, resnormConfig, hybridComputeManager.computeGridHybrid,
    setIsComputingGrid, setGridError, setComputationProgress, setComputationSummary,
    setGridResults, setTotalGridPoints, setActualComputedPoints,
    savedProfilesState.selectedProfile, profileActions,
    updateStatusMessage, configurationName, extendedSettings,
    hybridComputeManager, centralizedLimits.masterLimitResults, saveConfigurationNameIfNeeded
  ]);
  

  // Multi-select functions
  const _handleToggleMultiSelect = useCallback(() => {
    setIsMultiSelectMode(!isMultiSelectMode);
    setSelectedCircuits([]);
  }, [isMultiSelectMode]);

  const _handleSelectCircuit = useCallback(async (configId: string) => {
    if (isMultiSelectMode) {
      const newSelectedCircuits = selectedCircuits.includes(configId)
        ? selectedCircuits.filter(id => id !== configId)
        : [...selectedCircuits, configId];
      setSelectedCircuits(newSelectedCircuits);
    } else {
      // Auto-save current profile before switching (if there's an active one)
      if (activeConfigId && activeConfigId !== configId && sessionManagement.sessionState.userId) {
        try {
          console.log('Auto-saving current profile before switching...', activeConfigId);
          const success = await updateConfiguration(activeConfigId, {
            circuitParameters: parameters,
            gridSize,
            minFreq,
            maxFreq,
            numPoints
          });

          if (success) {
            console.log('Profile auto-saved before switch');
            updateStatusMessage(`Auto-saved changes to current profile before switching`);
          } else {
            console.warn('Failed to auto-save before switching - proceeding anyway');
            updateStatusMessage(`Could not auto-save before switching`);
          }
        } catch (error) {
          const errorMsg = error instanceof Error ? error.message : 'Unknown error';
          console.warn('Failed to auto-save before profile switch:', errorMsg);
          updateStatusMessage(`Auto-save failed: ${errorMsg}`);
        }
      }

      // Normal single selection behavior for regular profiles
      // Clear existing grid results when switching configurations
      resetComputationState();

      // Set active configuration in both hooks
      setActiveConfiguration(configId);
      sessionManagement.actions.setActiveCircuitConfig(configId);

      const config = circuitConfigurations?.find(c => c.id === configId);
      if (config) {
        // Load configuration settings into current state
        setGridSize(config.gridSize);
        // NOTE: Do NOT call updateDefaultGridSize here
        setMinFreq(config.minFreq);
        setMaxFreq(config.maxFreq);
        setNumPoints(config.numPoints);
        setParameters(config.circuitParameters);

        // Auto-fill configuration name with config name
        setConfigurationName(config.name);

        // Update frequencies based on loaded settings (direct update instead of using updateFrequencies)
        setMinFreq(config.minFreq);
        setMaxFreq(config.maxFreq);
        setNumPoints(config.numPoints);

        // Mark parameters as changed to enable recompute
        setParameterChanged(true);

        updateStatusMessage(`Loaded configuration: ${config.name} - Previous grid data cleared, ready to compute`);

        console.log('Loaded profile settings:', {
          name: config.name,
          parameters: config.circuitParameters,
          gridSize: config.gridSize,
          frequency: { min: config.minFreq, max: config.maxFreq, points: config.numPoints }
        });
      } else {
        console.warn('Configuration not found:', configId);
        updateStatusMessage(`Configuration not found`);
      }
    }
  }, [isMultiSelectMode, selectedCircuits, activeConfigId, sessionManagement.sessionState.userId, updateConfiguration, parameters, gridSize, minFreq, maxFreq, numPoints, resetComputationState, setActiveConfiguration, sessionManagement.actions, circuitConfigurations]);

  const _handleBulkDelete = useCallback(async () => {
    if (selectedCircuits.length === 0) return;
    
    // Check if currently selected configuration is being deleted
    const wasCurrentlySelectedDeleted = selectedCircuits.includes(activeConfigId || '');
    if (wasCurrentlySelectedDeleted) {
      resetComputationState();
    }
    
    // Remove selected circuit configurations
    const success = await deleteMultipleConfigurations(selectedCircuits);
    
    if (success) {
      if (wasCurrentlySelectedDeleted) {
        setActiveConfiguration(null);
        // Note: SessionManagement doesn't have a clear method, so we only set local state to null
      }
      
      const deletedCount = selectedCircuits.length;
      setSelectedCircuits([]);
      setIsMultiSelectMode(false);
      
      updateStatusMessage(
        wasCurrentlySelectedDeleted 
          ? `${deletedCount} circuit${deletedCount > 1 ? 's' : ''} deleted and grid data cleared`
          : `${deletedCount} circuit${deletedCount > 1 ? 's' : ''} deleted`
      );
    } else {
      updateStatusMessage('Failed to delete selected circuits');
    }
  }, [selectedCircuits, activeConfigId, deleteMultipleConfigurations]);
  
  // Current memory usage for performance monitoring
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const getCurrentMemoryUsage = useCallback((): number => {
    if (typeof window !== 'undefined' && 'performance' in window && 'memory' in performance) {
      const memory = (performance as typeof performance & { memory: { usedJSHeapSize: number } }).memory;
      return memory.usedJSHeapSize / (1024 * 1024);
    }
    return 0;
  }, []);
  
  // Initialize parameters only when no active profile is loaded
  useEffect(() => {
    // Only initialize defaults if no active profile is selected
    if (activeConfigId || sessionManagement.sessionState.currentCircuitConfigId) {
      console.log('Skipping parameter initialization - active profile loaded:', activeConfigId || sessionManagement.sessionState.currentCircuitConfigId);
      return;
    }

    // Calculate 50% values for each parameter range
    const rsRange = { min: 10, max: 10000 };
    const raRange = { min: 10, max: 10000 };
    const rbRange = { min: 10, max: 10000 };
    const caRange = { min: 0.1e-6, max: 50e-6 };
    const cbRange = { min: 0.1e-6, max: 50e-6 };

    const rs50 = rsRange.min + (rsRange.max - rsRange.min) * 0.5;
    const ra50 = raRange.min + (raRange.max - raRange.min) * 0.5;
    const rb50 = rbRange.min + (rbRange.max - rbRange.min) * 0.5;
    const ca50 = caRange.min + (caRange.max - caRange.min) * 0.5;
    const cb50 = cbRange.min + (cbRange.max - cbRange.min) * 0.5;

    console.log('Initializing default parameters (no active profile)');
    setParameters({
      Rsh: rs50,
      Ra: ra50,
      Ca: ca50,
      Rb: rb50,
      Cb: cb50,
      frequency_range: [minFreq, maxFreq]
    });

    updateStatusMessage('Initialized with default parameters at 50% of ranges');
  }, [updateStatusMessage, minFreq, maxFreq, activeConfigId, sessionManagement.sessionState.currentCircuitConfigId]);

  // Update reference parameters when slider values are initialized
  useEffect(() => {
    // Only create reference once on initial load
    if (referenceParams.Rsh === 24 && 
        referenceParams.Ra === 500 && 
        referenceParams.Ca === 0.5e-6 && 
        referenceParams.Rb === 500 && 
        referenceParams.Cb === 0.5e-6 && 
        referenceParams.frequency_range.length === 2 &&
        referenceParams.frequency_range[0] === minFreq &&
        referenceParams.frequency_range[1] === maxFreq) {
      // Create the reference object directly without calling the function
      setReferenceParams({
        Rsh: parameters.Rsh,
        Ra: parameters.Ra,
        Ca: parameters.Ca,
        Rb: parameters.Rb,
        Cb: parameters.Cb,
        frequency_range: parameters.frequency_range
      });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);  // Run only once on mount
  
  // We'll calculate TER and TEC in the component where they're needed
  
  // Create a reference model (current circuit parameters)
  const createReferenceModel = useCallback((): ModelSnapshot => {
    // Generate frequencies to compute impedance at (logarithmically spaced)
    let freqs: number[];
    if (frequencyPoints.length > 0) {
      // Use the pre-calculated frequency points if available
      freqs = [...frequencyPoints];
    } else {
      // Generate logarithmically spaced points from min to max
      freqs = [];
      const logMin = Math.log10(minFreq);
      const logMax = Math.log10(maxFreq);
      const logStep = (logMax - logMin) / (numPoints - 1);
      
      for (let i = 0; i < numPoints; i++) {
        const logValue = logMin + i * logStep;
        const frequency = Math.pow(10, logValue);
        freqs.push(frequency);
      }
      
      // Don't update state here - this would cause re-renders
      // Instead, just use the locally calculated frequencies
    }
    
    // Use user-controlled parameters for the reference model (ground truth)
    const { Rsh, Ra, Ca, Rb, Cb } = parameters;
    
    // Compute impedance at each frequency
    const impedanceData: ImpedancePoint[] = [];
    
    // Generate detailed logs for the first calculation (avoid calling updateStatusMessage in callback)
    
    for (const f of freqs) {
      // Calculate impedance for a single frequency
      const impedance = calculateCircuitImpedance(
        { Rsh, Ra, Ca, Rb, Cb, frequency_range: [minFreq, maxFreq] }, 
        f
      );
      
      // Log removed to avoid circular dependencies
      
      impedanceData.push({
        frequency: f,
        real: impedance.real,
        imaginary: impedance.imaginary,
        magnitude: Math.sqrt(impedance.real * impedance.real + impedance.imaginary * impedance.imaginary),
        phase: Math.atan2(impedance.imaginary, impedance.real) * (180 / Math.PI)
      });
    }
    
    return {
      id: 'dynamic-reference',
      name: 'Reference',
      parameters: { Rsh, Ra, Ca, Rb, Cb, frequency_range: [minFreq, maxFreq] },
      data: impedanceData,
      color: '#FFFFFF',
      isVisible: true,
      opacity: 1,
      resnorm: 0,
      timestamp: Date.now(),
      ter: Rsh + Ra + Rb
    };
  }, [parameters, minFreq, maxFreq, numPoints, frequencyPoints]);

  // Function to calculate impedance at a single frequency
  const calculateCircuitImpedance = (params: CircuitParameters, frequency: number) => {
    const { Rsh, Ra, Ca, Rb, Cb } = params;
    const omega = 2 * Math.PI * frequency;
    
    // Calculate impedance of apical membrane (Ra || Ca)
    // Za(ω) = Ra/(1+jωRaCa)
    const Za_real = Ra / (1 + Math.pow(omega * Ra * Ca, 2));
    const Za_imag = -omega * Ra * Ra * Ca / (1 + Math.pow(omega * Ra * Ca, 2));
    
    // Calculate impedance of basal membrane (Rb || Cb)
    // Zb(ω) = Rb/(1+jωRbCb)
    const Zb_real = Rb / (1 + Math.pow(omega * Rb * Cb, 2));
    const Zb_imag = -omega * Rb * Rb * Cb / (1 + Math.pow(omega * Rb * Cb, 2));
    
    // Calculate sum of membrane impedances (Za + Zb)
    const Zab_real = Za_real + Zb_real;
    const Zab_imag = Za_imag + Zb_imag;
    
    // Calculate parallel combination: Z_total = (Rsh * (Za + Zb)) / (Rsh + Za + Zb)
    // Numerator: Rsh * (Za + Zb)
    const num_real = Rsh * Zab_real;
    const num_imag = Rsh * Zab_imag;
    
    // Denominator: Rsh + Za + Zb
    const denom_real = Rsh + Zab_real;
    const denom_imag = Zab_imag;
    
    // Complex division: (num_real + j*num_imag) / (denom_real + j*denom_imag)
    const denom_mag_squared = denom_real * denom_real + denom_imag * denom_imag;
    
    const real = (num_real * denom_real + num_imag * denom_imag) / denom_mag_squared;
    const imaginary = (num_imag * denom_real - num_real * denom_imag) / denom_mag_squared;
    
    return { real, imaginary };
  };

  // Function to handle reference model toggle
  const toggleReferenceModel = useCallback(() => {
    if (referenceModelId === 'dynamic-reference') {
      // Hide reference model and mark as manually hidden
      setReferenceModelId(null);
      setReferenceModel(null);
      setManuallyHidden(true);
      updateStatusMessage('Reference model hidden');
    } else {
      // Show reference model - create a new one
      const newReferenceModel = createReferenceModel();
      setReferenceModel(newReferenceModel);
      setReferenceModelId('dynamic-reference');
      setManuallyHidden(false);
      updateStatusMessage('Reference model shown');
    }
  }, [referenceModelId, createReferenceModel, updateStatusMessage, setManuallyHidden]);

  // Create and show reference model by default
  useEffect(() => {
    // Create the reference model when parameters are initialized
    if (parameters.Rsh && !referenceModelId && !manuallyHidden) {
      const newReferenceModel = createReferenceModel();
      setReferenceModel(newReferenceModel);
      setReferenceModelId('dynamic-reference');
      updateStatusMessage('Reference model created with current parameters');
    }
  }, [parameters.Rsh, referenceModelId, createReferenceModel, manuallyHidden, updateStatusMessage]);

  // Update reference model when parameters change
  useEffect(() => {
    if (referenceModelId === 'dynamic-reference' && !manuallyHidden) {
      const updatedModel = createReferenceModel();
      setReferenceModel(updatedModel);
      updateStatusMessage('Reference model updated with current parameters');
    }
  }, [parameters, createReferenceModel, referenceModelId, manuallyHidden, updateStatusMessage]);

  // Authentication effect - show modal when not authenticated
  useEffect(() => {
    if (!authLoading && !user) {
      setShowAuthModal(true);
    }
  }, [user, authLoading]);

  // Synchronize configurationName and ALL parameters with activeConfiguration
  // This ensures center panel always reflects the selected circuit from sidebar
  useEffect(() => {
    if (activeConfiguration) {
      console.log('[SYNC] Loading parameters from activeConfiguration:', {
        id: activeConfiguration.id,
        name: activeConfiguration.name,
        parameters: activeConfiguration.circuitParameters
      });

      // Sync configuration name
      setConfigurationName(activeConfiguration.name);

      // Sync all circuit parameters to center panel
      setParameters(activeConfiguration.circuitParameters);
      setGridSize(activeConfiguration.gridSize);
      // NOTE: Do NOT call updateDefaultGridSize here - it causes infinite loops
      // The default grid size should only be set from user profile, not from circuits
      setMinFreq(activeConfiguration.minFreq);
      setMaxFreq(activeConfiguration.maxFreq);
      setNumPoints(activeConfiguration.numPoints);

      // Update frequencies
      updateFrequencies(activeConfiguration.minFreq, activeConfiguration.maxFreq, activeConfiguration.numPoints);
    }
    // Only run when activeConfiguration.id changes to avoid infinite loops
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeConfiguration?.id]);

  // Synchronize session's active circuit config with local state
  // Use a ref to track if we're in the middle of a user-initiated change
  const isSyncingRef = React.useRef(false);

  useEffect(() => {
    const syncActiveConfig = async () => {
      // Prevent re-entry during sync operations
      if (isSyncingRef.current) {
        return;
      }

      const sessionConfigId = sessionManagement.sessionState.currentCircuitConfigId;

      // Case 1: Session has config but local state doesn't - sync to local
      if (sessionConfigId && !activeConfigId) {
        console.log('Syncing session config to local state:', sessionConfigId);
        isSyncingRef.current = true;
        setActiveConfiguration(sessionConfigId);
        setTimeout(() => { isSyncingRef.current = false; }, 100);
        return;
      }

      // Case 2: Local has config but session doesn't - sync to session
      if (activeConfigId && !sessionConfigId) {
        console.log('Syncing local config to session:', activeConfigId);
        isSyncingRef.current = true;
        await sessionManagement.actions.setActiveCircuitConfig(activeConfigId);
        setTimeout(() => { isSyncingRef.current = false; }, 100);
        return;
      }

      // Case 3: Both exist but differ - LOCAL wins (user selection takes precedence)
      // Session sync should only happen on initial load, not override user selections
      if (sessionConfigId && activeConfigId && sessionConfigId !== activeConfigId) {
        console.log('Config mismatch detected - keeping local selection:', activeConfigId);
        isSyncingRef.current = true;
        await sessionManagement.actions.setActiveCircuitConfig(activeConfigId);
        setTimeout(() => { isSyncingRef.current = false; }, 100);
        return;
      }

      // Case 4: Both match or both are null - no action needed
    };

    // Only sync if session is ready
    if (sessionManagement.isReady) {
      syncActiveConfig();
    }
  }, [sessionManagement.sessionState.currentCircuitConfigId, activeConfigId, sessionManagement.isReady, sessionManagement.actions, setActiveConfiguration]);
  
  // Add event listeners for custom events
  useEffect(() => {
    // Handler for reference model toggle
    const handleToggleReference = (event: Event) => {
      // Check if this is a forced state change
      const customEvent = event as CustomEvent;
      const forceState = customEvent.detail?.forceState;
      
      if (forceState !== undefined) {
        // Handle forced state
        if (forceState) {
          // Force show reference model
          if (!referenceModelId) {
            const newReferenceModel = createReferenceModel();
            setReferenceModel(newReferenceModel);
            setReferenceModelId('dynamic-reference');
            setManuallyHidden(false);
            updateStatusMessage('Reference model shown');
          }
        } else {
          // Force hide reference model
          if (referenceModelId) {
            setReferenceModelId(null);
            setReferenceModel(null);
            setManuallyHidden(true);
            updateStatusMessage('Reference model hidden');
          }
        }
      } else {
        // Regular toggle
        toggleReferenceModel();
      }
    };
    
    // Handler for resnorm group toggle
    const handleToggleResnormGroup = (event: Event) => {
      const customEvent = event as CustomEvent;
      const { groupIndex } = customEvent.detail;
      
      // Toggle this group's visibility
      setHiddenGroups(prev => {
        if (prev.includes(groupIndex)) {
          return prev.filter(i => i !== groupIndex);
        } else {
          return [...prev, groupIndex];
        }
      });
      
      // Update status message
      const groupName = 
        groupIndex === 0 ? 'Very Good Fit' : 
        groupIndex === 1 ? 'Good Fit' : 
        groupIndex === 2 ? 'Moderate Fit' : 'Poor Fit';
      
      updateStatusMessage(
        hiddenGroups.includes(groupIndex) 
          ? `Showing ${groupName} category`
          : `Hiding ${groupName} category`
      );
    };
    
    // Add event listeners
    window.addEventListener('toggleReferenceModel', handleToggleReference);
    window.addEventListener('toggleResnormGroup', handleToggleResnormGroup);
    
    // Clean up event listeners on unmount
    return () => {
      window.removeEventListener('toggleReferenceModel', handleToggleReference);
      window.removeEventListener('toggleResnormGroup', handleToggleResnormGroup);
    };
  }, [toggleReferenceModel, hiddenGroups, referenceModelId, createReferenceModel, updateStatusMessage, setHiddenGroups, setManuallyHidden]);
  
  // Optimized helper function to map BackendMeshPoint to ModelSnapshot
  const mapBackendMeshToSnapshot = (meshPoint: BackendMeshPoint, index: number): ModelSnapshot => {
    // Extract parameters efficiently (they should already be in Farads)
    const { Rsh, Ra, Ca, Rb, Cb, frequency_range } = meshPoint.parameters;
    
    // Pre-calculate TER (Total Extracellular Resistance)
    const ter = Ra + Rb;
    
    // Only convert spectrum if it exists and has data
    const spectrumData = meshPoint.spectrum && meshPoint.spectrum.length > 0 
      ? toImpedancePoints(meshPoint.spectrum) as ImpedancePoint[]
      : [];
    
    return {
      id: `mesh-${index}`,
      name: `Mesh Point ${index}`,
      parameters: {
        Rsh,
        Ra,
        Ca,
        Rb,
        Cb,
        frequency_range: frequency_range || parameters.frequency_range
      },
      data: spectrumData,
      color: '#f97316',
      isVisible: true,
      opacity: 1,
      resnorm: meshPoint.resnorm,
      timestamp: Date.now(),
      ter
    };
  };



  // Updated grid computation using Web Workers for parallel processing
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const handleComputeRegressionMesh = useCallback(async () => {
    // Validate grid size
    if (gridSize < 2 || gridSize > 25) {
      updateStatusMessage('Points per dimension must be between 2 and 25');
      return;
    }

    const totalPointsToCompute = Math.pow(gridSize, 5);
    
    // Add warning for very large grids that might affect performance
    if (gridSize > 20) {
      const proceed = window.confirm(
        `⚠️ Warning: You're about to compute ${totalPointsToCompute.toLocaleString()} parameter combinations.\n\n` +
        `This will take several minutes and may make the page temporarily less responsive.\n\n` +
        `Consider using a smaller grid size (15-20) for faster results.\n\n` +
        `Continue with computation?`
      );
      if (!proceed) {
        updateStatusMessage('Grid computation cancelled by user');
        return;
      }
    }

    // Start comprehensive performance tracking
    const startTime = Date.now();
    let generationStartTime = 0;
    let computationStartTime = 0;
    let processingStartTime = 0;

    // Reset state and start performance tracking
    setIsComputingGrid(true);
    
    // Save current state before starting new computation
    saveComputationState();
    resetComputationState();
    setParameterChanged(false);
    setManuallyHidden(false);
    setCurrentPhases([]);
    
    // Initialize computation phase
    startPhase('initialization', 'Setting up parallel computation pipeline');
    
    // Set initial grid tracking
    setTotalGridPoints(totalPointsToCompute);
    
    const totalPoints = totalPointsToCompute;
    updateStatusMessage(`Starting parallel grid computation with ${gridSize} points per dimension...`);
    updateStatusMessage(`MATH: Computing all ${totalPoints.toLocaleString()} parameter combinations using ${navigator.hardwareConcurrency || 4} CPU cores`);
    updateStatusMessage(`MATH: Circuit model: Randles equivalent circuit (Rsh shunt resistance with parallel RC elements)`);
    updateStatusMessage(`MATH: Z(ω) = Rsh + Ra/(1+jωRaCa) + Rb/(1+jωRbCb)`);
    updateStatusMessage(`MATH: Using frequency range ${minFreq.toFixed(1)}-${maxFreq.toFixed(1)} Hz with ${numPoints} points`);

    try {
      // Enhanced progress callback with UI coordination for large datasets
      const handleProgress = (progress: WorkerProgress) => {
        // For high-load operations, queue updates and process with requestAnimationFrame
        if (progress.mainThreadLoad === 'high' || progress.total > 100000) {
          // Queue the update
          progressUpdateQueueRef.current.push(progress);
          
          // Process queue if not already processing
          if (!isProcessingUpdatesRef.current) {
            isProcessingUpdatesRef.current = true;
            
            requestAnimationFrame(() => {
              // Process latest update from queue
              const latestProgress = progressUpdateQueueRef.current.pop();
              if (latestProgress) {
                setComputationProgress(latestProgress);
              }
              
              // Clear queue and reset flag
              progressUpdateQueueRef.current = [];
              isProcessingUpdatesRef.current = false;
            });
          }
        } else {
          // Direct update for smaller datasets
          setComputationProgress(progress);
        }
        
        // Enhanced progress handling with detailed logging for large datasets
        if (progress.type === 'STREAMING_UPDATE' || progress.type === 'THROTTLE_UPDATE') {
          // Use requestAnimationFrame for smooth UI updates
          requestAnimationFrame(() => {
            setComputationProgress(progress);
          });
          
          // Enhanced logging for large dataset computations
          if (progress.mainThreadLoad === 'high') {
            if (progress.streamingBatch && progress.streamingBatch % 2 === 0) {
              updateStatusMessage(`[BATCH ${progress.streamingBatch}] ${progress.message || 'Processing large dataset...'}`);
            }
          } else {
            // Throttle status message updates for better performance
            if (progress.streamingBatch && progress.streamingBatch % 3 === 0) {
              updateStatusMessage(progress.message || 'Processing...');
            }
          }
        } else {
          // Regular progress updates with enhanced detail
          if (progress.message) {
            // Add visual indicators for different progress types
            const icon = progress.type === 'MATHEMATICAL_OPERATION' ? '[MATH]' : 
                        progress.type === 'WORKER_STATUS' ? '[WORK]' : 
                        progress.type === 'GENERATION_PROGRESS' ? '[GEN]' : '[PROC]';
            updateStatusMessage(`${icon} ${progress.message}`);
          }
          
          if (progress.operation) {
            updateStatusMessage(`[${progress.operation.toUpperCase()}] ${progress.message || 'Processing...'}`);
          }
        }
        
        // Handle memory pressure with user feedback
        if (progress.memoryPressure && progress.type === 'THROTTLE_UPDATE') {
          updateStatusMessage(`⚠️ High memory usage - throttling computation to maintain responsiveness`);
        }
        
        if (progress.equation) {
          updateStatusMessage(`[EQUATION] ${progress.equation}`);
        }
        
        if (progress.type === 'GENERATION_PROGRESS') {
          if (generationStartTime === 0) {
            generationStartTime = Date.now();
            completePhase('initialization');
            startPhase('grid-generation', 'Generating parameter space grid combinations');
          }
          if (progress.generated && progress.generated > 0) {
            console.log(`GENERATION DEBUG: generated=${progress.generated}, total=${progress.total}, processed=${progress.processed || 'unknown'}, skipped=${progress.skipped || 'unknown'}`);
            setActualComputedPoints(progress.generated);
            const skipped = progress.skipped || (progress.total - progress.generated);
            setSkippedPoints(skipped);
            updateStatusMessage(`Generating grid points: ${progress.generated.toLocaleString()}/${progress.total.toLocaleString()} (${progress.overallProgress.toFixed(1)}%) [${skipped.toLocaleString()} skipped by symmetric optimization]`);
          }
          
          // Add small delay for large grids to keep UI responsive
          if (totalPoints > 100000 && progress.generated && progress.generated % 100000 === 0) {
            setTimeout(() => {}, 10);
          }
        } else if (progress.type === 'CHUNK_PROGRESS') {
          if (computationStartTime === 0) {
            computationStartTime = Date.now();
            const generationTime = ((computationStartTime - generationStartTime) / 1000).toFixed(2);
            completePhase('grid-generation', { 
              pointsGenerated: progress.total,
              generationTime: generationTime + 's'
            });
            startPhase('impedance-computation', 'Computing complex impedance spectra in parallel');
          }
          const processedStr = progress.processed ? progress.processed.toLocaleString() : '0';
          const totalStr = progress.total.toLocaleString();
          const memoryWarning = progress.memoryPressure ? ' [MEMORY PRESSURE - Implementing backpressure]' : '';
          updateStatusMessage(`Computing spectra & resnorms: ${processedStr}/${totalStr} points (${progress.overallProgress.toFixed(1)}%)${memoryWarning}`);
          
          // Log memory pressure warnings
          if (progress.memoryPressure) {
            updateStatusMessage(`[MEMORY] High memory usage detected - workers implementing staggered processing`);
          }
          
          // Add periodic UI updates for better responsiveness
          if (totalPoints > 500000 && progress.processed && progress.processed % 100000 === 0) {
            setTimeout(() => {}, 5);
          }
        }
      };

      // Error callback for worker errors
      const handleError = (error: string) => {
        setGridError(`Parallel computation failed: ${error}`);
        updateStatusMessage(`Error in parallel computation: ${error}`);
        setIsComputingGrid(false);
      };

      // Create progress adapter for optimized compute manager
      const progressAdapter = (progress: {
        phase: string;
        progress: number;
        currentOperation: string;
        parametersProcessed?: number;
        memoryUsage?: number;
        estimatedTimeRemaining?: number;
      }) => {
        // Convert to WorkerProgress format
        handleProgress({
          type: progress.phase.includes('Stage 1') ? 'GENERATION_PROGRESS' : 
                progress.phase.includes('Stage 2') ? 'CHUNK_PROGRESS' : 
                progress.phase.includes('Stage 3') ? 'MATHEMATICAL_OPERATION' : 'COMPUTATION_START',
          total: 100,
          overallProgress: progress.progress,
          processed: progress.parametersProcessed,
          generated: progress.parametersProcessed
        });
      };
      
      // Run the parallel computation (with optimization if qualified)
      const computeResult = await optimizedComputeManager.computeGridOptimized(
        parameters,
        gridSize,
        minFreq,
        maxFreq,
        numPoints,
        performanceSettings,
        extendedSettings,
        resnormConfig,
        progressAdapter,
        handleError
      );

      const results = computeResult.results;

      if (results.length === 0) {
        throw new Error('No results returned from computation');
      }

      // Start processing phase timing
      processingStartTime = Date.now();
      const computationTime = ((processingStartTime - computationStartTime) / 1000).toFixed(2);
      completePhase('impedance-computation', {
        pointsProcessed: results.length,
        computationTime: computationTime + 's',
        parallelCores: navigator.hardwareConcurrency || 4
      });
      startPhase('resnorm-analysis', 'Analyzing residual norms and grouping results');

      // Sort results by resnorm
      const sortingStart = Date.now();
      const sortedResults = results.sort((a, b) => a.resnorm - b.resnorm);
      const sortingTime = ((Date.now() - sortingStart) / 1000).toFixed(3);
      
      updateStatusMessage(`Parallel computation complete: ${sortedResults.length} points analyzed`);
      updateStatusMessage(`[TIMING] Results sorted by resnorm in ${sortingTime}s`);

      // Calculate resnorm range for grouping
      const groupingStart = Date.now();
      const resnorms = sortedResults.map(p => p.resnorm).filter(r => r > 0);
      
      // Use iterative approach to avoid stack overflow with large arrays
      let minResnorm = Infinity;
      let maxResnorm = -Infinity;
      for (const resnorm of resnorms) {
        if (resnorm < minResnorm) minResnorm = resnorm;
        if (resnorm > maxResnorm) maxResnorm = resnorm;
      }

      // Intelligent memory management for large datasets
      const estimateMemoryUsage = (count: number) => {
        // Rough estimate: each model ~500 bytes + spectrum data
        const avgSpectrumSize = numPoints * 40; // bytes per spectrum point
        return count * (500 + avgSpectrumSize) / 1024 / 1024; // MB
      };

      const estimatedMemory = estimateMemoryUsage(sortedResults.length);
      setEstimatedMemoryUsage(estimatedMemory);
      
      // Use new user-controlled or automatic limit calculation
      const MAX_VISUALIZATION_MODELS = calculateEffectiveVisualizationLimit(sortedResults.length);
      
      // Log memory management strategy
      if (isUserControlledLimits) {
        updateStatusMessage(`User-controlled limits: Displaying ${MAX_VISUALIZATION_MODELS.toLocaleString()}/${sortedResults.length.toLocaleString()} points (${userVisualizationPercentage}%)`);
      } else {
        if (estimatedMemory > 500) {
          updateStatusMessage(`Large dataset detected (${estimatedMemory.toFixed(1)}MB estimated), using automatic aggressive sampling`);
        } else if (estimatedMemory > 200) {
          updateStatusMessage(`Medium dataset detected (${estimatedMemory.toFixed(1)}MB estimated), using automatic moderate sampling`);
        }
      }
      
      const shouldLimitModels = sortedResults.length > MAX_VISUALIZATION_MODELS;
      const modelsToProcess = shouldLimitModels ? sortedResults.slice(0, MAX_VISUALIZATION_MODELS) : sortedResults;
      
      if (shouldLimitModels) {
        setMemoryLimitedPoints(MAX_VISUALIZATION_MODELS);
        updateStatusMessage(`Memory management: Displaying ${MAX_VISUALIZATION_MODELS.toLocaleString()}/${sortedResults.length.toLocaleString()} points (${(sortedResults.length - MAX_VISUALIZATION_MODELS).toLocaleString()} hidden for performance)`);
      }
      
      updateStatusMessage(`Processing ${modelsToProcess.length}/${sortedResults.length} models for visualization ${shouldLimitModels ? '(limited for performance)' : ''}`);

      completePhase('resnorm-analysis', {
        pointsAnalyzed: sortedResults.length,
        resnormRange: {
          min: sortedResults[0]?.resnorm,
          max: sortedResults[sortedResults.length - 1]?.resnorm
        }
      });
      startPhase('data-processing', 'Converting results to visualization format');
      
      // Map to ModelSnapshot format for visualization (optimized)
      const mappingStart = Date.now();
      const models: ModelSnapshot[] = [];
      
      // Process in smaller chunks to avoid stack overflow with spread operator
      const CHUNK_SIZE = 1000; // Reduced from 10000 to prevent spread operator stack overflow
      for (let i = 0; i < modelsToProcess.length; i += CHUNK_SIZE) {
        const chunk = modelsToProcess.slice(i, i + CHUNK_SIZE);
        const chunkModels = chunk.map((point, localIdx) => 
          mapBackendMeshToSnapshot(point, i + localIdx)
        );
        
        // Use Array.prototype.push.apply instead of spread operator to avoid stack overflow
        Array.prototype.push.apply(models, chunkModels);
        
        // Allow UI to update during processing
        if (i % (CHUNK_SIZE * 10) === 0) {
          setTimeout(() => {
            updateStatusMessage(`Mapping models for visualization: ${Math.min(i + CHUNK_SIZE, modelsToProcess.length)}/${modelsToProcess.length}`);
          }, 1);
        }
      }
      
      const mappingTime = ((Date.now() - mappingStart) / 1000).toFixed(3);
      updateStatusMessage(`[TIMING] Model mapping completed in ${mappingTime}s`);

      // Create resnorm groups
      updateStatusMessage('Creating visualization groups...');
      const groups = createResnormGroups(models, minResnorm, maxResnorm);
      setResnormGroups(groups);
      const groupingTime = ((Date.now() - groupingStart) / 1000).toFixed(3);
      updateStatusMessage(`[TIMING] Visualization groups created in ${groupingTime}s`);

      // Add unique IDs for table display (chunked to avoid memory issues)
      const pointsWithIds: (BackendMeshPoint & { id: number })[] = [];
      const ID_CHUNK_SIZE = 5000;
      
      for (let i = 0; i < sortedResults.length; i += ID_CHUNK_SIZE) {
        const chunk = sortedResults.slice(i, i + ID_CHUNK_SIZE);
        const chunkWithIds = chunk.map((point, localIdx) => ({
          ...point,
          id: i + localIdx + 1
        }));
        Array.prototype.push.apply(pointsWithIds, chunkWithIds);
      }

      // Complete data processing phase
      completePhase('data-processing', {
        modelsProcessed: models.length,
        groupsCreated: groups.length,
        memoryOptimized: models.length < sortedResults.length
      });
      
      // Apply visualization filtering before setting results
      const filteredModels = applyVisualizationFiltering(models, groups, visualizationSettings);
      
      // Convert filtered models back to BackendMeshPoint format for consistency
      const filteredResults = filteredModels.map(model => {
        // Find the original BackendMeshPoint for this model
        const originalResult = sortedResults.find(result => 
          result.parameters.Rsh === model.parameters.Rsh &&
          result.parameters.Ra === model.parameters.Ra &&
          result.parameters.Ca === model.parameters.Ca &&
          result.parameters.Rb === model.parameters.Rb &&
          result.parameters.Cb === model.parameters.Cb
        );
        
        return originalResult || {
          parameters: model.parameters,
          spectrum: (model.data || []).map(point => ({
            freq: point.frequency,
            real: point.real,
            imag: point.imaginary,
            mag: point.magnitude,
            phase: point.phase
          })),
          resnorm: model.resnorm || 0
        };
      });
      
      // Filtered results ready for state update
      
      // Convert BackendMeshPoint to ModelSnapshot and update state
      const filteredModelSnapshots = filteredResults.map((result, index) => mapBackendMeshToSnapshot(result, index));
      setGridResults(filteredModelSnapshots);
      
      // Calculate final timing and generate comprehensive performance log
      const endTime = Date.now();
      const totalTime = endTime - startTime;
      
      // Generate comprehensive performance log
      const perfLog = generatePerformanceLog(currentPhases, totalTime, totalPoints);
      setPerformanceLog(perfLog);
      
      const totalTimeSec = (totalTime / 1000).toFixed(2);
      const generationTime = ((computationStartTime - generationStartTime) / 1000).toFixed(2);
      const spectrumTime = ((processingStartTime - computationStartTime) / 1000).toFixed(2);
      const processingTime = ((endTime - processingStartTime) / 1000).toFixed(2);

      // Log comprehensive performance breakdown
      updateStatusMessage(`[PERFORMANCE LOG] Computation completed in ${totalTimeSec}s`);
      updateStatusMessage(`[BOTTLENECK] Slowest phase: ${perfLog.bottleneck}`);
      updateStatusMessage(`[THROUGHPUT] ${perfLog.efficiency.throughput.toFixed(0)} points/second using ${perfLog.efficiency.cpuCores} CPU cores`);
      updateStatusMessage(`[MEMORY] ~${perfLog.efficiency.memoryUsage.toFixed(1)}MB estimated usage`);
      updateStatusMessage(`[BREAKDOWN] Grid: ${(perfLog.summary.gridGeneration/1000).toFixed(1)}s | Impedance: ${(perfLog.summary.impedanceComputation/1000).toFixed(1)}s | Analysis: ${(perfLog.summary.resnormAnalysis/1000).toFixed(1)}s | Processing: ${(perfLog.summary.dataProcessing/1000).toFixed(1)}s`);
      updateStatusMessage(`[RESULTS] ${sortedResults.length} computed points, ${filteredResults.length} rendered points, ${groups.length} resnorm groups created`);
      updateStatusMessage(`Resnorm range: ${minResnorm.toExponential(3)} to ${maxResnorm.toExponential(3)}`);
      updateStatusMessage(`Memory usage optimized: ${results.filter(r => r.spectrum.length > 0).length} points with full spectra`);
      updateStatusMessage(`Rendering efficiency: ${((filteredResults.length / sortedResults.length) * 100).toFixed(1)}% of computed models rendered`);
      
      // Create and show summary notification
      const summaryData: ComputationSummary = {
        title: 'Grid Computation Complete',
        totalTime: totalTimeSec,
        generationTime,
        computationTime: spectrumTime,
        processingTime,
        totalPoints,
        validPoints: sortedResults.length,
        groups: groups.length,
        cores: navigator.hardwareConcurrency || 4,
        throughput: totalPoints / parseFloat(totalTimeSec),
        type: 'success',
        duration: 8000
      };

      // Log the summary to activity log with detailed breakdown
      updateStatusMessage(`[COMPLETION SUMMARY] Total: ${totalTimeSec}s | Generation: ${generationTime}s | Computation: ${spectrumTime}s | Processing: ${processingTime}s | Throughput: ${(totalPoints / parseFloat(totalTimeSec)).toFixed(0)} pts/s`);

      // Show the notification
      setComputationSummary(summaryData);

      // Update reference model
      const updatedModel = createReferenceModel();
      setReferenceModel(updatedModel);

      if (!referenceModelId) {
        setReferenceModelId('dynamic-reference');
        updateStatusMessage('Reference model created and shown');
      } else {
        updateStatusMessage('Reference model updated');
      }

      // Mark the currently selected profile as computed
      if (savedProfilesState.selectedProfile) {
        await profileActions.updateProfile(savedProfilesState.selectedProfile, { isComputed: true });
      }

      setIsComputingGrid(false);
      setComputationProgress(null);
      
      // Keep sidebar open after computation completes (removed auto-collapse)
      
      // Always auto-save the configuration after successful computation with enhanced immediate saving
      const autoSaveName = configurationName.trim() || `Auto-saved ${new Date().toLocaleString('en-US', { 
        month: 'short', 
        day: 'numeric', 
        hour: '2-digit', 
        minute: '2-digit',
        hour12: false 
      })}`;
      
      console.log('Starting automatic profile save after grid computation...');
      try {
        const savedProfileId = await handleSaveProfile(autoSaveName, `Last action: Grid computation completed - ${sortedResults.length} models computed`);
        if (savedProfileId) {
          console.log('Profile auto-saved successfully:', savedProfileId);
          // Force immediate UI update
          setTimeout(() => {
            profileActions.refreshProfiles();
          }, 50);
          updateStatusMessage(`Configuration auto-saved as "${autoSaveName}" - visible in left menu`);
        } else {
          console.error('Auto-save failed - retrying...');
          // Retry with simplified name
          const retryName = `Auto-save ${Date.now()}`;
          const retryId = await handleSaveProfile(retryName, `Last action: Grid computation completed (retry save) - ${sortedResults.length} models computed`);
          if (retryId) {
            console.log('Retry auto-save successful:', retryId);
            updateStatusMessage(`Configuration saved as "${retryName}"`);
          }
        }
      } catch (error) {
        console.error('Auto-save error:', error);
        updateStatusMessage('Grid computed successfully but auto-save failed');
      }

      // Don't clear configuration name - keep it synced with saved profile

    } catch (error) {
      console.error("Error in parallel computation:", error);
      const errorMessage = error instanceof Error ? error.message : String(error);
      const errorTime = ((Date.now() - startTime) / 1000).toFixed(2);
      
      setGridError(`Parallel grid computation failed: ${errorMessage}`);
      updateStatusMessage(`[ERROR] Computation failed after ${errorTime}s: ${errorMessage}`);
      updateStatusMessage(`[TIMING] Error occurred during computation phase`);
      
      // Clear any pending computation on error
      profileActions.setPendingComputeProfile(null);
      
      // Show error notification
      const errorSummary: ComputationSummary = {
        title: '❌ Grid Computation Failed',
        totalTime: errorTime,
        generationTime: generationStartTime > 0 ? ((Date.now() - generationStartTime) / 1000).toFixed(2) : '0',
        computationTime: computationStartTime > 0 ? ((Date.now() - computationStartTime) / 1000).toFixed(2) : '0',
        processingTime: '0',
        totalPoints,
        validPoints: 0,
        groups: 0,
        cores: navigator.hardwareConcurrency || 4,
        throughput: 0,
        type: 'error',
        duration: 10000 // Show error longer
      };
      
      setComputationSummary(errorSummary);
      
      setIsComputingGrid(false);
      setComputationProgress(null);
    }
  }, [gridSize, updateStatusMessage, setIsComputingGrid, resetComputationState, setParameterChanged, setManuallyHidden, setTotalGridPoints, minFreq, maxFreq, numPoints, setComputationProgress, hybridComputeManager.computeGridHybrid, mapBackendMeshToSnapshot, setGridResults, setResnormGroups, setComputationSummary, visualizationSettings, applyVisualizationFiltering, calculateEffectiveVisualizationLimit, completePhase, createReferenceModel, currentPhases, generatePerformanceLog, parameters, isUserControlledLimits, performanceSettings, extendedSettings, referenceModelId, savedProfilesState.selectedProfile, setActualComputedPoints, setEstimatedMemoryUsage, setGridError, setMemoryLimitedPoints, setSkippedPoints, startPhase, userVisualizationPercentage, profileActions, configurationName]); // eslint-disable-line react-hooks/exhaustive-deps

  // Add pagination state - used by child components
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [currentPage, setCurrentPage] = useState(1);
  
  // Update frequency array when range changes - called internally
  const updateFrequencies = useCallback((min: number, max: number, points: number) => {
    // Validate frequency range
    if (min <= 0) {
      console.warn("Minimum frequency must be positive, setting to 0.01 Hz");
      min = 0.01;
    }
    if (max <= min) {
      console.warn("Maximum frequency must be greater than minimum, setting to min*10");
      max = min * 10;
    }
    
    // Create proper logarithmically spaced frequencies for EIS
    const frequencies: number[] = [];
    const logMin = Math.log10(min);
    const logMax = Math.log10(max);
    const logStep = (logMax - logMin) / (points - 1);
    
    for (let i = 0; i < points; i++) {
      const logValue = logMin + i * logStep;
      const frequency = Math.pow(10, logValue);
      frequencies.push(frequency);
    }
    
    setMinFreq(min);
    setMaxFreq(max);
    setNumPoints(points);
    
    // Update parameters with the new frequency range
    setParameters((prev: CircuitParameters) => ({
      ...prev,
      frequency_range: [min, max] // Keep the [min, max] format for backward compatibility
    }));
    
    // Store the full frequency array in a separate state variable
    setFrequencyPoints(frequencies);
    
    // Update status message with details about the frequency range
    updateStatusMessage(`Frequency range updated: ${min.toFixed(2)} Hz - ${max.toFixed(1)} Hz with ${points} points (logarithmic spacing)`);
    
    // Mark parameters as changed - will require recomputing the grid
    setParameterChanged(true);
  }, [setMinFreq, setMaxFreq, setNumPoints, setParameters, setFrequencyPoints, updateStatusMessage, setParameterChanged]);
  


  // No longer need TER and TEC display in the header
  // Removed unused variables and the resetToReference function

  // Initialize frequency points on component mount
  useEffect(() => {
    // Generate initial frequency points
    const initialFreqs: number[] = [];
    const logMin = Math.log10(minFreq);
    const logMax = Math.log10(maxFreq);
    const logStep = (logMax - logMin) / (numPoints - 1);
    
    for (let i = 0; i < numPoints; i++) {
      const logValue = logMin + i * logStep;
      const frequency = Math.pow(10, logValue);
      initialFreqs.push(frequency);
    }
    
    setFrequencyPoints(initialFreqs);
  }, [minFreq, maxFreq, numPoints]);

  // Separate useEffect for initializing the reference model
  useEffect(() => {
    if (frequencyPoints.length > 0 && !referenceModel && !manuallyHidden) {
      const initialReferenceModel = createReferenceModel();
      setReferenceModel(initialReferenceModel);
      setReferenceModelId('dynamic-reference');
    }
  }, [frequencyPoints, referenceModel, manuallyHidden, createReferenceModel]);

  // Update reference model when frequency settings change
  useEffect(() => {
    if (referenceModelId === 'dynamic-reference' && !manuallyHidden) {
      const updatedModel = createReferenceModel();
      setReferenceModel(updatedModel);
      updateStatusMessage('Reference model updated with new frequency range');
    }
  }, [minFreq, maxFreq, numPoints, frequencyPoints.length, referenceModelId, manuallyHidden, createReferenceModel, updateStatusMessage]);

  // Grid values handler removed - functionality moved to visualization layer

  // Note: Grid clearing functionality is now handled by resetComputationState from useComputationState hook

  // Wrapper for cancel computation that also clears pending profile
  const handleCancelComputation = useCallback(() => {
    updateStatusMessage('⚠️ Cancelling computation...');
    
    try {
      cancelComputation();
      setIsComputingGrid(false);
      resetComputationState();
      profileActions.setPendingComputeProfile(null);
      updateStatusMessage('Computation cancelled successfully');
    } catch (error) {
      console.error('Error cancelling computation:', error);
      updateStatusMessage('Error cancelling computation');
    }
  }, [cancelComputation, setIsComputingGrid, resetComputationState, updateStatusMessage, profileActions]);


  // Handler for saving circuit configurations (replaces legacy profile saving)

  const handleSaveProfile = useCallback(async (name: string, description?: string, forceNew?: boolean) => {
    const isAutoSave = forceNew === undefined; // Auto-save if forceNew is not specified
    console.log('Starting circuit configuration save:', { name, description, parameters, gridSize, minFreq, maxFreq, numPoints, isAutoSave, forceNew });

    if (!sessionManagement.sessionState.userId) {
      console.log('No user ID for saving circuit configuration - user not authenticated');
      if (!isAutoSave) {
        // Only show message for manual saves, not auto-saves
        updateStatusMessage('Please sign in to save circuit configurations');
      }
      return null;
    }

    // Handle different save modes
    if (forceNew === false && activeConfigId) {
      // Update mode: Update the currently active configuration
      console.log('Updating current circuit configuration:', activeConfigId);
      try {
        const success = await updateConfiguration(activeConfigId, {
          name: name || 'Updated Configuration',
          description: description || 'Updated configuration',
          circuitParameters: parameters,
          gridSize,
          minFreq,
          maxFreq,
          numPoints
        });
        
        if (success) {
          // Enhanced activity logging for updates
          const timestamp = new Date().toLocaleTimeString();
          const activityDetails = `Updated "${name}" - Grid: ${gridSize}^5, Freq: ${minFreq}-${maxFreq}Hz (${numPoints}pts), Parameters: Rs=${parameters.Rsh?.toFixed(1)}, Ra=${parameters.Ra?.toFixed(1)}, Ca=${(parameters.Ca * 1e6)?.toFixed(2)}µF, Rb=${parameters.Rb?.toFixed(1)}, Cb=${(parameters.Cb * 1e6)?.toFixed(2)}µF`;

          updateStatusMessage(`[${timestamp}] ✅ Updated "${name}" successfully`);
          console.log('Profile Update Activity:', activityDetails);
          return activeConfigId;
        } else {
          updateStatusMessage(`Failed to update configuration "${name}"`);
          return null;
        }
      } catch (error) {
        console.error('Configuration update error:', error);
        updateStatusMessage(`Error updating configuration: ${error instanceof Error ? error.message : 'Unknown error'}`);
        return null;
      }
    }
    
    if (forceNew === true) {
      // Save As New mode: Always create a new configuration
      console.log('Force creating new circuit configuration...');
    } else if (activeConfigId) {
      // Auto-save mode: Update existing active configuration
      console.log('Auto-saving to existing configuration:', activeConfigId);
      try {
        const success = await updateConfiguration(activeConfigId, {
          circuitParameters: parameters,
          gridSize,
          minFreq,
          maxFreq,
          numPoints
        });
        
        if (success) {
          // Enhanced activity logging for auto-saves
          const timestamp = new Date().toLocaleTimeString();
          const currentConfig = circuitConfigurations?.find(c => c.id === activeConfigId);
          updateStatusMessage(`[${timestamp}] Auto-saved "${currentConfig?.name || 'Current Profile'}"`);
          console.log('Manual Save Activity: Auto-saved existing profile');
          return activeConfigId;
        }
      } catch (error) {
        console.log('WARNING: Auto-save to existing config failed, creating new one:', error);
      }
    }
    
    // Create new configuration (either force new, no active config, or auto-save failed)
    console.log('Creating new circuit configuration...');
    try {
      // Ensure unique name for new configurations
      const baseName = name || 'New Circuit Configuration';
      const uniqueName = generateUniqueCircuitName(baseName);
      
      const newConfig = await createConfiguration({
        name: uniqueName,
        description: description || 'Circuit configuration saved from simulator',
        circuitParameters: parameters,
        gridSize,
        minFreq,
        maxFreq,
        numPoints
      });
      
      if (newConfig) {
        console.log('Circuit configuration created:', newConfig.id);
        // Set as active in both local state and session
        setActiveConfiguration(newConfig.id);
        await sessionManagement.actions.setActiveCircuitConfig(newConfig.id);
        // Enhanced activity logging for new profile creation
        const timestamp = new Date().toLocaleTimeString();
        const activityDetails = `Created "${newConfig.name}" - Grid: ${gridSize}^5, Freq: ${minFreq}-${maxFreq}Hz (${numPoints}pts), Parameters: Rs=${parameters.Rsh?.toFixed(1)}, Ra=${parameters.Ra?.toFixed(1)}, Ca=${(parameters.Ca * 1e6)?.toFixed(2)}µF, Rb=${parameters.Rb?.toFixed(1)}, Cb=${(parameters.Cb * 1e6)?.toFixed(2)}µF`;

        updateStatusMessage(`[${timestamp}] Created "${newConfig.name}" successfully`);
        console.log('New Profile Activity:', activityDetails);
        return newConfig.id;
      } else {
        console.error('Configuration creation returned null');
        updateStatusMessage(`Failed to save configuration "${name}"`);
        return null;
      }
    } catch (error) {
      console.error('Configuration creation error:', error);
      updateStatusMessage(`Error saving configuration "${name}": ${error instanceof Error ? error.message : 'Unknown error'}`);
      return null;
    }
  }, [gridSize, minFreq, maxFreq, numPoints, parameters, updateStatusMessage, activeConfigId, createConfiguration, updateConfiguration, sessionManagement, generateUniqueCircuitName]);

  // Function to create a new circuit with standard configuration (moved after handleSaveProfile)
  const _handleNewCircuit = useCallback(async () => {
    // Clear any existing results
    resetComputationState();

    router.push('/simulator', { scroll: false });

    // Reset to requested default configuration
    const newParameters = {
      Rsh: 870,
      Ra: 7500,
      Ca: 0.0000042000000000000004,
      Rb: 6210,
      Cb: 0.0000035,
      frequency_range: [0.1, 100000] as [number, number]
    };

    setParameters(newParameters);

    // Reset frequency settings
    setMinFreq(0.1);
    setMaxFreq(100000);
    setNumPoints(100);

    // Reset grid size to user's default
    setGridSize(defaultGridSize);

    // Generate unique name for new circuit
    const timestamp = new Date().toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false
    });
    const baseCircuitName = `New Circuit ${timestamp}`;
    const uniqueCircuitName = generateUniqueCircuitName(baseCircuitName);
    setConfigurationName(uniqueCircuitName);

    // Automatically create and select the new circuit profile
    try {
      console.log('Creating new circuit profile automatically...');
      const newProfileId = await handleSaveProfile(
        uniqueCircuitName,
        'Last action: New circuit configuration created',
        true  // forceNew: true to ensure a new circuit is created
      );

      if (newProfileId) {
        console.log('New circuit profile created:', newProfileId);
        updateStatusMessage(`New circuit "${uniqueCircuitName}" created and selected - ready to customize and compute`);
      } else {
        console.error('Failed to create new circuit profile');
        updateStatusMessage('New circuit created but failed to save profile - you can save manually');
      }
    } catch (error) {
      console.error('Error creating new circuit profile:', error);
      updateStatusMessage('New circuit created - profile creation failed, you can save manually');
    }
  }, [resetComputationState, updateStatusMessage, setGridSize, setParameters, setMinFreq, setMaxFreq, setNumPoints, setConfigurationName, generateUniqueCircuitName, defaultGridSize, handleSaveProfile]);





  // Note: Grid generation and spectrum calculation now handled by Web Workers


  // Track if we've loaded from localStorage to avoid hydration issues
  const [hasLoadedFromStorage, setHasLoadedFromStorage] = useState(false);

  // Track profile loading for computation
  // pendingComputeProfile is now handled by the useUserProfiles hook
  

// Handler for copying profile parameters
  const _handleCopyParams = useCallback((configId: string) => {
    const config = circuitConfigurations?.find(c => c.id === configId);
    if (config) {
      const paramsData = {
        profileName: config.name,
        gridSettings: {
          gridSize: config.gridSize,
          minFreq: config.minFreq,
          maxFreq: config.maxFreq,
          numPoints: config.numPoints
        },
        circuitParameters: {
          Rsh: config.circuitParameters.Rsh,
          Ra: config.circuitParameters.Ra,
          Ca: config.circuitParameters.Ca,
          Rb: config.circuitParameters.Rb,
          Cb: config.circuitParameters.Cb,
          frequency_range: config.circuitParameters.frequency_range
        }
      };

      // Copy to clipboard as formatted JSON
      const jsonString = JSON.stringify(paramsData, null, 2);
      
      if (navigator.clipboard && window.isSecureContext) {
        // Use the Clipboard API when available (HTTPS/localhost)
        navigator.clipboard.writeText(jsonString).then(() => {
          updateStatusMessage(`Parameters copied to clipboard from "${config.name}"`);
        }).catch(() => {
          updateStatusMessage(`Failed to copy parameters to clipboard`);
        });
      } else {
        // Fallback for non-secure contexts
        try {
          const textArea = document.createElement('textarea');
          textArea.value = jsonString;
          textArea.style.position = 'fixed';
          textArea.style.opacity = '0';
          document.body.appendChild(textArea);
          textArea.select();
          document.execCommand('copy');
          document.body.removeChild(textArea);
          updateStatusMessage(`Parameters copied to clipboard from "${config.name}"`);
        } catch {
          updateStatusMessage(`Failed to copy parameters to clipboard`);
        }
      }
    }
  }, [circuitConfigurations, updateStatusMessage]);

  // Removed unused tagged models state

  // Load tagged models from database for current user and profile
  const loadTaggedModelsFromDatabase = useCallback(async () => {
    if (!user) {
      console.log('No user logged in, skipping tagged models load');
      return;
    }

    // Use activeConfigId instead of legacy selectedProfile
    if (!activeConfigId) {
      console.log('No active circuit configuration, clearing tagged models');
      setTaggedModels(new Map());
      return;
    }

    try {
      const { TaggedModelsService } = await import('../../lib/taggedModelsService');
      const allModels = await TaggedModelsService.getAllUserTaggedModels(user.id);
      const filtered = allModels.filter(m => m.circuitConfigId === activeConfigId);

      const newTaggedModels = new Map<string, { tagName: string; profileId: string; resnormValue: number; taggedAt: number; notes?: string }>();
      filtered.forEach(m => {
        newTaggedModels.set(m.modelId, {
          tagName: m.tagName,
          profileId: m.circuitConfigId,
          resnormValue: m.resnormValue || 0,
          taggedAt: new Date(m.createdAt).getTime(),
          notes: m.notes || undefined
        });
      });

      setTaggedModels(newTaggedModels);
      if (filtered.length > 0) {
        updateStatusMessage(`Loaded ${filtered.length} tagged models for current circuit configuration`);
      }
    } catch (error) {
      console.error('Error loading tagged models:', error);
    }
  }, [user, activeConfigId, updateStatusMessage]); // FIXED: Use activeConfigId dependency

  // Load tagged models when user or active circuit config changes
  useEffect(() => {
    if (user && sessionManagement.isReady && activeConfigId) {
      loadTaggedModelsFromDatabase();
    }
  }, [user, sessionManagement.isReady, activeConfigId, loadTaggedModelsFromDatabase]);

  // Handle model tagging (kept for potential future use)
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const handleModelTag = useCallback(async (model: ModelSnapshot, tagName: string) => {
    if (!user || !activeConfigId) {
      console.warn('⚠️ Cannot tag model: no user or active circuit config');
      return;
    }

    try {
      if (!tagName || tagName.trim() === '') {
        // Remove tag if tagName is empty
        const modelKey = model.id;
        if (taggedModels.has(modelKey)) {
          // TODO: Remove from database
          const newTaggedModels = new Map(taggedModels);
          newTaggedModels.delete(modelKey);
          setTaggedModels(newTaggedModels);
          updateStatusMessage(`🗑️ Removed tag for model`);
        }
        return;
      }

      // Add new tag
      const { TaggedModelsService } = await import('../../lib/taggedModelsService');

      const taggedModel = await TaggedModelsService.createTaggedModel(user.id, {
        circuitConfigId: activeConfigId,
        modelId: model.id,
        tagName: tagName.trim(),
        resnormValue: model.resnorm ?? 0,
        isInteresting: true,
        metadata: model.parameters as unknown as Record<string, unknown>,
      });

      // Update local state
      const newTaggedModels = new Map(taggedModels);
      newTaggedModels.set(model.id, {
        tagName: taggedModel.tagName,
        profileId: taggedModel.circuitConfigId,
        resnormValue: taggedModel.resnormValue || 0,
        taggedAt: new Date(taggedModel.createdAt).getTime(),
        notes: taggedModel.notes
      });
      setTaggedModels(newTaggedModels);
      updateStatusMessage(`🏷️ Tagged model: ${tagName}`);

    } catch (error) {
      console.error('❌ Failed to tag model:', error);
      updateStatusMessage('❌ Failed to tag model');
    }
  }, [user, activeConfigId, taggedModels, updateStatusMessage]);


  // Set loading state as completed since useUserProfiles handles loading
  useEffect(() => {
    if (!hasLoadedFromStorage) {
      setHasLoadedFromStorage(true);
    }
  }, [hasLoadedFromStorage]);


  // Persist profiles to localStorage whenever they change (but only after initial load)
  useEffect(() => {
    if (typeof window !== 'undefined' && hasLoadedFromStorage) {
      try {
        localStorage.setItem('nei-viz-saved-profiles', JSON.stringify(savedProfilesState));
      } catch (error) {
        console.warn('Failed to save profiles to localStorage:', error);
      }
    }
  }, [savedProfilesState, hasLoadedFromStorage]);

  // Legacy auto-create sample profile functionality removed

  // Legacy pending computation logic removed - now handled by onComputeProfile with auto-computation


  // Modify the main content area to show the correct tab content
  return (
    <div className="h-screen bg-neutral-950 text-white flex overflow-hidden">
      {/* Main Content Area - Full Width Layout */}
      <div className="flex-1 relative overflow-hidden">
        {/* Main visualization area — fills full height under floating navbar */}
        <div className="absolute inset-0 flex overflow-hidden">
          <div className="flex-1 overflow-hidden relative">
            {/* Cancel overlay — only shown during computation */}
            {isComputingGrid && (
              <div className="absolute top-0 right-0 z-50 px-3 py-1.5 flex items-center gap-3 text-xs">
                <span className="text-neutral-500">
                  {computationProgress?.message ?? 'Computing...'}
                  {computationProgress?.processed && computationProgress?.total
                    ? ` · ${Math.round((computationProgress.processed / computationProgress.total) * 100)}%`
                    : ''}
                </span>
                <button
                  onClick={handleCancelComputation}
                  className="text-neutral-600 hover:text-red-400 transition-colors"
                >
                  Cancel
                </button>
              </div>
            )}

            {/* Main content - always rendered */}
            {gridError ? (
              <div className="p-4 bg-danger-light border border-danger rounded-lg text-sm text-danger">
                <div className="flex items-start">
                  <svg className="w-5 h-5 mr-2 mt-0.5 text-danger" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <div>
                    <p className="font-medium">Error occurred</p>
                    <p className="mt-1">{gridError}</p>
                  </div>
                </div>
              </div>
            ) : visualizationTab === 'activity' ? (
              <div className="h-full flex flex-col overflow-hidden">
                <h3 className="text-lg font-medium text-neutral-200 mb-4 px-2 flex-shrink-0">Activity Log</h3>
                
                {/* Performance Log Display */}
                {performanceLog && (
                  <div className="bg-neutral-800 border border-neutral-600 rounded p-3 mb-4 mx-2">
                    <h4 className="text-sm font-semibold text-neutral-200 mb-2">Latest Performance Report</h4>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="text-neutral-400">Total Duration:</div>
                      <div className="text-neutral-200 font-mono">{(performanceLog.totalDuration / 1000).toFixed(2)}s</div>
                      <div className="text-neutral-400">Bottleneck:</div>
                      <div className="text-neutral-300 font-mono">{performanceLog.bottleneck}</div>
                      <div className="text-neutral-400">Throughput:</div>
                      <div className="text-neutral-300 font-mono">{performanceLog.efficiency.throughput.toFixed(0)} pts/s</div>
                      <div className="text-neutral-400">Memory:</div>
                      <div className="text-neutral-300 font-mono">~{performanceLog.efficiency.memoryUsage.toFixed(1)}MB</div>
                    </div>
                    <div className="mt-2 pt-2 border-t border-neutral-600">
                      <div className="text-xs text-neutral-400 mb-1">Phase Breakdown:</div>
                      <div className="grid grid-cols-2 gap-1 text-xs font-mono">
                        <div>Grid: {(performanceLog.summary.gridGeneration/1000).toFixed(1)}s</div>
                        <div>Impedance: {(performanceLog.summary.impedanceComputation/1000).toFixed(1)}s</div>
                        <div>Analysis: {(performanceLog.summary.resnormAnalysis/1000).toFixed(1)}s</div>
                        <div>Processing: {(performanceLog.summary.dataProcessing/1000).toFixed(1)}s</div>
                      </div>
                    </div>
                  </div>
                )}
                
                <div className="flex-1 overflow-y-auto space-y-1 pr-4">
                  {logMessages.length === 0 ? (
                    <div className="text-neutral-400 text-sm italic px-2">No activity yet...</div>
                  ) : (
                    logMessages.map((message, index) => (
                      <div key={index} className="text-sm font-mono text-neutral-300 px-3 py-1.5 hover:bg-neutral-800/50 border-l-2 border-transparent hover:border-slate-500/50 transition-colors">
                        <span className="text-neutral-500 text-xs">[{message.time}]</span>{' '}
                        <span className="text-neutral-200 break-words">{message.message.length > 80 ? message.message.substring(0, 80) + '...' : message.message}</span>
                      </div>
                    ))
                  )}
                </div>
              </div>
            ) : (
              <AIModelTab
                groundTruthParams={parameters}
                minFreq={parameters.frequency_range[0]}
                maxFreq={parameters.frequency_range[1]}
                numPoints={numPoints}
                onZoomChange={setCanvasZoom}
                zoomActionsRef={zoomActionsRef}
              />
            )}
          </div>

        </div>

        {/* Floating navbar — overlays canvas with gradient fade */}
        <div className="absolute top-0 inset-x-0 z-50 pointer-events-none">
          <div className="pointer-events-auto bg-gradient-to-b from-[#0d0d0f] to-transparent pb-3">
            <SlimNavbar
              gridResults={gridResults}
              activeCircuitName={circuitConfigurationsRef.current?.find(c => c.id === activeConfigId)?.name}
            />
            <div className="h-0.5 relative">
              <div
                className="h-full bg-primary transition-all duration-500"
                style={{ width: isComputingGrid ? (computationProgress ? `${computationProgress.overallProgress}%` : '8%') : '0%', opacity: isComputingGrid ? 1 : 0 }}
              />
            </div>
          </div>
        </div>

        {/* Top-right HUD — zoom + docs + share | user tile */}
        <div className="absolute top-3 right-3 z-[60] flex items-center gap-2">
          {/* Main pill */}
          <div className={PILL_BASE}>
            <button onClick={() => zoomActionsRef.current?.zoomOut()} className={`${ICON_BTN} font-mono`} title="Zoom out">−</button>
            <button onClick={() => zoomActionsRef.current?.zoomReset()} className={MONO_VALUE} title="Reset zoom">
              {Math.round(canvasZoom * 100)}%
            </button>
            <button onClick={() => zoomActionsRef.current?.zoomIn()} className={`${ICON_BTN} font-mono`} title="Zoom in">+</button>

            <div className={DIVIDER} />

            <button onClick={() => setModelInfoModalOpen(true)} className={ICON_BTN} title="Circuit model">
              <svg className={ICON.md} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </button>

            <div className={DIVIDER} />

            <button disabled className={LABEL_BTN_DISABLED} title="Share (coming soon)">Share</button>
          </div>

          {/* User — separate rounded tile */}
          {user && (
            <button
              onClick={signOut}
              title={`${user.email} — click to sign out`}
              className="h-8 w-8 flex items-center justify-center rounded-lg bg-[#0d0d0f]/90 border border-[#1e1e24] text-[#9a9aa2] hover:text-[#dddde2] hover:border-[#2a2a33] backdrop-blur-sm transition-colors focus:outline-none"
            >
              <span className="text-[11px] font-semibold">{user.email[0].toUpperCase()}</span>
            </button>
          )}
        </div>
      </div>

      {/* Computation Summary Notification - Bottom Right */}
      <ComputationNotification 
        summary={computationSummary}
        onDismiss={() => setComputationSummary(null)}
      />


      {/* Model Info Modal */}
      <ModelInfoModal
        isOpen={modelInfoModalOpen}
        onClose={() => setModelInfoModalOpen(false)}
      />

      {/* Authentication Modal */}
      <AuthModal
        isOpen={showAuthModal}
        onClose={() => setShowAuthModal(false)}
        onAuthSuccess={() => {
          setShowAuthModal(false);
        }}
      />

      {/* Welcome modal — shown once on first visit */}
      {showWelcome && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 backdrop-blur-sm">
          <div className="bg-[#131316] border border-[#2a2a33] rounded-2xl p-8 max-w-lg w-full mx-4 shadow-2xl">
            <div className="mb-6">
              <h2 className="text-xl font-semibold text-[#dddde2] mb-1">SpideyPlot</h2>
              <p className="text-xs text-[#454549] uppercase tracking-widest">EIS analysis platform for RPE research</p>
            </div>

            <div className="space-y-4 mb-8">
              <p className="text-sm text-[#9a9aa2] leading-relaxed">
                SpideyPlot is a node-based pipeline for analyzing electrochemical impedance spectroscopy data from retinal pigment epithelium experiments.
              </p>

              <div className="bg-[#0d0d0f] border border-[#1e1e24] rounded-lg p-4 space-y-3">
                <p className="text-xs text-[#454549] uppercase tracking-widest mb-2">Workflow</p>
                <div className="flex items-start gap-3">
                  <span className="w-2 h-2 rounded-full bg-[#4ade80] flex-shrink-0 mt-1" />
                  <div>
                    <p className="text-xs font-medium text-[#dddde2]">Source</p>
                    <p className="text-xs text-[#454549]">Load real experimental data or generate synthetic parameter grids</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="w-2 h-2 rounded-full bg-[#60a5fa] flex-shrink-0 mt-1" />
                  <div>
                    <p className="text-xs font-medium text-[#dddde2]">Process</p>
                    <p className="text-xs text-[#454549]">Run ML-based parameter extraction or classical ECM fitting</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <span className="w-2 h-2 rounded-full bg-[#f97316] flex-shrink-0 mt-1" />
                  <div>
                    <p className="text-xs font-medium text-[#dddde2]">Visualize</p>
                    <p className="text-xs text-[#454549]">Inspect results with spider plots, Nyquist plots, and trajectory charts</p>
                  </div>
                </div>
              </div>

              <p className="text-xs text-[#454549]">
                Use the Add node button at the bottom of the canvas to get started.
              </p>
            </div>

            <button
              onClick={dismissWelcome}
              className="w-full py-2.5 bg-[#1e1e24] hover:bg-[#2a2a33] border border-[#3a3a46] hover:border-[#4a4a56] text-[#dddde2] text-sm font-medium rounded-lg transition-colors duration-100"
            >
              Get started
            </button>
          </div>
        </div>
      )}
    </div>
  );
} 

// Helper to convert backend spectrum to ImpedancePoint[] (types.ts shape)
function toImpedancePoints(spectrum: { freq: number; real: number; imag: number; mag: number; phase: number; }[]): ImpedancePoint[] {
  return spectrum.map((p): ImpedancePoint => ({
    frequency: p.freq,
    real: p.real,
    imaginary: p.imag,
    magnitude: p.mag,
    phase: p.phase
  }));
}

// --- Complex math operations now handled in Web Workers ---

// Optimized resnorm grouping logic using quartiles (4 groups)
const createResnormGroups = (models: ModelSnapshot[], minResnorm: number, maxResnorm: number): ResnormGroup[] => {
  // Limit processing for very large datasets to prevent memory issues
  const MAX_MODELS_PER_GROUP = 500000;
  const shouldSample = models.length > MAX_MODELS_PER_GROUP;

  // Extract all finite resnorm values for percentile calculation
  const validResnorms = models
    .map(m => m.resnorm)
    .filter(r => r !== undefined && isFinite(r)) as number[];

  if (validResnorms.length === 0) {
    console.warn('No valid resnorm values found for grouping');
    return [];
  }

  // Sort resnorms for percentile calculation
  validResnorms.sort((a, b) => a - b);

  // Calculate quartile thresholds
  const calculatePercentile = (arr: number[], percentile: number): number => {
    const index = Math.ceil((percentile / 100) * arr.length) - 1;
    return arr[Math.max(0, Math.min(index, arr.length - 1))];
  };

  const p25 = calculatePercentile(validResnorms, 25);
  const p50 = calculatePercentile(validResnorms, 50);
  const p75 = calculatePercentile(validResnorms, 75);

  // Pre-allocate arrays for each quartile group
  const q1: ModelSnapshot[] = [];
  const q2: ModelSnapshot[] = [];
  const q3: ModelSnapshot[] = [];
  const q4: ModelSnapshot[] = [];

  let processed = 0;
  const maxToProcess = shouldSample ? MAX_MODELS_PER_GROUP : models.length;
  const step = shouldSample ? Math.ceil(models.length / MAX_MODELS_PER_GROUP) : 1;

  for (let i = 0; i < models.length && processed < maxToProcess; i += step) {
    const model = models[i];
    if (!model || model.resnorm === undefined || !isFinite(model.resnorm)) continue;
    const resnorm = model.resnorm;
    if (resnorm <= p25) {
      q1.push({ ...model, color: '#059669' }); // Emerald-600
    } else if (resnorm <= p50) {
      q2.push({ ...model, color: '#10B981' }); // Emerald-500
    } else if (resnorm <= p75) {
      q3.push({ ...model, color: '#F59E0B' }); // Amber-500
    } else {
      q4.push({ ...model, color: '#DC2626' }); // Red-600
    }
    processed++;
  }

  // Create groups array with quartile labels
  const groups: ResnormGroup[] = [];
  if (q1.length > 0) {
    groups.push({
      range: [minResnorm, p25],
      color: '#059669',
      label: 'Q1 (Best 25%)',
      description: `Resnorm ≤ ${p25.toExponential(2)} (best ${q1.length} models)`,
      items: q1
    });
  }
  if (q2.length > 0) {
    groups.push({
      range: [p25, p50],
      color: '#10B981',
      label: 'Q2 (25-50%)',
      description: `${p25.toExponential(2)} < Resnorm ≤ ${p50.toExponential(2)} (${q2.length} models)`,
      items: q2
    });
  }
  if (q3.length > 0) {
    groups.push({
      range: [p50, p75],
      color: '#F59E0B',
      label: 'Q3 (50-75%)',
      description: `${p50.toExponential(2)} < Resnorm ≤ ${p75.toExponential(2)} (${q3.length} models)`,
      items: q3
    });
  }
  if (q4.length > 0) {
    groups.push({
      range: [p75, maxResnorm],
      color: '#DC2626',
      label: 'Q4 (Worst 25%)',
      description: `Resnorm > ${p75.toExponential(2)} (worst ${q4.length} models)`,
      items: q4
    });
  }
  return groups;
};
