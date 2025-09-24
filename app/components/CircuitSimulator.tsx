"use client";

import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import Image from 'next/image';
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
import { createParameterConfigManager } from './circuit-simulator/utils/parameterConfigManager';
import { ExtendedPerformanceSettings, DEFAULT_GPU_SETTINGS, DEFAULT_CPU_SETTINGS } from './circuit-simulator/types/gpuSettings';
import SimplifiedSettingsModal from './settings/SimplifiedSettingsModal';
import { useComputationState } from './circuit-simulator/hooks/useComputationState';
import { useUserProfiles } from '../hooks/useUserProfiles';
import { useSessionManagement } from '../hooks/useSessionManagement';
import { useCircuitConfigurations } from '../hooks/useCircuitConfigurations';
import { useUISettingsManager } from '../hooks/useUISettingsManager';
import { ProfilesService } from '../../lib/profilesService';
import { CentralizedLimitsManager, setGlobalLimitsManager } from './circuit-simulator/utils/centralizedLimits';

// Add imports for the new tab components at the top of the file
import { MathDetailsTab } from './circuit-simulator/MathDetailsTab';
import { VisualizerTab } from './circuit-simulator/VisualizerTab';
// import { SerializedSpiderPlotDemo } from './circuit-simulator/examples/SerializedSpiderPlotDemo';
// Removed OrchestratorTab - functionality integrated into VisualizerTab

import { PerformanceSettings, DEFAULT_PERFORMANCE_SETTINGS } from './circuit-simulator/controls/PerformanceControls';
import { ComputationNotification, ComputationSummary } from './circuit-simulator/notifications/ComputationNotification';
import { SavedProfiles } from './circuit-simulator/controls/SavedProfiles';
import { StaticRenderSettings, defaultStaticRenderSettings } from './circuit-simulator/controls/StaticRenderControls';
import { SavedProfile } from './circuit-simulator/types/savedProfiles';
import { CenterParameterPanel } from './circuit-simulator/controls/CenterParameterPanel';
import { ResnormConfig, ResnormMethod } from './circuit-simulator/utils/resnorm';
import { AuthModal } from './auth/AuthModal';
import { useAuth } from './auth/AuthProvider';
import { SlimNavbar } from './circuit-simulator/SlimNavbar';

// Remove empty interface and replace with type
type CircuitSimulatorProps = Record<string, never>;

export const CircuitSimulator: React.FC<CircuitSimulatorProps> = () => {
  // Authentication
  const { user, loading: authLoading, signOut } = useAuth();
  const [showAuthModal, setShowAuthModal] = useState(false);
  
  // State for user's default grid size
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
    deleteConfiguration, // ENABLED: For profile deletion
    deleteMultipleConfigurations, // ENABLED: For bulk deletion
    setActiveConfiguration
  } = useCircuitConfigurations(
    sessionManagement.sessionState.userId || undefined,
    sessionManagement.sessionState.currentCircuitConfigId
  );

  // UI Settings management with auto-save
  const {
    uiSettings,
    setActiveTab,
    setSplitPaneHeight, // eslint-disable-line @typescript-eslint/no-unused-vars
    setOpacityLevel, // eslint-disable-line @typescript-eslint/no-unused-vars
    setOpacityExponent,
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
  const [extendedSettings, setExtendedSettings] = useState<ExtendedPerformanceSettings>({
    useSymmetricGrid: true,
    maxComputationResults: 500000, // Compute all data, limit display separately
    gpuAcceleration: DEFAULT_GPU_SETTINGS,
    cpuSettings: DEFAULT_CPU_SETTINGS
  });
  
  // Settings modal state
  const [settingsModalOpen, setSettingsModalOpen] = useState(false);

  // Memory optimization settings
  const [visibleResultsLimit, setVisibleResultsLimit] = useState(100000); // Increased limit for comprehensive data analysis
  const [memoryOptimizationEnabled, setMemoryOptimizationEnabled] = useState(true);

  // SRD upload state
  const [, setSrdUploadMessage] = useState('');

  // Tagged models state - for model tagging and highlighting functionality
  const [taggedModels, setTaggedModels] = useState<Map<string, { tagName: string; profileId: string; resnormValue: number; taggedAt: number; notes?: string }>>(new Map());

  // Streamlined parameter configuration manager - initialized with defaults to avoid circular deps
  const [paramConfigManager] = useState(() => createParameterConfigManager({
    gridSize: 9,
    minFreq: 0.1,
    maxFreq: 100000,
    numPoints: 100,
    maxVisibleResults: 1000,
    memoryOptimizationEnabled: true
  }));

  // Parameter change subscription will be added after all state declarations
  
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
    resnormGroups, setResnormGroups,
    hiddenGroups, setHiddenGroups,
    logMessages,
    statusMessage,
    updateStatusMessage,
    resetComputationState,
    saveComputationState,
    restoreComputationState,
    clearAllComputationData, // eslint-disable-line @typescript-eslint/no-unused-vars
    lastComputedResults,
    updateDefaultGridSize
  } = useComputationState(defaultGridSize);

  // Handle master limit changes from settings modal (simplified version without status message)
  const handleMasterLimitChange = useCallback((percentage: number) => {
    console.log(`🎛️ Master limit changed to ${percentage}% (${Math.floor((percentage / 100) * Math.pow(gridSize, 5)).toLocaleString()} models)`);

    // DEBUG: If percentage is around 58%, log where this is coming from
    if (Math.abs(percentage - 58) < 1) {
      console.error(`🔥 FOUND THE CULPRIT! Master limit being set to ${percentage}% - this explains the incomplete grid!`);
      console.trace('Stack trace for 58% limit setting:');
    }

    const newLimits = centralizedLimits.updateMasterLimit(percentage);
    setCentralizedLimits(newLimits);
    setGlobalLimitsManager(newLimits);
  }, [centralizedLimits, gridSize]);

  // Serialized computation manager for efficient memory storage
  const [serializedManager, setSerializedManager] = useState<SerializedComputationManager | null>(null);

  // SINGLE-CIRCUIT MEMORY MANAGEMENT: Only keep current circuit in memory
  useEffect(() => {
    // Clear previous manager and force cleanup of all cached data
    if (serializedManager) {
      serializedManager.clearCaches();
      console.log('🧹 Cleared previous SerializedComputationManager and all caches');
    }

    // Clear UI state to prevent memory accumulation across computations
    setGridResults([]);
    setResnormGroups([]);
    setComputationSummary(null);

    // Update centralized limits manager for new grid size
    const newCentralizedLimits = CentralizedLimitsManager.fromGridSize(gridSize, centralizedLimits.masterLimitPercentage);
    setCentralizedLimits(newCentralizedLimits);
    setGlobalLimitsManager(newCentralizedLimits);
    console.log(`🎛️ Updated centralized limits manager for grid ${gridSize} (${newCentralizedLimits.masterLimitPercentage}% = ${newCentralizedLimits.masterLimitResults.toLocaleString()} models)`);

    // DEBUG: Check if the percentage is not 100%
    if (newCentralizedLimits.masterLimitPercentage !== 100) {
      console.warn(`⚠️ LIMITS DEBUG: Master limit is ${newCentralizedLimits.masterLimitPercentage}% instead of 100%! This explains incomplete grid generation.`);
      console.log(`🔧 FIXING: Forcing master limit back to 100% to compute complete grid`);

      // Force reset to 100% for complete grid computation
      const fixedLimits = CentralizedLimitsManager.fromGridSize(gridSize, 100);
      setCentralizedLimits(fixedLimits);
      setGlobalLimitsManager(fixedLimits);
      console.log(`✅ Fixed: Master limit reset to 100% (${fixedLimits.masterLimitResults.toLocaleString()} models)`);
    }

    // Create new optimized manager for current grid configuration
    const manager = createSerializedComputationManager(gridSize, 'standard');
    setSerializedManager(manager);

    const estimatedTraditionalMemory = Math.pow(gridSize, 5) * 4000 / (1024 * 1024);
    const estimatedOptimizedMemory = Math.pow(gridSize, 5) * 61 / (1024 * 1024);
    const reductionFactor = estimatedTraditionalMemory / estimatedOptimizedMemory;

    console.log(`🚀 SINGLE-CIRCUIT MODE: New manager for ${gridSize}^5 grid`);
    console.log(`📊 Memory optimization: ${estimatedTraditionalMemory.toFixed(1)}MB → ${estimatedOptimizedMemory.toFixed(1)}MB (${reductionFactor.toFixed(1)}x reduction)`);
  }, [gridSize, centralizedLimits.masterLimitPercentage, setGridResults, setResnormGroups, setComputationSummary]);

  // Convert CircuitConfigurations to SavedProfiles format for backward compatibility
  const convertedProfiles = useMemo((): SavedProfile[] => {
    if (!circuitConfigurations) return [];
    
    // Convert local circuit configurations
    const localProfiles = circuitConfigurations.map(config => ({
      id: config.id,
      name: config.name,
      description: config.description || '',
      created: new Date(config.createdAt).getTime(),
      lastModified: new Date(config.updatedAt).getTime(),
      gridSize: config.gridSize,
      minFreq: config.minFreq,
      maxFreq: config.maxFreq,
      numPoints: config.numPoints,
      groundTruthParams: config.circuitParameters,
      isComputed: config.isComputed,
      computationTime: config.computationTime,
      totalPoints: config.totalPoints,
      validPoints: config.validPoints,
      datasetSource: 'local' as const
    }));

    return localProfiles;
  }, [circuitConfigurations]);

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
  
  // Load user's default grid size when user changes
  useEffect(() => {
    if (user?.id && !authLoading) {
      ProfilesService.getUserDefaultGridSize(user.id).then((loadedDefaultGridSize) => {
        setDefaultGridSize(loadedDefaultGridSize);
        updateDefaultGridSize(loadedDefaultGridSize);
      });
    }
  }, [user?.id, authLoading, updateDefaultGridSize]);
  
  // Auto-restore computation state when switching to visualizer tab
  useEffect(() => {
    if (visualizationTab === 'visualizer' && resnormGroups.length === 0 && (lastComputedResults?.resnormGroups?.length || 0) > 0) {
      // Only restore if we're on visualizer tab, have no current data, but have saved data
      const restored = restoreComputationState();
      if (restored) {
        console.log('Auto-restored computation state when switching to visualizer tab');
      }
    }
  }, [visualizationTab, resnormGroups.length, lastComputedResults, restoreComputationState]);
  
  // Multi-select state for circuits is now managed by UI settings
  const selectedCircuits = uiSettings.selectedCircuits;
  const isMultiSelectMode = uiSettings.isMultiSelectMode;
  // Track if reference model was manually hidden
  // Manually hidden state is now managed by UI settings
  const manuallyHidden = uiSettings.manuallyHidden;
  // Configuration name for saving profiles
  const [configurationName, setConfigurationName] = useState<string>('');
  
  // Static rendering state
  const [staticRenderSettings, setStaticRenderSettings] = useState<StaticRenderSettings>(defaultStaticRenderSettings);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_isStaticRendering, _setIsStaticRendering] = useState<boolean>(false);

  // Saved profiles state is now handled by useUserProfiles hook

  
  // Visualization settings - these are now passed to child components but setters not used
   
  // Opacity level is now managed by UI settings
  const opacityLevel = uiSettings.opacityLevel;
  // Log scalar and opacity exponent are now managed by UI settings
  const logScalar = uiSettings.logScalar; // eslint-disable-line @typescript-eslint/no-unused-vars
  const opacityExponent = uiSettings.opacityExponent;
  
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
        console.log(`🔄 Parameter change detected: ${change.type}.${change.field}`);

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
          console.log('💾 Auto-saving configuration changes...', {
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
            console.log('✅ Configuration auto-saved successfully');

            // Log activity with timestamp
            const currentConfig = circuitConfigurations?.find(c => c.id === activeConfigId);
            const activityMessage = `Auto-saved "${currentConfig?.name || 'Current Profile'}" - Parameters: ${Object.entries(parameters).map(([k, v]) => `${k}=${typeof v === 'number' ? v.toFixed(3) : v}`).join(', ')}, Grid: ${gridSize}^5, Freq: ${minFreq}-${maxFreq}Hz (${numPoints}pts)`;

            // Update status with detailed info but don't spam
            const timestamp = new Date().toLocaleTimeString();
            updateStatusMessage(`[${timestamp}] Auto-saved configuration changes`);

            console.log('📝 Activity logged:', activityMessage);
          } else {
            console.warn('⚠️ Configuration auto-save failed');
            updateStatusMessage('⚠️ Auto-save failed - changes may be lost');
          }
        } catch (error) {
          const errorMsg = error instanceof Error ? error.message : String(error);
          console.error('❌ Auto-save error:', errorMsg, error);
          updateStatusMessage(`❌ Auto-save failed: ${errorMsg}`);
        }
      }
    }, 2000); // 2-second delay to avoid excessive saves during parameter adjustment

    return () => clearTimeout(autoSaveTimer);
  }, [parameters, gridSize, minFreq, maxFreq, numPoints, activeConfigId, sessionManagement.sessionState.userId, updateConfiguration, circuitConfigurations]);

  // SRD Upload Handlers
  const handleSRDUploaded = useCallback(async (manager: SerializedComputationManager, metadata: { title: string; totalResults: number; gridSize: number }) => {
    try {
      // Set the serialized manager as the active one
      setSerializedManager(manager);

      // Update grid size and frequency settings to match the SRD
      setGridSize(metadata.gridSize);
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

      // Generate ModelSnapshots for visualization using intelligent limits
      const smartLimit = calculateEffectiveVisualizationLimit(metadata.totalResults);
      const effectiveLimit = Math.min(smartLimit, metadata.totalResults);
      const modelSnapshots = manager.generateModelSnapshots(effectiveLimit);
      setGridResults(modelSnapshots);

      // Generate resnorm groups for visualization (required for spider plot display)
      if (modelSnapshots.length > 0) {
        const resnormValues = modelSnapshots.map(m => m.resnorm);
        const minResnorm = Math.min(...resnormValues);
        const maxResnorm = Math.max(...resnormValues);
        const groups = createResnormGroups(modelSnapshots, minResnorm, maxResnorm);
        setResnormGroups(groups);

        // Switch to VisualizerTab to show the uploaded results
        setActiveTab('visualizer');
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
          console.log('💾 SRD upload: User not authenticated, skipping auto-save profile creation');
          updateStatusMessage('SRD loaded successfully - sign in to save configurations');
          return;
        }

        try {
          const savedProfileId = await handleSaveProfile(profileName, profileDescription, true); // Force new profile
          if (savedProfileId) {
            // Set as active config to maintain tagged models and other data
            setActiveConfiguration(savedProfileId);
            console.log(`💾 Auto-saved SRD config as profile: ${savedProfileId}`);
          }
        } catch (saveError) {
          console.warn('Failed to auto-save SRD profile:', saveError);
        }
      }, 100); // Small delay to ensure handleSaveProfile is available

      // Update status
      const memoryReduction = manager.getStorageStats().reductionFactor;
      updateStatusMessage(`🚀 SRD uploaded: "${metadata.title}" - ${modelSnapshots.length.toLocaleString()}/${metadata.totalResults.toLocaleString()} models loaded (${memoryReduction.toFixed(1)}x memory optimized)`);
      setSrdUploadMessage(`Successfully loaded ${metadata.totalResults.toLocaleString()} results from "${metadata.title}"`);

      // Clear any previous errors
      setGridError('');

      console.log(`📥 SRD Upload Success:`, {
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
  
  // Moved handleNewCircuit after handleSaveProfile definition
  
  // Compute grid function
  const handleCompute = useCallback(async () => {
    if (isComputingGrid) {
      updateStatusMessage('Computation already in progress');
      return;
    }

    console.log('🚀 Starting handleCompute with parameters:', {
      parameters,
      gridSize,
      minFreq,
      maxFreq,
      numPoints,
      isComputingGrid
    });

    try {
      // Ensure there's always a profile before starting computation
      if (!savedProfilesState.selectedProfile) {
        console.log('📝 No profile selected - creating one before computation...');
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
            console.log('⚠️ Pre-compute profile creation failed - proceeding with computation anyway');
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

      console.log(`🔥 COMPUTATION DEBUG: masterLimitPercentage=${centralizedLimits.masterLimitPercentage}%, masterLimitResults=${centralizedLimits.masterLimitResults.toLocaleString()}, expected=${expectedFullGrid.toLocaleString()}, using=${masterLimitToUse.toLocaleString()}`);

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

          console.log(`🚀 MEMORY OPTIMIZED: ${serializedCount} results stored`);
          console.log(`📊 Memory: ${stats.traditionalSizeMB.toFixed(1)}MB → ${stats.serializedSizeMB.toFixed(1)}MB (${stats.reductionFactor.toFixed(1)}x reduction)`);

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

        console.log(`🏁 COMPUTATION COMPLETE: gridSize=${gridSize}, expected=${expectedGridPoints}, computed=${results.length}, previous_actualComputedPoints=${actualComputedPoints}`);

        if (results.length !== actualComputedPoints) {
          console.warn(`⚠️ MISMATCH: Final results (${results.length}) != generated count (${actualComputedPoints})`);
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
        console.log('🔧 Grid configuration automatically saved to localStorage');

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
          console.log('💾 Backup profile created in localStorage');
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

      console.error('❌ Computation Error Details:', {
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
        console.log('🧹 Forced garbage collection after computation');
      }
    }
  }, [
    isComputingGrid, gridSize, parameters, minFreq, maxFreq, numPoints,
    performanceSettings, resnormConfig, hybridComputeManager.computeGridHybrid,
    setIsComputingGrid, setGridError, setComputationProgress, setComputationSummary,
    setGridResults, setTotalGridPoints, setActualComputedPoints,
    savedProfilesState.selectedProfile, profileActions,
    updateStatusMessage, configurationName, extendedSettings,
    hybridComputeManager, centralizedLimits.masterLimitResults
  ]);
  

  // Multi-select functions
  const handleToggleMultiSelect = useCallback(() => {
    setIsMultiSelectMode(!isMultiSelectMode);
    setSelectedCircuits([]);
  }, [isMultiSelectMode]);
  
  const handleSelectCircuit = useCallback(async (configId: string) => {
    if (isMultiSelectMode) {
      const newSelectedCircuits = selectedCircuits.includes(configId)
        ? selectedCircuits.filter(id => id !== configId)
        : [...selectedCircuits, configId];
      setSelectedCircuits(newSelectedCircuits);
    } else {
      // Auto-save current profile before switching (if there's an active one)
      if (activeConfigId && activeConfigId !== configId && sessionManagement.sessionState.userId) {
        try {
          console.log('💾 Auto-saving current profile before switching...', activeConfigId);
          const success = await updateConfiguration(activeConfigId, {
            circuitParameters: parameters,
            gridSize,
            minFreq,
            maxFreq,
            numPoints
          });

          if (success) {
            console.log('✅ Profile auto-saved before switch');
            updateStatusMessage(`Auto-saved changes to current profile before switching`);
          } else {
            console.warn('⚠️ Failed to auto-save before switching - proceeding anyway');
            updateStatusMessage(`⚠️ Could not auto-save before switching`);
          }
        } catch (error) {
          const errorMsg = error instanceof Error ? error.message : 'Unknown error';
          console.warn('⚠️ Failed to auto-save before profile switch:', errorMsg);
          updateStatusMessage(`⚠️ Auto-save failed: ${errorMsg}`);
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
        updateDefaultGridSize(config.gridSize);
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

        console.log('🔄 Loaded profile settings:', {
          name: config.name,
          parameters: config.circuitParameters,
          gridSize: config.gridSize,
          frequency: { min: config.minFreq, max: config.maxFreq, points: config.numPoints }
        });
      } else {
        console.warn('⚠️ Configuration not found:', configId);
        updateStatusMessage(`⚠️ Configuration not found`);
      }
    }
  }, [isMultiSelectMode, selectedCircuits, activeConfigId, sessionManagement.sessionState.userId, updateConfiguration, parameters, gridSize, minFreq, maxFreq, numPoints, resetComputationState, setActiveConfiguration, sessionManagement.actions, circuitConfigurations]);
  
  const handleBulkDelete = useCallback(async () => {
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
      console.log('🔄 Skipping parameter initialization - active profile loaded:', activeConfigId || sessionManagement.sessionState.currentCircuitConfigId);
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

    console.log('🏁 Initializing default parameters (no active profile)');
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

  // Synchronize configurationName with activeConfiguration.name
  useEffect(() => {
    if (activeConfiguration && activeConfiguration.name !== configurationName) {
      console.log('Syncing configuration name:', activeConfiguration.name);
      setConfigurationName(activeConfiguration.name);
    }
  }, [activeConfiguration, configurationName]);

  // Synchronize session's active circuit config with local state
  useEffect(() => {
    const syncActiveConfig = async () => {
      const sessionConfigId = sessionManagement.sessionState.currentCircuitConfigId;
      
      // Case 1: Session has config but local state doesn't - sync to local
      if (sessionConfigId && !activeConfigId) {
        console.log('Syncing session config to local state:', sessionConfigId);
        setActiveConfiguration(sessionConfigId);
        return;
      }
      
      // Case 2: Local has config but session doesn't - sync to session
      if (activeConfigId && !sessionConfigId) {
        console.log('Syncing local config to session:', activeConfigId);
        await sessionManagement.actions.setActiveCircuitConfig(activeConfigId);
        return;
      }
      
      // Case 3: Both exist but differ - session wins (it's the source of truth)
      if (sessionConfigId && activeConfigId && sessionConfigId !== activeConfigId) {
        console.log('Resolving config mismatch - session wins:', sessionConfigId);
        setActiveConfiguration(sessionConfigId);
        return;
      }
      
      // Case 4: Both match or both are null - no action needed
    };

    // Only sync if session is ready
    if (sessionManagement.isReady) {
      syncActiveConfig();
    }
  }, [sessionManagement.sessionState.currentCircuitConfigId, activeConfigId, sessionManagement.isReady, sessionManagement.actions, setActiveConfiguration]);
  
  // Add event listeners for custom events from VisualizerTab
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

  // Helper function for Halton sequence (quasi-random number generation)
  const haltonSequence = useCallback((index: number, base: number): number => {
    let result = 0;
    let fraction = 1 / base;
    let i = index;
    while (i > 0) {
      result += (i % base) * fraction;
      i = Math.floor(i / base);
      fraction /= base;
    }
    return result;
  }, []);

  // Calculate synthetic resnorm based on parameter values and profile characteristics
  const calculateSyntheticResnorm = useCallback((Rsh: number, Ra: number, Ca: number, Rb: number, Cb: number, profile: SavedProfile): number => {
    // Use a realistic model that correlates with actual circuit behavior
    // Lower resistance and higher capacitance generally lead to better fits (lower resnorm)
    
    // Normalize parameters to 0-1 range
    const rsNorm = Math.log10(Rsh / 10) / Math.log10(1000);
    const raNorm = Math.log10(Ra / 10) / Math.log10(1000);
    const rbNorm = Math.log10(Rb / 10) / Math.log10(1000);
    const caNorm = Math.log10(Ca / 0.1e-6) / Math.log10(500);
    const cbNorm = Math.log10(Cb / 0.1e-6) / Math.log10(500);
    
    // Calculate base resnorm with realistic relationships
    const balanceFactor = Math.abs(raNorm - rbNorm); // Penalty for imbalanced resistances
    const capacitanceFactor = (caNorm + cbNorm) / 2; // Higher capacitance = better fit
    const frequencyFactor = Math.log10(profile.maxFreq / profile.minFreq) / 4; // Wider frequency range = more constraints
    
    // Combine factors to create realistic resnorm distribution
    let baseResnorm = 0.01 + 
                     (rsNorm * 0.02) + 
                     (balanceFactor * 0.05) + 
                     ((1 - capacitanceFactor) * 0.08) + 
                     (frequencyFactor * 0.03);
    
    // Add parameter-specific penalties
    if (Ra < 50 || Rb < 50) baseResnorm += 0.02; // Very low resistance penalty
    if (Ca < 1e-6 || Cb < 1e-6) baseResnorm += 0.03; // Very low capacitance penalty
    
    // Add some noise to create realistic distribution
    const noise = 1 + (Math.random() - 0.5) * 0.3; // ±15% noise
    
    return Math.max(0.001, Math.min(0.5, baseResnorm * noise));
  }, []);

  // Generate synthetic profile data for profiles that exceed worker limits
  const generateSyntheticProfileData = useCallback((profile: SavedProfile, sampleSize: number): ModelSnapshot[] => {
    console.log(`Generating ${sampleSize} synthetic data points for profile "${profile.name}" (grid: ${profile.gridSize}^5)`);
    
    const syntheticData: ModelSnapshot[] = [];
    const paramRanges = {
      Rsh: { min: 10, max: 10000 },
      Ra: { min: 10, max: 10000 },
      Rb: { min: 10, max: 10000 },
      Ca: { min: 0.1e-6, max: 50e-6 },
      Cb: { min: 0.1e-6, max: 50e-6 }
    };
    
    // Generate parameter combinations using quasi-random Halton sequence for better distribution
    for (let i = 0; i < sampleSize; i++) {
      // Use Halton sequence bases for better parameter space coverage
      const haltonBase2 = haltonSequence(i, 2);
      const haltonBase3 = haltonSequence(i, 3);
      const haltonBase5 = haltonSequence(i, 5);
      const haltonBase7 = haltonSequence(i, 7);
      const haltonBase11 = haltonSequence(i, 11);
      
      // Map to parameter ranges using logarithmic scaling
      const Rsh = paramRanges.Rsh.min * Math.pow(paramRanges.Rsh.max / paramRanges.Rsh.min, haltonBase2);
      const Ra = paramRanges.Ra.min * Math.pow(paramRanges.Ra.max / paramRanges.Ra.min, haltonBase3);
      const Rb = paramRanges.Rb.min * Math.pow(paramRanges.Rb.max / paramRanges.Rb.min, haltonBase5);
      const Ca = paramRanges.Ca.min * Math.pow(paramRanges.Ca.max / paramRanges.Ca.min, haltonBase7);
      const Cb = paramRanges.Cb.min * Math.pow(paramRanges.Cb.max / paramRanges.Cb.min, haltonBase11);
      
      // Calculate synthetic resnorm using realistic circuit behavior model
      const resnorm = calculateSyntheticResnorm(Rsh, Ra, Ca, Rb, Cb, profile);
      
      // Calculate TER for the synthetic model
      const ter = (Rsh * (Ra + Rb)) / (Rsh + Ra + Rb);
      
      // Generate minimal impedance spectrum for compatibility
      const spectrum: ImpedancePoint[] = [];
      const numFreqPoints = Math.min(profile.numPoints, 10); // Limit spectrum size for performance
      for (let j = 0; j < numFreqPoints; j++) {
        const logMin = Math.log10(profile.minFreq);
        const logMax = Math.log10(profile.maxFreq);
        const logFreq = logMin + (j / (numFreqPoints - 1)) * (logMax - logMin);
        const frequency = Math.pow(10, logFreq);
        
        // Calculate simple impedance at this frequency
        const omega = 2 * Math.PI * frequency;
        const Za_real = Ra / (1 + Math.pow(omega * Ra * Ca, 2));
        const Za_imag = -omega * Ra * Ra * Ca / (1 + Math.pow(omega * Ra * Ca, 2));
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
        const magnitude = Math.sqrt(real * real + imaginary * imaginary);
        const phase = Math.atan2(imaginary, real) * (180 / Math.PI);
        
        spectrum.push({
          frequency,
          real,
          imaginary,
          magnitude,
          phase
        });
      }
      
      syntheticData.push({
        id: `synthetic-${profile.id}-${i}`,
        name: `Synthetic ${profile.name} - Model ${i + 1}`,
        timestamp: Date.now(),
        parameters: {
          Rsh,
          Ra,
          Rb,
          Ca,
          Cb,
          frequency_range: [profile.minFreq, profile.maxFreq]
        },
        data: spectrum,
        resnorm,
        color: '#8B5CF6', // Purple color to distinguish synthetic data
        isVisible: true,
        opacity: 0.7,
        ter
      });
    }
    
    // Sort by resnorm to ensure proper quartile distribution
    return syntheticData.sort((a, b) => (a.resnorm || 0) - (b.resnorm || 0));
  }, [haltonSequence, calculateSyntheticResnorm]);

  // Generate mesh data from saved profile (currently unused, was for OrchestratorTab)
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const generateProfileMeshData = useCallback(async (profile: SavedProfile, sampleSize: number = 1000): Promise<ModelSnapshot[]> => {
    if (!profile) return [];
    
    try {
      // Use the profile's saved parameters to generate grid data
      const profileGridSize = profile.gridSize;
      const profileMinFreq = profile.minFreq;
      const profileMaxFreq = profile.maxFreq;
      const profileNumPoints = profile.numPoints;
      const profileGroundTruthParams = profile.groundTruthParams;
      
      // CRITICAL: Enforce safety limits to prevent crashes
      const WORKER_MAX_GRID_SIZE = 30; // Matches UI validation limit
      const WORKER_MAX_TOTAL_POINTS = 25000000; // 25M points hard limit (30^5 = 24.3M)
      
      const totalPoints = Math.pow(profileGridSize, 5);
      
      // Check if the profile exceeds worker capabilities
      if (profileGridSize > WORKER_MAX_GRID_SIZE || totalPoints > WORKER_MAX_TOTAL_POINTS) {
        console.warn(`Profile "${profile.name}" grid size ${profileGridSize}^5 = ${totalPoints.toLocaleString()} exceeds worker limits (max: ${WORKER_MAX_GRID_SIZE}^5 = ${Math.pow(WORKER_MAX_GRID_SIZE, 5).toLocaleString()})`);
        
        // Generate synthetic data instead to prevent crashes
        updateStatusMessage(`Profile "${profile.name}" exceeds worker limits - generating synthetic dataset...`);
        
        return generateSyntheticProfileData(profile, sampleSize);
      }
      
      // Additional safety check for very large grids
      const estimatedMemoryMB = (totalPoints * 800) / (1024 * 1024); // ~800 bytes per model
      if (estimatedMemoryMB > 4000) { // 4GB memory limit for high-end configurations
        console.warn(`Profile "${profile.name}" estimated memory usage ${estimatedMemoryMB.toFixed(0)}MB exceeds safe limits`);
        updateStatusMessage(`Profile "${profile.name}" memory requirements too high - using synthetic data...`);
        
        return generateSyntheticProfileData(profile, sampleSize);
      }
      
      // Use a conservative grid size that's guaranteed to work
      const safeGridSize = Math.min(profileGridSize, WORKER_MAX_GRID_SIZE);
      const actualSampleSize = Math.min(sampleSize, Math.pow(safeGridSize, 5));
      
      updateStatusMessage(`Generating grid data from profile "${profile.name}" (safe grid: ${safeGridSize}^5 = ${Math.pow(safeGridSize, 5).toLocaleString()})...`);
      
      // Use ultra-conservative performance settings for large computations
      const conservativeSettings: PerformanceSettings = {
        ...DEFAULT_PERFORMANCE_SETTINGS,
        useSymmetricGrid: true, // Enable symmetric optimization
        useParallelProcessing: true, // Keep parallel processing enabled
        enableSmartBatching: true, // Use smart batching
        chunkSize: Math.floor(DEFAULT_PERFORMANCE_SETTINGS.chunkSize * 0.5), // Use smaller chunks
        maxMemoryMB: Math.min(DEFAULT_PERFORMANCE_SETTINGS.maxMemoryMB, 1024), // Limit memory to 1GB
        enableProgressiveRendering: true, // Show progress
        performanceWarnings: true // Enable warnings
      };
      
      // Use the worker to generate mesh data with timeout protection
      const gridResults = await Promise.race([
        hybridComputeManager.computeGridHybrid(
          profileGroundTruthParams,
          safeGridSize,
          profileMinFreq,
          profileMaxFreq,
          profileNumPoints,
          conservativeSettings,
          extendedSettings,
          resnormConfig,
          (progress) => {
            // Update status with progress
            if (progress.overallProgress) {
              updateStatusMessage(`Generating profile data: ${Math.round(progress.overallProgress)}%`);
            }
          },
          (error) => {
            console.error('Profile generation error:', error);
            throw new Error(`Worker error: ${error}`);
          },
          centralizedLimits.masterLimitResults
        ),
        // 5 minute timeout to prevent infinite hangs
        new Promise<never>((_, reject) => 
          setTimeout(() => reject(new Error('Profile generation timeout (5 minutes)')), 5 * 60 * 1000)
        )
      ]);
      
      if (!gridResults || gridResults.results.length === 0) {
        throw new Error('No grid results generated');
      }
      
      // Convert BackendMeshPoint to ModelSnapshot format and sample if needed
      const modelSnapshots: ModelSnapshot[] = gridResults.results.map((result, index) => {
        const ter = (result.parameters.Rsh * (result.parameters.Ra + result.parameters.Rb)) / 
                   (result.parameters.Rsh + result.parameters.Ra + result.parameters.Rb);
        
        // Convert spectrum to ImpedancePoint format
        const data: ImpedancePoint[] = result.spectrum.map(spec => ({
          real: spec.real,
          imaginary: spec.imag,
          frequency: spec.freq,
          magnitude: spec.mag,
          phase: spec.phase
        }));
        
        return {
          id: `profile-${profile.id}-${index}`,
          name: `Profile ${profile.name} - Model ${index + 1}`,
          parameters: result.parameters,
          data,
          resnorm: result.resnorm,
          color: '#f97316',
          opacity: 0.7,
          isVisible: true,
          timestamp: Date.now(),
          ter
        };
      });
      
      // Sample down to requested size if needed
      if (modelSnapshots.length > actualSampleSize) {
        const step = Math.floor(modelSnapshots.length / actualSampleSize);
        const sampledModels = modelSnapshots.filter((_, index) => index % step === 0).slice(0, actualSampleSize);
        updateStatusMessage(`Sampled ${sampledModels.length} models from profile "${profile.name}"`);
        return sampledModels;
      }
      
      updateStatusMessage(`Generated ${modelSnapshots.length} models from profile "${profile.name}"`);
      return modelSnapshots;
      
    } catch (error) {
      console.error('Error generating profile mesh data:', error);
      updateStatusMessage(`Error generating profile data: ${error instanceof Error ? error.message : 'Unknown error'}`);
      
      // Fallback to synthetic data generation if worker computation fails
      try {
        updateStatusMessage(`Fallback: generating synthetic data for profile "${profile.name}"...`);
        return generateSyntheticProfileData(profile, sampleSize);
      } catch (syntheticError) {
        console.error('Synthetic data generation also failed:', syntheticError);
        updateStatusMessage(`Failed to generate any data for profile "${profile.name}"`);
        return [];
      }
    }
  }, [hybridComputeManager.computeGridHybrid, updateStatusMessage, generateSyntheticProfileData, resnormConfig, extendedSettings, hybridComputeManager, centralizedLimits.masterLimitResults]);

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
            console.log(`📊 GENERATION DEBUG: generated=${progress.generated}, total=${progress.total}, processed=${progress.processed || 'unknown'}, skipped=${progress.skipped || 'unknown'}`);
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
      
      console.log('🔄 Starting automatic profile save after grid computation...');
      try {
        const savedProfileId = await handleSaveProfile(autoSaveName, `Last action: Grid computation completed - ${sortedResults.length} models computed`);
        if (savedProfileId) {
          console.log('✅ Profile auto-saved successfully:', savedProfileId);
          // Force immediate UI update
          setTimeout(() => {
            profileActions.refreshProfiles();
          }, 50);
          updateStatusMessage(`Configuration auto-saved as "${autoSaveName}" - visible in left menu`);
        } else {
          console.error('❌ Auto-save failed - retrying...');
          // Retry with simplified name
          const retryName = `Auto-save ${Date.now()}`;
          const retryId = await handleSaveProfile(retryName, `Last action: Grid computation completed (retry save) - ${sortedResults.length} models computed`);
          if (retryId) {
            console.log('✅ Retry auto-save successful:', retryId);
            updateStatusMessage(`Configuration saved as "${retryName}"`);
          }
        }
      } catch (error) {
        console.error('❌ Auto-save error:', error);
        updateStatusMessage('Grid computed successfully but auto-save failed');
      }
      
      // Clear the configuration name after saving
      setConfigurationName('');

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
    console.log('💾 Starting circuit configuration save:', { name, description, parameters, gridSize, minFreq, maxFreq, numPoints, isAutoSave, forceNew });
    
    if (!sessionManagement.sessionState.userId) {
      console.log('💾 No user ID for saving circuit configuration - user not authenticated');
      if (!isAutoSave) {
        // Only show message for manual saves, not auto-saves
        updateStatusMessage('Please sign in to save circuit configurations');
      }
      return null;
    }

    // Handle different save modes
    if (forceNew === false && activeConfigId) {
      // Update mode: Update the currently active configuration
      console.log('🔄 Updating current circuit configuration:', activeConfigId);
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
          console.log('📝 Profile Update Activity:', activityDetails);
          return activeConfigId;
        } else {
          updateStatusMessage(`❌ Failed to update configuration "${name}"`);
          return null;
        }
      } catch (error) {
        console.error('❌ Configuration update error:', error);
        updateStatusMessage(`Error updating configuration: ${error instanceof Error ? error.message : 'Unknown error'}`);
        return null;
      }
    }
    
    if (forceNew === true) {
      // Save As New mode: Always create a new configuration
      console.log('🆕 Force creating new circuit configuration...');
    } else if (activeConfigId) {
      // Auto-save mode: Update existing active configuration
      console.log('🔄 Auto-saving to existing configuration:', activeConfigId);
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
          updateStatusMessage(`[${timestamp}] 💾 Auto-saved "${currentConfig?.name || 'Current Profile'}"`);
          console.log('📝 Manual Save Activity: Auto-saved existing profile');
          return activeConfigId;
        }
      } catch (error) {
        console.log('⚠️ Auto-save to existing config failed, creating new one:', error);
      }
    }
    
    // Create new configuration (either force new, no active config, or auto-save failed)
    console.log('🆕 Creating new circuit configuration...');
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
        console.log('✅ Circuit configuration created:', newConfig.id);
        // Set as active and sync with session
        await sessionManagement.actions.setActiveCircuitConfig(newConfig.id);
        // Enhanced activity logging for new profile creation
        const timestamp = new Date().toLocaleTimeString();
        const activityDetails = `Created "${newConfig.name}" - Grid: ${gridSize}^5, Freq: ${minFreq}-${maxFreq}Hz (${numPoints}pts), Parameters: Rs=${parameters.Rsh?.toFixed(1)}, Ra=${parameters.Ra?.toFixed(1)}, Ca=${(parameters.Ca * 1e6)?.toFixed(2)}µF, Rb=${parameters.Rb?.toFixed(1)}, Cb=${(parameters.Cb * 1e6)?.toFixed(2)}µF`;

        updateStatusMessage(`[${timestamp}] 🆕 Created "${newConfig.name}" successfully`);
        console.log('📝 New Profile Activity:', activityDetails);
        return newConfig.id;
      } else {
        console.error('❌ Configuration creation returned null');
        updateStatusMessage(`Failed to save configuration "${name}"`);
        return null;
      }
    } catch (error) {
      console.error('❌ Configuration creation error:', error);
      updateStatusMessage(`Error saving configuration "${name}": ${error instanceof Error ? error.message : 'Unknown error'}`);
      return null;
    }
  }, [gridSize, minFreq, maxFreq, numPoints, parameters, updateStatusMessage, activeConfigId, createConfiguration, updateConfiguration, sessionManagement, generateUniqueCircuitName]);

  // Function to create a new circuit with standard configuration (moved after handleSaveProfile)
  const handleNewCircuit = useCallback(async () => {
    // Clear any existing results
    resetComputationState();

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
      console.log('🆕 Creating new circuit profile automatically...');
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

  // Handle configuration name changes with auto-save
  const handleConfigurationNameChange = useCallback(async (newName: string) => {
    setConfigurationName(newName);
    
    // If there's an active configuration and the name is not empty, auto-update the configuration
    if (activeConfigId && activeConfiguration && newName.trim() && newName.trim() !== activeConfiguration.name) {
      try {
        const success = await updateConfiguration(activeConfigId, {
          name: newName.trim(),
          description: activeConfiguration.description,
          circuitParameters: parameters,
          gridSize,
          minFreq,
          maxFreq,
          numPoints
        });
        
        if (success) {
          console.log('✅ Configuration name updated automatically');
        }
      } catch (error) {
        console.error('❌ Failed to auto-update configuration name:', error);
      }
    }
  }, [activeConfigId, activeConfiguration, parameters, gridSize, minFreq, maxFreq, numPoints, updateConfiguration]);



  // Note: Grid generation and spectrum calculation now handled by Web Workers

  // State for collapsible sidebar
  const [leftNavCollapsed, setLeftNavCollapsed] = useState(false);

  // Track if we've loaded from localStorage to avoid hydration issues
  const [hasLoadedFromStorage, setHasLoadedFromStorage] = useState(false);
  // Removed unused sample profile state
  
  // Track profile loading for computation
  // pendingComputeProfile is now handled by the useUserProfiles hook
  

// Handler for copying profile parameters
  const handleCopyParams = useCallback((configId: string) => {
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
      console.log('🏷️ No user logged in, skipping tagged models load');
      return;
    }

    // Use activeConfigId instead of legacy selectedProfile
    if (!activeConfigId) {
      console.log('🏷️ No active circuit configuration, clearing tagged models');
      setTaggedModels(new Map());
      return;
    }

    try {
      console.log('🏷️ Loading tagged models for user:', user.id, 'circuit config:', activeConfigId);
      const { supabase } = await import('../../lib/supabase');
      
      // Query tagged models by user_id AND circuit_config_id (circuit-specific)
      const { data: dbTaggedModels, error } = await supabase
        .from('tagged_models')
        .select('*')
        .eq('user_id', user.id)
        .eq('circuit_config_id', activeConfigId) // FIXED: Use correct schema field
        .order('tagged_at', { ascending: false });

      if (error) {
        console.error('❌ Failed to load tagged models from database:', error);
        console.error('❌ Error details:', error);
        return;
      }

      if (dbTaggedModels && dbTaggedModels.length > 0) {
        console.log(`🏷️ Loaded ${dbTaggedModels.length} tagged models from database`);
        
        // Convert database tagged models to local state format
        const newTaggedModels = new Map<string, { tagName: string; profileId: string; resnormValue: number; taggedAt: number; notes?: string }>();
        
        dbTaggedModels.forEach(dbModel => {
          // Type cast to handle schema fields that aren't in generated types
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const model = dbModel as any;
          newTaggedModels.set(model.model_id, {
            tagName: model.tag_name,
            profileId: model.circuit_config_id || 'default', // FIXED: Use correct schema field
            resnormValue: model.resnorm_value || 0,
            taggedAt: new Date(model.tagged_at).getTime(),
            notes: model.notes || undefined
          });
        });

        setTaggedModels(newTaggedModels);
        updateStatusMessage(`📊 Loaded ${dbTaggedModels.length} tagged models for current circuit configuration`);
      } else {
        console.log('🏷️ No tagged models found for current circuit configuration');
        setTaggedModels(new Map());
      }
    } catch (error) {
      console.error('❌ Error loading tagged models from database:', error);
    }
  }, [user, activeConfigId, updateStatusMessage]); // FIXED: Use activeConfigId dependency

  // Load tagged models when user or active circuit config changes
  useEffect(() => {
    if (user && sessionManagement.isReady && activeConfigId) {
      loadTaggedModelsFromDatabase();
    }
  }, [user, sessionManagement.isReady, activeConfigId, loadTaggedModelsFromDatabase]);

  // Handle model tagging
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
        tagCategory: 'user',
        circuitParameters: model.parameters,
        resnormValue: model.resnorm,
        isInteresting: true
      });

      // Update local state
      const newTaggedModels = new Map(taggedModels);
      newTaggedModels.set(model.id, {
        tagName: taggedModel.tagName,
        profileId: taggedModel.circuitConfigId,
        resnormValue: taggedModel.resnormValue || 0,
        taggedAt: new Date(taggedModel.taggedAt).getTime(),
        notes: taggedModel.notes
      });
      setTaggedModels(newTaggedModels);
      updateStatusMessage(`🏷️ Tagged model: ${tagName}`);

    } catch (error) {
      console.error('❌ Failed to tag model:', error);
      updateStatusMessage('❌ Failed to tag model');
    }
  }, [user, activeConfigId, taggedModels, updateStatusMessage]);

  // Handle static render job creation
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const _handleCreateRenderJob = useCallback(async (settings: StaticRenderSettings, meshData: ModelSnapshot[]) => {
    _setIsStaticRendering(true);
    updateStatusMessage(`Starting high-quality export (${meshData.length.toLocaleString()} models)...`);
    
    try {
      // For now, just simulate the render process
      // TODO: Implement actual orchestrator worker integration
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Create a simple export for now using the existing SpiderPlot export
      const link = document.createElement('a');
      link.download = `spideyplot-export-${Date.now()}.${settings.format}`;
      
      // TODO: Replace with actual high-quality rendering
      updateStatusMessage(`Export completed (simulated) - ${settings.format.toUpperCase()} format`);
      
    } catch (error) {
      console.error('Static render error:', error);
      updateStatusMessage(`Export failed: ${error}`);
    } finally {
      _setIsStaticRendering(false);
    }
  }, [updateStatusMessage]);

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
    <div className="h-screen bg-black text-white flex overflow-hidden">
      {/* Left Navigation Sidebar - Full Height like ChatGPT */}
      <div 
        className={`${leftNavCollapsed ? 'w-16' : 'w-64'} bg-neutral-900 flex flex-col transition-all duration-300 ease-in-out h-full relative group`}
      >
        {/* Header with logo and collapse functionality */}
        <div className="p-4 flex-shrink-0 relative">
          <div className={`flex items-center transition-all duration-300 ${leftNavCollapsed ? 'justify-center' : 'justify-between'}`}>
            {/* Logo section - refresh when expanded, centered when collapsed */}
            <button
              onClick={() => {
                if (!leftNavCollapsed) {
                  window.location.reload();
                }
              }}
              className="w-10 h-10 flex items-center justify-center rounded-md hover:bg-neutral-700 transition-all duration-200"
              title={leftNavCollapsed ? "SpideyPlot" : "Refresh page"}
            >
              <Image 
                src="/logo.png"
                alt="SpideyPlot" 
                width={24}
                height={24}
                className="flex-shrink-0"
              />
            </button>
            
            {/* Collapse button - only visible when expanded */}
            {!leftNavCollapsed && (
              <button
                onClick={() => setLeftNavCollapsed(true)}
                className="w-10 h-10 flex items-center justify-center rounded-md hover:bg-neutral-700 transition-all duration-200"
                title="Collapse sidebar"
              >
                <svg
                  className="w-4 h-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <rect x="3" y="3" width="18" height="18" rx="2" strokeWidth={2} />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v18" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 8l-3 4 3 4" />
                </svg>
              </button>
            )}
          </div>
          
          
          {/* Expand button - positioned to appear over the logo when hovered */}
          {leftNavCollapsed && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                setLeftNavCollapsed(false);
              }}
              className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-10 h-10 flex items-center justify-center rounded-md bg-neutral-700 hover:bg-neutral-600 transition-all duration-200 opacity-0 group-hover:opacity-100 z-10"
              title="Expand sidebar"
            >
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <rect x="3" y="3" width="18" height="18" rx="2" strokeWidth={2} />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v18" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 8l3 4-3 4" />
              </svg>
            </button>
          )}
        </div>

        {/* Navigation content - show when expanded */}
        <div className={`flex-1 transition-all duration-300 ${leftNavCollapsed ? 'opacity-0 pointer-events-none' : 'opacity-100'} flex flex-col overflow-hidden`}>
          {/* New Circuit Button - Expanded */}
          <div className="p-3 pb-2 flex-shrink-0">
            <button
              onClick={handleNewCircuit}
              className="w-full bg-orange-600 hover:bg-orange-700 text-white px-4 py-2.5 rounded-md text-sm font-medium transition-colors duration-200 flex items-center justify-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              New Circuit
            </button>
          </div>
          
          {/* Navigation Tabs - Extended */}
          <div className="px-3 pb-3 space-y-1 flex-shrink-0">
            <button 
              className={`w-full text-left px-3 py-2.5 rounded-md text-sm font-medium transition-all duration-200 ${
                visualizationTab === 'visualizer' 
                  ? 'bg-neutral-800 text-white' 
                  : 'text-neutral-400 hover:bg-neutral-800 hover:text-white'
              }`}
              onClick={() => setVisualizationTab('visualizer')}
            >
              <div className="flex items-center gap-3">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                </svg>
                <span>Playground</span>
              </div>
            </button>
            <button 
              className={`w-full text-left px-3 py-2.5 rounded-md text-sm font-medium transition-all duration-200 ${
                visualizationTab === 'math'
                  ? 'bg-neutral-800 text-white' 
                  : 'text-neutral-400 hover:bg-neutral-800 hover:text-white'
              }`}
              onClick={() => setVisualizationTab('math')}
            >
              <div className="flex items-center gap-3">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
                <span>Model</span>
              </div>
            </button>
            <button 
              className={`w-full text-left px-3 py-2.5 rounded-md text-sm font-medium transition-all duration-200 ${
                visualizationTab === 'activity' 
                  ? 'bg-neutral-800 text-white' 
                  : 'text-neutral-400 hover:bg-neutral-800 hover:text-white'
              }`}
              onClick={() => setVisualizationTab('activity')}
            >
              <div className="flex items-center gap-3">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <span>Activity Log</span>
              </div>
            </button>
            {/* Orchestrator tab removed - functionality integrated into Playground */}
          </div>
          
          {/* Saved Profiles Section - Scrollable - Always Show */}
          <SavedProfiles
              profiles={convertedProfiles} // FIXED: Use circuit configurations instead of legacy profiles
              selectedProfile={activeConfigId} // FIXED: Use activeConfigId instead of selectedProfile
              onCopyParams={handleCopyParams}
              selectedCircuits={selectedCircuits}
              isMultiSelectMode={isMultiSelectMode}
              onToggleMultiSelect={handleToggleMultiSelect}
              onBulkDelete={handleBulkDelete}
              computingProfileId={isComputingGrid ? activeConfigId : null} // FIXED: Use activeConfigId
            onSelectProfile={handleSelectCircuit}
            onSelectProfileOriginal={(configId) => {
              // Clear existing grid results when switching configurations
              resetComputationState();
              
              // Set as active configuration
              setActiveConfiguration(configId);
              
              // Sync with session management
              sessionManagement.actions.setActiveCircuitConfig(configId);
              
              const config = circuitConfigurations?.find(c => c.id === configId);
              if (config) {
                // Load configuration settings into current state
                setGridSize(config.gridSize);
                updateDefaultGridSize(config.gridSize);
                setMinFreq(config.minFreq);
                setMaxFreq(config.maxFreq);
                setNumPoints(config.numPoints);
                setParameters(config.circuitParameters);
                
                // Auto-fill configuration name
                setConfigurationName(config.name);
                
                // Update frequencies based on loaded settings
                updateFrequencies(config.minFreq, config.maxFreq, config.numPoints);
                
                // Mark parameters as changed to enable recompute
                setParameterChanged(true);
                
                updateStatusMessage(`Loaded configuration: ${config.name} - Previous grid data cleared, ready to compute`);
              }
            }}
            onDeleteProfile={async (configId) => {
              // Clear grid results if we're deleting the currently active configuration
              const wasSelected = activeConfigId === configId;
              if (wasSelected) {
                resetComputationState();
              }
              
              const success = await deleteConfiguration(configId);
              if (success) {
                updateStatusMessage(wasSelected ? 'Configuration deleted and grid data cleared' : 'Configuration deleted');
              } else {
                updateStatusMessage('Failed to delete configuration');
              }
            }}
            onEditProfile={async (configId, name, description) => {
              const success = await updateConfiguration(configId, { name, description });
              if (success) {
                updateStatusMessage(`Configuration "${name}" updated`);
              } else {
                updateStatusMessage(`Failed to update configuration "${name}"`);
              }
            }}
            onEditParameters={(configId) => {
              const config = circuitConfigurations?.find(c => c.id === configId);
              if (config) {
                // Expand the sidebar
                setLeftNavCollapsed(false);
                
                // Clear existing grid results
                resetComputationState();
                
                // Load configuration settings into center panel for editing
                setGridSize(config.gridSize);
                updateDefaultGridSize(config.gridSize);
                setMinFreq(config.minFreq);
                setMaxFreq(config.maxFreq);
                setNumPoints(config.numPoints);
                setParameters(config.circuitParameters);
                
                // Auto-fill configuration name for editing
                setConfigurationName(config.name);
                
                // Update frequencies
                updateFrequencies(config.minFreq, config.maxFreq, config.numPoints);
                
                // Set as active configuration
                setActiveConfiguration(configId);
                sessionManagement.actions.setActiveCircuitConfig(configId);
                
                // Mark parameters as changed to enable editing mode
                setParameterChanged(true);
                
                updateStatusMessage(`Editing parameters for "${config.name}" - make changes in center panel`);
              }
            }}
            onComputeProfile={(configId) => {
              const config = circuitConfigurations?.find(c => c.id === configId);
              if (config) {
                
                // Clear existing grid results first
                resetComputationState();
                
                // Load configuration settings
                setGridSize(config.gridSize);
                updateDefaultGridSize(config.gridSize);
                setMinFreq(config.minFreq);
                setMaxFreq(config.maxFreq);
                setNumPoints(config.numPoints);
                setParameters(config.circuitParameters);
                
                // Auto-fill configuration name
                setConfigurationName(config.name);
                
                // Update frequencies
                updateFrequencies(config.minFreq, config.maxFreq, config.numPoints);
                
                // Set as active configuration
                setActiveConfiguration(configId);
                sessionManagement.actions.setActiveCircuitConfig(configId);
                
                // Mark parameters as loaded and ready for computation
                setParameterChanged(false); // Parameters are fresh from saved config
                
                updateStatusMessage(`Loading configuration "${config.name}" parameters...`);
                
                // Auto-trigger computation using the same function as center panel
                setTimeout(() => {
                  handleCompute();
                }, 100);
              }
            }}
            isCollapsed={leftNavCollapsed}
                />
          {/* End of Saved Profiles Section */}
        </div>
        
        {/* Collapsed state icons overlay - only show when collapsed */}
        <div className={`absolute inset-0 flex flex-col transition-all duration-300 ${leftNavCollapsed ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'}`}>
          {/* Skip header area - updated for wider collapsed bar */}
          <div className="h-20 flex-shrink-0"></div>
          
          {/* New Circuit Button - Collapsed */}
          <div className="flex-shrink-0 px-3 pb-2">
            <button
              onClick={handleNewCircuit}
              className="w-10 h-10 flex items-center justify-center rounded-md bg-orange-600 hover:bg-orange-700 text-white transition-colors duration-200"
              title="New Circuit"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
            </button>
          </div>
          
          {/* Vertical tab icons */}
          <div className="flex-1 flex flex-col items-center py-2 space-y-1">
                        <button 
              className={`w-10 h-10 flex items-center justify-center rounded-md transition-all duration-200 ${
                visualizationTab === 'visualizer' 
                  ? 'bg-neutral-800 text-white' 
                  : 'text-neutral-500 hover:bg-neutral-800 hover:text-white'
              }`}
              onClick={() => setVisualizationTab('visualizer')}
              title="Playground"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
              </svg>
            </button>
            <button 
              className={`w-10 h-10 flex items-center justify-center rounded-md transition-all duration-200 ${
                visualizationTab === 'math' 
                  ? 'bg-neutral-800 text-white' 
                  : 'text-neutral-500 hover:bg-neutral-800 hover:text-white'
              }`}
              onClick={() => setVisualizationTab('math')}
              title="Model"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
            </button>
            <button 
              className={`w-10 h-10 flex items-center justify-center rounded-md transition-all duration-200 ${
                visualizationTab === 'data' 
                  ? 'bg-neutral-800 text-white' 
                  : 'text-neutral-500 hover:bg-neutral-800 hover:text-white'
              }`}
              onClick={() => setVisualizationTab('data')}
              title="File Manager"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
              </svg>
            </button>
            <button 
              className={`w-10 h-10 flex items-center justify-center rounded-md transition-all duration-200 ${
                visualizationTab === 'activity' 
                  ? 'bg-neutral-800 text-white' 
                  : 'text-neutral-500 hover:bg-neutral-800 hover:text-white'
              }`}
              onClick={() => setVisualizationTab('activity')}
              title="Activity Log"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </button>
            {/* Orchestrator tab removed - functionality integrated into Playground */}
          </div>
        </div>
      </div>

      {/* Main Content Area - Full Width Layout */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Slim Navbar */}
        <SlimNavbar
          statusMessage={statusMessage}
          gridResults={gridResults}
          gridSize={parameters.gridSize}
          onSettingsOpen={() => setSettingsModalOpen(true)}
          user={user}
          onSignOut={signOut}
        />

        {/* Main visualization area with right sidebar */}
        <div className="flex-1 flex overflow-hidden">
          <div className={`flex-1 p-4 bg-neutral-950 overflow-hidden transition-all duration-300 ${leftNavCollapsed ? 'ml-0' : ''}`}>
            {isComputingGrid ? (
              <div className="flex flex-col h-full bg-neutral-900/50 border border-neutral-700 rounded-lg shadow-md overflow-hidden">
                {/* Header with Progress */}
                <div className="flex-shrink-0 p-4 border-b border-neutral-700">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center">
                      <div className="animate-spin rounded-full h-6 w-6 border-t-2 border-b-2 border-orange-400 mr-3"></div>
                      <div>
                        <p className="text-lg text-neutral-200 font-medium">
                          Mathematical Circuit Analysis
                        </p>
                        <p className="text-sm text-neutral-400">
                          {computationProgress ? 
                            `${computationProgress.overallProgress.toFixed(1)}% complete` : 
                            'Initializing computational pipeline...'}
                        </p>
                      </div>
                    </div>
                    {/* Cancel button */}
                    <button
                      onClick={handleCancelComputation}
                      className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-md text-sm font-medium transition-colors duration-200 flex items-center gap-2"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                      Cancel
                    </button>
                  </div>
                  
                  {/* Progress bar */}
                  {computationProgress && (
                    <div className="w-full">
                      <div className="w-full bg-neutral-700 rounded-full h-2">
                        <div 
                          className="bg-orange-500 h-2 rounded-full transition-all duration-300" 
                          style={{ width: `${computationProgress.overallProgress}%` }}
                        ></div>
                      </div>
                    </div>
                  )}
                </div>
                
                {/* Mathematical Process Details Panel */}
                <div className="flex-1 p-4 overflow-y-auto">
                  <div className="space-y-4">
                    {/* Current Operation */}
                    <div className="bg-neutral-800 rounded-lg p-4">
                      <div className="flex items-center mb-3">
                        <div className="w-2 h-2 bg-blue-400 rounded-full mr-2 animate-pulse"></div>
                        <h3 className="text-sm font-semibold text-neutral-200">
                          {computationProgress?.phase === 'initialization' && 'Initialization Phase'}
                          {computationProgress?.phase === 'grid_generation' && 'Parameter Grid Generation'}
                          {computationProgress?.phase === 'impedance_calculation' && 'Impedance Calculations'}
                          {computationProgress?.phase === 'resnorm_analysis' && 'Residual Norm Analysis'}
                          {computationProgress?.phase === 'data_aggregation' && 'Data Aggregation'}
                          {computationProgress?.phase === 'completion' && 'Finalizing Results'}
                          {!computationProgress?.phase && 'Setting up computation'}
                        </h3>
                      </div>
                      <p className="text-xs text-neutral-400 mb-3">
                        {computationProgress?.message || 'Preparing mathematical models...'}
                      </p>
                      
                      {/* Current equation being processed */}
                      {computationProgress?.equation && (
                        <div className="bg-neutral-900 rounded p-3 mb-3">
                          <h4 className="text-xs font-semibold text-neutral-300 mb-2">Current Equation:</h4>
                          <div className="font-mono text-xs text-green-300 overflow-x-auto">
                            {computationProgress.equation}
                          </div>
                        </div>
                      )}
                      
                      {/* Computation details */}
                      <div className="grid grid-cols-2 gap-3 text-xs">
                        {computationProgress?.workerCount && (
                          <div className="flex justify-between">
                            <span className="text-neutral-400">Workers:</span>
                            <span className="text-green-400 font-mono">{computationProgress.workerCount}</span>
                          </div>
                        )}
                        {computationProgress?.processed && computationProgress?.total && (
                          <div className="flex justify-between">
                            <span className="text-neutral-400">Progress:</span>
                            <span className="text-orange-400 font-mono">{Math.round((computationProgress.processed / computationProgress.total) * 100)}%</span>
                          </div>
                        )}
                        {computationProgress?.chunkIndex && computationProgress?.totalChunks && (
                          <div className="flex justify-between">
                            <span className="text-neutral-400">Chunk:</span>
                            <span className="text-purple-400 font-mono">{computationProgress.chunkIndex}/{computationProgress.totalChunks}</span>
                          </div>
                        )}
                        {computationProgress?.memoryPressure && (
                          <div className="flex justify-between">
                            <span className="text-neutral-400">Memory:</span>
                            <span className="text-yellow-400 font-mono">High usage</span>
                          </div>
                        )}
                      </div>
                    </div>
                    
                    {/* Recent Activity Log */}
                    <div className="bg-neutral-800 rounded-lg p-4">
                      <h3 className="text-sm font-semibold text-neutral-200 mb-3 flex items-center">
                        <svg className="w-4 h-4 mr-2 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                        Recent Activity
                      </h3>
                      <div className="bg-neutral-900 rounded p-3 max-h-32 overflow-y-auto">
                        <div className="space-y-1 font-mono text-xs">
                          {logMessages.length > 0 ? (
                            [...logMessages].reverse().slice(0, 8).map((log, index) => (
                              <div key={index} className="flex">
                                <span className="text-neutral-500 mr-2 flex-shrink-0">{log.time}</span>
                                <span className="text-neutral-300 truncate">{log.message}</span>
                              </div>
                            ))
                          ) : (
                            <div className="text-neutral-500 text-center py-2">
                              Activity log will appear here...
                            </div>
                          )}
                          {/* Current status at top */}
                          {statusMessage && (
                            <div className="flex border-t border-neutral-700 pt-2 mt-2">
                              <span className="text-orange-400 mr-2 flex-shrink-0">NOW</span>
                              <span className="text-blue-300 font-medium truncate">{statusMessage}</span>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ) : gridError ? (
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
            ) : visualizationTab === 'math' ? (
              <div className="h-full overflow-y-auto">
                <MathDetailsTab 
                  parameters={parameters}
                  minFreq={minFreq}
                  maxFreq={maxFreq}
                  numPoints={numPoints}
                  referenceModel={referenceModel}
                />
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
                      <div className="text-green-400 font-mono">{(performanceLog.totalDuration / 1000).toFixed(2)}s</div>
                      <div className="text-neutral-400">Bottleneck:</div>
                      <div className="text-orange-400 font-mono">{performanceLog.bottleneck}</div>
                      <div className="text-neutral-400">Throughput:</div>
                      <div className="text-orange-400 font-mono">{performanceLog.efficiency.throughput.toFixed(0)} pts/s</div>
                      <div className="text-neutral-400">Memory:</div>
                      <div className="text-purple-400 font-mono">~{performanceLog.efficiency.memoryUsage.toFixed(1)}MB</div>
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
            ) : visualizationTab === 'visualizer' ? (
              // Show visualization if we have grid results (current portion-based system)
              gridResults && gridResults.length > 0 ? (
                <div className="h-full">
                  <VisualizerTab
                    resnormGroups={resnormGroups.length > 0 ? resnormGroups : []}
                    gridResults={gridResults}
                    hiddenGroups={hiddenGroups}
                    opacityLevel={opacityLevel}
                    referenceModelId={referenceModelId}
                    gridSize={gridSize}
                    onGridValuesGenerated={() => {}}
                    opacityExponent={opacityExponent}
                    onOpacityExponentChange={setOpacityExponent}
                    userReferenceParams={parameters}
                    staticRenderSettings={staticRenderSettings}
                    onStaticRenderSettingsChange={setStaticRenderSettings}
                    groundTruthParams={parameters}
                    numPoints={numPoints}
                    resnormConfig={resnormConfig}
                    groupPortion={staticRenderSettings.groupPortion}
                    onGroupPortionChange={(value) => setStaticRenderSettings({...staticRenderSettings, groupPortion: value})}
                    taggedModels={taggedModels}
                    onModelTag={handleModelTag}
                  />
                </div>
              ) : (
                <div className="h-full flex items-start justify-center p-4 overflow-y-auto">
                  <div className="w-full max-w-4xl">
                    <CenterParameterPanel
                      circuitParams={parameters}
                      onCircuitParamsChange={setParameters}
                      gridSize={gridSize}
                      onGridSizeChange={setGridSize}
                      staticRenderSettings={staticRenderSettings}
                      onStaticRenderSettingsChange={setStaticRenderSettings}
                      minFreq={minFreq}
                      maxFreq={maxFreq}
                      onMinFreqChange={setMinFreq}
                      onMaxFreqChange={setMaxFreq}
                      numPoints={numPoints}
                      onNumPointsChange={setNumPoints}
                      onCompute={handleCompute}
                      onSaveProfile={handleSaveProfile}
                      isComputing={isComputingGrid}
                      configurationName={configurationName}
                      onConfigurationNameChange={handleConfigurationNameChange}
                      selectedProfileId={savedProfilesState.selectedProfile}
                      maxComputationResults={centralizedLimits.masterLimitResults}
                      onMaxComputationResultsChange={(value) => {
                        const percentage = (value / Math.pow(gridSize, 5)) * 100;
                        handleMasterLimitChange(percentage);
                      }}
                      onSRDUploaded={handleSRDUploaded}
                      onUploadError={handleSRDUploadError}
                    />
                  </div>
                </div>
              )
            ) : (
              <div className="h-full p-8 text-center text-neutral-400">
                <p>Invalid tab selection</p>
              </div>
            )}
          </div>

        </div>
      </div>
      
      {/* Computation Summary Notification - Bottom Right */}
      <ComputationNotification 
        summary={computationSummary}
        onDismiss={() => setComputationSummary(null)}
      />


      {/* Settings Modal */}
      <SimplifiedSettingsModal
        isOpen={settingsModalOpen}
        onClose={() => setSettingsModalOpen(false)}
        onSettingsChange={setExtendedSettings}
        gridSize={gridSize}
        totalPossibleResults={totalGridPoints}
        isComputing={isComputingGrid}
        centralizedLimits={centralizedLimits}
        onMasterLimitChange={handleMasterLimitChange}
      />

      {/* Authentication Modal */}
      <AuthModal 
        isOpen={showAuthModal}
        onClose={() => setShowAuthModal(false)}
        onAuthSuccess={() => {
          setShowAuthModal(false);
        }}
      />
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
