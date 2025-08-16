"use client";

import React, { useState, useEffect, useCallback } from 'react';
import Image from 'next/image';
import 'katex/dist/katex.min.css';
import { 
  BackendMeshPoint, 
  GridParameterArrays 
} from './circuit-simulator/types';
import { ModelSnapshot, ImpedancePoint, ResnormGroup, PerformanceLog, PipelinePhase } from './circuit-simulator/types';
import { CircuitParameters } from './circuit-simulator/types/parameters';
import { useWorkerManager, WorkerProgress } from './circuit-simulator/utils/workerManager';
import { useComputationState } from './circuit-simulator/hooks/useComputationState';

// Add imports for the new tab components at the top of the file
import { MathDetailsTab } from './circuit-simulator/MathDetailsTab';
import { DataTableTab } from './circuit-simulator/DataTableTab';
import { VisualizerTab } from './circuit-simulator/VisualizerTab';
// Removed OrchestratorTab - functionality integrated into VisualizerTab

import { PerformanceSettings, DEFAULT_PERFORMANCE_SETTINGS } from './circuit-simulator/controls/PerformanceControls';
import { ComputationNotification, ComputationSummary } from './circuit-simulator/notifications/ComputationNotification';
import { SavedProfiles } from './circuit-simulator/controls/SavedProfiles';
import { StaticRenderSettings, defaultStaticRenderSettings } from './circuit-simulator/controls/StaticRenderControls';
import { SavedProfile, SavedProfilesState } from './circuit-simulator/types/savedProfiles';
import { CenterParameterPanel } from './circuit-simulator/controls/CenterParameterPanel';

// Remove empty interface and replace with type
type CircuitSimulatorProps = Record<string, never>;

export const CircuitSimulator: React.FC<CircuitSimulatorProps> = () => {
  // Initialize worker manager for parallel computation
  const { computeGridParallel, cancelComputation } = useWorkerManager();
  
  // Add frequency control state - extended range for full Nyquist closure
  const [minFreq, setMinFreq] = useState<number>(0.1); // 0.1 Hz for full low-frequency closure
  const [maxFreq, setMaxFreq] = useState<number>(100000); // 100 kHz for high-frequency closure
  const [numPoints, setNumPoints] = useState<number>(100); // Default number of frequency points
  const [frequencyPoints, setFrequencyPoints] = useState<number[]>([]);
  
  // Use the new computation state hook
  const {
    gridResults, setGridResults,
    gridResultsWithIds, setGridResultsWithIds,
    gridSize, setGridSize,
    gridError, setGridError,
    isComputingGrid, setIsComputingGrid,
    computationProgress, setComputationProgress,
    computationSummary, setComputationSummary,
    skippedPoints, setSkippedPoints,
    totalGridPoints, setTotalGridPoints,
    actualComputedPoints, setActualComputedPoints,
    memoryLimitedPoints, setMemoryLimitedPoints,
    estimatedMemoryUsage, setEstimatedMemoryUsage,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars  
    userVisualizationPercentage,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    maxVisualizationPoints,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    isUserControlledLimits,
    calculateEffectiveVisualizationLimit,
    resnormGroups, setResnormGroups,
    hiddenGroups, setHiddenGroups,
    logMessages,
    statusMessage,
    updateStatusMessage,
    resetComputationState
  } = useComputationState();

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

  const [visualizationTab, setVisualizationTab] = useState<'visualizer' | 'math' | 'data' | 'activity'>('visualizer');
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_parameterChanged, setParameterChanged] = useState<boolean>(false);
  
  // Multi-select state for circuits
  const [selectedCircuits, setSelectedCircuits] = useState<string[]>([]);
  const [isMultiSelectMode, setIsMultiSelectMode] = useState<boolean>(false);
  // Track if reference model was manually hidden
  const [manuallyHidden, setManuallyHidden] = useState<boolean>(false);
  // Configuration name for saving profiles
  const [configurationName, setConfigurationName] = useState<string>('');
  
  // Static rendering state
  const [staticRenderSettings, setStaticRenderSettings] = useState<StaticRenderSettings>(defaultStaticRenderSettings);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_isStaticRendering, _setIsStaticRendering] = useState<boolean>(false);

  // Saved profiles state - start with empty state to avoid hydration mismatch
  const [savedProfilesState, setSavedProfilesState] = useState<SavedProfilesState>({
    profiles: [],
    selectedProfile: null
  });

  
  // Visualization settings - these are now passed to child components but setters not used
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [opacityLevel, setOpacityLevel] = useState<number>(0.7);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [logScalar, setLogScalar] = useState<number>(1.0);
  // Add opacity exponent state for spider plot and data table
  const [opacityExponent, setOpacityExponent] = useState<number>(5);
  
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
  
  // Function to create a new circuit with standard configuration
  const handleNewCircuit = useCallback(() => {
    // Clear any existing results
    resetComputationState();
    
    // Reset to standard starting configuration
    setParameters({
      Rsh: 100,
      Ra: 1000,
      Ca: 1.0e-6,
      Rb: 800,
      Cb: 0.8e-6,
      frequency_range: [0.1, 100000]
    });
    
    // Reset frequency settings
    setMinFreq(0.1);
    setMaxFreq(100000);
    setNumPoints(100);
    
    // Reset grid size
    setGridSize(5);
    
    // Reset configuration name
    setConfigurationName('');
    
    // Update status
    updateStatusMessage('New circuit created with standard configuration');
  }, [resetComputationState, updateStatusMessage, setGridSize]);
  
  // Multi-select functions
  const handleToggleMultiSelect = useCallback(() => {
    setIsMultiSelectMode(!isMultiSelectMode);
    setSelectedCircuits([]);
  }, [isMultiSelectMode]);
  
  const handleSelectCircuit = useCallback((profileId: string) => {
    if (isMultiSelectMode) {
      setSelectedCircuits(prev => 
        prev.includes(profileId) 
          ? prev.filter(id => id !== profileId)
          : [...prev, profileId]
      );
    } else {
      // Normal single selection behavior
      // Clear any pending computation
      setPendingComputeProfile(null);
      
      // Clear existing grid results when switching profiles
      resetComputationState();
      
      // Mark all profiles as not computed since we cleared the results
      setSavedProfilesState(prev => ({
        ...prev,
        selectedProfile: profileId,
        profiles: prev.profiles.map(p => ({ ...p, isComputed: false }))
      }));
      
      const profile = savedProfilesState.profiles.find(p => p.id === profileId);
      if (profile) {
        // Load profile settings into current state
        setGridSize(profile.gridSize);
        setMinFreq(profile.minFreq);
        setMaxFreq(profile.maxFreq);
        setNumPoints(profile.numPoints);
        setParameters(profile.groundTruthParams);
        
        // Update frequencies based on loaded settings
        updateFrequencies(profile.minFreq, profile.maxFreq, profile.numPoints);
        
        // Mark parameters as changed to enable recompute
        setParameterChanged(true);
        
        updateStatusMessage(`Loaded profile: ${profile.name} - Previous grid data cleared, ready to compute`);
      }
    }
  }, [isMultiSelectMode, resetComputationState, setSavedProfilesState, savedProfilesState.profiles, setGridSize, setMinFreq, setMaxFreq, setNumPoints, setParameters, setParameterChanged, updateStatusMessage]);
  
  const handleBulkDelete = useCallback(() => {
    if (selectedCircuits.length === 0) return;
    
    // Check if currently selected profile is being deleted
    const wasCurrentlySelectedDeleted = selectedCircuits.includes(savedProfilesState.selectedProfile || '');
    if (wasCurrentlySelectedDeleted) {
      resetComputationState();
    }
    
    // Remove selected profiles
    setSavedProfilesState(prev => ({
      ...prev,
      profiles: prev.profiles.filter(p => !selectedCircuits.includes(p.id)),
      selectedProfile: wasCurrentlySelectedDeleted ? null : prev.selectedProfile
    }));
    
    const deletedCount = selectedCircuits.length;
    setSelectedCircuits([]);
    setIsMultiSelectMode(false);
    
    updateStatusMessage(
      wasCurrentlySelectedDeleted 
        ? `${deletedCount} circuit${deletedCount > 1 ? 's' : ''} deleted and grid data cleared`
        : `${deletedCount} circuit${deletedCount > 1 ? 's' : ''} deleted`
    );
  }, [selectedCircuits, savedProfilesState.selectedProfile, resetComputationState, setSavedProfilesState, updateStatusMessage]);
  
  // Add state for grid parameter arrays
  const [gridParameterArrays, setGridParameterArrays] = useState<GridParameterArrays | null>(null);
  
  // Performance settings state
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [performanceSettings, setPerformanceSettings] = useState<PerformanceSettings>(DEFAULT_PERFORMANCE_SETTINGS);
  
  // Current memory usage for performance monitoring
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const getCurrentMemoryUsage = useCallback((): number => {
    if (typeof window !== 'undefined' && 'performance' in window && 'memory' in performance) {
      const memory = (performance as typeof performance & { memory: { usedJSHeapSize: number } }).memory;
      return memory.usedJSHeapSize / (1024 * 1024);
    }
    return 0;
  }, []);
  
  // Initialize parameters
  useEffect(() => {
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

    // Set default starting values at 50% of ranges
    setParameters({
      Rsh: rs50,
      Ra: ra50,
      Ca: ca50,
      Rb: rb50,
      Cb: cb50,
      frequency_range: [minFreq, maxFreq]
    });
    
    // Initial reference params will be set after the state updates
    updateStatusMessage('Initialized with default parameters at 50% of ranges');
  }, [updateStatusMessage, minFreq, maxFreq]);

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
  }, [referenceModelId, createReferenceModel, updateStatusMessage]);

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
  }, [toggleReferenceModel, hiddenGroups, referenceModelId, createReferenceModel, updateStatusMessage, setHiddenGroups]);
  
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
      color: '#3B82F6',
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
        
        const real = Rsh + Za_real + Zb_real;
        const imaginary = Za_imag + Zb_imag;
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
      const WORKER_MAX_GRID_SIZE = 20; // Hard limit from grid-worker.js
      const WORKER_MAX_TOTAL_POINTS = 10000000; // 10M points hard limit
      
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
      if (estimatedMemoryMB > 2000) { // 2GB memory limit
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
        computeGridParallel(
          profileGroundTruthParams,
          safeGridSize,
          profileMinFreq,
          profileMaxFreq,
          profileNumPoints,
          conservativeSettings,
          (progress) => {
            // Update status with progress
            if (progress.overallProgress) {
              updateStatusMessage(`Generating profile data: ${Math.round(progress.overallProgress)}%`);
            }
          },
          (error) => {
            console.error('Profile generation error:', error);
            throw new Error(`Worker error: ${error}`);
          }
        ),
        // 5 minute timeout to prevent infinite hangs
        new Promise<never>((_, reject) => 
          setTimeout(() => reject(new Error('Profile generation timeout (5 minutes)')), 5 * 60 * 1000)
        )
      ]);
      
      if (!gridResults || gridResults.length === 0) {
        throw new Error('No grid results generated');
      }
      
      // Convert BackendMeshPoint to ModelSnapshot format and sample if needed
      const modelSnapshots: ModelSnapshot[] = gridResults.map((result, index) => {
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
          color: '#3B82F6',
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
  }, [computeGridParallel, updateStatusMessage, generateSyntheticProfileData]);

  // Updated grid computation using Web Workers for parallel processing
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
      // Progress callback for worker updates
      const handleProgress = (progress: WorkerProgress) => {
        setComputationProgress(progress);
        
        if (progress.type === 'GENERATION_PROGRESS') {
          if (generationStartTime === 0) {
            generationStartTime = Date.now();
            completePhase('initialization');
            startPhase('grid-generation', 'Generating parameter space grid combinations');
          }
          if (progress.generated && progress.generated > 0) {
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

      // Run the parallel computation
      const results = await computeGridParallel(
        parameters,
        gridSize,
        minFreq,
        maxFreq,
        numPoints,
        performanceSettings,
        handleProgress,
        handleError
      );

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
      
      // Create filtered points with IDs
      const filteredPointsWithIds = filteredResults.map((point, idx) => ({
        ...point,
        id: idx + 1
      }));
      
      // Update state with filtered results
      setGridResults(filteredResults);
      setGridResultsWithIds(filteredPointsWithIds);
      
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
        setSavedProfilesState(prev => ({
          ...prev,
          profiles: prev.profiles.map(p => 
            p.id === savedProfilesState.selectedProfile 
              ? { ...p, isComputed: true, lastModified: Date.now() }
              : p
          )
        }));
      }

      setIsComputingGrid(false);
      setComputationProgress(null);
      
      // Auto-collapse sidebar and auto-save configuration when computation completes
      setLeftNavCollapsed(true);
      
      // Always auto-save the configuration after successful computation
      const autoSaveName = configurationName.trim() || `Auto-saved ${new Date().toLocaleString('en-US', { 
        month: 'short', 
        day: 'numeric', 
        hour: '2-digit', 
        minute: '2-digit',
        hour12: false 
      })}`;
      
      handleSaveProfile(autoSaveName, `Automatically saved after successful computation - ${sortedResults.length} models computed`);
      updateStatusMessage(`Configuration auto-saved as "${autoSaveName}" and sidebar collapsed`);
      
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
      setPendingComputeProfile(null);
      
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
  }, [gridSize, updateStatusMessage, setIsComputingGrid, resetComputationState, setParameterChanged, setManuallyHidden, setTotalGridPoints, minFreq, maxFreq, numPoints, setComputationProgress, computeGridParallel, mapBackendMeshToSnapshot, setGridResults, setGridResultsWithIds, setResnormGroups, setComputationSummary, visualizationSettings, applyVisualizationFiltering, calculateEffectiveVisualizationLimit, completePhase, createReferenceModel, currentPhases, generatePerformanceLog, parameters, isUserControlledLimits, performanceSettings, referenceModelId, savedProfilesState.selectedProfile, setActualComputedPoints, setEstimatedMemoryUsage, setGridError, setMemoryLimitedPoints, setSkippedPoints, startPhase, userVisualizationPercentage, setSavedProfilesState]); // eslint-disable-line react-hooks/exhaustive-deps

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

  // Handler for grid values generated by spider plot
  const handleGridValuesGenerated = useCallback((values: GridParameterArrays) => {
    if (!values) return;
    
    // Log the values for debugging
    console.log('Grid values received:', values);
    
    // Only update if values have actually changed
    setGridParameterArrays(prev => {
      if (!prev) return values;
      
      // Quick equality check for performance
      if (prev === values) return prev;
      
      // Check if any arrays have changed length
      const keys = Object.keys(values) as Array<keyof GridParameterArrays>;
      const hasLengthChanged = keys.some(key => 
        !prev[key] || prev[key].length !== values[key].length
      );
      
      if (hasLengthChanged) return values;
      
      // Deep compare arrays
      const hasChanged = keys.some(key => {
        const prevArr = prev[key];
        const newArr = values[key];
        return !prevArr.every((val, idx) => val === newArr[idx]);
      });
      
      return hasChanged ? values : prev;
    });
  }, []);

  // Note: Grid clearing functionality is now handled by resetComputationState from useComputationState hook

  // Wrapper for cancel computation that also clears pending profile
  const handleCancelComputation = useCallback(() => {
    cancelComputation();
    setPendingComputeProfile(null);
    updateStatusMessage('Computation cancelled');
  }, [cancelComputation, updateStatusMessage]);

  // Handler for saving a profile
  const handleSaveProfile = useCallback((name: string, description?: string) => {
    // Generate timestamp and ID only when actually saving (client-side)
    const now = Date.now();
    const randomId = Math.random().toString(36).substr(2, 9);
    
    const newProfile: SavedProfile = {
      id: `profile_${now}_${randomId}`,
      name,
      description,
      created: now,
      lastModified: now,
      
      // Grid computation settings
      gridSize,
      minFreq,
      maxFreq,
      numPoints,
      
      // Circuit parameters
      groundTruthParams: { ...parameters },
      
      // Computation status
      isComputed: false,
    };

    setSavedProfilesState(prev => ({
      ...prev,
      profiles: [...prev.profiles, newProfile],
      selectedProfile: newProfile.id
    }));

    updateStatusMessage(`Profile "${name}" saved with current settings`);
  }, [gridSize, minFreq, maxFreq, numPoints, parameters, updateStatusMessage]);



  // Note: Grid generation and spectrum calculation now handled by Web Workers

  // State for collapsible sidebar
  const [leftNavCollapsed, setLeftNavCollapsed] = useState(false);

  // Track if we've loaded from localStorage to avoid hydration issues
  const [hasLoadedFromStorage, setHasLoadedFromStorage] = useState(false);
  
  // Track profile loading for computation
  const [pendingComputeProfile, setPendingComputeProfile] = useState<string | null>(null);
  

  // Handler for copying profile parameters
  const handleCopyParams = useCallback((profileId: string) => {
    const profile = savedProfilesState.profiles.find(p => p.id === profileId);
    if (profile) {
      const paramsData = {
        profileName: profile.name,
        gridSettings: {
          gridSize: profile.gridSize,
          minFreq: profile.minFreq,
          maxFreq: profile.maxFreq,
          numPoints: profile.numPoints
        },
        circuitParameters: {
          Rsh: profile.groundTruthParams.Rsh,
          Ra: profile.groundTruthParams.Ra,
          Ca: profile.groundTruthParams.Ca,
          Rb: profile.groundTruthParams.Rb,
          Cb: profile.groundTruthParams.Cb,
          frequency_range: profile.groundTruthParams.frequency_range
        }
      };

      // Copy to clipboard as formatted JSON
      const jsonString = JSON.stringify(paramsData, null, 2);
      
      if (navigator.clipboard && window.isSecureContext) {
        // Use the Clipboard API when available (HTTPS/localhost)
        navigator.clipboard.writeText(jsonString).then(() => {
          updateStatusMessage(`Parameters copied to clipboard from "${profile.name}"`);
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
          updateStatusMessage(`Parameters copied to clipboard from "${profile.name}"`);
        } catch {
          updateStatusMessage(`Failed to copy parameters to clipboard`);
        }
      }
    }
  }, [savedProfilesState.profiles, updateStatusMessage]);

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

  // Load from localStorage after hydration
  useEffect(() => {
    if (typeof window !== 'undefined' && !hasLoadedFromStorage) {
      try {
        const saved = localStorage.getItem('nei-viz-saved-profiles');
        if (saved) {
          const parsedState = JSON.parse(saved);
          setSavedProfilesState(parsedState);
        }
      } catch (error) {
        console.warn('Failed to load saved profiles from localStorage:', error);
      }
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

  // Add a sample profile on first load (only after localStorage has been loaded)
  useEffect(() => {
    if (hasLoadedFromStorage && savedProfilesState.profiles.length === 0 && parameters.Rsh > 0) {
      const now = Date.now();
      const sampleProfile: SavedProfile = {
        id: 'sample_profile_default',
        name: 'Sample Configuration',
        description: 'Example profile showing a typical bioimpedance measurement setup',
        created: now,
        lastModified: now,
        gridSize: 5,
        minFreq: 1,
        maxFreq: 1000,
        numPoints: 20,
        groundTruthParams: {
          Rsh: 50,
          Ra: 1000,
          Ca: 1.0e-6,
          Rb: 800,
          Cb: 0.8e-6,
          frequency_range: [1, 1000]
        },
        isComputed: false,
      };

      setSavedProfilesState(prev => ({
        ...prev,
        profiles: [sampleProfile]
      }));
      
      updateStatusMessage('Welcome! A sample profile has been created to get you started.');
    }
  }, [hasLoadedFromStorage, parameters.Rsh, savedProfilesState.profiles.length, updateStatusMessage]);

  // Handle pending profile computation after parameters are loaded
  useEffect(() => {
    if (pendingComputeProfile && !isComputingGrid) {
      const profile = savedProfilesState.profiles.find(p => p.id === pendingComputeProfile);
      if (profile) {
        // Check if parameters match the profile (indicating they've been loaded)
        const paramsMatch = 
          gridSize === profile.gridSize &&
          minFreq === profile.minFreq &&
          maxFreq === profile.maxFreq &&
          numPoints === profile.numPoints &&
          Math.abs(parameters.Rsh - profile.groundTruthParams.Rsh) < 0.1 &&
          Math.abs(parameters.Ra - profile.groundTruthParams.Ra) < 0.1;
        
        if (paramsMatch) {
          updateStatusMessage(`Starting computation for profile "${profile.name}"...`);
          setPendingComputeProfile(null);
          handleComputeRegressionMesh();
        }
      } else {
        // Profile not found, clear pending
        setPendingComputeProfile(null);
      }
    }
  }, [pendingComputeProfile, isComputingGrid, savedProfilesState.profiles, gridSize, minFreq, maxFreq, numPoints, parameters.Rsh, parameters.Ra, handleComputeRegressionMesh, updateStatusMessage]);


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
                src="/spiderweb.png" 
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
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
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
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
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
              className="w-full bg-blue-600 hover:bg-blue-700 text-white px-4 py-2.5 rounded-md text-sm font-medium transition-colors duration-200 flex items-center justify-center gap-2"
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
                visualizationTab === 'data' 
                  ? 'bg-neutral-800 text-white' 
                  : 'text-neutral-400 hover:bg-neutral-800 hover:text-white'
              }`}
              onClick={() => setVisualizationTab('data')}
            >
              <div className="flex items-center gap-3">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                <span>Data Table</span>
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
          
          {/* Saved Profiles Section - Scrollable */}
          {hasLoadedFromStorage && (
            <SavedProfiles
              profiles={savedProfilesState.profiles}
              selectedProfile={savedProfilesState.selectedProfile}
              onCopyParams={handleCopyParams}
              selectedCircuits={selectedCircuits}
              isMultiSelectMode={isMultiSelectMode}
              onToggleMultiSelect={handleToggleMultiSelect}
              onBulkDelete={handleBulkDelete}
            onSelectProfile={handleSelectCircuit}
            onSelectProfileOriginal={(profileId) => {
              // Clear any pending computation
              setPendingComputeProfile(null);
              
              // Clear existing grid results when switching profiles
              resetComputationState();
              
              // Mark all profiles as not computed since we cleared the results
              setSavedProfilesState(prev => ({
                ...prev,
                selectedProfile: profileId,
                profiles: prev.profiles.map(p => ({ ...p, isComputed: false }))
              }));
              
              const profile = savedProfilesState.profiles.find(p => p.id === profileId);
              if (profile) {
                // Load profile settings into current state
                setGridSize(profile.gridSize);
                setMinFreq(profile.minFreq);
                setMaxFreq(profile.maxFreq);
                setNumPoints(profile.numPoints);
                setParameters(profile.groundTruthParams);
                
                // Update frequencies based on loaded settings
                updateFrequencies(profile.minFreq, profile.maxFreq, profile.numPoints);
                
                // Mark parameters as changed to enable recompute
                setParameterChanged(true);
                
                updateStatusMessage(`Loaded profile: ${profile.name} - Previous grid data cleared, ready to compute`);
              }
            }}
            onDeleteProfile={(profileId) => {
              // Clear grid results if we're deleting the currently selected profile
              const wasSelected = savedProfilesState.selectedProfile === profileId;
              if (wasSelected) {
                resetComputationState();
              }
              
              setSavedProfilesState(prev => ({
                ...prev,
                profiles: prev.profiles.filter(p => p.id !== profileId),
                selectedProfile: prev.selectedProfile === profileId ? null : prev.selectedProfile
              }));
              
              updateStatusMessage(wasSelected ? 'Profile deleted and grid data cleared' : 'Profile deleted');
            }}
            onEditProfile={(profileId, name, description) => {
              const now = Date.now();
              setSavedProfilesState(prev => ({
                ...prev,
                profiles: prev.profiles.map(p => 
                  p.id === profileId 
                    ? { ...p, name, description, lastModified: now }
                    : p
                )
              }));
              updateStatusMessage(`Profile "${name}" updated`);
            }}
            onEditParameters={(profileId) => {
              const profile = savedProfilesState.profiles.find(p => p.id === profileId);
              if (profile) {
                // Expand the sidebar
                setLeftNavCollapsed(false);
                
                // Clear existing grid results
                resetComputationState();
                
                // Load profile settings into center panel for editing
                setGridSize(profile.gridSize);
                setMinFreq(profile.minFreq);
                setMaxFreq(profile.maxFreq);
                setNumPoints(profile.numPoints);
                setParameters(profile.groundTruthParams);
                
                // Update frequencies
                updateFrequencies(profile.minFreq, profile.maxFreq, profile.numPoints);
                
                // Select this profile
                setSavedProfilesState(prev => ({ ...prev, selectedProfile: profileId }));
                
                // Mark parameters as changed to enable editing mode
                setParameterChanged(true);
                
                updateStatusMessage(`Editing parameters for "${profile.name}" - make changes in center panel`);
              }
            }}
            onComputeProfile={(profileId) => {
              const profile = savedProfilesState.profiles.find(p => p.id === profileId);
              if (profile) {
                // Clear existing grid results first
                resetComputationState();
                
                // Load profile settings
                setGridSize(profile.gridSize);
                setMinFreq(profile.minFreq);
                setMaxFreq(profile.maxFreq);
                setNumPoints(profile.numPoints);
                setParameters(profile.groundTruthParams);
                
                // Update frequencies
                updateFrequencies(profile.minFreq, profile.maxFreq, profile.numPoints);
                
                // Select this profile
                setSavedProfilesState(prev => ({ ...prev, selectedProfile: profileId }));
                
                // Set pending computation - useEffect will handle when parameters are loaded
                setPendingComputeProfile(profileId);
                
                updateStatusMessage(`Loading profile "${profile.name}" parameters...`);
              }
            }}
            isCollapsed={leftNavCollapsed}
                />
            )}
        </div>
        
        {/* Collapsed state icons overlay - only show when collapsed */}
        <div className={`absolute inset-0 flex flex-col transition-all duration-300 ${leftNavCollapsed ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'}`}>
          {/* Skip header area - updated for wider collapsed bar */}
          <div className="h-20 flex-shrink-0"></div>
          
          {/* New Circuit Button - Collapsed */}
          <div className="flex-shrink-0 px-2 pb-2">
            <button
              onClick={handleNewCircuit}
              className="w-10 h-10 flex items-center justify-center rounded-md bg-blue-600 hover:bg-blue-700 text-white transition-colors duration-200"
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
              title="Data Table"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
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

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Clean Header with integrated status */}
        <header className="px-6 py-3 flex items-center justify-between z-30 flex-shrink-0 bg-neutral-950">
          {/* Left side: compact status indicator */}
          <div className="flex items-center gap-3">
            {statusMessage && (
              <div className="bg-neutral-800 rounded-md px-3 py-1.5 text-xs max-w-xs truncate flex items-center">
                <svg className="w-3 h-3 mr-1.5 flex-shrink-0 text-neutral-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="truncate text-neutral-300">{statusMessage}</span>
              </div>
            )}
          </div>

          {/* Right side: Tab-specific info and toolbox button */}
          <div className="flex items-center gap-3">
            {/* Tab-specific status info */}
            {visualizationTab === 'visualizer' && (
              <div className="bg-neutral-800 rounded-md px-3 py-1.5 text-xs">
                <span className="text-neutral-400">Showing: </span>
                <span className="text-white font-medium">{gridResults.length.toLocaleString()}</span>
                
                {/* Show actual computed vs theoretical points */}
                {actualComputedPoints > 0 && actualComputedPoints !== totalGridPoints && (
                  <>
                    <span className="text-neutral-400 ml-1.5">of</span>
                    <span className="text-green-300 font-medium ml-1">{actualComputedPoints.toLocaleString()}</span>
                    <span className="text-neutral-400 ml-1">computed</span>
                  </>
                )}
                
                <span className="text-neutral-400 ml-1.5">/</span>
                <span className="text-amber-300 font-semibold ml-1">{totalGridPoints.toLocaleString()}</span>
                <span className="text-neutral-400 ml-1">total</span>
                
                {/* Show skipped points from symmetric optimization */}
                {skippedPoints > 0 && (
                  <>
                    <span className="text-neutral-400 ml-1.5">•</span>
                    <span className="text-orange-300 ml-1">{skippedPoints.toLocaleString()} skipped</span>
                  </>
                )}
                
                {/* Show memory-limited points */}
                {memoryLimitedPoints > 0 && memoryLimitedPoints < gridResults.length && (
                  <>
                    <span className="text-neutral-400 ml-1.5">•</span>
                    <span className="text-red-300 ml-1">{(gridResults.length - memoryLimitedPoints).toLocaleString()} hidden</span>
                  </>
                )}
                
                <span className="text-neutral-400 ml-1.5">|</span>
                <span className="text-neutral-400 ml-1.5">Freq: </span>
                <span className="text-white font-medium">{minFreq.toFixed(2)} - {maxFreq.toFixed(0)} Hz</span>
                
                {/* Show memory usage if significant */}
                {estimatedMemoryUsage > 100 && (
                  <>
                    <span className="text-neutral-400 ml-1.5">|</span>
                    <span className="text-purple-300 ml-1">{estimatedMemoryUsage.toFixed(0)}MB</span>
                  </>
                )}
              </div>
            )}
            
            {/* Orchestrator status removed - functionality integrated into Playground */}
            
            {visualizationTab === 'data' && (
              <div className="bg-neutral-800 rounded-md px-3 py-1.5 text-xs">
                <span className="text-neutral-400">Data: </span>
                <span className="text-white font-medium">{gridResults.length.toLocaleString()} models</span>
                <span className="text-neutral-400 ml-1.5">•</span>
                <span className="text-neutral-400 ml-1.5">Sortable view</span>
              </div>
            )}
            
            {visualizationTab === 'math' && (
              <div className="bg-neutral-800 rounded-md px-3 py-1.5 text-xs">
                <span className="text-neutral-400">Circuit: </span>
                <span className="text-white font-medium">Randles Model</span>
                <span className="text-neutral-400 ml-1.5">•</span>
                <span className="text-neutral-400 ml-1.5">Mathematical reference</span>
              </div>
            )}
            
          </div>
        </header>
        
        {/* Main visualization area */}
        <div className="flex-1 flex overflow-hidden">
          <div className="flex-1 p-4 bg-neutral-950 overflow-hidden">
            {isComputingGrid ? (
              <div className="flex flex-col items-center justify-center p-6 h-40 bg-neutral-900/50 border border-neutral-700 rounded-lg shadow-md">
                <div className="flex items-center mb-4">
                  <div className="animate-spin rounded-full h-6 w-6 border-t-2 border-b-2 border-neutral-400 mr-3"></div>
                  <div>
                    <p className="text-sm text-neutral-300 font-medium">
                      {computationProgress?.type === 'GENERATION_PROGRESS' 
                        ? 'Generating grid points...' 
                        : 'Computing in parallel...'}
                    </p>
                    <p className="text-xs text-neutral-400 mt-1">
                      {computationProgress ? 
                        `${computationProgress.overallProgress.toFixed(1)}% complete` : 
                        'Using multiple CPU cores for faster computation'}
                    </p>
                  </div>
                </div>
                
                {/* Progress bar */}
                {computationProgress && (
                  <div className="w-full max-w-md">
                    <div className="w-full bg-neutral-700 rounded-full h-2">
                      <div 
                        className="bg-neutral-400 h-2 rounded-full transition-all duration-300" 
                        style={{ width: `${computationProgress.overallProgress}%` }}
                      ></div>
                    </div>
                  </div>
                )}
                
                {/* Cancel button */}
                <button
                  onClick={handleCancelComputation}
                  className="mt-4 px-4 py-2 text-xs bg-red-600 hover:bg-red-700 text-white rounded-md transition-colors"
                >
                  Cancel Computation
                </button>
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
            ) : visualizationTab === 'data' ? (
              gridResults.length > 0 ? (
                <div className="h-full overflow-y-auto">
                  <DataTableTab 
                    gridResults={gridResults}
                    gridResultsWithIds={gridResultsWithIds}
                    resnormGroups={resnormGroups}
                    hiddenGroups={hiddenGroups}
                    maxGridPoints={Math.pow(gridSize, 5)}
                    gridSize={gridSize}
                    parameters={parameters}
                    gridParameterArrays={gridParameterArrays}
                    opacityExponent={opacityExponent}
                    groundTruthParams={parameters}
                  />
                </div>
              ) : (
                <div className="flex items-center justify-center h-32 bg-neutral-900/50 border border-neutral-700 rounded-lg shadow-md p-4">
                  <div className="text-center">
                    <svg className="w-8 h-8 mx-auto mb-2 text-neutral-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <p className="text-sm text-neutral-400">
                      No data to display. Compute a grid first.
                    </p>
                  </div>
                </div>
              )
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
                      <div className="text-blue-400 font-mono">{performanceLog.efficiency.throughput.toFixed(0)} pts/s</div>
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
              // Show visualization if we have results, otherwise show configuration panel
              (resnormGroups && resnormGroups.length > 0) ? (
                <div className="h-full">
                  <VisualizerTab 
                    resnormGroups={resnormGroups}
                    hiddenGroups={hiddenGroups}
                    opacityLevel={opacityLevel}
                    referenceModelId={referenceModelId}
                    gridSize={gridSize}
                    onGridValuesGenerated={handleGridValuesGenerated}
                    opacityExponent={opacityExponent}
                    onOpacityExponentChange={setOpacityExponent}
                    userReferenceParams={parameters}
                    staticRenderSettings={staticRenderSettings}
                    onStaticRenderSettingsChange={setStaticRenderSettings}
                    groundTruthParams={parameters}
                    performanceSettings={performanceSettings}
                    numPoints={numPoints}
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
                      onCompute={handleComputeRegressionMesh}
                      onSaveProfile={handleSaveProfile}
                      isComputing={isComputingGrid}
                      configurationName={configurationName}
                      onConfigurationNameChange={setConfigurationName}
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
