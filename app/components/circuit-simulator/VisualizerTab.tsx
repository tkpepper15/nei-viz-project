import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
// SpiderPlot 2D import removed - using only 3D visualization
import { SpiderPlot3D } from './visualizations/SpiderPlot3D';
import { TSNEPlot3D } from './visualizations/TSNEPlot3D';
import { PentagonGroundTruth } from './visualizations/PentagonGroundTruth';
import { PentagonBoxWhisker } from './visualizations/PentagonBoxWhisker';
import { ModelSnapshot, ResnormGroup } from './types';
import { GridParameterArrays } from './types';
import { CircuitParameters } from './types/parameters';
import { StaticRenderSettings } from './controls/StaticRenderControls';
import { ResnormConfig, calculateResnormWithConfig, calculate_impedance_spectrum } from './utils/resnorm';
import { calculateTER, calculateTEC, getTERRange, getTECRange, formatTER, formatTEC } from './utils/terTecCalculations';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { calculateTimeConstants } from './math/utils';
import { CollapsibleBottomPanel } from './panels/CollapsibleBottomPanel';

interface VisualizationSettings {
  groupPortion: number;
  selectedOpacityGroups: number[];
  visualizationType: 'spider3d' | 'nyquist' | 'tsne' | 'pentagon-gt' | 'pentagon-quartile';
  resnormMatchingPortion?: number; // For fine-tuning resnorm grouping
}

interface VisualizerTabProps {
  resnormGroups: ResnormGroup[];
  hiddenGroups: number[];
  opacityLevel: number;
  referenceModelId: string | null;
  gridSize: number;
  onGridValuesGenerated: (values: GridParameterArrays) => void;
  opacityExponent: number;
  onOpacityExponentChange: (value: number) => void;
  // Circuit parameters from parent
  userReferenceParams?: CircuitParameters | null;
  // View control props - removed unused controls
  showLabels?: boolean;
  // Visualization settings callback
  onVisualizationSettingsChange?: (settings: VisualizationSettings) => void;
  // Static render settings for visualization consistency
  staticRenderSettings: StaticRenderSettings;
  onStaticRenderSettingsChange: (settings: StaticRenderSettings) => void;
  
  // Circuit parameter management
  groundTruthParams: CircuitParameters;
  
  
  // Frequency configuration for Nyquist plot
  numPoints: number;
  
  // Resnorm configuration for method switching (now fixed to SSR)
  resnormConfig: ResnormConfig;
  
  // Grid results for single circuit selection
  gridResults?: ModelSnapshot[];
  
  // Group portion for model filtering - sync with parent CircuitSimulator
  groupPortion: number;
  onGroupPortionChange: (value: number) => void;
  
  // Tagged models support
  taggedModels?: Map<string, { tagName: string; profileId: string; resnormValue: number; taggedAt: number; notes?: string }>;
  onModelTag?: (model: ModelSnapshot, tagName: string, profileId?: string) => void;

  // TER/TEC filtering from bottom panel
  terTecFilteredModelIds?: string[];
  onTERTECFilterChange?: (filteredIds: string[]) => void;
}

export const VisualizerTab: React.FC<VisualizerTabProps> = ({
  resnormGroups,
  hiddenGroups: _hiddenGroups, // eslint-disable-line @typescript-eslint/no-unused-vars
  opacityLevel: _opacityLevel, // eslint-disable-line @typescript-eslint/no-unused-vars
  referenceModelId: _referenceModelId, // eslint-disable-line @typescript-eslint/no-unused-vars
  gridSize,
  onGridValuesGenerated: _onGridValuesGenerated, // eslint-disable-line @typescript-eslint/no-unused-vars
  opacityExponent,
  onOpacityExponentChange: _onOpacityExponentChange, // eslint-disable-line @typescript-eslint/no-unused-vars
  // Circuit parameters from parent
  userReferenceParams,
  // View control props
  showLabels: _showLabels, // eslint-disable-line @typescript-eslint/no-unused-vars
  // Visualization settings callback
  onVisualizationSettingsChange,
  // Static render settings for visualization consistency
  staticRenderSettings,
  onStaticRenderSettingsChange,
  // Circuit parameter management
  groundTruthParams,
  
  // Frequency configuration for Nyquist plot
  numPoints: _numPoints, // eslint-disable-line @typescript-eslint/no-unused-vars
  
  // Resnorm configuration (now fixed to SSR)
  resnormConfig,
  
  // Grid results for single circuit selection
  gridResults,
  
  // Group portion for model filtering - sync with parent CircuitSimulator
  groupPortion,
  onGroupPortionChange,
  
  // Tagged models support
  taggedModels,
  onModelTag,

  // TER/TEC filtering from bottom panel
  terTecFilteredModelIds = [],
  onTERTECFilterChange
}) => {

  // Removed old control states - now using compact sidebar design
  
  // Use static render settings instead of local state
  const showGroundTruth = staticRenderSettings.showGroundTruth;
  
  // Note: resnorm matching portion no longer used with new logarithmic approach
  // const [resnormMatchingPortion] = useState<number>(25); // Removed - replaced with logarithmic selection
  // Use ground truth parameters from toolbox (userReferenceParams) or direct groundTruthParams
  const effectiveGroundTruthParams = userReferenceParams || groundTruthParams;
  // Use visualization settings from static render settings - defaulting to 3D only
  const visualizationType = staticRenderSettings.visualizationType === 'nyquist' ? 'nyquist' :
                           staticRenderSettings.visualizationType === 'tsne' ? 'tsne' :
                           staticRenderSettings.visualizationType === 'pentagon-gt' ? 'pentagon-gt' :
                           staticRenderSettings.visualizationType === 'pentagon-quartile' ? 'pentagon-quartile' : 'spider3d';
  const view3D = visualizationType === 'spider3d';
  const setVisualizationType = (type: 'spider3d' | 'nyquist' | 'tsne' | 'pentagon-gt' | 'pentagon-quartile') => {
    onStaticRenderSettingsChange({
      ...staticRenderSettings,
      visualizationType: type
    });
  };
  
  // Nyquist plot log scale options
  // Removed nyquist log scale state - no longer needed without navigation controls
  
  // Removed old CollapsibleSection and formatTickValue - using compact sidebar design
  
  // Add view control state - use props if provided, otherwise internal state
  // const [internalZoomLevel, setInternalZoomLevel] = useState(1.0); // Removed unused
  
  // Convert groupPortion to slider value (1-100) for logarithmic scaling
  const groupPortionToSliderValue = useCallback((portion: number) => {
    const percentage = portion * 100; // Convert to 0-100%
    const minLog = Math.log10(0.01);
    const maxLog = Math.log10(100);
    const logValue = Math.log10(Math.max(0.01, percentage));
    const sliderValue = ((logValue - minLog) / (maxLog - minLog)) * 99 + 1;
    return Math.max(1, Math.min(100, Math.round(sliderValue)));
  }, []);

  // State for slider position (1-100)
  const [modelSelectionSlider, setModelSelectionSlider] = useState<number>(() => 
    groupPortionToSliderValue(groupPortion)
  );
  
  // Convert slider value to actual percentage for display and filtering
  const logModelPercent = useMemo(() => {
    const minLog = Math.log10(0.01);
    const maxLog = Math.log10(100);
    const logValue = minLog + (modelSelectionSlider - 1) / 99 * (maxLog - minLog);
    return Math.pow(10, logValue);
  }, [modelSelectionSlider]);

  const [selectedNyquistConfig, setSelectedNyquistConfig] = useState<number>(0);
  // State for active snapshot for ResnormDisplay
  const [activeSnapshot, setActiveSnapshot] = useState<ModelSnapshot | null>(null);
  
  // Sync when parent groupPortion changes
  useEffect(() => {
    const expectedSliderValue = groupPortionToSliderValue(groupPortion);
    if (Math.abs(expectedSliderValue - modelSelectionSlider) > 1) {
      setModelSelectionSlider(expectedSliderValue);
    }
  }, [groupPortion, groupPortionToSliderValue, modelSelectionSlider]);
  
  // State for resnorm range filtering
  const [selectedResnormRange, setSelectedResnormRange] = useState<{min: number; max: number} | null>(null);
  
  // State for current resnorm navigation and model highlighting
  const [currentResnorm, setCurrentResnorm] = useState<number | null>(null);
  const [highlightedModelId, setHighlightedModelId] = useState<string | null>(null);

  // State for granular model navigation
  const [navigationStepSize, setNavigationStepSize] = useState<number>(1);

  // State for comparison selection
  const [selectedComparisonModel, setSelectedComparisonModel] = useState<ModelSnapshot | null>(null);

  // State for TER/TEC filtering
  const [terTecFilterEnabled, setTerTecFilterEnabled] = useState<boolean>(false);
  const [terTecFilterType, setTerTecFilterType] = useState<'TER' | 'TEC'>('TER');
  const [terTecTargetValue, setTerTecTargetValue] = useState<number>(0);
  const [terTecTolerance, setTerTecTolerance] = useState<number>(5); // Percentage

  // State for tagged models - convert parent prop to local Map format
  const localTaggedModels = useMemo(() => {
    if (!taggedModels) return new Map<string, string>();
    const localMap = new Map<string, string>();
    taggedModels.forEach((tagData, modelId) => {
      localMap.set(modelId, tagData.tagName);
    });
    return localMap;
  }, [taggedModels]);
  
  // Loading state for smooth initialization
  const [isInitializing, setIsInitializing] = useState(true);

  // Panel state management with localStorage persistence
  const [leftPanelCollapsed, setLeftPanelCollapsed] = useState(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('visualizerLeftPanelCollapsed');
      return saved === 'true';
    }
    return false;
  });
  const [bottomPanelCollapsed, setBottomPanelCollapsed] = useState(false);
  const [bottomPanelHeight, setBottomPanelHeight] = useState(400);

  // Persist left panel collapse state to localStorage
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('visualizerLeftPanelCollapsed', String(leftPanelCollapsed));
    }
  }, [leftPanelCollapsed]);

  

  // Handle resnorm range change from ResnormDisplay
  const handleResnormRangeChange = useCallback((min: number, max: number) => {
    setSelectedResnormRange({ min, max });
  }, []);

  // Handle current resnorm change from 3D view hover
  const handleCurrentResnormChange = useCallback((resnorm: number | null) => {
    setCurrentResnorm(resnorm);
  }, []);

  // Handle resnorm selection (shift-click in 3D view or double-click in histogram)
  // This will be defined after modelsWithUpdatedResnorm to avoid dependency issues
  const handleResnormSelectRef = useRef<((resnorm: number) => void) | null>(null);

  // Handle tagged model selection for highlighting
  const handleTaggedModelSelect = useCallback((modelId: string) => {
    setHighlightedModelId(prevId => prevId === modelId ? null : modelId);
  }, []);

  // Handle model tagging - delegate to parent component
  const handleModelTag = useCallback((model: ModelSnapshot, tag: string) => {
    if (onModelTag) {
      if (tag.trim() === '') {
        // TODO: Handle tag removal - for now, just skip empty tags
        return;
      } else {
        onModelTag(model, tag);
      }
    }
  }, [onModelTag]);

  // Handle model selection - update comparison model and highlighted ID
  const handleModelSelect = useCallback((model: ModelSnapshot | null) => {
    if (model) {
      setSelectedComparisonModel(model);
      setHighlightedModelId(model.id);
      setCurrentResnorm(model.resnorm || null);
    } else {
      // Deselect - clear comparison model but keep ground truth
      setSelectedComparisonModel(null);
      setHighlightedModelId(null);
      setCurrentResnorm(null);
    }
  }, []);

  // Memoized tag colors to avoid recalculation
  const tagColors = useMemo(() => [
    '#FF6B9D', '#4ECDC4', '#45B7D1', '#96CEB4', 
    '#FFEAA7', '#DDA0DD', '#98D8E8', '#F8C471', 
    '#82E0AA', '#85C1E9'
  ], []);

  // Recalculate resnorm values when config changes for live updating (moved up to fix declaration order)
  const modelsWithUpdatedResnorm = useMemo(() => {
    // Use gridResults if available (new portion-based system), otherwise fall back to resnormGroups
    if (gridResults && gridResults.length > 0) {
      console.log('Using gridResults for visualization:', gridResults.length, 'models');
      
      // Get reference spectrum from user reference params (for future enhancement)
      // const referenceSpectrum = userReferenceParams ? 
      //   calculate_impedance_spectrum(userReferenceParams) : null;
      
      // gridResults are already ModelSnapshot[], so just return them
      return gridResults;
    }
    
    // Fallback to old resnormGroups system
    if (!resnormGroups || resnormGroups.length === 0) return [];
    
    // Get reference spectrum from user reference params
    const referenceSpectrum = userReferenceParams ? 
      calculate_impedance_spectrum(userReferenceParams) : null;
    
    if (!referenceSpectrum) return resnormGroups.flatMap(group => group.items);
    
    // Recalculate resnorm for all models with current config
    return resnormGroups.flatMap(group => 
      group.items.map(model => ({
        ...model,
        resnorm: calculateResnormWithConfig(
          model.data,
          referenceSpectrum,
          undefined,
          undefined,
          resnormConfig
        )
      }))
    );
  }, [gridResults, resnormGroups, userReferenceParams, resnormConfig]);

  // Handle initialization timing to prevent laggy appearance
  useEffect(() => {
    if (modelsWithUpdatedResnorm.length > 0) {
      // Small delay to allow render cycle to complete before showing complex visualizations
      const timer = setTimeout(() => {
        setIsInitializing(false);
      }, 100);
      
      return () => clearTimeout(timer);
    }
  }, [modelsWithUpdatedResnorm.length]);

  // Get all models sorted by resnorm for logarithmic selection
  const sortedModels = useMemo(() => {
    if (!modelsWithUpdatedResnorm.length) return [];
    return [...modelsWithUpdatedResnorm].sort((a, b) => (a.resnorm || Infinity) - (b.resnorm || Infinity));
  }, [modelsWithUpdatedResnorm]);

  // Define handleResnormSelect after modelsWithUpdatedResnorm is available
  const handleResnormSelect = useCallback((resnorm: number) => {
    setCurrentResnorm(resnorm);
    
    // Calculate a more intelligent range size based on the overall resnorm distribution
    const allResnorms = modelsWithUpdatedResnorm.map(m => m.resnorm || 0).filter(r => r > 0);
    if (allResnorms.length === 0) return;
    
    allResnorms.sort((a, b) => a - b);
    const totalRange = allResnorms[allResnorms.length - 1] - allResnorms[0];
    const dynamicRangeSize = Math.max(totalRange * 0.05, 0.001); // 5% of total range or minimum 0.001
    
    // Create a focused range centered on the selected resnorm value
    const newMin = Math.max(allResnorms[0], resnorm - dynamicRangeSize / 2);
    const newMax = Math.min(allResnorms[allResnorms.length - 1], resnorm + dynamicRangeSize / 2);
    
    setSelectedResnormRange({ 
      min: newMin, 
      max: newMax 
    });
  }, [modelsWithUpdatedResnorm]);

  // Update the ref for use in components that need it earlier
  handleResnormSelectRef.current = handleResnormSelect;

  // Set active snapshot to the best model for ResnormDisplay
  useEffect(() => {
    if (sortedModels.length > 0 && !activeSnapshot) {
      // Set the best model (lowest resnorm) as the active snapshot
      const bestModel = sortedModels[0];
      setActiveSnapshot(bestModel);
    }
  }, [sortedModels, activeSnapshot]);
  
  // Rendering mode state - unused after removing 2D plot support  
  // const [renderingMode] = useState<'auto' | 'interactive' | 'tile'>('auto');
  // const [forceMode] = useState<boolean>(false);
  
  // Use external props if provided, otherwise use internal state
  // const currentZoomLevel = zoomLevel !== undefined ? zoomLevel : internalZoomLevel; // Unused after UI cleanup
  // const currentShowLabels = showLabels !== undefined ? showLabels : staticRenderSettings.includeLabels; // Unused after removing 2D plot

  // Notify parent of visualization settings changes
  useEffect(() => {
    if (onVisualizationSettingsChange) {
      const settings: VisualizationSettings = {
        groupPortion: logModelPercent / 100,
        selectedOpacityGroups: [], // No longer using group-based selection
        visualizationType
      };
      onVisualizationSettingsChange(settings);
    }
  }, [logModelPercent, visualizationType, onVisualizationSettingsChange]);

  // Ground truth parameters now come from userReferenceParams (toolbox)

  // Zoom level removed after UI cleanup

  // Note: sortedModels will be defined after modelsWithUpdatedResnorm
  
  // Generate standard logarithmic frequencies for consistent Nyquist plots
  const standardFrequencies = useMemo(() => {
    const logMin = Math.log10(0.1); // 0.1 Hz
    const logMax = Math.log10(100000); // 100 kHz
    const pointsToUse = 200; // Use same point count as original implementation
    const logStep = (logMax - logMin) / (pointsToUse - 1);
    
    const freqs: number[] = [];
    for (let i = 0; i < pointsToUse; i++) {
      const logValue = logMin + i * logStep;
      freqs.push(Math.pow(10, logValue));
    }
    return freqs;
  }, []);
  
  // Calculate impedance for given parameters using same method as original
  const calculateNyquistImpedance = useCallback((params: CircuitParameters): { real: number; imag: number; frequency: number; magnitude: number }[] => {
    const impedancePoints: { real: number; imag: number; frequency: number; magnitude: number }[] = [];
    
    // Calculate time constants once outside the loop for efficiency
    const { tauA, tauB } = calculateTimeConstants(params);
    
    for (const freq of standardFrequencies) {
      const omega = 2 * Math.PI * freq;
      
      // Calculate individual membrane impedances: Z = R/(1 + jωRC)
      // Za (apical membrane)
      const denominatorA = 1 + Math.pow(omega * tauA, 2);
      const realA = params.Ra / denominatorA;
      const imagA = -params.Ra * omega * tauA / denominatorA;
      
      // Zb (basal membrane)  
      const denominatorB = 1 + Math.pow(omega * tauB, 2);
      const realB = params.Rb / denominatorB;
      const imagB = -params.Rb * omega * tauB / denominatorB;
      
      // Za + Zb (series combination of membranes)
      const realSeriesMembranes = realA + realB;
      const imagSeriesMembranes = imagA + imagB;
      
      // Total impedance: Z_total = (Rsh * (Za + Zb)) / (Rsh + Za + Zb) (parallel with shunt)
      const numeratorReal = params.Rsh * realSeriesMembranes;
      const numeratorImag = params.Rsh * imagSeriesMembranes;
      
      const denominatorReal = params.Rsh + realSeriesMembranes;
      const denominatorImag = imagSeriesMembranes;
      const denominatorMagSquared = denominatorReal * denominatorReal + denominatorImag * denominatorImag;
      
      const realTotal = (numeratorReal * denominatorReal + numeratorImag * denominatorImag) / denominatorMagSquared;
      const imagTotal = (numeratorImag * denominatorReal - numeratorReal * denominatorImag) / denominatorMagSquared;
      
      const magnitude = Math.sqrt(realTotal * realTotal + imagTotal * imagTotal);
      
      impedancePoints.push({
        real: realTotal,
        imag: -imagTotal, // Negative for proper Nyquist display convention
        frequency: freq,
        magnitude
      });
    }
    
    return impedancePoints;
  }, [standardFrequencies]);
  
  // Get filtered configurations for Nyquist single circuit selection
  const topConfigurations = useMemo(() => {
    if (!gridResults || !gridResults.length) return [];
    
    // Sort grid results by resnorm and take top configurations
    const sorted = [...gridResults]
      .filter(point => point.resnorm !== undefined && point.resnorm !== null)
      .sort((a, b) => (a.resnorm || 0) - (b.resnorm || 0))
      .slice(0, Math.max(50, Math.ceil(gridResults.length * logModelPercent / 100))); // Show top models based on percentage
    
    return sorted.map(point => ({
      point,
      resnorm: point.resnorm,
      nyquistData: calculateNyquistImpedance(point.parameters)
    }));
  }, [gridResults, logModelPercent, calculateNyquistImpedance]);
  
  // Get current configuration for Nyquist plot
  const currentNyquistConfig = useMemo(() => {
    return topConfigurations[selectedNyquistConfig] || null;
  }, [topConfigurations, selectedNyquistConfig]);
  
  // Auto-reset nyquist selection when data changes
  useEffect(() => {
    if (topConfigurations.length > 0 && selectedNyquistConfig >= topConfigurations.length) {
      setSelectedNyquistConfig(0);
    }
  }, [topConfigurations, selectedNyquistConfig]);

  // Automatic rendering mode selection - unused after removing 2D plot
  // const actualRenderingMode = useMemo(() => { ... }, [renderingMode, forceMode, resnormGroups, hiddenGroups]);

  // Performance metrics for display - removed unused after UI cleanup
  // Performance metrics removed after UI cleanup

  // modelsWithUpdatedResnorm and sortedModels moved up above useEffect to fix declaration order

  // Note: allAvailableModels removed as it was unused

  // Navigation window for efficient rendering (default 1000 models)
  const [navigationWindowSize] = useState<number>(1000);
  const [navigationOffset, setNavigationOffset] = useState<number>(0);

  // Get all available models based on logarithmic percentage selection
  const allFilteredModels: ModelSnapshot[] = useMemo(() => {
    if (!sortedModels.length) return [];

    // Calculate how many models to include in navigation dataset based on logarithmic percentage
    const calculatedModelCount = Math.max(1, Math.ceil(sortedModels.length * logModelPercent / 100));
    return sortedModels.slice(0, calculatedModelCount);
  }, [sortedModels, logModelPercent]);

  // Apply TER/TEC filtering if enabled
  const terTecFilteredModels: ModelSnapshot[] = useMemo(() => {
    // Priority 1: Bottom panel filter (from TER/TEC Analysis tab)
    if (terTecFilteredModelIds && terTecFilteredModelIds.length > 0) {
      const idSet = new Set(terTecFilteredModelIds);
      return allFilteredModels.filter(m => idSet.has(m.id));
    }

    // Priority 2: Left sidebar filter
    if (!terTecFilterEnabled || terTecTargetValue === 0) {
      return allFilteredModels;
    }

    const toleranceFraction = terTecTolerance / 100;
    const range = terTecFilterType === 'TER'
      ? getTERRange(terTecTargetValue, toleranceFraction)
      : getTECRange(terTecTargetValue, toleranceFraction);

    return allFilteredModels.filter(model => {
      const value = terTecFilterType === 'TER'
        ? calculateTER(model.parameters)
        : calculateTEC(model.parameters);
      return value >= range.min && value <= range.max;
    });
  }, [allFilteredModels, terTecFilterEnabled, terTecFilterType, terTecTargetValue, terTecTolerance, terTecFilteredModelIds]);

  // Get unique TER/TEC values for the slider/selector
  const uniqueTerTecValues = useMemo(() => {
    const values = allFilteredModels.map(m => {
      const value = terTecFilterType === 'TER'
        ? calculateTER(m.parameters)
        : calculateTEC(m.parameters);
      // For TEC (very small values), use more precision
      if (terTecFilterType === 'TEC') {
        // Round to 3 significant figures instead of 3 decimals
        const magnitude = Math.floor(Math.log10(Math.abs(value)));
        const scale = Math.pow(10, magnitude - 2);
        return Math.round(value / scale) * scale;
      }
      return Math.round(value * 1000) / 1000; // Round to 3 decimals for TER
    });
    const uniqueSet = new Set(values.filter(v => v > 0)); // Filter out zero/invalid values
    return Array.from(uniqueSet).sort((a, b) => a - b);
  }, [allFilteredModels, terTecFilterType]);

  // Granular model navigation function
  const navigateModel = useCallback((direction: 'first' | 'previous' | 'next' | 'last') => {
    const modelsToNavigate = terTecFilterEnabled ? terTecFilteredModels : allFilteredModels;
    if (!modelsToNavigate.length) return;

    const currentIndex = highlightedModelId ?
      modelsToNavigate.findIndex(m => m.id === highlightedModelId) : -1;

    let newIndex: number;

    switch (direction) {
      case 'first':
        newIndex = 0;
        break;
      case 'last':
        newIndex = modelsToNavigate.length - 1;
        break;
      case 'previous':
        if (currentIndex <= 0) {
          newIndex = modelsToNavigate.length - 1; // Wrap to last
        } else {
          newIndex = Math.max(0, currentIndex - navigationStepSize);
        }
        break;
      case 'next':
        if (currentIndex >= modelsToNavigate.length - 1 || currentIndex < 0) {
          newIndex = 0; // Wrap to first or start from beginning
        } else {
          newIndex = Math.min(modelsToNavigate.length - 1, currentIndex + navigationStepSize);
        }
        break;
      default:
        return;
    }

    const targetModel = modelsToNavigate[newIndex];
    if (targetModel) {
      setHighlightedModelId(targetModel.id);
      setCurrentResnorm(targetModel.resnorm || null);

      // If the current model is being used for comparison, update the comparison model too
      if (selectedComparisonModel && selectedComparisonModel.id === highlightedModelId) {
        setSelectedComparisonModel(targetModel);
      }
    }
  }, [allFilteredModels, terTecFilteredModels, terTecFilterEnabled, highlightedModelId, navigationStepSize, selectedComparisonModel]);

  // Get visible models within current navigation window
  const visibleModels: ModelSnapshot[] = useMemo(() => {
    // Use terTecFilteredModels which already handles both bottom panel and left sidebar filters
    const modelsToShow = terTecFilteredModels.length > 0 &&
                         (terTecFilterEnabled || (terTecFilteredModelIds && terTecFilteredModelIds.length > 0))
      ? terTecFilteredModels
      : allFilteredModels;

    if (!modelsToShow.length) return [];

    // Navigation window: show up to 1000 models at current offset position
    const startIndex = Math.max(0, Math.min(navigationOffset, modelsToShow.length - 1));
    const endIndex = Math.min(startIndex + navigationWindowSize, modelsToShow.length);

    return modelsToShow.slice(startIndex, endIndex);
  }, [allFilteredModels, terTecFilteredModels, terTecFilterEnabled, terTecFilteredModelIds, navigationOffset, navigationWindowSize]);

  // Memoized tagged model data for Nyquist plot to avoid recalculation
  const taggedModelNyquistData = useMemo(() => {
    if (localTaggedModels.size === 0) return [];

    return Array.from(localTaggedModels.entries()).map(([modelId, tagName]) => {
      // Use allFilteredModels instead of visibleModels for tagged models that might be outside current window
      const taggedModel = allFilteredModels.find(m => m.id === modelId);
      if (!taggedModel?.parameters) return null;

      const colorIndex = Array.from(localTaggedModels.keys()).indexOf(modelId) % tagColors.length;
      return {
        modelId,
        tagName,
        color: tagColors[colorIndex],
        nyquistData: calculateNyquistImpedance(taggedModel.parameters)
      };
    }).filter(Boolean);
  }, [localTaggedModels, allFilteredModels, tagColors, calculateNyquistImpedance]);

  // Memoized tagged model display data to avoid accessing visibleModels during render
  const taggedModelsDisplayData = useMemo(() => {
    if (localTaggedModels.size === 0) return [];

    return Array.from(localTaggedModels.entries()).map(([modelId, tagName]) => {
      // Use allFilteredModels instead of visibleModels to find tagged models that might be outside current window
      const taggedModel = allFilteredModels.find(m => m.id === modelId);
      const colorIndex = Array.from(localTaggedModels.keys()).indexOf(modelId) % tagColors.length;

      return {
        modelId,
        tagName,
        taggedModel,
        color: tagColors[colorIndex]
      };
    });
  }, [localTaggedModels, allFilteredModels, tagColors]);

  // Enhanced worker-assisted rendering for extremely large datasets - unused after removing 2D plot
  // const [isWorkerRendering, setIsWorkerRendering] = useState(false);
  // const [workerProgress, setWorkerProgress] = useState(0);
  // const [workerImageUrl, setWorkerImageUrl] = useState<string | null>(null);

  // Worker-assisted rendering removed after removing 2D plot support

  // Apply opacity strategy to visible models based on resnorm ranking
  const visibleModelsWithOpacity: ModelSnapshot[] = useMemo(() => {
    if (!visibleModels.length) return visibleModels;
    
    // Extract resnorm values for opacity calculation
    const resnorms = visibleModels.map(m => m.resnorm!).filter(r => r !== undefined);
    if (resnorms.length === 0) return visibleModels;
    
    const maxR = Math.max(...resnorms);
    const minR = Math.min(...resnorms);
    
    // Apply mathematical opacity strategy across all visible models
    return visibleModels.map(model => {
      let opacity = 0.8; // Default for equal resnorms
      
      if (model.resnorm !== undefined && maxR > minR) {
        // Apply mathematical formula: r_i = R_i / max(R)
        const r_i = model.resnorm / maxR;
        const min_r = minR / maxR; // Normalized minimum
        
        // Calculate opacity using: O_i = ((1 - r_i) / (1 - min(r_i)))^exponent
        if (Math.abs(1 - min_r) > 1e-10) {
          const baseOpacity = (1 - r_i) / (1 - min_r);
          opacity = Math.pow(baseOpacity, opacityExponent);
        }
      }
      
      // Clamp opacity to reasonable range (0.1 to 1.0)
      opacity = Math.max(0.1, Math.min(1.0, opacity));
      
      return {
        ...model,
        opacity
      };
    });
  }, [visibleModels, opacityExponent]);

  // Check if we have actual computed results to show (updated for gridResults-based system)
  const hasComputedResults = modelsWithUpdatedResnorm && modelsWithUpdatedResnorm.length > 0;
  
  // Debug logging for visualization state
  console.log('VisualizerTab state:', {
    hasComputedResults,
    modelsLength: modelsWithUpdatedResnorm?.length || 0,
    gridResultsLength: gridResults?.length || 0,
    resnormGroupsLength: resnormGroups?.length || 0
  });

  // Log the data being passed to bottom panel
  const bottomPanelConfigs = resnormGroups.length > 0 ? resnormGroups : [{
    range: [0, 100],
    color: '#3B82F6',
    label: 'Computed Results',
    description: 'All computed circuit models',
    items: modelsWithUpdatedResnorm || []
  }];

  console.log('Bottom panel configurations:', {
    configCount: bottomPanelConfigs.length,
    firstConfigItems: bottomPanelConfigs[0]?.items?.length || 0,
    sampleModel: bottomPanelConfigs[0]?.items?.[0]
  });


  return (
    <div className="h-full flex flex-col bg-neutral-900">
      {hasComputedResults ? (
        <>
          {/* Main Content Area - Split into visualization and bottom panel */}
          <div className="flex-1 flex min-h-0 relative">
            {/* Left Sidebar - Collapsible Control Panels (stops at bottom panel) */}
            <div className={`${leftPanelCollapsed ? 'w-0' : 'w-80'} bg-neutral-800 border-r border-neutral-700 flex flex-col transition-all duration-300 overflow-hidden absolute left-0 top-0 bottom-0 z-10`}>
              {/* Scrollable Content Container */}
              <div className="flex-1 overflow-y-auto p-3 space-y-4">
              <div className="flex items-center gap-4">
              {/* Visualization Type Selection */}
              <select
                value={visualizationType}
                onChange={(e) => setVisualizationType(e.target.value as 'spider3d' | 'nyquist' | 'tsne' | 'pentagon-gt' | 'pentagon-quartile')}
                className="px-3 py-1.5 text-sm bg-neutral-900 border border-neutral-600 rounded-md text-neutral-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
              >
                <option value="spider3d">Spider 3D</option>
                <option value="nyquist">Nyquist Plot</option>
                <option value="tsne">t-SNE 3D Space</option>
                <option value="pentagon-gt">Pentagon Ground Truth</option>
                <option value="pentagon-quartile">Pentagon Quartile</option>
              </select>
            </div> {/* End left toolbar controls */}
                {/* Ground Truth Control */}
                <div className="bg-neutral-700 rounded-lg p-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-neutral-200">Ground Truth</span>
                    <button
                      onClick={() => onStaticRenderSettingsChange({
                        ...staticRenderSettings,
                        showGroundTruth: !showGroundTruth
                      })}
                      className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                        showGroundTruth ? 'bg-green-600' : 'bg-neutral-600'
                      }`}
                    >
                      <span
                        className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform ${
                          showGroundTruth ? 'translate-x-5' : 'translate-x-1'
                        }`}
                      />
                    </button>
                  </div>
                  <p className="text-xs text-neutral-400 mt-2">
                    Reference Params
                  </p>
                </div>

                {/* 3D Controls - only show when 3D is enabled */}
                {view3D && (
                  <div className="bg-neutral-700 rounded-lg p-3">
                    <h4 className="text-sm font-medium text-neutral-200 mb-2">Plot Style</h4>

                    <div className="space-y-3">
                      {/* 3D Scale Control */}
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs text-neutral-400">Spread</span>
                          <span className="text-xs text-neutral-200 font-mono">{staticRenderSettings.resnormSpread.toFixed(1)}x</span>
                        </div>
                        <input
                          type="range"
                          min={0.1}
                          max={10.0}
                          step={0.1}
                          value={isNaN(staticRenderSettings.resnormSpread) ? 1.0 : staticRenderSettings.resnormSpread}
                          onChange={e => onStaticRenderSettingsChange({
                            ...staticRenderSettings,
                            resnormSpread: Number(e.target.value)
                          })}
                          className="w-full h-2 bg-neutral-600 rounded-lg appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-purple-500"
                        />
                      </div>

                      {/* Resnorm Center Control */}
                      <div>
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-neutral-400">Resnorm Center</span>
                          <button
                            onClick={() => onStaticRenderSettingsChange({
                              ...staticRenderSettings,
                              useResnormCenter: !staticRenderSettings.useResnormCenter
                            })}
                            className={`relative inline-flex h-4 w-8 items-center rounded-full transition-colors ${
                              staticRenderSettings.useResnormCenter ? 'bg-purple-600' : 'bg-neutral-600'
                            }`}
                          >
                            <span
                              className={`inline-block h-2 w-2 transform rounded-full bg-white transition-transform ${
                                staticRenderSettings.useResnormCenter ? 'translate-x-5' : 'translate-x-1'
                              }`}
                            />
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Model Filtering */}
                <div className="bg-neutral-700 rounded-lg p-3">
                  <h4 className="text-sm font-medium text-neutral-200 mb-2">Model Filtering</h4>

                  {/* Group Portion */}
                  <div className="space-y-3">
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs text-neutral-400">Group Portion</span>
                        <span className="text-xs text-neutral-200">{(groupPortion * 100).toFixed(1)}% ({Math.floor(allFilteredModels.length * groupPortion)} models)</span>
                      </div>

                      {/* Convert groupPortion (0-1) to logarithmic slider (1-100) for UI */}
                      <input
                        type="range"
                        min={1}
                        max={100}
                        step={1}
                        value={(() => {
                          const logPercent = groupPortion * 100;
                          if (logPercent <= 0.01) return 1;
                          const minLog = Math.log10(0.01);
                          const maxLog = Math.log10(100);
                          const currentLog = Math.log10(logPercent);
                          return Math.round(((currentLog - minLog) / (maxLog - minLog)) * 99 + 1);
                        })()}
                        onChange={e => {
                          const sliderValue = Number(e.target.value);
                          const minLog = Math.log10(0.01);
                          const maxLog = Math.log10(100);
                          const logValue = minLog + (sliderValue - 1) / 99 * (maxLog - minLog);
                          const logPercent = Math.pow(10, logValue);
                          const groupPortionValue = logPercent / 100; // Convert back to 0-1 range
                          onGroupPortionChange(groupPortionValue);
                        }}
                        className="w-full h-2 bg-neutral-600 rounded-lg appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-blue-500"
                      />

                      <div className="flex justify-between text-xs text-neutral-500 mt-1">
                        <span>0.01%</span>
                        <span className="text-orange-400 font-medium">Nav: 1-{Math.min(navigationWindowSize, allFilteredModels.length)} of {allFilteredModels.length.toLocaleString()}</span>
                        <span>100%</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* TER/TEC Filtering */}
                <div className="bg-neutral-700 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-sm font-medium text-neutral-200">TER/TEC Filter</h4>
                    <button
                      onClick={() => setTerTecFilterEnabled(!terTecFilterEnabled)}
                      className={`relative inline-flex h-4 w-8 items-center rounded-full transition-colors ${
                        terTecFilterEnabled ? 'bg-emerald-600' : 'bg-neutral-600'
                      }`}
                    >
                      <span
                        className={`inline-block h-2 w-2 transform rounded-full bg-white transition-transform ${
                          terTecFilterEnabled ? 'translate-x-5' : 'translate-x-1'
                        }`}
                      />
                    </button>
                  </div>

                  {terTecFilterEnabled && (
                    <div className="space-y-3 mt-3">
                      {/* Metric Type Toggle */}
                      <div className="flex gap-2">
                        <button
                          onClick={() => setTerTecFilterType('TER')}
                          className={`flex-1 px-2 py-1 text-xs rounded transition-colors ${
                            terTecFilterType === 'TER'
                              ? 'bg-blue-600 text-white'
                              : 'bg-neutral-800 text-neutral-400 hover:bg-neutral-750'
                          }`}
                        >
                          TER
                        </button>
                        <button
                          onClick={() => setTerTecFilterType('TEC')}
                          className={`flex-1 px-2 py-1 text-xs rounded transition-colors ${
                            terTecFilterType === 'TEC'
                              ? 'bg-blue-600 text-white'
                              : 'bg-neutral-800 text-neutral-400 hover:bg-neutral-750'
                          }`}
                        >
                          TEC
                        </button>
                      </div>

                      {/* Target Value Dropdown */}
                      <div>
                        <label className="block text-xs text-neutral-400 mb-1">
                          Target {terTecFilterType}
                        </label>
                        <select
                          value={terTecTargetValue}
                          onChange={(e) => setTerTecTargetValue(parseFloat(e.target.value) || 0)}
                          className="w-full px-2 py-1 text-xs bg-neutral-800 border border-neutral-600 rounded text-neutral-200 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                        >
                          <option value={0}>Select {terTecFilterType} value...</option>
                          {uniqueTerTecValues.map((value, index) => (
                            <option key={index} value={value}>
                              {terTecFilterType === 'TER' ? formatTER(value) : formatTEC(value)}
                            </option>
                          ))}
                        </select>
                        <div className="text-xs text-neutral-500 mt-1">
                          {uniqueTerTecValues.length} unique values
                        </div>
                      </div>

                      {/* Tolerance Slider */}
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-xs text-neutral-400">Tolerance</span>
                          <span className="text-xs text-neutral-200">±{terTecTolerance}%</span>
                        </div>
                        <input
                          type="range"
                          min="1"
                          max="20"
                          step="1"
                          value={terTecTolerance}
                          onChange={(e) => setTerTecTolerance(parseInt(e.target.value))}
                          className="w-full h-2 bg-neutral-600 rounded-lg appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-emerald-500"
                        />
                      </div>

                      {/* Filter Results */}
                      <div className="bg-neutral-800 rounded px-2 py-1.5">
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-neutral-400">Filtered Models</span>
                          <span className="text-xs font-semibold text-emerald-400">
                            {terTecFilteredModels.length.toLocaleString()}
                          </span>
                        </div>
                        {terTecTargetValue > 0 && (
                          <div className="text-xs text-neutral-500 mt-1">
                            {terTecFilterType === 'TER' ? formatTER(terTecTargetValue) : formatTEC(terTecTargetValue)}
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>


                {/* Nyquist Configuration Selector - only for Nyquist plot */}
                {visualizationType === 'nyquist' && topConfigurations.length > 0 && (
                  <div className="bg-neutral-700 rounded-lg p-3">
                    <h4 className="text-sm font-medium text-neutral-200 mb-2">Circuit Selection</h4>

                    <select
                      value={isNaN(selectedNyquistConfig) ? 0 : selectedNyquistConfig}
                      onChange={(e) => setSelectedNyquistConfig(Number(e.target.value))}
                      className="w-full bg-neutral-800 border border-neutral-600 rounded px-2 py-1 text-sm text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                    >
                      {topConfigurations.map((config, index) => (
                        <option key={index} value={index}>
                          #{index + 1} - Resnorm: {(config.resnorm || 0).toFixed(4)}
                        </option>
                      ))}
                    </select>

                    {currentNyquistConfig && (
                      <div className="mt-2 p-2 bg-neutral-800 rounded text-xs">
                        <div className="grid grid-cols-2 gap-1 text-neutral-300">
                          <div>Rs: {currentNyquistConfig.point.parameters.Rsh.toFixed(1)}Ω</div>
                          <div>Ra: {currentNyquistConfig.point.parameters.Ra.toFixed(1)}Ω</div>
                          <div>Ca: {(currentNyquistConfig.point.parameters.Ca * 1e6).toFixed(2)}µF</div>
                          <div>Rb: {currentNyquistConfig.point.parameters.Rb.toFixed(1)}Ω</div>
                          <div>Cb: {(currentNyquistConfig.point.parameters.Cb * 1e6).toFixed(2)}µF</div>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Model Selection & Navigation */}
                <div className="bg-neutral-700 rounded-lg p-3 space-y-3">
                  <h4 className="text-sm font-medium text-neutral-200">Model Selection</h4>

                  {/* Current Model Display */}
                  {highlightedModelId && (
                    <div className="bg-neutral-800 rounded px-3 py-2">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs text-neutral-400">Selected Model</span>
                        <button
                          onClick={() => setHighlightedModelId(null)}
                          className="text-neutral-400 hover:text-neutral-200 text-xs"
                          title="Clear selection"
                        >
                          ×
                        </button>
                      </div>
                      <div className="text-xs text-cyan-300">
                        ID: {highlightedModelId.slice(0, 8)}...
                      </div>
                      {(() => {
                        const selectedModel = allFilteredModels.find(m => m.id === highlightedModelId);
                        return selectedModel && (
                          <div className="text-xs text-neutral-300 mt-1">
                            Resnorm: {selectedModel.resnorm?.toFixed(4) || 'N/A'}
                          </div>
                        );
                      })()}
                    </div>
                  )}

                  {/* Granular Navigation Controls */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-neutral-400">Step Size</span>
                      <div className="flex items-center gap-2">
                        <input
                          type="text"
                          min="1"
                          max="100000"
                          value={navigationStepSize}
                          onChange={(e) => setNavigationStepSize(Number(e.target.value))}
                          className="w-full px-1 py-1 bg-neutral-800 border border-neutral-600 rounded text-neutral-200 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 transition-colors"
                          title={`Step size: ${navigationStepSize} models`}
                        />
                      </div>
                    </div>

                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => navigateModel('first')}
                        className="px-2 py-1 text-xs bg-neutral-600 hover:bg-neutral-500 rounded transition-colors"
                        title="Go to first model"
                        disabled={!allFilteredModels.length}
                      >
                        <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 16 16">
                          <path d="M8.354 1.146a.5.5 0 0 0-.708 0l-6 6a.5.5 0 0 0 0 .708l6 6a.5.5 0 0 0 .708-.708L2.707 7.5H14.5a.5.5 0 0 0 0-1H2.707l5.647-5.646a.5.5 0 0 0 0-.708z"/>
                          <path d="M1 1v14h1V1H1z"/>
                        </svg>
                      </button>
                      <button
                        onClick={() => navigateModel('previous')}
                        className="px-2 py-1 text-xs bg-neutral-600 hover:bg-neutral-500 rounded transition-colors"
                        title={`Previous ${navigationStepSize} model(s)`}
                        disabled={!allFilteredModels.length}
                      >
                        <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 16 16">
                          <path fillRule="evenodd" d="M11.354 1.646a.5.5 0 0 1 0 .708L5.707 8l5.647 5.646a.5.5 0 0 1-.708.708l-6-6a.5.5 0 0 1 0-.708l6-6a.5.5 0 0 1 .708 0z"/>
                          <path fillRule="evenodd" d="M6.354 1.646a.5.5 0 0 1 0 .708L.707 8l5.647 5.646a.5.5 0 0 1-.708.708l-6-6a.5.5 0 0 1 0-.708l6-6a.5.5 0 0 1 .708 0z"/>
                        </svg>
                      </button>
                      <button
                        onClick={() => navigateModel('next')}
                        className="px-2 py-1 text-xs bg-neutral-600 hover:bg-neutral-500 rounded transition-colors"
                        title={`Next ${navigationStepSize} model(s)`}
                        disabled={!allFilteredModels.length}
                      >
                        <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 16 16">
                          <path fillRule="evenodd" d="M4.646 1.646a.5.5 0 0 1 .708 0l6 6a.5.5 0 0 1 0 .708l-6 6a.5.5 0 0 1-.708-.708L10.293 8 4.646 2.354a.5.5 0 0 1 0-.708z"/>
                          <path fillRule="evenodd" d="M9.646 1.646a.5.5 0 0 1 .708 0l6 6a.5.5 0 0 1 0 .708l-6 6a.5.5 0 0 1-.708-.708L15.293 8 9.646 2.354a.5.5 0 0 1 0-.708z"/>
                        </svg>
                      </button>
                      <button
                        onClick={() => navigateModel('last')}
                        className="px-2 py-1 text-xs bg-neutral-600 hover:bg-neutral-500 rounded transition-colors"
                        title="Go to last model"
                        disabled={!allFilteredModels.length}
                      >
                        <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 16 16">
                          <path d="M7.646 1.146a.5.5 0 0 1 .708 0l6 6a.5.5 0 0 1 0 .708l-6 6a.5.5 0 0 1-.708-.708L13.293 7.5H1.5a.5.5 0 0 1 0-1h11.793L7.646 1.854a.5.5 0 0 1 0-.708z"/>
                          <path d="M15 1v14h-1V1h1z"/>
                        </svg>
                      </button>
                    </div>

                    {/* Model Position Indicator */}
                    {highlightedModelId && (
                      <div className="text-xs text-neutral-400 text-center">
                        {(() => {
                          const currentIndex = allFilteredModels.findIndex(m => m.id === highlightedModelId);
                          const currentModel = allFilteredModels.find(m => m.id === highlightedModelId);
                          const resnormValue = currentModel?.resnorm?.toFixed(4) || 'N/A';
                          if (currentIndex >= 0) {
                            return (
                              <div>
                                <div>Rank {currentIndex + 1} of {allFilteredModels.length}</div>
                                <div className="text-cyan-400">Resnorm: {resnormValue}</div>
                              </div>
                            );
                          }
                          return 'Not found';
                        })()}
                      </div>
                    )}
                  </div>
                </div>

                {/* Comparison Selection */}
                {(localTaggedModels.size > 0 || highlightedModelId) && (
                  <div className="bg-neutral-700 rounded-lg p-3">
                    <h4 className="text-sm font-medium text-neutral-200 mb-2">Compare with Reference</h4>

                    <select
                      className="w-full bg-neutral-800 border border-neutral-600 rounded px-2 py-1 text-sm text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                      value={selectedComparisonModel ? (localTaggedModels.has(selectedComparisonModel.id) ? `tagged:${selectedComparisonModel.id}` : 'current') : 'ground-truth'}
                      onChange={(e) => {
                        const value = e.target.value;
                        if (value === '' || value === 'ground-truth') {
                          // Select ground truth for comparison
                          setSelectedComparisonModel(null);
                          setHighlightedModelId(null);
                          setCurrentResnorm(null);
                        } else if (value.startsWith('tagged:')) {
                          // Select tagged model for comparison
                          const modelId = value.replace('tagged:', '');
                          const taggedModel = allFilteredModels.find(m => m.id === modelId);
                          if (taggedModel) {
                            setSelectedComparisonModel(taggedModel);
                            setHighlightedModelId(modelId);
                            setCurrentResnorm(taggedModel.resnorm || null);
                          } else {
                            console.warn('Tagged model not found in allFilteredModels:', modelId);
                          }
                        } else if (value === 'current' && highlightedModelId) {
                          // Use current selection for comparison
                          const currentModel = allFilteredModels.find(m => m.id === highlightedModelId);
                          if (currentModel) {
                            setSelectedComparisonModel(currentModel);
                            setCurrentResnorm(currentModel.resnorm || null);
                          } else {
                            console.warn('Current model not found in allFilteredModels:', highlightedModelId);
                          }
                        }
                      }}
                    >
                      <option value="ground-truth">Ground Truth (Reference)</option>
                      {highlightedModelId && (
                        <option value="current">Current Selection</option>
                      )}
                      {Array.from(localTaggedModels.entries()).map(([modelId, tagName]) => (
                        <option key={modelId} value={`tagged:${modelId}`}>
                          {tagName} {(() => {
                            const model = allFilteredModels.find(m => m.id === modelId);
                            return model ? `(${model.resnorm?.toFixed(4) || 'N/A'})` : '';
                          })()}
                        </option>
                      ))}
                    </select>

                    <div className="text-xs text-neutral-400 mt-2">
                      Select a model to compare in the bottom data panel
                    </div>
                  </div>
                )}

                {/* Tagged Models Display */}
                {localTaggedModels.size > 0 && (
                  <div className="bg-neutral-700 rounded-lg p-3">
                    <h4 className="text-sm font-medium text-neutral-200 mb-2">Tagged Models ({localTaggedModels.size})</h4>
                    <div className="space-y-2 max-h-32 overflow-y-auto">
                      {taggedModelsDisplayData.map(({ modelId, tagName, taggedModel, color }) => (
                        <div key={modelId} className={`flex items-center justify-between text-xs rounded px-2 py-1 transition-colors cursor-pointer ${
                          highlightedModelId === modelId ? 'bg-cyan-600/30 border border-cyan-400' : 'hover:bg-neutral-600/50'
                        }`}
                        onClick={() => handleTaggedModelSelect(modelId)}
                        title="Click to highlight in 3D view"
                        >
                          <div className="flex items-center gap-2">
                            <div
                              className="w-3 h-3 rounded-full"
                              style={{ backgroundColor: color }}
                            />
                            <span className="text-neutral-200">{tagName}</span>
                            {taggedModel && (
                              <span className="text-neutral-400">
                                (resnorm: {taggedModel.resnorm?.toFixed(4) || 'N/A'})
                              </span>
                            )}
                          </div>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleModelTag(taggedModel!, '');
                            }}
                            className="text-neutral-400 hover:text-neutral-200 px-1 rounded hover:bg-neutral-500"
                            title="Remove tag"
                          >
                            ×
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

              </div> {/* End scrollable container */}

              {/* Collapse toggle button */}
              <button
                onClick={() => setLeftPanelCollapsed(true)}
                className="absolute top-2 right-2 w-6 h-6 flex items-center justify-center rounded bg-neutral-700 hover:bg-neutral-600 transition-colors z-10"
                title="Collapse panel"
              >
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
              </button>
            </div> {/* End left sidebar */}

            {/* Expand button when collapsed */}
            {leftPanelCollapsed && (
              <button
                onClick={() => setLeftPanelCollapsed(false)}
                className="absolute left-0 top-1/2 -translate-y-1/2 w-6 h-12 flex items-center justify-center rounded-r bg-neutral-800 hover:bg-neutral-700 border-r border-t border-b border-neutral-600 transition-colors z-10"
                title="Expand panel"
              >
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </button>
            )}

            {/* Visualization and Bottom Panel Container */}
            <div className={`flex-1 flex flex-col min-h-0 ${leftPanelCollapsed ? '' : 'ml-80'} transition-all duration-300`}>
              {/* Primary Visualization Area */}
              <div className="flex-1 relative bg-neutral-950 min-h-0">
              {visualizationType === 'spider3d' ? (
                /* Spider 3D Visualization */
                <div className="w-full h-full relative overflow-hidden">
                  {isInitializing && visibleModelsWithOpacity.length > 0 ? (
                    <div className="absolute inset-0 bg-neutral-950 flex items-center justify-center">
                      <div className="text-white text-lg flex items-center gap-3">
                        <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                        <span>Initializing 3D visualization...</span>
                      </div>
                    </div>
                  ) : (
                    <SpiderPlot3D
                      models={visibleModelsWithOpacity}
                      referenceModel={showGroundTruth && effectiveGroundTruthParams ? {
                        id: 'ground-truth',
                        name: 'Ground Truth Reference',
                        timestamp: Date.now(),
                        parameters: effectiveGroundTruthParams,
                        data: [],
                        resnorm: 0,
                        color: '#FFFFFF',
                        isVisible: true,
                        opacity: 1
                      } : null}
                      responsive={true} // Enable responsive sizing
                      showControls={true}
                      gridSize={gridSize}
                      resnormSpread={staticRenderSettings.resnormSpread}
                      useResnormCenter={staticRenderSettings.useResnormCenter}
                      resnormRange={selectedResnormRange}
                      onModelTag={handleModelTag}
                      onModelSelect={handleModelSelect}
                      taggedModels={localTaggedModels}
                      currentResnorm={currentResnorm}
                      onResnormSelect={handleResnormSelect}
                      onCurrentResnormChange={handleCurrentResnormChange}
                      highlightedModelId={highlightedModelId}
                      selectedResnormRange={selectedResnormRange}
                    />
                  )}
                </div>
              ) : visualizationType === 'tsne' ? (
                /* Parameter Space 3D Visualization */
                <div className="w-full h-full relative overflow-hidden">
                  {isInitializing && visibleModelsWithOpacity.length > 0 ? (
                    <div className="absolute inset-0 bg-neutral-950 flex items-center justify-center">
                      <div className="text-white text-lg flex items-center gap-3">
                        <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                        <span>Computing parameter space...</span>
                      </div>
                    </div>
                  ) : (
                    <TSNEPlot3D
                      models={visibleModelsWithOpacity}
                      referenceModel={showGroundTruth && effectiveGroundTruthParams ? {
                        id: 'ground-truth',
                        name: 'Ground Truth Reference',
                        timestamp: Date.now(),
                        parameters: effectiveGroundTruthParams,
                        data: calculate_impedance_spectrum(effectiveGroundTruthParams), // Provide impedance spectrum
                        resnorm: 0,
                        color: '#FFFFFF',
                        isVisible: true,
                        opacity: 1
                      } : null}
                      responsive={true}
                    />
                  )}
                </div>
              ) : visualizationType === 'pentagon-gt' ? (
                /* Pentagon Ground Truth Visualization */
                <div className="w-full h-full relative overflow-hidden">
                  <PentagonGroundTruth
                    models={visibleModelsWithOpacity}
                    currentParameters={groundTruthParams}
                  />
                </div>
              ) : visualizationType === 'pentagon-quartile' ? (
                /* Pentagon Quartile Box-Whisker Visualization */
                <div className="w-full h-full relative overflow-hidden">
                  <PentagonBoxWhisker
                    models={visibleModelsWithOpacity}
                    currentParameters={effectiveGroundTruthParams}
                    gridSize={gridSize}
                  />
                </div>
              ) : (
                /* Nyquist Plot Visualization - Single Circuit */
                <div className="absolute inset-0 flex items-center justify-center">
                  {isInitializing && topConfigurations.length > 0 ? (
                    <div className="text-white text-lg flex items-center gap-3">
                      <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                      <span>Initializing Nyquist plot...</span>
                    </div>
                  ) : currentNyquistConfig ? (
                    <div className="w-full h-full p-4">
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                          <XAxis 
                            type="number"
                            dataKey="real"
                            name="Real(Z)"
                            stroke="#9CA3AF"
                            domain={['dataMin', 'dataMax']}
                            label={{ value: 'Real(Z) [Ω]', position: 'insideBottom', offset: -5, style: { textAnchor: 'middle', fill: '#9CA3AF' } }}
                          />
                          <YAxis 
                            type="number"
                            dataKey="imag"
                            name="-Imag(Z)"
                            stroke="#9CA3AF"
                            domain={['dataMin', 'dataMax']}
                            label={{ value: '-Imag(Z) [Ω]', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle', fill: '#9CA3AF' } }}
                          />
                          <Tooltip 
                            formatter={(value, name) => {
                              if (typeof value === 'number') {
                                return [value.toFixed(2) + ' Ω', name];
                              }
                              return [value, name];
                            }}
                            labelFormatter={(label) => {
                              if (typeof label === 'number') {
                                return `Frequency: ${label.toFixed(2)} Hz`;
                              }
                              return `Frequency: ${label} Hz`;
                            }}
                          />
                          <Scatter
                            name={`Config ${selectedNyquistConfig + 1}`}
                            data={currentNyquistConfig.nyquistData}
                            fill="#3B82F6"
                            line={true}
                            lineJointType="monotoneX"
                            lineType="joint"
                          />
                          {/* Ground Truth Line if enabled */}
                          {showGroundTruth && effectiveGroundTruthParams && (() => {
                            const gtNyquistData = calculateNyquistImpedance(effectiveGroundTruthParams);
                            return (
                              <Scatter
                                name="Ground Truth"
                                data={gtNyquistData}
                                fill="#FFFFFF"
                                line={true}
                                lineJointType="monotoneX"
                                lineType="joint"
                              />
                            );
                          })()}
                          
                          {/* Tagged Models Overlays */}
                          {taggedModelNyquistData.map((taggedData) => (
                            <Scatter
                              key={taggedData!.modelId}
                              name={`${taggedData!.tagName} (Tagged)`}
                              data={taggedData!.nyquistData}
                              fill={taggedData!.color}
                              line={true}
                              lineJointType="monotoneX"
                              lineType="joint"
                            />
                          ))}
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <div className="text-center text-neutral-400">
                      <div className="text-lg mb-2">No Configuration Selected</div>
                      <div className="text-sm">Adjust the model selection percentage to see circuits</div>
                    </div>
                  )}
                </div>
              )}
              {/* Bottom-right expand caret when panel is collapsed */}
              {bottomPanelCollapsed && (
                <button
                  onClick={() => setBottomPanelCollapsed(false)}
                  className="absolute bottom-4 right-4 bg-neutral-800 hover:bg-neutral-700 border border-neutral-600 rounded-md px-3 py-2 text-neutral-400 hover:text-neutral-200 transition-all duration-200 shadow-lg z-50 flex items-center gap-2"
                  title="Show data panel"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                  </svg>
                  <span className="text-xs">Data</span>
                </button>
              )}

              </div> {/* End Primary Visualization Area */}

              {/* Bottom Panel - PyCharm Style with Enhanced Scrolling */}
              <CollapsibleBottomPanel
                gridResults={gridResults || []}
                topConfigurations={(() => {
                  // Create ResnormGroup array with ground truth and selected comparison model
                  const configs: ResnormGroup[] = [];

                  // Always include ground truth as first configuration
                  if (groundTruthParams) {
                    const groundTruthModel: ModelSnapshot = {
                      id: 'ground-truth',
                      name: 'Ground Truth Reference',
                      timestamp: Date.now(),
                      parameters: groundTruthParams,
                      data: [],
                      resnorm: 0,
                      color: '#FFFFFF',
                      isVisible: true,
                      opacity: 1
                    };

                    configs.push({
                      range: [0, 0] as [number, number],
                      color: '#FFFFFF',
                      label: 'Ground Truth',
                      description: 'Reference circuit configuration',
                      items: [groundTruthModel]
                    });
                  }

                  // Add selected comparison model if available
                  if (selectedComparisonModel) {
                    configs.push({
                      range: [selectedComparisonModel.resnorm || 0, selectedComparisonModel.resnorm || 0] as [number, number],
                      color: '#3B82F6',
                      label: `Selected Model`,
                      description: `Resnorm: ${selectedComparisonModel.resnorm?.toFixed(4) || 'N/A'}`,
                      items: [selectedComparisonModel]
                    });
                  }

                  return configs;
                })()}
                currentParameters={groundTruthParams}
                selectedConfigIndex={selectedComparisonModel ? 1 : 0}
                onConfigurationSelect={(index: number) => {
                  // Handle configuration selection for table comparison
                  console.log('Selected configuration index:', index);
                  // Sync with comparison selection if needed
                  if (index === 0) {
                    // Ground truth selected
                    setSelectedComparisonModel(null);
                  } else if (selectedComparisonModel && index === 1) {
                    // Selected comparison model is already active
                    // No need to change anything
                  }
                }}
                highlightedModelId={highlightedModelId}
                gridSize={gridSize}
                isCollapsed={bottomPanelCollapsed}
                onToggleCollapse={setBottomPanelCollapsed}
                height={bottomPanelHeight}
                onHeightChange={setBottomPanelHeight}
                minHeight={120}
                maxHeight={500}
                // Resnorm-related props for new ResnormRangeTab
                currentResnorm={currentResnorm}
                onCurrentResnormChange={handleCurrentResnormChange}
                selectedResnormRange={selectedResnormRange}
                onResnormRangeChange={handleResnormRangeChange}
                onResnormSelect={handleResnormSelect}
                navigationOffset={navigationOffset}
                onNavigationOffsetChange={setNavigationOffset}
                navigationWindowSize={navigationWindowSize}
                taggedModels={localTaggedModels}
                tagColors={tagColors}
                onTERTECFilterChange={(filteredIds) => {
                  // Propagate TER/TEC filter to parent CircuitSimulator
                  if (onTERTECFilterChange) {
                    onTERTECFilterChange(filteredIds);
                  }
                }}
              />
            </div> {/* End Visualization and Bottom Panel Container */}

          </div> {/* End main layout container */}
        </>
      ) : (
        /* Empty State */
        <div className="flex-1 flex items-center justify-center bg-neutral-900">
          <div className="text-center max-w-md mx-auto p-8">
            <div className="w-16 h-16 mx-auto mb-4 bg-neutral-700 rounded-full flex items-center justify-center">
              <svg className="w-8 h-8 text-neutral-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <h3 className="text-lg font-medium text-neutral-200 mb-2">No Data to Visualize</h3>
            <p className="text-sm text-neutral-400">
              Run a parameter grid computation to generate visualization data.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
