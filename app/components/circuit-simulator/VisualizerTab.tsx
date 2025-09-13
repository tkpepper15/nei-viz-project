import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
// SpiderPlot 2D import removed - using only 3D visualization
import { SpiderPlot3D } from './visualizations/SpiderPlot3D';
import { ModelSnapshot, ResnormGroup } from './types';
import { GridParameterArrays } from './types';
import { CircuitParameters } from './types/parameters';
import { StaticRenderSettings } from './controls/StaticRenderControls';
import { ResnormConfig, calculateResnormWithConfig, calculate_impedance_spectrum } from './utils/resnorm';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { calculateTimeConstants } from './math/utils';
import ResnormDisplay from './insights/ResnormDisplay';

interface VisualizationSettings {
  groupPortion: number;
  selectedOpacityGroups: number[];
  visualizationType: 'spider3d' | 'nyquist';
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
  onModelTag
}) => {

  // Removed old control states - now using compact sidebar design
  
  // Use static render settings instead of local state
  const showGroundTruth = staticRenderSettings.showGroundTruth;
  
  // Note: resnorm matching portion no longer used with new logarithmic approach
  // const [resnormMatchingPortion] = useState<number>(25); // Removed - replaced with logarithmic selection
  // Use ground truth parameters from toolbox (userReferenceParams) or direct groundTruthParams
  const effectiveGroundTruthParams = userReferenceParams || groundTruthParams;
  // Use visualization settings from static render settings - defaulting to 3D only
  const visualizationType = staticRenderSettings.visualizationType === 'nyquist' ? 'nyquist' : 'spider3d';
  const view3D = visualizationType === 'spider3d';
  const setVisualizationType = (type: 'spider3d' | 'nyquist') => {
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
      console.log('🔄 Using gridResults for visualization:', gridResults.length, 'models');
      
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

  // Get visible models within current navigation window
  const visibleModels: ModelSnapshot[] = useMemo(() => {
    if (!allFilteredModels.length) return [];

    // Navigation window: show up to 1000 models at current offset position
    const startIndex = Math.max(0, Math.min(navigationOffset, allFilteredModels.length - 1));
    const endIndex = Math.min(startIndex + navigationWindowSize, allFilteredModels.length);

    return allFilteredModels.slice(startIndex, endIndex);
  }, [allFilteredModels, navigationOffset, navigationWindowSize]);

  // Memoized tagged model data for Nyquist plot to avoid recalculation
  const taggedModelNyquistData = useMemo(() => {
    if (localTaggedModels.size === 0) return [];
    
    return Array.from(localTaggedModels.entries()).map(([modelId, tagName]) => {
      const taggedModel = visibleModels.find(m => m.id === modelId);
      if (!taggedModel?.parameters) return null;
      
      const colorIndex = Array.from(localTaggedModels.keys()).indexOf(modelId) % tagColors.length;
      return {
        modelId,
        tagName,
        color: tagColors[colorIndex],
        nyquistData: calculateNyquistImpedance(taggedModel.parameters)
      };
    }).filter(Boolean);
  }, [localTaggedModels, visibleModels, tagColors, calculateNyquistImpedance]);

  // Memoized tagged model display data to avoid accessing visibleModels during render
  const taggedModelsDisplayData = useMemo(() => {
    if (localTaggedModels.size === 0) return [];
    
    return Array.from(localTaggedModels.entries()).map(([modelId, tagName]) => {
      const taggedModel = visibleModels.find(m => m.id === modelId);
      const colorIndex = Array.from(localTaggedModels.keys()).indexOf(modelId) % tagColors.length;
      
      return {
        modelId,
        tagName,
        taggedModel,
        color: tagColors[colorIndex]
      };
    });
  }, [localTaggedModels, visibleModels, tagColors]);

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
  console.log('📊 VisualizerTab state:', {
    hasComputedResults,
    modelsLength: modelsWithUpdatedResnorm?.length || 0,
    gridResultsLength: gridResults?.length || 0,
    resnormGroupsLength: resnormGroups?.length || 0
  });


  return (
    <div className="h-full flex flex-col bg-neutral-900">
      {hasComputedResults ? (
        <>
          {/* Top Toolbar - Clean and Minimal */}
          <div className="flex items-center justify-between px-4 py-2 bg-neutral-800 border-b border-neutral-700 flex-shrink-0">
            <div className="flex items-center gap-4">
              {/* Visualization Type Selection */}
              <select
                value={visualizationType}
                onChange={(e) => setVisualizationType(e.target.value as 'spider3d' | 'nyquist')}
                className="px-3 py-1.5 text-sm bg-neutral-900 border border-neutral-600 rounded-md text-neutral-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
              >
                <option value="spider3d">Spider 3D</option>
                <option value="nyquist">Nyquist Plot</option>
              </select>
            </div>

          </div>

          {/* Main Content Area */}
          <div className="flex-1 flex overflow-hidden">
            {/* Primary Visualization Area */}
            <div className="flex-1 relative bg-black">
              {visualizationType === 'spider3d' ? (
                /* Spider 3D Visualization */
                <div className="w-full h-full relative overflow-hidden">
                  {isInitializing && visibleModelsWithOpacity.length > 0 ? (
                    <div className="absolute inset-0 bg-black flex items-center justify-center">
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
                    taggedModels={localTaggedModels}
                    currentResnorm={currentResnorm}
                    onResnormSelect={handleResnormSelect}
                    onCurrentResnormChange={handleCurrentResnormChange}
                    highlightedModelId={highlightedModelId}
                    selectedResnormRange={selectedResnormRange}
                  />
                  )}
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
            </div>

            {/* Right Sidebar - Scrollable Control Panels */}
            <div className="w-80 bg-neutral-800 border-l border-neutral-700 flex flex-col overflow-hidden">
              {/* Scrollable Content Container */}
              <div className="flex-1 overflow-y-auto p-3 space-y-4">
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
                    Show reference parameters from toolbox
                  </p>
                </div>
                
                {/* Note: Resnorm distribution histogram moved to top-right overlay */}
                
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

                {/* Tagged Models Display - for all plot types but especially relevant for Nyquist */}
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

                {/* 3D Controls - only show when 3D is enabled */}
                {view3D && (
                  <div className="bg-neutral-700 rounded-lg p-3">
                    <h4 className="text-sm font-medium text-neutral-200 mb-2">3D Controls</h4>
                    
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
                          max={5.0}
                          step={0.1}
                          value={isNaN(staticRenderSettings.resnormSpread) ? 1.0 : staticRenderSettings.resnormSpread}
                          onChange={e => onStaticRenderSettingsChange({
                            ...staticRenderSettings,
                            resnormSpread: Number(e.target.value)
                          })}
                          className="w-full h-2 bg-neutral-600 rounded-lg appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-purple-500"
                        />
                      </div>

                      {/* 3D Rotation Center Control */}
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-neutral-400">Resnorm Center</span>
                        <button
                          onClick={() => onStaticRenderSettingsChange({
                            ...staticRenderSettings,
                            useResnormCenter: !staticRenderSettings.useResnormCenter
                          })}
                          className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                            staticRenderSettings.useResnormCenter ? 'bg-purple-600' : 'bg-neutral-600'
                          }`}
                        >
                          <span
                            className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform ${
                              staticRenderSettings.useResnormCenter ? 'translate-x-5' : 'translate-x-1'
                            }`}
                          />
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {/* Model Filtering */}
                <div className="bg-neutral-700 rounded-lg p-3">
                  <h4 className="text-sm font-medium text-neutral-200 mb-3">Model Filtering</h4>
                  
                  {/* Group Portion */}
                  <div className="mb-3">
                    <div className="flex items-center justify-between mb-2">
                      <label className="text-xs text-neutral-400">Group Portion</label>
                      <div className="text-right text-xs text-neutral-400">
                        <span className="font-mono">{logModelPercent.toFixed(logModelPercent < 1 ? 2 : 1)}%</span>
                        <span className="text-neutral-500 ml-2">({visibleModels.length.toLocaleString()} models)</span>
                      </div>
                    </div>
                    
                    <input
                      type="range"
                      min="1"
                      max="100"
                      value={isNaN(modelSelectionSlider) ? 50 : modelSelectionSlider}
                      onChange={(e) => {
                        const newSliderValue = Number(e.target.value);
                        setModelSelectionSlider(newSliderValue);
                        
                        // Convert slider value to groupPortion and sync with parent
                        const minLog = Math.log10(0.01);
                        const maxLog = Math.log10(100);
                        const logValue = minLog + (newSliderValue - 1) / 99 * (maxLog - minLog);
                        const logPercent = Math.pow(10, logValue);
                        const groupPortionValue = logPercent / 100; // Convert back to 0-1 range
                        
                        onGroupPortionChange(groupPortionValue);
                      }}
                      className="w-full h-2 bg-neutral-600 rounded-lg appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-blue-500 hover:[&::-webkit-slider-thumb]:bg-blue-400 transition-colors"
                    />
                    
                    {/* Navigation progress integrated into slider info */}
                    <div className="flex justify-between text-xs text-neutral-500 mt-1">
                      <span>0.01%</span>
                      {allFilteredModels.length > navigationWindowSize ? (
                        <span className="text-blue-400 font-mono">
                          Nav: {navigationOffset + 1}-{Math.min(navigationOffset + navigationWindowSize, allFilteredModels.length)} of {allFilteredModels.length.toLocaleString()}
                        </span>
                      ) : (
                        <span>100%</span>
                      )}
                    </div>

                    {/* Minimal navigation progress bar */}
                    {allFilteredModels.length > navigationWindowSize && (
                      <div className="relative mt-1 h-0.5 bg-neutral-700 rounded-full overflow-hidden">
                        <div
                          className="absolute h-full bg-blue-400 transition-all duration-300 ease-out"
                          style={{
                            left: `${(navigationOffset / allFilteredModels.length) * 100}%`,
                            width: `${Math.min((navigationWindowSize / allFilteredModels.length) * 100, 100 - (navigationOffset / allFilteredModels.length) * 100)}%`
                          }}
                        />
                      </div>
                    )}
                  </div>

                  {/* Resnorm Range */}
                  <div>
                    <label className="block text-xs text-neutral-400 mb-2">Resnorm Range</label>
                    <div className="bg-neutral-800/50 rounded border border-neutral-600">
                      <ResnormDisplay
                        models={allFilteredModels}
                        visibleModels={visibleModels}
                        navigationOffset={navigationOffset}
                        onNavigationOffsetChange={setNavigationOffset}
                        navigationWindowSize={navigationWindowSize}
                        onResnormRangeChange={handleResnormRangeChange}
                        currentResnorm={currentResnorm}
                        onResnormSelect={handleResnormSelect}
                        taggedModels={localTaggedModels}
                        tagColors={tagColors}
                      />
                    </div>
                  </div>
                </div>

                {/* Resnorm Method Display (SSR Only) */}
                <div className="bg-neutral-700 rounded-lg p-3">
                  <h4 className="text-sm font-medium text-neutral-200 mb-2">Resnorm Method</h4>
                  <div className="px-2 py-1.5 text-sm bg-neutral-800 border border-neutral-600 rounded text-neutral-200">
                    SSR - Sum of Squared Residuals
                  </div>
                  <p className="text-xs text-neutral-400 mt-2">
                    Fixed to SSR method for consistent impedance analysis
                  </p>
                </div>

              </div> {/* End scrollable container */}
            </div> {/* End right sidebar */}
          </div>
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