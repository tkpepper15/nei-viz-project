import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { SpiderPlot } from './visualizations/SpiderPlot';
import { SpiderPlot3D } from './visualizations/SpiderPlot3D';
import { NyquistPlot } from './visualizations/NyquistPlot';
import { ModelSnapshot, ResnormGroup } from './types';
import { GridParameterArrays } from './types';
import { CircuitParameters } from './types/parameters';
// import { ExportModal } from './controls/ExportModal'; // Removed export functionality
import { StaticRenderSettings } from './controls/StaticRenderControls';
import { PerformanceSettings } from './controls/PerformanceControls';

interface VisualizationSettings {
  groupPortion: number;
  selectedOpacityGroups: number[];
  visualizationType: 'spider2d' | 'spider3d' | 'nyquist';
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
  
  // Performance settings (unused in current simplified layout)
  performanceSettings: PerformanceSettings;
  
  // Frequency configuration for Nyquist plot
  numPoints: number;
}

export const VisualizerTab: React.FC<VisualizerTabProps> = ({
  resnormGroups,
  hiddenGroups,
  opacityLevel,
  referenceModelId: _referenceModelId, // eslint-disable-line @typescript-eslint/no-unused-vars
  gridSize,
  onGridValuesGenerated: _onGridValuesGenerated, // eslint-disable-line @typescript-eslint/no-unused-vars
  opacityExponent,
  onOpacityExponentChange,
  // Circuit parameters from parent
  userReferenceParams,
  // View control props
  showLabels,
  // Visualization settings callback
  onVisualizationSettingsChange,
  // Static render settings for visualization consistency
  staticRenderSettings,
  onStaticRenderSettingsChange,
  // Circuit parameter management
  groundTruthParams,
  // Performance settings (unused in current simplified layout)
  performanceSettings: _performanceSettings, // eslint-disable-line @typescript-eslint/no-unused-vars
  
  // Frequency configuration for Nyquist plot
  numPoints
}) => {

  // Removed old control states - now using compact sidebar design
  
  // Use static render settings instead of local state
  const chromaEnabled = staticRenderSettings.chromaEnabled;
  const showGroundTruth = staticRenderSettings.showGroundTruth;
  // Use ground truth parameters from toolbox (userReferenceParams) or direct groundTruthParams
  const effectiveGroundTruthParams = userReferenceParams || groundTruthParams;
  // Use visualization settings from static render settings
  const visualizationType = staticRenderSettings.visualizationType;
  const view3D = staticRenderSettings.visualizationType === 'spider3d';
  const setVisualizationType = (type: 'spider2d' | 'spider3d' | 'nyquist') => {
    onStaticRenderSettingsChange({
      ...staticRenderSettings,
      visualizationType: type
    });
  };
  
  // Nyquist plot log scale options
  const [nyquistLogScaleReal, setNyquistLogScaleReal] = useState(false);
  const [nyquistLogScaleImaginary, setNyquistLogScaleImaginary] = useState(false);
  
  // Removed old CollapsibleSection and formatTickValue - using compact sidebar design
  
  // Add view control state - use props if provided, otherwise internal state
  // const [internalZoomLevel, setInternalZoomLevel] = useState(1.0); // Removed unused
  
  // Add state for selected groups for opacity contrast (multi-select)
  // Use selected opacity groups from static render settings with safety check
  const selectedOpacityGroups = useMemo(() => {
    return Array.isArray(staticRenderSettings.selectedOpacityGroups) 
      ? staticRenderSettings.selectedOpacityGroups 
      : [0]; // Default to showing "Excellent" group
  }, [staticRenderSettings.selectedOpacityGroups]);
  const setSelectedOpacityGroups = useCallback((groups: number[]) => {
    onStaticRenderSettingsChange({
      ...staticRenderSettings,
      selectedOpacityGroups: groups
    });
  }, [staticRenderSettings, onStaticRenderSettingsChange]);
  
  // Add state for group portion filtering
  // NOTE: These visualization settings are applied after computation, not during.
  // To optimize computation based on these settings, they would need to be passed
  // to the computation pipeline and integrated into the web workers.
  // Use group portion from static render settings
  const groupPortion = staticRenderSettings.groupPortion;
  const setGroupPortion = (value: number) => {
    onStaticRenderSettingsChange({
      ...staticRenderSettings,
      groupPortion: value
    });
  };
  
  // Rendering mode state with automatic selection (simplified since rendering engine controls removed)
  const [renderingMode] = useState<'auto' | 'interactive' | 'tile'>('auto');
  const [forceMode] = useState<boolean>(false);
  
  // Use external props if provided, otherwise use internal state
  // const currentZoomLevel = zoomLevel !== undefined ? zoomLevel : internalZoomLevel; // Unused after UI cleanup
  const currentShowLabels = showLabels !== undefined ? showLabels : staticRenderSettings.includeLabels;

  // Notify parent of visualization settings changes
  useEffect(() => {
    if (onVisualizationSettingsChange) {
      const settings: VisualizationSettings = {
        groupPortion,
        selectedOpacityGroups,
        visualizationType
      };
      onVisualizationSettingsChange(settings);
    }
  }, [groupPortion, selectedOpacityGroups, visualizationType, onVisualizationSettingsChange]);

  // Ground truth parameters now come from userReferenceParams (toolbox)

  // Zoom level removed after UI cleanup

  // Reset selected groups if they don't exist or set default
  useEffect(() => {
    if (!resnormGroups || resnormGroups.length === 0) {
      setSelectedOpacityGroups([]);
      return;
    }
    
    // Check if we need to update the selected groups
    const currentSelected = selectedOpacityGroups;
    
    if (resnormGroups.length > 0 && currentSelected.length === 0) {
      // Default to "Excellent" group (index 0) when groups are available but none selected
      setSelectedOpacityGroups([0]);
      return;
    }
    
    // Filter out groups that no longer exist
    const validGroups = currentSelected.filter(groupIndex => groupIndex < resnormGroups.length);
    
    if (validGroups.length !== currentSelected.length) {
      setSelectedOpacityGroups(validGroups.length > 0 ? validGroups : [0]);
    }
  }, [resnormGroups, setSelectedOpacityGroups, selectedOpacityGroups]);

  // Automatic rendering mode selection based on dataset size
  const actualRenderingMode = useMemo(() => {
    if (renderingMode !== 'auto' || forceMode) {
      return renderingMode === 'auto' ? 'interactive' : renderingMode;
    }

    const totalModels = (resnormGroups || [])
      .filter((_, index) => !hiddenGroups.includes(index))
      .reduce((sum, group) => sum + group.items.length, 0);

    // Automatic selection thresholds
    if (totalModels > 50000) {
      return 'tile'; // Use tile-based for large datasets
    } else if (totalModels > 10000) {
      return 'tile'; // Use tile-based for medium-large datasets
    } else {
      return 'interactive'; // Use interactive for smaller datasets
    }
  }, [renderingMode, forceMode, resnormGroups, hiddenGroups]);

  // Performance metrics for display - removed unused after UI cleanup
  // Performance metrics removed after UI cleanup

  // Get all models from non-hidden groups for data preservation
  const allAvailableModels: ModelSnapshot[] = (resnormGroups || [])
    .filter((_, index) => !hiddenGroups.includes(index))
    .flatMap(group => group.items);

  // Get visible models based on selectedOpacityGroups and groupPortion for display
  const visibleModels: ModelSnapshot[] = (() => {
    // If no groups are selected for opacity, show NOTHING (not all models)
    if (selectedOpacityGroups.length === 0) {
      return [];
    }
    
    // Apply both group selection and group portion filtering
    const filteredModels: ModelSnapshot[] = [];
    
    (resnormGroups || [])
      .filter((_, index) => selectedOpacityGroups.includes(index) && !hiddenGroups.includes(index))
      .forEach(group => {
        // Apply groupPortion to each selected group
        const keepCount = Math.max(1, Math.floor(group.items.length * groupPortion));
        // Sort by resnorm (ascending = best fits first) and take the top portion
        const sortedItems = [...group.items].sort((a, b) => (a.resnorm || 0) - (b.resnorm || 0));
        filteredModels.push(...sortedItems.slice(0, keepCount));
      });
    
    return filteredModels;
  })();

  // Enhanced worker-assisted rendering for extremely large datasets
  const [isWorkerRendering, setIsWorkerRendering] = useState(false);
  const [workerProgress, setWorkerProgress] = useState(0);
  const [workerImageUrl, setWorkerImageUrl] = useState<string | null>(null);

  // Determine if we should use worker-assisted rendering
  const shouldUseWorkerRendering = useMemo(() => {
    const WORKER_THRESHOLD = 100000; // Use workers for >100k models
    return visibleModels.length > WORKER_THRESHOLD && actualRenderingMode === 'tile';
  }, [visibleModels.length, actualRenderingMode]);

  // Worker-assisted rendering function
  const renderWithWorkers = useCallback(async () => {
    if (!shouldUseWorkerRendering || isWorkerRendering) return;

    setIsWorkerRendering(true);
    setWorkerProgress(0);
    setWorkerImageUrl(null);

    try {
      const { sharedWorkerStrategy } = await import('./utils/sharedWorkerStrategy');
      
      const config = sharedWorkerStrategy.getOptimalConfiguration(visibleModels.length, 'medium');
      const estimate = sharedWorkerStrategy.estimateProcessingTime(visibleModels.length, config);
      
      console.log(`🚀 [VisualizerTab] Worker rendering ${visibleModels.length} models | Est. time: ${estimate.estimatedSeconds.toFixed(1)}s`);

      // Use same rendering parameters as orchestrator
      const renderParams = {
        format: 'png' as const,
        resolution: '1920x1080',
        quality: 95,
        includeLabels: true,
        chromaEnabled: true,
        selectedGroups: selectedOpacityGroups,
        backgroundColor: 'transparent' as const,
        opacityFactor: opacityLevel,
        visualizationMode: 'color' as const,
        opacityIntensity: 1.0
      };

      const imageUrl = await sharedWorkerStrategy.processSpiderPlotVisualization(
        visibleModels,
        renderParams,
        config,
        {
          onProgress: (progress, message) => {
            setWorkerProgress(progress * 100);
            console.log(`🎨 [VisualizerTab] ${message} | Progress: ${(progress * 100).toFixed(1)}%`);
          },
          onComplete: () => {
            console.log(`🎉 [VisualizerTab] Worker rendering completed`);
          },
          onError: (error) => {
            console.error(`❌ [VisualizerTab] Worker rendering error: ${error}`);
          }
        }
      );

      setWorkerImageUrl(imageUrl);
    } catch (error) {
      console.error('Worker rendering failed:', error);
    } finally {
      setIsWorkerRendering(false);
    }
  }, [shouldUseWorkerRendering, isWorkerRendering, visibleModels, selectedOpacityGroups, opacityLevel]);

  // Auto-trigger worker rendering for very large datasets
  useEffect(() => {
    if (shouldUseWorkerRendering && !isWorkerRendering && !workerImageUrl) {
      const timeout = setTimeout(() => {
        renderWithWorkers();
      }, 1000); // Delay to avoid immediate triggering on data changes
      
      return () => clearTimeout(timeout);
    }
  }, [shouldUseWorkerRendering, isWorkerRendering, workerImageUrl, renderWithWorkers]);

  // Apply opacity strategy to visible models
  const visibleModelsWithOpacity: ModelSnapshot[] = useMemo(() => {
    if (!visibleModels.length || !resnormGroups) return visibleModels;
    
    // Since filtering is now done at the computation level, we just need to apply opacity
    // Group models by their resnorm groups for opacity calculation
    const groupedModels: { [key: number]: ModelSnapshot[] } = {};
    
    visibleModels.forEach(model => {
      // Find which group this model belongs to
      const groupIndex = resnormGroups.findIndex(group => 
        group.items.some(item => item.id === model.id)
      );
      
      if (groupIndex !== -1) {
        if (!groupedModels[groupIndex]) groupedModels[groupIndex] = [];
        groupedModels[groupIndex].push(model);
      }
    });
    
    // Apply mathematical opacity strategy to each group
    const processedModels: ModelSnapshot[] = [];
    
    Object.keys(groupedModels).forEach(groupKey => {
      const groupIndex = parseInt(groupKey);
      const groupModels = groupedModels[groupIndex];
      
      if (!groupModels || groupModels.length === 0) return;
      
      // Extract resnorm values for opacity calculation
      const groupResnorms = groupModels.map(m => m.resnorm!).filter(r => r !== undefined);
      if (groupResnorms.length === 0) return;
      
      const maxR = Math.max(...groupResnorms);
      const minR = Math.min(...groupResnorms);
      
      // Apply mathematical opacity strategy within this group
      groupModels.forEach(model => {
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
        
        processedModels.push({
          ...model,
          opacity
        });
      });
    });
    
    return processedModels;
  }, [visibleModels, resnormGroups, opacityExponent]);

  // Check if we have actual computed results to show
  const hasComputedResults = resnormGroups && resnormGroups.length > 0 && allAvailableModels.length > 0;


  return (
    <div className="h-full flex flex-col bg-neutral-900">
      {hasComputedResults ? (
        <>
          {/* Top Toolbar - Compact and Clean */}
          <div className="flex items-center justify-between px-4 py-2 bg-neutral-800 border-b border-neutral-700 flex-shrink-0">
            <div className="flex items-center gap-4">
              <h3 className="text-sm font-semibold text-neutral-200">Visualizer</h3>
              
              {/* Visualization Type Selection */}
              <div className="flex bg-neutral-900 rounded-md overflow-hidden border border-neutral-600">
                <button
                  onClick={() => setVisualizationType('spider2d')}
                  className={`px-3 py-1.5 text-xs font-medium transition-all duration-200 ${
                    visualizationType === 'spider2d'
                      ? 'bg-blue-600 text-white'
                      : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-700'
                  }`}
                >
                  Spider 2D
                </button>
                <button
                  onClick={() => setVisualizationType('spider3d')}
                  className={`px-3 py-1.5 text-xs font-medium transition-all duration-200 ${
                    visualizationType === 'spider3d'
                      ? 'bg-purple-600 text-white'
                      : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-700'
                  }`}
                >
                  Spider 3D
                </button>
                <button
                  onClick={() => setVisualizationType('nyquist')}
                  className={`px-3 py-1.5 text-xs font-medium transition-all duration-200 ${
                    visualizationType === 'nyquist'
                      ? 'bg-green-600 text-white'
                      : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-700'
                  }`}
                >
                  Nyquist
                </button>
              </div>
            </div>

            {/* Top Right Controls */}
            <div className="flex items-center gap-3">
              {/* Ground Truth Toggle */}
              <div className="flex items-center gap-2">
                <span className="text-xs text-neutral-400">Ground Truth</span>
                <button
                  onClick={() => onStaticRenderSettingsChange({
                    ...staticRenderSettings,
                    showGroundTruth: !showGroundTruth
                  })}
                  className={`relative inline-flex h-4 w-7 items-center rounded-full transition-colors ${
                    showGroundTruth ? 'bg-green-600' : 'bg-neutral-600'
                  }`}
                >
                  <span
                    className={`inline-block h-2 w-2 transform rounded-full bg-white transition-transform ${
                      showGroundTruth ? 'translate-x-4' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>

              {/* Models Count */}
              <div className="text-xs text-neutral-400">
                {visibleModels.length.toLocaleString()} models
              </div>
            </div>
          </div>

          {/* Main Content Area */}
          <div className="flex-1 flex overflow-hidden">
            {/* Primary Visualization Area */}
            <div className="flex-1 relative bg-black">
              {(visualizationType === 'spider2d' || visualizationType === 'spider3d') ? (
                /* Spider Visualization */
                <>
                  {shouldUseWorkerRendering && workerImageUrl ? (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <img 
                        src={workerImageUrl} 
                        alt="Worker-rendered spider plot"
                        className="max-w-full max-h-full object-contain"
                        style={{ background: 'transparent' }}
                      />
                    </div>
                  ) : shouldUseWorkerRendering && isWorkerRendering ? (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-neutral-300">
                      <div className="text-lg font-medium mb-4">Rendering with Web Workers...</div>
                      <div className="w-64 bg-neutral-700 rounded-full h-2 mb-2">
                        <div 
                          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${workerProgress}%` }}
                        ></div>
                      </div>
                      <div className="text-sm">{workerProgress.toFixed(1)}%</div>
                      <div className="text-xs text-neutral-400 mt-2">
                        Processing {visibleModels.length.toLocaleString()} models across multiple workers
                      </div>
                    </div>
                  ) : view3D ? (
                    <div className="w-full h-full relative overflow-hidden">
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
                        chromaEnabled={chromaEnabled}
                        width={1920} // Use large fixed width for responsiveness
                        height={1080} // Use large fixed height for responsiveness
                        showControls={true}
                        gridSize={gridSize}
                        resnormScale={staticRenderSettings.resnormScale}
                      />
                    </div>
                  ) : (
                    <SpiderPlot
                      meshItems={visibleModelsWithOpacity}
                      opacityFactor={opacityLevel}
                      maxPolygons={actualRenderingMode === 'tile' ? 1000000 : 100000}
                      visualizationMode={chromaEnabled ? 'color' : 'opacity'}
                      gridSize={gridSize}
                      includeLabels={currentShowLabels}
                      backgroundColor="transparent"
                      showGroundTruth={showGroundTruth}
                      groundTruthParams={effectiveGroundTruthParams}
                    />
                  )}
                </>
              ) : (
                /* Nyquist Plot Visualization */
                <div className="absolute inset-0 flex items-center justify-center">
                  <NyquistPlot
                    groundTruthParams={effectiveGroundTruthParams}
                    resnormGroups={resnormGroups}
                    visibleModels={allAvailableModels}
                    width={window?.innerWidth ? Math.floor(window.innerWidth * 0.7) : 900}
                    height={window?.innerHeight ? Math.floor(window.innerHeight * 0.7) : 600}
                    showGroundTruth={showGroundTruth}
                    chromaEnabled={chromaEnabled}
                    numPoints={numPoints}
                    hiddenGroups={hiddenGroups}
                    selectedOpacityGroups={selectedOpacityGroups}
                  />
                </div>
              )}
            </div>

            {/* Right Sidebar - Compact Control Panels */}
            <div className="w-72 bg-neutral-800 border-l border-neutral-700 flex flex-col">
              {/* Resnorm Spectrum Panel */}
              <div className="p-3 border-b border-neutral-700">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-xs font-semibold text-neutral-200">Resnorm Spectrum</h4>
                  <span className="text-xs text-neutral-400">Models: {allAvailableModels.length.toLocaleString()}</span>
                </div>
                
                {/* Performance Groups */}
                <div className="space-y-2">
                  {resnormGroups && resnormGroups.length > 0 ? (
                    <div className="grid grid-cols-2 gap-1">
                      {(() => {
                        const allResnorms = resnormGroups.flatMap(group => 
                          group.items.map(item => item.resnorm).filter(r => r !== undefined)
                        ).sort((a, b) => a - b);
                        
                        const percentileGroups = [
                          { name: 'Excellent', index: 0, color: '#22c55e', range: { min: allResnorms[0], max: allResnorms[Math.floor(allResnorms.length * 0.25)] } },
                          { name: 'Good', index: 1, color: '#f59e0b', range: { min: allResnorms[Math.floor(allResnorms.length * 0.25)], max: allResnorms[Math.floor(allResnorms.length * 0.50)] } },
                          { name: 'Fair', index: 2, color: '#f97316', range: { min: allResnorms[Math.floor(allResnorms.length * 0.50)], max: allResnorms[Math.floor(allResnorms.length * 0.75)] } },
                          { name: 'Poor', index: 3, color: '#ef4444', range: { min: allResnorms[Math.floor(allResnorms.length * 0.75)], max: allResnorms[allResnorms.length - 1]} }
                        ];
                        
                        return percentileGroups.map((group) => {
                          const isSelected = selectedOpacityGroups.includes(group.index);
                          const groupData = resnormGroups[group.index];
                          const modelCount = groupData ? groupData.items.length : 0;
                          
                          return (
                            <button
                              key={group.index}
                              onClick={() => {
                                if (isSelected) {
                                  setSelectedOpacityGroups(selectedOpacityGroups.filter(i => i !== group.index));
                                } else {
                                  setSelectedOpacityGroups([...selectedOpacityGroups, group.index]);
                                }
                              }}
                              className={`px-2 py-1.5 text-xs font-medium rounded transition-all ${
                                isSelected
                                  ? 'text-white shadow-md'
                                  : 'bg-neutral-700 text-neutral-300 hover:bg-neutral-600'
                              }`}
                              style={{ backgroundColor: isSelected ? group.color : undefined }}
                            >
                              <div className="text-center">
                                <div>{group.name}</div>
                                <div className="text-[10px] opacity-70">{modelCount}</div>
                              </div>
                            </button>
                          );
                        });
                      })()}
                    </div>
                  ) : (
                    <div className="text-xs text-neutral-500 text-center py-2">No data</div>
                  )}
                </div>

                {/* Clear All button */}
                {selectedOpacityGroups.length > 0 && (
                  <button
                    onClick={() => setSelectedOpacityGroups([])}
                    className="w-full mt-2 px-2 py-1 text-xs font-medium rounded transition-colors bg-neutral-600 text-neutral-300 hover:bg-neutral-500"
                  >
                    Clear All
                  </button>
                )}
              </div>

              {/* Opacity Controls Panel */}
              <div className="p-3 border-b border-neutral-700">
                <h4 className="text-xs font-semibold text-neutral-200 mb-2">Opacity Controls</h4>
                
                <div className="space-y-3">
                  {/* Opacity Exponent */}
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-neutral-400">Contrast</span>
                      <span className="text-xs text-neutral-200 font-mono">{opacityExponent.toFixed(1)}</span>
                    </div>
                    <input
                      type="range"
                      min={1}
                      max={8}
                      step={0.1}
                      value={opacityExponent}
                      onChange={e => onOpacityExponentChange(Number(e.target.value))}
                      className="w-full h-1.5 bg-neutral-700 rounded-lg appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-blue-600"
                    />
                  </div>

                  {/* Group Portion */}
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-neutral-400">Portion</span>
                      <span className="text-xs text-neutral-200 font-mono">{Math.round(groupPortion * 100)}%</span>
                    </div>
                    <input
                      type="range"
                      min={1}
                      max={100}
                      step={1}
                      value={Math.max(1, groupPortion * 100)}
                      onChange={e => setGroupPortion(Math.max(0.01, Number(e.target.value) / 100))}
                      className="w-full h-1.5 bg-neutral-700 rounded-lg appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-blue-600"
                    />
                  </div>

                  {/* 3D Scale Control - only show when 3D is enabled */}
                  {view3D && (
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs text-neutral-400">3D Scale</span>
                        <span className="text-xs text-neutral-200 font-mono">{staticRenderSettings.resnormScale.toFixed(1)}x</span>
                      </div>
                      <input
                        type="range"
                        min={0.1}
                        max={5.0}
                        step={0.1}
                        value={staticRenderSettings.resnormScale}
                        onChange={e => onStaticRenderSettingsChange({
                          ...staticRenderSettings,
                          resnormScale: Number(e.target.value)
                        })}
                        className="w-full h-1.5 bg-neutral-700 rounded-lg appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-purple-500"
                      />
                    </div>
                  )}
                </div>
              </div>

              {/* Navigation Controls Panel */}
              <div className="p-3 flex-1">
                <h4 className="text-xs font-semibold text-neutral-200 mb-2">Navigation</h4>
                
                <div className="grid grid-cols-3 gap-1">
                  <button className="px-2 py-1.5 text-xs bg-neutral-700 hover:bg-neutral-600 text-neutral-300 rounded transition-colors">
                    Reset
                  </button>
                  <button className="px-2 py-1.5 text-xs bg-neutral-700 hover:bg-neutral-600 text-neutral-300 rounded transition-colors">
                    Focus
                  </button>
                  <button className="px-2 py-1.5 text-xs bg-neutral-700 hover:bg-neutral-600 text-neutral-300 rounded transition-colors">
                    Navigate
                  </button>
                </div>

                {/* Nyquist Plot Specific Controls */}
                {visualizationType === 'nyquist' && (
                  <div className="mt-4 space-y-2">
                    <h5 className="text-xs font-medium text-neutral-300">Axis Scaling</h5>
                    
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-neutral-400">Real Log</span>
                      <button
                        onClick={() => setNyquistLogScaleReal(!nyquistLogScaleReal)}
                        className={`relative inline-flex h-3 w-6 items-center rounded-full transition-colors ${
                          nyquistLogScaleReal ? 'bg-blue-600' : 'bg-neutral-600'
                        }`}
                      >
                        <span
                          className={`inline-block h-2 w-2 transform rounded-full bg-white transition-transform ${
                            nyquistLogScaleReal ? 'translate-x-3' : 'translate-x-0.5'
                          }`}
                        />
                      </button>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-neutral-400">Imaginary Log</span>
                      <button
                        onClick={() => setNyquistLogScaleImaginary(!nyquistLogScaleImaginary)}
                        className={`relative inline-flex h-3 w-6 items-center rounded-full transition-colors ${
                          nyquistLogScaleImaginary ? 'bg-blue-600' : 'bg-neutral-600'
                        }`}
                      >
                        <span
                          className={`inline-block h-2 w-2 transform rounded-full bg-white transition-transform ${
                            nyquistLogScaleImaginary ? 'translate-x-3' : 'translate-x-0.5'
                          }`}
                        />
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
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