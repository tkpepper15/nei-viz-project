import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { SpiderPlot } from './visualizations/SpiderPlot';
import { SpiderPlot3D } from './visualizations/SpiderPlot3D';
import { NyquistPlot } from './visualizations/NyquistPlot';
import { ModelSnapshot, ResnormGroup } from './types';
import { GridParameterArrays } from './types';
import { CircuitParameters } from './types/parameters';
// import { ExportModal } from './controls/ExportModal'; // Removed export functionality
import { StaticRenderSettings } from './controls/StaticRenderControls';
import { ParamSlider } from './ParamSlider';
import { generateLogSpace } from './utils/parameter-space';
import { PARAMETER_RANGES, faradToMicroFarad } from './types/parameters';
import { SaveProfileModal } from './controls/SaveProfileModal';
import { PerformanceControls, PerformanceSettings } from './controls/PerformanceControls';

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
  
  // Grid computation props
  setGridSize: (size: number) => void;
  minFreq: number;
  setMinFreq: (freq: number) => void;
  maxFreq: number;
  setMaxFreq: (freq: number) => void;
  numPoints: number;
  setNumPoints: (points: number) => void;
  updateFrequencies: (min: number, max: number, points: number) => void;
  updateStatusMessage: (message: string) => void;
  parameterChanged: boolean;
  setParameterChanged: (changed: boolean) => void;
  handleComputeRegressionMesh: () => void;
  isComputingGrid: boolean;
  onClearResults: () => void;
  hasGridResults: boolean;
  
  // Circuit parameter management
  groundTruthParams: CircuitParameters;
  setGroundTruthParams: (params: CircuitParameters | ((prev: CircuitParameters) => CircuitParameters)) => void;
  createReferenceModel: () => ModelSnapshot;
  setReferenceModel: (model: ModelSnapshot | null) => void;
  
  // Save profile functionality
  onSaveProfile: (name: string, description?: string) => void;
  
  // Performance settings
  performanceSettings: PerformanceSettings;
  setPerformanceSettings: (settings: PerformanceSettings) => void;
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
  // Grid computation props
  setGridSize,
  minFreq: _minFreq, // eslint-disable-line @typescript-eslint/no-unused-vars
  setMinFreq: _setMinFreq, // eslint-disable-line @typescript-eslint/no-unused-vars
  maxFreq: _maxFreq, // eslint-disable-line @typescript-eslint/no-unused-vars
  setMaxFreq: _setMaxFreq, // eslint-disable-line @typescript-eslint/no-unused-vars
  numPoints: _numPoints, // eslint-disable-line @typescript-eslint/no-unused-vars
  setNumPoints: _setNumPoints, // eslint-disable-line @typescript-eslint/no-unused-vars
  updateFrequencies: _updateFrequencies, // eslint-disable-line @typescript-eslint/no-unused-vars
  updateStatusMessage,
  parameterChanged,
  setParameterChanged: _setParameterChanged, // eslint-disable-line @typescript-eslint/no-unused-vars
  handleComputeRegressionMesh,
  isComputingGrid,
  onClearResults,
  hasGridResults,
  // Circuit parameter management
  groundTruthParams,
  setGroundTruthParams,
  createReferenceModel,
  setReferenceModel,
  // Save profile functionality
  onSaveProfile,
  // Performance settings
  performanceSettings,
  setPerformanceSettings
}) => {

  // const [isExportModalOpen, setIsExportModalOpen] = useState(false); // Removed export modal
  // const [showDistribution, setShowDistribution] = useState(false); // Unused after UI cleanup
  const [saveModalOpen, setSaveModalOpen] = useState(false);
  const [showGridControls, setShowGridControls] = useState(true);
  const [showCircuitParams, setShowCircuitParams] = useState(false);
  const [showPerformanceControls, setShowPerformanceControls] = useState(false); // eslint-disable-line @typescript-eslint/no-unused-vars
  
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
  
  // Collapsible section component for toolbox controls
  const CollapsibleSection: React.FC<{title: string; isOpen: boolean; toggleOpen: () => void; children: React.ReactNode; isFirst?: boolean}> = ({ 
    title, 
    isOpen, 
    toggleOpen, 
    children,
    isFirst = false
  }) => {
    return (
      <div className="border-b border-neutral-700 last:border-b-0">
        <button 
          className="w-full px-4 py-3 bg-neutral-800 text-neutral-300 text-sm font-bold flex items-center justify-between hover:bg-neutral-750 transition-colors"
          onClick={toggleOpen}
          style={isFirst ? { borderRadius: '0px 0px 0px 0px' } : {}}
        >
          <span className="font-bold">{title}</span>
          <svg 
            className={`w-4 h-4 transition-transform ${isOpen ? 'transform rotate-180' : ''}`} 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        <div className={`px-4 py-3 bg-neutral-800 transition-all ${isOpen ? 'block' : 'hidden'}`}>
          {children}
        </div>
      </div>
    );
  };
  
  // Helper function for formatting tick values
  const formatTickValue = (v: number, unit: string, digits = 2) =>
    unit === 'Ω' ? Number(v).toPrecision(3) : Number(v).toFixed(digits);
  
  // Add view control state - use props if provided, otherwise internal state
  // const [internalZoomLevel, setInternalZoomLevel] = useState(1.0); // Removed unused
  
  // Add state for selected groups for opacity contrast (multi-select)
  // Use selected opacity groups from static render settings with safety check
  const selectedOpacityGroups = useMemo(() => {
    return Array.isArray(staticRenderSettings.selectedOpacityGroups) 
      ? staticRenderSettings.selectedOpacityGroups 
      : [0]; // Default to showing "Excellent" group
  }, [staticRenderSettings.selectedOpacityGroups]);
  const setSelectedOpacityGroups = (groups: number[]) => {
    onStaticRenderSettingsChange({
      ...staticRenderSettings,
      selectedOpacityGroups: groups
    });
  };
  
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

  const controlsContent = (
    <div className="bg-neutral-800 rounded border border-neutral-700 min-h-[600px] flex flex-col">
      {/* Settings Header */}
      <div className="p-3 border-b border-neutral-600 flex-shrink-0">
        <h3 className="text-sm font-semibold text-neutral-200 mb-3">Settings</h3>
        
        

      </div>

      {/* Controls Content */}
      <div className="flex-1 min-h-0 overflow-y-auto custom-scrollbar">
        <div className="p-3 space-y-4"
             style={{ scrollbarWidth: 'thin', scrollbarColor: '#4B5563 #1F2937' }}>
        
        {/* Group Opacity Controls */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-medium text-neutral-200">Group Opacity</h4>
            <span className="text-xs text-neutral-400">
              {hasComputedResults ? `${visibleModels.length} models visible` : '0 models'}
            </span>
          </div>

          {/* Opacity System Info - removed after UI cleanup */}

          {/* Current Settings Display */}
          <div className="flex items-center justify-between text-xs">
            <span className="text-neutral-400">
              {selectedOpacityGroups.length === 0 ? (
                <span className="text-amber-400 flex items-center gap-1">
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                  </svg>
                  No groups selected
                </span>
              ) : (
                `${visibleModels.length} models visible`
              )}
            </span>
            <span className="font-mono text-neutral-200 bg-neutral-700 px-2 py-1 rounded text-xs">
              {selectedOpacityGroups.length > 0 ? '5-100%' : 'N/A'}
            </span>
          </div>

        {/* Show groups section - Percentile Tags */}
        <div className="space-y-3">
          <label className="text-xs font-medium text-neutral-300">Show Performance Groups:</label>
          
          {resnormGroups && resnormGroups.length > 0 ? (
            <div className="space-y-2">
              {/* Calculate resnorm ranges for each group */}
              {(() => {
                // Get all resnorm values and calculate percentiles
                const allResnorms = resnormGroups.flatMap(group => 
                  group.items.map(item => item.resnorm).filter(r => r !== undefined)
                ).sort((a, b) => a - b);
                
                const percentileGroups = [
                  { 
                    name: 'Excellent (0-25%)', 
                    index: 0, 
                    color: '#22c55e',
                    range: allResnorms.length > 0 ? {
                      min: allResnorms[0],
                      max: allResnorms[Math.floor(allResnorms.length * 0.25)]
                    } : { min: 0, max: 0 }
                  },
                  { 
                    name: 'Good (25-50%)', 
                    index: 1, 
                    color: '#f59e0b',
                    range: allResnorms.length > 0 ? {
                      min: allResnorms[Math.floor(allResnorms.length * 0.25)],
                      max: allResnorms[Math.floor(allResnorms.length * 0.50)]
                    } : { min: 0, max: 0 }
                  },
                  { 
                    name: 'Fair (50-75%)', 
                    index: 2, 
                    color: '#f97316',
                    range: allResnorms.length > 0 ? {
                      min: allResnorms[Math.floor(allResnorms.length * 0.50)],
                      max: allResnorms[Math.floor(allResnorms.length * 0.75)]
                    } : { min: 0, max: 0 }
                  },
                  { 
                    name: 'Poor (75-100%)', 
                    index: 3, 
                    color: '#ef4444',
                    range: allResnorms.length > 0 ? {
                      min: allResnorms[Math.floor(allResnorms.length * 0.75)],
                      max: allResnorms[allResnorms.length - 1]
                    } : { min: 0, max: 0 }
                  }
                ];
                
                return (
                  <div className="flex flex-wrap gap-2">
                    {percentileGroups.map((group) => {
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
                          className={`px-3 py-2 text-xs font-medium rounded-full transition-all border ${
                            isSelected
                              ? 'text-white shadow-md transform scale-105'
                              : 'bg-neutral-700 text-neutral-300 hover:bg-neutral-600 hover:text-white border-neutral-600'
                          }`}
                          style={{
                            backgroundColor: isSelected ? group.color : undefined,
                            borderColor: isSelected ? group.color : undefined
                          }}
                          title={`Resnorm range: ${group.range.min.toExponential(2)} - ${group.range.max.toExponential(2)}`}
                        >
                          <div className="flex flex-col items-center">
                            <span>{group.name}</span>
                            <span className="text-[10px] opacity-70">
                              {group.range.min.toExponential(1)} - {group.range.max.toExponential(1)}
                            </span>
                            <span className="text-[9px] opacity-60">{modelCount} models</span>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                );
              })()}
              
              {/* Clear All button */}
              {selectedOpacityGroups.length > 0 && (
                <button
                  onClick={() => setSelectedOpacityGroups([])}
                  className="w-full px-3 py-1.5 text-xs font-medium rounded transition-colors bg-neutral-600 text-neutral-300 hover:bg-neutral-500"
                >
                  Clear All ({selectedOpacityGroups.length})
                </button>
              )}
            </div>
          ) : (
            <div className="text-xs text-neutral-500 italic">
              Run computation to enable group selection
            </div>
          )}
        </div>

        {/* Opacity Control */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-neutral-300">Opacity:</label>
            <span className="text-xs text-neutral-200 font-mono">{opacityExponent.toFixed(1)}</span>
          </div>
          <input
            type="range"
            min={1}
            max={8}
            step={0.1}
            value={opacityExponent}
            onChange={e => onOpacityExponentChange(Number(e.target.value))}
            className="w-full h-2 bg-neutral-700 rounded-lg appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-blue-600 [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:shadow-md hover:[&::-webkit-slider-thumb]:bg-blue-500"
          />
          <div className="flex justify-between text-[10px] text-neutral-400">
            <span>Linear</span>
            <span>Very Stark</span>
          </div>
        </div>

        {/* Group Portion Control - Logarithmic Percentage */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-neutral-300">Group Portion:</label>
            <span className="text-xs text-neutral-200 font-mono">
              {Math.round(groupPortion * 100)}% 
              {visibleModels.length > 0 && (
                <span className="text-neutral-400 ml-1">
                  ({Math.max(1, Math.floor(visibleModels.length * groupPortion))} models)
                </span>
              )}
            </span>
          </div>
          <input
            type="range"
            min={1}
            max={100}
            step={1}
            value={Math.max(1, groupPortion * 100)}
            onChange={e => setGroupPortion(Math.max(0.01, Number(e.target.value) / 100))}
            className="w-full h-2 bg-neutral-700 rounded-lg appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-blue-600 [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:shadow-md hover:[&::-webkit-slider-thumb]:bg-blue-500"
          />
          <div className="flex justify-between text-[10px] text-neutral-400">
            <span>Best 1%</span>
            <span>All Models (100%)</span>
          </div>
        </div>

          
          {/* Nyquist Plot Specific Controls */}
          {visualizationType === 'nyquist' && (
            <div className="space-y-3 p-4 bg-neutral-900/50 rounded-lg border border-neutral-700">
              <div className="text-sm font-medium text-neutral-200">Axis Scaling</div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-neutral-300">Real Axis Log Scale</span>
                <button
                  onClick={() => setNyquistLogScaleReal(!nyquistLogScaleReal)}
                  className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                    nyquistLogScaleReal ? 'bg-blue-600' : 'bg-neutral-600'
                  }`}
                >
                  <span
                    className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform ${
                      nyquistLogScaleReal ? 'translate-x-5' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-neutral-300">Imaginary Axis Log Scale</span>
                <button
                  onClick={() => setNyquistLogScaleImaginary(!nyquistLogScaleImaginary)}
                  className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                    nyquistLogScaleImaginary ? 'bg-blue-600' : 'bg-neutral-600'
                  }`}
                >
                  <span
                    className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform ${
                      nyquistLogScaleImaginary ? 'translate-x-5' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
            </div>
          )}

          {/* Spider Visualization Specific Controls */}
          {(visualizationType === 'spider2d' || visualizationType === 'spider3d') && (
            <div className="space-y-3 p-4 bg-neutral-900/50 rounded-lg border border-neutral-700">
              <div className="text-sm font-medium text-neutral-200">Spider Visualization Controls</div>

              {/* 3D Resnorm Scale Control - only show when 3D is enabled */}
              {view3D && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-sm text-neutral-300">3D Scale:</label>
                    <span className="text-xs text-neutral-400 font-mono">{staticRenderSettings.resnormScale.toFixed(1)}x</span>
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
                    className="w-full h-2 bg-neutral-700 rounded-lg appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-purple-500 [&::-webkit-slider-thumb]:cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] text-neutral-400">
                    <span>Compressed</span>
                    <span>Expanded</span>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Ground Truth Controls */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-medium text-neutral-200">Ground Truth Overlay</h4>
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
          
          {showGroundTruth && (
            <div className="space-y-3 p-3 bg-neutral-900/50 rounded-lg border border-neutral-700">
              <div className="text-xs text-neutral-400 space-y-1">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-[1px] border-b border-dashed border-white"></div>
                  <span>White dashed polygon showing ground truth parameters</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 rounded-full bg-white"></div>
                  <span>White circles indicate exact parameter values</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-sm"></div>
                  <span>Overlay helps compare model fits to reference</span>
                </div>
              </div>
              
              <div className="space-y-2">
                <label className="text-xs font-medium text-neutral-300">Ground Truth Values:</label>
                <div className="text-xs text-neutral-400 space-y-1">
                  {effectiveGroundTruthParams ? (
                    <>
                      <div>R shunt: {effectiveGroundTruthParams.Rsh} Ω</div>
                      <div>Ra: {effectiveGroundTruthParams.Ra} Ω, Ca: {(effectiveGroundTruthParams.Ca * 1e6).toFixed(1)} µF</div>
                      <div>Rb: {effectiveGroundTruthParams.Rb} Ω, Cb: {(effectiveGroundTruthParams.Cb * 1e6).toFixed(1)} µF</div>
                      <div className="text-neutral-500 text-[10px] mt-2">
                        ✓ Values controlled by Toolbox parameters
                      </div>
                    </>
                  ) : (
                    <div>No ground truth parameters set</div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Intelligent Worker Status */}
        {shouldUseWorkerRendering && (
          <div className="space-y-2 border-t border-neutral-700 pt-3">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-medium text-neutral-300">Worker System</h4>
              <div className="text-xs text-green-400 font-mono">12 Workers</div>
            </div>
            
            {isWorkerRendering ? (
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
                  <span className="text-xs text-blue-300">Processing {visibleModels.length.toLocaleString()} models</span>
                </div>
                <div className="bg-neutral-700 rounded-full h-1">
                  <div 
                    className="bg-blue-500 h-1 rounded-full transition-all duration-300"
                    style={{ width: `${workerProgress}%` }}
                  />
                </div>
                <div className="text-[10px] text-neutral-400 flex justify-between">
                  <span>12-Worker Parallel Processing</span>
                  <span>{workerProgress.toFixed(1)}%</span>
                </div>
              </div>
            ) : (
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-green-400" />
                  <span className="text-xs text-green-300">12 workers ready</span>
                </div>
                <div className="text-[10px] text-neutral-500">
                  Auto-activates for {(50000).toLocaleString()}+ models
                </div>
              </div>
            )}
          </div>
        )}
        
        {/* Toolbox Controls - Migrated from ToolboxComponent */}
        <div className="space-y-0 border-t border-neutral-700 pt-3">
          {/* Grid Computation Section */}
          <CollapsibleSection 
            title="Grid Computation" 
            isOpen={showGridControls} 
            toggleOpen={() => setShowGridControls(!showGridControls)}
            isFirst={true}
          >
            <div className="space-y-3 pt-1">
              <div className="flex items-center justify-between gap-2">
                <label htmlFor="gridSize" className="text-sm font-medium text-neutral-200 flex-1">
                  Points per parameter
                </label>
                <input
                  type="number"
                  id="gridSize"
                  value={gridSize}
                  onChange={(e) => {
                    const size = Math.max(2, Math.min(25, parseInt(e.target.value) || 2));
                    setGridSize(size);
                    const totalPoints = Math.pow(size, 5);
                    const warningMessage = size > 15 ? ` - This will take significant computation time!` : '';
                    updateStatusMessage(`Grid size set to ${size} (${totalPoints.toLocaleString()} total points to compute)${warningMessage}`);
                  }} 
                  min="2"
                  max="25"
                  className="w-14 p-1 border border-neutral-700 rounded text-xs text-center bg-neutral-800 text-neutral-200 flex-shrink-0"
                />
              </div>

              {/* Save/Compute Button Group */}
              <div className="flex gap-2 mt-3">
                <button
                  onClick={() => setSaveModalOpen(true)}
                  disabled={isComputingGrid}
                  className={`flex-1 py-2.5 rounded-lg font-medium text-white transition-colors ${
                    isComputingGrid
                      ? 'bg-neutral-600 cursor-not-allowed'
                      : 'bg-green-600 hover:bg-green-700'
                  }`}
                  title="Save current parameters and settings as a profile"
                >
                  <span className="flex items-center justify-center">
                    <svg className="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                    Save Profile
                  </span>
                </button>
                
                <button
                  onClick={handleComputeRegressionMesh}
                  disabled={isComputingGrid}
                  className={`flex-1 py-2.5 rounded-lg font-medium text-white transition-colors ${
                    isComputingGrid
                      ? 'bg-neutral-600 cursor-not-allowed'
                      : parameterChanged 
                        ? 'bg-blue-600 hover:bg-blue-700' 
                        : 'bg-blue-600 hover:bg-blue-700'
                  }`}
                  title="Compute grid and display results in playground"
                >
                  {isComputingGrid ? (
                    <span className="flex items-center justify-center">
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Computing...
                    </span>
                  ) : (
                    <span className="flex items-center justify-center">
                      <svg className="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                      {parameterChanged ? 'Recompute' : 'Compute'}
                    </span>
                  )}
                </button>
              </div>
              
              {/* Clear Memory Button */}
              {hasGridResults && (
                <div className="mt-2">
                  <button
                    onClick={onClearResults}
                    disabled={isComputingGrid}
                    className="w-full py-2 px-3 rounded-md text-red-300 border border-red-600/50 hover:bg-red-600/10 hover:border-red-500 disabled:opacity-50 disabled:cursor-not-allowed font-medium transition-all duration-200 text-xs"
                    title="Clear grid results from memory to free up space"
                  >
                    <span className="flex items-center justify-center gap-2">
                      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                      Clear Results from Memory
                    </span>
                  </button>
                </div>
              )}
            </div>
          </CollapsibleSection>

          {/* Performance Controls Section */}
          <div className="border-b border-neutral-700 last:border-b-0">
            <PerformanceControls
              settings={performanceSettings}
              onChange={setPerformanceSettings}
              gridSize={gridSize}
            />
          </div>

          {/* Circuit Parameters Section */}
          <CollapsibleSection 
            title="Circuit Parameters" 
            isOpen={showCircuitParams} 
            toggleOpen={() => setShowCircuitParams(!showCircuitParams)}
          >
            <div className="space-y-4 pt-2">
              {/* Rsh Parameter */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-neutral-200">R shunt (Ω)</span>
                  <div className="relative">
                    <input
                      type="number"
                      value={effectiveGroundTruthParams.Rsh.toFixed(1)}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value);
                        if (!isNaN(val)) {
                          const clampedVal = Math.max(10, Math.min(10000, val));
                          setGroundTruthParams(prev => ({ ...prev, Rsh: clampedVal }));
                          if (val < 10 || val > 10000) {
                            updateStatusMessage(`⚠️ Rsh value ${val.toFixed(1)} is out of range (10-10000 Ω), clamped to ${clampedVal.toFixed(1)} Ω`);
                          } else {
                            updateStatusMessage(`Rsh set to ${val.toFixed(1)} Ω`);
                          }
                          if (_referenceModelId === 'dynamic-reference') {
                            const updatedModel = createReferenceModel();
                            setReferenceModel(updatedModel);
                          }
                        }
                      }}
                      className={`w-16 bg-neutral-700 text-white text-xs px-2 py-1 rounded border text-center focus:outline-none ${
                        effectiveGroundTruthParams.Rsh < 10 || effectiveGroundTruthParams.Rsh > 10000 
                          ? 'border-yellow-500 focus:border-yellow-400' 
                          : 'border-neutral-600 focus:border-blue-500'
                      }`}
                      step="10"
                    />
                    {(effectiveGroundTruthParams.Rsh < 10 || effectiveGroundTruthParams.Rsh > 10000) && (
                      <div className="absolute -top-1 -right-1 w-2 h-2 bg-yellow-500 rounded-full animate-pulse" />
                    )}
                  </div>
                </div>
                <ParamSlider 
                  label="" 
                  value={effectiveGroundTruthParams.Rsh} 
                  min={10} 
                  max={10000} 
                  step={10}
                  unit="Ω" 
                  onChange={(val: number) => {
                    setGroundTruthParams(prev => ({ ...prev, Rsh: val }));
                    updateStatusMessage(`Rsh set to ${val.toFixed(1)} Ω`);
                    if (_referenceModelId === 'dynamic-reference') {
                      const updatedModel = createReferenceModel();
                      setReferenceModel(updatedModel);
                    }
                  }} 
                  ticks={generateLogSpace(PARAMETER_RANGES.Rsh.min, PARAMETER_RANGES.Rsh.max, gridSize).map((v, i) => ({ level: i + 1, value: v }))}
                  log={true}
                  tickLabels={generateLogSpace(PARAMETER_RANGES.Rsh.min, PARAMETER_RANGES.Rsh.max, gridSize).map(v => formatTickValue(v, 'Ω'))}
                  readOnlyRange={false}
                />
              </div>

              {/* Ra Parameter */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-neutral-200">Ra (Ω)</span>
                  <div className="relative">
                    <input
                      type="number"
                      value={effectiveGroundTruthParams.Ra.toFixed(0)}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value);
                        if (!isNaN(val)) {
                          const clampedVal = Math.max(10, Math.min(10000, val));
                          setGroundTruthParams(prev => ({ ...prev, Ra: clampedVal }));
                          if (val < 10 || val > 10000) {
                            updateStatusMessage(`⚠️ Ra value ${val.toFixed(0)} is out of range (10-10000 Ω), clamped to ${clampedVal.toFixed(0)} Ω`);
                          } else {
                            updateStatusMessage(`Ra set to ${val.toFixed(0)} Ω`);
                          }
                          if (_referenceModelId === 'dynamic-reference') {
                            const updatedModel = createReferenceModel();
                            setReferenceModel(updatedModel);
                          }
                        }
                      }}
                      className={`w-16 bg-neutral-700 text-white text-xs px-2 py-1 rounded border text-center focus:outline-none ${
                        effectiveGroundTruthParams.Ra < 10 || effectiveGroundTruthParams.Ra > 10000 
                          ? 'border-yellow-500 focus:border-yellow-400' 
                          : 'border-neutral-600 focus:border-blue-500'
                      }`}
                      step="10"
                    />
                    {(effectiveGroundTruthParams.Ra < 10 || effectiveGroundTruthParams.Ra > 10000) && (
                      <div className="absolute -top-1 -right-1 w-2 h-2 bg-yellow-500 rounded-full animate-pulse" />
                    )}
                  </div>
                </div>
                <ParamSlider 
                  label="" 
                  value={effectiveGroundTruthParams.Ra} 
                  min={10} 
                  max={10000} 
                  step={10}
                  unit="Ω" 
                  onChange={(val: number) => {
                    setGroundTruthParams(prev => ({ ...prev, Ra: val }));
                    updateStatusMessage(`Ra set to ${val.toFixed(0)} Ω`);
                    if (_referenceModelId === 'dynamic-reference') {
                      const updatedModel = createReferenceModel();
                      setReferenceModel(updatedModel);
                    }
                  }} 
                  ticks={generateLogSpace(PARAMETER_RANGES.Ra.min, PARAMETER_RANGES.Ra.max, gridSize).map((v, i) => ({ level: i + 1, value: v }))}
                  log={true}
                  tickLabels={generateLogSpace(PARAMETER_RANGES.Ra.min, PARAMETER_RANGES.Ra.max, gridSize).map(v => formatTickValue(v, 'Ω'))}
                  readOnlyRange={false}
                />
              </div>

              {/* Ca Parameter */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-neutral-200">Ca (μF)</span>
                  <div className="relative">
                    <input
                      type="number"
                      value={(effectiveGroundTruthParams.Ca * 1e6).toFixed(2)}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value);
                        if (!isNaN(val)) {
                          const clampedVal = Math.max(0.1, Math.min(50, val));
                          setGroundTruthParams(prev => ({ ...prev, Ca: clampedVal / 1e6 }));
                          if (val < 0.1 || val > 50) {
                            updateStatusMessage(`⚠️ Ca value ${val.toFixed(2)} is out of range (0.1-50 μF), clamped to ${clampedVal.toFixed(2)} μF`);
                          } else {
                            updateStatusMessage(`Ca set to ${val.toFixed(2)} μF`);
                          }
                          if (_referenceModelId === 'dynamic-reference') {
                            const updatedModel = createReferenceModel();
                            setReferenceModel(updatedModel);
                          }
                        }
                      }}
                      className={`w-16 bg-neutral-700 text-white text-xs px-2 py-1 rounded border text-center focus:outline-none ${
                        (effectiveGroundTruthParams.Ca * 1e6) < 0.1 || (effectiveGroundTruthParams.Ca * 1e6) > 50 
                          ? 'border-yellow-500 focus:border-yellow-400' 
                          : 'border-neutral-600 focus:border-blue-500'
                      }`}
                      step="0.1"
                    />
                    {((effectiveGroundTruthParams.Ca * 1e6) < 0.1 || (effectiveGroundTruthParams.Ca * 1e6) > 50) && (
                      <div className="absolute -top-1 -right-1 w-2 h-2 bg-yellow-500 rounded-full animate-pulse" />
                    )}
                  </div>
                </div>
                <ParamSlider 
                  label="" 
                  value={effectiveGroundTruthParams.Ca * 1e6} 
                  min={0.1} 
                  max={50} 
                  step={0.1}
                  unit="μF" 
                  onChange={(val: number) => {
                    setGroundTruthParams(prev => ({ ...prev, Ca: val / 1e6 }));
                    updateStatusMessage(`Ca set to ${val.toFixed(2)} μF`);
                    if (_referenceModelId === 'dynamic-reference') {
                      const updatedModel = createReferenceModel();
                      setReferenceModel(updatedModel);
                    }
                  }} 
                  ticks={generateLogSpace(faradToMicroFarad(PARAMETER_RANGES.Ca.min), faradToMicroFarad(PARAMETER_RANGES.Ca.max), gridSize).map((v, i) => ({ level: i + 1, value: v }))}
                  transformValue={(v) => v.toFixed(2)}
                  log={true}
                  tickLabels={generateLogSpace(faradToMicroFarad(PARAMETER_RANGES.Ca.min), faradToMicroFarad(PARAMETER_RANGES.Ca.max), gridSize).map(v => formatTickValue(v, 'μF', 2))}
                  readOnlyRange={false}
                />
              </div>

              {/* Rb Parameter */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-neutral-200">Rb (Ω)</span>
                  <div className="relative">
                    <input
                      type="number"
                      value={effectiveGroundTruthParams.Rb.toFixed(0)}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value);
                        if (!isNaN(val)) {
                          const clampedVal = Math.max(10, Math.min(10000, val));
                          setGroundTruthParams(prev => ({ ...prev, Rb: clampedVal }));
                          if (val < 10 || val > 10000) {
                            updateStatusMessage(`⚠️ Rb value ${val.toFixed(0)} is out of range (10-10000 Ω), clamped to ${clampedVal.toFixed(0)} Ω`);
                          } else {
                            updateStatusMessage(`Rb set to ${val.toFixed(0)} Ω`);
                          }
                          if (_referenceModelId === 'dynamic-reference') {
                            const updatedModel = createReferenceModel();
                            setReferenceModel(updatedModel);
                          }
                        }
                      }}
                      className={`w-16 bg-neutral-700 text-white text-xs px-2 py-1 rounded border text-center focus:outline-none ${
                        effectiveGroundTruthParams.Rb < 10 || effectiveGroundTruthParams.Rb > 10000 
                          ? 'border-yellow-500 focus:border-yellow-400' 
                          : 'border-neutral-600 focus:border-blue-500'
                      }`}
                      step="10"
                    />
                    {(effectiveGroundTruthParams.Rb < 10 || effectiveGroundTruthParams.Rb > 10000) && (
                      <div className="absolute -top-1 -right-1 w-2 h-2 bg-yellow-500 rounded-full animate-pulse" />
                    )}
                  </div>
                </div>
                <ParamSlider 
                  label="" 
                  value={effectiveGroundTruthParams.Rb} 
                  min={10} 
                  max={10000} 
                  step={10}
                  unit="Ω" 
                  onChange={(val: number) => {
                    setGroundTruthParams(prev => ({ ...prev, Rb: val }));
                    updateStatusMessage(`Rb set to ${val.toFixed(0)} Ω`);
                    if (_referenceModelId === 'dynamic-reference') {
                      const updatedModel = createReferenceModel();
                      setReferenceModel(updatedModel);
                    }
                  }} 
                  ticks={generateLogSpace(PARAMETER_RANGES.Rb.min, PARAMETER_RANGES.Rb.max, gridSize).map((v, i) => ({ level: i + 1, value: v }))}
                  log={true}
                  tickLabels={generateLogSpace(PARAMETER_RANGES.Rb.min, PARAMETER_RANGES.Rb.max, gridSize).map(v => formatTickValue(v, 'Ω'))}
                  readOnlyRange={false}
                />
              </div>

              {/* Cb Parameter */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-neutral-200">Cb (μF)</span>
                  <div className="relative">
                    <input
                      type="number"
                      value={(effectiveGroundTruthParams.Cb * 1e6).toFixed(2)}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value);
                        if (!isNaN(val)) {
                          const clampedVal = Math.max(0.1, Math.min(50, val));
                          setGroundTruthParams(prev => ({ ...prev, Cb: clampedVal / 1e6 }));
                          if (val < 0.1 || val > 50) {
                            updateStatusMessage(`⚠️ Cb value ${val.toFixed(2)} is out of range (0.1-50 μF), clamped to ${clampedVal.toFixed(2)} μF`);
                          } else {
                            updateStatusMessage(`Cb set to ${val.toFixed(2)} μF`);
                          }
                          if (_referenceModelId === 'dynamic-reference') {
                            const updatedModel = createReferenceModel();
                            setReferenceModel(updatedModel);
                          }
                        }
                      }}
                      className={`w-16 bg-neutral-700 text-white text-xs px-2 py-1 rounded border text-center focus:outline-none ${
                        (effectiveGroundTruthParams.Cb * 1e6) < 0.1 || (effectiveGroundTruthParams.Cb * 1e6) > 50 
                          ? 'border-yellow-500 focus:border-yellow-400' 
                          : 'border-neutral-600 focus:border-blue-500'
                      }`}
                      step="0.1"
                    />
                    {((effectiveGroundTruthParams.Cb * 1e6) < 0.1 || (effectiveGroundTruthParams.Cb * 1e6) > 50) && (
                      <div className="absolute -top-1 -right-1 w-2 h-2 bg-yellow-500 rounded-full animate-pulse" />
                    )}
                  </div>
                </div>
                <ParamSlider 
                  label="" 
                  value={effectiveGroundTruthParams.Cb * 1e6} 
                  min={0.1} 
                  max={50} 
                  step={0.1}
                  unit="μF" 
                  onChange={(val: number) => {
                    setGroundTruthParams(prev => ({ ...prev, Cb: val / 1e6 }));
                    updateStatusMessage(`Cb set to ${val.toFixed(2)} μF`);
                    if (_referenceModelId === 'dynamic-reference') {
                      const updatedModel = createReferenceModel();
                      setReferenceModel(updatedModel);
                    }
                  }} 
                  ticks={generateLogSpace(faradToMicroFarad(PARAMETER_RANGES.Cb.min), faradToMicroFarad(PARAMETER_RANGES.Cb.max), gridSize).map((v, i) => ({ level: i + 1, value: v }))}
                  transformValue={(v) => v.toFixed(2)}
                  log={true}
                  tickLabels={generateLogSpace(faradToMicroFarad(PARAMETER_RANGES.Cb.min), faradToMicroFarad(PARAMETER_RANGES.Cb.max), gridSize).map(v => formatTickValue(v, 'μF', 2))}
                  readOnlyRange={false}
                />
              </div>
            </div>
          </CollapsibleSection>
        </div>
        
        {/* Save Profile Modal */}
        <SaveProfileModal
          isOpen={saveModalOpen}
          onClose={() => setSaveModalOpen(false)}
          onSave={(name, description) => {
            onSaveProfile(name, description);
            updateStatusMessage(`Profile "${name}" saved successfully`);
          }}
          defaultName={`Grid ${gridSize}x${gridSize}x${gridSize}x${gridSize}x${gridSize}`}
        />
        </div>
      </div>
    </div>
  );

  return (
    <div className="h-full flex gap-4 p-4 overflow-hidden">
      {/* Settings Section - Left Side - Scrollable */}
      <div className="w-80 flex-shrink-0 overflow-y-auto h-full">
        {controlsContent}
      </div>
      
      {/* Visualization Section - Right Side - Sticky */}
      <div className="flex-1 min-w-0 h-full overflow-hidden">
        {hasComputedResults ? (
          <>
            {/* ExportModal removed from settings cleanup */}
            
            <div className="bg-neutral-800 rounded border border-neutral-700 h-full flex flex-col">
              {/* Header with View Controls */}
              <div className="p-3 border-b border-neutral-600 flex-shrink-0">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-semibold text-neutral-200">Visualization</h3>
                  
                  {/* Visualization Type Selection - Moved to header */}
                  <div className="flex items-center gap-2">
                    <div className="flex bg-neutral-900 rounded-lg p-1 border border-neutral-600">
                      <button
                        onClick={() => setVisualizationType('spider2d')}
                        className={`px-2 py-1 text-xs font-medium rounded transition-all duration-200 ${
                          visualizationType === 'spider2d'
                            ? 'bg-blue-600 text-white shadow-sm'
                            : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-700'
                        }`}
                        title="2D multi-dimensional parameter visualization"
                      >
                        Spider 2D
                      </button>
                      <button
                        onClick={() => setVisualizationType('spider3d')}
                        className={`px-2 py-1 text-xs font-medium rounded transition-all duration-200 ${
                          visualizationType === 'spider3d'
                            ? 'bg-purple-600 text-white shadow-sm'
                            : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-700'
                        }`}
                        title="3D multi-dimensional parameter visualization"
                      >
                        Spider 3D
                      </button>
                      <button
                        onClick={() => setVisualizationType('nyquist')}
                        className={`px-2 py-1 text-xs font-medium rounded transition-all duration-200 ${
                          visualizationType === 'nyquist'
                            ? 'bg-green-600 text-white shadow-sm'
                            : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-700'
                        }`}
                        title="Real vs Imaginary impedance plot"
                      >
                        Nyquist
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* Rendering Mode Controls */}

              {/* Visualization Area */}
              <div className="flex-1 p-4 min-h-0">
                {(visualizationType === 'spider2d' || visualizationType === 'spider3d') ? (
                  /* Spider Visualization */
                  <div className="w-full h-full min-h-0 flex items-center justify-center">
                    <div className="spider-visualization w-full h-full min-h-0 flex items-center justify-center bg-neutral-900 rounded border border-neutral-600 relative">
                    <div className="w-full h-full max-w-full max-h-full">
                      {/* Conditionally render worker-assisted image or regular SpiderPlot */}
                      {shouldUseWorkerRendering && workerImageUrl ? (
                        <div className="w-full h-full flex items-center justify-center">
                          <img 
                            src={workerImageUrl} 
                            alt="Worker-rendered spider plot"
                            className="max-w-full max-h-full object-contain"
                            style={{ background: 'transparent' }}
                          />
                        </div>
                      ) : shouldUseWorkerRendering && isWorkerRendering ? (
                        <div className="w-full h-full flex flex-col items-center justify-center text-neutral-300">
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
                            width={800}
                            height={600}
                            showControls={true}
                            resnormScale={staticRenderSettings.resnormScale}
                          />
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
                    </div>
                    
                      {/* Removed performance overlay clutter */}
                    </div>
                  </div>
                ) : (
                  /* Nyquist Plot Visualization */
                  <div className="w-full h-full min-h-0 flex items-center justify-center">
                    <div className="w-full h-full bg-neutral-900 rounded border border-neutral-600 p-4 flex items-center justify-center">
                      <NyquistPlot
                        groundTruthParams={effectiveGroundTruthParams}
                        resnormGroups={resnormGroups}
                        visibleModels={allAvailableModels}
                        width={700}
                        height={500}
                        showGroundTruth={showGroundTruth}
                        chromaEnabled={chromaEnabled}
                        numPoints={_numPoints}
                        hiddenGroups={hiddenGroups}
                        selectedOpacityGroups={selectedOpacityGroups}
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>
          </>
        ) : (
          <div className="bg-neutral-800 rounded border border-neutral-700 min-h-[400px] flex items-center justify-center">
            <div className="text-center max-w-md mx-auto p-8">
              <div className="w-24 h-24 mx-auto mb-6 bg-neutral-700 rounded-full flex items-center justify-center">
                <svg className="w-12 h-12 text-neutral-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <h3 className="text-lg font-medium text-neutral-200 mb-2">No Data to Visualize</h3>
              <p className="text-sm text-neutral-400 mb-4">
                Run a parameter grid computation to generate spider plot visualization data.
              </p>
              <div className="text-xs text-neutral-500">
                Configure parameters and run computation using the toolbox controls.
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}; 