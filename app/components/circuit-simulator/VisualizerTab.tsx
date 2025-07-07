import React, { useState, useEffect } from 'react';
import { SpiderPlot } from './visualizations/SpiderPlot';
import { ModelSnapshot, ResnormGroup } from './types';
import { GridParameterArrays } from './types';
import { ExportModal } from './controls/ExportModal';
import { ResizableSplitPane } from './controls/ResizableSplitPane';

interface VisualizerTabProps {
  resnormGroups: ResnormGroup[];
  hiddenGroups: number[];
  opacityLevel: number;
  referenceModelId: string | null;
  gridSize: number;
  onGridValuesGenerated: (values: GridParameterArrays) => void;
  // View control props
  onZoomIn?: () => void;
  onZoomOut?: () => void;
  onResetZoom?: () => void;
  onToggleLabels?: () => void;
  zoomLevel?: number;
  showLabels?: boolean;
}

export const VisualizerTab: React.FC<VisualizerTabProps> = ({
  resnormGroups,
  hiddenGroups,
  opacityLevel,
  referenceModelId: _referenceModelId, // eslint-disable-line @typescript-eslint/no-unused-vars
  gridSize,
  onGridValuesGenerated: _onGridValuesGenerated, // eslint-disable-line @typescript-eslint/no-unused-vars
  // View control props
  onZoomIn,
  onZoomOut,
  onResetZoom,
  onToggleLabels,
  zoomLevel,
  showLabels
}) => {

  const [isExportModalOpen, setIsExportModalOpen] = useState(false);
  const [chromaEnabled, setChromaEnabled] = useState<boolean>(true); // Default to chroma enabled
  const [showDistribution, setShowDistribution] = useState(false);
  const [opacityIntensity, setOpacityIntensity] = useState<number>(1e-30);
  const [sliderValue, setSliderValue] = useState<number>(0);
  const [isClient, setIsClient] = useState(false);
  
  // Add view control state - use props if provided, otherwise internal state
  const [internalZoomLevel, setInternalZoomLevel] = useState(1.0);
  
  // Use external props if provided, otherwise use internal state
  const currentZoomLevel = zoomLevel !== undefined ? zoomLevel : internalZoomLevel;
  const currentShowLabels = showLabels !== undefined ? showLabels : true;

  // Zoom level is not available in optimized SpiderPlot
  useEffect(() => {
    setInternalZoomLevel(1.0);
  }, []);
  
  // Add state for selected groups for opacity contrast (multi-select)
  const [selectedOpacityGroups, setSelectedOpacityGroups] = useState<number[]>([0]); // [0] means "Excellent" group by default
  
  // Logarithmic scale mapping for opacity intensity
  // Maps slider 0-100 to log scale 1e-30 to 10 (extreme range refined for usability)
  // 
  // HOW OPACITY INTENSITY WORKS (REFINED):
  // The opacity calculation uses: Math.pow(normalizedResnorm, 1 / opacityIntensity)
  // Where normalizedResnorm is 0-1 (0 = worst resnorm, 1 = best resnorm)
  //
  // Refined examples with normalizedResnorm = 0.5 (middle quality):
  // - opacityIntensity = 10.0:     Math.pow(0.5, 1/10) = 0.5^0.1 ≈ 0.93    (93% opacity - very low contrast)
  // - opacityIntensity = 1.0:      Math.pow(0.5, 1/1.0) = 0.5^1 = 0.50     (50% opacity - linear)
  // - opacityIntensity = 0.1:      Math.pow(0.5, 1/0.1) = 0.5^10 ≈ 0.001   (0.1% opacity - high contrast)
  // - opacityIntensity = 1e-3:     Math.pow(0.5, 1/1e-3) = 0.5^1000 ≈ 0    (virtually invisible)
  // - opacityIntensity = 1e-8:     Math.pow(0.5, 1/1e-8) = 0.5^1e8 ≈ 0     (only perfect matches visible)
  // - opacityIntensity = 1e-30:    Math.pow(0.5, 1/1e-30) = 0.5^1e30 = 0   (extreme precision for high grids)
  // Dynamic contextual slider that adapts to selected groups' resnorm distributions
  // LOWER values = MORE CONTRAST (only the very best resnorms get any opacity)
  // HIGHER values = LESS CONTRAST (more models remain visible)
  const getContextualRange = (): { min: number; max: number; dynamic: boolean } => {
    if (selectedOpacityGroups.length === 0 || !resnormGroups || resnormGroups.length === 0) {
      return { min: 1e-30, max: 10, dynamic: false };
    }
    
    // Get resnorm values from selected groups
    const selectedResnorms = selectedOpacityGroups
      .flatMap(groupIndex => resnormGroups[groupIndex]?.items || [])
      .map(model => model.resnorm)
      .filter(r => r !== undefined && r > 0) as number[];
    
    if (selectedResnorms.length === 0) {
      return { min: 1e-30, max: 10, dynamic: false };
    }
    
    // Calculate contextual range based on selected groups
    const sortedResnorms = [...selectedResnorms].sort((a, b) => a - b);
    const minResnorm = sortedResnorms[0];
    const maxResnorm = sortedResnorms[sortedResnorms.length - 1];
    
    // Calculate percentile thresholds for better contrast scaling
    const p90 = sortedResnorms[Math.floor(sortedResnorms.length * 0.9)];
    
    // Use a dynamic range that focuses on the meaningful spread
    const contextualMin = Math.max(minResnorm * 0.1, 1e-15); // Slightly below minimum
    const contextualMax = Math.min(maxResnorm * 2, p90 * 10); // Focus on meaningful range
    
    return { min: contextualMin, max: contextualMax, dynamic: true };
  };
  
  const sliderToOpacityIntensity = (sliderVal: number): number => {
    const { min, max, dynamic } = getContextualRange();
    
    if (!dynamic) {
      // Fallback to fixed range if no context
      const minLog = Math.log10(1e-30);
      const maxLog = Math.log10(10);
      const logValue = minLog + (sliderVal / 100) * (maxLog - minLog);
      return Math.pow(10, logValue);
    }
    
    // Use contextual range for dynamic scaling
    const minLog = Math.log10(min);
    const maxLog = Math.log10(max);
    const logValue = minLog + (sliderVal / 100) * (maxLog - minLog);
    return Math.pow(10, logValue);
  };
  
  const opacityIntensityToSlider = (opacityValue: number): number => {
    const { min, max, dynamic } = getContextualRange();
    
    if (!dynamic) {
      // Fallback to fixed range
      const minLog = Math.log10(1e-30);
      const maxLog = Math.log10(10);
      const logValue = Math.log10(Math.max(1e-30, Math.min(10, opacityValue)));
      return ((logValue - minLog) / (maxLog - minLog)) * 100;
    }
    
    // Use contextual range for dynamic scaling
    const minLog = Math.log10(min);
    const maxLog = Math.log10(max);
    const logValue = Math.log10(Math.max(min, Math.min(max, opacityValue)));
    return ((logValue - minLog) / (maxLog - minLog)) * 100;
  };
  
  const handleOpacityChange = (newSliderValue: string) => {
    const sliderVal = parseFloat(newSliderValue);
    setSliderValue(sliderVal);
    const mappedValue = sliderToOpacityIntensity(sliderVal);
    setOpacityIntensity(mappedValue);
  };
  
  useEffect(() => {
    setIsClient(true);
    // Initialize slider position based on default opacity intensity (1e-30 = 0%)
    const initialSliderPos = opacityIntensityToSlider(1e-30);
    setSliderValue(initialSliderPos);
  }, []);

  // Recalculate slider position when selected groups change for dynamic scaling
  useEffect(() => {
    if (selectedOpacityGroups.length > 0 && resnormGroups && resnormGroups.length > 0) {
      // Reset to high contrast when groups change to provide consistent starting point
      const highContrastSliderPos = opacityIntensityToSlider(1e-20);
      setSliderValue(highContrastSliderPos);
      setOpacityIntensity(sliderToOpacityIntensity(highContrastSliderPos));
    }
  }, [selectedOpacityGroups, resnormGroups?.length, opacityIntensityToSlider, sliderToOpacityIntensity]);

  // Reset selected groups if they don't exist or set default (fixed infinite loop)
  useEffect(() => {
    if (!resnormGroups || resnormGroups.length === 0) {
      setSelectedOpacityGroups([]);
    } else if (resnormGroups.length > 0 && selectedOpacityGroups.length === 0) {
      // Default to "Excellent" group (index 0) when groups are available but none selected
      setSelectedOpacityGroups([0]);
    } else {
      // Filter out groups that no longer exist
      const validGroups = selectedOpacityGroups.filter(groupIndex => groupIndex < resnormGroups.length);
      
      if (validGroups.length !== selectedOpacityGroups.length) {
        setSelectedOpacityGroups(validGroups.length > 0 ? validGroups : [0]);
      }
    }
  }, [resnormGroups?.length]); // Only depend on resnormGroups.length to avoid infinite loop

  // Zoom control functions - not available in optimized SpiderPlot
  const handleZoomIn = () => {
    if (onZoomIn) {
      onZoomIn();
    }
  };

  const handleZoomOut = () => {
    if (onZoomOut) {
      onZoomOut();
    }
  };

  const handleResetZoom = () => {
    if (onResetZoom) {
      onResetZoom();
    }
  };

  const handleToggleLabels = () => {
    if (onToggleLabels) {
      onToggleLabels();
    }
  };

  // Get all models from non-hidden groups for data preservation
  const allAvailableModels: ModelSnapshot[] = (resnormGroups || [])
    .filter((_, index) => !hiddenGroups.includes(index))
    .flatMap(group => group.items);

  // Get visible models based on selectedOpacityGroups for display
  const visibleModels: ModelSnapshot[] = (() => {
    // If no groups are selected for opacity, show NOTHING (not all models)
    if (selectedOpacityGroups.length === 0) {
      return [];
    }
    
    // Only show models from selected opacity groups (and not hidden)
    return (resnormGroups || [])
      .filter((_, index) => selectedOpacityGroups.includes(index) && !hiddenGroups.includes(index))
      .flatMap(group => group.items);
  })();

  // Calculate resnorm statistics for distribution using iterative approach
  // Use only the selected groups' statistics if specific groups are selected
  const relevantModels = selectedOpacityGroups.length > 0 && resnormGroups && resnormGroups.length > 0
    ? selectedOpacityGroups.flatMap(groupIndex => resnormGroups[groupIndex]?.items || [])
    : visibleModels;
    
  const resnormValues = relevantModels.map(model => model.resnorm).filter(r => r !== undefined) as number[];
  const resnormStats = resnormValues.length > 0 ? (() => {
    // Use iterative approach to avoid stack overflow
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    for (const value of resnormValues) {
      if (value < min) min = value;
      if (value > max) max = value;
      sum += value;
    }
    const sortedValues = [...resnormValues].sort((a, b) => a - b);
    return {
      min,
      max,
      mean: sum / resnormValues.length,
      median: sortedValues[Math.floor(resnormValues.length / 2)]
    };
  })() : null;

  // Check if we have actual computed results to show
  const hasComputedResults = resnormGroups && resnormGroups.length > 0 && allAvailableModels.length > 0;

  // Split visualization content
  const spiderPlotContent = (
    <>
      {/* Export Modal */}
      <ExportModal 
        isOpen={isExportModalOpen}
        onClose={() => setIsExportModalOpen(false)}
        visibleModels={allAvailableModels}
        opacityLevel={opacityLevel}
        chromaEnabled={chromaEnabled}
        opacityIntensity={opacityIntensity}
        gridSize={gridSize}
      />

      {/* Spider Plot with Controls */}
      {hasComputedResults ? (
        <div className="bg-neutral-800 rounded border border-neutral-700 h-full flex flex-col">
          {/* Spider Plot Controls Bar */}
          <div className="flex-shrink-0 p-2 border-b border-neutral-600 bg-neutral-750">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-xs text-neutral-400">View:</span>
                <button
                  onClick={handleZoomIn}
                  className="px-2 py-1 text-xs bg-neutral-700 text-neutral-300 rounded hover:bg-neutral-600 transition-colors"
                  title="Zoom In"
                >
                  +
                </button>
                <button
                  onClick={handleZoomOut}
                  className="px-2 py-1 text-xs bg-neutral-700 text-neutral-300 rounded hover:bg-neutral-600 transition-colors"
                  title="Zoom Out"
                >
                  −
                </button>
                <button
                  onClick={handleResetZoom}
                  className="px-2 py-1 text-xs bg-neutral-700 text-neutral-300 rounded hover:bg-neutral-600 transition-colors"
                  title="Reset Zoom"
                >
                  ⌂
                </button>
                <button
                  onClick={handleToggleLabels}
                  className={`px-2 py-1 text-xs rounded transition-colors ${
                    currentShowLabels 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-neutral-700 text-neutral-300 hover:bg-neutral-600'
                  }`}
                  title="Toggle Labels"
                >
                  Labels
                </button>
              </div>
              <div className="flex items-center gap-2 text-xs text-neutral-400">
                <span>Zoom: {(currentZoomLevel * 100).toFixed(0)}%</span>
              </div>
            </div>
          </div>

          {/* Spider Plot Visualization */}
          <div className="flex-1 p-2 min-h-0">
            <div className="spider-visualization w-full h-full min-h-0 flex items-center justify-center">
              <div className="w-full h-full max-w-full max-h-full">
                <SpiderPlot
                  meshItems={visibleModels.filter(model => model.parameters !== undefined)}
                  opacityFactor={opacityLevel}
                  maxPolygons={100000}
                  visualizationMode={chromaEnabled ? 'color' : 'opacity'}
                  opacityIntensity={opacityIntensity}
                  gridSize={gridSize}
                  includeLabels={true}
                  backgroundColor="transparent"
                />
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-neutral-800 rounded border border-neutral-700 h-full flex items-center justify-center">
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
              Use the controls on the left to set parameters and start computation.
            </div>
          </div>
        </div>
      )}
    </>
  );

  const controlsContent = hasComputedResults ? (
    <div className="bg-neutral-800 rounded border border-neutral-700 h-full flex flex-col">
      {/* Controls Header */}
      <div className="p-4 border-b border-neutral-600 flex-shrink-0">
        <div className="flex gap-4 items-center">
          {/* Left side - Chroma toggle and Export buttons */}
          <div className="flex items-center gap-3">
            {/* Chroma Toggle */}
            <button
              onClick={() => setChromaEnabled(!chromaEnabled)}
              className={`px-3 py-2 text-xs font-medium rounded-md transition-colors ${
                chromaEnabled
                  ? 'bg-primary text-white'
                  : 'bg-neutral-700 text-neutral-300 hover:bg-neutral-600'
              }`}
            >
              {chromaEnabled ? 'Chroma' : 'Mono'}
            </button>

            {/* Export Action */}
            <button
              onClick={() => setIsExportModalOpen(true)}
              disabled={!hasComputedResults}
              className={`px-4 py-2 text-xs font-medium rounded-md transition-colors ${
                hasComputedResults 
                  ? 'bg-blue-600 hover:bg-blue-700 text-white'
                  : 'bg-neutral-600 text-neutral-400 cursor-not-allowed'
              }`}
            >
              Export Plot
            </button>
          </div>

          {/* Right side - Mode-specific info */}
          <div className="flex-1 flex justify-end items-center gap-4">
            {selectedOpacityGroups.length === 0 ? (
              <span className="text-xs text-amber-400 flex items-center gap-1">
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
                No groups selected
              </span>
            ) : (
              <span className="text-xs text-neutral-400">
                {visibleModels.length} models visible
              </span>
            )}
            <span className="text-xs font-mono text-neutral-200 bg-neutral-700 px-2 py-1 rounded border">
              {isClient 
                ? (opacityIntensity >= 0.01 
                    ? (opacityIntensity < 1 ? opacityIntensity.toFixed(2) : opacityIntensity.toFixed(1)) + 'x'
                    : opacityIntensity.toExponential(0) + 'x')
                : '1e-30x'
              }
            </span>
          </div>
        </div>
      </div>

      {/* Compact Content Panel */}
      <div className="flex-1 min-h-0 overflow-y-auto custom-scrollbar">
        <div className="p-3 space-y-3"
             style={{ scrollbarWidth: 'thin', scrollbarColor: '#4B5563 #1F2937' }}>
        {/* Opacity Contrast Controls */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-neutral-200">Opacity Contrast</h3>
            <button
              onClick={() => setShowDistribution(!showDistribution)}
              className={`px-2 py-1 text-xs font-medium rounded transition-colors ${
                showDistribution
                  ? 'bg-amber-500 text-white hover:bg-amber-600'
                  : 'bg-neutral-700 text-neutral-300 hover:bg-neutral-600'
              }`}
            >
              {showDistribution ? 'Hide' : 'Show'} Stats
            </button>
          </div>

          {/* Group Selection Toggle */}
          <div className="space-y-2">
            <label className="text-xs font-medium text-neutral-300">Show groups:</label>
            
            {/* Horizontal toggle buttons */}
            {resnormGroups && resnormGroups.length > 0 ? (
              <div className="flex gap-1 flex-wrap">
                {/* Individual group buttons only */}
                {resnormGroups.map((group, index) => {
                  const isSelected = selectedOpacityGroups.includes(index);
                  return (
                    <button
                      key={index}
                      onClick={() => {
                        // Toggle individual group selection
                        if (isSelected) {
                          setSelectedOpacityGroups(selectedOpacityGroups.filter(i => i !== index));
                        } else {
                          setSelectedOpacityGroups([...selectedOpacityGroups, index]);
                        }
                      }}
                      className={`px-2 py-1 text-xs font-medium rounded transition-all flex items-center gap-1 ${
                        isSelected
                          ? 'bg-blue-600 text-white shadow-sm border border-blue-500'
                          : 'bg-neutral-700 text-neutral-300 hover:bg-neutral-600 hover:text-white border border-transparent'
                      }`}
                    >
                      <div 
                        className="w-1.5 h-1.5 rounded-full" 
                        style={{ backgroundColor: group.color }}
                      />
                      {group.label.split(' ')[0]}
                      <span className="text-[10px] opacity-60">({group.items.length})</span>
                    </button>
                  );
                })}
                
                {/* Clear All button */}
                {selectedOpacityGroups.length > 0 && (
                  <button
                    onClick={() => setSelectedOpacityGroups([])}
                    className="px-2 py-1 text-xs font-medium rounded transition-colors bg-red-600/20 text-red-300 hover:bg-red-600/30 border border-red-600/30"
                  >
                    Clear
                  </button>
                )}
              </div>
            ) : (
              <div className="text-xs text-neutral-500 italic">
                Run computation to enable group selection
              </div>
            )}
          </div>
          
          {/* Opacity Intensity Slider */}
          {selectedOpacityGroups.length > 0 && (
            <div className="space-y-2">
              {/* Dynamic range indicator */}
              {(() => {
                const { min, max, dynamic } = getContextualRange();
                return dynamic && (
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-blue-400 font-medium flex items-center gap-1">
                      <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse"></div>
                      Dynamic Range
                    </span>
                    <span className="text-neutral-400 font-mono text-[10px]">
                      {min.toExponential(1)} - {max.toExponential(1)}
                    </span>
                  </div>
                );
              })()}

              <div className="relative">
                <div className="absolute top-1/2 left-0 right-0 h-1.5 bg-neutral-600 rounded transform -translate-y-1/2"></div>
                <div 
                  className={`absolute top-1/2 left-0 h-1.5 rounded transform -translate-y-1/2 transition-all duration-200 ${
                    getContextualRange().dynamic 
                      ? 'bg-gradient-to-r from-blue-500/60 to-blue-400' 
                      : 'bg-gradient-to-r from-primary/60 to-primary'
                  }`}
                  style={{ width: `${isClient ? sliderValue : 0}%` }}
                ></div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  value={isClient ? sliderValue : 0}
                  onChange={(e) => handleOpacityChange(e.target.value)}
                  className="relative w-full h-3 bg-transparent appearance-none cursor-pointer slider z-10"
                />
              </div>
              <div className="flex justify-between text-[10px] text-neutral-400">
                <span>High Contrast</span>
                <span>Low Contrast</span>
              </div>
              {getContextualRange().dynamic && (
                <div className="text-[10px] text-blue-300 text-center">
                  Range optimized for selected groups
                </div>
              )}
            </div>
          )}

          {/* Distribution Panel */}
          {showDistribution && resnormStats && selectedOpacityGroups.length > 0 && (
            <div className="space-y-2 p-3 bg-neutral-900 rounded border border-neutral-600">
              <h4 className="text-xs font-medium text-neutral-200">
                Stats
                {selectedOpacityGroups.length > 0 && resnormGroups && selectedOpacityGroups.length < resnormGroups.length && (
                  <span className="text-[10px] font-normal text-neutral-400 ml-2">
                    ({selectedOpacityGroups.map(i => resnormGroups[i]?.label.split(' ')[0]).join(', ')})
                  </span>
                )}
              </h4>
              
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-neutral-700 p-2 rounded">
                  <div className="text-[10px] text-neutral-400">Count</div>
                  <div className="text-xs font-mono text-neutral-200">{resnormValues.length}</div>
                </div>
                <div className="bg-neutral-700 p-2 rounded">
                  <div className="text-[10px] text-neutral-400">Min</div>
                  <div className="text-xs font-mono text-neutral-200">{resnormStats.min.toExponential(1)}</div>
                </div>
                <div className="bg-neutral-700 p-2 rounded">
                  <div className="text-[10px] text-neutral-400">Max</div>
                  <div className="text-xs font-mono text-neutral-200">{resnormStats.max.toExponential(1)}</div>
                </div>
                <div className="bg-neutral-700 p-2 rounded">
                  <div className="text-[10px] text-neutral-400">Median</div>
                  <div className="text-xs font-mono text-neutral-200">{resnormStats.median.toExponential(1)}</div>
                </div>
              </div>
            </div>
          )}
        </div>
        </div>
      </div>
    </div>
  ) : (
    <div className="bg-neutral-800 rounded border border-neutral-700 h-full flex items-center justify-center">
      <div className="text-center max-w-sm mx-auto p-6">
        <div className="w-16 h-16 mx-auto mb-4 bg-neutral-700 rounded-full flex items-center justify-center">
          <svg className="w-8 h-8 text-neutral-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4" />
          </svg>
        </div>
        <h4 className="text-sm font-medium text-neutral-300 mb-2">Controls Unavailable</h4>
        <p className="text-xs text-neutral-500">
          Run a computation to enable visualization controls.
        </p>
      </div>
    </div>
  );

  return (
    <div className="h-full">
      <ResizableSplitPane 
        defaultSplitHeight={70}
        minTopHeight={400}
        minBottomHeight={180}
      >
        <div key="spider-plot">{spiderPlotContent}</div>
        <div key="controls">{controlsContent}</div>
      </ResizableSplitPane>
    </div>
  );
}; 