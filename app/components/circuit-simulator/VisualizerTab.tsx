import React, { useState, useEffect } from 'react';
import { BaseSpiderPlot } from './BaseSpiderPlot';
import { ModelSnapshot, ResnormGroup } from './utils/types';
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
}

export const VisualizerTab: React.FC<VisualizerTabProps> = ({
  resnormGroups,
  hiddenGroups,
  opacityLevel,
  referenceModelId,
  gridSize,
  onGridValuesGenerated
}) => {

  const [isExportModalOpen, setIsExportModalOpen] = useState(false);
  const [chromaEnabled, setChromaEnabled] = useState<boolean>(true); // Default to chroma enabled
  const [showDistribution, setShowDistribution] = useState(false);
  const [opacityIntensity, setOpacityIntensity] = useState<number>(1e-30);
  const [sliderValue, setSliderValue] = useState<number>(0);
  const [isClient, setIsClient] = useState(false);
  
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
    if (selectedOpacityGroups.length === 0 || resnormGroups.length === 0) {
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
    if (selectedOpacityGroups.length > 0 && resnormGroups.length > 0) {
      // Reset to high contrast when groups change to provide consistent starting point
      const highContrastSliderPos = opacityIntensityToSlider(1e-20);
      setSliderValue(highContrastSliderPos);
      setOpacityIntensity(sliderToOpacityIntensity(highContrastSliderPos));
    }
  }, [selectedOpacityGroups, resnormGroups.length]);

  // Reset selected groups if they don't exist or set default (fixed infinite loop)
  useEffect(() => {
    if (resnormGroups.length === 0) {
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
  }, [resnormGroups.length]); // Only depend on resnormGroups.length to avoid infinite loop

  // Get all models from non-hidden groups for data preservation
  const allAvailableModels: ModelSnapshot[] = resnormGroups
    .filter((_, index) => !hiddenGroups.includes(index))
    .flatMap(group => group.items);

  // Get visible models based on selectedOpacityGroups for display
  const visibleModels: ModelSnapshot[] = (() => {
    // If no groups are selected for opacity, show all non-hidden groups
    if (selectedOpacityGroups.length === 0) {
      return allAvailableModels;
    }
    
    // Only show models from selected opacity groups (and not hidden)
    return resnormGroups
      .filter((_, index) => selectedOpacityGroups.includes(index) && !hiddenGroups.includes(index))
      .flatMap(group => group.items);
  })();

  // Calculate resnorm statistics for distribution using iterative approach
  // Use only the selected groups' statistics if specific groups are selected
  const relevantModels = selectedOpacityGroups.length > 0 && resnormGroups.length > 0
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
  const hasComputedResults = resnormGroups.length > 0 && allAvailableModels.length > 0;

  // Split visualization content
  const spiderPlotContent = (
    <>
      {/* Export Modal */}
      <ExportModal 
        isOpen={isExportModalOpen}
        onClose={() => setIsExportModalOpen(false)}
        visibleModels={allAvailableModels}
        referenceModelId={referenceModelId}
        opacityLevel={opacityLevel}
        onGridValuesGenerated={onGridValuesGenerated}
        chromaEnabled={chromaEnabled}
        opacityIntensity={opacityIntensity}
        gridSize={gridSize}
      />

      {/* Spider Plot Visualization - Only show when we have computed results */}
      {hasComputedResults ? (
        <div className="bg-neutral-800 rounded border border-neutral-700 h-full">
          <div className="spider-visualization h-full w-full">
            <BaseSpiderPlot
              meshItems={visibleModels.filter(model => model.parameters !== undefined)}
              referenceId={referenceModelId || undefined}
              opacityFactor={opacityLevel}
              gridSize={gridSize}
              onGridValuesGenerated={onGridValuesGenerated}
              chromaEnabled={chromaEnabled}
              opacityIntensity={opacityIntensity}
              selectedOpacityGroups={selectedOpacityGroups}
              resnormGroups={resnormGroups}
            />
          </div>
        </div>
      ) : (
        <div className="bg-neutral-800 rounded border border-neutral-700 text-center h-full flex items-center justify-center">
          <div className="text-neutral-400 text-sm">
            No data to visualize. Run a computation to generate results.
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
            <span className="text-xs text-neutral-400">
              {visibleModels.length} models visible
            </span>
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

      {/* Dynamic Content Panel - Scrollable only if needed */}
      <div className="p-4 flex-1 min-h-0">
        {/* Show Slider Controls */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-neutral-200">Opacity Contrast</h3>
            <button
              onClick={() => setShowDistribution(!showDistribution)}
              className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
                showDistribution
                  ? 'bg-amber-500 text-white hover:bg-amber-600'
                  : 'bg-neutral-700 text-neutral-300 hover:bg-neutral-600'
              }`}
            >
              {showDistribution ? 'Hide' : 'Show'} Distribution
            </button>
          </div>

          {/* Group Selection Toggle */}
          <div className="space-y-2">
            <label className="text-xs font-medium text-neutral-300">Apply contrast to:</label>
            
            {/* Horizontal toggle buttons */}
            {resnormGroups.length > 0 ? (
              <div className="flex gap-1.5 flex-wrap">
                {/* All Groups button */}
                <button
                  onClick={() => {
                    // Toggle all groups - if all selected, deselect all; otherwise select all
                    if (selectedOpacityGroups.length === resnormGroups.length) {
                      setSelectedOpacityGroups([]);
                    } else {
                      setSelectedOpacityGroups(resnormGroups.map((_, index) => index));
                    }
                  }}
                  className={`px-2.5 py-1.5 text-xs font-medium rounded transition-colors flex items-center gap-1.5 ${
                    selectedOpacityGroups.length === resnormGroups.length
                      ? 'bg-blue-600 text-white shadow-sm border border-blue-500'
                      : 'bg-neutral-700 text-neutral-300 hover:bg-neutral-600 hover:text-white border border-transparent'
                  }`}
                >
                  <div className="w-2 h-2 rounded-full bg-gradient-to-r from-emerald-400 to-red-500 opacity-70" />
                  All{selectedOpacityGroups.length === resnormGroups.length ? ' ✓' : ''}
                </button>
                
                {/* Individual group buttons */}
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
                      className={`px-2.5 py-1.5 text-xs font-medium rounded transition-all flex items-center gap-1.5 ${
                        isSelected
                          ? 'bg-blue-600 text-white shadow-sm border border-blue-500'
                          : 'bg-neutral-700 text-neutral-300 hover:bg-neutral-600 hover:text-white border border-transparent'
                      }`}
                    >
                      <div 
                        className="w-2 h-2 rounded-full" 
                        style={{ backgroundColor: group.color }}
                      />
                      {group.label.split(' ')[0]}{isSelected ? ' ✓' : ''}
                      <span className="text-[10px] opacity-60 ml-0.5">({group.items.length})</span>
                    </button>
                  );
                })}
              </div>
            ) : (
              <div className="text-xs text-neutral-500 italic">
                Run computation to enable group selection
              </div>
            )}
            

          </div>
          
          <div className="space-y-2">
            {/* Dynamic range indicator */}
            {(() => {
              const { min, max, dynamic } = getContextualRange();
              return dynamic && (
                <div className="flex items-center justify-between text-xs mb-2">
                  <span className="text-blue-400 font-medium flex items-center gap-1">
                    <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                    Dynamic Range Active
                  </span>
                  <span className="text-neutral-400 font-mono">
                    {min.toExponential(1)} - {max.toExponential(1)}
                  </span>
                </div>
              );
            })()}

            <div className="relative">
              <div className="absolute top-1/2 left-0 right-0 h-2 bg-neutral-600 rounded-lg transform -translate-y-1/2"></div>
              <div 
                className={`absolute top-1/2 left-0 h-2 rounded-lg transform -translate-y-1/2 transition-all duration-200 ${
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
                className="relative w-full h-4 bg-transparent appearance-none cursor-pointer slider z-10"
              />
            </div>
            <div className="flex justify-between text-xs text-neutral-400">
              <span>High Contrast</span>
              <span>Low Contrast</span>
            </div>
            {getContextualRange().dynamic && (
              <div className="text-xs text-blue-300 text-center mt-1">
                Contrast optimized for selected groups
              </div>
            )}
          </div>

          {/* Distribution Panel for Opacity Mode */}
          {showDistribution && resnormStats && (
            <div className="mt-6 space-y-4 p-4 bg-neutral-900 rounded-lg border border-neutral-600">
              <h4 className="text-sm font-medium text-neutral-200">
                Distribution Analysis
                {selectedOpacityGroups.length > 0 && selectedOpacityGroups.length < resnormGroups.length && (
                                    <span className="text-xs font-normal text-neutral-400 ml-2">
                  ({selectedOpacityGroups.map(i => resnormGroups[i]?.label.split(' ')[0]).join(', ')})
                </span>
                )}
              </h4>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="bg-neutral-700 p-2 rounded">
                  <div className="text-xs text-neutral-400">Count</div>
                  <div className="text-sm font-mono text-neutral-200">{resnormValues.length}</div>
                </div>
                <div className="bg-neutral-700 p-2 rounded">
                  <div className="text-xs text-neutral-400">Min</div>
                  <div className="text-sm font-mono text-neutral-200">{resnormStats.min.toExponential(2)}</div>
                </div>
                <div className="bg-neutral-700 p-2 rounded">
                  <div className="text-xs text-neutral-400">Max</div>
                  <div className="text-sm font-mono text-neutral-200">{resnormStats.max.toExponential(2)}</div>
                </div>
                <div className="bg-neutral-700 p-2 rounded">
                  <div className="text-xs text-neutral-400">Median</div>
                  <div className="text-sm font-mono text-neutral-200">{resnormStats.median.toExponential(2)}</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  ) : (
    <div className="bg-neutral-800 rounded border border-neutral-700 h-full flex items-center justify-center">
      <div className="text-neutral-400 text-sm">
        No controls available. Run a computation to enable controls.
      </div>
    </div>
  );

  return (
    <div className="h-full">
      <ResizableSplitPane 
        defaultSplitHeight={35}
        minTopHeight={250}
        minBottomHeight={200}
      >
        <div key="spider-plot">{spiderPlotContent}</div>
        <div key="controls">{controlsContent}</div>
      </ResizableSplitPane>
    </div>
  );
}; 