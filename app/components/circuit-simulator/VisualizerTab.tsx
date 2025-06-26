import React, { useState, useEffect } from 'react';
import { BaseSpiderPlot } from './BaseSpiderPlot';
import { ModelSnapshot, ResnormGroup } from './utils/types';
import { GridParameterArrays } from './types';
import { ExportModal } from './controls/ExportModal';

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
  const [visualizationMode, setVisualizationMode] = useState<'color' | 'opacity'>('opacity');
  const [showDistribution, setShowDistribution] = useState(false);
  const [opacityIntensity, setOpacityIntensity] = useState<number>(1e-30);
  const [sliderValue, setSliderValue] = useState<number>(0);
  const [isClient, setIsClient] = useState(false);
  
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
  //
  // LOWER values = MORE CONTRAST (only the very best resnorms get any opacity)
  // HIGHER values = LESS CONTRAST (more models remain visible)
  const sliderToOpacityIntensity = (sliderVal: number): number => {
    const minLog = Math.log10(1e-30);  // log10(1e-30) = -30
    const maxLog = Math.log10(10);     // log10(10) = 1
    const logValue = minLog + (sliderVal / 100) * (maxLog - minLog);
    return Math.pow(10, logValue);
  };
  
  const opacityIntensityToSlider = (opacityValue: number): number => {
    const minLog = Math.log10(1e-30);
    const maxLog = Math.log10(10);
    const logValue = Math.log10(Math.max(1e-30, Math.min(10, opacityValue)));
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

  // Get all visible models across groups
  const visibleModels: ModelSnapshot[] = resnormGroups
    .filter((_, index) => !hiddenGroups.includes(index))
    .flatMap(group => group.items);

  // Calculate resnorm statistics for distribution using iterative approach
  const resnormValues = visibleModels.map(model => model.resnorm).filter(r => r !== undefined) as number[];
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

  return (
    <div className="space-y-4">
      {/* Main Controls Panel */}
      <div className="bg-neutral-800 rounded-lg p-4 border border-neutral-700">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
          
          {/* Visualization Mode */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-neutral-200">Visualization Mode</label>
            <div className="bg-neutral-700 rounded-md flex overflow-hidden">
              <button
                onClick={() => setVisualizationMode('color')}
                className={`flex-1 px-3 py-2 text-xs font-medium transition-colors ${
                  visualizationMode === 'color'
                    ? 'bg-blue-500 text-white'
                    : 'text-neutral-300 hover:bg-neutral-600'
                }`}
              >
                Color Groups
              </button>
              <button
                onClick={() => setVisualizationMode('opacity')}
                className={`flex-1 px-3 py-2 text-xs font-medium transition-colors ${
                  visualizationMode === 'opacity'
                    ? 'bg-blue-500 text-white'
                    : 'text-neutral-300 hover:bg-neutral-600'
                }`}
              >
                Monochrome
              </button>
            </div>
          </div>

          {/* Opacity Control - Extended */}
          <div className="lg:col-span-2 space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-neutral-200">Opacity Contrast</label>
              <span className="text-xs font-mono text-neutral-400 bg-neutral-700 px-2 py-1 rounded">
                {isClient 
                  ? (opacityIntensity >= 0.01 
                      ? (opacityIntensity < 1 ? opacityIntensity.toFixed(2) : opacityIntensity.toFixed(1)) + 'x'
                      : opacityIntensity.toExponential(0) + 'x')
                  : '1e-30x'
                }
              </span>
            </div>
            <div className="space-y-1">
              <input
                type="range"
                min="0"
                max="100"
                step="1"
                value={isClient ? sliderValue : 0}
                onChange={(e) => handleOpacityChange(e.target.value)}
                className="w-full h-3 bg-neutral-600 rounded-lg appearance-none cursor-pointer slider"
              />
              <div className="flex justify-between text-[10px] text-neutral-500">
                <span>1e-30x</span>
                <span>10x</span>
              </div>
            </div>
          </div>

          {/* Export Action */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-neutral-200">Export</label>
            <button
              onClick={() => setIsExportModalOpen(true)}
              className="w-full px-3 py-2 text-xs font-medium rounded-md transition-colors bg-blue-600 hover:bg-blue-700 text-white"
            >
              Export Plot
            </button>
          </div>

        </div>
      </div>

      {/* Spider Plot - Pentagon-based visualization */}
      <div className="spider-visualization">
        <BaseSpiderPlot
          meshItems={visibleModels.filter(model => model.parameters !== undefined)}
          referenceId={referenceModelId || undefined}
          opacityFactor={opacityLevel}
          gridSize={gridSize}
          onGridValuesGenerated={onGridValuesGenerated}
          visualizationMode={visualizationMode}
          opacityIntensity={opacityIntensity}
        />
      </div>

      {/* Distribution Toggle */}
      <div className="flex justify-center">
        <button
          onClick={() => setShowDistribution(!showDistribution)}
          className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
            showDistribution
              ? 'bg-amber-500 text-white hover:bg-amber-600'
              : 'bg-neutral-700 text-neutral-300 hover:bg-neutral-600'
          }`}
        >
          {showDistribution ? 'Hide' : 'Show'} Resnorm Distribution
        </button>
      </div>

      {/* Model Groups Control Panel */}
      <div className="bg-neutral-800 rounded-lg p-4 border border-neutral-700">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-neutral-200">Model Groups</h3>
          <span className="text-xs text-neutral-400">
            {visibleModels.length} total models
          </span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {resnormGroups.map((group, index) => {
            const isHidden = hiddenGroups.includes(index);
            
            return (
              <button
                key={index}
                onClick={() => {
                  // Dispatch custom event to toggle group visibility
                  window.dispatchEvent(new CustomEvent('toggleResnormGroup', {
                    detail: { groupIndex: index }
                  }));
                }}
                className={`p-3 rounded-lg border transition-all text-left ${
                  isHidden 
                    ? 'border-neutral-600 bg-neutral-700/30 opacity-50 hover:opacity-75' 
                    : 'border-neutral-600 bg-neutral-700 hover:bg-neutral-600 hover:border-neutral-500'
                }`}
              >
                <div className="flex items-center space-x-2 mb-2">
                  <div 
                    className="w-3 h-3 rounded-full border border-neutral-500" 
                    style={{ backgroundColor: group.color }}
                  />
                  <span className="text-xs font-medium text-neutral-200">
                    {group.label}
                  </span>
                  {isHidden && <span className="text-[10px] text-neutral-500 ml-auto">Hidden</span>}
                </div>
                <div className="text-[10px] text-neutral-400 mb-1">
                  {group.items.length} models
                </div>
                <div className="text-[9px] text-neutral-500 leading-tight">
                  {group.description}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Resnorm Distribution Panel */}
      {showDistribution && resnormStats && (
        <div className="bg-neutral-800 rounded-lg border border-neutral-700 p-4">
          <h4 className="text-sm font-medium text-white mb-3">Resnorm Distribution ({resnormValues.length} models)</h4>
          
          {/* Statistics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div className="text-center">
              <div className="text-xs text-neutral-400">Minimum</div>
              <div className="text-sm font-mono text-green-400">{resnormStats.min.toExponential(2)}</div>
            </div>
            <div className="text-center">
              <div className="text-xs text-neutral-400">Maximum</div>
              <div className="text-sm font-mono text-red-400">{resnormStats.max.toExponential(2)}</div>
            </div>
            <div className="text-center">
              <div className="text-xs text-neutral-400">Mean</div>
              <div className="text-sm font-mono text-blue-400">{resnormStats.mean.toExponential(2)}</div>
            </div>
            <div className="text-center">
              <div className="text-xs text-neutral-400">Median</div>
              <div className="text-sm font-mono text-cyan-400">{resnormStats.median.toExponential(2)}</div>
            </div>
          </div>

          {/* Simple Histogram */}
          <div className="space-y-2">
            <div className="text-xs text-neutral-400 mb-2">Distribution (logarithmic scale)</div>
            <div className="flex items-end h-20 space-x-1">
              {Array.from({ length: 20 }, (_, i) => {
                const logMin = Math.log10(resnormStats.min);
                const logMax = Math.log10(resnormStats.max);
                const binStart = Math.pow(10, logMin + (i / 20) * (logMax - logMin));
                const binEnd = Math.pow(10, logMin + ((i + 1) / 20) * (logMax - logMin));
                const count = resnormValues.filter(r => r >= binStart && r < binEnd).length;
                const maxCount = Math.max(...Array.from({ length: 20 }, (_, j) => {
                  const bStart = Math.pow(10, logMin + (j / 20) * (logMax - logMin));
                  const bEnd = Math.pow(10, logMin + ((j + 1) / 20) * (logMax - logMin));
                  return resnormValues.filter(r => r >= bStart && r < bEnd).length;
                }));
                const height = maxCount > 0 ? (count / maxCount) * 100 : 0;
                
                return (
                  <div
                    key={i}
                    className="bg-gradient-to-t from-blue-500 to-cyan-400 rounded-t-sm flex-1 transition-all hover:opacity-80"
                    style={{ height: `${height}%`, minHeight: count > 0 ? '2px' : '0px' }}
                    title={`Range: ${binStart.toExponential(1)} - ${binEnd.toExponential(1)}, Count: ${count}`}
                  />
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Export Modal */}
      <ExportModal
        isOpen={isExportModalOpen}
        onClose={() => setIsExportModalOpen(false)}
        visibleModels={visibleModels}
        referenceModelId={referenceModelId}
        opacityLevel={opacityLevel}
        onGridValuesGenerated={onGridValuesGenerated}
        visualizationMode={visualizationMode}
        opacityIntensity={opacityIntensity}
        gridSize={gridSize}
      />
    </div>
  );
}; 