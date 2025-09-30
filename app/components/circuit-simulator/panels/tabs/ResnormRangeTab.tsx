/**
 * Resnorm Range Tab
 * =================
 *
 * Dedicated tab for resnorm range selection and histogram scrubbing
 * - Interactive histogram with range selection
 * - Model navigation and filtering
 * - Resnorm value distribution analysis
 */

import React, { useState, useMemo, useCallback } from 'react';
import { BottomPanelTabProps } from '../CollapsibleBottomPanel';
import ResnormDisplay from '../../insights/ResnormDisplay';

interface ResnormRangeTabProps extends BottomPanelTabProps {
  // Current resnorm state from parent
  currentResnorm?: number | null;

  // Resnorm range selection
  selectedResnormRange?: {min: number; max: number} | null;
  onResnormRangeChange?: (min: number, max: number) => void;
  onResnormSelect?: (resnorm: number) => void;

  // Navigation controls
  navigationOffset?: number;
  onNavigationOffsetChange?: (offset: number) => void;
  navigationWindowSize?: number;

  // Tagged models for visualization
  taggedModels?: Map<string, string>;
  tagColors?: string[];
}

export const ResnormRangeTab: React.FC<ResnormRangeTabProps> = ({
  gridResults,
  isVisible,
  currentResnorm,
  selectedResnormRange,
  onResnormRangeChange,
  onResnormSelect,
  navigationOffset = 0,
  onNavigationOffsetChange,
  navigationWindowSize = 1000,
  taggedModels = new Map(),
  tagColors = [
    '#FF6B9D', '#4ECDC4', '#45B7D1', '#96CEB4',
    '#FFEAA7', '#DDA0DD', '#98D8E8', '#F8C471',
    '#82E0AA', '#85C1E9'
  ]
}) => {
  const [localNavigationOffset, setLocalNavigationOffset] = useState(navigationOffset);

  // Handle navigation offset changes
  const handleNavigationOffsetChange = useCallback((offset: number) => {
    setLocalNavigationOffset(offset);
    if (onNavigationOffsetChange) {
      onNavigationOffsetChange(offset);
    }
  }, [onNavigationOffsetChange]);

  // Handle resnorm range selection
  const handleResnormRangeChange = useCallback((min: number, max: number) => {
    if (onResnormRangeChange) {
      onResnormRangeChange(min, max);
    }
  }, [onResnormRangeChange]);

  // Handle resnorm value selection
  const handleResnormSelect = useCallback((resnorm: number) => {
    if (onResnormSelect) {
      onResnormSelect(resnorm);
    }
  }, [onResnormSelect]);

  // Sort models by resnorm for consistent navigation
  const sortedModels = useMemo(() => {
    if (!gridResults || gridResults.length === 0) return [];
    return [...gridResults]
      .filter(model => model.resnorm !== undefined && model.resnorm !== null && !isNaN(model.resnorm))
      .sort((a, b) => (a.resnorm || 0) - (b.resnorm || 0));
  }, [gridResults]);

  // Get visible models within navigation window
  const visibleModels = useMemo(() => {
    if (!sortedModels.length) return [];
    const startIndex = Math.max(0, Math.min(localNavigationOffset, sortedModels.length - 1));
    const endIndex = Math.min(startIndex + navigationWindowSize, sortedModels.length);
    return sortedModels.slice(startIndex, endIndex);
  }, [sortedModels, localNavigationOffset, navigationWindowSize]);

  // Calculate resnorm statistics
  const resnormStats = useMemo(() => {
    if (!sortedModels.length) return null;

    const resnorms = sortedModels.map(m => m.resnorm || 0);
    const min = Math.min(...resnorms);
    const max = Math.max(...resnorms);
    const mean = resnorms.reduce((sum, val) => sum + val, 0) / resnorms.length;

    // Calculate median
    const sortedResnorms = [...resnorms].sort((a, b) => a - b);
    const median = sortedResnorms.length % 2 === 0
      ? (sortedResnorms[sortedResnorms.length / 2 - 1] + sortedResnorms[sortedResnorms.length / 2]) / 2
      : sortedResnorms[Math.floor(sortedResnorms.length / 2)];

    return { min, max, mean, median, count: resnorms.length };
  }, [sortedModels]);

  if (!isVisible) return null;

  return (
    <div className="flex-1 overflow-y-auto min-h-0 bg-neutral-900">
      <div className="p-4 space-y-4">

        {/* Resnorm Statistics Summary */}
        {resnormStats && (
          <div className="bg-neutral-800 rounded-lg p-4">
            <h3 className="text-sm font-medium text-neutral-200 mb-3">Resnorm Distribution</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-neutral-400">Total Models:</span>
                <span className="ml-2 font-mono text-neutral-200">{resnormStats.count.toLocaleString()}</span>
              </div>
              <div>
                <span className="text-neutral-400">Min:</span>
                <span className="ml-2 font-mono text-green-400">{resnormStats.min.toExponential(3)}</span>
              </div>
              <div>
                <span className="text-neutral-400">Max:</span>
                <span className="ml-2 font-mono text-red-400">{resnormStats.max.toExponential(3)}</span>
              </div>
              <div>
                <span className="text-neutral-400">Median:</span>
                <span className="ml-2 font-mono text-blue-400">{resnormStats.median.toExponential(3)}</span>
              </div>
            </div>
          </div>
        )}

        {/* Selected Range Display */}
        {selectedResnormRange && (
          <div className="bg-blue-900/20 rounded-lg p-3 border border-blue-700/30">
            <h4 className="text-xs font-medium text-blue-200 mb-2">Selected Range</h4>
            <div className="text-sm text-blue-300">
              <div>Min: <span className="font-mono">{selectedResnormRange.min.toExponential(3)}</span></div>
              <div>Max: <span className="font-mono">{selectedResnormRange.max.toExponential(3)}</span></div>
              <div className="text-xs text-blue-400 mt-1">
                Models in range: {sortedModels.filter(m =>
                  m.resnorm !== undefined &&
                  m.resnorm >= selectedResnormRange.min &&
                  m.resnorm <= selectedResnormRange.max
                ).length}
              </div>
            </div>
          </div>
        )}

        {/* Current Resnorm Display */}
        {currentResnorm !== null && currentResnorm !== undefined && (
          <div className="bg-orange-900/20 rounded-lg p-3 border border-orange-700/30">
            <h4 className="text-xs font-medium text-orange-200 mb-2">Current Selection</h4>
            <div className="text-sm text-orange-300">
              Resnorm: <span className="font-mono">{currentResnorm.toExponential(3)}</span>
            </div>
          </div>
        )}

        {/* Enhanced Histogram Scrubber */}
        <div className="bg-neutral-800 rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-neutral-200">Interactive Histogram</h3>
            {sortedModels.length > navigationWindowSize && (
              <div className="text-xs text-neutral-400">
                Showing {localNavigationOffset + 1} - {Math.min(localNavigationOffset + navigationWindowSize, sortedModels.length)} of {sortedModels.length.toLocaleString()}
              </div>
            )}
          </div>

          {/* ResnormDisplay Component - Optimized for bottom panel */}
          <ResnormDisplay
            models={sortedModels}
            visibleModels={visibleModels}
            navigationOffset={localNavigationOffset}
            onNavigationOffsetChange={handleNavigationOffsetChange}
            navigationWindowSize={navigationWindowSize}
            onResnormRangeChange={handleResnormRangeChange}
            currentResnorm={currentResnorm}
            onResnormSelect={handleResnormSelect}
            taggedModels={taggedModels}
            tagColors={tagColors}
          />
        </div>

        {/* Navigation Controls for Large Datasets */}
        {sortedModels.length > navigationWindowSize && (
          <div className="bg-neutral-800 rounded-lg p-4">
            <h3 className="text-sm font-medium text-neutral-200 mb-3">Dataset Navigation</h3>

            <div className="space-y-3">
              {/* Navigation slider */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-neutral-400">Position in Dataset</span>
                  <span className="text-xs text-neutral-200 font-mono">
                    {Math.round((localNavigationOffset / Math.max(1, sortedModels.length - navigationWindowSize)) * 100)}%
                  </span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={Math.max(0, sortedModels.length - navigationWindowSize)}
                  step={Math.max(1, Math.floor((sortedModels.length - navigationWindowSize) / 100))}
                  value={localNavigationOffset}
                  onChange={(e) => handleNavigationOffsetChange(Number(e.target.value))}
                  className="w-full h-2 bg-neutral-600 rounded-lg appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-purple-500"
                />
              </div>

              {/* Quick navigation buttons */}
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleNavigationOffsetChange(0)}
                  className="px-3 py-1 text-xs bg-neutral-600 hover:bg-neutral-500 rounded transition-colors"
                  disabled={localNavigationOffset === 0}
                >
                  First
                </button>
                <button
                  onClick={() => handleNavigationOffsetChange(Math.max(0, localNavigationOffset - navigationWindowSize))}
                  className="px-3 py-1 text-xs bg-neutral-600 hover:bg-neutral-500 rounded transition-colors"
                  disabled={localNavigationOffset === 0}
                >
                  Previous
                </button>
                <button
                  onClick={() => handleNavigationOffsetChange(Math.min(sortedModels.length - navigationWindowSize, localNavigationOffset + navigationWindowSize))}
                  className="px-3 py-1 text-xs bg-neutral-600 hover:bg-neutral-500 rounded transition-colors"
                  disabled={localNavigationOffset >= sortedModels.length - navigationWindowSize}
                >
                  Next
                </button>
                <button
                  onClick={() => handleNavigationOffsetChange(Math.max(0, sortedModels.length - navigationWindowSize))}
                  className="px-3 py-1 text-xs bg-neutral-600 hover:bg-neutral-500 rounded transition-colors"
                  disabled={localNavigationOffset >= sortedModels.length - navigationWindowSize}
                >
                  Last
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Usage Instructions */}
        <div className="bg-blue-900/20 rounded-lg p-3 border border-blue-700/30">
          <h4 className="text-xs font-medium text-blue-200 mb-2">How It Works</h4>
          <div className="text-xs text-blue-300/90 space-y-2">
            <div><strong>Range Selection:</strong> Drag the orange handles to filter models shown in 3D plot</div>
            <div><strong>Model Navigation:</strong> Double-click histogram or use arrow buttons to highlight specific models</div>
            <div><strong>3D Integration:</strong> Hover over 3D plot updates the golden cursor position here</div>
            <div className="text-blue-400 mt-2">
              <em>Current resnorm (gold line) • Selected range (orange handles) • Tagged models (colored)</em>
            </div>
          </div>
        </div>

        {/* No Data State */}
        {(!gridResults || gridResults.length === 0) && (
          <div className="text-center py-8 text-neutral-400">
            <div className="text-lg mb-2">No Resnorm Data Available</div>
            <div className="text-sm">Run a computation to generate resnorm distribution data</div>
          </div>
        )}

      </div>
    </div>
  );
};

export default ResnormRangeTab;