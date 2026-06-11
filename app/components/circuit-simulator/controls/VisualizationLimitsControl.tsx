import React from 'react';
import * as Checkbox from '@radix-ui/react-checkbox';

interface VisualizationLimitsControlProps {
  userVisualizationPercentage: number;
  setUserVisualizationPercentage: (value: number) => void;
  maxVisualizationPoints: number;
  setMaxVisualizationPoints: (value: number) => void;
  isUserControlledLimits: boolean;
  setIsUserControlledLimits: (value: boolean) => void;
  totalComputedPoints: number;
  currentlyDisplayed: number;
  estimatedMemory: number;
}

export const VisualizationLimitsControl: React.FC<VisualizationLimitsControlProps> = ({
  userVisualizationPercentage,
  setUserVisualizationPercentage,
  maxVisualizationPoints,
  setMaxVisualizationPoints,
  isUserControlledLimits,
  setIsUserControlledLimits,
  totalComputedPoints,
  currentlyDisplayed,
  estimatedMemory
}) => {
  // Calculate what would be shown with current settings
  const projectedDisplayed = isUserControlledLimits 
    ? Math.min(
        Math.floor((userVisualizationPercentage / 100) * totalComputedPoints),
        maxVisualizationPoints,
        totalComputedPoints
      )
    : currentlyDisplayed;

  const projectedMemory = isUserControlledLimits 
    ? (projectedDisplayed * 1000) / 1024 / 1024 // Rough estimate in MB
    : estimatedMemory;

  // Predefined percentage options for quick selection
  const quickPercentages = [10, 25, 50, 75, 90, 100];

  // Predefined max point options
  const maxPointOptions = [
    { value: 50000, label: '50K' },
    { value: 100000, label: '100K' },
    { value: 250000, label: '250K' },
    { value: 500000, label: '500K' },
    { value: 1000000, label: '1M' }
  ];

  return (
    <div className="bg-neutral-800 rounded p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-medium text-neutral-300">Visualization Limits</h3>
        <div className="flex items-center gap-2">
          <Checkbox.Root
            id="user-controlled-limits"
            checked={isUserControlledLimits}
            onCheckedChange={(checked) => setIsUserControlledLimits(checked === true)}
            className="w-4 h-4 rounded border border-neutral-500 bg-neutral-700 flex items-center justify-center data-[state=checked]:bg-primary data-[state=checked]:border-primary focus:outline-none focus-visible:ring-1 focus-visible:ring-primary"
          >
            <Checkbox.Indicator>
              <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
              </svg>
            </Checkbox.Indicator>
          </Checkbox.Root>
          <label htmlFor="user-controlled-limits" className="text-xs text-neutral-300 cursor-pointer">
            Custom Control
          </label>
        </div>
      </div>

      {/* Current Status */}
      <div className="bg-neutral-800/50 border border-border rounded p-3 text-xs space-y-1">
        <div className="flex justify-between">
          <span className="text-neutral-400">Currently Showing:</span>
          <span className="text-white font-medium">{currentlyDisplayed.toLocaleString()}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-neutral-400">Total Computed:</span>
          <span className="text-neutral-200">{totalComputedPoints.toLocaleString()}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-neutral-400">Current %:</span>
          <span className="text-neutral-300">
            {totalComputedPoints > 0 ? ((currentlyDisplayed / totalComputedPoints) * 100).toFixed(1) : '0'}%
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-neutral-400">Est. Memory:</span>
          <span className="text-neutral-300">
            {estimatedMemory.toFixed(1)} MB
          </span>
        </div>
      </div>

      {isUserControlledLimits && (
        <div className="space-y-4">
          {/* Percentage Slider */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-xs text-neutral-300">Display Percentage</label>
              <span className="text-xs font-mono text-neutral-300">{userVisualizationPercentage}%</span>
            </div>
            
            <input
              type="range"
              min="1"
              max="100"
              step="1"
              value={isNaN(userVisualizationPercentage) ? 100 : userVisualizationPercentage}
              onChange={(e) => setUserVisualizationPercentage(Number(e.target.value))}
              className="w-full h-2 bg-neutral-600 rounded-lg appearance-none cursor-pointer slider"
            />
            
            {/* Quick percentage buttons */}
            <div className="flex gap-1">
              {quickPercentages.map(pct => (
                <button
                  key={pct}
                  onClick={() => setUserVisualizationPercentage(pct)}
                  className={`px-2 py-1 text-xs rounded transition-colors ${
                    userVisualizationPercentage === pct
                      ? 'bg-neutral-600 text-neutral-100'
                      : 'bg-neutral-600 text-neutral-300 hover:bg-neutral-500'
                  }`}
                >
                  {pct}%
                </button>
              ))}
            </div>
          </div>

          {/* Maximum Points Limit */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-xs text-neutral-300">Max Points Cap</label>
              <span className="text-xs font-mono text-neutral-300">{maxVisualizationPoints.toLocaleString()}</span>
            </div>
            
            <div className="flex gap-1">
              {maxPointOptions.map(option => (
                <button
                  key={option.value}
                  onClick={() => setMaxVisualizationPoints(option.value)}
                  className={`px-2 py-1 text-xs rounded transition-colors ${
                    maxVisualizationPoints === option.value
                      ? 'bg-neutral-600 text-neutral-100'
                      : 'bg-neutral-600 text-neutral-300 hover:bg-neutral-500'
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          {/* Projection Preview */}
          <div className="bg-neutral-800/50 border border-border rounded p-3 text-xs space-y-1">
            <div className="text-neutral-300 font-medium mb-1">Projected with Settings:</div>
            <div className="flex justify-between">
              <span className="text-neutral-400">Will Display:</span>
              <span className="text-neutral-200 font-medium font-mono">{projectedDisplayed.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-400">Projected %:</span>
              <span className="text-neutral-300">
                {totalComputedPoints > 0 ? ((projectedDisplayed / totalComputedPoints) * 100).toFixed(1) : '0'}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-400">Est. Memory:</span>
              <span className="text-neutral-300">
                {projectedMemory.toFixed(1)} MB
              </span>
            </div>
            {projectedMemory > 500 && (
              <div className="text-red-300 text-xs mt-1">
                ⚠️ High memory usage - may impact performance
              </div>
            )}
          </div>
        </div>
      )}

      {/* Help Text */}
      <div className="text-xs text-neutral-400 space-y-1">
        <div>• <strong>Auto Mode:</strong> System manages limits based on memory usage</div>
        <div>• <strong>Custom Mode:</strong> You control percentage and maximum points</div>
        <div>• <strong>Max Recommended:</strong> 1M points for optimal performance</div>
      </div>
    </div>
  );
}; 