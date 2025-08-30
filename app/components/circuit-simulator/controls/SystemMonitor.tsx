"use client";

import React, { useState } from 'react';

interface GridFilterSettings {
  enableSmartFiltering: boolean;
  visibilityPercentage: number;
  maxVisiblePoints: number;
  filterMode: 'best_resnorm' | 'distributed' | 'random';
  resnormThreshold: number;
  adaptiveFiltering: boolean;
}

interface SystemMonitorProps {
  gridSize?: number;
  totalGridPoints?: number;
  computedGridPoints?: number;
  onGridFilterChanged?: (settings: GridFilterSettings) => void;
}

// Calculate smart grid filtering recommendations
const calculateGridRecommendations = (gridSize: number) => {
  const totalPoints = Math.pow(gridSize, 5);
  
  if (totalPoints <= 243) { // 3^5
    return { percentage: 100, recommendation: 'Show all points' };
  } else if (totalPoints <= 1024) { // 4^5
    return { percentage: 75, recommendation: 'Show 75% best' };
  } else if (totalPoints <= 3125) { // 5^5
    return { percentage: 50, recommendation: 'Show 50% best' };
  } else if (totalPoints <= 7776) { // 6^5
    return { percentage: 25, recommendation: 'Show 25% best' };
  } else {
    return { percentage: 10, recommendation: 'Performance mode' };
  }
};

const GridFilterControls: React.FC<{
  gridSize: number;
  totalPoints: number;
  computedPoints: number;
  onFilterChanged: (settings: GridFilterSettings) => void;
}> = ({ gridSize, totalPoints, computedPoints, onFilterChanged }) => {
  const recommendations = calculateGridRecommendations(gridSize);
  const [filterSettings, setFilterSettings] = useState<GridFilterSettings>({
    enableSmartFiltering: totalPoints > 1000,
    visibilityPercentage: recommendations.percentage,
    maxVisiblePoints: Math.floor(totalPoints * (recommendations.percentage / 100)),
    filterMode: 'best_resnorm',
    resnormThreshold: 0.1,
    adaptiveFiltering: true
  });

  const handleSettingChange = (key: keyof GridFilterSettings, value: boolean | number | string) => {
    const newSettings = { 
      ...filterSettings, 
      [key]: value,
      // Update maxVisiblePoints when percentage changes
      ...(key === 'visibilityPercentage' && { 
        maxVisiblePoints: Math.floor(computedPoints * (value as number / 100)) 
      })
    };
    setFilterSettings(newSettings);
    onFilterChanged(newSettings);
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-bold text-neutral-200">Smart Filtering</span>
        <input
          type="checkbox"
          checked={filterSettings.enableSmartFiltering}
          onChange={(e) => handleSettingChange('enableSmartFiltering', e.target.checked)}
          className="w-3 h-3 accent-blue-500"
        />
      </div>

      {filterSettings.enableSmartFiltering && (
        <div className="space-y-3">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs text-neutral-400">Visible Points</span>
              <span className="text-xs text-neutral-300 font-mono">
                {Math.floor((filterSettings.visibilityPercentage / 100) * computedPoints).toLocaleString()}
              </span>
            </div>
            <input
              type="range"
              min="5"
              max="100"
              step="5"
              value={isNaN(filterSettings.visibilityPercentage) ? 50 : filterSettings.visibilityPercentage}
              onChange={(e) => handleSettingChange('visibilityPercentage', parseInt(e.target.value))}
              className="w-full h-2 bg-neutral-700 rounded appearance-none cursor-pointer slider"
              style={{
                background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${filterSettings.visibilityPercentage}%, #374151 ${filterSettings.visibilityPercentage}%, #374151 100%)`
              }}
            />
            <div className="flex justify-between text-[10px] text-neutral-500">
              <span>5%</span>
              <span className="text-blue-300 font-medium">{filterSettings.visibilityPercentage}%</span>
              <span>100%</span>
            </div>
            <div className="text-center text-[10px] text-blue-300">
              {recommendations.recommendation}
            </div>
          </div>

          {/* Filter Mode Selection */}
          <div className="space-y-2">
            <span className="text-xs text-neutral-400">Filter Mode</span>
            <select
              value={filterSettings.filterMode}
              onChange={(e) => handleSettingChange('filterMode', e.target.value)}
              className="w-full text-xs bg-neutral-700 border border-neutral-600 rounded px-2 py-1 text-neutral-200"
            >
              <option value="best_resnorm">Best Resnorm</option>
              <option value="distributed">Distributed</option>
              <option value="random">Random</option>
            </select>
          </div>

          {/* Adaptive Filtering Toggle */}
          <div className="flex items-center justify-between">
            <span className="text-xs text-neutral-400">Adaptive filtering</span>
            <input
              type="checkbox"
              checked={filterSettings.adaptiveFiltering}
              onChange={(e) => handleSettingChange('adaptiveFiltering', e.target.checked)}
              className="w-3 h-3 accent-blue-500"
            />
          </div>
        </div>
      )}
    </div>
  );
};

export const SystemMonitor: React.FC<SystemMonitorProps> = ({
  gridSize = 3,
  totalGridPoints = 0,
  computedGridPoints = 0,
  onGridFilterChanged
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const getSystemStatus = () => {
    if (totalGridPoints > 8000) {
      return { color: 'text-red-400', text: 'Hi', dotColor: 'bg-red-400' };
    }
    if (totalGridPoints > 3000) {
      return { color: 'text-yellow-400', text: 'Med', dotColor: 'bg-yellow-400' };
    }
    return { color: 'text-green-400', text: 'Lo', dotColor: 'bg-green-400' };
  };

  const systemStatus = getSystemStatus();

  return (
    <div className="border-b border-neutral-700 last:border-b-0">
      {/* Header */}
      <button 
        className="w-full px-4 py-3 bg-neutral-800 text-neutral-300 text-sm font-medium flex items-center justify-between hover:bg-neutral-750 transition-colors"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-3">
          <span className="text-sm font-bold">Performance</span>
          <div className="flex items-center gap-2 text-xs">
            <div className={`w-2 h-2 rounded-full ${systemStatus.dotColor}`}></div>
            <span className={`${systemStatus.color} font-medium`}>{systemStatus.text}</span>
            <span className="text-neutral-500">•</span>
            <span className="text-neutral-400">
              {computedGridPoints.toLocaleString()} pts
            </span>
          </div>
        </div>
        <svg 
          className={`w-4 h-4 transition-transform ${isExpanded ? 'transform rotate-180' : ''}`} 
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isExpanded && (
        <div className="px-4 py-3 bg-neutral-800 transition-all space-y-3">
          {/* Grid Statistics */}
          <div className="flex justify-between items-center text-xs">
            <span className="text-neutral-400">Grid: {gridSize}^5</span>
            <span className="text-neutral-300 font-mono">{totalGridPoints.toLocaleString()} total</span>
          </div>
          
          {/* Grid Filter Controls */}
          <GridFilterControls
            gridSize={gridSize}
            totalPoints={totalGridPoints}
            computedPoints={computedGridPoints}
            onFilterChanged={onGridFilterChanged || (() => {})}
          />
          
          {/* Quick Performance Settings */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-bold text-neutral-200">Auto-optimize</span>
              <input type="checkbox" defaultChecked className="w-3 h-3" />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm font-bold text-neutral-200">Performance warnings</span>
              <input type="checkbox" defaultChecked className="w-3 h-3" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}; 