import React from 'react';
import { StaticRenderSettings } from './StaticRenderControls';
import { CircuitParameters } from '../types/parameters';
import { SavedProfiles } from './SavedProfiles';
import { SavedProfilesState } from '../types/savedProfiles';

interface ToolboxSidebarProps {
  // Grid settings
  gridSize: number;
  setGridSize: (size: number) => void;
  minFreq: number;
  setMinFreq: (freq: number) => void;
  maxFreq: number;
  setMaxFreq: (freq: number) => void;
  numPoints: number;
  setNumPoints: (points: number) => void;
  
  // Static render settings
  staticRenderSettings: StaticRenderSettings;
  onStaticRenderSettingsChange: (settings: StaticRenderSettings) => void;
  
  // Circuit parameters
  groundTruthParams?: CircuitParameters | null;
  
  // Actions
  handleComputeRegressionMesh: () => void;
  isComputingGrid: boolean;
  
  // Profile management
  savedProfilesState: SavedProfilesState;
  hasLoadedFromStorage: boolean;
  handleCopyParams: (profileId: string) => void;
  onSelectProfile: (profileId: string) => void;
  onDeleteProfile: (profileId: string) => void;
  onEditProfile: (profileId: string, name: string, description?: string) => void;
}

export const ToolboxSidebar: React.FC<ToolboxSidebarProps> = ({
  gridSize,
  setGridSize,
  minFreq,
  setMinFreq,
  maxFreq,
  setMaxFreq,
  numPoints,
  setNumPoints,
  staticRenderSettings,
  onStaticRenderSettingsChange,
  groundTruthParams,
  handleComputeRegressionMesh,
  isComputingGrid,
  savedProfilesState,
  hasLoadedFromStorage,
  handleCopyParams,
  onSelectProfile,
  onDeleteProfile,
  onEditProfile
}) => {
  const visualizationType = staticRenderSettings.visualizationType;
  const setVisualizationType = (type: 'spider2d' | 'spider3d' | 'nyquist') => {
    onStaticRenderSettingsChange({
      ...staticRenderSettings,
      visualizationType: type
    });
  };

  const groupPortion = staticRenderSettings.groupPortion;
  const setGroupPortion = (value: number) => {
    onStaticRenderSettingsChange({
      ...staticRenderSettings,
      groupPortion: value
    });
  };

  return (
    <div className="w-80 bg-neutral-900 border-r border-neutral-700 flex flex-col h-full">
      <div className="flex-1 overflow-y-auto">
        <div className="p-4 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-neutral-200">Circuit Controls</h3>
          <div className="flex items-center gap-2">
            <span className="text-xs text-neutral-400">Auto</span>
            <button className="relative inline-flex h-5 w-9 items-center rounded-full bg-neutral-600 transition-colors">
              <span className="inline-block h-3 w-3 transform rounded-full bg-white transition-transform translate-x-1"></span>
            </button>
          </div>
        </div>

        {/* Grid Configuration */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-sm font-medium text-neutral-300">Grid Size</label>
            <span className="text-xs text-neutral-400 font-mono">{gridSize}</span>
          </div>
          <input
            type="range"
            min={5}
            max={20}
            value={gridSize}
            onChange={(e) => setGridSize(Number(e.target.value))}
            className="w-full h-2 bg-neutral-700 rounded-lg appearance-none cursor-pointer"
          />
          <div className="flex justify-between text-xs text-neutral-500 mt-1">
            <span>5 (Fast)</span>
            <span>20 (Detailed)</span>
          </div>
        </div>

        {/* Visualization Type Controls */}
        <div>
          <label className="text-sm font-medium text-neutral-300 mb-2 block">Visualization Type</label>
          <div className="grid grid-cols-3 gap-2">
            <button
              onClick={() => setVisualizationType('spider2d')}
              className={`px-2 py-2 text-xs font-medium rounded transition-colors ${
                visualizationType === 'spider2d' 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700'
              }`}
            >
              Spider 2D
            </button>
            <button
              onClick={() => setVisualizationType('spider3d')}
              className={`px-2 py-2 text-xs font-medium rounded transition-colors ${
                visualizationType === 'spider3d' 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700'
              }`}
            >
              Spider 3D
            </button>
            <button
              onClick={() => setVisualizationType('nyquist')}
              className={`px-2 py-2 text-xs font-medium rounded transition-colors ${
                visualizationType === 'nyquist' 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700'
              }`}
            >
              Nyquist
            </button>
          </div>
        </div>

        {/* Group Proportion Slider */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-sm font-medium text-neutral-300">Group Proportion</label>
            <span className="text-xs text-neutral-400 font-mono">{(groupPortion * 100).toFixed(0)}%</span>
          </div>
          <input
            type="range"
            min={0.1}
            max={1.0}
            step={0.1}
            value={groupPortion}
            onChange={(e) => setGroupPortion(Number(e.target.value))}
            className="w-full h-2 bg-neutral-700 rounded-lg appearance-none cursor-pointer"
          />
          <div className="flex justify-between text-xs text-neutral-500 mt-1">
            <span>10%</span>
            <span>100%</span>
          </div>
        </div>

        {/* Frequency Controls */}
        <div>
          <label className="text-sm font-medium text-neutral-300 mb-3 block">Frequency Range</label>
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-neutral-400 mb-1">Min (Hz)</label>
              <input
                type="number"
                value={minFreq}
                onChange={(e) => setMinFreq(Number(e.target.value))}
                className="w-full px-2 py-1.5 bg-neutral-800 border border-neutral-600 rounded text-white text-xs"
                step="0.1"
                min="0.01"
              />
            </div>
            <div>
              <label className="block text-xs text-neutral-400 mb-1">Max (Hz)</label>
              <input
                type="number"
                value={maxFreq}
                onChange={(e) => setMaxFreq(Number(e.target.value))}
                className="w-full px-2 py-1.5 bg-neutral-800 border border-neutral-600 rounded text-white text-xs"
                step="1"
                min="1"
              />
            </div>
            <div>
              <label className="block text-xs text-neutral-400 mb-1">Points</label>
              <input
                type="number"
                value={numPoints}
                onChange={(e) => setNumPoints(Number(e.target.value))}
                className="w-full px-2 py-1.5 bg-neutral-800 border border-neutral-600 rounded text-white text-xs"
                step="10"
                min="10"
                max="1000"
              />
            </div>
          </div>
        </div>

        {/* Circuit Parameters */}
        <div>
          <label className="text-sm font-medium text-neutral-300 mb-3 block">Circuit Parameters</label>
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-neutral-400 mb-1">R shunt (Ω)</label>
              <input
                type="number"
                value={groundTruthParams?.Rsh || 100}
                readOnly
                className="w-full px-2 py-1.5 bg-neutral-800 border border-neutral-600 rounded text-neutral-400 text-xs"
              />
            </div>
            <div>
              <label className="block text-xs text-neutral-400 mb-1">Ra (Ω)</label>
              <input
                type="number"
                value={groundTruthParams?.Ra || 1000}
                readOnly
                className="w-full px-2 py-1.5 bg-neutral-800 border border-neutral-600 rounded text-neutral-400 text-xs"
              />
            </div>
            <div>
              <label className="block text-xs text-neutral-400 mb-1">Ca (μF)</label>
              <input
                type="number"
                value={((groundTruthParams?.Ca || 1e-6) * 1e6).toFixed(2)}
                readOnly
                className="w-full px-2 py-1.5 bg-neutral-800 border border-neutral-600 rounded text-neutral-400 text-xs"
              />
            </div>
            <div>
              <label className="block text-xs text-neutral-400 mb-1">Rb (Ω)</label>
              <input
                type="number"
                value={groundTruthParams?.Rb || 5000}
                readOnly
                className="w-full px-2 py-1.5 bg-neutral-800 border border-neutral-600 rounded text-neutral-400 text-xs"
              />
            </div>
            <div>
              <label className="block text-xs text-neutral-400 mb-1">Cb (μF)</label>
              <input
                type="number"
                value={((groundTruthParams?.Cb || 5e-6) * 1e6).toFixed(2)}
                readOnly
                className="w-full px-2 py-1.5 bg-neutral-800 border border-neutral-600 rounded text-neutral-400 text-xs"
              />
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="space-y-2 pt-2">
          <button 
            onClick={handleComputeRegressionMesh}
            disabled={isComputingGrid}
            className="w-full py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-neutral-600 text-white text-sm font-medium rounded transition-colors"
          >
            {isComputingGrid ? 'Computing...' : 'Compute Grid'}
          </button>
          <button className="w-full py-2 text-sm font-medium text-neutral-300 hover:text-white transition-colors border border-neutral-600 rounded hover:bg-neutral-800">
            Reset Parameters
          </button>
        </div>

        {/* Status */}
        <div className="flex items-center gap-2 text-xs">
          <div className="w-2 h-2 bg-green-400 rounded-full"></div>
          <span className="text-neutral-400">Ready • Low Load</span>
        </div>

        {/* Saved Profiles Section */}
        {hasLoadedFromStorage && (
          <div className="border-t border-neutral-700 pt-4 mt-4">
            <SavedProfiles
              profiles={savedProfilesState.profiles}
              selectedProfile={savedProfilesState.selectedProfile}
              onCopyParams={handleCopyParams}
              onSelectProfile={onSelectProfile}
              onDeleteProfile={onDeleteProfile}
              onEditProfile={onEditProfile}
              onComputeProfile={onSelectProfile}
              isCollapsed={false}
            />
          </div>
        )}
        </div>
      </div>
    </div>
  );
};