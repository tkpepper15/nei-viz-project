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
  const setVisualizationType = (type: 'spider3d' | 'nyquist') => {
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
        {/* Compact Header */}
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium text-neutral-200">Controls</h3>
          <div className="flex items-center text-xs text-green-400">
            <div className="w-2 h-2 bg-green-400 rounded-full mr-1"></div>
            Ready
          </div>
        </div>

        {/* Grid Configuration */}
        <div className="bg-neutral-800/50 rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-medium text-neutral-300">Grid</label>
            <span className="text-xs text-neutral-400 font-mono">{gridSize}×{gridSize}</span>
          </div>
          <input
            type="range"
            min={5}
            max={20}
            value={gridSize}
            onChange={(e) => setGridSize(Number(e.target.value))}
            className="w-full h-1.5 bg-neutral-700 rounded-lg appearance-none cursor-pointer"
          />
          <div className="flex justify-between text-xs text-neutral-500 mt-1">
            <span>Fast</span>
            <span>Precise</span>
          </div>
        </div>

        {/* Visualization Type - Compact */}
        <div className="bg-neutral-800/50 rounded-lg p-3">
          <label className="text-xs font-medium text-neutral-300 mb-2 block">View</label>
          <div className="grid grid-cols-2 gap-1 text-xs">
            {[
              { value: 'spider3d', label: '3D' },
              { value: 'nyquist', label: 'Nyq' }
            ].map(option => (
              <button
                key={option.value}
                onClick={() => setVisualizationType(option.value as 'spider3d' | 'nyquist')}
                className={`px-2 py-1 rounded text-center transition-colors ${
                  visualizationType === option.value
                    ? 'bg-blue-600 text-white'
                    : 'bg-neutral-700 text-neutral-200 hover:bg-neutral-600'
                }`}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>

        {/* Group Proportion - Compact */}
        <div className="bg-neutral-800/50 rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-medium text-neutral-300">Models</label>
            <span className="text-xs text-neutral-400 font-mono">{(groupPortion * 100).toFixed(0)}%</span>
          </div>
          <input
            type="range"
            min={0.1}
            max={1.0}
            step={0.1}
            value={groupPortion}
            onChange={(e) => setGroupPortion(Number(e.target.value))}
            className="w-full h-1.5 bg-neutral-700 rounded-lg appearance-none cursor-pointer"
          />
        </div>

        {/* Frequency Controls - Compact */}
        <div className="bg-neutral-800/50 rounded-lg p-3">
          <label className="text-xs font-medium text-neutral-300 mb-2 block">Frequency</label>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <input
                type="number"
                value={minFreq}
                onChange={(e) => setMinFreq(Number(e.target.value))}
                className="w-full px-2 py-1 bg-neutral-700 border border-neutral-600 rounded text-white text-xs"
                placeholder="Min Hz"
                step="0.1"
                min="0.01"
              />
            </div>
            <div>
              <input
                type="number"
                value={maxFreq}
                onChange={(e) => setMaxFreq(Number(e.target.value))}
                className="w-full px-2 py-1 bg-neutral-700 border border-neutral-600 rounded text-white text-xs"
                placeholder="Max Hz"
                step="1"
                min="1"
              />
            </div>
          </div>
          <div className="mt-2">
            <input
              type="number"
              value={numPoints}
              onChange={(e) => setNumPoints(Number(e.target.value))}
              className="w-full px-2 py-1 bg-neutral-700 border border-neutral-600 rounded text-white text-xs"
              placeholder="Points"
              step="10"
              min="10"
              max="1000"
            />
          </div>
        </div>

        {/* Circuit Parameters - Compact Display */}
        <div className="bg-neutral-800/50 rounded-lg p-3">
          <label className="text-xs font-medium text-neutral-300 mb-2 block">Parameters</label>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="text-neutral-400">
              Rs: <span className="text-neutral-200 font-mono">{(groundTruthParams?.Rsh || 100).toFixed(0)}Ω</span>
            </div>
            <div className="text-neutral-400">
              Ra: <span className="text-neutral-200 font-mono">{(groundTruthParams?.Ra || 1000).toFixed(0)}Ω</span>
            </div>
            <div className="text-neutral-400">
              Ca: <span className="text-neutral-200 font-mono">{((groundTruthParams?.Ca || 1e-6) * 1e6).toFixed(1)}µF</span>
            </div>
            <div className="text-neutral-400">
              Rb: <span className="text-neutral-200 font-mono">{(groundTruthParams?.Rb || 5000).toFixed(0)}Ω</span>
            </div>
            <div className="text-neutral-400 col-span-2">
              Cb: <span className="text-neutral-200 font-mono">{((groundTruthParams?.Cb || 5e-6) * 1e6).toFixed(1)}µF</span>
            </div>
          </div>
        </div>

        {/* Action Button - Prominent */}
        <div className="pt-2">
          <button 
            onClick={handleComputeRegressionMesh}
            disabled={isComputingGrid}
            className="w-full py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-neutral-600 text-white text-sm font-semibold rounded-lg transition-colors shadow-lg"
          >
            {isComputingGrid ? 'Computing Grid...' : 'Compute Grid'}
          </button>
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