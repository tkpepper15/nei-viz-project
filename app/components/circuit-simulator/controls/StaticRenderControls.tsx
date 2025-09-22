import React, { useState, useCallback } from 'react';
import { CircuitParameters } from '../types/parameters';
import { ModelSnapshot } from '../types';

// Static render settings interface
export interface StaticRenderSettings {
  includeLabels: boolean;
  backgroundColor: 'transparent' | 'white' | 'black';
  opacityLevel: number;
  opacityExponent: number;
  groupPortion: number;
  selectedOpacityGroups: number[];
  showGroundTruth: boolean;
  visualizationType: 'spider3d' | 'nyquist' | 'residual';
  resnormSpread: number;
  useResnormCenter: boolean; // New: rotate around resnorm center instead of grid front
  liveRendering: boolean;
  resolution: string;
  format: 'png' | 'svg' | 'pdf' | 'webp' | 'jpg';
  quality: number;
}

// Default static render settings
export const defaultStaticRenderSettings: StaticRenderSettings = {
  includeLabels: true,
  backgroundColor: 'white',
  opacityLevel: 0.7,
  opacityExponent: 1.0,
  groupPortion: 0.2, // Show best 20% by default
  selectedOpacityGroups: [0], // Default to excellent performance group only
  showGroundTruth: true,
  visualizationType: 'spider3d',
  resnormSpread: 1.0,
  useResnormCenter: false, // Default to grid front rotation
  liveRendering: true, // Default to live rendering
  resolution: '1920x1080',
  format: 'png',
  quality: 95,
};

// Props interface
interface StaticRenderControlsProps {
  settings: StaticRenderSettings;
  onSettingsChange: (settings: StaticRenderSettings) => void;
  groundTruthParams: CircuitParameters;
  onGroundTruthChange: (params: CircuitParameters) => void;
  meshData: ModelSnapshot[]; // Mesh data for rendering
  onCreateRenderJob: (settings: StaticRenderSettings, meshData: ModelSnapshot[]) => void;
  isRendering: boolean;
}

export const StaticRenderControls: React.FC<StaticRenderControlsProps> = ({
  settings,
  onSettingsChange,
  groundTruthParams,
  onGroundTruthChange,
  meshData,
  onCreateRenderJob,
  isRendering,
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const updateSetting = useCallback((key: keyof StaticRenderSettings, value: StaticRenderSettings[keyof StaticRenderSettings]) => {
    onSettingsChange({ ...settings, [key]: value });
  }, [settings, onSettingsChange]);

  const handleGroundTruthChange = useCallback((field: keyof CircuitParameters, value: number | number[]) => {
    onGroundTruthChange({ ...groundTruthParams, [field]: value });
  }, [groundTruthParams, onGroundTruthChange]);

  const handleRender = useCallback(() => {
    onCreateRenderJob(settings, meshData);
  }, [settings, meshData, onCreateRenderJob]);

  const resolutionOptions = [
    { value: '1920x1080', label: 'Full HD (1920×1080)' },
    { value: '2560x1440', label: '1440p (2560×1440)' },
    { value: '3840x2160', label: '4K (3840×2160)' },
    { value: '1280x720', label: 'HD (1280×720)' },
  ];

  const formatOptions = [
    { value: 'png', label: 'PNG (High Quality)' },
    { value: 'jpg', label: 'JPEG (Smaller Size)' },
    { value: 'webp', label: 'WebP (Modern)' },
    { value: 'svg', label: 'SVG (Vector)' },
    { value: 'pdf', label: 'PDF (Document)' },
  ];

  return (
    <div className="bg-white p-4 rounded-lg border border-gray-200">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Static Export</h3>
        <button
          onClick={handleRender}
          disabled={isRendering || meshData.length === 0}
          className={`px-4 py-2 rounded font-medium ${
            isRendering || meshData.length === 0
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-blue-600 text-white hover:bg-blue-700'
          }`}
        >
          {isRendering ? 'Rendering...' : '🎨 Generate High-Quality Export'}
        </button>
      </div>

      {/* Basic Settings */}
      <div className="space-y-4">
        {/* Format and Resolution */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Format
            </label>
            <select
              value={settings.format}
              onChange={(e) => updateSetting('format', e.target.value as StaticRenderSettings['format'])}
              className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
            >
              {formatOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Resolution
            </label>
            <select
              value={settings.resolution}
              onChange={(e) => updateSetting('resolution', e.target.value)}
              className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
            >
              {resolutionOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Quality Slider (for lossy formats) */}
        {(settings.format === 'jpg' || settings.format === 'webp') && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Quality: {settings.quality}%
            </label>
            <input
              type="range"
              min="50"
              max="100"
              step="5"
              value={settings.quality}
              onChange={(e) => updateSetting('quality', parseInt(e.target.value))}
              className="w-full"
            />
          </div>
        )}

        {/* Basic Visualization Settings */}
        <div className="grid grid-cols-2 gap-4">

          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={settings.includeLabels}
              onChange={(e) => updateSetting('includeLabels', e.target.checked)}
              className="rounded"
            />
            <span className="text-sm text-gray-700">Include Labels</span>
          </label>
        </div>

        {/* Background Color */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Background
          </label>
          <select
            value={settings.backgroundColor}
            onChange={(e) => updateSetting('backgroundColor', e.target.value as StaticRenderSettings['backgroundColor'])}
            className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
          >
            <option value="white">White</option>
            <option value="transparent">Transparent</option>
            <option value="black">Black</option>
          </select>
        </div>

        {/* Group Portion */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <label className="text-sm font-medium text-gray-700">Group Portion:</label>
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={(() => {
                  // Display the actual logarithmic percentage that matches the filtering
                  const groupPortionPercent = settings.groupPortion * 100;
                  return groupPortionPercent < 1 
                    ? groupPortionPercent.toFixed(2) 
                    : groupPortionPercent.toFixed(1);
                })()}
                onChange={(e) => {
                  const displayValue = parseFloat(e.target.value);
                  if (!isNaN(displayValue) && displayValue > 0 && displayValue <= 100) {
                    updateSetting('groupPortion', displayValue / 100);
                  }
                }}
                className="w-16 bg-neutral-700 text-white text-xs px-1 py-1 rounded border border-neutral-600 focus:border-blue-500 focus:outline-none text-center"
              />
              <span className="text-xs text-gray-600">% of best fits</span>
            </div>
          </div>
          <input
            type="range"
            min="0.1"
            max="1.0"
            step="0.1"
            value={isNaN(settings.groupPortion) ? 1.0 : settings.groupPortion}
            onChange={(e) => updateSetting('groupPortion', parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
      </div>

      {/* Advanced Settings Toggle */}
      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="mt-4 text-sm text-blue-600 hover:text-blue-800"
      >
        {showAdvanced ? '▼ Hide Advanced Settings' : '▶ Show Advanced Settings'}
      </button>

      {/* Advanced Settings */}
      {showAdvanced && (
        <div className="mt-4 p-4 bg-gray-50 rounded border space-y-4">
          {/* Opacity Controls */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Opacity Level: {settings.opacityLevel.toFixed(1)}
              </label>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.1"
                value={isNaN(settings.opacityLevel) ? 0.5 : settings.opacityLevel}
                onChange={(e) => updateSetting('opacityLevel', parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Opacity Curve: {settings.opacityExponent.toFixed(1)}
              </label>
              <input
                type="range"
                min="0.1"
                max="5.0"
                step="0.1"
                value={isNaN(settings.opacityExponent) ? 1.0 : settings.opacityExponent}
                onChange={(e) => updateSetting('opacityExponent', parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
          </div>

          {/* Ground Truth Settings */}
          <div>
            <label className="flex items-center space-x-2 mb-2">
              <input
                type="checkbox"
                checked={settings.showGroundTruth}
                onChange={(e) => updateSetting('showGroundTruth', e.target.checked)}
                className="rounded"
              />
              <span className="text-sm font-medium text-gray-700">Show Ground Truth Overlay</span>
            </label>

            {settings.showGroundTruth && (
              <div className="grid grid-cols-2 gap-2 mt-2">
                <div>
                  <label className="block text-xs text-gray-600">R shunt (Ω)</label>
                  <input
                    type="number"
                    value={isNaN(groundTruthParams.Rsh) ? 100 : groundTruthParams.Rsh}
                    onChange={(e) => handleGroundTruthChange('Rsh', parseFloat(e.target.value))}
                    className="w-full p-1 text-sm border border-gray-300 rounded"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-600">Ra (Ω)</label>
                  <input
                    type="number"
                    value={isNaN(groundTruthParams.Ra) ? 1000 : groundTruthParams.Ra}
                    onChange={(e) => handleGroundTruthChange('Ra', parseFloat(e.target.value))}
                    className="w-full p-1 text-sm border border-gray-300 rounded"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-600">Ca (µF)</label>
                  <input
                    type="number"
                    value={isNaN(groundTruthParams.Ca * 1e6) ? 0 : groundTruthParams.Ca * 1e6}
                    onChange={(e) => handleGroundTruthChange('Ca', parseFloat(e.target.value) / 1e6)}
                    className="w-full p-1 text-sm border border-gray-300 rounded"
                    step="0.1"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-600">Rb (Ω)</label>
                  <input
                    type="number"
                    value={isNaN(groundTruthParams.Rb) ? 5000 : groundTruthParams.Rb}
                    onChange={(e) => handleGroundTruthChange('Rb', parseFloat(e.target.value))}
                    className="w-full p-1 text-sm border border-gray-300 rounded"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-600">Cb (µF)</label>
                  <input
                    type="number"
                    value={isNaN(groundTruthParams.Cb * 1e6) ? 0 : groundTruthParams.Cb * 1e6}
                    onChange={(e) => handleGroundTruthChange('Cb', parseFloat(e.target.value) / 1e6)}
                    className="w-full p-1 text-sm border border-gray-300 rounded"
                    step="0.1"
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Model Count Info */}
      {meshData.length > 0 && (
        <div className="mt-4 p-2 bg-blue-50 rounded text-sm text-blue-800">
          Ready to export {meshData.length.toLocaleString()} models
        </div>
      )}
    </div>
  );
};