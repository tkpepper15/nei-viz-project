import React, { useState, useCallback } from 'react';
import { CircuitParameters } from '../types/parameters';
import { ModelSnapshot } from '../types';

// Static render settings interface
export interface StaticRenderSettings {
  chromaEnabled: boolean;
  includeLabels: boolean;
  backgroundColor: 'transparent' | 'white' | 'black';
  opacityLevel: number;
  opacityExponent: number;
  groupPortion: number;
  selectedOpacityGroups: number[];
  showGroundTruth: boolean;
  visualizationType: 'spider' | 'nyquist';
  view3D: boolean;
  resnormScale: number;
  liveRendering: boolean;
  resolution: string;
  format: 'png' | 'svg' | 'pdf' | 'webp' | 'jpg';
  quality: number;
}

// Default static render settings
export const defaultStaticRenderSettings: StaticRenderSettings = {
  chromaEnabled: true,
  includeLabels: true,
  backgroundColor: 'white',
  opacityLevel: 0.7,
  opacityExponent: 1.0,
  groupPortion: 0.2, // Show best 20% by default
  selectedOpacityGroups: [0, 1, 2, 3], // All quartiles
  showGroundTruth: true,
  visualizationType: 'spider',
  view3D: false,
  resnormScale: 1.0,
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
              checked={settings.chromaEnabled}
              onChange={(e) => updateSetting('chromaEnabled', e.target.checked)}
              className="rounded"
            />
            <span className="text-sm text-gray-700">Color Mode</span>
          </label>

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

        {/* Model Portion */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Model Portion: {Math.round(settings.groupPortion * 100)}% of best fits
          </label>
          <input
            type="range"
            min="0.1"
            max="1.0"
            step="0.1"
            value={settings.groupPortion}
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
                value={settings.opacityLevel}
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
                value={settings.opacityExponent}
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
                  <label className="block text-xs text-gray-600">Rs (Ω)</label>
                  <input
                    type="number"
                    value={groundTruthParams.Rs}
                    onChange={(e) => handleGroundTruthChange('Rs', parseFloat(e.target.value))}
                    className="w-full p-1 text-sm border border-gray-300 rounded"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-600">Ra (Ω)</label>
                  <input
                    type="number"
                    value={groundTruthParams.Ra}
                    onChange={(e) => handleGroundTruthChange('Ra', parseFloat(e.target.value))}
                    className="w-full p-1 text-sm border border-gray-300 rounded"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-600">Ca (µF)</label>
                  <input
                    type="number"
                    value={groundTruthParams.Ca * 1e6}
                    onChange={(e) => handleGroundTruthChange('Ca', parseFloat(e.target.value) / 1e6)}
                    className="w-full p-1 text-sm border border-gray-300 rounded"
                    step="0.1"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-600">Rb (Ω)</label>
                  <input
                    type="number"
                    value={groundTruthParams.Rb}
                    onChange={(e) => handleGroundTruthChange('Rb', parseFloat(e.target.value))}
                    className="w-full p-1 text-sm border border-gray-300 rounded"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-600">Cb (µF)</label>
                  <input
                    type="number"
                    value={groundTruthParams.Cb * 1e6}
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