"use client";

import React, { useState, useEffect } from 'react';

export interface PerformanceSettings {
  // Memory management
  maxMemoryMB: number;
  memoryWarningThreshold: number; // percentage
  memoryEmergencyThreshold: number; // percentage
  
  // Chunk processing
  chunkSize: number;
  gcFrequency: number; // chunks between garbage collection
  midChunkMemoryChecks: boolean;
  
  // Quality and limits
  maxPolygons: number;
  maxRenderTimeMs: number;
  preRenderMemoryCheck: boolean;
  
  // Canvas settings
  canvasWidth: number;
  canvasHeight: number;
  enableHighQualityRendering: boolean;
  
  // Advanced settings
  yieldTimeMs: number;
  memoryCheckFrequency: number;
  autoQualityAdjustment: boolean;
  enableDetailedLogging: boolean;
}

// Detect system memory and create intelligent defaults
const getSystemMemoryMB = (): number => {
  if (typeof navigator !== 'undefined' && 'deviceMemory' in navigator) {
    // navigator.deviceMemory gives memory in GB, convert to MB
    const navigatorWithMemory = navigator as Navigator & { deviceMemory: number };
    return navigatorWithMemory.deviceMemory * 1024;
  }
  
  // Fallback: estimate based on performance.memory if available
  if (typeof window !== 'undefined' && 'performance' in window && 'memory' in performance) {
    const memory = (performance as typeof performance & { memory: { jsHeapSizeLimit: number } }).memory;
    // jsHeapSizeLimit is typically around 2-4GB on most systems, use as rough estimate
    const estimatedSystemMemory = (memory.jsHeapSizeLimit / (1024 * 1024)) * 2; // Assume heap limit is ~50% of available
    return Math.min(estimatedSystemMemory, 32768); // Cap at 32GB for safety
  }
  
  // Conservative fallback for unknown systems
  return 8192; // 8GB default assumption
};

const createIntelligentDefaults = (): PerformanceSettings => {
  const systemMemoryMB = getSystemMemoryMB();
  
  // Use a percentage of system memory for JavaScript operations
  // Conservative: use 10-25% of system memory depending on total amount
  let maxMemoryMB: number;
  let maxPolygons: number;
  let qualityLevel: string;
  
  if (systemMemoryMB >= 16384) { // 16GB+
    maxMemoryMB = Math.floor(systemMemoryMB * 0.25); // Use 25% of system memory
    maxPolygons = 100000;
    qualityLevel = "High-end system detected";
  } else if (systemMemoryMB >= 8192) { // 8GB+
    maxMemoryMB = Math.floor(systemMemoryMB * 0.20); // Use 20% of system memory
    maxPolygons = 50000;
    qualityLevel = "Mid-range system detected";
  } else if (systemMemoryMB >= 4096) { // 4GB+
    maxMemoryMB = Math.floor(systemMemoryMB * 0.15); // Use 15% of system memory
    maxPolygons = 25000;
    qualityLevel = "Entry system detected";
  } else { // <4GB
    maxMemoryMB = Math.floor(systemMemoryMB * 0.10); // Use 10% of system memory
    maxPolygons = 10000;
    qualityLevel = "Limited system detected";
  }
  
  console.log(`System Memory Detection: ${systemMemoryMB}MB total, ${maxMemoryMB}MB allocated for visualization (${qualityLevel})`);
  
  return {
    maxMemoryMB,
    memoryWarningThreshold: 70,
    memoryEmergencyThreshold: 85,
    chunkSize: 25,
    gcFrequency: 3,
    midChunkMemoryChecks: true,
    maxPolygons,
    maxRenderTimeMs: 30000,
    preRenderMemoryCheck: true,
    canvasWidth: 800,
    canvasHeight: 800,
    enableHighQualityRendering: true,
    yieldTimeMs: 10,
    memoryCheckFrequency: 5,
    autoQualityAdjustment: true,
    enableDetailedLogging: false,
  };
};

export const DEFAULT_PERFORMANCE_SETTINGS: PerformanceSettings = createIntelligentDefaults();

interface PerformanceControlsProps {
  settings: PerformanceSettings;
  onChange: (settings: PerformanceSettings) => void;
  currentMemoryUsage?: number;
  gridSize?: number;
  onReset?: () => void;
}

export const PerformanceControls: React.FC<PerformanceControlsProps> = ({
  settings,
  onChange,
  currentMemoryUsage = 0,
  gridSize = 5,
  onReset
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [activeTab, setActiveTab] = useState<'memory' | 'chunks' | 'quality' | 'advanced'>('memory');
  const [isClient, setIsClient] = useState(false);

  // Set client flag after hydration to avoid SSR mismatch
  useEffect(() => {
    setIsClient(true);
  }, []);

  const updateSetting = <K extends keyof PerformanceSettings>(
    key: K,
    value: PerformanceSettings[K]
  ) => {
    onChange({ ...settings, [key]: value });
  };

  const getMemoryStatus = () => {
    // During SSR or before hydration, show safe defaults to prevent mismatch
    if (!isClient || currentMemoryUsage === 0) {
      return { color: 'text-gray-400', text: 'Unknown' };
    }
    
    const warningLevel = settings.maxMemoryMB * (settings.memoryWarningThreshold / 100);
    const emergencyLevel = settings.maxMemoryMB * (settings.memoryEmergencyThreshold / 100);
    
    if (currentMemoryUsage >= emergencyLevel) {
      return { color: 'text-red-400', text: 'Critical' };
    } else if (currentMemoryUsage >= warningLevel) {
      return { color: 'text-yellow-400', text: 'Warning' };
    } else {
      return { color: 'text-green-400', text: 'Safe' };
    }
  };

  const getRecommendedSettings = () => {
    const totalGridPoints = Math.pow(gridSize, 5);
    const systemMemory = getSystemMemoryMB();
    
    if (gridSize >= 8 || totalGridPoints > 32768) {
      return {
        maxMemoryMB: Math.min(Math.floor(systemMemory * 0.15), 1024),
        maxPolygons: 10000,
        chunkSize: 20,
        label: "Ultra Conservative (Grid 8+)"
      };
    } else if (gridSize >= 6) {
      return {
        maxMemoryMB: Math.min(Math.floor(systemMemory * 0.20), 2048),
        maxPolygons: 25000,
        chunkSize: 30,
        label: "Conservative (Grid 6-7)"
      };
    } else if (gridSize >= 4) {
      return {
        maxMemoryMB: Math.min(Math.floor(systemMemory * 0.25), 3072),
        maxPolygons: 50000,
        chunkSize: 50,
        label: "Balanced (Grid 4-5)"
      };
    } else {
      return {
        maxMemoryMB: Math.min(Math.floor(systemMemory * 0.30), 4096),
        maxPolygons: 100000,
        chunkSize: 75,
        label: "Performance (Grid 2-3)"
      };
    }
  };

  const applyRecommended = () => {
    const recommended = getRecommendedSettings();
    onChange({
      ...settings,
      maxMemoryMB: recommended.maxMemoryMB,
      maxPolygons: recommended.maxPolygons,
      chunkSize: recommended.chunkSize,
    });
  };

  const resetToDefaults = () => {
    onChange(DEFAULT_PERFORMANCE_SETTINGS);
    onReset?.();
  };

  const memoryStatus = getMemoryStatus();
  const recommended = getRecommendedSettings();

  return (
    <div className="border border-neutral-700 rounded overflow-hidden">
      {/* Header */}
      <button 
        className="w-full p-3 bg-neutral-800 text-neutral-200 text-sm font-medium flex items-center justify-between hover:bg-neutral-600"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-3">
          <span>Performance Controls</span>
          <div className="flex items-center gap-2 text-xs">
            <span className="text-neutral-400">Memory:</span>
            <span className={memoryStatus.color}>
              {isClient ? currentMemoryUsage.toFixed(0) : '0'}MB / {settings.maxMemoryMB}MB
            </span>
            <span className={`px-2 py-0.5 rounded text-xs ${memoryStatus.color} bg-neutral-900`}>
              {memoryStatus.text}
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
        <div className="bg-neutral-900">
          {/* System Info & Quick Actions */}
          <div className="p-3 border-b border-neutral-700 bg-neutral-800">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-neutral-400">System Memory: {getSystemMemoryMB().toLocaleString()}MB detected</span>
              <span className="text-xs text-green-400">Using intelligent defaults</span>
            </div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-neutral-400">Quick Settings for Grid {gridSize}</span>
              <span className="text-xs text-blue-400">{recommended.label}</span>
            </div>
            <div className="flex gap-2">
              <button
                onClick={applyRecommended}
                className="px-3 py-1 text-xs bg-blue-600 hover:bg-blue-700 text-white rounded"
              >
                Apply Recommended
              </button>
              <button
                onClick={resetToDefaults}
                className="px-3 py-1 text-xs bg-neutral-700 hover:bg-neutral-600 text-white rounded"
              >
                Reset to Defaults
              </button>
            </div>
          </div>

          {/* Tabs */}
          <div className="flex border-b border-neutral-700">
            {[
              { key: 'memory', label: 'Memory' },
              { key: 'chunks', label: 'Processing' },
              { key: 'quality', label: 'Quality' },
              { key: 'advanced', label: 'Advanced' }
            ].map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key as 'memory' | 'chunks' | 'quality' | 'advanced')}
                className={`flex-1 p-2 text-xs font-medium transition-colors ${
                  activeTab === tab.key
                    ? 'bg-neutral-900 text-white border-b-2 border-blue-500'
                    : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-600'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div className="p-4 space-y-4">
            {activeTab === 'memory' && (
              <div className="space-y-4">
                <h4 className="text-sm font-medium text-neutral-200 mb-3">Memory Management</h4>
                
                {/* Memory Limit */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-xs text-neutral-300">Memory Limit (MB)</label>
                    <span className="text-xs font-mono text-neutral-400">{settings.maxMemoryMB}MB</span>
                  </div>
                  <input
                    type="range"
                    min="100"
                    max="8192"
                    step="50"
                    value={settings.maxMemoryMB}
                    onChange={(e) => updateSetting('maxMemoryMB', parseInt(e.target.value))}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-neutral-500">
                    <span>100MB</span>
                    <span>8192MB (8GB)</span>
                  </div>
                </div>

                {/* Warning Threshold */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-xs text-neutral-300">Warning Threshold</label>
                    <span className="text-xs font-mono text-yellow-400">
                      {settings.memoryWarningThreshold}% ({(settings.maxMemoryMB * settings.memoryWarningThreshold / 100).toFixed(0)}MB)
                    </span>
                  </div>
                  <input
                    type="range"
                    min="50"
                    max="90"
                    step="5"
                    value={settings.memoryWarningThreshold}
                    onChange={(e) => updateSetting('memoryWarningThreshold', parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>

                {/* Emergency Threshold */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-xs text-neutral-300">Emergency Stop Threshold</label>
                    <span className="text-xs font-mono text-red-400">
                      {settings.memoryEmergencyThreshold}% ({(settings.maxMemoryMB * settings.memoryEmergencyThreshold / 100).toFixed(0)}MB)
                    </span>
                  </div>
                  <input
                    type="range"
                    min="60"
                    max="95"
                    step="5"
                    value={settings.memoryEmergencyThreshold}
                    onChange={(e) => updateSetting('memoryEmergencyThreshold', parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>

                {/* Pre-render Check */}
                <div className="flex items-center justify-between">
                  <label className="text-xs text-neutral-300">Pre-render Memory Check</label>
                  <input
                    type="checkbox"
                    checked={settings.preRenderMemoryCheck}
                    onChange={(e) => updateSetting('preRenderMemoryCheck', e.target.checked)}
                    className="rounded"
                  />
                </div>
              </div>
            )}

            {activeTab === 'chunks' && (
              <div className="space-y-4">
                <h4 className="text-sm font-medium text-neutral-200 mb-3">Processing Settings</h4>
                
                {/* Chunk Size */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-xs text-neutral-300">Chunk Size (polygons)</label>
                    <span className="text-xs font-mono text-neutral-400">{settings.chunkSize}</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="100"
                    step="1"
                    value={settings.chunkSize}
                    onChange={(e) => updateSetting('chunkSize', parseInt(e.target.value))}
                    className="w-full"
                  />
                  <div className="text-xs text-neutral-500">
                    Smaller chunks = less memory spikes, but slower processing
                  </div>
                </div>

                {/* GC Frequency */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-xs text-neutral-300">Garbage Collection Frequency</label>
                    <span className="text-xs font-mono text-neutral-400">Every {settings.gcFrequency} chunks</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    step="1"
                    value={settings.gcFrequency}
                    onChange={(e) => updateSetting('gcFrequency', parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>

                {/* Mid-chunk Memory Checks */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-xs text-neutral-300">Mid-chunk Memory Checks</label>
                    <input
                      type="checkbox"
                      checked={settings.midChunkMemoryChecks}
                      onChange={(e) => updateSetting('midChunkMemoryChecks', e.target.checked)}
                      className="rounded"
                    />
                  </div>
                  {settings.midChunkMemoryChecks && (
                    <div className="space-y-2 ml-4">
                      <div className="flex items-center justify-between">
                        <label className="text-xs text-neutral-400">Check every N polygons</label>
                        <span className="text-xs font-mono text-neutral-500">{settings.memoryCheckFrequency}</span>
                      </div>
                      <input
                        type="range"
                        min="1"
                        max="10"
                        step="1"
                        value={settings.memoryCheckFrequency}
                        onChange={(e) => updateSetting('memoryCheckFrequency', parseInt(e.target.value))}
                        className="w-full"
                      />
                    </div>
                  )}
                </div>

                {/* Yield Time */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-xs text-neutral-300">Yield Time (ms)</label>
                    <span className="text-xs font-mono text-neutral-400">{settings.yieldTimeMs}ms</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    step="5"
                    value={settings.yieldTimeMs}
                    onChange={(e) => updateSetting('yieldTimeMs', parseInt(e.target.value))}
                    className="w-full"
                  />
                  <div className="text-xs text-neutral-500">
                    Time to yield to browser between chunks (keeps UI responsive)
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'quality' && (
              <div className="space-y-4">
                <h4 className="text-sm font-medium text-neutral-200 mb-3">Quality & Limits</h4>
                
                {/* Max Polygons */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-xs text-neutral-300">Max Polygons</label>
                    <span className="text-xs font-mono text-neutral-400">{settings.maxPolygons.toLocaleString()}</span>
                  </div>
                  <input
                    type="range"
                    min="1000"
                    max="500000"
                    step="1000"
                    value={settings.maxPolygons}
                    onChange={(e) => updateSetting('maxPolygons', parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>

                {/* Max Render Time */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-xs text-neutral-300">Max Render Time</label>
                    <span className="text-xs font-mono text-neutral-400">{(settings.maxRenderTimeMs / 1000).toFixed(1)}s</span>
                  </div>
                  <input
                    type="range"
                    min="1000"
                    max="60000"
                    step="1000"
                    value={settings.maxRenderTimeMs}
                    onChange={(e) => updateSetting('maxRenderTimeMs', parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>

                {/* Canvas Size */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-xs text-neutral-300">Canvas Size</label>
                    <span className="text-xs font-mono text-neutral-400">{settings.canvasWidth} × {settings.canvasHeight}</span>
                  </div>
                  <select
                    value={`${settings.canvasWidth}x${settings.canvasHeight}`}
                    onChange={(e) => {
                      const [width, height] = e.target.value.split('x').map(Number);
                      updateSetting('canvasWidth', width);
                      updateSetting('canvasHeight', height);
                    }}
                    className="w-full p-1 bg-neutral-800 border border-neutral-700 rounded text-xs text-neutral-200"
                  >
                    <option value="400x400">400 × 400 (Fast)</option>
                    <option value="600x600">600 × 600 (Balanced)</option>
                    <option value="800x800">800 × 800 (Default)</option>
                    <option value="1000x1000">1000 × 1000 (High Quality)</option>
                    <option value="1200x1200">1200 × 1200 (Ultra)</option>
                  </select>
                </div>

                {/* High Quality Rendering */}
                <div className="flex items-center justify-between">
                  <label className="text-xs text-neutral-300">High Quality Rendering</label>
                  <input
                    type="checkbox"
                    checked={settings.enableHighQualityRendering}
                    onChange={(e) => updateSetting('enableHighQualityRendering', e.target.checked)}
                    className="rounded"
                  />
                </div>
              </div>
            )}

            {activeTab === 'advanced' && (
              <div className="space-y-4">
                <h4 className="text-sm font-medium text-neutral-200 mb-3">Advanced Settings</h4>
                
                {/* Auto Quality Adjustment */}
                <div className="flex items-center justify-between">
                  <label className="text-xs text-neutral-300">Auto Quality Adjustment</label>
                  <input
                    type="checkbox"
                    checked={settings.autoQualityAdjustment}
                    onChange={(e) => updateSetting('autoQualityAdjustment', e.target.checked)}
                    className="rounded"
                  />
                </div>

                {/* Detailed Logging */}
                <div className="flex items-center justify-between">
                  <label className="text-xs text-neutral-300">Detailed Logging</label>
                  <input
                    type="checkbox"
                    checked={settings.enableDetailedLogging}
                    onChange={(e) => updateSetting('enableDetailedLogging', e.target.checked)}
                    className="rounded"
                  />
                </div>

                {/* Performance Presets */}
                <div className="space-y-2">
                  <label className="text-xs text-neutral-300">Performance Presets</label>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      onClick={() => onChange({
                        ...settings,
                        maxMemoryMB: 200,
                        chunkSize: 10,
                        maxPolygons: 5000,
                        gcFrequency: 1,
                        yieldTimeMs: 50,
                      })}
                      className="px-2 py-1 text-xs bg-red-800 hover:bg-red-700 text-white rounded"
                    >
                      Ultra Safe
                    </button>
                    <button
                      onClick={() => onChange({
                        ...settings,
                        maxMemoryMB: 1024,
                        chunkSize: 25,
                        maxPolygons: 25000,
                        gcFrequency: 2,
                        yieldTimeMs: 20,
                      })}
                      className="px-2 py-1 text-xs bg-yellow-700 hover:bg-yellow-600 text-white rounded"
                    >
                      Balanced
                    </button>
                    <button
                      onClick={() => onChange({
                        ...settings,
                        maxMemoryMB: 2048,
                        chunkSize: 50,
                        maxPolygons: 100000,
                        gcFrequency: 3,
                        yieldTimeMs: 10,
                      })}
                      className="px-2 py-1 text-xs bg-green-700 hover:bg-green-600 text-white rounded"
                    >
                      Performance
                    </button>
                    <button
                      onClick={() => onChange({
                        ...settings,
                        maxMemoryMB: 4096,
                        chunkSize: 100,
                        maxPolygons: 500000,
                        gcFrequency: 5,
                        yieldTimeMs: 5,
                      })}
                      className="px-2 py-1 text-xs bg-blue-700 hover:bg-blue-600 text-white rounded"
                    >
                      Maximum
                    </button>
                  </div>
                </div>

                {/* Current Settings Summary */}
                <div className="mt-4 p-3 bg-neutral-800 rounded">
                  <div className="text-xs text-neutral-400 mb-2">Current Configuration:</div>
                  <div className="grid grid-cols-2 gap-2 text-xs font-mono text-neutral-300">
                    <div>Memory: {settings.maxMemoryMB}MB</div>
                    <div>Chunks: {settings.chunkSize}</div>
                    <div>Polygons: {settings.maxPolygons.toLocaleString()}</div>
                    <div>Canvas: {settings.canvasWidth}x{settings.canvasHeight}</div>
                    <div>GC: Every {settings.gcFrequency}</div>
                    <div>Yield: {settings.yieldTimeMs}ms</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}; 