"use client";

import React, { useState } from 'react';
import { ImportResultsDialog } from './ImportResultsDialog';
import { ExportedResultsMetadata } from '../utils/fileExport';
import { BackendMeshPoint } from '../types';

export interface PerformanceSettings {
  // User-facing controls
  autoOptimize: boolean; // Master toggle for all optimizations
  performanceWarnings: boolean; // Show performance warnings and tips
  
  // Internal optimization settings (controlled by autoOptimize)
  useSymmetricGrid: boolean; // Skip duplicate Ra/Ca <-> Rb/Cb combinations
  useParallelProcessing: boolean; // Use multiple CPU cores
  enableSmartBatching: boolean; // Optimize batch sizes automatically
  enableHighQualityRendering: boolean;
  autoQualityAdjustment: boolean;
  enableDetailedLogging: boolean;
  enableProgressiveRendering: boolean; // Render results as they come in
  
  // High Parameter Discovery Mode
  enableHighParameterMode: boolean; // Stream results to files for massive grids
  highParameterFilePath: string; // Directory path for result files
  topModelsToKeep: number; // Number of top models to keep in memory
  autoExportThreshold: number; // Auto-enable file mode above this many models
  enableResultImport: boolean; // Allow importing previous results
  
  // Memory and performance limits
  maxMemoryMB: number;
  chunkSize: number;
  maxPolygons: number;
  maxRenderTimeMs: number;
  preRenderMemoryCheck: boolean;
  canvasWidth: number;
  canvasHeight: number;
  gcFrequency: number;
  memoryWarningThreshold: number;
  memoryEmergencyThreshold: number;
  midChunkMemoryChecks: boolean;
  memoryCheckFrequency: number;
  yieldTimeMs: number;
}

const createOptimizedDefaults = (): PerformanceSettings => {
  return {
    autoOptimize: true, // Enable all optimizations by default
    performanceWarnings: true, // Show performance warnings
    useSymmetricGrid: true, // Enable symmetric grid optimization by default
    useParallelProcessing: true, // Enable multi-core processing
    enableSmartBatching: true, // Optimize batch sizes automatically
    enableHighQualityRendering: true,
    autoQualityAdjustment: true,
    enableDetailedLogging: false,
    enableProgressiveRendering: true, // Show results as they compute
    
    // High Parameter Discovery Mode defaults
    enableHighParameterMode: false, // Disabled by default
    highParameterFilePath: 'circuit_results', // Default file prefix
    topModelsToKeep: 5000, // Keep top 5K models in memory
    autoExportThreshold: 100000, // Auto-enable file mode above 100K models
    enableResultImport: true, // Allow importing results
    
    // Memory and performance limits - default values
    maxMemoryMB: 512,
    chunkSize: 1000,
    maxPolygons: 10000,
    maxRenderTimeMs: 5000,
    preRenderMemoryCheck: true,
    canvasWidth: 800,
    canvasHeight: 600,
    gcFrequency: 100,
    memoryWarningThreshold: 400,
    memoryEmergencyThreshold: 480,
    midChunkMemoryChecks: true,
    memoryCheckFrequency: 50,
    yieldTimeMs: 16,
  };
};

export const DEFAULT_PERFORMANCE_SETTINGS: PerformanceSettings = createOptimizedDefaults();

interface PerformanceControlsProps {
  settings: PerformanceSettings;
  onChange: (settings: PerformanceSettings) => void;
  gridSize?: number;
  onImportResults?: (results: BackendMeshPoint[], metadata: ExportedResultsMetadata) => void;
}

export const PerformanceControls: React.FC<PerformanceControlsProps> = ({
  settings,
  onChange,
  gridSize = 5,
  onImportResults
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isImportDialogOpen, setIsImportDialogOpen] = useState(false);

  const handleAutoOptimizeChange = (enabled: boolean) => {
    onChange({
      ...settings,
      autoOptimize: enabled,
      // Update all internal optimizations based on auto optimize setting
      useSymmetricGrid: enabled,
      useParallelProcessing: enabled,
      enableSmartBatching: enabled,
      enableHighQualityRendering: enabled,
      autoQualityAdjustment: enabled,
      enableDetailedLogging: false, // Keep logging off even when auto-optimizing
      enableProgressiveRendering: enabled
    });
  };

  const handleWarningsChange = (enabled: boolean) => {
    onChange({
      ...settings,
      performanceWarnings: enabled,
    });
  };

  const handleHighParameterModeChange = (enabled: boolean) => {
    onChange({
      ...settings,
      enableHighParameterMode: enabled,
      // Auto-adjust related settings when enabling high parameter mode
      enableProgressiveRendering: enabled, // Important for large datasets
      enableSmartBatching: enabled, // Critical for performance
    });
  };

  const handleImportResults = (results: BackendMeshPoint[], metadata: ExportedResultsMetadata) => {
    if (onImportResults) {
      onImportResults(results, metadata);
    }
    setIsImportDialogOpen(false);
  };

  // Calculate performance status based on grid size
  const getPerformanceStatus = () => {
    const totalGridPoints = Math.pow(gridSize, 5);
    if (totalGridPoints > 8000) {
      return { color: 'text-red-400', text: 'Hi', dotColor: 'bg-red-400' };
    }
    if (totalGridPoints > 3000) {
      return { color: 'text-yellow-400', text: 'Med', dotColor: 'bg-yellow-400' };
    }
    return { color: 'text-green-400', text: 'Lo', dotColor: 'bg-green-400' };
  };

  const performanceStatus = getPerformanceStatus();

  return (
    <div className="border border-neutral-700 rounded overflow-hidden">
      {/* Header */}
      <button 
        className="w-full p-3 bg-neutral-800 text-neutral-200 text-sm font-medium flex items-center justify-between hover:bg-neutral-600 transition-colors"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-3">
          <span>Performance</span>
          <div className="flex items-center gap-2 text-xs">
            <div className={`w-2 h-2 rounded-full ${performanceStatus.dotColor}`}></div>
            <span className={`${performanceStatus.color} font-medium`}>{performanceStatus.text}</span>
            <span className="text-neutral-500">•</span>
            <span className="text-neutral-400">{Math.pow(gridSize, 5).toLocaleString()} pts</span>
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
        <div className="bg-neutral-900 p-4 space-y-4">
          {/* Auto Optimize Toggle */}
          <div className="flex items-center justify-between p-3 bg-neutral-800 rounded border border-neutral-700">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-sm font-medium text-neutral-200">Auto Optimize</span>
                <span className="px-2 py-0.5 text-xs bg-green-600 text-white rounded">
                  Recommended
                </span>
              </div>
              <p className="text-xs text-neutral-400">
                Enable all performance optimizations including symmetric grid computation, parallel processing, and smart batching.
              </p>
            </div>
            <div className="flex-shrink-0 ml-3">
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={settings.autoOptimize}
                  onChange={(e) => handleAutoOptimizeChange(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="relative w-11 h-6 bg-neutral-600 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-neutral-600 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-neutral-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-neutral-600 peer-checked:bg-blue-600"></div>
              </label>
            </div>
          </div>

          {/* High Parameter Discovery Mode Toggle */}
          <div className="flex items-center justify-between p-3 bg-neutral-800 rounded border border-neutral-700">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-sm font-medium text-neutral-200">High Parameter Discovery</span>
                <span className="px-2 py-0.5 text-xs bg-purple-600 text-white rounded">
                  Large Grids
                </span>
              </div>
              <p className="text-xs text-neutral-400">
                Stream results to files for massive parameter grids (100K+ models). Prevents browser crashes and enables unlimited exploration.
              </p>
              {settings.enableHighParameterMode && (
                <div className="mt-2 p-2 bg-purple-900/20 border border-purple-600/30 rounded text-xs text-purple-300">
                  ⚡ File mode active: Top {settings.topModelsToKeep} models kept in memory, rest streamed to files
                </div>
              )}
            </div>
            <div className="flex-shrink-0 ml-3">
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={settings.enableHighParameterMode}
                  onChange={(e) => handleHighParameterModeChange(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="relative w-11 h-6 bg-neutral-600 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-purple-300 dark:peer-focus:ring-purple-800 rounded-full peer dark:bg-neutral-600 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-neutral-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-neutral-600 peer-checked:bg-purple-600"></div>
              </label>
            </div>
          </div>

          {/* Import Results Button - Only show when High Parameter Mode is enabled */}
          {settings.enableHighParameterMode && settings.enableResultImport && onImportResults && (
            <div className="p-3 bg-neutral-800 rounded border border-neutral-700">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm font-medium text-neutral-200">Resume Analysis</span>
                    <span className="px-2 py-0.5 text-xs bg-blue-600 text-white rounded">
                      Import
                    </span>
                  </div>
                  <p className="text-xs text-neutral-400 mb-3">
                    Import previously exported results to continue your analysis with saved top models and metadata.
                  </p>
                </div>
              </div>
              <button
                onClick={() => setIsImportDialogOpen(true)}
                className="w-full bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700 transition-colors text-sm font-medium flex items-center justify-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10" />
                </svg>
                Import Results
              </button>
            </div>
          )}

          {/* Performance Warnings Toggle */}
          <div className="flex items-center justify-between p-3 bg-neutral-800 rounded border border-neutral-700">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-sm font-medium text-neutral-200">Performance Warnings</span>
              </div>
              <p className="text-xs text-neutral-400">
                Show warnings and tips for large computations and performance bottlenecks.
              </p>
            </div>
            <div className="flex-shrink-0 ml-3">
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={settings.performanceWarnings}
                  onChange={(e) => handleWarningsChange(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="relative w-11 h-6 bg-neutral-600 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-neutral-600 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-neutral-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-neutral-600 peer-checked:bg-blue-600"></div>
              </label>
            </div>
          </div>


          {/* Performance Status */}
          {settings.performanceWarnings && (
            <div className="p-3 bg-neutral-800 rounded border border-neutral-700">
              <div className="text-xs font-medium text-neutral-300 mb-2">⚡ Performance Status</div>
              {settings.autoOptimize ? (
                <p className="text-xs text-green-400">
                  All optimizations enabled. Expected ~50% faster computation with symmetric grid optimization.
                </p>
              ) : (
                <p className="text-xs text-yellow-400">
                  Optimizations disabled. Enable Auto Optimize for best performance on large grids.
                </p>
              )}
            </div>
          )}
        </div>
      )}
      
      {/* Import Results Dialog */}
      <ImportResultsDialog
        isOpen={isImportDialogOpen}
        onClose={() => setIsImportDialogOpen(false)}
        onImport={handleImportResults}
      />
    </div>
  );
}; 