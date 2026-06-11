"use client";

import React, { useState } from 'react';
import * as Switch from '@radix-ui/react-switch';
import { ImportResultsDialog } from './ImportResultsDialog';
import { ExportedResultsMetadata } from '../utils/fileExport';
import { BackendMeshPoint } from '../types';

export interface PerformanceSettings {
  autoOptimize: boolean;
  performanceWarnings: boolean;
  useSymmetricGrid: boolean;
  useParallelProcessing: boolean;
  enableSmartBatching: boolean;
  enableHighQualityRendering: boolean;
  autoQualityAdjustment: boolean;
  enableDetailedLogging: boolean;
  enableProgressiveRendering: boolean;
  enableHighParameterMode: boolean;
  highParameterFilePath: string;
  topModelsToKeep: number;
  autoExportThreshold: number;
  enableResultImport: boolean;
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

const createOptimizedDefaults = (): PerformanceSettings => ({
  autoOptimize: true,
  performanceWarnings: true,
  useSymmetricGrid: true,
  useParallelProcessing: true,
  enableSmartBatching: true,
  enableHighQualityRendering: true,
  autoQualityAdjustment: true,
  enableDetailedLogging: false,
  enableProgressiveRendering: true,
  enableHighParameterMode: false,
  highParameterFilePath: 'circuit_results',
  topModelsToKeep: 5000,
  autoExportThreshold: 100000,
  enableResultImport: true,
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
});

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
      useSymmetricGrid: enabled,
      useParallelProcessing: enabled,
      enableSmartBatching: enabled,
      enableHighQualityRendering: enabled,
      autoQualityAdjustment: enabled,
      enableDetailedLogging: false,
      enableProgressiveRendering: enabled
    });
  };

  const handleWarningsChange = (enabled: boolean) => {
    onChange({ ...settings, performanceWarnings: enabled });
  };

  const handleHighParameterModeChange = (enabled: boolean) => {
    onChange({
      ...settings,
      enableHighParameterMode: enabled,
      enableProgressiveRendering: enabled,
      enableSmartBatching: enabled,
    });
  };

  const handleImportResults = (results: BackendMeshPoint[], metadata: ExportedResultsMetadata) => {
    if (onImportResults) {
      onImportResults(results, metadata);
    }
    setIsImportDialogOpen(false);
  };

  const getPerformanceStatus = () => {
    const totalGridPoints = Math.pow(gridSize, 5);
    if (totalGridPoints > 8000) return { color: 'text-neutral-400', text: 'hi', dotColor: 'bg-neutral-500' };
    if (totalGridPoints > 3000) return { color: 'text-neutral-400', text: 'med', dotColor: 'bg-neutral-500' };
    return { color: 'text-neutral-400', text: 'lo', dotColor: 'bg-neutral-500' };
  };

  const performanceStatus = getPerformanceStatus();

  return (
    <div className="border border-neutral-800 rounded overflow-hidden">
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
          className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isExpanded && (
        <div className="bg-neutral-900 p-4 space-y-4">
          {/* Auto Optimize */}
          <div className="flex items-center justify-between p-3 bg-neutral-800 rounded border border-neutral-800">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-sm font-medium text-neutral-200">Auto Optimize</span>
                <span className="px-1.5 py-0.5 text-xs bg-neutral-700 text-neutral-300 rounded border border-border">Recommended</span>
              </div>
              <p className="text-xs text-neutral-400">
                Enable all performance optimizations including symmetric grid computation, parallel processing, and smart batching.
              </p>
            </div>
            <Switch.Root
              checked={settings.autoOptimize}
              onCheckedChange={handleAutoOptimizeChange}
              className="flex-shrink-0 ml-3 w-11 h-6 bg-neutral-600 rounded-full relative data-[state=checked]:bg-primary transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-primary cursor-pointer"
            >
              <Switch.Thumb className="block w-5 h-5 bg-white rounded-full shadow transition-transform translate-x-0.5 data-[state=checked]:translate-x-[22px]" />
            </Switch.Root>
          </div>

          {/* High Parameter Discovery */}
          <div className="flex items-center justify-between p-3 bg-neutral-800 rounded border border-neutral-800">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-sm font-medium text-neutral-200">High Parameter Discovery</span>
                <span className="px-1.5 py-0.5 text-xs bg-neutral-700 text-neutral-300 rounded border border-border">Large Grids</span>
              </div>
              <p className="text-xs text-neutral-400">
                Stream results to files for massive parameter grids (100K+ models). Prevents browser crashes and enables unlimited exploration.
              </p>
              {settings.enableHighParameterMode && (
                <div className="mt-2 p-2 bg-neutral-800/50 border border-border rounded text-xs text-neutral-400">
                  File mode active: Top {settings.topModelsToKeep} models kept in memory, rest streamed to files
                </div>
              )}
            </div>
            <Switch.Root
              checked={settings.enableHighParameterMode}
              onCheckedChange={handleHighParameterModeChange}
              className="flex-shrink-0 ml-3 w-11 h-6 bg-neutral-600 rounded-full relative data-[state=checked]:bg-primary transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-primary cursor-pointer"
            >
              <Switch.Thumb className="block w-5 h-5 bg-white rounded-full shadow transition-transform translate-x-0.5 data-[state=checked]:translate-x-[22px]" />
            </Switch.Root>
          </div>

          {/* Import Results */}
          {settings.enableHighParameterMode && settings.enableResultImport && onImportResults && (
            <div className="p-3 bg-neutral-800 rounded border border-neutral-800">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm font-medium text-neutral-200">Resume Analysis</span>
                    <span className="px-1.5 py-0.5 text-xs bg-neutral-700 text-neutral-300 rounded border border-border">Import</span>
                  </div>
                  <p className="text-xs text-neutral-400 mb-3">
                    Import previously exported results to continue your analysis with saved top models and metadata.
                  </p>
                </div>
              </div>
              <button
                onClick={() => setIsImportDialogOpen(true)}
                className="w-full bg-neutral-700 text-neutral-200 px-4 py-2 rounded border border-border hover:bg-neutral-600 transition-colors text-sm font-medium flex items-center justify-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10" />
                </svg>
                Import Results
              </button>
            </div>
          )}

          {/* Performance Warnings */}
          <div className="flex items-center justify-between p-3 bg-neutral-800 rounded border border-neutral-800">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-sm font-medium text-neutral-200">Performance Warnings</span>
              </div>
              <p className="text-xs text-neutral-400">
                Show warnings and tips for large computations and performance bottlenecks.
              </p>
            </div>
            <Switch.Root
              checked={settings.performanceWarnings}
              onCheckedChange={handleWarningsChange}
              className="flex-shrink-0 ml-3 w-11 h-6 bg-neutral-600 rounded-full relative data-[state=checked]:bg-primary transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-primary cursor-pointer"
            >
              <Switch.Thumb className="block w-5 h-5 bg-white rounded-full shadow transition-transform translate-x-0.5 data-[state=checked]:translate-x-[22px]" />
            </Switch.Root>
          </div>

          {/* Performance Status */}
          {settings.performanceWarnings && (
            <div className="p-3 bg-neutral-800 rounded border border-neutral-800">
              <div className="text-xs font-medium text-neutral-300 mb-2">Performance Status</div>
              {settings.autoOptimize ? (
                <p className="text-xs text-neutral-300">
                  All optimizations enabled. Expected ~50% faster computation with symmetric grid optimization.
                </p>
              ) : (
                <p className="text-xs text-neutral-400">
                  Optimizations disabled. Enable Auto Optimize for best performance on large grids.
                </p>
              )}
            </div>
          )}
        </div>
      )}

      <ImportResultsDialog
        isOpen={isImportDialogOpen}
        onClose={() => setIsImportDialogOpen(false)}
        onImport={handleImportResults}
      />
    </div>
  );
};
