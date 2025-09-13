/**
 * Advanced Optimization Controls
 * UI for configuring the 3-stage optimized compute pipeline
 */

import React, { useState, useCallback } from 'react';
import { OptimizedPipelineConfig } from '../utils/optimizedComputePipeline';
import { OptimizedComputeManagerConfig } from '../utils/optimizedComputeManager';

interface OptimizationControlsProps {
  config: OptimizedComputeManagerConfig;
  onConfigChange: (config: Partial<OptimizedComputeManagerConfig>) => void;
  isComputing?: boolean;
  lastOptimizationStats?: {
    lastRun?: {
      timestamp: number;
      gridSize: number;
      totalParams: number;
      finalCandidates: number;
      reductionRatio: number;
      processingTime: number;
      stageBreakdown: {
        stage1: number;
        stage2: number;
        stage3: number;
      };
    };
  };
}

export const OptimizationControls: React.FC<OptimizationControlsProps> = ({
  config,
  onConfigChange,
  isComputing = false,
  lastOptimizationStats
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showStats, setShowStats] = useState(false);

  const handleToggleOptimization = useCallback(() => {
    onConfigChange({
      enableOptimizedPipeline: !config.enableOptimizedPipeline
    });
  }, [config.enableOptimizedPipeline, onConfigChange]);

  const handleThresholdChange = useCallback((value: number) => {
    onConfigChange({
      optimizationThreshold: value
    });
  }, [onConfigChange]);

  const handlePipelineConfigChange = useCallback((
    key: keyof OptimizedPipelineConfig,
    value: number | boolean
  ) => {
    onConfigChange({
      pipelineConfig: {
        ...config.pipelineConfig,
        [key]: value
      }
    });
  }, [config.pipelineConfig, onConfigChange]);

  const formatNumber = (num: number) => {
    if (num > 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num > 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  const getOptimizationStatusColor = () => {
    if (!config.enableOptimizedPipeline) return 'text-gray-500';
    if (isComputing) return 'text-blue-400';
    return 'text-green-400';
  };

  const getOptimizationStatusText = () => {
    if (!config.enableOptimizedPipeline) return 'Disabled';
    if (isComputing) return 'Computing...';
    return 'Ready';
  };

  return (
    <div className="space-y-4 p-4 bg-neutral-800 rounded-lg border border-neutral-600">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <h3 className="text-lg font-semibold text-white">
            🚀 Advanced Optimization
          </h3>
          <div className={`text-sm ${getOptimizationStatusColor()}`}>
            {getOptimizationStatusText()}
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          {lastOptimizationStats && (
            <button
              onClick={() => setShowStats(!showStats)}
              className="px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
              title="Show optimization statistics"
            >
              Stats
            </button>
          )}
          
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="px-3 py-1 text-xs bg-neutral-600 text-white rounded hover:bg-neutral-700 transition-colors"
          >
            {showAdvanced ? 'Simple' : 'Advanced'}
          </button>
        </div>
      </div>

      {/* Main Toggle */}
      <div className="flex items-center justify-between p-3 bg-neutral-700 rounded">
        <div>
          <label className="text-white font-medium">Enable 3-Stage Pipeline</label>
          <div className="text-sm text-neutral-300 mt-1">
            Fingerprinting → Coarse SSR → Full SSR optimization
          </div>
          <div className="text-xs text-neutral-400 mt-1">
            Recommended for grid sizes &gt; {formatNumber(config.optimizationThreshold)} parameters
          </div>
        </div>
        
        <button
          onClick={handleToggleOptimization}
          disabled={isComputing}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-neutral-800 ${
            config.enableOptimizedPipeline ? 'bg-blue-600' : 'bg-neutral-600'
          } ${isComputing ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
              config.enableOptimizedPipeline ? 'translate-x-6' : 'translate-x-1'
            }`}
          />
        </button>
      </div>

      {/* Threshold Configuration */}
      {config.enableOptimizedPipeline && (
        <div className="space-y-3">
          <div>
            <label className="block text-sm font-medium text-white mb-2">
              Optimization Threshold: {formatNumber(config.optimizationThreshold)} parameters
            </label>
            <input
              type="range"
              min={1000}
              max={100000}
              step={1000}
              value={config.optimizationThreshold}
              onChange={(e) => handleThresholdChange(Number(e.target.value))}
              disabled={isComputing}
              className="w-full h-2 bg-neutral-600 rounded-lg appearance-none cursor-pointer slider-thumb:appearance-none slider-thumb:w-4 slider-thumb:h-4 slider-thumb:rounded-full slider-thumb:bg-blue-500"
            />
            <div className="flex justify-between text-xs text-neutral-400 mt-1">
              <span>1K</span>
              <span>100K</span>
            </div>
          </div>
          
          <div className="text-xs text-neutral-400">
            Use optimization when parameter count exceeds this threshold
          </div>
        </div>
      )}

      {/* Advanced Settings */}
      {showAdvanced && config.enableOptimizedPipeline && (
        <div className="space-y-4 border-t border-neutral-600 pt-4">
          <h4 className="text-md font-medium text-white">Pipeline Configuration</h4>
          
          <div className="grid grid-cols-2 gap-4">
            {/* Stage Frequencies */}
            <div>
              <label className="block text-sm font-medium text-white mb-1">
                Fingerprint Frequencies
              </label>
              <input
                type="number"
                min={6}
                max={12}
                value={config.pipelineConfig.fingerprintFrequencies}
                onChange={(e) => handlePipelineConfigChange('fingerprintFrequencies', Number(e.target.value))}
                disabled={isComputing}
                className="w-full px-3 py-2 bg-neutral-700 border border-neutral-600 rounded text-white text-sm focus:border-blue-500 focus:outline-none"
              />
              <div className="text-xs text-neutral-400 mt-1">6-12 frequencies for spectral grouping</div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-white mb-1">
                Full SSR Frequencies  
              </label>
              <input
                type="number"
                min={20}
                max={40}
                value={config.pipelineConfig.fullFrequencies}
                onChange={(e) => handlePipelineConfigChange('fullFrequencies', Number(e.target.value))}
                disabled={isComputing}
                className="w-full px-3 py-2 bg-neutral-700 border border-neutral-600 rounded text-white text-sm focus:border-blue-500 focus:outline-none"
              />
              <div className="text-xs text-neutral-400 mt-1">20-40 frequencies for final SSR</div>
            </div>
            
            {/* Result Limits */}
            <div>
              <label className="block text-sm font-medium text-white mb-1">
                Coarse Survivors
              </label>
              <input
                type="number"
                min={1000}
                max={10000}
                step={500}
                value={config.pipelineConfig.topMSurvivors}
                onChange={(e) => handlePipelineConfigChange('topMSurvivors', Number(e.target.value))}
                disabled={isComputing}
                className="w-full px-3 py-2 bg-neutral-700 border border-neutral-600 rounded text-white text-sm focus:border-blue-500 focus:outline-none"
              />
              <div className="text-xs text-neutral-400 mt-1">Candidates surviving Stage 2</div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-white mb-1">
                Final Top-K
              </label>
              <input
                type="number"
                min={100}
                max={2000}
                step={100}
                value={config.pipelineConfig.finalTopK}
                onChange={(e) => handlePipelineConfigChange('finalTopK', Number(e.target.value))}
                disabled={isComputing}
                className="w-full px-3 py-2 bg-neutral-700 border border-neutral-600 rounded text-white text-sm focus:border-blue-500 focus:outline-none"
              />
              <div className="text-xs text-neutral-400 mt-1">Final candidates returned</div>
            </div>

            {/* Memory Settings */}
            <div>
              <label className="block text-sm font-medium text-white mb-1">
                Chunk Size
              </label>
              <input
                type="number"
                min={5000}
                max={50000}
                step={5000}
                value={config.pipelineConfig.chunkSize}
                onChange={(e) => handlePipelineConfigChange('chunkSize', Number(e.target.value))}
                disabled={isComputing}
                className="w-full px-3 py-2 bg-neutral-700 border border-neutral-600 rounded text-white text-sm focus:border-blue-500 focus:outline-none"
              />
              <div className="text-xs text-neutral-400 mt-1">Parameters per processing chunk</div>
            </div>

            <div>
              <label className="block text-sm font-medium text-white mb-1">
                Tolerance for Ties (%)
              </label>
              <input
                type="number"
                min={0.1}
                max={5.0}
                step={0.1}
                value={(config.pipelineConfig.toleranceForTies || 0.01) * 100}
                onChange={(e) => handlePipelineConfigChange('toleranceForTies', Number(e.target.value) / 100)}
                disabled={isComputing}
                className="w-full px-3 py-2 bg-neutral-700 border border-neutral-600 rounded text-white text-sm focus:border-blue-500 focus:outline-none"
              />
              <div className="text-xs text-neutral-400 mt-1">Include near-ties within this %</div>
            </div>
          </div>

          {/* Advanced Options */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium text-white">Local Optimization</label>
                <div className="text-xs text-neutral-400">Nelder-Mead refinement on final candidates</div>
              </div>
              <button
                onClick={() => handlePipelineConfigChange('enableLocalOptimization', !config.pipelineConfig.enableLocalOptimization)}
                disabled={isComputing}
                className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-neutral-800 ${
                  config.pipelineConfig.enableLocalOptimization ? 'bg-blue-600' : 'bg-neutral-600'
                } ${isComputing ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                <span
                  className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform ${
                    config.pipelineConfig.enableLocalOptimization ? 'translate-x-5' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium text-white">Fallback to Original</label>
                <div className="text-xs text-neutral-400">Use original pipeline if optimization fails</div>
              </div>
              <button
                onClick={() => onConfigChange({ fallbackToOriginal: !config.fallbackToOriginal })}
                disabled={isComputing}
                className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-neutral-800 ${
                  config.fallbackToOriginal ? 'bg-blue-600' : 'bg-neutral-600'
                } ${isComputing ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                <span
                  className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform ${
                    config.fallbackToOriginal ? 'translate-x-5' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Optimization Statistics */}
      {showStats && lastOptimizationStats?.lastRun && (
        <div className="border-t border-neutral-600 pt-4">
          <h4 className="text-md font-medium text-white mb-3">Last Optimization Run</h4>
          
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-neutral-300">Grid Size:</span>
                <span className="text-white">{lastOptimizationStats.lastRun.gridSize}^5</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-300">Total Parameters:</span>
                <span className="text-white">{formatNumber(lastOptimizationStats.lastRun.totalParams)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-300">Final Candidates:</span>
                <span className="text-green-400">{lastOptimizationStats.lastRun.finalCandidates}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-300">Reduction Ratio:</span>
                <span className="text-blue-400">{lastOptimizationStats.lastRun.reductionRatio.toFixed(0)}x</span>
              </div>
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-neutral-300">Total Time:</span>
                <span className="text-white">{(lastOptimizationStats.lastRun.processingTime / 1000).toFixed(1)}s</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-300">Stage 1:</span>
                <span className="text-yellow-400">{(lastOptimizationStats.lastRun.stageBreakdown.stage1 / 1000).toFixed(1)}s</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-300">Stage 2:</span>
                <span className="text-yellow-400">{(lastOptimizationStats.lastRun.stageBreakdown.stage2 / 1000).toFixed(1)}s</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-300">Stage 3:</span>
                <span className="text-yellow-400">{(lastOptimizationStats.lastRun.stageBreakdown.stage3 / 1000).toFixed(1)}s</span>
              </div>
            </div>
          </div>
          
          <div className="mt-3 p-2 bg-green-900/20 border border-green-500/30 rounded text-sm">
            <span className="text-green-400">✓ </span>
            <span className="text-white">
              Processed {formatNumber(lastOptimizationStats.lastRun.totalParams)} parameters 
              {lastOptimizationStats.lastRun.reductionRatio.toFixed(0)}x faster than brute force
            </span>
          </div>
        </div>
      )}
    </div>
  );
};