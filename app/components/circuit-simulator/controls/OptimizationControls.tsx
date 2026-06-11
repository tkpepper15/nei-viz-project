import React, { useState, useCallback } from 'react';
import * as Switch from '@radix-ui/react-switch';
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

  const getOptimizationStatusText = () => {
    if (!config.enableOptimizedPipeline) return 'Disabled';
    if (isComputing) return 'Computing...';
    return 'Ready';
  };

  return (
    <div className="space-y-4 p-4 bg-neutral-800 rounded border border-border">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <h3 className="text-xs font-medium text-neutral-300">
            Advanced Optimization
          </h3>
          <span className="text-xs text-neutral-500">
            {getOptimizationStatusText()}
          </span>
        </div>

        <div className="flex items-center space-x-2">
          {lastOptimizationStats && (
            <button
              onClick={() => setShowStats(!showStats)}
              className="px-2 py-1 text-xs bg-neutral-700 text-neutral-300 rounded border border-border hover:bg-neutral-600 transition-colors"
              title="Show optimization statistics"
            >
              Stats
            </button>
          )}

          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="px-2 py-1 text-xs bg-neutral-700 text-neutral-300 rounded border border-border hover:bg-neutral-600 transition-colors"
          >
            {showAdvanced ? 'Simple' : 'Advanced'}
          </button>
        </div>
      </div>

      {/* Main Toggle */}
      <div className="flex items-center justify-between p-3 bg-neutral-800/50 border border-border rounded">
        <div>
          <label className="text-xs font-medium text-neutral-200">Enable 3-Stage Pipeline</label>
          <div className="text-xs text-neutral-400 mt-0.5">
            Fingerprinting → Coarse SSR → Full SSR optimization
          </div>
          <div className="text-xs text-neutral-500 mt-0.5">
            Recommended for grid sizes &gt; {formatNumber(config.optimizationThreshold)} parameters
          </div>
        </div>

        <Switch.Root
          checked={config.enableOptimizedPipeline}
          onCheckedChange={handleToggleOptimization}
          disabled={isComputing}
          className={`w-11 h-6 bg-neutral-600 rounded-full relative data-[state=checked]:bg-primary transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-primary cursor-pointer ${isComputing ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          <Switch.Thumb className="block w-4 h-4 bg-white rounded-full shadow transition-transform translate-x-1 data-[state=checked]:translate-x-6" />
        </Switch.Root>
      </div>

      {/* Threshold Configuration */}
      {config.enableOptimizedPipeline && (
        <div className="space-y-3">
          <div>
            <label className="block text-xs font-medium text-neutral-300 mb-2">
              Threshold: {formatNumber(config.optimizationThreshold)} parameters
            </label>
            <input
              type="range"
              min={1000}
              max={100000}
              step={1000}
              value={config.optimizationThreshold}
              onChange={(e) => handleThresholdChange(Number(e.target.value))}
              disabled={isComputing}
              className="w-full h-2 bg-neutral-600 rounded appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary"
            />
            <div className="flex justify-between text-xs text-neutral-500 mt-1">
              <span>1K</span>
              <span>100K</span>
            </div>
          </div>

          <div className="text-xs text-neutral-500">
            Use optimization when parameter count exceeds this threshold
          </div>
        </div>
      )}

      {/* Advanced Settings */}
      {showAdvanced && config.enableOptimizedPipeline && (
        <div className="space-y-4 border-t border-border pt-4">
          <h4 className="text-xs font-medium text-neutral-300">Pipeline Configuration</h4>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-medium text-neutral-400 mb-1">
                Fingerprint Frequencies
              </label>
              <input
                type="number"
                min={6}
                max={12}
                value={config.pipelineConfig.fingerprintFrequencies}
                onChange={(e) => handlePipelineConfigChange('fingerprintFrequencies', Number(e.target.value))}
                disabled={isComputing}
                className="w-full px-2 py-1.5 bg-neutral-700 border border-border rounded text-neutral-200 text-xs focus:border-primary focus:outline-none"
              />
              <div className="text-xs text-neutral-500 mt-1">6-12 frequencies for spectral grouping</div>
            </div>

            <div>
              <label className="block text-xs font-medium text-neutral-400 mb-1">
                Full SSR Frequencies
              </label>
              <input
                type="number"
                min={20}
                max={40}
                value={config.pipelineConfig.fullFrequencies}
                onChange={(e) => handlePipelineConfigChange('fullFrequencies', Number(e.target.value))}
                disabled={isComputing}
                className="w-full px-2 py-1.5 bg-neutral-700 border border-border rounded text-neutral-200 text-xs focus:border-primary focus:outline-none"
              />
              <div className="text-xs text-neutral-500 mt-1">20-40 frequencies for final SSR</div>
            </div>

            <div>
              <label className="block text-xs font-medium text-neutral-400 mb-1">
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
                className="w-full px-2 py-1.5 bg-neutral-700 border border-border rounded text-neutral-200 text-xs focus:border-primary focus:outline-none"
              />
              <div className="text-xs text-neutral-500 mt-1">Candidates surviving Stage 2</div>
            </div>

            <div>
              <label className="block text-xs font-medium text-neutral-400 mb-1">
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
                className="w-full px-2 py-1.5 bg-neutral-700 border border-border rounded text-neutral-200 text-xs focus:border-primary focus:outline-none"
              />
              <div className="text-xs text-neutral-500 mt-1">Final candidates returned</div>
            </div>

            <div>
              <label className="block text-xs font-medium text-neutral-400 mb-1">
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
                className="w-full px-2 py-1.5 bg-neutral-700 border border-border rounded text-neutral-200 text-xs focus:border-primary focus:outline-none"
              />
              <div className="text-xs text-neutral-500 mt-1">Parameters per processing chunk</div>
            </div>

            <div>
              <label className="block text-xs font-medium text-neutral-400 mb-1">
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
                className="w-full px-2 py-1.5 bg-neutral-700 border border-border rounded text-neutral-200 text-xs focus:border-primary focus:outline-none"
              />
              <div className="text-xs text-neutral-500 mt-1">Include near-ties within this %</div>
            </div>
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <label className="text-xs font-medium text-neutral-300">Local Optimization</label>
                <div className="text-xs text-neutral-500">Nelder-Mead refinement on final candidates</div>
              </div>
              <Switch.Root
                checked={config.pipelineConfig.enableLocalOptimization}
                onCheckedChange={(checked) => handlePipelineConfigChange('enableLocalOptimization', checked)}
                disabled={isComputing}
                className={`w-9 h-5 bg-neutral-600 rounded-full relative data-[state=checked]:bg-primary transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-primary cursor-pointer ${isComputing ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                <Switch.Thumb className="block w-3 h-3 bg-white rounded-full shadow transition-transform translate-x-1 data-[state=checked]:translate-x-5" />
              </Switch.Root>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <label className="text-xs font-medium text-neutral-300">Fallback to Original</label>
                <div className="text-xs text-neutral-500">Use original pipeline if optimization fails</div>
              </div>
              <Switch.Root
                checked={config.fallbackToOriginal}
                onCheckedChange={(checked) => onConfigChange({ fallbackToOriginal: checked })}
                disabled={isComputing}
                className={`w-9 h-5 bg-neutral-600 rounded-full relative data-[state=checked]:bg-primary transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-primary cursor-pointer ${isComputing ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                <Switch.Thumb className="block w-3 h-3 bg-white rounded-full shadow transition-transform translate-x-1 data-[state=checked]:translate-x-5" />
              </Switch.Root>
            </div>
          </div>
        </div>
      )}

      {/* Optimization Statistics */}
      {showStats && lastOptimizationStats?.lastRun && (
        <div className="border-t border-border pt-4">
          <h4 className="text-xs font-medium text-neutral-300 mb-3">Last Optimization Run</h4>

          <div className="grid grid-cols-2 gap-4 text-xs">
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-neutral-400">Grid Size:</span>
                <span className="text-neutral-200 font-mono">{lastOptimizationStats.lastRun.gridSize}^5</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-400">Total Parameters:</span>
                <span className="text-neutral-200 font-mono">{formatNumber(lastOptimizationStats.lastRun.totalParams)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-400">Final Candidates:</span>
                <span className="text-neutral-200 font-mono">{lastOptimizationStats.lastRun.finalCandidates}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-400">Reduction Ratio:</span>
                <span className="text-neutral-200 font-mono">{lastOptimizationStats.lastRun.reductionRatio.toFixed(0)}x</span>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-neutral-400">Total Time:</span>
                <span className="text-neutral-200 font-mono">{(lastOptimizationStats.lastRun.processingTime / 1000).toFixed(1)}s</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-400">Stage 1:</span>
                <span className="text-neutral-300 font-mono">{(lastOptimizationStats.lastRun.stageBreakdown.stage1 / 1000).toFixed(1)}s</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-400">Stage 2:</span>
                <span className="text-neutral-300 font-mono">{(lastOptimizationStats.lastRun.stageBreakdown.stage2 / 1000).toFixed(1)}s</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-400">Stage 3:</span>
                <span className="text-neutral-300 font-mono">{(lastOptimizationStats.lastRun.stageBreakdown.stage3 / 1000).toFixed(1)}s</span>
              </div>
            </div>
          </div>

          <div className="mt-3 p-2 bg-neutral-800/50 border border-border rounded text-xs text-neutral-400">
            Processed {formatNumber(lastOptimizationStats.lastRun.totalParams)} parameters{' '}
            {lastOptimizationStats.lastRun.reductionRatio.toFixed(0)}x faster than brute force
          </div>
        </div>
      )}
    </div>
  );
};
