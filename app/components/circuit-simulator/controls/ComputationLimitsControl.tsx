import React, { useState, useCallback, useMemo } from 'react';

interface ComputationLimitsConfig {
  maxComputationResults: number;
  maxDisplayResults: number;
  memoryLimitMB: number;
  autoScale: boolean;
  // Master limit - single variable to control all computation limits
  masterLimitPercentage: number; // 0-100, percentage of total possible results
}

interface ComputationLimitsControlProps {
  gridSize: number;
  totalPossibleResults: number;
  currentConfig: ComputationLimitsConfig;
  onConfigChange: (config: ComputationLimitsConfig) => void;
  className?: string;
}

export const ComputationLimitsControl: React.FC<ComputationLimitsControlProps> = ({
  gridSize,
  totalPossibleResults,
  currentConfig,
  onConfigChange,
  className = ''
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Transparent math calculations
  const calculations = useMemo(() => {
    const avgModelSize = 2000; // bytes per model (parameters + spectrum data)
    const totalMemoryMB = (totalPossibleResults * avgModelSize) / (1024 * 1024);

    // Calculate master limit based results
    const masterLimitResults = Math.floor((currentConfig.masterLimitPercentage / 100) * totalPossibleResults);

    // Auto-scaling thresholds
    const memoryThresholds = {
      low: 500,      // Under 500MB - compute all
      medium: 2000,  // Under 2GB - compute 250K max
      high: 8000     // Under 8GB - compute 100K max
    };

    let recommendedComputation: number;
    let recommendedDisplay: number;
    let recommendedMasterPercentage: number;
    let reasoning: string;

    if (totalMemoryMB < memoryThresholds.low) {
      recommendedComputation = totalPossibleResults;
      recommendedDisplay = Math.min(100000, totalPossibleResults);
      recommendedMasterPercentage = 100;
      reasoning = `Low memory usage (${totalMemoryMB.toFixed(0)}MB) - compute all ${totalPossibleResults.toLocaleString()} models`;
    } else if (totalMemoryMB < memoryThresholds.medium) {
      recommendedComputation = Math.min(250000, totalPossibleResults);
      recommendedDisplay = Math.min(50000, recommendedComputation);
      recommendedMasterPercentage = Math.min(100, (250000 / totalPossibleResults) * 100);
      reasoning = `Medium memory usage (${totalMemoryMB.toFixed(0)}MB) - compute top 250K models`;
    } else if (totalMemoryMB < memoryThresholds.high) {
      recommendedComputation = Math.min(100000, totalPossibleResults);
      recommendedDisplay = Math.min(25000, recommendedComputation);
      recommendedMasterPercentage = Math.min(100, (100000 / totalPossibleResults) * 100);
      reasoning = `High memory usage (${totalMemoryMB.toFixed(0)}MB) - compute top 100K models`;
    } else {
      recommendedComputation = Math.min(50000, totalPossibleResults);
      recommendedDisplay = Math.min(10000, recommendedComputation);
      recommendedMasterPercentage = Math.min(100, (50000 / totalPossibleResults) * 100);
      reasoning = `Very high memory usage (${totalMemoryMB.toFixed(0)}MB) - compute top 50K models`;
    }

    return {
      totalMemoryMB,
      recommendedComputation,
      recommendedDisplay,
      recommendedMasterPercentage,
      reasoning,
      masterLimitResults,
      currentMemoryMB: (currentConfig.maxComputationResults * avgModelSize) / (1024 * 1024),
      compressionRatio: currentConfig.maxComputationResults / totalPossibleResults
    };
  }, [totalPossibleResults, currentConfig.maxComputationResults, currentConfig.masterLimitPercentage]);

  const handleAutoScale = useCallback(() => {
    const newConfig: ComputationLimitsConfig = {
      maxComputationResults: calculations.recommendedComputation,
      maxDisplayResults: calculations.recommendedDisplay,
      memoryLimitMB: Math.ceil(calculations.currentMemoryMB),
      autoScale: true,
      masterLimitPercentage: calculations.recommendedMasterPercentage
    };
    onConfigChange(newConfig);
  }, [calculations, onConfigChange]);

  const handleMasterLimitChange = useCallback((percentage: number) => {
    const masterLimitResults = Math.floor((percentage / 100) * totalPossibleResults);
    const newConfig: ComputationLimitsConfig = {
      maxComputationResults: masterLimitResults,
      maxDisplayResults: Math.min(currentConfig.maxDisplayResults, masterLimitResults),
      memoryLimitMB: currentConfig.memoryLimitMB,
      autoScale: false,
      masterLimitPercentage: percentage
    };
    onConfigChange(newConfig);
  }, [totalPossibleResults, currentConfig, onConfigChange]);

  const handleManualChange = useCallback((field: keyof ComputationLimitsConfig, value: number | boolean) => {
    const newConfig = { ...currentConfig, [field]: value, autoScale: false };
    onConfigChange(newConfig);
  }, [currentConfig, onConfigChange]);

  return (
    <div className={`bg-neutral-800 border border-neutral-700 rounded-lg p-4 ${className}`}>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-neutral-200">Computation Limits</h3>
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-xs text-neutral-400 hover:text-neutral-200"
        >
          {showAdvanced ? 'Simple' : 'Advanced'}
        </button>
      </div>

      {/* Grid Analysis */}
      <div className="grid grid-cols-2 gap-4 mb-4 text-xs">
        <div>
          <span className="text-neutral-400">Grid Size:</span>
          <span className="text-neutral-200 ml-2">{gridSize}^5 = {totalPossibleResults.toLocaleString()}</span>
        </div>
        <div>
          <span className="text-neutral-400">Total Memory:</span>
          <span className="text-neutral-200 ml-2">{calculations.totalMemoryMB.toFixed(0)} MB</span>
        </div>
      </div>

      {/* Master Limit Control - Single Variable to Rule Them All */}
      <div className="mb-4 p-3 bg-blue-900/20 border border-blue-700/50 rounded-lg">
        <div className="flex items-center justify-between mb-2">
          <label className="text-sm font-medium text-blue-200">Master Limit</label>
          <span className="text-sm text-blue-300 font-mono">
            {currentConfig.masterLimitPercentage.toFixed(1)}% = {calculations.masterLimitResults.toLocaleString()} models
          </span>
        </div>
        <input
          type="range"
          min="0.1"
          max="100"
          step="0.1"
          value={currentConfig.masterLimitPercentage}
          onChange={(e) => handleMasterLimitChange(parseFloat(e.target.value))}
          className="w-full accent-blue-500"
        />
        <div className="flex justify-between text-xs text-blue-400 mt-1">
          <span>0.1%</span>
          <span className="text-center">Single slider controls ALL computation limits</span>
          <span>100%</span>
        </div>
      </div>

      {/* Current Configuration */}
      <div className="space-y-3">
        <div>
          <div className="flex items-center justify-between mb-1">
            <label className="text-xs text-neutral-400">Computation Limit</label>
            <span className="text-xs text-neutral-300">
              {currentConfig.maxComputationResults.toLocaleString()}
            </span>
          </div>
          <input
            type="range"
            min="1000"
            max={Math.min(500000, totalPossibleResults)}
            step="1000"
            value={currentConfig.maxComputationResults}
            onChange={(e) => handleManualChange('maxComputationResults', parseInt(e.target.value))}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-neutral-500 mt-1">
            <span>1K</span>
            <span>{(currentConfig.maxComputationResults / totalPossibleResults * 100).toFixed(1)}% of total</span>
            <span>{Math.min(500, totalPossibleResults / 1000).toFixed(0)}K</span>
          </div>
        </div>

        <div>
          <div className="flex items-center justify-between mb-1">
            <label className="text-xs text-neutral-400">Display Limit</label>
            <span className="text-xs text-neutral-300">
              {currentConfig.maxDisplayResults.toLocaleString()}
            </span>
          </div>
          <input
            type="range"
            min="100"
            max={Math.min(100000, currentConfig.maxComputationResults)}
            step="100"
            value={currentConfig.maxDisplayResults}
            onChange={(e) => handleManualChange('maxDisplayResults', parseInt(e.target.value))}
            className="w-full"
          />
        </div>
      </div>

      {/* Auto-Scale Recommendation */}
      <div className="mt-4 p-3 bg-neutral-900 rounded border border-neutral-600">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-neutral-400">Recommendation</span>
          <button
            onClick={handleAutoScale}
            className="text-xs px-2 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded"
          >
            Apply Auto-Scale
          </button>
        </div>
        <p className="text-xs text-neutral-300 leading-relaxed">{calculations.reasoning}</p>

        {showAdvanced && (
          <div className="mt-3 space-y-2 text-xs text-neutral-400">
            <div className="flex justify-between">
              <span>Recommended Master Limit:</span>
              <span>{calculations.recommendedMasterPercentage.toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span>Current Master Limit:</span>
              <span>{currentConfig.masterLimitPercentage.toFixed(1)}% = {calculations.masterLimitResults.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span>Recommended Computation:</span>
              <span>{calculations.recommendedComputation.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span>Recommended Display:</span>
              <span>{calculations.recommendedDisplay.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span>Current Memory Usage:</span>
              <span>{calculations.currentMemoryMB.toFixed(0)} MB</span>
            </div>
            <div className="flex justify-between">
              <span>Compression Ratio:</span>
              <span>{(calculations.compressionRatio * 100).toFixed(1)}%</span>
            </div>
          </div>
        )}
      </div>

      {/* Status Indicators */}
      <div className="mt-3 flex items-center gap-2 text-xs">
        <div className={`w-2 h-2 rounded-full ${
          calculations.currentMemoryMB < 500 ? 'bg-green-500' :
          calculations.currentMemoryMB < 2000 ? 'bg-yellow-500' : 'bg-red-500'
        }`} />
        <span className="text-neutral-400">
          {calculations.currentMemoryMB < 500 ? 'Low' :
           calculations.currentMemoryMB < 2000 ? 'Medium' : 'High'} memory usage
        </span>
      </div>
    </div>
  );
};

// Default computation limits based on system capabilities
export const getDefaultComputationLimits = (gridSize: number): ComputationLimitsConfig => {
  const totalPossibleResults = Math.pow(gridSize, 5);
  const avgModelSize = 2000;
  const totalMemoryMB = (totalPossibleResults * avgModelSize) / (1024 * 1024);

  if (totalMemoryMB < 500) {
    return {
      maxComputationResults: totalPossibleResults,
      maxDisplayResults: Math.min(100000, totalPossibleResults),
      memoryLimitMB: Math.ceil(totalMemoryMB),
      autoScale: true,
      masterLimitPercentage: 100
    };
  } else if (totalMemoryMB < 2000) {
    return {
      maxComputationResults: totalPossibleResults, // Compute all for medium memory usage
      maxDisplayResults: 50000,
      memoryLimitMB: 500,
      autoScale: true,
      masterLimitPercentage: 100
    };
  } else {
    return {
      maxComputationResults: totalPossibleResults, // Compute all for higher memory usage
      maxDisplayResults: 25000,
      memoryLimitMB: 200,
      autoScale: true,
      masterLimitPercentage: 100
    };
  }
};

export type { ComputationLimitsConfig };