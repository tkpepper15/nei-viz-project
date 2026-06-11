'use client';

import React, { useState, useCallback } from 'react';
import { CircuitParameters } from '../circuit-simulator/types/parameters';

interface TestAnalysisProps {
  testConfig: {
    profileName: string;
    gridSettings: {
      gridSize: number;
      minFreq: number;
      maxFreq: number;
      numPoints: number;
    };
    circuitParameters: CircuitParameters;
  };
}

export function TestAnalysis({ testConfig }: TestAnalysisProps) {
  const [analysis, setAnalysis] = useState<{
    totalCombinations: number;
    memoryEstimate: number;
    timeEstimate: number;
    recommendations: string[];
    severity: 'safe' | 'warning' | 'danger';
  } | null>(null);

  const analyzeConfiguration = useCallback(() => {
    const { gridSize, numPoints } = testConfig.gridSettings;
    
    // Calculate total parameter combinations (5 parameters)
    const totalCombinations = Math.pow(gridSize, 5);
    
    // Estimate memory usage (rough calculation)
    const bytesPerResult = 
      8 * 4 + // 4 doubles for resnorm + circuit params  
      8 * numPoints * 4 + // spectrum data (freq, real, imag, mag per point)
      100; // overhead
    const memoryEstimate = (totalCombinations * bytesPerResult) / (1024 * 1024); // MB
    
    // Estimate computation time (very rough)
    const operationsPerCombination = numPoints * 50; // rough estimate
    const totalOperations = totalCombinations * operationsPerCombination;
    const operationsPerSecond = 1000000; // conservative estimate
    const timeEstimate = totalOperations / operationsPerSecond; // seconds
    
    const recommendations: string[] = [];
    let severity: 'safe' | 'warning' | 'danger' = 'safe';
    
    // Analysis and recommendations
    if (totalCombinations > 100000) {
      severity = 'danger';
      recommendations.push(`Very large grid: ${totalCombinations.toLocaleString()} combinations may exceed memory limits`);
      recommendations.push(`Consider reducing grid size from ${gridSize} to 8-12 for testing`);
    } else if (totalCombinations > 10000) {
      severity = 'warning';
      recommendations.push(`Large grid: ${totalCombinations.toLocaleString()} combinations - expect longer computation time`);
    }
    
    if (memoryEstimate > 2000) {
      severity = 'danger';
      recommendations.push(`High memory usage: ~${memoryEstimate.toFixed(0)}MB estimated`);
      recommendations.push('GPU computation may fail due to memory limits');
      recommendations.push('Reduce maxComputationResults or grid size');
    } else if (memoryEstimate > 500) {
      if (severity !== 'danger') severity = 'warning';
      recommendations.push(`Moderate memory usage: ~${memoryEstimate.toFixed(0)}MB estimated`);
    }
    
    if (timeEstimate > 300) {
      if (severity !== 'danger') severity = 'warning';
      recommendations.push(`Long computation time: ~${(timeEstimate/60).toFixed(1)} minutes estimated`);
    }
    
    // Check parameter values
    const { Ca, Cb } = testConfig.circuitParameters;
    if (Ca < 1e-12 || Cb < 1e-12) {
      recommendations.push('Very small capacitance values may cause numerical precision issues');
    }
    
    if (recommendations.length === 0) {
      recommendations.push('Configuration looks reasonable for testing');
    }
    
    setAnalysis({
      totalCombinations,
      memoryEstimate,
      timeEstimate,
      recommendations,
      severity
    });
  }, [testConfig]);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'safe': return 'text-green-800 dark:text-green-200 bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800';
      case 'warning': return 'text-yellow-800 dark:text-yellow-200 bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800';
      case 'danger': return 'text-red-800 dark:text-red-200 bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800';
      default: return 'text-gray-800 dark:text-gray-200 bg-gray-50 dark:bg-gray-900/20 border-gray-200 dark:border-gray-800';
    }
  };

  const generateSmallerTestConfig = () => {
    return {
      ...testConfig,
      profileName: testConfig.profileName + ' (Reduced)',
      gridSettings: {
        ...testConfig.gridSettings,
        gridSize: Math.min(8, testConfig.gridSettings.gridSize), // Reduce to max 8
        numPoints: Math.min(50, testConfig.gridSettings.numPoints) // Reduce frequency points
      }
    };
  };

  return (
    <div className="max-w-4xl mx-auto p-6 bg-white dark:bg-neutral-900 rounded-lg shadow-lg">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-neutral-900 dark:text-white mb-2">Configuration Analysis</h2>
        <p className="text-neutral-600 dark:text-neutral-400">
          Analyzing test configuration for potential issues
        </p>
      </div>

      <div className="mb-6">
        <button
          onClick={analyzeConfiguration}
          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
        >
          Analyze Configuration
        </button>
      </div>

      {analysis && (
        <div className="space-y-6">
          {/* Summary */}
          <div className={`p-4 rounded-lg border ${getSeverityColor(analysis.severity)}`}>
            <h3 className="font-medium mb-2">Analysis Summary</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div>
                <span className="font-medium">Parameter Combinations:</span>
                <br />
                <span className="font-mono">{analysis.totalCombinations.toLocaleString()}</span>
              </div>
              <div>
                <span className="font-medium">Memory Estimate:</span>
                <br />
                <span className="font-mono">{analysis.memoryEstimate.toFixed(1)} MB</span>
              </div>
              <div>
                <span className="font-medium">Time Estimate:</span>
                <br />
                <span className="font-mono">
                  {analysis.timeEstimate < 60 
                    ? `${analysis.timeEstimate.toFixed(1)}s` 
                    : `${(analysis.timeEstimate/60).toFixed(1)}m`}
                </span>
              </div>
            </div>
          </div>

          {/* Recommendations */}
          <div className="bg-neutral-50 dark:bg-neutral-800 rounded-lg p-4">
            <h3 className="font-medium mb-3 text-neutral-900 dark:text-white">Recommendations</h3>
            <ul className="space-y-2">
              {analysis.recommendations.map((rec, i) => (
                <li key={i} className="flex items-start space-x-2">
                  <span className="text-blue-600 dark:text-blue-400 mt-1">•</span>
                  <span className="text-neutral-700 dark:text-neutral-300 text-sm">{rec}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Alternative Configuration */}
          {analysis.severity === 'danger' && (
            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
              <h3 className="font-medium mb-3 text-blue-800 dark:text-blue-200">Suggested Test Configuration</h3>
              <div className="bg-neutral-900 text-green-400 font-mono text-sm p-3 rounded overflow-x-auto">
                <pre>{JSON.stringify(generateSmallerTestConfig(), null, 2)}</pre>
              </div>
              <p className="text-blue-700 dark:text-blue-300 text-sm mt-2">
                This reduced configuration should be more manageable for testing while preserving the same parameter relationships.
              </p>
            </div>
          )}

          {/* Parameter Details */}
          <div className="bg-neutral-50 dark:bg-neutral-800 rounded-lg p-4">
            <h3 className="font-medium mb-3 text-neutral-900 dark:text-white">Parameter Details</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <h4 className="font-medium text-neutral-800 dark:text-neutral-200 mb-2">Circuit Parameters</h4>
                <div className="space-y-1 font-mono text-xs">
                  <div>Rsh (Shunt): {testConfig.circuitParameters.Rsh.toLocaleString()} Ω</div>
                  <div>Ra (Branch A): {testConfig.circuitParameters.Ra.toLocaleString()} Ω</div>
                  <div>Ca (Branch A): {testConfig.circuitParameters.Ca.toExponential(2)} F</div>
                  <div>Rb (Branch B): {testConfig.circuitParameters.Rb.toLocaleString()} Ω</div>
                  <div>Cb (Branch B): {testConfig.circuitParameters.Cb.toExponential(2)} F</div>
                </div>
              </div>
              <div>
                <h4 className="font-medium text-neutral-800 dark:text-neutral-200 mb-2">Grid Settings</h4>
                <div className="space-y-1 font-mono text-xs">
                  <div>Grid Size: {testConfig.gridSettings.gridSize} per dimension</div>
                  <div>Freq Range: {testConfig.gridSettings.minFreq} - {testConfig.gridSettings.maxFreq.toLocaleString()} Hz</div>
                  <div>Freq Points: {testConfig.gridSettings.numPoints}</div>
                  <div>Total Combinations: {analysis.totalCombinations.toLocaleString()}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}