'use client';

import React, { useState, useCallback } from 'react';
import { useWorkerManager } from '../circuit-simulator/utils/workerManager';
import { useHybridComputeManager } from '../circuit-simulator/utils/hybridComputeManager';
import { ExtendedPerformanceSettings, DEFAULT_GPU_SETTINGS, DEFAULT_CPU_SETTINGS } from '../circuit-simulator/types/gpuSettings';
import { DEFAULT_PERFORMANCE_SETTINGS } from '../circuit-simulator/controls/PerformanceControls';
import { ResnormConfig, ResnormMethod } from '../circuit-simulator/utils/resnorm';
import { CircuitParameters } from '../circuit-simulator/types/parameters';

interface ComputationTestProps {
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

export function ComputationTest({ testConfig }: ComputationTestProps) {
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState<{ results: unknown[]; usedGPU: boolean; benchmarkData?: Record<string, unknown> } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);

  const cpuWorkerManager = useWorkerManager();
  const hybridComputeManager = useHybridComputeManager(cpuWorkerManager);

  const addLog = useCallback((message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev, `[${timestamp}] ${message}`]);
    console.log(`[ComputationTest] ${message}`);
  }, []);

  const runTest = useCallback(async () => {
    setIsRunning(true);
    setError(null);
    setResults(null);
    setLogs([]);

    try {
      addLog('Starting computation test...');
      addLog(`Profile: ${testConfig.profileName}`);
      addLog(`Grid Size: ${testConfig.gridSettings.gridSize}x${testConfig.gridSettings.gridSize}`);
      addLog(`Frequency Range: ${testConfig.gridSettings.minFreq} - ${testConfig.gridSettings.maxFreq} Hz`);
      addLog(`Frequency Points: ${testConfig.gridSettings.numPoints}`);
      addLog(`Circuit Parameters: Rsh=${testConfig.circuitParameters.Rsh}, Ra=${testConfig.circuitParameters.Ra}, Ca=${testConfig.circuitParameters.Ca}, Rb=${testConfig.circuitParameters.Rb}, Cb=${testConfig.circuitParameters.Cb}`);

      // Calculate total computation points
      const totalPoints = Math.pow(testConfig.gridSettings.gridSize, 5);
      addLog(`Total parameter combinations: ${totalPoints.toLocaleString()}`);

      if (totalPoints > 1000000) {
        addLog('⚠️  WARNING: Large computation detected - this may take a while or hit memory limits');
      }

      // Extended performance settings for testing
      const extendedSettings: ExtendedPerformanceSettings = {
        useSymmetricGrid: true,
        maxComputationResults: Math.min(5000, totalPoints), // Limit results
        gpuAcceleration: {
          ...DEFAULT_GPU_SETTINGS,
          enabled: true, // Enable GPU for testing
          fallbackToCPU: true // Allow fallback
        },
        cpuSettings: DEFAULT_CPU_SETTINGS
      };

      // Resnorm configuration
      const resnormConfig: ResnormConfig = {
        method: ResnormMethod.SSR,
        useRangeAmplification: false,
        useFrequencyWeighting: true
      };

      addLog('Initializing computation with hybrid manager...');
      addLog(`Extended settings: GPU=${extendedSettings.gpuAcceleration.enabled}, CPU cores=${extendedSettings.cpuSettings.maxWorkers}`);
      addLog(`Max results limit: ${extendedSettings.maxComputationResults}`);

      // Check if WebGPU is available
      try {
        const gpuCapabilities = await hybridComputeManager.getGPUCapabilities();
        addLog(`WebGPU status: ${gpuCapabilities.supported ? 'Available' : 'Not available'}`);
        if (gpuCapabilities.supported) {
          addLog(`GPU device: ${gpuCapabilities.vendor} ${gpuCapabilities.deviceType}`);
        }
      } catch (gpuError) {
        addLog(`GPU detection error: ${gpuError}`);
      }

      const startTime = performance.now();
      
      const result = await hybridComputeManager.computeGridHybrid(
        testConfig.circuitParameters,
        testConfig.gridSettings.gridSize,
        testConfig.gridSettings.minFreq,
        testConfig.gridSettings.maxFreq,
        testConfig.gridSettings.numPoints,
        DEFAULT_PERFORMANCE_SETTINGS,
        extendedSettings,
        resnormConfig,
        (progress) => {
          addLog(`Progress: ${progress.phase} - ${Math.round(progress.overallProgress)}% (${progress.message})`);
        },
        (error) => {
          addLog(`ERROR: ${error}`);
          setError(error);
        },
        extendedSettings.maxComputationResults
      );

      const endTime = performance.now();
      const totalTime = endTime - startTime;

      addLog(`✅ Computation completed successfully!`);
      addLog(`Used ${result.usedGPU ? 'GPU' : 'CPU'} acceleration`);
      addLog(`Total time: ${(totalTime / 1000).toFixed(2)}s`);
      addLog(`Results: ${result.results.length} parameter sets`);

      if (result.benchmarkData) {
        addLog(`Benchmark - Parameters/sec: ${result.benchmarkData.parametersPerSecond.toFixed(0)}`);
        addLog(`Benchmark - Memory used: ${(result.benchmarkData.memoryUsed / 1024 / 1024).toFixed(1)}MB`);
      }

      // Sample some results for display
      const sampleResults = result.results.slice(0, 5);
      addLog(`Sample results (top 5 by resnorm):`);
      sampleResults.forEach((res, i) => {
        addLog(`  ${i + 1}. Resnorm: ${res.resnorm.toFixed(6)}, Rsh: ${res.parameters.Rsh.toFixed(1)}, Ra: ${res.parameters.Ra.toFixed(1)}`);
      });

      setResults(result);

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : String(err);
      addLog(`❌ Computation failed: ${errorMessage}`);
      setError(errorMessage);
    } finally {
      setIsRunning(false);
    }
  }, [testConfig, hybridComputeManager, addLog]);

  return (
    <div className="max-w-4xl mx-auto p-6 bg-white dark:bg-neutral-900 rounded-lg shadow-lg">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-neutral-900 dark:text-white mb-2">Computation Test</h2>
        <p className="text-neutral-600 dark:text-neutral-400">
          Testing configuration: <span className="font-mono text-blue-600 dark:text-blue-400">{testConfig.profileName}</span>
        </p>
      </div>

      <div className="mb-6">
        <button
          onClick={runTest}
          disabled={isRunning}
          className={`px-6 py-3 rounded-lg font-medium text-white transition-colors ${
            isRunning 
              ? 'bg-gray-500 cursor-not-allowed' 
              : 'bg-blue-600 hover:bg-blue-700'
          }`}
        >
          {isRunning ? 'Running Test...' : 'Run Computation Test'}
        </button>
      </div>

      {error && (
        <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
          <h3 className="text-red-800 dark:text-red-200 font-medium mb-2">Error</h3>
          <p className="text-red-700 dark:text-red-300 text-sm font-mono">{error}</p>
        </div>
      )}

      {results && (
        <div className="mb-6 p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
          <h3 className="text-green-800 dark:text-green-200 font-medium mb-2">Success</h3>
          <p className="text-green-700 dark:text-green-300 text-sm">
            Computation completed with {results.results.length} results using {results.usedGPU ? 'GPU' : 'CPU'} acceleration
          </p>
        </div>
      )}

      <div className="bg-neutral-50 dark:bg-neutral-800 rounded-lg p-4">
        <h3 className="text-neutral-900 dark:text-white font-medium mb-3">Computation Log</h3>
        <div className="bg-black dark:bg-neutral-900 text-green-400 font-mono text-sm p-4 rounded max-h-96 overflow-y-auto">
          {logs.length === 0 ? (
            <div className="text-neutral-500">No logs yet. Click &quot;Run Computation Test&quot; to start.</div>
          ) : (
            logs.map((log, i) => (
              <div key={i} className="mb-1">{log}</div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}