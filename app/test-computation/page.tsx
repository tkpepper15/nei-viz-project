'use client';

import React, { useState } from 'react';
import { ComputationTest } from '../components/test/ComputationTest';
import { TestAnalysis } from '../components/test/TestAnalysis';
import testConfigLarge from '../../data/sample-outputs/test_1.json';
import testConfigSmall from '../../data/sample-outputs/test_1_small.json';

export default function TestComputationPage() {
  const [useSmallConfig, setUseSmallConfig] = useState(true);
  const activeConfig = useSmallConfig ? testConfigSmall as any : testConfigLarge as any; // eslint-disable-line @typescript-eslint/no-explicit-any

  return (
    <div className="min-h-screen bg-neutral-100 dark:bg-neutral-950 py-8">
      {/* Configuration Selector */}
      <div className="max-w-4xl mx-auto mb-8 p-6 bg-white dark:bg-neutral-900 rounded-lg shadow-lg">
        <h1 className="text-3xl font-bold text-neutral-900 dark:text-white mb-4">WebGPU Computation Testing</h1>
        <div className="space-y-4">
          <p className="text-neutral-600 dark:text-neutral-400">
            Test your circuit computation with different configurations to identify performance limits.
          </p>
          
          <div className="flex space-x-4">
            <button
              onClick={() => setUseSmallConfig(true)}
              className={`px-6 py-3 rounded-lg font-medium transition-colors ${
                useSmallConfig 
                  ? 'bg-green-600 hover:bg-green-700 text-white' 
                  : 'bg-neutral-200 hover:bg-neutral-300 text-neutral-700 dark:bg-neutral-700 dark:hover:bg-neutral-600 dark:text-neutral-300'
              }`}
            >
              ✅ Manageable Config (6^5 = 7,776 params)
            </button>
            
            <button
              onClick={() => setUseSmallConfig(false)}
              className={`px-6 py-3 rounded-lg font-medium transition-colors ${
                !useSmallConfig 
                  ? 'bg-red-600 hover:bg-red-700 text-white' 
                  : 'bg-neutral-200 hover:bg-neutral-300 text-neutral-700 dark:bg-neutral-700 dark:hover:bg-neutral-600 dark:text-neutral-300'
              }`}
            >
              ⚠️  Original Config (16^5 = 1M+ params)
            </button>
          </div>
          
          <div className="text-sm text-neutral-600 dark:text-neutral-400">
            <strong>Currently testing:</strong> {activeConfig.profileName}
            <br />
            <strong>Parameters:</strong> {Math.pow(activeConfig.gridSettings.gridSize, 5).toLocaleString()} combinations
            <br />
            <strong>Memory estimate:</strong> ~{Math.round((Math.pow(activeConfig.gridSettings.gridSize, 5) * activeConfig.gridSettings.numPoints * 32) / (1024 * 1024))} MB
          </div>
        </div>
      </div>

      <div className="space-y-8">
        <TestAnalysis testConfig={activeConfig} />
        <ComputationTest testConfig={activeConfig} />
      </div>
    </div>
  );
}