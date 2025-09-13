/**
 * Serialized Spider Plot Demo
 * ===========================
 * 
 * Demonstrates integration of serialized computation with spider plot rendering.
 * Shows seamless transition from traditional to serialized approach.
 */

"use client";

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { SpiderPlot3D } from '../visualizations/SpiderPlot3D';
import { useAdaptiveComputation } from '../hooks/useSerializedComputation';
import { BackendMeshPoint, ModelSnapshot } from '../types';
import { CircuitParameters } from '../types/parameters';
import { ConfigUtils } from '../utils/configSerializer';
// import { FrequencyUtils } from '../utils/frequencySerializer';

interface SerializedSpiderPlotDemoProps {
  traditionalResults?: BackendMeshPoint[];
  enableSerialization?: boolean;
  gridSize?: number;
  maxVisualizationPoints?: number;
}

export const SerializedSpiderPlotDemo: React.FC<SerializedSpiderPlotDemoProps> = ({
  traditionalResults = [],
  enableSerialization = true,
  gridSize = 15,
  maxVisualizationPoints = 1000
}) => {
  
  // Use adaptive computation hook that handles both traditional and serialized
  const computationHook = useAdaptiveComputation(traditionalResults, enableSerialization);
  
  // Demo state
  const [filterSettings, setFilterSettings] = useState({
    resnormMin: 0,
    resnormMax: 10,
    showParameterFilter: false,
    parameterFilters: {
      Rsh: { min: 10, max: 10000 },
      Ra: { min: 10, max: 10000 }
    }
  });
  
  const [demoMode, setDemoMode] = useState<'all' | 'best' | 'filtered' | 'resnorm'>('all');
  const [isGeneratingDemo, setIsGeneratingDemo] = useState(false);
  
  // Generate demo data if no results provided
  const generateDemoResults = useCallback(async () => {
    if (traditionalResults.length > 0 || computationHook.hasResults) return;
    
    setIsGeneratingDemo(true);
    console.log('🎯 Generating demo data for serialized spider plot...');
    
    try {
      // Generate sample configuration IDs
      const configIds = ConfigUtils.generateAllConfigIds(Math.min(gridSize, 5)); // Small grid for demo
      const sampleCount = Math.min(100, configIds.length);
      
      // Create demo BackendMeshPoints
      const demoResults: BackendMeshPoint[] = [];
      
      for (let i = 0; i < sampleCount; i++) {
        const config = configIds[i];
        const serializer = ConfigUtils.createStandard(gridSize);
        const params = serializer.deserializeConfig(config);
        
        // Convert to CircuitParameters format
        const circuitParams: CircuitParameters = {
          Rsh: params.rsh,
          Ra: params.ra,
          Ca: params.ca,
          Rb: params.rb,
          Cb: params.cb,
          frequency_range: [0.1, 100000] as [number, number]
        };
        
        // Simulate resnorm (lower values for lower resistance)
        const resnorm = (1.0 / (params.ra * params.ca * 1e6)) * (1 + Math.random() * 0.2);
        
        demoResults.push({
          parameters: circuitParams,
          resnorm,
          spectrum: [] // Empty for demo
        });
      }
      
      // Store in serialized format
      await computationHook.storeResults(demoResults);
      
      console.log(`✅ Generated ${demoResults.length} demo results`);
    } catch (error) {
      console.error('❌ Error generating demo results:', error);
    } finally {
      setIsGeneratingDemo(false);
    }
  }, [traditionalResults.length, computationHook, gridSize]);
  
  // Auto-generate demo data
  useEffect(() => {
    generateDemoResults();
  }, [generateDemoResults]);
  
  // Get models based on demo mode
  const displayedModels = useMemo((): ModelSnapshot[] => {
    if (!computationHook.hasResults) return [];
    
    switch (demoMode) {
      case 'best':
        return computationHook.getBestResults(50);
        
      case 'filtered':
        return computationHook.filterByParameters(filterSettings.parameterFilters);
        
      case 'resnorm':
        return computationHook.filterByResnorm(filterSettings.resnormMin, filterSettings.resnormMax);
        
      case 'all':
      default:
        return computationHook.getModelSnapshots(maxVisualizationPoints);
    }
  }, [computationHook, demoMode, filterSettings, maxVisualizationPoints]);
  
  // Storage statistics
  const storageStats = useMemo(() => computationHook.getStorageStats(), [computationHook]);
  
  // Clear cache handler
  const handleClearCache = useCallback(() => {
    computationHook.clearCache();
    console.log('🧹 Cache cleared');
  }, [computationHook]);
  
  return (
    <div className="w-full h-full flex flex-col space-y-4">
      
      {/* Header with Statistics */}
      <div className="bg-gray-900 rounded-lg p-4 space-y-3">
        <h2 className="text-xl font-bold text-white flex items-center space-x-2">
          <span>📊 Serialized Spider Plot Demo</span>
          {enableSerialization && <span className="text-green-400 text-sm">[Serialization Enabled]</span>}
        </h2>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div className="bg-gray-800 rounded p-3">
            <div className="text-gray-400">Total Results</div>
            <div className="text-white font-mono">{computationHook.resultCount.toLocaleString()}</div>
          </div>
          
          <div className="bg-gray-800 rounded p-3">
            <div className="text-gray-400">Displayed</div>
            <div className="text-white font-mono">{displayedModels.length.toLocaleString()}</div>
          </div>
          
          {storageStats && (
            <>
              <div className="bg-gray-800 rounded p-3">
                <div className="text-gray-400">Storage</div>
                <div className="text-green-400 font-mono">{storageStats.serializedSizeMB.toFixed(2)} MB</div>
                <div className="text-gray-500 text-xs">vs {storageStats.traditionalSizeMB.toFixed(1)} MB</div>
              </div>
              
              <div className="bg-gray-800 rounded p-3">
                <div className="text-gray-400">Efficiency</div>
                <div className="text-green-400 font-mono">{storageStats.reductionFactor.toFixed(0)}x smaller</div>
              </div>
            </>
          )}
        </div>
      </div>
      
      {/* Demo Controls */}
      <div className="bg-gray-900 rounded-lg p-4 space-y-3">
        <h3 className="text-lg font-semibold text-white">Demo Controls</h3>
        
        <div className="flex flex-wrap gap-2">
          {(['all', 'best', 'filtered', 'resnorm'] as const).map((mode) => (
            <button
              key={mode}
              onClick={() => setDemoMode(mode)}
              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                demoMode === mode 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {mode === 'all' && '🌐 All Results'}
              {mode === 'best' && '🏆 Best 50'}
              {mode === 'filtered' && '🔍 Parameter Filter'}
              {mode === 'resnorm' && '📊 Resnorm Range'}
            </button>
          ))}
        </div>
        
        {/* Filter Controls */}
        {demoMode === 'resnorm' && (
          <div className="flex items-center space-x-4">
            <label className="text-sm text-gray-400">
              Min Resnorm:
              <input
                type="number"
                value={filterSettings.resnormMin}
                onChange={(e) => setFilterSettings(prev => ({ ...prev, resnormMin: parseFloat(e.target.value) || 0 }))}
                className="ml-2 w-20 px-2 py-1 bg-gray-800 text-white rounded text-xs"
                step="0.001"
                min="0"
              />
            </label>
            <label className="text-sm text-gray-400">
              Max Resnorm:
              <input
                type="number"
                value={filterSettings.resnormMax}
                onChange={(e) => setFilterSettings(prev => ({ ...prev, resnormMax: parseFloat(e.target.value) || 10 }))}
                className="ml-2 w-20 px-2 py-1 bg-gray-800 text-white rounded text-xs"
                step="0.001"
                min="0"
              />
            </label>
          </div>
        )}
        
        <div className="flex items-center space-x-4">
          <button
            onClick={handleClearCache}
            className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white rounded text-sm"
          >
            🧹 Clear Cache
          </button>
          
          <button
            onClick={generateDemoResults}
            disabled={isGeneratingDemo}
            className="px-3 py-1 bg-green-600 hover:bg-green-700 text-white rounded text-sm disabled:opacity-50"
          >
            {isGeneratingDemo ? '⏳ Generating...' : '🎲 Generate Demo Data'}
          </button>
        </div>
      </div>
      
      {/* Spider Plot Visualization */}
      <div className="flex-1 bg-gray-900 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-white mb-4">
          3D Spider Plot - {demoMode.charAt(0).toUpperCase() + demoMode.slice(1)} View
        </h3>
        
        {displayedModels.length > 0 ? (
          <div className="w-full h-96">
            <SpiderPlot3D
              models={displayedModels}
              width={800}
              height={400}
              showControls={true}
              showAdvancedControls={true}
              responsive={true}
              showLabels={true}
            />
          </div>
        ) : (
          <div className="w-full h-96 flex items-center justify-center text-gray-500">
            {isGeneratingDemo ? '⏳ Generating demo data...' : '📊 No data to display'}
          </div>
        )}
      </div>
      
      {/* Technical Info */}
      <div className="bg-gray-900 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-white mb-2">Technical Implementation</h3>
        <div className="text-sm text-gray-300 space-y-1">
          <div>✅ <strong>Procedural Rendering:</strong> Parameters generated from config IDs on-demand</div>
          <div>✅ <strong>Ultra-compact Storage:</strong> Only config IDs + resnorms stored (~45 bytes vs 2.4KB)</div>
          <div>✅ <strong>Seamless Integration:</strong> Drop-in replacement for BackendMeshPoint[]</div>
          <div>✅ <strong>Performance Caching:</strong> Intelligent caching of generated parameters</div>
          <div>✅ <strong>Adaptive Filtering:</strong> Direct config ID manipulation for ultra-fast filtering</div>
          {computationHook.isCompatibleWithExistingSystem && (
            <div>✅ <strong>System Compatibility:</strong> Fully compatible with existing spider plot</div>
          )}
        </div>
      </div>
      
    </div>
  );
};