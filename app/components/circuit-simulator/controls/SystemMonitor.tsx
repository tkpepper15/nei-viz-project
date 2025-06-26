"use client";

import React, { useState } from 'react';

interface GridFilterSettings {
  enableSmartFiltering: boolean;
  visibilityPercentage: number;
  maxVisiblePoints: number;
  filterMode: 'best_resnorm' | 'distributed' | 'random';
}

interface SystemMonitorProps {
  gridSize?: number;
  totalGridPoints?: number;
  computedGridPoints?: number;
  onGridFilterChanged?: (settings: GridFilterSettings) => void;
}

// Calculate smart grid filtering recommendations
const calculateGridRecommendations = (gridSize: number) => {
  const totalPoints = Math.pow(gridSize, 5);
  
  if (totalPoints <= 243) { // 3^5
    return { percentage: 100, maxVisible: totalPoints, recommendation: 'Show all points' };
  } else if (totalPoints <= 1024) { // 4^5
    return { percentage: 75, maxVisible: Math.floor(totalPoints * 0.75), recommendation: 'Show 75% best resnorm' };
  } else if (totalPoints <= 3125) { // 5^5
    return { percentage: 50, maxVisible: Math.floor(totalPoints * 0.5), recommendation: 'Show 50% best resnorm' };
  } else if (totalPoints <= 7776) { // 6^5
    return { percentage: 25, maxVisible: Math.floor(totalPoints * 0.25), recommendation: 'Show 25% best resnorm' };
  } else {
    return { percentage: 10, maxVisible: Math.floor(totalPoints * 0.1), recommendation: 'Show 10% best resnorm (Performance Mode)' };
  }
};

const MetricCard: React.FC<{
  title: string;
  value: string | number;
  unit: string;
  status: 'good' | 'warning' | 'critical';
  subtitle?: string;
}> = ({ title, value, unit, status, subtitle }) => {
  const statusColors = {
    good: 'text-green-400 border-green-400/20 bg-green-400/5',
    warning: 'text-yellow-400 border-yellow-400/20 bg-yellow-400/5',
    critical: 'text-red-400 border-red-400/20 bg-red-400/5'
  };

  return (
    <div className={`border rounded-md p-3 ${statusColors[status]} transition-all`}>
      <div className="text-center space-y-1">
        <h4 className="text-xs font-medium text-neutral-300">{title}</h4>
        {subtitle && <p className="text-[10px] text-neutral-500">{subtitle}</p>}
        <div className="flex items-baseline justify-center gap-1 mt-2">
          <span className="text-2xl font-bold leading-none">{value}</span>
          <span className="text-xs text-neutral-400">{unit}</span>
        </div>
      </div>
    </div>
  );
};

const GridFilterControls: React.FC<{
  gridSize: number;
  totalPoints: number;
  computedPoints: number;
  onFilterChanged: (settings: GridFilterSettings) => void;
}> = ({ gridSize, totalPoints, computedPoints, onFilterChanged }) => {
  const recommendations = calculateGridRecommendations(gridSize);
  const [filterSettings, setFilterSettings] = useState<GridFilterSettings>({
    enableSmartFiltering: totalPoints > 1000,
    visibilityPercentage: recommendations.percentage,
    maxVisiblePoints: recommendations.maxVisible,
    filterMode: 'best_resnorm'
  });

  const handleSettingChange = (key: keyof GridFilterSettings, value: boolean | number | string) => {
    const newSettings = { ...filterSettings, [key]: value };
    setFilterSettings(newSettings);
    onFilterChanged(newSettings);
  };

  return (
    <div className="bg-neutral-800/50 rounded-lg p-4 border border-neutral-600/30">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-sm font-medium text-neutral-300">Smart Grid Filtering</h4>
        <div className="flex items-center gap-2">
          <span className="text-xs text-neutral-400">Auto-Optimize</span>
          <input
            type="checkbox"
            checked={filterSettings.enableSmartFiltering}
            onChange={(e) => handleSettingChange('enableSmartFiltering', e.target.checked)}
            className="w-3 h-3"
          />
        </div>
      </div>

      {filterSettings.enableSmartFiltering && (
        <div className="space-y-4">
          {/* Visibility Control */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-xs text-neutral-400">Visible Points:</label>
              <span className="text-xs text-neutral-300 font-mono">
                {Math.floor((filterSettings.visibilityPercentage / 100) * computedPoints).toLocaleString()} / {computedPoints.toLocaleString()}
              </span>
            </div>
            <input
              type="range"
              min="5"
              max="100"
              value={filterSettings.visibilityPercentage}
              onChange={(e) => handleSettingChange('visibilityPercentage', parseInt(e.target.value))}
              className="w-full h-2 bg-neutral-700 rounded appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-neutral-500">
              <span>5%</span>
              <span className="text-neutral-300">{filterSettings.visibilityPercentage}%</span>
              <span>100%</span>
            </div>
          </div>

          {/* Filter Mode */}
          <div className="space-y-2">
            <label className="text-xs text-neutral-400">Filter Mode:</label>
            <select
              value={filterSettings.filterMode}
              onChange={(e) => handleSettingChange('filterMode', e.target.value)}
              className="w-full text-xs bg-neutral-700 border border-neutral-600 rounded px-3 py-2"
            >
              <option value="best_resnorm">Best Resnorm (Recommended)</option>
              <option value="distributed">Distributed Sampling</option>
              <option value="random">Random Sampling</option>
            </select>
          </div>

          {/* Performance Recommendation */}
          <div className="bg-blue-900/20 border border-blue-400/30 rounded p-3">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-blue-400 text-xs">•</span>
              <span className="text-xs font-medium text-blue-300">Recommendation</span>
            </div>
            <p className="text-xs text-blue-200">{recommendations.recommendation}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export const SystemMonitor: React.FC<SystemMonitorProps> = ({
  gridSize = 3,
  totalGridPoints = 0,
  computedGridPoints = 0,
  onGridFilterChanged
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [activeTab, setActiveTab] = useState<'overview' | 'grid' | 'alerts'>('overview');

  // Enhanced computer information detection
  const getComputerInfo = () => {
    if (typeof navigator === 'undefined') {
      return { deviceType: 'Unknown', platform: 'Unknown', cores: 'Unknown', memory: 'Unknown', architecture: 'Unknown' };
    }

    const userAgent = navigator.userAgent;
    const platform = navigator.platform;
    const coresRaw = navigator.hardwareConcurrency;
    const cores = coresRaw || 'Unknown';
    
    // Detect device type and architecture
    let deviceType = 'Unknown';
    let architecture = 'Unknown';
    let memoryEstimate = 'Unknown';
    
    // Enhanced Mac detection
    if (userAgent.includes('Mac')) {
      if (userAgent.includes('MacIntel')) {
        // Detect Apple Silicon vs Intel
        if (platform.includes('MacIntel') && typeof coresRaw === 'number' && coresRaw >= 8) {
          // Likely Apple Silicon (M1/M2/M3 typically have 8+ cores)
          if (coresRaw >= 10) {
            deviceType = 'MacBook Pro (M2 Pro/Max)';
            architecture = 'Apple Silicon';
            memoryEstimate = '16GB - 96GB';
          } else if (coresRaw >= 8) {
            deviceType = 'MacBook (M1/M2)';
            architecture = 'Apple Silicon';
            memoryEstimate = '8GB - 24GB';
          }
        } else {
          deviceType = 'MacBook (Intel)';
          architecture = 'x86_64';
          memoryEstimate = '8GB - 32GB';
        }
      } else {
        deviceType = 'Mac';
        architecture = 'Unknown';
      }
    } else if (userAgent.includes('Windows')) {
      deviceType = 'Windows PC';
      architecture = platform.includes('Win32') ? 'x86_64' : 'Unknown';
      memoryEstimate = '4GB - 64GB+';
    } else if (userAgent.includes('Linux')) {
      deviceType = 'Linux';
      architecture = platform.includes('Linux x86_64') ? 'x86_64' : platform.includes('Linux aarch64') ? 'ARM64' : 'Unknown';
      memoryEstimate = '2GB - 128GB+';
    } else if (userAgent.includes('CrOS')) {
      deviceType = 'Chromebook';
      architecture = 'x86_64/ARM';
      memoryEstimate = '4GB - 16GB';
    }

    // Get available memory if supported
    let actualMemory = memoryEstimate;
    if ('memory' in performance) {
      const memInfo = (performance as { memory?: { jsHeapSizeLimit?: number } }).memory;
      if (memInfo?.jsHeapSizeLimit) {
        const limitGB = (memInfo.jsHeapSizeLimit / (1024 * 1024 * 1024)).toFixed(1);
        actualMemory = `~${limitGB}GB available`;
      }
    }
    
    return { 
      deviceType, 
      platform, 
      cores, 
      memory: actualMemory,
      architecture
    };
  };

  const computerInfo = getComputerInfo();

  // Static system info (no constant updates to prevent infinite loops)
  const getMemoryStatus = (): 'good' | 'warning' | 'critical' => {
    if (totalGridPoints > 5000) return 'critical';
    if (totalGridPoints > 2000) return 'warning';
    return 'good';
  };

  const getGridComplexity = (): 'good' | 'warning' | 'critical' => {
    if (totalGridPoints > 8000) return 'critical';
    if (totalGridPoints > 4000) return 'warning';
    return 'good';
  };

  const getPerformanceStatus = (): 'good' | 'warning' | 'critical' => {
    if (totalGridPoints > 8000) return 'critical';
    if (totalGridPoints > 4000) return 'warning';
    return 'good';
  };

  const systemInfo = {
    memory: {
      estimated: Math.min(100, (totalGridPoints / 10000) * 20),
      status: getMemoryStatus()
    },
    grid: {
      complexity: getGridComplexity(),
      impact: totalGridPoints > 10000 ? 'High' : totalGridPoints > 5000 ? 'Medium' : 'Low'
    },
    performance: {
      expected: getPerformanceStatus(),
      efficiency: Math.max(50, 100 - (totalGridPoints / 100))
    }
  };

  const getSystemStatus = () => {
    if (systemInfo.memory.status === 'critical' || systemInfo.grid.complexity === 'critical') {
      return { status: 'critical', color: 'text-red-400', text: 'Critical' };
    }
    if (systemInfo.memory.status === 'warning' || systemInfo.grid.complexity === 'warning') {
      return { status: 'warning', color: 'text-yellow-400', text: 'Warning' };
    }
    return { status: 'good', color: 'text-green-400', text: 'Optimal' };
  };

  const systemStatus = getSystemStatus();

  return (
    <div className="border border-neutral-700 rounded-lg overflow-hidden bg-neutral-900">
      {/* Header */}
      <button 
        className="w-full px-4 py-3 bg-neutral-800/80 backdrop-blur text-neutral-200 text-sm font-medium flex items-center justify-between hover:bg-neutral-750 transition-all"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-3">
          <span className="text-sm font-semibold">Performance</span>
          <div className="flex items-center gap-2 text-xs">
            <div className={`w-2 h-2 rounded-full ${systemStatus.color.replace('text-', 'bg-')}`}></div>
            <span className={`${systemStatus.color} font-medium`}>{systemStatus.text}</span>
            <span className="text-neutral-500">•</span>
            <span className="text-neutral-400">
              {computedGridPoints.toLocaleString()} pts
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
        <div className="bg-neutral-900/50 backdrop-blur">
          {/* Tab Navigation */}
          <div className="flex border-b border-neutral-700/50 bg-neutral-800/30">
            {[
              { id: 'overview', label: 'Overview' },
              { id: 'grid', label: 'Grid Control' },
              { id: 'alerts', label: 'Performance' }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as 'overview' | 'grid' | 'alerts')}
                className={`flex-1 px-4 py-2.5 text-xs font-medium transition-all ${
                  activeTab === tab.id 
                    ? 'bg-blue-500/20 text-blue-300 border-b-2 border-blue-400' 
                    : 'text-neutral-400 hover:text-neutral-300 hover:bg-neutral-700/30'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          <div className="p-4 space-y-4">
            {/* Overview Tab */}
            {activeTab === 'overview' && (
              <div className="space-y-4">
                {/* System Status Cards - Vertical Layout */}
                <div className="space-y-3">
                  <MetricCard
                    title="Memory Load"
                    value={systemInfo.memory.estimated.toFixed(0)}
                    unit="%"
                    status={systemInfo.memory.status}
                    subtitle="Estimated usage"
                  />
                  
                  <MetricCard
                    title="Grid Complexity"
                    value={totalGridPoints.toLocaleString()}
                    unit="pts"
                    status={systemInfo.grid.complexity}
                    subtitle={`${systemInfo.grid.impact} impact`}
                  />
                  
                  <MetricCard
                    title="Efficiency"
                    value={systemInfo.performance.efficiency.toFixed(0)}
                    unit="%"
                    status={systemInfo.performance.expected}
                    subtitle="Expected performance"
                  />
                </div>

                {/* Computer Information */}
                <div className="bg-neutral-800/30 rounded-lg p-4 border border-neutral-600/20">
                  <h4 className="text-sm font-medium text-neutral-300 mb-3">System Information</h4>
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between items-center">
                      <span className="text-neutral-400">Device:</span>
                      <span className="text-neutral-300 font-mono font-medium">{computerInfo.deviceType}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-neutral-400">Architecture:</span>
                      <span className="text-neutral-300 font-mono font-medium">{computerInfo.architecture}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-neutral-400">CPU Cores:</span>
                      <span className="text-neutral-300 font-mono font-medium">{computerInfo.cores}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-neutral-400">Memory:</span>
                      <span className="text-neutral-300 font-mono font-medium">{computerInfo.memory}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-neutral-400">Platform:</span>
                      <span className="text-neutral-300 font-mono font-medium text-[10px]">{computerInfo.platform}</span>
                    </div>
                  </div>
                </div>

                {/* Grid Analysis */}
                <div className="bg-neutral-800/30 rounded-lg p-4 border border-neutral-600/20">
                  <h4 className="text-sm font-medium text-neutral-300 mb-3">Grid Analysis</h4>
                  <div className="grid grid-cols-2 gap-4 text-xs">
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-neutral-400">Total Points:</span>
                        <span className="text-neutral-300 font-mono font-medium">{totalGridPoints.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-neutral-400">Computed:</span>
                        <span className="text-neutral-300 font-mono font-medium">{computedGridPoints.toLocaleString()}</span>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-neutral-400">Grid Size:</span>
                        <span className="text-neutral-300 font-mono font-medium">{gridSize}^5</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-neutral-400">Progress:</span>
                        <span className="text-neutral-300 font-mono font-medium">
                          {totalGridPoints > 0 ? ((computedGridPoints / totalGridPoints) * 100).toFixed(0) : 0}%
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Grid Control Tab */}
            {activeTab === 'grid' && (
              <div className="space-y-4">
                <GridFilterControls
                  gridSize={gridSize}
                  totalPoints={totalGridPoints}
                  computedPoints={computedGridPoints}
                  onFilterChanged={onGridFilterChanged || (() => {})}
                />
                
                {/* Performance Impact Analysis */}
                <div className="bg-neutral-800/30 rounded-lg p-4 border border-neutral-600/20">
                  <h4 className="text-sm font-medium text-neutral-300 mb-3">Performance Impact</h4>
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-neutral-400">Rendering Impact:</span>
                      <span className={
                        totalGridPoints > 8000 ? 'text-red-400' :
                        totalGridPoints > 4000 ? 'text-yellow-400' : 'text-green-400'
                      }>
                        {totalGridPoints > 8000 ? 'Critical' : totalGridPoints > 4000 ? 'Moderate' : 'Minimal'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-neutral-400">Memory Impact:</span>
                      <span className={
                        totalGridPoints > 5000 ? 'text-red-400' :
                        totalGridPoints > 2000 ? 'text-yellow-400' : 'text-green-400'
                      }>
                        {totalGridPoints > 5000 ? 'High' : totalGridPoints > 2000 ? 'Medium' : 'Low'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-neutral-400">Recommendation:</span>
                      <span className="text-blue-300">
                        {calculateGridRecommendations(gridSize).recommendation}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Settings Tab */}
            {activeTab === 'alerts' && (
              <div className="space-y-4">
                {/* Performance Settings */}
                <div className="bg-neutral-800/30 rounded-lg p-4 border border-neutral-600/20">
                  <h4 className="text-sm font-medium text-neutral-300 mb-3">Performance Settings</h4>
                  <div className="space-y-3 text-xs">
                    <div className="flex items-center justify-between">
                      <span className="text-neutral-400">Auto-optimize grid filtering</span>
                      <input type="checkbox" defaultChecked className="w-3 h-3" />
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-neutral-400">Enable performance warnings</span>
                      <input type="checkbox" defaultChecked className="w-3 h-3" />
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-neutral-400">Smart memory management</span>
                      <input type="checkbox" defaultChecked className="w-3 h-3" />
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-neutral-400">High-quality rendering</span>
                      <input type="checkbox" defaultChecked className="w-3 h-3" />
                    </div>
                  </div>
                </div>

                {/* Memory Management */}
                <div className="bg-neutral-800/30 rounded-lg p-4 border border-neutral-600/20">
                  <h4 className="text-sm font-medium text-neutral-300 mb-3">Memory Management</h4>
                  <div className="space-y-3">
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="text-neutral-400">Max Memory (MB)</span>
                        <span className="text-neutral-300">2048</span>
                      </div>
                      <input
                        type="range"
                        min="512"
                        max="8192"
                        defaultValue="2048"
                        className="w-full h-1 bg-neutral-700 rounded appearance-none cursor-pointer"
                      />
                      <div className="flex justify-between text-[10px] text-neutral-500">
                        <span>512MB</span>
                        <span>8GB</span>
                      </div>
                    </div>
                    
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="text-neutral-400">Chunk Size</span>
                        <span className="text-neutral-300">25</span>
                      </div>
                      <input
                        type="range"
                        min="10"
                        max="100"
                        defaultValue="25"
                        className="w-full h-1 bg-neutral-700 rounded appearance-none cursor-pointer"
                      />
                      <div className="flex justify-between text-[10px] text-neutral-500">
                        <span>10</span>
                        <span>100</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Rendering Quality */}
                <div className="bg-neutral-800/30 rounded-lg p-4 border border-neutral-600/20">
                  <h4 className="text-sm font-medium text-neutral-300 mb-3">Rendering Quality</h4>
                  <div className="space-y-3">
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="text-neutral-400">Max Polygons</span>
                        <span className="text-neutral-300">50k</span>
                      </div>
                      <input
                        type="range"
                        min="5000"
                        max="200000"
                        defaultValue="50000"
                        className="w-full h-1 bg-neutral-700 rounded appearance-none cursor-pointer"
                      />
                      <div className="flex justify-between text-[10px] text-neutral-500">
                        <span>5k</span>
                        <span>200k</span>
                      </div>
                    </div>
                    
                    <div className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="text-neutral-400">Canvas Size</span>
                        <span className="text-neutral-300">800x800</span>
                      </div>
                      <select 
                        className="w-full text-xs bg-neutral-700 border border-neutral-600 rounded px-2 py-1"
                        defaultValue="800"
                      >
                        <option value="600">600x600</option>
                        <option value="800">800x800</option>
                        <option value="1024">1024x1024</option>
                        <option value="1200">1200x1200</option>
                      </select>
                    </div>
                  </div>
                </div>



                {/* Status Summary */}
                <div className="text-center py-4 text-neutral-500">
                  <p className="text-sm font-medium">System Status: <span className={systemStatus.color}>{systemStatus.text}</span></p>
                  <p className="text-xs text-neutral-600 mt-1">
                    Monitoring {totalGridPoints.toLocaleString()} grid points
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}; 