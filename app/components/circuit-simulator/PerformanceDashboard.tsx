"use client";

import React, { useState, useEffect, useMemo } from 'react';
import { RenderSessionMetrics, PerformanceMonitor } from './utils/performanceMonitor';

interface PerformanceDashboardProps {
  performanceMonitor: PerformanceMonitor;
  isVisible: boolean;
  onToggle: () => void;
}

export const PerformanceDashboard: React.FC<PerformanceDashboardProps> = ({
  performanceMonitor,
  isVisible,
  onToggle
}) => {
  const [currentSession, setCurrentSession] = useState<RenderSessionMetrics | null>(null);
  const [updateInterval, setUpdateInterval] = useState<NodeJS.Timeout | null>(null);

  // Update dashboard every 500ms during rendering
  useEffect(() => {
    if (isVisible) {
      const interval = setInterval(() => {
        const session = performanceMonitor.getCurrentSession();
        if (session) {
          setCurrentSession({ ...session });
        }
      }, 500);
      setUpdateInterval(interval);
      
      return () => {
        if (interval) clearInterval(interval);
      };
    }
  }, [isVisible, performanceMonitor]);

  // Cleanup interval
  useEffect(() => {
    return () => {
      if (updateInterval) clearInterval(updateInterval);
    };
  }, [updateInterval]);

  // Calculate performance analytics
  const analytics = useMemo(() => {
    if (!currentSession) return null;

    // Stage analysis
    const stages = new Map<string, { totalTime: number; count: number; avgTime: number }>();
    currentSession.stages.forEach(stage => {
      if (!stages.has(stage.stage)) {
        stages.set(stage.stage, { totalTime: 0, count: 0, avgTime: 0 });
      }
      const stageData = stages.get(stage.stage)!;
      stageData.totalTime += stage.duration;
      stageData.count += 1;
      stageData.avgTime = stageData.totalTime / stageData.count;
    });

    // Tile performance analysis
    const tiles = currentSession.tiles;
    const tileMetrics = {
      totalTiles: tiles.length,
      averageTime: tiles.length > 0 ? tiles.reduce((sum, tile) => sum + tile.renderTime, 0) / tiles.length : 0,
      fastestTile: tiles.reduce((min, tile) => tile.renderTime < min.renderTime ? tile : min, tiles[0] || { renderTime: Infinity }),
      slowestTile: tiles.reduce((max, tile) => tile.renderTime > max.renderTime ? tile : max, tiles[0] || { renderTime: 0 }),
      efficiency: tiles.length > 0 ? tiles.reduce((sum, tile) => sum + (tile.polygonCount / tile.renderTime * 1000), 0) / tiles.length : 0
    };

    // Memory analysis
    const memoryPoints = currentSession.stages.map(s => s.memoryUsage?.used || 0).filter(m => m > 0);
    const memoryAnalysis = {
      current: memoryPoints[memoryPoints.length - 1] || 0,
      peak: Math.max(...memoryPoints, 0),
      average: memoryPoints.length > 0 ? memoryPoints.reduce((a, b) => a + b, 0) / memoryPoints.length : 0,
      trend: memoryPoints.length > 5 ? 
        (memoryPoints.slice(-3).reduce((a, b) => a + b, 0) / 3) - 
        (memoryPoints.slice(-6, -3).reduce((a, b) => a + b, 0) / 3) : 0
    };

    return {
      stages: Array.from(stages.entries()).map(([name, data]) => ({ name, ...data })),
      tiles: tileMetrics,
      memory: memoryAnalysis,
      session: currentSession
    };
  }, [currentSession]);

  const formatBytes = (bytes: number) => {
    const mb = bytes / (1024 * 1024);
    return mb > 1024 ? `${(mb / 1024).toFixed(1)}GB` : `${mb.toFixed(1)}MB`;
  };

  const formatTime = (ms: number) => {
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    const seconds = ms / 1000;
    return seconds < 60 ? `${seconds.toFixed(1)}s` : `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(0)}s`;
  };

  const getPerformanceColor = (value: number, thresholds: { good: number; warning: number }) => {
    if (value <= thresholds.good) return 'text-green-400';
    if (value <= thresholds.warning) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getCrashRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'text-green-400';
      case 'medium': return 'text-yellow-400';
      case 'high': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  if (!isVisible) {
    return (
      <button
        onClick={onToggle}
        className="fixed bottom-4 right-4 bg-neutral-800 hover:bg-neutral-700 text-white p-3 rounded-lg shadow-lg border border-neutral-600 z-50"
        title="Show Performance Dashboard"
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      </button>
    );
  }

  return (
    <div className="fixed top-4 right-4 w-96 bg-neutral-900/95 backdrop-blur-sm border border-neutral-700 rounded-lg shadow-2xl z-50 max-h-[80vh] overflow-y-auto">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-neutral-700">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
          Performance Monitor
        </h3>
        <button
          onClick={onToggle}
          className="text-neutral-400 hover:text-white transition-colors"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <div className="p-4 space-y-4">
        {/* Session Overview */}
        {analytics?.session && (
          <div className="bg-neutral-800 rounded-lg p-3">
            <h4 className="text-sm font-medium text-white mb-2">Session Overview</h4>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-neutral-400">Duration:</span>
                <span className="text-white ml-2">
                  {analytics.session.totalDuration ? formatTime(analytics.session.totalDuration) : 'Running...'}
                </span>
              </div>
              <div>
                <span className="text-neutral-400">Polygons:</span>
                <span className="text-white ml-2">{analytics.session.totalPolygons.toLocaleString()}</span>
              </div>
              <div>
                <span className="text-neutral-400">Tiles:</span>
                <span className="text-white ml-2">{analytics.tiles.totalTiles}</span>
              </div>
              <div>
                <span className="text-neutral-400">Crash Risk:</span>
                <span className={`ml-2 font-medium ${getCrashRiskColor(analytics.session.crashRisk)}`}>
                  {analytics.session.crashRisk.toUpperCase()}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Memory Usage */}
        {analytics?.memory && (
          <div className="bg-neutral-800 rounded-lg p-3">
            <h4 className="text-sm font-medium text-white mb-2">Memory Usage</h4>
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-neutral-400">Current:</span>
                <span className={getPerformanceColor(analytics.memory.current / (1024 * 1024), { good: 100, warning: 200 })}>
                  {formatBytes(analytics.memory.current)}
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-neutral-400">Peak:</span>
                <span className={getPerformanceColor(analytics.memory.peak / (1024 * 1024), { good: 150, warning: 300 })}>
                  {formatBytes(analytics.memory.peak)}
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-neutral-400">Trend:</span>
                <span className={analytics.memory.trend > 0 ? 'text-red-400' : 'text-green-400'}>
                  {analytics.memory.trend > 0 ? '↗' : '↘'} {formatBytes(Math.abs(analytics.memory.trend))}
                </span>
              </div>
              {analytics.session.memoryLeaks && (
                <div className="text-xs text-red-400 font-medium">
                  ⚠️ Memory leaks detected!
                </div>
              )}
            </div>
          </div>
        )}

        {/* Tile Performance */}
        {analytics?.tiles && analytics.tiles.totalTiles > 0 && (
          <div className="bg-neutral-800 rounded-lg p-3">
            <h4 className="text-sm font-medium text-white mb-2">Tile Performance</h4>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-neutral-400">Average Time:</span>
                <span className={getPerformanceColor(analytics.tiles.averageTime, { good: 100, warning: 500 })}>
                  {formatTime(analytics.tiles.averageTime)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-400">Efficiency:</span>
                <span className={getPerformanceColor(1000 / analytics.tiles.efficiency, { good: 1, warning: 10 })}>
                  {analytics.tiles.efficiency.toFixed(0)} polygons/sec
                </span>
              </div>
              {analytics.tiles.fastestTile && (
                <div className="flex justify-between">
                  <span className="text-neutral-400">Fastest:</span>
                  <span className="text-green-400">
                    {formatTime(analytics.tiles.fastestTile.renderTime)}
                  </span>
                </div>
              )}
              {analytics.tiles.slowestTile && (
                <div className="flex justify-between">
                  <span className="text-neutral-400">Slowest:</span>
                  <span className="text-red-400">
                    {formatTime(analytics.tiles.slowestTile.renderTime)}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Stage Breakdown */}
        {analytics?.stages && analytics.stages.length > 0 && (
          <div className="bg-neutral-800 rounded-lg p-3">
            <h4 className="text-sm font-medium text-white mb-2">Stage Breakdown</h4>
            <div className="space-y-1 text-xs">
                             {analytics.stages
                 .sort((a, b) => b.totalTime - a.totalTime)
                 .slice(0, 5)
                 .map((stage) => (
                   <div key={stage.name} className="flex justify-between">
                    <span className="text-neutral-400 truncate" title={stage.name}>
                      {stage.name.replace(/^(tile-render-|session_)/, '')}:
                    </span>
                    <span className="text-white ml-2">
                      {formatTime(stage.totalTime)}
                      {stage.count > 1 && (
                        <span className="text-neutral-500 ml-1">
                          (×{stage.count})
                        </span>
                      )}
                    </span>
                  </div>
                ))}
            </div>
          </div>
        )}

        {/* Performance Warnings */}
        {analytics?.session && (
          <div className="bg-neutral-800 rounded-lg p-3">
            <h4 className="text-sm font-medium text-white mb-2">Optimization Tips</h4>
            <div className="space-y-1 text-xs text-neutral-300">
              {analytics.tiles.averageTime > 500 && (
                <div className="text-yellow-400">• Consider reducing tile size or polygon count</div>
              )}
              {analytics.memory.current > 200 * 1024 * 1024 && (
                <div className="text-red-400">• High memory usage detected</div>
              )}
              {analytics.tiles.efficiency < 1000 && (
                <div className="text-yellow-400">• Low rendering efficiency - check for bottlenecks</div>
              )}
              {analytics.session.crashRisk === 'high' && (
                <div className="text-red-400">• High crash risk - consider reducing quality</div>
              )}
              {analytics.session.memoryLeaks && (
                <div className="text-red-400">• Memory leaks detected - restart may be needed</div>
              )}
              {analytics.tiles.averageTime < 100 && analytics.tiles.efficiency > 2000 && (
                <div className="text-green-400">• Performance is optimal!</div>
              )}
            </div>
          </div>
        )}

        {/* Debug Actions */}
        <div className="bg-neutral-800 rounded-lg p-3">
          <h4 className="text-sm font-medium text-white mb-2">Debug Actions</h4>
          <div className="flex gap-2">
            <button
              onClick={() => performanceMonitor.forceGarbageCollection()}
              className="px-2 py-1 text-xs bg-blue-600 hover:bg-blue-700 text-white rounded"
            >
              Force GC
            </button>
            <button
              onClick={() => console.log('Session Data:', performanceMonitor.getCurrentSession())}
              className="px-2 py-1 text-xs bg-green-600 hover:bg-green-700 text-white rounded"
            >
              Log Data
            </button>
            <button
              onClick={() => performanceMonitor.clearSessions()}
              className="px-2 py-1 text-xs bg-red-600 hover:bg-red-700 text-white rounded"
            >
              Clear
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}; 