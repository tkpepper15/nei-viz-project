export interface PerformanceMetrics {
  timestamp: number;
  stage: string;
  duration: number;
  memoryUsage?: {
    used: number;
    total: number;
    percentage: number;
  };
  details?: Record<string, unknown>;
}

export interface TileMetrics {
  tileId: string;
  renderTime: number;
  polygonCount: number;
  memoryBefore: number;
  memoryAfter: number;
  canvasSize: number;
  workerIndex: number;
}

export interface RenderSessionMetrics {
  sessionId: string;
  startTime: number;
  endTime?: number;
  totalDuration?: number;
  stages: PerformanceMetrics[];
  tiles: TileMetrics[];
  totalPolygons: number;
  finalImageSize: number;
  peakMemoryUsage: number;
  averageTileTime: number;
  slowestTile: TileMetrics | null;
  fastestTile: TileMetrics | null;
  memoryLeaks: boolean;
  crashRisk: 'low' | 'medium' | 'high';
}

export class PerformanceMonitor {
  private sessions: Map<string, RenderSessionMetrics> = new Map();
  private currentSession: string | null = null;
  private memoryCheckInterval: NodeJS.Timeout | null = null;
  private performanceObserver: PerformanceObserver | null = null;
  private onUpdate?: (metrics: RenderSessionMetrics) => void;

  constructor(onUpdate?: (metrics: RenderSessionMetrics) => void) {
    this.onUpdate = onUpdate;
    this.initializePerformanceObserver();
  }

  private initializePerformanceObserver() {
    if (typeof window !== 'undefined' && 'PerformanceObserver' in window) {
      this.performanceObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach(entry => {
          if (entry.name.includes('tile-render')) {
            this.logPerformanceEntry(entry);
          }
        });
      });
      
      try {
        this.performanceObserver.observe({ entryTypes: ['measure', 'mark'] });
      } catch (error) {
        console.warn('Performance Observer not supported:', error);
      }
    }
  }

  private logPerformanceEntry(entry: PerformanceEntry) {
    if (this.currentSession) {
      const session = this.sessions.get(this.currentSession);
      if (session) {
        session.stages.push({
          timestamp: Date.now(),
          stage: entry.name,
          duration: entry.duration,
          memoryUsage: this.getMemoryUsage(),
          details: { entryType: entry.entryType }
        });
      }
    }
  }

  startSession(sessionId: string, totalPolygons: number): RenderSessionMetrics {
    this.endCurrentSession();
    
    const session: RenderSessionMetrics = {
      sessionId,
      startTime: Date.now(),
      stages: [],
      tiles: [],
      totalPolygons,
      finalImageSize: 0,
      peakMemoryUsage: 0,
      averageTileTime: 0,
      slowestTile: null,
      fastestTile: null,
      memoryLeaks: false,
      crashRisk: 'low'
    };

    this.sessions.set(sessionId, session);
    this.currentSession = sessionId;

    // Start memory monitoring
    this.startMemoryMonitoring();
    
    this.logStage('session_start', 0, {
      totalPolygons,
      initialMemory: this.getMemoryUsage()
    });

    console.log(`🚀 [PERFORMANCE] Session ${sessionId} started - ${totalPolygons.toLocaleString()} polygons`);
    return session;
  }

  endSession(): RenderSessionMetrics | null {
    if (!this.currentSession) return null;

    const session = this.sessions.get(this.currentSession);
    if (session) {
      session.endTime = Date.now();
      session.totalDuration = session.endTime - session.startTime;
      
      // Calculate analytics
      this.calculateSessionAnalytics(session);
      
      this.logStage('session_end', 0, {
        totalDuration: session.totalDuration,
        finalMemory: this.getMemoryUsage(),
        crashRisk: session.crashRisk
      });

      console.log(`🏁 [PERFORMANCE] Session ${session.sessionId} completed:`, this.formatSessionSummary(session));
    }

    this.stopMemoryMonitoring();
    this.currentSession = null;
    return session || null;
  }

  private endCurrentSession() {
    if (this.currentSession) {
      this.endSession();
    }
  }

  logStage(stage: string, duration: number, details?: Record<string, unknown>) {
    if (!this.currentSession) return;

    const session = this.sessions.get(this.currentSession);
    if (session) {
      const memoryUsage = this.getMemoryUsage();
      
      // Track peak memory
      if (memoryUsage && memoryUsage.used > session.peakMemoryUsage) {
        session.peakMemoryUsage = memoryUsage.used;
      }

      const metric: PerformanceMetrics = {
        timestamp: Date.now(),
        stage,
        duration,
        memoryUsage,
        details
      };

      session.stages.push(metric);

      // Check for performance issues
      this.checkPerformanceWarnings(session, metric);

      if (this.onUpdate) {
        this.onUpdate(session);
      }

      console.log(`⏱️  [PERF] ${stage}: ${duration.toFixed(1)}ms | Memory: ${memoryUsage?.percentage.toFixed(1)}%`, details);
    }
  }

  logTileComplete(tileMetrics: TileMetrics) {
    if (!this.currentSession) return;

    const session = this.sessions.get(this.currentSession);
    if (session) {
      session.tiles.push(tileMetrics);

      // Update fastest/slowest tiles
      if (!session.fastestTile || tileMetrics.renderTime < session.fastestTile.renderTime) {
        session.fastestTile = tileMetrics;
      }
      if (!session.slowestTile || tileMetrics.renderTime > session.slowestTile.renderTime) {
        session.slowestTile = tileMetrics;
      }

      // Calculate running average
      session.averageTileTime = session.tiles.reduce((sum, tile) => sum + tile.renderTime, 0) / session.tiles.length;

      console.log(`🔲 [TILE] ${tileMetrics.tileId}: ${tileMetrics.renderTime.toFixed(1)}ms | ${tileMetrics.polygonCount} polygons | Worker ${tileMetrics.workerIndex}`);

      if (this.onUpdate) {
        this.onUpdate(session);
      }
    }
  }

  private checkPerformanceWarnings(session: RenderSessionMetrics, metric: PerformanceMetrics) {
    // Memory usage warnings
    if (metric.memoryUsage) {
      if (metric.memoryUsage.percentage > 85) {
        session.crashRisk = 'high';
        console.warn(`⚠️  [MEMORY] Critical memory usage: ${metric.memoryUsage.percentage.toFixed(1)}%`);
      } else if (metric.memoryUsage.percentage > 70) {
        session.crashRisk = 'medium';
        console.warn(`⚠️  [MEMORY] High memory usage: ${metric.memoryUsage.percentage.toFixed(1)}%`);
      }
    }

    // Long operation warnings
    if (metric.duration > 5000) {
      console.warn(`⚠️  [PERFORMANCE] Long operation detected: ${metric.stage} took ${metric.duration.toFixed(1)}ms`);
    }

    // Check for memory leaks
    if (session.stages.length > 5) {
      const recentMemory = session.stages.slice(-5).map(s => s.memoryUsage?.used || 0);
      const isIncreasing = recentMemory.every((val, i) => i === 0 || val >= recentMemory[i - 1]);
      if (isIncreasing && recentMemory[recentMemory.length - 1] > recentMemory[0] * 1.5) {
        session.memoryLeaks = true;
        console.warn(`⚠️  [MEMORY LEAK] Detected increasing memory usage pattern`);
      }
    }
  }

  private calculateSessionAnalytics(session: RenderSessionMetrics) {
    // Final calculations
    if (session.tiles.length > 0) {
      session.averageTileTime = session.tiles.reduce((sum, tile) => sum + tile.renderTime, 0) / session.tiles.length;
    }

    // Analyze stages for bottlenecks
    const stageTimes = new Map<string, number[]>();
    session.stages.forEach(stage => {
      if (!stageTimes.has(stage.stage)) {
        stageTimes.set(stage.stage, []);
      }
      stageTimes.get(stage.stage)!.push(stage.duration);
    });

    console.log(`📊 [ANALYTICS] Stage breakdown:`, Array.from(stageTimes.entries()).map(([stage, times]) => ({
      stage,
      totalTime: times.reduce((a, b) => a + b, 0),
      avgTime: times.reduce((a, b) => a + b, 0) / times.length,
      count: times.length
    })));
  }

  private startMemoryMonitoring() {
    this.memoryCheckInterval = setInterval(() => {
      const memoryUsage = this.getMemoryUsage();
      if (memoryUsage && this.currentSession) {
        const session = this.sessions.get(this.currentSession);
        if (session && memoryUsage.used > session.peakMemoryUsage) {
          session.peakMemoryUsage = memoryUsage.used;
        }
      }
    }, 1000);
  }

  private stopMemoryMonitoring() {
    if (this.memoryCheckInterval) {
      clearInterval(this.memoryCheckInterval);
      this.memoryCheckInterval = null;
    }
  }

  private getMemoryUsage() {
    if (typeof window !== 'undefined' && 'performance' in window && 'memory' in performance) {
      const memory = (performance as typeof performance & { memory: { usedJSHeapSize: number; jsHeapSizeLimit: number; totalJSHeapSize: number } }).memory;
      return {
        used: memory.usedJSHeapSize,
        total: memory.totalJSHeapSize,
        percentage: (memory.usedJSHeapSize / memory.totalJSHeapSize) * 100
      };
    }
    return undefined;
  }

  private formatSessionSummary(session: RenderSessionMetrics): string {
    return `
📊 Session Summary:
├─ Duration: ${session.totalDuration?.toFixed(0)}ms
├─ Polygons: ${session.totalPolygons.toLocaleString()}
├─ Tiles: ${session.tiles.length}
├─ Avg Tile Time: ${session.averageTileTime.toFixed(1)}ms
├─ Peak Memory: ${(session.peakMemoryUsage / 1024 / 1024).toFixed(1)}MB
├─ Crash Risk: ${session.crashRisk}
└─ Memory Leaks: ${session.memoryLeaks ? 'DETECTED' : 'None'}`;
  }

  // Performance benchmarking utilities
  startTiming(label: string): string {
    const markName = `${label}-start`;
    performance.mark(markName);
    return markName;
  }

  endTiming(startMark: string, label: string): number {
    const endMark = `${label}-end`;
    const measureName = `tile-render-${label}`;
    
    performance.mark(endMark);
    performance.measure(measureName, startMark, endMark);
    
    const measure = performance.getEntriesByName(measureName)[0];
    return measure.duration;
  }

  // Memory cleanup utilities
  forceGarbageCollection() {
    if (typeof window !== 'undefined' && 'gc' in window) {
      (window as typeof window & { gc: () => void }).gc();
      console.log('🗑️  [MEMORY] Forced garbage collection');
    }
  }

  getCurrentSession(): RenderSessionMetrics | null {
    return this.currentSession ? this.sessions.get(this.currentSession) || null : null;
  }

  getAllSessions(): RenderSessionMetrics[] {
    return Array.from(this.sessions.values());
  }

  clearSessions() {
    this.sessions.clear();
    console.log('🧹 [CLEANUP] Cleared all performance sessions');
  }

  destroy() {
    this.endCurrentSession();
    this.stopMemoryMonitoring();
    if (this.performanceObserver) {
      this.performanceObserver.disconnect();
    }
    this.clearSessions();
  }
} 