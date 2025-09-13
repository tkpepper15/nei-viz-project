/**
 * Serialized Computation Manager
 * =============================
 * 
 * Integrates the serialization system with the existing rendering engine.
 * Stores only config IDs + resnorms, then procedurally generates full objects.
 */

import { BackendMeshPoint, ModelSnapshot } from '../types';
import { CircuitParameters } from '../types/parameters';
import { ConfigSerializer, ConfigId, SerializedCircuitParameters } from './configSerializer';
import { FrequencySerializer, FrequencyConfig } from './frequencySerializer';

// Serialized computation result - ultra-compact storage
interface SerializedResult {
  configId: string;           // e.g. "15_01_02_04_03_05"
  frequencyConfigId: string;  // e.g. "L_1.0E-01_1.0E+05_100"
  resnorm: number;
  computationTime?: number;
  timestamp?: string;
}

// Manager configuration
interface SerializedComputationConfig {
  gridSize: number;
  frequencyPreset: string;
  cacheEnabled: boolean;
  maxCacheSize: number;
}

export class SerializedComputationManager {
  private configSerializer: ConfigSerializer;
  private frequencySerializer: FrequencySerializer;
  private results: SerializedResult[] = [];
  private config: SerializedComputationConfig;
  
  // Performance caches
  private parameterCache = new Map<string, CircuitParameters>();
  private spectrumCache = new Map<string, Array<{freq: number; real: number; imag: number; mag: number; phase: number}>>();
  private modelSnapshotCache = new Map<string, ModelSnapshot>();
  
  // Pagination support with global resnorm ordering
  private sortedResults: SerializedResult[] = [];
  private resultsSorted: boolean = false;
  
  constructor(config: Partial<SerializedComputationConfig> = {}) {
    this.config = {
      gridSize: 9,
      frequencyPreset: 'standard',
      cacheEnabled: true,
      maxCacheSize: 10000,
      ...config
    };
    
    this.configSerializer = new ConfigSerializer(this.config.gridSize);
    this.frequencySerializer = new FrequencySerializer();
    
    console.log(`🚀 SerializedComputationManager initialized: ${this.config.gridSize}x5 grid, ${this.config.frequencyPreset} frequency preset`);
  }
  
  /**
   * Store computation results in serialized format
   * Converts BackendMeshPoint[] to ultra-compact SerializedResult[]
   */
  storeResults(backendResults: BackendMeshPoint[], frequencyPreset: string = 'standard'): number {
    const startTime = Date.now();
    const freqConfig = this.frequencySerializer.getPreset(frequencyPreset);
    
    if (!freqConfig) {
      throw new Error(`Unknown frequency preset: ${frequencyPreset}`);
    }
    
    const serializedResults: SerializedResult[] = [];
    const frequencyConfigId = freqConfig.toId();
    
    for (const result of backendResults) {
      try {
        // Convert CircuitParameters to serialized format
        const serializedParams: SerializedCircuitParameters = {
          rsh: result.parameters.Rsh,
          ra: result.parameters.Ra,
          ca: result.parameters.Ca,
          rb: result.parameters.Rb,
          cb: result.parameters.Cb
        };
        
        // Find matching config ID (procedural reverse lookup)
        const configId = this.findConfigIdForParameters(serializedParams);
        
        if (configId) {
          serializedResults.push({
            configId: configId.toString(),
            frequencyConfigId,
            resnorm: result.resnorm,
            computationTime: undefined,
            timestamp: new Date().toISOString()
          });
        }
      } catch (error) {
        console.warn('Failed to serialize result:', error);
      }
    }
    
    this.results = serializedResults;
    // Invalidate sorted cache when new results are added
    this.resultsSorted = false;
    this.sortedResults = [];
    
    const compressionRatio = this.calculateCompressionRatio(backendResults.length, serializedResults.length);
    const duration = Date.now() - startTime;
    
    console.log(`✅ Stored ${serializedResults.length}/${backendResults.length} results in ${duration}ms`);
    console.log(`📊 Compression: ${compressionRatio.traditional.toFixed(1)}MB → ${compressionRatio.serialized.toFixed(1)}MB (${compressionRatio.reductionFactor.toFixed(0)}x smaller)`);
    
    return serializedResults.length;
  }
  
  /**
   * Generate BackendMeshPoint[] from serialized results - procedural reconstruction
   */
  generateBackendMeshPoints(maxResults?: number): BackendMeshPoint[] {
    const startTime = Date.now();
    const results = maxResults ? this.results.slice(0, maxResults) : this.results;
    const backendPoints: BackendMeshPoint[] = [];
    
    // Get frequency configuration
    const firstResult = results[0];
    if (!firstResult) return [];
    
    const freqConfig = FrequencyConfig.fromId(firstResult.frequencyConfigId);
    const frequencies = freqConfig.generateFrequencies();
    const frequencyRange: [number, number] = [frequencies[0], frequencies[frequencies.length - 1]];
    
    let cacheHits = 0;
    let regenerated = 0;
    
    for (const result of results) {
      try {
        // Get or generate circuit parameters
        let circuitParams: CircuitParameters;
        const cacheKey = result.configId;
        
        if (this.config.cacheEnabled && this.parameterCache.has(cacheKey)) {
          circuitParams = this.parameterCache.get(cacheKey)!;
          cacheHits++;
        } else {
          // Procedurally regenerate parameters from config ID
          const configId = ConfigId.fromString(result.configId);
          const serializedParams = this.configSerializer.deserializeConfig(configId);
          
          circuitParams = {
            Rsh: serializedParams.rsh,
            Ra: serializedParams.ra,
            Ca: serializedParams.ca,
            Rb: serializedParams.rb,
            Cb: serializedParams.cb,
            frequency_range: frequencyRange
          };
          
          if (this.config.cacheEnabled) {
            this.parameterCache.set(cacheKey, circuitParams);
          }
          regenerated++;
        }
        
        // Generate spectrum on-demand (for now, empty array for performance)
        // In full implementation, this would calculate impedance spectrum
        const spectrum: Array<{freq: number; real: number; imag: number; mag: number; phase: number}> = [];
        
        backendPoints.push({
          parameters: circuitParams,
          resnorm: result.resnorm,
          spectrum
        });
      } catch (error) {
        console.warn(`Failed to generate backend point for ${result.configId}:`, error);
      }
    }
    
    const duration = Date.now() - startTime;
    console.log(`🔧 Generated ${backendPoints.length} BackendMeshPoints in ${duration}ms`);
    console.log(`📋 Cache performance: ${cacheHits} hits, ${regenerated} regenerated`);
    
    return backendPoints;
  }
  
  /**
   * Generate ModelSnapshot[] for spider plot rendering - procedural reconstruction
   */
  generateModelSnapshots(maxResults?: number): ModelSnapshot[] {
    const startTime = Date.now();
    const results = maxResults ? this.results.slice(0, maxResults) : this.results;
    const modelSnapshots: ModelSnapshot[] = [];
    
    let cacheHits = 0;
    let regenerated = 0;
    
    for (let i = 0; i < results.length; i++) {
      const result = results[i];
      
      try {
        const cacheKey = `${result.configId}_${result.frequencyConfigId}`;
        
        if (this.config.cacheEnabled && this.modelSnapshotCache.has(cacheKey)) {
          modelSnapshots.push(this.modelSnapshotCache.get(cacheKey)!);
          cacheHits++;
          continue;
        }
        
        // Procedurally regenerate from config ID
        const configId = ConfigId.fromString(result.configId);
        const serializedParams = this.configSerializer.deserializeConfig(configId);
        const freqConfig = FrequencyConfig.fromId(result.frequencyConfigId);
        const frequencies = freqConfig.generateFrequencies();
        
        const circuitParams: CircuitParameters = {
          Rsh: serializedParams.rsh,
          Ra: serializedParams.ra,
          Ca: serializedParams.ca,
          Rb: serializedParams.rb,
          Cb: serializedParams.cb,
          frequency_range: [frequencies[0], frequencies[frequencies.length - 1]]
        };
        
        // Create ModelSnapshot with procedural data
        const modelSnapshot: ModelSnapshot = {
          id: `model_${result.configId}_${i}`,
          name: `Config ${result.configId}`,
          timestamp: Date.now(),
          parameters: circuitParams,
          data: [], // Empty for spider plot (impedance not needed)
          resnorm: result.resnorm,
          color: this.generateColorForResnorm(result.resnorm),
          isVisible: true,
          opacity: 1.0,
          ter: undefined
        };
        
        if (this.config.cacheEnabled) {
          this.modelSnapshotCache.set(cacheKey, modelSnapshot);
        }
        
        modelSnapshots.push(modelSnapshot);
        regenerated++;
        
      } catch (error) {
        console.warn(`Failed to generate ModelSnapshot for ${result.configId}:`, error);
      }
    }
    
    const duration = Date.now() - startTime;
    console.log(`🎨 Generated ${modelSnapshots.length} ModelSnapshots in ${duration}ms`);
    console.log(`📋 Cache performance: ${cacheHits} hits, ${regenerated} regenerated`);
    
    return modelSnapshots;
  }
  
  /**
   * Filter results by parameter ranges - direct config ID filtering (ultra-fast)
   */
  filterByParameters(filters: Partial<{
    Rsh: { min: number; max: number };
    Ra: { min: number; max: number };
    Ca: { min: number; max: number };
    Rb: { min: number; max: number };
    Cb: { min: number; max: number };
  }>): SerializedResult[] {
    const startTime = Date.now();
    
    const filtered = this.results.filter(result => {
      try {
        const configId = ConfigId.fromString(result.configId);
        const params = this.configSerializer.deserializeConfig(configId);
        
        // Check each filter condition
        for (const [param, range] of Object.entries(filters)) {
          if (range) {
            const value = params[param as keyof SerializedCircuitParameters] as number;
            if (value < range.min || value > range.max) {
              return false;
            }
          }
        }
        
        return true;
      } catch {
        return false;
      }
    });
    
    const duration = Date.now() - startTime;
    console.log(`🔍 Filtered ${this.results.length} → ${filtered.length} results in ${duration}ms`);
    
    return filtered;
  }
  
  /**
   * Filter by resnorm range - direct numeric filtering (ultra-fast)
   */
  filterByResnorm(min: number, max: number): SerializedResult[] {
    const startTime = Date.now();
    
    const filtered = this.results.filter(result => 
      result.resnorm >= min && result.resnorm <= max
    );
    
    const duration = Date.now() - startTime;
    console.log(`📊 Resnorm filter ${this.results.length} → ${filtered.length} results in ${duration}ms`);
    
    return filtered;
  }
  
  /**
   * Get best N results - direct numeric sorting (ultra-fast)
   * Defaults to 1000 results for performance, but allows override
   */
  getBestResults(n: number = 1000): SerializedResult[] {
    const startTime = Date.now();
    
    // Sort by resnorm and take top N (globally ordered across all results)
    const sorted = [...this.results].sort((a, b) => a.resnorm - b.resnorm);
    const best = sorted.slice(0, n);
    
    const duration = Date.now() - startTime;
    console.log(`🏆 Retrieved top ${n} of ${this.results.length} results (${(n/this.results.length*100).toFixed(1)}%) in ${duration}ms`);
    
    // Warn if requesting more than recommended limit
    if (n > 1000) {
      console.warn(`⚠️  Requesting ${n} results exceeds recommended limit of 1000 for optimal performance`);
    }
    
    return best;
  }
  
  /**
   * Get paginated results with global resnorm ordering
   */
  getPaginatedResults(pageNumber: number, pageSize: number = 100): {
    results: SerializedResult[];
    totalPages: number;
    totalResults: number;
    currentPage: number;
    hasNextPage: boolean;
    hasPrevPage: boolean;
  } {
    const startTime = Date.now();
    
    // Ensure results are sorted globally by resnorm
    if (!this.resultsSorted) {
      this.sortedResults = [...this.results].sort((a, b) => a.resnorm - b.resnorm);
      this.resultsSorted = true;
    }
    
    const totalResults = this.sortedResults.length;
    const totalPages = Math.ceil(totalResults / pageSize);
    const startIndex = (pageNumber - 1) * pageSize;
    const endIndex = Math.min(startIndex + pageSize, totalResults);
    
    const pageResults = this.sortedResults.slice(startIndex, endIndex);
    
    const duration = Date.now() - startTime;
    console.log(`📄 Page ${pageNumber}/${totalPages}: ${pageResults.length} results in ${duration}ms`);
    
    return {
      results: pageResults,
      totalPages,
      totalResults,
      currentPage: pageNumber,
      hasNextPage: pageNumber < totalPages,
      hasPrevPage: pageNumber > 1
    };
  }
  
  /**
   * Get storage statistics
   */
  getStorageStats() {
    const traditionalSize = this.results.length * this.estimateTraditionalSize();
    const serializedSize = this.results.length * this.estimateSerializedSize();
    const reductionFactor = traditionalSize / serializedSize;
    
    return {
      totalResults: this.results.length,
      traditionalSizeMB: traditionalSize / (1024 * 1024),
      serializedSizeMB: serializedSize / (1024 * 1024),
      reductionFactor,
      cacheStats: {
        parameterCacheSize: this.parameterCache.size,
        spectrumCacheSize: this.spectrumCache.size,
        modelCacheSize: this.modelSnapshotCache.size
      }
    };
  }
  
  /**
   * Clear all caches including pagination cache
   */
  clearCaches(): void {
    this.parameterCache.clear();
    this.spectrumCache.clear();
    this.modelSnapshotCache.clear();
    this.sortedResults = [];
    this.resultsSorted = false;
    console.log('🧹 All caches cleared including pagination cache');
  }
  
  // Private helper methods
  
  private findConfigIdForParameters(params: SerializedCircuitParameters): ConfigId | null {
    try {
      // Use relative tolerance matching for more robust parameter identification
      const relativeMatch = (expected: number, actual: number, relativeTolerance: number): boolean => {
        if (expected === 0 && actual === 0) return true;
        if (expected === 0 || actual === 0) return false;
        return Math.abs((expected - actual) / expected) < relativeTolerance;
      };
      
      // Find closest match within the parameter grid
      for (let rsh = 1; rsh <= this.config.gridSize; rsh++) {
        for (let ra = 1; ra <= this.config.gridSize; ra++) {
          for (let ca = 1; ca <= this.config.gridSize; ca++) {
            for (let rb = 1; rb <= this.config.gridSize; rb++) {
              for (let cb = 1; cb <= this.config.gridSize; cb++) {
                const testConfig = this.configSerializer.serializeConfig(ra, rb, rsh, ca, cb);
                const testParams = this.configSerializer.deserializeConfig(testConfig);
                
                // Use relative tolerance (1% for resistances, 5% for capacitances due to precision)
                if (
                  relativeMatch(testParams.rsh, params.rsh, 0.01) &&
                  relativeMatch(testParams.ra, params.ra, 0.01) &&
                  relativeMatch(testParams.ca, params.ca, 0.05) &&
                  relativeMatch(testParams.rb, params.rb, 0.01) &&
                  relativeMatch(testParams.cb, params.cb, 0.05)
                ) {
                  return testConfig;
                }
              }
            }
          }
        }
      }
      
      console.warn('No matching config found for parameters:', params);
    } catch (error) {
      console.warn('Parameter lookup failed:', error);
    }
    
    return null;
  }
  
  private generateColorForResnorm(resnorm: number): string {
    // Generate color based on resnorm value for consistent coloring
    const hue = Math.min(240, Math.max(0, 240 - (Math.log10(resnorm + 1) * 40)));
    return `hsl(${hue}, 70%, 50%)`;
  }
  
  private calculateCompressionRatio(originalCount: number, serializedCount: number) {
    const traditionalSize = originalCount * this.estimateTraditionalSize();
    const serializedSize = serializedCount * this.estimateSerializedSize();
    
    return {
      traditional: traditionalSize / (1024 * 1024), // MB
      serialized: serializedSize / (1024 * 1024),   // MB
      reductionFactor: traditionalSize / serializedSize
    };
  }
  
  private estimateTraditionalSize(): number {
    // Estimate size of a BackendMeshPoint
    return (
      5 * 8 +        // CircuitParameters (5 floats)
      8 +            // resnorm
      100 * 5 * 8    // spectrum array (estimated 100 points * 5 values * 8 bytes)
    );
  }
  
  private estimateSerializedSize(): number {
    // Estimate size of a SerializedResult
    return (
      20 +  // configId string
      25 +  // frequencyConfigId string
      8 +   // resnorm
      8     // optional fields
    );
  }
}

/**
 * Factory function to create manager with sensible defaults
 */
export function createSerializedComputationManager(gridSize: number = 9, frequencyPreset: string = 'standard'): SerializedComputationManager {
  return new SerializedComputationManager({
    gridSize,
    frequencyPreset,
    cacheEnabled: true,
    maxCacheSize: 10000
  });
}

/**
 * Utility functions for integration with existing components
 */
export const SerializedComputationUtils = {
  /**
   * Convert existing BackendMeshPoint[] to serialized storage
   */
  migrateFromBackendMeshPoints: (results: BackendMeshPoint[], gridSize: number = 9): SerializedComputationManager => {
    const manager = createSerializedComputationManager(gridSize);
    manager.storeResults(results);
    return manager;
  },
  
  /**
   * Create manager from NPZ data migration
   */
  createFromNPZData: (npzResults: Array<{Rsh?: number; Ra?: number; Ca?: number; Rb?: number; Cb?: number; rsh?: number; ra?: number; ca?: number; rb?: number; cb?: number; resnorm?: number; spectrum?: unknown[]}>, gridSize: number = 9): SerializedComputationManager => {
    // Convert NPZ format to BackendMeshPoint format first
    const backendResults: BackendMeshPoint[] = npzResults.map((result) => ({
      parameters: {
        Rsh: result.Rsh || result.rsh || 0,
        Ra: result.Ra || result.ra || 0,
        Ca: result.Ca || result.ca || 0,
        Rb: result.Rb || result.rb || 0,
        Cb: result.Cb || result.cb || 0,
        frequency_range: [1, 10000] as [number, number]
      },
      resnorm: result.resnorm || 0,
      spectrum: (result.spectrum as Array<{freq: number; real: number; imag: number; mag: number; phase: number}>) || []
    }));
    
    return SerializedComputationUtils.migrateFromBackendMeshPoints(backendResults, gridSize);
  }
};