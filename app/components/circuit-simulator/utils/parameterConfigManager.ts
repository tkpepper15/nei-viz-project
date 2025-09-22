/**
 * Parameter Configuration Manager
 * ==============================
 *
 * Streamlined parameter configuration object that ensures dynamic changes
 * are properly reflected throughout the system while maintaining memory efficiency.
 */

import { CircuitParameters } from '../types/parameters';
import { ConfigSerializer, SerializedCircuitParameters } from './configSerializer';

export interface ParameterConfigState {
  // Core circuit parameters
  parameters: CircuitParameters;

  // Grid configuration
  gridSize: number;

  // Frequency configuration
  minFreq: number;
  maxFreq: number;
  numPoints: number;

  // Memory optimization settings
  maxVisibleResults: number;
  memoryOptimizationEnabled: boolean;

  // Computation tracking
  lastUpdated: number;
  configHash: string;
}

export interface ParameterChangeEvent {
  type: 'parameter' | 'grid' | 'frequency' | 'optimization';
  field: string;
  oldValue: unknown;
  newValue: unknown;
  timestamp: number;
  configId?: string;
}

/**
 * Manages parameter configuration state with automatic serialization and change tracking
 */
export class ParameterConfigManager {
  private state: ParameterConfigState;
  private listeners: ((state: ParameterConfigState, change?: ParameterChangeEvent) => void)[] = [];
  private configSerializer: ConfigSerializer;

  constructor(initialState: Partial<ParameterConfigState> = {}) {
    this.state = {
      parameters: {
        Rsh: 1000,
        Ra: 2000,
        Ca: 1e-6,
        Rb: 3000,
        Cb: 2e-6,
        frequency_range: [0.1, 100000]
      },
      gridSize: 9,
      minFreq: 0.1,
      maxFreq: 100000,
      numPoints: 100,
      maxVisibleResults: 1000,
      memoryOptimizationEnabled: true,
      lastUpdated: Date.now(),
      configHash: '',
      ...initialState
    };

    this.configSerializer = new ConfigSerializer(this.state.gridSize);
    this.state.configHash = this.calculateConfigHash();
  }

  /**
   * Get current parameter configuration state
   */
  getState(): ParameterConfigState {
    return { ...this.state };
  }

  /**
   * Update circuit parameters with automatic change tracking
   */
  updateParameters(newParams: Partial<CircuitParameters>): void {
    const changes: ParameterChangeEvent[] = [];

    for (const [key, value] of Object.entries(newParams)) {
      const oldValue = this.state.parameters[key as keyof CircuitParameters];
      if (oldValue !== value) {
        changes.push({
          type: 'parameter',
          field: key,
          oldValue,
          newValue: value,
          timestamp: Date.now()
        });
      }
    }

    // Update parameters and derived values
    this.state.parameters = { ...this.state.parameters, ...newParams };
    this.state.parameters.frequency_range = [this.state.minFreq, this.state.maxFreq];
    this.updateStateMetadata();

    // Notify listeners of changes
    changes.forEach(change => this.notifyListeners(change));

    console.log(`📊 Updated ${changes.length} parameter(s):`, changes.map(c => `${c.field}: ${c.oldValue} → ${c.newValue}`));
  }

  /**
   * Update grid size with automatic serializer reconfiguration
   */
  updateGridSize(newGridSize: number): void {
    if (newGridSize === this.state.gridSize) return;

    const change: ParameterChangeEvent = {
      type: 'grid',
      field: 'gridSize',
      oldValue: this.state.gridSize,
      newValue: newGridSize,
      timestamp: Date.now()
    };

    this.state.gridSize = newGridSize;

    // Reconfigure serializer for new grid size
    this.configSerializer = new ConfigSerializer(newGridSize);
    this.updateStateMetadata();

    this.notifyListeners(change);
    console.log(`🔧 Grid size updated: ${change.oldValue} → ${change.newValue}`);
  }

  /**
   * Update frequency range with parameter synchronization
   */
  updateFrequencyRange(minFreq?: number, maxFreq?: number, numPoints?: number): void {
    const changes: ParameterChangeEvent[] = [];

    if (minFreq !== undefined && minFreq !== this.state.minFreq) {
      changes.push({
        type: 'frequency',
        field: 'minFreq',
        oldValue: this.state.minFreq,
        newValue: minFreq,
        timestamp: Date.now()
      });
      this.state.minFreq = minFreq;
    }

    if (maxFreq !== undefined && maxFreq !== this.state.maxFreq) {
      changes.push({
        type: 'frequency',
        field: 'maxFreq',
        oldValue: this.state.maxFreq,
        newValue: maxFreq,
        timestamp: Date.now()
      });
      this.state.maxFreq = maxFreq;
    }

    if (numPoints !== undefined && numPoints !== this.state.numPoints) {
      changes.push({
        type: 'frequency',
        field: 'numPoints',
        oldValue: this.state.numPoints,
        newValue: numPoints,
        timestamp: Date.now()
      });
      this.state.numPoints = numPoints;
    }

    // Synchronize frequency range in parameters
    if (changes.length > 0) {
      this.state.parameters.frequency_range = [this.state.minFreq, this.state.maxFreq];
      this.updateStateMetadata();

      changes.forEach(change => this.notifyListeners(change));
      console.log(`🔄 Frequency range updated:`, changes.map(c => `${c.field}: ${c.oldValue} → ${c.newValue}`));
    }
  }

  /**
   * Update memory optimization settings
   */
  updateOptimizationSettings(maxVisibleResults?: number, enabled?: boolean): void {
    const changes: ParameterChangeEvent[] = [];

    if (maxVisibleResults !== undefined && maxVisibleResults !== this.state.maxVisibleResults) {
      changes.push({
        type: 'optimization',
        field: 'maxVisibleResults',
        oldValue: this.state.maxVisibleResults,
        newValue: maxVisibleResults,
        timestamp: Date.now()
      });
      this.state.maxVisibleResults = maxVisibleResults;
    }

    if (enabled !== undefined && enabled !== this.state.memoryOptimizationEnabled) {
      changes.push({
        type: 'optimization',
        field: 'memoryOptimizationEnabled',
        oldValue: this.state.memoryOptimizationEnabled,
        newValue: enabled,
        timestamp: Date.now()
      });
      this.state.memoryOptimizationEnabled = enabled;
    }

    if (changes.length > 0) {
      this.updateStateMetadata();
      changes.forEach(change => this.notifyListeners(change));
      console.log(`⚙️ Optimization settings updated:`, changes.map(c => `${c.field}: ${c.oldValue} → ${c.newValue}`));
    }
  }

  /**
   * Get serialized configuration ID for current parameters
   */
  getConfigId(): string | null {
    try {
      const serializedParams: SerializedCircuitParameters = {
        rsh: this.state.parameters.Rsh,
        ra: this.state.parameters.Ra,
        ca: this.state.parameters.Ca,
        rb: this.state.parameters.Rb,
        cb: this.state.parameters.Cb
      };

      // Find matching config ID
      return this.findConfigIdForParameters(serializedParams);
    } catch (error) {
      console.warn('Failed to generate config ID:', error);
      return null;
    }
  }

  /**
   * Get memory usage estimation
   */
  getMemoryEstimation(): {
    totalPoints: number;
    traditionalSizeMB: number;
    optimizedSizeMB: number;
    reductionFactor: number;
    isHighMemory: boolean;
  } {
    const totalPoints = Math.pow(this.state.gridSize, 5);
    const traditionalSizeMB = totalPoints * 4000 / (1024 * 1024); // 4KB per model
    const optimizedSizeMB = totalPoints * 61 / (1024 * 1024); // 61 bytes per serialized result
    const reductionFactor = traditionalSizeMB / optimizedSizeMB;

    return {
      totalPoints,
      traditionalSizeMB,
      optimizedSizeMB,
      reductionFactor,
      isHighMemory: traditionalSizeMB > 100 // > 100MB considered high memory
    };
  }

  /**
   * Subscribe to parameter changes
   */
  subscribe(listener: (state: ParameterConfigState, change?: ParameterChangeEvent) => void): () => void {
    this.listeners.push(listener);

    // Return unsubscribe function
    return () => {
      const index = this.listeners.indexOf(listener);
      if (index > -1) {
        this.listeners.splice(index, 1);
      }
    };
  }

  /**
   * Get configuration hash for change detection
   */
  getConfigHash(): string {
    return this.state.configHash;
  }

  /**
   * Reset to default configuration
   */
  reset(): void {
    const oldState = { ...this.state };

    this.state = {
      parameters: {
        Rsh: 1000,
        Ra: 2000,
        Ca: 1e-6,
        Rb: 3000,
        Cb: 2e-6,
        frequency_range: [0.1, 100000]
      },
      gridSize: 9,
      minFreq: 0.1,
      maxFreq: 100000,
      numPoints: 100,
      maxVisibleResults: 1000,
      memoryOptimizationEnabled: true,
      lastUpdated: Date.now(),
      configHash: ''
    };

    this.configSerializer = new ConfigSerializer(this.state.gridSize);
    this.state.configHash = this.calculateConfigHash();

    const change: ParameterChangeEvent = {
      type: 'parameter',
      field: 'reset',
      oldValue: oldState,
      newValue: this.state,
      timestamp: Date.now()
    };

    this.notifyListeners(change);
    console.log('🔄 Configuration reset to defaults');
  }

  // Private methods

  private updateStateMetadata(): void {
    this.state.lastUpdated = Date.now();
    this.state.configHash = this.calculateConfigHash();
  }

  private calculateConfigHash(): string {
    const hashInput = JSON.stringify({
      params: this.state.parameters,
      grid: this.state.gridSize,
      freq: [this.state.minFreq, this.state.maxFreq, this.state.numPoints]
    });

    // Simple hash function for change detection
    let hash = 0;
    for (let i = 0; i < hashInput.length; i++) {
      const char = hashInput.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }

    return hash.toString(36);
  }

  private findConfigIdForParameters(params: SerializedCircuitParameters): string | null {
    try {
      // Use relative tolerance matching for parameter identification
      const relativeMatch = (expected: number, actual: number, tolerance: number): boolean => {
        if (expected === 0 && actual === 0) return true;
        if (expected === 0 || actual === 0) return false;
        return Math.abs((expected - actual) / expected) < tolerance;
      };

      // Search parameter grid for closest match
      for (let rsh = 1; rsh <= this.state.gridSize; rsh++) {
        for (let ra = 1; ra <= this.state.gridSize; ra++) {
          for (let ca = 1; ca <= this.state.gridSize; ca++) {
            for (let rb = 1; rb <= this.state.gridSize; rb++) {
              for (let cb = 1; cb <= this.state.gridSize; cb++) {
                const testConfig = this.configSerializer.serializeConfig(ra, rb, rsh, ca, cb);
                const testParams = this.configSerializer.deserializeConfig(testConfig);

                if (
                  relativeMatch(testParams.rsh, params.rsh, 0.01) &&
                  relativeMatch(testParams.ra, params.ra, 0.01) &&
                  relativeMatch(testParams.ca, params.ca, 0.05) &&
                  relativeMatch(testParams.rb, params.rb, 0.01) &&
                  relativeMatch(testParams.cb, params.cb, 0.05)
                ) {
                  return testConfig.toString();
                }
              }
            }
          }
        }
      }
    } catch (error) {
      console.warn('Parameter lookup failed:', error);
    }

    return null;
  }

  private notifyListeners(change?: ParameterChangeEvent): void {
    this.listeners.forEach(listener => {
      try {
        listener(this.state, change);
      } catch (error) {
        console.error('Error in parameter change listener:', error);
      }
    });
  }
}

/**
 * Factory function for creating parameter configuration manager
 */
export function createParameterConfigManager(initialConfig?: Partial<ParameterConfigState>): ParameterConfigManager {
  return new ParameterConfigManager(initialConfig);
}