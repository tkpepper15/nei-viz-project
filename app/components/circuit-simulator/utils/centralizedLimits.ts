/**
 * Centralized Computation Limits System
 *
 * This module provides a single, unified control system for all computation limits
 * throughout the application, eliminating scattered hardcoded values.
 */

import { ComputationLimitsConfig } from '../controls/ComputationLimitsControl';

export interface CentralizedLimitsConfig {
  masterLimitPercentage: number;
  totalPossibleResults: number;
  gridSize: number;
}

export class CentralizedLimitsManager {
  private config: CentralizedLimitsConfig;

  constructor(config: CentralizedLimitsConfig) {
    this.config = config;
  }

  // Master limit calculations
  get masterLimitResults(): number {
    return Math.floor((this.config.masterLimitPercentage / 100) * this.config.totalPossibleResults);
  }

  get masterLimitPercentage(): number {
    return this.config.masterLimitPercentage;
  }

  // Worker Manager limits
  get workerManagerLimit(): number {
    return this.masterLimitResults;
  }

  // WebGPU Manager limits
  get webgpuManagerLimit(): number {
    return this.masterLimitResults;
  }

  // Hybrid Compute Manager limits
  get hybridComputeManagerLimit(): number {
    return this.masterLimitResults;
  }

  // Visualization limits
  get visualizationLimit(): number {
    // For visualization, apply intelligent scaling based on master limit
    const masterResults = this.masterLimitResults;

    if (masterResults <= 10000) {
      return masterResults; // Show all for small datasets
    } else if (masterResults <= 100000) {
      return Math.min(50000, masterResults); // Cap at 50K for medium datasets
    } else {
      return Math.min(100000, Math.floor(masterResults * 0.5)); // Show 50% for large datasets
    }
  }

  // Display limits for UI
  get displayLimit(): number {
    const vizLimit = this.visualizationLimit;
    return Math.min(25000, vizLimit); // UI display cap at 25K for performance
  }

  // Memory estimation
  get estimatedMemoryMB(): number {
    const avgModelSize = 2000; // bytes per model
    return (this.masterLimitResults * avgModelSize) / (1024 * 1024);
  }

  // Update master limit
  updateMasterLimit(percentage: number): CentralizedLimitsManager {
    return new CentralizedLimitsManager({
      ...this.config,
      masterLimitPercentage: Math.max(0.1, Math.min(100, percentage))
    });
  }

  // Convert to ComputationLimitsConfig for UI
  toComputationLimitsConfig(): ComputationLimitsConfig {
    return {
      maxComputationResults: this.masterLimitResults,
      maxDisplayResults: this.displayLimit,
      memoryLimitMB: Math.ceil(this.estimatedMemoryMB),
      autoScale: false,
      masterLimitPercentage: this.config.masterLimitPercentage
    };
  }

  // Static factory methods for common scenarios
  static fromGridSize(gridSize: number, masterLimitPercentage: number = 100): CentralizedLimitsManager {
    const totalPossibleResults = Math.pow(gridSize, 5);
    return new CentralizedLimitsManager({
      masterLimitPercentage,
      totalPossibleResults,
      gridSize
    });
  }

  static fromComputationConfig(config: ComputationLimitsConfig, gridSize: number): CentralizedLimitsManager {
    const totalPossibleResults = Math.pow(gridSize, 5);
    return new CentralizedLimitsManager({
      masterLimitPercentage: config.masterLimitPercentage,
      totalPossibleResults,
      gridSize
    });
  }
}

// Global instance for the current session (can be updated)
let globalLimitsManager: CentralizedLimitsManager | null = null;

export const getGlobalLimitsManager = (): CentralizedLimitsManager | null => globalLimitsManager;

export const setGlobalLimitsManager = (manager: CentralizedLimitsManager): void => {
  globalLimitsManager = manager;
};

export const updateGlobalMasterLimit = (percentage: number): void => {
  if (globalLimitsManager) {
    globalLimitsManager = globalLimitsManager.updateMasterLimit(percentage);
  }
};

// Utility functions for backward compatibility
export const getMasterLimitResults = (): number => {
  return globalLimitsManager?.masterLimitResults ?? 500000; // Fallback to 500K
};

export const getVisualizationLimit = (): number => {
  return globalLimitsManager?.visualizationLimit ?? 100000; // Fallback to 100K
};

export const getWorkerManagerLimit = (): number => {
  return globalLimitsManager?.workerManagerLimit ?? 500000; // Fallback to 500K
};

export const getWebGPUManagerLimit = (): number => {
  return globalLimitsManager?.webgpuManagerLimit ?? 500000; // Fallback to 500K
};

export const getHybridComputeManagerLimit = (): number => {
  return globalLimitsManager?.hybridComputeManagerLimit ?? 500000; // Fallback to 500K
};