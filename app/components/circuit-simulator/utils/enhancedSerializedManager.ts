/**
 * Enhanced Serialized Data Manager
 * ================================
 *
 * Intelligently detects existing serialized data and avoids unnecessary recomputation.
 * Integrates seamlessly with the existing rendering pipeline while providing massive performance gains.
 */

import { BackendMeshPoint, ModelSnapshot } from '../types';
import { CircuitParameters } from '../types/parameters';
import { SerializedComputationManager } from './serializedComputationManager';

export interface SerializedDataState {
  hasData: boolean;
  dataCount: number;
  compressionRatio: number;
  gridSize: number;
  frequencyPreset: string;
  lastUpdated: number;
  isReady: boolean;
}

export interface SerializedImportResult {
  success: boolean;
  dataCount: number;
  gridSize: number;
  error?: string;
  compressionInfo?: {
    originalSize: string;
    compressedSize: string;
    ratio: number;
  };
}

/**
 * Enhanced manager that intelligently handles serialized data imports and computation avoidance
 */
export class EnhancedSerializedManager {
  private serializedManager: SerializedComputationManager | null = null;
  private state: SerializedDataState;
  private listeners: ((state: SerializedDataState) => void)[] = [];

  constructor() {
    this.state = {
      hasData: false,
      dataCount: 0,
      compressionRatio: 0,
      gridSize: 9,
      frequencyPreset: 'standard',
      lastUpdated: 0,
      isReady: true
    };
  }

  /**
   * Get current serialized data state
   */
  getState(): SerializedDataState {
    return { ...this.state };
  }

  /**
   * Subscribe to state changes
   */
  subscribe(listener: (state: SerializedDataState) => void): () => void {
    this.listeners.push(listener);
    return () => {
      const index = this.listeners.indexOf(listener);
      if (index > -1) {
        this.listeners.splice(index, 1);
      }
    };
  }

  /**
   * Notify all listeners of state changes
   */
  private notifyListeners(): void {
    this.listeners.forEach(listener => listener(this.state));
  }

  /**
   * Import serialized data from file - this is the key method that should avoid recomputation
   */
  async importSerializedData(file: File): Promise<SerializedImportResult> {
    console.log('📥 Importing serialized data from file:', file.name);

    try {
      // Check if this is an SRD file (Serialized Resnorm Data)
      if (file.name.endsWith('.srd') || file.name.endsWith('.json')) {
        return await this.importSRDFile(file);
      }

      // Check if this is NPZ data that can be converted
      if (file.name.endsWith('.npz')) {
        console.log('🔄 Converting NPZ data to serialized format...');
        return await this.importNPZData(file);
      }

      // Try to parse as JSON data
      const text = await file.text();
      const data = JSON.parse(text);

      return await this.importJSONData(data, file.name);

    } catch (error) {
      console.error('❌ Failed to import serialized data:', error);
      return {
        success: false,
        dataCount: 0,
        gridSize: 9,
        error: `Import failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      };
    }
  }

  /**
   * Import SRD file format (native serialized format)
   */
  private async importSRDFile(file: File): Promise<SerializedImportResult> {
    try {
      console.log('📋 Importing SRD file...');

      // Use the existing SRD import mechanism
      this.serializedManager = await SerializedComputationManager.importFromSRD(file);

      const stats = this.serializedManager.getStorageStats();
      const config = this.serializedManager.getConfig();

      // Update state
      this.state = {
        hasData: true,
        dataCount: stats.totalResults,
        compressionRatio: stats.reductionFactor,
        gridSize: config.gridSize,
        frequencyPreset: config.frequencyPreset,
        lastUpdated: Date.now(),
        isReady: true
      };

      this.notifyListeners();

      console.log(`✅ SRD import successful: ${stats.totalResults.toLocaleString()} results`);

      return {
        success: true,
        dataCount: stats.totalResults,
        gridSize: config.gridSize,
        compressionInfo: {
          originalSize: `${stats.traditionalSizeMB.toFixed(1)}MB`,
          compressedSize: `${stats.serializedSizeMB.toFixed(1)}MB`,
          ratio: stats.reductionFactor
        }
      };

    } catch (error) {
      console.error('❌ SRD import failed:', error);
      throw error;
    }
  }

  /**
   * Import NPZ data and convert to serialized format
   */
  private async importNPZData(_file: File): Promise<SerializedImportResult> { // eslint-disable-line @typescript-eslint/no-unused-vars
    try {
      console.log('🔄 Converting NPZ to serialized format...');

      // For now, return an error since NPZ processing requires additional setup
      // In a full implementation, this would extract NPZ data and convert it
      return {
        success: false,
        dataCount: 0,
        gridSize: 9,
        error: 'NPZ import not yet implemented. Please use JSON or SRD format.'
      };

    } catch (error) {
      console.error('❌ NPZ import failed:', error);
      throw error;
    }
  }

  /**
   * Import JSON data (common format for computational results)
   */
  private async importJSONData(data: unknown, filename: string): Promise<SerializedImportResult> {
    try {
      console.log('📄 Importing JSON data...');

      // Detect data format and structure
      let backendResults: BackendMeshPoint[] = [];
      let gridSize = 9; // Default

      // Handle different JSON formats
      const dataRecord = data as Record<string, unknown>;

      if (Array.isArray(data)) {
        // Array of results
        backendResults = this.convertArrayToBackendMeshPoints(data);
      } else if (dataRecord.results && Array.isArray(dataRecord.results)) {
        // Object with results array
        backendResults = this.convertArrayToBackendMeshPoints(dataRecord.results);
        gridSize = (dataRecord.gridSize || dataRecord.grid_size || 9) as number;
      } else if (dataRecord.gridResults && Array.isArray(dataRecord.gridResults)) {
        // SpideyPlot format
        backendResults = this.convertSpideyPlotResults(dataRecord.gridResults);
        gridSize = (dataRecord.gridSize || 9) as number;
      } else {
        throw new Error('Unrecognized JSON format. Expected array of results or object with results property.');
      }

      if (backendResults.length === 0) {
        throw new Error('No valid results found in JSON data');
      }

      // Create serialized manager and store results
      this.serializedManager = new SerializedComputationManager({
        gridSize,
        frequencyPreset: 'standard',
        cacheEnabled: true,
        maxCacheSize: Math.max(10000, backendResults.length)
      });

      const storedCount = this.serializedManager.storeResults(backendResults, 'standard');
      const stats = this.serializedManager.getStorageStats();

      // Update state
      this.state = {
        hasData: true,
        dataCount: storedCount,
        compressionRatio: stats.reductionFactor,
        gridSize,
        frequencyPreset: 'standard',
        lastUpdated: Date.now(),
        isReady: true
      };

      this.notifyListeners();

      console.log(`✅ JSON import successful: ${storedCount.toLocaleString()} results from ${filename}`);

      return {
        success: true,
        dataCount: storedCount,
        gridSize,
        compressionInfo: {
          originalSize: `${stats.traditionalSizeMB.toFixed(1)}MB`,
          compressedSize: `${stats.serializedSizeMB.toFixed(1)}MB`,
          ratio: stats.reductionFactor
        }
      };

    } catch (error) {
      console.error('❌ JSON import failed:', error);
      throw error;
    }
  }

  /**
   * Convert generic array to BackendMeshPoint format
   */
  private convertArrayToBackendMeshPoints(data: unknown[]): BackendMeshPoint[] {
    return data.map((item) => {
      // Handle different parameter naming conventions
      const itemAny = item as Record<string, unknown>;
      const parameters: CircuitParameters = {
        Rsh: (itemAny.Rsh || itemAny.rsh || itemAny.Rs || itemAny.rs || 1000) as number,
        Ra: (itemAny.Ra || itemAny.ra || itemAny.R1 || itemAny.r1 || 2000) as number,
        Ca: (itemAny.Ca || itemAny.ca || itemAny.C1 || itemAny.c1 || 1e-6) as number,
        Rb: (itemAny.Rb || itemAny.rb || itemAny.R2 || itemAny.r2 || 3000) as number,
        Cb: (itemAny.Cb || itemAny.cb || itemAny.C2 || itemAny.c2 || 2e-6) as number,
        frequency_range: (itemAny.frequency_range || itemAny.freq_range || [0.1, 100000]) as [number, number]
      };

      return {
        parameters,
        resnorm: (itemAny.resnorm || itemAny.residual_norm || itemAny.error || Math.random()) as number,
        spectrum: (itemAny.spectrum || itemAny.impedance_spectrum || []) as Array<{freq: number; real: number; imag: number; mag: number; phase: number}>
      };
    });
  }

  /**
   * Convert SpideyPlot specific format to BackendMeshPoint
   */
  private convertSpideyPlotResults(data: unknown[]): BackendMeshPoint[] {
    return data.map((item) => {
      const itemAny = item as Record<string, unknown>;
      return {
        parameters: (itemAny.parameters as CircuitParameters) || {
          Rsh: 1000, Ra: 2000, Ca: 1e-6, Rb: 3000, Cb: 2e-6,
          frequency_range: [0.1, 100000] as [number, number]
        },
        resnorm: (itemAny.resnorm || 0) as number,
        spectrum: (itemAny.data || itemAny.spectrum || []) as Array<{freq: number; real: number; imag: number; mag: number; phase: number}>
      };
    });
  }

  /**
   * Generate ModelSnapshots from existing serialized data - NO RECOMPUTATION
   */
  generateModelSnapshots(maxResults?: number): ModelSnapshot[] {
    if (!this.serializedManager || !this.state.hasData) {
      console.log('⚠️ No serialized data available for model generation');
      return [];
    }

    console.log('🚀 Generating model snapshots from existing serialized data (no recomputation)');

    const snapshots = this.serializedManager.generateModelSnapshots(maxResults);

    console.log(`✅ Generated ${snapshots.length} model snapshots from serialized data`);

    return snapshots;
  }

  /**
   * Generate BackendMeshPoints from existing serialized data - NO RECOMPUTATION
   */
  generateBackendMeshPoints(maxResults?: number): BackendMeshPoint[] {
    if (!this.serializedManager || !this.state.hasData) {
      console.log('⚠️ No serialized data available for backend mesh generation');
      return [];
    }

    console.log('🚀 Generating backend mesh points from existing serialized data (no recomputation)');

    const meshPoints = this.serializedManager.generateBackendMeshPoints(maxResults);

    console.log(`✅ Generated ${meshPoints.length} backend mesh points from serialized data`);

    return meshPoints;
  }

  /**
   * Get paginated results for large datasets
   */
  getPaginatedResults(pageNumber: number, pageSize: number = 1000) {
    if (!this.serializedManager || !this.state.hasData) {
      return {
        modelSnapshots: [],
        pagination: {
          totalPages: 0,
          currentPage: 1,
          hasNextPage: false,
          hasPrevPage: false,
          totalResults: 0,
          resnormRange: { min: 0, max: 0, avgCurrentPage: 0 },
          pageInfo: { startIndex: 0, endIndex: 0 }
        }
      };
    }

    return this.serializedManager.generatePaginatedModelSnapshots(pageNumber, pageSize);
  }

  /**
   * Export current data to SRD format
   */
  exportToSRD(filename: string, metadata: { title: string; description?: string; author?: string }): boolean {
    if (!this.serializedManager || !this.state.hasData) {
      console.error('❌ No data available for export');
      return false;
    }

    try {
      this.serializedManager.exportToSRD(filename, metadata);
      console.log(`✅ Exported data to ${filename}`);
      return true;
    } catch (error) {
      console.error('❌ Export failed:', error);
      return false;
    }
  }

  /**
   * Clear all serialized data
   */
  clearData(): void {
    if (this.serializedManager) {
      this.serializedManager.clearCaches();
      this.serializedManager = null;
    }

    this.state = {
      hasData: false,
      dataCount: 0,
      compressionRatio: 0,
      gridSize: 9,
      frequencyPreset: 'standard',
      lastUpdated: 0,
      isReady: true
    };

    this.notifyListeners();
    console.log('🧹 Serialized data cleared');
  }

  /**
   * Check if the manager has usable data
   */
  hasUsableData(): boolean {
    return this.state.hasData && this.state.dataCount > 0;
  }

  /**
   * Get performance statistics
   */
  getPerformanceStats() {
    if (!this.serializedManager) {
      return null;
    }

    return this.serializedManager.getStorageStats();
  }

  /**
   * Check if current grid size matches serialized data
   */
  isGridSizeCompatible(gridSize: number): boolean {
    return this.state.gridSize === gridSize;
  }
}

/**
 * Factory function to create enhanced serialized manager
 */
export function createEnhancedSerializedManager(): EnhancedSerializedManager {
  return new EnhancedSerializedManager();
}