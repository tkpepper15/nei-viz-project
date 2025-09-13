/**
 * Serialized Computation Hook
 * ==========================
 * 
 * Integrates serialized computation management with existing spider plot rendering.
 * Provides seamless transition from traditional BackendMeshPoint storage to serialized.
 */

import { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import { BackendMeshPoint, ModelSnapshot, ResnormGroup } from '../types';
import { 
  SerializedComputationManager, 
  createSerializedComputationManager
} from '../utils/serializedComputationManager';

// Hook configuration
interface UseSerializedComputationConfig {
  gridSize: number;
  frequencyPreset: string;
  enableSerialization: boolean;
  maxCacheSize: number;
  autoMigrationEnabled: boolean;
}

// Hook return interface
interface UseSerializedComputationReturn {
  // Core management
  manager: SerializedComputationManager | null;
  isSerializationEnabled: boolean;
  
  // Result storage and retrieval
  storeResults: (results: BackendMeshPoint[]) => Promise<number>;
  getBackendMeshPoints: (maxResults?: number) => BackendMeshPoint[];
  getModelSnapshots: (maxResults?: number) => ModelSnapshot[];
  
  // Filtering and analysis
  filterByParameters: (filters: Partial<{
    Rsh: { min: number; max: number };
    Ra: { min: number; max: number };
    Ca: { min: number; max: number };
    Rb: { min: number; max: number };
    Cb: { min: number; max: number };
  }>) => ModelSnapshot[];
  filterByResnorm: (min: number, max: number) => ModelSnapshot[];
  getBestResults: (n?: number) => ModelSnapshot[];
  
  // Pagination support with global resnorm ordering
  getPaginatedResults: (pageNumber: number, pageSize?: number) => {
    results: ModelSnapshot[];
    totalPages: number;
    totalResults: number;
    currentPage: number;
    hasNextPage: boolean;
    hasPrevPage: boolean;
  };
  
  // Statistics and monitoring
  getStorageStats: () => {
    totalResults: number;
    traditionalSizeMB: number;
    serializedSizeMB: number;
    reductionFactor: number;
    cacheStats: {
      parameterCacheSize: number;
      spectrumCacheSize: number;
      modelCacheSize: number;
    };
  } | null;
  
  // Migration utilities
  migrateFromBackendResults: (results: BackendMeshPoint[]) => Promise<void>;
  clearCache: () => void;
  
  // Integration helpers
  generateResnormGroups: (maxResults?: number) => ResnormGroup[];
  isCompatibleWithExistingSystem: boolean;
  
  // Status
  hasResults: boolean;
  resultCount: number;
  isProcessing: boolean;
}

const DEFAULT_CONFIG: UseSerializedComputationConfig = {
  gridSize: 15,
  frequencyPreset: 'standard',
  enableSerialization: true,
  maxCacheSize: 10000,
  autoMigrationEnabled: true
};

export function useSerializedComputation(
  config: Partial<UseSerializedComputationConfig> = {}
): UseSerializedComputationReturn {
  
  const finalConfig = { ...DEFAULT_CONFIG, ...config };
  
  // State management
  const [manager, setManager] = useState<SerializedComputationManager | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [hasResults, setHasResults] = useState(false);
  const [resultCount, setResultCount] = useState(0);
  
  // Cache for expensive operations
  const modelSnapshotCache = useRef(new Map<string, ModelSnapshot[]>());
  const backendPointsCache = useRef(new Map<string, BackendMeshPoint[]>());
  const resnormGroupsCache = useRef<ResnormGroup[] | null>(null);
  
  // Initialize manager
  useEffect(() => {
    if (finalConfig.enableSerialization && !manager) {
      const newManager = createSerializedComputationManager(
        finalConfig.gridSize, 
        finalConfig.frequencyPreset
      );
      setManager(newManager);
      console.log(`🚀 SerializedComputation hook initialized: ${finalConfig.gridSize}x5 grid`);
    }
  }, [finalConfig.enableSerialization, finalConfig.gridSize, finalConfig.frequencyPreset, manager]);
  
  // Store computation results
  const storeResults = useCallback(async (results: BackendMeshPoint[]): Promise<number> => {
    if (!manager || !finalConfig.enableSerialization) {
      console.warn('❌ Serialization not enabled - results not stored');
      return 0;
    }
    
    setIsProcessing(true);
    
    try {
      const storedCount = manager.storeResults(results, finalConfig.frequencyPreset);
      setHasResults(storedCount > 0);
      setResultCount(storedCount);
      
      // Clear caches when new results are stored
      modelSnapshotCache.current.clear();
      backendPointsCache.current.clear();
      resnormGroupsCache.current = null;
      
      console.log(`✅ Stored ${storedCount}/${results.length} results in serialized format`);
      return storedCount;
    } catch (error) {
      console.error('❌ Error storing serialized results:', error);
      return 0;
    } finally {
      setIsProcessing(false);
    }
  }, [manager, finalConfig.enableSerialization, finalConfig.frequencyPreset]);
  
  // Get BackendMeshPoints (procedurally generated)
  const getBackendMeshPoints = useCallback((maxResults?: number): BackendMeshPoint[] => {
    if (!manager || !finalConfig.enableSerialization) return [];
    
    const cacheKey = `backend_${maxResults || 'all'}`;
    
    if (backendPointsCache.current.has(cacheKey)) {
      return backendPointsCache.current.get(cacheKey)!;
    }
    
    const points = manager.generateBackendMeshPoints(maxResults);
    backendPointsCache.current.set(cacheKey, points);
    
    return points;
  }, [manager, finalConfig.enableSerialization]);
  
  // Get ModelSnapshots for spider plot (procedurally generated)
  const getModelSnapshots = useCallback((maxResults?: number): ModelSnapshot[] => {
    if (!manager || !finalConfig.enableSerialization) return [];
    
    const cacheKey = `models_${maxResults || 'all'}`;
    
    if (modelSnapshotCache.current.has(cacheKey)) {
      return modelSnapshotCache.current.get(cacheKey)!;
    }
    
    const snapshots = manager.generateModelSnapshots(maxResults);
    modelSnapshotCache.current.set(cacheKey, snapshots);
    
    return snapshots;
  }, [manager, finalConfig.enableSerialization]);
  
  // Filter by circuit parameters 
  const filterByParameters = useCallback((filters: Partial<{
    Rsh: { min: number; max: number };
    Ra: { min: number; max: number };
    Ca: { min: number; max: number };
    Rb: { min: number; max: number };
    Cb: { min: number; max: number };
  }>): ModelSnapshot[] => {
    if (!manager) return [];
    
    const serializedResults = manager.filterByParameters(filters);
    
    // Convert to ModelSnapshots
    return serializedResults.map((result) => {
      // This could be cached or optimized further
      const tempManager = createSerializedComputationManager(finalConfig.gridSize);
      tempManager['results'] = [result]; // Direct access for single result
      return tempManager.generateModelSnapshots(1)[0];
    }).filter(Boolean);
  }, [manager, finalConfig.gridSize]);
  
  // Filter by resnorm range
  const filterByResnorm = useCallback((min: number, max: number): ModelSnapshot[] => {
    if (!manager) return [];
    
    const serializedResults = manager.filterByResnorm(min, max);
    
    // Convert to ModelSnapshots  
    return serializedResults.map((result) => {
      const tempManager = createSerializedComputationManager(finalConfig.gridSize);
      tempManager['results'] = [result]; // Direct access for single result
      return tempManager.generateModelSnapshots(1)[0];
    }).filter(Boolean);
  }, [manager, finalConfig.gridSize]);
  
  // Get best N results
  const getBestResults = useCallback((n: number = 1000): ModelSnapshot[] => {
    if (!manager) return [];
    
    // Use the manager's new default of 1000 results for optimal performance
    const serializedResults = manager.getBestResults(n);
    
    // Convert to ModelSnapshots
    return serializedResults.map((result) => {
      const tempManager = createSerializedComputationManager(finalConfig.gridSize);
      tempManager['results'] = [result]; // Direct access for single result
      return tempManager.generateModelSnapshots(1)[0];
    }).filter(Boolean);
  }, [manager, finalConfig.gridSize]);
  
  // Generate resnorm groups for existing visualization components
  const generateResnormGroups = useCallback((maxResults?: number): ResnormGroup[] => {
    if (!manager) return [];
    
    if (resnormGroupsCache.current && !maxResults) {
      return resnormGroupsCache.current;
    }
    
    const snapshots = getModelSnapshots(maxResults);
    
    // Sort by resnorm
    const sortedSnapshots = [...snapshots].sort((a, b) => a.resnorm! - b.resnorm!);
    
    if (sortedSnapshots.length === 0) return [];
    
    // Create percentile groups (matching existing system)
    const groups: ResnormGroup[] = [
      {
        range: [0, 0.25],
        color: '#10B981', // Green
        label: 'Excellent (Top 25%)',
        description: 'Best performing models',
        items: []
      },
      {
        range: [0.25, 0.5],
        color: '#F59E0B', // Yellow
        label: 'Good (25-50%)',
        description: 'Above average performance',
        items: []
      },
      {
        range: [0.5, 0.75],
        color: '#F97316', // Orange
        label: 'Fair (50-75%)',
        description: 'Below average performance',
        items: []
      },
      {
        range: [0.75, 1.0],
        color: '#EF4444', // Red
        label: 'Poor (Bottom 25%)',
        description: 'Lowest performing models',
        items: []
      }
    ];
    
    // Distribute models into groups based on percentile
    const totalCount = sortedSnapshots.length;
    
    sortedSnapshots.forEach((snapshot, index) => {
      const percentile = index / totalCount;
      
      let groupIndex = 3; // Default to last group
      if (percentile < 0.25) groupIndex = 0;
      else if (percentile < 0.5) groupIndex = 1;
      else if (percentile < 0.75) groupIndex = 2;
      
      groups[groupIndex].items.push(snapshot);
    });
    
    // Filter out empty groups
    const nonEmptyGroups = groups.filter(group => group.items.length > 0);
    
    if (!maxResults) {
      resnormGroupsCache.current = nonEmptyGroups;
    }
    
    return nonEmptyGroups;
  }, [manager, getModelSnapshots]);
  
  // Migration from existing BackendMeshPoint results
  const migrateFromBackendResults = useCallback(async (results: BackendMeshPoint[]): Promise<void> => {
    if (!finalConfig.autoMigrationEnabled) return;
    
    console.log(`🔄 Migrating ${results.length} BackendMeshPoints to serialized format`);
    
    try {
      await storeResults(results);
      console.log('✅ Migration completed successfully');
    } catch (error) {
      console.error('❌ Migration failed:', error);
    }
  }, [finalConfig.autoMigrationEnabled, storeResults]);
  
  // Get storage statistics
  const getStorageStats = useCallback(() => {
    return manager?.getStorageStats() || null;
  }, [manager]);
  
  // Clear all caches
  const clearCache = useCallback(() => {
    manager?.clearCaches();
    modelSnapshotCache.current.clear();
    backendPointsCache.current.clear();
    resnormGroupsCache.current = null;
    console.log('🧹 All caches cleared');
  }, [manager]);
  
  // Get paginated results with global resnorm ordering
  const getPaginatedResults = useCallback((pageNumber: number, pageSize: number = 100) => {
    if (!manager) {
      return {
        results: [],
        totalPages: 0,
        totalResults: 0,
        currentPage: pageNumber,
        hasNextPage: false,
        hasPrevPage: false
      };
    }
    
    const paginationInfo = manager.getPaginatedResults(pageNumber, pageSize);
    
    // Convert serialized results to ModelSnapshots
    const modelSnapshots = paginationInfo.results.map((result) => {
      const tempManager = createSerializedComputationManager(finalConfig.gridSize);
      tempManager['results'] = [result]; // Direct access for single result
      return tempManager.generateModelSnapshots(1)[0];
    }).filter(Boolean);
    
    return {
      ...paginationInfo,
      results: modelSnapshots
    };
  }, [manager, finalConfig.gridSize]);
  
  // Compatibility check
  const isCompatibleWithExistingSystem = useMemo(() => {
    return finalConfig.enableSerialization && manager !== null;
  }, [finalConfig.enableSerialization, manager]);
  
  return {
    // Core management
    manager,
    isSerializationEnabled: finalConfig.enableSerialization,
    
    // Result storage and retrieval
    storeResults,
    getBackendMeshPoints,
    getModelSnapshots,
    
    // Filtering and analysis
    filterByParameters,
    filterByResnorm,
    getBestResults,
    getPaginatedResults,
    
    // Statistics and monitoring
    getStorageStats,
    
    // Migration utilities
    migrateFromBackendResults,
    clearCache,
    
    // Integration helpers
    generateResnormGroups,
    isCompatibleWithExistingSystem,
    
    // Status
    hasResults,
    resultCount,
    isProcessing
  };
}

/**
 * Utility hook for seamless integration with existing components
 * Automatically decides whether to use serialized or traditional approach
 */
export function useAdaptiveComputation(
  traditionalResults: BackendMeshPoint[], 
  enableSerialization: boolean = true
) {
  const serializedHook = useSerializedComputation({ 
    enableSerialization,
    autoMigrationEnabled: true 
  });
  
  // Auto-migration effect
  useEffect(() => {
    if (enableSerialization && traditionalResults.length > 0 && !serializedHook.hasResults) {
      serializedHook.migrateFromBackendResults(traditionalResults);
    }
  }, [traditionalResults, enableSerialization, serializedHook]);
  
  // Return appropriate data source
  return {
    ...serializedHook,
    
    // Adaptive methods that fall back to traditional if serialization disabled
    getModelSnapshots: (maxResults?: number) => {
      if (enableSerialization && serializedHook.hasResults) {
        return serializedHook.getModelSnapshots(maxResults);
      }
      
      // Fallback to traditional conversion
      const limited = maxResults ? traditionalResults.slice(0, maxResults) : traditionalResults;
      return limited.map((result, index): ModelSnapshot => ({
        id: `traditional_${index}`,
        name: `Config ${index + 1}`,
        timestamp: Date.now(),
        parameters: result.parameters,
        data: result.spectrum?.map(s => ({
          real: s.real,
          imaginary: s.imag,
          frequency: s.freq,
          magnitude: s.mag,
          phase: s.phase
        })) || [],
        resnorm: result.resnorm,
        color: `hsl(${Math.min(240, Math.max(0, 240 - (Math.log10(result.resnorm + 1) * 40)))}, 70%, 50%)`,
        isVisible: true,
        opacity: 1.0
      }));
    },
    
    // Status that considers both sources
    hasResults: serializedHook.hasResults || traditionalResults.length > 0,
    resultCount: serializedHook.hasResults ? serializedHook.resultCount : traditionalResults.length
  };
}