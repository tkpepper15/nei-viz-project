/**
 * Serialized Compute Integration
 * =============================
 * 
 * Integrates the serialized computation system into the main compute flow.
 * Automatically stores results in serialized format when compute completes.
 */

import { BackendMeshPoint, ModelSnapshot } from '../types';
import { SerializedComputationManager, createSerializedComputationManager } from './serializedComputationManager';

export interface SerializedComputeResult {
  traditionalResults: BackendMeshPoint[];
  serializedManager: SerializedComputationManager;
  storageStats: {
    traditionalSizeMB: number;
    serializedSizeMB: number;
    reductionFactor: number;
    storedCount: number;
  };
  modelSnapshots: ModelSnapshot[];
}

/**
 * Integrates serialized storage into the existing compute pipeline
 */
export async function integrateSerializedStorage(
  computeResults: BackendMeshPoint[],
  gridSize: number = 9,
  frequencyPreset: string = 'standard'
): Promise<SerializedComputeResult> {
  
  console.log(`🔄 Integrating ${computeResults.length} results with serialized storage...`);
  const startTime = Date.now();
  
  // Create serialized computation manager
  const serializedManager = createSerializedComputationManager(gridSize, frequencyPreset);
  
  // Store results in serialized format
  const storedCount = await serializedManager.storeResults(computeResults, frequencyPreset);
  
  // Generate ModelSnapshots from serialized data (procedural)
  const modelSnapshots = serializedManager.generateModelSnapshots();
  
  // Get storage statistics
  const stats = serializedManager.getStorageStats();
  
  const storageStats = {
    traditionalSizeMB: stats?.traditionalSizeMB || 0,
    serializedSizeMB: stats?.serializedSizeMB || 0,
    reductionFactor: stats?.reductionFactor || 1,
    storedCount
  };
  
  const duration = Date.now() - startTime;
  console.log(`✅ Serialized integration complete in ${duration}ms:`);
  console.log(`📊 Storage: ${storageStats.traditionalSizeMB.toFixed(1)}MB → ${storageStats.serializedSizeMB.toFixed(1)}MB (${storageStats.reductionFactor.toFixed(0)}x smaller)`);
  console.log(`🎨 Generated ${modelSnapshots.length} ModelSnapshots procedurally`);
  
  return {
    traditionalResults: computeResults,
    serializedManager,
    storageStats,
    modelSnapshots
  };
}

/**
 * Hook integration helper - returns enhanced state with serialization
 */
export function useSerializedComputeIntegration(
  enabled: boolean = true
) {
  return {
    integrateResults: (results: BackendMeshPoint[], gridSize?: number, frequencyPreset?: string) => {
      if (!enabled) {
        return null;
      }
      return integrateSerializedStorage(results, gridSize, frequencyPreset);
    },
    isEnabled: enabled
  };
}

/**
 * Drop-in replacement for traditional ModelSnapshot generation
 * Uses serialized system for better efficiency
 */
export function generateModelSnapshotsFromSerialized(
  serializedManager: SerializedComputationManager,
  maxResults?: number
): ModelSnapshot[] {
  return serializedManager.generateModelSnapshots(maxResults);
}

/**
 * Helper to convert traditional compute flow to use serialized storage
 */
export async function enhanceComputeResultsWithSerialization(
  computeFunction: () => Promise<{ results: BackendMeshPoint[] }>,
  gridSize: number = 9,
  frequencyPreset: string = 'standard',
  onSerializationComplete?: (stats: SerializedComputeResult['storageStats']) => void
): Promise<SerializedComputeResult> {
  
  // Run traditional computation
  const computeResult = await computeFunction();
  
  // Integrate with serialized storage
  const serializedResult = await integrateSerializedStorage(
    computeResult.results,
    gridSize,
    frequencyPreset
  );
  
  // Notify about serialization completion
  if (onSerializationComplete) {
    onSerializationComplete(serializedResult.storageStats);
  }
  
  return serializedResult;
}

/**
 * Demo function showing how to integrate serialization into existing compute flow
 */
export function demonstrateComputeIntegration() {
  console.log(`
🎯 SERIALIZED COMPUTE INTEGRATION READY!

To integrate with existing compute flow:

1. After your existing computation completes:
   const serializedResult = await integrateSerializedStorage(
     computeResults,    // Your existing BackendMeshPoint[]
     gridSize,         // Grid size used in computation
     'standard'        // Frequency preset
   );

2. Use the procedurally generated ModelSnapshots:
   const modelSnapshots = serializedResult.modelSnapshots;
   // These are generated from serialized data, not stored arrays!

3. Storage efficiency achieved:
   console.log(\`Storage reduced by \${serializedResult.storageStats.reductionFactor}x\`);

4. Continue using existing visualization:
   <SpiderPlot3D models={modelSnapshots} />
   // Works exactly the same, but data comes from serialized system!

Your compute flow remains the same, but now includes automatic serialization! 🚀
  `);
}