// Shared Worker Strategy for Both Playground and Orchestrator Tabs
// This utility provides consistent parallel processing across the application

import { ModelSnapshot } from '../types';
import { getOrchestratorWorkerManager } from './orchestratorWorkerManager';

interface SharedWorkerOptions {
  threshold?: number; // Dataset size threshold for parallel processing
  maxWorkers?: number; // Maximum number of workers to use
  chunkStrategy?: 'size' | 'count'; // How to split data
  enableProgressTracking?: boolean; // Enable progress tracking
}

interface SharedWorkerCallbacks {
  onProgress?: (progress: number, message: string) => void;
  onChunkComplete?: (chunkIndex: number, totalChunks: number) => void;
  onComplete?: (result: unknown) => void;
  onError?: (error: string) => void;
}

export class SharedWorkerStrategy {
  private static instance: SharedWorkerStrategy | null = null;
  
  private constructor() {}
  
  public static getInstance(): SharedWorkerStrategy {
    if (!SharedWorkerStrategy.instance) {
      SharedWorkerStrategy.instance = new SharedWorkerStrategy();
    }
    return SharedWorkerStrategy.instance;
  }

  /**
   * Determines if parallel processing should be used based on dataset size
   */
  public shouldUseParallelProcessing(
    dataSize: number, 
    options: SharedWorkerOptions = {}
  ): boolean {
    const threshold = options.threshold || 3000;
    return dataSize > threshold;
  }

  /**
   * Splits data into optimal chunks for parallel processing
   */
  public createDataChunks<T>(
    data: T[], 
    options: SharedWorkerOptions = {}
  ): T[][] {
    const maxWorkers = options.maxWorkers || 12;
    const strategy = options.chunkStrategy || 'count';
    
    if (strategy === 'size') {
      // Split by chunk size
      const chunkSize = Math.ceil(data.length / maxWorkers);
      const chunks: T[][] = [];
      
      for (let i = 0; i < data.length; i += chunkSize) {
        chunks.push(data.slice(i, i + chunkSize));
      }
      
      return chunks.filter(chunk => chunk.length > 0);
    } else {
      // Split by worker count
      const chunkSize = Math.ceil(data.length / maxWorkers);
      const chunks: T[][] = [];
      
      for (let i = 0; i < maxWorkers && i * chunkSize < data.length; i++) {
        const start = i * chunkSize;
        const end = Math.min(start + chunkSize, data.length);
        chunks.push(data.slice(start, end));
      }
      
      return chunks;
    }
  }

  /**
   * Process spider plot visualization with automatic parallel processing
   */
  public async processSpiderPlotVisualization(
    meshData: ModelSnapshot[],
    renderParams: unknown,
    options: SharedWorkerOptions = {},
    callbacks: SharedWorkerCallbacks = {}
  ): Promise<string> {
    
    console.log(`🔄 [SharedWorkerStrategy] Processing ${meshData.length} models...`);
    
    // Determine processing strategy
    if (!this.shouldUseParallelProcessing(meshData.length, options)) {
      console.log(`📊 [SharedWorkerStrategy] Using single-thread processing (${meshData.length} < ${options.threshold || 3000})`);
      return this.processSingleThread(meshData, renderParams, callbacks);
    }
    
    console.log(`⚡ [SharedWorkerStrategy] Using parallel processing (${meshData.length} models)`);
    return this.processParallel(meshData, renderParams, options, callbacks);
  }

  /**
   * Single-threaded processing for smaller datasets
   */
  private async processSingleThread(
    meshData: ModelSnapshot[],
    renderParams: unknown,
    callbacks: SharedWorkerCallbacks
  ): Promise<string> {
    return new Promise((resolve, reject) => {
      const jobId = `single_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      getOrchestratorWorkerManager().addJob(
        jobId,
        meshData,
        renderParams,
        'high',
        {
          onProgress: (progress) => {
            callbacks.onProgress?.(progress, `Processing ${meshData.length} models...`);
          },
          onComplete: (result) => {
            if (result.success && result.imageUrl) {
              callbacks.onComplete?.(result);
              resolve(result.imageUrl);
            } else {
              const error = result.error || 'Unknown rendering error';
              callbacks.onError?.(error);
              reject(new Error(error));
            }
          },
          onError: (error) => {
            callbacks.onError?.(error);
            reject(new Error(error));
          }
        }
      );
    });
  }

  /**
   * Parallel processing for large datasets
   */
  private async processParallel(
    meshData: ModelSnapshot[],
    renderParams: unknown,
    options: SharedWorkerOptions,
    callbacks: SharedWorkerCallbacks
  ): Promise<string> {
    
    // Create data chunks
    const chunks = this.createDataChunks(meshData, options);
    
    console.log(`🔀 [SharedWorkerStrategy] Created ${chunks.length} chunks (${chunks.map(c => c.length).join(', ')} models each)`);
    
    // Create jobs for each chunk
    const chunkJobs = chunks.map((chunk, index) => {
      const chunkJobId = `parallel_${Date.now()}_chunk_${index}`;
      
      return new Promise<string>((resolve, reject) => {
        getOrchestratorWorkerManager().addJob(
          chunkJobId,
          chunk,
          {
            ...renderParams,
            chunkIndex: index,
            totalChunks: chunks.length
          },
          'high',
          {
            onProgress: (progress) => {
              const chunkProgress = Math.round(progress * 100);
              const overallProgress = ((index + progress) / chunks.length) * 100;
              
              callbacks.onProgress?.(
                overallProgress / 100,
                `Processing chunk ${index + 1}/${chunks.length} (${chunkProgress}%)`
              );
            },
            onComplete: (result) => {
              if (result.success && result.imageUrl) {
                console.log(`✅ Chunk ${index + 1}/${chunks.length} completed (${chunk.length} models)`);
                callbacks.onChunkComplete?.(index, chunks.length);
                resolve(result.imageUrl);
              } else {
                const error = result.error || `Chunk ${index} rendering error`;
                reject(new Error(error));
              }
            },
            onError: (error) => {
              reject(new Error(`Chunk ${index} error: ${error}`));
            }
          }
        );
      });
    });

    try {
      // Wait for all chunks to complete
      const chunkResults = await Promise.all(chunkJobs);
      
      console.log(`🎨 [SharedWorkerStrategy] All ${chunks.length} chunks completed. Using first result.`);
      
      // For now, return the first chunk result
      // TODO: Implement canvas composition for true parallel rendering
      const finalResult = chunkResults[0];
      callbacks.onComplete?.({ imageUrl: finalResult });
      
      return finalResult;
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown parallel processing error';
      console.error('❌ [SharedWorkerStrategy] Parallel processing failed:', errorMessage);
      callbacks.onError?.(errorMessage);
      throw error;
    }
  }

  /**
   * Get optimal processing configuration based on dataset characteristics
   */
  public getOptimalConfiguration(
    dataSize: number,
    complexity: 'low' | 'medium' | 'high' = 'medium'
  ): SharedWorkerOptions {
    // Base configuration with 12 workers by default
    const config: SharedWorkerOptions = {
      threshold: 3000,
      maxWorkers: 12,
      chunkStrategy: 'count',
      enableProgressTracking: true
    };

    // Adjust based on dataset size - use 12 workers for better performance
    if (dataSize > 50000) {
      config.threshold = 2000; // Use parallel processing earlier
      config.maxWorkers = 12;
    } else if (dataSize > 10000) {
      config.threshold = 3000;
      config.maxWorkers = 12;
    } else {
      config.maxWorkers = 12;
    }

    // Adjust based on complexity
    if (complexity === 'high') {
      config.threshold = 1500; // Use parallel processing much earlier
      config.maxWorkers = 12;
      config.chunkStrategy = 'size'; // Better for complex processing
    } else if (complexity === 'low') {
      config.threshold = 5000; // Use single-thread longer
      config.maxWorkers = 8;
    }

    return config;
  }

  /**
   * Estimate processing time based on dataset size and configuration
   */
  public estimateProcessingTime(
    dataSize: number,
    options: SharedWorkerOptions = {}
  ): { estimatedSeconds: number; complexity: string } {
    const baseTimePerModel = 0.001; // ~1ms per model
    const parallelSpeedup = options.maxWorkers || 6;
    
    let estimatedSeconds: number;
    let complexity: string;
    
    if (this.shouldUseParallelProcessing(dataSize, options)) {
      estimatedSeconds = (dataSize * baseTimePerModel) / parallelSpeedup;
      complexity = dataSize > 50000 ? 'Very High' : dataSize > 10000 ? 'High' : 'Medium';
    } else {
      estimatedSeconds = dataSize * baseTimePerModel;
      complexity = 'Low';
    }
    
    return {
      estimatedSeconds: Math.max(0.1, estimatedSeconds),
      complexity
    };
  }
}

// Export singleton instance
export const sharedWorkerStrategy = SharedWorkerStrategy.getInstance();