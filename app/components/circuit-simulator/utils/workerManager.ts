import { useRef, useCallback, useEffect } from 'react';
import { BackendMeshPoint } from '../types';
import { CircuitParameters } from '../types/parameters';
import { PerformanceSettings } from '../controls/PerformanceControls';
import { ResnormConfig } from './resnorm';

export interface WorkerProgress {
  type: 'CHUNK_PROGRESS' | 'GENERATION_PROGRESS' | 'COMPUTATION_START' | 'MATHEMATICAL_OPERATION' | 'MEMORY_STATUS' | 'WORKER_STATUS' | 'STREAMING_UPDATE' | 'THROTTLE_UPDATE';
  chunkIndex?: number;
  totalChunks?: number;
  chunkProgress?: number;
  processed?: number;
  total: number;
  generated?: number;
  skipped?: number;
  overallProgress: number;
  memoryPressure?: boolean;
  // Enhanced logging fields
  message?: string;
  equation?: string;
  operation?: string;
  phase?: 'initialization' | 'grid_generation' | 'impedance_calculation' | 'resnorm_analysis' | 'data_aggregation' | 'completion';
  workerCount?: number;
  chunkSize?: number;
  frequency?: number;
  parametersProcessed?: number;
  // Enhanced responsiveness fields
  streamingBatch?: number;
  throttleDelay?: number;
  mainThreadLoad?: 'low' | 'medium' | 'high';
  adaptiveChunkSize?: number;
}

export interface WorkerResult {
  topResults: Array<{
    parameters: CircuitParameters;
    resnorm: number;
    spectrum: Array<{ freq: number; real: number; imag: number; mag: number; phase: number }>;
  }>;
  otherResults: Array<{
    parameters: CircuitParameters;
    resnorm: number;
  }>;
  totalProcessed: number;
}

export interface UseWorkerManagerReturn {
  computeGridParallel: (
    groundTruthParams: CircuitParameters,
    gridSize: number,
    minFreq: number,
    maxFreq: number,
    numPoints: number,
    performanceSettings: PerformanceSettings,
    resnormConfig: ResnormConfig,
    onProgress: (progress: WorkerProgress) => void,
    onError: (error: string) => void
  ) => Promise<BackendMeshPoint[]>;
  cancelComputation: () => void;
  isComputing: boolean;
}

interface WorkerPoolItem {
  worker: Worker;
  isIdle: boolean;
  lastUsed: number;
  taskId: string | null;
}

export function useWorkerManager(): UseWorkerManagerReturn {
  const workerPoolRef = useRef<WorkerPoolItem[]>([]);
  const isComputingRef = useRef(false);
  const cancelTokenRef = useRef<{ cancelled: boolean }>({ cancelled: false });
  const activePromisesRef = useRef<Set<() => void>>(new Set());
  const workerTimeoutsRef = useRef<Map<string, NodeJS.Timeout>>(new Map());
  const sharedDataRef = useRef<{
    frequencies: number[] | null;
    referenceSpectrum: Array<{ freq: number; real: number; imag: number; mag: number; phase: number }> | null;
    isInitialized: boolean;
  }>({ frequencies: null, referenceSpectrum: null, isInitialized: false });

  // Optimized worker allocation to prevent halfway stalls
  const getOptimalWorkerCount = useCallback((totalPoints: number) => {
    const cores = navigator.hardwareConcurrency || 4;
    const memoryMB = (performance as unknown as { memory?: { usedJSHeapSize?: number } })?.memory?.usedJSHeapSize ? 
      Math.round((performance as unknown as { memory: { usedJSHeapSize: number } }).memory.usedJSHeapSize / 1024 / 1024) : 100;
    
    console.log(`Worker allocation: ${totalPoints} points, ${cores} cores, ~${memoryMB}MB memory`);
    
    // Ultra-conservative allocation for very large datasets to prevent stalls
    if (totalPoints > 500000) {
      console.log('🔴 Massive dataset detected - using minimal workers to prevent stalls');
      return Math.min(2, cores); // Only 2 workers for massive datasets
    }
    if (totalPoints > 200000) {
      console.log('🟠 Very large dataset - conservative worker allocation');
      return Math.min(3, cores); // Conservative for very large
    }
    if (totalPoints > 100000) {
      console.log('🟡 Large dataset - moderate worker allocation');
      return Math.min(4, cores); // Moderate for large
    }
    if (totalPoints > 50000) {
      console.log('🟢 Medium dataset - standard worker allocation');
      return Math.min(cores - 1, 4); // Standard for medium
    }
    
    console.log('🔵 Small dataset - full worker allocation');
    return Math.min(cores, 6); // Full allocation for small datasets
  }, []);

  // Enhanced worker pool management
  const initializeWorkerPool = useCallback((count: number) => {
    // Clean up existing workers
    workerPoolRef.current.forEach(item => item.worker.terminate());
    workerPoolRef.current = [];

    // Create new worker pool
    for (let i = 0; i < count; i++) {
      const worker = new Worker('/grid-worker.js');
      workerPoolRef.current.push({
        worker,
        isIdle: true,
        lastUsed: Date.now(),
        taskId: null
      });
    }
  }, []);

  // Get an idle worker from the pool
  const getIdleWorker = useCallback((): WorkerPoolItem | null => {
    const idleWorker = workerPoolRef.current.find(item => item.isIdle);
    if (idleWorker) {
      idleWorker.isIdle = false;
      idleWorker.lastUsed = Date.now();
      return idleWorker;
    }
    return null;
  }, []);

  // Return worker to pool with cleanup
  const returnWorkerToPool = useCallback((taskId: string) => {
    const worker = workerPoolRef.current.find(item => item.taskId === taskId);
    if (worker) {
      worker.isIdle = true;
      worker.taskId = null;
      
      // Clear any timeouts for this task
      const timeout = workerTimeoutsRef.current.get(taskId);
      if (timeout) {
        clearTimeout(timeout);
        workerTimeoutsRef.current.delete(taskId);
      }
    }
  }, []);

  // Force terminate all workers and clean up resources
  const forceTerminateAllWorkers = useCallback(() => {
    console.log('Force terminating all workers and cleaning up resources...');
    
    // Clear all timeouts
    workerTimeoutsRef.current.forEach(timeout => clearTimeout(timeout));
    workerTimeoutsRef.current.clear();
    
    // Reject all active promises
    activePromisesRef.current.forEach(rejectFn => {
      try {
        rejectFn();
      } catch (error) {
        console.warn('Error rejecting promise during cleanup:', error);
      }
    });
    activePromisesRef.current.clear();
    
    // Terminate all workers
    workerPoolRef.current.forEach(item => {
      try {
        item.worker.terminate();
      } catch (error) {
        console.warn('Error terminating worker:', error);
      }
    });
    workerPoolRef.current = [];
    
    // Reset shared data
    sharedDataRef.current = {
      frequencies: null,
      referenceSpectrum: null,
      isInitialized: false
    };
    
    console.log('All workers terminated and resources cleaned up');
  }, []);

  // Initialize shared data using transferable objects for zero-copy transfer
  const initializeSharedData = useCallback(async (
    frequencies: number[],
    referenceSpectrum: Array<{ freq: number; real: number; imag: number; mag: number; phase: number }>,
    onProgress: (progress: WorkerProgress) => void
  ) => {
    if (sharedDataRef.current.isInitialized) return;

    onProgress({
      type: 'MATHEMATICAL_OPERATION',
      total: frequencies.length,
      overallProgress: 3,
      phase: 'initialization',
      message: 'Converting data to transferable format for zero-copy worker transfer',
      operation: 'Data optimization'
    });

    // Convert to transferable format for efficient transfer
    const frequencyBuffer = new Float64Array(frequencies);
    const spectrumSize = referenceSpectrum.length * 5; // freq, real, imag, mag, phase
    const spectrumBuffer = new Float64Array(spectrumSize);
    
    // Pack spectrum data into typed array
    referenceSpectrum.forEach((point, i) => {
      const offset = i * 5;
      spectrumBuffer[offset] = point.freq;
      spectrumBuffer[offset + 1] = point.real;
      spectrumBuffer[offset + 2] = point.imag;
      spectrumBuffer[offset + 3] = point.mag;
      spectrumBuffer[offset + 4] = point.phase;
    });

    onProgress({
      type: 'MATHEMATICAL_OPERATION',
      total: frequencies.length,
      overallProgress: 5,
      phase: 'initialization',
      message: 'Transferring data to workers using zero-copy transferable objects',
      operation: 'Efficient data transfer'
    });

    // Send transferable data to all workers with zero-copy transfer
    const initPromises = workerPoolRef.current.map((item, index) => {
      return new Promise<void>((resolve, reject) => {
        const handleInit = (e: MessageEvent) => {
          const { type } = e.data;
          if (type === 'SHARED_DATA_INITIALIZED') {
            item.worker.removeEventListener('message', handleInit);
            resolve();
          } else if (type === 'ERROR') {
            item.worker.removeEventListener('message', handleInit);
            reject(new Error(e.data.data.message));
          }
        };

        // Create separate buffers for each worker (required for transferable objects)
        const workerFreqBuffer = new Float64Array(frequencyBuffer);
        const workerSpectrumBuffer = new Float64Array(spectrumBuffer);

        item.worker.addEventListener('message', handleInit);
        
        // Use transferable objects for zero-copy transfer
        item.worker.postMessage({
          type: 'INITIALIZE_SHARED_DATA_TRANSFERABLE',
          data: {
            frequencyBuffer: workerFreqBuffer,
            spectrumBuffer: workerSpectrumBuffer,
            spectrumLength: referenceSpectrum.length,
            workerId: index
          }
        }, [workerFreqBuffer.buffer, workerSpectrumBuffer.buffer]); // Transfer ownership
      });
    });

    await Promise.all(initPromises);
    
    sharedDataRef.current = {
      frequencies,
      referenceSpectrum,
      isInitialized: true
    };

    onProgress({
      type: 'MATHEMATICAL_OPERATION',
      total: frequencies.length,
      overallProgress: 8,
      phase: 'initialization',
      message: `Shared data initialized in ${workerPoolRef.current.length} workers`,
      operation: 'Data optimization complete'
    });
  }, []);

  // Split grid points into chunks for parallel processing
  const chunkArray = useCallback(<T>(array: T[], chunkSize: number): T[][] => {
    const chunks: T[][] = [];
    for (let i = 0; i < array.length; i += chunkSize) {
      chunks.push(array.slice(i, i + chunkSize));
    }
    return chunks;
  }, []);

  // Generate frequencies array
  const generateFrequencies = useCallback((minFreq: number, maxFreq: number, numPoints: number): number[] => {
    const frequencies: number[] = [];
    const logMin = Math.log10(minFreq);
    const logMax = Math.log10(maxFreq);
    const logStep = (logMax - logMin) / (numPoints - 1);

    for (let i = 0; i < numPoints; i++) {
      const logValue = logMin + i * logStep;
      const frequency = Math.pow(10, logValue);
      frequencies.push(frequency);
    }

    return frequencies;
  }, []);

  // Calculate reference spectrum on main thread (single computation)
  const calculateReferenceSpectrum = useCallback((
    params: CircuitParameters, 
    frequencies: number[]
  ): Array<{ freq: number; real: number; imag: number; mag: number; phase: number }> => {
    const { Rsh, Ra, Ca, Rb, Cb } = params;
    
    return frequencies.map(freq => {
      const omega = 2 * Math.PI * freq;
      
      // Za = Ra/(1+jωRaCa)
      const za_denom = 1 + Math.pow(omega * Ra * Ca, 2);
      const za_real = Ra / za_denom;
      const za_imag = -omega * Ra * Ra * Ca / za_denom;
      
      // Zb = Rb/(1+jωRbCb)
      const zb_denom = 1 + Math.pow(omega * Rb * Cb, 2);
      const zb_real = Rb / zb_denom;
      const zb_imag = -omega * Rb * Rb * Cb / zb_denom;
      
      // Z_eq = Rsh + Za + Zb
      const real = Rsh + za_real + zb_real;
      const imag = za_imag + zb_imag;
      
      const magnitude = Math.sqrt(real * real + imag * imag);
      const phase = Math.atan2(imag, real) * (180 / Math.PI);
      
      return {
        freq,
        real,
        imag,
        mag: magnitude,
        phase
      };
    });
  }, []);

  // Main computation function
  const computeGridParallel = useCallback(async (
    groundTruthParams: CircuitParameters,
    gridSize: number,
    minFreq: number,
    maxFreq: number,
    numPoints: number,
    performanceSettings: PerformanceSettings,
    resnormConfig: ResnormConfig,
    onProgress: (progress: WorkerProgress) => void,
    onError: (error: string) => void
  ): Promise<BackendMeshPoint[]> => {
    
    if (isComputingRef.current) {
      throw new Error('Computation already in progress');
    }

    isComputingRef.current = true;
    cancelTokenRef.current = { cancelled: false };
    
    // Reset shared data state to ensure clean initialization
    sharedDataRef.current.isInitialized = false;

    try {
      // Determine total points first
      const totalPoints = Math.pow(gridSize, 5);
      
      // Generate frequency array
      onProgress({
        type: 'MATHEMATICAL_OPERATION',
        total: totalPoints,
        generated: 0,
        overallProgress: 2,
        phase: 'initialization',
        message: `Generating logarithmic frequency array (${numPoints} points)`,
        equation: 'f(i) = 10^(log₁₀(fₘᵢₙ) + i × (log₁₀(fₘₐₓ) - log₁₀(fₘᵢₙ))/(n-1))',
        operation: 'Frequency discretization'
      });
      const frequencies = generateFrequencies(minFreq, maxFreq, numPoints);
      
      // Calculate reference spectrum
      onProgress({
        type: 'MATHEMATICAL_OPERATION',
        total: totalPoints,
        generated: 0,
        overallProgress: 3,
        phase: 'initialization',
        message: 'Computing reference impedance spectrum for ground truth parameters',
        equation: 'Z_ref(ω) = Rs + Ra/(1+jωRaCa) + Rb/(1+jωRbCb)',
        operation: 'Reference spectrum calculation'
      });
      const referenceSpectrum = calculateReferenceSpectrum(groundTruthParams, frequencies);
      
      // Determine worker count and chunk size based on grid size
      const workerCount = getOptimalWorkerCount(totalPoints);
      
      onProgress({
        type: 'WORKER_STATUS',
        total: totalPoints,
        generated: 0,
        overallProgress: 4,
        phase: 'initialization',
        message: `Optimizing parallel computation: ${workerCount} workers for ${totalPoints.toLocaleString()} parameter combinations`,
        operation: 'Worker optimization',
        workerCount
      });
      
      // Anti-stall chunk sizing strategy based on insights from AskJS discussion
      let chunkSize: number;
      
      // Ultra-conservative sizing to prevent halfway stalls
      if (totalPoints > 500000) {
        chunkSize = 500; // Micro-chunks for massive datasets
        console.log('🔴 Using micro-chunks (500) for massive dataset to prevent stalls');
      } else if (totalPoints > 200000) {
        chunkSize = 800; // Very small chunks to prevent stalls
        console.log('🟠 Using very small chunks (800) for very large dataset');
      } else if (totalPoints > 100000) {
        chunkSize = 1200; // Small chunks for large datasets
        console.log('🟡 Using small chunks (1200) for large dataset');
      } else if (totalPoints > 50000) {
        chunkSize = 1800; // Medium chunks
        console.log('🟢 Using medium chunks (1800) for medium dataset');
      } else if (totalPoints > 25000) {
        chunkSize = 2500; // Larger chunks for smaller datasets
      } else {
        chunkSize = Math.ceil(totalPoints / workerCount); // Dynamic for very small datasets
      }
      
      // Safety cap based on worker count to prevent memory overload
      const maxChunkSize = Math.floor(15000 / workerCount); // Very conservative limit
      chunkSize = Math.min(chunkSize, maxChunkSize);
      
      console.log(`Final chunk size: ${chunkSize}, Total chunks: ${Math.ceil(totalPoints / chunkSize)}`);
      
      // Additional memory pressure prevention
      if (totalPoints > 100000 && chunkSize > 1000) {
        chunkSize = 1000;
        console.log('🚨 Force-limited chunk size to 1000 for large dataset anti-stall protection');
      }

      onProgress({
        type: 'COMPUTATION_START',
        total: totalPoints,
        generated: 0,
        overallProgress: 0,
        phase: 'initialization',
        message: 'Initializing computation environment',
        workerCount,
        chunkSize
      });

      onProgress({
        type: 'MATHEMATICAL_OPERATION',
        total: totalPoints,
        generated: 0,
        overallProgress: 1,
        phase: 'initialization',
        message: 'Setting up Randles equivalent circuit model',
        equation: 'Z(ω) = Rs + Ra/(1+jωRaCa) + Rb/(1+jωRbCb)',
        operation: 'Circuit model initialization'
      });

      // For large grids (>100k points), use streaming approach
      if (totalPoints > 100000) {
        return await computeGridStreaming(
          gridSize,
          frequencies,
          referenceSpectrum,
          minFreq,
          maxFreq,
          workerCount,
          chunkSize,
          performanceSettings,
          resnormConfig,
          onProgress,
          onError,
          groundTruthParams
        );
      }

      // Initialize worker pool for enhanced performance
      initializeWorkerPool(workerCount);
      
      // Initialize shared data once to minimize repeated transfers
      await initializeSharedData(frequencies, referenceSpectrum, onProgress);
      
      // Generate grid points using a single worker first
      const gridWorker = workerPoolRef.current[0].worker;

      const gridPoints = await new Promise<CircuitParameters[]>((resolve, reject) => {
        let generationProgress = 0;

        gridWorker.onmessage = (e) => {
          const { type, data } = e.data;
          
          if (cancelTokenRef.current.cancelled) {
            reject(new Error('Computation cancelled'));
            return;
          }

          switch (type) {
            case 'GENERATION_PROGRESS':
              generationProgress = (data.generated / data.total) * 10; // 10% for generation
              const percentComplete = Math.round((data.generated / data.total) * 100);
              onProgress({
                type: 'GENERATION_PROGRESS',
                total: data.total,
                generated: data.generated,
                skipped: data.skipped,
                overallProgress: generationProgress,
                phase: 'grid_generation',
                message: `Generating parameter grid: ${data.generated.toLocaleString()}/${data.total.toLocaleString()} combinations (${percentComplete}%)`,
                operation: 'Parameter space sampling'
              });
              break;
              
            case 'GRID_POINTS_GENERATED':
              resolve(data.gridPoints);
              break;
              
            case 'ERROR':
              reject(new Error(data.message));
              break;
          }
        };

        gridWorker.onerror = (error) => {
          reject(new Error(`Worker error: ${error.message}`));
        };

        gridWorker.postMessage({
          type: 'GENERATE_GRID_POINTS',
          data: { 
            gridSize,
            useSymmetricGrid: performanceSettings.useSymmetricGrid,
            resnormConfig,
            groundTruthParams
          }
        });
      });

      if (cancelTokenRef.current.cancelled) {
        throw new Error('Computation cancelled');
      }

      // Worker pool already initialized above
      
      // Split grid points into chunks
      const chunks = chunkArray(gridPoints, chunkSize);
      
      onProgress({
        type: 'MATHEMATICAL_OPERATION',
        total: totalPoints,
        processed: 0,
        overallProgress: 10,
        phase: 'impedance_calculation',
        message: `Starting parallel impedance calculations across ${chunks.length} chunks`,
        equation: 'For each parameter set: Z(ω) = Rs + Za(ω) + Zb(ω)',
        operation: 'Parallel impedance computation',
        chunkSize: chunks.length
      });

      // Process chunks with streaming results and anti-stall protection
      const processChunksWithStreaming = async () => {
        const streamedResults: BackendMeshPoint[] = [];
        
        // Ultra-conservative concurrency to prevent stalls
        let maxConcurrentChunks: number;
        if (totalPoints > 200000) {
          maxConcurrentChunks = 1; // Sequential processing for very large datasets
          console.log('🔴 Using sequential processing to prevent stalls');
        } else if (totalPoints > 100000) {
          maxConcurrentChunks = 2; // Minimal concurrency for large datasets
          console.log('🟡 Using minimal concurrency (2) for large dataset');
        } else {
          maxConcurrentChunks = Math.min(workerCount, 3); // Standard for smaller datasets
        }
        
        let totalProcessedCount = 0;
        
        // Process chunks in batches to maintain responsiveness
        for (let batchStart = 0; batchStart < chunks.length; batchStart += maxConcurrentChunks) {
          if (cancelTokenRef.current.cancelled) {
            throw new Error('Computation cancelled');
          }
          
          const batchEnd = Math.min(batchStart + maxConcurrentChunks, chunks.length);
          const batchChunks = chunks.slice(batchStart, batchEnd);
          
          // Process current batch with optimized worker pool
          const batchPromises = batchChunks.map((chunk, localIndex) => {
            const chunkIndex = batchStart + localIndex;
            const taskId = `task-${chunkIndex}-${Date.now()}`;
            
            // Get an idle worker from the pool
            const workerItem = getIdleWorker();
            if (!workerItem) {
              throw new Error('No idle workers available');
            }
            
            workerItem.taskId = taskId;
            const worker = workerItem.worker;

            return new Promise<WorkerResult>((resolve, reject) => {
              // Add this promise's reject function to active promises for cancellation
              const cleanup = () => {
                worker.removeEventListener('message', handleMessage);
                returnWorkerToPool(taskId);
                activePromisesRef.current.delete(cleanup);
                reject(new Error('Computation cancelled'));
              };
              activePromisesRef.current.add(cleanup);
              
              // Set worker timeout to prevent indefinite hanging
              const timeoutMs = totalPoints > 100000 ? 300000 : 180000; // 5min for large, 3min for smaller
              const timeout = setTimeout(() => {
                console.warn(`Worker timeout for task ${taskId} after ${timeoutMs}ms`);
                worker.removeEventListener('message', handleMessage);
                returnWorkerToPool(taskId);
                activePromisesRef.current.delete(cleanup);
                reject(new Error(`Worker timeout after ${timeoutMs}ms`));
              }, timeoutMs);
              
              workerTimeoutsRef.current.set(taskId, timeout);

              const handleMessage = (e: MessageEvent) => {
                const { type, data } = e.data;

                if (cancelTokenRef.current.cancelled) {
                  cleanup();
                  return;
                }

            switch (type) {
                  case 'CHUNK_PROGRESS':
                    const progressPercent = 10 + (totalProcessedCount / totalPoints) * 90;
                    const chunkPercent = Math.round((data.chunkProgress || 0) * 100);
                    
                    // More aggressive throttling for large datasets  
                    const shouldUpdate = totalPoints > 100000 ? 
                      (chunkPercent % 20 === 0 || data.chunkProgress === 1) :
                      (chunkPercent % 10 === 0 || data.chunkProgress === 1);
                      
                    if (shouldUpdate) {
                      onProgress({
                        type: 'STREAMING_UPDATE',
                        chunkIndex: data.chunkIndex,
                        totalChunks: chunks.length,
                        chunkProgress: data.chunkProgress,
                        processed: totalProcessedCount,
                        total: totalPoints,
                        overallProgress: progressPercent,
                        phase: 'impedance_calculation',
                        message: `Streaming batch ${Math.floor(batchStart / maxConcurrentChunks) + 1}: Processing chunk ${chunkIndex + 1}/${chunks.length} (${streamedResults.length} models computed)`,
                        operation: 'Streaming computation',
                        streamingBatch: Math.floor(batchStart / maxConcurrentChunks) + 1,
                        mainThreadLoad: totalPoints > 100000 ? 'high' : totalPoints > 50000 ? 'medium' : 'low'
                      });
                    }
                    break;
                
                  case 'CHUNK_COMPLETE':
                    worker.removeEventListener('message', handleMessage);
                    returnWorkerToPool(taskId);
                    activePromisesRef.current.delete(cleanup);
                    clearTimeout(timeout);
                    
                    // Immediately convert and stream results to prevent memory accumulation
                    const chunkResults: BackendMeshPoint[] = [
                      ...data.topResults.map((r: { 
                        parameters: CircuitParameters; 
                        spectrum: Array<{ freq: number; real: number; imag: number; mag: number; phase: number }>; 
                        resnorm: number 
                      }) => ({
                        parameters: {
                          Rsh: r.parameters.Rsh,
                          Ra: r.parameters.Ra,
                          Ca: r.parameters.Ca,
                          Rb: r.parameters.Rb,
                          Cb: r.parameters.Cb,
                          frequency_range: [minFreq, maxFreq] as [number, number]
                        },
                        spectrum: r.spectrum,
                        resnorm: r.resnorm
                      })),
                      ...data.otherResults.map((r: { parameters: CircuitParameters; resnorm: number }) => ({
                        parameters: {
                          Rsh: r.parameters.Rsh,
                          Ra: r.parameters.Ra,
                          Ca: r.parameters.Ca,
                          Rb: r.parameters.Rb,
                          Cb: r.parameters.Cb,
                          frequency_range: [minFreq, maxFreq] as [number, number]
                        },
                        spectrum: [], // Empty spectrum for memory efficiency
                        resnorm: r.resnorm
                      }))
                    ];
                    
                    streamedResults.push(...chunkResults);
                    totalProcessedCount += data.totalProcessed;
                    
                    console.log(`Streamed ${chunkResults.length} results, total: ${streamedResults.length}`);
                    resolve(data); // Return original worker result for compatibility
                    break;
                
                  case 'MEMORY_PRESSURE':
                    // Handle memory pressure with streaming approach
                    console.warn(`Worker memory pressure detected:`, data.message, data.estimatedMemory);
                    const safeProgress = Math.min(95, 10 + (totalProcessedCount / totalPoints) * 90);
                    
                    onProgress({
                      type: 'THROTTLE_UPDATE',
                      chunkIndex: data.chunkIndex,
                      totalChunks: chunks.length,
                      processed: totalProcessedCount,
                      total: totalPoints,
                      overallProgress: safeProgress,
                      memoryPressure: true,
                      message: `Memory pressure detected - ${streamedResults.length} models streamed so far`,
                      throttleDelay: 250,
                      mainThreadLoad: 'high'
                    });
                    
                    // Add adaptive delay based on memory pressure (handled in worker)
                    break;
                
                  case 'PARTIAL_RESULTS':
                    // Handle streaming results with memory-conscious buffering
                    console.log(`Worker streaming ${data.totalPartialCount} partial results`);
                    
                    // For large datasets, yield control is handled in the worker thread
                    break;
                
                  case 'ERROR':
                    worker.removeEventListener('message', handleMessage);
                    returnWorkerToPool(taskId);
                    activePromisesRef.current.delete(cleanup);
                    clearTimeout(timeout);
                    reject(new Error(data.message));
                    break;
            }
          };

          worker.addEventListener('message', handleMessage);
          
          worker.onerror = (error) => {
            worker.removeEventListener('message', handleMessage);
            returnWorkerToPool(taskId);
            activePromisesRef.current.delete(cleanup);
            clearTimeout(timeout);
            reject(new Error(`Worker error: ${error.message}`));
          };

              // Send only minimal data - frequencies and referenceSpectrum already initialized
              worker.postMessage({
                type: 'COMPUTE_GRID_CHUNK_OPTIMIZED',
                data: {
                  chunkParams: chunk,
                  chunkIndex,
                  totalChunks: chunks.length,
                  resnormConfig,
                  taskId
                }
              });
            });
          });
          
          // Wait for current batch to complete with streaming results
          await Promise.all(batchPromises);
          
          // Yield control to main thread between batches for better responsiveness
          if (batchEnd < chunks.length) {
            // Force garbage collection hint
            if (globalThis.gc) globalThis.gc();
            
            await new Promise(resolve => setTimeout(resolve, 20)); // Longer delay for large datasets
            
            // Report batch completion with memory status
            onProgress({
              type: 'STREAMING_UPDATE',
              processed: totalProcessedCount,
              total: totalPoints,
              overallProgress: 10 + (totalProcessedCount / totalPoints) * 90,
              phase: 'impedance_calculation',
              message: `Batch ${Math.floor(batchStart / maxConcurrentChunks) + 1}/${Math.ceil(chunks.length / maxConcurrentChunks)} complete - ${streamedResults.length} models streamed`,
              operation: 'Streaming batch processing',
              streamingBatch: Math.floor(batchStart / maxConcurrentChunks) + 1,
              mainThreadLoad: totalPoints > 100000 ? 'high' : 'medium'
            });
          }
        }
        
        console.log(`Streaming computation complete: ${streamedResults.length} total results`);
        return streamedResults;
      };
      
      // Execute the streaming processing
      const streamedResults = await processChunksWithStreaming();
      
      if (cancelTokenRef.current.cancelled) {
        throw new Error('Computation cancelled');
      }

      // Sort results by resnorm for consistent ordering
      streamedResults.sort((a, b) => a.resnorm - b.resnorm);
      
      // Send final completion update
      onProgress({
        type: 'CHUNK_PROGRESS',
        chunkIndex: chunks.length,
        totalChunks: chunks.length,
        processed: totalPoints,
        total: totalPoints,
        overallProgress: 100,
        phase: 'completion',
        message: `Streaming computation complete: ${streamedResults.length.toLocaleString()} models processed`,
        operation: 'Streaming results finalized'
      });

      return streamedResults;

    } catch (error) {
      onError(error instanceof Error ? error.message : String(error));
      throw error;
    } finally {
      isComputingRef.current = false;
    }
  }, [
    generateFrequencies,
    calculateReferenceSpectrum,
    getOptimalWorkerCount,
    chunkArray,
    initializeWorkerPool,
    initializeSharedData,
    getIdleWorker,
    returnWorkerToPool
  ]);  

  // New streaming computation function for large grids
  const computeGridStreaming = useCallback(async (
    gridSize: number,
    frequencies: number[],
    referenceSpectrum: { freq: number; real: number; imag: number; mag: number; phase: number; }[],
    minFreq: number,
    maxFreq: number,
    workerCount: number,
    chunkSize: number,
    performanceSettings: PerformanceSettings,
    resnormConfig: ResnormConfig,
    onProgress: (progress: WorkerProgress) => void,
    onError: (error: string) => void,
    groundTruthParams: CircuitParameters
  ): Promise<BackendMeshPoint[]> => {
    
    const totalPoints = Math.pow(gridSize, 5);
    
    // Initialize worker pool for streaming computation  
    initializeWorkerPool(workerCount);
    
    // Initialize shared data for streaming computation
    await initializeSharedData(frequencies, referenceSpectrum, onProgress);
    
    // Use a single worker for grid generation with streaming
    const gridWorker = workerPoolRef.current[0].worker;
    const computationWorkers = workerPoolRef.current.slice(1);
    
    const allResults: BackendMeshPoint[] = [];
    let totalProcessed = 0;
    
    // Start grid generation streaming
    const gridGenerationPromise = new Promise<void>((resolve, reject) => {
      const pendingChunks: CircuitParameters[][] = [];
      let isGenerationComplete = false;
      
      gridWorker.onmessage = (e) => {
        const { type, data } = e.data;
        
        if (cancelTokenRef.current.cancelled) {
          reject(new Error('Computation cancelled'));
          return;
        }
        
                 switch (type) {
           case 'GENERATION_PROGRESS':
             onProgress({
               type: 'GENERATION_PROGRESS',
               total: data.total,
               generated: data.generated,
               overallProgress: (data.generated / data.total) * 10
             });
             break;
             
           case 'GRID_CHUNK_READY':
             pendingChunks.push(data.chunk);
             
             if (data.progress.isComplete) {
               isGenerationComplete = true;
             }
             break;
            
          case 'ERROR':
            reject(new Error(data.message));
            break;
        }
      };
      
      gridWorker.onerror = (error) => {
        reject(new Error(`Grid worker error: ${error.message}`));
      };
      
      // Start streaming grid generation
      gridWorker.postMessage({
        type: 'GENERATE_GRID_STREAM',
        data: {
          gridSize,
          useSymmetricGrid: performanceSettings.useSymmetricGrid,
          resnormConfig,
          chunkSize: Math.min(chunkSize, 5000), // Smaller chunks for streaming
          groundTruthParams
        }
      });
      
      // Process chunks as they arrive
      const processChunks = async () => {
        while (!isGenerationComplete || pendingChunks.length > 0) {
          if (pendingChunks.length > 0) {
            const chunk = pendingChunks.shift()!;
            
            // Process this chunk with available workers
            const workerIndex = totalProcessed % computationWorkers.length;
            const worker = computationWorkers[workerIndex].worker;
            
            try {
              const result = await new Promise<WorkerResult>((resolve, reject) => {
                const handleMessage = (e: MessageEvent) => {
                  const { type, data } = e.data;
                  
                  if (cancelTokenRef.current.cancelled) {
                    worker.removeEventListener('message', handleMessage);
                    reject(new Error('Computation cancelled'));
                    return;
                  }
                  
                  switch (type) {
                    case 'CHUNK_COMPLETE':
                      worker.removeEventListener('message', handleMessage);
                      resolve(data);
                      break;
                      
                    case 'ERROR':
                      worker.removeEventListener('message', handleMessage);
                      reject(new Error(data.message));
                      break;
                  }
                };
                
                worker.addEventListener('message', handleMessage);
                
                worker.onerror = (error) => {
                  worker.removeEventListener('message', handleMessage);
                  reject(new Error(`Worker error: ${error.message}`));
                };
                
                worker.postMessage({
                  type: 'COMPUTE_GRID_CHUNK',
                  data: {
                    chunkParams: chunk,
                    frequencyArray: frequencies,
                    chunkIndex: totalProcessed,
                    totalChunks: -1, // Unknown for streaming
                    referenceSpectrum,
                    resnormConfig
                  }
                });
              });
              
              // Convert results to BackendMeshPoint format
              const chunkResults: BackendMeshPoint[] = [
                ...result.topResults.map(r => ({
                  parameters: {
                    Rsh: r.parameters.Rsh,
                    Ra: r.parameters.Ra,
                    Ca: r.parameters.Ca,
                    Rb: r.parameters.Rb,
                    Cb: r.parameters.Cb,
                    frequency_range: [minFreq, maxFreq] as [number, number]
                  },
                  spectrum: r.spectrum,
                  resnorm: r.resnorm
                })),
                ...result.otherResults.map(r => ({
                  parameters: {
                    Rsh: r.parameters.Rsh,
                    Ra: r.parameters.Ra,
                    Ca: r.parameters.Ca,
                    Rb: r.parameters.Rb,
                    Cb: r.parameters.Cb,
                    frequency_range: [minFreq, maxFreq] as [number, number]
                  },
                  spectrum: [], // Empty spectrum for memory efficiency
                  resnorm: r.resnorm
                }))
              ];
              
              allResults.push(...chunkResults);
              totalProcessed += result.totalProcessed;
              
              // Report progress
              const progressPercent = 10 + (totalProcessed / totalPoints) * 90;
              onProgress({
                type: 'CHUNK_PROGRESS',
                total: totalPoints,
                processed: totalProcessed,
                overallProgress: progressPercent
              });
              
                         } catch (error) {
               onError(error instanceof Error ? error.message : String(error));
               reject(error);
               return;
             }
          } else {
            // Wait a bit before checking again
            await new Promise(resolve => setTimeout(resolve, 10));
          }
        }
        
        resolve();
      };
      
      processChunks();
    });
    
    await gridGenerationPromise;
    
    if (cancelTokenRef.current.cancelled) {
      throw new Error('Computation cancelled');
    }
    
    onProgress({
      type: 'CHUNK_PROGRESS',
      total: totalPoints,
      processed: totalPoints,
      overallProgress: 100
    });
    
    return allResults;
  }, [initializeWorkerPool]);

  // Enhanced cancel computation with immediate force termination
  const cancelComputation = useCallback(() => {
    console.log('🛑 Cancel computation triggered - force terminating all workers');
    
    cancelTokenRef.current.cancelled = true;
    isComputingRef.current = false;
    
    // Force terminate all workers and clean up immediately
    forceTerminateAllWorkers();
    
    console.log('✅ Cancellation complete - all workers terminated');
  }, [forceTerminateAllWorkers]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      workerPoolRef.current.forEach(item => item.worker.terminate());
    };
  }, []);

  return {
    computeGridParallel,
    cancelComputation,
    isComputing: isComputingRef.current
  };
} 