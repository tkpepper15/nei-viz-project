import { useRef, useCallback, useEffect } from 'react';
import { BackendMeshPoint } from '../types';
import { CircuitParameters } from '../types/parameters';
import { PerformanceSettings } from '../controls/PerformanceControls';

export interface WorkerProgress {
  type: 'CHUNK_PROGRESS' | 'GENERATION_PROGRESS';
  chunkIndex?: number;
  totalChunks?: number;
  chunkProgress?: number;
  processed?: number;
  total: number;
  generated?: number;
  overallProgress: number;
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
    onProgress: (progress: WorkerProgress) => void,
    onError: (error: string) => void
  ) => Promise<BackendMeshPoint[]>;
  cancelComputation: () => void;
  isComputing: boolean;
}

export function useWorkerManager(): UseWorkerManagerReturn {
  const workersRef = useRef<Worker[]>([]);
  const isComputingRef = useRef(false);
  const cancelTokenRef = useRef<{ cancelled: boolean }>({ cancelled: false });

  // Determine optimal number of workers (usually CPU cores - 1)
  const getOptimalWorkerCount = useCallback(() => {
    const cores = navigator.hardwareConcurrency || 4;
    return Math.min(cores - 1, 8); // Cap at 8 workers to avoid too much overhead
  }, []);

  // Create workers
  const createWorkers = useCallback((count: number) => {
    // Clean up existing workers
    workersRef.current.forEach(worker => worker.terminate());
    workersRef.current = [];

    // Create new workers
    for (let i = 0; i < count; i++) {
      const worker = new Worker('/grid-worker.js');
      workersRef.current.push(worker);
    }
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
    const { Rs, Ra, Ca, Rb, Cb } = params;
    
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
      
      // Z_eq = Rs + Za + Zb
      const real = Rs + za_real + zb_real;
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
    onProgress: (progress: WorkerProgress) => void,
    onError: (error: string) => void
  ): Promise<BackendMeshPoint[]> => {
    
    if (isComputingRef.current) {
      throw new Error('Computation already in progress');
    }

    isComputingRef.current = true;
    cancelTokenRef.current = { cancelled: false };

    try {
      // Generate frequency array
      const frequencies = generateFrequencies(minFreq, maxFreq, numPoints);
      
      // Calculate reference spectrum
      const referenceSpectrum = calculateReferenceSpectrum(groundTruthParams, frequencies);
      
      // Determine worker count and chunk size based on grid size
      const totalPoints = Math.pow(gridSize, 5);
      const workerCount = getOptimalWorkerCount();
      
      // Adaptive chunk sizing based on total points
      let chunkSize: number;
      if (totalPoints <= 10000) {
        chunkSize = Math.ceil(totalPoints / workerCount);
      } else if (totalPoints <= 100000) {
        chunkSize = 5000;
      } else if (totalPoints <= 1000000) {
        chunkSize = 10000;
      } else {
        chunkSize = 25000; // Large chunks for massive grids
      }

      onProgress({
        type: 'GENERATION_PROGRESS',
        total: totalPoints,
        generated: 0,
        overallProgress: 0
      });

      // Generate grid points using a single worker first
      createWorkers(1);
      const gridWorker = workersRef.current[0];

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
              onProgress({
                type: 'GENERATION_PROGRESS',
                total: data.total,
                generated: data.generated,
                overallProgress: generationProgress
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
            useSymmetricGrid: performanceSettings.useSymmetricGrid 
          }
        });
      });

      if (cancelTokenRef.current.cancelled) {
        throw new Error('Computation cancelled');
      }

      // Now create multiple workers for parallel computation
      createWorkers(workerCount);
      
      // Split grid points into chunks
      const chunks = chunkArray(gridPoints, chunkSize);
      const results: WorkerResult[] = [];
      
      onProgress({
        type: 'CHUNK_PROGRESS',
        total: totalPoints,
        processed: 0,
        overallProgress: 10 // Start at 10% after generation
      });

      // Process chunks in parallel using all workers
      const chunkPromises = chunks.map((chunk, chunkIndex) => {
        const workerIndex = chunkIndex % workerCount;
        const worker = workersRef.current[workerIndex];

        return new Promise<WorkerResult>((resolve, reject) => {
          const handleMessage = (e: MessageEvent) => {
            const { type, data } = e.data;

            if (cancelTokenRef.current.cancelled) {
              worker.removeEventListener('message', handleMessage);
              reject(new Error('Computation cancelled'));
              return;
            }

            switch (type) {
              case 'CHUNK_PROGRESS':
                const overallProcessed = results.reduce((sum, r) => sum + r.totalProcessed, 0) + data.processed;
                const progressPercent = 10 + (overallProcessed / totalPoints) * 90; // 10-100%
                
                onProgress({
                  type: 'CHUNK_PROGRESS',
                  chunkIndex: data.chunkIndex,
                  totalChunks: chunks.length,
                  chunkProgress: data.chunkProgress,
                  processed: overallProcessed,
                  total: totalPoints,
                  overallProgress: progressPercent
                });
                break;
                
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
            reject(new Error(`Worker ${workerIndex} error: ${error.message}`));
          };

          worker.postMessage({
            type: 'COMPUTE_GRID_CHUNK',
            data: {
              chunkParams: chunk,
              frequencyArray: frequencies,
              chunkIndex,
              totalChunks: chunks.length,
              referenceSpectrum
            }
          });
        });
      });

      // Wait for all chunks to complete
      const chunkResults = await Promise.all(chunkPromises);
      
      if (cancelTokenRef.current.cancelled) {
        throw new Error('Computation cancelled');
      }

      // Combine results from all chunks
      const allTopResults: WorkerResult['topResults'] = [];
      const allOtherResults: WorkerResult['otherResults'] = [];

      for (const result of chunkResults) {
        allTopResults.push(...result.topResults);
        allOtherResults.push(...result.otherResults);
      }

      // Sort all results by resnorm and limit to reasonable size
      allTopResults.sort((a, b) => a.resnorm - b.resnorm);
      allOtherResults.sort((a, b) => a.resnorm - b.resnorm);

      // Keep top 5000 results with spectra, rest without spectra
      const topResults = allTopResults.slice(0, 5000);
      const otherResults = [
        ...allTopResults.slice(5000).map(r => ({ parameters: r.parameters, resnorm: r.resnorm })),
        ...allOtherResults.slice(0, 45000) // Total cap at 50k results
      ];

      // Convert to BackendMeshPoint format
      const finalResults: BackendMeshPoint[] = [
        ...topResults.map(r => ({
          parameters: {
            Rs: r.parameters.Rs,
            Ra: r.parameters.Ra,
            Ca: r.parameters.Ca,
            Rb: r.parameters.Rb,
            Cb: r.parameters.Cb,
            frequency_range: [minFreq, maxFreq] as [number, number]
          },
          spectrum: r.spectrum,
          resnorm: r.resnorm
        })),
        ...otherResults.map(r => ({
          parameters: {
            Rs: r.parameters.Rs,
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

      onProgress({
        type: 'CHUNK_PROGRESS',
        total: totalPoints,
        processed: totalPoints,
        overallProgress: 100
      });

      return finalResults;

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
    createWorkers
  ]);

  // Cancel computation
  const cancelComputation = useCallback(() => {
    cancelTokenRef.current.cancelled = true;
    isComputingRef.current = false;
    
    // Terminate all workers
    workersRef.current.forEach(worker => worker.terminate());
    workersRef.current = [];
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      workersRef.current.forEach(worker => worker.terminate());
    };
  }, []);

  return {
    computeGridParallel,
    cancelComputation,
    isComputing: isComputingRef.current
  };
} 