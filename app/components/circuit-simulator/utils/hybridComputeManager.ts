import { useRef, useCallback, useEffect } from 'react';
import { BackendMeshPoint } from '../types';
import { CircuitParameters } from '../types/parameters';
import { PerformanceSettings } from '../controls/PerformanceControls';
import { ResnormConfig } from './resnorm';
import { WorkerProgress, UseWorkerManagerReturn } from './workerManager';
import { WebGPUManager, WebGPUComputeResult } from './webgpuManager';
import { ExtendedPerformanceSettings, WebGPUCapabilities } from '../types/gpuSettings';

export interface HybridComputeResult {
  results: BackendMeshPoint[];
  usedGPU: boolean;
  benchmarkData?: {
    totalTime: number;
    computeTime: number;
    parametersProcessed: number;
    parametersPerSecond: number;
    memoryUsed: number;
  };
}

export interface UseHybridComputeManagerReturn {
  computeGridHybrid: (
    groundTruthParams: CircuitParameters,
    gridSize: number,
    minFreq: number,
    maxFreq: number,
    numPoints: number,
    performanceSettings: PerformanceSettings,
    extendedSettings: ExtendedPerformanceSettings,
    resnormConfig: ResnormConfig,
    onProgress: (progress: WorkerProgress) => void,
    onError: (error: string) => void,
    maxComputationResults?: number
  ) => Promise<HybridComputeResult>;
  cancelComputation: () => void;
  isComputing: boolean;
  getGPUCapabilities: () => Promise<WebGPUCapabilities>;
}

export function useHybridComputeManager(
  cpuComputeManager: UseWorkerManagerReturn
): UseHybridComputeManagerReturn {
  const webgpuManagerRef = useRef<WebGPUManager | null>(null);
  const isComputingRef = useRef(false);
  const cancelTokenRef = useRef<{ cancelled: boolean }>({ cancelled: false });

  // Initialize WebGPU manager
  useEffect(() => {
    webgpuManagerRef.current = new WebGPUManager();
    
    return () => {
      if (webgpuManagerRef.current) {
        webgpuManagerRef.current.dispose();
      }
    };
  }, []);

  const getGPUCapabilities = useCallback(async () => {
    if (!webgpuManagerRef.current) {
      webgpuManagerRef.current = new WebGPUManager();
    }
    return await webgpuManagerRef.current.initialize();
  }, []);

  const shouldUseGPU = useCallback((
    extendedSettings: ExtendedPerformanceSettings,
    totalParameters: number,
    capabilities: WebGPUCapabilities
  ): boolean => {
    // Check if GPU acceleration is enabled
    if (!extendedSettings.gpuAcceleration.enabled) {
      return false;
    }

    // Check if WebGPU is supported
    if (!capabilities?.supported) {
      console.log('📝 GPU disabled: WebGPU not supported');
      return false;
    }

    // For small datasets, CPU might be faster due to setup overhead
    const GPU_THRESHOLD = 5000; // Parameters below this use CPU
    if (totalParameters < GPU_THRESHOLD) {
      console.log(`📝 GPU disabled: Dataset too small (${totalParameters} < ${GPU_THRESHOLD})`);
      return false;
    }

    // Check if batch size is appropriate
    if (totalParameters > extendedSettings.gpuAcceleration.maxBatchSize * 2) {
      console.log(`📝 GPU enabled: Large dataset (${totalParameters} parameters)`);
      return true;
    }

    // For discrete GPUs, prefer GPU for medium datasets
    if (capabilities.deviceType === 'discrete' && totalParameters >= GPU_THRESHOLD) {
      console.log(`📝 GPU enabled: Discrete GPU with medium dataset`);
      return true;
    }

    // For integrated GPUs, be more conservative
    if (capabilities.deviceType === 'integrated' && totalParameters >= GPU_THRESHOLD * 2) {
      console.log(`📝 GPU enabled: Integrated GPU with larger dataset`);
      return true;
    }

    console.log(`📝 GPU disabled: Dataset not suitable for GPU acceleration`);
    return false;
  }, []);

  const generateGridPoints = useCallback(async (
    gridSize: number,
    performanceSettings: PerformanceSettings,
    resnormConfig: ResnormConfig,
    groundTruthParams: CircuitParameters
  ): Promise<CircuitParameters[]> => {
    // Use the CPU worker manager's grid generation
    // This is a placeholder - we'll need to extract the grid generation logic
    // from the CPU worker manager or create a shared utility
    
    // For now, generate a simple grid (this should match the CPU version)
    const gridPoints: CircuitParameters[] = [];
    
    // This is a simplified version - the actual implementation should match
    // the sophisticated grid generation in your existing worker
    const paramRanges = {
      Rsh: [100, 10000],
      Ra: [50, 5000], 
      Ca: [1e-8, 1e-4],
      Rb: [30, 3000],
      Cb: [1e-8, 1e-4]
    };

    const stepSize = gridSize > 1 ? 1 / (gridSize - 1) : 0;
    
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        for (let k = 0; k < gridSize; k++) {
          for (let l = 0; l < gridSize; l++) {
            for (let m = 0; m < gridSize; m++) {
              const params: CircuitParameters = {
                Rsh: paramRanges.Rsh[0] + (paramRanges.Rsh[1] - paramRanges.Rsh[0]) * i * stepSize,
                Ra: paramRanges.Ra[0] + (paramRanges.Ra[1] - paramRanges.Ra[0]) * j * stepSize,
                Ca: paramRanges.Ca[0] * Math.pow(paramRanges.Ca[1] / paramRanges.Ca[0], k * stepSize),
                Rb: paramRanges.Rb[0] + (paramRanges.Rb[1] - paramRanges.Rb[0]) * l * stepSize,
                Cb: paramRanges.Cb[0] * Math.pow(paramRanges.Cb[1] / paramRanges.Cb[0], m * stepSize),
                frequency_range: groundTruthParams.frequency_range
              };
              gridPoints.push(params);
            }
          }
        }
      }
    }
    
    return gridPoints;
  }, []);

  const convertGPUResultsToBackendMeshPoints = useCallback((
    gpuResults: WebGPUComputeResult,
    minFreq: number,
    maxFreq: number
  ): BackendMeshPoint[] => {
    return gpuResults.results.map(result => ({
      parameters: {
        ...result.parameters,
        frequency_range: [minFreq, maxFreq] as [number, number]
      },
      spectrum: result.spectrum,
      resnorm: result.resnorm
    }));
  }, []);

  const computeGridHybrid = useCallback(async (
    groundTruthParams: CircuitParameters,
    gridSize: number,
    minFreq: number,
    maxFreq: number,
    numPoints: number,
    performanceSettings: PerformanceSettings,
    extendedSettings: ExtendedPerformanceSettings,
    resnormConfig: ResnormConfig,
    onProgress: (progress: WorkerProgress) => void,
    onError: (error: string) => void,
    maxComputationResults: number = 5000
  ): Promise<HybridComputeResult> => {
    
    if (isComputingRef.current) {
      throw new Error('Computation already in progress');
    }

    isComputingRef.current = true;
    cancelTokenRef.current = { cancelled: false };

    const startTime = performance.now();
    const totalParameters = Math.pow(gridSize, 5);

    try {
      // Initialize WebGPU and get capabilities
      const capabilities = await getGPUCapabilities();
      
      // Decide whether to use GPU or CPU
      const useGPU = shouldUseGPU(extendedSettings, totalParameters, capabilities);

      if (useGPU && webgpuManagerRef.current) {
        console.log('🚀 Starting GPU-accelerated computation');
        
        onProgress({
          type: 'COMPUTATION_START',
          total: totalParameters,
          overallProgress: 0,
          phase: 'initialization',
          message: 'Initializing GPU acceleration',
          workerCount: 1 // GPU uses single compute pipeline
        });

        // Generate frequency array
        const frequencies = Array.from({ length: numPoints }, (_, i) => {
          const logMin = Math.log10(minFreq);
          const logMax = Math.log10(maxFreq);
          const logValue = logMin + (i / (numPoints - 1)) * (logMax - logMin);
          return Math.pow(10, logValue);
        });

        // Calculate reference spectrum (reuse CPU logic)
        const referenceSpectrum = frequencies.map(freq => {
          const omega = 2 * Math.PI * freq;
          const { Rsh, Ra, Ca, Rb, Cb } = groundTruthParams;
          
          // Za = Ra/(1+jωRaCa)  
          const za_denom = 1 + Math.pow(omega * Ra * Ca, 2);
          const za_real = Ra / za_denom;
          const za_imag = -omega * Ra * Ra * Ca / za_denom;
          
          // Zb = Rb/(1+jωRbCb)
          const zb_denom = 1 + Math.pow(omega * Rb * Cb, 2);
          const zb_real = Rb / zb_denom; 
          const zb_imag = -omega * Rb * Rb * Cb / zb_denom;
          
          // Z_total = Rsh + Za + Zb
          const real = Rsh + za_real + zb_real;
          const imag = za_imag + zb_imag;
          const magnitude = Math.sqrt(real * real + imag * imag);
          const phase = Math.atan2(imag, real) * (180 / Math.PI);
          
          return { freq, real, imag, mag: magnitude, phase };
        });

        // Generate parameter grid  
        onProgress({
          type: 'GENERATION_PROGRESS',
          total: totalParameters,
          generated: 0,
          overallProgress: 5,
          phase: 'grid_generation',
          message: 'Generating parameter combinations for GPU processing'
        });

        const gridPoints = await generateGridPoints(gridSize, performanceSettings, resnormConfig, groundTruthParams);

        onProgress({
          type: 'GENERATION_PROGRESS', 
          total: totalParameters,
          generated: gridPoints.length,
          overallProgress: 15,
          phase: 'grid_generation',
          message: `Generated ${gridPoints.length.toLocaleString()} parameter combinations`
        });

        // Process in batches if needed
        const batchSize = Math.min(gridPoints.length, extendedSettings.gpuAcceleration.maxBatchSize);
        const batches = [];
        
        for (let i = 0; i < gridPoints.length; i += batchSize) {
          batches.push(gridPoints.slice(i, i + batchSize));
        }

        console.log(`📊 GPU processing: ${batches.length} batches of up to ${batchSize} parameters each`);

        const allResults: BackendMeshPoint[] = [];
        let totalProcessed = 0;

        // Process each batch
        for (let batchIndex = 0; batchIndex < batches.length; batchIndex++) {
          if (cancelTokenRef.current.cancelled) {
            throw new Error('Computation cancelled');
          }

          const batch = batches[batchIndex];
          
          onProgress({
            type: 'CHUNK_PROGRESS',
            chunkIndex: batchIndex,
            totalChunks: batches.length,
            processed: totalProcessed,
            total: gridPoints.length,
            overallProgress: 15 + (totalProcessed / gridPoints.length) * 75,
            phase: 'impedance_calculation',
            message: `GPU batch ${batchIndex + 1}/${batches.length}: Processing ${batch.length} parameters`
          });

          // Run GPU computation
          const gpuResult = await webgpuManagerRef.current.computeCircuitGrid(
            batch,
            frequencies,
            referenceSpectrum,
            extendedSettings.gpuAcceleration
          );

          // Convert results
          const batchResults = convertGPUResultsToBackendMeshPoints(gpuResult, minFreq, maxFreq);
          allResults.push(...batchResults);
          totalProcessed += batch.length;

          onProgress({
            type: 'CHUNK_PROGRESS',
            chunkIndex: batchIndex + 1,
            totalChunks: batches.length,
            processed: totalProcessed,
            total: gridPoints.length,
            overallProgress: 15 + (totalProcessed / gridPoints.length) * 75,
            phase: 'impedance_calculation',
            message: `GPU batch complete: ${totalProcessed}/${gridPoints.length} parameters processed`
          });
        }

        // Sort by resnorm and limit results
        allResults.sort((a, b) => a.resnorm - b.resnorm);
        const finalResults = allResults.slice(0, maxComputationResults);

        const endTime = performance.now();
        const totalTime = endTime - startTime;

        onProgress({
          type: 'CHUNK_PROGRESS',
          processed: totalProcessed,
          total: gridPoints.length,
          overallProgress: 100,
          phase: 'completion',
          message: `GPU computation complete: ${finalResults.length} results in ${(totalTime/1000).toFixed(1)}s`
        });

        console.log(`✅ GPU computation complete: ${totalProcessed} parameters in ${totalTime.toFixed(2)}ms`);

        return {
          results: finalResults,
          usedGPU: true,
          benchmarkData: {
            totalTime,
            computeTime: totalTime * 0.8, // Rough estimate
            parametersProcessed: totalProcessed,
            parametersPerSecond: totalProcessed / (totalTime / 1000),
            memoryUsed: totalProcessed * 1000 // Rough estimate
          }
        };

      } else {
        // Fallback to CPU computation
        console.log('🔧 Falling back to CPU computation');
        
        if (extendedSettings.gpuAcceleration.enabled && !capabilities?.supported) {
          onProgress({
            type: 'COMPUTATION_START',
            total: totalParameters,
            overallProgress: 0,
            phase: 'initialization',
            message: 'GPU not available - using CPU workers',
            workerCount: extendedSettings.cpuSettings.maxWorkers
          });
        }

        const cpuResults = await cpuComputeManager.computeGridParallel(
          groundTruthParams,
          gridSize,
          minFreq,
          maxFreq,
          numPoints,
          performanceSettings,
          resnormConfig,
          onProgress,
          onError,
          maxComputationResults
        );

        const endTime = performance.now();

        return {
          results: cpuResults,
          usedGPU: false,
          benchmarkData: {
            totalTime: endTime - startTime,
            computeTime: (endTime - startTime) * 0.9,
            parametersProcessed: cpuResults.length,
            parametersPerSecond: cpuResults.length / ((endTime - startTime) / 1000),
            memoryUsed: cpuResults.length * 800 // Rough estimate
          }
        };
      }

    } catch (error) {
      // If GPU fails and fallback is enabled, try CPU
      if (extendedSettings.gpuAcceleration.enabled && 
          extendedSettings.gpuAcceleration.fallbackToCPU && 
          error instanceof Error && 
          error.message.includes('GPU')) {
        
        console.warn('🔄 GPU computation failed, falling back to CPU:', error.message);
        
        onProgress({
          type: 'COMPUTATION_START',
          total: totalParameters,
          overallProgress: 0,
          phase: 'initialization',
          message: 'GPU failed - falling back to CPU workers'
        });

        try {
          const cpuResults = await cpuComputeManager.computeGridParallel(
            groundTruthParams,
            gridSize,
            minFreq,
            maxFreq,
            numPoints,
            performanceSettings,
            resnormConfig,
            onProgress,
            onError,
            maxComputationResults
          );

          const endTime = performance.now();

          return {
            results: cpuResults,
            usedGPU: false,
            benchmarkData: {
              totalTime: endTime - startTime,
              computeTime: (endTime - startTime) * 0.9,
              parametersProcessed: cpuResults.length,
              parametersPerSecond: cpuResults.length / ((endTime - startTime) / 1000),
              memoryUsed: cpuResults.length * 800
            }
          };
        } catch (cpuError) {
          onError(cpuError instanceof Error ? cpuError.message : String(cpuError));
          throw cpuError;
        }
      }

      onError(error instanceof Error ? error.message : String(error));
      throw error;
    } finally {
      isComputingRef.current = false;
    }
  }, [cpuComputeManager, getGPUCapabilities, shouldUseGPU, generateGridPoints, convertGPUResultsToBackendMeshPoints]);

  const cancelComputation = useCallback(() => {
    console.log('🛑 Cancelling hybrid computation');
    cancelTokenRef.current.cancelled = true;
    isComputingRef.current = false;
    
    // Cancel CPU computation
    cpuComputeManager.cancelComputation();
    
    // GPU computation cancellation is handled by the cancel token check
  }, [cpuComputeManager]);

  return {
    computeGridHybrid,
    cancelComputation,
    isComputing: isComputingRef.current,
    getGPUCapabilities
  };
}