/**
 * Optimized Compute Manager
 * Integration layer between the new 3-stage pipeline and existing UI/workflow
 * Provides backward compatibility while enabling the new optimization features
 */

import { useCallback, useRef, useState } from 'react';
import { CircuitParameters } from '../types/parameters';
import { BackendMeshPoint } from '../types';
import { PerformanceSettings } from '../controls/PerformanceControls';
import { ExtendedPerformanceSettings } from '../types/gpuSettings';
import { ResnormConfig } from './resnorm';
import { WorkerProgress } from './workerManager';
import { 
  OptimizedComputePipeline, 
  OptimizedPipelineConfig, 
  DEFAULT_PIPELINE_CONFIG,
  PipelineProgress,
  PipelineResult 
} from './optimizedComputePipeline';
import { ParameterGeneratorConfig } from './streamingParameterIterator';

export interface OptimizedComputeResult {
  results: BackendMeshPoint[];
  usedOptimizedPipeline: boolean;
  benchmarkData: {
    totalTime: number;
    parametersProcessed: number;
    parametersPerSecond: number;
    memoryUsed: number;
    reductionRatio?: number;
    stageBreakdown?: {
      fingerprinting: number;
      coarseSSR: number;
      fullSSR: number;
    };
  };
  optimizationStats?: {
    uniqueFingerprints: number;
    stage1Representatives: number;
    stage2Survivors: number;
    finalCandidates: number;
  };
}

export interface OptimizedComputeManagerConfig {
  enableOptimizedPipeline: boolean;
  optimizationThreshold: number;    // Use optimization for grid sizes > this threshold
  fallbackToOriginal: boolean;      // Fallback to original computation on errors
  maxGridSizeForOptimization: number; // Don't optimize beyond this size (memory safety)
  
  // Pipeline configuration
  pipelineConfig: Partial<OptimizedPipelineConfig>;
}

const DEFAULT_MANAGER_CONFIG: OptimizedComputeManagerConfig = {
  enableOptimizedPipeline: true,
  optimizationThreshold: 10000,     // Use optimization for >10k parameter combinations
  fallbackToOriginal: true,
  maxGridSizeForOptimization: 1000000, // 1M parameters max for optimization
  
  pipelineConfig: {
    ...DEFAULT_PIPELINE_CONFIG,
    chunkSize: 15000,                // Slightly smaller chunks for better progress reporting
    finalTopK: 1000,                 // Return more results for visualization
    toleranceForTies: 0.01,          // Include more near-ties
  }
};

export interface UseOptimizedComputeManagerReturn {
  computeGridOptimized: (
    groundTruthParams: CircuitParameters,
    gridSize: number,
    minFreq: number,
    maxFreq: number,
    numPoints: number,
    performanceSettings: PerformanceSettings,
    extendedSettings: ExtendedPerformanceSettings,
    resnormConfig: ResnormConfig,
    onProgress: (progress: {
      phase: string;
      progress: number;
      currentOperation: string;
      parametersProcessed?: number;
      memoryUsage?: number;
      estimatedTimeRemaining?: number;
    }) => void,
    onError: (error: string) => void
  ) => Promise<OptimizedComputeResult>;
  
  isOptimizedEnabled: () => boolean;
  getOptimizationStats: () => {
    lastRun?: {
      timestamp: number;
      gridSize: number;
      totalParams: number;
      finalCandidates: number;
      reductionRatio: number;
      processingTime: number;
      stageBreakdown: {
        stage1: number;
        stage2: number;
        stage3: number;
      };
    };
  } | null;
  updateConfig: (config: Partial<OptimizedComputeManagerConfig>) => void;
}

/**
 * Hook for managing optimized compute pipeline
 */
export function useOptimizedComputeManager(
  originalComputeFunction: (
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
    maxResults?: number
  ) => Promise<{
    results: BackendMeshPoint[];
    benchmarkData?: {
      totalTime: number;
      computeTime: number;
      parametersProcessed: number;
      parametersPerSecond: number;
      memoryUsed: number;
    };
  }>, // Reference to existing compute function for fallback
  initialConfig: Partial<OptimizedComputeManagerConfig> = {}
): UseOptimizedComputeManagerReturn {
  
  const configRef = useRef<OptimizedComputeManagerConfig>({
    ...DEFAULT_MANAGER_CONFIG,
    ...initialConfig
  });
  
  const [optimizationStats, setOptimizationStats] = useState<{
    lastRun?: {
      timestamp: number;
      gridSize: number;
      totalParams: number;
      finalCandidates: number;
      reductionRatio: number;
      processingTime: number;
      stageBreakdown: {
        stage1: number;
        stage2: number;
        stage3: number;
      };
    };
  } | null>(null);
  const isComputingRef = useRef(false);
  
  /**
   * Determine if optimization should be used
   */
  const shouldUseOptimization = useCallback((
    gridSize: number,
    extendedSettings: ExtendedPerformanceSettings
  ): boolean => {
    const config = configRef.current;
    
    // Check if optimization is globally enabled
    if (!config.enableOptimizedPipeline) return false;
    
    // Check if user has explicitly disabled optimization
    if (extendedSettings.optimization?.enableAdvancedOptimization === false) return false;
    
    const totalParams = Math.pow(gridSize, 5);
    
    // Check threshold limits
    if (totalParams < config.optimizationThreshold) {
      console.log(`📊 Grid size ${gridSize}^5 = ${totalParams} below optimization threshold ${config.optimizationThreshold}, using original pipeline`);
      return false;
    }
    
    if (totalParams > config.maxGridSizeForOptimization) {
      console.log(`⚠️ Grid size ${gridSize}^5 = ${totalParams} exceeds optimization limit ${config.maxGridSizeForOptimization}, using original pipeline`);
      return false;
    }
    
    console.log(`🚀 Grid size ${gridSize}^5 = ${totalParams} qualified for optimized pipeline`);
    return true;
  }, []);
  
  /**
   * Generate reference spectrum from ground truth parameters
   */
  const generateReferenceSpectrum = useCallback((
    groundTruthParams: CircuitParameters,
    minFreq: number,
    maxFreq: number,
    numPoints: number
  ): Array<{ frequency: number; real: number; imag: number }> => {
    const frequencies: number[] = [];
    const logMin = Math.log10(minFreq);
    const logMax = Math.log10(maxFreq);
    
    // Generate log-spaced frequencies
    for (let i = 0; i < numPoints; i++) {
      const logValue = logMin + (i / (numPoints - 1)) * (logMax - logMin);
      frequencies.push(Math.pow(10, logValue));
    }
    
    // Calculate reference impedance spectrum
    return frequencies.map(frequency => {
      const omega = 2.0 * Math.PI * frequency;
      const tau_a = groundTruthParams.Ra * groundTruthParams.Ca;
      const tau_b = groundTruthParams.Rb * groundTruthParams.Cb;
      
      // Za = Ra / (1 + jωτa)
      const za_denom_real = 1.0;
      const za_denom_imag = omega * tau_a;
      const za_denom_mag_sq = za_denom_real * za_denom_real + za_denom_imag * za_denom_imag;
      const za_real = groundTruthParams.Ra * za_denom_real / za_denom_mag_sq;
      const za_imag = -groundTruthParams.Ra * za_denom_imag / za_denom_mag_sq;
      
      // Zb = Rb / (1 + jωτb)
      const zb_denom_real = 1.0;
      const zb_denom_imag = omega * tau_b;
      const zb_denom_mag_sq = zb_denom_real * zb_denom_real + zb_denom_imag * zb_denom_imag;
      const zb_real = groundTruthParams.Rb * zb_denom_real / zb_denom_mag_sq;
      const zb_imag = -groundTruthParams.Rb * zb_denom_imag / zb_denom_mag_sq;
      
      // Z_series = Za + Zb
      const z_series_real = za_real + zb_real;
      const z_series_imag = za_imag + zb_imag;
      
      // Z_total = (Rsh * Z_series) / (Rsh + Z_series)
      const num_real = groundTruthParams.Rsh * z_series_real;
      const num_imag = groundTruthParams.Rsh * z_series_imag;
      const denom_real = groundTruthParams.Rsh + z_series_real;
      const denom_imag = z_series_imag;
      const denom_mag_sq = denom_real * denom_real + denom_imag * denom_imag;
      
      return {
        frequency,
        real: (num_real * denom_real + num_imag * denom_imag) / denom_mag_sq,
        imag: (num_imag * denom_real - num_real * denom_imag) / denom_mag_sq
      };
    });
  }, []);
  
  /**
   * Main optimized compute function
   */
  const computeGridOptimized = useCallback(async (
    groundTruthParams: CircuitParameters,
    gridSize: number,
    minFreq: number,
    maxFreq: number,
    numPoints: number,
    performanceSettings: PerformanceSettings,
    extendedSettings: ExtendedPerformanceSettings,
    resnormConfig: ResnormConfig,
    onProgress: (progress: {
      phase: string;
      progress: number;
      currentOperation: string;
      parametersProcessed?: number;
      memoryUsage?: number;
      estimatedTimeRemaining?: number;
    }) => void,
    onError: (error: string) => void
  ): Promise<OptimizedComputeResult> => {
    
    if (isComputingRef.current) {
      throw new Error('Computation already in progress');
    }
    
    isComputingRef.current = true;
    const startTime = performance.now();
    
    try {
      const config = configRef.current;
      const useOptimization = shouldUseOptimization(gridSize, extendedSettings);
      
      if (!useOptimization) {
        // Fall back to original computation
        console.log('🔄 Using original computation pipeline...');
        
        // Create adapter for original function's WorkerProgress callback
        const originalProgressAdapter = (progress: WorkerProgress) => {
          onProgress({
            phase: `${progress.type}`,
            progress: progress.overallProgress,
            currentOperation: progress.type,
            parametersProcessed: progress.processed,
            memoryUsage: progress.memoryPressure ? 100 : 50, // Rough estimate
            estimatedTimeRemaining: undefined
          });
        };
        
        const originalResult = await originalComputeFunction(
          groundTruthParams,
          gridSize,
          minFreq,
          maxFreq,
          numPoints,
          performanceSettings,
          extendedSettings,
          resnormConfig,
          originalProgressAdapter,
          onError,
          5000 // maxResults
        );
        
        return {
          results: originalResult.results,
          usedOptimizedPipeline: false,
          benchmarkData: {
            totalTime: originalResult.benchmarkData?.totalTime || 0,
            parametersProcessed: originalResult.benchmarkData?.parametersProcessed || 0,
            parametersPerSecond: originalResult.benchmarkData?.parametersPerSecond || 0,
            memoryUsed: originalResult.benchmarkData?.memoryUsed || 0
          }
        };
      }
      
      // Use optimized pipeline
      console.log('🚀 Using optimized 3-stage pipeline...');
      
      // Generate reference spectrum
      const referenceSpectrum = generateReferenceSpectrum(
        groundTruthParams,
        minFreq,
        maxFreq,
        numPoints
      );
      
      // Configure parameter generator
      const parameterConfig: Partial<ParameterGeneratorConfig> = {
        gridSize,
        chunkSize: config.pipelineConfig.chunkSize,
        groundTruthParams,
        useSymmetricOptimization: extendedSettings.optimization?.useSymmetricOptimization !== false,
        samplingMode: 'logarithmic'
      };
      
      // Create and run optimized pipeline
      const pipeline = new OptimizedComputePipeline(
        referenceSpectrum,
        config.pipelineConfig,
        parameterConfig
      );
      
      // Convert progress callback
      const progressAdapter = (progress: PipelineProgress) => {
        onProgress({
          phase: `Stage ${progress.stage}`,
          progress: progress.overallProgress,
          currentOperation: progress.currentOperation,
          parametersProcessed: progress.parametersProcessed,
          memoryUsage: progress.memoryUsage,
          estimatedTimeRemaining: progress.estimatedTimeRemaining
        });
      };
      
      const result: PipelineResult = await pipeline.runPipeline(
        progressAdapter,
        (stage, stageResults) => {
          console.log(`✅ Stage ${stage} completed:`, stageResults);
        }
      );
      
      const totalTime = performance.now() - startTime;
      
      // Update optimization stats
      setOptimizationStats({
        lastRun: {
          timestamp: Date.now(),
          gridSize,
          totalParams: result.benchmarkData.parametersProcessed,
          finalCandidates: result.finalCandidates.length,
          reductionRatio: result.benchmarkData.reductionRatio,
          processingTime: totalTime,
          stageBreakdown: {
            stage1: result.benchmarkData.stage1Time,
            stage2: result.benchmarkData.stage2Time,
            stage3: result.benchmarkData.stage3Time
          }
        }
      });
      
      return {
        results: result.finalCandidates,
        usedOptimizedPipeline: true,
        benchmarkData: {
          totalTime,
          parametersProcessed: result.benchmarkData.parametersProcessed,
          parametersPerSecond: result.benchmarkData.parametersProcessed / (totalTime / 1000),
          memoryUsed: result.benchmarkData.memoryPeakUsage,
          reductionRatio: result.benchmarkData.reductionRatio,
          stageBreakdown: {
            fingerprinting: result.benchmarkData.stage1Time,
            coarseSSR: result.benchmarkData.stage2Time,
            fullSSR: result.benchmarkData.stage3Time
          }
        },
        optimizationStats: {
          uniqueFingerprints: result.benchmarkData.uniqueFingerprints,
          stage1Representatives: result.stageResults.stage1Representatives,
          stage2Survivors: result.stageResults.stage2Survivors,
          finalCandidates: result.stageResults.stage3Final
        }
      };
      
    } catch (error) {
      console.error('❌ Optimized pipeline failed:', error);
      
      if (configRef.current.fallbackToOriginal && originalComputeFunction) {
        console.log('🔄 Falling back to original pipeline...');
        onProgress({ 
          phase: 'Fallback',
          progress: 0,
          currentOperation: 'Switching to original computation method'
        });
        
        // Create adapter for original function's WorkerProgress callback
        const fallbackProgressAdapter = (progress: WorkerProgress) => {
          onProgress({
            phase: `Fallback: ${progress.type}`,
            progress: progress.overallProgress,
            currentOperation: progress.type,
            parametersProcessed: progress.processed,
            memoryUsage: progress.memoryPressure ? 100 : 50,
            estimatedTimeRemaining: undefined
          });
        };
        
        try {
          const fallbackResult = await originalComputeFunction(
            groundTruthParams,
            gridSize,
            minFreq,
            maxFreq,
            numPoints,
            performanceSettings,
            extendedSettings,
            resnormConfig,
            fallbackProgressAdapter,
            onError,
            5000 // maxResults
          );
          
          return {
            results: fallbackResult.results,
            usedOptimizedPipeline: false,
            benchmarkData: {
              totalTime: fallbackResult.benchmarkData?.totalTime || 0,
              parametersProcessed: fallbackResult.benchmarkData?.parametersProcessed || 0,
              parametersPerSecond: fallbackResult.benchmarkData?.parametersPerSecond || 0,
              memoryUsed: fallbackResult.benchmarkData?.memoryUsed || 0
            }
          };
        } catch (fallbackError) {
          onError(`Both optimized and fallback computations failed: ${error}`);
          throw fallbackError;
        }
      } else {
        onError(`Optimized computation failed: ${error}`);
        throw error;
      }
    } finally {
      isComputingRef.current = false;
    }
  }, [shouldUseOptimization, generateReferenceSpectrum, originalComputeFunction]);
  
  /**
   * Check if optimization is enabled
   */
  const isOptimizedEnabled = useCallback((): boolean => {
    return configRef.current.enableOptimizedPipeline;
  }, []);
  
  /**
   * Get current optimization statistics
   */
  const getOptimizationStats = useCallback(() => {
    return optimizationStats;
  }, [optimizationStats]);
  
  /**
   * Update configuration
   */
  const updateConfig = useCallback((config: Partial<OptimizedComputeManagerConfig>) => {
    configRef.current = {
      ...configRef.current,
      ...config
    };
  }, []);
  
  return {
    computeGridOptimized,
    isOptimizedEnabled,
    getOptimizationStats,
    updateConfig
  };
}