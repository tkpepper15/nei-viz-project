/**
 * Optimized 3-Stage Compute Pipeline
 * Implements the proposed coarse → fine screening approach
 * Stage 1: Fingerprinting + cheap resnorm
 * Stage 2: Coarse SSR on representatives  
 * Stage 3: Full SSR refinement + local optimization
 */

// Type for performance.memory API
interface PerformanceMemory {
  usedJSHeapSize: number;
  totalJSHeapSize: number;
  jsHeapSizeLimit: number;
}

declare global {
  interface Performance {
    memory?: PerformanceMemory;
  }
}

import { BackendMeshPoint } from '../types';
import { CircuitParameters } from '../types/parameters';
import { TopKHeap, HeapItem } from './topKHeap';
import { 
  SpectralFingerprintManager, 
  FingerprintConfig,
  generateLogFrequencies,
  computeFullSSR
} from './spectralFingerprinting';
import { 
  StreamingParameterGenerator,
  ParameterGeneratorConfig
} from './streamingParameterIterator';

export interface OptimizedPipelineConfig {
  // Stage parameters from the proposed plan
  fingerprintFrequencies: number;        // 8 (fingerprint)
  approximationFrequencies: number;      // 5 (cheap screening)
  coarseFrequencies: number;            // 12 (coarse SSR)
  fullFrequencies: number;              // 20 (full SSR, increase to 40 if needed)
  
  // Filtering parameters
  quantizationBin: number;              // 0.05 dex for fingerprint grouping
  chunkSize: number;                    // 20,000 (tune based on memory)
  topMSurvivors: number;               // 5,000 (coarse survivors)
  finalTopK: number;                   // 500 (final top-K)
  
  // Optimization parameters
  candidatesPerFingerprint: number;     // 3 (max candidates per spectral group)
  toleranceForTies: number;            // 0.005 (0.5% for near-ties)
  enableLocalOptimization: boolean;     // Optional Nelder-Mead refinement
  
  // Hardware optimization
  useWebWorkers: boolean;
  maxConcurrentChunks: number;
  enableMemoryOptimization: boolean;
}

export const DEFAULT_PIPELINE_CONFIG: OptimizedPipelineConfig = {
  fingerprintFrequencies: 8,
  approximationFrequencies: 5,
  coarseFrequencies: 12,
  fullFrequencies: 20,
  
  quantizationBin: 0.05,
  chunkSize: 20000,
  topMSurvivors: 5000,
  finalTopK: 500,
  
  candidatesPerFingerprint: 3,
  toleranceForTies: 0.005,
  enableLocalOptimization: false,
  
  useWebWorkers: true,
  maxConcurrentChunks: navigator.hardwareConcurrency || 4,
  enableMemoryOptimization: true
};

export interface PipelineResult {
  finalCandidates: BackendMeshPoint[];
  benchmarkData: {
    totalTime: number;
    stage1Time: number;      // Fingerprinting
    stage2Time: number;      // Coarse SSR
    stage3Time: number;      // Full SSR
    parametersProcessed: number;
    uniqueFingerprints: number;
    reductionRatio: number;  // Original params / final candidates
    memoryPeakUsage: number;
  };
  stageResults: {
    stage1Representatives: number;
    stage2Survivors: number;
    stage3Final: number;
  };
}

export interface PipelineProgress {
  stage: 1 | 2 | 3;
  stageProgress: number;     // 0-100
  overallProgress: number;   // 0-100
  currentOperation: string;
  parametersProcessed: number;
  memoryUsage?: number;
  estimatedTimeRemaining?: number;
}

export interface StageResults {
  stage1Representatives?: number;
  stage2Survivors?: number;
  stage3Final?: number;
  processingTime?: number;
  memoryUsed?: number;
  representatives?: number;
  uniqueFingerprints?: number;
  survivors?: number;
  finalCandidates?: number;
}

/**
 * High-Performance Optimized Compute Pipeline
 */
export class OptimizedComputePipeline {
  private config: OptimizedPipelineConfig;
  private fingerprinter: SpectralFingerprintManager;
  private parameterGenerator: StreamingParameterGenerator;
  private referenceSpectrum: Array<{ frequency: number; real: number; imag: number }>;
  
  // Pipeline state
  private isRunning = false;
  private currentStage: 1 | 2 | 3 = 1;
  private startTime = 0;
  private stageStartTime = 0;
  
  // Progress tracking
  private onProgress?: (progress: PipelineProgress) => void;
  private onStageComplete?: (stage: number, results: StageResults) => void;
  
  constructor(
    referenceSpectrum: Array<{ frequency: number; real: number; imag: number }>,
    pipelineConfig: Partial<OptimizedPipelineConfig> = {},
    parameterConfig: Partial<ParameterGeneratorConfig> = {}
  ) {
    this.config = { ...DEFAULT_PIPELINE_CONFIG, ...pipelineConfig };
    this.referenceSpectrum = referenceSpectrum;
    
    // Initialize fingerprint configuration
    const fingerprintConfig: FingerprintConfig = {
      fingerprintFrequencies: generateLogFrequencies(1, 10000, this.config.fingerprintFrequencies),
      approximationFrequencies: generateLogFrequencies(1, 10000, this.config.approximationFrequencies),
      quantizationBin: this.config.quantizationBin,
      candidatesPerFingerprint: this.config.candidatesPerFingerprint
    };
    
    this.fingerprinter = new SpectralFingerprintManager(fingerprintConfig, referenceSpectrum);
    
    // Initialize parameter generator
    const generatorConfig: ParameterGeneratorConfig = {
      gridSize: 25, // Default grid size, can be overridden
      chunkSize: this.config.chunkSize,
      useSymmetricOptimization: true,
      samplingMode: 'logarithmic',
      ...parameterConfig
    };
    
    this.parameterGenerator = new StreamingParameterGenerator(generatorConfig);
  }
  
  /**
   * Helper method to create complete CircuitParameters with frequency_range
   */
  private createCompleteParameters(baseParams: { Rsh: number; Ra: number; Ca: number; Rb: number; Cb: number }): CircuitParameters {
    return {
      ...baseParams,
      frequency_range: [
        this.referenceSpectrum[0].frequency,
        this.referenceSpectrum[this.referenceSpectrum.length - 1].frequency
      ]
    };
  }
  
  /**
   * Run the complete 3-stage pipeline
   */
  async runPipeline(
    progressCallback?: (progress: PipelineProgress) => void,
    stageCompleteCallback?: (stage: number, results: StageResults) => void
  ): Promise<PipelineResult> {
    if (this.isRunning) {
      throw new Error('Pipeline is already running');
    }
    
    this.isRunning = true;
    this.startTime = performance.now();
    this.onProgress = progressCallback;
    this.onStageComplete = stageCompleteCallback;
    
    let stage1Representatives: HeapItem[] = [];
    let stage2Survivors: HeapItem[] = [];
    let stage3Final: HeapItem[] = [];
    
    let memoryPeakUsage = 0;
    
    try {
      // Stage 1: Fingerprinting + Cheap Screening
      console.log('🔍 Stage 1: Spectral fingerprinting + cheap resnorm screening...');
      this.currentStage = 1;
      this.stageStartTime = performance.now();
      
      const stage1Result = await this.runStage1();
      stage1Representatives = stage1Result.representatives;
      memoryPeakUsage = Math.max(memoryPeakUsage, stage1Result.memoryUsage);
      
      const stage1Time = performance.now() - this.stageStartTime;
      console.log(`✅ Stage 1 complete: ${stage1Representatives.length} representatives from ${stage1Result.uniqueFingerprints} unique fingerprints (${stage1Time.toFixed(0)}ms)`);
      
      this.onStageComplete?.(1, { representatives: stage1Representatives.length, uniqueFingerprints: stage1Result.uniqueFingerprints });
      
      // Stage 2: Coarse SSR on Representatives
      if (stage1Representatives.length > 0) {
        console.log('📊 Stage 2: Coarse SSR screening on representatives...');
        this.currentStage = 2;
        this.stageStartTime = performance.now();
        
        const stage2Result = await this.runStage2(stage1Representatives);
        stage2Survivors = stage2Result.survivors;
        memoryPeakUsage = Math.max(memoryPeakUsage, stage2Result.memoryUsage);
        
        const stage2Time = performance.now() - this.stageStartTime;
        console.log(`✅ Stage 2 complete: ${stage2Survivors.length} survivors from ${stage1Representatives.length} representatives (${stage2Time.toFixed(0)}ms)`);
        
        this.onStageComplete?.(2, { survivors: stage2Survivors.length });
      }
      
      // Stage 3: Full SSR Refinement
      if (stage2Survivors.length > 0) {
        console.log('🎯 Stage 3: Full SSR refinement + final selection...');
        this.currentStage = 3;
        this.stageStartTime = performance.now();
        
        const stage3Result = await this.runStage3(stage2Survivors);
        stage3Final = stage3Result.finalCandidates;
        memoryPeakUsage = Math.max(memoryPeakUsage, stage3Result.memoryUsage);
        
        const stage3Time = performance.now() - this.stageStartTime;
        console.log(`✅ Stage 3 complete: ${stage3Final.length} final candidates (${stage3Time.toFixed(0)}ms)`);
        
        this.onStageComplete?.(3, { finalCandidates: stage3Final.length });
      }
      
      const totalTime = performance.now() - this.startTime;
      const totalParams = this.parameterGenerator.getTotalCombinations();
      
      // Convert HeapItems to BackendMeshPoints with spectrum data
      const finalResults: BackendMeshPoint[] = stage3Final.map((item, index) => {
        // Create complete CircuitParameters with frequency_range first
        const completeParameters = this.createCompleteParameters(item.parameters);
        
        // Generate spectrum for each final candidate using complete parameters
        const spectrum = this.generateSpectrum(completeParameters);
        
        return {
          id: index,
          parameters: completeParameters,
          resnorm: item.resnorm,
          spectrum
        };
      });
      
      const result: PipelineResult = {
        finalCandidates: finalResults,
        benchmarkData: {
          totalTime,
          stage1Time: stage1Result.processingTime,
          stage2Time: stage2Survivors.length > 0 ? performance.now() - this.stageStartTime : 0,
          stage3Time: stage3Final.length > 0 ? performance.now() - this.stageStartTime : 0,
          parametersProcessed: totalParams,
          uniqueFingerprints: stage1Result.uniqueFingerprints,
          reductionRatio: totalParams / Math.max(finalResults.length, 1),
          memoryPeakUsage
        },
        stageResults: {
          stage1Representatives: stage1Representatives.length,
          stage2Survivors: stage2Survivors.length,
          stage3Final: stage3Final.length
        }
      };
      
      console.log(`🎉 Pipeline complete: ${totalParams.toLocaleString()} → ${finalResults.length} (${(result.benchmarkData.reductionRatio).toFixed(0)}x reduction) in ${totalTime.toFixed(0)}ms`);
      
      return result;
      
    } finally {
      this.isRunning = false;
    }
  }
  
  /**
   * Stage 1: Fingerprinting + Cheap Resnorm Screening
   */
  private async runStage1(): Promise<{
    representatives: HeapItem[];
    uniqueFingerprints: number;
    memoryUsage: number;
    processingTime: number;
  }> {
    const startTime = performance.now();
    let processedChunks = 0;
    let totalProcessed = 0;
    const allRepresentatives: HeapItem[] = [];
    let uniqueFingerprints = 0;
    const totalChunks = Math.ceil(this.parameterGenerator.getTotalCombinations() / this.config.chunkSize);
    
    // Process parameter chunks through fingerprinting
    for (const chunk of this.parameterGenerator.generateChunks()) {
      const chunkResult = this.fingerprinter.processChunk(chunk.parameters);
      totalProcessed += chunkResult.processed;
      uniqueFingerprints = chunkResult.uniqueFingerprints; // Updated with latest count
      
      // Collect representatives from this chunk
      allRepresentatives.push(...chunkResult.representatives);
      processedChunks++;
      
      // Report progress
      const stageProgress = (processedChunks / totalChunks) * 100;
      this.reportProgress(stageProgress, `Processing chunk ${processedChunks}/${totalChunks}`, totalProcessed);
      
      // Memory management: force garbage collection periodically
      if (this.config.enableMemoryOptimization && processedChunks % 10 === 0) {
        if (globalThis.gc) {
          globalThis.gc();
        }
      }
    }
    
    // Sort representatives by resnorm and take the top ones
    const representatives = allRepresentatives
      .sort((a, b) => a.resnorm - b.resnorm)
      .slice(0, this.config.topMSurvivors);
    
    const processingTime = performance.now() - startTime;
    const memoryUsage = this.estimateMemoryUsage();
    
    return {
      representatives,
      uniqueFingerprints,
      memoryUsage,
      processingTime
    };
  }
  
  /**
   * Stage 2: Coarse SSR Screening on Representatives
   */
  private async runStage2(representatives: HeapItem[]): Promise<{
    survivors: HeapItem[];
    memoryUsage: number;
  }> {
    const coarseFrequencies = generateLogFrequencies(1, 10000, this.config.coarseFrequencies);
    const topMHeap = new TopKHeap(this.config.topMSurvivors);
    
    let processed = 0;
    
    // Process representatives in chunks for memory efficiency
    for (let i = 0; i < representatives.length; i += this.config.chunkSize) {
      const chunk = representatives.slice(i, Math.min(i + this.config.chunkSize, representatives.length));
      
      for (const item of chunk) {
        try {
          // Compute coarse SSR using reduced frequency set
          const coarseSpectrum = Array.from(coarseFrequencies).map(freq => {
            // Find closest reference frequency
            const closest = this.findClosestReferencePoint(freq);
            return closest;
          });
          
          const completeParams = this.createCompleteParameters(item.parameters);
          const coarseSSR = computeFullSSR(completeParams, coarseSpectrum);
          
          const refinedItem: HeapItem = {
            ...item,
            resnorm: coarseSSR,
            metadata: {
              ...item.metadata,
              stage: 2
            }
          };
          
          topMHeap.push(refinedItem);
          processed++;
          
        } catch (error) {
          console.warn('Error in coarse SSR for parameter set:', item.parameters, error);
        }
      }
      
      // Report progress
      const progress = (processed / representatives.length) * 100;
      this.reportProgress(progress, `Coarse SSR: ${processed}/${representatives.length}`, processed);
    }
    
    return {
      survivors: topMHeap.getSortedItems(),
      memoryUsage: this.estimateMemoryUsage()
    };
  }
  
  /**
   * Stage 3: Full SSR Refinement + Final Selection
   */
  private async runStage3(survivors: HeapItem[]): Promise<{
    finalCandidates: HeapItem[];
    memoryUsage: number;
  }> {
    const finalHeap = new TopKHeap(this.config.finalTopK);
    let processed = 0;
    
    // Process survivors for full SSR computation
    for (const item of survivors) {
      try {
        // Compute full complex SSR on complete reference spectrum
        const completeParams = this.createCompleteParameters(item.parameters);
        const fullSSR = computeFullSSR(completeParams, this.referenceSpectrum);
        
        const finalItem: HeapItem = {
          ...item,
          resnorm: fullSSR,
          metadata: {
            ...item.metadata,
            stage: 3
          }
        };
        
        finalHeap.push(finalItem);
        processed++;
        
      } catch (error) {
        console.warn('Error in full SSR for parameter set:', item.parameters, error);
      }
      
      // Report progress
      const progress = (processed / survivors.length) * 100;
      this.reportProgress(progress, `Full SSR: ${processed}/${survivors.length}`, processed);
    }
    
    let finalCandidates = finalHeap.getSortedItems();
    
    // Include near-ties within tolerance
    if (finalCandidates.length > 0) {
      const bestResnorm = finalCandidates[0].resnorm;
      const toleranceThreshold = bestResnorm * (1 + this.config.toleranceForTies);
      
      // Find all candidates within tolerance from the full survivor list
      const nearTies = survivors.filter(item => {
        const completeParams = this.createCompleteParameters(item.parameters);
        const fullSSR = computeFullSSR(completeParams, this.referenceSpectrum);
        return fullSSR <= toleranceThreshold && !finalCandidates.some(final => final.parameters === item.parameters);
      });
      
      // Add near-ties to final results
      finalCandidates = [...finalCandidates, ...nearTies.slice(0, Math.max(0, this.config.finalTopK - finalCandidates.length))];
    }
    
    return {
      finalCandidates,
      memoryUsage: this.estimateMemoryUsage()
    };
  }
  
  /**
   * Report progress to callback
   */
  private reportProgress(stageProgress: number, operation: string, parametersProcessed: number): void {
    if (!this.onProgress) return;
    
    const stageWeights = [0.6, 0.3, 0.1]; // Stage 1 is most expensive
    const overallProgress = stageWeights.slice(0, this.currentStage - 1).reduce((sum, w) => sum + w * 100, 0) + 
                           stageWeights[this.currentStage - 1] * stageProgress;
    
    this.onProgress({
      stage: this.currentStage,
      stageProgress,
      overallProgress,
      currentOperation: operation,
      parametersProcessed,
      memoryUsage: this.estimateMemoryUsage(),
      estimatedTimeRemaining: this.estimateTimeRemaining(overallProgress)
    });
  }
  
  /**
   * Find closest reference spectrum point to a target frequency
   */
  private findClosestReferencePoint(targetFrequency: number): { frequency: number; real: number; imag: number } {
    let closest = this.referenceSpectrum[0];
    let minDiff = Math.abs(this.referenceSpectrum[0].frequency - targetFrequency);
    
    for (const point of this.referenceSpectrum) {
      const diff = Math.abs(point.frequency - targetFrequency);
      if (diff < minDiff) {
        minDiff = diff;
        closest = point;
      }
    }
    
    return closest;
  }
  
  /**
   * Estimate current memory usage
   */
  private estimateMemoryUsage(): number {
    // Rough estimation based on active data structures
    // Use proper memory API with fallback
    if ('memory' in performance && performance.memory) {
      return performance.memory.usedJSHeapSize || 0;
    }
    return 0;
  }
  
  /**
   * Estimate time remaining based on current progress
   */
  private estimateTimeRemaining(overallProgress: number): number {
    if (overallProgress <= 0) return 0;
    
    const elapsedTime = performance.now() - this.startTime;
    const totalEstimatedTime = elapsedTime / (overallProgress / 100);
    return Math.max(0, totalEstimatedTime - elapsedTime);
  }
  
  /**
   * Generate spectrum data for a parameter set
   */
  private generateSpectrum(params: CircuitParameters): Array<{ freq: number; real: number; imag: number; mag: number; phase: number }> {
    return this.referenceSpectrum.map(refPoint => {
      const omega = 2.0 * Math.PI * refPoint.frequency;
      const tau_a = params.Ra * params.Ca;
      const tau_b = params.Rb * params.Cb;
      
      // Za = Ra / (1 + jωτa)
      const za_denom_real = 1.0;
      const za_denom_imag = omega * tau_a;
      const za_denom_mag_sq = za_denom_real * za_denom_real + za_denom_imag * za_denom_imag;
      const za_real = params.Ra * za_denom_real / za_denom_mag_sq;
      const za_imag = -params.Ra * za_denom_imag / za_denom_mag_sq;
      
      // Zb = Rb / (1 + jωτb)
      const zb_denom_real = 1.0;
      const zb_denom_imag = omega * tau_b;
      const zb_denom_mag_sq = zb_denom_real * zb_denom_real + zb_denom_imag * zb_denom_imag;
      const zb_real = params.Rb * zb_denom_real / zb_denom_mag_sq;
      const zb_imag = -params.Rb * zb_denom_imag / zb_denom_mag_sq;
      
      // Z_series = Za + Zb
      const z_series_real = za_real + zb_real;
      const z_series_imag = za_imag + zb_imag;
      
      // Z_total = (Rsh * Z_series) / (Rsh + Z_series)
      const num_real = params.Rsh * z_series_real;
      const num_imag = params.Rsh * z_series_imag;
      const denom_real = params.Rsh + z_series_real;
      const denom_imag = z_series_imag;
      const denom_mag_sq = denom_real * denom_real + denom_imag * denom_imag;
      
      const impedance_real = (num_real * denom_real + num_imag * denom_imag) / denom_mag_sq;
      const impedance_imag = (num_imag * denom_real - num_real * denom_imag) / denom_mag_sq;
      
      const magnitude = Math.sqrt(impedance_real * impedance_real + impedance_imag * impedance_imag);
      const phase = Math.atan2(impedance_imag, impedance_real);
      
      return {
        freq: refPoint.frequency,
        real: impedance_real,
        imag: impedance_imag,
        mag: magnitude,
        phase
      };
    });
  }

  /**
   * Get color for resnorm value
   */
  private getResnormColor(resnorm: number, bestResnorm: number): string {
    const ratio = resnorm / bestResnorm;
    if (ratio <= 1.1) return '#22c55e'; // Green for best
    if (ratio <= 1.5) return '#f59e0b'; // Yellow for good
    if (ratio <= 2.0) return '#f97316'; // Orange for okay
    return '#ef4444'; // Red for poor
  }
  
  /**
   * Get opacity for resnorm value
   */
  private getResnormOpacity(resnorm: number, allResults: HeapItem[]): number {
    if (allResults.length === 0) return 1.0;
    
    const resnorms = allResults.map(item => item.resnorm).sort((a, b) => a - b);
    const index = resnorms.findIndex(r => r >= resnorm);
    const normalized = index / resnorms.length;
    
    return Math.max(0.3, 1.0 - normalized * 0.7);
  }
  
  /**
   * Cancel the running pipeline
   */
  cancel(): void {
    this.isRunning = false;
  }
  
  /**
   * Get pipeline statistics
   */
  getStats() {
    return {
      isRunning: this.isRunning,
      currentStage: this.currentStage,
      config: this.config,
      totalCombinations: this.parameterGenerator.getTotalCombinations(),
      estimatedMemoryPerChunk: this.parameterGenerator.getEstimatedMemoryPerChunk()
    };
  }
}