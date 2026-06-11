/**
 * Streaming Parameter Iterator
 * Generates parameter combinations without materializing all 9.7M objects in memory
 * Uses generators for memory-efficient traversal of parameter space
 */

import { CircuitParameters, PARAMETER_RANGES } from '../types/parameters';

export interface ParameterGeneratorConfig {
  gridSize: number;
  chunkSize: number;
  groundTruthParams?: CircuitParameters;
  includeCornersAndCenter?: boolean;
  useSymmetricOptimization?: boolean;
  samplingMode: 'logarithmic' | 'linear' | 'hybrid';
}

export interface ParameterChunk {
  parameters: CircuitParameters[];
  chunkIndex: number;
  startIndex: number;
  endIndex: number;
  totalCombinations: number;
}

export interface ParameterRange {
  min: number;
  max: number;
  values: Float64Array;
  isLogarithmic: boolean;
}

/**
 * Generate parameter values with optional ground truth inclusion
 */
function generateParameterValues(
  min: number, 
  max: number, 
  gridSize: number, 
  groundTruthValue?: number,
  samplingMode: 'logarithmic' | 'linear' = 'logarithmic'
): Float64Array {
  const values = new Float64Array(gridSize);
  
  if (samplingMode === 'logarithmic') {
    const logMin = Math.log10(min);
    const logMax = Math.log10(max);
    
    for (let i = 0; i < gridSize; i++) {
      const logValue = logMin + (i / (gridSize - 1)) * (logMax - logMin);
      values[i] = Math.pow(10, logValue);
    }
  } else {
    // Linear sampling
    for (let i = 0; i < gridSize; i++) {
      values[i] = min + (i / (gridSize - 1)) * (max - min);
    }
  }
  
  // Include ground truth value if provided (replace closest grid point)
  if (groundTruthValue !== undefined && groundTruthValue >= min && groundTruthValue <= max) {
    let closestIndex = 0;
    let minDiff = Math.abs(values[0] - groundTruthValue);
    
    for (let i = 1; i < gridSize; i++) {
      const diff = Math.abs(values[i] - groundTruthValue);
      if (diff < minDiff) {
        minDiff = diff;
        closestIndex = i;
      }
    }
    
    values[closestIndex] = groundTruthValue;
  }
  
  return values;
}

/**
 * Convert 5D index to parameter combination
 */
function indexToParameters(
  index: number,
  ranges: {
    Rsh: ParameterRange;
    Ra: ParameterRange;
    Rb: ParameterRange;
    Ca: ParameterRange;
    Cb: ParameterRange;
  },
  gridSize: number
): CircuitParameters {
  // Convert linear index to 5D coordinates
  const cbIndex = index % gridSize;
  const temp1 = Math.floor(index / gridSize);
  
  const rbIndex = temp1 % gridSize;
  const temp2 = Math.floor(temp1 / gridSize);
  
  const caIndex = temp2 % gridSize;
  const temp3 = Math.floor(temp2 / gridSize);
  
  const raIndex = temp3 % gridSize;
  const rshIndex = Math.floor(temp3 / gridSize);
  
  return {
    Rsh: ranges.Rsh.values[rshIndex],
    Ra: ranges.Ra.values[raIndex],
    Ca: ranges.Ca.values[caIndex],
    Rb: ranges.Rb.values[rbIndex],
    Cb: ranges.Cb.values[cbIndex],
    frequency_range: [1.0, 10000.0] // Standard frequency range
  };
}

/**
 * Check if parameter combination should be skipped (symmetric optimization)
 */
function shouldSkipSymmetric(params: CircuitParameters): boolean {
  // Skip if tau_a > tau_b (we'll get equivalent from swapped version)
  const tau_a = params.Ra * params.Ca;
  const tau_b = params.Rb * params.Cb;
  return tau_a > tau_b;
}

/**
 * Streaming Parameter Generator Class
 */
export class StreamingParameterGenerator {
  private config: ParameterGeneratorConfig;
  private ranges: {
    Rsh: ParameterRange;
    Ra: ParameterRange;
    Rb: ParameterRange;
    Ca: ParameterRange;
    Cb: ParameterRange;
  };
  private totalCombinations: number;
  
  constructor(config: ParameterGeneratorConfig) {
    this.config = config;
    this.totalCombinations = Math.pow(config.gridSize, 5);
    
    // Initialize parameter ranges
    this.ranges = {
      Rsh: {
        min: PARAMETER_RANGES.Rsh.min,
        max: PARAMETER_RANGES.Rsh.max,
        values: generateParameterValues(
          PARAMETER_RANGES.Rsh.min,
          PARAMETER_RANGES.Rsh.max,
          config.gridSize,
          config.groundTruthParams?.Rsh,
          'linear' // Resistance typically linear
        ),
        isLogarithmic: false
      },
      Ra: {
        min: PARAMETER_RANGES.Ra.min,
        max: PARAMETER_RANGES.Ra.max,
        values: generateParameterValues(
          PARAMETER_RANGES.Ra.min,
          PARAMETER_RANGES.Ra.max,
          config.gridSize,
          config.groundTruthParams?.Ra,
          'linear'
        ),
        isLogarithmic: false
      },
      Rb: {
        min: PARAMETER_RANGES.Rb.min,
        max: PARAMETER_RANGES.Rb.max,
        values: generateParameterValues(
          PARAMETER_RANGES.Rb.min,
          PARAMETER_RANGES.Rb.max,
          config.gridSize,
          config.groundTruthParams?.Rb,
          'linear'
        ),
        isLogarithmic: false
      },
      Ca: {
        min: PARAMETER_RANGES.Ca.min,
        max: PARAMETER_RANGES.Ca.max,
        values: generateParameterValues(
          PARAMETER_RANGES.Ca.min,
          PARAMETER_RANGES.Ca.max,
          config.gridSize,
          config.groundTruthParams?.Ca,
          'logarithmic' // Capacitance typically logarithmic
        ),
        isLogarithmic: true
      },
      Cb: {
        min: PARAMETER_RANGES.Cb.min,
        max: PARAMETER_RANGES.Cb.max,
        values: generateParameterValues(
          PARAMETER_RANGES.Cb.min,
          PARAMETER_RANGES.Cb.max,
          config.gridSize,
          config.groundTruthParams?.Cb,
          'logarithmic'
        ),
        isLogarithmic: true
      }
    };
  }
  
  /**
   * Generator function that yields parameter chunks
   */
  *generateChunks(): Generator<ParameterChunk, void, unknown> {
    let chunkIndex = 0;
    let processedCount = 0;
    
    for (let startIndex = 0; startIndex < this.totalCombinations; startIndex += this.config.chunkSize) {
      const endIndex = Math.min(startIndex + this.config.chunkSize, this.totalCombinations);
      const chunkParameters: CircuitParameters[] = [];
      
      for (let index = startIndex; index < endIndex; index++) {
        const params = indexToParameters(index, this.ranges, this.config.gridSize);
        
        // Apply symmetric optimization if enabled
        if (this.config.useSymmetricOptimization && shouldSkipSymmetric(params)) {
          continue;
        }
        
        // Validate parameters
        if (this.isValidParameterSet(params)) {
          chunkParameters.push(params);
        }
      }
      
      processedCount += chunkParameters.length;
      
      yield {
        parameters: chunkParameters,
        chunkIndex,
        startIndex,
        endIndex,
        totalCombinations: this.totalCombinations
      };
      
      chunkIndex++;
    }
    
    console.log(`📊 Parameter generation complete: ${processedCount}/${this.totalCombinations} valid combinations processed`);
  }
  
  /**
   * Generate specific parameter combinations (corners, center, etc.)
   */
  *generateSpecialPoints(): Generator<CircuitParameters[], void, unknown> {
    const specialPoints: CircuitParameters[] = [];
    
    if (this.config.includeCornersAndCenter) {
      // Add corner points (min/max combinations)
      const cornerIndices = [
        [0, 0, 0, 0, 0], // All minimums
        [0, 0, 0, 0, this.config.gridSize - 1],
        [0, 0, 0, this.config.gridSize - 1, 0],
        [0, 0, this.config.gridSize - 1, 0, 0],
        [0, this.config.gridSize - 1, 0, 0, 0],
        [this.config.gridSize - 1, 0, 0, 0, 0],
        [this.config.gridSize - 1, this.config.gridSize - 1, this.config.gridSize - 1, this.config.gridSize - 1, this.config.gridSize - 1], // All maximums
      ];
      
      for (const indices of cornerIndices) {
        const params: CircuitParameters = {
          Rsh: this.ranges.Rsh.values[indices[0]],
          Ra: this.ranges.Ra.values[indices[1]],
          Ca: this.ranges.Ca.values[indices[2]],
          Rb: this.ranges.Rb.values[indices[3]],
          Cb: this.ranges.Cb.values[indices[4]],
          frequency_range: [1.0, 10000.0]
        };
        
        if (this.isValidParameterSet(params)) {
          specialPoints.push(params);
        }
      }
      
      // Add center point
      const centerIndex = Math.floor(this.config.gridSize / 2);
      const centerParams: CircuitParameters = {
        Rsh: this.ranges.Rsh.values[centerIndex],
        Ra: this.ranges.Ra.values[centerIndex],
        Ca: this.ranges.Ca.values[centerIndex],
        Rb: this.ranges.Rb.values[centerIndex],
        Cb: this.ranges.Cb.values[centerIndex],
        frequency_range: [1.0, 10000.0]
      };
      
      if (this.isValidParameterSet(centerParams)) {
        specialPoints.push(centerParams);
      }
    }
    
    // Add ground truth if provided
    if (this.config.groundTruthParams && this.isValidParameterSet(this.config.groundTruthParams)) {
      specialPoints.push(this.config.groundTruthParams);
    }
    
    yield specialPoints;
  }
  
  /**
   * Validate parameter set
   */
  private isValidParameterSet(params: CircuitParameters): boolean {
    // Check for finite values
    if (!isFinite(params.Rsh) || !isFinite(params.Ra) || !isFinite(params.Ca) || 
        !isFinite(params.Rb) || !isFinite(params.Cb)) {
      return false;
    }
    
    // Check for positive values
    if (params.Rsh <= 0 || params.Ra <= 0 || params.Ca <= 0 || 
        params.Rb <= 0 || params.Cb <= 0) {
      return false;
    }
    
    // Check ranges
    if (params.Rsh < PARAMETER_RANGES.Rsh.min || params.Rsh > PARAMETER_RANGES.Rsh.max ||
        params.Ra < PARAMETER_RANGES.Ra.min || params.Ra > PARAMETER_RANGES.Ra.max ||
        params.Ca < PARAMETER_RANGES.Ca.min || params.Ca > PARAMETER_RANGES.Ca.max ||
        params.Rb < PARAMETER_RANGES.Rb.min || params.Rb > PARAMETER_RANGES.Rb.max ||
        params.Cb < PARAMETER_RANGES.Cb.min || params.Cb > PARAMETER_RANGES.Cb.max) {
      return false;
    }
    
    return true;
  }
  
  /**
   * Get total number of combinations
   */
  getTotalCombinations(): number {
    return this.totalCombinations;
  }
  
  /**
   * Get estimated memory usage per chunk
   */
  getEstimatedMemoryPerChunk(): number {
    // Rough estimate: 32 bytes per parameter set + overhead
    return this.config.chunkSize * 32;
  }
  
  /**
   * Get generator configuration
   */
  getConfig(): ParameterGeneratorConfig {
    return { ...this.config };
  }
  
  /**
   * Get parameter ranges information
   */
  getRangesInfo() {
    return Object.entries(this.ranges).map(([key, range]) => ({
      parameter: key,
      min: range.min,
      max: range.max,
      count: range.values.length,
      isLogarithmic: range.isLogarithmic,
      values: range.values.slice(0, 5) // First 5 values for preview
    }));
  }
}