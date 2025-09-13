/**
 * TypeScript Frequency Serializer
 * ===============================
 * 
 * Port of the Python FrequencySerializer for use in React components.
 * Handles frequency configuration with scientific notation serialization.
 */

export class FrequencyConfig {
  constructor(
    public readonly minFreq: number,     // Minimum frequency (Hz)
    public readonly maxFreq: number,     // Maximum frequency (Hz) 
    public readonly nPoints: number,     // Number of frequency points
    public readonly spacing: 'log' | 'linear' = 'log'  // Frequency spacing
  ) {
    if (minFreq <= 0) throw new Error('Minimum frequency must be positive');
    if (maxFreq <= minFreq) throw new Error('Maximum frequency must be greater than minimum');
    if (nPoints < 2) throw new Error('Number of points must be at least 2');
  }
  
  toId(): string {
    const spacingCode = this.spacing === 'log' ? 'L' : 'N';
    const minSci = this.minFreq.toExponential(1);
    const maxSci = this.maxFreq.toExponential(1);
    return `${spacingCode}_${minSci}_${maxSci}_${this.nPoints.toString().padStart(3, '0')}`;
  }
  
  static fromId(freqId: string): FrequencyConfig {
    const parts = freqId.split('_');
    if (parts.length !== 4) {
      throw new Error(`Invalid frequency ID format: ${freqId}`);
    }
    
    const [spacingCode, minSci, maxSci, nPointsStr] = parts;
    
    const spacing = spacingCode === 'L' ? 'log' : 'linear';
    const minFreq = parseFloat(minSci);
    const maxFreq = parseFloat(maxSci);
    const nPoints = parseInt(nPointsStr, 10);
    
    return new FrequencyConfig(minFreq, maxFreq, nPoints, spacing);
  }
  
  generateFrequencies(): number[] {
    const frequencies: number[] = [];
    
    if (this.spacing === 'log') {
      const logMin = Math.log10(this.minFreq);
      const logMax = Math.log10(this.maxFreq);
      const logStep = (logMax - logMin) / (this.nPoints - 1);
      
      for (let i = 0; i < this.nPoints; i++) {
        const logValue = logMin + i * logStep;
        frequencies.push(Math.pow(10, logValue));
      }
    } else {
      const step = (this.maxFreq - this.minFreq) / (this.nPoints - 1);
      
      for (let i = 0; i < this.nPoints; i++) {
        frequencies.push(this.minFreq + i * step);
      }
    }
    
    return frequencies;
  }
  
  getFrequencyInfo() {
    const frequencies = this.generateFrequencies();
    const decades = this.spacing === 'log' ? Math.log10(this.maxFreq / this.minFreq) : null;
    
    return {
      id: this.toId(),
      config: {
        minFreq: this.minFreq,
        maxFreq: this.maxFreq,
        nPoints: this.nPoints,
        spacing: this.spacing
      },
      frequencies: {
        first: `${frequencies[0].toExponential(2)} Hz`,
        last: `${frequencies[frequencies.length - 1].toExponential(2)} Hz`,
        center: `${frequencies[Math.floor(frequencies.length / 2)].toExponential(2)} Hz`,
        stepSize: this.spacing === 'linear' ? `${(frequencies[1] - frequencies[0]).toExponential(2)} Hz` : null,
        decadeSpan: decades ? `${decades.toFixed(1)} decades` : null,
        pointsPerDecade: decades ? `${(this.nPoints / decades).toFixed(1)}` : null
      },
      arrayPreview: {
        first5: frequencies.slice(0, 5).map(f => f.toExponential(2)),
        last5: frequencies.slice(-5).map(f => f.toExponential(2))
      }
    };
  }
}

export class FrequencySerializer {
  private presets: Map<string, FrequencyConfig>;
  
  constructor() {
    this.presets = new Map();
    this.createStandardPresets();
  }
  
  private createStandardPresets(): void {
    this.presets.set('standard', new FrequencyConfig(1e-1, 1e5, 100, 'log'));        // L_1.0E-01_1.0E+05_100
    this.presets.set('high_res', new FrequencyConfig(1e-2, 1e6, 500, 'log'));        // L_1.0E-02_1.0E+06_500
    this.presets.set('fast', new FrequencyConfig(1e0, 1e4, 50, 'log'));              // L_1.0E+00_1.0E+04_050
    this.presets.set('low_freq', new FrequencyConfig(1e-3, 1e2, 200, 'log'));        // L_1.0E-03_1.0E+02_200
    this.presets.set('linear', new FrequencyConfig(1e1, 1e3, 100, 'linear'));        // N_1.0E+01_1.0E+03_100
    this.presets.set('ultra_wide', new FrequencyConfig(1e-4, 1e7, 1000, 'log'));     // L_1.0E-04_1.0E+07_1000
    this.presets.set('mid_range', new FrequencyConfig(1e1, 1e5, 250, 'log'));        // L_1.0E+01_1.0E+05_250
  }
  
  getPreset(name: string): FrequencyConfig | undefined {
    return this.presets.get(name);
  }
  
  listPresets(): string[] {
    return Array.from(this.presets.keys());
  }
  
  getPresetInfo(name: string) {
    const preset = this.presets.get(name);
    return preset ? preset.getFrequencyInfo() : null;
  }
  
  createCustomConfig(minFreq: number, maxFreq: number, nPoints: number, spacing: 'log' | 'linear' = 'log'): FrequencyConfig {
    return new FrequencyConfig(minFreq, maxFreq, nPoints, spacing);
  }
  
  registerPreset(name: string, config: FrequencyConfig): void {
    this.presets.set(name, config);
  }
  
  getAllPresetInfo() {
    const info: Record<string, ReturnType<FrequencyConfig['getFrequencyInfo']>> = {};
    for (const [name, config] of this.presets) {
      info[name] = config.getFrequencyInfo();
    }
    return info;
  }
}

export class ComputationResult {
  constructor(
    public readonly circuitConfigId: string,      // e.g. "15_01_02_04_03_05"
    public readonly frequencyConfigId: string,    // e.g. "L_1.0E-01_1.0E+05_100"
    public readonly resnorm: number,              // Computed residual norm
    public readonly computationTime?: number,
    public readonly timestamp?: string
  ) {}
  
  toCompactString(): string {
    return `${this.circuitConfigId}|${this.frequencyConfigId}|${this.resnorm.toExponential(6)}`;
  }
  
  static fromCompactString(compactStr: string): ComputationResult {
    const parts = compactStr.split('|');
    if (parts.length !== 3) {
      throw new Error(`Invalid compact string format: ${compactStr}`);
    }
    
    const [circuitId, freqId, resnormStr] = parts;
    return new ComputationResult(circuitId, freqId, parseFloat(resnormStr));
  }
  
  getStorageSize(): number {
    return this.toCompactString().length;
  }
}

/**
 * Utility functions
 */
export const FrequencyUtils = {
  /**
   * Create standard frequency serializer
   */
  createStandard(): FrequencySerializer {
    return new FrequencySerializer();
  },
  
  /**
   * Get most commonly used presets
   */
  getCommonPresets(): string[] {
    return ['standard', 'high_res', 'fast', 'low_freq'];
  },
  
  /**
   * Calculate storage efficiency compared to traditional arrays
   */
  calculateStorageEfficiency(results: ComputationResult[]): {
    traditionalSizeBytes: number;
    serializedSizeBytes: number;
    reductionFactor: number;
  } {
    if (results.length === 0) {
      return { traditionalSizeBytes: 0, serializedSizeBytes: 0, reductionFactor: 0 };
    }
    
    // Get frequency config to estimate array sizes
    const firstResult = results[0];
    const freqConfig = FrequencyConfig.fromId(firstResult.frequencyConfigId);
    
    const traditionalSize = results.length * (
      5 * 8 +                           // Circuit parameters (5 floats)
      freqConfig.nPoints * 8 +          // Frequency array
      freqConfig.nPoints * 16           // Complex impedance array
    );
    
    const serializedSize = results.reduce((sum, result) => sum + result.getStorageSize(), 0);
    
    return {
      traditionalSizeBytes: traditionalSize,
      serializedSizeBytes: serializedSize,
      reductionFactor: traditionalSize / serializedSize
    };
  },
  
  /**
   * Generate computation results for testing
   */
  generateTestResults(configIds: string[], frequencyPreset: string = 'standard', count: number = 100): ComputationResult[] {
    const results: ComputationResult[] = [];
    const serializer = new FrequencySerializer();
    const freqConfig = serializer.getPreset(frequencyPreset);
    
    if (!freqConfig) {
      throw new Error(`Unknown frequency preset: ${frequencyPreset}`);
    }
    
    const frequencyConfigId = freqConfig.toId();
    
    for (let i = 0; i < Math.min(count, configIds.length); i++) {
      const resnorm = Math.random() * 10; // Random resnorm for testing
      results.push(new ComputationResult(
        configIds[i],
        frequencyConfigId,
        resnorm,
        Math.random() * 5, // Random computation time
        new Date().toISOString()
      ));
    }
    
    return results.sort((a, b) => a.resnorm - b.resnorm);
  }
};