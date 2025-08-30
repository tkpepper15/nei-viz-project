/**
 * Parameter Space Sampling Utilities
 * 
 * Based on 2024 best practices for parameter space exploration:
 * - Logarithmic scaling for parameters that vary across orders of magnitude
 * - Clean separation of concerns between UI and mathematical operations
 * - Efficient sampling methods following scikit-learn conventions
 */

export interface SamplingRange {
  min: number;
  max: number;
  scale: 'linear' | 'logarithmic';
}

/**
 * Parameter Space Sampler
 * 
 * Handles conversion between different parameter representations:
 * - Raw values (0-1 normalized range)
 * - Display values (actual parameter values)
 * - UI slider values (integer positions)
 */
export class ParameterSpaceSampler {
  private readonly range: SamplingRange;
  
  constructor(range: SamplingRange) {
    this.range = range;
    this.validateRange();
  }
  
  private validateRange(): void {
    if (this.range.min >= this.range.max) {
      throw new Error(`Invalid range: min (${this.range.min}) must be less than max (${this.range.max})`);
    }
    
    if (this.range.scale === 'logarithmic' && this.range.min <= 0) {
      throw new Error(`Logarithmic scale requires positive minimum value, got ${this.range.min}`);
    }
  }
  
  /**
   * Convert normalized value (0-1) to actual parameter value
   */
  denormalize(normalizedValue: number): number {
    const clamped = Math.max(0, Math.min(1, normalizedValue));
    
    if (this.range.scale === 'logarithmic') {
      const logMin = Math.log10(this.range.min);
      const logMax = Math.log10(this.range.max);
      const logValue = logMin + clamped * (logMax - logMin);
      return Math.pow(10, logValue);
    } else {
      return this.range.min + clamped * (this.range.max - this.range.min);
    }
  }
  
  /**
   * Convert actual parameter value to normalized value (0-1)
   */
  normalize(actualValue: number): number {
    const clamped = Math.max(this.range.min, Math.min(this.range.max, actualValue));
    
    if (this.range.scale === 'logarithmic') {
      const logMin = Math.log10(this.range.min);
      const logMax = Math.log10(this.range.max);
      const logValue = Math.log10(clamped);
      return (logValue - logMin) / (logMax - logMin);
    } else {
      return (clamped - this.range.min) / (this.range.max - this.range.min);
    }
  }
  
  /**
   * Convert UI slider position (integer) to normalized value
   */
  sliderToNormalized(sliderValue: number, sliderMin: number = 0, sliderMax: number = 100): number {
    const clamped = Math.max(sliderMin, Math.min(sliderMax, sliderValue));
    return (clamped - sliderMin) / (sliderMax - sliderMin);
  }
  
  /**
   * Convert normalized value to UI slider position (integer)
   */
  normalizedToSlider(normalizedValue: number, sliderMin: number = 0, sliderMax: number = 100): number {
    const clamped = Math.max(0, Math.min(1, normalizedValue));
    return Math.round(sliderMin + clamped * (sliderMax - sliderMin));
  }
  
  /**
   * Direct conversion from slider to actual value
   */
  sliderToActual(sliderValue: number, sliderMin: number = 0, sliderMax: number = 100): number {
    const normalized = this.sliderToNormalized(sliderValue, sliderMin, sliderMax);
    return this.denormalize(normalized);
  }
  
  /**
   * Direct conversion from actual value to slider
   */
  actualToSlider(actualValue: number, sliderMin: number = 0, sliderMax: number = 100): number {
    const normalized = this.normalize(actualValue);
    return this.normalizedToSlider(normalized, sliderMin, sliderMax);
  }
  
  /**
   * Generate sample points across the parameter space
   */
  generateSamples(numSamples: number): number[] {
    const samples: number[] = [];
    
    for (let i = 0; i < numSamples; i++) {
      const normalized = i / (numSamples - 1);
      samples.push(this.denormalize(normalized));
    }
    
    return samples;
  }
  
  /**
   * Get appropriate step size for UI controls
   */
  getStepSize(currentValue: number): number {
    if (this.range.scale === 'logarithmic') {
      // Dynamic step size based on order of magnitude
      const orderOfMagnitude = Math.floor(Math.log10(Math.max(0.01, currentValue)));
      return Math.pow(10, orderOfMagnitude - 2); // 1% of order of magnitude
    } else {
      // Linear step size: 1% of total range
      return (this.range.max - this.range.min) * 0.01;
    }
  }
  
  /**
   * Format value for display with appropriate precision
   */
  formatValue(value: number): string {
    if (this.range.scale === 'logarithmic') {
      if (value >= 100) return value.toFixed(0);
      if (value >= 10) return value.toFixed(1);
      if (value >= 1) return value.toFixed(2);
      return value.toFixed(3);
    } else {
      const range = this.range.max - this.range.min;
      if (range >= 1000) return value.toFixed(0);
      if (range >= 100) return value.toFixed(1);
      return value.toFixed(2);
    }
  }
}

/**
 * Pre-configured sampler for group portion (percentage values)
 * Range: 0.01% to 100% with logarithmic scaling
 */
export const groupPortionSampler = new ParameterSpaceSampler({
  min: 0.01,
  max: 100,
  scale: 'logarithmic'
});

/**
 * Convert group portion (0-1 decimal) to percentage for display
 */
export function groupPortionToPercentage(groupPortion: number): number {
  return Math.max(0.01, Math.min(100, groupPortion * 100));
}

/**
 * Convert percentage to group portion (0-1 decimal)
 */
export function percentageToGroupPortion(percentage: number): number {
  return Math.max(0.0001, Math.min(1, percentage / 100));
}

/**
 * Calculate number of models to show based on total and group portion
 */
export function calculateShownModels(totalModels: number, groupPortion: number): number {
  return Math.max(1, Math.floor(totalModels * Math.max(0.0001, Math.min(1, groupPortion))));
}