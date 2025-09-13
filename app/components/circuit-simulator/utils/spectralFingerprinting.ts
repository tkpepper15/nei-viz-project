/**
 * Spectral Fingerprinting System for Circuit Parameter Optimization
 * Groups parameter sets by their impedance spectral characteristics
 * This dramatically reduces computational load while preserving mathematical accuracy
 */

import { CircuitParameters } from '../types/parameters';
import { FingerprintHeap, HeapItem } from './topKHeap';

export interface SpectralFingerprint {
  key: string;                    // Quantized fingerprint key for grouping
  logMagnitudes: Float64Array;    // Log10 magnitudes at fingerprint frequencies
  parameters: CircuitParameters;   // Original parameters that generated this fingerprint
  cheapResnorm: number;           // Fast screening resnorm (magnitude-only, few frequencies)
}

export interface FingerprintConfig {
  fingerprintFrequencies: Float64Array;  // 6-8 log-spaced frequencies for fingerprinting
  approximationFrequencies: Float64Array; // 5 frequencies for cheap resnorm screening
  quantizationBin: number;        // e.g., 0.05 dex for grouping similar spectra
  candidatesPerFingerprint: number; // Max candidates to keep per fingerprint group
}

/**
 * Default configuration based on the proposed plan
 */
export const DEFAULT_FINGERPRINT_CONFIG: FingerprintConfig = {
  // 8 log-spaced fingerprint frequencies from 1Hz to 10kHz
  fingerprintFrequencies: new Float64Array([
    1.0, 2.154, 4.642, 10.0, 21.54, 46.42, 100.0, 215.4
  ]),
  
  // 5 approximation frequencies for cheap screening
  approximationFrequencies: new Float64Array([
    1.0, 10.0, 100.0, 1000.0, 10000.0
  ]),
  
  quantizationBin: 0.05, // 0.05 dex = ~12% magnitude bins
  candidatesPerFingerprint: 3
};

/**
 * Generate log-spaced frequency array
 */
export function generateLogFrequencies(minFreq: number, maxFreq: number, numPoints: number): Float64Array {
  const logMin = Math.log10(minFreq);
  const logMax = Math.log10(maxFreq);
  const frequencies = new Float64Array(numPoints);
  
  for (let i = 0; i < numPoints; i++) {
    const logValue = logMin + (i / (numPoints - 1)) * (logMax - logMin);
    frequencies[i] = Math.pow(10, logValue);
  }
  
  return frequencies;
}

/**
 * Calculate impedance magnitude for a single frequency (optimized)
 * Uses time-constant reparameterization for speed: τa = Ra*Ca, τb = Rb*Cb
 */
function calculateMagnitude(params: CircuitParameters, omega: number): number {
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
  
  const z_total_real = (num_real * denom_real + num_imag * denom_imag) / denom_mag_sq;
  const z_total_imag = (num_imag * denom_real - num_real * denom_imag) / denom_mag_sq;
  
  return Math.sqrt(z_total_real * z_total_real + z_total_imag * z_total_imag);
}

/**
 * Calculate complex impedance for full SSR computation
 */
function calculateComplexImpedance(params: CircuitParameters, omega: number): { real: number; imag: number } {
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
  
  return {
    real: (num_real * denom_real + num_imag * denom_imag) / denom_mag_sq,
    imag: (num_imag * denom_real - num_real * denom_imag) / denom_mag_sq
  };
}

/**
 * Generate spectral fingerprint for a parameter set
 */
export function generateFingerprint(
  params: CircuitParameters,
  config: FingerprintConfig
): SpectralFingerprint {
  const logMagnitudes = new Float64Array(config.fingerprintFrequencies.length);
  
  // Compute log magnitudes at fingerprint frequencies
  for (let i = 0; i < config.fingerprintFrequencies.length; i++) {
    const frequency = config.fingerprintFrequencies[i];
    const omega = 2.0 * Math.PI * frequency;
    const magnitude = calculateMagnitude(params, omega);
    logMagnitudes[i] = Math.log10(magnitude);
  }
  
  // Quantize to create grouping key
  const quantizedValues = new Float64Array(logMagnitudes.length);
  for (let i = 0; i < logMagnitudes.length; i++) {
    quantizedValues[i] = Math.round(logMagnitudes[i] / config.quantizationBin) * config.quantizationBin;
  }
  
  // Create string key for grouping
  const key = Array.from(quantizedValues)
    .map(val => val.toFixed(2))
    .join(',');
  
  return {
    key,
    logMagnitudes,
    parameters: params,
    cheapResnorm: 0 // Will be computed separately
  };
}

/**
 * Compute cheap resnorm for screening (magnitude-only, few frequencies)
 */
export function computeCheapResnorm(
  params: CircuitParameters,
  referenceMagnitudes: Float64Array,
  config: FingerprintConfig
): number {
  let totalError = 0;
  let validPoints = 0;
  
  for (let i = 0; i < config.approximationFrequencies.length; i++) {
    const frequency = config.approximationFrequencies[i];
    const omega = 2.0 * Math.PI * frequency;
    const testMagnitude = calculateMagnitude(params, omega);
    const refMagnitude = referenceMagnitudes[i];
    
    if (isFinite(testMagnitude) && isFinite(refMagnitude)) {
      const error = Math.abs(testMagnitude - refMagnitude);
      totalError += error;
      validPoints++;
    }
  }
  
  return validPoints > 0 ? totalError / validPoints : Infinity;
}

/**
 * Compute full complex SSR for final refinement
 */
export function computeFullSSR(
  params: CircuitParameters,
  referenceSpectrum: Array<{ frequency: number; real: number; imag: number }>,
  frequencyWeights?: Float64Array
): number {
  let totalSSR = 0;
  let validPoints = 0;
  
  for (let i = 0; i < referenceSpectrum.length; i++) {
    const { frequency, real: refReal, imag: refImag } = referenceSpectrum[i];
    const omega = 2.0 * Math.PI * frequency;
    const impedance = calculateComplexImpedance(params, omega);
    
    if (isFinite(impedance.real) && isFinite(impedance.imag)) {
      const realDiff = impedance.real - refReal;
      const imagDiff = impedance.imag - refImag;
      const ssr = realDiff * realDiff + imagDiff * imagDiff;
      
      // Apply frequency weighting if provided
      const weight = frequencyWeights ? frequencyWeights[i] : 1.0;
      totalSSR += ssr * weight;
      validPoints++;
    }
  }
  
  return validPoints > 0 ? totalSSR / validPoints : Infinity;
}

/**
 * Spectral Fingerprinting Manager
 * Orchestrates the fingerprint-based screening process
 */
export class SpectralFingerprintManager {
  private config: FingerprintConfig;
  private fingerprintHeap: FingerprintHeap;
  private referenceMagnitudes: Float64Array;
  private referenceSpectrum: Array<{ frequency: number; real: number; imag: number }>;
  
  constructor(
    config: FingerprintConfig = DEFAULT_FINGERPRINT_CONFIG,
    referenceSpectrum: Array<{ frequency: number; real: number; imag: number }>
  ) {
    this.config = config;
    this.fingerprintHeap = new FingerprintHeap(config.candidatesPerFingerprint);
    this.referenceSpectrum = referenceSpectrum;
    
    // Precompute reference magnitudes at approximation frequencies
    this.referenceMagnitudes = new Float64Array(config.approximationFrequencies.length);
    for (let i = 0; i < config.approximationFrequencies.length; i++) {
      const targetFreq = config.approximationFrequencies[i];
      
      // Find closest reference frequency
      let closestIndex = 0;
      let minDiff = Math.abs(referenceSpectrum[0].frequency - targetFreq);
      
      for (let j = 1; j < referenceSpectrum.length; j++) {
        const diff = Math.abs(referenceSpectrum[j].frequency - targetFreq);
        if (diff < minDiff) {
          minDiff = diff;
          closestIndex = j;
        }
      }
      
      const { real, imag } = referenceSpectrum[closestIndex];
      this.referenceMagnitudes[i] = Math.sqrt(real * real + imag * imag);
    }
  }
  
  /**
   * Process a chunk of parameters through fingerprinting
   */
  processChunk(parameterChunk: CircuitParameters[]): {
    processed: number;
    uniqueFingerprints: number;
    representatives: HeapItem[];
  } {
    let processed = 0;
    
    for (const params of parameterChunk) {
      try {
        // Generate spectral fingerprint
        const fingerprint = generateFingerprint(params, this.config);
        
        // Compute cheap resnorm for screening
        const cheapResnorm = computeCheapResnorm(params, this.referenceMagnitudes, this.config);
        
        // Add to fingerprint heap (grouped by spectral characteristics)
        const heapItem: HeapItem = {
          resnorm: cheapResnorm,
          parameters: params,
          fingerprint: fingerprint.key,
          metadata: {
            stage: 1
          }
        };
        
        this.fingerprintHeap.addCandidate(heapItem, fingerprint.key);
        processed++;
        
      } catch (error) {
        console.warn('Error processing parameter set:', params, error);
      }
    }
    
    const representatives = this.fingerprintHeap.getRepresentatives();
    const stats = this.fingerprintHeap.getStats();
    
    return {
      processed,
      uniqueFingerprints: stats.uniqueFingerprints,
      representatives
    };
  }
  
  /**
   * Refine representatives with full SSR computation
   */
  refineWithFullSSR(representatives: HeapItem[]): HeapItem[] {
    const refined: HeapItem[] = [];
    
    for (const item of representatives) {
      try {
        // Create complete parameters with frequency_range
        const completeParams: CircuitParameters = {
          ...item.parameters,
          frequency_range: [
            this.referenceSpectrum[0].frequency,
            this.referenceSpectrum[this.referenceSpectrum.length - 1].frequency
          ]
        };
        const fullSSR = computeFullSSR(completeParams, this.referenceSpectrum);
        
        refined.push({
          ...item,
          resnorm: fullSSR,
          metadata: {
            ...item.metadata,
            stage: 3
          }
        });
        
      } catch (error) {
        console.warn('Error computing full SSR for:', item.parameters, error);
      }
    }
    
    return refined.sort((a, b) => a.resnorm - b.resnorm);
  }
  
  /**
   * Get current statistics
   */
  getStats() {
    return this.fingerprintHeap.getStats();
  }
  
  /**
   * Clear all data
   */
  clear(): void {
    this.fingerprintHeap.clear();
  }
}