/**
 * Utility functions for calculating residual norms and circuit parameters
 */

import { CircuitParameters } from '../types/parameters';

/**
 * Available resnorm calculation methods for EIS parameter fitting
 * MAE is preferred for EIS as noted in battery research literature
 */
export enum ResnormMethod {
  MAE = 'mae',          // Mean Absolute Error (preferred for EIS parameter extraction)
  SSR = 'ssr',          // Sum of Squared Residuals (classic least-squares in complex plane)
  RMSE = 'rmse'         // Root Mean Square Error
}

/**
 * Configuration for resnorm calculation
 */
export interface ResnormConfig {
  method: ResnormMethod;
  useRangeAmplification?: boolean;
  useFrequencyWeighting?: boolean;
}

// Define local interfaces for clear types
export interface Impedance {
  real: number;
  imaginary: number;
}

export interface ImpedancePoint {
  frequency: number;
  real: number;
  imaginary: number;
  magnitude: number;
  phase: number;
}

/**
 * Complex number implementation for impedance calculations
 */
export interface Complex {
  real: number;
  imag: number;
}

/**
 * Create a complex number
 * @param real Real part
 * @param imag Imaginary part
 * @returns Complex number
 */
export function createComplex(real: number, imag: number): Complex {
  return { real, imag };
}

/**
 * Calculate impedance at a given frequency
 * 
 * Z_total = Rsh || (Za + Zb) where:
 * - Za = Ra/(1+jωRaCa) 
 * - Zb = Rb/(1+jωRbCb)
 * 
 * @param params Circuit parameters
 * @param frequency Frequency in Hz
 * @returns Complex impedance with real and imaginary parts
 */
export const calculateImpedance = (params: CircuitParameters, frequency: number): Impedance => {
  const { Rsh, Ra, Ca, Rb, Cb } = params;
  const omega = 2 * Math.PI * frequency;
  
  // Calculate impedance of apical membrane (Ra || Ca)
  // Z_parallel = (Ra) / (1 + jωRaCa)
  const Za_real = Ra / (1 + Math.pow(omega * Ra * Ca, 2));
  const Za_imag = -omega * Ra * Ra * Ca / (1 + Math.pow(omega * Ra * Ca, 2));
  
  // Calculate impedance of basal membrane (Rb || Cb)
  // Z_parallel = (Rb) / (1 + jωRbCb)
  const Zb_real = Rb / (1 + Math.pow(omega * Rb * Cb, 2));
  const Zb_imag = -omega * Rb * Rb * Cb / (1 + Math.pow(omega * Rb * Cb, 2));
  
  // Calculate sum of membrane impedances (Za + Zb)
  const Zab_real = Za_real + Zb_real;
  const Zab_imag = Za_imag + Zb_imag;
  
  // Calculate parallel combination: Z_total = (Rsh * (Za + Zb)) / (Rsh + Za + Zb)
  // Numerator: Rsh * (Za + Zb)
  const num_real = Rsh * Zab_real;
  const num_imag = Rsh * Zab_imag;
  
  // Denominator: Rsh + Za + Zb
  const denom_real = Rsh + Zab_real;
  const denom_imag = Zab_imag;
  
  // Complex division: (num_real + j*num_imag) / (denom_real + j*denom_imag)
  const denom_mag_squared = denom_real * denom_real + denom_imag * denom_imag;
  
  const real = (num_real * denom_real + num_imag * denom_imag) / denom_mag_squared;
  const imaginary = (num_imag * denom_real - num_real * denom_imag) / denom_mag_squared;
  
  return { real, imaginary };
};

/**
 * Calculate the complete impedance spectrum for a given set of parameters
 * 
 * @param params Circuit parameters
 * @returns Array of impedance points across the frequency range
 */
export const calculate_impedance_spectrum = (params: CircuitParameters): ImpedancePoint[] => {
  const spectrum: ImpedancePoint[] = [];
  
  // Extract frequency range
  const [min_freq, max_freq] = params.frequency_range;
  
  // Generate logarithmically spaced frequencies (20 points by default)
  const num_points = 20;
  const frequencies: number[] = [];
  const logMin = Math.log10(min_freq);
  const logMax = Math.log10(max_freq);
  const logStep = (logMax - logMin) / (num_points - 1);
  
  for (let i = 0; i < num_points; i++) {
    const logValue = logMin + i * logStep;
    const frequency = Math.pow(10, logValue);
    frequencies.push(frequency);
  }
  
  // Calculate impedance at each frequency
  for (const frequency of frequencies) {
    const impedance = calculateImpedance(params, frequency);
    const magnitude = Math.sqrt(Math.pow(impedance.real, 2) + Math.pow(impedance.imaginary, 2));
    const phase = Math.atan2(impedance.imaginary, impedance.real) * (180 / Math.PI);
    
    spectrum.push({
      frequency,
      real: impedance.real,
      imaginary: impedance.imaginary,
      magnitude,
      phase
    });
  }
  
  return spectrum;
};

/**
 * Calculate residual norm (resnorm) between reference and test data using various methods.
 * Supports MAE, SSR, and RMSE methods.
 * 
 * @param testData Array of impedance points to compare against reference
 * @param referenceData Array of reference impedance points
 * @param logFunction Optional logging function for debugging
 * @param frequency Optional frequency for single point comparison
 * @param config Configuration object with method and parameters
 * @returns The calculated resnorm value using specified method
 */
export const calculateResnormWithConfig = (
  testData: ImpedancePoint[] | ImpedancePoint,
  referenceData: ImpedancePoint[] | number,
  logFunction?: ((message: string) => void) | number,
  frequency?: number,
  config: ResnormConfig = { method: ResnormMethod.MAE }
): number => {
  // Handle the case where we're passing individual points
  if (!Array.isArray(testData) && typeof referenceData === 'number' && typeof logFunction === 'number') {
    // In this case:
    // testData is a single ImpedancePoint
    // referenceData is the real part of reference impedance
    // logFunction is actually the imaginary part of reference impedance
    // frequency is the frequency at which to compare
    
    const testPoint = testData;
    const refReal = referenceData;
    const refImag = logFunction as number;
    const refFreq = frequency ?? testPoint.frequency;
    
    // Create a synthetic reference point
    const refPoint: ImpedancePoint = {
      frequency: refFreq,
      real: refReal,
      imaginary: refImag,
      magnitude: Math.sqrt(refReal * refReal + refImag * refImag),
      phase: Math.atan2(refImag, refReal) * (180 / Math.PI)
    };
    
    // Convert to arrays and call the main implementation
    return calculateResnormWithConfig([testPoint], [refPoint], undefined, undefined, config);
  }
  
  // Handle arrays case (main implementation)
  const testDataArray = Array.isArray(testData) ? testData : [testData];
  const referenceDataArray = Array.isArray(referenceData) ? referenceData : [referenceData as unknown as ImpedancePoint];
  const logger = typeof logFunction === 'function' ? logFunction : undefined;

  // Ensure we have data to compare
  if (!testDataArray || !referenceDataArray || testDataArray.length === 0 || referenceDataArray.length === 0) {
    if (logger) logger("No data to compare");
    return Infinity;
  }

  // Match frequencies between test and reference data
  const matchedData: Array<{
    frequency: number;
    test: Impedance;
    reference: Impedance;
  }> = [];

  // Process each test data point
  for (const testPoint of testDataArray) {
    // Safety check for testPoint structure
    if (!testPoint || typeof testPoint.frequency !== 'number' || !isFinite(testPoint.frequency)) {
      if (logger) logger("Invalid test point encountered, skipping");
      continue;
    }
    
    // Find matching reference point with the same frequency
    const refPoint = referenceDataArray.find(p => 
      p && typeof p.frequency === 'number' && isFinite(p.frequency) &&
      Math.abs(p.frequency - testPoint.frequency) / testPoint.frequency < 0.01);
    
    if (refPoint) {
      matchedData.push({
        frequency: testPoint.frequency,
        test: { real: testPoint.real, imaginary: testPoint.imaginary },
        reference: { real: refPoint.real, imaginary: refPoint.imaginary }
      });
    }
  }

  if (matchedData.length === 0) {
    if (logger) logger("No matching frequency points found");
    return Infinity;
  }

  if (logger) logger(`Matched ${matchedData.length} frequency points for comparison`);

  // const n = matchedData.length; // Not currently used but may be needed for future methods
  let totalError = 0;
  let sumWeights = 0;

  for (const point of matchedData) {
    // Calculate reference magnitude to check for zero division
    const refMag = Math.sqrt(point.reference.real * point.reference.real + point.reference.imaginary * point.reference.imaginary);
    
    // Skip points with zero reference magnitude to avoid division by zero
    if (refMag === 0) {
      if (logger) logger(`Skipping point at ${point.frequency} Hz with zero reference magnitude`);
      continue;
    }
    
    // Calculate base error components
    const realDiff = point.test.real - point.reference.real;
    const imagDiff = point.test.imaginary - point.reference.imaginary;
    const complexMagnitudeError = Math.sqrt(realDiff * realDiff + imagDiff * imagDiff);
    
    // Calculate error based on selected method
    let error: number;
    switch (config.method) {
      case ResnormMethod.MAE:
        error = complexMagnitudeError; // Absolute error in complex magnitude
        break;
      case ResnormMethod.SSR:
        error = complexMagnitudeError * complexMagnitudeError; // Squared error
        break;
      case ResnormMethod.RMSE:
        error = complexMagnitudeError * complexMagnitudeError; // Will be square rooted later
        break;
      default:
        error = complexMagnitudeError; // Default to MAE
    }
    
    // Apply frequency weighting if enabled
    let weight = 1.0;
    if (config.useFrequencyWeighting && point.frequency > 0) {
      weight = Math.pow(point.frequency, -0.5);
    }
    
    totalError += weight * error;
    sumWeights += weight;
    
    if (logger) {
      const weightStr = config.useFrequencyWeighting ? ` (weight: ${weight.toFixed(4)})` : '';
      logger(`Freq: ${point.frequency.toFixed(2)} Hz, ${config.method.toUpperCase()}: ${error.toFixed(4)}${weightStr}`);
    }
  }

  // Calculate base resnorm
  if (sumWeights === 0) return Infinity;
  
  // Calculate final value based on method
  let baseResnorm: number;
  switch (config.method) {
    case ResnormMethod.MAE:
      baseResnorm = totalError / sumWeights; // Mean absolute error
      break;
    case ResnormMethod.SSR:
      baseResnorm = totalError / sumWeights; // Mean squared error
      break;
    case ResnormMethod.RMSE:
      baseResnorm = Math.sqrt(totalError / sumWeights); // Root mean squared error
      break;
    default:
      baseResnorm = totalError / sumWeights; // Default to MAE
  }
  
  // Apply range amplification if enabled
  let finalResnorm = baseResnorm;
  
  if (config.useRangeAmplification && matchedData.length > 1) {
    // Range Amplification: Amplifier = sqrt(log10(ωmax/ωmin))
    const frequencies = matchedData.map(d => d.frequency);
    const ωmax = Math.max(...frequencies);
    const ωmin = Math.min(...frequencies);
    const amplifier = Math.sqrt(Math.log10(ωmax / ωmin));
    
    // Final Resnorm = Resnorm · Amplifier
    finalResnorm = baseResnorm * amplifier;
    
    if (logger) {
      logger(`Range amplification: ωmax=${ωmax.toFixed(2)}, ωmin=${ωmin.toFixed(2)}, amplifier=${amplifier.toFixed(4)}`);
    }
  }
  
  if (logger) {
    logger(`Base resnorm (SSR): ${baseResnorm.toFixed(6)}, Final resnorm: ${finalResnorm.toFixed(6)}`);
  }

  return finalResnorm;
};

/**
 * Legacy calculateResnorm function for backward compatibility
 * @deprecated Use calculateResnormWithConfig instead
 * Now always uses SSR method
 */
export const calculateResnorm = (
  testData: ImpedancePoint[] | ImpedancePoint,
  referenceData: ImpedancePoint[] | number,
  logFunction?: ((message: string) => void) | number,
  frequency?: number,
  useRangeAmplification: boolean = false,
  useFrequencyWeighting: boolean = false,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  _useMaeMethod: boolean = true // Ignored - always uses SSR now
): number => {
  const config: ResnormConfig = {
    method: ResnormMethod.SSR,
    useRangeAmplification,
    useFrequencyWeighting
  };
  
  return calculateResnormWithConfig(testData, referenceData, logFunction, frequency, config);
};

/**
 * Calculate impedance ratio imaginary high-frequency penalty
 * @param Ca Apical capacitance
 * @param Cb Basal capacitance
 * @param target_zr_imag_hf Target impedance ratio imaginary high-frequency value
 * @returns Penalty value
 */
export function impedanceRatioImaginaryHighFreqPenalty(Ca: number, Cb: number, target_zr_imag_hf: number): number {
  // Calculate effective capacitance at high frequency
  const x = Ca === Cb ? Ca : (Ca * Cb) / (Ca - Cb);
  const dx = x - target_zr_imag_hf;
  return dx * dx; // Square of difference as penalty
}

/**
 * Calculate junction resistance ratio penalty
 * @param Ra Apical resistance
 * @param Rb Basal resistance
 * @param Rsh Shunt resistance
 * @param target_jrr Target junction resistance ratio
 * @param use_log10_ratio Whether to use log10 of ratio
 * @returns Penalty value
 */
export function jrrPenalty(Ra: number, Rb: number, Rsh: number, target_jrr: number, use_log10_ratio: boolean): number {
  if (use_log10_ratio) {
    const dx = Math.log10((Ra + Rb) / Rsh) - Math.log10(target_jrr);
    return dx * dx;
  } else {
    const dx = (Ra + Rb) / Rsh - target_jrr;
    return dx * dx;
  }
}

/**
 * Calculate capacitance ratio penalty
 * @param Ca Apical capacitance
 * @param Cb Basal capacitance
 * @param target_cap_ratio Target capacitance ratio
 * @param use_log10_ratio Whether to use log10 of ratio
 * @returns Penalty value
 */
export function capacitanceRatio(Ca: number, Cb: number, target_cap_ratio: number, use_log10_ratio: boolean): number {
  if (use_log10_ratio) {
    const dx = Math.log10(Ca/Cb) - Math.log10(target_cap_ratio);
    return dx * dx;
  } else {
    const dx = Ca/Cb - target_cap_ratio;
    return dx * dx;
  }
}

/**
 * Calculate numerical total capacitance penalty
 * @param w Angular frequencies
 * @param Z Complex impedances
 * @param Ca Apical capacitance
 * @param Cb Basal capacitance
 * @returns Penalty value
 */
export function numericalCtPenalty(w: number[], Z: Complex[], Ca: number, Cb: number): number {
  if (w.length <= 1) return 0;

  // Calculate target Ct using numerical method
  const target_Ct = calculateCt(w, Z.map(z => z.real));
  const modeled_Ct = 1 / (1/Ca + 1/Cb); // Series capacitance formula
  const dx = modeled_Ct - target_Ct;
  return dx * dx;
}

/**
 * Calculate total capacitance from impedance data
 * @param w Angular frequencies
 * @param Z_real Real parts of impedances
 * @returns Calculated total capacitance
 */
function calculateCt(w: number[], Z_real: number[]): number {
  const n = Z_real.length;
  if (n <= 1) return 0;

  // Calculate numerical derivative
  const dZdr = new Array(n-1);
  for (let i = 0; i < n-1; i++) {
    dZdr[i] = (Z_real[i+1] - Z_real[i]) / (w[i+1] - w[i]);
  }

  // Find minimum derivative point
  const minDeriv = Math.min(...dZdr);
  const minDerivIndex = dZdr.indexOf(minDeriv);

  // Calculate Ct using the point of minimum derivative
  // This is based on the fact that at inflection point, imaginary part is maximized
  return -1 / (w[minDerivIndex] * Z_real[minDerivIndex]);
}

/**
 * Ground truth impedance point from impedance_model_comparison.py
 * To be used as a reference for resnorm calculations
 */
export const groundTruthImpedance: ImpedancePoint = {
  real: 24.0196796232508,
  imaginary: -1.15693826250742,
  frequency: 10000,
  magnitude: 24.0475261793571,
  phase: Math.atan2(-1.15693826250742, 24.0196796232508) * (180 / Math.PI)
};

/**
 * Extended ground truth impedance dataset based on the provided frequency values
 * This includes a complete set of impedance measurements across various frequencies
 */
export const groundTruthDataset: ImpedancePoint[] = [
  // Primary data point from user
  {
    real: 24.0196796232508,
    imaginary: -1.15693826250742, // This is -Z'' converted to imaginary
    frequency: 10000,
    magnitude: 24.0475261793571, // Value provided by user
    phase: -2.75759250933683 // Properly store the phase in degrees
  },
  {
    real: 19.0140411028486,
    imaginary: -7.52984606443549,
    frequency: 5000,
    magnitude: Math.sqrt(19.0140411028486**2 + 7.52984606443549**2),
    phase: Math.atan2(-7.52984606443549, 19.0140411028486) * (180 / Math.PI)
  },
  {
    real: 20.7560176016118,
    imaginary: -16.8293998859934,
    frequency: 2025,
    magnitude: Math.sqrt(20.7560176016118**2 + 16.8293998859934**2),
    phase: Math.atan2(-16.8293998859934, 20.7560176016118) * (180 / Math.PI)
  },
  {
    real: 22.3597923401363,
    imaginary: -21.3644829082237,
    frequency: 1575,
    magnitude: Math.sqrt(22.3597923401363**2 + 21.3644829082237**2),
    phase: Math.atan2(-21.3644829082237, 22.3597923401363) * (180 / Math.PI)
  },
  {
    real: 25.4208517108672,
    imaginary: -28.5723160076139,
    frequency: 1125,
    magnitude: Math.sqrt(25.4208517108672**2 + 28.5723160076139**2),
    phase: Math.atan2(-28.5723160076139, 25.4208517108672) * (180 / Math.PI)
  }
];

// TER calculation moved to centralized math/utils.ts
export { calculateTER } from '../math/utils';

/**
 * Calculate TEC (Transepithelial Capacitance)
 */
export const calculateTEC = (params: CircuitParameters): number => {
  // TEC is the series capacitance of Ca and Cb
  return (params.Ca * params.Cb) / (params.Ca + params.Cb);
};
