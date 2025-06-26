import { BackendMeshPoint } from './types';
import { CircuitParameters } from './types/parameters';

// Utility functions for circuit simulator

/**
 * Generate logarithmically spaced frequencies
 */
export const generateFrequencies = (start: number, end: number, points: number): number[] => {
  const frequencies: number[] = [];
  const logStart = Math.log10(start);
  const logEnd = Math.log10(end);
  const step = (logEnd - logStart) / (points - 1);
  
  for (let i = 0; i < points; i++) {
    frequencies.push(Math.pow(10, logStart + i * step));
  }
  return frequencies;
};

/**
 * Generate logarithmically spaced values between min and max
 */
export const generateLogSpace = (min: number, max: number, count: number): number[] => {
  const result: number[] = [];
  const logMin = Math.log10(min);
  const logMax = Math.log10(max);
  
  for (let i = 0; i < count; i++) {
    // Log interpolation: 10^(logMin + i * (logMax - logMin) / (count - 1))
    const logValue = logMin + i * (logMax - logMin) / (count - 1);
    const value = Math.pow(10, logValue);
    result.push(value);
  }
  
  return result;
};

/**
 * Generate linear spaced values between min and max
 */
export const generateLinSpace = (min: number, max: number, count: number): number[] => {
  const result: number[] = [];
  for (let i = 0; i < count; i++) {
    // Linear interpolation: min + i * (max - min) / (count - 1)
    const value = min + i * (max - min) / (count - 1);
    result.push(value);
  }
  return result;
};

/**
 * Calculate complex impedance of an RC element at a given frequency
 * Returns [real part, imaginary part]
 */
export const calculateRCImpedance = (r: number, c: number, freq: number): [number, number] => {
  const omega = 2 * Math.PI * freq; // Angular frequency
  
  // For a parallel RC circuit: Z = R / (1 + jωRC)
  // Separate into real and imaginary parts by multiplying numerator and denominator by (1 - jωRC)
  
  if (c <= 0 || freq <= 0) {
    // Handle edge cases to prevent NaN or division by zero
    return [r, 0]; // Pure resistance if no capacitance
  }
  
  const wRC = omega * r * c;
  const denominator = 1 + wRC * wRC;
  
  // Real part: R / (1 + (ωRC)²)
  const realPart = r / denominator;
  
  // Imaginary part: -ωR²C / (1 + (ωRC)²)
  const imagPart = -wRC * r / denominator;
  
  return [realPart, imagPart];
};

/**
 * Calculate the total impedance of the Randles circuit model 
 * Z_total = (Rs * (Za + Zb)) / (Rs + Za + Zb) where Za = Ra/(1+jωRaCa) and Zb = Rb/(1+jωRbCb)
 * Returns [real part, imaginary part]
 */
export const calculateTotalImpedance = (params: {
  Rs: number;
  Ra: number;
  Ca: number;
  Rb: number;
  Cb: number;
}, freq: number): [number, number] => {
  // Calculate parallel Ra-Ca impedance using Za = Ra/(1+jωRaCa)
  const [realA, imagA] = calculateRCImpedance(params.Ra, params.Ca, freq);
  
  // Calculate parallel Rb-Cb impedance using Zb = Rb/(1+jωRbCb)
  const [realB, imagB] = calculateRCImpedance(params.Rb, params.Cb, freq);
  
  // Calculate sum of membrane impedances (Za + Zb)
  const Zab_real = realA + realB;
  const Zab_imag = imagA + imagB;
  
  // Calculate parallel combination: Z_total = (Rs * (Za + Zb)) / (Rs + Za + Zb)
  // Numerator: Rs * (Za + Zb)
  const num_real = params.Rs * Zab_real;
  const num_imag = params.Rs * Zab_imag;
  
  // Denominator: Rs + Za + Zb
  const denom_real = params.Rs + Zab_real;
  const denom_imag = Zab_imag;
  
  // Complex division: (num_real + j*num_imag) / (denom_real + j*denom_imag)
  const denom_mag_squared = denom_real * denom_real + denom_imag * denom_imag;
  
  const realTotal = (num_real * denom_real + num_imag * denom_imag) / denom_mag_squared;
  const imagTotal = (num_imag * denom_real - num_real * denom_imag) / denom_mag_squared;
  
  return [realTotal, imagTotal];
};

/**
 * Calculate the impedance spectrum across a frequency range
 */
export const calculateImpedanceSpectrum = (params: {
  Rs: number;
  Ra: number;
  Ca: number;
  Rb: number;
  Cb: number;
}, frequencies: number[]): Array<{freq: number; real: number; imag: number; mag: number; phase: number}> => {
  return frequencies.map(freq => {
    const [real, imag] = calculateTotalImpedance(params, freq);
    const mag = Math.sqrt(real * real + imag * imag); // Magnitude |Z|
    const phase = Math.atan2(imag, real) * (180 / Math.PI); // Phase in degrees
    
    return { freq, real, imag, mag, phase };
  });
};

/**
 * Calculate physically meaningful resnorm based on impedance spectrum differences
 */
export const calculatePhysicalResnorm = (
  testParams: {
    Rs: number;
    Ra: number;
    Ca: number;
    Rb: number;
    Cb: number;
  },
  referenceParams: {
    Rs: number;
    Ra: number;
    Ca: number;
    Rb: number;
    Cb: number;
  },
  frequencies: number[],
  preCalculatedRefSpectrum?: Array<{freq: number; real: number; imag: number; mag: number; phase: number}>,
  logFunction?: (message: string) => void
): number => {
  // Calculate impedance spectrum for test parameters
  const testSpectrum = calculateImpedanceSpectrum(testParams, frequencies);
  
  // Use pre-calculated spectrum if provided, otherwise calculate it
  const refSpectrum = preCalculatedRefSpectrum || calculateImpedanceSpectrum(referenceParams, frequencies);
  
  if (logFunction && testParams.Rs === referenceParams.Rs && 
      testParams.Ra === referenceParams.Ra && testParams.Ca === referenceParams.Ca &&
      testParams.Rb === referenceParams.Rb && testParams.Cb === referenceParams.Cb) {
    // If this is the reference point, log detailed math for educational purposes
    logFunction(`MATH DETAIL: Impedance calculation for reference parameters:`);
    logFunction(`MATH DETAIL: For RC circuit, Z = R/(1+jωRC) where ω = 2πf`);
    logFunction(`MATH DETAIL: For parallel combination, Ztotal = (Rs * (Za + Zb)) / (Rs + Za + Zb)`);
    
    const sampleFreqs = [frequencies[0], frequencies[Math.floor(frequencies.length/2)], frequencies[frequencies.length-1]];
    sampleFreqs.forEach(freq => {
      const [realA, imagA] = calculateRCImpedance(referenceParams.Ra, referenceParams.Ca, freq);
      const [realB, imagB] = calculateRCImpedance(referenceParams.Rb, referenceParams.Cb, freq);
      const [realTotal, imagTotal] = calculateTotalImpedance(referenceParams, freq);
      
      logFunction(`MATH DETAIL: At f=${freq.toFixed(2)}Hz:`);
      logFunction(`MATH DETAIL:   Za = ${realA.toFixed(4)} ${imagA >= 0 ? '+' : ''}${imagA.toFixed(4)}j Ω`);
      logFunction(`MATH DETAIL:   Zb = ${realB.toFixed(4)} ${imagB >= 0 ? '+' : ''}${imagB.toFixed(4)}j Ω`);
      logFunction(`MATH DETAIL:   Ztotal = (Rs * (Za + Zb)) / (Rs + Za + Zb) = ${realTotal.toFixed(4)} ${imagTotal >= 0 ? '+' : ''}${imagTotal.toFixed(4)}j Ω`);
    });
  }
  
  // Calculate weighted normalized residuals across frequency spectrum
  // Weight low frequencies higher as they're more important for cell characterization
  let sumSquaredResiduals = 0;
  let weightSum = 0;
  
  const residuals: Array<{freq: number; weight: number; residual: number}> = [];
  
  for (let i = 0; i < frequencies.length; i++) {
    const freq = frequencies[i];
    const logFreq = Math.log10(freq);
    
    // Compute weight as 1/log(f) to give more weight to lower frequencies
    // This is biophysically relevant as lower frequencies probe deeper cell properties
    const weight = 1 / Math.max(1, logFreq + 1);
    weightSum += weight;
    
    // Calculate complex impedance differences for real and imaginary parts
    const realDiff = Math.abs(testSpectrum[i].real - refSpectrum[i].real) / Math.max(1, Math.abs(refSpectrum[i].real));
    const imagDiff = Math.abs(testSpectrum[i].imag - refSpectrum[i].imag) / Math.max(1, Math.abs(refSpectrum[i].imag));
    
    // Use Euclidean distance in complex plane (normalized)
    const euclideanDist = Math.sqrt(realDiff*realDiff + imagDiff*imagDiff);
    
    // Also consider magnitude and phase differences
    const magnitudeDiff = Math.abs(testSpectrum[i].mag - refSpectrum[i].mag) / refSpectrum[i].mag;
    const phaseDiff = Math.abs(testSpectrum[i].phase - refSpectrum[i].phase) / 90; // Normalize to 90 degrees
    
    // Combine all metrics with appropriate weights reflecting their importance in EIS analysis
    // Standard practice in EIS analysis weights magnitude more heavily than phase
    const combinedDiff = (0.4 * euclideanDist) + (0.4 * magnitudeDiff) + (0.2 * phaseDiff);
    
    // Add weighted squared residual (standard form for resnorm calculation)
    const residual = weight * combinedDiff * combinedDiff;
    sumSquaredResiduals += residual;
    
    residuals.push({ freq, weight, residual });
  }
  
  // Log detailed residual calculations for specific test points if requested
  if (logFunction && 
      (Math.abs(testParams.Rs - referenceParams.Rs) < 0.01 * referenceParams.Rs ||
       Math.abs(testParams.Ra - referenceParams.Ra) < 0.01 * referenceParams.Ra ||
       Math.abs(testParams.Ca - referenceParams.Ca) < 0.01 * referenceParams.Ca ||
       Math.abs(testParams.Rb - referenceParams.Rb) < 0.01 * referenceParams.Rb ||
       Math.abs(testParams.Cb - referenceParams.Cb) < 0.01 * referenceParams.Cb)) {
    
    // Log representative residuals (low, mid, high freq)
    const lowIdx = 0;
    const midIdx = Math.floor(residuals.length / 2);
    const highIdx = residuals.length - 1;
    
    logFunction(`MATH DETAIL: Residual calculation for test point close to reference:`);
    logFunction(`MATH DETAIL: Rs=${testParams.Rs.toFixed(2)}, Ra=${testParams.Ra.toFixed(0)}, Ca=${(testParams.Ca*1e6).toFixed(2)}μF, Rb=${testParams.Rb.toFixed(0)}, Cb=${(testParams.Cb*1e6).toFixed(2)}μF`);
    logFunction(`MATH DETAIL: Low frequency (${residuals[lowIdx].freq.toFixed(2)}Hz): weight=${residuals[lowIdx].weight.toFixed(4)}, residual=${residuals[lowIdx].residual.toExponential(4)}`);
    logFunction(`MATH DETAIL: Mid frequency (${residuals[midIdx].freq.toFixed(2)}Hz): weight=${residuals[midIdx].weight.toFixed(4)}, residual=${residuals[midIdx].residual.toExponential(4)}`);
    logFunction(`MATH DETAIL: High frequency (${residuals[highIdx].freq.toFixed(2)}Hz): weight=${residuals[highIdx].weight.toFixed(4)}, residual=${residuals[highIdx].residual.toExponential(4)}`);
  }
  
  // Normalize by sum of weights - this is standard practice in weighted least squares
  const normalizedResnorm = Math.sqrt(sumSquaredResiduals / weightSum);
  
  return normalizedResnorm;
};

/**
 * Generate grid points for parameter space exploration
 */
export function generateGridPoints(
  referenceParams: CircuitParameters,
  paramBounds: {
    Rs: { min: number; max: number };
    Ra: { min: number; max: number };
    Ca: { min: number; max: number };
    Rb: { min: number; max: number };
    Cb: { min: number; max: number };
  },
  pointsPerDim: number,
  logFunction?: (message: string) => void
): BackendMeshPoint[] {
  // Generate logarithmically spaced values for each parameter
  const generateLogValues = (min: number, max: number, count: number): number[] => {
    const values: number[] = [];
    const logMin = Math.log10(min);
    const logMax = Math.log10(max);
    const step = (logMax - logMin) / (count - 1);
    
    for (let i = 0; i < count; i++) {
      const logVal = logMin + i * step;
      const value = Math.pow(10, logVal);
      values.push(value);
    }
    
    return values;
  };

  // Generate values for each parameter
  const rsValues = generateLogValues(paramBounds.Rs.min, paramBounds.Rs.max, pointsPerDim);
  const raValues = generateLogValues(paramBounds.Ra.min, paramBounds.Ra.max, pointsPerDim);
  const rbValues = generateLogValues(paramBounds.Rb.min, paramBounds.Rb.max, pointsPerDim);
  const caValues = generateLogValues(paramBounds.Ca.min, paramBounds.Ca.max, pointsPerDim);
  const cbValues = generateLogValues(paramBounds.Cb.min, paramBounds.Cb.max, pointsPerDim);

  if (logFunction) {
    logFunction(`Parameter Value Distribution`);
    logFunction(`Rs (Ω):\n[${rsValues.map(v => v.toFixed(1))}]\nCount: ${rsValues.length}`);
    logFunction(`Ra (Ω):\n[${raValues.map(v => v.toFixed(1))}]\nCount: ${raValues.length}`);
    logFunction(`Rb (Ω):\n[${rbValues.map(v => v.toFixed(1))}]\nCount: ${rbValues.length}`);
    logFunction(`Ca (µF):\n[${caValues.map(v => (v * 1e6).toFixed(2))}]\nCount: ${caValues.length}`);
    logFunction(`Cb (µF):\n[${cbValues.map(v => (v * 1e6).toFixed(2))}]\nCount: ${cbValues.length}`);
  }

  // Generate all combinations
  const gridPoints: BackendMeshPoint[] = [];
  
  for (const Rs of rsValues) {
    for (const Ra of raValues) {
      for (const Ca of caValues) {
        for (const Rb of rbValues) {
          for (const Cb of cbValues) {
            gridPoints.push({
              parameters: {
                Rs,
                Ra,
                Ca,  // Keep in Farads
                Rb,
                Cb,  // Keep in Farads
                frequency_range: referenceParams.frequency_range
              },
              spectrum: [],  // Will be filled later
              resnorm: 0     // Will be calculated later
            });
          }
        }
      }
    }
  }

  return gridPoints;
} 