import { ImpedancePoint, CircuitParameters } from '../types';

/**
 * Format a numeric value with proper unit
 */
export const formatValue = (value: number, unit: string): string => {
  if (Math.abs(value) >= 1000) {
    const exponent = Math.floor(Math.log10(value));
    const base = value / Math.pow(10, exponent);
    return `${base.toFixed(1)}×10${superscript(exponent)} ${unit}`;
  }
  return `${value.toFixed(1)} ${unit}`;
};

/**
 * Convert a number to superscript format
 */
export const superscript = (num: number): string => {
  const digits = num.toString().split('');
  const superscripts: { [key: string]: string } = {
    '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
    '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
    '-': '⁻'
  };
  return digits.map(d => superscripts[d] || d).join('');
};

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
 * Calculate Za (apical impedance)
 */
export const calculateZa = (f: number, parameters: CircuitParameters): { real: number; imaginary: number } => {
  const omega = 2 * Math.PI * f;
  const real = parameters.ra;
  const imaginary = -1 / (omega * parameters.ca);
  return { real, imaginary };
};

/**
 * Calculate Zb (basal impedance)
 */
export const calculateZb = (f: number, parameters: CircuitParameters): { real: number; imaginary: number } => {
  const omega = 2 * Math.PI * f;
  const real = parameters.rb;
  const imaginary = -1 / (omega * parameters.cb);
  return { real, imaginary };
};

/**
 * Calculate total impedance for given frequency and parameters
 */
export const calculateImpedance = (f: number, parameters: CircuitParameters): ImpedancePoint => {
  // Calculate impedance for the given frequency and circuit parameters
  const omega = 2 * Math.PI * f;
  
  // Calculate Za (impedance of Ra-Ca parallel branch)
  const jwRaCa = omega * parameters.ra * parameters.ca;
  const za_real = parameters.ra / (1 + jwRaCa * jwRaCa);
  const za_imag = -jwRaCa * parameters.ra / (1 + jwRaCa * jwRaCa);
  
  // Calculate Zb (impedance of Rb-Cb parallel branch)
  const jwRbCb = omega * parameters.rb * parameters.cb;
  const zb_real = parameters.rb / (1 + jwRbCb * jwRbCb);
  const zb_imag = -jwRbCb * parameters.rb / (1 + jwRbCb * jwRbCb);
  
  // Add RC impedances (Za and Zb are in series)
  const z_series_real = za_real + zb_real;
  const z_series_imag = za_imag + zb_imag;
  
  // Add Rs in series
  const real = parameters.Rs + z_series_real;
  const imaginary = z_series_imag;
  
  // Calculate magnitude and phase for convenience
  const magnitude = Math.sqrt(real * real + imaginary * imaginary);
  const phase = Math.atan2(imaginary, real) * (180 / Math.PI);
  
  // Log debug info for important frequencies
  if (f === 10000 || f === 5000 || f === 2025 || f === 1575 || f === 1125) {
    console.log(`Impedance at ${f}Hz:`, { 
      real: real.toExponential(6), 
      imaginary: imaginary.toExponential(6),
      magnitude: magnitude.toExponential(6),
      phase: phase.toFixed(2)
    });
  }
  
  return {
    real,
    imaginary,
    frequency: f,
    magnitude,
    phase
  };
};

/**
 * Normalize a value linearly between min and max
 */
export const normalizeLinear = (value: number, min: number, max: number): number => {
  return (value - min) / (max - min);
};

/**
 * Normalize a value logarithmically between min and max
 */
export const normalizeLog = (value: number, min: number, max: number): number => {
  return (Math.log10(value) - Math.log10(min)) / (Math.log10(max) - Math.log10(min));
};

/**
 * Calculate Total Epithelial Resistance
 */
export const calculateTER = (parameters: CircuitParameters): number => {
  return parameters.ra + parameters.rb; // TER is sum of Ra and Rb
};

// Export functions from other files
export { calculateResnorm } from './resnorm';
export { calculateImpedanceSpectrum } from './impedance';
export { computeRegressionMesh } from './meshComputation';
export { generateGridPoints } from '../utils'; 