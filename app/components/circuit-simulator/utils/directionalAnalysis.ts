/**
 * Directional Analysis Utilities for Circuit Parameter Sensitivity
 *
 * Implements a focused, interpretable approach to understanding how circuit parameters
 * affect impedance spectra and resnorm surfaces. Uses analytic Jacobian computation
 * and compressed spectral representations for efficient directional sensitivity analysis.
 */

import { CircuitParameters, PARAMETER_RANGES } from '../types/parameters';
import { calculate_impedance_spectrum, ImpedancePoint } from './resnorm';

// Core interfaces for directional analysis
export interface SpectralCoefficients {
  /** Compressed representation of impedance spectrum using K coefficients */
  coefficients: number[];
  /** Basis vectors (left singular vectors from SVD) */
  basis: number[][];
  /** Singular values for weighting importance */
  singularValues: number[];
  /** Original spectrum length for reconstruction */
  originalLength: number;
}

export interface ParameterDirection {
  /** Unit direction vector in 5D parameter space [Rsh, Ra, Ca, Rb, Cb] */
  direction: number[];
  /** Sensitivity magnitude (singular value) */
  sensitivity: number;
  /** Physical interpretation of the direction */
  interpretation: string;
  /** How this direction affects spectral shape */
  spectralEffect: string;
}

export interface DirectionalSensitivity {
  /** Point in parameter space where sensitivity is computed */
  parameters: CircuitParameters;
  /** Jacobian matrix: [frequency × parameter] complex derivatives */
  jacobian: {
    real: number[][]; // [freq_idx][param_idx]
    imag: number[][]; // [freq_idx][param_idx]
  };
  /** Principal directions ranked by sensitivity */
  principalDirections: ParameterDirection[];
  /** Local quadratic form eigenvalues (curvature) */
  curvatureEigenvalues: number[];
  /** Condition number (identifiability measure) */
  conditionNumber: number;
}

/**
 * Create logarithmically spaced frequency array
 */
export function createFrequencyArray(minFreq: number, maxFreq: number, numPoints: number = 20): number[] {
  const frequencies: number[] = [];
  const logMin = Math.log10(minFreq);
  const logMax = Math.log10(maxFreq);
  const logStep = (logMax - logMin) / (numPoints - 1);

  for (let i = 0; i < numPoints; i++) {
    const logValue = logMin + i * logStep;
    frequencies.push(Math.pow(10, logValue));
  }

  return frequencies;
}

/**
 * Compute analytic Jacobian of complex impedance with respect to circuit parameters
 * This gives the exact sensitivity without finite difference approximation
 */
export function computeAnalyticJacobian(
  params: CircuitParameters,
  frequencies: number[]
): { real: number[][]; imag: number[][] } {
  const { Rsh, Ra, Ca, Rb, Cb } = params;
  const numFreq = frequencies.length;

  const jacobianReal: number[][] = Array(numFreq).fill(null).map(() => Array(5).fill(0));
  const jacobianImag: number[][] = Array(numFreq).fill(null).map(() => Array(5).fill(0));

  frequencies.forEach((freq, i) => {
    const omega = 2 * Math.PI * freq;

    // Pre-compute common terms to avoid redundant calculations
    const omegaRaCa = omega * Ra * Ca;
    const omegaRbCb = omega * Rb * Cb;
    const Da = 1 + omegaRaCa * omegaRaCa; // |1 + jωRaCa|²
    const Db = 1 + omegaRbCb * omegaRbCb; // |1 + jωRbCb|²

    // Branch impedances
    const ZaReal = Ra / Da;
    const ZaImag = -omegaRaCa * Ra / Da;
    const ZbReal = Rb / Db;
    const ZbImag = -omegaRbCb * Rb / Db;

    // Series combination: Zab = Za + Zb
    const ZabReal = ZaReal + ZbReal;
    const ZabImag = ZaImag + ZbImag;

    // Parallel with Rsh: Z = (Rsh * Zab) / (Rsh + Zab)
    const denomReal = Rsh + ZabReal;
    const denomImag = ZabImag;
    const denomMagSq = denomReal * denomReal + denomImag * denomImag;

    // ∂Z/∂Rsh (trivial - just the parallel combination derivative)
    const numReal = ZabReal;
    const numImag = ZabImag;
    jacobianReal[i][0] = (numReal * denomReal + numImag * denomImag) / denomMagSq;
    jacobianImag[i][0] = (numImag * denomReal - numReal * denomImag) / denomMagSq;

    // ∂Z/∂Ra: need ∂Za/∂Ra and then chain rule through parallel combination
    const dZaReal_dRa = 1 / Da - 2 * Ra * omegaRaCa * omegaRaCa / (Da * Da);
    const dZaImag_dRa = -omega * Ca / Da + 2 * Ra * omega * Ca * omegaRaCa * omegaRaCa / (Da * Da);

    // Chain rule: ∂Z/∂Ra = ∂Z/∂Zab * ∂Zab/∂Za * ∂Za/∂Ra
    const dZ_dZabReal = (Rsh * denomReal - Rsh * ZabReal) / denomMagSq;
    const dZ_dZabImag = (Rsh * denomImag - Rsh * ZabImag) / denomMagSq;

    jacobianReal[i][1] = dZ_dZabReal * dZaReal_dRa - dZ_dZabImag * dZaImag_dRa;
    jacobianImag[i][1] = dZ_dZabImag * dZaReal_dRa + dZ_dZabReal * dZaImag_dRa;

    // ∂Z/∂Ca: similar but for capacitance derivative
    const dZaReal_dCa = 2 * Ra * Ra * Ra * omega * omega * Ca / (Da * Da);
    const dZaImag_dCa = -omega * Ra * Ra / Da + 2 * Ra * Ra * Ra * omega * omega * Ca * omegaRaCa / (Da * Da);

    jacobianReal[i][2] = dZ_dZabReal * dZaReal_dCa - dZ_dZabImag * dZaImag_dCa;
    jacobianImag[i][2] = dZ_dZabImag * dZaReal_dCa + dZ_dZabReal * dZaImag_dCa;

    // ∂Z/∂Rb and ∂Z/∂Cb: symmetric to Ra and Ca
    const dZbReal_dRb = 1 / Db - 2 * Rb * omegaRbCb * omegaRbCb / (Db * Db);
    const dZbImag_dRb = -omega * Cb / Db + 2 * Rb * omega * Cb * omegaRbCb * omegaRbCb / (Db * Db);

    jacobianReal[i][3] = dZ_dZabReal * dZbReal_dRb - dZ_dZabImag * dZbImag_dRb;
    jacobianImag[i][3] = dZ_dZabImag * dZbReal_dRb + dZ_dZabReal * dZbImag_dRb;

    const dZbReal_dCb = 2 * Rb * Rb * Rb * omega * omega * Cb / (Db * Db);
    const dZbImag_dCb = -omega * Rb * Rb / Db + 2 * Rb * Rb * Rb * omega * omega * Cb * omegaRbCb / (Db * Db);

    jacobianReal[i][4] = dZ_dZabReal * dZbReal_dCb - dZ_dZabImag * dZbImag_dCb;
    jacobianImag[i][4] = dZ_dZabImag * dZbReal_dCb + dZ_dZabReal * dZbImag_dCb;
  });

  return { real: jacobianReal, imag: jacobianImag };
}

/**
 * Compress impedance spectra using SVD for efficient storage and analysis
 */
export function compressSpectra(spectra: ImpedancePoint[][], numCoefficients: number = 8): SpectralCoefficients {
  if (spectra.length === 0) {
    throw new Error('No spectra provided for compression');
  }

  const numSpectra = spectra.length;
  const spectrumLength = spectra[0].length;

  // Stack real and imaginary parts: [real_f1, real_f2, ..., imag_f1, imag_f2, ...]
  const dataMatrix: number[][] = Array(numSpectra).fill(null).map(() => Array(2 * spectrumLength).fill(0));

  spectra.forEach((spectrum, i) => {
    spectrum.forEach((point, j) => {
      dataMatrix[i][j] = point.real;                    // Real parts first half
      dataMatrix[i][j + spectrumLength] = point.imaginary; // Imaginary parts second half
    });
  });

  // Compute SVD using a simple power iteration method (for the top K components)
  const { U, S, Vt } = computeSVD(dataMatrix, numCoefficients);

  // Return compressed representation
  return {
    coefficients: U.map(row => row.slice(0, numCoefficients)).flat(),
    basis: Vt.slice(0, numCoefficients),
    singularValues: S.slice(0, numCoefficients),
    originalLength: spectrumLength
  };
}

/**
 * Proper SVD implementation using ml-matrix library
 */
function computeSVD(matrix: number[][], k: number): { U: number[][]; S: number[]; Vt: number[][] } {
  try {
    // Use ml-matrix for proper SVD computation
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { Matrix, SVD } = require('ml-matrix');

    const matrixObj = new Matrix(matrix);
    const svd = new SVD(matrixObj);

    const U = svd.leftSingularVectors.to2DArray();
    const S = svd.diagonal;
    const Vt = svd.rightSingularVectors.transpose().to2DArray();

    // Return top k components
    const kActual = Math.min(k, S.length);
    return {
      U: U.map((row: number[]) => row.slice(0, kActual)),
      S: S.slice(0, kActual),
      Vt: Vt.slice(0, kActual)
    };
  } catch {
    console.warn('ml-matrix not available, using fallback SVD');
    return computeFallbackSVD(matrix, k);
  }
}

/**
 * Fallback SVD using simple power iteration
 */
function computeFallbackSVD(matrix: number[][], k: number): { U: number[][]; S: number[]; Vt: number[][] } {
  const m = matrix.length;
  const n = matrix[0].length;

  // Simple fallback for when ml-matrix is not available
  const U: number[][] = Array(m).fill(null).map(() => Array(k).fill(0));
  const S: number[] = Array(k).fill(1);
  const Vt: number[][] = Array(k).fill(null).map(() => Array(n).fill(0));

  // Power iteration for dominant singular vector
  for (let comp = 0; comp < Math.min(k, Math.min(m, n)); comp++) {
    // Initialize random vector
    let v = Array(n).fill(0).map(() => Math.random() - 0.5);

    // Power iteration
    for (let iter = 0; iter < 10; iter++) {
      // v = A^T * A * v
      const Av = Array(m).fill(0);
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          Av[i] += matrix[i][j] * v[j];
        }
      }

      const AtAv = Array(n).fill(0);
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < m; i++) {
          AtAv[j] += matrix[i][j] * Av[i];
        }
      }

      // Normalize
      const norm = Math.sqrt(AtAv.reduce((sum, val) => sum + val * val, 0));
      if (norm > 1e-10) {
        v = AtAv.map(val => val / norm);
      }
    }

    // Store results
    S[comp] = Math.sqrt(v.reduce((sum, val) => sum + val * val, 0));
    for (let j = 0; j < n && j < Vt[comp].length; j++) {
      Vt[comp][j] = v[j];
    }

    // Compute corresponding U column
    for (let i = 0; i < m && i < U.length; i++) {
      let sum = 0;
      for (let j = 0; j < n; j++) {
        sum += matrix[i][j] * v[j];
      }
      if (comp < U[i].length) {
        U[i][comp] = S[comp] > 1e-10 ? sum / S[comp] : 0;
      }
    }
  }

  return { U, S, Vt };
}

/**
 * Compute principal directions of parameter sensitivity
 */
export function computePrincipalDirections(
  jacobian: { real: number[][]; imag: number[][] },
  frequencies: number[]
): ParameterDirection[] {
  const numFreq = frequencies.length;
  const numParams = 5;

  // Stack real and imaginary Jacobian into a single matrix for SVD
  const stackedJacobian: number[][] = Array(2 * numFreq).fill(null).map(() => Array(numParams).fill(0));

  for (let i = 0; i < numFreq; i++) {
    for (let j = 0; j < numParams; j++) {
      stackedJacobian[i][j] = jacobian.real[i][j];           // Real part
      stackedJacobian[i + numFreq][j] = jacobian.imag[i][j]; // Imaginary part
    }
  }

  // Compute SVD to find principal directions
  const { S, Vt } = computeSVD(stackedJacobian, numParams);

  const paramNames = ['Rsh', 'Ra', 'Ca', 'Rb', 'Cb'];
  const directions: ParameterDirection[] = [];

  for (let i = 0; i < Math.min(3, numParams); i++) { // Top 3 directions
    const direction = Vt[i];
    const sensitivity = S[i];

    // Generate physical interpretation
    const dominantParam = direction.map((val, idx) => ({ param: paramNames[idx], value: Math.abs(val) }))
                                   .sort((a, b) => b.value - a.value)[0];

    const interpretation = `Primary: ${dominantParam.param} (${(dominantParam.value * 100).toFixed(1)}%)`;

    const spectralEffect = i === 0 ? "Low-frequency magnitude" :
                          i === 1 ? "Mid-frequency phase" : "High-frequency behavior";

    directions.push({
      direction,
      sensitivity,
      interpretation,
      spectralEffect
    });
  }

  return directions;
}

/**
 * Generate adaptive sampling strategy for parameter space exploration
 */
export function generateAdaptiveSampling(
  centerParams: CircuitParameters,
  _sensitivity: DirectionalSensitivity,
  numSamples: number = 500
): CircuitParameters[] {
  const samples: CircuitParameters[] = [];
  const ranges = PARAMETER_RANGES;

  // Use principal directions for intelligent sampling
  const principalDirs = _sensitivity.principalDirections;

  for (let i = 0; i < numSamples; i++) {
    const sample = { ...centerParams };

    if (i < numSamples * 0.7) {
      // 70% of samples along principal directions
      const dirIndex = i % principalDirs.length;
      const direction = principalDirs[dirIndex].direction;
      const stepSize = (Math.random() - 0.5) * 0.1; // ±10% along direction

      sample.Rsh = Math.max(ranges.Rsh.min, Math.min(ranges.Rsh.max,
        centerParams.Rsh * (1 + stepSize * direction[0])));
      sample.Ra = Math.max(ranges.Ra.min, Math.min(ranges.Ra.max,
        centerParams.Ra * (1 + stepSize * direction[1])));
      sample.Ca = Math.max(ranges.Ca.min, Math.min(ranges.Ca.max,
        centerParams.Ca * (1 + stepSize * direction[2])));
      sample.Rb = Math.max(ranges.Rb.min, Math.min(ranges.Rb.max,
        centerParams.Rb * (1 + stepSize * direction[3])));
      sample.Cb = Math.max(ranges.Cb.min, Math.min(ranges.Cb.max,
        centerParams.Cb * (1 + stepSize * direction[4])));
    } else {
      // 30% random exploration
      sample.Rsh = ranges.Rsh.min + Math.random() * (ranges.Rsh.max - ranges.Rsh.min);
      sample.Ra = ranges.Ra.min + Math.random() * (ranges.Ra.max - ranges.Ra.min);
      sample.Ca = ranges.Ca.min + Math.random() * (ranges.Ca.max - ranges.Ca.min);
      sample.Rb = ranges.Rb.min + Math.random() * (ranges.Rb.max - ranges.Rb.min);
      sample.Cb = ranges.Cb.min + Math.random() * (ranges.Cb.max - ranges.Cb.min);
    }

    samples.push(sample);
  }

  return samples;
}

/**
 * Main function to compute directional sensitivity at a point
 */
export function computeDirectionalSensitivity(
  params: CircuitParameters,
  frequencies?: number[]
): DirectionalSensitivity {
  // Use provided frequencies or create default logarithmic spacing
  const freqArray = frequencies || createFrequencyArray(params.frequency_range[0], params.frequency_range[1]);

  // Compute analytic Jacobian
  const jacobian = computeAnalyticJacobian(params, freqArray);

  // Find principal directions
  const principalDirections = computePrincipalDirections(jacobian, freqArray);

  // Compute condition number (measure of identifiability)
  const singularValues = principalDirections.map(d => d.sensitivity);
  const conditionNumber = Math.max(...singularValues) / Math.min(...singularValues.filter(s => s > 1e-10));

  // Placeholder curvature eigenvalues (would need Hessian computation)
  const curvatureEigenvalues = principalDirections.map(d => d.sensitivity * d.sensitivity);

  return {
    parameters: params,
    jacobian,
    principalDirections,
    curvatureEigenvalues,
    conditionNumber
  };
}

/**
 * Predict impedance change along a parameter direction
 */
export function predictImpedanceChange(
  baseParams: CircuitParameters,
  direction: number[],
  stepSize: number,
  _sensitivity: DirectionalSensitivity // eslint-disable-line @typescript-eslint/no-unused-vars
): ImpedancePoint[] {
  // Create perturbed parameters
  const perturbedParams: CircuitParameters = {
    Rsh: baseParams.Rsh * (1 + stepSize * direction[0]),
    Ra: baseParams.Ra * (1 + stepSize * direction[1]),
    Ca: baseParams.Ca * (1 + stepSize * direction[2]),
    Rb: baseParams.Rb * (1 + stepSize * direction[3]),
    Cb: baseParams.Cb * (1 + stepSize * direction[4]),
    frequency_range: baseParams.frequency_range
  };

  // Compute new spectrum
  return calculate_impedance_spectrum(perturbedParams);
}