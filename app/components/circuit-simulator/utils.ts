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
 * (Rs in series with parallel Ra-Ca and parallel Rb-Cb)
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
  
  // Sum impedances in series (Rs + Za + Zb)
  const realTotal = params.Rs + realA + realB;
  const imagTotal = imagA + imagB;
  
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
    logFunction(`MATH DETAIL: For series components, Ztotal = Z1 + Z2 + ...`);
    
    const sampleFreqs = [frequencies[0], frequencies[Math.floor(frequencies.length/2)], frequencies[frequencies.length-1]];
    sampleFreqs.forEach(freq => {
      const [realA, imagA] = calculateRCImpedance(referenceParams.Ra, referenceParams.Ca, freq);
      const [realB, imagB] = calculateRCImpedance(referenceParams.Rb, referenceParams.Cb, freq);
      const realTotal = referenceParams.Rs + realA + realB;
      const imagTotal = imagA + imagB;
      
      logFunction(`MATH DETAIL: At f=${freq.toFixed(2)}Hz:`);
      logFunction(`MATH DETAIL:   Za = ${realA.toFixed(4)} ${imagA >= 0 ? '+' : ''}${imagA.toFixed(4)}j Ω`);
      logFunction(`MATH DETAIL:   Zb = ${realB.toFixed(4)} ${imagB >= 0 ? '+' : ''}${imagB.toFixed(4)}j Ω`);
      logFunction(`MATH DETAIL:   Ztotal = Rs + Za + Zb = ${referenceParams.Rs} + (${realA.toFixed(4)} ${imagA >= 0 ? '+' : ''}${imagA.toFixed(4)}j) + (${realB.toFixed(4)} ${imagB >= 0 ? '+' : ''}${imagB.toFixed(4)}j) = ${realTotal.toFixed(4)} ${imagTotal >= 0 ? '+' : ''}${imagTotal.toFixed(4)}j Ω`);
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
export const generateGridPoints = (
  referenceParams: {
    Rs: number;
    Ra: number;
    Ca: number;
    Rb: number;
    Cb: number;
    frequency_range: number[];
  }, 
  paramBounds: {
    Rs: { min: number; max: number };
    Ra: { min: number; max: number };
    Ca: { min: number; max: number };
    Rb: { min: number; max: number };
    Cb: { min: number; max: number };
  },
  pointsPerDim: number = 3,
  logFunction?: (message: string) => void
): Array<{
  parameters: {
    Rs: number;
    Ra: number;
    Ca: number;
    Rb: number;
    Cb: number;
    frequency_range: number[];
  };
  resnorm: number;
  alpha: number;
}> => {
  // Log the generation process
  if (logFunction) {
    logFunction(`MATH: Generating grid points for exploring parameter space`);
    logFunction(`MATH: Reference parameters: Rs=${referenceParams.Rs}, Ra=${referenceParams.Ra}, Ca=${(referenceParams.Ca * 1e6).toFixed(2)}μF, Rb=${referenceParams.Rb}, Cb=${(referenceParams.Cb * 1e6).toFixed(2)}μF`);
    logFunction(`MATH: Parameter bounds: Rs=[${paramBounds.Rs.min}, ${paramBounds.Rs.max}], Ra=[${paramBounds.Ra.min}, ${paramBounds.Ra.max}], Ca=[${(paramBounds.Ca.min * 1e6).toFixed(2)}, ${(paramBounds.Ca.max * 1e6).toFixed(2)}]μF, Rb=[${paramBounds.Rb.min}, ${paramBounds.Rb.max}], Cb=[${(paramBounds.Cb.min * 1e6).toFixed(2)}, ${(paramBounds.Cb.max * 1e6).toFixed(2)}]μF`);
  }

  // Function to generate grid values for a parameter
  const generateParamValues = (
    param: string,
    min: number,
    max: number, 
    refValue: number,
    pointsPerDim: number
  ): number[] => {
    // Create an array of values clustered more densely around the reference
    // but also ensure we cover the entire range to explore the parameter space
    
    // Previously we used a linear grid or weighted grid that favored the reference point too much
    // Now we'll use a mixture of reference-focused and exploration-focused points
    
    const values: number[] = [];
    
    // Add the min and max bounds to ensure full range coverage
    values.push(min);
    values.push(max);
    
    // Add the reference value
    if (refValue >= min && refValue <= max) {
      values.push(refValue);
    }
    
    // Add some points near reference (25% of points)
    const nearRefCount = Math.max(1, Math.floor(pointsPerDim * 0.25));
    const refRangeMin = Math.max(min, refValue * 0.85);
    const refRangeMax = Math.min(max, refValue * 1.15);
    
    for (let i = 0; i < nearRefCount; i++) {
      const t = (i + 1) / (nearRefCount + 1);
      values.push(refRangeMin + t * (refRangeMax - refRangeMin));
    }
    
    // Add points across the full range (75% of points)
    const fullRangeCount = pointsPerDim - nearRefCount - 3; // -3 for min, max, ref
    for (let i = 0; i < fullRangeCount; i++) {
      const t = (i + 1) / (fullRangeCount + 1);
      values.push(min + t * (max - min));
    }
    
    // Sort and deduplicate values
    return [...new Set(values)].sort((a, b) => a - b);
  };

  // Generate parameter values for each dimension
  const rsValues = generateParamValues('Rs', paramBounds.Rs.min, paramBounds.Rs.max, referenceParams.Rs, pointsPerDim);
  const raValues = generateParamValues('Ra', paramBounds.Ra.min, paramBounds.Ra.max, referenceParams.Ra, pointsPerDim);
  const caValues = generateParamValues('Ca', paramBounds.Ca.min, paramBounds.Ca.max, referenceParams.Ca, pointsPerDim);
  const rbValues = generateParamValues('Rb', paramBounds.Rb.min, paramBounds.Rb.max, referenceParams.Rb, pointsPerDim);
  const cbValues = generateParamValues('Cb', paramBounds.Cb.min, paramBounds.Cb.max, referenceParams.Cb, pointsPerDim);

  if (logFunction) {
    logFunction(`MATH: Grid points per parameter: Rs=${rsValues.length}, Ra=${raValues.length}, Ca=${caValues.length}, Rb=${rbValues.length}, Cb=${cbValues.length}`);
    logFunction(`MATH: Total points in grid: ${rsValues.length * raValues.length * caValues.length * rbValues.length * cbValues.length}`);
    
    // Log a subset of the values for each parameter to show distribution
    logFunction(`MATH: Rs values sample: [${rsValues.slice(0, 3).join(', ')}... ${rsValues.slice(-3).join(', ')}]`);
    logFunction(`MATH: Ra values sample: [${raValues.slice(0, 3).join(', ')}... ${raValues.slice(-3).join(', ')}]`);
    logFunction(`MATH: Ca values sample: [${caValues.slice(0, 3).map(v => (v * 1e6).toFixed(2)).join(', ')}... ${caValues.slice(-3).map(v => (v * 1e6).toFixed(2)).join(', ')}]μF`);
  }

  // Generate all grid points
  const gridPoints: Array<{
    parameters: {
      Rs: number;
      Ra: number;
      Ca: number;
      Rb: number;
      Cb: number;
      frequency_range: number[];
    };
    resnorm: number;
    alpha: number;
  }> = [];
  
  // Generate ALL combinations for full parameter space exploration
  // This uses nested loops to create the full Cartesian product of all parameter values
  for (const rs of rsValues) {
    for (const ra of raValues) {
      for (const ca of caValues) {
        for (const rb of rbValues) {
          for (const cb of cbValues) {
            gridPoints.push({
              parameters: {
                ...referenceParams,
                Rs: rs,
                Ra: ra,
                Ca: ca,
                Rb: rb,
                Cb: cb
              },
              resnorm: 0,
              alpha: 1
            });
          }
        }
      }
    }
  }
  
  if (logFunction) {
    logFunction(`MATH: Generated ${gridPoints.length} grid points using full Cartesian product`);
    logFunction(`MATH: Exploring all possible parameter combinations for comprehensive visualization`);
  }
  
  return gridPoints;
}; 