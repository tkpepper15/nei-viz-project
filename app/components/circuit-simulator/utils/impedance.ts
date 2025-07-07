import { Complex } from '../types';
import { CircuitParameters } from '../types/parameters';

export interface ImpedancePoint {
    real: number;
    imag: number;
    frequency: number;
    magnitude: number;
    phase: number;
}

/**
 * Calculate the impedance of a single membrane (apical or basal)
 * Z(ω) = R/(1+jωRC)
 */
export const calculateMembraneImpedance = (
    R: number,
    C: number,
    omega: number
): Complex => {
    // Z = R/(1+jωRC)
    const denom = {
        real: 1,
        imag: omega * R * C
    };
    const denom_mag_squared = denom.real * denom.real + denom.imag * denom.imag;
    
    return {
        real: R * denom.real / denom_mag_squared,
        imag: -R * denom.imag / denom_mag_squared
    };
};

/**
 * Calculate the equivalent impedance of the epithelial model
 * Z_total = (Rs * (Za + Zb)) / (Rs + Za + Zb) where Za(ω) = Ra/(1+jωRaCa) and Zb(ω) = Rb/(1+jωRbCb)
 */
export const calculateEquivalentImpedance = (
    params: CircuitParameters,
    omega: number
): Complex => {
    // Calculate individual membrane impedances using the formula Ra/(1+jωRaCa)
    // For Za
    const Za_denom = {
        real: 1, 
        imag: omega * params.Ra * params.Ca
    };
    const Za_denom_mag_squared = Za_denom.real * Za_denom.real + Za_denom.imag * Za_denom.imag;
    const Za = {
        real: params.Ra * Za_denom.real / Za_denom_mag_squared,
        imag: -params.Ra * Za_denom.imag / Za_denom_mag_squared
    };
    
    // For Zb
    const Zb_denom = {
        real: 1, 
        imag: omega * params.Rb * params.Cb
    };
    const Zb_denom_mag_squared = Zb_denom.real * Zb_denom.real + Zb_denom.imag * Zb_denom.imag;
    const Zb = {
        real: params.Rb * Zb_denom.real / Zb_denom_mag_squared,
        imag: -params.Rb * Zb_denom.imag / Zb_denom_mag_squared
    };

    // Calculate sum of membrane impedances (Za + Zb)
    const Zab = {
        real: Za.real + Zb.real,
        imag: Za.imag + Zb.imag
    };

    // Calculate parallel combination: Z_total = (Rs * (Za + Zb)) / (Rs + Za + Zb)
    // Numerator: Rs * (Za + Zb)
    const numerator = {
        real: params.Rs * Zab.real,
        imag: params.Rs * Zab.imag
    };
    
    // Denominator: Rs + Za + Zb
    const denominator = {
        real: params.Rs + Zab.real,
        imag: Zab.imag
    };
    
    // Complex division: numerator / denominator
    const denom_mag_squared = denominator.real * denominator.real + denominator.imag * denominator.imag;
    
    return {
        real: (numerator.real * denominator.real + numerator.imag * denominator.imag) / denom_mag_squared,
        imag: (numerator.imag * denominator.real - numerator.real * denominator.imag) / denom_mag_squared
    };
};

/**
 * Calculate the complete impedance spectrum for a given frequency range
 */
export const calculateImpedanceSpectrum = (
    params: CircuitParameters,
    frequencies: number[]
): ImpedancePoint[] => {
    return frequencies.map(freq => {
        const omega = 2 * Math.PI * freq;
        const Z = calculateEquivalentImpedance(params, omega);
        
        return {
            frequency: freq,
            real: Z.real,
            imag: Z.imag,
            magnitude: Math.sqrt(Z.real * Z.real + Z.imag * Z.imag),
            phase: Math.atan2(Z.imag, Z.real) * (180 / Math.PI)
        };
    });
};

/**
 * Calculate TER (Transepithelial Resistance)
 * TER = Rs * (Ra + Rb) / (Rs + Ra + Rb)
 */
export const calculateTER = (params: CircuitParameters): number => {
    const numerator = params.Rs * (params.Ra + params.Rb);
    const denominator = params.Rs + params.Ra + params.Rb;
    return numerator / denominator;
};

/**
 * Calculate TEC (Transepithelial Capacitance)
 * TEC = CaCb/(Ca+Cb)
 */
export const calculateTEC = (params: CircuitParameters): number => {
    return (params.Ca * params.Cb) / (params.Ca + params.Cb);
};

/**
 * Calculate Nyquist plot characteristics
 */
export const calculateNyquistCharacteristics = (
    impedanceData: ImpedancePoint[]
): {
    TER: number;
    highFreqIntercept: number;
    semicircleArea: number;
} => {
    // Find low-frequency intercept (TER)
    const lowFreqPoint = impedanceData[impedanceData.length - 1];
    
    // Find high-frequency intercept
    const highFreqPoint = impedanceData[0];
    
    // Calculate semicircle area (related to TEC)
    let area = 0;
    for (let i = 0; i < impedanceData.length - 1; i++) {
        const current = impedanceData[i];
        const next = impedanceData[i + 1];
        area += Math.abs(
            (current.real - highFreqPoint.real) * (next.imag - current.imag) -
            (current.imag - next.imag) * (next.real - current.real)
        ) / 2;
    }
    
    return {
        TER: lowFreqPoint.real,
        highFreqIntercept: highFreqPoint.real,
        semicircleArea: area
    };
};

// Calculate physical resnorm between two parameter sets
export const calculatePhysicalResnorm = (
    testParams: CircuitParameters,
    referenceParams: CircuitParameters,
    frequencies: number[],
    refSpectrum: { freq: number; real: number; imag: number; mag: number; phase: number; }[],
    logFunction?: (message: string) => void
) => {
    // Calculate test spectrum
    const testSpectrum = calculateImpedanceSpectrum(testParams, frequencies);
    
    // Calculate residuals
    const residuals = testSpectrum.map((test, i) => {
        const ref = refSpectrum[i];
        const freq = test.frequency;
        
        // Calculate frequency-dependent weight
        // Higher weight for lower frequencies (more important for physical interpretation)
        const weight = 1 / Math.sqrt(freq);
        
        // Calculate complex residual
        const realResidual = (test.real - ref.real) / ref.real;
        const imagResidual = (test.imag - ref.imag) / ref.imag;
        
        // Calculate magnitude of complex residual
        const residual = Math.sqrt(realResidual * realResidual + imagResidual * imagResidual);
        
        return {
            freq,
            weight,
            residual
        };
    });
    
    // Calculate weighted sum of squared residuals
    let sumSquaredResiduals = 0;
    let weightSum = 0;
    
    residuals.forEach(({ weight, residual }) => {
        sumSquaredResiduals += weight * residual * residual;
        weightSum += weight;
    });
    
    // Log detailed residuals if requested and parameters are close to reference
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

// Calculate resnorm between two impedance spectra
export const calculateResnorm = (reference: ImpedancePoint[], testData: ImpedancePoint[]): number => {
    if (reference.length !== testData.length) {
        throw new Error('Reference and test spectra must have the same length');
    }

    let sumSquaredDiff = 0;
    for (let i = 0; i < reference.length; i++) {
        const ref = reference[i];
        const testPoint = testData[i];
        
        // Calculate complex residual
        const realResidual = (testPoint.real - ref.real) / ref.real;
        const imagResidual = (testPoint.imag - ref.imag) / ref.imag;
        
        // Calculate magnitude of complex residual
        const residual = Math.sqrt(realResidual * realResidual + imagResidual * imagResidual);
        sumSquaredDiff += residual * residual;
    }

    return Math.sqrt(sumSquaredDiff / reference.length);
}; 