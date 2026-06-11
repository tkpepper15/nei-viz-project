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
 * Calculate the total impedance
 * Z_total = (Rsh * (Za + Zb)) / (Rsh + Za + Zb) where Za(ω) = Ra/(1+jωRaCa) and Zb(ω) = Rb/(1+jωRbCb)
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

    // Calculate parallel combination: Z_total = (Rsh * (Za + Zb)) / (Rsh + Za + Zb)
    // Numerator: Rsh * (Za + Zb)
    const numerator = {
        real: params.Rsh * Zab.real,
        imag: params.Rsh * Zab.imag
    };
    
    // Denominator: Rsh + Za + Zb
    const denominator = {
        real: params.Rsh + Zab.real,
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

// TER calculation moved to centralized math/utils.ts
export { calculateTER } from '../math/utils';

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
        (Math.abs(testParams.Rsh - referenceParams.Rsh) < 0.01 * referenceParams.Rsh ||
         Math.abs(testParams.Ra - referenceParams.Ra) < 0.01 * referenceParams.Ra ||
         Math.abs(testParams.Ca - referenceParams.Ca) < 0.01 * referenceParams.Ca ||
         Math.abs(testParams.Rb - referenceParams.Rb) < 0.01 * referenceParams.Rb ||
         Math.abs(testParams.Cb - referenceParams.Cb) < 0.01 * referenceParams.Cb)) {
        
        // Log representative residuals (low, mid, high freq)
        const lowIdx = 0;
        const midIdx = Math.floor(residuals.length / 2);
        const highIdx = residuals.length - 1;
        
        logFunction(`MATH DETAIL: Residual calculation for test point close to reference:`);
        logFunction(`MATH DETAIL: Rsh=${testParams.Rsh.toFixed(2)}, Ra=${testParams.Ra.toFixed(0)}, Ca=${(testParams.Ca*1e6).toFixed(2)}μF, Rb=${testParams.Rb.toFixed(0)}, Cb=${(testParams.Cb*1e6).toFixed(2)}μF`);
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

// =============================================================================
// BERTRAND SYSTEM THEORY COEFFICIENTS
// =============================================================================
// From: Bertrand et al. (1998) Biophysical Journal 75:2743-2756
// "System for Dynamic Measurements of Membrane Capacitance in Intact
// Epithelial Monolayers"
//
// These three unique coefficients fully describe the monolayer impedance:
// Z(s) = (N1*s + N0) / (s^2 + D1*s + N0/Rm)
// =============================================================================

export interface BertrandCoefficients {
    N1: number;  // Inverse total capacitance (1/F) - HIGHLY IDENTIFIABLE
    N0: number;  // Mixed conductance-capacitance (1/(Ohm*F^2)) - POORLY IDENTIFIABLE
    D1: number;  // Damping coefficient (1/s = Hz) - MODERATELY IDENTIFIABLE
    Rm: number;  // TER for reference (Ohm)
}

/**
 * Calculate Bertrand coefficient N1 (Equation 4a from Bertrand et al. 1998)
 *
 * N1 = 1/Ca + 1/Cb = 1/TEC
 *
 * This coefficient represents the inverse of total monolayer capacitance
 * (series combination of apical and basolateral capacitances).
 *
 * IDENTIFIABILITY: Highly identifiable (~1% precision at 99% confidence)
 * Monte Carlo simulations show this coefficient can be estimated with
 * excellent precision across all experimental conditions.
 */
export const calculateBertrandN1 = (params: CircuitParameters): number => {
    return 1.0 / params.Ca + 1.0 / params.Cb;
};

/**
 * Calculate Bertrand coefficient N0 (Equation 4b from Bertrand et al. 1998)
 *
 * N0 = (1/Ra + 1/Rb) / (Ca*Cb)
 *
 * This coefficient involves both conductances (1/R) and capacitances.
 *
 * IDENTIFIABILITY: Poorly identifiable (~50% precision under basal conditions)
 * N0 is strongly correlated with D1 (correlation approaches unity).
 * Precision improves when membrane conductances increase.
 */
export const calculateBertrandN0 = (params: CircuitParameters): number => {
    const conductanceSum = 1.0 / params.Ra + 1.0 / params.Rb;
    const capacitanceProduct = params.Ca * params.Cb;
    return conductanceSum / capacitanceProduct;
};

/**
 * Calculate Bertrand coefficient D1 (Equation 4c from Bertrand et al. 1998)
 *
 * D1 = N1 * (1/Rm - 1/(Ra+Rb)) + 1/(Ra*Ca) + 1/(Rb*Cb)
 *
 * This simplifies to:
 * D1 = N1/Rsh + 1/tau_a + 1/tau_b
 *
 * where tau_a = Ra*Ca, tau_b = Rb*Cb
 *
 * IDENTIFIABILITY: Moderately identifiable (~10% precision)
 * Correlated with Rm (transepithelial resistance) and N0.
 */
export const calculateBertrandD1 = (params: CircuitParameters): number => {
    const N1 = calculateBertrandN1(params);
    const tauA = params.Ra * params.Ca;
    const tauB = params.Rb * params.Cb;

    // D1 = N1/Rsh + 1/tau_a + 1/tau_b
    return N1 / params.Rsh + 1.0 / tauA + 1.0 / tauB;
};

/**
 * Calculate all three Bertrand impedance coefficients.
 *
 * These three coefficients uniquely describe the monolayer impedance
 * and can be estimated directly from impedance measurements.
 *
 * The impedance transfer function (Equation 3 from Bertrand et al.):
 * Z(s) = (N1*s + N0) / (s^2 + D1*s + N0/Rm)
 *
 * In the frequency domain (s = j*omega):
 * Z(jw) = (N0 + j*w*N1) / ((N0/Rm - w^2) + j*w*D1)
 */
export const calculateBertrandCoefficients = (params: CircuitParameters): BertrandCoefficients => {
    const N1 = calculateBertrandN1(params);
    const N0 = calculateBertrandN0(params);
    const D1 = calculateBertrandD1(params);

    // Compute Rm (= TER) for reference
    const rTrans = params.Ra + params.Rb;
    const Rm = (params.Rsh * rTrans) / (params.Rsh + rTrans);

    return { N1, N0, D1, Rm };
};

/**
 * Calculate impedance from Bertrand coefficients.
 *
 * This is the forward model using the system theory formulation:
 * Z(jw) = (N0 + j*w*N1) / ((N0/Rm - w^2) + j*w*D1)
 *
 * This can be used to verify that the Bertrand representation
 * produces identical impedance spectra to the circuit model.
 */
export const calculateImpedanceFromBertrand = (
    coefficients: BertrandCoefficients,
    omega: number
): Complex => {
    const { N1, N0, D1, Rm } = coefficients;

    // Numerator: N0 + j*w*N1
    const numReal = N0;
    const numImag = omega * N1;

    // Denominator: (N0/Rm - w^2) + j*w*D1
    const denReal = N0 / Rm - omega * omega;
    const denImag = omega * D1;

    // Complex division
    const denMagSq = denReal * denReal + denImag * denImag;

    return {
        real: (numReal * denReal + numImag * denImag) / denMagSq,
        imag: (numImag * denReal - numReal * denImag) / denMagSq
    };
};

/**
 * Get the identifiability status for each Bertrand coefficient.
 * Based on Bertrand et al. (1998) Monte Carlo analysis.
 */
export const getBertrandIdentifiability = (): {
    N1: { level: 'high'; precision: string; description: string };
    N0: { level: 'low'; precision: string; description: string };
    D1: { level: 'moderate'; precision: string; description: string };
} => {
    return {
        N1: {
            level: 'high',
            precision: '~1%',
            description: 'Inverse total capacitance - highly identifiable from impedance'
        },
        N0: {
            level: 'low',
            precision: '~50%',
            description: 'Mixed conductance-capacitance - poorly identifiable, correlated with D1'
        },
        D1: {
            level: 'moderate',
            precision: '~10%',
            description: 'Damping coefficient - relates to time constants and shunt'
        }
    };
}; 