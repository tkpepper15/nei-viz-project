import { CircuitParameters } from '../types/parameters';

export interface ParameterRange {
    min: number;
    max: number;
}

export interface ParameterSpace {
    Rs: ParameterRange;
    Ra: ParameterRange;
    Ca: ParameterRange;
    Rb: ParameterRange;
    Cb: ParameterRange;
}

/**
 * Generate a logarithmic space between min and max values
 */
export const generateLogSpace = (min: number, max: number, points: number): number[] => {
    const logMin = Math.log10(min);
    const logMax = Math.log10(max);
    const step = (logMax - logMin) / (points - 1);
    return Array.from({ length: points }, (_, i) => 
        Math.pow(10, logMin + i * step)
    );
};

/**
 * Generate a linear space between min and max values
 */
export const generateLinearSpace = (min: number, max: number, points: number): number[] => {
    const step = (max - min) / (points - 1);
    return Array.from({ length: points }, (_, i) => min + i * step);
};

/**
 * Generate a mesh of parameter combinations
 */
export const generateParameterMesh = (
    space: ParameterSpace,
    resolution: number
): CircuitParameters[] => {
    // Generate parameter values
    const RsValues = generateLogSpace(space.Rs.min, space.Rs.max, resolution);
    const RaValues = generateLogSpace(space.Ra.min, space.Ra.max, resolution);
    const CaValues = generateLogSpace(space.Ca.min, space.Ca.max, resolution);
    const RbValues = generateLogSpace(space.Rb.min, space.Rb.max, resolution);
    const CbValues = generateLogSpace(space.Cb.min, space.Cb.max, resolution);

    // Generate all combinations
    const combinations: CircuitParameters[] = [];
    for (const Rs of RsValues) {
        for (const Ra of RaValues) {
            for (const Ca of CaValues) {
                for (const Rb of RbValues) {
                    for (const Cb of CbValues) {
                        combinations.push({
                            Rs,
                            Ra,
                            Ca,
                            Rb,
                            Cb,
                            frequency_range: [1, 1000] as [number, number] // Default frequency range
                        });
                    }
                }
            }
        }
    }

    return combinations;
};

/**
 * Calculate alpha value for visualization based on resnorm
 */
export const calculateAlpha = (
    resnorm: number,
    minResnorm: number,
    maxResnorm: number
): number => {
    // Handle edge cases
    if (!isFinite(resnorm) || !isFinite(minResnorm) || !isFinite(maxResnorm)) {
        return 0.0; // Return fully transparent for invalid values
    }

    if (maxResnorm === minResnorm) {
        return 1.0; // All points are equally good
    }

    // Ensure resnorm is within bounds
    resnorm = Math.max(minResnorm, Math.min(resnorm, maxResnorm));

    // Normalize between 0 and 1
    const normalized = (resnorm - minResnorm) / (maxResnorm - minResnorm);

    // Use sigmoid function to create smooth transition
    const alpha = 1 / (1 + Math.exp(5 * (normalized - 0.5)));

    // Ensure alpha is between 0 and 1
    return Math.max(0.0, Math.min(1.0, alpha));
}; 