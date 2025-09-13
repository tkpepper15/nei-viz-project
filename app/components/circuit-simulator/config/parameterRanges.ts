/**
 * Canonical parameter ranges for EIS circuit simulation
 * Used across all computation implementations to ensure consistency
 */

export interface ParameterRanges {
  Rsh: [number, number];
  Ra: [number, number];
  Ca: [number, number];
  Rb: [number, number];
  Cb: [number, number];
}

/**
 * Standard parameter ranges based on RPE cell research literature
 * These ranges should be used consistently across all implementations
 */
export const CANONICAL_PARAMETER_RANGES: ParameterRanges = {
  Rsh: [10, 10000],      // Shunt resistance (Ω)
  Ra: [10, 10000],       // Apical resistance (Ω) 
  Ca: [0.1e-6, 50e-6],   // Apical capacitance (F)
  Rb: [10, 10000],       // Basal resistance (Ω)
  Cb: [0.1e-6, 50e-6]    // Basal capacitance (F)
};

/**
 * Default ground truth parameters for testing and validation
 */
export const DEFAULT_GROUND_TRUTH = {
  Rsh: 24,        // Shunt resistance (Ω)
  Ra: 500,        // Apical resistance (Ω) 
  Ca: 0.5e-6,     // Apical capacitance (F)
  Rb: 500,        // Basal resistance (Ω)
  Cb: 0.5e-6      // Basal capacitance (F)
};