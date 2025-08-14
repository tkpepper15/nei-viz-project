export interface CircuitParameters {
  Rsh: number;
  Ra: number;
  Ca: number;
  Rb: number;
  Cb: number;
  frequency_range: [number, number];
}

/**
 * Centralized parameter range configuration for consistent grid generation
 * across all visualizations (2D spider plot, 3D spider plot, data table)
 */
export const PARAMETER_RANGES = {
  Rsh: { min: 10, max: 10000 },     // Shunt resistance (Ω)
  Ra: { min: 10, max: 10000 },      // Apical resistance (Ω)
  Ca: { min: 0.1e-6, max: 50e-6 },  // Apical capacitance (F) - 0.1 to 50 μF
  Rb: { min: 10, max: 10000 },      // Basal resistance (Ω)  
  Cb: { min: 0.1e-6, max: 50e-6 },  // Basal capacitance (F) - 0.1 to 50 μF
  frequency: { min: 1, max: 10000 } // Frequency range (Hz)
} as const;

/**
 * Default grid size for parameter space exploration
 * This creates 5^5 = 3,125 total parameter combinations
 */
export const DEFAULT_GRID_SIZE = 5;

/**
 * Grid size limits for performance management
 */
export const GRID_SIZE_LIMITS = {
  min: 2,
  max: 20,
  recommended: 5,
  performanceWarning: 10 // Show warning above this size
} as const;

/**
 * Convert capacitance from Farads to microFarads for display
 */
export const faradToMicroFarad = (farad: number): number => farad * 1e6;

/**
 * Convert capacitance from microFarads to Farads for calculation
 */
export const microFaradToFarad = (microFarad: number): number => microFarad * 1e-6; 