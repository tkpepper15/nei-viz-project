/**
 * Circuit parameters for the modified Randles equivalent circuit model
 * Used in electrochemical impedance spectroscopy (EIS) analysis for retinal pigment epithelium (RPE)
 *
 * Circuit topology:
 *       Rs (Shunt Resistance)
 *   ────[Rs]────┬──────────┬──────
 *               │          │
 *           [Ra]│      [Rb]│
 *               │          │
 *           [Ca]│      [Cb]│
 *               │          │
 *               └──────────┘
 */
export interface CircuitParameters {
  /** Shunt resistance (Ω) - parallel resistance across the entire circuit */
  Rsh: number;
  /** Apical resistance (Ω) - resistance of apical membrane */
  Ra: number;
  /** Apical capacitance (F) - capacitance of apical membrane */
  Ca: number;
  /** Basal resistance (Ω) - resistance of basal membrane */
  Rb: number;
  /** Basal capacitance (F) - capacitance of basal membrane */
  Cb: number;
  /** Frequency range for impedance measurements [min_freq, max_freq] in Hz */
  frequency_range: [number, number];
}

/**
 * Parameter constraints for optimization fitting
 * Used to implement constraint-based optimization as described in ISER 2024 paper
 * to resolve underdetermined system identification
 */
export interface ParameterConstraints {
  /** Whether to constrain Cb parameter during fitting */
  constrainCb: boolean;
  /** Fixed value for Cb when constrained (F) */
  cbConstraintValue?: number;
  /** Minimum allowed values for each parameter */
  minValues?: Partial<Omit<CircuitParameters, 'frequency_range'>>;
  /** Maximum allowed values for each parameter */
  maxValues?: Partial<Omit<CircuitParameters, 'frequency_range'>>;
  /** Whether to enforce physical relationships (e.g., Ca/Cb ratios) */
  enforcePhysicalConstraints?: boolean;
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

/**
 * Default parameter constraints configuration
 * Implements exploratory phase optimization approach from ISER 2024 paper
 */
export const DEFAULT_PARAMETER_CONSTRAINTS: ParameterConstraints = {
  constrainCb: false,
  cbConstraintValue: undefined,
  enforcePhysicalConstraints: false
};

/**
 * Validate circuit parameters against constraints
 * @param params Circuit parameters to validate
 * @param constraints Parameter constraints to enforce
 * @returns Validation result with any constraint violations
 */
export function validateParameterConstraints(
  params: CircuitParameters,
  constraints: ParameterConstraints
): { isValid: boolean; violations: string[] } {
  const violations: string[] = [];

  // Check Cb constraint
  if (constraints.constrainCb && constraints.cbConstraintValue !== undefined) {
    if (Math.abs(params.Cb - constraints.cbConstraintValue) > 1e-12) {
      violations.push(`Cb must be constrained to ${constraints.cbConstraintValue} F`);
    }
  }

  // Check minimum values
  if (constraints.minValues) {
    for (const [key, minValue] of Object.entries(constraints.minValues)) {
      const paramValue = params[key as keyof Omit<CircuitParameters, 'frequency_range'>];
      if (typeof paramValue === 'number' && typeof minValue === 'number' && paramValue < minValue) {
        violations.push(`${key} (${paramValue}) must be >= ${minValue}`);
      }
    }
  }

  // Check maximum values
  if (constraints.maxValues) {
    for (const [key, maxValue] of Object.entries(constraints.maxValues)) {
      const paramValue = params[key as keyof Omit<CircuitParameters, 'frequency_range'>];
      if (typeof paramValue === 'number' && typeof maxValue === 'number' && paramValue > maxValue) {
        violations.push(`${key} (${paramValue}) must be <= ${maxValue}`);
      }
    }
  }

  // Check physical relationships if enabled
  if (constraints.enforcePhysicalConstraints) {
    // Example: Check if capacitance ratio is physically reasonable
    const capacitanceRatio = params.Ca / params.Cb;
    if (capacitanceRatio < 0.1 || capacitanceRatio > 10) {
      violations.push(`Ca/Cb ratio (${capacitanceRatio.toFixed(2)}) should be between 0.1 and 10 for physical validity`);
    }

    // Example: Check if resistance ratio is reasonable
    const resistanceRatio = params.Ra / params.Rb;
    if (resistanceRatio < 0.1 || resistanceRatio > 10) {
      violations.push(`Ra/Rb ratio (${resistanceRatio.toFixed(2)}) should be between 0.1 and 10 for physical validity`);
    }
  }

  return {
    isValid: violations.length === 0,
    violations
  };
}

/**
 * Apply parameter constraints to a given parameter set
 * Modifies parameters in-place to satisfy constraints
 * @param params Circuit parameters to constrain
 * @param constraints Parameter constraints to apply
 * @returns Modified parameters satisfying constraints
 */
export function applyParameterConstraints(
  params: CircuitParameters,
  constraints: ParameterConstraints
): CircuitParameters {
  const constrainedParams = { ...params };

  // Apply Cb constraint
  if (constraints.constrainCb && constraints.cbConstraintValue !== undefined) {
    constrainedParams.Cb = constraints.cbConstraintValue;
  }

  // Apply minimum value constraints
  if (constraints.minValues) {
    for (const [key, minValue] of Object.entries(constraints.minValues)) {
      const currentValue = constrainedParams[key as keyof Omit<CircuitParameters, 'frequency_range'>];
      if (typeof currentValue === 'number' && typeof minValue === 'number' && currentValue < minValue) {
        (constrainedParams as Record<string, unknown>)[key] = minValue;
      }
    }
  }

  // Apply maximum value constraints
  if (constraints.maxValues) {
    for (const [key, maxValue] of Object.entries(constraints.maxValues)) {
      const currentValue = constrainedParams[key as keyof Omit<CircuitParameters, 'frequency_range'>];
      if (typeof currentValue === 'number' && typeof maxValue === 'number' && currentValue > maxValue) {
        (constrainedParams as Record<string, unknown>)[key] = maxValue;
      }
    }
  }

  return constrainedParams;
} 