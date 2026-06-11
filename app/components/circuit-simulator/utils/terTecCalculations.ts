/**
 * TER/TEC Calculation Utilities
 * ==============================
 *
 * Utilities for calculating Transepithelial Resistance (TER) and
 * Transepithelial Capacitance (TEC) from circuit parameters and impedance data.
 */

import { CircuitParameters } from '../types/parameters';


/**
 * Calculate TER (Transepithelial Resistance) - DC resistance
 * TER = Rsh * (Ra + Rb) / (Rsh + Ra + Rb)
 *
 * This is the parallel combination of Rsh with the series combination of Ra and Rb.
 *
 * @param params - Circuit parameters containing Rsh, Ra, Rb
 * @returns TER value in Ohms (Ω)
 */
export function calculateTER(params: CircuitParameters): number {
  const numerator = params.Rsh * (params.Ra + params.Rb);
  const denominator = params.Rsh + params.Ra + params.Rb;

  if (denominator === 0) {
    return 0; // Avoid division by zero
  }

  return numerator / denominator;
}

/**
 * Calculate TEC (Transepithelial Capacitance) from circuit parameters
 *
 * For the circuit topology: Rₛₕ ∥ [(Rₐ ∥ Cₐ) + (Rᵦ ∥ Cᵦ)]
 * The apical and basal capacitors are in series, so:
 *
 * TEC = (Ca × Cb) / (Ca + Cb)
 *
 * This is the series combination of two capacitors.
 *
 * @param params - Circuit parameters
 * @returns TEC value in Farads (F)
 */
export function calculateTEC(params: CircuitParameters): number {
  if (!params.Ca || !params.Cb || params.Ca <= 0 || params.Cb <= 0) {
    return 0;
  }

  // Series capacitor formula: 1/C_total = 1/Ca + 1/Cb
  // Which simplifies to: C_total = (Ca × Cb) / (Ca + Cb)
  const tec = (params.Ca * params.Cb) / (params.Ca + params.Cb);

  if (!isFinite(tec) || tec <= 0) {
    return 0;
  }

  return tec;
}


/**
 * Group models by TER value within a tolerance range
 *
 * @param ter - Target TER value
 * @param tolerance - Tolerance as a fraction (e.g., 0.05 = ±5%)
 * @returns Object with min and max TER range
 */
export function getTERRange(ter: number, tolerance: number = 0.05): { min: number; max: number } {
  const delta = ter * tolerance;
  return {
    min: ter - delta,
    max: ter + delta
  };
}

/**
 * Group models by TEC value within a tolerance range
 *
 * @param tec - Target TEC value
 * @param tolerance - Tolerance as a fraction (e.g., 0.05 = ±5%)
 * @returns Object with min and max TEC range
 */
export function getTECRange(tec: number, tolerance: number = 0.05): { min: number; max: number } {
  const delta = tec * tolerance;
  return {
    min: tec - delta,
    max: tec + delta
  };
}

/**
 * Format TER value for display
 *
 * @param ter - TER value in Ohms
 * @returns Formatted string with appropriate units
 */
export function formatTER(ter: number): string {
  if (ter >= 1000) {
    return `${(ter / 1000).toFixed(2)} kΩ`;
  }
  return `${ter.toFixed(2)} Ω`;
}

/**
 * Format TEC value for display
 *
 * @param tec - TEC value in Farads
 * @returns Formatted string with appropriate units
 */
export function formatTEC(tec: number): string {
  if (tec >= 1e-6) {
    return `${(tec * 1e6).toFixed(2)} µF`;
  } else if (tec >= 1e-9) {
    return `${(tec * 1e9).toFixed(2)} nF`;
  } else if (tec >= 1e-12) {
    return `${(tec * 1e12).toFixed(2)} pF`;
  }
  return `${tec.toExponential(2)} F`;
}
