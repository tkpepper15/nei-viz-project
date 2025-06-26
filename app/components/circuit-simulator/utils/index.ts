// Export core utilities
export { calculateImpedanceSpectrum } from './impedance';
export { computeRegressionMesh } from './meshComputation';
export { calculateResnorm } from './resnorm';

// Export spider plot utilities
export { generateGridValues, getRefValue, isValidParameterKey } from './spider-utils';

// Format value with appropriate units
export const formatValue = (value: number, unit?: string): string => {
  if (unit === "μF" || unit?.includes("F")) {
    // Handle capacitance values (convert from F to μF)
    return `${(value * 1e6).toFixed(2)} μF`;
  } else if (unit === "Ω" || !unit) {
    // Handle resistance values
    if (value >= 1000) {
      return `${(value / 1000).toFixed(2)} kΩ`;
    } else {
      return `${value.toFixed(2)} Ω`;
    }
  } else {
    // Generic formatting
    return `${value.toFixed(3)}${unit ? ' ' + unit : ''}`;
  }
}; 