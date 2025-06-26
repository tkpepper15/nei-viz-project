import { ModelSnapshot } from './types';
import { CircuitParameters } from '../types/parameters';
import { GridParameterArrays } from './types';

// Update type checking for parameters
export const isValidParameterKey = (key: string): key is keyof CircuitParameters => {
  const validKeys = ['Rs', 'Ra', 'Ca', 'Rb', 'Cb'];
  return validKeys.includes(key);
};

// Function to safely get reference value
export const getRefValue = (params: CircuitParameters, key: keyof CircuitParameters): number => {
  const value = params[key];
  return typeof value === 'number' ? value : 0;
};

// Function to generate grid values
export const generateGridValues = (items: ModelSnapshot[]): GridParameterArrays => {
  // Ensure we have valid items
  if (!items || items.length === 0) {
    console.warn('No valid items provided to generateGridValues');
    return {
      Rs: [],
      Ra: [],
      Ca: [],
      Rb: [],
      Cb: []
    };
  }

  const paramSets = {
    Rs: new Set<number>(),
    Ra: new Set<number>(),
    Ca: new Set<number>(),
    Rb: new Set<number>(),
    Cb: new Set<number>()
  };

  // Collect values from all items with validation
  items.forEach(item => {
    if (!item?.parameters) {
      console.warn('Invalid item in generateGridValues:', item);
      return;
    }

    const params = item.parameters;
    
    // Validate and add each parameter
    if (typeof params.Rs === 'number' && isFinite(params.Rs)) {
      paramSets.Rs.add(params.Rs);
    }
    if (typeof params.Ra === 'number' && isFinite(params.Ra)) {
      paramSets.Ra.add(params.Ra);
    }
    if (typeof params.Ca === 'number' && isFinite(params.Ca)) {
      paramSets.Ca.add(params.Ca);
    }
    if (typeof params.Rb === 'number' && isFinite(params.Rb)) {
      paramSets.Rb.add(params.Rb);
    }
    if (typeof params.Cb === 'number' && isFinite(params.Cb)) {
      paramSets.Cb.add(params.Cb);
    }
  });

  // Convert Sets to sorted Arrays with validation
  const result: GridParameterArrays = {
    Rs: Array.from(paramSets.Rs).sort((a, b) => a - b),
    Ra: Array.from(paramSets.Ra).sort((a, b) => a - b),
    Ca: Array.from(paramSets.Ca).sort((a, b) => a - b),
    Rb: Array.from(paramSets.Rb).sort((a, b) => a - b),
    Cb: Array.from(paramSets.Cb).sort((a, b) => a - b)
  };

  // Ensure we have at least one value for each parameter
  Object.entries(result).forEach(([key, values]) => {
    if (values.length === 0) {
      console.warn(`No valid values found for parameter ${key}`);
      result[key as keyof GridParameterArrays] = [0]; // Default value
    }
  });

  // Log the generated values for debugging
  console.log('Generated grid values:', result);

  return result;
}; 