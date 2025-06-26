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
  const paramSets = {
    Rs: new Set<number>(),
    Ra: new Set<number>(),
    Ca: new Set<number>(),
    Rb: new Set<number>(),
    Cb: new Set<number>()
  };

  // Collect values from all items
  items.forEach(item => {
    const params = item.parameters;
    paramSets.Rs.add(params.Rs);
    paramSets.Ra.add(params.Ra);
    paramSets.Ca.add(params.Ca);
    paramSets.Rb.add(params.Rb);
    paramSets.Cb.add(params.Cb);
  });

  // Convert Sets to sorted Arrays
  return {
    Rs: Array.from(paramSets.Rs).sort((a, b) => a - b),
    Ra: Array.from(paramSets.Ra).sort((a, b) => a - b),
    Ca: Array.from(paramSets.Ca).sort((a, b) => a - b),
    Rb: Array.from(paramSets.Rb).sort((a, b) => a - b),
    Cb: Array.from(paramSets.Cb).sort((a, b) => a - b)
  };
}; 