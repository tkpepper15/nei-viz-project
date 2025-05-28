"use client";

import React, { useMemo, useEffect, useState, useCallback } from 'react';
import {
  Radar,
  RadarChart,
  PolarAngleAxis,
  ResponsiveContainer,
  PolarGrid,
  PolarRadiusAxis,
} from 'recharts';
import { ModelSnapshot } from '../types';

// Structure to hold info about each radar line (model)
interface SpiderState {
  name: string;
  color: string;
  dataKey: string;
  resnorm?: number;
  opacity: number;
}

// Define Parameter interface and SpiderData type
interface Parameter {
  key: string;
  displayName: string;
  unit?: string;
  min?: number;
  max?: number;
}

// Define PolarAxisProps for proper typing
// interface PolarAxisProps {
//   cx: number;
//   cy: number;
//   radius: number;
//   payload: {
//     value: number;
//   };
// }

export interface SpiderData {
  data: Record<string, string | number>[];
  states: SpiderState[];
  referenceValues: Record<string, number>;
  parameters: Parameter[];
}

// Add dataKey to ModelSnapshot interface via declaration merging
declare module '../types' {
  interface ModelSnapshot {
    dataKey?: string;
  }
}

// Update the CircuitParameters type to include both uppercase and lowercase variants
type CircuitParameters = {
  ra: number;
  ca: number;
  rb: number;
  cb: number;
  Rs: number;
  // Also include uppercase variants for compatibility
  Ra?: number;
  Ca?: number;
  Rb?: number;
  Cb?: number;
};

// Function to ensure consistent parameter handling
const normalizeParameters = (params: Record<string, number | [number, number]>): CircuitParameters => {
  // Create a new object without frequency_range
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const { frequency_range, ...rest } = params;
  const normalized = { ...rest } as CircuitParameters;
  
  // Handle parameter conversion to Rs
  if ('rBlank' in params && params.rBlank !== undefined) {
    normalized.Rs = params.rBlank as number;
  } else if ('rShunt' in params && params.rShunt !== undefined) {
    normalized.Rs = params.rShunt as number;
  } else if (!('Rs' in params) || normalized.Rs === undefined) {
    normalized.Rs = 0; // Default value if none is present
  }
  
  // Ensure all parameters have sensible defaults
  normalized.ra = normalized.ra || normalized.Ra || 0;
  normalized.ca = normalized.ca || normalized.Ca || 0;
  normalized.rb = normalized.rb || normalized.Rb || 0;
  normalized.cb = normalized.cb || normalized.Cb || 0;
  
  return normalized;
};

// Function to calculate impedance from circuit parameters at a given frequency
// This follows the Randles equivalent circuit model
const calculateImpedance = (params: CircuitParameters, frequency: number): { real: number; imag: number } => {
  const w = 2 * Math.PI * frequency; // Angular frequency
  
  // Calculate impedances of the RC elements
  // Za = Ra / (1 + jwRaCa)
  const za_real = params.ra / (1 + Math.pow(w * params.ra * params.ca, 2));
  const za_imag = -w * Math.pow(params.ra, 2) * params.ca / (1 + Math.pow(w * params.ra * params.ca, 2));
  
  // Zb = Rb / (1 + jwRbCb)
  const zb_real = params.rb / (1 + Math.pow(w * params.rb * params.cb, 2));
  const zb_imag = -w * Math.pow(params.rb, 2) * params.cb / (1 + Math.pow(w * params.rb * params.cb, 2));
  
  // Add Za and Zb (since they are in series with each other in the Randles circuit)
  const z_series_real = za_real + zb_real;
  const z_series_imag = za_imag + zb_imag;
  
  // Add Rs in series with the combined Za+Zb
  const total_real = params.Rs + z_series_real;
  const total_imag = z_series_imag;
  
  return { real: total_real, imag: total_imag };
};

// Calculate resnorm between two sets of impedances across frequencies
const calculateImpedanceResnorm = (
  ref: Array<{ real: number; imag: number }>,
  test: Array<{ real: number; imag: number }>
): number => {
  if (ref.length !== test.length || ref.length === 0) return Infinity;
  
  let sumSquaredDiff = 0;
  let sumSquaredRef = 0;
  
  for (let i = 0; i < ref.length; i++) {
    const realDiff = ref[i].real - test[i].real;
    const imagDiff = ref[i].imag - test[i].imag;
    const squaredDiff = realDiff * realDiff + imagDiff * imagDiff;
    
    const refMagSquared = ref[i].real * ref[i].real + ref[i].imag * ref[i].imag;
    
    sumSquaredDiff += squaredDiff;
    sumSquaredRef += refMagSquared;
  }
  
  return sumSquaredDiff / sumSquaredRef;
};

// Function to create a color scale based on resnorm values
const createResnormColorScale = (range: { min: number; max: number }) => {
  return (resnorm: number): string => {
    // Default for invalid resnorm
    if (!Number.isFinite(resnorm)) return 'hsl(0, 75%, 50%)'; // Red for invalid/infinite

    // If resnorm is 0 or very small, return pure green (reference point)
    if (resnorm < 1e-10) return 'hsl(120, 80%, 40%)';

    // For very small range of values, use a different approach
    if (range.max - range.min < 1e-6) {
      return 'hsl(200, 75%, 45%)'; // Use a fixed blue color
    }

    // Use log scale for better distribution since resnorm values can vary by orders of magnitude
    const logMin = Math.log10(Math.max(range.min, 1e-10)); // Avoid log(0)
    const logMax = Math.log10(Math.max(range.max, 1e-9));
    const logResnorm = Math.log10(Math.max(resnorm, 1e-10));
    
    // Invert: lower resnorm (better fit) = higher value (greener)
    const colorValue = 1 - Math.min(1, Math.max(0, (logResnorm - logMin) / (logMax - logMin)));
    
    // Generate color using a better gradient from green to yellow to orange to red
    let hue;
    if (colorValue > 0.75) {
      // Green to yellowish-green (120° to 90°)
      hue = 120 - (1 - colorValue) * 4 * 30;
    } else if (colorValue > 0.5) {
      // Yellowish-green to yellow (90° to 60°)
      hue = 90 - (0.75 - colorValue) * 4 * 30;
    } else if (colorValue > 0.25) {
      // Yellow to orange (60° to 30°)
      hue = 60 - (0.5 - colorValue) * 4 * 30;
    } else {
      // Orange to red (30° to 0°)
      hue = 30 - colorValue * 4 * 30;
    }
    
    return `hsl(${hue}, 85%, 45%)`; // More saturated for better visibility
  };
};

// Ensure the interface has the correct property names to match what's being passed from CircuitSimulator
interface SpiderPlotProps {
  meshItems: ModelSnapshot[];
  referenceId?: string;
  opacityFactor?: number; // Add opacity factor parameter
}

export const SpiderPlot: React.FC<SpiderPlotProps> = ({ 
  meshItems,
  referenceId,
  opacityFactor = 0.7 // Default value if not provided
}) => {
  // State to hold min/max resnorm for opacity scaling
  const [resnormRange, setResnormRange] = useState({ min: 0, max: 1 });
  const [referenceItem, setReferenceItem] = useState<ModelSnapshot | null>(null);
  
  // Standard test frequencies for impedance calculations
  const testFrequencies = useMemo(() => {
    // Generate logarithmically spaced frequencies between 0.1 Hz and 100 kHz
    const frequencies: number[] = [];
    const start = 0.1;
    const end = 100000;
    const points = 20;
    
    const logStart = Math.log10(start);
    const logEnd = Math.log10(end);
    const step = (logEnd - logStart) / (points - 1);
    
    for (let i = 0; i < points; i++) {
      frequencies.push(Math.pow(10, logStart + i * step));
    }
    
    return frequencies;
  }, []);
  
  // Handle reference ID changes from parent
  useEffect(() => {
    if (referenceId && meshItems.length > 0) {
      const selectedReference = meshItems.find((item: ModelSnapshot) => item.id === referenceId);
      if (selectedReference) {
        setReferenceItem(selectedReference);
      }
    }
  }, [referenceId, meshItems]);

  // Calculate min/max resnorm when meshItems change and identify reference item
  useEffect(() => {
    // Skip if no mesh items or if reference already set via prop
    if (meshItems.length === 0) return;
    
    // Sort items first by resnorm
    const sortedItems = [...meshItems].sort((a, b) => {
      const aResnorm = a.resnorm || Infinity;
      const bResnorm = b.resnorm || Infinity;
      return aResnorm - bResnorm;
    });
    
    // Find our reference item - either from prop or default criteria
    let selectedRef: ModelSnapshot | null = null;
    
    if (referenceId) {
      selectedRef = meshItems.find((item: ModelSnapshot) => item.id === referenceId) || null;
    } 
    
    if (!selectedRef) {
      // Otherwise find reference/ground truth item by default criteria
      selectedRef = sortedItems.find((item: ModelSnapshot) => 
      item.id === 'ground-truth' || 
      item.name === 'Ground Truth' || 
      Math.abs(item.resnorm || 1) < 1e-10
    ) || (sortedItems.length > 0 ? sortedItems[0] : null);
    }
    
    if (selectedRef) {
      setReferenceItem(selectedRef);
    }
    
    // Calculate impedance-based resnorms if we have a reference
    if (selectedRef) {
      // Calculate reference impedances across frequencies
      const normalizedRefParams = normalizeParameters(selectedRef.parameters);
      const refImpedances = testFrequencies.map(freq => 
        calculateImpedance(normalizedRefParams, freq)
      );
      
      // Calculate resnorms for all non-reference items
      const impedanceResnorms = sortedItems
        .filter((item: ModelSnapshot) => item.id !== selectedRef.id)
        .map((item: ModelSnapshot) => {
          const normalizedParams = normalizeParameters(item.parameters);
          const testImpedances = testFrequencies.map(freq => 
            calculateImpedance(normalizedParams, freq)
          );
          return calculateImpedanceResnorm(refImpedances, testImpedances);
        })
        .filter((res: number) => Number.isFinite(res));
      
      if (impedanceResnorms.length > 0) {
        const min = Math.min(...impedanceResnorms);
        const max = Math.max(...impedanceResnorms);
        setResnormRange({ min, max });
      } else {
        // Fallback to using the original resnorms
        const originalResnorms = sortedItems
          .filter((item: ModelSnapshot) => item.id !== selectedRef.id)
          .map((item: ModelSnapshot) => item.resnorm)
          .filter((res: number | undefined) => res !== undefined && Number.isFinite(res)) as number[];
          
        if (originalResnorms.length > 0) {
          const min = Math.min(...originalResnorms);
          const max = Math.max(...originalResnorms);
          setResnormRange({ min, max });
        } else {
          setResnormRange({ min: 0, max: 1 }); // Default
        }
      }
    } else {
      // No reference, use original resnorms
      const resnorms = sortedItems
        .map((item: ModelSnapshot) => item.resnorm)
        .filter((res: number | undefined) => res !== undefined && Number.isFinite(res)) as number[];
        
      if (resnorms.length > 0) {
        const min = Math.min(...resnorms);
        const max = Math.max(...resnorms);
        setResnormRange({ min, max });
      } else {
        setResnormRange({ min: 0, max: 1 }); // Default
      }
    }
  }, [meshItems, testFrequencies]);

  // Recalculate impedance-based resnorms when reference item changes manually
  useEffect(() => {
    if (!referenceItem || meshItems.length === 0) return;
    
    // Calculate reference impedances across frequencies
    const normalizedRefParams = normalizeParameters(referenceItem.parameters);
    const refImpedances = testFrequencies.map(freq => 
      calculateImpedance(normalizedRefParams, freq)
    );
    
    // Calculate resnorms for all non-reference items
    const impedanceResnorms = meshItems
      .filter((item: ModelSnapshot) => item.id !== referenceItem.id)
      .map((item: ModelSnapshot) => {
        const normalizedParams = normalizeParameters(item.parameters);
        const testImpedances = testFrequencies.map(freq => 
          calculateImpedance(normalizedParams, freq)
        );
        return calculateImpedanceResnorm(refImpedances, testImpedances);
      })
      .filter((res: number) => Number.isFinite(res));
    
    if (impedanceResnorms.length > 0) {
      const min = Math.min(...impedanceResnorms);
      const max = Math.max(...impedanceResnorms);
      setResnormRange({ min, max });
    }
  }, [referenceItem, testFrequencies, meshItems]);

  // Function to calculate opacity based on resnorm and opacity factor
  const calculateLineOpacity = useCallback(
    (resnorm: number | undefined, isReference: boolean): number => {
    if (isReference) return 1; // Reference always fully visible
    if (resnorm === undefined || !Number.isFinite(resnorm)) return 0.4; // Default for unknown
    
    // Use log scale to map resnorm to opacity
    const { min, max } = resnormRange;
    
    // Handle edge cases
    if (min === max) return opacityFactor;
    
    // Base opacity is inversely related to resnorm (lower resnorm = higher quality = more visible)
    const logMin = Math.log10(Math.max(min, 1e-10));
    const logMax = Math.log10(Math.max(max, 1e-9));
    const logResnorm = Math.log10(Math.max(resnorm, 1e-10));
    
    // Higher quality (lower resnorm) items are more visible
    const normalizedQuality = 1 - Math.min(1, Math.max(0, (logResnorm - logMin) / (logMax - logMin)));
    
    // Create dramatically more steps with higher opacity values
    const stepCount = Math.max(5, Math.round(60 * opacityFactor)); // 5 to 60 steps (increased for more granularity)
    
    // Apply a more aggressive curve for the step calculation
    // Use an adaptive power function that changes behavior based on opacity setting
    const adaptivePower = () => {
      if (opacityFactor < 0.3) {
        // Gentler curve at low opacity settings
        return 0.8;
      } else if (opacityFactor < 0.6) {
        // Moderate curve at medium opacity settings
        return 0.6 + (opacityFactor * 0.8);
      } else {
        // Very aggressive curve at high opacity settings
        return 0.4 + (opacityFactor * 1.8);
      }
    };
    
    const power = adaptivePower();
    // Use polynomial interpolation for smoother steps
    const polyCurve = Math.pow(normalizedQuality, 1/power);
    const stepIndex = Math.floor(stepCount * polyCurve);
    const fraction = (stepCount * polyCurve) - stepIndex; // Get fractional part for smooth interpolation
    
    // Smooth quantization using interpolation between steps
    const step1 = 1 - (stepIndex / stepCount);
    const step2 = 1 - ((stepIndex + 1) / stepCount);
    const smoothQuantized = step1 * (1 - fraction) + step2 * fraction;
    
    // Enhanced contrast factor - super aggressive but with smoother transitions
    const getContrastCurve = () => {
      // Multi-segment contrast curve for more refined control
      if (opacityFactor < 0.3) {
        // Low opacity regime: mild contrast (2.0-3.0)
        return 2.0 + (opacityFactor * 3.3);
      } else if (opacityFactor < 0.6) {
        // Mid opacity regime: medium contrast (3.0-5.5)
        return 3.0 + ((opacityFactor - 0.3) * 8.3);
      } else if (opacityFactor < 0.8) {
        // High opacity regime: high contrast (5.5-8.0)
        return 5.5 + ((opacityFactor - 0.6) * 12.5);
      } else {
        // Extreme opacity regime: extreme contrast (8.0-15.0)
        return 8.0 + ((opacityFactor - 0.8) * 35);
      }
    };
    
    const contrastFactor = getContrastCurve();
    const enhancedOpacity = Math.pow(1 - smoothQuantized, contrastFactor);
    
    // Apply adaptive transform power based on opacity setting
    const getTransformPower = () => {
      if (opacityFactor < 0.4) {
        // Gentle transform at low settings: 0.4-0.6
        return 0.4 + (opacityFactor * 0.5);
      } else if (opacityFactor < 0.7) {
        // Medium transform at mid settings: 0.6-1.0
        return 0.6 + ((opacityFactor - 0.4) * 1.33);
      } else {
        // Strong transform at high settings: 1.0-2.0
        return 1.0 + ((opacityFactor - 0.7) * 3.33);
      }
    };
    
    const logTransformPower = getTransformPower();
    
    // Use a multi-stage sigmoid for more refined control over the transition curve
    const getSigmoidParams = () => {
      // Return midpoint and steepness values that vary with opacity settings
      if (opacityFactor < 0.3) {
        // Low setting: gentle slope, centered lower
        return { midpoint: 0.6, steepness: 3.0 + (opacityFactor * 10) };
      } else if (opacityFactor < 0.6) {
        // Medium setting: moderate slope, centered in middle
        return { midpoint: 0.65, steepness: 6.0 + ((opacityFactor - 0.3) * 16.7) };
      } else if (opacityFactor < 0.8) {
        // High setting: steep slope, centered higher
        return { midpoint: 0.7, steepness: 11.0 + ((opacityFactor - 0.6) * 40) };
      } else {
        // Extreme setting: very steep slope
        return { midpoint: 0.75, steepness: 19.0 + ((opacityFactor - 0.8) * 100) };
      }
    };
    
    const { midpoint, steepness } = getSigmoidParams();
    
    // Calculate sigmoid value using the refined parameters
    const sigmoidInput = (Math.pow(enhancedOpacity, logTransformPower) - midpoint) * steepness;
    const sigmoidOpacity = 1 / (1 + Math.exp(-sigmoidInput));
    
    // Scale the sigmoid with adaptive min/max values
    const getMinOpacity = () => {
      // More granular control over minimum opacity
      if (opacityFactor < 0.3) {
        // Low setting: relatively high minimum (0.15-0.08)
        return 0.15 - (opacityFactor * 0.23);
      } else if (opacityFactor < 0.6) {
        // Medium setting: medium minimum (0.08-0.04)
        return 0.08 - ((opacityFactor - 0.3) * 0.13);
      } else if (opacityFactor < 0.8) {
        // High setting: low minimum (0.04-0.02)
        return 0.04 - ((opacityFactor - 0.6) * 0.1);
      } else {
        // Extreme setting: very low minimum (0.02-0.005)
        return 0.02 - ((opacityFactor - 0.8) * 0.075);
      }
    };
    
    const minOpacity = getMinOpacity();
    const maxOpacity = Math.min(0.98, 0.85 + (opacityFactor * 0.15)); // Dynamic max based on opacity factor
    const scaledOpacity = minOpacity + (sigmoidOpacity * (maxOpacity - minOpacity));
    
    return Math.max(minOpacity, Math.min(maxOpacity, scaledOpacity));
    },
    [resnormRange, opacityFactor]
  );

  // Function to format parameter values based on parameter type
  const formatParameterValue = (value: number | [number, number], paramKey: string): string => {
    // Skip if value is an array (frequency_range)
    if (Array.isArray(value)) {
      return `[${value[0]}-${value[1]}]`;
    }
    
    // Handle special parameters with unit conversions
    if (paramKey === 'ca' || paramKey === 'Ca' || paramKey === 'cb' || paramKey === 'Cb') {
      return `${(value * 1e6).toFixed(2)} μF`;
    } else if (paramKey === 'Rs' || paramKey === 'ra' || paramKey === 'Ra' || paramKey === 'rb' || paramKey === 'Rb') {
      return `${value.toFixed(1)} Ω`;
    } else if (paramKey === 'TER' || paramKey === 'ter') {
      return `${value.toFixed(1)} Ω`;
    } else if (paramKey === 'resnorm') {
      return value.toExponential(2);
    }
    
    // Default formatting for other numeric values
    return value.toFixed(2);
  };

  // Helper function to apply normalization effects to normalized values
  const applyNormalizationEffects = useCallback(
    (
      normalizedValue: number,
      referenceItem: ModelSnapshot | null | undefined,
      paramName: string,
      originalValue: number
    ): number => {
      // For reference point, add a slight boost to make it more visible
      if (referenceItem) {
        const normalizedRefParams = normalizeParameters(referenceItem.parameters);
        let refValue: number | undefined;
        
        if (paramName === 'Rs') {
          refValue = normalizedRefParams.Rs;
        } else {
          refValue = normalizedRefParams[paramName as keyof typeof normalizedRefParams];
        }
        
        // Use absolute comparison with small tolerance for floating point
        if (refValue !== undefined && Math.abs(originalValue - refValue) < 1e-10) {
          // Add a slight boost for better visibility of the reference
          normalizedValue = Math.min(1, normalizedValue + 0.05);
        }
      }
      
      return Math.max(0, Math.min(1, normalizedValue)); // Ensure value is between 0 and 1
    },
    []
  );

  // Enhanced function to normalize parameter values
  const normalizeParameterValue = useCallback(
    (
      value: number, 
      paramName: string, 
      referenceItem: ModelSnapshot | null | undefined,
      meshItems: ModelSnapshot[]
    ): number => {
      // If no meshItems or this is a non-numeric value, return 0.5 (middle)
      if (!meshItems.length || typeof value !== 'number' || !Number.isFinite(value)) {
        return 0.5;
      }
      
      // Get reference parameter value if available
      let refValue: number | undefined;
      if (referenceItem) {
        const refParams = normalizeParameters(referenceItem.parameters);
        refValue = paramName === 'Rs' 
          ? refParams.Rs 
          : refParams[paramName as keyof typeof refParams];
      }
      
      // Get all values for this parameter across all meshItems
      const allValues = meshItems
        .map((item: ModelSnapshot) => {
          const normalizedParams = normalizeParameters(item.parameters);
          let paramValue: number | undefined;
          
          // Handle special case for Rs
          if (paramName === 'Rs') {
            paramValue = normalizedParams.Rs;
          } else {
            paramValue = normalizedParams[paramName as keyof typeof normalizedParams];
          }
          
          return typeof paramValue === 'number' ? paramValue : NaN;
        })
        .filter((v: number) => Number.isFinite(v));
      
      // If we have no valid values, return 0.5 (middle of the scale)
      if (!allValues.length) return 0.5;
      
      // Get min and max values
      const minValue = Math.min(...allValues);
      const maxValue = Math.max(...allValues);
      
      // If min and max are the same, return 0.5 (middle of the scale)
      if (maxValue === minValue) return 0.5;
      
      // Use reference-centered normalization if a reference value exists
      if (refValue !== undefined && Number.isFinite(refValue)) {
        // Determine visualization strategy based on parameter type
        if (paramName === 'ca' || paramName === 'cb') {
          // For capacitance - use ratio to reference (log scale)
          // A value of 0.5 means equal to reference
          // Values < 0.5 mean smaller than reference, > 0.5 mean larger
          const ratio = value / refValue;
          
          // Map to [0,1] range with 0.5 being the reference value
          // Use asymmetric scaling to handle wide ranges
          let normalizedValue;
          if (ratio < 1) {
            // Less than reference (scale from 0 to 0.5)
            normalizedValue = 0.5 * (Math.log10(Math.max(ratio, 0.01)) + 2) / 2;
          } else {
            // Greater than reference (scale from 0.5 to 1)
            normalizedValue = 0.5 + 0.5 * Math.min(1, Math.log10(ratio) / 2);
          }
          
          return applyNormalizationEffects(normalizedValue, referenceItem, paramName, value);
        } 
        else if (paramName === 'ra' || paramName === 'rb' || paramName === 'Rs') {
          // For resistance - use ratio to reference (log scale)
          const ratio = value / refValue;
          
          // Map to [0,1] range with 0.5 being the reference value
          let normalizedValue;
          if (ratio < 1) {
            // Less than reference (scale from 0 to 0.5)
            normalizedValue = 0.5 * (Math.log10(Math.max(ratio, 0.01)) + 2) / 2;
          } else {
            // Greater than reference (scale from 0.5 to 1)
            normalizedValue = 0.5 + 0.5 * Math.min(1, Math.log10(ratio) / 2);
          }
          
          return applyNormalizationEffects(normalizedValue, referenceItem, paramName, value);
        }
      }
      
      // Fallback to the standard normalization if no reference available
      // Choose normalization method based on parameter type
      if (paramName === 'ca' || paramName === 'cb') {
        // For capacitance values (always positive), use logarithmic scale
        const safeValue = Math.max(value, 1e-15);
        const safeMin = Math.max(minValue, 1e-15);
        const safeMax = Math.max(maxValue, 1e-14);
        
        const logMin = Math.log10(safeMin);
        const logMax = Math.log10(safeMax);
        const logValue = Math.log10(safeValue);
        
        const normalizedValue = (logValue - logMin) / (logMax - logMin);
        return applyNormalizationEffects(normalizedValue, referenceItem, paramName, value);
      } else {
        // For resistance values
        const range = maxValue - minValue;
        
        if (minValue > 0 && maxValue / minValue > 10) {
          // Use log scale for wide ranges
          const safeValue = Math.max(value, minValue * 0.1);
          const safeMin = Math.max(minValue, 1e-15);
          
          const logMin = Math.log10(safeMin);
          const logMax = Math.log10(maxValue);
          const logValue = Math.log10(safeValue);
          
          const normalizedValue = (logValue - logMin) / (logMax - logMin);
          return applyNormalizationEffects(normalizedValue, referenceItem, paramName, value);
        } else {
          // Use linear scale for narrower ranges
          const normalizedValue = (value - minValue) / range;
          return applyNormalizationEffects(normalizedValue, referenceItem, paramName, value);
        }
      }
    },
    [applyNormalizationEffects]
  );

  // Generate a key for the reference model that will change when its parameters change
  const referenceKey = useMemo(() => {
    if (!referenceItem) return 'no-reference';
    
    // Create a string that includes all parameter values - this will change when any parameter changes
    const { Rs, ra, ca, rb, cb } = normalizeParameters(referenceItem.parameters);
    return `ref-${referenceItem.id}-${Rs}-${ra}-${ca}-${rb}-${cb}`;
  }, [referenceItem]);

  // Calculate spiderData from meshItems
  const spiderData = useMemo(() => {
    if (!meshItems.length) {
      return { 
        data: [],
        states: [],
        referenceValues: {},
        parameters: [] 
      };
    }
    
    // Sort items by resnorm to ensure consistent rendering and assignment of colors
    const sortedItems = [...meshItems].sort((a, b) => {
      // Reference item (or lowest resnorm) always comes first
      if (a.id === referenceItem?.id) return -1;
      if (b.id === referenceItem?.id) return 1;
      
      const aResnorm = a.resnorm || Infinity;
      const bResnorm = b.resnorm || Infinity;
      return aResnorm - bResnorm;
    });

    // Get the color generator function
    const getColor = createResnormColorScale(resnormRange);
    
    // Create radar entries for each mesh item
    const radarStates: SpiderState[] = sortedItems.map((item, index) => {
      // Use custom color if provided, otherwise generate based on resnorm
      const color = item.id === referenceItem?.id
        ? '#000000' // Black for reference/ground truth
        : item.color || getColor(item.resnorm || 0);
      
      // Calculate appropriate opacity based on resnorm
      const opacity = calculateLineOpacity(item.resnorm, item.id === referenceItem?.id);
        
      return {
        name: item.name || `Model ${index + 1}`,
        color,
        dataKey: `model_${item.id || index}`,
        resnorm: item.resnorm,
        opacity
      };
    });
    
    // Initialize radar data with all parameters
    // First, identify all possible parameters from all items
    const parameterKeys = new Set<string>();
    for (const item of sortedItems) {
      if (item.parameters) {
        // Normalize parameter keys - replace rBlank with Rs
        Object.keys(item.parameters).forEach(key => {
          if (key === 'rBlank') {
            parameterKeys.add('Rs');
          } else if (key !== 'frequency_range') {
            parameterKeys.add(key);
          }
        });
      }
    }
    
    // Filter out any frequency_range parameter
    parameterKeys.delete('frequency_range');
    
    // Create parameter info with display names, units, and electrophysiological significance
    const paramInfo: Parameter[] = Array.from(parameterKeys).map(key => {
      // Customize display name and unit based on parameter
      let displayName = key;
      let unit = '';
      
      switch (key) {
        case 'Rs':
          displayName = 'R_s';
          unit = 'Ω';
          break;
        case 'ra':
          displayName = 'R_a';
          unit = 'Ω';
          break;
        case 'ca':
          displayName = 'C_a';
          unit = 'μF';
          break;
        case 'rb':
          displayName = 'R_b';
          unit = 'Ω';
          break;
        case 'cb':
          displayName = 'C_b';
          unit = 'μF';
          break;
      }
      
      return { key, displayName, unit };
    });
    
    // Pre-calculate parameter values for each item to avoid doing it repeatedly during render
    const parameterValues: Record<string, Record<string, number>> = {};
    
    // For each parameter
    paramInfo.forEach(param => {
      parameterValues[param.key] = {};
      
      // For each state (item)
      radarStates.forEach(state => {
        const meshItem = sortedItems.find((item: ModelSnapshot) => `model_${item.id || ''}` === state.dataKey);
        if (meshItem && meshItem.parameters) {
          const normalizedParams = normalizeParameters(meshItem.parameters);
          
          // Get value based on parameter key
          let value: number | undefined;
          
          // Special handling for Rs
          if (param.key === 'Rs') {
            value = normalizedParams.Rs;
          } else {
            value = normalizedParams[param.key as keyof typeof normalizedParams];
          }
          
          if (value !== undefined && typeof value === 'number' && Number.isFinite(value)) {
            const normalizedValue = normalizeParameterValue(
              value, 
              param.key, 
              referenceItem, 
              meshItems
            );
            
            // Ensure the value is not NaN or infinite
            parameterValues[param.key][state.dataKey] = Number.isFinite(normalizedValue) ? normalizedValue : 0.5;
          } else {
            parameterValues[param.key][state.dataKey] = 0.5; // Default for missing/invalid
          }
        }
      });
    });
    
    // Create radar data structure with precalculated values
    const radarData = paramInfo.map(param => {
      const dataPoint: Record<string, string | number> = {
        parameter: param.key,
        displayName: param.displayName,
        unit: param.unit || '',
      };
      
      // Add data for each state (already normalized)
      radarStates.forEach(state => {
        if (parameterValues[param.key] && parameterValues[param.key][state.dataKey]) {
          dataPoint[state.dataKey] = parameterValues[param.key][state.dataKey];
        } else {
          dataPoint[state.dataKey] = 0.5; // Fallback to middle value
        }
      });
      
      return dataPoint;
    });
    
    // Get reference values from referenceItem or first item
    const referenceValues = referenceItem
      ? { ...normalizeParameters(referenceItem.parameters) }
      : sortedItems.length > 0
        ? { ...normalizeParameters(sortedItems[0].parameters) }
        : {};
    
    return {
      data: radarData,
      states: radarStates,
      referenceValues,
      parameters: paramInfo
    };
  }, [meshItems, referenceItem, resnormRange, normalizeParameterValue, calculateLineOpacity]);
  
  // Custom tick component for radar axis labels
  const CustomTick = () => {
    // Don't render anything in the CustomTick component
    // We'll handle all axis labels in the AxisValueIndicators component
    return null;
  };

  // Render axis values in a more visible way
  const AxisValueIndicators = () => {
    if (!meshItems.length || !referenceItem) return null;
    
    // Define exact positions for each parameter label instead of calculating angles
    const parameterPositions = [
      { key: 'Rs', className: 'axis-top', position: { x: '50%', y: '5%' } },
      { key: 'ra', className: 'axis-top-right', position: { x: '85%', y: '28%' } },
      { key: 'ca', className: 'axis-bottom-right', position: { x: '75%', y: '80%' } },
      { key: 'rb', className: 'axis-bottom-left', position: { x: '25%', y: '80%' } },
      { key: 'cb', className: 'axis-top-left', position: { x: '15%', y: '28%' } },
    ];
    
    // Map parameter keys to display names
    const displayNames: Record<string, string> = {
      'Rs': 'Rs (Shunt)',
      'ra': 'Ra (Apical)',
      'ca': 'Ca (Apical)',
      'rb': 'Rb (Basal)',
      'cb': 'Cb (Basal)',
    };
    
    // Calculate min/max values for each parameter across all models
    const paramRanges: Record<string, {min: number, max: number, ref: number}> = {};
    
    // Use reference item values
    const refParams = normalizeParameters(referenceItem.parameters);
    
    parameterPositions.forEach(pos => {
      const key = pos.key;
      const refValue = refParams[key as keyof typeof refParams];
      
      // Find min/max for this parameter across all models
      const values = meshItems
        .map(item => {
          const params = normalizeParameters(item.parameters);
          return params[key as keyof typeof params];
        })
        .filter(val => val !== undefined && Number.isFinite(val)) as number[];
      
      const min = Math.min(...values);
      const max = Math.max(...values);
      
      paramRanges[key] = {
        min,
        max,
        ref: refValue || 0
      };
    });
    
    // Fix the type guard function
    const isValidParameterKey = (k: string, obj: { parameters?: Record<string, unknown> }): boolean => {
      return obj.parameters !== undefined && typeof obj.parameters === 'object' && k in obj.parameters;
    };

    // Update the generateTicks function to return correct tick values with value and position properties
    const generateTicks = (key: string) => {
      // Get reference value for normalization
      const refValue = refParams[key as keyof typeof refParams] || 0;
      
      // Skip if refValue is an array or not a number
      if (Array.isArray(refValue) || typeof refValue !== 'number') {
        return [];
      }
      
      // Find parameter min/max across all mesh items
      const allItems = meshItems.filter(i => 
        isValidParameterKey(key, i) && 
        i.parameters[key as keyof typeof i.parameters] !== undefined && 
        !Array.isArray(i.parameters[key as keyof typeof i.parameters]) && 
        typeof i.parameters[key as keyof typeof i.parameters] === 'number'
      );
      
      if (allItems.length === 0) return [];
      
      const values = allItems.map(i => i.parameters[key as keyof typeof i.parameters] as number);
      
      // Calculate a reasonable range - always include reference value plus extra room for context
      const min = Math.min(...values);
      const max = Math.max(...values);
      
      // Add the reference value to ensure it's included
      const valueRange = [min, refValue, max].sort((a, b) => a - b);
      const rangeMin = valueRange[0];
      const rangeMax = valueRange[valueRange.length - 1];
      
      // Generate tick values (5 is usually a good number)
      const tickStep = (rangeMax - rangeMin) / 4;
      
      // Return objects with value and position properties
      return [
        { value: rangeMin, position: 0 },
        { value: rangeMin + tickStep, position: 0.25 },
        { value: rangeMin + 2 * tickStep, position: 0.5 },
        { value: rangeMin + 3 * tickStep, position: 0.75 },
        { value: rangeMax, position: 1 }
      ];
    };

    return (
      <div className="absolute inset-0 pointer-events-none" aria-hidden="true">
        {/* Parameter axis labels with reference values */}
        {parameterPositions.map(position => {
          const paramKey = position.key;
          const refValue = referenceItem.parameters[paramKey as keyof typeof referenceItem.parameters];
          
          if (refValue === undefined) return null;
          
          return (
            <div 
              key={paramKey} 
              className={`spider-parameter-label ${position.className}`}
                  style={{
                position: 'absolute', 
                left: position.position.x, 
                top: position.position.y, 
                transform: 'translate(-50%, -50%)' 
              }}
            >
              <div className="spider-parameter-name">{displayNames[paramKey]}</div>
              <div className="spider-parameter-value">{formatParameterValue(refValue, paramKey)}</div>
            </div>
          );
        })}

        {/* Add tick marks with normalized values along axes */}
        <svg className="absolute inset-0 w-full h-full" aria-hidden="true">
          {/* Axis lines connecting center to each parameter */}
          <g className="axis-lines">
            <line x1="50%" y1="50%" x2="50%" y2="10%" stroke="var(--spider-grid-color)" strokeOpacity="0.6" strokeWidth="1.5" />
            <line x1="50%" y1="50%" x2="82%" y2="30%" stroke="var(--spider-grid-color)" strokeOpacity="0.6" strokeWidth="1.5" />
            <line x1="50%" y1="50%" x2="73%" y2="75%" stroke="var(--spider-grid-color)" strokeOpacity="0.6" strokeWidth="1.5" />
            <line x1="50%" y1="50%" x2="27%" y2="75%" stroke="var(--spider-grid-color)" strokeOpacity="0.6" strokeWidth="1.5" />
            <line x1="50%" y1="50%" x2="18%" y2="30%" stroke="var(--spider-grid-color)" strokeOpacity="0.6" strokeWidth="1.5" />
          </g>

          {/* Dynamic tick marks for Rs (top) */}
          <g className="axis-ticks">
            {generateTicks('Rs').map((tick, idx) => {
              // Calculate position along the axis line
              const yPos = 50 - ((50 - 10) * tick.position);
              return (
                <g key={`Rs-tick-${idx}`} transform={`translate(50%, ${yPos}%)`}>
                  <circle cx="0" cy="0" r="2" fill="var(--spider-grid-color)" opacity="0.8" />
                  <text x="8" y="0" fontSize="9" fill="var(--circuit-text-secondary)" dominantBaseline="middle" className="tick-text">
                    {formatParameterValue(tick.value, 'Rs')}
                  </text>
                </g>
              );
            })}
          </g>

          {/* Dynamic tick marks for Ra (top-right) */}
          <g className="axis-ticks">
            {generateTicks('ra').map((tick, idx) => {
              // Calculate position along the axis line
              const xPos = 50 + ((82 - 50) * tick.position);
              const yPos = 50 - ((50 - 30) * tick.position);
              return (
                <g key={`ra-tick-${idx}`} transform={`translate(${xPos}%, ${yPos}%)`}>
                  <circle cx="0" cy="0" r="2" fill="var(--spider-grid-color)" opacity="0.8" />
                  <text x="6" y="-3" fontSize="9" fill="var(--circuit-text-secondary)" dominantBaseline="middle" className="tick-text">
                    {formatParameterValue(tick.value, 'ra')}
                  </text>
                </g>
              );
            })}
          </g>

          {/* Dynamic tick marks for Ca (bottom-right) */}
          <g className="axis-ticks">
            {generateTicks('ca').map((tick, idx) => {
              // Calculate position along the axis line
              const xPos = 50 + ((73 - 50) * tick.position);
              const yPos = 50 + ((75 - 50) * tick.position);
              return (
                <g key={`ca-tick-${idx}`} transform={`translate(${xPos}%, ${yPos}%)`}>
                  <circle cx="0" cy="0" r="2" fill="var(--spider-grid-color)" opacity="0.8" />
                  <text x="6" y="0" fontSize="9" fill="var(--circuit-text-secondary)" dominantBaseline="middle" className="tick-text">
                    {formatParameterValue(tick.value, 'ca')}
                  </text>
                </g>
              );
            })}
          </g>

          {/* Dynamic tick marks for Rb (bottom-left) */}
          <g className="axis-ticks">
            {generateTicks('rb').map((tick, idx) => {
              // Calculate position along the axis line
              const xPos = 50 - ((50 - 27) * tick.position);
              const yPos = 50 + ((75 - 50) * tick.position);
              return (
                <g key={`rb-tick-${idx}`} transform={`translate(${xPos}%, ${yPos}%)`}>
                  <circle cx="0" cy="0" r="2" fill="var(--spider-grid-color)" opacity="0.8" />
                  <text x="-6" y="0" fontSize="9" fill="var(--circuit-text-secondary)" dominantBaseline="middle" textAnchor="end" className="tick-text">
                    {formatParameterValue(tick.value, 'rb')}
                  </text>
                </g>
              );
            })}
          </g>

          {/* Dynamic tick marks for Cb (top-left) */}
          <g className="axis-ticks">
            {generateTicks('cb').map((tick, idx) => {
              // Calculate position along the axis line
              const xPos = 50 - ((50 - 18) * tick.position);
              const yPos = 50 - ((50 - 30) * tick.position);
              return (
                <g key={`cb-tick-${idx}`} transform={`translate(${xPos}%, ${yPos}%)`}>
                  <circle cx="0" cy="0" r="2" fill="var(--spider-grid-color)" opacity="0.8" />
                  <text x="-6" y="-3" fontSize="9" fill="var(--circuit-text-secondary)" dominantBaseline="middle" textAnchor="end" className="tick-text">
                    {formatParameterValue(tick.value, 'cb')}
                  </text>
                </g>
              );
            })}
          </g>

          {/* Center point */}
          <g className="center-point">
            <circle cx="50%" cy="50%" r="3" fill="var(--primary)" opacity="0.8" />
          </g>
        </svg>
      </div>
    );
  };

  return (
    <div className="h-full w-full">
          {meshItems.length === 0 ? (
        <div className="flex items-center justify-center h-[380px] text-circuit-text-secondary">
              <p>No data available for visualization</p>
            </div>
          ) : (
        <div className="h-full spider-visualization relative">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart
                  cx="50%"
                  cy="50%"
                  outerRadius="75%"
                  data={spiderData.data}
                  key={referenceKey}
                >
                  {/* SVG definitions for filters */}
                  <defs>
                    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                      <feGaussianBlur stdDeviation="2.5" result="coloredBlur"/>
                      <feMerge>
                        <feMergeNode in="coloredBlur"/>
                        <feMergeNode in="SourceGraphic"/>
                      </feMerge>
                    </filter>
                  </defs>
                  
                  {/* PolarGrid represents the parameter grid, values are normalized around the reference shape */}
                  <PolarGrid 
                    gridType="polygon"
                    strokeOpacity={0.4}
                    stroke="var(--spider-grid-color)"
                    radialLines={true}
                    className="spider-grid-line"
                  />
                  <PolarAngleAxis
                    dataKey="displayName"
                    tick={<CustomTick />}
                    axisLineType="polygon"
                    tickLine={false}
                    className="spider-axis-line"
                  />
                  <PolarRadiusAxis 
                    tick={false} 
                    axisLine={false}
                    domain={[0, 1]}
                  />
                  
                  {/* Radar lines for all non-reference models first */}
                  {spiderData.states
                    .filter(state => state.dataKey !== `model_${referenceItem?.id || ''}`)
                    // Sort by resnorm so better fits are on top
                    .sort((a, b) => (a.resnorm || Infinity) - (b.resnorm || Infinity))
                    .map((state) => {
                      const hasValidData = spiderData.data.every(dataPoint => 
                        state.dataKey in dataPoint && 
                        typeof dataPoint[state.dataKey] === 'number' &&
                        Number.isFinite(dataPoint[state.dataKey] as number)
                      );
                      
                      if (!hasValidData) return null;
                      
                      return (
                        <Radar
                          key={state.dataKey}
                          name={state.name}
                          dataKey={state.dataKey}
                          stroke={state.color}
                          fill={state.color}
                          fillOpacity={0}
                          strokeWidth={1.5}
                          strokeOpacity={state.opacity}
                          strokeDasharray="0"
                          dot={false}
                          isAnimationActive={false}
                        />
                      );
                    })}
                    
                  {/* Reference model rendered last (on top) */}
                  {(() => {
                    const referenceState = spiderData.states.find(
                      state => state.dataKey === `model_${referenceItem?.id || ''}`
                    );
                    
                    if (!referenceState) return null;
                    
                    const hasValidData = spiderData.data.every(dataPoint => 
                      referenceState.dataKey in dataPoint && 
                      typeof dataPoint[referenceState.dataKey] === 'number' &&
                      Number.isFinite(dataPoint[referenceState.dataKey] as number)
                    );
                    
                    if (!hasValidData) return null;
                    
                    return (
                      <Radar
                        key={referenceState.dataKey}
                        name="Reference Model"
                        dataKey={referenceState.dataKey}
                        stroke="var(--spider-ref-color)"
                        fill="var(--spider-ref-color)"
                        fillOpacity={0}
                        strokeWidth={3} // Increased for maximum visibility
                        strokeOpacity={1}
                        strokeDasharray="5,3" // Adjust dash pattern for better visibility
                        dot={false}
                        isAnimationActive={false}
                        style={{ zIndex: 1000, filter: 'url(#glow)' }}
                      />
                    );
                  })()}
                </RadarChart>
              </ResponsiveContainer>
          
          {/* Only use the AxisValueIndicators component for all labels */}
          <AxisValueIndicators />
          
          {/* Reference model indicator */}
          <div className="absolute bottom-2 right-2 bg-neutral-900/80 rounded px-3 py-1.5 text-xs border border-neutral-600">
            <div className="flex items-center">
              <div className="w-4 h-0.5 bg-white mr-2 opacity-90 dash-pattern"></div>
              <span className="text-white font-medium">Reference Model</span>
            </div>
          </div>
            </div>
          )}
    </div>
  );
};