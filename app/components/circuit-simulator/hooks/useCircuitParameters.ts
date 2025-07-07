import { useState, useEffect, useCallback } from 'react';
import { CircuitParameters } from '../types/parameters';

export const useCircuitParameters = (initialMinFreq: number, initialMaxFreq: number) => {
  // Frequency control state
  const [minFreq, setMinFreq] = useState<number>(initialMinFreq);
  const [maxFreq, setMaxFreq] = useState<number>(initialMaxFreq);
  const [numPoints, setNumPoints] = useState<number>(20);
  const [frequencyPoints, setFrequencyPoints] = useState<number[]>([]);

  // Circuit parameters state
  const [parameters, setParameters] = useState<CircuitParameters>({
    Rs: 50,
    Ra: 100,
    Ca: 0.5e-6, // 0.5 microfarads (converted to farads)
    Rb: 100,
    Cb: 0.5e-6, // 0.5 microfarads (converted to farads)
    frequency_range: [0.1, 10000]
  });

  // Reference parameters state
  const [referenceParams, setReferenceParams] = useState<CircuitParameters>({
    Rs: 24,
    Ra: 500,
    Ca: 0.5e-6, // 0.5 microfarads (converted to farads)
    Rb: 500,
    Cb: 0.5e-6, // 0.5 microfarads (converted to farads)
    frequency_range: [initialMinFreq, initialMaxFreq]
  });

  // Ground truth parameters state
  const [groundTruthParams, setGroundTruthParams] = useState<CircuitParameters>({
    Rs: 24,
    Ra: 500,
    Ca: 0.5e-6, // 0.5 microfarads (converted to farads)
    Rb: 500,
    Cb: 0.5e-6, // 0.5 microfarads (converted to farads)
    frequency_range: [initialMinFreq, initialMaxFreq]
  });

  // Parameter change tracking
  const [parameterChanged, setParameterChanged] = useState<boolean>(false);

  // Initialize parameters to 50% of ranges
  const initializeParameters = useCallback(() => {
    const rsRange = { min: 10, max: 10000 };
    const raRange = { min: 10, max: 10000 };
    const rbRange = { min: 10, max: 10000 };
    const caRange = { min: 0.1e-6, max: 50e-6 };
    const cbRange = { min: 0.1e-6, max: 50e-6 };

    const rs50 = rsRange.min + (rsRange.max - rsRange.min) * 0.5;
    const ra50 = raRange.min + (raRange.max - raRange.min) * 0.5;
    const rb50 = rbRange.min + (rbRange.max - rbRange.min) * 0.5;
    const ca50 = caRange.min + (caRange.max - caRange.min) * 0.5;
    const cb50 = cbRange.min + (cbRange.max - cbRange.min) * 0.5;

    setGroundTruthParams({
      Rs: rs50,
      Ra: ra50,
      Ca: ca50,
      Rb: rb50,
      Cb: cb50,
      frequency_range: [minFreq, maxFreq]
    });
  }, [minFreq, maxFreq]);

  // Update reference parameters when ground truth changes
  useEffect(() => {
    if (referenceParams.Rs === 24 && 
        referenceParams.Ra === 500 && 
        referenceParams.Ca === 0.5e-6 && 
        referenceParams.Rb === 500 && 
        referenceParams.Cb === 0.5e-6 && 
        referenceParams.frequency_range.length === 2 &&
        referenceParams.frequency_range[0] === minFreq &&
        referenceParams.frequency_range[1] === maxFreq) {
      setReferenceParams({
        Rs: groundTruthParams.Rs,
        Ra: groundTruthParams.Ra,
        Ca: groundTruthParams.Ca,
        Rb: groundTruthParams.Rb,
        Cb: groundTruthParams.Cb,
        frequency_range: groundTruthParams.frequency_range
      });
    }
  }, [groundTruthParams, minFreq, maxFreq, referenceParams]);

  return {
    // Frequency state
    minFreq,
    setMinFreq,
    maxFreq,
    setMaxFreq,
    numPoints,
    setNumPoints,
    frequencyPoints,
    setFrequencyPoints,

    // Parameters state
    parameters,
    setParameters,
    referenceParams,
    setReferenceParams,
    groundTruthParams,
    setGroundTruthParams,

    // Parameter change tracking
    parameterChanged,
    setParameterChanged,

    // Utility functions
    initializeParameters,
  };
};