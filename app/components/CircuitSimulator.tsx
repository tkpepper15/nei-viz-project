"use client";

import React, { useState, useEffect, useCallback } from 'react';
import 'katex/dist/katex.min.css';
import { CircuitParameters } from './circuit-simulator/utils/impedance';
import { BackendMeshPoint, ModelSnapshot, ResnormGroup } from './circuit-simulator/utils/types';
import { generateGridPoints } from './circuit-simulator/utils';
import { ParamSlider } from './circuit-simulator';
import { ImpedancePoint as TypesImpedancePoint } from './circuit-simulator/utils/types';

// Add imports for the new tab components at the top of the file
import MathDetailsTab from './circuit-simulator/MathDetailsTab';
import DataTableTab from './circuit-simulator/DataTableTab';
import VisualizerTab from './circuit-simulator/VisualizerTab';

// Remove empty interface and replace with type
type CircuitSimulatorProps = Record<string, never>;

export const CircuitSimulator: React.FC<CircuitSimulatorProps> = () => {
  // Add frequency control state
  const [minFreq, setMinFreq] = useState<number>(0.1); // 0.1 Hz
  const [maxFreq, setMaxFreq] = useState<number>(10000); // 10 kHz
  const [numPoints, setNumPoints] = useState<number>(20); // Default number of frequency points
  const [frequencyPoints, setFrequencyPoints] = useState<number[]>([]);

  // State for the circuit simulator
  const [gridResults, setGridResults] = useState<BackendMeshPoint[]>([]);
  const [gridResultsWithIds, setGridResultsWithIds] = useState<(BackendMeshPoint & { id: number })[]>([]);
  const [logMessages, setLogMessages] = useState<{time: string; message: string}[]>([]);
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [gridSettingsOpen, setGridSettingsOpen] = useState<boolean>(true);
  const [circuitParamsOpen, setCircuitParamsOpen] = useState<boolean>(true);
  const [gridSize, setGridSize] = useState<number>(3);
  const [gridError, setGridError] = useState<string | null>(null);
  const [isComputingGrid, setIsComputingGrid] = useState<boolean>(false);
  const [resnormGroups, setResnormGroups] = useState<ResnormGroup[]>([]);
  const [hiddenGroups, setHiddenGroups] = useState<number[]>([]);
  const [currentTab, setCurrentTab] = useState<'toolbox' | 'activity'>('toolbox');
  const [visualizationTab, setVisualizationTab] = useState<'visualizer' | 'math' | 'data'>('visualizer');
  const [parameterChanged, setParameterChanged] = useState<boolean>(false);
  // Track if reference model was manually hidden
  const [manuallyHidden, setManuallyHidden] = useState<boolean>(false);
  
  // Visualization settings - these are now passed to child components but setters not used
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [opacityLevel, setOpacityLevel] = useState<number>(0.7);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [logScalar, setLogScalar] = useState<number>(1.0);
  
  const [referenceParams, setReferenceParams] = useState<CircuitParameters>({
    Rs: 24,
    Ra: 500,
    Ca: 3e-6,
    Rb: 500,
    Cb: 3e-6,
    frequency_range: [minFreq, maxFreq]
  });
  
  // Reference model state
  const [referenceModelId, setReferenceModelId] = useState<string | null>(null);
  const [referenceModel, setReferenceModel] = useState<ModelSnapshot | null>(null);
  
  // Ground truth parameters state
  const [groundTruthParams, setGroundTruthParams] = useState<CircuitParameters>({
    Rs: 24,
    Ra: 500,
    Ca: 3e-6,
    Rb: 500,
    Cb: 3e-6,
    frequency_range: [minFreq, maxFreq]
  });
  
  // Use a single parameters state object
  const [parameters, setParameters] = useState<CircuitParameters>({
    Rs: 50,
    Ra: 100,
    Ca: 0.000001,
    Rb: 100,
    Cb: 0.000001,
    frequency_range: [0.1, 10000]
  });
  
  // Update status message to also store in log history
  const updateStatusMessage = useCallback((message: string) => {
    setStatusMessage(message);
    const timestamp = new Date().toLocaleTimeString();
    // Categorize messages for better visibility in log
    let formattedMessage = message;
    
    // Add category tags based on message content
    if (message.includes('Comput') || message.includes('Grid')) {
      formattedMessage = `[Grid] ${message}`;
    } else if (message.includes('Parameters') || message.includes('parameter')) {
      formattedMessage = `[Params] ${message}`;
    } else if (message.includes('model') || message.includes('Model')) {
      formattedMessage = `[Visual] ${message}`;
    } else if (message.includes('Sort')) {
      formattedMessage = `[Table] ${message}`;
    }
    
    setLogMessages(prev => [...prev.slice(-49), { time: timestamp, message: formattedMessage }]);
    
    // Auto-switch to activity tab for certain important operations
    if (message.includes('Grid computed successfully') || message.includes('Error')) {
      setCurrentTab('activity');
    }
  }, []);

  // Initialize parameters
  useEffect(() => {
    // Set default starting values
    setGroundTruthParams({
      Rs: 24,
      Ra: 500,
      Ca: 3e-6,
      Rb: 500,
      Cb: 3e-6,
      frequency_range: [minFreq, maxFreq]
    });
    
    // Initial reference params will be set after the state updates
    updateStatusMessage('Initialized with default parameters');
  }, [updateStatusMessage, minFreq, maxFreq]);

  // Update reference parameters when slider values are initialized
  useEffect(() => {
    // Only create reference once on initial load
    if (referenceParams.Rs === 24 && 
        referenceParams.Ra === 500 && 
        referenceParams.Ca === 3e-6 && 
        referenceParams.Rb === 500 && 
        referenceParams.Cb === 3e-6 && 
        referenceParams.frequency_range.length === 2 &&
        referenceParams.frequency_range[0] === minFreq &&
        referenceParams.frequency_range[1] === maxFreq) {
      // Create the reference object directly without calling the function
      setReferenceParams({
        Rs: groundTruthParams.Rs,
        Ra: groundTruthParams.Ra,
        Ca: groundTruthParams.Ca,
        Rb: groundTruthParams.Rb,
        Cb: groundTruthParams.Cb,
        frequency_range: groundTruthParams.frequency_range
      });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);  // Run only once on mount
  
  // We'll calculate TER and TEC in the component where they're needed
  
  // Define the ImpedancePoint interface if not already defined
  interface ImpedancePoint {
    frequency: number;
    real: number;
    imaginary: number;
    magnitude: number;
    phase: number;
  }

  // Create a reference model (current circuit parameters)
  const createReferenceModel = useCallback((): ModelSnapshot => {
    // Generate frequencies to compute impedance at (logarithmically spaced)
    let freqs: number[];
    if (frequencyPoints.length > 0) {
      // Use the pre-calculated frequency points if available
      freqs = [...frequencyPoints];
    } else {
      // Generate logarithmically spaced points from min to max
      freqs = [];
      const logMin = Math.log10(minFreq);
      const logMax = Math.log10(maxFreq);
      const logStep = (logMax - logMin) / (numPoints - 1);
      
      for (let i = 0; i < numPoints; i++) {
        const logValue = logMin + i * logStep;
        const frequency = Math.pow(10, logValue);
        freqs.push(frequency);
      }
      
      // Don't update state here - this would cause re-renders
      // Instead, just use the locally calculated frequencies
    }
    
    // Use groundTruthParams instead of parameters for the reference model
    const { Rs, Ra, Ca, Rb, Cb } = groundTruthParams;
    
    // Compute impedance at each frequency
    const impedanceData: ImpedancePoint[] = [];
    
    // Generate detailed logs for the first calculation
    const isLogging = logMessages.length === 0;
    if (isLogging) {
      updateStatusMessage(`MATH: Calculating impedance for reference model at ${freqs.length} frequency points`);
      updateStatusMessage(`MATH: Circuit parameters: Rs=${Rs.toFixed(2)}Ω, Ra=${Ra.toFixed(2)}Ω, Ca=${Ca.toExponential(2)}F, Rb=${Rb.toFixed(2)}Ω, Cb=${Cb.toExponential(2)}F`);
    }
    
    for (const f of freqs) {
      // Calculate impedance for a single frequency
      const impedance = calculateCircuitImpedance(
        { Rs, Ra, Ca, Rb, Cb, frequency_range: [minFreq, maxFreq] }, 
        f
      );
      
      // Log first few frequency points for visibility
      if (isLogging && impedanceData.length < 3) {
        updateStatusMessage(`MATH: At f=${f.toFixed(2)} Hz: Z = ${impedance.real.toFixed(2)} - j${Math.abs(impedance.imaginary).toFixed(2)} Ω`);
      }
      
      impedanceData.push({
        frequency: f,
        real: impedance.real,
        imaginary: impedance.imaginary,
        magnitude: Math.sqrt(impedance.real * impedance.real + impedance.imaginary * impedance.imaginary),
        phase: Math.atan2(impedance.imaginary, impedance.real) * (180 / Math.PI)
      });
    }
    
    return {
      id: 'dynamic-reference',
      name: 'Reference',
      data: impedanceData,
      parameters: { Rs, Ra, Ca, Rb, Cb, frequency_range: [minFreq, maxFreq] },
      timestamp: Date.now(),
      color: '#FFFFFF',
      isVisible: true,
      opacity: 1,
      resnorm: 0
    };
  }, [groundTruthParams, minFreq, maxFreq, numPoints, frequencyPoints, logMessages.length, updateStatusMessage]);

  // Function to calculate impedance at a single frequency
  const calculateCircuitImpedance = (params: CircuitParameters, frequency: number) => {
    const { Rs, Ra, Ca, Rb, Cb } = params;
    const omega = 2 * Math.PI * frequency;
    
    // Calculate impedance of apical membrane (Ra || Ca)
    const Za_real = Ra / (1 + Math.pow(omega * Ra * Ca, 2));
    const Za_imag = -omega * Ra * Ra * Ca / (1 + Math.pow(omega * Ra * Ca, 2));
    
    // Calculate impedance of basal membrane (Rb || Cb)
    const Zb_real = Rb / (1 + Math.pow(omega * Rb * Cb, 2));
    const Zb_imag = -omega * Rb * Rb * Cb / (1 + Math.pow(omega * Rb * Cb, 2));
    
    // Calculate total impedance
    const real = Rs + Za_real + Zb_real;
    const imaginary = Za_imag + Zb_imag;
    
    return { real, imaginary };
  };

  // Function to handle reference model toggle
  const toggleReferenceModel = useCallback(() => {
    if (referenceModelId === 'dynamic-reference') {
      // Hide reference model and mark as manually hidden
      setReferenceModelId(null);
      setReferenceModel(null);
      setManuallyHidden(true);
      updateStatusMessage('Reference model hidden');
    } else {
      // Show reference model - create a new one
      const newReferenceModel = createReferenceModel();
      setReferenceModel(newReferenceModel);
      setReferenceModelId('dynamic-reference');
      setManuallyHidden(false);
      updateStatusMessage('Reference model shown');
    }
  }, [referenceModelId, createReferenceModel, updateStatusMessage, setManuallyHidden]);

  // Create and show reference model by default
  useEffect(() => {
    // Create the reference model when groundTruthParams are initialized
    if (groundTruthParams.Rs && !referenceModelId && !manuallyHidden) {
      const newReferenceModel = createReferenceModel();
      setReferenceModel(newReferenceModel);
      setReferenceModelId('dynamic-reference');
      updateStatusMessage('Reference model created with current parameters');
    }
  }, [groundTruthParams.Rs, referenceModelId, createReferenceModel, updateStatusMessage, manuallyHidden]);

  // Update reference model when parameters change
  useEffect(() => {
    if (referenceModelId === 'dynamic-reference' && !manuallyHidden) {
      const updatedModel = createReferenceModel();
      setReferenceModel(updatedModel);
      updateStatusMessage('Reference model updated with current parameters');
    }
  }, [groundTruthParams, createReferenceModel, referenceModelId, updateStatusMessage, manuallyHidden]);
  
  // Add event listeners for custom events from VisualizerTab
  useEffect(() => {
    // Handler for reference model toggle
    const handleToggleReference = (event: Event) => {
      // Check if this is a forced state change
      const customEvent = event as CustomEvent;
      const forceState = customEvent.detail?.forceState;
      
      if (forceState !== undefined) {
        // Handle forced state
        if (forceState) {
          // Force show reference model
          if (!referenceModelId) {
            const newReferenceModel = createReferenceModel();
            setReferenceModel(newReferenceModel);
            setReferenceModelId('dynamic-reference');
            setManuallyHidden(false);
            updateStatusMessage('Reference model shown');
          }
        } else {
          // Force hide reference model
          if (referenceModelId) {
            setReferenceModelId(null);
            setReferenceModel(null);
            setManuallyHidden(true);
            updateStatusMessage('Reference model hidden');
          }
        }
      } else {
        // Regular toggle
        toggleReferenceModel();
      }
    };
    
    // Handler for resnorm group toggle
    const handleToggleResnormGroup = (event: Event) => {
      const customEvent = event as CustomEvent;
      const { groupIndex } = customEvent.detail;
      
      // Toggle this group's visibility
      setHiddenGroups(prev => {
        if (prev.includes(groupIndex)) {
          return prev.filter(i => i !== groupIndex);
        } else {
          return [...prev, groupIndex];
        }
      });
      
      // Update status message
      const groupName = 
        groupIndex === 0 ? 'Very Good Fit' : 
        groupIndex === 1 ? 'Good Fit' : 
        groupIndex === 2 ? 'Moderate Fit' : 'Poor Fit';
      
      updateStatusMessage(
        hiddenGroups.includes(groupIndex) 
          ? `Showing ${groupName} category`
          : `Hiding ${groupName} category`
      );
    };
    
    // Add event listeners
    window.addEventListener('toggleReferenceModel', handleToggleReference);
    window.addEventListener('toggleResnormGroup', handleToggleResnormGroup);
    
    // Clean up event listeners on unmount
    return () => {
      window.removeEventListener('toggleReferenceModel', handleToggleReference);
      window.removeEventListener('toggleResnormGroup', handleToggleResnormGroup);
    };
  }, [toggleReferenceModel, hiddenGroups, updateStatusMessage, referenceModelId]);
  
  // Helper function to map BackendMeshPoint to ModelSnapshot
  const mapBackendMeshToSnapshot = (meshPoint: BackendMeshPoint, index: number): ModelSnapshot => {
    // Determine color based on resnorm
    let color = '#3B82F6'; // Default blue
    
    // Simple color mapping based on resnorm thresholds
    // These should align with the thresholds used for resnorm groups
    if (meshPoint.resnorm !== undefined) {
      if (meshPoint.resnorm <= 0.001) {
        color = '#10B981'; // Green - Very good
      } else if (meshPoint.resnorm <= 0.01) {
        color = '#3B82F6'; // Blue - Good
      } else if (meshPoint.resnorm <= 0.1) {
        color = '#F59E0B'; // Amber - Moderate
      } else {
        color = '#EF4444'; // Red - Poor
      }
    }
    
    return {
      id: `mesh-${index}`,
      name: `Mesh Point ${index}`,
      timestamp: Date.now(),
      parameters: {
        Rs: meshPoint.parameters.Rs,
        Ra: meshPoint.parameters.Ra,
        Ca: meshPoint.parameters.Ca,
        Rb: meshPoint.parameters.Rb,
        Cb: meshPoint.parameters.Cb,
        frequency_range: meshPoint.parameters.frequency_range || parameters.frequency_range
      },
      data: toImpedancePoints(meshPoint.spectrum) as TypesImpedancePoint[],
      resnorm: meshPoint.resnorm,
      color: color,
      opacity: meshPoint.alpha || 1,
      isVisible: true,
      ter: meshPoint.parameters.Ra + meshPoint.parameters.Rb
    };
  };

  // Update the grid computation to use consistent parameter names
  const handleComputeRegressionMesh = async () => {
    // Validate grid size
    if (gridSize < 2 || gridSize > 10) {
      updateStatusMessage('Points per dimension must be between 2 and 10');
      setTimeout(() => setStatusMessage(''), 3000);
      return;
    }

    setIsComputingGrid(true); // Start loading
    setGridResults([]); // Clear previous results
    setGridError(null); // Clear previous errors
    setResnormGroups([]); // Clear previous groups
    setParameterChanged(false); // Reset parameter changed flag
    setManuallyHidden(false); // Reset manually hidden flag
    
    // Clear previous logs and switch to the activity tab
    setLogMessages([]);
    setCurrentTab('activity');
    
    updateStatusMessage(`Starting grid computation with ${gridSize} points per dimension...`);
    updateStatusMessage(`MATH: Using ${gridSize} points per dimension for 5 parameters exploring full parameter space`);
    updateStatusMessage(`MATH: Computing all ${Math.pow(gridSize, 5).toLocaleString()} possible parameter combinations`);
    updateStatusMessage(`MATH: Circuit model: Randles equivalent circuit (Rs in series with RC elements)`);
    updateStatusMessage(`MATH: Z(ω) = Rs + Ra/(1+jωRaCa) + Rb/(1+jωRbCb)`);
    updateStatusMessage(`MATH: Using frequency range ${minFreq.toFixed(1)}-${maxFreq.toFixed(1)} Hz with ${numPoints} points`);
    updateStatusMessage(`MATH: Frequency range affects which parameters can be uniquely identified (underdetermined system)`);

    // Use ground truth parameters as reference
    const referenceParams: CircuitParameters = {
        Rs: groundTruthParams.Rs,
        Ra: groundTruthParams.Ra,
        Ca: groundTruthParams.Ca,
        Rb: groundTruthParams.Rb,
        Cb: groundTruthParams.Cb,
        frequency_range: parameters.frequency_range
    };

    try {
      // Generate grid points using our improved algorithm
        const gridPoints = generateGridPoints(
            {
                Rs: referenceParams.Rs,
                Ra: referenceParams.Ra,
                Ca: referenceParams.Ca,
                Rb: referenceParams.Rb,
                Cb: referenceParams.Cb,
                frequency_range: parameters.frequency_range
            },
            {
                Rs: { min: 10, max: 30 },
                Ra: { min: 100, max: 1000 },
                Ca: { min: 1e-6, max: 5e-6 },
                Rb: { min: 100, max: 1000 },
                Cb: { min: 1e-6, max: 5e-6 }
            },
            gridSize,
            updateStatusMessage
        );

        // Generate frequency array for spectrum calculations - use the correct logarithmic spacing
        const frequencyArray: number[] = [];
        const logMin = Math.log10(minFreq);
        const logMax = Math.log10(maxFreq);
        const logStep = (logMax - logMin) / (numPoints - 1);

        for (let i = 0; i < numPoints; i++) {
            const logValue = logMin + i * logStep;
            const frequency = Math.pow(10, logValue);
            frequencyArray.push(frequency);
        }

        updateStatusMessage(`MATH: Using ${frequencyArray.length} frequency points from ${minFreq.toExponential(2)} to ${maxFreq.toExponential(2)} Hz`);

        // Update parameter references in grid points
        const updatedPoints = gridPoints.map(point => ({
            ...point,
            parameters: {
                Rs: point.parameters.Rs,
                Ra: point.parameters.Ra,
                Ca: point.parameters.Ca,
                Rb: point.parameters.Rb,
                Cb: point.parameters.Cb,
                frequency_range: [minFreq, maxFreq] as [number, number] // Use current frequency range with proper typing
            }
        }));

        // Calculate impedance spectra for all points
        const pointsWithSpectra = updatedPoints.map(point => {
            const spectrum = calculateImpedanceSpectrum({
                Rs: point.parameters.Rs,
                Ra: point.parameters.Ra,
                Ca: point.parameters.Ca,
                Rb: point.parameters.Rb,
                Cb: point.parameters.Cb
            }, frequencyArray); // Use our generated frequency array
            return {
                ...point,
                spectrum
            };
        });

        // Calculate reference spectrum using ground truth parameters
        const referenceSpectrum = calculateImpedanceSpectrum({
          Rs: groundTruthParams.Rs,
          Ra: groundTruthParams.Ra,
          Ca: groundTruthParams.Ca,
          Rb: groundTruthParams.Rb,
          Cb: groundTruthParams.Cb
        }, frequencyArray); // Use same frequency array for consistency
        
        // Calculate resnorm for each point
        const pointsWithResnorm = pointsWithSpectra.map(point => {
            // Create clean parameter objects without frequency_range
            const cleanParams: CircuitParameters = {
                Rs: point.parameters.Rs,
                Ra: point.parameters.Ra,
                Ca: point.parameters.Ca,
                Rb: point.parameters.Rb,
                Cb: point.parameters.Cb,
                frequency_range: parameters.frequency_range
            };
            
            const cleanRefParams: CircuitParameters = {
                Rs: groundTruthParams.Rs,
                Ra: groundTruthParams.Ra,
                Ca: groundTruthParams.Ca,
                Rb: groundTruthParams.Rb,
                Cb: groundTruthParams.Cb,
                frequency_range: parameters.frequency_range
            };
            
            const testModel: ModelSnapshot = {
                id: `test-${Date.now()}`,
                name: 'Test Model',
                timestamp: Date.now(),
                parameters: cleanParams,
                data: toImpedancePoints(point.spectrum),
                color: '#3B82F6',
                isVisible: true,
                opacity: 1,
                resnorm: point.resnorm
            };

            const referenceModel: ModelSnapshot = {
                id: 'reference',
                name: 'Ground Truth Model',
                timestamp: Date.now(),
                parameters: cleanRefParams,
                data: toImpedancePoints(referenceSpectrum),
                color: '#10B981',
                isVisible: true,
                opacity: 1,
                resnorm: 0
            };

            const resnorm = calculateResnorm(referenceModel, testModel);
            return {
                ...point,
                resnorm
            };
        });
        
        // Sort by resnorm
        const sortedPoints = pointsWithResnorm.sort((a, b) => a.resnorm - b.resnorm);
        
        // Update state with results
        setGridResults(sortedPoints);
        updateStatusMessage(`Grid computation complete: ${sortedPoints.length} points analyzed`);
        
        // Find min/max resnorm values for scaling
        let minResnorm = Infinity;
        let maxResnorm = 0;
        sortedPoints.forEach(point => {
            if (point.resnorm > 0) { // Skip the reference point which has resnorm = 0
                minResnorm = Math.min(minResnorm, point.resnorm);
                maxResnorm = Math.max(maxResnorm, point.resnorm);
            }
        });
        
        // Log resnorm range
        updateStatusMessage(`MATH: Resnorm range: ${minResnorm.toExponential(3)} to ${maxResnorm.toExponential(3)}`);
        updateStatusMessage(`MATH: This wide range indicates the underdetermined nature of the system`);
        
        // Provide information about frequency range effect on resnorm
        if (maxFreq - minFreq < 100) {
          updateStatusMessage(`MATH: Narrow frequency range (${minFreq}-${maxFreq} Hz) makes the system more underdetermined`);
          updateStatusMessage(`MATH: Many parameter sets can produce similar impedance patterns in this frequency range`);
        } else if (maxFreq - minFreq > 500) {
          updateStatusMessage(`MATH: Wide frequency range (${minFreq}-${maxFreq} Hz) improves parameter separation`);
          updateStatusMessage(`MATH: Different parameter combinations are more distinguishable across this frequency range`);
        }
        
        // Apply log scaling for better visualization - map to [0.15, 1] range
        sortedPoints.forEach(point => {
            if (point.resnorm === 0) {
                // Reference point gets full opacity
                point.alpha = 1;
            } else {
                // Map resnorm to alpha value using log scale - better points have higher opacity
                const logMin = Math.log10(minResnorm);
                const logMax = Math.log10(maxResnorm);
                const logVal = Math.log10(point.resnorm);
                
                // Invert scale: 1 = best fit (lowest resnorm), 0.15 = worst fit (highest resnorm)
                point.alpha = 1 - (0.85 * ((logVal - logMin) / (logMax - logMin)));
                
                // Ensure alpha is within bounds
                point.alpha = Math.max(0.15, Math.min(0.95, point.alpha));
            }
        });
        
        // We'll show all computed points, not just a percentage
        updateStatusMessage(`Grid computation complete. All ${sortedPoints.length} points will be displayed.`);
        
        // Reset to page 1 when results change
        setCurrentPage(1);
        
        // Map to ModelSnapshot format for visualization - will be filtered by maxGridPoints in the rendering
        const models = sortedPoints.map((point, idx) => 
            mapBackendMeshToSnapshot(point, idx)
        );
        
        // Calculate resnorm grouping thresholds using logarithmic scale
        const groups: ResnormGroup[] = [];
        
        // Add detailed logging about the resnorm range and grouping
        updateStatusMessage(`MATH: Resnorm calculation uses weighted RMSE of normalized impedance differences`);
        updateStatusMessage(`MATH: Lower frequencies are weighted more heavily (weight = 1/log10(f))`);
        updateStatusMessage(`MATH: Residuals are normalized by reference impedance magnitude at each frequency`);
        updateStatusMessage(`MATH: Min resnorm: ${minResnorm.toExponential(5)}, Max resnorm: ${maxResnorm.toExponential(5)}`);
        
        // Calculate frequency range ratio for dynamic thresholds
        const freqRangeRatio = maxFreq / minFreq;
        const isNarrowRange = freqRangeRatio < 100;
        const isWideRange = freqRangeRatio > 1000;

        // Adjust threshold multipliers based on frequency range
        const thresholdMultiplier = isNarrowRange ? 1.2 : 
                                   isWideRange ? 2.0 : 1.5;

        // Log the effect of frequency range on thresholds
        updateStatusMessage(`MATH: Frequency range ratio: ${freqRangeRatio.toFixed(1)}, threshold adjustment: ${thresholdMultiplier.toFixed(1)}x`);
        updateStatusMessage(`MATH: ${isNarrowRange ? "Narrow" : isWideRange ? "Wide" : "Moderate"} frequency range affects parameter sensitivity`);

        // Group 1: Very good fits (reference and very close matches)
        // With our improved normalization, thresholds are different
        const veryGoodThreshold = Math.min(0.03 * thresholdMultiplier, minResnorm * 1.5 * thresholdMultiplier);

        // Group 2: Good fits - more conservative threshold
        const goodThreshold = Math.min(0.08 * thresholdMultiplier, minResnorm * 3 * thresholdMultiplier);

        // Group 3: Moderate fits
        const moderateThreshold = Math.min(0.2 * thresholdMultiplier, minResnorm * 8 * thresholdMultiplier);
        
        // Group 4: Poor fits (everything else)
        
        // Log the thresholds for educational purposes
        updateStatusMessage(`MATH: Grouping thresholds: Very Good (0-${veryGoodThreshold.toExponential(3)}), Good (${veryGoodThreshold.toExponential(3)}-${goodThreshold.toExponential(3)}), Moderate (${goodThreshold.toExponential(3)}-${moderateThreshold.toExponential(3)}), Poor (>${moderateThreshold.toExponential(3)})`);
        updateStatusMessage(`MATH: Resnorm < 0.05 typically indicates excellent model fit in impedance spectroscopy`);
        
        // Create the groups with appropriate colors
        const veryGoodGroup: ResnormGroup = {
            range: [0, veryGoodThreshold] as [number, number],
            color: '#10B981', // Green
            items: models
              .filter(m => m.resnorm !== undefined && m.resnorm <= veryGoodThreshold)
              .map(item => ({
                ...item,
                color: '#10B981' // Ensure every item has the group color
              }))
        };
        
        const goodGroup: ResnormGroup = {
            range: [veryGoodThreshold, goodThreshold] as [number, number],
            color: '#3B82F6', // Blue
            items: models
              .filter(m => m.resnorm !== undefined && m.resnorm > veryGoodThreshold && m.resnorm <= goodThreshold)
              .map(item => ({
                ...item,
                color: '#3B82F6' // Ensure every item has the group color
              }))
        };
        
        const moderateGroup: ResnormGroup = {
            range: [goodThreshold, moderateThreshold] as [number, number],
            color: '#F59E0B', // Amber
            items: models
              .filter(m => m.resnorm !== undefined && m.resnorm > goodThreshold && m.resnorm <= moderateThreshold)
              .map(item => ({
                ...item,
                color: '#F59E0B' // Ensure every item has the group color
              }))
        };
        
        const poorGroup: ResnormGroup = {
            range: [moderateThreshold, maxResnorm] as [number, number],
            color: '#EF4444', // Red
            items: models
              .filter(m => m.resnorm !== undefined && m.resnorm > moderateThreshold)
              .map(item => ({
                ...item,
                color: '#EF4444' // Ensure every item has the group color
              }))
        };
        
        // Only add groups that have items
        if (veryGoodGroup.items.length > 0) groups.push(veryGoodGroup);
        if (goodGroup.items.length > 0) groups.push(goodGroup);
        if (moderateGroup.items.length > 0) groups.push(moderateGroup);
        if (poorGroup.items.length > 0) groups.push(poorGroup);
        
        // Log a detailed sample calculation for educational purposes - choose a few representative points
        const samplePoints = [
          // Best fit point
          sortedPoints[0],
          // Middle point
          sortedPoints[Math.floor(sortedPoints.length / 2)],
          // Worst fit point
          sortedPoints[sortedPoints.length - 1]
        ];
        
        updateStatusMessage(`MATH: Detailed resnorm calculations for sample grid points`);
        
        for (const point of samplePoints) {
          const pointType = point === sortedPoints[0] ? "Best fit" : 
                           point === sortedPoints[sortedPoints.length - 1] ? "Worst fit" : "Median fit";
          
          updateStatusMessage(`MATH: --- ${pointType} point (resnorm = ${point.resnorm.toExponential(5)}) ---`);
          updateStatusMessage(`MATH: Parameters: Rs=${point.parameters.Rs.toFixed(1)}Ω, Ra=${point.parameters.Ra.toFixed(0)}Ω, Ca=${(point.parameters.Ca*1e6).toFixed(2)}μF, Rb=${point.parameters.Rb.toFixed(0)}Ω, Cb=${(point.parameters.Cb*1e6).toFixed(2)}μF`);
          
          // Sample a few frequencies for the log - first, middle, last
          if (point.spectrum && point.spectrum.length > 0) {
            const freqSamples = [
              point.spectrum[0],
              point.spectrum[Math.floor(point.spectrum.length / 2)],
              point.spectrum[point.spectrum.length - 1]
            ];
            
            for (const freqSample of freqSamples) {
              // Find matching reference frequency point
              const refSample = referenceSpectrum.find(ref => 
                Math.abs(ref.freq - freqSample.freq) / freqSample.freq < 0.001
              );
              
              if (refSample) {
                const freq = freqSample.freq;
                const weight = 1 / Math.max(1, Math.log10(freq));
                const refMagnitude = Math.sqrt(refSample.real * refSample.real + refSample.imag * refSample.imag);
                const normFactor = refMagnitude > 0 ? refMagnitude : 1;
                
                const realResidual = (freqSample.real - refSample.real) / normFactor;
                const imagResidual = (freqSample.imag - refSample.imag) / normFactor;
                const residualSquared = (realResidual * realResidual + imagResidual * imagResidual) * weight;
                
                updateStatusMessage(`MATH:   At f=${freq.toFixed(1)}Hz: weight=${weight.toFixed(3)}, normalized residual=${Math.sqrt(realResidual*realResidual + imagResidual*imagResidual).toExponential(3)}, contribution=${residualSquared.toExponential(3)}`);
              }
            }
          }
        }
        
        setResnormGroups(groups);
        updateStatusMessage(`Grid visualization ready with ${models.length} points grouped by resnorm`);
        setIsComputingGrid(false);
        
        // Add unique IDs based on resnorm sorting
        const sortedByResnorm = [...sortedPoints].sort((a, b) => a.resnorm - b.resnorm);
        const pointsWithIds = sortedByResnorm.map((point, idx) => ({
            ...point,
            id: idx + 1 // ID starts from 1
        }));
        
        // Store all computed points in both state variables
        setGridResultsWithIds(pointsWithIds);
        setGridResults(sortedPoints);
        updateStatusMessage(`Stored and displaying all ${sortedPoints.length} computed grid points.`);
        
        // Always update reference model after computing grid
        const updatedModel = createReferenceModel();
        setReferenceModel(updatedModel);
        
        // If reference model is not visible, make it visible
        if (!referenceModelId) {
          setReferenceModelId('dynamic-reference');
          updateStatusMessage('Reference model created and shown');
        } else {
          updateStatusMessage('Reference model updated');
        }
        
        // Add to the logs after the frequency range is updated
        updateStatusMessage(`MATH: Frequency range updated to ${minFreq.toExponential(2)}-${maxFreq.toExponential(2)} Hz with ${numPoints} points`);
        updateStatusMessage(`MATH: Resnorm calculation is frequency-sensitive. Changing the range affects parameter sensitivity:`);
        if (minFreq < 1.0) {
          updateStatusMessage(`MATH: Very low frequencies (< 1 Hz) highlight capacitive effects (Ca, Cb)`);
        }
        if (minFreq < 10.0 && maxFreq > 100) {
          updateStatusMessage(`MATH: Mid-range frequencies (10-100 Hz) highlight RC time constants`);
        }
        if (maxFreq > 1000) {
          updateStatusMessage(`MATH: High frequencies (> 1000 Hz) emphasize series resistance (Rs)`);
        }

        // Also add these status messages during grid computation, before calculating spectra
        updateStatusMessage(`MATH: Frequency range ${minFreq.toExponential(2)}-${maxFreq.toExponential(2)} Hz affects parameter sensitivity:`);
        if (maxFreq / minFreq < 100) {
          updateStatusMessage(`MATH: Narrow frequency range (ratio ${(maxFreq/minFreq).toFixed(1)}) means fewer distinguishing features`);
          updateStatusMessage(`MATH: Some parameter combinations may appear similar in this range`);
        } else if (maxFreq / minFreq > 1000) {
          updateStatusMessage(`MATH: Wide frequency range (ratio ${(maxFreq/minFreq).toFixed(1)}) provides better parameter discrimination`);
          updateStatusMessage(`MATH: This helps distinguish between different circuit parameter sets`);
        }
        
    } catch (error) {
        console.error("Error computing grid:", error);
        setGridError(`Grid computation failed: ${error instanceof Error ? error.message : String(error)}`);
        updateStatusMessage(`Error computing grid: ${error instanceof Error ? error.message : String(error)}`);
        setIsComputingGrid(false);
    }
  };

  // Function to apply parameters from a grid result
  const handleApplyParameters = (point: BackendMeshPoint) => {
    setParameters({
      Rs: point.parameters.Rs,
      Ra: point.parameters.Ra,
      Ca: point.parameters.Ca,
      Rb: point.parameters.Rb,
      Cb: point.parameters.Cb,
      frequency_range: point.parameters.frequency_range
    });
    setParameterChanged(true);
    updateStatusMessage(`Applied parameters from grid result with resnorm ${point.resnorm.toExponential(3)}`);
  };

  // Collapsible section component - Modernized
  const CollapsibleSection = ({ 
    title, 
    isOpen, 
    toggleOpen, 
    children 
  }: { 
    title: string; 
    isOpen: boolean; 
    toggleOpen: () => void; 
    children: React.ReactNode 
  }) => {
    return (
      <div className="collapsible-section">
        <button
          onClick={toggleOpen}
          className="section-header group"
        >
          <span className="flex items-center">
            {title === "Grid Computation" && (
              <svg className="w-4 h-4 mr-1.5 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
              </svg>
            )}
            {title === "Circuit Parameters" && (
              <svg className="w-4 h-4 mr-1.5 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
              </svg>
            )}
            {title}
          </span>
          <svg 
            className={`icon-small transition-transform duration-300 ease-in-out ${isOpen ? 'transform rotate-180' : ''}`} 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        <div
          className={`transition-all duration-300 ease-in-out ${
            isOpen ? 'max-h-[2000px] opacity-100' : 'max-h-0 opacity-0 overflow-hidden'
          }`}
        >
          <div className="section-content">
            {children}
          </div>
        </div>
      </div>
    );
  };

  // Add pagination state - used by child components
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [currentPage, setCurrentPage] = useState(1);
  
  // Update frequency array when range changes - called internally
  const updateFrequencies = (min: number, max: number, points: number) => {
    // Validate frequency range
    if (min <= 0) {
      console.warn("Minimum frequency must be positive, setting to 0.01 Hz");
      min = 0.01;
    }
    if (max <= min) {
      console.warn("Maximum frequency must be greater than minimum, setting to min*10");
      max = min * 10;
    }
    
    // Create proper logarithmically spaced frequencies for EIS
    const frequencies: number[] = [];
    const logMin = Math.log10(min);
    const logMax = Math.log10(max);
    const logStep = (logMax - logMin) / (points - 1);
    
    for (let i = 0; i < points; i++) {
      const logValue = logMin + i * logStep;
      const frequency = Math.pow(10, logValue);
      frequencies.push(frequency);
    }
    
    setMinFreq(min);
    setMaxFreq(max);
    setNumPoints(points);
    
    // Update parameters with the new frequency range
    setParameters(prev => ({
      ...prev,
      frequency_range: [min, max] // Keep the [min, max] format for backward compatibility
    }));
    
    // Store the full frequency array in a separate state variable
    setFrequencyPoints(frequencies);
    
    // Update status message with details about the frequency range
    updateStatusMessage(`Frequency range updated: ${min.toFixed(2)} Hz - ${max.toFixed(1)} Hz with ${points} points (logarithmic spacing)`);
    
    // Mark parameters as changed - will require recomputing the grid
    setParameterChanged(true);
  };
  
  // Helper functions for logarithmic slider
  const logToLinearSlider = (logValue: number) => {
    // Convert a log scale value to linear slider position (0-100)
    const minLog = Math.log10(0.01);
    const maxLog = Math.log10(10000);
    const scale = (Math.log10(logValue) - minLog) / (maxLog - minLog);
    return scale * 100;
  };
  
  const linearSliderToLog = (sliderPosition: number) => {
    // Convert linear slider position (0-100) to log scale value
    const minLog = Math.log10(0.01);
    const maxLog = Math.log10(10000);
    const logValue = minLog + (sliderPosition / 100) * (maxLog - minLog);
    return Math.pow(10, logValue);
  };

  // No longer need TER and TEC display in the header
  // Removed unused variables and the resetToReference function

  // Initialize frequency points on component mount
  useEffect(() => {
    // Generate initial frequency points
    const initialFreqs: number[] = [];
    const logMin = Math.log10(minFreq);
    const logMax = Math.log10(maxFreq);
    const logStep = (logMax - logMin) / (numPoints - 1);
    
    for (let i = 0; i < numPoints; i++) {
      const logValue = logMin + i * logStep;
      const frequency = Math.pow(10, logValue);
      initialFreqs.push(frequency);
    }
    
    setFrequencyPoints(initialFreqs);
  }, [minFreq, maxFreq, numPoints]);

  // Separate useEffect for initializing the reference model
  useEffect(() => {
    if (frequencyPoints.length > 0 && !referenceModel && !manuallyHidden) {
      const initialReferenceModel = createReferenceModel();
      setReferenceModel(initialReferenceModel);
      setReferenceModelId('dynamic-reference');
    }
  }, [frequencyPoints, referenceModel, manuallyHidden, createReferenceModel]);

  // Update reference model when frequency parameters change
  useEffect(() => {
    // Only update if reference model is already visible and not manually hidden
    if (referenceModelId === 'dynamic-reference' && !manuallyHidden && frequencyPoints.length > 0) {
      const updatedModel = createReferenceModel();
      setReferenceModel(updatedModel);
      updateStatusMessage('Reference model updated with new frequency range');
    }
  }, [minFreq, maxFreq, numPoints, frequencyPoints.length, referenceModelId, manuallyHidden, createReferenceModel, updateStatusMessage]);

  // Modify the main content area to show the correct tab content
  return (
    <div className="circuit-container">
      <style jsx>{`
        .pagination-button {
          @apply p-1.5 rounded-md text-neutral-400 hover:text-primary hover:bg-neutral-800/50 transition-colors;
        }
        .pagination-button:disabled {
          @apply opacity-50 cursor-not-allowed hover:text-neutral-400 hover:bg-transparent;
        }
        .pagination-button svg {
          @apply w-4 h-4;
        }
        .slider-marker {
          @apply absolute w-px h-3 bg-neutral-600/50 rounded-full bottom-0;
        }
        .slider-marker-label {
          @apply absolute -bottom-5 transform -translate-x-1/2 text-[10px] text-neutral-500;
        }
        .slider-track {
          @apply absolute h-1 bg-neutral-700 rounded-full w-full top-1/2 -mt-0.5;
        }
        .slider-track-active {
          @apply absolute h-1 bg-primary/60 rounded-full top-1/2 -mt-0.5;
        }
        .slider-thumb {
          @apply absolute w-3 h-3 bg-primary rounded-full -mt-1 cursor-pointer transform -translate-x-1/2 hover:scale-110 transition-transform;
        }
        .slider-thumb-label {
          @apply absolute -top-5 left-1/2 transform -translate-x-1/2 text-xs text-neutral-400;
        }
        
        /* Dual range slider styles */
        .range_container {
          position: relative;
          width: 100%;
        }
        
        .sliders_control {
          position: relative;
          min-height: 30px;
          margin-bottom: 16px;
        }
        
        .form_control {
          position: relative;
          display: flex;
          justify-content: space-between;
          margin-top: 16px;
        }
        
        .form_control_container {
          flex: 1;
        }
        
        .form_control_container:first-child {
          margin-right: 12px;
        }
        
        .form_control_container__time {
          @apply text-[11px] text-neutral-400 mb-1;
        }
        
        .form_control_container__time__input {
          @apply w-full p-1.5 border border-neutral-700 rounded text-xs text-center bg-neutral-800 text-neutral-200;
          transition: all 0.15s ease;
        }
        
        .form_control_container__time__input:focus {
          @apply border-primary outline-none;
          box-shadow: 0 0 0 1px var(--primary-focus);
        }
        
        .form_control_container__time__input:hover:not(:focus) {
          @apply border-neutral-600;
        }
        
        /* Styling for all range inputs */
        input[type="range"] {
          -webkit-appearance: none;
          -moz-appearance: none;
          appearance: none;
          cursor: pointer;
          background: transparent;
        }
        
        /* Thumb styling for WebKit browsers */
        input[type="range"]::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 16px;
          height: 16px;
          border-radius: 50%;
          background: var(--primary);
          cursor: pointer;
          border: 2px solid var(--neutral-900);
          box-shadow: 0 0 2px rgba(0, 0, 0, 0.3);
          margin-top: -7px;
          transition: all 0.2s ease;
          z-index: 30;
          position: relative;
        }
        
        input[type="range"]::-webkit-slider-thumb:hover {
          background: var(--primary-focus);
          transform: scale(1.1);
        }
        
        input[type="range"]::-webkit-slider-thumb:active {
          background: var(--primary-focus);
          transform: scale(1.15);
          box-shadow: 0 0 5px var(--primary);
        }
        
        /* Track styling for WebKit browsers */
        input[type="range"]::-webkit-slider-runnable-track {
          width: 100%;
          height: 2px;
          cursor: pointer;
          background: transparent;
          border-radius: 2px;
        }
        
        /* Thumb styling for Firefox */
        input[type="range"]::-moz-range-thumb {
          width: 16px;
          height: 16px;
          border-radius: 50%;
          background: var(--primary);
          cursor: pointer;
          border: 2px solid var(--neutral-900);
          box-shadow: 0 0 2px rgba(0, 0, 0, 0.3);
          transition: all 0.2s ease;
          z-index: 30;
          position: relative;
        }
        
        input[type="range"]::-moz-range-thumb:hover {
          background: var(--primary-focus);
          transform: scale(1.1);
        }
        
        input[type="range"]::-moz-range-thumb:active {
          background: var(--primary-focus);
          transform: scale(1.15);
          box-shadow: 0 0 5px var(--primary);
        }
        
        /* Track styling for Firefox */
        input[type="range"]::-moz-range-track {
          width: 100%;
          height: 2px;
          cursor: pointer;
          background: transparent;
          border-radius: 2px;
        }
      `}</style>
      
      {/* Header - Modernized */}
      <header className="circuit-header">
        <div className="flex items-center gap-4">
          <h1 className="text-title">Circuit Simulator</h1>
          
          {/* Visualization tabs */}
          <div className="flex bg-neutral-800 rounded-md overflow-hidden">
            <button 
              className={`px-4 py-1.5 text-sm font-medium ${visualizationTab === 'visualizer' ? 'bg-primary text-white' : 'text-neutral-300 hover:bg-neutral-700 transition-colors'}`}
              onClick={() => setVisualizationTab('visualizer')}
            >
              Visualizer
            </button>
            <button 
              className={`px-4 py-1.5 text-sm font-medium ${visualizationTab === 'data' ? 'bg-primary text-white' : 'text-neutral-300 hover:bg-neutral-700 transition-colors'}`}
              onClick={() => setVisualizationTab('data')}
            >
              Data
            </button>
            <button 
              className={`px-4 py-1.5 text-sm font-medium ${visualizationTab === 'math' ? 'bg-primary text-white' : 'text-neutral-300 hover:bg-neutral-700 transition-colors'}`}
              onClick={() => setVisualizationTab('math')}
            >
              Math
            </button>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {gridResults.length > 0 && (
            <div className="chip bg-primary/10 text-primary border border-primary/20">
              <span className="chip-indicator" style={{ backgroundColor: 'var(--success)' }}></span>
              {Math.pow(gridSize, 5).toLocaleString()} Grid Points
            </div>
          )}
        </div>
      </header>

      {/* Main Content */}
      <div className="circuit-content">
        {/* Sidebar */}
        <div className="circuit-sidebar">
          {/* Sidebar Tabs */}
          <div className="circuit-tabs">
            <button
              className={`circuit-tab ${currentTab === 'toolbox' ? 'circuit-tab-active' : 'circuit-tab-inactive'}`}
              onClick={() => setCurrentTab('toolbox')}
            >
              <span>Toolbox</span>
            </button>
            <button
              className={`circuit-tab ${currentTab === 'activity' ? 'circuit-tab-active' : 'circuit-tab-inactive'}`}
              onClick={() => setCurrentTab('activity')}
            >
              <span>Activity Log</span>
            </button>
          </div>
              
          {/* Sidebar Content - scrollable */}
          <div className="flex-1 overflow-y-auto">
            {currentTab === 'toolbox' && (
              <div className="p-4 space-y-4">
                {/* Grid Computation Section */}
                <CollapsibleSection 
                  title="Grid Computation" 
                  isOpen={gridSettingsOpen} 
                  toggleOpen={() => setGridSettingsOpen(!gridSettingsOpen)}
                >
                  <div className="space-y-3 pt-1">
                    <div className="flex items-center justify-between gap-2">
                      <label htmlFor="gridSize" className="slider-label-text">
                        Points per parameter:
                      </label>
                                              <input
                          type="number"
                          id="gridSize"
                          value={gridSize}
                          onChange={(e) => {
                            const size = Math.max(2, Math.min(10, parseInt(e.target.value) || 2));
                            setGridSize(size);
                            const totalPoints = Math.pow(size, 5);
                            updateStatusMessage(`Grid size set to ${size} (${totalPoints.toLocaleString()} total points to compute)`);
                          }} 
                          min="2"
                          max="10"
                          className="w-16 p-1 border rounded text-xs text-center"
                        />
                    </div>

                    {/* Frequency Range Slider - New addition */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <label className="slider-label-text font-medium text-xs text-neutral-300">
                          Frequency range:
                        </label>
                        <span className="text-xs font-mono text-neutral-300">
                          {minFreq < 1 ? minFreq.toFixed(2) + ' Hz' : minFreq < 1000 ? minFreq.toFixed(1) + ' Hz' : (minFreq/1000).toFixed(1) + ' kHz'} - 
                          {maxFreq < 1000 ? maxFreq.toFixed(1) + ' Hz' : (maxFreq/1000).toFixed(1) + ' kHz'}
                        </span>
                      </div>
                      
                      <div className="range_container">
                        <div className="sliders_control relative h-8">
                          {/* Slider track */}
                          <div className="slider-track"></div>
                          <div className="slider-track-active" style={{ 
                            left: `${logToLinearSlider(minFreq)}%`, 
                            width: `${logToLinearSlider(maxFreq) - logToLinearSlider(minFreq)}%` 
                          }}></div>
                          
                          {/* Min frequency slider */}
                          <input 
                            id="fromSlider"
                            type="range" 
                            min={0}
                            max={100}
                            step={0.1}
                            value={logToLinearSlider(minFreq)}
                            onChange={(e) => {
                              // Convert linear slider position to logarithmic value
                              const sliderPos = parseFloat(e.target.value);
                              const logValue = linearSliderToLog(sliderPos);
                              const value = Math.max(0.01, Math.min(maxFreq - 0.01, logValue));
                              setMinFreq(value);
                              updateFrequencies(value, maxFreq, numPoints);
                              setParameterChanged(true);
                            }}
                            onMouseUp={() => {
                              updateStatusMessage(`Frequency range updated to ${minFreq < 1000 ? minFreq.toFixed(2) + ' Hz' : (minFreq/1000).toFixed(2) + ' kHz'} - ${maxFreq < 1000 ? maxFreq.toFixed(1) + ' Hz' : (maxFreq/1000).toFixed(1) + ' kHz'}. Recompute grid to see effect.`);
                            }}
                            className="absolute top-1/2 -translate-y-1/2 appearance-none w-full h-8 bg-transparent z-10"
                          />
                          
                          {/* Max frequency slider */}
                          <input 
                            id="toSlider"
                            type="range" 
                            min={0}
                            max={100}
                            step={0.1}
                            value={logToLinearSlider(maxFreq)}
                            onChange={(e) => {
                              // Convert linear slider position to logarithmic value
                              const sliderPos = parseFloat(e.target.value);
                              const logValue = linearSliderToLog(sliderPos);
                              const value = Math.max(minFreq + 0.01, Math.min(10000, logValue));
                              setMaxFreq(value);
                              updateFrequencies(minFreq, value, numPoints);
                              setParameterChanged(true);
                            }}
                            onMouseUp={() => {
                              updateStatusMessage(`Frequency range updated to ${minFreq < 1000 ? minFreq.toFixed(2) + ' Hz' : (minFreq/1000).toFixed(2) + ' kHz'} - ${maxFreq < 1000 ? maxFreq.toFixed(1) + ' Hz' : (maxFreq/1000).toFixed(1) + ' kHz'}. Recompute grid to see effect.`);
                            }}
                            className="absolute top-1/2 -translate-y-1/2 appearance-none w-full h-8 bg-transparent z-20"
                          />
                          
                          {/* Tick marks at logarithmic intervals */}
                          {[0.01, 0.1, 1, 10, 100, 1000, 10000].map((tick) => (
                            <div key={tick} className="slider-marker" style={{ 
                              left: `${logToLinearSlider(tick)}%` 
                            }}>
                              <span className="slider-marker-label">{tick < 1000 ? (tick < 1 ? tick.toFixed(2) : tick) : `${tick/1000}k`}</span>
                            </div>
                          ))}
                        </div>
                        
                        <div className="form_control flex justify-between mt-6">
                          <div className="form_control_container">
                            <div className="form_control_container__time text-[11px] text-neutral-400 mb-1">Min Frequency (Hz)</div>
                            <input 
                              className="form_control_container__time__input w-24 p-1.5 border rounded text-xs text-center" 
                              type="number" 
                              id="fromInput" 
                              value={parseFloat(minFreq.toFixed(2))}
                              min={0.01}
                              max={maxFreq - 0.01}
                              step={0.01}
                              onChange={(e) => {
                                const value = parseFloat(e.target.value);
                                if (!isNaN(value) && value >= 0.01 && value < maxFreq) {
                                  setMinFreq(value);
                                  updateFrequencies(value, maxFreq, numPoints);
                                  setParameterChanged(true);
                                }
                              }}
                            />
                          </div>
                          <div className="form_control_container">
                            <div className="form_control_container__time text-[11px] text-neutral-400 mb-1">Max Frequency (Hz)</div>
                            <input 
                              className="form_control_container__time__input w-24 p-1.5 border rounded text-xs text-center" 
                              type="number" 
                              id="toInput" 
                              value={parseFloat(maxFreq.toFixed(1))}
                              min={minFreq + 0.01}
                              max={10000}
                              step={0.1}
                              onChange={(e) => {
                                const value = parseFloat(e.target.value);
                                if (!isNaN(value) && value > minFreq && value <= 10000) {
                                  setMaxFreq(value);
                                  updateFrequencies(minFreq, value, numPoints);
                                  setParameterChanged(true);
                                }
                              }}
                            />
                          </div>
                        </div>
                      </div>
                      
                      {/* Frequency range impact message */}
                      <div className="text-[11px] text-neutral-400 mt-3 italic pl-1"></div>

                      {/* Frequency Points control */}
                      <div className="flex items-center justify-between">
                        <label className="slider-label-text font-medium text-xs text-neutral-300">
                          Frequency points:
                        </label>
                        <span className="text-xs font-mono text-neutral-300">
                          {numPoints} points
                        </span>
                      </div>
                      <div className="mt-2 relative h-8">
                        <div className="slider-track"></div>
                        <div className="slider-track-active" style={{ 
                          left: '0%', 
                          width: `${(numPoints - 10) / (200 - 10) * 100}%`
                        }}></div>
                        <input 
                          type="range" 
                          min={10} 
                          max={200} 
                          step={1}
                          value={numPoints}
                          onChange={(e) => {
                            const points = parseInt(e.target.value);
                            setNumPoints(points);
                            updateFrequencies(minFreq, maxFreq, points);
                            setParameterChanged(true);
                          }}
                          onMouseUp={() => {
                            updateStatusMessage(`Frequency points set to ${numPoints}. Recompute grid to see effect.`);
                          }}
                          className="absolute top-1/2 -translate-y-1/2 appearance-none w-full h-8 bg-transparent" 
                          style={{
                            zIndex: 10
                          }}
                        />
                        <div className="flex justify-between px-1 mt-4 text-[10px] text-neutral-500">
                          <span>10</span>
                          <span>100</span>
                          <span>200</span>
                        </div>
                      </div>
                      <div className="flex justify-between items-center mt-2">
                        <div className="text-[11px] text-neutral-400">Number of frequency points in range</div>
                        <input 
                          type="number" 
                          min={10} 
                          max={200} 
                          step={1}
                          value={numPoints}
                          onChange={(e) => {
                            const value = parseInt(e.target.value);
                            if (!isNaN(value) && value >= 10 && value <= 200) {
                              setNumPoints(value);
                              updateFrequencies(minFreq, maxFreq, value);
                              setParameterChanged(true);
                              updateStatusMessage(`Frequency points set to ${value}. Recompute grid to see effect.`);
                            }
                          }}
                          className="w-16 p-1 border border-neutral-700 rounded text-xs text-center bg-neutral-800 text-neutral-200"
                        />
                      </div>
                    </div>
                    
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <label className="slider-label-text font-medium text-xs text-neutral-300 block">
                          Grid Summary:
                        </label>
                      </div>
                      <div className="text-xs text-neutral-400 space-y-1 mt-2">
                        <div className="flex justify-between">
                          <span className="text-[11px]">Total points to compute:</span>
                          <span className="font-medium text-neutral-300">{Math.pow(gridSize, 5).toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-[11px]">Computed points:</span>
                          <span className="font-medium text-neutral-300">
                            {gridResults.length > 0 ? gridResults.length.toLocaleString() : Math.pow(gridSize, 5).toLocaleString()}
                          </span>
                        </div>
                        {parameterChanged && (
                          <div className="text-amber-500 text-[11px]">Parameters modified</div>
                        )}
                      </div>
                    </div>

                    <button
                      onClick={handleComputeRegressionMesh}
                      disabled={isComputingGrid}
                      className={`w-full mt-2 py-2.5 rounded-lg ${
                        isComputingGrid
                          ? 'bg-gray-600 cursor-not-allowed'
                          : parameterChanged 
                            ? 'bg-amber-600 hover:bg-amber-700' 
                            : 'button-primary'
                      }`}
                    >
                      {isComputingGrid ? (
                        <span className="flex items-center justify-center">
                          <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                          </svg>
                          Computing...
                        </span>
                      ) : (parameterChanged ? 'Recompute Grid' : 'Compute Grid')}
                    </button>
                    
                    {/* Apply Best Fit Button - only shown when grid results are available */}
                    {gridResults.length > 0 && gridResults[0].resnorm > 0 && (
                      <button
                        onClick={() => handleApplyParameters(gridResults[0])}
                        className="button-success w-full mt-2"
                      >
                        Use Best Fit (Resnorm: {gridResults[0].resnorm.toExponential(3)})
                      </button>
                    )}
                  </div>
                </CollapsibleSection>

                {/* Circuit Parameters Section */}
                <CollapsibleSection 
                  title="Circuit Parameters" 
                  isOpen={circuitParamsOpen} 
                  toggleOpen={() => setCircuitParamsOpen(!circuitParamsOpen)}
                >
                  <div className="space-y-4 pt-1">
                    <div className="space-y-3">
                      <ParamSlider 
                        label="Rs" 
                        value={groundTruthParams.Rs} 
                        min={10} 
                        max={30} 
                        step={0.1}
                        unit="Ω" 
                        onChange={(val: number) => {
                          setGroundTruthParams(prev => ({ ...prev, Rs: val }));
                          updateStatusMessage(`Rs set to ${val.toFixed(1)} Ω`);
                          // Update reference model immediately if visible
                          if (referenceModelId === 'dynamic-reference') {
                            const updatedModel = createReferenceModel();
                            setReferenceModel(updatedModel);
                          }
                        }} 
                      />
                      
                      <ParamSlider 
                        label="Ra" 
                        value={groundTruthParams.Ra} 
                        min={100} 
                        max={1000} 
                        step={1}
                        unit="Ω" 
                        onChange={(val: number) => {
                          setGroundTruthParams(prev => ({ ...prev, Ra: val }));
                          updateStatusMessage(`Ra set to ${val.toFixed(0)} Ω`);
                          // Update reference model immediately if visible
                          if (referenceModelId === 'dynamic-reference') {
                            const updatedModel = createReferenceModel();
                            setReferenceModel(updatedModel);
                          }
                        }} 
                      />
                      
                      <ParamSlider 
                        label="Ca" 
                        value={groundTruthParams.Ca * 1e6} 
                        min={1} 
                        max={5} 
                        step={0.01}
                        unit="μF" 
                        onChange={(val: number) => {
                          setGroundTruthParams(prev => ({ ...prev, Ca: val / 1e6 }));
                          updateStatusMessage(`Ca set to ${val.toFixed(2)} μF`);
                          // Update reference model immediately if visible
                          if (referenceModelId === 'dynamic-reference') {
                            const updatedModel = createReferenceModel();
                            setReferenceModel(updatedModel);
                          }
                        }} 
                      />
                      
                      <ParamSlider 
                        label="Rb" 
                        value={groundTruthParams.Rb} 
                        min={100} 
                        max={1000} 
                        step={1}
                        unit="Ω" 
                        onChange={(val: number) => {
                          setGroundTruthParams(prev => ({ ...prev, Rb: val }));
                          updateStatusMessage(`Rb set to ${val.toFixed(0)} Ω`);
                          // Update reference model immediately if visible
                          if (referenceModelId === 'dynamic-reference') {
                            const updatedModel = createReferenceModel();
                            setReferenceModel(updatedModel);
                          }
                        }} 
                      />
                      
                      <ParamSlider 
                        label="Cb" 
                        value={groundTruthParams.Cb * 1e6} 
                        min={1} 
                        max={5} 
                        step={0.01}
                        unit="μF" 
                        onChange={(val: number) => {
                          setGroundTruthParams(prev => ({ ...prev, Cb: val / 1e6 }));
                          updateStatusMessage(`Cb set to ${val.toFixed(2)} μF`);
                          // Update reference model immediately if visible
                          if (referenceModelId === 'dynamic-reference') {
                            const updatedModel = createReferenceModel();
                            setReferenceModel(updatedModel);
                          }
                        }} 
                      />
                    </div>
                  </div>
                </CollapsibleSection>


              </div>
            )}
              
            {currentTab === 'activity' && (
              <div className="p-4">
                <div className="text-xs space-y-0 text-neutral-400 max-h-full overflow-y-auto">
                  {logMessages.length === 0 ? (
                    <p className="italic text-neutral-500 p-3 text-center">No activity yet</p>
                  ) : (
                    logMessages.map((log, idx) => (
                      <div key={idx} className="py-2 px-2 border-b border-neutral-800 last:border-0">
                        <span className="text-neutral-500 mr-2 text-[10px]">{log.time}</span>
                        <span className={`
                          ${log.message.includes('Error') ? 'text-danger' : ''}
                          ${log.message.includes('Success') || log.message.includes('computed successfully') ? 'text-success' : ''}
                          ${log.message.includes('MATH:') ? 'text-primary font-medium' : ''}
                        `}>{log.message}</span>
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
                  
        {/* Main Content Area */}
        <div className="circuit-main">
          {/* Status message bar */}
          {statusMessage && (
            <div className="circuit-status">
              <div className="flex items-center">
                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                {statusMessage}
              </div>
            </div>
          )}
                  
          {/* Main visualization area */}
          <div className="circuit-visualization">
            {isComputingGrid ? (
              <div className="flex items-center justify-center p-6 h-40 bg-neutral-100/5 border border-neutral-700 rounded-lg shadow-md">
                <div className="animate-spin rounded-full h-6 w-6 border-t-2 border-b-2 border-primary mr-3"></div>
                <div>
                  <p className="text-sm text-primary font-medium">Computing grid points...</p>
                  <p className="text-xs text-neutral-400 mt-1">This may take a moment depending on grid size</p>
                </div>
              </div>
            ) : gridError ? (
              <div className="p-4 bg-danger-light border border-danger rounded-lg text-sm text-danger">
                <div className="flex items-start">
                  <svg className="w-5 h-5 mr-2 mt-0.5 text-danger" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <div>
                    <p className="font-medium">Error occurred</p>
                    <p className="mt-1">{gridError}</p>
                  </div>
                </div>
              </div>
            ) : visualizationTab === 'math' ? (
              <MathDetailsTab 
                parameters={parameters}
                minFreq={minFreq}
                maxFreq={maxFreq}
                numPoints={numPoints}
                referenceModel={referenceModel}
              />
            ) : visualizationTab === 'data' ? (
              gridResults.length > 0 ? (
                <DataTableTab 
                  gridResults={gridResults}
                  gridResultsWithIds={gridResultsWithIds}
                  resnormGroups={resnormGroups}
                  hiddenGroups={hiddenGroups}
                  maxGridPoints={Math.pow(gridSize, 5)}
                  gridSize={gridSize}
                  parameters={parameters}
                  groundTruthParams={groundTruthParams}
                />
              ) : (
                <div className="flex items-center justify-center h-40 bg-neutral-100/5 border border-neutral-700 rounded-lg shadow-md p-5">
                  <div className="text-center">
                    <svg className="w-10 h-10 mx-auto mb-3 text-neutral-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <p className="text-sm text-neutral-400">
                      No data to display. Compute a grid first.
                    </p>
                  </div>
                </div>
              )
            ) : visualizationTab === 'visualizer' ? (
              <div className="space-y-5">
                {/* Spider Plot Visualization - Always render regardless of data */}
                <VisualizerTab 
                  resnormGroups={resnormGroups}
                  hiddenGroups={hiddenGroups}
                  maxGridPoints={Math.pow(gridSize, 5)}
                  opacityLevel={opacityLevel}
                  logScalar={logScalar}
                  referenceModelId={referenceModelId}
                  referenceModel={referenceModel}
                  minFreq={minFreq}
                  maxFreq={maxFreq}
                />
              </div>
            ) : (
              <div className="flex items-center justify-center h-40 bg-neutral-100/5 border border-neutral-700 rounded-lg shadow-md p-5">
                <div className="text-center">
                  <svg className="w-10 h-10 mx-auto mb-3 text-neutral-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  <p className="text-sm text-neutral-400">
                    Unknown tab selected. Please select a valid tab.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
} 

// Helper to convert backend spectrum to ImpedancePoint[] (types.ts shape)
function toImpedancePoints(spectrum: { freq?: number; frequency?: number; real: number; imag?: number; imaginary?: number; mag?: number; magnitude?: number; phase: number; }[], forImpedance?: boolean): TypesImpedancePoint[] {
  return spectrum.map(p => ({
    frequency: p.frequency ?? p.freq ?? 0,
    real: p.real,
    imaginary: forImpedance ? (p.imag ?? p.imaginary ?? 0) : (p.imag ?? 0),
    magnitude: p.magnitude ?? p.mag ?? 0,
    phase: p.phase
  }));
}

// --- Utility functions for circuit simulation ---

// Complex number type
type Complex = {
  real: number;
  imag: number;
};

// Complex number operations
const complex = {
  add: (a: Complex, b: Complex): Complex => ({
    real: a.real + b.real,
    imag: a.imag + b.imag
  }),
  multiply: (a: Complex, b: Complex): Complex => ({
    real: a.real * b.real - a.imag * b.imag,
    imag: a.real * b.imag + a.imag * b.real
  }),
  divide: (a: Complex, b: Complex): Complex => {
    const denom = b.real * b.real + b.imag * b.imag;
    return {
      real: (a.real * b.real + a.imag * b.imag) / denom,
      imag: (a.imag * b.real - a.real * b.imag) / denom
    };
  }
};

// Impedance spectrum calculation for epithelial model
function calculateImpedanceSpectrum(params: {
  Rs: number;
  Ra: number;
  Rb: number;
  Ca: number;
  Cb: number;
}, freqs: number[]): { freq: number; real: number; imag: number; mag: number; phase: number }[] {
  return freqs.map(freq => {
    const omega = 2 * Math.PI * freq;
    
    // Za = Ra/(1+jωRaCa)
    const Za_denom = complex.add(
      { real: 1, imag: 0 },
      { real: 0, imag: omega * params.Ra * params.Ca }
    );
    const Za = complex.divide(
      { real: params.Ra, imag: 0 },
      Za_denom
    );
    
    // Zb = Rb/(1+jωRbCb)
    const Zb_denom = complex.add(
      { real: 1, imag: 0 },
      { real: 0, imag: omega * params.Rb * params.Cb }
    );
    const Zb = complex.divide(
      { real: params.Rb, imag: 0 },
      Zb_denom
    );
    
    // Z_eq = Rs + Za + Zb
    const Z_eq = complex.add(
      { real: params.Rs, imag: 0 },
      complex.add(Za, Zb)
    );
    
    // Magnitude and phase
    const magnitude = Math.sqrt(Z_eq.real * Z_eq.real + Z_eq.imag * Z_eq.imag);
    const phase = Math.atan2(Z_eq.imag, Z_eq.real) * (180 / Math.PI);
    return {
      freq,
      real: Z_eq.real,
      imag: Z_eq.imag,
      mag: magnitude,
      phase
    };
  });
}

// These calculateTER and calculateTEC functions are now handled in the MathDetailsTab component
// TEC is calculated as CaCb/(Ca+Cb)

// Calculate resnorm between reference and test models
function calculateResnorm(reference: ModelSnapshot, test: ModelSnapshot): number {
  if (!reference.data.length || !test.data.length) return Infinity;
  
  // Collect all impedance points from both models
  const referenceData = reference.data;
  const testData = test.data;
  
  // Define a logging function that adds detailed calculations to the status log
  const logFunction = (message: string) => {
    // Add to console but not UI to avoid too much noise
    console.log(message);
  };
  
  // Extract frequency range from the data
  const minFreq = Math.min(...referenceData.map(p => p.frequency));
  const maxFreq = Math.max(...referenceData.map(p => p.frequency));
  
  // Log the frequency range being used
  logFunction(`Resnorm calculation using frequency range: ${minFreq.toFixed(2)} - ${maxFreq.toFixed(2)} Hz (${referenceData.length} points)`);
  
  // Calculate the weighted sum of squared residuals
  let sumWeightedSquaredResiduals = 0;
  let sumWeights = 0;
  
  // Go through each frequency point
  for (let i = 0; i < Math.min(referenceData.length, testData.length); i++) {
    const refPoint = referenceData[i];
    const testPoint = testData[i];
    
    if (!refPoint || !testPoint) continue;
    
    // Ensure the frequencies match (within tolerance)
    if (Math.abs(refPoint.frequency - testPoint.frequency) / refPoint.frequency > 0.001) {
      continue; // Skip if frequencies don't match
    }
    
    // Extract real and imaginary components
    const refReal = refPoint.real;
    const refImag = refPoint.imaginary ?? 0;
    const testReal = testPoint.real;
    const testImag = testPoint.imaginary ?? 0;
    
    // Enhanced frequency weighting: give more weight to specific frequency ranges
    // Low frequencies are weighted heavily as they reflect capacitive behavior
    const freq = refPoint.frequency;
    
    // Calculate logarithmic weight
    // Frequencies closer to the low end of the range get higher weight
    // This ensures frequency range changes have significant impact
    let frequencyWeight;
    
    // Determine which weighting scheme to use based on frequency range span
    // If the range spans less than 2 decades, use a more aggressive weighting
    const isNarrowRange = maxFreq / minFreq < 100;
    
    if (freq < 10) {
      // Low frequencies: enhanced weight for capacitive effects
      frequencyWeight = isNarrowRange ? 5.0 / Math.max(0.1, Math.log10(freq)) : 3.0 / Math.max(0.1, Math.log10(freq));
    } else if (freq < 1000) {
      // Mid frequencies: moderate weight for RC time constants
      frequencyWeight = isNarrowRange ? 2.5 / Math.max(0.5, Math.log10(freq)) : 1.5 / Math.max(0.5, Math.log10(freq));
    } else {
      // Higher frequencies: lower weight
      frequencyWeight = isNarrowRange ? 1.5 / Math.max(1.0, Math.log10(freq)) : 1.0 / Math.max(1.0, Math.log10(freq));
    }
    
    // Calculate the normalized residuals using reference magnitude
    const refMagnitude = Math.sqrt(refReal * refReal + refImag * refImag);
    const normFactor = refMagnitude > 0 ? refMagnitude : 1;
    
    // Calculate normalized residuals - separate real and imaginary components
    const realResidual = (testReal - refReal) / normFactor;
    const imagResidual = (testImag - refImag) / normFactor;
    
    // Calculate the weighted squared Euclidean distance in the complex plane
    // Give slightly more emphasis to imaginary component at low frequencies (capacitive behavior)
    const realWeight = freq < 100 ? 1.0 : 1.5;  // More weight to real component at higher freqs
    const imagWeight = freq < 100 ? 1.5 : 1.0;  // More weight to imaginary component at lower freqs
    
    const squaredResidual = (realWeight * realResidual * realResidual + 
                             imagWeight * imagResidual * imagResidual) * frequencyWeight;
    
    // Add to running total
    sumWeightedSquaredResiduals += squaredResidual;
    sumWeights += frequencyWeight;
    
    // Log detailed calculations for a few representative points
    if (i === 0 || i === Math.floor(referenceData.length / 2) || i === referenceData.length - 1) {
      logFunction(`Resnorm detail at ${refPoint.frequency.toExponential(2)} Hz:
  Reference: Z = ${refReal.toExponential(4)} + j${refImag.toExponential(4)} Ω
  Test: Z = ${testReal.toExponential(4)} + j${testImag.toExponential(4)} Ω
  |Z_ref| = ${refMagnitude.toExponential(4)} Ω
  Normalized residuals: real=${realResidual.toExponential(4)}, imag=${imagResidual.toExponential(4)}
  Frequency weight: ${frequencyWeight.toFixed(4)}
  Component weights: real=${realWeight.toFixed(1)}, imag=${imagWeight.toFixed(1)}
  Weighted squared residual: ${squaredResidual.toExponential(4)}`);
    }
  }
  
  // Apply an amplification factor to make differences more visible
  // This helps distinguish between small but significant differences
  const freqRangeRatio = maxFreq / minFreq;
  const rangeBasedAmplifier = freqRangeRatio < 100 ? 3.0 : 
                              freqRangeRatio < 1000 ? 2.5 : 2.0;
  
  // Calculate the final resnorm with dynamic amplification
  const finalResnorm = Math.sqrt(sumWeightedSquaredResiduals / sumWeights) * rangeBasedAmplifier;
  
  logFunction(`Final resnorm calculation:
  Total frequency points: ${Math.min(referenceData.length, testData.length)}
  Sum of weighted squared residuals: ${sumWeightedSquaredResiduals.toExponential(4)}
  Sum of weights: ${sumWeights.toFixed(4)}
  Frequency range ratio: ${freqRangeRatio.toFixed(1)} (${minFreq.toExponential(2)}-${maxFreq.toExponential(2)} Hz)
  Range-based amplifier: ${rangeBasedAmplifier.toFixed(1)}
  Raw resnorm value: ${(Math.sqrt(sumWeightedSquaredResiduals / sumWeights)).toExponential(4)}
  Amplified resnorm value (×${rangeBasedAmplifier}): ${finalResnorm.toExponential(4)}`);
  
  return finalResnorm;
}
