"use client";

import React, { useState, useEffect, useCallback } from 'react';
import 'katex/dist/katex.min.css';
import { 
  BackendMeshPoint, 
  GridParameterArrays 
} from './circuit-simulator/types';
import { ModelSnapshot, ImpedancePoint as TypesImpedancePoint, ResnormGroup } from './circuit-simulator/utils/types';
import { CircuitParameters } from './circuit-simulator/types/parameters';
import { useWorkerManager, WorkerProgress } from './circuit-simulator/utils/workerManager';

// Add imports for the new tab components at the top of the file
import { MathDetailsTab } from './circuit-simulator/MathDetailsTab';
import { DataTableTab } from './circuit-simulator/DataTableTab';
import { VisualizerTab } from './circuit-simulator/VisualizerTab';

// Import the new ToolboxComponent
import { ToolboxComponent } from './circuit-simulator/controls/ToolboxComponent';
import { PerformanceSettings, DEFAULT_PERFORMANCE_SETTINGS } from './circuit-simulator/controls/PerformanceControls';
import { ComputationNotification, ComputationSummary } from './circuit-simulator/notifications/ComputationNotification';

// Remove empty interface and replace with type
type CircuitSimulatorProps = Record<string, never>;

export const CircuitSimulator: React.FC<CircuitSimulatorProps> = () => {
  // Initialize worker manager for parallel computation
  const { computeGridParallel, cancelComputation } = useWorkerManager();
  
  // Add frequency control state
  const [minFreq, setMinFreq] = useState<number>(0.1); // 0.1 Hz
  const [maxFreq, setMaxFreq] = useState<number>(10000); // 10 kHz
  const [numPoints, setNumPoints] = useState<number>(20); // Default number of frequency points
  const [frequencyPoints, setFrequencyPoints] = useState<number[]>([]);
  
  // Add progress tracking for worker computation
  const [computationProgress, setComputationProgress] = useState<WorkerProgress | null>(null);

  // State for the circuit simulator
  const [gridResults, setGridResults] = useState<BackendMeshPoint[]>([]);
  const [gridResultsWithIds, setGridResultsWithIds] = useState<(BackendMeshPoint & { id: number })[]>([]);
  const [logMessages, setLogMessages] = useState<{time: string; message: string}[]>([]);
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [gridSize, setGridSize] = useState<number>(3);
  const [gridError, setGridError] = useState<string | null>(null);
  const [isComputingGrid, setIsComputingGrid] = useState<boolean>(false);
  const [resnormGroups, setResnormGroups] = useState<ResnormGroup[]>([]);
  // Initialize hidden groups state - start with only top 25% (excellent group) visible for performance
  const [hiddenGroups, setHiddenGroups] = useState<number[]>([1, 2, 3, 4]); // Hide all except group 0 (excellent)
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
    Ca: 0.5e-6, // 0.5 microfarads (converted to farads)
    Rb: 500,
    Cb: 0.5e-6, // 0.5 microfarads (converted to farads)
    frequency_range: [minFreq, maxFreq]
  });
  
  // Reference model state
  const [referenceModelId, setReferenceModelId] = useState<string | null>(null);
  const [referenceModel, setReferenceModel] = useState<ModelSnapshot | null>(null);
  
  // Ground truth parameters state
  const [groundTruthParams, setGroundTruthParams] = useState<CircuitParameters>({
    Rs: 24,
    Ra: 500,
    Ca: 0.5e-6, // 0.5 microfarads (converted to farads)
    Rb: 500,
    Cb: 0.5e-6, // 0.5 microfarads (converted to farads)
    frequency_range: [minFreq, maxFreq]
  });
  
  // Use a single parameters state object
  const [parameters, setParameters] = useState<CircuitParameters>({
    Rs: 50,
    Ra: 100,
    Ca: 0.5e-6, // 0.5 microfarads (converted to farads)
    Rb: 100,
    Cb: 0.5e-6, // 0.5 microfarads (converted to farads)
    frequency_range: [0.1, 10000]
  });
  
  // Add state for grid parameter arrays
  const [gridParameterArrays, setGridParameterArrays] = useState<GridParameterArrays | null>(null);
  
  // Add state for computation summary notification
  const [computationSummary, setComputationSummary] = useState<ComputationSummary | null>(null);
  
  // Performance settings state
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [performanceSettings, setPerformanceSettings] = useState<PerformanceSettings>(DEFAULT_PERFORMANCE_SETTINGS);
  
  // Current memory usage for performance monitoring
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const getCurrentMemoryUsage = useCallback((): number => {
    if (typeof window !== 'undefined' && 'performance' in window && 'memory' in performance) {
      const memory = (performance as typeof performance & { memory: { usedJSHeapSize: number } }).memory;
      return memory.usedJSHeapSize / (1024 * 1024);
    }
    return 0;
  }, []);
  
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
  }, []);

  // Initialize parameters
  useEffect(() => {
    // Calculate 50% values for each parameter range
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

    // Set default starting values at 50% of ranges
    setGroundTruthParams({
      Rs: rs50,
      Ra: ra50,
      Ca: ca50,
      Rb: rb50,
      Cb: cb50,
      frequency_range: [minFreq, maxFreq]
    });
    
    // Initial reference params will be set after the state updates
    updateStatusMessage('Initialized with default parameters at 50% of ranges');
  }, [updateStatusMessage, minFreq, maxFreq]);

  // Update reference parameters when slider values are initialized
  useEffect(() => {
    // Only create reference once on initial load
    if (referenceParams.Rs === 24 && 
        referenceParams.Ra === 500 && 
        referenceParams.Ca === 0.5e-6 && 
        referenceParams.Rb === 500 && 
        referenceParams.Cb === 0.5e-6 && 
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
    const impedanceData: TypesImpedancePoint[] = [];
    
    // Generate detailed logs for the first calculation (avoid calling updateStatusMessage in callback)
    
    for (const f of freqs) {
      // Calculate impedance for a single frequency
      const impedance = calculateCircuitImpedance(
        { Rs, Ra, Ca, Rb, Cb, frequency_range: [minFreq, maxFreq] }, 
        f
      );
      
      // Log removed to avoid circular dependencies
      
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
      parameters: { Rs, Ra, Ca, Rb, Cb, frequency_range: [minFreq, maxFreq] },
      data: impedanceData,
      color: '#FFFFFF',
      isVisible: true,
      opacity: 1,
      resnorm: 0,
      timestamp: Date.now(),
      ter: Rs + Ra + Rb
    };
  }, [groundTruthParams, minFreq, maxFreq, numPoints, frequencyPoints]);

  // Function to calculate impedance at a single frequency
  const calculateCircuitImpedance = (params: CircuitParameters, frequency: number) => {
    const { Rs, Ra, Ca, Rb, Cb } = params;
    const omega = 2 * Math.PI * frequency;
    
    // Calculate impedance of apical membrane (Ra || Ca)
    // Za(ω) = Ra/(1+jωRaCa)
    const Za_real = Ra / (1 + Math.pow(omega * Ra * Ca, 2));
    const Za_imag = -omega * Ra * Ra * Ca / (1 + Math.pow(omega * Ra * Ca, 2));
    
    // Calculate impedance of basal membrane (Rb || Cb)
    // Zb(ω) = Rb/(1+jωRbCb)
    const Zb_real = Rb / (1 + Math.pow(omega * Rb * Cb, 2));
    const Zb_imag = -omega * Rb * Rb * Cb / (1 + Math.pow(omega * Rb * Cb, 2));
    
    // Calculate sum of membrane impedances (Za + Zb)
    const Zab_real = Za_real + Zb_real;
    const Zab_imag = Za_imag + Zb_imag;
    
    // Calculate parallel combination: Z_total = (Rs * (Za + Zb)) / (Rs + Za + Zb)
    // Numerator: Rs * (Za + Zb)
    const num_real = Rs * Zab_real;
    const num_imag = Rs * Zab_imag;
    
    // Denominator: Rs + Za + Zb
    const denom_real = Rs + Zab_real;
    const denom_imag = Zab_imag;
    
    // Complex division: (num_real + j*num_imag) / (denom_real + j*denom_imag)
    const denom_mag_squared = denom_real * denom_real + denom_imag * denom_imag;
    
    const real = (num_real * denom_real + num_imag * denom_imag) / denom_mag_squared;
    const imaginary = (num_imag * denom_real - num_real * denom_imag) / denom_mag_squared;
    
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
  }, [referenceModelId, createReferenceModel]);

  // Create and show reference model by default
  useEffect(() => {
    // Create the reference model when groundTruthParams are initialized
    if (groundTruthParams.Rs && !referenceModelId && !manuallyHidden) {
      const newReferenceModel = createReferenceModel();
      setReferenceModel(newReferenceModel);
      setReferenceModelId('dynamic-reference');
      updateStatusMessage('Reference model created with current parameters');
    }
  }, [groundTruthParams.Rs, referenceModelId, createReferenceModel, manuallyHidden, updateStatusMessage]);

  // Update reference model when parameters change
  useEffect(() => {
    if (referenceModelId === 'dynamic-reference' && !manuallyHidden) {
      const updatedModel = createReferenceModel();
      setReferenceModel(updatedModel);
      updateStatusMessage('Reference model updated with current parameters');
    }
  }, [groundTruthParams, createReferenceModel, referenceModelId, manuallyHidden, updateStatusMessage]);
  
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
  }, [toggleReferenceModel, hiddenGroups, referenceModelId, createReferenceModel, updateStatusMessage]);
  
  // Optimized helper function to map BackendMeshPoint to ModelSnapshot
  const mapBackendMeshToSnapshot = (meshPoint: BackendMeshPoint, index: number): ModelSnapshot => {
    // Extract parameters efficiently (they should already be in Farads)
    const { Rs, Ra, Ca, Rb, Cb, frequency_range } = meshPoint.parameters;
    
    // Pre-calculate TER (Total Extracellular Resistance)
    const ter = Ra + Rb;
    
    // Only convert spectrum if it exists and has data
    const spectrumData = meshPoint.spectrum && meshPoint.spectrum.length > 0 
      ? toImpedancePoints(meshPoint.spectrum) as TypesImpedancePoint[]
      : [];
    
    return {
      id: `mesh-${index}`,
      name: `Mesh Point ${index}`,
      parameters: {
        Rs,
        Ra,
        Ca,
        Rb,
        Cb,
        frequency_range: frequency_range || parameters.frequency_range
      },
      data: spectrumData,
      color: '#3B82F6',
      isVisible: true,
      opacity: 1,
      resnorm: meshPoint.resnorm,
      timestamp: Date.now(),
      ter
    };
  };

  // Updated grid computation using Web Workers for parallel processing
  const handleComputeRegressionMesh = async () => {
    // Validate grid size
    if (gridSize < 2 || gridSize > 25) {
      updateStatusMessage('Points per dimension must be between 2 and 25');
      setTimeout(() => setStatusMessage(''), 3000);
      return;
    }

    const totalPointsToCompute = Math.pow(gridSize, 5);
    
    // Add warning for very large grids that might affect performance
    if (gridSize > 20) {
      const proceed = window.confirm(
        `⚠️ Warning: You're about to compute ${totalPointsToCompute.toLocaleString()} parameter combinations.\n\n` +
        `This will take several minutes and may make the page temporarily less responsive.\n\n` +
        `Consider using a smaller grid size (15-20) for faster results.\n\n` +
        `Continue with computation?`
      );
      if (!proceed) {
        updateStatusMessage('Grid computation cancelled by user');
        return;
      }
    }

    // Start timing
    const startTime = Date.now();
    let generationStartTime = 0;
    let computationStartTime = 0;
    let processingStartTime = 0;

    setIsComputingGrid(true);
    setGridResults([]);
    setGridError(null);
    setResnormGroups([]);
    setParameterChanged(false);
    setManuallyHidden(false);
    setComputationProgress(null);
    
    // Clear previous logs
    setLogMessages([]);
    
    const totalPoints = totalPointsToCompute;
    updateStatusMessage(`Starting parallel grid computation with ${gridSize} points per dimension...`);
    updateStatusMessage(`MATH: Computing all ${totalPoints.toLocaleString()} parameter combinations using ${navigator.hardwareConcurrency || 4} CPU cores`);
    updateStatusMessage(`MATH: Circuit model: Randles equivalent circuit (Rs in series with RC elements)`);
    updateStatusMessage(`MATH: Z(ω) = Rs + Ra/(1+jωRaCa) + Rb/(1+jωRbCb)`);
    updateStatusMessage(`MATH: Using frequency range ${minFreq.toFixed(1)}-${maxFreq.toFixed(1)} Hz with ${numPoints} points`);

    try {
      // Progress callback for worker updates
      const handleProgress = (progress: WorkerProgress) => {
        setComputationProgress(progress);
        
        if (progress.type === 'GENERATION_PROGRESS') {
          if (generationStartTime === 0) {
            generationStartTime = Date.now();
            updateStatusMessage(`[TIMING] Grid generation phase started`);
          }
          if (progress.generated && progress.generated > 0) {
            updateStatusMessage(`Generating grid points: ${progress.generated.toLocaleString()}/${progress.total.toLocaleString()} (${progress.overallProgress.toFixed(1)}%)`);
          }
          
          // Add small delay for large grids to keep UI responsive
          if (totalPoints > 100000 && progress.generated && progress.generated % 50000 === 0) {
            setTimeout(() => {}, 10);
          }
        } else if (progress.type === 'CHUNK_PROGRESS') {
          if (computationStartTime === 0) {
            computationStartTime = Date.now();
            const generationTime = ((computationStartTime - generationStartTime) / 1000).toFixed(2);
            updateStatusMessage(`[TIMING] Grid generation completed in ${generationTime}s, starting spectrum computation`);
          }
          const processedStr = progress.processed ? progress.processed.toLocaleString() : '0';
          const totalStr = progress.total.toLocaleString();
          updateStatusMessage(`Computing spectra & resnorms: ${processedStr}/${totalStr} points (${progress.overallProgress.toFixed(1)}%)`);
          
          // Add periodic UI updates for better responsiveness
          if (totalPoints > 500000 && progress.processed && progress.processed % 100000 === 0) {
            setTimeout(() => {}, 5);
          }
        }
      };

      // Error callback for worker errors
      const handleError = (error: string) => {
        setGridError(`Parallel computation failed: ${error}`);
        updateStatusMessage(`Error in parallel computation: ${error}`);
        setIsComputingGrid(false);
      };

      // Run the parallel computation
      const results = await computeGridParallel(
        groundTruthParams,
        gridSize,
        minFreq,
        maxFreq,
        numPoints,
        handleProgress,
        handleError
      );

      if (results.length === 0) {
        throw new Error('No results returned from computation');
      }

      // Start processing phase timing
      processingStartTime = Date.now();
      const computationTime = ((processingStartTime - computationStartTime) / 1000).toFixed(2);
      updateStatusMessage(`[TIMING] Spectrum computation completed in ${computationTime}s, starting results processing`);

      // Sort results by resnorm
      const sortingStart = Date.now();
      const sortedResults = results.sort((a, b) => a.resnorm - b.resnorm);
      const sortingTime = ((Date.now() - sortingStart) / 1000).toFixed(3);
      
      updateStatusMessage(`Parallel computation complete: ${sortedResults.length} points analyzed`);
      updateStatusMessage(`[TIMING] Results sorted by resnorm in ${sortingTime}s`);

      // Calculate resnorm range for grouping
      const groupingStart = Date.now();
      const resnorms = sortedResults.map(p => p.resnorm).filter(r => r > 0);
      
      // Use iterative approach to avoid stack overflow with large arrays
      let minResnorm = Infinity;
      let maxResnorm = -Infinity;
      for (const resnorm of resnorms) {
        if (resnorm < minResnorm) minResnorm = resnorm;
        if (resnorm > maxResnorm) maxResnorm = resnorm;
      }

      // Intelligent memory management for large datasets
      const estimateMemoryUsage = (count: number) => {
        // Rough estimate: each model ~500 bytes + spectrum data
        const avgSpectrumSize = numPoints * 40; // bytes per spectrum point
        return count * (500 + avgSpectrumSize) / 1024 / 1024; // MB
      };

      let MAX_VISUALIZATION_MODELS = 100000;
      const estimatedMemory = estimateMemoryUsage(sortedResults.length);
      
      // Adaptive limits based on dataset size and estimated memory usage
      if (estimatedMemory > 500) { // > 500MB
        MAX_VISUALIZATION_MODELS = 50000;
        updateStatusMessage(`Large dataset detected (${estimatedMemory.toFixed(1)}MB estimated), using aggressive sampling`);
      } else if (estimatedMemory > 200) { // > 200MB  
        MAX_VISUALIZATION_MODELS = 75000;
        updateStatusMessage(`Medium dataset detected (${estimatedMemory.toFixed(1)}MB estimated), using moderate sampling`);
      }
      
      const shouldLimitModels = sortedResults.length > MAX_VISUALIZATION_MODELS;
      const modelsToProcess = shouldLimitModels ? sortedResults.slice(0, MAX_VISUALIZATION_MODELS) : sortedResults;
      
      updateStatusMessage(`Processing ${modelsToProcess.length}/${sortedResults.length} models for visualization ${shouldLimitModels ? '(limited for performance)' : ''}`);

      // Map to ModelSnapshot format for visualization (optimized)
      const mappingStart = Date.now();
      const models: ModelSnapshot[] = [];
      
      // Process in smaller chunks to avoid stack overflow with spread operator
      const CHUNK_SIZE = 1000; // Reduced from 10000 to prevent spread operator stack overflow
      for (let i = 0; i < modelsToProcess.length; i += CHUNK_SIZE) {
        const chunk = modelsToProcess.slice(i, i + CHUNK_SIZE);
        const chunkModels = chunk.map((point, localIdx) => 
          mapBackendMeshToSnapshot(point, i + localIdx)
        );
        
        // Use Array.prototype.push.apply instead of spread operator to avoid stack overflow
        Array.prototype.push.apply(models, chunkModels);
        
        // Allow UI to update during processing
        if (i % (CHUNK_SIZE * 10) === 0) {
          setTimeout(() => {
            updateStatusMessage(`Mapping models for visualization: ${Math.min(i + CHUNK_SIZE, modelsToProcess.length)}/${modelsToProcess.length}`);
          }, 1);
        }
      }
      
      const mappingTime = ((Date.now() - mappingStart) / 1000).toFixed(3);
      updateStatusMessage(`[TIMING] Model mapping completed in ${mappingTime}s`);

      // Create resnorm groups
      updateStatusMessage('Creating visualization groups...');
      const groups = createResnormGroups(models, minResnorm, maxResnorm);
      setResnormGroups(groups);
      const groupingTime = ((Date.now() - groupingStart) / 1000).toFixed(3);
      updateStatusMessage(`[TIMING] Visualization groups created in ${groupingTime}s`);

      // Add unique IDs for table display (chunked to avoid memory issues)
      const pointsWithIds: (BackendMeshPoint & { id: number })[] = [];
      const ID_CHUNK_SIZE = 5000;
      
      for (let i = 0; i < sortedResults.length; i += ID_CHUNK_SIZE) {
        const chunk = sortedResults.slice(i, i + ID_CHUNK_SIZE);
        const chunkWithIds = chunk.map((point, localIdx) => ({
          ...point,
          id: i + localIdx + 1
        }));
        Array.prototype.push.apply(pointsWithIds, chunkWithIds);
      }

      // Update state
      setGridResults(sortedResults);
      setGridResultsWithIds(pointsWithIds);
      
      // Calculate final timing
      const endTime = Date.now();
      const totalTime = ((endTime - startTime) / 1000).toFixed(2);
      const generationTime = ((computationStartTime - generationStartTime) / 1000).toFixed(2);
      const spectrumTime = ((processingStartTime - computationStartTime) / 1000).toFixed(2);
      const processingTime = ((endTime - processingStartTime) / 1000).toFixed(2);

      // Log detailed timing breakdown
      updateStatusMessage(`[TIMING] Results processing completed in ${processingTime}s`);
      updateStatusMessage(`[SUMMARY] Grid computation completed successfully in ${totalTime}s total`);
      updateStatusMessage(`[BREAKDOWN] Grid generation: ${generationTime}s | Spectrum computation: ${spectrumTime}s | Results processing: ${processingTime}s`);
      updateStatusMessage(`[PERFORMANCE] Processed ${totalPoints.toLocaleString()} parameter combinations using ${navigator.hardwareConcurrency || 4} CPU cores`);
      updateStatusMessage(`[RESULTS] ${sortedResults.length} valid points, ${groups.length} resnorm groups created`);
      updateStatusMessage(`Resnorm range: ${minResnorm.toExponential(3)} to ${maxResnorm.toExponential(3)}`);
      updateStatusMessage(`Memory usage optimized: ${results.filter(r => r.spectrum.length > 0).length} points with full spectra`);
      
      // Create and show summary notification
      const summaryData: ComputationSummary = {
        title: 'Grid Computation Complete',
        totalTime,
        generationTime,
        computationTime: spectrumTime,
        processingTime,
        totalPoints,
        validPoints: sortedResults.length,
        groups: groups.length,
        cores: navigator.hardwareConcurrency || 4,
        throughput: totalPoints / parseFloat(totalTime),
        type: 'success',
        duration: 8000
      };

      // Log the summary to activity log with detailed breakdown
      updateStatusMessage(`[COMPLETION SUMMARY] Total: ${totalTime}s | Generation: ${generationTime}s | Computation: ${spectrumTime}s | Processing: ${processingTime}s | Throughput: ${(totalPoints / parseFloat(totalTime)).toFixed(0)} pts/s`);

      // Show the notification
      setComputationSummary(summaryData);

      // Update reference model
      const updatedModel = createReferenceModel();
      setReferenceModel(updatedModel);

      if (!referenceModelId) {
        setReferenceModelId('dynamic-reference');
        updateStatusMessage('Reference model created and shown');
      } else {
        updateStatusMessage('Reference model updated');
      }

      setIsComputingGrid(false);
      setComputationProgress(null);

    } catch (error) {
      console.error("Error in parallel computation:", error);
      const errorMessage = error instanceof Error ? error.message : String(error);
      const errorTime = ((Date.now() - startTime) / 1000).toFixed(2);
      
      setGridError(`Parallel grid computation failed: ${errorMessage}`);
      updateStatusMessage(`[ERROR] Computation failed after ${errorTime}s: ${errorMessage}`);
      updateStatusMessage(`[TIMING] Error occurred during computation phase`);
      
      // Show error notification
      const errorSummary: ComputationSummary = {
        title: '❌ Grid Computation Failed',
        totalTime: errorTime,
        generationTime: generationStartTime > 0 ? ((Date.now() - generationStartTime) / 1000).toFixed(2) : '0',
        computationTime: computationStartTime > 0 ? ((Date.now() - computationStartTime) / 1000).toFixed(2) : '0',
        processingTime: '0',
        totalPoints,
        validPoints: 0,
        groups: 0,
        cores: navigator.hardwareConcurrency || 4,
        throughput: 0,
        type: 'error',
        duration: 10000 // Show error longer
      };
      
      setComputationSummary(errorSummary);
      
      setIsComputingGrid(false);
      setComputationProgress(null);
    }
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
    setParameters((prev: CircuitParameters) => ({
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

  // Update reference model when frequency settings change
  useEffect(() => {
    if (referenceModelId === 'dynamic-reference' && !manuallyHidden) {
      const updatedModel = createReferenceModel();
      setReferenceModel(updatedModel);
      updateStatusMessage('Reference model updated with new frequency range');
    }
  }, [minFreq, maxFreq, numPoints, frequencyPoints.length, referenceModelId, manuallyHidden, createReferenceModel, updateStatusMessage]);

  // Handler for grid values generated by spider plot
  const handleGridValuesGenerated = useCallback((values: GridParameterArrays) => {
    if (!values) return;
    
    // Log the values for debugging
    console.log('Grid values received:', values);
    
    // Only update if values have actually changed
    setGridParameterArrays(prev => {
      if (!prev) return values;
      
      // Quick equality check for performance
      if (prev === values) return prev;
      
      // Check if any arrays have changed length
      const keys = Object.keys(values) as Array<keyof GridParameterArrays>;
      const hasLengthChanged = keys.some(key => 
        !prev[key] || prev[key].length !== values[key].length
      );
      
      if (hasLengthChanged) return values;
      
      // Deep compare arrays
      const hasChanged = keys.some(key => {
        const prevArr = prev[key];
        const newArr = values[key];
        return !prevArr.every((val, idx) => val === newArr[idx]);
      });
      
      return hasChanged ? values : prev;
    });
  }, []);

  // Note: Grid generation and spectrum calculation now handled by Web Workers

  // State for collapsible sidebar
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // Modify the main content area to show the correct tab content
  return (
    <div className="min-h-screen bg-slate-900 text-white flex flex-col">
      <div className="flex flex-col h-full">
        {/* Header - Sticky and Modernized */}
        <header className="sticky top-0 z-50 bg-slate-800 border-b border-slate-700 px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            {/* Sidebar toggle button */}
            <button
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              className="p-2 rounded-md bg-neutral-700 hover:bg-neutral-600 transition-colors"
              title={sidebarCollapsed ? "Show Sidebar" : "Hide Sidebar"}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                {sidebarCollapsed ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7M19 19l-7-7 7-7" />
                )}
              </svg>
            </button>

            <h1 className="text-2xl font-bold text-white">Circuit Simulator</h1>
            
            {/* Visualization tabs */}
            <div className="flex bg-neutral-800 rounded-md overflow-hidden">
              <button 
                className={`px-4 py-1.5 text-sm font-medium ${visualizationTab === 'visualizer' ? 'bg-primary text-white' : 'text-neutral-300 hover:bg-neutral-700 transition-colors'}`}
                onClick={() => setVisualizationTab('visualizer')}
              >
                Visualizer
              </button>
              <button 
                className={`px-4 py-1.5 text-sm font-medium ${visualizationTab === 'math' ? 'bg-primary text-white' : 'text-neutral-300 hover:bg-neutral-700 transition-colors'}`}
                onClick={() => setVisualizationTab('math')}
              >
                Math Details
              </button>
              <button 
                className={`px-4 py-1.5 text-sm font-medium ${visualizationTab === 'data' ? 'bg-primary text-white' : 'text-neutral-300 hover:bg-neutral-700 transition-colors'}`}
                onClick={() => setVisualizationTab('data')}
              >
                Data Table
              </button>
            </div>
          </div>
          
          {/* Grid point counter and frequency info */}
          <div className="flex items-center gap-4">
            <div className="bg-neutral-800 rounded-md px-3 py-1.5 text-xs">
              <span className="text-neutral-400">Grid Points: </span>
              <span className="text-white font-medium">{gridResults.length}</span>
              <span className="text-neutral-400 ml-1.5">|</span>
              <span className="text-neutral-400 ml-1.5">Freq: </span>
              <span className="text-white font-medium">{minFreq.toFixed(2)} - {maxFreq.toFixed(0)} Hz</span>
            </div>
          </div>
        </header>
        
        {/* Main content with collapsible sidebar and visualization area */}
        <div className="flex flex-1 overflow-hidden">
          {/* Collapsible Sidebar */}
          <div className={`transition-all duration-300 ease-in-out ${sidebarCollapsed ? 'w-0 -ml-4' : 'w-[350px]'} overflow-hidden`}>
            <ToolboxComponent 
              // Grid computation props
              gridSize={gridSize}
              setGridSize={setGridSize}
              minFreq={minFreq}
              setMinFreq={setMinFreq}
              maxFreq={maxFreq}
              setMaxFreq={setMaxFreq}
              numPoints={numPoints}
              setNumPoints={setNumPoints}
              updateFrequencies={updateFrequencies}
              updateStatusMessage={updateStatusMessage}
              parameterChanged={parameterChanged}
              setParameterChanged={setParameterChanged}
              handleComputeRegressionMesh={handleComputeRegressionMesh}
              isComputingGrid={isComputingGrid}
              gridResults={gridResults}
              
              // Circuit parameters props
              groundTruthParams={groundTruthParams}
              setGroundTruthParams={setGroundTruthParams}
              referenceModelId={referenceModelId}
              createReferenceModel={createReferenceModel}
              setReferenceModel={setReferenceModel}
              
              // Utility functions
              logToLinearSlider={logToLinearSlider}
              linearSliderToLog={linearSliderToLog}
              
              // Log messages
              logMessages={logMessages}
            />
          </div>
                  
          {/* Main Content Area */}
          <div className="flex-1 bg-slate-900 overflow-y-auto">
            {/* Status message bar */}
            {statusMessage && (
              <div className="bg-blue-600 text-white px-4 py-2 text-sm border-b border-slate-700">
                <div className="flex items-center">
                  <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  {statusMessage}
                </div>
              </div>
            )}
                    
            {/* Main visualization area */}
            <div className="p-6 bg-slate-900" style={{ minHeight: '600px' }}>
              {isComputingGrid ? (
                <div className="flex flex-col items-center justify-center p-6 h-40 bg-neutral-100/5 border border-neutral-700 rounded-lg shadow-md">
                  <div className="flex items-center mb-4">
                    <div className="animate-spin rounded-full h-6 w-6 border-t-2 border-b-2 border-primary mr-3"></div>
                    <div>
                      <p className="text-sm text-primary font-medium">
                        {computationProgress?.type === 'GENERATION_PROGRESS' 
                          ? 'Generating grid points...' 
                          : 'Computing in parallel...'}
                      </p>
                      <p className="text-xs text-neutral-400 mt-1">
                        {computationProgress ? 
                          `${computationProgress.overallProgress.toFixed(1)}% complete` : 
                          'Using multiple CPU cores for faster computation'}
                      </p>
                    </div>
                  </div>
                  
                  {/* Progress bar */}
                  {computationProgress && (
                    <div className="w-full max-w-md">
                      <div className="w-full bg-neutral-700 rounded-full h-2">
                        <div 
                          className="bg-primary h-2 rounded-full transition-all duration-300" 
                          style={{ width: `${computationProgress.overallProgress}%` }}
                        ></div>
                      </div>
                    </div>
                  )}
                  
                  {/* Cancel button */}
                  <button
                    onClick={cancelComputation}
                    className="mt-4 px-4 py-2 text-xs bg-red-600 hover:bg-red-700 text-white rounded-md transition-colors"
                  >
                    Cancel Computation
                  </button>
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
                  gridParameterArrays={gridParameterArrays}
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
                    gridParameterArrays={gridParameterArrays}
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
                  <VisualizerTab 
                    resnormGroups={resnormGroups}
                    hiddenGroups={hiddenGroups}
                    opacityLevel={opacityLevel}
                    referenceModelId={referenceModelId}
                    gridSize={gridSize}
                    onGridValuesGenerated={handleGridValuesGenerated}
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
      
      {/* Computation Summary Notification */}
      <ComputationNotification 
        summary={computationSummary}
        onDismiss={() => setComputationSummary(null)}
      />
    </div>
  );
} 

// Helper to convert backend spectrum to ImpedancePoint[] (types.ts shape)
function toImpedancePoints(spectrum: { freq: number; real: number; imag: number; mag: number; phase: number; }[]): TypesImpedancePoint[] {
  return spectrum.map((p): TypesImpedancePoint => ({
    frequency: p.freq,
    real: p.real,
    imaginary: p.imag,
    magnitude: p.mag,
    phase: p.phase
  }));
}

// --- Complex math operations now handled in Web Workers ---

// Note: Complex number operations, impedance calculations, and resnorm calculations
// are now handled efficiently in Web Workers for parallel processing

// Optimized resnorm grouping logic using percentiles instead of hardline boundaries
const createResnormGroups = (models: ModelSnapshot[], minResnorm: number, maxResnorm: number): ResnormGroup[] => {
  // Limit processing for very large datasets to prevent memory issues
  const MAX_MODELS_PER_GROUP = 50000;
  const shouldSample = models.length > MAX_MODELS_PER_GROUP;
  
  console.log(`Creating percentile-based resnorm groups for ${models.length} models (sampling: ${shouldSample})`);
  
  // Extract all finite resnorm values for percentile calculation
  const validResnorms = models
    .map(m => m.resnorm)
    .filter(r => r !== undefined && isFinite(r)) as number[];
  
  if (validResnorms.length === 0) {
    console.warn('No valid resnorm values found for grouping');
    return [];
  }

  // Sort resnorms for percentile calculation
  validResnorms.sort((a, b) => a - b);
  
  // Calculate percentile thresholds
  const calculatePercentile = (arr: number[], percentile: number): number => {
    const index = Math.ceil((percentile / 100) * arr.length) - 1;
    return arr[Math.max(0, Math.min(index, arr.length - 1))];
  };

  const percentileThresholds = {
    p25: calculatePercentile(validResnorms, 25),  // Top 25% (best fits)
    p50: calculatePercentile(validResnorms, 50),  // Top 50% (good fits)
    p75: calculatePercentile(validResnorms, 75),  // Top 75% (moderate fits)
    p90: calculatePercentile(validResnorms, 90)   // Top 90% (acceptable fits)
  };

  console.log('Percentile thresholds:', percentileThresholds);

  // Pre-allocate arrays for each percentile group
  const excellentItems: ModelSnapshot[] = [];  // Top 25%
  const goodItems: ModelSnapshot[] = [];       // 25-50%
  const moderateItems: ModelSnapshot[] = [];   // 50-75%
  const acceptableItems: ModelSnapshot[] = []; // 75-90%
  const poorItems: ModelSnapshot[] = [];       // Bottom 10%

  // Single pass classification with performance optimization
  let processed = 0;
  const maxToProcess = shouldSample ? MAX_MODELS_PER_GROUP : models.length;
  const step = shouldSample ? Math.ceil(models.length / MAX_MODELS_PER_GROUP) : 1;

  for (let i = 0; i < models.length && processed < maxToProcess; i += step) {
    const model = models[i];
    
    if (!model || model.resnorm === undefined || !isFinite(model.resnorm)) continue;
    
    const resnorm = model.resnorm;
    
    // Classify into percentile groups with performance-optimized colors
    if (resnorm <= percentileThresholds.p25) {
      excellentItems.push({ ...model, color: '#059669' }); // Emerald-600 (excellent)
    } else if (resnorm <= percentileThresholds.p50) {
      goodItems.push({ ...model, color: '#10B981' });      // Emerald-500 (good)
    } else if (resnorm <= percentileThresholds.p75) {
      moderateItems.push({ ...model, color: '#F59E0B' }); // Amber-500 (moderate)
    } else if (resnorm <= percentileThresholds.p90) {
      acceptableItems.push({ ...model, color: '#F97316' }); // Orange-500 (acceptable)
    } else {
      poorItems.push({ ...model, color: '#DC2626' });      // Red-600 (poor)
    }
    
    processed++;
    
    // Progress logging for large datasets
    if (processed % 10000 === 0) {
      console.log(`Processed ${processed}/${maxToProcess} models for percentile grouping`);
    }
  }

  console.log(`Percentile group sizes: Excellent(≤25%): ${excellentItems.length}, Good(25-50%): ${goodItems.length}, Moderate(50-75%): ${moderateItems.length}, Acceptable(75-90%): ${acceptableItems.length}, Poor(>90%): ${poorItems.length}`);

  // Create groups array with performance focus - excellent group visible by default
  const groups: ResnormGroup[] = [];

  if (excellentItems.length > 0) {
    groups.push({
      range: [minResnorm, percentileThresholds.p25] as [number, number],
      color: '#059669',
      label: 'Excellent Fit (Top 25%)',
      description: `Resnorm ≤ ${percentileThresholds.p25.toExponential(2)} (best ${excellentItems.length} models)`,
      items: excellentItems
    });
  }

  if (goodItems.length > 0) {
    groups.push({
      range: [percentileThresholds.p25, percentileThresholds.p50] as [number, number],
      color: '#10B981',
      label: 'Good Fit (25-50%)',
      description: `${percentileThresholds.p25.toExponential(2)} < Resnorm ≤ ${percentileThresholds.p50.toExponential(2)} (${goodItems.length} models)`,
      items: goodItems
    });
  }

  if (moderateItems.length > 0) {
    groups.push({
      range: [percentileThresholds.p50, percentileThresholds.p75] as [number, number],
      color: '#F59E0B',
      label: 'Moderate Fit (50-75%)',
      description: `${percentileThresholds.p50.toExponential(2)} < Resnorm ≤ ${percentileThresholds.p75.toExponential(2)} (${moderateItems.length} models)`,
      items: moderateItems
    });
  }

  if (acceptableItems.length > 0) {
    groups.push({
      range: [percentileThresholds.p75, percentileThresholds.p90] as [number, number],
      color: '#F97316',
      label: 'Acceptable Fit (75-90%)',
      description: `${percentileThresholds.p75.toExponential(2)} < Resnorm ≤ ${percentileThresholds.p90.toExponential(2)} (${acceptableItems.length} models)`,
      items: acceptableItems
    });
  }

  if (poorItems.length > 0) {
    groups.push({
      range: [percentileThresholds.p90, maxResnorm] as [number, number],
      color: '#DC2626',
      label: 'Poor Fit (Bottom 10%)',
      description: `Resnorm > ${percentileThresholds.p90.toExponential(2)} (worst ${poorItems.length} models)`,
      items: poorItems
    });
  }

  return groups;
};
