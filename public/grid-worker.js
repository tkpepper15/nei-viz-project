// Enhanced Web Worker for efficient grid computation with optimized data transfer
// This runs on a separate thread to avoid blocking the UI

// Shared data cache to minimize transfer overhead
let sharedFrequencies = null;
let sharedReferenceSpectrum = null;
let workerId = null;

// Complex number operations
const complex = {
  add: (a, b) => ({
    real: a.real + b.real,
    imag: a.imag + b.imag
  }),
  multiply: (a, b) => ({
    real: a.real * b.real - a.imag * b.imag,
    imag: a.real * b.imag + a.imag * b.real
  }),
  divide: (a, b) => {
    const denom = b.real * b.real + b.imag * b.imag;
    return {
      real: (a.real * b.real + a.imag * b.imag) / denom,
      imag: (a.imag * b.real - a.real * b.imag) / denom
    };
  }
};

// Calculate impedance spectrum for a single parameter set using parallel configuration
function calculateImpedanceSpectrum(params, freqs) {
  const { Rsh, Ra, Ca, Rb, Cb } = params;
  
  return freqs.map(freq => {
    const omega = 2 * Math.PI * freq;
    
    // Za = Ra/(1+jωRaCa)
    const Za_denom = complex.add(
      { real: 1, imag: 0 },
      { real: 0, imag: omega * Ra * Ca }
    );
    const Za = complex.divide(
      { real: Ra, imag: 0 },
      Za_denom
    );
    
    // Zb = Rb/(1+jωRbCb)
    const Zb_denom = complex.add(
      { real: 1, imag: 0 },
      { real: 0, imag: omega * Rb * Cb }
    );
    const Zb = complex.divide(
      { real: Rb, imag: 0 },
      Zb_denom
    );
    
    // Calculate sum of membrane impedances (Za + Zb)
    const Zab = complex.add(Za, Zb);
    
    // Calculate parallel combination: Z_total = (Rsh * (Za + Zb)) / (Rsh + Za + Zb)
    // Numerator: Rsh * (Za + Zb)
    const numerator = complex.multiply(
      { real: Rsh, imag: 0 },
      Zab
    );
    
    // Denominator: Rsh + Za + Zb
    const denominator = complex.add(
      { real: Rsh, imag: 0 },
      Zab
    );
    
    // Z_total = numerator / denominator
    const Z_total = complex.divide(numerator, denominator);
    
    const magnitude = Math.sqrt(Z_total.real * Z_total.real + Z_total.imag * Z_total.imag);
    const phase = Math.atan2(Z_total.imag, Z_total.real) * (180 / Math.PI);
    
    return {
      freq,
      real: Z_total.real,
      imag: Z_total.imag,
      mag: magnitude,
      phase
    };
  });
}

// Calculate resnorm using configurable methods
function calculateResnorm(referenceSpectrum, testSpectrum, resnormConfig = { method: 'mae' }) {
  if (!referenceSpectrum.length || !testSpectrum.length) return Infinity;
  
  let totalError = 0;
  let sumWeights = 0;
  const n = Math.min(referenceSpectrum.length, testSpectrum.length);
  
  for (let i = 0; i < n; i++) {
    const refPoint = referenceSpectrum[i];
    const testPoint = testSpectrum[i];
    
    // Skip if frequencies don't match (within tolerance)
    if (Math.abs(refPoint.freq - testPoint.freq) / refPoint.freq > 0.001) {
      continue;
    }
    
    // Calculate magnitudes
    const testMag = Math.sqrt(testPoint.real * testPoint.real + testPoint.imag * testPoint.imag);
    const refMag = Math.sqrt(refPoint.real * refPoint.real + refPoint.imag * refPoint.imag);
    
    // Skip points with zero reference magnitude to avoid division by zero
    if (refMag === 0) {
      continue;
    }
    
    // Calculate normalized residual (relative error) - EIS standard practice
    const normalizedResidual = (testMag - refMag) / refMag;
    
    let error;
    
    // Calculate error based on selected method
    switch (resnormConfig.method) {
      case 'ssr':
        // Sum of Squared Residuals: sqrt((real_diff)² + (imag_diff)²)
        const realDiff = testPoint.real - refPoint.real;
        const imagDiff = testPoint.imag - refPoint.imag;
        error = Math.sqrt(realDiff * realDiff + imagDiff * imagDiff);
        break;
        
      case 'mae':
        error = Math.abs(normalizedResidual);
        break;
        
      case 'rmse':
        error = normalizedResidual * normalizedResidual; // Will take sqrt later
        break;
        
        
      default:
        error = Math.abs(normalizedResidual); // Default to normalized MAE
    }
    
    // Apply frequency weighting if enabled
    let weight = 1.0;
    if (resnormConfig.useFrequencyWeighting && refPoint.freq > 0) {
      weight = Math.pow(refPoint.freq, -0.5);
    }
    
    totalError += weight * error;
    sumWeights += weight;
  }
  
  // Calculate final result based on method
  if (sumWeights === 0) return Infinity;
  
  switch (resnormConfig.method) {
    case 'ssr':
      return totalError / sumWeights;  // Mean of squared residuals
      
    case 'mae':
      return totalError / sumWeights;  // Simple average
      
    case 'rmse':
      return Math.sqrt(totalError / sumWeights);  // Root mean squared error
      
    default:
      return totalError / sumWeights;
  }
}

// Generate logarithmic space
function generateLogSpace(min, max, num) {
  const logMin = Math.log10(min);
  const logMax = Math.log10(max);
  const step = (logMax - logMin) / (num - 1);
  
  const result = [];
  for (let i = 0; i < num; i++) {
    result.push(Math.pow(10, logMin + i * step));
  }
  return result;
}

// Generate logarithmic space with ground truth value included
function generateLogSpaceWithReference(min, max, num, referenceValue) {
  // First generate standard logarithmic space
  const logValues = generateLogSpace(min, max, num);
  
  // If reference value is within range and not already close to a sampled point
  if (referenceValue >= min && referenceValue <= max) {
    // Find the closest existing point
    let closestIndex = 0;
    let minDistance = Math.abs(Math.log10(logValues[0]) - Math.log10(referenceValue));
    
    for (let i = 1; i < logValues.length; i++) {
      const distance = Math.abs(Math.log10(logValues[i]) - Math.log10(referenceValue));
      if (distance < minDistance) {
        minDistance = distance;
        closestIndex = i;
      }
    }
    
    // If the closest point is more than 5% away in log space, replace it with reference value
    const logRef = Math.log10(referenceValue);
    const logClosest = Math.log10(logValues[closestIndex]);
    const relativeError = Math.abs(logRef - logClosest) / Math.abs(logRef);
    
    if (relativeError > 0.05) {
      logValues[closestIndex] = referenceValue;
      // Re-sort the array to maintain order
      logValues.sort((a, b) => a - b);
    }
  }
  
  return logValues;
}

// Stream grid points in chunks to prevent memory overflow
function* streamGridPoints(gridSize, useSymmetricGrid, inputGroundTruth, resnormConfig = { method: 'mae' }) {
  const totalPoints = Math.pow(gridSize, 5);
  
  // Ground truth parameters (reference values to ensure are included)
  // Use input parameters if provided, otherwise use defaults
  const groundTruthParams = inputGroundTruth || {
    Rsh: 24,        // Shunt resistance (Ω)
    Ra: 500,       // Apical resistance (Ω) 
    Ca: 0.5e-6,    // Apical capacitance (F)
    Rb: 500,       // Basal resistance (Ω)
    Cb: 0.5e-6     // Basal capacitance (F)
  };
  
  // Generate parameter ranges with ground truth values included
  // Use consistent ranges: all resistance parameters use 10-10000 Ω
  const rsValues = generateLogSpaceWithReference(10, 10000, gridSize, groundTruthParams.Rsh);
  const raValues = generateLogSpaceWithReference(10, 10000, gridSize, groundTruthParams.Ra);
  const rbValues = generateLogSpaceWithReference(10, 10000, gridSize, groundTruthParams.Rb);
  const caValues = generateLogSpaceWithReference(0.1e-6, 50e-6, gridSize, groundTruthParams.Ca);
  const cbValues = generateLogSpaceWithReference(0.1e-6, 50e-6, gridSize, groundTruthParams.Cb);
  
  // Debug: Check if ground truth values are included
  const debugInfo = {
    Rsh_included: rsValues.includes(groundTruthParams.Rsh),
    Ra_included: raValues.includes(groundTruthParams.Ra),
    Rb_included: rbValues.includes(groundTruthParams.Rb),
    Ca_included: caValues.includes(groundTruthParams.Ca),
    Cb_included: cbValues.includes(groundTruthParams.Cb),
    rs_closest: rsValues.reduce((prev, curr) => Math.abs(curr - groundTruthParams.Rsh) < Math.abs(prev - groundTruthParams.Rsh) ? curr : prev),
    ra_closest: raValues.reduce((prev, curr) => Math.abs(curr - groundTruthParams.Ra) < Math.abs(prev - groundTruthParams.Ra) ? curr : prev),
    rb_closest: rbValues.reduce((prev, curr) => Math.abs(curr - groundTruthParams.Rb) < Math.abs(prev - groundTruthParams.Rb) ? curr : prev),
    ca_closest: caValues.reduce((prev, curr) => Math.abs(curr - groundTruthParams.Ca) < Math.abs(prev - groundTruthParams.Ca) ? curr : prev),
    cb_closest: cbValues.reduce((prev, curr) => Math.abs(curr - groundTruthParams.Cb) < Math.abs(prev - groundTruthParams.Cb) ? curr : prev)
  };
  
  console.log('Ground Truth Parameter Inclusion Debug:', debugInfo);
  
  let generatedCount = 0;
  
  // Use iterative approach to avoid stack overflow
  for (let pointIndex = 0; pointIndex < totalPoints; pointIndex++) {
    // Convert linear index to 5D coordinates
    let temp = pointIndex;
    const cbIndex = temp % gridSize;
    temp = Math.floor(temp / gridSize);
    const rbIndex = temp % gridSize;
    temp = Math.floor(temp / gridSize);
    const caIndex = temp % gridSize;
    temp = Math.floor(temp / gridSize);
    const raIndex = temp % gridSize;
    const rsIndex = Math.floor(temp / gridSize);
    
    const Rsh = rsValues[rsIndex];
    const Ra = raValues[raIndex];
    const Ca = caValues[caIndex];
    const Rb = rbValues[rbIndex];
    const Cb = cbValues[cbIndex];
    
    // Symmetric grid optimization: skip duplicates where Ra/Ca > Rb/Cb
    if (useSymmetricGrid) {
      // Calculate time constants tau = RC for comparison
      const tauA = Ra * Ca;
      const tauB = Rb * Cb;
      
      // Skip this combination if tauA > tauB (we'll get the equivalent from the swapped version)
      if (tauA > tauB) {
        continue;
      }
      
      // If time constants are equal, enforce Ra <= Rb to break ties
      if (Math.abs(tauA - tauB) < 1e-15 && Ra > Rb) {
        continue;
      }
    }
    
    generatedCount++;
    
    // Yield the point and progress info
    yield {
      point: { Rsh, Ra, Ca, Rb, Cb },
      progress: {
        generated: generatedCount,
        processed: pointIndex + 1,
        total: totalPoints
      }
    };
  }
}

// Optimized chunk processing function with minimal data transfer
async function processChunkOptimized(chunkParams, frequencyArray, chunkIndex, totalChunks, referenceSpectrum, resnormConfig, taskId, maxComputationResults = 5000) {
  // Ultra-efficient Top-N heap for massive datasets
  const MAX_RESULTS_DURING_COMPUTATION = maxComputationResults; // User-configurable limit
  const FINAL_TOP_RESULTS = maxComputationResults; // Final top results to return (user configurable)
  
  // Efficient min-heap implementation for Top-N results (O(log n) operations)
  class MinHeap {
    constructor(maxSize) {
      this.heap = [];
      this.maxSize = maxSize;
    }
    
    parent(index) { return Math.floor((index - 1) / 2); }
    leftChild(index) { return 2 * index + 1; }
    rightChild(index) { return 2 * index + 2; }
    
    swap(i, j) {
      [this.heap[i], this.heap[j]] = [this.heap[j], this.heap[i]];
    }
    
    heapifyUp(index) {
      if (index > 0 && this.heap[this.parent(index)].resnorm > this.heap[index].resnorm) {
        this.swap(index, this.parent(index));
        this.heapifyUp(this.parent(index));
      }
    }
    
    heapifyDown(index) {
      let smallest = index;
      const left = this.leftChild(index);
      const right = this.rightChild(index);
      
      if (left < this.heap.length && this.heap[left].resnorm < this.heap[smallest].resnorm) {
        smallest = left;
      }
      if (right < this.heap.length && this.heap[right].resnorm < this.heap[smallest].resnorm) {
        smallest = right;
      }
      if (smallest !== index) {
        this.swap(index, smallest);
        this.heapifyDown(smallest);
      }
    }
    
    insert(item) {
      if (this.heap.length < this.maxSize) {
        // Heap not full, just add
        this.heap.push(item);
        this.heapifyUp(this.heap.length - 1);
      } else if (item.resnorm < this.heap[0].resnorm) {
        // Replace worst (root) with better item
        this.heap[0] = item;
        this.heapifyDown(0);
      }
      // Otherwise ignore (item is worse than our worst)
    }
    
    getAll() {
      return [...this.heap].sort((a, b) => a.resnorm - b.resnorm);
    }
    
    size() {
      return this.heap.length;
    }
  }
  
  const topResults = new MinHeap(MAX_RESULTS_DURING_COMPUTATION);
  let processedCount = 0;
  
  // Enhanced adaptive batch sizing based on chunk size
  let batchSize = 25;
  let memoryThreshold = 100 * 1024 * 1024;
  
  if (chunkParams.length > 80000) {
    batchSize = 2; // Ultra micro-batches for massive chunks
    memoryThreshold = 30 * 1024 * 1024;
  } else if (chunkParams.length > 50000) {
    batchSize = 3;
    memoryThreshold = 40 * 1024 * 1024;
  } else if (chunkParams.length > 25000) {
    batchSize = 5;
  } else if (chunkParams.length > 10000) {
    batchSize = 10;
  }
  
  let estimatedMemoryUsage = 0;
  const bytesPerResult = 800;
  
  for (let i = 0; i < chunkParams.length; i += batchSize) {
    const batch = chunkParams.slice(i, i + batchSize);
    
    // Enhanced memory pressure monitoring
    estimatedMemoryUsage = topResults.size() * bytesPerResult;
    if (estimatedMemoryUsage > memoryThreshold) {
      self.postMessage({
        type: 'MEMORY_PRESSURE',
        data: {
          chunkIndex,
          taskId,
          estimatedMemory: Math.round(estimatedMemoryUsage / 1024 / 1024) + 'MB',
          resultsCount: topResults.size(),
          message: 'High memory usage detected in optimized worker'
        }
      });
      
      // Aggressive memory management for large datasets
      if (topResults.size() > 3000) {
        // For heap-based approach, we don't need to splice - just report current status
        const currentResults = topResults.getAll().slice(0, 500);
        self.postMessage({
          type: 'PARTIAL_RESULTS',
          data: {
            chunkIndex,
            taskId,
            partialResults: currentResults,
            totalPartialCount: topResults.size()
          }
        });
        estimatedMemoryUsage = topResults.size() * bytesPerResult;
      }
      
      // Yield more frequently for large datasets
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    
    // Enhanced yielding for very large datasets
    if (chunkParams.length > 100000 && i % (batchSize * 3) === 0) {
      await new Promise(resolve => setTimeout(resolve, 30));
    } else if (chunkParams.length > 80000 && i % (batchSize * 5) === 0) {
      await new Promise(resolve => setTimeout(resolve, 25));
    } else if (chunkParams.length > 50000 && i % (batchSize * 8) === 0) {
      await new Promise(resolve => setTimeout(resolve, 15));
    }
    
    for (const params of batch) {
      try {
        if (!params || typeof params.Rsh !== 'number') continue;
        
        const spectrum = calculateImpedanceSpectrum(params, frequencyArray);
        const resnorm = calculateResnorm(referenceSpectrum, spectrum, resnormConfig);
        
        if (isFinite(resnorm)) {
          // Ultra-efficient Top-N: store minimal data during computation
          // Only store parameters + resnorm during computation phase
          topResults.insert({
            parameters: params,
            resnorm: resnorm
            // NO spectrum data during computation - saves massive memory
          });
          
          processedCount++;
        }
      } catch (error) {
        console.warn('Parameter processing error:', error);
      }
    }
    
    // Progressive refinement: send best results found so far
    const progress = (i + batchSize) / chunkParams.length;
    if (i % (batchSize * 50) === 0 || progress >= 1) {
      // Send current best results for progressive visualization
      const currentBest = topResults.getAll().slice(0, 100);
      
      self.postMessage({
        type: 'CHUNK_PROGRESS',
        data: {
          chunkIndex,
          taskId,
          totalChunks,
          chunkProgress: progress,
          processed: Math.min(i + batchSize, chunkParams.length),
          total: chunkParams.length,
          // Progressive refinement data
          currentBestResults: currentBest,
          currentBestResnorm: currentBest.length > 0 ? currentBest[0].resnorm : null,
          totalQualityResults: topResults.size()
        }
      });
    }
  }
  
  // Get final results from heap (already optimized)
  const finalResults = topResults.getAll();
  
  // Now add spectrum data ONLY to top results for final output
  const topResultsWithSpectra = [];
  const otherResults = [];
  
  // Re-compute spectra only for the top results
  for (let i = 0; i < finalResults.length; i++) {
    const result = finalResults[i];
    
    if (i < FINAL_TOP_RESULTS) {
      // Re-calculate spectrum for top results only  
      const spectrum = calculateImpedanceSpectrum(result.parameters, frequencyArray);
      topResultsWithSpectra.push({
        parameters: result.parameters,
        resnorm: result.resnorm,
        spectrum: spectrum
      });
    } else {
      // No spectrum for lower-priority results
      otherResults.push({
        parameters: result.parameters,
        resnorm: result.resnorm
      });
    }
  }
  
  self.postMessage({
    type: 'CHUNK_COMPLETE',
    data: {
      chunkIndex,
      taskId,
      topResults: topResultsWithSpectra,
      otherResults: otherResults,
      totalProcessed: processedCount
    }
  });
}

// Main worker message handler
self.onmessage = async function(e) {
  const { type, data } = e.data;
  
  try {
    switch (type) {
      case 'INITIALIZE_SHARED_DATA_TRANSFERABLE': {
        const { frequencyBuffer, spectrumBuffer, spectrumLength, workerId: id } = data;
        
        console.log(`Worker ${id}: Received transferable data - frequencies: ${frequencyBuffer.length}, spectrum: ${spectrumBuffer.length}`);
        
        // Convert transferred buffers back to usable format
        sharedFrequencies = Array.from(frequencyBuffer);
        
        // Reconstruct reference spectrum from packed buffer
        sharedReferenceSpectrum = [];
        for (let i = 0; i < spectrumLength; i++) {
          const offset = i * 5;
          sharedReferenceSpectrum.push({
            freq: spectrumBuffer[offset],
            real: spectrumBuffer[offset + 1],
            imag: spectrumBuffer[offset + 2],
            mag: spectrumBuffer[offset + 3],
            phase: spectrumBuffer[offset + 4]
          });
        }
        
        workerId = id;
        
        console.log(`Worker ${id}: Successfully initialized with ${sharedFrequencies.length} frequencies and ${sharedReferenceSpectrum.length} spectrum points`);
        
        self.postMessage({
          type: 'SHARED_DATA_INITIALIZED',
          data: { workerId: id }
        });
        break;
      }

      case 'INITIALIZE_SHARED_DATA': {
        const { frequencies, referenceSpectrum, workerId: id } = data;
        
        // Fallback for non-transferable initialization
        sharedFrequencies = frequencies;
        sharedReferenceSpectrum = referenceSpectrum;
        workerId = id;
        
        self.postMessage({
          type: 'SHARED_DATA_INITIALIZED',
          data: { workerId: id }
        });
        break;
      }

      case 'COMPUTE_GRID_CHUNK_OPTIMIZED': {
        const { 
          chunkParams, 
          chunkIndex, 
          totalChunks, 
          resnormConfig = { method: 'mae' },
          taskId,
          maxComputationResults = 5000
        } = data;
        
        // Use cached shared data instead of transferring it each time
        if (!sharedFrequencies || !sharedReferenceSpectrum) {
          throw new Error('Shared data not initialized. Call INITIALIZE_SHARED_DATA first.');
        }
        
        await processChunkOptimized(
          chunkParams, 
          sharedFrequencies, 
          chunkIndex, 
          totalChunks, 
          sharedReferenceSpectrum, 
          resnormConfig,
          taskId,
          maxComputationResults
        );
        break;
      }

      case 'COMPUTE_GRID_CHUNK': {
        const { 
          chunkParams, 
          frequencyArray, 
          chunkIndex, 
          totalChunks,
          referenceSpectrum,
          resnormConfig = { method: 'mae' },
          maxComputationResults = 5000
        } = data;
        
        // Process chunk with ultra-efficient Top-N heap for massive datasets
        const MAX_RESULTS_DURING_COMPUTATION = maxComputationResults; // User-configurable limit
        
        // Efficient min-heap implementation for Top-N results (O(log n) operations)
        class MinHeap {
          constructor(maxSize) {
            this.heap = [];
            this.maxSize = maxSize;
          }
          
          parent(index) { return Math.floor((index - 1) / 2); }
          leftChild(index) { return 2 * index + 1; }
          rightChild(index) { return 2 * index + 2; }
          
          swap(i, j) {
            [this.heap[i], this.heap[j]] = [this.heap[j], this.heap[i]];
          }
          
          heapifyUp(index) {
            if (index > 0 && this.heap[this.parent(index)].resnorm > this.heap[index].resnorm) {
              this.swap(index, this.parent(index));
              this.heapifyUp(this.parent(index));
            }
          }
          
          heapifyDown(index) {
            let smallest = index;
            const left = this.leftChild(index);
            const right = this.rightChild(index);
            
            if (left < this.heap.length && this.heap[left].resnorm < this.heap[smallest].resnorm) {
              smallest = left;
            }
            if (right < this.heap.length && this.heap[right].resnorm < this.heap[smallest].resnorm) {
              smallest = right;
            }
            if (smallest !== index) {
              this.swap(index, smallest);
              this.heapifyDown(smallest);
            }
          }
          
          insert(item) {
            if (this.heap.length < this.maxSize) {
              // Heap not full, just add
              this.heap.push(item);
              this.heapifyUp(this.heap.length - 1);
            } else if (item.resnorm < this.heap[0].resnorm) {
              // Replace worst (root) with better item
              this.heap[0] = item;
              this.heapifyDown(0);
            }
            // Otherwise ignore (item is worse than our worst)
          }
          
          getAll() {
            return [...this.heap].sort((a, b) => a.resnorm - b.resnorm);
          }
          
          size() {
            return this.heap.length;
          }
        }
        
        // Initialize heap with lightweight data structure
        const topResults = new MinHeap(MAX_RESULTS_DURING_COMPUTATION);
        let processedCount = 0;
        
        // Ultra-aggressive adaptive batch sizing with enhanced memory pressure monitoring
        let batchSize = 25; // Start conservative
        let memoryThreshold = 100 * 1024 * 1024; // 100MB default memory limit per worker
        
        // Adaptive thresholds based on dataset size to prevent UI blocking at 80k points
        if (chunkParams.length > 80000) {
          batchSize = 3; // Micro-batches for massive chunks (unresponsive at 80k)
          memoryThreshold = 40 * 1024 * 1024; // Stricter memory limit for large datasets
        } else if (chunkParams.length > 50000) {
          batchSize = 5; // Very small batches for huge chunks
          memoryThreshold = 60 * 1024 * 1024; // Reduced memory threshold
        } else if (chunkParams.length > 25000) {
          batchSize = 8; // Smaller batches for very large chunks
        } else if (chunkParams.length > 10000) {
          batchSize = 15; // Small batches for large chunks
        } else if (chunkParams.length < 1000) {
          batchSize = 50; // Larger batches for small chunks
        }
        
        // Lightweight memory estimation (much lower with heap approach)
        const bytesPerResult = 150; // Much smaller without spectrum data during computation
        
        for (let i = 0; i < chunkParams.length; i += batchSize) {
          const batch = chunkParams.slice(i, i + batchSize);
          
          // Critical: Frequent async yields to prevent worker thread blocking
          if (i % (batchSize * 2) === 0) {  // Yield every 2 batches
            // Non-blocking yield to event loop - prevents 55% hang
            await new Promise(resolve => setTimeout(resolve, 0));
          }
          
          // Additional yields for massive datasets
          if (chunkParams.length > 100000 && i % batchSize === 0) {
            // Yield every single batch for very large datasets
            await new Promise(resolve => setTimeout(resolve, 1));
          }
          
          for (const params of batch) {
            try {
              // Validate parameters before computation
              if (!params || typeof params.Rsh !== 'number' || 
                  typeof params.Ra !== 'number' || typeof params.Ca !== 'number' ||
                  typeof params.Rb !== 'number' || typeof params.Cb !== 'number') {
                continue; // Skip invalid parameters
              }
              
              // Calculate spectrum
              const spectrum = calculateImpedanceSpectrum(params, frequencyArray);
              
              // Calculate resnorm
              const resnorm = calculateResnorm(referenceSpectrum, spectrum, resnormConfig);
              
              // Skip infinite or NaN resnorms
              if (!isFinite(resnorm) || isNaN(resnorm)) {
                continue;
              }
              
              // Ultra-efficient Top-N: store minimal data during computation
              // Only store parameters + resnorm during computation phase
              topResults.insert({
                parameters: params,
                resnorm: resnorm
                // NO spectrum data during computation - saves massive memory
              });
              
              processedCount++;
            } catch (error) {
              // Continue processing even if one parameter set fails
              console.warn('Error processing parameter set:', params, error.message);
              continue;
            }
          }
          
          // Report progress within chunk
          const progress = Math.min(100, Math.round((i + batchSize) / chunkParams.length * 100));
          self.postMessage({
            type: 'CHUNK_PROGRESS',
            data: {
              chunkIndex,
              totalChunks,
              chunkProgress: progress,
              processed: Math.min(i + batchSize, chunkParams.length),
              total: chunkParams.length
            }
          });
        }
        
        // Get final results from heap (already optimized)
        const finalResults = topResults.getAll();
        
        // Now add spectrum data ONLY to top results for final output
        const FINAL_TOP_RESULTS = maxComputationResults;
        const topResultsWithSpectra = [];
        const otherResults = [];
        
        // Re-compute spectra only for the top results
        for (let i = 0; i < finalResults.length; i++) {
          const result = finalResults[i];
          
          if (i < FINAL_TOP_RESULTS) {
            // Re-calculate spectrum for top results only  
            const spectrum = calculateImpedanceSpectrum(result.parameters, frequencyArray);
            topResultsWithSpectra.push({
              parameters: result.parameters,
              resnorm: result.resnorm,
              spectrum: spectrum
            });
          } else {
            // No spectrum for lower-priority results
            otherResults.push({
              parameters: result.parameters,
              resnorm: result.resnorm
            });
          }
        }
        
        self.postMessage({
          type: 'CHUNK_COMPLETE',
          data: {
            chunkIndex,
            topResults: topResultsWithSpectra,
            otherResults: otherResults,
            totalProcessed: processedCount
          }
        });
        
        break;
      }
      
      case 'GENERATE_GRID_POINTS': {
        const { gridSize, useSymmetricGrid, resnormConfig = { method: 'mae' }, groundTruthParams: inputGroundTruth } = data;
        
        // Safety check for grid size to prevent memory issues
        if (gridSize > 20) {
          throw new Error('Grid size too large: maximum allowed is 20 points per dimension');
        }
        
        const totalPoints = Math.pow(gridSize, 5);
        if (totalPoints > 10000000) { // 10 million points
          throw new Error(`Grid too large: ${totalPoints.toLocaleString()} points would exceed memory limits`);
        }
        
        // Use streaming approach to prevent memory overflow
        const gridPoints = [];
        const stream = streamGridPoints(gridSize, useSymmetricGrid, inputGroundTruth, resnormConfig);
        let lastProgressReport = 0;
        
        for (const { point, progress } of stream) {
          gridPoints.push(point);
          
          // Report progress periodically to avoid flooding the main thread
          if (progress.generated - lastProgressReport >= 1000 || progress.processed === progress.total) {
            self.postMessage({
              type: 'GENERATION_PROGRESS',
              data: {
                generated: progress.generated,
                total: progress.total,
                processed: progress.processed,
                skipped: progress.processed - progress.generated
              }
            });
            lastProgressReport = progress.generated;
          }
          
          // Yield control periodically to prevent blocking
          if (progress.generated % 10000 === 0) {
            // Small delay to allow other operations and garbage collection
            setTimeout(() => {}, 0);
          }
        }
        
        self.postMessage({
          type: 'GRID_POINTS_GENERATED',
          data: { gridPoints }
        });
        
        break;
      }
      
      case 'GENERATE_GRID_STREAM': {
        const { gridSize, useSymmetricGrid, resnormConfig = { method: 'mae' }, chunkSize = 1000, groundTruthParams: inputGroundTruth } = data;
        
        // Safety check for grid size
        if (gridSize > 20) {
          throw new Error('Grid size too large: maximum allowed is 20 points per dimension');
        }
        
        const totalPoints = Math.pow(gridSize, 5);
        if (totalPoints > 10000000) {
          throw new Error(`Grid too large: ${totalPoints.toLocaleString()} points would exceed memory limits`);
        }
        
        // Stream grid points in chunks to prevent memory overflow
        const stream = streamGridPoints(gridSize, useSymmetricGrid, inputGroundTruth, resnormConfig);
        let currentChunk = [];
        let totalGenerated = 0;
        
        for (const { point, progress } of stream) {
          currentChunk.push(point);
          totalGenerated = progress.generated;
          
          // Send chunk when it reaches the target size
          if (currentChunk.length >= chunkSize) {
            self.postMessage({
              type: 'GRID_CHUNK_READY',
              data: {
                chunk: currentChunk,
                progress: {
                  generated: totalGenerated,
                  total: progress.total,
                  processed: progress.processed
                }
              }
            });
            
                         // Clear chunk to free memory
             currentChunk = [];
             
             // Small delay to prevent blocking and allow garbage collection
             setTimeout(() => {}, 0);
          }
        }
        
        // Send any remaining points
        if (currentChunk.length > 0) {
          self.postMessage({
            type: 'GRID_CHUNK_READY',
            data: {
              chunk: currentChunk,
              progress: {
                generated: totalGenerated,
                total: totalPoints,
                processed: totalPoints,
                isComplete: true
              }
            }
          });
        }
        
        break;
      }
      
      default:
        self.postMessage({
          type: 'ERROR',
          data: { message: `Unknown message type: ${type}` }
        });
    }
  } catch (error) {
    self.postMessage({
      type: 'ERROR',
      data: { 
        message: error.message,
        stack: error.stack 
      }
    });
  }
}; 