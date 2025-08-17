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

// Calculate resnorm between two spectra using formula: (1/n) * sqrt(sum(wi * ri^2))
function calculateResnorm(referenceSpectrum, testSpectrum, useFrequencyWeighting = false) {
  if (!referenceSpectrum.length || !testSpectrum.length) return Infinity;
  
  let sumWeightedSquaredResiduals = 0;
  let sumWeights = 0;
  const n = Math.min(referenceSpectrum.length, testSpectrum.length);
  
  for (let i = 0; i < n; i++) {
    const refPoint = referenceSpectrum[i];
    const testPoint = testSpectrum[i];
    
    // Skip if frequencies don't match (within tolerance)
    if (Math.abs(refPoint.freq - testPoint.freq) / refPoint.freq > 0.001) {
      continue;
    }
    
    // Calculate residuals for real and imaginary components
    const realResidual = testPoint.real - refPoint.real;
    const imagResidual = testPoint.imag - refPoint.imag;
    
    // Calculate squared residual (ri^2)
    const squaredResidual = realResidual * realResidual + imagResidual * imagResidual;
    
    // Apply frequency weighting if enabled
    let weight = 1.0;
    if (useFrequencyWeighting && refPoint.freq > 0) {
      // Frequency weighting: w = f^(-0.5) emphasizes low frequencies
      weight = Math.pow(refPoint.freq, -0.5);
    }
    
    sumWeightedSquaredResiduals += weight * squaredResidual;
    sumWeights += weight;
  }
  
  // Calculate final resnorm with weighting
  if (sumWeights === 0) return Infinity;
  
  if (useFrequencyWeighting) {
    // Weighted resnorm: sqrt(sum(wi * ri^2) / sum(wi))
    return Math.sqrt(sumWeightedSquaredResiduals / sumWeights);
  } else {
    // Standard resnorm: (1/n) * sqrt(sum(ri^2))
    return (1 / n) * Math.sqrt(sumWeightedSquaredResiduals);
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
function* streamGridPoints(gridSize, useSymmetricGrid, inputGroundTruth, useFrequencyWeighting = false) {
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
async function processChunkOptimized(chunkParams, frequencyArray, chunkIndex, totalChunks, referenceSpectrum, useFrequencyWeighting, taskId) {
  const results = [];
  
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
    estimatedMemoryUsage = results.length * bytesPerResult;
    if (estimatedMemoryUsage > memoryThreshold) {
      self.postMessage({
        type: 'MEMORY_PRESSURE',
        data: {
          chunkIndex,
          taskId,
          estimatedMemory: Math.round(estimatedMemoryUsage / 1024 / 1024) + 'MB',
          resultsCount: results.length,
          message: 'High memory usage detected in optimized worker'
        }
      });
      
      // Aggressive memory management for large datasets
      if (results.length > 3000) {
        const partialResults = results.splice(0, 2000);
        self.postMessage({
          type: 'PARTIAL_RESULTS',
          data: {
            chunkIndex,
            taskId,
            partialResults: partialResults.slice(0, 500),
            totalPartialCount: partialResults.length
          }
        });
        partialResults.length = 0;
        estimatedMemoryUsage = results.length * bytesPerResult;
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
        const resnorm = calculateResnorm(referenceSpectrum, spectrum, useFrequencyWeighting);
        
        if (isFinite(resnorm)) {
          results.push({
            parameters: params,
            spectrum: results.length < 1000 ? spectrum : [], // Only keep spectrum for top results
            resnorm
          });
        }
      } catch (error) {
        console.warn('Parameter processing error:', error);
      }
    }
    
    // Throttled progress reporting for better performance
    const progress = (i + batchSize) / chunkParams.length;
    if (i % (batchSize * 10) === 0 || progress >= 1) {
      self.postMessage({
        type: 'CHUNK_PROGRESS',
        data: {
          chunkIndex,
          taskId,
          totalChunks,
          chunkProgress: progress,
          processed: Math.min(i + batchSize, chunkParams.length),
          total: chunkParams.length
        }
      });
    }
  }
  
  // Optimized result processing
  results.sort((a, b) => a.resnorm - b.resnorm);
  
  const topResults = results.slice(0, 800); // Slightly reduced to save memory
  const otherResults = results.slice(800).map(r => ({
    parameters: r.parameters,
    resnorm: r.resnorm
  }));
  
  self.postMessage({
    type: 'CHUNK_COMPLETE',
    data: {
      chunkIndex,
      taskId,
      topResults,
      otherResults,
      totalProcessed: results.length
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
          useFrequencyWeighting = false,
          taskId
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
          useFrequencyWeighting,
          taskId
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
          useFrequencyWeighting = false
        } = data;
        
        // Process chunk with aggressive memory management for large datasets
        const results = [];
        
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
        
        // Memory usage estimation
        let estimatedMemoryUsage = 0;
        const bytesPerResult = 800; // Estimated bytes per result object
        
        for (let i = 0; i < chunkParams.length; i += batchSize) {
          const batch = chunkParams.slice(i, i + batchSize);
          
          // Memory pressure check - implement back-pressure if needed
          estimatedMemoryUsage = results.length * bytesPerResult;
          if (estimatedMemoryUsage > memoryThreshold) {
            // Trigger intermediate cleanup and result streaming
            self.postMessage({
              type: 'MEMORY_PRESSURE',
              data: {
                chunkIndex,
                estimatedMemory: Math.round(estimatedMemoryUsage / 1024 / 1024) + 'MB',
                resultsCount: results.length,
                message: 'High memory usage detected, implementing backpressure'
              }
            });
            
            // Wait for memory pressure to subside (simple backpressure)
            await new Promise(resolve => setTimeout(resolve, 100));
            
            // Implement progressive data streaming if results are large
            if (results.length > 5000) {
              // Stream partial results and clear buffer
              const partialResults = results.splice(0, 3000); // Take first 3000 results
              self.postMessage({
                type: 'PARTIAL_RESULTS',
                data: {
                  chunkIndex,
                  partialResults: partialResults.slice(0, 1000), // Only send top 1000
                  totalPartialCount: partialResults.length
                }
              });
              // Clear memory references
              partialResults.length = 0;
              estimatedMemoryUsage = results.length * bytesPerResult;
            }
          }
          
          // Enhanced staggered processing with adaptive delays for UI responsiveness
          if (chunkParams.length > 80000 && i % (batchSize * 5) === 0) {
            // More frequent delays for datasets > 80k to prevent UI freezing
            await new Promise(resolve => setTimeout(resolve, 20));
          } else if (chunkParams.length > 50000 && i % (batchSize * 10) === 0) {
            // Medium frequency delays for large datasets
            await new Promise(resolve => setTimeout(resolve, 10));
          } else if (chunkParams.length > 25000 && i % (batchSize * 20) === 0) {
            // Small delay every 20 batches for very large chunks to prevent blocking
            await new Promise(resolve => setTimeout(resolve, 5));
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
              const resnorm = calculateResnorm(referenceSpectrum, spectrum, useFrequencyWeighting);
              
              // Skip infinite or NaN resnorms
              if (!isFinite(resnorm) || isNaN(resnorm)) {
                continue;
              }
              
              // Only store essential data
              results.push({
                parameters: params,
                resnorm: resnorm,
                spectrum: spectrum // Only needed for top results
              });
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
        
        // Sort results by resnorm and keep only top results to save memory
        results.sort((a, b) => a.resnorm - b.resnorm);
        
        // Only keep essential spectrum data for best results
        const topResults = results.slice(0, 1000); // Keep top 1000 per chunk
        const otherResults = results.slice(1000).map(r => ({
          parameters: r.parameters,
          resnorm: r.resnorm
          // Remove spectrum to save memory
        }));
        
        self.postMessage({
          type: 'CHUNK_COMPLETE',
          data: {
            chunkIndex,
            topResults,
            otherResults,
            totalProcessed: results.length
          }
        });
        
        break;
      }
      
      case 'GENERATE_GRID_POINTS': {
        const { gridSize, useSymmetricGrid, useFrequencyWeighting = false, groundTruthParams: inputGroundTruth } = data;
        
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
        const stream = streamGridPoints(gridSize, useSymmetricGrid, inputGroundTruth, useFrequencyWeighting);
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
        const { gridSize, useSymmetricGrid, useFrequencyWeighting = false, chunkSize = 1000, groundTruthParams: inputGroundTruth } = data;
        
        // Safety check for grid size
        if (gridSize > 20) {
          throw new Error('Grid size too large: maximum allowed is 20 points per dimension');
        }
        
        const totalPoints = Math.pow(gridSize, 5);
        if (totalPoints > 10000000) {
          throw new Error(`Grid too large: ${totalPoints.toLocaleString()} points would exceed memory limits`);
        }
        
        // Stream grid points in chunks to prevent memory overflow
        const stream = streamGridPoints(gridSize, useSymmetricGrid, inputGroundTruth, useFrequencyWeighting);
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