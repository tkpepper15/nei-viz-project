// Web Worker for efficient grid computation
// This runs on a separate thread to avoid blocking the UI

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
  const { Rs, Ra, Ca, Rb, Cb } = params;
  
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
    
    // Calculate parallel combination: Z_total = (Rs * (Za + Zb)) / (Rs + Za + Zb)
    // Numerator: Rs * (Za + Zb)
    const numerator = complex.multiply(
      { real: Rs, imag: 0 },
      Zab
    );
    
    // Denominator: Rs + Za + Zb
    const denominator = complex.add(
      { real: Rs, imag: 0 },
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

// Calculate resnorm between two spectra using simplified formula: (1/n) * sqrt(sum(ri^2))
function calculateResnorm(referenceSpectrum, testSpectrum) {
  if (!referenceSpectrum.length || !testSpectrum.length) return Infinity;
  
  let sumSquaredResiduals = 0;
  const n = Math.min(referenceSpectrum.length, testSpectrum.length);
  
  for (let i = 0; i < n; i++) {
    const refPoint = referenceSpectrum[i];
    const testPoint = testSpectrum[i];
    
    // Skip if frequencies don't match (within tolerance)
    if (Math.abs(refPoint.freq - testPoint.freq) / refPoint.freq > 0.001) {
      continue;
    }
    
    // Calculate residuals for real and imaginary components (no normalization)
    const realResidual = testPoint.real - refPoint.real;
    const imagResidual = testPoint.imag - refPoint.imag;
    
    // Calculate squared residual (ri^2)
    const squaredResidual = realResidual * realResidual + imagResidual * imagResidual;
    
    sumSquaredResiduals += squaredResidual;
  }
  
  // Calculate final resnorm: (1/n) * sqrt(sum(ri^2))
  return (1 / n) * Math.sqrt(sumSquaredResiduals);
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

// Main worker message handler
self.onmessage = function(e) {
  const { type, data } = e.data;
  
  try {
    switch (type) {
      case 'COMPUTE_GRID_CHUNK': {
        const { 
          chunkParams, 
          frequencyArray, 
          chunkIndex, 
          totalChunks,
          referenceSpectrum 
        } = data;
        
        // Process chunk efficiently with memory management
        const results = [];
        
        // Adaptive batch size based on chunk size to prevent memory issues
        let batchSize = 50;
        if (chunkParams.length > 10000) {
          batchSize = 25; // Smaller batches for large chunks
        } else if (chunkParams.length < 1000) {
          batchSize = 100; // Larger batches for small chunks
        }
        
        for (let i = 0; i < chunkParams.length; i += batchSize) {
          const batch = chunkParams.slice(i, i + batchSize);
          
          for (const params of batch) {
            try {
              // Validate parameters before computation
              if (!params || typeof params.Rs !== 'number' || 
                  typeof params.Ra !== 'number' || typeof params.Ca !== 'number' ||
                  typeof params.Rb !== 'number' || typeof params.Cb !== 'number') {
                continue; // Skip invalid parameters
              }
              
              // Calculate spectrum
              const spectrum = calculateImpedanceSpectrum(params, frequencyArray);
              
              // Calculate resnorm directly (avoid creating model objects)
              const resnorm = calculateResnorm(referenceSpectrum, spectrum);
              
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
        const { gridSize } = data;
        
        // Safety check for grid size to prevent memory issues
        if (gridSize > 25) {
          throw new Error('Grid size too large: maximum allowed is 25 points per dimension');
        }
        
        const totalPoints = Math.pow(gridSize, 5);
        if (totalPoints > 10000000) { // 10 million points
          throw new Error(`Grid too large: ${totalPoints.toLocaleString()} points would exceed memory limits`);
        }
        
        // Generate parameter ranges - use log space for Rs for consistency
        const rsValues = generateLogSpace(10, 10000, gridSize);
        const raValues = generateLogSpace(10, 10000, gridSize);
        const rbValues = generateLogSpace(10, 10000, gridSize);
        const caValues = generateLogSpace(0.1e-6, 50e-6, gridSize);
        const cbValues = generateLogSpace(0.1e-6, 50e-6, gridSize);
        
        // Generate all combinations using iterative approach to avoid stack overflow
        const gridPoints = [];
        
        // Use iterative approach instead of nested loops
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
          
          // Generate point from indices
          gridPoints.push({
            Rs: rsValues[rsIndex],
            Ra: raValues[raIndex],
            Ca: caValues[caIndex],
            Rb: rbValues[rbIndex],
            Cb: cbValues[cbIndex]
          });
          
          // Report progress for large grids
          if ((pointIndex + 1) % 10000 === 0 || pointIndex === totalPoints - 1) {
            self.postMessage({
              type: 'GENERATION_PROGRESS',
              data: {
                generated: pointIndex + 1,
                total: totalPoints
              }
            });
          }
        }
        
        self.postMessage({
          type: 'GRID_POINTS_GENERATED',
          data: { gridPoints }
        });
        
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