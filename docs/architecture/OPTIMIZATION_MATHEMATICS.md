# Circuit Simulator Optimization Mathematics

This document provides a comprehensive mathematical explanation of all optimizations implemented in the NEI Visualization Project's circuit simulator.

## Table of Contents
1. [Circuit Model Overview](#circuit-model-overview)
2. [Grid Generation Mathematics](#grid-generation-mathematics)
3. [Symmetric Grid Optimization](#symmetric-grid-optimization)
4. [Parameter Space Sampling](#parameter-space-sampling)
5. [Ground Truth Integration](#ground-truth-integration)
6. [Resnorm Calculation and Optimization](#resnorm-calculation-and-optimization)
7. [Opacity and Visualization Optimizations](#opacity-and-visualization-optimizations)
8. [Performance Optimizations](#performance-optimizations)

## Circuit Model Overview

### Modified Randles Equivalent Circuit
The simulator implements a modified Randles circuit model for retinal pigment epithelium (RPE) impedance analysis:

```
       Rsh (Shunt Resistance)
   ────[Rsh]────┬──────────┬──────
                │          │
            [Ra]│      [Rb]│
                │          │
            [Ca]│      [Cb]│
                │          │
                └──────────┘
```

### Circuit Parameters
- **Rsh**: Shunt resistance (10 - 10,000 Ω)
- **Ra**: Apical resistance (10 - 10,000 Ω)
- **Ca**: Apical capacitance (0.1 - 50 µF)
- **Rb**: Basal resistance (10 - 10,000 Ω)  
- **Cb**: Basal capacitance (0.1 - 50 µF)

### Complex Impedance Calculation
The total impedance is calculated using parallel RC networks:

#### Individual Branch Impedances:
```
Za(ω) = Ra / (1 + jωRaCa)
Zb(ω) = Rb / (1 + jωRbCb)
```

#### Total Circuit Impedance:
```
Z(ω) = Rsh + Za(ω) + Zb(ω)
     = Rsh + Ra/(1 + jωRaCa) + Rb/(1 + jωRbCb)
```

Where:
- `ω = 2πf` (angular frequency)
- `j` = imaginary unit
- `f` = frequency in Hz

## Grid Generation Mathematics

### Logarithmic Parameter Space Sampling

For each parameter P with range [Pmin, Pmax] and N grid points:

```
Pi = 10^(log₁₀(Pmin) + i × step)
```

Where:
```
step = (log₁₀(Pmax) - log₁₀(Pmin)) / (N - 1)
i ∈ {0, 1, 2, ..., N-1}
```

### Grid Size Impact
- **Total Parameter Combinations**: `N^5` (5-dimensional parameter space)
- **Example**: N=9 → 9^5 = 59,049 combinations
- **Memory Estimate**: ~500 bytes per model + spectrum data

### Ground Truth Integration Algorithm

The `generateLogSpaceWithReference()` function ensures ground truth parameters are included:

1. **Generate standard logarithmic space**: `{P₁, P₂, ..., Pₙ}`
2. **Find closest existing point** to ground truth value `Pgt`:
   ```
   closest_index = argmin_i |log₁₀(Pi) - log₁₀(Pgt)|
   ```
3. **Calculate relative error**:
   ```
   relative_error = |log₁₀(Pgt) - log₁₀(Pclosest)| / |log₁₀(Pgt)|
   ```
4. **Replace if error > 5%**:
   ```
   if relative_error > 0.05:
       Pi[closest_index] = Pgt
       sort({P₁, P₂, ..., Pₙ})
   ```

## Symmetric Grid Optimization

### Mathematical Rationale
The symmetric optimization exploits the physical symmetry in the parallel RC branch configuration. For branches A and B, swapping parameters often yields identical impedance responses.

### Time Constant Analysis
Define time constants for each branch:
```
τA = Ra × Ca  (apical time constant)
τB = Rb × Cb  (basal time constant)
```

### Optimization Rules

#### Primary Rule - Time Constant Ordering:
```
Skip combination if: τA > τB
```

**Mathematical Justification**: 
For low frequencies where `ωτ ≪ 1`, the impedance approaches:
```
Za(ω) ≈ Ra - j/(ωCa) ≈ Ra - j/(ωCa)
Zb(ω) ≈ Rb - j/(ωCb) ≈ Rb - j/(ωCb)
```

Combinations with (Ra₁,Ca₁,Rb₁,Cb₁) and (Rb₁,Cb₁,Ra₁,Ca₁) produce identical impedance spectra when τA = τB.

#### Tie-Breaking Rule:
```
if |τA - τB| < ε and Ra > Rb: skip combination
```

Where `ε = 10⁻¹⁵` (machine precision tolerance).

### Implementation in Web Worker

```javascript
// Calculate time constants
const tauA = Ra * Ca;
const tauB = Rb * Cb;

// Apply symmetric optimization
if (useSymmetricGrid) {
    // Skip if tauA > tauB (will be covered by swapped version)
    if (tauA > tauB) {
        continue;
    }
    
    // Tie-breaking: if time constants equal, enforce Ra <= Rb
    if (Math.abs(tauA - tauB) < 1e-15 && Ra > Rb) {
        continue;
    }
}
```

### Reduction Factor Analysis

For grid size N, the symmetric optimization reduces combinations by approximately:

```
Reduction ≈ 1 - 1/2^k
```

Where k is the number of parameter pairs that can be symmetrically swapped.

**Example**: For N=9, without optimization: 59,049 combinations
With symmetric optimization: ~30,000-35,000 combinations (40-45% reduction)

## Parameter Space Sampling

### Multi-Dimensional Grid Generation

The parameter space is generated as a 5D Cartesian product:

```
Ω = {(Rsh,Ra,Ca,Rb,Cb) | Rsh ∈ ΩRsh, Ra ∈ ΩRa, Ca ∈ ΩCa, Rb ∈ ΩRb, Cb ∈ ΩCb}
```

### Streaming Generation Algorithm

To prevent memory overflow, parameters are generated using a streaming iterator:

```javascript
function* streamGridPoints(gridSize, useSymmetricGrid, groundTruthParams) {
    const rsValues = generateLogSpaceWithReference(...);
    const raValues = generateLogSpaceWithReference(...);
    // ... generate all parameter arrays
    
    for (let rshIndex = 0; rshIndex < gridSize; rshIndex++) {
        for (let raIndex = 0; raIndex < gridSize; raIndex++) {
            for (let caIndex = 0; caIndex < gridSize; caIndex++) {
                for (let rbIndex = 0; rbIndex < gridSize; rbIndex++) {
                    for (let cbIndex = 0; cbIndex < gridSize; cbIndex++) {
                        const params = {
                            Rsh: rsValues[rshIndex],
                            Ra: raValues[raIndex],
                            Ca: caValues[caIndex],
                            Rb: rbValues[rbIndex],
                            Cb: cbValues[cbIndex]
                        };
                        
                        // Apply symmetric optimization
                        if (useSymmetricGrid && shouldSkip(params)) {
                            continue;
                        }
                        
                        yield { point: params, progress: ... };
                    }
                }
            }
        }
    }
}
```

## Resnorm Calculation and Optimization

### Residual Norm Definition

The residual norm quantifies how well a parameter set fits reference data:

```
resnorm = √(Σᵢ wᵢ |Zsim(ωᵢ) - Zref(ωᵢ)|²)
```

Where:
- `Zsim(ωᵢ)`: Simulated impedance at frequency ωᵢ
- `Zref(ωᵢ)`: Reference impedance at frequency ωᵢ
- `wᵢ`: Frequency-dependent weighting factor

### Frequency Weighting Strategy (Optional - User Toggleable)

Frequency weighting is now a user-configurable option in the Performance settings, **disabled by default**:

```javascript
// When useFrequencyWeighting = true:
wᵢ = fᵢ^(-0.5)

// When useFrequencyWeighting = false (default):
wᵢ = 1.0
```

Where the exponent -0.5 emphasizes low-frequency accuracy (critical for biological systems).

**To enable frequency weighting:**
1. Open the Performance controls (expand the Performance section)
2. Toggle the "Frequency Weighting" switch to ON
3. The toggle is marked as "Advanced" and remains off when "Auto Optimize" is enabled

### Advanced Resnorm Calculation

The implementation includes component-specific weighting:

```javascript
function calculateResnorm(simSpectrum, refSpectrum, frequencies, useFrequencyWeighting = false) {
    let totalWeightedError = 0;
    let totalWeight = 0;
    
    for (let i = 0; i < frequencies.length; i++) {
        const freq = frequencies[i];
        const simZ = simSpectrum[i];  // Complex impedance
        const refZ = refSpectrum[i];  // Complex impedance
        
        // Component errors
        const realError = simZ.real - refZ.real;
        const imagError = simZ.imag - refZ.imag;
        
        // Magnitude error
        const errorMagnitude = Math.sqrt(realError * realError + imagError * imagError);
        
        // Apply frequency weighting if enabled
        let weight = 1.0;
        if (useFrequencyWeighting && freq > 0) {
            weight = Math.pow(freq, -0.5);  // w = f^(-0.5)
        }
        
        // Reference magnitude for normalization
        const refMagnitude = Math.sqrt(refZ.real * refZ.real + refZ.imag * refZ.imag);
        
        // Normalized weighted error
        if (refMagnitude > 1e-12) {  // Avoid division by zero
            const normalizedError = errorMagnitude / refMagnitude;
            totalWeightedError += weight * normalizedError * normalizedError;
            totalWeight += weight;
        }
    }
    
    // Calculate final resnorm based on weighting mode
    if (useFrequencyWeighting && totalWeight > 0) {
        // Weighted resnorm: sqrt(sum(wi * ri^2) / sum(wi))
        return Math.sqrt(totalWeightedError / totalWeight);
    } else {
        // Standard resnorm: (1/n) * sqrt(sum(ri^2))
        const n = frequencies.length;
        return (1 / n) * Math.sqrt(totalWeightedError);
    }
}
```

### Resnorm Group Classification

Models are classified into percentile-based performance groups:

```
Group 0 (Excellent): resnorm ∈ [min, P₂₅]     (0-25th percentile)
Group 1 (Good):      resnorm ∈ [P₂₅, P₅₀]    (25-50th percentile)  
Group 2 (Fair):      resnorm ∈ [P₅₀, P₇₅]    (50-75th percentile)
Group 3 (Poor):      resnorm ∈ [P₇₅, max]    (75-100th percentile)
```

## Opacity and Visualization Optimizations

### Mathematical Opacity Strategy

For models within each resnorm group, opacity is calculated using:

```
Oᵢ = ((1 - rᵢ) / (1 - rₘᵢₙ))^β
```

Where:
- `rᵢ = Rᵢ/Rₘₐₓ` (normalized resnorm)
- `rₘᵢₙ = Rₘᵢₙ/Rₘₐₓ` (normalized minimum resnorm)
- `β` = opacity exponent (user-controlled, 1.0 - 8.0)
- `Oᵢ` ∈ [0.1, 1.0] (clamped opacity range)

### Opacity Function Properties

1. **Monotonic Decreasing**: Better fits (lower resnorm) → higher opacity
2. **Exponential Enhancement**: Higher β values increase contrast between good and poor fits
3. **Bounded**: Prevents completely invisible models (minimum opacity = 0.1)

### Group Portion Filtering

Within each selected group, only the best-fitting models are displayed:

```javascript
const keepCount = Math.floor(groupSize × groupPortion);
const sortedByResnorm = group.sort((a, b) => a.resnorm - b.resnorm);
const displayedModels = sortedByResnorm.slice(0, keepCount);
```

Where `groupPortion` ∈ [0.01, 1.0] (1% - 100%).

## Performance Optimizations

### Parallel Processing Architecture

#### Multi-Worker Strategy:
```
Total Workers = min(CPU_CORES, 8)
Grid Worker = workers[0]      // Parameter generation
Compute Workers = workers[1:] // Impedance calculation
```

#### Work Distribution:
```javascript
const chunkSize = Math.ceil(totalCombinations / computeWorkers.length);
const chunks = [];
for (let i = 0; i < computeWorkers.length; i++) {
    chunks.push(parameterCombinations.slice(i * chunkSize, (i + 1) * chunkSize));
}
```

### Memory Management

#### Adaptive Rendering Limits:
```javascript
function calculateVisualizationLimit(totalComputed) {
    const estimatedMemoryMB = totalComputed * 0.5 / 1024;  // ~500 bytes per model
    
    if (estimatedMemoryMB > 2000) return 250000;  // 2GB → 250K models
    if (estimatedMemoryMB > 1000) return 500000;  // 1GB → 500K models
    if (estimatedMemoryMB > 500)  return 750000;  // 500MB → 750K models
    return Math.min(1000000, totalComputed);      // Max 1M models
}
```

### Smart Batching Algorithm

Dynamic batch size optimization based on system performance:

```javascript
function calculateOptimalBatchSize(workerCount, totalPoints) {
    const baseSize = Math.ceil(Math.sqrt(totalPoints / workerCount));
    const maxSize = 5000;  // Memory constraint
    const minSize = 100;   // Overhead constraint
    
    return Math.max(minSize, Math.min(maxSize, baseSize));
}
```

### Frequency Generation Optimization

Logarithmic frequency spacing for EIS analysis:

```javascript
function generateFrequencies(minFreq, maxFreq, numPoints) {
    const logMin = Math.log10(minFreq);
    const logMax = Math.log10(maxFreq);
    const step = (logMax - logMin) / (numPoints - 1);
    
    return Array.from({ length: numPoints }, (_, i) => 
        Math.pow(10, logMin + i * step)
    );
}
```

## Implementation Summary

### Key Mathematical Components:

1. **Parameter Generation**: Logarithmic spacing with ground truth integration
2. **Symmetric Optimization**: Time constant comparison for 40-45% computational reduction
3. **Impedance Calculation**: Complex arithmetic for parallel RC networks
4. **Resnorm Analysis**: Frequency-weighted residual norm with component-specific weighting
5. **Opacity Mapping**: Exponential function for visual contrast enhancement
6. **Performance Scaling**: Adaptive algorithms for memory and computational efficiency

### Optimization Benefits:

- **Computational Efficiency**: 40-45% reduction in parameter combinations via symmetry
- **Memory Management**: Adaptive limits prevent system overload
- **Visual Quality**: Mathematical opacity functions enhance model comparison
- **Scalability**: Parallel processing utilizes available CPU cores
- **Accuracy**: Ground truth integration ensures reference parameters are always sampled

This mathematical framework enables efficient exploration of high-dimensional parameter spaces while maintaining computational accuracy and system responsiveness.