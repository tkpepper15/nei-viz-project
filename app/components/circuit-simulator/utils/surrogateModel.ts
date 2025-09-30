/**
 * Surrogate Model for Circuit Parameter-to-Spectrum Mapping
 *
 * Implements a focused Gaussian Process Regression approach for modeling
 * the relationship between circuit parameters and compressed spectral coefficients.
 * Provides uncertainty quantification for adaptive sampling and directional insights.
 */

import { CircuitParameters } from '../types/parameters';

// Core interfaces for surrogate modeling
export interface TrainingData {
  /** Input parameters */
  parameters: CircuitParameters[];
  /** Compressed spectral coefficients */
  spectralCoefficients: number[][];
  /** Resnorm values for each parameter set */
  resnorms: number[];
}

export interface GPRPrediction {
  /** Mean prediction */
  mean: number[];
  /** Prediction variance (uncertainty) */
  variance: number[];
  /** 95% confidence bounds */
  confidenceBounds: { lower: number[]; upper: number[] };
}

export interface SurrogateModel {
  /** Predict spectral coefficients from parameters */
  predict: (params: CircuitParameters) => GPRPrediction;
  /** Get prediction uncertainty at a point */
  getUncertainty: (params: CircuitParameters) => number;
  /** Get gradient for sensitivity analysis */
  getGradient: (params: CircuitParameters) => number[][];
  /** Update model with new data */
  update: (newData: TrainingData) => void;
  /** Get model confidence across parameter space */
  getConfidenceMap: (parameterGrid: CircuitParameters[]) => number[];
}

/**
 * Simple kernel functions for Gaussian Process
 */
export class KernelFunctions {
  /**
   * RBF (Gaussian) kernel with automatic relevance determination
   */
  static rbf(x1: number[], x2: number[], lengthScales: number[], variance: number = 1): number {
    let squaredDistance = 0;
    for (let i = 0; i < x1.length; i++) {
      const diff = (x1[i] - x2[i]) / lengthScales[i];
      squaredDistance += diff * diff;
    }
    return variance * Math.exp(-0.5 * squaredDistance);
  }

  /**
   * Matern 3/2 kernel (good for twice-differentiable functions)
   */
  static matern32(x1: number[], x2: number[], lengthScales: number[], variance: number = 1): number {
    let distance = 0;
    for (let i = 0; i < x1.length; i++) {
      const diff = Math.abs(x1[i] - x2[i]) / lengthScales[i];
      distance += diff * diff;
    }
    distance = Math.sqrt(distance);
    const sqrt3 = Math.sqrt(3);
    return variance * (1 + sqrt3 * distance) * Math.exp(-sqrt3 * distance);
  }
}

/**
 * Lightweight Gaussian Process Regression implementation
 * Focused on the core functionality needed for directional analysis
 */
export class SimplifiedGPR implements SurrogateModel {
  private trainingInputs: number[][] = [];
  private trainingOutputs: number[][] = [];
  private kernelMatrix: number[][] = [];
  private kernelInverse: number[][] = [];
  private lengthScales: number[] = [1, 1, 1, 1, 1]; // [Rsh, Ra, Ca, Rb, Cb]
  private noiseVariance: number = 1e-4;
  private signalVariance: number = 1.0;
  private numCoefficients: number = 8;

  constructor(numSpectralCoefficients: number = 8) {
    this.numCoefficients = numSpectralCoefficients;
  }

  /**
   * Convert circuit parameters to normalized input vector
   */
  private paramToVector(params: CircuitParameters): number[] {
    // Normalize parameters to [0, 1] range for better kernel behavior
    return [
      (params.Rsh - 10) / (10000 - 10),
      (params.Ra - 10) / (10000 - 10),
      (params.Ca - 0.1e-6) / (50e-6 - 0.1e-6),
      (params.Rb - 10) / (10000 - 10),
      (params.Cb - 0.1e-6) / (50e-6 - 0.1e-6)
    ];
  }

  /**
   * Compute kernel matrix
   */
  private computeKernelMatrix(inputs: number[][]): number[][] {
    const n = inputs.length;
    const K = Array(n).fill(null).map(() => Array(n).fill(0));

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        K[i][j] = KernelFunctions.matern32(inputs[i], inputs[j], this.lengthScales, this.signalVariance);
        if (i === j) {
          K[i][j] += this.noiseVariance; // Add noise to diagonal
        }
      }
    }

    return K;
  }

  /**
   * Simple matrix inversion using Cholesky decomposition
   */
  private invertMatrix(matrix: number[][]): number[][] {
    const n = matrix.length;

    // For demo purposes, use a simplified pseudo-inverse approach
    // In production, use a proper linear algebra library
    const inverse = Array(n).fill(null).map(() => Array(n).fill(0));

    // Initialize as identity and solve approximately
    for (let i = 0; i < n; i++) {
      inverse[i][i] = 1 / (matrix[i][i] + 1e-6);
    }

    return inverse;
  }

  /**
   * Train the GP model
   */
  train(data: TrainingData): void {
    this.trainingInputs = data.parameters.map(p => this.paramToVector(p));
    this.trainingOutputs = data.spectralCoefficients;

    // Compute kernel matrix and its inverse
    this.kernelMatrix = this.computeKernelMatrix(this.trainingInputs);
    this.kernelInverse = this.invertMatrix(this.kernelMatrix);

    // Simple hyperparameter optimization (fit length scales)
    this.optimizeHyperparameters();
  }

  /**
   * Simplified hyperparameter optimization
   */
  private optimizeHyperparameters(): void {
    // For each parameter, estimate characteristic length scale
    for (let param = 0; param < 5; param++) {
      const values = this.trainingInputs.map(x => x[param]);
      const range = Math.max(...values) - Math.min(...values);
      this.lengthScales[param] = Math.max(0.1, range / 3); // Reasonable default
    }
  }

  /**
   * Make prediction for new parameters
   */
  predict(params: CircuitParameters): GPRPrediction {
    const testInput = this.paramToVector(params);
    const n = this.trainingInputs.length;

    if (n === 0) {
      // No training data - return prior
      const meanPred = Array(this.numCoefficients).fill(0);
      const varPred = Array(this.numCoefficients).fill(this.signalVariance);
      return {
        mean: meanPred,
        variance: varPred,
        confidenceBounds: {
          lower: meanPred.map((m, i) => m - 1.96 * Math.sqrt(varPred[i])),
          upper: meanPred.map((m, i) => m + 1.96 * Math.sqrt(varPred[i]))
        }
      };
    }

    // Compute kernel vector between test point and training points
    const kStar = this.trainingInputs.map(trainInput =>
      KernelFunctions.matern32(testInput, trainInput, this.lengthScales, this.signalVariance)
    );

    // Predict mean: k* K^(-1) y
    const meanPredictions: number[] = Array(this.numCoefficients).fill(0);
    for (let coeff = 0; coeff < this.numCoefficients; coeff++) {
      let prediction = 0;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          prediction += kStar[i] * this.kernelInverse[i][j] *
                       (this.trainingOutputs[j][coeff] || 0);
        }
      }
      meanPredictions[coeff] = prediction;
    }

    // Predict variance: k** - k* K^(-1) k*
    let variance = this.signalVariance;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        variance -= kStar[i] * this.kernelInverse[i][j] * kStar[j];
      }
    }
    variance = Math.max(1e-6, variance); // Ensure positive

    const variancePredictions = Array(this.numCoefficients).fill(variance);

    return {
      mean: meanPredictions,
      variance: variancePredictions,
      confidenceBounds: {
        lower: meanPredictions.map((m, i) => m - 1.96 * Math.sqrt(variancePredictions[i])),
        upper: meanPredictions.map((m, i) => m + 1.96 * Math.sqrt(variancePredictions[i]))
      }
    };
  }

  /**
   * Get prediction uncertainty (scalar measure)
   */
  getUncertainty(params: CircuitParameters): number {
    const prediction = this.predict(params);
    return prediction.variance.reduce((sum, v) => sum + v, 0) / prediction.variance.length;
  }

  /**
   * Compute gradient via finite differences
   */
  getGradient(params: CircuitParameters): number[][] {
    const eps = 1e-4;
    const basePred = this.predict(params);
    const gradient: number[][] = Array(5).fill(null).map(() => Array(this.numCoefficients).fill(0));

    const paramNames: (keyof CircuitParameters)[] = ['Rsh', 'Ra', 'Ca', 'Rb', 'Cb'];

    paramNames.forEach((paramName, paramIdx) => {
      const perturbedParams = { ...params };
      const originalValue = perturbedParams[paramName] as number;

      // Forward difference
      (perturbedParams[paramName] as number) = originalValue * (1 + eps);
      const forwardPred = this.predict(perturbedParams);

      // Compute gradient
      for (let coeffIdx = 0; coeffIdx < this.numCoefficients; coeffIdx++) {
        gradient[paramIdx][coeffIdx] =
          (forwardPred.mean[coeffIdx] - basePred.mean[coeffIdx]) / (originalValue * eps);
      }
    });

    return gradient;
  }

  /**
   * Update model with new training data
   */
  update(newData: TrainingData): void {
    // Simple approach: retrain with combined data
    this.trainingInputs.push(...newData.parameters.map(p => this.paramToVector(p)));
    this.trainingOutputs.push(...newData.spectralCoefficients);

    // Recompute kernel matrices
    this.kernelMatrix = this.computeKernelMatrix(this.trainingInputs);
    this.kernelInverse = this.invertMatrix(this.kernelMatrix);
  }

  /**
   * Get confidence map across parameter grid
   */
  getConfidenceMap(parameterGrid: CircuitParameters[]): number[] {
    return parameterGrid.map(params => this.getUncertainty(params));
  }
}

/**
 * Active sampling strategy using acquisition functions
 */
export class ActiveSampler {
  private gpr: SimplifiedGPR;

  constructor(gpr: SimplifiedGPR) {
    this.gpr = gpr;
  }

  /**
   * Upper Confidence Bound acquisition function
   */
  upperConfidenceBound(params: CircuitParameters, beta: number = 2.0): number {
    const prediction = this.gpr.predict(params);
    const meanQuality = -prediction.mean.reduce((sum, val) => sum + Math.abs(val), 0); // Negative for minimization
    const uncertainty = Math.sqrt(prediction.variance.reduce((sum, val) => sum + val, 0));

    return meanQuality + beta * uncertainty;
  }

  /**
   * Expected Improvement acquisition function
   */
  expectedImprovement(params: CircuitParameters, bestValue: number = 0, xi: number = 0.01): number {
    const prediction = this.gpr.predict(params);
    const mean = prediction.mean.reduce((sum, val) => sum + val, 0);
    const std = Math.sqrt(prediction.variance.reduce((sum, val) => sum + val, 0));

    if (std === 0) return 0;

    const improvement = bestValue - mean - xi;
    const z = improvement / std;

    // Simplified normal CDF and PDF
    const phi = 0.5 * (1 + this.erf(z / Math.sqrt(2)));
    const phiPrime = Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);

    return improvement * phi + std * phiPrime;
  }

  /**
   * Error function approximation
   */
  private erf(x: number): number {
    // Abramowitz and Stegun approximation
    const a1 =  0.254829592;
    const a2 = -0.284496736;
    const a3 =  1.421413741;
    const a4 = -1.453152027;
    const a5 =  1.061405429;
    const p  =  0.3275911;

    const sign = x >= 0 ? 1 : -1;
    x = Math.abs(x);

    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

    return sign * y;
  }

  /**
   * Select next sampling points based on acquisition function
   */
  selectNextSamples(
    candidateParams: CircuitParameters[],
    numSamples: number = 10,
    acquisitionType: 'ucb' | 'ei' = 'ucb'
  ): CircuitParameters[] {
    const scores = candidateParams.map(params => {
      return acquisitionType === 'ucb'
        ? this.upperConfidenceBound(params)
        : this.expectedImprovement(params);
    });

    // Sort by acquisition score and take top samples
    const indexedScores = scores.map((score, index) => ({ score, index }));
    indexedScores.sort((a, b) => b.score - a.score);

    return indexedScores
      .slice(0, numSamples)
      .map(item => candidateParams[item.index]);
  }
}

/**
 * Factory function to create and train a surrogate model
 */
export function createSurrogateModel(
  trainingData: TrainingData,
  numSpectralCoefficients: number = 8
): { model: SimplifiedGPR; sampler: ActiveSampler } {
  const gpr = new SimplifiedGPR(numSpectralCoefficients);
  gpr.train(trainingData);

  const sampler = new ActiveSampler(gpr);

  return { model: gpr, sampler };
}