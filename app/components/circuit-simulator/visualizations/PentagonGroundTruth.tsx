/**
 * Pentagon Ground Truth Visualization Component
 * ==========================================
 *
 * Real-time pentagon pattern analysis for continuous simulation testing.
 * Generates and compares pentagon shapes based on circuit parameters.
 */

'use client';

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { CircuitParameters, DEFAULT_GRID_SIZE } from '../types/parameters';
import { ModelSnapshot } from '../types';
import {
  PlayIcon,
  PauseIcon,
  BeakerIcon
} from '@heroicons/react/24/outline';
import { calculateResnormWithConfig, calculate_impedance_spectrum, ResnormMethod } from '../utils/resnorm';

interface PentagonGroundTruthProps {
  models: ModelSnapshot[];
  currentParameters: CircuitParameters;
  isVisible?: boolean;
}

interface PentagonGroundTruth {
  vertices: { x: number; y: number }[];
  normalized: boolean;
  timestamp: number;
}

interface PentagonAnalysisConfig {
  gridSize: number; // Deprecated - kept for compatibility
  gridPointsPerParam: number; // Number of grid points per parameter (2-20)
  realTimeUpdates: boolean;
  maxIterations: number;
  exportResults: boolean;
}

interface PentagonCandidate {
  parameters: CircuitParameters;
  vertices: { x: number; y: number }[];
  euclideanDistance: number; // Pentagon geometry distance
  frequencyResnorm: number;   // Frequency spectrum resnorm (from 3D spider plot pipeline)
  modelSnapshot: ModelSnapshot; // Full model data compatible with 3D spider plot
  conditionTags: string[];
  parameterAnalysis: ParameterAnalysis; // Statistical analysis of parameter impact
  timestamp: number;
}

// Parameter sensitivity analysis interface
interface ParameterAnalysis {
  gridCoordinates: { Rsh: number; Ra: number; Rb: number; Ca: number; Cb: number }; // Grid position for each parameter
  parameterDeviations: { Rsh: number; Ra: number; Rb: number; Ca: number; Cb: number }; // Deviation from ground truth
  marginalImpact: { Rsh: number; Ra: number; Rb: number; Ca: number; Cb: number }; // Impact on resnorm per parameter
  sensitivityScore: number; // Overall sensitivity of this configuration
  dominantParameters: string[]; // Parameters with highest impact
}

interface PentagonAnalysisState {
  isRunning: boolean;
  progress: number;
  currentIteration: number;
  bestCandidates: PentagonCandidate[];
  currentCandidate: PentagonCandidate | null;
  groundTruth: PentagonGroundTruth | null;
  groundTruthParameters: CircuitParameters | null;
  groundTruthSpectrum: ReturnType<typeof calculate_impedance_spectrum> | null; // For resnorm calculation
  config: PentagonAnalysisConfig;
}

export const PentagonGroundTruth: React.FC<PentagonGroundTruthProps> = ({
  models,
  currentParameters,
  isVisible = true
}) => {
  const [pentagonAnalysis, setPentagonAnalysis] = useState<PentagonAnalysisState>({
    isRunning: false,
    progress: 0,
    currentIteration: 0,
    bestCandidates: [],
    currentCandidate: null,
    groundTruth: null,
    groundTruthParameters: null,
    groundTruthSpectrum: null,
    config: {
      gridSize: DEFAULT_GRID_SIZE, // Deprecated - kept for compatibility
      gridPointsPerParam: DEFAULT_GRID_SIZE, // Grid points per parameter using consistent default
      realTimeUpdates: true,
      maxIterations: Math.pow(DEFAULT_GRID_SIZE, 5), // DEFAULT_GRID_SIZE^5
      exportResults: true
    }
  });

  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Generate ground truth from best model in current grid
  const generateGroundTruthFromGrid = useCallback((): CircuitParameters => {
    if (models.length === 0) return currentParameters;

    // Find the model with the lowest resnorm (best fit)
    const bestModel = models.reduce((best, current) => {
      const currentResnorm = current.resnorm ?? Infinity;
      const bestResnorm = best.resnorm ?? Infinity;
      return currentResnorm < bestResnorm ? current : best;
    });

    return bestModel.parameters;
  }, [models, currentParameters]);

  // Generate ground truth pentagon from current parameters
  const generateGroundTruthPentagon = useCallback((params: CircuitParameters): PentagonGroundTruth => {
    // Normalize each parameter to [0,1] range using logarithmic scaling that matches parameter ranges
    const normalizedParams = {
      Rsh: (Math.log10(Math.max(params.Rsh, 100)) - Math.log10(100)) / (Math.log10(10000) - Math.log10(100)), // log(100Ω) to log(10kΩ)
      Ra: (Math.log10(Math.max(params.Ra, 500)) - Math.log10(500)) / (Math.log10(10000) - Math.log10(500)),   // log(500Ω) to log(10kΩ)
      Rb: (Math.log10(Math.max(params.Rb, 1000)) - Math.log10(1000)) / (Math.log10(15000) - Math.log10(1000)), // log(1kΩ) to log(15kΩ)
      Ca: (Math.log10(Math.max(params.Ca, 0.1e-6)) - Math.log10(0.1e-6)) / (Math.log10(10e-6) - Math.log10(0.1e-6)), // log(0.1µF) to log(10µF)
      Cb: (Math.log10(Math.max(params.Cb, 0.1e-6)) - Math.log10(0.1e-6)) / (Math.log10(10e-6) - Math.log10(0.1e-6))  // log(0.1µF) to log(10µF)
    };

    // Create pentagon vertices with equal angular spacing and parameter-based radii
    const vertices = [
      {
        x: 0.5 + Math.cos(0) * normalizedParams.Rsh * 0.4,
        y: 0.5 + Math.sin(0) * normalizedParams.Rsh * 0.4
      },
      {
        x: 0.5 + Math.cos(2 * Math.PI / 5) * normalizedParams.Ra * 0.4,
        y: 0.5 + Math.sin(2 * Math.PI / 5) * normalizedParams.Ra * 0.4
      },
      {
        x: 0.5 + Math.cos(4 * Math.PI / 5) * normalizedParams.Rb * 0.4,
        y: 0.5 + Math.sin(4 * Math.PI / 5) * normalizedParams.Rb * 0.4
      },
      {
        x: 0.5 + Math.cos(6 * Math.PI / 5) * normalizedParams.Ca * 0.4,
        y: 0.5 + Math.sin(6 * Math.PI / 5) * normalizedParams.Ca * 0.4
      },
      {
        x: 0.5 + Math.cos(8 * Math.PI / 5) * normalizedParams.Cb * 0.4,
        y: 0.5 + Math.sin(8 * Math.PI / 5) * normalizedParams.Cb * 0.4
      }
    ];

    return {
      vertices,
      normalized: true,
      timestamp: Date.now()
    };
  }, []);

  // Calculate Euclidean distance between two pentagons
  const calculatePentagonDistance = useCallback((pentagon1: { x: number; y: number }[], pentagon2: { x: number; y: number }[]): number => {
    let totalDistance = 0;
    for (let i = 0; i < 5; i++) {
      const dx = pentagon1[i].x - pentagon2[i].x;
      const dy = pentagon1[i].y - pentagon2[i].y;
      totalDistance += Math.sqrt(dx * dx + dy * dy);
    }
    return totalDistance / 5; // Average distance per vertex
  }, []);

  // Generate condition tags based on parameter values
  const generateConditionTags = useCallback((params: CircuitParameters): string[] => {
    const tags: string[] = [];

    if (params.Ra < 1000) tags.push('low Ra');
    else if (params.Ra > 5000) tags.push('high Ra');

    if (params.Rb < 2000) tags.push('low Rb');
    else if (params.Rb > 8000) tags.push('high Rb');

    if (params.Ca < 0.5e-6) tags.push('low Ca');
    else if (params.Ca > 2e-6) tags.push('high Ca');

    if (params.Cb < 1e-6) tags.push('low Cb');
    else if (params.Cb > 4e-6) tags.push('high Cb');

    if (params.Rsh < 500) tags.push('low Rsh');
    else if (params.Rsh > 2000) tags.push('high Rsh');

    return tags;
  }, []);

  // Calculate comprehensive parameter analysis with sensitivity insights
  const calculateParameterAnalysis = useCallback((
    candidateParams: CircuitParameters,
    groundTruthParams: CircuitParameters | null,
    iteration: number,
    gridPointsPerParam: number,
    candidateResnorm: number,
    groundTruthResnorm: number = 0
  ): ParameterAnalysis => {
    // Calculate grid coordinates (which grid point for each parameter)
    const paramKeys = ['Rsh', 'Ra', 'Rb', 'Ca', 'Cb'] as const;
    const gridCoordinates = {} as { [K in typeof paramKeys[number]]: number };

    // Calculate current position in the 5D grid
    const totalSteps = Math.pow(gridPointsPerParam, 5);
    const iterationClamped = Math.min(iteration, totalSteps - 1);

    paramKeys.forEach((key, i) => {
      const divisor = Math.pow(gridPointsPerParam, 4 - i);
      gridCoordinates[key] = Math.floor(iterationClamped / divisor) % gridPointsPerParam;
    });

    // Calculate parameter deviations from ground truth (logarithmic percentage)
    const parameterDeviations = {} as { [K in typeof paramKeys[number]]: number };
    if (groundTruthParams) {
      paramKeys.forEach(key => {
        const candidate = candidateParams[key];
        const groundTruth = groundTruthParams[key];
        // Calculate logarithmic deviation percentage
        parameterDeviations[key] = ((Math.log10(candidate) - Math.log10(groundTruth)) / Math.log10(groundTruth)) * 100;
      });
    } else {
      paramKeys.forEach(key => {
        parameterDeviations[key] = 0;
      });
    }

    // Calculate marginal impact (how much each parameter affects resnorm)
    const marginalImpact = {} as { [K in typeof paramKeys[number]]: number };
    const resnormDelta = candidateResnorm - groundTruthResnorm;

    paramKeys.forEach(key => {
      const deviation = Math.abs(parameterDeviations[key]);
      // Impact = resnorm change per percentage deviation (smaller = more sensitive)
      marginalImpact[key] = deviation > 0 ? Math.abs(resnormDelta) / deviation : 0;
    });

    // Calculate overall sensitivity score
    const totalDeviation = Object.values(parameterDeviations).reduce((sum, dev) => sum + Math.abs(dev), 0);
    const sensitivityScore = totalDeviation > 0 ? Math.abs(resnormDelta) / totalDeviation : 0;

    // Identify dominant parameters (top 2 with highest marginal impact)
    const dominantParameters = Object.entries(marginalImpact)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 2)
      .map(([key]) => key);

    // Comprehensive logging with pattern analysis
    console.group(`🔬 Pentagon Analysis - Iteration ${iteration + 1}`);
    console.log(`📊 Grid Coordinates:`, gridCoordinates);
    console.log(`📈 Parameter Values:`, {
      Rsh: `${candidateParams.Rsh.toFixed(0)}Ω (grid: ${gridCoordinates.Rsh})`,
      Ra: `${candidateParams.Ra.toFixed(0)}Ω (grid: ${gridCoordinates.Ra})`,
      Rb: `${candidateParams.Rb.toFixed(0)}Ω (grid: ${gridCoordinates.Rb})`,
      Ca: `${(candidateParams.Ca * 1e6).toFixed(2)}µF (grid: ${gridCoordinates.Ca})`,
      Cb: `${(candidateParams.Cb * 1e6).toFixed(2)}µF (grid: ${gridCoordinates.Cb})`
    });
    console.log(`📉 Parameter Deviations from Ground Truth:`,
      Object.fromEntries(paramKeys.map(key => [key, `${parameterDeviations[key].toFixed(1)}%`]))
    );
    console.log(`🎯 Frequency Resnorm:`, candidateResnorm.toFixed(6));
    console.log(`⚡ Marginal Impact (resnorm/% change):`,
      Object.fromEntries(paramKeys.map(key => [key, marginalImpact[key].toFixed(4)]))
    );
    console.log(`🏆 Dominant Parameters:`, dominantParameters);
    console.log(`📊 Sensitivity Score:`, sensitivityScore.toFixed(4));
    console.groupEnd();

    return {
      gridCoordinates,
      parameterDeviations,
      marginalImpact,
      sensitivityScore,
      dominantParameters
    };
  }, []);

  // Generate comprehensive statistical overview from all candidates
  const generateStatisticalOverview = useCallback((candidates: PentagonCandidate[]) => {
    if (candidates.length === 0) return;

    const paramKeys = ['Rsh', 'Ra', 'Rb', 'Ca', 'Cb'] as const;

    // Calculate parameter sensitivity statistics
    const parameterStats = {} as { [K in typeof paramKeys[number]]: {
      avgMarginalImpact: number;
      maxMarginalImpact: number;
      minMarginalImpact: number;
      stdDev: number;
      dominanceCount: number;
      sensitivityRank: number;
    }};

    paramKeys.forEach(key => {
      const impacts = candidates.map(c => c.parameterAnalysis.marginalImpact[key]).filter(i => i > 0);
      const dominanceCount = candidates.filter(c => c.parameterAnalysis.dominantParameters.includes(key)).length;

      if (impacts.length > 0) {
        const avg = impacts.reduce((sum, val) => sum + val, 0) / impacts.length;
        const max = Math.max(...impacts);
        const min = Math.min(...impacts);
        const variance = impacts.reduce((sum, val) => sum + Math.pow(val - avg, 2), 0) / impacts.length;
        const stdDev = Math.sqrt(variance);

        parameterStats[key] = {
          avgMarginalImpact: avg,
          maxMarginalImpact: max,
          minMarginalImpact: min,
          stdDev,
          dominanceCount,
          sensitivityRank: 0 // Will be calculated below
        };
      } else {
        parameterStats[key] = {
          avgMarginalImpact: 0,
          maxMarginalImpact: 0,
          minMarginalImpact: 0,
          stdDev: 0,
          dominanceCount: 0,
          sensitivityRank: 5
        };
      }
    });

    // Calculate sensitivity rankings (lower avgMarginalImpact = higher sensitivity = lower rank number)
    const sortedParams = [...paramKeys].sort((a, b) => parameterStats[a].avgMarginalImpact - parameterStats[b].avgMarginalImpact);
    sortedParams.forEach((param, index) => {
      parameterStats[param].sensitivityRank = index + 1;
    });

    // Generate insights
    const mostSensitiveParam = sortedParams[0];
    const leastSensitiveParam = sortedParams[sortedParams.length - 1];
    const mostDominantParam = paramKeys.reduce((a, b) =>
      parameterStats[a].dominanceCount > parameterStats[b].dominanceCount ? a : b
    );

    // Comprehensive statistical logging
    console.group(`📊 PENTAGON ANALYSIS - STATISTICAL OVERVIEW (${candidates.length} candidates)`);

    console.log(`🏆 PARAMETER SENSITIVITY RANKING (1=most sensitive):`);
    sortedParams.forEach((param, index) => {
      const stats = parameterStats[param];
      console.log(`  ${index + 1}. ${param}: avg impact ${stats.avgMarginalImpact.toFixed(4)}, dominance ${stats.dominanceCount}x, σ=${stats.stdDev.toFixed(4)}`);
    });

    console.log(`\n🎯 KEY INSIGHTS:`);
    console.log(`  • Most Sensitive Parameter: ${mostSensitiveParam} (smallest margins → biggest resnorm impact)`);
    console.log(`  • Most Dominant Parameter: ${mostDominantParam} (appeared in ${parameterStats[mostDominantParam].dominanceCount} top-2 rankings)`);
    console.log(`  • Least Sensitive Parameter: ${leastSensitiveParam} (largest margins needed for resnorm change)`);

    console.log(`\n📈 RESIDUAL THINKING IN GEOMETRIC SPACE:`);
    console.log(`  • Pentagon vertex radius directly correlates with frequency-domain impedance`);
    console.log(`  • Capacitance vertices (Ca, Cb) create phase-domain geometry patterns`);
    console.log(`  • Resistance vertices (Rsh, Ra, Rb) dominate magnitude-domain geometry`);
    console.log(`  • Geometric distance reflects frequency spectrum resnorm through parameter space mapping`);

    console.log(`\n🧮 PARAMETER STATISTICS DETAILED:`);
    paramKeys.forEach(key => {
      const stats = parameterStats[key];
      console.log(`  ${key}: μ=${stats.avgMarginalImpact.toFixed(6)}, σ=${stats.stdDev.toFixed(6)}, max=${stats.maxMarginalImpact.toFixed(6)}, min=${stats.minMarginalImpact.toFixed(6)}`);
    });

    console.groupEnd();

    return {
      parameterStats,
      mostSensitiveParam,
      leastSensitiveParam,
      mostDominantParam,
      totalCandidates: candidates.length
    };
  }, []);

  // Define parameter ranges for systematic scanning (logarithmic spacing)
  const parameterRanges = {
    Rsh: { min: Math.log10(100), max: Math.log10(10000) },      // log10(100Ω) to log10(10kΩ)
    Ra: { min: Math.log10(500), max: Math.log10(10000) },       // log10(500Ω) to log10(10kΩ)
    Rb: { min: Math.log10(1000), max: Math.log10(15000) },      // log10(1kΩ) to log10(15kΩ)
    Ca: { min: Math.log10(0.1e-6), max: Math.log10(10e-6) },    // log10(0.1µF) to log10(10µF)
    Cb: { min: Math.log10(0.1e-6), max: Math.log10(10e-6) }     // log10(0.1µF) to log10(10µF)
  };

  // Generate systematic parameter combinations with logarithmic spacing
  const generateSystematicParameters = useCallback((iteration: number, gridPointsPerParam: number): CircuitParameters => {
    // Convert linear iteration to 5D grid coordinates
    const totalSteps = Math.pow(gridPointsPerParam, 5);
    const normalizedIteration = iteration % totalSteps;

    // Extract 5D indices
    const indices: number[] = [];
    let remaining = normalizedIteration;
    for (let i = 0; i < 5; i++) {
      indices.push(remaining % gridPointsPerParam);
      remaining = Math.floor(remaining / gridPointsPerParam);
    }

    // Map indices to logarithmically spaced parameter values
    const paramKeys = ['Rsh', 'Ra', 'Rb', 'Ca', 'Cb'] as const;
    const parameters: Partial<CircuitParameters> = {};

    paramKeys.forEach((key, i) => {
      const range = parameterRanges[key];
      const stepValue = indices[i] / (gridPointsPerParam - 1 || 1); // 0 to 1

      // Logarithmic interpolation: log_min + stepValue * (log_max - log_min)
      const logValue = range.min + stepValue * (range.max - range.min);

      // Convert back to linear scale: 10^logValue
      parameters[key] = Math.pow(10, logValue);
    });

    return {
      ...parameters,
      frequency_range: currentParameters.frequency_range
    } as CircuitParameters;
  }, [currentParameters.frequency_range]);

  // Generate ModelSnapshot with full frequency spectrum and resnorm (integrates with 3D spider plot pipeline)
  const generateModelSnapshot = useCallback((
    parameters: CircuitParameters,
    groundTruthSpectrum: ReturnType<typeof calculate_impedance_spectrum> | null
  ): { snapshot: ModelSnapshot; resnorm: number } => {
    // Calculate complete frequency spectrum using existing pipeline
    // ✅ VERIFIED: Uses entire frequency range (20 logarithmic points from 0.1Hz to 100kHz)
    const impedanceSpectrum = calculate_impedance_spectrum(parameters);

    // Calculate resnorm using existing pipeline (same as 3D spider plot)
    // ✅ VERIFIED: Compares entire frequency spectrum using MAE across all 20 frequency points
    let resnorm = 0;
    if (groundTruthSpectrum) {
      resnorm = calculateResnormWithConfig(
        impedanceSpectrum,         // Full frequency spectrum for candidate
        groundTruthSpectrum,       // Full frequency spectrum for ground truth
        undefined,                 // No frequency weighting (uses all points equally)
        undefined,                 // No component weighting (resistive + capacitive)
        { method: ResnormMethod.MAE } // Mean Absolute Error across all frequency points
      );
    }

    // Create ModelSnapshot compatible with 3D spider plot visualization
    const snapshot: ModelSnapshot = {
      id: `pentagon-candidate-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      name: `Pentagon Candidate`,
      timestamp: Date.now(),
      parameters,
      data: impedanceSpectrum, // Full frequency spectrum for spider plot integration
      resnorm,
      color: '#FF6B6B', // Pentagon analysis color
      isVisible: true,
      opacity: 1.0
    };

    return { snapshot, resnorm };
  }, []);

  // Start pentagon ground truth analysis
  const startPentagonAnalysis = useCallback(() => {
    if (!pentagonAnalysis.groundTruth) {
      const groundTruth = generateGroundTruthPentagon(currentParameters);
      setPentagonAnalysis(prev => ({
        ...prev,
        groundTruth,
        groundTruthParameters: currentParameters,
        isRunning: true,
        progress: 0,
        currentIteration: 0,
        bestCandidates: []
      }));
    } else {
      setPentagonAnalysis(prev => ({
        ...prev,
        isRunning: true
      }));
    }

    // Start the analysis loop
    const analysisLoop = () => {
      setPentagonAnalysis(prev => {
        if (!prev.isRunning || prev.currentIteration >= prev.config.maxIterations) {
          // Generate final statistical overview when analysis completes
          if (prev.bestCandidates.length > 0) {
            setTimeout(() => generateStatisticalOverview(prev.bestCandidates), 100);
          }
          return { ...prev, isRunning: false };
        }

        // Generate systematic candidate parameters across full ranges
        const candidateParams = generateSystematicParameters(prev.currentIteration, prev.config.gridPointsPerParam);

        // Generate candidate pentagon
        const candidatePentagon = generateGroundTruthPentagon(candidateParams);

        // Calculate distance to ground truth
        const distance = prev.groundTruth ?
          calculatePentagonDistance(candidatePentagon.vertices, prev.groundTruth.vertices) :
          Infinity;

        // Generate condition tags
        const conditionTags = generateConditionTags(candidateParams);

        // Generate unified ModelSnapshot with frequency spectrum and resnorm (integrated with 3D spider plot pipeline)
        const { snapshot, resnorm: frequencyResnorm } = generateModelSnapshot(candidateParams, prev.groundTruthSpectrum);

        // Calculate comprehensive parameter analysis with sensitivity insights
        const groundTruthResnorm = prev.groundTruthSpectrum ?
          (prev.bestCandidates.length > 0 ? prev.bestCandidates[0].frequencyResnorm : 0) : 0;
        const parameterAnalysis = calculateParameterAnalysis(
          candidateParams,
          prev.groundTruthParameters,
          prev.currentIteration,
          prev.config.gridPointsPerParam,
          frequencyResnorm,
          groundTruthResnorm
        );

        const candidate: PentagonCandidate = {
          parameters: candidateParams,
          vertices: candidatePentagon.vertices,
          euclideanDistance: distance,
          frequencyResnorm, // Frequency spectrum based resnorm using existing pipeline
          modelSnapshot: snapshot, // Full ModelSnapshot for 3D spider plot integration
          conditionTags,
          parameterAnalysis, // Statistical analysis of parameter impact
          timestamp: Date.now()
        };

        // Update best candidates
        const updatedBestCandidates = [...prev.bestCandidates, candidate]
          .sort((a, b) => a.euclideanDistance - b.euclideanDistance)
          .slice(0, 10); // Keep top 10

        return {
          ...prev,
          currentIteration: prev.currentIteration + 1,
          progress: (prev.currentIteration / prev.config.maxIterations) * 100,
          currentCandidate: candidate,
          bestCandidates: updatedBestCandidates
        };
      });
    };

    // Run analysis loop
    const intervalId = setInterval(analysisLoop, 50); // 20 FPS for real-time updates

    // Cleanup on unmount
    return () => clearInterval(intervalId);
  }, [currentParameters, generateGroundTruthPentagon, calculatePentagonDistance, generateConditionTags, pentagonAnalysis.groundTruth]);

  // Render pentagon visualization
  const renderPentagonVisualization = useCallback(() => {
    if (!canvasRef.current || !pentagonAnalysis.groundTruth) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const canvasWidth = canvas.width;
    const canvasHeight = canvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    // Calculate pentagon area (center 60% of canvas)
    const padding = 0.2;
    const pentagonWidth = canvasWidth * (1 - 2 * padding);
    const pentagonHeight = canvasHeight * (1 - 2 * padding);
    const offsetX = canvasWidth * padding;
    const offsetY = canvasHeight * padding;

    // Draw ground truth pentagon (blue)
    ctx.strokeStyle = '#3B82F6';
    ctx.lineWidth = 3;
    ctx.beginPath();
    const gtVertices = pentagonAnalysis.groundTruth.vertices;
    ctx.moveTo(gtVertices[0].x * pentagonWidth + offsetX, gtVertices[0].y * pentagonHeight + offsetY);
    for (let i = 1; i < 5; i++) {
      ctx.lineTo(gtVertices[i].x * pentagonWidth + offsetX, gtVertices[i].y * pentagonHeight + offsetY);
    }
    ctx.closePath();
    ctx.stroke();

    // Draw current candidate (red)
    if (pentagonAnalysis.currentCandidate) {
      ctx.strokeStyle = '#EF4444';
      ctx.lineWidth = 2;
      ctx.beginPath();
      const candVertices = pentagonAnalysis.currentCandidate.vertices;
      ctx.moveTo(candVertices[0].x * pentagonWidth + offsetX, candVertices[0].y * pentagonHeight + offsetY);
      for (let i = 1; i < 5; i++) {
        ctx.lineTo(candVertices[i].x * pentagonWidth + offsetX, candVertices[i].y * pentagonHeight + offsetY);
      }
      ctx.closePath();
      ctx.stroke();
    }

    // Draw best candidate (green)
    if (pentagonAnalysis.bestCandidates.length > 0) {
      const bestCandidate = pentagonAnalysis.bestCandidates[0];
      ctx.strokeStyle = '#10B981';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      const bestVertices = bestCandidate.vertices;
      ctx.moveTo(bestVertices[0].x * pentagonWidth + offsetX, bestVertices[0].y * pentagonHeight + offsetY);
      for (let i = 1; i < 5; i++) {
        ctx.lineTo(bestVertices[i].x * pentagonWidth + offsetX, bestVertices[i].y * pentagonHeight + offsetY);
      }
      ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Add parameter labels for ground truth vertices
    if (pentagonAnalysis.groundTruth) {
      const labels = ['Rsh', 'Ra', 'Rb', 'Ca', 'Cb'];
      ctx.fillStyle = '#FFFFFF';
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';

      pentagonAnalysis.groundTruth.vertices.forEach((vertex, i) => {
        const x = vertex.x * pentagonWidth + offsetX;
        const y = vertex.y * pentagonHeight + offsetY;

        // Draw label slightly outside the vertex
        const labelOffset = 15;
        const angle = (i * 2 * Math.PI) / 5;
        const labelX = x + Math.cos(angle) * labelOffset;
        const labelY = y + Math.sin(angle) * labelOffset;

        ctx.fillText(labels[i], labelX, labelY);
      });
    }
  }, [pentagonAnalysis]);

  // Update pentagon visualization
  useEffect(() => {
    if (isVisible) {
      renderPentagonVisualization();
    }
  }, [isVisible, pentagonAnalysis, renderPentagonVisualization]);

  return (
    <div className="w-full h-full bg-black text-white p-4">
      <div className="h-full flex flex-col">
        {/* Header Controls */}
        <div className="space-y-3 mb-4">
          {/* Title and Action Buttons */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-neutral-300">
              <BeakerIcon className="w-4 h-4" />
              <span className="font-medium">Pentagon Ground Truth</span>
              <span className="text-xs text-neutral-500">({models.length.toLocaleString()} models)</span>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => {
                  const bestParams = generateGroundTruthFromGrid();
                  const groundTruth = generateGroundTruthPentagon(bestParams);
                  const groundTruthSpectrum = calculate_impedance_spectrum(bestParams);
                  setPentagonAnalysis(prev => ({
                    ...prev,
                    groundTruth,
                    groundTruthParameters: bestParams,
                    groundTruthSpectrum
                  }));
                }}
                className="px-2 py-1 bg-emerald-600 hover:bg-emerald-700 text-white text-xs rounded transition-colors"
              >
                Set Best Model
              </button>
              <button
                onClick={() => {
                  const groundTruth = generateGroundTruthPentagon(currentParameters);
                  const groundTruthSpectrum = calculate_impedance_spectrum(currentParameters);
                  setPentagonAnalysis(prev => ({
                    ...prev,
                    groundTruth,
                    groundTruthParameters: currentParameters,
                    groundTruthSpectrum
                  }));
                }}
                className="px-2 py-1 bg-blue-600 hover:bg-blue-700 text-white text-xs rounded transition-colors"
              >
                Set Ground Truth
              </button>
              <button
                onClick={() => {
                  setPentagonAnalysis(prev => ({
                    ...prev,
                    groundTruth: null,
                    groundTruthParameters: null,
                    groundTruthSpectrum: null, // Clear frequency spectrum for unified computation
                    isRunning: false,
                    progress: 0,
                    currentIteration: 0,
                    bestCandidates: [],
                    currentCandidate: null
                  }));
                }}
                className="px-2 py-1 bg-neutral-600 hover:bg-neutral-700 text-white text-xs rounded transition-colors"
              >
                Reset
              </button>
              {pentagonAnalysis.isRunning ? (
                <button
                  onClick={() => setPentagonAnalysis(prev => ({ ...prev, isRunning: false }))}
                  className="px-2 py-1 bg-red-600 hover:bg-red-700 text-white text-xs rounded transition-colors flex items-center gap-1"
                >
                  <PauseIcon className="w-3 h-3" />
                  Pause
                </button>
              ) : (
                <button
                  onClick={startPentagonAnalysis}
                  disabled={!pentagonAnalysis.groundTruth}
                  className="px-2 py-1 bg-orange-600 hover:bg-orange-700 disabled:bg-neutral-600 text-white text-xs rounded transition-colors flex items-center gap-1"
                >
                  <PlayIcon className="w-3 h-3" />
                  Analyze
                </button>
              )}
            </div>
          </div>

          {/* Analysis Configuration */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 p-3 bg-neutral-800/50 rounded border border-neutral-700">
            {/* Grid Points Control */}
            <div className="space-y-1">
              <label className="text-xs text-neutral-400">Grid Points Per Parameter</label>
              <div className="flex items-center gap-2">
                <input
                  type="range"
                  min="2"
                  max="12"
                  step="1"
                  value={pentagonAnalysis.config.gridPointsPerParam}
                  onChange={(e) => {
                    const gridPoints = parseInt(e.target.value);
                    setPentagonAnalysis(prev => ({
                      ...prev,
                      config: {
                        ...prev.config,
                        gridPointsPerParam: gridPoints,
                        maxIterations: Math.pow(gridPoints, 5)
                      }
                    }));
                  }}
                  className="flex-1 h-1"
                />
                <span className="text-xs text-neutral-300 w-8">
                  {pentagonAnalysis.config.gridPointsPerParam}
                </span>
              </div>
              <div className="text-xs text-neutral-500">
                {pentagonAnalysis.config.gridPointsPerParam}⁵ = {Math.pow(pentagonAnalysis.config.gridPointsPerParam, 5).toLocaleString()} points
              </div>
            </div>

            {/* Real-time Updates Toggle */}
            <div className="space-y-1">
              <label className="text-xs text-neutral-400">Real-time Updates</label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={pentagonAnalysis.config.realTimeUpdates}
                  onChange={(e) => setPentagonAnalysis(prev => ({
                    ...prev,
                    config: { ...prev.config, realTimeUpdates: e.target.checked }
                  }))}
                  className="w-3 h-3"
                />
                <span className="text-xs text-neutral-300">
                  {pentagonAnalysis.config.realTimeUpdates ? 'Enabled' : 'Disabled'}
                </span>
              </label>
            </div>
          </div>

          {/* Ground Truth Values Display */}
          {pentagonAnalysis.groundTruth && pentagonAnalysis.groundTruthParameters && (
            <div className="p-3 bg-neutral-800/30 rounded border border-neutral-700">
              <div className="text-xs text-neutral-400 mb-2">Ground Truth Parameters</div>
              <div className="grid grid-cols-5 gap-3 text-xs">
                <div className="text-center">
                  <div className="text-neutral-500">Rsh</div>
                  <div className="text-white font-mono">{pentagonAnalysis.groundTruthParameters.Rsh.toFixed(0)}Ω</div>
                </div>
                <div className="text-center">
                  <div className="text-neutral-500">Ra</div>
                  <div className="text-white font-mono">{pentagonAnalysis.groundTruthParameters.Ra.toFixed(0)}Ω</div>
                </div>
                <div className="text-center">
                  <div className="text-neutral-500">Rb</div>
                  <div className="text-white font-mono">{pentagonAnalysis.groundTruthParameters.Rb.toFixed(0)}Ω</div>
                </div>
                <div className="text-center">
                  <div className="text-neutral-500">Ca</div>
                  <div className="text-white font-mono">{(pentagonAnalysis.groundTruthParameters.Ca * 1e6).toFixed(1)}µF</div>
                </div>
                <div className="text-center">
                  <div className="text-neutral-500">Cb</div>
                  <div className="text-white font-mono">{(pentagonAnalysis.groundTruthParameters.Cb * 1e6).toFixed(1)}µF</div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Main Content */}
        {pentagonAnalysis.groundTruth ? (
          <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-4 min-h-0">

            {/* Left: Pentagon Visualization */}
            <div className="lg:col-span-2 bg-neutral-800/30 rounded border border-neutral-700">
              <div className="p-3 border-b border-neutral-700">
                <div className="text-sm text-neutral-300">Pentagon Visualization</div>
              </div>
              <div className="p-3">
                <canvas
                  ref={canvasRef}
                  width={500}
                  height={350}
                  className="w-full bg-neutral-900 rounded border border-neutral-600"
                />
                <div className="text-xs text-neutral-500 mt-2 flex justify-center gap-4">
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-0.5 bg-blue-400"></span>Ground Truth
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-0.5 bg-red-400"></span>Current
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-0.5 bg-green-400 border-dashed"></span>Best Match
                  </span>
                </div>
              </div>
            </div>

            {/* Right: Compact Statistics */}
            <div className="space-y-3">
              {/* Progress */}
              <div className="bg-neutral-800/30 rounded border border-neutral-700 p-3">
                <div className="text-xs text-neutral-400 mb-2">Progress</div>
                <div className="w-full bg-neutral-700 rounded-full h-2">
                  <div
                    className="bg-orange-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${pentagonAnalysis.progress}%` }}
                  />
                </div>
                <div className="text-xs text-neutral-500 mt-1 flex justify-between">
                  <span>{pentagonAnalysis.currentIteration.toLocaleString()}</span>
                  <span>{pentagonAnalysis.progress.toFixed(1)}%</span>
                </div>
              </div>

              {/* Current Analysis */}
              {pentagonAnalysis.currentCandidate && (
                <div className="bg-neutral-800/30 rounded border border-neutral-700 p-3">
                  <div className="text-xs text-neutral-400 mb-2">Current</div>
                  <div className="text-sm text-neutral-200">
                    {pentagonAnalysis.currentCandidate.euclideanDistance.toFixed(4)}
                  </div>
                  <div className="text-xs text-neutral-500">
                    {pentagonAnalysis.currentCandidate.conditionTags.slice(0, 2).join(', ') || 'No tags'}
                  </div>
                </div>
              )}

              {/* Best Match */}
              {pentagonAnalysis.bestCandidates.length > 0 && (
                <div className="bg-neutral-800/30 rounded border border-neutral-700 p-3">
                  <div className="text-xs text-neutral-400 mb-2">Best Match</div>
                  <div className="text-sm text-emerald-400">
                    {pentagonAnalysis.bestCandidates[0].euclideanDistance.toFixed(4)}
                  </div>
                  <div className="text-xs text-neutral-500">
                    {pentagonAnalysis.bestCandidates[0].conditionTags.slice(0, 2).join(', ') || 'No tags'}
                  </div>
                </div>
              )}

              {/* Parameter Sensitivity Insights */}
              {pentagonAnalysis.bestCandidates.length > 0 && (
                <div className="bg-neutral-800/30 rounded border border-neutral-700 p-3">
                  <div className="text-xs text-neutral-400 mb-2">Parameter Sensitivity</div>
                  <div className="space-y-1">
                    <div className="text-xs text-orange-400">
                      Dominant: {pentagonAnalysis.bestCandidates[0].parameterAnalysis.dominantParameters.join(', ')}
                    </div>
                    <div className="text-xs text-blue-400">
                      Frequency Resnorm: {pentagonAnalysis.bestCandidates[0].frequencyResnorm.toFixed(6)}
                    </div>
                    <div className="text-xs text-neutral-500">
                      Sensitivity: {pentagonAnalysis.bestCandidates[0].parameterAnalysis.sensitivityScore.toFixed(4)}
                    </div>
                  </div>
                </div>
              )}

              {/* Top Rankings */}
              {pentagonAnalysis.bestCandidates.length > 1 && (
                <div className="bg-neutral-800/30 rounded border border-neutral-700 p-3">
                  <div className="text-xs text-neutral-400 mb-2">Top 5</div>
                  <div className="space-y-1 max-h-28 overflow-y-auto">
                    {pentagonAnalysis.bestCandidates.slice(0, 5).map((candidate, index) => (
                      <div key={index} className="text-xs text-neutral-500 flex justify-between">
                        <span>#{index + 1}</span>
                        <span className="font-mono">{candidate.euclideanDistance.toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center text-neutral-500 max-w-md">
              <BeakerIcon className="w-8 h-8 mx-auto mb-3 text-neutral-600" />
              <div className="text-base mb-2 text-neutral-400">Pentagon Analysis Ready</div>
              <div className="text-sm mb-4 text-neutral-500">
                Set ground truth using your best model or current circuit parameters to begin systematic pattern analysis
              </div>
              <div className="text-xs text-neutral-600 leading-relaxed">
                Systematically scans parameter space to find pentagon shapes that match your ground truth target
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PentagonGroundTruth;