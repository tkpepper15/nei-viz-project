/**
 * Unified Analysis Tab Component
 * =============================
 *
 * Consolidates correlation analysis, directional sensitivity, and pentagon ground truth system
 * for comprehensive pattern analysis and continuous simulation testing.
 */

'use client';

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { CircuitParameters } from '../../types/parameters';
import { ModelSnapshot, ResnormGroup } from '../../types';
import {
  computeDirectionalSensitivity,
  DirectionalSensitivity,
  createFrequencyArray
} from '../../utils/directionalAnalysis';
import DirectionalSensitivityPlot from '../../visualizations/DirectionalSensitivityPlot';
import CorrelationHeatmap from '../../visualizations/CorrelationHeatmap';
import {
  ChartBarIcon,
  ArrowPathIcon,
  PlayIcon,
  PauseIcon,
  BeakerIcon
} from '@heroicons/react/24/outline';

interface AnalysisTabProps {
  gridResults: ModelSnapshot[];
  topConfigurations: ResnormGroup[];
  currentParameters: CircuitParameters;
  selectedConfigIndex: number;
  onConfigurationSelect: (index: number) => void;
  isVisible: boolean;
  highlightedModelId?: string | null;
  gridSize?: number;
}

type AnalysisMode = 'correlations' | 'directional' | 'pentagon-gt';

interface PentagonGroundTruth {
  vertices: { x: number; y: number }[];
  normalized: boolean;
  timestamp: number;
}

interface PentagonAnalysisConfig {
  gridSize: number; // Will use 9^5 = 59,049 combinations
  perturbationRadius: number; // For parameter variation around ground truth
  realTimeUpdates: boolean;
  maxIterations: number;
  exportResults: boolean;
}

interface PentagonCandidate {
  parameters: CircuitParameters;
  vertices: { x: number; y: number }[];
  euclideanDistance: number;
  conditionTags: string[];
  timestamp: number;
}

interface PentagonAnalysisState {
  isRunning: boolean;
  progress: number;
  currentIteration: number;
  bestCandidates: PentagonCandidate[];
  currentCandidate: PentagonCandidate | null;
  groundTruth: PentagonGroundTruth | null;
  config: PentagonAnalysisConfig;
}

export const AnalysisTab: React.FC<AnalysisTabProps> = ({
  gridResults,
  currentParameters,
  isVisible
}) => {
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>('correlations');
  const [directionalResults, setDirectionalResults] = useState<DirectionalSensitivity | null>(null);
  const [pentagonAnalysis, setPentagonAnalysis] = useState<PentagonAnalysisState>({
    isRunning: false,
    progress: 0,
    currentIteration: 0,
    bestCandidates: [],
    currentCandidate: null,
    groundTruth: null,
    config: {
      gridSize: 9,
      perturbationRadius: 0.1,
      realTimeUpdates: true,
      maxIterations: 59049, // 9^5
      exportResults: true
    }
  });

  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Generate ground truth pentagon from current parameters
  const generateGroundTruthPentagon = useCallback((params: CircuitParameters): PentagonGroundTruth => {
    // Convert circuit parameters to normalized pentagon vertices
    // This creates a reference shape based on the parameter values
    const vertices = [
      { x: Math.cos(0) * (params.Rsh / 10000), y: Math.sin(0) * (params.Rsh / 10000) },
      { x: Math.cos(2 * Math.PI / 5) * (params.Ra / 10000), y: Math.sin(2 * Math.PI / 5) * (params.Ra / 10000) },
      { x: Math.cos(4 * Math.PI / 5) * (params.Rb / 10000), y: Math.sin(4 * Math.PI / 5) * (params.Rb / 10000) },
      { x: Math.cos(6 * Math.PI / 5) * (params.Ca * 1e6), y: Math.sin(6 * Math.PI / 5) * (params.Ca * 1e6) },
      { x: Math.cos(8 * Math.PI / 5) * (params.Cb * 1e6), y: Math.sin(8 * Math.PI / 5) * (params.Cb * 1e6) }
    ];

    // Normalize vertices to [0, 1] range
    const minX = Math.min(...vertices.map(v => v.x));
    const maxX = Math.max(...vertices.map(v => v.x));
    const minY = Math.min(...vertices.map(v => v.y));
    const maxY = Math.max(...vertices.map(v => v.y));

    const normalized = vertices.map(v => ({
      x: (v.x - minX) / (maxX - minX),
      y: (v.y - minY) / (maxY - minY)
    }));

    return {
      vertices: normalized,
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

  // Start pentagon ground truth analysis
  const startPentagonAnalysis = useCallback(() => {
    if (!pentagonAnalysis.groundTruth) {
      const groundTruth = generateGroundTruthPentagon(currentParameters);
      setPentagonAnalysis(prev => ({
        ...prev,
        groundTruth,
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
          return { ...prev, isRunning: false };
        }

        // Generate candidate parameters
        const baseParams = currentParameters;
        const candidateParams: CircuitParameters = {
          Rsh: baseParams.Rsh + (Math.random() - 0.5) * 2 * prev.config.perturbationRadius * baseParams.Rsh,
          Ra: baseParams.Ra + (Math.random() - 0.5) * 2 * prev.config.perturbationRadius * baseParams.Ra,
          Rb: baseParams.Rb + (Math.random() - 0.5) * 2 * prev.config.perturbationRadius * baseParams.Rb,
          Ca: baseParams.Ca + (Math.random() - 0.5) * 2 * prev.config.perturbationRadius * baseParams.Ca,
          Cb: baseParams.Cb + (Math.random() - 0.5) * 2 * prev.config.perturbationRadius * baseParams.Cb,
          frequency_range: baseParams.frequency_range
        };

        // Generate candidate pentagon
        const candidatePentagon = generateGroundTruthPentagon(candidateParams);

        // Calculate distance to ground truth
        const distance = prev.groundTruth ?
          calculatePentagonDistance(candidatePentagon.vertices, prev.groundTruth.vertices) :
          Infinity;

        // Generate condition tags
        const conditionTags = generateConditionTags(candidateParams);

        const candidate: PentagonCandidate = {
          parameters: candidateParams,
          vertices: candidatePentagon.vertices,
          euclideanDistance: distance,
          conditionTags,
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

  // Compute directional sensitivity
  const computeDirectional = useCallback(async () => {
    if (gridResults.length < 10) return;

    try {
      const frequencies = createFrequencyArray(0.1, 100000, 100);
      const sensitivity = computeDirectionalSensitivity(
        currentParameters,
        frequencies
      );
      setDirectionalResults(sensitivity);
    } catch (error) {
      console.error('Directional analysis failed:', error);
    }
  }, [currentParameters, gridResults]);

  // Render pentagon visualization
  const renderPentagonVisualization = useCallback(() => {
    if (!canvasRef.current || !pentagonAnalysis.groundTruth) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw ground truth pentagon (blue)
    ctx.strokeStyle = '#3B82F6';
    ctx.lineWidth = 3;
    ctx.beginPath();
    const gtVertices = pentagonAnalysis.groundTruth.vertices;
    ctx.moveTo(gtVertices[0].x * width, gtVertices[0].y * height);
    for (let i = 1; i < 5; i++) {
      ctx.lineTo(gtVertices[i].x * width, gtVertices[i].y * height);
    }
    ctx.closePath();
    ctx.stroke();

    // Draw current candidate (red)
    if (pentagonAnalysis.currentCandidate) {
      ctx.strokeStyle = '#EF4444';
      ctx.lineWidth = 2;
      ctx.beginPath();
      const candVertices = pentagonAnalysis.currentCandidate.vertices;
      ctx.moveTo(candVertices[0].x * width, candVertices[0].y * height);
      for (let i = 1; i < 5; i++) {
        ctx.lineTo(candVertices[i].x * width, candVertices[i].y * height);
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
      ctx.moveTo(bestVertices[0].x * width, bestVertices[0].y * height);
      for (let i = 1; i < 5; i++) {
        ctx.lineTo(bestVertices[i].x * width, bestVertices[i].y * height);
      }
      ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [pentagonAnalysis]);

  // Update pentagon visualization
  useEffect(() => {
    if (analysisMode === 'pentagon-gt' && isVisible) {
      renderPentagonVisualization();
    }
  }, [analysisMode, isVisible, pentagonAnalysis, renderPentagonVisualization]);

  const renderModeContent = () => {
    switch (analysisMode) {
      case 'correlations':
        return (
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-sm text-neutral-400">
              <ChartBarIcon className="w-4 h-4" />
              <span>Parameter correlation analysis ({gridResults.length.toLocaleString()} models)</span>
            </div>
            <CorrelationHeatmap
              models={gridResults}
              width={600}
              height={400}
              className="mx-auto"
            />
          </div>
        );

      case 'directional':
        return (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm text-neutral-400">
                <ArrowPathIcon className="w-4 h-4" />
                <span>Directional sensitivity analysis</span>
              </div>
              <button
                onClick={computeDirectional}
                className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded transition-colors"
              >
                Compute Sensitivity
              </button>
            </div>
            {directionalResults ? (
              <DirectionalSensitivityPlot
                sensitivity={directionalResults}
                groundTruth={currentParameters}
                width={600}
                height={400}
              />
            ) : (
              <div className="text-center text-neutral-500 py-8">
                Click &quot;Compute Sensitivity&quot; to analyze parameter directionality
              </div>
            )}
          </div>
        );

      case 'pentagon-gt':
        return (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm text-neutral-400">
                <BeakerIcon className="w-4 h-4" />
                <span>Pentagon Ground Truth System</span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => {
                    const groundTruth = generateGroundTruthPentagon(currentParameters);
                    setPentagonAnalysis(prev => ({ ...prev, groundTruth }));
                  }}
                  className="px-3 py-1 bg-green-600 hover:bg-green-700 text-white text-sm rounded transition-colors"
                >
                  Load Ground Truth
                </button>
                {pentagonAnalysis.isRunning ? (
                  <button
                    onClick={() => setPentagonAnalysis(prev => ({ ...prev, isRunning: false }))}
                    className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded transition-colors flex items-center gap-1"
                  >
                    <PauseIcon className="w-3 h-3" />
                    Pause
                  </button>
                ) : (
                  <button
                    onClick={startPentagonAnalysis}
                    disabled={!pentagonAnalysis.groundTruth}
                    className="px-3 py-1 bg-blue-600 hover:bg-blue-700 disabled:bg-neutral-600 text-white text-sm rounded transition-colors flex items-center gap-1"
                  >
                    <PlayIcon className="w-3 h-3" />
                    Start Analysis
                  </button>
                )}
              </div>
            </div>

            {/* Progress and Stats */}
            {pentagonAnalysis.groundTruth && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div className="space-y-3">
                  <div className="bg-neutral-800 rounded p-3">
                    <div className="text-sm font-medium text-neutral-200 mb-2">Analysis Progress</div>
                    <div className="w-full bg-neutral-700 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${pentagonAnalysis.progress}%` }}
                      />
                    </div>
                    <div className="text-xs text-neutral-400 mt-1">
                      {pentagonAnalysis.currentIteration.toLocaleString()} / {pentagonAnalysis.config.maxIterations.toLocaleString()} iterations
                    </div>
                  </div>

                  {pentagonAnalysis.bestCandidates.length > 0 && (
                    <div className="bg-neutral-800 rounded p-3">
                      <div className="text-sm font-medium text-neutral-200 mb-2">Best Match</div>
                      <div className="text-xs text-neutral-400">
                        Distance: {pentagonAnalysis.bestCandidates[0].euclideanDistance.toFixed(4)}
                      </div>
                      <div className="text-xs text-neutral-400">
                        Tags: {pentagonAnalysis.bestCandidates[0].conditionTags.join(', ')}
                      </div>
                    </div>
                  )}
                </div>

                <div className="bg-neutral-800 rounded p-3">
                  <div className="text-sm font-medium text-neutral-200 mb-2">Pentagon Visualization</div>
                  <canvas
                    ref={canvasRef}
                    width={300}
                    height={200}
                    className="border border-neutral-600 rounded w-full"
                  />
                  <div className="text-xs text-neutral-400 mt-1">
                    <span className="text-blue-400">■</span> Ground Truth
                    <span className="text-red-400 ml-4">■</span> Current
                    <span className="text-green-400 ml-4">■</span> Best Match
                  </div>
                </div>
              </div>
            )}
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="space-y-4">
      {/* Mode Selection */}
      <div className="flex items-center gap-1 bg-neutral-800 rounded p-1">
        <button
          onClick={() => setAnalysisMode('correlations')}
          className={`px-3 py-2 text-sm rounded transition-colors flex items-center gap-2 ${
            analysisMode === 'correlations'
              ? 'bg-blue-600 text-white'
              : 'text-neutral-400 hover:text-neutral-200'
          }`}
        >
          <ChartBarIcon className="w-4 h-4" />
          Correlations
        </button>
        <button
          onClick={() => setAnalysisMode('directional')}
          className={`px-3 py-2 text-sm rounded transition-colors flex items-center gap-2 ${
            analysisMode === 'directional'
              ? 'bg-blue-600 text-white'
              : 'text-neutral-400 hover:text-neutral-200'
          }`}
        >
          <ArrowPathIcon className="w-4 h-4" />
          Directional
        </button>
        <button
          onClick={() => setAnalysisMode('pentagon-gt')}
          className={`px-3 py-2 text-sm rounded transition-colors flex items-center gap-2 ${
            analysisMode === 'pentagon-gt'
              ? 'bg-blue-600 text-white'
              : 'text-neutral-400 hover:text-neutral-200'
          }`}
        >
          <BeakerIcon className="w-4 h-4" />
          Pentagon GT
        </button>
      </div>

      {/* Content */}
      {renderModeContent()}
    </div>
  );
};

export default AnalysisTab;