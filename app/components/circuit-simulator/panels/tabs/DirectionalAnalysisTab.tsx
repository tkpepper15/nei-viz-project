/**
 * Directional Analysis Tab Component
 *
 * Integrates directional sensitivity analysis into the existing SpideyPlot interface.
 * Provides a comprehensive workflow for understanding parameter directionality:
 * 1. Compute sensitivity analysis at ground truth
 * 2. Generate adaptive sampling strategy
 * 3. Build surrogate model for uncertainty quantification
 * 4. Visualize directional insights and predictions
 */

'use client';

import React, { useState, useCallback, useMemo } from 'react';
import { CircuitParameters } from '../../types/parameters';
import { ImpedancePoint, calculate_impedance_spectrum } from '../../utils/resnorm';
import {
  computeDirectionalSensitivity,
  generateAdaptiveSampling,
  compressSpectra,
  DirectionalSensitivity,
  ParameterDirection,
  createFrequencyArray
} from '../../utils/directionalAnalysis';
import {
  createSurrogateModel,
  TrainingData,
  SimplifiedGPR,
  ActiveSampler
} from '../../utils/surrogateModel';
import DirectionalSensitivityPlot from '../../visualizations/DirectionalSensitivityPlot';
import {
  exportAnalysisPackage,
  ExportableResults,
  createAnalysisSummary
} from '../../utils/dataExport';

interface DirectionalAnalysisTabProps {
  /** Ground truth circuit parameters */
  groundTruthParams: CircuitParameters;
  /** Callback when analysis results change */
  onAnalysisUpdate?: (results: AnalysisResults) => void;
  /** Whether analysis is currently running */
  isComputing?: boolean;
}

interface AnalysisResults {
  sensitivity: DirectionalSensitivity;
  adaptiveSamples: CircuitParameters[];
  surrogateModel: SimplifiedGPR;
  activeSampler: ActiveSampler;
  uncertaintyMap: { params: CircuitParameters; uncertainty: number }[];
  spectralCompression: unknown;
}

const DirectionalAnalysisTab: React.FC<DirectionalAnalysisTabProps> = ({
  groundTruthParams,
  onAnalysisUpdate,
  isComputing = false
}) => {
  const [analysisResults, setAnalysisResults] = useState<AnalysisResults | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStage, setAnalysisStage] = useState<string>('');
  const [numSamples, setNumSamples] = useState(500);
  const [selectedDirection, setSelectedDirection] = useState<ParameterDirection | null>(null);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);

  /**
   * Run complete directional analysis workflow
   */
  const runDirectionalAnalysis = useCallback(async () => {
    if (!groundTruthParams || isComputing) return;

    setIsAnalyzing(true);
    try {
      // Stage 1: Compute directional sensitivity at ground truth
      setAnalysisStage('Computing directional sensitivity...');
      const frequencies = createFrequencyArray(
        groundTruthParams.frequency_range[0],
        groundTruthParams.frequency_range[1],
        20
      );

      const sensitivity = computeDirectionalSensitivity(groundTruthParams, frequencies);

      // Stage 2: Generate adaptive sampling points
      setAnalysisStage('Generating adaptive sampling strategy...');
      const adaptiveSamples = generateAdaptiveSampling(groundTruthParams, sensitivity, numSamples);

      // Stage 3: Compute spectra for training data
      setAnalysisStage('Computing spectra for surrogate training...');
      const trainingSpectra: ImpedancePoint[][] = [];
      const trainingResnorms: number[] = [];

      // Compute first 50 samples for initial training
      const initialSamples = adaptiveSamples.slice(0, Math.min(50, adaptiveSamples.length));
      for (const params of initialSamples) {
        const spectrum = calculate_impedance_spectrum(params);
        trainingSpectra.push(spectrum);
        // Simple resnorm calculation (could use more sophisticated method)
        const resnorm = spectrum.reduce((sum, point) => sum + point.magnitude, 0) / spectrum.length;
        trainingResnorms.push(resnorm);
      }

      // Stage 4: Compress spectra for efficient modeling
      setAnalysisStage('Compressing spectral data...');
      const spectralCompression = compressSpectra(trainingSpectra, 8);

      // Create training data for surrogate model
      const trainingData: TrainingData = {
        parameters: initialSamples,
        spectralCoefficients: trainingSpectra.map((spectrum) => {
          // Simple coefficient extraction (in practice, use SVD coefficients)
          return spectrum.slice(0, 8).map(point => point.magnitude);
        }),
        resnorms: trainingResnorms
      };

      // Stage 5: Train surrogate model
      setAnalysisStage('Training surrogate model...');
      const { model: surrogateModel, sampler: activeSampler } = createSurrogateModel(trainingData, 8);

      // Stage 6: Generate uncertainty map
      setAnalysisStage('Computing uncertainty map...');
      const uncertaintySamples = adaptiveSamples.slice(50, 150); // Next 100 samples for uncertainty
      const uncertaintyMap = uncertaintySamples.map(params => ({
        params,
        uncertainty: surrogateModel.getUncertainty(params)
      }));

      const results: AnalysisResults = {
        sensitivity,
        adaptiveSamples,
        surrogateModel,
        activeSampler,
        uncertaintyMap,
        spectralCompression
      };

      setAnalysisResults(results);
      if (onAnalysisUpdate) {
        onAnalysisUpdate(results);
      }

    } catch (error) {
      console.error('Error in directional analysis:', error);
    } finally {
      setIsAnalyzing(false);
      setAnalysisStage('');
    }
  }, [groundTruthParams, isComputing, numSamples, onAnalysisUpdate]);

  /**
   * Handle direction selection from visualization
   */
  const handleDirectionSelect = useCallback((direction: ParameterDirection) => {
    setSelectedDirection(direction);
  }, []);

  // Suppress unused variable warning - function reserved for future use
  void handleDirectionSelect;

  /**
   * Export analysis results
   */
  const handleExportResults = useCallback(() => {
    if (!analysisResults) return;

    const exportData: ExportableResults = {
      sensitivity: analysisResults.sensitivity,
      adaptiveSamples: analysisResults.adaptiveSamples,
      uncertaintyMap: analysisResults.uncertaintyMap,
      spectralCompression: analysisResults.spectralCompression,
      metadata: {
        exportDate: new Date().toISOString(),
        version: '1.0.0',
        analysisType: 'directional_sensitivity'
      }
    };

    exportAnalysisPackage(exportData, 'spideyplot_directional_analysis');
  }, [analysisResults]);

  /**
   * Run active learning to find next sampling points
   */
  const handleActiveLearning = useCallback(() => {
    if (!analysisResults) return;

    console.log('Starting active learning...');
    // Generate candidate points
    const candidateParams = generateAdaptiveSampling(
      groundTruthParams,
      analysisResults.sensitivity,
      100 // Generate 100 candidate points
    );

    // Use acquisition function to select best points
    const nextSamples = analysisResults.activeSampler.selectNextSamples(candidateParams, 10);

    console.log('Recommended next sampling points:', nextSamples);

    // Could update state to show recommended points
    setAnalysisStage('Active learning complete - see console for recommended sampling points');
  }, [analysisResults, groundTruthParams]);

  /**
   * Generate directional insights summary
   */
  const directionalInsights = useMemo(() => {
    if (!analysisResults) return null;

    const { sensitivity } = analysisResults;
    const insights = [];

    // Overall identifiability
    const conditionNumber = sensitivity.conditionNumber;
    if (conditionNumber < 10) {
      insights.push({
        type: 'success',
        message: `Good parameter identifiability (condition number: ${conditionNumber.toFixed(1)})`
      });
    } else if (conditionNumber < 100) {
      insights.push({
        type: 'warning',
        message: `Moderate parameter identifiability (condition number: ${conditionNumber.toFixed(1)})`
      });
    } else {
      insights.push({
        type: 'error',
        message: `Poor parameter identifiability (condition number: ${conditionNumber.toFixed(1)})`
      });
    }

    // Most sensitive direction
    if (sensitivity.principalDirections.length > 0) {
      const topDirection = sensitivity.principalDirections[0];
      insights.push({
        type: 'info',
        message: `Most sensitive direction: ${topDirection.interpretation}`
      });
      insights.push({
        type: 'info',
        message: `Primary spectral effect: ${topDirection.spectralEffect}`
      });
    }

    // Prediction confidence
    if (analysisResults.uncertaintyMap.length > 0) {
      const avgUncertainty = analysisResults.uncertaintyMap.reduce((sum, u) => sum + u.uncertainty, 0) / analysisResults.uncertaintyMap.length;
      insights.push({
        type: 'info',
        message: `Average model uncertainty: ${avgUncertainty.toFixed(4)}`
      });
    }

    return insights;
  }, [analysisResults]);

  return (
    <div className="p-6 bg-gray-900 min-h-screen">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex justify-between items-center">
          <h1 className="text-3xl font-bold text-white">Directional Analysis</h1>
          <div className="flex space-x-3">
            <button
              onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}
              className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-500"
            >
              {showAdvancedSettings ? 'Hide' : 'Show'} Settings
            </button>
            <button
              onClick={runDirectionalAnalysis}
              disabled={isAnalyzing || isComputing}
              className="px-6 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-600 disabled:cursor-not-allowed"
            >
              {isAnalyzing ? 'Analyzing...' : 'Run Analysis'}
            </button>
          </div>
        </div>

        {/* Advanced Settings */}
        {showAdvancedSettings && (
          <div className="bg-gray-800 rounded-lg p-4 space-y-4">
            <h3 className="text-lg font-semibold text-white">Analysis Settings</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-gray-300 mb-1">Number of Samples</label>
                <input
                  type="number"
                  value={numSamples}
                  onChange={(e) => setNumSamples(parseInt(e.target.value))}
                  min={100}
                  max={2000}
                  className="w-full px-3 py-1 bg-gray-700 text-white rounded"
                />
              </div>
              <div>
                <label className="block text-sm text-gray-300 mb-1">Spectral Coefficients</label>
                <select className="w-full px-3 py-1 bg-gray-700 text-white rounded">
                  <option value={6}>6 coefficients</option>
                  <option value={8} selected>8 coefficients</option>
                  <option value={10}>10 coefficients</option>
                </select>
              </div>
            </div>
          </div>
        )}

        {/* Analysis Progress */}
        {isAnalyzing && (
          <div className="bg-gray-800 rounded-lg p-4">
            <div className="flex items-center space-x-3">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
              <span className="text-white">{analysisStage}</span>
            </div>
          </div>
        )}

        {/* Directional Insights */}
        {directionalInsights && (
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-white mb-3">Key Insights</h3>
            <div className="space-y-2">
              {directionalInsights.map((insight, index) => (
                <div key={index} className={`p-2 rounded text-sm ${{
                  success: 'bg-green-900 text-green-100',
                  warning: 'bg-yellow-900 text-yellow-100',
                  error: 'bg-red-900 text-red-100',
                  info: 'bg-blue-900 text-blue-100'
                }[insight.type]}`}>
                  {insight.message}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Main Visualization */}
        {analysisResults && (
          <DirectionalSensitivityPlot
            sensitivity={analysisResults.sensitivity}
            groundTruth={groundTruthParams}
            uncertaintyMap={analysisResults.uncertaintyMap}
            width={800}
            height={500}
          />
        )}

        {/* Selected Direction Details */}
        {selectedDirection && (
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-white mb-3">Selected Direction Analysis</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <h4 className="text-md font-medium text-gray-300 mb-2">Physical Meaning</h4>
                <p className="text-sm text-white">{selectedDirection.interpretation}</p>
                <p className="text-sm text-gray-400 mt-1">{selectedDirection.spectralEffect}</p>
              </div>
              <div>
                <h4 className="text-md font-medium text-gray-300 mb-2">Quantitative Impact</h4>
                <p className="text-sm text-white">
                  Sensitivity Magnitude: {selectedDirection.sensitivity.toFixed(4)}
                </p>
                <p className="text-sm text-gray-400">
                  Direction Vector: [{selectedDirection.direction.map(d => d.toFixed(3)).join(', ')}]
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Quick Actions */}
        {analysisResults && (
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-white mb-3">Next Steps</h3>
            <div className="grid grid-cols-3 gap-4">
              <button
                onClick={handleExportResults}
                className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-500 transition-colors"
              >
                Export Results
              </button>
              <button
                onClick={handleActiveLearning}
                className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-500 transition-colors"
              >
                Active Learning
              </button>
              <button
                onClick={() => {
                  if (analysisResults) {
                    const summary = createAnalysisSummary({
                      sensitivity: analysisResults.sensitivity,
                      adaptiveSamples: analysisResults.adaptiveSamples,
                      uncertaintyMap: analysisResults.uncertaintyMap,
                      spectralCompression: analysisResults.spectralCompression,
                      metadata: {
                        exportDate: new Date().toISOString(),
                        version: '1.0.0',
                        analysisType: 'directional_sensitivity'
                      }
                    });
                    console.log(summary);
                    alert('Analysis summary printed to console');
                  }
                }}
                className="px-4 py-2 bg-orange-600 text-white rounded hover:bg-orange-500 transition-colors"
              >
                Show Summary
              </button>
            </div>
          </div>
        )}

        {/* Ground Truth Display */}
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-3">Ground Truth Parameters</h3>
          <div className="grid grid-cols-5 gap-4 text-sm">
            <div>
              <span className="text-gray-400">Rsh:</span>
              <span className="text-white ml-2">{groundTruthParams.Rsh.toFixed(1)} Ω</span>
            </div>
            <div>
              <span className="text-gray-400">Ra:</span>
              <span className="text-white ml-2">{groundTruthParams.Ra.toFixed(1)} Ω</span>
            </div>
            <div>
              <span className="text-gray-400">Ca:</span>
              <span className="text-white ml-2">{(groundTruthParams.Ca * 1e6).toFixed(1)} μF</span>
            </div>
            <div>
              <span className="text-gray-400">Rb:</span>
              <span className="text-white ml-2">{groundTruthParams.Rb.toFixed(1)} Ω</span>
            </div>
            <div>
              <span className="text-gray-400">Cb:</span>
              <span className="text-white ml-2">{(groundTruthParams.Cb * 1e6).toFixed(1)} μF</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DirectionalAnalysisTab;