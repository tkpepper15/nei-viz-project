/**
 * Directional Sensitivity Visualization Component
 *
 * Displays parameter sensitivity analysis results including:
 * - Principal direction vectors in parameter space
 * - Sensitivity magnitude heatmaps
 * - Spectral changes along parameter directions
 * - Uncertainty maps from surrogate models
 */

'use client';

import React, { useMemo, useState } from 'react';
import { CircuitParameters } from '../types/parameters';
import { DirectionalSensitivity, ParameterDirection } from '../utils/directionalAnalysis';

interface DirectionalSensitivityPlotProps {
  /** Sensitivity analysis results */
  sensitivity: DirectionalSensitivity;
  /** Ground truth parameters for reference */
  groundTruth: CircuitParameters;
  /** Surrogate model predictions for uncertainty visualization */
  uncertaintyMap?: { params: CircuitParameters; uncertainty: number }[];
  /** Callback for parameter selection */
  onParameterSelect?: (params: CircuitParameters) => void;
  /** Width of the visualization */
  width?: number;
  /** Height of the visualization */
  height?: number;
}

interface ParameterVisualizationProps {
  directions: ParameterDirection[];
  onDirectionSelect?: (direction: ParameterDirection) => void;
}

/**
 * Parameter space direction visualization
 */
const ParameterDirectionPlot: React.FC<ParameterVisualizationProps> = ({
  directions,
  onDirectionSelect
}) => {
  const [selectedDirection, setSelectedDirection] = useState<number>(0);

  const paramNames = ['Rsh', 'Ra', 'Ca', 'Rb', 'Cb'];

  const handleDirectionClick = (index: number) => {
    setSelectedDirection(index);
    if (onDirectionSelect) {
      onDirectionSelect(directions[index]);
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <h3 className="text-lg font-semibold text-white mb-4">Principal Parameter Directions</h3>

      {/* Direction selector */}
      <div className="flex space-x-2 mb-4">
        {directions.map((dir, index) => (
          <button
            key={index}
            onClick={() => handleDirectionClick(index)}
            className={`px-3 py-1 rounded text-sm ${
              selectedDirection === index
                ? 'bg-blue-500 text-white'
                : 'bg-gray-600 text-gray-300 hover:bg-gray-500'
            }`}
          >
            Dir {index + 1}
          </button>
        ))}
      </div>

      {/* Selected direction details */}
      {directions[selectedDirection] && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <h4 className="text-sm font-medium text-gray-300 mb-2">Direction Components</h4>
              <div className="space-y-1">
                {directions[selectedDirection].direction.map((component, index) => (
                  <div key={index} className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">{paramNames[index]}:</span>
                    <div className="flex items-center">
                      <div
                        className="h-2 bg-blue-400 mr-2 rounded"
                        style={{
                          width: `${Math.abs(component) * 50}px`,
                          backgroundColor: component >= 0 ? '#3b82f6' : '#ef4444'
                        }}
                      />
                      <span className="text-sm text-white w-12 text-right">
                        {component.toFixed(3)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h4 className="text-sm font-medium text-gray-300 mb-2">Analysis</h4>
              <div className="space-y-2 text-sm">
                <div>
                  <span className="text-gray-400">Sensitivity:</span>
                  <span className="text-white ml-2">
                    {directions[selectedDirection].sensitivity.toFixed(4)}
                  </span>
                </div>
                <div>
                  <span className="text-gray-400">Interpretation:</span>
                  <span className="text-white ml-2">
                    {directions[selectedDirection].interpretation}
                  </span>
                </div>
                <div>
                  <span className="text-gray-400">Spectral Effect:</span>
                  <span className="text-white ml-2">
                    {directions[selectedDirection].spectralEffect}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Vector visualization */}
          <div className="mt-4">
            <h4 className="text-sm font-medium text-gray-300 mb-2">Direction Vector</h4>
            <svg width="400" height="100" className="bg-gray-700 rounded">
              {directions[selectedDirection].direction.map((component, index) => {
                const barWidth = Math.abs(component) * 60;
                const x = 60 + index * 70;
                const y = 50;
                const isPositive = component >= 0;

                return (
                  <g key={index}>
                    {/* Parameter label */}
                    <text
                      x={x}
                      y={y + 25}
                      className="fill-gray-300 text-xs"
                      textAnchor="middle"
                    >
                      {paramNames[index]}
                    </text>

                    {/* Direction bar */}
                    <rect
                      x={x - barWidth/2}
                      y={isPositive ? y - barWidth : y}
                      width={barWidth}
                      height={Math.abs(barWidth)}
                      fill={isPositive ? '#3b82f6' : '#ef4444'}
                      opacity={0.8}
                    />

                    {/* Value label */}
                    <text
                      x={x}
                      y={y - barWidth - 5}
                      className="fill-white text-xs"
                      textAnchor="middle"
                    >
                      {component.toFixed(2)}
                    </text>
                  </g>
                );
              })}

              {/* Zero line */}
              <line
                x1={30}
                y1={50}
                x2={370}
                y2={50}
                stroke="#6b7280"
                strokeWidth={1}
                strokeDasharray="2,2"
              />
            </svg>
          </div>
        </div>
      )}
    </div>
  );
};

/**
 * Sensitivity heatmap for parameter pairs
 */
const SensitivityHeatmap: React.FC<{
  sensitivity: DirectionalSensitivity;
  width: number;
  height: number;
}> = ({ sensitivity, width, height }) => {
  const paramNames = ['Rsh', 'Ra', 'Ca', 'Rb', 'Cb'];

  // Compute pairwise sensitivities from Jacobian
  const pairwiseSensitivities = useMemo(() => {
    const { jacobian } = sensitivity;
    const matrix: number[][] = Array(5).fill(null).map(() => Array(5).fill(0));

    for (let i = 0; i < 5; i++) {
      for (let j = 0; j < 5; j++) {
        if (i === j) {
          // Diagonal: total variance explained by this parameter
          let variance = 0;
          for (let freq = 0; freq < jacobian.real.length; freq++) {
            variance += jacobian.real[freq][i] ** 2 + jacobian.imag[freq][i] ** 2;
          }
          matrix[i][j] = variance;
        } else {
          // Off-diagonal: cross-correlation of sensitivities
          let correlation = 0;
          for (let freq = 0; freq < jacobian.real.length; freq++) {
            correlation += jacobian.real[freq][i] * jacobian.real[freq][j] +
                          jacobian.imag[freq][i] * jacobian.imag[freq][j];
          }
          matrix[i][j] = Math.abs(correlation);
        }
      }
    }

    return matrix;
  }, [sensitivity]);

  const maxSensitivity = Math.max(...pairwiseSensitivities.flat());
  const cellSize = Math.min(width / 6, height / 6);

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <h3 className="text-lg font-semibold text-white mb-4">Parameter Sensitivity Matrix</h3>

      <svg width={width} height={height}>
        {pairwiseSensitivities.map((row, i) =>
          row.map((value, j) => {
            const intensity = value / maxSensitivity;
            const x = (j + 0.5) * cellSize;
            const y = (i + 0.5) * cellSize;

            return (
              <g key={`${i}-${j}`}>
                <rect
                  x={x}
                  y={y}
                  width={cellSize * 0.8}
                  height={cellSize * 0.8}
                  fill={`rgba(59, 130, 246, ${intensity})`}
                  stroke="#374151"
                  strokeWidth={1}
                />
                <text
                  x={x + cellSize * 0.4}
                  y={y + cellSize * 0.5}
                  className="fill-white text-xs"
                  textAnchor="middle"
                  dominantBaseline="middle"
                >
                  {value.toExponential(1)}
                </text>
              </g>
            );
          })
        )}

        {/* Row labels */}
        {paramNames.map((name, i) => (
          <text
            key={`row-${i}`}
            x={10}
            y={(i + 1) * cellSize}
            className="fill-gray-300 text-sm"
            textAnchor="middle"
            dominantBaseline="middle"
          >
            {name}
          </text>
        ))}

        {/* Column labels */}
        {paramNames.map((name, j) => (
          <text
            key={`col-${j}`}
            x={(j + 1) * cellSize}
            y={25}
            className="fill-gray-300 text-sm"
            textAnchor="middle"
          >
            {name}
          </text>
        ))}
      </svg>
    </div>
  );
};

/**
 * Main directional sensitivity visualization component
 */
const DirectionalSensitivityPlot: React.FC<DirectionalSensitivityPlotProps> = ({
  sensitivity,
  groundTruth, // eslint-disable-line @typescript-eslint/no-unused-vars
  uncertaintyMap,
  width = 800,
  height = 600
}) => {
  const [selectedTab, setSelectedTab] = useState<'directions' | 'heatmap' | 'uncertainty'>('directions');
  const [selectedDirection, setSelectedDirection] = useState<ParameterDirection | null>(null); // eslint-disable-line @typescript-eslint/no-unused-vars

  const handleDirectionSelect = (direction: ParameterDirection) => {
    setSelectedDirection(direction);
  };

  return (
    <div className="w-full space-y-4">
      {/* Tab selector */}
      <div className="flex space-x-2">
        {[
          { key: 'directions', label: 'Principal Directions' },
          { key: 'heatmap', label: 'Sensitivity Matrix' },
          { key: 'uncertainty', label: 'Uncertainty Map' }
        ].map(tab => (
          <button
            key={tab.key}
            onClick={() => setSelectedTab(tab.key as 'directions' | 'heatmap' | 'uncertainty')}
            className={`px-4 py-2 rounded ${
              selectedTab === tab.key
                ? 'bg-blue-500 text-white'
                : 'bg-gray-600 text-gray-300 hover:bg-gray-500'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div style={{ width, height }}>
        {selectedTab === 'directions' && (
          <ParameterDirectionPlot
            directions={sensitivity.principalDirections}
            onDirectionSelect={handleDirectionSelect}
          />
        )}

        {selectedTab === 'heatmap' && (
          <SensitivityHeatmap
            sensitivity={sensitivity}
            width={width}
            height={height}
          />
        )}

        {selectedTab === 'uncertainty' && uncertaintyMap && (
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-white mb-4">Model Uncertainty Map</h3>
            <div className="text-gray-300">
              <p>Uncertainty visualization shows where the surrogate model</p>
              <p>has high prediction variance - indicating areas needing more sampling.</p>
              <div className="mt-4">
                <p className="text-sm">
                  Total uncertainty points: {uncertaintyMap.length}
                </p>
                <p className="text-sm">
                  Max uncertainty: {Math.max(...uncertaintyMap.map(u => u.uncertainty)).toFixed(4)}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Summary statistics */}
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-white mb-2">Analysis Summary</h3>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-400">Condition Number:</span>
            <span className="text-white ml-2">{sensitivity.conditionNumber.toFixed(2)}</span>
          </div>
          <div>
            <span className="text-gray-400">Principal Directions:</span>
            <span className="text-white ml-2">{sensitivity.principalDirections.length}</span>
          </div>
          <div>
            <span className="text-gray-400">Most Sensitive:</span>
            <span className="text-white ml-2">
              {sensitivity.principalDirections[0]?.interpretation || 'None'}
            </span>
          </div>
          <div>
            <span className="text-gray-400">Identifiability:</span>
            <span className="text-white ml-2">
              {sensitivity.conditionNumber < 10 ? 'Good' :
               sensitivity.conditionNumber < 100 ? 'Moderate' : 'Poor'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DirectionalSensitivityPlot;