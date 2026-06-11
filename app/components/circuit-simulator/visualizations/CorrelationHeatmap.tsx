"use client";

import React, { useMemo, useRef, useEffect, useState } from 'react';
import { ModelSnapshot } from '../types';

interface CorrelationHeatmapProps {
  models: ModelSnapshot[];
  width?: number;
  height?: number;
  className?: string;
}

interface CorrelationMatrix {
  parameters: string[];
  matrix: number[][];
  labels: string[];
}

// Calculate correlation matrix for circuit parameters and resnorm
function calculateCorrelationMatrix(models: ModelSnapshot[]): CorrelationMatrix {
  if (models.length < 2) {
    return {
      parameters: [],
      matrix: [],
      labels: []
    };
  }

  // Extract data for correlation calculation
  const data = models
    .filter(m => m.parameters && typeof m.resnorm === 'number' && !isNaN(m.resnorm))
    .map(model => ({
      resnorm: model.resnorm || 0,
      rsh: model.parameters.Rsh,
      ra: model.parameters.Ra,
      rb: model.parameters.Rb,
      ca: model.parameters.Ca,
      cb: model.parameters.Cb
    }));

  if (data.length < 2) {
    return {
      parameters: [],
      matrix: [],
      labels: []
    };
  }

  const parameters = ['Resnorm', 'Rsh (Ω)', 'Ra (Ω)', 'Rb (Ω)', 'Ca (F)', 'Cb (F)'];
  const keys = ['resnorm', 'rsh', 'ra', 'rb', 'ca', 'cb'] as const;

  // Calculate means
  const means = keys.map(key =>
    data.reduce((sum, d) => sum + d[key], 0) / data.length
  );

  // Calculate correlation matrix
  const matrix: number[][] = [];

  for (let i = 0; i < keys.length; i++) {
    matrix[i] = [];
    for (let j = 0; j < keys.length; j++) {
      if (i === j) {
        matrix[i][j] = 1.0;
      } else {
        // Calculate Pearson correlation coefficient
        let numerator = 0;
        let sumSqX = 0;
        let sumSqY = 0;

        for (const point of data) {
          const x = point[keys[i]] - means[i];
          const y = point[keys[j]] - means[j];
          numerator += x * y;
          sumSqX += x * x;
          sumSqY += y * y;
        }

        const denominator = Math.sqrt(sumSqX * sumSqY);
        matrix[i][j] = denominator === 0 ? 0 : numerator / denominator;
      }
    }
  }

  return {
    parameters: [...keys],
    matrix,
    labels: parameters
  };
}

// Color scale for correlation values (-1 to 1)
function getCorrelationColor(value: number): string {
  // Clamp value between -1 and 1
  const clampedValue = Math.max(-1, Math.min(1, value));

  // Blue for negative, white for zero, red for positive correlations
  if (clampedValue < 0) {
    const intensity = Math.abs(clampedValue);
    const blueValue = Math.floor(255 - (intensity * 155)); // 100-255
    return `rgb(${blueValue}, ${blueValue}, 255)`;
  } else if (clampedValue > 0) {
    const intensity = clampedValue;
    const redValue = Math.floor(255 - (intensity * 155)); // 100-255
    return `rgb(255, ${redValue}, ${redValue})`;
  } else {
    return 'rgb(255, 255, 255)'; // White for zero correlation
  }
}

// Format correlation value for display
function formatCorrelation(value: number): string {
  if (Math.abs(value) < 0.001) return '0';
  return value.toFixed(3);
}

export const CorrelationHeatmap: React.FC<CorrelationHeatmapProps> = ({
  models,
  width = 600,
  height = 500,
  className = ''
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredCell, setHoveredCell] = useState<{row: number, col: number, value: number} | null>(null);
  const [mousePos, setMousePos] = useState<{x: number, y: number}>({x: 0, y: 0});

  // Calculate correlation matrix
  const correlationData = useMemo(() => calculateCorrelationMatrix(models), [models]);

  // Draw heatmap on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !correlationData.matrix.length) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    canvas.width = width;
    canvas.height = height;

    // Clear canvas
    ctx.fillStyle = '#1f2937'; // neutral-800
    ctx.fillRect(0, 0, width, height);

    const numParams = correlationData.labels.length;
    if (numParams === 0) return;

    // Calculate cell dimensions
    const labelWidth = 120;
    const labelHeight = 30;
    const heatmapWidth = width - labelWidth - 20;
    const heatmapHeight = height - labelHeight - 20;
    const cellWidth = heatmapWidth / numParams;
    const cellHeight = heatmapHeight / numParams;

    // Draw parameter labels (y-axis)
    ctx.fillStyle = '#e5e7eb'; // neutral-200
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    for (let i = 0; i < numParams; i++) {
      const y = labelHeight + (i + 0.5) * cellHeight;
      ctx.fillText(correlationData.labels[i], labelWidth - 5, y);
    }

    // Draw parameter labels (x-axis) - rotated
    ctx.save();
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';

    for (let j = 0; j < numParams; j++) {
      const x = labelWidth + (j + 0.5) * cellWidth;
      ctx.save();
      ctx.translate(x, labelHeight - 5);
      ctx.rotate(-Math.PI / 4); // 45 degree rotation
      ctx.fillText(correlationData.labels[j], 0, 0);
      ctx.restore();
    }
    ctx.restore();

    // Draw heatmap cells
    for (let i = 0; i < numParams; i++) {
      for (let j = 0; j < numParams; j++) {
        const value = correlationData.matrix[i][j];
        const x = labelWidth + j * cellWidth;
        const y = labelHeight + i * cellHeight;

        // Fill cell with correlation color
        ctx.fillStyle = getCorrelationColor(value);
        ctx.fillRect(x, y, cellWidth, cellHeight);

        // Draw cell border
        ctx.strokeStyle = '#374151'; // neutral-700
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, cellWidth, cellHeight);

        // Draw correlation value text
        ctx.fillStyle = Math.abs(value) > 0.5 ? '#ffffff' : '#000000';
        ctx.font = '11px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(
          formatCorrelation(value),
          x + cellWidth / 2,
          y + cellHeight / 2
        );
      }
    }

  }, [correlationData, width, height]);

  // Handle mouse events for hover tooltip
  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !correlationData.matrix.length) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    const numParams = correlationData.labels.length;
    const labelWidth = 120;
    const labelHeight = 30;
    const heatmapWidth = width - labelWidth - 20;
    const heatmapHeight = height - labelHeight - 20;
    const cellWidth = heatmapWidth / numParams;
    const cellHeight = heatmapHeight / numParams;

    // Check if mouse is over heatmap area
    if (mouseX >= labelWidth && mouseX < labelWidth + heatmapWidth &&
        mouseY >= labelHeight && mouseY < labelHeight + heatmapHeight) {

      const col = Math.floor((mouseX - labelWidth) / cellWidth);
      const row = Math.floor((mouseY - labelHeight) / cellHeight);

      if (row >= 0 && row < numParams && col >= 0 && col < numParams) {
        setHoveredCell({
          row,
          col,
          value: correlationData.matrix[row][col]
        });
        setMousePos({x: event.clientX, y: event.clientY});
        return;
      }
    }

    setHoveredCell(null);
  };

  const handleMouseLeave = () => {
    setHoveredCell(null);
  };

  if (!correlationData.matrix.length) {
    return (
      <div className={`flex items-center justify-center bg-neutral-800 rounded-lg ${className}`}>
        <div className="text-center">
          <p className="text-neutral-400 mb-2">No correlation data available</p>
          <p className="text-sm text-neutral-500">
            Need at least 2 models with valid parameters and resnorm values
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={`relative ${className}`}>
      <div className="mb-4">
        <h3 className="text-lg font-medium text-neutral-200 mb-2">
          Parameter Correlation Matrix
        </h3>
        <p className="text-sm text-neutral-400">
          Correlation coefficients between circuit parameters and resnorm values.
          Blue indicates negative correlation, red indicates positive correlation.
        </p>
      </div>

      <div className="relative inline-block">
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          className="border border-neutral-600 rounded cursor-crosshair"
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        />

        {/* Hover tooltip */}
        {hoveredCell && (
          <div
            className="fixed z-50 bg-neutral-900 border border-neutral-600 rounded px-3 py-2 text-sm pointer-events-none"
            style={{
              left: mousePos.x + 10,
              top: mousePos.y - 10,
              transform: mousePos.x > window.innerWidth - 200 ? 'translateX(-100%)' : 'none'
            }}
          >
            <div className="text-neutral-200 font-medium">
              {correlationData.labels[hoveredCell.row]} ↔ {correlationData.labels[hoveredCell.col]}
            </div>
            <div className="text-neutral-300">
              Correlation: <span className="font-mono">{hoveredCell.value.toFixed(6)}</span>
            </div>
            <div className="text-xs text-neutral-400 mt-1">
              {Math.abs(hoveredCell.value) > 0.7 ? 'Strong' :
               Math.abs(hoveredCell.value) > 0.3 ? 'Moderate' : 'Weak'}
              {hoveredCell.value > 0 ? ' positive' : ' negative'} correlation
            </div>
          </div>
        )}
      </div>

      {/* Color scale legend */}
      <div className="mt-4 flex items-center justify-center">
        <div className="flex items-center space-x-4">
          <span className="text-sm text-neutral-400">Correlation:</span>
          <div className="flex items-center space-x-2">
            <span className="text-xs text-neutral-500">-1.0</span>
            <div className="flex h-4 w-32 rounded overflow-hidden">
              {Array.from({length: 20}, (_, i) => {
                const value = -1 + (i / 19) * 2; // -1 to 1
                return (
                  <div
                    key={i}
                    className="flex-1 h-full"
                    style={{backgroundColor: getCorrelationColor(value)}}
                  />
                );
              })}
            </div>
            <span className="text-xs text-neutral-500">+1.0</span>
          </div>
        </div>
      </div>

      {/* Statistics summary */}
      <div className="mt-4 text-xs text-neutral-400">
        <span>Computed from {models.filter(m => m.parameters && typeof m.resnorm === 'number').length} models</span>
      </div>
    </div>
  );
};

export default CorrelationHeatmap;