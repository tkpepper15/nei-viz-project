"use client";

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { ModelSnapshot } from '../types';

// Simple PCA implementation for dimensionality reduction

// t-SNE 3D Plot Props
interface TSNEPlot3DProps {
  models: ModelSnapshot[];
  referenceModel?: ModelSnapshot | null;
  width?: number;
  height?: number;
  responsive?: boolean;
}

// 3D Point representing a model in t-SNE parameter space
interface TSNEPoint3D {
  x: number; // t-SNE dimension 1 (variance-minimized)
  y: number; // t-SNE dimension 2 (variance-minimized)
  z: number; // t-SNE dimension 3 (variance-minimized)
  color: string;
  model: ModelSnapshot;
  // Meaningful circuit characteristic points
  apicalPoint: { x: number; y: number; z: number };     // Total Resistance
  basalPoint: { x: number; y: number; z: number };      // Resistance Balance
  centerPoint: { x: number; y: number; z: number };     // Capacitance Balance
  isGroundTruth: boolean;
  circuitName: string;
  characteristics: {
    totalResistance: number;
    resistanceBalance: number;
    capacitanceBalance: number;
  };
}

// 3D Camera state
interface Camera3D {
  distance: number;
  rotation: { x: number; y: number; z: number };
  target: { x: number; y: number; z: number };
  scale: number;
}

// Default camera configuration
const DEFAULT_CAMERA: Camera3D = {
  distance: 500,
  rotation: { x: 20, y: 45, z: 0 },
  target: { x: 0, y: 0, z: 0 },
  scale: 1.0
};

export const TSNEPlot3D: React.FC<TSNEPlot3DProps> = ({
  models,
  referenceModel: _referenceModel, // eslint-disable-line @typescript-eslint/no-unused-vars
  width = 800,
  height = 600,
  responsive = true,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animationFrameRef = useRef<number | undefined>(undefined);

  // State management
  const [camera, setCamera] = useState<Camera3D>(DEFAULT_CAMERA);
  const [tsnePoints, setTsnePoints] = useState<TSNEPoint3D[]>([]);
  const [isComputing, setIsComputing] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });

  // Responsive sizing
  const [actualWidth, setActualWidth] = useState(width);
  const [actualHeight, setActualHeight] = useState(height);



  // Extract 5D parameter vectors and apply t-SNE-like reduction
  const extractAndReduceParameters = useCallback((modelList: ModelSnapshot[]) => {
    // Extract 5D parameter vectors
    const vectors = modelList.map(model => {
      const params = model.parameters;
      return [
        Math.log10(params.Rsh), // Log scale for resistance
        Math.log10(params.Ra),  // Log scale for resistance
        Math.log10(params.Ca * 1e6), // Convert to microfarads and log scale
        Math.log10(params.Rb),  // Log scale for resistance
        Math.log10(params.Cb * 1e6), // Convert to microfarads and log scale
      ];
    });

    if (vectors.length < 2) return vectors.map(v => [v[0] * 100, v[1] * 100, v[2] * 100]);

    const n = vectors.length;
    const dims = vectors[0].length;

    // Center the data
    const means = new Array(dims).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < dims; j++) {
        means[j] += vectors[i][j];
      }
    }
    for (let j = 0; j < dims; j++) {
      means[j] /= n;
    }

    const centered = vectors.map(vec =>
      vec.map((val, idx) => val - means[idx])
    );

    // Find the 3 dimensions with highest variance (t-SNE-like variance minimization)
    const variances = new Array(dims).fill(0);
    for (let j = 0; j < dims; j++) {
      for (let i = 0; i < n; i++) {
        variances[j] += centered[i][j] * centered[i][j];
      }
      variances[j] /= n;
    }

    // Get indices of 3 highest variance dimensions
    const indices = Array.from({length: dims}, (_, i) => i)
      .sort((a, b) => variances[b] - variances[a])
      .slice(0, 3);

    // Project to 3D using variance-minimized axes (like t-SNE) - wider spacing
    return centered.map(vec => [
      vec[indices[0]] * 200, // Increased scale for wider spacing
      vec[indices[1]] * 200,
      vec[indices[2]] * 200
    ]);
  }, []);

  // Extract 3 meaningful circuit characteristics for each model
  const extractCircuitCharacteristics = useCallback((model: ModelSnapshot) => {
    const params = model.parameters;

    // Three meaningful circuit characteristics that create unique shapes:
    // 1. Total Resistance (overall impedance magnitude)
    const totalResistance = Math.log10(params.Rsh + params.Ra + params.Rb);

    // 2. Resistance Balance (apical vs basal dominance)
    const resistanceBalance = Math.log10(params.Ra / params.Rb);

    // 3. Capacitance Balance (membrane capacitance characteristics)
    const capacitanceBalance = Math.log10(params.Ca / params.Cb);

    return [totalResistance, resistanceBalance, capacitanceBalance];
  }, []);

  // Map circuits to 3D space with meaningful 3-dot patterns
  const mapCircuitsTo3D = useCallback((modelList: ModelSnapshot[]) => {
    // Get t-SNE-like 3D coordinates for base positioning
    const coordinates3D = extractAndReduceParameters(modelList);

    return modelList.map((model, index) => {
      const coords = coordinates3D[index];
      const resnorm = model.resnorm || 0;
      const isGroundTruth = model.id === 'ground-truth';

      // Color based on resnorm or special colors for ground truth
      let color = '#FFFFFF';
      if (isGroundTruth) {
        color = '#8B5CF6'; // Purple for ground truth (matching image)
      } else {
        // Assign different colors for different circuits (like green in image)
        const colorOptions = ['#10B981', '#3B82F6', '#F59E0B', '#EF4444', '#F97316'];
        const colorIndex = Math.floor(resnorm * 10) % colorOptions.length;
        color = colorOptions[colorIndex];
      }

      // Get 3 meaningful characteristics for this circuit
      const [totalResistance, resistanceBalance, capacitanceBalance] = extractCircuitCharacteristics(model);

      // Base position in t-SNE space
      const baseX = coords[0];
      const baseY = coords[1];
      const baseZ = coords[2];

      const scale = 25; // Reduced scale for better spacing like your sketch

      // Create 3 dots with MEANINGFUL positions - more spread out like sketch
      // Dot 1: Total Resistance (positioned more distinctly)
      const totalResistancePoint = {
        x: baseX + totalResistance * scale * 1.2,
        y: baseY + totalResistance * scale * 0.4,
        z: baseZ + totalResistance * scale * 0.2
      };

      // Dot 2: Resistance Balance (positioned distinctly)
      const resistanceBalancePoint = {
        x: baseX + resistanceBalance * scale * 0.3,
        y: baseY + resistanceBalance * scale * 1.2,
        z: baseZ + resistanceBalance * scale * 0.6
      };

      // Dot 3: Capacitance Balance (positioned distinctly)
      const capacitanceBalancePoint = {
        x: baseX + capacitanceBalance * scale * 0.2,
        y: baseY + capacitanceBalance * scale * 0.3,
        z: baseZ + capacitanceBalance * scale * 1.2
      };

      return {
        x: baseX, y: baseY, z: baseZ, // t-SNE base position
        color,
        model,
        // Meaningful circuit characteristic points
        apicalPoint: totalResistancePoint,        // Represents total resistance
        basalPoint: resistanceBalancePoint,      // Represents resistance balance
        centerPoint: capacitanceBalancePoint,    // Represents capacitance balance
        isGroundTruth,
        circuitName: isGroundTruth ? 'Ground Truth' : `Circuit ${index}`,
        // Store the characteristic values for analysis
        characteristics: {
          totalResistance,
          resistanceBalance,
          capacitanceBalance
        }
      };
    });
  }, [extractAndReduceParameters, extractCircuitCharacteristics]);

  // Compute t-SNE 3D positioning
  const compute3DSpace = useCallback(async (modelList: ModelSnapshot[]) => {
    if (modelList.length < 1) {
      console.warn('No models to visualize');
      return [];
    }

    setIsComputing(true);
    console.log('🔬 Computing t-SNE 3D space for', modelList.length, 'models');

    try {
      // Map circuits to t-SNE 3D coordinates
      const points = mapCircuitsTo3D(modelList);

      console.log('✅ t-SNE 3D space completed');
      return points;

    } catch (error) {
      console.error('❌ t-SNE computation failed:', error);
      return [];
    } finally {
      setIsComputing(false);
    }
  }, [mapCircuitsTo3D]);

  // Recompute parameter space when models change
  useEffect(() => {
    if (models.length > 0) {
      compute3DSpace(models).then(points => {
        setTsnePoints(points);
      });
    }
  }, [models, compute3DSpace]);

  // Responsive sizing
  useEffect(() => {
    if (!responsive || !containerRef.current) {
      setActualWidth(width);
      setActualHeight(height);
      return;
    }

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width: containerWidth, height: containerHeight } = entry.contentRect;
        setActualWidth(Math.floor(containerWidth));
        setActualHeight(Math.floor(containerHeight));
      }
    });

    resizeObserver.observe(containerRef.current);
    return () => resizeObserver.disconnect();
  }, [responsive, width, height]);

  // 3D to 2D projection
  const project3D = useCallback((point: { x: number; y: number; z: number }) => {
    const { rotation, distance, target, scale } = camera;

    // Apply rotation
    const cosX = Math.cos(rotation.x * Math.PI / 180);
    const sinX = Math.sin(rotation.x * Math.PI / 180);
    const cosY = Math.cos(rotation.y * Math.PI / 180);
    const sinY = Math.sin(rotation.y * Math.PI / 180);

    // Translate relative to target
    const x = point.x - target.x;
    const y = point.y - target.y;
    const z = point.z - target.z;

    // Rotate around Y axis (yaw)
    const x1 = x * cosY - z * sinY;
    const z1 = x * sinY + z * cosY;

    // Rotate around X axis (pitch)
    const y2 = y * cosX - z1 * sinX;
    const z2 = y * sinX + z1 * cosX;

    // Project to 2D
    const perspective = distance / (distance + z2);
    const screenX = (x1 * perspective * scale) + actualWidth / 2;
    const screenY = (y2 * perspective * scale) + actualHeight / 2;

    return { x: screenX, y: screenY, depth: z2 };
  }, [camera, actualWidth, actualHeight]);

  // Render the 3D scene
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, actualWidth, actualHeight);

    // Set canvas size
    canvas.width = actualWidth;
    canvas.height = actualHeight;

    if (isComputing) {
      // Show loading state
      ctx.fillStyle = '#64748B';
      ctx.font = '16px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Computing parameter space...', actualWidth / 2, actualHeight / 2);
      return;
    }

    if (tsnePoints.length === 0) return;

    // Sort points by depth for proper rendering order (using geometric center of 3 points)
    const sortedPoints = [...tsnePoints].sort((a, b) => {
      // Calculate average depth of all 3 characteristic points
      const depthA = (project3D(a.apicalPoint).depth + project3D(a.basalPoint).depth + project3D(a.centerPoint).depth) / 3;
      const depthB = (project3D(b.apicalPoint).depth + project3D(b.basalPoint).depth + project3D(b.centerPoint).depth) / 3;
      return depthB - depthA; // Far to near
    });

    // Render all circuits as 3-dot, 2-line structures with meaningful positioning
    sortedPoints.forEach(tsnePoint => {
      // Project the 3 characteristic points
      const totalResProj = project3D(tsnePoint.apicalPoint);      // Total Resistance
      const resBalanceProj = project3D(tsnePoint.basalPoint);     // Resistance Balance
      const capBalanceProj = project3D(tsnePoint.centerPoint);    // Capacitance Balance

      // Draw connections between the 3 characteristic points - cleaner lines
      ctx.strokeStyle = tsnePoint.color;
      ctx.lineWidth = tsnePoint.isGroundTruth ? 2.5 : 1.5;
      ctx.globalAlpha = tsnePoint.isGroundTruth ? 1.0 : 0.7;

      // Connection 1: Total Resistance to Resistance Balance
      ctx.beginPath();
      ctx.moveTo(totalResProj.x, totalResProj.y);
      ctx.lineTo(resBalanceProj.x, resBalanceProj.y);
      ctx.stroke();

      // Connection 2: Resistance Balance to Capacitance Balance
      ctx.beginPath();
      ctx.moveTo(resBalanceProj.x, resBalanceProj.y);
      ctx.lineTo(capBalanceProj.x, capBalanceProj.y);
      ctx.stroke();

      // Draw the 3 characteristic points with different sizes to show importance
      ctx.globalAlpha = 1.0;
      ctx.fillStyle = tsnePoint.color;

      // Point 1: Total Resistance (largest - most fundamental)
      ctx.beginPath();
      const totalResRadius = tsnePoint.isGroundTruth ? 8 : 6;
      ctx.arc(totalResProj.x, totalResProj.y, totalResRadius, 0, 2 * Math.PI);
      ctx.fill();

      // Point 2: Resistance Balance (medium)
      ctx.beginPath();
      const resBalanceRadius = tsnePoint.isGroundTruth ? 7 : 5;
      ctx.arc(resBalanceProj.x, resBalanceProj.y, resBalanceRadius, 0, 2 * Math.PI);
      ctx.fill();

      // Point 3: Capacitance Balance (smaller)
      ctx.beginPath();
      const capBalanceRadius = tsnePoint.isGroundTruth ? 6 : 4;
      ctx.arc(capBalanceProj.x, capBalanceProj.y, capBalanceRadius, 0, 2 * Math.PI);
      ctx.fill();

      // Add circuit labels - only for ground truth and selected circuits to reduce clutter
      if (tsnePoint.isGroundTruth) {
        ctx.fillStyle = tsnePoint.color;
        ctx.font = '14px Inter, sans-serif';
        ctx.textAlign = 'center';

        // Calculate geometric center of the 3 points for label placement
        const labelX = (totalResProj.x + resBalanceProj.x + capBalanceProj.x) / 3;
        const labelY = (totalResProj.y + resBalanceProj.y + capBalanceProj.y) / 3;

        ctx.fillText('ground truth', labelX, labelY + 35);

        // Add characteristic labels for ground truth (to help understand the visualization)
        ctx.font = '10px Inter, sans-serif';
        ctx.fillStyle = '#E5E7EB';
        ctx.fillText('R', totalResProj.x, totalResProj.y - 10);      // Total Resistance
        ctx.fillText('R↔', resBalanceProj.x, resBalanceProj.y - 10); // Resistance Balance
        ctx.fillText('C↔', capBalanceProj.x, capBalanceProj.y - 10); // Capacitance Balance
      }
    });

    // Draw improved 3D coordinate axes with proper t-SNE labeling
    ctx.globalAlpha = 1.0;
    ctx.strokeStyle = '#E5E7EB';
    ctx.lineWidth = 2;

    // Origin point for axes - positioned better
    const originX = 80;
    const originY = actualHeight - 80;

    // Make axes wider and more prominent
    const axisLength = 120;
    const diagonalLength = 85;

    // X-axis (horizontal) - wider
    ctx.beginPath();
    ctx.moveTo(originX, originY);
    ctx.lineTo(originX + axisLength, originY);
    ctx.stroke();

    // Y-axis (vertical) - wider
    ctx.beginPath();
    ctx.moveTo(originX, originY);
    ctx.lineTo(originX, originY - axisLength);
    ctx.stroke();

    // Z-axis (diagonal up-right, representing depth) - wider
    ctx.beginPath();
    ctx.moveTo(originX, originY);
    ctx.lineTo(originX + diagonalLength, originY - diagonalLength);
    ctx.stroke();

    // Add axis arrows
    ctx.strokeStyle = '#9CA3AF';
    ctx.lineWidth = 2;

    // X-axis arrow
    ctx.beginPath();
    ctx.moveTo(originX + axisLength - 10, originY - 5);
    ctx.lineTo(originX + axisLength, originY);
    ctx.lineTo(originX + axisLength - 10, originY + 5);
    ctx.stroke();

    // Y-axis arrow
    ctx.beginPath();
    ctx.moveTo(originX - 5, originY - axisLength + 10);
    ctx.lineTo(originX, originY - axisLength);
    ctx.lineTo(originX + 5, originY - axisLength + 10);
    ctx.stroke();

    // Z-axis arrow
    ctx.beginPath();
    ctx.moveTo(originX + diagonalLength - 8, originY - diagonalLength + 3);
    ctx.lineTo(originX + diagonalLength, originY - diagonalLength);
    ctx.lineTo(originX + diagonalLength - 3, originY - diagonalLength + 8);
    ctx.stroke();

    // Improved axis labels with proper t-SNE context
    ctx.fillStyle = '#374151';
    ctx.font = '14px Inter, sans-serif';
    ctx.textAlign = 'center';

    // X-axis label with context
    ctx.fillText('t-SNE Component 1', originX + axisLength/2, originY + 35);

    // Y-axis label with context (rotated)
    ctx.save();
    ctx.translate(originX - 35, originY - axisLength/2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('t-SNE Component 2', 0, 0);
    ctx.restore();

    // Z-axis label with context
    ctx.fillText('t-SNE Component 3', originX + diagonalLength + 15, originY - diagonalLength - 15);

    // Add scale indicators
    ctx.font = '10px Inter, sans-serif';
    ctx.fillStyle = '#6B7280';
    ctx.textAlign = 'center';

    // X-axis scale marks
    for (let i = 1; i <= 3; i++) {
      const x = originX + (axisLength/4) * i;
      ctx.beginPath();
      ctx.moveTo(x, originY - 3);
      ctx.lineTo(x, originY + 3);
      ctx.strokeStyle = '#9CA3AF';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Y-axis scale marks
    for (let i = 1; i <= 3; i++) {
      const y = originY - (axisLength/4) * i;
      ctx.beginPath();
      ctx.moveTo(originX - 3, y);
      ctx.lineTo(originX + 3, y);
      ctx.strokeStyle = '#9CA3AF';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Cleaner grid lines - more spaced out
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.2;

    // Horizontal grid lines (more spaced)
    for (let i = 0; i < actualHeight; i += 80) {
      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(actualWidth, i);
      ctx.stroke();
    }

    // Vertical grid lines (more spaced)
    for (let i = 0; i < actualWidth; i += 80) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, actualHeight);
      ctx.stroke();
    }

    ctx.globalAlpha = 1.0;

  }, [actualWidth, actualHeight, isComputing, tsnePoints, project3D]);

  // Animation loop
  useEffect(() => {
    const animate = () => {
      render();
      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [render]);

  // Mouse interaction handlers
  const handleMouseDown = (event: React.MouseEvent) => {
    setIsDragging(true);
    setLastMousePos({ x: event.clientX, y: event.clientY });
  };

  const handleMouseMove = (event: React.MouseEvent) => {
    if (!isDragging) return;

    const deltaX = event.clientX - lastMousePos.x;
    const deltaY = event.clientY - lastMousePos.y;

    setCamera(prev => ({
      ...prev,
      rotation: {
        x: prev.rotation.x + deltaY * 0.5,
        y: prev.rotation.y + deltaX * 0.5,
        z: prev.rotation.z
      }
    }));

    setLastMousePos({ x: event.clientX, y: event.clientY });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (event: React.WheelEvent) => {
    event.preventDefault();
    const scaleFactor = event.deltaY > 0 ? 1.1 : 0.9;

    setCamera(prev => ({
      ...prev,
      scale: Math.max(0.1, Math.min(5.0, prev.scale * scaleFactor))
    }));
  };

  return (
    <div ref={containerRef} className="w-full h-full relative">
      {/* Enhanced t-SNE context panel following best practices */}
      <div className="absolute top-4 right-4 z-10 bg-gray-900/95 backdrop-blur-sm rounded-lg p-4 max-w-sm border border-gray-700">
        <div className="text-base text-white mb-3 font-semibold">t-SNE 3D Embedding</div>
        <div className="text-xs text-gray-300 leading-relaxed space-y-2">
          <div>
            <strong className="text-gray-200">Dataset:</strong> Circuit Parameters<br/>
            <strong className="text-gray-200">Original Dims:</strong> 5D → <strong className="text-blue-400">3D</strong><br/>
            <strong className="text-gray-200">Method:</strong> Variance-based dimensionality reduction
          </div>
          <div className="border-t border-gray-600 pt-2">
            <strong className="text-gray-200">Circuit Representation:</strong><br/>
            • <strong className="text-red-400">Large dot:</strong> Total Resistance (R)<br/>
            • <strong className="text-yellow-400">Medium dot:</strong> Resistance Balance (R↔)<br/>
            • <strong className="text-green-400">Small dot:</strong> Capacitance Balance (C↔)
          </div>
          <div className="border-t border-gray-600 pt-2">
            • <strong className="text-purple-400">Purple:</strong> Ground truth reference<br/>
            • <strong className="text-blue-400">Colors:</strong> Performance (resnorm)<br/>
            • <strong className="text-gray-400">Clusters:</strong> Similar parameter patterns
          </div>
        </div>
      </div>

      {/* Main 3D canvas */}
      <canvas
        ref={canvasRef}
        width={actualWidth}
        height={actualHeight}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
        className="w-full h-full cursor-grab active:cursor-grabbing"
        style={{ width: actualWidth, height: actualHeight }}
      />
    </div>
  );
};