import React, { useEffect, useRef, useState } from 'react';
import { ModelSnapshot } from '../types';

// 3D Spider Plot Props
interface SpiderPlot3DProps {
  models: ModelSnapshot[];
  referenceModel?: ModelSnapshot | null;
  chromaEnabled?: boolean;
  width?: number;
  height?: number;
  showControls?: boolean;
  resnormScale?: number;
  onRotationChange?: (rotation: { x: number; y: number; z: number }) => void;
}

// 3D Point representing a model in parameter space with resnorm as Z-axis
interface Point3D {
  x: number; // Parameter dimension 1 (e.g., tau_a = Ra * Ca)
  y: number; // Parameter dimension 2 (e.g., tau_b = Rb * Cb) 
  z: number; // Resnorm value
  color: string;
  opacity: number;
  model: ModelSnapshot;
}

// 3D Polygon representing a radar chart model
interface Polygon3D {
  vertices: Point3D[];
  color: string;
  opacity: number;
  model: ModelSnapshot;
  zHeight: number;
}

// Mini-map state for resnorm spectrum visualization
interface MinimapState {
  totalRange: { min: number; max: number };
  visibleRange: { min: number; max: number };
  dataDistribution: number[];
}

// 3D Rotation state
interface Rotation3D {
  x: number; // Pitch
  y: number; // Yaw  
  z: number; // Roll
}

// Camera/View state
interface Camera3D {
  distance: number;
  fov: number;
  target: { x: number; y: number; z: number };
}

export const SpiderPlot3D: React.FC<SpiderPlot3DProps> = ({
  models,
  referenceModel,
  chromaEnabled = true,
  width = 800,
  height = 600,
  showControls = true,
  resnormScale = 1.0,
  onRotationChange,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [rotation, setRotation] = useState<Rotation3D>({ x: -30, y: 45, z: 0 });
  const [camera, setCamera] = useState<Camera3D>({
    distance: 8,
    fov: 45,
    target: { x: 0, y: 0, z: 2.5 }
  });
  const [isDragging, setIsDragging] = useState(false);
  const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });
  
  // Mini-map state for resnorm spectrum visualization
  const [minimapState, setMinimapState] = useState<MinimapState>({
    totalRange: { min: 0, max: 1 },
    visibleRange: { min: 0, max: 1 },
    dataDistribution: []
  });

  // Convert models to 3D radar polygons
  const convert3DPolygons = React.useMemo(() => {
    if (!models.length) return [];

    // Parameter definitions for radar chart
    const params = ['Rs', 'Ra', 'Ca', 'Rb', 'Cb'];
    const paramRanges = {
      Rs: { min: 10, max: 10000 },
      Ra: { min: 10, max: 10000 },
      Rb: { min: 10, max: 10000 },
      Ca: { min: 0.1e-6, max: 50e-6 },
      Cb: { min: 0.1e-6, max: 50e-6 }
    };

    // Calculate resnorm range for Z-axis scaling
    const resnorms = models.map(m => m.resnorm || 0).filter(r => r > 0);
    const minResnorm = Math.min(...resnorms);
    const maxResnorm = Math.max(...resnorms);
    const resnormRange = maxResnorm - minResnorm;
    
    // Update minimap state with current data range
    const visibleResnorms = resnorms.slice(0, 1000); // Limit for performance
    const currentMinVisible = Math.min(...visibleResnorms);
    const currentMaxVisible = Math.max(...visibleResnorms);
    
    // Create histogram for data distribution
    const histogramBins = 20;
    const distribution = new Array(histogramBins).fill(0);
    resnorms.forEach(r => {
      const binIndex = Math.floor(((r - minResnorm) / resnormRange) * (histogramBins - 1));
      distribution[Math.max(0, Math.min(histogramBins - 1, binIndex))]++;
    });
    
    // Update minimap state
    setMinimapState({
      totalRange: { min: minResnorm, max: maxResnorm },
      visibleRange: { min: currentMinVisible, max: currentMaxVisible },
      dataDistribution: distribution
    });

    return models.map(model => {
      const resnorm = model.resnorm || 0;
      
      // Calculate Z position based on resnorm (fixed: lower resnorm = lower Z, closer to viewer)
      const normalizedResnorm = resnormRange > 0 
        ? (resnorm - minResnorm) / resnormRange 
        : 0;
      const zHeight = normalizedResnorm * 5.0 * resnormScale; // Scale Z from 0 to 5 * scale (lower resnorm = lower Z)

      // Determine color based on resnorm quartiles
      const q25 = resnorms[Math.floor(resnorms.length * 0.25)];
      const q50 = resnorms[Math.floor(resnorms.length * 0.50)];
      const q75 = resnorms[Math.floor(resnorms.length * 0.75)];

      let color = '#22c55e'; // Green (best quartile)
      if (resnorm > q75) color = '#ef4444'; // Red (worst quartile)
      else if (resnorm > q50) color = '#f97316'; // Orange (third quartile)  
      else if (resnorm > q25) color = '#f59e0b'; // Yellow (second quartile)

      // Calculate radar polygon vertices
      const vertices = params.map((param, i) => {
        const value = model.parameters[param as keyof typeof model.parameters] as number;
        const range = paramRanges[param as keyof typeof paramRanges];
        
        // Normalize parameter value to 0-1 range (logarithmic)
        const logMin = Math.log10(range.min);
        const logMax = Math.log10(range.max);
        const logValue = Math.log10(Math.max(range.min, Math.min(range.max, value)));
        const normalizedValue = (logValue - logMin) / (logMax - logMin);
        
        // Calculate radar position (pentagon)
        const angle = (i * 2 * Math.PI) / params.length - Math.PI / 2;
        const radius = normalizedValue * 2.0; // Scale to radius of 2
        
        return {
          x: Math.cos(angle) * radius,
          y: Math.sin(angle) * radius,
          z: zHeight
        };
      });

      return {
        vertices,
        color: chromaEnabled ? color : '#CCCCCC',
        opacity: Math.max(0.2, 1.0 - normalizedResnorm * 0.7),
        model,
        zHeight
      };
    });
  }, [models, chromaEnabled, resnormScale]);

  // 3D to 2D projection with orthographic-like perspective to reduce distortion
  const project3D = React.useCallback((point: Point3D): { x: number; y: number; visible: boolean } => {
    // Apply rotation
    const rad = (deg: number) => (deg * Math.PI) / 180;
    const cos = Math.cos;
    const sin = Math.sin;

    // Rotation matrices
    const rx = rad(rotation.x);
    const ry = rad(rotation.y);
    const rz = rad(rotation.z);

    // Apply rotations (order: Z, Y, X)
    let x = point.x;
    let y = point.y;
    let z = point.z;

    // Z rotation
    const x1 = x * cos(rz) - y * sin(rz);
    const y1 = x * sin(rz) + y * cos(rz);
    x = x1; y = y1;

    // Y rotation  
    const x2 = x * cos(ry) + z * sin(ry);
    const z2 = -x * sin(ry) + z * cos(ry);
    x = x2; z = z2;

    // X rotation
    const y3 = y * cos(rx) - z * sin(rx);
    const z3 = y * sin(rx) + z * cos(rx);
    y = y3; z = z3;

    // Apply camera distance
    z += camera.distance;

    // Orthographic-style projection to reduce fisheye distortion
    if (z <= 0.1) return { x: 0, y: 0, visible: false };

    // Use a more conservative perspective calculation
    // Reduce the FOV effect by using a smaller divisor
    const baseScale = Math.min(width, height) * 0.12; // Fixed scale factor
    const perspectiveFactor = Math.max(0.7, 1.0 - (camera.distance - 8) * 0.02); // Subtle perspective
    const scale = baseScale * perspectiveFactor;
    
    const projX = width / 2 + x * scale;
    const projY = height / 2 - y * scale;

    // Reasonable visibility bounds
    const margin = 50;
    return { 
      x: projX, 
      y: projY, 
      visible: projX >= -margin && projX <= width + margin && projY >= -margin && projY <= height + margin
    };
  }, [rotation, camera, width, height]);

  // Draw 3D radar grid
  const draw3DRadarGrid = React.useCallback((ctx: CanvasRenderingContext2D) => {
    const params = ['Rs', 'Ra', 'Ca', 'Rb', 'Cb'];
    
    ctx.save();
    ctx.strokeStyle = '#505050';
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.6;

    // Draw base pentagon outline only
    const baseRadius = 2.0;
    const baseVertices = params.map((_, i) => {
      const angle = (i * 2 * Math.PI) / params.length - Math.PI / 2;
      return project3D({
        x: Math.cos(angle) * baseRadius,
        y: Math.sin(angle) * baseRadius,
        z: 0
      });
    }).filter(p => p.visible);

    if (baseVertices.length === params.length) {
      ctx.beginPath();
      ctx.moveTo(baseVertices[0].x, baseVertices[0].y);
      for (let i = 1; i < baseVertices.length; i++) {
        ctx.lineTo(baseVertices[i].x, baseVertices[i].y);
      }
      ctx.closePath();
      ctx.stroke();
    }

    // Draw radial axes (only at base level)
    ctx.strokeStyle = '#606060';
    ctx.globalAlpha = 0.4;
    params.forEach((_, i) => {
      const angle = (i * 2 * Math.PI) / params.length - Math.PI / 2;
      const start = project3D({ x: 0, y: 0, z: 0 });
      const end = project3D({ 
        x: Math.cos(angle) * baseRadius, 
        y: Math.sin(angle) * baseRadius, 
        z: 0 
      });
      
      if (start.visible && end.visible) {
        ctx.beginPath();
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
        ctx.stroke();
      }
    });

    ctx.restore();
  }, [project3D]);

  // Draw 3D polygon wireframe with depth lines
  const draw3DPolygon = React.useCallback((ctx: CanvasRenderingContext2D, polygon: Polygon3D) => {
    const projectedVertices = polygon.vertices
      .map((vertex: Point3D) => project3D(vertex))
      .filter((p) => p.visible);

    if (projectedVertices.length < 3) return;

    ctx.save();
    
    // Draw vertical lines from base to polygon (depth indicators)
    ctx.strokeStyle = polygon.color;
    ctx.lineWidth = 1;
    ctx.globalAlpha = Math.max(0.2, polygon.opacity * 0.5);
    ctx.setLineDash([2, 2]);
    
    polygon.vertices.forEach(vertex => {
      const topPoint = project3D(vertex);
      const basePoint = project3D({ x: vertex.x, y: vertex.y, z: 0 });
      
      if (topPoint.visible && basePoint.visible) {
        ctx.beginPath();
        ctx.moveTo(basePoint.x, basePoint.y);
        ctx.lineTo(topPoint.x, topPoint.y);
        ctx.stroke();
      }
    });
    
    // Draw wireframe polygon
    ctx.globalAlpha = Math.max(0.6, polygon.opacity);
    ctx.lineWidth = 1.8;
    ctx.setLineDash([]);

    ctx.beginPath();
    ctx.moveTo(projectedVertices[0].x, projectedVertices[0].y);
    for (let i = 1; i < projectedVertices.length; i++) {
      ctx.lineTo(projectedVertices[i].x, projectedVertices[i].y);
    }
    ctx.closePath();
    ctx.stroke();

    // Draw vertices as dots for better visibility
    ctx.fillStyle = polygon.color;
    ctx.globalAlpha = Math.min(1.0, polygon.opacity + 0.4);
    projectedVertices.forEach(vertex => {
      ctx.beginPath();
      ctx.arc(vertex.x, vertex.y, 2.5, 0, 2 * Math.PI);
      ctx.fill();
    });

    ctx.restore();
  }, [project3D]);

  // Draw enhanced 3D radar labels with value markers
  const draw3DRadarLabels = React.useCallback((ctx: CanvasRenderingContext2D) => {
    const params = [
      { name: 'Rs (Ω)', desc: 'Shunt', range: { min: 10, max: 10000 } },
      { name: 'Ra (Ω)', desc: 'Apical R', range: { min: 10, max: 10000 } },
      { name: 'Ca (µF)', desc: 'Apical C', range: { min: 0.1, max: 50 } },
      { name: 'Rb (Ω)', desc: 'Basal R', range: { min: 10, max: 10000 } },
      { name: 'Cb (µF)', desc: 'Basal C', range: { min: 0.1, max: 50 } }
    ];
    
    ctx.save();
    
    // Draw all parameter labels with enhanced styling and value markers
    params.forEach((param, i) => {
      const angle = (i * 2 * Math.PI) / params.length - Math.PI / 2;
      
      // Position labels further out
      const labelRadius = 2.8;
      const label3DPoint = project3D({
        x: Math.cos(angle) * labelRadius,
        y: Math.sin(angle) * labelRadius,
        z: 0.5
      });
      
      const labelX = label3DPoint.x;
      const labelY = label3DPoint.y;
      
      // Draw enhanced label background
      ctx.font = 'bold 13px Arial';
      const textWidth = ctx.measureText(param.name).width;
      const backgroundWidth = Math.max(textWidth + 16, 80);
      
      ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
      ctx.fillRect(labelX - backgroundWidth/2, labelY - 15, backgroundWidth, 35);
      ctx.strokeStyle = '#444444';
      ctx.lineWidth = 1;
      ctx.strokeRect(labelX - backgroundWidth/2, labelY - 15, backgroundWidth, 35);
      
      // Draw parameter name (bolder)
      ctx.fillStyle = '#FFFFFF';
      ctx.font = 'bold 13px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(param.name, labelX, labelY - 5);
      
      // Draw value markers at intervals
      const numMarkers = 3;
      const markerTexts = [];
      
      for (let j = 0; j < numMarkers; j++) {
        const ratio = j / (numMarkers - 1);
        let value: number;
        
        if (param.name.includes('Ω')) {
          // Logarithmic scale for resistance
          const logMin = Math.log10(param.range.min);
          const logMax = Math.log10(param.range.max);
          value = Math.pow(10, logMin + ratio * (logMax - logMin));
          markerTexts.push(value >= 1000 ? `${(value/1000).toFixed(1)}k` : value.toFixed(0));
        } else {
          // Linear scale for capacitance
          value = param.range.min + ratio * (param.range.max - param.range.min);
          markerTexts.push(value < 1 ? value.toFixed(1) : value.toFixed(0));
        }
      }
      
      // Draw value markers
      ctx.fillStyle = '#CCCCCC';
      ctx.font = '9px Arial';
      const markerText = markerTexts.join(' | ');
      ctx.fillText(markerText, labelX, labelY + 10);
    });
    
    ctx.restore();
  }, [project3D]);

  // Draw 3D orientation cube
  const drawOrientationCube = React.useCallback((ctx: CanvasRenderingContext2D) => {
    const cubeSize = 60;
    const cubeX = width / 2 - cubeSize / 2;
    const cubeY = 15;
    
    ctx.save();
    
    // Draw cube background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.fillRect(cubeX - 10, cubeY - 5, cubeSize + 20, cubeSize + 20);
    ctx.strokeStyle = '#666666';
    ctx.lineWidth = 1;
    ctx.strokeRect(cubeX - 10, cubeY - 5, cubeSize + 20, cubeSize + 20);
    
    // Define cube vertices in local space
    const cubeVertices = [
      { x: -1, y: -1, z: -1 }, // 0
      { x:  1, y: -1, z: -1 }, // 1
      { x:  1, y:  1, z: -1 }, // 2
      { x: -1, y:  1, z: -1 }, // 3
      { x: -1, y: -1, z:  1 }, // 4
      { x:  1, y: -1, z:  1 }, // 5
      { x:  1, y:  1, z:  1 }, // 6
      { x: -1, y:  1, z:  1 }  // 7
    ];
    
    // Project cube vertices with current rotation
    const projectedCubeVertices = cubeVertices.map(vertex => {
      // Apply same rotation as main view
      const rad = (deg: number) => (deg * Math.PI) / 180;
      const cos = Math.cos;
      const sin = Math.sin;
      
      const rx = rad(rotation.x);
      const ry = rad(rotation.y);
      const rz = rad(rotation.z);
      
      let x = vertex.x;
      let y = vertex.y;
      let z = vertex.z;
      
      // Z rotation
      const x1 = x * cos(rz) - y * sin(rz);
      const y1 = x * sin(rz) + y * cos(rz);
      x = x1; y = y1;
      
      // Y rotation
      const x2 = x * cos(ry) + z * sin(ry);
      const z2 = -x * sin(ry) + z * cos(ry);
      x = x2; z = z2;
      
      // X rotation
      const y3 = y * cos(rx) - z * sin(rx);
      const z3 = y * sin(rx) + z * cos(rx);
      y = y3; z = z3;
      
      // Scale and position for cube display
      const scale = 15;
      return {
        x: cubeX + cubeSize/2 + x * scale,
        y: cubeY + cubeSize/2 - y * scale,
        z: z
      };
    });
    
    // Define cube faces with colors
    const faces = [
      { indices: [0, 1, 2, 3], color: '#FF6B6B', label: '-Z' }, // Front face
      { indices: [4, 7, 6, 5], color: '#4ECDC4', label: '+Z' }, // Back face
      { indices: [0, 4, 5, 1], color: '#45B7D1', label: '-Y' }, // Bottom face
      { indices: [3, 2, 6, 7], color: '#96CEB4', label: '+Y' }, // Top face
      { indices: [0, 3, 7, 4], color: '#FFEAA7', label: '-X' }, // Left face
      { indices: [1, 5, 6, 2], color: '#DDA0DD', label: '+X' }  // Right face
    ];
    
    // Sort faces by average Z depth
    const sortedFaces = faces.map(face => ({
      ...face,
      avgZ: face.indices.reduce((sum, i) => sum + projectedCubeVertices[i].z, 0) / face.indices.length
    })).sort((a, b) => a.avgZ - b.avgZ);
    
    // Draw faces
    sortedFaces.forEach(face => {
      const vertices = face.indices.map(i => projectedCubeVertices[i]);
      
      ctx.fillStyle = face.color;
      ctx.globalAlpha = 0.7;
      ctx.beginPath();
      ctx.moveTo(vertices[0].x, vertices[0].y);
      for (let i = 1; i < vertices.length; i++) {
        ctx.lineTo(vertices[i].x, vertices[i].y);
      }
      ctx.closePath();
      ctx.fill();
      
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 1;
      ctx.globalAlpha = 1.0;
      ctx.stroke();
    });
    
    // Draw axis lines
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 2;
    ctx.globalAlpha = 1.0;
    
    // X axis (red)
    const xStart = { x: cubeX + cubeSize/2, y: cubeY + cubeSize/2, z: 0 };
    ctx.strokeStyle = '#FF4444';
    ctx.beginPath();
    ctx.moveTo(xStart.x, xStart.y);
    ctx.lineTo(xStart.x + 20, xStart.y);
    ctx.stroke();
    
    // Y axis (green)
    ctx.strokeStyle = '#44FF44';
    ctx.beginPath();
    ctx.moveTo(xStart.x, xStart.y);
    ctx.lineTo(xStart.x, xStart.y - 20);
    ctx.stroke();
    
    // Z axis (blue)
    ctx.strokeStyle = '#4444FF';
    ctx.beginPath();
    ctx.moveTo(xStart.x, xStart.y);
    ctx.lineTo(xStart.x + 10, xStart.y + 10);
    ctx.stroke();
    
    // Draw axis labels
    ctx.fillStyle = '#FFFFFF';
    ctx.font = '10px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('X', xStart.x + 25, xStart.y + 3);
    ctx.fillText('Y', xStart.x - 3, xStart.y - 25);
    ctx.fillText('Z', xStart.x + 15, xStart.y + 20);
    
    ctx.restore();
  }, [width, rotation]);

  // Draw clean resnorm Z-axis with proper formatting
  const drawResnormAxis = React.useCallback((ctx: CanvasRenderingContext2D) => {
    if (!models.length) return;
    
    // Calculate resnorm range
    const resnorms = models.map(m => m.resnorm || 0).filter(r => r > 0);
    const minResnorm = Math.min(...resnorms);
    const maxResnorm = Math.max(...resnorms);
    
    ctx.save();
    
    // Position axis away from main chart
    const axisX = 3.2;
    const axisY = 3.2;
    
    // Draw Z-axis line with gradient
    const axisStart = project3D({ x: axisX, y: axisY, z: 0 });
    const axisEnd = project3D({ x: axisX, y: axisY, z: 5 });
    
    if (axisStart.visible && axisEnd.visible) {
      // Main axis line
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 3;
      ctx.globalAlpha = 0.9;
      ctx.beginPath();
      ctx.moveTo(axisStart.x, axisStart.y);
      ctx.lineTo(axisEnd.x, axisEnd.y);
      ctx.stroke();
      
      // Draw axis ticks and labels with better formatting
      ctx.font = 'bold 11px Arial';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      
      const numTicks = 6;
      for (let i = 0; i < numTicks; i++) {
        const z = (i / (numTicks - 1)) * 5;
        const normalizedZ = i / (numTicks - 1);
        const resnormValue = maxResnorm - (normalizedZ * (maxResnorm - minResnorm));
        
        const tickPoint = project3D({ x: axisX, y: axisY, z: z });
        const tickEndPoint = project3D({ x: axisX + 0.15, y: axisY, z: z });
        
        if (tickPoint.visible && tickEndPoint.visible) {
          // Draw tick mark
          ctx.strokeStyle = '#FFFFFF';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(tickPoint.x, tickPoint.y);
          ctx.lineTo(tickEndPoint.x, tickEndPoint.y);
          ctx.stroke();
          
          // Format value label with proper scientific notation
          let labelText: string;
          if (resnormValue >= 1) {
            labelText = resnormValue.toFixed(2);
          } else if (resnormValue >= 0.01) {
            labelText = resnormValue.toFixed(3);
          } else {
            labelText = resnormValue.toExponential(2);
          }
          
          // Draw label with background
          const labelWidth = ctx.measureText(labelText).width;
          ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
          ctx.fillRect(tickEndPoint.x + 8, tickEndPoint.y - 8, labelWidth + 6, 16);
          
          ctx.fillStyle = '#FFFFFF';
          ctx.fillText(labelText, tickEndPoint.x + 11, tickEndPoint.y);
        }
      }
      
      // Draw axis title with background
      const titlePoint = project3D({ x: axisX + 0.4, y: axisY, z: 2.5 });
      if (titlePoint.visible) {
        ctx.save();
        ctx.translate(titlePoint.x, titlePoint.y);
        ctx.rotate(-Math.PI / 2);
        
        // Title background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(-25, -10, 50, 20);
        
        // Title text
        ctx.fillStyle = '#FFFFFF';
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Resnorm', 0, 0);
        ctx.restore();
      }
    }
    
    ctx.restore();
  }, [models, project3D]);

  // Draw resnorm spectrum mini-map
  const drawResnormMinimap = React.useCallback((ctx: CanvasRenderingContext2D) => {
    const mapWidth = 180;
    const mapHeight = 120;
    const mapX = width - mapWidth - 15;
    const mapY = 15;
    
    ctx.save();
    
    // Draw minimap background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.fillRect(mapX, mapY, mapWidth, mapHeight);
    ctx.strokeStyle = '#666666';
    ctx.lineWidth = 1;
    ctx.strokeRect(mapX, mapY, mapWidth, mapHeight);
    
    // Draw title
    ctx.fillStyle = '#FFFFFF';
    ctx.font = '11px Arial';
    ctx.textAlign = 'left';
    ctx.fillText('Resnorm Spectrum', mapX + 5, mapY + 15);
    
    // Draw data distribution histogram
    const histHeight = 60;
    const histY = mapY + 25;
    const barWidth = (mapWidth - 20) / minimapState.dataDistribution.length;
    
    const maxCount = Math.max(...minimapState.dataDistribution, 1);
    
    ctx.fillStyle = '#4A90E2';
    minimapState.dataDistribution.forEach((count, i) => {
      const barHeight = (count / maxCount) * histHeight;
      const barX = mapX + 10 + i * barWidth;
      const barY = histY + histHeight - barHeight;
      
      ctx.fillRect(barX, barY, barWidth - 1, barHeight);
    });
    
    // Draw visible range indicator
    const totalRange = minimapState.totalRange.max - minimapState.totalRange.min;
    if (totalRange > 0) {
      const visibleStart = ((minimapState.visibleRange.min - minimapState.totalRange.min) / totalRange) * (mapWidth - 20);
      const visibleWidth = ((minimapState.visibleRange.max - minimapState.visibleRange.min) / totalRange) * (mapWidth - 20);
      
      // Draw visible range rectangle
      ctx.strokeStyle = '#FF6B35';
      ctx.lineWidth = 2;
      ctx.strokeRect(mapX + 10 + visibleStart, histY, visibleWidth, histHeight);
      
      // Draw range values
      ctx.fillStyle = '#FFFFFF';
      ctx.font = '9px Arial';
      ctx.textAlign = 'left';
      ctx.fillText(`Min: ${minimapState.totalRange.min.toExponential(2)}`, mapX + 5, mapY + mapHeight - 25);
      ctx.fillText(`Max: ${minimapState.totalRange.max.toExponential(2)}`, mapX + 5, mapY + mapHeight - 15);
      
      ctx.fillStyle = '#FF6B35';
      ctx.fillText(`Visible: ${minimapState.visibleRange.min.toExponential(2)} - ${minimapState.visibleRange.max.toExponential(2)}`, mapX + 5, mapY + mapHeight - 5);
    }
    
    // Draw current render info
    ctx.fillStyle = '#CCCCCC';
    ctx.font = '9px Arial';
    ctx.textAlign = 'right';
    ctx.fillText(`Models: ${models.length}`, mapX + mapWidth - 5, mapY + 15);
    
    ctx.restore();
  }, [width, models.length, minimapState]);

  // Draw procedural rendering status and context
  const drawProceduralContext = React.useCallback((ctx: CanvasRenderingContext2D) => {
    const statusWidth = 250;
    const statusHeight = 40;
    const statusX = 15;
    const statusY = height - statusHeight - 15;
    
    ctx.save();
    
    // Draw status background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
    ctx.fillRect(statusX, statusY, statusWidth, statusHeight);
    ctx.strokeStyle = '#555555';
    ctx.lineWidth = 1;
    ctx.strokeRect(statusX, statusY, statusWidth, statusHeight);
    
    // Determine rendering status
    const totalPossibleModels = 100000; // Theoretical maximum
    const renderingMode = models.length > 10000 ? 'Procedural' : 'Direct';
    const renderPercentage = Math.min(100, (models.length / totalPossibleModels) * 100);
    
    // Draw status text
    ctx.fillStyle = '#FFFFFF';
    ctx.font = '11px Arial';
    ctx.textAlign = 'left';
    ctx.fillText(`Rendering: ${renderingMode} Mode | ${models.length.toLocaleString()} models (${renderPercentage.toFixed(1)}%)`, statusX + 8, statusY + 15);
    
    // Draw performance indicator
    const perfColor = models.length > 50000 ? '#FF6B35' : models.length > 10000 ? '#FFB84D' : '#4CAF50';
    const perfText = models.length > 50000 ? 'High Load' : models.length > 10000 ? 'Medium Load' : 'Optimal';
    
    ctx.fillStyle = perfColor;
    ctx.fillText(`Performance: ${perfText}`, statusX + 8, statusY + 30);
    
    // Draw data quality indicator
    const dataQuality = minimapState.visibleRange.max - minimapState.visibleRange.min;
    const qualityText = dataQuality > (minimapState.totalRange.max - minimapState.totalRange.min) * 0.5 ? 'Full Range' : 'Partial Range';
    
    ctx.fillStyle = '#CCCCCC';
    ctx.textAlign = 'right';
    ctx.fillText(`Data: ${qualityText}`, statusX + statusWidth - 8, statusY + 30);
    
    ctx.restore();
  }, [height, models.length, minimapState]);

  // Render 3D visualization
  const render3D = React.useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Set canvas background
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);

    // Draw 3D radar chart grid
    draw3DRadarGrid(ctx);

    // Sort polygons by average Z-depth for proper rendering order
    const sortedPolygons = [...convert3DPolygons].sort((a, b) => {
      return b.zHeight - a.zHeight; // Front to back for proper transparency
    });

    // Render each 3D radar polygon
    sortedPolygons.forEach(polygon => {
      draw3DPolygon(ctx, polygon);
    });

    // Render reference model if provided
    if (referenceModel) {
      const params = ['Rs', 'Ra', 'Ca', 'Rb', 'Cb'];
      const paramRanges = {
        Rs: { min: 10, max: 10000 },
        Ra: { min: 10, max: 10000 },
        Rb: { min: 10, max: 10000 },
        Ca: { min: 0.1e-6, max: 50e-6 },
        Cb: { min: 0.1e-6, max: 50e-6 }
      };

      // Create reference polygon at base level (z=0)
      const refVertices = params.map((param, i) => {
        const value = referenceModel.parameters[param as keyof typeof referenceModel.parameters] as number;
        const range = paramRanges[param as keyof typeof paramRanges];
        
        const logMin = Math.log10(range.min);
        const logMax = Math.log10(range.max);
        const logValue = Math.log10(Math.max(range.min, Math.min(range.max, value)));
        const normalizedValue = (logValue - logMin) / (logMax - logMin);
        
        const angle = (i * 2 * Math.PI) / params.length - Math.PI / 2;
        const radius = normalizedValue * 2.0;
        
        return {
          x: Math.cos(angle) * radius,
          y: Math.sin(angle) * radius,
          z: 0 // Reference at base level
        };
      });

      const projectedRefVertices = refVertices
        .map(vertex => project3D(vertex))
        .filter(p => p.visible);

      if (projectedRefVertices.length === params.length) {
        ctx.save();
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 3]);
        ctx.globalAlpha = 0.9;

        ctx.beginPath();
        ctx.moveTo(projectedRefVertices[0].x, projectedRefVertices[0].y);
        for (let i = 1; i < projectedRefVertices.length; i++) {
          ctx.lineTo(projectedRefVertices[i].x, projectedRefVertices[i].y);
        }
        ctx.closePath();
        ctx.stroke();

        // Draw reference vertices as larger dots
        ctx.fillStyle = '#FFFFFF';
        ctx.setLineDash([]);
        ctx.globalAlpha = 1.0;
        projectedRefVertices.forEach(vertex => {
          ctx.beginPath();
          ctx.arc(vertex.x, vertex.y, 3, 0, 2 * Math.PI);
          ctx.fill();
        });

        ctx.restore();
      }
    }

    // Draw parameter labels and resnorm axis
    draw3DRadarLabels(ctx);
    drawResnormAxis(ctx);
    
    // Draw minimap overlay
    drawResnormMinimap(ctx);
    
    // Draw procedural rendering context
    drawProceduralContext(ctx);
    
    // Draw orientation cube
    drawOrientationCube(ctx);

  }, [convert3DPolygons, project3D, referenceModel, width, height, draw3DRadarGrid, draw3DPolygon, draw3DRadarLabels, drawResnormAxis, drawResnormMinimap, drawProceduralContext, drawOrientationCube]);

  // Mouse interaction handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setLastMousePos({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;

    const deltaX = e.clientX - lastMousePos.x;
    const deltaY = e.clientY - lastMousePos.y;

    const newRotation = {
      x: Math.max(-90, Math.min(90, rotation.x + deltaY * 0.5)),
      y: rotation.y + deltaX * 0.5,
      z: rotation.z
    };

    setRotation(newRotation);
    setLastMousePos({ x: e.clientX, y: e.clientY });

    if (onRotationChange) {
      onRotationChange(newRotation);
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 1.15 : 0.85;
    setCamera(prev => ({
      ...prev,
      distance: Math.max(3, Math.min(25, prev.distance * delta))
    }));
  };

  // Render effect
  useEffect(() => {
    render3D();
  }, [render3D]);

  return (
    <div className="relative bg-black rounded-lg overflow-hidden">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="cursor-grab active:cursor-grabbing"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      />
      
      {showControls && (
        <div className="absolute bottom-4 right-4 bg-black bg-opacity-75 p-3 rounded text-white text-sm">
          <div className="space-y-2">
            <div>Drag: Rotate view</div>
            <div>Scroll: Zoom in/out</div>
            <div className="border-t border-gray-600 pt-2">
              <div>Pitch: {rotation.x.toFixed(1)}°</div>
              <div>Yaw: {rotation.y.toFixed(1)}°</div>
              <div>Roll: {rotation.z.toFixed(1)}°</div>
              <div>Zoom: {(1/camera.distance * 10).toFixed(1)}x</div>
            </div>
          </div>
        </div>
      )}

      {models.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center text-gray-400">
          <div className="text-center">
            <div className="text-lg mb-2">No 3D data available</div>
            <div className="text-sm">Generate grid results to see 3D visualization</div>
          </div>
        </div>
      )}
    </div>
  );
};