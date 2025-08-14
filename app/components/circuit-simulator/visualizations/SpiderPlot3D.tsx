"use client";

import React, { useEffect, useRef, useState } from 'react';
import { ModelSnapshot } from '../types';
import { PARAMETER_RANGES, DEFAULT_GRID_SIZE, faradToMicroFarad } from '../types/parameters';

// 3D Spider Plot Props
interface SpiderPlot3DProps {
  models: ModelSnapshot[];
  referenceModel?: ModelSnapshot | null;
  chromaEnabled?: boolean;
  width?: number;
  height?: number;
  showControls?: boolean;
  showAdvancedControls?: boolean; // New toggle for detailed info
  resnormScale?: number;
  gridSize?: number; // Number of points per parameter dimension
  onRotationChange?: (rotation: { x: number; y: number; z: number }) => void;
}

// Spectrum navigation state for traversing resnorm sections
interface SpectrumNavigation {
  enabled: boolean;
  currentSection: number; // Current resnorm section index
  totalSections: number; // Total number of sections
  modelsPerSection: number; // Models to show per section
  sectionRange: { min: number; max: number; }; // Current section's resnorm range
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

// Modern 3D Camera/View state with professional navigation
interface Camera3D {
  distance: number;
  fov: number;
  target: { x: number; y: number; z: number };
  position: { x: number; y: number; z: number };
  scale: number; // Global scale factor for all elements
  minDistance: number;
  maxDistance: number;
}

// Navigation modes for different interaction behaviors  
type NavigationMode = 'orbit' | 'pan' | 'zoom';

// Enhanced interaction state
interface InteractionState {
  isDragging: boolean;
  dragMode: NavigationMode;
  lastMousePos: { x: number; y: number };
  isShiftPressed: boolean;
  isCtrlPressed: boolean;
}

export const SpiderPlot3D: React.FC<SpiderPlot3DProps> = ({
  models,
  referenceModel,
  chromaEnabled = true,
  width = 800,
  height = 600,
  showControls = true,
  showAdvancedControls = false,
  resnormScale = 1.0,
  gridSize = DEFAULT_GRID_SIZE,
  onRotationChange,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [rotation, setRotation] = useState<Rotation3D>({ x: -30, y: 45, z: 0 });
  const [camera, setCamera] = useState<Camera3D>({
    distance: 12,
    fov: 60,
    target: { x: 0, y: 0, z: 2.5 },
    position: { x: 0, y: 0, z: 0 },
    scale: 1.0,
    minDistance: 2,
    maxDistance: 50
  });
  const [interaction, setInteraction] = useState<InteractionState>({
    isDragging: false,
    dragMode: 'orbit',
    lastMousePos: { x: 0, y: 0 },
    isShiftPressed: false,
    isCtrlPressed: false
  });
  
  // Spectrum navigation state
  const [spectrumNav, setSpectrumNav] = useState<SpectrumNavigation>({
    enabled: false,
    currentSection: 0,
    totalSections: 10, // Default to 10 sections
    modelsPerSection: 100, // Show 100 models per section
    sectionRange: { min: 0, max: 1 }
  });
  
  // Mini-map state for resnorm spectrum visualization
  const [minimapState, setMinimapState] = useState<MinimapState>({
    totalRange: { min: 0, max: 1 },
    visibleRange: { min: 0, max: 1 },
    dataDistribution: []
  });

  // Filter models based on spectrum navigation
  const filteredModels = React.useMemo(() => {
    if (!models.length || !spectrumNav.enabled) return models;
    
    // Sort models by resnorm for consistent sectioning
    const sortedModels = [...models].sort((a, b) => (a.resnorm || 0) - (b.resnorm || 0));
    
    const startIndex = spectrumNav.currentSection * spectrumNav.modelsPerSection;
    const endIndex = Math.min(startIndex + spectrumNav.modelsPerSection, sortedModels.length);
    
    const sectionModels = sortedModels.slice(startIndex, endIndex);
    
    // Update section range
    if (sectionModels.length > 0) {
      const minResnorm = Math.min(...sectionModels.map(m => m.resnorm || 0));
      const maxResnorm = Math.max(...sectionModels.map(m => m.resnorm || 0));
      setSpectrumNav(prev => ({
        ...prev,
        sectionRange: { min: minResnorm, max: maxResnorm },
        totalSections: Math.ceil(sortedModels.length / prev.modelsPerSection)
      }));
    }
    
    return sectionModels;
  }, [models, spectrumNav.enabled, spectrumNav.currentSection, spectrumNav.modelsPerSection]);
  
  // Convert models to 3D radar polygons
  const convert3DPolygons = React.useMemo(() => {
    if (!filteredModels.length) return [];

    // Parameter definitions for radar chart (shared with reference model)
    const params = ['Rsh', 'Ra', 'Ca', 'Rb', 'Cb'];
    const paramRanges = {
      Rsh: PARAMETER_RANGES.Rsh,
      Ra: PARAMETER_RANGES.Ra,
      Rb: PARAMETER_RANGES.Rb,
      Ca: PARAMETER_RANGES.Ca,
      Cb: PARAMETER_RANGES.Cb
    };

    // Calculate resnorm range for Z-axis scaling
    const resnorms = filteredModels.map(m => m.resnorm || 0).filter(r => r > 0);
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

    return filteredModels.map(model => {
      const resnorm = model.resnorm || 0;
      
      // Calculate Z position based on resnorm (FIXED: lower resnorm = lower Z, closer to ground truth)
      const normalizedResnorm = resnormRange > 0 
        ? (resnorm - minResnorm) / resnormRange 
        : 0;
      // CORRECT: Better models (lower resnorm) should be closer to ground truth (lower Z)
      const zHeight = normalizedResnorm * 5.0 * resnormScale; // Scale Z from 0 to 5 (lower resnorm = lower Z)

      // Determine color based on resnorm quartiles (FIXED: lower resnorm = better = green)
      const sortedResnorms = [...resnorms].sort((a, b) => a - b);
      const q25 = sortedResnorms[Math.floor(sortedResnorms.length * 0.25)];
      const q50 = sortedResnorms[Math.floor(sortedResnorms.length * 0.50)];
      const q75 = sortedResnorms[Math.floor(sortedResnorms.length * 0.75)];

      let color = '#ef4444'; // Red (worst quartile - highest resnorm)
      if (resnorm <= q25) color = '#22c55e'; // Green (best quartile - lowest resnorm)
      else if (resnorm <= q50) color = '#f59e0b'; // Yellow (second quartile)  
      else if (resnorm <= q75) color = '#f97316'; // Orange (third quartile)

      // Calculate radar polygon vertices
      const vertices = params.map((param, i) => {
        const value = model.parameters[param as keyof typeof model.parameters] as number;
        const range = paramRanges[param as keyof typeof paramRanges];
        
        // Normalize parameter value to 0-1 range (logarithmic)
        const logMin = Math.log10(range.min);
        const logMax = Math.log10(range.max);
        const logValue = Math.log10(Math.max(range.min, Math.min(range.max, value)));
        const normalizedValue = (logValue - logMin) / (logMax - logMin);
        
        // Calculate radar position (pentagon) - restored original bounds
        const angle = (i * 2 * Math.PI) / params.length - Math.PI / 2;
        const radius = normalizedValue * 2.0; // Back to original 2.0 for undistorted interpretation
        
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
  }, [filteredModels, chromaEnabled, resnormScale]);

  // Restored original projection with navigation improvements
  const project3D = React.useCallback((point: Point3D): { x: number; y: number; visible: boolean; depth: number } => {
    // Apply rotation
    const rad = (deg: number) => (deg * Math.PI) / 180;
    const cos = Math.cos;
    const sin = Math.sin;

    // Rotation matrices
    const rx = rad(rotation.x);
    const ry = rad(rotation.y);
    const rz = rad(rotation.z);

    // Apply rotations (order: Z, Y, X) - back to original method
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

    // Apply camera distance and target offset
    x -= camera.target.x;
    y -= camera.target.y;
    z += camera.distance - camera.target.z;

    // Original-style projection to avoid distortion
    if (z <= 0.1) return { x: 0, y: 0, visible: false, depth: 0 };

    // Restored conservative perspective calculation
    const baseScale = Math.min(width, height) * 0.12; // Back to original scale
    const perspectiveFactor = Math.max(0.7, 1.0 - (camera.distance - 8) * 0.02);
    const scale = baseScale * perspectiveFactor;
    
    const projX = width / 2 + x * scale;
    const projY = height / 2 - y * scale;

    // Original visibility bounds
    const margin = 50;
    return { 
      x: projX, 
      y: projY, 
      visible: projX >= -margin && projX <= width + margin && projY >= -margin && projY <= height + margin,
      depth: z
    };
  }, [rotation, camera, width, height]);

  // Draw 3D radar grid with dynamic scaling
  const draw3DRadarGrid = React.useCallback((ctx: CanvasRenderingContext2D) => {
    const params = ['Rsh', 'Ra', 'Ca', 'Rb', 'Cb'];
    
    ctx.save();
    ctx.strokeStyle = camera.scale > 0.5 ? '#505050' : '#707070'; // Adapt grid visibility to scale
    ctx.lineWidth = Math.max(1, camera.scale);
    ctx.globalAlpha = 0.6;

    // Draw base pentagon outline with original radius
    const baseRadius = 2.0; // Restored original size
    const baseVertices = params.map((_, i) => {
      const angle = (i * 2 * Math.PI) / params.length - Math.PI / 2;
      return project3D({
        x: Math.cos(angle) * baseRadius,
        y: Math.sin(angle) * baseRadius,
        z: 0,
        color: '#000000',
        opacity: 1,
        model: null as any // eslint-disable-line @typescript-eslint/no-explicit-any
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

    // Draw radial axes with improved visibility
    ctx.strokeStyle = '#606060';
    ctx.globalAlpha = Math.min(0.6, camera.scale * 0.4); // Scale-aware transparency
    params.forEach((_, i) => {
      const angle = (i * 2 * Math.PI) / params.length - Math.PI / 2;
      const start = project3D({ x: 0, y: 0, z: 0, color: '#000000', opacity: 1, model: null as any }); // eslint-disable-line @typescript-eslint/no-explicit-any
      const end = project3D({ 
        x: Math.cos(angle) * baseRadius, 
        y: Math.sin(angle) * baseRadius, 
        z: 0,
        color: '#000000',
        opacity: 1,
        model: null as any // eslint-disable-line @typescript-eslint/no-explicit-any
      });
      
      if (start.visible && end.visible) {
        ctx.beginPath();
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
        ctx.stroke();
      }
    });

    ctx.restore();
  }, [project3D, camera.scale]);

  // Draw 3D polygon wireframe with depth lines and adaptive styling
  const draw3DPolygon = React.useCallback((ctx: CanvasRenderingContext2D, polygon: Polygon3D) => {
    const projectedVertices = polygon.vertices
      .map((vertex: Point3D) => ({ vertex, projected: project3D(vertex) }))
      .filter((p) => p.projected.visible);

    if (projectedVertices.length < 3) return;

    ctx.save();
    
    // Calculate average depth for adaptive styling
    const avgDepth = projectedVertices.reduce((sum, p) => sum + p.projected.depth, 0) / projectedVertices.length;
    const depthFactor = Math.max(0.3, Math.min(1.0, 10 / avgDepth)); // Closer = more visible
    
    // Draw vertical lines from base to polygon (depth indicators) with adaptive styling
    ctx.strokeStyle = polygon.color;
    ctx.lineWidth = Math.max(0.5, depthFactor * camera.scale);
    ctx.globalAlpha = Math.max(0.15, polygon.opacity * 0.4 * depthFactor);
    ctx.setLineDash([3 * camera.scale, 3 * camera.scale]);
    
    projectedVertices.forEach(({ vertex, projected }) => {
      const basePoint = project3D({ x: vertex.x, y: vertex.y, z: 0, color: '#000000', opacity: 1, model: null as any }); // eslint-disable-line @typescript-eslint/no-explicit-any
      
      if (projected.visible && basePoint.visible) {
        ctx.beginPath();
        ctx.moveTo(basePoint.x, basePoint.y);
        ctx.lineTo(projected.x, projected.y);
        ctx.stroke();
      }
    });
    
    // Draw wireframe polygon with adaptive line width
    ctx.globalAlpha = Math.max(0.5, polygon.opacity * depthFactor);
    ctx.lineWidth = Math.max(1, 2 * depthFactor * camera.scale);
    ctx.setLineDash([]);

    const projectedOnly = projectedVertices.map(p => p.projected);
    ctx.beginPath();
    ctx.moveTo(projectedOnly[0].x, projectedOnly[0].y);
    for (let i = 1; i < projectedOnly.length; i++) {
      ctx.lineTo(projectedOnly[i].x, projectedOnly[i].y);
    }
    ctx.closePath();
    ctx.stroke();

    // Draw vertices as adaptive-sized dots
    ctx.fillStyle = polygon.color;
    ctx.globalAlpha = Math.min(1.0, (polygon.opacity + 0.4) * depthFactor);
    const dotSize = Math.max(1.5, 3 * depthFactor * camera.scale);
    projectedOnly.forEach(vertex => {
      ctx.beginPath();
      ctx.arc(vertex.x, vertex.y, dotSize, 0, 2 * Math.PI);
      ctx.fill();
    });

    ctx.restore();
  }, [project3D, camera.scale]);

  // Draw enhanced 3D radar labels with value markers
  const draw3DRadarLabels = React.useCallback((ctx: CanvasRenderingContext2D) => {
    const params = [
      { name: 'Rsh (Ω)', desc: 'Shunt', range: PARAMETER_RANGES.Rsh },
      { name: 'Ra (Ω)', desc: 'Apical R', range: PARAMETER_RANGES.Ra },
      { name: 'Ca (µF)', desc: 'Apical C', range: { min: faradToMicroFarad(PARAMETER_RANGES.Ca.min), max: faradToMicroFarad(PARAMETER_RANGES.Ca.max) } },
      { name: 'Rb (Ω)', desc: 'Basal R', range: PARAMETER_RANGES.Rb },
      { name: 'Cb (µF)', desc: 'Basal C', range: { min: faradToMicroFarad(PARAMETER_RANGES.Cb.min), max: faradToMicroFarad(PARAMETER_RANGES.Cb.max) } }
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
        z: 0.5,
        color: '#000000',
        opacity: 1,
        model: null as any // eslint-disable-line @typescript-eslint/no-explicit-any
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
      
      // Draw parameter name with adaptive font
      ctx.fillStyle = '#FFFFFF';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(param.name, labelX, labelY - 6 * camera.scale);
      
      // Draw value markers at intervals
      const numMarkers = gridSize || 5;
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
  }, [project3D, camera.scale]);

  // Draw modern 3D orientation cube with better visibility
  const drawOrientationCube = React.useCallback((ctx: CanvasRenderingContext2D) => {
    const cubeSize = Math.max(50, 70 * camera.scale);
    const cubeX = width - cubeSize - 20;
    const cubeY = height - cubeSize - 80; // Move to bottom-right
    
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
  }, [width, height, rotation, camera.scale]);

  // Draw dynamic resnorm Z-axis that scales with 3D transformations
  const drawResnormAxis = React.useCallback((ctx: CanvasRenderingContext2D) => {
    if (!models.length) return;
    
    // Calculate resnorm range
    const resnorms = models.map(m => m.resnorm || 0).filter(r => r > 0);
    const minResnorm = Math.min(...resnorms);
    const maxResnorm = Math.max(...resnorms);
    
    ctx.save();
    
    // Dynamic axis positioning relative to camera and scale
    const axisOffset = 5.0 / camera.scale; // Scale-aware positioning
    const axisX = axisOffset;
    const axisY = axisOffset;
    const zAxisHeight = 5.0 * resnormScale; // Respect resnormScale prop
    
    // Draw Z-axis line with improved scaling
    const axisStart = project3D({ x: axisX, y: axisY, z: 0, color: '#000000', opacity: 1, model: null as any }); // eslint-disable-line @typescript-eslint/no-explicit-any
    const axisEnd = project3D({ x: axisX, y: axisY, z: zAxisHeight, color: '#000000', opacity: 1, model: null as any }); // eslint-disable-line @typescript-eslint/no-explicit-any
    
    if (axisStart.visible && axisEnd.visible) {
      // Dynamic axis line with scale-aware styling
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = Math.max(2, 3 * camera.scale);
      ctx.globalAlpha = Math.min(0.9, 0.5 + camera.scale * 0.4);
      ctx.beginPath();
      ctx.moveTo(axisStart.x, axisStart.y);
      ctx.lineTo(axisEnd.x, axisEnd.y);
      ctx.stroke();
      
      // Draw axis ticks and labels with scale-aware formatting
      const fontSize = Math.max(9, 11 * camera.scale);
      ctx.font = `bold ${fontSize}px Arial`;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      
      const numTicks = Math.max(4, Math.min(8, Math.floor(6 * camera.scale))); // Adaptive tick count
      for (let i = 0; i < numTicks; i++) {
        const z = (i / (numTicks - 1)) * zAxisHeight;
        const normalizedZ = i / (numTicks - 1);
        const resnormValue = minResnorm + (normalizedZ * (maxResnorm - minResnorm));
        
        const tickSize = 0.2 / camera.scale; // Scale-aware tick size
        const tickPoint = project3D({ x: axisX, y: axisY, z: z, color: '#000000', opacity: 1, model: null as any }); // eslint-disable-line @typescript-eslint/no-explicit-any
        const tickEndPoint = project3D({ x: axisX + tickSize, y: axisY, z: z, color: '#000000', opacity: 1, model: null as any }); // eslint-disable-line @typescript-eslint/no-explicit-any
        
        if (tickPoint.visible && tickEndPoint.visible) {
          // Draw scale-aware tick mark
          ctx.strokeStyle = '#FFFFFF';
          ctx.lineWidth = Math.max(1, 2 * camera.scale);
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
          
          // Draw label with scale-aware background
          const labelWidth = ctx.measureText(labelText).width;
          const labelOffset = 8 * camera.scale;
          const labelHeight = fontSize + 4;
          ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
          ctx.fillRect(tickEndPoint.x + labelOffset, tickEndPoint.y - labelHeight/2, labelWidth + 6, labelHeight);
          
          ctx.fillStyle = '#FFFFFF';
          ctx.fillText(labelText, tickEndPoint.x + labelOffset + 3, tickEndPoint.y);
        }
      }
      
      // Draw scale-aware axis title
      const titleOffset = 0.6 / camera.scale;
      const titlePoint = project3D({ x: axisX + titleOffset, y: axisY, z: zAxisHeight / 2, color: '#000000', opacity: 1, model: null as any }); // eslint-disable-line @typescript-eslint/no-explicit-any
      if (titlePoint.visible) {
        ctx.save();
        ctx.translate(titlePoint.x, titlePoint.y);
        ctx.rotate(-Math.PI / 2);
        
        const titleFontSize = Math.max(12, 14 * camera.scale);
        const titleWidth = 60 * camera.scale;
        const titleHeight = titleFontSize + 8;
        
        // Title background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(-titleWidth/2, -titleHeight/2, titleWidth, titleHeight);
        
        // Title text
        ctx.fillStyle = '#FFFFFF';
        ctx.font = `bold ${titleFontSize}px Arial`;
        ctx.textAlign = 'center';
        ctx.fillText('Resnorm', 0, 0);
        ctx.restore();
      }
    }
    
    ctx.restore();
  }, [models, project3D, camera.scale, resnormScale]);

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
      // @ts-expect-error - Complex type incompatibility with Polygon3D vertices
      draw3DPolygon(ctx, polygon);
    });

    // Render reference model if provided
    if (referenceModel) {
      const params = ['Rsh', 'Ra', 'Ca', 'Rb', 'Cb'];
      const paramRanges = {
        Rsh: PARAMETER_RANGES.Rsh,
        Ra: PARAMETER_RANGES.Ra,
        Rb: PARAMETER_RANGES.Rb,
        Ca: PARAMETER_RANGES.Ca,
        Cb: PARAMETER_RANGES.Cb
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
        const radius = normalizedValue * 2.0; // Match original radar bounds
        
        return {
          x: Math.cos(angle) * radius,
          y: Math.sin(angle) * radius,
          z: 0 // Reference at base level
        };
      });

      const projectedRefVertices = refVertices
        .map(vertex => project3D({
          ...vertex,
          color: '#FFFFFF',
          opacity: 1,
          model: referenceModel
        }))
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

  // Professional 3D navigation handlers (Blender/Onshape-style)
  const handleMouseDown = (e: React.MouseEvent) => {
    // Determine interaction mode based on mouse button and modifiers
    let dragMode: NavigationMode = 'orbit';
    if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
      dragMode = 'pan';
    } else if (e.button === 2 || (e.button === 0 && e.ctrlKey)) {
      dragMode = 'zoom';
    }

    setInteraction({
      isDragging: true,
      dragMode,
      lastMousePos: { x: e.clientX, y: e.clientY },
      isShiftPressed: e.shiftKey,
      isCtrlPressed: e.ctrlKey
    });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!interaction.isDragging) return;

    const deltaX = e.clientX - interaction.lastMousePos.x;
    const deltaY = e.clientY - interaction.lastMousePos.y;
    const sensitivity = 0.5;

    if (interaction.dragMode === 'orbit') {
      // Orbit around target
      const newRotation = {
        x: Math.max(-89, Math.min(89, rotation.x + deltaY * sensitivity)),
        y: (rotation.y + deltaX * sensitivity) % 360,
        z: rotation.z
      };
      setRotation(newRotation);
      if (onRotationChange) {
        onRotationChange(newRotation);
      }
    } else if (interaction.dragMode === 'pan') {
      // Pan the target (world space movement)
      const panSensitivity = 0.005 * camera.distance;
      setCamera(prev => ({
        ...prev,
        target: {
          x: prev.target.x - deltaX * panSensitivity,
          y: prev.target.y + deltaY * panSensitivity,
          z: prev.target.z
        }
      }));
    } else if (interaction.dragMode === 'zoom') {
      // Zoom by dragging
      const zoomSensitivity = 0.01;
      const zoomDelta = 1 + (deltaY * zoomSensitivity);
      setCamera(prev => ({
        ...prev,
        distance: Math.max(prev.minDistance, Math.min(prev.maxDistance, prev.distance * zoomDelta))
      }));
    }

    setInteraction(prev => ({
      ...prev,
      lastMousePos: { x: e.clientX, y: e.clientY }
    }));
  };

  const handleMouseUp = () => {
    setInteraction(prev => ({ ...prev, isDragging: false }));
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const zoomSpeed = 0.1;
    const zoomFactor = e.deltaY > 0 ? (1 + zoomSpeed) : (1 - zoomSpeed);
    
    if (e.shiftKey) {
      // Shift+scroll for scale adjustment
      setCamera(prev => ({
        ...prev,
        scale: Math.max(0.1, Math.min(10, prev.scale * zoomFactor))
      }));
    } else {
      // Normal scroll for distance zoom
      setCamera(prev => ({
        ...prev,
        distance: Math.max(prev.minDistance, Math.min(prev.maxDistance, prev.distance * zoomFactor))
      }));
    }
  };

  // Keyboard shortcuts for navigation
  const handleKeyDown = React.useCallback((e: KeyboardEvent) => {
    if (e.target !== canvasRef.current) return;
    
    const moveSpeed = 0.5;
    const rotateSpeed = 5;
    
    switch (e.key.toLowerCase()) {
      case 'w': // Move forward
      case 'arrowup':
        setCamera(prev => ({ ...prev, distance: Math.max(prev.minDistance, prev.distance - moveSpeed) }));
        break;
      case 's': // Move back  
      case 'arrowdown':
        setCamera(prev => ({ ...prev, distance: Math.min(prev.maxDistance, prev.distance + moveSpeed) }));
        break;
      case 'a': // Rotate left
      case 'arrowleft':
        setRotation(prev => ({ ...prev, y: (prev.y - rotateSpeed) % 360 }));
        break;
      case 'd': // Rotate right
      case 'arrowright':
        setRotation(prev => ({ ...prev, y: (prev.y + rotateSpeed) % 360 }));
        break;
      case 'q': // Rotate up
        setRotation(prev => ({ ...prev, x: Math.max(-89, prev.x - rotateSpeed) }));
        break;
      case 'e': // Rotate down
        setRotation(prev => ({ ...prev, x: Math.min(89, prev.x + rotateSpeed) }));
        break;
      case 'r': // Reset view
        setRotation({ x: -30, y: 45, z: 0 });
        setCamera(prev => ({ ...prev, distance: 12, target: { x: 0, y: 0, z: 2.5 }, scale: 1.0 }));
        break;
      case 'f': // Focus/fit view
        setCamera(prev => ({ ...prev, target: { x: 0, y: 0, z: 2.5 }, distance: 12, scale: 1.0 }));
        break;
    }
  }, []);

  // Context menu prevention for right-click pan
  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
  };

  // Keyboard event listeners
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Make canvas focusable for keyboard events
    canvas.tabIndex = 0;
    canvas.addEventListener('keydown', handleKeyDown);
    
    return () => {
      canvas.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleKeyDown]);

  // Render effect
  useEffect(() => {
    render3D();
  }, [render3D]);

  return (
    <div className="relative bg-black rounded-lg overflow-hidden border border-neutral-700">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className={`${
          interaction.isDragging 
            ? interaction.dragMode === 'pan' ? 'cursor-move' 
              : interaction.dragMode === 'zoom' ? 'cursor-ns-resize'
              : 'cursor-grabbing'
            : 'cursor-grab'
        } focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50`}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
        onContextMenu={handleContextMenu}
        onFocus={() => canvasRef.current?.focus()}
      />
      
      {/* Navigation Toolbar */}
      <div className="absolute top-4 left-4 bg-gray-900 bg-opacity-80 rounded p-2">
        <div className="flex flex-col space-y-2">
          {/* View Controls */}
          <div className="flex space-x-1">
            <button
              onClick={() => {
                setRotation({ x: -30, y: 45, z: 0 });
                setCamera(prev => ({ ...prev, distance: 12, target: { x: 0, y: 0, z: 2.5 }, scale: 1.0 }));
              }}
              className="px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-white text-xs transition-colors"
              title="Reset View (R)"
            >
              Reset
            </button>
            <button
              onClick={() => setCamera(prev => ({ ...prev, target: { x: 0, y: 0, z: 2.5 }, distance: 12, scale: 1.0 }))}
              className="px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-white text-xs transition-colors"
              title="Focus View (F)"
            >
              Focus
            </button>
            <button
              onClick={() => setSpectrumNav(prev => ({ ...prev, enabled: !prev.enabled, currentSection: 0 }))}
              className={`px-2 py-1 rounded text-white text-xs transition-colors ${
                spectrumNav.enabled ? 'bg-blue-600 hover:bg-blue-700' : 'bg-gray-700 hover:bg-gray-600'
              }`}
              title="Navigate Spectrum Sections"
            >
              {spectrumNav.enabled ? 'Exit Nav' : 'Navigate'}
            </button>
          </div>
          
          {/* Spectrum Navigation Controls */}
          {spectrumNav.enabled && (
            <div className="flex flex-col space-y-1 border-t border-gray-600 pt-2">
              {/* Section Slider */}
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setSpectrumNav(prev => ({ ...prev, currentSection: Math.max(0, prev.currentSection - 1) }))}
                  disabled={spectrumNav.currentSection === 0}
                  className="px-1 py-0.5 bg-gray-600 hover:bg-gray-500 disabled:bg-gray-800 disabled:cursor-not-allowed rounded text-white text-xs"
                >
                  ◀
                </button>
                <input
                  type="range"
                  min="0"
                  max={Math.max(0, spectrumNav.totalSections - 1)}
                  value={spectrumNav.currentSection}
                  onChange={(e) => setSpectrumNav(prev => ({ ...prev, currentSection: parseInt(e.target.value) }))}
                  className="flex-1 h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                  style={{
                    background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${(spectrumNav.currentSection / Math.max(1, spectrumNav.totalSections - 1)) * 100}%, #4b5563 ${(spectrumNav.currentSection / Math.max(1, spectrumNav.totalSections - 1)) * 100}%, #4b5563 100%)`
                  }}
                />
                <button
                  onClick={() => setSpectrumNav(prev => ({ ...prev, currentSection: Math.min(prev.totalSections - 1, prev.currentSection + 1) }))}
                  disabled={spectrumNav.currentSection >= spectrumNav.totalSections - 1}
                  className="px-1 py-0.5 bg-gray-600 hover:bg-gray-500 disabled:bg-gray-800 disabled:cursor-not-allowed rounded text-white text-xs"
                >
                  ▶
                </button>
              </div>
              
              {/* Section Info */}
              <div className="text-xs text-gray-300">
                <div>Section {spectrumNav.currentSection + 1} / {spectrumNav.totalSections}</div>
                <div className="text-gray-400">
                  Resnorm: {spectrumNav.sectionRange.min.toFixed(3)} - {spectrumNav.sectionRange.max.toFixed(3)}
                </div>
                <div className="text-gray-400">
                  Showing {filteredModels.length} / {models.length} models
                </div>
              </div>
              
              {/* Models per section control */}
              <div className="flex items-center space-x-2">
                <span className="text-xs text-gray-400">Per section:</span>
                <select
                  value={spectrumNav.modelsPerSection}
                  onChange={(e) => setSpectrumNav(prev => ({ ...prev, modelsPerSection: parseInt(e.target.value), currentSection: 0 }))}
                  className="bg-gray-700 text-white text-xs rounded px-1 py-0.5"
                >
                  <option value={50}>50</option>
                  <option value={100}>100</option>
                  <option value={200}>200</option>
                  <option value={500}>500</option>
                </select>
              </div>
            </div>
          )}
        </div>
      </div>
      
      {showControls && (
        <div className="absolute bottom-4 right-4">
          <div className="bg-black bg-opacity-80 p-3 rounded text-white text-xs">
            <div className="space-y-2">
              <div>Drag: Rotate • Shift+Drag: Pan • Scroll: Zoom</div>
              {showAdvancedControls && (
                <>
                  <div className="border-t border-gray-600 pt-2 space-y-1">
                    <div>Pitch: {rotation.x.toFixed(1)}° • Yaw: {rotation.y.toFixed(1)}°</div>
                    <div>Distance: {camera.distance.toFixed(1)} • Scale: {camera.scale.toFixed(2)}x</div>
                  </div>
                </>
              )}
              <button
                onClick={() => {/* Toggle advanced controls via parent component */}}
                className="text-xs text-blue-400 hover:text-blue-300 underline"
              >
                {showAdvancedControls ? 'Hide' : 'Show'} Details
              </button>
            </div>
          </div>
        </div>
      )}

      {filteredModels.length === 0 && models.length > 0 && spectrumNav.enabled && (
        <div className="absolute inset-0 flex items-center justify-center text-gray-400 bg-black bg-opacity-50">
          <div className="text-center p-8 bg-gray-900 bg-opacity-80 rounded-lg">
            <div className="text-lg mb-2 text-white">No models in this section</div>
            <div className="text-sm mb-4">Try adjusting the section or models per section</div>
          </div>
        </div>
      )}
      
      {models.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center text-gray-400 bg-black bg-opacity-50">
          <div className="text-center p-8 bg-gray-900 bg-opacity-80 rounded-lg">
            <div className="text-lg mb-2 text-white">No 3D data available</div>
            <div className="text-sm mb-4">Generate grid results to see 3D visualization</div>
            <div className="text-xs text-gray-500">
              Professional 3D navigation + spectrum traversal ready when data is loaded
            </div>
          </div>
        </div>
      )}
    </div>
  );
};