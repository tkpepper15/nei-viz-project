"use client";

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { ModelSnapshot } from '../types';
import { PARAMETER_RANGES, DEFAULT_GRID_SIZE, faradToMicroFarad } from '../types/parameters';

// 3D Spider Plot Props
interface SpiderPlot3DProps {
  models: ModelSnapshot[];
  referenceModel?: ModelSnapshot | null;
  width?: number;
  height?: number;
  showControls?: boolean;
  showAdvancedControls?: boolean; // New toggle for detailed info
  resnormSpread?: number;
  gridSize?: number; // Number of points per parameter dimension
  onRotationChange?: (rotation: { x: number; y: number; z: number }) => void;
  useResnormCenter?: boolean; // New: rotate around resnorm center instead of grid front
  responsive?: boolean; // New: enable responsive sizing
  showLabels?: boolean; // New: toggle for parameter labels, values, and resnorm markers
  resnormRange?: { min: number; max: number } | null; // New: filter models by resnorm range
  onModelTag?: (model: ModelSnapshot, tag: string) => void; // Callback for tagging models
  taggedModels?: Map<string, string>; // Map of model id to tag name
}

// Navigation removed - now handled by resnorm distribution section

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

// Dynamic parameter configuration based on CircuitParameters type
const getCircuitParameters = () => {
  const parameterKeys = ['Rsh', 'Ra', 'Ca', 'Rb', 'Cb'] as const;
  return parameterKeys.map(key => {
    const range = PARAMETER_RANGES[key];
    const isCapacitance = key === 'Ca' || key === 'Cb';
    return {
      key,
      name: isCapacitance ? `${key} (µF)` : `${key} (Ω)`,
      desc: key === 'Rsh' ? 'Shunt' : 
            key === 'Ra' ? 'Apical R' :
            key === 'Ca' ? 'Apical C' :
            key === 'Rb' ? 'Basal R' : 'Basal C',
      range: isCapacitance ? 
        { min: faradToMicroFarad(range.min), max: faradToMicroFarad(range.max) } : 
        range
    };
  });
};

export const SpiderPlot3D: React.FC<SpiderPlot3DProps> = ({
  models,
  referenceModel,
  width = 800,
  height = 600,
  showControls = true,
  showAdvancedControls: _showAdvancedControls = false, // eslint-disable-line @typescript-eslint/no-unused-vars
  resnormSpread = 1.0,
  gridSize = DEFAULT_GRID_SIZE,
  onRotationChange,
  useResnormCenter = false,
  responsive = true,
  showLabels = true, // eslint-disable-line @typescript-eslint/no-unused-vars
  resnormRange = null,
  onModelTag,
  taggedModels = new Map(),
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Responsive sizing state
  const [actualWidth, setActualWidth] = useState(width);
  const [actualHeight, setActualHeight] = useState(height);
  
  // Component initialization state to prevent laggy first render
  const [isComponentReady, setIsComponentReady] = useState(false);
  
  // Initialize component after first mount to prevent initial lag
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsComponentReady(true);
    }, 50); // Small delay to allow DOM to settle
    
    return () => clearTimeout(timer);
  }, []);

  // Initialize user session for database integration
  // Removed database initialization - using parent callback system instead
  
  // Responsive sizing effect
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

    return () => {
      resizeObserver.disconnect();
    };
  }, [responsive, width, height]);
  
  // Get dynamic circuit parameters configuration
  const circuitParams = React.useMemo(() => getCircuitParameters(), []);
  const paramKeys = React.useMemo(() => circuitParams.map(p => p.key), [circuitParams]);

  // Utility function to check if a point is inside a polygon
  const isPointInPolygon = useCallback((x: number, y: number, polygon: {x: number; y: number}[]): boolean => {
    if (polygon.length < 3) return false;
    
    let inside = false;
    let j = polygon.length - 1;
    
    for (let i = 0; i < polygon.length; i++) {
      if (((polygon[i].y > y) !== (polygon[j].y > y)) &&
          (x < (polygon[j].x - polygon[i].x) * (y - polygon[i].y) / (polygon[j].y - polygon[i].y) + polygon[i].x)) {
        inside = !inside;
      }
      j = i;
    }
    
    return inside;
  }, []);

  
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
  
  // Model selection and tagging state
  const [hoveredModel, setHoveredModel] = useState<ModelSnapshot | null>(null);
  const [showTagDialog, setShowTagDialog] = useState<{model: ModelSnapshot; x: number; y: number} | null>(null);
  
  // Tagging state
  const [isTagging, setIsTagging] = useState<boolean>(false);
  

  // Filter models based on spectrum navigation and resnorm range
  const filteredModels = React.useMemo(() => {
    let filteredByRange = models;
    
    // Apply resnorm range filtering first if provided
    if (resnormRange) {
      filteredByRange = models.filter(model => {
        const modelResnorm = model.resnorm || 0;
        return modelResnorm >= resnormRange.min && modelResnorm <= resnormRange.max;
      });
    }
    
    // Navigation removed - resnorm distribution section handles granular selection
    return filteredByRange;
  }, [models, resnormRange]);
  
  // Convert models to 3D radar polygons
  const convert3DPolygons = React.useMemo(() => {
    if (!filteredModels.length) return [];

    // Parameter definitions for radar chart (shared with reference model)
    const params = paramKeys;
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
    

    return filteredModels.map(model => {
      const resnorm = model.resnorm || 0;
      
      // Calculate Z position based on resnorm (FIXED: lower resnorm = lower Z, closer to ground truth)
      const normalizedResnorm = resnormRange > 0 
        ? (resnorm - minResnorm) / resnormRange 
        : 0;
      // CORRECT: Better models (lower resnorm) should be closer to ground truth (lower Z)
      const zHeight = normalizedResnorm * 5.0 * resnormSpread; // Scale Z from 0 to 5 (lower resnorm = lower Z)

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
        color: color,
        opacity: Math.max(0.2, 1.0 - normalizedResnorm * 0.7),
        model,
        zHeight
      };
    });
  }, [filteredModels, resnormSpread, paramKeys]);


  // Calculate resnorm center for rotation pivot
  const resnormCenter = React.useMemo(() => {
    if (!useResnormCenter || !filteredModels.length) {
      return { x: 0, y: 0, z: 2.5 }; // Default center
    }

    // Calculate the weighted center of all resnorm points
    const resnorms = filteredModels.map(m => m.resnorm || 0).filter(r => r > 0);
    if (resnorms.length === 0) return { x: 0, y: 0, z: 2.5 };

    const minResnorm = Math.min(...resnorms);
    const maxResnorm = Math.max(...resnorms);
    const resnormRange = maxResnorm - minResnorm;
    
    // Calculate center Z based on median resnorm
    const sortedResnorms = [...resnorms].sort((a, b) => a - b);
    const medianResnorm = sortedResnorms[Math.floor(sortedResnorms.length / 2)];
    const normalizedMedian = resnormRange > 0 ? (medianResnorm - minResnorm) / resnormRange : 0;
    const centerZ = normalizedMedian * 5.0 * resnormSpread;

    return { x: 0, y: 0, z: centerZ };
  }, [useResnormCenter, filteredModels, resnormSpread]);

  // Enhanced projection with configurable rotation center
  const project3D = React.useCallback((point: Point3D): { x: number; y: number; visible: boolean; depth: number } => {
    // Apply rotation
    const rad = (deg: number) => (deg * Math.PI) / 180;
    const cos = Math.cos;
    const sin = Math.sin;

    // Rotation matrices
    const rx = rad(rotation.x);
    const ry = rad(rotation.y);
    const rz = rad(rotation.z);

    // Use configurable rotation center
    const rotationCenter = useResnormCenter ? resnormCenter : { x: 0, y: 0, z: 0 };

    // Translate to rotation center
    let x = point.x - rotationCenter.x;
    let y = point.y - rotationCenter.y;
    let z = point.z - rotationCenter.z;

    // Apply rotations (order: Z, Y, X)
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

    // Translate back from rotation center
    x += rotationCenter.x;
    y += rotationCenter.y;
    z += rotationCenter.z;

    // Apply camera distance and target offset
    x -= camera.target.x;
    y -= camera.target.y;
    z += camera.distance - camera.target.z;

    // Projection with depth testing
    if (z <= 0.1) return { x: 0, y: 0, visible: false, depth: 0 };

    // Conservative perspective calculation
    const baseScale = Math.min(actualWidth, actualHeight) * 0.12; // Use actual dimensions
    const perspectiveFactor = Math.max(0.7, 1.0 - (camera.distance - 8) * 0.02);
    const scale = baseScale * perspectiveFactor;
    
    const projX = actualWidth / 2 + x * scale; // Use actual dimensions
    const projY = actualHeight / 2 - y * scale; // Use actual dimensions

    // Visibility bounds with actual dimensions
    const margin = 50;
    return { 
      x: projX, 
      y: projY, 
      visible: projX >= -margin && projX <= actualWidth + margin && projY >= -margin && projY <= actualHeight + margin,
      depth: z
    };
  }, [rotation, camera, actualWidth, actualHeight, useResnormCenter, resnormCenter]);

  // Function to find which model polygon contains a click point
  const findModelAtPoint = useCallback((clickX: number, clickY: number): ModelSnapshot | null => {
    // Check polygons from front to back (reverse order since we render back to front)
    const sortedPolygons = [...convert3DPolygons].sort((a, b) => b.zHeight - a.zHeight);
    
    for (const polygon of sortedPolygons) {
      const projectedVertices = polygon.vertices
        .map(vertex => project3D(vertex))
        .filter(p => p.visible);
      
      if (projectedVertices.length >= 3) {
        const polygonPoints = projectedVertices.map(p => ({ x: p.x, y: p.y }));
        
        if (isPointInPolygon(clickX, clickY, polygonPoints)) {
          return polygon.model;
        }
      }
    }
    
    return null;
  }, [convert3DPolygons, project3D, isPointInPolygon]);

  // Draw 3D radar grid with dynamic scaling
  const draw3DRadarGrid = React.useCallback((ctx: CanvasRenderingContext2D) => {
    const params = paramKeys;
    
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
  }, [project3D, camera.scale, paramKeys]);

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
    
    // Check if this model is hovered or tagged
    const isHovered = hoveredModel?.id === polygon.model.id;
    const isTagged = taggedModels.has(polygon.model.id);
    const tagName = taggedModels.get(polygon.model.id);
    
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
    
    // Draw wireframe polygon with adaptive line width and highlighting
    const baseOpacity = Math.max(0.5, polygon.opacity * depthFactor);
    const baseLineWidth = Math.max(1, 2 * depthFactor * camera.scale);
    
    // Enhanced styling for hovered/tagged models
    if (isHovered) {
      ctx.globalAlpha = Math.min(1.0, baseOpacity + 0.3);
      ctx.lineWidth = baseLineWidth * 2;
      ctx.strokeStyle = '#FFD700'; // Gold highlight for hover
    } else if (isTagged) {
      ctx.globalAlpha = Math.min(1.0, baseOpacity + 0.2);
      ctx.lineWidth = baseLineWidth * 1.5;
      ctx.strokeStyle = '#FF6B9D'; // Pink highlight for tagged
    } else {
      ctx.globalAlpha = baseOpacity;
      ctx.lineWidth = baseLineWidth;
      ctx.strokeStyle = polygon.color;
    }
    ctx.setLineDash([]);

    const projectedOnly = projectedVertices.map(p => p.projected);
    ctx.beginPath();
    ctx.moveTo(projectedOnly[0].x, projectedOnly[0].y);
    for (let i = 1; i < projectedOnly.length; i++) {
      ctx.lineTo(projectedOnly[i].x, projectedOnly[i].y);
    }
    ctx.closePath();
    ctx.stroke();

    // Draw vertices as adaptive-sized dots with highlighting
    if (isHovered) {
      ctx.fillStyle = '#FFD700';
      ctx.globalAlpha = Math.min(1.0, (polygon.opacity + 0.6) * depthFactor);
    } else if (isTagged) {
      ctx.fillStyle = '#FF6B9D';
      ctx.globalAlpha = Math.min(1.0, (polygon.opacity + 0.5) * depthFactor);
    } else {
      ctx.fillStyle = polygon.color;
      ctx.globalAlpha = Math.min(1.0, (polygon.opacity + 0.4) * depthFactor);
    }
    
    const baseDotSize = Math.max(1.5, 3 * depthFactor * camera.scale);
    const dotSize = isHovered ? baseDotSize * 1.5 : (isTagged ? baseDotSize * 1.2 : baseDotSize);
    
    projectedOnly.forEach(vertex => {
      ctx.beginPath();
      ctx.arc(vertex.x, vertex.y, dotSize, 0, 2 * Math.PI);
      ctx.fill();
    });
    
    // Draw tag label if model is tagged
    if (isTagged && tagName) {
      const centerX = projectedOnly.reduce((sum, v) => sum + v.x, 0) / projectedOnly.length;
      const centerY = projectedOnly.reduce((sum, v) => sum + v.y, 0) / projectedOnly.length;
      
      ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
      ctx.font = 'bold 10px Arial';
      const textWidth = ctx.measureText(tagName).width;
      ctx.fillRect(centerX - textWidth/2 - 3, centerY - 8, textWidth + 6, 14);
      
      ctx.fillStyle = '#FFFFFF';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(tagName, centerX, centerY);
    }

    ctx.restore();
  }, [project3D, camera.scale, hoveredModel, taggedModels]);

  // Draw enhanced 3D radar labels with intelligent value markers (min/max/median only)
  const draw3DRadarLabels = React.useCallback((ctx: CanvasRenderingContext2D) => {
    const params = circuitParams;
    
    ctx.save();
    
    // Draw all parameter labels with enhanced styling and intelligent value markers
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
      
      // Draw intelligent value markers (min/max/median only)
      let minValue: number, maxValue: number, medianValue: number;
      
      if (param.name.includes('Ω')) {
        // Logarithmic scale for resistance
        const logMin = Math.log10(param.range.min);
        const logMax = Math.log10(param.range.max);
        minValue = param.range.min;
        maxValue = param.range.max;
        medianValue = Math.pow(10, (logMin + logMax) / 2);
      } else {
        // Linear scale for capacitance
        minValue = param.range.min;
        maxValue = param.range.max;
        medianValue = (param.range.min + param.range.max) / 2;
      }
      
      // Format values intelligently
      const formatValue = (value: number, isResistance: boolean) => {
        if (isResistance) {
          return value >= 1000 ? `${(value/1000).toFixed(1)}k` : value.toFixed(0);
        } else {
          return value < 1 ? value.toFixed(1) : value.toFixed(0);
        }
      };
      
      const isResistance = param.name.includes('Ω');
      const minText = formatValue(minValue, isResistance);
      const medianText = formatValue(medianValue, isResistance);
      const maxText = formatValue(maxValue, isResistance);
      
      // Draw compact value markers (min/median/max)
      ctx.fillStyle = '#CCCCCC';
      ctx.font = '9px Arial';
      const markerText = `${minText} | ${medianText} | ${maxText}`;
      ctx.fillText(markerText, labelX, labelY + 10);
    });
    
    ctx.restore();
  }, [project3D, camera.scale, circuitParams]);

  // Draw modern 3D orientation cube with dynamic positioning
  const drawOrientationCube = React.useCallback((ctx: CanvasRenderingContext2D) => {
    const cubeSize = Math.max(60, 80 * camera.scale); // Increased size
    const margin = 10; // Small margin from edge
    const cubeX = actualWidth - cubeSize - margin;
    const cubeY = margin; // Position at top-right corner
    
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
  }, [actualWidth, rotation, camera.scale]);

  // Draw resnorm labels inside the core of the 3D spider plot
  const drawResnormAxis = React.useCallback((ctx: CanvasRenderingContext2D) => {
    if (!filteredModels.length) return;
    
    // Calculate resnorm range from filtered models (respects resnorm range and navigation filtering)
    const resnorms = filteredModels.map(m => m.resnorm || 0).filter(r => r > 0);
    const minResnorm = Math.min(...resnorms);
    const maxResnorm = Math.max(...resnorms);
    
    ctx.save();
    
    // Position resnorm labels at the CENTER of the spider plot (x=0, y=0)
    const coreX = 0;
    const coreY = 0;
    const zAxisHeight = 5.0 * resnormSpread; // Respect resnormSpread prop
    
    // Draw a subtle central vertical line for reference
    const axisStart = project3D({ x: coreX, y: coreY, z: 0, color: '#000000', opacity: 1, model: null as any }); // eslint-disable-line @typescript-eslint/no-explicit-any
    const axisEnd = project3D({ x: coreX, y: coreY, z: zAxisHeight, color: '#000000', opacity: 1, model: null as any }); // eslint-disable-line @typescript-eslint/no-explicit-any
    
    if (axisStart.visible && axisEnd.visible) {
      // Draw subtle central reference line
      ctx.strokeStyle = '#666666';
      ctx.lineWidth = Math.max(1, 2 * camera.scale);
      ctx.globalAlpha = 0.3;
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      ctx.moveTo(axisStart.x, axisStart.y);
      ctx.lineTo(axisEnd.x, axisEnd.y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    
    // Draw resnorm value labels positioned inside the core at different Z heights
    const fontSize = Math.max(10, Math.min(14, 12 * camera.scale));
    ctx.font = `bold ${fontSize}px Arial`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    const numLabels = Math.max(3, Math.min(6, 5)); // Fixed optimal label count for readability
    for (let i = 0; i < numLabels; i++) {
      const z = (i / (numLabels - 1)) * zAxisHeight;
      const normalizedZ = i / (numLabels - 1);
      const resnormValue = minResnorm + (normalizedZ * (maxResnorm - minResnorm));
      
      // Position labels at the center of the spider plot
      const labelPoint = project3D({ x: coreX, y: coreY, z: z, color: '#000000', opacity: 1, model: null as any }); // eslint-disable-line @typescript-eslint/no-explicit-any
      
      if (labelPoint.visible) {
        // Format value label with proper scientific notation
        let labelText: string;
        if (resnormValue >= 1) {
          labelText = resnormValue.toFixed(2);
        } else if (resnormValue >= 0.01) {
          labelText = resnormValue.toFixed(3);
        } else {
          labelText = resnormValue.toExponential(2);
        }
        
        // Draw enhanced circular background for core labels
        const labelWidth = ctx.measureText(labelText).width;
        const labelHeight = fontSize + 4;
        const backgroundRadius = Math.max(labelWidth/2 + 6, labelHeight/2 + 2);
        
        // Draw circular background
        ctx.globalAlpha = 0.9;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
        ctx.beginPath();
        ctx.arc(labelPoint.x, labelPoint.y, backgroundRadius, 0, 2 * Math.PI);
        ctx.fill();
        
        // Draw circular border with correct color scheme
        ctx.strokeStyle = i === 0 ? '#22c55e' : i === numLabels - 1 ? '#ef4444' : '#fbbf24'; // Green for min (better), red for max (worse), yellow for middle
        ctx.lineWidth = 2;
        ctx.globalAlpha = 1.0;
        ctx.stroke();
        
        // Draw text at center
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(labelText, labelPoint.x, labelPoint.y);
        
        // Add small indicator for what this value represents
        if (i === 0) {
          ctx.fillStyle = '#22c55e';
          ctx.font = `${Math.max(8, fontSize - 2)}px Arial`;
          ctx.fillText('MIN', labelPoint.x, labelPoint.y + fontSize/2 + 3);
        } else if (i === numLabels - 1) {
          ctx.fillStyle = '#ef4444';
          ctx.font = `${Math.max(8, fontSize - 2)}px Arial`;
          ctx.fillText('MAX', labelPoint.x, labelPoint.y + fontSize/2 + 3);
        }
      }
    }
    
    // Title removed as requested
    
    
    ctx.restore();
  }, [filteredModels, project3D, camera.scale, resnormSpread]);


  // Draw parameter values list with clean table formatting
  const drawParameterValuesList = React.useCallback((ctx: CanvasRenderingContext2D) => {
    const params = circuitParams;
    const listWidth = 380;
    const listHeight = 200;
    const margin = 10;
    const listX = margin;
    const listY = actualHeight - listHeight - margin;
    
    ctx.save();
    
    // Draw background with subtle styling
    ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
    ctx.fillRect(listX, listY, listWidth, listHeight);
    ctx.strokeStyle = '#666666';
    ctx.lineWidth = 1;
    ctx.strokeRect(listX, listY, listWidth, listHeight);
    
    // Draw header
    ctx.fillStyle = '#FFFFFF';
    ctx.font = 'bold 14px Arial';
    ctx.textAlign = 'left';
    ctx.fillText('Parameter Grid Values', listX + 10, listY + 18);
    
    const numPoints = gridSize || 5;
    ctx.fillStyle = '#CCCCCC';
    ctx.font = '10px Arial';
    ctx.fillText(`Grid: ${numPoints} points per parameter`, listX + 10, listY + 32);
    
    // Table setup
    const tableStartY = listY + 45;
    const rowHeight = 24;
    const colWidth = (listWidth - 20) / (numPoints + 1); // +1 for parameter name column
    
    // Draw table header with grid points
    ctx.fillStyle = '#888888';
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';
    for (let j = 0; j < numPoints; j++) {
      const colX = listX + 10 + (j + 1) * colWidth + colWidth / 2;
      ctx.fillText(`${j + 1}`, colX, tableStartY - 5);
    }
    
    // Draw parameter rows with clean formatting
    params.forEach((param, i) => {
      if (i >= 6) return; // Limit to fit in available space
      
      const rowY = tableStartY + (i * rowHeight);
      
      // Parameter name column
      ctx.fillStyle = '#FFFFFF';
      ctx.font = 'bold 10px Arial';
      ctx.textAlign = 'left';
      const paramLabel = param.key; // Use short form (Ra, Rb, Ca, Cb, Rsh)
      ctx.fillText(paramLabel, listX + 10, rowY + 12);
      
      // Generate and display values in fixed columns
      for (let j = 0; j < numPoints; j++) {
        const ratio = j / (numPoints - 1);
        let value: number;
        let displayValue: string;
        
        if (param.name.includes('Ω')) {
          // Logarithmic scale for resistance
          const logMin = Math.log10(param.range.min);
          const logMax = Math.log10(param.range.max);
          value = Math.pow(10, logMin + ratio * (logMax - logMin));
          
          if (value >= 10000) {
            displayValue = `${(value/1000).toFixed(0)}k`;
          } else if (value >= 1000) {
            displayValue = `${(value/1000).toFixed(1)}k`;
          } else {
            displayValue = `${value.toFixed(0)}`;
          }
        } else {
          // Linear scale for capacitance
          value = param.range.min + ratio * (param.range.max - param.range.min);
          if (value < 1) {
            displayValue = `${value.toFixed(1)}`;
          } else if (value < 10) {
            displayValue = `${value.toFixed(1)}`;
          } else {
            displayValue = `${value.toFixed(0)}`;
          }
        }
        
        // Draw value in column with monospace alignment
        const colX = listX + 10 + (j + 1) * colWidth + colWidth / 2;
        ctx.fillStyle = '#CCCCCC';
        ctx.font = '9px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(displayValue, colX, rowY + 12);
      }
      
      // Draw subtle row separator
      if (i < params.length - 1) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(listX + 5, rowY + rowHeight - 2);
        ctx.lineTo(listX + listWidth - 5, rowY + rowHeight - 2);
        ctx.stroke();
      }
    });
    
    // Draw vertical column separators
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    for (let j = 0; j <= numPoints; j++) {
      const colX = listX + 10 + j * colWidth;
      ctx.beginPath();
      ctx.moveTo(colX, tableStartY - 10);
      ctx.lineTo(colX, tableStartY + params.length * rowHeight - 5);
      ctx.stroke();
    }
    
    ctx.restore();
  }, [actualHeight, circuitParams, gridSize]);

  // Render 3D visualization with enhanced smoothness
  const render3D = React.useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d', { 
      alpha: true, 
      desynchronized: true,
      willReadFrequently: false
    }) as CanvasRenderingContext2D | null;
    if (!ctx) return;

    // Enable smoothing for better rendering quality
    if ('imageSmoothingEnabled' in ctx) {
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
    }
    
    // Clear canvas with anti-aliasing
    ctx.clearRect(0, 0, actualWidth, actualHeight);

    // Set canvas background
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, actualWidth, actualHeight);

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
      const params = paramKeys;
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
    
    
    // Draw parameter values list
    drawParameterValuesList(ctx);
    
    // Draw orientation cube
    drawOrientationCube(ctx);

  }, [convert3DPolygons, project3D, referenceModel, actualWidth, actualHeight, draw3DRadarGrid, draw3DPolygon, draw3DRadarLabels, drawResnormAxis, drawParameterValuesList, drawOrientationCube, paramKeys]);

  // Professional 3D navigation handlers (Blender/Onshape-style) with model selection
  const handleMouseDown = (e: React.MouseEvent) => {
    // Check for model selection on left click without modifiers
    if (e.button === 0 && !e.shiftKey && !e.ctrlKey && onModelTag) {
      const canvas = canvasRef.current;
      if (canvas) {
        const rect = canvas.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const clickY = e.clientY - rect.top;
        
        const clickedModel = findModelAtPoint(clickX, clickY);
        if (clickedModel) {
          // Show tag dialog
          setShowTagDialog({
            model: clickedModel,
            x: e.clientX,
            y: e.clientY
          });
          return; // Don't start dragging if we're showing tag dialog
        }
      }
    }
    
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
    // Handle hover detection for model highlighting
    if (!interaction.isDragging) {
      const canvas = canvasRef.current;
      if (canvas) {
        const rect = canvas.getBoundingClientRect();
        const hoverX = e.clientX - rect.left;
        const hoverY = e.clientY - rect.top;
        
        const hoveredModelFound = findModelAtPoint(hoverX, hoverY);
        setHoveredModel(hoveredModelFound);
      }
      return;
    }

    const deltaX = e.clientX - interaction.lastMousePos.x;
    const deltaY = e.clientY - interaction.lastMousePos.y;
    const sensitivity = 0.3; // Reduced for smoother interaction

    if (interaction.dragMode === 'orbit') {
      // Orbit around target with smooth interpolation
      const smoothedDeltaX = deltaX * sensitivity;
      const smoothedDeltaY = deltaY * sensitivity;
      const newRotation = {
        x: Math.max(-89, Math.min(89, rotation.x + smoothedDeltaY)),
        y: (rotation.y + smoothedDeltaX) % 360,
        z: rotation.z
      };
      setRotation(newRotation);
      if (onRotationChange) {
        onRotationChange(newRotation);
      }
    } else if (interaction.dragMode === 'pan') {
      // Pan the target with smooth movement
      const panSensitivity = 0.003 * camera.distance; // Reduced for smoother panning
      setCamera(prev => ({
        ...prev,
        target: {
          x: prev.target.x - deltaX * panSensitivity,
          y: prev.target.y + deltaY * panSensitivity,
          z: prev.target.z
        }
      }));
    } else if (interaction.dragMode === 'zoom') {
      // Zoom by dragging with smooth scaling
      const zoomSensitivity = 0.008; // Reduced for smoother zooming
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
    const zoomSpeed = 0.05; // Reduced for smoother zooming
    const zoomFactor = e.deltaY > 0 ? (1 + zoomSpeed) : (1 - zoomSpeed);
    
    if (e.shiftKey) {
      // Shift+scroll for smooth scale adjustment
      setCamera(prev => ({
        ...prev,
        scale: Math.max(0.1, Math.min(10, prev.scale * zoomFactor))
      }));
    } else {
      // Normal scroll for smooth distance zoom
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
    <div ref={containerRef} className="relative bg-black overflow-hidden w-full h-full">
      {!isComponentReady ? (
        <div className="absolute inset-0 bg-black flex items-center justify-center">
          <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
        </div>
      ) : null}
      {isComponentReady && (
        <canvas
          ref={canvasRef}
        width={actualWidth}
        height={actualHeight}
        className={`w-full h-full object-contain ${
          interaction.isDragging 
            ? interaction.dragMode === 'pan' ? 'cursor-move' 
              : interaction.dragMode === 'zoom' ? 'cursor-ns-resize'
              : 'cursor-grabbing'
            : 'cursor-grab'
        } focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50`}
        style={{ maxWidth: '100%', maxHeight: '100%' }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
        onContextMenu={handleContextMenu}
        onFocus={() => canvasRef.current?.focus()}
      />
      )}
      
      {/* Reset Button - Clean positioning */}
      <button
        onClick={() => {
          setRotation({ x: -30, y: 45, z: 0 });
          setCamera(prev => ({ ...prev, distance: 12, target: { x: 0, y: 0, z: 2.5 }, scale: 1.0 }));
        }}
        className="absolute top-4 left-4 px-3 py-1.5 bg-gray-900 bg-opacity-90 hover:bg-opacity-100 rounded text-white text-xs transition-all z-10 border border-gray-600"
        title="Reset View (R)"
      >
        Reset
      </button>
      
      {showControls && (
        <div className="absolute bottom-4 right-4">
          <div className="bg-black bg-opacity-80 p-3 rounded text-white text-xs">
            <div className="space-y-1">
              <div>Drag: Rotate • Shift+Drag: Pan • Scroll: Zoom</div>
              <div className="text-xs text-gray-400">R: Reset</div>
            </div>
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
      
      {models.length > 50000 && (
        <div className="absolute inset-0 flex items-center justify-center text-gray-400 bg-black bg-opacity-50">
          <div className="text-center p-8 bg-gray-900 bg-opacity-80 rounded-lg">
            <div className="text-xl text-orange-400 mb-4">⚠️ Exceeds Rendering Capacity</div>
            <div className="text-neutral-300 mb-2">
              Too many models to render in 3D: {models.length.toLocaleString()}
            </div>
            <div className="text-sm text-neutral-400 mb-4">
              Maximum safe limit: 50,000 models
            </div>
            <div className="text-xs text-neutral-500">
              Try reducing the resnorm matching portion or grid size for better performance
            </div>
          </div>
        </div>
      )}

      {/* Tag Dialog */}
      {showTagDialog && (
        <>
          {/* Backdrop */}
          <div 
            className="fixed inset-0 bg-black bg-opacity-50 z-40"
            onClick={() => setShowTagDialog(null)}
          />
          
          {/* Dialog */}
          <div 
            className="fixed z-50 bg-gray-900 border border-gray-600 rounded-lg p-3 shadow-xl"
            style={{ 
              left: Math.min(showTagDialog.x + 10, window.innerWidth - 200), 
              top: Math.max(10, showTagDialog.y - 50)
            }}
          >
            <div className="text-white text-sm mb-2">
              Tag Model 
              <span className="text-gray-400 text-xs ml-2">
                (Resnorm: {showTagDialog.model.resnorm?.toFixed(4) || 'N/A'})
              </span>
            </div>
            <input
              type="text"
              placeholder="Enter tag name..."
              className="w-40 px-2 py-1 text-xs bg-gray-800 border border-gray-600 rounded text-white focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
              onKeyDown={async (e) => {
                if (e.key === 'Enter') {
                  const tagName = e.currentTarget.value.trim();
                  if (tagName && showTagDialog && onModelTag) {
                    setIsTagging(true);
                    try {
                      // Use parent callback for tagging
                      onModelTag(showTagDialog.model, tagName);
                    } catch (error) {
                      console.error('Failed to tag model:', error);
                    } finally {
                      setIsTagging(false);
                    }
                  }
                  setShowTagDialog(null);
                } else if (e.key === 'Escape') {
                  setShowTagDialog(null);
                }
              }}
              autoFocus
            />
            <div className="flex gap-2 mt-2">
              <button
                className="px-2 py-1 text-xs bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 transition-colors rounded text-white"
                disabled={isTagging}
                onClick={async () => {
                  if (!showTagDialog || !onModelTag) return;
                  
                  const input = document.querySelector('input[placeholder="Enter tag name..."]') as HTMLInputElement;
                  const tagName = input?.value.trim();
                  if (!tagName) return;
                  
                  setIsTagging(true);
                  try {
                    // Use parent callback for tagging
                    onModelTag(showTagDialog.model, tagName);
                  } catch (error) {
                    console.error('Failed to tag model:', error);
                  } finally {
                    setIsTagging(false);
                  }
                  setShowTagDialog(null);
                }}
              >
                {isTagging ? 'Saving...' : 'Tag'}
              </button>
              <button
                className="px-2 py-1 text-xs bg-gray-600 hover:bg-gray-700 transition-colors rounded text-white"
                onClick={() => setShowTagDialog(null)}
              >
                Cancel
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
};