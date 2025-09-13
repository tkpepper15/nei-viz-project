"use client";

import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { ModelSnapshot } from '../types';
import { PARAMETER_RANGES, DEFAULT_GRID_SIZE, faradToMicroFarad, CircuitParameters } from '../types/parameters';

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
  currentResnorm?: number | null; // Current resnorm value being navigated
  onResnormSelect?: (resnorm: number) => void; // Callback for shift-click resnorm selection
  onCurrentResnormChange?: (resnorm: number | null) => void; // Callback when user hovers over resnorm levels
  highlightedModelId?: string | null; // ID of model to highlight (for tagged model selection)
  selectedResnormRange?: { min: number; max: number } | null; // Currently selected range in histogram
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
  currentResnorm = null,
  onResnormSelect,
  onCurrentResnormChange,
  highlightedModelId = null,
  selectedResnormRange = null,
}) => {
  // Debug component props
  console.log('🐛 SpiderPlot3D: Component initialized with:', {
    modelsCount: models.length,
    gridSize,
    resnormSpread,
    hasReferenceModel: !!referenceModel,
    resnormRange,
    selectedResnormRange
  });

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
  
  // Separate visual and computational resnorm states
  const [visualResnorm, setVisualResnorm] = useState<number | null>(null); // For smooth UI interpolation only
  const [isModelSelected, setIsModelSelected] = useState<boolean>(false);
  
  // Refs for debounced computational updates
  const lastComputationalResnorm = useRef<number | null>(null);
  const computationalUpdateThreshold = 0.005; // Only update parent for significant changes
  
  // High-performance throttling for ultra-smooth interactions
  const lastMouseMoveTime = useRef<number>(0);
  const lastDragTime = useRef<number>(0);
  const mouseMoveThrottleMs = 10; // ~100fps for balanced performance and smoothness
  const dragThrottleMs = 3; // ~333fps for buttery-smooth rotation
  
  // Performance optimization refs
  const lastRenderTime = useRef<number>(0);
  const renderThrottleMs = 16; // ~60fps cap for smooth rendering
  

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
    if (!filteredModels.length) {
      console.warn('🔴 SpiderPlot3D: No filtered models available for 3D conversion');
      return [];
    }

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
    

    console.log('🟢 SpiderPlot3D: Converting', filteredModels.length, 'models to 3D polygons');
    
    return filteredModels.map(model => {
      const resnorm = model.resnorm || 0;
      
      // Validate model parameters
      if (!model.parameters) {
        console.warn('🔴 SpiderPlot3D: Model missing parameters:', model.id);
        return null;
      }
      
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
        
        // Validate parameter value
        if (typeof value !== 'number' || !isFinite(value)) {
          console.warn('🔴 SpiderPlot3D: Invalid parameter value for', param, ':', value, 'in model', model.id);
        }
        
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
          z: zHeight,
          color: color,
          opacity: Math.max(0.2, 1.0 - normalizedResnorm * 0.7),
          model: model
        };
      });

      // Log first model for debugging
      if (model.id === filteredModels[0]?.id) {
        console.log('🟢 SpiderPlot3D: First model vertex sample:', {
          modelId: model.id,
          resnorm,
          normalizedResnorm,
          zHeight,
          firstVertex: vertices[0],
          parameters: model.parameters
        });
      }

      return {
        vertices,
        color: color,
        opacity: Math.max(0.2, 1.0 - normalizedResnorm * 0.7),
        model,
        zHeight
      };
    }).filter(polygon => polygon !== null); // Filter out invalid polygons
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
        .map(vertex => project3D(vertex as Point3D))
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

  // Draw ground truth cross-section line through the entire resnorm spectrum
  const drawGroundTruthCrossSection = React.useCallback((ctx: CanvasRenderingContext2D) => {
    if (!referenceModel || !filteredModels.length) return;

    ctx.save();
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.8;
    ctx.setLineDash([6, 3]);

    // Calculate the full resnorm range from all filtered models
    const resnorms = filteredModels.map(m => m.resnorm || 0).filter(r => r > 0);
    if (resnorms.length === 0) return;

    // Get ground truth parameter positions (same calculation as in convert3DPolygons)
    const params = paramKeys;
    const paramRanges = {
      Rsh: PARAMETER_RANGES.Rsh,
      Ra: PARAMETER_RANGES.Ra,
      Rb: PARAMETER_RANGES.Rb,
      Ca: PARAMETER_RANGES.Ca,
      Cb: PARAMETER_RANGES.Cb
    };

    // Calculate ground truth vertices at different Z levels throughout the resnorm spectrum
    const numLevels = 20; // Number of levels to draw the continuous line
    const groundTruthVertices = Array.from({ length: numLevels }, (_, levelIndex) => {
      // Calculate Z height for this level
      const normalizedZ = levelIndex / (numLevels - 1);
      const zHeight = normalizedZ * 5.0 * resnormSpread; // Match the Z scaling from convert3DPolygons

      // Calculate ground truth parameter positions at this Z level
      return params.map((param, i) => {
        const value = referenceModel.parameters[param as keyof typeof referenceModel.parameters] as number;
        const range = paramRanges[param as keyof typeof paramRanges];
        
        // Normalize parameter value to 0-1 range (logarithmic) - same as convert3DPolygons
        const logMin = Math.log10(range.min);
        const logMax = Math.log10(range.max);
        const logValue = Math.log10(Math.max(range.min, Math.min(range.max, value)));
        const normalizedValue = (logValue - logMin) / (logMax - logMin);
        
        // Calculate radar position (pentagon) - same as convert3DPolygons
        const angle = (i * 2 * Math.PI) / params.length - Math.PI / 2;
        const radius = normalizedValue * 2.0;
        
        return {
          x: Math.cos(angle) * radius,
          y: Math.sin(angle) * radius,
          z: zHeight,
          color: '#FFFFFF',
          opacity: 1,
          model: referenceModel
        };
      });
    });

    // Draw vertical lines connecting ground truth vertices at each parameter position
    for (let paramIndex = 0; paramIndex < params.length; paramIndex++) {
      ctx.beginPath();
      let pathStarted = false;

      for (let levelIndex = 0; levelIndex < numLevels; levelIndex++) {
        const vertex = groundTruthVertices[levelIndex][paramIndex];
        const projected = project3D(vertex);

        if (projected.visible) {
          if (!pathStarted) {
            ctx.moveTo(projected.x, projected.y);
            pathStarted = true;
          } else {
            ctx.lineTo(projected.x, projected.y);
          }
        }
      }

      if (pathStarted) {
        ctx.stroke();
      }
    }

    ctx.restore();
  }, [referenceModel, filteredModels, resnormSpread, paramKeys, project3D]);

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
    
    // Check if this model is hovered, tagged, or highlighted
    const isHovered = hoveredModel?.id === polygon.model.id;
    const isTagged = taggedModels.has(polygon.model.id);
    const isHighlighted = highlightedModelId === polygon.model.id;
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
    
    // Enhanced styling for hovered/tagged/highlighted models
    if (isHovered) {
      ctx.globalAlpha = Math.min(1.0, baseOpacity + 0.3);
      ctx.lineWidth = baseLineWidth * 2;
      ctx.strokeStyle = '#FFFF00'; // Yellow highlight for hover
    } else if (isHighlighted) {
      ctx.globalAlpha = Math.min(1.0, baseOpacity + 0.4);
      ctx.lineWidth = baseLineWidth * 2.5;
      ctx.strokeStyle = '#00FFFF'; // Cyan highlight for selected model
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
      ctx.fillStyle = '#FFFF00';
      ctx.globalAlpha = Math.min(1.0, (polygon.opacity + 0.6) * depthFactor);
    } else if (isHighlighted) {
      ctx.fillStyle = '#00FFFF';
      ctx.globalAlpha = Math.min(1.0, (polygon.opacity + 0.7) * depthFactor);
    } else if (isTagged) {
      ctx.fillStyle = '#FF6B9D';
      ctx.globalAlpha = Math.min(1.0, (polygon.opacity + 0.5) * depthFactor);
    } else {
      ctx.fillStyle = polygon.color;
      ctx.globalAlpha = Math.min(1.0, (polygon.opacity + 0.4) * depthFactor);
    }
    
    const baseDotSize = Math.max(1.5, 3 * depthFactor * camera.scale);
    const dotSize = isHovered ? baseDotSize * 1.5 : 
                    (isHighlighted ? baseDotSize * 2 : 
                     (isTagged ? baseDotSize * 1.2 : baseDotSize));
    
    projectedOnly.forEach(vertex => {
      ctx.beginPath();
      ctx.arc(vertex.x, vertex.y, dotSize, 0, 2 * Math.PI);
      ctx.fill();
    });
    
    // Draw tag label if model is tagged - positioned outside the radar like parameter labels
    if (isTagged && tagName) {
      // Find the vertex with the maximum radius to position the label outside
      const centerX = projectedOnly.reduce((sum, v) => sum + v.x, 0) / projectedOnly.length;
      const centerY = projectedOnly.reduce((sum, v) => sum + v.y, 0) / projectedOnly.length;
      
      // Calculate direction from center to furthest vertex
      const screenCenter = project3D({ x: 0, y: 0, z: polygon.zHeight, color: '#000000', opacity: 1, model: polygon.model });
      if (screenCenter.visible) {
        const dirX = centerX - screenCenter.x;
        const dirY = centerY - screenCenter.y;
        const dirLength = Math.sqrt(dirX * dirX + dirY * dirY);
        
        if (dirLength > 0) {
          // Position label outside the polygon in the direction of the polygon center
          const labelDistance = Math.max(80, dirLength + 40); // Position well outside
          const labelX = screenCenter.x + (dirX / dirLength) * labelDistance;
          const labelY = screenCenter.y + (dirY / dirLength) * labelDistance;
          
          // Draw enhanced label background
          ctx.font = 'bold 11px Arial';
          const textWidth = ctx.measureText(tagName).width;
          const backgroundWidth = textWidth + 12;
          const backgroundHeight = 20;
          
          ctx.fillStyle = 'rgba(255, 107, 157, 0.9)'; // Pink background for tagged models
          ctx.fillRect(labelX - backgroundWidth/2, labelY - backgroundHeight/2, backgroundWidth, backgroundHeight);
          ctx.strokeStyle = '#FF6B9D';
          ctx.lineWidth = 2;
          ctx.strokeRect(labelX - backgroundWidth/2, labelY - backgroundHeight/2, backgroundWidth, backgroundHeight);
          
          // Draw tag name
          ctx.fillStyle = '#FFFFFF';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(tagName, labelX, labelY);
          
          // Draw connecting line from polygon to label
          ctx.strokeStyle = '#FF6B9D';
          ctx.lineWidth = 1;
          ctx.globalAlpha = 0.6;
          ctx.setLineDash([2, 2]);
          ctx.beginPath();
          ctx.moveTo(centerX, centerY);
          ctx.lineTo(labelX - (dirX / dirLength) * (backgroundWidth/2 + 5), labelY - (dirY / dirLength) * (backgroundHeight/2 + 5));
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.globalAlpha = 1.0;
        }
      }
    }

    ctx.restore();
  }, [project3D, camera.scale, hoveredModel, taggedModels, highlightedModelId]);

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

  // Refs for caching crosshair model lookups
  const lastResnormRef = useRef<number | null>(null);
  const lastModelRef = useRef<ModelSnapshot | null>(null);

  // Find nearest discrete grid model for computational accuracy
  const findNearestGridModel = useCallback((resnorm: number): ModelSnapshot | null => {
    if (!filteredModels.length) return null;
    
    let closestModel: ModelSnapshot | null = null;
    let closestDistance = Infinity;
    
    filteredModels.forEach(model => {
      const modelResnorm = model.resnorm || 0;
      const distance = Math.abs(modelResnorm - resnorm);
      if (distance < closestDistance) {
        closestDistance = distance;
        closestModel = model;
      }
    });
    
    return closestModel;
  }, [filteredModels]);

  // Find nearest actual model for polygon highlighting during crosshair movement
  const findNearestActualModel = useCallback((resnorm: number): ModelSnapshot | null => {
    if (!filteredModels.length) return null;
    
    // Use cached result if resnorm hasn't changed significantly
    const resnormThreshold = 0.01;
    if (lastResnormRef.current !== null && lastModelRef.current !== null &&
        Math.abs(lastResnormRef.current - resnorm) < resnormThreshold) {
      return lastModelRef.current;
    }
    
    let closestModel: ModelSnapshot | null = null;
    let closestDistance = Infinity;
    
    // Find the actual polygon model closest to the crosshair resnorm
    filteredModels.forEach(model => {
      const modelResnorm = model.resnorm || 0;
      const distance = Math.abs(modelResnorm - resnorm);
      if (distance < closestDistance) {
        closestDistance = distance;
        closestModel = model;
      }
    });
    
    // Cache the result for performance
    lastResnormRef.current = resnorm;
    lastModelRef.current = closestModel;
    
    return closestModel;
  }, [filteredModels]);
  
  // Smooth parameter interpolation for visual display only
  const getInterpolatedParameters = useCallback((resnorm: number): CircuitParameters | null => {
    if (!filteredModels.length) return null;
    
    // Find the two closest models for interpolation
    const sortedModels = [...filteredModels].sort((a, b) => (a.resnorm || 0) - (b.resnorm || 0));
    
    // Find bounding models
    let lowerModel: ModelSnapshot | null = null;
    let upperModel: ModelSnapshot | null = null;
    
    for (let i = 0; i < sortedModels.length - 1; i++) {
      const currentResnorm = sortedModels[i].resnorm || 0;
      const nextResnorm = sortedModels[i + 1].resnorm || 0;
      
      if (resnorm >= currentResnorm && resnorm <= nextResnorm) {
        lowerModel = sortedModels[i];
        upperModel = sortedModels[i + 1];
        break;
      }
    }
    
    // If no bounds found, use nearest model
    if (!lowerModel || !upperModel) {
      const nearest = findNearestGridModel(resnorm);
      return nearest?.parameters || null;
    }
    
    // Interpolate parameters between bounds
    const lowerResnorm = lowerModel.resnorm || 0;
    const upperResnorm = upperModel.resnorm || 0;
    const t = (resnorm - lowerResnorm) / (upperResnorm - lowerResnorm);
    
    const interpolated: CircuitParameters = {
      Rsh: lowerModel.parameters.Rsh + t * (upperModel.parameters.Rsh - lowerModel.parameters.Rsh),
      Ra: lowerModel.parameters.Ra + t * (upperModel.parameters.Ra - lowerModel.parameters.Ra),
      Rb: lowerModel.parameters.Rb + t * (upperModel.parameters.Rb - lowerModel.parameters.Rb),
      Ca: lowerModel.parameters.Ca + t * (upperModel.parameters.Ca - lowerModel.parameters.Ca),
      Cb: lowerModel.parameters.Cb + t * (upperModel.parameters.Cb - lowerModel.parameters.Cb),
      frequency_range: lowerModel.parameters.frequency_range // Use same frequency range
    };
    
    return interpolated;
  }, [filteredModels, findNearestGridModel]);
  
  // Memoized crosshair model with exact match detection for yellow selection
  const crosshairModel = useMemo((): ModelSnapshot | null => {
    const activeResnorm = isModelSelected ? currentResnorm : visualResnorm;
    if (!activeResnorm || !filteredModels.length) return null;
    
    // Early return if no significant change in resnorm
    const threshold = 0.001;
    
    if (lastResnormRef.current && Math.abs(lastResnormRef.current - activeResnorm) < threshold) {
      return lastModelRef.current;
    }
    
    // For locked selections (yellow), only show exact model matches
    if (isModelSelected) {
      const exactModel = findNearestGridModel(activeResnorm);
      // Only show yellow if we have a very close match (within 0.005% tolerance)
      const exactTolerance = 0.00005;
      if (exactModel && exactModel.resnorm !== undefined && Math.abs(exactModel.resnorm - activeResnorm) <= exactTolerance) {
        lastResnormRef.current = activeResnorm;
        lastModelRef.current = exactModel;
        return exactModel;
      }
      // No exact match for locked selection - return null to hide yellow
      lastResnormRef.current = activeResnorm;
      lastModelRef.current = null;
      return null;
    }
    
    // For free scrolling (cyan), create smooth interpolated model
    const interpolatedParams = getInterpolatedParameters(activeResnorm);
    if (!interpolatedParams) {
      // Fallback to nearest discrete model if interpolation fails
      const nearest = findNearestGridModel(activeResnorm);
      lastResnormRef.current = activeResnorm;
      lastModelRef.current = nearest;
      return nearest;
    }
    
    // Create synthetic model for smooth visual display
    const syntheticModel: ModelSnapshot = {
      id: `interpolated_${activeResnorm.toFixed(6)}`,
      name: `Interpolated Model`,
      timestamp: Date.now(),
      parameters: interpolatedParams,
      data: [], // Empty data since this is visual-only
      resnorm: activeResnorm,
      color: '#00FFFF',
      isVisible: true,
      opacity: 0.8
    };
    
    lastResnormRef.current = activeResnorm;
    lastModelRef.current = syntheticModel;
    return syntheticModel;
  }, [isModelSelected, currentResnorm, visualResnorm, getInterpolatedParameters, findNearestGridModel, filteredModels.length]);

  // Draw pentagonal outline showing current crosshair model, grid, and ground truth
  const drawPentagonalOutline = React.useCallback((ctx: CanvasRenderingContext2D) => {
    const panelWidth = Math.max(120, 140 * camera.scale);
    const panelHeight = Math.max(120, 140 * camera.scale);
    const margin = 15;
    const panelX = actualWidth - panelWidth - margin;
    const panelY = margin;
    
    ctx.save();
    
    // Draw panel background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
    ctx.fillRect(panelX - 10, panelY - 5, panelWidth + 20, panelHeight + 20);
    ctx.strokeStyle = '#666666';
    ctx.lineWidth = 1;
    ctx.strokeRect(panelX - 10, panelY - 5, panelWidth + 20, panelHeight + 20);
    
    // Pentagon center and radius
    const centerX = panelX + panelWidth / 2;
    const centerY = panelY + panelHeight / 2;
    const radius = Math.min(panelWidth, panelHeight) / 3;
    
    // Get parameter configuration
    const params = paramKeys;
    
    // Helper function to draw pentagon outline
    const drawPentagon = (centerX: number, centerY: number, values: number[], color: string, lineWidth: number, isDashed = false) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      if (isDashed) {
        ctx.setLineDash([4, 4]);
      } else {
        ctx.setLineDash([]);
      }
      
      const vertices = params.map((param, i) => {
        const paramRange = PARAMETER_RANGES[param as keyof typeof PARAMETER_RANGES];
        const value = values[i];
        
        // Normalize parameter value to 0-1 range (logarithmic)
        const logMin = Math.log10(paramRange.min);
        const logMax = Math.log10(paramRange.max);
        const logValue = Math.log10(Math.max(paramRange.min, Math.min(paramRange.max, value)));
        const normalizedValue = (logValue - logMin) / (logMax - logMin);
        
        // Calculate pentagon position (matching 3D spider plot parameter arrangements)
        const angle = (i * 2 * Math.PI) / params.length - Math.PI / 2;
        const vertexRadius = normalizedValue * radius;
        
        return {
          x: centerX + Math.cos(angle) * vertexRadius,
          y: centerY + Math.sin(angle) * vertexRadius
        };
      });
      
      // Draw pentagon
      ctx.beginPath();
      ctx.moveTo(vertices[0].x, vertices[0].y);
      for (let i = 1; i < vertices.length; i++) {
        ctx.lineTo(vertices[i].x, vertices[i].y);
      }
      ctx.closePath();
      ctx.stroke();
      
      // Draw vertices as dots
      ctx.fillStyle = color;
      const dotSize = lineWidth / 2 + 1;
      vertices.forEach(vertex => {
        ctx.beginPath();
        ctx.arc(vertex.x, vertex.y, dotSize, 0, 2 * Math.PI);
        ctx.fill();
      });
      
      ctx.setLineDash([]); // Reset dash
    };
    
    // Use the memoized crosshair model
    
    // Draw outer pentagon boundary (max values)
    const maxValues = params.map(param => PARAMETER_RANGES[param as keyof typeof PARAMETER_RANGES].max);
    drawPentagon(centerX, centerY, maxValues, '#333333', 1);
    
    // Draw ground truth pentagon if available
    if (referenceModel?.parameters) {
      const gtValues = params.map(param => referenceModel.parameters[param as keyof typeof referenceModel.parameters] as number);
      drawPentagon(centerX, centerY, gtValues, '#FFFFFF', 2, true);
    }
    
    // Draw crosshair model pentagon (dynamic based on crosshair position)
    if (crosshairModel && crosshairModel.parameters) {
      const crosshairValues = params.map(param => (crosshairModel.parameters as any)[param as keyof CircuitParameters] as number); // eslint-disable-line @typescript-eslint/no-explicit-any
      // Only show yellow if exact model match exists (consistent with 3D crosshair)
      const hasExactMatch = isModelSelected && crosshairModel !== null && !crosshairModel.id.startsWith('interpolated_');
      const crosshairColor = hasExactMatch ? '#FFFF00' : '#00FFFF'; 
      const lineWidth = hasExactMatch ? 3 : 2;
      drawPentagon(centerX, centerY, crosshairValues, crosshairColor, lineWidth);
    } else if (gridSize && gridSize > 1) {
      // Fallback to grid pentagon if no crosshair model
      const gridValues = params.map(param => {
        const range = PARAMETER_RANGES[param as keyof typeof PARAMETER_RANGES];
        const logMin = Math.log10(range.min);
        const logMax = Math.log10(range.max);
        const logMid = (logMin + logMax) / 2;
        return Math.pow(10, logMid);
      });
      drawPentagon(centerX, centerY, gridValues, '#666666', 1);
    }
    
    // Draw parameter labels around the pentagon
    ctx.fillStyle = '#CCCCCC';
    ctx.font = '9px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    params.forEach((param, i) => {
      const angle = (i * 2 * Math.PI) / params.length - Math.PI / 2 + Math.PI;
      const labelRadius = radius + 25;
      const labelX = centerX + Math.cos(angle) * labelRadius;
      const labelY = centerY + Math.sin(angle) * labelRadius;
      
      ctx.fillText(param, labelX, labelY);
    });
    
    // Draw legend with dynamic crosshair indicator
    ctx.textAlign = 'left';
    ctx.font = '8px Arial';
    let legendY = panelY + panelHeight - 45; // More space for additional legend item
    
    // Crosshair model indicator
    if (crosshairModel && crosshairModel.parameters) {
      const crosshairColor = isModelSelected ? '#FFFF00' : '#00FFFF';
      ctx.strokeStyle = crosshairColor;
      ctx.lineWidth = isModelSelected ? 3 : 2;
      ctx.beginPath();
      ctx.moveTo(panelX, legendY);
      ctx.lineTo(panelX + 8, legendY);
      ctx.stroke();
      ctx.fillStyle = '#CCCCCC';
      const labelText = isModelSelected ? 'Selected' : 'Crosshair';
      ctx.fillText(labelText, panelX + 12, legendY + 1);
      legendY += 12;
    }
    
    // Ground truth indicator (if available)
    if (referenceModel?.parameters) {
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(panelX, legendY);
      ctx.lineTo(panelX + 8, legendY);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#CCCCCC';
      ctx.fillText('Truth', panelX + 12, legendY + 1);
      legendY += 12;
    }
    
    // Boundary indicator
    ctx.strokeStyle = '#333333';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(panelX, legendY);
    ctx.lineTo(panelX + 8, legendY);
    ctx.stroke();
    ctx.fillStyle = '#CCCCCC';
    ctx.fillText('Max', panelX + 12, legendY + 1);
    
    ctx.restore();
  }, [actualWidth, referenceModel, paramKeys, gridSize, camera.scale, crosshairModel, isModelSelected]);

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

    // Draw selected resnorm range highlight (simplified - no cylindrical grid)
    if (selectedResnormRange && selectedResnormRange.min >= minResnorm && selectedResnormRange.max <= maxResnorm) {
      const normalizedMinResnorm = (selectedResnormRange.min - minResnorm) / (maxResnorm - minResnorm);
      const normalizedMaxResnorm = (selectedResnormRange.max - minResnorm) / (maxResnorm - minResnorm);
      const minZ = normalizedMinResnorm * zAxisHeight;
      const maxZ = normalizedMaxResnorm * zAxisHeight;
      
      // Draw simplified range indicators at center axis
      const rangeHeight = maxZ - minZ;
      
      if (rangeHeight > 0.1) {
        ctx.strokeStyle = '#FFFF00'; // Yellow highlight color
        ctx.lineWidth = Math.max(2, 3 * camera.scale);
        ctx.globalAlpha = 0.8;
        ctx.setLineDash([]);
        
        // Draw vertical range line at center
        const bottomPoint = project3D({ x: coreX, y: coreY, z: minZ, color: '#FFFF00', opacity: 1, model: null as any }); // eslint-disable-line @typescript-eslint/no-explicit-any
        const topPoint = project3D({ x: coreX, y: coreY, z: maxZ, color: '#FFFF00', opacity: 1, model: null as any }); // eslint-disable-line @typescript-eslint/no-explicit-any
        
        if (bottomPoint.visible && topPoint.visible) {
          ctx.beginPath();
          ctx.moveTo(bottomPoint.x, bottomPoint.y);
          ctx.lineTo(topPoint.x, topPoint.y);
          ctx.stroke();
          
          // Add range end markers
          ctx.fillStyle = '#FFFF00';
          ctx.globalAlpha = 1.0;
          const markerSize = Math.max(3, 4 * camera.scale);
          
          ctx.beginPath();
          ctx.arc(bottomPoint.x, bottomPoint.y, markerSize, 0, 2 * Math.PI);
          ctx.fill();
          
          ctx.beginPath();
          ctx.arc(topPoint.x, topPoint.y, markerSize, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
      
      // Draw range labels
      ctx.globalAlpha = 1.0;
      const centerZ = (minZ + maxZ) / 2;
      const centerPoint = project3D({ x: coreX + 2.5, y: coreY, z: centerZ, color: '#FFFF00', opacity: 1, model: null as any }); // eslint-disable-line @typescript-eslint/no-explicit-any
      if (centerPoint.visible) {
        const fontSize = Math.max(9, 11 * camera.scale);
        ctx.font = `bold ${fontSize}px Arial`;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = 'rgba(255, 215, 0, 0.9)';
        const rangeText = `Range: ${selectedResnormRange.min < 1 ? selectedResnormRange.min.toFixed(3) : selectedResnormRange.min.toFixed(2)} - ${selectedResnormRange.max < 1 ? selectedResnormRange.max.toFixed(3) : selectedResnormRange.max.toFixed(2)}`;
        const textWidth = ctx.measureText(rangeText).width;
        ctx.fillRect(centerPoint.x - 4, centerPoint.y - fontSize/2 - 2, textWidth + 8, fontSize + 4);
        
        ctx.fillStyle = '#000000';
        ctx.fillText(rangeText, centerPoint.x, centerPoint.y);
      }
    }

    // Draw crosshair - always visible when currentResnorm or visualResnorm is available
    const activeVisualResnorm = isModelSelected ? currentResnorm : (visualResnorm || currentResnorm);
    if (activeVisualResnorm !== null && activeVisualResnorm >= minResnorm && activeVisualResnorm <= maxResnorm) {
      const normalizedCurrentResnorm = (activeVisualResnorm - minResnorm) / (maxResnorm - minResnorm);
      const currentZ = normalizedCurrentResnorm * zAxisHeight;
      
      // Enhanced styling for different interaction modes
      const isLocked = isModelSelected && crosshairModel !== null; // Only yellow if exact model match
      const isHovering = visualResnorm !== null && !isModelSelected;
      
      // Color coding: Yellow (exact locked model) > Cyan (hovering) > Blue (scrubber control)
      ctx.strokeStyle = isLocked ? '#FFFF00' : (isHovering ? '#00FFFF' : '#0080FF'); 
      ctx.lineWidth = Math.max(2, isLocked ? 4 : (isHovering ? 3 : 2.5) * camera.scale);
      ctx.globalAlpha = isLocked ? 1.0 : (isHovering ? 0.8 : 0.6);
      ctx.setLineDash(isLocked ? [] : (isHovering ? [4, 4] : [2, 6])); // Different dash patterns for each mode
      
      // Draw 5-prong crosshair aligned with pentagon parameter directions
      const crosshairLength = 1.8; // Slightly larger crosshair
      
      // Calculate center point once for performance
      const centerPoint = project3D({ x: 0, y: 0, z: currentZ, color: ctx.strokeStyle, opacity: 1, model: null as any }); // eslint-disable-line @typescript-eslint/no-explicit-any
      
      if (centerPoint.visible) {
        // Draw 5 prongs pointing to each parameter direction
        for (let i = 0; i < 5; i++) {
          // Calculate angle for each parameter (matching 3D spider plot layout)
          const angle = (i * 2 * Math.PI) / 5 - Math.PI / 2;
          
          // Calculate prong endpoints
          const prongX = Math.cos(angle) * crosshairLength;
          const prongY = Math.sin(angle) * crosshairLength;
          
          // Draw prong from center to parameter direction
          const prongPoint = project3D({ x: prongX, y: prongY, z: currentZ, color: ctx.strokeStyle, opacity: 1, model: null as any }); // eslint-disable-line @typescript-eslint/no-explicit-any
          
          if (prongPoint.visible) {
            ctx.beginPath();
            ctx.moveTo(centerPoint.x, centerPoint.y);
            ctx.lineTo(prongPoint.x, prongPoint.y);
            ctx.stroke();
          }
        }
        
        // Add center dot
        ctx.fillStyle = ctx.strokeStyle;
        ctx.globalAlpha = 1.0;
        const dotSize = isLocked ? 4 : 3;
        ctx.beginPath();
        ctx.arc(centerPoint.x, centerPoint.y, dotSize, 0, 2 * Math.PI);
        ctx.fill();
      }
      
      ctx.setLineDash([]); // Reset dash pattern
      
      // Add resnorm value label next to crosshair
      const labelPoint = project3D({ x: crosshairLength + 0.5, y: 0, z: currentZ, color: ctx.strokeStyle, opacity: 1, model: null as any }); // eslint-disable-line @typescript-eslint/no-explicit-any
      if (labelPoint.visible) {
        const fontSize = Math.max(9, 10 * camera.scale);
        ctx.font = `bold ${fontSize}px Arial`;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        
        const labelText = activeVisualResnorm < 1 ? activeVisualResnorm.toFixed(3) : activeVisualResnorm.toFixed(2);
        
        // Background for label
        const textWidth = ctx.measureText(labelText).width;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(labelPoint.x - 2, labelPoint.y - fontSize/2 - 1, textWidth + 4, fontSize + 2);
        
        // Label text
        ctx.fillStyle = ctx.strokeStyle;
        ctx.fillText(labelText, labelPoint.x, labelPoint.y);
      }
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
  }, [filteredModels, project3D, camera.scale, resnormSpread, currentResnorm, selectedResnormRange, isModelSelected, visualResnorm]);


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
    
    // Draw header with circuit icon
    ctx.fillStyle = '#FFFFFF';
    ctx.font = 'bold 14px Arial';
    ctx.textAlign = 'left';

    // Draw circuit icon (simple resistor/capacitor symbol)
    const iconX = listX + 10;
    const iconY = listY + 8;
    const iconSize = 12;

    // Resistor symbol (zig-zag)
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(iconX, iconY + iconSize/2);
    ctx.lineTo(iconX + 2, iconY + 2);
    ctx.lineTo(iconX + 4, iconY + iconSize - 2);
    ctx.lineTo(iconX + 6, iconY + 2);
    ctx.lineTo(iconX + 8, iconY + iconSize - 2);
    ctx.lineTo(iconX + 10, iconY + iconSize/2);
    ctx.stroke();

    // Connection lines
    ctx.beginPath();
    ctx.moveTo(iconX - 2, iconY + iconSize/2);
    ctx.lineTo(iconX, iconY + iconSize/2);
    ctx.moveTo(iconX + 10, iconY + iconSize/2);
    ctx.lineTo(iconX + 12, iconY + iconSize/2);
    ctx.stroke();

    ctx.fillText('Circuit Configuration Matrix', listX + 25, listY + 18);
    
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

  // Throttled 3D rendering for optimal performance
  const render3D = React.useCallback(() => {
    const currentTime = performance.now();
    if (currentTime - lastRenderTime.current < renderThrottleMs) {
      return; // Skip render if too frequent
    }
    lastRenderTime.current = currentTime;
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

    // Draw ground truth cross-section line (before polygons for proper layering)
    drawGroundTruthCrossSection(ctx);

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
    
    // Draw pentagonal outline
    drawPentagonalOutline(ctx);

  }, [convert3DPolygons, project3D, referenceModel, actualWidth, actualHeight, draw3DRadarGrid, drawGroundTruthCrossSection, draw3DPolygon, draw3DRadarLabels, drawResnormAxis, drawParameterValuesList, drawPentagonalOutline, paramKeys]);

  // Professional 3D navigation handlers (Blender/Onshape-style) with model selection
  const handleMouseDown = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    
    // Check for shift-click resnorm selection (locks crosshair)
    if (e.button === 0 && e.shiftKey) {
      const selectedResnorm = findResnormAtPoint(clickX, clickY);
      if (selectedResnorm !== null) {
        // Lock crosshair to the discrete selected resnorm
        setIsModelSelected(true);
        setVisualResnorm(selectedResnorm);
        // Always send discrete resnorm for computational accuracy
        const nearestModel = findNearestGridModel(selectedResnorm);
        const discreteResnorm = nearestModel?.resnorm || selectedResnorm;
        if (onResnormSelect) {
          onResnormSelect(discreteResnorm);
        }
        return; // Don't start dragging
      }
    }
    
    // Check for ctrl-click to unlock crosshair (free scrolling mode)
    if (e.button === 0 && e.ctrlKey && !e.shiftKey) {
      // Unlock crosshair for free scrolling
      setIsModelSelected(false);
      setVisualResnorm(null);
      // Clear computational state
      lastComputationalResnorm.current = null;
      if (onCurrentResnormChange) {
        onCurrentResnormChange(null);
      }
      return; // Don't start dragging
    }
    
    // Check for model selection on left click without modifiers
    if (e.button === 0 && !e.shiftKey && !e.ctrlKey && onModelTag) {
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
    
    // Determine interaction mode based on mouse button and modifiers
    let dragMode: NavigationMode = 'orbit';
    if (e.button === 1) {
      dragMode = 'pan';
    } else if (e.button === 2) {
      dragMode = 'zoom';
    } else if (e.button === 0 && e.shiftKey) {
      // For shift+left click that didn't hit a resnorm point, use pan mode
      dragMode = 'pan';
    } else if (e.button === 0 && e.altKey) {
      // Use Alt+left click for zoom instead of Ctrl+left click
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

  // Precise crosshair detection with strict alignment validation
  const findSmoothResnormAtPoint = useCallback((screenX: number, screenY: number): number | null => {
    if (!filteredModels.length) return null;
    
    const resnorms = filteredModels.map(m => m.resnorm || 0).filter(r => r > 0);
    const minResnorm = Math.min(...resnorms);
    const maxResnorm = Math.max(...resnorms);
    const zAxisHeight = 5.0 * resnormSpread;
    
    let bestResnormMatch: number | null = null;
    let minDistanceToAxis = Infinity;
    
    // Balanced precision for smooth performance
    const numTestPoints = 75; // Optimized for performance while maintaining smoothness
    const testAngles = 1; // Single central axis for direct resnorm mapping
    
    for (let i = 0; i < numTestPoints; i++) {
      const normalizedZ = i / (numTestPoints - 1);
      const z = normalizedZ * zAxisHeight;
      const resnormValue = minResnorm + normalizedZ * (maxResnorm - minResnorm);
      
      // Test points along central Z-axis for direct resnorm mapping
      for (let angle = 0; angle < testAngles; angle++) {
        const testX = 0; // Central axis
        const testY = 0; // Central axis
        
        const testPoint = project3D({ x: testX, y: testY, z, color: '#000000', opacity: 1, model: null as any }); // eslint-disable-line @typescript-eslint/no-explicit-any
        if (testPoint.visible) {
          const distance = Math.sqrt(Math.pow(testPoint.x - screenX, 2) + Math.pow(testPoint.y - screenY, 2));
          // Stricter detection radius for precise alignment
          if (distance < 35 && distance < minDistanceToAxis) {
            minDistanceToAxis = distance;
            bestResnormMatch = resnormValue;
          }
        }
      }
    }
    
    // Only return match if it's close enough (strict alignment validation)
    return minDistanceToAxis < 30 ? bestResnormMatch : null;
  }, [filteredModels, resnormSpread, project3D]);

  // Legacy function for discrete resnorm selection (kept for backward compatibility)
  const findResnormAtPoint = useCallback((screenX: number, screenY: number): number | null => {
    return findSmoothResnormAtPoint(screenX, screenY);
  }, [findSmoothResnormAtPoint]);
  
  // Debounced computational update to parent (only discrete grid values)
  const updateComputationalResnorm = useCallback((visualValue: number) => {
    if (!onCurrentResnormChange) return;
    
    // Find nearest discrete grid model
    const nearestModel = findNearestGridModel(visualValue);
    const discreteResnorm = nearestModel?.resnorm || null;
    
    // Only update if significantly different from last computational value
    if (discreteResnorm !== null && 
        (!lastComputationalResnorm.current || 
         Math.abs(discreteResnorm - lastComputationalResnorm.current) >= computationalUpdateThreshold)) {
      
      lastComputationalResnorm.current = discreteResnorm;
      onCurrentResnormChange(discreteResnorm);
    }
  }, [onCurrentResnormChange, findNearestGridModel, computationalUpdateThreshold]);

  const handleMouseMove = (e: React.MouseEvent) => {
    // Handle hover detection for model highlighting and resnorm detection
    if (!interaction.isDragging) {
      const canvas = canvasRef.current;
      if (canvas) {
        const rect = canvas.getBoundingClientRect();
        const hoverX = e.clientX - rect.left;
        const hoverY = e.clientY - rect.top;
        
        // Throttled smooth crosshair scrolling and resnorm detection
        const currentTime = performance.now();
        if (models.length > 0 && canvas.offsetParent !== null && 
            (currentTime - lastMouseMoveTime.current) > mouseMoveThrottleMs) {
          
          lastMouseMoveTime.current = currentTime;
          
          // Precise crosshair scrolling with strict alignment (only when no model is selected)
          if (!isModelSelected) {
            const smoothResnorm = findSmoothResnormAtPoint(hoverX, hoverY);
            if (smoothResnorm !== null) {
              // Smooth interpolation for ultra-precise visual movement
              const currentVisual = visualResnorm || smoothResnorm;
              const diff = smoothResnorm - currentVisual;
              const interpolationFactor = 0.15; // Smooth interpolation
              const newVisualResnorm = Math.abs(diff) > 0.001 
                ? currentVisual + diff * interpolationFactor
                : smoothResnorm;
              
              setVisualResnorm(newVisualResnorm);
              
              // Debounced computational sync (discrete values only)
              updateComputationalResnorm(newVisualResnorm);
              
              // Set hovered model to nearest actual model for proper polygon highlighting
              const nearestModel = findNearestActualModel(newVisualResnorm);
              setHoveredModel(nearestModel);
            } else {
              // Clear visual crosshair and hover when outside strict detection area
              setVisualResnorm(null);
              setHoveredModel(null);
              // Clear computational state as well
              if (onCurrentResnormChange && lastComputationalResnorm.current !== null) {
                lastComputationalResnorm.current = null;
                onCurrentResnormChange(null);
              }
            }
          } else {
            // When model is selected, use nearest actual model for highlighting
            const activeResnorm = currentResnorm || visualResnorm;
            if (activeResnorm) {
              const nearestModel = findNearestActualModel(activeResnorm);
              setHoveredModel(nearestModel);
            } else {
              setHoveredModel(null);
            }
            
            // Legacy discrete resnorm detection when model is selected
            const hoveredResnorm = findResnormAtPoint(hoverX, hoverY);
            if (onCurrentResnormChange) {
              onCurrentResnormChange(hoveredResnorm);
            }
          }
        }
      }
      return;
    }

    // Throttle drag operations for smooth performance
    const currentDragTime = performance.now();
    if ((currentDragTime - lastDragTime.current) < dragThrottleMs) {
      // Update interaction state but skip heavy operations
      setInteraction(prev => ({ ...prev, lastMousePos: { x: e.clientX, y: e.clientY } }));
      return;
    }
    lastDragTime.current = currentDragTime;

    const deltaX = e.clientX - interaction.lastMousePos.x;
    const deltaY = e.clientY - interaction.lastMousePos.y;
    
    // Enhanced sensitivity with acceleration for ultra-smooth controls
    const baseSensitivity = 0.5;
    const accelerationFactor = Math.min(1.5, Math.sqrt(deltaX * deltaX + deltaY * deltaY) / 10);
    const sensitivity = baseSensitivity * accelerationFactor;

    if (interaction.dragMode === 'orbit') {
      // Ultra-smooth orbit with momentum-based smoothing
      const smoothedDeltaX = deltaX * sensitivity * 0.8; // Reduced for smoothness
      const smoothedDeltaY = deltaY * sensitivity * 0.8;
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
      const panSensitivity = 0.004 * camera.distance; // Slightly increased for responsiveness
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
      const zoomSensitivity = 0.01; // Slightly increased for responsiveness
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

  const handleMouseLeave = () => {
    // Clear hover states and stop dragging when mouse leaves the canvas
    setHoveredModel(null);
    // Clear computational state
    lastComputationalResnorm.current = null;
    if (onCurrentResnormChange) {
      onCurrentResnormChange(null);
    }
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

  // Legacy render scheduling (replaced by advanced frame scheduler above)

  // Main rendering trigger
  useEffect(() => {
    render3D();
  }, [render3D]);

  // Additional effect to handle visibility changes and navigation
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (!document.hidden && canvasRef.current) {
        // Re-render when page becomes visible again
        setTimeout(() => {
          render3D();
        }, 100);
      }
    };

    const handleFocus = () => {
      if (canvasRef.current) {
        // Re-render when canvas gains focus after navigation
        setTimeout(() => {
          render3D();
        }, 50);
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    const currentCanvas = canvasRef.current;
    if (currentCanvas) {
      currentCanvas.addEventListener('focus', handleFocus);
    }

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      if (currentCanvas) {
        currentCanvas.removeEventListener('focus', handleFocus);
      }
    };
  }, [render3D]);

  // Force re-render when component becomes visible
  useEffect(() => {
    if (containerRef.current) {
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting && entry.intersectionRatio > 0.1) {
            // Component is visible, ensure it renders
            setTimeout(() => {
              render3D();
            }, 100);
          }
        });
      }, { threshold: 0.1 });

      observer.observe(containerRef.current);

      return () => observer.disconnect();
    }
  }, [render3D]);

  // Remove complex scheduling - just trigger render on data changes

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
        onMouseLeave={handleMouseLeave}
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
              <div>Shift+Click: Select Resnorm Level</div>
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