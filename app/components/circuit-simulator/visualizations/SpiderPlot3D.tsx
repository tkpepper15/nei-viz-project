"use client";

import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { InformationCircleIcon } from '@heroicons/react/24/outline';
import { ModelSnapshot } from '../types';
import { PARAMETER_RANGES, DEFAULT_GRID_SIZE, faradToMicroFarad, CircuitParameters } from '../types/parameters';

// Brighten a hex color by adding a flat amount (0–1) to each channel.
function lightenHex(hex: string, amount: number): string {
  const num = parseInt(hex.replace('#', ''), 16);
  const r = Math.min(255, ((num >> 16) & 0xff) + Math.round(amount * 255));
  const g = Math.min(255, ((num >> 8) & 0xff) + Math.round(amount * 255));
  const b = Math.min(255, (num & 0xff) + Math.round(amount * 255));
  return `#${r.toString(16).padStart(2,'0')}${g.toString(16).padStart(2,'0')}${b.toString(16).padStart(2,'0')}`;
}

// Custom axis configuration — overrides the default Rsh/Ra/Ca/Rb/Cb circuit parameter axes.
// normRange must be in the same units as the value stored in model.parameters[key].
export interface SpiderAxisConfig {
  key: 'Rsh' | 'Ra' | 'Ca' | 'Rb' | 'Cb';
  name: string;
  normRange: { min: number; max: number };
}

// 3D Spider Plot Props
interface SpiderPlot3DProps {
  models: ModelSnapshot[];
  referenceModel?: ModelSnapshot | null;
  width?: number;
  height?: number;
  resnormSpread?: number;
  gridSize?: number;
  onRotationChange?: (rotation: { x: number; y: number; z: number }) => void;
  useResnormCenter?: boolean;
  responsive?: boolean;
  resnormRange?: { min: number; max: number } | null;
  onModelTag?: (model: ModelSnapshot, tag: string) => void;
  onModelSelect?: (model: ModelSnapshot | null) => void;
  taggedModels?: Map<string, string>;
  currentResnorm?: number | null;
  onResnormSelect?: (resnorm: number) => void;
  onCurrentResnormChange?: (resnorm: number | null) => void;
  highlightedModelId?: string | null;
  selectedResnormRange?: { min: number; max: number } | null;
  axisConfig?: SpiderAxisConfig[];
  modelColors?: Map<string, string>;
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

// Enhanced interaction state with momentum tracking
interface InteractionState {
  isDragging: boolean;
  dragMode: NavigationMode;
  lastMousePos: { x: number; y: number };
  isShiftPressed: boolean;
  isCtrlPressed: boolean;
  velocity: { x: number; y: number }; // Track mouse velocity for momentum
  lastUpdateTime: number; // For calculating momentum
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
  resnormSpread = 1.0,
  gridSize = DEFAULT_GRID_SIZE,
  onRotationChange,
  useResnormCenter = false,
  responsive = true,
  resnormRange = null,
  onModelTag,
  onModelSelect,
  taggedModels = new Map(),
  currentResnorm = null,
  onResnormSelect,
  onCurrentResnormChange,
  highlightedModelId = null,
  selectedResnormRange = null,
  axisConfig,
  modelColors,
}) => {

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Responsive sizing state
  const [actualWidth, setActualWidth] = useState(width);
  const [actualHeight, setActualHeight] = useState(height);
  
  const isComponentReady = true;

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
  
  // Get dynamic circuit parameters configuration — overridable via axisConfig prop
  const circuitParams = React.useMemo(() => {
    if (axisConfig) {
      return axisConfig.map(cfg => ({
        key: cfg.key,
        name: cfg.name,
        desc: cfg.name,
        range: cfg.normRange,
      }));
    }
    return getCircuitParameters();
  }, [axisConfig]);
  const paramKeys = React.useMemo(() => circuitParams.map(p => p.key), [circuitParams]);

  // Normalization ranges — matches model.parameters units for all axes
  const normRanges = React.useMemo(() => {
    if (axisConfig) {
      const r: Record<string, { min: number; max: number }> = {};
      axisConfig.forEach(cfg => { r[cfg.key] = cfg.normRange; });
      return r;
    }
    return {
      Rsh: PARAMETER_RANGES.Rsh,
      Ra:  PARAMETER_RANGES.Ra,
      Ca:  PARAMETER_RANGES.Ca,
      Rb:  PARAMETER_RANGES.Rb,
      Cb:  PARAMETER_RANGES.Cb,
    } as Record<string, { min: number; max: number }>;
  }, [axisConfig]);

  // Utility function to check if a point is inside a polygon with tolerance
  const isPointInPolygon = useCallback((x: number, y: number, polygon: {x: number; y: number}[], tolerance: number = 0): boolean => {
    if (polygon.length < 3) return false;

    // First check exact polygon containment
    let inside = false;
    let j = polygon.length - 1;

    for (let i = 0; i < polygon.length; i++) {
      if (((polygon[i].y > y) !== (polygon[j].y > y)) &&
          (x < (polygon[j].x - polygon[i].x) * (y - polygon[i].y) / (polygon[j].y - polygon[i].y) + polygon[i].x)) {
        inside = !inside;
      }
      j = i;
    }

    if (inside) return true;

    // If tolerance is provided, also check distance to edges and vertices
    if (tolerance > 0) {
      // Check distance to each vertex
      for (const vertex of polygon) {
        const dx = x - vertex.x;
        const dy = y - vertex.y;
        if (Math.sqrt(dx * dx + dy * dy) <= tolerance) {
          return true;
        }
      }

      // Check distance to each edge
      for (let i = 0; i < polygon.length; i++) {
        const j = (i + 1) % polygon.length;
        const p1 = polygon[i];
        const p2 = polygon[j];

        // Calculate distance from point to line segment
        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const lengthSquared = dx * dx + dy * dy;

        if (lengthSquared === 0) continue;

        // Project point onto line segment
        const t = Math.max(0, Math.min(1, ((x - p1.x) * dx + (y - p1.y) * dy) / lengthSquared));
        const projX = p1.x + t * dx;
        const projY = p1.y + t * dy;

        const distX = x - projX;
        const distY = y - projY;
        const distance = Math.sqrt(distX * distX + distY * distY);

        if (distance <= tolerance) {
          return true;
        }
      }
    }

    return false;
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
    isCtrlPressed: false,
    velocity: { x: 0, y: 0 },
    lastUpdateTime: performance.now()
  });
  
  // Model selection and tagging state
  const [hoveredModel, setHoveredModel] = useState<ModelSnapshot | null>(null);
  const [selectedModelForTag, setSelectedModelForTag] = useState<ModelSnapshot | null>(null);
  const [tagInputValue, setTagInputValue] = useState<string>('');

  // Tagging state
  const [isTagging, setIsTagging] = useState<boolean>(false);

  // Info tooltip state
  const [showInfoTooltip, setShowInfoTooltip] = useState<boolean>(false);
  
  // Separate visual and computational resnorm states
  const [visualResnorm, setVisualResnorm] = useState<number | null>(null); // For smooth UI interpolation only
  const [isModelSelected, setIsModelSelected] = useState<boolean>(false);
  
  // Refs for debounced computational updates
  const lastComputationalResnorm = useRef<number | null>(null);
  const computationalUpdateThreshold = 0.005; // Only update parent for significant changes
  
  // Enhanced performance throttling for ultra-smooth interactions
  const lastMouseMoveTime = useRef<number>(0);
  const lastDragTime = useRef<number>(0);
  const mouseMoveThrottleMs = 8; // ~125fps for improved smoothness
  const dragThrottleMs = 2; // ~500fps for ultra-buttery rotation
  const wheelThrottleMs = 16; // ~60fps for smooth wheel events
  const lastWheelTime = useRef<number>(0);
  
  // Performance optimization refs
  const lastRenderTime = useRef<number>(0);
  const renderThrottleMs = 16; // ~60fps cap for smooth rendering

  // Tag label bounding boxes for click hit testing, populated each render
  const tagLabelBoundsRef = useRef<Map<string, { x: number; y: number; w: number; h: number }>>(new Map());
  

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

    const params = paramKeys;

    // Calculate resnorm range for Z-axis scaling
    const resnorms = filteredModels.map(m => m.resnorm || 0).filter(r => r > 0);
    const minResnorm = Math.min(...resnorms);
    const maxResnorm = Math.max(...resnorms);
    const resnormRange = maxResnorm - minResnorm;

    // Compute quartile thresholds once outside the map (avoids O(n² log n))
    const sortedResnorms = [...resnorms].sort((a, b) => a - b);
    const q25 = sortedResnorms[Math.floor(sortedResnorms.length * 0.25)];
    const q50 = sortedResnorms[Math.floor(sortedResnorms.length * 0.50)];
    const q75 = sortedResnorms[Math.floor(sortedResnorms.length * 0.75)];

    return filteredModels.map(model => {
      const resnorm = model.resnorm || 0;

      if (!model.parameters) return null;

      const normalizedResnorm = resnormRange > 0
        ? (resnorm - minResnorm) / resnormRange
        : 0;
      const zHeight = normalizedResnorm * 5.0 * resnormSpread;

      // Use per-model color override when provided (e.g. for impedance/comparison mode),
      // otherwise fall back to quartile-based coloring.
      let color: string;
      const overrideColor = modelColors?.get(model.id);
      if (overrideColor) {
        color = overrideColor;
      } else {
        color = '#ef4444';
        if (resnorm <= q25) color = '#22c55e';
        else if (resnorm <= q50) color = '#f59e0b';
        else if (resnorm <= q75) color = '#f97316';
      }

      // Calculate radar polygon vertices
      const vertices = params.map((param, i) => {
        const value = model.parameters[param as keyof typeof model.parameters] as number;
        const range = normRanges[param];

        // Normalize parameter value to 0-1 range (logarithmic)
        const logMin = Math.log10(range.min);
        const logMax = Math.log10(range.max);
        const logValue = Math.log10(Math.max(range.min, Math.min(range.max, value)));
        const normalizedValue = (logValue - logMin) / (logMax - logMin);

        const angle = (i * 2 * Math.PI) / params.length - Math.PI / 2;
        const radius = normalizedValue * 2.0;

        return {
          x: Math.cos(angle) * radius,
          y: Math.sin(angle) * radius,
          z: zHeight,
          color: color,
          opacity: Math.max(0.2, 1.0 - normalizedResnorm * 0.7),
          model: model
        };
      });

      return {
        vertices,
        color: color,
        opacity: Math.max(0.2, 1.0 - normalizedResnorm * 0.7),
        model,
        zHeight
      };
    }).filter(polygon => polygon !== null);
  }, [filteredModels, resnormSpread, paramKeys, normRanges, modelColors]);


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

    // Improved depth testing - prevent models from disappearing
    if (z <= 0.01) return { x: 0, y: 0, visible: false, depth: 0 };

    // Dynamic perspective calculation adapted to camera distance
    const baseScale = Math.min(actualWidth, actualHeight) * 0.12;
    // Perspective factor that works across our 2-50 distance range
    const perspectiveFactor = Math.max(0.3, Math.min(2.0, 12 / camera.distance));
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

  // Pre-sorted polygons (front-to-back) — avoids O(n log n) sort on every click/hover
  const sortedPolygonsForHit = useMemo(
    () => [...convert3DPolygons].sort((a, b) => b.zHeight - a.zHeight),
    [convert3DPolygons]
  );

  // Function to find which model polygon contains a click point
  const findModelAtPoint = useCallback((clickX: number, clickY: number): ModelSnapshot | null => {
    const sortedPolygons = sortedPolygonsForHit;

    // Check tag label bounding boxes first — clicking a label selects its model
    for (const [modelId, bounds] of tagLabelBoundsRef.current) {
      if (clickX >= bounds.x && clickX <= bounds.x + bounds.w &&
          clickY >= bounds.y && clickY <= bounds.y + bounds.h) {
        const model = sortedPolygons.find(p => p.model.id === modelId)?.model ?? null;
        if (model) return model;
      }
    }

    // Use adaptive tolerance based on camera zoom - more tolerance when zoomed out
    const baseTolerance = 15; // Base pixel tolerance
    const scaledTolerance = baseTolerance / Math.max(0.5, camera.scale); // Adjust for zoom level

    // First pass: exact containment check
    for (const polygon of sortedPolygons) {
      const projectedVertices = polygon.vertices
        .map(vertex => project3D(vertex as Point3D))
        .filter(p => p.visible);

      if (projectedVertices.length >= 3) {
        const polygonPoints = projectedVertices.map(p => ({ x: p.x, y: p.y }));

        if (isPointInPolygon(clickX, clickY, polygonPoints, 0)) {
          return polygon.model;
        }
      }
    }

    // Second pass: check with tolerance if no exact match
    for (const polygon of sortedPolygons) {
      const projectedVertices = polygon.vertices
        .map(vertex => project3D(vertex as Point3D))
        .filter(p => p.visible);

      if (projectedVertices.length >= 3) {
        const polygonPoints = projectedVertices.map(p => ({ x: p.x, y: p.y }));

        if (isPointInPolygon(clickX, clickY, polygonPoints, scaledTolerance)) {
          return polygon.model;
        }
      }
    }

    return null;
  }, [sortedPolygonsForHit, project3D, isPointInPolygon, camera.scale, tagLabelBoundsRef]);

  // Draw 3D radar grid with dynamic scaling
  const draw3DRadarGrid = React.useCallback((ctx: CanvasRenderingContext2D) => {
    const params = paramKeys;
    
    ctx.save();
    ctx.strokeStyle = '#252529';
    ctx.lineWidth = Math.max(0.6, camera.scale * 0.8);
    ctx.globalAlpha = 0.35;

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
      // Faint filled base plane — anchoring element for 3D spatial reading
      ctx.beginPath();
      ctx.moveTo(baseVertices[0].x, baseVertices[0].y);
      for (let i = 1; i < baseVertices.length; i++) {
        ctx.lineTo(baseVertices[i].x, baseVertices[i].y);
      }
      ctx.closePath();
      ctx.fillStyle = 'rgba(30,30,36,0.18)';
      ctx.fill();
      ctx.stroke();
    }

    // Draw radial axes with improved visibility
    ctx.strokeStyle = '#252529';
    ctx.globalAlpha = Math.min(0.35, camera.scale * 0.25);
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

    // Calculate the full resnorm range from all filtered models
    const resnorms = filteredModels.map(m => m.resnorm || 0).filter(r => r > 0);
    if (resnorms.length === 0) return;

    // Get ground truth parameter positions (same calculation as in convert3DPolygons)
    const params = paramKeys;

    // Calculate ground truth vertices at different Z levels throughout the resnorm spectrum
    const numLevels = 30; // Increased levels for smoother line
    const groundTruthVertices = Array.from({ length: numLevels }, (_, levelIndex) => {
      // Calculate Z height for this level
      const normalizedZ = levelIndex / (numLevels - 1);
      const zHeight = normalizedZ * 5.0 * resnormSpread; // Match the Z scaling from convert3DPolygons

      // Calculate ground truth parameter positions at this Z level
      return params.map((param, i) => {
        const value = referenceModel.parameters[param as keyof typeof referenceModel.parameters] as number;
        const range = normRanges[param];

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
          color: '#9a9a9a',
          opacity: 1,
          model: referenceModel
        };
      });
    });

    // Ground truth cross-section — muted, dashed, quiet presence
    ctx.strokeStyle = '#555b63';
    ctx.lineWidth = Math.max(1.2, 2 * camera.scale);
    ctx.globalAlpha = 0.55;
    ctx.setLineDash([6, 4]);

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

    // Draw horizontal connecting lines at each Z level to create continuous 3D structure
    ctx.setLineDash([4, 2]); // Different dash pattern for horizontal lines
    ctx.globalAlpha = 0.7;
    ctx.lineWidth = Math.max(2, 3 * camera.scale);

    for (let levelIndex = 0; levelIndex < numLevels; levelIndex += 3) { // Every 3rd level for clarity
      const levelVertices = groundTruthVertices[levelIndex];
      const projectedVertices = levelVertices.map(v => project3D(v)).filter(p => p.visible);

      if (projectedVertices.length >= 3) {
        ctx.beginPath();
        ctx.moveTo(projectedVertices[0].x, projectedVertices[0].y);

        for (let i = 1; i < projectedVertices.length; i++) {
          ctx.lineTo(projectedVertices[i].x, projectedVertices[i].y);
        }
        ctx.closePath(); // Close the pentagon at this level
        ctx.stroke();
      }
    }

    // Redraw vertical axis lines — no glow
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 0.3;
    ctx.lineWidth = Math.max(0.6, 1.2 * camera.scale);
    ctx.setLineDash([]);

    // Vertical axis lines
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
  }, [referenceModel, filteredModels, resnormSpread, paramKeys, normRanges, project3D]);

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
    
    // Check if this model is hovered, tagged, highlighted, or selected for tagging
    const isHovered = hoveredModel?.id === polygon.model.id && !selectedModelForTag; // Don't show hover if model is selected
    const isTagged = taggedModels.has(polygon.model.id);
    const isHighlighted = highlightedModelId === polygon.model.id || selectedModelForTag?.id === polygon.model.id;
    const tagName = taggedModels.get(polygon.model.id);
    
    // Draw vertical lines from base to polygon (depth indicators) with adaptive styling
    ctx.strokeStyle = polygon.color;
    ctx.lineWidth = Math.max(0.5, depthFactor * camera.scale);
    ctx.globalAlpha = Math.max(0.1, polygon.opacity * 0.28 * depthFactor);
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
    
    // Draw filled wafer plate polygon
    const baseOpacity = Math.max(0., polygon.opacity * depthFactor * 0.8);
    const baseLineWidth = Math.max(0.6, 1.5 * depthFactor * camera.scale);

    const projectedOnly = projectedVertices.map(p => p.projected);
    ctx.beginPath();
    ctx.moveTo(projectedOnly[0].x, projectedOnly[0].y);
    for (let i = 1; i < projectedOnly.length; i++) {
      ctx.lineTo(projectedOnly[i].x, projectedOnly[i].y);
    }
    ctx.closePath();

    // Fill — hover/highlight/tagged use lightened version of own color, not global hues
    if (isHovered) {
      ctx.globalAlpha = Math.min(0.7, baseOpacity + 0.3);
      ctx.fillStyle = lightenHex(polygon.color, 0.3);
    } else if (isHighlighted) {
      ctx.globalAlpha = Math.min(0.75, baseOpacity + 0.35);
      ctx.fillStyle = lightenHex(polygon.color, 0.2);
    } else if (isTagged) {
      ctx.globalAlpha = Math.min(0.6, baseOpacity + 0.2);
      ctx.fillStyle = lightenHex(polygon.color, 0.15);
    } else {
      ctx.globalAlpha = Math.min(0.45, baseOpacity + 0.15);
      ctx.fillStyle = polygon.color;
    }
    ctx.fill();

    // Outline
    ctx.globalAlpha = Math.min(1.0, baseOpacity + 0.25);
    if (isHovered) {
      ctx.lineWidth = baseLineWidth * 1.8;
      ctx.strokeStyle = lightenHex(polygon.color, 0.3);
    } else if (isHighlighted) {
      ctx.lineWidth = baseLineWidth * 2.5;
      ctx.strokeStyle = '#ffffff';
    } else if (isTagged) {
      ctx.lineWidth = baseLineWidth * 1.4;
      ctx.strokeStyle = lightenHex(polygon.color, 0.2);
    } else {
      ctx.lineWidth = Math.max(0.5, baseLineWidth * 0.6);
      ctx.strokeStyle = polygon.color;
    }
    ctx.setLineDash([]);
    ctx.stroke();

    // Vertex dots
    if (isHovered) {
      ctx.fillStyle = lightenHex(polygon.color, 0.3);
      ctx.globalAlpha = Math.min(0.9, (polygon.opacity + 0.4) * depthFactor);
    } else if (isHighlighted) {
      ctx.fillStyle = '#ffffff';
      ctx.globalAlpha = 0.9;
    } else if (isTagged) {
      ctx.fillStyle = lightenHex(polygon.color, 0.2);
      ctx.globalAlpha = Math.min(0.8, (polygon.opacity + 0.35) * depthFactor);
    } else {
      ctx.fillStyle = polygon.color;
      ctx.globalAlpha = Math.min(0.8, (polygon.opacity + 0.3) * depthFactor);
    }

    const baseDotSize = Math.max(1.5, 3 * depthFactor * camera.scale);
    const dotSize = isHovered ? baseDotSize * 1.4 :
                    (isHighlighted ? baseDotSize * 1.8 :
                     (isTagged ? baseDotSize * 1.2 : baseDotSize));

    projectedOnly.forEach(vertex => {
      ctx.beginPath();
      ctx.arc(vertex.x, vertex.y, dotSize, 0, 2 * Math.PI);
      ctx.fill();
    });
    
    // Draw tag label if model is tagged - positioned outside the radar like parameter labels
    if (isTagged && tagName) {
      const polyCenterX = projectedOnly.reduce((sum, v) => sum + v.x, 0) / projectedOnly.length;
      const polyCenterY = projectedOnly.reduce((sum, v) => sum + v.y, 0) / projectedOnly.length;

      const screenCenter = project3D({ x: 0, y: 0, z: polygon.zHeight, color: '#000000', opacity: 1, model: polygon.model });
      if (screenCenter.visible) {
        const dirX = polyCenterX - screenCenter.x;
        const dirY = polyCenterY - screenCenter.y;
        const dirLength = Math.sqrt(dirX * dirX + dirY * dirY);

        if (dirLength > 0) {
          const labelDistance = Math.max(80, dirLength + 40);
          let labelX = screenCenter.x + (dirX / dirLength) * labelDistance;
          let labelY = screenCenter.y + (dirY / dirLength) * labelDistance;

          ctx.font = 'bold 11px Arial';
          const textWidth = ctx.measureText(tagName).width;
          const backgroundWidth = textWidth + 12;
          const backgroundHeight = 20;

          // Clamp label within canvas bounds with a small margin
          const margin = 4;
          labelX = Math.max(backgroundWidth / 2 + margin, Math.min(actualWidth - backgroundWidth / 2 - margin, labelX));
          labelY = Math.max(backgroundHeight / 2 + margin, Math.min(actualHeight - backgroundHeight / 2 - margin, labelY));

          // Store bounds for click hit testing
          tagLabelBoundsRef.current.set(polygon.model.id, {
            x: labelX - backgroundWidth / 2,
            y: labelY - backgroundHeight / 2,
            w: backgroundWidth,
            h: backgroundHeight
          });

          // Draw connecting line first (behind label)
          ctx.globalAlpha = 0.45;
          ctx.strokeStyle = lightenHex(polygon.color, 0.2);
          ctx.lineWidth = 0.8;
          ctx.setLineDash([2, 3]);
          ctx.beginPath();
          ctx.moveTo(polyCenterX, polyCenterY);
          ctx.lineTo(labelX - (dirX / dirLength) * (backgroundWidth / 2 + 5), labelY - (dirY / dirLength) * (backgroundHeight / 2 + 5));
          ctx.stroke();
          ctx.setLineDash([]);

          // Draw label background and text at full opacity
          ctx.globalAlpha = 1.0;
          ctx.fillStyle = 'rgba(13,13,15,0.92)';
          ctx.fillRect(labelX - backgroundWidth / 2, labelY - backgroundHeight / 2, backgroundWidth, backgroundHeight);
          ctx.strokeStyle = lightenHex(polygon.color, 0.2);
          ctx.lineWidth = 0.8;
          ctx.strokeRect(labelX - backgroundWidth / 2, labelY - backgroundHeight / 2, backgroundWidth, backgroundHeight);

          ctx.fillStyle = '#dddde2';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(tagName, labelX, labelY);
        }
      }
    }

    ctx.restore();
  }, [project3D, camera.scale, hoveredModel, taggedModels, highlightedModelId, selectedModelForTag, actualWidth, actualHeight, tagLabelBoundsRef]);

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
      
      // Draw label background
      ctx.font = '11px Arial';
      const textWidth = ctx.measureText(param.name).width;
      const backgroundWidth = Math.max(textWidth + 16, 72);

      ctx.fillStyle = 'rgba(13,13,15,0.88)';
      ctx.fillRect(labelX - backgroundWidth/2, labelY - 15, backgroundWidth, 32);
      ctx.strokeStyle = '#252529';
      ctx.lineWidth = 0.8;
      ctx.strokeRect(labelX - backgroundWidth/2, labelY - 15, backgroundWidth, 32);

      // Draw parameter name
      ctx.fillStyle = '#7a7a82';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(param.name, labelX, labelY - 6 * camera.scale);
      
      // Draw intelligent value markers (min/max/median only)
      // Always use log-scale median since all normalization is logarithmic
      const logMin = Math.log10(Math.max(1e-15, param.range.min));
      const logMax = Math.log10(Math.max(1e-15, param.range.max));
      const minValue  = param.range.min;
      const maxValue  = param.range.max;
      const medianValue = Math.pow(10, (logMin + logMax) / 2);

      const formatValue = (v: number) => {
        if (v >= 1000) return `${(v / 1000).toFixed(1)}k`;
        if (v >= 1)    return v.toFixed(0);
        if (v >= 0.01) return v.toFixed(2);
        return v.toExponential(1);
      };

      const minText    = formatValue(minValue);
      const medianText = formatValue(medianValue);
      const maxText    = formatValue(maxValue);
      
      // Draw compact value markers (min/median/max)
      ctx.fillStyle = '#454549';
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
      color: '#7a7a82',
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
    ctx.fillStyle = 'rgba(13,13,15,0.92)';
    ctx.fillRect(panelX - 10, panelY - 5, panelWidth + 20, panelHeight + 20);
    ctx.strokeStyle = '#2a2a33';
    ctx.lineWidth = 0.8;
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
        const paramRange = normRanges[param];
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
    drawPentagon(centerX, centerY, maxValues, '#252529', 0.8);
    
    // Draw ground truth pentagon if available
    if (referenceModel?.parameters) {
      const gtValues = params.map(param => referenceModel.parameters[param as keyof typeof referenceModel.parameters] as number);
      drawPentagon(centerX, centerY, gtValues, '#555b63', 1.2, true);
    }

    // Draw crosshair model pentagon (dynamic based on crosshair position)
    if (crosshairModel && crosshairModel.parameters) {
      const crosshairValues = params.map(param => (crosshairModel.parameters as any)[param as keyof CircuitParameters] as number); // eslint-disable-line @typescript-eslint/no-explicit-any
      const hasExactMatch = isModelSelected && crosshairModel !== null && !crosshairModel.id.startsWith('interpolated_');
      // Selected: white outline; crosshair: muted highlight
      const crosshairColor = hasExactMatch ? '#dddde2' : '#7a7a82';
      const lineWidth = hasExactMatch ? 2 : 1.2;
      drawPentagon(centerX, centerY, crosshairValues, crosshairColor, lineWidth);
    } else if (gridSize && gridSize > 1) {
      const gridValues = params.map(param => {
        const range = normRanges[param];
        const logMin = Math.log10(range.min);
        const logMax = Math.log10(range.max);
        const logMid = (logMin + logMax) / 2;
        return Math.pow(10, logMid);
      });
      drawPentagon(centerX, centerY, gridValues, '#3a3a46', 0.8);
    }

    // Draw parameter labels around the pentagon
    ctx.fillStyle = '#7a7a82';
    ctx.font = '9px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    params.forEach((param, i) => {
      const angle = (i * 2 * Math.PI) / params.length - Math.PI / 2;
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
      const crosshairColor = isModelSelected ? '#dddde2' : '#7a7a82';
      ctx.strokeStyle = crosshairColor;
      ctx.lineWidth = isModelSelected ? 1.5 : 0.8;
      ctx.beginPath();
      ctx.moveTo(panelX, legendY);
      ctx.lineTo(panelX + 8, legendY);
      ctx.stroke();
      ctx.fillStyle = '#7a7a82';
      const labelText = isModelSelected ? 'Selected' : 'Crosshair';
      ctx.fillText(labelText, panelX + 12, legendY + 1);
      legendY += 12;
    }

    // Ground truth indicator (if available)
    if (referenceModel?.parameters) {
      ctx.strokeStyle = '#555b63';
      ctx.lineWidth = 0.8;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(panelX, legendY);
      ctx.lineTo(panelX + 8, legendY);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#7a7a82';
      ctx.fillText('Truth', panelX + 12, legendY + 1);
      legendY += 12;
    }

    // Boundary indicator
    ctx.strokeStyle = '#252529';
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    ctx.moveTo(panelX, legendY);
    ctx.lineTo(panelX + 8, legendY);
    ctx.stroke();
    ctx.fillStyle = '#454549';
    ctx.fillText('Max', panelX + 12, legendY + 1);
    
    ctx.restore();
  }, [actualWidth, referenceModel, paramKeys, normRanges, gridSize, camera.scale, crosshairModel, isModelSelected]);

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
      ctx.strokeStyle = '#3a3a46';
      ctx.lineWidth = Math.max(0.8, 1.5 * camera.scale);
      ctx.globalAlpha = 0.4;
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
        ctx.strokeStyle = '#7a7a82';
        ctx.lineWidth = Math.max(1.2, 2 * camera.scale);
        ctx.globalAlpha = 0.7;
        ctx.setLineDash([]);
        
        // Draw vertical range line at center
        const bottomPoint = project3D({ x: coreX, y: coreY, z: minZ, color: '#7a7a82', opacity: 1, model: null as any }); // eslint-disable-line @typescript-eslint/no-explicit-any
        const topPoint = project3D({ x: coreX, y: coreY, z: maxZ, color: '#7a7a82', opacity: 1, model: null as any }); // eslint-disable-line @typescript-eslint/no-explicit-any
        
        if (bottomPoint.visible && topPoint.visible) {
          ctx.beginPath();
          ctx.moveTo(bottomPoint.x, bottomPoint.y);
          ctx.lineTo(topPoint.x, topPoint.y);
          ctx.stroke();
          
          // Add range end markers
          ctx.fillStyle = '#7a7a82';
          ctx.globalAlpha = 0.8;
          const markerSize = Math.max(2, 3 * camera.scale);
          
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
      const centerPoint = project3D({ x: coreX + 2.5, y: coreY, z: centerZ, color: '#7a7a82', opacity: 1, model: null as any }); // eslint-disable-line @typescript-eslint/no-explicit-any
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
      
      // Color coding — all muted, differ by brightness and dash pattern
      ctx.strokeStyle = isLocked ? '#dddde2' : (isHovering ? '#7a7a82' : '#454549');
      ctx.lineWidth = Math.max(1.2, isLocked ? 2.5 : (isHovering ? 2 : 1.5) * camera.scale);
      ctx.globalAlpha = isLocked ? 0.9 : (isHovering ? 0.65 : 0.45);
      ctx.setLineDash(isLocked ? [] : (isHovering ? [4, 4] : [2, 6]));
      
      // Draw 5-prong crosshair aligned with pentagon parameter directions
      const crosshairLength = 1.8; // Slightly larger crosshair
      
      // Calculate center point once for performance
      const centerPoint = project3D({ x: 0, y: 0, z: currentZ, color: '#7a7a82', opacity: 1, model: null as any }); // eslint-disable-line @typescript-eslint/no-explicit-any
      
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
        ctx.strokeStyle = '#3a3a46';
        ctx.lineWidth = 0.8;
        ctx.globalAlpha = 0.6;
        ctx.stroke();

        // Draw text at center
        ctx.fillStyle = '#454549';
        ctx.fillText(labelText, labelPoint.x, labelPoint.y);

        // Add small indicator for what this value represents
        if (i === 0) {
          ctx.fillStyle = '#3d7a60'; // viz-muted green for min (better)
          ctx.font = `${Math.max(8, fontSize - 2)}px Arial`;
          ctx.fillText('MIN', labelPoint.x, labelPoint.y + fontSize/2 + 3);
        } else if (i === numLabels - 1) {
          ctx.fillStyle = '#7a3a3a'; // muted error for max (worse)
          ctx.font = `${Math.max(8, fontSize - 2)}px Arial`;
          ctx.fillText('MAX', labelPoint.x, labelPoint.y + fontSize/2 + 3);
        }
      }
    }
    
    // Title removed as requested
    
    
    ctx.restore();
  }, [filteredModels, project3D, camera.scale, resnormSpread, currentResnorm, selectedResnormRange, isModelSelected, visualResnorm]);


  // Circuit configuration matrix moved to bottom panel - overlay removed from 3D view

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

    // Clear tag label bounds before repopulating during this render pass
    tagLabelBoundsRef.current.clear();

    // Render each 3D radar polygon
    sortedPolygons.forEach(polygon => {
      draw3DPolygon(ctx, polygon);
    });

    // Render reference model if provided
    if (referenceModel) {
      const params = paramKeys;

      // Create reference polygon at base level (z=0)
      const refVertices = params.map((param, i) => {
        const value = referenceModel.parameters[param as keyof typeof referenceModel.parameters] as number;
        const range = normRanges[param];
        
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
          color: '#9a9a9a',
          opacity: 1,
          model: referenceModel
        }))
        .filter(p => p.visible);

      if (projectedRefVertices.length === params.length) {
        ctx.save();
        ctx.strokeStyle = '#555b63';
        ctx.lineWidth = 1.2;
        ctx.setLineDash([5, 4]);
        ctx.globalAlpha = 0.6;

        ctx.beginPath();
        ctx.moveTo(projectedRefVertices[0].x, projectedRefVertices[0].y);
        for (let i = 1; i < projectedRefVertices.length; i++) {
          ctx.lineTo(projectedRefVertices[i].x, projectedRefVertices[i].y);
        }
        ctx.closePath();
        ctx.stroke();

        // Draw reference vertices as dots
        ctx.fillStyle = '#555b63';
        ctx.setLineDash([]);
        ctx.globalAlpha = 0.7;
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
    
    
    // Circuit configuration matrix moved to bottom panel - overlay removed
    
    // Draw pentagonal outline
    drawPentagonalOutline(ctx);

  }, [convert3DPolygons, project3D, referenceModel, actualWidth, actualHeight, draw3DRadarGrid, drawGroundTruthCrossSection, draw3DPolygon, draw3DRadarLabels, drawResnormAxis, drawPentagonalOutline, paramKeys, normRanges, tagLabelBoundsRef]);

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
    if (e.button === 0 && !e.shiftKey && !e.ctrlKey) {
      const clickedModel = findModelAtPoint(clickX, clickY);
      if (clickedModel) {
        // Only allow one model to be selected at a time
        // Clear previous selection if clicking on a different model
        if (selectedModelForTag?.id === clickedModel.id) {
          // Clicking same model again - deselect it
          setSelectedModelForTag(null);
          setTagInputValue('');
          if (onModelSelect) {
            onModelSelect(null);
          }
        } else {
          // Select new model (this automatically clears the previous one)
          setSelectedModelForTag(clickedModel);
          // Pre-fill with existing tag if available
          const existingTag = taggedModels.get(clickedModel.id);
          setTagInputValue(existingTag || '');
          // Notify parent of selection
          if (onModelSelect) {
            onModelSelect(clickedModel);
          }
        }
        return; // Don't start dragging
      } else {
        // Clicked empty space - clear selection
        setSelectedModelForTag(null);
        setTagInputValue('');
        if (onModelSelect) {
          onModelSelect(null);
        }
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
      isCtrlPressed: e.ctrlKey,
      velocity: { x: 0, y: 0 },
      lastUpdateTime: Date.now()
    });
  };

  // Granular model detection with adaptive precision and zoom-aware radius
  const findSmoothResnormAtPoint = useCallback((screenX: number, screenY: number): number | null => {
    if (!filteredModels.length) return null;

    const resnorms = filteredModels.map(m => m.resnorm || 0).filter(r => r > 0);
    const minResnorm = Math.min(...resnorms);
    const maxResnorm = Math.max(...resnorms);
    const zAxisHeight = 5.0 * resnormSpread;

    let bestResnormMatch: number | null = null;
    let minDistanceToAxis = Infinity;

    // Adaptive precision based on model density and zoom level
    const modelDensity = filteredModels.length / (maxResnorm - minResnorm || 1);
    const zoomFactor = Math.max(0.5, Math.min(3.0, 12 / camera.distance)); // Scale with zoom
    const basePrecision = Math.min(500, Math.max(100, Math.floor(modelDensity * 50)));
    const numTestPoints = Math.floor(basePrecision * zoomFactor); // More precision when zoomed in

    // Zoom-adaptive detection radius
    const baseRadius = 25;
    const detectionRadius = Math.floor(baseRadius / zoomFactor);
    const strictRadius = Math.floor(detectionRadius * 0.8);

    for (let i = 0; i < numTestPoints; i++) {
      const normalizedZ = i / (numTestPoints - 1);
      const z = normalizedZ * zAxisHeight;
      const resnormValue = minResnorm + normalizedZ * (maxResnorm - minResnorm);

      // Test along central axis for precise resnorm mapping
      const testX = 0;
      const testY = 0;

      const testPoint = project3D({ x: testX, y: testY, z, color: '#000000', opacity: 1, model: null as any }); // eslint-disable-line @typescript-eslint/no-explicit-any
      if (testPoint.visible) {
        const distance = Math.sqrt(Math.pow(testPoint.x - screenX, 2) + Math.pow(testPoint.y - screenY, 2));
        if (distance < detectionRadius && distance < minDistanceToAxis) {
          minDistanceToAxis = distance;
          bestResnormMatch = resnormValue;
        }
      }
    }

    // Return match if within strict validation radius
    return minDistanceToAxis < strictRadius ? bestResnormMatch : null;
  }, [filteredModels, resnormSpread, project3D, camera.distance]);

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
            // Try direct model hit detection first for better accuracy
            const directModel = findModelAtPoint(hoverX, hoverY);

            if (directModel) {
              // Direct hit on a model polygon - use its exact resnorm
              const modelResnorm = directModel.resnorm || 0;
              setVisualResnorm(modelResnorm);
              setHoveredModel(directModel);
              updateComputationalResnorm(modelResnorm);
            } else {
              // Fallback to smooth resnorm detection
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
      setInteraction(prev => ({
        ...prev,
        lastMousePos: { x: e.clientX, y: e.clientY },
        lastUpdateTime: currentDragTime
      }));
      return;
    }
    lastDragTime.current = currentDragTime;

    const deltaX = e.clientX - interaction.lastMousePos.x;
    const deltaY = e.clientY - interaction.lastMousePos.y;
    const currentTime = performance.now();
    const deltaTime = currentTime - interaction.lastUpdateTime;

    // Calculate velocity for momentum-based smoothing
    const currentVelocity = {
      x: deltaTime > 0 ? deltaX / deltaTime * 16.67 : 0, // Normalize to ~60fps
      y: deltaTime > 0 ? deltaY / deltaTime * 16.67 : 0
    };

    // Smooth velocity with exponential averaging
    const velocitySmoothing = 0.3;
    const prevVelocity = interaction.velocity || { x: 0, y: 0 };
    const smoothedVelocity = {
      x: prevVelocity.x * (1 - velocitySmoothing) + currentVelocity.x * velocitySmoothing,
      y: prevVelocity.y * (1 - velocitySmoothing) + currentVelocity.y * velocitySmoothing
    };

    // Enhanced sensitivity with velocity-based acceleration for ultra-smooth controls
    const velocity = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
    const baseSensitivity = 0.65; // Slightly increased for better responsiveness
    const velocityFactor = Math.min(1.8, 1 + (velocity / 18)); // Progressive acceleration
    const sensitivity = baseSensitivity * velocityFactor;

    if (interaction.dragMode === 'orbit') {
      // Ultra-smooth orbit with adaptive smoothing based on velocity
      const smoothingFactor = Math.max(0.7, 1 - (velocity / 50)); // Less smoothing for fast movements
      const smoothedDeltaX = deltaX * sensitivity * smoothingFactor;
      const smoothedDeltaY = deltaY * sensitivity * smoothingFactor;

      // Apply rotation with smooth interpolation
      setRotation(prev => {
        const targetX = Math.max(-89, Math.min(89, prev.x + smoothedDeltaY));
        const targetY = (prev.y + smoothedDeltaX) % 360;
        const newRotation = {
          x: prev.x + (targetX - prev.x) * 0.85, // Smooth interpolation
          y: prev.y + (targetY - prev.y) * 0.85,
          z: prev.z
        };
        if (onRotationChange) {
          onRotationChange(newRotation);
        }
        return newRotation;
      });
    } else if (interaction.dragMode === 'pan') {
      // Enhanced pan with velocity-aware smoothing
      const panSensitivity = 0.005 * camera.distance * (1 + velocity * 0.01);
      const smoothingFactor = Math.max(0.8, 1 - (velocity / 30));

      setCamera(prev => {
        const targetX = prev.target.x - deltaX * panSensitivity;
        const targetY = prev.target.y + deltaY * panSensitivity;
        return {
          ...prev,
          target: {
            x: prev.target.x + (targetX - prev.target.x) * smoothingFactor,
            y: prev.target.y + (targetY - prev.target.y) * smoothingFactor,
            z: prev.target.z
          }
        };
      });
    } else if (interaction.dragMode === 'zoom') {
      // Professional zoom by dragging with proper bounds
      const zoomSensitivity = 0.025 * (1 + velocity * 0.001);
      const zoomDelta = 1 + (deltaY * zoomSensitivity);

      setCamera(prev => {
        // Use our improved zoom bounds (2-50)
        const targetDistance = Math.max(2, Math.min(50, prev.distance * zoomDelta));
        return {
          ...prev,
          distance: prev.distance + (targetDistance - prev.distance) * 0.85 // Smooth interpolation
        };
      });
    }

    // Update interaction state with velocity tracking
    setInteraction(prev => ({
      ...prev,
      lastMousePos: { x: e.clientX, y: e.clientY },
      velocity: smoothedVelocity,
      lastUpdateTime: currentTime
    }));
  };

  const handleMouseUp = useCallback(() => {
    setInteraction(prev => ({
      ...prev,
      isDragging: false,
      velocity: { x: 0, y: 0 } // Reset velocity when drag ends
    }));
  }, []);

  const handleMouseLeave = useCallback(() => {
    // Clear hover states and stop dragging when mouse leaves the canvas
    setHoveredModel(null);
    // Clear computational state
    lastComputationalResnorm.current = null;
    if (onCurrentResnormChange) {
      onCurrentResnormChange(null);
    }
    setInteraction(prev => ({
      ...prev,
      isDragging: false,
      velocity: { x: 0, y: 0 } // Reset velocity when leaving canvas
    }));
  }, [onCurrentResnormChange]);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();

    // Throttle wheel events for smoother performance
    const currentTime = performance.now();
    if ((currentTime - lastWheelTime.current) < wheelThrottleMs) {
      return;
    }
    lastWheelTime.current = currentTime;

    // Enhanced zoom with progressive sensitivity
    const baseZoomSpeed = 0.20;
    const wheelDelta = Math.abs(e.deltaY);
    const progressiveMultiplier = Math.min(4.0, 1 + (wheelDelta / 60));
    const zoomSpeed = baseZoomSpeed * progressiveMultiplier;
    const zoomFactor = e.deltaY > 0 ? (1 + zoomSpeed) : (1 - zoomSpeed);

    if (e.shiftKey) {
      // Shift+scroll for smooth scale adjustment with easing
      setCamera(prev => {
        const newScale = Math.max(0.1, Math.min(10, prev.scale * zoomFactor));
        return {
          ...prev,
          scale: prev.scale + (newScale - prev.scale) * 0.8 // Smooth interpolation
        };
      });
    } else {
      // Normal scroll for smooth distance zoom with proper bounds
      setCamera(prev => {
        const newDistance = Math.max(2, Math.min(50, prev.distance * zoomFactor));
        return {
          ...prev,
          distance: prev.distance + (newDistance - prev.distance) * 0.9 // Smooth interpolation
        };
      });
    }
  }, [wheelThrottleMs]);

  // Enhanced keyboard shortcuts for smooth navigation
  const handleKeyDown = React.useCallback((e: KeyboardEvent) => {
    if (e.target !== canvasRef.current) return;

    // Adaptive speeds based on current camera distance and scale
    // const moveSpeed = 0.3 * camera.distance * 0.1; // Scale with distance (unused)
    const rotateSpeed = 3; // Reduced for smoother rotation
    
    switch (e.key.toLowerCase()) {
      case 'w': // Move forward (zoom in)
      case 'arrowup':
        setCamera(prev => {
          const currentDistance = prev.distance;
          const zoomStep = currentDistance * 0.08; // 8% for smooth keyboard input
          const newDistance = Math.max(2, currentDistance - zoomStep);
          return { ...prev, distance: newDistance };
        });
        break;
      case 's': // Move back (zoom out)
      case 'arrowdown':
        setCamera(prev => {
          const currentDistance = prev.distance;
          const zoomStep = currentDistance * 0.08; // 8% for smooth keyboard input
          const newDistance = Math.min(50, currentDistance + zoomStep);
          return { ...prev, distance: newDistance };
        });
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
      case '+': // Zoom in
      case '=': // Plus key without shift
        setCamera(prev => {
          const currentDistance = prev.distance;
          const zoomStep = currentDistance * 0.15; // 15% step
          const newDistance = Math.max(2, currentDistance - zoomStep);
          return { ...prev, distance: newDistance };
        });
        break;
      case '-': // Zoom out
      case '_': // Minus key with shift
        setCamera(prev => {
          const currentDistance = prev.distance;
          const zoomStep = currentDistance * 0.15; // 15% step
          const newDistance = Math.min(50, currentDistance + zoomStep);
          return { ...prev, distance: newDistance };
        });
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

  // Main rendering trigger
  useEffect(() => {
    render3D();
  }, [render3D]);

  // Additional effect to handle visibility changes and navigation
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (!document.hidden && canvasRef.current) {
        requestAnimationFrame(() => render3D());
      }
    };

    const handleFocus = () => {
      if (canvasRef.current) {
        requestAnimationFrame(() => render3D());
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
            requestAnimationFrame(() => render3D());
          }
        });
      }, { threshold: 0.1 });

      observer.observe(containerRef.current);

      return () => observer.disconnect();
    }
  }, [render3D]);

  // Remove complex scheduling - just trigger render on data changes

  return (
    <div ref={containerRef} className="relative bg-neutral-950 overflow-hidden w-full h-full" onWheel={handleWheel}>
      {!isComponentReady ? (
        <div className="absolute inset-0 bg-neutral-950 flex items-center justify-center">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
        </div>
      ) : null}
      {isComponentReady && (
        <canvas
          ref={canvasRef}
        width={actualWidth}
        height={actualHeight}
        className={`w-full h-full object-contain focus:outline-none ${
          interaction.isDragging
            ? interaction.dragMode === 'pan' ? 'cursor-move'
              : interaction.dragMode === 'zoom' ? 'cursor-ns-resize'
              : 'cursor-grabbing'
            : 'cursor-grab'
        }`}
        style={{ maxWidth: '100%', maxHeight: '100%' }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onContextMenu={handleContextMenu}
        onFocus={() => canvasRef.current?.focus()}
      />
      )}
      
      {/* Control Buttons */}
      <div className="absolute top-4 left-4 flex items-center gap-2 z-10">
        <button
          onClick={() => {
            setRotation({ x: -30, y: 45, z: 0 });
            setCamera(prev => ({ ...prev, distance: 12, target: { x: 0, y: 0, z: 2.5 }, scale: 1.0 }));
          }}
          className="p-1 bg-neutral-900/80 hover:bg-neutral-800 rounded text-neutral-500 hover:text-neutral-300 transition-colors border border-neutral-800 focus:outline-none"
          title="Reset View"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </button>

        <button
          onClick={() => {
            setCamera(prev => {
              // Professional 3D zoom: proper bounds with smooth scaling
              const currentDistance = prev.distance;
              const zoomStep = currentDistance * 0.15; // 15% of current distance
              const newDistance = Math.max(2, currentDistance - zoomStep);
              return {
                ...prev,
                distance: newDistance
              };
            });
          }}
          className="p-1 bg-neutral-900/80 hover:bg-neutral-800 rounded text-neutral-500 hover:text-neutral-300 transition-colors border border-neutral-800 focus:outline-none"
          title="Zoom In (Camera closer)"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v6m3-3H7" />
          </svg>
        </button>

        <button
          onClick={() => {
            setCamera(prev => {
              // Professional 3D zoom: proper bounds with smooth scaling
              const currentDistance = prev.distance;
              const zoomStep = currentDistance * 0.15; // 15% of current distance
              const newDistance = Math.min(50, currentDistance + zoomStep);
              return {
                ...prev,
                distance: newDistance
              };
            });
          }}
          className="p-1 bg-neutral-900/80 hover:bg-neutral-800 rounded text-neutral-500 hover:text-neutral-300 transition-colors border border-neutral-800 focus:outline-none"
          title="Zoom Out (Camera further)"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM13 10H7" />
          </svg>
        </button>

        <div className="relative">
          <button
            onClick={() => setShowInfoTooltip(!showInfoTooltip)}
            onMouseEnter={() => setShowInfoTooltip(true)}
            onMouseLeave={() => setShowInfoTooltip(false)}
            className="p-1 bg-surface hover:bg-neutral-700 rounded text-neutral-300 transition-colors border border-border"
          >
            <InformationCircleIcon className="w-4 h-4" />
          </button>

          {showInfoTooltip && (
            <div className="absolute top-full left-0 mt-2 bg-surface border border-border p-3 rounded text-neutral-200 text-xs whitespace-nowrap z-50">
              <div className="space-y-1">
                <div>Drag: Rotate • Shift+Drag: Pan • Scroll: Zoom</div>
                <div>Shift+Click: Select Resnorm Level</div>
                <div className="text-neutral-400">R: Reset • +/-: Zoom</div>
              </div>
            </div>
          )}
        </div>
      </div>
      

      
      {models.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/50">
          <div className="text-center p-8 bg-surface border border-border rounded-lg">
            <div className="text-sm mb-2 text-neutral-200">No 3D data available</div>
            <div className="text-xs text-neutral-400">Generate grid results to see 3D visualization</div>
          </div>
        </div>
      )}
      
      {models.length > 50000 && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/50">
          <div className="text-center p-8 bg-surface border border-border rounded-lg">
            <div className="text-sm font-medium text-neutral-200 mb-3">Exceeds Rendering Capacity</div>
            <div className="text-xs text-neutral-400 mb-2">
              {models.length.toLocaleString()} models — max 50,000
            </div>
            <div className="text-xs text-neutral-500">
              Reduce resnorm matching portion or grid size
            </div>
          </div>
        </div>
      )}

      {/* Inline Tag Input - Non-blocking */}
      {selectedModelForTag && onModelTag && (
        <div
          className="absolute bottom-4 left-4 z-30 bg-neutral-800 border border-neutral-600 rounded-lg p-3 shadow-xl"
          style={{ maxWidth: '350px' }}
        >
          <div className="flex items-center justify-between mb-2">
            <div className="text-neutral-200 text-sm">
              Tag Model
              <span className="text-neutral-400 text-xs ml-2">
                (Resnorm: {selectedModelForTag.resnorm?.toFixed(4) || 'N/A'})
              </span>
            </div>
            <button
              onClick={() => {
                setSelectedModelForTag(null);
                setTagInputValue('');
                if (onModelSelect) {
                  onModelSelect(null);
                }
              }}
              className="text-neutral-400 hover:text-neutral-200 text-lg leading-none"
              title="Close"
            >
              ×
            </button>
          </div>
          <input
            type="text"
            value={tagInputValue}
            onChange={(e) => setTagInputValue(e.target.value)}
            placeholder={`Model ${selectedModelForTag.id.slice(0, 8)}...`}
            className="w-full px-3 py-2 text-sm bg-neutral-800 border border-neutral-700 rounded text-neutral-200 placeholder-neutral-600 focus:outline-none focus:border-neutral-600"
            onKeyDown={async (e) => {
              if (e.key === 'Enter') {
                const tagName = tagInputValue.trim();
                if (tagName && selectedModelForTag && onModelTag) {
                  setIsTagging(true);
                  try {
                    onModelTag(selectedModelForTag, tagName);
                  } catch (error) {
                    console.error('Failed to tag model:', error);
                  } finally {
                    setIsTagging(false);
                  }
                }
                // Keep model selected after tagging, just clear the input
                setTagInputValue('');
              } else if (e.key === 'Escape') {
                // Escape clears selection entirely
                setSelectedModelForTag(null);
                setTagInputValue('');
                if (onModelSelect) {
                  onModelSelect(null);
                }
              }
            }}
            autoFocus
          />
          <div className="flex gap-2 mt-2">
            <button
              className="flex-1 px-3 py-1.5 text-xs bg-primary hover:bg-primary-dark disabled:bg-neutral-700 disabled:text-neutral-500 transition-colors rounded text-neutral-100"
              disabled={isTagging || !tagInputValue.trim()}
              onClick={async () => {
                const tagName = tagInputValue.trim();
                if (!tagName || !selectedModelForTag || !onModelTag) return;

                setIsTagging(true);
                try {
                  onModelTag(selectedModelForTag, tagName);
                } catch (error) {
                  console.error('Failed to tag model:', error);
                } finally {
                  setIsTagging(false);
                }
                // Keep model selected after tagging, just clear the input
                setTagInputValue('');
              }}
            >
              {isTagging ? 'Saving...' : 'Save Tag'}
            </button>
            <button
              className="px-3 py-1.5 text-xs bg-neutral-700 hover:bg-neutral-600 transition-colors rounded text-neutral-300"
              onClick={() => {
                setSelectedModelForTag(null);
                setTagInputValue('');
                if (onModelSelect) {
                  onModelSelect(null);
                }
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
};