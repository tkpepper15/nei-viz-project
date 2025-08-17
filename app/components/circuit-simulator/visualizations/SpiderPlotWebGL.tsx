"use client";

import React, { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Line, Text, Html } from '@react-three/drei';
import * as THREE from 'three';
import { ModelSnapshot } from '../types';
import { CircuitParameters, PARAMETER_RANGES, DEFAULT_GRID_SIZE, faradToMicroFarad } from '../types/parameters';

interface SpiderPlotWebGLProps {
  meshItems: ModelSnapshot[];
  referenceId?: string | null;
  opacityFactor: number;
  maxPolygons: number;
  onExportImage?: (canvas: HTMLCanvasElement) => void;
  visualizationMode?: 'color' | 'opacity';
  gridSize?: number;
  includeLabels?: boolean;
  backgroundColor?: 'transparent' | 'white' | 'black';
  groundTruthParams?: CircuitParameters;
  showGroundTruth?: boolean;
  width?: number;
  height?: number;
}

// Performance monitoring hook
function usePerformanceMonitor() {
  const [fps, setFps] = useState(60);
  const [frameTime, setFrameTime] = useState(0);
  const frameCount = useRef(0);
  const lastTime = useRef(performance.now());
  const frameStartTime = useRef(0);

  useFrame(() => {
    frameStartTime.current = performance.now();
    frameCount.current++;
    
    if (frameCount.current % 60 === 0) {
      const currentTime = performance.now();
      const actualFps = 60000 / (currentTime - lastTime.current);
      setFps(Math.round(actualFps));
      lastTime.current = currentTime;
    }
  });

  useFrame(() => {
    const frameEnd = performance.now();
    setFrameTime(frameEnd - frameStartTime.current);
  });

  return { fps, frameTime };
}

// Get dynamic circuit parameter configuration
function getCircuitParameters() {
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
}

// Instanced spider polygons for high performance
function InstancedSpiderPolygons({ 
  models, 
  opacityFactor, 
  visualizationMode 
}: { 
  models: ModelSnapshot[]; 
  opacityFactor: number;
  visualizationMode: 'color' | 'opacity';
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const circuitParams = useMemo(() => getCircuitParameters(), []);
  const paramKeys = useMemo(() => circuitParams.map(p => p.key), [circuitParams]);

  // Convert models to 3D positions and colors
  const { positions, colors, scales } = useMemo(() => {
    if (!models.length) return { positions: [], colors: [], scales: [] };

    const paramRanges = {
      Rsh: PARAMETER_RANGES.Rsh,
      Ra: PARAMETER_RANGES.Ra,
      Rb: PARAMETER_RANGES.Rb,
      Ca: PARAMETER_RANGES.Ca,
      Cb: PARAMETER_RANGES.Cb
    };

    // Calculate resnorm range for Z-axis and color coding
    const resnorms = models.map(m => m.resnorm || 0).filter(r => r > 0);
    const minResnorm = Math.min(...resnorms);
    const maxResnorm = Math.max(...resnorms);
    const resnormRange = maxResnorm - minResnorm;

    // Calculate quartile thresholds
    const sortedResnorms = [...resnorms].sort((a, b) => a - b);
    const q25 = sortedResnorms[Math.floor(sortedResnorms.length * 0.25)];
    const q50 = sortedResnorms[Math.floor(sortedResnorms.length * 0.50)];
    const q75 = sortedResnorms[Math.floor(sortedResnorms.length * 0.75)];

    const positions: THREE.Vector3[] = [];
    const colors: THREE.Color[] = [];
    const scales: number[] = [];

    models.forEach(model => {
      const resnorm = model.resnorm || 0;
      
      // Calculate Z position based on resnorm (lower resnorm = lower Z)
      const normalizedResnorm = resnormRange > 0 
        ? (resnorm - minResnorm) / resnormRange 
        : 0;
      const zHeight = normalizedResnorm * 5.0;

      // Determine color based on resnorm quartiles
      let color: THREE.Color;
      if (visualizationMode === 'color') {
        if (resnorm <= q25) color = new THREE.Color('#22c55e'); // Green (best)
        else if (resnorm <= q50) color = new THREE.Color('#f59e0b'); // Yellow
        else if (resnorm <= q75) color = new THREE.Color('#f97316'); // Orange
        else color = new THREE.Color('#ef4444'); // Red (worst)
      } else {
        color = new THREE.Color('#3B82F6'); // Blue for opacity mode
      }

      // Calculate radar polygon center position
      let avgX = 0, avgY = 0;
      let validParams = 0;

      paramKeys.forEach((param, i) => {
        const value = model.parameters[param as keyof typeof model.parameters] as number;
        if (typeof value === 'number') {
          const range = paramRanges[param as keyof typeof paramRanges];
          
          // Normalize parameter value (logarithmic)
          const logMin = Math.log10(range.min);
          const logMax = Math.log10(range.max);
          const logValue = Math.log10(Math.max(range.min, Math.min(range.max, value)));
          const normalizedValue = (logValue - logMin) / (logMax - logMin);
          
          // Calculate radar position
          const angle = (i * 2 * Math.PI) / paramKeys.length - Math.PI / 2;
          const radius = normalizedValue * 2.0;
          
          avgX += Math.cos(angle) * radius;
          avgY += Math.sin(angle) * radius;
          validParams++;
        }
      });

      if (validParams > 0) {
        avgX /= validParams;
        avgY /= validParams;
        
        positions.push(new THREE.Vector3(avgX, avgY, zHeight));
        colors.push(color);
        scales.push(Math.max(0.2, 1.0 - normalizedResnorm * 0.7));
      }
    });

    return { positions, colors, scales };
  }, [models, paramKeys, visualizationMode]);

  // Update instanced mesh when data changes
  useEffect(() => {
    if (!meshRef.current || positions.length === 0) return;

    const dummy = new THREE.Object3D();
    
    positions.forEach((position, index) => {
      dummy.position.copy(position);
      dummy.scale.setScalar(scales[index] * opacityFactor);
      dummy.updateMatrix();
      meshRef.current!.setMatrixAt(index, dummy.matrix);
      meshRef.current!.setColorAt(index, colors[index]);
    });
    
    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) {
      meshRef.current.instanceColor.needsUpdate = true;
    }
  }, [positions, colors, scales, opacityFactor]);

  if (positions.length === 0) return null;

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, positions.length]}>
      <sphereGeometry args={[0.05, 8, 6]} />
      <meshStandardMaterial transparent opacity={0.8} />
    </instancedMesh>
  );
}

// Spider grid with WebGL optimization
function SpiderGrid({ gridSize = 5 }: { gridSize?: number }) {
  const circuitParams = useMemo(() => getCircuitParameters(), []);
  const paramKeys = useMemo(() => circuitParams.map(p => p.key), [circuitParams]);

  const gridLines = useMemo(() => {
    const lines: THREE.Vector3[][] = [];
    const angleStep = (2 * Math.PI) / paramKeys.length;
    const maxRadius = 2.0;

    // Radial axes
    paramKeys.forEach((_, i) => {
      const angle = i * angleStep - Math.PI / 2;
      lines.push([
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(
          Math.cos(angle) * maxRadius,
          Math.sin(angle) * maxRadius,
          0
        )
      ]);
    });

    // Concentric grid levels
    for (let level = 1; level <= gridSize; level++) {
      const radius = (maxRadius * level) / gridSize;
      const levelPoints: THREE.Vector3[] = [];
      
      paramKeys.forEach((_, i) => {
        const angle = i * angleStep - Math.PI / 2;
        levelPoints.push(new THREE.Vector3(
          Math.cos(angle) * radius,
          Math.sin(angle) * radius,
          0
        ));
      });
      
      // Close the polygon
      levelPoints.push(levelPoints[0]);
      lines.push(levelPoints);
    }

    return lines;
  }, [paramKeys, gridSize]);

  return (
    <group>
      {gridLines.map((points, index) => (
        <Line
          key={index}
          points={points}
          color="#4B5563"
          lineWidth={1}
          transparent
          opacity={0.3}
        />
      ))}
    </group>
  );
}

// Parameter labels with 3D positioning
function ParameterLabels() {
  const circuitParams = useMemo(() => getCircuitParameters(), []);
  const paramKeys = useMemo(() => circuitParams.map(p => p.key), [circuitParams]);

  const labelPositions = useMemo(() => {
    const angleStep = (2 * Math.PI) / paramKeys.length;
    const labelDistance = 2.8;
    
    return circuitParams.map((param, i) => {
      const angle = i * angleStep - Math.PI / 2;
      return {
        position: new THREE.Vector3(
          Math.cos(angle) * labelDistance,
          Math.sin(angle) * labelDistance,
          0.5
        ),
        name: param.name,
        key: param.key
      };
    });
  }, [circuitParams, paramKeys]);

  return (
    <group>
      {labelPositions.map((label) => (
        <Text
          key={label.key}
          position={label.position}
          fontSize={0.2}
          color="#FFFFFF"
          anchorX="center"
          anchorY="middle"
          font="/fonts/Inter-Bold.woff"
        >
          {label.name}
        </Text>
      ))}
    </group>
  );
}

// Ground truth overlay
function GroundTruthOverlay({ 
  groundTruthParams 
}: { 
  groundTruthParams: CircuitParameters 
}) {
  const circuitParams = useMemo(() => getCircuitParameters(), []);
  const paramKeys = useMemo(() => circuitParams.map(p => p.key), [circuitParams]);

  const groundTruthPoints = useMemo(() => {
    const paramRanges = {
      Rsh: PARAMETER_RANGES.Rsh,
      Ra: PARAMETER_RANGES.Ra,
      Rb: PARAMETER_RANGES.Rb,
      Ca: PARAMETER_RANGES.Ca,
      Cb: PARAMETER_RANGES.Cb
    };

    const points: THREE.Vector3[] = [];
    const angleStep = (2 * Math.PI) / paramKeys.length;

    paramKeys.forEach((param, i) => {
      const angle = i * angleStep - Math.PI / 2;
      const range = paramRanges[param as keyof typeof paramRanges];
      const groundTruthValue = groundTruthParams[param as keyof CircuitParameters];
      
      // Calculate logarithmic position
      const logMin = Math.log10(range.min);
      const logMax = Math.log10(range.max);
      const logValue = Math.log10(groundTruthValue as number);
      const normalizedValue = (logValue - logMin) / (logMax - logMin);
      const clampedValue = Math.max(0, Math.min(1, normalizedValue));
      
      // Calculate position
      const radius = 2.0 * clampedValue;
      points.push(new THREE.Vector3(
        Math.cos(angle) * radius,
        Math.sin(angle) * radius,
        0
      ));
    });

    // Close the polygon
    points.push(points[0]);
    return points;
  }, [groundTruthParams, paramKeys]);

  return (
    <group>
      <Line
        points={groundTruthPoints}
        color="#FFFFFF"
        lineWidth={2}
        dashed
        dashSize={0.1}
        gapSize={0.05}
        transparent
        opacity={0.9}
      />
      {groundTruthPoints.slice(0, -1).map((point, index) => (
        <mesh key={index} position={point}>
          <sphereGeometry args={[0.06]} />
          <meshStandardMaterial color="#FFFFFF" />
        </mesh>
      ))}
    </group>
  );
}

// Removed AnimatedCamera - not used in basic WebGL version

// Performance HUD
function PerformanceHUD() {
  const { fps, frameTime } = usePerformanceMonitor();
  
  return (
    <Html position={[3, 3, 0]}>
      <div className="bg-black/80 text-white p-2 rounded text-xs font-mono">
        <div>FPS: {fps}</div>
        <div>Frame: {frameTime.toFixed(1)}ms</div>
      </div>
    </Html>
  );
}

// Main WebGL Spider Plot Scene
function SpiderPlotScene({ 
  models, 
  opacityFactor, 
  visualizationMode, 
  gridSize, 
  includeLabels, 
  showGroundTruth, 
  groundTruthParams 
}: {
  models: ModelSnapshot[];
  opacityFactor: number;
  visualizationMode: 'color' | 'opacity';
  gridSize: number;
  includeLabels: boolean;
  showGroundTruth: boolean;
  groundTruthParams?: CircuitParameters;
}) {
  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.6} />
      <directionalLight position={[10, 10, 5]} intensity={0.8} />
      <pointLight position={[-10, -10, -5]} intensity={0.4} />

      {/* Spider grid */}
      <SpiderGrid gridSize={gridSize} />

      {/* Parameter labels */}
      {includeLabels && <ParameterLabels />}

      {/* Instanced spider polygons for performance */}
      <InstancedSpiderPolygons 
        models={models}
        opacityFactor={opacityFactor}
        visualizationMode={visualizationMode}
      />

      {/* Ground truth overlay */}
      {showGroundTruth && groundTruthParams && (
        <GroundTruthOverlay groundTruthParams={groundTruthParams} />
      )}

      {/* Performance monitoring */}
      <PerformanceHUD />

      {/* Controls */}
      <OrbitControls 
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        dampingFactor={0.05}
        enableDamping
        maxDistance={20}
        minDistance={2}
      />
    </>
  );
}

// Main component
const SpiderPlotWebGL: React.FC<SpiderPlotWebGLProps> = ({
  meshItems,
  opacityFactor,
  maxPolygons,
  visualizationMode = 'color',
  gridSize = DEFAULT_GRID_SIZE,
  includeLabels = true,
  backgroundColor = 'transparent',
  groundTruthParams,
  showGroundTruth = false,
  width: _width = 800, // eslint-disable-line @typescript-eslint/no-unused-vars
  height: _height = 600 // eslint-disable-line @typescript-eslint/no-unused-vars
}) => {
  // Filter and limit models for performance
  const visibleModels = useMemo(() => {
    if (!meshItems?.length) return [];
    
    const validModels = meshItems.filter(model => 
      model.parameters && 
      typeof model.parameters.Rsh === 'number' &&
      typeof model.parameters.Ra === 'number' &&
      typeof model.parameters.Rb === 'number' &&
      typeof model.parameters.Ca === 'number' &&
      typeof model.parameters.Cb === 'number'
    );
    
    return validModels.slice(0, maxPolygons);
  }, [meshItems, maxPolygons]);

  const [cameraPosition] = useState<[number, number, number]>([10, 10, 10]);

  return (
    <div className="relative w-full h-full min-h-[500px] max-h-[800px] aspect-square max-w-4xl mx-auto">
      <div className="w-full h-full rounded-lg overflow-hidden border border-neutral-700/50 bg-gradient-to-br from-neutral-900/50 to-neutral-800/30">
        <Canvas
          camera={{ 
            position: cameraPosition, 
            fov: 45,
            near: 0.1,
            far: 1000
          }}
          gl={{ 
            antialias: true,
            powerPreference: "high-performance",
            alpha: backgroundColor === 'transparent'
          }}
          style={{ 
            background: backgroundColor === 'white' ? '#ffffff' : 
                       backgroundColor === 'black' ? '#000000' : '#0f172a'
          }}
        >
          <SpiderPlotScene
            models={visibleModels}
            opacityFactor={opacityFactor}
            visualizationMode={visualizationMode}
            gridSize={gridSize}
            includeLabels={includeLabels}
            showGroundTruth={showGroundTruth}
            groundTruthParams={groundTruthParams}
          />
        </Canvas>
      </div>

      {/* Control hints */}
      <div className="absolute bottom-4 right-4 bg-neutral-800/90 backdrop-blur-sm border border-neutral-700 rounded-lg p-3 text-xs text-neutral-200">
        <div>WebGL Optimized | Models: {visibleModels.length.toLocaleString()}</div>
        <div className="text-neutral-400">Drag: Rotate • Scroll: Zoom • Shift+Drag: Pan</div>
      </div>

      {/* Gradient legend */}
      <div className="absolute bottom-4 left-4 bg-neutral-800/90 backdrop-blur-sm border border-neutral-700 rounded-lg p-3">
        <div className="text-xs font-medium text-neutral-200 mb-2">Resnorm Scale</div>
        <div className="flex items-center gap-2">
          <div className="w-16 h-3 rounded bg-gradient-to-r from-green-500 via-yellow-500 to-red-500"></div>
          <div className="text-xs text-neutral-400">
            <span className="text-green-400">Low</span> → <span className="text-red-400">High</span>
          </div>
        </div>
        <div className="flex justify-between text-xs text-neutral-500 mt-1">
          <span>Best</span>
          <span>Worst</span>
        </div>
      </div>
    </div>
  );
};

export default SpiderPlotWebGL;