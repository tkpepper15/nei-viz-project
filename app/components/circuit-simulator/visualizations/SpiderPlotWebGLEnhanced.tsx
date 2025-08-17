"use client";

import React, { useRef, useMemo, useState, useEffect, Suspense } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Line, Text, Html, useProgress } from '@react-three/drei';
import { useSpring, animated, config } from '@react-spring/three';
import * as THREE from 'three';
import { ModelSnapshot } from '../types';
import { CircuitParameters, PARAMETER_RANGES, DEFAULT_GRID_SIZE, faradToMicroFarad } from '../types/parameters';

interface SpiderPlotWebGLEnhancedProps {
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
  enableAnimations?: boolean;
  enableAdvancedInteractions?: boolean;
}

// Loading fallback component
function LoadingFallback() {
  const { progress } = useProgress();
  
  return (
    <Html center>
      <div className="bg-black/80 text-white p-4 rounded-lg text-center">
        <div className="text-lg font-semibold mb-2">Loading WebGL Spider Plot</div>
        <div className="w-32 bg-gray-700 rounded-full h-2">
          <div 
            className="bg-blue-500 h-2 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
        <div className="text-sm text-gray-300 mt-2">{Math.round(progress)}%</div>
      </div>
    </Html>
  );
}

// Advanced performance monitoring with frame time histogram
function useAdvancedPerformanceMonitor() {
  const [fps, setFps] = useState(60);
  const [frameTime, setFrameTime] = useState(0);
  const [memoryUsage, setMemoryUsage] = useState(0);
  const [drawCalls, setDrawCalls] = useState(0);
  const frameCount = useRef(0);
  const lastTime = useRef(performance.now());
  const frameStartTime = useRef(0);
  const frameHistory = useRef<number[]>([]);

  useFrame(({ gl }) => {
    frameStartTime.current = performance.now();
    frameCount.current++;
    
    // Update FPS every 60 frames
    if (frameCount.current % 60 === 0) {
      const currentTime = performance.now();
      const actualFps = 60000 / (currentTime - lastTime.current);
      setFps(Math.round(actualFps));
      lastTime.current = currentTime;
      
      // Update memory usage if available
      const memory = (performance as Performance & { memory?: { usedJSHeapSize: number } }).memory;
      if (memory) {
        setMemoryUsage(memory.usedJSHeapSize / 1024 / 1024);
      }
      
      // Track draw calls
      setDrawCalls(gl.info.render.calls);
    }
  });

  useFrame(() => {
    const frameEnd = performance.now();
    const currentFrameTime = frameEnd - frameStartTime.current;
    setFrameTime(currentFrameTime);
    
    // Maintain frame time history for smoothing
    frameHistory.current.push(currentFrameTime);
    if (frameHistory.current.length > 30) {
      frameHistory.current.shift();
    }
  });

  const avgFrameTime = useMemo(() => {
    if (frameHistory.current.length === 0) return frameTime;
    return frameHistory.current.reduce((a, b) => a + b, 0) / frameHistory.current.length;
  }, [frameTime]);

  return { fps, frameTime: avgFrameTime, memoryUsage, drawCalls };
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

// GPU-optimized instanced spider points with level-of-detail (LOD)
function InstancedSpiderPoints({ 
  models, 
  opacityFactor, 
  visualizationMode,
  enableAnimations = true
}: { 
  models: ModelSnapshot[]; 
  opacityFactor: number;
  visualizationMode: 'color' | 'opacity';
  enableAnimations?: boolean;
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const { camera } = useThree();
  const circuitParams = useMemo(() => getCircuitParameters(), []);
  const paramKeys = useMemo(() => circuitParams.map(p => p.key), [circuitParams]);

  // Animate appearance of new points
  const { scale } = useSpring({
    scale: models.length > 0 ? 1 : 0,
    config: enableAnimations ? config.gentle : { duration: 0 }
  });

  // Process models into GPU-friendly format
  const { positions, colors, scales } = useMemo(() => {
    if (!models.length) return { positions: [], colors: [], scales: [] };

    const paramRanges = {
      Rsh: PARAMETER_RANGES.Rsh,
      Ra: PARAMETER_RANGES.Ra,
      Rb: PARAMETER_RANGES.Rb,
      Ca: PARAMETER_RANGES.Ca,
      Cb: PARAMETER_RANGES.Cb
    };

    // Calculate resnorm statistics
    const resnorms = models.map(m => m.resnorm || 0).filter(r => r > 0);
    const minResnorm = Math.min(...resnorms);
    const maxResnorm = Math.max(...resnorms);
    const resnormRange = maxResnorm - minResnorm;

    // Calculate quartile thresholds for consistent coloring
    const sortedResnorms = [...resnorms].sort((a, b) => a - b);
    const q25 = sortedResnorms[Math.floor(sortedResnorms.length * 0.25)];
    const q50 = sortedResnorms[Math.floor(sortedResnorms.length * 0.50)];
    const q75 = sortedResnorms[Math.floor(sortedResnorms.length * 0.75)];

    const positions: THREE.Vector3[] = [];
    const colors: THREE.Color[] = [];
    const scales: number[] = [];
    // Removed metadata array as it's not used in this version

    models.forEach(model => {
      const resnorm = model.resnorm || 0;
      
      // Calculate Z position (lower resnorm = better = lower Z)
      const normalizedResnorm = resnormRange > 0 
        ? (resnorm - minResnorm) / resnormRange 
        : 0;
      const zHeight = normalizedResnorm * 5.0;

      // Determine color based on performance quartiles
      let color: THREE.Color;
      if (visualizationMode === 'color') {
        if (resnorm <= q25) color = new THREE.Color('#22c55e'); // Green (best quartile)
        else if (resnorm <= q50) color = new THREE.Color('#f59e0b'); // Yellow (second quartile)
        else if (resnorm <= q75) color = new THREE.Color('#f97316'); // Orange (third quartile)
        else color = new THREE.Color('#ef4444'); // Red (worst quartile)
      } else {
        // Opacity-based visualization
        const intensity = 1.0 - normalizedResnorm * 0.7;
        color = new THREE.Color(0x3B82F6).multiplyScalar(intensity);
      }

      // Calculate polygon center from radar chart vertices
      let avgX = 0, avgY = 0;
      let validParams = 0;

      paramKeys.forEach((param, i) => {
        const value = model.parameters[param as keyof typeof model.parameters] as number;
        if (typeof value === 'number') {
          const range = paramRanges[param as keyof typeof paramRanges];
          
          // Logarithmic normalization for parameters
          const logMin = Math.log10(range.min);
          const logMax = Math.log10(range.max);
          const logValue = Math.log10(Math.max(range.min, Math.min(range.max, value)));
          const normalizedValue = (logValue - logMin) / (logMax - logMin);
          
          // Calculate radar chart position
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
        scales.push(Math.max(0.1, 1.0 - normalizedResnorm * 0.6));
      }
    });

    return { positions, colors, scales };
  }, [models, paramKeys, visualizationMode]);

  // Level-of-detail based on camera distance
  const [lodLevel, setLodLevel] = useState(0);
  
  useFrame(() => {
    if (!meshRef.current) return;
    
    const distance = camera.position.distanceTo(new THREE.Vector3(0, 0, 2.5));
    const newLodLevel = distance < 5 ? 0 : distance < 10 ? 1 : 2;
    
    if (newLodLevel !== lodLevel) {
      setLodLevel(newLodLevel);
    }
  });

  // Update instance matrices and colors
  useEffect(() => {
    if (!meshRef.current || positions.length === 0) return;

    const dummy = new THREE.Object3D();
    const currentScale = scale.get();
    
    positions.forEach((position, index) => {
      dummy.position.copy(position);
      
      // Apply scale with animation and LOD
      const instanceScale = scales[index] * opacityFactor * currentScale;
      const lodScale = lodLevel === 0 ? 1 : lodLevel === 1 ? 0.7 : 0.4;
      dummy.scale.setScalar(instanceScale * lodScale);
      
      dummy.updateMatrix();
      meshRef.current!.setMatrixAt(index, dummy.matrix);
      meshRef.current!.setColorAt(index, colors[index]);
    });
    
    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) {
      meshRef.current.instanceColor.needsUpdate = true;
    }
  }, [positions, colors, scales, opacityFactor, scale, lodLevel]);

  if (positions.length === 0) return null;

  // Choose geometry based on LOD
  const geometry = lodLevel === 0 
    ? <sphereGeometry args={[0.06, 12, 8]} />
    : lodLevel === 1 
    ? <sphereGeometry args={[0.06, 8, 6]} />
    : <sphereGeometry args={[0.06, 6, 4]} />;

  return (
    <animated.group scale={scale}>
      <instancedMesh ref={meshRef} args={[undefined, undefined, positions.length]}>
        {geometry}
        <meshStandardMaterial transparent opacity={0.8} />
      </instancedMesh>
    </animated.group>
  );
}

// Enhanced spider grid with smooth transitions
function AnimatedSpiderGrid({ 
  gridSize = 5,
  visible: _visible = true // eslint-disable-line @typescript-eslint/no-unused-vars 
}: { 
  gridSize?: number;
  visible?: boolean;
}) {
  const circuitParams = useMemo(() => getCircuitParameters(), []);
  const paramKeys = useMemo(() => circuitParams.map(p => p.key), [circuitParams]);

  // Removed opacity spring - using direct opacity in Line components

  const gridLines = useMemo(() => {
    const lines: THREE.Vector3[][] = [];
    const angleStep = (2 * Math.PI) / paramKeys.length;
    const maxRadius = 2.0;

    // Radial axes with enhanced styling
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
      
      levelPoints.push(levelPoints[0]); // Close polygon
      lines.push(levelPoints);
    }

    return lines;
  }, [paramKeys, gridSize]);

  return (
    <animated.group>
      {gridLines.map((points, index) => (
        <Line
          key={index}
          points={points}
          color={index < paramKeys.length ? "#6B7280" : "#4B5563"} // Brighter axes
          lineWidth={index < paramKeys.length ? 1.5 : 1}
          transparent
          opacity={index < paramKeys.length ? 0.5 : 0.3}
        />
      ))}
    </animated.group>
  );
}

// Smooth parameter labels with fade-in animations
function AnimatedParameterLabels({ 
  visible = true 
}: { 
  visible?: boolean 
}) {
  const circuitParams = useMemo(() => getCircuitParameters(), []);

  const { scale } = useSpring({
    scale: visible ? 1 : 0,
    config: config.gentle
  });

  const labelPositions = useMemo(() => {
    const angleStep = (2 * Math.PI) / circuitParams.length;
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
  }, [circuitParams]);

  return (
    <animated.group scale={scale}>
      {labelPositions.map((label) => (
        <Text
          key={label.key}
          position={label.position}
          fontSize={0.18}
          color="#F3F4F6"
          anchorX="center"
          anchorY="middle"
          outlineWidth={0.02}
          outlineColor="#000000"
          fontWeight="bold"
        >
          {label.name}
        </Text>
      ))}
    </animated.group>
  );
}

// Enhanced ground truth with pulsing animation
function AnimatedGroundTruthOverlay({ 
  groundTruthParams,
  visible = true 
}: { 
  groundTruthParams: CircuitParameters;
  visible?: boolean;
}) {
  const circuitParams = useMemo(() => getCircuitParameters(), []);
  const paramKeys = useMemo(() => circuitParams.map(p => p.key), [circuitParams]);

  // Pulsing animation for emphasis
  const { scale } = useSpring({
    from: { scale: 0.8 },
    to: async (next) => {
      if (!visible) {
        await next({ scale: 0 });
        return;
      }
      while (true) {
        await next({ scale: 1.1 });
        await next({ scale: 1.0 });
      }
    },
    config: config.slow
  });

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

    points.push(points[0]); // Close polygon
    return points;
  }, [groundTruthParams, paramKeys]);

  return (
    <animated.group scale={scale}>
      <Line
        points={groundTruthPoints}
        color="#FFFFFF"
        lineWidth={3}
        dashed
        dashSize={0.15}
        gapSize={0.05}
      />
      {groundTruthPoints.slice(0, -1).map((point, idx) => (
        <mesh key={idx} position={point}>
          <sphereGeometry args={[0.08]} />
          <meshStandardMaterial 
            color="#FFFFFF"
            emissive="#FFFFFF"
            emissiveIntensity={0.2}
          />
        </mesh>
      ))}
    </animated.group>
  );
}

// Advanced performance HUD with detailed metrics
function AdvancedPerformanceHUD() {
  const { fps, frameTime, memoryUsage, drawCalls } = useAdvancedPerformanceMonitor();
  
  const getPerformanceColor = (fps: number) => {
    if (fps >= 50) return '#22c55e'; // Green
    if (fps >= 30) return '#f59e0b'; // Yellow
    return '#ef4444'; // Red
  };

  return (
    <Html position={[3.5, 3, 0]}>
      <div className="bg-black/90 text-white p-3 rounded-lg text-xs font-mono border border-gray-600">
        <div className="mb-2 text-center font-semibold text-blue-400">WebGL Performance</div>
        <div className="space-y-1">
          <div style={{ color: getPerformanceColor(fps) }}>
            FPS: {fps}
          </div>
          <div>Frame: {frameTime.toFixed(1)}ms</div>
          <div>Memory: {memoryUsage.toFixed(1)}MB</div>
          <div>Draws: {drawCalls}</div>
        </div>
        <div className="mt-2 pt-2 border-t border-gray-600 text-gray-400">
          <div className="text-xs">WebGL + GPU Acceleration</div>
          <div className="text-xs">60fps Target: {fps >= 50 ? '✓' : '✗'}</div>
        </div>
      </div>
    </Html>
  );
}

// Enhanced controls with gesture support
function EnhancedControls({
  enableAdvancedInteractions = true
}: {
  enableAdvancedInteractions?: boolean;
}) {
  const controlsRef = useRef<any>(null); // eslint-disable-line @typescript-eslint/no-explicit-any
  const { camera } = useThree();

  // Simplified camera state for this version
  const cameraState = {
    position: [10, 10, 10] as [number, number, number],
    target: [0, 0, 2.5] as [number, number, number]
  };

  const { animatedPosition, animatedTarget } = useSpring({
    animatedPosition: cameraState.position,
    animatedTarget: cameraState.target,
    config: config.gentle
  });

  // Enhanced gesture controls (simplified for this version)
  useEffect(() => {
    if (!enableAdvancedInteractions) return;
    
    // Custom gesture handling could be added here
    // For now, using standard OrbitControls
  }, [enableAdvancedInteractions]);

  useFrame(() => {
    if (enableAdvancedInteractions) {
      camera.position.set(...animatedPosition.get());
      camera.lookAt(...animatedTarget.get());
    }
  });

  return (
    <OrbitControls 
      ref={controlsRef}
      enablePan={true}
      enableZoom={!enableAdvancedInteractions} // Use custom zoom if advanced
      enableRotate={true}
      dampingFactor={0.05}
      enableDamping
      maxDistance={50}
      minDistance={2}
      maxPolarAngle={Math.PI * 0.8}
      minPolarAngle={Math.PI * 0.1}
    />
  );
}

// Main enhanced scene
function EnhancedSpiderPlotScene({ 
  models, 
  opacityFactor, 
  visualizationMode, 
  gridSize, 
  includeLabels, 
  showGroundTruth, 
  groundTruthParams,
  enableAnimations,
  enableAdvancedInteractions
}: {
  models: ModelSnapshot[];
  opacityFactor: number;
  visualizationMode: 'color' | 'opacity';
  gridSize: number;
  includeLabels: boolean;
  showGroundTruth: boolean;
  groundTruthParams?: CircuitParameters;
  enableAnimations: boolean;
  enableAdvancedInteractions: boolean;
}) {
  return (
    <>
      {/* Enhanced lighting setup */}
      <ambientLight intensity={0.4} />
      <directionalLight position={[10, 10, 5]} intensity={1.0} castShadow />
      <pointLight position={[-10, -10, -5]} intensity={0.6} />
      <pointLight position={[5, -5, 5]} intensity={0.4} color="#4ECDC4" />

      {/* Animated spider grid */}
      <AnimatedSpiderGrid gridSize={gridSize} visible={true} />

      {/* Animated parameter labels */}
      {includeLabels && <AnimatedParameterLabels visible={true} />}

      {/* GPU-optimized spider points with LOD */}
      <InstancedSpiderPoints 
        models={models}
        opacityFactor={opacityFactor}
        visualizationMode={visualizationMode}
        enableAnimations={enableAnimations}
      />

      {/* Animated ground truth overlay */}
      {showGroundTruth && groundTruthParams && (
        <AnimatedGroundTruthOverlay 
          groundTruthParams={groundTruthParams}
          visible={showGroundTruth}
        />
      )}

      {/* Advanced performance monitoring */}
      <AdvancedPerformanceHUD />

      {/* Enhanced controls with gestures */}
      <EnhancedControls enableAdvancedInteractions={enableAdvancedInteractions} />
    </>
  );
}

// Main enhanced component
const SpiderPlotWebGLEnhanced: React.FC<SpiderPlotWebGLEnhancedProps> = ({
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
  height: _height = 600, // eslint-disable-line @typescript-eslint/no-unused-vars
  enableAnimations = true,
  enableAdvancedInteractions = true
}) => {
  // Intelligent model filtering with performance optimization
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
    
    // Simplified performance scaling for this version
    const dynamicLimit = maxPolygons;
    
    return validModels.slice(0, dynamicLimit);
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
            alpha: backgroundColor === 'transparent',
            // Enable additional WebGL optimizations
            stencil: false,
            depth: true,
            logarithmicDepthBuffer: true
          }}
          style={{ 
            background: backgroundColor === 'white' ? '#ffffff' : 
                       backgroundColor === 'black' ? '#000000' : '#0f172a'
          }}
          shadows
        >
          <Suspense fallback={<LoadingFallback />}>
            <EnhancedSpiderPlotScene
              models={visibleModels}
              opacityFactor={opacityFactor}
              visualizationMode={visualizationMode}
              gridSize={gridSize}
              includeLabels={includeLabels}
              showGroundTruth={showGroundTruth}
              groundTruthParams={groundTruthParams}
              enableAnimations={enableAnimations}
              enableAdvancedInteractions={enableAdvancedInteractions}
            />
          </Suspense>
        </Canvas>
      </div>

      {/* Enhanced control hints */}
      <div className="absolute bottom-4 right-4 bg-neutral-800/95 backdrop-blur-sm border border-neutral-700 rounded-lg p-3 text-xs text-neutral-200">
        <div className="flex items-center gap-2 mb-1">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
          <span className="font-semibold">WebGL Enhanced</span>
        </div>
        <div>Models: {visibleModels.length.toLocaleString()}</div>
        <div className="text-neutral-400 mt-1">
          {enableAdvancedInteractions ? 'Advanced' : 'Standard'} Controls
        </div>
        <div className="text-neutral-400">
          Drag: Rotate • Scroll: Zoom • Shift+Drag: Pan
        </div>
      </div>

      {/* Enhanced gradient legend */}
      <div className="absolute bottom-4 left-4 bg-neutral-800/95 backdrop-blur-sm border border-neutral-700 rounded-lg p-3">
        <div className="text-xs font-medium text-neutral-200 mb-2">Resnorm Performance</div>
        <div className="flex items-center gap-2">
          <div className="w-20 h-4 rounded bg-gradient-to-r from-green-500 via-yellow-500 via-orange-500 to-red-500 shadow-lg"></div>
          <div className="text-xs text-neutral-400">
            <span className="text-green-400">Best</span> → <span className="text-red-400">Worst</span>
          </div>
        </div>
        <div className="flex justify-between text-xs text-neutral-500 mt-1">
          <span>Q1</span>
          <span>Q2</span>
          <span>Q3</span>
          <span>Q4</span>
        </div>
        <div className="text-xs text-neutral-400 mt-2">
          {enableAnimations ? '✓ Animations' : '✗ Static'} • 
          {enableAdvancedInteractions ? '✓ Enhanced' : '✗ Basic'}
        </div>
      </div>
    </div>
  );
};

export default SpiderPlotWebGLEnhanced;