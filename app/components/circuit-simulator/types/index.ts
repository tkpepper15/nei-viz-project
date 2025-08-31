import type { CircuitParameters } from './parameters';

// Export important types
export type { CircuitParameters };

// Core application types
export type VisualizationType = 'nyquist' | 'spider' | 'sensitivity' | 'residuals' | 'correlation';
export type ReferenceType = 'ground-truth' | 'custom';

// Complex number interface
export interface Complex {
  real: number;
  imag: number;
}

// Impedance data structures
export interface ImpedancePoint {
  real: number;
  imaginary: number;
  frequency: number;
  magnitude: number;
  phase: number;
}

// Model and state management
export interface ModelSnapshot {
  id: string;
  name: string;
  timestamp: number;
  parameters: CircuitParameters;
  data: ImpedancePoint[];
  resnorm?: number;
  color: string;
  isVisible: boolean;
  opacity: number;
  ter?: number;  // Total electrode resistance
}

export interface StateFolder {
  id: string;
  name: string;
  items: ModelSnapshot[];
  isVisible: boolean;
  isExpanded?: boolean;
  parentId?: string;
}

// Resnorm and analysis types
export interface ResnormSettings {
  lambda_penalty: number;
  penalty_method: 'zr_imag_hf' | 'jrr' | 'zr_imag_hf_and_cap_ratio' | 'zr_imag_hf_and_ct' | 'zr_imag_hf_and_ct_and_jrr' | 'membrane_ct_and_jrr' | 'cap_ratio_and_jrr' | 'membrane_ct_and_cap_ratio_and_jrr' | 'zr_imag_hf_and_jrr' | 'none';
  target_zr_imag_hf: number;
  target_jrr: number;
  target_cap_ratio: number;
  use_log10_ratio: boolean;
}

export interface ResnormGroup {
  range: [number, number];
  color: string;
  label: string;
  description: string;
  items: ModelSnapshot[];
}

export interface ResnormGroupItem {
  id: string;
  name: string;
  timestamp: number;
  parameters: CircuitParameters;
  data: ImpedancePoint[];
  resnorm?: number;
  color: string;
  isVisible: boolean;
  opacity: number;
  ter?: number;
  dataKey?: string;
}

// Parameter relationships
export interface ParameterRelationship {
  sourceParam: string;
  targetParam: string;
  relationship: 'proportional' | 'inverse' | 'custom';
  factor?: number;
  customFunction?: string;
}

// Spider plot visualization types
export interface SpiderDataPoint {
  parameter: string;
  [key: string]: string | number | undefined;  // For dynamic state keys (state0, state1, etc.)
}

export interface SpiderState {
  name: string;
  color: string;
  dataKey: string;
  resnorm?: number;
  isActive?: boolean;
}

export interface SpiderData {
  data: SpiderDataPoint[];
  states: SpiderState[];
}

// Radar chart visualization
export interface RadarDataPoint {
  parameter: string;
  fullValue: number;
  displayValue: string;
  [key: string]: number | string;
}

// Control component interfaces
export interface ControlProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  onChange: (value: number) => void;
  multiplier?: number;
}

export interface CompactControlProps {
  label: string;
  value: number;
  unit: string;
  onChange: (value: number) => void;
  multiplier?: number;
}

// Backend and computation types
export interface BackendMeshPoint {
  id?: number;
  parameters: CircuitParameters;
  resnorm: number;
  alpha?: number;
  spectrum: {
    freq: number;
    real: number;
    imag: number;
    mag: number;
    phase: number;
  }[];
}

export interface GridParameterArrays {
  Rsh: number[];
  Ra: number[];
  Rb: number[];
  Ca: number[];
  Cb: number[];
}

// Component props interfaces
export interface CircuitSimulatorProps {
  groundTruthDataset?: ImpedancePoint[];
}

// Performance tracking interfaces
export interface PipelinePhase {
  name: string;
  startTime: number;
  endTime?: number;
  duration?: number;
  description: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  details?: Record<string, unknown>;
}

export interface PerformanceLog {
  totalDuration: number;
  phases: PipelinePhase[];
  bottleneck: string;
  efficiency: {
    parallelization: number; // 0-1 scale
    memoryUsage: number; // MB
    throughput: number; // points/second
    cpuCores: number;
  };
  summary: {
    gridGeneration: number;
    impedanceComputation: number;
    resnormAnalysis: number;
    dataProcessing: number;
    rendering: number;
  };
} 