import { CircuitParameters } from './types/parameters';

export interface ImpedancePoint {
  real: number;
  imaginary: number;
  magnitude: number;
  phase: number;
  frequency: number;
}

export type ReferenceType = 'ground-truth' | 'custom';

export interface ModelSnapshot {
  id: string;
  name: string;
  parameters: CircuitParameters;
  resnorm?: number;
  color: string;
  timestamp: number;
  data: ImpedancePoint[];
  isVisible: boolean;
  opacity: number;
  ter?: number;
}

export interface StateFolder {
  id: string;
  name: string;
  items: ModelSnapshot[];
  isVisible: boolean;
  isExpanded?: boolean;
  timestamp?: number;
  parentId?: string;
}

export interface ResnormSettings {
  lambda_penalty: number;
  penalty_method: 'zr_imag_hf' | 'jrr' | 'zr_imag_hf_and_cap_ratio' | 'zr_imag_hf_and_ct' | 'zr_imag_hf_and_ct_and_jrr' | 'membrane_ct_and_jrr' | 'cap_ratio_and_jrr' | 'membrane_ct_and_cap_ratio_and_jrr' | 'zr_imag_hf_and_jrr' | 'none';
  target_zr_imag_hf: number;
  target_jrr: number;
  target_cap_ratio: number;
  use_log10_ratio: boolean;
}

export interface SpiderDataPoint {
  parameter: string;
  [key: string]: string | number | undefined;  // For dynamic state keys (state0, state1, etc.)
}

export interface SpiderState {
  name: string;
  color: string;
  dataKey: string;
}

export interface SpiderData {
  data: SpiderDataPoint[];
  states: SpiderState[];
}

// Control component props interfaces
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

export interface BackendMeshPoint {
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
  Rs: number[];
  Ra: number[];
  Rb: number[];
  Ca: number[];
  Cb: number[];
}

export interface CircuitSimulatorProps {
  groundTruthDataset?: ImpedancePoint[];
} 