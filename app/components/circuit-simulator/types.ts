export interface ImpedancePoint {
  real: number;
  imaginary: number;
  frequency: number;
  magnitude: number;
  phase: number;
}

export interface CircuitParameters {
  Rs: number;
  ra: number;
  ca: number;
  rb: number;
  cb: number;
}

export type ReferenceType = 'ground-truth' | 'custom';

export interface ModelSnapshot {
  id: string;
  name: string;
  parameters: {
    Rs: number;
    ra: number;
    ca: number;
    rb: number;
    cb: number;
    R_blank: number;
    frequency_range: [number, number];
  };
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
  parameters: {
    Rs: number;
    ra: number;
    ca: number;
    rb: number;
    cb: number;
    frequency_range: number[];
  };
  resnorm: number;
  alpha: number;
}

export interface ResnormGroupItem {
  id: string;
  name: string;
  parameters: {
    Rs: number;
    ra: number;
    ca: number;
    rb: number;
    cb: number;
    [key: string]: number;
  };
  resnorm?: number;
  color: string;
  timestamp: number;
  data: ImpedancePoint[];
  isVisible: boolean;
  opacity: number;
  ter?: number;
}

export interface ResnormGroup {
  range: [number, number];
  color: string;
  items: ResnormGroupItem[];
}

export interface CircuitSimulatorProps {
  groundTruthDataset?: ImpedancePoint[];
} 