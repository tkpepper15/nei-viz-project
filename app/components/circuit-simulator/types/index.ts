import { CircuitParameters } from './parameters';

export type VisualizationType = 'nyquist' | 'spider' | 'sensitivity' | 'residuals' | 'correlation';

export interface ImpedancePoint {
  real: number;
  imaginary: number;
  frequency: number;
  magnitude: number;
  phase: number;
}

export interface ModelSnapshot {
  id: string;
  name: string;
  timestamp: number;
  parameters: CircuitParameters;
  data: ImpedancePoint[];
  resnorm?: number;
  color: string;
  isVisible: boolean;
  ter?: number;  // Total electrode resistance
}

export interface StateFolder {
  id: string;
  name: string;
  items: ModelSnapshot[];
  isVisible: boolean;
}

export interface ResnormSettings {
  lambda_penalty: number;
  penalty_method: 'zr_imag_hf' | 'jrr' | 'zr_imag_hf_and_cap_ratio' | 'zr_imag_hf_and_ct' | 'zr_imag_hf_and_ct_and_jrr' | 'membrane_ct_and_jrr' | 'cap_ratio_and_jrr' | 'membrane_ct_and_cap_ratio_and_jrr' | 'zr_imag_hf_and_jrr' | 'none';
  target_zr_imag_hf: number;
  target_jrr: number;
  target_cap_ratio: number;
  use_log10_ratio: boolean;
}

export interface ParameterRelationship {
  sourceParam: string;
  targetParam: string;
  relationship: 'proportional' | 'inverse' | 'custom';
  factor?: number;
  customFunction?: string;
}

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
  states: Array<{
    name: string;
    color: string;
    dataKey: string;
    resnorm?: number;
    isActive: boolean;
  }>;
}

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

export type CompactControlProps = Omit<ControlProps, 'min' | 'max' | 'step'>;

export interface RadarDataPoint {
  parameter: string;
  [key: string]: number | string;
} 