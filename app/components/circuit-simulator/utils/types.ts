import { CircuitParameters } from '../types/parameters';

export interface Complex {
    real: number;
    imag: number;
}

export interface BackendMeshPoint {
    id?: number;
    resnorm: number;
    spectrum: {
        freq: number;
        real: number;
        imag: number;
        mag: number;
        phase: number;
    }[];
    parameters: CircuitParameters;
    alpha?: number;
}

export interface ImpedancePoint {
    frequency: number;
    real: number;
    imaginary: number;
    magnitude: number;
    phase: number;
}

export interface ResnormGroup {
    range: [number, number];
    color: string;
    label: string;
    description: string;
    items: ModelSnapshot[];
}

export interface ModelSnapshot {
    id: string;
    name: string;
    parameters: CircuitParameters;
    data: ImpedancePoint[];
    color: string;
    isVisible: boolean;
    opacity: number;
    resnorm: number;
    timestamp?: number;
    ter?: number;
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

export interface GridParameterArrays {
    Rs: number[];
    Ra: number[];
    Ca: number[];
    Rb: number[];
    Cb: number[];
}

export interface RadarDataPoint {
    parameter: string;
    fullValue: number;
    displayValue: string;
    [key: string]: number | string;
}

export interface SpiderData {
    data: SpiderDataPoint[];
    states: SpiderState[];
}

export interface SpiderDataPoint {
    parameter: string;
    [key: string]: number | string;
}

export interface SpiderState {
    name: string;
    color: string;
    dataKey: string;
} 