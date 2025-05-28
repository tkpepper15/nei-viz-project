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
    items: ModelSnapshot[];
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
    opacity: number;
    ter?: number;
    dataKey?: string;
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