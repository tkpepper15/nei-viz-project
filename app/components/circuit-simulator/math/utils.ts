import { CircuitParameters } from '../types/parameters';

export function calculateTER(params: CircuitParameters): number {
  return (params.Rsh * (params.Ra + params.Rb)) / (params.Rsh + params.Ra + params.Rb);
}

export function calculateTimeConstants(params: CircuitParameters): { tauA: number; tauB: number } {
  return {
    tauA: params.Ra * params.Ca,
    tauB: params.Rb * params.Cb
  };
}