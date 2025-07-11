import { CircuitParameters } from './parameters';

export interface SavedProfile {
  id: string;
  name: string;
  description?: string;
  created: number; // timestamp
  lastModified: number; // timestamp
  
  // Grid computation settings
  gridSize: number;
  minFreq: number;
  maxFreq: number;
  numPoints: number;
  
  // Circuit parameters
  groundTruthParams: CircuitParameters;
  
  // Computation status
  isComputed: boolean;
  computationTime?: number; // time taken to compute in seconds
  totalPoints?: number; // total computed points
  validPoints?: number; // valid computed points
}

export interface SavedProfilesState {
  profiles: SavedProfile[];
  selectedProfile: string | null;
} 