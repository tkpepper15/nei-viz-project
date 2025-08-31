/**
 * Minimal Database Service for Vercel Deployment
 * Contains only the essential types and functions needed for tagging system
 */

// Essential type definitions
export interface SessionEnvironment {
  nodeOptions: string
  computeWorkers: number
  memoryLimit: string
  cacheSize: string
  debugMode: boolean
  [key: string]: string | number | boolean | undefined
}

export interface VisualizationSettings {
  groupPortion: number
  selectedOpacityGroups: number[]
  visualizationType: 'spider2d' | 'spider3d' | 'nyquist'
  chromaEnabled: boolean
  resnormSpread: number
  useResnormCenter: boolean
  showLabels: boolean
  [key: string]: string | number | boolean | number[] | undefined
}

export interface PerformanceSettings {
  maxWorkers: number
  chunkSize: number
  memoryThreshold: number
  enableCaching: boolean
  adaptiveLimits: boolean
  [key: string]: string | number | boolean | undefined
}

// Minimal service implementation - just exports the types
export default class DatabaseService {
  // Placeholder for compatibility - actual functionality is in useSessionManagement
  static async getActiveSession() {
    throw new Error('Use useSessionManagement hook instead')
  }
  
  static async getUserSessions() {
    throw new Error('Use useSessionManagement hook instead')
  }
  
  static async updateSessionActivity() {
    throw new Error('Use useSessionManagement hook instead')
  }
  
  static async initializeUserSession() {
    throw new Error('Use useSessionManagement hook instead')
  }
}