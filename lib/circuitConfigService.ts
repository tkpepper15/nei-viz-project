import { CircuitParameters } from '../app/components/circuit-simulator/types/parameters';

export interface UISettings {
  activeTab: 'visualizer' | 'math' | 'data' | 'activity' | 'serialized' | 'sweeper';
  splitPaneHeight: number;
  opacityLevel: number;
  opacityExponent: number;
  logScalar: number;
  visualizationMode: 'color' | 'opacity';
  backgroundColor: 'transparent' | 'white' | 'black';
  showGroundTruth: boolean;
  includeLabels: boolean;
  maxPolygons: number;
  useSymmetricGrid: boolean;
  adaptiveLimit: boolean;
  maxMemoryUsage: number;
  referenceModelVisible: boolean;
  manuallyHidden: boolean;
  isMultiSelectMode: boolean;
  selectedCircuits: string[];
  windowPositions?: { [key: string]: { x: number; y: number; width: number; height: number } };
  sidebarCollapsed?: boolean;
  toolboxPositions?: { [key: string]: { x: number; y: number } };
}

export interface CircuitConfiguration {
  id: string;
  userId: string;
  name: string;
  description?: string;
  isPublic: boolean;
  circuitParameters: CircuitParameters;
  gridSize: number;
  minFreq: number;
  maxFreq: number;
  numPoints: number;
  isComputed: boolean;
  computationTime?: number;
  totalPoints?: number;
  validPoints?: number;
  computationResults?: unknown;
  uiSettings?: UISettings;
  createdAt: string;
  updatedAt: string;
}

export interface CreateCircuitConfigRequest {
  name: string;
  description?: string;
  circuitParameters: CircuitParameters;
  gridSize: number;
  minFreq: number;
  maxFreq: number;
  numPoints: number;
  uiSettings?: UISettings;
}

const CONFIGS_KEY = 'nei-viz-circuit-configs';

function loadConfigs(): CircuitConfiguration[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = localStorage.getItem(CONFIGS_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveConfigs(configs: CircuitConfiguration[]): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(CONFIGS_KEY, JSON.stringify(configs));
}

export class CircuitConfigService {
  static async getUserCircuitConfigurations(_userId: string): Promise<CircuitConfiguration[]> {
    return loadConfigs();
  }

  static async createCircuitConfiguration(
    userId: string,
    req: CreateCircuitConfigRequest,
  ): Promise<CircuitConfiguration> {
    const configs = loadConfigs();
    const config: CircuitConfiguration = {
      id: `config-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      userId,
      name: req.name,
      description: req.description,
      isPublic: false,
      circuitParameters: req.circuitParameters,
      gridSize: req.gridSize,
      minFreq: req.minFreq,
      maxFreq: req.maxFreq,
      numPoints: req.numPoints,
      isComputed: false,
      uiSettings: req.uiSettings,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };
    configs.push(config);
    saveConfigs(configs);
    return config;
  }

  static async updateCircuitConfiguration(
    configId: string,
    updates: Partial<CircuitConfiguration>,
  ): Promise<CircuitConfiguration> {
    const configs = loadConfigs();
    const idx = configs.findIndex(c => c.id === configId);
    if (idx === -1) throw new Error(`Circuit configuration ${configId} not found`);
    configs[idx] = { ...configs[idx], ...updates, updatedAt: new Date().toISOString() };
    saveConfigs(configs);
    return configs[idx];
  }

  static async deleteCircuitConfiguration(configId: string): Promise<boolean> {
    const configs = loadConfigs();
    const filtered = configs.filter(c => c.id !== configId);
    if (filtered.length === configs.length) return false;
    saveConfigs(filtered);
    return true;
  }

  static async deleteMultipleCircuitConfigurations(configIds: string[]): Promise<boolean> {
    const set = new Set(configIds);
    saveConfigs(loadConfigs().filter(c => !set.has(c.id)));
    return true;
  }

  static async getCircuitConfiguration(configId: string): Promise<CircuitConfiguration | null> {
    return loadConfigs().find(c => c.id === configId) ?? null;
  }

  static async updateUISettings(configId: string, uiSettings: UISettings): Promise<boolean> {
    try {
      await CircuitConfigService.updateCircuitConfiguration(configId, { uiSettings });
      return true;
    } catch {
      return false;
    }
  }
}
