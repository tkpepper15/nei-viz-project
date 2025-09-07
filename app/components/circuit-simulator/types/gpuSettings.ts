export interface GPUAccelerationSettings {
  enabled: boolean;
  preferWebGPU: boolean;
  fallbackToCPU: boolean;
  maxBatchSize: number;
  deviceType: 'discrete' | 'integrated' | 'cpu';
  enableProfiling: boolean;
  memoryThreshold: number; // MB
}

export interface CPUSettings {
  maxWorkers: number;
  chunkSize: number;
}

export interface ExtendedPerformanceSettings {
  useSymmetricGrid: boolean;
  maxComputationResults: number;
  gpuAcceleration: GPUAccelerationSettings;
  cpuSettings: CPUSettings;
}

export interface WebGPUCapabilities {
  supported: boolean;
  adapter: GPUAdapter | null;
  device: GPUDevice | null;
  features: string[];
  limits: Record<string, number>;
  deviceType: 'discrete' | 'integrated' | 'cpu' | 'unknown';
  vendor: string;
  architecture: string;
}

export interface WebGPUBenchmarkResult {
  totalTime: number;
  computeTime: number;
  dataTransferTime: number;
  parametersProcessed: number;
  parametersPerSecond: number;
  gpuMemoryUsed: number;
  cpuMemoryUsed: number;
}

export const DEFAULT_GPU_SETTINGS: GPUAccelerationSettings = {
  enabled: false, // Default to false until user explicitly enables
  preferWebGPU: true,
  fallbackToCPU: true,
  maxBatchSize: 65536, // 64K parameter sets per batch
  deviceType: 'discrete',
  enableProfiling: false,
  memoryThreshold: 1024 // 1GB
};

export const DEFAULT_CPU_SETTINGS: CPUSettings = {
  maxWorkers: navigator.hardwareConcurrency || 4,
  chunkSize: 5000
};