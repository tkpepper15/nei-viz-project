// WebGPU Type Definitions
declare global {
  interface Navigator {
    gpu?: GPU;
  }

  interface GPU {
    requestAdapter(options?: GPURequestAdapterOptions): Promise<GPUAdapter | null>;
  }

  interface GPURequestAdapterOptions {
    powerPreference?: 'low-power' | 'high-performance';
    forceFallbackAdapter?: boolean;
  }

  interface GPUAdapter {
    requestDevice(descriptor?: GPUDeviceDescriptor): Promise<GPUDevice>;
    info: GPUAdapterInfo;
    features: GPUSupportedFeatures;
    limits: GPUSupportedLimits;
  }

  interface GPUDeviceDescriptor {
    requiredFeatures?: string[];
    requiredLimits?: Record<string, number>;
  }

  interface GPUDevice {
    createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
    createShaderModule(descriptor: { code: string }): GPUShaderModule;
    createComputePipeline(descriptor: GPUComputePipelineDescriptor): GPUComputePipeline;
    createBindGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup;
    createCommandEncoder(): GPUCommandEncoder;
    queue: GPUQueue;
    destroy(): void;
    features: GPUSupportedFeatures;
    limits: GPUSupportedLimits;
  }

  interface GPUAdapterInfo {
    vendor: string;
    architecture: string;
    device: string;
    description: string;
  }

  interface GPUSupportedFeatures extends Set<string> {}

  interface GPUSupportedLimits {
    maxTextureDimension1D: number;
    maxTextureDimension2D: number;
    maxTextureDimension3D: number;
    maxTextureArrayLayers: number;
    maxBindGroups: number;
    maxStorageBufferBindingSize: number;
    maxComputeWorkgroupStorageSize: number;
    maxComputeInvocationsPerWorkgroup: number;
    maxComputeWorkgroupSizeX: number;
    maxComputeWorkgroupSizeY: number;
    maxComputeWorkgroupSizeZ: number;
    maxStorageBuffersPerShaderStage: number;
    maxBufferSize: number;
  }

  interface GPUBuffer {
    size: number;
    mapAsync(mode: number, offset?: number, size?: number): Promise<void>;
    getMappedRange(offset?: number, size?: number): ArrayBuffer;
    unmap(): void;
    destroy(): void;
  }

  interface GPUBufferDescriptor {
    size: number;
    usage: number;
    mappedAtCreation?: boolean;
  }

  interface GPUComputePipeline {
    getBindGroupLayout(index: number): GPUBindGroupLayout;
  }

  interface GPUComputePipelineDescriptor {
    layout: 'auto' | GPUPipelineLayout;
    compute: GPUProgrammableStage;
  }

  interface GPUProgrammableStage {
    module: GPUShaderModule;
    entryPoint: string;
  }

  interface GPUShaderModule {}

  interface GPUBindGroup {}

  interface GPUBindGroupDescriptor {
    layout: GPUBindGroupLayout;
    entries: GPUBindGroupEntry[];
  }

  interface GPUBindGroupEntry {
    binding: number;
    resource: GPUBindingResource;
  }

  interface GPUBindingResource {
    buffer?: GPUBuffer;
    offset?: number;
    size?: number;
  }

  interface GPUBindGroupLayout {}

  interface GPUPipelineLayout {}

  interface GPUCommandEncoder {
    beginComputePass(descriptor?: { label?: string }): GPUComputePassEncoder;
    copyBufferToBuffer(source: GPUBuffer, sourceOffset: number, destination: GPUBuffer, destinationOffset: number, size: number): void;
    finish(): GPUCommandBuffer;
  }

  interface GPUComputePassEncoder {
    setPipeline(pipeline: GPUComputePipeline): void;
    setBindGroup(index: number, bindGroup: GPUBindGroup): void;
    dispatchWorkgroups(workgroupCountX: number, workgroupCountY?: number, workgroupCountZ?: number): void;
    end(): void;
  }

  interface GPUCommandBuffer {}

  interface GPUQueue {
    submit(commandBuffers: GPUCommandBuffer[]): void;
    writeBuffer(buffer: GPUBuffer, bufferOffset: number, data: ArrayBufferView, dataOffset?: number, size?: number): void;
  }

  // Constants
  const GPUBufferUsage: {
    MAP_READ: number;
    MAP_WRITE: number;
    COPY_SRC: number;
    COPY_DST: number;
    INDEX: number;
    VERTEX: number;
    UNIFORM: number;
    STORAGE: number;
    INDIRECT: number;
    QUERY_RESOLVE: number;
  };

  const GPUMapMode: {
    READ: number;
    WRITE: number;
  };
}

export {};