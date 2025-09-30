import { CircuitParameters } from '../types/parameters';
import { WebGPUCapabilities, WebGPUBenchmarkResult, GPUAccelerationSettings } from '../types/gpuSettings';
import { getWebGPUManagerLimit } from './centralizedLimits';

export interface WebGPUComputeResult {
  results: Array<{
    parameters: CircuitParameters;
    spectrum: Array<{ freq: number; real: number; imag: number; mag: number; phase: number }>;
    resnorm: number;
  }>;
  benchmarkData: WebGPUBenchmarkResult;
  memoryUsed: number;
}

export class WebGPUManager {
  private adapter: GPUAdapter | null = null;
  private device: GPUDevice | null = null;
  private computeShader: string | null = null;
  private isInitialized: boolean = false;
  private capabilities: WebGPUCapabilities | null = null;

  async initialize(): Promise<WebGPUCapabilities> {
    const startTime = performance.now();
    
    // Check WebGPU support
    if (!navigator.gpu) {
      return this.createUnsupportedCapabilities('WebGPU not available in this browser');
    }

    try {
      // Request adapter with configurable power preference
      this.adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance', // Will be configurable via settings
        forceFallbackAdapter: false
      });

      if (!this.adapter) {
        return this.createUnsupportedCapabilities('No WebGPU adapter available');
      }

      // Request device with required features
      this.device = await this.adapter.requestDevice({
        requiredFeatures: [],
        requiredLimits: {
          maxComputeWorkgroupSizeX: 256,
          maxComputeWorkgroupSizeY: 1,
          maxComputeWorkgroupSizeZ: 1,
          maxComputeInvocationsPerWorkgroup: 256
        }
      });

      // Load compute shader
      this.computeShader = await this.loadComputeShader();
      
      this.isInitialized = true;
      
      // Create capabilities object
      this.capabilities = {
        supported: true,
        adapter: this.adapter,
        device: this.device,
        features: Array.from(this.adapter.features),
        limits: this.getLimitsObject(this.adapter.limits),
        deviceType: this.getDeviceType(this.adapter.info || {}),
        vendor: this.adapter.info?.vendor || 'unknown',
        architecture: this.adapter.info?.architecture || 'unknown'
      };

      console.log('SUCCESS: WebGPU initialized successfully:', {
        device: this.capabilities.deviceType,
        vendor: this.capabilities.vendor,
        features: this.capabilities.features.length,
        initTime: `${(performance.now() - startTime).toFixed(2)}ms`
      });

      return this.capabilities;

    } catch (error) {
      console.warn('ERROR: WebGPU initialization failed:', error);
      return this.createUnsupportedCapabilities(error instanceof Error ? error.message : 'Unknown error');
    }
  }

  private async loadComputeShader(): Promise<string> {
    try {
      const response = await fetch('/webgpu-compute.wgsl');
      if (!response.ok) {
        throw new Error(`Failed to load shader: ${response.status}`);
      }
      return await response.text();
    } catch {
      // Fallback to inline shader if file loading fails
      console.warn('Failed to load external shader, using inline version');
      return this.getInlineShader();
    }
  }

  private getInlineShader(): string {
    // Simplified inline version of the compute shader
    return `
      struct CircuitParams {
        Rs: f32, Ra: f32, Ca: f32, Rb: f32, Cb: f32,
        pad0: f32, pad1: f32, pad2: f32,
      };
      
      struct ComplexNumber { real: f32, imag: f32, };
      
      struct ImpedanceResult {
        frequency: f32, real: f32, imag: f32, magnitude: f32,
        phase: f32, resnorm: f32, pad0: f32, pad1: f32,
      };
      
      @group(0) @binding(0) var<storage, read> circuit_parameters: array<CircuitParams>;
      @group(0) @binding(1) var<storage, read> frequencies: array<f32>;
      @group(0) @binding(2) var<storage, read> reference_spectrum: array<ComplexNumber>;
      @group(0) @binding(3) var<storage, read_write> results: array<ImpedanceResult>;
      @group(1) @binding(0) var<uniform> config: vec4<u32>;
      
      fn complex_divide(a: ComplexNumber, b: ComplexNumber) -> ComplexNumber {
        let denom = b.real * b.real + b.imag * b.imag;
        return ComplexNumber((a.real * b.real + a.imag * b.imag) / denom,
                            (a.imag * b.real - a.real * b.imag) / denom);
      }
      
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let param_idx = global_id.x;
        if (param_idx >= config.x) { return; }
        
        let params = circuit_parameters[param_idx];
        let freq = frequencies[0];
        let omega = 2.0 * 3.141592653589793 * freq;
        
        // Za = Ra/(1+jωRaCa)
        let za = complex_divide(ComplexNumber(params.Ra, 0.0), 
                               ComplexNumber(1.0, omega * params.Ra * params.Ca));
        
        // Zb = Rb/(1+jωRbCb)
        let zb = complex_divide(ComplexNumber(params.Rb, 0.0),
                               ComplexNumber(1.0, omega * params.Rb * params.Cb));
        
        // Z = Rs + Za + Zb
        let z_real = params.Rs + za.real + zb.real;
        let z_imag = za.imag + zb.imag;
        let magnitude = sqrt(z_real * z_real + z_imag * z_imag);
        let phase = atan2(z_imag, z_real) * (180.0 / 3.141592653589793);
        
        results[param_idx] = ImpedanceResult(freq, z_real, z_imag, magnitude, phase, 0.0, 0.0, 0.0);
      }
    `;
  }

  private createUnsupportedCapabilities(reason: string): WebGPUCapabilities {
    console.warn(`WARNING: WebGPU not supported: ${reason}`);
    return {
      supported: false,
      adapter: null,
      device: null,
      features: [],
      limits: {},
      deviceType: 'unknown',
      vendor: 'unknown',
      architecture: 'unknown'
    };
  }

  private getLimitsObject(limits: GPUSupportedLimits): Record<string, number> {
    return {
      maxTextureDimension1D: limits.maxTextureDimension1D,
      maxTextureDimension2D: limits.maxTextureDimension2D,
      maxTextureDimension3D: limits.maxTextureDimension3D,
      maxTextureArrayLayers: limits.maxTextureArrayLayers,
      maxBindGroups: limits.maxBindGroups,
      maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
    };
  }

  private getDeviceType(info: GPUAdapterInfo): 'discrete' | 'integrated' | 'cpu' | 'unknown' {
    const description = (info.description || '').toLowerCase();
    if (description.includes('discrete') || description.includes('nvidia') || description.includes('radeon')) {
      return 'discrete';
    }
    if (description.includes('integrated') || description.includes('intel')) {
      return 'integrated';
    }
    if (description.includes('cpu') || description.includes('software')) {
      return 'cpu';
    }
    return 'unknown';
  }

  async computeCircuitGrid(
    circuitParameters: CircuitParameters[],
    frequencies: number[],
    referenceSpectrum: Array<{ freq: number; real: number; imag: number; mag: number; phase: number }>,
    settings: GPUAccelerationSettings,
    maxResults: number = getWebGPUManagerLimit(),
    progressCallback?: (progress: number) => void
  ): Promise<WebGPUComputeResult> {
    if (!this.isInitialized || !this.device || !this.computeShader) {
      throw new Error('WebGPU not initialized');
    }

    // Device validation will be handled by the actual operations failing

    const startTime = performance.now();
    const totalParams = circuitParameters.length;
    const freqCount = Math.min(frequencies.length, 256); // Shader limit

    // Calculate optimal chunk size based on GPU memory limits
    const maxMemoryMB = Math.min(
      settings.memoryThreshold || 1024, // User setting
      this.capabilities?.limits.maxBufferSize ? this.capabilities.limits.maxBufferSize / (1024 * 1024) : 512 // GPU limit
    );
    
    // Each param needs: param data (8 floats) + result data (freqCount * 8 floats)
    const bytesPerParam = (8 + freqCount * 8) * 4; // 4 bytes per float
    const maxParamsPerChunk = Math.floor((maxMemoryMB * 1024 * 1024 * 0.8) / bytesPerParam); // 80% of available memory
    const chunkSize = Math.min(
      maxParamsPerChunk,
      Math.max(1000, settings.maxBatchSize || 10000) // Min 1k, respect user setting
    );

    console.log(`LAUNCH: WebGPU streaming compute: ${totalParams.toLocaleString()} params in chunks of ${chunkSize.toLocaleString()} (${Math.ceil(totalParams / chunkSize)} chunks)`);
    console.log(`  MEMORY: per chunk: ${(chunkSize * bytesPerParam / 1024 / 1024).toFixed(1)}MB`);

    try {
      // Create compute pipeline (reuse across chunks)
      const shaderModule = this.device.createShaderModule({
        code: this.computeShader
      });

      const computePipeline = this.device.createComputePipeline({
        layout: 'auto',
        compute: {
          module: shaderModule,
          entryPoint: 'main'
        }
      });

      // Maintain top-K results pool
      const topKResults: Array<{
        parameters: CircuitParameters;
        spectrum: Array<{ freq: number; real: number; imag: number; mag: number; phase: number }>;
        resnorm: number;
      }> = [];

      let totalGpuMemoryUsed = 0;
      let totalComputeTime = 0;
      let totalDataTransferTime = 0;
      let processedParams = 0;

      // Process in chunks
      for (let chunkStart = 0; chunkStart < totalParams; chunkStart += chunkSize) {
        const chunkEnd = Math.min(chunkStart + chunkSize, totalParams);
        const chunkParams = circuitParameters.slice(chunkStart, chunkEnd);
        const actualChunkSize = chunkParams.length;

        // Report progress
        const progress = (chunkStart / totalParams) * 100;
        progressCallback?.(progress);

        const chunkNum = Math.floor(chunkStart / chunkSize) + 1;
        const totalChunks = Math.ceil(totalParams / chunkSize);
        console.log(`  CHUNK: ${chunkNum}/${totalChunks}: processing ${actualChunkSize.toLocaleString()} params (${chunkStart}-${chunkEnd-1})`);

        const chunkStartTime = performance.now();

        // Create buffers for this chunk only
        const { paramBuffer, freqBuffer, refSpectrumBuffer, resultBuffer, configBuffer } = 
          this.createBuffers(chunkParams, frequencies.slice(0, freqCount), referenceSpectrum, freqCount);

        const bufferTime = performance.now();

        // Create bind groups
        const bindGroup0 = this.device.createBindGroup({
          layout: computePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: paramBuffer } },
            { binding: 1, resource: { buffer: freqBuffer } },
            { binding: 2, resource: { buffer: refSpectrumBuffer } },
            { binding: 3, resource: { buffer: resultBuffer } }
          ]
        });

        const bindGroup1 = this.device.createBindGroup({
          layout: computePipeline.getBindGroupLayout(1),
          entries: [
            { binding: 0, resource: { buffer: configBuffer } }
          ]
        });

        // Execute compute shader
        const commandEncoder = this.device.createCommandEncoder();
        const computePass = commandEncoder.beginComputePass({
          label: `Circuit Computation Chunk ${chunkNum}`
        });

        computePass.setPipeline(computePipeline);
        computePass.setBindGroup(0, bindGroup0);
        computePass.setBindGroup(1, bindGroup1);
        computePass.dispatchWorkgroups(Math.ceil(actualChunkSize / 64)); // 64 = workgroup size
        computePass.end();

        // Copy results back
        const stagingBuffer = this.device.createBuffer({
          size: resultBuffer.size,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        commandEncoder.copyBufferToBuffer(resultBuffer, 0, stagingBuffer, 0, resultBuffer.size);
        
        const computeEndTime = performance.now();
        this.device.queue.submit([commandEncoder.finish()]);

        // Read results with error handling
        let chunkResults: Array<{
          parameters: CircuitParameters;
          spectrum: Array<{ freq: number; real: number; imag: number; mag: number; phase: number }>;
          resnorm: number;
        }> = [];
        try {
          await stagingBuffer.mapAsync(GPUMapMode.READ);
          const rawResults = new Float32Array(stagingBuffer.getMappedRange());

          // Parse chunk results
          chunkResults = this.parseResults(rawResults, actualChunkSize, freqCount, chunkParams);

          // Update top-K results pool
          topKResults.push(...chunkResults);
          topKResults.sort((a, b) => a.resnorm - b.resnorm);
          if (topKResults.length > maxResults * 2) { // Keep 2x buffer to ensure best results
            topKResults.splice(maxResults * 2);
          }
        } catch (error) {
          console.error('ERROR: WebGPU Buffer Error:', error);
          // Continue processing other chunks even if this one fails
        } finally {
          // Always cleanup buffers regardless of success/failure
          try {
            stagingBuffer.unmap();
          } catch {
            // Buffer might not be mapped, ignore this error
          }
          
          // Destroy buffers
          [paramBuffer, freqBuffer, refSpectrumBuffer, resultBuffer, configBuffer, stagingBuffer]
            .forEach(buffer => {
              try {
                buffer.destroy();
              } catch (destroyError) {
                console.warn('Buffer destroy failed:', destroyError);
              }
            });
        }

        // Track metrics (only if processing succeeded)
        if (chunkResults.length > 0) {
          const chunkTotalTime = performance.now() - chunkStartTime;
          const chunkComputeTime = computeEndTime - bufferTime;
          const chunkDataTime = chunkTotalTime - chunkComputeTime; // Simplified calculation
          
          totalComputeTime += chunkComputeTime;
          totalDataTransferTime += chunkDataTime;
          totalGpuMemoryUsed = Math.max(totalGpuMemoryUsed, this.estimateGPUMemory(actualChunkSize, freqCount));
          processedParams += actualChunkSize;
        }

        const bestResnorm = chunkResults.length > 0 ? chunkResults[0].resnorm : 'N/A';
        const actualChunkTime = performance.now() - chunkStartTime;
        console.log(`    SUCCESS: Chunk ${chunkNum} completed: ${actualChunkTime.toFixed(2)}ms, best resnorm: ${typeof bestResnorm === 'number' ? bestResnorm.toFixed(6) : bestResnorm}`);
      }

      progressCallback?.(100);

      const endTime = performance.now();
      const totalTime = endTime - startTime;

      // Final sort and limit
      const finalResults = topKResults
        .sort((a, b) => a.resnorm - b.resnorm)
        .slice(0, maxResults);

      const benchmarkData: WebGPUBenchmarkResult = {
        totalTime,
        computeTime: totalComputeTime,
        dataTransferTime: totalDataTransferTime,
        parametersProcessed: processedParams,
        parametersPerSecond: processedParams / (totalTime / 1000),
        gpuMemoryUsed: totalGpuMemoryUsed,
        cpuMemoryUsed: finalResults.length * 1000 // Rough estimate
      };

      console.log('SUCCESS: WebGPU streaming computation completed:', {
        totalParams: totalParams.toLocaleString(),
        chunks: Math.ceil(totalParams / chunkSize),
        chunkSize: chunkSize.toLocaleString(),
        frequencies: freqCount,
        totalTime: `${totalTime.toFixed(2)}ms`,
        computeTime: `${totalComputeTime.toFixed(2)}ms`,
        dataTransferTime: `${totalDataTransferTime.toFixed(2)}ms`,
        resultCount: `${finalResults.length}/${totalParams}`,
        peakMemoryPerChunk: `${(totalGpuMemoryUsed / 1024 / 1024).toFixed(1)}MB`,
        paramPerSec: `${benchmarkData.parametersPerSecond.toFixed(0)}/s`,
        bestResnorm: finalResults[0]?.resnorm.toFixed(6) || 'N/A'
      });

      return {
        results: finalResults,
        benchmarkData,
        memoryUsed: totalGpuMemoryUsed
      };

    } catch (error) {
      console.error('ERROR: WebGPU computation failed:', error);
      throw error;
    }
  }

  private createBuffers(
    params: CircuitParameters[],
    frequencies: number[],
    refSpectrum: Array<{ freq: number; real: number; imag: number; mag: number; phase: number }>,
    freqCount: number
  ) {
    if (!this.device) throw new Error('Device not available');

    // Circuit parameters buffer (8 floats per param for alignment)
    const paramData = new Float32Array(params.length * 8);
    params.forEach((param, i) => {
      const offset = i * 8;
      paramData[offset] = param.Rsh;
      paramData[offset + 1] = param.Ra;
      paramData[offset + 2] = param.Ca;
      paramData[offset + 3] = param.Rb;
      paramData[offset + 4] = param.Cb;
      // padding: offset + 5, 6, 7
    });

    const paramBuffer = this.device.createBuffer({
      size: paramData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    this.device.queue.writeBuffer(paramBuffer, 0, paramData);

    // Frequency buffer
    const freqData = new Float32Array(frequencies);
    const freqBuffer = this.device.createBuffer({
      size: freqData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    this.device.queue.writeBuffer(freqBuffer, 0, freqData);

    // Reference spectrum buffer (2 floats per complex number)
    const refData = new Float32Array(refSpectrum.length * 2);
    refSpectrum.forEach((point, i) => {
      refData[i * 2] = point.real;
      refData[i * 2 + 1] = point.imag;
    });

    const refSpectrumBuffer = this.device.createBuffer({
      size: refData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    this.device.queue.writeBuffer(refSpectrumBuffer, 0, refData);

    // Result buffer (8 floats per result)
    const resultBuffer = this.device.createBuffer({
      size: params.length * freqCount * 8 * 4, // 8 floats * 4 bytes per float
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // Configuration uniform buffer
    const configData = new Uint32Array([params.length, freqCount, 0, 0]); // mae method = 0
    const configBuffer = this.device.createBuffer({
      size: configData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.device.queue.writeBuffer(configBuffer, 0, configData);

    return { paramBuffer, freqBuffer, refSpectrumBuffer, resultBuffer, configBuffer };
  }

  private parseResults(
    rawData: Float32Array,
    paramCount: number,
    freqCount: number,
    originalParams: CircuitParameters[],
  ) {
    const results: Array<{
      parameters: CircuitParameters;
      spectrum: Array<{ freq: number; real: number; imag: number; mag: number; phase: number }>;
      resnorm: number;
    }> = [];

    for (let paramIdx = 0; paramIdx < paramCount; paramIdx++) {
      const spectrum: Array<{ freq: number; real: number; imag: number; mag: number; phase: number }> = [];
      let resnorm = 0;

      for (let freqIdx = 0; freqIdx < freqCount; freqIdx++) {
        const resultIdx = (paramIdx * freqCount + freqIdx) * 8; // 8 floats per result
        
        if (resultIdx + 7 < rawData.length) {
          spectrum.push({
            freq: rawData[resultIdx],
            real: rawData[resultIdx + 1],
            imag: rawData[resultIdx + 2],
            mag: rawData[resultIdx + 3],
            phase: rawData[resultIdx + 4]
          });
          
          if (freqIdx === 0) {
            resnorm = rawData[resultIdx + 5]; // Get resnorm from first frequency point
            
            // Validate resnorm - GPU computation might return invalid values
            if (!isFinite(resnorm) || resnorm < 0) {
              console.warn(`WARNING: WebGPU: Invalid resnorm ${resnorm} for parameter ${paramIdx}, skipping`);
              resnorm = Infinity; // Mark as invalid
            }
          }
        }
      }

      // Only include results with valid resnorms and non-empty spectra
      if (isFinite(resnorm) && resnorm > 0 && spectrum.length > 0) {
        results.push({
          parameters: originalParams[paramIdx],
          spectrum,
          resnorm
        });
      }
    }

    console.log(`SUCCESS: WebGPU parseResults: ${results.length}/${paramCount} valid results parsed`);
    return results;
  }

  private estimateGPUMemory(paramCount: number, freqCount: number): number {
    // Rough estimation in bytes
    const paramBuffer = paramCount * 8 * 4; // 8 floats * 4 bytes
    const freqBuffer = freqCount * 4; // 1 float * 4 bytes  
    const refBuffer = freqCount * 2 * 4; // 2 floats * 4 bytes
    const resultBuffer = paramCount * freqCount * 8 * 4; // 8 floats * 4 bytes
    const overhead = 1024 * 1024; // 1MB overhead

    return paramBuffer + freqBuffer + refBuffer + resultBuffer + overhead;
  }

  getCapabilities(): WebGPUCapabilities | null {
    return this.capabilities;
  }

  isAvailable(): boolean {
    return this.isInitialized && this.capabilities?.supported === true;
  }

  async dispose(): Promise<void> {
    if (this.device) {
      await this.device.destroy();
      this.device = null;
    }
    this.adapter = null;
    this.isInitialized = false;
    this.capabilities = null;
    console.log('CLEANUP: WebGPU resources disposed');
  }
}