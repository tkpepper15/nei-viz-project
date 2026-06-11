/**
 * TypeScript Config Serializer
 * ============================
 * 
 * Port of the Python ConfigSerializer for use in React components.
 * Provides 1-based indexing and procedural parameter generation.
 */

export interface SerializedCircuitParameters {
  rsh: number;  // Shunt resistance (Ω)
  ra: number;   // Apical resistance (Ω) 
  ca: number;   // Apical capacitance (F)
  rb: number;   // Basal resistance (Ω)
  cb: number;   // Basal capacitance (F)
}

export class ConfigId {
  constructor(
    public readonly gridSize: number,
    public readonly rshIdx: number,  // 1-based index
    public readonly raIdx: number,   // 1-based index
    public readonly caIdx: number,   // 1-based index
    public readonly rbIdx: number,   // 1-based index
    public readonly cbIdx: number    // 1-based index
  ) {
    // Validation
    if (gridSize < 2) throw new Error('Grid size must be at least 2');
    if (rshIdx < 1 || rshIdx > gridSize) throw new Error(`rsh_idx must be 1-${gridSize}`);
    if (raIdx < 1 || raIdx > gridSize) throw new Error(`ra_idx must be 1-${gridSize}`);
    if (caIdx < 1 || caIdx > gridSize) throw new Error(`ca_idx must be 1-${gridSize}`);
    if (rbIdx < 1 || rbIdx > gridSize) throw new Error(`rb_idx must be 1-${gridSize}`);
    if (cbIdx < 1 || cbIdx > gridSize) throw new Error(`cb_idx must be 1-${gridSize}`);
  }
  
  toString(): string {
    return `${this.gridSize.toString().padStart(2, '0')}_${this.rshIdx.toString().padStart(2, '0')}_${this.raIdx.toString().padStart(2, '0')}_${this.caIdx.toString().padStart(2, '0')}_${this.rbIdx.toString().padStart(2, '0')}_${this.cbIdx.toString().padStart(2, '0')}`;
  }
  
  static fromString(configStr: string): ConfigId {
    const parts = configStr.split('_');
    if (parts.length !== 6) {
      throw new Error(`Invalid config string format: ${configStr}`);
    }

    const [gridSizeStr, rshIdxStr, raIdxStr, caIdxStr, rbIdxStr, cbIdxStr] = parts;

    const gridSize = parseInt(gridSizeStr, 10);
    let rshIdx = parseInt(rshIdxStr, 10);
    let raIdx = parseInt(raIdxStr, 10);
    let caIdx = parseInt(caIdxStr, 10);
    let rbIdx = parseInt(rbIdxStr, 10);
    let cbIdx = parseInt(cbIdxStr, 10);

    // Auto-detect and convert 0-based indices to 1-based for backward compatibility
    // If any index is 0, assume the entire config uses 0-based indexing
    const isZeroBased = rshIdx === 0 || raIdx === 0 || caIdx === 0 || rbIdx === 0 || cbIdx === 0;
    if (isZeroBased) {
      rshIdx += 1;
      raIdx += 1;
      caIdx += 1;
      rbIdx += 1;
      cbIdx += 1;
    }

    return new ConfigId(gridSize, rshIdx, raIdx, caIdx, rbIdx, cbIdx);
  }
  
  toLinearIndex(): number {
    // Convert to 0-based for calculation
    const r0 = this.rshIdx - 1;
    const a0 = this.raIdx - 1;
    const c0 = this.caIdx - 1;
    const b0 = this.rbIdx - 1;
    const d0 = this.cbIdx - 1;
    
    return ((((r0 * this.gridSize + a0) * this.gridSize + c0) * this.gridSize + b0) * this.gridSize + d0);
  }
  
  static fromLinearIndex(linearIndex: number, gridSize: number): ConfigId {
    let remaining = linearIndex;
    
    const cbIdx = (remaining % gridSize) + 1;
    remaining = Math.floor(remaining / gridSize);
    
    const rbIdx = (remaining % gridSize) + 1;
    remaining = Math.floor(remaining / gridSize);
    
    const caIdx = (remaining % gridSize) + 1;
    remaining = Math.floor(remaining / gridSize);
    
    const raIdx = (remaining % gridSize) + 1;
    remaining = Math.floor(remaining / gridSize);
    
    const rshIdx = remaining + 1;
    
    return new ConfigId(gridSize, rshIdx, raIdx, caIdx, rbIdx, cbIdx);
  }
}

export class ConfigSerializer {
  private parameterGrids: {
    rsh: number[];
    ra: number[];
    ca: number[];
    rb: number[];
    cb: number[];
  };
  
  constructor(public readonly gridSize: number) {
    if (gridSize < 2) throw new Error('Grid size must be at least 2');
    
    // Generate parameter grids using logarithmic spacing
    this.parameterGrids = {
      rsh: this.generateLogGrid(10, 10000, gridSize),      // 10Ω to 10kΩ
      ra: this.generateLogGrid(10, 10000, gridSize),       // 10Ω to 10kΩ
      ca: this.generateLogGrid(0.1e-6, 50e-6, gridSize),  // 0.1μF to 50μF
      rb: this.generateLogGrid(10, 10000, gridSize),       // 10Ω to 10kΩ
      cb: this.generateLogGrid(0.1e-6, 50e-6, gridSize)   // 0.1μF to 50μF
    };
  }
  
  private generateLogGrid(min: number, max: number, points: number): number[] {
    const logMin = Math.log10(min);
    const logMax = Math.log10(max);
    const step = (logMax - logMin) / (points - 1);
    
    const grid: number[] = [];
    for (let i = 0; i < points; i++) {
      const logValue = logMin + i * step;
      grid.push(Math.pow(10, logValue));
    }
    
    return grid;
  }
  
  serializeConfig(raIdx: number, rbIdx: number, rshIdx: number, caIdx: number, cbIdx: number): ConfigId {
    return new ConfigId(this.gridSize, rshIdx, raIdx, caIdx, rbIdx, cbIdx);
  }
  
  deserializeConfig(configId: ConfigId): SerializedCircuitParameters {
    // Convert from 1-based to 0-based for array access
    const rshValue = this.parameterGrids.rsh[configId.rshIdx - 1];
    const raValue = this.parameterGrids.ra[configId.raIdx - 1];
    const caValue = this.parameterGrids.ca[configId.caIdx - 1];
    const rbValue = this.parameterGrids.rb[configId.rbIdx - 1];
    const cbValue = this.parameterGrids.cb[configId.cbIdx - 1];
    
    return {
      rsh: rshValue,
      ra: raValue,
      ca: caValue,
      rb: rbValue,
      cb: cbValue
    };
  }
  
  getAllParameterValues(): {
    rsh: { index: number; value: number; }[];
    ra: { index: number; value: number; }[];
    ca: { index: number; value: number; }[];
    rb: { index: number; value: number; }[];
    cb: { index: number; value: number; }[];
  } {
    return {
      rsh: this.parameterGrids.rsh.map((value, i) => ({ index: i + 1, value })),
      ra: this.parameterGrids.ra.map((value, i) => ({ index: i + 1, value })),
      ca: this.parameterGrids.ca.map((value, i) => ({ index: i + 1, value })),
      rb: this.parameterGrids.rb.map((value, i) => ({ index: i + 1, value })),
      cb: this.parameterGrids.cb.map((value, i) => ({ index: i + 1, value }))
    };
  }
  
  getTotalConfigurations(): number {
    return Math.pow(this.gridSize, 5);
  }
}

/**
 * Utility functions
 */
export const ConfigUtils = {
  /**
   * Create a config serializer with standard grid size
   */
  createStandard(gridSize: number = 9): ConfigSerializer {
    return new ConfigSerializer(gridSize);
  },
  
  /**
   * Generate all possible configuration IDs for a grid size
   */
  generateAllConfigIds(gridSize: number): ConfigId[] {
    const configs: ConfigId[] = [];
    
    for (let rsh = 1; rsh <= gridSize; rsh++) {
      for (let ra = 1; ra <= gridSize; ra++) {
        for (let ca = 1; ca <= gridSize; ca++) {
          for (let rb = 1; rb <= gridSize; rb++) {
            for (let cb = 1; cb <= gridSize; cb++) {
              configs.push(new ConfigId(gridSize, rsh, ra, ca, rb, cb));
            }
          }
        }
      }
    }
    
    return configs;
  },
  
  /**
   * Find config ID that best matches given parameters
   */
  findBestMatchingConfig(params: SerializedCircuitParameters, gridSize: number): ConfigId | null {
    const serializer = new ConfigSerializer(gridSize);
    let bestMatch: ConfigId | null = null;
    let bestScore = Infinity;
    
    // Search all possible configs for best match
    for (let rsh = 1; rsh <= gridSize; rsh++) {
      for (let ra = 1; ra <= gridSize; ra++) {
        for (let ca = 1; ca <= gridSize; ca++) {
          for (let rb = 1; rb <= gridSize; rb++) {
            for (let cb = 1; cb <= gridSize; cb++) {
              const configId = new ConfigId(gridSize, rsh, ra, ca, rb, cb);
              const testParams = serializer.deserializeConfig(configId);
              
              // Calculate similarity score
              const score = 
                Math.abs(Math.log(testParams.rsh) - Math.log(params.rsh)) +
                Math.abs(Math.log(testParams.ra) - Math.log(params.ra)) +
                Math.abs(Math.log(testParams.ca) - Math.log(params.ca)) +
                Math.abs(Math.log(testParams.rb) - Math.log(params.rb)) +
                Math.abs(Math.log(testParams.cb) - Math.log(params.cb));
              
              if (score < bestScore) {
                bestScore = score;
                bestMatch = configId;
              }
            }
          }
        }
      }
    }
    
    return bestScore < 0.1 ? bestMatch : null; // Only return if reasonably close
  }
};