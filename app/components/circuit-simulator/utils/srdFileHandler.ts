/**
 * SRD File Handler
 * ================
 *
 * Handles download and upload of Serialized Resnorm Data (.srd) files.
 * Provides validation, error handling, and integration with SerializedComputationManager.
 */

interface SRDMetadata {
  version: string;
  format: string;
  created: string;
  title: string;
  description?: string;
  gridSize: number;
  frequencyPreset: string;
  totalResults: number;
  compressionRatio: number;
  memoryOptimized: boolean;
  author?: string;
  tags?: string[];
}

interface SerializedResult {
  configId: string;
  frequencyConfigId: string;
  resnorm: number;
  computationTime?: number;
  timestamp?: string;
}

interface SRDData {
  metadata: SRDMetadata;
  serializedResults: SerializedResult[];
}

export type { SRDMetadata, SerializedResult, SRDData };

/**
 * File validation error types
 */
export class SRDValidationError extends Error {
  constructor(message: string, public code: string) {
    super(message);
    this.name = 'SRDValidationError';
  }
}

/**
 * SRD File Handler - handles serialized resnorm data files
 */
export class SRDFileHandler {
  private static readonly CURRENT_VERSION = '1.0';
  private static readonly EXPECTED_FORMAT = 'SpideyPlot-SerializedResnormData';
  private static readonly MAX_FILE_SIZE = 500 * 1024 * 1024; // 500MB limit
  private static readonly MAX_RESULTS = 10_000_000; // 10M results limit

  /**
   * Download serialized data as .srd file
   */
  static download(data: SRDData, filename: string): void {
    try {
      // Validate data before download
      this.validateSRDData(data);

      // Ensure proper file extension
      const finalFilename = filename.endsWith('.srd') ? filename : `${filename}.srd`;

      // Create formatted JSON
      const jsonData = JSON.stringify(data, null, 2);

      // Create and trigger download
      const blob = new Blob([jsonData], { type: 'application/json' });
      const url = URL.createObjectURL(blob);

      const link = document.createElement('a');
      link.href = url;
      link.download = finalFilename;

      // Trigger download
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      // Cleanup
      URL.revokeObjectURL(url);

      console.log(`📥 SRD file downloaded: ${finalFilename}`);
      console.log(`📊 Data: ${data.serializedResults.length.toLocaleString()} results, ${data.metadata.compressionRatio.toFixed(1)}% compression`);

    } catch (error) {
      console.error('❌ SRD download failed:', error);
      throw new SRDValidationError(
        `Failed to download SRD file: ${error instanceof Error ? error.message : 'Unknown error'}`,
        'DOWNLOAD_ERROR'
      );
    }
  }

  /**
   * Upload and parse .srd file
   */
  static async upload(file: File): Promise<SRDData> {
    try {
      // Basic file validation
      await this.validateFile(file);

      // Read file content
      const content = await this.readFileContent(file);

      // Parse JSON
      let parsedData: unknown;
      try {
        parsedData = JSON.parse(content);
      } catch {
        throw new SRDValidationError('Invalid JSON format', 'INVALID_JSON');
      }

      // Validate SRD format
      const srdData = this.validateAndCast(parsedData);

      console.log(`📤 SRD file uploaded: ${file.name}`);
      console.log(`📊 Data: ${srdData.serializedResults.length.toLocaleString()} results from grid ${srdData.metadata.gridSize}^5`);

      return srdData;

    } catch (error) {
      console.error('❌ SRD upload failed:', error);
      throw error; // Re-throw for caller handling
    }
  }

  /**
   * Validate SRD data structure
   */
  static validateSRDData(data: unknown): asserts data is SRDData {
    if (!data || typeof data !== 'object') {
      throw new SRDValidationError('Invalid data: must be an object', 'INVALID_DATA');
    }

    const obj = data as Record<string, unknown>;

    // Validate metadata
    if (!obj.metadata || typeof obj.metadata !== 'object') {
      throw new SRDValidationError('Missing or invalid metadata', 'MISSING_METADATA');
    }

    const metadata = obj.metadata as Record<string, unknown>;

    // Check required metadata fields
    const requiredFields = ['version', 'format', 'created', 'title', 'gridSize', 'frequencyPreset', 'totalResults'];
    for (const field of requiredFields) {
      if (!(field in metadata)) {
        throw new SRDValidationError(`Missing required metadata field: ${field}`, 'MISSING_FIELD');
      }
    }

    // Validate version
    if (metadata.version !== this.CURRENT_VERSION) {
      throw new SRDValidationError(
        `Unsupported version: ${metadata.version}. Expected: ${this.CURRENT_VERSION}`,
        'VERSION_MISMATCH'
      );
    }

    // Validate format
    if (metadata.format !== this.EXPECTED_FORMAT) {
      throw new SRDValidationError(
        `Invalid format: ${metadata.format}. Expected: ${this.EXPECTED_FORMAT}`,
        'FORMAT_MISMATCH'
      );
    }

    // Validate serialized results
    if (!Array.isArray(obj.serializedResults)) {
      throw new SRDValidationError('Invalid serializedResults: must be an array', 'INVALID_RESULTS');
    }

    // Validate result count
    if (obj.serializedResults.length > this.MAX_RESULTS) {
      throw new SRDValidationError(
        `Too many results: ${obj.serializedResults.length}. Maximum: ${this.MAX_RESULTS.toLocaleString()}`,
        'TOO_MANY_RESULTS'
      );
    }

    // Sample validation of serialized results structure
    if (obj.serializedResults.length > 0) {
      const firstResult = obj.serializedResults[0] as Record<string, unknown>;
      const requiredResultFields = ['configId', 'frequencyConfigId', 'resnorm'];

      for (const field of requiredResultFields) {
        if (!(field in firstResult)) {
          throw new SRDValidationError(`Missing result field: ${field}`, 'INVALID_RESULT_STRUCTURE');
        }
      }

      // Validate config ID format (e.g., "15_01_02_04_03_05")
      if (typeof firstResult.configId !== 'string' || !firstResult.configId.match(/^\d+(_\d+){5}$/)) {
        throw new SRDValidationError('Invalid configId format', 'INVALID_CONFIG_ID');
      }

      // Validate resnorm is a number
      if (typeof firstResult.resnorm !== 'number' || !isFinite(firstResult.resnorm)) {
        throw new SRDValidationError('Invalid resnorm: must be a finite number', 'INVALID_RESNORM');
      }
    }
  }

  /**
   * Check if data is valid SRD format (non-throwing)
   */
  static isValidSRD(data: unknown): data is SRDData {
    try {
      this.validateSRDData(data);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Get metadata from SRD data
   */
  static getMetadata(data: SRDData): SRDMetadata {
    return { ...data.metadata };
  }

  /**
   * Create SRD data structure from components
   */
  static createSRDData(
    serializedResults: SerializedResult[],
    metadata: Partial<SRDMetadata> & Pick<SRDMetadata, 'title' | 'gridSize' | 'frequencyPreset'>
  ): SRDData {
    const fullMetadata: SRDMetadata = {
      version: this.CURRENT_VERSION,
      format: this.EXPECTED_FORMAT,
      created: new Date().toISOString(),
      totalResults: serializedResults.length,
      compressionRatio: this.estimateCompressionRatio(serializedResults.length),
      memoryOptimized: true,
      ...metadata
    };

    return {
      metadata: fullMetadata,
      serializedResults
    };
  }

  /**
   * Estimate file size for given number of results
   */
  static estimateFileSize(resultCount: number): { sizeBytes: number; sizeFormatted: string } {
    // Estimate ~150 bytes per serialized result + metadata overhead
    const sizeBytes = (resultCount * 150) + 1000; // 1KB metadata overhead

    let sizeFormatted: string;
    if (sizeBytes < 1024) {
      sizeFormatted = `${sizeBytes} B`;
    } else if (sizeBytes < 1024 * 1024) {
      sizeFormatted = `${(sizeBytes / 1024).toFixed(1)} KB`;
    } else {
      sizeFormatted = `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB`;
    }

    return { sizeBytes, sizeFormatted };
  }

  // Private helper methods

  private static async validateFile(file: File): Promise<void> {
    // Check file extension
    if (!file.name.toLowerCase().endsWith('.srd')) {
      throw new SRDValidationError('Invalid file type. Expected .srd file', 'INVALID_FILE_TYPE');
    }

    // Check file size
    if (file.size > this.MAX_FILE_SIZE) {
      throw new SRDValidationError(
        `File too large: ${(file.size / (1024 * 1024)).toFixed(1)}MB. Maximum: ${this.MAX_FILE_SIZE / (1024 * 1024)}MB`,
        'FILE_TOO_LARGE'
      );
    }

    // Check if file is empty
    if (file.size === 0) {
      throw new SRDValidationError('File is empty', 'EMPTY_FILE');
    }
  }

  private static async readFileContent(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onload = (event) => {
        const content = event.target?.result;
        if (typeof content === 'string') {
          resolve(content);
        } else {
          reject(new SRDValidationError('Failed to read file content', 'READ_ERROR'));
        }
      };

      reader.onerror = () => {
        reject(new SRDValidationError('File read error', 'READ_ERROR'));
      };

      reader.readAsText(file);
    });
  }

  private static validateAndCast(data: unknown): SRDData {
    this.validateSRDData(data);
    return data as SRDData;
  }

  private static estimateCompressionRatio(resultCount: number): number {
    // Traditional storage: ~4KB per model
    // Serialized storage: ~150 bytes per result
    const traditionalSize = resultCount * 4096;
    const serializedSize = resultCount * 150;
    return ((1 - serializedSize / traditionalSize) * 100);
  }
}

/**
 * React hook for handling SRD file operations
 */
export function useSRDFileHandler() {
  const downloadSRD = (data: SRDData, filename: string) => {
    try {
      SRDFileHandler.download(data, filename);
      return { success: true, error: null };
    } catch (error) {
      return {
        success: false,
        error: error instanceof SRDValidationError ? error.message : 'Download failed'
      };
    }
  };

  const uploadSRD = async (file: File) => {
    try {
      const data = await SRDFileHandler.upload(file);
      return { success: true, data, error: null };
    } catch (error) {
      return {
        success: false,
        data: null,
        error: error instanceof SRDValidationError ? error.message : 'Upload failed'
      };
    }
  };

  return {
    downloadSRD,
    uploadSRD,
    isValidSRD: SRDFileHandler.isValidSRD,
    estimateFileSize: SRDFileHandler.estimateFileSize
  };
}