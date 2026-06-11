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

      // Ensure proper file extension - prefer .json since it's more standard
      const finalFilename = filename.endsWith('.json') || filename.endsWith('.srd') ? filename : `${filename}.json`;

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
   * Upload and parse .srd or .json file
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

      // Check if this is a strict SRD format or flexible JSON
      let srdData: SRDData;
      try {
        // Try strict SRD validation first
        srdData = this.validateAndCast(parsedData);
      } catch {
        // If strict validation fails, try to convert flexible JSON to SRD format
        console.log('📄 Not strict SRD format, attempting flexible JSON parsing...');
        srdData = this.convertFlexibleJsonToSRD(parsedData, file.name);
      }

      console.log(`📤 File uploaded: ${file.name}`);
      console.log(`📊 Data: ${srdData.serializedResults.length.toLocaleString()} results from grid ${srdData.metadata.gridSize}^5`);

      return srdData;

    } catch (error) {
      console.error('❌ File upload failed:', error);
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
    // Check file extension - accept both .srd and .json since SRD is just JSON
    const fileName = file.name.toLowerCase();
    if (!fileName.endsWith('.srd') && !fileName.endsWith('.json')) {
      throw new SRDValidationError('Invalid file type. Expected .srd or .json file', 'INVALID_FILE_TYPE');
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

  /**
   * Convert flexible JSON to SRD format
   */
  private static convertFlexibleJsonToSRD(data: unknown, filename: string): SRDData {
    if (!data || typeof data !== 'object') {
      throw new SRDValidationError('Invalid JSON: must be an object', 'INVALID_JSON');
    }

    const obj = data as Record<string, unknown>;

    // Handle different JSON formats
    let serializedResults: SerializedResult[];
    let metadata: Partial<SRDMetadata> = {};

    // Check if it's already an array of results
    if (Array.isArray(data)) {
      serializedResults = this.normalizeResultArray(data);
    }
    // Check if it has a results array property
    else if (Array.isArray(obj.results)) {
      serializedResults = this.normalizeResultArray(obj.results);
      // Try to extract metadata if present
      if (obj.metadata && typeof obj.metadata === 'object') {
        metadata = obj.metadata as Partial<SRDMetadata>;
      }
    }
    // Check if it has serializedResults already
    else if (Array.isArray(obj.serializedResults)) {
      serializedResults = this.normalizeResultArray(obj.serializedResults);
      if (obj.metadata && typeof obj.metadata === 'object') {
        metadata = obj.metadata as Partial<SRDMetadata>;
      }
    }
    // Check for other common patterns
    else if (Array.isArray(obj.data)) {
      serializedResults = this.normalizeResultArray(obj.data);
    }
    // Check for exported models from the app
    else if (Array.isArray(obj.models)) {
      serializedResults = this.normalizeResultArray(obj.models);
    }
    else {
      throw new SRDValidationError('No valid results array found in JSON. Expected "results", "serializedResults", "data", or "models" array.', 'NO_RESULTS_ARRAY');
    }

    // Extract additional metadata from exported format
    let extractedTitle = metadata.title || filename.replace(/\.(json|srd)$/i, '');
    let extractedDescription = metadata.description || 'Imported from JSON file';

    // Check if this is an exported model file with exportDate
    if (obj.exportDate && obj.referenceParameters) {
      const exportDate = new Date(obj.exportDate as string);
      extractedTitle = `Exported Models - ${exportDate.toLocaleDateString()}`;
      extractedDescription = `Exported from spider plot on ${exportDate.toLocaleString()}`;
    }

    // Create SRD-compatible metadata
    const srdMetadata: SRDMetadata = {
      version: this.CURRENT_VERSION,
      format: this.EXPECTED_FORMAT,
      created: obj.exportDate as string || new Date().toISOString(),
      title: extractedTitle,
      description: extractedDescription,
      gridSize: metadata.gridSize || this.inferGridSize(serializedResults),
      frequencyPreset: metadata.frequencyPreset || 'custom',
      totalResults: serializedResults.length,
      compressionRatio: this.estimateCompressionRatio(serializedResults.length),
      memoryOptimized: true,
      author: metadata.author,
      tags: metadata.tags
    };

    return {
      metadata: srdMetadata,
      serializedResults
    };
  }

  /**
   * Normalize result array to SRD format
   */
  private static normalizeResultArray(results: unknown[]): SerializedResult[] {
    return results.map((item, index) => {
      if (!item || typeof item !== 'object') {
        throw new SRDValidationError(`Invalid result at index ${index}: must be an object`, 'INVALID_RESULT');
      }

      const result = item as Record<string, unknown>;

      // Extract required fields with fallbacks
      let configId = result.configId || result.id || result.config_id;
      let frequencyConfigId = result.frequencyConfigId || result.freq_id || result.frequency_id || 'freq_0';
      const resnorm = result.resnorm || result.residual || result.error;

      // Generate proper ConfigId format (gridSize_rshIdx_raIdx_caIdx_rbIdx_cbIdx)
      // If configId doesn't match the expected format, generate a synthetic one
      if (!configId || typeof configId !== 'string' || !configId.match(/^\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$/)) {
        // For imported data, generate synthetic config IDs based on index
        // This allows the data to be loaded even if the original format doesn't match
        const gridSize = 9; // Default grid size for imports
        const maxIdx = gridSize - 1; // eslint-disable-line @typescript-eslint/no-unused-vars

        // Distribute indices across the grid space using the result index
        const totalCombinations = Math.pow(gridSize, 5);
        const scaledIndex = Math.floor((index / results.length) * totalCombinations);

        // Convert scaled index to 5D grid coordinates
        let remaining = scaledIndex;
        const cbIdx = remaining % gridSize; remaining = Math.floor(remaining / gridSize);
        const rbIdx = remaining % gridSize; remaining = Math.floor(remaining / gridSize);
        const caIdx = remaining % gridSize; remaining = Math.floor(remaining / gridSize);
        const raIdx = remaining % gridSize; remaining = Math.floor(remaining / gridSize);
        const rshIdx = remaining % gridSize;

        configId = `${gridSize.toString().padStart(2, '0')}_${rshIdx.toString().padStart(2, '0')}_${raIdx.toString().padStart(2, '0')}_${caIdx.toString().padStart(2, '0')}_${rbIdx.toString().padStart(2, '0')}_${cbIdx.toString().padStart(2, '0')}`;

        console.log(`🔧 Generated synthetic configId for index ${index}: ${configId} (original: ${result.configId || result.id || 'none'})`);
      }
      if (typeof frequencyConfigId !== 'string') {
        frequencyConfigId = 'freq_0';
      }
      if (typeof resnorm !== 'number' || !isFinite(resnorm)) {
        throw new SRDValidationError(`Invalid resnorm at index ${index}: must be a finite number`, 'INVALID_RESNORM');
      }

      return {
        configId: configId as string,
        frequencyConfigId: frequencyConfigId as string,
        resnorm: resnorm as number,
        computationTime: typeof result.computationTime === 'number' ? result.computationTime : undefined,
        timestamp: typeof result.timestamp === 'string' ? result.timestamp : undefined
      };
    });
  }

  /**
   * Infer grid size from results
   */
  private static inferGridSize(results: SerializedResult[]): number {
    const totalResults = results.length;
    // Common grid sizes: 3^5=243, 5^5=3125, 7^5=16807, 9^5=59049
    const commonSizes = [3, 5, 7, 9, 11, 13, 15, 17, 19];

    for (const size of commonSizes) {
      if (Math.pow(size, 5) === totalResults) {
        return size;
      }
    }

    // If no exact match, estimate
    const estimated = Math.round(Math.pow(totalResults, 1/5));
    console.warn(`⚠️ Could not determine exact grid size. Estimated: ${estimated} (${totalResults} results)`);
    return Math.max(3, estimated);
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