import { BackendMeshPoint } from '../types';
import { CircuitParameters } from '../types/parameters';

export interface StreamingExportConfig {
  filePrefix: string;
  maxModelsPerFile: number;
  topModelsToKeep: number;
  exportFullSpectrum: boolean;
}

export interface ExportedResultsMetadata {
  totalModels: number;
  files: string[];
  topModelsCount: number;
  gridSize: number;
  computationDate: string;
  parameters: {
    minFreq: number;
    maxFreq: number;
    numPoints: number;
  };
}

export interface CompactModelData {
  id: number;
  parameters: CircuitParameters;
  resnorm: number;
  spectrum?: Array<{ freq: number; real: number; imag: number; mag: number; phase: number }>;
}

/**
 * Streaming file export utility for large parameter grids
 * Prevents browser memory crashes by streaming results to downloadable files
 */
export class StreamingFileExporter {
  private config: StreamingExportConfig;
  private exportedCount: number = 0;
  private currentFileIndex: number = 1;
  private topModels: CompactModelData[] = [];
  private metadata: ExportedResultsMetadata;

  constructor(config: StreamingExportConfig, gridSize: number, freqParams: { minFreq: number; maxFreq: number; numPoints: number }) {
    this.config = config;
    this.metadata = {
      totalModels: 0,
      files: [],
      topModelsCount: 0,
      gridSize,
      computationDate: new Date().toISOString(),
      parameters: freqParams
    };
  }

  /**
   * Add a batch of models for export
   * Automatically manages file chunking and top model selection
   */
  addModelBatch(models: BackendMeshPoint[]): { topModelsUpdate: CompactModelData[], filesGenerated: string[] } {
    const compactModels: CompactModelData[] = models.map((model, index) => ({
      id: this.exportedCount + index,
      parameters: model.parameters,
      resnorm: model.resnorm,
      spectrum: this.config.exportFullSpectrum ? model.spectrum : undefined
    }));

    // Update total count
    this.exportedCount += models.length;
    this.metadata.totalModels = this.exportedCount;

    // Update top models (keep only the best)
    this.topModels.push(...compactModels);
    this.topModels.sort((a, b) => a.resnorm - b.resnorm); // Sort by resnorm (lower is better)
    
    if (this.topModels.length > this.config.topModelsToKeep) {
      this.topModels = this.topModels.slice(0, this.config.topModelsToKeep);
    }

    // Generate file for this batch (exclude top models to save space)
    const modelsToExport = compactModels.filter(model => 
      !this.topModels.some(top => top.id === model.id)
    );

    const filesGenerated: string[] = [];

    if (modelsToExport.length > 0) {
      // Split into multiple files if batch is too large
      const chunks = this.chunkArray(modelsToExport, this.config.maxModelsPerFile);
      
      for (const chunk of chunks) {
        const fileName = `${this.config.filePrefix}_batch_${this.currentFileIndex.toString().padStart(3, '0')}.json`;
        const fileContent = {
          metadata: {
            batchIndex: this.currentFileIndex,
            modelsInFile: chunk.length,
            exportDate: new Date().toISOString()
          },
          models: chunk
        };

        this.downloadFile(fileName, JSON.stringify(fileContent, null, 2));
        filesGenerated.push(fileName);
        this.metadata.files.push(fileName);
        this.currentFileIndex++;
      }
    }

    return {
      topModelsUpdate: [...this.topModels], // Return copy of current top models
      filesGenerated
    };
  }

  /**
   * Finalize export and generate metadata file
   */
  finalizeExport(): { topModels: CompactModelData[], metadataFile: string } {
    // Generate final metadata file
    this.metadata.topModelsCount = this.topModels.length;
    
    const metadataFileName = `${this.config.filePrefix}_metadata.json`;
    const metadataContent = {
      ...this.metadata,
      summary: {
        totalModelsComputed: this.metadata.totalModels,
        topModelsKept: this.metadata.topModelsCount,
        filesGenerated: this.metadata.files.length,
        compressionRatio: this.metadata.topModelsCount / this.metadata.totalModels
      },
      instructions: {
        howToImport: "Use the Import Results button to load this dataset back into the application",
        fileStructure: "Each batch file contains models excluded from the top performers",
        topModelsLocation: "Top models are kept in browser memory for immediate analysis"
      }
    };

    this.downloadFile(metadataFileName, JSON.stringify(metadataContent, null, 2));

    // Also generate top models file for easy sharing
    const topModelsFileName = `${this.config.filePrefix}_top_models.json`;
    const topModelsContent = {
      metadata: {
        description: `Top ${this.topModels.length} best performing models from ${this.metadata.totalModels} computed`,
        sortedBy: "resnorm (ascending - lower is better)",
        exportDate: new Date().toISOString()
      },
      models: this.topModels
    };

    this.downloadFile(topModelsFileName, JSON.stringify(topModelsContent, null, 2));

    console.log(`🎯 High Parameter Export Complete:
      📊 ${this.metadata.totalModels.toLocaleString()} total models computed
      ⭐ ${this.metadata.topModelsCount} top models kept in memory  
      📁 ${this.metadata.files.length} batch files generated
      💾 Files: ${metadataFileName}, ${topModelsFileName}, + batch files`);

    return {
      topModels: this.topModels,
      metadataFile: metadataFileName
    };
  }

  /**
   * Generate progress report
   */
  getExportProgress(): { 
    modelsExported: number, 
    filesGenerated: number, 
    topModelsCount: number,
    memoryFootprint: string
  } {
    return {
      modelsExported: this.exportedCount,
      filesGenerated: this.metadata.files.length,
      topModelsCount: this.topModels.length,
      memoryFootprint: `${this.topModels.length} models in memory vs ${this.exportedCount} total`
    };
  }

  private chunkArray<T>(array: T[], chunkSize: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < array.length; i += chunkSize) {
      chunks.push(array.slice(i, i + chunkSize));
    }
    return chunks;
  }

  private downloadFile(fileName: string, content: string): void {
    const blob = new Blob([content], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    a.style.display = 'none';
    
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
}

/**
 * File import utility for resuming analysis from exported results
 */
export class ResultsFileImporter {
  
  /**
   * Import metadata from exported results
   */
  static async importMetadata(file: File): Promise<ExportedResultsMetadata> {
    const content = await this.readFileAsText(file);
    const data = JSON.parse(content);
    
    if (!data.totalModels || !data.files) {
      throw new Error('Invalid metadata file format');
    }
    
    return data;
  }

  /**
   * Import top models from exported results
   */
  static async importTopModels(file: File): Promise<CompactModelData[]> {
    const content = await this.readFileAsText(file);
    const data = JSON.parse(content);
    
    if (!data.models || !Array.isArray(data.models)) {
      throw new Error('Invalid top models file format');
    }
    
    return data.models;
  }

  /**
   * Convert imported models back to BackendMeshPoint format
   */
  static convertToBackendMeshPoints(compactModels: CompactModelData[]): BackendMeshPoint[] {
    return compactModels.map(model => ({
      parameters: model.parameters,
      resnorm: model.resnorm,
      spectrum: model.spectrum || [] // Empty array if spectrum not included
    }));
  }

  private static readFileAsText(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsText(file);
    });
  }
}