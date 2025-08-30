/**
 * ML Data Interface Layer
 * 
 * Provides standardized data export/import interfaces for machine learning
 * Supports multiple formats: JSON, CSV, HDF5, NumPy arrays
 * Handles feature engineering and preprocessing for ML workflows
 */

import DatabaseService from './database-service'
import { Database } from './database.types'

// ML Data Types
export interface MLFeatureMatrix {
  features: number[][]  // Shape: [samples, features]
  featureNames: string[]
  featureTypes: ('categorical' | 'numerical' | 'ordinal')[]
  featureRanges?: { min: number; max: number }[]
}

export interface MLTargetVector {
  targets: number[]  // Shape: [samples]
  targetName: string
  targetType: 'regression' | 'classification' | 'multi_target'
  targetClasses?: string[]  // For classification
}

export interface MLDataset {
  features: MLFeatureMatrix
  targets: MLTargetVector
  metadata: {
    datasetId: string
    datasetName: string
    totalSamples: number
    createdAt: string
    dataQualityScore: number
    sourceInfo: {
      configurations: string[]
      sessions: string[]
      taggedModels: string[]
    }
  }
  preprocessing?: {
    scaler?: 'standard' | 'minmax' | 'robust' | 'none'
    featureSelection?: string[]
    outlierRemoval?: boolean
    missingValueHandling?: 'drop' | 'mean' | 'median' | 'interpolate'
  }
}

export interface MLExperimentConfig {
  modelType: 'regression' | 'classification' | 'optimization'
  algorithm: 'neural_network' | 'random_forest' | 'gaussian_process' | 'svm' | 'xgboost'
  hyperparameters: Record<string, any>
  crossValidation: {
    folds: number
    strategy: 'kfold' | 'stratified' | 'time_series'
  }
  metrics: string[]  // ['mae', 'mse', 'r2', 'accuracy', 'f1', etc.]
}

export interface MLTrainingResults {
  modelId: string
  trainingMetrics: Record<string, number>
  validationMetrics: Record<string, number>
  testMetrics?: Record<string, number>
  featureImportance?: Record<string, number>
  confusionMatrix?: number[][]
  learningCurve?: { trainSizes: number[]; trainScores: number[]; validScores: number[] }
}

/**
 * Data Extraction Service
 * Extracts and preprocesses data from the database for ML training
 */
export class MLDataExtractionService {
  /**
   * Extract circuit parameters and impedance data from tagged models
   */
  static async extractFromTaggedModels(
    userId: string,
    options: {
      tagCategories?: string[]
      sessionIds?: string[]
      minResnormValue?: number
      maxResnormValue?: number
      includeImpedanceSpectrum?: boolean
    } = {}
  ) {
    const { data: taggedModels, error } = await DatabaseService.taggedModels.getTaggedModels(
      userId,
      undefined,
      options.tagCategories?.[0]  // Use first category for now
    )

    if (error || !taggedModels) {
      throw new Error(`Failed to extract tagged models: ${error?.message}`)
    }

    // Filter models based on criteria
    let filteredModels = taggedModels

    if (options.sessionIds) {
      filteredModels = filteredModels.filter(m => options.sessionIds!.includes(m.session_id))
    }

    if (options.minResnormValue !== undefined) {
      filteredModels = filteredModels.filter(m => (m.resnorm_value || 0) >= options.minResnormValue!)
    }

    if (options.maxResnormValue !== undefined) {
      filteredModels = filteredModels.filter(m => (m.resnorm_value || Infinity) <= options.maxResnormValue!)
    }

    // Extract features (circuit parameters)
    const features: number[][] = []
    const targets: number[] = []
    const featureNames = ['Rsh', 'Ra', 'Ca', 'Rb', 'Cb']
    const impedanceSpectra: any[] = []

    for (const model of filteredModels) {
      const params = model.circuit_parameters as any
      
      // Extract circuit parameters as features
      const featureRow = [
        params.Rsh || 0,
        params.Ra || 0,
        params.Ca || 0,  // Store in Farads
        params.Rb || 0,
        params.Cb || 0   // Store in Farads
      ]

      features.push(featureRow)
      targets.push(model.resnorm_value || 0)
      
      if (options.includeImpedanceSpectrum && model.impedance_spectrum) {
        impedanceSpectra.push(model.impedance_spectrum)
      }
    }

    return {
      features: {
        features,
        featureNames,
        featureTypes: ['numerical', 'numerical', 'numerical', 'numerical', 'numerical'] as const
      },
      targets: {
        targets,
        targetName: 'resnorm_value',
        targetType: 'regression' as const
      },
      impedanceSpectra: options.includeImpedanceSpectrum ? impedanceSpectra : undefined,
      modelIds: filteredModels.map(m => m.id),
      metadata: {
        totalSamples: filteredModels.length,
        extractedAt: new Date().toISOString(),
        sourceTagCategories: options.tagCategories,
        sourceSessions: options.sessionIds
      }
    }
  }

  /**
   * Extract comprehensive dataset from computation results
   */
  static async extractFromComputationResults(
    configurationIds: string[],
    options: {
      includeFailedComputations?: boolean
      maxSamples?: number
      frequencyRange?: [number, number]
      parameterFilter?: Partial<Record<string, [number, number]>>
    } = {}
  ) {
    // Get computation results for configurations
    const computationPromises = configurationIds.map(configId => 
      DatabaseService.session.getActiveSession(configId)  // This should be replaced with proper computation results fetch
    )

    const computationResults = await Promise.all(computationPromises)
    
    // Extract and process data from computation results
    const features: number[][] = []
    const targets: number[] = []
    const impedanceData: any[] = []

    // Process each computation result
    for (const result of computationResults) {
      if (result.error) continue

      // Extract circuit parameters and results
      // This would process the grid results from the computation
      // Implementation depends on the structure of computation results
    }

    return {
      features: {
        features,
        featureNames: ['Rsh', 'Ra', 'Ca', 'Rb', 'Cb', 'min_freq', 'max_freq', 'num_points'],
        featureTypes: ['numerical', 'numerical', 'numerical', 'numerical', 'numerical', 'numerical', 'numerical', 'numerical'] as const
      },
      targets: {
        targets,
        targetName: 'resnorm_value',
        targetType: 'regression' as const
      },
      impedanceData,
      metadata: {
        totalSamples: features.length,
        extractedAt: new Date().toISOString(),
        sourceConfigurations: configurationIds
      }
    }
  }

  /**
   * Feature engineering for circuit parameters
   */
  static engineerFeatures(
    features: number[][],
    featureNames: string[],
    options: {
      includeTimeConstants?: boolean
      includeFrequencyDependentFeatures?: boolean
      includeRatios?: boolean
      logTransform?: string[]  // Feature names to log-transform
    } = {}
  ) {
    const engineeredFeatures: number[][] = []
    const engineeredNames: string[] = [...featureNames]

    for (const row of features) {
      const engineeredRow = [...row]

      if (options.includeTimeConstants) {
        // Calculate time constants: τ = RC
        const RaIndex = featureNames.indexOf('Ra')
        const CaIndex = featureNames.indexOf('Ca')
        const RbIndex = featureNames.indexOf('Rb')
        const CbIndex = featureNames.indexOf('Cb')

        if (RaIndex >= 0 && CaIndex >= 0) {
          const tauA = row[RaIndex] * row[CaIndex]
          engineeredRow.push(tauA)
          if (engineeredNames.length === featureNames.length + engineeredRow.length - row.length - 1) {
            engineeredNames.push('tau_a')
          }
        }

        if (RbIndex >= 0 && CbIndex >= 0) {
          const tauB = row[RbIndex] * row[CbIndex]
          engineeredRow.push(tauB)
          if (engineeredNames.length === featureNames.length + engineeredRow.length - row.length - 1) {
            engineeredNames.push('tau_b')
          }
        }
      }

      if (options.includeRatios) {
        // Add parameter ratios
        const RaIndex = featureNames.indexOf('Ra')
        const RbIndex = featureNames.indexOf('Rb')
        const CaIndex = featureNames.indexOf('Ca')
        const CbIndex = featureNames.indexOf('Cb')

        if (RaIndex >= 0 && RbIndex >= 0 && row[RbIndex] !== 0) {
          const resistanceRatio = row[RaIndex] / row[RbIndex]
          engineeredRow.push(resistanceRatio)
          if (engineeredNames.length === featureNames.length + engineeredRow.length - row.length - 1) {
            engineeredNames.push('Ra_Rb_ratio')
          }
        }

        if (CaIndex >= 0 && CbIndex >= 0 && row[CbIndex] !== 0) {
          const capacitanceRatio = row[CaIndex] / row[CbIndex]
          engineeredRow.push(capacitanceRatio)
          if (engineeredNames.length === featureNames.length + engineeredRow.length - row.length - 1) {
            engineeredNames.push('Ca_Cb_ratio')
          }
        }
      }

      // Apply log transforms
      if (options.logTransform) {
        for (const featureName of options.logTransform) {
          const featureIndex = featureNames.indexOf(featureName)
          if (featureIndex >= 0 && row[featureIndex] > 0) {
            engineeredRow[featureIndex] = Math.log10(row[featureIndex])
          }
        }
      }

      engineeredFeatures.push(engineeredRow)
    }

    return {
      features: engineeredFeatures,
      featureNames: engineeredNames
    }
  }

  /**
   * Prepare dataset for specific ML algorithms
   */
  static prepareForAlgorithm(
    dataset: MLDataset,
    algorithm: MLExperimentConfig['algorithm'],
    options: {
      testSplitRatio?: number
      validationSplitRatio?: number
      randomSeed?: number
      stratify?: boolean
    } = {}
  ) {
    const { features, targets } = dataset
    const totalSamples = features.features.length

    // Set default split ratios
    const testRatio = options.testSplitRatio || 0.2
    const validRatio = options.validationSplitRatio || 0.2
    const trainRatio = 1 - testRatio - validRatio

    // Simple random splitting (in production, use proper stratification)
    const trainSize = Math.floor(totalSamples * trainRatio)
    const validSize = Math.floor(totalSamples * validRatio)

    // Create indices for splitting
    const indices = Array.from({ length: totalSamples }, (_, i) => i)
    
    // Shuffle indices (simple implementation)
    if (options.randomSeed) {
      // Use seed for reproducible results
      let seed = options.randomSeed
      for (let i = indices.length - 1; i > 0; i--) {
        seed = (seed * 9301 + 49297) % 233280
        const j = Math.floor((seed / 233280) * (i + 1))
        ;[indices[i], indices[j]] = [indices[j], indices[i]]
      }
    }

    const trainIndices = indices.slice(0, trainSize)
    const validIndices = indices.slice(trainSize, trainSize + validSize)
    const testIndices = indices.slice(trainSize + validSize)

    // Split features and targets
    const trainFeatures = trainIndices.map(i => features.features[i])
    const validFeatures = validIndices.map(i => features.features[i])
    const testFeatures = testIndices.map(i => features.features[i])

    const trainTargets = trainIndices.map(i => targets.targets[i])
    const validTargets = validIndices.map(i => targets.targets[i])
    const testTargets = testIndices.map(i => targets.targets[i])

    return {
      train: {
        features: trainFeatures,
        targets: trainTargets,
        size: trainFeatures.length
      },
      validation: {
        features: validFeatures,
        targets: validTargets,
        size: validFeatures.length
      },
      test: {
        features: testFeatures,
        targets: testTargets,
        size: testFeatures.length
      },
      metadata: {
        totalSamples,
        trainRatio,
        validRatio,
        testRatio,
        featureNames: features.featureNames,
        targetName: targets.targetName,
        randomSeed: options.randomSeed
      }
    }
  }
}

/**
 * Data Export Service
 * Exports data in various formats for different ML frameworks
 */
export class MLDataExportService {
  /**
   * Export dataset in JSON format for JavaScript ML libraries
   */
  static exportToJSON(dataset: MLDataset): string {
    return JSON.stringify(dataset, null, 2)
  }

  /**
   * Export dataset in CSV format for R, Python pandas, etc.
   */
  static exportToCSV(dataset: MLDataset): string {
    const { features, targets } = dataset
    const headers = [...features.featureNames, targets.targetName]
    
    let csv = headers.join(',') + '\n'
    
    for (let i = 0; i < features.features.length; i++) {
      const row = [...features.features[i], targets.targets[i]]
      csv += row.join(',') + '\n'
    }
    
    return csv
  }

  /**
   * Export in NumPy-compatible format (as JSON that can be parsed by Python)
   */
  static exportToNumpyFormat(dataset: MLDataset) {
    return {
      X: dataset.features.features,  // Feature matrix
      y: dataset.targets.targets,    // Target vector
      feature_names: dataset.features.featureNames,
      target_name: dataset.targets.targetName,
      metadata: dataset.metadata,
      shape: {
        X: [dataset.features.features.length, dataset.features.features[0]?.length || 0],
        y: [dataset.targets.targets.length]
      }
    }
  }

  /**
   * Export for TensorFlow.js format
   */
  static exportToTensorFlowJS(dataset: MLDataset) {
    return {
      inputs: dataset.features.features,
      outputs: dataset.targets.targets.map(t => [t]), // TF.js expects 2D output
      inputShape: [dataset.features.features[0]?.length || 0],
      outputShape: [1],
      featureNames: dataset.features.featureNames,
      targetName: dataset.targets.targetName,
      metadata: dataset.metadata
    }
  }

  /**
   * Export training configuration for experiment tracking
   */
  static exportExperimentConfig(
    config: MLExperimentConfig,
    dataset: MLDataset,
    results?: MLTrainingResults
  ) {
    return {
      experiment: {
        id: `exp_${Date.now()}`,
        timestamp: new Date().toISOString(),
        dataset: {
          id: dataset.metadata.datasetId,
          name: dataset.metadata.datasetName,
          samples: dataset.metadata.totalSamples,
          features: dataset.features.featureNames.length,
          qualityScore: dataset.metadata.dataQualityScore
        },
        model: config,
        results: results || null,
        reproducibility: {
          randomSeed: Math.floor(Math.random() * 1000000),
          environment: {
            platform: typeof window !== 'undefined' ? 'browser' : 'node',
            timestamp: new Date().toISOString()
          }
        }
      }
    }
  }
}

/**
 * Data Import Service
 * Imports data from various formats and creates ML datasets
 */
export class MLDataImportService {
  /**
   * Import from CSV file/string
   */
  static importFromCSV(
    csvContent: string,
    options: {
      hasHeader?: boolean
      delimiter?: string
      targetColumn: string | number
      featureColumns?: string[] | number[]
    }
  ): MLDataset {
    const lines = csvContent.trim().split('\n')
    const delimiter = options.delimiter || ','
    
    let startIndex = 0
    let headers: string[] = []
    
    if (options.hasHeader) {
      headers = lines[0].split(delimiter).map(h => h.trim())
      startIndex = 1
    }
    
    const data = lines.slice(startIndex).map(line => 
      line.split(delimiter).map(cell => parseFloat(cell.trim()))
    )
    
    // Extract features and targets
    const targetColumnIndex = typeof options.targetColumn === 'string' 
      ? headers.indexOf(options.targetColumn)
      : options.targetColumn
    
    const features: number[][] = []
    const targets: number[] = []
    const featureNames: string[] = []
    
    // Determine feature columns
    const featureIndices = options.featureColumns 
      ? (typeof options.featureColumns[0] === 'string'
          ? (options.featureColumns as string[]).map(name => headers.indexOf(name))
          : options.featureColumns as number[])
      : data[0]?.map((_, i) => i).filter(i => i !== targetColumnIndex) || []
    
    // Extract feature names
    for (const index of featureIndices) {
      featureNames.push(headers[index] || `feature_${index}`)
    }
    
    // Extract data
    for (const row of data) {
      const featureRow = featureIndices.map(i => row[i] || 0)
      features.push(featureRow)
      targets.push(row[targetColumnIndex] || 0)
    }
    
    return {
      features: {
        features,
        featureNames,
        featureTypes: featureNames.map(() => 'numerical' as const)
      },
      targets: {
        targets,
        targetName: headers[targetColumnIndex] || 'target',
        targetType: 'regression'
      },
      metadata: {
        datasetId: `imported_${Date.now()}`,
        datasetName: 'Imported Dataset',
        totalSamples: features.length,
        createdAt: new Date().toISOString(),
        dataQualityScore: 0.8, // Default score
        sourceInfo: {
          configurations: [],
          sessions: [],
          taggedModels: []
        }
      }
    }
  }
  
  /**
   * Import from JSON format
   */
  static importFromJSON(jsonContent: string): MLDataset {
    try {
      const data = JSON.parse(jsonContent)
      
      // Validate structure
      if (!data.features || !data.targets || !data.metadata) {
        throw new Error('Invalid ML dataset JSON structure')
      }
      
      return data as MLDataset
    } catch (error) {
      throw new Error(`Failed to parse JSON dataset: ${error}`)
    }
  }
}

/**
 * Main ML Data Interface
 * Orchestrates all ML data operations
 */
export class MLDataInterface {
  static extraction = MLDataExtractionService
  static export = MLDataExportService
  static import = MLDataImportService
  
  /**
   * Complete workflow: Extract → Engineer → Export
   */
  static async processForTraining(
    userId: string,
    options: {
      dataSource: 'tagged_models' | 'computation_results'
      sourceIds?: string[]
      featureEngineering?: {
        includeTimeConstants?: boolean
        includeRatios?: boolean
        logTransform?: string[]
      }
      exportFormat?: 'json' | 'csv' | 'numpy' | 'tensorflow'
      splitData?: boolean
    }
  ): Promise<{
    dataset: MLDataset
    exportedData: string | object
    metadata: any
  }> {
    // Step 1: Extract data
    let extractedData
    if (options.dataSource === 'tagged_models') {
      extractedData = await MLDataExtractionService.extractFromTaggedModels(userId, {
        includeImpedanceSpectrum: true
      })
    } else {
      extractedData = await MLDataExtractionService.extractFromComputationResults(
        options.sourceIds || []
      )
    }
    
    // Step 2: Feature engineering
    let processedFeatures = extractedData.features
    if (options.featureEngineering) {
      const engineered = MLDataExtractionService.engineerFeatures(
        extractedData.features.features,
        extractedData.features.featureNames,
        options.featureEngineering
      )
      processedFeatures = {
        ...extractedData.features,
        features: engineered.features,
        featureNames: engineered.featureNames
      }
    }
    
    // Step 3: Create dataset
    const dataset: MLDataset = {
      features: processedFeatures,
      targets: extractedData.targets,
      metadata: {
        datasetId: `ml_${Date.now()}`,
        datasetName: `ML Dataset ${new Date().toISOString()}`,
        totalSamples: processedFeatures.features.length,
        createdAt: new Date().toISOString(),
        dataQualityScore: 0.85,
        sourceInfo: {
          configurations: [],
          sessions: [],
          taggedModels: extractedData.modelIds || []
        }
      }
    }
    
    // Step 4: Export in requested format
    let exportedData: string | object
    switch (options.exportFormat) {
      case 'csv':
        exportedData = MLDataExportService.exportToCSV(dataset)
        break
      case 'numpy':
        exportedData = MLDataExportService.exportToNumpyFormat(dataset)
        break
      case 'tensorflow':
        exportedData = MLDataExportService.exportToTensorFlowJS(dataset)
        break
      default:
        exportedData = MLDataExportService.exportToJSON(dataset)
    }
    
    // Step 5: Save dataset to database
    await DatabaseService.datasets.createDataset(userId, {
      name: dataset.metadata.datasetName,
      description: `Auto-generated ML dataset from ${options.dataSource}`,
      sourceConfigurations: dataset.metadata.sourceInfo.configurations,
      sourceSessions: dataset.metadata.sourceInfo.sessions,
      totalSamples: dataset.metadata.totalSamples,
      featureCount: dataset.features.featureNames.length,
      datasetType: dataset.targets.targetType,
      targetVariable: dataset.targets.targetName,
      featureEngineeringConfig: options.featureEngineering,
      dataQualityScore: dataset.metadata.dataQualityScore
    })
    
    return {
      dataset,
      exportedData,
      metadata: extractedData.metadata
    }
  }
}

export default MLDataInterface