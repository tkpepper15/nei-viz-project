/**
 * Database Integration Tests
 * 
 * Comprehensive test suite for the SQL strategy implementation
 * Tests all database operations and ML interfaces
 */

import DatabaseService from '../database-service'
import MLDataInterface from '../ml-data-interface'
import { Database } from '../database.types'

// Mock user ID for testing
const TEST_USER_ID = 'test-user-123'
const TEST_SESSION_NAME = 'Test Session'

// Test data
const TEST_CIRCUIT_PARAMETERS = {
  Rsh: 1000,
  Ra: 500,
  Ca: 1e-6,
  Rb: 800,
  Cb: 2e-6
}

const TEST_MODEL_DATA = {
  modelId: 'model-123',
  tagName: 'optimal',
  tagCategory: 'user' as const,
  circuitParameters: TEST_CIRCUIT_PARAMETERS,
  resnormValue: 0.05,
  impedanceSpectrum: {
    frequencies: [1, 10, 100, 1000],
    real: [900, 800, 700, 600],
    imaginary: [100, 200, 300, 400]
  }
}

describe('Database Service Integration Tests', () => {
  let testSessionId: string
  let testConfigId: string

  beforeAll(async () => {
    // Initialize test session
    const { data: session, error } = await DatabaseService.session.createSession(
      TEST_USER_ID,
      TEST_SESSION_NAME,
      'Test session for integration tests',
      {
        nodeOptions: '--max-old-space-size=8192',
        computeWorkers: 4,
        memoryLimit: '8GB',
        debugMode: true
      },
      {
        groupPortion: 0.2,
        selectedOpacityGroups: [0],
        visualizationType: 'spider3d',
        chromaEnabled: true,
        resnormSpread: 1.0,
        useResnormCenter: false,
        showLabels: true
      }
    )

    if (error) {
      throw new Error(`Failed to create test session: ${error.message}`)
    }

    testSessionId = session!.id
  })

  afterAll(async () => {
    // Clean up test data
    if (testSessionId) {
      await DatabaseService.session.deleteSession(testSessionId)
    }
  })

  describe('Session Management', () => {
    test('should create and retrieve active session', async () => {
      const { data: activeSession, error } = await DatabaseService.session.getActiveSession(TEST_USER_ID)
      
      expect(error).toBeNull()
      expect(activeSession).toBeTruthy()
      expect(activeSession?.session_name).toBe(TEST_SESSION_NAME)
      expect(activeSession?.session_id).toBe(testSessionId)
    })

    test('should update session environment variables', async () => {
      const updatedEnvironment = {
        debugMode: false,
        customSettings: { testKey: 'testValue' }
      }

      const { error } = await DatabaseService.session.updateSession(testSessionId, {
        environment_variables: updatedEnvironment
      })

      expect(error).toBeNull()

      // Verify update
      const { data: session } = await DatabaseService.session.getActiveSession(TEST_USER_ID)
      expect(session?.environment_variables).toMatchObject(updatedEnvironment)
    })

    test('should update session activity', async () => {
      const { error } = await DatabaseService.session.updateSessionActivity(testSessionId)
      expect(error).toBeNull()
    })

    test('should increment session statistics', async () => {
      const { error } = await DatabaseService.session.incrementSessionStats(
        testSessionId,
        5,
        1000,
        '00:05:30'
      )

      expect(error).toBeNull()
    })
  })

  describe('Tagged Models', () => {
    let taggedModelId: string

    test('should tag a model', async () => {
      const { data: taggedModel, error } = await DatabaseService.taggedModels.tagModel(
        TEST_USER_ID,
        testSessionId,
        'test-config-id',
        TEST_MODEL_DATA
      )

      expect(error).toBeNull()
      expect(taggedModel).toBeTruthy()
      expect(taggedModel!.tag_name).toBe('optimal')
      expect(taggedModel!.model_id).toBe('model-123')

      taggedModelId = taggedModel!.id
    })

    test('should retrieve tagged models', async () => {
      const { data: taggedModels, error } = await DatabaseService.taggedModels.getTaggedModels(
        TEST_USER_ID,
        testSessionId
      )

      expect(error).toBeNull()
      expect(taggedModels).toBeTruthy()
      expect(taggedModels!.length).toBeGreaterThan(0)
      expect(taggedModels![0].tag_name).toBe('optimal')
    })

    test('should get ML training candidates', async () => {
      // First mark the model for ML training
      await DatabaseService.taggedModels.updateTaggedModel(taggedModelId, {
        isMLTrainingData: true,
        mlRelevanceScore: 0.9
      })

      const { data: candidates, error } = await DatabaseService.taggedModels.getMLTrainingCandidates(
        TEST_USER_ID,
        0.8
      )

      expect(error).toBeNull()
      expect(candidates).toBeTruthy()
      expect(candidates!.length).toBeGreaterThan(0)
      expect(candidates![0].ml_relevance_score).toBeGreaterThanOrEqual(0.8)
    })

    test('should update tagged model', async () => {
      const { error } = await DatabaseService.taggedModels.updateTaggedModel(taggedModelId, {
        notes: 'Updated test notes',
        mlRelevanceScore: 0.95
      })

      expect(error).toBeNull()

      // Verify update
      const { data: taggedModels } = await DatabaseService.taggedModels.getTaggedModels(
        TEST_USER_ID,
        testSessionId
      )
      const updatedModel = taggedModels?.find(m => m.id === taggedModelId)
      expect(updatedModel?.notes).toBe('Updated test notes')
      expect(updatedModel?.ml_relevance_score).toBe(0.95)
    })
  })

  describe('Parameter Exploration', () => {
    let explorationId: string

    test('should create exploration session', async () => {
      const { data: exploration, error } = await DatabaseService.exploration.createExplorationSession(
        TEST_USER_ID,
        testSessionId,
        {
          name: 'Test Parameter Exploration',
          type: 'manual',
          parameterRanges: {
            Rsh: [100, 2000],
            Ra: [50, 1000],
            Ca: [0.1e-6, 10e-6],
            Rb: [100, 1500],
            Cb: [0.5e-6, 5e-6]
          },
          samplingStrategy: {
            type: 'grid',
            pointsPerParameter: 5
          },
          totalCombinations: 3125
        }
      )

      expect(error).toBeNull()
      expect(exploration).toBeTruthy()
      expect(exploration!.exploration_name).toBe('Test Parameter Exploration')

      explorationId = exploration!.id
    })

    test('should update exploration progress', async () => {
      const { error } = await DatabaseService.exploration.updateExplorationProgress(
        explorationId,
        {
          successfulComputations: 100,
          failedComputations: 5,
          progressPercentage: 25.0,
          status: 'active'
        }
      )

      expect(error).toBeNull()
    })

    test('should get exploration sessions', async () => {
      const { data: sessions, error } = await DatabaseService.exploration.getExplorationSessions(
        TEST_USER_ID,
        'active'
      )

      expect(error).toBeNull()
      expect(sessions).toBeTruthy()
      expect(sessions!.length).toBeGreaterThan(0)
    })
  })

  describe('ML Datasets', () => {
    let datasetId: string

    test('should create ML dataset', async () => {
      const { data: dataset, error } = await DatabaseService.datasets.createDataset(
        TEST_USER_ID,
        {
          name: 'Test ML Dataset',
          description: 'Dataset created for testing',
          sourceConfigurations: ['config-1', 'config-2'],
          sourceSessions: [testSessionId],
          totalSamples: 1000,
          featureCount: 5,
          datasetType: 'regression',
          targetVariable: 'resnorm_value',
          dataQualityScore: 0.85
        }
      )

      expect(error).toBeNull()
      expect(dataset).toBeTruthy()
      expect(dataset!.dataset_name).toBe('Test ML Dataset')
      expect(dataset!.dataset_type).toBe('regression')

      datasetId = dataset!.id
    })

    test('should get ML datasets', async () => {
      const { data: datasets, error } = await DatabaseService.datasets.getDatasets(
        TEST_USER_ID,
        'regression',
        500
      )

      expect(error).toBeNull()
      expect(datasets).toBeTruthy()
      expect(datasets!.length).toBeGreaterThan(0)
    })

    test('should update dataset paths', async () => {
      const { error } = await DatabaseService.datasets.updateDatasetPaths(
        datasetId,
        {
          featureMatrixPath: '/data/features.h5',
          targetVectorPath: '/data/targets.h5',
          metadataPath: '/data/metadata.json'
        }
      )

      expect(error).toBeNull()
    })

    test('should generate dataset from tags', async () => {
      const { data: datasetId, error } = await DatabaseService.datasets.generateDatasetFromTags(
        TEST_USER_ID,
        'Auto Generated Dataset',
        ['optimal', 'user']
      )

      expect(error).toBeNull()
      expect(datasetId).toBeTruthy()
    })
  })

  describe('ML Models', () => {
    let modelId: string

    test('should save ML model', async () => {
      const { data: model, error } = await DatabaseService.models.saveModel(
        TEST_USER_ID,
        {
          name: 'Test Neural Network',
          modelType: 'neural_network',
          description: 'Test model for parameter optimization',
          trainingConfig: {
            layers: [128, 64, 32, 1],
            activation: 'relu',
            optimizer: 'adam',
            epochs: 100
          },
          hyperparameters: {
            learning_rate: 0.001,
            batch_size: 32,
            dropout: 0.2
          },
          modelPath: '/models/test_nn.h5',
          trainingMetrics: {
            loss: 0.05,
            mae: 0.03,
            r2: 0.92
          },
          validationMetrics: {
            loss: 0.06,
            mae: 0.035,
            r2: 0.89
          }
        }
      )

      expect(error).toBeNull()
      expect(model).toBeTruthy()
      expect(model!.model_name).toBe('Test Neural Network')
      expect(model!.model_type).toBe('neural_network')

      modelId = model!.id
    })

    test('should get ML models', async () => {
      const { data: models, error } = await DatabaseService.models.getModels(
        TEST_USER_ID,
        'neural_network'
      )

      expect(error).toBeNull()
      expect(models).toBeTruthy()
      expect(models!.length).toBeGreaterThan(0)
    })

    test('should deploy ML model', async () => {
      const { error } = await DatabaseService.models.deployModel(
        modelId,
        'https://api.example.com/ml/predict'
      )

      expect(error).toBeNull()
    })

    test('should update model metrics', async () => {
      const { error } = await DatabaseService.models.updateModelMetrics(
        modelId,
        {
          testMetrics: {
            loss: 0.055,
            mae: 0.032,
            r2: 0.91
          }
        }
      )

      expect(error).toBeNull()
    })
  })

  describe('Visualization Snapshots', () => {
    test('should save visualization snapshot', async () => {
      const { data: snapshot, error } = await DatabaseService.snapshots.saveSnapshot(
        TEST_USER_ID,
        testSessionId,
        'test-config-id',
        {
          name: 'Test Visualization Snapshot',
          description: 'Snapshot created during testing',
          visualizationType: 'spider3d',
          cameraPosition: {
            x: 0,
            y: 0,
            z: 10,
            rotation: { x: -30, y: 45, z: 0 }
          },
          filterSettings: {
            resnormRange: { min: 0, max: 1 },
            groupPortion: 0.2
          },
          displaySettings: {
            chromaEnabled: true,
            showLabels: true
          }
        }
      )

      expect(error).toBeNull()
      expect(snapshot).toBeTruthy()
      expect(snapshot!.snapshot_name).toBe('Test Visualization Snapshot')
      expect(snapshot!.visualization_type).toBe('spider3d')
    })

    test('should get visualization snapshots', async () => {
      const { data: snapshots, error } = await DatabaseService.snapshots.getSnapshots(
        TEST_USER_ID,
        testSessionId
      )

      expect(error).toBeNull()
      expect(snapshots).toBeTruthy()
      expect(snapshots!.length).toBeGreaterThan(0)
    })
  })

  describe('Parameter Optimization', () => {
    let jobId: string

    test('should create optimization job', async () => {
      const { data: job, error } = await DatabaseService.optimization.createOptimizationJob(
        TEST_USER_ID,
        testSessionId,
        {
          name: 'Test Optimization Job',
          algorithm: 'genetic',
          objectiveFunction: 'minimize_resnorm',
          parameterBounds: {
            Rsh: [100, 2000],
            Ra: [50, 1000],
            Ca: [0.1e-6, 10e-6],
            Rb: [100, 1500],
            Cb: [0.5e-6, 5e-6]
          },
          maxIterations: 100,
          populationSize: 50
        }
      )

      expect(error).toBeNull()
      expect(job).toBeTruthy()
      expect(job!.job_name).toBe('Test Optimization Job')
      expect(job!.optimization_algorithm).toBe('genetic')

      jobId = job!.id
    })

    test('should update job progress', async () => {
      const { error } = await DatabaseService.optimization.updateJobProgress(
        jobId,
        {
          currentIteration: 25,
          bestObjectiveValue: 0.045,
          progressPercentage: 25.0,
          status: 'running'
        }
      )

      expect(error).toBeNull()
    })

    test('should get optimization jobs', async () => {
      const { data: jobs, error } = await DatabaseService.optimization.getOptimizationJobs(
        TEST_USER_ID,
        'running'
      )

      expect(error).toBeNull()
      expect(jobs).toBeTruthy()
      expect(jobs!.length).toBeGreaterThan(0)
    })
  })
})

describe('ML Data Interface Tests', () => {
  test('should extract data from tagged models', async () => {
    // This would require actual tagged models in the database
    // For now, we'll test the interface structure
    const extractedData = await MLDataInterface.extraction.extractFromTaggedModels(
      TEST_USER_ID,
      {
        tagCategories: ['optimal', 'user'],
        includeImpedanceSpectrum: true
      }
    )

    expect(extractedData).toHaveProperty('features')
    expect(extractedData).toHaveProperty('targets')
    expect(extractedData).toHaveProperty('metadata')
    expect(extractedData.features).toHaveProperty('featureNames')
    expect(extractedData.targets).toHaveProperty('targetName')
  })

  test('should engineer features', async () => {
    const testFeatures = [
      [1000, 500, 1e-6, 800, 2e-6],
      [1200, 600, 1.5e-6, 900, 2.5e-6]
    ]
    const featureNames = ['Rsh', 'Ra', 'Ca', 'Rb', 'Cb']

    const engineered = MLDataInterface.extraction.engineerFeatures(
      testFeatures,
      featureNames,
      {
        includeTimeConstants: true,
        includeRatios: true,
        logTransform: ['Rsh', 'Ra', 'Rb']
      }
    )

    expect(engineered.features).toHaveLength(2)
    expect(engineered.features[0]).toHaveLength(featureNames.length + 4) // +2 time constants, +2 ratios
    expect(engineered.featureNames).toContain('tau_a')
    expect(engineered.featureNames).toContain('tau_b')
    expect(engineered.featureNames).toContain('Ra_Rb_ratio')
    expect(engineered.featureNames).toContain('Ca_Cb_ratio')
  })

  test('should export data in different formats', async () => {
    const testDataset = {
      features: {
        features: [[1, 2, 3], [4, 5, 6]],
        featureNames: ['f1', 'f2', 'f3'],
        featureTypes: ['numerical', 'numerical', 'numerical'] as const
      },
      targets: {
        targets: [0.1, 0.2],
        targetName: 'resnorm',
        targetType: 'regression' as const
      },
      metadata: {
        datasetId: 'test-dataset',
        datasetName: 'Test Dataset',
        totalSamples: 2,
        createdAt: new Date().toISOString(),
        dataQualityScore: 0.9,
        sourceInfo: {
          configurations: [],
          sessions: [],
          taggedModels: []
        }
      }
    }

    // Test JSON export
    const jsonExport = MLDataInterface.export.exportToJSON(testDataset)
    expect(jsonExport).toContain('features')
    expect(jsonExport).toContain('targets')

    // Test CSV export
    const csvExport = MLDataInterface.export.exportToCSV(testDataset)
    expect(csvExport).toContain('f1,f2,f3,resnorm')
    expect(csvExport).toContain('1,2,3,0.1')

    // Test NumPy format
    const numpyExport = MLDataInterface.export.exportToNumpyFormat(testDataset)
    expect(numpyExport).toHaveProperty('X')
    expect(numpyExport).toHaveProperty('y')
    expect(numpyExport).toHaveProperty('feature_names')
    expect(numpyExport.shape.X).toEqual([2, 3])
    expect(numpyExport.shape.y).toEqual([2])

    // Test TensorFlow.js format
    const tfExport = MLDataInterface.export.exportToTensorFlowJS(testDataset)
    expect(tfExport).toHaveProperty('inputs')
    expect(tfExport).toHaveProperty('outputs')
    expect(tfExport.inputShape).toEqual([3])
    expect(tfExport.outputShape).toEqual([1])
  })

  test('should import data from CSV', async () => {
    const csvData = 'f1,f2,f3,target\n1,2,3,0.1\n4,5,6,0.2\n7,8,9,0.3'
    
    const imported = MLDataInterface.import.importFromCSV(csvData, {
      hasHeader: true,
      targetColumn: 'target'
    })

    expect(imported.features.features).toHaveLength(3)
    expect(imported.targets.targets).toHaveLength(3)
    expect(imported.features.featureNames).toEqual(['f1', 'f2', 'f3'])
    expect(imported.targets.targetName).toBe('target')
    expect(imported.features.features[0]).toEqual([1, 2, 3])
    expect(imported.targets.targets[0]).toBe(0.1)
  })
})

describe('Complete ML Workflow Tests', () => {
  test('should process complete ML workflow', async () => {
    // This is an integration test that would require a full database setup
    // For now, we'll test the workflow structure
    const workflowResult = await MLDataInterface.processForTraining(
      TEST_USER_ID,
      {
        dataSource: 'tagged_models',
        featureEngineering: {
          includeTimeConstants: true,
          includeRatios: true,
          logTransform: ['Rsh', 'Ra', 'Rb']
        },
        exportFormat: 'json',
        splitData: true
      }
    )

    expect(workflowResult).toHaveProperty('dataset')
    expect(workflowResult).toHaveProperty('exportedData')
    expect(workflowResult).toHaveProperty('metadata')
  })
})