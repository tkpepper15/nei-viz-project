/**
 * Comprehensive Database Service Layer
 * 
 * Provides clean interfaces for all database operations including:
 * - Session management with environment variables
 * - Tagged model storage for ML training
 * - Parameter exploration tracking
 * - ML dataset preparation and model management
 * - Visualization state snapshots
 * - Parameter optimization jobs
 */

import { supabase } from './supabase'
import { Database, Json } from './database.types'
import { CircuitParameters, ImpedancePoint } from '../app/components/circuit-simulator/types'

// Type aliases for cleaner code
type Tables = Database['public']['Tables']
type UserSessionInsert = Tables['user_sessions']['Insert']
type UserSessionUpdate = Tables['user_sessions']['Update']
type TaggedModelInsert = Tables['tagged_models']['Insert']
type ParameterExplorationSessionInsert = Tables['parameter_exploration_sessions']['Insert']
type MLTrainingDatasetInsert = Tables['ml_training_datasets']['Insert']
type MLModelInsert = Tables['ml_models']['Insert']
type VisualizationSnapshotInsert = Tables['visualization_snapshots']['Insert']
type ParameterOptimizationJobInsert = Tables['parameter_optimization_jobs']['Insert']

// Additional type definitions for better type safety
export interface FilterSettings {
  [key: string]: unknown
}

export interface DisplaySettings {
  [key: string]: unknown
}

export interface PanelStates {
  [key: string]: boolean | string | number
}

export interface SliderValues {
  [key: string]: number
}

export interface CameraPosition {
  x: number
  y: number
  z: number
  rotation: { x: number; y: number; z: number }
}

export interface UserAnnotations {
  [key: string]: unknown
}

export interface MLObjectives {
  [key: string]: unknown
}

export interface MLConstraints {
  [key: string]: unknown
}

export interface MLRecommendations {
  [key: string]: unknown
}

export interface TrainingConfig {
  [key: string]: unknown
}

export interface Hyperparameters {
  [key: string]: number | string | boolean
}

export interface MLMetrics {
  [key: string]: number | string | unknown
}

export interface ParameterBounds {
  [key: string]: { min: number; max: number }
}

export interface OptimizationConstraints {
  [key: string]: unknown
}

export interface FeatureEngineeringConfig {
  [key: string]: unknown
}

export interface SamplingStrategy {
  type: 'grid' | 'random' | 'latin_hypercube' | 'sobol'
  [key: string]: unknown
}

export interface ParameterRanges {
  [parameter: string]: { min: number; max: number; count?: number }
}

// Extended interfaces for better type safety
export interface SessionEnvironment {
  nodeOptions?: string
  computeWorkers?: number
  memoryLimit?: string
  cacheSize?: string
  debugMode?: boolean
  customSettings?: Record<string, unknown>
}

export interface VisualizationSettings {
  groupPortion: number
  selectedOpacityGroups: number[]
  visualizationType: 'spider2d' | 'spider3d' | 'nyquist'
  chromaEnabled: boolean
  resnormSpread: number
  useResnormCenter: boolean
  showLabels: boolean
}

export interface PerformanceSettings {
  maxWorkers: number
  chunkSize: number
  memoryThreshold: number
  enableCaching: boolean
  adaptiveLimits: boolean
}

export interface TaggingContext {
  cameraPosition?: { x: number; y: number; z: number; rotation: { x: number; y: number; z: number } }
  currentZoom?: number
  filterSettings?: FilterSettings
  timestamp: string
  sessionContext?: Record<string, unknown>
}

export interface MLDatasetMetrics {
  featureImportance?: Record<string, number>
  correlationMatrix?: number[][]
  dataDistribution?: Record<string, unknown>
  outlierAnalysis?: Record<string, unknown>
  qualityScore: number
}

/**
 * Session Management Service
 * Handles user sessions with environment variables and settings
 */
export class SessionService {
  static async createSession(
    userId: string,
    sessionName: string,
    description?: string,
    environmentVariables?: SessionEnvironment,
    visualizationSettings?: VisualizationSettings,
    performanceSettings?: PerformanceSettings
  ) {
    const sessionData: UserSessionInsert = {
      user_id: userId,
      session_name: sessionName,
      description,
      environment_variables: (environmentVariables || {}) as Json,
      visualization_settings: (visualizationSettings || {}) as Json,
      performance_settings: (performanceSettings || {}) as Json
    }

    const { data, error } = await supabase
      .from('user_sessions')
      .insert(sessionData)
      .select()
      .single()

    return { data, error }
  }

  static async getActiveSession(userId: string) {
    // Get the most recent active session directly instead of using RPC
    const { data, error } = await supabase
      .from('user_sessions')
      .select('*')
      .eq('user_id', userId)
      .eq('is_active', true)
      .order('last_accessed', { ascending: false })
      .limit(1)
      .single()

    return { data: data || null, error }
  }

  static async updateSession(sessionId: string, updates: UserSessionUpdate) {
    const { data, error } = await supabase
      .from('user_sessions')
      .update({ ...updates, updated_at: new Date().toISOString() })
      .eq('id', sessionId)
      .select()
      .single()

    return { data, error }
  }

  static async updateSessionActivity(sessionId: string) {
    const { error } = await supabase
      .from('user_sessions')
      .update({ 
        last_accessed: new Date().toISOString(),
        is_active: true 
      })
      .eq('id', sessionId)

    return { error }
  }

  static async getUserSessions(userId: string) {
    const { data, error } = await supabase
      .from('user_sessions')
      .select('*')
      .eq('user_id', userId)
      .order('last_accessed', { ascending: false })

    return { data, error }
  }

  static async deleteSession(sessionId: string) {
    const { error } = await supabase
      .from('user_sessions')
      .delete()
      .eq('id', sessionId)

    return { error }
  }

  static async incrementSessionStats(
    sessionId: string,
    computationCount: number = 1,
    modelsGenerated: number = 0,
    computationTime: string = '0 seconds'
  ) {
    const { data, error } = await supabase
      .from('user_sessions')
      .update({
        total_computations: computationCount,
        total_models_generated: modelsGenerated,
        total_computation_time: computationTime,
        updated_at: new Date().toISOString()
      })
      .eq('id', sessionId)
      .select()
      .single()

    return { data, error }
  }
}

/**
 * Tagged Model Service
 * Manages tagged models from 3D spider plot interactions
 */
export class TaggedModelService {
  static async tagModel(
    userId: string,
    sessionId: string,
    configurationId: string,
    modelData: {
      modelId: string
      tagName: string
      tagCategory?: 'user' | 'ml_generated' | 'optimal' | 'interesting'
      circuitParameters: CircuitParameters
      resnormValue?: number
      impedanceSpectrum?: ImpedancePoint[]
      taggingContext?: TaggingContext
      notes?: string
      mlRelevanceScore?: number
    }
  ) {
    const tagData: TaggedModelInsert = {
      user_id: userId,
      session_id: sessionId,
      configuration_id: configurationId,
      model_id: modelData.modelId,
      tag_name: modelData.tagName,
      tag_category: modelData.tagCategory || 'user',
      circuit_parameters: modelData.circuitParameters as unknown as Json,
      resnorm_value: modelData.resnormValue,
      impedance_spectrum: modelData.impedanceSpectrum as unknown as Json,
      tagging_context: modelData.taggingContext as unknown as Json,
      notes: modelData.notes,
      ml_relevance_score: modelData.mlRelevanceScore
    }

    const { data, error } = await supabase
      .from('tagged_models')
      .insert(tagData)
      .select()
      .single()

    return { data, error }
  }

  static async getTaggedModels(
    userId: string,
    sessionId?: string,
    tagCategory?: string,
    limit?: number
  ) {
    let query = supabase
      .from('tagged_models')
      .select('*')
      .eq('user_id', userId)
      .order('tagged_at', { ascending: false })

    if (sessionId) {
      query = query.eq('session_id', sessionId)
    }

    if (tagCategory) {
      query = query.eq('tag_category', tagCategory)
    }

    if (limit) {
      query = query.limit(limit)
    }

    const { data, error } = await query

    return { data, error }
  }

  static async updateTaggedModel(modelId: string, updates: {
    tagName?: string
    tagCategory?: string
    notes?: string
    mlRelevanceScore?: number
    isMLTrainingData?: boolean
  }) {
    const { data, error } = await supabase
      .from('tagged_models')
      .update(updates)
      .eq('id', modelId)
      .select()
      .single()

    return { data, error }
  }

  static async deleteTaggedModel(modelId: string) {
    const { error } = await supabase
      .from('tagged_models')
      .delete()
      .eq('id', modelId)

    return { error }
  }

  static async getMLTrainingCandidates(userId: string, minRelevanceScore: number = 0.7) {
    const { data, error } = await supabase
      .from('tagged_models')
      .select('*')
      .eq('user_id', userId)
      .gte('ml_relevance_score', minRelevanceScore)
      .not('impedance_spectrum', 'is', null)
      .order('ml_relevance_score', { ascending: false })

    return { data, error }
  }

  static async markForMLTraining(modelIds: string[]) {
    const { data, error } = await supabase
      .from('tagged_models')
      .update({ is_ml_training_data: true })
      .in('id', modelIds)
      .select()

    return { data, error }
  }
}

/**
 * Parameter Exploration Service
 * Tracks detailed parameter space exploration sessions
 */
export class ParameterExplorationService {
  static async createExplorationSession(
    userId: string,
    sessionId: string,
    explorationData: {
      name: string
      type?: 'manual' | 'automated' | 'ml_guided'
      parameterRanges: ParameterRanges
      samplingStrategy: SamplingStrategy
      totalCombinations?: number
      mlObjectives?: MLObjectives
      mlConstraints?: MLConstraints
    }
  ) {
    const data: ParameterExplorationSessionInsert = {
      user_id: userId,
      session_id: sessionId,
      exploration_name: explorationData.name,
      exploration_type: explorationData.type || 'manual',
      parameter_ranges: explorationData.parameterRanges,
      sampling_strategy: explorationData.samplingStrategy,
      total_parameter_combinations: explorationData.totalCombinations,
      ml_objectives: explorationData.mlObjectives,
      ml_constraints: explorationData.mlConstraints
    }

    const { data: result, error } = await supabase
      .from('parameter_exploration_sessions')
      .insert(data)
      .select()
      .single()

    return { data: result, error }
  }

  static async updateExplorationProgress(
    explorationId: string,
    progress: {
      successfulComputations?: number
      failedComputations?: number
      progressPercentage?: number
      status?: 'active' | 'completed' | 'paused' | 'failed'
      averageComputationTime?: string
      mlRecommendations?: MLRecommendations
    }
  ) {
    const updates: Record<string, unknown> = { ...progress }
    if (progress.status === 'completed') {
      updates.completed_at = new Date().toISOString()
    }

    const { data, error } = await supabase
      .from('parameter_exploration_sessions')
      .update(updates)
      .eq('id', explorationId)
      .select()
      .single()

    return { data, error }
  }

  static async getExplorationSessions(userId: string, status?: string) {
    let query = supabase
      .from('parameter_exploration_sessions')
      .select('*')
      .eq('user_id', userId)
      .order('started_at', { ascending: false })

    if (status) {
      query = query.eq('status', status)
    }

    const { data, error } = await query

    return { data, error }
  }
}

/**
 * ML Training Dataset Service
 * Manages ML-ready datasets for training
 */
export class MLDatasetService {
  static async createDataset(
    userId: string,
    datasetData: {
      name: string
      description?: string
      sourceConfigurations: string[]
      sourceSessions: string[]
      totalSamples: number
      featureCount: number
      datasetType: 'regression' | 'classification' | 'optimization'
      targetVariable?: string
      featureEngineeringConfig?: FeatureEngineeringConfig
      dataQualityScore?: number
    }
  ) {
    const data: MLTrainingDatasetInsert = {
      user_id: userId,
      dataset_name: datasetData.name,
      description: datasetData.description,
      source_configurations: datasetData.sourceConfigurations,
      source_sessions: datasetData.sourceSessions,
      total_samples: datasetData.totalSamples,
      feature_count: datasetData.featureCount,
      dataset_type: datasetData.datasetType,
      target_variable: datasetData.targetVariable,
      feature_engineering_config: datasetData.featureEngineeringConfig,
      data_quality_score: datasetData.dataQualityScore
    }

    const { data: result, error } = await supabase
      .from('ml_training_datasets')
      .insert(data)
      .select()
      .single()

    return { data: result, error }
  }

  static async generateDatasetFromTags(
    userId: string,
    datasetName: string,
    tagCategories: string[] = ['optimal', 'interesting']
  ) {
    const { data, error } = await supabase
      .rpc('generate_ml_dataset_from_tags', {
        user_uuid: userId,
        dataset_name_param: datasetName,
        target_tag_categories: tagCategories
      })

    return { data, error }
  }

  static async getDatasets(
    userId: string,
    datasetType?: string,
    minSamples: number = 100
  ) {
    const { data, error } = await supabase
      .rpc('get_ml_training_data', {
        user_uuid: userId,
        dataset_type: datasetType,
        min_samples: minSamples
      })

    return { data, error }
  }

  static async updateDatasetPaths(
    datasetId: string,
    paths: {
      featureMatrixPath?: string
      targetVectorPath?: string
      metadataPath?: string
    }
  ) {
    const { data, error } = await supabase
      .from('ml_training_datasets')
      .update({
        feature_matrix_path: paths.featureMatrixPath,
        target_vector_path: paths.targetVectorPath,
        metadata_path: paths.metadataPath,
        updated_at: new Date().toISOString()
      })
      .eq('id', datasetId)
      .select()
      .single()

    return { data, error }
  }

  static async updateDatasetQuality(
    datasetId: string,
    metrics: {
      dataQualityScore: number
      completenessPercentage: number
      outlierPercentage: number
    }
  ) {
    const { data, error } = await supabase
      .from('ml_training_datasets')
      .update({
        data_quality_score: metrics.dataQualityScore,
        completeness_percentage: metrics.completenessPercentage,
        outlier_percentage: metrics.outlierPercentage,
        updated_at: new Date().toISOString()
      })
      .eq('id', datasetId)
      .select()
      .single()

    return { data, error }
  }
}

/**
 * ML Model Service
 * Manages trained machine learning models
 */
export class MLModelService {
  static async saveModel(
    userId: string,
    modelData: {
      name: string
      version?: number
      modelType: string
      description?: string
      trainingDatasetId?: string
      trainingConfig: TrainingConfig
      hyperparameters: Hyperparameters
      modelPath: string
      trainingMetrics?: MLMetrics
      validationMetrics?: MLMetrics
      testMetrics?: MLMetrics
      modelSizeBytes?: number
      trainingDuration?: string
    }
  ) {
    const data: MLModelInsert = {
      user_id: userId,
      model_name: modelData.name,
      model_version: modelData.version || 1,
      model_type: modelData.modelType,
      description: modelData.description,
      training_dataset_id: modelData.trainingDatasetId,
      training_config: modelData.trainingConfig,
      hyperparameters: modelData.hyperparameters,
      model_path: modelData.modelPath,
      training_metrics: modelData.trainingMetrics,
      validation_metrics: modelData.validationMetrics,
      test_metrics: modelData.testMetrics,
      model_size_bytes: modelData.modelSizeBytes,
      training_duration: modelData.trainingDuration
    }

    const { data: result, error } = await supabase
      .from('ml_models')
      .insert(data)
      .select()
      .single()

    return { data: result, error }
  }

  static async getModels(userId: string, modelType?: string, deployedOnly?: boolean) {
    let query = supabase
      .from('ml_models')
      .select(`
        *,
        ml_training_datasets (
          dataset_name,
          total_samples,
          data_quality_score
        )
      `)
      .eq('user_id', userId)
      .order('created_at', { ascending: false })

    if (modelType) {
      query = query.eq('model_type', modelType)
    }

    if (deployedOnly) {
      query = query.eq('is_deployed', true)
    }

    const { data, error } = await query

    return { data, error }
  }

  static async deployModel(
    modelId: string,
    deploymentEndpoint: string
  ) {
    const { data, error } = await supabase
      .from('ml_models')
      .update({
        is_deployed: true,
        deployment_endpoint: deploymentEndpoint,
        updated_at: new Date().toISOString()
      })
      .eq('id', modelId)
      .select()
      .single()

    return { data, error }
  }

  static async updateModelMetrics(
    modelId: string,
    metrics: {
      trainingMetrics?: MLMetrics
      validationMetrics?: MLMetrics
      testMetrics?: MLMetrics
    }
  ) {
    const { data, error } = await supabase
      .from('ml_models')
      .update({
        ...metrics,
        updated_at: new Date().toISOString()
      })
      .eq('id', modelId)
      .select()
      .single()

    return { data, error }
  }
}

/**
 * Visualization Snapshot Service
 * Manages complete visualization state snapshots
 */
export class VisualizationSnapshotService {
  static async saveSnapshot(
    userId: string,
    sessionId: string,
    configurationId: string,
    snapshotData: {
      name: string
      description?: string
      visualizationType: 'spider2d' | 'spider3d' | 'nyquist'
      cameraPosition?: CameraPosition
      filterSettings?: FilterSettings
      displaySettings?: DisplaySettings
      panelStates?: PanelStates
      sliderValues?: SliderValues
      selectedModels?: string[]
      userAnnotations?: UserAnnotations
      isPublic?: boolean
      sharedWith?: string[]
    }
  ) {
    const data: VisualizationSnapshotInsert = {
      user_id: userId,
      session_id: sessionId,
      configuration_id: configurationId,
      snapshot_name: snapshotData.name,
      description: snapshotData.description,
      visualization_type: snapshotData.visualizationType,
      camera_position: snapshotData.cameraPosition,
      filter_settings: snapshotData.filterSettings,
      display_settings: snapshotData.displaySettings,
      panel_states: snapshotData.panelStates,
      slider_values: snapshotData.sliderValues,
      selected_models: snapshotData.selectedModels || [],
      user_annotations: snapshotData.userAnnotations,
      is_public: snapshotData.isPublic || false,
      shared_with: snapshotData.sharedWith || []
    }

    const { data: result, error } = await supabase
      .from('visualization_snapshots')
      .insert(data)
      .select()
      .single()

    return { data: result, error }
  }

  static async getSnapshots(
    userId: string,
    sessionId?: string,
    includePublic: boolean = false
  ) {
    let query = supabase
      .from('visualization_snapshots')
      .select('*')
      .order('created_at', { ascending: false })

    if (includePublic) {
      query = query.or(`user_id.eq.${userId},is_public.eq.true,shared_with.cs.["${userId}"]`)
    } else {
      query = query.eq('user_id', userId)
    }

    if (sessionId) {
      query = query.eq('session_id', sessionId)
    }

    const { data, error } = await query

    return { data, error }
  }

  static async restoreSnapshot(snapshotId: string) {
    const { data, error } = await supabase
      .from('visualization_snapshots')
      .select('*')
      .eq('id', snapshotId)
      .single()

    return { data, error }
  }
}

/**
 * Parameter Optimization Service
 * Manages automated parameter optimization jobs
 */
export class ParameterOptimizationService {
  static async createOptimizationJob(
    userId: string,
    sessionId: string,
    jobData: {
      name: string
      algorithm: 'genetic' | 'bayesian' | 'grid_search' | 'random_search'
      objectiveFunction: string
      parameterBounds: ParameterBounds
      constraints?: OptimizationConstraints
      maxIterations?: number
      convergenceTolerance?: number
      populationSize?: number
      mlModelId?: string
    }
  ) {
    const data: ParameterOptimizationJobInsert = {
      user_id: userId,
      session_id: sessionId,
      ml_model_id: jobData.mlModelId,
      job_name: jobData.name,
      optimization_algorithm: jobData.algorithm,
      objective_function: jobData.objectiveFunction,
      parameter_bounds: jobData.parameterBounds,
      constraints: jobData.constraints,
      max_iterations: jobData.maxIterations || 100,
      convergence_tolerance: jobData.convergenceTolerance || 1e-6,
      population_size: jobData.populationSize || 50
    }

    const { data: result, error } = await supabase
      .from('parameter_optimization_jobs')
      .insert(data)
      .select()
      .single()

    return { data: result, error }
  }

  static async updateJobProgress(
    jobId: string,
    progress: {
      currentIteration?: number
      bestParameters?: Record<string, number>
      bestObjectiveValue?: number
      convergenceHistory?: Record<string, unknown>
      progressPercentage?: number
      status?: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
      errorMessage?: string
      estimatedCompletion?: string
      cpuHoursUsed?: number
    }
  ) {
    const updates: Record<string, unknown> = { ...progress }
    if (progress.status === 'completed' || progress.status === 'failed') {
      updates.completed_at = new Date().toISOString()
    }

    const { data, error } = await supabase
      .from('parameter_optimization_jobs')
      .update(updates)
      .eq('id', jobId)
      .select()
      .single()

    return { data, error }
  }

  static async getOptimizationJobs(
    userId: string,
    status?: string
  ) {
    let query = supabase
      .from('parameter_optimization_jobs')
      .select(`
        *,
        ml_models (
          model_name,
          model_type
        )
      `)
      .eq('user_id', userId)
      .order('started_at', { ascending: false })

    if (status) {
      query = query.eq('status', status)
    }

    const { data, error } = await query

    return { data, error }
  }

  static async cancelOptimizationJob(jobId: string) {
    const { data, error } = await supabase
      .from('parameter_optimization_jobs')
      .update({
        status: 'cancelled',
        completed_at: new Date().toISOString()
      })
      .eq('id', jobId)
      .select()
      .single()

    return { data, error }
  }
}

/**
 * Comprehensive Database Service
 * Main service that aggregates all functionality
 */
export class DatabaseService {
  static session = SessionService
  static taggedModels = TaggedModelService
  static exploration = ParameterExplorationService
  static datasets = MLDatasetService
  static models = MLModelService
  static snapshots = VisualizationSnapshotService
  static optimization = ParameterOptimizationService

  /**
   * Initialize user's first session
   */
  static async initializeUserSession(
    userId: string,
    sessionName: string = 'Default Session'
  ) {
    // Create default session
    const { data: session, error: sessionError } = await SessionService.createSession(
      userId,
      sessionName,
      'Automatically created default session',
      {
        nodeOptions: '--max-old-space-size=8192',
        computeWorkers: navigator.hardwareConcurrency || 4,
        memoryLimit: '8GB',
        cacheSize: '1GB',
        debugMode: false
      },
      {
        groupPortion: 0.2,
        selectedOpacityGroups: [0],
        visualizationType: 'spider3d',
        chromaEnabled: true,
        resnormSpread: 1.0,
        useResnormCenter: false,
        showLabels: true
      },
      {
        maxWorkers: navigator.hardwareConcurrency || 4,
        chunkSize: 1000,
        memoryThreshold: 7000,
        enableCaching: true,
        adaptiveLimits: true
      }
    )

    if (sessionError) {
      return { data: null, error: sessionError }
    }

    return { data: session, error: null }
  }

  /**
   * Get complete user context for ML experimentation
   */
  static async getUserMLContext(userId: string) {
    const [
      { data: sessions, error: sessionsError },
      { data: taggedModels, error: taggedError },
      { data: datasets, error: datasetsError },
      { data: models, error: modelsError },
      { data: explorationSessions, error: explorationError }
    ] = await Promise.all([
      SessionService.getUserSessions(userId),
      TaggedModelService.getTaggedModels(userId),
      MLDatasetService.getDatasets(userId),
      MLModelService.getModels(userId),
      ParameterExplorationService.getExplorationSessions(userId)
    ])

    return {
      data: {
        sessions,
        taggedModels,
        datasets,
        models,
        explorationSessions
      },
      errors: {
        sessionsError,
        taggedError,
        datasetsError,
        modelsError,
        explorationError
      }
    }
  }

  /**
   * Export data for ML training in standardized format
   */
  static async exportMLTrainingData(
    userId: string,
    options: {
      includeTaggedModels?: boolean
      includeComputationResults?: boolean
      includeVisualizationData?: boolean
      datasetId?: string
      format?: 'json' | 'csv' | 'hdf5'
    } = {}
  ) {
    const exportData: Record<string, unknown> = {
      metadata: {
        userId,
        exportedAt: new Date().toISOString(),
        format: options.format || 'json'
      },
      data: {}
    }

    if (options.includeTaggedModels) {
      const { data: taggedModels } = await TaggedModelService.getMLTrainingCandidates(userId)
      exportData.data.taggedModels = taggedModels
    }

    if (options.datasetId) {
      const { data: dataset } = await MLDatasetService.getDatasets(userId)
      exportData.data.dataset = dataset?.find(d => d.dataset_id === options.datasetId)
    }

    // Add computation results and visualization data as needed
    
    return { data: exportData, error: null }
  }
}

export default DatabaseService