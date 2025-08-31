/**
 * Simplified Session Database Service
 * 
 * Provides clean interfaces for session-only database operations:
 * - Session management with environment variables  
 * - Saved profiles (what was in localStorage)
 * - Lightweight tagged models (references only)
 * - User preferences and settings
 */

import { supabase } from './supabase'

// Simple type definitions for session-only storage
export interface SessionEnvironment {
  nodeOptions?: string
  computeWorkers?: number
  memoryLimit?: string
  cacheSize?: string
  debugMode?: boolean
  [key: string]: string | number | boolean | undefined
}

export interface VisualizationSettings {
  groupPortion: number
  selectedOpacityGroups: number[]
  visualizationType: 'spider2d' | 'spider3d' | 'nyquist'
  chromaEnabled: boolean
  resnormSpread: number
  useResnormCenter: boolean
  showLabels: boolean
  [key: string]: string | number | boolean | number[] | undefined
}

export interface PerformanceSettings {
  maxWorkers: number
  chunkSize: number
  memoryThreshold: number
  enableCaching: boolean
  adaptiveLimits: boolean
  [key: string]: string | number | boolean | undefined
}

export interface SavedProfile {
  id: string
  name: string
  parameters: {
    Rsh: number
    Ra: number
    Ca: number
    Rb: number
    Cb: number
  }
  frequencyRange: [number, number]
  description?: string
  createdAt: string
}

export interface UserSession {
  id: string
  user_id: string
  session_name: string
  description?: string
  environment_variables: SessionEnvironment
  visualization_settings: VisualizationSettings
  performance_settings: PerformanceSettings
  saved_profiles: SavedProfile[]
  is_active: boolean
  last_accessed: string
  created_at: string
  updated_at: string
}

export interface TaggedModel {
  id: string
  user_id: string
  session_id: string
  model_id: string
  tag_name: string
  tag_category: string
  circuit_parameters: {
    Rsh: number
    Ra: number
    Ca: number
    Rb: number
    Cb: number
  }
  resnorm_value?: number
  tagged_at: string
  notes?: string
  is_interesting: boolean
}

/**
 * Session Management Service
 */
class SessionService {
  // Create a new session
  async createSession(
    userId: string,
    sessionName: string,
    description?: string,
    environment?: SessionEnvironment,
    visualization?: VisualizationSettings,
    performance?: PerformanceSettings
  ) {
    const { data, error } = await supabase
      .from('user_sessions')
      .insert({
        user_id: userId,
        session_name: sessionName,
        description,
        environment_variables: environment || {},
        visualization_settings: visualization || {
          groupPortion: 0.2,
          selectedOpacityGroups: [0],
          visualizationType: 'spider3d',
          chromaEnabled: true,
          resnormSpread: 1.0,
          useResnormCenter: false,
          showLabels: true
        },
        performance_settings: performance || {
          maxWorkers: navigator.hardwareConcurrency || 4,
          chunkSize: 1000,
          memoryThreshold: 7000,
          enableCaching: true,
          adaptiveLimits: true
        },
        saved_profiles: []
      })
      .select()
      .single()

    return { data, error }
  }

  // Get user's active session
  async getActiveSession(userId: string) {
    const { data, error } = await supabase
      .from('user_sessions')
      .select('*')
      .eq('user_id', userId)
      .eq('is_active', true)
      .order('last_accessed', { ascending: false })
      .limit(1)
      .single()

    return { data, error }
  }

  // Get all user sessions
  async getUserSessions(userId: string) {
    const { data, error } = await supabase
      .from('user_sessions')
      .select('*')
      .eq('user_id', userId)
      .order('last_accessed', { ascending: false })

    return { data, error }
  }

  // Update session data
  async updateSession(sessionId: string, updates: Partial<UserSession>) {
    const { data, error } = await supabase
      .from('user_sessions')
      .update(updates)
      .eq('id', sessionId)
      .select()
      .single()

    return { data, error }
  }

  // Update session activity (last accessed)
  async updateSessionActivity(sessionId: string) {
    const { error } = await supabase
      .from('user_sessions')
      .update({ last_accessed: new Date().toISOString() })
      .eq('id', sessionId)

    return { error }
  }

  // Add a saved profile to session
  async addSavedProfile(sessionId: string, profile: SavedProfile) {
    // First get current profiles
    const { data: session, error: fetchError } = await supabase
      .from('user_sessions')
      .select('saved_profiles')
      .eq('id', sessionId)
      .single()

    if (fetchError) return { error: fetchError }

    const currentProfiles = session.saved_profiles || []
    const updatedProfiles = [...currentProfiles, profile]

    const { data, error } = await supabase
      .from('user_sessions')
      .update({ saved_profiles: updatedProfiles })
      .eq('id', sessionId)
      .select()
      .single()

    return { data, error }
  }

  // Remove a saved profile
  async removeSavedProfile(sessionId: string, profileId: string) {
    // First get current profiles
    const { data: session, error: fetchError } = await supabase
      .from('user_sessions')
      .select('saved_profiles')
      .eq('id', sessionId)
      .single()

    if (fetchError) return { error: fetchError }

    const currentProfiles = session.saved_profiles || []
    const updatedProfiles = currentProfiles.filter((p: SavedProfile) => p.id !== profileId)

    const { data, error } = await supabase
      .from('user_sessions')
      .update({ saved_profiles: updatedProfiles })
      .eq('id', sessionId)
      .select()
      .single()

    return { data, error }
  }
}

/**
 * Tagged Models Service (Lightweight)
 */
class TaggedModelService {
  // Tag a model (lightweight - just reference data)
  async tagModel(
    userId: string,
    sessionId: string,
    modelData: {
      modelId: string
      tagName: string
      tagCategory?: string
      circuitParameters: Record<string, number>
      resnormValue?: number
      notes?: string
      isInteresting?: boolean
    }
  ) {
    const { data, error } = await supabase
      .from('tagged_models')
      .insert({
        user_id: userId,
        session_id: sessionId,
        model_id: modelData.modelId,
        tag_name: modelData.tagName,
        tag_category: modelData.tagCategory || 'user',
        circuit_parameters: modelData.circuitParameters,
        resnorm_value: modelData.resnormValue,
        notes: modelData.notes,
        is_interesting: modelData.isInteresting || false
      })
      .select()
      .single()

    return { data, error }
  }

  // Get tagged models for a session
  async getSessionTaggedModels(sessionId: string) {
    const { data, error } = await supabase
      .from('tagged_models')
      .select('*')
      .eq('session_id', sessionId)
      .order('tagged_at', { ascending: false })

    return { data, error }
  }

  // Get all user's tagged models
  async getUserTaggedModels(userId: string) {
    const { data, error } = await supabase
      .from('tagged_models')
      .select('*')
      .eq('user_id', userId)
      .order('tagged_at', { ascending: false })

    return { data, error }
  }

  // Get interesting models (for local ML processing)
  async getInterestingModels(userId: string) {
    const { data, error } = await supabase
      .from('tagged_models')
      .select('*')
      .eq('user_id', userId)
      .eq('is_interesting', true)
      .order('tagged_at', { ascending: false })

    return { data, error }
  }

  // Delete a tagged model
  async deleteTaggedModel(modelId: string) {
    const { error } = await supabase
      .from('tagged_models')
      .delete()
      .eq('id', modelId)

    return { error }
  }
}

/**
 * Simplified Database Service
 */
class SessionDatabaseService {
  public session = new SessionService()
  public taggedModels = new TaggedModelService()

  // Initialize user session (main entry point)
  async initializeUserSession(userId: string, sessionName: string) {
    // Try to get existing active session first
    const { data: existingSession } = await this.session.getActiveSession(userId)
    
    if (existingSession) {
      return { data: existingSession, error: null }
    }

    // Create new session if none exists
    return await this.session.createSession(userId, sessionName)
  }

  // Get session summary for user
  async getSessionSummary(userId: string) {
    const { data: sessions, error } = await this.session.getUserSessions(userId)
    if (error) return { data: null, error }

    const summary = {
      totalSessions: sessions?.length || 0,
      activeSession: sessions?.find(s => s.is_active),
      recentSessions: sessions?.slice(0, 5) || []
    }

    return { data: summary, error: null }
  }
}

const sessionDatabaseService = new SessionDatabaseService()
export default sessionDatabaseService