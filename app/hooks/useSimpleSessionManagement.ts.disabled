/**
 * Simplified Session Management Hook
 * 
 * Provides session management focused on replacing localStorage functionality
 * - Environment variables and settings persistence
 * - Saved profiles management
 * - Lightweight tagged models
 */

import { useState, useEffect, useCallback } from 'react'
import { getUser } from '../../lib/supabase'
import SessionDatabaseService, { 
  SessionEnvironment, 
  VisualizationSettings, 
  PerformanceSettings,
  SavedProfile
} from '../../lib/session-database-service'

export interface SessionState {
  userId: string | null
  sessionId: string | null
  sessionName: string | null
  isLoading: boolean
  error: string | null
  environment: SessionEnvironment | null
  visualizationSettings: VisualizationSettings | null
  performanceSettings: PerformanceSettings | null
  savedProfiles: SavedProfile[]
}

export interface SessionActions {
  createSession: (name: string, description?: string) => Promise<boolean>
  switchSession: (sessionId: string) => Promise<boolean>
  updateEnvironment: (environment: Partial<SessionEnvironment>) => Promise<boolean>
  updateVisualizationSettings: (settings: Partial<VisualizationSettings>) => Promise<boolean>
  updatePerformanceSettings: (settings: Partial<PerformanceSettings>) => Promise<boolean>
  addSavedProfile: (profile: Omit<SavedProfile, 'id' | 'createdAt'>) => Promise<boolean>
  removeSavedProfile: (profileId: string) => Promise<boolean>
  tagModel: (modelData: {
    modelId: string
    tagName: string
    tagCategory?: string
    circuitParameters: Record<string, number>
    resnormValue?: number
    notes?: string
    isInteresting?: boolean
  }) => Promise<boolean>
  refreshSession: () => Promise<boolean>
}

export const useSimpleSessionManagement = () => {
  const [sessionState, setSessionState] = useState<SessionState>({
    userId: null,
    sessionId: null,
    sessionName: null,
    isLoading: true,
    error: null,
    environment: null,
    visualizationSettings: null,
    performanceSettings: null,
    savedProfiles: []
  })

  // Initialize session on mount
  useEffect(() => {
    const initializeSession = async () => {
      setSessionState(prev => ({ ...prev, isLoading: true, error: null }))
      
      try {
        const user = await getUser()
        if (!user) {
          setSessionState(prev => ({
            ...prev,
            isLoading: false,
            error: 'User not authenticated'
          }))
          return
        }

        // Get or create active session
        const { data: newSession, error } = await SessionDatabaseService.initializeUserSession(
          user.id,
          `Session ${new Date().toLocaleDateString()}`
        )
        
        if (error || !newSession) {
          throw new Error(error?.message || 'Failed to create session')
        }

        await SessionDatabaseService.session.updateSessionActivity(newSession.id)

        setSessionState({
          userId: user.id,
          sessionId: newSession.id,
          sessionName: newSession.session_name,
          isLoading: false,
          error: null,
          environment: newSession.environment_variables as SessionEnvironment,
          visualizationSettings: newSession.visualization_settings as VisualizationSettings,
          performanceSettings: newSession.performance_settings as PerformanceSettings,
          savedProfiles: []
        })
      } catch (error) {
        console.error('Failed to initialize session:', error)
        setSessionState(prev => ({
          ...prev,
          isLoading: false,
          error: error instanceof Error ? error.message : 'Unknown error'
        }))
      }
    }

    initializeSession()
  }, [])

  // Create new session
  const createSession = useCallback(async (name: string, description?: string): Promise<boolean> => {
    if (!sessionState.userId) return false

    try {
      const { data: newSession, error } = await SessionDatabaseService.session.createSession(
        sessionState.userId,
        name,
        description
      )

      if (error || !newSession) return false

      setSessionState(prev => ({
        ...prev,
        sessionId: newSession.id,
        sessionName: newSession.session_name,
        environment: newSession.environment_variables as SessionEnvironment,
        visualizationSettings: newSession.visualization_settings as VisualizationSettings,
        performanceSettings: newSession.performance_settings as PerformanceSettings,
        savedProfiles: []
      }))

      return true
    } catch (error) {
      console.error('Error creating session:', error)
      return false
    }
  }, [sessionState.userId])

  // Switch to existing session
  const switchSession = useCallback(async (sessionId: string): Promise<boolean> => {
    if (!sessionState.userId) return false

    try {
      await SessionDatabaseService.session.updateSessionActivity(sessionId)
      
      const { data: sessions } = await SessionDatabaseService.session.getUserSessions(sessionState.userId)
      const session = sessions?.find(s => s.id === sessionId)
      
      if (!session) return false

      setSessionState(prev => ({
        ...prev,
        sessionId: session.id,
        sessionName: session.session_name,
        environment: session.environment_variables as SessionEnvironment,
        visualizationSettings: session.visualization_settings as VisualizationSettings,
        performanceSettings: session.performance_settings as PerformanceSettings,
        savedProfiles: []
      }))

      return true
    } catch (error) {
      console.error('Error switching session:', error)
      return false
    }
  }, [sessionState.userId])

  // Update environment variables
  const updateEnvironment = useCallback(async (environment: Partial<SessionEnvironment>): Promise<boolean> => {
    if (!sessionState.sessionId) return false

    try {
      const updatedEnvironment = { ...sessionState.environment, ...environment }
      const { error } = await SessionDatabaseService.session.updateSession(sessionState.sessionId, {
        environment_variables: updatedEnvironment
      })

      if (error) return false

      setSessionState(prev => ({
        ...prev,
        environment: updatedEnvironment
      }))

      return true
    } catch (error) {
      console.error('Error updating environment:', error)
      return false
    }
  }, [sessionState.sessionId, sessionState.environment])

  // Update visualization settings
  const updateVisualizationSettings = useCallback(async (settings: Partial<VisualizationSettings>): Promise<boolean> => {
    if (!sessionState.sessionId) return false

    try {
      const updatedSettings = { ...sessionState.visualizationSettings, ...settings }
      const { error } = await SessionDatabaseService.session.updateSession(sessionState.sessionId, {
        visualization_settings: updatedSettings as VisualizationSettings
      })

      if (error) return false

      setSessionState(prev => ({
        ...prev,
        visualizationSettings: updatedSettings as VisualizationSettings
      }))

      return true
    } catch (error) {
      console.error('Error updating visualization settings:', error)
      return false
    }
  }, [sessionState.sessionId, sessionState.visualizationSettings])

  // Update performance settings
  const updatePerformanceSettings = useCallback(async (settings: Partial<PerformanceSettings>): Promise<boolean> => {
    if (!sessionState.sessionId) return false

    try {
      const updatedSettings = { ...sessionState.performanceSettings, ...settings }
      const { error } = await SessionDatabaseService.session.updateSession(sessionState.sessionId, {
        performance_settings: updatedSettings as PerformanceSettings
      })

      if (error) return false

      setSessionState(prev => ({
        ...prev,
        performanceSettings: updatedSettings as PerformanceSettings
      }))

      return true
    } catch (error) {
      console.error('Error updating performance settings:', error)
      return false
    }
  }, [sessionState.sessionId, sessionState.performanceSettings])

  // Add saved profile
  const addSavedProfile = useCallback(async (profile: Omit<SavedProfile, 'id' | 'createdAt'>): Promise<boolean> => {
    if (!sessionState.sessionId) return false

    try {
      const newProfile: SavedProfile = {
        ...profile,
        id: `profile_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        createdAt: new Date().toISOString()
      }

      const { data, error } = await SessionDatabaseService.session.addSavedProfile(sessionState.sessionId, newProfile)

      if (error || !data) return false

      setSessionState(prev => ({
        ...prev,
        savedProfiles: [newProfile, ...prev.savedProfiles]
      }))

      return true
    } catch (error) {
      console.error('Error adding saved profile:', error)
      return false
    }
  }, [sessionState.sessionId])

  // Remove saved profile
  const removeSavedProfile = useCallback(async (profileId: string): Promise<boolean> => {
    if (!sessionState.sessionId) return false

    try {
      const { data, error } = await SessionDatabaseService.session.removeSavedProfile(sessionState.sessionId, profileId)

      if (error || !data) return false

      setSessionState(prev => ({
        ...prev,
        savedProfiles: prev.savedProfiles.filter(p => p.id !== profileId)
      }))

      return true
    } catch (error) {
      console.error('Error removing saved profile:', error)
      return false
    }
  }, [sessionState.sessionId])

  // Tag a model (lightweight)
  const tagModel = useCallback(async (modelData: {
    modelId: string
    tagName: string
    tagCategory?: string
    circuitParameters: Record<string, number>
    resnormValue?: number
    notes?: string
    isInteresting?: boolean
  }): Promise<boolean> => {
    if (!sessionState.userId || !sessionState.sessionId) return false

    try {
      const { error } = await SessionDatabaseService.taggedModels.tagModel(
        sessionState.userId,
        sessionState.sessionId,
        modelData
      )

      return !error
    } catch (error) {
      console.error('Error tagging model:', error)
      return false
    }
  }, [sessionState.userId, sessionState.sessionId])

  // Refresh session data
  const refreshSession = useCallback(async (): Promise<boolean> => {
    if (!sessionState.userId || !sessionState.sessionId) return false

    try {
      const { data: activeSession } = await SessionDatabaseService.session.getActiveSession(sessionState.userId)
      if (activeSession && activeSession.id === sessionState.sessionId) {
        setSessionState(prev => ({
          ...prev,
          environment: activeSession.environment_variables as SessionEnvironment,
          visualizationSettings: activeSession.visualization_settings as VisualizationSettings,
          performanceSettings: activeSession.performance_settings as PerformanceSettings,
          savedProfiles: prev.savedProfiles // Keep existing saved profiles
        }))
        return true
      }
      return false
    } catch (error) {
      console.error('Error refreshing session:', error)
      return false
    }
  }, [sessionState.userId, sessionState.sessionId])

  const actions: SessionActions = {
    createSession,
    switchSession,
    updateEnvironment,
    updateVisualizationSettings,
    updatePerformanceSettings,
    addSavedProfile,
    removeSavedProfile,
    tagModel,
    refreshSession
  }

  return {
    sessionState,
    actions,
    isReady: !sessionState.isLoading && sessionState.sessionId !== null,
    hasError: sessionState.error !== null
  }
}