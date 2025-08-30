/**
 * Session Management Hook
 * 
 * Provides centralized session management for the application
 * Handles user authentication, session creation, and environment variables
 */

import { useState, useEffect, useCallback } from 'react'
import { getUser } from '../../lib/supabase'
import DatabaseService, { SessionEnvironment, VisualizationSettings, PerformanceSettings } from '../../lib/database-service'

export interface SessionState {
  userId: string | null
  sessionId: string | null
  sessionName: string | null
  isLoading: boolean
  error: string | null
  environment: SessionEnvironment | null
  visualizationSettings: VisualizationSettings | null
  performanceSettings: PerformanceSettings | null
}

export interface SessionActions {
  createSession: (name: string, description?: string) => Promise<boolean>
  switchSession: (sessionId: string) => Promise<boolean>
  updateEnvironment: (environment: Partial<SessionEnvironment>) => Promise<boolean>
  updateVisualizationSettings: (settings: Partial<VisualizationSettings>) => Promise<boolean>
  updatePerformanceSettings: (settings: Partial<PerformanceSettings>) => Promise<boolean>
  incrementSessionStats: (computations: number, models: number, time: string) => Promise<boolean>
  tagModel: (modelData: {
    modelId: string
    tagName: string
    tagCategory?: string
    circuitParameters: any
    resnormValue?: number
    impedanceSpectrum?: any
    context?: any
    notes?: string
  }) => Promise<boolean>
  refreshSession: () => Promise<boolean>
}

export const useSessionManagement = () => {
  const [sessionState, setSessionState] = useState<SessionState>({
    userId: null,
    sessionId: null,
    sessionName: null,
    isLoading: true,
    error: null,
    environment: null,
    visualizationSettings: null,
    performanceSettings: null
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
        const { data: activeSession } = await DatabaseService.session.getActiveSession(user.id)
        if (activeSession) {
          // Update session activity
          await DatabaseService.session.updateSessionActivity(activeSession.session_id)
          
          setSessionState({
            userId: user.id,
            sessionId: activeSession.session_id,
            sessionName: activeSession.session_name,
            isLoading: false,
            error: null,
            environment: activeSession.environment_variables as SessionEnvironment,
            visualizationSettings: activeSession.visualization_settings as VisualizationSettings,
            performanceSettings: activeSession.performance_settings as PerformanceSettings
          })
        } else {
          // Create default session with unique timestamp
          let newSession = null;
          let error = null;
          
          // Try creating a session with different naming strategies to avoid unique constraint violations
          const attempts = [
            `Session ${new Date().toLocaleDateString()} ${new Date().toLocaleTimeString()}`,
            `Session ${Date.now()}`,
            `Default Session ${crypto.randomUUID().substring(0, 8)}`
          ];
          
          for (const sessionName of attempts) {
            const result = await DatabaseService.initializeUserSession(user.id, sessionName);
            if (!result.error && result.data) {
              newSession = result.data;
              error = null;
              break;
            } else if (result.error?.message?.includes('duplicate key value violates unique constraint')) {
              // Try next name variation
              continue;
            } else {
              // Different error, stop trying
              error = result.error;
              break;
            }
          }
          
          if (error || !newSession) {
            throw new Error(error?.message || 'Failed to create session after multiple attempts')
          }

          setSessionState({
            userId: user.id,
            sessionId: newSession.id,
            sessionName: newSession.session_name,
            isLoading: false,
            error: null,
            environment: newSession.environment_variables as SessionEnvironment,
            visualizationSettings: newSession.visualization_settings as VisualizationSettings,
            performanceSettings: newSession.performance_settings as PerformanceSettings
          })
        }
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
    if (!sessionState.userId) {
      console.error('No user ID available')
      return false
    }

    try {
      const { data: newSession, error } = await DatabaseService.session.createSession(
        sessionState.userId,
        name,
        description,
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

      if (error || !newSession) {
        console.error('Failed to create session:', error)
        return false
      }

      setSessionState(prev => ({
        ...prev,
        sessionId: newSession.id,
        sessionName: newSession.session_name,
        environment: newSession.environment_variables as SessionEnvironment,
        visualizationSettings: newSession.visualization_settings as VisualizationSettings,
        performanceSettings: newSession.performance_settings as PerformanceSettings
      }))

      return true
    } catch (error) {
      console.error('Error creating session:', error)
      return false
    }
  }, [sessionState.userId])

  // Switch to existing session
  const switchSession = useCallback(async (sessionId: string): Promise<boolean> => {
    try {
      await DatabaseService.session.updateSessionActivity(sessionId)
      
      // Get session details
      const { data: sessions } = await DatabaseService.session.getUserSessions(sessionState.userId!)
      const session = sessions?.find(s => s.id === sessionId)
      
      if (!session) {
        console.error('Session not found')
        return false
      }

      setSessionState(prev => ({
        ...prev,
        sessionId: session.id,
        sessionName: session.session_name,
        environment: session.environment_variables as SessionEnvironment,
        visualizationSettings: session.visualization_settings as VisualizationSettings,
        performanceSettings: session.performance_settings as PerformanceSettings
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
      const { error } = await DatabaseService.session.updateSession(sessionState.sessionId, {
        environment_variables: updatedEnvironment
      })

      if (error) {
        console.error('Failed to update environment:', error)
        return false
      }

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
      const { error } = await DatabaseService.session.updateSession(sessionState.sessionId, {
        visualization_settings: updatedSettings
      })

      if (error) {
        console.error('Failed to update visualization settings:', error)
        return false
      }

      setSessionState(prev => ({
        ...prev,
        visualizationSettings: updatedSettings
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
      const { error } = await DatabaseService.session.updateSession(sessionState.sessionId, {
        performance_settings: updatedSettings
      })

      if (error) {
        console.error('Failed to update performance settings:', error)
        return false
      }

      setSessionState(prev => ({
        ...prev,
        performanceSettings: updatedSettings
      }))

      return true
    } catch (error) {
      console.error('Error updating performance settings:', error)
      return false
    }
  }, [sessionState.sessionId, sessionState.performanceSettings])

  // Increment session statistics
  const incrementSessionStats = useCallback(async (
    computations: number,
    models: number,
    time: string
  ): Promise<boolean> => {
    if (!sessionState.sessionId) return false

    try {
      const { error } = await DatabaseService.session.incrementSessionStats(
        sessionState.sessionId,
        computations,
        models,
        time
      )

      if (error) {
        console.error('Failed to update session stats:', error)
        return false
      }

      return true
    } catch (error) {
      console.error('Error updating session stats:', error)
      return false
    }
  }, [sessionState.sessionId])

  // Tag a model
  const tagModel = useCallback(async (modelData: {
    modelId: string
    tagName: string
    tagCategory?: string
    circuitParameters: any
    resnormValue?: number
    impedanceSpectrum?: any
    context?: any
    notes?: string
  }): Promise<boolean> => {
    if (!sessionState.userId || !sessionState.sessionId) return false

    try {
      // Use simplified approach - direct Supabase insert to match simplified schema
      const { supabase } = await import('../../lib/supabase')
      const { error } = await supabase
        .from('tagged_models')
        .insert({
          user_id: sessionState.userId,
          session_id: sessionState.sessionId,
          model_id: modelData.modelId,
          tag_name: modelData.tagName,
          tag_category: modelData.tagCategory || 'user',
          circuit_parameters: modelData.circuitParameters,
          resnorm_value: modelData.resnormValue,
          notes: modelData.notes,
          is_interesting: true
        })

      if (error) {
        console.error('Failed to tag model:', error)
        return false
      }

      return true
    } catch (error) {
      console.error('Error tagging model:', error)
      return false
    }
  }, [sessionState.userId, sessionState.sessionId])

  // Refresh session data
  const refreshSession = useCallback(async (): Promise<boolean> => {
    if (!sessionState.userId || !sessionState.sessionId) return false

    try {
      const { data: activeSession } = await DatabaseService.session.getActiveSession(sessionState.userId)
      if (activeSession && activeSession.session_id === sessionState.sessionId) {
        setSessionState(prev => ({
          ...prev,
          environment: activeSession.environment_variables as SessionEnvironment,
          visualizationSettings: activeSession.visualization_settings as VisualizationSettings,
          performanceSettings: activeSession.performance_settings as PerformanceSettings
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
    incrementSessionStats,
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