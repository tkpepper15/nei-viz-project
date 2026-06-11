/**
 * Self-hosted session management — localStorage backed, no network calls.
 */

import { useState, useEffect, useCallback } from 'react'
import { TaggedModelsService } from '../../lib/taggedModelsService'

export interface SessionState {
  userId: string | null
  sessionId: string | null
  sessionName: string | null
  currentCircuitConfigId: string | null
  isLoading: boolean
  error: string | null
}

export interface SessionActions {
  tagModel: (modelData: {
    circuitConfigId: string
    modelId: string
    tagName: string
    tagCategory?: string
    circuitParameters: Record<string, unknown>
    resnormValue?: number
    notes?: string
    isInteresting?: boolean
  }) => Promise<boolean>

  untagModel: (untagData: {
    modelId: string
    circuitConfigId: string
  }) => Promise<boolean>

  setActiveCircuitConfig: (configId: string | null) => Promise<void>
}

const SESSION_KEY = 'nei-viz-session-state'

function loadSessionConfig(): string | null {
  if (typeof window === 'undefined') return null
  try {
    const raw = localStorage.getItem(SESSION_KEY)
    return raw ? JSON.parse(raw).currentCircuitConfigId ?? null : null
  } catch {
    return null
  }
}

function saveSessionConfig(configId: string | null): void {
  if (typeof window === 'undefined') return
  localStorage.setItem(SESSION_KEY, JSON.stringify({ currentCircuitConfigId: configId }))
}

export const useSessionManagement = () => {
  const [sessionState, setSessionState] = useState<SessionState>({
    userId: 'local-user',
    sessionId: 'local-session',
    sessionName: 'Local Session',
    currentCircuitConfigId: null,
    isLoading: true,
    error: null,
  })

  useEffect(() => {
    const configId = loadSessionConfig()
    setSessionState(prev => ({ ...prev, currentCircuitConfigId: configId, isLoading: false }))
  }, [])

  const tagModel = useCallback(async (modelData: {
    circuitConfigId: string
    modelId: string
    tagName: string
    tagCategory?: string
    circuitParameters: Record<string, unknown>
    resnormValue?: number
    notes?: string
    isInteresting?: boolean
  }): Promise<boolean> => {
    try {
      await TaggedModelsService.createTaggedModel('local-user', {
        circuitConfigId: modelData.circuitConfigId,
        modelId: modelData.modelId,
        tagName: modelData.tagName,
        resnormValue: modelData.resnormValue ?? 0,
        notes: modelData.notes,
        isInteresting: modelData.isInteresting,
        metadata: modelData.circuitParameters,
      })
      return true
    } catch {
      return false
    }
  }, [])

  const untagModel = useCallback(async (untagData: {
    modelId: string
    circuitConfigId: string
  }): Promise<boolean> => {
    try {
      const all = await TaggedModelsService.getAllUserTaggedModels('local-user')
      const match = all.find(
        m => m.modelId === untagData.modelId && m.circuitConfigId === untagData.circuitConfigId,
      )
      if (match) {
        await TaggedModelsService.deleteTaggedModel(match.id)
      }
      return true
    } catch {
      return false
    }
  }, [])

  const setActiveCircuitConfig = useCallback(async (configId: string | null) => {
    saveSessionConfig(configId)
    setSessionState(prev => ({ ...prev, currentCircuitConfigId: configId }))
  }, [])

  const actions: SessionActions = { tagModel, untagModel, setActiveCircuitConfig }

  return {
    sessionState,
    actions,
    isReady: !sessionState.isLoading,
    hasError: false,
  }
}
