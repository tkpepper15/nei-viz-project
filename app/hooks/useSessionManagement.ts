/**
 * Simplified Session Management Hook for Deployment
 * 
 * Contains only the essential tagging functionality that's working
 */

import { useState, useEffect, useCallback } from 'react'
import { getUser } from '../../lib/supabase'

export interface SessionState {
  userId: string | null
  sessionId: string | null
  sessionName: string | null
  isLoading: boolean
  error: string | null
}

export interface SessionActions {
  tagModel: (modelData: {
    modelId: string
    tagName: string
    tagCategory?: string
    circuitParameters: Record<string, unknown>
    resnormValue?: number
    notes?: string
    configurationId?: string
  }) => Promise<boolean>
}

export const useSessionManagement = () => {
  const [sessionState, setSessionState] = useState<SessionState>({
    userId: null,
    sessionId: null,
    sessionName: null,
    isLoading: true,
    error: null
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

        // Get or create active session - use direct Supabase call
        const { supabase } = await import('../../lib/supabase');
        const { data: activeSession } = await supabase
          .from('user_sessions')
          .select('*')
          .eq('user_id', user.id)
          .eq('is_active', true)
          .order('last_accessed', { ascending: false })
          .limit(1)
          .single();
        
        if (activeSession) {
          // Update session activity
          await supabase
            .from('user_sessions')
            .update({ 
              last_accessed: new Date().toISOString(),
              is_active: true 
            })
            .eq('id', activeSession.id);
          
          setSessionState({
            userId: user.id,
            sessionId: activeSession.id,
            sessionName: activeSession.session_name,
            isLoading: false,
            error: null
          })
        } else {
          // Create a simple session
          const { data: newSession, error: sessionError } = await supabase
            .from('user_sessions')
            .insert({
              user_id: user.id,
              session_name: `Session ${new Date().toLocaleDateString()}`,
              description: 'Auto-created session'
            })
            .select()
            .single();

          if (sessionError || !newSession) {
            throw new Error(sessionError?.message || 'Failed to create session')
          }

          setSessionState({
            userId: user.id,
            sessionId: newSession.id,
            sessionName: newSession.session_name,
            isLoading: false,
            error: null
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

  // Tag a model - the core functionality that was working
  const tagModel = useCallback(async (modelData: {
    modelId: string
    tagName: string
    tagCategory?: string
    circuitParameters: Record<string, unknown>
    resnormValue?: number
    notes?: string
    configurationId?: string
  }): Promise<boolean> => {
    console.log('🏷️ TagModel called with:', { 
      modelId: modelData.modelId, 
      tagName: modelData.tagName,
      hasUserId: !!sessionState.userId,
      hasSessionId: !!sessionState.sessionId,
      userId: sessionState.userId,
      sessionId: sessionState.sessionId,
      isLoading: sessionState.isLoading,
      error: sessionState.error
    });

    // Debug: Let's also test basic Supabase connectivity
    console.log('🔗 Testing basic Supabase connectivity...');
    try {
      const { supabase: testSupabase } = await import('../../lib/supabase');
      const { data: authUser, error: authError } = await testSupabase.auth.getUser();
      console.log('🔗 Auth check:', { user: authUser?.user?.id, error: authError });
    } catch (connectError) {
      console.error('🔗 Supabase connection failed:', connectError);
    }

    if (!sessionState.userId) {
      console.error('❌ No userId available for tagging');
      return false;
    }

    if (!sessionState.sessionId) {
      console.error('❌ No sessionId available for tagging');
      return false;
    }

    try {
      // Use simplified approach - direct Supabase insert to match simplified schema
      const { supabase } = await import('../../lib/supabase')
      
      const insertData = {
        user_id: sessionState.userId,
        session_id: sessionState.sessionId,
        configuration_id: modelData.configurationId || 'default', // Use passed configurationId or default
        model_id: modelData.modelId,
        tag_name: modelData.tagName,
        tag_category: modelData.tagCategory || 'user',
        circuit_parameters: JSON.stringify(modelData.circuitParameters), // Convert to JSON string like existing data
        resnorm_value: modelData.resnormValue,
        notes: modelData.notes,
        is_interesting: false // Add missing field from schema
      };

      console.log('🏷️ Attempting to insert tag with data:', insertData);

      // First, let's check if the table exists by testing a simple select
      console.log('🔍 Testing table access first...');
      const { data: testData, error: testError } = await supabase
        .from('tagged_models')
        .select('*', { count: 'exact', head: true });

      console.log('🔍 Table access test result:', { count: testData, error: testError });

      if (testError) {
        console.error('❌ Table access test failed:', testError);
        console.error('❌ Table test error details:', JSON.stringify(testError, null, 2));
        return false;
      }

      const { data, error } = await supabase
        .from('tagged_models')
        .insert(insertData)
        .select()

      console.log('🏷️ Full Supabase response:', { data, error });

      if (error) {
        console.error('❌ Supabase error tagging model:', error);
        console.error('❌ Raw error object:', JSON.stringify(error, null, 2));
        console.error('❌ Error details:', { 
          message: error.message, 
          details: error.details, 
          hint: error.hint, 
          code: error.code 
        });
        return false
      }

      console.log('✅ Successfully tagged model:', data);
      return true
    } catch (error) {
      console.error('❌ Exception while tagging model:', error);
      if (error instanceof Error) {
        console.error('Error message:', error.message);
        console.error('Error stack:', error.stack);
      }
      return false
    }
  }, [sessionState.userId, sessionState.sessionId, sessionState.isLoading, sessionState.error])

  const actions: SessionActions = {
    tagModel
  }

  return {
    sessionState,
    actions,
    isReady: !sessionState.isLoading && sessionState.sessionId !== null,
    hasError: sessionState.error !== null
  }
}