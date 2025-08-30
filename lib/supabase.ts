import { createClient } from '@supabase/supabase-js'
import { Database } from './database.types'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

// Create a single supabase client for interacting with your database
export const supabase = createClient<Database>(supabaseUrl, supabaseAnonKey)

// Client-side auth helpers
export const getUser = async () => {
  const { data: { user } } = await supabase.auth.getUser()
  return user
}

export const signIn = async (email: string, password: string) => {
  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password,
  })
  return { data, error }
}

export const signUp = async (email: string, password: string, metadata?: Record<string, any>) => {
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
    options: {
      data: metadata
    }
  })
  return { data, error }
}

export const signOut = async () => {
  const { error } = await supabase.auth.signOut()
  return { error }
}

export const resetPassword = async (email: string) => {
  const { data, error } = await supabase.auth.resetPasswordForEmail(email, {
    redirectTo: `${process.env.NEXT_PUBLIC_SITE_URL}/auth/reset-password`,
  })
  return { data, error }
}

// Profile helpers
export const getUserProfile = async (userId: string) => {
  const { data, error } = await supabase
    .from('user_profiles')
    .select('*')
    .eq('id', userId)
    .single()
  
  return { data, error }
}

export const updateUserProfile = async (userId: string, updates: {
  username?: string
  full_name?: string
  avatar_url?: string
}) => {
  const { data, error } = await supabase
    .from('user_profiles')
    .update(updates)
    .eq('id', userId)
    .select()
    .single()
  
  return { data, error }
}

// Configuration helpers
export const getUserConfigurations = async (userId: string) => {
  const { data, error } = await supabase
    .rpc('get_user_configurations', { user_uuid: userId })
  
  return { data, error }
}

export const saveConfiguration = async (config: Database['public']['Tables']['saved_configurations']['Insert']) => {
  const { data, error } = await supabase
    .from('saved_configurations')
    .insert(config)
    .select()
    .single()
  
  return { data, error }
}

export const updateConfiguration = async (
  configId: string, 
  updates: Database['public']['Tables']['saved_configurations']['Update']
) => {
  const { data, error } = await supabase
    .from('saved_configurations')
    .update(updates)
    .eq('id', configId)
    .select()
    .single()
  
  return { data, error }
}

export const deleteConfiguration = async (configId: string) => {
  const { error } = await supabase
    .from('saved_configurations')
    .delete()
    .eq('id', configId)
  
  return { error }
}

export const getPublicConfigurations = async (limit: number = 50) => {
  const { data, error } = await supabase
    .from('saved_configurations')
    .select(`
      id,
      name,
      description,
      created_at,
      is_computed,
      circuit_parameters,
      user_profiles!inner(full_name, username)
    `)
    .eq('is_public', true)
    .order('created_at', { ascending: false })
    .limit(limit)
  
  return { data, error }
}

// Sharing helpers
export const shareConfiguration = async (
  configId: string,
  sharedWithUserId: string,
  permissionLevel: 'read' | 'write' | 'admin' = 'read',
  message?: string,
  expiresAt?: string
) => {
  const user = await getUser()
  if (!user) throw new Error('User not authenticated')
  
  const { data, error } = await supabase
    .from('shared_configurations')
    .insert({
      configuration_id: configId,
      shared_with: sharedWithUserId,
      shared_by: user.id,
      permission_level: permissionLevel,
      share_message: message,
      expires_at: expiresAt
    })
    .select()
    .single()
  
  return { data, error }
}

export const getSharedConfigurations = async (userId: string) => {
  const { data, error } = await supabase
    .from('shared_configurations_with_details')
    .select('*')
    .eq('shared_with', userId)
    .eq('is_active', true)
  
  return { data, error }
}

// Computation results helpers
export const saveComputationResults = async (results: Database['public']['Tables']['computation_results']['Insert']) => {
  const { data, error } = await supabase
    .from('computation_results')
    .insert(results)
    .select()
    .single()
  
  return { data, error }
}

export const getComputationResults = async (configId: string) => {
  const { data, error } = await supabase
    .from('computation_results')
    .select('*')
    .eq('configuration_id', configId)
    .order('computed_at', { ascending: false })
    .limit(1)
    .single()
  
  return { data, error }
}

// Real-time subscriptions
export const subscribeToUserConfigurations = (
  userId: string,
  callback: (payload: any) => void
) => {
  return supabase
    .channel('user_configurations')
    .on(
      'postgres_changes',
      {
        event: '*',
        schema: 'public',
        table: 'saved_configurations',
        filter: `user_id=eq.${userId}`
      },
      callback
    )
    .subscribe()
}

// Enhanced ML and session management helpers
export const getUserSessions = async (userId: string) => {
  const { data, error } = await supabase
    .from('user_sessions')
    .select('*')
    .eq('user_id', userId)
    .order('last_accessed', { ascending: false })
  
  return { data, error }
}

export const getActiveSession = async (userId: string) => {
  const { data, error } = await supabase
    .rpc('get_active_user_session', { user_uuid: userId })
  
  return { data: data?.[0] || null, error }
}

export const createSession = async (sessionData: Database['public']['Tables']['user_sessions']['Insert']) => {
  const { data, error } = await supabase
    .from('user_sessions')
    .insert(sessionData)
    .select()
    .single()
  
  return { data, error }
}

export const tagModel = async (tagData: Database['public']['Tables']['tagged_models']['Insert']) => {
  const { data, error } = await supabase
    .from('tagged_models')
    .insert(tagData)
    .select()
    .single()
  
  return { data, error }
}

export const getTaggedModels = async (userId: string, sessionId?: string) => {
  let query = supabase
    .from('tagged_models')
    .select('*')
    .eq('user_id', userId)
    .order('tagged_at', { ascending: false })

  if (sessionId) {
    query = query.eq('session_id', sessionId)
  }

  const { data, error } = await query
  return { data, error }
}

export const createMLDataset = async (datasetData: Database['public']['Tables']['ml_training_datasets']['Insert']) => {
  const { data, error } = await supabase
    .from('ml_training_datasets')
    .insert(datasetData)
    .select()
    .single()
  
  return { data, error }
}

export const getMLDatasets = async (userId: string) => {
  const { data, error } = await supabase
    .from('ml_training_datasets')
    .select('*')
    .eq('user_id', userId)
    .eq('is_active', true)
    .order('created_at', { ascending: false })
  
  return { data, error }
}

export const saveMLModel = async (modelData: Database['public']['Tables']['ml_models']['Insert']) => {
  const { data, error } = await supabase
    .from('ml_models')
    .insert(modelData)
    .select()
    .single()
  
  return { data, error }
}

export const getMLModels = async (userId: string) => {
  const { data, error } = await supabase
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
  
  return { data, error }
}

export const saveVisualizationSnapshot = async (snapshotData: Database['public']['Tables']['visualization_snapshots']['Insert']) => {
  const { data, error } = await supabase
    .from('visualization_snapshots')
    .insert(snapshotData)
    .select()
    .single()
  
  return { data, error }
}

export const getVisualizationSnapshots = async (userId: string, sessionId?: string) => {
  let query = supabase
    .from('visualization_snapshots')
    .select('*')
    .eq('user_id', userId)
    .order('created_at', { ascending: false })

  if (sessionId) {
    query = query.eq('session_id', sessionId)
  }

  const { data, error } = await query
  return { data, error }
}

export const createOptimizationJob = async (jobData: Database['public']['Tables']['parameter_optimization_jobs']['Insert']) => {
  const { data, error } = await supabase
    .from('parameter_optimization_jobs')
    .insert(jobData)
    .select()
    .single()
  
  return { data, error }
}

export const getOptimizationJobs = async (userId: string) => {
  const { data, error } = await supabase
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
  
  return { data, error }
}

// Batch operations for performance
export const batchInsertTaggedModels = async (models: Database['public']['Tables']['tagged_models']['Insert'][]) => {
  const { data, error } = await supabase
    .from('tagged_models')
    .insert(models)
    .select()
  
  return { data, error }
}

export const batchUpdateSessionStats = async (sessionId: string, stats: {
  total_computations?: number
  total_models_generated?: number
  total_computation_time?: string
}) => {
  const { data, error } = await supabase
    .from('user_sessions')
    .update({
      ...stats,
      last_accessed: new Date().toISOString(),
      updated_at: new Date().toISOString()
    })
    .eq('id', sessionId)
    .select()
    .single()
  
  return { data, error }
}

// Real-time subscriptions for ML workflows
export const subscribeToMLTrainingProgress = (
  userId: string,
  callback: (payload: any) => void
) => {
  return supabase
    .channel('ml_training_progress')
    .on(
      'postgres_changes',
      {
        event: 'UPDATE',
        schema: 'public',
        table: 'parameter_optimization_jobs',
        filter: `user_id=eq.${userId}`
      },
      callback
    )
    .on(
      'postgres_changes',
      {
        event: 'UPDATE',
        schema: 'public',
        table: 'ml_models',
        filter: `user_id=eq.${userId}`
      },
      callback
    )
    .subscribe()
}

export const subscribeToTaggedModels = (
  userId: string,
  sessionId: string,
  callback: (payload: any) => void
) => {
  return supabase
    .channel('tagged_models_updates')
    .on(
      'postgres_changes',
      {
        event: '*',
        schema: 'public',
        table: 'tagged_models',
        filter: `user_id=eq.${userId}&session_id=eq.${sessionId}`
      },
      callback
    )
    .subscribe()
}

// Server-side client (for API routes)
export const createServerSupabaseClient = () => {
  return createClient<Database>(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!,
    {
      auth: {
        autoRefreshToken: false,
        persistSession: false
      }
    }
  )
}