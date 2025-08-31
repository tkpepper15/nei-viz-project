import { supabase } from './supabase';
import { SavedProfile } from '../app/components/circuit-simulator/types/savedProfiles';
import { CircuitParameters } from '../app/components/circuit-simulator/types/parameters';
import { Json } from './database.types';

export interface DatabaseProfile {
  id: string;
  user_id: string;
  name: string;
  description?: string;
  parameters: CircuitParameters;
  grid_size: number;
  min_freq: number;
  max_freq: number;
  num_points: number;
  is_computed: boolean;
  computation_time?: number;
  total_points?: number;
  valid_points?: number;
  computation_results?: unknown;
  created_at: string;
  updated_at: string;
}

export class ProfilesService {
  static async getUserProfiles(userId: string): Promise<SavedProfile[]> {
    // Enhanced debugging and validation
    console.log('📦 ProfilesService: Fetching profiles for user:', userId);
    
    // Validate auth state first
    const { data: { user } } = await supabase.auth.getUser();
    
    // Only log auth details if there's an issue
    if (!user || user.id !== userId) {
      console.log('🔐 Auth state check:', { 
        hasUser: !!user, 
        authUserId: user?.id,
        requestedUserId: userId,
        userRole: user?.role,
        isAnonymous: user?.is_anonymous 
      });
    }

    if (!user) {
      console.error('❌ No authenticated user found');
      throw new Error('User not authenticated');
    }

    if (user.id !== userId) {
      console.warn('⚠️ Auth user ID mismatch:', { authId: user.id, requestedId: userId });
    }

    // Test table existence first (reduced logging when successful)
    try {
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      const { data: tableCheck, error: tableError } = await supabase
        .from('saved_configurations')
        .select('count')
        .limit(1);
      
      if (tableError) {
        console.error('❌ Table access error:', {
          message: tableError.message,
          code: tableError.code,
          hint: tableError.hint,
          details: tableError.details
        });
        
        if (tableError.code === 'PGRST116' || tableError.message?.includes('does not exist')) {
          throw new Error('saved_configurations table does not exist - run database migration');
        }
        
        throw tableError;
      }
      
      // Only log success in debug scenarios
      if (process.env.NODE_ENV === 'development') {
        console.log('✅ Table access confirmed');
      }
    } catch (e) {
      console.error('❌ Table check failed:', e);
      throw e;
    }

    // Now attempt to fetch user profiles
    const { data, error } = await supabase
      .from('saved_configurations')
      .select('*')
      .eq('user_id', userId)
      .order('updated_at', { ascending: false });

    console.log('📊 Query result:', { 
      hasData: !!data, 
      dataLength: data?.length || 0,
      hasError: !!error 
    });

    if (error) {
      console.error('❌ Error fetching user profiles from saved_configurations table:');
      console.error('Full error object:', JSON.stringify(error, null, 2));
      console.error('Error details:', {
        message: error.message,
        code: error.code,
        hint: error.hint,
        details: error.details
      });
      console.error('Query details:', {
        table: 'saved_configurations',
        userId: userId,
        orderBy: 'updated_at'
      });
      throw error;
    }

    if (!data) {
      console.log('📭 No profiles found for user, returning empty array');
      return [];
    }

    const profiles = data.map(this.convertDatabaseProfileToSavedProfile);
    console.log('✅ Successfully fetched', profiles.length, 'profiles');
    
    return profiles;
  }

  static async createProfile(
    userId: string, 
    name: string, 
    parameters: CircuitParameters,
    gridSize: number,
    minFreq: number,
    maxFreq: number,
    numPoints: number,
    description?: string
  ): Promise<SavedProfile> {
    const { data, error } = await supabase
      .from('saved_configurations')
      .insert({
        user_id: userId,
        name: name,
        description,
        circuit_parameters: parameters as unknown as Json,
        grid_size: gridSize,
        min_frequency: minFreq,
        max_frequency: maxFreq,
        num_points: numPoints,
        is_computed: false
      })
      .select()
      .single();

    if (error) {
      console.error('Error creating profile:', error);
      console.error('Error details:', error.message, error.code, error.hint);
      throw error;
    }

    return this.convertDatabaseProfileToSavedProfile(data);
  }

  static async updateProfile(
    profileId: string, 
    updates: Partial<Pick<DatabaseProfile, 'name' | 'description' | 'parameters' | 'is_computed' | 'computation_results'>>
  ): Promise<SavedProfile> {
    // Map updates to correct column names  
    const mappedUpdates: Record<string, unknown> = {};
    if (updates.name) mappedUpdates.name = updates.name;
    if (updates.description !== undefined) mappedUpdates.description = updates.description;
    if (updates.parameters) mappedUpdates.circuit_parameters = updates.parameters as unknown as Json;
    mappedUpdates.updated_at = new Date().toISOString();

    const { data, error } = await supabase
      .from('saved_configurations')
      .update(mappedUpdates)
      .eq('id', profileId)
      .select()
      .single();

    if (error) {
      console.error('Error updating profile:', error);
      throw error;
    }

    return this.convertDatabaseProfileToSavedProfile(data);
  }

  static async deleteProfile(profileId: string): Promise<void> {
    const { error } = await supabase
      .from('saved_configurations')
      .delete()
      .eq('id', profileId);

    if (error) {
      console.error('Error deleting profile:', error);
      throw error;
    }
  }

  static async deleteMultipleProfiles(profileIds: string[]): Promise<void> {
    const { error } = await supabase
      .from('saved_configurations')
      .delete()
      .in('id', profileIds);

    if (error) {
      console.error('Error deleting profiles:', error);
      throw error;
    }
  }

  private static convertDatabaseProfileToSavedProfile(dbProfile: Record<string, unknown>): SavedProfile {
    return {
      id: dbProfile.id as string,
      name: dbProfile.name as string,
      description: dbProfile.description as string,
      groundTruthParams: dbProfile.circuit_parameters as CircuitParameters,
      gridSize: dbProfile.grid_size as number,
      minFreq: dbProfile.min_frequency as number,
      maxFreq: dbProfile.max_frequency as number,  
      numPoints: dbProfile.num_points as number,
      isComputed: dbProfile.is_computed as boolean,
      computationTime: dbProfile.computation_time as number,
      totalPoints: dbProfile.total_points as number,
      validPoints: dbProfile.valid_points as number,
      created: new Date(dbProfile.created_at as string).getTime(),
      lastModified: new Date(dbProfile.updated_at as string).getTime()
    };
  }
}