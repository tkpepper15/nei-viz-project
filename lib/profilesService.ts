/* eslint-disable @typescript-eslint/no-explicit-any */
// ^ Disabled because database schema differs from generated types

import { supabase } from './supabase';
import { SavedProfile } from '../app/components/circuit-simulator/types/savedProfiles';
import { CircuitParameters } from '../app/components/circuit-simulator/types/parameters';
import { Json } from './database.types';

// This interface matches the NEW user_profiles table schema (user metadata only)
export interface UserProfileRow {
  id: string;
  user_id: string;
  username: string | null;
  full_name: string | null;
  avatar_url: string | null;
  default_grid_size: number | null;
  default_min_freq: number | null;
  default_max_freq: number | null;
  default_num_points: number | null;
  created_at: string | null;
  updated_at: string | null;
}

export class ProfilesService {
  static async getUserDefaultGridSize(userId: string): Promise<number> {
    try {
      const { data, error } = await supabase
        .from('user_profiles')
        .select('default_grid_size')
        .eq('user_id', userId)
        .single();

      if (error || !data) {
        console.log('📏 No user profile found or error, using default grid size: 9');
        return 9; // Default fallback
      }

      const gridSize = (data as any).default_grid_size || 9;
      console.log(`📏 Retrieved user default grid size: ${gridSize}`);
      return gridSize;
    } catch (error) {
      console.error('❌ Error fetching user default grid size:', error);
      return 9; // Fallback default
    }
  }

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
      const { data: tableCheck, error: tableError } = await (supabase as any)
        .from('user_profiles')
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
          throw new Error('user_profiles table does not exist - run database migration');
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
    const { data, error } = await (supabase as any)
      .from('user_profiles')
      .select('*')
      .eq('user_id', userId)
      .order('updated_at', { ascending: false });

    console.log('📊 Query result:', { 
      hasData: !!data, 
      dataLength: data?.length || 0,
      hasError: !!error 
    });

    if (error) {
      console.error('❌ Error fetching user profiles from user_profiles table:');
      console.error('Full error object:', JSON.stringify(error, null, 2));
      console.error('Error details:', {
        message: error.message,
        code: error.code,
        hint: error.hint,
        details: error.details
      });
      console.error('Query details:', {
        table: 'user_profiles',
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
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    userId: string, 
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    name: string, 
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    parameters: CircuitParameters,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    gridSize: number,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    minFreq: number,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    maxFreq: number,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    numPoints: number,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    description?: string
  ): Promise<SavedProfile> {
    // DEPRECATED: This method is no longer supported with the new schema
    // Circuit configurations should use CircuitConfigService instead
    console.warn('⚠️ ProfilesService.createProfile is deprecated - use CircuitConfigService instead');
    throw new Error('ProfilesService.createProfile is deprecated - circuit configurations should use CircuitConfigService');
  }

  static async updateProfile(
    profileId: string, 
    updates: any // Legacy compatibility - this method is deprecated
  ): Promise<SavedProfile> {
    // DEPRECATED: This method is no longer supported with the new schema
    // Circuit configurations should use CircuitConfigService instead
    console.warn('⚠️ ProfilesService.updateProfile is deprecated - use CircuitConfigService instead');
    throw new Error('ProfilesService.updateProfile is deprecated - circuit configurations should use CircuitConfigService');
    
    // Map updates to correct column names  
    const mappedUpdates: Record<string, unknown> = {};
    if (updates.name) mappedUpdates.name = updates.name;
    if (updates.description !== undefined) mappedUpdates.description = updates.description;
    if (updates.parameters) mappedUpdates.parameters = updates.parameters as unknown as Json;
    mappedUpdates.updated_at = new Date().toISOString();

    const { data, error } = await (supabase as any)
      .from('user_profiles')
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
    const { error } = await (supabase as any)
      .from('user_profiles')
      .delete()
      .eq('id', profileId);

    if (error) {
      console.error('Error deleting profile:', error);
      throw error;
    }
  }

  static async deleteMultipleProfiles(profileIds: string[]): Promise<void> {
    const { error } = await (supabase as any)
      .from('user_profiles')
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
      description: (dbProfile.description as string) || '',
      groundTruthParams: dbProfile.parameters as CircuitParameters,
      gridSize: dbProfile.grid_size as number || 10,
      minFreq: Number(dbProfile.min_freq) || 0.1,
      maxFreq: Number(dbProfile.max_freq) || 100000,  
      numPoints: dbProfile.num_points as number || 100,
      isComputed: dbProfile.is_computed as boolean || false,
      computationTime: dbProfile.computation_time ? Number(dbProfile.computation_time) : undefined,
      totalPoints: dbProfile.total_points as number | undefined,
      validPoints: dbProfile.valid_points as number | undefined,
      created: new Date(dbProfile.created_at as string).getTime(),
      lastModified: new Date(dbProfile.updated_at as string).getTime()
    };
  }
}