/* eslint-disable @typescript-eslint/no-explicit-any */
// ^ Disabled because database schema may differ from generated types

import { supabase } from './supabase';
import { CircuitParameters } from '../app/components/circuit-simulator/types/parameters';
import { Json } from './database.types';

// Circuit Configuration interfaces
export interface CircuitConfiguration {
  id: string;
  userId: string;
  name: string;
  description?: string;
  isPublic: boolean;
  circuitParameters: CircuitParameters;
  gridSize: number;
  minFreq: number;
  maxFreq: number;
  numPoints: number;
  isComputed: boolean;
  computationTime?: number;
  totalPoints?: number;
  validPoints?: number;
  computationResults?: unknown;
  createdAt: string;
  updatedAt: string;
}

export interface CreateCircuitConfigRequest {
  name: string;
  description?: string;
  circuitParameters: CircuitParameters;
  gridSize: number;
  minFreq: number;
  maxFreq: number;
  numPoints: number;
}

// Database row interface (matches actual schema)
interface CircuitConfigRow {
  id: string;
  user_id: string;
  name: string;
  description?: string | null;
  is_public: boolean;
  circuit_parameters: Json;
  grid_size: number;
  min_freq: number;
  max_freq: number;
  num_points: number;
  is_computed: boolean;
  computation_time?: number | null;
  total_points?: number | null;
  valid_points?: number | null;
  computation_results?: Json | null;
  created_at: string;
  updated_at: string;
}

export class CircuitConfigService {
  /**
   * Get all circuit configurations for a user (persist across sessions)
   */
  static async getUserCircuitConfigurations(userId: string): Promise<CircuitConfiguration[]> {
    console.log('🔄 Fetching circuit configurations for user:', userId);

    try {
      const { data, error } = await (supabase as any)
        .from('circuit_configurations')
        .select('*')
        .eq('user_id', userId)
        .order('updated_at', { ascending: false });

      if (error) {
        console.error('❌ Error fetching circuit configurations:', error);
        throw error;
      }

      if (!data || data.length === 0) {
        console.log('📋 No circuit configurations found for user');
        return [];
      }

      console.log(`✅ Found ${data.length} circuit configurations`);
      return data.map(this.convertDatabaseRowToCircuitConfig);

    } catch (error) {
      console.error('❌ Exception in getUserCircuitConfigurations:', error);
      throw error;
    }
  }

  /**
   * Create new circuit configuration
   */
  static async createCircuitConfiguration(
    userId: string,
    config: CreateCircuitConfigRequest
  ): Promise<CircuitConfiguration> {
    console.log('🔄 Creating circuit configuration:', config.name);

    try {
      // Add frequency_range to circuit_parameters to satisfy schema constraint
      const circuitParamsWithFrequency = {
        ...config.circuitParameters,
        frequency_range: [config.minFreq, config.maxFreq]
      };

      const { data, error } = await (supabase as any)
        .from('circuit_configurations')
        .insert({
          user_id: userId,
          name: config.name,
          description: config.description,
          circuit_parameters: circuitParamsWithFrequency as unknown as Json,
          grid_size: config.gridSize,
          min_freq: config.minFreq,
          max_freq: config.maxFreq,
          num_points: config.numPoints,
          is_computed: false,
          is_public: false
        })
        .select()
        .single();

      if (error) {
        console.error('❌ Error creating circuit configuration:', error);
        throw error;
      }

      console.log('✅ Circuit configuration created successfully:', data.id);
      return this.convertDatabaseRowToCircuitConfig(data);

    } catch (error) {
      console.error('❌ Exception in createCircuitConfiguration:', error);
      throw error;
    }
  }

  /**
   * Update existing circuit configuration
   */
  static async updateCircuitConfiguration(
    configId: string,
    updates: Partial<CreateCircuitConfigRequest & {
      isComputed?: boolean;
      computationTime?: number;
      totalPoints?: number;
      validPoints?: number;
      computationResults?: unknown;
    }>
  ): Promise<CircuitConfiguration> {
    console.log('🔄 Updating circuit configuration:', configId);

    try {
      // Map updates to correct column names  
      const mappedUpdates: Record<string, unknown> = {};
      if (updates.name) mappedUpdates.name = updates.name;
      if (updates.description !== undefined) mappedUpdates.description = updates.description;
      
      // Handle circuit parameters with frequency_range
      if (updates.circuitParameters || updates.minFreq || updates.maxFreq) {
        const currentMinFreq = updates.minFreq || 0.1;
        const currentMaxFreq = updates.maxFreq || 100000;
        const circuitParamsWithFrequency = {
          ...(updates.circuitParameters || {}),
          frequency_range: [currentMinFreq, currentMaxFreq]
        };
        mappedUpdates.circuit_parameters = circuitParamsWithFrequency as unknown as Json;
      }
      
      if (updates.gridSize) mappedUpdates.grid_size = updates.gridSize;
      if (updates.minFreq) mappedUpdates.min_freq = updates.minFreq;
      if (updates.maxFreq) mappedUpdates.max_freq = updates.maxFreq;
      if (updates.numPoints) mappedUpdates.num_points = updates.numPoints;
      if (updates.isComputed !== undefined) mappedUpdates.is_computed = updates.isComputed;
      if (updates.computationTime !== undefined) mappedUpdates.computation_time = updates.computationTime;
      if (updates.totalPoints !== undefined) mappedUpdates.total_points = updates.totalPoints;
      if (updates.validPoints !== undefined) mappedUpdates.valid_points = updates.validPoints;
      if (updates.computationResults !== undefined) mappedUpdates.computation_results = updates.computationResults as unknown as Json;
      
      mappedUpdates.updated_at = new Date().toISOString();

      const { data, error } = await (supabase as any)
        .from('circuit_configurations')
        .update(mappedUpdates)
        .eq('id', configId)
        .select()
        .single();

      if (error) {
        console.error('❌ Error updating circuit configuration:', error);
        throw error;
      }

      console.log('✅ Circuit configuration updated successfully');
      return this.convertDatabaseRowToCircuitConfig(data);

    } catch (error) {
      console.error('❌ Exception in updateCircuitConfiguration:', error);
      throw error;
    }
  }

  /**
   * Delete circuit configuration (cascades to tagged models)
   */
  static async deleteCircuitConfiguration(configId: string): Promise<void> {
    console.log('🔄 Deleting circuit configuration:', configId);

    try {
      const { error } = await (supabase as any)
        .from('circuit_configurations')
        .delete()
        .eq('id', configId);

      if (error) {
        console.error('❌ Error deleting circuit configuration:', error);
        throw error;
      }

      console.log('✅ Circuit configuration deleted successfully (tagged models cascade deleted)');

    } catch (error) {
      console.error('❌ Exception in deleteCircuitConfiguration:', error);
      throw error;
    }
  }

  /**
   * Delete multiple circuit configurations
   */
  static async deleteMultipleCircuitConfigurations(configIds: string[]): Promise<void> {
    console.log('🔄 Deleting multiple circuit configurations:', configIds.length);

    try {
      const { error } = await (supabase as any)
        .from('circuit_configurations')
        .delete()
        .in('id', configIds);

      if (error) {
        console.error('❌ Error deleting circuit configurations:', error);
        throw error;
      }

      console.log('✅ Circuit configurations deleted successfully');

    } catch (error) {
      console.error('❌ Exception in deleteMultipleCircuitConfigurations:', error);
      throw error;
    }
  }

  /**
   * Get specific circuit configuration by ID
   */
  static async getCircuitConfiguration(configId: string): Promise<CircuitConfiguration | null> {
    try {
      const { data, error } = await (supabase as any)
        .from('circuit_configurations')
        .select('*')
        .eq('id', configId)
        .single();

      if (error) {
        if (error.code === 'PGRST116') {
          return null; // Not found
        }
        throw error;
      }

      return this.convertDatabaseRowToCircuitConfig(data);

    } catch (error) {
      console.error('❌ Exception in getCircuitConfiguration:', error);
      throw error;
    }
  }

  /**
   * Convert database row to CircuitConfiguration interface
   */
  private static convertDatabaseRowToCircuitConfig(row: CircuitConfigRow): CircuitConfiguration {
    return {
      id: row.id,
      userId: row.user_id,
      name: row.name,
      description: row.description || undefined,
      isPublic: row.is_public,
      circuitParameters: row.circuit_parameters as unknown as CircuitParameters,
      gridSize: row.grid_size,
      minFreq: Number(row.min_freq),
      maxFreq: Number(row.max_freq),
      numPoints: row.num_points,
      isComputed: row.is_computed,
      computationTime: row.computation_time ? Number(row.computation_time) : undefined,
      totalPoints: row.total_points || undefined,
      validPoints: row.valid_points || undefined,
      computationResults: row.computation_results,
      createdAt: row.created_at,
      updatedAt: row.updated_at
    };
  }
}