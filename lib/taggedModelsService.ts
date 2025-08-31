/* eslint-disable @typescript-eslint/no-explicit-any */
// ^ Disabled because database schema may differ from generated types

import { supabase } from './supabase';
import { CircuitParameters } from '../app/components/circuit-simulator/types/parameters';
import { Json } from './database.types';

// Tagged Model interfaces
export interface TaggedModel {
  id: string;
  userId: string;
  sessionId?: string;
  circuitConfigId: string; // Links to circuit configuration
  modelId: string;
  tagName: string;
  tagCategory: string;
  circuitParameters: CircuitParameters;
  resnormValue?: number;
  taggedAt: string;
  notes?: string;
  isInteresting: boolean;
}

export interface CreateTaggedModelRequest {
  circuitConfigId: string; // REQUIRED
  sessionId?: string;
  modelId: string;
  tagName: string;
  tagCategory?: string;
  circuitParameters: CircuitParameters;
  resnormValue?: number;
  notes?: string;
  isInteresting?: boolean;
}

// Database row interface (matches actual schema)
interface TaggedModelRow {
  id: string;
  user_id: string;
  session_id?: string | null;
  circuit_config_id: string;
  model_id: string;
  tag_name: string;
  tag_category: string;
  circuit_parameters: Json;
  resnorm_value?: number | null;
  tagged_at: string;
  notes?: string | null;
  is_interesting: boolean;
}

export class TaggedModelsService {
  /**
   * Get tagged models for specific circuit configuration
   * This is the main method - tagged models are circuit-specific
   */
  static async getTaggedModelsForCircuit(
    circuitConfigId: string, 
    userId: string
  ): Promise<TaggedModel[]> {
    console.log('🔄 Fetching tagged models for circuit configuration:', circuitConfigId);

    try {
      const { data, error } = await (supabase as any)
        .from('tagged_models')
        .select('*')
        .eq('circuit_config_id', circuitConfigId)
        .eq('user_id', userId)
        .order('tagged_at', { ascending: false });

      if (error) {
        console.error('❌ Error fetching tagged models for circuit:', error);
        throw error;
      }

      if (!data || data.length === 0) {
        console.log('📋 No tagged models found for this circuit configuration');
        return [];
      }

      console.log(`✅ Found ${data.length} tagged models for circuit configuration`);
      return data.map(this.convertDatabaseRowToTaggedModel);

    } catch (error) {
      console.error('❌ Exception in getTaggedModelsForCircuit:', error);
      throw error;
    }
  }

  /**
   * Create tagged model linked to circuit configuration
   */
  static async createTaggedModel(
    userId: string,
    modelData: CreateTaggedModelRequest
  ): Promise<TaggedModel> {
    console.log('🔄 Creating tagged model:', {
      circuitConfigId: modelData.circuitConfigId,
      modelId: modelData.modelId,
      tagName: modelData.tagName
    });

    try {
      const insertData = {
        user_id: userId,
        session_id: modelData.sessionId || null,
        circuit_config_id: modelData.circuitConfigId,
        model_id: modelData.modelId,
        tag_name: modelData.tagName,
        tag_category: modelData.tagCategory || 'user',
        circuit_parameters: modelData.circuitParameters as unknown as Json,
        resnorm_value: modelData.resnormValue || null,
        notes: modelData.notes || null,
        is_interesting: modelData.isInteresting || false
      };

      const { data, error } = await (supabase as any)
        .from('tagged_models')
        .insert(insertData)
        .select()
        .single();

      if (error) {
        console.error('❌ Error creating tagged model:', error);
        throw error;
      }

      console.log('✅ Tagged model created successfully:', data.id);
      return this.convertDatabaseRowToTaggedModel(data);

    } catch (error) {
      console.error('❌ Exception in createTaggedModel:', error);
      throw error;
    }
  }

  /**
   * Update tagged model (notes, category, etc.)
   */
  static async updateTaggedModel(
    taggedModelId: string, 
    updates: Partial<{
      tagName: string;
      tagCategory: string;
      notes: string;
      isInteresting: boolean;
    }>
  ): Promise<TaggedModel> {
    console.log('🔄 Updating tagged model:', taggedModelId);

    try {
      // Map updates to correct column names
      const mappedUpdates: Record<string, unknown> = {};
      if (updates.tagName) mappedUpdates.tag_name = updates.tagName;
      if (updates.tagCategory) mappedUpdates.tag_category = updates.tagCategory;
      if (updates.notes !== undefined) mappedUpdates.notes = updates.notes;
      if (updates.isInteresting !== undefined) mappedUpdates.is_interesting = updates.isInteresting;

      const { data, error } = await (supabase as any)
        .from('tagged_models')
        .update(mappedUpdates)
        .eq('id', taggedModelId)
        .select()
        .single();

      if (error) {
        console.error('❌ Error updating tagged model:', error);
        throw error;
      }

      console.log('✅ Tagged model updated successfully');
      return this.convertDatabaseRowToTaggedModel(data);

    } catch (error) {
      console.error('❌ Exception in updateTaggedModel:', error);
      throw error;
    }
  }

  /**
   * Delete tagged model
   */
  static async deleteTaggedModel(taggedModelId: string): Promise<void> {
    console.log('🔄 Deleting tagged model:', taggedModelId);

    try {
      const { error } = await (supabase as any)
        .from('tagged_models')
        .delete()
        .eq('id', taggedModelId);

      if (error) {
        console.error('❌ Error deleting tagged model:', error);
        throw error;
      }

      console.log('✅ Tagged model deleted successfully');

    } catch (error) {
      console.error('❌ Exception in deleteTaggedModel:', error);
      throw error;
    }
  }

  /**
   * Delete multiple tagged models
   */
  static async deleteMultipleTaggedModels(taggedModelIds: string[]): Promise<void> {
    console.log('🔄 Deleting multiple tagged models:', taggedModelIds.length);

    try {
      const { error } = await (supabase as any)
        .from('tagged_models')
        .delete()
        .in('id', taggedModelIds);

      if (error) {
        console.error('❌ Error deleting tagged models:', error);
        throw error;
      }

      console.log('✅ Tagged models deleted successfully');

    } catch (error) {
      console.error('❌ Exception in deleteMultipleTaggedModels:', error);
      throw error;
    }
  }

  /**
   * Get tagged model by ID
   */
  static async getTaggedModel(taggedModelId: string): Promise<TaggedModel | null> {
    try {
      const { data, error } = await (supabase as any)
        .from('tagged_models')
        .select('*')
        .eq('id', taggedModelId)
        .single();

      if (error) {
        if (error.code === 'PGRST116') {
          return null; // Not found
        }
        throw error;
      }

      return this.convertDatabaseRowToTaggedModel(data);

    } catch (error) {
      console.error('❌ Exception in getTaggedModel:', error);
      throw error;
    }
  }

  /**
   * Get all tagged models for a user (across all circuit configurations)
   * Use this for analytics or overview purposes
   */
  static async getAllUserTaggedModels(userId: string): Promise<TaggedModel[]> {
    console.log('🔄 Fetching all tagged models for user:', userId);

    try {
      const { data, error } = await (supabase as any)
        .from('tagged_models')
        .select('*')
        .eq('user_id', userId)
        .order('tagged_at', { ascending: false });

      if (error) {
        console.error('❌ Error fetching user tagged models:', error);
        throw error;
      }

      console.log(`✅ Found ${data?.length || 0} total tagged models for user`);
      return data ? data.map(this.convertDatabaseRowToTaggedModel) : [];

    } catch (error) {
      console.error('❌ Exception in getAllUserTaggedModels:', error);
      throw error;
    }
  }

  /**
   * Get interesting tagged models for circuit configuration
   */
  static async getInterestingTaggedModels(
    circuitConfigId: string,
    userId: string
  ): Promise<TaggedModel[]> {
    console.log('🔄 Fetching interesting tagged models for circuit:', circuitConfigId);

    try {
      const { data, error } = await (supabase as any)
        .from('tagged_models')
        .select('*')
        .eq('circuit_config_id', circuitConfigId)
        .eq('user_id', userId)
        .eq('is_interesting', true)
        .order('resnorm_value', { ascending: true }); // Best resnorm first

      if (error) {
        console.error('❌ Error fetching interesting tagged models:', error);
        throw error;
      }

      console.log(`✅ Found ${data?.length || 0} interesting tagged models`);
      return data ? data.map(this.convertDatabaseRowToTaggedModel) : [];

    } catch (error) {
      console.error('❌ Exception in getInterestingTaggedModels:', error);
      throw error;
    }
  }

  /**
   * Convert database row to TaggedModel interface
   */
  private static convertDatabaseRowToTaggedModel(row: TaggedModelRow): TaggedModel {
    return {
      id: row.id,
      userId: row.user_id,
      sessionId: row.session_id || undefined,
      circuitConfigId: row.circuit_config_id,
      modelId: row.model_id,
      tagName: row.tag_name,
      tagCategory: row.tag_category,
      circuitParameters: row.circuit_parameters as unknown as CircuitParameters,
      resnormValue: row.resnorm_value || undefined,
      taggedAt: row.tagged_at,
      notes: row.notes || undefined,
      isInteresting: row.is_interesting
    };
  }
}