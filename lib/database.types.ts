// TypeScript types generated from Supabase database schema
export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export interface Database {
  public: {
    Tables: {
      user_sessions: {
        Row: {
          id: string
          user_id: string
          session_name: string
          description: string | null
          environment_variables: Json
          visualization_settings: Json
          performance_settings: Json
          is_active: boolean
          last_accessed: string
          created_at: string
          updated_at: string
          total_computations: number
          total_models_generated: number
          total_computation_time: string
        }
        Insert: {
          id?: string
          user_id: string
          session_name: string
          description?: string | null
          environment_variables?: Json
          visualization_settings?: Json
          performance_settings?: Json
          is_active?: boolean
          last_accessed?: string
          created_at?: string
          updated_at?: string
          total_computations?: number
          total_models_generated?: number
          total_computation_time?: string
        }
        Update: {
          id?: string
          user_id?: string
          session_name?: string
          description?: string | null
          environment_variables?: Json
          visualization_settings?: Json
          performance_settings?: Json
          is_active?: boolean
          last_accessed?: string
          created_at?: string
          updated_at?: string
          total_computations?: number
          total_models_generated?: number
          total_computation_time?: string
        }
        Relationships: [
          {
            foreignKeyName: "user_sessions_user_id_fkey"
            columns: ["user_id"]
            isOneToOne: false
            referencedRelation: "users"
            referencedColumns: ["id"]
          }
        ]
      }
      tagged_models: {
        Row: {
          id: string
          user_id: string
          session_id: string
          configuration_id: string
          model_id: string
          tag_name: string
          tag_category: string
          circuit_parameters: Json
          resnorm_value: number | null
          impedance_spectrum: Json | null
          tagged_at: string
          tagging_context: Json | null
          notes: string | null
          ml_relevance_score: number | null
          is_ml_training_data: boolean
        }
        Insert: {
          id?: string
          user_id: string
          session_id: string
          configuration_id: string
          model_id: string
          tag_name: string
          tag_category?: string
          circuit_parameters: Json
          resnorm_value?: number | null
          impedance_spectrum?: Json | null
          tagged_at?: string
          tagging_context?: Json | null
          notes?: string | null
          ml_relevance_score?: number | null
          is_ml_training_data?: boolean
        }
        Update: {
          id?: string
          user_id?: string
          session_id?: string
          configuration_id?: string
          model_id?: string
          tag_name?: string
          tag_category?: string
          circuit_parameters?: Json
          resnorm_value?: number | null
          impedance_spectrum?: Json | null
          tagged_at?: string
          tagging_context?: Json | null
          notes?: string | null
          ml_relevance_score?: number | null
          is_ml_training_data?: boolean
        }
        Relationships: [
          {
            foreignKeyName: "tagged_models_user_id_fkey"
            columns: ["user_id"]
            isOneToOne: false
            referencedRelation: "users"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "tagged_models_session_id_fkey"
            columns: ["session_id"]
            isOneToOne: false
            referencedRelation: "user_sessions"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "tagged_models_configuration_id_fkey"
            columns: ["configuration_id"]
            isOneToOne: false
            referencedRelation: "saved_configurations"
            referencedColumns: ["id"]
          }
        ]
      }
      parameter_exploration_sessions: {
        Row: {
          id: string
          user_id: string
          session_id: string
          exploration_name: string
          exploration_type: string
          parameter_ranges: Json
          sampling_strategy: Json
          total_parameter_combinations: number | null
          successful_computations: number | null
          failed_computations: number | null
          started_at: string
          completed_at: string | null
          total_duration: string | null
          average_computation_time: string | null
          ml_objectives: Json | null
          ml_constraints: Json | null
          ml_recommendations: Json | null
          status: string
          progress_percentage: number
        }
        Insert: {
          id?: string
          user_id: string
          session_id: string
          exploration_name: string
          exploration_type?: string
          parameter_ranges: Json
          sampling_strategy: Json
          total_parameter_combinations?: number | null
          successful_computations?: number | null
          failed_computations?: number | null
          started_at?: string
          completed_at?: string | null
          total_duration?: string | null
          average_computation_time?: string | null
          ml_objectives?: Json | null
          ml_constraints?: Json | null
          ml_recommendations?: Json | null
          status?: string
          progress_percentage?: number
        }
        Update: {
          id?: string
          user_id?: string
          session_id?: string
          exploration_name?: string
          exploration_type?: string
          parameter_ranges?: Json
          sampling_strategy?: Json
          total_parameter_combinations?: number | null
          successful_computations?: number | null
          failed_computations?: number | null
          started_at?: string
          completed_at?: string | null
          total_duration?: string | null
          average_computation_time?: string | null
          ml_objectives?: Json | null
          ml_constraints?: Json | null
          ml_recommendations?: Json | null
          status?: string
          progress_percentage?: number
        }
        Relationships: [
          {
            foreignKeyName: "parameter_exploration_sessions_user_id_fkey"
            columns: ["user_id"]
            isOneToOne: false
            referencedRelation: "users"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "parameter_exploration_sessions_session_id_fkey"
            columns: ["session_id"]
            isOneToOne: false
            referencedRelation: "user_sessions"
            referencedColumns: ["id"]
          }
        ]
      }
      ml_training_datasets: {
        Row: {
          id: string
          user_id: string
          dataset_name: string
          dataset_version: number
          description: string | null
          source_configurations: string[]
          source_sessions: string[]
          total_samples: number
          feature_count: number
          dataset_type: string
          target_variable: string | null
          feature_engineering_config: Json | null
          feature_matrix_path: string | null
          target_vector_path: string | null
          metadata_path: string | null
          data_quality_score: number | null
          completeness_percentage: number | null
          outlier_percentage: number | null
          created_at: string
          updated_at: string
          is_active: boolean
        }
        Insert: {
          id?: string
          user_id: string
          dataset_name: string
          dataset_version?: number
          description?: string | null
          source_configurations: string[]
          source_sessions: string[]
          total_samples: number
          feature_count: number
          dataset_type: string
          target_variable?: string | null
          feature_engineering_config?: Json | null
          feature_matrix_path?: string | null
          target_vector_path?: string | null
          metadata_path?: string | null
          data_quality_score?: number | null
          completeness_percentage?: number | null
          outlier_percentage?: number | null
          created_at?: string
          updated_at?: string
          is_active?: boolean
        }
        Update: {
          id?: string
          user_id?: string
          dataset_name?: string
          dataset_version?: number
          description?: string | null
          source_configurations?: string[]
          source_sessions?: string[]
          total_samples?: number
          feature_count?: number
          dataset_type?: string
          target_variable?: string | null
          feature_engineering_config?: Json | null
          feature_matrix_path?: string | null
          target_vector_path?: string | null
          metadata_path?: string | null
          data_quality_score?: number | null
          completeness_percentage?: number | null
          outlier_percentage?: number | null
          created_at?: string
          updated_at?: string
          is_active?: boolean
        }
        Relationships: [
          {
            foreignKeyName: "ml_training_datasets_user_id_fkey"
            columns: ["user_id"]
            isOneToOne: false
            referencedRelation: "users"
            referencedColumns: ["id"]
          }
        ]
      }
      ml_models: {
        Row: {
          id: string
          user_id: string
          training_dataset_id: string | null
          model_name: string
          model_version: number
          model_type: string
          description: string | null
          training_config: Json
          hyperparameters: Json
          training_metrics: Json | null
          validation_metrics: Json | null
          test_metrics: Json | null
          model_path: string
          model_size_bytes: number | null
          is_deployed: boolean
          deployment_endpoint: string | null
          trained_at: string
          training_duration: string | null
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          user_id: string
          training_dataset_id?: string | null
          model_name: string
          model_version?: number
          model_type: string
          description?: string | null
          training_config: Json
          hyperparameters: Json
          training_metrics?: Json | null
          validation_metrics?: Json | null
          test_metrics?: Json | null
          model_path: string
          model_size_bytes?: number | null
          is_deployed?: boolean
          deployment_endpoint?: string | null
          trained_at?: string
          training_duration?: string | null
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          training_dataset_id?: string | null
          model_name?: string
          model_version?: number
          model_type?: string
          description?: string | null
          training_config?: Json
          hyperparameters?: Json
          training_metrics?: Json | null
          validation_metrics?: Json | null
          test_metrics?: Json | null
          model_path?: string
          model_size_bytes?: number | null
          is_deployed?: boolean
          deployment_endpoint?: string | null
          trained_at?: string
          training_duration?: string | null
          created_at?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "ml_models_user_id_fkey"
            columns: ["user_id"]
            isOneToOne: false
            referencedRelation: "users"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "ml_models_training_dataset_id_fkey"
            columns: ["training_dataset_id"]
            isOneToOne: false
            referencedRelation: "ml_training_datasets"
            referencedColumns: ["id"]
          }
        ]
      }
      visualization_snapshots: {
        Row: {
          id: string
          user_id: string
          session_id: string
          configuration_id: string
          snapshot_name: string
          description: string | null
          visualization_type: string
          camera_position: Json | null
          filter_settings: Json | null
          display_settings: Json | null
          panel_states: Json | null
          slider_values: Json | null
          selected_models: string[]
          user_annotations: Json | null
          is_public: boolean
          shared_with: string[]
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          user_id: string
          session_id: string
          configuration_id: string
          snapshot_name: string
          description?: string | null
          visualization_type: string
          camera_position?: Json | null
          filter_settings?: Json | null
          display_settings?: Json | null
          panel_states?: Json | null
          slider_values?: Json | null
          selected_models?: string[]
          user_annotations?: Json | null
          is_public?: boolean
          shared_with?: string[]
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          session_id?: string
          configuration_id?: string
          snapshot_name?: string
          description?: string | null
          visualization_type?: string
          camera_position?: Json | null
          filter_settings?: Json | null
          display_settings?: Json | null
          panel_states?: Json | null
          slider_values?: Json | null
          selected_models?: string[]
          user_annotations?: Json | null
          is_public?: boolean
          shared_with?: string[]
          created_at?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "visualization_snapshots_user_id_fkey"
            columns: ["user_id"]
            isOneToOne: false
            referencedRelation: "users"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "visualization_snapshots_session_id_fkey"
            columns: ["session_id"]
            isOneToOne: false
            referencedRelation: "user_sessions"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "visualization_snapshots_configuration_id_fkey"
            columns: ["configuration_id"]
            isOneToOne: false
            referencedRelation: "saved_configurations"
            referencedColumns: ["id"]
          }
        ]
      }
      parameter_optimization_jobs: {
        Row: {
          id: string
          user_id: string
          session_id: string
          ml_model_id: string | null
          job_name: string
          optimization_algorithm: string
          objective_function: string
          parameter_bounds: Json
          constraints: Json | null
          max_iterations: number
          convergence_tolerance: number
          population_size: number
          current_iteration: number
          best_parameters: Json | null
          best_objective_value: number | null
          convergence_history: Json | null
          started_at: string
          completed_at: string | null
          estimated_completion: string | null
          cpu_hours_used: number
          status: string
          error_message: string | null
          progress_percentage: number
        }
        Insert: {
          id?: string
          user_id: string
          session_id: string
          ml_model_id?: string | null
          job_name: string
          optimization_algorithm: string
          objective_function: string
          parameter_bounds: Json
          constraints?: Json | null
          max_iterations?: number
          convergence_tolerance?: number
          population_size?: number
          current_iteration?: number
          best_parameters?: Json | null
          best_objective_value?: number | null
          convergence_history?: Json | null
          started_at?: string
          completed_at?: string | null
          estimated_completion?: string | null
          cpu_hours_used?: number
          status?: string
          error_message?: string | null
          progress_percentage?: number
        }
        Update: {
          id?: string
          user_id?: string
          session_id?: string
          ml_model_id?: string | null
          job_name?: string
          optimization_algorithm?: string
          objective_function?: string
          parameter_bounds?: Json
          constraints?: Json | null
          max_iterations?: number
          convergence_tolerance?: number
          population_size?: number
          current_iteration?: number
          best_parameters?: Json | null
          best_objective_value?: number | null
          convergence_history?: Json | null
          started_at?: string
          completed_at?: string | null
          estimated_completion?: string | null
          cpu_hours_used?: number
          status?: string
          error_message?: string | null
          progress_percentage?: number
        }
        Relationships: [
          {
            foreignKeyName: "parameter_optimization_jobs_user_id_fkey"
            columns: ["user_id"]
            isOneToOne: false
            referencedRelation: "users"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "parameter_optimization_jobs_session_id_fkey"
            columns: ["session_id"]
            isOneToOne: false
            referencedRelation: "user_sessions"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "parameter_optimization_jobs_ml_model_id_fkey"
            columns: ["ml_model_id"]
            isOneToOne: false
            referencedRelation: "ml_models"
            referencedColumns: ["id"]
          }
        ]
      }
      user_profiles: {
        Row: {
          id: string
          username: string | null
          full_name: string | null
          avatar_url: string | null
          created_at: string
          updated_at: string
        }
        Insert: {
          id: string
          username?: string | null
          full_name?: string | null
          avatar_url?: string | null
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          username?: string | null
          full_name?: string | null
          avatar_url?: string | null
          created_at?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "user_profiles_id_fkey"
            columns: ["id"]
            isOneToOne: true
            referencedRelation: "users"
            referencedColumns: ["id"]
          }
        ]
      }
      saved_configurations: {
        Row: {
          id: string
          user_id: string
          name: string
          description: string | null
          is_public: boolean
          created_at: string
          updated_at: string
          grid_size: number
          min_frequency: number
          max_frequency: number
          num_points: number
          circuit_parameters: Json
          is_computed: boolean
          computation_time: number | null
          total_points: number | null
          valid_points: number | null
        }
        Insert: {
          id?: string
          user_id: string
          name: string
          description?: string | null
          is_public?: boolean
          created_at?: string
          updated_at?: string
          grid_size: number
          min_frequency: number
          max_frequency: number
          num_points: number
          circuit_parameters: Json
          is_computed?: boolean
          computation_time?: number | null
          total_points?: number | null
          valid_points?: number | null
        }
        Update: {
          id?: string
          user_id?: string
          name?: string
          description?: string | null
          is_public?: boolean
          created_at?: string
          updated_at?: string
          grid_size?: number
          min_frequency?: number
          max_frequency?: number
          num_points?: number
          circuit_parameters?: Json
          is_computed?: boolean
          computation_time?: number | null
          total_points?: number | null
          valid_points?: number | null
        }
        Relationships: [
          {
            foreignKeyName: "saved_configurations_user_id_fkey"
            columns: ["user_id"]
            isOneToOne: false
            referencedRelation: "users"
            referencedColumns: ["id"]
          }
        ]
      }
      computation_results: {
        Row: {
          id: string
          configuration_id: string
          grid_results: Json
          resnorm_groups: Json | null
          grid_parameter_arrays: Json | null
          performance_metrics: Json | null
          computed_at: string
          computation_duration: number
          result_size_bytes: number | null
          grid_size: number
          total_points: number
          valid_points: number
        }
        Insert: {
          id?: string
          configuration_id: string
          grid_results: Json
          resnorm_groups?: Json | null
          grid_parameter_arrays?: Json | null
          performance_metrics?: Json | null
          computed_at?: string
          computation_duration: number
          result_size_bytes?: number | null
          grid_size: number
          total_points: number
          valid_points: number
        }
        Update: {
          id?: string
          configuration_id?: string
          grid_results?: Json
          resnorm_groups?: Json | null
          grid_parameter_arrays?: Json | null
          performance_metrics?: Json | null
          computed_at?: string
          computation_duration?: number
          result_size_bytes?: number | null
          grid_size?: number
          total_points?: number
          valid_points?: number
        }
        Relationships: [
          {
            foreignKeyName: "computation_results_configuration_id_fkey"
            columns: ["configuration_id"]
            isOneToOne: false
            referencedRelation: "saved_configurations"
            referencedColumns: ["id"]
          }
        ]
      }
      shared_configurations: {
        Row: {
          id: string
          configuration_id: string
          shared_with: string
          shared_by: string
          permission_level: string
          shared_at: string
          expires_at: string | null
          last_accessed: string | null
          share_message: string | null
          is_active: boolean
        }
        Insert: {
          id?: string
          configuration_id: string
          shared_with: string
          shared_by: string
          permission_level?: string
          shared_at?: string
          expires_at?: string | null
          last_accessed?: string | null
          share_message?: string | null
          is_active?: boolean
        }
        Update: {
          id?: string
          configuration_id?: string
          shared_with?: string
          shared_by?: string
          permission_level?: string
          shared_at?: string
          expires_at?: string | null
          last_accessed?: string | null
          share_message?: string | null
          is_active?: boolean
        }
        Relationships: [
          {
            foreignKeyName: "shared_configurations_configuration_id_fkey"
            columns: ["configuration_id"]
            isOneToOne: false
            referencedRelation: "saved_configurations"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "shared_configurations_shared_by_fkey"
            columns: ["shared_by"]
            isOneToOne: false
            referencedRelation: "users"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "shared_configurations_shared_with_fkey"
            columns: ["shared_with"]
            isOneToOne: false
            referencedRelation: "users"
            referencedColumns: ["id"]
          }
        ]
      }
    }
    Views: {
      shared_configurations_with_details: {
        Row: {
          id: string | null
          configuration_id: string | null
          shared_with: string | null
          shared_by: string | null
          permission_level: string | null
          shared_at: string | null
          expires_at: string | null
          last_accessed: string | null
          share_message: string | null
          is_active: boolean | null
          configuration_name: string | null
          configuration_description: string | null
          is_computed: boolean | null
          configuration_created_at: string | null
          owner_name: string | null
          owner_username: string | null
          shared_by_name: string | null
          shared_by_username: string | null
        }
        Relationships: [
          {
            foreignKeyName: "shared_configurations_configuration_id_fkey"
            columns: ["configuration_id"]
            isOneToOne: false
            referencedRelation: "saved_configurations"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "shared_configurations_shared_by_fkey"
            columns: ["shared_by"]
            isOneToOne: false
            referencedRelation: "users"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "shared_configurations_shared_with_fkey"
            columns: ["shared_with"]
            isOneToOne: false
            referencedRelation: "users"
            referencedColumns: ["id"]
          }
        ]
      }
    }
    Functions: {
      get_user_configurations: {
        Args: {
          user_uuid?: string
        }
        Returns: {
          id: string
          name: string
          description: string
          created_at: string
          updated_at: string
          is_computed: boolean
          is_owner: boolean
          permission_level: string
          owner_name: string
        }[]
      }
    }
    Enums: {
      [_ in never]: never
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

// Additional type definitions for the application
export interface CircuitParameters {
  Rsh: number      // Shunt resistance (Ω)
  Ra: number       // Apical resistance (Ω)  
  Ca: number       // Apical capacitance (F)
  Rb: number       // Basal resistance (Ω)
  Cb: number       // Basal capacitance (F)
  frequency_range: [number, number] // [min, max] Hz
}

export interface SavedConfigurationWithProfile extends Database['public']['Tables']['saved_configurations']['Row'] {
  user_profiles: {
    full_name: string | null
    username: string | null
  } | null
}

export interface ComputationResultsData {
  gridResults: Record<string, unknown>[]
  resnormGroups: Record<string, unknown>[]
  gridParameterArrays: Record<string, unknown>[]
  performanceMetrics: {
    totalComputeTime: number
    averagePointTime: number
    memoryUsage: string
    workerCount: number
  }
}

// Auth types
export interface AuthUser {
  id: string
  email: string
  user_metadata?: {
    full_name?: string
    avatar_url?: string
  }
}

export interface UserProfile {
  id: string
  username: string | null
  full_name: string | null
  avatar_url: string | null
  created_at: string
  updated_at: string
}