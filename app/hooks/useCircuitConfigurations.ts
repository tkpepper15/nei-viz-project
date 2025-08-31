import { useState, useEffect, useCallback } from 'react';
import { CircuitConfigService, CircuitConfiguration, CreateCircuitConfigRequest } from '../../lib/circuitConfigService';

// Removed unused interface - props are passed directly to hook

interface CircuitConfigurationsState {
  configurations: CircuitConfiguration[];
  activeConfigId: string | null;
  loading: boolean;
  error: string | null;
}

export const useCircuitConfigurations = (userId?: string) => {
  const [state, setState] = useState<CircuitConfigurationsState>({
    configurations: [],
    activeConfigId: null,
    loading: false,
    error: null
  });

  // Load all circuit configurations for the user
  const loadConfigurations = useCallback(async () => {
    if (!userId) {
      setState(prev => ({
        ...prev,
        configurations: [],
        activeConfigId: null,
        loading: false,
        error: null
      }));
      return;
    }

    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      console.log('🔄 Loading circuit configurations for user:', userId);
      const configs = await CircuitConfigService.getUserCircuitConfigurations(userId);
      
      setState(prev => ({
        ...prev,
        configurations: configs,
        // Auto-select first configuration as active if none is selected
        activeConfigId: prev.activeConfigId || (configs.length > 0 ? configs[0].id : null),
        loading: false
      }));

      console.log(`✅ Loaded ${configs.length} circuit configurations`);
      
      // If there's no active config but we have configurations, set the first one as active
      if (configs.length > 0 && !state.activeConfigId) {
        console.log(`🔄 Auto-setting first configuration as active: ${configs[0].id}`);
      }

    } catch (error) {
      console.error('❌ Error loading circuit configurations:', error);
      setState(prev => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to load configurations'
      }));
    }
  }, [userId]);

  // Load configurations when userId changes
  useEffect(() => {
    loadConfigurations();
  }, [loadConfigurations]);

  // Create new circuit configuration
  const createConfiguration = useCallback(async (
    config: CreateCircuitConfigRequest
  ): Promise<CircuitConfiguration | null> => {
    if (!userId) {
      console.error('❌ No user ID available for creating configuration');
      return null;
    }

    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      console.log('🔄 Creating new circuit configuration:', config.name);
      const newConfig = await CircuitConfigService.createCircuitConfiguration(userId, config);
      
      setState(prev => ({
        ...prev,
        configurations: [newConfig, ...prev.configurations],
        activeConfigId: newConfig.id, // Set as active
        loading: false
      }));

      console.log('✅ Circuit configuration created:', newConfig.id);
      return newConfig;

    } catch (error) {
      console.error('❌ Error creating circuit configuration:', error);
      setState(prev => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to create configuration'
      }));
      return null;
    }
  }, [userId]);

  // Update existing circuit configuration
  const updateConfiguration = useCallback(async (
    configId: string,
    updates: Parameters<typeof CircuitConfigService.updateCircuitConfiguration>[1]
  ): Promise<boolean> => {
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      console.log('🔄 Updating circuit configuration:', configId);
      const updatedConfig = await CircuitConfigService.updateCircuitConfiguration(configId, updates);
      
      setState(prev => ({
        ...prev,
        configurations: prev.configurations.map(config =>
          config.id === configId ? updatedConfig : config
        ),
        loading: false
      }));

      console.log('✅ Circuit configuration updated');
      return true;

    } catch (error) {
      console.error('❌ Error updating circuit configuration:', error);
      setState(prev => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to update configuration'
      }));
      return false;
    }
  }, []);

  // Delete circuit configuration
  const deleteConfiguration = useCallback(async (configId: string): Promise<boolean> => {
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      console.log('🔄 Deleting circuit configuration:', configId);
      await CircuitConfigService.deleteCircuitConfiguration(configId);
      
      setState(prev => ({
        ...prev,
        configurations: prev.configurations.filter(config => config.id !== configId),
        activeConfigId: prev.activeConfigId === configId ? null : prev.activeConfigId,
        loading: false
      }));

      console.log('✅ Circuit configuration deleted (tagged models cascade deleted)');
      return true;

    } catch (error) {
      console.error('❌ Error deleting circuit configuration:', error);
      setState(prev => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to delete configuration'
      }));
      return false;
    }
  }, []);

  // Delete multiple circuit configurations
  const deleteMultipleConfigurations = useCallback(async (configIds: string[]): Promise<boolean> => {
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      console.log('🔄 Deleting multiple circuit configurations:', configIds.length);
      await CircuitConfigService.deleteMultipleCircuitConfigurations(configIds);
      
      setState(prev => ({
        ...prev,
        configurations: prev.configurations.filter(config => !configIds.includes(config.id)),
        activeConfigId: configIds.includes(prev.activeConfigId || '') ? null : prev.activeConfigId,
        loading: false
      }));

      console.log('✅ Circuit configurations deleted');
      return true;

    } catch (error) {
      console.error('❌ Error deleting circuit configurations:', error);
      setState(prev => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to delete configurations'
      }));
      return false;
    }
  }, []);

  // Set active configuration (local state only - session management handles persistence)
  const setActiveConfiguration = useCallback((configId: string | null) => {
    console.log('🔄 Setting active configuration locally:', configId);
    setState(prev => ({
      ...prev,
      activeConfigId: configId
    }));
  }, []);

  // Get active configuration object
  const activeConfiguration = state.configurations.find(
    config => config.id === state.activeConfigId
  ) || null;

  // Refresh configurations (useful after external changes)
  const refreshConfigurations = useCallback(() => {
    loadConfigurations();
  }, [loadConfigurations]);

  // Get configuration by ID
  const getConfigurationById = useCallback((configId: string): CircuitConfiguration | null => {
    return state.configurations.find(config => config.id === configId) || null;
  }, [state.configurations]);

  return {
    // State
    configurations: state.configurations,
    activeConfigId: state.activeConfigId,
    activeConfiguration,
    loading: state.loading,
    error: state.error,

    // Actions
    createConfiguration,
    updateConfiguration,
    deleteConfiguration,
    deleteMultipleConfigurations,
    setActiveConfiguration,
    refreshConfigurations,
    getConfigurationById,

    // Utilities
    hasConfigurations: state.configurations.length > 0,
    configurationCount: state.configurations.length
  };
};