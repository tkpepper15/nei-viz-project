import { useState } from 'react';

// NPZ functionality has been disabled - this is a stub implementation
/* eslint-disable @typescript-eslint/no-unused-vars */
export const useNPZData = () => {
  const [isLoading] = useState(false);
  const [error] = useState<string | null>(null);
  
  // Return empty/disabled state
  const emptyDatasets: unknown[] = [];
  
  return {
    // Data - all empty
    datasets: emptyDatasets,
    publicDatasets: emptyDatasets,
    localDatasets: emptyDatasets,
    mergedDatasets: emptyDatasets,
    totalDatasets: 0,
    availableDatasets: emptyDatasets,
    loadedDataset: null,
    loadedResults: null,
    
    // State
    isLoading,
    error,
    user: null,
    isAvailable: false,
    hasNPZSupport: false,
    
    // Actions - all no-ops
    fetchDatasets: async () => ({ success: false, data: [] }),
    loadDataset: async (..._args: unknown[]) => false,
    getBestResults: async () => null,
    formatCircuitParameters: () => [],
    saveDatasetReference: async () => ({ success: false }),
    updateAvailability: async () => ({ success: false }),
    tagModel: async () => ({ success: false }),
    getTaggedModels: async () => ({ data: [], error: null }),
    getStats: async () => ({ data: [], error: null }),
    
    // Computed
    isEmpty: true,
    hasLocalDatasets: false,
    hasPublicDatasets: false
  };
};