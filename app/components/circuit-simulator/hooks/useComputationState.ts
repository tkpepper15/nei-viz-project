import { useState, useCallback } from 'react';
import { BackendMeshPoint, GridParameterArrays, ResnormGroup } from '../types';
import { WorkerProgress } from '../utils/workerManager';
import { ComputationSummary } from '../notifications/ComputationNotification';

export const useComputationState = () => {
  // Grid computation state
  const [gridResults, setGridResults] = useState<BackendMeshPoint[]>([]);
  const [gridResultsWithIds, setGridResultsWithIds] = useState<(BackendMeshPoint & { id: number })[]>([]);
  const [gridSize, setGridSize] = useState<number>(3);
  const [gridError, setGridError] = useState<string | null>(null);
  const [isComputingGrid, setIsComputingGrid] = useState<boolean>(false);
  const [gridParameterArrays, setGridParameterArrays] = useState<GridParameterArrays | null>(null);

  // Progress tracking
  const [computationProgress, setComputationProgress] = useState<WorkerProgress | null>(null);
  const [computationSummary, setComputationSummary] = useState<ComputationSummary | null>(null);

  // Resnorm groups for analysis
  const [resnormGroups, setResnormGroups] = useState<ResnormGroup[]>([]);
  const [hiddenGroups, setHiddenGroups] = useState<number[]>([]);

  // Activity log state
  const [logMessages, setLogMessages] = useState<{time: string, message: string}[]>([]);
  const [statusMessage, setStatusMessage] = useState<string>('');

  // Update status message and log
  const updateStatusMessage = useCallback((message: string) => {
    setStatusMessage(message);
    const timestamp = new Date().toLocaleTimeString();
    
    // Categorize messages for better visibility in log
    let formattedMessage = message;
    
    if (message.includes('Comput') || message.includes('Grid')) {
      formattedMessage = `[Grid] ${message}`;
    } else if (message.includes('Parameters') || message.includes('parameter')) {
      formattedMessage = `[Params] ${message}`;
    } else if (message.includes('model') || message.includes('Model')) {
      formattedMessage = `[Visual] ${message}`;
    } else if (message.includes('Sort')) {
      formattedMessage = `[Table] ${message}`;
    }
    
    setLogMessages(prev => [...prev.slice(-49), { time: timestamp, message: formattedMessage }]);
  }, []);

  // Clear computation state
  const clearComputationState = useCallback(() => {
    setGridResults([]);
    setGridResultsWithIds([]);
    setGridError(null);
    setIsComputingGrid(false);
    setComputationProgress(null);
    setComputationSummary(null);
    setResnormGroups([]);
    setGridParameterArrays(null);
  }, []);

  // Reset computation progress
  const resetComputationProgress = useCallback(() => {
    setComputationProgress(null);
    setComputationSummary(null);
  }, []);

  return {
    // Grid results
    gridResults,
    setGridResults,
    gridResultsWithIds,
    setGridResultsWithIds,
    gridSize,
    setGridSize,
    gridError,
    setGridError,
    isComputingGrid,
    setIsComputingGrid,
    gridParameterArrays,
    setGridParameterArrays,

    // Progress tracking
    computationProgress,
    setComputationProgress,
    computationSummary,
    setComputationSummary,

    // Resnorm analysis
    resnormGroups,
    setResnormGroups,
    hiddenGroups,
    setHiddenGroups,

    // Activity log
    logMessages,
    setLogMessages,
    statusMessage,
    setStatusMessage,
    updateStatusMessage,

    // Utility functions
    clearComputationState,
    resetComputationProgress,
  };
};