import { useState, useCallback } from 'react';
import { BackendMeshPoint, GridParameterArrays, ResnormGroup } from '../types';
import { WorkerProgress } from '../utils/workerManager';
import { ComputationSummary } from '../notifications/ComputationNotification';

export const useComputationState = () => {
  // Grid computation state
  const [gridResults, setGridResults] = useState<BackendMeshPoint[]>([]);
  const [gridResultsWithIds, setGridResultsWithIds] = useState<(BackendMeshPoint & { id: number })[]>([]);
  const [gridSize, setGridSize] = useState<number>(5);
  const [gridError, setGridError] = useState<string | null>(null);
  const [isComputingGrid, setIsComputingGrid] = useState<boolean>(false);
  const [gridParameterArrays, setGridParameterArrays] = useState<GridParameterArrays | null>(null);

  // Progress tracking
  const [computationProgress, setComputationProgress] = useState<WorkerProgress | null>(null);
  const [computationSummary, setComputationSummary] = useState<ComputationSummary | null>(null);
  
  // Track skipped points from symmetric optimization
  const [skippedPoints, setSkippedPoints] = useState<number>(0);
  const [totalGridPoints, setTotalGridPoints] = useState<number>(0);
  const [actualComputedPoints, setActualComputedPoints] = useState<number>(0);
  
  // Track memory management limitations
  const [memoryLimitedPoints, setMemoryLimitedPoints] = useState<number>(0);
  const [estimatedMemoryUsage, setEstimatedMemoryUsage] = useState<number>(0);
  
  // User-controlled visualization limits
  const [userVisualizationPercentage, setUserVisualizationPercentage] = useState<number>(100); // Default 100%
  const [maxVisualizationPoints, setMaxVisualizationPoints] = useState<number>(1000000); // 1M max
  const [isUserControlledLimits, setIsUserControlledLimits] = useState<boolean>(false);

  // Resnorm groups for analysis
  const [resnormGroups, setResnormGroups] = useState<ResnormGroup[]>([]);
  const [hiddenGroups, setHiddenGroups] = useState<number[]>([]);

  // Activity log state
  const [logMessages, setLogMessages] = useState<{time: string, message: string}[]>([]);
  const [statusMessage, setStatusMessage] = useState<string>('');

  // Helper functions
  const addLogMessage = useCallback((message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogMessages(prev => [...prev, { time: timestamp, message }]);
  }, []);

  const updateStatusMessage = useCallback((message: string) => {
    setStatusMessage(message);
    addLogMessage(message);
  }, [addLogMessage]);

  const clearLogs = useCallback(() => {
    setLogMessages([]);
    setStatusMessage('');
  }, []);

  const resetComputationState = useCallback(() => {
    setGridResults([]);
    setGridResultsWithIds([]);
    setGridError(null);
    setResnormGroups([]);
    setHiddenGroups([]);
    setComputationProgress(null);
    setComputationSummary(null);
    setSkippedPoints(0);
    setTotalGridPoints(0);
    setActualComputedPoints(0);
    setMemoryLimitedPoints(0);
    setEstimatedMemoryUsage(0);
  }, []);

  // Calculate effective visualization limit based on user preferences
  const calculateEffectiveVisualizationLimit = useCallback((totalComputed: number) => {
    if (!isUserControlledLimits) {
      // Use automatic memory-based limits (original behavior with higher limits)
      const avgSpectrumSize = 20 * 40; // bytes per spectrum point (rough estimate)
      const estimatedMemory = totalComputed * (500 + avgSpectrumSize) / 1024 / 1024; // MB
      
      // Much more aggressive limits - prioritize user experience over conservative memory management
      if (estimatedMemory > 2000) return 250000; // 2GB -> 250K points
      if (estimatedMemory > 1000) return 500000; // 1GB -> 500K points  
      if (estimatedMemory > 500) return 750000;  // 500MB -> 750K points
      if (estimatedMemory > 200) return 1000000; // 200MB -> 1M points (our max)
      return Math.min(1000000, totalComputed); // Default to 1M max
    }
    
    // Use user-controlled limits
    const userLimit = Math.floor((userVisualizationPercentage / 100) * totalComputed);
    return Math.min(userLimit, maxVisualizationPoints, totalComputed);
  }, [isUserControlledLimits, userVisualizationPercentage, maxVisualizationPoints]);

  return {
    // Grid computation state
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
    
    // Symmetric optimization tracking
    skippedPoints,
    setSkippedPoints,
    totalGridPoints,
    setTotalGridPoints,
    actualComputedPoints,
    setActualComputedPoints,
    
    // Memory management tracking
    memoryLimitedPoints,
    setMemoryLimitedPoints,
    estimatedMemoryUsage,
    setEstimatedMemoryUsage,
    
    // User-controlled visualization limits
    userVisualizationPercentage,
    setUserVisualizationPercentage,
    maxVisualizationPoints,
    setMaxVisualizationPoints,
    isUserControlledLimits,
    setIsUserControlledLimits,
    calculateEffectiveVisualizationLimit,

    // Resnorm groups
    resnormGroups,
    setResnormGroups,
    hiddenGroups,
    setHiddenGroups,

    // Activity log
    logMessages,
    statusMessage,
    addLogMessage,
    updateStatusMessage,
    clearLogs,
    resetComputationState,
  };
};