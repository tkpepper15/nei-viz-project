import { useState, useCallback } from 'react';
import { ModelSnapshot, GridParameterArrays, ResnormGroup } from '../types';
import { WorkerProgress } from '../utils/workerManager';
import { ComputationSummary } from '../notifications/ComputationNotification';

export const useComputationState = (initialGridSize: number = 9) => {
  // Grid computation state - now using ModelSnapshot for modern serialized workflow
  const [gridResults, setGridResults] = useState<ModelSnapshot[]>([]);
  const [gridSize, setGridSize] = useState<number>(initialGridSize);
  const [defaultGridSize, setDefaultGridSize] = useState<number>(initialGridSize);
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
  
  // User-controlled computation result limits
  const [maxComputationResults, setMaxComputationResults] = useState<number>(500000); // Top N results to keep during computation

  // Resnorm groups for analysis
  const [resnormGroups, setResnormGroups] = useState<ResnormGroup[]>([]);
  const [hiddenGroups, setHiddenGroups] = useState<number[]>([]);
  
  // Preserve last computed results for tab switching
  const [lastComputedResults, setLastComputedResults] = useState<{
    resnormGroups: ResnormGroup[];
    gridResults: ModelSnapshot[];
    computationSummary: ComputationSummary | null;
  } | null>(null);

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

  // Update the default grid size (for when user preferences are loaded)
  const updateDefaultGridSize = useCallback((newDefaultGridSize: number) => {
    setDefaultGridSize(newDefaultGridSize);
    // If the current grid size is still the old default, update it too
    if (gridSize === defaultGridSize) {
      setGridSize(newDefaultGridSize);
    }
  }, [gridSize, defaultGridSize]);

  // Save current computed results before clearing
  const saveComputationState = useCallback(() => {
    if (resnormGroups.length > 0 || gridResults.length > 0) {
      setLastComputedResults({
        resnormGroups,
        gridResults,
        computationSummary
      });
    }
  }, [resnormGroups, gridResults, computationSummary]);

  // Restore previously computed results
  const restoreComputationState = useCallback(() => {
    if (lastComputedResults) {
      setResnormGroups(lastComputedResults.resnormGroups);
      setGridResults(lastComputedResults.gridResults);
      setComputationSummary(lastComputedResults.computationSummary);
      setGridError(null);
      updateStatusMessage('Restored previous computation results');
      return true;
    }
    return false;
  }, [lastComputedResults, updateStatusMessage]);

  // Reset only when starting a new computation
  const resetComputationState = useCallback(() => {
    setGridResults([]);
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

  // Clear all computation data including saved state
  const clearAllComputationData = useCallback(() => {
    resetComputationState();
    setLastComputedResults(null);
  }, [resetComputationState]);

  // Calculate effective visualization limit based on user preferences
  const calculateEffectiveVisualizationLimit = useCallback((totalComputed: number) => {
    if (!isUserControlledLimits) {
      // Use automatic memory-based limits (original behavior with higher limits)
      // Memory calculations simplified since we show all data for smaller grids
      
      // Enhanced limits for parameter experimentation (focused on top models)
      if (totalComputed > 1000000) {
        // For massive grids (1M+ models), show top few thousand only  
        const displayLimit = Math.min(100000, Math.max(10000, Math.floor(totalComputed * 0.1))); // 10% of total for massive grids
        console.log(`Massive grid detected: ${totalComputed.toLocaleString()} models computed, displaying top ${displayLimit.toLocaleString()} (${(displayLimit/totalComputed*100).toFixed(3)}%)`);
        return displayLimit;
      } else {
        // For smaller grids, show all computed data
        return totalComputed; // Show all computed models
      }
    }
    
    // Use user-controlled limits
    const userLimit = Math.floor((userVisualizationPercentage / 100) * totalComputed);
    return Math.min(userLimit, maxVisualizationPoints, totalComputed);
  }, [isUserControlledLimits, userVisualizationPercentage, maxVisualizationPoints]);

  return {
    // Grid computation state
    gridResults,
    setGridResults,
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
    
    // User-controlled computation limits
    maxComputationResults,
    setMaxComputationResults,

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
    
    // State management
    resetComputationState,
    saveComputationState,
    restoreComputationState,
    clearAllComputationData,
    lastComputedResults,

    // Grid size management
    defaultGridSize,
    updateDefaultGridSize,
  };
};