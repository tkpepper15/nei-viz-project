import { ModelSnapshot } from '../types';

export interface GridFilterSettings {
  enableSmartFiltering: boolean;
  visibilityPercentage: number;
  maxVisiblePoints: number;
  filterMode: 'best_resnorm' | 'distributed' | 'random';
  resnormThreshold: number;
  adaptiveFiltering: boolean;
}

export interface FilteredGridResult {
  visibleModels: ModelSnapshot[];
  hiddenCount: number;
  performanceImpact: 'low' | 'medium' | 'high' | 'critical';
  recommendation: string;
}

// Calculate smart filtering recommendations based on grid size
export const calculateSmartFilteringRecommendations = (gridSize: number) => {
  const totalPoints = Math.pow(gridSize, 5);
  
  if (totalPoints <= 243) { // 3^5
    return { 
      percentage: 100, 
      maxVisible: totalPoints,
      recommendation: 'Show all points - no filtering needed',
      mode: 'none' as const
    };
  } else if (totalPoints <= 1024) { // 4^5
    return { 
      percentage: 75, 
      maxVisible: Math.floor(totalPoints * 0.75),
      recommendation: 'Show 75% best resnorm - manageable performance',
      mode: 'light' as const
    };
  } else if (totalPoints <= 3125) { // 5^5
    return { 
      percentage: 50, 
      maxVisible: Math.floor(totalPoints * 0.5),
      recommendation: 'Show 50% best resnorm - balanced view',
      mode: 'moderate' as const
    };
  } else if (totalPoints <= 7776) { // 6^5
    return { 
      percentage: 25, 
      maxVisible: Math.floor(totalPoints * 0.25),
      recommendation: 'Show 25% best resnorm - performance focused',
      mode: 'aggressive' as const
    };
  } else if (totalPoints <= 16807) { // 7^5
    return { 
      percentage: 15, 
      maxVisible: Math.floor(totalPoints * 0.15),
      recommendation: 'Show 15% best resnorm - high performance mode',
      mode: 'ultra' as const
    };
  } else {
    return { 
      percentage: 10, 
      maxVisible: Math.floor(totalPoints * 0.1),
      recommendation: 'Show 10% best resnorm - critical performance mode',
      mode: 'critical' as const
    };
  }
};

// Sort models by resnorm (best first)
const sortByResnorm = (models: ModelSnapshot[]): ModelSnapshot[] => {
  return [...models].sort((a, b) => {
    const aResnorm = a.resnorm || Infinity;
    const bResnorm = b.resnorm || Infinity;
    return aResnorm - bResnorm;
  });
};

// Distributed sampling across parameter space
const distributedSampling = (models: ModelSnapshot[], count: number): ModelSnapshot[] => {
  if (models.length <= count) return models;
  
  const step = models.length / count;
  const sampled: ModelSnapshot[] = [];
  
  for (let i = 0; i < count; i++) {
    const index = Math.floor(i * step);
    if (index < models.length) {
      sampled.push(models[index]);
    }
  }
  
  return sampled;
};

// Random sampling
const randomSampling = (models: ModelSnapshot[], count: number): ModelSnapshot[] => {
  if (models.length <= count) return models;
  
  const shuffled = [...models].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, count);
};

// Smart grid filtering implementation
export const applySmartGridFiltering = (
  models: ModelSnapshot[],
  settings: GridFilterSettings
): FilteredGridResult => {
  // If smart filtering is disabled, return all models
  if (!settings.enableSmartFiltering) {
    return {
      visibleModels: models,
      hiddenCount: 0,
      performanceImpact: models.length > 5000 ? 'critical' : 
                        models.length > 2000 ? 'high' :
                        models.length > 1000 ? 'medium' : 'low',
      recommendation: 'Smart filtering disabled - showing all points'
    };
  }

  const targetCount = Math.floor((settings.visibilityPercentage / 100) * models.length);
  const maxAllowed = Math.min(targetCount, settings.maxVisiblePoints);
  
  let filteredModels: ModelSnapshot[];
  
  switch (settings.filterMode) {
    case 'best_resnorm':
      const sortedModels = sortByResnorm(models);
      filteredModels = sortedModels.slice(0, maxAllowed);
      break;
      
    case 'distributed':
      const sortedForDistribution = sortByResnorm(models);
      filteredModels = distributedSampling(sortedForDistribution, maxAllowed);
      break;
      
    case 'random':
      filteredModels = randomSampling(models, maxAllowed);
      break;
      
    default:
      filteredModels = models.slice(0, maxAllowed);
  }
  
  const hiddenCount = models.length - filteredModels.length;
  
  // Calculate performance impact
  let performanceImpact: 'low' | 'medium' | 'high' | 'critical';
  if (filteredModels.length > 5000) {
    performanceImpact = 'critical';
  } else if (filteredModels.length > 2000) {
    performanceImpact = 'high';
  } else if (filteredModels.length > 1000) {
    performanceImpact = 'medium';
  } else {
    performanceImpact = 'low';
  }
  
  // Generate recommendation
  let recommendation: string;
  
  if (hiddenCount === 0) {
    recommendation = 'Showing all computed points';
  } else if (settings.filterMode === 'best_resnorm') {
    recommendation = `Showing ${filteredModels.length.toLocaleString()} best models (${hiddenCount.toLocaleString()} hidden)`;
  } else {
    recommendation = `Applied ${settings.filterMode} sampling: ${filteredModels.length.toLocaleString()} visible`;
  }
  
  return {
    visibleModels: filteredModels,
    hiddenCount,
    performanceImpact,
    recommendation
  };
};

// Create default filter settings based on grid size
export const createDefaultFilterSettings = (gridSize: number): GridFilterSettings => {
  const recommendations = calculateSmartFilteringRecommendations(gridSize);
  
  return {
    enableSmartFiltering: gridSize > 3, // Auto-enable for grid size > 3
    visibilityPercentage: recommendations.percentage,
    maxVisiblePoints: recommendations.maxVisible,
    filterMode: 'best_resnorm',
    resnormThreshold: 0.1,
    adaptiveFiltering: true
  };
};

// Adaptive filtering based on system performance
export const getAdaptiveFilterSettings = (
  currentSettings: GridFilterSettings,
  systemMetrics: {
    memoryUsage: number;
    cpuUsage: number;
    fps: number;
  }
): GridFilterSettings => {
  if (!currentSettings.adaptiveFiltering) {
    return currentSettings;
  }
  
  let adjustedPercentage = currentSettings.visibilityPercentage;
  
  // Reduce visibility if system is under stress
  if (systemMetrics.memoryUsage > 80 || systemMetrics.cpuUsage > 85 || systemMetrics.fps < 30) {
    adjustedPercentage = Math.max(10, adjustedPercentage * 0.8);
  } else if (systemMetrics.memoryUsage < 50 && systemMetrics.cpuUsage < 60 && systemMetrics.fps > 45) {
    // Increase visibility if system has headroom
    adjustedPercentage = Math.min(100, adjustedPercentage * 1.1);
  }
  
  return {
    ...currentSettings,
    visibilityPercentage: Math.round(adjustedPercentage)
  };
};

// Export utility functions for grid analysis
export const analyzeGridComplexity = (gridSize: number) => {
  const totalPoints = Math.pow(gridSize, 5);
  const recommendations = calculateSmartFilteringRecommendations(gridSize);
  
  return {
    totalPoints,
    recommendedVisible: recommendations.maxVisible,
    complexity: recommendations.mode,
    memoryEstimate: totalPoints * 0.5, // KB estimate per point
    renderingCost: totalPoints > 10000 ? 'high' : totalPoints > 5000 ? 'medium' : 'low'
  };
}; 