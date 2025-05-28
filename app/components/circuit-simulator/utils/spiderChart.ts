import { ModelSnapshot, SpiderData, SpiderDataPoint, SpiderState } from '../types';

// Generate data for spider/radar chart
export const generateSpiderChartData = (snapshots: ModelSnapshot[]): SpiderData => {
  if (!snapshots.length) {
    return { data: [], states: [] };
  }
  
  // Get parameter names (first level keys from parameters)
  const paramNames = Object.keys(snapshots[0].parameters);
  
  // Normalize parameters across all snapshots
  const normalizedParams = normalizeParameters(snapshots, paramNames);
  
  // Create data points for the chart
  const data: SpiderDataPoint[] = paramNames.map(param => {
    const dataPoint: SpiderDataPoint = { parameter: param };
    snapshots.forEach(snapshot => {
      dataPoint[snapshot.id] = normalizedParams[snapshot.id][param];
    });
    return dataPoint;
  });
  
  // Create state definitions for the chart
  const states: SpiderState[] = snapshots.map(snapshot => ({
    name: snapshot.name,
    color: snapshot.color,
    dataKey: snapshot.id
  }));
  
  return { data, states };
};

// Normalize parameters to values between 0-1 for the radar chart
const normalizeParameters = (snapshots: ModelSnapshot[], paramNames: string[]): Record<string, Record<string, number>> => {
  const result: Record<string, Record<string, number>> = {};
  
  // Find min/max values for each parameter
  const minMax: Record<string, { min: number, max: number }> = {};
  
  paramNames.forEach(param => {
    // Skip frequency_range parameter which is an array
    if (param === 'frequency_range') return;

    const values = snapshots.map(snapshot => snapshot.parameters[param as keyof typeof snapshot.parameters]);
    // Filter out undefined values and provide defaults
    const filteredValues = values.map(v => v ?? 0).filter(v => typeof v === 'number');
    minMax[param] = {
      min: Math.min(...filteredValues),
      max: Math.max(...filteredValues)
    };
  });
  
  // Normalize values for each snapshot
  snapshots.forEach(snapshot => {
    result[snapshot.id] = {};
    paramNames.forEach(param => {
      // Skip frequency_range parameter which is an array
      if (param === 'frequency_range') {
        result[snapshot.id][param] = 0.5; // Set to middle value
        return;
      }

      const paramKey = param as keyof typeof snapshot.parameters;
      const value = snapshot.parameters[paramKey] ?? 0; // Default to 0 if undefined
      
      // Skip if not a number
      if (typeof value !== 'number') {
        result[snapshot.id][param] = 0.5; // Default value
        return;
      }
      
      const { min, max } = minMax[param];
      
      // Handle case where min and max are the same (avoid division by zero)
      if (min === max) {
        result[snapshot.id][param] = 0.5; // Set to middle value
      } else {
        result[snapshot.id][param] = (value - min) / (max - min);
      }
    });
  });
  
  return result;
}; 