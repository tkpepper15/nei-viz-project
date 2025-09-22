/**
 * Diagnostics Tab
 * ===============
 *
 * Provides system diagnostics, performance metrics, and error tracking
 * similar to PyCharm's inspection and error reporting tools.
 */

import React, { useMemo, useState, useEffect } from 'react';
import { BottomPanelTabProps } from '../CollapsibleBottomPanel';
import { ExclamationTriangleIcon, CheckCircleIcon, InformationCircleIcon, ClockIcon } from '@heroicons/react/24/outline';

interface DiagnosticItem {
  id: string;
  type: 'error' | 'warning' | 'info' | 'success';
  title: string;
  message: string;
  timestamp: Date;
  details?: string;
  actionable?: boolean;
}

interface PerformanceMetric {
  name: string;
  value: number;
  unit: string;
  status: 'good' | 'warning' | 'critical';
  threshold?: number;
}

export const DiagnosticsTab: React.FC<BottomPanelTabProps> = ({
  gridResults,
  topConfigurations,
  currentParameters,
  selectedConfigIndex,
  isVisible,
  highlightedModelId
}) => {
  const [diagnostics, setDiagnostics] = useState<DiagnosticItem[]>([]);
  const [activeFilter, setActiveFilter] = useState<'all' | 'error' | 'warning' | 'info' | 'success'>('all');
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Get highlighted circuit parameters from the spider plot
  const highlightedModel = useMemo(() => {
    if (!highlightedModelId || topConfigurations.length === 0) {
      return null;
    }

    // Search through all configurations for the highlighted model
    for (const config of topConfigurations) {
      if (config.items && config.items.length > 0) {
        const model = config.items.find(item => item.id === highlightedModelId);
        if (model) {
          return {
            id: model.id,
            name: model.name || 'Highlighted Model',
            parameters: model.parameters || {},
            resnorm: model.resnorm || 0,
            configLabel: config.label || 'Unknown Configuration'
          };
        }
      }
    }
    return null;
  }, [highlightedModelId, topConfigurations]);

  // Use highlighted model parameters if available, otherwise fall back to current parameters
  const activeParameters = highlightedModel?.parameters || currentParameters;

  // Debug logging for highlighted model integration
  useEffect(() => {
    console.log('🔧 DiagnosticsTab highlighted model update:', {
      highlightedModelId,
      highlightedModel: highlightedModel ? {
        id: highlightedModel.id,
        name: highlightedModel.name,
        configLabel: highlightedModel.configLabel,
        resnorm: highlightedModel.resnorm
      } : null,
      usingActiveParameters: highlightedModel ? 'highlighted model' : 'current parameters',
      parametersAvailable: Object.keys(activeParameters).length
    });
  }, [highlightedModelId, highlightedModel, activeParameters]);

  // Performance metrics calculation
  const performanceMetrics = useMemo(() => {
    const metrics: PerformanceMetric[] = [];

    // Grid computation metrics
    if (gridResults.length > 0) {
      const totalPossibleConfigs = Math.pow(9, 5); // Assuming 9x9x9x9x9 grid
      const completionRate = (gridResults.length / totalPossibleConfigs) * 100;

      metrics.push({
        name: 'Grid Completion',
        value: completionRate,
        unit: '%',
        status: completionRate > 80 ? 'good' : completionRate > 40 ? 'warning' : 'critical',
        threshold: 80
      });

      // Memory usage estimation
      const estimatedMemoryMB = gridResults.length * 0.5; // ~500 bytes per result
      metrics.push({
        name: 'Memory Usage',
        value: estimatedMemoryMB,
        unit: 'MB',
        status: estimatedMemoryMB < 100 ? 'good' : estimatedMemoryMB < 500 ? 'warning' : 'critical',
        threshold: 100
      });
    }

    // Configuration quality metrics
    if (topConfigurations.length > 0) {
      const avgResnorm = topConfigurations.reduce((sum, config) => sum + ((config.range[0] + config.range[1]) / 2), 0) / topConfigurations.length;

      metrics.push({
        name: 'Average Resnorm',
        value: avgResnorm,
        unit: '',
        status: avgResnorm < 10 ? 'good' : avgResnorm < 50 ? 'warning' : 'critical',
        threshold: 10
      });

      const bestResnorm = Math.min(...topConfigurations.map(config => config.range[0]));
      metrics.push({
        name: 'Best Resnorm',
        value: bestResnorm,
        unit: '',
        status: bestResnorm < 5 ? 'good' : bestResnorm < 20 ? 'warning' : 'critical',
        threshold: 5
      });
    }

    // Parameter validation using active parameters (highlighted model or current)
    const parameterIssues = validateParameters(activeParameters);
    if (parameterIssues.length === 0) {
      metrics.push({
        name: 'Parameter Validation',
        value: 100,
        unit: '%',
        status: 'good'
      });
    } else {
      metrics.push({
        name: 'Parameter Issues',
        value: parameterIssues.length,
        unit: 'issues',
        status: parameterIssues.length > 3 ? 'critical' : 'warning'
      });
    }

    return metrics;
  }, [gridResults, topConfigurations, activeParameters]);

  // Parameter validation function
  const validateParameters = (params: { Rsh: number; Ra: number; Rb: number; Ca: number; Cb: number; frequency_range?: number[] }) => {
    const issues: string[] = [];

    if (params.Rsh <= 0) issues.push('Shunt resistance must be positive');
    if (params.Ra <= 0) issues.push('Apical resistance must be positive');
    if (params.Rb <= 0) issues.push('Basal resistance must be positive');
    if (params.Ca <= 0) issues.push('Apical capacitance must be positive');
    if (params.Cb <= 0) issues.push('Basal capacitance must be positive');

    if (params.Rsh > 100000) issues.push('Shunt resistance unusually high (>100kΩ)');
    if (params.Ca > 100e-6) issues.push('Apical capacitance unusually high (>100μF)');
    if (params.Cb > 100e-6) issues.push('Basal capacitance unusually high (>100μF)');

    if (params.frequency_range && params.frequency_range[0] >= params.frequency_range[1]) {
      issues.push('Minimum frequency must be less than maximum frequency');
    }

    return issues;
  };

  // Generate diagnostics based on current state
  useEffect(() => {
    if (!isVisible || !autoRefresh) return;

    const newDiagnostics: DiagnosticItem[] = [];

    // Add highlighted model information if available
    if (highlightedModel) {
      newDiagnostics.push({
        id: 'highlighted-model',
        type: 'success',
        title: 'Highlighted Circuit Selected',
        message: `Analyzing circuit: ${highlightedModel.name} from ${highlightedModel.configLabel}`,
        timestamp: new Date(),
        details: `Model ID: ${highlightedModel.id}, Resnorm: ${highlightedModel.resnorm.toFixed(6)}`
      });
    }

    // Parameter validation diagnostics using active parameters
    const parameterIssues = validateParameters(activeParameters);
    parameterIssues.forEach((issue, index) => {
      newDiagnostics.push({
        id: `param-${index}`,
        type: 'warning',
        title: highlightedModel ? 'Highlighted Model Parameter Issue' : 'Parameter Validation',
        message: issue,
        timestamp: new Date(),
        actionable: true
      });
    });

    // Grid computation diagnostics
    if (gridResults.length === 0) {
      newDiagnostics.push({
        id: 'no-results',
        type: 'info',
        title: 'No Computation Results',
        message: 'No grid computation results available. Start a computation to see data.',
        timestamp: new Date(),
        actionable: true
      });
    } else {
      newDiagnostics.push({
        id: 'computation-success',
        type: 'success',
        title: 'Computation Complete',
        message: `Successfully computed ${gridResults.length} configurations.`,
        timestamp: new Date()
      });
    }

    // Configuration selection diagnostics
    if (topConfigurations.length > 0) {
      const selectedConfig = topConfigurations[selectedConfigIndex];
      if (selectedConfig) {
        const resnorm = (selectedConfig.range[0] + selectedConfig.range[1]) / 2;

        if (resnorm < 5) {
          newDiagnostics.push({
            id: 'excellent-fit',
            type: 'success',
            title: 'Excellent Parameter Fit',
            message: `Selected configuration has excellent resnorm: ${resnorm.toFixed(4)}`,
            timestamp: new Date()
          });
        } else if (resnorm > 50) {
          newDiagnostics.push({
            id: 'poor-fit',
            type: 'warning',
            title: 'Poor Parameter Fit',
            message: `Selected configuration has high resnorm: ${resnorm.toFixed(4)}. Consider different parameters.`,
            timestamp: new Date(),
            actionable: true
          });
        }
      }
    }

    // Performance diagnostics
    performanceMetrics.forEach(metric => {
      if (metric.status === 'critical') {
        newDiagnostics.push({
          id: `perf-${metric.name}`,
          type: 'error',
          title: 'Performance Issue',
          message: `${metric.name}: ${metric.value}${metric.unit} is critically high.`,
          timestamp: new Date(),
          details: metric.threshold ? `Recommended threshold: ${metric.threshold}${metric.unit}` : undefined,
          actionable: true
        });
      } else if (metric.status === 'warning') {
        newDiagnostics.push({
          id: `perf-warn-${metric.name}`,
          type: 'warning',
          title: 'Performance Warning',
          message: `${metric.name}: ${metric.value}${metric.unit} approaching threshold.`,
          timestamp: new Date(),
          details: metric.threshold ? `Recommended threshold: ${metric.threshold}${metric.unit}` : undefined
        });
      }
    });

    setDiagnostics(newDiagnostics);
  }, [isVisible, autoRefresh, activeParameters, gridResults, topConfigurations, selectedConfigIndex, performanceMetrics, highlightedModel]);

  // Filter diagnostics
  const filteredDiagnostics = useMemo(() => {
    return activeFilter === 'all'
      ? diagnostics
      : diagnostics.filter(d => d.type === activeFilter);
  }, [diagnostics, activeFilter]);

  // Get icon for diagnostic type
  const getIcon = (type: DiagnosticItem['type']) => {
    switch (type) {
      case 'error':
        return <ExclamationTriangleIcon className="w-4 h-4 text-red-400" />;
      case 'warning':
        return <ExclamationTriangleIcon className="w-4 h-4 text-yellow-400" />;
      case 'success':
        return <CheckCircleIcon className="w-4 h-4 text-green-400" />;
      case 'info':
        return <InformationCircleIcon className="w-4 h-4 text-blue-400" />;
      default:
        return <InformationCircleIcon className="w-4 h-4 text-neutral-400" />;
    }
  };

  // Get color classes for diagnostic type
  const getColorClasses = (type: DiagnosticItem['type']) => {
    switch (type) {
      case 'error':
        return 'border-red-500/20 bg-red-900/10';
      case 'warning':
        return 'border-yellow-500/20 bg-yellow-900/10';
      case 'success':
        return 'border-green-500/20 bg-green-900/10';
      case 'info':
        return 'border-blue-500/20 bg-blue-900/10';
      default:
        return 'border-neutral-500/20 bg-neutral-900/10';
    }
  };

  if (!isVisible) {
    return <div className="h-full flex items-center justify-center text-neutral-500">
      Loading diagnostics...
    </div>;
  }

  return (
    <div className="h-full flex flex-col bg-neutral-900 text-white">
      {/* Header with filters and controls */}
      <div className="p-4 border-b border-neutral-700 bg-neutral-800/50">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h3 className="text-lg font-semibold text-neutral-200">System Diagnostics</h3>
            {highlightedModel && (
              <p className="text-xs text-yellow-400 mt-1">
                🎯 Analyzing: {highlightedModel.name} (Resnorm: {highlightedModel.resnorm.toFixed(4)})
              </p>
            )}
          </div>
          <div className="flex items-center space-x-3">
            <label className="flex items-center space-x-2 text-sm">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="rounded bg-neutral-700 border-neutral-600"
              />
              <span className="text-neutral-400">Auto-refresh</span>
            </label>
            <span className="text-xs text-neutral-500">
              <ClockIcon className="w-3 h-3 inline mr-1" />
              {new Date().toLocaleTimeString()}
            </span>
          </div>
        </div>

        {/* Filter tabs */}
        <div className="flex space-x-1">
          {[
            { id: 'all', label: 'All', count: diagnostics.length },
            { id: 'error', label: 'Errors', count: diagnostics.filter(d => d.type === 'error').length },
            { id: 'warning', label: 'Warnings', count: diagnostics.filter(d => d.type === 'warning').length },
            { id: 'info', label: 'Info', count: diagnostics.filter(d => d.type === 'info').length },
            { id: 'success', label: 'Success', count: diagnostics.filter(d => d.type === 'success').length }
          ].map(filter => (
            <button
              key={filter.id}
              onClick={() => setActiveFilter(filter.id as 'all' | 'error' | 'warning' | 'info' | 'success')}
              className={`px-3 py-1.5 text-xs font-medium rounded transition-colors flex items-center space-x-1 ${
                activeFilter === filter.id
                  ? 'bg-neutral-700 text-white'
                  : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-700/50'
              }`}
            >
              <span>{filter.label}</span>
              {filter.count > 0 && (
                <span className="bg-neutral-600 text-white text-xs px-1.5 py-0.5 rounded-full min-w-[18px] text-center">
                  {filter.count}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Performance metrics */}
      <div className="p-4 border-b border-neutral-700 bg-neutral-800/20">
        <h4 className="text-sm font-medium text-neutral-300 mb-3">Performance Metrics</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {performanceMetrics.map((metric, index) => (
            <div key={index} className={`p-3 rounded-lg border ${getColorClasses(metric.status as DiagnosticItem['type'])}`}>
              <div className="flex items-center space-x-2 mb-1">
                {getIcon(metric.status as DiagnosticItem['type'])}
                <span className="text-xs text-neutral-400">{metric.name}</span>
              </div>
              <div className="font-mono text-lg text-neutral-200">
                {metric.value.toFixed(metric.unit === '%' ? 1 : 3)}{metric.unit}
              </div>
              {metric.threshold && (
                <div className="text-xs text-neutral-500 mt-1">
                  Threshold: {metric.threshold}{metric.unit}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Diagnostics list - Enhanced scrollable container */}
      <div className="flex-1 overflow-y-auto min-h-0">
        {filteredDiagnostics.length > 0 ? (
          <div className="p-4 space-y-3">
            {filteredDiagnostics.map(diagnostic => (
              <div
                key={diagnostic.id}
                className={`p-4 rounded-lg border ${getColorClasses(diagnostic.type)}`}
              >
                <div className="flex items-start space-x-3">
                  {getIcon(diagnostic.type)}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-1">
                      <h5 className="text-sm font-medium text-neutral-200">{diagnostic.title}</h5>
                      <span className="text-xs text-neutral-500">
                        {diagnostic.timestamp.toLocaleTimeString()}
                      </span>
                    </div>
                    <p className="text-sm text-neutral-300 mb-2">{diagnostic.message}</p>
                    {diagnostic.details && (
                      <p className="text-xs text-neutral-400 bg-neutral-800/50 p-2 rounded">
                        {diagnostic.details}
                      </p>
                    )}
                    {diagnostic.actionable && (
                      <div className="mt-2">
                        <span className="inline-block text-xs bg-blue-600 text-white px-2 py-1 rounded">
                          Action Required
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="h-full flex items-center justify-center text-neutral-500">
            <div className="text-center">
              <CheckCircleIcon className="w-12 h-12 mx-auto mb-3 text-green-500" />
              <p>No {activeFilter === 'all' ? '' : activeFilter} diagnostics found</p>
              <p className="text-sm mt-1">System is running smoothly</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DiagnosticsTab;