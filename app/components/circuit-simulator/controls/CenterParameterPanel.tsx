"use client";

import React, { useState, useCallback, useMemo } from 'react';
import { CircuitParameters } from '../types/parameters';
import { StaticRenderSettings } from './StaticRenderControls';
import { EnhancedInput } from './EnhancedInput';
import { groupPortionSampler, groupPortionToPercentage, percentageToGroupPortion, calculateShownModels } from '../utils/parameterSampling';


// Collapsible section component
interface CollapsibleSectionProps {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({ 
  title, 
  children, 
  defaultOpen = false 
}) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="border border-neutral-700 rounded-lg bg-neutral-800/50">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-neutral-700/50 transition-colors rounded-t-lg"
      >
        <span className="text-sm font-medium text-neutral-200">{title}</span>
        <svg
          className={`w-4 h-4 transition-transform ${isOpen ? 'transform rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {isOpen && (
        <div className="px-4 pb-4">
          {children}
        </div>
      )}
    </div>
  );
};

// Points calculator component
interface PointsCalculatorProps {
  gridSize: number;
}

const PointsCalculator: React.FC<PointsCalculatorProps> = ({ gridSize }) => {
  const totalPoints = useMemo(() => Math.pow(gridSize, 5), [gridSize]);
  
  const getLoadColor = useCallback((points: number) => {
    if (points < 1000) return 'text-green-400';
    if (points < 10000) return 'text-yellow-400';
    if (points < 100000) return 'text-orange-400';
    return 'text-red-400';
  }, []);

  const getLoadText = useCallback((points: number) => {
    if (points < 1000) return 'Low';
    if (points < 10000) return 'Medium';
    if (points < 100000) return 'High';
    return 'Extreme';
  }, []);

  return (
    <div className={`text-sm font-medium ${getLoadColor(totalPoints)}`}>
      {totalPoints.toLocaleString()} pts ({getLoadText(totalPoints)})
    </div>
  );
};

// Main component props
interface CenterParameterPanelProps {
  circuitParams: CircuitParameters;
  onCircuitParamsChange: (params: CircuitParameters) => void;
  gridSize: number;
  onGridSizeChange: (size: number) => void;
  staticRenderSettings: StaticRenderSettings;
  onStaticRenderSettingsChange: (settings: StaticRenderSettings) => void;
  minFreq: number;
  maxFreq: number;
  onMinFreqChange: (freq: number) => void;
  onMaxFreqChange: (freq: number) => void;
  numPoints: number;
  onNumPointsChange: (points: number) => void;
  onCompute: () => void;
  onSaveProfile: (name: string, description?: string, forceNew?: boolean) => void;
  isComputing: boolean;
  configurationName?: string;
  onConfigurationNameChange?: (name: string) => void;
  selectedProfileId?: string | null; // To know if there's a current profile to update
  maxComputationResults: number;
  onMaxComputationResultsChange: (limit: number) => void;
  // Removed resnorm config props since method is fixed to SSR
}

export const CenterParameterPanel: React.FC<CenterParameterPanelProps> = ({
  circuitParams,
  onCircuitParamsChange,
  gridSize,
  onGridSizeChange,
  staticRenderSettings,
  onStaticRenderSettingsChange,
  minFreq,
  maxFreq,
  onMinFreqChange,
  onMaxFreqChange,
  numPoints,
  onNumPointsChange,
  onCompute,
  onSaveProfile,
  isComputing,
  configurationName = '',
  onConfigurationNameChange,
  selectedProfileId,
  maxComputationResults,
  onMaxComputationResultsChange
  // Removed resnorm config params since method is fixed to SSR
}) => {
  // Circuit parameter change handlers
  const handleCircuitParamChange = useCallback((param: keyof CircuitParameters, value: number) => {
    if (param === 'frequency_range') return; // Handle separately
    if (!circuitParams) return; // Guard against undefined circuitParams
    
    onCircuitParamsChange({
      ...circuitParams,
      [param]: value
    });
  }, [circuitParams, onCircuitParamsChange]);


  // Visualization type handler
  const handleVisualizationTypeChange = useCallback((type: 'spider3d' | 'nyquist') => {
    onStaticRenderSettingsChange({
      ...staticRenderSettings,
      visualizationType: type
    });
  }, [staticRenderSettings, onStaticRenderSettingsChange]);

  // Clean group portion handling using the parameter sampling utility
  const currentGroupPercentage = useMemo(() => {
    return groupPortionToPercentage(staticRenderSettings.groupPortion);
  }, [staticRenderSettings.groupPortion]);

  const handleGroupPortionPercentageChange = useCallback((percentageValue: number) => {
    const newGroupPortion = percentageToGroupPortion(percentageValue);
    onStaticRenderSettingsChange({
      ...staticRenderSettings,
      groupPortion: newGroupPortion
    });
  }, [staticRenderSettings, onStaticRenderSettingsChange]);

  // Removed unused resnorm method handler since method is fixed to SSR

  // Calculate actual number of models being shown using the utility
  const totalModels = useMemo(() => Math.pow(gridSize, 5), [gridSize]);
  const shownModels = useMemo(() => 
    calculateShownModels(totalModels, staticRenderSettings.groupPortion), 
    [totalModels, staticRenderSettings.groupPortion]
  );

  // Enhanced validation function that checks for valid numbers and ranges
  const validateParameter = useCallback((value: number, min: number, max: number): boolean => {
    return !isNaN(value) && isFinite(value) && value >= min && value <= max;
  }, []);

  // Check if all circuit parameters are valid (numeric and within range)
  const circuitParamsValid = useMemo(() => {
    if (!circuitParams || typeof circuitParams !== 'object') {
      return false;
    }
    
    return (
      validateParameter(circuitParams.Rsh, 10, 10000) &&
      validateParameter(circuitParams.Ra, 10, 10000) &&
      validateParameter(circuitParams.Ca, 0.1e-6, 50e-6) &&
      validateParameter(circuitParams.Rb, 10, 10000) &&
      validateParameter(circuitParams.Cb, 0.1e-6, 50e-6)
    );
  }, [circuitParams, validateParameter]);


  // Frequency range validation (convert Hz to kHz for validation)
  const frequencyRangeValid = useMemo(() => {
    // Check for the specific warning conditions
    const hasMinMaxWarning = minFreq >= maxFreq;
    const hasNegativeWarning = minFreq <= 0 || maxFreq <= 0;
    
    // If either warning condition is true, validation fails
    if (hasMinMaxWarning || hasNegativeWarning) {
      return false;
    }
    
    // Additional range validation
    return (
      validateParameter(minFreq / 1000, 0.0001, 10000) && // Allow down to 0.1 Hz (0.0001 kHz)
      validateParameter(maxFreq / 1000, 0.0001, 10000) &&
      minFreq < maxFreq
    );
  }, [minFreq, maxFreq, validateParameter]);

  // Number of frequency points validation
  const numPointsValid = useMemo(() => {
    return validateParameter(numPoints, 10, 1000);
  }, [numPoints, validateParameter]);


  // Grid size validation
  const gridSizeValid = useMemo(() => {
    return validateParameter(gridSize, 2, 30);
  }, [gridSize, validateParameter]);

  // Overall validation state
  const allValid = circuitParamsValid && frequencyRangeValid && numPointsValid && gridSizeValid;

  // Get validation errors for user feedback
  const getValidationErrors = useCallback(() => {
    const errors: string[] = [];
    
    if (!circuitParamsValid) {
      errors.push("Circuit parameters have invalid values");
    }
    if (!frequencyRangeValid) {
      errors.push("Frequency range is invalid (check min < max and valid ranges)");
    }
    if (!numPointsValid) {
      errors.push("Number of frequency points must be between 10 and 1000");
    }
    if (!gridSizeValid) {
      errors.push("Grid size must be between 2 and 30 points");
    }
    
    return errors;
  }, [circuitParamsValid, frequencyRangeValid, numPointsValid, gridSizeValid]);

  const validationErrors = getValidationErrors();

  return (
    <div className="w-full">
      {/* Welcome Header */}
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-white mb-2">What are you looking to visualize?</h1>
        <p className="text-neutral-400 text-lg">Configure your circuit parameters for the electrochemical impedance spectroscopy simulation</p>
      </div>

      {/* Configuration Name Input - Moved to top */}
      {onConfigurationNameChange && (
        <div className="mb-6">
          <label className="block text-sm font-medium text-neutral-300 mb-2">
            Configuration Name
          </label>
          <input
            type="text"
            value={configurationName}
            onChange={(e) => onConfigurationNameChange(e.target.value)}
            placeholder="Enter configuration name..."
            className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-neutral-200 placeholder-neutral-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
          />
        </div>
      )}

      <div className="space-y-6">
          {/* Configuration Section */}
          <CollapsibleSection title="Configuration" defaultOpen={true}>
            <div className="space-y-6 mt-4">
              {/* Frequency Range */}
              <div>
                <label className="block text-sm font-medium text-neutral-300 mb-3">Frequency Range</label>
                
                {/* Min and Max Frequency Inputs */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs text-neutral-400 mb-1">Min Frequency (Hz)</label>
                    <input
                      type="text"
                      value={minFreq === 0 ? '' : minFreq.toString()}
                      onChange={(e) => {
                        const value = e.target.value.trim();
                        if (value === '') {
                          onMinFreqChange(0);
                          return;
                        }
                        
                        // Parse value, supporting scientific notation
                        const numValue = parseFloat(value);
                        if (!isNaN(numValue) && numValue >= 0) {
                          onMinFreqChange(numValue);
                        }
                        // If invalid, don't update but allow user to keep typing
                      }}
                      className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded text-neutral-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                      placeholder="0.1 or 1e2"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-neutral-400 mb-1">Max Frequency (Hz)</label>
                    <input
                      type="text"
                      value={maxFreq === 0 ? '' : maxFreq.toString()}
                      onChange={(e) => {
                        const value = e.target.value.trim();
                        if (value === '') {
                          onMaxFreqChange(0);
                          return;
                        }
                        
                        // Parse value, supporting scientific notation
                        const numValue = parseFloat(value);
                        if (!isNaN(numValue) && numValue >= 0) {
                          onMaxFreqChange(numValue);
                        }
                        // If invalid, don't update but allow user to keep typing
                      }}
                      className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded text-neutral-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                      placeholder="1e5 or 100000"
                    />
                  </div>
                </div>

                {/* Frequency Range Validation Warnings */}
                {(minFreq >= maxFreq) && (
                  <div className="mb-3 p-2 bg-red-900/20 border border-red-700/50 rounded text-red-300 text-xs">
                    ⚠️ Warning: Minimum frequency must be less than maximum frequency
                  </div>
                )}
                {(minFreq <= 0 || maxFreq <= 0) && (
                  <div className="mb-3 p-2 bg-red-900/20 border border-red-700/50 rounded text-red-300 text-xs">
                    ⚠️ Warning: Frequencies must be positive values
                  </div>
                )}
              </div>

              {/* Frequency Points */}
              <div>
                <EnhancedInput
                  label="Frequency Points"
                  value={numPoints}
                  onChange={onNumPointsChange}
                  min={10}
                  max={1000}
                  step={1}
                  showSlider={true}
                  unit=""
                />
              </div>


              {/* Grid Configuration */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Points per Parameter */}
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium text-neutral-200">
                      Points per Parameter
                    </label>
                    <PointsCalculator gridSize={gridSize} />
                  </div>
                  <EnhancedInput
                    label=""
                    value={gridSize}
                    onChange={onGridSizeChange}
                    min={2}
                    max={30}
                    step={1}
                    showSlider={true}
                  />
                </div>

                {/* Computation Result Limit */}
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium text-neutral-200">
                      Max Results to Store
                    </label>
                    <div className="text-xs text-neutral-400">
                      Top {maxComputationResults.toLocaleString()} kept
                    </div>
                  </div>
                  <select
                    value={maxComputationResults}
                    onChange={(e) => onMaxComputationResultsChange(parseInt(e.target.value))}
                    className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-neutral-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                  >
                    <option value={500}>500 (Ultra Fast)</option>
                    <option value={1000}>1,000 (Fast)</option>
                    <option value={2000}>2,000 (Balanced)</option>
                    <option value={3000}>3,000 (Quality)</option>
                    <option value={5000}>5,000 (High Quality)</option>
                  </select>
                  <div className="text-xs text-neutral-500">
                    Higher limits find better results but use more memory
                  </div>
                </div>
              </div>

              {/* Visualization Controls */}
              <div className="grid grid-cols-1 gap-6">
                {/* Group Portion - moved from above */}
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium text-neutral-200">
                      Group Portion (%)
                    </label>
                    <div className="text-sm text-neutral-400">
                      {shownModels.toLocaleString()} models
                    </div>
                  </div>
                  <EnhancedInput
                    label=""
                    value={parseFloat(groupPortionSampler.formatValue(currentGroupPercentage))}
                    onChange={handleGroupPortionPercentageChange}
                    min={0.01}
                    max={100}
                    step={groupPortionSampler.getStepSize(currentGroupPercentage)}
                    showSlider={true}
                  />
                  <div className="flex justify-between text-xs text-neutral-500">
                  </div>
                </div>
              </div>
            </div>
          </CollapsibleSection>

          {/* Circuit Parameters Section */}
          <CollapsibleSection title="Circuit Parameters" defaultOpen={true}>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-4">
              <EnhancedInput
                label="R shunt"
                value={isNaN(circuitParams?.Rsh || 0) ? 0 : (circuitParams?.Rsh || 0)}
                onChange={(value) => handleCircuitParamChange('Rsh', value)}
                unit="Ω"
                min={10}
                max={10000}
                step={10}
                showSlider={true}
              />
              <EnhancedInput
                label="Ra"
                value={isNaN(circuitParams?.Ra || 0) ? 0 : (circuitParams?.Ra || 0)}
                onChange={(value) => handleCircuitParamChange('Ra', value)}
                unit="Ω"
                min={10}
                max={10000}
                step={10}
                showSlider={true}
              />
              <EnhancedInput
                label="Ca"
                value={isNaN((circuitParams?.Ca || 0) * 1e6) ? 0 : (circuitParams?.Ca || 0) * 1e6}
                onChange={(value) => handleCircuitParamChange('Ca', value / 1e6)}
                unit="μF"
                min={0.1}
                max={50}
                step={0.1}
                showSlider={true}
              />
              <EnhancedInput
                label="Rb"
                value={isNaN(circuitParams?.Rb || 0) ? 0 : (circuitParams?.Rb || 0)}
                onChange={(value) => handleCircuitParamChange('Rb', value)}
                unit="Ω"
                min={10}
                max={10000}
                step={100}
                showSlider={true}
              />
              <EnhancedInput
                label="Cb"
                value={isNaN((circuitParams?.Cb || 0) * 1e6) ? 0 : (circuitParams?.Cb || 0) * 1e6}
                onChange={(value) => handleCircuitParamChange('Cb', value / 1e6)}
                unit="μF"
                min={0.1}
                max={50}
                step={0.1}
                showSlider={true}
              />
            </div>
          </CollapsibleSection>

          {/* Visualization Settings */}
          <CollapsibleSection title="Visualization Settings" defaultOpen={true}>
            <div className="space-y-4 mt-4">
              {/* Visualization Type */}
              <div>
                <label className="block text-sm font-medium text-neutral-200 mb-2">
                  Visualization Type
                </label>
                <select
                  value={staticRenderSettings.visualizationType === 'nyquist' ? 'nyquist' : 'spider3d'}
                  onChange={(e) => handleVisualizationTypeChange(e.target.value as 'spider3d' | 'nyquist')}
                  className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-neutral-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                >
                  <option value="spider3d">Spider 3D</option>
                  <option value="nyquist">Nyquist Plot</option>
                </select>
              </div>

              {/* Resnorm Method - Fixed to SSR */}
              <div>
                <label className="block text-sm font-medium text-neutral-200 mb-2">
                  Resnorm Method
                </label>
                <div className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-neutral-200">
                  SSR - Sum of Squared Residuals (Fixed)
                </div>
                <p className="text-xs text-neutral-400 mt-1">
                  Classic least-squares in complex plane - equal weight to real and imaginary parts
                </p>
              </div>

            </div>
          </CollapsibleSection>


          {/* Action Buttons */}
          <div className="flex gap-3 pt-4">
            {selectedProfileId && (
              <button
                onClick={() => {
                  const name = configurationName.trim() || 'Untitled Configuration';
                  onSaveProfile(name, 'Last action: Parameters updated from center panel', false);
                }}
                disabled={isComputing || !allValid}
                className="flex-1 px-4 py-3 bg-green-600 hover:bg-green-700 disabled:bg-neutral-700 disabled:text-neutral-400 text-white font-medium rounded-lg transition-colors"
              >
                Save
              </button>
            )}
            <button
              onClick={() => {
                const name = configurationName.trim() || 'Untitled Configuration';
                onSaveProfile(name, 'Last action: New profile saved from center panel', true);
              }}
              disabled={isComputing || !allValid}
              className={`flex-1 px-4 py-3 ${selectedProfileId ? 'bg-amber-600 hover:bg-amber-700' : 'bg-green-600 hover:bg-green-700'} disabled:bg-neutral-700 disabled:text-neutral-400 text-white font-medium rounded-lg transition-colors`}
            >
              {selectedProfileId ? 'Save As New' : 'Save Profile'}
            </button>
            <button
              onClick={onCompute}
              disabled={isComputing || !allValid}
              className="flex-1 px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-neutral-700 disabled:text-neutral-400 text-white font-medium rounded-lg transition-colors relative overflow-hidden"
            >
              {isComputing && (
                <div className="absolute inset-0 bg-blue-700">
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-blue-500/30 to-transparent animate-pulse"></div>
                </div>
              )}
              <span className="relative z-10">
                {isComputing ? 'Computing...' : 'Compute'}
              </span>
            </button>
          </div>

          {!allValid && validationErrors.length > 0 && (
            <div className="bg-yellow-900/30 border border-yellow-500/50 rounded-lg p-3 text-yellow-200 text-sm space-y-2">
              <div className="font-medium">⚠️ Validation Errors:</div>
              <ul className="list-disc list-inside space-y-1">
                {validationErrors.map((error, index) => (
                  <li key={index} className="text-xs">{error}</li>
                ))}
              </ul>
              <div className="text-xs text-yellow-300 mt-2">
                Fix these errors to enable computation. You can enter any values, but they must be within valid ranges to proceed.
              </div>
            </div>
          )}
        </div>
    </div>
  );
};