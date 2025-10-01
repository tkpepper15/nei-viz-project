"use client";

import React, { useState, useCallback, useMemo, useEffect } from 'react';
import { CircuitParameters } from '../types/parameters';
import { StaticRenderSettings } from './StaticRenderControls';
import { EnhancedInput } from './EnhancedInput';
import { groupPortionSampler, groupPortionToPercentage, percentageToGroupPortion, calculateShownModels } from '../utils/parameterSampling';
import { SRDUploadInterface } from './SRDUploadInterface';
import { SerializedComputationManager } from '../utils/serializedComputationManager';


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
  onConfigurationNameBlur?: (name: string) => void;
  selectedProfileId?: string | null; // To know if there's a current profile to update
  // New SRD upload functionality
  onSRDUploaded?: (manager: SerializedComputationManager, metadata: { title: string; totalResults: number; gridSize: number }) => void;
  onUploadError?: (error: string) => void;
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
  onConfigurationNameBlur,
  selectedProfileId,
  onSRDUploaded,
  onUploadError
  // Removed resnorm config params since method is fixed to SSR
}) => {
  // Operation mode state - compute or upload
  const [operationMode, setOperationMode] = useState<'compute' | 'upload'>('compute');

  // Simple local state for immediate UI feedback
  const [localConfigName, setLocalConfigName] = useState(configurationName);
  const [isEditingName, setIsEditingName] = useState(false);

  // Only update local state when prop changes AND we're not currently editing
  useEffect(() => {
    if (!isEditingName) {
      setLocalConfigName(configurationName);
    }
  }, [configurationName, isEditingName]);
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
            value={localConfigName}
            onChange={(e) => {
              setLocalConfigName(e.target.value);
              // Update parent state immediately for responsive UI
              onConfigurationNameChange?.(e.target.value);
            }}
            onFocus={() => setIsEditingName(true)}
            onBlur={(e) => {
              setIsEditingName(false);
              // Trigger save when user finishes editing
              onConfigurationNameBlur?.(e.target.value);
            }}
            placeholder="Enter configuration name..."
            className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-neutral-200 placeholder-neutral-500 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 transition-colors"
          />
        </div>
      )}

      {/* Operation Mode Toggle */}
      <div className="mb-6">
        <div className="flex bg-neutral-800 border border-neutral-600 rounded-lg p-1">
          <button
            className={`flex-1 px-4 py-2.5 text-sm font-medium rounded-md transition-all duration-200 ${
              operationMode === 'compute'
                ? 'bg-orange-600 text-white shadow-sm'
                : 'text-neutral-400 hover:text-white hover:bg-neutral-700'
            }`}
            onClick={() => setOperationMode('compute')}
          >
            <div className="flex items-center justify-center gap-2">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
              </svg>
              Compute Grid
            </div>
          </button>
          <button
            className={`flex-1 px-4 py-2.5 text-sm font-medium rounded-md transition-all duration-200 ${
              operationMode === 'upload'
                ? 'bg-green-600 text-white shadow-sm'
                : 'text-neutral-400 hover:text-white hover:bg-neutral-700'
            }`}
            onClick={() => setOperationMode('upload')}
          >
            <div className="flex items-center justify-center gap-2">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              Upload Data
            </div>
          </button>
        </div>
        <div className="mt-2 text-xs text-neutral-500 text-center">
          {operationMode === 'compute'
            ? 'Configure parameters and compute new grid results'
            : 'Upload pre-computed serialized resnorm data (.json files)'
          }
        </div>
      </div>

      {/* Conditional Content Based on Operation Mode */}
      {operationMode === 'compute' ? (
        <div className="space-y-8">
          {/* Configuration Section */}
          <CollapsibleSection title="Configuration" defaultOpen={true}>
            <div className="space-y-8">
              {/* Frequency Range */}
              <div>
                <label className="block text-sm font-medium text-neutral-300 mb-4">Frequency Range</label>

                {/* Min and Max Frequency Inputs */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-xs text-neutral-400 mb-2">Min Frequency (Hz)</label>
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
                      className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded text-neutral-200 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 transition-colors"
                      placeholder="0.1 or 1e2"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-neutral-400 mb-2">Max Frequency (Hz)</label>
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
                      className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded text-neutral-200 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 transition-colors"
                      placeholder="1e5 or 100000"
                    />
                  </div>
                </div>

                {/* Frequency Range Validation Warnings */}
                {(minFreq >= maxFreq || minFreq <= 0 || maxFreq <= 0) && (
                  <div className="mt-4 p-3 bg-red-900/20 border border-red-700/50 rounded-lg">
                    <div className="text-red-300 text-sm space-y-1">
                      {(minFreq >= maxFreq) && (
                        <div>⚠️ Minimum frequency must be less than maximum frequency</div>
                      )}
                      {(minFreq <= 0 || maxFreq <= 0) && (
                        <div>⚠️ Frequencies must be positive values</div>
                      )}
                    </div>
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
                <div>
                  <div className="flex items-center justify-between mb-3">
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
              {/* Group Portion Control */}
              <div>
                <div className="flex items-center justify-between mb-3">
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
              </div>
              </div>
            </div>
          </CollapsibleSection>

          {/* Circuit Parameters Section */}
          <CollapsibleSection title="Circuit Parameters" defaultOpen={true}>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
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
            <div className="space-y-6">
              {/* Visualization Type */}
              <div>
                <label className="block text-sm font-medium text-neutral-200 mb-3">
                  Visualization Type
                </label>
                <select
                  value={staticRenderSettings.visualizationType === 'nyquist' ? 'nyquist' : 'spider3d'}
                  onChange={(e) => handleVisualizationTypeChange(e.target.value as 'spider3d' | 'nyquist')}
                  className="w-full px-3 py-2 bg-neutral-800 border border-neutral-600 rounded-lg text-neutral-200 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 transition-colors"
                >
                  <option value="spider3d">Spider 3D</option>
                  <option value="nyquist">Nyquist Plot</option>
                </select>
              </div>
            </div>
          </CollapsibleSection>


          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 pt-6">
            {selectedProfileId && (
              <button
                onClick={() => {
                  const name = configurationName.trim() || 'Untitled Configuration';
                  onSaveProfile(name, 'Last action: Parameters updated from center panel', false);
                }}
                disabled={isComputing || !allValid}
                className="flex-1 px-6 py-3 bg-green-600 hover:bg-green-700 disabled:bg-neutral-700 disabled:text-neutral-400 text-white font-medium rounded-lg transition-colors"
              >
                Update Profile
              </button>
            )}
            <button
              onClick={() => {
                const name = configurationName.trim() || 'Untitled Configuration';
                onSaveProfile(name, 'Last action: New profile saved from center panel', true);
              }}
              disabled={isComputing || !allValid}
              className={`flex-1 px-6 py-3 ${selectedProfileId ? 'bg-amber-600 hover:bg-amber-700' : 'bg-green-600 hover:bg-green-700'} disabled:bg-neutral-700 disabled:text-neutral-400 text-white font-medium rounded-lg transition-colors`}
            >
              {selectedProfileId ? 'Save As New' : 'Save Profile'}
            </button>
            <button
              onClick={onCompute}
              disabled={isComputing || !allValid}
              className="flex-1 px-8 py-3 bg-orange-600 hover:bg-orange-700 disabled:bg-neutral-700 disabled:text-neutral-400 text-white font-bold rounded-lg transition-colors relative overflow-hidden"
            >
              {isComputing && (
                <div className="absolute inset-0 bg-orange-700">
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-orange-500/30 to-transparent animate-pulse"></div>
                </div>
              )}
              <span className="relative z-10">
                {isComputing ? 'Computing...' : 'Compute Grid'}
              </span>
            </button>
          </div>

          {!allValid && validationErrors.length > 0 && (
            <div className="bg-yellow-900/30 border border-yellow-500/50 rounded-lg p-4 mt-6">
              <div className="text-yellow-200 font-medium mb-3">⚠️ Validation Errors:</div>
              <ul className="list-disc list-inside space-y-2 text-yellow-200">
                {validationErrors.map((error, index) => (
                  <li key={index} className="text-sm">{error}</li>
                ))}
              </ul>
              <div className="text-sm text-yellow-300 mt-3 pt-3 border-t border-yellow-500/30">
                Fix these errors to enable computation. You can enter any values, but they must be within valid ranges to proceed.
              </div>
            </div>
          )}
        </div>
      ) : (
        /* Upload Mode */
        <div className="space-y-6">
          <div className="text-center mb-6">
            <h2 className="text-2xl font-semibold text-white mb-2">Upload Serialized Data</h2>
            <p className="text-neutral-400">
              Skip computation entirely by uploading pre-computed .json files with resnorm analysis results
            </p>
          </div>

          {/* Upload Interface */}
          {onSRDUploaded && onUploadError ? (
            <SRDUploadInterface
              onSRDUploaded={(manager, metadata) => {
                // Update grid configuration to match uploaded data
                onGridSizeChange(metadata.gridSize);

                // Create a descriptive configuration name if none exists
                if (onConfigurationNameChange && !localConfigName) {
                  const timestamp = new Date().toLocaleDateString();
                  const newName = `${metadata.title} (Uploaded ${timestamp})`;
                  setLocalConfigName(newName);
                  onConfigurationNameChange(newName);
                }

                // Call the parent handler
                onSRDUploaded(manager, metadata);
              }}
              onError={onUploadError}
              className="w-full"
            />
          ) : (
            <div className="bg-orange-900/20 border border-orange-600/50 rounded-lg p-6 text-center">
              <svg className="w-12 h-12 text-orange-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
              <h3 className="text-lg font-semibold text-orange-200 mb-2">Upload Not Available</h3>
              <p className="text-orange-300 text-sm">
                SRD upload functionality is not configured for this component instance.
              </p>
            </div>
          )}


        </div>
      )}
    </div>
  );
};