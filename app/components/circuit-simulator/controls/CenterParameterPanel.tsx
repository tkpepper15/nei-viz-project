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
    <div className="border border-border rounded">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-3 py-2 flex items-center justify-between text-left hover:bg-neutral-800/40 transition-colors"
      >
        <span className="text-xs font-medium text-neutral-400">{title}</span>
        <svg
          className={`w-3 h-3 text-neutral-500 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {isOpen && (
        <div className="px-3 pb-3 pt-1">
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
    if (points < 10000) return 'text-neutral-400';
    if (points < 100000) return 'text-neutral-400';
    return 'text-neutral-500';
  }, []);

  const getLoadText = useCallback((points: number) => {
    if (points < 1000) return 'lo';
    if (points < 10000) return 'med';
    if (points < 100000) return 'hi';
    return 'extreme';
  }, []);

  return (
    <div className={`text-xs font-mono ${getLoadColor(totalPoints)}`}>
      {totalPoints.toLocaleString()} ({getLoadText(totalPoints)})
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
    <div className="w-full space-y-4">
      {/* Circuit name */}
      {onConfigurationNameChange && (
        <input
          type="text"
          value={localConfigName}
          onChange={(e) => { setLocalConfigName(e.target.value); onConfigurationNameChange?.(e.target.value); }}
          onFocus={() => setIsEditingName(true)}
          onBlur={(e) => { setIsEditingName(false); onConfigurationNameBlur?.(e.target.value); }}
          placeholder="Circuit name..."
          className="w-full px-0 py-1 text-sm bg-transparent border-b border-border text-neutral-200 placeholder-neutral-700 focus:outline-none focus:border-primary transition-colors"
        />
      )}

      {operationMode === 'compute' ? (
        <div className="space-y-3">
          {/* Circuit Parameters — primary, always first */}
          <CollapsibleSection title="Parameters" defaultOpen={true}>
            <div className="grid grid-cols-2 xl:grid-cols-3 gap-3">
              <EnhancedInput label="Rsh" value={isNaN(circuitParams?.Rsh || 0) ? 0 : (circuitParams?.Rsh || 0)} onChange={(v) => handleCircuitParamChange('Rsh', v)} unit="Ω" min={10} max={10000} step={10} showSlider={true} />
              <EnhancedInput label="Ra" value={isNaN(circuitParams?.Ra || 0) ? 0 : (circuitParams?.Ra || 0)} onChange={(v) => handleCircuitParamChange('Ra', v)} unit="Ω" min={10} max={10000} step={10} showSlider={true} />
              <EnhancedInput label="Ca" value={isNaN((circuitParams?.Ca || 0) * 1e6) ? 0 : (circuitParams?.Ca || 0) * 1e6} onChange={(v) => handleCircuitParamChange('Ca', v / 1e6)} unit="μF" min={0.1} max={50} step={0.1} showSlider={true} />
              <EnhancedInput label="Rb" value={isNaN(circuitParams?.Rb || 0) ? 0 : (circuitParams?.Rb || 0)} onChange={(v) => handleCircuitParamChange('Rb', v)} unit="Ω" min={10} max={10000} step={100} showSlider={true} />
              <EnhancedInput label="Cb" value={isNaN((circuitParams?.Cb || 0) * 1e6) ? 0 : (circuitParams?.Cb || 0) * 1e6} onChange={(v) => handleCircuitParamChange('Cb', v / 1e6)} unit="μF" min={0.1} max={50} step={0.1} showSlider={true} />
            </div>
          </CollapsibleSection>

          {/* Frequency */}
          <CollapsibleSection title="Frequency" defaultOpen={true}>
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-[10px] text-neutral-600 mb-1">Min (Hz)</label>
                  <input
                    type="text"
                    value={minFreq === 0 ? '' : minFreq.toString()}
                    onChange={(e) => {
                      const value = e.target.value.trim();
                      if (value === '') { onMinFreqChange(0); return; }
                      const numValue = parseFloat(value);
                      if (!isNaN(numValue) && numValue >= 0) onMinFreqChange(numValue);
                    }}
                    className="w-full px-2 py-1.5 text-xs bg-neutral-800 border border-neutral-800 rounded text-neutral-200 focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary transition-colors"
                    placeholder="0.1"
                  />
                </div>
                <div>
                  <label className="block text-[10px] text-neutral-600 mb-1">Max (Hz)</label>
                  <input
                    type="text"
                    value={maxFreq === 0 ? '' : maxFreq.toString()}
                    onChange={(e) => {
                      const value = e.target.value.trim();
                      if (value === '') { onMaxFreqChange(0); return; }
                      const numValue = parseFloat(value);
                      if (!isNaN(numValue) && numValue >= 0) onMaxFreqChange(numValue);
                    }}
                    className="w-full px-2 py-1.5 text-xs bg-neutral-800 border border-neutral-800 rounded text-neutral-200 focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary transition-colors"
                    placeholder="1e5"
                  />
                </div>
              </div>
              {(minFreq >= maxFreq || minFreq <= 0 || maxFreq <= 0) && (
                <p className="text-xs text-red-400">
                  {minFreq >= maxFreq ? 'Min must be less than max.' : 'Frequencies must be positive.'}
                </p>
              )}
              <EnhancedInput label="Points" value={numPoints} onChange={onNumPointsChange} min={10} max={1000} step={1} showSlider={true} unit="" />
            </div>
          </CollapsibleSection>

          {/* Grid */}
          <CollapsibleSection title="Grid" defaultOpen={true}>
            <div className="space-y-3">
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <label className="text-[10px] text-neutral-600">Points per parameter</label>
                  <PointsCalculator gridSize={gridSize} />
                </div>
                <EnhancedInput label="" value={gridSize} onChange={onGridSizeChange} min={2} max={30} step={1} showSlider={true} />
              </div>
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <label className="text-[10px] text-neutral-600">Group portion</label>
                  <span className="text-[10px] text-neutral-600 font-mono">{shownModels.toLocaleString()} shown</span>
                </div>
                <EnhancedInput label="" value={parseFloat(groupPortionSampler.formatValue(currentGroupPercentage))} onChange={handleGroupPortionPercentageChange} min={0.01} max={100} step={groupPortionSampler.getStepSize(currentGroupPercentage)} showSlider={true} />
              </div>
            </div>
          </CollapsibleSection>

          {/* Display — secondary, collapsed by default */}
          <CollapsibleSection title="Display" defaultOpen={false}>
            <select
              value={staticRenderSettings.visualizationType === 'nyquist' ? 'nyquist' : 'spider3d'}
              onChange={(e) => handleVisualizationTypeChange(e.target.value as 'spider3d' | 'nyquist')}
              className="w-full px-2 py-1.5 text-xs bg-neutral-800 border border-neutral-800 rounded text-neutral-200 focus:outline-none focus:ring-1 focus:ring-primary transition-colors"
            >
              <option value="spider3d">Spider 3D</option>
              <option value="nyquist">Nyquist Plot</option>
            </select>
          </CollapsibleSection>

          {/* Validation errors */}
          {!allValid && validationErrors.length > 0 && (
            <div className="border border-border rounded p-2.5 text-xs text-neutral-400 space-y-0.5 bg-neutral-900">
              {validationErrors.map((error, index) => <div key={index}>{error}</div>)}
            </div>
          )}

          {/* Run controls */}
          <div className="space-y-2 pt-1">
            <button
              onClick={onCompute}
              disabled={isComputing || !allValid}
              className="w-full px-4 py-2 text-xs bg-primary hover:bg-primary-dark disabled:bg-neutral-800 disabled:text-neutral-600 text-white font-medium rounded transition-colors"
            >
              {isComputing ? 'Computing...' : 'Run'}
            </button>
            <div className="flex gap-2">
              {selectedProfileId && (
                <button
                  onClick={() => onSaveProfile(configurationName.trim() || 'Untitled', 'Parameters updated', false)}
                  disabled={isComputing || !allValid}
                  className="flex-1 px-3 py-1.5 text-xs bg-neutral-800 hover:bg-neutral-700 disabled:opacity-40 text-neutral-400 rounded border border-border transition-colors"
                >
                  Update
                </button>
              )}
              <button
                onClick={() => onSaveProfile(configurationName.trim() || 'Untitled', 'New profile saved', true)}
                disabled={isComputing || !allValid}
                className="flex-1 px-3 py-1.5 text-xs bg-neutral-800 hover:bg-neutral-700 disabled:opacity-40 text-neutral-400 rounded border border-border transition-colors"
              >
                {selectedProfileId ? 'Save as new' : 'Save'}
              </button>
            </div>
            {onSRDUploaded && (
              <button
                onClick={() => setOperationMode('upload')}
                className="w-full text-center text-[10px] text-neutral-600 hover:text-neutral-400 transition-colors pt-1"
              >
                Upload SRD data instead
              </button>
            )}
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          {onSRDUploaded && onUploadError ? (
            <SRDUploadInterface
              onSRDUploaded={(manager, metadata) => {
                onGridSizeChange(metadata.gridSize);
                if (onConfigurationNameChange && !localConfigName) {
                  const newName = `${metadata.title} (${new Date().toLocaleDateString()})`;
                  setLocalConfigName(newName);
                  onConfigurationNameChange(newName);
                }
                onSRDUploaded(manager, metadata);
              }}
              onError={onUploadError}
              className="w-full"
            />
          ) : (
            <p className="text-xs text-neutral-500 text-center py-4">Upload not configured.</p>
          )}
          <button
            onClick={() => setOperationMode('compute')}
            className="w-full text-center text-[10px] text-neutral-600 hover:text-neutral-400 transition-colors"
          >
            Back to compute
          </button>
        </div>
      )}
    </div>
  );
};