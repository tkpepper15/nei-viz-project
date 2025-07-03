import React, { useState } from 'react';
import { CircuitParameters } from '../../circuit-simulator/types/parameters';
import { ParamSlider } from '../ParamSlider';
import { ModelSnapshot } from '../../circuit-simulator/utils/types';
import { generateLogSpace } from '../../circuit-simulator/utils/parameter-space';

import { SaveProfileModal } from './SaveProfileModal';
import { PerformanceControls, PerformanceSettings } from './PerformanceControls';

// Props interface for the ToolboxComponent
interface ToolboxComponentProps {
  // Grid computation
  gridSize: number;
  setGridSize: (size: number) => void;
  minFreq: number;
  setMinFreq: (freq: number) => void;
  maxFreq: number;
  setMaxFreq: (freq: number) => void;
  numPoints: number;
  setNumPoints: (points: number) => void;
  updateFrequencies: (min: number, max: number, points: number) => void;
  updateStatusMessage: (message: string) => void;
  parameterChanged: boolean;
  setParameterChanged: (changed: boolean) => void;
  handleComputeRegressionMesh: () => void;
  isComputingGrid: boolean;
  onClearResults: () => void;
  hasGridResults: boolean;
  
  // Circuit parameters
  groundTruthParams: CircuitParameters;
  setGroundTruthParams: (params: CircuitParameters | ((prev: CircuitParameters) => CircuitParameters)) => void;
  referenceModelId: string | null;
  createReferenceModel: () => ModelSnapshot;
  setReferenceModel: (model: ModelSnapshot | null) => void;
  
  // Save profile functionality
  onSaveProfile: (name: string, description?: string) => void;
  
  // Performance settings
  performanceSettings: PerformanceSettings;
  setPerformanceSettings: (settings: PerformanceSettings) => void;
}

// CollapsibleSection component definition
interface CollapsibleSectionProps {
  title: string;
  isOpen: boolean;
  toggleOpen: () => void;
  children: React.ReactNode;
  isFirst?: boolean;
}

const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({ 
  title, 
  isOpen, 
  toggleOpen, 
  children,
  isFirst = false
}) => {
  return (
    <div className="border-b border-neutral-700 last:border-b-0">
      <button 
        className="w-full px-4 py-3 bg-neutral-800 text-neutral-300 text-sm font-bold flex items-center justify-between hover:bg-neutral-750 transition-colors"
        onClick={toggleOpen}
        style={isFirst ? { borderRadius: '0px 0px 0px 0px' } : {}}
      >
        <span className="font-bold">{title}</span>
        <svg 
          className={`w-4 h-4 transition-transform ${isOpen ? 'transform rotate-180' : ''}`} 
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      <div className={`px-4 py-3 bg-neutral-800 transition-all ${isOpen ? 'block' : 'hidden'}`}>
        {children}
      </div>
    </div>
  );
};

const formatTickValue = (v: number, unit: string, digits = 2) =>
  unit === 'Ω' ? Number(v).toPrecision(3) : Number(v).toFixed(digits);

export const ToolboxComponent: React.FC<ToolboxComponentProps> = ({
  // Grid computation props
  gridSize,
  setGridSize,
  minFreq,
  setMinFreq,
  maxFreq,
  setMaxFreq,
  numPoints,
  setNumPoints,
  updateFrequencies,
  updateStatusMessage,
  parameterChanged,
  setParameterChanged,
  handleComputeRegressionMesh,
  isComputingGrid,
  onClearResults,
  hasGridResults,
  
  // Circuit parameters props
  groundTruthParams,
  setGroundTruthParams,
  referenceModelId,
  createReferenceModel,
  setReferenceModel,
  
  // Save profile props
  onSaveProfile,
  
  // Performance settings props
  performanceSettings,
  setPerformanceSettings,
}) => {
  // Local state for collapsible sections and tab management
  const [gridSettingsOpen, setGridSettingsOpen] = useState(true);
  const [circuitParamsOpen, setCircuitParamsOpen] = useState(false);
  const [saveModalOpen, setSaveModalOpen] = useState(false);

  return (
    <div className="w-full">
      <div className="space-y-0">
        {/* Grid Computation Section */}
        <CollapsibleSection 
          title="Grid Computation" 
          isOpen={gridSettingsOpen} 
          toggleOpen={() => setGridSettingsOpen(!gridSettingsOpen)}
          isFirst={true}
        >
          <div className="space-y-3 pt-1">
            <div className="flex items-center justify-between gap-2">
              <label htmlFor="gridSize" className="text-sm font-medium text-neutral-200 flex-1">
                Points per parameter
              </label>
              <input
                type="number"
                id="gridSize"
                value={gridSize}
                onChange={(e) => {
                  const size = Math.max(2, Math.min(25, parseInt(e.target.value) || 2));
                  setGridSize(size);
                  const totalPoints = Math.pow(size, 5);
                  const warningMessage = size > 15 ? ` - This will take significant computation time!` : '';
                  updateStatusMessage(`Grid size set to ${size} (${totalPoints.toLocaleString()} total points to compute)${warningMessage}`);
                }} 
                min="2"
                max="25"
                className="w-14 p-1 border border-neutral-700 rounded text-xs text-center bg-neutral-800 text-neutral-200 flex-shrink-0"
              />
            </div>

            {/* Frequency Range Controls */}
            <div className="space-y-3">
              {/* Frequency Range Label */}
              <div className="text-sm font-medium text-neutral-200">Frequency Range</div>

              {/* Dual Range Slider */}
              <div className="flex items-center gap-3">
                <div className="text-xs text-neutral-400">0.01</div>
                
                <div className="flex-1 relative">
                  {/* Slider Track */}
                  <div className="relative h-2 bg-neutral-700 rounded-full">
                                          {/* Active range between thumbs */}
                      <div 
                        className="absolute h-2 bg-neutral-500 rounded-full" 
                        style={{ 
                          left: `${((Math.log10(minFreq) - Math.log10(0.01)) / (Math.log10(10000) - Math.log10(0.01))) * 100}%`,
                          width: `${((Math.log10(maxFreq) - Math.log10(minFreq)) / (Math.log10(10000) - Math.log10(0.01))) * 100}%`
                        }}
                      />
                  </div>
                  
                  {/* Min Frequency Slider */}
                  <input 
                    type="range" 
                    min={0} 
                    max={100} 
                    step={1}
                    value={((Math.log10(minFreq) - Math.log10(0.01)) / (Math.log10(10000) - Math.log10(0.01))) * 100}
                    onChange={(e) => {
                      const sliderPos = parseFloat(e.target.value);
                      const logMin = Math.log10(0.01);
                      const logMax = Math.log10(10000);
                      const logValue = logMin + (sliderPos / 100) * (logMax - logMin);
                      const value = Math.pow(10, logValue);
                      // Refined smoothing with better precision control
                      const roundedValue = value < 0.1 ? Math.round(value * 1000) / 1000 : 
                                         value < 1 ? Math.round(value * 100) / 100 : 
                                         value < 10 ? Math.round(value * 10) / 10 : 
                                         value < 100 ? Math.round(value) :
                                         Math.round(value / 5) * 5;
                      const clampedValue = Math.max(0.01, Math.min(maxFreq - 0.01, roundedValue));
                      setMinFreq(clampedValue);
                      updateFrequencies(clampedValue, maxFreq, numPoints);
                      setParameterChanged(true);
                    }}
                    className="absolute top-0 h-2 opacity-0 cursor-pointer z-30"
                    style={{ 
                      left: '0%',
                      width: '65%'
                    }}
                  />
                  
                  {/* Max Frequency Slider */}
                  <input 
                    type="range" 
                    min={0} 
                    max={100} 
                    step={1}
                    value={((Math.log10(maxFreq) - Math.log10(0.01)) / (Math.log10(10000) - Math.log10(0.01))) * 100}
                    onChange={(e) => {
                      const sliderPos = parseFloat(e.target.value);
                      const logMin = Math.log10(0.01);
                      const logMax = Math.log10(10000);
                      const logValue = logMin + (sliderPos / 100) * (logMax - logMin);
                      const value = Math.pow(10, logValue);
                      // Refined smoothing with better precision control
                      const roundedValue = value < 1 ? Math.round(value * 100) / 100 : 
                                         value < 10 ? Math.round(value * 10) / 10 : 
                                         value < 100 ? Math.round(value) :
                                         value < 1000 ? Math.round(value / 5) * 5 :
                                         Math.round(value / 50) * 50;
                      const clampedValue = Math.max(minFreq + 0.01, Math.min(10000, roundedValue));
                      setMaxFreq(clampedValue);
                      updateFrequencies(minFreq, clampedValue, numPoints);
                      setParameterChanged(true);
                    }}
                    className="absolute top-0 h-2 opacity-0 cursor-pointer z-20"
                    style={{ 
                      right: '0%',
                      width: '65%'
                    }}
                  />
                  
                                      {/* Visual thumb indicators */}
                    <div 
                      className="absolute top-0 w-4 h-4 bg-neutral-500 border border-neutral-600 rounded-full shadow-lg -translate-y-1 -translate-x-2 pointer-events-none"
                      style={{
                        left: `${((Math.log10(minFreq) - Math.log10(0.01)) / (Math.log10(10000) - Math.log10(0.01))) * 100}%`
                      }}
                    />
                    <div 
                      className="absolute top-0 w-4 h-4 bg-neutral-500 border border-neutral-600 rounded-full shadow-lg -translate-y-1 -translate-x-2 pointer-events-none"
                      style={{
                        left: `${((Math.log10(maxFreq) - Math.log10(0.01)) / (Math.log10(10000) - Math.log10(0.01))) * 100}%`
                      }}
                    />
                </div>

                <div className="text-xs text-neutral-400">10k</div>
              </div>

              {/* Manual Input Fields */}
              <div className="flex gap-3">
                <div className="flex-1">
                  <div className="text-xs text-neutral-400 mb-1">Min (Hz)</div>
                  <input
                    type="number"
                    value={minFreq.toFixed(minFreq < 1 ? 2 : minFreq < 10 ? 1 : 0)}
                    onChange={(e) => {
                      const value = parseFloat(e.target.value);
                      if (!isNaN(value)) {
                        const clampedValue = Math.max(0.01, Math.min(maxFreq - 0.01, value));
                        setMinFreq(clampedValue);
                        updateFrequencies(clampedValue, maxFreq, numPoints);
                        setParameterChanged(true);
                      }
                    }}
                    className="w-full bg-neutral-700 text-white text-xs px-2 py-1 rounded border border-neutral-600 focus:border-blue-500 focus:outline-none text-center"
                  />
                </div>
                <div className="flex-1">
                  <div className="text-xs text-neutral-400 mb-1">Max (Hz)</div>
                  <input
                    type="number"
                    value={maxFreq.toFixed(maxFreq < 10 ? 1 : 0)}
                    onChange={(e) => {
                      const value = parseFloat(e.target.value);
                      if (!isNaN(value)) {
                        const clampedValue = Math.max(minFreq + 0.01, Math.min(10000, value));
                        setMaxFreq(clampedValue);
                        updateFrequencies(minFreq, clampedValue, numPoints);
                        setParameterChanged(true);
                      }
                    }}
                    className="w-full bg-neutral-700 text-white text-xs px-2 py-1 rounded border border-neutral-600 focus:border-blue-500 focus:outline-none text-center"
                  />
                </div>
                <div className="flex-1">
                  <div className="text-xs text-neutral-400 mb-1">Points</div>
                  <input
                    type="number"
                    value={numPoints}
                    onChange={(e) => {
                      const points = Math.max(10, Math.min(200, parseInt(e.target.value) || 10));
                      setNumPoints(points);
                      updateFrequencies(minFreq, maxFreq, points);
                      setParameterChanged(true);
                      updateStatusMessage(`Frequency points set to ${points}`);
                    }}
                    min="10"
                    max="200"
                    className="w-full bg-neutral-700 text-white text-xs px-2 py-1 rounded border border-neutral-600 focus:border-blue-500 focus:outline-none text-center"
                  />
                </div>
              </div>
            </div>

            {/* Save/Compute Button Group */}
            <div className="flex gap-2 mt-3">
              <button
                onClick={() => setSaveModalOpen(true)}
                disabled={isComputingGrid}
                className={`flex-1 py-2.5 rounded-lg font-medium text-white transition-colors ${
                  isComputingGrid
                    ? 'bg-neutral-600 cursor-not-allowed'
                    : 'bg-green-600 hover:bg-green-700'
                }`}
                title="Save current parameters and settings as a profile"
              >
                <span className="flex items-center justify-center">
                  <svg className="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  Save Profile
                </span>
              </button>
              
              <button
                onClick={handleComputeRegressionMesh}
                disabled={isComputingGrid}
                className={`flex-1 py-2.5 rounded-lg font-medium text-white transition-colors ${
                  isComputingGrid
                    ? 'bg-neutral-600 cursor-not-allowed'
                    : parameterChanged 
                      ? 'bg-blue-600 hover:bg-blue-700' 
                      : 'bg-blue-600 hover:bg-blue-700'
                }`}
                title="Compute grid and display results in playground"
              >
                {isComputingGrid ? (
                  <span className="flex items-center justify-center">
                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Computing...
                  </span>
                ) : (
                  <span className="flex items-center justify-center">
                    <svg className="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    {parameterChanged ? 'Recompute' : 'Compute'}
                  </span>
                )}
              </button>
            </div>
            
            {/* Clear Memory Button */}
            {hasGridResults && (
              <div className="mt-2">
                <button
                  onClick={onClearResults}
                  disabled={isComputingGrid}
                  className="w-full py-2 px-3 rounded-md text-red-300 border border-red-600/50 hover:bg-red-600/10 hover:border-red-500 disabled:opacity-50 disabled:cursor-not-allowed font-medium transition-all duration-200 text-xs"
                  title="Clear grid results from memory to free up space"
                >
                  <span className="flex items-center justify-center gap-2">
                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                    Clear Results from Memory
                  </span>
                </button>
              </div>
            )}
          </div>
        </CollapsibleSection>

        {/* Performance Controls Section */}
        <PerformanceControls
          settings={performanceSettings}
          onChange={setPerformanceSettings}
          gridSize={gridSize}
        />

        {/* Circuit Parameters Section */}
        <CollapsibleSection 
          title="Circuit Parameters" 
          isOpen={circuitParamsOpen} 
          toggleOpen={() => setCircuitParamsOpen(!circuitParamsOpen)}
        >
          <div className="space-y-4 pt-2">
            {/* Rs Parameter */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-neutral-200">Rs (Ω)</span>
                <input
                  type="number"
                  value={groundTruthParams.Rs.toFixed(1)}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    if (!isNaN(val) && val >= 10 && val <= 10000) {
                      setGroundTruthParams(prev => ({ ...prev, Rs: val }));
                      updateStatusMessage(`Rs set to ${val.toFixed(1)} Ω`);
                      if (referenceModelId === 'dynamic-reference') {
                        const updatedModel = createReferenceModel();
                        setReferenceModel(updatedModel);
                      }
                    }
                  }}
                  className="w-16 bg-neutral-700 text-white text-xs px-2 py-1 rounded border border-neutral-600 focus:border-blue-500 focus:outline-none text-center"
                  step="10"
                  min="10"
                  max="10000"
                />
              </div>
              <ParamSlider 
                label="" 
                value={groundTruthParams.Rs} 
                min={10} 
                max={10000} 
                step={10}
                unit="Ω" 
                onChange={(val: number) => {
                  setGroundTruthParams(prev => ({ ...prev, Rs: val }));
                  updateStatusMessage(`Rs set to ${val.toFixed(1)} Ω`);
                  if (referenceModelId === 'dynamic-reference') {
                    const updatedModel = createReferenceModel();
                    setReferenceModel(updatedModel);
                  }
                }} 
                ticks={generateLogSpace(10, 10000, gridSize).map((v, i) => ({ level: i + 1, value: v }))}
                log={true}
                tickLabels={generateLogSpace(10, 10000, gridSize).map(v => formatTickValue(v, 'Ω'))}
                readOnlyRange={false}
              />
            </div>

            {/* Ra Parameter */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-neutral-200">Ra (Ω)</span>
                <input
                  type="number"
                  value={groundTruthParams.Ra.toFixed(0)}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    if (!isNaN(val) && val >= 10 && val <= 10000) {
                      setGroundTruthParams(prev => ({ ...prev, Ra: val }));
                      updateStatusMessage(`Ra set to ${val.toFixed(0)} Ω`);
                      if (referenceModelId === 'dynamic-reference') {
                        const updatedModel = createReferenceModel();
                        setReferenceModel(updatedModel);
                      }
                    }
                  }}
                  className="w-16 bg-neutral-700 text-white text-xs px-2 py-1 rounded border border-neutral-600 focus:border-blue-500 focus:outline-none text-center"
                  step="10"
                  min="10"
                  max="10000"
                />
              </div>
              <ParamSlider 
                label="" 
                value={groundTruthParams.Ra} 
                min={10} 
                max={10000} 
                step={10}
                unit="Ω" 
                onChange={(val: number) => {
                  setGroundTruthParams(prev => ({ ...prev, Ra: val }));
                  updateStatusMessage(`Ra set to ${val.toFixed(0)} Ω`);
                  if (referenceModelId === 'dynamic-reference') {
                    const updatedModel = createReferenceModel();
                    setReferenceModel(updatedModel);
                  }
                }} 
                ticks={generateLogSpace(10, 10000, gridSize).map((v, i) => ({ level: i + 1, value: v }))}
                log={true}
                tickLabels={generateLogSpace(10, 10000, gridSize).map(v => formatTickValue(v, 'Ω'))}
                readOnlyRange={false}
              />
            </div>

            {/* Ca Parameter */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-neutral-200">Ca (μF)</span>
                <input
                  type="number"
                  value={(groundTruthParams.Ca * 1e6).toFixed(2)}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    if (!isNaN(val) && val >= 0.1 && val <= 50) {
                      setGroundTruthParams(prev => ({ ...prev, Ca: val / 1e6 }));
                      updateStatusMessage(`Ca set to ${val.toFixed(2)} μF`);
                      if (referenceModelId === 'dynamic-reference') {
                        const updatedModel = createReferenceModel();
                        setReferenceModel(updatedModel);
                      }
                    }
                  }}
                  className="w-16 bg-neutral-700 text-white text-xs px-2 py-1 rounded border border-neutral-600 focus:border-blue-500 focus:outline-none text-center"
                  step="0.1"
                  min="0.1"
                  max="50"
                />
              </div>
              <ParamSlider 
                label="" 
                value={groundTruthParams.Ca * 1e6} 
                min={0.1} 
                max={50} 
                step={0.1}
                unit="μF" 
                onChange={(val: number) => {
                  setGroundTruthParams(prev => ({ ...prev, Ca: val / 1e6 }));
                  updateStatusMessage(`Ca set to ${val.toFixed(2)} μF`);
                  if (referenceModelId === 'dynamic-reference') {
                    const updatedModel = createReferenceModel();
                    setReferenceModel(updatedModel);
                  }
                }} 
                ticks={generateLogSpace(0.1, 50, gridSize).map((v, i) => ({ level: i + 1, value: v }))}
                transformValue={(v) => v.toFixed(2)}
                log={true}
                tickLabels={generateLogSpace(0.1, 50, gridSize).map(v => formatTickValue(v, 'μF', 2))}
                readOnlyRange={false}
              />
            </div>

            {/* Rb Parameter */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-neutral-200">Rb (Ω)</span>
                <input
                  type="number"
                  value={groundTruthParams.Rb.toFixed(0)}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    if (!isNaN(val) && val >= 10 && val <= 10000) {
                      setGroundTruthParams(prev => ({ ...prev, Rb: val }));
                      updateStatusMessage(`Rb set to ${val.toFixed(0)} Ω`);
                      if (referenceModelId === 'dynamic-reference') {
                        const updatedModel = createReferenceModel();
                        setReferenceModel(updatedModel);
                      }
                    }
                  }}
                  className="w-16 bg-neutral-700 text-white text-xs px-2 py-1 rounded border border-neutral-600 focus:border-blue-500 focus:outline-none text-center"
                  step="10"
                  min="10"
                  max="10000"
                />
              </div>
              <ParamSlider 
                label="" 
                value={groundTruthParams.Rb} 
                min={10} 
                max={10000} 
                step={10}
                unit="Ω" 
                onChange={(val: number) => {
                  setGroundTruthParams(prev => ({ ...prev, Rb: val }));
                  updateStatusMessage(`Rb set to ${val.toFixed(0)} Ω`);
                  if (referenceModelId === 'dynamic-reference') {
                    const updatedModel = createReferenceModel();
                    setReferenceModel(updatedModel);
                  }
                }} 
                ticks={generateLogSpace(10, 10000, gridSize).map((v, i) => ({ level: i + 1, value: v }))}
                log={true}
                tickLabels={generateLogSpace(10, 10000, gridSize).map(v => formatTickValue(v, 'Ω'))}
                readOnlyRange={false}
              />
            </div>

            {/* Cb Parameter */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-neutral-200">Cb (μF)</span>
                <input
                  type="number"
                  value={(groundTruthParams.Cb * 1e6).toFixed(2)}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    if (!isNaN(val) && val >= 0.1 && val <= 50) {
                      setGroundTruthParams(prev => ({ ...prev, Cb: val / 1e6 }));
                      updateStatusMessage(`Cb set to ${val.toFixed(2)} μF`);
                      if (referenceModelId === 'dynamic-reference') {
                        const updatedModel = createReferenceModel();
                        setReferenceModel(updatedModel);
                      }
                    }
                  }}
                  className="w-16 bg-neutral-700 text-white text-xs px-2 py-1 rounded border border-neutral-600 focus:border-blue-500 focus:outline-none text-center"
                  step="0.1"
                  min="0.1"
                  max="50"
                />
              </div>
              <ParamSlider 
                label="" 
                value={groundTruthParams.Cb * 1e6} 
                min={0.1} 
                max={50} 
                step={0.1}
                unit="μF" 
                onChange={(val: number) => {
                  setGroundTruthParams(prev => ({ ...prev, Cb: val / 1e6 }));
                  updateStatusMessage(`Cb set to ${val.toFixed(2)} μF`);
                  if (referenceModelId === 'dynamic-reference') {
                    const updatedModel = createReferenceModel();
                    setReferenceModel(updatedModel);
                  }
                }} 
                ticks={generateLogSpace(0.1, 50, gridSize).map((v, i) => ({ level: i + 1, value: v }))}
                transformValue={(v) => v.toFixed(2)}
                log={true}
                tickLabels={generateLogSpace(0.1, 50, gridSize).map(v => formatTickValue(v, 'μF', 2))}
                readOnlyRange={false}
              />
            </div>
          </div>
        </CollapsibleSection>
      </div>
      
      {/* Save Profile Modal */}
      <SaveProfileModal
        isOpen={saveModalOpen}
        onClose={() => setSaveModalOpen(false)}
        onSave={(name, description) => {
          onSaveProfile(name, description);
          updateStatusMessage(`Profile "${name}" saved successfully`);
        }}
        defaultName={`Grid ${gridSize}x${gridSize}x${gridSize}x${gridSize}x${gridSize}`}
      />
    </div>
  );
}; 