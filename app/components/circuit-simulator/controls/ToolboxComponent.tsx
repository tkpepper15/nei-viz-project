import React, { useState } from 'react';
import { CircuitParameters } from '../../circuit-simulator/types/parameters';
import { ParamSlider } from '../ParamSlider';
import { BackendMeshPoint, ModelSnapshot } from '../../circuit-simulator/utils/types';
import { generateLogSpace } from '../../circuit-simulator/utils/parameter-space';

import { SystemMonitor } from './SystemMonitor';

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
  gridResults: BackendMeshPoint[];
  
  // Circuit parameters
  groundTruthParams: CircuitParameters;
  setGroundTruthParams: (params: CircuitParameters | ((prev: CircuitParameters) => CircuitParameters)) => void;
  referenceModelId: string | null;
  createReferenceModel: () => ModelSnapshot;
  setReferenceModel: (model: ModelSnapshot | null) => void;
  

}

// CollapsibleSection component definition
interface CollapsibleSectionProps {
  title: string;
  isOpen: boolean;
  toggleOpen: () => void;
  children: React.ReactNode;
}

const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({ 
  title, 
  isOpen, 
  toggleOpen, 
  children 
}) => {
  return (
    <div className="border border-neutral-700 rounded overflow-hidden">
      <button 
        className="w-full p-3 bg-neutral-800 text-neutral-200 text-sm font-medium flex items-center justify-between hover:bg-neutral-600"
        onClick={toggleOpen}
      >
        <span>{title}</span>
        <svg 
          className={`w-4 h-4 transition-transform ${isOpen ? 'transform rotate-180' : ''}`} 
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      <div className={`p-3 bg-neutral-900 transition-all ${isOpen ? 'block' : 'hidden'}`}>
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
  gridResults,
  
  // Circuit parameters props
  groundTruthParams,
  setGroundTruthParams,
  referenceModelId,
  createReferenceModel,
  setReferenceModel,

}) => {
  // Local state for collapsible sections and tab management
  const [gridSettingsOpen, setGridSettingsOpen] = useState(true);
  const [circuitParamsOpen, setCircuitParamsOpen] = useState(false);

  return (
    <div className="circuit-sidebar h-full flex flex-col">
      {/* Toolbox Header */}
      <div className="bg-neutral-800 border-b border-neutral-700 p-3 flex-shrink-0">
        <h3 className="text-sm font-medium text-neutral-200">Toolbox</h3>
      </div>
          
      {/* Sidebar Content - scrollable */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-3 space-y-3">
          {/* Grid Computation Section */}
          <CollapsibleSection 
            title="Grid Computation" 
            isOpen={gridSettingsOpen} 
            toggleOpen={() => setGridSettingsOpen(!gridSettingsOpen)}
          >
            <div className="space-y-3 pt-1">
              <div className="flex items-center justify-between gap-2">
                <label htmlFor="gridSize" className="text-sm font-medium text-neutral-200 flex-1">
                  Points per parameter:
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
                        className="absolute h-2 bg-blue-500 rounded-full" 
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
                      className="absolute top-0 w-4 h-4 bg-blue-500 border border-neutral-600 rounded-full shadow-lg -translate-y-1 -translate-x-2 pointer-events-none"
                      style={{
                        left: `${((Math.log10(minFreq) - Math.log10(0.01)) / (Math.log10(10000) - Math.log10(0.01))) * 100}%`
                      }}
                    />
                    <div 
                      className="absolute top-0 w-4 h-4 bg-blue-500 border border-neutral-600 rounded-full shadow-lg -translate-y-1 -translate-x-2 pointer-events-none"
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

              <button
                onClick={handleComputeRegressionMesh}
                disabled={isComputingGrid}
                className={`w-full mt-2 py-2.5 rounded-lg font-medium text-white transition-colors ${
                  isComputingGrid
                    ? 'bg-gray-600 cursor-not-allowed'
                    : parameterChanged 
                      ? 'bg-amber-600 hover:bg-amber-700' 
                      : 'bg-blue-600 hover:bg-blue-700'
                }`}
              >
                {isComputingGrid ? (
                  <span className="flex items-center justify-center">
                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Computing...
                  </span>
                ) : (parameterChanged ? 'Recompute Grid' : 'Compute Grid')}
              </button>
            </div>
          </CollapsibleSection>

          {/* Enhanced Performance & System Monitor */}
          <SystemMonitor
            gridSize={gridSize}
            totalGridPoints={Math.pow(gridSize, 5)}
            computedGridPoints={gridResults.length}
            onGridFilterChanged={(settings) => {
              updateStatusMessage(`Grid filtering updated: ${settings.enableSmartFiltering ? `${settings.visibilityPercentage}% visible` : 'Disabled'}`);
            }}
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
                  <span className="text-xs text-neutral-400">{gridSize} points</span>
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
                  readOnlyRange={true}
                />
              </div>

              {/* Ra Parameter */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-neutral-200">Ra (Ω)</span>
                  <span className="text-xs text-neutral-400">{gridSize} points</span>
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
                  readOnlyRange={true}
                />
              </div>

              {/* Ca Parameter */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-neutral-200">Ca (μF)</span>
                  <span className="text-xs text-neutral-400">{gridSize} points</span>
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
                  readOnlyRange={true}
                />
              </div>

              {/* Rb Parameter */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-neutral-200">Rb (Ω)</span>
                  <span className="text-xs text-neutral-400">{gridSize} points</span>
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
                  readOnlyRange={true}
                />
              </div>

              {/* Cb Parameter */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-neutral-200">Cb (μF)</span>
                  <span className="text-xs text-neutral-400">{gridSize} points</span>
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
                  readOnlyRange={true}
                />
              </div>
            </div>
          </CollapsibleSection>


        </div>
      </div>
    </div>
  );
}; 