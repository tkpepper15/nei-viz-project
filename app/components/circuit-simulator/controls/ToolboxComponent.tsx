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
  
  // Utility functions
  logToLinearSlider: (logValue: number) => number;
  linearSliderToLog: (sliderPosition: number) => number;
  
  // Log messages
  logMessages: { time: string; message: string }[];
  

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
    <div className="border border-neutral-700 rounded-lg overflow-hidden">
      <button 
        className="w-full p-3 bg-neutral-800 text-neutral-200 text-sm font-medium flex items-center justify-between"
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
  
  // Utility functions
  logToLinearSlider,
  linearSliderToLog,
  
  // Log messages
  logMessages
}) => {
  // Local state for collapsible sections and tab management
  const [currentTab, setCurrentTab] = useState<'toolbox' | 'activity'>('toolbox');
  const [gridSettingsOpen, setGridSettingsOpen] = useState(true);
  const [circuitParamsOpen, setCircuitParamsOpen] = useState(false);

  return (
    <div className="circuit-sidebar">
      {/* Tab Switcher */}
      <div className="circuit-tabs">
        <button
          className={`circuit-tab ${currentTab === 'toolbox' ? 'circuit-tab-active' : 'circuit-tab-inactive'}`}
          onClick={() => setCurrentTab('toolbox')}
        >
          <span>Toolbox</span>
        </button>
        <button
          className={`circuit-tab ${currentTab === 'activity' ? 'circuit-tab-active' : 'circuit-tab-inactive'}`}
          onClick={() => setCurrentTab('activity')}
        >
          <span>Activity Log</span>
        </button>
      </div>
          
      {/* Sidebar Content - scrollable */}
      <div className="flex-1 overflow-y-auto">
        {currentTab === 'toolbox' && (
          <div className="p-4 space-y-4">
            {/* Grid Computation Section */}
            <CollapsibleSection 
              title="Grid Computation" 
              isOpen={gridSettingsOpen} 
              toggleOpen={() => setGridSettingsOpen(!gridSettingsOpen)}
            >
              <div className="space-y-3 pt-1">
                <div className="flex items-center justify-between gap-2">
                  <label htmlFor="gridSize" className="slider-label-text">
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
                    className="w-16 p-1 border border-neutral-700 rounded text-xs text-center bg-neutral-800 text-neutral-200"
                  />
                </div>

                {/* Frequency Range Slider */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="slider-label-text font-medium text-xs text-neutral-300">
                      Frequency range:
                    </label>
                    <span className="text-xs font-mono text-neutral-300">
                      {minFreq < 1 ? minFreq.toFixed(2) + ' Hz' : minFreq < 1000 ? minFreq.toFixed(1) + ' Hz' : (minFreq/1000).toFixed(1) + ' kHz'} - 
                      {maxFreq < 1000 ? maxFreq.toFixed(1) + ' Hz' : (maxFreq/1000).toFixed(1) + ' kHz'}
                    </span>
                  </div>
                  
                  <div className="range_container">
                    <div className="sliders_control relative h-8">
                      {/* Slider track */}
                      <div className="slider-track"></div>
                      <div className="slider-track-active" style={{ 
                        left: `${logToLinearSlider(minFreq)}%`, 
                        width: `${logToLinearSlider(maxFreq) - logToLinearSlider(minFreq)}%` 
                      }}></div>
                      
                      {/* Min frequency slider */}
                      <input 
                        id="fromSlider"
                        type="range" 
                        min={0}
                        max={100}
                        step={0.1}
                        value={logToLinearSlider(minFreq)}
                        onChange={(e) => {
                          // Convert linear slider position to logarithmic value
                          const sliderPos = parseFloat(e.target.value);
                          const logValue = linearSliderToLog(sliderPos);
                          const value = Math.max(0.01, Math.min(maxFreq - 0.01, logValue));
                          setMinFreq(value);
                          updateFrequencies(value, maxFreq, numPoints);
                          setParameterChanged(true);
                        }}
                        onMouseUp={() => {
                          updateStatusMessage(`Frequency range updated to ${minFreq < 1000 ? minFreq.toFixed(2) + ' Hz' : (minFreq/1000).toFixed(2) + ' kHz'} - ${maxFreq < 1000 ? maxFreq.toFixed(1) + ' Hz' : (maxFreq/1000).toFixed(1) + ' kHz'}. Recompute grid to see effect.`);
                        }}
                        className="absolute top-1/2 -translate-y-1/2 appearance-none w-full h-8 bg-transparent z-10"
                      />
                      
                      {/* Max frequency slider */}
                      <input 
                        id="toSlider"
                        type="range" 
                        min={0}
                        max={100}
                        step={0.1}
                        value={logToLinearSlider(maxFreq)}
                        onChange={(e) => {
                          // Convert linear slider position to logarithmic value
                          const sliderPos = parseFloat(e.target.value);
                          const logValue = linearSliderToLog(sliderPos);
                          const value = Math.max(minFreq + 0.01, Math.min(10000, logValue));
                          setMaxFreq(value);
                          updateFrequencies(minFreq, value, numPoints);
                          setParameterChanged(true);
                        }}
                        onMouseUp={() => {
                          updateStatusMessage(`Frequency range updated to ${minFreq < 1000 ? minFreq.toFixed(2) + ' Hz' : (minFreq/1000).toFixed(2) + ' kHz'} - ${maxFreq < 1000 ? maxFreq.toFixed(1) + ' Hz' : (maxFreq/1000).toFixed(1) + ' kHz'}. Recompute grid to see effect.`);
                        }}
                        className="absolute top-1/2 -translate-y-1/2 appearance-none w-full h-8 bg-transparent z-20"
                      />
                      

                    </div>
                    
                    <div className="form_control flex justify-between mt-6">
                      <div className="form_control_container">
                        <div className="form_control_container__time text-[11px] text-neutral-400 mb-1">Min Frequency (Hz)</div>
                        <input 
                          className="form_control_container__time__input w-24 p-1.5 border border-neutral-700 rounded text-xs text-center bg-neutral-800 text-neutral-200" 
                          type="number" 
                          id="fromInput" 
                          value={parseFloat(minFreq.toFixed(2))}
                          min={0.01}
                          max={maxFreq - 0.01}
                          step={0.01}
                          onChange={(e) => {
                            const value = parseFloat(e.target.value);
                            if (!isNaN(value) && value >= 0.01 && value < maxFreq) {
                              setMinFreq(value);
                              updateFrequencies(value, maxFreq, numPoints);
                              setParameterChanged(true);
                            }
                          }}
                        />
                      </div>
                      <div className="form_control_container">
                        <div className="form_control_container__time text-[11px] text-neutral-400 mb-1">Max Frequency (Hz)</div>
                        <input 
                          className="form_control_container__time__input w-24 p-1.5 border border-neutral-700 rounded text-xs text-center bg-neutral-800 text-neutral-200" 
                          type="number" 
                          id="toInput" 
                          value={parseFloat(maxFreq.toFixed(1))}
                          min={minFreq + 0.01}
                          max={10000}
                          step={0.1}
                          onChange={(e) => {
                            const value = parseFloat(e.target.value);
                            if (!isNaN(value) && value > minFreq && value <= 10000) {
                              setMaxFreq(value);
                              updateFrequencies(minFreq, value, numPoints);
                              setParameterChanged(true);
                            }
                          }}
                        />
                      </div>
                    </div>
                  </div>
                  
                  {/* Frequency range impact message */}
                  <div className="text-[11px] text-neutral-400 mt-3 italic pl-1"></div>

                  {/* Frequency Points control */}
                  <div className="flex items-center justify-between">
                    <label className="slider-label-text font-medium text-xs text-neutral-300">
                      Frequency points:
                    </label>
                    <span className="text-xs font-mono text-neutral-300">
                      {numPoints} points
                    </span>
                  </div>
                  <div className="mt-2 relative h-8">
                    <div className="slider-track"></div>
                    <div className="slider-track-active" style={{ 
                      left: '0%', 
                      width: `${(numPoints - 10) / (200 - 10) * 100}%`
                    }}></div>
                    <input 
                      type="range" 
                      min={10} 
                      max={200} 
                      step={1}
                      value={numPoints}
                      onChange={(e) => {
                        const points = parseInt(e.target.value);
                        setNumPoints(points);
                        updateFrequencies(minFreq, maxFreq, points);
                        setParameterChanged(true);
                      }}
                      onMouseUp={() => {
                        updateStatusMessage(`Frequency points set to ${numPoints}. Recompute grid to see effect.`);
                      }}
                      className="absolute top-1/2 -translate-y-1/2 appearance-none w-full h-8 bg-transparent" 
                      style={{
                        zIndex: 10
                      }}
                    />
                    <div className="flex justify-between px-1 mt-4 text-[10px] text-neutral-500">
                      <span>10</span>
                      <span>100</span>
                      <span>200</span>
                    </div>
                  </div>
                  <div className="flex justify-between items-center mt-2">
                    <div className="text-[11px] text-neutral-400">Number of frequency points in range</div>
                    <input 
                      type="number" 
                      min={10} 
                      max={200} 
                      step={1}
                      value={numPoints}
                      onChange={(e) => {
                        const value = parseInt(e.target.value);
                        if (!isNaN(value) && value >= 10 && value <= 200) {
                          setNumPoints(value);
                          updateFrequencies(minFreq, maxFreq, value);
                          setParameterChanged(true);
                        }
                      }}
                      className="w-16 p-1 border border-neutral-700 rounded text-xs text-center bg-neutral-800 text-neutral-200"
                    />
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
        )}
          
        {currentTab === 'activity' && (
          <div className="p-4">
            <div className="text-xs space-y-0 text-neutral-400 max-h-full overflow-y-auto">
              {logMessages.length === 0 ? (
                <p className="italic text-neutral-500 p-3 text-center">No activity yet</p>
              ) : (
                logMessages.map((log, idx) => (
                  <div key={idx} className="py-2 px-2 border-b border-neutral-800 last:border-0">
                    <span className="text-neutral-500 mr-2 text-[10px]">{log.time}</span>
                    <span className={`
                      ${log.message.includes('Error') ? 'text-danger' : ''}
                      ${log.message.includes('Success') || log.message.includes('computed successfully') ? 'text-success' : ''}
                      ${log.message.includes('MATH:') ? 'text-primary font-medium' : ''}
                    `}>{log.message}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}; 