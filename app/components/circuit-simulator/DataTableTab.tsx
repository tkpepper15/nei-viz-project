import React, { useState, useMemo } from 'react';
import { BackendMeshPoint, ResnormGroup, GridParameterArrays } from './types';
import { CircuitParameters } from './types/parameters';

interface DataTableTabProps {
  gridResults: BackendMeshPoint[];
  gridResultsWithIds: (BackendMeshPoint & { id: number })[];
  resnormGroups: ResnormGroup[];
  hiddenGroups: number[];
  maxGridPoints: number;
  gridSize: number;
  parameters: CircuitParameters;
  groundTruthParams: CircuitParameters;
  gridParameterArrays: GridParameterArrays | null;
  opacityExponent: number;
  minFreq: number;
  maxFreq: number;
}

interface ImpedanceCalculation {
  frequency: number;
  Za_real: number;
  Za_imag: number;
  Zb_real: number;
  Zb_imag: number;
  Za_plus_Zb_real: number;
  Za_plus_Zb_imag: number;
  Z_total_real: number;
  Z_total_imag: number;
  magnitude: number;
  phase: number;
}

// Add formatValue function to handle unit conversions
const formatValue = (value: number, isCapacitance: boolean): string => {
  // Safety check for undefined/null/NaN values
  if (typeof value !== 'number' || !isFinite(value) || isNaN(value)) {
    return isCapacitance ? '0.00' : '0.0';
  }
  
  if (isCapacitance) {
    // Convert from Farads to µF and format with 2 decimal places
    return (value * 1e6).toFixed(2);
  }
  // For resistance values, show 1 decimal place for values < 1000, otherwise no decimals
  return value < 1000 ? value.toFixed(1) : value.toFixed(0);
};

// Calculate impedance at a specific frequency for given parameters
const calculateImpedanceAtFrequency = (params: CircuitParameters, frequency: number): ImpedanceCalculation => {
  const { Rsh, Ra, Ca, Rb, Cb } = params;
  const omega = 2 * Math.PI * frequency;
  
  // Za = Ra/(1+jωRaCa)
  const Za_denom_real = 1;
  const Za_denom_imag = omega * Ra * Ca;
  const Za_denom_mag_squared = Za_denom_real * Za_denom_real + Za_denom_imag * Za_denom_imag;
  const Za_real = Ra * Za_denom_real / Za_denom_mag_squared;
  const Za_imag = -Ra * Za_denom_imag / Za_denom_mag_squared;
  
  // Zb = Rb/(1+jωRbCb)
  const Zb_denom_real = 1;
  const Zb_denom_imag = omega * Rb * Cb;
  const Zb_denom_mag_squared = Zb_denom_real * Zb_denom_real + Zb_denom_imag * Zb_denom_imag;
  const Zb_real = Rb * Zb_denom_real / Zb_denom_mag_squared;
  const Zb_imag = -Rb * Zb_denom_imag / Zb_denom_mag_squared;
  
  // Za + Zb
  const Za_plus_Zb_real = Za_real + Zb_real;
  const Za_plus_Zb_imag = Za_imag + Zb_imag;
  
  // Z_total = (Rsh * (Za + Zb)) / (Rsh + Za + Zb)
  const numerator_real = Rsh * Za_plus_Zb_real;
  const numerator_imag = Rsh * Za_plus_Zb_imag;
  
  const denominator_real = Rsh + Za_plus_Zb_real;
  const denominator_imag = Za_plus_Zb_imag;
  const denominator_mag_squared = denominator_real * denominator_real + denominator_imag * denominator_imag;
  
  const Z_total_real = (numerator_real * denominator_real + numerator_imag * denominator_imag) / denominator_mag_squared;
  const Z_total_imag = (numerator_imag * denominator_real - numerator_real * denominator_imag) / denominator_mag_squared;
  
  const magnitude = Math.sqrt(Z_total_real * Z_total_real + Z_total_imag * Z_total_imag);
  const phase = Math.atan2(Z_total_imag, Z_total_real) * (180 / Math.PI);
  
  return {
    frequency,
    Za_real,
    Za_imag,
    Zb_real,
    Zb_imag,
    Za_plus_Zb_real,
    Za_plus_Zb_imag,
    Z_total_real,
    Z_total_imag,
    magnitude,
    phase
  };
};

// This function is available for future use if needed
// const calculateImpedanceSpectrum = (params: CircuitParameters): ImpedancePoint[] => { ... }

export const DataTableTab: React.FC<DataTableTabProps> = ({
  gridResults,
  gridResultsWithIds,
  resnormGroups,
  hiddenGroups,
  maxGridPoints,
  gridSize,
  parameters,
  groundTruthParams,
  gridParameterArrays,
  opacityExponent,
  minFreq,
  maxFreq
}) => {
  // Table sorting state - ALL HOOKS MUST BE CALLED BEFORE ANY EARLY RETURNS
  const [sortColumn, setSortColumn] = useState<string>('resnorm');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [currentPage, setCurrentPage] = useState<number>(1);
  const resultsPerPage = 100;
  
  // Frequency selection and calculation breakdown state
  const [selectedParams, setSelectedParams] = useState<CircuitParameters | null>(null);
  const [selectedFrequency, setSelectedFrequency] = useState<number>(() => {
    // Start with middle frequency on log scale
    return Math.sqrt(minFreq * maxFreq);
  });
  const [showCalculationBreakdown, setShowCalculationBreakdown] = useState<boolean>(false);
  
  // Generate frequency array for slider using the actual simulator frequency range
  const frequencyArray = useMemo(() => {
    const frequencies = [];
    const startFreq = minFreq;
    const endFreq = maxFreq;
    const numPoints = 100;
    
    // Generate logarithmic frequency spacing
    const logStart = Math.log10(startFreq);
    const logEnd = Math.log10(endFreq);
    const step = (logEnd - logStart) / (numPoints - 1);
    
    for (let i = 0; i < numPoints; i++) {
      frequencies.push(Math.pow(10, logStart + i * step));
    }
    
    return frequencies;
  }, [minFreq, maxFreq]);
  
  // Find current frequency index
  const currentFreqIndex = useMemo(() => {
    return frequencyArray.findIndex(f => Math.abs(f - selectedFrequency) === Math.min(...frequencyArray.map(freq => Math.abs(freq - selectedFrequency))));
  }, [frequencyArray, selectedFrequency]);

  // Ensure hiddenGroups is always number[] for type safety
  const hiddenGroupsNum: number[] = hiddenGroups.map(Number);

  // Function to handle sorting
  const handleSort = (column: string) => {
    const isAsc = sortColumn === column && sortDirection === 'asc';
    setSortDirection(isAsc ? 'desc' : 'asc');
    setSortColumn(column);
  };

  // Memoized function to get sorted grid results
  const getSortedGridResults = useMemo(() => {
    if (gridResultsWithIds.length === 0) return [];
    
    const sorted = [...gridResultsWithIds].sort((a, b) => {
      let aValue, bValue;
      
      // Extract the values for comparison based on the sort column
      switch (sortColumn) {
        case 'id':
          aValue = a.id;
          bValue = b.id;
          break;
        case 'Rsh':
          aValue = a.parameters.Rsh;
          bValue = b.parameters.Rsh;
          break;
        case 'Ra':
          aValue = a.parameters.Ra;
          bValue = b.parameters.Ra;
          break;
        case 'Ca':
          aValue = a.parameters.Ca;
          bValue = b.parameters.Ca;
          break;
        case 'Rb':
          aValue = a.parameters.Rb;
          bValue = b.parameters.Rb;
          break;
        case 'Cb':
          aValue = a.parameters.Cb;
          bValue = b.parameters.Cb;
          break;
        case 'resnorm':
        default:
          aValue = a.resnorm || 0;
          bValue = b.resnorm || 0;
          break;
      }
      
      // Apply the sort direction
      if (sortDirection === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });

    // Apply maxGridPoints limit - if maxGridPoints equals total points, show all
    const totalPoints = Math.pow(gridSize, 5);
    if (maxGridPoints >= totalPoints) {
      return sorted; // Show all points
    }
    return sorted.slice(0, Math.min(maxGridPoints, sorted.length));
  }, [gridResultsWithIds, sortColumn, sortDirection, maxGridPoints, gridSize]);

  // Calculate pagination values
  const displayCount = Math.min(maxGridPoints, gridResults.length);
  const totalPages = Math.ceil(displayCount / resultsPerPage);
  const startIndex = (currentPage - 1) * resultsPerPage;
  const endIndex = Math.min(startIndex + resultsPerPage, displayCount);
  
  // Function to get paginated grid results
  const getPaginatedGridResults = () => {
    return getSortedGridResults.slice(startIndex, endIndex);
  };

  // Function to handle page change
  const handlePageChange = (page: number) => {
    setCurrentPage(page);
    // Scroll table into view
    const tableElement = document.querySelector('.data-table');
    if (tableElement) {
      tableElement.scrollIntoView({ behavior: 'smooth' });
    }
  };

  // Render sorting arrow
  const renderSortArrow = (column: string) => {
    if (sortColumn !== column) {
      return (
        <svg className="w-3 h-3 text-neutral-500 opacity-30" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
        </svg>
      );
    }
    
    return sortDirection === 'asc' ? (
      <svg className="w-3 h-3 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
      </svg>
    ) : (
      <svg className="w-3 h-3 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
      </svg>
    );
  };

  // Calculate dynamic resnorm percentiles from currently sorted data
  const resnormPercentiles = useMemo(() => {
    const sortedData = getSortedGridResults;
    const resnormValues = sortedData.map((point: BackendMeshPoint & { id: number }) => point.resnorm || 0).filter((r: number) => r > 0);
    
    if (resnormValues.length === 0) {
      return { p25: 0, p50: 0, p75: 0, p90: 0 };
    }
    
    // Sort resnorm values for percentile calculation
    const sortedResnorms = [...resnormValues].sort((a, b) => a - b);
    
    const calculatePercentile = (arr: number[], percentile: number): number => {
      const index = Math.ceil((percentile / 100) * arr.length) - 1;
      return arr[Math.max(0, Math.min(index, arr.length - 1))];
    };
    
    return {
      p25: calculatePercentile(sortedResnorms, 25),  // Top 25% (best fits)
      p50: calculatePercentile(sortedResnorms, 50),  // Top 50% (good fits)
      p75: calculatePercentile(sortedResnorms, 75),  // Top 75% (moderate fits)
      p90: calculatePercentile(sortedResnorms, 90)   // Top 90% (acceptable fits)
    };
  }, [getSortedGridResults]);

  // Get quality label and color based on resnorm value and dynamic percentiles
  const getQualityInfo = (point: BackendMeshPoint & { id: number }): { qualityLabel: string; groupColor: string; groupIndex: number; qualityCategory: number } => {
    const resnorm = point.resnorm || 0;
    
    let qualityCategory = 4; // Default to Poor
    let groupColor = '#DC2626'; // Default red
    let qualityLabel = "Poor";
    
    // Categorize based on dynamic percentiles
    if (resnorm <= resnormPercentiles.p25) {
      qualityCategory = 0;
      groupColor = '#059669'; // Emerald-600
      qualityLabel = "Excellent";
    } else if (resnorm <= resnormPercentiles.p50) {
      qualityCategory = 1;
      groupColor = '#10B981'; // Emerald-500
      qualityLabel = "Good";
    } else if (resnorm <= resnormPercentiles.p75) {
      qualityCategory = 2;
      groupColor = '#F59E0B'; // Amber-500
      qualityLabel = "Moderate";
    } else if (resnorm <= resnormPercentiles.p90) {
      qualityCategory = 3;
      groupColor = '#F97316'; // Orange-500
      qualityLabel = "Acceptable";
    } else {
      qualityCategory = 4;
      groupColor = '#DC2626'; // Red-600
      qualityLabel = "Poor";
    }
    
    // Find corresponding group index from original resnorm groups for consistency
    const groupIndex = resnormGroups.findIndex(group => 
      group.color === groupColor || 
      (qualityCategory === 0 && group.label.includes('Excellent')) ||
      (qualityCategory === 1 && group.label.includes('Good')) ||
      (qualityCategory === 2 && group.label.includes('Moderate')) ||
      (qualityCategory === 3 && group.label.includes('Acceptable')) ||
      (qualityCategory === 4 && group.label.includes('Poor'))
    );
    
    return { qualityLabel, groupColor, groupIndex: groupIndex >= 0 ? groupIndex : Number(qualityCategory), qualityCategory };
  };

  // Calculate unique values for each parameter from actual grid arrays
  const uniqueValues = useMemo(() => {
    if (!gridParameterArrays) {
      return {
        Rsh: [],
        Ra: [],
        Rb: [],
        Ca: [],
        Cb: []
      };
    }
    
    return {
      Rsh: [...gridParameterArrays.Rsh].sort((a, b) => a - b),
      Ra: [...gridParameterArrays.Ra].sort((a, b) => a - b),
      Rb: [...gridParameterArrays.Rb].sort((a, b) => a - b),
      Ca: [...gridParameterArrays.Ca].map(v => v * 1e6).sort((a, b) => a - b), // Convert to µF
      Cb: [...gridParameterArrays.Cb].map(v => v * 1e6).sort((a, b) => a - b)  // Convert to µF
    };
  }, [gridParameterArrays]);

  // Early return AFTER all hooks are called - this prevents the React hooks rules violation
  if (!parameters || typeof parameters.Rsh !== 'number' || !minFreq || !maxFreq) {
    return (
      <div className="flex items-center justify-center h-32 bg-neutral-900/50 border border-neutral-700 rounded-lg shadow-md p-4">
        <div className="text-center">
          <p className="text-sm text-red-400">
            Invalid circuit parameters. Please set valid parameter values.
          </p>
        </div>
      </div>
    );
  }

  // Add grid parameter info display
  const renderGridInfo = () => {
    if (!gridParameterArrays) return null;

    const getMinMax = (values: number[]) => {
      if (!values || values.length === 0) return { min: 0, max: 0 };
      const validValues = values.filter(v => typeof v === 'number' && !isNaN(v));
      if (validValues.length === 0) return { min: 0, max: 0 };
      return {
        min: Math.min(...validValues),
        max: Math.max(...validValues)
      };
    };

    const ranges = {
      Rsh: getMinMax(gridParameterArrays.Rsh),
      Ra: getMinMax(gridParameterArrays.Ra),
      Rb: getMinMax(gridParameterArrays.Rb),
      Ca: getMinMax(gridParameterArrays.Ca),
      Cb: getMinMax(gridParameterArrays.Cb)
    };

    return (
      <div className="mb-4 p-3 bg-neutral-800/30 rounded-md">
        <h3 className="text-sm font-medium text-neutral-200 mb-2">Grid Parameter Ranges</h3>
        <div className="grid grid-cols-5 gap-4 text-xs">
          <div>
            <span className="text-neutral-400">R shunt: </span>
            <span className="font-mono text-neutral-300">
              {ranges.Rsh.min.toFixed(1)} - {ranges.Rsh.max.toFixed(1)} Ω
            </span>
          </div>
          <div>
            <span className="text-neutral-400">Ra: </span>
            <span className="font-mono text-neutral-300">
              {ranges.Ra.min.toFixed(1)} - {ranges.Ra.max.toFixed(1)} Ω
            </span>
          </div>
          <div>
            <span className="text-neutral-400">Rb: </span>
            <span className="font-mono text-neutral-300">
              {ranges.Rb.min.toFixed(1)} - {ranges.Rb.max.toFixed(1)} Ω
            </span>
          </div>
          <div>
            <span className="text-neutral-400">Ca: </span>
            <span className="font-mono text-neutral-300">
              {ranges.Ca.min.toFixed(2)} - {ranges.Ca.max.toFixed(2)} µF
            </span>
          </div>
          <div>
            <span className="text-neutral-400">Cb: </span>
            <span className="font-mono text-neutral-300">
              {ranges.Cb.min.toFixed(2)} - {ranges.Cb.max.toFixed(2)} µF
            </span>
          </div>
        </div>
      </div>
    );
  };

  // Find the group for this point
  const getGroupForPoint = (point: BackendMeshPoint & { id: number }) => {
    // Find which group this point belongs to based on resnorm range
    const groupIndex = resnormGroups.findIndex(group => {
      const resnorm = point.resnorm || 0;
      return resnorm >= group.range[0] && resnorm <= group.range[1];
    });
    
    return groupIndex >= 0 ? resnormGroups[groupIndex] : null;
  };

  return (
    <div className="space-y-4">
      {/* Frequency Selector */}
      <div className="bg-blue-900/20 rounded-lg p-4 border border-blue-700/30">
        <h3 className="text-sm font-medium text-blue-200 mb-3 flex items-center">
          <svg className="w-4 h-4 mr-2 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
          Selected Circuit Impedance @ Frequency
        </h3>
        
        <div className="space-y-4">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <label className="text-sm text-blue-200">Frequency:</label>
              <span className="font-mono text-blue-100 text-sm min-w-[80px]">
                {selectedFrequency < 1000 ? 
                  `${selectedFrequency.toFixed(1)} Hz` : 
                  `${(selectedFrequency/1000).toFixed(2)} kHz`
                }
              </span>
            </div>
            
            {/* Quick frequency presets */}
            <div className="flex items-center gap-1">
              <span className="text-xs text-blue-300/70">Quick:</span>
              {[1, 10, 100, 1000, 10000].filter(f => f >= minFreq && f <= maxFreq).map(freq => (
                <button
                  key={freq}
                  onClick={() => setSelectedFrequency(freq)}
                  className={`text-xs px-2 py-1 rounded transition-colors ${
                    Math.abs(selectedFrequency - freq) < 1 
                      ? 'bg-blue-600 text-blue-100' 
                      : 'bg-blue-800/40 hover:bg-blue-700/60 text-blue-200'
                  }`}
                >
                  {freq >= 1000 ? `${freq/1000}k` : `${freq}`}
                </button>
              ))}
            </div>
            
            <div className="text-xs text-blue-300/70">
              Real and imaginary impedance values calculated at this frequency
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs text-blue-300">
              <span>{minFreq < 1 ? `${minFreq} Hz` : `${minFreq.toFixed(0)} Hz`}</span>
              <div className="flex-1 relative">
                <input
                  type="range"
                  min={0}
                  max={frequencyArray.length - 1}
                  value={isNaN(currentFreqIndex) ? 0 : currentFreqIndex}
                  onChange={(e) => {
                    const index = parseInt(e.target.value);
                    setSelectedFrequency(frequencyArray[index]);
                  }}
                  className="w-full h-2 bg-neutral-700 rounded-lg appearance-none cursor-pointer"
                  style={{
                    background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${(currentFreqIndex / (frequencyArray.length - 1)) * 100}%, #374151 ${(currentFreqIndex / (frequencyArray.length - 1)) * 100}%, #374151 100%)`,
                    WebkitAppearance: 'none',
                    MozAppearance: 'none',
                  }}
                />
              </div>
              <span>{maxFreq >= 1000 ? `${(maxFreq/1000).toFixed(0)} kHz` : `${maxFreq.toFixed(0)} Hz`}</span>
            </div>
            <div className="text-xs text-blue-400">
              100 frequency points available • Position {currentFreqIndex + 1}/100 • Range: {minFreq.toFixed(1)} - {maxFreq.toLocaleString()} Hz
            </div>
          </div>
          
          {/* Show reference impedance calculation at selected frequency */}
          <div className="mt-4 p-3 bg-blue-800/10 rounded border border-blue-600/20">
            <div className="text-xs text-blue-200 mb-2 font-medium">Reference Model Impedance:</div>
            <div className="grid grid-cols-3 gap-4 text-xs">
              {(() => {
                const refParams = groundTruthParams || parameters;
                
                // Safety check to prevent errors with undefined parameters
                if (!refParams || !refParams.Rsh || selectedFrequency === undefined || selectedFrequency === null) {
                  return <div className="text-red-400 text-sm">Invalid parameters or frequency</div>;
                }
                
                const refCalc = calculateImpedanceAtFrequency(refParams, selectedFrequency);
                return (
                  <>
                    <div>
                      <span className="text-blue-300">Real:</span>
                      <span className="font-mono text-blue-100 ml-1">{refCalc.Z_total_real.toFixed(2)} Ω</span>
                    </div>
                    <div>
                      <span className="text-blue-300">Imag:</span>
                      <span className="font-mono text-blue-100 ml-1">{refCalc.Z_total_imag.toFixed(2)} Ω</span>
                    </div>
                    <div>
                      <span className="text-blue-300">|Z|:</span>
                      <span className="font-mono text-blue-100 ml-1">{refCalc.magnitude.toFixed(2)} Ω</span>
                    </div>
                  </>
                );
              })()}
            </div>
          </div>
        </div>
      </div>
      
      {/* Add grid info section */}
      {renderGridInfo()}

      {/* Dynamic Resnorm Percentiles Section */}
      <div className="bg-blue-900/20 rounded-lg p-4 border border-blue-700/30">
        <h3 className="text-sm font-medium text-blue-200 mb-3 flex items-center">
          <svg className="w-4 h-4 mr-2 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          Dynamic Resnorm Categories 
          <span className="text-xs text-blue-400 ml-2 font-normal">
            (Updates with sorting: {sortColumn} {sortDirection})
          </span>
        </h3>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
          <div className="bg-emerald-900/30 p-2.5 rounded border border-emerald-700/40">
            <div className="flex items-center gap-2 mb-1">
              <div className="w-3 h-3 rounded-full bg-emerald-600"></div>
              <span className="text-emerald-200 font-medium">Excellent (≤25%)</span>
            </div>
            <div className="font-mono text-emerald-300">
              ≤ {resnormPercentiles.p25.toExponential(2)}
            </div>
          </div>
          
          <div className="bg-green-900/30 p-2.5 rounded border border-green-700/40">
            <div className="flex items-center gap-2 mb-1">
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
              <span className="text-green-200 font-medium">Good (25-50%)</span>
            </div>
            <div className="font-mono text-green-300">
              ≤ {resnormPercentiles.p50.toExponential(2)}
            </div>
          </div>
          
          <div className="bg-amber-900/30 p-2.5 rounded border border-amber-700/40">
            <div className="flex items-center gap-2 mb-1">
              <div className="w-3 h-3 rounded-full bg-amber-500"></div>
              <span className="text-amber-200 font-medium">Moderate (50-75%)</span>
            </div>
            <div className="font-mono text-amber-300">
              ≤ {resnormPercentiles.p75.toExponential(2)}
            </div>
          </div>
          
          <div className="bg-orange-900/30 p-2.5 rounded border border-orange-700/40">
            <div className="flex items-center gap-2 mb-1">
              <div className="w-3 h-3 rounded-full bg-orange-500"></div>
              <span className="text-orange-200 font-medium">Acceptable (75-90%)</span>
            </div>
            <div className="font-mono text-orange-300">
              ≤ {resnormPercentiles.p90.toExponential(2)}
            </div>
          </div>
        </div>
        
        <div className="mt-3 text-xs text-blue-300/70">
          <span className="font-medium">Note:</span> Categories are calculated dynamically from currently displayed data.
          Poor fits (&gt;90%) are shown in red.
        </div>
      </div>

      {/* Parameter Value Distribution - only show when grid data is available */}
      {gridParameterArrays && (
        <div className="bg-neutral-800/50 rounded-lg p-4 border border-neutral-700">
          <h3 className="text-sm font-medium text-neutral-300 mb-3 flex items-center">
            <svg className="w-4 h-4 mr-2 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            </svg>
            Parameter Value Distribution (Grid Generation Values)
          </h3>
        
        <div className="space-y-3 text-sm">
          <div>
            <span className="text-neutral-400 font-medium">R shunt (Ω):</span>
            <div className="mt-1 font-mono text-xs text-neutral-300 break-all">
              [{uniqueValues.Rsh.map(v => v.toFixed(1)).join(', ')}]
            </div>
            <div className="text-xs text-neutral-500 mt-1">Count: {uniqueValues.Rsh.length}</div>
          </div>
          
          <div>
            <span className="text-neutral-400 font-medium">Ra (Ω):</span>
            <div className="mt-1 font-mono text-xs text-neutral-300 break-all">
              [{uniqueValues.Ra.map(v => v.toFixed(1)).join(', ')}]
            </div>
            <div className="text-xs text-neutral-500 mt-1">Count: {uniqueValues.Ra.length}</div>
          </div>
          
          <div>
            <span className="text-neutral-400 font-medium">Rb (Ω):</span>
            <div className="mt-1 font-mono text-xs text-neutral-300 break-all">
              [{uniqueValues.Rb.map(v => v.toFixed(1)).join(', ')}]
            </div>
            <div className="text-xs text-neutral-500 mt-1">Count: {uniqueValues.Rb.length}</div>
          </div>
          
          <div>
            <span className="text-neutral-400 font-medium">Ca (µF):</span>
            <div className="mt-1 font-mono text-xs text-neutral-300 break-all">
              [{uniqueValues.Ca.map(v => v.toFixed(2)).join(', ')}]
            </div>
            <div className="text-xs text-neutral-500 mt-1">Count: {uniqueValues.Ca.length}</div>
          </div>
          
          <div>
            <span className="text-neutral-400 font-medium">Cb (µF):</span>
            <div className="mt-1 font-mono text-xs text-neutral-300 break-all">
              [{uniqueValues.Cb.map(v => v.toFixed(2)).join(', ')}]
            </div>
            <div className="text-xs text-neutral-500 mt-1">Count: {uniqueValues.Cb.length}</div>
          </div>
        </div>
        </div>
      )}

      <div className="card">
        <header className="card-header">
          <div className="flex items-center">
            <svg className="w-4 h-4 mr-2 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <h3 className="text-sm font-medium text-neutral-300">Grid Results Data</h3>
          </div>
          <p className="text-xs text-neutral-500 mt-2">
            Showing all {gridResults.length} grid points, grouped by quality score
          </p>
        </header>
          
        <div className="px-5 pb-5">
          <div className="flex justify-between items-center mb-3">
            <h4 className="text-xs font-medium text-neutral-300 flex items-center">
              <svg className="w-3.5 h-3.5 mr-1.5 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
              Parameters Data
            </h4>
            <div className="text-xs text-neutral-500">
              Showing {startIndex + 1}-{Math.min(endIndex, displayCount)} of {gridResults.length} results ({Math.round((maxGridPoints / Math.pow(gridSize, 5)) * 100)}% of total {Math.pow(gridSize, 5).toLocaleString()})
            </div>
          </div>
          
          <table className="min-w-full divide-y divide-neutral-700 data-table">
            <thead>
              <tr className="bg-neutral-800 text-neutral-300 text-xs">
                <th className="p-2 text-left cursor-pointer" onClick={() => handleSort('id')}>
                  <div className="flex items-center justify-between">
                    <span>ID</span>
                    {renderSortArrow('id')}
                  </div>
                </th>
                <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-neutral-400 uppercase tracking-wider">
                  <button 
                    className="flex items-center gap-1 hover:text-neutral-200 transition-colors"
                    onClick={() => handleSort('Rsh')}
                  >
                    R shunt (Ω) {renderSortArrow('Rsh')}
                  </button>
                </th>
                <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-neutral-400 uppercase tracking-wider">
                  <button 
                    className="flex items-center gap-1 hover:text-neutral-200 transition-colors"
                    onClick={() => handleSort('Ra')}
                  >
                    Ra (Ω) {renderSortArrow('Ra')}
                  </button>
                </th>
                <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-neutral-400 uppercase tracking-wider">
                  <button 
                    className="flex items-center gap-1 hover:text-neutral-200 transition-colors"
                    onClick={() => handleSort('Ca')}
                  >
                    Ca (µF) {renderSortArrow('Ca')}
                  </button>
                </th>
                <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-neutral-400 uppercase tracking-wider">
                  <button 
                    className="flex items-center gap-1 hover:text-neutral-200 transition-colors"
                    onClick={() => handleSort('Rb')}
                  >
                    Rb (Ω) {renderSortArrow('Rb')}
                  </button>
                </th>
                <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-neutral-400 uppercase tracking-wider">
                  <button 
                    className="flex items-center gap-1 hover:text-neutral-200 transition-colors"
                    onClick={() => handleSort('Cb')}
                  >
                    Cb (µF) {renderSortArrow('Cb')}
                  </button>
                </th>
                <th className="p-2 text-left cursor-pointer" onClick={() => handleSort('resnorm')}>
                  <div className="flex items-center justify-between">
                    <span>Fit Quality (Resnorm)</span>
                    {renderSortArrow('resnorm')}
                  </div>
                </th>
                <th className="p-2 text-left">Real (Ω)</th>
                <th className="p-2 text-left">Imag (Ω)</th>
                <th className="p-2 text-left">|Z| (Ω)</th>
                <th className="p-2 text-left">Opacity</th>
                <th className="p-2 text-left">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-700">
              {/* Reference Row */}
              {(() => {
                const refParams = groundTruthParams || parameters;
                
                // Safety check to prevent errors with undefined parameters
                if (!refParams || !refParams.Rsh || selectedFrequency === undefined || selectedFrequency === null) {
                  return null;
                }
                
                const refCalc = calculateImpedanceAtFrequency(refParams, selectedFrequency);
                return (
                  <tr className="bg-primary/10 hover:bg-primary/15 transition-colors border-t border-neutral-700">
                    <td className="p-2 font-medium text-primary">Ref</td>
                    <td className="px-3 py-2 text-sm font-mono whitespace-nowrap">
                      {formatValue(refParams.Rsh, false)}
                    </td>
                    <td className="px-3 py-2 text-sm font-mono whitespace-nowrap">
                      {formatValue(refParams.Ra, false)}
                    </td>
                    <td className="px-3 py-2 text-sm font-mono whitespace-nowrap">
                      {formatValue(refParams.Ca, true)}
                    </td>
                    <td className="px-3 py-2 text-sm font-mono whitespace-nowrap">
                      {formatValue(refParams.Rb, false)}
                    </td>
                    <td className="px-3 py-2 text-sm font-mono whitespace-nowrap">
                      {formatValue(refParams.Cb, true)}
                    </td>
                    <td className="p-2">
                      <div className="flex items-center space-x-2">
                        <span className="w-2.5 h-2.5 rounded-full bg-primary"></span>
                        <span className="font-medium text-primary mr-2">Perfect</span>
                        <span className="text-xs font-mono text-primary">0.00E+00</span>
                      </div>
                    </td>
                    <td className="p-2 text-xs font-mono">{refCalc.Z_total_real.toFixed(2)}</td>
                    <td className="p-2 text-xs font-mono">{refCalc.Z_total_imag.toFixed(2)}</td>
                    <td className="p-2 text-xs font-mono">{refCalc.magnitude.toFixed(2)}</td>
                    <td className="p-2 text-xs font-mono">1.000</td>
                    <td className="p-2">
                      <button 
                        onClick={() => {
                          setSelectedParams(refParams);
                          setShowCalculationBreakdown(true);
                        }}
                        className="text-xs bg-primary/20 hover:bg-primary/30 text-primary px-2 py-1 rounded transition-colors"
                      >
                        Show Math
                      </button>
                    </td>
                  </tr>
                );
              })()}
              
              {/* Paginated Results */}
              {getPaginatedGridResults()
                // Show all points regardless of visibility status in the spider plot
                .map((point: BackendMeshPoint & { id: number }, idx: number) => {
                  const { qualityLabel, groupColor, groupIndex } = getQualityInfo(point);
                  const isHidden = groupIndex >= 0 && hiddenGroupsNum.includes(groupIndex);
                  
                  const group = getGroupForPoint(point);
                  let opacity = 1;
                  if (group && group.items.length > 1 && point.resnorm !== undefined) {
                    const groupResnorms = group.items.map(item => item.resnorm).filter(r => r !== undefined) as number[];
                    const minGroupResnorm = Math.min(...groupResnorms);
                    const maxGroupResnorm = Math.max(...groupResnorms);
                    if (maxGroupResnorm > minGroupResnorm) {
                      const normalizedResnorm = (point.resnorm - minGroupResnorm) / (maxGroupResnorm - minGroupResnorm);
                      const mapped = 1 - Math.pow(normalizedResnorm, opacityExponent);
                      opacity = 0.05 + (Math.max(0, Math.min(mapped, 1)) * 0.95);
                    }
                  }
                  
                  // Safety check for parameter data
                  if (!point.parameters || !point.parameters.Rsh || selectedFrequency === undefined || selectedFrequency === null) {
                    return null; // Skip this row if parameters are invalid
                  }
                  
                  const pointCalc = calculateImpedanceAtFrequency(point.parameters, selectedFrequency);
                  
                  return (
                    <tr 
                      key={`result-${idx}`} 
                      className={`${idx % 2 === 0 ? 'bg-neutral-900' : 'bg-neutral-800/60'} 
                        hover:bg-neutral-800/80 transition-colors border-t border-neutral-700
                        ${isHidden ? 'opacity-50' : ''}`}
                    >
                      <td className="p-2 font-medium">{point.id}</td>
                      <td className="px-3 py-2 text-sm font-mono whitespace-nowrap">
                        {formatValue(point.parameters.Rsh, false)}
                      </td>
                      <td className="px-3 py-2 text-sm font-mono whitespace-nowrap">
                        {formatValue(point.parameters.Ra, false)}
                      </td>
                      <td className="px-3 py-2 text-sm font-mono whitespace-nowrap">
                        {formatValue(point.parameters.Ca, true)}
                      </td>
                      <td className="px-3 py-2 text-sm font-mono whitespace-nowrap">
                        {formatValue(point.parameters.Rb, false)}
                      </td>
                      <td className="px-3 py-2 text-sm font-mono whitespace-nowrap">
                        {formatValue(point.parameters.Cb, true)}
                      </td>
                      <td className="p-2">
                        <div className="flex items-center space-x-2">
                          <span 
                            className="w-2.5 h-2.5 rounded-full"
                            style={{ backgroundColor: groupColor }}
                          ></span>
                          <span className="font-medium mr-2" style={{ color: groupColor }}>{qualityLabel}</span>
                          <span className="text-xs font-mono text-neutral-400">{point.resnorm.toExponential(3)}</span>
                          {isHidden && (
                            <span className="text-xs text-neutral-500 ml-1">(hidden)</span>
                          )}
                        </div>
                      </td>
                      <td className="p-2 text-xs font-mono">{pointCalc.Z_total_real.toFixed(2)}</td>
                      <td className="p-2 text-xs font-mono">{pointCalc.Z_total_imag.toFixed(2)}</td>
                      <td className="p-2 text-xs font-mono">{pointCalc.magnitude.toFixed(2)}</td>
                      <td className="p-2 text-xs font-mono">{opacity.toFixed(3)}</td>
                      <td className="p-2">
                        <button 
                          onClick={() => {
                            setSelectedParams(point.parameters);
                            setShowCalculationBreakdown(true);
                          }}
                          className="text-xs bg-neutral-700 hover:bg-neutral-600 text-neutral-300 px-2 py-1 rounded transition-colors"
                        >
                          Show Math
                        </button>
                      </td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
          
          {/* Pagination Controls */}
          {totalPages > 1 && (
            <div className="mt-4 flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => handlePageChange(1)}
                  disabled={currentPage === 1}
                  className={`p-1.5 rounded-md text-neutral-400 ${currentPage === 1 ? 'opacity-50 cursor-not-allowed' : 'hover:text-primary hover:bg-neutral-800/50'}`}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
                  </svg>
                </button>
                <button
                  onClick={() => handlePageChange(currentPage - 1)}
                  disabled={currentPage === 1}
                  className={`p-1.5 rounded-md text-neutral-400 ${currentPage === 1 ? 'opacity-50 cursor-not-allowed' : 'hover:text-primary hover:bg-neutral-800/50'}`}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                  </svg>
                </button>
                
                <span className="text-sm text-neutral-400">
                  Page {currentPage} of {totalPages}
                </span>
                
                <button
                  onClick={() => handlePageChange(currentPage + 1)}
                  disabled={currentPage === totalPages}
                  className={`p-1.5 rounded-md text-neutral-400 ${currentPage === totalPages ? 'opacity-50 cursor-not-allowed' : 'hover:text-primary hover:bg-neutral-800/50'}`}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </button>
                <button
                  onClick={() => handlePageChange(totalPages)}
                  disabled={currentPage === totalPages}
                  className={`p-1.5 rounded-md text-neutral-400 ${currentPage === totalPages ? 'opacity-50 cursor-not-allowed' : 'hover:text-primary hover:bg-neutral-800/50'}`}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                  </svg>
                </button>
              </div>
              
              <div className="text-xs text-neutral-400">
                <span className="font-medium text-neutral-300">TER</span> = (Rsh × (Ra + Rb)) / (Rsh + Ra + Rb) = {(() => {
                  const refParams = groundTruthParams || parameters;
                  const ter = (refParams.Rsh * (refParams.Ra + refParams.Rb)) / (refParams.Rsh + (refParams.Ra + refParams.Rb));
                  return ter.toFixed(1);
                })()} Ω (reference)
              </div>
            </div>
          )}
        </div>
      </div>
      
      {/* Calculation Breakdown Modal */}
      {showCalculationBreakdown && selectedParams && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-neutral-900 rounded-lg border border-neutral-700 max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-medium text-neutral-200">Impedance Calculation Breakdown</h2>
                <button 
                  onClick={() => setShowCalculationBreakdown(false)}
                  className="text-neutral-400 hover:text-neutral-200 p-1"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              {(() => {
                // Safety check for parameter data
                if (!selectedParams || !selectedParams.Rsh || selectedFrequency === undefined || selectedFrequency === null) {
                  return <div className="text-red-400">Invalid parameters or frequency selected</div>;
                }
                
                const calc = calculateImpedanceAtFrequency(selectedParams, selectedFrequency);
                const omega = 2 * Math.PI * selectedFrequency;
                
                return (
                  <div className="space-y-6">
                    {/* Parameters */}
                    <div className="bg-neutral-800/50 rounded-lg p-4">
                      <h3 className="text-lg font-medium text-neutral-200 mb-3">Circuit Parameters</h3>
                      <div className="grid grid-cols-3 gap-4 text-sm">
                        <div><span className="text-neutral-400">Rsh:</span> <span className="font-mono text-neutral-200">{selectedParams.Rsh.toFixed(2)} Ω</span></div>
                        <div><span className="text-neutral-400">Ra:</span> <span className="font-mono text-neutral-200">{selectedParams.Ra.toFixed(2)} Ω</span></div>
                        <div><span className="text-neutral-400">Ca:</span> <span className="font-mono text-neutral-200">{(selectedParams.Ca * 1e6).toFixed(2)} μF</span></div>
                        <div><span className="text-neutral-400">Rb:</span> <span className="font-mono text-neutral-200">{selectedParams.Rb.toFixed(2)} Ω</span></div>
                        <div><span className="text-neutral-400">Cb:</span> <span className="font-mono text-neutral-200">{(selectedParams.Cb * 1e6).toFixed(2)} μF</span></div>
                        <div><span className="text-neutral-400">f:</span> <span className="font-mono text-neutral-200">{selectedFrequency.toLocaleString()} Hz</span></div>
                      </div>
                      <div className="mt-3 text-sm">
                        <span className="text-neutral-400">ω = 2πf = </span>
                        <span className="font-mono text-neutral-200">{omega.toExponential(3)} rad/s</span>
                      </div>
                      
                      {/* Unit Conversion Note */}
                      <div className="mt-3 p-3 bg-blue-900/10 rounded border border-blue-700/20">
                        <div className="text-xs text-blue-200 font-medium mb-2">📝 Unit Conversions Used:</div>
                        <div className="text-xs text-blue-300/80 space-y-1">
                          <div>• Capacitance: Internal storage in Farads (F) → Display in microfarads (μF) × 10⁶</div>
                          <div>• Ca = {selectedParams.Ca.toExponential(3)} F = {(selectedParams.Ca * 1e6).toFixed(2)} μF</div>
                          <div>• Cb = {selectedParams.Cb.toExponential(3)} F = {(selectedParams.Cb * 1e6).toFixed(2)} μF</div>
                        </div>
                      </div>
                    </div>
                    
                    {/* Step-by-step calculation */}
                    <div className="bg-neutral-800/50 rounded-lg p-4">
                      <h3 className="text-lg font-medium text-neutral-200 mb-4">Step-by-Step Calculation</h3>
                      
                      <div className="space-y-4 text-sm">
                        {/* Za calculation */}
                        <div className="border border-neutral-600 rounded p-3">
                          <h4 className="font-medium text-green-400 mb-2">Step 1: Calculate Za = Ra/(1+jωRaCa)</h4>
                          <div className="space-y-2 font-mono text-xs">
                            <div>ωRaCa = {omega.toExponential(3)} rad/s × {selectedParams.Ra.toFixed(2)} Ω × {selectedParams.Ca.toExponential(3)} F = {(omega * selectedParams.Ra * selectedParams.Ca).toExponential(3)} (dimensionless)</div>
                            <div>Denominator: 1 + j({(omega * selectedParams.Ra * selectedParams.Ca).toExponential(3)})</div>
                            <div>|Denominator|² = 1² + ({(omega * selectedParams.Ra * selectedParams.Ca).toExponential(3)})² = {(1 + Math.pow(omega * selectedParams.Ra * selectedParams.Ca, 2)).toExponential(3)}</div>
                            <div className="text-green-300">Za = {calc.Za_real.toFixed(3)} + j({calc.Za_imag.toFixed(3)}) Ω</div>
                          </div>
                        </div>
                        
                        {/* Zb calculation */}
                        <div className="border border-neutral-600 rounded p-3">
                          <h4 className="font-medium text-blue-400 mb-2">Step 2: Calculate Zb = Rb/(1+jωRbCb)</h4>
                          <div className="space-y-2 font-mono text-xs">
                            <div>ωRbCb = {omega.toExponential(3)} rad/s × {selectedParams.Rb.toFixed(2)} Ω × {selectedParams.Cb.toExponential(3)} F = {(omega * selectedParams.Rb * selectedParams.Cb).toExponential(3)} (dimensionless)</div>
                            <div>Denominator: 1 + j({(omega * selectedParams.Rb * selectedParams.Cb).toExponential(3)})</div>
                            <div>|Denominator|² = 1² + ({(omega * selectedParams.Rb * selectedParams.Cb).toExponential(3)})² = {(1 + Math.pow(omega * selectedParams.Rb * selectedParams.Cb, 2)).toExponential(3)}</div>
                            <div className="text-blue-300">Zb = {calc.Zb_real.toFixed(3)} + j({calc.Zb_imag.toFixed(3)}) Ω</div>
                          </div>
                        </div>
                        
                        {/* Za + Zb */}
                        <div className="border border-neutral-600 rounded p-3">
                          <h4 className="font-medium text-purple-400 mb-2">Step 3: Calculate Za + Zb</h4>
                          <div className="space-y-2 font-mono text-xs">
                            <div>Real: {calc.Za_real.toFixed(3)} Ω + {calc.Zb_real.toFixed(3)} Ω = {calc.Za_plus_Zb_real.toFixed(3)} Ω</div>
                            <div>Imaginary: ({calc.Za_imag.toFixed(3)}) Ω + ({calc.Zb_imag.toFixed(3)}) Ω = {calc.Za_plus_Zb_imag.toFixed(3)} Ω</div>
                            <div className="text-purple-300">Za + Zb = {calc.Za_plus_Zb_real.toFixed(3)} + j({calc.Za_plus_Zb_imag.toFixed(3)}) Ω</div>
                          </div>
                        </div>
                        
                        {/* Final calculation */}
                        <div className="border border-neutral-600 rounded p-3">
                          <h4 className="font-medium text-yellow-400 mb-2">Step 4: Calculate Z_total = (Rsh × (Za + Zb)) / (Rsh + Za + Zb)</h4>
                          <div className="space-y-2 font-mono text-xs">
                            <div>Numerator: {selectedParams.Rsh.toFixed(2)} Ω × ({calc.Za_plus_Zb_real.toFixed(3)} + j{calc.Za_plus_Zb_imag.toFixed(3)}) Ω</div>
                            <div>Denominator: {selectedParams.Rsh.toFixed(2)} Ω + ({calc.Za_plus_Zb_real.toFixed(3)} + j{calc.Za_plus_Zb_imag.toFixed(3)}) Ω</div>
                            <div>Denominator: {(selectedParams.Rsh + calc.Za_plus_Zb_real).toFixed(3)} + j({calc.Za_plus_Zb_imag.toFixed(3)}) Ω</div>
                            <div className="text-yellow-300 text-lg font-semibold mt-2">Z_total = {calc.Z_total_real.toFixed(3)} + j({calc.Z_total_imag.toFixed(3)}) Ω</div>
                            <div className="text-yellow-300">|Z| = {calc.magnitude.toFixed(3)} Ω, φ = {calc.phase.toFixed(2)}°</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })()}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}; 