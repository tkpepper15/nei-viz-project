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
}

// Add formatValue function to handle unit conversions
const formatValue = (value: number, isCapacitance: boolean): string => {
  if (isCapacitance) {
    // Convert from Farads to µF and format with 2 decimal places
    return (value * 1e6).toFixed(2);
  }
  // For resistance values, show 1 decimal place for values < 1000, otherwise no decimals
  return value < 1000 ? value.toFixed(1) : value.toFixed(0);
};

export const DataTableTab: React.FC<DataTableTabProps> = ({
  gridResults,
  gridResultsWithIds,
  resnormGroups,
  hiddenGroups,
  maxGridPoints,
  gridSize,
  parameters,
  groundTruthParams,
  gridParameterArrays
}) => {
  // Table sorting state
  const [sortColumn, setSortColumn] = useState<string>('resnorm');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [currentPage, setCurrentPage] = useState<number>(1);
  const resultsPerPage = 100;

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
        case 'Rs':
          aValue = a.parameters.Rs;
          bValue = b.parameters.Rs;
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
  const getQualityInfo = (point: BackendMeshPoint & { id: number }) => {
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
    
    return { qualityLabel, groupColor, groupIndex: groupIndex >= 0 ? groupIndex : qualityCategory, qualityCategory };
  };

  // Calculate unique values for each parameter
  const uniqueValues = useMemo(() => {
    const values = {
      Rs: new Set<number>(),
      Ra: new Set<number>(),
      Rb: new Set<number>(),
      Ca: new Set<number>(),
      Cb: new Set<number>()
    };

    gridResults.forEach(point => {
      values.Rs.add(Number(point.parameters.Rs.toFixed(2)));
      values.Ra.add(Number(point.parameters.Ra.toFixed(2)));
      values.Rb.add(Number(point.parameters.Rb.toFixed(2)));
      values.Ca.add(Number((point.parameters.Ca * 1e6).toFixed(2))); // Convert to µF
      values.Cb.add(Number((point.parameters.Cb * 1e6).toFixed(2))); // Convert to µF
    });

    return {
      Rs: Array.from(values.Rs).sort((a, b) => a - b),
      Ra: Array.from(values.Ra).sort((a, b) => a - b),
      Rb: Array.from(values.Rb).sort((a, b) => a - b),
      Ca: Array.from(values.Ca).sort((a, b) => a - b),
      Cb: Array.from(values.Cb).sort((a, b) => a - b)
    };
  }, [gridResults]);

  // Add grid parameter info display
  const renderGridInfo = () => {
    if (!gridParameterArrays) return null;

    const getMinMax = (values: number[]) => ({
      min: Math.min(...values),
      max: Math.max(...values)
    });

    const ranges = {
      Rs: getMinMax(gridParameterArrays.Rs),
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
            <span className="text-neutral-400">Rs: </span>
            <span className="font-mono text-neutral-300">
              {ranges.Rs.min.toFixed(1)} - {ranges.Rs.max.toFixed(1)} Ω
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

  return (
    <div className="space-y-4">
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

      {/* Debug section showing unique values */}
      <div className="bg-neutral-800/50 rounded-lg p-4 border border-neutral-700">
        <h3 className="text-sm font-medium text-neutral-300 mb-3 flex items-center">
          <svg className="w-4 h-4 mr-2 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
          </svg>
          Parameter Value Distribution
        </h3>
        
        <div className="space-y-3 text-sm">
          <div>
            <span className="text-neutral-400 font-medium">Rs (Ω):</span>
            <div className="mt-1 font-mono text-xs text-neutral-300 break-all">
              [{uniqueValues.Rs.map(v => v.toFixed(1)).join(', ')}]
            </div>
            <div className="text-xs text-neutral-500 mt-1">Count: {uniqueValues.Rs.length}</div>
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
                    onClick={() => handleSort('Rs')}
                  >
                    Rs (Ω) {renderSortArrow('Rs')}
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
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-700">
              {/* Reference Row */}
              <tr className="bg-primary/10 hover:bg-primary/15 transition-colors border-t border-neutral-700">
                <td className="p-2 font-medium text-primary">Ref</td>
                <td className="px-3 py-2 text-sm font-mono whitespace-nowrap">
                  {formatValue(groundTruthParams?.Rs || parameters.Rs, false)}
                </td>
                <td className="px-3 py-2 text-sm font-mono whitespace-nowrap">
                  {formatValue(groundTruthParams?.Ra || parameters.Ra, false)}
                </td>
                <td className="px-3 py-2 text-sm font-mono whitespace-nowrap">
                  {formatValue(groundTruthParams?.Ca || parameters.Ca, true)}
                </td>
                <td className="px-3 py-2 text-sm font-mono whitespace-nowrap">
                  {formatValue(groundTruthParams?.Rb || parameters.Rb, false)}
                </td>
                <td className="px-3 py-2 text-sm font-mono whitespace-nowrap">
                  {formatValue(groundTruthParams?.Cb || parameters.Cb, true)}
                </td>
                <td className="p-2">
                  <div className="flex items-center space-x-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-primary"></span>
                    <span className="font-medium text-primary mr-2">Perfect</span>
                    <span className="text-xs font-mono text-primary">0.00E+00</span>
                  </div>
                </td>
              </tr>
              
              {/* Paginated Results */}
              {getPaginatedGridResults()
                // Show all points regardless of visibility status in the spider plot
                .map((point: BackendMeshPoint & { id: number }, idx: number) => {
                  const { qualityLabel, groupColor, groupIndex } = getQualityInfo(point);
                  // Calculate visibility status for UI indication
                  const isHidden = groupIndex >= 0 && hiddenGroups.includes(groupIndex);
                  
                  return (
                    <tr 
                      key={`result-${idx}`} 
                      className={`${idx % 2 === 0 ? 'bg-neutral-900' : 'bg-neutral-800/60'} 
                        hover:bg-neutral-800/80 transition-colors border-t border-neutral-700
                        ${isHidden ? 'opacity-50' : ''}`}
                    >
                      <td className="p-2 font-medium">{point.id}</td>
                      <td className="px-3 py-2 text-sm font-mono whitespace-nowrap">
                        {formatValue(point.parameters.Rs, false)}
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
                <span className="font-medium text-neutral-300">TER</span> = Ra + Rb = {((groundTruthParams?.Ra || parameters.Ra) + (groundTruthParams?.Rb || parameters.Rb)).toFixed(0)} Ω (reference)
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}; 