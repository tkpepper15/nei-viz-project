import React, { useState } from 'react';
import { BackendMeshPoint, ResnormGroup } from '../circuit-simulator/utils/types';
import { CircuitParameters } from '../circuit-simulator/utils/impedance';

type DataTableTabProps = {
  gridResults: BackendMeshPoint[];
  gridResultsWithIds: (BackendMeshPoint & { id: number })[];
  resnormGroups: ResnormGroup[];
  hiddenGroups: number[];
  maxGridPoints: number;
  gridSize: number;
  parameters: CircuitParameters;
  groundTruthParams?: CircuitParameters;
};

const DataTableTab: React.FC<DataTableTabProps> = ({
  gridResults,
  gridResultsWithIds,
  resnormGroups,
  hiddenGroups,
  maxGridPoints,
  gridSize,
  parameters,
  groundTruthParams
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

  // Function to get sorted grid results
  const getSortedGridResults = () => {
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
  };

  // Calculate pagination values
  const displayCount = Math.min(maxGridPoints, gridResults.length);
  const totalPages = Math.ceil(displayCount / resultsPerPage);
  const startIndex = (currentPage - 1) * resultsPerPage;
  const endIndex = Math.min(startIndex + resultsPerPage, displayCount);
  
  // Function to get paginated grid results
  const getPaginatedGridResults = () => {
    return getSortedGridResults().slice(startIndex, endIndex);
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

  // Get quality label and color based on resnorm group
  const getQualityInfo = (point: BackendMeshPoint & { id: number }) => {
    const groupIndex = resnormGroups.findIndex(group => 
      group.items.some(item => 
        item.parameters.Rs === point.parameters.Rs &&
        item.parameters.Ra === point.parameters.Ra &&
        item.parameters.Ca === point.parameters.Ca &&
        item.parameters.Rb === point.parameters.Rb &&
        item.parameters.Cb === point.parameters.Cb
      )
    );
    
    // Determine quality category based on the group's color
    let qualityCategory = 1; // Default to Good
    let groupColor = '#3B82F6'; // Default blue
    
    if (groupIndex >= 0) {
      const group = resnormGroups[groupIndex];
      switch (group.color) {
        case '#10B981': // Very Good - Green
          qualityCategory = 0;
          groupColor = '#10B981';
          break;
        case '#3B82F6': // Good - Blue
          qualityCategory = 1;
          groupColor = '#3B82F6';
          break;
        case '#F59E0B': // Moderate - Amber
          qualityCategory = 2;
          groupColor = '#F59E0B';
          break;
        case '#EF4444': // Poor - Red
          qualityCategory = 3;
          groupColor = '#EF4444';
          break;
      }
    }
    
    // Get quality label based on quality category
    let qualityLabel = "Good";
    if (qualityCategory === 0) qualityLabel = "Very Good";
    else if (qualityCategory === 1) qualityLabel = "Good";
    else if (qualityCategory === 2) qualityLabel = "Moderate";
    else if (qualityCategory === 3) qualityLabel = "Poor";
    
    return { qualityLabel, groupColor, groupIndex, qualityCategory };
  };

  return (
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
        
        <table className="data-table w-full border border-neutral-700 rounded-lg overflow-hidden">
          <thead>
            <tr className="bg-neutral-800 text-neutral-300 text-xs">
              <th className="p-2 text-left cursor-pointer" onClick={() => handleSort('id')}>
                <div className="flex items-center justify-between">
                  <span>ID</span>
                  {renderSortArrow('id')}
                </div>
              </th>
              <th className="p-2 text-left cursor-pointer" onClick={() => handleSort('Rs')}>
                <div className="flex items-center justify-between">
                  <span>Rs (Ω)</span>
                  {renderSortArrow('Rs')}
                </div>
              </th>
              <th className="p-2 text-left cursor-pointer" onClick={() => handleSort('Ra')}>
                <div className="flex items-center justify-between">
                  <span>Ra (Ω)</span>
                  {renderSortArrow('Ra')}
                </div>
              </th>
              <th className="p-2 text-left cursor-pointer" onClick={() => handleSort('Ca')}>
                <div className="flex items-center justify-between">
                  <span>Ca (μF)</span>
                  {renderSortArrow('Ca')}
                </div>
              </th>
              <th className="p-2 text-left cursor-pointer" onClick={() => handleSort('Rb')}>
                <div className="flex items-center justify-between">
                  <span>Rb (Ω)</span>
                  {renderSortArrow('Rb')}
                </div>
              </th>
              <th className="p-2 text-left cursor-pointer" onClick={() => handleSort('Cb')}>
                <div className="flex items-center justify-between">
                  <span>Cb (μF)</span>
                  {renderSortArrow('Cb')}
                </div>
              </th>
              <th className="p-2 text-left cursor-pointer" onClick={() => handleSort('resnorm')}>
                <div className="flex items-center justify-between">
                  <span>Fit Quality (Resnorm)</span>
                  {renderSortArrow('resnorm')}
                </div>
              </th>
            </tr>
          </thead>
          <tbody>
            {/* Reference Row */}
            <tr className="bg-primary/10 hover:bg-primary/15 transition-colors border-t border-neutral-700">
              <td className="p-2 font-medium text-primary">Ref</td>
              <td className="p-2">{groundTruthParams?.Rs.toFixed(2) || parameters.Rs.toFixed(2)}</td>
              <td className="p-2">{groundTruthParams?.Ra.toFixed(0) || parameters.Ra.toFixed(0)}</td>
              <td className="p-2">{((groundTruthParams?.Ca || parameters.Ca) * 1e6).toFixed(2)}</td>
              <td className="p-2">{groundTruthParams?.Rb.toFixed(0) || parameters.Rb.toFixed(0)}</td>
              <td className="p-2">{((groundTruthParams?.Cb || parameters.Cb) * 1e6).toFixed(2)}</td>
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
              .map((point, idx) => {
                const { qualityLabel, groupColor, groupIndex } = getQualityInfo(point);
                // Calculate visibility status for UI indication
                const isHidden = groupIndex >= 0 && hiddenGroups.includes(groupIndex);
                
                return (
                  <tr 
                    key={idx} 
                    className={`${idx % 2 === 0 ? 'bg-neutral-900' : 'bg-neutral-800/60'} 
                      hover:bg-neutral-800/80 transition-colors border-t border-neutral-700
                      ${isHidden ? 'opacity-50' : ''}`}
                  >
                    <td className="p-2 font-medium">{point.id}</td>
                    <td className="p-2">{point.parameters.Rs.toFixed(2)}</td>
                    <td className="p-2">{point.parameters.Ra.toFixed(0)}</td>
                    <td className="p-2">{(point.parameters.Ca * 1e6).toFixed(2)}</td>
                    <td className="p-2">{point.parameters.Rb.toFixed(0)}</td>
                    <td className="p-2">{(point.parameters.Cb * 1e6).toFixed(2)}</td>
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
  );
};

export default DataTableTab; 