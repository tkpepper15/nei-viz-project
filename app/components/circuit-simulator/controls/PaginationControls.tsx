"use client";

import React from 'react';

interface PaginationInfo {
  totalPages: number;
  currentPage: number;
  hasNextPage: boolean;
  hasPrevPage: boolean;
  totalResults: number;
  resnormRange: { min: number; max: number; avgCurrentPage: number };
  pageInfo: { startIndex: number; endIndex: number };
}

interface PaginationControlsProps {
  pagination: PaginationInfo;
  pageSize: number;
  onPageChange: (page: number) => void;
  onPageSizeChange: (size: number) => void;
  isLoading?: boolean;
  className?: string;
}

export const PaginationControls: React.FC<PaginationControlsProps> = ({
  pagination,
  pageSize,
  onPageChange,
  onPageSizeChange,
  isLoading = false,
  className = ''
}) => {
  // Keyboard navigation support
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle if not typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
          case 'ArrowLeft':
            e.preventDefault();
            if (pagination.hasPrevPage && !isLoading) {
              onPageChange(pagination.currentPage - 1);
            }
            break;
          case 'ArrowRight':
            e.preventDefault();
            if (pagination.hasNextPage && !isLoading) {
              onPageChange(pagination.currentPage + 1);
            }
            break;
          case 'Home':
            e.preventDefault();
            if (!isLoading && pagination.currentPage > 1) {
              onPageChange(1);
            }
            break;
          case 'End':
            e.preventDefault();
            if (!isLoading && pagination.currentPage < pagination.totalPages) {
              onPageChange(pagination.totalPages);
            }
            break;
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [pagination, onPageChange, isLoading]);
  // Generate page number array for pagination buttons
  const getVisiblePages = () => {
    const { currentPage, totalPages } = pagination;
    const maxVisible = 7; // Show max 7 page buttons

    if (totalPages <= maxVisible) {
      return Array.from({ length: totalPages }, (_, i) => i + 1);
    }

    let start = Math.max(1, currentPage - Math.floor(maxVisible / 2));
    const end = Math.min(totalPages, start + maxVisible - 1);

    // Adjust start if we're near the end
    if (end === totalPages) {
      start = Math.max(1, end - maxVisible + 1);
    }

    const pages: (number | string)[] = [];

    // Add first page and ellipsis if needed
    if (start > 1) {
      pages.push(1);
      if (start > 2) {
        pages.push('...');
      }
    }

    // Add visible page range
    for (let i = start; i <= end; i++) {
      pages.push(i);
    }

    // Add ellipsis and last page if needed
    if (end < totalPages) {
      if (end < totalPages - 1) {
        pages.push('...');
      }
      pages.push(totalPages);
    }

    return pages;
  };

  const visiblePages = getVisiblePages();

  // Format large numbers for display
  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  // Enhanced jump to page with input validation
  const [jumpPageInput, setJumpPageInput] = React.useState('');
  const [showJumpInput, setShowJumpInput] = React.useState(false);

  const handleJumpToPage = () => {
    if (showJumpInput && jumpPageInput.trim()) {
      const pageNum = parseInt(jumpPageInput, 10);
      if (pageNum >= 1 && pageNum <= pagination.totalPages) {
        onPageChange(pageNum);
        setJumpPageInput('');
        setShowJumpInput(false);
      }
    } else {
      setShowJumpInput(true);
      setTimeout(() => {
        const input = document.querySelector('.page-jump-input') as HTMLInputElement;
        if (input) input.focus();
      }, 100);
    }
  };

  const handleJumpInputKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleJumpToPage();
    } else if (e.key === 'Escape') {
      setShowJumpInput(false);
      setJumpPageInput('');
    }
  };

  // Quick jump buttons for large datasets
  const getQuickJumpButtons = () => {
    const { totalPages } = pagination;
    const jumps = [];

    if (totalPages > 10) {
      jumps.push({ label: 'First', page: 1 });
    }

    if (totalPages > 100) {
      jumps.push({ label: '10%', page: Math.ceil(totalPages * 0.1) });
      jumps.push({ label: '25%', page: Math.ceil(totalPages * 0.25) });
      jumps.push({ label: '50%', page: Math.ceil(totalPages * 0.5) });
      jumps.push({ label: '75%', page: Math.ceil(totalPages * 0.75) });
      jumps.push({ label: '90%', page: Math.ceil(totalPages * 0.9) });
    }

    if (totalPages > 10) {
      jumps.push({ label: 'Last', page: totalPages });
    }

    return jumps;
  };

  const quickJumps = getQuickJumpButtons();

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Main Pagination Info */}
      <div className="bg-neutral-800 border border-neutral-600 rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <h4 className="text-sm font-medium text-neutral-200">Result Navigation</h4>
            {isLoading && (
              <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
            )}
            {/* Keyboard shortcuts tooltip */}
            <div className="relative group">
              <svg className="w-3 h-3 text-neutral-500 hover:text-neutral-300 cursor-help" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <div className="absolute bottom-full left-0 mb-2 w-48 p-2 text-xs bg-neutral-900 border border-neutral-600 rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity z-10">
                <div className="font-medium mb-1">Keyboard Shortcuts:</div>
                <div>Ctrl+← Previous page</div>
                <div>Ctrl+→ Next page</div>
                <div>Ctrl+Home First page</div>
                <div>Ctrl+End Last page</div>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="text-xs text-neutral-400">
              {formatNumber(pagination.pageInfo.startIndex + 1)}-{formatNumber(pagination.pageInfo.endIndex + 1)} of {formatNumber(pagination.totalResults)}
            </div>
            {/* Progress indicator */}
            <div className="text-xs text-blue-400 font-mono">
              {((pagination.currentPage / pagination.totalPages) * 100).toFixed(1)}%
            </div>
          </div>
        </div>

        {/* Progress bar */}
        <div className="w-full bg-neutral-700 rounded-full h-1.5 mb-3 overflow-hidden">
          <div
            className="bg-blue-500 h-1.5 rounded-full transition-all duration-300 ease-out"
            style={{ width: `${(pagination.currentPage / pagination.totalPages) * 100}%` }}
          ></div>
        </div>

        {/* Resnorm Range Info */}
        <div className="grid grid-cols-3 gap-2 text-xs text-neutral-400 mb-4">
          <div>
            <div className="text-neutral-300 font-medium">Min</div>
            <div className="text-green-400">{pagination.resnormRange.min.toFixed(3)}</div>
          </div>
          <div>
            <div className="text-neutral-300 font-medium">Page Avg</div>
            <div className="text-blue-400">{pagination.resnormRange.avgCurrentPage.toFixed(3)}</div>
          </div>
          <div>
            <div className="text-neutral-300 font-medium">Max</div>
            <div className="text-red-400">{pagination.resnormRange.max.toFixed(3)}</div>
          </div>
        </div>

        {/* Enhanced Page Size Selector */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <label className="text-xs text-neutral-400">Results per page:</label>
            <select
              value={pageSize}
              onChange={(e) => onPageSizeChange(parseInt(e.target.value))}
              disabled={isLoading}
              className="px-3 py-1.5 bg-neutral-700 border border-neutral-600 rounded-md text-xs text-neutral-200 focus:outline-none focus:ring-1 focus:ring-blue-500 transition-colors hover:bg-neutral-600"
            >
              <option value={250}>250</option>
              <option value={500}>500</option>
              <option value={1000}>1,000</option>
              <option value={2000}>2,000</option>
              <option value={5000}>5,000</option>
            </select>
          </div>

          {/* Show estimated pages info */}
          <div className="text-xs text-neutral-500">
            ~{Math.ceil(pagination.totalResults / pageSize).toLocaleString()} pages
          </div>
        </div>

        {/* Navigation Controls */}
        <div className="flex items-center justify-between">
          {/* Previous/Next Buttons */}
          <div className="flex items-center gap-1">
            <button
              onClick={() => onPageChange(1)}
              disabled={!pagination.hasPrevPage || isLoading}
              className="px-2 py-1 bg-neutral-700 hover:bg-neutral-600 disabled:bg-neutral-800 disabled:text-neutral-500 text-xs text-neutral-200 rounded transition-colors flex items-center gap-1"
              title="First Page"
            >
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
              </svg>
            </button>
            <button
              onClick={() => onPageChange(pagination.currentPage - 1)}
              disabled={!pagination.hasPrevPage || isLoading}
              className="px-2 py-1 bg-neutral-700 hover:bg-neutral-600 disabled:bg-neutral-800 disabled:text-neutral-500 text-xs text-neutral-200 rounded transition-colors flex items-center gap-1"
              title="Previous Page"
            >
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
              Prev
            </button>
          </div>

          {/* Current Page Indicator */}
          <div className="text-xs text-neutral-300 font-medium">
            Page {pagination.currentPage} of {pagination.totalPages}
          </div>

          {/* Next/Last Buttons */}
          <div className="flex items-center gap-1">
            <button
              onClick={() => onPageChange(pagination.currentPage + 1)}
              disabled={!pagination.hasNextPage || isLoading}
              className="px-2 py-1 bg-neutral-700 hover:bg-neutral-600 disabled:bg-neutral-800 disabled:text-neutral-500 text-xs text-neutral-200 rounded transition-colors flex items-center gap-1"
              title="Next Page"
            >
              Next
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </button>
            <button
              onClick={() => onPageChange(pagination.totalPages)}
              disabled={!pagination.hasNextPage || isLoading}
              className="px-2 py-1 bg-neutral-700 hover:bg-neutral-600 disabled:bg-neutral-800 disabled:text-neutral-500 text-xs text-neutral-200 rounded transition-colors flex items-center gap-1"
              title="Last Page"
            >
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Enhanced Page Number Buttons with Jump Input */}
      {pagination.totalPages > 1 && (
        <div className="space-y-2">
          <div className="flex flex-wrap items-center justify-center gap-1">
            {visiblePages.map((page, index) => (
              <React.Fragment key={index}>
                {page === '...' ? (
                  showJumpInput ? (
                    <div className="flex items-center gap-1">
                      <input
                        type="number"
                        min="1"
                        max={pagination.totalPages}
                        value={jumpPageInput}
                        onChange={(e) => setJumpPageInput(e.target.value)}
                        onKeyDown={handleJumpInputKeyPress}
                        onBlur={() => {
                          setTimeout(() => {
                            if (!jumpPageInput.trim()) {
                              setShowJumpInput(false);
                            }
                          }, 150);
                        }}
                        className="page-jump-input w-16 px-2 py-1 text-xs bg-neutral-700 border border-neutral-600 rounded text-neutral-200 focus:outline-none focus:ring-1 focus:ring-blue-500"
                        placeholder="Page"
                      />
                      <button
                        onClick={handleJumpToPage}
                        className="px-2 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
                      >
                        Go
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={handleJumpToPage}
                      className="px-2 py-1 text-xs text-neutral-400 hover:text-blue-400 transition-colors hover:bg-neutral-700 rounded"
                      title="Jump to specific page"
                    >
                      ...
                    </button>
                  )
                ) : (
                  <button
                    onClick={() => onPageChange(page as number)}
                    disabled={isLoading}
                    className={`px-2 py-1 text-xs rounded transition-all duration-200 min-w-[28px] ${
                      page === pagination.currentPage
                        ? 'bg-blue-600 text-white shadow-md transform scale-105'
                        : 'bg-neutral-700 text-neutral-300 hover:bg-neutral-600 hover:text-white'
                    } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                  >
                    {page}
                  </button>
                )}
              </React.Fragment>
            ))}
          </div>

          {/* Keyboard shortcuts hint */}
          {showJumpInput && (
            <div className="text-xs text-neutral-500 text-center">
              Press Enter to jump, Escape to cancel
            </div>
          )}
        </div>
      )}

      {/* Quick Jump Buttons for Large Datasets */}
      {quickJumps.length > 0 && (
        <div className="border-t border-neutral-700 pt-3">
          <div className="text-xs text-neutral-400 mb-2">Quick Jump:</div>
          <div className="flex flex-wrap items-center gap-1">
            {quickJumps.map((jump) => (
              <button
                key={jump.label}
                onClick={() => onPageChange(jump.page)}
                disabled={isLoading || pagination.currentPage === jump.page}
                className="px-2 py-1 bg-neutral-700 hover:bg-neutral-600 disabled:bg-neutral-800 disabled:text-neutral-500 text-xs text-neutral-300 rounded transition-colors"
                title={`Page ${jump.page}`}
              >
                {jump.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Loading Indicator */}
      {isLoading && (
        <div className="flex items-center justify-center py-2">
          <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin mr-2"></div>
          <span className="text-xs text-neutral-400">Loading page...</span>
        </div>
      )}
    </div>
  );
};