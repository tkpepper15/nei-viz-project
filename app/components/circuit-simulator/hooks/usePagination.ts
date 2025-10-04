import { useState, useMemo } from 'react';

export interface PaginationConfig {
  itemsPerPage: number;
  currentPage: number;
}

export interface PaginationResult<T> {
  currentItems: T[];
  currentPage: number;
  totalPages: number;
  totalItems: number;
  hasNextPage: boolean;
  hasPreviousPage: boolean;
  nextPage: () => void;
  previousPage: () => void;
  goToPage: (page: number) => void;
  setItemsPerPage: (count: number) => void;
}

/**
 * Custom hook for pagination of large datasets
 * Optimized for circuit model visualization to prevent hydration errors
 */
export function usePagination<T>(
  items: T[],
  initialItemsPerPage: number = 1000
): PaginationResult<T> {
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPageState] = useState(initialItemsPerPage);

  const paginatedData = useMemo(() => {
    const totalItems = items.length;
    const totalPages = Math.ceil(totalItems / itemsPerPage);
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    const currentItems = items.slice(startIndex, endIndex);

    return {
      currentItems,
      totalPages,
      totalItems,
      hasNextPage: currentPage < totalPages,
      hasPreviousPage: currentPage > 1,
    };
  }, [items, currentPage, itemsPerPage]);

  const nextPage = () => {
    if (paginatedData.hasNextPage) {
      setCurrentPage(p => p + 1);
    }
  };

  const previousPage = () => {
    if (paginatedData.hasPreviousPage) {
      setCurrentPage(p => p - 1);
    }
  };

  const goToPage = (page: number) => {
    const validPage = Math.max(1, Math.min(page, paginatedData.totalPages));
    setCurrentPage(validPage);
  };

  const setItemsPerPage = (count: number) => {
    setItemsPerPageState(count);
    setCurrentPage(1); // Reset to first page when changing items per page
  };

  return {
    currentItems: paginatedData.currentItems,
    currentPage,
    totalPages: paginatedData.totalPages,
    totalItems: paginatedData.totalItems,
    hasNextPage: paginatedData.hasNextPage,
    hasPreviousPage: paginatedData.hasPreviousPage,
    nextPage,
    previousPage,
    goToPage,
    setItemsPerPage,
  };
}
