import { useState, useEffect, useCallback } from 'react';
import { useClientSide } from './useClientSide';

/**
 * Custom hook for safe localStorage operations that prevents SSR hydration mismatches
 */
export const useLocalStorage = <T>(
  key: string,
  initialValue: T
): [T, (value: T | ((prev: T) => T)) => void, boolean] => {
  const isClient = useClientSide();
  const [storedValue, setStoredValue] = useState<T>(initialValue);
  const [hasLoaded, setHasLoaded] = useState(false);

  // Load from localStorage only on client side
  useEffect(() => {
    if (!isClient || hasLoaded) return;

    try {
      const item = localStorage.getItem(key);
      if (item) {
        const parsed = JSON.parse(item);
        setStoredValue(parsed);
      }
    } catch (error) {
      console.warn(`Failed to load from localStorage key "${key}":`, error);
    } finally {
      setHasLoaded(true);
    }
  }, [key, isClient, hasLoaded]);

  // Save to localStorage
  const setValue = useCallback((value: T | ((prev: T) => T)) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      
      if (isClient && hasLoaded) {
        localStorage.setItem(key, JSON.stringify(valueToStore));
      }
    } catch (error) {
      console.warn(`Failed to save to localStorage key "${key}":`, error);
    }
  }, [key, storedValue, isClient, hasLoaded]);

  return [storedValue, setValue, hasLoaded];
};