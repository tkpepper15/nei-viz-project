import { useCallback } from 'react';
import { useClientSide } from './useClientSide';

/**
 * Custom hook for generating unique IDs safely on client-side only
 */
export const useIdGeneration = () => {
  const isClient = useClientSide();

  const generateUniqueId = useCallback((prefix: string = 'id'): string => {
    if (!isClient) {
      // Return a deterministic fallback ID for SSR
      return `${prefix}_ssr_fallback`;
    }

    // Generate client-side ID
    const timestamp = Date.now();
    const random = Math.random().toString(36).substr(2, 9);
    return `${prefix}_${timestamp}_${random}`;
  }, [isClient]);

  const generateDeterministicNoise = useCallback((
    seed: number,
    amplitude: number = 0.15
  ): number => {
    // Use deterministic function for consistent results across renders
    return 1 + (Math.sin(seed * 0.01234) * amplitude);
  }, []);

  return {
    generateUniqueId,
    generateDeterministicNoise,
    isClient
  };
};