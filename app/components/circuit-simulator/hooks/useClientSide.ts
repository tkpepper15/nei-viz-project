import { useState, useEffect } from 'react';

/**
 * Custom hook to handle client-side hydration and prevent SSR mismatches
 * Returns true only after the component has hydrated on the client side
 */
export const useClientSide = () => {
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    // This effect only runs on the client side after hydration
    setIsClient(true);
  }, []);

  return isClient;
};