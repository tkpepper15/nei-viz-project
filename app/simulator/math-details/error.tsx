'use client';

import { useEffect } from 'react';

export default function MathDetailsError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('Math details error:', error);
  }, [error]);

  return (
    <div className="h-screen flex items-center justify-center bg-neutral-950">
      <div className="max-w-md text-center">
        <h2 className="text-2xl font-bold text-red-400 mb-4">Math Details Error</h2>
        <p className="text-neutral-400 mb-6">{error.message}</p>
        <button
          onClick={reset}
          className="px-6 py-3 bg-orange-600 hover:bg-orange-700 rounded transition-colors"
        >
          Try again
        </button>
      </div>
    </div>
  );
}
