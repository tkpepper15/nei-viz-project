'use client';

import React from 'react';

interface SlimNavbarProps {
  gridResults?: unknown[];
  activeCircuitName?: string;
}

export const SlimNavbar: React.FC<SlimNavbarProps> = ({
  gridResults = [],
  activeCircuitName,
}) => {
  const computedCount = gridResults.length;

  return (
    <div className="h-11 px-4 flex items-center flex-shrink-0 relative">
      <span className="text-[#dddde2] font-semibold tracking-tight text-sm select-none">
        SpideyPlot
      </span>

      {(activeCircuitName || computedCount > 0) && (
        <div className="flex items-center gap-2 text-xs absolute left-1/2 -translate-x-1/2">
          {activeCircuitName && (
            <span className="text-[#9a9aa2] truncate max-w-40">{activeCircuitName}</span>
          )}
          {computedCount > 0 && (
            <>
              {activeCircuitName && <span className="text-[#2a2a33]">·</span>}
              <span className="text-[#454549] font-mono tabular-nums">
                {computedCount.toLocaleString()} models
              </span>
            </>
          )}
        </div>
      )}
    </div>
  );
};
