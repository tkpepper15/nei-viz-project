/**
 * Model Info Button Component
 * ===========================
 *
 * Button to open the Model Info modal.
 * Styled to match the Settings button.
 */

'use client';

import React from 'react';

interface ModelInfoButtonProps {
  onClick: () => void;
  className?: string;
}

export const ModelInfoButton: React.FC<ModelInfoButtonProps> = ({
  onClick,
  className = ''
}) => {
  return (
    <button
      onClick={onClick}
      className={`px-2 py-1 bg-neutral-800 hover:bg-neutral-700 rounded text-neutral-400 hover:text-neutral-200 transition-colors flex items-center ${className}`}
      title="Circuit Model Info"
    >
      <svg className="w-4 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    </button>
  );
};

export default ModelInfoButton;
