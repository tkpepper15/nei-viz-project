"use client";

import React, { useState, useEffect } from 'react';

export interface ComputationSummary {
  title: string;
  totalTime: string;
  generationTime: string;
  computationTime: string;
  processingTime: string;
  totalPoints: number;
  validPoints: number;
  groups: number;
  cores: number;
  throughput: number;
  type: 'success' | 'error';
  duration?: number;
}

interface ComputationNotificationProps {
  summary: ComputationSummary | null;
  onDismiss: () => void;
}

export const ComputationNotification: React.FC<ComputationNotificationProps> = ({
  summary,
  onDismiss
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [isExiting, setIsExiting] = useState(false);

  useEffect(() => {
    if (summary) {
      setIsVisible(true);
      setIsExiting(false);
      
      // Auto-dismiss after specified duration
      const timer = setTimeout(() => {
        handleDismiss();
      }, summary.duration || 6000);

      return () => clearTimeout(timer);
    }
  }, [summary]);

  const handleDismiss = () => {
    setIsExiting(true);
    setTimeout(() => {
      setIsVisible(false);
      onDismiss();
    }, 300); // Animation duration
  };

  if (!summary || !isVisible) return null;

  const bgColor = summary.type === 'success' ? 'bg-green-900/90' : 'bg-red-900/90';
  const borderColor = summary.type === 'success' ? 'border-green-500' : 'border-red-500';
  const iconColor = summary.type === 'success' ? 'text-green-400' : 'text-red-400';

  return (
    <div className="fixed bottom-4 right-4 z-50">
      <div
        className={`
          ${bgColor} ${borderColor} border rounded-lg shadow-xl p-4 min-w-[400px] max-w-[500px]
          transform transition-all duration-300 ease-in-out
          ${isExiting ? 'translate-y-full opacity-0' : 'translate-y-0 opacity-100'}
        `}
      >
        {/* Header */}
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-2">
            {summary.type === 'success' ? (
              <svg className={`w-5 h-5 ${iconColor}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            ) : (
              <svg className={`w-5 h-5 ${iconColor}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            )}
            <h3 className="text-white font-semibold text-sm">{summary.title}</h3>
          </div>
          <button
            onClick={handleDismiss}
            className="text-neutral-400 hover:text-white transition-colors p-1"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="space-y-3">
          {/* Overall timing */}
          <div className="bg-black/20 rounded p-3">
            <div className="text-white font-medium text-lg mb-2">
              Total Time: {summary.totalTime}s
            </div>
            <div className="text-xs text-neutral-300">
              Processed {summary.totalPoints.toLocaleString()} parameter combinations
            </div>
          </div>

          {/* Phase breakdown */}
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="bg-blue-900/40 rounded p-2">
              <div className="text-blue-300 font-medium">Generation</div>
              <div className="text-white">{summary.generationTime}s</div>
            </div>
            <div className="bg-purple-900/40 rounded p-2">
              <div className="text-purple-300 font-medium">Computation</div>
              <div className="text-white">{summary.computationTime}s</div>
            </div>
            <div className="bg-orange-900/40 rounded p-2">
              <div className="text-orange-300 font-medium">Processing</div>
              <div className="text-white">{summary.processingTime}s</div>
            </div>
          </div>

          {/* Performance metrics */}
          <div className="flex justify-between items-center text-xs text-neutral-300">
            <div>
              <span className="text-white font-medium">{summary.throughput.toFixed(0)}</span> points/sec
            </div>
            <div>
              <span className="text-white font-medium">{summary.cores}</span> CPU cores
            </div>
            <div>
              <span className="text-white font-medium">{summary.groups}</span> result groups
            </div>
          </div>

          {/* Results summary */}
          <div className="bg-neutral-800/50 rounded p-2 text-xs">
            <div className="text-neutral-300">
              {summary.type === 'success' ? (
                <>
                  Results: <span className="text-white font-medium">{summary.validPoints}</span> valid points 
                  in <span className="text-white font-medium">{summary.groups}</span> resnorm groups
                </>
              ) : (
                <>
                  <span className="text-red-300">Computation failed</span> - 
                  check activity log for details
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ComputationNotification; 