"use client";

import React from 'react';

interface ComputationNotificationProps {
  showNotification: boolean;
  computationLogs: string[];
  isLoadingMesh: boolean;
  setShowNotification: (show: boolean) => void;
}

export const ComputationNotification: React.FC<ComputationNotificationProps> = ({
  showNotification,
  computationLogs,
  isLoadingMesh,
  setShowNotification
}) => {
  if (!showNotification || computationLogs.length === 0) return null;

  return (
    <div className="fixed bottom-4 right-4 z-50 bg-white shadow-lg border border-gray-200 rounded-md p-3 max-w-xs">
      <div className="flex justify-between items-center mb-1">
        <h3 className="font-medium text-sm text-gray-700">Computation Logs</h3>
        <button 
          onClick={() => setShowNotification(false)}
          className="text-gray-400 hover:text-gray-600 text-sm"
        >
          ×
        </button>
      </div>
      <div className="max-h-32 overflow-y-auto text-xs text-gray-600">
        {computationLogs.slice(-3).map((log, i) => (
          <div key={i} className="py-1">
            {log}
          </div>
        ))}
      </div>
      {isLoadingMesh && (
        <div className="mt-1 flex items-center text-xs text-blue-500">
          <svg className="animate-spin -ml-1 mr-2 h-3 w-3 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          Processing...
        </div>
      )}
    </div>
  );
};

export default ComputationNotification; 