'use client';

import React, { useState, useEffect } from 'react';
import {
  XMarkIcon,
  CpuChipIcon
} from '@heroicons/react/24/outline';
import { useAuth } from '../auth/AuthProvider';
import { useUserSettings } from '../../hooks/useUserSettings';
import { ComputationLimitsControl, ComputationLimitsConfig, getDefaultComputationLimits } from '../circuit-simulator/controls/ComputationLimitsControl';
import { CentralizedLimitsManager } from '../circuit-simulator/utils/centralizedLimits';

// Simplified settings interface without GPU complexity
interface SimplePerformanceSettings {
  useSymmetricGrid: boolean;
  maxComputationResults: number;
}

interface SimplifiedSettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSettingsChange: (settings: SimplePerformanceSettings) => void;
  gridSize: number;
  totalPossibleResults: number;
  isComputing?: boolean;
  centralizedLimits?: CentralizedLimitsManager;
  onMasterLimitChange?: (percentage: number) => void;
}

const SimplifiedSettingsModal: React.FC<SimplifiedSettingsModalProps> = ({
  isOpen,
  onClose,
  onSettingsChange,
  gridSize,
  totalPossibleResults,
  isComputing = false,
  centralizedLimits,
  onMasterLimitChange
}) => {
  const { user } = useAuth();
  const { settings: userSettings, updateSettings, isLoading } = useUserSettings(user?.id);

  const [currentSettings, setCurrentSettings] = useState<SimplePerformanceSettings>({
    useSymmetricGrid: true,
    maxComputationResults: getDefaultComputationLimits(gridSize).maxComputationResults
  });

  const [computationLimits, setComputationLimits] = useState<ComputationLimitsConfig>(
    centralizedLimits?.toComputationLimitsConfig() ?? getDefaultComputationLimits(gridSize)
  );

  // Load settings from user profile
  useEffect(() => {
    if (userSettings?.performance_settings && typeof userSettings.performance_settings === 'object') {
      // Extract only the simple settings we need
      const simpleSettings: SimplePerformanceSettings = {
        useSymmetricGrid: userSettings.performance_settings.useSymmetricGrid ?? true,
        maxComputationResults: userSettings.performance_settings.maxComputationResults ?? getDefaultComputationLimits(gridSize).maxComputationResults
      };
      setCurrentSettings(simpleSettings);
    }
  }, [userSettings, gridSize]);

  const handleSettingsUpdate = (newSettings: Partial<SimplePerformanceSettings>) => {
    const updatedSettings = { ...currentSettings, ...newSettings };
    setCurrentSettings(updatedSettings);
    onSettingsChange(updatedSettings);

    if (user && !isLoading) {
      updateSettings({ performance_settings: updatedSettings });
    }
  };

  const handleComputationLimitsChange = (config: ComputationLimitsConfig) => {
    setComputationLimits(config);
    handleSettingsUpdate({
      maxComputationResults: config.maxComputationResults
    });

    // If master limit percentage changed and we have the callback, update the centralized limits
    if (onMasterLimitChange && config.masterLimitPercentage !== computationLimits.masterLimitPercentage) {
      onMasterLimitChange(config.masterLimitPercentage);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-neutral-800 rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-hidden border border-neutral-700">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-neutral-700 bg-neutral-900">
          <h2 className="text-xl font-semibold text-white">Performance Settings</h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-neutral-700 rounded-lg transition-colors"
            disabled={isComputing}
          >
            <XMarkIcon className="w-5 h-5 text-neutral-400" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6 overflow-y-auto max-h-[70vh]">

          {/* Computation Limits */}
          <ComputationLimitsControl
            gridSize={gridSize}
            totalPossibleResults={totalPossibleResults}
            currentConfig={computationLimits}
            onConfigChange={handleComputationLimitsChange}
          />

          {/* Optimization Settings */}
          <div className="bg-neutral-900 rounded-lg p-4 border border-neutral-700">
            <div className="flex items-center gap-3 mb-4">
              <CpuChipIcon className="w-5 h-5 text-green-400" />
              <h3 className="text-lg font-medium text-white">Optimization</h3>
            </div>

            <div className="space-y-4">
              <label className="flex items-center gap-3">
                <input
                  type="checkbox"
                  checked={currentSettings.useSymmetricGrid}
                  onChange={(e) => handleSettingsUpdate({
                    useSymmetricGrid: e.target.checked
                  })}
                  className="rounded"
                />
                <div>
                  <span className="text-sm text-neutral-300">Use symmetric grid optimization</span>
                  <div className="text-xs text-neutral-500">Reduces computation time for symmetric circuit parameters</div>
                </div>
              </label>
            </div>
          </div>

          {/* Performance Settings */}
          <div className="bg-neutral-900 rounded-lg p-4 border border-neutral-700">
            <div className="flex items-center gap-3 mb-4">
              <CpuChipIcon className="w-5 h-5 text-blue-400" />
              <h3 className="text-lg font-medium text-white">Performance</h3>
            </div>

            <div className="space-y-4">
              <div className="text-sm text-neutral-300">
                <div className="mb-2">Available CPU cores: <span className="text-white font-medium">{navigator.hardwareConcurrency || 'Unknown'}</span></div>
                <div className="text-xs text-neutral-500">
                  Computations will automatically use all available cores for optimal performance.
                </div>
              </div>
            </div>
          </div>

          {/* System Info */}
          <div className="bg-neutral-900 rounded-lg p-4 border border-neutral-700">
            <h3 className="text-lg font-medium text-white mb-3">System Information</h3>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-neutral-400">CPU Cores:</span>
                <span className="text-neutral-200 ml-2">{navigator.hardwareConcurrency || 'Unknown'}</span>
              </div>
              <div>
                <span className="text-neutral-400">Memory:</span>
                <span className="text-neutral-200 ml-2">
                  {navigator.deviceMemory ? `~${navigator.deviceMemory}GB` : 'Unknown'}
                </span>
              </div>
              <div>
                <span className="text-neutral-400">User Agent:</span>
                <span className="text-neutral-200 ml-2 text-xs">{navigator.userAgent.split(' ')[0]}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-neutral-700 bg-neutral-900">
          <div className="flex items-center justify-between">
            <div className="text-xs text-neutral-400">
              Settings auto-save{user ? ' to your account' : ' locally'}
            </div>
            <button
              onClick={onClose}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors text-sm"
              disabled={isComputing}
            >
              Done
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SimplifiedSettingsModal;