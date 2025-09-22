'use client';

import React, { useState, useEffect } from 'react';
import {
  XMarkIcon,
  ComputerDesktopIcon,
  CpuChipIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline';
import { ExtendedPerformanceSettings } from '../circuit-simulator/types/gpuSettings';
import { DEFAULT_GPU_SETTINGS, DEFAULT_CPU_SETTINGS } from '../circuit-simulator/types/gpuSettings';
import { WebGPUManager } from '../circuit-simulator/utils/webgpuManager';
import { WebGPUCapabilities } from '../circuit-simulator/types/gpuSettings';
import { useAuth } from '../auth/AuthProvider';
import { useUserSettings } from '../../hooks/useUserSettings';
import { ComputationLimitsControl, ComputationLimitsConfig, getDefaultComputationLimits } from '../circuit-simulator/controls/ComputationLimitsControl';
import { CentralizedLimitsManager } from '../circuit-simulator/utils/centralizedLimits';

interface SimplifiedSettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSettingsChange: (settings: ExtendedPerformanceSettings) => void;
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

  const [currentSettings, setCurrentSettings] = useState<ExtendedPerformanceSettings>({
    useSymmetricGrid: true,
    maxComputationResults: getDefaultComputationLimits(gridSize).maxComputationResults,
    gpuAcceleration: DEFAULT_GPU_SETTINGS,
    cpuSettings: DEFAULT_CPU_SETTINGS
  });

  const [computationLimits, setComputationLimits] = useState<ComputationLimitsConfig>(
    centralizedLimits?.toComputationLimitsConfig() ?? getDefaultComputationLimits(gridSize)
  );

  const [webgpuStatus, setWebgpuStatus] = useState<WebGPUCapabilities | null>(null);
  const [isCheckingWebGPU, setIsCheckingWebGPU] = useState(false);

  // Load settings from user profile
  useEffect(() => {
    if (userSettings?.performanceSettings) {
      setCurrentSettings(userSettings.performanceSettings);
    }
  }, [userSettings]);

  // Check WebGPU capabilities
  useEffect(() => {
    if (isOpen && !webgpuStatus && !isCheckingWebGPU) {
      setIsCheckingWebGPU(true);
      const webgpuManager = new WebGPUManager();
      webgpuManager.initialize().then(caps => {
        setWebgpuStatus(caps);
        setIsCheckingWebGPU(false);
      }).catch(() => {
        setWebgpuStatus({
          supported: false,
          adapter: null,
          initTime: 0,
          error: 'WebGPU initialization failed'
        });
        setIsCheckingWebGPU(false);
      });
    }
  }, [isOpen, webgpuStatus, isCheckingWebGPU]);

  const handleSettingsUpdate = (newSettings: Partial<ExtendedPerformanceSettings>) => {
    const updatedSettings = { ...currentSettings, ...newSettings };
    setCurrentSettings(updatedSettings);
    onSettingsChange(updatedSettings);

    if (user && !isLoading) {
      updateSettings({ performanceSettings: updatedSettings });
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

          {/* GPU Settings */}
          <div className="bg-neutral-900 rounded-lg p-4 border border-neutral-700">
            <div className="flex items-center gap-3 mb-4">
              <ComputerDesktopIcon className="w-5 h-5 text-blue-400" />
              <h3 className="text-lg font-medium text-white">GPU Acceleration</h3>
            </div>

            {webgpuStatus && (
              <div className={`flex items-center gap-2 mb-4 p-3 rounded-lg ${
                webgpuStatus.supported ? 'bg-green-900/30 border border-green-700' : 'bg-red-900/30 border border-red-700'
              }`}>
                {webgpuStatus.supported ? (
                  <CheckCircleIcon className="w-5 h-5 text-green-400" />
                ) : (
                  <ExclamationTriangleIcon className="w-5 h-5 text-red-400" />
                )}
                <div>
                  <div className="text-sm font-medium text-white">
                    WebGPU {webgpuStatus.supported ? 'Available' : 'Not Available'}
                  </div>
                  {webgpuStatus.adapter && (
                    <div className="text-xs text-neutral-400">
                      {webgpuStatus.adapter.info?.vendor} {webgpuStatus.adapter.info?.device}
                    </div>
                  )}
                </div>
              </div>
            )}

            <div className="space-y-4">
              <label className="flex items-center gap-3">
                <input
                  type="checkbox"
                  checked={currentSettings.gpuAcceleration?.enabled}
                  onChange={(e) => handleSettingsUpdate({
                    gpuAcceleration: { ...currentSettings.gpuAcceleration, enabled: e.target.checked }
                  })}
                  className="rounded"
                  disabled={!webgpuStatus?.supported}
                />
                <span className="text-sm text-neutral-300">Enable GPU acceleration</span>
              </label>

              {currentSettings.gpuAcceleration?.enabled && (
                <div className="ml-6 space-y-3">
                  <div>
                    <label className="block text-xs text-neutral-400 mb-1">Workgroup Size</label>
                    <select
                      value={currentSettings.gpuAcceleration.workgroupSize}
                      onChange={(e) => handleSettingsUpdate({
                        gpuAcceleration: { ...currentSettings.gpuAcceleration, workgroupSize: parseInt(e.target.value) }
                      })}
                      className="w-full p-2 bg-neutral-800 border border-neutral-600 rounded text-sm"
                    >
                      <option value={64}>64 (Default)</option>
                      <option value={128}>128 (Higher performance)</option>
                      <option value={256}>256 (Max performance)</option>
                    </select>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* CPU Settings */}
          <div className="bg-neutral-900 rounded-lg p-4 border border-neutral-700">
            <div className="flex items-center gap-3 mb-4">
              <CpuChipIcon className="w-5 h-5 text-green-400" />
              <h3 className="text-lg font-medium text-white">CPU Settings</h3>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-xs text-neutral-400 mb-1">
                  Worker Threads ({currentSettings.cpuSettings?.workerCount || 'auto'})
                </label>
                <input
                  type="range"
                  min="1"
                  max={navigator.hardwareConcurrency || 8}
                  value={currentSettings.cpuSettings?.workerCount || navigator.hardwareConcurrency || 4}
                  onChange={(e) => handleSettingsUpdate({
                    cpuSettings: { ...currentSettings.cpuSettings, workerCount: parseInt(e.target.value) }
                  })}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-neutral-500 mt-1">
                  <span>1</span>
                  <span>Auto: {navigator.hardwareConcurrency || 4}</span>
                  <span>{navigator.hardwareConcurrency || 8}</span>
                </div>
              </div>

              <div>
                <label className="block text-xs text-neutral-400 mb-1">
                  Chunk Size ({currentSettings.cpuSettings?.chunkSize.toLocaleString()})
                </label>
                <input
                  type="range"
                  min="100"
                  max="10000"
                  step="100"
                  value={currentSettings.cpuSettings?.chunkSize || 1000}
                  onChange={(e) => handleSettingsUpdate({
                    cpuSettings: { ...currentSettings.cpuSettings, chunkSize: parseInt(e.target.value) }
                  })}
                  className="w-full"
                />
              </div>

              <label className="flex items-start gap-3">
                <input
                  type="checkbox"
                  checked={currentSettings.useSymmetricGrid}
                  onChange={(e) => handleSettingsUpdate({ useSymmetricGrid: e.target.checked })}
                  className="rounded mt-1"
                />
                <div>
                  <span className="text-sm text-neutral-300">Symmetric Grid Optimization</span>
                  <p className="text-xs text-neutral-500 mt-1">
                    Skip τA &gt; τB combinations (~37% reduction). ⚠️ May exclude valid biological parameters.
                  </p>
                </div>
              </label>
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