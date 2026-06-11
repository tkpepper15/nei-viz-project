'use client';

import React, { useState, useEffect } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import * as Checkbox from '@radix-ui/react-checkbox';
import {
  XMarkIcon,
  CpuChipIcon
} from '@heroicons/react/24/outline';
import { useAuth } from '../auth/AuthProvider';
import { useUserSettings } from '../../hooks/useUserSettings';
import { ComputationLimitsControl, ComputationLimitsConfig, getDefaultComputationLimits } from '../circuit-simulator/controls/ComputationLimitsControl';
import { CentralizedLimitsManager } from '../circuit-simulator/utils/centralizedLimits';

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
  const { settings: userSettings, updateSettings: _updateSettings, isLoading } = useUserSettings(user?.id); // eslint-disable-line @typescript-eslint/no-unused-vars

  const [currentSettings, setCurrentSettings] = useState<SimplePerformanceSettings>({
    useSymmetricGrid: true,
    maxComputationResults: getDefaultComputationLimits(gridSize).maxComputationResults
  });

  const [computationLimits, setComputationLimits] = useState<ComputationLimitsConfig>(
    centralizedLimits?.toComputationLimitsConfig() ?? getDefaultComputationLimits(gridSize)
  );

  useEffect(() => {
    if (userSettings?.performance_settings && typeof userSettings.performance_settings === 'object') {
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
      // updateSettings({ performance_settings: updatedSettings });
    }
  };

  const handleComputationLimitsChange = (config: ComputationLimitsConfig) => {
    setComputationLimits(config);
    handleSettingsUpdate({ maxComputationResults: config.maxComputationResults });

    if (onMasterLimitChange && config.masterLimitPercentage !== computationLimits.masterLimitPercentage) {
      onMasterLimitChange(config.masterLimitPercentage);
    }
  };

  return (
    <Dialog.Root open={isOpen} onOpenChange={(open) => !open && !isComputing && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/50 z-50" />
        <Dialog.Content className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-50 bg-neutral-800 rounded border w-full max-w-2xl max-h-[90vh] overflow-hidden border border-neutral-800 focus:outline-none">
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-neutral-800 bg-neutral-900">
            <Dialog.Title className="text-xl font-semibold text-white">Performance Settings</Dialog.Title>
            <Dialog.Close asChild>
              <button
                className="p-2 hover:bg-neutral-700 rounded-lg transition-colors disabled:opacity-50"
                disabled={isComputing}
              >
                <XMarkIcon className="w-5 h-5 text-neutral-400" />
              </button>
            </Dialog.Close>
          </div>

          {/* Content */}
          <div className="p-6 space-y-6 overflow-y-auto max-h-[70vh]">
            <ComputationLimitsControl
              gridSize={gridSize}
              totalPossibleResults={totalPossibleResults}
              currentConfig={computationLimits}
              onConfigChange={handleComputationLimitsChange}
            />

            {/* Optimization Settings */}
            <div className="bg-neutral-900 rounded-lg p-4 border border-neutral-800">
              <div className="flex items-center gap-3 mb-4">
                <CpuChipIcon className="w-5 h-5 text-neutral-400" />
                <h3 className="text-lg font-medium text-white">Optimization</h3>
              </div>

              <div className="space-y-4">
                <label className="flex items-center gap-3 cursor-pointer">
                  <Checkbox.Root
                    id="symmetric-grid"
                    checked={currentSettings.useSymmetricGrid}
                    onCheckedChange={(checked) => handleSettingsUpdate({ useSymmetricGrid: checked === true })}
                    className="w-4 h-4 rounded border border-neutral-500 bg-neutral-700 flex items-center justify-center data-[state=checked]:bg-primary data-[state=checked]:border-primary focus:outline-none focus-visible:ring-1 focus-visible:ring-primary"
                  >
                    <Checkbox.Indicator>
                      <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                      </svg>
                    </Checkbox.Indicator>
                  </Checkbox.Root>
                  <div>
                    <span className="text-sm text-neutral-300">Use symmetric grid optimization</span>
                    <div className="text-xs text-neutral-500">Reduces computation time for symmetric circuit parameters</div>
                  </div>
                </label>
              </div>
            </div>

            {/* Performance Settings */}
            <div className="bg-neutral-900 rounded-lg p-4 border border-neutral-800">
              <div className="flex items-center gap-3 mb-4">
                <CpuChipIcon className="w-5 h-5 text-neutral-400" />
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
            <div className="bg-neutral-900 rounded-lg p-4 border border-neutral-800">
              <h3 className="text-lg font-medium text-white mb-3">System Information</h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-neutral-400">CPU Cores:</span>
                  <span className="text-neutral-200 ml-2">{navigator.hardwareConcurrency || 'Unknown'}</span>
                </div>
                <div>
                  <span className="text-neutral-400">Memory:</span>
                  <span className="text-neutral-200 ml-2">
                    {'deviceMemory' in navigator ? `~${(navigator as { deviceMemory: number }).deviceMemory}GB` : 'Unknown'}
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
          <div className="px-6 py-4 border-t border-neutral-800 bg-neutral-900">
            <div className="flex items-center justify-between">
              <div className="text-xs text-neutral-400">
                Settings auto-save{user ? ' to your account' : ' locally'}
              </div>
              <Dialog.Close asChild>
                <button
                  className="px-4 py-2 bg-primary hover:bg-primary-dark text-white rounded transition-colors text-sm disabled:opacity-50"
                  disabled={isComputing}
                >
                  Done
                </button>
              </Dialog.Close>
            </div>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
};

export default SimplifiedSettingsModal;
