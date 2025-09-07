'use client';

import React, { useState, useEffect } from 'react';
import { 
  CogIcon,
  XMarkIcon,
  ComputerDesktopIcon,
  CpuChipIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline';
import { ExtendedPerformanceSettings, GPUAccelerationSettings, CPUSettings } from '../circuit-simulator/types/gpuSettings';
import { DEFAULT_GPU_SETTINGS, DEFAULT_CPU_SETTINGS } from '../circuit-simulator/types/gpuSettings';
import { WebGPUManager } from '../circuit-simulator/utils/webgpuManager';
import { WebGPUCapabilities } from '../circuit-simulator/types/gpuSettings';
import { useAuth } from '../auth/AuthProvider';
import { useUserSettings } from '../../hooks/useUserSettings';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSettingsChange: (settings: ExtendedPerformanceSettings) => void;
}

interface TabButtonProps {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}

function TabButton({ active, onClick, icon, label }: TabButtonProps) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center space-x-2 px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
        active 
          ? 'bg-blue-600 text-white' 
          : 'text-neutral-400 hover:text-white hover:bg-neutral-700'
      }`}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}

export function SettingsModal({ isOpen, onClose, onSettingsChange }: SettingsModalProps) {
  const { user } = useAuth();
  const { settings: userSettings, updateSettings, isLoading } = useUserSettings(user?.id);
  const [activeTab, setActiveTab] = useState<'performance' | 'computation' | 'system'>('performance');
  const [currentSettings, setCurrentSettings] = useState<ExtendedPerformanceSettings>({
    useSymmetricGrid: true,
    maxComputationResults: 5000,
    gpuAcceleration: DEFAULT_GPU_SETTINGS,
    cpuSettings: DEFAULT_CPU_SETTINGS
  });
  const [capabilities, setCapabilities] = useState<WebGPUCapabilities | null>(null);
  const [isInitializing, setIsInitializing] = useState(true);
  const [webgpuManager] = useState(() => new WebGPUManager());

  // Load user settings when available
  useEffect(() => {
    if (userSettings?.performance_settings) {
      try {
        const performanceSettings = typeof userSettings.performance_settings === 'string' 
          ? JSON.parse(userSettings.performance_settings)
          : userSettings.performance_settings;
        
        if (performanceSettings.gpuAcceleration || performanceSettings.cpuSettings) {
          setCurrentSettings(prev => ({
            ...prev,
            ...performanceSettings
          }));
        }
      } catch (error) {
        console.warn('Failed to parse user performance settings:', error);
      }
    }
  }, [userSettings]);

  // Initialize WebGPU capabilities
  useEffect(() => {
    const initializeWebGPU = async () => {
      setIsInitializing(true);
      try {
        const caps = await webgpuManager.initialize();
        setCapabilities(caps);
        
        // Auto-enable GPU if supported and discrete GPU detected
        if (caps.supported && caps.deviceType === 'discrete') {
          const newGpuSettings = { ...currentSettings.gpuAcceleration, enabled: true };
          handleSettingsUpdate({
            gpuAcceleration: newGpuSettings
          });
        }
      } catch (error) {
        console.error('Failed to initialize WebGPU:', error);
      } finally {
        setIsInitializing(false);
      }
    };

    if (isOpen) {
      initializeWebGPU();
    }

    return () => {
      webgpuManager.dispose();
    };
  }, [isOpen]);

  const handleSettingsUpdate = async (newSettings: Partial<ExtendedPerformanceSettings>) => {
    const updated = { ...currentSettings, ...newSettings };
    setCurrentSettings(updated);
    onSettingsChange(updated);

    // Save to database if user is authenticated
    if (user && !isLoading) {
      try {
        await updateSettings({
          performance_settings: JSON.stringify(updated)
        });
      } catch (error) {
        console.error('Failed to save settings:', error);
      }
    }
  };

  const handleGpuSettingsChange = (newSettings: Partial<GPUAccelerationSettings>) => {
    handleSettingsUpdate({
      gpuAcceleration: { ...currentSettings.gpuAcceleration, ...newSettings }
    });
  };

  const handleCpuSettingsChange = (newSettings: Partial<CPUSettings>) => {
    handleSettingsUpdate({
      cpuSettings: { ...currentSettings.cpuSettings, ...newSettings }
    });
  };

  const getDeviceTypeBadge = (deviceType: string) => {
    const colors = {
      discrete: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
      integrated: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200', 
      cpu: 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200',
      unknown: 'bg-gray-100 text-gray-600 dark:bg-gray-900 dark:text-gray-400'
    };
    
    const colorClass = colors[deviceType as keyof typeof colors] || colors.unknown;
    
    return (
      <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${colorClass}`}>
        {deviceType.charAt(0).toUpperCase() + deviceType.slice(1)}
      </span>
    );
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-neutral-800 rounded-xl shadow-2xl w-full max-w-4xl max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-neutral-200 dark:border-neutral-700">
          <div className="flex items-center space-x-3">
            <CogIcon className="h-6 w-6 text-neutral-600 dark:text-neutral-400" />
            <h2 className="text-xl font-semibold text-neutral-900 dark:text-white">Settings</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-700 transition-colors"
          >
            <XMarkIcon className="h-5 w-5" />
          </button>
        </div>

        <div className="flex h-[600px]">
          {/* Sidebar Navigation */}
          <div className="w-64 bg-neutral-50 dark:bg-neutral-900 p-4 space-y-2">
            <TabButton
              active={activeTab === 'performance'}
              onClick={() => setActiveTab('performance')}
              icon={<ComputerDesktopIcon className="h-5 w-5" />}
              label="GPU & Performance"
            />
            <TabButton
              active={activeTab === 'computation'}
              onClick={() => setActiveTab('computation')}
              icon={<CpuChipIcon className="h-5 w-5" />}
              label="Computation"
            />
            <TabButton
              active={activeTab === 'system'}
              onClick={() => setActiveTab('system')}
              icon={<ChartBarIcon className="h-5 w-5" />}
              label="System Info"
            />
          </div>

          {/* Main Content */}
          <div className="flex-1 p-6 overflow-y-auto">
            {activeTab === 'performance' && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-medium text-neutral-900 dark:text-white mb-4">GPU Acceleration</h3>
                  
                  {/* GPU Status */}
                  <div className="bg-neutral-50 dark:bg-neutral-800 rounded-lg p-4 mb-4">
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-sm font-medium">WebGPU Status</span>
                      {isInitializing ? (
                        <span className="text-sm text-neutral-500">Detecting...</span>
                      ) : capabilities?.supported ? (
                        <span className="text-sm text-green-600 dark:text-green-400">Available</span>
                      ) : (
                        <span className="text-sm text-red-600 dark:text-red-400">Not Available</span>
                      )}
                    </div>
                    
                    {capabilities?.supported && (
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="text-neutral-600 dark:text-neutral-400">Device Type:</span>
                          <div className="mt-1">{getDeviceTypeBadge(capabilities.deviceType)}</div>
                        </div>
                        <div>
                          <span className="text-neutral-600 dark:text-neutral-400">Vendor:</span>
                          <div className="mt-1 font-medium">{capabilities.vendor}</div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* GPU Settings */}
                  {capabilities?.supported && (
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <label className="text-sm font-medium text-neutral-900 dark:text-white">
                            Enable GPU Acceleration
                          </label>
                          <p className="text-sm text-neutral-600 dark:text-neutral-400">
                            Use WebGPU for parallel computation
                          </p>
                        </div>
                        <input
                          type="checkbox"
                          checked={currentSettings.gpuAcceleration.enabled}
                          onChange={(e) => handleGpuSettingsChange({ enabled: e.target.checked })}
                          className="h-4 w-4 rounded border-neutral-300 text-blue-600 focus:ring-blue-500"
                        />
                      </div>

                      {currentSettings.gpuAcceleration.enabled && (
                        <div className="ml-4 space-y-4 border-l-2 border-blue-200 dark:border-blue-800 pl-4">
                          <div>
                            <label className="text-sm font-medium text-neutral-900 dark:text-white">
                              Batch Size: {currentSettings.gpuAcceleration.maxBatchSize.toLocaleString()}
                            </label>
                            <input
                              type="range"
                              min="1024"
                              max="131072"
                              step="1024"
                              value={currentSettings.gpuAcceleration.maxBatchSize}
                              onChange={(e) => handleGpuSettingsChange({ maxBatchSize: parseInt(e.target.value) })}
                              className="w-full mt-2"
                            />
                            <p className="text-xs text-neutral-600 dark:text-neutral-400">
                              Larger batches use more memory but may be faster
                            </p>
                          </div>

                          <div>
                            <label className="text-sm font-medium text-neutral-900 dark:text-white">
                              Memory Limit: {currentSettings.gpuAcceleration.memoryThreshold}MB
                            </label>
                            <input
                              type="range"
                              min="256"
                              max="4096"
                              step="256"
                              value={currentSettings.gpuAcceleration.memoryThreshold}
                              onChange={(e) => handleGpuSettingsChange({ memoryThreshold: parseInt(e.target.value) })}
                              className="w-full mt-2"
                            />
                          </div>

                          <div className="flex items-center justify-between">
                            <label className="text-sm font-medium text-neutral-900 dark:text-white">
                              Fallback to CPU on Error
                            </label>
                            <input
                              type="checkbox"
                              checked={currentSettings.gpuAcceleration.fallbackToCPU}
                              onChange={(e) => handleGpuSettingsChange({ fallbackToCPU: e.target.checked })}
                              className="h-4 w-4 rounded border-neutral-300 text-blue-600 focus:ring-blue-500"
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {!capabilities?.supported && !isInitializing && (
                    <div className="bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-lg p-4">
                      <p className="text-sm text-orange-800 dark:text-orange-200">
                        WebGPU is not available. Computation will use Web Workers (CPU).
                      </p>
                    </div>
                  )}
                </div>

                {/* CPU Settings */}
                <div>
                  <h3 className="text-lg font-medium text-neutral-900 dark:text-white mb-4">CPU Settings</h3>
                  
                  <div className="space-y-4">
                    <div>
                      <label className="text-sm font-medium text-neutral-900 dark:text-white">
                        Worker Threads: {currentSettings.cpuSettings.maxWorkers}
                      </label>
                      <input
                        type="range"
                        min="1"
                        max={navigator.hardwareConcurrency || 8}
                        step="1"
                        value={currentSettings.cpuSettings.maxWorkers}
                        onChange={(e) => handleCpuSettingsChange({ maxWorkers: parseInt(e.target.value) })}
                        className="w-full mt-2"
                      />
                      <p className="text-xs text-neutral-600 dark:text-neutral-400">
                        Available cores: {navigator.hardwareConcurrency || 'Unknown'}
                      </p>
                    </div>

                    <div>
                      <label className="text-sm font-medium text-neutral-900 dark:text-white">
                        Chunk Size: {currentSettings.cpuSettings.chunkSize.toLocaleString()}
                      </label>
                      <input
                        type="range"
                        min="1000"
                        max="20000"
                        step="1000"
                        value={currentSettings.cpuSettings.chunkSize}
                        onChange={(e) => handleCpuSettingsChange({ chunkSize: parseInt(e.target.value) })}
                        className="w-full mt-2"
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'computation' && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-medium text-neutral-900 dark:text-white mb-4">General Settings</h3>
                  
                  <div className="space-y-4">
                    <div>
                      <label className="text-sm font-medium text-neutral-900 dark:text-white">
                        Max Results: {currentSettings.maxComputationResults.toLocaleString()}
                      </label>
                      <input
                        type="range"
                        min="1000"
                        max="50000"
                        step="1000"
                        value={currentSettings.maxComputationResults}
                        onChange={(e) => handleSettingsUpdate({ maxComputationResults: parseInt(e.target.value) })}
                        className="w-full mt-2"
                      />
                      <div className="flex justify-between text-xs text-neutral-500 mt-1">
                        <span>1K</span>
                        <span>50K</span>
                      </div>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div>
                        <label className="text-sm font-medium text-neutral-900 dark:text-white">Symmetric Grid</label>
                        <p className="text-sm text-neutral-600 dark:text-neutral-400">
                          Use symmetric parameter sampling
                        </p>
                      </div>
                      <input
                        type="checkbox"
                        checked={currentSettings.useSymmetricGrid}
                        onChange={(e) => handleSettingsUpdate({ useSymmetricGrid: e.target.checked })}
                        className="h-4 w-4 rounded border-neutral-300 text-blue-600 focus:ring-blue-500"
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'system' && (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-medium text-neutral-900 dark:text-white mb-4">System Information</h3>
                  
                  <div className="bg-neutral-50 dark:bg-neutral-800 rounded-lg p-4 space-y-3 text-sm">
                    <div className="flex justify-between">
                      <span className="text-neutral-600 dark:text-neutral-400">CPU Cores:</span>
                      <span className="font-mono">{navigator.hardwareConcurrency || 'Unknown'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-neutral-600 dark:text-neutral-400">Memory Usage:</span>
                      <span className="font-mono">
                        {(performance as unknown as { memory?: { usedJSHeapSize?: number } })?.memory?.usedJSHeapSize ? 
                         `${Math.round((performance as unknown as { memory: { usedJSHeapSize: number } }).memory.usedJSHeapSize / 1024 / 1024)}MB` : 
                         'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-neutral-600 dark:text-neutral-400">WebGPU Support:</span>
                      <span className={`font-medium ${
                        capabilities?.supported ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                      }`}>
                        {capabilities?.supported ? 'Supported' : 'Not Available'}
                      </span>
                    </div>
                    {user && (
                      <div className="flex justify-between">
                        <span className="text-neutral-600 dark:text-neutral-400">Settings Sync:</span>
                        <span className="text-green-600 dark:text-green-400 font-medium">
                          {isLoading ? 'Syncing...' : 'Active'}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="border-t border-neutral-200 dark:border-neutral-700 p-4">
          <div className="text-xs text-neutral-500 text-center">
            Settings are automatically saved{user ? ' to your account' : ' locally'}
          </div>
        </div>
      </div>
    </div>
  );
}