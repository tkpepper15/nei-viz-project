'use client';

import React, { useState, useEffect } from 'react';
import { 
  CogIcon,
  XMarkIcon,
  ComputerDesktopIcon,
  CpuChipIcon,
  ChartBarIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  RocketLaunchIcon,
  CircleStackIcon
} from '@heroicons/react/24/outline';
import { ExtendedPerformanceSettings, GPUAccelerationSettings, CPUSettings } from '../circuit-simulator/types/gpuSettings';
import { DEFAULT_GPU_SETTINGS, DEFAULT_CPU_SETTINGS } from '../circuit-simulator/types/gpuSettings';
import { WebGPUManager } from '../circuit-simulator/utils/webgpuManager';
import { WebGPUCapabilities } from '../circuit-simulator/types/gpuSettings';
import { useAuth } from '../auth/AuthProvider';
import { useUserSettings } from '../../hooks/useUserSettings';
import { OptimizationControls } from '../circuit-simulator/controls/OptimizationControls';
import { OptimizedComputeManagerConfig } from '../circuit-simulator/utils/optimizedComputeManager';
import { NPZDatasetManager } from '../npz/NPZDatasetManager';
import styles from './SettingsModal.module.css';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSettingsChange: (settings: ExtendedPerformanceSettings) => void;
  optimizationConfig?: OptimizedComputeManagerConfig;
  onOptimizationConfigChange?: (config: Partial<OptimizedComputeManagerConfig>) => void;
  optimizationStats?: {
    lastRun?: {
      timestamp: number;
      gridSize: number;
      totalParams: number;
      finalCandidates: number;
      reductionRatio: number;
      processingTime: number;
      stageBreakdown: {
        stage1: number;
        stage2: number;
        stage3: number;
      };
    };
  };
  isComputing?: boolean;
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
      className={`flex items-center space-x-3 px-4 py-3 text-sm font-medium rounded-xl transition-all duration-200 w-full ${
        active 
          ? 'bg-blue-600 text-white shadow-lg' 
          : 'text-neutral-300 hover:text-white hover:bg-neutral-700'
      }`}
    >
      <div className="w-5 h-5">{icon}</div>
      <span>{label}</span>
    </button>
  );
}

function SystemSpecs() {
  const [gpuInfo, setGpuInfo] = useState<string>('Detecting...');
  const [memoryInfo, setMemoryInfo] = useState<string>('N/A');
  const [platformInfo, setPlatformInfo] = useState<string>('Detecting...');
  const [architectureInfo, setArchitectureInfo] = useState<string>('Detecting...');
  const [browserInfo, setBrowserInfo] = useState<string>('Detecting...');
  
  useEffect(() => {
    const detectSystemInfo = async () => {
      // Get GPU info
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl') as WebGLRenderingContext | null;
      if (gl) {
        const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
        if (debugInfo) {
          const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
          const vendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL);
          setGpuInfo(renderer ? `${renderer}` : (vendor || 'WebGL Available'));
        } else {
          setGpuInfo('WebGL Available');
        }
      } else {
        setGpuInfo('No WebGL Support');
      }
      
      // Get memory info
      const perfMemory = (performance as unknown as { memory?: { usedJSHeapSize?: number; totalJSHeapSize?: number } })?.memory;
      if (perfMemory && perfMemory.usedJSHeapSize && perfMemory.totalJSHeapSize) {
        const usedMB = Math.round(perfMemory.usedJSHeapSize / 1024 / 1024);
        const totalMB = Math.round(perfMemory.totalJSHeapSize / 1024 / 1024);
        setMemoryInfo(`${usedMB}MB / ${totalMB}MB`);
      }

      // Detect platform and architecture more accurately
      const detectPlatformAndArch = () => {
        const userAgent = navigator.userAgent;
        const platform = navigator.platform;
        
        // Detect macOS and architecture
        if (userAgent.includes('Mac OS X') || userAgent.includes('macOS') || platform.includes('Mac')) {
          setPlatformInfo('macOS');
          
          // Multiple methods to detect Apple Silicon
          let isAppleSilicon = false;
          
          // Method 1: Check WebGL renderer
          const canvas = document.createElement('canvas');
          const gl = canvas.getContext('webgl') as WebGLRenderingContext;
          if (gl) {
            const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
            if (debugInfo) {
              const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
              if (renderer && (renderer.includes('Apple M') || renderer.includes('Apple GPU') || renderer.includes('Apple'))) {
                isAppleSilicon = true;
              }
            }
          }
          
          // Method 2: Check CPU cores (M1 has 8, M2 has 8, M3 varies but usually 8+)
          const cores = navigator.hardwareConcurrency;
          if (!isAppleSilicon && cores && cores >= 8) {
            // Additional heuristic: Apple Silicon usually has high core count
            // Combined with other factors
            const hasHighCoreCount = cores >= 8;
            const isModernUserAgent = userAgent.includes('Version/1') || userAgent.includes('Safari/6'); // Recent Safari versions
            if (hasHighCoreCount && isModernUserAgent) {
              isAppleSilicon = true;
            }
          }
          
          // Method 3: Check for absence of Intel-specific indicators
          if (!isAppleSilicon) {
            const noIntelIndicators = !userAgent.includes('Intel') && 
                                    !platform.includes('Intel') &&
                                    !userAgent.includes('x86');
            if (noIntelIndicators && cores && cores >= 8) {
              isAppleSilicon = true;
            }
          }
          
          if (isAppleSilicon) {
            setArchitectureInfo('Apple Silicon (ARM64)');
          } else {
            // Likely Intel Mac
            setArchitectureInfo('Intel x64');
          }
        }
        // Windows detection
        else if (userAgent.includes('Windows') || platform.includes('Win')) {
          setPlatformInfo('Windows');
          if (userAgent.includes('WOW64') || userAgent.includes('x64')) {
            setArchitectureInfo('x64');
          } else if (userAgent.includes('ARM64')) {
            setArchitectureInfo('ARM64');
          } else {
            setArchitectureInfo('x86/x64');
          }
        }
        // Linux detection
        else if (userAgent.includes('Linux') || platform.includes('Linux')) {
          setPlatformInfo('Linux');
          if (userAgent.includes('x86_64') || userAgent.includes('amd64')) {
            setArchitectureInfo('x64');
          } else if (userAgent.includes('aarch64') || userAgent.includes('arm64')) {
            setArchitectureInfo('ARM64');
          } else if (userAgent.includes('armv7') || userAgent.includes('armv6')) {
            setArchitectureInfo('ARM32');
          } else {
            setArchitectureInfo('Unknown');
          }
        }
        // Mobile platforms
        else if (userAgent.includes('iPhone') || userAgent.includes('iPad')) {
          setPlatformInfo('iOS');
          setArchitectureInfo('ARM64');
        }
        else if (userAgent.includes('Android')) {
          setPlatformInfo('Android');
          if (userAgent.includes('arm64') || userAgent.includes('aarch64')) {
            setArchitectureInfo('ARM64');
          } else if (userAgent.includes('armv7')) {
            setArchitectureInfo('ARM32');
          } else {
            setArchitectureInfo('ARM');
          }
        }
        // Fallback
        else {
          setPlatformInfo(platform || 'Unknown');
          setArchitectureInfo('Unknown');
        }
      };

      detectPlatformAndArch();

      // Detect browser info
      const detectBrowser = () => {
        const userAgent = navigator.userAgent;
        if (userAgent.includes('Chrome') && userAgent.includes('Safari')) {
          const chromeMatch = userAgent.match(/Chrome\/(\d+)/);
          const version = chromeMatch ? chromeMatch[1] : 'Unknown';
          setBrowserInfo(`Chrome ${version}`);
        } else if (userAgent.includes('Firefox')) {
          const firefoxMatch = userAgent.match(/Firefox\/(\d+)/);
          const version = firefoxMatch ? firefoxMatch[1] : 'Unknown';
          setBrowserInfo(`Firefox ${version}`);
        } else if (userAgent.includes('Safari') && !userAgent.includes('Chrome')) {
          const safariMatch = userAgent.match(/Version\/(\d+)/);
          const version = safariMatch ? safariMatch[1] : 'Unknown';
          setBrowserInfo(`Safari ${version}`);
        } else if (userAgent.includes('Edge')) {
          const edgeMatch = userAgent.match(/Edge?\/(\d+)/);
          const version = edgeMatch ? edgeMatch[1] : 'Unknown';
          setBrowserInfo(`Edge ${version}`);
        } else {
          setBrowserInfo('Unknown Browser');
        }
      };

      detectBrowser();
    };

    detectSystemInfo();
  }, []);

  return (
    <div className="bg-neutral-800 rounded-xl p-4 space-y-3 border border-neutral-700">
      <h4 className="text-sm font-semibold text-white mb-3">System Information</h4>
      <div className="space-y-2 text-xs">
        <div className="flex justify-between">
          <span className="text-neutral-400">CPU Cores:</span>
          <span className="text-white font-mono">{navigator.hardwareConcurrency || 'Unknown'}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-neutral-400">Memory Usage:</span>
          <span className="text-white font-mono">{memoryInfo}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-neutral-400">GPU:</span>
          <span className="text-white text-right max-w-48 truncate" title={gpuInfo}>{gpuInfo}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-neutral-400">Platform:</span>
          <span className="text-white font-mono">{platformInfo}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-neutral-400">Architecture:</span>
          <span className="text-white font-mono">{architectureInfo}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-neutral-400">Browser:</span>
          <span className="text-white font-mono">{browserInfo}</span>
        </div>
      </div>
    </div>
  );
}

export function SettingsModal({ 
  isOpen, 
  onClose, 
  onSettingsChange,
  optimizationConfig,
  onOptimizationConfigChange,
  optimizationStats,
  isComputing = false
}: SettingsModalProps) {
  const { user } = useAuth();
  const { settings: userSettings, updateSettings, isLoading } = useUserSettings(user?.id);
  const [activeTab, setActiveTab] = useState<'performance' | 'computation' | 'optimization' | 'datasets' | 'system'>('performance');
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
    const badges = {
      discrete: { bg: 'bg-green-500/20', text: 'text-green-300', dot: 'bg-green-400' },
      integrated: { bg: 'bg-blue-500/20', text: 'text-blue-300', dot: 'bg-blue-400' },
      cpu: { bg: 'bg-gray-500/20', text: 'text-gray-300', dot: 'bg-gray-400' },
      unknown: { bg: 'bg-gray-500/20', text: 'text-gray-400', dot: 'bg-gray-500' }
    };
    
    const badge = badges[deviceType as keyof typeof badges] || badges.unknown;
    
    return (
      <div className={`inline-flex items-center space-x-1.5 px-2 py-1 rounded-full text-xs font-medium ${badge.bg} ${badge.text}`}>
        <div className={`w-1.5 h-1.5 rounded-full ${badge.dot}`}></div>
        <span>{deviceType.charAt(0).toUpperCase() + deviceType.slice(1)}</span>
      </div>
    );
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 backdrop-blur-sm">
      <div className="bg-neutral-900 rounded-2xl shadow-2xl w-full max-w-4xl max-h-[90vh] overflow-hidden border border-neutral-700">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-neutral-700">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-600/20 rounded-lg">
              <CogIcon className="h-5 w-5 text-blue-400" />
            </div>
            <h2 className="text-xl font-semibold text-white">Performance Settings</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-neutral-400 hover:text-white rounded-lg hover:bg-neutral-800 transition-colors"
          >
            <XMarkIcon className="h-5 w-5" />
          </button>
        </div>

        <div className="flex h-[600px]">
          {/* Sidebar Navigation */}
          <div className="w-64 bg-neutral-800 p-6 space-y-3">
            <TabButton
              active={activeTab === 'performance'}
              onClick={() => setActiveTab('performance')}
              icon={<ComputerDesktopIcon />}
              label="GPU & Performance"
            />
            <TabButton
              active={activeTab === 'computation'}
              onClick={() => setActiveTab('computation')}
              icon={<CpuChipIcon />}
              label="CPU & Computation"
            />
            <TabButton
              active={activeTab === 'optimization'}
              onClick={() => setActiveTab('optimization')}
              icon={<RocketLaunchIcon />}
              label="Advanced Optimization"
            />
            <TabButton
              active={activeTab === 'datasets'}
              onClick={() => setActiveTab('datasets')}
              icon={<CircleStackIcon />}
              label="NPZ Datasets"
            />
            <TabButton
              active={activeTab === 'system'}
              onClick={() => setActiveTab('system')}
              icon={<ChartBarIcon />}
              label="System Info"
            />
          </div>

          {/* Main Content */}
          <div className="flex-1 p-6 overflow-y-auto bg-neutral-900">
            {activeTab === 'performance' && (
              <div className="space-y-6">
                {/* WebGPU Status Card */}
                <div className="bg-neutral-800 rounded-xl p-5 border border-neutral-700">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-white">WebGPU Acceleration</h3>
                    {isInitializing ? (
                      <div className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse"></div>
                        <span className="text-sm text-yellow-300">Detecting...</span>
                      </div>
                    ) : capabilities?.supported ? (
                      <div className="flex items-center space-x-2">
                        <CheckCircleIcon className="w-5 h-5 text-green-400" />
                        <span className="text-sm text-green-400 font-medium">Available</span>
                      </div>
                    ) : (
                      <div className="flex items-center space-x-2">
                        <ExclamationTriangleIcon className="w-5 h-5 text-red-400" />
                        <span className="text-sm text-red-400 font-medium">Not Available</span>
                      </div>
                    )}
                  </div>
                  
                  {capabilities?.supported && (
                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div>
                        <span className="text-xs text-neutral-400">Device Type</span>
                        <div className="mt-1">{getDeviceTypeBadge(capabilities.deviceType)}</div>
                      </div>
                      <div>
                        <span className="text-xs text-neutral-400">Vendor</span>
                        <div className="mt-1 text-sm font-medium text-white">{capabilities.vendor}</div>
                      </div>
                    </div>
                  )}

                  {/* GPU Settings */}
                  {capabilities?.supported && (
                    <div className="space-y-4">
                      <div className="flex items-center justify-between p-3 bg-neutral-700 rounded-lg">
                        <div>
                          <label className="text-sm font-medium text-white">Enable WebGPU</label>
                          <p className="text-xs text-neutral-400">Use GPU for all computations (recommended)</p>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer">
                          <input
                            type="checkbox"
                            checked={currentSettings.gpuAcceleration.enabled}
                            onChange={(e) => handleGpuSettingsChange({ enabled: e.target.checked })}
                            className="sr-only peer"
                          />
                          <div className="w-11 h-6 bg-neutral-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                        </label>
                      </div>

                      {currentSettings.gpuAcceleration.enabled && (
                        <div className="space-y-4 pl-4 border-l-2 border-blue-500">
                          <div>
                            <div className="flex justify-between items-center mb-2">
                              <label className="text-sm font-medium text-white">Batch Size</label>
                              <span className="text-sm text-blue-300">{currentSettings.gpuAcceleration.maxBatchSize.toLocaleString()}</span>
                            </div>
                            <input
                              type="range"
                              min="1024"
                              max="131072"
                              step="1024"
                              value={currentSettings.gpuAcceleration.maxBatchSize}
                              onChange={(e) => handleGpuSettingsChange({ maxBatchSize: parseInt(e.target.value) })}
                              className={`w-full h-2 bg-neutral-700 rounded-lg appearance-none cursor-pointer ${styles.slider}`}
                            />
                            <p className="text-xs text-neutral-400 mt-1">Parameters per GPU batch</p>
                          </div>

                          <div>
                            <div className="flex justify-between items-center mb-2">
                              <label className="text-sm font-medium text-white">Memory Limit</label>
                              <span className="text-sm text-blue-300">{currentSettings.gpuAcceleration.memoryThreshold}MB</span>
                            </div>
                            <input
                              type="range"
                              min="256"
                              max="4096"
                              step="256"
                              value={currentSettings.gpuAcceleration.memoryThreshold}
                              onChange={(e) => handleGpuSettingsChange({ memoryThreshold: parseInt(e.target.value) })}
                              className={`w-full h-2 bg-neutral-700 rounded-lg appearance-none cursor-pointer ${styles.slider}`}
                            />
                          </div>

                          <div className="flex items-center justify-between p-3 bg-neutral-700 rounded-lg">
                            <div>
                              <label className="text-sm font-medium text-white">CPU Fallback</label>
                              <p className="text-xs text-neutral-400">Auto-fallback if GPU fails</p>
                            </div>
                            <label className="relative inline-flex items-center cursor-pointer">
                              <input
                                type="checkbox"
                                checked={currentSettings.gpuAcceleration.fallbackToCPU}
                                onChange={(e) => handleGpuSettingsChange({ fallbackToCPU: e.target.checked })}
                                className="sr-only peer"
                              />
                              <div className="w-11 h-6 bg-neutral-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                            </label>
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {!capabilities?.supported && !isInitializing && (
                    <div className="bg-orange-500/10 border border-orange-500/20 rounded-lg p-4">
                      <div className="flex items-start space-x-3">
                        <ExclamationTriangleIcon className="w-5 h-5 text-orange-400 mt-0.5" />
                        <div>
                          <h4 className="text-sm font-medium text-orange-300">WebGPU Not Available</h4>
                          <p className="text-sm text-orange-200 mt-1">
                            Your browser doesn&apos;t support WebGPU. Computations will use CPU workers.
                          </p>
                          <p className="text-xs text-orange-300 mt-2">
                            Try Chrome 113+, Edge 113+, or Firefox 113+ for GPU acceleration.
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {activeTab === 'computation' && (
              <div className="space-y-6">
                {/* CPU Settings */}
                <div className="bg-neutral-800 rounded-xl p-5 border border-neutral-700">
                  <h3 className="text-lg font-semibold text-white mb-4">CPU Worker Settings</h3>
                  
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <label className="text-sm font-medium text-white">Worker Threads</label>
                        <span className="text-sm text-blue-300">{currentSettings.cpuSettings.maxWorkers}</span>
                      </div>
                      <input
                        type="range"
                        min="1"
                        max={navigator.hardwareConcurrency || 8}
                        step="1"
                        value={currentSettings.cpuSettings.maxWorkers}
                        onChange={(e) => handleCpuSettingsChange({ maxWorkers: parseInt(e.target.value) })}
                        className="w-full h-2 bg-neutral-700 rounded-lg appearance-none cursor-pointer slider"
                      />
                      <p className="text-xs text-neutral-400 mt-1">
                        Available cores: {navigator.hardwareConcurrency || 'Unknown'}
                      </p>
                    </div>

                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <label className="text-sm font-medium text-white">Chunk Size</label>
                        <span className="text-sm text-blue-300">{currentSettings.cpuSettings.chunkSize.toLocaleString()}</span>
                      </div>
                      <input
                        type="range"
                        min="1000"
                        max="20000"
                        step="1000"
                        value={currentSettings.cpuSettings.chunkSize}
                        onChange={(e) => handleCpuSettingsChange({ chunkSize: parseInt(e.target.value) })}
                        className="w-full h-2 bg-neutral-700 rounded-lg appearance-none cursor-pointer slider"
                      />
                      <p className="text-xs text-neutral-400 mt-1">Parameters per CPU batch</p>
                    </div>
                  </div>
                </div>

                {/* Computation Limits */}
                <div className="bg-neutral-800 rounded-xl p-5 border border-neutral-700">
                  <h3 className="text-lg font-semibold text-white mb-4">Computation Limits</h3>
                  
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <label className="text-sm font-medium text-white">Max Results</label>
                        <span className="text-sm text-blue-300">{currentSettings.maxComputationResults.toLocaleString()}</span>
                      </div>
                      <input
                        type="range"
                        min="1000"
                        max="50000"
                        step="1000"
                        value={currentSettings.maxComputationResults}
                        onChange={(e) => handleSettingsUpdate({ maxComputationResults: parseInt(e.target.value) })}
                        className="w-full h-2 bg-neutral-700 rounded-lg appearance-none cursor-pointer slider"
                      />
                      <div className="flex justify-between text-xs text-neutral-500 mt-1">
                        <span>1K</span>
                        <span>50K</span>
                      </div>
                    </div>
                    
                    <div className="flex items-center justify-between p-3 bg-neutral-700 rounded-lg">
                      <div>
                        <label className="text-sm font-medium text-white">Symmetric Grid</label>
                        <p className="text-xs text-neutral-400">Use symmetric parameter sampling</p>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input
                          type="checkbox"
                          checked={currentSettings.useSymmetricGrid}
                          onChange={(e) => handleSettingsUpdate({ useSymmetricGrid: e.target.checked })}
                          className="sr-only peer"
                        />
                        <div className="w-11 h-6 bg-neutral-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                      </label>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'optimization' && (
              <div className="space-y-6">
                {optimizationConfig && onOptimizationConfigChange ? (
                  <OptimizationControls
                    config={optimizationConfig}
                    onConfigChange={onOptimizationConfigChange}
                    isComputing={isComputing}
                    lastOptimizationStats={optimizationStats}
                  />
                ) : (
                  <div className="bg-neutral-800 rounded-xl p-5 border border-neutral-700">
                    <div className="flex items-center space-x-3 text-neutral-400">
                      <RocketLaunchIcon className="w-5 h-5" />
                      <span className="text-sm">Advanced optimization not available in this context</span>
                    </div>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'datasets' && (
              <div className="space-y-6">
                <div className="bg-neutral-800 rounded-xl p-5 border border-neutral-700">
                  <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                    <CircleStackIcon className="w-5 h-5 mr-2" />
                    NPZ Dataset Management
                  </h3>
                  <p className="text-neutral-400 text-sm mb-6">
                    Manage pre-computed circuit datasets. These NPZ files contain massive parameter sweeps 
                    generated offline for instant visualization.
                  </p>
                  <NPZDatasetManager />
                </div>
              </div>
            )}

            {activeTab === 'system' && (
              <div className="space-y-6">
                <SystemSpecs />
                
                {/* WebGPU System Info */}
                {capabilities?.supported && (
                  <div className="bg-neutral-800 rounded-xl p-5 border border-neutral-700">
                    <h4 className="text-sm font-semibold text-white mb-3">WebGPU Details</h4>
                    <div className="space-y-2 text-xs">
                      <div className="flex justify-between">
                        <span className="text-neutral-400">Device:</span>
                        <span className="text-white">{capabilities.deviceType}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-neutral-400">Vendor:</span>
                        <span className="text-white">{capabilities.vendor}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-neutral-400">Architecture:</span>
                        <span className="text-white">{capabilities.architecture}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-neutral-400">Features:</span>
                        <span className="text-white">{capabilities.features.length}</span>
                      </div>
                    </div>
                  </div>
                )}
                
                {user && (
                  <div className="bg-green-500/10 border border-green-500/20 rounded-xl p-4">
                    <div className="flex items-center space-x-2">
                      <CheckCircleIcon className="w-5 h-5 text-green-400" />
                      <span className="text-sm text-green-300 font-medium">
                        {isLoading ? 'Syncing settings...' : 'Settings synced to account'}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="border-t border-neutral-700 p-4 bg-neutral-800">
          <div className="text-xs text-neutral-400 text-center">
            ⚡ Settings auto-save{user ? ' to your account' : ' locally'} • Requires modern browser for WebGPU
          </div>
        </div>
      </div>
    </div>
  );
}