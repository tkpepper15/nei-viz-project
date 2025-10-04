"use client";

import React, { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../components/auth/AuthProvider';
import { AuthModal } from '../components/auth/AuthModal';
import { SlimNavbar } from '../components/circuit-simulator/SlimNavbar';
import { PentagonGroundTruth } from '../components/circuit-simulator/visualizations/PentagonGroundTruth';
import { CircuitParameters } from '../components/circuit-simulator/types/parameters';
import { ModelSnapshot } from '../components/circuit-simulator/types';
import { useEnhancedUserSettings } from '../hooks/useEnhancedUserSettings';
import { useCircuitConfigurations } from '../hooks/useCircuitConfigurations';
import { CircuitConfiguration } from '../../lib/circuitConfigService';

// Default circuit parameters
const defaultParameters: CircuitParameters = {
  Rsh: 1000,
  Ra: 2000,
  Ca: 1e-6,
  Rb: 5000,
  Cb: 2e-6,
  frequency_range: [0.1, 100000]
};

export default function SweeperPage() {
  // Authentication
  const { user, loading: authLoading, signOut } = useAuth();
  const [showAuthModal, setShowAuthModal] = useState(false);

  // User settings and profiles
  const { settings: userSettings, isLoading: settingsLoading, updateSettings } = useEnhancedUserSettings(user?.id);
  const { configurations: savedProfiles } = useCircuitConfigurations(user?.id);

  // Component state
  const [parameters, setParameters] = useState<CircuitParameters>(defaultParameters);
  const [models] = useState<ModelSnapshot[]>([]); // Will be populated by PentagonGroundTruth analysis
  const [isComputingGrid] = useState(false); // Currently not used but kept for future analysis state

  // Profile management
  const [selectedProfileId, setSelectedProfileId] = useState<string | null>(null);

  // Load user's saved parameters when settings are available
  useEffect(() => {
    if (userSettings?.circuit_parameters) {
      setParameters(prev => ({
        ...prev,
        ...userSettings.circuit_parameters,
        frequency_range: userSettings.circuit_parameters?.frequency_range || prev.frequency_range
      }));
    }
  }, [userSettings]);

  // Handle profile selection
  const handleProfileSelect = useCallback((config: CircuitConfiguration) => {
    setSelectedProfileId(config.id);
    setParameters({
      ...config.circuitParameters,
      frequency_range: config.circuitParameters.frequency_range || [0.1, 100000]
    });

    // Update user settings with selected profile parameters
    if (updateSettings) {
      updateSettings({
        circuit_parameters: config.circuitParameters
      });
    }
  }, [updateSettings]);

  // Handle parameter changes
  const handleParameterChange = useCallback((newParams: CircuitParameters) => {
    setParameters(newParams);

    // Auto-save to user settings
    if (updateSettings) {
      updateSettings({
        circuit_parameters: newParams
      });
    }
  }, [updateSettings]);

  // Show authentication modal for non-authenticated users
  useEffect(() => {
    if (!authLoading && !user) {
      setShowAuthModal(true);
    }
  }, [authLoading, user]);

  if (authLoading || settingsLoading) {
    return (
      <div className="min-h-screen bg-neutral-900 flex items-center justify-center">
        <div className="text-neutral-400">Loading Sweeper...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-neutral-900 text-neutral-100 flex flex-col">
      {/* Authentication Modal */}
      <AuthModal
        isOpen={showAuthModal}
        onClose={() => setShowAuthModal(false)}
        onAuthSuccess={() => setShowAuthModal(false)}
      />


      {/* Navigation */}
      <SlimNavbar
        statusMessage={isComputingGrid ? "Sweeper analysis running..." : "Sweeper ready"}
        gridResults={models}
        gridSize={userSettings?.visualization_settings?.gridSize || 9}
        onSettingsOpen={() => {}} // No settings for sweeper page
        onModelInfoOpen={() => {}} // TODO: Implement model info modal
        user={user}
        onSignOut={signOut}
      />

      {/* Main Content */}
      <div className="flex-1 flex">
        {/* Left Panel: Profile Selection and Parameters */}
        <div className="w-80 bg-neutral-800 border-r border-neutral-700 flex flex-col">
          {/* Profile Selection */}
          <div className="p-4 border-b border-neutral-700">
            <h2 className="text-lg font-semibold text-neutral-200 mb-3">Circuit Profiles</h2>
            <div className="space-y-2">
              {savedProfiles.length > 0 ? (
                savedProfiles.map((config) => (
                  <button
                    key={config.id}
                    onClick={() => handleProfileSelect(config)}
                    className={`w-full p-3 text-left rounded border transition-colors ${
                      selectedProfileId === config.id
                        ? 'bg-orange-600 text-white border-orange-500'
                        : 'bg-neutral-700 text-neutral-300 border-neutral-600 hover:bg-neutral-600'
                    }`}
                  >
                    <div className="font-medium">{config.name}</div>
                    {config.description && (
                      <div className="text-xs text-neutral-400 mt-1">{config.description}</div>
                    )}
                  </button>
                ))
              ) : (
                <div className="text-center text-neutral-500 py-4">
                  No saved profiles yet. Configure parameters below and save your first profile in the main application.
                </div>
              )}
            </div>
          </div>

          {/* Circuit Parameters */}
          <div className="flex-1 p-4 overflow-y-auto">
            <h3 className="text-md font-medium text-neutral-300 mb-3">Circuit Parameters</h3>
            <div className="space-y-3">
              {/* Resistance Parameters */}
              <div>
                <label className="block text-xs text-neutral-400 mb-1">Rsh (Ω)</label>
                <input
                  type="number"
                  value={parameters.Rsh}
                  onChange={(e) => handleParameterChange({...parameters, Rsh: parseFloat(e.target.value) || 0})}
                  className="w-full p-2 bg-neutral-800 border border-neutral-600 rounded text-neutral-200 text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-neutral-400 mb-1">Ra (Ω)</label>
                <input
                  type="number"
                  value={parameters.Ra}
                  onChange={(e) => handleParameterChange({...parameters, Ra: parseFloat(e.target.value) || 0})}
                  className="w-full p-2 bg-neutral-800 border border-neutral-600 rounded text-neutral-200 text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-neutral-400 mb-1">Rb (Ω)</label>
                <input
                  type="number"
                  value={parameters.Rb}
                  onChange={(e) => handleParameterChange({...parameters, Rb: parseFloat(e.target.value) || 0})}
                  className="w-full p-2 bg-neutral-800 border border-neutral-600 rounded text-neutral-200 text-sm"
                />
              </div>

              {/* Capacitance Parameters */}
              <div>
                <label className="block text-xs text-neutral-400 mb-1">Ca (µF)</label>
                <input
                  type="number"
                  step="0.1"
                  value={(parameters.Ca * 1e6).toFixed(1)}
                  onChange={(e) => handleParameterChange({...parameters, Ca: (parseFloat(e.target.value) || 0) / 1e6})}
                  className="w-full p-2 bg-neutral-800 border border-neutral-600 rounded text-neutral-200 text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-neutral-400 mb-1">Cb (µF)</label>
                <input
                  type="number"
                  step="0.1"
                  value={(parameters.Cb * 1e6).toFixed(1)}
                  onChange={(e) => handleParameterChange({...parameters, Cb: (parseFloat(e.target.value) || 0) / 1e6})}
                  className="w-full p-2 bg-neutral-800 border border-neutral-600 rounded text-neutral-200 text-sm"
                />
              </div>

              {/* Frequency Range */}
              <div>
                <label className="block text-xs text-neutral-400 mb-1">Min Frequency (Hz)</label>
                <input
                  type="number"
                  step="0.1"
                  value={parameters.frequency_range[0]}
                  onChange={(e) => handleParameterChange({
                    ...parameters,
                    frequency_range: [parseFloat(e.target.value) || 0.1, parameters.frequency_range[1]]
                  })}
                  className="w-full p-2 bg-neutral-800 border border-neutral-600 rounded text-neutral-200 text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-neutral-400 mb-1">Max Frequency (Hz)</label>
                <input
                  type="number"
                  value={parameters.frequency_range[1]}
                  onChange={(e) => handleParameterChange({
                    ...parameters,
                    frequency_range: [parameters.frequency_range[0], parseFloat(e.target.value) || 100000]
                  })}
                  className="w-full p-2 bg-neutral-800 border border-neutral-600 rounded text-neutral-200 text-sm"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Right Panel: Sweeper Visualization */}
        <div className="flex-1 flex flex-col">
          <div className="p-4 border-b border-neutral-700">
            <h1 className="text-xl font-bold text-neutral-100">Parameter Sweeper</h1>
            <p className="text-sm text-neutral-400 mt-1">
              Geometric analysis of circuit parameter relationships through frequency-domain pentagon visualization
            </p>
          </div>

          <div className="flex-1 p-4">
            <PentagonGroundTruth
              models={models}
              currentParameters={parameters}
            />
          </div>
        </div>
      </div>
    </div>
  );
}