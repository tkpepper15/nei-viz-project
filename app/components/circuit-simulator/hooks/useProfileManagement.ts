import { useCallback, useEffect } from 'react';
import { SavedProfile, SavedProfilesState } from '../types/savedProfiles';
import { CircuitParameters } from '../types/parameters';
import { useLocalStorage } from './useLocalStorage';
import { useIdGeneration } from './useIdGeneration';

interface UseProfileManagementProps {
  gridSize: number;
  minFreq: number;
  maxFreq: number;
  numPoints: number;
  groundTruthParams: CircuitParameters;
  updateStatusMessage: (message: string) => void;
}

/**
 * Custom hook to manage saved profiles with localStorage persistence
 */
export const useProfileManagement = ({
  gridSize,
  minFreq,
  maxFreq,
  numPoints,
  groundTruthParams,
  updateStatusMessage
}: UseProfileManagementProps) => {
  const { generateUniqueId } = useIdGeneration();
  
  const initialState: SavedProfilesState = {
    profiles: [],
    selectedProfile: null
  };

  const [savedProfilesState, setSavedProfilesState, hasLoadedFromStorage] = useLocalStorage(
    'nei-viz-saved-profiles',
    initialState
  );

  // Add sample profile on first load if no profiles exist
  useEffect(() => {
    if (
      hasLoadedFromStorage && 
      groundTruthParams.Rsh > 0
    ) {
      // Check if specific default profiles exist
      const hasSampleProfile = savedProfilesState.profiles.some(p => p.id === 'sample_profile_default');
      const hasTestConfig1 = savedProfilesState.profiles.some(p => p.id === 'test_config_1');
      
      const profilesToAdd: SavedProfile[] = [];
      
      if (!hasSampleProfile) {
        const sampleProfile: SavedProfile = {
          id: 'sample_profile_default',
          name: 'Sample Configuration',
          description: 'Example profile showing a typical bioimpedance measurement setup',
          created: Date.now(),
          lastModified: Date.now(),
          gridSize: 5,
          minFreq: 0.1,
          maxFreq: 100000,
          numPoints: 100,
          groundTruthParams: {
            Rsh: 50,
            Ra: 1000,
            Ca: 1.0e-6,
            Rb: 800,
            Cb: 0.8e-6,
            frequency_range: [0.1, 100000]
          },
          isComputed: false,
        };
        profilesToAdd.push(sampleProfile);
      }

      if (!hasTestConfig1) {
        const testConfig1: SavedProfile = {
          id: 'test_config_1',
          name: 'Test Config #1',
          description: 'Optimized parameters for ideal Nyquist semicircular arcs - demonstrates proper EIS behavior',
          created: Date.now(),
          lastModified: Date.now(),
          gridSize: 5,
          minFreq: 0.1,
          maxFreq: 100000,
          numPoints: 100,
          groundTruthParams: {
            Rsh: 50,      // Balanced shunt resistance
            Ra: 1000,     // Higher apical resistance for clear semicircles
            Ca: 1.0e-6,   // 1.0 μF for proper time constants
            Rb: 800,      // Asymmetric basal resistance for distinct arcs
            Cb: 0.8e-6,   // 0.8 μF for frequency separation
            frequency_range: [0.1, 100000]
          },
          isComputed: false,
        };
        profilesToAdd.push(testConfig1);
      }

      if (profilesToAdd.length > 0) {
        setSavedProfilesState(prev => ({
          ...prev,
          profiles: [...prev.profiles, ...profilesToAdd]
        }));
      }
    }
  }, [
    hasLoadedFromStorage, 
    savedProfilesState.profiles, 
    groundTruthParams.Rsh,
    setSavedProfilesState
  ]);

  const handleSaveProfile = useCallback((name: string, description?: string) => {
    const profileId = generateUniqueId('profile');
    const now = Date.now();
    
    const newProfile: SavedProfile = {
      id: profileId,
      name,
      description,
      created: now,
      lastModified: now,
      
      // Grid computation settings
      gridSize,
      minFreq,
      maxFreq,
      numPoints,
      
      // Circuit parameters
      groundTruthParams: { ...groundTruthParams },
      
      // Computation status
      isComputed: false,
    };

    setSavedProfilesState(prev => ({
      ...prev,
      profiles: [...prev.profiles, newProfile],
      selectedProfile: newProfile.id
    }));

    updateStatusMessage(`Profile "${name}" saved with current settings`);
  }, [
    generateUniqueId,
    gridSize,
    minFreq,
    maxFreq,
    numPoints,
    groundTruthParams,
    setSavedProfilesState,
    updateStatusMessage
  ]);

  const handleCopyParams = useCallback((profileId: string) => {
    const profile = savedProfilesState.profiles.find(p => p.id === profileId);
    if (!profile) return;

    const paramsData = {
      profileName: profile.name,
      gridSettings: {
        gridSize: profile.gridSize,
        minFreq: profile.minFreq,
        maxFreq: profile.maxFreq,
        numPoints: profile.numPoints
      },
      circuitParameters: {
        Rsh: profile.groundTruthParams.Rsh,
        Ra: profile.groundTruthParams.Ra,
        Ca: profile.groundTruthParams.Ca,
        Rb: profile.groundTruthParams.Rb,
        Cb: profile.groundTruthParams.Cb,
        frequency_range: profile.groundTruthParams.frequency_range
      }
    };

    const jsonString = JSON.stringify(paramsData, null, 2);
    
    if (typeof window !== 'undefined' && navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(jsonString).then(() => {
        updateStatusMessage(`Parameters copied to clipboard from "${profile.name}"`);
      }).catch(() => {
        updateStatusMessage('Failed to copy parameters to clipboard');
      });
    } else {
      // Fallback for non-secure contexts
      try {
        if (typeof document !== 'undefined') {
          const textArea = document.createElement('textarea');
          textArea.value = jsonString;
          document.body.appendChild(textArea);
          textArea.select();
          document.execCommand('copy');
          document.body.removeChild(textArea);
          updateStatusMessage(`Parameters copied to clipboard from "${profile.name}"`);
        }
      } catch (error) {
        console.error('Clipboard fallback failed:', error);
        updateStatusMessage('Failed to copy parameters to clipboard');
      }
    }
  }, [savedProfilesState.profiles, updateStatusMessage]);

  return {
    savedProfilesState,
    setSavedProfilesState,
    hasLoadedFromStorage,
    handleSaveProfile,
    handleCopyParams
  };
};