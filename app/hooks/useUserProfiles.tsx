"use client";

import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../components/auth/AuthProvider';
import { ProfilesService } from '../../lib/profilesService';
import { SavedProfile, SavedProfilesState } from '../components/circuit-simulator/types/savedProfiles';
import { CircuitParameters } from '../components/circuit-simulator/types/parameters';

export const useUserProfiles = () => {
  const { user, loading: authLoading } = useAuth();
  const [profilesState, setProfilesState] = useState<SavedProfilesState>({
    profiles: [],
    selectedProfile: null,
    pendingComputeProfile: null,
    isMultiSelectMode: false,
    selectedCircuits: []
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Enhanced localStorage fallback system
  const loadProfilesFromLocalStorage = useCallback((): SavedProfile[] => {
    try {
      // Try multiple localStorage keys for backward compatibility
      const keys = [
        `user-profiles-${user?.id}`, // User-specific profiles
        'nei-viz-saved-profiles',     // App-specific profiles
        'savedProfiles'               // Legacy profiles
      ];
      
      for (const key of keys) {
        const stored = localStorage.getItem(key);
        if (stored) {
          console.log('📂 Found profiles in localStorage key:', key);
          const data = JSON.parse(stored);
          const profiles = Array.isArray(data) ? data : (data.profiles || []);
          if (profiles.length > 0) {
            return profiles;
          }
        }
      }
      
      console.log('📭 No profiles found in localStorage');
    } catch (error) {
      console.error('❌ Error loading profiles from localStorage:', error);
    }
    return [];
  }, [user]);

  // Load profiles when user changes
  const loadProfiles = useCallback(async () => {
    if (!user) {
      console.log('👤 No user found, clearing profiles');
      setProfilesState(prev => ({ ...prev, profiles: [] }));
      return;
    }

    console.log('📥 Loading profiles for user:', user.id);
    setLoading(true);
    setError(null);
    
    try {
      console.log('🔄 Attempting to load profiles from database...');
      const profiles = await ProfilesService.getUserProfiles(user.id);
      console.log('✅ Loaded', profiles.length, 'profiles from Supabase');
      setProfilesState(prev => ({ ...prev, profiles }));
      setError(null);
    } catch (err) {
      console.error('❌ Error loading profiles from Supabase:');
      console.error('Error object:', err);
      console.error('Error stringified:', JSON.stringify(err, null, 2));
      
      const localProfiles = loadProfilesFromLocalStorage();
      console.log('📂 Loaded', localProfiles.length, 'profiles from localStorage fallback');
      setProfilesState(prev => ({ ...prev, profiles: localProfiles }));
      
      // If we have local profiles, partially clear the error
      if (localProfiles.length > 0) {
        setError(prev => prev ? `${prev} (Using local storage)` : null);
      }
    } finally {
      setLoading(false);
    }
  }, [user, loadProfilesFromLocalStorage]);

  useEffect(() => {
    loadProfiles();
  }, [loadProfiles]);

  // Enhanced localStorage save
  const saveProfilesToLocalStorage = useCallback((profiles: SavedProfile[]) => {
    try {
      if (user) {
        // Save user-specific profiles
        localStorage.setItem(`user-profiles-${user.id}`, JSON.stringify(profiles));
        console.log('💾 Saved', profiles.length, 'profiles to localStorage for user:', user.id);
      }
      
      // Also save to legacy location for backward compatibility
      localStorage.setItem('nei-viz-saved-profiles', JSON.stringify({ profiles }));
    } catch (error) {
      console.error('❌ Failed to save profiles to localStorage:', error);
    }
  }, [user]);

  // Helper function to create profile in localStorage when no user is authenticated
  const createLocalProfile = useCallback((
    name: string, 
    parameters: CircuitParameters,
    gridSize: number,
    minFreq: number,
    maxFreq: number,
    numPoints: number,
    description?: string
  ): SavedProfile => {
    console.log('📱 Creating local profile (no database):', name);
    const now = Date.now();
    const newProfile: SavedProfile = {
      id: crypto.randomUUID(),
      name,
      description: description || 'Local profile (created without authentication)',
      groundTruthParams: parameters,
      gridSize,
      minFreq,
      maxFreq,
      numPoints,
      created: now,
      lastModified: now,
      isComputed: false
    };
    
    // Add to local state
    const updatedProfiles = [...profilesState.profiles, newProfile];
    setProfilesState(prev => ({ ...prev, profiles: updatedProfiles }));
    saveProfilesToLocalStorage(updatedProfiles);
    
    console.log('📦 Local profile created successfully:', newProfile.id);
    return newProfile;
  }, [profilesState.profiles, saveProfilesToLocalStorage]);

  const createProfile = useCallback(async (
    name: string, 
    parameters: CircuitParameters,
    gridSize: number,
    minFreq: number,
    maxFreq: number,
    numPoints: number,
    description?: string
  ): Promise<SavedProfile | null> => {
    console.log('👤 CreateProfile called:', { hasUser: !!user, userId: user?.id, name, authLoading });
    
    if (authLoading) {
      console.warn('⏳ Authentication still loading - waiting before creating profile...');
      // Wait a moment for auth to complete, then retry
      await new Promise(resolve => setTimeout(resolve, 500));
      if (!user) {
        console.warn('⚠️ No authenticated user after waiting - creating local profile instead');
        return createLocalProfile(name, parameters, gridSize, minFreq, maxFreq, numPoints, description);
      }
    }
    
    if (!user) {
      console.warn('⚠️ No authenticated user - creating local profile instead');
      return createLocalProfile(name, parameters, gridSize, minFreq, maxFreq, numPoints, description);
    }

    try {
      console.log('🔄 Calling ProfilesService.createProfile...');
      const newProfile = await ProfilesService.createProfile(user.id, name, parameters, gridSize, minFreq, maxFreq, numPoints, description);
      console.log('✅ ProfilesService.createProfile succeeded:', newProfile);
      
      const updatedProfiles = [newProfile, ...profilesState.profiles];
      setProfilesState(prev => ({
        ...prev,
        profiles: updatedProfiles
      }));
      
      // Also save to localStorage as backup
      saveProfilesToLocalStorage(updatedProfiles);
      
      console.log('📦 Profile state updated, total profiles:', updatedProfiles.length);
      return newProfile;
    } catch (err) {
      console.error('Error creating profile:', err);
      setError('Failed to create profile - falling back to localStorage');
      
      // Fallback: create profile in localStorage
      const now = Date.now();
      const randomId = Math.random().toString(36).substr(2, 9);
      const fallbackProfile: SavedProfile = {
        id: `profile_${now}_${randomId}`,
        name,
        description,
        created: now,
        lastModified: now,
        gridSize,
        minFreq,
        maxFreq,
        numPoints,
        groundTruthParams: parameters,
        isComputed: false,
      };
      
      const updatedProfiles = [fallbackProfile, ...profilesState.profiles];
      setProfilesState(prev => ({
        ...prev,
        profiles: updatedProfiles
      }));
      
      // Save fallback profile to localStorage
      saveProfilesToLocalStorage(updatedProfiles);
      
      return fallbackProfile;
    }
  }, [user, authLoading, profilesState.profiles, saveProfilesToLocalStorage, createLocalProfile]);

  const updateProfile = useCallback(async (
    profileId: string, 
    updates: { 
      name?: string; 
      description?: string; 
      parameters?: CircuitParameters; 
      isComputed?: boolean; 
      computationResults?: unknown;
    }
  ): Promise<void> => {
    try {
      // Only update the profile fields that exist in the database
      const profileUpdates = {
        name: updates.name,
        description: updates.description,
        parameters: updates.parameters,
        is_computed: updates.isComputed,
        computation_results: updates.computationResults
      };
      
      // Filter out undefined values
      const filteredUpdates = Object.fromEntries(
        Object.entries(profileUpdates).filter(([, value]) => value !== undefined)
      );

      let updatedProfile;
      if (Object.keys(filteredUpdates).length > 0) {
        // Only call updateProfile if there are actual profile fields to update
        updatedProfile = await ProfilesService.updateProfile(profileId, filteredUpdates);
      } else {
        // If no profile fields to update, just find the existing profile
        updatedProfile = profilesState.profiles.find(p => p.id === profileId);
        if (!updatedProfile) {
          throw new Error('Profile not found');
        }
      }
      
      const updatedProfiles = profilesState.profiles.map(p => {
        if (p.id === profileId) {
          // Merge the updated profile (tagged models now handled via database)
          return {
            ...updatedProfile
          };
        }
        return p;
      });
      setProfilesState(prev => ({
        ...prev,
        profiles: updatedProfiles
      }));
      
      // Save to localStorage
      saveProfilesToLocalStorage(updatedProfiles);
    } catch (err) {
      console.error('Error updating profile:', err);
      setError('Failed to update profile');
    }
  }, [profilesState.profiles, saveProfilesToLocalStorage]);

  const deleteProfile = useCallback(async (profileId: string): Promise<void> => {
    try {
      await ProfilesService.deleteProfile(profileId);
      setProfilesState(prev => ({
        ...prev,
        profiles: prev.profiles.filter(p => p.id !== profileId),
        selectedProfile: prev.selectedProfile === profileId ? null : prev.selectedProfile
      }));
    } catch (err) {
      console.error('Error deleting profile:', err);
      setError('Failed to delete profile');
    }
  }, []);

  const deleteMultipleProfiles = useCallback(async (profileIds: string[]): Promise<void> => {
    try {
      await ProfilesService.deleteMultipleProfiles(profileIds);
      setProfilesState(prev => ({
        ...prev,
        profiles: prev.profiles.filter(p => !profileIds.includes(p.id)),
        selectedCircuits: []
      }));
    } catch (err) {
      console.error('Error deleting profiles:', err);
      setError('Failed to delete profiles');
    }
  }, []);

  const selectProfile = useCallback((profileId: string | null) => {
    setProfilesState(prev => ({ ...prev, selectedProfile: profileId }));
  }, []);

  const toggleMultiSelectMode = useCallback(() => {
    setProfilesState(prev => ({
      ...prev,
      isMultiSelectMode: !prev.isMultiSelectMode,
      selectedCircuits: []
    }));
  }, []);

  const toggleCircuitSelection = useCallback((profileId: string) => {
    setProfilesState(prev => {
      const isSelected = prev.selectedCircuits.includes(profileId);
      return {
        ...prev,
        selectedCircuits: isSelected 
          ? prev.selectedCircuits.filter(id => id !== profileId)
          : [...prev.selectedCircuits, profileId]
      };
    });
  }, []);

  const setPendingComputeProfile = useCallback((profileId: string | null) => {
    setProfilesState(prev => ({ ...prev, pendingComputeProfile: profileId }));
  }, []);

  return {
    profilesState,
    loading,
    error,
    actions: {
      createProfile,
      updateProfile,
      deleteProfile,
      deleteMultipleProfiles,
      selectProfile,
      toggleMultiSelectMode,
      toggleCircuitSelection,
      setPendingComputeProfile,
      refreshProfiles: loadProfiles
    }
  };
};