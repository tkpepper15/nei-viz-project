import React, { useState, useMemo } from 'react';
import { SavedProfile } from '../types/savedProfiles';
import { ProfileCard } from './ProfileCard';
import { EditProfileModal } from './EditProfileModal';

interface SavedProfilesProps {
  profiles: SavedProfile[];
  selectedProfile: string | null;
  onSelectProfile: (profileId: string) => void;
  onSelectProfileOriginal?: (profileId: string) => void;
  onDeleteProfile: (profileId: string) => void;
  onEditProfile: (profileId: string, name: string, description?: string) => void;
  onEditParameters?: (profileId: string) => void;
  onComputeProfile: (profileId: string) => void;
  onCopyParams?: (profileId: string) => void;
  isCollapsed: boolean;
  onRestart?: () => void;
  // Multi-select functionality
  selectedCircuits?: string[];
  isMultiSelectMode?: boolean;
  onToggleMultiSelect?: () => void;
  onBulkDelete?: () => void;
  // Computing state
  computingProfileId?: string | null;
}

// Filter options removed as requested

export const SavedProfiles: React.FC<SavedProfilesProps> = ({
  profiles,
  selectedProfile,
  onSelectProfile,
  onSelectProfileOriginal,
  onDeleteProfile,
  onEditProfile,
  onEditParameters,
  onComputeProfile,
  onCopyParams,
  isCollapsed,
  onRestart: _onRestart, // eslint-disable-line @typescript-eslint/no-unused-vars
  selectedCircuits = [],
  isMultiSelectMode = false,
  onToggleMultiSelect,
  onBulkDelete,
  computingProfileId = null
}) => {
  const [editingProfile, setEditingProfile] = useState<SavedProfile | null>(null);
  const [searchTerm, setSearchTerm] = useState('');

  // Search-only profiles (no filtering or sorting)
  const processedProfiles = useMemo(() => {
    // Ensure profiles is always an array
    if (!Array.isArray(profiles)) {
      return [];
    }
    
    let filtered = profiles;

    // Apply search filter only
    if (searchTerm.trim()) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(profile => 
        profile.name.toLowerCase().includes(term) ||
        (profile.description && profile.description.toLowerCase().includes(term))
      );
    }

    return filtered;
  }, [profiles, searchTerm]);



  if (isCollapsed) {
    return (
      <div className="flex flex-col items-center py-2 space-y-1">
        {processedProfiles.slice(0, 3).map((profile) => (
          <button
            key={profile.id}
            onClick={() => onSelectProfile(profile.id)}
            className={`w-10 h-10 flex items-center justify-center rounded-md transition-all duration-200 ${
              selectedProfile === profile.id
                ? 'bg-neutral-800 text-white'
                : 'text-neutral-500 hover:bg-neutral-800 hover:text-white'
            }`}
            title={profile.name}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth={2} 
                d={profile.isComputed 
                  ? "M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" 
                  : "M8 7v8a2 2 0 002 2h6M8 7V5a2 2 0 012-2h4.586a1 1 0 01.707.293l4.414 4.414a1 1 0 01.293.707V15a2 2 0 01-2 2v0M8 7H6a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2v-2"
                } 
              />
            </svg>
          </button>
        ))}
        {processedProfiles.length > 3 && (
          <div className="text-xs text-neutral-500 mt-1">
            +{processedProfiles.length - 3}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col min-h-0">
      {/* Sticky Header */}
      <div className="sticky top-0 z-10 bg-neutral-900 px-3 py-2 border-b border-neutral-700">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-medium text-neutral-200">Circuits</h3>
            <p className="text-xs text-neutral-400">
              {isMultiSelectMode && selectedCircuits.length > 0 
                ? `${selectedCircuits.length} selected`
                : `${processedProfiles.length} of ${profiles.length} circuit${profiles.length !== 1 ? 's' : ''}`
              }
            </p>
          </div>
          <div className="flex items-center gap-1">
            {/* Multi-select controls */}
            {profiles.length > 0 && (
              <>
                {isMultiSelectMode ? (
                  <>
                    {selectedCircuits.length > 0 && onBulkDelete && (
                      <button
                        onClick={onBulkDelete}
                        className="w-8 h-8 flex items-center justify-center rounded-md bg-red-600 hover:bg-red-700 text-white transition-colors"
                        title={`Delete ${selectedCircuits.length} circuit${selectedCircuits.length > 1 ? 's' : ''}`}
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    )}
                    {onToggleMultiSelect && (
                      <button
                        onClick={onToggleMultiSelect}
                        className="w-8 h-8 flex items-center justify-center rounded-md bg-neutral-700 hover:bg-neutral-600 text-neutral-300 hover:text-white transition-colors"
                        title="Exit multi-select"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    )}
                  </>
                ) : (
                  onToggleMultiSelect && (
                    <button
                      onClick={onToggleMultiSelect}
                      className="px-2 py-1 text-xs rounded-md bg-neutral-700 hover:bg-neutral-600 text-neutral-300 hover:text-white transition-colors font-medium"
                      title="Select multiple circuits"
                    >
                      Select
                    </button>
                  )
                )}
              </>
            )}
          </div>
        </div>
      </div>
      
      {/* Filtering and Sorting Controls */}
      {profiles.length > 0 && (
        <div className="px-3 py-2 bg-neutral-800/50 border-b border-neutral-700">
          {/* Search Only */}
          <input
            type="text"
            placeholder="Search circuits..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full px-2 py-1 text-xs bg-neutral-700 border border-neutral-600 rounded text-neutral-200 placeholder-neutral-400 focus:outline-none focus:border-blue-500"
          />
        </div>
      )}
      
      {/* Scrollable Profiles List */}
      <div className="flex-1 overflow-y-auto scrollbar-thin scrollbar-track-neutral-900 scrollbar-thumb-neutral-700 hover:scrollbar-thumb-neutral-600">
        {profiles.length === 0 ? (
          <div className="p-4 text-center">
            <svg className="w-8 h-8 mx-auto mb-2 text-neutral-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 7v8a2 2 0 002 2h6M8 7V5a2 2 0 012-2h4.586a1 1 0 01.707.293l4.414 4.414a1 1 0 01.293.707V15a2 2 0 01-2 2v0M8 7H6a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2v-2" />
            </svg>
            <p className="text-sm text-neutral-400">No saved profiles</p>
            <p className="text-xs text-neutral-500 mt-1">
              Use &quot;Save Profile&quot; to create your first one
            </p>
          </div>
        ) : (
          <div className="space-y-1 p-2">
            {processedProfiles.map((profile) => (
              <ProfileCard
                key={profile.id}
                profile={profile}
                isSelected={selectedProfile === profile.id}
                onDelete={() => onDeleteProfile(profile.id)}
                onEdit={() => setEditingProfile(profile)}
                onEditParameters={onEditParameters ? () => onEditParameters(profile.id) : undefined}
                onCompute={() => onComputeProfile(profile.id)}
                onCopyParams={onCopyParams ? () => onCopyParams(profile.id) : undefined}
                isMultiSelectMode={isMultiSelectMode}
                isSelectedForDelete={selectedCircuits.includes(profile.id)}
                isComputing={computingProfileId === profile.id}
                onToggleSelect={() => onSelectProfile(profile.id)}
                onClick={() => onSelectProfileOriginal ? onSelectProfileOriginal(profile.id) : onSelectProfile(profile.id)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Edit Profile Modal */}
      <EditProfileModal
        profile={editingProfile}
        isOpen={!!editingProfile}
        onClose={() => setEditingProfile(null)}
        onSave={(profileId, name, description) => {
          onEditProfile(profileId, name, description);
          setEditingProfile(null);
        }}
      />
    </div>
  );
}; 