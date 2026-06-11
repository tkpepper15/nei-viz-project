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
  onNewCircuit?: () => void;
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
  onNewCircuit,
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
      <div className="sticky top-0 z-10 bg-surface px-3 py-2 border-b border-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-medium text-neutral-600 uppercase tracking-wider">
              {isMultiSelectMode && selectedCircuits.length > 0
                ? `${selectedCircuits.length} selected`
                : 'Circuits'}
            </span>
            {!isMultiSelectMode && profiles.length > 0 && (
              <span className="text-[10px] text-neutral-700 tabular-nums">{profiles.length}</span>
            )}
          </div>
          <div className="flex items-center gap-1">
            {isMultiSelectMode ? (
              <>
                {selectedCircuits.length > 0 && onBulkDelete && (
                  <button
                    onClick={onBulkDelete}
                    className="w-6 h-6 flex items-center justify-center rounded hover:bg-danger/20 text-neutral-500 hover:text-danger transition-colors"
                    title={`Delete ${selectedCircuits.length} circuit${selectedCircuits.length > 1 ? 's' : ''}`}
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                )}
                {onToggleMultiSelect && (
                  <button
                    onClick={onToggleMultiSelect}
                    className="w-6 h-6 flex items-center justify-center rounded text-neutral-500 hover:text-neutral-200 hover:bg-neutral-800 transition-colors"
                    title="Exit selection"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                )}
              </>
            ) : (
              <>
                {profiles.length > 0 && onToggleMultiSelect && (
                  <button
                    onClick={onToggleMultiSelect}
                    className="w-6 h-6 flex items-center justify-center rounded text-neutral-600 hover:text-neutral-300 hover:bg-neutral-800 transition-colors"
                    title="Select multiple"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                    </svg>
                  </button>
                )}
                {onNewCircuit && (
                  <button
                    onClick={onNewCircuit}
                    className="w-6 h-6 flex items-center justify-center rounded text-neutral-600 hover:text-neutral-200 hover:bg-neutral-800 transition-colors"
                    title="New circuit"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                    </svg>
                  </button>
                )}
              </>
            )}
          </div>
        </div>
      </div>
      
      {/* Filtering and Sorting Controls */}
      {profiles.length > 0 && (
        <div className="px-3 py-2 border-b border-border">
          <input
            type="text"
            placeholder="Search..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full px-2 py-1 text-xs bg-neutral-800 border border-neutral-800 rounded text-neutral-200 placeholder-neutral-600 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-colors"
          />
        </div>
      )}
      
      {/* Scrollable Profiles List */}
      <div className="flex-1 overflow-y-auto scrollbar-thin scrollbar-track-neutral-900 scrollbar-thumb-neutral-700 hover:scrollbar-thumb-neutral-600">
        {profiles.length === 0 ? (
          <div className="p-6 text-center">
            <p className="text-xs text-neutral-500">No circuits saved</p>
            <p className="text-xs text-neutral-600 mt-0.5">Use &quot;New Circuit&quot; to get started</p>
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