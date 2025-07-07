import React, { useState } from 'react';
import { SavedProfile } from '../types/savedProfiles';
import { ProfileCard } from './ProfileCard';
import { EditProfileModal } from './EditProfileModal';

interface SavedProfilesProps {
  profiles: SavedProfile[];
  selectedProfile: string | null;
  onSelectProfile: (profileId: string) => void;
  onDeleteProfile: (profileId: string) => void;
  onEditProfile: (profileId: string, name: string, description?: string) => void;
  onComputeProfile: (profileId: string) => void;
  onCopyParams?: (profileId: string) => void;
  isCollapsed: boolean;
}

export const SavedProfiles: React.FC<SavedProfilesProps> = ({
  profiles,
  selectedProfile,
  onSelectProfile,
  onDeleteProfile,
  onEditProfile,
  onComputeProfile,
  onCopyParams,
  isCollapsed
}) => {
  const [editingProfile, setEditingProfile] = useState<SavedProfile | null>(null);



  if (isCollapsed) {
    return (
      <div className="flex flex-col items-center py-2 space-y-1">
        {profiles.slice(0, 3).map((profile) => (
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
        {profiles.length > 3 && (
          <div className="text-xs text-neutral-500 mt-1">
            +{profiles.length - 3}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col min-h-0">
      {/* Sticky Header */}
      <div className="sticky top-0 z-10 bg-neutral-900 px-3 py-2 border-b border-neutral-700">
        <h3 className="text-sm font-medium text-neutral-200">Saved Profiles</h3>
        <p className="text-xs text-neutral-400">
          {profiles.length} profile{profiles.length !== 1 ? 's' : ''}
        </p>
      </div>
      
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
            {profiles.map((profile) => (
              <ProfileCard
                key={profile.id}
                profile={profile}
                isSelected={selectedProfile === profile.id}
                onDelete={() => onDeleteProfile(profile.id)}
                onEdit={() => setEditingProfile(profile)}
                onCompute={() => onComputeProfile(profile.id)}
                onCopyParams={onCopyParams ? () => onCopyParams(profile.id) : undefined}
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