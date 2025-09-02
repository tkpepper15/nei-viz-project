import React, { useState } from 'react';
import { SavedProfile } from '../types/savedProfiles';

interface ProfileCardProps {
  profile: SavedProfile;
  isSelected: boolean;
  onDelete: () => void;
  onCompute: () => void;
  onEdit: () => void;
  onEditParameters?: () => void;
  onCopyParams?: () => void;
  // Multi-select functionality
  isMultiSelectMode?: boolean;
  isSelectedForDelete?: boolean;
  onToggleSelect?: () => void;
  onClick?: () => void;
  // Computing state
  isComputing?: boolean;
}

export const ProfileCard: React.FC<ProfileCardProps> = ({
  profile,
  isSelected,
  onDelete,
  onCompute,
  onEdit,
  onEditParameters,
  onCopyParams,
  isMultiSelectMode = false,
  isSelectedForDelete = false,
  onToggleSelect,
  onClick,
  isComputing = false
}) => {
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);

  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) {
      return 'Today';
    } else if (diffDays === 1) {
      return 'Yesterday';
    } else if (diffDays < 7) {
      return `${diffDays} days ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

    return (
    <div
      className={`group relative rounded-lg transition-all duration-200 hover:bg-neutral-800/50 ${
        isSelected ? 'bg-neutral-700 border border-neutral-600' : ''
      } ${
        isSelectedForDelete ? 'ring-2 ring-blue-500' : ''
      } ${
        isComputing ? 'ring-2 ring-green-500 bg-green-900/20' : ''
      } ${
        isMultiSelectMode ? 'cursor-pointer' : ''
      }`}
      onClick={isMultiSelectMode ? onToggleSelect : onClick}
    >
      <div className="flex items-center p-3 gap-3">
        {/* Main Content - Non-clickable */}
        <div className="flex-1 min-w-0">
          {/* Profile Name */}
          <h4 className="text-sm font-medium text-white truncate mb-1 flex items-center gap-2">
            {profile.name}
            {isComputing && (
              <span className="flex items-center gap-1 text-xs text-green-400">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                Computing...
              </span>
            )}
          </h4>
          
          {/* Grid Info */}
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs text-neutral-300 bg-neutral-700/60 px-2 py-0.5 rounded">
              {profile.gridSize}⁵ grid
            </span>
            <span className="text-xs text-neutral-400">•</span>
            <span className="text-xs text-neutral-300">
              {profile.minFreq}-{profile.maxFreq} Hz
            </span>
          </div>
          
          {/* Description */}
          {profile.description && (
            <p className="text-xs text-neutral-400 truncate mb-1">
              {profile.description}
            </p>
          )}
          
          {/* Tagged Models Note: Now loaded from database when viewing */}
          
          {/* Time */}
          <div className="text-xs text-neutral-400">
            {formatDate(profile.created)}
          </div>
        </div>
        
        {/* Three Dots Menu */}
        <div className="relative">
          <button
            onClick={(e) => {
              e.stopPropagation();
              setOpenMenuId(openMenuId === profile.id ? null : profile.id);
            }}
            className="w-8 h-8 flex items-center justify-center rounded-md hover:bg-neutral-700 transition-colors opacity-0 group-hover:opacity-100"
          >
            <svg className="w-4 h-4 text-neutral-400" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 8c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"/>
            </svg>
          </button>
          
          {/* Dropdown Menu */}
          {openMenuId === profile.id && (
            <>
              {/* Backdrop */}
              <div 
                className="fixed inset-0 z-10"
                onClick={() => setOpenMenuId(null)}
              />
              
              {/* Menu */}
              <div className="absolute right-0 top-full mt-1 w-40 bg-neutral-800 rounded-lg shadow-lg border border-neutral-600 py-1 z-20">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onCompute();
                    setOpenMenuId(null);
                  }}
                  className="w-full text-left px-3 py-2 text-sm text-neutral-200 hover:bg-neutral-700 flex items-center gap-2"
                >
                  <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  Compute
                </button>
                
                {onEditParameters && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onEditParameters();
                      setOpenMenuId(null);
                    }}
                    className="w-full text-left px-3 py-2 text-sm text-neutral-200 hover:bg-neutral-700 flex items-center gap-2"
                  >
                    <svg className="w-4 h-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4" />
                    </svg>
                    Edit Parameters
                  </button>
                )}
                
                
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onEdit();
                    setOpenMenuId(null);
                  }}
                  className="w-full text-left px-3 py-2 text-sm text-neutral-200 hover:bg-neutral-700 flex items-center gap-2"
                >
                  <svg className="w-4 h-4 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                  </svg>
                  Edit Name
                </button>
                
                {onCopyParams && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onCopyParams();
                      setOpenMenuId(null);
                    }}
                    className="w-full text-left px-3 py-2 text-sm text-neutral-200 hover:bg-neutral-700 flex items-center gap-2"
                  >
                    <svg className="w-4 h-4 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
                    </svg>
                    Copy Params
                  </button>
                )}
                
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    if (typeof window !== 'undefined' && window.confirm(`Delete profile ${profile.name}?`)) {
                      onDelete();
                    }
                    setOpenMenuId(null);
                  }}
                  className="w-full text-left px-3 py-2 text-sm text-red-400 hover:bg-neutral-700 flex items-center gap-2"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                  Delete
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}; 