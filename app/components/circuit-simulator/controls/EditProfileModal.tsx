import React, { useState, useEffect } from 'react';
import { SavedProfile } from '../types/savedProfiles';

interface EditProfileModalProps {
  profile: SavedProfile | null;
  isOpen: boolean;
  onClose: () => void;
  onSave: (profileId: string, name: string, description?: string) => void;
}

export const EditProfileModal: React.FC<EditProfileModalProps> = ({
  profile,
  isOpen,
  onClose,
  onSave
}) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Update form when profile changes
  useEffect(() => {
    if (profile) {
      setName(profile.name);
      setDescription(profile.description || '');
    }
  }, [profile]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!profile || !name.trim()) return;

    setIsSubmitting(true);
    try {
      onSave(profile.id, name.trim(), description.trim() || undefined);
      onClose();
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = () => {
    if (!isSubmitting) {
      onClose();
    }
  };

  if (!isOpen || !profile) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-neutral-800 rounded-lg shadow-xl border border-neutral-600 w-full max-w-md mx-4">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-neutral-700">
          <h3 className="text-lg font-semibold text-white">Edit Profile</h3>
          <button
            onClick={handleClose}
            disabled={isSubmitting}
            className="w-8 h-8 flex items-center justify-center rounded-md hover:bg-neutral-700 transition-colors text-neutral-400 hover:text-white disabled:opacity-50"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-4 space-y-4">
          {/* Profile Name */}
          <div>
            <label htmlFor="edit-profile-name" className="block text-sm font-medium text-neutral-200 mb-2">
              Profile Name <span className="text-red-400">*</span>
            </label>
            <input
              id="edit-profile-name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter profile name"
              maxLength={50}
              disabled={isSubmitting}
              className="w-full px-3 py-2 bg-neutral-700 border border-neutral-600 rounded-md text-white placeholder-neutral-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
              required
            />
            <div className="text-xs text-neutral-400 mt-1">
              {name.length}/50 characters
            </div>
          </div>

          {/* Description */}
          <div>
            <label htmlFor="edit-profile-description" className="block text-sm font-medium text-neutral-200 mb-2">
              Description (Optional)
            </label>
            <textarea
              id="edit-profile-description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Enter description"
              maxLength={200}
              disabled={isSubmitting}
              rows={3}
              className="w-full px-3 py-2 bg-neutral-700 border border-neutral-600 rounded-md text-white placeholder-neutral-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 resize-none"
            />
            <div className="text-xs text-neutral-400 mt-1">
              {description.length}/200 characters
            </div>
          </div>

          {/* Profile Info Display */}
          <div className="bg-neutral-700/50 rounded-md p-3 space-y-2">
            <div className="text-xs text-neutral-400">Profile Settings</div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-neutral-300 bg-neutral-600 px-2 py-0.5 rounded">
                {profile.gridSize}⁵ grid
              </span>
              <span className="text-xs text-neutral-400">•</span>
              <span className="text-xs text-neutral-300">
                {profile.minFreq}-{profile.maxFreq} Hz
              </span>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-2 pt-2">
            <button
              type="button"
              onClick={handleClose}
              disabled={isSubmitting}
              className="flex-1 px-4 py-2 bg-neutral-600 hover:bg-neutral-500 text-white rounded-md transition-colors disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting || !name.trim()}
              className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
            >
              {isSubmitting ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                  Saving...
                </>
              ) : (
                'Save Changes'
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}; 