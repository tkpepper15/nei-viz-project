import React, { useState, useEffect } from 'react';

interface SaveProfileModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (name: string, description?: string) => void;
  defaultName?: string;
}

export const SaveProfileModal: React.FC<SaveProfileModalProps> = ({
  isOpen,
  onClose,
  onSave,
  defaultName = ''
}) => {
  const [name, setName] = useState(defaultName);
  const [description, setDescription] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Reset form when modal opens
  useEffect(() => {
    if (isOpen) {
      setName(defaultName);
      setDescription('');
      setIsSubmitting(false);
    }
  }, [isOpen, defaultName]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!name.trim()) return;
    
    setIsSubmitting(true);
    
    try {
      await onSave(name.trim(), description.trim() || undefined);
      onClose();
    } catch (error) {
      console.error('Error saving profile:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Modal */}
      <div 
        className="relative bg-neutral-800 rounded-lg shadow-2xl border border-neutral-600 w-full max-w-md mx-4"
        onKeyDown={handleKeyDown}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-neutral-700">
          <h2 className="text-lg font-semibold text-neutral-200">Save Profile</h2>
          <button
            onClick={onClose}
            className="w-8 h-8 flex items-center justify-center rounded-md hover:bg-neutral-700 transition-colors text-neutral-400 hover:text-neutral-200"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        
        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6">
          <div className="space-y-4">
            {/* Name Input */}
            <div>
              <label htmlFor="profileName" className="block text-sm font-medium text-neutral-200 mb-2">
                Profile Name *
              </label>
              <input
                id="profileName"
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Enter profile name..."
                className="w-full px-3 py-2 bg-neutral-700 border border-neutral-600 rounded-md text-neutral-200 placeholder-neutral-400 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
                autoFocus
                required
                maxLength={50}
              />
              <p className="text-xs text-neutral-400 mt-1">
                {name.length}/50 characters
              </p>
            </div>
            
            {/* Description Input */}
            <div>
              <label htmlFor="profileDescription" className="block text-sm font-medium text-neutral-200 mb-2">
                Description (optional)
              </label>
              <textarea
                id="profileDescription"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Brief description of this configuration..."
                className="w-full px-3 py-2 bg-neutral-700 border border-neutral-600 rounded-md text-neutral-200 placeholder-neutral-400 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none resize-none"
                rows={3}
                maxLength={200}
              />
              <p className="text-xs text-neutral-400 mt-1">
                {description.length}/200 characters
              </p>
            </div>
            
            {/* Save info */}
            <div className="bg-neutral-900/50 rounded-md p-3 border border-neutral-700">
              <p className="text-xs text-neutral-400 leading-relaxed">
                This will save your current grid size, frequency range, circuit parameters, and other settings. 
                You can compute it later from the saved profiles list.
              </p>
            </div>
          </div>
          
          {/* Actions */}
          <div className="flex gap-3 mt-6">
            <button
              type="button"
              onClick={onClose}
              disabled={isSubmitting}
              className="flex-1 px-4 py-2 text-sm font-medium text-neutral-300 bg-neutral-700 hover:bg-neutral-600 disabled:opacity-50 disabled:cursor-not-allowed rounded-md transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!name.trim() || isSubmitting}
              className="flex-1 px-4 py-2 text-sm font-medium text-white bg-green-600 hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-md transition-colors"
            >
              {isSubmitting ? (
                <span className="flex items-center justify-center">
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Saving...
                </span>
              ) : (
                'Save Profile'
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}; 