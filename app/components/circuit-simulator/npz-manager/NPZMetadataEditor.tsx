import React, { useState } from 'react';

interface NPZDataset {
  id: string;
  filename: string;
  title: string;
  description: string;
  tags: string[];
  size: number;
  created: string;
  lastModified: string;
  numResults: number;
  gridSize: number;
  frequencyRange: [number, number];
  author?: string;
}

interface NPZMetadataEditorProps {
  dataset: NPZDataset;
  onSave: (updatedDataset: NPZDataset) => void;
  onCancel: () => void;
  isOpen: boolean;
}

export const NPZMetadataEditor: React.FC<NPZMetadataEditorProps> = ({
  dataset,
  onSave,
  onCancel,
  isOpen
}) => {
  const [editedDataset, setEditedDataset] = useState<NPZDataset>({ ...dataset });
  const [tagInput, setTagInput] = useState('');

  if (!isOpen) return null;

  const addTag = () => {
    if (tagInput.trim() && !editedDataset.tags.includes(tagInput.trim())) {
      setEditedDataset(prev => ({
        ...prev,
        tags: [...prev.tags, tagInput.trim()]
      }));
      setTagInput('');
    }
  };

  const removeTag = (tagToRemove: string) => {
    setEditedDataset(prev => ({
      ...prev,
      tags: prev.tags.filter(tag => tag !== tagToRemove)
    }));
  };

  const handleSave = () => {
    const updatedDataset = {
      ...editedDataset,
      lastModified: new Date().toISOString()
    };
    onSave(updatedDataset);
  };

  const formatFileSize = (bytes: number) => {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unit = 0;
    while (size >= 1024 && unit < units.length - 1) {
      size /= 1024;
      unit++;
    }
    return `${size.toFixed(1)} ${units[unit]}`;
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-neutral-800 rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-neutral-700">
          <div className="flex items-center gap-2">
            <svg className="w-5 h-5 text-blue-400" fill="currentColor" viewBox="0 0 20 20">
              <path d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zM11.379 5.793L3 14.172V17h2.828l8.38-8.379-2.83-2.828z" />
            </svg>
            <h2 className="text-lg font-semibold text-white">Edit Dataset Metadata</h2>
          </div>
          <button
            onClick={onCancel}
            className="text-neutral-400 hover:text-white transition-colors"
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          {/* File Info */}
          <div className="bg-neutral-700/50 rounded-lg p-3">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-neutral-400">Filename:</span>
                <div className="text-white font-mono">{dataset.filename}</div>
              </div>
              <div>
                <span className="text-neutral-400">Size:</span>
                <div className="text-white">{formatFileSize(dataset.size)}</div>
              </div>
              <div>
                <span className="text-neutral-400">Results:</span>
                <div className="text-white">{dataset.numResults.toLocaleString()}</div>
              </div>
              <div>
                <span className="text-neutral-400">Grid Size:</span>
                <div className="text-white">{dataset.gridSize}⁵</div>
              </div>
              <div>
                <span className="text-neutral-400">Frequency Range:</span>
                <div className="text-white">{dataset.frequencyRange[0]}-{dataset.frequencyRange[1]} Hz</div>
              </div>
              <div>
                <span className="text-neutral-400">Created:</span>
                <div className="text-white">{new Date(dataset.created).toLocaleDateString()}</div>
              </div>
            </div>
          </div>

          {/* Editable Fields */}
          <div>
            <label className="block text-sm font-medium text-neutral-200 mb-2">
              Title
            </label>
            <input
              type="text"
              value={editedDataset.title}
              onChange={(e) => setEditedDataset(prev => ({ ...prev, title: e.target.value }))}
              placeholder="Dataset title"
              className="w-full px-3 py-2 bg-neutral-700 border border-neutral-600 rounded text-white placeholder-neutral-400"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-neutral-200 mb-2">
              Description
            </label>
            <textarea
              value={editedDataset.description}
              onChange={(e) => setEditedDataset(prev => ({ ...prev, description: e.target.value }))}
              placeholder="Describe this dataset..."
              rows={4}
              className="w-full px-3 py-2 bg-neutral-700 border border-neutral-600 rounded text-white placeholder-neutral-400 resize-none"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-neutral-200 mb-2">
              Author (optional)
            </label>
            <input
              type="text"
              value={editedDataset.author || ''}
              onChange={(e) => setEditedDataset(prev => ({ ...prev, author: e.target.value }))}
              placeholder="Your name or organization"
              className="w-full px-3 py-2 bg-neutral-700 border border-neutral-600 rounded text-white placeholder-neutral-400"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-neutral-200 mb-2">
              Tags
            </label>
            <div className="flex gap-2 mb-3">
              <input
                type="text"
                value={tagInput}
                onChange={(e) => setTagInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && addTag()}
                placeholder="Add tag..."
                className="flex-1 px-3 py-2 bg-neutral-700 border border-neutral-600 rounded text-white placeholder-neutral-400"
              />
              <button
                onClick={addTag}
                className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors"
              >
                Add
              </button>
            </div>
            <div className="flex flex-wrap gap-2">
              {editedDataset.tags.map(tag => (
                <span
                  key={tag}
                  className="px-2 py-1 bg-neutral-600 text-white text-sm rounded flex items-center gap-1"
                >
                  {tag}
                  <button
                    onClick={() => removeTag(tag)}
                    className="text-neutral-400 hover:text-white transition-colors"
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-3 p-4 border-t border-neutral-700">
          <button
            onClick={onCancel}
            className="px-4 py-2 bg-neutral-600 hover:bg-neutral-500 text-white rounded transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded transition-colors"
          >
            Save Changes
          </button>
        </div>
      </div>
    </div>
  );
};