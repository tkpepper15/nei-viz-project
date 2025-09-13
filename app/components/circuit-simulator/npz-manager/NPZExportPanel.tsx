import React, { useState } from 'react';
import { ModelSnapshot } from '../types';

interface NPZExportPanelProps {
  results: ModelSnapshot[];
  onExport: (filename: string, metadata: NPZMetadata) => void;
  isExporting?: boolean;
}

interface NPZMetadata {
  title: string;
  description: string;
  tags: string[];
  gridSize: number;
  frequencyRange: [number, number];
  numFrequencies: number;
  author?: string;
  created: string;
}

export const NPZExportPanel: React.FC<NPZExportPanelProps> = ({
  results,
  onExport,
  isExporting = false
}) => {
  const [filename, setFilename] = useState('');
  const [metadata, setMetadata] = useState<NPZMetadata>({
    title: '',
    description: '',
    tags: [],
    gridSize: 0,
    frequencyRange: [0.1, 100000],
    numFrequencies: 100,
    created: new Date().toISOString()
  });
  const [tagInput, setTagInput] = useState('');

  const addTag = () => {
    if (tagInput.trim() && !metadata.tags.includes(tagInput.trim())) {
      setMetadata(prev => ({
        ...prev,
        tags: [...prev.tags, tagInput.trim()]
      }));
      setTagInput('');
    }
  };

  const removeTag = (tagToRemove: string) => {
    setMetadata(prev => ({
      ...prev,
      tags: prev.tags.filter(tag => tag !== tagToRemove)
    }));
  };

  const handleExport = () => {
    if (!filename.trim()) {
      alert('Please enter a filename');
      return;
    }
    
    const fullFilename = filename.endsWith('.npz') ? filename : `${filename}.npz`;
    onExport(fullFilename, metadata);
  };

  const canExport = results.length > 0 && filename.trim();

  return (
    <div className="bg-neutral-800 rounded-lg p-4 space-y-4">
      <div className="flex items-center gap-2 mb-4">
        <svg className="w-5 h-5 text-blue-400" fill="currentColor" viewBox="0 0 20 20">
          <path d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z" />
        </svg>
        <h3 className="text-lg font-semibold text-white">Export NPZ Dataset</h3>
      </div>

      {results.length === 0 ? (
        <div className="text-neutral-400 text-center py-8">
          No computation results to export. Run a computation first.
        </div>
      ) : (
        <>
          <div className="bg-neutral-700/50 rounded p-3">
            <div className="text-sm text-neutral-300">
              Ready to export <span className="text-white font-semibold">{results.length.toLocaleString()}</span> computation results
            </div>
          </div>

          {/* Filename */}
          <div>
            <label className="block text-sm font-medium text-neutral-200 mb-2">
              Filename
            </label>
            <input
              type="text"
              value={filename}
              onChange={(e) => setFilename(e.target.value)}
              placeholder="my-circuit-analysis"
              className="w-full px-3 py-2 bg-neutral-700 border border-neutral-600 rounded text-white placeholder-neutral-400"
            />
            <div className="text-xs text-neutral-400 mt-1">
              .npz extension will be added automatically
            </div>
          </div>

          {/* Title */}
          <div>
            <label className="block text-sm font-medium text-neutral-200 mb-2">
              Title
            </label>
            <input
              type="text"
              value={metadata.title}
              onChange={(e) => setMetadata(prev => ({ ...prev, title: e.target.value }))}
              placeholder="Circuit Analysis Results"
              className="w-full px-3 py-2 bg-neutral-700 border border-neutral-600 rounded text-white placeholder-neutral-400"
            />
          </div>

          {/* Description */}
          <div>
            <label className="block text-sm font-medium text-neutral-200 mb-2">
              Description
            </label>
            <textarea
              value={metadata.description}
              onChange={(e) => setMetadata(prev => ({ ...prev, description: e.target.value }))}
              placeholder="Describe this dataset..."
              rows={3}
              className="w-full px-3 py-2 bg-neutral-700 border border-neutral-600 rounded text-white placeholder-neutral-400 resize-none"
            />
          </div>

          {/* Tags */}
          <div>
            <label className="block text-sm font-medium text-neutral-200 mb-2">
              Tags
            </label>
            <div className="flex gap-2 mb-2">
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
              {metadata.tags.map(tag => (
                <span
                  key={tag}
                  className="px-2 py-1 bg-neutral-600 text-white text-sm rounded flex items-center gap-1"
                >
                  {tag}
                  <button
                    onClick={() => removeTag(tag)}
                    className="text-neutral-400 hover:text-white"
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          </div>

          {/* Export Button */}
          <button
            onClick={handleExport}
            disabled={!canExport || isExporting}
            className="w-full px-4 py-3 bg-green-600 hover:bg-green-700 disabled:bg-neutral-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
          >
            {isExporting ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Exporting...
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M3 17a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1v-2zM6.293 6.707a1 1 0 010-1.414l3-3a1 1 0 011.414 0l3 3a1 1 0 01-1.414 1.414L11 5.414V13a1 1 0 11-2 0V5.414L7.707 6.707a1 1 0 01-1.414 0z" />
                </svg>
                Export NPZ File
              </>
            )}
          </button>
        </>
      )}
    </div>
  );
};