import React, { useState, useEffect } from 'react';

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

interface NPZLibraryManagerProps {
  onLoadDataset: (dataset: NPZDataset) => void;
  onEditMetadata: (dataset: NPZDataset) => void;
  onDeleteDataset: (datasetId: string) => void;
}

export const NPZLibraryManager: React.FC<NPZLibraryManagerProps> = ({
  onLoadDataset,
  onEditMetadata,
  onDeleteDataset
}) => {
  const [datasets, setDatasets] = useState<NPZDataset[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTag, setSelectedTag] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<'name' | 'created' | 'size'>('created');

  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = () => {
    // Load datasets from localStorage
    try {
      const stored = localStorage.getItem('npz-datasets');
      if (stored) {
        setDatasets(JSON.parse(stored));
      }
    } catch (error) {
      console.error('Error loading NPZ datasets:', error);
    }
  };

  const filteredDatasets = datasets
    .filter(dataset => {
      const matchesSearch = dataset.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           dataset.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           dataset.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
      const matchesTag = !selectedTag || dataset.tags.includes(selectedTag);
      return matchesSearch && matchesTag;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return a.title.localeCompare(b.title);
        case 'created':
          return new Date(b.created).getTime() - new Date(a.created).getTime();
        case 'size':
          return b.size - a.size;
        default:
          return 0;
      }
    });

  const allTags = Array.from(new Set(datasets.flatMap(d => d.tags))).sort();

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

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="bg-neutral-800 rounded-lg p-4 space-y-4">
      <div className="flex items-center gap-2 mb-4">
        <svg className="w-5 h-5 text-purple-400" fill="currentColor" viewBox="0 0 20 20">
          <path d="M7 3a1 1 0 000 2h6a1 1 0 100-2H7zM4 7a1 1 0 011-1h10a1 1 0 110 2H5a1 1 0 01-1-1zM2 11a2 2 0 012-2h12a2 2 0 012 2v4a2 2 0 01-2 2H4a2 2 0 01-2-2v-4z" />
        </svg>
        <h3 className="text-lg font-semibold text-white">NPZ Library</h3>
        <span className="text-sm text-neutral-400">({datasets.length} datasets)</span>
      </div>

      {datasets.length === 0 ? (
        <div className="text-center py-8">
          <div className="w-16 h-16 bg-neutral-700 rounded-lg flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-neutral-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
          </div>
          <div className="text-white font-medium mb-2">No NPZ datasets found</div>
          <div className="text-neutral-400 text-sm">Import or export datasets to get started</div>
        </div>
      ) : (
        <>
          {/* Search and Filters */}
          <div className="flex gap-3 mb-4">
            <div className="flex-1">
              <input
                type="text"
                placeholder="Search datasets..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full px-3 py-2 bg-neutral-700 border border-neutral-600 rounded text-white placeholder-neutral-400"
              />
            </div>
            <select
              value={selectedTag || ''}
              onChange={(e) => setSelectedTag(e.target.value || null)}
              className="px-3 py-2 bg-neutral-700 border border-neutral-600 rounded text-white"
            >
              <option value="">All tags</option>
              {allTags.map(tag => (
                <option key={tag} value={tag}>{tag}</option>
              ))}
            </select>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as 'created' | 'name' | 'size')}
              className="px-3 py-2 bg-neutral-700 border border-neutral-600 rounded text-white"
            >
              <option value="created">Latest</option>
              <option value="name">Name</option>
              <option value="size">Size</option>
            </select>
          </div>

          {/* Dataset List */}
          <div className="space-y-3">
            {filteredDatasets.map(dataset => (
              <div key={dataset.id} className="bg-neutral-700/50 rounded-lg p-4 hover:bg-neutral-700 transition-colors">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <h4 className="font-medium text-white">{dataset.title || dataset.filename}</h4>
                      <span className="text-xs text-neutral-400">({formatFileSize(dataset.size)})</span>
                    </div>
                    
                    {dataset.description && (
                      <p className="text-sm text-neutral-300 mb-2">{dataset.description}</p>
                    )}
                    
                    <div className="flex items-center gap-4 text-xs text-neutral-400 mb-2">
                      <span>{dataset.gridSize}⁵ grid</span>
                      <span>{dataset.numResults.toLocaleString()} results</span>
                      <span>{dataset.frequencyRange[0]}-{dataset.frequencyRange[1]} Hz</span>
                      <span>{formatDate(dataset.created)}</span>
                    </div>
                    
                    {dataset.tags.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {dataset.tags.map(tag => (
                          <span
                            key={tag}
                            className="px-2 py-1 bg-neutral-600 text-neutral-200 text-xs rounded cursor-pointer hover:bg-neutral-500"
                            onClick={() => setSelectedTag(tag)}
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                  
                  <div className="flex gap-2 ml-4">
                    <button
                      onClick={() => onLoadDataset(dataset)}
                      className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded transition-colors"
                    >
                      Load
                    </button>
                    <button
                      onClick={() => onEditMetadata(dataset)}
                      className="px-3 py-1.5 bg-neutral-600 hover:bg-neutral-500 text-white text-sm rounded transition-colors"
                    >
                      Edit
                    </button>
                    <button
                      onClick={() => onDeleteDataset(dataset.id)}
                      className="px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white text-sm rounded transition-colors"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
};