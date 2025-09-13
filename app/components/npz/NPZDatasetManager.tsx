'use client';

import React, { useState, useEffect } from 'react';
import { useNPZData } from '../../hooks/useNPZData';
import { 
  CloudIcon, 
  ComputerDesktopIcon, 
  ExclamationTriangleIcon,
  ArrowDownTrayIcon,
  LinkIcon,
  EyeIcon,
  DocumentTextIcon
} from '@heroicons/react/24/outline';

interface NPZDataset {
  filename: string;
  configurationName?: string;
  datasetSource?: 'local' | 'public';
  isPublic?: boolean;
  storageType?: string;
  is_available?: boolean;
  isLinkedToProfile?: boolean;
  n_parameters?: number;
  n_frequencies?: number;
  size_mb?: number;
  file_size_mb?: number;
  grid_size?: number;
  circuitParameters?: Record<string, number>;
  username?: string;
}

interface NPZDatasetManagerProps {
  onDatasetSelect?: (dataset: NPZDataset) => void;
}

export const NPZDatasetManager: React.FC<NPZDatasetManagerProps> = ({ 
  onDatasetSelect
}) => {
  const { 
    mergedDatasets, 
    publicDatasets, 
    isLoading, 
    error, 
    user,
    fetchDatasets,
    loadDataset
  } = useNPZData();

  const [selectedDataset, setSelectedDataset] = useState<NPZDataset | null>(null);
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  useEffect(() => {
    fetchDatasets();
    checkAPIStatus();
  }, [fetchDatasets]);

  const checkAPIStatus = async () => {
    try {
      const response = await fetch('http://localhost:5001/api/status');
      const data = await response.json();
      setApiStatus(data.status === 'online' ? 'online' : 'offline');
    } catch {
      setApiStatus('offline');
    }
  };

  const handleLoadDataset = async (dataset: NPZDataset) => {
    const success = await loadDataset(dataset.filename);
    if (success) {
      setSelectedDataset(dataset);
      onDatasetSelect?.(dataset);
    }
  };

  const formatFileSize = (mb: number) => {
    if (mb < 1) return `${(mb * 1024).toFixed(0)} KB`;
    if (mb < 1024) return `${mb.toFixed(1)} MB`;
    return `${(mb / 1024).toFixed(1)} GB`;
  };

  const getStorageIcon = (storageType: string, isAvailable: boolean) => {
    if (!isAvailable) {
      return <ExclamationTriangleIcon className="w-5 h-5 text-red-500" />;
    }
    
    switch (storageType) {
      case 'cloud':
        return <CloudIcon className="w-5 h-5 text-blue-500" />;
      case 'local':
        return <ComputerDesktopIcon className="w-5 h-5 text-green-500" />;
      case 'hybrid':
        return <LinkIcon className="w-5 h-5 text-purple-500" />;
      default:
        return <ComputerDesktopIcon className="w-5 h-5 text-gray-500" />;
    }
  };

  const getStatusBadge = (dataset: NPZDataset) => {
    if (!dataset.isLinkedToProfile) {
      return (
        <span className="px-2 py-1 text-xs rounded-full bg-gray-100 text-gray-600">
          Local Only
        </span>
      );
    }

    if (dataset.is_available) {
      return (
        <span className="px-2 py-1 text-xs rounded-full bg-green-100 text-green-700">
          ✓ Available
        </span>
      );
    }

    return (
      <span className="px-2 py-1 text-xs rounded-full bg-red-100 text-red-700">
        ⚠ Offline
      </span>
    );
  };

  if (isLoading && mergedDatasets.length === 0) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        <span className="ml-3 text-gray-600">Loading datasets...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* API Status Header */}
      <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
        <div className="flex items-center space-x-3">
          <div className={`w-3 h-3 rounded-full ${
            apiStatus === 'online' ? 'bg-green-500' : 
            apiStatus === 'offline' ? 'bg-red-500' : 'bg-yellow-500'
          }`} />
          <span className="font-medium">
            NPZ Backend API: {apiStatus === 'checking' ? 'Checking...' : 
                              apiStatus === 'online' ? 'Online' : 'Offline'}
          </span>
        </div>
        <button
          onClick={checkAPIStatus}
          className="px-3 py-1 text-sm bg-white border rounded-md hover:bg-gray-50"
        >
          Refresh Status
        </button>
      </div>

      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-700">{error}</p>
        </div>
      )}

      {/* My Datasets */}
      {user && (
        <div>
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <DocumentTextIcon className="w-5 h-5 mr-2" />
            My NPZ Datasets ({mergedDatasets.length})
          </h3>
          
          {mergedDatasets.length === 0 ? (
            <div className="text-center py-8 bg-gray-50 rounded-lg">
              <ComputerDesktopIcon className="w-12 h-12 mx-auto text-gray-400 mb-3" />
              <p className="text-gray-600">No NPZ datasets found</p>
              <p className="text-sm text-gray-500 mt-1">
                Run Python computations to generate datasets
              </p>
            </div>
          ) : (
            <div className="grid gap-4">
              {mergedDatasets.map((dataset) => (
                <div key={(dataset as any).filename} 
                     className={`p-4 border rounded-lg hover:shadow-md transition-shadow cursor-pointer ${
                       selectedDataset?.filename === (dataset as any).filename ? 'border-blue-500 bg-blue-50' : 'border-gray-200'
                     }`}
                     onClick={() => handleLoadDataset(dataset as NPZDataset)}>
                  
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                        {getStorageIcon(dataset.storageType, (dataset as any).is_available !== false)}
                        <h4 className="font-medium text-gray-900 flex items-center gap-2">
                          {dataset.configurationName || dataset.filename}
                          {/* Server-side indicator */}
                          {(dataset.datasetSource === 'public' || dataset.isPublic || dataset.storageType === 'cloud') && (
                            <svg className="w-4 h-4 text-blue-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                              <title>Server-side dataset</title>
                              <path fillRule="evenodd" d="M2 5a2 2 0 012-2h12a2 2 0 012 2v10a2 2 0 01-2 2H4a2 2 0 01-2-2V5zm3.293 1.293a1 1 0 011.414 0l3 3a1 1 0 010 1.414l-3 3a1 1 0 01-1.414-1.414L7.586 10 5.293 7.707a1 1 0 010-1.414zM11 12a1 1 0 100 2h3a1 1 0 100-2h-3z" clipRule="evenodd" />
                            </svg>
                          )}
                          {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                          {(dataset as any).type === 'precomputed' && (
                            <span className="ml-2 px-2 py-1 text-xs rounded-full bg-purple-100 text-purple-700">
                              Pre-computed
                            </span>
                          )}
                        </h4>
                        {getStatusBadge(dataset as NPZDataset)}
                      </div>
                      
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm text-gray-600">
                        <div>
                          <span className="font-medium">Parameters:</span>
                          <br />
                          {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                          {(dataset as any).n_parameters?.toLocaleString() || 'Unknown'}
                        </div>
                        <div>
                          <span className="font-medium">Frequencies:</span>
                          <br />
                          {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                          {(dataset as any).n_frequencies || 'Unknown'}
                        </div>
                        <div>
                          <span className="font-medium">File Size:</span>
                          <br />
                          {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                          {formatFileSize((dataset as any).size_mb || (dataset as any).file_size_mb || 0)}
                        </div>
                        <div>
                          <span className="font-medium">Grid Size:</span>
                          <br />
                          {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                          {(dataset as any).grid_size || 'Unknown'}
                        </div>
                      </div>
                      
                      {dataset.configurationName && (
                        <div className="mt-2 space-y-1">
                          <div className="text-sm">
                            <span className="text-blue-600">
                              🔗 Linked to: {dataset.configurationName}
                            </span>
                          </div>
                          {dataset.circuitParameters && (
                            <div className="text-xs text-gray-500">
                              <span className="font-medium">Circuit:</span> 
                              {' '}Rsh={Math.round(dataset.circuitParameters.Rsh)}Ω, 
                              {' '}Ra={Math.round(dataset.circuitParameters.Ra)}Ω, 
                              {' '}Ca={(dataset.circuitParameters.Ca * 1e6).toFixed(1)}µF
                              {dataset.isPublic && (
                                <span className="ml-2 px-1 py-0.5 text-xs rounded bg-green-100 text-green-700">
                                  Public
                                </span>
                              )}
                            </div>
                          )}
                          {dataset.username && (
                            <div className="text-xs text-gray-500">
                              <span className="font-medium">By:</span> {dataset.username}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                    
                    <div className="flex space-x-2">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleLoadDataset(dataset as NPZDataset);
                        }}
                        className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
                        disabled={apiStatus !== 'online'}
                      >
                        <EyeIcon className="w-4 h-4 inline mr-1" />
                        View
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Public Datasets */}
      {publicDatasets.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <CloudIcon className="w-5 h-5 mr-2" />
            Public Datasets ({publicDatasets.length})
          </h3>
          
          <div className="grid gap-4">
            {publicDatasets.map((dataset) => (
              <div key={dataset.id} className="p-4 border border-gray-200 rounded-lg bg-gray-50">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <CloudIcon className="w-5 h-5 text-blue-500" />
                      <h4 className="font-medium text-gray-900">
                        {dataset.saved_configurations?.name}
                      </h4>
                      <span className="px-2 py-1 text-xs rounded-full bg-blue-100 text-blue-700">
                        Public
                      </span>
                    </div>
                    
                    <div className="text-sm text-gray-600 mb-2">
                      {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                      By {(dataset.saved_configurations as any)?.user_profiles?.full_name || 'Unknown User'}
                    </div>
                    
                    <div className="grid grid-cols-3 gap-4 text-sm text-gray-600">
                      <div>
                        <span className="font-medium">Parameters:</span> {dataset.n_parameters.toLocaleString()}
                      </div>
                      <div>
                        <span className="font-medium">Frequencies:</span> {dataset.n_frequencies}
                      </div>
                      <div>
                        <span className="font-medium">Size:</span> {formatFileSize(dataset.file_size_mb)}
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex space-x-2">
                    <button
                      className="px-3 py-1 text-sm bg-gray-600 text-white rounded hover:bg-gray-700"
                      disabled
                    >
                      <ArrowDownTrayIcon className="w-4 h-4 inline mr-1" />
                      Download
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Quick Stats */}
      {(mergedDatasets.length > 0 || publicDatasets.length > 0) && (
        <div className="p-4 bg-blue-50 rounded-lg">
          <h4 className="font-medium text-blue-900 mb-2">Dataset Overview</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-blue-700">My Datasets:</span>
              <br />
              <span className="font-bold text-blue-900">{mergedDatasets.length}</span>
            </div>
            <div>
              <span className="text-blue-700">Public Available:</span>
              <br />
              <span className="font-bold text-blue-900">{publicDatasets.length}</span>
            </div>
            <div>
              <span className="text-blue-700">Linked to Profiles:</span>
              <br />
              <span className="font-bold text-blue-900">
                {mergedDatasets.filter(d => d.isLinkedToProfile).length}
              </span>
            </div>
            <div>
              <span className="text-blue-700">API Status:</span>
              <br />
              <span className={`font-bold ${apiStatus === 'online' ? 'text-green-900' : 'text-red-900'}`}>
                {apiStatus.toUpperCase()}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};