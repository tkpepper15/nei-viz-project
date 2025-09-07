"use client";

import React, { useState, useRef } from 'react';
import { ResultsFileImporter, ExportedResultsMetadata } from '../utils/fileExport';
import { BackendMeshPoint } from '../types';

interface ImportResultsDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onImport: (results: BackendMeshPoint[], metadata: ExportedResultsMetadata) => void;
}

export const ImportResultsDialog: React.FC<ImportResultsDialogProps> = ({
  isOpen,
  onClose,
  onImport
}) => {
  const [importStep, setImportStep] = useState<'select' | 'uploading' | 'processing' | 'success' | 'error'>('select');
  const [importedMetadata, setImportedMetadata] = useState<ExportedResultsMetadata | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>('');
  const metadataFileRef = useRef<HTMLInputElement>(null);
  const topModelsFileRef = useRef<HTMLInputElement>(null);

  const handleImport = async () => {
    const metadataFile = metadataFileRef.current?.files?.[0];
    const topModelsFile = topModelsFileRef.current?.files?.[0];

    if (!metadataFile || !topModelsFile) {
      setErrorMessage('Please select both metadata and top models files');
      setImportStep('error');
      return;
    }

    setImportStep('uploading');
    setErrorMessage('');

    try {
      // Import metadata
      const metadata = await ResultsFileImporter.importMetadata(metadataFile);
      setImportedMetadata(metadata);
      
      setImportStep('processing');
      
      // Import top models
      const compactModels = await ResultsFileImporter.importTopModels(topModelsFile);
      
      // Convert to BackendMeshPoint format
      const results = ResultsFileImporter.convertToBackendMeshPoints(compactModels);
      
      setImportStep('success');
      
      // Pass results to parent
      onImport(results, metadata);
      
      // Close dialog after short delay
      setTimeout(() => {
        onClose();
        setImportStep('select');
        setImportedMetadata(null);
      }, 2000);
      
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Failed to import results');
      setImportStep('error');
    }
  };

  const handleClose = () => {
    onClose();
    setImportStep('select');
    setImportedMetadata(null);
    setErrorMessage('');
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-neutral-800 rounded-lg border border-neutral-700 p-6 w-96 max-h-[80vh] overflow-y-auto">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-neutral-200">Import Results</h3>
          <button
            onClick={handleClose}
            className="text-neutral-400 hover:text-neutral-200"
            disabled={importStep === 'uploading' || importStep === 'processing'}
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {importStep === 'select' && (
          <div className="space-y-4">
            <p className="text-sm text-neutral-400">
              Import previously exported results to resume your analysis. You&apos;ll need both the metadata file and top models file.
            </p>
            
            <div>
              <label className="block text-sm font-medium text-neutral-300 mb-2">
                Metadata File (*_metadata.json)
              </label>
              <input
                ref={metadataFileRef}
                type="file"
                accept=".json"
                className="block w-full text-sm text-neutral-400 bg-neutral-700 border border-neutral-600 rounded-lg cursor-pointer focus:outline-none focus:border-blue-500"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-neutral-300 mb-2">
                Top Models File (*_top_models.json)
              </label>
              <input
                ref={topModelsFileRef}
                type="file"
                accept=".json"
                className="block w-full text-sm text-neutral-400 bg-neutral-700 border border-neutral-600 rounded-lg cursor-pointer focus:outline-none focus:border-blue-500"
              />
            </div>

            <div className="flex gap-3 mt-6">
              <button
                onClick={handleImport}
                className="flex-1 bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700 transition-colors font-medium"
              >
                Import Results
              </button>
              <button
                onClick={handleClose}
                className="px-4 py-2 bg-neutral-700 text-neutral-300 rounded hover:bg-neutral-600 transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {importStep === 'uploading' && (
          <div className="text-center py-8">
            <div className="w-8 h-8 border-2 border-purple-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-neutral-300">Reading files...</p>
          </div>
        )}

        {importStep === 'processing' && importedMetadata && (
          <div className="text-center py-8">
            <div className="w-8 h-8 border-2 border-purple-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-neutral-300 mb-2">Processing results...</p>
            <div className="text-xs text-neutral-400 space-y-1">
              <p>📊 Total models: {importedMetadata.totalModels.toLocaleString()}</p>
              <p>⭐ Top models: {importedMetadata.topModelsCount}</p>
              <p>📅 Computed: {new Date(importedMetadata.computationDate).toLocaleDateString()}</p>
            </div>
          </div>
        )}

        {importStep === 'success' && importedMetadata && (
          <div className="text-center py-8">
            <div className="w-12 h-12 bg-green-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <p className="text-green-400 font-medium mb-2">Import Successful!</p>
            <div className="text-xs text-neutral-400 space-y-1">
              <p>📊 {importedMetadata.totalModels.toLocaleString()} models from computation</p>
              <p>⭐ {importedMetadata.topModelsCount} top performers loaded</p>
              <p>🎯 Grid size: {importedMetadata.gridSize}</p>
            </div>
          </div>
        )}

        {importStep === 'error' && (
          <div className="text-center py-8">
            <div className="w-12 h-12 bg-red-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </div>
            <p className="text-red-400 font-medium mb-2">Import Failed</p>
            <p className="text-sm text-neutral-400">{errorMessage}</p>
            <button
              onClick={() => setImportStep('select')}
              className="mt-4 bg-neutral-700 text-neutral-300 px-4 py-2 rounded hover:bg-neutral-600 transition-colors"
            >
              Try Again
            </button>
          </div>
        )}
      </div>
    </div>
  );
};