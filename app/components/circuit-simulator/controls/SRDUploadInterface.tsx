"use client";

import React, { useState, useCallback, useRef } from 'react';
// Note: useSRDFileHandler removed since we now use SerializedComputationManager.importFromSRD directly
import { SerializedComputationManager } from '../utils/serializedComputationManager';

interface SRDUploadInterfaceProps {
  onSRDUploaded: (manager: SerializedComputationManager, metadata: { title: string; totalResults: number; gridSize: number }) => void;
  onError: (error: string) => void;
  className?: string;
}

export const SRDUploadInterface: React.FC<SRDUploadInterfaceProps> = ({
  onSRDUploaded,
  onError,
  className = ''
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  // Note: uploadSRD removed since we now use SerializedComputationManager.importFromSRD directly

  const handleFileUpload = useCallback(async (file: File) => {
    setIsUploading(true);
    setUploadProgress('Validating file...');

    try {
      // Single-pass upload and validation to avoid race conditions
      setUploadProgress('Processing data file...');

      // Create SerializedComputationManager directly from file (includes validation)
      const manager = await SerializedComputationManager.importFromSRD(file);

      // Get metadata from the manager after successful import
      const exportCheck = manager.canExport();
      if (!exportCheck.canExport) {
        throw new Error('Invalid SRD data: no results available');
      }

      setUploadProgress('Finalizing import...');

      // Extract metadata from the successfully imported manager
      const preview = manager.getExportPreview();
      const metadata = {
        title: file.name.replace(/\.(srd|json|csv)$/, ''), // Use filename as fallback
        totalResults: preview.resultCount,
        gridSize: parseInt(preview.gridConfiguration.split('^')[0]) // Extract grid size from "9^5 parameters"
      };

      setUploadProgress('Complete!');

      // Notify parent component
      onSRDUploaded(manager, metadata);

      setTimeout(() => {
        setUploadProgress('');
        setIsUploading(false);
      }, 1000);

    } catch (error) {
      console.error('SRD upload error:', error);
      onError(error instanceof Error ? error.message : 'Upload failed');
      setIsUploading(false);
      setUploadProgress('');
    }
  }, [onSRDUploaded, onError]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    const validFile = files.find(file => {
      const fileName = file.name.toLowerCase();
      return fileName.endsWith('.json') || fileName.endsWith('.srd');
    });

    if (validFile) {
      handleFileUpload(validFile);
    } else {
      onError('Please select a valid .json file');
    }
  }, [handleFileUpload, onError]);

  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
  }, [handleFileUpload]);

  const handleClick = useCallback(() => {
    if (!isUploading) {
      fileInputRef.current?.click();
    }
  }, [isUploading]);

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Upload Area */}
      <div
        className={`
          relative border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-200
          ${isDragging
            ? 'border-orange-400 bg-orange-900/20'
            : isUploading
              ? 'border-green-400 bg-green-900/20'
              : 'border-neutral-600 bg-neutral-800/50 hover:border-orange-400 hover:bg-orange-900/10'
          }
        `}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".json,.srd"
          onChange={handleFileInputChange}
          className="hidden"
          disabled={isUploading}
        />

        {isUploading ? (
          <div className="flex flex-col items-center gap-3">
            <div className="w-8 h-8 border-2 border-green-400 border-t-transparent rounded-full animate-spin"></div>
            <div className="text-green-400 font-medium">Uploading Data File</div>
            <div className="text-sm text-neutral-400">{uploadProgress}</div>
          </div>
        ) : isDragging ? (
          <div className="flex flex-col items-center gap-3">
            <svg className="w-12 h-12 text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
            <div className="text-orange-400 font-medium">Drop JSON file here</div>
            <div className="text-sm text-neutral-400">Release to upload</div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3">
            <svg className="w-12 h-12 text-neutral-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <div className="text-neutral-200 font-medium">Upload Circuit Analysis Data</div>
            <div className="text-sm text-neutral-400">
              Drag & drop your <span className="text-orange-400 font-mono">.json</span> file here or <span className="text-orange-400">click to browse</span>
            </div>
            <div className="text-xs text-neutral-500">
              Supports files up to 500MB with up to 10M results
            </div>
          </div>
        )}
      </div>

      {/* Upload Info */}
      <div className="bg-neutral-800/50 border border-neutral-700 rounded-lg p-4">
        <h4 className="text-sm font-medium text-neutral-200 mb-2 flex items-center gap-2">
          <svg className="w-4 h-4 text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          About JSON Files
        </h4>
        <div className="text-xs text-neutral-400 space-y-1">
          <div>• <strong className="text-neutral-300">Circuit Analysis Data</strong> - Complete computation results in JSON format</div>
          <div>• <strong className="text-neutral-300">Structured data</strong> - Human-readable and programmatically accessible</div>
          <div>• <strong className="text-neutral-300">Instant loading</strong> - Skip computation entirely</div>
          <div>• <strong className="text-neutral-300">Compatible</strong> with all grid sizes and frequency presets</div>
        </div>
      </div>

      {/* File Requirements */}
      <div className="bg-orange-900/20 border border-orange-600/50 rounded-lg p-3">
        <h4 className="text-sm font-medium text-orange-200 mb-1 flex items-center gap-2">
          <svg className="w-4 h-4 text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
          Requirements
        </h4>
        <div className="text-xs text-orange-300">
          Only <span className="font-mono bg-orange-900/50 px-1 rounded">.json</span> files are supported.
          Use the export feature in the visualizer to create JSON files.
        </div>
      </div>
    </div>
  );
};