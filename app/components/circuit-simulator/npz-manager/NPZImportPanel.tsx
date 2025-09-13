import React, { useState, useRef } from 'react';

interface NPZImportPanelProps {
  onImport: (file: File) => void;
  isImporting?: boolean;
  supportedFormats?: string[];
}

export const NPZImportPanel: React.FC<NPZImportPanelProps> = ({
  onImport,
  isImporting = false,
  supportedFormats = ['.npz']
}) => {
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (isValidFile(file)) {
        onImport(file);
      } else {
        alert(`Invalid file type. Supported formats: ${supportedFormats.join(', ')}`);
      }
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (isValidFile(file)) {
        onImport(file);
      } else {
        alert(`Invalid file type. Supported formats: ${supportedFormats.join(', ')}`);
      }
    }
  };

  const isValidFile = (file: File) => {
    return supportedFormats.some(format => 
      file.name.toLowerCase().endsWith(format.toLowerCase())
    );
  };

  const openFileDialog = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="bg-neutral-800 rounded-lg p-4 space-y-4">
      <div className="flex items-center gap-2 mb-4">
        <svg className="w-5 h-5 text-green-400" fill="currentColor" viewBox="0 0 20 20">
          <path d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" />
        </svg>
        <h3 className="text-lg font-semibold text-white">Import NPZ Dataset</h3>
      </div>

      {/* File Drop Zone */}
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          dragActive 
            ? 'border-blue-400 bg-blue-400/10' 
            : 'border-neutral-600 hover:border-neutral-500'
        } ${isImporting ? 'pointer-events-none opacity-50' : 'cursor-pointer'}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={openFileDialog}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={supportedFormats.join(',')}
          onChange={handleFileSelect}
          className="hidden"
          disabled={isImporting}
        />
        
        {isImporting ? (
          <div className="flex flex-col items-center gap-3">
            <div className="w-8 h-8 border-4 border-blue-400 border-t-transparent rounded-full animate-spin" />
            <div className="text-white font-medium">Importing dataset...</div>
            <div className="text-neutral-400 text-sm">Processing NPZ file</div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3">
            <div className="w-12 h-12 bg-neutral-700 rounded-lg flex items-center justify-center">
              <svg className="w-6 h-6 text-neutral-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <div className="text-white font-medium">
              {dragActive ? 'Drop your NPZ file here' : 'Click to select or drag & drop'}
            </div>
            <div className="text-neutral-400 text-sm">
              Supported formats: {supportedFormats.join(', ')}
            </div>
          </div>
        )}
      </div>

      {/* Import Instructions */}
      <div className="bg-neutral-700/50 rounded-lg p-3">
        <h4 className="text-sm font-medium text-neutral-200 mb-2">Import Instructions:</h4>
        <ul className="text-xs text-neutral-300 space-y-1">
          <li>• NPZ files contain pre-computed circuit analysis results</li>
          <li>• Imported datasets will be available for immediate visualization</li>
          <li>• You can edit metadata and tags after importing</li>
          <li>• Files are stored locally in your browser</li>
        </ul>
      </div>

      {/* Format Info */}
      <div className="text-xs text-neutral-400 bg-neutral-900/50 rounded p-2">
        <strong>NPZ Format:</strong> NumPy compressed archive containing circuit parameters, 
        impedance spectra, resnorm values, and metadata for efficient storage and sharing.
      </div>
    </div>
  );
};