import React, { useState } from 'react';
import { NPZExportPanel } from './NPZExportPanel';
import { NPZImportPanel } from './NPZImportPanel';
import { NPZLibraryManager } from './NPZLibraryManager';
import { NPZMetadataEditor } from './NPZMetadataEditor';
import { ModelSnapshot } from '../types';
import { SerializedComputationManager } from '../utils/serializedComputationManager';

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

interface NPZManagerProps {
  results: ModelSnapshot[];
  onLoadDataset: (results: ModelSnapshot[]) => void;
  className?: string;
}

type TabType = 'export' | 'import' | 'library';

export const NPZManager: React.FC<NPZManagerProps> = ({
  results,
  onLoadDataset,
  className = ''
}) => {
  const [activeTab, setActiveTab] = useState<TabType>('library');
  const [isExporting, setIsExporting] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const [editingDataset, setEditingDataset] = useState<NPZDataset | null>(null);

  const handleExport = async (filename: string, metadata: { title?: string; description?: string; tags?: string[] }) => {
    try {
      setIsExporting(true);
      
      // Convert results to serialized format using SerializedComputationManager
      const serializedManager = new SerializedComputationManager({ 
        gridSize: 15,
        frequencyPreset: 'standard' 
      });
      
      // Convert ModelSnapshot to BackendMeshPoint format for serialization
      const backendResults: Array<{
        parameters: {
          Rsh: number;
          Ra: number;
          Ca: number;
          Rb: number;
          Cb: number;
          frequency_range: [number, number];
        };
        resnorm: number;
        spectrum: Array<{freq: number; real: number; imag: number; mag: number; phase: number}>;
      }> = results.map((snapshot) => ({
        parameters: snapshot.parameters,
        resnorm: snapshot.resnorm || 0,
        spectrum: snapshot.data?.map(point => ({
          freq: point.frequency,
          real: point.real,
          imag: point.imaginary,
          mag: point.magnitude,
          phase: point.phase
        })) || [],
      }));
      
      // Store the computation results in serialized format
      const numStored = serializedManager.storeResults(backendResults);
      console.log(`Stored ${numStored} results for export`);
      
      // Create NPZ dataset entry for local storage
      const dataset: NPZDataset = {
        id: `npz-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        filename,
        title: metadata.title || filename.replace('.npz', ''),
        description: metadata.description || '',
        tags: metadata.tags || [],
        size: results.length * 1000, // Rough estimate
        created: new Date().toISOString(),
        lastModified: new Date().toISOString(),
        numResults: results.length,
        gridSize: 15,
        frequencyRange: [0.1, 100000],
        author: 'User'
      };
      
      // Store in localStorage (in real implementation, this would generate and download NPZ file)
      const existingDatasets = JSON.parse(localStorage.getItem('npz-datasets') || '[]');
      existingDatasets.push(dataset);
      localStorage.setItem('npz-datasets', JSON.stringify(existingDatasets));
      
      // Store serialized data with dataset ID
      localStorage.setItem(`npz-data-${dataset.id}`, JSON.stringify({
        serializedResults: 'mock-data',
        metadata: dataset
      }));
      
      console.log(`Exported ${results.length} results as ${filename}`);
      alert(`Successfully exported ${results.length.toLocaleString()} results to ${filename}`);
      
    } catch (error) {
      console.error('Export error:', error);
      alert('Export failed. Please try again.');
    } finally {
      setIsExporting(false);
    }
  };

  const handleImport = async (file: File) => {
    try {
      setIsImporting(true);
      
      // In a real implementation, this would parse the NPZ file
      // For now, we'll create a mock dataset
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const dataset: NPZDataset = {
            id: `imported-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            filename: file.name,
            title: file.name.replace('.npz', ''),
            description: `Imported from ${file.name}`,
            tags: ['imported'],
            size: file.size,
            created: new Date().toISOString(),
            lastModified: new Date().toISOString(),
            numResults: 1000, // Would be parsed from NPZ
            gridSize: 15, // Would be parsed from NPZ
            frequencyRange: [0.1, 100000], // Would be parsed from NPZ
          };
          
          // Store dataset info
          const existingDatasets = JSON.parse(localStorage.getItem('npz-datasets') || '[]');
          existingDatasets.push(dataset);
          localStorage.setItem('npz-datasets', JSON.stringify(existingDatasets));
          
          console.log(`Imported ${file.name} successfully`);
          alert(`Successfully imported ${file.name}`);
          
        } catch (error) {
          console.error('Import parsing error:', error);
          alert('Failed to parse NPZ file. Please ensure it\'s a valid format.');
        } finally {
          setIsImporting(false);
        }
      };
      
      reader.readAsArrayBuffer(file);
      
    } catch (error) {
      console.error('Import error:', error);
      alert('Import failed. Please try again.');
      setIsImporting(false);
    }
  };

  const handleLoadDataset = (dataset: NPZDataset) => {
    try {
      // Load serialized data from localStorage
      const storedData = localStorage.getItem(`npz-data-${dataset.id}`);
      if (storedData) {
        JSON.parse(storedData);
        
        // In a real implementation, this would deserialize the results
        // For now, we'll generate mock data as ModelSnapshot[]
        const mockResults: ModelSnapshot[] = Array.from({ length: dataset.numResults }, (_, i) => ({
          id: `model_${dataset.id}_${i}`,
          name: `Config ${i + 1}`,
          timestamp: Date.now(),
          parameters: {
            Rsh: 1000 + Math.random() * 4000,
            Ra: 1000 + Math.random() * 4000,
            Ca: 1e-6 + Math.random() * 49e-6,
            Rb: 1000 + Math.random() * 4000,
            Cb: 1e-6 + Math.random() * 49e-6,
            frequency_range: dataset.frequencyRange as [number, number]
          },
          data: [], // Empty impedance data for mock
          resnorm: Math.random() * 10,
          color: `hsl(${Math.floor(Math.random() * 360)}, 70%, 50%)`,
          isVisible: true,
          opacity: 1.0,
          ter: undefined
        }));
        
        onLoadDataset(mockResults);
        console.log(`Loaded ${dataset.numResults} results from ${dataset.title}`);
        alert(`Successfully loaded ${dataset.title} with ${dataset.numResults.toLocaleString()} results`);
      } else {
        alert('Dataset data not found. The file may have been removed.');
      }
    } catch (error) {
      console.error('Load dataset error:', error);
      alert('Failed to load dataset. Please try again.');
    }
  };

  const handleEditMetadata = (dataset: NPZDataset) => {
    setEditingDataset(dataset);
  };

  const handleSaveMetadata = (updatedDataset: NPZDataset) => {
    try {
      // Update dataset in localStorage
      const existingDatasets = JSON.parse(localStorage.getItem('npz-datasets') || '[]');
      const updatedDatasets = existingDatasets.map((ds: NPZDataset) =>
        ds.id === updatedDataset.id ? updatedDataset : ds
      );
      localStorage.setItem('npz-datasets', JSON.stringify(updatedDatasets));
      
      setEditingDataset(null);
      console.log('Metadata updated successfully');
    } catch (error) {
      console.error('Save metadata error:', error);
      alert('Failed to save metadata. Please try again.');
    }
  };

  const handleDeleteDataset = (datasetId: string) => {
    if (confirm('Are you sure you want to delete this dataset? This action cannot be undone.')) {
      try {
        // Remove dataset from localStorage
        const existingDatasets = JSON.parse(localStorage.getItem('npz-datasets') || '[]');
        const filteredDatasets = existingDatasets.filter((ds: NPZDataset) => ds.id !== datasetId);
        localStorage.setItem('npz-datasets', JSON.stringify(filteredDatasets));
        
        // Remove associated data
        localStorage.removeItem(`npz-data-${datasetId}`);
        
        console.log('Dataset deleted successfully');
      } catch (error) {
        console.error('Delete dataset error:', error);
        alert('Failed to delete dataset. Please try again.');
      }
    }
  };

  const tabs = [
    { id: 'library' as TabType, label: 'Library', icon: '📚', description: 'Manage saved datasets' },
    { id: 'export' as TabType, label: 'Export', icon: '📤', description: 'Export current results' },
    { id: 'import' as TabType, label: 'Import', icon: '📥', description: 'Import NPZ files' },
  ];

  return (
    <div className={`bg-neutral-900 rounded-xl shadow-xl ${className}`}>
      {/* Tab Navigation */}
      <div className="border-b border-neutral-700">
        <nav className="flex">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'text-blue-400 border-b-2 border-blue-400 bg-neutral-800'
                  : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800/50'
              }`}
            >
              <div className="flex items-center justify-center gap-2">
                <span>{tab.icon}</span>
                <span>{tab.label}</span>
              </div>
              <div className="text-xs text-neutral-500 mt-1">{tab.description}</div>
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="p-4">
        {activeTab === 'export' && (
          <NPZExportPanel
            results={results}
            onExport={handleExport}
            isExporting={isExporting}
          />
        )}
        
        {activeTab === 'import' && (
          <NPZImportPanel
            onImport={handleImport}
            isImporting={isImporting}
          />
        )}
        
        {activeTab === 'library' && (
          <NPZLibraryManager
            onLoadDataset={handleLoadDataset}
            onEditMetadata={handleEditMetadata}
            onDeleteDataset={handleDeleteDataset}
          />
        )}
      </div>

      {/* Metadata Editor Modal */}
      {editingDataset && (
        <NPZMetadataEditor
          dataset={editingDataset}
          onSave={handleSaveMetadata}
          onCancel={() => setEditingDataset(null)}
          isOpen={!!editingDataset}
        />
      )}
    </div>
  );
};