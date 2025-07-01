import React, { useState } from 'react';
import { ResnormGroup } from './utils/types';

interface OrchestratorTabProps {
  resnormGroups: ResnormGroup[];
}

interface RenderJob {
  id: string;
  name: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  imageUrl?: string;
  thumbnailUrl?: string;
  parameters: {
    format: 'png' | 'svg' | 'pdf';
    resolution: string;
    includeLabels: boolean;
    chromaEnabled: boolean;
    selectedGroups: number[];
  };
  createdAt: Date;
}

export const OrchestratorTab: React.FC<OrchestratorTabProps> = ({
  resnormGroups
}) => {
  const [renderJobs, setRenderJobs] = useState<RenderJob[]>([]);
  const [isCreatingJob, setIsCreatingJob] = useState(false);
  
  // Job creation form state
  const [jobName, setJobName] = useState('');
  const [format, setFormat] = useState<'png' | 'svg' | 'pdf'>('png');
  const [resolution, setResolution] = useState('1920x1080');
  const [includeLabels, setIncludeLabels] = useState(true);
  const [chromaEnabled, setChromaEnabled] = useState(true);
  const [selectedGroups, setSelectedGroups] = useState<number[]>([]);

  const handleCreateJob = () => {
    if (!jobName.trim()) return;
    
    const newJob: RenderJob = {
      id: `job_${Date.now()}`,
      name: jobName,
      status: 'pending',
      progress: 0,
      parameters: {
        format,
        resolution,
        includeLabels,
        chromaEnabled,
        selectedGroups: selectedGroups.length > 0 ? selectedGroups : resnormGroups.map((_, i) => i)
      },
      createdAt: new Date()
    };
    
    setRenderJobs(prev => [newJob, ...prev]);
    
    // Reset form
    setJobName('');
    setIsCreatingJob(false);
    
    // Simulate processing (in real implementation, this would be handled by a worker)
    setTimeout(() => {
      setRenderJobs(prev => prev.map(job => 
        job.id === newJob.id ? { ...job, status: 'processing' } : job
      ));
      
      // Simulate progress updates
      let progress = 0;
      const progressInterval = setInterval(() => {
        progress += Math.random() * 20;
        if (progress >= 100) {
          progress = 100;
          clearInterval(progressInterval);
          setRenderJobs(prev => prev.map(job => 
            job.id === newJob.id ? { 
              ...job, 
              status: 'completed', 
              progress: 100,
              imageUrl: `/api/renders/${newJob.id}.${format}`,
              thumbnailUrl: `/api/renders/${newJob.id}_thumb.png`
            } : job
          ));
        } else {
          setRenderJobs(prev => prev.map(job => 
            job.id === newJob.id ? { ...job, progress } : job
          ));
        }
      }, 500);
    }, 1000);
  };

  const handleDeleteJob = (jobId: string) => {
    setRenderJobs(prev => prev.filter(job => job.id !== jobId));
  };

  const handleDownload = (job: RenderJob) => {
    if (job.imageUrl) {
      // Create a temporary download link
      const link = document.createElement('a');
      link.href = job.imageUrl;
      link.download = `${job.name}.${job.parameters.format}`;
      link.click();
    }
  };

  const getStatusColor = (status: RenderJob['status']) => {
    switch (status) {
      case 'pending': return 'text-amber-400';
      case 'processing': return 'text-blue-400';
      case 'completed': return 'text-green-400';
      case 'failed': return 'text-red-400';
      default: return 'text-neutral-400';
    }
  };

  const getStatusIcon = (status: RenderJob['status']) => {
    switch (status) {
      case 'pending':
        return (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
      case 'processing':
        return (
          <svg className="w-4 h-4 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        );
      case 'completed':
        return (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        );
      case 'failed':
        return (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        );
      default:
        return null;
    }
  };

  return (
    <div className="h-full flex flex-col bg-neutral-950 p-4">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-white mb-2">Orchestrator</h2>
        <p className="text-neutral-400 text-sm">
          Generate and download high-quality spider plot images in batch
        </p>
      </div>

      {/* Create Job Section */}
      <div className="bg-neutral-800 rounded-lg border border-neutral-700 p-4 mb-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-white">Create Render Job</h3>
          <button
            onClick={() => setIsCreatingJob(!isCreatingJob)}
            className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-sm transition-colors"
          >
            {isCreatingJob ? 'Cancel' : 'New Job'}
          </button>
        </div>

        {isCreatingJob && (
          <div className="space-y-4">
            {/* Job Name */}
            <div>
              <label className="block text-sm font-medium text-neutral-200 mb-2">Job Name</label>
              <input
                type="text"
                value={jobName}
                onChange={(e) => setJobName(e.target.value)}
                placeholder="Enter job name..."
                className="w-full bg-neutral-700 text-white px-3 py-2 rounded-md border border-neutral-600 focus:border-blue-500 focus:outline-none"
              />
            </div>

            {/* Format and Resolution */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-neutral-200 mb-2">Format</label>
                <select
                  value={format}
                  onChange={(e) => setFormat(e.target.value as 'png' | 'svg' | 'pdf')}
                  className="w-full bg-neutral-700 text-white px-3 py-2 rounded-md border border-neutral-600 focus:border-blue-500 focus:outline-none"
                >
                  <option value="png">PNG</option>
                  <option value="svg">SVG</option>
                  <option value="pdf">PDF</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-neutral-200 mb-2">Resolution</label>
                <select
                  value={resolution}
                  onChange={(e) => setResolution(e.target.value)}
                  className="w-full bg-neutral-700 text-white px-3 py-2 rounded-md border border-neutral-600 focus:border-blue-500 focus:outline-none"
                >
                  <option value="1920x1080">1920x1080 (HD)</option>
                  <option value="2560x1440">2560x1440 (QHD)</option>
                  <option value="3840x2160">3840x2160 (4K)</option>
                  <option value="7680x4320">7680x4320 (8K)</option>
                </select>
              </div>
            </div>

            {/* Options */}
            <div className="flex gap-4">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={includeLabels}
                  onChange={(e) => setIncludeLabels(e.target.checked)}
                  className="rounded border-neutral-600"
                />
                <span className="text-sm text-neutral-200">Include Labels</span>
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={chromaEnabled}
                  onChange={(e) => setChromaEnabled(e.target.checked)}
                  className="rounded border-neutral-600"
                />
                <span className="text-sm text-neutral-200">Color Mode</span>
              </label>
            </div>

            {/* Group Selection */}
            <div>
              <label className="block text-sm font-medium text-neutral-200 mb-2">Groups to Include</label>
              <div className="flex gap-2 flex-wrap">
                <button
                  onClick={() => setSelectedGroups(selectedGroups.length === resnormGroups.length ? [] : resnormGroups.map((_, i) => i))}
                  className={`px-3 py-1.5 text-xs rounded-md border transition-colors ${
                    selectedGroups.length === resnormGroups.length
                      ? 'bg-blue-600 text-white border-blue-500'
                      : 'bg-neutral-700 text-neutral-300 border-neutral-600 hover:bg-neutral-600'
                  }`}
                >
                  {selectedGroups.length === resnormGroups.length ? 'Deselect All' : 'Select All'}
                </button>
                {resnormGroups.map((group, index) => (
                  <button
                    key={index}
                    onClick={() => {
                      if (selectedGroups.includes(index)) {
                        setSelectedGroups(selectedGroups.filter(i => i !== index));
                      } else {
                        setSelectedGroups([...selectedGroups, index]);
                      }
                    }}
                    className={`px-3 py-1.5 text-xs rounded-md border transition-colors flex items-center gap-1.5 ${
                      selectedGroups.includes(index)
                        ? 'bg-blue-600 text-white border-blue-500'
                        : 'bg-neutral-700 text-neutral-300 border-neutral-600 hover:bg-neutral-600'
                    }`}
                  >
                    <div className="w-2 h-2 rounded-full" style={{ backgroundColor: group.color }} />
                    {group.label.split(' ')[0]}
                  </button>
                ))}
              </div>
            </div>

            <button
              onClick={handleCreateJob}
              disabled={!jobName.trim()}
              className="w-full py-2.5 bg-green-600 hover:bg-green-700 disabled:bg-neutral-600 disabled:cursor-not-allowed text-white rounded-md font-medium transition-colors"
            >
              Create Render Job
            </button>
          </div>
        )}
      </div>

      {/* Jobs List */}
      <div className="flex-1 bg-neutral-800 rounded-lg border border-neutral-700 overflow-hidden">
        <div className="p-4 border-b border-neutral-700">
          <h3 className="text-lg font-medium text-white">Render Jobs ({renderJobs.length})</h3>
        </div>
        
        <div className="p-4 overflow-y-auto custom-scrollbar max-h-[500px]">
          {renderJobs.length === 0 ? (
            <div className="text-center py-8">
              <svg className="w-12 h-12 mx-auto mb-4 text-neutral-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <p className="text-neutral-400 text-sm">No render jobs yet</p>
              <p className="text-neutral-500 text-xs mt-1">Create a new job to get started</p>
            </div>
          ) : (
            <div className="space-y-3">
              {renderJobs.map((job) => (
                <div key={job.id} className="bg-neutral-700 rounded-lg border border-neutral-600 p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <h4 className="font-medium text-white">{job.name}</h4>
                        <div className={`flex items-center gap-1 ${getStatusColor(job.status)}`}>
                          {getStatusIcon(job.status)}
                          <span className="text-xs capitalize">{job.status}</span>
                        </div>
                      </div>
                      <p className="text-xs text-neutral-400">
                        {job.parameters.format.toUpperCase()} • {job.parameters.resolution} • 
                        {job.parameters.includeLabels ? ' Labels' : ' No Labels'} • 
                        {job.parameters.chromaEnabled ? ' Color' : ' Mono'}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      {job.status === 'completed' && (
                        <button
                          onClick={() => handleDownload(job)}
                          className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded text-xs transition-colors"
                        >
                          Download
                        </button>
                      )}
                      <button
                        onClick={() => handleDeleteJob(job.id)}
                        className="px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white rounded text-xs transition-colors"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                  
                  {job.status === 'processing' && (
                    <div className="w-full bg-neutral-600 rounded-full h-2">
                      <div 
                        className="bg-blue-500 h-2 rounded-full transition-all duration-300" 
                        style={{ width: `${job.progress}%` }}
                      />
                    </div>
                  )}
                  
                  {job.status === 'completed' && job.thumbnailUrl && (
                    <div className="mt-3">
                      <img 
                        src={job.thumbnailUrl} 
                        alt={`Preview of ${job.name}`}
                        className="w-32 h-20 object-cover rounded border border-neutral-600"
                      />
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}; 