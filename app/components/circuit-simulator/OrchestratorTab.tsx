import React, { useState, useCallback, useRef, useEffect } from 'react';
import { ResnormGroup } from './utils/types';
import { SavedProfile } from './types/savedProfiles';

interface OrchestratorTabProps {
  resnormGroups: ResnormGroup[];
  savedProfiles?: SavedProfile[];
  onComputeProfiles?: (profileIds: string[]) => Promise<void>;
}

interface RenderJob {
  id: string;
  name: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  imageUrl?: string;
  thumbnailUrl?: string;
  parameters: {
    format: 'png' | 'svg' | 'pdf' | 'webp' | 'jpg';
    resolution: string;
    quality: number;
    includeLabels: boolean;
    chromaEnabled: boolean;
    selectedGroups: number[];
    backgroundColor: 'transparent' | 'white' | 'black';
  };
  createdAt: Date;
  completedAt?: Date;
  fileSize?: string;
  error?: string;
}

interface DownloadOptions {
  format: 'png' | 'svg' | 'pdf' | 'webp' | 'jpg';
  quality: number;
  scale: number;
}

export const OrchestratorTab: React.FC<OrchestratorTabProps> = ({
  resnormGroups,
  savedProfiles = [],
  onComputeProfiles
}) => {
  const [renderJobs, setRenderJobs] = useState<RenderJob[]>([]);
  const [isCreatingJob, setIsCreatingJob] = useState(false);
  const [hasLoadedJobs, setHasLoadedJobs] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // Profile computation state
  const [selectedProfiles, setSelectedProfiles] = useState<string[]>([]);
  const [isComputingProfiles, setIsComputingProfiles] = useState(false);
  const [computeProgress, setComputeProgress] = useState<{ current: number; total: number } | null>(null);
  
  // Job creation form state
  const [jobName, setJobName] = useState('');
  const [format, setFormat] = useState<'png' | 'svg' | 'pdf' | 'webp' | 'jpg'>('png');
  const [resolution, setResolution] = useState('1920x1080');
  const [quality, setQuality] = useState(95);
  const [includeLabels, setIncludeLabels] = useState(true);
  const [chromaEnabled, setChromaEnabled] = useState(true);
  const [backgroundColor, setBackgroundColor] = useState<'transparent' | 'white' | 'black'>('transparent');
  const [selectedGroups, setSelectedGroups] = useState<number[]>([]);

  // Download options for individual jobs
  const [downloadOptions, setDownloadOptions] = useState<{ [jobId: string]: DownloadOptions }>({});

  // Helper function for file size formatting
  const formatFileSize = useCallback((bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }, []);

  // Load render jobs from localStorage on mount
  useEffect(() => {
    if (typeof window !== 'undefined' && !hasLoadedJobs) {
      try {
        const savedJobs = localStorage.getItem('nei-viz-render-jobs');
        if (savedJobs) {
          const jobs = JSON.parse(savedJobs) as RenderJob[];
          // Convert date strings back to Date objects
          const parsedJobs = jobs.map(job => ({
            ...job,
            createdAt: new Date(job.createdAt),
            completedAt: job.completedAt ? new Date(job.completedAt) : undefined
          }));
          setRenderJobs(parsedJobs);
        }
      } catch (error) {
        console.warn('Failed to load render jobs from localStorage:', error);
      }
      setHasLoadedJobs(true);
    }
  }, [hasLoadedJobs]);

  // Save render jobs to localStorage whenever they change (but only after initial load)
  useEffect(() => {
    if (typeof window !== 'undefined' && hasLoadedJobs) {
      try {
        localStorage.setItem('nei-viz-render-jobs', JSON.stringify(renderJobs));
      } catch (error) {
        console.warn('Failed to save render jobs to localStorage:', error);
      }
    }
  }, [renderJobs, hasLoadedJobs]);

  // Generate image from current visualization
  const generateImage = useCallback(async (job: RenderJob): Promise<{ imageUrl: string; thumbnailUrl: string; fileSize: string }> => {
    return new Promise((resolve, reject) => {
      try {
        // Create high-quality canvas for rendering
        const canvas = document.createElement('canvas');
        const [width, height] = job.parameters.resolution.split('x').map(Number);
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d')!;

        // Enable high-quality rendering
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';

        // Set background
        if (job.parameters.backgroundColor === 'white') {
          ctx.fillStyle = '#ffffff';
          ctx.fillRect(0, 0, width, height);
        } else if (job.parameters.backgroundColor === 'black') {
          ctx.fillStyle = '#000000';
          ctx.fillRect(0, 0, width, height);
        } else {
          // Transparent background - clear the canvas
          ctx.clearRect(0, 0, width, height);
        }

        // Generate actual spider plot visualization
        const centerX = width / 2;
        const centerY = height / 2;
        const maxRadius = Math.min(width, height) * 0.4;

        // Draw spider plot grid (5 parameters: Rs, Ra, Rb, Ca, Cb)
        const params = ['Rs', 'Ra', 'Rb', 'Ca', 'Cb'];
        const angleStep = (2 * Math.PI) / params.length;

        // Grid styling
        ctx.strokeStyle = job.parameters.chromaEnabled ? '#4B5563' : '#6B7280';
        ctx.lineWidth = Math.max(1, width / 1000);

        // Draw concentric circles
        for (let i = 1; i <= 5; i++) {
          const radius = (maxRadius * i) / 5;
          ctx.beginPath();
          ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
          ctx.stroke();
        }

        // Draw radial axes
        ctx.lineWidth = Math.max(1, width / 800);
        for (let i = 0; i < params.length; i++) {
          const angle = i * angleStep - Math.PI / 2;
          const endX = centerX + Math.cos(angle) * maxRadius;
          const endY = centerY + Math.sin(angle) * maxRadius;
          
          ctx.beginPath();
          ctx.moveTo(centerX, centerY);
          ctx.lineTo(endX, endY);
          ctx.stroke();
        }

        // Add parameter labels if requested
        if (job.parameters.includeLabels) {
          ctx.fillStyle = job.parameters.chromaEnabled ? '#E5E7EB' : '#D1D5DB';
          ctx.font = `${Math.max(12, width / 50)}px Inter, sans-serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';

          for (let i = 0; i < params.length; i++) {
            const angle = i * angleStep - Math.PI / 2;
            const labelDistance = maxRadius + Math.max(20, width / 40);
            const labelX = centerX + Math.cos(angle) * labelDistance;
            const labelY = centerY + Math.sin(angle) * labelDistance;
            
            ctx.fillText(params[i], labelX, labelY);
          }
        }

        // Draw selected resnorm groups data
        const renderedCount = { total: 0 };
        job.parameters.selectedGroups.forEach((groupIndex) => {
          if (groupIndex < resnormGroups.length) {
            const group = resnormGroups[groupIndex];
            ctx.fillStyle = group.color;
            ctx.strokeStyle = group.color;
            ctx.globalAlpha = job.parameters.chromaEnabled ? 0.6 : 0.8;
            ctx.lineWidth = Math.max(0.5, width / 2000);

            // Render polygons for this group (limit for performance)
            const maxPolygonsPerGroup = Math.min(1000, Math.floor(10000 / job.parameters.selectedGroups.length));
            const polygonsToRender = group.items.slice(0, maxPolygonsPerGroup);
            
            polygonsToRender.forEach((model) => {
              if (model.parameters) {
                // Normalize parameters to 0-1 range for visualization
                const normalizedValues = [
                  Math.min(1, Math.max(0, (model.parameters.Rs - 10) / (10000 - 10))),
                  Math.min(1, Math.max(0, (model.parameters.Ra - 10) / (10000 - 10))),
                  Math.min(1, Math.max(0, (model.parameters.Rb - 10) / (10000 - 10))),
                  Math.min(1, Math.max(0, (model.parameters.Ca - 0.1e-6) / (50e-6 - 0.1e-6))),
                  Math.min(1, Math.max(0, (model.parameters.Cb - 0.1e-6) / (50e-6 - 0.1e-6)))
                ];

                // Draw spider polygon
                ctx.beginPath();
                for (let j = 0; j < normalizedValues.length; j++) {
                  const angle = j * angleStep - Math.PI / 2;
                  const radius = normalizedValues[j] * maxRadius;
                  const x = centerX + Math.cos(angle) * radius;
                  const y = centerY + Math.sin(angle) * radius;
                  
                  if (j === 0) {
                    ctx.moveTo(x, y);
                  } else {
                    ctx.lineTo(x, y);
                  }
                }
                ctx.closePath();
                ctx.stroke();
                renderedCount.total++;
              }
            });
          }
        });

        ctx.globalAlpha = 1;

        // Add title and info if labels are enabled
        if (job.parameters.includeLabels) {
          ctx.fillStyle = job.parameters.chromaEnabled ? '#F9FAFB' : '#E5E7EB';
          ctx.font = `bold ${Math.max(16, width / 40)}px Inter, sans-serif`;
          ctx.textAlign = 'center';
          ctx.fillText('Spider Plot Visualization', centerX, Math.max(30, height / 25));
          
          ctx.font = `${Math.max(12, width / 80)}px Inter, sans-serif`;
          ctx.fillText(
            `${job.parameters.selectedGroups.length} Groups • ${renderedCount.total.toLocaleString()} Polygons • ${job.parameters.resolution}`,
            centerX,
            height - Math.max(20, height / 40)
          );
        }

        // Convert to requested format with proper quality
        let imageUrl: string;
        let fileSize: string;

        if (job.parameters.format === 'png') {
          imageUrl = canvas.toDataURL('image/png');
          fileSize = formatFileSize(imageUrl.length * 0.75); // Estimate based on base64 length
        } else if (job.parameters.format === 'jpg') {
          imageUrl = canvas.toDataURL('image/jpeg', job.parameters.quality / 100);
          fileSize = formatFileSize(imageUrl.length * 0.75 * (job.parameters.quality / 100));
        } else if (job.parameters.format === 'webp') {
          // WebP support detection
          const webpSupported = canvas.toDataURL('image/webp').indexOf('image/webp') === 5;
          if (webpSupported) {
            imageUrl = canvas.toDataURL('image/webp', job.parameters.quality / 100);
            fileSize = formatFileSize(imageUrl.length * 0.6 * (job.parameters.quality / 100)); // WebP is more efficient
          } else {
            // Fallback to PNG
            imageUrl = canvas.toDataURL('image/png');
            fileSize = formatFileSize(imageUrl.length * 0.75);
          }
        } else {
          // SVG/PDF fallback to PNG
          imageUrl = canvas.toDataURL('image/png');
          fileSize = formatFileSize(imageUrl.length * 0.75);
        }

        // Create high-quality thumbnail
        const thumbCanvas = document.createElement('canvas');
        thumbCanvas.width = 320;
        thumbCanvas.height = 200;
        const thumbCtx = thumbCanvas.getContext('2d')!;
        thumbCtx.imageSmoothingEnabled = true;
        thumbCtx.imageSmoothingQuality = 'high';
        
        // Draw thumbnail with proper aspect ratio
        const aspectRatio = width / height;
        let thumbWidth = 320;
        let thumbHeight = 200;
        let offsetX = 0;
        let offsetY = 0;
        
        if (aspectRatio > 320/200) {
          thumbHeight = 320 / aspectRatio;
          offsetY = (200 - thumbHeight) / 2;
        } else {
          thumbWidth = 200 * aspectRatio;
          offsetX = (320 - thumbWidth) / 2;
        }
        
        // Dark background for thumbnail
        thumbCtx.fillStyle = '#1F2937';
        thumbCtx.fillRect(0, 0, 320, 200);
        
        thumbCtx.drawImage(canvas, offsetX, offsetY, thumbWidth, thumbHeight);
        const thumbnailUrl = thumbCanvas.toDataURL('image/png');

        resolve({ imageUrl, thumbnailUrl, fileSize });
        
      } catch (error) {
        console.error('Image generation failed:', error);
        reject(new Error(`Failed to generate image: ${error instanceof Error ? error.message : 'Unknown error'}`));
      }
    });
  }, [resnormGroups, formatFileSize]);

  // Handle profile computation with progress tracking
  const handleComputeSelectedProfiles = async () => {
    if (selectedProfiles.length === 0 || !onComputeProfiles) return;
    
    setIsComputingProfiles(true);
    setComputeProgress({ current: 0, total: selectedProfiles.length });
    
    try {
      for (let i = 0; i < selectedProfiles.length; i++) {
        setComputeProgress({ current: i, total: selectedProfiles.length });
        await onComputeProfiles([selectedProfiles[i]]);
        // Add a small delay to show progress
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      setComputeProgress({ current: selectedProfiles.length, total: selectedProfiles.length });
    } catch (error) {
      console.error('Failed to compute profiles:', error);
    } finally {
      setTimeout(() => {
        setIsComputingProfiles(false);
        setComputeProgress(null);
      }, 1000);
    }
  };

  const handleCreateJob = () => {
    if (!jobName.trim()) return;
    
    const newJob: RenderJob = {
      id: `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name: jobName,
      status: 'pending',
      progress: 0,
      parameters: {
        format,
        resolution,
        quality,
        includeLabels,
        chromaEnabled,
        backgroundColor,
        selectedGroups: selectedGroups.length > 0 ? selectedGroups : resnormGroups.map((_, i) => i)
      },
      createdAt: new Date()
    };
    
    setRenderJobs(prev => [newJob, ...prev]);
    
    // Reset form
    setJobName('');
    setIsCreatingJob(false);
    
    // Start processing
    setTimeout(async () => {
      setRenderJobs(prev => prev.map(job => 
        job.id === newJob.id ? { ...job, status: 'processing', progress: 0 } : job
      ));
      
      // Realistic processing simulation based on job complexity
      const [width, height] = newJob.parameters.resolution.split('x').map(Number);
      const totalPixels = width * height;
      const groupCount = newJob.parameters.selectedGroups.length;
      const complexityFactor = (totalPixels / 1000000) * groupCount; // Normalize complexity
      const estimatedDuration = Math.min(8000, Math.max(1000, complexityFactor * 500)); // 1-8 seconds
      
      // Multi-stage progress simulation
      const stages = [
        { name: 'preparing', duration: 0.1, startProgress: 0 },
        { name: 'rendering_grid', duration: 0.15, startProgress: 10 },
        { name: 'processing_groups', duration: 0.6, startProgress: 25 },
        { name: 'finalizing', duration: 0.15, startProgress: 85 }
      ];
      
             let currentStageIndex = 0;
       let stageStartTime = Date.now();
       const overallStartTime = Date.now();
      
      const progressInterval = setInterval(() => {
        const elapsed = Date.now() - overallStartTime;
        const stageElapsed = Date.now() - stageStartTime;
        const currentStage = stages[currentStageIndex];
        
        if (currentStage) {
          const stageDuration = currentStage.duration * estimatedDuration;
          const stageProgress = Math.min(1, stageElapsed / stageDuration);
          
          // Calculate realistic progress with some variance
          let progress = currentStage.startProgress + (stageProgress * (
            currentStageIndex < stages.length - 1 
              ? stages[currentStageIndex + 1].startProgress - currentStage.startProgress
              : 100 - currentStage.startProgress
          ));
          
          // Add slight randomness for realism
          progress += (Math.random() - 0.5) * 2;
          progress = Math.max(currentStage.startProgress, Math.min(progress, 
            currentStageIndex < stages.length - 1 ? stages[currentStageIndex + 1].startProgress : 100
          ));
          
          setRenderJobs(prev => prev.map(job => 
            job.id === newJob.id ? { ...job, progress: Math.round(progress) } : job
          ));
          
          // Move to next stage
          if (stageProgress >= 1 && currentStageIndex < stages.length - 1) {
            currentStageIndex++;
            stageStartTime = Date.now();
          }
          
          // Complete the job
          if (elapsed >= estimatedDuration || progress >= 99) {
            clearInterval(progressInterval);
            
            setRenderJobs(prev => prev.map(job => 
              job.id === newJob.id ? { ...job, progress: 100 } : job
            ));
            
            // Generate the actual image
            generateImage(newJob).then(({ imageUrl, thumbnailUrl, fileSize }) => {
              setRenderJobs(prev => prev.map(job => 
                job.id === newJob.id ? { 
                  ...job, 
                  status: 'completed', 
                  progress: 100,
                  imageUrl,
                  thumbnailUrl,
                  fileSize,
                  completedAt: new Date()
                } : job
              ));
            }).catch((error) => {
              console.error('Image generation failed:', error);
              setRenderJobs(prev => prev.map(job => 
                job.id === newJob.id ? { 
                  ...job, 
                  status: 'failed', 
                  progress: 0,
                  error: error.message || 'Failed to generate image'
                } : job
              ));
            });
          }
        }
      }, 100); // More frequent updates for smoother progress
    }, 500);
  };

  const handleDeleteJob = useCallback((jobId: string) => {
    setRenderJobs(prev => prev.filter(job => job.id !== jobId));
    setDownloadOptions(prev => {
      const newOptions = { ...prev };
      delete newOptions[jobId];
      return newOptions;
    });
  }, []);

  const handleClearAllJobs = useCallback(() => {
    const confirm = window.confirm('Are you sure you want to clear all render jobs? This action cannot be undone.');
    if (confirm) {
      setRenderJobs([]);
      setDownloadOptions({});
    }
  }, []);

  const handleClearCompletedJobs = useCallback(() => {
    setRenderJobs(prev => prev.filter(job => job.status !== 'completed'));
    setDownloadOptions(prev => {
      const newOptions = { ...prev };
      renderJobs.forEach(job => {
        if (job.status === 'completed') {
          delete newOptions[job.id];
        }
      });
      return newOptions;
    });
  }, [renderJobs]);

  const handleDownload = useCallback((job: RenderJob, customOptions?: DownloadOptions) => {
    if (!job.imageUrl) {
      console.error('No image URL available for download');
      return;
    }

    try {
      const options = customOptions || downloadOptions[job.id] || {
        format: job.parameters.format,
        quality: job.parameters.quality,
        scale: 1
      };

      // If custom options are provided, regenerate with those settings
      if (customOptions && (
        customOptions.format !== job.parameters.format || 
        customOptions.quality !== job.parameters.quality || 
        customOptions.scale !== 1
      )) {
        // Create a new canvas with custom settings
        const img = new Image();
        img.onload = () => {
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d')!;
          
          // Apply scale
          canvas.width = img.width * options.scale;
          canvas.height = img.height * options.scale;
          
          ctx.imageSmoothingEnabled = true;
          ctx.imageSmoothingQuality = 'high';
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          
          // Convert to desired format
          let dataUrl: string;
          if (options.format === 'png') {
            dataUrl = canvas.toDataURL('image/png');
          } else if (options.format === 'jpg') {
            dataUrl = canvas.toDataURL('image/jpeg', options.quality / 100);
          } else if (options.format === 'webp') {
            dataUrl = canvas.toDataURL('image/webp', options.quality / 100);
          } else {
            dataUrl = canvas.toDataURL('image/png'); // Fallback
          }
          
          // Download the processed image
          downloadDataUrl(dataUrl, `${job.name.replace(/[^a-z0-9]/gi, '_')}.${options.format}`);
        };
        img.src = job.imageUrl;
      } else {
        // Download original image
        downloadDataUrl(job.imageUrl, `${job.name.replace(/[^a-z0-9]/gi, '_')}.${job.parameters.format}`);
      }
    } catch (error) {
      console.error('Download failed:', error);
    }
  }, [downloadOptions]);

  // Helper function to download data URL
  const downloadDataUrl = useCallback((dataUrl: string, filename: string) => {
    try {
      // Create download link
      const link = document.createElement('a');
      link.href = dataUrl;
      link.download = filename;
      
      // Add to DOM, click, and remove
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      console.log(`Downloaded: ${filename}`);
    } catch (error) {
      console.error('Failed to download file:', error);
      
      // Fallback: open in new window
      try {
        const newWindow = window.open(dataUrl, '_blank');
        if (!newWindow) {
          alert('Download failed. Please check your browser settings and try again.');
        }
      } catch (fallbackError) {
        console.error('Fallback download also failed:', fallbackError);
        alert('Download failed. Please try a different browser or check your settings.');
      }
    }
  }, []);

  const updateDownloadOptions = (jobId: string, options: Partial<DownloadOptions>) => {
    setDownloadOptions(prev => ({
      ...prev,
      [jobId]: { ...prev[jobId], ...options } as DownloadOptions
    }));
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
    <div className="h-full bg-neutral-950 flex">
      {/* Hidden canvas for image generation */}
      <canvas ref={canvasRef} className="hidden" />
      
      {/* Left Panel - Job Creation and Profile Management */}
      <div className="w-1/3 min-w-[400px] max-w-[500px] bg-neutral-900 border-r border-neutral-700 flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-neutral-700">
          <h2 className="text-xl font-bold text-white mb-2">Orchestrator</h2>
          <p className="text-neutral-400 text-sm">
            Generate and download high-quality spider plot images
          </p>
        </div>

        <div className="flex-1 overflow-y-auto">
          {/* Saved Profiles Section */}
          {savedProfiles.length > 0 && (
            <div className="p-6 border-b border-neutral-700">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium text-white">Batch Compute</h3>
                <div className="text-sm text-neutral-400">
                  {selectedProfiles.length}/{savedProfiles.length}
                </div>
              </div>

              {isComputingProfiles && computeProgress && (
                <div className="mb-4 p-3 bg-blue-600/10 border border-blue-600/20 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-blue-300">Computing profiles...</span>
                    <span className="text-sm text-blue-300">
                      {computeProgress.current}/{computeProgress.total}
                    </span>
                  </div>
                  <div className="w-full bg-neutral-700 rounded-full h-2">
                    <div 
                      className="bg-blue-500 h-2 rounded-full transition-all duration-300" 
                      style={{ width: `${(computeProgress.current / computeProgress.total) * 100}%` }}
                    />
                  </div>
                </div>
              )}

              <div className="space-y-3 mb-4">
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setSelectedProfiles(
                      selectedProfiles.length === savedProfiles.length 
                        ? [] 
                        : savedProfiles.map(p => p.id)
                    )}
                    className="px-3 py-1.5 text-xs bg-neutral-700 hover:bg-neutral-600 text-neutral-300 rounded-md border border-neutral-600 transition-colors"
                  >
                    {selectedProfiles.length === savedProfiles.length ? 'Deselect All' : 'Select All'}
                  </button>
                </div>

                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {savedProfiles.map((profile) => (
                    <div
                      key={profile.id}
                      onClick={() => {
                        if (selectedProfiles.includes(profile.id)) {
                          setSelectedProfiles(selectedProfiles.filter(id => id !== profile.id));
                        } else {
                          setSelectedProfiles([...selectedProfiles, profile.id]);
                        }
                      }}
                      className={`cursor-pointer p-3 rounded-lg border transition-all duration-200 ${
                        selectedProfiles.includes(profile.id)
                          ? 'bg-blue-600/20 border-blue-500'
                          : 'bg-neutral-800 border-neutral-600 hover:bg-neutral-700'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <h4 className="text-sm font-medium text-white truncate">{profile.name}</h4>
                        <div className="flex items-center gap-1">
                          {profile.isComputed && (
                            <svg className="w-3 h-3 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                          )}
                          {selectedProfiles.includes(profile.id) && (
                            <svg className="w-3 h-3 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-neutral-400">
                        <span className="bg-neutral-700 px-1.5 py-0.5 rounded">
                          {profile.gridSize}⁵ grid
                        </span>
                        <span>{profile.minFreq}-{profile.maxFreq} Hz</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <button
                onClick={handleComputeSelectedProfiles}
                disabled={selectedProfiles.length === 0 || isComputingProfiles}
                className="w-full py-2.5 bg-green-600 hover:bg-green-700 disabled:bg-neutral-600 disabled:cursor-not-allowed text-white rounded-md font-medium transition-colors flex items-center justify-center gap-2"
              >
                {isComputingProfiles ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                    Computing...
                  </>
                ) : (
                  <>
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    Compute Selected
                  </>
                )}
              </button>
            </div>
          )}

          {/* Create Job Section */}
          <div className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-white">Create Render Job</h3>
              <button
                onClick={() => setIsCreatingJob(!isCreatingJob)}
                className={`px-3 py-1.5 rounded-md text-sm transition-colors ${
                  isCreatingJob 
                    ? 'bg-red-600 hover:bg-red-700 text-white'
                    : 'bg-blue-600 hover:bg-blue-700 text-white'
                }`}
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
                    className="w-full bg-neutral-800 text-white px-3 py-2 rounded-md border border-neutral-600 focus:border-blue-500 focus:outline-none"
                  />
                </div>

                {/* Format and Resolution */}
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-sm font-medium text-neutral-200 mb-2">Format</label>
                    <select
                      value={format}
                      onChange={(e) => setFormat(e.target.value as typeof format)}
                      className="w-full bg-neutral-800 text-white px-3 py-2 rounded-md border border-neutral-600 focus:border-blue-500 focus:outline-none"
                    >
                      <option value="png">PNG</option>
                      <option value="jpg">JPEG</option>
                      <option value="webp">WebP</option>
                      <option value="svg">SVG</option>
                      <option value="pdf">PDF</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-neutral-200 mb-2">Resolution</label>
                    <select
                      value={resolution}
                      onChange={(e) => setResolution(e.target.value)}
                      className="w-full bg-neutral-800 text-white px-3 py-2 rounded-md border border-neutral-600 focus:border-blue-500 focus:outline-none"
                    >
                      <option value="1280x720">1280x720 (HD)</option>
                      <option value="1920x1080">1920x1080 (FHD)</option>
                      <option value="2560x1440">2560x1440 (QHD)</option>
                      <option value="3840x2160">3840x2160 (4K)</option>
                      <option value="7680x4320">7680x4320 (8K)</option>
                    </select>
                  </div>
                </div>

                {/* Quality and Background */}
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-sm font-medium text-neutral-200 mb-2">
                      Quality ({quality}%)
                    </label>
                    <input
                      type="range"
                      min="10"
                      max="100"
                      value={quality}
                      onChange={(e) => setQuality(Number(e.target.value))}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-neutral-200 mb-2">Background</label>
                    <select
                      value={backgroundColor}
                      onChange={(e) => setBackgroundColor(e.target.value as typeof backgroundColor)}
                      className="w-full bg-neutral-800 text-white px-3 py-2 rounded-md border border-neutral-600 focus:border-blue-500 focus:outline-none"
                    >
                      <option value="transparent">Transparent</option>
                      <option value="white">White</option>
                      <option value="black">Black</option>
                    </select>
                  </div>
                </div>

                {/* Options */}
                <div className="grid grid-cols-2 gap-4">
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
        </div>
      </div>

      {/* Right Panel - Job Queue and Downloads */}
      <div className="flex-1 flex flex-col bg-neutral-950">
        {/* Header */}
        <div className="p-6 border-b border-neutral-700">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-bold text-white">Render Jobs</h3>
            <div className="flex items-center gap-4 text-sm text-neutral-400">
              <span>{renderJobs.filter(j => j.status === 'completed').length} completed</span>
              <span>{renderJobs.filter(j => j.status === 'processing').length} processing</span>
              <span>{renderJobs.filter(j => j.status === 'pending').length} pending</span>
            </div>
          </div>
          
          {/* Job Management Actions */}
          {renderJobs.length > 0 && (
            <div className="flex items-center justify-between">
              <div className="text-sm text-neutral-500">
                Total: {renderJobs.length} jobs
              </div>
              <div className="flex items-center gap-2">
                {renderJobs.filter(j => j.status === 'completed').length > 0 && (
                  <button
                    onClick={handleClearCompletedJobs}
                    className="px-3 py-1.5 text-xs bg-amber-600/20 hover:bg-amber-600/30 text-amber-300 border border-amber-600/30 rounded-md transition-colors"
                  >
                    Clear Completed
                  </button>
                )}
                <button
                  onClick={handleClearAllJobs}
                  className="px-3 py-1.5 text-xs bg-red-600/20 hover:bg-red-600/30 text-red-300 border border-red-600/30 rounded-md transition-colors"
                >
                  Clear All
                </button>
              </div>
            </div>
          )}
        </div>
        
        {/* Jobs List */}
        <div className="flex-1 p-6 overflow-y-auto">
          {renderJobs.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="w-24 h-24 mb-6 bg-neutral-800 rounded-full flex items-center justify-center">
                <svg className="w-12 h-12 text-neutral-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </div>
              <h4 className="text-lg font-medium text-neutral-300 mb-2">No render jobs yet</h4>
              <p className="text-neutral-500 text-sm max-w-md">
                Create your first render job to generate high-quality images of your spider plot visualizations
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {renderJobs.map((job) => (
                <div key={job.id} className="bg-neutral-800 rounded-lg border border-neutral-700 overflow-hidden">
                  <div className="p-4">
                    {/* Job Header */}
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <h4 className="font-medium text-white">{job.name}</h4>
                          <div className={`flex items-center gap-1.5 ${getStatusColor(job.status)}`}>
                            {getStatusIcon(job.status)}
                            <span className="text-xs font-medium capitalize">{job.status}</span>
                          </div>
                          {job.status === 'completed' && job.fileSize && (
                            <span className="text-xs text-neutral-400 bg-neutral-700 px-2 py-0.5 rounded">
                              {job.fileSize}
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-3 text-xs text-neutral-400">
                          <span className="font-mono">{job.parameters.format.toUpperCase()}</span>
                          <span>{job.parameters.resolution}</span>
                          <span>{job.parameters.quality}% quality</span>
                          <span>{job.parameters.selectedGroups.length} groups</span>
                        </div>
                      </div>
                      
                      {/* Actions */}
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => handleDeleteJob(job.id)}
                          className="p-2 text-neutral-400 hover:text-red-400 hover:bg-red-600/10 rounded-md transition-colors"
                          title="Delete job"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        </button>
                      </div>
                    </div>
                    
                    {/* Progress Bar */}
                    {job.status === 'processing' && (
                      <div className="mb-4">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-xs text-neutral-400">Processing...</span>
                          <span className="text-xs text-neutral-400">{Math.round(job.progress)}%</span>
                        </div>
                        <div className="w-full bg-neutral-700 rounded-full h-2">
                          <div 
                            className="bg-blue-500 h-2 rounded-full transition-all duration-300" 
                            style={{ width: `${job.progress}%` }}
                          />
                        </div>
                      </div>
                    )}
                    
                    {/* Error Message */}
                    {job.status === 'failed' && job.error && (
                      <div className="mb-4 p-3 bg-red-600/10 border border-red-600/20 rounded-lg">
                        <p className="text-sm text-red-400">{job.error}</p>
                      </div>
                    )}
                    
                    {/* Completed Job Details */}
                    {job.status === 'completed' && (
                      <div className="border-t border-neutral-600 pt-4 mt-4">
                        <div className="flex items-start gap-4">
                          {/* Thumbnail */}
                          {job.thumbnailUrl && (
                            <div className="flex-shrink-0">
                              <img 
                                src={job.thumbnailUrl} 
                                alt={`Preview of ${job.name}`}
                                className="w-40 h-25 object-cover rounded border border-neutral-600 cursor-pointer hover:border-neutral-500 transition-colors"
                                onClick={() => {
                                  if (job.imageUrl || job.thumbnailUrl) {
                                    const modal = document.createElement('div');
                                    modal.className = 'fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4';
                                    modal.onclick = () => document.body.removeChild(modal);
                                    const img = document.createElement('img');
                                    img.src = job.imageUrl || job.thumbnailUrl!;
                                    img.className = 'max-w-full max-h-full rounded';
                                    modal.appendChild(img);
                                    document.body.appendChild(modal);
                                  }
                                }}
                              />
                              <div className="text-center mt-1">
                                <button
                                  onClick={() => {
                                    if (job.imageUrl || job.thumbnailUrl) {
                                      const modal = document.createElement('div');
                                      modal.className = 'fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4';
                                      modal.onclick = () => document.body.removeChild(modal);
                                      const img = document.createElement('img');
                                      img.src = job.imageUrl || job.thumbnailUrl!;
                                      img.className = 'max-w-full max-h-full rounded';
                                      modal.appendChild(img);
                                      document.body.appendChild(modal);
                                    }
                                  }}
                                  className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
                                >
                                  View Full Size
                                </button>
                              </div>
                            </div>
                          )}
                          
                          {/* Job Info and Download Options */}
                          <div className="flex-1">
                            {/* Completion Info */}
                            <div className="mb-4 p-3 bg-green-600/10 border border-green-600/20 rounded-lg">
                              <div className="flex items-center gap-2 mb-2">
                                <svg className="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                                <span className="text-sm text-green-300 font-medium">Render Complete</span>
                              </div>
                              <div className="text-xs text-green-200/80 space-y-1">
                                <div>Created: {job.createdAt.toLocaleString()}</div>
                                {job.completedAt && (
                                  <div>Completed: {job.completedAt.toLocaleString()}</div>
                                )}
                                <div>Groups: {job.parameters.selectedGroups.length} • Resolution: {job.parameters.resolution}</div>
                                <div>Background: {job.parameters.backgroundColor} • Labels: {job.parameters.includeLabels ? 'Yes' : 'No'}</div>
                              </div>
                            </div>

                            {/* Download Section */}
                            <div className="flex items-center justify-between mb-3">
                              <h5 className="text-sm font-medium text-white">Download Options</h5>
                              <button
                                onClick={() => handleDownload(job)}
                                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-sm font-medium transition-colors flex items-center gap-2"
                              >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                </svg>
                                Download
                              </button>
                            </div>
                            
                            <div className="grid grid-cols-3 gap-3 text-xs">
                              <div>
                                <label className="block text-neutral-400 mb-1">Format</label>
                                <select
                                  value={downloadOptions[job.id]?.format || job.parameters.format}
                                  onChange={(e) => updateDownloadOptions(job.id, { format: e.target.value as DownloadOptions['format'] })}
                                  className="w-full bg-neutral-700 text-white px-2 py-1 rounded border border-neutral-600 text-xs"
                                >
                                  <option value="png">PNG</option>
                                  <option value="jpg">JPEG</option>
                                  <option value="webp">WebP</option>
                                  <option value="svg">SVG</option>
                                  <option value="pdf">PDF</option>
                                </select>
                              </div>
                              <div>
                                <label className="block text-neutral-400 mb-1">Quality</label>
                                <select
                                  value={downloadOptions[job.id]?.quality || job.parameters.quality}
                                  onChange={(e) => updateDownloadOptions(job.id, { quality: Number(e.target.value) })}
                                  className="w-full bg-neutral-700 text-white px-2 py-1 rounded border border-neutral-600 text-xs"
                                >
                                  <option value={50}>50%</option>
                                  <option value={75}>75%</option>
                                  <option value={90}>90%</option>
                                  <option value={95}>95%</option>
                                  <option value={100}>100%</option>
                                </select>
                              </div>
                              <div>
                                <label className="block text-neutral-400 mb-1">Scale</label>
                                <select
                                  value={downloadOptions[job.id]?.scale || 1}
                                  onChange={(e) => updateDownloadOptions(job.id, { scale: Number(e.target.value) })}
                                  className="w-full bg-neutral-700 text-white px-2 py-1 rounded border border-neutral-600 text-xs"
                                >
                                  <option value={0.5}>0.5x</option>
                                  <option value={1}>1x</option>
                                  <option value={1.5}>1.5x</option>
                                  <option value={2}>2x</option>
                                </select>
                              </div>
                            </div>

                            {/* Quick Download Buttons */}
                            <div className="mt-3 flex gap-2">
                              <button
                                onClick={() => handleDownload(job, { format: 'png', quality: 100, scale: 1 })}
                                className="px-3 py-1.5 text-xs bg-neutral-700 hover:bg-neutral-600 text-white rounded border border-neutral-600 transition-colors"
                              >
                                PNG (High Quality)
                              </button>
                              <button
                                onClick={() => handleDownload(job, { format: 'jpg', quality: 90, scale: 1 })}
                                className="px-3 py-1.5 text-xs bg-neutral-700 hover:bg-neutral-600 text-white rounded border border-neutral-600 transition-colors"
                              >
                                JPEG (Compressed)
                              </button>
                              <button
                                onClick={() => handleDownload(job, { format: 'png', quality: 100, scale: 2 })}
                                className="px-3 py-1.5 text-xs bg-neutral-700 hover:bg-neutral-600 text-white rounded border border-neutral-600 transition-colors"
                              >
                                PNG (2x Scale)
                              </button>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}; 