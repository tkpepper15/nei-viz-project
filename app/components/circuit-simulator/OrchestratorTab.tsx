import React, { useState, useRef, useEffect, useCallback, JSX } from 'react';
import { ResnormGroup, ModelSnapshot } from './types';
import { SavedProfile } from './types/savedProfiles';
import { SpiderPlot } from './visualizations/SpiderPlot';
import { ResizableSplitPane } from './controls/ResizableSplitPane';

interface OrchestratorTabProps {
  resnormGroups: ResnormGroup[];
  savedProfiles?: SavedProfile[];
  generateProfileMeshData?: (profile: SavedProfile, sampleSize?: number) => ModelSnapshot[];
}


interface RenderJob {
  id: string;
  name: string;
  status: 'pending' | 'computing' | 'rendering' | 'completed' | 'failed';
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
    opacityFactor: number;
    visualizationMode: 'color' | 'opacity';
    opacityIntensity: number;
  };
  profileIds?: string[];
  meshData?: ModelSnapshot[];
  createdAt: Date;
  completedAt?: Date;
  fileSize?: string;
  error?: string;
}

// Temporarily commented out unused interface
/*
interface DownloadOptions {
  format: 'png' | 'svg' | 'pdf' | 'webp' | 'jpg';
  quality: number;
  scale: number;
}
*/

export const OrchestratorTab: React.FC<OrchestratorTabProps> = ({
  resnormGroups,
  savedProfiles = [],
  generateProfileMeshData: _generateProfileMeshData // eslint-disable-line @typescript-eslint/no-unused-vars
}): JSX.Element => {
  const [renderJobs, setRenderJobs] = useState<RenderJob[]>([]);
  const [hasLoadedJobs, setHasLoadedJobs] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // Profile computation state
  const [selectedProfiles, setSelectedProfiles] = useState<string[]>([]);
  
  // Simplified image generation settings - matching playground design
  const [format, setFormat] = useState<'png' | 'svg' | 'pdf' | 'webp' | 'jpg'>('png');
  const [resolution, setResolution] = useState('1920x1080');
  const [quality, setQuality] = useState(95);
  const [includeLabels, setIncludeLabels] = useState(true);
  const [chromaEnabled, setChromaEnabled] = useState(true);
  const [backgroundColor, setBackgroundColor] = useState<'transparent' | 'white' | 'black'>('transparent');
  
  // Opacity contrast controls from playground
  const [showDistribution, setShowDistribution] = useState(false);
  const [opacityIntensity, setOpacityIntensity] = useState<number>(1e-30);
  const [sliderValue, setSliderValue] = useState<number>(0);
  const [selectedOpacityGroups, setSelectedOpacityGroups] = useState<number[]>([0]);
  const [isClient, setIsClient] = useState(false);
  const [previewMeshData, setPreviewMeshData] = useState<ModelSnapshot[]>([]);
  const [opacityLevel] = useState<number>(0.7);

  // Helper function for file size formatting
  const formatFileSize = useCallback((bytes: number): string => { // eslint-disable-line @typescript-eslint/no-unused-vars
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }, []);
  
  // Generate sample mesh data for preview
  const generateSampleMeshData = useCallback(() => {
    const sampleData: ModelSnapshot[] = [];
    
    // Generate 135 sample points (similar to 3^5 grid)
    for (let i = 0; i < 135; i++) {
      const resnorm = Math.random() * 0.1 + 0.001; // Random resnorm between 0.001 and 0.101
      sampleData.push({
        id: `sample-${i}`,
        name: `Sample ${i}`,
        timestamp: Date.now(),
        parameters: {
          Rs: 10 + Math.random() * 9990,
          Ra: 10 + Math.random() * 9990,
          Rb: 10 + Math.random() * 9990,
          Ca: 0.1e-6 + Math.random() * 49.9e-6,
          Cb: 0.1e-6 + Math.random() * 49.9e-6,
          frequency_range: [0.1, 10000]
        },
        data: [], // Empty impedance data for preview
        resnorm,
        color: '#3B82F6',
        isVisible: true,
        opacity: 1.0
      });
    }
    
    return sampleData;
  }, []);
  
  // Opacity contrast functions from playground
  const getContextualRange = useCallback((): { min: number; max: number; dynamic: boolean } => {
    if (selectedOpacityGroups.length === 0 || !resnormGroups || resnormGroups.length === 0) {
      return { min: 1e-30, max: 10, dynamic: false };
    }
    
    const selectedResnorms = selectedOpacityGroups
      .flatMap(groupIndex => resnormGroups[groupIndex]?.items || [])
      .map(model => model.resnorm)
      .filter(r => r !== undefined && r > 0) as number[];
    
    if (selectedResnorms.length === 0) {
      return { min: 1e-30, max: 10, dynamic: false };
    }
    
    const sortedResnorms = [...selectedResnorms].sort((a, b) => a - b);
    const minResnorm = sortedResnorms[0];
    const maxResnorm = sortedResnorms[sortedResnorms.length - 1];
    const p90 = sortedResnorms[Math.floor(sortedResnorms.length * 0.9)];
    
    const contextualMin = Math.max(minResnorm * 0.1, 1e-15);
    const contextualMax = Math.min(maxResnorm * 2, p90 * 10);
    
    return { min: contextualMin, max: contextualMax, dynamic: true };
  }, [selectedOpacityGroups, resnormGroups]);
  
  const sliderToOpacityIntensity = useCallback((sliderVal: number): number => {
    const { min, max, dynamic } = getContextualRange();
    
    if (!dynamic) {
      const minLog = Math.log10(1e-30);
      const maxLog = Math.log10(10);
      const logValue = minLog + (sliderVal / 100) * (maxLog - minLog);
      return Math.pow(10, logValue);
    }
    
    const minLog = Math.log10(min);
    const maxLog = Math.log10(max);
    const logValue = minLog + (sliderVal / 100) * (maxLog - minLog);
    return Math.pow(10, logValue);
  }, [getContextualRange]);
  
  const opacityIntensityToSlider = useCallback((opacityValue: number): number => {
    const { min, max, dynamic } = getContextualRange();
    
    if (!dynamic) {
      const minLog = Math.log10(1e-30);
      const maxLog = Math.log10(10);
      const logValue = Math.log10(Math.max(1e-30, Math.min(10, opacityValue)));
      return ((logValue - minLog) / (maxLog - minLog)) * 100;
    }
    
    const minLog = Math.log10(min);
    const maxLog = Math.log10(max);
    const logValue = Math.log10(Math.max(min, Math.min(max, opacityValue)));
    return ((logValue - minLog) / (maxLog - minLog)) * 100;
  }, [getContextualRange]);
  
  const handleOpacityChange = useCallback((newSliderValue: string) => {
    const sliderVal = parseFloat(newSliderValue);
    setSliderValue(sliderVal);
    const mappedValue = sliderToOpacityIntensity(sliderVal);
    setOpacityIntensity(mappedValue);
  }, [sliderToOpacityIntensity]);
  
  // Initialize preview data and client state
  useEffect(() => {
    setIsClient(true);
    setPreviewMeshData(generateSampleMeshData());
    const initialSliderPos = opacityIntensityToSlider(1e-30);
    setSliderValue(initialSliderPos);
  }, [generateSampleMeshData, opacityIntensityToSlider]);
  
  // Generate resnorm groups from preview data
  const generatePreviewResnormGroups = useCallback((meshData: ModelSnapshot[]): ResnormGroup[] => {
    if (!meshData.length) return [];
    
    const resnorms = meshData.map(m => m.resnorm).filter(r => r !== undefined) as number[];
    if (resnorms.length === 0) return [];
    
    const sortedResnorms = [...resnorms].sort((a, b) => a - b);
    const q1 = sortedResnorms[Math.floor(sortedResnorms.length * 0.25)];
    const q2 = sortedResnorms[Math.floor(sortedResnorms.length * 0.5)];
    const q3 = sortedResnorms[Math.floor(sortedResnorms.length * 0.75)];
    const q4 = sortedResnorms[Math.floor(sortedResnorms.length * 0.9)];
    
    return [
      {
        range: [0, q1] as [number, number],
        color: '#059669',
        label: 'Excellent',
        description: 'Top 25% fits',
        items: meshData.filter(m => m.resnorm !== undefined && m.resnorm <= q1)
      },
      {
        range: [q1, q2] as [number, number],
        color: '#10B981',
        label: 'Good',
        description: '25-50% fits',
        items: meshData.filter(m => m.resnorm !== undefined && m.resnorm > q1 && m.resnorm <= q2)
      },
      {
        range: [q2, q3] as [number, number],
        color: '#F59E0B',
        label: 'Moderate',
        description: '50-75% fits',
        items: meshData.filter(m => m.resnorm !== undefined && m.resnorm > q2 && m.resnorm <= q3)
      },
      {
        range: [q3, q4] as [number, number],
        color: '#F97316',
        label: 'Acceptable',
        description: '75-90% fits',
        items: meshData.filter(m => m.resnorm !== undefined && m.resnorm > q3 && m.resnorm <= q4)
      },
      {
        range: [q4, Math.max(...resnorms)] as [number, number],
        color: '#DC2626',
        label: 'Poor',
        description: 'Bottom 10% fits',
        items: meshData.filter(m => m.resnorm !== undefined && m.resnorm > q4)
      }
    ].filter(g => g.items.length > 0);
  }, []);
  
  // Generate preview resnorm groups
  const previewResnormGroups = generatePreviewResnormGroups(previewMeshData);
  
  // Get visible models for preview
  const getVisiblePreviewModels = (): ModelSnapshot[] => {
    if (selectedOpacityGroups.length === 0) {
      return previewMeshData;
    }
    
    return previewResnormGroups
      .filter((_, index) => selectedOpacityGroups.includes(index))
      .flatMap(group => group.items);
  };
  
  // Calculate resnorm statistics for distribution
  const calculateResnormStats = (models: ModelSnapshot[]) => {
    const resnormValues = models.map(model => model.resnorm).filter(r => r !== undefined) as number[];
    if (resnormValues.length === 0) return null;
    
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    for (const value of resnormValues) {
      if (value < min) min = value;
      if (value > max) max = value;
      sum += value;
    }
    const sortedValues = [...resnormValues].sort((a, b) => a - b);
    return {
      min,
      max,
      mean: sum / resnormValues.length,
      median: sortedValues[Math.floor(resnormValues.length / 2)],
      count: resnormValues.length
    };
  };
  
  // Download functionality
  const downloadPreviewImage = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    try {
      const dataUrl = canvas.toDataURL(`image/${format}`, quality / 100);
      const link = document.createElement('a');
      link.href = dataUrl;
      link.download = `spider-plot-preview.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error('Download failed:', error);
    }
  }, [format, quality]);
  
  // Update selected groups when preview groups change
  useEffect(() => {
    if (previewResnormGroups.length > 0 && selectedOpacityGroups.length === 0) {
      setSelectedOpacityGroups([0]);
    }
  }, [previewResnormGroups.length, selectedOpacityGroups.length]);

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

  /*
  // Render actual spider plot exactly matching playground formatting
  const renderRealSpiderPlot = useCallback((
    canvas: HTMLCanvasElement, 
    parameters: RenderJob['parameters'], 
    meshData: ModelSnapshot[]
  ) => {
    const ctx = canvas.getContext('2d', { alpha: parameters.backgroundColor === 'transparent' });
    if (!ctx) throw new Error('Failed to get canvas context');

    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const maxRadius = Math.min(width, height) * 0.35;
    
    const params = ['Rs', 'Ra', 'Rb', 'Ca', 'Cb'] as const;
    const angleStep = (2 * Math.PI) / params.length;

    // Calculate dynamic parameter ranges from actual data (matching playground exactly)
    const paramRanges = params.reduce((acc, param) => {
      const values = meshData
        .filter(item => item?.parameters && typeof item.parameters[param] === 'number')
        .map(item => item.parameters[param]);
      
      if (values.length > 0) {
        let min = Infinity;
        let max = -Infinity;
        for (const value of values) {
          if (value < min) min = value;
          if (value > max) max = value;
        }
        acc[param] = { min, max };
      } else {
        // Use default ranges if no data
        const defaults = {
          Rs: { min: 10, max: 10000 },
          Ra: { min: 10, max: 10000 },
          Rb: { min: 10, max: 10000 },
          Ca: { min: 0.1e-6, max: 50e-6 },
          Cb: { min: 0.1e-6, max: 50e-6 }
        };
        acc[param] = defaults[param] || { min: 0, max: 1 };
      }
      return acc;
    }, {} as Record<string, { min: number; max: number }>);

    // Enable high-quality rendering
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    // Set background to match playground
    if (parameters.backgroundColor === 'white') {
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, width, height);
    } else if (parameters.backgroundColor === 'black') {
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, width, height);
    } else {
      // Default dark background like playground
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, width, height);
    }

    // Draw pentagon grid structure (matching playground PolarGrid with gridType="polygon")
    ctx.strokeStyle = '#4B5563';
    ctx.lineWidth = 0.5;
    ctx.globalAlpha = 0.3;

    // Draw 5 concentric pentagon grids (matching playground tickCount={5})
    for (let level = 1; level <= 5; level++) {
      const radius = (maxRadius * level) / 5;
      
      // Draw pentagon grid at this level
      ctx.beginPath();
      for (let i = 0; i <= params.length; i++) {
        const angle = i * angleStep - Math.PI / 2;
        const x = centerX + Math.cos(angle) * radius;
        const y = centerY + Math.sin(angle) * radius;
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }

    // Draw radial axes (matching playground)
    ctx.globalAlpha = 1;
    ctx.lineWidth = 0.5;
    for (let i = 0; i < params.length; i++) {
      const angle = i * angleStep - Math.PI / 2;
      const endX = centerX + Math.cos(angle) * maxRadius;
      const endY = centerY + Math.sin(angle) * maxRadius;
      
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(endX, endY);
      ctx.stroke();
    }



    // Parameter labels (matching playground PolarAngleAxis formatting exactly)
    if (parameters.includeLabels) {
      ctx.fillStyle = '#E5E7EB';
      ctx.font = '600 13px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      for (let i = 0; i < params.length; i++) {
        const param = params[i];
        const angle = i * angleStep - Math.PI / 2;
        const labelDistance = maxRadius + 30;
        const labelX = centerX + Math.cos(angle) * labelDistance;
        const labelY = centerY + Math.sin(angle) * labelDistance;
        
        // Enhanced axis labels with units and ranges (matching playground)
        const ranges = {
          Rs: '10Ω - 10kΩ',
          Ra: '10Ω - 10kΩ', 
          Rb: '10Ω - 10kΩ',
          Ca: '0.1 - 50µF',
          Cb: '0.1 - 50µF'
        };
        
        // Draw parameter name
        ctx.font = '600 13px Inter, sans-serif';
        ctx.fillStyle = '#E5E7EB';
        ctx.fillText(param, labelX, labelY - 8);
        
        // Draw range
        ctx.font = '400 10px Inter, sans-serif';
        ctx.fillStyle = '#E5E7EB';
        ctx.fillText(ranges[param], labelX, labelY + 8);
      }
    }

    // Render mesh data polygons with playground-style normalization
    if (meshData && meshData.length > 0) {
      meshData.forEach((model) => {
        if (!model.parameters) return;

        // Normalize parameters using LINEAR scaling (matching playground exactly)
        const normalizedValues = params.map(param => {
          const value = model.parameters[param];
          const range = paramRanges[param];
          
          // Linear normalization (NOT logarithmic) - matching playground
          const normalizedValue = (value - range.min) / Math.max(range.max - range.min, 1e-10);
          return Math.max(0, Math.min(1, normalizedValue));
        });

        // Calculate opacity (matching playground calculateLogOpacity exactly)
        let opacity = 0.7;
        if (parameters.visualizationMode === 'opacity' && model.resnorm !== undefined) {
          const allResnorms = meshData.map(m => m.resnorm).filter(r => r !== undefined) as number[];
          if (allResnorms.length > 1) {
            let minResnorm = Infinity;
            let maxResnorm = -Infinity;
            for (const r of allResnorms) {
              if (r < minResnorm) minResnorm = r;
              if (r > maxResnorm) maxResnorm = r;
            }
            
            const safeMin = Math.max(minResnorm, 1e-10);
            const safeMax = Math.max(maxResnorm, safeMin * 10);
            const logMin = Math.log10(safeMin);
            const logMax = Math.log10(safeMax);
            const logRange = Math.max(logMax - logMin, 1e-10);
            
            const safeResnorm = Math.max(model.resnorm, safeMin);
            const logResnorm = Math.log10(safeResnorm);
            const normalizedLog = (logResnorm - logMin) / logRange;
            
            // Invert so lower resnorm (better fit) = higher opacity
            const inverted = 1 - Math.max(0, Math.min(1, normalizedLog));
            
            // Apply intensity factor
            const intensified = Math.pow(inverted, 1 / parameters.opacityIntensity);
            
            // Map to opacity range 0.05 to 1.0
            opacity = Math.max(0.05, Math.min(1.0, 0.05 + intensified * 0.95));
          }
        }

        // Assign colors based on resnorm quartiles (matching playground color scheme)
        let color = model.color;
        
        if (!color && model.resnorm !== undefined) {
          // Calculate resnorm quartiles from all models to assign colors
          const allResnorms = meshData.map(m => m.resnorm).filter(r => r !== undefined) as number[];
          if (allResnorms.length > 0) {
            // Sort to find quartiles
            const sortedResnorms = [...allResnorms].sort((a, b) => a - b);
            const q1 = sortedResnorms[Math.floor(sortedResnorms.length * 0.25)];
            const q2 = sortedResnorms[Math.floor(sortedResnorms.length * 0.5)];
            const q3 = sortedResnorms[Math.floor(sortedResnorms.length * 0.75)];
            
            // Assign colors based on quartiles (better fit = cooler colors)
            if (model.resnorm <= q1) {
              color = '#059669'; // Emerald-600 (excellent fit - bottom quartile)
            } else if (model.resnorm <= q2) {
              color = '#10B981'; // Emerald-500 (good fit - second quartile)
            } else if (model.resnorm <= q3) {
              color = '#F59E0B'; // Amber-500 (moderate fit - third quartile)
            } else {
              color = '#DC2626'; // Red-600 (poor fit - top quartile)
            }
          } else {
            color = '#3B82F6'; // Default blue
          }
        } else if (!color) {
          color = '#3B82F6'; // Default blue
        }

        // Set stroke properties (matching playground Radar component)
        ctx.strokeStyle = color;
        ctx.lineWidth = model.id === 'dynamic-reference' ? 2 : 1;
        ctx.globalAlpha = opacity * parameters.opacityFactor;

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
      });
    }

    ctx.globalAlpha = 1;

    // Add title and metadata (matching playground style)
    if (parameters.includeLabels) {
      ctx.fillStyle = '#F9FAFB';
      ctx.font = 'bold 18px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Spider Plot - Cell Electrophysiology', centerX, 40);
      
      ctx.fillStyle = '#9CA3AF';
      ctx.font = '12px Inter, sans-serif';
      const infoText = `${meshData?.length || 0} Models • ${parameters.resolution} • ${parameters.visualizationMode} mode`;
      ctx.fillText(infoText, centerX, height - 30);
    }
  }, []);
  */

  /*
  // Generate SVG representation using real data
  const generateSVGFromCanvas = useCallback((canvas: HTMLCanvasElement, parameters: RenderJob['parameters']): string => {
    // For SVG, we'll create a simplified version since the real rendering is complex
    // This is a placeholder - in production you'd want to recreate the full logic in SVG
    const width = canvas.width;
    const height = canvas.height;
    
    let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`;
    
    // Background
    if (parameters.backgroundColor === 'white') {
      svg += `<rect width="${width}" height="${height}" fill="#ffffff"/>`;
    } else if (parameters.backgroundColor === 'black') {
      svg += `<rect width="${width}" height="${height}" fill="#000000"/>`;
    }
    
    // Add text indicating this is a simplified SVG version
    svg += `<text x="${width/2}" y="${height/2}" text-anchor="middle" font-family="Inter, sans-serif" font-size="16" fill="#888">SVG export - Canvas data converted</text>`;
    
    svg += '</svg>';
    return `data:image/svg+xml;base64,${btoa(svg)}`;
  }, []);
  */

  /*
  // Generate image from current visualization
  const generateImage = useCallback(async (job: RenderJob): Promise<{ imageUrl: string; thumbnailUrl: string; fileSize: string }> => {
    return new Promise((resolve, reject) => {
      let canvas: HTMLCanvasElement | null = null;
      let thumbCanvas: HTMLCanvasElement | null = null;
      
      try {
        // Validate job parameters
        if (!job.parameters.resolution || !job.parameters.format) {
          throw new Error('Invalid job parameters: missing resolution or format');
        }

        // Create high-quality canvas for rendering
        canvas = document.createElement('canvas');
        const [width, height] = job.parameters.resolution.split('x').map(Number);
        
        // Validate resolution
        if (isNaN(width) || isNaN(height) || width <= 0 || height <= 0) {
          throw new Error(`Invalid resolution: ${job.parameters.resolution}`);
        }
        
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d', { alpha: job.parameters.backgroundColor === 'transparent' });
        
        if (!ctx) {
          throw new Error('Failed to get canvas context');
        }

        // Use the real spider plot rendering with actual mesh data
        const meshData = job.meshData || [];
        renderRealSpiderPlot(canvas, job.parameters, meshData);

        // Convert to requested format with proper quality and error handling
        let imageUrl: string;
        let fileSize: string;

        try {
          switch (job.parameters.format) {
            case 'png':
              imageUrl = canvas.toDataURL('image/png');
              fileSize = formatFileSize(imageUrl.length * 0.75);
              break;
              
            case 'jpg':
              // Force white background for JPEG (no transparency support)
              if (job.parameters.backgroundColor === 'transparent') {
                const jpegCanvas = document.createElement('canvas');
                jpegCanvas.width = width;
                jpegCanvas.height = height;
                const jpegCtx = jpegCanvas.getContext('2d')!;
                jpegCtx.fillStyle = '#ffffff';
                jpegCtx.fillRect(0, 0, width, height);
                jpegCtx.drawImage(canvas, 0, 0);
                imageUrl = jpegCanvas.toDataURL('image/jpeg', job.parameters.quality / 100);
                jpegCanvas.remove();
              } else {
                imageUrl = canvas.toDataURL('image/jpeg', job.parameters.quality / 100);
              }
              fileSize = formatFileSize(imageUrl.length * 0.75 * (job.parameters.quality / 100));
              break;
              
            case 'webp':
              // WebP support detection
              const webpSupported = canvas.toDataURL('image/webp').indexOf('image/webp') === 5;
              if (webpSupported) {
                imageUrl = canvas.toDataURL('image/webp', job.parameters.quality / 100);
                fileSize = formatFileSize(imageUrl.length * 0.6 * (job.parameters.quality / 100));
              } else {
                console.warn('WebP not supported, falling back to PNG');
                imageUrl = canvas.toDataURL('image/png');
                fileSize = formatFileSize(imageUrl.length * 0.75);
              }
              break;
              
            case 'svg':
              // Generate SVG representation
              imageUrl = generateSVGFromCanvas(canvas, job.parameters);
              fileSize = formatFileSize(imageUrl.length * 0.5); // SVG is typically smaller
              break;
              
            case 'pdf':
              // For PDF, we'll use PNG as the base and indicate it needs PDF conversion
              console.warn('PDF export requires server-side conversion, using PNG as base');
              imageUrl = canvas.toDataURL('image/png');
              fileSize = formatFileSize(imageUrl.length * 0.75);
              break;
              
            default:
              throw new Error(`Unsupported format: ${job.parameters.format}`);
          }
        } catch (formatError) {
          console.error('Format conversion failed:', formatError);
          // Fallback to PNG
          imageUrl = canvas.toDataURL('image/png');
          fileSize = formatFileSize(imageUrl.length * 0.75);
        }

        // Create high-quality thumbnail
        thumbCanvas = document.createElement('canvas');
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

        // Clean up canvas resources
        canvas.remove();
        thumbCanvas.remove();

        resolve({ imageUrl, thumbnailUrl, fileSize });
        
      } catch (error) {
        console.error('Image generation failed:', error);
        // Clean up on error
        if (canvas) canvas.remove();
        if (thumbCanvas) thumbCanvas.remove();
        reject(new Error(`Failed to generate image: ${error instanceof Error ? error.message : 'Unknown error'}`));
      }
    });
  }, [formatFileSize, generateSVGFromCanvas, renderRealSpiderPlot]);
  */

  /*
  // Generate resnorm groups from mesh data for proper grouping
  const generateResnormGroupsFromMesh = useCallback((meshData: ModelSnapshot[]) => {
    if (!meshData.length) return [];
    
    const resnorms = meshData.map(m => m.resnorm).filter(r => r !== undefined) as number[];
    if (resnorms.length === 0) return [];
    
    // Sort to find quartiles
    const sortedResnorms = [...resnorms].sort((a, b) => a - b);
    const q1 = sortedResnorms[Math.floor(sortedResnorms.length * 0.25)];
    const q2 = sortedResnorms[Math.floor(sortedResnorms.length * 0.5)];
    const q3 = sortedResnorms[Math.floor(sortedResnorms.length * 0.75)];
    
    // Create groups with proper ranges
    const groups = [
      {
        range: [0, q1] as [number, number],
        color: '#059669',
        label: 'Excellent',
        description: 'Top 25% fits',
        items: meshData.filter(m => m.resnorm !== undefined && m.resnorm <= q1)
      },
      {
        range: [q1, q2] as [number, number],
        color: '#10B981',
        label: 'Good',
        description: '25-50% fits',
        items: meshData.filter(m => m.resnorm !== undefined && m.resnorm > q1 && m.resnorm <= q2)
      },
      {
        range: [q2, q3] as [number, number],
        color: '#F59E0B',
        label: 'Moderate',
        description: '50-75% fits',
        items: meshData.filter(m => m.resnorm !== undefined && m.resnorm > q2 && m.resnorm <= q3)
      },
      {
        range: [q3, Math.max(...resnorms)] as [number, number],
        color: '#DC2626',
        label: 'Poor',
        description: 'Bottom 25% fits',
        items: meshData.filter(m => m.resnorm !== undefined && m.resnorm > q3)
      }
    ];
    
    return groups.filter(g => g.items.length > 0);
  }, []);
  */

  /*
  // Generate mesh data for profiles when needed
  const generateMeshDataForProfile = useCallback((profileId: string) => {
    if (profileMeshData[profileId] || !generateProfileMeshData) return;
    
    const profile = savedProfiles.find(p => p.id === profileId);
    if (profile) {
      const meshData = generateProfileMeshData(profile, 500); // Generate 500 sample points
      setProfileMeshData(prev => ({
        ...prev,
        [profileId]: meshData
      }));
    }
  }, [profileMeshData, generateProfileMeshData, savedProfiles]);
  */

  /*
  // Function to render a job for a specific profile
  const renderProfileJob = useCallback(async (profileId: string) => {
    // Get mesh data for this profile
    let meshData = profileMeshData[profileId] || [];
    
    // Generate mesh data if not available
    if (meshData.length === 0) {
      generateMeshDataForProfile(profileId);
      meshData = profileMeshData[profileId] || [];
    }
    
    // Find the job for this profile
    const job = renderJobs.find(j => j.profileIds?.includes(profileId));
    if (!job) return;
    
    try {
      // Generate dynamic resnorm groups from mesh data
      const dynamicGroups = generateResnormGroupsFromMesh(meshData);
      
      // Update job with dynamic groups information
      const updatedJob = {
        ...job,
        meshData,
        parameters: {
          ...job.parameters,
          selectedGroups: dynamicGroups.map((_, i) => i) // Select all generated groups
        }
      };
      
      // Update status to rendering
      setRenderJobs(prev => prev.map(j => 
        j.id === job.id ? { ...j, status: 'rendering', progress: 0, meshData } : j
      ));
      
      // Simulate rendering progress
      for (let i = 10; i <= 90; i += 10) {
        setRenderJobs(prev => prev.map(j => 
          j.id === job.id ? { ...j, progress: i } : j
        ));
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      
      // Generate the actual image
      const { imageUrl, thumbnailUrl, fileSize } = await generateImage(updatedJob);
      
      setRenderJobs(prev => prev.map(j => 
        j.id === job.id ? { 
          ...j, 
          status: 'completed', 
          progress: 100,
          imageUrl,
          thumbnailUrl,
          fileSize,
          completedAt: new Date(),
          parameters: {
            ...j.parameters,
            selectedGroups: dynamicGroups.map((_, i) => i)
          }
        } : j
      ));
      
    } catch (error) {
      console.error(`Failed to render job for profile ${profileId}:`, error);
      setRenderJobs(prev => prev.map(j => 
        j.id === job.id ? { 
          ...j, 
          status: 'failed', 
          error: error instanceof Error ? error.message : 'Rendering failed'
        } : j
      ));
    }
  }, [profileMeshData, generateMeshDataForProfile, renderJobs, generateImage, generateResnormGroupsFromMesh]);
  */

  /*
  const handleDeleteJob = useCallback((jobId: string) => {
    setRenderJobs(prev => prev.filter(job => job.id !== jobId));
    setDownloadOptions(prev => {
      const newOptions = { ...prev };
      delete newOptions[jobId];
      return newOptions;
    });
  }, []);

  // Temporarily commented out unused functions
  /*
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
  */

  /*
  // Helper function to download data URL with improved error handling
  const downloadDataUrl = useCallback((dataUrl: string, filename: string) => {
    try {
      // Validate data URL
      if (!dataUrl || !dataUrl.startsWith('data:')) {
        throw new Error('Invalid data URL');
      }
      
      // Handle SVG downloads differently
      if (dataUrl.startsWith('data:image/svg+xml')) {
        // For SVG, decode and create blob
        const svgData = dataUrl.replace('data:image/svg+xml;base64,', '');
        const svgBlob = new Blob([atob(svgData)], { type: 'image/svg+xml' });
        const url = URL.createObjectURL(svgBlob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        // Clean up object URL
        setTimeout(() => URL.revokeObjectURL(url), 100);
      } else {
        // Standard data URL download
        const link = document.createElement('a');
        link.href = dataUrl;
        link.download = filename;
        
        // Add to DOM, click, and remove
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }
      
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
  */

  /*
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
          
          // Convert to desired format with proper error handling
          let dataUrl: string;
          try {
            switch (options.format) {
              case 'png':
                dataUrl = canvas.toDataURL('image/png');
                break;
              case 'jpg':
                // Force white background for JPEG
                const jpegCanvas = document.createElement('canvas');
                jpegCanvas.width = canvas.width;
                jpegCanvas.height = canvas.height;
                const jpegCtx = jpegCanvas.getContext('2d')!;
                jpegCtx.fillStyle = '#ffffff';
                jpegCtx.fillRect(0, 0, jpegCanvas.width, jpegCanvas.height);
                jpegCtx.drawImage(canvas, 0, 0);
                dataUrl = jpegCanvas.toDataURL('image/jpeg', options.quality / 100);
                jpegCanvas.remove();
                break;
              case 'webp':
                const webpSupported = canvas.toDataURL('image/webp').indexOf('image/webp') === 5;
                if (webpSupported) {
                  dataUrl = canvas.toDataURL('image/webp', options.quality / 100);
                } else {
                  console.warn('WebP not supported, falling back to PNG');
                  dataUrl = canvas.toDataURL('image/png');
                }
                break;
              default:
                dataUrl = canvas.toDataURL('image/png'); // Fallback
            }
          } catch (formatError) {
            console.error('Format conversion failed:', formatError);
            dataUrl = canvas.toDataURL('image/png'); // Safe fallback
          }
          
          // Download the processed image
          downloadDataUrl(dataUrl, `${job.name.replace(/[^a-z0-9]/gi, '_')}.${options.format}`);
          
          // Clean up canvas
          canvas.remove();
        };
        
        img.onerror = (error) => {
          console.error('Failed to load image for download:', error);
          alert('Failed to load image for download. Please try again.');
        };
        
        img.src = job.imageUrl;
      } else {
        // Download original image
        downloadDataUrl(job.imageUrl, `${job.name.replace(/[^a-z0-9]/gi, '_')}.${job.parameters.format}`);
      }
    } catch (error) {
      console.error('Download failed:', error);
    }
  }, [downloadOptions, downloadDataUrl]);
  */

  /*
  const updateDownloadOptions = (jobId: string, options: Partial<DownloadOptions>) => {
    setDownloadOptions(prev => ({
      ...prev,
      [jobId]: { ...prev[jobId], ...options } as DownloadOptions
    }));
  };

  const getStatusColor = (status: RenderJob['status']) => {
    switch (status) {
      case 'pending': return 'text-amber-400';
      case 'computing': return 'text-purple-400';
      case 'rendering': return 'text-blue-400';
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
      case 'computing':
        return (
          <svg className="w-4 h-4 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        );
      case 'rendering':
        return (
          <svg className="w-4 h-4 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
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
  */

  // Temporarily commented out unused functions
  /*
  // Add missing handleComputeAndRender function
  const handleComputeAndRender = useCallback(async () => {
    if (selectedProfiles.length === 0) return;
    
    // Create render jobs for each selected profile
    const newJobs: RenderJob[] = selectedProfiles.map(profileId => {
      const profile = savedProfiles.find(p => p.id === profileId);
      return {
        id: `job-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        name: profile?.name || 'Unnamed Profile',
        status: 'pending' as const,
        progress: 0,
        parameters: {
          format,
          resolution,
          quality,
          includeLabels,
          chromaEnabled,
          selectedGroups,
          backgroundColor,
          opacityFactor,
          visualizationMode,
          opacityIntensity
        },
        profileIds: [profileId],
        createdAt: new Date()
      };
    });
    
    // Add jobs to the queue
    setRenderJobs(prev => [...prev, ...newJobs]);
    
    // Process each job
    for (const job of newJobs) {
      if (job.profileIds && job.profileIds.length > 0) {
        await renderProfileJob(job.profileIds[0]);
      }
    }
  }, [selectedProfiles, savedProfiles, format, resolution, quality, includeLabels, chromaEnabled, selectedGroups, backgroundColor, opacityFactor, visualizationMode, opacityIntensity, renderProfileJob]);

  // Add missing deleteJob function
  const deleteJob = useCallback((jobId: string) => {
    handleDeleteJob(jobId);
  }, [handleDeleteJob]);

  // Add missing createRenderJob function
  const createRenderJob = useCallback(() => {
    if (savedProfiles.length === 0) return;
    handleComputeAndRender();
  }, [savedProfiles.length, handleComputeAndRender]);
  */

  // Function to create render jobs for selected profiles
  const handleCreateRenderJobs = useCallback(() => {
    if (selectedProfiles.length === 0) return;
    
    const newJobs: RenderJob[] = selectedProfiles.map(profileId => {
      const profile = savedProfiles.find(p => p.id === profileId);
      return {
        id: `job-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        name: profile?.name || 'Unnamed Profile',
        status: 'pending' as const,
        progress: 0,
        parameters: {
          format,
          resolution,
          quality,
          includeLabels,
          chromaEnabled,
          selectedGroups: selectedOpacityGroups.length > 0 ? selectedOpacityGroups : resnormGroups.map((_, i) => i),
          backgroundColor,
          opacityFactor: opacityLevel,
          visualizationMode: chromaEnabled ? 'color' : 'opacity',
          opacityIntensity
        },
        profileIds: [profileId],
        createdAt: new Date()
      };
    });
    
    setRenderJobs(prev => [...prev, ...newJobs]);
    
    // Simulate job processing with realistic timing
    newJobs.forEach(job => {
      // Start computing phase
      setTimeout(() => {
        setRenderJobs(prev => prev.map(j => 
          j.id === job.id ? { ...j, status: 'computing', progress: 25 } : j
        ));
      }, 500);
      
      // Move to rendering phase
      setTimeout(() => {
        setRenderJobs(prev => prev.map(j => 
          j.id === job.id ? { ...j, status: 'rendering', progress: 75 } : j
        ));
      }, 1500);
      
      // Complete the job
      setTimeout(() => {
        setRenderJobs(prev => prev.map(j => 
          j.id === job.id ? { ...j, status: 'completed', progress: 100, completedAt: new Date() } : j
        ));
      }, 2000 + Math.random() * 3000);
    });
  }, [selectedProfiles, savedProfiles, format, resolution, quality, includeLabels, chromaEnabled, resnormGroups, backgroundColor, selectedOpacityGroups, opacityLevel, opacityIntensity]);

  // Function to delete a render job
  const handleDeleteJob = useCallback((jobId: string) => {
    setRenderJobs(prev => prev.filter(job => job.id !== jobId));
  }, []);

  // Hidden canvas for image generation
  const canvasElement = <canvas ref={canvasRef} className="hidden" />;
  
  // Get current visible models and stats
  const visiblePreviewModels = getVisiblePreviewModels();
  const resnormStats = calculateResnormStats(visiblePreviewModels);
  
  // Preview spider plot content
  const previewContent = (
    <div className="bg-neutral-800 rounded-lg border border-neutral-700 h-full flex flex-col">
      {/* Preview Header */}
      <div className="flex-shrink-0 p-3 border-b border-neutral-600 bg-neutral-750">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
            <span className="text-sm font-medium text-neutral-200">Live Preview</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-neutral-400">
              {visiblePreviewModels.length} models
            </span>
            <button
              onClick={downloadPreviewImage}
              className="px-3 py-1.5 text-xs bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors flex items-center gap-1"
            >
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Download
            </button>
          </div>
        </div>
      </div>
      
      {/* Spider Plot */}
      <div className="flex-1 p-3 min-h-0">
        <div className="w-full h-full flex items-center justify-center">
          <div className="spider-visualization w-full h-full max-w-full max-h-full">
            <SpiderPlot
              meshItems={visiblePreviewModels.filter(model => model.parameters !== undefined)}
              opacityFactor={opacityLevel}
              maxPolygons={100000}
              visualizationMode={chromaEnabled ? 'color' : 'opacity'}
              opacityIntensity={opacityIntensity}
              gridSize={8}
              includeLabels={includeLabels}
              backgroundColor={backgroundColor}
            />
          </div>
        </div>
      </div>
    </div>
  );

  // Controls content - enhanced with opacity contrast tooling
  const controlsContent = (
    <div className="h-full bg-neutral-800 rounded-lg border border-neutral-700 flex flex-col">
      {/* Controls Header */}
      <div className="p-4 border-b border-neutral-600 flex-shrink-0">
        <div className="flex gap-4 items-center">
          {/* Left side - Chroma toggle and Create Jobs button */}
          <div className="flex items-center gap-3">
            {/* Chroma Toggle */}
            <button
              onClick={() => setChromaEnabled(!chromaEnabled)}
              className={`px-3 py-2 text-xs font-medium rounded-md transition-colors ${
                chromaEnabled
                  ? 'bg-primary text-white'
                  : 'bg-neutral-700 text-neutral-300 hover:bg-neutral-600'
              }`}
            >
              {chromaEnabled ? 'Chroma' : 'Mono'}
            </button>

            {/* Create Jobs Button */}
            <button
              onClick={handleCreateRenderJobs}
              disabled={selectedProfiles.length === 0}
              className={`px-4 py-2 text-xs font-medium rounded-md transition-colors ${
                selectedProfiles.length > 0
                  ? 'bg-blue-600 hover:bg-blue-700 text-white'
                  : 'bg-neutral-600 text-neutral-400 cursor-not-allowed'
              }`}
            >
              Create Jobs ({selectedProfiles.length})
            </button>
          </div>

          {/* Right side - Status info */}
          <div className="flex-1 flex justify-end items-center gap-4">
            <span className="text-xs text-neutral-400">
              {selectedProfiles.length} profiles selected
            </span>
            <span className="text-xs font-mono text-neutral-200 bg-neutral-700 px-2 py-1 rounded border">
              {format.toUpperCase()} • {resolution}
            </span>
          </div>
        </div>
      </div>

      {/* Scrollable Controls */}
      <div className="flex-1 min-h-0 overflow-y-auto custom-scrollbar">
        <div className="p-4 space-y-6">
          {/* Opacity Contrast Controls */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium text-neutral-200">Opacity Contrast</h3>
              <button
                onClick={() => setShowDistribution(!showDistribution)}
                className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
                  showDistribution
                    ? 'bg-amber-500 text-white hover:bg-amber-600'
                    : 'bg-neutral-700 text-neutral-300 hover:bg-neutral-600'
                }`}
              >
                {showDistribution ? 'Hide' : 'Show'} Distribution
              </button>
            </div>

            {/* Apply contrast to groups */}
            <div className="space-y-2">
              <label className="text-xs font-medium text-neutral-300">Apply contrast to:</label>
              
              {previewResnormGroups.length > 0 ? (
                <div className="flex gap-1.5 flex-wrap">
                  {/* All Groups button */}
                  <button
                    onClick={() => {
                      if (selectedOpacityGroups.length === previewResnormGroups.length) {
                        setSelectedOpacityGroups([]);
                      } else {
                        setSelectedOpacityGroups(previewResnormGroups.map((_, index) => index));
                      }
                    }}
                    className={`px-2.5 py-1.5 text-xs font-medium rounded transition-colors flex items-center gap-1.5 ${
                      selectedOpacityGroups.length === previewResnormGroups.length
                        ? 'bg-blue-600 text-white shadow-sm border border-blue-500'
                        : 'bg-neutral-700 text-neutral-300 hover:bg-neutral-600 hover:text-white border border-transparent'
                    }`}
                  >
                    <div className="w-2 h-2 rounded-full bg-gradient-to-r from-emerald-400 to-red-500 opacity-70" />
                    All{selectedOpacityGroups.length === previewResnormGroups.length ? ' ✓' : ''}
                  </button>
                  
                  {/* Individual group buttons */}
                  {previewResnormGroups.map((group, index) => {
                    const isSelected = selectedOpacityGroups.includes(index);
                    return (
                      <button
                        key={index}
                        onClick={() => {
                          if (isSelected) {
                            setSelectedOpacityGroups(selectedOpacityGroups.filter(i => i !== index));
                          } else {
                            setSelectedOpacityGroups([...selectedOpacityGroups, index]);
                          }
                        }}
                        className={`px-2.5 py-1.5 text-xs font-medium rounded transition-all flex items-center gap-1.5 ${
                          isSelected
                            ? 'bg-blue-600 text-white shadow-sm border border-blue-500'
                            : 'bg-neutral-700 text-neutral-300 hover:bg-neutral-600 hover:text-white border border-transparent'
                        }`}
                      >
                        <div 
                          className="w-2 h-2 rounded-full" 
                          style={{ backgroundColor: group.color }}
                        />
                        {group.label.split(' ')[0]}{isSelected ? ' ✓' : ''}
                        <span className="text-[10px] opacity-60 ml-0.5">({group.items.length})</span>
                      </button>
                    );
                  })}
                </div>
              ) : (
                <div className="text-xs text-neutral-500 italic">
                  Preview data groups will appear here
                </div>
              )}
            </div>
            
            {/* Opacity Slider */}
            <div className="space-y-2">
              {/* Dynamic range indicator */}
              {(() => {
                const { min, max, dynamic } = getContextualRange();
                return dynamic && (
                  <div className="flex items-center justify-between text-xs mb-2">
                    <span className="text-blue-400 font-medium flex items-center gap-1">
                      <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                      Dynamic Range Active
                    </span>
                    <span className="text-neutral-400 font-mono">
                      {min.toExponential(1)} - {max.toExponential(1)}
                    </span>
                  </div>
                );
              })()}

              <div className="relative">
                <div className="absolute top-1/2 left-0 right-0 h-2 bg-neutral-600 rounded-lg transform -translate-y-1/2"></div>
                <div 
                  className={`absolute top-1/2 left-0 h-2 rounded-lg transform -translate-y-1/2 transition-all duration-200 ${
                    getContextualRange().dynamic 
                      ? 'bg-gradient-to-r from-blue-500/60 to-blue-400' 
                      : 'bg-gradient-to-r from-primary/60 to-primary'
                  }`}
                  style={{ width: `${isClient ? sliderValue : 0}%` }}
                ></div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  value={isClient ? sliderValue : 0}
                  onChange={(e) => handleOpacityChange(e.target.value)}
                  className="relative w-full h-4 bg-transparent appearance-none cursor-pointer slider z-10"
                />
              </div>
              <div className="flex justify-between text-xs text-neutral-400">
                <span>High Contrast</span>
                <span>Low Contrast</span>
              </div>
              {getContextualRange().dynamic && (
                <div className="text-xs text-blue-300 text-center mt-1">
                  Contrast optimized for selected groups
                </div>
              )}
            </div>

            {/* Distribution Panel */}
            {showDistribution && resnormStats && (
              <div className="mt-4 space-y-3 p-3 bg-neutral-900 rounded-lg border border-neutral-600">
                <h4 className="text-sm font-medium text-neutral-200">
                  Distribution Analysis
                  {selectedOpacityGroups.length > 0 && previewResnormGroups && selectedOpacityGroups.length < previewResnormGroups.length && (
                    <span className="text-xs font-normal text-neutral-400 ml-2">
                      ({selectedOpacityGroups.map(i => previewResnormGroups[i]?.label.split(' ')[0]).join(', ')})
                    </span>
                  )}
                </h4>
                
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-neutral-700 p-2 rounded">
                    <div className="text-xs text-neutral-400">Count</div>
                    <div className="text-sm font-mono text-neutral-200">{resnormStats.count}</div>
                  </div>
                  <div className="bg-neutral-700 p-2 rounded">
                    <div className="text-xs text-neutral-400">Min</div>
                    <div className="text-sm font-mono text-neutral-200">{resnormStats.min.toExponential(2)}</div>
                  </div>
                  <div className="bg-neutral-700 p-2 rounded">
                    <div className="text-xs text-neutral-400">Max</div>
                    <div className="text-sm font-mono text-neutral-200">{resnormStats.max.toExponential(2)}</div>
                  </div>
                  <div className="bg-neutral-700 p-2 rounded">
                    <div className="text-xs text-neutral-400">Median</div>
                    <div className="text-sm font-mono text-neutral-200">{resnormStats.median.toExponential(2)}</div>
                  </div>
                </div>
              </div>
            )}
          </div>
          
          {/* Profile Selection */}
          {savedProfiles.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-medium text-neutral-200">Select Profiles</h3>
                <div className="text-xs text-neutral-400">
                  {selectedProfiles.length}/{savedProfiles.length}
                </div>
              </div>

              <div className="space-y-2">
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

                <div className="space-y-2 max-h-32 overflow-y-auto">
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
                      className={`cursor-pointer p-2 rounded-md border transition-all duration-200 ${
                        selectedProfiles.includes(profile.id)
                          ? 'bg-blue-600/20 border-blue-500'
                          : 'bg-neutral-800 border-neutral-600 hover:bg-neutral-700'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <h4 className="text-sm font-medium text-white truncate">{profile.name}</h4>
                        <div className="flex items-center gap-1">
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
            </div>
          )}

          {/* Render Settings */}
          <div className="space-y-3">
            <h3 className="text-sm font-medium text-neutral-200">Render Settings</h3>
            
            {/* Format and Resolution */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-medium text-neutral-300 mb-1">Format</label>
                <select
                  value={format}
                  onChange={(e) => setFormat(e.target.value as typeof format)}
                  className="w-full bg-neutral-700 text-white px-2 py-1.5 rounded-md border border-neutral-600 focus:border-blue-500 focus:outline-none text-xs"
                >
                  <option value="png">PNG</option>
                  <option value="jpg">JPEG</option>
                  <option value="webp">WebP</option>
                  <option value="svg">SVG</option>
                  <option value="pdf">PDF</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-medium text-neutral-300 mb-1">Resolution</label>
                <select
                  value={resolution}
                  onChange={(e) => setResolution(e.target.value)}
                  className="w-full bg-neutral-700 text-white px-2 py-1.5 rounded-md border border-neutral-600 focus:border-blue-500 focus:outline-none text-xs"
                >
                  <option value="1920x1080">1920x1080</option>
                  <option value="2560x1440">2560x1440</option>
                  <option value="3840x2160">3840x2160</option>
                  <option value="7680x4320">7680x4320</option>
                </select>
              </div>
            </div>

            {/* Quality and Background */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-medium text-neutral-300 mb-1">
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
                <label className="block text-xs font-medium text-neutral-300 mb-1">Background</label>
                <select
                  value={backgroundColor}
                  onChange={(e) => setBackgroundColor(e.target.value as typeof backgroundColor)}
                  className="w-full bg-neutral-700 text-white px-2 py-1.5 rounded-md border border-neutral-600 focus:border-blue-500 focus:outline-none text-xs"
                >
                  <option value="transparent">Transparent</option>
                  <option value="white">White</option>
                  <option value="black">Black</option>
                </select>
              </div>
            </div>

            {/* Include Labels Toggle */}
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={includeLabels}
                onChange={(e) => setIncludeLabels(e.target.checked)}
                className="rounded border-neutral-600"
              />
              <span className="text-xs text-neutral-300">Include Labels</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  // Status icon helpers
  const getStatusIcon = (status: RenderJob['status']) => {
    switch (status) {
      case 'pending':
        return <div className="w-2 h-2 bg-amber-400 rounded-full animate-pulse" />;
      case 'computing':
      case 'rendering':
        return <div className="w-2 h-2 bg-blue-400 rounded-full animate-spin" />;
      case 'completed':
        return <div className="w-2 h-2 bg-green-400 rounded-full" />;
      case 'failed':
        return <div className="w-2 h-2 bg-red-400 rounded-full" />;
      default:
        return <div className="w-2 h-2 bg-neutral-400 rounded-full" />;
    }
  };

  const getStatusColor = (status: RenderJob['status']) => {
    switch (status) {
      case 'pending': return 'text-amber-400 bg-amber-400/10';
      case 'computing': 
      case 'rendering': return 'text-blue-400 bg-blue-400/10';
      case 'completed': return 'text-green-400 bg-green-400/10';
      case 'failed': return 'text-red-400 bg-red-400/10';
      default: return 'text-neutral-400 bg-neutral-400/10';
    }
  };

  // Bottom Panel - Job Queue and Downloads
  const rightPanelContent = (
    <div className="flex-1 flex flex-col bg-neutral-900 border-t border-neutral-700">
      {/* Header */}
      <div className="p-4 border-b border-neutral-700 flex-shrink-0">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-bold text-white">Render Jobs</h3>
            <p className="text-xs text-neutral-400">
              {renderJobs.length} total • {renderJobs.filter(j => j.status === 'completed').length} completed
            </p>
          </div>
          {renderJobs.length > 0 && (
            <button
              onClick={() => setRenderJobs([])}
              className="px-3 py-1.5 text-xs bg-red-600/20 hover:bg-red-600/30 text-red-300 border border-red-600/30 rounded-md transition-colors"
            >
              Clear All
            </button>
          )}
        </div>
      </div>
      
      {/* Jobs List */}
      <div className="flex-1 p-4 overflow-y-auto">
        {renderJobs.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-16 h-16 mb-4 bg-neutral-800 rounded-full flex items-center justify-center">
              <svg className="w-8 h-8 text-neutral-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
            <h4 className="text-sm font-medium text-neutral-300 mb-2">No render jobs yet</h4>
            <p className="text-xs text-neutral-500 max-w-xs">
              Select profiles and click &quot;Create Jobs&quot; to generate high-quality visualizations
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {renderJobs.map((job) => (
              <div key={job.id} className="bg-neutral-800 rounded-lg border border-neutral-700 p-4">
                {/* Job Header */}
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-3">
                    {getStatusIcon(job.status)}
                    <div>
                      <h4 className="text-sm font-medium text-white truncate">{job.name}</h4>
                      <div className="flex items-center gap-2 text-xs text-neutral-400">
                        <span className="font-mono">{job.parameters.format.toUpperCase()}</span>
                        <span>•</span>
                        <span>{job.parameters.resolution}</span>
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={() => handleDeleteJob(job.id)}
                    className="p-1.5 text-neutral-400 hover:text-red-400 hover:bg-red-600/10 rounded-md transition-colors"
                    title="Delete job"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                </div>
                
                {/* Status Badge */}
                <div className="mb-3">
                  <span className={`inline-block px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(job.status)}`}>
                    {job.status}
                  </span>
                </div>
                
                {/* Progress Bar for active jobs */}
                {(job.status === 'computing' || job.status === 'rendering') && (
                  <div className="mb-3">
                    <div className="w-full bg-neutral-700 rounded-full h-1.5">
                      <div 
                        className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                        style={{ width: `${job.progress}%` }}
                      />
                    </div>
                  </div>
                )}
                
                {/* Job Details */}
                <div className="text-xs text-neutral-500 mb-3">
                  <div>Created: {job.createdAt.toLocaleTimeString()}</div>
                  {job.completedAt && (
                    <div>Completed: {job.completedAt.toLocaleTimeString()}</div>
                  )}
                </div>
                
                {/* Download Button for completed jobs */}
                {job.status === 'completed' && (
                  <button 
                    onClick={() => {
                      // Create a temporary download link
                      const canvas = document.createElement('canvas');
                      canvas.width = 800;
                      canvas.height = 600;
                      const ctx = canvas.getContext('2d');
                      if (ctx) {
                        ctx.fillStyle = '#1f2937';
                        ctx.fillRect(0, 0, 800, 600);
                        ctx.fillStyle = '#ffffff';
                        ctx.font = '20px Arial';
                        ctx.textAlign = 'center';
                        ctx.fillText('Spider Plot - ' + job.name, 400, 300);
                        
                        canvas.toBlob((blob) => {
                          if (blob) {
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = `${job.name.replace(/[^a-z0-9]/gi, '_')}.${job.parameters.format}`;
                            a.click();
                            URL.revokeObjectURL(url);
                          }
                        }, `image/${job.parameters.format}`);
                      }
                    }}
                    className="w-full px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-xs font-medium transition-colors flex items-center justify-center gap-2"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    Download {job.parameters.format.toUpperCase()}
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );

  // Main return: Three-panel layout with preview
  return (
    <div className="h-full">
      {canvasElement}
      <ResizableSplitPane 
        defaultSplitHeight={45}
        minTopHeight={320}
        minBottomHeight={280}
      >
        <div key="preview-and-controls" className="h-full flex">
          {/* Left Controls Panel */}
          <div className="w-[380px] flex-shrink-0 flex flex-col bg-neutral-900 border-r border-neutral-800">
            {controlsContent}
          </div>
          
          {/* Right Preview Panel */}
          <div className="flex-1 flex flex-col bg-neutral-950 p-4">
            {previewContent}
          </div>
        </div>
        
        {/* Bottom Jobs Panel */}
        <div key="jobs" className="h-full">
          {rightPanelContent}
        </div>
      </ResizableSplitPane>
    </div>
  );
};