// Orchestrator Worker Pool Manager with Staggered Job Queue
// Manages multiple orchestrator workers for parallel spider plot rendering

// Import types
import { ModelSnapshot } from '../types';
import { CircuitParameters } from '../types/parameters';

// Worker job interface
interface RenderJobParameters {
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
  baseOpacity?: number;
  maxOpacity?: number;
  opacityStrategy?: string;
  color?: string;
  strokeWidth?: number;
  strokeColor?: string;
  showGroundTruth?: boolean;
  groundTruthParams?: CircuitParameters;
}

interface WorkerResult {
  success: boolean;
  imageUrl?: string;
  error?: string;
  renderTime?: number;
  memoryUsage?: MemoryUsage;
  modelsRendered?: number;
}

interface MemoryUsage {
  used: number;
  total: number;
  limit: number;
  percentage: number;
}

interface WorkerJob {
  id: string;
  meshData: ModelSnapshot[];
  parameters: RenderJobParameters;
  priority: 'low' | 'medium' | 'high';
  timestamp: number;
  onProgress?: (progress: number, intermediateImageUrl?: string) => void;
  onComplete?: (result: WorkerResult) => void;
  onError?: (error: string) => void;
}

interface WorkerInstance {
  worker: Worker;
  id: number;
  busy: boolean;
  currentJobId?: string;
  totalJobsCompleted: number;
  totalRenderTime: number;
  lastMemoryUsage?: MemoryUsage;
}

interface QueueMetrics {
  totalJobs: number;
  completedJobs: number;
  failedJobs: number;
  averageRenderTime: number;
  queueLength: number;
  activeWorkers: number;
}

export class OrchestratorWorkerManager {
  private workers: WorkerInstance[] = [];
  private jobQueue: WorkerJob[] = [];
  private completedJobs: Map<string, WorkerResult> = new Map();
  private failedJobs: Map<string, string> = new Map();
  private maxWorkers: number;
  private isInitialized = false;
  private jobCallbacks: Map<string, {
    onProgress?: (progress: number, intermediateImageUrl?: string) => void;
    onComplete?: (result: WorkerResult) => void;
    onError?: (error: string) => void;
  }> = new Map();
  private metrics: QueueMetrics = {
    totalJobs: 0,
    completedJobs: 0,
    failedJobs: 0,
    averageRenderTime: 0,
    queueLength: 0,
    activeWorkers: 0
  };

  constructor(maxWorkers?: number) {
    // Use 12 workers for maximum parallel processing performance
    // Only check hardware in browser environment
    if (typeof window !== 'undefined' && typeof navigator !== 'undefined') {
      this.maxWorkers = maxWorkers || 12;
    } else {
      this.maxWorkers = 12; // Default for SSR
    }
    
    console.log(`🏗️ [OrchestratorWorkerManager] Initialized with ${this.maxWorkers} workers`);
  }

  // Initialize worker pool
  async initialize(): Promise<void> {
    if (this.isInitialized) return;
    
    // Check if we're in browser environment
    if (typeof window === 'undefined' || typeof Worker === 'undefined') {
      console.log('🚫 [OrchestratorWorkerManager] Workers not available (SSR environment)');
      return;
    }

    console.log(`🚀 [OrchestratorWorkerManager] Starting ${this.maxWorkers} orchestrator workers...`);

    for (let i = 0; i < this.maxWorkers; i++) {
      try {
        const worker = new Worker('/orchestrator-worker.js');
        const workerInstance: WorkerInstance = {
          worker,
          id: i,
          busy: false,
          totalJobsCompleted: 0,
          totalRenderTime: 0
        };

        // Set up worker message handling
        worker.onmessage = (e) => this.handleWorkerMessage(workerInstance, e);
        worker.onerror = (error) => this.handleWorkerError(workerInstance, error);

        // Initialize worker
        worker.postMessage({ type: 'init', data: { workerIndex: i } });

        this.workers.push(workerInstance);
      } catch (error) {
        console.error(`❌ [OrchestratorWorkerManager] Failed to create worker ${i}:`, error);
      }
    }

    // Wait for all workers to be ready
    await this.waitForWorkersReady();
    this.isInitialized = true;
    this.startQueueProcessor();

    console.log(`✅ [OrchestratorWorkerManager] All ${this.workers.length} workers ready`);
  }

  // Wait for all workers to report ready
  private async waitForWorkersReady(): Promise<void> {
    return new Promise((resolve) => {
      let readyCount = 0;
      const checkReady = () => {
        readyCount++;
        if (readyCount >= this.workers.length) {
          resolve();
        }
      };

      this.workers.forEach(workerInstance => {
        const originalHandler = workerInstance.worker.onmessage;
        workerInstance.worker.onmessage = (e) => {
          if (e.data.type === 'ready') {
            checkReady();
          }
          if (originalHandler) originalHandler.call(workerInstance.worker, e);
        };
      });
    });
  }

  // Add job to queue with priority and staggered execution
  addJob(
    id: string,
    meshData: ModelSnapshot[],
    parameters: RenderJobParameters,
    priority: 'low' | 'medium' | 'high' = 'medium',
    callbacks?: {
      onProgress?: (progress: number, intermediateImageUrl?: string) => void;
      onComplete?: (result: WorkerResult) => void;
      onError?: (error: string) => void;
    }
  ): void {
    const job: WorkerJob = {
      id,
      meshData,
      parameters,
      priority,
      timestamp: Date.now(),
      ...callbacks
    };

    // Insert job in priority order
    this.insertJobByPriority(job);
    this.metrics.totalJobs++;
    this.updateQueueMetrics();

    console.log(`📋 [OrchestratorWorkerManager] Job ${id} added to queue (priority: ${priority}, models: ${meshData.length}, queue: ${this.jobQueue.length})`);
    
    // Immediately try to process to ensure worker distribution
    setTimeout(() => this.processQueue(), 5);
  }

  // Insert job maintaining priority order
  private insertJobByPriority(job: WorkerJob): void {
    const priorityOrder = { high: 3, medium: 2, low: 1 };
    const jobPriority = priorityOrder[job.priority];

    let insertIndex = this.jobQueue.length;
    for (let i = 0; i < this.jobQueue.length; i++) {
      const queuedJobPriority = priorityOrder[this.jobQueue[i].priority];
      if (jobPriority > queuedJobPriority) {
        insertIndex = i;
        break;
      }
    }

    this.jobQueue.splice(insertIndex, 0, job);
  }

  // Process job queue with true parallel execution
  private async processQueue(): Promise<void> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    // Find available workers and assign ALL available jobs
    const availableWorkers = this.workers.filter(w => !w.busy);
    const jobsToProcess = Math.min(availableWorkers.length, this.jobQueue.length);

    if (jobsToProcess === 0) {
      if (this.jobQueue.length > 0) {
        console.log(`⏳ [OrchestratorWorkerManager] ${this.jobQueue.length} jobs queued, but all ${this.workers.length} workers busy`);
      }
      return;
    }

    const busyWorkerIds = this.workers.filter(w => w.busy).map(w => w.id);
    const availableWorkerIds = availableWorkers.map(w => w.id);
    console.log(`🔄 [OrchestratorWorkerManager] Processing ${jobsToProcess} jobs | Available workers: [${availableWorkerIds.join(', ')}] | Busy workers: [${busyWorkerIds.join(', ')}] | Queue: ${this.jobQueue.length}`);

    // Assign jobs to all available workers simultaneously
    const assignmentPromises = [];
    for (let i = 0; i < jobsToProcess; i++) {
      const worker = availableWorkers[i];
      const job = this.jobQueue.shift()!;

      console.log(`📋 [OrchestratorWorkerManager] Assigning job ${job.id} to worker ${worker.id}`);
      assignmentPromises.push(this.assignJobToWorker(worker, job));
    }

    // Execute all assignments in parallel
    await Promise.all(assignmentPromises);
    this.updateQueueMetrics();
    
    // Continue processing if there are more jobs
    if (this.jobQueue.length > 0) {
      setTimeout(() => this.processQueue(), 25);
    }
  }

  // Assign job to specific worker with promise-based tracking
  private async assignJobToWorker(worker: WorkerInstance, job: WorkerJob): Promise<void> {
    worker.busy = true;
    worker.currentJobId = job.id;

    console.log(`🎨 [OrchestratorWorkerManager] Assigning job ${job.id} (${job.meshData.length} models) to worker ${worker.id}`);

    // Store job callbacks before sending to worker
    this.storeJobCallbacks(job);

    worker.worker.postMessage({
      type: 'renderJob',
      data: {
        id: job.id,
        meshData: job.meshData,
        parameters: job.parameters
      }
    });

    // Brief stagger to prevent message queue overload
    await new Promise(resolve => setTimeout(resolve, 5));
  }

  // Store job callbacks in worker instance for message handling
  private storeJobCallbacks(job: WorkerJob): void {
    this.jobCallbacks.set(job.id, {
      onProgress: job.onProgress,
      onComplete: job.onComplete,
      onError: job.onError
    });
  }

  // Handle worker messages
  private handleWorkerMessage(workerInstance: WorkerInstance, e: MessageEvent): void {
    const { type, jobId, result, error, progress, intermediateImageUrl } = e.data;

    switch (type) {
      case 'progress':
        this.handleJobProgress(jobId, progress, intermediateImageUrl);
        break;

      case 'jobComplete':
        this.handleJobComplete(workerInstance, jobId, result);
        break;

      case 'jobError':
        this.handleJobError(workerInstance, jobId, error);
        break;

      case 'metrics':
        workerInstance.lastMemoryUsage = e.data.memoryUsage;
        break;

      case 'ready':
        console.log(`✅ [OrchestratorWorkerManager] Worker ${workerInstance.id} ready`);
        break;

      default:
        console.warn(`Unknown message type from worker ${workerInstance.id}:`, type);
    }
  }

  // Handle job progress updates
  private handleJobProgress(jobId: string, progress: number, intermediateImageUrl?: string): void {
    const callbacks = this.jobCallbacks.get(jobId);
    if (callbacks?.onProgress) {
      callbacks.onProgress(progress, intermediateImageUrl);
    }
  }

  // Handle job completion
  private handleJobComplete(workerInstance: WorkerInstance, jobId: string, result: WorkerResult): void {
    console.log(`✅ [OrchestratorWorkerManager] Job ${jobId} completed by worker ${workerInstance.id} in ${result.renderTime?.toFixed(1)}ms`);

    // Update worker stats
    const wasBusy = workerInstance.busy;
    workerInstance.busy = false;
    workerInstance.currentJobId = undefined;
    
    console.log(`🔄 [OrchestratorWorkerManager] Worker ${workerInstance.id} status: ${wasBusy ? 'busy' : 'available'} → available | Queue length: ${this.jobQueue.length}`);
    workerInstance.totalJobsCompleted++;
    if (result.renderTime) {
      workerInstance.totalRenderTime += result.renderTime;
    }

    // Store result and call completion callback
    this.completedJobs.set(jobId, result);
    const callbacks = this.jobCallbacks.get(jobId);
    if (callbacks?.onComplete) {
      callbacks.onComplete(result);
    }

    // Clean up callbacks
    this.jobCallbacks.delete(jobId);

    // Update metrics
    this.metrics.completedJobs++;
    this.updateQueueMetrics();

    // Process next jobs in queue
    this.processQueue();
  }

  // Handle job errors
  private handleJobError(workerInstance: WorkerInstance, jobId: string, error: string): void {
    console.error(`❌ [OrchestratorWorkerManager] Job ${jobId} failed on worker ${workerInstance.id}:`, error);

    // Update worker state
    workerInstance.busy = false;
    workerInstance.currentJobId = undefined;

    // Store error and call error callback
    this.failedJobs.set(jobId, error);
    const callbacks = this.jobCallbacks.get(jobId);
    if (callbacks?.onError) {
      callbacks.onError(error);
    }

    // Clean up callbacks
    this.jobCallbacks.delete(jobId);

    // Update metrics
    this.metrics.failedJobs++;
    this.updateQueueMetrics();

    // Process next jobs in queue
    this.processQueue();
  }

  // Handle worker errors
  private handleWorkerError(workerInstance: WorkerInstance, error: ErrorEvent): void {
    console.error(`💥 [OrchestratorWorkerManager] Worker ${workerInstance.id} error:`, error);
    
    // Mark worker as not busy and clear current job
    if (workerInstance.currentJobId) {
      this.handleJobError(workerInstance, workerInstance.currentJobId, `Worker error: ${error.message}`);
    }
  }

  // Start queue processor for continuous monitoring with adaptive intervals
  private startQueueProcessor(): void {
    let processingInterval: NodeJS.Timeout | null = null;
    
    const adaptiveProcessor = () => {
      const queueLength = this.jobQueue.length;
      const activeWorkers = this.workers.filter(w => w.busy).length;
      
      if (queueLength > 0) {
        this.processQueue();
        
        // Adaptive intervals based on queue pressure
        let nextInterval = 100; // Default
        if (queueLength > 10) nextInterval = 50; // High queue pressure
        if (queueLength > 20) nextInterval = 25; // Very high pressure
        if (activeWorkers === 0 && queueLength > 0) nextInterval = 10; // No workers active but jobs waiting
        
        if (processingInterval) clearTimeout(processingInterval);
        processingInterval = setTimeout(adaptiveProcessor, nextInterval);
      } else {
        // No jobs in queue, check less frequently
        if (processingInterval) clearTimeout(processingInterval);
        processingInterval = setTimeout(adaptiveProcessor, 250);
      }
    };
    
    // Start the adaptive processor
    adaptiveProcessor();
    
    // Additional background monitoring for stuck workers
    setInterval(() => {
      const stuckWorkers = this.workers.filter(w => w.busy && !w.currentJobId);
      if (stuckWorkers.length > 0) {
        console.warn(`⚠️ [OrchestratorWorkerManager] Found ${stuckWorkers.length} potentially stuck workers, resetting...`);
        stuckWorkers.forEach(worker => {
          worker.busy = false;
          worker.currentJobId = undefined;
        });
        this.processQueue(); // Try to assign jobs to reset workers
      }
    }, 5000); // Check for stuck workers every 5 seconds
  }

  // Update queue metrics
  private updateQueueMetrics(): void {
    this.metrics.queueLength = this.jobQueue.length;
    this.metrics.activeWorkers = this.workers.filter(w => w.busy).length;
    
    // Calculate average render time
    const totalRenderTime = this.workers.reduce((sum, w) => sum + w.totalRenderTime, 0);
    const totalJobs = this.workers.reduce((sum, w) => sum + w.totalJobsCompleted, 0);
    this.metrics.averageRenderTime = totalJobs > 0 ? totalRenderTime / totalJobs : 0;
  }

  // Cancel specific job
  cancelJob(jobId: string): boolean {
    // Remove from queue if not started
    const queueIndex = this.jobQueue.findIndex(job => job.id === jobId);
    if (queueIndex >= 0) {
      this.jobQueue.splice(queueIndex, 1);
      this.updateQueueMetrics();
      console.log(`🛑 [OrchestratorWorkerManager] Job ${jobId} cancelled (removed from queue)`);
      return true;
    }

    // Cancel if currently running
    const runningWorker = this.workers.find(w => w.currentJobId === jobId);
    if (runningWorker) {
      runningWorker.worker.postMessage({ type: 'cancel' });
      console.log(`🛑 [OrchestratorWorkerManager] Job ${jobId} cancellation requested on worker ${runningWorker.id}`);
      return true;
    }

    return false;
  }

  // Cancel all jobs
  cancelAllJobs(): void {
    // Clear queue
    this.jobQueue = [];
    
    // Cancel running jobs
    this.workers.forEach(worker => {
      if (worker.busy) {
        worker.worker.postMessage({ type: 'cancel' });
      }
    });

    console.log(`🛑 [OrchestratorWorkerManager] All jobs cancelled`);
    this.updateQueueMetrics();
  }

  // Get current metrics
  getMetrics(): QueueMetrics {
    this.updateQueueMetrics();
    return { ...this.metrics };
  }

  // Get worker statistics
  getWorkerStats(): WorkerStats[] {
    return this.workers.map(w => ({
      id: w.id,
      busy: w.busy,
      currentJobId: w.currentJobId,
      totalJobsCompleted: w.totalJobsCompleted,
      averageRenderTime: w.totalJobsCompleted > 0 ? w.totalRenderTime / w.totalJobsCompleted : 0,
      lastMemoryUsage: w.lastMemoryUsage
    }));
  }

  // Cleanup and terminate all workers
  terminate(): void {
    console.log(`🧹 [OrchestratorWorkerManager] Terminating ${this.workers.length} workers...`);
    
    this.workers.forEach(worker => {
      worker.worker.terminate();
    });
    
    this.workers = [];
    this.jobQueue = [];
    this.completedJobs.clear();
    this.failedJobs.clear();
    this.jobCallbacks.clear();
    this.isInitialized = false;
  }
}

interface WorkerStats {
  id: number;
  busy: boolean;
  currentJobId?: string;
  totalJobsCompleted: number;
  averageRenderTime: number;
  lastMemoryUsage?: MemoryUsage;
}

// Global persistent worker manager that survives page navigation
let globalOrchestratorWorkerManager: OrchestratorWorkerManager | null = null;

export const getOrchestratorWorkerManager = (): OrchestratorWorkerManager => {
  if (!globalOrchestratorWorkerManager) {
    globalOrchestratorWorkerManager = new OrchestratorWorkerManager();
    
    // Only initialize in browser environment
    if (typeof window !== 'undefined') {
      // Initialize for background processing
      globalOrchestratorWorkerManager.initialize().catch(error => {
        console.error('Failed to initialize global orchestrator worker manager:', error);
      });
      
      // Prevent termination on page unload - keep workers running
      window.addEventListener('beforeunload', (event) => {
        const queueLength = globalOrchestratorWorkerManager?.getMetrics().queueLength || 0;
        if (queueLength > 0) {
          // Warn user about ongoing jobs
          event.preventDefault();
          event.returnValue = `You have ${queueLength} rendering jobs in progress. Leaving will cancel them.`;
          return event.returnValue;
        }
      });
      
      // Handle visibility changes to optimize worker performance
      document.addEventListener('visibilitychange', () => {
        if (globalOrchestratorWorkerManager) {
          const metrics = globalOrchestratorWorkerManager.getMetrics();
          console.log(`🔄 [GlobalWorkerManager] Page visibility: ${document.hidden ? 'hidden' : 'visible'} | Queue: ${metrics.queueLength} | Active: ${metrics.activeWorkers}`);
        }
      });
    }
  }
  
  return globalOrchestratorWorkerManager;
};

// Export singleton instance - only initialize in browser
export const orchestratorWorkerManager = typeof window !== 'undefined' ? getOrchestratorWorkerManager() : null as OrchestratorWorkerManager;