/**
 * Intelligent Multi-Worker Management System
 * 
 * Features:
 * - Dynamic worker pool management (up to 12 workers)
 * - Task-specific worker assignment and specialization
 * - Automatic load balancing and idle worker termination
 * - Promise-based task execution with cancellation support
 * - Transferable object optimization for large datasets
 * - Performance monitoring and adaptive scaling
 */

import { ModelSnapshot } from '../types';

export interface WorkerTask<T = unknown> {
  id: string;
  type: WorkerTaskType;
  data: T;
  transferables?: Transferable[];
  priority: 'low' | 'normal' | 'high' | 'critical';
  timeout?: number;
  onProgress?: (progress: number, message?: string) => void;
}

export interface WorkerResult<T = unknown> {
  id: string;
  success: boolean;
  data?: T;
  error?: string;
  duration: number;
  workerId: string;
}

export type WorkerTaskType = 'spider-plot' | 'resnorm-calculation' | 'data-processing' | 'image-generation' | 'mesh-computation';

interface WorkerInstance {
  id: string;
  worker: Worker;
  type: WorkerTaskType | 'general';
  busy: boolean;
  taskCount: number;
  lastUsed: number;
  performance: {
    avgDuration: number;
    taskHistory: number[];
  };
}

interface PendingTask<T> {
  task: WorkerTask<T>;
  resolve: (result: WorkerResult<T>) => void;
  reject: (error: Error) => void;
  startTime: number;
  abortController?: AbortController;
}

class IntelligentWorkerManager {
  private workers: Map<string, WorkerInstance> = new Map();
  private pendingTasks: Map<string, PendingTask<unknown>> = new Map();
  private taskQueue: Array<PendingTask<unknown>> = [];
  private maxWorkers: number = 12;
  private idleTimeout: number = 60000; // 60 seconds
  private cleanupInterval: NodeJS.Timeout | null = null;
  
  // Worker script URLs for different task types
  private workerScripts: Record<WorkerTaskType, string> = {
    'spider-plot': '/orchestrator-worker.js',
    'resnorm-calculation': '/grid-worker.js',
    'data-processing': '/enhanced-tile-worker.js',
    'image-generation': '/orchestrator-worker.js',
    'mesh-computation': '/grid-worker.js'
  };

  constructor() {
    this.startCleanupTimer();
    this.initializeWorkerPool();
  }

  /**
   * Initialize optimal number of workers based on hardware
   */
  private initializeWorkerPool(): void {
    // Check if we're in browser environment
    if (typeof window === 'undefined' || typeof navigator === 'undefined') {
      this.maxWorkers = 4; // Default for SSR
      console.log('🚫 [IntelligentWorkerManager] SSR environment detected - workers will initialize on client');
      return;
    }
    
    const hardwareConcurrency = navigator.hardwareConcurrency || 4;
    // Use 2x hardware concurrency but cap at 12 for optimal performance
    this.maxWorkers = Math.min(12, Math.max(4, hardwareConcurrency * 2));
    
    console.log(`🧠 [IntelligentWorkerManager] Initialized with max ${this.maxWorkers} workers (Hardware: ${hardwareConcurrency} cores)`);
  }

  /**
   * Execute a task with intelligent worker assignment
   */
  async runTask<T>(task: WorkerTask<T>): Promise<WorkerResult<T>> {
    return new Promise((resolve, reject) => {
      const pendingTask: PendingTask<T> = {
        task,
        resolve,
        reject,
        startTime: performance.now(),
        abortController: new AbortController()
      };

      this.pendingTasks.set(task.id, pendingTask);

      // Try to assign immediately or queue
      if (!this.tryAssignTask(pendingTask)) {
        // Add to priority queue
        this.addToQueue(pendingTask);
        console.log(`📋 [IntelligentWorkerManager] Task ${task.id} queued (${this.taskQueue.length} in queue)`);
      }
    });
  }

  /**
   * Try to assign task to best available worker
   */
  private tryAssignTask<T, R>(pendingTask: PendingTask<T, R>): boolean {
    const { task } = pendingTask;
    
    // Find best worker for this task type
    let bestWorker = this.findBestWorker(task.type);
    
    if (!bestWorker && this.workers.size < this.maxWorkers) {
      // Create new specialized worker
      bestWorker = this.createWorker(task.type);
    }
    
    if (!bestWorker) {
      // All workers busy, find least busy general worker
      bestWorker = this.findLeastBusyWorker();
    }

    if (bestWorker && !bestWorker.busy) {
      this.assignTaskToWorker(pendingTask, bestWorker);
      return true;
    }

    return false;
  }

  /**
   * Find the best worker for a specific task type
   */
  private findBestWorker(taskType: WorkerTaskType): WorkerInstance | null {
    const candidates = Array.from(this.workers.values())
      .filter(w => !w.busy && (w.type === taskType || w.type === 'general'))
      .sort((a, b) => {
        // Prefer specialized workers, then by performance
        if (a.type === taskType && b.type !== taskType) return -1;
        if (b.type === taskType && a.type !== taskType) return 1;
        return a.performance.avgDuration - b.performance.avgDuration;
      });

    return candidates[0] || null;
  }

  /**
   * Find least busy worker when no specialized worker is available
   */
  private findLeastBusyWorker(): WorkerInstance | null {
    const available = Array.from(this.workers.values())
      .filter(w => !w.busy)
      .sort((a, b) => a.taskCount - b.taskCount);

    return available[0] || null;
  }

  /**
   * Create a new worker instance
   */
  private createWorker(type: WorkerTaskType | 'general' = 'general'): WorkerInstance | null {
    // Check if we're in browser environment
    if (typeof window === 'undefined' || typeof Worker === 'undefined') {
      console.log('🚫 [IntelligentWorkerManager] Cannot create worker in SSR environment');
      return null;
    }
    
    const workerId = `worker-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const scriptUrl = type !== 'general' ? this.workerScripts[type] : this.workerScripts['spider-plot'];
    
    const worker = new Worker(scriptUrl);
    const workerInstance: WorkerInstance = {
      id: workerId,
      worker,
      type,
      busy: false,
      taskCount: 0,
      lastUsed: Date.now(),
      performance: {
        avgDuration: 0,
        taskHistory: []
      }
    };

    // Set up worker message handling
    worker.onmessage = (event) => this.handleWorkerMessage(workerId, event);
    worker.onerror = (error) => this.handleWorkerError(workerId, error);

    this.workers.set(workerId, workerInstance);
    console.log(`🆕 [IntelligentWorkerManager] Created ${type} worker: ${workerId} (Total: ${this.workers.size})`);
    
    return workerInstance;
  }

  /**
   * Assign a specific task to a worker
   */
  private assignTaskToWorker<T, R>(pendingTask: PendingTask<T, R>, worker: WorkerInstance): void {
    const { task } = pendingTask;
    
    worker.busy = true;
    worker.taskCount++;
    worker.lastUsed = Date.now();

    console.log(`🎯 [IntelligentWorkerManager] Assigning ${task.type} task ${task.id} to worker ${worker.id}`);

    // Send task to worker with transferables if available
    if (task.transferables && task.transferables.length > 0) {
      worker.worker.postMessage({
        taskId: task.id,
        type: task.type,
        data: task.data,
        timestamp: pendingTask.startTime
      }, task.transferables);
    } else {
      worker.worker.postMessage({
        taskId: task.id,
        type: task.type,
        data: task.data,
        timestamp: pendingTask.startTime
      });
    }

    // Set up timeout if specified
    if (task.timeout) {
      setTimeout(() => {
        if (this.pendingTasks.has(task.id)) {
          this.handleTaskTimeout(task.id);
        }
      }, task.timeout);
    }
  }

  /**
   * Handle worker messages
   */
  private handleWorkerMessage(workerId: string, event: MessageEvent): void {
    const { taskId, type, data, error, progress, message } = event.data;
    
    if (type === 'progress' && progress !== undefined) {
      const pendingTask = this.pendingTasks.get(taskId);
      if (pendingTask?.task.onProgress) {
        pendingTask.task.onProgress(progress, message);
      }
      return;
    }

    if (type === 'complete' || type === 'error') {
      this.handleTaskCompletion(workerId, taskId, { data, error, type });
    }
  }

  /**
   * Handle task completion
   */
  private handleTaskCompletion(workerId: string, taskId: string, result: unknown): void {
    const worker = this.workers.get(workerId);
    const pendingTask = this.pendingTasks.get(taskId);

    if (!worker || !pendingTask) return;

    // Update worker status
    worker.busy = false;
    worker.lastUsed = Date.now();

    const duration = performance.now() - pendingTask.startTime;
    
    // Update performance metrics
    worker.performance.taskHistory.push(duration);
    if (worker.performance.taskHistory.length > 10) {
      worker.performance.taskHistory = worker.performance.taskHistory.slice(-10);
    }
    worker.performance.avgDuration = worker.performance.taskHistory.reduce((a, b) => a + b, 0) / worker.performance.taskHistory.length;

    // Create result
    const workerResult: WorkerResult = {
      id: taskId,
      success: result.type === 'complete',
      data: result.data,
      error: result.error,
      duration,
      workerId
    };

    // Resolve or reject the pending task
    if (result.type === 'complete') {
      pendingTask.resolve(workerResult);
    } else {
      pendingTask.reject(new Error(result.error || 'Worker task failed'));
    }

    // Clean up
    this.pendingTasks.delete(taskId);

    console.log(`✅ [IntelligentWorkerManager] Task ${taskId} completed by ${workerId} in ${duration.toFixed(2)}ms`);

    // Process next task in queue
    this.processQueue();
  }

  /**
   * Handle worker errors
   */
  private handleWorkerError(workerId: string, error: ErrorEvent): void {
    console.error(`❌ [IntelligentWorkerManager] Worker ${workerId} error:`, error);
    
    const worker = this.workers.get(workerId);
    if (worker) {
      worker.busy = false;
      // Find any pending tasks for this worker and reject them
      for (const [taskId, pendingTask] of this.pendingTasks.entries()) {
        pendingTask.reject(new Error(`Worker ${workerId} encountered an error: ${error.message}`));
        this.pendingTasks.delete(taskId);
      }
    }
  }

  /**
   * Handle task timeout
   */
  private handleTaskTimeout(taskId: string): void {
    const pendingTask = this.pendingTasks.get(taskId);
    if (pendingTask) {
      pendingTask.reject(new Error(`Task ${taskId} timed out`));
      this.pendingTasks.delete(taskId);
      console.warn(`⏰ [IntelligentWorkerManager] Task ${taskId} timed out`);
    }
  }

  /**
   * Add task to priority queue
   */
  private addToQueue<T, R>(pendingTask: PendingTask<T, R>): void {
    const priorityOrder = { critical: 0, high: 1, normal: 2, low: 3 };
    const taskPriority = priorityOrder[pendingTask.task.priority];
    
    // Insert in priority order
    let insertIndex = this.taskQueue.length;
    for (let i = 0; i < this.taskQueue.length; i++) {
      const queuePriority = priorityOrder[this.taskQueue[i].task.priority];
      if (taskPriority < queuePriority) {
        insertIndex = i;
        break;
      }
    }
    
    this.taskQueue.splice(insertIndex, 0, pendingTask);
  }

  /**
   * Process next task in queue
   */
  private processQueue(): void {
    if (this.taskQueue.length === 0) return;

    const nextTask = this.taskQueue.shift();
    if (nextTask && this.tryAssignTask(nextTask)) {
      // Successfully assigned, continue processing
      this.processQueue();
    } else if (nextTask) {
      // Put it back at the front
      this.taskQueue.unshift(nextTask);
    }
  }

  /**
   * Cancel a specific task
   */
  cancelTask(taskId: string): boolean {
    const pendingTask = this.pendingTasks.get(taskId);
    if (pendingTask) {
      pendingTask.abortController?.abort();
      pendingTask.reject(new Error(`Task ${taskId} was cancelled`));
      this.pendingTasks.delete(taskId);
      
      // Remove from queue if present
      const queueIndex = this.taskQueue.findIndex(t => t.task.id === taskId);
      if (queueIndex !== -1) {
        this.taskQueue.splice(queueIndex, 1);
      }
      
      console.log(`🚫 [IntelligentWorkerManager] Cancelled task ${taskId}`);
      return true;
    }
    return false;
  }

  /**
   * Get manager status and metrics
   */
  getStatus() {
    const workerStats = Array.from(this.workers.values()).map(w => ({
      id: w.id,
      type: w.type,
      busy: w.busy,
      taskCount: w.taskCount,
      avgDuration: w.performance.avgDuration
    }));

    return {
      totalWorkers: this.workers.size,
      busyWorkers: Array.from(this.workers.values()).filter(w => w.busy).length,
      pendingTasks: this.pendingTasks.size,
      queuedTasks: this.taskQueue.length,
      workers: workerStats
    };
  }

  /**
   * Start cleanup timer for idle workers
   */
  private startCleanupTimer(): void {
    // Only start cleanup timer in browser environment
    if (typeof window === 'undefined') {
      return;
    }
    
    this.cleanupInterval = setInterval(() => {
      this.cleanupIdleWorkers();
    }, 30000); // Check every 30 seconds
  }

  /**
   * Clean up idle workers
   */
  private cleanupIdleWorkers(): void {
    const now = Date.now();
    const workersToTerminate: string[] = [];

    for (const [workerId, worker] of this.workers.entries()) {
      if (!worker.busy && (now - worker.lastUsed) > this.idleTimeout) {
        workersToTerminate.push(workerId);
      }
    }

    for (const workerId of workersToTerminate) {
      this.terminateWorker(workerId);
    }

    if (workersToTerminate.length > 0) {
      console.log(`🧹 [IntelligentWorkerManager] Cleaned up ${workersToTerminate.length} idle workers`);
    }
  }

  /**
   * Terminate a specific worker
   */
  private terminateWorker(workerId: string): void {
    const worker = this.workers.get(workerId);
    if (worker) {
      worker.worker.terminate();
      this.workers.delete(workerId);
      console.log(`💀 [IntelligentWorkerManager] Terminated worker ${workerId}`);
    }
  }

  /**
   * Shutdown all workers and cleanup
   */
  shutdown(): void {
    console.log(`🛑 [IntelligentWorkerManager] Shutting down all workers...`);
    
    // Cancel all pending tasks
    for (const pendingTask of this.pendingTasks.values()) {
      pendingTask.reject(new Error('Worker manager is shutting down'));
    }
    this.pendingTasks.clear();
    this.taskQueue.length = 0;

    // Terminate all workers
    for (const worker of this.workers.values()) {
      worker.worker.terminate();
    }
    this.workers.clear();

    // Clear cleanup timer
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }
  }
}

// Export singleton instance
export const intelligentWorkerManager = new IntelligentWorkerManager();

// Utility functions for common task types
export const WorkerTasks = {
  /**
   * Create spider plot rendering task
   */
  createSpiderPlotTask(
    models: ModelSnapshot[],
    renderParams: unknown,
    options: {
      priority?: 'low' | 'normal' | 'high' | 'critical';
      onProgress?: (progress: number, message?: string) => void;
      timeout?: number;
    } = {}
  ): WorkerTask {
    return {
      id: `spider-plot-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type: 'spider-plot',
      data: { models, renderParams },
      priority: options.priority || 'normal',
      onProgress: options.onProgress,
      timeout: options.timeout || 300000 // 5 minute default timeout
    };
  },

  /**
   * Create resnorm calculation task
   */
  createResnormTask(
    gridData: unknown,
    options: {
      priority?: 'low' | 'normal' | 'high' | 'critical';
      onProgress?: (progress: number, message?: string) => void;
    } = {}
  ): WorkerTask {
    return {
      id: `resnorm-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type: 'resnorm-calculation',
      data: gridData,
      priority: options.priority || 'normal',
      onProgress: options.onProgress
    };
  }
};

// React hook for using the worker manager
export function useIntelligentWorkerManager() {
  const runTask = <T, R>(task: WorkerTask<T, R>) => intelligentWorkerManager.runTask(task);
  const cancelTask = (taskId: string) => intelligentWorkerManager.cancelTask(taskId);
  const getStatus = () => intelligentWorkerManager.getStatus();

  // Cleanup on unmount
  React.useEffect(() => {
    return () => {
      // Note: Don't shutdown on unmount as other components might be using it
      // Only shutdown on app-level cleanup
    };
  }, []);

  return {
    runTask,
    cancelTask,
    getStatus
  };
}