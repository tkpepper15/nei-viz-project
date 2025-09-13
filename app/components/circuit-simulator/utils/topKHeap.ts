/**
 * High-performance Top-K Min-Heap for maintaining best parameter candidates
 * Optimized for circuit parameter fitting applications
 */

export interface HeapItem {
  resnorm: number;
  parameters: {
    Rsh: number;
    Ra: number;
    Ca: number;
    Rb: number;
    Cb: number;
  };
  index?: number;
  fingerprint?: string;
  metadata?: {
    stage?: number;
    processingTime?: number;
    chunkIndex?: number;
  };
}

export class TopKHeap {
  private heap: HeapItem[] = [];
  private readonly maxSize: number;
  
  constructor(maxSize: number = 5000) {
    this.maxSize = maxSize;
  }

  /**
   * Add item to heap, maintaining Top-K by lowest resnorm
   * O(log K) complexity
   */
  push(item: HeapItem): boolean {
    if (this.heap.length < this.maxSize) {
      // Heap not full, add item
      this.heap.push(item);
      this.bubbleUp(this.heap.length - 1);
      return true;
    } else {
      // Heap full, only add if better than worst
      if (item.resnorm < this.heap[0].resnorm) {
        this.heap[0] = item;
        this.bubbleDown(0);
        return true;
      }
      return false;
    }
  }

  /**
   * Add multiple items efficiently
   */
  pushMany(items: HeapItem[]): number {
    let addedCount = 0;
    for (const item of items) {
      if (this.push(item)) {
        addedCount++;
      }
    }
    return addedCount;
  }

  /**
   * Get current worst resnorm (heap root)
   */
  getWorstResnorm(): number {
    return this.heap.length > 0 ? this.heap[0].resnorm : Infinity;
  }

  /**
   * Get current best resnorm
   */
  getBestResnorm(): number {
    if (this.heap.length === 0) return Infinity;
    return Math.min(...this.heap.map(item => item.resnorm));
  }

  /**
   * Get all items sorted by resnorm (ascending - best first)
   */
  getSortedItems(): HeapItem[] {
    return [...this.heap].sort((a, b) => a.resnorm - b.resnorm);
  }

  /**
   * Get top N items (best N)
   */
  getTopN(n: number): HeapItem[] {
    return this.getSortedItems().slice(0, Math.min(n, this.heap.length));
  }

  /**
   * Current size of heap
   */
  size(): number {
    return this.heap.length;
  }

  /**
   * Check if heap is at capacity
   */
  isFull(): boolean {
    return this.heap.length >= this.maxSize;
  }

  /**
   * Clear all items
   */
  clear(): void {
    this.heap.length = 0;
  }

  /**
   * Get heap statistics
   */
  getStats(): {
    size: number;
    maxSize: number;
    bestResnorm: number;
    worstResnorm: number;
    medianResnorm: number;
  } {
    const sorted = this.getSortedItems();
    const medianIndex = Math.floor(sorted.length / 2);
    
    return {
      size: this.heap.length,
      maxSize: this.maxSize,
      bestResnorm: sorted[0]?.resnorm ?? Infinity,
      worstResnorm: sorted[sorted.length - 1]?.resnorm ?? Infinity,
      medianResnorm: sorted[medianIndex]?.resnorm ?? Infinity
    };
  }

  /**
   * Filter items by tolerance (keep near-ties)
   */
  getItemsWithinTolerance(tolerance: number = 0.005): HeapItem[] {
    const sorted = this.getSortedItems();
    if (sorted.length === 0) return [];
    
    const bestResnorm = sorted[0].resnorm;
    const threshold = bestResnorm * (1 + tolerance);
    
    return sorted.filter(item => item.resnorm <= threshold);
  }

  private bubbleUp(index: number): void {
    while (index > 0) {
      const parentIndex = Math.floor((index - 1) / 2);
      
      // Max-heap property: parent >= child (we want worst at root for easy removal)
      if (this.heap[parentIndex].resnorm >= this.heap[index].resnorm) {
        break;
      }
      
      this.swap(parentIndex, index);
      index = parentIndex;
    }
  }

  private bubbleDown(index: number): void {
    while (true) {
      let maxIndex = index;
      const leftChild = 2 * index + 1;
      const rightChild = 2 * index + 2;
      
      // Find child with maximum resnorm
      if (leftChild < this.heap.length && 
          this.heap[leftChild].resnorm > this.heap[maxIndex].resnorm) {
        maxIndex = leftChild;
      }
      
      if (rightChild < this.heap.length && 
          this.heap[rightChild].resnorm > this.heap[maxIndex].resnorm) {
        maxIndex = rightChild;
      }
      
      if (maxIndex === index) {
        break;
      }
      
      this.swap(index, maxIndex);
      index = maxIndex;
    }
  }

  private swap(i: number, j: number): void {
    const temp = this.heap[i];
    this.heap[i] = this.heap[j];
    this.heap[j] = temp;
  }
}

/**
 * Specialized heap for fingerprint-based deduplication
 */
export class FingerprintHeap {
  private fingerprintMap = new Map<string, HeapItem[]>();
  private readonly candidatesPerKey: number;
  
  constructor(candidatesPerKey: number = 3) {
    this.candidatesPerKey = candidatesPerKey;
  }

  /**
   * Add item, grouped by fingerprint
   */
  addCandidate(item: HeapItem, fingerprint: string): void {
    if (!this.fingerprintMap.has(fingerprint)) {
      this.fingerprintMap.set(fingerprint, []);
    }
    
    const candidates = this.fingerprintMap.get(fingerprint)!;
    candidates.push(item);
    
    // Keep only best candidates per fingerprint
    candidates.sort((a, b) => a.resnorm - b.resnorm);
    if (candidates.length > this.candidatesPerKey) {
      candidates.splice(this.candidatesPerKey);
    }
  }

  /**
   * Get all representative candidates
   */
  getRepresentatives(): HeapItem[] {
    const representatives: HeapItem[] = [];
    
    for (const candidates of this.fingerprintMap.values()) {
      representatives.push(...candidates);
    }
    
    return representatives.sort((a, b) => a.resnorm - b.resnorm);
  }

  /**
   * Get statistics about fingerprint diversity
   */
  getStats(): {
    uniqueFingerprints: number;
    totalCandidates: number;
    averageCandidatesPerFingerprint: number;
  } {
    const uniqueFingerprints = this.fingerprintMap.size;
    const totalCandidates = Array.from(this.fingerprintMap.values())
      .reduce((sum, candidates) => sum + candidates.length, 0);
    
    return {
      uniqueFingerprints,
      totalCandidates,
      averageCandidatesPerFingerprint: totalCandidates / uniqueFingerprints || 0
    };
  }

  clear(): void {
    this.fingerprintMap.clear();
  }
}