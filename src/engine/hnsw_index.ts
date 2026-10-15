import { vectorDistance, DistanceMetric, cosineSimilarity } from './similarity';
import {
  quantizeSQ8,
  dotProductSQ8,
  quantizeBQ,
  hammingDistanceBQ,
  QuantizedSQ8Vector,
  QuantizedBQVector,
} from './quantization';

export type QuantizationMode = 'f32' | 'sq8' | 'bq';

export interface HNSWNode {
  id: string;
  vector: Float32Array;
  quantizedSQ8?: QuantizedSQ8Vector;
  quantizedBQ?: QuantizedBQVector;
  level: number;
  /**
   * Neighbors per layer. Index 0 is bottom layer (layer 0).
   * Index l contains neighbor node indices on layer l.
   */
  neighbors: number[][];
}

export interface HNSWOptions {
  M?: number;              // Max connections per node on layers > 0 (default: 16)
  M0?: number;             // Max connections on layer 0 (default: 32)
  efConstruction?: number; // Search depth during index build (default: 64)
  efSearch?: number;       // Search depth during query (default: 32)
  metric?: DistanceMetric; // 'cosine' | 'dot' | 'euclidean' (default: 'cosine')
  quantization?: QuantizationMode; // 'f32' | 'sq8' | 'bq' (default: 'f32')
}

export interface SearchResult {
  id: string;
  distance: number;
  score: number; // Normalized similarity score [0.0, 1.0]
}

export interface SerializedHNSW {
  M: number;
  M0: number;
  efConstruction: number;
  efSearch: number;
  metric: DistanceMetric;
  quantization?: QuantizationMode;
  entryPointIndex: number | null;
  maxLevel: number;
  nodes: Array<{
    id: string;
    vector: number[];
    level: number;
    neighbors: number[][];
  }>;
}

/**
 * Priority queue item for best-first search.
 */
interface QueueItem {
  index: number;
  dist: number;
}

/**
 * In-memory Hierarchical Navigable Small World (HNSW) graph.
 * Supports native hardware SIMD, SQ8 Int8 quantization (-75% RAM), and 1-bit BQ (-96% RAM).
 */
export class HNSWIndex {
  public readonly M: number;
  public readonly M0: number;
  public readonly efConstruction: number;
  public efSearch: number;
  public readonly metric: DistanceMetric;
  public readonly quantization: QuantizationMode;
  private readonly mL: number;

  private nodes: HNSWNode[] = [];
  private idToIndex: Map<string, number> = new Map();
  private deletedSet: Set<number> = new Set();
  private entryPointIndex: number | null = null;
  private maxLevel = -1;

  constructor(options: HNSWOptions = {}) {
    this.M = options.M ?? 16;
    this.M0 = options.M0 ?? this.M * 2;
    this.efConstruction = Math.max(options.efConstruction ?? 64, this.M);
    this.efSearch = options.efSearch ?? 32;
    this.metric = options.metric ?? 'cosine';
    this.quantization = options.quantization ?? 'f32';
    this.mL = 1.0 / Math.log(this.M);
  }

  public get size(): number {
    return this.nodes.length;
  }

  public get activeSize(): number {
    return this.nodes.length - this.deletedSet.size;
  }

  public get entryPoint(): number | null {
    return this.entryPointIndex;
  }

  public get currentMaxLevel(): number {
    return this.maxLevel;
  }

  /**
   * Generates a random level for a new node with exponential decay.
   */
  private getRandomLevel(): number {
    const r = Math.random();
    if (r === 0) return 0;
    return Math.floor(-Math.log(r) * this.mL);
  }

  /**
   * Computes distance between query vector and a stored node.
   * Leverages SQ8 integer dot product or 1-bit Hamming POPCNT distance when quantized.
   */
  private dist(
    queryVector: Float32Array,
    nodeIndex: number,
    querySQ8?: QuantizedSQ8Vector,
    queryBQ?: QuantizedBQVector
  ): number {
    const node = this.nodes[nodeIndex];

    if (this.quantization === 'sq8' && node.quantizedSQ8) {
      const qSQ8 = querySQ8 || quantizeSQ8(queryVector);
      const dot = dotProductSQ8(qSQ8, node.quantizedSQ8);
      return 1.0 - Math.max(-1.0, Math.min(1.0, dot));
    }

    if (this.quantization === 'bq' && node.quantizedBQ) {
      const qBQ = queryBQ || quantizeBQ(queryVector);
      const dist = hammingDistanceBQ(qBQ, node.quantizedBQ);
      return dist / qBQ.dimension;
    }

    return vectorDistance(queryVector, node.vector, this.metric, true);
  }

  /**
   * Computes distance between two stored internal graph nodes.
   */
  private nodeDist(idxA: number, idxB: number): number {
    const a = this.nodes[idxA];
    const b = this.nodes[idxB];

    if (this.quantization === 'sq8' && a.quantizedSQ8 && b.quantizedSQ8) {
      const dot = dotProductSQ8(a.quantizedSQ8, b.quantizedSQ8);
      return 1.0 - Math.max(-1.0, Math.min(1.0, dot));
    }

    if (this.quantization === 'bq' && a.quantizedBQ && b.quantizedBQ) {
      const dist = hammingDistanceBQ(a.quantizedBQ, b.quantizedBQ);
      return dist / a.quantizedBQ.dimension;
    }

    return vectorDistance(a.vector, b.vector, this.metric, true);
  }

  /**
   * Greedy search on a single layer to find the closest element to the query.
   */
  private searchLayerGreedy(
    queryVector: Float32Array,
    enterPointIndex: number,
    level: number,
    querySQ8?: QuantizedSQ8Vector,
    queryBQ?: QuantizedBQVector
  ): number {
    let curr = enterPointIndex;
    let currDist = this.dist(queryVector, curr, querySQ8, queryBQ);

    let changed = true;
    while (changed) {
      changed = false;
      const neighbors = this.nodes[curr].neighbors[level] || [];
      for (let i = 0; i < neighbors.length; i++) {
        const neighborIdx = neighbors[i];
        if (this.deletedSet.has(neighborIdx)) continue;
        const d = this.dist(queryVector, neighborIdx, querySQ8, queryBQ);
        if (d < currDist) {
          currDist = d;
          curr = neighborIdx;
          changed = true;
        }
      }
    }
    return curr;
  }

  /**
   * Search layer using beam search with a given ef candidate pool.
   */
  private searchLayer(
    queryVector: Float32Array,
    enterPointIndices: number[],
    ef: number,
    level: number,
    querySQ8?: QuantizedSQ8Vector,
    queryBQ?: QuantizedBQVector
  ): QueueItem[] {
    const visited = new Set<number>();
    const candidates: QueueItem[] = [];
    const nearest: QueueItem[] = [];

    for (const ep of enterPointIndices) {
      const d = this.dist(queryVector, ep, querySQ8, queryBQ);
      visited.add(ep);
      const item = { index: ep, dist: d };
      this.insertSorted(candidates, item);
      this.insertSorted(nearest, item);
    }

    while (candidates.length > 0) {
      const curr = candidates.shift()!;
      const furthestNearestDist = nearest[nearest.length - 1].dist;

      if (curr.dist > furthestNearestDist && nearest.length >= ef) {
        break;
      }

      const neighbors = this.nodes[curr.index].neighbors[level] || [];
      for (let i = 0; i < neighbors.length; i++) {
        const neighborIdx = neighbors[i];
        if (this.deletedSet.has(neighborIdx)) continue;
        if (visited.has(neighborIdx)) continue;
        visited.add(neighborIdx);

        const d = this.dist(queryVector, neighborIdx, querySQ8, queryBQ);
        const worstNearest = nearest[nearest.length - 1];

        if (d < worstNearest.dist || nearest.length < ef) {
          const item = { index: neighborIdx, dist: d };
          this.insertSorted(candidates, item);
          this.insertSorted(nearest, item);

          if (nearest.length > ef) {
            nearest.pop();
          }
        }
      }
    }

    return nearest;
  }

  private insertSorted(arr: QueueItem[], item: QueueItem): void {
    let low = 0;
    let high = arr.length;
    while (low < high) {
      const mid = (low + high) >>> 1;
      if (arr[mid].dist < item.dist) {
        low = mid + 1;
      } else {
        high = mid;
      }
    }
    arr.splice(low, 0, item);
  }

  /**
   * Selects neighbors using simple heuristic or nearest selection.
   */
  private selectNeighbors(candidates: QueueItem[], maxNeighbors: number): number[] {
    return candidates.slice(0, maxNeighbors).map((c) => c.index);
  }

  /**
   * Connects two nodes bidirectionally at a specific layer, pruning if degree exceeds limit.
   */
  private addBidirectionalConnection(
    nodeAIdx: number,
    nodeBIdx: number,
    level: number,
    maxDegree: number
  ): void {
    const nodeA = this.nodes[nodeAIdx];
    const nodeB = this.nodes[nodeBIdx];

    if (!nodeA.neighbors[level].includes(nodeBIdx)) {
      nodeA.neighbors[level].push(nodeBIdx);
      if (nodeA.neighbors[level].length > maxDegree) {
        this.pruneNeighbors(nodeAIdx, level, maxDegree);
      }
    }

    if (!nodeB.neighbors[level].includes(nodeAIdx)) {
      nodeB.neighbors[level].push(nodeAIdx);
      if (nodeB.neighbors[level].length > maxDegree) {
        this.pruneNeighbors(nodeBIdx, level, maxDegree);
      }
    }
  }

  /**
   * Prunes neighbors of a node on a given layer back to maxDegree based on distance.
   */
  private pruneNeighbors(nodeIdx: number, level: number, maxDegree: number): void {
    const node = this.nodes[nodeIdx];
    const currentNeighbors = node.neighbors[level];
    if (currentNeighbors.length <= maxDegree) return;

    const scored: QueueItem[] = currentNeighbors.map((nIdx) => ({
      index: nIdx,
      dist: this.nodeDist(nodeIdx, nIdx),
    }));

    scored.sort((a, b) => a.dist - b.dist);
    node.neighbors[level] = scored.slice(0, maxDegree).map((s) => s.index);
  }

  /**
   * Inserts a vector into the HNSW graph with optional SQ8 or BQ quantization.
   */
  public insert(id: string, vector: Float32Array): void {
    const qSQ8 = this.quantization === 'sq8' ? quantizeSQ8(vector) : undefined;
    const qBQ = this.quantization === 'bq' ? quantizeBQ(vector) : undefined;

    if (this.idToIndex.has(id)) {
      const existingIdx = this.idToIndex.get(id)!;
      this.nodes[existingIdx].vector = vector;
      this.nodes[existingIdx].quantizedSQ8 = qSQ8;
      this.nodes[existingIdx].quantizedBQ = qBQ;
      return;
    }

    const level = this.getRandomLevel();
    const nodeIdx = this.nodes.length;
    const newNode: HNSWNode = {
      id,
      vector,
      quantizedSQ8: qSQ8,
      quantizedBQ: qBQ,
      level,
      neighbors: Array.from({ length: level + 1 }, () => []),
    };

    this.nodes.push(newNode);
    this.idToIndex.set(id, nodeIdx);

    if (this.entryPointIndex === null) {
      this.entryPointIndex = nodeIdx;
      this.maxLevel = level;
      return;
    }

    let currEnterPoint = this.entryPointIndex;
    const topLevel = this.maxLevel;

    // Pre-calculate quantized representations for routing if needed
    const querySQ8 = qSQ8;
    const queryBQ = qBQ;

    // Phase 1: Greedy traverse down to level + 1
    for (let l = topLevel; l > level; l--) {
      currEnterPoint = this.searchLayerGreedy(vector, currEnterPoint, l, querySQ8, queryBQ);
    }

    // Phase 2: Beam search from min(level, topLevel) down to layer 0
    let enterPoints = [currEnterPoint];
    const startLevel = Math.min(level, topLevel);

    for (let l = startLevel; l >= 0; l--) {
      const candidates = this.searchLayer(vector, enterPoints, this.efConstruction, l, querySQ8, queryBQ);
      const maxConn = l === 0 ? this.M0 : this.M;
      const selectedNeighbors = this.selectNeighbors(candidates, maxConn);

      for (const neighborIdx of selectedNeighbors) {
        this.addBidirectionalConnection(nodeIdx, neighborIdx, l, maxConn);
      }

      enterPoints = candidates.map((c) => c.index);
    }

    if (level > this.maxLevel) {
      this.maxLevel = level;
      this.entryPointIndex = nodeIdx;
    }
  }

  /**
   * Lazy deletes a node from the HNSW index by tombstoning its index.
   * Returns true if deleted, false if not found.
   */
  public delete(id: string): boolean {
    const idx = this.idToIndex.get(id);
    if (idx === undefined) {
      return false;
    }
    this.deletedSet.add(idx);
    this.idToIndex.delete(id);
    return true;
  }

  /**
   * Performs Approximate Nearest Neighbor (ANN) search with optional metadata filter predicate.
   */
  public search(
    queryVector: Float32Array,
    topK: number,
    efSearchOrFilter?: number | ((id: string) => boolean),
    filterFn?: (id: string) => boolean
  ): SearchResult[] {
    if (this.nodes.length === 0 || this.entryPointIndex === null || this.activeSize === 0) {
      return [];
    }

    let actualEfSearch = this.efSearch;
    let actualFilter = filterFn;

    if (typeof efSearchOrFilter === 'function') {
      actualFilter = efSearchOrFilter;
    } else if (typeof efSearchOrFilter === 'number') {
      actualEfSearch = efSearchOrFilter;
    }

    let currEnterPoint = this.entryPointIndex;
    const querySQ8 = this.quantization === 'sq8' ? quantizeSQ8(queryVector) : undefined;
    const queryBQ = this.quantization === 'bq' ? quantizeBQ(queryVector) : undefined;

    // 1. Greedy search from top level down to level 1
    for (let l = this.maxLevel; l > 0; l--) {
      currEnterPoint = this.searchLayerGreedy(queryVector, currEnterPoint, l, querySQ8, queryBQ);
    }

    // 2. Beam search on layer 0 with efSearch
    const ef = Math.max(actualFilter ? actualEfSearch * 2 : actualEfSearch, topK);
    const candidates = this.searchLayer(queryVector, [currEnterPoint], ef, 0, querySQ8, queryBQ);

    const nonDeleted = candidates.filter((item) => !this.deletedSet.has(item.index));

    const filtered = actualFilter
      ? nonDeleted.filter((item) => actualFilter!(this.nodes[item.index].id))
      : nonDeleted;

    return filtered.slice(0, topK).map((item) => {
      const node = this.nodes[item.index];
      let score: number;

      if (this.quantization === 'bq') {
        score = Math.max(0.0, Math.min(1.0, 1.0 - item.dist));
      } else if (this.metric === 'cosine') {
        score = Math.max(0.0, Math.min(1.0, 1.0 - item.dist));
      } else if (this.metric === 'dot') {
        score = Math.max(0.0, Math.min(1.0, (-item.dist + 1.0) / 2.0));
      } else {
        score = 1.0 / (1.0 + item.dist);
      }

      return {
        id: node.id,
        distance: item.dist,
        score,
      };
    });
  }

  /**
   * Exact k-NN Brute Force Search (linear scan O(N)) with optional filter.
   */
  public bruteForceSearch(
    queryVector: Float32Array,
    topK: number,
    filterFn?: (id: string) => boolean
  ): SearchResult[] {
    const scored: Array<{ id: string; distance: number; score: number }> = [];

    for (let i = 0; i < this.nodes.length; i++) {
      if (this.deletedSet.has(i)) continue;
      const node = this.nodes[i];
      if (filterFn && !filterFn(node.id)) continue;

      const d = vectorDistance(queryVector, node.vector, this.metric, true);
      const score = Math.max(0.0, Math.min(1.0, cosineSimilarity(queryVector, node.vector, true)));
      scored.push({ id: node.id, distance: d, score });
    }

    scored.sort((a, b) => a.distance - b.distance);
    return scored.slice(0, topK);
  }

  /**
   * Retrieves vector by node ID.
   */
  public getVector(id: string): Float32Array | null {
    const idx = this.idToIndex.get(id);
    if (idx === undefined || this.deletedSet.has(idx)) return null;
    return this.nodes[idx].vector;
  }

  /**
   * Returns all stored nodes for visualization and batch export.
   */
  public getAllNodes(): Array<{ id: string; vector: Float32Array; level: number }> {
    return this.nodes
      .filter((_, idx) => !this.deletedSet.has(idx))
      .map((n) => ({ id: n.id, vector: n.vector, level: n.level }));
  }

  /**
   * Serializes the HNSW graph into a JSON-compatible snapshot excluding deleted nodes.
   */
  public serialize(): SerializedHNSW {
    const oldToNew = new Map<number, number>();
    let newIdx = 0;
    for (let i = 0; i < this.nodes.length; i++) {
      if (!this.deletedSet.has(i)) {
        oldToNew.set(i, newIdx++);
      }
    }

    let newEntryPointIndex: number | null = null;
    let newMaxLevel = -1;

    if (this.entryPointIndex !== null && oldToNew.has(this.entryPointIndex)) {
      newEntryPointIndex = oldToNew.get(this.entryPointIndex)!;
      newMaxLevel = this.maxLevel;
    } else if (oldToNew.size > 0) {
      let bestOldIdx = -1;
      let bestLevel = -1;
      for (let i = 0; i < this.nodes.length; i++) {
        if (!this.deletedSet.has(i) && this.nodes[i].level > bestLevel) {
          bestLevel = this.nodes[i].level;
          bestOldIdx = i;
        }
      }
      if (bestOldIdx !== -1) {
        newEntryPointIndex = oldToNew.get(bestOldIdx)!;
        newMaxLevel = bestLevel;
      }
    }

    const serializedNodes: Array<{
      id: string;
      vector: number[];
      level: number;
      neighbors: number[][];
    }> = [];

    for (let i = 0; i < this.nodes.length; i++) {
      if (this.deletedSet.has(i)) continue;
      const n = this.nodes[i];
      serializedNodes.push({
        id: n.id,
        vector: Array.from(n.vector),
        level: n.level,
        neighbors: n.neighbors.map((layer) =>
          layer
            .filter((neighborIdx) => oldToNew.has(neighborIdx))
            .map((neighborIdx) => oldToNew.get(neighborIdx)!)
        ),
      });
    }

    return {
      M: this.M,
      M0: this.M0,
      efConstruction: this.efConstruction,
      efSearch: this.efSearch,
      metric: this.metric,
      quantization: this.quantization,
      entryPointIndex: newEntryPointIndex,
      maxLevel: newMaxLevel,
      nodes: serializedNodes,
    };
  }

  /**
   * Restores an HNSW index from a serialized snapshot.
   */
  public static deserialize(data: SerializedHNSW): HNSWIndex {
    const index = new HNSWIndex({
      M: data.M,
      M0: data.M0,
      efConstruction: data.efConstruction,
      efSearch: data.efSearch,
      metric: data.metric,
      quantization: data.quantization ?? 'f32',
    });

    index.entryPointIndex = data.entryPointIndex;
    index.maxLevel = data.maxLevel;
    index.nodes = data.nodes.map((n) => {
      const vec = new Float32Array(n.vector);
      return {
        id: n.id,
        vector: vec,
        quantizedSQ8: index.quantization === 'sq8' ? quantizeSQ8(vec) : undefined,
        quantizedBQ: index.quantization === 'bq' ? quantizeBQ(vec) : undefined,
        level: n.level,
        neighbors: n.neighbors.map((layer) => [...layer]),
      };
    });

    data.nodes.forEach((n, idx) => {
      index.idToIndex.set(n.id, idx);
    });

    return index;
  }

  /**
   * Rebuilds the index from scratch using only non-deleted nodes.
   * Cleans up tombstoned nodes and re-optimizes graph connectivity.
   */
  public compact(): void {
    const freshIndex = new HNSWIndex({
      M: this.M,
      M0: this.M0,
      efConstruction: this.efConstruction,
      efSearch: this.efSearch,
      metric: this.metric,
      quantization: this.quantization,
    });

    for (let i = 0; i < this.nodes.length; i++) {
      if (this.deletedSet.has(i)) continue;
      const node = this.nodes[i];
      freshIndex.insert(node.id, node.vector);
    }

    this.nodes = freshIndex.nodes;
    this.idToIndex = freshIndex.idToIndex;
    this.deletedSet.clear();
    this.entryPointIndex = freshIndex.entryPointIndex;
    this.maxLevel = freshIndex.maxLevel;
  }

  public clear(): void {
    this.nodes = [];
    this.idToIndex.clear();
    this.deletedSet.clear();
    this.entryPointIndex = null;
    this.maxLevel = -1;
  }
}
