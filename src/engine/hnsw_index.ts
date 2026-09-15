import { vectorDistance, DistanceMetric, cosineSimilarity } from './similarity';

export interface HNSWNode {
  id: string;
  vector: Float32Array;
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
 * Fully compatible with WebAssembly and Web Worker environments.
 */
export class HNSWIndex {
  public readonly M: number;
  public readonly M0: number;
  public readonly efConstruction: number;
  public efSearch: number;
  public readonly metric: DistanceMetric;
  private readonly mL: number;

  private nodes: HNSWNode[] = [];
  private idToIndex: Map<string, number> = new Map();
  private entryPointIndex: number | null = null;
  private maxLevel = -1;

  constructor(options: HNSWOptions = {}) {
    this.M = options.M ?? 16;
    this.M0 = options.M0 ?? this.M * 2;
    this.efConstruction = Math.max(options.efConstruction ?? 64, this.M);
    this.efSearch = options.efSearch ?? 32;
    this.metric = options.metric ?? 'cosine';
    this.mL = 1.0 / Math.log(this.M);
  }

  public get size(): number {
    return this.nodes.length;
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
   */
  private dist(queryVector: Float32Array, nodeIndex: number): number {
    return vectorDistance(queryVector, this.nodes[nodeIndex].vector, this.metric, true);
  }

  /**
   * Greedy search on a single layer to find the closest element to the query.
   */
  private searchLayerGreedy(queryVector: Float32Array, enterPointIndex: number, level: number): number {
    let curr = enterPointIndex;
    let currDist = this.dist(queryVector, curr);

    let changed = true;
    while (changed) {
      changed = false;
      const neighbors = this.nodes[curr].neighbors[level] || [];
      for (let i = 0; i < neighbors.length; i++) {
        const neighborIdx = neighbors[i];
        const d = this.dist(queryVector, neighborIdx);
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
    level: number
  ): QueueItem[] {
    const visited = new Set<number>();
    // Candidates pool: min-heap ordered by distance (smallest dist first)
    const candidates: QueueItem[] = [];
    // Result set: tracks the top-ef nearest nodes encountered (sorted ascending by dist)
    const nearest: QueueItem[] = [];

    for (const ep of enterPointIndices) {
      const d = this.dist(queryVector, ep);
      visited.add(ep);
      const item = { index: ep, dist: d };
      this.insertSorted(candidates, item);
      this.insertSorted(nearest, item);
    }

    while (candidates.length > 0) {
      // Extract nearest candidate
      const curr = candidates.shift()!;
      const furthestNearestDist = nearest[nearest.length - 1].dist;

      if (curr.dist > furthestNearestDist && nearest.length >= ef) {
        break;
      }

      const neighbors = this.nodes[curr.index].neighbors[level] || [];
      for (let i = 0; i < neighbors.length; i++) {
        const neighborIdx = neighbors[i];
        if (visited.has(neighborIdx)) continue;
        visited.add(neighborIdx);

        const d = this.dist(queryVector, neighborIdx);
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
    // Return top-M candidate indices
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
      dist: this.dist(node.vector, nIdx),
    }));

    scored.sort((a, b) => a.dist - b.dist);
    node.neighbors[level] = scored.slice(0, maxDegree).map((s) => s.index);
  }

  /**
   * Inserts a vector into the HNSW graph.
   */
  public insert(id: string, vector: Float32Array): void {
    // If ID already exists, update vector in place or ignore
    if (this.idToIndex.has(id)) {
      const existingIdx = this.idToIndex.get(id)!;
      this.nodes[existingIdx].vector = vector;
      return;
    }

    const level = this.getRandomLevel();
    const nodeIdx = this.nodes.length;
    const newNode: HNSWNode = {
      id,
      vector,
      level,
      neighbors: Array.from({ length: level + 1 }, () => []),
    };

    this.nodes.push(newNode);
    this.idToIndex.set(id, nodeIdx);

    // If first node in graph
    if (this.entryPointIndex === null) {
      this.entryPointIndex = nodeIdx;
      this.maxLevel = level;
      return;
    }

    let currEnterPoint = this.entryPointIndex;
    const topLevel = this.maxLevel;

    // Phase 1: Traverse from top layer down to node's level + 1 using greedy search
    for (let l = topLevel; l > level; l--) {
      currEnterPoint = this.searchLayerGreedy(vector, currEnterPoint, l);
    }

    // Phase 2: From min(level, topLevel) down to layer 0, perform beam search and link neighbors
    let enterPoints = [currEnterPoint];
    const startLevel = Math.min(level, topLevel);

    for (let l = startLevel; l >= 0; l--) {
      const candidates = this.searchLayer(vector, enterPoints, this.efConstruction, l);
      const maxConn = l === 0 ? this.M0 : this.M;
      const selectedNeighbors = this.selectNeighbors(candidates, maxConn);

      for (const neighborIdx of selectedNeighbors) {
        this.addBidirectionalConnection(nodeIdx, neighborIdx, l, maxConn);
      }

      enterPoints = candidates.map((c) => c.index);
    }

    // Update global entry point if new node has a higher level
    if (level > this.maxLevel) {
      this.maxLevel = level;
      this.entryPointIndex = nodeIdx;
    }
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
    if (this.nodes.length === 0 || this.entryPointIndex === null) {
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

    // 1. Greedy search from top level down to level 1
    for (let l = this.maxLevel; l > 0; l--) {
      currEnterPoint = this.searchLayerGreedy(queryVector, currEnterPoint, l);
    }

    // 2. Beam search on layer 0 with efSearch
    const ef = Math.max(actualFilter ? actualEfSearch * 2 : actualEfSearch, topK);
    const candidates = this.searchLayer(queryVector, [currEnterPoint], ef, 0);

    const filtered = actualFilter
      ? candidates.filter((item) => actualFilter!(this.nodes[item.index].id))
      : candidates;

    // Return top-K candidates
    return filtered.slice(0, topK).map((item) => {
      const node = this.nodes[item.index];
      // Convert distance back to similarity score [0.0 - 1.0]
      let score: number;
      if (this.metric === 'cosine') {
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
   * Used for benchmark ground truth calculation and recall verification.
   */
  public bruteForceSearch(
    queryVector: Float32Array,
    topK: number,
    filterFn?: (id: string) => boolean
  ): SearchResult[] {
    const scored: Array<{ id: string; distance: number; score: number }> = [];

    for (let i = 0; i < this.nodes.length; i++) {
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
    if (idx === undefined) return null;
    return this.nodes[idx].vector;
  }

  /**
   * Returns all stored nodes for visualization and batch export.
   */
  public getAllNodes(): Array<{ id: string; vector: Float32Array; level: number }> {
    return this.nodes.map((n) => ({ id: n.id, vector: n.vector, level: n.level }));
  }

  /**
   * Serializes the HNSW graph into a JSON-compatible snapshot.
   */
  public serialize(): SerializedHNSW {
    return {
      M: this.M,
      M0: this.M0,
      efConstruction: this.efConstruction,
      efSearch: this.efSearch,
      metric: this.metric,
      entryPointIndex: this.entryPointIndex,
      maxLevel: this.maxLevel,
      nodes: this.nodes.map((n) => ({
        id: n.id,
        vector: Array.from(n.vector),
        level: n.level,
        neighbors: n.neighbors.map((layer) => [...layer]),
      })),
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
    });

    index.entryPointIndex = data.entryPointIndex;
    index.maxLevel = data.maxLevel;
    index.nodes = data.nodes.map((n) => ({
      id: n.id,
      vector: new Float32Array(n.vector),
      level: n.level,
      neighbors: n.neighbors.map((layer) => [...layer]),
    }));

    data.nodes.forEach((n, idx) => {
      index.idToIndex.set(n.id, idx);
    });

    return index;
  }

  public clear(): void {
    this.nodes = [];
    this.idToIndex.clear();
    this.entryPointIndex = null;
    this.maxLevel = -1;
  }
}
