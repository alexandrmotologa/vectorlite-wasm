import { describe, it, expect } from 'vitest';
import { HNSWIndex } from '../src/engine/hnsw_index';
import { normalize } from '../src/engine/similarity';

describe('HNSW In-Memory Vector Index', () => {
  it('inserts vectors and retrieves nearest neighbor with high recall', () => {
    const hnsw = new HNSWIndex({ M: 16, efConstruction: 64, efSearch: 32 });
    const dim = 64;
    const numVectors = 150;

    // Deterministic pseudo-random vectors
    for (let i = 0; i < numVectors; i++) {
      const raw = new Float32Array(dim);
      for (let d = 0; d < dim; d++) {
        raw[d] = Math.sin(i * 100 + d);
      }
      hnsw.insert(`vec-${i}`, normalize(raw));
    }

    expect(hnsw.size).toBe(numVectors);

    // Test query: search for vec-42
    const targetVector = hnsw.getVector('vec-42')!;
    const results = hnsw.search(targetVector, 5);

    expect(results.length).toBe(5);
    expect(results[0].id).toBe('vec-42');
    expect(results[0].score).toBeCloseTo(1.0, 3);
  });

  it('maintains >= 90% recall compared to brute-force k-NN ground truth', () => {
    const hnsw = new HNSWIndex({ M: 16, efConstruction: 64, efSearch: 64 });
    const dim = 32;
    const numVectors = 200;

    for (let i = 0; i < numVectors; i++) {
      const raw = new Float32Array(dim);
      for (let d = 0; d < dim; d++) {
        raw[d] = Math.cos(i * 37 + d * 13);
      }
      hnsw.insert(`id-${i}`, normalize(raw));
    }

    // Generate test query
    const query = new Float32Array(dim);
    for (let d = 0; d < dim; d++) {
      query[d] = Math.sin(d * 19);
    }
    const normalizedQuery = normalize(query);

    const k = 10;
    const groundTruth = hnsw.bruteForceSearch(normalizedQuery, k);
    const annResults = hnsw.search(normalizedQuery, k);

    const groundTruthSet = new Set(groundTruth.map((r) => r.id));
    const hits = annResults.filter((r) => groundTruthSet.has(r.id)).length;
    const recall = hits / k;

    // HNSW should achieve >= 90% recall on 200 vectors
    expect(recall).toBeGreaterThanOrEqual(0.9);
  });

  it('serializes and deserializes cleanly without data loss', () => {
    const hnsw = new HNSWIndex({ M: 8, efConstruction: 32 });
    const dim = 16;

    for (let i = 0; i < 20; i++) {
      const raw = new Float32Array(dim);
      for (let d = 0; d < dim; d++) {
        raw[d] = i + d;
      }
      hnsw.insert(`item-${i}`, normalize(raw));
    }

    const snapshot = hnsw.serialize();
    const restored = HNSWIndex.deserialize(snapshot);

    expect(restored.size).toBe(20);

    const query = hnsw.getVector('item-7')!;
    const originalRes = hnsw.search(query, 3);
    const restoredRes = restored.search(query, 3);

    expect(originalRes[0].id).toBe(restoredRes[0].id);
  });

  it('supports node deletion with lazy tombstoning and activeSize tracking', () => {
    const hnsw = new HNSWIndex({ M: 8, efConstruction: 32 });
    const dim = 8;

    for (let i = 0; i < 10; i++) {
      const raw = new Float32Array(dim).fill(i + 1);
      hnsw.insert(`node-${i}`, normalize(raw));
    }

    expect(hnsw.size).toBe(10);
    expect(hnsw.activeSize).toBe(10);

    // Deleting existing node returns true
    expect(hnsw.delete('node-3')).toBe(true);
    expect(hnsw.delete('node-7')).toBe(true);

    // Deleting again or non-existent returns false
    expect(hnsw.delete('node-3')).toBe(false);
    expect(hnsw.delete('node-999')).toBe(false);

    // Size remains 10 (lazy tombstone), activeSize becomes 8
    expect(hnsw.size).toBe(10);
    expect(hnsw.activeSize).toBe(8);

    // getVector returns null for deleted nodes
    expect(hnsw.getVector('node-3')).toBeNull();
    expect(hnsw.getVector('node-7')).toBeNull();
    expect(hnsw.getVector('node-0')).not.toBeNull();

    // getAllNodes excludes deleted nodes
    const allNodes = hnsw.getAllNodes();
    expect(allNodes.length).toBe(8);
    expect(allNodes.some((n) => n.id === 'node-3' || n.id === 'node-7')).toBe(false);

    // Querying exact vector of node-3 does NOT return node-3 in search or bruteForceSearch
    const queryNode3 = normalize(new Float32Array(dim).fill(4)); // node-3 vector
    const searchRes = hnsw.search(queryNode3, 5);
    expect(searchRes.some((r) => r.id === 'node-3')).toBe(false);

    const bfRes = hnsw.bruteForceSearch(queryNode3, 5);
    expect(bfRes.some((r) => r.id === 'node-3')).toBe(false);

    // Clear resets deletedSet as well
    hnsw.clear();
    expect(hnsw.size).toBe(0);
    expect(hnsw.activeSize).toBe(0);
  });

  it('serializes excluding deleted nodes and renumbers indices', () => {
    const hnsw = new HNSWIndex({ M: 8, efConstruction: 32 });
    const dim = 8;

    for (let i = 0; i < 5; i++) {
      const raw = new Float32Array(dim).fill(i + 1);
      hnsw.insert(`item-${i}`, normalize(raw));
    }

    hnsw.delete('item-1');
    hnsw.delete('item-3');

    const serialized = hnsw.serialize();
    expect(serialized.nodes.length).toBe(3);
    expect(serialized.nodes.map((n) => n.id)).toEqual(['item-0', 'item-2', 'item-4']);

    // Check that neighbor references are all valid indices < 3
    for (const node of serialized.nodes) {
      for (const layer of node.neighbors) {
        for (const neighborIdx of layer) {
          expect(neighborIdx).toBeGreaterThanOrEqual(0);
          expect(neighborIdx).toBeLessThan(3);
        }
      }
    }

    // Restore and verify
    const restored = HNSWIndex.deserialize(serialized);
    expect(restored.size).toBe(3);
    expect(restored.activeSize).toBe(3);
    expect(restored.getVector('item-1')).toBeNull();
    expect(restored.getVector('item-0')).not.toBeNull();
  });

  it('compacts the index by rebuilding without deleted nodes', () => {
    const hnsw = new HNSWIndex({ M: 8, efConstruction: 32 });
    const dim = 8;

    for (let i = 0; i < 20; i++) {
      const raw = new Float32Array(dim);
      for (let d = 0; d < dim; d++) raw[d] = i * 10 + d;
      hnsw.insert(`entry-${i}`, normalize(raw));
    }

    // Delete half the nodes
    for (let i = 0; i < 10; i++) {
      hnsw.delete(`entry-${i}`);
    }

    expect(hnsw.size).toBe(20);
    expect(hnsw.activeSize).toBe(10);

    hnsw.compact();

    // After compaction, physical size matches activeSize
    expect(hnsw.size).toBe(10);
    expect(hnsw.activeSize).toBe(10);

    // Remaining nodes are intact and searchable
    const query = hnsw.getVector('entry-15')!;
    expect(query).not.toBeNull();
    const results = hnsw.search(query, 3);
    expect(results[0].id).toBe('entry-15');
    expect(results.some((r) => r.id === 'entry-5')).toBe(false);
  });
});

