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
});
