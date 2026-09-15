import { describe, it, expect } from 'vitest';
import { HNSWIndex } from '../src/engine/hnsw_index';
import { BM25Index } from '../src/engine/bm25';
import { normalize } from '../src/engine/similarity';

describe('Metadata Filtering in HNSW & BM25', () => {
  it('filters HNSW search results based on predicate', () => {
    const hnsw = new HNSWIndex({ M: 8, efConstruction: 32 });
    const dim = 16;

    for (let i = 0; i < 30; i++) {
      const vec = new Float32Array(dim);
      for (let d = 0; d < dim; d++) {
        vec[d] = i + d;
      }
      const category = i % 2 === 0 ? 'finance' : 'tech';
      hnsw.insert(`${category}-chunk-${i}`, normalize(vec));
    }

    const query = hnsw.getVector('finance-chunk-10')!;
    // Search only tech chunks
    const results = hnsw.search(query, 5, 32, (id) => id.startsWith('tech-'));

    expect(results.length).toBeGreaterThan(0);
    for (const r of results) {
      expect(r.id.startsWith('tech-')).toBe(true);
    }
  });

  it('filters BM25 search results based on predicate', () => {
    const bm25 = new BM25Index();

    bm25.addDocument('report-2024.pdf-chunk-1', 'Financial revenue quarterly projections.');
    bm25.addDocument('report-2025.pdf-chunk-1', 'Financial revenue quarterly projections for next year.');
    bm25.addDocument('system.md-chunk-1', 'Server revenue architecture.');

    // Only allow 2025 report chunks
    const results = bm25.search('Financial revenue', 5, (id) => id.includes('2025'));

    expect(results.length).toBe(1);
    expect(results[0].id).toBe('report-2025.pdf-chunk-1');
  });
});
