import { describe, it, expect } from 'vitest';
import { BM25Index } from '../src/engine/bm25';
import { reciprocalRankFusion } from '../src/engine/hybrid';

describe('BM25 Lexical & Hybrid Search', () => {
  it('retrieves exact keyword matches with high relevance', () => {
    const bm25 = new BM25Index();

    bm25.addDocument('doc-1', 'PostgreSQL database connection pooling with PgBouncer.');
    bm25.addDocument('doc-2', 'Redis in-memory caching and distributed lock mechanisms.');
    bm25.addDocument('doc-3', 'Kubernetes pod orchestration, ingress controllers, and service meshes.');

    const results = bm25.search('PgBouncer pooling', 3);
    expect(results.length).toBeGreaterThan(0);
    expect(results[0].id).toBe('doc-1');
  });

  it('merges dense and sparse rankings with reciprocal rank fusion', () => {
    const dense = [
      { id: 'doc-A', distance: 0.1, score: 0.9 },
      { id: 'doc-B', distance: 0.2, score: 0.8 },
      { id: 'doc-C', distance: 0.3, score: 0.7 },
    ];

    const sparse = [
      { id: 'doc-B', score: 4.5 },
      { id: 'doc-D', score: 3.2 },
      { id: 'doc-A', score: 2.1 },
    ];

    const hybrid = reciprocalRankFusion(dense, sparse, 4);

    expect(hybrid.length).toBe(4);
    // doc-B appeared high in both dense and sparse, so it should rank first or second
    expect(['doc-A', 'doc-B']).toContain(hybrid[0].id);
    expect(hybrid[0].combinedScore).toBe(1.0); // Normalized top score
  });
});
