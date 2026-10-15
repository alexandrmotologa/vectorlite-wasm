import { describe, it, expect } from 'vitest';
import { FastTSNE } from '../src/engine/tsne';
import { normalize } from '../src/engine/similarity';

describe('FastTSNE Engine', () => {
  it('handles empty or single vector input gracefully', () => {
    expect(FastTSNE.project([])).toEqual([]);
    const single = [normalize(new Float32Array([1, 0, 0]))];
    const res = FastTSNE.project(single);
    expect(res).toHaveLength(1);
    expect(res[0]).toEqual({ x: 0.5, y: 0.5 });
  });

  it('projects two vectors into normalized 2D bounds', () => {
    const v1 = normalize(new Float32Array([1, 0, 0, 0]));
    const v2 = normalize(new Float32Array([0, 1, 0, 0]));
    const points = FastTSNE.project([v1, v2], { iterations: 20 });
    expect(points).toHaveLength(2);
    for (const pt of points) {
      expect(pt.x).toBeGreaterThanOrEqual(0);
      expect(pt.x).toBeLessThanOrEqual(1);
      expect(pt.y).toBeGreaterThanOrEqual(0);
      expect(pt.y).toBeLessThanOrEqual(1);
    }
  });

  it('projects distinct clusters of high-dimensional vectors', () => {
    // Cluster 1: around [1, 0, 0, 0]
    // Cluster 2: around [0, 0, 1, 0]
    const cluster1 = [
      normalize(new Float32Array([1.0, 0.05, 0.0, 0.0])),
      normalize(new Float32Array([0.95, 0.1, 0.0, 0.0])),
      normalize(new Float32Array([0.98, 0.0, 0.05, 0.0])),
    ];
    const cluster2 = [
      normalize(new Float32Array([0.0, 0.0, 1.0, 0.05])),
      normalize(new Float32Array([0.0, 0.05, 0.95, 0.1])),
      normalize(new Float32Array([0.05, 0.0, 0.98, 0.0])),
    ];
    const all = [...cluster1, ...cluster2];
    const points = FastTSNE.project(all, { perplexity: 5, iterations: 50, learningRate: 80 });

    expect(points).toHaveLength(6);
    for (const pt of points) {
      expect(pt.x).toBeGreaterThanOrEqual(0.04);
      expect(pt.x).toBeLessThanOrEqual(0.96);
      expect(pt.y).toBeGreaterThanOrEqual(0.04);
      expect(pt.y).toBeLessThanOrEqual(0.96);
    }
  });
});
