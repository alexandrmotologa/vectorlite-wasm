import { describe, it, expect } from 'vitest';
import {
  dotProduct,
  normalize,
  cosineSimilarity,
  euclideanDistance,
  vectorDistance,
} from '../src/engine/similarity';

describe('Similarity & Vector Math', () => {
  it('computes dot product accurately with loop unrolling', () => {
    const a = new Float32Array([1, 2, 3, 4, 5, 6, 7]);
    const b = new Float32Array([7, 6, 5, 4, 3, 2, 1]);
    // 1*7 + 2*6 + 3*5 + 4*4 + 5*3 + 6*2 + 7*1 = 7 + 12 + 15 + 16 + 15 + 12 + 7 = 84
    expect(dotProduct(a, b)).toBeCloseTo(84.0, 4);
  });

  it('normalizes vector to unit length', () => {
    const a = new Float32Array([3, 4]); // norm = 5
    const normalized = normalize(a);
    expect(normalized[0]).toBeCloseTo(0.6, 4);
    expect(normalized[1]).toBeCloseTo(0.8, 4);
    expect(dotProduct(normalized, normalized)).toBeCloseTo(1.0, 4);
  });

  it('computes cosine similarity correctly', () => {
    const a = new Float32Array([1, 0, 0]);
    const b = new Float32Array([1, 0, 0]);
    const c = new Float32Array([0, 1, 0]);

    expect(cosineSimilarity(a, b, false)).toBeCloseTo(1.0, 4);
    expect(cosineSimilarity(a, c, false)).toBeCloseTo(0.0, 4);
  });

  it('computes euclidean distance accurately', () => {
    const a = new Float32Array([0, 0, 0]);
    const b = new Float32Array([3, 4, 0]);
    expect(euclideanDistance(a, b)).toBeCloseTo(5.0, 4);
  });

  it('computes distance metrics consistently', () => {
    const a = normalize(new Float32Array([1, 2, 3]));
    const b = normalize(new Float32Array([1, 2, 3]));
    expect(vectorDistance(a, b, 'cosine')).toBeCloseTo(0.0, 4);
  });
});
