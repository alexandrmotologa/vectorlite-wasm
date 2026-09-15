import { describe, it, expect } from 'vitest';
import {
  quantizeSQ8,
  dequantizeSQ8,
  dotProductSQ8,
  quantizeBQ,
  hammingDistanceBQ,
  cosineSimilarityBQ,
} from '../src/engine/quantization';
import { dotProduct, normalize, cosineSimilarity } from '../src/engine/similarity';

describe('Vector Quantization Engine (SQ8 & BQ)', () => {
  it('compresses Float32 vectors to Int8 SQ8 with >0.98 similarity correlation', () => {
    const dim = 384;
    const a = normalize(new Float32Array(Array.from({ length: dim }, (_, i) => Math.sin(i * 3 + 1))));
    const b = normalize(new Float32Array(Array.from({ length: dim }, (_, i) => Math.cos(i * 5 + 2))));

    const exactDot = dotProduct(a, b);

    const qA = quantizeSQ8(a);
    const qB = quantizeSQ8(b);

    // Verify data storage size: 384 bytes instead of 1,536 bytes
    expect(qA.data.byteLength).toBe(384);

    const approxDot = dotProductSQ8(qA, qB);

    // SQ8 reconstruction
    const dequantA = dequantizeSQ8(qA);
    const reconSim = cosineSimilarity(a, dequantA);

    expect(reconSim).toBeGreaterThan(0.99); // Near-perfect fidelity
    expect(Math.abs(exactDot - approxDot)).toBeLessThan(0.05);
  });

  it('compresses to 1-bit Binary Quantization (BQ) with 32x memory reduction', () => {
    const dim = 384;
    const a = normalize(new Float32Array(Array.from({ length: dim }, (_, i) => Math.sin(i * 1.5))));
    const b = normalize(new Float32Array(Array.from({ length: dim }, (_, i) => Math.sin(i * 1.5 + 0.1)))); // Very close to a
    const c = normalize(new Float32Array(Array.from({ length: dim }, (_, i) => -Math.sin(i * 1.5)))); // Opposite of a

    const bqA = quantizeBQ(a);
    const bqB = quantizeBQ(b);
    const bqC = quantizeBQ(c);

    // 384 / 32 = 12 uint32 words = 48 bytes
    expect(bqA.bits.byteLength).toBe(48);

    const distClose = hammingDistanceBQ(bqA, bqB);
    const distFar = hammingDistanceBQ(bqA, bqC);

    expect(distClose).toBeLessThan(distFar);
    expect(cosineSimilarityBQ(bqA, bqB)).toBeGreaterThan(0.7);
    expect(cosineSimilarityBQ(bqA, bqC)).toBeLessThan(0);
  });
});
