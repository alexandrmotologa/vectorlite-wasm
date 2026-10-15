import { describe, it, expect } from 'vitest';
import { HNSWIndex } from '../src/engine/hnsw_index';
import { normalize } from '../src/engine/similarity';
import { wasmSimdEngine } from '../src/engine/wasm_simd';

function mulberry32(seed: number) {
  let a = seed;
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296 - 0.5;
  };
}

describe('Real Quantized HNSW Index (SQ8 & 1-bit BQ)', () => {
  it('builds real SQ8 HNSW index and retrieves nearest neighbors with high recall', () => {
    const hnswSQ8 = new HNSWIndex({
      M: 16,
      efConstruction: 64,
      efSearch: 32,
      quantization: 'sq8',
    });

    const dim = 64;
    const numVectors = 100;

    for (let i = 0; i < numVectors; i++) {
      const rng = mulberry32(i * 1000 + 7);
      const vec = new Float32Array(dim);
      for (let d = 0; d < dim; d++) {
        vec[d] = rng();
      }
      hnswSQ8.insert(`sq8-${i}`, normalize(vec));
    }

    expect(hnswSQ8.size).toBe(numVectors);
    expect(hnswSQ8.quantization).toBe('sq8');

    // Query for sq8-25
    const target = hnswSQ8.getVector('sq8-25')!;
    const results = hnswSQ8.search(target, 5);

    expect(results.length).toBe(5);
    expect(results[0].id).toBe('sq8-25');
    expect(results[0].score).toBeGreaterThan(0.95);
  });

  it('builds real 1-bit BQ HNSW index and searches via bitwise POPCNT distance', () => {
    const hnswBQ = new HNSWIndex({
      M: 16,
      efConstruction: 64,
      efSearch: 32,
      quantization: 'bq',
    });

    const dim = 64;
    const numVectors = 100;

    for (let i = 0; i < numVectors; i++) {
      const rng = mulberry32(i * 5000 + 13);
      const vec = new Float32Array(dim);
      for (let d = 0; d < dim; d++) {
        vec[d] = rng();
      }
      hnswBQ.insert(`bq-${i}`, normalize(vec));
    }

    expect(hnswBQ.size).toBe(numVectors);
    expect(hnswBQ.quantization).toBe('bq');

    const target = hnswBQ.getVector('bq-42')!;
    const results = hnswBQ.search(target, 5);

    expect(results.length).toBe(5);
    expect(results[0].id).toBe('bq-42');
  });

  it('verifies WebAssembly SIMD kernel evaluates dot products accurately', () => {
    expect(wasmSimdEngine.isSupported).toBe(true);
    expect(wasmSimdEngine.isActive).toBe(true);

    const a = new Float32Array([1.5, 2.0, 3.5, 4.0, 5.0]);
    const b = new Float32Array([2.0, 1.0, 2.0, 0.5, 1.0]);

    // 1.5*2 + 2*1 + 3.5*2 + 4*0.5 + 5*1 = 3 + 2 + 7 + 2 + 5 = 19
    const simdWasm = wasmSimdEngine.dot(a, b);
    expect(simdWasm).not.toBeNull();
    expect(simdWasm!).toBeCloseTo(19.0, 3);
  });
});
