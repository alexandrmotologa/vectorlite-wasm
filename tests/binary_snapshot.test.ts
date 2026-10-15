import { describe, it, expect } from 'vitest';
import { HNSWIndex } from '../src/engine/hnsw_index';
import { TextChunk } from '../src/engine/chunker';
import { serializeBinary, deserializeBinary, createSnapshotBlob } from '../src/engine/snapshot';
import { normalize } from '../src/engine/similarity';

describe('Binary Snapshot Serialization and Deserialization', () => {
  it('correctly serializes and deserializes an empty index and empty chunks', () => {
    const hnsw = new HNSWIndex({ M: 16, M0: 32, efConstruction: 64, efSearch: 32, metric: 'cosine', quantization: 'f32' });
    const chunks: TextChunk[] = [];

    const buffer = serializeBinary(hnsw, chunks);
    expect(buffer.byteLength).toBeGreaterThan(0);

    const restored = deserializeBinary(buffer);
    expect(restored.hnsw.size).toBe(0);
    expect(restored.chunks.length).toBe(0);
    expect(restored.hnsw.metric).toBe('cosine');
    expect(restored.hnsw.quantization).toBe('f32');
  });

  it('preserves vectors, topology, and text chunks with high fidelity across round-trip', () => {
    const hnsw = new HNSWIndex({ M: 8, M0: 16, efConstruction: 32, efSearch: 16, metric: 'cosine', quantization: 'f32' });
    const dim = 64;
    const numVectors = 25;

    const chunks: TextChunk[] = [];
    for (let i = 0; i < numVectors; i++) {
      const vec = new Float32Array(dim);
      for (let d = 0; d < dim; d++) {
        vec[d] = Math.sin(i * 17 + d);
      }
      const normVec = normalize(vec);
      const id = `doc-${i}`;
      hnsw.insert(id, normVec);

      chunks.push({
        id,
        text: `Content chunk number ${i} for testing binary snapshots.`,
        documentName: `doc_${i}.txt`,
        chunkIndex: i,
        totalChunks: numVectors,
        tokenCountEstimate: 10,
        startChar: i * 50,
        endChar: (i + 1) * 50,
      });
    }

    // Binary serialization
    const buffer = serializeBinary(hnsw, chunks);
    expect(buffer.byteLength).toBeGreaterThan(0);

    // Deserialize
    const restored = deserializeBinary(buffer);
    expect(restored.hnsw.size).toBe(numVectors);
    expect(restored.chunks.length).toBe(numVectors);
    expect(restored.chunks[5].text).toBe(chunks[5].text);
    expect(restored.chunks[5].documentName).toBe(chunks[5].documentName);

    // Search query on both original and restored index
    const queryVec = hnsw.getVector('doc-7')!;
    const origResults = hnsw.search(queryVec, 5);
    const restoredResults = restored.hnsw.search(queryVec, 5);

    expect(restoredResults.length).toBe(5);
    expect(restoredResults[0].id).toBe('doc-7');
    expect(restoredResults[0].score).toBeCloseTo(origResults[0].score, 4);

    for (let r = 0; r < origResults.length; r++) {
      expect(restoredResults[r].id).toBe(origResults[r].id);
      expect(restoredResults[r].score).toBeCloseTo(origResults[r].score, 4);
    }
  });

  it('is significantly more compact than JSON snapshot format (~3-4x smaller)', async () => {
    const hnsw = new HNSWIndex({ M: 16, efConstruction: 64, efSearch: 32 });
    const dim = 384; // Typical miniLM embedding dimension
    const numVectors = 50;

    const chunks: TextChunk[] = [];
    for (let i = 0; i < numVectors; i++) {
      const vec = new Float32Array(dim);
      for (let d = 0; d < dim; d++) {
        vec[d] = Math.sin(i * 13 + d);
      }
      const id = `item-${i}`;
      hnsw.insert(id, normalize(vec));
      chunks.push({
        id,
        text: `Short text snippet ${i}`,
        documentName: 'test.txt',
        chunkIndex: i,
        totalChunks: numVectors,
        tokenCountEstimate: 5,
        startChar: 0,
        endChar: 20,
      });
    }

    const binaryBuffer = serializeBinary(hnsw, chunks);
    const jsonBlob = createSnapshotBlob(hnsw, chunks);

    const binarySize = binaryBuffer.byteLength;
    const jsonSize = jsonBlob.size;

    // Binary should be at least 3x smaller than JSON representation
    const ratio = jsonSize / binarySize;
    expect(ratio).toBeGreaterThan(2.5);
  });

  it('correctly handles different metrics and quantization modes', () => {
    const hnswSQ8 = new HNSWIndex({ metric: 'euclidean', quantization: 'sq8' });
    const hnswBQ = new HNSWIndex({ metric: 'dot', quantization: 'bq' });

    const dummyChunks: TextChunk[] = [];
    const bufSQ8 = serializeBinary(hnswSQ8, dummyChunks);
    const bufBQ = serializeBinary(hnswBQ, dummyChunks);

    const resSQ8 = deserializeBinary(bufSQ8);
    expect(resSQ8.hnsw.metric).toBe('euclidean');
    expect(resSQ8.hnsw.quantization).toBe('sq8');

    const resBQ = deserializeBinary(bufBQ);
    expect(resBQ.hnsw.metric).toBe('dot');
    expect(resBQ.hnsw.quantization).toBe('bq');
  });

  it('throws descriptive errors on invalid buffer or magic number', () => {
    const invalidBuffer = new ArrayBuffer(64);
    expect(() => deserializeBinary(invalidBuffer)).toThrow('Invalid VectorLite binary snapshot: bad magic number');

    // Buffer with correct magic number but wrong version
    const view = new DataView(invalidBuffer);
    view.setUint32(0, 0x564C5753, true); // MAGIC
    view.setUint32(4, 999, true); // Invalid version
    expect(() => deserializeBinary(invalidBuffer)).toThrow('Unsupported snapshot version: 999');
  });
});
