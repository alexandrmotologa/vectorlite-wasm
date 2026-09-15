# VectorLite-Wasm: API Reference

VectorLite-Wasm provides standalone, zero-dependency TypeScript modules for client-side vector search, quantization, hybrid retrieval, syntax-aware chunking, and local grounded RAG.

---

## 1. `HNSWIndex`

Hierarchical Navigable Small World in-memory graph index with predicate filtering.

```typescript
import { HNSWIndex } from 'vectorlite-wasm/engine/hnsw_index';

const index = new HNSWIndex({
  M: 16,               // Max outgoing edges per node on layer > 0 (default: 16)
  M0: 32,              // Max edges on base layer 0 (default: 32)
  efConstruction: 64,  // Search beam width during index build (default: 64)
  efSearch: 32,        // Search beam width during query (default: 32)
  metric: 'cosine',    // Distance metric: 'cosine' | 'dot' | 'euclidean'
});
```

### Methods

#### `insert(id: string, vector: Float32Array): void`
Inserts a high-dimensional vector into the graph. Vectors are normalized to unit length when using `'cosine'` metric.

#### `search(query: Float32Array, topK: number, efSearch?: number, filterFn?: (id: string) => boolean): SearchResult[]`
Performs approximate nearest neighbor search using beam routing. Optionally accepts a predicate function `filterFn` to scope matches to specific documents or metadata.

```typescript
// Search only within 'systems_architecture.md'
const results = index.search(queryVector, 5, (id) => chunkMap.get(id)?.documentName === 'systems_architecture.md');
```

#### `bruteForceSearch(query: Float32Array, topK: number, filterFn?: (id: string) => boolean): SearchResult[]`
Executes an exact linear scan across all stored vectors. Useful for computing ground truth recall in benchmarks.

#### `serialize(): SerializedHNSW`
Serializes the graph topology and vector values into a JSON-compatible object.

#### `HNSWIndex.deserialize(data: SerializedHNSW): HNSWIndex`
Restores an in-memory index from a serialized snapshot.

---

## 2. `BM25Index`

In-memory Okapi BM25 ranking algorithm for sparse lexical keyword retrieval with predicate filtering.

```typescript
import { BM25Index } from 'vectorlite-wasm/engine/bm25';

const bm25 = new BM25Index(1.2, 0.75); // k1 = 1.2, b = 0.75

// Add documents
bm25.addDocument('doc-1', 'PostgreSQL connection pooling with PgBouncer.');
bm25.addDocument('doc-2', 'Redis cache clusters and distributed locks.');

// Search exact terms with optional filter
const results = bm25.search('PgBouncer pooling', 5, (id) => id.startsWith('doc'));
```

---

## 3. `reciprocalRankFusion`

Combines dense vector search results and sparse lexical results into a unified ranking using Reciprocal Rank Fusion.

```typescript
import { reciprocalRankFusion } from 'vectorlite-wasm/engine/hybrid';

const hybridResults = reciprocalRankFusion(denseResults, sparseResults, 10, {
  k: 60,               // RRF constant (default: 60)
  denseWeight: 1.0,    // Dense multiplier (default: 1.0)
  sparseWeight: 0.8,   // BM25 multiplier (default: 0.8)
});
```

---

## 4. `CodeAwareChunker` & `CodeChunker`

Syntax-aware code splitter preserving functions, classes, interfaces, and structs intact across Python, TypeScript, Rust, Go, JavaScript, and JSON.

```typescript
import { CodeChunker } from 'vectorlite-wasm/engine/code_chunker';

// Check if file is supported code
const isCode = CodeChunker.isSupportedCodeFile('cluster_node.py'); // true

// Split preserving natural block boundaries
const chunks = CodeChunker.split(sourceCode, 'cluster_node.py', {
  maxChunkLines: 45,
  overlapLines: 8,
});
```

---

## 5. `ScalarQuantizer` (SQ8) & `BinaryQuantizer` (BQ)

Memory compaction utilities reducing vector RAM usage by up to 96.8%.

```typescript
import { ScalarQuantizer, BinaryQuantizer } from 'vectorlite-wasm/engine/quantization';

// 8-bit Int8 Scalar Quantization (75% RAM reduction)
const sq8 = ScalarQuantizer.quantize(floatVector);
const restored = ScalarQuantizer.dequantize(sq8);
const simSQ8 = ScalarQuantizer.dotProductInt8(sq8_a, sq8_b);

// 1-bit Binary Quantization (96.8% RAM reduction)
const bq = BinaryQuantizer.quantize(floatVector);
const hammingDist = BinaryQuantizer.hammingDistance(bq_a, bq_b);
const bqSimilarity = BinaryQuantizer.similarity(bq_a, bq_b);
```

---

## 6. `LocalRAGSynthesizer`

In-browser grounded answer extraction generating natural language responses with anchored citations `[1]`, `[2]`.

```typescript
import { LocalRAGSynthesizer } from 'vectorlite-wasm/engine/rag_synthesizer';

const answer = LocalRAGSynthesizer.synthesize(userQuery, retrievedChunks);

console.log(answer.answer);
// "HNSW builds a multi-layer graph [1] using skip-list hierarchy [2]..."
console.log(answer.citations);
// [{ id: 1, chunkId: 'c-1', documentName: 'systems_architecture.md', snippet: '...' }]
```

---

## 7. `PCAProjector` (2D & 3D Orbital Projection)

Fast power-iteration Principal Component Analysis projecting 384-dimensional vector clusters to 2D and 3D coordinate spaces.

```typescript
import { PCAProjector } from 'vectorlite-wasm/engine/pca';

const pca = new PCAProjector();

// 2D Projection
const points2D = pca.fitTransform(vectorList, 25);

// 3D Orbital Projection
const points3D = pca.project3D(vectorList, 25);
// Returns: Array<{ id: string; x: number; y: number; z: number }>
```
