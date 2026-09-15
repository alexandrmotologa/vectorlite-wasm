# VectorLite-Wasm: API Reference

VectorLite-Wasm provides standalone, zero-dependency TypeScript modules for client-side vector search and text chunking.

---

## 1. `HNSWIndex`

Hierarchical Navigable Small World in-memory graph index.

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
Inserts a high-dimensional vector into the graph. Vectors should be normalized to unit length when using `'cosine'` metric.

#### `search(query: Float32Array, topK: number, efSearch?: number): SearchResult[]`
Performs approximate nearest neighbor search using beam routing.
Returns array of:
```typescript
interface SearchResult {
  id: string;
  distance: number;
  score: number; // Normalized similarity score [0.0, 1.0]
}
```

#### `bruteForceSearch(query: Float32Array, topK: number): SearchResult[]`
Executes an exact linear scan across all stored vectors. Useful for computing ground truth recall in benchmarks.

#### `serialize(): SerializedHNSW`
Serializes the graph topology and vector values into a JSON-compatible object.

#### `HNSWIndex.deserialize(data: SerializedHNSW): HNSWIndex`
Restores an in-memory index from a serialized snapshot.

---

## 2. `BM25Index`

In-memory Okapi BM25 ranking algorithm for sparse lexical keyword retrieval.

```typescript
import { BM25Index } from 'vectorlite-wasm/engine/bm25';

const bm25 = new BM25Index(1.2, 0.75); // k1 = 1.2, b = 0.75

// Add documents
bm25.addDocument('doc-1', 'PostgreSQL connection pooling with PgBouncer.');
bm25.addDocument('doc-2', 'Redis cache clusters and distributed locks.');

// Search exact terms
const results = bm25.search('PgBouncer pooling', 5);
// Returns: [{ id: 'doc-1', score: 2.84 }]
```

---

## 3. `reciprocalRankFusion`

Combines dense vector search results and sparse lexical results into a unified ranking.

```typescript
import { reciprocalRankFusion } from 'vectorlite-wasm/engine/hybrid';

const hybridResults = reciprocalRankFusion(denseResults, sparseResults, 10, {
  k: 60,               // RRF constant (default: 60)
  denseWeight: 1.0,    // Dense multiplier (default: 1.0)
  sparseWeight: 0.8,   // BM25 multiplier (default: 0.8)
});
```

---

## 4. `RecursiveChunker`

Hierarchical text splitter preserving Markdown headers, paragraph boundaries, and sentences.

```typescript
import { RecursiveChunker } from 'vectorlite-wasm/engine/chunker';

const chunker = new RecursiveChunker({
  chunkSize: 500,     // Max characters per chunk (default: 600)
  chunkOverlap: 100,  // Character overlap between chunks (default: 120)
});

const chunks = chunker.split(documentText, 'architecture_guide.md');
```

---

## 5. `FastPCA`

Online Principal Component Analysis using power iteration for projecting embeddings into 2D coordinates.

```typescript
import { FastPCA } from 'vectorlite-wasm/engine/pca';

// Fit model on dataset vectors
const pcaModel = FastPCA.fit(vectors, 15); // 15 power iterations

// Project vector into [0, 1] 2D space
const point2D = FastPCA.project(queryVector, pcaModel);
// { x: 0.45, y: 0.72 }
```

---

## 6. Snapshot Export & Import (`.vlite`)

```typescript
import { createSnapshotBlob, downloadSnapshot, parseSnapshot } from 'vectorlite-wasm/engine/snapshot';

// Trigger browser file download
downloadSnapshot(hnsw, chunks, 'knowledge-base.vlite');

// Parse uploaded file
const { hnsw, chunks } = await parseSnapshot(file);
```
