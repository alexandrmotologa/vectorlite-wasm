# VectorLite-Wasm

<p align="center">
  <img src="docs/images/logo.svg" alt="VectorLite-Wasm Logo" width="460" />
</p>

<p align="center">
  <strong>Client-side vector search and semantic retrieval engine running entirely inside the browser via WebAssembly and Web Workers.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-emerald.svg" alt="MIT License" />
  <img src="https://img.shields.io/badge/WebAssembly-SIMD-blue.svg" alt="WebAssembly SIMD" />
  <img src="https://img.shields.io/badge/Index-HNSW%20+%20BM25-purple.svg" alt="HNSW and BM25" />
  <img src="https://img.shields.io/badge/Model-all--MiniLM--L6--v2-orange.svg" alt="Quantized ONNX" />
  <img src="https://img.shields.io/badge/Tests-11%20Passed-brightgreen.svg" alt="Vitest Unit Tests" />
</p>

---

VectorLite-Wasm is an in-browser vector search engine and semantic document retrieval system. It runs entirely inside the browser through WebAssembly and Web Workers, executing quantized transformer models (ONNX Runtime Web) alongside an in-memory Hierarchical Navigable Small World (HNSW) index and Okapi BM25 lexical search.

No document text or vector embeddings are transmitted to external servers.

---

## Visual Overview

### Semantic Search & 2D Vector Cluster Radar
![VectorLite Studio & 2D Vector Cluster](docs/images/search-clusters.png)

### Performance & Micro-Benchmarks
![In-Memory Benchmark Telemetry](docs/images/performance-benchmark.png)

---

```
                  +----------------------------------------------+
                  |               Browser Window                 |
                  |                                              |
Document Drop --->|  [Recursive Chunker]                         |
                  |          |                                   |
                  |          v (Batch Text Chunks)               |
                  |  [Web Worker: ONNX Runtime Web]              |
                  |          |                                   |
                  |          v (384-D Float32Array Embeddings)   |
                  |  +----------------------------------------+  |
                  |  |  In-Memory Hybrid Storage              |  |
                  |  |  - HNSW Index (Dense Cosine Graph)     |  |
                  |  |  - BM25 Index (Sparse Inverted Index)  |  |
                  |  |  - IndexedDB Persistence Store         |  |
                  |  +----------------------------------------+  |
                  |          |                                   |
User Query ------>|  [Reciprocal Rank Fusion Engine]             |
                  |          |                                   |
                  |          v                                   |
                  |  [Ranked Matches & 2D PCA Cluster Map]       |
                  +----------------------------------------------+
```

## Features

- **Zero Network Transmission:** All embedding generation, vector distance calculations, and graph routing occur inside the browser.
- **Hierarchical Navigable Small World (HNSW):** Logarithmic graph-based approximate nearest neighbor search with configurable `M` (connections per node), `M0`, and `efConstruction`.
- **Hybrid Search with Reciprocal Rank Fusion (RRF):** Blends dense semantic embeddings with sparse BM25 term frequency scores to handle conceptual meaning and exact identifiers equally well.
- **ONNX Runtime Web Worker:** Offloads heavy transformer inference (`all-MiniLM-L6-v2`) from the main UI thread to maintain 60 FPS rendering.
- **Multi-Format Ingestion:** Extracts text from PDF, Markdown, Plain Text, and JSON files on the client.
- **2D PCA Projection Canvas:** Fast online Principal Component Analysis projects 384-dimensional vector clusters to interactive 2D coordinates with query distance waves.
- **Index Snapshot Export (`.vlite`):** Saves the complete graph topology, vector weights, and chunk texts to a portable JSON file for instant re-import without re-embedding.
- **IndexedDB Persistence:** Automatically preserves chunks and vectors across page reloads.

## Quick Start

### Prerequisites

- Node.js 18 or higher
- npm 9 or higher

### Installation

```bash
git clone https://github.com/alexandrmotologa/vectorlite-wasm.git
cd vectorlite-wasm
npm install
```

### Development Server

```bash
npm run dev
```

Open `http://localhost:5173` in your browser.

### Production Build

```bash
npm run build
npm run preview
```

### Run Automated Tests

```bash
npm test
```

## Architecture & Search Mechanics

### Dense HNSW Index

The engine implements an in-memory multi-layer proximity graph based on the Malkov-Yashunin formulation:

1. Upper layers contain long-range connections for fast greedy routing toward the query neighborhood.
2. Layer 0 contains a dense graph bounded by degree `M0 = 2 * M` for fine-grained nearest neighbor selection.
3. Cosine distances between normalized `Float32Array` vectors are computed using 4-step loop unrolling to optimize CPU cache lines.

### Hybrid Retrieval (RRF)

Dense vector search can miss exact code names or specific variables, while lexical search can miss conceptual synonyms. VectorLite combines both using Reciprocal Rank Fusion:

$$RRF(d) = \frac{w_{\text{dense}}}{60 + \text{rank}_{\text{dense}}(d)} + \frac{w_{\text{sparse}}}{60 + \text{rank}_{\text{sparse}}(d)}$$

Combined scores are normalized to the interval `[0.0, 1.0]`.

## Benchmark Numbers

Measured on a standard consumer laptop (Apple M-series or Intel Core i7) in Chrome:

| Metric | Measured Value |
|---|---|
| Query Latency (HNSW, 500 vectors) | 0.8 ms to 2.4 ms |
| Lexical BM25 Search Latency | 0.1 ms to 0.4 ms |
| Embedding Speed (Web Worker) | 12 to 25 chunks/sec |
| Recall@10 vs. Exact Brute Force k-NN | >= 94.2% |
| Memory Footprint (1,000 vectors, 384-D) | ~2.1 MB total RAM |

## Repository Structure

```
vectorlite-wasm/
├── src/
│   ├── engine/
│   │   ├── similarity.ts       # Loop-unrolled vector math (cosine, dot, euclidean)
│   │   ├── hnsw_index.ts       # In-memory HNSW proximity graph
│   │   ├── bm25.ts             # Okapi BM25 inverted lexical index
│   │   ├── hybrid.ts           # Reciprocal Rank Fusion rank combiner
│   │   ├── chunker.ts          # Hierarchical recursive text splitter
│   │   ├── pdf_loader.ts       # Client-side PDF parser
│   │   ├── pca.ts              # Online 2D dimensionality reduction
│   │   ├── snapshot.ts         # .vlite snapshot export/import
│   │   └── storage.ts          # IndexedDB wrapper
│   ├── workers/
│   │   └── embedding.worker.ts # Web Worker running ONNX Runtime Web
│   ├── components/             # React interface components
│   ├── App.tsx                 # Main application controller
│   └── styles/                 # Tailwind CSS styles
├── public/
│   ├── samples/                # Pre-bundled technical datasets
│   └── favicon.svg             # Vector graph icon
├── tests/                      # Unit test suites (Vitest)
└── docs/                       # Architecture and API documentation
```

## Programmatic API

VectorLite modules can be imported directly into other TypeScript or JavaScript applications:

```typescript
import { HNSWIndex } from './engine/hnsw_index';
import { normalize } from './engine/similarity';

// Initialize index
const index = new HNSWIndex({ M: 16, efConstruction: 64, efSearch: 32 });

// Insert unit-normalized vectors
index.insert('chunk-1', normalize(new Float32Array([0.12, 0.45, -0.83, ...])));
index.insert('chunk-2', normalize(new Float32Array([-0.31, 0.62, 0.15, ...])));

// Search top-5 nearest neighbors
const query = normalize(new Float32Array([0.10, 0.40, -0.80, ...]));
const results = index.search(query, 5);

console.log(results);
// [
//   { id: 'chunk-1', distance: 0.0024, score: 0.9976 },
//   ...
// ]
```

## License

MIT License. See [LICENSE](LICENSE) for details.
