# VectorLite-Wasm: Architecture Specification

## 1. System Overview

VectorLite-Wasm is a client-side vector database and semantic retrieval engine designed to execute entirely inside the web browser. The architecture decouples model inference, graph indexing, and user interface rendering across dedicated execution contexts:

1. **Main UI Thread:** Handles document file ingestion, DOM updates, user input debouncing, and 60 FPS Canvas rendering.
2. **Dedicated Web Worker:** Hosts the Transformers.js inference engine and ONNX Runtime WebAssembly binaries to compute 384-dimensional dense embeddings without blocking the browser event loop.
3. **In-Memory Graph Index:** Maintains a Hierarchical Navigable Small World (HNSW) proximity graph and an Okapi BM25 sparse inverted index directly in heap memory.
4. **Persistent Storage Layer:** Uses IndexedDB for durable local persistence of raw text chunks, pre-calculated Float32Array embeddings, and serialized graph topologies.

```
+-------------------------------------------------------------------------+
|                              Browser Runtime                            |
|                                                                         |
|  +---------------------+                     +-----------------------+  |
|  |   Main UI Thread    |                     | Dedicated Web Worker  |  |
|  |                     |  postMessage (Text) |                       |  |
|  |  - React 19 UI      | ------------------> |  - ONNX Runtime Web   |  |
|  |  - 2D Canvas (PCA)  |                     |  - all-MiniLM-L6-v2   |  |
|  |  - IndexedDB Sync   | <------------------ |  - Float32Array Buffs |  |
|  +---------------------+  postMessage (Vec)  +-----------------------+  |
|             |                                                           |
|             v                                                           |
|  +-------------------------------------------------------------------+  |
|  |                       In-Memory Search Layer                      |  |
|  |                                                                   |  |
|  |  +----------------------------+   +----------------------------+  |  |
|  |  | HNSW Proximity Graph       |   | BM25 Inverted Index        |  |  |
|  |  | - Multi-layer skip routing |   | - Term frequency / IDF     |  |  |
|  |  | - M=16, M0=32              |   | - Stopword pruning         |  |  |
|  |  | - 4x loop-unrolled cosine  |   | - Token length tracking    |  |  |
|  |  +----------------------------+   +----------------------------+  |  |
|  |                 \                               /                 |  |
|  |                  \                             /                  |  |
|  |                   v                           v                   |  |
|  |                  +-----------------------------+                  |  |
|  |                  | Reciprocal Rank Fusion (RRF)|                  |  |
|  |                  +-----------------------------+                  |  |
|  +-------------------------------------------------------------------+  |
+-------------------------------------------------------------------------+
```

## 2. HNSW Proximity Graph Mechanics

VectorLite implements an in-memory Hierarchical Navigable Small World (HNSW) structure based on the Malkov-Yashunin algorithm.

### Multi-Layer Hierarchy

Each vector inserted into the graph is assigned a maximum layer $l$ using an exponential decay distribution:

$$l = \lfloor -\ln(\text{uniform}(0, 1)) \cdot m_L \rfloor$$

where $m_L = \frac{1}{\ln(M)}$. This produces a hierarchical structure where higher layers contain sparse, long-range skip links and layer 0 contains dense local connectivity.

### Graph Parameters

- **$M$ (default: 16):** Maximum outgoing connections per node on layers $l > 0$.
- **$M_0$ (default: 32):** Maximum connections on the base layer (layer 0) to ensure high clustering and graph connectivity.
- **$efConstruction$ (default: 64):** Candidate search beam width during node insertion. Controls graph build quality and recall accuracy.
- **$efSearch$ (default: 32):** Candidate beam width during nearest-neighbor queries.

### Neighbor Pruning

When a node's neighbor list exceeds $M$ (or $M_0$ on layer 0), connections are pruned by sorting candidates by ascending distance and retaining the closest $M$ elements. This prevents high-degree hubs from degrading logarithmic traversal times.

## 3. Vector Arithmetic & Memory Optimization

High-dimensional vector distance calculations (384 dimensions for `all-MiniLM-L6-v2`) represent the primary compute cost during nearest neighbor search.

### Loop-Unrolled Dot Product

To maximize CPU cache utilization and facilitate compiler auto-vectorization, dot product calculations use 4-step loop unrolling:

```typescript
export function dotProduct(a: Float32Array, b: Float32Array): number {
  const len = a.length;
  let sum = 0;
  let i = 0;
  const limit = len - 3;

  for (; i < limit; i += 4) {
    sum += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
  }

  for (; i < len; i++) {
    sum += a[i] * b[i];
  }

  return sum;
}
```

Because all embeddings are normalized to unit length ($L_2 = 1.0$) upon generation, cosine similarity simplifies directly to the inner dot product:

$$\text{CosineSimilarity}(u, v) = u \cdot v$$

Cosine distance is then computed as $1.0 - (u \cdot v)$.

## 4. Hybrid Search with Reciprocal Rank Fusion

To address cases where dense embeddings miss exact lexical tokens (such as configuration parameters or function names), VectorLite runs parallel dense and sparse retrieval passes.

1. **Dense Retrieval:** HNSW traverses the graph and returns the top $2K$ nearest semantic neighbors with similarity scores.
2. **Sparse Retrieval:** Okapi BM25 tokenizes the query, applies Robertson-Spärck Jones IDF scoring against inverted term postings, and returns the top $2K$ keyword matches.
3. **Rank Fusion:** Reciprocal Rank Fusion combines the separate rankings into a unified score:

$$RRF(d) = \frac{w_{\text{dense}}}{60 + \text{rank}_{\text{dense}}(d)} + \frac{w_{\text{sparse}}}{60 + \text{rank}_{\text{sparse}}(d)}$$

Combined scores are normalized against the top result so that match percentages range predictably from 0% to 100%.

## 5. 2D Vector Projection via Online Fast PCA

The interactive cluster canvas visualizes high-dimensional embeddings by projecting them onto the first two principal components ($v_1$ and $v_2$).

1. The centroid mean vector $\bar{x} = \frac{1}{N} \sum x_i$ is computed across all indexed vectors.
2. All vectors are mean-centered: $c_i = x_i - \bar{x}$.
3. Power iteration isolates the dominant eigenvector $v_1$.
4. The centered dataset is deflated by $v_1$ to find the orthogonal second eigenvector $v_2$:
   $$c'_i = c_i - (c_i \cdot v_1) v_1$$
5. Any vector $x$ (including user queries) is projected into 2D coordinates via inner products with the principal axes:
   $$\text{Coord}_{2D}(x) = \left((x - \bar{x}) \cdot v_1, \; (x - \bar{x}) \cdot v_2\right)$$
   Normalized bounds scale coordinates to the HTML5 Canvas dimensions.
