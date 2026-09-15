# AI Engineering & Local RAG Architecture Manual

## 1. Local Retrieval-Augmented Generation (RAG)
Retrieval-Augmented Generation (RAG) grounds language model outputs in external authoritative documents. Traditional RAG setups require transmitting sensitive documents to remote third-party vector databases and inference APIs, introducing latency, cloud expenses, and privacy exposures. Client-side local RAG executes the entire embedding, vector indexing, and retrieval pipeline directly within the user's browser runtime through WebAssembly and Web Workers.

## 2. In-Browser Embedding Models via ONNX Runtime Web
Using Transformers.js, modern transformer models can be executed in the browser via ONNX Runtime Web.
- **all-MiniLM-L6-v2:** A 6-layer BERT-based model yielding 384-dimensional sentence embeddings. The 8-bit quantized ONNX variant weighs approximately 23MB, downloading in seconds and providing strong semantic representations.
- **bge-small-en-v1.5:** Trained by BAAI, offering state-of-the-art semantic retrieval benchmarks for dense search with 384 dimensions.
- **Web Worker Isolation:** Heavy matrix multiplications and tokenizations run off the UI thread to prevent interface freezing and maintain 60 FPS user interaction.

## 3. Chunking Strategies: Preserving Semantic Context
Effective vector retrieval depends heavily on document preprocessing.
- **Fixed-Length Chunking:** Slices text at arbitrary character or token counts. Often bifurcates sentences, disrupting semantic continuity.
- **Recursive Character Splitting:** Inspects structural markers hierarchically: Markdown headers, paragraph breaks, sentence punctuation, and words.
- **Contextual Chunk Headers:** Prepending document titles or parent section names to each chunk prevents orphaned chunks from losing contextual anchors during embedding.

## 4. Hybrid Search: Dense Vector + Sparse BM25
While dense vector embeddings excel at grasping semantic intent (e.g. mapping "latency spike" to "slow response"), they can fail on exact token lookups such as error codes (`ERR_CONNECTION_REFUSED`), specific variable names (`efConstruction`), or numerical thresholds.
- **Dense HNSW:** Finds conceptual neighbors in continuous vector space.
- **Sparse BM25:** Computes term-frequency inverse document frequency on exact lexical tokens.
- **Reciprocal Rank Fusion (RRF):** Merges dense and sparse rankings into a unified relevance score without needing score calibration across disparate mathematical distributions.

## 5. Client-Side Vector Persistence
In-browser applications utilize IndexedDB for durable local storage:
- **Zero Cloud Leakage:** Text chunks and high-dimensional Float32Array vectors persist locally on the user's device.
- **Index Snapshots:** Storing pre-built HNSW graph connections alongside embeddings allows instantaneous index re-hydration on subsequent page visits without requiring model re-inference.
