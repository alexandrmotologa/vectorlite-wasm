import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { Header } from './components/Header';
import { DocumentDropzone } from './components/DocumentDropzone';
import { SearchBar, SearchMode } from './components/SearchBar';
import { SearchResults } from './components/SearchResults';
import { VectorClusterCanvas } from './components/VectorClusterCanvas';
import { ChunkInspectorModal } from './components/ChunkInspectorModal';
import { BenchmarkStats } from './components/BenchmarkStats';
import { SnapshotManager } from './components/SnapshotManager';
import { HNSWIndex } from './engine/hnsw_index';
import { BM25Index } from './engine/bm25';
import { RecursiveChunker, TextChunk } from './engine/chunker';
import { VectorStorage } from './engine/storage';
import { reciprocalRankFusion, HybridSearchResult } from './engine/hybrid';
import { WorkerOutMessage, WorkerInMessage } from './workers/embedding.worker';
import { ShieldCheck, Cpu, Compass } from 'lucide-react';

export const App: React.FC = () => {
  // Engines
  const hnswRef = useRef<HNSWIndex>(new HNSWIndex({ M: 16, efConstruction: 64, efSearch: 32 }));
  const bm25Ref = useRef<BM25Index>(new BM25Index());
  const storageRef = useRef<VectorStorage>(new VectorStorage());
  const workerRef = useRef<Worker | null>(null);

  // States
  const [modelStatus, setModelStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  const [modelName] = useState('Xenova/all-MiniLM-L6-v2');
  const [downloadProgress, setDownloadProgress] = useState<{ file: string; progress: number } | null>(null);

  const [chunks, setChunks] = useState<TextChunk[]>([]);
  const [vectors, setVectors] = useState<Array<{ id: string; vector: Float32Array }>>([]);
  const [searchResults, setSearchResults] = useState<HybridSearchResult[]>([]);
  const [currentQuery, setCurrentQuery] = useState('');
  const [queryVector, setQueryVector] = useState<Float32Array | null>(null);
  const [queryLatencyMs, setQueryLatencyMs] = useState<number | null>(null);

  const [isProcessing, setIsProcessing] = useState(false);
  const [processStatus, setProcessStatus] = useState('');
  const [batchProgress, setBatchProgress] = useState<{ current: number; total: number } | null>(null);
  const [inspectingItem, setInspectingItem] = useState<{ chunk: TextChunk; result?: HybridSearchResult } | null>(null);

  // Pending resolver for Web Worker async request-response handling
  const pendingBatchResolver = useRef<((res: Array<{ id: string; vector: Float32Array }>) => void) | null>(null);
  const pendingQueryResolver = useRef<((vec: Float32Array) => void) | null>(null);

  // Map chunk IDs to TextChunk objects
  const chunksMap = useMemo(() => {
    const map = new Map<string, TextChunk>();
    for (const c of chunks) {
      map.set(c.id, c);
    }
    return map;
  }, [chunks]);

  const totalTokens = useMemo(() => {
    return chunks.reduce((acc, c) => acc + c.tokenCountEstimate, 0);
  }, [chunks]);

  // Initialize Web Worker and rehydrate from IndexedDB on startup
  useEffect(() => {
    const worker = new Worker(new URL('./workers/embedding.worker.ts', import.meta.url), {
      type: 'module',
    });
    workerRef.current = worker;

    worker.onmessage = (e: MessageEvent<WorkerOutMessage>) => {
      const msg = e.data;

      if (msg.type === 'STATUS') {
        setModelStatus(msg.status);
      } else if (msg.type === 'PROGRESS') {
        setDownloadProgress({ file: msg.file, progress: msg.progress });
      } else if (msg.type === 'BATCH_PROGRESS') {
        setBatchProgress({ current: msg.current, total: msg.total });
      } else if (msg.type === 'BATCH_RESULT') {
        if (pendingBatchResolver.current) {
          pendingBatchResolver.current(msg.results);
          pendingBatchResolver.current = null;
        }
      } else if (msg.type === 'QUERY_RESULT') {
        if (pendingQueryResolver.current) {
          pendingQueryResolver.current(msg.vector);
          pendingQueryResolver.current = null;
        }
      } else if (msg.type === 'ERROR') {
        setModelStatus('error');
        setProcessStatus(`Error: ${msg.error}`);
        setIsProcessing(false);
      }
    };

    // Send init to worker
    worker.postMessage({ type: 'INIT', modelName } as WorkerInMessage);

    // Rehydrate from IndexedDB
    const rehydrate = async () => {
      try {
        const storedChunks = await storageRef.current.getAllChunks();
        const storedVectors = await storageRef.current.getAllVectors();

        if (storedChunks.length > 0 && storedVectors.length > 0) {
          for (const item of storedVectors) {
            hnswRef.current.insert(item.id, item.vector);
          }
          for (const c of storedChunks) {
            bm25Ref.current.addDocument(c.id, c.text);
          }
          setChunks(storedChunks);
          setVectors(storedVectors);
        }
      } catch (err) {
        console.warn('Storage rehydration skipped or failed:', err);
      }
    };

    rehydrate();

    return () => {
      worker.terminate();
    };
  }, [modelName]);

  // Handle adding new document
  const handleAddDocument = async (filename: string, content: string) => {
    setIsProcessing(true);
    setProcessStatus('Chunking document...');

    try {
      const chunker = new RecursiveChunker({ chunkSize: 550, chunkOverlap: 100 });
      const newChunks = chunker.split(content, filename);

      if (newChunks.length === 0) {
        throw new Error('No chunks generated from document.');
      }

      setProcessStatus(`Embedding ${newChunks.length} chunks via ONNX runtime...`);
      setBatchProgress({ current: 0, total: newChunks.length });

      // Request embeddings from worker
      const worker = workerRef.current;
      if (!worker) throw new Error('Web Worker not ready.');

      const embeddingResults = await new Promise<Array<{ id: string; vector: Float32Array }>>((resolve) => {
        pendingBatchResolver.current = resolve;
        worker.postMessage({
          type: 'EMBED_BATCH',
          items: newChunks.map((c) => ({ id: c.id, text: c.text })),
        } as WorkerInMessage);
      });

      setProcessStatus('Indexing vectors into HNSW graph...');

      // Insert into HNSW & BM25
      for (const item of embeddingResults) {
        hnswRef.current.insert(item.id, item.vector);
      }
      for (const c of newChunks) {
        bm25Ref.current.addDocument(c.id, c.text);
      }

      // Persist in IndexedDB
      await storageRef.current.saveChunks(newChunks);
      await storageRef.current.saveVectors(embeddingResults);
      await storageRef.current.saveDocument({
        name: filename,
        sizeBytes: content.length,
        chunkCount: newChunks.length,
        createdAt: Date.now(),
      });

      setChunks((prev) => [...prev, ...newChunks]);
      setVectors((prev) => [...prev, ...embeddingResults]);

      setProcessStatus('Ready');
    } finally {
      setIsProcessing(false);
      setBatchProgress(null);
    }
  };

  // Execute search query
  const handleSearch = useCallback(
    async (query: string, mode: SearchMode, topK: number) => {
      if (chunks.length === 0) return;
      setCurrentQuery(query);
      const t0 = performance.now();

      if (mode === 'sparse') {
        // Pure BM25
        const bm25Res = bm25Ref.current.search(query, topK);
        const hybridResults = bm25Res.map((r) => ({
          id: r.id,
          combinedScore: r.score / (bm25Res[0]?.score || 1.0),
          sparseScore: r.score,
        }));
        const t1 = performance.now();
        setSearchResults(hybridResults);
        setQueryLatencyMs(t1 - t0);
        return;
      }

      // If dense or hybrid, generate query embedding via worker
      const worker = workerRef.current;
      if (!worker) return;

      const qVec = await new Promise<Float32Array>((resolve) => {
        pendingQueryResolver.current = resolve;
        worker.postMessage({
          type: 'EMBED_QUERY',
          query,
        } as WorkerInMessage);
      });

      setQueryVector(qVec);

      if (mode === 'dense') {
        const denseRes = hnswRef.current.search(qVec, topK);
        const hybridResults = denseRes.map((r) => ({
          id: r.id,
          combinedScore: r.score,
          denseScore: r.score,
        }));
        const t1 = performance.now();
        setSearchResults(hybridResults);
        setQueryLatencyMs(t1 - t0);
      } else {
        // Hybrid (Dense + Sparse RRF)
        const denseRes = hnswRef.current.search(qVec, topK * 2);
        const sparseRes = bm25Ref.current.search(query, topK * 2);
        const fused = reciprocalRankFusion(denseRes, sparseRes, topK);
        const t1 = performance.now();
        setSearchResults(fused);
        setQueryLatencyMs(t1 - t0);
      }
    },
    [chunks]
  );

  // Restore snapshot
  const handleImportSnapshot = (restoredHnsw: HNSWIndex, restoredChunks: TextChunk[]) => {
    hnswRef.current = restoredHnsw;
    bm25Ref.current.clear();
    for (const c of restoredChunks) {
      bm25Ref.current.addDocument(c.id, c.text);
    }
    const nodes = restoredHnsw.getAllNodes();
    setChunks(restoredChunks);
    setVectors(nodes.map((n) => ({ id: n.id, vector: n.vector })));
    setSearchResults([]);
  };

  // Clear all data
  const handleClearAll = async () => {
    if (!window.confirm('Reset and clear all indexed documents and vectors?')) return;
    await storageRef.current.clearAll();
    hnswRef.current.clear();
    bm25Ref.current.clear();
    setChunks([]);
    setVectors([]);
    setSearchResults([]);
    setQueryVector(null);
    setQueryLatencyMs(null);
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 bg-grid-pattern selection:bg-brand-500/20 selection:text-brand-300">
      <Header
        modelStatus={modelStatus}
        modelName={modelName}
        vectorCount={vectors.length}
        totalTokens={totalTokens}
        downloadProgress={downloadProgress}
      />

      <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8 w-full">
        {/* Top Hero Pitch / Architecture Badge */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 p-5 rounded-2xl bg-gradient-to-r from-slate-900 via-slate-900 to-slate-950 border border-slate-800">
          <div className="space-y-1">
            <div className="flex items-center space-x-2">
              <span className="flex h-2 w-2 relative">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-brand-500"></span>
              </span>
              <span className="text-xs font-mono font-semibold text-brand-400 uppercase tracking-wider">
                100% Client-Side WebAssembly Pipeline
              </span>
            </div>
            <p className="text-sm text-slate-300">
              Zero cloud latency, zero external API costs, zero data leakage. Embeddings, HNSW graph indexing, and hybrid BM25 retrieval run locally in your browser.
            </p>
          </div>

          <div className="flex items-center space-x-2 shrink-0 text-xs font-mono">
            <span className="px-3 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-slate-300 flex items-center space-x-1.5">
              <ShieldCheck className="w-3.5 h-3.5 text-brand-400" />
              <span>Offline Ready</span>
            </span>
            <span className="px-3 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-slate-300 flex items-center space-x-1.5">
              <Cpu className="w-3.5 h-3.5 text-indigo-400" />
              <span>SIMD Math</span>
            </span>
          </div>
        </div>

        {/* Ingestion & Search Panels */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Column: Dropzone & Snapshots (5 cols) */}
          <div className="lg:col-span-5 space-y-6">
            <DocumentDropzone
              onAddDocument={handleAddDocument}
              isProcessing={isProcessing}
              processStatus={processStatus}
              batchProgress={batchProgress}
            />

            <SnapshotManager
              hnsw={hnswRef.current}
              chunks={chunks}
              onImportSnapshot={handleImportSnapshot}
              onClearAll={handleClearAll}
            />
          </div>

          {/* Right Column: Search Bar & Results (7 cols) */}
          <div className="lg:col-span-7 space-y-6">
            <SearchBar
              onSearch={handleSearch}
              isSearching={isProcessing}
              queryLatencyMs={queryLatencyMs}
              totalVectors={vectors.length}
            />

            <SearchResults
              results={searchResults}
              chunksMap={chunksMap}
              onInspectChunk={(chunk, res) => setInspectingItem({ chunk, result: res })}
              query={currentQuery}
            />

            {/* Empty State when no results and no documents */}
            {searchResults.length === 0 && (
              <div className="p-8 rounded-2xl bg-slate-900/40 border border-dashed border-slate-800 text-center space-y-3">
                <Compass className="w-8 h-8 text-slate-600 mx-auto" />
                <h3 className="text-sm font-semibold text-slate-400">
                  {vectors.length === 0
                    ? 'No documents loaded in local vector engine'
                    : 'Enter a search query above to see ranked matches'}
                </h3>
                <p className="text-xs text-slate-500 max-w-md mx-auto">
                  {vectors.length === 0
                    ? 'Click "Load Systems Architecture" on the left to test semantic retrieval with 6 technical chunks in seconds.'
                    : 'Query vectors are generated on the fly via Web Worker and matched against the in-memory HNSW index.'}
                </p>
              </div>
            )}
          </div>
        </div>

        {/* 2D Vector Cluster Visualizer */}
        <VectorClusterCanvas
          vectors={vectors}
          chunksMap={chunksMap}
          queryVector={queryVector}
          searchResults={searchResults}
          onSelectChunk={(chunk) => setInspectingItem({ chunk })}
        />

        {/* Engine Performance & Micro-Benchmarks */}
        <BenchmarkStats
          hnsw={hnswRef.current}
          lastLatencyMs={queryLatencyMs}
          totalTokens={totalTokens}
        />
      </main>

      {/* Chunk & Vector Inspector Modal */}
      {inspectingItem && (
        <ChunkInspectorModal
          chunk={inspectingItem.chunk}
          result={inspectingItem.result}
          vector={hnswRef.current.getVector(inspectingItem.chunk.id) || undefined}
          onClose={() => setInspectingItem(null)}
        />
      )}

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 py-6 mt-12 text-xs text-slate-500 font-mono">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center space-x-2">
            <span>VectorLite-Wasm</span>
            <span>•</span>
            <span>Client-Side ONNX WebAssembly & HNSW</span>
          </div>
          <div>MIT License • Built by Alexandr Motologa</div>
        </div>
      </footer>
    </div>
  );
};
