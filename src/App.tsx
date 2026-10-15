import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { Header } from './components/Header';
import { DocumentDropzone } from './components/DocumentDropzone';
import { SearchBar, SearchMode } from './components/SearchBar';
import { SearchResults } from './components/SearchResults';
import { RAGAnswerBox } from './components/RAGAnswerBox';
import { VectorClusterCanvas } from './components/VectorClusterCanvas';
import { ChunkInspectorModal } from './components/ChunkInspectorModal';
import { BenchmarkStats } from './components/BenchmarkStats';
import { SnapshotManager } from './components/SnapshotManager';
import { DocumentLibrary } from './components/DocumentLibrary';
import { HNSWIndex, QuantizationMode } from './engine/hnsw_index';
import { BM25Index } from './engine/bm25';
import { RecursiveChunker, TextChunk } from './engine/chunker';
import { CodeChunker } from './engine/code_chunker';
import { LocalRAGSynthesizer, RAGAnswer } from './engine/rag_synthesizer';
import { VectorStorage } from './engine/storage';
import { reciprocalRankFusion, HybridSearchResult } from './engine/hybrid';
import { WorkerOutMessage, WorkerInMessage } from './workers/embedding.worker';
import { RAGWorkerInMessage, RAGWorkerOutMessage } from './workers/rag.worker';
import { isWasmSimdActive } from './engine/similarity';
import { ShieldCheck, Cpu, Compass } from 'lucide-react';

export const App: React.FC = () => {
  // Engines
  const hnswRef = useRef<HNSWIndex>(new HNSWIndex({ M: 16, efConstruction: 64, efSearch: 32, quantization: 'f32' }));
  const bm25Ref = useRef<BM25Index>(new BM25Index());
  const storageRef = useRef<VectorStorage>(new VectorStorage());
  const workerRef = useRef<Worker | null>(null);
  const ragWorkerRef = useRef<Worker | null>(null);

  // States
  const [modelStatus, setModelStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  const [modelName, setModelName] = useState('Xenova/all-MiniLM-L6-v2');
  const [embeddingDevice, setEmbeddingDevice] = useState<'wasm' | 'webgpu'>('wasm');
  const [hasWebGpu, setHasWebGpu] = useState(false);
  const embeddingDeviceRef = useRef(embeddingDevice);
  embeddingDeviceRef.current = embeddingDevice;
  const [isFallback, setIsFallback] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState<{ file: string; progress: number } | null>(null);

  // Detect WebGPU support on mount
  useEffect(() => {
    if (typeof navigator !== 'undefined' && 'gpu' in navigator && navigator.gpu) {
      setHasWebGpu(true);
    }
  }, []);

  // Theme State
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    return (localStorage.getItem('vlw_theme') as 'dark' | 'light') || 'dark';
  });

  const toggleTheme = () => {
    const next = theme === 'dark' ? 'light' : 'dark';
    setTheme(next);
    localStorage.setItem('vlw_theme', next);
  };

  // RAG States
  const [ragAnswer, setRagAnswer] = useState<RAGAnswer | null>(null);
  const [ragHistory, setRagHistory] = useState<Array<{ role: 'user' | 'assistant'; content: string }>>([]);
  const [ragMode, setRagMode] = useState<'in_browser_llm' | 'ollama' | 'extractive'>('in_browser_llm');
  const [ollamaEndpoint, setOllamaEndpoint] = useState('http://localhost:11434');
  const [ollamaModel, setOllamaModel] = useState('llama3.2');
  const [isGeneratingRAG, setIsGeneratingRAG] = useState(false);
  const [ragGenerationStatus, setRagGenerationStatus] = useState('');

  const [chunks, setChunks] = useState<TextChunk[]>([]);
  const [vectors, setVectors] = useState<Array<{ id: string; vector: Float32Array }>>([]);
  const [searchResults, setSearchResults] = useState<HybridSearchResult[]>([]);
  const [currentQuery, setCurrentQuery] = useState('');
  const currentQueryRef = useRef(currentQuery);
  currentQueryRef.current = currentQuery;
  const [queryVector, setQueryVector] = useState<Float32Array | null>(null);
  const [queryLatencyMs, setQueryLatencyMs] = useState<number | null>(null);

  const [chunkSize, setChunkSize] = useState(550);
  const [chunkOverlap, setChunkOverlap] = useState(100);

  const [isProcessing, setIsProcessing] = useState(false);
  const [processStatus, setProcessStatus] = useState('');
  const [batchProgress, setBatchProgress] = useState<{ current: number; total: number } | null>(null);
  const [inspectingItem, setInspectingItem] = useState<{ chunk: TextChunk; result?: HybridSearchResult } | null>(null);

  // Pending resolver for Web Worker async request-response handling
  const pendingBatchResolver = useRef<((res: Array<{ id: string; vector: Float32Array }>) => void) | null>(null);
  const pendingQueryResolver = useRef<((vec: Float32Array) => void) | null>(null);
  const isSimdAvailable = useMemo(() => isWasmSimdActive(), []);

  // Map chunk IDs to TextChunk objects
  const chunksMap = useMemo(() => {
    const map = new Map<string, TextChunk>();
    for (const c of chunks) {
      map.set(c.id, c);
    }
    return map;
  }, [chunks]);

  const availableDocuments = useMemo(() => {
    const set = new Set<string>();
    for (const c of chunks) {
      set.add(c.documentName);
    }
    return Array.from(set).sort();
  }, [chunks]);

  const documentInfos = useMemo(() => {
    const docMap = new Map<string, { chunkCount: number; tokenCount: number; addedAt: number }>();
    for (const c of chunks) {
      const existing = docMap.get(c.documentName);
      if (existing) {
        existing.chunkCount++;
        existing.tokenCount += c.tokenCountEstimate;
      } else {
        docMap.set(c.documentName, {
          chunkCount: 1,
          tokenCount: c.tokenCountEstimate,
          addedAt: Date.now(),
        });
      }
    }
    return Array.from(docMap.entries()).map(([name, info]) => ({
      name,
      chunkCount: info.chunkCount,
      tokenCount: info.tokenCount,
      addedAt: info.addedAt,
    }));
  }, [chunks]);

  const totalTokens = useMemo(() => {
    return chunks.reduce((acc, c) => acc + c.tokenCountEstimate, 0);
  }, [chunks]);

  // Helper to trigger grounded RAG generation via dedicated RAG worker
  const executeRAG = useCallback(
    (query: string, topChunks: TextChunk[], mode = ragMode) => {
      if (topChunks.length === 0 || !query.trim()) {
        setRagAnswer(null);
        return;
      }
      currentQueryRef.current = query;

      setIsGeneratingRAG(true);
      setRagGenerationStatus(
        mode === 'in_browser_llm'
          ? 'Running in-browser generative ONNX LLM...'
          : mode === 'ollama'
          ? 'Querying local Ollama model...'
          : 'Extracting key sentence evidence...'
      );

      const worker = ragWorkerRef.current;
      if (!worker) {
        // Synchronous fallback
        const fallback = LocalRAGSynthesizer.synthesize(query, topChunks);
        setRagAnswer(fallback);
        setIsGeneratingRAG(false);
        return;
      }

      worker.postMessage({
        type: 'GENERATE',
        query,
        contexts: topChunks.map((c, i) => ({
          index: i + 1,
          chunkId: c.id,
          documentName: c.documentName,
          chunkIndex: c.chunkIndex,
          text: c.text,
        })),
        backend: mode,
        ollamaEndpoint,
        ollamaModel,
        history: ragHistory.slice(-6), // Last 3 Q&A pairs
      } as RAGWorkerInMessage);
    },
    [ragMode, ollamaEndpoint, ollamaModel, ragHistory]
  );

  // Initialize Web Workers and rehydrate from IndexedDB on startup
  useEffect(() => {
    // 1. Embedding Worker
    const embWorker = new Worker(new URL('./workers/embedding.worker.ts', import.meta.url), {
      type: 'module',
    });
    workerRef.current = embWorker;

    embWorker.onmessage = (e: MessageEvent<WorkerOutMessage>) => {
      const msg = e.data;

      if (msg.type === 'STATUS') {
        setModelStatus(msg.status);
        setIsFallback(Boolean(msg.isFallback));
        if (msg.device) {
          setEmbeddingDevice(msg.device);
        }
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
        if (pendingBatchResolver.current) {
          pendingBatchResolver.current([]);
          pendingBatchResolver.current = null;
        }
        if (pendingQueryResolver.current) {
          pendingQueryResolver.current(new Float32Array(384));
          pendingQueryResolver.current = null;
        }
      }
    };

    embWorker.postMessage({ type: 'INIT', modelName, device: embeddingDeviceRef.current } as WorkerInMessage);

    // 2. Generative RAG Worker
    const rWorker = new Worker(new URL('./workers/rag.worker.ts', import.meta.url), {
      type: 'module',
    });
    ragWorkerRef.current = rWorker;

    rWorker.onmessage = (e: MessageEvent<RAGWorkerOutMessage>) => {
      const msg = e.data;
      if (msg.type === 'RAG_STATUS') {
        if (msg.message) setRagGenerationStatus(msg.message);
      } else if (msg.type === 'RAG_STREAM_TOKEN') {
        setRagGenerationStatus(msg.accumulated);
      } else if (msg.type === 'RAG_RESULT') {
        const answer = {
          summary: msg.summary,
          bulletPoints: msg.bulletPoints,
          citations: msg.citations,
          confidence: msg.confidence,
          backendUsed: msg.backendUsed,
          modelTitle: msg.modelTitle,
        };
        setRagAnswer(answer);
        setIsGeneratingRAG(false);
        // Append to conversation history
        setRagHistory(prev => [
          ...prev,
          { role: 'user' as const, content: currentQueryRef.current || currentQuery },
          { role: 'assistant' as const, content: answer.summary + (answer.bulletPoints.length > 0 ? '\n' + answer.bulletPoints.join('\n') : '') },
        ]);
      } else if (msg.type === 'RAG_ERROR') {
        console.warn('RAG worker error:', msg.error);
        setIsGeneratingRAG(false);
      }
    };

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
      embWorker.terminate();
      rWorker.terminate();
    };
  }, [modelName]);

  // Global keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl+K or / to focus search (when not in an input)
      if ((e.key === 'k' && (e.ctrlKey || e.metaKey)) || (e.key === '/' && !(e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement))) {
        e.preventDefault();
        const searchInput = document.querySelector<HTMLInputElement>('input[placeholder*="Search concepts"], input[placeholder*="Load a document"]');
        searchInput?.focus();
      }
      // Escape to close inspector modal
      if (e.key === 'Escape' && inspectingItem) {
        setInspectingItem(null);
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [inspectingItem]);

  // Handle Model Switching
  const handleSelectModel = (newModel: string) => {
    if (newModel === modelName) return;

    if (vectors.length > 0) {
      const confirmSwitch = window.confirm(
        `Switching model to ${newModel.split('/').pop()} will initialize a new embedding space. Do you want to reset current vectors for clean semantic consistency?`
      );
      if (confirmSwitch) {
        hnswRef.current.clear();
        bm25Ref.current.clear();
        storageRef.current.clearAll();
        setChunks([]);
        setVectors([]);
        setSearchResults([]);
        setRagAnswer(null);
        setQueryVector(null);
      }
    }

    setModelName(newModel);
    setModelStatus('loading');
    workerRef.current?.postMessage({ type: 'INIT', modelName: newModel, device: embeddingDevice } as WorkerInMessage);
  };

  // Handle WebGPU Device Toggling
  const handleToggleDevice = () => {
    if (!hasWebGpu) return;
    const nextDevice = embeddingDevice === 'webgpu' ? 'wasm' : 'webgpu';
    setEmbeddingDevice(nextDevice);
    setModelStatus('loading');
    workerRef.current?.postMessage({
      type: 'INIT',
      modelName,
      device: nextDevice,
    } as WorkerInMessage);
  };

  // Real in-memory quantization switching
  const handleSwitchQuantization = (mode: QuantizationMode) => {
    const prevHnsw = hnswRef.current;
    if (prevHnsw.quantization === mode) return;

    const newHnsw = new HNSWIndex({
      M: prevHnsw.M,
      M0: prevHnsw.M0,
      efConstruction: prevHnsw.efConstruction,
      efSearch: prevHnsw.efSearch,
      metric: prevHnsw.metric,
      quantization: mode,
    });

    for (const v of vectors) {
      newHnsw.insert(v.id, v.vector);
    }

    hnswRef.current = newHnsw;
  };

  const [, setEfSearch] = useState(hnswRef.current.efSearch);

  const handleEfSearchChange = (efSearch: number) => {
    hnswRef.current.efSearch = efSearch;
    setEfSearch(efSearch);
  };

  // Handle adding new document
  const handleAddDocument = async (filename: string, content: string) => {
    setIsProcessing(true);
    setProcessStatus('Chunking document...');

    try {
      let newChunks: TextChunk[];

      if (CodeChunker.isSupportedCodeFile(filename)) {
        setProcessStatus('Applying syntax-aware code chunker...');
        newChunks = CodeChunker.split(content, filename, { maxChunkLines: 40, overlapLines: 8 });
      } else {
        const chunker = new RecursiveChunker({ chunkSize, chunkOverlap });
        newChunks = chunker.split(content, filename);
      }

      if (newChunks.length === 0) {
        throw new Error('No chunks generated from document.');
      }

      setProcessStatus(`Embedding ${newChunks.length} chunks via ONNX runtime...`);
      setBatchProgress({ current: 0, total: newChunks.length });

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

      for (const item of embeddingResults) {
        hnswRef.current.insert(item.id, item.vector);
      }
      for (const c of newChunks) {
        bm25Ref.current.addDocument(c.id, c.text);
      }

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
    async (query: string, mode: SearchMode, topK: number, documentFilter?: string | null) => {
      if (chunks.length === 0) return;
      setCurrentQuery(query);
      const t0 = performance.now();

      const filterFn = documentFilter
        ? (id: string) => chunksMap.get(id)?.documentName === documentFilter
        : undefined;

      if (mode === 'sparse') {
        const bm25Res = bm25Ref.current.search(query, topK, filterFn);
        const hybridResults = bm25Res.map((r) => ({
          id: r.id,
          combinedScore: r.score / (bm25Res[0]?.score || 1.0),
          sparseScore: r.score,
        }));
        const t1 = performance.now();
        setSearchResults(hybridResults);
        setQueryLatencyMs(t1 - t0);

        const topChunks = hybridResults
          .slice(0, 4)
          .map((r) => chunksMap.get(r.id))
          .filter(Boolean) as TextChunk[];
        executeRAG(query, topChunks);
        return;
      }

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
        const denseRes = hnswRef.current.search(qVec, topK, filterFn);
        const hybridResults = denseRes.map((r) => ({
          id: r.id,
          combinedScore: r.score,
          denseScore: r.score,
        }));
        const t1 = performance.now();
        setSearchResults(hybridResults);
        setQueryLatencyMs(t1 - t0);

        const topChunks = hybridResults
          .slice(0, 4)
          .map((r) => chunksMap.get(r.id))
          .filter(Boolean) as TextChunk[];
        executeRAG(query, topChunks);
      } else {
        const denseRes = hnswRef.current.search(qVec, topK * 2, filterFn);
        const sparseRes = bm25Ref.current.search(query, topK * 2, filterFn);
        const fused = reciprocalRankFusion(denseRes, sparseRes, topK);
        const t1 = performance.now();
        setSearchResults(fused);
        setQueryLatencyMs(t1 - t0);

        const topChunks = fused
          .slice(0, 4)
          .map((r) => chunksMap.get(r.id))
          .filter(Boolean) as TextChunk[];
        executeRAG(query, topChunks);
      }
    },
    [chunks, chunksMap, executeRAG]
  );

  // Handle Find Similar
  const handleFindSimilar = useCallback(
    (chunk: TextChunk) => {
      const vec = hnswRef.current.getVector(chunk.id);
      if (!vec) return;

      const t0 = performance.now();
      const similarQueryTitle = `Similar to [${chunk.documentName} #${chunk.chunkIndex}]`;
      setCurrentQuery(similarQueryTitle);
      setQueryVector(vec);

      const rawResults = hnswRef.current.search(vec, 7);
      const filtered =
        rawResults.length > 1 ? rawResults.filter((r) => r.id !== chunk.id) : rawResults;

      const hybridResults: HybridSearchResult[] = filtered.slice(0, 5).map((r) => ({
        id: r.id,
        combinedScore: r.score,
        denseScore: r.score,
      }));

      const t1 = performance.now();
      setSearchResults(hybridResults);
      setQueryLatencyMs(t1 - t0);

      const topChunks = hybridResults
        .slice(0, 4)
        .map((r) => chunksMap.get(r.id))
        .filter(Boolean) as TextChunk[];
      executeRAG(similarQueryTitle, topChunks);
    },
    [chunksMap, executeRAG]
  );

  // Re-generate RAG answer when switching mode or clicking refresh
  const handleRegenerateRAG = (newMode?: 'in_browser_llm' | 'ollama' | 'extractive') => {
    const targetMode = newMode || ragMode;
    if (newMode) setRagMode(newMode);

    if (searchResults.length > 0 && currentQuery.trim()) {
      const topChunks = searchResults
        .slice(0, 4)
        .map((r) => chunksMap.get(r.id))
        .filter(Boolean) as TextChunk[];
      executeRAG(currentQuery, topChunks, targetMode);
    }
  };

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
    setEfSearch(restoredHnsw.efSearch);
    setSearchResults([]);
    setRagAnswer(null);
  };

  // Delete a specific document and its associated chunks and vectors
  const handleDeleteDocument = async (documentName: string) => {
    // Find chunk IDs belonging to this document
    const docChunkIds = chunks
      .filter((c) => c.documentName === documentName)
      .map((c) => c.id);

    if (docChunkIds.length === 0) return;

    // Delete from HNSW index
    for (const id of docChunkIds) {
      hnswRef.current.delete(id);
    }
    // Compact HNSW graph after bulk deletion
    hnswRef.current.compact();

    // Delete from BM25 index
    for (const id of docChunkIds) {
      bm25Ref.current.removeDocument(id);
    }

    // Remove from IndexedDB
    await storageRef.current.deleteDocumentData(documentName, docChunkIds);

    // Update React state
    setChunks((prev) => prev.filter((c) => c.documentName !== documentName));
    setVectors((prev) => prev.filter((v) => !docChunkIds.includes(v.id)));
    setSearchResults((prev) => prev.filter((r) => !docChunkIds.includes(r.id)));
    setRagAnswer(null);
    setInspectingItem((prev) => (prev?.chunk.documentName === documentName ? null : prev));
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
    setRagAnswer(null);
    setRagHistory([]);
    setQueryVector(null);
    setQueryLatencyMs(null);
  };

  return (
    <div 
      data-theme={theme}
      className={`min-h-screen flex flex-col selection:bg-brand-500/20 selection:text-brand-300 transition-colors duration-300 ${
        theme === 'light' 
          ? 'bg-white text-slate-900' 
          : 'bg-slate-950 text-slate-100 bg-grid-pattern'
      }`}
    >
      <Header
        modelStatus={modelStatus}
        modelName={modelName}
        isFallback={isFallback}
        isWasmSimd={isSimdAvailable}
        hasWebGpu={hasWebGpu}
        embeddingDevice={embeddingDevice}
        onToggleDevice={handleToggleDevice}
        vectorCount={vectors.length}
        totalTokens={totalTokens}
        downloadProgress={downloadProgress}
        onSelectModel={handleSelectModel}
        theme={theme}
        onToggleTheme={toggleTheme}
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
                100% Client-Side WebAssembly & Generative RAG
              </span>
            </div>
            <p className="text-sm text-slate-300">
              Zero cloud latency, zero external API costs, zero data leakage. Embeddings, HNSW graph indexing, hardware SIMD math, SQ8/BQ memory quantization, and grounded local LLM synthesis run entirely in your browser.
            </p>
          </div>

          <div className="flex items-center space-x-2 shrink-0 text-xs font-mono">
            <span className="px-3 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-slate-300 flex items-center space-x-1.5">
              <ShieldCheck className="w-3.5 h-3.5 text-brand-400" />
              <span>Offline Ready</span>
            </span>
            <span className="px-3 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-slate-300 flex items-center space-x-1.5">
              <Cpu className="w-3.5 h-3.5 text-indigo-400" />
              <span>128-bit SIMD</span>
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
              chunkSize={chunkSize}
              chunkOverlap={chunkOverlap}
              onChunkSizeChange={setChunkSize}
              onChunkOverlapChange={setChunkOverlap}
            />

            <DocumentLibrary
              documents={documentInfos}
              onDeleteDocument={handleDeleteDocument}
              isProcessing={isProcessing}
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
              availableDocuments={availableDocuments}
            />

            {/* Local Grounded Generative RAG Answer Box */}
            <RAGAnswerBox
              answer={ragAnswer}
              query={currentQuery}
              isGenerating={isGeneratingRAG}
              generationStatus={ragGenerationStatus}
              activeMode={ragMode}
              onChangeMode={(m) => handleRegenerateRAG(m)}
              onRegenerate={() => handleRegenerateRAG()}
              ollamaEndpoint={ollamaEndpoint}
              ollamaModel={ollamaModel}
              onOllamaEndpointChange={setOllamaEndpoint}
              onOllamaModelChange={setOllamaModel}
              history={ragHistory}
            />

            <SearchResults
              results={searchResults}
              chunksMap={chunksMap}
              onInspectChunk={(chunk, res) => setInspectingItem({ chunk, result: res })}
              onFindSimilar={handleFindSimilar}
              query={currentQuery}
            />

            {/* Empty State when no results and no documents */}
            {searchResults.length === 0 && (
              <div className="p-8 rounded-2xl bg-slate-900/40 border border-dashed border-slate-800 text-center space-y-3">
                <Compass className="w-8 h-8 text-slate-600 mx-auto" />
                <h3 className="text-sm font-semibold text-slate-400">
                  {vectors.length === 0
                    ? 'No documents loaded in local vector engine'
                    : 'Enter a search query above to see ranked matches & generated answers'}
                </h3>
                <p className="text-xs text-slate-500 max-w-md mx-auto">
                  {vectors.length === 0
                    ? 'Click "Systems Architecture" or "Raft Python Code" on the left to test semantic retrieval and generative RAG.'
                    : 'Query vectors are generated on the fly via Web Worker and matched against the in-memory HNSW index.'}
                </p>
              </div>
            )}
          </div>
        </div>

        {/* 2D / 3D Vector Cluster Visualizer */}
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
          onSwitchQuantization={handleSwitchQuantization}
          onEfSearchChange={handleEfSearchChange}
        />
      </main>

      {/* Chunk & Vector Inspector Modal */}
      {inspectingItem && (
        <ChunkInspectorModal
          chunk={inspectingItem.chunk}
          result={inspectingItem.result}
          vector={hnswRef.current.getVector(inspectingItem.chunk.id) || undefined}
          onClose={() => setInspectingItem(null)}
          onFindSimilar={handleFindSimilar}
        />
      )}

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 py-6 mt-12 text-xs text-slate-500 font-mono">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center space-x-2">
            <span>VectorLite-Wasm</span>
            <span>•</span>
            <span>Client-Side ONNX WebAssembly, Hardware SIMD & Local Generative RAG</span>
          </div>
          <div>MIT License • Built by Alexandr Motologa</div>
        </div>
      </footer>
    </div>
  );
};
