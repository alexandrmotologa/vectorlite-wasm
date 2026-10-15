import React, { useState, useRef } from 'react';
import { UploadCloud, FileText, BookOpen, Layers, AlertCircle, Loader2, Globe, Code2, Sliders } from 'lucide-react';
import { extractTextFromPDF } from '../engine/pdf_loader';

interface DocumentDropzoneProps {
  onAddDocument: (filename: string, content: string) => Promise<void>;
  isProcessing: boolean;
  processStatus: string;
  batchProgress: { current: number; total: number } | null;
  chunkSize?: number;
  chunkOverlap?: number;
  onChunkSizeChange?: (size: number) => void;
  onChunkOverlapChange?: (overlap: number) => void;
}

export const DocumentDropzone: React.FC<DocumentDropzoneProps> = ({
  onAddDocument,
  isProcessing,
  processStatus,
  batchProgress,
  chunkSize,
  chunkOverlap,
  onChunkSizeChange,
  onChunkOverlapChange,
}) => {
  const [activeTab, setActiveTab] = useState<'upload' | 'url'>('upload');
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [urlInput, setUrlInput] = useState('');
  const [isFetchingUrl, setIsFetchingUrl] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileProcess = async (file: File) => {
    setError(null);
    try {
      if (file.type === 'application/pdf' || file.name.endsWith('.pdf')) {
        const result = await extractTextFromPDF(file);
        await onAddDocument(file.name, result.text);
      } else {
        const text = await file.text();
        await onAddDocument(file.name, text);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to parse file';
      setError(msg);
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (isProcessing) return;

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      for (let i = 0; i < files.length; i++) {
        await handleFileProcess(files[i]);
      }
    }
  };

  const handleSelectFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      for (let i = 0; i < e.target.files.length; i++) {
        await handleFileProcess(e.target.files[i]);
      }
      e.target.value = '';
    }
  };

  const handleUrlFetch = async (targetUrl: string) => {
    if (!targetUrl.trim()) return;
    setError(null);
    setIsFetchingUrl(true);

    try {
      const res = await fetch(targetUrl.trim());
      if (!res.ok) {
        throw new Error(`HTTP Error ${res.status}: ${res.statusText}`);
      }
      const text = await res.text();

      // Derive file name from URL path
      const urlObj = new URL(targetUrl.trim());
      const pathSegments = urlObj.pathname.split('/').filter(Boolean);
      const filename = pathSegments.pop() || 'remote_document.txt';

      await onAddDocument(filename, text);
      setUrlInput('');
    } catch (err: unknown) {
      const msg =
        err instanceof Error
          ? err.message
          : 'Failed to fetch from URL. Make sure URL has CORS enabled (e.g. raw.githubusercontent.com).';
      setError(`URL Fetch failed: ${msg}`);
    } finally {
      setIsFetchingUrl(false);
    }
  };

  const loadSample = async (samplePath: string, sampleName: string) => {
    setError(null);
    try {
      const res = await fetch(samplePath);
      if (!res.ok) throw new Error(`Could not fetch sample dataset ${samplePath}`);
      const text = await res.text();
      await onAddDocument(sampleName, text);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to load sample dataset';
      setError(msg);
    }
  };

  return (
    <div className="bg-white dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 rounded-2xl p-4 sm:p-6 backdrop-blur-sm shadow-xs">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Layers className="w-5 h-5 text-brand-500 dark:text-brand-400" />
          <h2 className="text-base font-semibold text-slate-900 dark:text-white">Document Ingestion & Chunking</h2>
        </div>
        <span className="text-xs font-mono text-slate-600 dark:text-slate-400 bg-slate-100 dark:bg-slate-800/80 px-2.5 py-1 rounded-md border border-slate-200 dark:border-slate-700/60">
          Markdown • Code • PDF
        </span>
      </div>

      {/* Tabs: Upload / Dropzone vs Direct URL */}
      <div className="flex items-center space-x-2 mb-3">
        <button
          type="button"
          onClick={() => setActiveTab('upload')}
          className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
            activeTab === 'upload'
              ? 'bg-slate-100 dark:bg-slate-800 text-brand-600 dark:text-brand-300 border border-slate-300 dark:border-slate-700 font-semibold shadow-xs'
              : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800/50'
          }`}
        >
          <UploadCloud className="w-3.5 h-3.5" />
          <span>Local Files & Code</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('url')}
          className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
            activeTab === 'url'
              ? 'bg-slate-100 dark:bg-slate-800 text-brand-600 dark:text-brand-300 border border-slate-300 dark:border-slate-700 font-semibold shadow-xs'
              : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800/50'
          }`}
        >
          <Globe className="w-3.5 h-3.5" />
          <span>Direct URL / GitHub Raw</span>
        </button>
      </div>

      {activeTab === 'upload' ? (
        /* Drag & Drop Area */
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
          onClick={() => !isProcessing && fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all duration-200 ${
            isDragOver
              ? 'border-brand-500 bg-brand-500/10'
              : 'border-slate-300 dark:border-slate-800 hover:border-slate-400 dark:hover:border-slate-700 bg-slate-50/60 dark:bg-slate-950/40 hover:bg-slate-50 dark:hover:bg-slate-950/70'
          } ${isProcessing ? 'pointer-events-none opacity-60' : ''}`}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".txt,.md,.markdown,.pdf,.json,.ts,.js,.tsx,.jsx,.py,.rs,.go,.java,.c,.cpp,.html,.css"
            onChange={handleSelectFile}
            className="hidden"
          />

          <div className="flex flex-col items-center justify-center space-y-2">
            <div className="w-12 h-12 rounded-xl bg-slate-100 dark:bg-slate-800/70 flex items-center justify-center text-slate-600 dark:text-slate-300">
              {isProcessing ? (
                <Loader2 className="w-6 h-6 text-brand-500 dark:text-brand-400 animate-spin" />
              ) : (
                <UploadCloud className="w-6 h-6 text-brand-500 dark:text-brand-400" />
              )}
            </div>
            <div>
              <p className="text-sm font-medium text-slate-800 dark:text-slate-200">
                {isProcessing ? 'Processing & Generating Embeddings...' : 'Drop files or click to browse'}
              </p>
              <p className="text-xs text-slate-500 mt-0.5">
                Markdown, text, PDF, or code files (.ts, .py, .rs, .go, .js).
              </p>
            </div>
          </div>
        </div>
      ) : (
        /* Direct URL Ingestion Panel */
        <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 space-y-3">
          <label className="block text-xs font-medium text-slate-700 dark:text-slate-300">
            Enter Raw File URL (e.g., GitHub raw or public CORS text):
          </label>
          <div className="flex items-center space-x-2">
            <input
              type="url"
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              placeholder="https://raw.githubusercontent.com/user/repo/main/README.md"
              className="flex-1 px-3 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700/80 rounded-lg text-xs font-mono text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:border-brand-500"
            />
            <button
              type="button"
              disabled={isFetchingUrl || isProcessing || !urlInput.trim()}
              onClick={() => handleUrlFetch(urlInput)}
              className="px-4 py-2 bg-brand-500 hover:bg-brand-400 text-slate-950 font-semibold text-xs rounded-lg transition-colors disabled:opacity-50 flex items-center space-x-1.5"
            >
              {isFetchingUrl ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Fetching...</span>
                </>
              ) : (
                <span>Fetch & Index</span>
              )}
            </button>
          </div>

          {/* Quick Preset URLs */}
          <div className="pt-2 text-[11px] text-slate-500 dark:text-slate-400 space-y-1 font-mono">
            <span className="text-slate-600 dark:text-slate-500 block">Quick examples:</span>
            <div className="flex flex-wrap gap-1.5">
              <button
                type="button"
                onClick={() =>
                  setUrlInput(
                    'https://raw.githubusercontent.com/tiangolo/fastapi/master/README.md'
                  )
                }
                className="px-2 py-0.5 rounded bg-white dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 border border-slate-200 dark:border-slate-800 text-slate-700 dark:text-slate-300 text-[10px]"
              >
                FastAPI README.md
              </button>
              <button
                type="button"
                onClick={() =>
                  setUrlInput(
                    'https://raw.githubusercontent.com/BurntSushi/ripgrep/master/crates/core/main.rs'
                  )
                }
                className="px-2 py-0.5 rounded bg-white dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 border border-slate-200 dark:border-slate-800 text-slate-700 dark:text-slate-300 text-[10px]"
              >
                ripgrep main.rs (Rust)
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Progress or Status Bar */}
      {isProcessing && (
        <div className="mt-4 p-3.5 rounded-xl bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
          <div className="flex items-center justify-between text-xs mb-1.5 font-mono">
            <span className="text-brand-600 dark:text-brand-300 flex items-center space-x-1.5">
              <Loader2 className="w-3 h-3 animate-spin" />
              <span>{processStatus}</span>
            </span>
            {batchProgress && (
              <span className="text-slate-600 dark:text-slate-400">
                {batchProgress.current} / {batchProgress.total} chunks (
                {Math.round((batchProgress.current / batchProgress.total) * 100)}%)
              </span>
            )}
          </div>
          {batchProgress && (
            <div className="w-full bg-slate-200 dark:bg-slate-800 rounded-full h-1.5 overflow-hidden">
              <div
                className="bg-brand-500 h-1.5 rounded-full transition-all duration-300"
                style={{ width: `${(batchProgress.current / batchProgress.total) * 100}%` }}
              />
            </div>
          )}
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="mt-3 p-3 rounded-lg bg-red-50 dark:bg-red-950/50 border border-red-200 dark:border-red-800/60 text-xs text-red-600 dark:text-red-300 flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 shrink-0 text-red-500 dark:text-red-400" />
          <span>{error}</span>
        </div>
      )}

      {/* Advanced Chunking Settings */}
      <div className="mt-3">
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-[11px] text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 font-mono flex items-center space-x-1"
        >
          <Sliders className="w-3 h-3" />
          <span>{showAdvanced ? 'Hide' : 'Show'} Chunking Settings</span>
        </button>
        {showAdvanced && (
          <div className="mt-2 p-3 rounded-lg bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 space-y-2 text-xs font-mono">
            <div className="flex items-center justify-between">
              <label className="text-slate-600 dark:text-slate-400">Chunk Size (chars):</label>
              <div className="flex items-center space-x-2">
                <input
                  type="range"
                  min={200}
                  max={2000}
                  step={50}
                  value={chunkSize || 550}
                  onChange={(e) => onChunkSizeChange?.(Number(e.target.value))}
                  className="w-24 accent-brand-500"
                />
                <span className="text-slate-900 dark:text-white font-bold w-10 text-right">{chunkSize || 550}</span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <label className="text-slate-600 dark:text-slate-400">Overlap (chars):</label>
              <div className="flex items-center space-x-2">
                <input
                  type="range"
                  min={0}
                  max={500}
                  step={25}
                  value={chunkOverlap || 100}
                  onChange={(e) => onChunkOverlapChange?.(Number(e.target.value))}
                  className="w-24 accent-brand-500"
                />
                <span className="text-slate-900 dark:text-white font-bold w-10 text-right">{chunkOverlap || 100}</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Quick-load Sample Buttons */}
      <div className="mt-4 pt-4 border-t border-slate-200 dark:border-slate-800/60">
        <div className="flex items-center justify-between mb-2.5">
          <span className="text-xs font-medium text-slate-600 dark:text-slate-400 flex items-center space-x-1.5">
            <BookOpen className="w-3.5 h-3.5 text-brand-500 dark:text-brand-400" />
            <span>Pre-bundled Datasets:</span>
          </span>
          <span className="text-[11px] text-slate-500">Instant test without uploading files</span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 lg:grid-cols-1 xl:grid-cols-3 gap-2.5">
          <button
            type="button"
            disabled={isProcessing}
            onClick={() => loadSample('/samples/systems_architecture.md', 'systems_architecture.md')}
            className="flex flex-col justify-between p-2.5 rounded-xl bg-slate-50 dark:bg-slate-950 hover:bg-slate-100 dark:hover:bg-slate-800/80 border border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700 text-left transition-colors disabled:opacity-50"
          >
            <div className="flex items-center space-x-2 mb-1">
              <FileText className="w-3.5 h-3.5 text-brand-500 dark:text-brand-400 shrink-0" />
              <p className="text-xs font-semibold text-slate-800 dark:text-slate-200">Systems Architecture</p>
            </div>
            <p className="text-[10px] text-slate-500">Raft, LSM, HNSW</p>
          </button>

          <button
            type="button"
            disabled={isProcessing}
            onClick={() => loadSample('/samples/ai_engineering_guide.md', 'ai_engineering_guide.md')}
            className="flex flex-col justify-between p-2.5 rounded-xl bg-slate-50 dark:bg-slate-950 hover:bg-slate-100 dark:hover:bg-slate-800/80 border border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700 text-left transition-colors disabled:opacity-50"
          >
            <div className="flex items-center space-x-2 mb-1">
              <FileText className="w-3.5 h-3.5 text-indigo-500 dark:text-indigo-400 shrink-0" />
              <p className="text-xs font-semibold text-slate-800 dark:text-slate-200">AI Engineering</p>
            </div>
            <p className="text-[10px] text-slate-500">RAG, Hybrid Search</p>
          </button>

          <button
            type="button"
            disabled={isProcessing}
            onClick={() => loadSample('/samples/raft_node.py', 'raft_node.py')}
            className="flex flex-col justify-between p-2.5 rounded-xl bg-slate-50 dark:bg-slate-950 hover:bg-slate-100 dark:hover:bg-slate-800/80 border border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700 text-left transition-colors disabled:opacity-50"
          >
            <div className="flex items-center space-x-2 mb-1">
              <Code2 className="w-3.5 h-3.5 text-amber-500 dark:text-amber-400 shrink-0" />
              <p className="text-xs font-semibold text-slate-800 dark:text-slate-200">Raft Python Code</p>
            </div>
            <p className="text-[10px] text-slate-500">Syntax-aware chunking</p>
          </button>
        </div>
      </div>
    </div>
  );
};
