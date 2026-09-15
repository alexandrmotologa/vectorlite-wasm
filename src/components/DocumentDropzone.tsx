import React, { useState, useRef } from 'react';
import { UploadCloud, FileText, BookOpen, Layers, AlertCircle, Loader2 } from 'lucide-react';
import { extractTextFromPDF } from '../engine/pdf_loader';

interface DocumentDropzoneProps {
  onAddDocument: (filename: string, content: string) => Promise<void>;
  isProcessing: boolean;
  processStatus: string;
  batchProgress: { current: number; total: number } | null;
}

export const DocumentDropzone: React.FC<DocumentDropzoneProps> = ({
  onAddDocument,
  isProcessing,
  processStatus,
  batchProgress,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
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

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      await handleFileProcess(file);
    }
  };

  const handleSelectFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      await handleFileProcess(file);
      e.target.value = '';
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
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 backdrop-blur-sm">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Layers className="w-5 h-5 text-brand-400" />
          <h2 className="text-base font-semibold text-white">Document Ingestion & Chunking</h2>
        </div>
        <span className="text-xs font-mono text-slate-400 bg-slate-800/80 px-2.5 py-1 rounded-md border border-slate-700/60">
          Markdown • Plain Text • PDF
        </span>
      </div>

      {/* Drag & Drop Area */}
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        onClick={() => !isProcessing && fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all duration-200 ${
          isDragOver
            ? 'border-brand-400 bg-brand-500/10'
            : 'border-slate-800 hover:border-slate-700 bg-slate-950/40 hover:bg-slate-950/70'
        } ${isProcessing ? 'pointer-events-none opacity-60' : ''}`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".txt,.md,.markdown,.pdf,.json"
          onChange={handleSelectFile}
          className="hidden"
        />

        <div className="flex flex-col items-center justify-center space-y-2">
          <div className="w-12 h-12 rounded-xl bg-slate-800/70 flex items-center justify-center text-slate-300">
            {isProcessing ? (
              <Loader2 className="w-6 h-6 text-brand-400 animate-spin" />
            ) : (
              <UploadCloud className="w-6 h-6 text-brand-400" />
            )}
          </div>
          <div>
            <p className="text-sm font-medium text-slate-200">
              {isProcessing ? 'Processing & Generating Embeddings...' : 'Drop files here or click to browse'}
            </p>
            <p className="text-xs text-slate-500 mt-0.5">
              Processes text locally. Nothing is uploaded to remote servers.
            </p>
          </div>
        </div>
      </div>

      {/* Progress or Status Bar */}
      {isProcessing && (
        <div className="mt-4 p-3.5 rounded-xl bg-slate-950 border border-slate-800">
          <div className="flex items-center justify-between text-xs mb-1.5 font-mono">
            <span className="text-brand-300 flex items-center space-x-1.5">
              <Loader2 className="w-3 h-3 animate-spin" />
              <span>{processStatus}</span>
            </span>
            {batchProgress && (
              <span className="text-slate-400">
                {batchProgress.current} / {batchProgress.total} chunks (
                {Math.round((batchProgress.current / batchProgress.total) * 100)}%)
              </span>
            )}
          </div>
          {batchProgress && (
            <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
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
        <div className="mt-3 p-3 rounded-lg bg-red-950/50 border border-red-800/60 text-xs text-red-300 flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 shrink-0 text-red-400" />
          <span>{error}</span>
        </div>
      )}

      {/* Quick-load Sample Buttons */}
      <div className="mt-4 pt-4 border-t border-slate-800/60">
        <div className="flex items-center justify-between mb-2.5">
          <span className="text-xs font-medium text-slate-400 flex items-center space-x-1.5">
            <BookOpen className="w-3.5 h-3.5 text-brand-400" />
            <span>Pre-bundled Datasets:</span>
          </span>
          <span className="text-[11px] text-slate-500">Instant test without uploading files</span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
          <button
            type="button"
            disabled={isProcessing}
            onClick={() => loadSample('/samples/systems_architecture.md', 'systems_architecture.md')}
            className="flex items-center justify-between px-3.5 py-2 rounded-xl bg-slate-950 hover:bg-slate-800/80 border border-slate-800 hover:border-slate-700 text-left transition-colors disabled:opacity-50"
          >
            <div className="flex items-center space-x-2.5">
              <FileText className="w-4 h-4 text-brand-400 shrink-0" />
              <div>
                <p className="text-xs font-semibold text-slate-200">Systems Architecture</p>
                <p className="text-[10px] text-slate-500">Raft, LSM, HNSW, Caching</p>
              </div>
            </div>
            <span className="text-[10px] font-mono text-brand-400 bg-brand-950/60 px-2 py-0.5 rounded border border-brand-800/40">
              Load
            </span>
          </button>

          <button
            type="button"
            disabled={isProcessing}
            onClick={() => loadSample('/samples/ai_engineering_guide.md', 'ai_engineering_guide.md')}
            className="flex items-center justify-between px-3.5 py-2 rounded-xl bg-slate-950 hover:bg-slate-800/80 border border-slate-800 hover:border-slate-700 text-left transition-colors disabled:opacity-50"
          >
            <div className="flex items-center space-x-2.5">
              <FileText className="w-4 h-4 text-indigo-400 shrink-0" />
              <div>
                <p className="text-xs font-semibold text-slate-200">AI Engineering Guide</p>
                <p className="text-[10px] text-slate-500">RAG, Hybrid Search, Embeddings</p>
              </div>
            </div>
            <span className="text-[10px] font-mono text-indigo-400 bg-indigo-950/60 px-2 py-0.5 rounded border border-indigo-800/40">
              Load
            </span>
          </button>
        </div>
      </div>
    </div>
  );
};
