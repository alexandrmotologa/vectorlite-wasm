import React, { useRef, useState } from 'react';
import { Download, Upload, Trash2, FolderDown, Check, AlertCircle } from 'lucide-react';
import { HNSWIndex } from '../engine/hnsw_index';
import { TextChunk } from '../engine/chunker';
import { downloadSnapshot, parseSnapshot, serializeBinary, deserializeBinary } from '../engine/snapshot';

interface SnapshotManagerProps {
  hnsw: HNSWIndex;
  chunks: TextChunk[];
  onImportSnapshot: (hnsw: HNSWIndex, chunks: TextChunk[]) => void;
  onClearAll: () => Promise<void>;
}

export const SnapshotManager: React.FC<SnapshotManagerProps> = ({
  hnsw,
  chunks,
  onImportSnapshot,
  onClearAll,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const binaryFileInputRef = useRef<HTMLInputElement>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleExport = () => {
    if (hnsw.size === 0) {
      setErrorMsg('Cannot export empty knowledge base. Load documents first.');
      return;
    }
    setErrorMsg(null);
    downloadSnapshot(hnsw, chunks);
    setSuccessMsg('Snapshot exported as .vlite file.');
    setTimeout(() => setSuccessMsg(null), 3000);
  };

  const handleBinaryExport = () => {
    if (hnsw.size === 0) {
      setErrorMsg('Cannot export empty knowledge base. Load documents first.');
      return;
    }
    setErrorMsg(null);
    try {
      const buffer = serializeBinary(hnsw, chunks);
      const blob = new Blob([buffer], { type: 'application/octet-stream' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `vectorlite_snapshot_${Date.now()}.vlws`;
      a.click();
      URL.revokeObjectURL(url);
      setSuccessMsg('Binary snapshot exported as .vlws file (~4x smaller).');
      setTimeout(() => setSuccessMsg(null), 3000);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to export binary snapshot';
      setErrorMsg(msg);
    }
  };

  const handleBinaryImport = async (file: File) => {
    const buffer = await file.arrayBuffer();
    const { hnsw: restoredHnsw, chunks: restoredChunks } = deserializeBinary(buffer);
    onImportSnapshot(restoredHnsw, restoredChunks);
    setSuccessMsg(`Restored ${restoredChunks.length} chunks from binary snapshot.`);
    setTimeout(() => setSuccessMsg(null), 3000);
  };

  const handleBinaryFileInputChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    setErrorMsg(null);

    try {
      await handleBinaryImport(file);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to parse binary snapshot file';
      setErrorMsg(msg);
    } finally {
      e.target.value = '';
    }
  };

  const handleImportFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    setErrorMsg(null);

    try {
      const { hnsw: restoredHnsw, chunks: restoredChunks } = await parseSnapshot(file);
      onImportSnapshot(restoredHnsw, restoredChunks);
      setSuccessMsg(`Restored ${restoredChunks.length} chunks from snapshot.`);
      setTimeout(() => setSuccessMsg(null), 3000);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to parse snapshot file';
      setErrorMsg(msg);
    } finally {
      e.target.value = '';
    }
  };

  return (
    <div className="bg-white dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 rounded-2xl p-4 sm:p-6 backdrop-blur-sm shadow-xs">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <FolderDown className="w-5 h-5 text-brand-500 dark:text-brand-400" />
          <h2 className="text-base font-semibold text-slate-900 dark:text-white">Index Snapshots & Persistence</h2>
        </div>
        <span className="text-xs font-mono text-slate-600 dark:text-slate-400 bg-slate-100 dark:bg-slate-800/80 px-2.5 py-1 rounded-md border border-slate-200 dark:border-slate-700/60">
          .vlws &amp; .vlite
        </span>
      </div>

      <p className="text-xs text-slate-600 dark:text-slate-400 mb-4 leading-relaxed">
        Export your indexed vectors, graph topology, and document chunks into a portable snapshot file. Binary export (.vlws) is ~4x smaller and faster than JSON.
      </p>

      {/* Notifications */}
      {successMsg && (
        <div className="mb-3 p-3 rounded-lg bg-brand-50 dark:bg-brand-950/60 border border-brand-200 dark:border-brand-800/60 text-xs text-brand-700 dark:text-brand-300 flex items-center space-x-2 font-mono">
          <Check className="w-4 h-4 text-brand-500 dark:text-brand-400 shrink-0" />
          <span>{successMsg}</span>
        </div>
      )}
      {errorMsg && (
        <div className="mb-3 p-3 rounded-lg bg-red-50 dark:bg-red-950/60 border border-red-200 dark:border-red-800/60 text-xs text-red-600 dark:text-red-300 flex items-center space-x-2 font-mono">
          <AlertCircle className="w-4 h-4 text-red-500 dark:text-red-400 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Symmetrical 2x2 Action Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 mb-3">
        {/* Binary Export Button */}
        <button
          type="button"
          disabled={hnsw.size === 0}
          onClick={handleBinaryExport}
          className="flex items-center justify-center space-x-2 px-3 py-2.5 rounded-xl bg-brand-600 hover:bg-brand-500 text-white font-medium text-xs transition-colors border border-brand-500 shadow-xs disabled:opacity-50"
        >
          <Download className="w-4 h-4 text-white" />
          <span>Export Binary (.vlws)</span>
        </button>

        {/* Binary Import Button */}
        <input
          ref={binaryFileInputRef}
          type="file"
          accept=".vlws"
          onChange={handleBinaryFileInputChange}
          className="hidden"
        />
        <button
          type="button"
          onClick={() => binaryFileInputRef.current?.click()}
          className="flex items-center justify-center space-x-2 px-3 py-2.5 rounded-xl bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-800 dark:text-white font-medium text-xs transition-colors border border-slate-300 dark:border-slate-700 shadow-xs"
        >
          <Upload className="w-4 h-4 text-brand-500 dark:text-brand-400" />
          <span>Import Binary (.vlws)</span>
        </button>

        {/* JSON Export Button */}
        <button
          type="button"
          disabled={hnsw.size === 0}
          onClick={handleExport}
          className="flex items-center justify-center space-x-2 px-3 py-2.5 rounded-xl bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-800 dark:text-white font-medium text-xs transition-colors border border-slate-300 dark:border-slate-700 shadow-xs disabled:opacity-50"
        >
          <Download className="w-4 h-4 text-slate-500 dark:text-slate-400" />
          <span>Export JSON (.vlite)</span>
        </button>

        {/* JSON Import Button */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".vlite,.json"
          onChange={handleImportFile}
          className="hidden"
        />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="flex items-center justify-center space-x-2 px-3 py-2.5 rounded-xl bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-800 dark:text-white font-medium text-xs transition-colors border border-slate-300 dark:border-slate-700 shadow-xs"
        >
          <Upload className="w-4 h-4 text-indigo-500 dark:text-indigo-400" />
          <span>Import JSON (.vlite)</span>
        </button>
      </div>

      {/* Action Footer: Status & Reset */}
      <div className="flex items-center justify-between pt-3 border-t border-slate-200 dark:border-slate-800/80">
        <span className="text-[11px] font-mono text-slate-500">
          Index status: {hnsw.size} vectors • {chunks.length} chunks
        </span>
        <button
          type="button"
          disabled={hnsw.size === 0 && chunks.length === 0}
          onClick={onClearAll}
          className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-red-50 dark:bg-red-950/40 hover:bg-red-100 dark:hover:bg-red-900/60 text-red-600 dark:text-red-300 font-medium text-xs transition-colors border border-red-200 dark:border-red-800/50 disabled:opacity-40"
        >
          <Trash2 className="w-3.5 h-3.5 text-red-500 dark:text-red-400" />
          <span>Reset All</span>
        </button>
      </div>
    </div>
  );
};
