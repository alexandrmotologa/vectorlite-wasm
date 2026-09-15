import React, { useRef, useState } from 'react';
import { Download, Upload, Trash2, FolderDown, Check, AlertCircle } from 'lucide-react';
import { HNSWIndex } from '../engine/hnsw_index';
import { TextChunk } from '../engine/chunker';
import { downloadSnapshot, parseSnapshot } from '../engine/snapshot';

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
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 backdrop-blur-sm">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <FolderDown className="w-5 h-5 text-brand-400" />
          <h2 className="text-base font-semibold text-white">Index Snapshots & Persistence</h2>
        </div>
        <span className="text-xs font-mono text-slate-400 bg-slate-800/80 px-2.5 py-1 rounded-md border border-slate-700/60">
          .vlite format
        </span>
      </div>

      <p className="text-xs text-slate-400 mb-4 leading-relaxed">
        Export your indexed vectors, graph topology, and document chunks into a portable snapshot file. Re-import anywhere to search instantly without re-embedding.
      </p>

      {/* Notifications */}
      {successMsg && (
        <div className="mb-3 p-3 rounded-lg bg-brand-950/60 border border-brand-800/60 text-xs text-brand-300 flex items-center space-x-2 font-mono">
          <Check className="w-4 h-4 text-brand-400 shrink-0" />
          <span>{successMsg}</span>
        </div>
      )}
      {errorMsg && (
        <div className="mb-3 p-3 rounded-lg bg-red-950/60 border border-red-800/60 text-xs text-red-300 flex items-center space-x-2 font-mono">
          <AlertCircle className="w-4 h-4 text-red-400 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-3">
        {/* Export Button */}
        <button
          type="button"
          disabled={hnsw.size === 0}
          onClick={handleExport}
          className="flex items-center space-x-2 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-medium text-xs transition-colors border border-slate-700 disabled:opacity-50"
        >
          <Download className="w-4 h-4 text-brand-400" />
          <span>Export .vlite Snapshot</span>
        </button>

        {/* Import Button */}
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
          className="flex items-center space-x-2 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-medium text-xs transition-colors border border-slate-700"
        >
          <Upload className="w-4 h-4 text-indigo-400" />
          <span>Import Snapshot</span>
        </button>

        {/* Clear Database Button */}
        <button
          type="button"
          disabled={hnsw.size === 0 && chunks.length === 0}
          onClick={onClearAll}
          className="ml-auto flex items-center space-x-1.5 px-3 py-2 rounded-xl bg-red-950/40 hover:bg-red-900/60 text-red-300 font-medium text-xs transition-colors border border-red-800/50 disabled:opacity-40"
        >
          <Trash2 className="w-3.5 h-3.5 text-red-400" />
          <span>Reset All</span>
        </button>
      </div>
    </div>
  );
};
