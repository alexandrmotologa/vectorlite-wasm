import React, { useState } from 'react';
import { X, FileText, Hash, Copy, Check, Binary } from 'lucide-react';
import { TextChunk } from '../engine/chunker';
import { HybridSearchResult } from '../engine/hybrid';

interface ChunkInspectorModalProps {
  chunk: TextChunk;
  result?: HybridSearchResult;
  vector?: Float32Array;
  onClose: () => void;
}

export const ChunkInspectorModal: React.FC<ChunkInspectorModalProps> = ({
  chunk,
  result,
  vector,
  onClose,
}) => {
  const [copiedText, setCopiedText] = useState(false);
  const [copiedVector, setCopiedVector] = useState(false);

  const handleCopyText = () => {
    navigator.clipboard.writeText(chunk.text);
    setCopiedText(true);
    setTimeout(() => setCopiedText(false), 1500);
  };

  const handleCopyVector = () => {
    if (!vector) return;
    navigator.clipboard.writeText(JSON.stringify(Array.from(vector)));
    setCopiedVector(true);
    setTimeout(() => setCopiedVector(false), 1500);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="relative w-full max-w-2xl bg-slate-900 border border-slate-700/80 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh]">
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-950/50">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-lg bg-brand-500/20 text-brand-400 flex items-center justify-center border border-brand-500/30">
              <FileText className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-white font-mono">
                {chunk.documentName} • Chunk #{chunk.chunkIndex}
              </h3>
              <p className="text-xs text-slate-400 font-mono">ID: {chunk.id}</p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-5">
          {/* Metadata Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 text-xs font-mono">
            <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-slate-500 block text-[10px]">Position</span>
              <span className="text-slate-200 font-semibold">
                {chunk.chunkIndex} / {chunk.totalChunks}
              </span>
            </div>
            <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-slate-500 block text-[10px]">Tokens Est.</span>
              <span className="text-brand-300 font-semibold">~{chunk.tokenCountEstimate}</span>
            </div>
            <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-slate-500 block text-[10px]">Characters</span>
              <span className="text-slate-200 font-semibold">{chunk.text.length} chars</span>
            </div>
            <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-slate-500 block text-[10px]">Match Score</span>
              <span className="text-brand-400 font-semibold">
                {result ? `${Math.round(result.combinedScore * 100)}%` : 'N/A'}
              </span>
            </div>
          </div>

          {/* Full Chunk Text */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-medium text-slate-300 flex items-center space-x-1.5">
                <Hash className="w-3.5 h-3.5 text-brand-400" />
                <span>Raw Text Content:</span>
              </span>
              <button
                type="button"
                onClick={handleCopyText}
                className="text-xs font-mono text-slate-400 hover:text-white flex items-center space-x-1 transition-colors"
              >
                {copiedText ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-brand-400" />
                    <span className="text-brand-400">Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5" />
                    <span>Copy Text</span>
                  </>
                )}
              </button>
            </div>
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs text-slate-300 leading-relaxed whitespace-pre-wrap selection:bg-brand-500/30">
              {chunk.text}
            </div>
          </div>

          {/* Vector Embeddings Inspection */}
          {vector && (
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-slate-300 flex items-center space-x-1.5">
                  <Binary className="w-3.5 h-3.5 text-indigo-400" />
                  <span>Float32Array Embedding ({vector.length} dimensions):</span>
                </span>
                <button
                  type="button"
                  onClick={handleCopyVector}
                  className="text-xs font-mono text-slate-400 hover:text-white flex items-center space-x-1 transition-colors"
                >
                  {copiedVector ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-indigo-400" />
                      <span className="text-indigo-400">Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5" />
                      <span>Copy Full Vector</span>
                    </>
                  )}
                </button>
              </div>
              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 font-mono text-[11px] text-slate-400 overflow-x-auto">
                <div className="grid grid-cols-4 sm:grid-cols-8 gap-2">
                  {Array.from(vector.slice(0, 16)).map((val, idx) => (
                    <div key={idx} className="bg-slate-900 px-2 py-1 rounded text-center">
                      <span className="text-[9px] text-slate-500 block">d{idx}</span>
                      <span className="text-slate-200">{val.toFixed(3)}</span>
                    </div>
                  ))}
                </div>
                <div className="mt-2 text-right text-[10px] text-slate-500">
                  + {vector.length - 16} additional dimensions
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3 border-t border-slate-800 bg-slate-950 flex justify-end">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
