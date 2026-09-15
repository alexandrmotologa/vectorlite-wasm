import React, { useState } from 'react';
import { FileText, Copy, Check, ExternalLink, Sparkles, Compass, Tag } from 'lucide-react';
import { TextChunk } from '../engine/chunker';
import { HybridSearchResult } from '../engine/hybrid';

interface SearchResultsProps {
  results: HybridSearchResult[];
  chunksMap: Map<string, TextChunk>;
  onInspectChunk: (chunk: TextChunk, result: HybridSearchResult) => void;
  onFindSimilar?: (chunk: TextChunk) => void;
  query: string;
}

export const SearchResults: React.FC<SearchResultsProps> = ({
  results,
  chunksMap,
  onInspectChunk,
  onFindSimilar,
  query,
}) => {
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const handleCopy = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 1800);
  };

  // Helper to highlight query words in excerpt
  const renderHighlightedText = (text: string, searchQuery: string) => {
    if (!searchQuery.trim()) return text;
    const words = searchQuery
      .toLowerCase()
      .split(/\s+/)
      .filter((w) => w.length > 2)
      .map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));

    if (words.length === 0) return text;
    const regex = new RegExp(`(${words.join('|')})`, 'gi');
    const parts = text.split(regex);

    return parts.map((part, i) =>
      regex.test(part) ? (
        <mark key={i} className="bg-brand-500/25 text-brand-200 px-0.5 rounded font-medium">
          {part}
        </mark>
      ) : (
        part
      )
    );
  };

  if (results.length === 0) {
    return null;
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-white flex items-center space-x-2">
          <Sparkles className="w-4 h-4 text-brand-400" />
          <span>Top Matching Chunks ({results.length})</span>
        </h3>
        <span className="text-xs font-mono text-slate-400">Ranked by Reciprocal Rank Fusion / Similarity</span>
      </div>

      <div className="grid grid-cols-1 gap-3.5">
        {results.map((res, idx) => {
          const chunk = chunksMap.get(res.id);
          if (!chunk) return null;

          const scorePercent = Math.round(res.combinedScore * 100);

          return (
            <div
              key={res.id}
              className="bg-slate-900/80 border border-slate-800 hover:border-slate-700/90 rounded-xl p-4 transition-all duration-200 group"
            >
              {/* Card Header: Rank, Document Info, Score Badge */}
              <div className="flex items-start justify-between gap-3 mb-2.5">
                <div className="flex items-center space-x-2 flex-wrap gap-y-1">
                  <span className="w-6 h-6 rounded-md bg-slate-800 flex items-center justify-center font-mono font-bold text-xs text-brand-400 border border-slate-700">
                    #{idx + 1}
                  </span>

                  <div className="flex items-center space-x-1.5 text-xs font-mono text-slate-300">
                    <FileText className="w-3.5 h-3.5 text-slate-400" />
                    <span className="font-semibold text-white">{chunk.documentName}</span>
                    <span className="text-slate-500">•</span>
                    <span className="text-slate-400">
                      Chunk {chunk.chunkIndex} of {chunk.totalChunks}
                    </span>
                  </div>
                </div>

                {/* Badges & Scores */}
                <div className="flex items-center space-x-2 shrink-0 font-mono">
                  {res.denseScore !== undefined && (
                    <span
                      title={`Dense similarity: ${res.denseScore.toFixed(3)} (Rank #${res.denseRank})`}
                      className="hidden sm:flex items-center space-x-1 text-[10px] px-2 py-0.5 rounded-full bg-slate-950 text-slate-300 border border-slate-800"
                    >
                      <Compass className="w-3 h-3 text-brand-400" />
                      <span>Dense: {Math.round(res.denseScore * 100)}%</span>
                    </span>
                  )}

                  {res.sparseScore !== undefined && (
                    <span
                      title={`BM25 Score: ${res.sparseScore.toFixed(2)} (Rank #${res.sparseRank})`}
                      className="hidden sm:flex items-center space-x-1 text-[10px] px-2 py-0.5 rounded-full bg-slate-950 text-slate-300 border border-slate-800"
                    >
                      <Tag className="w-3 h-3 text-indigo-400" />
                      <span>BM25: {res.sparseScore.toFixed(1)}</span>
                    </span>
                  )}

                  {/* Main Combined Score Badge */}
                  <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-brand-950/80 text-brand-300 border border-brand-800/60 flex items-center space-x-1">
                    <span>{scorePercent}% match</span>
                  </span>
                </div>
              </div>

              {/* Chunk Text Body with Highlight */}
              <div className="text-xs text-slate-300 leading-relaxed font-mono bg-slate-950/60 p-3 rounded-lg border border-slate-850 line-clamp-4">
                {renderHighlightedText(chunk.text, query)}
              </div>

              {/* Footer Actions */}
              <div className="mt-3 flex items-center justify-between text-xs pt-2 border-t border-slate-800/50">
                <span className="text-[11px] text-slate-500 font-mono">
                  ~{chunk.tokenCountEstimate} tokens ({chunk.text.length} chars)
                </span>

                <div className="flex items-center space-x-2">
                  {onFindSimilar && (
                    <button
                      type="button"
                      onClick={() => onFindSimilar(chunk)}
                      className="flex items-center space-x-1 text-slate-400 hover:text-brand-300 px-2 py-1 rounded hover:bg-slate-800 transition-colors"
                      title="Run search using this chunk's embedding vector as the query"
                    >
                      <Compass className="w-3.5 h-3.5 text-brand-400" />
                      <span>Find Similar</span>
                    </button>
                  )}

                  <button
                    type="button"
                    onClick={() => handleCopy(res.id, chunk.text)}
                    className="flex items-center space-x-1 text-slate-400 hover:text-white px-2 py-1 rounded hover:bg-slate-800 transition-colors"
                  >
                    {copiedId === res.id ? (
                      <>
                        <Check className="w-3.5 h-3.5 text-brand-400" />
                        <span className="text-brand-400">Copied</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5" />
                        <span>Copy</span>
                      </>
                    )}
                  </button>

                  <button
                    type="button"
                    onClick={() => onInspectChunk(chunk, res)}
                    className="flex items-center space-x-1 text-brand-400 hover:text-brand-300 px-2 py-1 rounded hover:bg-brand-950/40 transition-colors"
                  >
                    <ExternalLink className="w-3.5 h-3.5" />
                    <span>Inspect Vector</span>
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
