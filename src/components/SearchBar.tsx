import React, { useState } from 'react';
import { Search, Sliders, Zap, Tag, Compass, Sparkles } from 'lucide-react';

export type SearchMode = 'hybrid' | 'dense' | 'sparse';

interface SearchBarProps {
  onSearch: (query: string, mode: SearchMode, topK: number) => void;
  isSearching: boolean;
  queryLatencyMs: number | null;
  totalVectors: number;
}

const SAMPLE_QUERIES = [
  'consensus algorithms and leader election failure',
  'difference between B-Trees and LSM-Trees',
  'HNSW hierarchical proximity graph traversal',
  'cache stampede mitigation and invalidation',
  'local client-side RAG without cloud leakage',
];

export const SearchBar: React.FC<SearchBarProps> = ({
  onSearch,
  isSearching,
  queryLatencyMs,
  totalVectors,
}) => {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<SearchMode>('hybrid');
  const [topK, setTopK] = useState(5);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    onSearch(query.trim(), mode, topK);
  };

  const handleSampleClick = (sample: string) => {
    setQuery(sample);
    onSearch(sample, mode, topK);
  };

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 backdrop-blur-sm">
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Search Input Bar */}
        <div className="relative flex items-center">
          <div className="absolute left-4 text-slate-400 pointer-events-none">
            <Search className="w-5 h-5 text-brand-400" />
          </div>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={
              totalVectors === 0
                ? 'Load a document or sample dataset above to start searching...'
                : 'Search concepts, architecture patterns, or exact keywords...'
            }
            className="w-full pl-12 pr-32 py-3.5 bg-slate-950/80 border border-slate-700/80 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-brand-500/50 focus:border-brand-500 font-sans text-sm transition-all"
          />

          <button
            type="submit"
            disabled={isSearching || !query.trim() || totalVectors === 0}
            className="absolute right-2 px-4 py-2 rounded-lg bg-gradient-to-r from-brand-500 to-brand-600 hover:from-brand-400 hover:to-brand-500 text-slate-950 font-semibold text-xs transition-all shadow-md shadow-brand-500/20 disabled:opacity-50 disabled:pointer-events-none flex items-center space-x-1.5"
          >
            {isSearching ? (
              <span>Searching...</span>
            ) : (
              <>
                <Zap className="w-3.5 h-3.5 fill-slate-950" />
                <span>Search</span>
              </>
            )}
          </button>
        </div>

        {/* Query Controls: Mode Toggle & Top-K */}
        <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
          {/* Search Mode Toggles */}
          <div className="flex items-center space-x-1 bg-slate-950 p-1 rounded-xl border border-slate-800">
            <button
              type="button"
              onClick={() => setMode('hybrid')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg font-medium transition-all ${
                mode === 'hybrid'
                  ? 'bg-brand-500 text-slate-950 font-semibold shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>Hybrid (Dense + BM25)</span>
            </button>

            <button
              type="button"
              onClick={() => setMode('dense')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg font-medium transition-all ${
                mode === 'dense'
                  ? 'bg-brand-500 text-slate-950 font-semibold shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Compass className="w-3.5 h-3.5" />
              <span>Dense HNSW</span>
            </button>

            <button
              type="button"
              onClick={() => setMode('sparse')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg font-medium transition-all ${
                mode === 'sparse'
                  ? 'bg-brand-500 text-slate-950 font-semibold shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Tag className="w-3.5 h-3.5" />
              <span>Sparse BM25</span>
            </button>
          </div>

          {/* Top-K Selector & Latency Pill */}
          <div className="flex items-center space-x-3 font-mono">
            <div className="flex items-center space-x-2 text-slate-400 bg-slate-950 px-3 py-1.5 rounded-xl border border-slate-800">
              <Sliders className="w-3.5 h-3.5 text-slate-400" />
              <span>Top-K:</span>
              <select
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="bg-transparent text-white font-semibold focus:outline-none cursor-pointer"
              >
                <option value={3} className="bg-slate-900 text-white">3</option>
                <option value={5} className="bg-slate-900 text-white">5</option>
                <option value={10} className="bg-slate-900 text-white">10</option>
                <option value={20} className="bg-slate-900 text-white">20</option>
              </select>
            </div>

            {queryLatencyMs !== null && (
              <div className="px-2.5 py-1.5 rounded-xl bg-brand-950/60 border border-brand-800/40 text-brand-300 text-[11px] flex items-center space-x-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-brand-400 animate-pulse" />
                <span>Latency: {queryLatencyMs.toFixed(2)} ms</span>
              </div>
            )}
          </div>
        </div>

        {/* Quick Sample Queries */}
        <div className="pt-2 flex items-center space-x-2 overflow-x-auto text-xs pb-1">
          <span className="text-slate-500 shrink-0">Try queries:</span>
          {SAMPLE_QUERIES.map((sq, i) => (
            <button
              key={i}
              type="button"
              onClick={() => handleSampleClick(sq)}
              className="shrink-0 px-2.5 py-1 rounded-lg bg-slate-950 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 text-slate-400 hover:text-slate-200 transition-colors"
            >
              {sq}
            </button>
          ))}
        </div>
      </form>
    </div>
  );
};
