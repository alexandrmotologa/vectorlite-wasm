import React, { useState } from 'react';
import { Activity, Gauge, Cpu, CheckCircle2, Play, Zap } from 'lucide-react';
import { HNSWIndex } from '../engine/hnsw_index';

interface BenchmarkStatsProps {
  hnsw: HNSWIndex;
  lastLatencyMs: number | null;
  totalTokens: number;
}

interface BenchmarkResult {
  iterations: number;
  avgLatencyMs: number;
  p50Ms: number;
  p95Ms: number;
  p99Ms: number;
  qps: number;
}

export const BenchmarkStats: React.FC<BenchmarkStatsProps> = ({
  hnsw,
  lastLatencyMs,
  totalTokens,
}) => {
  const [isRunning, setIsRunning] = useState(false);
  const [benchResult, setBenchResult] = useState<BenchmarkResult | null>(null);

  // Estimate in-memory RAM size
  const vectorMemoryBytes = hnsw.size * 384 * 4; // Float32Array
  const graphLinksBytes = hnsw.size * 16 * 4 * 2; // Average 16 neighbors per node (4 bytes int)
  const totalKb = Math.round((vectorMemoryBytes + graphLinksBytes) / 1024);

  const runBenchmark = () => {
    if (hnsw.size === 0) return;
    setIsRunning(true);

    setTimeout(() => {
      const iterations = 100;
      const latencies: number[] = [];
      const dim = 384;

      for (let i = 0; i < iterations; i++) {
        // Generate test vector
        const q = new Float32Array(dim);
        for (let d = 0; d < dim; d++) {
          q[d] = Math.sin(i * 17 + d);
        }

        const t0 = performance.now();
        hnsw.search(q, 5);
        const t1 = performance.now();
        latencies.push(t1 - t0);
      }

      latencies.sort((a, b) => a - b);
      const totalTime = latencies.reduce((sum, val) => sum + val, 0);
      const avg = totalTime / iterations;
      const p50 = latencies[Math.floor(iterations * 0.5)];
      const p95 = latencies[Math.floor(iterations * 0.95)];
      const p99 = latencies[Math.floor(iterations * 0.99)];
      const qps = Math.round((iterations / totalTime) * 1000);

      setBenchResult({
        iterations,
        avgLatencyMs: avg,
        p50Ms: p50,
        p95Ms: p95,
        p99Ms: p99,
        qps,
      });
      setIsRunning(false);
    }, 50);
  };

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 backdrop-blur-sm">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Activity className="w-5 h-5 text-brand-400" />
          <h2 className="text-base font-semibold text-white">Engine Performance & Micro-Benchmarks</h2>
        </div>

        <button
          type="button"
          disabled={isRunning || hnsw.size === 0}
          onClick={runBenchmark}
          className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-brand-300 hover:text-brand-200 transition-colors border border-slate-700 disabled:opacity-50"
        >
          <Play className="w-3.5 h-3.5 fill-current" />
          <span>{isRunning ? 'Benchmarking...' : 'Run 100-Query Test'}</span>
        </button>
      </div>

      {/* Grid Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
        <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
          <div className="flex items-center space-x-1.5 text-slate-500 mb-1">
            <Gauge className="w-3.5 h-3.5 text-brand-400" />
            <span>Search Latency</span>
          </div>
          <span className="text-sm font-bold text-white">
            {lastLatencyMs !== null ? `${lastLatencyMs.toFixed(2)} ms` : '< 2 ms'}
          </span>
          <span className="text-[10px] text-slate-500 block mt-0.5">Target: &lt; 10ms</span>
        </div>

        <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
          <div className="flex items-center space-x-1.5 text-slate-500 mb-1">
            <Cpu className="w-3.5 h-3.5 text-indigo-400" />
            <span>HNSW Parameters</span>
          </div>
          <span className="text-sm font-bold text-white">
            M={hnsw.M} • M0={hnsw.M0}
          </span>
          <span className="text-[10px] text-slate-500 block mt-0.5">efConst: {hnsw.efConstruction}</span>
        </div>

        <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
          <div className="flex items-center space-x-1.5 text-slate-500 mb-1">
            <Zap className="w-3.5 h-3.5 text-amber-400" />
            <span>Memory Footprint</span>
          </div>
          <span className="text-sm font-bold text-white">
            {totalKb > 0 ? `${totalKb} KB` : 'Minimal'}
          </span>
          <span className="text-[10px] text-slate-500 block mt-0.5">In-memory graph</span>
        </div>

        <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
          <div className="flex items-center space-x-1.5 text-slate-500 mb-1">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            <span>Max Graph Level</span>
          </div>
          <span className="text-sm font-bold text-white">
            Layer {hnsw.currentMaxLevel >= 0 ? hnsw.currentMaxLevel : 0}
          </span>
          <span className="text-[10px] text-slate-500 block mt-0.5">
            {hnsw.size} nodes • ~{totalTokens} tokens
          </span>
        </div>
      </div>

      {/* Benchmark Results Card */}
      {benchResult && (
        <div className="mt-4 p-4 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs">
          <div className="flex items-center justify-between mb-3 text-slate-400 pb-2 border-b border-slate-850">
            <span className="text-brand-300 font-bold">100-Query Latency Distribution</span>
            <span className="text-slate-500">Throughput: {benchResult.qps.toLocaleString()} QPS</span>
          </div>

          <div className="grid grid-cols-4 gap-2 text-center">
            <div className="bg-slate-900/60 p-2 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-500 block">Avg</span>
              <span className="text-white font-bold">{benchResult.avgLatencyMs.toFixed(3)} ms</span>
            </div>
            <div className="bg-slate-900/60 p-2 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-500 block">p50</span>
              <span className="text-brand-300 font-bold">{benchResult.p50Ms.toFixed(3)} ms</span>
            </div>
            <div className="bg-slate-900/60 p-2 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-500 block">p95</span>
              <span className="text-amber-300 font-bold">{benchResult.p95Ms.toFixed(3)} ms</span>
            </div>
            <div className="bg-slate-900/60 p-2 rounded-lg border border-slate-800">
              <span className="text-[10px] text-slate-500 block">p99</span>
              <span className="text-red-400 font-bold">{benchResult.p99Ms.toFixed(3)} ms</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
