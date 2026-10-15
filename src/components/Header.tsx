import React from 'react';
import { Database, Cpu, HardDrive, CheckCircle2, Loader2, Sparkles, Zap, AlertTriangle, Sun, Moon } from 'lucide-react';

interface HeaderProps {
  modelStatus: 'idle' | 'loading' | 'ready' | 'error';
  modelName: string;
  vectorCount: number;
  totalTokens: number;
  downloadProgress: { file: string; progress: number } | null;
  onSelectModel: (modelName: string) => void;
  isFallback?: boolean;
  isWasmSimd?: boolean;
  hasWebGpu?: boolean;
  embeddingDevice?: 'wasm' | 'webgpu';
  onToggleDevice?: () => void;
  theme?: 'dark' | 'light';
  onToggleTheme?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  modelStatus,
  modelName,
  vectorCount,
  totalTokens,
  downloadProgress,
  onSelectModel,
  isFallback = false,
  isWasmSimd = true,
  hasWebGpu = false,
  embeddingDevice = 'wasm',
  onToggleDevice,
  theme = 'dark',
  onToggleTheme,
}) => {
  const isLight = theme === 'light';

  return (
    <header className={`border-b sticky top-0 z-40 transition-colors backdrop-blur-md ${
      isLight
        ? 'bg-white/95 border-slate-200 text-slate-900 shadow-xs'
        : 'bg-slate-950/90 border-slate-800/80 text-slate-100'
    }`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 min-h-16 py-2.5 flex items-center justify-between gap-3">
        {/* Logo & Title */}
        <div className="flex items-center space-x-3 shrink-0">
          <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-gradient-to-br from-brand-400 to-brand-600 flex items-center justify-center shadow-md shadow-brand-500/20 shrink-0">
            <Database className="w-5 h-5 text-slate-950 stroke-[2.5]" />
          </div>
          <div className="min-w-0">
            <div className="flex items-center space-x-2">
              <h1 className={`text-base sm:text-lg font-bold tracking-tight font-mono whitespace-nowrap ${
                isLight ? 'text-slate-900' : 'text-white'
              }`}>
                VectorLite<span className="text-brand-500 font-sans">.wasm</span>
              </h1>
              <span className={`text-[10px] font-mono font-medium px-2 py-0.5 rounded-full border whitespace-nowrap shrink-0 ${
                isLight
                  ? 'bg-slate-100 text-brand-700 border-slate-200'
                  : 'bg-slate-800 text-brand-300 border-slate-700'
              }`}>
                v0.3.0 • Genuine Wasm
              </span>
            </div>
            <p className={`text-[11px] sm:text-xs truncate max-w-[200px] sm:max-w-none hidden xs:block ${
              isLight ? 'text-slate-500' : 'text-slate-400'
            }`}>
              Zero-backend local semantic search & generative RAG
            </p>
          </div>
        </div>

        {/* Desktop Telemetry Pills & Model Selector (visible on md+) */}
        <div className="hidden md:flex items-center space-x-1.5 lg:space-x-2 text-xs font-mono shrink-0">
          {/* WebAssembly SIMD Badge */}
          <div
            className={`flex items-center space-x-1 px-2 py-1 lg:px-2.5 lg:py-1.5 rounded-lg border text-[11px] whitespace-nowrap transition-colors ${
              isLight
                ? isWasmSimd ? 'bg-indigo-50 border-indigo-200 text-indigo-700' : 'bg-slate-100 border-slate-200 text-slate-600'
                : isWasmSimd ? 'bg-slate-900 border-indigo-500/40 text-indigo-300' : 'bg-slate-900 border-slate-800 text-slate-400'
            }`}
            title="128-bit hardware WebAssembly SIMD vector execution"
          >
            <Zap className={`w-3.5 h-3.5 ${isWasmSimd ? 'text-indigo-500 fill-indigo-500/20' : 'text-slate-400'}`} />
            <span><span className="hidden xl:inline">WASM </span>SIMD: {isWasmSimd ? 'Active' : 'Unrolled'}</span>
          </div>

          {/* WebGPU Hardware Acceleration Badge */}
          {hasWebGpu ? (
            <button
              type="button"
              onClick={onToggleDevice}
              className={`flex items-center space-x-1 px-2 py-1 lg:px-2.5 lg:py-1.5 rounded-lg border text-[11px] transition-colors cursor-pointer whitespace-nowrap ${
                isLight
                  ? embeddingDevice === 'webgpu'
                    ? 'bg-amber-50 border-amber-300 text-amber-800 hover:bg-amber-100'
                    : 'bg-slate-100 border-slate-200 text-slate-600 hover:border-slate-300'
                  : embeddingDevice === 'webgpu'
                    ? 'bg-slate-900 border-amber-500/50 text-amber-300 hover:bg-slate-800/80'
                    : 'bg-slate-900 border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200'
              }`}
              title={
                embeddingDevice === 'webgpu'
                  ? 'WebGPU active for matrix operations. Click to switch to CPU/WASM'
                  : 'WebGPU supported. Click to enable WebGPU hardware acceleration'
              }
            >
              {embeddingDevice === 'webgpu' ? (
                <Zap className="w-3.5 h-3.5 text-amber-500 fill-amber-500/20" />
              ) : (
                <Cpu className="w-3.5 h-3.5 text-slate-400" />
              )}
              <span>{embeddingDevice === 'webgpu' ? 'WebGPU Active' : 'WebGPU Ready'}</span>
            </button>
          ) : (
            <div
              className={`flex items-center space-x-1 px-2 py-1 lg:px-2.5 lg:py-1.5 rounded-lg border text-[11px] whitespace-nowrap ${
                isLight ? 'bg-slate-100 border-slate-200 text-slate-600' : 'bg-slate-900 border-slate-800 text-slate-500'
              }`}
              title="WebGPU not detected in this browser environment. Using CPU/WASM."
            >
              <Cpu className="w-3.5 h-3.5 text-slate-400" />
              <span>CPU/WASM</span>
            </div>
          )}

          {/* Model Selector Dropdown & Real Status */}
          <div className={`flex items-center space-x-1.5 px-2 py-1 lg:px-2.5 lg:py-1.5 rounded-lg border ${
            isLight ? 'bg-slate-100 border-slate-200' : 'bg-slate-900 border-slate-800'
          }`}>
            {modelStatus === 'loading' ? (
              <>
                <Loader2 className="w-3.5 h-3.5 text-amber-500 animate-spin" />
                <span className={isLight ? 'text-slate-700' : 'text-slate-300'}>
                  {downloadProgress ? `${downloadProgress.file}: ${downloadProgress.progress}%` : 'Loading ONNX...'}
                </span>
              </>
            ) : (
              <>
                {isFallback ? (
                  <span title="Running in offline N-gram hash fallback">
                    <AlertTriangle className="w-3.5 h-3.5 text-amber-500 shrink-0" />
                  </span>
                ) : modelStatus === 'ready' ? (
                  <CheckCircle2 className="w-3.5 h-3.5 text-brand-500 shrink-0" />
                ) : (
                  <Cpu className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                )}
                <select
                  value={modelName}
                  onChange={(e) => onSelectModel(e.target.value)}
                  className={`bg-transparent font-semibold focus:outline-none cursor-pointer text-xs max-w-[130px] sm:max-w-[160px] lg:max-w-[190px] truncate ${
                    isLight ? 'text-slate-800 hover:text-black' : 'text-slate-200 hover:text-white'
                  }`}
                  title="Select embedding model"
                >
                  <option value="Xenova/all-MiniLM-L6-v2" className={isLight ? 'bg-white text-slate-900' : 'bg-slate-900 text-white'}>
                    all-MiniLM-L6-v2 (384-D)
                  </option>
                  <option value="Xenova/bge-small-en-v1.5" className={isLight ? 'bg-white text-slate-900' : 'bg-slate-900 text-white'}>
                    bge-small-en-v1.5
                  </option>
                  <option value="Xenova/multilingual-e5-small" className={isLight ? 'bg-white text-slate-900' : 'bg-slate-900 text-white'}>
                    multilingual-e5-small
                  </option>
                </select>
                {isFallback && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded font-medium bg-amber-100 dark:bg-amber-950/80 text-amber-800 dark:text-amber-300 border border-amber-300 dark:border-amber-800/60">
                    Offline
                  </span>
                )}
              </>
            )}
          </div>

          {/* Stored Vectors & Tokens - Always visible on md+ */}
          <div className={`flex items-center space-x-1.5 px-2 py-1 lg:px-2.5 lg:py-1.5 rounded-lg border ${
            isLight ? 'bg-slate-100 border-slate-200' : 'bg-slate-900 border-slate-800'
          }`}>
            <Sparkles className="w-3.5 h-3.5 text-brand-500 shrink-0" />
            <span className={`hidden xl:inline ${isLight ? 'text-slate-600' : 'text-slate-400'}`}>Vectors:</span>
            <span className={`font-semibold ${isLight ? 'text-slate-900' : 'text-white'}`}>{vectorCount.toLocaleString()}</span>
            <span className={`xl:hidden text-[10px] ${isLight ? 'text-slate-500' : 'text-slate-400'}`}>vecs</span>
          </div>

          <div className={`flex items-center space-x-1.5 px-2 py-1 lg:px-2.5 lg:py-1.5 rounded-lg border ${
            isLight ? 'bg-slate-100 border-slate-200' : 'bg-slate-900 border-slate-800'
          }`}>
            <HardDrive className="w-3.5 h-3.5 text-indigo-500 shrink-0" />
            <span className={`hidden xl:inline ${isLight ? 'text-slate-600' : 'text-slate-400'}`}>Tokens:</span>
            <span className={`font-semibold ${isLight ? 'text-slate-900' : 'text-white'}`}>~{totalTokens >= 1000 ? `${(totalTokens / 1000).toFixed(1)}k` : totalTokens}</span>
            <span className={`xl:hidden text-[10px] ${isLight ? 'text-slate-500' : 'text-slate-400'}`}>tok</span>
          </div>
        </div>

        {/* Right Actions: Compact info for tablet/mobile + Theme Toggle & GitHub */}
        <div className="flex items-center space-x-1.5 sm:space-x-2 shrink-0">
          {/* Mobile/Small Tablet Compact Vector Count */}
          <div className={`md:hidden flex items-center space-x-1.5 px-2 py-1 rounded-lg border text-xs font-mono ${
            isLight ? 'bg-slate-100 border-slate-200 text-slate-700' : 'bg-slate-900 border-slate-800 text-slate-300'
          }`}>
            <Sparkles className="w-3 h-3 text-brand-500" />
            <span className="font-semibold">{vectorCount}v</span>
            <span className="text-slate-400 dark:text-slate-600">•</span>
            <span className="text-slate-500 dark:text-slate-400">~{totalTokens >= 1000 ? `${Math.round(totalTokens / 1000)}k` : totalTokens}t</span>
          </div>

          {onToggleTheme && (
            <button
              type="button"
              onClick={onToggleTheme}
              className={`p-2 rounded-lg transition-colors border ${
                isLight
                  ? 'bg-slate-100 hover:bg-slate-200 border-slate-200 text-slate-700 hover:text-black'
                  : 'bg-slate-900 hover:bg-slate-800 border-slate-800 text-slate-400 hover:text-white'
              }`}
              title={`Switch to ${isLight ? 'dark' : 'light'} mode`}
            >
              {isLight ? <Moon className="w-4 h-4 text-slate-700" /> : <Sun className="w-4 h-4 text-amber-400" />}
            </button>
          )}

          <a
            href="https://github.com/alexandrmotologa/vectorlite-wasm"
            target="_blank"
            rel="noreferrer"
            className={`flex items-center space-x-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border transition-colors ${
              isLight
                ? 'bg-slate-100 hover:bg-slate-200 border-slate-200 text-slate-800 hover:text-black'
                : 'bg-slate-900 hover:bg-slate-800 border-slate-800 text-slate-300 hover:text-white'
            }`}
          >
            <svg className="w-3.5 h-3.5 fill-current" viewBox="0 0 24 24">
              <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
            </svg>
            <span className="hidden sm:inline">GitHub</span>
          </a>
        </div>
      </div>
    </header>
  );
};
