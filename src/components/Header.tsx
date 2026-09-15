import React from 'react';
import { Database, Cpu, HardDrive, CheckCircle2, Loader2, Sparkles } from 'lucide-react';

interface HeaderProps {
  modelStatus: 'idle' | 'loading' | 'ready' | 'error';
  modelName: string;
  vectorCount: number;
  totalTokens: number;
  downloadProgress: { file: string; progress: number } | null;
}

export const Header: React.FC<HeaderProps> = ({
  modelStatus,
  modelName,
  vectorCount,
  totalTokens,
  downloadProgress,
}) => {
  return (
    <header className="border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Logo & Title */}
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-400 to-brand-600 flex items-center justify-center shadow-lg shadow-brand-500/20">
            <Database className="w-5 h-5 text-slate-950 stroke-[2.5]" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-lg font-bold tracking-tight text-white font-mono">
                VectorLite<span className="text-brand-400 font-sans">.wasm</span>
              </h1>
              <span className="text-[10px] font-mono font-medium px-2 py-0.5 rounded-full bg-slate-800 text-brand-300 border border-slate-700">
                v0.1.0 • Client Wasm
              </span>
            </div>
            <p className="text-xs text-slate-400">Zero-backend local semantic search & vector engine</p>
          </div>
        </div>

        {/* Telemetry Status Pills */}
        <div className="hidden md:flex items-center space-x-3 text-xs font-mono">
          {/* Model Status */}
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800">
            {modelStatus === 'loading' ? (
              <>
                <Loader2 className="w-3.5 h-3.5 text-amber-400 animate-spin" />
                <span className="text-slate-300">
                  {downloadProgress ? `${downloadProgress.file}: ${downloadProgress.progress}%` : 'Loading ONNX model...'}
                </span>
              </>
            ) : modelStatus === 'ready' ? (
              <>
                <CheckCircle2 className="w-3.5 h-3.5 text-brand-400" />
                <span className="text-slate-300">{modelName.split('/').pop()}</span>
                <span className="text-[10px] text-brand-400 bg-brand-950 px-1.5 py-0.2 rounded border border-brand-800/60">384-D</span>
              </>
            ) : (
              <>
                <Cpu className="w-3.5 h-3.5 text-slate-500" />
                <span className="text-slate-400">Model standby</span>
              </>
            )}
          </div>

          {/* Stored Vectors */}
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800">
            <Sparkles className="w-3.5 h-3.5 text-brand-400" />
            <span className="text-slate-400">Vectors:</span>
            <span className="text-white font-semibold">{vectorCount.toLocaleString()}</span>
          </div>

          {/* Tokens Indexed */}
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800">
            <HardDrive className="w-3.5 h-3.5 text-indigo-400" />
            <span className="text-slate-400">Tokens:</span>
            <span className="text-white font-semibold">~{totalTokens.toLocaleString()}</span>
          </div>
        </div>

        {/* External Links */}
        <div className="flex items-center space-x-2">
          <a
            href="https://github.com/alexandrmotologa/vectorlite-wasm"
            target="_blank"
            rel="noreferrer"
            className="flex items-center space-x-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 text-slate-300 transition-colors"
          >
            <svg className="w-3.5 h-3.5 fill-current" viewBox="0 0 24 24">
              <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
            </svg>
            <span>GitHub</span>
          </a>
        </div>
      </div>
    </header>
  );
};
