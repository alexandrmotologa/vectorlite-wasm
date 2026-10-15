import React, { useState } from 'react';
import { Sparkles, Copy, Check, Quote, Loader2, RefreshCw, Settings } from 'lucide-react';
import { RAGAnswer } from '../engine/rag_synthesizer';

interface RAGAnswerBoxProps {
  answer: RAGAnswer | null;
  query: string;
  isGenerating?: boolean;
  generationStatus?: string;
  activeMode?: 'in_browser_llm' | 'ollama' | 'extractive';
  onChangeMode?: (mode: 'in_browser_llm' | 'ollama' | 'extractive') => void;
  onRegenerate?: () => void;
  ollamaEndpoint?: string;
  ollamaModel?: string;
  onOllamaEndpointChange?: (endpoint: string) => void;
  onOllamaModelChange?: (model: string) => void;
  history?: Array<{ role: 'user' | 'assistant'; content: string }>;
}

export const RAGAnswerBox: React.FC<RAGAnswerBoxProps> = ({
  answer,
  query,
  isGenerating = false,
  generationStatus = '',
  activeMode = 'in_browser_llm',
  onChangeMode,
  onRegenerate,
  ollamaEndpoint,
  ollamaModel,
  onOllamaEndpointChange,
  onOllamaModelChange,
  history,
}) => {
  const [copied, setCopied] = useState(false);
  const [showOllamaSettings, setShowOllamaSettings] = useState(false);

  if (!query.trim() && !isGenerating && !answer) return null;

  const handleCopy = () => {
    if (!answer) return;
    let text = `${answer.summary}\n\n`;
    if (answer.bulletPoints.length > 0) {
      text += answer.bulletPoints.map((bp) => `• ${bp}`).join('\n') + '\n\n';
    }
    text += 'Sources:\n';
    answer.citations.forEach((c) => {
      text += `[${c.index}] ${c.documentName} (Chunk #${c.chunkIndex})\n`;
    });

    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const getBadgeStyle = (backend?: string) => {
    switch (backend) {
      case 'in_browser_llm':
        return 'bg-emerald-950/80 text-emerald-300 border-emerald-800/60';
      case 'ollama':
        return 'bg-purple-950/80 text-purple-300 border-purple-800/60';
      default:
        return 'bg-slate-900 text-slate-300 border-slate-700/60';
    }
  };

  return (
    <div className="bg-gradient-to-br from-slate-900 via-slate-900/90 to-brand-950/30 border border-brand-500/30 rounded-2xl p-5 shadow-xl relative overflow-hidden">
      {/* Header Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3 pb-3 border-b border-slate-800/60">
        <div className="flex items-center space-x-2">
          <div className="w-6 h-6 rounded-lg bg-brand-500/20 text-brand-400 flex items-center justify-center border border-brand-500/30">
            <Sparkles className="w-3.5 h-3.5" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white tracking-wide flex items-center space-x-2">
              <span>Grounded RAG Synthesis</span>
              {answer?.modelTitle && (
                <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${getBadgeStyle(answer.backendUsed)}`}>
                  {answer.modelTitle}
                </span>
              )}
            </h3>
          </div>
        </div>

        {/* Mode Selector & Action Buttons */}
        <div className="flex items-center space-x-2">
          {onChangeMode && (
            <div className="flex items-center space-x-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-[11px] font-mono">
              <button
                type="button"
                onClick={() => onChangeMode('in_browser_llm')}
                className={`px-2 py-0.5 rounded transition-colors ${
                  activeMode === 'in_browser_llm'
                    ? 'bg-slate-800 text-brand-300 font-semibold shadow-sm'
                    : 'text-slate-400 hover:text-white'
                }`}
                title="Run in-browser quantized ONNX generative LLM (LaMini-Flan-T5)"
              >
                ONNX LLM
              </button>
              <button
                type="button"
                onClick={() => onChangeMode('ollama')}
                className={`px-2 py-0.5 rounded transition-colors ${
                  activeMode === 'ollama'
                    ? 'bg-slate-800 text-purple-300 font-semibold shadow-sm'
                    : 'text-slate-400 hover:text-white'
                }`}
                title="Route generation to local Ollama daemon (e.g. Llama 3 / Mistral)"
              >
                Ollama
              </button>
              <button
                type="button"
                onClick={() => onChangeMode('extractive')}
                className={`px-2 py-0.5 rounded transition-colors ${
                  activeMode === 'extractive'
                    ? 'bg-slate-800 text-slate-200 font-semibold shadow-sm'
                    : 'text-slate-400 hover:text-white'
                }`}
                title="Instant extractive sentence salience ranker (zero download)"
              >
                Extractive
              </button>
            </div>
          )}

          {activeMode === 'ollama' && onOllamaEndpointChange && (
            <button
              type="button"
              onClick={() => setShowOllamaSettings(!showOllamaSettings)}
              className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
              title="Ollama connection settings"
            >
              <Settings className="w-3.5 h-3.5" />
            </button>
          )}

          {onRegenerate && (
            <button
              type="button"
              disabled={isGenerating}
              onClick={onRegenerate}
              className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors disabled:opacity-50"
              title="Regenerate answer"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isGenerating ? 'animate-spin text-brand-400' : ''}`} />
            </button>
          )}

          {answer && (
            <button
              type="button"
              onClick={handleCopy}
              className="flex items-center space-x-1 text-xs font-mono text-slate-400 hover:text-white px-2 py-1 rounded-lg hover:bg-slate-800 border border-transparent hover:border-slate-700 transition-colors"
            >
              {copied ? (
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
          )}
        </div>
      </div>

      {showOllamaSettings && activeMode === 'ollama' && (
        <div className="mb-3 p-3 rounded-lg bg-slate-950/80 border border-slate-800 space-y-2 text-xs font-mono">
          <div className="flex items-center space-x-2">
            <label className="text-slate-400 shrink-0 w-16">Endpoint:</label>
            <input
              type="text"
              value={ollamaEndpoint || 'http://localhost:11434'}
              onChange={(e) => onOllamaEndpointChange?.(e.target.value)}
              className="flex-1 px-2 py-1 bg-slate-900 border border-slate-700 rounded text-white placeholder-slate-500 focus:outline-none focus:border-brand-400"
              placeholder="http://localhost:11434"
            />
          </div>
          <div className="flex items-center space-x-2">
            <label className="text-slate-400 shrink-0 w-16">Model:</label>
            <input
              type="text"
              value={ollamaModel || 'llama3.2'}
              onChange={(e) => onOllamaModelChange?.(e.target.value)}
              className="flex-1 px-2 py-1 bg-slate-900 border border-slate-700 rounded text-white placeholder-slate-500 focus:outline-none focus:border-brand-400"
              placeholder="llama3.2"
            />
          </div>
          <p className="text-[10px] text-slate-500">Ollama must be running locally. Supported models: llama3.2, mistral, gemma2, phi3, etc.</p>
        </div>
      )}

      {/* Loading State */}
      {isGenerating && (
        <div className="py-3 space-y-2 font-mono text-xs">
          {generationStatus && generationStatus.length > 40 ? (
            <div className="text-sm text-slate-200 leading-relaxed font-sans whitespace-pre-wrap">
              {generationStatus}
              <span className="inline-block w-2 h-4 bg-brand-400 animate-pulse ml-0.5 align-text-bottom" />
            </div>
          ) : (
            <div className="flex items-center justify-center space-x-2 text-brand-300">
              <Loader2 className="w-4 h-4 animate-spin text-brand-400" />
              <span>{generationStatus || 'Synthesizing grounded answer with local LLM...'}</span>
            </div>
          )}
          <p className="text-[11px] text-slate-500 text-center">
            Retrieval complete. Processing context evidence inside Web Worker.
          </p>
        </div>
      )}

      {/* Answer Body */}
      {!isGenerating && answer && (
        <>
          {history && history.length > 2 && (
            <details className="mb-3 text-xs">
              <summary className="text-slate-500 cursor-pointer hover:text-slate-300 font-mono">
                Previous conversation ({Math.floor(history.length / 2)} exchanges)
              </summary>
              <div className="mt-2 space-y-1.5 max-h-32 overflow-y-auto pr-1">
                {history.slice(0, -2).map((h, i) => (
                  <div key={i} className={`text-[11px] font-mono px-2 py-1 rounded ${h.role === 'user' ? 'text-brand-300 bg-brand-950/30' : 'text-slate-400 bg-slate-950/50'}`}>
                    <span className="font-bold">{h.role === 'user' ? 'Q:' : 'A:'}</span> {h.content.slice(0, 200)}{h.content.length > 200 ? '...' : ''}
                  </div>
                ))}
              </div>
            </details>
          )}

          <div className="text-sm text-slate-200 leading-relaxed font-sans mb-3">
            {answer.summary}
          </div>

          {answer.bulletPoints.length > 0 && (
            <ul className="space-y-1.5 mb-4 text-xs text-slate-300">
              {answer.bulletPoints.map((bp, i) => (
                <li key={i} className="flex items-start space-x-2">
                  <span className="text-brand-400 font-bold">•</span>
                  <span>{bp}</span>
                </li>
              ))}
            </ul>
          )}

          {/* Citations Footer */}
          {answer.citations.length > 0 && (
            <div className="pt-3 border-t border-slate-800/70 flex items-center flex-wrap gap-2 text-xs font-mono">
              <span className="text-slate-500 text-[11px] flex items-center space-x-1 mr-1">
                <Quote className="w-3 h-3 text-slate-400" />
                <span>Citations:</span>
              </span>
              {answer.citations.map((c) => (
                <div
                  key={c.index}
                  className="flex items-center space-x-1.5 px-2 py-0.5 rounded-md bg-slate-950/80 border border-slate-800 text-[11px] text-slate-300"
                >
                  <span className="text-brand-400 font-bold">[{c.index}]</span>
                  <span className="truncate max-w-[150px]">{c.documentName}</span>
                  <span className="text-slate-500">#{c.chunkIndex}</span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
};
