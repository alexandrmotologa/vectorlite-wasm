import React, { useState } from 'react';
import { Sparkles, Copy, Check, Quote, ShieldCheck } from 'lucide-react';
import { RAGAnswer } from '../engine/rag_synthesizer';

interface RAGAnswerBoxProps {
  answer: RAGAnswer | null;
  query: string;
}

export const RAGAnswerBox: React.FC<RAGAnswerBoxProps> = ({ answer, query }) => {
  const [copied, setCopied] = useState(false);

  if (!answer || !query.trim()) return null;

  const handleCopy = () => {
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

  return (
    <div className="bg-gradient-to-br from-slate-900 via-slate-900/90 to-brand-950/30 border border-brand-500/30 rounded-2xl p-5 shadow-xl relative overflow-hidden">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <div className="w-6 h-6 rounded-lg bg-brand-500/20 text-brand-400 flex items-center justify-center border border-brand-500/30">
            <Sparkles className="w-3.5 h-3.5" />
          </div>
          <h3 className="text-sm font-bold text-white tracking-wide">
            Synthesized Grounded Answer (Local RAG)
          </h3>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-brand-950 text-brand-300 border border-brand-800/60 flex items-center space-x-1">
            <ShieldCheck className="w-3 h-3 text-brand-400" />
            <span>{Math.round(answer.confidence * 100)}% confidence</span>
          </span>
        </div>

        <button
          type="button"
          onClick={handleCopy}
          className="flex items-center space-x-1 text-xs font-mono text-slate-400 hover:text-white px-2.5 py-1 rounded-lg hover:bg-slate-800 border border-transparent hover:border-slate-700 transition-colors"
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
      </div>

      {/* Main Answer Summary */}
      <div className="text-sm text-slate-200 leading-relaxed font-sans mb-3">
        {answer.summary}
      </div>

      {/* Additional Grounded Bullet Points */}
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

      {/* Citation Sources Bar */}
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
    </div>
  );
};
