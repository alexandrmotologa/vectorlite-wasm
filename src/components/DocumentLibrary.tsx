import React from 'react';
import { Library, Trash2, FileText, Code2, FileSpreadsheet } from 'lucide-react';

export interface DocumentInfo {
  name: string;
  chunkCount: number;
  tokenCount: number;
  addedAt: number;
}

interface DocumentLibraryProps {
  documents: DocumentInfo[];
  onDeleteDocument: (documentName: string) => void;
  isProcessing: boolean;
}

const getFileIcon = (name: string) => {
  if (/\.(ts|tsx|js|jsx|py|rs|go|java|c|cpp)$/.test(name)) return <Code2 className="w-3.5 h-3.5 text-amber-400" />;
  if (/\.pdf$/.test(name)) return <FileSpreadsheet className="w-3.5 h-3.5 text-red-400" />;
  return <FileText className="w-3.5 h-3.5 text-brand-400" />;
};

export const DocumentLibrary: React.FC<DocumentLibraryProps> = ({
  documents,
  onDeleteDocument,
  isProcessing,
}) => {
  if (documents.length === 0) return null;

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 backdrop-blur-sm">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <Library className="w-4.5 h-4.5 text-brand-400" />
          <h2 className="text-sm font-semibold text-white">Indexed Documents</h2>
        </div>
        <span className="text-xs font-mono text-slate-400 bg-slate-800/80 px-2 py-0.5 rounded-md">
          {documents.length} {documents.length === 1 ? 'document' : 'documents'}
        </span>
      </div>

      <div className="space-y-2">
        {documents.map((doc) => (
          <div
            key={doc.name}
            className="flex items-center justify-between p-2.5 rounded-lg bg-slate-950/60 border border-slate-800 hover:border-slate-700 transition-colors group"
          >
            <div className="flex items-center space-x-2.5 min-w-0">
              {getFileIcon(doc.name)}
              <div className="min-w-0">
                <p className="text-xs font-semibold text-slate-200 truncate">{doc.name}</p>
                <p className="text-[10px] font-mono text-slate-500">
                  {doc.chunkCount} chunks • ~{doc.tokenCount.toLocaleString()} tokens
                </p>
              </div>
            </div>
            <button
              type="button"
              disabled={isProcessing}
              onClick={() => {
                if (window.confirm(`Remove "${doc.name}" and its ${doc.chunkCount} chunks from the index?`)) {
                  onDeleteDocument(doc.name);
                }
              }}
              className="p-1.5 rounded-lg text-slate-500 hover:text-red-400 hover:bg-red-950/40 opacity-0 group-hover:opacity-100 transition-all disabled:opacity-30"
              title={`Delete ${doc.name}`}
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};
