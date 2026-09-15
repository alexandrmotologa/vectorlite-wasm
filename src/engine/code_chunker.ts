import { TextChunk } from './chunker';

export interface CodeChunkerOptions {
  maxChunkLines?: number; // Maximum lines per code chunk (default: 45)
  overlapLines?: number;  // Overlap in lines (default: 8)
}

/**
 * Syntax-aware code chunker for programming languages and structured data.
 * Keeps functions, classes, struct declarations, and logical blocks intact.
 */
export class CodeAwareChunker {
  private readonly maxLines: number;
  private readonly overlapLines: number;

  constructor(options: CodeChunkerOptions = {}) {
    this.maxLines = options.maxChunkLines ?? 45;
    this.overlapLines = options.overlapLines ?? 8;
  }

  /**
   * Splits source code into contextually coherent chunks.
   */
  public splitCode(code: string, filename: string): TextChunk[] {
    const lines = code.split('\n');
    if (lines.length === 0) return [];

    const boundaries = this.detectBlockBoundaries(lines, filename);
    const rawChunks: string[] = [];

    let currentStart = 0;
    let bIdx = 0;

    while (currentStart < lines.length) {
      let targetEnd = Math.min(lines.length, currentStart + this.maxLines);

      // Try to snap to the closest natural block boundary before targetEnd
      while (bIdx < boundaries.length && boundaries[bIdx] <= targetEnd) {
        if (boundaries[bIdx] > currentStart + 15) {
          targetEnd = boundaries[bIdx];
        }
        bIdx++;
      }

      const chunkLines = lines.slice(currentStart, targetEnd);
      const chunkText = chunkLines.join('\n').trim();
      if (chunkText) {
        rawChunks.push(chunkText);
      }

      if (targetEnd >= lines.length) {
        break;
      }

      // Step forward with overlap
      currentStart = Math.max(currentStart + 1, targetEnd - this.overlapLines);
    }

    const totalChunks = rawChunks.length;
    let charCursor = 0;

    return rawChunks.map((text, idx) => {
      const startChar = code.indexOf(text.slice(0, 30), charCursor);
      const actualStart = startChar !== -1 ? startChar : charCursor;
      const actualEnd = actualStart + text.length;
      charCursor = actualEnd;

      return {
        id: `${filename}-code-${idx + 1}`,
        documentName: filename,
        chunkIndex: idx + 1,
        totalChunks,
        text,
        startChar: actualStart,
        endChar: actualEnd,
        tokenCountEstimate: Math.max(1, Math.round(text.length / 4)),
      };
    });
  }

  /**
   * Detects line numbers where new top-level declarations or blocks begin.
   */
  private detectBlockBoundaries(lines: number[] | string[], filename: string): number[] {
    const ext = filename.split('.').pop()?.toLowerCase() || '';
    const boundaries: number[] = [];

    // Language regex patterns for declarations
    const patterns = [
      /^(export\s+)?(async\s+)?function\s+/,
      /^(export\s+)?(const|let|var)\s+\w+\s*=\s*(async\s+)?\(/,
      /^(export\s+)?class\s+/,
      /^(export\s+)?interface\s+/,
      /^(export\s+)?type\s+\w+\s*=/,
      /^(pub\s+)?fn\s+/,             // Rust
      /^(pub\s+)?struct\s+/,         // Rust
      /^(pub\s+)?impl\s+/,           // Rust
      /^func\s+(\([^)]+\)\s+)?\w+/,  // Go
      /^type\s+\w+\s+struct/,        // Go
      /^def\s+\w+\s*\(/,             // Python
      /^class\s+\w+.*:/,             // Python
      /^@\w+/,                       // Decorators
    ];

    for (let i = 0; i < lines.length; i++) {
      const line = String(lines[i]).trim();
      if (!line) continue;

      // In JSON, array/object boundaries
      if (ext === 'json' && (line.startsWith('{') || line.startsWith('}') || line.startsWith('['))) {
        boundaries.push(i);
        continue;
      }

      for (const pat of patterns) {
        if (pat.test(line)) {
          boundaries.push(i);
          break;
        }
      }
    }

    return boundaries;
  }
}

export const CodeChunker = {
  isSupportedCodeFile(filename: string): boolean {
    const ext = filename.split('.').pop()?.toLowerCase();
    return [
      'ts',
      'tsx',
      'js',
      'jsx',
      'py',
      'rs',
      'go',
      'java',
      'c',
      'cpp',
      'h',
      'hpp',
      'html',
      'css',
      'json',
      'yaml',
      'yml',
    ].includes(ext || '');
  },

  split(code: string, filename: string, options?: CodeChunkerOptions): TextChunk[] {
    const chunker = new CodeAwareChunker(options);
    return chunker.splitCode(code, filename);
  },
};

