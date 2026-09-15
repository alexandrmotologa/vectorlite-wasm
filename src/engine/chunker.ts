export interface TextChunk {
  id: string;
  documentName: string;
  chunkIndex: number;
  totalChunks: number;
  text: string;
  startChar: number;
  endChar: number;
  tokenCountEstimate: number;
}

export interface ChunkerOptions {
  chunkSize?: number;      // Maximum character count per chunk (default: 600)
  chunkOverlap?: number;   // Character overlap between consecutive chunks (default: 120)
  documentName?: string;   // Source document title or file name
}

/**
 * Hierarchical recursive character splitter.
 * Splitting precedence:
 * 1. Markdown H1/H2/H3 headers
 * 2. Paragraph breaks (\n\n)
 * 3. Line breaks (\n)
 * 4. Sentence terminators (. ! ?)
 * 5. Whitespace
 */
export class RecursiveChunker {
  private readonly chunkSize: number;
  private readonly chunkOverlap: number;

  constructor(options: ChunkerOptions = {}) {
    this.chunkSize = options.chunkSize ?? 600;
    this.chunkOverlap = options.chunkOverlap ?? 120;
  }

  public split(text: string, documentName = 'document.txt'): TextChunk[] {
    const rawChunks = this.recursiveSplit(text, [
      /(?=\n#{1,3}\s)/, // Markdown headings
      /\n\s*\n/,        // Paragraphs
      /\n/,             // Line breaks
      /(?<=[.!?])\s+/,  // Sentence boundaries
      /\s+/,            // Words
    ]);

    // Merge smaller pieces into chunks up to chunkSize with chunkOverlap
    const merged: string[] = [];
    let current = '';

    for (let i = 0; i < rawChunks.length; i++) {
      const piece = rawChunks[i].trim();
      if (!piece) continue;

      if (!current) {
        current = piece;
      } else if (current.length + 1 + piece.length <= this.chunkSize) {
        current += '\n' + piece;
      } else {
        merged.push(current);
        // Form overlap from end of current chunk
        if (this.chunkOverlap > 0 && current.length > this.chunkOverlap) {
          const overlapSlice = current.slice(current.length - this.chunkOverlap).trim();
          current = overlapSlice ? overlapSlice + '\n' + piece : piece;
        } else {
          current = piece;
        }
      }
    }

    if (current.trim().length > 0) {
      merged.push(current.trim());
    }

    // Build structured chunk objects
    let charCursor = 0;
    const totalChunks = merged.length;

    return merged.map((chunkText, idx) => {
      const startChar = text.indexOf(chunkText.slice(0, 30), charCursor);
      const actualStart = startChar !== -1 ? startChar : charCursor;
      const actualEnd = actualStart + chunkText.length;
      charCursor = actualEnd;

      // Estimate tokens: ~4 chars per token for English text
      const tokenCountEstimate = Math.max(1, Math.round(chunkText.length / 4));

      return {
        id: `${documentName}-chunk-${idx + 1}`,
        documentName,
        chunkIndex: idx + 1,
        totalChunks,
        text: chunkText,
        startChar: actualStart,
        endChar: actualEnd,
        tokenCountEstimate,
      };
    });
  }

  private recursiveSplit(text: string, separators: RegExp[]): string[] {
    if (separators.length === 0 || text.length <= this.chunkSize) {
      return [text];
    }

    const [currentSep, ...remainingSeps] = separators;
    const parts = text.split(currentSep);

    const result: string[] = [];
    for (const part of parts) {
      if (part.length > this.chunkSize && remainingSeps.length > 0) {
        result.push(...this.recursiveSplit(part, remainingSeps));
      } else {
        result.push(part);
      }
    }

    return result;
  }
}
