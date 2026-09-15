import { TextChunk } from './chunker';

export interface RAGCitation {
  index: number;
  chunkId: string;
  documentName: string;
  chunkIndex: number;
  excerpt: string;
}

export interface RAGAnswer {
  summary: string;
  bulletPoints: string[];
  citations: RAGCitation[];
  confidence: number;
}

/**
 * Client-side grounded RAG synthesizer.
 * Extracts and composes key authoritative evidence from top-K retrieved chunks
 * with verifiable citation anchors.
 */
export class LocalRAGSynthesizer {
  public static synthesize(query: string, chunks: TextChunk[]): RAGAnswer {
    if (chunks.length === 0) {
      return {
        summary: 'No relevant information found in the indexed documents to answer this query.',
        bulletPoints: [],
        citations: [],
        confidence: 0,
      };
    }

    const queryTerms = query
      .toLowerCase()
      .replace(/[^a-z0-9_\s]/g, ' ')
      .split(/\s+/)
      .filter((t) => t.length > 2);

    const citations: RAGCitation[] = [];
    const scoredSentences: Array<{
      sentence: string;
      score: number;
      citationIndex: number;
    }> = [];

    chunks.slice(0, 5).forEach((chunk, chunkIdx) => {
      const citationIndex = chunkIdx + 1;
      citations.push({
        index: citationIndex,
        chunkId: chunk.id,
        documentName: chunk.documentName,
        chunkIndex: chunk.chunkIndex,
        excerpt: chunk.text.slice(0, 180) + '...',
      });

      // Split into sentences
      const sentences = chunk.text
        .replace(/\n+/g, ' ')
        .split(/(?<=[.!?])\s+/)
        .map((s) => s.trim())
        .filter((s) => s.length > 20 && !s.startsWith('#'));

      for (const sent of sentences) {
        const lower = sent.toLowerCase();
        let termMatches = 0;
        for (const term of queryTerms) {
          if (lower.includes(term)) termMatches++;
        }

        if (termMatches > 0) {
          // Score based on term overlap density and rank of source chunk
          const density = termMatches / Math.max(1, queryTerms.length);
          const rankWeight = 1.0 / (1.0 + chunkIdx * 0.25);
          const score = density * rankWeight;

          scoredSentences.push({
            sentence: sent,
            score,
            citationIndex,
          });
        }
      }
    });

    // Sort by relevance score
    scoredSentences.sort((a, b) => b.score - a.score);

    // Pick top unique sentences
    const picked: Array<{ sentence: string; citationIndex: number }> = [];
    const seenPhrases = new Set<string>();

    for (const item of scoredSentences) {
      const key = item.sentence.slice(0, 40).toLowerCase();
      if (!seenPhrases.has(key)) {
        seenPhrases.add(key);
        picked.push({
          sentence: item.sentence,
          citationIndex: item.citationIndex,
        });
        if (picked.length >= 4) break;
      }
    }

    if (picked.length === 0) {
      // Fallback to first chunk's leading sentence
      const fallbackChunk = chunks[0];
      const lead = fallbackChunk.text.split(/(?<=[.!?])\s+/)[0] || fallbackChunk.text.slice(0, 150);
      return {
        summary: `${lead} [1]`,
        bulletPoints: [],
        citations: citations.slice(0, 1),
        confidence: 0.5,
      };
    }

    const summary = `${picked[0].sentence} [${picked[0].citationIndex}]`;
    const bulletPoints = picked.slice(1).map((p) => `${p.sentence} [${p.citationIndex}]`);

    return {
      summary,
      bulletPoints,
      citations,
      confidence: Math.min(0.98, 0.6 + picked.length * 0.1),
    };
  }
}
