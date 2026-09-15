/**
 * In-memory Okapi BM25 ranking algorithm for sparse lexical search.
 * Provides exact keyword, function name, and token matching to complement dense embeddings.
 */

const ENGLISH_STOPWORDS = new Set([
  'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t', 'as', 'at',
  'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'can\'t', 'cannot',
  'could', 'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during', 'each',
  'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d',
  'he\'ll', 'he\'s', 'her', 'here', 'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i',
  'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself', 'let\'s',
  'me', 'more', 'most', 'mustn\'t', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or',
  'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll',
  'she\'s', 'should', 'shouldn\'t', 'so', 'some', 'such', 'than', 'that', 'that\'s', 'the', 'their', 'theirs',
  'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve',
  'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll',
  'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which', 'while',
  'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll',
  'you\'re', 'you\'ve', 'your', 'yours', 'yourself', 'yourselves'
]);

export interface BM25SearchResult {
  id: string;
  score: number;
}

export class BM25Index {
  private readonly k1: number;
  private readonly b: number;

  private docLengths: Map<string, number> = new Map();
  private totalDocLength = 0;
  // term -> (docId -> frequency)
  private invertedIndex: Map<string, Map<string, number>> = new Map();

  constructor(k1 = 1.2, b = 0.75) {
    this.k1 = k1;
    this.b = b;
  }

  public get size(): number {
    return this.docLengths.size;
  }

  public tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^a-z0-9_\-\s]/g, ' ')
      .split(/\s+/)
      .filter((t) => t.length > 1 && !ENGLISH_STOPWORDS.has(t));
  }

  public addDocument(id: string, text: string): void {
    const tokens = this.tokenize(text);
    if (this.docLengths.has(id)) {
      this.removeDocument(id);
    }

    const docLen = tokens.length;
    this.docLengths.set(id, docLen);
    this.totalDocLength += docLen;

    const termFreqs = new Map<string, number>();
    for (const token of tokens) {
      termFreqs.set(token, (termFreqs.get(token) || 0) + 1);
    }

    for (const [term, freq] of termFreqs.entries()) {
      let posting = this.invertedIndex.get(term);
      if (!posting) {
        posting = new Map();
        this.invertedIndex.set(term, posting);
      }
      posting.set(id, freq);
    }
  }

  public removeDocument(id: string): void {
    const docLen = this.docLengths.get(id);
    if (docLen === undefined) return;

    this.totalDocLength -= docLen;
    this.docLengths.delete(id);

    for (const posting of this.invertedIndex.values()) {
      posting.delete(id);
    }
  }

  public search(query: string, topK = 10): BM25SearchResult[] {
    const queryTokens = this.tokenize(query);
    if (queryTokens.length === 0 || this.docLengths.size === 0) {
      return [];
    }

    const N = this.docLengths.size;
    const avgdl = this.totalDocLength / Math.max(1, N);
    const scores = new Map<string, number>();

    for (const term of queryTokens) {
      const posting = this.invertedIndex.get(term);
      if (!posting) continue;

      const df = posting.size;
      // Robertson-Spärck Jones IDF formula with add-1 smoothing
      const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));

      for (const [docId, tf] of posting.entries()) {
        const docLen = this.docLengths.get(docId) || avgdl;
        const numerator = tf * (this.k1 + 1);
        const denominator = tf + this.k1 * (1 - this.b + this.b * (docLen / avgdl));
        const termScore = idf * (numerator / denominator);

        scores.set(docId, (scores.get(docId) || 0) + termScore);
      }
    }

    const results: BM25SearchResult[] = Array.from(scores.entries()).map(([id, score]) => ({
      id,
      score,
    }));

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }

  public clear(): void {
    this.docLengths.clear();
    this.totalDocLength = 0;
    this.invertedIndex.clear();
  }
}
