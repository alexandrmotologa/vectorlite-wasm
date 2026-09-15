import { SearchResult } from './hnsw_index';
import { BM25SearchResult } from './bm25';

export interface HybridSearchResult {
  id: string;
  combinedScore: number;
  denseScore?: number;
  denseRank?: number;
  sparseScore?: number;
  sparseRank?: number;
}

export interface RRFOptions {
  k?: number;            // Smoothing constant (standard: 60)
  denseWeight?: number;  // Weight multiplier for dense results (default: 1.0)
  sparseWeight?: number; // Weight multiplier for sparse BM25 results (default: 0.8)
}

/**
 * Combines dense vector search rankings and sparse lexical (BM25) rankings
 * using Reciprocal Rank Fusion (RRF).
 */
export function reciprocalRankFusion(
  denseResults: SearchResult[],
  sparseResults: BM25SearchResult[],
  topK = 10,
  options: RRFOptions = {}
): HybridSearchResult[] {
  const k = options.k ?? 60;
  const denseWeight = options.denseWeight ?? 1.0;
  const sparseWeight = options.sparseWeight ?? 0.8;

  const scoreMap = new Map<string, {
    combinedScore: number;
    denseScore?: number;
    denseRank?: number;
    sparseScore?: number;
    sparseRank?: number;
  }>();

  // 1. Process Dense Ranking
  denseResults.forEach((res, index) => {
    const rank = index + 1;
    const rrfContribution = denseWeight / (k + rank);

    scoreMap.set(res.id, {
      combinedScore: rrfContribution,
      denseScore: res.score,
      denseRank: rank,
    });
  });

  // 2. Process Sparse (BM25) Ranking
  sparseResults.forEach((res, index) => {
    const rank = index + 1;
    const rrfContribution = sparseWeight / (k + rank);

    const existing = scoreMap.get(res.id);
    if (existing) {
      existing.combinedScore += rrfContribution;
      existing.sparseScore = res.score;
      existing.sparseRank = rank;
    } else {
      scoreMap.set(res.id, {
        combinedScore: rrfContribution,
        sparseScore: res.score,
        sparseRank: rank,
      });
    }
  });

  const merged: HybridSearchResult[] = Array.from(scoreMap.entries()).map(([id, data]) => ({
    id,
    combinedScore: data.combinedScore,
    denseScore: data.denseScore,
    denseRank: data.denseRank,
    sparseScore: data.sparseScore,
    sparseRank: data.sparseRank,
  }));

  // Sort descending by combinedScore
  merged.sort((a, b) => b.combinedScore - a.combinedScore);

  // Normalize scores to [0.0, 1.0] relative to highest rank for clean UI display
  const maxScore = merged.length > 0 ? merged[0].combinedScore : 1.0;
  if (maxScore > 0) {
    for (const item of merged) {
      item.combinedScore = Math.min(1.0, item.combinedScore / maxScore);
    }
  }

  return merged.slice(0, topK);
}
