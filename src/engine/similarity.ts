/**
 * High-performance vector math operations for Float32Array vectors.
 * Features 4x loop unrolling for optimal CPU cache utilization and SIMD pipelining.
 */

export type DistanceMetric = 'cosine' | 'dot' | 'euclidean';

/**
 * Computes the dot product of two Float32Array vectors.
 * Uses 4-step loop unrolling.
 */
export function dotProduct(a: Float32Array, b: Float32Array): number {
  const len = a.length;
  let sum = 0;
  let i = 0;
  const limit = len - 3;

  for (; i < limit; i += 4) {
    sum += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
  }

  for (; i < len; i++) {
    sum += a[i] * b[i];
  }

  return sum;
}

/**
 * Computes Euclidean (L2) norm of a Float32Array.
 */
export function norm(a: Float32Array): number {
  return Math.sqrt(dotProduct(a, a));
}

/**
 * In-place normalization of a Float32Array to unit length (L2 norm = 1.0).
 */
export function normalizeInPlace(a: Float32Array): Float32Array {
  const n = norm(a);
  if (n === 0 || Math.abs(n - 1.0) < 1e-6) return a;
  const invNorm = 1.0 / n;
  for (let i = 0; i < a.length; i++) {
    a[i] *= invNorm;
  }
  return a;
}

/**
 * Returns a new unit-normalized copy of a Float32Array.
 */
export function normalize(a: Float32Array): Float32Array {
  const copy = new Float32Array(a);
  return normalizeInPlace(copy);
}

/**
 * Computes cosine similarity between two vectors (range: -1.0 to 1.0).
 * Assumes unit vectors by default for maximum speed; falls back to full formula if unnormalized.
 */
export function cosineSimilarity(a: Float32Array, b: Float32Array, assumedNormalized = false): number {
  const dot = dotProduct(a, b);
  if (assumedNormalized) {
    return Math.max(-1.0, Math.min(1.0, dot));
  }
  const normA = norm(a);
  const normB = norm(b);
  if (normA === 0 || normB === 0) return 0;
  return Math.max(-1.0, Math.min(1.0, dot / (normA * normB)));
}

/**
 * Computes squared Euclidean distance between two vectors.
 */
export function squaredEuclideanDistance(a: Float32Array, b: Float32Array): number {
  const len = a.length;
  let sum = 0;
  let i = 0;
  const limit = len - 3;

  for (; i < limit; i += 4) {
    const d0 = a[i] - b[i];
    const d1 = a[i + 1] - b[i + 1];
    const d2 = a[i + 2] - b[i + 2];
    const d3 = a[i + 3] - b[i + 3];
    sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
  }

  for (; i < len; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }

  return sum;
}

/**
 * Computes Euclidean distance between two vectors.
 */
export function euclideanDistance(a: Float32Array, b: Float32Array): number {
  return Math.sqrt(squaredEuclideanDistance(a, b));
}

/**
 * Computes distance based on selected metric.
 * Lower distance means higher similarity.
 */
export function vectorDistance(
  a: Float32Array,
  b: Float32Array,
  metric: DistanceMetric = 'cosine',
  isNormalized = true
): number {
  switch (metric) {
    case 'cosine': {
      // Cosine distance: 1.0 - cosine_similarity (range: 0 to 2)
      return 1.0 - cosineSimilarity(a, b, isNormalized);
    }
    case 'dot': {
      // Inverted dot product distance for maximum inner product search
      return -dotProduct(a, b);
    }
    case 'euclidean': {
      return euclideanDistance(a, b);
    }
  }
}
