import { dotProduct } from './similarity';

export interface Point2D {
  x: number;
  y: number;
}

export interface PCAModel {
  mean: Float32Array;
  v1: Float32Array;
  v2: Float32Array;
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

/**
 * Fast online Principal Component Analysis (PCA) using power iteration.
 * Projects 384-dimensional embeddings to a 2D plane for interactive spatial cluster visualization.
 */
export class FastPCA {
  /**
   * Computes the 2D projection model for an array of vectors.
   */
  public static fit(vectors: Float32Array[], iterations = 15): PCAModel {
    const n = vectors.length;
    if (n === 0) {
      throw new Error('Cannot fit PCA on empty vector set.');
    }
    const dim = vectors[0].length;

    // 1. Compute Mean Vector
    const mean = new Float32Array(dim);
    for (let i = 0; i < n; i++) {
      const vec = vectors[i];
      for (let d = 0; d < dim; d++) {
        mean[d] += vec[d];
      }
    }
    const invN = 1.0 / n;
    for (let d = 0; d < dim; d++) {
      mean[d] *= invN;
    }

    // 2. Mean-center all vectors
    const centered: Float32Array[] = new Array(n);
    for (let i = 0; i < n; i++) {
      const c = new Float32Array(dim);
      const vec = vectors[i];
      for (let d = 0; d < dim; d++) {
        c[d] = vec[d] - mean[d];
      }
      centered[i] = c;
    }

    // 3. Power iteration for 1st Principal Component (v1)
    const v1 = new Float32Array(dim);
    for (let d = 0; d < dim; d++) {
      v1[d] = Math.sin(d + 1); // Deterministic non-zero initialization
    }
    FastPCA.normalizeVectorInPlace(v1);

    for (let iter = 0; iter < iterations; iter++) {
      const nextV = new Float32Array(dim);
      for (let i = 0; i < n; i++) {
        const dot = dotProduct(centered[i], v1);
        for (let d = 0; d < dim; d++) {
          nextV[d] += dot * centered[i][d];
        }
      }
      FastPCA.normalizeVectorInPlace(nextV);
      v1.set(nextV);
    }

    // 4. Deflate centered vectors by v1 to find orthogonal 2nd component (v2)
    const deflated: Float32Array[] = new Array(n);
    for (let i = 0; i < n; i++) {
      const dot = dotProduct(centered[i], v1);
      const def = new Float32Array(dim);
      for (let d = 0; d < dim; d++) {
        def[d] = centered[i][d] - dot * v1[d];
      }
      deflated[i] = def;
    }

    const v2 = new Float32Array(dim);
    for (let d = 0; d < dim; d++) {
      v2[d] = Math.cos(d + 1);
    }
    FastPCA.normalizeVectorInPlace(v2);

    for (let iter = 0; iter < iterations; iter++) {
      const nextV = new Float32Array(dim);
      for (let i = 0; i < n; i++) {
        const dot = dotProduct(deflated[i], v2);
        for (let d = 0; d < dim; d++) {
          nextV[d] += dot * deflated[i][d];
        }
      }
      FastPCA.normalizeVectorInPlace(nextV);
      v2.set(nextV);
    }

    // 5. Compute min/max bounds for normalization
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    for (let i = 0; i < n; i++) {
      const x = dotProduct(centered[i], v1);
      const y = dotProduct(centered[i], v2);
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }

    // Add margin if bounds are equal
    if (minX === maxX) { minX -= 1; maxX += 1; }
    if (minY === maxY) { minY -= 1; maxY += 1; }

    return { mean, v1, v2, minX, maxX, minY, maxY };
  }

  /**
   * Projects a single vector into normalized [0, 1] 2D coordinates.
   */
  public static project(vector: Float32Array, model: PCAModel): Point2D {
    const dim = vector.length;
    const c = new Float32Array(dim);
    for (let d = 0; d < dim; d++) {
      c[d] = vector[d] - model.mean[d];
    }

    const rawX = dotProduct(c, model.v1);
    const rawY = dotProduct(c, model.v2);

    const normX = (rawX - model.minX) / (model.maxX - model.minX);
    const normY = (rawY - model.minY) / (model.maxY - model.minY);

    return {
      x: Math.max(0, Math.min(1, normX)),
      y: Math.max(0, Math.min(1, normY)),
    };
  }

  private static normalizeVectorInPlace(v: Float32Array): void {
    let sum = 0;
    for (let i = 0; i < v.length; i++) {
      sum += v[i] * v[i];
    }
    const mag = Math.sqrt(sum);
    if (mag === 0) return;
    const inv = 1.0 / mag;
    for (let i = 0; i < v.length; i++) {
      v[i] *= inv;
    }
  }
}
