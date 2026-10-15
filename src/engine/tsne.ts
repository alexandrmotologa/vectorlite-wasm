import { cosineSimilarity } from './similarity';

export interface TSNEPoint {
  x: number;
  y: number;
}

export interface TSNEOptions {
  perplexity?: number; // default: 15
  learningRate?: number; // default: 100
  iterations?: number; // default: 120
}

export class FastTSNE {
  /**
   * Runs 2D t-SNE projection on high-dimensional vectors.
   * Returns normalized [0, 1] 2D coordinates for each input vector.
   */
  public static project(vectors: Float32Array[], options: TSNEOptions = {}): TSNEPoint[] {
    const N = vectors.length;
    if (N < 2) {
      return vectors.map(() => ({ x: 0.5, y: 0.5 }));
    }

    const perplexity = Math.min(options.perplexity ?? 15, Math.max(1, (N - 1) / 3));
    const iterations = options.iterations ?? 120;
    const eta = options.learningRate ?? 100;

    // 1. Compute pairwise squared Euclidean distances from cosine similarity or vectors
    const distances: Float32Array[] = [];
    for (let i = 0; i < N; i++) {
      const row = new Float32Array(N);
      for (let j = 0; j < N; j++) {
        if (i === j) {
          row[j] = 0;
        } else {
          // Cosine distance squared for normalized vectors: 2 * (1 - cos(a,b))
          const cos = cosineSimilarity(vectors[i], vectors[j], true);
          row[j] = Math.max(0, 2 * (1 - cos));
        }
      }
      distances.push(row);
    }

    // 2. Compute P-conditional probabilities with fixed sigma approximation
    const P = Array.from({ length: N }, () => new Float32Array(N));
    for (let i = 0; i < N; i++) {
      let sumP = 0;
      for (let j = 0; j < N; j++) {
        if (i !== j) {
          const val = Math.exp(-distances[i][j] / (2 * Math.max(0.01, perplexity * 0.1)));
          P[i][j] = val;
          sumP += val;
        }
      }
      if (sumP > 0) {
        const invSum = 1.0 / sumP;
        for (let j = 0; j < N; j++) {
          P[i][j] *= invSum;
        }
      }
    }

    // Symmetrize P: P_ij = (P_i|j + P_j|i) / (2N)
    const inv2N = 1.0 / (2 * N);
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const sym = (P[i][j] + P[j][i]) * inv2N;
        P[i][j] = sym;
        P[j][i] = sym;
      }
    }

    // 3. Initialize Y coordinates with small random values
    const Y: TSNEPoint[] = [];
    const prevY: TSNEPoint[] = [];
    for (let i = 0; i < N; i++) {
      // Deterministic initial coordinates using mulberry or simple pseudo-random
      const rx = (Math.sin(i * 12.9898) * 43758.5453) % 1;
      const ry = (Math.cos(i * 78.233) * 43758.5453) % 1;
      Y.push({ x: rx * 0.1, y: ry * 0.1 });
      prevY.push({ x: rx * 0.1, y: ry * 0.1 });
    }

    // 4. Gradient descent iterations
    const gains = Array.from({ length: N }, () => ({ x: 1.0, y: 1.0 }));

    for (let iter = 0; iter < iterations; iter++) {
      // Compute Q-distribution (Student-t kernel)
      let sumQ = 0;
      const num = Array.from({ length: N }, () => new Float32Array(N));
      for (let i = 0; i < N; i++) {
        for (let j = i + 1; j < N; j++) {
          const dx = Y[i].x - Y[j].x;
          const dy = Y[i].y - Y[j].y;
          const qVal = 1.0 / (1.0 + dx * dx + dy * dy);
          num[i][j] = qVal;
          num[j][i] = qVal;
          sumQ += 2 * qVal;
        }
      }

      const invSumQ = sumQ > 0 ? 1.0 / sumQ : 0;

      // Compute gradients
      for (let i = 0; i < N; i++) {
        let gradX = 0;
        let gradY = 0;
        for (let j = 0; j < N; j++) {
          if (i === j) continue;
          const qVal = num[i][j] * invSumQ;
          const mult = 4.0 * (P[i][j] - qVal) * num[i][j];
          gradX += mult * (Y[i].x - Y[j].x);
          gradY += mult * (Y[i].y - Y[j].y);
        }

        // Adaptive update with momentum
        const momentum = iter < 20 ? 0.5 : 0.8;
        const dx = Y[i].x - prevY[i].x;
        const dy = Y[i].y - prevY[i].y;

        gains[i].x = Math.sign(gradX) !== Math.sign(dx) ? gains[i].x + 0.2 : Math.max(0.01, gains[i].x * 0.8);
        gains[i].y = Math.sign(gradY) !== Math.sign(dy) ? gains[i].y + 0.2 : Math.max(0.01, gains[i].y * 0.8);

        const newDx = momentum * dx - eta * gains[i].x * gradX;
        const newDy = momentum * dy - eta * gains[i].y * gradY;

        prevY[i].x = Y[i].x;
        prevY[i].y = Y[i].y;

        Y[i].x += newDx;
        Y[i].y += newDy;
      }
    }

    // 5. Normalize Y coordinates to [0, 1]
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < N; i++) {
      if (Y[i].x < minX) minX = Y[i].x;
      if (Y[i].x > maxX) maxX = Y[i].x;
      if (Y[i].y < minY) minY = Y[i].y;
      if (Y[i].y > maxY) maxY = Y[i].y;
    }

    const rangeX = Math.max(1e-5, maxX - minX);
    const rangeY = Math.max(1e-5, maxY - minY);

    return Y.map(pt => ({
      x: 0.05 + 0.9 * ((pt.x - minX) / rangeX),
      y: 0.05 + 0.9 * ((pt.y - minY) / rangeY),
    }));
  }
}
