/**
 * Vector Quantization Suite for in-browser memory reduction:
 * 1. Scalar Quantization (SQ8 / Int8): 75% RAM reduction (1.5KB -> 384 bytes per vector).
 * 2. Binary Quantization (BQ / 1-bit): 96.8% RAM reduction (1.5KB -> 48 bytes per vector).
 */

export interface QuantizedSQ8Vector {
  data: Int8Array;
  scale: number;
}

export interface QuantizedBQVector {
  bits: Uint32Array;
  dimension: number;
}

/**
 * Compresses a Float32Array into an Int8Array using symmetric scalar quantization.
 */
export function quantizeSQ8(vector: Float32Array): QuantizedSQ8Vector {
  const len = vector.length;
  let maxAbs = 0;

  for (let i = 0; i < len; i++) {
    const abs = Math.abs(vector[i]);
    if (abs > maxAbs) maxAbs = abs;
  }

  const scale = maxAbs === 0 ? 1 : maxAbs / 127;
  const invScale = 1.0 / scale;
  const data = new Int8Array(len);

  for (let i = 0; i < len; i++) {
    data[i] = Math.max(-128, Math.min(127, Math.round(vector[i] * invScale)));
  }

  return { data, scale };
}

/**
 * Reconstructs an approximate Float32Array vector from an SQ8 quantized vector.
 */
export function dequantizeSQ8(quantized: QuantizedSQ8Vector): Float32Array {
  const len = quantized.data.length;
  const result = new Float32Array(len);
  const scale = quantized.scale;

  for (let i = 0; i < len; i++) {
    result[i] = quantized.data[i] * scale;
  }

  return result;
}

/**
 * High-speed approximate dot product between two SQ8 quantized vectors.
 * Accumulates in standard 32-bit integer arithmetic.
 */
export function dotProductSQ8(a: QuantizedSQ8Vector, b: QuantizedSQ8Vector): number {
  const len = a.data.length;
  let intSum = 0;
  let i = 0;
  const limit = len - 3;

  for (; i < limit; i += 4) {
    intSum +=
      a.data[i] * b.data[i] +
      a.data[i + 1] * b.data[i + 1] +
      a.data[i + 2] * b.data[i + 2] +
      a.data[i + 3] * b.data[i + 3];
  }

  for (; i < len; i++) {
    intSum += a.data[i] * b.data[i];
  }

  return intSum * a.scale * b.scale;
}

/**
 * Compresses a Float32Array into a 1-bit packed binary vector (32 dimensions per Uint32 word).
 */
export function quantizeBQ(vector: Float32Array): QuantizedBQVector {
  const dim = vector.length;
  const wordCount = Math.ceil(dim / 32);
  const bits = new Uint32Array(wordCount);

  for (let i = 0; i < dim; i++) {
    if (vector[i] > 0) {
      const wordIdx = i >>> 5; // Math.floor(i / 32)
      const bitIdx = i & 31;   // i % 32
      bits[wordIdx] |= 1 << bitIdx;
    }
  }

  return { bits, dimension: dim };
}

/**
 * Computes Hamming distance between two binary quantized vectors using bitwise XOR and POPCNT.
 */
export function hammingDistanceBQ(a: QuantizedBQVector, b: QuantizedBQVector): number {
  const words = Math.min(a.bits.length, b.bits.length);
  let distance = 0;

  for (let i = 0; i < words; i++) {
    const xor = a.bits[i] ^ b.bits[i];
    distance += popcount32(xor);
  }

  return distance;
}

/**
 * Converts Hamming distance to approximate cosine similarity:
 * cos(theta) = cos(pi * (hammingDistance / dimension))
 */
export function cosineSimilarityBQ(a: QuantizedBQVector, b: QuantizedBQVector): number {
  const dist = hammingDistanceBQ(a, b);
  const normalizedDist = dist / a.dimension;
  return Math.cos(Math.PI * normalizedDist);
}

/**
 * 32-bit population count (number of set bits) using bitwise parallel logic.
 */
function popcount32(x: number): number {
  x = x - ((x >>> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  x = (x + (x >>> 4)) & 0x0f0f0f0f;
  x = x + (x >>> 8);
  x = x + (x >>> 16);
  return x & 0x3f;
}
