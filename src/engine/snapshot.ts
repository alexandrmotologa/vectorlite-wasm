import { HNSWIndex, SerializedHNSW } from './hnsw_index';
import { TextChunk } from './chunker';

export interface VLiteSnapshot {
  version: '1.0';
  type: 'vectorlite-snapshot';
  createdAt: string;
  metadata: {
    totalChunks: number;
    dimension: number;
    modelName: string;
    description?: string;
  };
  hnsw: SerializedHNSW;
  chunks: TextChunk[];
}

/**
 * Creates a downloadable .vlite snapshot blob from current state.
 */
export function createSnapshotBlob(
  hnsw: HNSWIndex,
  chunks: TextChunk[],
  modelName = 'Xenova/all-MiniLM-L6-v2'
): Blob {
  const serializedHnsw = hnsw.serialize();
  const dimension = serializedHnsw.nodes.length > 0 ? serializedHnsw.nodes[0].vector.length : 384;

  const snapshot: VLiteSnapshot = {
    version: '1.0',
    type: 'vectorlite-snapshot',
    createdAt: new Date().toISOString(),
    metadata: {
      totalChunks: chunks.length,
      dimension,
      modelName,
    },
    hnsw: serializedHnsw,
    chunks,
  };

  return new Blob([JSON.stringify(snapshot, null, 2)], {
    type: 'application/json',
  });
}

/**
 * Triggers browser download of the .vlite snapshot file.
 */
export function downloadSnapshot(
  hnsw: HNSWIndex,
  chunks: TextChunk[],
  filename = 'vectorlite-knowledge-base.vlite'
): void {
  const blob = createSnapshotBlob(hnsw, chunks);
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Restores HNSW index and text chunks from an imported snapshot file or text.
 */
export async function parseSnapshot(
  input: File | string
): Promise<{ hnsw: HNSWIndex; chunks: TextChunk[]; modelName: string }> {
  let content: string;
  if (input instanceof File) {
    content = await input.text();
  } else {
    content = input;
  }

  const parsed = JSON.parse(content) as VLiteSnapshot;
  if (parsed.type !== 'vectorlite-snapshot') {
    throw new Error('Invalid file format: missing vectorlite-snapshot signature.');
  }

  const hnsw = HNSWIndex.deserialize(parsed.hnsw);
  return {
    hnsw,
    chunks: parsed.chunks,
    modelName: parsed.metadata.modelName || 'Xenova/all-MiniLM-L6-v2',
  };
}

/**
 * Binary snapshot format:
 * [4 bytes] Magic number: 0x564C5753 ('VLWS')
 * [4 bytes] Version: 1
 * [4 bytes] Number of nodes
 * [4 bytes] Vector dimension
 * [4 bytes] M
 * [4 bytes] M0
 * [4 bytes] efConstruction
 * [4 bytes] efSearch
 * [4 bytes] Metric enum (0=cosine, 1=dot, 2=euclidean)
 * [4 bytes] Quantization enum (0=f32, 1=sq8, 2=bq)
 * [4 bytes] Entry point index (-1 if null)
 * [4 bytes] Max level
 * [4 bytes] Chunks JSON byte length
 * [N bytes] Chunks JSON (UTF-8)
 * For each node:
 *   [var bytes] ID as null-terminated UTF-8
 *   [4 bytes] Level
 *   [dim*4 bytes] Vector data (Float32)
 *   For each layer (0..level):
 *     [4 bytes] Neighbor count
 *     [count*4 bytes] Neighbor indices
 */

const MAGIC = 0x564C5753; // 'VLWS'
const VERSION = 1;

const METRIC_MAP: Record<string, number> = { cosine: 0, dot: 1, euclidean: 2 };
const METRIC_REVERSE: Record<number, string> = { 0: 'cosine', 1: 'dot', 2: 'euclidean' };
const QUANT_MAP: Record<string, number> = { f32: 0, sq8: 1, bq: 2 };
const QUANT_REVERSE: Record<number, string> = { 0: 'f32', 1: 'sq8', 2: 'bq' };

export function serializeBinary(hnsw: HNSWIndex, chunks: TextChunk[]): ArrayBuffer {
  const serialized = hnsw.serialize();
  const nodes = serialized.nodes;
  const dim = nodes.length > 0 ? nodes[0].vector.length : 384;

  const chunksJson = new TextEncoder().encode(JSON.stringify(chunks));

  // Calculate total size
  const HEADER_SIZE = 13 * 4; // 13 uint32 fields
  let nodesSize = 0;
  for (const node of nodes) {
    const idBytes = new TextEncoder().encode(node.id);
    nodesSize += idBytes.length + 1; // id + null terminator
    nodesSize += 4; // level
    nodesSize += dim * 4; // vector
    for (const layer of node.neighbors) {
      nodesSize += 4; // neighbor count
      nodesSize += layer.length * 4; // neighbor indices
    }
  }

  const totalSize = HEADER_SIZE + chunksJson.length + nodesSize;
  const buffer = new ArrayBuffer(totalSize);
  const view = new DataView(buffer);
  const uint8 = new Uint8Array(buffer);
  let offset = 0;

  // Header
  view.setUint32(offset, MAGIC, true); offset += 4;
  view.setUint32(offset, VERSION, true); offset += 4;
  view.setUint32(offset, nodes.length, true); offset += 4;
  view.setUint32(offset, dim, true); offset += 4;
  view.setUint32(offset, serialized.M, true); offset += 4;
  view.setUint32(offset, serialized.M0, true); offset += 4;
  view.setUint32(offset, serialized.efConstruction, true); offset += 4;
  view.setUint32(offset, serialized.efSearch, true); offset += 4;
  view.setUint32(offset, METRIC_MAP[serialized.metric] ?? 0, true); offset += 4;
  view.setUint32(offset, QUANT_MAP[serialized.quantization ?? 'f32'] ?? 0, true); offset += 4;
  view.setInt32(offset, serialized.entryPointIndex ?? -1, true); offset += 4;
  view.setInt32(offset, serialized.maxLevel, true); offset += 4;
  view.setUint32(offset, chunksJson.length, true); offset += 4;

  // Chunks JSON
  uint8.set(chunksJson, offset);
  offset += chunksJson.length;

  // Nodes
  for (const node of nodes) {
    const idBytes = new TextEncoder().encode(node.id);
    uint8.set(idBytes, offset); offset += idBytes.length;
    uint8[offset] = 0; offset += 1; // null terminator

    view.setUint32(offset, node.level, true); offset += 4;

    // Vector as raw Float32
    const vecFloat = new Float32Array(node.vector);
    new Uint8Array(buffer, offset, dim * 4).set(new Uint8Array(vecFloat.buffer));
    offset += dim * 4;

    // Neighbors per layer
    for (const layer of node.neighbors) {
      view.setUint32(offset, layer.length, true); offset += 4;
      for (const idx of layer) {
        view.setUint32(offset, idx, true); offset += 4;
      }
    }
  }

  return buffer;
}

export function deserializeBinary(buffer: ArrayBuffer): { hnsw: HNSWIndex; chunks: TextChunk[] } {
  const view = new DataView(buffer);
  const uint8 = new Uint8Array(buffer);
  let offset = 0;

  const magic = view.getUint32(offset, true); offset += 4;
  if (magic !== MAGIC) throw new Error('Invalid VectorLite binary snapshot: bad magic number');

  const version = view.getUint32(offset, true); offset += 4;
  if (version !== VERSION) throw new Error(`Unsupported snapshot version: ${version}`);

  const nodeCount = view.getUint32(offset, true); offset += 4;
  const dim = view.getUint32(offset, true); offset += 4;
  const M = view.getUint32(offset, true); offset += 4;
  const M0 = view.getUint32(offset, true); offset += 4;
  const efConstruction = view.getUint32(offset, true); offset += 4;
  const efSearch = view.getUint32(offset, true); offset += 4;
  const metricNum = view.getUint32(offset, true); offset += 4;
  const quantNum = view.getUint32(offset, true); offset += 4;
  const entryPointIndex = view.getInt32(offset, true); offset += 4;
  const maxLevel = view.getInt32(offset, true); offset += 4;
  const chunksJsonLen = view.getUint32(offset, true); offset += 4;

  const chunksJsonBytes = uint8.slice(offset, offset + chunksJsonLen);
  const chunks: TextChunk[] = JSON.parse(new TextDecoder().decode(chunksJsonBytes));
  offset += chunksJsonLen;

  const nodes: SerializedHNSW['nodes'] = [];
  for (let i = 0; i < nodeCount; i++) {
    // Read null-terminated ID
    let idEnd = offset;
    while (uint8[idEnd] !== 0) idEnd++;
    const id = new TextDecoder().decode(uint8.slice(offset, idEnd));
    offset = idEnd + 1;

    const level = view.getUint32(offset, true); offset += 4;

    const vector = Array.from(new Float32Array(buffer.slice(offset, offset + dim * 4)));
    offset += dim * 4;

    const neighbors: number[][] = [];
    for (let l = 0; l <= level; l++) {
      const count = view.getUint32(offset, true); offset += 4;
      const layer: number[] = [];
      for (let n = 0; n < count; n++) {
        layer.push(view.getUint32(offset, true)); offset += 4;
      }
      neighbors.push(layer);
    }

    nodes.push({ id, vector, level, neighbors });
  }

  const serialized: SerializedHNSW = {
    M, M0, efConstruction, efSearch,
    metric: (METRIC_REVERSE[metricNum] || 'cosine') as any,
    quantization: (QUANT_REVERSE[quantNum] || 'f32') as any,
    entryPointIndex: entryPointIndex === -1 ? null : entryPointIndex,
    maxLevel,
    nodes,
  };

  const hnsw = HNSWIndex.deserialize(serialized);
  return { hnsw, chunks };
}
