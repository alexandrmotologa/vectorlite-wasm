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
