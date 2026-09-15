import { openDB, IDBPDatabase } from 'idb';
import { TextChunk } from './chunker';
import { SerializedHNSW } from './hnsw_index';

const DB_NAME = 'vectorlite_wasm_db';
const DB_VERSION = 1;

export interface StoredDocument {
  name: string;
  sizeBytes: number;
  chunkCount: number;
  createdAt: number;
}

export interface StoredVector {
  id: string;
  vector: Float32Array;
}

export interface StoredSnapshot {
  id: string;
  title: string;
  timestamp: number;
  chunkCount: number;
  hnswData: SerializedHNSW;
  chunks: TextChunk[];
}

export class VectorStorage {
  private dbPromise: Promise<IDBPDatabase>;

  constructor() {
    this.dbPromise = openDB(DB_NAME, DB_VERSION, {
      upgrade(db) {
        if (!db.objectStoreNames.contains('documents')) {
          db.createObjectStore('documents', { keyPath: 'name' });
        }
        if (!db.objectStoreNames.contains('chunks')) {
          const chunkStore = db.createObjectStore('chunks', { keyPath: 'id' });
          chunkStore.createIndex('documentName', 'documentName');
        }
        if (!db.objectStoreNames.contains('vectors')) {
          db.createObjectStore('vectors', { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains('snapshots')) {
          db.createObjectStore('snapshots', { keyPath: 'id' });
        }
      },
    });
  }

  public async saveDocument(doc: StoredDocument): Promise<void> {
    const db = await this.dbPromise;
    await db.put('documents', doc);
  }

  public async getDocuments(): Promise<StoredDocument[]> {
    const db = await this.dbPromise;
    return db.getAll('documents');
  }

  public async saveChunks(chunks: TextChunk[]): Promise<void> {
    const db = await this.dbPromise;
    const tx = db.transaction('chunks', 'readwrite');
    for (const chunk of chunks) {
      await tx.store.put(chunk);
    }
    await tx.done;
  }

  public async getChunk(id: string): Promise<TextChunk | undefined> {
    const db = await this.dbPromise;
    return db.get('chunks', id);
  }

  public async getAllChunks(): Promise<TextChunk[]> {
    const db = await this.dbPromise;
    return db.getAll('chunks');
  }

  public async saveVectors(vectors: Array<{ id: string; vector: Float32Array }>): Promise<void> {
    const db = await this.dbPromise;
    const tx = db.transaction('vectors', 'readwrite');
    for (const v of vectors) {
      await tx.store.put(v);
    }
    await tx.done;
  }

  public async getAllVectors(): Promise<StoredVector[]> {
    const db = await this.dbPromise;
    return db.getAll('vectors');
  }

  public async saveSnapshot(snapshot: StoredSnapshot): Promise<void> {
    const db = await this.dbPromise;
    await db.put('snapshots', snapshot);
  }

  public async getSnapshots(): Promise<StoredSnapshot[]> {
    const db = await this.dbPromise;
    return db.getAll('snapshots');
  }

  public async clearAll(): Promise<void> {
    const db = await this.dbPromise;
    const tx = db.transaction(['documents', 'chunks', 'vectors', 'snapshots'], 'readwrite');
    await tx.objectStore('documents').clear();
    await tx.objectStore('chunks').clear();
    await tx.objectStore('vectors').clear();
    await tx.objectStore('snapshots').clear();
    await tx.done;
  }
}
