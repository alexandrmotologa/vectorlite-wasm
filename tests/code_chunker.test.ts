import { describe, it, expect } from 'vitest';
import { CodeAwareChunker } from '../src/engine/code_chunker';

describe('Code-Aware Syntax Chunker', () => {
  it('splits TypeScript code while keeping function and class blocks intact', () => {
    const tsCode = `
import { Database } from './db';

export interface UserConfig {
  id: string;
  theme: 'dark' | 'light';
}

export class UserManager {
  private db: Database;

  constructor(db: Database) {
    this.db = db;
  }

  public async getUser(id: string): Promise<UserConfig> {
    return this.db.find(id);
  }
}

export function computeHash(data: string): string {
  let hash = 0;
  for (let i = 0; i < data.length; i++) {
    hash = (hash << 5) - hash + data.charCodeAt(i);
  }
  return String(hash);
}
    `.trim();

    const chunker = new CodeAwareChunker({ maxChunkLines: 20, overlapLines: 4 });
    const chunks = chunker.splitCode(tsCode, 'user_manager.ts');

    expect(chunks.length).toBeGreaterThan(0);
    expect(chunks[0].documentName).toBe('user_manager.ts');

    // Verify declaration preservation
    const combined = chunks.map((c) => c.text).join('\n');
    expect(combined).toContain('class UserManager');
    expect(combined).toContain('function computeHash');
  });
});
