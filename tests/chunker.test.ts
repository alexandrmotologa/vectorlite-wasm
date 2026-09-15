import { describe, it, expect } from 'vitest';
import { RecursiveChunker } from '../src/engine/chunker';

describe('Recursive Character Chunker', () => {
  it('splits long markdown document into structured chunks', () => {
    const markdown = `
# Distributed Systems Architecture

A distributed system is a collection of autonomous computing entities that communicate over a network.

## Consensus Algorithms

Consensus algorithms allow a collection of machines to work as a coherent group that can survive failures.
Examples include Raft and Paxos. Raft relies on leader election and log replication.

## Partitioning & Sharding

Horizontal scaling requires partitioning data across nodes. Consistent hashing is widely adopted to minimize key remapping when nodes join or leave the cluster.
    `.trim();

    const chunker = new RecursiveChunker({ chunkSize: 180, chunkOverlap: 40 });
    const chunks = chunker.split(markdown, 'distributed_systems.md');

    expect(chunks.length).toBeGreaterThan(1);
    expect(chunks[0].documentName).toBe('distributed_systems.md');
    expect(chunks[0].totalChunks).toBe(chunks.length);
    expect(chunks[0].tokenCountEstimate).toBeGreaterThan(0);

    // Text continuity
    const allText = chunks.map((c) => c.text).join(' ');
    expect(allText).toContain('Consistent hashing');
    expect(allText).toContain('Consensus algorithms');
  });
});
