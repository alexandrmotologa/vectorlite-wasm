import { describe, it, expect } from 'vitest';

describe('RAG Conversation History', () => {
  const sampleHistory: Array<{ role: 'user' | 'assistant'; content: string }> = [
    { role: 'user', content: 'What is HNSW?' },
    { role: 'assistant', content: 'HNSW is Hierarchical Navigable Small World, a graph-based vector index.' },
    { role: 'user', content: 'How does it compare to flat index?' },
    { role: 'assistant', content: 'It provides logarithmic search complexity at the cost of indexing memory.' },
    { role: 'user', content: 'What quantization options are supported?' },
    { role: 'assistant', content: 'VectorLite-WASM supports SQ8 scalar and BQ 1-bit binary quantization.' },
  ];

  it('correctly slices the last 3 Q&A pairs (6 items) for worker message payload', () => {
    const extendedHistory = [
      { role: 'user' as const, content: 'Old question 1' },
      { role: 'assistant' as const, content: 'Old answer 1' },
      ...sampleHistory,
    ];
    expect(extendedHistory.length).toBe(8);

    const payloadHistory = extendedHistory.slice(-6);
    expect(payloadHistory.length).toBe(6);
    expect(payloadHistory[0].content).toBe('What is HNSW?');
    expect(payloadHistory[payloadHistory.length - 1].content).toContain('VectorLite-WASM supports');
  });

  it('correctly formats Ollama history block', () => {
    const historyBlock = sampleHistory.map(h =>
      h.role === 'user' ? `User: ${h.content}` : `Assistant: ${h.content}`
    ).join('\n');

    expect(historyBlock).toContain('User: What is HNSW?');
    expect(historyBlock).toContain('Assistant: HNSW is Hierarchical Navigable Small World');
    expect(historyBlock.split('\n').length).toBe(6);
  });

  it('correctly formats In-Browser LLM history context with truncation', () => {
    const historyContext = sampleHistory.slice(-4).map(h =>
      h.role === 'user' ? `Q: ${h.content.slice(0, 80)}` : `A: ${h.content.slice(0, 120)}`
    ).join(' ');

    expect(historyContext).toContain('Q: How does it compare to flat index?');
    expect(historyContext).toContain('A: It provides logarithmic search complexity');
    expect(historyContext).toContain('Q: What quantization options are supported?');
  });

  it('correctly slices previous exchanges for UI display', () => {
    // When history has 6 entries (3 exchanges), previous exchanges before the current one should be the first 4 (2 exchanges)
    const previous = sampleHistory.slice(0, -2);
    expect(previous.length).toBe(4);
    expect(Math.floor(sampleHistory.length / 2)).toBe(3);
    expect(previous[0].role).toBe('user');
    expect(previous[1].role).toBe('assistant');
  });
});
