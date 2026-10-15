let generator: any = null;
let currentGenerativeModel = 'Xenova/LaMini-Flan-T5-77M';

export interface RAGWorkerInMessage {
  type: 'INIT' | 'GENERATE';
  modelName?: string;
  query?: string;
  contexts?: Array<{
    index: number;
    chunkId: string;
    documentName: string;
    chunkIndex: number;
    text: string;
  }>;
  backend?: 'in_browser_llm' | 'ollama' | 'extractive';
  ollamaEndpoint?: string;
  ollamaModel?: string;
  history?: Array<{ role: 'user' | 'assistant'; content: string }>;
}

export type RAGWorkerOutMessage =
  | { type: 'RAG_STATUS'; status: 'idle' | 'loading' | 'ready' | 'error'; message?: string; progress?: number; file?: string }
  | {
      type: 'RAG_RESULT';
      summary: string;
      bulletPoints: string[];
      citations: Array<{
        index: number;
        chunkId: string;
        documentName: string;
        chunkIndex: number;
        excerpt: string;
      }>;
      confidence: number;
      backendUsed: 'in_browser_llm' | 'ollama' | 'extractive';
      modelTitle: string;
    }
  | { type: 'RAG_STREAM_TOKEN'; token: string; accumulated: string }
  | { type: 'RAG_ERROR'; error: string };

self.onmessage = async (e: MessageEvent<RAGWorkerInMessage>) => {
  const { type, modelName, query, contexts, backend, ollamaEndpoint, ollamaModel, history } = e.data;

  try {
    if (type === 'INIT') {
      const targetModel = modelName || currentGenerativeModel;
      self.postMessage({
        type: 'RAG_STATUS',
        status: 'loading',
        message: `Loading in-browser LLM ${targetModel}...`,
      });

      try {
        const { pipeline, env } = await import('@xenova/transformers');
        env.allowLocalModels = false;
        env.useBrowserCache = true;

        generator = await pipeline('text2text-generation', targetModel, {
          quantized: true,
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          progress_callback: (prog: any) => {
            if (prog.status === 'progress') {
              self.postMessage({
                type: 'RAG_STATUS',
                status: 'loading',
                file: prog.file || 'model.onnx',
                progress: Math.round(prog.progress || 0),
                message: `Downloading generative LLM: ${Math.round(prog.progress || 0)}%`,
              });
            }
          },
        });

        currentGenerativeModel = targetModel;
        self.postMessage({
          type: 'RAG_STATUS',
          status: 'ready',
          message: `In-browser LLM ${targetModel} ready.`,
        });
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        self.postMessage({
          type: 'RAG_STATUS',
          status: 'error',
          message: `Failed to load in-browser LLM: ${msg}. Falling back to extractive mode.`,
        });
      }
      return;
    }

    if (type === 'GENERATE') {
      if (!query || !contexts || contexts.length === 0) {
        self.postMessage({
          type: 'RAG_RESULT',
          summary: 'No relevant context found in indexed documents to formulate an answer.',
          bulletPoints: [],
          citations: [],
          confidence: 0,
          backendUsed: backend || 'extractive',
          modelTitle: 'None',
        });
        return;
      }

      const citations = contexts.slice(0, 5).map((c) => ({
        index: c.index,
        chunkId: c.chunkId,
        documentName: c.documentName,
        chunkIndex: c.chunkIndex,
        excerpt: c.text.slice(0, 160).trim() + '...',
      }));

      // Option 1: Local Ollama endpoint
      if (backend === 'ollama') {
        const endpoint = (ollamaEndpoint || 'http://localhost:11434').replace(/\/+$/, '');
        const targetOllamaModel = ollamaModel || 'llama3.2';

        const contextBlock = contexts
          .slice(0, 4)
          .map((c) => `[Source ${c.index}] (${c.documentName}):\n${c.text.slice(0, 500)}`)
          .join('\n\n');

        const systemPrompt = `You are a factual, concise grounded RAG assistant. Answer the user query using only the supplied context. Use bracketed citation numbers like [1], [2] when referencing source facts.`;
        const historyBlock = (history || []).map(h => 
          h.role === 'user' ? `User: ${h.content}` : `Assistant: ${h.content}`
        ).join('\n');

        const prompt = `${systemPrompt}\n\n${historyBlock ? `Previous conversation:\n${historyBlock}\n\n` : ''}Context:\n${contextBlock}\n\nUser Question: ${query}\n\nConcise Grounded Answer:`;

        const res = await fetch(`${endpoint}/api/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: targetOllamaModel,
            prompt,
            stream: true,
            options: { temperature: 0.2, num_predict: 250 },
          }),
        });

        if (!res.ok) {
          throw new Error(`Ollama HTTP ${res.status}: ${res.statusText}`);
        }

        // Stream NDJSON tokens
        const reader = res.body?.getReader();
        if (!reader) throw new Error('No readable stream from Ollama');

        const decoder = new TextDecoder();
        let fullResponse = '';
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (!line.trim()) continue;
            try {
              const parsed = JSON.parse(line);
              if (parsed.response) {
                fullResponse += parsed.response;
                self.postMessage({
                  type: 'RAG_STREAM_TOKEN',
                  token: parsed.response,
                  accumulated: fullResponse,
                } as RAGWorkerOutMessage);
              }
            } catch { /* skip malformed lines */ }
          }
        }

        if (buffer.trim()) {
          try {
            const parsed = JSON.parse(buffer);
            if (parsed.response) {
              fullResponse += parsed.response;
              self.postMessage({
                type: 'RAG_STREAM_TOKEN',
                token: parsed.response,
                accumulated: fullResponse,
              } as RAGWorkerOutMessage);
            }
          } catch { /* skip malformed lines */ }
        }

        const rawAnswer = fullResponse;
        const lines = rawAnswer.split('\n').map((l) => l.trim()).filter(Boolean);
        const summary = lines[0] || 'Answer synthesized from local Ollama.';
        const bulletPoints = lines.slice(1, 4);

        self.postMessage({
          type: 'RAG_RESULT',
          summary,
          bulletPoints,
          citations,
          confidence: 0.95,
          backendUsed: 'ollama',
          modelTitle: `Ollama (${targetOllamaModel})`,
        });
        return;
      }

      // Option 2: In-Browser Generative ONNX LLM
      if (backend === 'in_browser_llm') {
        if (!generator) {
          // Initialize generator on-demand if not already loaded
          const { pipeline, env } = await import('@xenova/transformers');
          env.allowLocalModels = false;
          env.useBrowserCache = true;

          self.postMessage({
            type: 'RAG_STATUS',
            status: 'loading',
            message: 'Loading in-browser generative LLM (LaMini-Flan-T5)...',
          });

          generator = await pipeline('text2text-generation', currentGenerativeModel, {
            quantized: true,
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            progress_callback: (prog: any) => {
              if (prog.status === 'progress') {
                self.postMessage({
                  type: 'RAG_STATUS',
                  status: 'loading',
                  file: prog.file || 'model.onnx',
                  progress: Math.round(prog.progress || 0),
                  message: `Downloading generative LLM: ${Math.round(prog.progress || 0)}%`,
                });
              }
            },
          });
        }

        const contextBlock = contexts
          .slice(0, 3)
          .map((c) => `[${c.index}] ${c.text.slice(0, 350)}`)
          .join(' ');

        const historyContext = (history || [])
          .slice(-4)
          .map((h) => (h.role === 'user' ? `Q: ${h.content.slice(0, 80)}` : `A: ${h.content.slice(0, 120)}`))
          .join(' ');

        const prompt = `${historyContext ? historyContext + ' ' : ''}Context: ${contextBlock} Question: ${query} Answer the question concisely using the context:`;

        const output = await generator(prompt, {
          max_new_tokens: 140,
          temperature: 0.2,
          do_sample: false,
        });

        const generatedText = (output[0]?.generated_text || '').trim();
        const cleanSummary = generatedText || 'Could not synthesize answer from context.';

        // Extract key factual sentences for bullet points
        const bulletPoints: string[] = [];
        const sentences = cleanSummary.split(/(?<=[.!?])\s+/).filter((s: string) => s.length > 15);
        if (sentences.length > 1) {
          bulletPoints.push(...sentences.slice(1, 3));
        }

        self.postMessage({
          type: 'RAG_RESULT',
          summary: sentences[0] || cleanSummary,
          bulletPoints,
          citations,
          confidence: 0.92,
          backendUsed: 'in_browser_llm',
          modelTitle: `In-Browser ONNX (${currentGenerativeModel.split('/').pop()})`,
        });
        return;
      }

      // Option 3: Extractive Salience (Fast Instant Mode)
      const queryTerms = query
        .toLowerCase()
        .replace(/[^a-z0-9_\s]/g, ' ')
        .split(/\s+/)
        .filter((t) => t.length > 2);

      const scoredSentences: Array<{ sentence: string; score: number; citationIndex: number }> = [];

      contexts.slice(0, 5).forEach((ctx, chunkIdx) => {
        const sentences = ctx.text
          .replace(/\n+/g, ' ')
          .split(/(?<=[.!?])\s+/)
          .map((s) => s.trim())
          .filter((s) => s.length > 20 && !s.startsWith('#'));

        for (const sent of sentences) {
          const lower = sent.toLowerCase();
          let termMatches = 0;
          for (const term of queryTerms) {
            if (lower.includes(term)) termMatches++;
          }
          if (termMatches > 0) {
            const density = termMatches / Math.max(1, queryTerms.length);
            const rankWeight = 1.0 / (1.0 + chunkIdx * 0.25);
            scoredSentences.push({
              sentence: sent,
              score: density * rankWeight,
              citationIndex: ctx.index,
            });
          }
        }
      });

      scoredSentences.sort((a, b) => b.score - a.score);
      const picked: Array<{ sentence: string; citationIndex: number }> = [];
      const seen = new Set<string>();

      for (const item of scoredSentences) {
        const key = item.sentence.slice(0, 35).toLowerCase();
        if (!seen.has(key)) {
          seen.add(key);
          picked.push(item);
          if (picked.length >= 3) break;
        }
      }

      const summary = picked.length > 0
        ? `${picked[0].sentence} [${picked[0].citationIndex}]`
        : `${contexts[0].text.slice(0, 140)}... [1]`;

      const bulletPoints = picked.slice(1).map((p) => `${p.sentence} [${p.citationIndex}]`);

      self.postMessage({
        type: 'RAG_RESULT',
        summary,
        bulletPoints,
        citations,
        confidence: Math.min(0.88, 0.55 + picked.length * 0.1),
        backendUsed: 'extractive',
        modelTitle: 'Extractive Salience Ranker (Instant)',
      });
    }
  } catch (err: unknown) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    self.postMessage({ type: 'RAG_ERROR', error: errorMsg });
  }
};
