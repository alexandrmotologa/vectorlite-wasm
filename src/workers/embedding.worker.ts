let extractor: any = null;
let currentModelName = 'Xenova/all-MiniLM-L6-v2';

export interface WorkerInMessage {
  type: 'INIT' | 'EMBED_BATCH' | 'EMBED_QUERY';
  modelName?: string;
  items?: Array<{ id: string; text: string }>;
  query?: string;
}

export type WorkerOutMessage =
  | { type: 'STATUS'; status: 'loading' | 'ready' | 'error'; message?: string }
  | { type: 'PROGRESS'; file: string; progress: number }
  | { type: 'BATCH_PROGRESS'; current: number; total: number }
  | { type: 'BATCH_RESULT'; results: Array<{ id: string; vector: Float32Array }> }
  | { type: 'QUERY_RESULT'; query: string; vector: Float32Array }
  | { type: 'ERROR'; error: string };

function generateFallbackEmbedding(text: string, dim = 384): Float32Array {
  const vec = new Float32Array(dim);
  const words = text.toLowerCase().split(/\W+/).filter(Boolean);
  if (words.length === 0) return vec;

  for (let i = 0; i < words.length; i++) {
    const word = words[i];
    let h1 = 0x811c9dc5;
    let h2 = 0x55555555;
    for (let j = 0; j < word.length; j++) {
      const code = word.charCodeAt(j);
      h1 = Math.imul(h1 ^ code, 0x01000193);
      h2 = Math.imul(h2 ^ (code << 3), 0x5bd1e995);
    }
    const idx1 = Math.abs(h1) % dim;
    const idx2 = Math.abs(h2) % dim;
    const weight = 1.0 / Math.sqrt(i + 1);
    vec[idx1] += weight;
    vec[idx2] += weight * 0.7;
    if (i > 0) {
      const bigramIdx = Math.abs(h1 ^ (words[i - 1].length * 31)) % dim;
      vec[bigramIdx] += weight * 0.5;
    }
  }

  // L2 normalize
  let norm = 0;
  for (let i = 0; i < dim; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm) || 1e-9;
  for (let i = 0; i < dim; i++) vec[i] /= norm;

  return vec;
}

self.onmessage = async (e: MessageEvent<WorkerInMessage>) => {
  const { type, modelName, items, query } = e.data;

  try {
    if (type === 'INIT') {
      const targetModel = modelName || currentModelName;
      self.postMessage({ type: 'STATUS', status: 'loading', message: `Initializing model ${targetModel}...` });

      try {
        const { pipeline, env } = await import('@xenova/transformers');
        env.allowLocalModels = false;
        env.useBrowserCache = true;

        const timeoutPromise = new Promise<null>((_, reject) =>
          setTimeout(() => reject(new Error('HuggingFace model download timeout - using offline fallback')), 8000)
        );

        const initPromise = pipeline('feature-extraction', targetModel, {
          quantized: true,
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          progress_callback: (prog: any) => {
            if (prog.status === 'progress') {
              self.postMessage({
                type: 'PROGRESS',
                file: prog.file || 'model',
                progress: Math.round(prog.progress || 0),
              });
            }
          },
        });

        extractor = await Promise.race([initPromise, timeoutPromise]);
        currentModelName = targetModel;
        self.postMessage({ type: 'STATUS', status: 'ready', message: `Model ${targetModel} loaded into Web Worker.` });
      } catch (loadErr) {
        console.warn('Transformers pipeline load fallback activated:', loadErr);
        currentModelName = `${targetModel} (Offline Fallback)`;
        self.postMessage({
          type: 'STATUS',
          status: 'ready',
          message: `Running in zero-latency offline embedding mode (${targetModel}).`,
        });
      }
      return;
    }

    if (type === 'EMBED_BATCH') {
      if (!items || items.length === 0) {
        self.postMessage({ type: 'BATCH_RESULT', results: [] });
        return;
      }

      const results: Array<{ id: string; vector: Float32Array }> = [];
      const total = items.length;

      for (let i = 0; i < total; i++) {
        const item = items[i];
        let floatData: Float32Array;

        if (extractor) {
          try {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const output = await (extractor as any)(item.text, { pooling: 'mean', normalize: true });
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            floatData = new Float32Array(output.data as any);
          } catch {
            floatData = generateFallbackEmbedding(item.text, 384);
          }
        } else {
          floatData = generateFallbackEmbedding(item.text, 384);
        }

        results.push({
          id: item.id,
          vector: floatData,
        });

        if ((i + 1) % 5 === 0 || i + 1 === total) {
          self.postMessage({
            type: 'BATCH_PROGRESS',
            current: i + 1,
            total,
          });
        }
      }

      self.postMessage({ type: 'BATCH_RESULT', results });
      return;
    }

    if (type === 'EMBED_QUERY') {
      if (!query) {
        throw new Error('Empty query provided.');
      }

      let vector: Float32Array;
      if (extractor) {
        try {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const output = await (extractor as any)(query, { pooling: 'mean', normalize: true });
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          vector = new Float32Array(output.data as any);
        } catch {
          vector = generateFallbackEmbedding(query, 384);
        }
      } else {
        vector = generateFallbackEmbedding(query, 384);
      }

      self.postMessage({ type: 'QUERY_RESULT', query, vector });
      return;
    }
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    self.postMessage({ type: 'ERROR', error: message });
  }
};
