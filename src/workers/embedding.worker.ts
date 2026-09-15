import { pipeline, env, FeatureExtractionPipeline } from '@xenova/transformers';

// Configure transformers.js for in-browser client execution
env.allowLocalModels = false;
env.useBrowserCache = true;

let extractor: FeatureExtractionPipeline | null = null;
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

self.onmessage = async (e: MessageEvent<WorkerInMessage>) => {
  const { type, modelName, items, query } = e.data;

  try {
    if (type === 'INIT') {
      const targetModel = modelName || currentModelName;
      self.postMessage({ type: 'STATUS', status: 'loading', message: `Initializing model ${targetModel}...` });

      extractor = await pipeline('feature-extraction', targetModel, {
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

      currentModelName = targetModel;
      self.postMessage({ type: 'STATUS', status: 'ready', message: `Model ${targetModel} loaded into Web Worker.` });
      return;
    }

    if (type === 'EMBED_BATCH') {
      if (!extractor) {
        throw new Error('Embedding pipeline not initialized. Send INIT first.');
      }
      if (!items || items.length === 0) {
        self.postMessage({ type: 'BATCH_RESULT', results: [] });
        return;
      }

      const results: Array<{ id: string; vector: Float32Array }> = [];
      const total = items.length;

      for (let i = 0; i < total; i++) {
        const item = items[i];
        // Generate embedding with mean pooling and L2 normalization
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const output = await (extractor as any)(item.text, { pooling: 'mean', normalize: true });
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const floatData = new Float32Array(output.data as any);

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
      if (!extractor) {
        throw new Error('Embedding pipeline not initialized.');
      }
      if (!query) {
        throw new Error('Empty query provided.');
      }

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const output = await (extractor as any)(query, { pooling: 'mean', normalize: true });
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const vector = new Float32Array(output.data as any);

      self.postMessage({ type: 'QUERY_RESULT', query, vector });
      return;
    }
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    self.postMessage({ type: 'ERROR', error: message });
  }
};
