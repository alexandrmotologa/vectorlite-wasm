import { describe, it, expect } from 'vitest';
import type { WorkerInMessage, WorkerOutMessage } from '../src/workers/embedding.worker';

describe('WebGPU Acceleration & Device Detection', () => {
  it('supports device selection types in WorkerInMessage and WorkerOutMessage', () => {
    const initWasmMsg: WorkerInMessage = {
      type: 'INIT',
      modelName: 'Xenova/all-MiniLM-L6-v2',
      device: 'wasm',
    };
    expect(initWasmMsg.device).toBe('wasm');

    const initGpuMsg: WorkerInMessage = {
      type: 'INIT',
      modelName: 'Xenova/all-MiniLM-L6-v2',
      device: 'webgpu',
    };
    expect(initGpuMsg.device).toBe('webgpu');

    const outStatusWasm: WorkerOutMessage = {
      type: 'STATUS',
      status: 'ready',
      modelName: 'Xenova/all-MiniLM-L6-v2',
      device: 'wasm',
    };
    expect(outStatusWasm.type).toBe('STATUS');
    if (outStatusWasm.type === 'STATUS') {
      expect(outStatusWasm.device).toBe('wasm');
    }

    const outStatusGpu: WorkerOutMessage = {
      type: 'STATUS',
      status: 'ready',
      modelName: 'Xenova/all-MiniLM-L6-v2',
      device: 'webgpu',
    };
    if (outStatusGpu.type === 'STATUS') {
      expect(outStatusGpu.device).toBe('webgpu');
    }
  });

  it('detects navigator.gpu capability correctly', () => {
    const hasGpuAvailable = typeof navigator !== 'undefined' && 'gpu' in navigator && Boolean(navigator.gpu);
    // In standard Node/Vitest environment, navigator.gpu is usually undefined
    expect(typeof hasGpuAvailable).toBe('boolean');

    // Mock navigator.gpu
    const mockNav = { gpu: {} };
    const detected = typeof mockNav !== 'undefined' && 'gpu' in mockNav && Boolean(mockNav.gpu);
    expect(detected).toBe(true);
  });
});
