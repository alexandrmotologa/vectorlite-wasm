import { getWasmSimdBinary } from './wasm_simd_binary';

interface WasmSIMDExports {
  memory: WebAssembly.Memory;
  dot_f32: (ptrA: number, ptrB: number, len: number) => number;
  squared_euclidean_f32: (ptrA: number, ptrB: number, len: number) => number;
}

class WasmSIMDKernel {
  private instance: WasmSIMDExports | null = null;
  private memoryView: Float32Array | null = null;
  private supported = false;

  constructor() {
    this.init();
  }

  public get isSupported(): boolean {
    return this.supported;
  }

  public get isActive(): boolean {
    return this.instance !== null;
  }

  private init(): void {
    if (typeof WebAssembly === 'undefined' || typeof atob === 'undefined') {
      this.supported = false;
      return;
    }

    try {
      const bytes = getWasmSimdBinary();
      const bufferSource = bytes.buffer as BufferSource;
      const isValid = WebAssembly.validate(bufferSource);
      if (!isValid) {
        this.supported = false;
        return;
      }

      const mod = new WebAssembly.Module(bufferSource);
      const inst = new WebAssembly.Instance(mod);
      this.instance = inst.exports as unknown as WasmSIMDExports;
      this.memoryView = new Float32Array(this.instance.memory.buffer);
      this.supported = true;
    } catch (err) {
      console.warn('WebAssembly SIMD kernel initialization fallback:', err);
      this.supported = false;
      this.instance = null;
      this.memoryView = null;
    }
  }

  /**
   * Computes dot product using hardware 128-bit SIMD WebAssembly kernel.
   * If WASM SIMD is not active or supported, returns null to fall back to unrolled JS.
   */
  public dot(a: Float32Array, b: Float32Array): number | null {
    if (!this.instance || !this.memoryView) return null;
    const len = a.length;
    if (len !== b.length) return null;

    // Slot A at index 0, Slot B at index 512 (allowing up to 512 dimensions in page 1)
    if (len > 512) {
      return null;
    }

    // Refresh memory view if buffer detached
    if (this.memoryView.buffer !== this.instance.memory.buffer) {
      this.memoryView = new Float32Array(this.instance.memory.buffer);
    }

    this.memoryView.set(a, 0);
    this.memoryView.set(b, 512);

    const ptrA = 0;
    const ptrB = 512 * 4;

    return this.instance.dot_f32(ptrA, ptrB, len);
  }

  /**
   * Computes squared Euclidean distance using hardware 128-bit SIMD WebAssembly kernel.
   */
  public squaredEuclidean(a: Float32Array, b: Float32Array): number | null {
    if (!this.instance || !this.memoryView) return null;
    const len = a.length;
    if (len !== b.length || len > 512) return null;

    if (this.memoryView.buffer !== this.instance.memory.buffer) {
      this.memoryView = new Float32Array(this.instance.memory.buffer);
    }

    this.memoryView.set(a, 0);
    this.memoryView.set(b, 512);

    const ptrA = 0;
    const ptrB = 512 * 4;

    return this.instance.squared_euclidean_f32(ptrA, ptrB, len);
  }
}

export const wasmSimdEngine = new WasmSIMDKernel();
