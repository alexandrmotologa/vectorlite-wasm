/**
 * Pre-compiled WebAssembly SIMD Kernel (dot_f32, squared_euclidean_f32).
 * Compiled from WAT with 128-bit hardware SIMD vector instructions.
 */
export const WASM_SIMD_BASE64 = 'AGFzbQEAAAABCAFgA39/fwF9AwMCAAAFBAEBAgQHLAMGbWVtb3J5AgAHZG90X2YzMgAAFXNxdWFyZWRfZXVjbGlkZWFuX2YzMgABCt0CAqQBAwF/AXsBff0MAAAAAAAAAAAAAAAAAAAAACEEAkADQCADQQRqIAJLDQEgBCAAIANBAnRq/QAEACABIANBAnRq/QAEAP3mAf3kASEEIANBBGohAwwACwsgBP0fACAE/R8BkiAE/R8CIAT9HwOSkiEFAkADQCADIAJPDQEgBSAAIANBAnRqKgIAIAEgA0ECdGoqAgCUkiEFIANBAWohAwwACwsgBQu0AQMBfwJ7An39DAAAAAAAAAAAAAAAAAAAAAAhBAJAA0AgA0EEaiACSw0BIAAgA0ECdGr9AAQAIAEgA0ECdGr9AAQA/eUBIQUgBCAFIAX95gH95AEhBCADQQRqIQMMAAsLIAT9HwAgBP0fAZIgBP0fAiAE/R8DkpIhBgJAA0AgAyACTw0BIAAgA0ECdGoqAgAgASADQQJ0aioCAJMhByAGIAcgB5SSIQYgA0EBaiEDDAALCyAGCw==';

export function getWasmSimdBinary(): Uint8Array {
  const binaryString = atob(WASM_SIMD_BASE64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}
