const fs = require('fs');
const path = require('path');
const wabt = require('wabt');

async function buildWasm() {
  const wabtModule = await wabt();

  const wat = `
  (module
    (memory (export "memory") 2 4)

    ;; 1. SIMD Float32 Dot Product
    (func (export "dot_f32") (param $ptrA i32) (param $ptrB i32) (param $len i32) (result f32)
      (local $i i32)
      (local $sum v128)
      (local $total f32)
      (local.set $sum (v128.const i32x4 0 0 0 0))

      ;; SIMD 4-lane Loop
      (block $break_simd
        (loop $loop_simd
          (br_if $break_simd (i32.gt_u (i32.add (local.get $i) (i32.const 4)) (local.get $len)))
          (local.set $sum
            (f32x4.add
              (local.get $sum)
              (f32x4.mul
                (v128.load (i32.add (local.get $ptrA) (i32.shl (local.get $i) (i32.const 2))))
                (v128.load (i32.add (local.get $ptrB) (i32.shl (local.get $i) (i32.const 2))))
              )
            )
          )
          (local.set $i (i32.add (local.get $i) (i32.const 4)))
          (br $loop_simd)
        )
      )

      ;; Horizontal sum of 4 lanes
      (local.set $total
        (f32.add
          (f32.add (f32x4.extract_lane 0 (local.get $sum)) (f32x4.extract_lane 1 (local.get $sum)))
          (f32.add (f32x4.extract_lane 2 (local.get $sum)) (f32x4.extract_lane 3 (local.get $sum)))
        )
      )

      ;; Remainder scalar loop
      (block $break_rem
        (loop $loop_rem
          (br_if $break_rem (i32.ge_u (local.get $i) (local.get $len)))
          (local.set $total
            (f32.add
              (local.get $total)
              (f32.mul
                (f32.load (i32.add (local.get $ptrA) (i32.shl (local.get $i) (i32.const 2))))
                (f32.load (i32.add (local.get $ptrB) (i32.shl (local.get $i) (i32.const 2))))
              )
            )
          )
          (local.set $i (i32.add (local.get $i) (i32.const 1)))
          (br $loop_rem)
        )
      )

      (local.get $total)
    )

    ;; 2. SIMD Float32 Squared Euclidean Distance
    (func (export "squared_euclidean_f32") (param $ptrA i32) (param $ptrB i32) (param $len i32) (result f32)
      (local $i i32)
      (local $sum v128)
      (local $diff v128)
      (local $total f32)
      (local $d f32)
      (local.set $sum (v128.const i32x4 0 0 0 0))

      (block $break_simd
        (loop $loop_simd
          (br_if $break_simd (i32.gt_u (i32.add (local.get $i) (i32.const 4)) (local.get $len)))
          (local.set $diff
            (f32x4.sub
              (v128.load (i32.add (local.get $ptrA) (i32.shl (local.get $i) (i32.const 2))))
              (v128.load (i32.add (local.get $ptrB) (i32.shl (local.get $i) (i32.const 2))))
            )
          )
          (local.set $sum
            (f32x4.add
              (local.get $sum)
              (f32x4.mul (local.get $diff) (local.get $diff))
            )
          )
          (local.set $i (i32.add (local.get $i) (i32.const 4)))
          (br $loop_simd)
        )
      )

      (local.set $total
        (f32.add
          (f32.add (f32x4.extract_lane 0 (local.get $sum)) (f32x4.extract_lane 1 (local.get $sum)))
          (f32.add (f32x4.extract_lane 2 (local.get $sum)) (f32x4.extract_lane 3 (local.get $sum)))
        )
      )

      (block $break_rem
        (loop $loop_rem
          (br_if $break_rem (i32.ge_u (local.get $i) (local.get $len)))
          (local.set $d
            (f32.sub
              (f32.load (i32.add (local.get $ptrA) (i32.shl (local.get $i) (i32.const 2))))
              (f32.load (i32.add (local.get $ptrB) (i32.shl (local.get $i) (i32.const 2))))
            )
          )
          (local.set $total
            (f32.add (local.get $total) (f32.mul (local.get $d) (local.get $d)))
          )
          (local.set $i (i32.add (local.get $i) (i32.const 1)))
          (br $loop_rem)
        )
      )

      (local.get $total)
    )
  )
  `;

  const parsed = wabtModule.parseWat('simd_kernel.wat', wat, { simd: true });
  const { buffer } = parsed.toBinary({ log: false });

  const base64 = Buffer.from(buffer).toString('base64');
  const targetFile = path.join(__dirname, '..', 'src', 'engine', 'wasm_simd_binary.ts');

  const content = `/**
 * Pre-compiled WebAssembly SIMD Kernel (dot_f32, squared_euclidean_f32).
 * Compiled from WAT with 128-bit hardware SIMD vector instructions.
 */
export const WASM_SIMD_BASE64 = '${base64}';

export function getWasmSimdBinary(): Uint8Array {
  const binaryString = atob(WASM_SIMD_BASE64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}
`;

  fs.writeFileSync(targetFile, content, 'utf8');
  console.log(`Successfully compiled WebAssembly SIMD kernel (${buffer.length} bytes) to ${targetFile}`);
}

buildWasm().catch(console.error);
