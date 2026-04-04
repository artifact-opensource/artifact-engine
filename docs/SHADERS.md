# Artifact Engine — Compute Shaders

> Version 0.1.0 · GLSL 4.50 / SPIR-V · CONFIDENTIAL

All compute shaders are written in GLSL 4.50 (`#version 450`) and compiled to SPIR-V via `glslc` or `glslangValidator` at build time. Shaders are loaded from `{shader_dir}/*.spv` at engine startup.

Every shader shares the same **push constant layout** (32 bytes), which is used to pass per-dispatch parameters without descriptor set updates:

```glsl
layout(push_constant) uniform Params {
    uint M;        // primary dimension (rows / element count / token_id)
    uint N;        // secondary dimension (cols / hidden_size)
    uint K;        // tertiary dimension (inner dim / K tensor size)
    uint stride;   // general-purpose stride
    float scale;   // scale factor (eps / freq_base / temperature)
    uint offset;   // position offset / buffer offset
    uint head_dim; // attention head dimension
    uint n_heads;  // number of attention heads
};
```

---

## 1. matmul.comp — Tiled Matrix Multiplication

**Operation:** `C[M,N] = A[M,K] × B[K,N] × scale`

**Workgroup:** `local_size_x = 16, local_size_y = 16`

**Bindings:**
| Binding | Buffer | Access | Content |
|---------|--------|--------|---------|
| 0 | BufA | readonly | Input matrix A, float32, row-major [M × K] |
| 1 | BufB | readonly | Input matrix B, float32, row-major [K × N] |
| 2 | BufC | writeonly | Output matrix C, float32, row-major [M × N] |

**Push Constants Used:** `M`, `N`, `K`, `scale`

**Dispatch:** `groups_x = ceil(N / 16)`, `groups_y = ceil(M / 16)`, `groups_z = 1`

**Algorithm:**
- Uses 16×16 shared memory tiles (`tileA[16][16]`, `tileB[16][16]`) for coalesced global memory access
- Iterates over K dimension in tiles of 16
- Each thread computes one element of C: loads its tile contribution, synchronizes via `barrier()`, accumulates inner products
- Final value is multiplied by `scale` (defaults to 1.0 for standard matmul)
- Bounds checking: threads with `row >= M` or `col >= N` early-exit
- Zero-padding for tiles that extend past matrix boundaries

**Workgroup Sizes for Common Models:**
| Model | Hidden (H) | FFN (I) | Dispatch (H×H) | Dispatch (H×I) |
|-------|-----------|---------|----------------|----------------|
| Qwen 3.5 9B | 3584 | 18944 | 224 × 224 | 1184 × 224 |
| LLaMA 7B | 4096 | 11008 | 256 × 256 | 688 × 256 |

---

## 2. rmsnorm.comp — RMS Layer Normalization

**Operation:** `out[i] = x[i] × rsqrt(mean(x²) + eps) × weight[i]`

**Workgroup:** `local_size_x = 256`

**Bindings:**
| Binding | Buffer | Access | Content |
|---------|--------|--------|---------|
| 0 | BufX | readonly | Input vector, float32 [N] |
| 1 | BufWeight | readonly | Normalization weights, float32 [N] |
| 2 | BufOut | writeonly | Output vector, float32 [N] |

**Push Constants Used:** `M` (n_rows), `N` (hidden_size), `scale` (epsilon), `offset` (row offset)

**Dispatch:** `groups_x = 1` per row (one workgroup per token), `groups_y = 1`, `groups_z = 1`

**Algorithm:**
- **Pass 1:** Parallel sum-of-squares reduction using shared memory (`partial_sum[256]`)
  - Each thread accumulates its slice: `for (i = tid; i < N; i += 256)`
  - Tree reduction: 128 → 64 → 32 → ... → 1 step with `barrier()` synchronization
- **Pass 2:** Compute `inv_rms = 1.0 / sqrt(sum_sq / N + eps)`, then `out[i] = x[i] * inv_rms * weight[i]`
- Stride loop handles hidden_size > 256 (256 threads process N elements in chunks)

**Performance Notes:**
- Single workgroup per row — limited to 256 threads regardless of hidden_size
- For large hidden sizes (>4096), each thread processes multiple elements
- The two-pass approach requires only one `barrier()` per reduction step

---

## 3. rope.comp — Rotary Position Embedding

**Operation:** Applies rotary position encoding to Q and K tensors in-place.

For each pair `(x[2i], x[2i+1])` within a head:
```
freq = 1.0 / pow(base, 2i / head_dim)
angle = position × freq
x_new[2i]   = x[2i]   × cos(angle) - x[2i+1] × sin(angle)
x_new[2i+1] = x[2i]   × sin(angle) + x[2i+1] × cos(angle)
```

**Workgroup:** `local_size_x = 256`

**Bindings:**
| Binding | Buffer | Access | Content |
|---------|--------|--------|---------|
| 0 | BufQ | read/write | Query tensor, float32, in-place modification |
| 1 | BufK | read/write | Key tensor, float32, in-place modification |

**Push Constants Used:**
- `M` — number of tokens in batch
- `N` — total Q size per token (n_heads × head_dim)
- `K` (as `K_size`) — total K size per token (n_kv_heads × head_dim)
- `scale` — RoPE frequency base (e.g., 10000.0 for standard, 1000000.0 for extended context)
- `offset` — starting position index
- `head_dim` — dimension per head
- `n_heads` — number of Q heads

**Dispatch:** `groups_x = ceil(head_dim / 2 / 256)`, `groups_y = 1`, `groups_z = 1`

**Algorithm:**
- Each thread handles one pair index across all tokens and all heads
- Iterates over `M` tokens and `n_heads` Q heads, then `n_kv_heads` K heads (GQA-aware)
- `n_kv_heads` is derived: `K_size / head_dim`
- Frequency computation: `pow(base, 2i/dim)` uses GLSL built-in `pow()`
- Trigonometric functions: `cos()` and `sin()` on the computed angle

**Notes:**
- Supports GQA natively — Q and K can have different numbers of heads
- Operates in-place to avoid extra buffer allocation
- `offset` parameter enables incremental position encoding during autoregressive generation

---

## 4. softmax.comp — Numerically Stable Softmax

**Operation:** `softmax(x)[i] = exp(x[i] - max(x)) / Σ exp(x[j] - max(x))`

In-place, applied per row.

**Workgroup:** `local_size_x = 256`

**Bindings:**
| Binding | Buffer | Access | Content |
|---------|--------|--------|---------|
| 0 | BufX | read/write | Input/output matrix, float32, in-place [M × N] |

**Push Constants Used:** `M` (rows), `N` (cols — softmax dimension), `scale` (pre-scale factor, e.g., 1/√d_k)

**Dispatch:** `groups_x = M` (one workgroup per row), `groups_y = 1`, `groups_z = 1`

**Algorithm (three passes per row):**
1. **Pre-scale:** If `scale ≠ 1.0`, multiply all elements by `scale` (for attention scaling)
2. **Find max:** Parallel reduction over shared memory (`shared_data[256]`) — tree reduction from 128 down to 1
3. **Exp + sum:** Compute `exp(x[i] - row_max)`, write back, accumulate sum — parallel reduction for sum
4. **Normalize:** Multiply each element by `1.0 / row_sum`

**Performance Notes:**
- Three barrier-synchronized passes, each iterating `ceil(N / 256)` times
- Max subtraction prevents `exp()` overflow (numerically stable)
- Single workgroup per row — same constraint as rmsnorm for very wide dimensions

---

## 5. silu.comp — SiLU Activation

**Operation:** `silu(x) = x × sigmoid(x) = x / (1 + exp(-x))` — in-place

**Workgroup:** `local_size_x = 256`

**Bindings:**
| Binding | Buffer | Access | Content |
|---------|--------|--------|---------|
| 0 | BufX | read/write | Input/output vector, float32, in-place |

**Push Constants Used:** `M` (number of elements)

**Dispatch:** `groups_x = ceil(M / 256)`, `groups_y = 1`, `groups_z = 1`

**Algorithm:**
- Pure element-wise: each thread processes one element
- `x / (1.0 + exp(-x))` — numerically equivalent to `x * sigmoid(x)`
- Used for the FFN gate activation in SwiGLU architecture

---

## 6. add.comp — Element-wise Addition

**Operation:** `Out[i] = A[i] + B[i]`

**Workgroup:** `local_size_x = 256`

**Bindings:**
| Binding | Buffer | Access | Content |
|---------|--------|--------|---------|
| 0 | BufA | readonly | Input vector A, float32 |
| 1 | BufB | readonly | Input vector B, float32 |
| 2 | BufOut | writeonly | Output vector, float32 |

**Push Constants Used:** `M` (number of elements)

**Dispatch:** `groups_x = ceil(M / 256)`, `groups_y = 1`, `groups_z = 1`

**Usage:** Residual connections (`hidden += layer_output`)

---

## 7. mul.comp — Element-wise Multiplication

**Operation:** `Out[i] = A[i] × B[i]`

**Workgroup:** `local_size_x = 256`

**Bindings:**
| Binding | Buffer | Access | Content |
|---------|--------|--------|---------|
| 0 | BufA | readonly | Input vector A, float32 |
| 1 | BufB | readonly | Input vector B, float32 |
| 2 | BufOut | writeonly | Output vector, float32 |

**Push Constants Used:** `M` (number of elements)

**Dispatch:** `groups_x = ceil(M / 256)`, `groups_y = 1`, `groups_z = 1`

**Usage:** FFN gating (`gate_output × up_output` in SwiGLU)

---

## 8. embedding.comp — Token Embedding Lookup

**Operation:** `Out[K × N + idx] = Table[M × N + idx]` — copies one embedding row

**Workgroup:** `local_size_x = 256`

**Bindings:**
| Binding | Buffer | Access | Content |
|---------|--------|--------|---------|
| 0 | BufTable | readonly | Embedding table, float32 [vocab_size × N] |
| 1 | BufOut | writeonly | Output vector, float32 [N] |

**Push Constants Used:** `M` (token_id), `N` (embedding dim), `K` (output row offset)

**Dispatch:** `groups_x = ceil(N / 256)`, `groups_y = 1`, `groups_z = 1`

**Algorithm:**
- Copies `N` floats from row `M` (token_id) of the embedding table to row `K` of the output buffer
- `K` parameter enables writing to different positions in a batch output buffer

---

## 9. dequant_q4k.comp — Q4_K Dequantization + Fused Matrix Multiply

**Operation:** `C[row, col] = Σ_k dequant(A_q4k[row, k]) × B[k, col]`

Fuses dequantization with matrix multiplication to minimize memory bandwidth — the quantized weights are never materialized as full fp32 in VRAM.

**Workgroup:** `local_size_x = 256`

**Bindings:**
| Binding | Buffer | Access | Content |
|---------|--------|--------|---------|
| 0 | BufA (A_raw) | readonly | Quantized weight matrix, uint32 array (Q4_K packed) |
| 1 | BufB | readonly | Input activation matrix, float32 [K × N] |
| 2 | BufC | writeonly | Output matrix, float32 [M × N] |

**Push Constants Used:** `M` (rows of A/C), `N` (cols of B/C), `K` (number of columns in A, in elements)

**Dispatch:** `groups_x = ceil(M × N / 256)`, `groups_y = 1`, `groups_z = 1`

### Q4_K Block Format (144 bytes per 256 elements)

```
┌──────────────────────────────────────────────┐
│ d_dmin:  2 × fp16 packed into uint32         │  ← Super-block scales
│          d = unpackHalf2x16(d_dmin).x        │
│          dmin = unpackHalf2x16(d_dmin).y     │
├──────────────────────────────────────────────┤
│ scales:  12 bytes (3 × uint32)               │  ← 8 sub-block scales, 6-bit each
│          8 sub-blocks of 32 elements          │
│          Each has a 6-bit scale + 6-bit min   │
├──────────────────────────────────────────────┤
│ qs:      128 bytes (32 × uint32)             │  ← 256 4-bit quantized values
│          Each uint32 holds 8 nibbles          │
│          Value range: 0–15, centered at 8     │
└──────────────────────────────────────────────┘
```

**Dequantization formula:**
```
float val = block_scale × (float(nibble) - 8.0)
```

Where `block_scale = d × raw_6bit_scale` for each sub-block.

**Algorithm:**
- Each thread computes one element `C[row, col]`
- Iterates over `K / 256` blocks per row
- For each block: reads header (d, dmin), reads 3 scale words, then iterates 8 sub-blocks × 32 values
- Nibble extraction: `(packed >> (n * 4)) & 0xF`
- fp16 unpacking: uses GLSL built-in `unpackHalf2x16()`

**Performance Notes:**
- This shader is bandwidth-bound — Q4_K reduces memory access by ~4.5× vs fp32
- The fused approach avoids a separate dequant → matmul pipeline (would double memory bandwidth)
- Sub-optimal for very large matrices — each thread iterates over the entire K dimension sequentially
- Future optimization: tiled approach similar to matmul.comp, with shared memory for dequantized tiles

---

## Shader Compilation

### Build-time (via CMake)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
# Shaders are compiled as part of the build via add_custom_command
```

### Manual Compilation

```bash
# Using glslc (from Vulkan SDK)
glslc -O shaders/matmul.comp -o build/shaders/matmul.spv
glslc -O shaders/rmsnorm.comp -o build/shaders/rmsnorm.spv
glslc -O shaders/rope.comp -o build/shaders/rope.spv
glslc -O shaders/softmax.comp -o build/shaders/softmax.spv
glslc -O shaders/silu.comp -o build/shaders/silu.spv
glslc -O shaders/add.comp -o build/shaders/add.spv
glslc -O shaders/mul.comp -o build/shaders/mul.spv
glslc -O shaders/embedding.comp -o build/shaders/embedding.spv
glslc -O shaders/dequant_q4k.comp -o build/shaders/dequant_q4k.spv

# Using glslangValidator (alternative)
glslangValidator -V shaders/matmul.comp -o build/shaders/matmul.spv
```

### Compiled Sizes

```
add.spv           ~800 bytes
mul.spv           ~800 bytes
silu.spv          ~900 bytes
embedding.spv     ~900 bytes
rmsnorm.spv       ~2.4 KB
matmul.spv        ~2.8 KB
rope.spv          ~3.2 KB
softmax.spv       ~3.0 KB
dequant_q4k.spv   ~4.8 KB
```

---

## Memory Barrier Strategy

Between every shader dispatch pair that has a write→read dependency, `vk_barrier()` inserts:

```c
VkMemoryBarrier barrier = {
    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
};
vkCmdPipelineBarrier(cmd_buf,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  // src
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  // dst
    0, 1, &barrier, 0, NULL, 0, NULL);
```

This is a global memory barrier — it ensures all writes from the previous dispatch are visible to the next. It is inserted after:
- Embedding lookup (before layer processing)
- RMS norm (before QKV projections)
- QKV projections (before RoPE)
- RoPE (before attention)
- Attention output projection (before residual add)
- Residual add (before FFN norm)
- FFN norm (before gate/up projections)
- Gate/up projections (before SiLU)
- SiLU (before gate × up multiply)
- Gate × up multiply (before down projection)
- Down projection (before residual add)

Total barriers per layer: **11**. This is conservative — some could be merged or relaxed with buffer-level barriers, which is a future optimization.
