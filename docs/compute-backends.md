# Compute Backends

Artifact Engine supports three compute backends: Vulkan, CPU, and DirectML. All backends implement the same interface. This document covers the internals of each.

## Backend Interface

Every backend implements these operations:

```c
int  compute_init(Engine *eng);                    // Initialize backend
void compute_cleanup(Engine *eng);                 // Release resources

void compute_matmul(float *out, float *a, float *b, int M, int N, int K);
void compute_softmax(float *x, int len);
void compute_rmsnorm(float *out, float *x, float *weight, int len, float eps);
void compute_rope(float *q, float *k, int head_dim, int n_heads, int pos, float freq_base);
void compute_silu(float *x, int len);
void compute_add(float *out, float *a, float *b, int len);
void compute_mul(float *out, float *a, float *b, int len);
void compute_embed(float *out, float *table, int token_id, int dim);
```

The engine calls these functions through a function pointer table set at initialization. Swapping backends requires only changing which function pointers are installed — no other code changes.

---

## Vulkan Backend

**File:** `src/vulkan_compute.c` (698 lines)
**Header:** `include/vulkan_compute.h`
**Shaders:** `shaders/*.comp` (8 shaders, 310 lines GLSL total)

### Initialization

1. Create Vulkan instance (no validation layers in release, enabled in debug)
2. Enumerate physical devices, select by index (`--gpu-device`)
3. Find a compute-capable queue family
4. Create logical device and command pool
5. Load SPIR-V shader modules from `shaders/*.spv`
6. Create compute pipelines (one per shader)
7. Allocate GPU buffers for model weights, KV cache, and activation scratch space
8. Upload model weights to device-local memory

### GPU Buffer Layout

```
Device-Local Memory:
┌────────────────────────────────┐
│ Model Weights (read-only)      │  Largest allocation — full model
├────────────────────────────────┤
│ KV Cache (read-write)          │  2 × layers × ctx_len × kv_dim
├────────────────────────────────┤
│ Activation Buffer A            │  hidden_dim floats
├────────────────────────────────┤
│ Activation Buffer B            │  hidden_dim floats
├────────────────────────────────┤
│ Logits Buffer                  │  vocab_size floats
└────────────────────────────────┘

Host-Visible Memory:
┌────────────────────────────────┐
│ Staging Buffer                 │  For CPU↔GPU transfers
└────────────────────────────────┘
```

### Shader Dispatch

Each operation records a command buffer with:

1. Bind the compute pipeline for the operation
2. Bind descriptor sets pointing to input/output buffers
3. Push constants with operation parameters (dimensions, position, etc.)
4. Dispatch compute workgroups
5. Pipeline barrier (compute→compute) for data dependencies

Example — matrix multiply dispatch:

```c
vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, matmul_pipeline);
vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &desc_set, 0, NULL);

MatmulPushConstants pc = { .M = M, .N = N, .K = K };
vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

// 16×16 workgroups for matmul
vkCmdDispatch(cmd, (N + 15) / 16, (M + 15) / 16, 1);
```

### Compute Shaders

**matmul.comp** — Dense matrix multiplication
```glsl
// Workgroup: 16×16 threads
// Each thread computes one element of the output matrix
// Shared memory tiling for cache efficiency
layout(local_size_x = 16, local_size_y = 16) in;

shared float tile_A[16][16];
shared float tile_B[16][16];

void main() {
    // Tiled matrix multiply with shared memory
    float sum = 0.0;
    for (int t = 0; t < (K + 15) / 16; t++) {
        // Load tiles into shared memory
        // Compute partial dot products
        // Accumulate
    }
    C[row * N + col] = sum;
}
```

**softmax.comp** — Numerically stable row-wise softmax
```glsl
// Two-pass algorithm:
// Pass 1: find max value (for numerical stability)
// Pass 2: compute exp(x - max) and sum
// Pass 3: normalize by sum
```

**rmsnorm.comp** — RMS normalization
```glsl
// Compute: out = x * weight / sqrt(mean(x²) + eps)
// Single pass to compute mean of squares
// Then normalize each element
```

**rope.comp** — Rotary positional embedding
```glsl
// Apply rotation to Q and K vectors based on position
// For each pair (x_2i, x_2i+1):
//   cos_θ = cos(pos * freq_base^(-2i/dim))
//   sin_θ = sin(pos * freq_base^(-2i/dim))
//   out_2i   = x_2i * cos_θ - x_2i+1 * sin_θ
//   out_2i+1 = x_2i * sin_θ + x_2i+1 * cos_θ
```

**silu.comp** — SiLU activation (SwiGLU component)
```glsl
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

**add.comp, mul.comp** — Element-wise operations
```glsl
// Residual connections and element-wise gating
```

**embed.comp** — Token embedding lookup
```glsl
// out[i] = embedding_table[token_id * dim + i]
```

### Synchronization

The Vulkan backend batches operations within a transformer layer into a single command buffer submission. Pipeline barriers enforce data dependencies between operations (e.g., matmul output must be visible before softmax reads it). Only one `vkQueueSubmit` + `vkQueueWaitIdle` per generated token to minimize synchronization overhead.

### Device Selection

When `--list-devices` is passed:

```
Available Vulkan devices:
  0: NVIDIA GeForce RTX 4090    (discrete, 24576 MB)
  1: Intel UHD Graphics 770     (integrated, 4096 MB)

Use --gpu-device N to select.
```

---

## CPU Backend

**File:** `src/cpu_compute.c` (378 lines)
**Header:** `include/cpu_compute.h`

### Design

Pure C implementation with no assembly or intrinsics. Memory layout is SIMD-friendly (contiguous float arrays, aligned to 64 bytes) so the compiler auto-vectorizes critical loops with `-O2` or higher.

### Matrix Multiply

The CPU matmul uses a loop-tiled approach:

```c
void cpu_matmul(float *out, float *A, float *B, int M, int N, int K) {
    // Zero output
    memset(out, 0, M * N * sizeof(float));

    // Tiled multiplication — better cache behavior than naive triple loop
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = A[i * K + k];
            for (int j = 0; j < N; j++) {
                out[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}
```

The inner loop over `j` is auto-vectorized by GCC/Clang. The `i,k,j` loop order keeps `B` access sequential, maximizing L1 cache hits.

### Multi-threading

When `--threads N` is specified (N > 1), matrix multiplication splits work across threads using pthreads. Each thread handles a contiguous range of output rows:

```c
// Thread i handles rows [start, end) where:
// start = i * (M / n_threads)
// end = (i + 1) * (M / n_threads)   (last thread handles remainder)
```

All other operations (softmax, rmsnorm, rope, silu, add, mul, embed) run single-threaded — they are memory-bandwidth-bound, not compute-bound, and threading overhead exceeds the benefit.

### Performance

The CPU backend is significantly slower than Vulkan for inference, but it:
- Requires zero setup (no GPU driver, no Vulkan SDK)
- Produces a tiny binary (~235KB on Linux with static linking)
- Runs on any machine with a C compiler
- Is useful for testing, CI/CD, and resource-constrained environments

---

## DirectML Backend

**File:** `src/directml_compute.c` (748 lines)

### Purpose

DirectML (Direct Machine Learning) is a DirectX 12-based API for GPU compute on Windows and Xbox. The DirectML backend enables Artifact Engine to run on:

- Xbox Series X|S (RDNA 2 GPU, 10–12 TFLOPS)
- Windows PCs with any DX12 GPU
- Scenarios where Vulkan is unavailable

### Initialization

1. Create D3D12 device
2. Create DirectML device on top of D3D12
3. Create command queue, command allocator, command list
4. Create operator descriptors for each operation (matmul, softmax, etc.)
5. Compile operators into a reusable command list
6. Allocate GPU buffers (D3D12 heaps) for weights, KV cache, activations

### Frame Capture Pipeline

The DirectML backend includes a unique feature: frame capture. This enables a "game companion" architecture where Artifact Engine:

1. Captures the current frame from a running game (via D3D12 shared surfaces)
2. Encodes the frame as a visual token (future: VLM integration)
3. Feeds it into the LLM context alongside text

This is designed for Xbox scenarios where the engine runs alongside a game, providing real-time AI assistance based on what's happening on screen.

### Xbox-Specific Considerations

- UWP sandbox: file access restricted to app package and DevelopmentFiles
- Memory budget: must stay within the UWP memory limit (~1GB for background apps)
- No Vulkan on Xbox: DirectML is the only GPU compute path
- Model discovery: scans DevelopmentFiles and SMB shares (see [Model Fetching](model-fetching.md))

---

## Backend Comparison

| Feature | Vulkan | CPU | DirectML |
|---------|--------|-----|----------|
| GPU acceleration | ✅ | ❌ | ✅ |
| Linux | ✅ | ✅ | ❌ |
| Windows | ✅ | ✅ | ✅ |
| Xbox | ❌ | ❌ | ✅ |
| Binary size | ~250KB | ~235KB | ~300KB |
| Setup required | Vulkan SDK | None | DX12 runtime |
| Frame capture | ❌ | ❌ | ✅ |
| Multi-thread | N/A (GPU) | Optional | N/A (GPU) |
| Shader format | SPIR-V | N/A | HLSL (compiled) |

## Adding a New Backend

To add a backend (e.g., Metal for macOS):

1. Create `src/metal_compute.c` and `include/metal_compute.h`
2. Implement all functions in the compute interface
3. Add a backend selection case in `engine.c`:
   ```c
   if (strcmp(backend, "metal") == 0) {
       eng->compute_matmul = metal_matmul;
       eng->compute_softmax = metal_softmax;
       // ... etc
   }
   ```
4. Add the new source file to `CMakeLists.txt`
5. Gate compilation with a CMake option (e.g., `-DMETAL=ON`)

The engine does not need any other changes — the function pointer interface isolates backends completely.
