# Artifact Engine — System Architecture

> Version 0.1.0 · Artifact Virtual · CONFIDENTIAL

## Overview

Artifact Engine is a from-scratch GPU-accelerated LLM inference engine built on the Vulkan compute API. It loads quantized GGUF model files, dequantizes weights on-GPU via compute shaders, runs the full transformer forward pass, and exposes an OpenAI-compatible HTTP API — all in ~3,600 lines of C and GLSL with zero external dependencies beyond Vulkan and libc.

The design targets **any Vulkan-capable GPU** — AMD RDNA 2 (Xbox Series X), NVIDIA, Intel Arc — without requiring CUDA, ROCm, or vendor-specific compute frameworks.

---

## System Layers

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   HTTP/1.1 Server  (src/http_server.c, 571 lines)                    │
│   ─ POSIX sockets, no dependencies                                   │
│   ─ OpenAI-compatible JSON API                                       │
│   ─ SSE streaming for token-by-token output                          │
│   ─ CORS headers for browser clients                                 │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Transformer Engine  (src/engine.c, 525 lines)                      │
│   ─ Model weight upload (GGUF → GPU)                                 │
│   ─ KV cache allocation and management                               │
│   ─ Layer-by-layer forward pass orchestration                        │
│   ─ Token sampling (temperature, top-k, top-p)                       │
│   ─ Autoregressive generation loop                                   │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Vulkan Compute Backend  (src/vulkan_compute.c, 676 lines)          │
│   ─ Device discovery (prefers discrete GPU)                          │
│   ─ SPIR-V shader loading + pipeline creation                        │
│   ─ GPU buffer management (device-local + staging)                   │
│   ─ Compute dispatch with push constants                             │
│   ─ Memory barriers + synchronization                                │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   GGUF Loader  (src/gguf.c, 526 lines)                               │
│   ─ Memory-mapped file I/O (mmap on Linux, MapViewOfFile on Windows) │
│   ─ Header + metadata + tensor info parsing                          │
│   ─ Architecture extraction (Qwen2, LLaMA, etc.)                     │
│   ─ Zero-copy tensor data access via mmap pointers                   │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   9 Compute Shaders  (shaders/*.comp, GLSL 4.50)                     │
│   ─ matmul, rmsnorm, rope, softmax, silu, add, mul, embedding,      │
│     dequant_q4k                                                      │
│   ─ Compiled to SPIR-V at build time via glslc/glslangValidator      │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Vulkan 1.2 Driver  (AMD, NVIDIA, Intel, Xbox)                      │
│   ─ Compute queue only — no graphics pipeline, no render passes      │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Startup Sequence

```
main.c
  │
  ├─→ [1] engine_init()
  │     └─→ vk_init()
  │           ├─ vkCreateInstance (Vulkan 1.2, "Artifact Engine")
  │           ├─ Enumerate physical devices → pick best (discrete > integrated)
  │           ├─ Find compute queue family (prefer dedicated compute)
  │           ├─ vkCreateDevice + get compute queue
  │           ├─ Create command pool, command buffer, fence
  │           ├─ Create descriptor pool (1024 storage buffer descriptors, 256 sets)
  │           └─ For each of 15 shader IDs:
  │                 ├─ Load SPIR-V from disk
  │                 ├─ vkCreateShaderModule
  │                 ├─ Create descriptor set layout (1-4 storage buffer bindings)
  │                 ├─ Create pipeline layout (with push_constants)
  │                 └─ vkCreateComputePipelines
  │
  ├─→ [2] engine_load_model()
  │     ├─→ gguf_load() (mmap the file)
  │     │     ├─ Validate magic (0x46554747 = "GGUF")
  │     │     ├─ Parse version, n_tensors, n_kv
  │     │     ├─ Read all key-value metadata pairs
  │     │     ├─ Read all tensor info (name, dims, type, offset)
  │     │     ├─ Compute tensor byte sizes from quantization type
  │     │     └─ Locate data section (aligned to 32 bytes or general.alignment)
  │     │
  │     ├─→ gguf_extract_arch()
  │     │     └─ Read: vocab_size, hidden_size, n_layers, n_heads,
  │     │        n_kv_heads, intermediate_size, max_position,
  │     │        rope_freq_base, rms_norm_eps, tokenizer IDs
  │     │
  │     └─→ Upload weights to GPU (per-tensor: staging buffer → device copy)
  │           ├─ token_embd.weight                [vocab_size × hidden_size]
  │           ├─ blk.{L}.attn_norm.weight         [hidden_size] × n_layers
  │           ├─ blk.{L}.attn_q.weight            [hidden_size × n_heads·head_dim]
  │           ├─ blk.{L}.attn_k.weight            [hidden_size × n_kv_heads·head_dim]
  │           ├─ blk.{L}.attn_v.weight            [hidden_size × n_kv_heads·head_dim]
  │           ├─ blk.{L}.attn_output.weight       [n_heads·head_dim × hidden_size]
  │           ├─ blk.{L}.ffn_norm.weight          [hidden_size] × n_layers
  │           ├─ blk.{L}.ffn_gate.weight          [hidden_size × intermediate_size]
  │           ├─ blk.{L}.ffn_up.weight            [hidden_size × intermediate_size]
  │           ├─ blk.{L}.ffn_down.weight          [intermediate_size × hidden_size]
  │           ├─ output_norm.weight                [hidden_size]
  │           └─ output.weight (or tied to token_embd) [hidden_size × vocab_size]
  │
  ├─→ [3] engine_alloc_cache()
  │     ├─ KV cache: k[n_layers], v[n_layers], each [max_seq × n_kv_heads × head_dim]
  │     └─ Scratch buffers: hidden, residual, norm_out, q, k, v, attn_out,
  │        attn_scores, ffn_gate_out, ffn_up_out, ffn_down_out, logits
  │
  └─→ [4] server_start()  (blocks, accepting connections)
```

### Inference Flow (Single Token Generation)

```
HTTP POST /v1/chat/completions
  │
  ├─→ parse_messages() → extract {role, content} pairs
  ├─→ Build prompt with ChatML template (<|im_start|> format)
  ├─→ engine_tokenize() → byte-level token IDs + BOS
  │
  ├─→ engine_generate(prompt_tokens, params, callback)
  │     │
  │     ├─→ engine_forward(prompt_tokens)    ← Prefill phase
  │     │     │
  │     │     │  For each token in prompt:
  │     │     ├─→ vk_begin_compute()
  │     │     ├─→ vk_embedding()             ← Token → hidden state
  │     │     ├─→ vk_barrier()
  │     │     │
  │     │     │  For each layer L:
  │     │     ├─→ forward_layer(L, position)
  │     │     │     │
  │     │     │     ├─ [1] vk_rmsnorm(hidden → norm_out, attn_norm_weight)
  │     │     │     ├─ [2] vk_matmul(norm_out @ Wq → q)
  │     │     │     │      vk_matmul(norm_out @ Wk → k)
  │     │     │     │      vk_matmul(norm_out @ Wv → v)
  │     │     │     ├─ [3] vk_rope(q, k, position, freq_base)
  │     │     │     ├─ [4] TODO: KV cache write-back
  │     │     │     ├─ [5] TODO: Multi-head attention with GQA
  │     │     │     ├─ [6] TODO: Causal mask + softmax
  │     │     │     ├─ [7] TODO: Attention @ V
  │     │     │     ├─ [8] vk_matmul(attn_out @ Wo → proj)
  │     │     │     ├─ [9] vk_add(hidden + proj → hidden)     ← Residual
  │     │     │     ├─ [10] vk_rmsnorm(hidden → norm_out, ffn_norm_weight)
  │     │     │     ├─ [11] vk_matmul(norm_out @ Wgate → gate)
  │     │     │     │       vk_matmul(norm_out @ Wup → up)
  │     │     │     ├─ [12] vk_silu(gate)
  │     │     │     ├─ [13] vk_mul(gate * up → gate)
  │     │     │     ├─ [14] vk_matmul(gate @ Wdown → ffn_out)
  │     │     │     └─ [15] vk_add(hidden + ffn_out → hidden)  ← Residual
  │     │     │
  │     │     ├─→ vk_rmsnorm(hidden → norm_out, output_norm)
  │     │     ├─→ vk_matmul(norm_out @ output_weight → logits)
  │     │     └─→ vk_submit_and_wait()      ← GPU executes all queued ops
  │     │
  │     └─→ Decode loop (token-by-token):
  │           ├─ vk_download(logits → CPU)
  │           ├─ sample_token(logits, params) → next_token_id
  │           ├─ stream_token_cb() → SSE chunk to HTTP client
  │           └─ engine_forward(&next_token, 1) → next logits
  │
  └─→ send_sse_done()  ← "data: [DONE]\n\n"
```

---

## Memory Management

### Three Memory Domains

```
┌─────────────────────────┐     ┌────────────────────────┐
│     System RAM (CPU)     │     │       VRAM (GPU)        │
│                          │     │                         │
│  ┌──────────────────┐   │     │  ┌─────────────────┐   │
│  │  GGUF mmap       │───────────→ (zero-copy read)  │   │
│  │  (entire file)   │   │     │  │                  │   │
│  └──────────────────┘   │     │  │  Model Weights   │   │
│                          │     │  │  (device-local)  │   │
│  ┌──────────────────┐   │     │  │                  │   │
│  │  Staging Buffers  │◄─────────→ (upload/download) │   │
│  │  (host-visible,   │   │     │  │  KV Cache       │   │
│  │   host-coherent)  │   │     │  │  (device-local)  │   │
│  └──────────────────┘   │     │  │                  │   │
│                          │     │  │  Scratch Buffers │   │
│  ┌──────────────────┐   │     │  │  (device-local)  │   │
│  │  CPU logits buf   │   │     │  │                  │   │
│  │  (for sampling)   │   │     │  │  Logits Output   │   │
│  └──────────────────┘   │     │  └─────────────────┘   │
└─────────────────────────┘     └────────────────────────┘
```

### GGUF Memory Mapping

The GGUF file is memory-mapped (`mmap` on Linux, `MapViewOfFile` on Windows) for zero-copy access to tensor data. The kernel's virtual memory system pages in data on demand — no explicit read of the entire file into RAM.

- `MADV_SEQUENTIAL` hint on Linux for optimal readahead during initial weight upload
- Tensor data pointers are direct offsets into the mmap region: `(uint8_t*)gf->data + tensor->offset`
- The mmap stays live for the process lifetime but is only actively used during `engine_load_model()` — once weights are uploaded to GPU, the mmap pages can be evicted by the OS under memory pressure

### GPU Buffer Allocation

Two buffer types via `vk_alloc_buffer()`:

| Type | Usage Flags | Memory Properties | Purpose |
|------|------------|-------------------|---------|
| **Device-local** | STORAGE_BUFFER \| TRANSFER_DST \| TRANSFER_SRC | DEVICE_LOCAL | Model weights, KV cache, scratch buffers — fast GPU access |
| **Staging** | TRANSFER_SRC \| TRANSFER_DST | HOST_VISIBLE \| HOST_COHERENT | Temporary buffers for CPU↔GPU data transfer |

Upload path: `vk_upload()` allocates a temporary staging buffer → `memcpy()` from CPU → `vkCmdCopyBuffer()` → wait on fence → free staging. Download path: inverse.

Memory tracking: `ctx->used_memory` is incremented on every allocation and decremented on free, available via `vk_memory_used()`.

### KV Cache Layout

```
For each layer L (0 to n_layers-1):
  cache.k[L]:  [max_seq_len × n_kv_heads × head_dim] float32
  cache.v[L]:  [max_seq_len × n_kv_heads × head_dim] float32
```

Cache is pre-allocated for the full context window. `cache.seq_len` tracks how many positions are filled. Reset via `engine_reset()` (sets seq_len = 0, buffers remain allocated).

**Note:** KV cache write-back from the scratch Q/K/V buffers into the cache is currently a TODO — the scaffolding (allocation, pointers, position tracking) is in place but the copy shader dispatch is not yet wired.

### Scratch Buffer Layout

All scratch buffers are sized for single-token inference (the generation use case). Sizes for a model with hidden_size `H`, intermediate_size `I`, vocab_size `V`, `n_heads` attention heads, `n_kv_heads` KV heads, `head_dim` `d`:

| Buffer | Size (bytes) | Purpose |
|--------|-------------|---------|
| hidden | H × 4 | Current hidden state |
| residual | H × 4 | Residual connection storage |
| norm_out | H × 4 | Output of RMS normalization |
| q | n_heads × d × 4 | Query projection |
| k | n_kv_heads × d × 4 | Key projection |
| v | n_kv_heads × d × 4 | Value projection |
| attn_out | H × 4 | Attention output |
| attn_scores | n_heads × max_seq × 4 | Attention score matrix |
| ffn_gate_out | I × 4 | FFN gate projection |
| ffn_up_out | I × 4 | FFN up projection |
| ffn_down_out | H × 4 | FFN down projection |
| logits | V × 4 | Final logit output |

---

## Shader Pipeline

### Pipeline Architecture

All shaders share a common push constant structure:

```c
typedef struct {
    uint32_t M;        // rows / element count / token_id
    uint32_t N;        // cols / hidden_size
    uint32_t K;        // inner dim / K size
    uint32_t stride;   // general-purpose stride
    float    scale;    // temperature / eps / freq_base
    uint32_t offset;   // position offset / buffer offset
    uint32_t head_dim; // attention head dimension
    uint32_t n_heads;  // number of attention heads
} push_constants;     // 32 bytes total
```

Each shader pipeline has:
- A `VkDescriptorSetLayout` with 1–4 storage buffer bindings
- A `VkPipelineLayout` with the descriptor set + push constant range
- A `VkComputePipeline` with `main` as entry point

### Dispatch Flow

```
vk_begin_compute()       ← Reset + begin command buffer
  │
  ├─ vk_dispatch(shader, buffers, push_constants, group_counts)
  │    ├─ Allocate descriptor set from pool
  │    ├─ Write buffer descriptors
  │    ├─ Bind pipeline + descriptors
  │    ├─ Push constants
  │    └─ vkCmdDispatch(gx, gy, gz)
  │
  ├─ vk_barrier()         ← Shader write → shader read memory barrier
  │    └─ vkCmdPipelineBarrier(COMPUTE → COMPUTE)
  │
  ├─ ... (more dispatches + barriers) ...
  │
  └─ vk_submit_and_wait() ← End command buffer, submit to queue, wait on fence
       └─ vkResetDescriptorPool() ← Free all descriptor sets for reuse
```

### Shader-to-Pipeline Mapping

15 `shader_id` enum values map to 9 physical SPIR-V files. Several IDs share the same shader (with different push constants) or use placeholders:

| shader_id | SPIR-V File | Bindings | Status |
|-----------|-------------|----------|--------|
| SHADER_MATMUL | matmul.spv | 3 (A, B, C) | ✅ Implemented |
| SHADER_MATMUL_Q4K | dequant_q4k.spv | 3 (A_quant, B, C) | ✅ Implemented |
| SHADER_MATMUL_Q8 | matmul.spv | 3 | Placeholder (TODO: Q8 shader) |
| SHADER_ATTENTION | matmul.spv | 3 | Placeholder (TODO: fused attention) |
| SHADER_RMSNORM | rmsnorm.spv | 3 (X, W, Out) | ✅ Implemented |
| SHADER_ROPE | rope.spv | 2 (Q, K) | ✅ Implemented |
| SHADER_SILU | silu.spv | 1 (X, in-place) | ✅ Implemented |
| SHADER_SOFTMAX | softmax.spv | 1 (X, in-place) | ✅ Implemented |
| SHADER_EMBEDDING | embedding.spv | 2 (Table, Out) | ✅ Implemented |
| SHADER_DEQUANT_Q4K | dequant_q4k.spv | 3 | ✅ Implemented |
| SHADER_DEQUANT_Q8 | matmul.spv | 3 | Placeholder |
| SHADER_ADD | add.spv | 3 (A, B, Out) | ✅ Implemented |
| SHADER_MUL | mul.spv | 3 (A, B, Out) | ✅ Implemented |
| SHADER_COPY | matmul.spv | 2 | Placeholder (TODO: copy shader) |
| SHADER_SAMPLE | matmul.spv | 2 | Placeholder (TODO: GPU sampling) |

---

## Vulkan Initialization Details

### Device Selection

The engine enumerates all physical devices and scores them:
- Discrete GPU: score 100
- Integrated GPU: score 50
- Other (CPU/virtual): score 10

Highest score wins. Within the selected device, the engine prefers a **dedicated compute queue** (one without `VK_QUEUE_GRAPHICS_BIT`) for non-contention with any display workloads. Falls back to a combined graphics+compute queue if needed.

### API Version

Vulkan 1.2 is required (`VK_API_VERSION_1_2`). No extensions are explicitly enabled — the engine uses only core Vulkan 1.2 features. This maximizes portability across drivers and platforms.

### Descriptor Pool

A single `VkDescriptorPool` with capacity for 1024 storage buffer descriptors across 256 sets. After each `vk_submit_and_wait()`, the pool is reset (`vkResetDescriptorPool`) to reclaim all sets. This avoid-fragmentation approach works because inference is synchronous — all descriptor sets from a batch are freed together.

---

## Source File Map

```
artifact-engine/
├── include/
│   ├── gguf.h              233 lines — GGUF format defs, quant types, tensor info, API
│   ├── vulkan_compute.h    196 lines — Vulkan context, GPU buffers, shader dispatch API
│   ├── engine.h            124 lines — Model weights, KV cache, forward pass, generation API
│   ├── http_server.h        48 lines — Chat message types, server config, start/stop API
│   └── vulkan/              Vulkan SDK headers (vendored)
│       └── *.h, *.hpp
├── src/
│   ├── main.c              195 lines — CLI parsing, startup orchestration, signal handling
│   ├── gguf.c              526 lines — GGUF loader (mmap, binary reader, metadata extraction)
│   ├── vulkan_compute.c    676 lines — Vulkan init, buffer mgmt, shader dispatch, cleanup
│   ├── engine.c            525 lines — Weight upload, forward pass, sampling, generation
│   └── http_server.c       571 lines — HTTP server, JSON parsing, SSE streaming, routing
├── shaders/
│   ├── matmul.comp          70 lines — Tiled 16×16 matrix multiplication
│   ├── rmsnorm.comp         60 lines — RMS normalization with parallel reduction
│   ├── rope.comp            81 lines — Rotary position embedding
│   ├── softmax.comp         82 lines — Numerically stable softmax
│   ├── silu.comp            29 lines — SiLU activation
│   ├── add.comp             23 lines — Element-wise addition
│   ├── mul.comp             23 lines — Element-wise multiplication
│   ├── embedding.comp       24 lines — Token embedding lookup
│   └── dequant_q4k.comp   110 lines — Q4_K dequantization + fused matmul
├── build/
│   ├── artifact-engine      68 KB Linux ELF binary
│   ├── artifact-engine.exe 427 KB Windows PE binary
│   └── shaders/*.spv        Compiled SPIR-V shaders
└── CMakeLists.txt          CMake build configuration
```

**Total:** 3,596 lines of C + GLSL (excluding vendored Vulkan headers).

---

## Cross-Platform Strategy

The codebase compiles for both Linux and Windows with platform-specific abstractions:

| Feature | Linux | Windows |
|---------|-------|---------|
| Memory mapping | `mmap()` / `munmap()` | `CreateFileMapping()` / `MapViewOfFile()` |
| Sockets | POSIX sockets | Winsock2 (`WSAStartup`, `closesocket`) |
| Threading | `pthread` | Not yet (TODO: `CreateThread`) |
| Signal handling | `signal(SIGPIPE, SIG_IGN)` | N/A |
| Vulkan | `libvulkan.so` | `vulkan-1.dll` |

The CMakeLists.txt handles platform-specific linking (`ws2_32` on Windows, `pthread` on Linux).
