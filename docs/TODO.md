# Artifact Engine — TODO

> Pinned development items · CONFIDENTIAL

Items are ordered by priority. Items marked with the current status reflect what exists in the codebase as of v0.1.0.

---

## 🔴 Critical (Required for functional inference)

### Multi-Head Attention with GQA
- **Status:** Scaffolded but not implemented
- **Location:** `src/engine.c` → `forward_layer()` steps 4–7
- **What exists:** QKV projections are computed and RoPE is applied. The attention output projection (Wo) and residual connection are wired. Everything between RoPE and the output projection is TODO.
- **What's needed:**
  1. **KV cache write-back:** After RoPE, copy current K and V into `cache.k[layer]` and `cache.v[layer]` at position `pos`. Requires a buffer copy dispatch or a dedicated copy shader.
  2. **Attention score computation:** For each Q head, compute `Q_head @ K_cache^T` (matmul with the full cached K up to `seq_len`). Must handle GQA — multiple Q heads share a single KV head (`n_kv_heads < n_heads`, typically `n_kv_heads = n_heads / group_size`).
  3. **Causal masking:** Apply causal mask before softmax — positions > current position should be `-inf` (or very large negative). Needs either a dedicated mask shader or integration into the softmax shader.
  4. **Softmax:** Already implemented. Apply per-head over the sequence dimension with `scale = 1.0 / sqrt(head_dim)`.
  5. **Attention output:** Compute `softmax_scores @ V_cache` — weighted sum of cached values.
  6. **Concatenate heads:** Reshape multi-head output back to `[hidden_size]`.
- **Approach options:**
  - A) Compose from existing shaders (matmul + softmax + copy) — simpler, more dispatches + barriers
  - B) Fused attention shader (like Flash Attention) — better performance, more complex GLSL
  - Recommend A first, then optimize to B

### BPE Tokenizer
- **Status:** Byte-level placeholder
- **Location:** `src/engine.c` → `engine_tokenize()` and `engine_detokenize()`
- **What exists:** BOS token + raw byte-to-index mapping. This produces incorrect tokenization for all real models.
- **What's needed:**
  1. Extract BPE merge rules from GGUF metadata (`tokenizer.ggml.merges` array)
  2. Extract vocabulary tokens from GGUF (`tokenizer.ggml.tokens` array)
  3. Implement BPE merge algorithm: iteratively merge the most frequent byte/token pairs
  4. Handle special tokens: `<|im_start|>`, `<|im_end|>`, BOS, EOS, PAD
  5. Detokenization: map token IDs back to strings via the vocabulary array
- **Impact:** Without proper BPE, the model receives garbage input and produces garbage output. This is the #1 blocker for actual usable inference.

---

## 🟡 Important (Performance / correctness)

### Q4_K_M Fused Dequant + Matmul Performance
- **Status:** Implemented but sub-optimal
- **Location:** `shaders/dequant_q4k.comp`
- **Current approach:** Each thread computes one output element by iterating over the entire K dimension — O(K) work per thread with no shared memory tiling
- **Optimization needed:**
  1. Tiled approach (like `matmul.comp`) — load dequantized tiles into shared memory, then accumulate
  2. Vectorized loads (`uvec4` instead of `uint`) for better bandwidth utilization
  3. Warp-level operations for sub-block dequantization
  4. Consider separate dequant → tiled matmul pipeline if fused shader complexity is too high
- **Expected impact:** 3-5× throughput improvement on bandwidth-bound models

### KV Cache Write-Back
- **Status:** Cache allocation exists, write-back does not
- **Location:** `src/engine.c` → `forward_layer()` step 4
- **What's needed:** After computing and RoPE-encoding K and V for the current position, copy them into the appropriate cache slots. This requires either:
  - A `vk_copy` dispatch using a copy shader (SHADER_COPY is allocated but placeholder)
  - OR use `vkCmdCopyBuffer` with offset calculations for the correct position within the cache
- **Dependency:** Required by multi-head attention (above)

### Causal Masking
- **Status:** Not implemented
- **What's needed:** During attention score computation, positions beyond the current position must be masked out (set to -inf before softmax). Options:
  1. Modify `softmax.comp` to accept a mask parameter
  2. Add a dedicated `mask.comp` shader that applies the causal mask in-place
  3. Build the mask into the attention score computation
- **Dependency:** Required by multi-head attention

---

## 🟢 Enhancement (Quality / features)

### Batch Inference Support
- **Status:** Forward pass processes one token at a time
- **Location:** `src/engine.c` → `engine_forward()`
- **What's needed:** Process the entire prompt in parallel during prefill (instead of token-by-token). Scratch buffers would need to be `[seq_len × dim]` instead of `[1 × dim]`. Matmul workgroups would handle M > 1 rows. KV cache would be written in bulk.
- **Impact:** Dramatic prefill speedup (currently O(n) dispatches, should be O(1) for the prompt)

### Thread Pool for HTTP Server
- **Status:** Synchronous single-connection handling
- **Location:** `src/http_server.c` → `handle_request()` called in accept loop
- **What's needed:** `pthread_create` per connection (simple) or a fixed thread pool with work queue (better). Since inference is GPU-bound and single-threaded on the host side, the main benefit is keeping the accept loop responsive during generation.
- **Platform note:** Windows would need `CreateThread` or `_beginthreadex`

### Large Request Body Handling
- **Status:** Single 64 KB `recv()` call
- **Location:** `src/http_server.c` → `handle_request()`
- **What's needed:** Parse `Content-Length` header, read the full body in a loop until all bytes received. Required for prompts with long conversation history.

### Top-K and Top-P Sampling
- **Status:** Temperature + basic probability sampling implemented. Top-k and top-p are stubbed.
- **Location:** `src/engine.c` → `sample_token()`
- **What's needed:**
  - Top-k: partial sort to find k-th largest logit, zero out everything below
  - Top-p: sort probabilities descending, cumulative sum, cutoff at threshold
  - Both need an efficient sort on `vocab_size` elements (~32K–150K). Can be CPU-side since it runs once per token.

### Non-Streaming Response (Full Generation Buffer)
- **Status:** Placeholder message returned
- **Location:** `src/http_server.c` → `handle_chat_completions()` (non-streaming branch)
- **What's needed:** Collect all generated tokens into a buffer (via callback), then build the full response JSON. The streaming path works correctly; the non-streaming path needs the collection buffer.

### Repetition Penalty
- **Status:** Defined in `sample_params` (1.1) but not applied
- **What's needed:** Track recently generated token IDs, divide their logits by `repeat_penalty` before sampling

### GPU-Side Sampling
- **Status:** SHADER_SAMPLE allocated but placeholder
- **What's needed:** Temperature scaling and argmax on GPU to avoid downloading the full logits vector to CPU. Would save one `vk_download()` per token.

---

## 🔵 Nice-to-Have (Polish / research)

### Dedicated Q8_0 Shader
- SHADER_MATMUL_Q8 currently maps to `matmul.spv` (fp32 matmul) — no actual Q8 dequantization
- Q8_0 format: 32 bytes per block of 32 elements (34 bytes = 32 × int8 + fp16 scale)
- Simpler than Q4_K but still benefits from fused dequant+matmul

### Buffer Copy Shader
- SHADER_COPY currently maps to `matmul.spv` (placeholder)
- Needed for KV cache management and general buffer operations

### Fused Attention Shader
- Replace the compose-from-primitives attention with a single fused shader
- Flash Attention-style: tiled Q×K^T computation with online softmax
- Would dramatically reduce memory barriers and temporary buffers

### Memory Pool / Sub-allocator
- Current approach: one `vkAllocateMemory` per buffer
- Vulkan has a limit on simultaneous allocations (~4096 on some drivers)
- A sub-allocator (allocate large chunks, sub-divide) would be more efficient

### Multi-Model Support
- Currently single model loaded per process
- Allow runtime model switching or multiple models with separate engines

### Tensor Parallelism
- Split model across multiple GPUs (e.g., if running on a multi-GPU system)
- Would require inter-device buffer copies and synchronized dispatch

### Weight Tying Detection
- Currently tries to load `output.weight`, falls back to `token_embd.weight`
- Could explicitly detect weight tying from GGUF metadata

### Chat Template Auto-Detection
- Currently hardcoded ChatML format (`<|im_start|>`)
- Should read `tokenizer.chat_template` from GGUF metadata (Jinja2 format used by HF)
- Different models use different templates (ChatML, LLaMA, Alpaca, etc.)

---

## 📊 Status Summary

| Component | Status |
|-----------|--------|
| GGUF loader (mmap, metadata, tensor info) | ✅ Complete |
| GGUF architecture extraction | ✅ Complete |
| Vulkan init (device, queue, shaders, pipelines) | ✅ Complete |
| GPU buffer management (alloc, upload, download, free) | ✅ Complete |
| Compute dispatch + barriers | ✅ Complete |
| 9 compute shaders (compiled to SPIR-V) | ✅ Complete |
| Weight upload to GPU (all layers) | ✅ Complete |
| KV cache allocation | ✅ Complete |
| Embedding lookup | ✅ Complete |
| RMS normalization | ✅ Complete |
| QKV projections | ✅ Complete |
| RoPE (GQA-aware) | ✅ Complete |
| FFN (SwiGLU: gate + up + SiLU + mul + down) | ✅ Complete |
| Residual connections | ✅ Complete |
| Final norm + LM head | ✅ Complete |
| Token sampling (temperature + basic) | ✅ Complete |
| Autoregressive generation loop | ✅ Complete |
| HTTP server (POSIX sockets) | ✅ Complete |
| SSE streaming | ✅ Complete |
| /v1/chat/completions | ✅ Complete |
| /v1/models | ✅ Complete |
| /health | ✅ Complete |
| CORS support | ✅ Complete |
| Cross-platform (Linux + Windows) | ✅ Complete |
| **KV cache write-back** | 🔴 TODO |
| **Multi-head attention** | 🔴 TODO |
| **Causal masking** | 🔴 TODO |
| **BPE tokenizer** | 🔴 TODO |
| Q4_K_M perf optimization | 🟡 TODO |
| Batch prefill | 🟢 TODO |
| Thread pool | 🟢 TODO |
| Top-k / top-p sampling | 🟢 TODO |
| Non-streaming response buffer | 🟢 TODO |
