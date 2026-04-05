# Architecture

This document describes the internal architecture of Artifact Engine — how data flows from a GGUF file on disk to tokens generated over HTTP.

## Design Principles

1. **Zero dependencies.** No external libraries. The Vulkan headers are vendored. Everything else is implemented from scratch.
2. **Single binary.** One executable, under 250KB compiled. No shared libraries, no runtime requirements beyond a GPU driver (or nothing at all for the CPU backend).
3. **Layered abstraction.** Each component (GGUF parser, tokenizer, compute backend, HTTP server) is isolated behind a clean C interface. Backends are swappable at startup.
4. **Minimal allocation.** Large buffers (model weights, KV cache, activations) are allocated once at model load time. Inference runs with zero heap allocation in the hot path.

## System Overview

```
                        ┌─────────────┐
                        │   main.c    │
                        │  CLI parse  │
                        │  init loop  │
                        └──────┬──────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
       ┌──────▼──────┐ ┌──────▼──────┐ ┌───────▼───────┐
       │  engine.c   │ │http_server.c│ │ model_fetch.c │
       │  Inference  │ │  HTTP API   │ │  Downloader   │
       │  Pipeline   │ │  Listener   │ │  LAN Discover │
       └──────┬──────┘ └──────┬──────┘ └───────────────┘
              │               │
              │    ┌──────────┘
              │    │  (HTTP request triggers inference)
              ▼    ▼
       ┌─────────────────┐
       │    engine.c     │
       │ Token generation│
       │ Sampling loop   │
       └────────┬────────┘
                │
     ┌──────────┼──────────┐
     │          │          │
┌────▼────┐┌───▼───┐┌─────▼─────┐
│ gguf.c  ││token- ││  compute  │
│ Model   ││izer.c ││  backend  │
│ loader  ││ BPE   ││(selected) │
└─────────┘└───────┘└─────┬─────┘
                          │
              ┌───────────┼───────────┐
              │           │           │
       ┌──────▼──┐ ┌──────▼──┐ ┌─────▼──────┐
       │ Vulkan  │ │  CPU    │ │  DirectML  │
       │ SPIR-V  │ │ Pure C  │ │  DX12      │
       │ shaders │ │ fallback│ │  Xbox/Win  │
       └─────────┘ └─────────┘ └────────────┘
```

## Component Responsibilities

### main.c (Entry Point)

- Parses command-line arguments (`--model`, `--port`, `--backend`, `--ctx-len`, etc.)
- Initializes the engine, tokenizer, and selected compute backend
- Optionally fetches a model via HTTP or LAN discovery before loading
- Starts the HTTP server and enters the main event loop
- Handles graceful shutdown (SIGINT/SIGTERM)

### engine.c (Inference Core)

The engine is the central orchestrator. It owns:

- **Model state** — weight tensors, layer configuration, hyperparameters extracted from GGUF metadata
- **KV cache** — pre-allocated key/value buffers for each attention layer, sized to `ctx_len × n_heads × head_dim`
- **Inference loop** — the autoregressive token generation pipeline:

```
for each new token:
  1. Embed token ID → hidden state vector
  2. For each transformer layer:
     a. RMSNorm(hidden)
     b. QKV projection (matmul)
     c. Apply RoPE to Q and K
     d. Store K,V in cache at current position
     e. Attention: softmax(Q·K^T / √d) · V
     f. Output projection (matmul)
     g. Residual add
     h. RMSNorm(hidden)
     i. FFN: gate_proj, up_proj → SiLU → down_proj
     j. Residual add
  3. Final RMSNorm
  4. Language model head (matmul → logits)
  5. Sample next token from logits
  6. If EOS or max_tokens → stop
```

- **Sampling** — temperature scaling, greedy (argmax), and probabilistic sampling from the logit distribution
- **Token management** — tracks the generated sequence, manages context window, handles chat template formatting

### gguf.c (Model Loader)

Parses the GGUF binary format:

1. **Header** — magic number validation, version check (v2/v3), tensor count, metadata KV count
2. **Metadata** — reads all key-value pairs: architecture name, hidden size, head count, layer count, vocabulary, RoPE parameters, etc.
3. **Tensor index** — name, shape, data type (F32, F16, Q4_K_M, Q5_K_M, etc.), file offset for each tensor
4. **Tensor data** — maps or reads tensor data from the file into engine buffers

The parser handles all GGUF value types: uint8, int8, uint16, int16, uint32, int32, float32, float64, bool, string, and arrays thereof.

Key metadata fields extracted:

| Key | Purpose |
|-----|---------|
| `general.architecture` | Model family (llama, qwen2, mistral) |
| `*.embedding_length` | Hidden dimension size |
| `*.block_count` | Number of transformer layers |
| `*.attention.head_count` | Number of attention heads |
| `*.attention.head_count_kv` | Number of KV heads (for GQA) |
| `*.rope.freq_base` | RoPE base frequency |
| `*.context_length` | Maximum supported context |
| `tokenizer.ggml.tokens` | Vocabulary tokens |
| `tokenizer.ggml.merges` | BPE merge rules |

### tokenizer.c (BPE Tokenizer)

Full byte-pair encoding implementation:

1. **Vocabulary loading** — reads token strings and IDs from GGUF metadata
2. **Merge table** — loads ordered merge pairs (e.g., `"t h" → "th"`) and builds a priority-indexed lookup
3. **Encoding** — converts input text to token IDs:
   - UTF-8 byte splitting
   - Pre-tokenization (whitespace, punctuation boundaries)
   - Iterative merge application (highest-priority merges first)
   - Special token injection (BOS, EOS, chat template markers)
4. **Decoding** — converts token IDs back to text:
   - Token ID → string lookup
   - Byte-level token reassembly
   - UTF-8 reconstruction

Handles edge cases: multi-byte UTF-8 sequences, byte-fallback tokens (`<0x00>` through `<0xFF>`), added tokens (chat templates), and unknown token graceful fallback.

### http_server.c (API Layer)

Minimal HTTP/1.1 server built on raw POSIX sockets:

- Single-threaded accept loop with per-request handling
- Parses HTTP method, path, headers, and JSON body
- Routes to endpoint handlers
- Constructs JSON responses with proper Content-Length and Content-Type headers

Endpoints:

| Method | Path | Handler |
|--------|------|---------|
| POST | `/v1/chat/completions` | Chat completion (main inference) |
| GET | `/v1/models` | List loaded models |
| GET | `/health` | Health check (200 OK) |

The server formats chat messages using the model's chat template (Chatml for Qwen, Llama-style for Llama/Mistral) before passing to the tokenizer.

### model_fetch.c (Model Acquisition)

Three model acquisition methods:

1. **HTTP download** — fetches a GGUF file from any URL with progress reporting. Supports resume on interruption.
2. **LAN auto-discovery** — broadcasts a UDP discovery packet on the local network. Other Artifact Engine instances respond with their model inventory. Automatically pulls missing models.
3. **Xbox path scanning** — searches Xbox-specific paths (`DevelopmentFiles\`, `LocalState\`, SMB shares) for GGUF files.

### Compute Backends

All backends implement the same interface (`compute_init`, `compute_matmul`, `compute_softmax`, `compute_rmsnorm`, `compute_rope`, `compute_silu`, `compute_add`, `compute_mul`, `compute_embed`, `compute_cleanup`). The engine calls these functions without knowing which backend is active.

See [Compute Backends](compute-backends.md) for implementation details of each backend.

## Memory Layout

### Model Weights

Loaded contiguously from the GGUF file. For quantized models (Q4_K_M), weights remain in their quantized representation and are dequantized on-the-fly during computation. This minimizes memory usage — a Q4_K_M 9B model uses ~5.5GB vs ~18GB for F16.

### KV Cache

Pre-allocated at model load time:

```
Total KV cache size = 2 × n_layers × ctx_len × n_kv_heads × head_dim × sizeof(float)
```

For a 9B model with 32 layers, 8 KV heads, 128-dim heads, and 2048 context:
```
2 × 32 × 2048 × 8 × 128 × 4 = 512 MB
```

### Activation Buffers

Two ping-pong buffers of size `hidden_dim × sizeof(float)` (e.g., 2 × 4096 × 4 = 32KB for a 4096-dim model). Reused across layers — no per-layer allocation.

### Vulkan GPU Memory

When using the Vulkan backend, model weights and KV cache are uploaded to GPU device-local memory. Activation buffers use host-visible memory for CPU↔GPU transfer. Shader dispatch uses push constants for per-operation parameters (matrix dimensions, head count, etc.).

## Threading Model

Artifact Engine is single-threaded by design for the inference hot path. The HTTP server accepts one request at a time. This simplifies GPU synchronization and eliminates race conditions on the KV cache.

The CPU backend optionally uses multiple threads for matrix multiplication (controlled by `--threads`), but all other operations remain single-threaded.

## Error Handling

All functions return error codes or null pointers on failure. The engine maintains a state machine:

```
UNINITIALIZED → LOADING → READY → GENERATING → READY
                   ↓                    ↓
                 ERROR               ERROR → READY (recoverable)
```

Fatal errors (out of memory, GPU device lost, corrupt model file) cause a clean shutdown with a diagnostic message. Recoverable errors (malformed HTTP request, tokenization failure) return an error response without affecting engine state.
