# Artifact Engine

> GPU-accelerated LLM inference from scratch. Vulkan compute. Pure C. Zero dependencies.

**Artifact Virtual · v0.1.0 · CONFIDENTIAL**

---

## What Is This

A complete LLM inference engine built from the ground up in C and GLSL — no PyTorch, no ONNX, no llama.cpp. Loads quantized GGUF model files, runs the full transformer forward pass on GPU via Vulkan compute shaders, and serves an OpenAI-compatible HTTP API.

Designed to run on **any Vulkan-capable GPU** — AMD RDNA 2 (Xbox Series X), NVIDIA, Intel Arc — without CUDA or vendor lock-in.

```
3,596 lines of C + GLSL
68 KB Linux binary, 427 KB Windows binary
9 compute shaders compiled to SPIR-V
Zero runtime dependencies beyond Vulkan + libc
```

---

## Architecture

```
                    ┌─────────────────────────────┐
                    │    HTTP API (:8080)          │
                    │    OpenAI-compatible         │
                    │    SSE streaming             │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │    Transformer Engine        │
                    │    Per-layer forward pass    │
                    │    Autoregressive generation │
                    │    Token sampling            │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │    Vulkan Compute Backend    │
                    │    Shader dispatch           │
                    │    Buffer management         │
                    │    Push constants            │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │    GGUF Loader               │
                    │    Memory-mapped I/O         │
                    │    Quantization support      │
                    │    Architecture extraction   │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │    GPU (Vulkan 1.2)          │
                    │    Any vendor: AMD/NVIDIA/   │
                    │    Intel/Xbox                │
                    └─────────────────────────────┘
```

### Components

| File | Lines | Purpose |
|------|-------|---------|
| `src/main.c` | 195 | CLI, startup, signal handling |
| `src/gguf.c` | 526 | GGUF file loader (mmap, metadata, tensor parsing) |
| `src/vulkan_compute.c` | 676 | Vulkan initialization, buffer management, shader dispatch |
| `src/engine.c` | 525 | Transformer forward pass, KV cache, sampling, generation |
| `src/http_server.c` | 571 | HTTP/1.1 server, JSON parsing, SSE streaming |
| `include/gguf.h` | 233 | GGUF format definitions, quantization types |
| `include/vulkan_compute.h` | 196 | Vulkan context, GPU buffer, dispatch API |
| `include/engine.h` | 124 | Model weights, KV cache, engine API |
| `include/http_server.h` | 48 | Server types and config |
| `shaders/*.comp` | 502 | 9 GLSL compute shaders |

---

## Build

### Quick Start (Linux)

```bash
# Install Vulkan SDK
sudo apt install vulkan-sdk libvulkan-dev glslang-tools

# Compile shaders
mkdir -p build/shaders
for s in matmul rmsnorm rope softmax silu add mul embedding dequant_q4k; do
    glslc -O shaders/${s}.comp -o build/shaders/${s}.spv
done

# Build
gcc -O2 -std=c11 -I include \
    src/main.c src/gguf.c src/vulkan_compute.c src/engine.c src/http_server.c \
    -o build/artifact-engine -lvulkan -lm -lpthread
```

### Windows Cross-Compile (MinGW)

```bash
sudo apt install gcc-mingw-w64-x86-64

x86_64-w64-mingw32-gcc -O2 -std=c11 -D_WIN32 -I include \
    src/main.c src/gguf.c src/vulkan_compute.c src/engine.c src/http_server.c \
    -o build/artifact-engine.exe -lvulkan-1 -lws2_32 -lm
```

### CMake

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

See [docs/BUILD.md](docs/BUILD.md) for full platform-specific instructions.

---

## Usage

### CLI Options

```
artifact-engine --model <path.gguf> [options]

Options:
  --model <path>    Path to GGUF model file (required)
  --port <port>     HTTP server port (default: 8080)
  --host <host>     HTTP server host (default: 0.0.0.0)
  --ctx <length>    Max context length (default: 4096)
  --shaders <dir>   Path to compiled shaders (default: ./shaders)
  --info            Print model info and exit
  --bench           Run inference benchmark
  --help            Show help
```

### Run as Server

```bash
./artifact-engine --model qwen3.5-9b-q4_k_m.gguf --port 8080
```

Output:
```
╔══════════════════════════════════════╗
║       ARTIFACT ENGINE v0.1.0         ║
║       Artifact Virtual               ║
╚══════════════════════════════════════╝

[1/4] Initializing Vulkan...
vk: GPU: AMD Radeon RX 6900 XT
vk: VRAM: 16.00 GB
[2/4] Loading model: qwen3.5-9b-q4_k_m.gguf
engine: uploading 36 layers to GPU...
engine: model loaded — 6.48 GB VRAM used
[3/4] Allocating KV cache (ctx=4096)...
[4/4] Starting HTTP server on 0.0.0.0:8080...

╔══════════════════════════════════════╗
║  Artifact Engine — Listening         ║
║  http://0.0.0.0:8080                 ║
╚══════════════════════════════════════╝
```

### Inspect Model (--info)

```bash
./artifact-engine --model qwen3.5-9b-q4_k_m.gguf --info
```

Prints GGUF metadata, architecture parameters, tensor type histogram, and data section size — without initializing Vulkan or uploading to GPU.

### Benchmark (--bench)

```bash
./artifact-engine --model qwen3.5-9b-q4_k_m.gguf --bench
```

Loads the model, tokenizes a short prompt, generates 32 tokens with greedy decoding, and reports tokens/second.

---

## API

The server exposes an **OpenAI-compatible** REST API. Any client that works with OpenAI's API works here by changing the base URL:

### Chat Completion

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,
    "max_tokens": 512,
    "stream": true
  }'
```

### Streaming (SSE)

With `"stream": true`, tokens arrive as Server-Sent Events:

```
data: {"id":"chatcmpl-...","choices":[{"delta":{"content":"The"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","choices":[{"delta":{"content":" capital"},"finish_reason":null}]}

data: [DONE]
```

### Python Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
response = client.chat.completions.create(
    model="qwen2",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

### Health Check

```bash
curl http://localhost:8080/health
# {"status":"ok","model":"qwen2","vram_used":7654321000,"vram_total":10737418240}
```

### List Models

```bash
curl http://localhost:8080/v1/models
# {"object":"list","data":[{"id":"qwen2","object":"model","owned_by":"artifact-virtual","context_length":32768}]}
```

See [docs/API.md](docs/API.md) for complete request/response schemas.

---

## Compute Shaders

9 GLSL 4.50 compute shaders, compiled to SPIR-V:

| Shader | Workgroup | Op | Bindings |
|--------|-----------|-----|----------|
| `matmul.comp` | 16×16 | C = A×B (tiled, shared memory) | 3: A, B, C |
| `rmsnorm.comp` | 256 | RMS normalization (parallel reduction) | 3: X, W, Out |
| `rope.comp` | 256 | Rotary position embedding (GQA-aware) | 2: Q, K (in-place) |
| `softmax.comp` | 256 | Numerically stable softmax (3-pass) | 1: X (in-place) |
| `silu.comp` | 256 | SiLU activation: x·σ(x) | 1: X (in-place) |
| `add.comp` | 256 | Element-wise addition | 3: A, B, Out |
| `mul.comp` | 256 | Element-wise multiplication | 3: A, B, Out |
| `embedding.comp` | 256 | Token embedding lookup | 2: Table, Out |
| `dequant_q4k.comp` | 256 | Q4_K dequant + fused matmul | 3: A_quant, B, C |

All shaders share a 32-byte push constant block for per-dispatch parameters (M, N, K, scale, offset, head_dim, n_heads).

See [docs/SHADERS.md](docs/SHADERS.md) for detailed per-shader documentation.

---

## GGUF Support

Artifact Engine loads models in the [GGUF format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) — the standard format for quantized LLMs.

### Supported Features

- **Versions:** GGUF v2 and v3
- **Memory mapping:** mmap (Linux) / MapViewOfFile (Windows) for zero-copy tensor access
- **Metadata types:** uint8–uint64, int8–int64, float32, float64, bool, string, arrays
- **Architecture extraction:** Reads vocab_size, hidden_size, n_layers, n_heads, n_kv_heads, intermediate_size, max_position, rope_freq_base, rms_norm_eps from model-prefixed keys
- **Tokenizer metadata:** BOS/EOS/PAD token IDs
- **Alignment:** Respects `general.alignment` for data section offset

### Quantization Types

All types are defined in `gguf.h` with block sizes and bytes-per-block:

| Type | Block Size | Bytes/Block | Bits/Weight | Status |
|------|-----------|-------------|-------------|--------|
| F32 | 1 | 4 | 32 | ✅ Upload to GPU |
| F16 | 1 | 2 | 16 | ✅ Upload to GPU |
| Q4_K | 256 | 144 | ~4.5 | ✅ Fused dequant+matmul shader |
| Q8_0 | 32 | 34 | ~8.5 | ⚠️ Upload only (no dequant shader yet) |
| Q4_0 | 32 | 18 | 4.5 | Defined, no shader |
| Q5_K | 256 | 176 | ~5.5 | Defined, no shader |
| Q6_K | 256 | 210 | ~6.6 | Defined, no shader |
| BF16 | 1 | 2 | 16 | Defined, no shader |

**Primary target:** Q4_K_M (Q4_K quantization with medium quality) — the best quality-to-size ratio for ~10 GB VRAM.

### Supported Architectures

The architecture extraction (`gguf_extract_arch()`) reads model-prefixed keys and works with any architecture that follows the GGUF convention. Tested/targeted:

- **qwen2** — Qwen 3.5 series (primary target)
- **llama** — LLaMA 2/3, Mistral, Yi, etc.
- Any architecture with standard attention + SwiGLU FFN

GQA (Grouped Query Attention) is supported natively — `n_kv_heads` can differ from `n_heads`.

---

## Target Hardware

### Primary: Xbox Series X

- AMD RDNA 2, ~10 GB usable VRAM, Vulkan 1.2 via GDK
- Qwen 3.5 9B Q4_K_M: 6.6 GB model + 0.5 GB KV cache = **~7.1 GB** — fits
- All LAN devices connect to `http://<xbox-ip>:8080`

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for full Xbox deployment guide.

### Also Works On

| GPU | VRAM | Max Model |
|-----|------|-----------|
| RTX 4090 | 24 GB | 30B Q4_K_M |
| RTX 3080 | 10 GB | 9B Q4_K_M |
| RTX 2050 | 4 GB | 4B Q4_K_M |
| RX 7900 XT | 20 GB | 20B Q4_K_M |
| Intel Arc A770 | 16 GB | 13B Q4_K_M |
| Xbox Series X | ~10 GB | 9B Q4_K_M |
| Any Vulkan GPU | Varies | Depends on VRAM |

---

## Performance Notes

- **Binary size:** 68 KB (Linux), 427 KB (Windows) — entire engine in a single executable
- **Startup:** Vulkan init + shader compilation takes ~1s. Model loading is I/O bound (mmap + GPU upload)
- **VRAM tracking:** Real-time via `vk_memory_used()` / `vk_memory_total()`, exposed on `/health`
- **Synchronous execution:** All GPU operations are submitted as a single command buffer per forward pass, waited on via fence. This is simple and correct; async/pipelined execution is a future optimization.
- **Memory barriers:** 11 barriers per transformer layer (conservative — every write→read dependency). Can be reduced with buffer-level barriers.
- **Matmul:** 16×16 tiled with shared memory. Competitive for medium matrices; larger tile sizes (32×32, 64×64) would improve throughput on high-end GPUs.
- **Q4_K dequant:** Fused dequant+matmul avoids materializing fp32 weights in VRAM. Current implementation is per-element (not tiled) — optimization headroom is significant.

---

## Current Status

**What works end-to-end:**
- GGUF loading, metadata extraction, weight upload to GPU
- Full transformer forward pass (embedding → N layers → logits)
- FFN with SwiGLU (gate + up + SiLU + mul + down + residual)
- RoPE with GQA support
- Token sampling with temperature
- HTTP API with SSE streaming
- Cross-platform build (Linux + Windows)

**What's still TODO:**
- Multi-head attention (QKV projections done, attention mechanism not wired)
- KV cache write-back (allocated, not populated)
- Causal masking
- BPE tokenizer (byte-level placeholder)
- See [docs/TODO.md](docs/TODO.md) for full list

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, data flow, memory management |
| [docs/API.md](docs/API.md) | OpenAI-compatible API reference |
| [docs/SHADERS.md](docs/SHADERS.md) | All 9 compute shaders: specs, algorithms, workgroups |
| [docs/BUILD.md](docs/BUILD.md) | Build for Linux, Windows (MinGW + MSVC), shader compilation |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Xbox Series X deployment guide |
| [docs/TODO.md](docs/TODO.md) | Development roadmap and status |

---

## License

Private · Artifact Virtual · All rights reserved.
