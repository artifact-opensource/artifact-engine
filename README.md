<p align="center">
  <img src="Assets/StoreLogo.png" alt="Artifact Engine" width="120" />
</p>

<h1 align="center">Artifact Engine</h1>

<p align="center">
  <strong>GPU-accelerated LLM inference. Vulkan compute. Pure C. Zero dependencies.</strong>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#features">Features</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#api">API</a> ·
  <a href="docs/">Documentation</a>
</p>

---

Artifact Engine is a from-scratch LLM inference runtime written in pure C with Vulkan compute shaders. No frameworks. No Python. No runtime dependencies beyond a GPU driver. It loads GGUF models, tokenizes with a built-in BPE implementation, runs multi-head grouped-query attention on the GPU, and serves completions over an OpenAI-compatible HTTP API.

The entire stack — GGUF parser, BPE tokenizer, KV cache, Vulkan pipeline, HTTP server, model fetcher — is implemented in ~5,800 lines of C and GLSL. The compiled binary is under 250KB.

**Designed for:** Running real LLMs (Qwen, Llama, Mistral) on constrained hardware — Xbox Series X, laptops with 4GB VRAM, headless Linux servers — with no ecosystem overhead.

---

## Quickstart

**Linux (Vulkan)**
```bash
# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run
./artifact-engine --model models/qwen-3.5-9b-q4_k_m.gguf --port 8080
```

**Linux (CPU-only, no GPU required)**
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCPU_ONLY=ON
make -j$(nproc)

./artifact-engine --model models/qwen-3.5-9b-q4_k_m.gguf --port 8080 --backend cpu
```

**Windows (MSVC)**
```cmd
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release

artifact-engine.exe --model models\qwen-3.5-9b-q4_k_m.gguf --port 8080
```

**Xbox Series X**
```cmd
build_xbox.bat
package_xbox.bat
:: Deploy via Xbox Device Portal
curl -sk -u user:pass -X POST -F "file=@ArtifactEngine.appx" ^
  https://XBOX_IP:11443/api/app/packagemanager/package
```

**Query the API**
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-3.5-9b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

---

## Features

**Inference**
- GGUF model loading (v2/v3) with full metadata parsing
- BPE tokenizer with Unicode normalization, special token handling, and merge-pair decoding
- Multi-head grouped-query attention (GQA) with RoPE positional encoding
- KV cache with configurable context length
- Greedy and temperature-based sampling
- Streaming token generation

**Compute Backends**
- **Vulkan** — SPIR-V compute shaders for matrix multiply, softmax, RMSNorm, RoPE, residual add, element-wise ops. Runs on any Vulkan 1.0+ GPU.
- **CPU** — Pure C fallback. SIMD-friendly memory layout, no GPU required. Verified on Qwen 3.5 9B at 235KB binary size.
- **DirectML** — Windows/Xbox backend for RDNA 2+ GPUs. Frame capture pipeline for game companion use cases.

**Networking**
- OpenAI-compatible HTTP API (`/v1/chat/completions`, `/v1/models`)
- Built-in HTTP model fetcher — download GGUF files from URLs
- LAN auto-discovery — pull models from other Artifact Engine instances on the local network via UDP broadcast
- Xbox SMB discovery — auto-detect models in DevelopmentFiles and local app paths

**Platforms**
- Linux (x86_64, aarch64)
- Windows 10/11 (MSVC)
- Xbox Series X|S (UWP, packaged as .appx)

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                     main.c                           │
│              CLI parsing · server init               │
├──────────────────────────────────────────────────────┤
│                    engine.c                          │
│         Model orchestration · inference loop         │
│         Token generation · sampling · KV cache       │
├────────────┬────────────┬────────────────────────────┤
│  gguf.c    │tokenizer.c │      http_server.c         │
│  Model I/O │ BPE encode │  OpenAI-compatible API     │
│  Metadata  │ BPE decode │  /v1/chat/completions      │
│  Tensors   │ Merges     │  /v1/models                │
├────────────┴────────────┴────────────────────────────┤
│               Compute Backend (selectable)           │
│  ┌──────────────┬──────────────┬──────────────────┐  │
│  │vulkan_compute│ cpu_compute  │directml_compute  │  │
│  │ SPIR-V       │ Pure C       │ DirectX 12       │  │
│  │ Any GPU      │ Any CPU      │ Xbox / Windows   │  │
│  └──────────────┴──────────────┴──────────────────┘  │
├──────────────────────────────────────────────────────┤
│                   model_fetch.c                      │
│       HTTP download · LAN discovery · Xbox SMB       │
└──────────────────────────────────────────────────────┘
```

**Source breakdown:**

| Component | File | Lines | Purpose |
|-----------|------|------:|---------|
| Entry point | `src/main.c` | 437 | CLI, config, server lifecycle |
| Engine core | `src/engine.c` | 546 | Model load, inference, token generation |
| GGUF parser | `src/gguf.c` | 479 | Binary format parsing, tensor extraction |
| Tokenizer | `src/tokenizer.c` | 491 | BPE encode/decode, merge pairs, special tokens |
| HTTP server | `src/http_server.c` | 467 | Socket server, OpenAI API, JSON response |
| Vulkan backend | `src/vulkan_compute.c` | 698 | Device init, pipeline, shader dispatch |
| CPU backend | `src/cpu_compute.c` | 378 | Matrix ops, attention, softmax in pure C |
| DirectML backend | `src/directml_compute.c` | 748 | DX12 compute, frame capture, game companion |
| Model fetcher | `src/model_fetch.c` | 534 | HTTP download, LAN broadcast, Xbox paths |
| Compute shaders | `shaders/*.comp` | 310 | GLSL 450 → SPIR-V, 8 shader kernels |

---

## Compute Shaders

Eight GLSL 450 compute shaders compiled to SPIR-V:

| Shader | Operation | Workgroup |
|--------|-----------|-----------|
| `matmul.comp` | Dense matrix multiply (MxNxK) | 16×16 |
| `softmax.comp` | Row-wise softmax with numerical stability | 256×1 |
| `rmsnorm.comp` | RMS normalization with learned scale | 256×1 |
| `rope.comp` | Rotary positional embedding | 256×1 |
| `add.comp` | Element-wise vector addition (residuals) | 256×1 |
| `mul.comp` | Element-wise vector multiply | 256×1 |
| `silu.comp` | SiLU activation (x · σ(x)) | 256×1 |
| `embed.comp` | Token embedding lookup | 256×1 |

Compile shaders:
```bash
cd shaders
for f in *.comp; do
  glslangValidator -V "$f" -o "${f%.comp}.spv"
done
```

---

## API Reference

### POST /v1/chat/completions

OpenAI-compatible chat completion endpoint.

**Request:**
```json
{
  "model": "qwen-3.5-9b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain transformers in one paragraph."}
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-artifact-1",
  "object": "chat.completion",
  "created": 1743811200,
  "model": "qwen-3.5-9b",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Transformers are a neural network architecture..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 87,
    "total_tokens": 111
  }
}
```

### GET /v1/models

List loaded models.

**Response:**
```json
{
  "object": "list",
  "data": [{
    "id": "qwen-3.5-9b",
    "object": "model",
    "owned_by": "artifact-engine"
  }]
}
```

### GET /health

Returns `200 OK` when the server is ready.

---

## Supported Models

Any GGUF-formatted model with a supported architecture. Tested with:

| Model | Parameters | Quantization | VRAM Required |
|-------|-----------|-------------|---------------|
| Qwen 2.5 3B | 3B | Q4_K_M | ~2.5 GB |
| Qwen 3.5 9B | 9B | Q4_K_M | ~6 GB |
| Llama 3.1 8B | 8B | Q4_K_M | ~5.5 GB |
| Mistral 7B | 7B | Q4_K_M | ~5 GB |

Place `.gguf` files in the `models/` directory, or pass `--model /path/to/model.gguf`.

---

## Building from Source

**Prerequisites:**
- C compiler (GCC 11+, Clang 14+, or MSVC 2022)
- CMake 3.16+
- Vulkan SDK (for GPU backend) — or build with `-DCPU_ONLY=ON`
- `glslangValidator` (for shader compilation, included in Vulkan SDK)

**Linux:**
```bash
git clone https://github.com/artifact-opensource/artifact-engine.git
cd artifact-engine
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**Windows:**
```cmd
git clone https://github.com/artifact-opensource/artifact-engine.git
cd artifact-engine
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

**CMake Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `CPU_ONLY` | `OFF` | Build without Vulkan (CPU backend only) |
| `CMAKE_BUILD_TYPE` | `Release` | `Debug` / `Release` / `RelWithDebInfo` |

---

## Configuration

All configuration is via command-line flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--model PATH` | *required* | Path to GGUF model file |
| `--port N` | `8080` | HTTP server port |
| `--backend TYPE` | `vulkan` | Compute backend: `vulkan`, `cpu`, `directml` |
| `--ctx-len N` | `2048` | Maximum context length |
| `--threads N` | *auto* | CPU threads (CPU backend) |
| `--gpu-device N` | `0` | Vulkan physical device index |
| `--fetch URL` | — | Download model from URL before starting |
| `--lan-pull` | — | Auto-discover and pull models from LAN |

---

## Xbox Deployment

Artifact Engine runs natively on Xbox Series X|S as a UWP application, using the DirectML compute backend.

1. Enable **Developer Mode** on the Xbox
2. Build and package:
   ```cmd
   build_xbox.bat
   package_xbox.bat
   ```
3. Deploy via Xbox Device Portal:
   ```cmd
   curl -sk -u user:pass -X POST -F "file=@ArtifactEngine.appx" ^
     https://XBOX_IP:11443/api/app/packagemanager/package
   ```
4. Models are auto-discovered from `DevelopmentFiles\` on the Xbox or pulled via LAN from a host machine running Artifact Engine with `--lan-pull`.

See [docs/xbox-deployment.md](docs/xbox-deployment.md) for the complete guide.

---

## Project Status

**Current version: v0.5.0**

| Milestone | Status |
|-----------|--------|
| GGUF parser (v2/v3) | ✅ Complete |
| BPE tokenizer | ✅ Complete |
| Multi-head GQA attention | ✅ Complete |
| KV cache | ✅ Complete |
| Vulkan compute backend | ✅ Complete |
| CPU compute backend | ✅ Complete |
| DirectML compute backend | ✅ Complete |
| OpenAI-compatible HTTP API | ✅ Complete |
| HTTP model fetcher | ✅ Complete |
| LAN model auto-discovery | ✅ Complete |
| Xbox packaging & deployment | ✅ Complete |
| Frame capture (game companion) | ✅ Complete |
| Streaming responses | 🔄 In progress |
| Batched inference | 📋 Planned |
| LoRA adapter loading | 📋 Planned |
| Metal compute backend (macOS) | 📋 Planned |

---

## License

Private · [Artifact Virtual](https://artifactvirtual.com) · All rights reserved.
