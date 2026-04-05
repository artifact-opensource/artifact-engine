# Changelog

All notable changes to Artifact Engine.

## v0.5.0 — DirectML Backend + Frame Capture

**Date:** 2026-04-04

Three new modules for the AI companion architecture:

- **DirectML compute backend** — `src/directml_compute.c` (748 lines). Full DX12/DirectML implementation supporting matrix multiply, softmax, RMSNorm, RoPE, SiLU, element-wise ops, and embedding lookup. Targets Xbox Series X|S (RDNA 2) and Windows PCs with DX12 GPUs.
- **Frame capture pipeline** — captures D3D12 frames from running games via shared surfaces. Enables the game companion architecture where the LLM has visual context of what's on screen.
- **Xbox process lifetime management** — proper UWP lifecycle handling (suspend/resume), background task registration, and memory budget enforcement for Xbox background execution.

## v0.4.0 — Xbox Process Lifetime + Edge Download Paths

**Date:** 2026-04-04

- Fixed Xbox UWP process lifetime management for background execution
- Added Edge browser download path scanning for model auto-discovery
- Improved model path resolution on Windows

## v0.3.0 — HTTP Model Fetcher + LAN Discovery

**Date:** 2026-04-04

- **HTTP model fetcher** — download GGUF models from any URL with `--fetch`. Progress reporting, resume support, redirect following.
- **LAN auto-discovery** — UDP broadcast protocol for finding Artifact Engine instances on the local network. Automatically pulls models from discovered peers.
- **Xbox SMB discovery** — scans Xbox DevelopmentFiles and network SMB shares for GGUF files.
- **`model_fetch.c`** — 534 lines. New module: `src/model_fetch.c` + `include/model_fetch.h`.

## v0.2.2 — Xbox Model Auto-Detect

**Date:** 2026-04-03

- Auto-detect GGUF models in Xbox DevelopmentFiles and local app storage paths
- Path scanning for standard Xbox sideload locations

## v0.2.1 — CPU Backend + Windows Build

**Date:** 2026-04-03

- **CPU compute backend** — `src/cpu_compute.c` (378 lines). Pure C matrix operations, multi-head attention, softmax, RMSNorm, RoPE, SiLU. No GPU required.
- **Windows MSVC build** — CMake configuration for Visual Studio 2022. Produces a 235KB binary.
- Verified inference on Qwen 3.5 9B with CPU backend.

## v0.2.0 — BPE Tokenizer

**Date:** 2026-04-03

- **BPE tokenizer** — `src/tokenizer.c` (491 lines). Full byte-pair encoding with merge table, Unicode handling, special tokens, byte-fallback reassembly.
- Vocabulary and merge rules loaded directly from GGUF metadata.
- Chat template support (Chatml, Llama format).

## v0.1.1 — Multi-Head GQA Attention + KV Cache

**Date:** 2026-04-02

- Multi-head grouped-query attention (GQA) with configurable head count ratio
- RoPE (rotary positional embedding) in the attention layer
- KV cache with configurable context length — pre-allocated at model load time
- Autoregressive token generation loop with temperature sampling

## v0.1.0 — Initial Release

**Date:** 2026-04-02

First release. GPU-accelerated LLM inference from scratch.

- **GGUF parser** — `src/gguf.c` (479 lines). Parses GGUF v2/v3 headers, metadata, tensor descriptors, and weight data.
- **Vulkan compute backend** — `src/vulkan_compute.c` (698 lines). 8 SPIR-V compute shaders: matmul, softmax, RMSNorm, RoPE, SiLU, add, mul, embed.
- **HTTP server** — `src/http_server.c` (467 lines). OpenAI-compatible `/v1/chat/completions` and `/v1/models` endpoints.
- **Engine core** — `src/engine.c` (546 lines). Model loading, inference loop, token generation, sampling.
- 3,596 lines of C + GLSL. Zero dependencies beyond Vulkan.
- Target: Qwen 3.5 9B on Xbox Series X (RDNA 2, Vulkan).
- Xbox packaging: AppxManifest.xml, build_xbox.bat, package_xbox.bat.
