# Artifact Engine

GPU-accelerated LLM inference engine built on Vulkan compute shaders. Designed to run on any GPU with Vulkan support — AMD, NVIDIA, Intel, including Xbox Series X (RDNA 2).

## Architecture

```
┌─────────────────────────────────────────────┐
│              Artifact Engine                  │
├─────────────────────────────────────────────┤
│  HTTP API (OpenAI-compatible)                │
│  /v1/chat/completions, /v1/models, /health   │
├─────────────────────────────────────────────┤
│  Transformer Forward Pass                    │
│  Token embedding → N layers → LM head        │
├─────────────────────────────────────────────┤
│  Vulkan Compute Pipeline                     │
│  Buffer management, shader dispatch,         │
│  synchronization, memory mapping             │
├─────────────────────────────────────────────┤
│  GGUF Weight Loader                          │
│  Quantized weight loading (Q4_K_M, Q8_0)     │
│  Metadata parsing, tensor mapping            │
├─────────────────────────────────────────────┤
│  Vulkan Device (RDNA 2 / Any Vulkan GPU)     │
└─────────────────────────────────────────────┘
```

## Core Compute Shaders (GLSL)

| Shader | Operation | Description |
|--------|-----------|-------------|
| matmul.comp | Matrix multiplication | Core of attention + FFN |
| attention.comp | Scaled dot-product attention | QKV → attention output |
| rmsnorm.comp | RMS Layer Normalization | Pre-attention + pre-FFN norm |
| rope.comp | Rotary Position Embedding | Position encoding |
| silu.comp | SiLU activation | FFN gate activation |
| softmax.comp | Softmax | Attention weights normalization |
| embedding.comp | Token embedding lookup | Token ID → vector |
| dequant.comp | Dequantization | Q4/Q8 → FP16/FP32 on GPU |
| add.comp | Element-wise addition | Residual connections |
| mul.comp | Element-wise multiplication | FFN gating |
| copy.comp | Buffer copy | KV cache management |
| sample.comp | Token sampling | Temperature, top-k, top-p |

## Build

```bash
# Prerequisites: Vulkan SDK, C compiler (gcc/clang/MSVC)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Or directly:
gcc -O2 -o artifact-engine src/*.c -lvulkan -lm -lpthread
```

## Run

```bash
./artifact-engine --model qwen3.5-9b-q4_k_m.gguf --port 8080
# Now accessible at http://localhost:8080/v1/chat/completions
```

## Target Hardware

- **Primary:** Xbox Series X (AMD RDNA 2, ~10GB usable VRAM)
- **Also:** Any Vulkan-capable GPU (NVIDIA, AMD, Intel)
- **Model:** Qwen 3.5 9B Q4_K_M (6.6 GB)

## Status

Phase 1: MVP — GGUF loader + Vulkan compute + HTTP API

## License

Private — Artifact Virtual
