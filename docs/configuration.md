# Configuration

All Artifact Engine configuration is done through command-line flags. There are no configuration files.

## Command-Line Reference

### Required

| Flag | Description |
|------|-------------|
| `--model PATH` | Path to the GGUF model file. Can be absolute or relative. |

### Server

| Flag | Default | Description |
|------|---------|-------------|
| `--port N` | `8080` | HTTP server listen port. |
| `--host ADDR` | `0.0.0.0` | Bind address. Use `127.0.0.1` to restrict to localhost. |

### Compute

| Flag | Default | Description |
|------|---------|-------------|
| `--backend TYPE` | `vulkan` | Compute backend: `vulkan`, `cpu`, or `directml`. |
| `--gpu-device N` | `0` | Vulkan physical device index. Use `--list-devices` to see available GPUs. |
| `--threads N` | *auto-detect* | Number of CPU threads for the CPU backend. Defaults to the number of logical cores. |

### Inference

| Flag | Default | Description |
|------|---------|-------------|
| `--ctx-len N` | `2048` | Maximum context length (tokens). Determines KV cache size. |
| `--temp F` | `0.7` | Default sampling temperature. Overridden per-request via the API. |
| `--max-tokens N` | `512` | Default maximum tokens to generate. Overridden per-request via the API. |

### Model Fetching

| Flag | Default | Description |
|------|---------|-------------|
| `--fetch URL` | — | Download a GGUF model from a URL before starting. Saves to `models/`. |
| `--lan-pull` | — | Enable LAN auto-discovery. Broadcasts a UDP probe and pulls models from responding peers. |
| `--lan-port N` | `8081` | UDP port for LAN model discovery. |

### Diagnostics

| Flag | Description |
|------|-------------|
| `--help` | Print usage and exit. |
| `--version` | Print version and exit. |
| `--list-devices` | List available Vulkan devices and exit. |
| `--verbose` | Enable verbose logging to stderr. |

## Usage Examples

**Basic — load a model and serve on default port:**
```bash
./artifact-engine --model models/qwen-3.5-9b-q4_k_m.gguf
```

**CPU backend with custom thread count:**
```bash
./artifact-engine --model models/qwen-3.5-9b-q4_k_m.gguf --backend cpu --threads 8
```

**Specific GPU on a multi-GPU system:**
```bash
# List available GPUs
./artifact-engine --list-devices

# Use the second GPU
./artifact-engine --model models/qwen-3.5-9b-q4_k_m.gguf --gpu-device 1
```

**Extended context with custom port:**
```bash
./artifact-engine --model models/qwen-3.5-9b-q4_k_m.gguf --ctx-len 4096 --port 9090
```

**Download a model and start serving:**
```bash
./artifact-engine --fetch https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf --port 8080
```

**LAN model pull — find models on other machines:**
```bash
./artifact-engine --lan-pull --model auto --port 8080
```

**Localhost-only binding for security:**
```bash
./artifact-engine --model models/qwen-3.5-9b-q4_k_m.gguf --host 127.0.0.1 --port 8080
```

## Memory Requirements

The primary memory consumers are model weights and the KV cache. Use this to plan your `--ctx-len` setting:

**Model weights (approximate, Q4_K_M quantization):**

| Model Size | Weight Memory |
|-----------|---------------|
| 3B params | ~2.0 GB |
| 7B params | ~4.5 GB |
| 9B params | ~5.5 GB |
| 13B params | ~8.0 GB |

**KV cache per 1024 context tokens:**

| Model Hidden Dim | KV Heads | Cache / 1K tokens |
|-------------------|----------|-------------------|
| 2048 (3B) | 4 | ~32 MB |
| 4096 (7-9B) | 8 | ~128 MB |
| 5120 (13B) | 8 | ~160 MB |

**Example:** Qwen 3.5 9B with `--ctx-len 2048` requires ~5.5 GB (weights) + ~256 MB (KV cache) ≈ **5.75 GB** total.

## Backend Selection Guide

| Scenario | Recommended Backend |
|----------|-------------------|
| Linux with NVIDIA/AMD/Intel GPU | `vulkan` |
| Linux server, no GPU | `cpu` |
| Windows with any GPU | `vulkan` or `directml` |
| Xbox Series X/S | `directml` |
| Minimal binary, portability | `cpu` |
| Lowest latency on modern GPU | `vulkan` |

## Environment Variables

Artifact Engine does not read environment variables for configuration. All settings are explicit via CLI flags. This is intentional — no hidden state, no surprising behavior from inherited environments.

The Vulkan backend respects standard Vulkan environment variables (`VK_ICD_FILENAMES`, `VK_LAYER_PATH`, etc.) for driver and validation layer configuration.
