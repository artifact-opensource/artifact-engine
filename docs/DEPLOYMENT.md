# Artifact Engine — Xbox Series X Deployment Guide

> Version 0.1.0 · CONFIDENTIAL

## Overview

The Xbox Series X has an AMD RDNA 2 GPU with ~10 GB usable VRAM and 16 GB unified RAM — enough to run quantized LLMs up to 9B parameters (Q4_K_M). Artifact Engine is built on Vulkan, which is supported on Xbox via the Game Development Kit (GDK).

This guide covers setting up the Xbox in Developer Mode, deploying Artifact Engine, loading a model, and running it as a local inference server that all devices on your LAN can connect to.

---

## Target Configuration

| Spec | Xbox Series X |
|------|--------------|
| GPU | AMD RDNA 2, 52 CUs, 12.15 TFLOPS |
| VRAM | ~10 GB usable (16 GB unified, shared with system) |
| CPU | AMD Zen 2, 8 cores, 3.8 GHz |
| RAM | 16 GB GDDR6 (unified) |
| Storage | 1 TB NVMe |
| Vulkan | Supported via GDK (Vulkan 1.2 compute) |
| Network | Gigabit Ethernet |

**Target Model:** Qwen 3.5 9B Q4_K_M (6.6 GB GGUF file, ~7 GB VRAM loaded)

---

## Step 1: Enable Developer Mode

1. On your Xbox, go to **Store** and search for **"Xbox Dev Mode"** (free app)
2. Install and launch the Dev Mode app
3. Follow the activation instructions — you'll need a Microsoft Partner Center developer account ($19 one-time fee) at https://partner.microsoft.com/
4. The Xbox will restart into **Developer Mode** — a separate environment from retail mode
5. Note the Xbox's **IP address** shown on the Dev Mode home screen (e.g., `192.168.1.50`)
6. Access **Xbox Device Portal** from a browser: `https://192.168.1.50:11443`

### Developer Mode Notes
- Dev Mode and Retail Mode are separate partitions — you can switch between them
- In Dev Mode, the full GPU and system resources are available
- Network access is unrestricted — the Xbox can serve HTTP to the LAN

---

## Step 2: Build Artifact Engine for Xbox

The Xbox runs a Windows 10-based OS. Build a Windows x64 binary:

### Option A: Cross-compile from Linux (MinGW)

```bash
cd /path/to/artifact-engine

# Compile shaders
mkdir -p build/shaders
for s in matmul rmsnorm rope softmax silu add mul embedding dequant_q4k; do
    glslc -O shaders/${s}.comp -o build/shaders/${s}.spv
done

# Cross-compile
x86_64-w64-mingw32-gcc -O2 -std=c11 -D_WIN32 -I include \
    src/main.c src/gguf.c src/vulkan_compute.c src/engine.c src/http_server.c \
    -o build/artifact-engine.exe \
    -lvulkan-1 -lws2_32 -lm
```

### Option B: Build on Windows (MSVC/GDK)

For best Xbox compatibility, build with the GDK toolchain which includes the Vulkan headers and libs targeting the Xbox runtime:

```cmd
:: Using Developer Command Prompt for VS 2022
cl /O2 /std:c11 /I include /I "%GDK_PATH%\Include" ^
    src\main.c src\gguf.c src\vulkan_compute.c src\engine.c src\http_server.c ^
    /Fe:artifact-engine.exe ^
    /link /LIBPATH:"%GDK_PATH%\Lib" vulkan-1.lib ws2_32.lib
```

---

## Step 3: Download the Model

Download a quantized GGUF model. Recommended: **Qwen 3.5 9B Q4_K_M** (~6.6 GB).

```bash
# On a PC with fast internet
# From HuggingFace (example URL — find the actual GGUF release)
wget https://huggingface.co/Qwen/Qwen3.5-9B-GGUF/resolve/main/qwen3.5-9b-q4_k_m.gguf
```

Alternative models that fit in 10 GB VRAM:
- Qwen 3.5 4B Q8_0 (~4.5 GB)
- LLaMA 3 8B Q4_K_M (~5.0 GB)
- Mistral 7B v0.3 Q4_K_M (~4.4 GB)
- Phi-3 Mini 4B Q8_0 (~4.2 GB)

---

## Step 4: Deploy to Xbox

### Via Device Portal (recommended for first deployment)

1. Open Xbox Device Portal: `https://<xbox-ip>:11443`
2. Navigate to **File Explorer** → Browse to a deployment directory (e.g., `D:\ArtifactEngine\`)
3. Upload:
   - `artifact-engine.exe`
   - `shaders/` directory (all `.spv` files)
   - Your GGUF model file (e.g., `qwen3.5-9b-q4_k_m.gguf`)

### Via Network Share (faster for large files)

1. In Device Portal, go to **Settings** → **Developer Settings** → Enable **Device Discovery**
2. The Xbox exposes SMB shares — map `\\<xbox-ip>\DevelopmentFiles` from your PC
3. Copy the files directly via the network share

### File Layout on Xbox

```
D:\ArtifactEngine\
├── artifact-engine.exe
├── shaders\
│   ├── matmul.spv
│   ├── rmsnorm.spv
│   ├── rope.spv
│   ├── softmax.spv
│   ├── silu.spv
│   ├── add.spv
│   ├── mul.spv
│   ├── embedding.spv
│   └── dequant_q4k.spv
└── models\
    └── qwen3.5-9b-q4_k_m.gguf
```

---

## Step 5: Run Artifact Engine

### Via Device Portal

1. Go to **Run a command** (or SSH into the Xbox)
2. Execute:

```cmd
cd D:\ArtifactEngine
artifact-engine.exe --model models\qwen3.5-9b-q4_k_m.gguf --port 8080 --host 0.0.0.0 --ctx 4096 --shaders shaders
```

### Expected Output

```
╔══════════════════════════════════════╗
║       ARTIFACT ENGINE v0.1.0         ║
║       Artifact Virtual               ║
╚══════════════════════════════════════╝

[1/4] Initializing Vulkan...
vk: GPU: Microsoft Xbox Series X (RDNA 2)
vk: VRAM: 10.00 GB
vk: Max workgroup: 1024, Max buffer: 2.00 GB
vk: initialized (15 shaders loaded)

[2/4] Loading model: models\qwen3.5-9b-q4_k_m.gguf
gguf: version=3, tensors=291, kv_pairs=26
gguf: data section at offset 3424, size 6.63 GB
engine: uploading 36 layers to GPU...
  layer 4/36 — VRAM: 0.89 GB / 10.00 GB
  layer 8/36 — VRAM: 1.68 GB / 10.00 GB
  ...
  layer 36/36 — VRAM: 6.48 GB / 10.00 GB
engine: model loaded — 6.48 GB VRAM used

[3/4] Allocating KV cache (ctx=4096)...
engine: KV cache allocated (ctx=4096) — total VRAM: 7.12 GB / 10.00 GB

[4/4] Starting HTTP server on 0.0.0.0:8080...
╔══════════════════════════════════════╗
║  Artifact Engine — Listening         ║
║  http://0.0.0.0:8080                 ║
╚══════════════════════════════════════╝
```

---

## Step 6: Connect from LAN Devices

Once the engine is running on Xbox, any device on the same network can connect:

### From Spore (AEGIS / Android)

Configure Spore to use the Xbox as its LLM backend:

```json
{
  "provider": "openai",
  "model": "qwen2",
  "base_url": "http://192.168.1.50:8080/v1"
}
```

### From Any OpenAI-Compatible Client

```bash
# From any machine on the LAN
curl http://192.168.1.50:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello from the Xbox!"}],
    "temperature": 0.7,
    "stream": true
  }'
```

### From Python

```python
from openai import OpenAI

client = OpenAI(base_url="http://192.168.1.50:8080/v1", api_key="none")
response = client.chat.completions.create(
    model="qwen2",
    messages=[{"role": "user", "content": "What can you do?"}]
)
print(response.choices[0].message.content)
```

---

## Step 7: Run as Background Service

To keep Artifact Engine running when you're not at the Device Portal:

### Via Scheduled Task

Create a startup task in Device Portal → **Xbox Settings** → **App Management** → **Run at Startup**.

### Via Script

Create `D:\ArtifactEngine\start.bat`:

```bat
@echo off
cd /d D:\ArtifactEngine
artifact-engine.exe --model models\qwen3.5-9b-q4_k_m.gguf --port 8080 --host 0.0.0.0 --ctx 4096 --shaders shaders > engine.log 2>&1
```

---

## Network Topology: Artifact Cloud

```
┌────────────────────────────────────────────────────────────┐
│                    Local Network (LAN)                      │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Dragonfly    │    │   Victus     │    │    AEGIS     │  │
│  │  (Mach6/AVA)  │    │  (GPU forge) │    │  (Z Fold 5)  │  │
│  │  .13          │    │  .8          │    │  .3/.7       │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                    │                    │          │
│         └────────────┬───────┴────────────┬──────┘          │
│                      │                    │                  │
│              ┌───────▼────────────────────▼───────┐         │
│              │         Xbox Series X              │         │
│              │     Artifact Engine :8080           │         │
│              │     Qwen 3.5 9B Q4_K_M             │         │
│              │     RDNA 2, 10 GB VRAM             │         │
│              │         .50 (example)              │         │
│              └───────────────┬────────────────────┘         │
│                              │                               │
│                  ┌───────────▼───────────┐                  │
│                  │  Cloudflare Tunnel    │                  │
│                  │  (internet-facing)    │                  │
│                  └──────────────────────┘                   │
└────────────────────────────────────────────────────────────┘
```

All LAN devices connect to `http://<xbox-ip>:8080`. For internet access, a Cloudflare Tunnel can expose the endpoint securely without opening router ports.

---

## Troubleshooting

### "vk: no Vulkan-capable GPU found"

- Ensure you're running in Dev Mode (not Retail Mode)
- The GDK's Vulkan runtime must be present — it should be included in the Xbox system image for Dev Mode
- Try running `vulkaninfo` equivalent if available in the Xbox dev tools

### Out of VRAM

- Reduce context length: `--ctx 2048` instead of `--ctx 4096`
- Use a more aggressively quantized model (Q4_0 instead of Q4_K_M)
- Use a smaller model (4B instead of 9B)

### Slow Inference

- RDNA 2 compute throughput depends on shader occupancy — the current matmul shader uses 16×16 workgroups; 32×32 may be better for RDNA 2
- Ensure no other apps are running in Dev Mode consuming GPU resources
- Check GPU temperature via Device Portal system stats

### Network Connection Refused

- Verify firewall settings in Dev Mode — port 8080 must be open
- Check that `--host 0.0.0.0` is set (not `127.0.0.1`)
- Verify Xbox IP via Device Portal home screen

### Model Too Large

Qwen 3.5 9B Q4_K_M fits in 10 GB. If using a larger model:
```
Model size + KV cache + scratch must be < ~10 GB
```

For a 9B Q4_K_M model at ctx=4096:
- Model weights: ~6.6 GB
- KV cache: ~0.5 GB (36 layers × 2 × 4096 × 512 × 4 bytes)
- Scratch: ~0.1 GB
- **Total: ~7.2 GB** — fits with ~2.8 GB headroom

---

## Security Considerations

- The HTTP API has **no authentication** — it should only be exposed on trusted networks
- Do not expose port 8080 directly to the internet without a reverse proxy + auth
- Use Cloudflare Tunnel with access policies for internet-facing deployments
- The Xbox in Dev Mode has reduced security sandboxing — treat it as a development device
