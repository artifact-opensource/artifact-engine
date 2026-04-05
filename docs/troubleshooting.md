# Troubleshooting

Common issues and solutions when building, running, and deploying Artifact Engine.

## Build Issues

### "vulkan/vulkan.h: No such file or directory"

The Vulkan SDK is not installed.

**Fix:** Either install the Vulkan SDK or build without it:

```bash
# Install Vulkan SDK (Ubuntu/Debian)
sudo apt install vulkan-sdk

# Or build CPU-only (no Vulkan needed)
cmake .. -DCPU_ONLY=ON
```

### "glslangValidator: command not found"

The shader compiler is not installed. It's included in the Vulkan SDK.

**Fix:**
```bash
# Ubuntu/Debian
sudo apt install glslang-tools

# Or install the full Vulkan SDK
```

### CMake version too old

Artifact Engine requires CMake 3.16+.

**Fix:**
```bash
# Install latest CMake
pip install cmake

# Or download from cmake.org
```

### Linker errors on Windows (unresolved externals)

Usually caused by 32-bit vs 64-bit mismatch.

**Fix:** Ensure you're building for x64:
```cmd
cmake .. -G "Visual Studio 17 2022" -A x64
```

### "Cannot find -lvulkan"

The Vulkan loader library is not installed.

**Fix:**
```bash
# Ubuntu/Debian
sudo apt install libvulkan-dev

# Fedora
sudo dnf install vulkan-loader-devel
```

## Runtime Issues

### "Failed to create Vulkan instance"

No Vulkan-capable GPU driver is installed, or the Vulkan runtime is missing.

**Diagnosis:**
```bash
# Check Vulkan availability
vulkaninfo --summary

# If vulkaninfo is not found:
sudo apt install vulkan-tools
vulkaninfo --summary
```

**Fix:**
- Install or update your GPU driver (NVIDIA, AMD, or Intel)
- Ensure the Vulkan ICD (installable client driver) is present
- As a workaround, use the CPU backend: `--backend cpu`

### "No suitable Vulkan device found"

Vulkan is available but no GPU meets the requirements (needs compute queue support).

**Diagnosis:**
```bash
# List all Vulkan devices
./artifact-engine --list-devices
```

**Fix:**
- Update your GPU driver
- If using integrated graphics, ensure Vulkan support is enabled in BIOS
- Use `--gpu-device N` to try a different device
- Fall back to CPU: `--backend cpu`

### "Out of GPU memory" or "VK_ERROR_OUT_OF_DEVICE_MEMORY"

The model + KV cache exceeds available VRAM.

**Diagnosis:** Check model size vs available VRAM:
- Q4_K_M 3B ≈ 2.0 GB
- Q4_K_M 7B ≈ 4.5 GB
- Q4_K_M 9B ≈ 5.5 GB
- KV cache adds ~128 MB per 1K context tokens (for 7-9B models)

**Fix:**
- Use a smaller model
- Reduce context length: `--ctx-len 512` or `--ctx-len 1024`
- Close other GPU-intensive applications
- Use the CPU backend for models that don't fit in VRAM: `--backend cpu`

### "Not a GGUF file" or "GGUF magic mismatch"

The model file is not in GGUF format, is corrupted, or is a partial download.

**Fix:**
- Verify the file with `xxd model.gguf | head -1` — should start with `4755 4646` (GGUF magic)
- Re-download the model if the file is truncated
- Ensure you downloaded the `.gguf` file, not an HTML page (common with incorrect URLs)

### "Unsupported GGUF version N"

The model uses a GGUF version that Artifact Engine doesn't support.

**Fix:** Artifact Engine supports GGUF v2 and v3. If the model uses a newer version, check for an updated version of Artifact Engine or re-quantize the model with a compatible tool.

### "Missing key: {architecture}.embedding_length"

The GGUF file doesn't contain expected metadata. This usually means the model uses an unsupported architecture.

**Fix:**
- Check `general.architecture` in the model metadata — supported: `llama`, `qwen2`, `mistral`
- Use a model from a supported family

### HTTP API returns no response or hangs

The server is busy generating tokens. With large models or long prompts, the first token can take several seconds.

**Diagnosis:**
- Check stderr output — the server logs when it starts generating
- Test with a minimal prompt: `{"messages":[{"role":"user","content":"Hi"}],"max_tokens":1}`

**Fix:**
- Use `--verbose` to see detailed timing
- Reduce prompt length
- Use a smaller model
- Check if the GPU is thermal throttling

### "Address already in use" (EADDRINUSE)

Another process is using the specified port.

**Fix:**
```bash
# Find what's using the port
lsof -i :8080

# Use a different port
./artifact-engine --model model.gguf --port 9090
```

## Model Issues

### Model generates garbage or repetitive text

**Possible causes:**
1. Wrong chat template applied (e.g., Chatml template on a Llama model)
2. Temperature too high
3. Model is too small for the task
4. Corrupt model file

**Fix:**
- Try temperature 0.0 (greedy) to rule out sampling issues
- Verify the model architecture matches what Artifact Engine expects
- Re-download the model
- Try a different model

### Inference is very slow

**Possible causes:**
1. Using CPU backend instead of Vulkan
2. Using integrated GPU instead of discrete GPU
3. Model too large for available VRAM (thrashing between GPU and system memory)
4. Context length too long

**Diagnosis:**
```bash
# Run with verbose output
./artifact-engine --model model.gguf --verbose

# Check which device is being used
./artifact-engine --list-devices
```

**Fix:**
- Ensure the Vulkan backend is active: `--backend vulkan`
- Select the correct GPU: `--gpu-device N`
- Use a smaller quantization (Q4_K_M instead of Q5_K_M)
- Reduce context length

### Token count seems wrong

The tokenizer produces different token counts than Python tokenizers for the same text.

**Explanation:** Artifact Engine's BPE tokenizer loads merge rules directly from the GGUF file. Minor differences from HuggingFace tokenizers can occur due to:
- Pre-tokenization regex differences
- Unicode normalization handling
- Added token matching order

These differences are typically 1-3% and do not affect generation quality.

## Network Issues

### LAN discovery doesn't find other instances

**Diagnosis:**
```bash
# Check if UDP broadcast works
# On the server machine:
nc -lu 8081

# On the client machine:
echo "AE_DISCOVER" | nc -u -b 255.255.255.255 8081
```

**Fix:**
- Ensure both machines are on the same subnet
- Check firewall rules — UDP port 8081 must be open
- Some networks block broadcast traffic (common in enterprise/hotel WiFi)

### Model download fails or is slow

**Fix:**
- Check your internet connection
- Try a different model URL
- If behind a proxy, the built-in HTTP client doesn't support proxy configuration — download the model externally and use `--model /path/to/model.gguf`

## Getting Help

If your issue isn't covered here:

1. Run with `--verbose` and capture the full output
2. Note your OS, GPU model, driver version, and model file
3. Check the GGUF file integrity: `sha256sum model.gguf`
4. File an issue with the diagnostic information
