# Model Fetching

Artifact Engine includes a built-in model acquisition system. Models can be downloaded from HTTP URLs, discovered on the local network, or found on Xbox-specific paths. No external tools (wget, huggingface-cli, etc.) are required.

**File:** `src/model_fetch.c` (534 lines)
**Header:** `include/model_fetch.h`

## HTTP Download

Download a GGUF model from any URL:

```bash
./artifact-engine --fetch https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf --port 8080
```

### Behavior

1. Parses the URL and extracts the filename
2. Creates a TCP connection to the host (port 443 for HTTPS, 80 for HTTP)
3. Sends an HTTP GET request with standard headers
4. Reads the response, following redirects (up to 5 hops)
5. Writes the file to `models/{filename}` with progress reporting
6. Validates the file starts with the GGUF magic number (`0x46475547`)
7. On success, loads the model and starts serving

### Progress Reporting

During download, progress is printed to stderr:

```
Downloading: qwen2.5-3b-instruct-q4_k_m.gguf
  [################............] 58% (1.2 GB / 2.1 GB) 45.3 MB/s
```

### Resume Support

If a partial download exists (e.g., from an interrupted fetch), the fetcher sends a `Range` header to resume from where it left off. The server must support range requests (most CDNs and Hugging Face do).

### Error Handling

- **DNS failure** — "Failed to resolve host: huggingface.co" (fatal)
- **Connection refused** — "Connection to host:port failed" (fatal)
- **HTTP 404** — "Model not found at URL" (fatal)
- **HTTP 5xx** — Retry up to 3 times with exponential backoff
- **Corrupt download** — GGUF magic validation fails → delete partial file, report error
- **Disk full** — Write fails → report error with available space

## LAN Auto-Discovery

Artifact Engine instances on the same local network can share models automatically.

### Protocol

The LAN discovery protocol uses UDP broadcast:

```
┌─────────┐                        ┌─────────┐
│ Client  │  UDP broadcast:8081    │ Server  │
│ (needs  │ ─────────────────────► │ (has    │
│  model) │  "AE_DISCOVER"        │  model) │
│         │                        │         │
│         │  UDP unicast response  │         │
│         │ ◄───────────────────── │         │
│         │  "AE_OFFER model_name  │         │
│         │   size_bytes port"     │         │
│         │                        │         │
│         │  TCP connect to port   │         │
│         │ ─────────────────────► │         │
│         │  HTTP GET /model       │         │
│         │ ◄───────────────────── │         │
│         │  (raw GGUF data)       │         │
└─────────┘                        └─────────┘
```

### Usage

**On the machine that has the model (server):**
```bash
# Just run normally — LAN discovery server is always active
./artifact-engine --model models/qwen-3.5-9b-q4_k_m.gguf --port 8080
```

**On the machine that needs the model (client):**
```bash
# Auto-discover and pull
./artifact-engine --lan-pull --model auto --port 8080
```

### Discovery Packet Format

**Request (broadcast):**
```
AE_DISCOVER\n
```

**Response (unicast):**
```
AE_OFFER {model_name} {size_bytes} {tcp_port}\n
```

Example:
```
AE_OFFER qwen-3.5-9b-q4_k_m.gguf 5368709120 8080
```

### Network Requirements

- Client and server must be on the same broadcast domain (same subnet)
- UDP port 8081 must not be blocked by firewall
- TCP port (server's HTTP port) must be accessible for the file transfer

## Xbox Model Paths

On Xbox (UWP environment), the model fetcher scans platform-specific paths to find GGUF files.

### Search Order

1. **DevelopmentFiles** — Xbox Developer Mode shared folder, accessible via the Xbox Device Portal file manager
   ```
   D:\DevelopmentFiles\*.gguf
   D:\DevelopmentFiles\models\*.gguf
   ```

2. **Local app data** — the UWP app's private storage
   ```
   LocalState\*.gguf
   LocalState\models\*.gguf
   ```

3. **SMB shares** — network shares discovered via NetBIOS/SMB browsing
   ```
   \\{host}\{share}\*.gguf
   ```

### Deploying Models to Xbox

**Method 1: Xbox Device Portal (recommended)**

1. Open the Xbox Device Portal in a browser: `https://XBOX_IP:11443`
2. Navigate to File Explorer → DevelopmentFiles
3. Upload the `.gguf` file
4. Artifact Engine will find it automatically on next start

**Method 2: SMB from a LAN host**

1. Share a folder containing the model on your PC
2. Artifact Engine scans SMB shares on the local network
3. If a GGUF file is found, it copies it to local storage

**Method 3: LAN pull from another Artifact Engine instance**

1. Run Artifact Engine with the model on your PC
2. Start Artifact Engine on Xbox with `--lan-pull`
3. The Xbox instance discovers the PC instance and pulls the model

## Model Directory

All downloaded models are stored in the `models/` directory relative to the executable (or the current working directory). The directory is created automatically if it doesn't exist.

```
models/
├── qwen-3.5-9b-q4_k_m.gguf     (5.4 GB)
├── llama-3.1-8b-q4_k_m.gguf    (4.9 GB)
└── mistral-7b-q4_k_m.gguf      (4.4 GB)
```

## Where to Get Models

GGUF models are available from:

- **Hugging Face** — search for "GGUF" on huggingface.co. Most popular models have community-quantized GGUF versions.
- **TheBloke** — prolific quantizer with hundreds of GGUF models: `https://huggingface.co/TheBloke`
- **Official repos** — Qwen, Meta, Mistral publish official GGUF versions of their models.

Recommended quantization: **Q4_K_M** for the best quality/size balance on 4-8GB VRAM.

Direct download URLs (Hugging Face):
```
https://huggingface.co/{org}/{model}/resolve/main/{filename}.gguf
```

Example:
```bash
./artifact-engine --fetch https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
```
