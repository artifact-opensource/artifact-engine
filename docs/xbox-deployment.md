# Xbox Deployment

Complete guide to running Artifact Engine on Xbox Series X|S.

## Overview

Artifact Engine runs on Xbox as a UWP (Universal Windows Platform) application using the DirectML compute backend. The Xbox Series X has an RDNA 2 GPU with 10-12 TFLOPS of compute — more than enough for real-time LLM inference on quantized models.

## Prerequisites

**Hardware:**
- Xbox Series X or Series S
- Network connection (for deployment and model transfer)

**Xbox Configuration:**
- Developer Mode enabled ([Microsoft guide](https://learn.microsoft.com/en-us/windows/uwp/xbox-apps/devkit-activation))
- Xbox Device Portal accessible at `https://XBOX_IP:11443`

**Build Machine (Windows):**
- Windows 10/11
- Visual Studio 2022 with:
  - Desktop development with C++ workload
  - Universal Windows Platform development workload
- Xbox GDK (Game Development Kit)
- MakeAppx.exe (included in Windows SDK)

## Build

### Step 1: Build the Binary

```cmd
build_xbox.bat
```

This script:
1. Configures CMake for UWP/Xbox targeting
2. Compiles with the DirectML backend enabled
3. Links against DirectX 12 and DirectML libraries
4. Outputs `artifact-engine.exe` for Xbox

### Step 2: Package as .appx

```cmd
package_xbox.bat
```

This script:
1. Collects the binary and assets into a staging directory
2. Includes `AppxManifest.xml` with the app identity and capabilities
3. Runs `MakeAppx.exe pack` to create `ArtifactEngine.appx`

### Step 3: Deploy to Xbox

```cmd
curl -sk -u USERNAME:PASSWORD -X POST ^
  -F "file=@ArtifactEngine.appx" ^
  https://XBOX_IP:11443/api/app/packagemanager/package
```

Replace:
- `USERNAME:PASSWORD` with your Xbox Device Portal credentials
- `XBOX_IP` with your Xbox's IP address

Alternatively, deploy through the Xbox Device Portal web UI:
1. Open `https://XBOX_IP:11443` in a browser
2. Navigate to **My games & apps** → **Install app**
3. Upload `ArtifactEngine.appx`
4. Click **Install**

## Model Deployment

The Xbox UWP sandbox restricts file system access. Models must be placed in accessible locations.

### Method 1: DevelopmentFiles (Recommended)

DevelopmentFiles is a shared folder accessible via the Xbox Device Portal.

1. Open the Xbox Device Portal: `https://XBOX_IP:11443`
2. Navigate to **File Explorer** → **DevelopmentFiles**
3. Create a `models` subfolder
4. Upload your `.gguf` file

Artifact Engine scans `DevelopmentFiles\` and `DevelopmentFiles\models\` automatically.

### Method 2: LAN Pull

If another machine on the network is running Artifact Engine with a model:

1. Start Artifact Engine on the PC:
   ```cmd
   artifact-engine.exe --model models\qwen-3.5-9b-q4_k_m.gguf --port 8080
   ```
2. Artifact Engine on Xbox will discover it via UDP broadcast and pull the model automatically

### Method 3: SMB Share

1. Share a folder containing the model on your PC
2. Ensure the Xbox can reach the PC on the local network
3. Artifact Engine scans network shares for GGUF files

## AppxManifest.xml

The manifest defines the app identity, capabilities, and visual assets:

```xml
<?xml version="1.0" encoding="utf-8"?>
<Package xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"
         xmlns:mp="http://schemas.microsoft.com/appx/2014/phone/manifest"
         xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10">

  <Identity Name="ArtifactVirtual.ArtifactEngine"
            Publisher="CN=Artifact Virtual"
            Version="0.5.0.0"
            ProcessorArchitecture="x64" />

  <Properties>
    <DisplayName>Artifact Engine</DisplayName>
    <PublisherDisplayName>Artifact Virtual</PublisherDisplayName>
    <Description>LLM inference engine — Vulkan compute, pure C</Description>
    <Logo>Assets\StoreLogo.png</Logo>
  </Properties>

  <Resources>
    <Resource Language="en-us"/>
  </Resources>

  <Dependencies>
    <TargetDeviceFamily Name="Windows.Xbox"
                        MinVersion="10.0.19041.0"
                        MaxVersionTested="10.0.22621.0" />
  </Dependencies>

  <Applications>
    <Application Id="App"
                 Executable="artifact-engine.exe"
                 EntryPoint="ArtifactEngine.App">
      <uap:VisualElements
        DisplayName="Artifact Engine"
        Square150x150Logo="Assets\Square150x150Logo.png"
        Square44x44Logo="Assets\Square44x44Logo.png"
        Description="LLM Inference Engine"
        BackgroundColor="transparent">
      </uap:VisualElements>
    </Application>
  </Applications>

  <Capabilities>
    <Capability Name="internetClient" />
    <Capability Name="internetClientServer" />
    <Capability Name="privateNetworkClientServer" />
  </Capabilities>

</Package>
```

Key capabilities:
- **internetClient** — for HTTP model fetching from external URLs
- **internetClientServer** — for serving the HTTP API
- **privateNetworkClientServer** — for LAN model discovery and SMB access

## Network Configuration

### HTTP API Access

Once running, the Artifact Engine HTTP API is accessible from other devices on the local network:

```bash
# From your PC
curl http://XBOX_IP:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello from Xbox!"}]}'
```

### Port Forwarding

Xbox Developer Mode allows inbound connections by default. If the API is not reachable:

1. Open Xbox Device Portal → **Networking**
2. Verify the Xbox IP address
3. Check that port 8080 is not blocked by your router

## Memory Considerations

Xbox Series X has 16GB total memory shared between system, game, and apps. UWP background apps are limited to approximately 1GB of memory.

**Implications:**
- Use quantized models (Q4_K_M or lower)
- Keep context length reasonable (`--ctx-len 1024` or `--ctx-len 2048`)
- A 3B Q4_K_M model (~2GB weights + ~128MB KV cache) fits comfortably
- A 9B Q4_K_M model (~5.5GB) requires running as a foreground app or with elevated memory allocation

**Recommended models for Xbox:**

| Model | Quant | Total Memory | Fits In |
|-------|-------|-------------|---------|
| Qwen 2.5 3B | Q4_K_M | ~2.2 GB | Background app |
| Phi-3 Mini 3.8B | Q4_K_M | ~2.5 GB | Background app |
| Llama 3.1 8B | Q4_K_M | ~5.5 GB | Foreground only |
| Qwen 3.5 9B | Q4_K_M | ~6.0 GB | Foreground only |

## Game Companion Architecture

The DirectML backend supports frame capture for a "game companion" use case:

```
┌──────────────────────────────────────┐
│            Xbox Series X             │
│                                      │
│  ┌──────────┐    ┌────────────────┐  │
│  │   Game   │    │ Artifact Engine│  │
│  │ (D3D12)  │    │  (DirectML)   │  │
│  │          │───►│               │  │
│  │          │frame│  LLM + Vision │  │
│  │          │capture│             │  │
│  └──────────┘    └───────┬────────┘  │
│                          │           │
│                     HTTP API         │
│                          │           │
└──────────────────────────┼───────────┘
                           │
                    ┌──────▼──────┐
                    │ Companion   │
                    │ App (phone/ │
                    │ tablet/PC)  │
                    └─────────────┘
```

The companion app sends requests to the Xbox's HTTP API, which can include visual context from the running game. This enables scenarios like:
- Real-time game strategy suggestions
- NPC dialogue generation based on game state
- Accessibility narration of on-screen content

## Troubleshooting

**"App won't install"**
- Ensure Developer Mode is enabled on Xbox
- Check that the .appx was built for x64 architecture
- Verify the certificate matches (sideloading may require self-signed cert trust)

**"Model not found"**
- Check DevelopmentFiles via the Device Portal file browser
- Ensure the .gguf file uploaded completely (check file size)
- Try specifying the full path: `--model D:\DevelopmentFiles\models\model.gguf`

**"Out of memory"**
- Use a smaller model (3B recommended for background execution)
- Reduce context length: `--ctx-len 512`
- Close other apps on the Xbox

**"HTTP API unreachable"**
- Verify Xbox IP address in Device Portal → Networking
- Check that the port is not in use by another app
- Try a different port: `--port 9090`

**"DirectML initialization failed"**
- Ensure the Xbox OS is up to date
- Check that the D3D12 runtime is available (should be by default on Xbox)
