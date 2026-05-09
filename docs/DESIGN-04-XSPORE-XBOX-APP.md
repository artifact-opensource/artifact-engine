# DESIGN-04: XSpore — Xbox UWP Game Mode Application
## Symbiote XSpore — Chat + Dashboard + Game Launcher

**Author:** AVA (Artifact Virtual)  
**Date:** April 6, 2026  
**Version:** 1.0  
**Status:** Architecture Blueprint  
**Dependencies:** DESIGN-01 (GS Renderer), DESIGN-03 (XSpore Pipeline), Artifact Engine

---

## 1. Vision

Click the XSpore icon on Xbox Home → full-screen game-mode app launches → you're talking to Symbiote. Chat on the left, 3D Gaussian viewport on the right, dashboard below. Say "show me a galaxy" and it renders. Say "launch Halo" and it launches. Say "what's my Cthulu P&L" and it shows the trading dashboard.

**This is Symbiote's living room.** The Xbox becomes the primary consumer interface for the entire Artifact Virtual ecosystem — LLM chat, 3D generation, trading dashboards, game launching, system monitoring — all from the couch with a controller.

### Why Xbox Game Mode
- **5GB RAM** (vs 1GB in app mode) — fits LLM + renderer + UI
- **Full GPU access** — RDNA 2, 12 TFLOPs, DirectX 12 Ultimate
- **4+2 CPU cores** — dedicated compute for inference
- **DX12** — hardware-accelerated rendering pipeline
- **Controller-native** — designed for 10-foot UI from day one

### Name
**XSpore** = a spore that grows worlds from seeds. The Xbox is the terrarium.  
**Symbiote XSpore** = the full branded experience on Xbox Home.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SYMBIOTE XSPORE (UWP Game)               │
│                                                              │
│  ┌──────────────┐  ┌───────────────────┐  ┌──────────────┐ │
│  │   CHAT        │  │   3D VIEWPORT     │  │  DASHBOARD   │ │
│  │   Panel       │  │   Gaussian Splat  │  │  Panel       │ │
│  │              │  │   Renderer        │  │              │ │
│  │  Controller  │  │   (DESIGN-01)     │  │  Cthulu P&L  │ │
│  │  keyboard    │  │                   │  │  System Mon   │ │
│  │  + voice     │  │   Camera orbit    │  │  Game Library │ │
│  │              │  │   via right stick │  │              │ │
│  └──────┬───────┘  └────────┬──────────┘  └──────┬───────┘ │
│         │                   │                     │         │
│  ┌──────▼───────────────────▼─────────────────────▼───────┐ │
│  │              XSPORE CORE ENGINE (C)                     │ │
│  │                                                         │ │
│  │  ┌─────────┐ ┌──────────┐ ┌─────────┐ ┌─────────────┐ │ │
│  │  │ UI Mgr  │ │ Scene Mgr│ │ Net Mgr │ │ App Launcher│ │ │
│  │  │ (D3D12) │ │ (GS Pipe)│ │ (HTTP)  │ │ (UWP API)   │ │ │
│  │  └─────────┘ └──────────┘ └─────────┘ └─────────────┘ │ │
│  └─────────────────────────────────────────────────────────┘ │
│                              │                               │
│                     ┌────────▼────────┐                      │
│                     │    D3D12 GPU    │                      │
│                     │  RDNA 2 (Xbox)  │                      │
│                     └─────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
    ┌────▼────┐         ┌────▼────┐          ┌────▼────┐
    │ Victus  │         │Dragonfly│          │  Xbox   │
    │ LLM     │         │ Mach6   │          │ Retail  │
    │ Qwen 3.5│         │ Cthulu  │          │ Games   │
    └─────────┘         └─────────┘          └─────────┘
```

### 2.1 Component Breakdown

| Component | Role | Technology |
|-----------|------|------------|
| **UI Manager** | Immediate-mode GPU-rendered interface | D3D12 + nuklear (single-header C) |
| **Scene Manager** | Gaussian splat rendering pipeline | Compute shaders (DESIGN-01) |
| **Network Manager** | LAN communication with Victus/Dragonfly | HTTP client (WinHTTP) |
| **App Launcher** | Launch Xbox games from within XSpore | UWP `LaunchUriAsync` / protocol activation |
| **Input Manager** | Controller mapping + virtual keyboard | XInput / GamePad API |
| **Chat Engine** | Message history, streaming responses | Ring buffer + text layout |

### 2.2 Data Flow

```
User Input (Controller/Voice/Keyboard)
    │
    ├─→ Text Chat → HTTP POST → Victus (Qwen 3.5) or Dragonfly (Mach6)
    │                              │
    │                              ▼
    │                         LLM Response (streamed)
    │                              │
    │                    ┌─────────┼─────────┐
    │                    │         │         │
    │                    ▼         ▼         ▼
    │              Plain text   XSpore    Command
    │              → Chat       JSON      → Execute
    │              panel        → Render  (launch/query)
    │
    ├─→ Camera Control → Right stick → Orbit/zoom viewport
    ├─→ Dashboard Nav → D-pad → Switch dashboard panels
    └─→ Game Launch → A button on game tile → Protocol URI
```

---

## 3. User Interface — 10-Foot Design

### 3.1 Layout Modes

**Mode 1: CHAT (default on launch)**
```
┌──────────────────────────────────────────────────┐
│  SYMBIOTE XSPORE                    🔮  ⚙️  ❌  │
├──────────────────────────────────────────────────┤
│                                                   │
│   ┌─────────────────────────────────────────┐    │
│   │                                         │    │
│   │          CHAT HISTORY                   │    │
│   │                                         │    │
│   │  [AVA] Welcome back. What shall we      │    │
│   │        build today?                     │    │
│   │                                         │    │
│   │  [You] Show me a crystal cave           │    │
│   │                                         │    │
│   │  [AVA] Generating... ████████░░ 80%     │    │
│   │                                         │    │
│   └─────────────────────────────────────────┘    │
│                                                   │
│   ┌─────────────────────────────────────────┐    │
│   │ > Type or press Y for voice...    [Send]│    │
│   └─────────────────────────────────────────┘    │
│                                                   │
│   [LB] Chat  [RB] Viewport  [Menu] Dashboard     │
├──────────────────────────────────────────────────┤
│  🟢 Victus: Online  │  🟢 Dragonfly: Online      │
└──────────────────────────────────────────────────┘
```

**Mode 2: VIEWPORT (RB from chat, or auto on scene generation)**
```
┌──────────────────────────────────────────────────┐
│  SYMBIOTE XSPORE              Crystal Cave  🔮   │
├─────────────┬────────────────────────────────────┤
│             │                                     │
│  CHAT       │     3D GAUSSIAN VIEWPORT            │
│  (narrow)   │                                     │
│             │     ┌───────────────────────┐       │
│  [AVA]      │     │                       │       │
│  Crystal    │     │    *** * ** *          │       │
│  cave with  │     │   ** CAVE ***         │       │
│  3 chambers │     │    *** * ** *          │       │
│  and        │     │   crystals glow       │       │
│  glowing    │     │    amethyst purple    │       │
│  amethyst   │     │                       │       │
│  forma-     │     └───────────────────────┘       │
│  tions...   │                                     │
│             │  Right Stick: orbit  LT/RT: zoom    │
│             │  X: wireframe  Y: screenshot        │
│  [>_]       │                                     │
├─────────────┴────────────────────────────────────┤
│  Gaussians: 45,231  │  FPS: 60  │  GPU: 34%      │
└──────────────────────────────────────────────────┘
```

**Mode 3: DASHBOARD (Menu button)**
```
┌──────────────────────────────────────────────────┐
│  SYMBIOTE XSPORE — DASHBOARD               🔮   │
├──────────────────────────────────────────────────┤
│                                                   │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐ │
│  │ CTHULU │  │ SYSTEM │  │ GAMES  │  │ WYRM   │ │
│  │ Trading│  │ Health │  │ Launch │  │ Train  │ │
│  └───┬────┘  └────────┘  └────────┘  └────────┘ │
│      │                                            │
│  ┌───▼──────────────────────────────────────────┐│
│  │  CTHULU — Live Trading Dashboard             ││
│  │                                               ││
│  │  Balance: $25.12        P&L Today: +$0.43    ││
│  │  Open: 1 (EURUSDm# L)  Win Rate: 62%        ││
│  │                                               ││
│  │  ┌─── EUR/USD H1 ───────────────────────┐   ││
│  │  │  1.0842 ─╱╲──╱╲─                     │   ││
│  │  │         ╱  ╲╱  ╲───────              │   ││
│  │  │  1.0831                               │   ││
│  │  └──────────────────────────────────────┘   ││
│  │                                               ││
│  │  Last Signal: BUY EURUSDm# @ 1.0835 (A+)    ││
│  │  Tentacles: 32/32 active | Confluence: 87%    ││
│  └───────────────────────────────────────────────┘│
│                                                   │
│  [D-pad] Navigate  [A] Select  [B] Back  [Y] Ref │
└──────────────────────────────────────────────────┘
```

### 3.2 Dashboard Panels

| Panel | Data Source | Refresh |
|-------|------------|---------|
| **Cthulu Trading** | Dragonfly:9002 `/account`, `/positions`, `/ticks` | 1s |
| **System Health** | Dragonfly Mach6 `/health`, Victus ping | 10s |
| **Games** | Xbox installed games (PackageManager API) | On open |
| **WYRM Training** | Victus/Xbox training logs (heartbeat.json) | 30s |
| **Gaussian Gallery** | Local saved scenes (.xspore files) | On open |
| **Network** | LAN topology, bandwidth, latency | 5s |

### 3.3 Color Scheme

```c
// Artifact Virtual brand — deep space purple
#define AV_BG_PRIMARY     0x0D0B1AFF  // Near-black purple
#define AV_BG_SECONDARY   0x1A1433FF  // Dark purple panel
#define AV_BG_TERTIARY    0x261E47FF  // Lighter purple hover
#define AV_ACCENT_PRIMARY 0x8B5CF6FF  // Vivid purple (brand)
#define AV_ACCENT_GOLD    0xD4A843FF  // Gold accent
#define AV_TEXT_PRIMARY   0xE8E0F0FF  // Off-white
#define AV_TEXT_SECONDARY 0x9B8FBFFF  // Muted lavender
#define AV_SUCCESS        0x22C55EFF  // Green
#define AV_WARNING        0xF59E0BFF  // Amber
#define AV_ERROR          0xEF4444FF  // Red
#define AV_CHAT_USER      0x3B82F6FF  // Blue (user messages)
#define AV_CHAT_AVA       0x8B5CF6FF  // Purple (AVA messages)
```

### 3.4 Typography

Xbox 10-foot UI requirements:
- **Minimum text:** 24px (readable at 3m / 10ft)
- **Chat body:** 28px regular
- **Headers:** 36px bold
- **HUD elements:** 20px (bottom status bar only)
- **Font:** Segoe UI (Xbox system font, always available) or embedded FiraSans

Font rendering: `stb_truetype.h` (single-header, rasterize to texture atlas, GPU text rendering via quads).

### 3.5 Controller Mapping

```
┌──────────────────────────────────────┐
│           CONTROLLER MAP             │
├──────────────────────────────────────┤
│  LB / RB      → Switch modes        │
│                 (Chat/Viewport/Dash) │
│  Left Stick   → Scroll / Navigate   │
│  Right Stick  → Camera orbit (3D)   │
│  LT / RT      → Zoom in/out (3D)   │
│  A             → Select / Send msg  │
│  B             → Back / Cancel      │
│  X             → Toggle wireframe   │
│  Y             → Voice input / Ref  │
│  Menu          → Dashboard toggle   │
│  View          → Screenshot / Share │
│  D-pad         → Dashboard nav      │
│  D-pad Up      → Quick command menu │
└──────────────────────────────────────┘
```

---

## 4. Technical Implementation

### 4.1 Rendering Pipeline — D3D12

Xbox UWP in game mode = DirectX 12. No XAML, no WebView — pure GPU rendering.

```c
// Core render loop — 60 FPS target
typedef struct xspore_app {
    // D3D12 core
    ID3D12Device*              device;
    ID3D12CommandQueue*        cmd_queue;
    IDXGISwapChain4*           swap_chain;
    ID3D12DescriptorHeap*      rtv_heap;
    ID3D12DescriptorHeap*      srv_heap;     // for textures/fonts
    ID3D12Resource*            render_targets[2]; // double buffer
    ID3D12CommandAllocator*    cmd_alloc[2];
    ID3D12GraphicsCommandList* cmd_list;
    UINT                       frame_index;
    HANDLE                     fence_event;
    ID3D12Fence*               fence;
    UINT64                     fence_value;

    // UI state
    struct nk_context          nk_ctx;       // nuklear immediate-mode UI
    struct nk_d3d12_state      nk_d3d12;     // nuklear D3D12 backend
    xspore_mode                current_mode; // CHAT, VIEWPORT, DASHBOARD
    
    // Chat
    chat_history               chat;
    char                       input_buf[4096];
    bool                       awaiting_response;
    
    // 3D Viewport  
    gs_scene*                  active_scene;
    gs_camera                  camera;
    float                      orbit_yaw, orbit_pitch, orbit_dist;
    
    // Dashboard
    dashboard_state            dash;
    
    // Network
    net_client                 victus;       // LLM inference
    net_client                 dragonfly;    // Mach6 + Cthulu
    
    // App launcher
    game_entry*                installed_games;
    int                        num_games;
    
    // Input
    XINPUT_STATE               gamepad;
    bool                       virtual_kb_open;
} xspore_app;

typedef enum {
    MODE_CHAT,
    MODE_VIEWPORT,
    MODE_DASHBOARD,
    MODE_GAME_LIBRARY
} xspore_mode;
```

### 4.2 Immediate-Mode UI — nuklear

[nuklear](https://github.com/Immediate-Mode-UI/Nuklear) is a single-header ANSI C library (~18K lines) that provides:
- Layout engine (rows, columns, groups, scrollable regions)
- Widgets (buttons, text fields, labels, sliders, progress bars, trees)
- Theming / styling
- Text input with cursor
- Zero allocations (caller provides memory)

**Why nuklear over Dear ImGui:**
- Pure C (our engine is C, no C++ dependency)
- Single header (embed directly, no build system changes)
- Has a D3D12 backend
- MIT license
- Battle-tested in game UIs

```c
// Example: Chat panel rendering
void render_chat_panel(struct nk_context *ctx, xspore_app *app, 
                       float x, float y, float w, float h) {
    if (nk_begin(ctx, "Chat", nk_rect(x, y, w, h), 
                 NK_WINDOW_NO_SCROLLBAR | NK_WINDOW_BORDER)) {
        // Chat history (scrollable)
        nk_layout_row_dynamic(ctx, h - 80, 1);
        if (nk_group_scrolled_begin(ctx, &app->chat.scroll, "messages", 0)) {
            for (int i = 0; i < app->chat.count; i++) {
                chat_msg *msg = &app->chat.messages[i];
                struct nk_color col = msg->is_user ? 
                    nk_rgb(59, 130, 246) : nk_rgb(139, 92, 246);
                
                nk_layout_row_dynamic(ctx, 0, 1);  // auto-height
                nk_label_colored_wrap(ctx, msg->text, col);
            }
            nk_group_scrolled_end(ctx);
        }
        
        // Input bar
        nk_layout_row_template_begin(ctx, 50);
        nk_layout_row_template_push_dynamic(ctx);  // text field
        nk_layout_row_template_push_static(ctx, 80);  // send button
        nk_layout_row_template_end(ctx);
        
        nk_edit_string(ctx, NK_EDIT_FIELD, app->input_buf, 
                       &app->input_len, sizeof(app->input_buf), nk_filter_default);
        
        if (nk_button_label(ctx, "Send") || 
            (app->gamepad.Gamepad.wButtons & XINPUT_GAMEPAD_A)) {
            send_chat_message(app);
        }
    }
    nk_end(ctx);
}
```

### 4.3 Network Layer — LAN Communication

```c
// HTTP client using WinHTTP (available in UWP)
typedef struct net_client {
    char     host[64];
    uint16_t port;
    bool     connected;
    float    latency_ms;
    uint64_t last_ping;
} net_client;

// Endpoints
// Victus (LLM):
//   POST http://192.168.1.15:8080/v1/chat/completions (OpenAI-compatible)
//   GET  http://192.168.1.15:8080/v1/models
//
// Dragonfly (Mach6 + services):
//   GET  http://192.168.1.13:9002/health          (Cthulu health)
//   GET  http://192.168.1.13:9002/account          (Trading account)
//   GET  http://192.168.1.13:9002/positions         (Open positions)
//   GET  http://192.168.1.13:9002/ticks?symbol=X   (Latest ticks)
//   POST http://192.168.1.13:18789/v1/chat         (Mach6 gateway — talk to AVA)
//
// All communication is LAN-only. No internet required for core functionality.

// Streaming response handler (for chat)
typedef void (*stream_callback)(const char *chunk, int len, void *userdata);

int net_post_streaming(net_client *client, const char *path, 
                       const char *json_body, stream_callback cb, void *userdata);
```

### 4.4 Game Launcher

Xbox UWP apps can launch other apps via protocol URIs:

```c
// Game library — enumerate installed games
typedef struct game_entry {
    char     name[128];
    char     package_family[256];
    char     aumid[256];          // Application User Model ID
    char     logo_path[MAX_PATH]; // tile image
    uint64_t install_size;
    bool     is_game;
} game_entry;

// Launch a game from within XSpore
// Uses IApplicationActivationManager or ShellExecute with ms-xbox-*: protocol
void launch_game(const char *aumid) {
    // UWP protocol activation
    wchar_t uri[512];
    swprintf(uri, 512, L"shell:AppsFolder\\%S", aumid);
    ShellExecuteW(NULL, L"open", uri, NULL, NULL, SW_SHOWNORMAL);
    // XSpore suspends when game takes foreground
    // Resumes when user switches back
}

// Enumerate installed packages
// Uses Windows.Management.Deployment.PackageManager
// Filter: SignatureKind == Store || Developer, IsBundle = false
// Extract: DisplayName, Logo, InstalledLocation, IsGame
```

### 4.5 Chat Integration with Mach6

The chat doesn't just talk to a raw LLM — it talks to **Mach6 (Symbiote)**, meaning it talks to AVA. Full tool access, memory, personality.

```c
// Chat message to Mach6 gateway
// POST http://192.168.1.13:18789/v1/chat
// {
//   "message": "what's my trading P&L today?",
//   "session": "xbox-xspore",
//   "stream": true
// }
//
// Response includes:
// - Text reply (streamed)
// - Optional scene generation (XSpore JSON embedded)
// - Optional commands (dashboard switch, game launch, etc.)
//
// AVA can respond with embedded commands:
// [XSPORE:SCENE]{...json...}[/XSPORE:SCENE]  → render 3D scene
// [XSPORE:DASH:CTHULU]                        → switch to trading dashboard
// [XSPORE:LAUNCH:GameAUMID]                   → launch a game
// [XSPORE:CHART]{...data...}[/XSPORE:CHART]   → render 2D chart
```

### 4.6 Voice Input

Xbox has a built-in microphone on the controller (via headset) and Kinect-style voice capture.

```c
// Voice input options:
// 1. Xbox Virtual Keyboard (system-provided, controller-friendly)
//    - Triggered by Y button or focus on text field
//    - OS handles all input, returns string
//
// 2. Speech-to-text via Victus
//    - Capture audio on Xbox → stream to Victus → Whisper transcription
//    - POST http://192.168.1.15:8080/v1/audio/transcriptions
//    - Requires microphone access (Capability: microphone)
//
// 3. Xbox System Speech Recognition
//    - Windows.Media.SpeechRecognition (UWP API)
//    - Works offline for simple commands
//    - Online mode for full dictation
```

---

## 5. Gaussian Splat Viewport Integration

The 3D viewport is the showpiece. When AVA generates a scene (via XSpore pipeline), it renders in real-time with full camera control.

### 5.1 Rendering Pipeline (Xbox D3D12)

From DESIGN-01, adapted for D3D12:

```
GS Scene Data (CPU)
    │
    ▼
[Preprocess Compute Shader]  ─── frustum cull + SH eval
    │
    ▼  
[Project Compute Shader]     ─── 3D→2D + covariance
    │
    ▼
[Radix Sort]                 ─── depth-sort by tile
    │
    ▼
[Rasterize Compute Shader]  ─── alpha-blend splats per tile
    │
    ▼
Render Target → Composite with UI → Swap Chain → Display
```

### 5.2 Performance Budget

| Resource | Budget | Notes |
|----------|--------|-------|
| GPU Compute | 60% | GS rendering (4 dispatch calls) |
| GPU Graphics | 20% | UI rendering (nuklear quads + text) |
| GPU Available | 20% | Headroom for effects, transitions |
| CPU Thread 0 | Input + UI logic | Main thread |
| CPU Thread 1 | Network I/O | Async HTTP |
| CPU Thread 2 | Scene compilation | XSpore JSON → Gaussians |
| CPU Thread 3 | Audio / Voice | Optional |
| RAM | ~3GB | Scene (500MB) + UI (200MB) + Net (100MB) + LLM cache (2GB) |

### 5.3 Camera Controller

```c
void update_camera_from_gamepad(xspore_app *app, float dt) {
    XINPUT_STATE state;
    XInputGetState(0, &state);
    
    // Right stick: orbit
    float rx = state.Gamepad.sThumbRX / 32768.0f;
    float ry = state.Gamepad.sThumbRY / 32768.0f;
    
    // Dead zone
    if (fabsf(rx) < 0.15f) rx = 0;
    if (fabsf(ry) < 0.15f) ry = 0;
    
    app->orbit_yaw   += rx * 3.0f * dt;  // radians/sec
    app->orbit_pitch  = clampf(app->orbit_pitch + ry * 2.0f * dt, 
                               -1.4f, 1.4f);
    
    // Triggers: zoom
    float lt = state.Gamepad.bLeftTrigger / 255.0f;
    float rt = state.Gamepad.bRightTrigger / 255.0f;
    app->orbit_dist = clampf(app->orbit_dist + (lt - rt) * 5.0f * dt, 
                             0.5f, 100.0f);
    
    // Compute camera position on sphere around target
    float cx = app->orbit_dist * cosf(app->orbit_pitch) * sinf(app->orbit_yaw);
    float cy = app->orbit_dist * sinf(app->orbit_pitch);
    float cz = app->orbit_dist * cosf(app->orbit_pitch) * cosf(app->orbit_yaw);
    
    app->camera.position[0] = app->camera.target[0] + cx;
    app->camera.position[1] = app->camera.target[1] + cy;
    app->camera.position[2] = app->camera.target[2] + cz;
}
```

---

## 6. AppxManifest

```xml
<?xml version="1.0" encoding="utf-8"?>
<Package xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"
         xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10"
         xmlns:rescap="http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities"
         IgnorableNamespaces="uap rescap">

  <Identity Name="ArtifactVirtual.SymbioteXSpore"
            Publisher="CN=ArtifactVirtual"
            Version="1.0.0.0"
            ProcessorArchitecture="x64" />

  <Properties>
    <DisplayName>Symbiote XSpore</DisplayName>
    <PublisherDisplayName>Artifact Virtual</PublisherDisplayName>
    <Logo>Assets\StoreLogo.png</Logo>
  </Properties>

  <Dependencies>
    <TargetDeviceFamily Name="Windows.Xbox" 
                        MinVersion="10.0.22621.0" 
                        MaxVersionTested="10.0.26100.0" />
    <TargetDeviceFamily Name="Windows.Universal" 
                        MinVersion="10.0.22621.0" 
                        MaxVersionTested="10.0.26100.0" />
  </Dependencies>

  <Resources>
    <Resource Language="en-us" />
  </Resources>

  <Applications>
    <Application Id="App"
                 Executable="xspore.exe"
                 EntryPoint="Windows.FullTrustApplication">
      <uap:VisualElements
        DisplayName="Symbiote XSpore"
        Description="AI-powered 3D world generation, chat, and dashboard — by Artifact Virtual"
        BackgroundColor="#0D0B1A"
        Square150x150Logo="Assets\Square150x150Logo.png"
        Square44x44Logo="Assets\Square44x44Logo.png">
        <uap:SplashScreen Image="Assets\SplashScreen.png" BackgroundColor="#0D0B1A" />
      </uap:VisualElements>
    </Application>
  </Applications>

  <Capabilities>
    <Capability Name="internetClient" />
    <Capability Name="internetClientServer" />
    <Capability Name="privateNetworkClientServer" />
    <uap:Capability Name="userAccountInformation" />
    <DeviceCapability Name="microphone" />
    <DeviceCapability Name="gamepad" />
    <rescap:Capability Name="broadFileSystemAccess" />
    <rescap:Capability Name="runFullTrust" />
    <rescap:Capability Name="packageManagement" />
  </Capabilities>

</Package>
```

---

## 7. Source File Layout

```
projects/xspore/
├── src/
│   ├── main.c                 # Entry point, Win32 message pump, D3D12 init
│   ├── app.c                  # xspore_app lifecycle (init, update, render, shutdown)
│   ├── app.h                  # Core types and state
│   ├── d3d12_init.c           # Device, swap chain, command queue, fence
│   ├── d3d12_init.h
│   ├── ui_chat.c              # Chat panel rendering + input handling
│   ├── ui_dashboard.c         # Dashboard panels (Cthulu, system, training)
│   ├── ui_viewport.c          # 3D viewport with camera controls
│   ├── ui_games.c             # Game library browser + launcher
│   ├── ui_theme.c             # Artifact Virtual color scheme + fonts
│   ├── ui_theme.h
│   ├── net.c                  # WinHTTP async client (LLM + Cthulu + Mach6)
│   ├── net.h
│   ├── gs_renderer.c          # Gaussian splat D3D12 compute pipeline
│   ├── gs_renderer.h
│   ├── xspore_parser.c        # Parse LLM output for [XSPORE:*] commands
│   ├── input.c                # XInput gamepad + virtual keyboard triggers
│   ├── input.h
│   └── voice.c                # Speech recognition (optional, Phase 3)
├── shaders/
│   ├── gs_preprocess.hlsl     # Frustum cull + SH eval
│   ├── gs_project.hlsl        # 3D→2D projection
│   ├── gs_sort.hlsl           # Radix sort
│   ├── gs_render.hlsl         # Alpha-blend rasterizer
│   ├── ui_vertex.hlsl         # nuklear vertex shader
│   └── ui_pixel.hlsl          # nuklear pixel shader (textured quads)
├── lib/
│   ├── nuklear.h              # Single-header UI library
│   ├── nuklear_d3d12.h        # D3D12 backend for nuklear
│   ├── stb_truetype.h         # Font rasterization
│   └── stb_image.h            # Image loading (game tiles, assets)
├── assets/
│   ├── StoreLogo.png          # 50x50
│   ├── Square44x44Logo.png    # App list icon
│   ├── Square150x150Logo.png  # Tile icon
│   ├── SplashScreen.png       # 620x300 launch screen
│   ├── ava-avatar.png         # Chat avatar for AVA
│   └── FiraSans-Regular.ttf   # Embedded font (fallback)
├── AppxManifest.xml
├── build_xspore.bat           # MSVC compile + APPX package
├── deploy_xspore.sh           # Dragonfly → Victus → Xbox pipeline
└── README.md
```

**Estimated LOC:** ~4,000-5,000 C + ~600 HLSL + headers

---

## 8. Build Pipeline

```bash
# On Dragonfly: package source
tar czf xspore-src.tar.gz -C projects/xspore src/ shaders/ lib/ assets/ \
    AppxManifest.xml build_xspore.bat

# Transfer to Victus
scp xspore-src.tar.gz victus:C:/workspace/

# On Victus (SSH): build + package
ssh victus "cd C:\workspace && tar xzf xspore-src.tar.gz && build_xspore.bat --package"

# Deploy to Xbox (from Dragonfly)
APPX=$(ssh victus "dir /b C:\workspace\build\*.appx")
scp victus:C:/workspace/build/$APPX /tmp/
curl -k -X POST \
  -F "file=@/tmp/$APPX;filename=XSpore.appx" \
  "https://192.168.1.15:11443/api/app/packagemanager/package?package=XSpore.appx" \
  -u artifact:sirius

# Or via SMB (faster for large packages)
smbclient //192.168.1.15/DevelopmentFiles -U DevToolsUser -c "put /tmp/$APPX xspore/XSpore.appx"
```

---

## 9. Implementation Phases

### Phase 1: Skeleton (3-4 days)
- D3D12 initialization + swap chain + render loop
- nuklear integration with D3D12 backend
- Font loading (stb_truetype → texture atlas)
- Three-panel layout (chat, viewport placeholder, dashboard placeholder)
- Controller input (XInput)
- Build + deploy pipeline verified on Xbox

**Deliverable:** App launches on Xbox, shows purple UI with placeholder panels, controller navigates between modes.

### Phase 2: Chat + Network (2-3 days)
- WinHTTP async client
- Mach6 gateway integration (POST /v1/chat with streaming)
- Chat history with message bubbles
- Virtual keyboard trigger on text input focus
- LLM response streaming (word-by-word appearance)
- [XSPORE:*] command parsing

**Deliverable:** Talk to AVA from Xbox. Streamed responses. Full conversation history.

### Phase 3: 3D Viewport (3-4 days)
- Port DESIGN-01 compute shaders to HLSL/D3D12
- GS preprocess → project → sort → render pipeline
- XSpore JSON → procedural Gaussian generation
- Camera orbit controller (right stick + triggers)
- Scene compositing with UI overlay
- Wireframe toggle, screenshot capture

**Deliverable:** Say "red sphere" → AVA generates XSpore JSON → renders as Gaussians → orbit with controller.

### Phase 4: Dashboard + Games (2-3 days)
- Cthulu trading dashboard (live data from Dragonfly:9002)
- System health panel
- WYRM training monitor
- Game library enumeration (PackageManager API)
- Game launching (protocol activation)
- Chart rendering (sparklines, bar charts via nuklear custom draw)

**Deliverable:** Full dashboard with live trading data. Launch any installed game from within XSpore.

### Phase 5: Polish (2-3 days)
- Splash screen animation
- Transitions between modes (fade/slide)
- Sound effects (UI navigation, message received, scene complete)
- Voice input integration
- Scene gallery (save/load .xspore files)
- Performance profiling + optimization

**Deliverable:** Production-quality Xbox app.

**Total estimate: 12-17 days for full implementation.**

---

## 10. Dependencies

| Library | Purpose | Size | License |
|---------|---------|------|---------|
| nuklear.h | Immediate-mode UI | 18K lines, single header | MIT |
| nuklear_d3d12.h | D3D12 rendering backend | ~2K lines | MIT |
| stb_truetype.h | Font rasterization | Single header | MIT/Public Domain |
| stb_image.h | Image loading (PNG) | Single header | MIT/Public Domain |
| D3D12 SDK | DirectX 12 API | System (Xbox) | Microsoft |
| XInput | Controller input | System (Xbox) | Microsoft |
| WinHTTP | HTTP client | System (Xbox) | Microsoft |

**Zero external downloads.** Everything is either single-header C (bundled) or Xbox system API. No NuGet, no npm, no pip. Pure C. Compiles with MSVC on Victus, runs on Xbox.

---

## 11. What This Means

This isn't an app. It's the **console interface for Symbiote.** 

Every piece of Artifact Virtual's infrastructure — AVA, Cthulu, WYRM, HEKTOR, Mach6 — converges here. The Xbox becomes the consumer endpoint that Ali (or anyone) walks up to, picks up a controller, and interacts with the entire system through natural conversation and 3D visualization.

The LLM generates scenes. The engine renders them. The dashboard monitors everything. The chat connects to AVA. All of it — game mode, 5GB RAM, full GPU, D3D12 — for the cost of an Xbox that was already sitting there.

"A spore grows worlds from seeds. The Xbox is the terrarium."
