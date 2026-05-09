# DESIGN-01: Gaussian Splat Native Renderer
## Artifact Engine — Workstream 1

**Author:** AVA (Artifact Virtual)  
**Date:** April 6, 2026  
**Version:** 1.0  
**Status:** Architecture Blueprint — Ready for Implementation  
**Engine Version:** v0.7.0 (8,536 LOC C, 11 compute shaders)

---

## 1. Objective

Add a native 3D Gaussian Splatting (3DGS) renderer to Artifact Engine, enabling real-time photorealistic scene rendering on Xbox Series X, Windows, and Linux from the same C codebase. The renderer will share the engine's existing Vulkan compute infrastructure and integrate with the LLM inference pipeline, creating a unified binary that can **think** (language) and **see** (3D vision).

### Success Criteria
- Load and render a 1M Gaussian scene at **≥60 FPS @ 1080p** on Xbox Series X (RDNA 2)
- Load and render at **≥100 FPS @ 1080p** on Victus (RTX 2050)
- Support PLY, SPLAT, and SPZ file formats
- Memory budget: **≤2 GB VRAM** for a standard scene
- Integration: accessible via existing HTTP API (`POST /v1/render`)
- Latency: first frame **<500ms** after load

---

## 2. Architecture Overview

### 2.1 Current Engine Architecture

```
┌─────────────────────────────────────────────────┐
│                  Artifact Engine v0.7.0          │
├─────────────┬───────────────┬───────────────────┤
│   main.c    │  http_server  │   companion.c     │
│  (675 LOC)  │  (571 LOC)   │   (835 LOC)       │
├─────────────┴───────────────┴───────────────────┤
│                   engine.c (1,412 LOC)          │
│        Transformer forward pass + generate      │
│   Hybrid: Attention / DeltaNet / Mamba / Mamba2 │
├─────────────────────────────────────────────────┤
│             Compute Backends (pick one)          │
│  ┌──────────────┬──────────────┬──────────────┐ │
│  │ vulkan_      │ directml_    │ cpu_         │ │
│  │ compute.c    │ compute.c    │ compute.c    │ │
│  │ (713 LOC)    │ (1,460 LOC)  │ (1,016 LOC)  │ │
│  │ 11 shaders   │ DirectML API │ Pure C       │ │
│  └──────────────┴──────────────┴──────────────┘ │
├─────────────────────────────────────────────────┤
│   gguf.c (639) │ tokenizer.c (402) │ fetch (296)│
└─────────────────────────────────────────────────┘
```

### 2.2 Extended Architecture (with GS)

```
┌───────────────────────────────────────────────────────┐
│                 Artifact Engine v0.8.0                 │
├──────────────┬──────────────┬─────────────────────────┤
│   main.c     │ http_server  │  companion.c            │
│  +--splat    │ +/v1/render  │  +vision_3d integration │
│  +--scene    │ +/v1/scene   │                         │
├──────────────┴──────────────┴─────────────────────────┤
│     engine.c (LLM)     │    gs_engine.c (3DGS) [NEW] │
│     Forward pass        │    Load → Sort → Render     │
│     Token generation    │    Camera → Framebuffer     │
├─────────────────────────┴─────────────────────────────┤
│                  Compute Backends                      │
│  ┌──────────────┬──────────────┬────────────────────┐ │
│  │ vulkan_      │ directml_    │ cpu_               │ │
│  │ compute.c    │ compute.c    │ compute.c          │ │
│  │ +4 GS shaders│ +DX12 compute│ +GS CPU fallback   │ │
│  └──────────────┴──────────────┴────────────────────┘ │
├───────────────────────────────────────────────────────┤
│ gguf.c │ tokenizer.c │ gs_loader.c [NEW] │ gs_proc.c │
│        │             │ PLY/SPLAT/SPZ     │ [NEW]     │
│        │             │ parser            │ procedural │
└───────────────────────────────────────────────────────┘
```

### 2.3 New Files

| File | LOC (est.) | Purpose |
|------|-----------|---------|
| `src/gs_engine.c` | 600-800 | GS renderer: load, sort, render, camera |
| `src/gs_loader.c` | 400-500 | PLY/SPLAT/SPZ file parsers |
| `src/gs_procedural.c` | 300-400 | Procedural Gaussian generation (spheres, cubes, etc.) |
| `include/gs_engine.h` | 150 | Public API for GS subsystem |
| `include/gs_loader.h` | 80 | Loader interface |
| `include/gs_procedural.h` | 60 | Procedural generator interface |
| `shaders/gs_project.comp` | 80-100 | 3D→2D Gaussian projection |
| `shaders/gs_sort.comp` | 150-200 | GPU radix sort (per-tile) |
| `shaders/gs_render.comp` | 120-150 | Tile-based alpha compositing |
| `shaders/gs_preprocess.comp` | 60-80 | Frustum culling + tile assignment |

**Total new code:** ~2,000-2,500 LOC C + ~500 LOC GLSL/HLSL

---

## 3. Data Structures

### 3.1 Gaussian Representation (GPU)

```c
/* 
 * Packed Gaussian — 236 bytes per splat (59 floats)
 * Aligned for GPU structured buffer access.
 */
typedef struct __attribute__((packed)) {
    float pos[3];           /* World position (x, y, z) */
    float scale[3];         /* Log-scale (sx, sy, sz) — exp() before use */
    float rot[4];           /* Rotation quaternion (w, x, y, z), normalized */
    float opacity;          /* Sigmoid-space opacity — sigmoid() before use */
    float sh[48];           /* Spherical harmonics: 3 channels × 16 coeffs (degree 3) */
} gaussian_splat;

/*
 * Compact Gaussian — 32 bytes per splat (antimatter15 SPLAT format)
 * Position + scale + rotation + RGBA, no view-dependent color.
 */
typedef struct __attribute__((packed)) {
    float    pos[3];        /* World position */
    float    scale[3];      /* Log-scale */
    uint8_t  rot[4];        /* Quantized rotation (uint8 → normalize to [-1,1]) */
    uint8_t  rgba[4];       /* Base color + alpha */
} gaussian_compact;
```

### 3.2 Sort Key

```c
/*
 * Sort key for depth ordering.
 * Pack (tile_id, depth) into a single uint64_t for radix sort.
 * Upper 32 bits = tile_id, lower 32 bits = depth (float as uint32 via IEEE 754).
 */
typedef struct {
    uint32_t tile_id;       /* Screen tile index (row * tiles_x + col) */
    uint32_t depth_bits;    /* Float depth reinterpreted as uint32 (preserves order for positive depths) */
    uint32_t gaussian_idx;  /* Original index into gaussian array */
} gs_sort_key;
```

### 3.3 Scene

```c
/*
 * A loaded Gaussian Splat scene.
 */
typedef struct {
    /* Raw Gaussian data on GPU */
    gpu_buffer  gaussians;      /* Structured buffer of gaussian_splat[] */
    uint32_t    n_gaussians;    /* Total Gaussian count */
    
    /* Derived buffers (computed per frame) */
    gpu_buffer  projected;      /* 2D screen-space means + cov2d [n_visible × 8 floats] */
    gpu_buffer  sort_keys;      /* Sort keys [n_visible × 12 bytes] */
    gpu_buffer  sort_values;    /* Sorted indices [n_visible × 4 bytes] */
    gpu_buffer  tile_ranges;    /* Per-tile (start, count) into sorted array */
    gpu_buffer  framebuffer;    /* Output RGBA [width × height × 4 floats] */
    
    /* Scratch for radix sort */
    gpu_buffer  sort_scratch_keys;
    gpu_buffer  sort_scratch_vals;
    gpu_buffer  prefix_sums;
    
    /* Bounding box (for frustum culling) */
    float       bbox_min[3];
    float       bbox_max[3];
    
    /* Scene metadata */
    uint32_t    sh_degree;      /* 0 = RGB only, 1-3 = SH bands */
    bool        loaded;
} gs_scene;
```

### 3.4 Camera

```c
typedef struct {
    float   pos[3];         /* Camera world position */
    float   target[3];      /* Look-at target */
    float   up[3];          /* Up vector */
    float   fov_y;          /* Vertical field of view (radians) */
    float   aspect;         /* Width / Height */
    float   near;           /* Near clip plane */
    float   far;            /* Far clip plane */
    
    /* Derived (computed from above) */
    float   view[16];       /* 4×4 view matrix (row-major) */
    float   proj[16];       /* 4×4 projection matrix */
    float   viewproj[16];   /* Combined view-projection */
    
    /* Viewport */
    uint32_t width;
    uint32_t height;
    uint32_t tile_size;     /* 16 (default) — screen divided into tile_size² blocks */
    uint32_t tiles_x;       /* ceil(width / tile_size) */
    uint32_t tiles_y;       /* ceil(height / tile_size) */
} gs_camera;
```

---

## 4. Rendering Pipeline

The GS rendering pipeline runs entirely on the GPU via compute shaders. Four stages, all dispatched in a single command buffer:

```
  gaussians[]     camera
       │             │
       ▼             ▼
┌─────────────────────────┐
│  Stage 1: PREPROCESS    │  gs_preprocess.comp
│  Frustum cull           │  Input:  gaussians[], viewproj, viewport
│  Compute screen bounds  │  Output: visible_indices[], n_visible (atomic)
│  Assign to tiles        │  Dispatch: ceil(n_gaussians / 256) × 1 × 1
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  Stage 2: PROJECT       │  gs_project.comp
│  3D Gaussian → 2D splat │  Input:  gaussians[visible], view, proj
│  Compute 2D cov + depth │  Output: projected[], sort_keys[]
│  Pack sort keys          │  Dispatch: ceil(n_visible / 256) × 1 × 1
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  Stage 3: SORT          │  gs_sort.comp
│  GPU radix sort         │  Input:  sort_keys[]
│  Sort by (tile, depth)  │  Output: sorted_indices[], tile_ranges[]
│  8-pass, 4-bit radix    │  Dispatch: per-pass, varies
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  Stage 4: RENDER        │  gs_render.comp
│  Per-tile alpha blend   │  Input:  projected[], sorted_indices[], tile_ranges[]
│  Front-to-back compose  │  Output: framebuffer[width × height × RGBA]
│  Early termination      │  Dispatch: tiles_x × tiles_y × 1
│  @ α > 0.9999           │  Workgroup: tile_size × tile_size × 1
└────────────┬────────────┘
             ▼
        framebuffer[]
        (display / readback / HTTP response)
```

### 4.1 Stage 1: Preprocess (gs_preprocess.comp)

```glsl
#version 450
layout(local_size_x = 256) in;

struct Gaussian {
    vec3  pos;
    vec3  scale;
    vec4  rot;
    float opacity;
    float sh[48];
};

layout(set=0, binding=0) readonly buffer Gaussians { Gaussian gs[]; };
layout(set=0, binding=1) writeonly buffer VisibleIdx { uint visible[]; };
layout(set=0, binding=2) buffer AtomicCount { uint n_visible; };

layout(push_constant) uniform Params {
    mat4  viewproj;
    vec4  frustum_planes[6];  /* packed as (nx, ny, nz, d) */
    uvec2 viewport;           /* (width, height) */
    float near_clip;
    float far_clip;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= gs.length()) return;
    
    vec3 p = gs[idx].pos;
    
    /* Frustum culling — test against 6 planes */
    /* Use 3σ bounding sphere: radius = max(exp(scale)) * 3.0 */
    float radius = 3.0 * max(max(exp(gs[idx].scale.x), exp(gs[idx].scale.y)), 
                              exp(gs[idx].scale.z));
    
    for (int i = 0; i < 6; i++) {
        if (dot(frustum_planes[i].xyz, p) + frustum_planes[i].w < -radius)
            return;  /* Outside frustum */
    }
    
    /* Passed culling — add to visible list */
    uint slot = atomicAdd(n_visible, 1);
    visible[slot] = idx;
}
```

### 4.2 Stage 2: Project (gs_project.comp)

The projection stage transforms each 3D Gaussian into a 2D screen-space ellipse. This is the mathematical core of 3DGS:

**3D Covariance from quaternion + scale:**
```
Σ₃ₓ₃ = R · S · Sᵀ · Rᵀ
```
where `R = rotation_matrix(quaternion)`, `S = diag(exp(scale))`.

**Project to 2D:**
```
Σ₂ₓ₂ = J · W · Σ₃ₓ₃ · Wᵀ · Jᵀ
```
where `W = view matrix (3×3)`, `J = Jacobian of perspective projection`.

The Jacobian of perspective projection at point `t = (tx, ty, tz)` in camera space:
```
J = | 1/tz   0    -tx/tz² |
    | 0     1/tz  -ty/tz² |
```
(The focal length is folded into the projection matrix.)

**Output per visible Gaussian (8 floats):**
- `screen_xy` (2 floats): 2D center in pixels
- `cov2d` (3 floats): upper triangle of 2×2 covariance [a, b, d] where Σ₂ = [[a,b],[b,d]]
- `depth` (1 float): Z distance for sorting
- `color_rgb` (3 floats): evaluated SH at camera direction (or base RGB if SH degree 0)
- `opacity` (1 float): sigmoid(raw_opacity)

### 4.3 Stage 3: Sort (gs_sort.comp)

GPU radix sort on 64-bit keys (tile_id:32 | depth:32). This is the performance-critical stage.

**Algorithm:** 8-pass radix sort, 4 bits per pass:
1. For each pass (8 total, processing 4 bits at a time):
   a. **Histogram:** Count occurrences of each 4-bit digit (16 buckets) per workgroup
   b. **Prefix sum:** Exclusive scan across workgroups to get global offsets
   c. **Scatter:** Write elements to sorted positions

**Reference:** VkRadixSort (Vulkan), CUB DeviceRadixSort (CUDA).

**Key insight:** For GS rendering, sort stability doesn't matter — we only need correct depth ordering within each tile. This allows optimizations:
- Sort only within tiles (reduces problem size)
- Use 32-bit depth sort within each tile (skip tile_id in sort key if tiles are pre-bucketed)

**Performance target:** Sort 2M keys in <2ms on RDNA 2.

**Alternative for v0.8.0 MVP:** Bitonic sort (simpler to implement, O(n·log²n) vs O(n·k) for radix). Slower but correct. Upgrade to radix in v0.9.0.

### 4.4 Stage 4: Render (gs_render.comp)

```glsl
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

/* Per-tile: iterate through sorted Gaussians, alpha-blend front-to-back */
void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    if (pixel.x >= viewport.x || pixel.y >= viewport.y) return;
    
    uint tile_x = pixel.x / TILE_SIZE;
    uint tile_y = pixel.y / TILE_SIZE;
    uint tile_id = tile_y * tiles_x + tile_x;
    
    /* Get range of sorted Gaussians for this tile */
    uint start = tile_ranges[tile_id * 2];
    uint count = tile_ranges[tile_id * 2 + 1];
    
    vec3 color = vec3(0.0);
    float transmittance = 1.0;  /* starts at 1, decreases */
    
    for (uint i = 0; i < count; i++) {
        if (transmittance < 0.0001) break;  /* Early termination */
        
        uint gs_idx = sorted_indices[start + i];
        
        /* Read 2D Gaussian params */
        vec2 mean = projected[gs_idx].screen_xy;
        vec3 cov2d = projected[gs_idx].cov2d;  /* [a, b, d] */
        vec3 rgb = projected[gs_idx].color;
        float alpha_raw = projected[gs_idx].opacity;
        
        /* Evaluate 2D Gaussian at this pixel */
        vec2 d = vec2(pixel) + 0.5 - mean;
        
        /* Inverse of 2×2 covariance: [[a,b],[b,d]] → det = a*d - b*b */
        float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
        if (det <= 0.0) continue;
        
        float inv_det = 1.0 / det;
        float power = -0.5 * (cov2d.z * d.x * d.x - 2.0 * cov2d.y * d.x * d.y 
                              + cov2d.x * d.y * d.y) * inv_det;
        
        if (power > 0.0) continue;   /* Outside Gaussian */
        if (power < -4.0) continue;  /* Too far from center (3σ ≈ exp(-4.5)) */
        
        float gaussian = exp(power);
        float alpha = min(0.99, alpha_raw * gaussian);
        
        if (alpha < 1.0/255.0) continue;  /* Invisible */
        
        /* Front-to-back alpha compositing */
        color += transmittance * alpha * rgb;
        transmittance *= (1.0 - alpha);
    }
    
    /* Write to framebuffer (premultiplied alpha → add background) */
    vec3 bg = vec3(0.0, 0.0, 0.0);  /* Black background */
    color += transmittance * bg;
    
    uint fb_idx = pixel.y * viewport.x + pixel.x;
    framebuffer[fb_idx] = vec4(color, 1.0 - transmittance);
}
```

---

## 5. File Format Loaders

### 5.1 PLY Loader

The standard 3DGS output format. Binary little-endian PLY with custom properties:

```
ply
format binary_little_endian 1.0
element vertex 1234567
property float x
property float y
property float z
property float nx        (unused — normal_x)
property float ny
property float nz
property float f_dc_0    (SH degree 0, channel 0 = R)
property float f_dc_1    (SH degree 0, channel 1 = G)
property float f_dc_2    (SH degree 0, channel 2 = B)
property float f_rest_0  (SH higher degrees, 45 floats)
...
property float f_rest_44
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0     (quaternion w)
property float rot_1     (quaternion x)
property float rot_2     (quaternion y)
property float rot_3     (quaternion z)
end_header
<binary data>
```

**Per vertex:** 62 floats × 4 bytes = 248 bytes.  
**1M Gaussians:** ~237 MB.

Parser strategy:
1. Read ASCII header, extract element count and property list
2. Compute byte offset per property and total stride
3. `mmap()` the file (same pattern as `gguf.c`)
4. Copy properties into our `gaussian_splat` struct layout with field remapping
5. Upload to GPU via staging buffer

### 5.2 SPLAT Loader

Antimatter15's compact format. No header beyond a magic byte. 32 bytes per Gaussian:

```
Offset  Size  Field
0       12    position (3 × float32)
12      12    scale (3 × float32)
24      4     rotation (4 × uint8, normalized to [-1,1])
28      4     RGBA (4 × uint8)
```

**1M Gaussians:** ~30.5 MB. No SH — RGB only.

### 5.3 SPZ Loader (Niantic Compressed)

Quantized + entropy coded. Significantly smaller. Header + compressed blocks.
Defer to v0.9.0 — PLY and SPLAT cover MVP.

---

## 6. Spherical Harmonics Evaluation

View-dependent color from SH coefficients. At render time, compute the viewing direction `d = normalize(camera_pos - gaussian_pos)`, then evaluate:

```
/* SH degree 0 (constant) — 1 coefficient per channel */
color = SH_C0 * sh[0..2]

/* SH degree 1 — 3 coefficients per channel */
+ SH_C1 * (-y * sh[3..5] + z * sh[6..8] - x * sh[9..11])

/* SH degree 2 — 5 coefficients per channel */
+ SH_C2_0 * (xy * sh[12..14])
+ SH_C2_1 * (yz * sh[15..17])
+ SH_C2_2 * ((2zz - xx - yy) * sh[18..20])
+ SH_C2_3 * (xz * sh[21..23])
+ SH_C2_4 * ((xx - yy) * sh[24..26])

/* SH degree 3 — 7 coefficients per channel */
... (16 total coefficients per channel)

color = max(color + 0.5, 0.0)  /* DC offset + clamp */
```

Constants: `SH_C0 = 0.28209479`, `SH_C1 = 0.48860251`, etc.

For SPLAT format (no SH), just use the RGBA directly — much cheaper.

**Optimization:** Precompute SH at load time if camera is static (e.g., for single-view rendering). For interactive viewing, evaluate per-frame in the project shader.

---

## 7. Compute Backend Integration

### 7.1 Vulkan (Primary — Linux, Victus)

Add 4 new shader IDs to `vulkan_compute.h`:

```c
typedef enum {
    /* ... existing 17 shaders ... */
    SHADER_GS_PREPROCESS,    /* Frustum cull + visibility */
    SHADER_GS_PROJECT,       /* 3D→2D projection */
    SHADER_GS_SORT,          /* GPU radix sort */
    SHADER_GS_RENDER,        /* Tile-based alpha blend */
    SHADER_COUNT             /* Update count */
} shader_id;
```

New shaders use the existing `vk_dispatch()` infrastructure — same descriptor set layout, push constants, command buffer recording. No Vulkan API changes needed.

**Memory allocation:** GS buffers use `vk_alloc_device()` for GPU-local storage. Framebuffer uses `vk_alloc_staging()` if CPU readback is needed (HTTP response), or stays device-local for display pass-through.

### 7.2 DirectML (Xbox Series X)

Xbox uses DirectX 12 compute, not Vulkan. The `directml_compute.c` backend already has DX12 device creation, command queues, and resource management.

**Translation strategy:**
- GLSL compute shaders → cross-compile to HLSL via `glslangValidator --target-env vulkan1.0` + `spirv-cross --hlsl`
- Or write HLSL versions directly (preferred for Xbox-specific optimizations)
- DX12 structured buffers = Vulkan storage buffers
- DX12 root signatures ≈ Vulkan pipeline layouts
- DX12 dispatch groups = Vulkan dispatch groups

**Xbox-specific optimizations:**
- Use `ExecuteIndirect` for variable-length dispatch (n_visible not known until preprocess runs)
- Wave intrinsics (`WaveActiveSum`, `WavePrefixSum`) for efficient prefix sums in sort
- LDS (Local Data Share) = shared memory — same 32KB/workgroup as Vulkan

### 7.3 CPU Fallback

For development/testing, implement scalar C versions of all stages in `cpu_compute.c`:

```c
void cpu_gs_preprocess(const gaussian_splat* gs, uint32_t n, 
                       const float viewproj[16], uint32_t* visible, uint32_t* n_visible);
void cpu_gs_project(const gaussian_splat* gs, const uint32_t* visible, uint32_t n_visible,
                    const gs_camera* cam, gs_projected* out);
void cpu_gs_sort(gs_sort_key* keys, uint32_t n);  /* qsort wrapper */
void cpu_gs_render(const gs_projected* projected, const uint32_t* sorted,
                   const uint32_t* tile_ranges, const gs_camera* cam,
                   float* framebuffer);
```

Performance: ~2-5 FPS for 1M Gaussians on CPU. Useful for correctness validation.

---

## 8. Public API

### 8.1 C API (gs_engine.h)

```c
#ifndef GS_ENGINE_H
#define GS_ENGINE_H

#include "vulkan_compute.h"
#include <stdbool.h>
#include <stdint.h>

/* Forward declarations */
typedef struct gs_scene gs_scene;
typedef struct gs_camera gs_camera;

/* ═══ Scene Management ═══ */

/* Load a Gaussian Splat scene from file (PLY, SPLAT, or SPZ) */
gs_scene* gs_load(vk_context* vk, const char* path);

/* Load from procedural generator output */
gs_scene* gs_load_from_buffer(vk_context* vk, const float* data, 
                               uint32_t n_gaussians, uint32_t sh_degree);

/* Free a scene and all GPU resources */
void gs_free(vk_context* vk, gs_scene* scene);

/* Get scene info */
uint32_t gs_get_count(const gs_scene* scene);
void gs_get_bbox(const gs_scene* scene, float min[3], float max[3]);

/* ═══ Camera ═══ */

/* Create a camera for the given viewport */
gs_camera* gs_camera_create(uint32_t width, uint32_t height, float fov_y);

/* Set camera position/orientation */
void gs_camera_lookat(gs_camera* cam, 
                      float eye_x, float eye_y, float eye_z,
                      float target_x, float target_y, float target_z,
                      float up_x, float up_y, float up_z);

/* Orbit camera controls */
void gs_camera_orbit(gs_camera* cam, float delta_yaw, float delta_pitch, float delta_distance);

/* Update derived matrices (call after any position change) */
void gs_camera_update(gs_camera* cam);

void gs_camera_free(gs_camera* cam);

/* ═══ Rendering ═══ */

/* Render scene from camera to internal framebuffer.
 * Returns number of visible Gaussians rendered. */
uint32_t gs_render(vk_context* vk, gs_scene* scene, const gs_camera* cam);

/* Read back framebuffer to CPU memory.
 * Output: RGBA float32, caller must allocate width*height*4*sizeof(float). */
bool gs_readback(vk_context* vk, const gs_scene* scene, float* rgba_out);

/* Read back framebuffer as 8-bit RGBA (for PNG/JPEG encoding).
 * Output: RGBA uint8, caller must allocate width*height*4. */
bool gs_readback_u8(vk_context* vk, const gs_scene* scene, uint8_t* rgba_out);

/* ═══ Render Stats ═══ */

typedef struct {
    uint32_t total_gaussians;
    uint32_t visible_gaussians;
    float    preprocess_ms;
    float    project_ms;
    float    sort_ms;
    float    render_ms;
    float    total_ms;
    size_t   gpu_memory_bytes;
} gs_render_stats;

void gs_get_stats(const gs_scene* scene, gs_render_stats* stats);

#endif /* GS_ENGINE_H */
```

### 8.2 HTTP API Extensions

Add to `http_server.c`:

```
POST /v1/render
{
    "scene": "path/to/scene.ply",
    "camera": {
        "position": [0, 0, 5],
        "target": [0, 0, 0],
        "up": [0, 1, 0],
        "fov": 60,
        "width": 1920,
        "height": 1080
    },
    "format": "png"  // or "raw" for float32 RGBA
}
→ Binary PNG response (or raw float32 buffer)

GET /v1/scene/info
→ { "n_gaussians": 1234567, "bbox_min": [...], "bbox_max": [...], "sh_degree": 3 }

POST /v1/scene/load
{ "path": "path/to/scene.ply" }
→ { "status": "loaded", "n_gaussians": 1234567 }
```

---

## 9. Memory Budget

### 9.1 Per-Scene (1M Gaussians)

| Buffer | Size | Notes |
|--------|------|-------|
| `gaussians` (full PLY) | 236 MB | 236 bytes × 1M |
| `gaussians` (compact SPLAT) | 30.5 MB | 32 bytes × 1M |
| `projected` | 32 MB | 32 bytes × 1M (8 floats) |
| `sort_keys` | 12 MB | 12 bytes × 1M |
| `sort_scratch` | 24 MB | 2× sort_keys for ping-pong |
| `tile_ranges` | 0.5 MB | 8 bytes × 64K tiles (1080p) |
| `framebuffer` | 32 MB | 1920×1080 × 4 × 4 bytes (float32 RGBA) |
| **Total (PLY)** | **~337 MB** | |
| **Total (SPLAT)** | **~131 MB** | |

Well within the Xbox's 16 GB unified memory and the RTX 2050's 4 GB VRAM.

### 9.2 Scaling

| Gaussians | PLY VRAM | SPLAT VRAM | 
|-----------|----------|------------|
| 500K | 169 MB | 66 MB |
| 1M | 337 MB | 131 MB |
| 3M | 1.01 GB | 393 MB |
| 5M | 1.68 GB | 655 MB |
| 10M | 3.37 GB | 1.31 GB |

**Recommendation:** Target 1-3M Gaussians for Xbox. 5M+ for Victus. Compression (SPZ) can extend range further.

---

## 10. Performance Model

### 10.1 Theoretical Throughput

**Xbox Series X (RDNA 2, 52 CUs, 12 TFLOPS):**
- Preprocess (1M): ~0.3 ms (simple culling, memory bound)
- Project (500K visible): ~0.5 ms (moderate ALU)
- Sort (500K): ~1.5 ms (radix sort, memory bandwidth limited)
- Render (1080p, 128×68 tiles): ~2.0 ms (per-tile iteration, early termination helps)
- **Total: ~4.3 ms → ~230 FPS**

**RTX 2050 (2048 CUDA cores, 4.5 TFLOPS):**
- Slightly lower compute, but NVIDIA hardware excels at atomic operations
- **Estimated: ~6 ms → ~166 FPS**

**CPU fallback (i3-1005G1, 4 threads):**
- **Estimated: 200-500 ms → 2-5 FPS** (viable for testing only)

### 10.2 Bottleneck Analysis

1. **Sort is the bottleneck** — always. 50-60% of frame time.
2. **Memory bandwidth** — Gaussians are large. SPLAT format (32 bytes) is 7.4× smaller than PLY (236 bytes), proportionally faster for memory-bound stages.
3. **Tile occupancy** — Dense scenes with many overlapping Gaussians slow down the render stage. Early termination (T < 0.0001) is critical.
4. **SH evaluation** — If using full degree-3 SH, the project stage does significant ALU work. Degree 1 or RGB-only is 4-16× cheaper.

---

## 11. Build Integration

### 11.1 Shader Compilation

Add to build pipeline (existing `build_xbox.bat` pattern):

```batch
@rem Compile GS shaders to SPIR-V
glslangValidator -V shaders/gs_preprocess.comp -o shaders/gs_preprocess.spv
glslangValidator -V shaders/gs_project.comp -o shaders/gs_project.spv
glslangValidator -V shaders/gs_sort.comp -o shaders/gs_sort.spv
glslangValidator -V shaders/gs_render.comp -o shaders/gs_render.spv

@rem Cross-compile to HLSL for DirectML/Xbox (if needed)
spirv-cross --hlsl --shader-model 60 shaders/gs_project.spv --output shaders/gs_project.hlsl
```

### 11.2 Source Files

Add to MSVC compile command in `build_xbox.bat`:
```batch
cl.exe /nologo /O2 /DCPU_ONLY ^
    src/main.c src/engine.c src/gguf.c src/tokenizer.c ^
    src/cpu_compute.c src/http_server.c src/model_fetch.c ^
    src/companion.c src/frame_capture.c ^
    src/gs_engine.c src/gs_loader.c src/gs_procedural.c ^
    ...
```

---

## 12. Implementation Phases

### Phase 1: MVP — Static Scene Rendering (v0.8.0)
**Target: 1-2 weeks**
- [ ] `gs_loader.c` — PLY parser (mmap-based, matching gguf.c pattern)
- [ ] `gs_loader.c` — SPLAT parser (trivial, 32 bytes/record)
- [ ] `gs_engine.c` — Scene management, camera math (view/proj matrices)
- [ ] `gs_render.comp` — Simplified single-pass renderer (no tiling, no sort — brute force)
- [ ] CPU fallback in `cpu_compute.c`
- [ ] `--splat` CLI flag in `main.c`
- [ ] Render single frame, output PNG via HTTP `/v1/render`

### Phase 2: Performance — Tiled Sort (v0.8.5)
**Target: 1 week**
- [ ] `gs_preprocess.comp` — Frustum culling
- [ ] `gs_project.comp` — Full 3D→2D with covariance
- [ ] `gs_sort.comp` — Bitonic sort (simpler first, radix later)
- [ ] `gs_render.comp` — Tile-based with early termination
- [ ] Proper SH evaluation (degree 0-3)
- [ ] Performance profiling on RTX 2050

### Phase 3: Xbox Deploy (v0.9.0)
**Target: 1 week**
- [ ] HLSL shader ports (or SPIR-V → HLSL cross-compile)
- [ ] DirectML backend integration
- [ ] APPX build + deploy to Xbox
- [ ] Test with real-world scenes (Mip-NeRF360 dataset)
- [ ] Interactive camera controls (gamepad/keyboard)

### Phase 4: Integration (v1.0.0)
**Target: 2 weeks**
- [ ] Radix sort upgrade (replace bitonic)
- [ ] SPZ loader
- [ ] XSpore pipeline integration (Workstream 3)
- [ ] WYRM specialist head integration (Workstream 2)
- [ ] Companion mode: vision + 3DGS scene understanding
- [ ] Streaming large scenes (hierarchical LOD)

---

## 13. Test Scenes

| Scene | Gaussians | Source | Purpose |
|-------|-----------|--------|---------|
| Single Gaussian | 1 | Generated | Verify projection math |
| 1000 random | 1K | Generated | Sort + render correctness |
| Lego (Synthetic) | ~300K | NeRF Synthetic | Standard benchmark |
| Bicycle | ~6M | Mip-NeRF360 | Stress test |
| Garden | ~5.8M | Mip-NeRF360 | Outdoor scene |
| Procedural sphere | 10K | gs_procedural.c | XSpore integration test |

Download links for pre-trained scenes: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip (14 GB)

---

## 14. References

1. Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering," SIGGRAPH 2023
2. 3DGS.cpp (shg8) — Vulkan Compute reference implementation
3. gsplat (nerfstudio) — CUDA reference, 4× less VRAM
4. antimatter15/splat — SPLAT format specification
5. VkRadixSort — Vulkan GPU radix sort reference

---

## 15. Open Questions

1. **Display pipeline on Xbox:** The current engine is headless (HTTP server). For real-time interactive rendering, we need a swap chain (D3D12 present queue). Should we add a minimal display backend, or pipe through Xbox's compositor?

2. **Shared memory with LLM:** When running companion mode (LLM + vision), should GS rendering and LLM inference share the same command queue, or use separate queues? Separate is safer for latency but wastes GPU bandwidth.

3. **Level of Detail:** For large scenes (>5M), implement hierarchical 3DGS (Kerbl 2024) from the start, or defer?

4. **Training on-device:** Should Artifact Engine support 3DGS training (not just rendering)? This would require differentiable rasterization + Adam optimizer on GPU. Significantly more complex but makes the engine self-sufficient.

---

*"Structure is not modality. A cell doesn't have a 'text mode.' It responds to stimuli."*

This renderer gives the engine eyes. WYRM gives it understanding. The convergence is one binary that processes structure — text, vision, 3D — because there is no such thing as multi-modal.
