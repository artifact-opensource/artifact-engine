# DESIGN-03: XSpore — Natural Language → Gaussian Generation Pipeline
## Artifact Engine — Workstream 3

**Author:** AVA (Artifact Virtual)  
**Date:** April 6, 2026  
**Version:** 1.0  
**Status:** Architecture Blueprint  
**Dependencies:** DESIGN-01 (GS Renderer), Qwen 3.5 9B (Victus)

---

## 1. Overview

**XSpore** is the pipeline that transforms natural language into rendered 3D Gaussian Splat scenes. It bridges the LLM inference engine (already running Qwen 3.5 9B on Victus/Xbox) with the Gaussian Splat renderer (DESIGN-01), creating an end-to-end system where a user says "show me a crystal cave" and sees a photorealistic 3D hologram.

The name: a spore is a reproductive unit that grows into something complex from something simple. XSpore takes a seed (text) and grows it into a 3D world.

### Design Principles
1. **Procedural first** — Don't ask the LLM to output raw floats. Let it describe; let the engine generate.
2. **Layered complexity** — Simple scenes (a sphere) should render in <100ms. Complex scenes (a forest) take longer but follow the same pipeline.
3. **Deterministic** — Same prompt + same seed = same scene. Reproducibility matters.
4. **Streamable** — Gaussians can be generated and rendered incrementally. First primitives appear fast; detail refines over time.

---

## 2. Architecture — End-to-End Data Flow

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  User     │───▶│  Qwen 3.5    │───▶│  Scene       │───▶│  GS Renderer │
│  Prompt   │    │  (LLM)       │    │  Compiler    │    │  (DESIGN-01) │
│  "red     │    │  Structured  │    │  Procedural  │    │  Project →   │
│   sphere" │    │  JSON output │    │  Generator   │    │  Sort →      │
└──────────┘    └──────────────┘    │  → Gaussians │    │  Render      │
                                     └──────────────┘    └──────┬───────┘
                                                                 │
                                                          ┌──────▼───────┐
                                                          │  Framebuffer │
                                                          │  → Display   │
                                                          │  → PNG/HTTP  │
                                                          └──────────────┘
```

### 2.1 Stage 1: Prompt → Structured Scene Description

The LLM receives a system prompt that instructs it to output a **scene description** in a strict JSON schema. The LLM does NOT output Gaussian parameters — it outputs high-level primitives, materials, and spatial relationships.

**Why this works:** LLMs are excellent at spatial reasoning and scene composition at an abstract level. They fail at producing precise floating-point coordinates. By separating "what to create" (LLM's job) from "how to create it" (engine's job), each component does what it's good at.

### 2.2 Stage 2: Scene Description → Gaussian Parameters

A **procedural generator** built into Artifact Engine (C) takes the structured scene description and generates the actual Gaussian splat parameters. This is deterministic math — sphere = sample points on surface with appropriate normals, scales, colors.

### 2.3 Stage 3: Gaussian Parameters → Rendered Frame

Standard GS rendering pipeline from DESIGN-01. Already designed.

---

## 3. Scene Description Schema

### 3.1 XSpore Scene Format (JSON)

```json
{
  "version": "1.0",
  "scene": {
    "name": "Red Sphere",
    "background": [0.1, 0.1, 0.15],
    "ambient_light": 0.3,
    "camera": {
      "position": [0, 0, 5],
      "target": [0, 0, 0],
      "fov": 60
    },
    "objects": [
      {
        "type": "sphere",
        "position": [0, 0, 0],
        "radius": 1.0,
        "material": {
          "color": [1.0, 0.2, 0.1],
          "roughness": 0.3,
          "emission": 0.0,
          "opacity": 1.0
        },
        "density": 5000,
        "name": "main_sphere"
      }
    ],
    "lights": [
      {
        "type": "point",
        "position": [3, 4, 5],
        "color": [1, 1, 1],
        "intensity": 2.0
      }
    ]
  }
}
```

### 3.2 Primitive Types

| Primitive | Parameters | Gaussian Count (at density=1.0) |
|-----------|------------|-------------------------------|
| `sphere` | position, radius | 4πr² × density |
| `cube` | position, size(xyz) | 6 × face_area × density |
| `cylinder` | position, radius, height | (2πr² + 2πrh) × density |
| `torus` | position, major_r, minor_r | 4π²Rr × density |
| `plane` | position, normal, size(wh) | w × h × density |
| `cone` | position, radius, height | πr(r + √(h²+r²)) × density |
| `point_cloud` | points[], colors[] | len(points) |
| `mesh` | vertices[], faces[] | Σ face_area × density |
| `text_3d` | string, font_size, position | varies by glyph |
| `heightmap` | data[][], size, scale | w × h of data |

### 3.3 Material Properties

```json
{
  "color": [r, g, b],          // Base color [0-1]
  "emission": 0.0,              // Emissive intensity [0-∞)
  "emission_color": [r, g, b],  // Emissive color (default = color)
  "roughness": 0.5,             // Surface roughness [0-1]
  "metallic": 0.0,              // Metallic factor [0-1]
  "opacity": 1.0,               // Transparency [0-1]
  "texture": "checkerboard",    // Procedural texture name (optional)
  "texture_scale": 1.0          // Texture UV scale
}
```

**Mapping to SH coefficients:**
- `color` → SH degree 0 (DC component): `sh[0..2] = (color - 0.5) / SH_C0`
- `roughness` → SH degree 1-3 magnitude: lower roughness = stronger specular = higher SH amplitudes
- `emission` → additive on top of SH evaluation: `final_color = sh_color + emission * emission_color`
- `metallic` → modulates SH view-dependence: metallic surfaces have colored specular
- `opacity` → direct: `gaussian.opacity = logit(material.opacity)`

### 3.4 Scene Modifiers

```json
{
  "modifiers": [
    {
      "type": "scatter",
      "source": "main_sphere",
      "count": 50,
      "radius": 3.0,
      "scale_range": [0.1, 0.3],
      "color_variation": 0.2
    },
    {
      "type": "noise",
      "target": "ground_plane",
      "amplitude": 0.5,
      "frequency": 2.0,
      "octaves": 4
    },
    {
      "type": "glow",
      "target": "crystal",
      "radius": 0.5,
      "intensity": 1.5,
      "color": [0.5, 0.8, 1.0]
    }
  ]
}
```

---

## 4. Prompt Engineering

### 4.1 System Prompt

```
You are a 3D scene architect. Given a description, output a JSON scene 
in XSpore format. Use only these primitive types: sphere, cube, cylinder, 
torus, plane, cone, point_cloud, text_3d, heightmap.

Rules:
- Position objects in a coordinate system where Y is up
- Scene should fit within a 20-unit cube centered at origin
- Set density between 1000 (sparse) and 10000 (detailed)
- Use modifiers for complex effects (scatter, noise, glow)
- Camera should be positioned to see the whole scene
- Output ONLY valid JSON, no explanation

Scene description:
```

### 4.2 Example: "Create a red sphere"

**Prompt:** `Create a red sphere floating above a dark ground plane`

**Expected LLM output:**
```json
{
  "version": "1.0",
  "scene": {
    "name": "Floating Red Sphere",
    "background": [0.02, 0.02, 0.05],
    "ambient_light": 0.2,
    "camera": {
      "position": [3, 3, 6],
      "target": [0, 1, 0],
      "fov": 50
    },
    "objects": [
      {
        "type": "sphere",
        "position": [0, 2, 0],
        "radius": 1.0,
        "material": {
          "color": [0.95, 0.15, 0.1],
          "roughness": 0.2,
          "emission": 0.1
        },
        "density": 5000
      },
      {
        "type": "plane",
        "position": [0, 0, 0],
        "normal": [0, 1, 0],
        "size": [10, 10],
        "material": {
          "color": [0.1, 0.1, 0.12],
          "roughness": 0.8
        },
        "density": 2000
      }
    ],
    "lights": [
      {"type": "point", "position": [4, 6, 3], "color": [1, 0.95, 0.9], "intensity": 2.5},
      {"type": "point", "position": [-2, 3, -1], "color": [0.3, 0.4, 0.8], "intensity": 1.0}
    ]
  }
}
```

**Gaussian generation:** sphere (5000 density × 4π(1.0)² ≈ 62,831 Gaussians) + plane (2000 × 100 = 200,000 Gaussians) = ~263K total. Renders at >200 FPS.

### 4.3 Example: "Show me a glowing crystal"

**Prompt:** `A glowing blue crystal jutting from rocky ground, with scattered smaller crystals around it`

**Expected output includes:**
- Main crystal: elongated `cylinder` with low density, high emission
- Ground: `heightmap` or `plane` with noise modifier
- Small crystals: `scatter` modifier on the main crystal
- Glow: `glow` modifier with blue color
- Dramatic lighting from below

### 4.4 Complex Scene: "A miniature city at night"

The LLM would output:
- Multiple `cube` objects for buildings (varying heights)
- `plane` for streets with grid texture
- Scattered `sphere` objects for streetlights (high emission)
- `text_3d` for signage
- Dark background, multiple warm point lights
- Camera positioned at elevated angle

---

## 5. Procedural Gaussian Generator (gs_procedural.c)

### 5.1 API

```c
#ifndef GS_PROCEDURAL_H
#define GS_PROCEDURAL_H

#include "gs_engine.h"
#include <stdint.h>

/* ═══ Primitive Generation ═══ */

/* Generate Gaussians for a sphere */
uint32_t gs_gen_sphere(gaussian_splat* out, uint32_t max_count,
                       float cx, float cy, float cz, float radius,
                       const float color[3], float roughness, float emission,
                       float opacity, uint32_t density);

/* Generate Gaussians for a cube */
uint32_t gs_gen_cube(gaussian_splat* out, uint32_t max_count,
                     float cx, float cy, float cz,
                     float sx, float sy, float sz,
                     const float color[3], float roughness, float emission,
                     float opacity, uint32_t density);

/* Generate Gaussians for a cylinder */
uint32_t gs_gen_cylinder(gaussian_splat* out, uint32_t max_count,
                         float cx, float cy, float cz,
                         float radius, float height,
                         const float color[3], float roughness, float emission,
                         float opacity, uint32_t density);

/* Generate Gaussians for a plane */
uint32_t gs_gen_plane(gaussian_splat* out, uint32_t max_count,
                      float cx, float cy, float cz,
                      float nx, float ny, float nz,
                      float width, float height,
                      const float color[3], float roughness, float emission,
                      float opacity, uint32_t density);

/* Generate Gaussians for a torus */
uint32_t gs_gen_torus(gaussian_splat* out, uint32_t max_count,
                      float cx, float cy, float cz,
                      float major_r, float minor_r,
                      const float color[3], float roughness, float emission,
                      float opacity, uint32_t density);

/* Generate Gaussians for a cone */
uint32_t gs_gen_cone(gaussian_splat* out, uint32_t max_count,
                     float cx, float cy, float cz,
                     float radius, float height,
                     const float color[3], float roughness, float emission,
                     float opacity, uint32_t density);

/* Generate Gaussians for 3D text */
uint32_t gs_gen_text3d(gaussian_splat* out, uint32_t max_count,
                       const char* text, float font_size,
                       float cx, float cy, float cz,
                       const float color[3], uint32_t density);

/* Generate Gaussians from a heightmap */
uint32_t gs_gen_heightmap(gaussian_splat* out, uint32_t max_count,
                          const float* heights, uint32_t w, uint32_t h,
                          float world_sx, float world_sz, float height_scale,
                          const float color[3], float roughness,
                          uint32_t density);

/* ═══ Modifiers ═══ */

/* Scatter: duplicate objects with random placement */
uint32_t gs_mod_scatter(gaussian_splat* out, uint32_t max_count,
                        const gaussian_splat* source, uint32_t source_count,
                        uint32_t copies, float radius,
                        float scale_min, float scale_max,
                        float color_variation, uint32_t seed);

/* Noise: perturb positions with Perlin/simplex noise */
void gs_mod_noise(gaussian_splat* gs, uint32_t count,
                  float amplitude, float frequency, uint32_t octaves,
                  uint32_t seed);

/* Glow: add emissive halo Gaussians around existing objects */
uint32_t gs_mod_glow(gaussian_splat* out, uint32_t max_count,
                     const gaussian_splat* source, uint32_t source_count,
                     float glow_radius, float intensity,
                     const float glow_color[3]);

/* ═══ Scene Compilation ═══ */

/* Parse XSpore JSON and generate all Gaussians */
gs_scene* gs_compile_xspore(vk_context* vk, const char* json, uint32_t json_len);

/* Parse XSpore JSON from file */
gs_scene* gs_compile_xspore_file(vk_context* vk, const char* path);

#endif /* GS_PROCEDURAL_H */
```

### 5.2 Sphere Generation Algorithm

The core of procedural generation. Other primitives follow similar patterns.

```c
uint32_t gs_gen_sphere(gaussian_splat* out, uint32_t max_count,
                       float cx, float cy, float cz, float radius,
                       const float color[3], float roughness, float emission,
                       float opacity, uint32_t density) {
    /* Surface area = 4πr². Gaussians = area × density */
    float area = 4.0f * M_PI * radius * radius;
    uint32_t target = (uint32_t)(area * density);
    if (target > max_count) target = max_count;
    
    /* Fibonacci sphere sampling — uniform point distribution on sphere */
    float golden_ratio = (1.0f + sqrtf(5.0f)) / 2.0f;
    float golden_angle = 2.0f * M_PI / (golden_ratio * golden_ratio);
    
    /* Scale for each Gaussian — covers surface without gaps */
    float splat_radius = radius * sqrtf(4.0f * M_PI / target) * 0.5f;
    float log_scale = logf(splat_radius);
    
    for (uint32_t i = 0; i < target; i++) {
        /* Fibonacci sphere point */
        float t = (float)i / (float)(target - 1);
        float phi = acosf(1.0f - 2.0f * t);
        float theta = golden_angle * i;
        
        float nx = sinf(phi) * cosf(theta);
        float ny = sinf(phi) * sinf(theta);
        float nz = cosf(phi);
        
        /* Position on sphere surface */
        out[i].pos[0] = cx + radius * nx;
        out[i].pos[1] = cy + radius * ny;
        out[i].pos[2] = cz + radius * nz;
        
        /* Scale: flat disc aligned with surface */
        out[i].scale[0] = log_scale;        /* tangent direction 1 */
        out[i].scale[1] = log_scale;        /* tangent direction 2 */
        out[i].scale[2] = log_scale - 1.5f; /* normal direction (thin) */
        
        /* Rotation: align Z-axis of Gaussian with surface normal */
        quaternion_from_normal(nx, ny, nz, out[i].rot);
        
        /* Opacity */
        out[i].opacity = logitf(opacity);  /* inverse sigmoid */
        
        /* SH coefficients — degree 0 = base color */
        float sh_c0 = 0.28209479f;
        out[i].sh[0] = (color[0] - 0.5f) / sh_c0;
        out[i].sh[1] = (color[1] - 0.5f) / sh_c0;
        out[i].sh[2] = (color[2] - 0.5f) / sh_c0;
        
        /* Degree 1 SH — view-dependent shading from normal direction */
        float sh_c1 = 0.48860251f;
        float specular = (1.0f - roughness) * 0.5f;
        out[i].sh[3] = -ny * specular / sh_c1;  /* Y component */
        out[i].sh[4] = -ny * specular / sh_c1;
        out[i].sh[5] = -ny * specular / sh_c1;
        out[i].sh[6] =  nz * specular / sh_c1;  /* Z component */
        out[i].sh[7] =  nz * specular / sh_c1;
        out[i].sh[8] =  nz * specular / sh_c1;
        out[i].sh[9] = -nx * specular / sh_c1;  /* X component */
        out[i].sh[10] = -nx * specular / sh_c1;
        out[i].sh[11] = -nx * specular / sh_c1;
        
        /* Higher SH degrees: zero for procedural (can add later) */
        for (int j = 12; j < 48; j++) out[i].sh[j] = 0.0f;
        
        /* Emission: boost SH DC term */
        if (emission > 0.0f) {
            out[i].sh[0] += emission / sh_c0;
            out[i].sh[1] += emission / sh_c0;
            out[i].sh[2] += emission / sh_c0;
        }
    }
    
    return target;
}
```

### 5.3 JSON Parser

Minimal JSON parser for XSpore format (no external deps):

```c
/* Lightweight JSON parser — handles XSpore schema only.
 * Not a general-purpose parser. Expects well-formed JSON from LLM.
 * Falls back gracefully on malformed input (skip object, warn). */

typedef struct {
    const char* json;
    uint32_t    pos;
    uint32_t    len;
} json_parser;

/* Parse entire XSpore scene → array of primitives + metadata */
xspore_scene* xspore_parse(const char* json, uint32_t len);

/* Each primitive has type, position, material, density */
typedef struct {
    primitive_type type;
    float pos[3];
    float params[8];   /* type-specific: radius, size, height, etc. */
    xspore_material material;
    uint32_t density;
    char name[64];
} xspore_object;

typedef struct {
    float background[3];
    float ambient;
    gs_camera camera;
    xspore_object* objects;
    uint32_t n_objects;
    xspore_light* lights;
    uint32_t n_lights;
    xspore_modifier* modifiers;
    uint32_t n_modifiers;
} xspore_scene;
```

---

## 6. 2D Gaussian Mode — Image Generation

A simpler application: generate 2D Gaussian splats to create images, diagrams, and visualizations. No 3D projection needed — just position, 2D covariance, color, and opacity.

### 6.1 Use Cases
- **Market visualization** — Cthulu data as heatmaps, price surfaces, volatility clouds
- **Artistic rendering** — Impressionist-style images from text descriptions
- **UI elements** — Soft, glowing interface components
- **Data visualization** — Scatter plots, density maps, flow fields
- **Diagrams** — Architecture diagrams with soft edges and glow effects

### 6.2 2D Gaussian Struct

```c
typedef struct {
    float pos[2];       /* Screen position (x, y) in pixels */
    float cov[3];       /* 2×2 covariance upper triangle [a, b, d] */
    float color[3];     /* RGB [0-1] */
    float opacity;      /* Alpha [0-1] */
} gaussian_2d;
```

32 bytes per splat. Rendering is trivial — no projection, no sorting (render in order), just evaluate 2D Gaussian at each pixel and alpha-blend.

### 6.3 2D Render Pipeline

```
Input: gaussian_2d[] (ordered by layer)
For each pixel (x, y):
    color = background
    for each gaussian:
        d = (x,y) - gaussian.pos
        power = -0.5 * d^T * inv(cov) * d
        if power > -4.0:
            alpha = gaussian.opacity * exp(power)
            color = color * (1-alpha) + gaussian.color * alpha
    output pixel = color
```

Single compute shader, embarrassingly parallel. Expected: 1M 2D Gaussians at 1080p in <5ms.

---

## 7. Network Architecture (Distributed)

### 7.1 Topology

```
┌─────────────────┐     HTTP/WS      ┌──────────────────┐
│  Xbox Series X   │◄───────────────▶│  Victus           │
│  Artifact Engine │     LAN          │  Qwen 3.5 9B      │
│  GS Renderer     │                  │  (GGUF inference)  │
│  Display         │                  │                    │
└─────────────────┘                  └──────────────────┘
```

### 7.2 API (Xbox → Victus)

**Request:** `POST http://192.168.1.15:8080/v1/chat/completions`
```json
{
  "model": "qwen3.5-9b",
  "messages": [
    {"role": "system", "content": "<XSpore system prompt>"},
    {"role": "user", "content": "Create a red sphere"}
  ],
  "temperature": 0.7,
  "max_tokens": 2048,
  "response_format": {"type": "json_object"}
}
```

**Response:** Standard OpenAI-compatible JSON with XSpore scene in content.

### 7.3 Local Inference Mode

If the model fits on Xbox (Q3_K_M = 4.67 GB, Xbox has 16 GB unified), run inference locally using Artifact Engine's own LLM pipeline. Zero network latency.

**Memory budget with both LLM + GS:**
- Qwen 3.5 9B Q3_K_M: ~4.7 GB
- KV cache (4096 ctx): ~1.5 GB
- GS scene (1M splats SPLAT format): ~0.13 GB
- GS render buffers: ~0.1 GB
- **Total: ~6.4 GB** — fits in Xbox's 10 GB fast pool

### 7.4 Latency Budget

| Stage | Local (Xbox) | Remote (Victus) |
|-------|-------------|-----------------|
| LLM inference | 2-5s (CPU) | 1-3s (RTX 2050) |
| Network transfer | 0 | 10-50ms |
| JSON parse | <1ms | <1ms |
| Gaussian generation | 10-50ms | 10-50ms |
| GS rendering | 5-15ms | 5-15ms |
| **Total** | **2-5s** | **1-3s** |

Interactive editing (adjusting camera, colors) is instant — only initial generation needs LLM.

---

## 8. Integration with Existing Engine Modes

### 8.1 New CLI Flag

```
artifact-engine --splat scene.ply           # Render a pre-trained GS scene
artifact-engine --xspore "red sphere"       # Generate + render from text
artifact-engine --xspore-file scene.xspore  # Load XSpore JSON file
artifact-engine --xspore-server             # Interactive XSpore server mode
```

### 8.2 Companion Mode Integration

The companion (game AI) can use XSpore to visualize its understanding:

```c
/* In companion.c, when companion has spatial information: */
if (companion_wants_to_show_map(comp)) {
    const char* scene_json = companion_generate_map_xspore(comp);
    gs_scene* vis = gs_compile_xspore(&eng.vk, scene_json, strlen(scene_json));
    gs_render(&eng.vk, vis, &cam);
    /* Display alongside game view */
}
```

### 8.3 HTTP API

```
POST /v1/xspore/generate
{
    "prompt": "A crystal cave with bioluminescent mushrooms",
    "width": 1920,
    "height": 1080,
    "format": "png"
}
→ PNG image

POST /v1/xspore/scene
{
    "prompt": "A crystal cave with bioluminescent mushrooms"
}
→ XSpore JSON (scene description, can be edited and re-rendered)

POST /v1/xspore/render
{
    "scene": { ... XSpore JSON ... },
    "camera": { "position": [0, 2, 5], "target": [0, 0, 0], "fov": 60 },
    "width": 1920,
    "height": 1080,
    "format": "png"
}
→ PNG image
```

---

## 9. File Format: .xspore

The `.xspore` file format is simply JSON with the schema defined in Section 3. It can also be embedded in other formats:

```
*.xspore        — Scene description (JSON)
*.xspore.gz     — Compressed scene description
*.xsplat        — Pre-compiled scene (binary Gaussians + metadata)
```

### 9.1 Pre-compiled Format (.xsplat)

For scenes that don't change, pre-compile to binary for instant loading:

```
Header (32 bytes):
  magic:     "XSPT" (4 bytes)
  version:   uint32 (1)
  n_gauss:   uint32
  sh_degree: uint32
  bbox_min:  float[3]
  bbox_max:  float[3]

Body:
  gaussian_splat[n_gauss]  — packed, same as DESIGN-01 struct
```

---

## 10. Implementation Plan

### Phase 1: Procedural Primitive Engine (Week 1)
- [ ] `gs_procedural.c` — sphere, cube, plane, cylinder generators
- [ ] Fibonacci sphere sampling, surface-aligned Gaussians
- [ ] Material → SH coefficient mapping
- [ ] Simple JSON parser for XSpore format
- [ ] CLI: `--xspore "sphere at 0,0,0 radius 1 color red"` (structured text, not free-form yet)
- [ ] Output: .ply file (compatible with any GS viewer)
- [ ] Test: generate sphere, render with 3DGS.cpp viewer

### Phase 2: LLM Integration (Week 2)
- [ ] System prompt engineering for Qwen 3.5
- [ ] HTTP client in Artifact Engine (call Victus for inference)
- [ ] Response parser: extract JSON from LLM output (handle markdown code blocks, preamble)
- [ ] Validation: check schema, clamp values, warn on invalid primitives
- [ ] CLI: `--xspore "a red sphere"` (free-form, goes through LLM)
- [ ] Test: 10 diverse prompts → validate all produce renderable scenes

### Phase 3: Xbox Integration (Week 3)
- [ ] Wire XSpore pipeline into Artifact Engine's main loop
- [ ] Display pipeline (either swap chain or frame-to-texture)
- [ ] Gamepad controls: navigate camera, trigger new generations
- [ ] On-screen text overlay for prompt input
- [ ] Test: full loop on Xbox — voice prompt (future) → LLM → render → display

### Phase 4: Interactive + Voice (Week 4+)
- [ ] Scene editing: "make it bluer", "add more crystals" → LLM modifies existing scene
- [ ] Voice input via Xbox microphone → STT → XSpore prompt
- [ ] Scene history: undo/redo, save/load
- [ ] 2D Gaussian mode for Cthulu market visualization
- [ ] Performance optimization: cache LLM responses, incremental scene updates

---

## 11. Dependencies

| Dependency | Status | Blocker? |
|-----------|--------|----------|
| DESIGN-01 GS Renderer | Blueprint done | Yes — need at least Phase 1 MVP |
| Qwen 3.5 9B on Victus | ✅ Downloaded | No |
| Artifact Engine HTTP server | ✅ Working | No |
| Xbox APPX deploy pipeline | ✅ Working | No |
| Victus HTTP server for inference | Needs setup | Minor — just run engine on Victus |
| JSON parser (C) | Not started | No — simple to write, ~200 LOC |
| Xbox display pipeline | Not started | Phase 3 only |

---

## 12. Risks

| Risk | Impact | Mitigation |
|------|--------|-----------|
| LLM outputs invalid JSON | Scene fails to compile | Robust parser with fallbacks; retry with simplified prompt |
| LLM outputs boring/flat scenes | Underwhelming visuals | Curate system prompt library; add post-processing modifiers |
| Procedural Gaussians look artificial | Obviously computer-generated | Add noise, variation, imperfection to generators |
| Xbox memory pressure (LLM + GS) | OOM crashes | Budget enforcer; use SPLAT format (7× smaller); reduce density |
| Latency too high for interactive | Poor UX | Cache common scenes; pre-generate primitives; stream incrementally |

---

## 13. Future Directions

1. **WYRM Integration** — When the WYRM Gaussian specialist head (DESIGN-02) is trained, bypass the LLM prompt pipeline entirely. WYRM generates Gaussian parameters directly from its latent space. XSpore becomes a thin rendering wrapper.

2. **Physics** — Add rigid body physics to Gaussians. Objects fall, collide, bounce. Gaussian positions update per physics tick.

3. **Animation** — Time-varying scenes. Extend XSpore JSON with keyframes: `"animation": [{"t": 0, "position": [0,0,0]}, {"t": 1, "position": [0,2,0]}]`. Interpolate Gaussian parameters over time → 4DGS.

4. **Multi-user** — Multiple Xbox clients viewing the same GS scene from different cameras. Scene state on Victus, streaming sorted Gaussians per-client.

5. **Capture → Edit** — Use phone/COLMAP to capture a real scene as GS, load into XSpore, let LLM edit it: "remove the table", "change wall color to blue".

---

*"A spore takes a seed and grows it into a world. XSpore takes words and grows them into light."*
