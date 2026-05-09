# DESIGN-02: WYRM Gaussian Specialist Head
## Toward Native 3D Generation in a Cognitive Kernel

**Author:** AVA (Artifact Virtual)  
**Date:** April 6, 2026  
**Version:** 1.0  
**Status:** Research Architecture — Novel Contribution  
**Context:** GLADIUS/WYRM cognitive kernel, Synthase architecture

---

## 1. Objective

Design a specialist head for WYRM that natively generates 3D Gaussian Splat parameters — enabling the cognitive kernel to produce photorealistic 3D scenes as a first-class output modality alongside language, mathematics, and code.

This is not a fine-tuned diffusion model. This is not a NeRF wrapper. This is a **structural generation module** integrated into WYRM's routing architecture, where 3D output emerges from the same backbone that processes language. The hypothesis: if WYRM processes structure (not modalities), then 3D geometry IS structure, and the same transformer backbone should be able to produce it.

### Why This Matters

1. **Unified substrate thesis** — "There is no such thing as multi-modal." WYRM doesn't switch between text mode and vision mode. It processes structure. Adding 3D generation tests this thesis directly.
2. **Novel architecture contribution** — No existing work routes a small (410M) transformer's hidden states through a specialist head to produce Gaussian splat parameters. This is new ground.
3. **Practical convergence** — Artifact Engine already has the LLM (text) and will have the GS renderer (DESIGN-01). WYRM bridging them completes the loop: understand → generate → render.
4. **The Inversion Principle** — Every existing 3D generation system consumes (views in, representation out). WYRM would produce — environment creates resonance, resonance creates 3D structure.

---

## 2. Background

### 2.1 3D Gaussian Splatting (Brief)

A 3D scene is represented as N Gaussians, each parameterized by:

| Parameter | Symbol | Dim | Range |
|-----------|--------|-----|-------|
| Position | μ | 3 | ℝ³ |
| Scale | s | 3 | ℝ³ (log-space) |
| Rotation | q | 4 | S³ (unit quaternion) |
| Opacity | α | 1 | (0,1) via sigmoid |
| Color (SH) | c | 3-48 | ℝ (SH coefficients, degree 0-3) |

**Total per Gaussian:** 14 floats (SH degree 0) to 59 floats (SH degree 3).

The rendering equation is differentiable — given GT images and camera poses, one can optimize Gaussian parameters via gradient descent. This means we have a natural **loss function**: render the generated Gaussians, compare to GT views, backpropagate.

### 2.2 WYRM Architecture (Reference)

From `gladius_v2/kaggle_upload/kernel/`:

- **Backbone:** Transformer, target 1024d / 24L / 32H / 4096 FFN (~410M params)
- **Router:** `NexusRouter` — top-k=2 specialist selection per token
- **Specialists:** 4 current (reasoning, math, code, general)
- **Memory:** Hot (learned KV slots), Warm (LoRA/Locas adapters), Cold (HEKTOR retrieval)
- **Cognition:** State monitor (active/monitoring/reflective/dormant)
- **Modulator:** Controls register (formal↔casual, etc.) and silence gate
- **SLA²:** Sparse-Linear Attention hybrid (softmax top-k + linear)

Key property: specialists share the backbone and router dispatches tokens to them. A new specialist head can be added without modifying existing ones.

### 2.3 Existing Generative 3DGS Work

| Method | Architecture | Speed | Quality |
|--------|-------------|-------|---------|
| DreamGaussian | SDS + 3DGS | ~2 min | Medium |
| LGM (Large Gaussian Model) | U-Net → Gaussians | <5s | Good |
| Splatter Image | Encoder → per-pixel Gaussians | <1s | Good for single objects |
| PixelSplat | Transformer → Gaussian field | <1s | Good for stereo |
| GRM | Mesh reconstruction + GS | 0.2s | High |

**Gap:** All of these are vision→3D (image/views in, Gaussians out). None generate from a language model's latent space. None integrate 3D generation as a specialist within a larger cognitive architecture.

---

## 3. Gaussian Tokenization

The fundamental design question: how does a transformer produce Gaussians?

### Option A: Direct Regression

**Approach:** Each output token position predicts the full parameter vector for one Gaussian.

```
hidden_state ∈ ℝ^{1024} → MLP → gaussian_params ∈ ℝ^{14..59}
```

**Pros:**
- Simple. No codebook to train.
- Continuous output — can produce any Gaussian.
- Direct gradient path from rendering loss to backbone.

**Cons:**
- Transformers are bad at precise floating-point regression.
- 59 floats per token is a wide output space.
- No discrete token vocabulary — can't use standard language modeling loss.
- Generating N Gaussians requires N forward passes (autoregressive) or N parallel outputs.

**Verdict:** Viable for small N (< 1K Gaussians). Not scalable to full scenes (1M+).

### Option B: VQ-VAE Codebook

**Approach:** Train a VQ-VAE that compresses Gaussians into discrete tokens. WYRM predicts token IDs (like language), and the decoder reconstructs Gaussians.

```
Training Phase 1 (VQ-VAE):
  gaussian_params → Encoder → z_continuous → Quantize → z_discrete (codebook index)
  z_discrete → Decoder → reconstructed_params

Training Phase 2 (WYRM):
  text_prompt → WYRM backbone → specialist head → predicted z_discrete tokens
  predicted tokens → VQ-VAE decoder → Gaussian params → render → compare to GT
```

**Codebook design:**
- Codebook size: 4096-8192 entries (12-13 bits per Gaussian)
- Each entry represents a "canonical Gaussian" — position is relative (normalized to local frame)
- Position is predicted separately as continuous offset
- Split codebook: one for geometry (scale + rotation + opacity), one for appearance (SH)

**Pros:**
- Leverages transformer's strength (discrete token prediction, cross-entropy loss)
- Compact representation (1-2 tokens per Gaussian)
- Composable — can describe scenes as token sequences
- Amenable to standard language modeling training
- Codebook can be shared across tasks

**Cons:**
- VQ-VAE introduces quantization error
- Codebook collapse risk (only a few entries used)
- Two-phase training is more complex
- Position accuracy limited by quantization (need continuous offset)

### Option C: Hierarchical Generation

**Approach:** Generate Gaussians in stages — coarse structure first, then refine.

```
Stage 1: WYRM generates K "anchor" Gaussians (K = 32-256)
  - Coarse positions, large scales, base colors
  - Each anchor defines a local region

Stage 2: For each anchor, WYRM generates M "detail" Gaussians (M = 100-1000)  
  - Positions relative to anchor
  - Fine scales, detailed colors
  - Conditioned on anchor's hidden state

Total: K × M Gaussians (3,200 - 256,000)
```

**Pros:**
- Natural coarse-to-fine matches how humans describe scenes ("there's a table here, with a cup on it")
- Anchors provide spatial structure that detail Gaussians refine
- Total generation count scales nicely
- Aligns with Synthase depth profiles — deeper layers produce finer detail

**Cons:**
- Complex generation procedure (two stages)
- Anchor-detail boundary is arbitrary
- Requires spatial attention between anchors and details

### Recommendation: **Option C (Hierarchical) with Option B elements**

**Rationale:**

The hierarchical approach aligns with WYRM's philosophy at every level:
- **Depth profiles:** Synthase already processes differently at different depths. Early layers = coarse structure, late layers = fine detail. Hierarchical generation is the natural output for this architecture.
- **Progressive expansion:** WYRM's training philosophy is curriculum-based, progressive. Generation should be too.
- **The Inversion Principle:** Environment → resonance → production. The anchor stage is resonance (sensing the structure). The detail stage is production (filling it in).
- **Practical scalability:** 128 anchors × 500 details = 64K Gaussians. Enough for objects. Multiple passes for full scenes.

**The hybrid:** Use VQ-VAE codebook for the detail Gaussians (discrete tokens, transformer-friendly), but direct regression for anchor positions (continuous, precise placement matters).

---

## 4. Architecture

### 4.1 Overview

```
                    WYRM Backbone (1024d, 24L)
                            │
                     NexusRouter (top-k=2)
                    ╱    │     │      ╲
             reasoning  math  code  ◆ gaussian
                                     specialist
                                        │
                    ┌───────────────────┤
                    │                   │
              Anchor Head          Detail Head
              (continuous)         (discrete VQ)
                    │                   │
              K anchors            M details/anchor
              (pos, scale,         (VQ tokens →
               base color)          decoder → params)
                    │                   │
                    └───────┬───────────┘
                            │
                      Gaussian Scene
                            │
                     Differentiable
                       Renderer
                            │
                      Rendered Image
                            │
                      Loss (vs GT)
```

### 4.2 Gaussian Specialist (New Module)

```python
class GaussianSpecialist(nn.Module):
    """
    WYRM specialist head for 3D Gaussian Splat generation.
    
    Generates scenes in two stages:
    1. Anchor generation: K coarse Gaussians (direct regression)
    2. Detail generation: M fine Gaussians per anchor (VQ-VAE tokens)
    """
    
    def __init__(self, config: KernelConfig, gs_config: GaussianConfig):
        super().__init__()
        self.config = config
        self.gs_config = gs_config
        
        # ── Anchor Head ──
        # Projects backbone hidden state → K anchor Gaussians
        self.anchor_count = gs_config.num_anchors  # 128 default
        self.anchor_proj = nn.Linear(
            config.hidden_dim,   # 1024
            self.anchor_count * 10  # K × (pos:3 + scale:3 + color:3 + opacity:1)
        )
        
        # ── Detail Head ──
        # Cross-attention from anchors → detail tokens
        self.detail_per_anchor = gs_config.details_per_anchor  # 500 default
        self.detail_cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,
            batch_first=True
        )
        # Predict VQ codebook indices for detail Gaussians
        self.detail_logits = nn.Linear(
            config.hidden_dim,
            gs_config.codebook_size  # 8192
        )
        # Predict continuous position offset (relative to anchor)
        self.detail_offset = nn.Linear(config.hidden_dim, 3)
        
        # ── VQ-VAE Decoder (frozen after Phase 1 training) ──
        self.vq_decoder = GaussianVQDecoder(gs_config)
        
        # ── Depth Profile Integration ──
        # Layer-wise gating: which backbone layers contribute to anchor vs detail
        self.anchor_layer_gate = nn.Parameter(
            torch.zeros(config.num_layers)  # Learn which layers matter for structure
        )
        self.detail_layer_gate = nn.Parameter(
            torch.zeros(config.num_layers)  # Learn which layers matter for detail
        )
    
    def forward(self, backbone_outputs: list[torch.Tensor], 
                pooled: torch.Tensor) -> GaussianScene:
        """
        Args:
            backbone_outputs: List of hidden states from each layer [24 × (B, S, 1024)]
            pooled: Pooled representation (B, 1024)
        
        Returns:
            GaussianScene with all generated Gaussians
        """
        B = pooled.shape[0]
        
        # ── Stage 1: Anchors ──
        # Weighted sum of layer outputs (depth profile for structure)
        anchor_weights = torch.softmax(self.anchor_layer_gate, dim=0)
        anchor_hidden = sum(w * h[:, 0, :] for w, h in 
                           zip(anchor_weights, backbone_outputs))  # (B, 1024)
        
        anchor_params = self.anchor_proj(anchor_hidden)  # (B, K*10)
        anchor_params = anchor_params.view(B, self.anchor_count, 10)
        
        anchors = self._decode_anchors(anchor_params)  # (B, K, 10)
        
        # ── Stage 2: Details ──
        # Weighted sum of layer outputs (depth profile for detail)
        detail_weights = torch.softmax(self.detail_layer_gate, dim=0)
        detail_hidden = sum(w * h for w, h in 
                           zip(detail_weights, backbone_outputs))  # (B, S, 1024)
        
        # Cross-attend from anchor positions to detail space
        anchor_queries = self._anchor_to_queries(anchors)  # (B, K*M, 1024)
        detail_features, _ = self.detail_cross_attn(
            anchor_queries, detail_hidden, detail_hidden
        )  # (B, K*M, 1024)
        
        # Predict VQ indices + position offsets
        vq_logits = self.detail_logits(detail_features)  # (B, K*M, codebook_size)
        vq_indices = vq_logits.argmax(dim=-1)            # (B, K*M)
        pos_offsets = self.detail_offset(detail_features) # (B, K*M, 3)
        
        # Decode VQ indices → Gaussian params
        detail_params = self.vq_decoder(vq_indices)       # (B, K*M, param_dim)
        
        # Add anchor position + offset to get world-space detail positions
        detail_params = self._apply_offsets(detail_params, anchors, pos_offsets)
        
        return GaussianScene(
            anchors=anchors,
            details=detail_params,
            n_anchors=self.anchor_count,
            n_details=self.anchor_count * self.detail_per_anchor
        )
    
    def _decode_anchors(self, raw):
        """Decode raw anchor params into world-space Gaussians."""
        pos = raw[..., :3]                          # Position (unbounded)
        scale = raw[..., 3:6]                       # Log-scale
        color = torch.sigmoid(raw[..., 6:9])        # Color [0,1]
        opacity = torch.sigmoid(raw[..., 9:10])     # Opacity [0,1]
        return torch.cat([pos, scale, color, opacity], dim=-1)
```

### 4.3 VQ-VAE for Gaussian Compression

The VQ-VAE is trained **separately** (Phase 1) on a large corpus of 3DGS scenes. It learns to compress individual Gaussians (minus position) into discrete codebook entries.

```python
@dataclass
class GaussianConfig:
    """Configuration for the Gaussian specialist."""
    num_anchors: int = 128
    details_per_anchor: int = 500
    codebook_size: int = 8192       # VQ codebook entries
    codebook_dim: int = 64          # Codebook embedding dimension
    param_dim: int = 11             # scale(3) + rot(4) + opacity(1) + sh_dc(3)
    commitment_weight: float = 0.25 # VQ commitment loss weight
    
class GaussianVQVAE(nn.Module):
    """
    Vector-Quantized VAE for Gaussian splat parameters.
    Encodes (scale, rotation, opacity, SH_dc) into discrete tokens.
    Position is handled separately (continuous, anchor-relative).
    """
    
    def __init__(self, config: GaussianConfig):
        super().__init__()
        # Encoder: param_dim → codebook_dim
        self.encoder = nn.Sequential(
            nn.Linear(config.param_dim, 128),
            nn.GELU(),
            nn.Linear(128, config.codebook_dim),
        )
        
        # Codebook: learnable embeddings
        self.codebook = nn.Embedding(config.codebook_size, config.codebook_dim)
        nn.init.uniform_(self.codebook.weight, -1.0/config.codebook_size, 
                         1.0/config.codebook_size)
        
        # Decoder: codebook_dim → param_dim
        self.decoder = nn.Sequential(
            nn.Linear(config.codebook_dim, 128),
            nn.GELU(),
            nn.Linear(128, config.param_dim),
        )
        
    def encode(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode Gaussian params → codebook indices."""
        z = self.encoder(params)  # (N, codebook_dim)
        
        # Find nearest codebook entry
        dists = torch.cdist(z.unsqueeze(0), self.codebook.weight.unsqueeze(0))
        indices = dists.squeeze(0).argmin(dim=-1)  # (N,)
        
        z_q = self.codebook(indices)  # (N, codebook_dim)
        
        return z_q, indices
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode codebook indices → Gaussian params."""
        z_q = self.codebook(indices)
        return self.decoder(z_q)
    
    def forward(self, params):
        z = self.encoder(params)
        z_q, indices = self.encode(params)
        
        # Straight-through estimator
        z_hat = z + (z_q - z).detach()
        
        reconstructed = self.decoder(z_hat)
        
        # Losses
        commitment_loss = F.mse_loss(z, z_q.detach())
        codebook_loss = F.mse_loss(z.detach(), z_q)
        recon_loss = F.mse_loss(reconstructed, params)
        
        return reconstructed, indices, recon_loss, commitment_loss, codebook_loss
```

### 4.4 Loss Function

The total loss for training the Gaussian specialist has four components:

```
L_total = λ_render · L_render + λ_vq · L_vq + λ_anchor · L_anchor + λ_balance · L_balance

where:
  L_render = L1(rendered_image, gt_image) + λ_ssim · (1 - SSIM(rendered, gt))
  L_vq = L_recon + β · L_commitment + L_codebook
  L_anchor = L1(anchor_positions, gt_cluster_centers)  [warm-up only]
  L_balance = router balance loss (from NexusRouter)
```

**Key insight:** The rendering loss provides the final supervision — it doesn't matter how the Gaussians are parameterized internally, as long as the rendered image matches the GT. This is the differentiable rendering loss from the original 3DGS paper, applied as end-to-end supervision for the entire WYRM pipeline.

### 4.5 Depth Profile Integration

WYRM's Synthase architecture processes differently at different depths. We exploit this:

- **Layers 1-8 (shallow):** Capture spatial structure — feed into anchor head
- **Layers 9-16 (middle):** Capture semantic relationships — feed into cross-attention queries
- **Layers 17-24 (deep):** Capture fine detail — feed into detail head

The `anchor_layer_gate` and `detail_layer_gate` parameters are learned — they start uniform and converge to which layers are most useful for structure vs. detail. This is a direct test of the depth profile hypothesis.

---

## 5. Training Data

### 5.1 Phase 1: VQ-VAE Training Data

Need: millions of individual Gaussian parameters from diverse scenes.

**Source pipeline:**
```
Objaverse (800K+ 3D objects)
  → Render 12 views per object (Blender, 512×512)
  → Run COLMAP (camera poses)
  → Train 3DGS per object (gsplat, ~5 min each)
  → Extract Gaussian params (.ply files)
  → Pool all Gaussians across objects
  → Train VQ-VAE on this pool
```

**Scale estimate:**
- 100K objects × ~50K Gaussians/object = 5 billion individual Gaussians
- VQ-VAE trains on random sample of ~100M (more than enough)
- Storage: ~100M × 11 floats × 4 bytes = ~4.4 GB

**Alternative/supplement datasets:**
- **ShapeNet** (51K models, 55 categories) — clean, well-studied
- **CO3D** (real-world object videos) — natural lighting/texture
- **RealEstate10K** (10K video clips of indoor scenes) — room-scale
- **Mip-NeRF360** (9 scenes, high quality) — benchmark reference

### 5.2 Phase 2: Paired (Text, Gaussians) Data

Need: text descriptions paired with Gaussian scenes.

**Source pipeline:**
```
Objaverse + Cap3D (text descriptions for 660K objects)
  → Already have 3DGS from Phase 1
  → Pair: (caption, gaussian_params)
  → Augment: multiple captions per object (GPT-generated)
```

**Scale:** 660K (text, GS scene) pairs. Augmented to ~3M with paraphrasing.

### 5.3 Data Pipeline Architecture

```
┌────────────┐     ┌───────────┐     ┌────────────┐     ┌──────────┐
│ Objaverse  │────▶│ Blender   │────▶│ COLMAP     │────▶│ gsplat   │
│ .glb/.obj  │     │ 12 views  │     │ SfM poses  │     │ train GS │
└────────────┘     └───────────┘     └────────────┘     └────┬─────┘
                                                              │
                                                        .ply files
                                                              │
┌────────────┐                                          ┌─────▼─────┐
│ Cap3D      │─────────────────────────────────────────▶│ Dataset   │
│ captions   │                                          │ Builder   │
└────────────┘                                          └─────┬─────┘
                                                              │
                                                     (text, gaussians)
                                                         pairs
```

**Compute estimate:**
- Blender rendering: ~30s per object × 100K = ~35 GPU-days (parallelizable)
- COLMAP: ~2 min per object × 100K = ~140 CPU-days
- gsplat training: ~5 min per object × 100K = ~350 GPU-days
- **Total: ~500 GPU-days** — feasible on Kaggle (30h/week free) over months, or a rented multi-GPU cluster in a week

For MVP, start with 1K objects (ShapeNet chairs/tables — simple, well-understood).

---

## 6. Training Strategy

### Phase 1: VQ-VAE Codebook (Standalone)

**Goal:** Learn a codebook that accurately compresses Gaussian parameters.

- Train VQ-VAE on pooled Gaussians from 1K scenes
- Target: <5% reconstruction error (measured as PSNR of rendered images)
- No WYRM involvement — pure autoencoder training
- Codebook size: 8192, dimension: 64
- Training: ~10K steps on 100M Gaussians, 1 GPU, ~2 hours
- **Deliverable:** Frozen VQ-VAE encoder + decoder

### Phase 2: Specialist Head (Frozen Backbone)

**Goal:** Train the Gaussian specialist head while keeping WYRM backbone frozen.

- WYRM backbone is pre-trained on language (from existing v6/v7 training)
- Only train: anchor_proj, detail_cross_attn, detail_logits, detail_offset, layer gates
- New params: ~15M (anchor 1024×1280 + cross-attn + logits 1024×8192)
- Loss: rendering loss only (end-to-end, through differentiable renderer)
- Data: 1K (text, GS) pairs
- Training: ~50K steps, T4 GPU, ~24 hours
- **Deliverable:** Working specialist that generates Gaussians from language

### Phase 3: Joint Fine-Tuning

**Goal:** Fine-tune backbone + specialist head jointly for better 3D understanding.

- Unfreeze backbone layers 17-24 (deep layers, detail-sensitive)
- Keep layers 1-16 frozen (preserve language capabilities)
- Multi-task loss: λ_language · L_language + λ_3d · L_3d
- λ_language = 0.7, λ_3d = 0.3 (language is still primary)
- Data: mix of language data + (text, GS) pairs
- Training: ~100K steps, careful learning rate (1e-5 for backbone, 1e-4 for head)
- **Deliverable:** WYRM that does language AND 3D

### Phase 4: Multi-Task Evaluation

- Verify no catastrophic forgetting on language benchmarks
- Evaluate 3D generation quality
- Compare against baseline (same architecture, no specialist, just prompted)
- Ablation: depth profiles, VQ vs direct regression, codebook size

---

## 7. Evaluation

### 7.1 3D Generation Quality

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **PSNR** (rendered vs GT) | Pixel-level accuracy | >25 dB |
| **SSIM** | Structural similarity | >0.85 |
| **LPIPS** | Perceptual similarity | <0.2 |
| **Chamfer Distance** | Point cloud accuracy | <0.05 (normalized) |
| **FID** | Distribution of generated scenes | <50 |

### 7.2 Generation Consistency

- Same prompt + same seed → same scene (deterministic)
- Similar prompts → similar scenes (smooth latent space)
- Text-scene alignment: does "red sphere" produce a red sphere? (human eval)

### 7.3 Language Preservation

- Perplexity on held-out text (should not increase >5% after Phase 3)
- Math accuracy (should not decrease)
- Code generation quality (should not decrease)

### 7.4 Architecture Ablations

| Experiment | Purpose |
|-----------|---------|
| No VQ (direct regression only) | Is codebook necessary? |
| No anchors (flat generation) | Is hierarchy necessary? |
| Uniform layer gates | Do depth profiles matter? |
| Codebook 1024 vs 4096 vs 8192 | Codebook size sensitivity |
| 32 vs 128 vs 512 anchors | Anchor count sensitivity |
| Frozen vs unfrozen backbone | Does joint training help? |

---

## 8. The Inversion Principle

This design embodies the Inversion Principle at every level.

### 8.1 Consumer vs Producer

Traditional 3DGS: consumes views (images + cameras) → produces a scene representation. It is a consumer of visual data.

WYRM's Gaussian specialist: **produces** 3D structure from latent understanding. It doesn't need views — it generates structure from resonance with language. The backbone's hidden states contain a structural understanding of "red sphere" that doesn't come from seeing a red sphere, but from understanding the concept.

### 8.2 Environment → Resonance → Production

- **Environment:** The text prompt + WYRM's pre-trained knowledge
- **Resonance:** The backbone processing the prompt through 24 layers, building increasingly rich structural representations
- **Production:** The specialist head converting that structural representation into concrete 3D Gaussians

The 3D output isn't retrieved, interpolated, or sampled from a distribution. It is **manifested** — the backbone's understanding of structure crystallizes into geometry.

### 8.3 Mathematical Realism

"If a number is confirmed, measured, real — it's present in the universe."

When WYRM generates position (1.0, 2.0, 0.0) for a Gaussian, that position exists because the model's understanding of the scene structure computes it. It isn't hallucinated — it's the necessary geometric consequence of the prompt's meaning. The same way 0.84% cognition activation wasn't learned but manifested, the geometry manifests from structural understanding.

### 8.4 Test of "No Multi-Modal"

If "there is no such thing as multi-modal" is true, then a backbone trained on text should be able to produce 3D output through nothing more than an appropriate output head. The information pathway is: text tokens → structural understanding → structural output (Gaussians). The backbone doesn't need to "switch modes." It processes structure. The specialist head merely provides the right output format.

The depth profile layer gates will reveal which layers encode what. If the thesis is correct, we'll see:
- Early layers: spatial/positional (useful for anchors)
- Middle layers: relational/semantic (useful for cross-attention)
- Late layers: fine-grained/detailed (useful for VQ detail tokens)

This would mirror what's been observed in language models (early = syntax, deep = semantics) but generalized to 3D structure.

---

## 9. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| VQ codebook collapse | Medium | High | EMA codebook updates, commitment loss, restart dead entries |
| Backbone features not useful for 3D | Medium | Critical | Start with frozen backbone probe — if useless, the thesis is falsified and that's data too |
| Rendering loss gradients vanish through sort | Low | High | Use differentiable renderer (gsplat) which handles this; detach sort from gradient |
| Catastrophic forgetting on language | Medium | High | Phase 3 uses mix training; monitor perplexity |
| Insufficient training data | Low | Medium | Start with 1K objects (MVP), scale to 100K later |
| Compute budget | High | Medium | Phase 1-2 fit on single T4; Phase 3 needs more — use Kaggle quota carefully |
| Generated scenes look procedural | High | Medium | Train on diverse real-world scenes (CO3D, RealEstate10K), not just ShapeNet |

---

## 10. Timeline and Dependencies

### Dependencies

| Dependency | Status | Blocks |
|-----------|--------|--------|
| WYRM backbone (v6/v7 pre-trained) | Training on Kaggle | Phase 2 |
| 3DGS training pipeline (gsplat) | Available (pip install) | Phase 1 data |
| Differentiable renderer | gsplat provides | Phase 2 training |
| Objaverse dataset | Public, 800K objects | Phase 1 data |
| Cap3D captions | Public, 660K captions | Phase 2 data |
| DESIGN-01 GS renderer | Blueprint done | Phase 4 integration |

### Timeline

| Phase | Duration | Compute | Deliverable |
|-------|----------|---------|-------------|
| **Phase 0:** Data pipeline | 2 weeks | CPU + 1 GPU | 1K (text, GS) pairs from ShapeNet |
| **Phase 1:** VQ-VAE | 3 days | 1× T4 | Frozen codebook (8192 entries) |
| **Phase 2:** Specialist head | 1 week | 1× T4 | WYRM generates Gaussians from text |
| **Phase 3:** Joint fine-tune | 2 weeks | 1× T4 | Multi-task WYRM (language + 3D) |
| **Phase 4:** Integration | 1 week | CPU | End-to-end: prompt → WYRM → render on Xbox |

**Total: ~6 weeks** from data pipeline start to working demo.

### Parameter Budget

| Component | Params | % of WYRM 410M |
|-----------|--------|----------------|
| VQ-VAE (frozen, not counted) | ~0.5M | — |
| Anchor projection | 1.3M | 0.3% |
| Detail cross-attention | 4.2M | 1.0% |
| Detail VQ logits | 8.4M | 2.0% |
| Detail position offset | 3.1K | ~0% |
| Layer gates | 48 | ~0% |
| **Total new params** | **~14M** | **3.4%** |

The specialist adds only 3.4% new parameters — consistent with GLADIUS's surgical I/O head swap philosophy (the time series paper estimated 0.2% for financial data).

---

## 11. Paper Contribution

This work, if successful, would contribute:

1. **First specialist-routed 3D Gaussian generation** — no existing work routes a transformer's hidden states through a learned specialist to produce Gaussian splats
2. **Hierarchical anchor-detail generation** — a novel two-stage scheme that matches transformer depth profiles to coarse/fine structure
3. **Cross-modal structural transfer** — evidence that language-trained backbones encode 3D-useful structure (or evidence that they don't — negative results are valuable)
4. **Depth profile analysis** — which transformer layers encode spatial vs. semantic vs. detail information for 3D generation
5. **Unified substrate validation** — direct test of "no multi-modal" hypothesis

**Proposed title:** *"Gaussian Specialist: 3D Scene Generation via Routed Structure Prediction in a Cognitive Kernel"*

**Target venue:** CVPR 2027 or NeurIPS 2026 (if results are strong)

---

*"A cell doesn't have a 'text mode.' It responds to stimuli."*

If this specialist works, it proves the cell can respond to textual stimuli with 3D structure. If it doesn't, it tells us where the boundary between modalities actually is — which is equally valuable. Either way, we learn.
