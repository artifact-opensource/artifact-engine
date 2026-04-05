/*
 * Artifact Engine — Transformer Forward Pass (Hybrid Architecture)
 *
 * Supports five modes:
 *   1. Pure attention (classic transformers: llama, qwen2, etc.)
 *   2. Pure DeltaNet (all linear attention)
 *   3. Hybrid DeltaNet+Attention (Qwen3.5-style: every Nth layer is full attention)
 *   4. Pure Mamba (S6) / Pure Mamba-2 (SSD)
 *   5. Hybrid Mamba+Attention (Jamba, Falcon-H1, Samba-style)
 *
 * The layer_type per layer controls dispatch. Backward compatible with
 * existing pure-attention models.
 */

#define _GNU_SOURCE
#include "../include/engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
  #include <windows.h>
#endif

/* ───── Engine Init ───── */

bool engine_init(engine* eng, const char* shader_dir) {
    memset(eng, 0, sizeof(*eng));
    return vk_init(&eng->vk, shader_dir);
}

/* ───── Upload tensor from GGUF to GPU ───── */

static bool upload_tensor(engine* eng, gpu_buffer* dst, 
                          const char* name, bool required) {
    const gguf_tensor_info* ti = gguf_find_tensor(eng->gguf, name);
    if (!ti) {
        if (required) {
            fprintf(stderr, "engine: required tensor '%s' not found\n", name);
        }
        return !required;
    }
    
    const void* data = gguf_tensor_data(eng->gguf, ti);
    
#ifdef CPU_ONLY
    /* CPU mode: point directly to mmap'd GGUF data — zero-copy */
    memset(dst, 0, sizeof(*dst));
    dst->mapped = (void*)data;  /* const-cast: CPU ops read-only for weights */
    dst->size = ti->n_bytes;
#else
    if (!vk_alloc_device(&eng->vk, dst, ti->n_bytes)) {
        fprintf(stderr, "engine: failed to allocate GPU buffer for '%s' (%zu bytes)\n",
                name, (size_t)ti->n_bytes);
        return false;
    }
    
    if (!vk_upload(&eng->vk, dst, data, ti->n_bytes)) {
        fprintf(stderr, "engine: failed to upload '%s' to GPU\n", name);
        return false;
    }
#endif
    
    /* Store tensor type metadata for quantization-aware dispatch */
    dst->dtype = (uint32_t)ti->type;
    dst->n_rows = (ti->n_dims >= 2) ? (uint32_t)ti->dims[1] : 1;
    dst->n_cols = (uint32_t)ti->dims[0];
    
    return true;
}

/* ───── Determine layer type for hybrid models ───── */

static layer_type get_layer_type_deltanet(const model_arch* arch, uint32_t layer_idx) {
    if (arch->full_attn_interval == 0) {
        return LAYER_ATTENTION;  /* pure attention model */
    }
    /* Qwen3.5 pattern: layers (interval-1), (2*interval-1), ... are attention */
    if (((layer_idx + 1) % arch->full_attn_interval) == 0) {
        return LAYER_ATTENTION;
    }
    return LAYER_DELTANET;
}

/* Check if a specific layer has Mamba/SSM weights by probing tensor names */
static bool layer_has_ssm_in(const gguf_file* gf, uint32_t layer_idx) {
    char name[128];
    snprintf(name, sizeof(name), "blk.%u.ssm_in.weight", layer_idx);
    return gguf_find_tensor(gf, name) != NULL;
}

/* ───── Load Model ───── */

bool engine_load_model(engine* eng, const char* model_path) {
    printf("engine: loading %s\n", model_path);
    
    /* Load GGUF */
    eng->gguf = gguf_load(model_path);
    if (!eng->gguf) return false;
    
    gguf_print_info(eng->gguf);
    
    /* Extract architecture */
    if (!gguf_extract_arch(eng->gguf, &eng->arch)) return false;
    
    /* Detect model type and determine layer types */
    bool is_qwen35  = (strcmp(eng->arch.arch, "qwen35") == 0);
    bool is_mamba   = (strcmp(eng->arch.arch, "mamba") == 0);
    bool is_mamba2  = (strcmp(eng->arch.arch, "mamba2") == 0);
    bool is_jamba   = (strcmp(eng->arch.arch, "jamba") == 0);
    bool is_falcon  = (strcmp(eng->arch.arch, "falcon_h1") == 0);
    bool is_hybrid_deltanet = (is_qwen35 && eng->arch.full_attn_interval > 0 && eng->arch.ssm_d_inner > 0);
    bool is_hybrid_mamba = (is_jamba || is_falcon);
    bool is_hybrid = (is_hybrid_deltanet || is_hybrid_mamba || is_mamba || is_mamba2);
    
    eng->weights.is_hybrid = is_hybrid;
    
    if (is_mamba) {
        printf("engine: MAMBA (S6) model — all %u layers are Mamba\n", eng->arch.n_layers);
    } else if (is_mamba2) {
        printf("engine: MAMBA-2 (SSD) model — all %u layers are Mamba-2\n", eng->arch.n_layers);
    } else if (is_hybrid_mamba) {
        printf("engine: HYBRID %s model — mixed Attention + Mamba layers\n", eng->arch.arch);
    } else if (is_hybrid_deltanet) {
        printf("engine: HYBRID model detected — interval=%u, %u DeltaNet + %u attention layers\n",
               eng->arch.full_attn_interval,
               eng->arch.n_layers - eng->arch.n_layers / eng->arch.full_attn_interval,
               eng->arch.n_layers / eng->arch.full_attn_interval);
    }
    
    printf("engine: uploading %u layers to GPU...\n", eng->arch.n_layers);
    
    /* Allocate layer array */
    eng->weights.n_layers = eng->arch.n_layers;
    eng->weights.layers = calloc(eng->arch.n_layers, sizeof(hybrid_layer));
    if (!eng->weights.layers) return false;
    
    /* Upload token embeddings */
    if (!upload_tensor(eng, &eng->weights.token_embd, "token_embd.weight", true))
        return false;
    
    /* Upload per-layer weights */
    uint32_t attn_cache_idx = 0;
    uint32_t recurrent_cache_idx = 0;
    
    for (uint32_t l = 0; l < eng->arch.n_layers; l++) {
        char name[128];
        hybrid_layer* layer = &eng->weights.layers[l];
        
        /* Determine layer type */
        if (is_mamba) {
            layer->type = LAYER_MAMBA;
        } else if (is_mamba2) {
            layer->type = LAYER_MAMBA2;
        } else if (is_hybrid_mamba) {
            /* Jamba/Falcon-H1: probe for ssm_in tensor to detect Mamba layers */
            if (layer_has_ssm_in(eng->gguf, l)) {
                /* Detect Mamba1 vs Mamba2 by checking for ssm_x tensor (Mamba1 only) */
                char probe[128];
                snprintf(probe, sizeof(probe), "blk.%u.ssm_x.weight", l);
                if (gguf_find_tensor(eng->gguf, probe)) {
                    layer->type = LAYER_MAMBA;
                } else {
                    layer->type = LAYER_MAMBA2;
                }
            } else {
                layer->type = LAYER_ATTENTION;
            }
        } else if (is_hybrid_deltanet) {
            layer->type = get_layer_type_deltanet(&eng->arch, l);
        } else {
            layer->type = LAYER_ATTENTION;
        }
        
        #define UPLOAD_LAYER(field, tensor_name, req) \
            snprintf(name, sizeof(name), "blk.%u.%s", l, tensor_name); \
            if (!upload_tensor(eng, &layer->field, name, req)) { if (req) return false; }
        
        /* Common: attention norm (all layers have this) */
        UPLOAD_LAYER(attn_norm, "attn_norm.weight", true);
        
        /* FFN: Mamba/Mamba2 pure models may not have FFN.
         * Hybrid models (Jamba) always have FFN on all layers.
         * Try loading FFN — not required for pure Mamba. */
        bool has_ffn = false;
        {
            snprintf(name, sizeof(name), "blk.%u.ffn_gate.weight", l);
            has_ffn = (gguf_find_tensor(eng->gguf, name) != NULL);
        }
        
        if (has_ffn) {
            /* FFN norm: try post_attention_norm first (qwen35), fallback to ffn_norm */
            snprintf(name, sizeof(name), "blk.%u.post_attention_norm.weight", l);
            if (gguf_find_tensor(eng->gguf, name)) {
                UPLOAD_LAYER(post_attn_norm, "post_attention_norm.weight", true);
            } else {
                UPLOAD_LAYER(ffn_norm, "ffn_norm.weight", true);
            }
            
            UPLOAD_LAYER(ffn_gate, "ffn_gate.weight", true);
            UPLOAD_LAYER(ffn_up,   "ffn_up.weight",   true);
            UPLOAD_LAYER(ffn_down, "ffn_down.weight",  true);
        }
        
        if (layer->type == LAYER_ATTENTION) {
            /* ── Attention layer weights ── */
            layer->cache_index = attn_cache_idx++;
            
            UPLOAD_LAYER(wq, "attn_q.weight", true);
            UPLOAD_LAYER(wk, "attn_k.weight", true);
            UPLOAD_LAYER(wv, "attn_v.weight", true);
            UPLOAD_LAYER(wo, "attn_output.weight", true);
            
            /* Optional per-head norms (qwen35 attention layers) */
            UPLOAD_LAYER(attn_q_norm, "attn_q_norm.weight", false);
            UPLOAD_LAYER(attn_k_norm, "attn_k_norm.weight", false);
            
        } else if (layer->type == LAYER_DELTANET) {
            /* ── DeltaNet layer weights ── */
            layer->cache_index = recurrent_cache_idx++;
            
            UPLOAD_LAYER(wqkv,      "attn_qkv.weight",    true);
            UPLOAD_LAYER(wqkv_gate, "attn_gate.weight",    true);
            UPLOAD_LAYER(ssm_alpha, "ssm_alpha.weight",    true);
            UPLOAD_LAYER(ssm_beta,  "ssm_beta.weight",     true);
            UPLOAD_LAYER(ssm_conv1d,"ssm_conv1d.weight",   true);
            UPLOAD_LAYER(ssm_norm,  "ssm_norm.weight",     true);
            UPLOAD_LAYER(ssm_out,   "ssm_out.weight",      true);
            
            snprintf(name, sizeof(name), "blk.%u.ssm_a", l);
            if (!upload_tensor(eng, &layer->ssm_a, name, true)) return false;
            
            snprintf(name, sizeof(name), "blk.%u.ssm_dt.bias", l);
            if (!upload_tensor(eng, &layer->ssm_dt, name, true)) return false;
            
        } else if (layer->type == LAYER_MAMBA) {
            /* ── Mamba-1 (S6) layer weights ── */
            layer->cache_index = recurrent_cache_idx++;
            
            UPLOAD_LAYER(ssm_in,      "ssm_in.weight",      true);
            UPLOAD_LAYER(ssm_conv1d,  "ssm_conv1d.weight",  true);
            UPLOAD_LAYER(ssm_conv1d_b,"ssm_conv1d.bias",    true);
            UPLOAD_LAYER(ssm_x,       "ssm_x.weight",       true);
            UPLOAD_LAYER(ssm_out,     "ssm_out.weight",      true);
            
            /* ssm_dt has weight and bias as separate tensors */
            snprintf(name, sizeof(name), "blk.%u.ssm_dt.weight", l);
            if (!upload_tensor(eng, &layer->ssm_dt_w, name, true)) return false;
            
            snprintf(name, sizeof(name), "blk.%u.ssm_dt.bias", l);
            if (!upload_tensor(eng, &layer->ssm_dt_b, name, true)) return false;
            
            /* ssm_a and ssm_d have no .weight suffix */
            snprintf(name, sizeof(name), "blk.%u.ssm_a", l);
            if (!upload_tensor(eng, &layer->ssm_a, name, true)) return false;
            
            snprintf(name, sizeof(name), "blk.%u.ssm_d", l);
            if (!upload_tensor(eng, &layer->ssm_d, name, true)) return false;
            
        } else if (layer->type == LAYER_MAMBA2) {
            /* ── Mamba-2 (SSD) layer weights ── */
            layer->cache_index = recurrent_cache_idx++;
            
            UPLOAD_LAYER(ssm_in,      "ssm_in.weight",      true);
            UPLOAD_LAYER(ssm_conv1d,  "ssm_conv1d.weight",  true);
            UPLOAD_LAYER(ssm_conv1d_b,"ssm_conv1d.bias",    true);
            UPLOAD_LAYER(ssm_out,     "ssm_out.weight",      true);
            
            /* ssm_dt.bias for Mamba2 */
            snprintf(name, sizeof(name), "blk.%u.ssm_dt.bias", l);
            if (!upload_tensor(eng, &layer->ssm_dt_b, name, true)) return false;
            
            /* ssm_a [1, n_head] */
            snprintf(name, sizeof(name), "blk.%u.ssm_a", l);
            if (!upload_tensor(eng, &layer->ssm_a, name, true)) return false;
            
            /* ssm_d [1, n_head] */
            snprintf(name, sizeof(name), "blk.%u.ssm_d", l);
            if (!upload_tensor(eng, &layer->ssm_d, name, true)) return false;
            
            /* Per-group RMS norm */
            UPLOAD_LAYER(ssm_norm, "ssm_norm.weight", true);
        }
        
        #undef UPLOAD_LAYER
        
        if ((l + 1) % 4 == 0 || l == eng->arch.n_layers - 1) {
            const char* type_str = "ATTN";
            if (layer->type == LAYER_DELTANET) type_str = "DELTA";
            else if (layer->type == LAYER_MAMBA) type_str = "MAMBA";
            else if (layer->type == LAYER_MAMBA2) type_str = "MAMBA2";
            
            printf("  layer %u/%u [%s] — VRAM: %.2f GB / %.2f GB\n",
                   l + 1, eng->arch.n_layers, type_str,
                   (double)vk_memory_used(&eng->vk) / (1024.0*1024.0*1024.0),
                   (double)vk_memory_total(&eng->vk) / (1024.0*1024.0*1024.0));
        }
    }
    
    /* Upload output norm + LM head */
    if (!upload_tensor(eng, &eng->weights.output_norm, "output_norm.weight", true))
        return false;
    
    if (!upload_tensor(eng, &eng->weights.output, "output.weight", false)) {
        printf("engine: output.weight not found — using weight tying\n");
        eng->weights.output = eng->weights.token_embd;
    }
    
    /* Load tokenizer vocabulary */
    if (!tokenizer_load(&eng->tok, eng->gguf)) {
        fprintf(stderr, "engine: warning — tokenizer failed to load, using byte-level fallback\n");
    }
    
    const gguf_kv* tokens_kv = gguf_find_kv(eng->gguf, "tokenizer.ggml.tokens");
    if (tokens_kv && tokens_kv->type == GGUF_TYPE_ARRAY) {
        eng->vocab_size = (uint32_t)tokens_kv->value.arr.count;
        printf("engine: vocabulary size: %u tokens\n", eng->vocab_size);
    } else {
        eng->vocab_size = eng->arch.vocab_size;
    }
    
    eng->loaded = true;
    printf("engine: model loaded — %.2f GB VRAM used (hybrid=%s)\n",
           (double)vk_memory_used(&eng->vk) / (1024.0*1024.0*1024.0),
           is_hybrid ? "yes" : "no");
    
    return true;
}

/* ───── Cache Allocation ───── */

bool engine_alloc_cache(engine* eng, uint32_t max_seq_len) {
    uint32_t H = eng->arch.hidden_size;
    uint32_t I = eng->arch.intermediate_size;
    uint32_t V = eng->vocab_size;
    
    /* Count layer types */
    uint32_t n_attn_layers = 0;
    uint32_t n_delta_layers = 0;
    uint32_t n_mamba_layers = 0;
    uint32_t n_mamba2_layers = 0;
    
    for (uint32_t l = 0; l < eng->arch.n_layers; l++) {
        switch (eng->weights.layers[l].type) {
            case LAYER_ATTENTION: n_attn_layers++; break;
            case LAYER_DELTANET:  n_delta_layers++; break;
            case LAYER_MAMBA:     n_mamba_layers++; break;
            case LAYER_MAMBA2:    n_mamba2_layers++; break;
        }
    }
    
    uint32_t n_recurrent_layers = n_delta_layers + n_mamba_layers + n_mamba2_layers;
    
    /* ── KV Cache for attention layers ── */
    eng->cache.n_layers = n_attn_layers;
    eng->cache.max_seq = max_seq_len;
    eng->cache.seq_len = 0;
    
    if (n_attn_layers > 0) {
        size_t kv_size = (size_t)max_seq_len * eng->arch.n_kv_heads * eng->arch.head_dim * sizeof(float);
        eng->cache.k = calloc(n_attn_layers, sizeof(gpu_buffer));
        eng->cache.v = calloc(n_attn_layers, sizeof(gpu_buffer));
        
        for (uint32_t i = 0; i < n_attn_layers; i++) {
            if (!vk_alloc_device(&eng->vk, &eng->cache.k[i], kv_size)) return false;
            if (!vk_alloc_device(&eng->vk, &eng->cache.v[i], kv_size)) return false;
        }
        printf("engine: KV cache: %u attention layers × %u ctx = %.2f MB\n",
               n_attn_layers, max_seq_len,
               (double)(2 * n_attn_layers * kv_size) / (1024.0*1024.0));
    }
    
    /* ── Recurrent state for all recurrent layers ── */
    eng->recurrent.n_recurrent_layers = n_recurrent_layers;
    
    if (n_recurrent_layers > 0) {
        eng->recurrent.conv_state = calloc(n_recurrent_layers, sizeof(gpu_buffer));
        eng->recurrent.ssm_state  = calloc(n_recurrent_layers, sizeof(gpu_buffer));
        
        /* Allocate per-layer — sizes differ by type */
        for (uint32_t l = 0; l < eng->arch.n_layers; l++) {
            hybrid_layer* layer = &eng->weights.layers[l];
            if (layer->type == LAYER_ATTENTION) continue;
            
            uint32_t ci = layer->cache_index;
            size_t conv_state_size = 0;
            size_t ssm_state_size  = 0;
            
            if (layer->type == LAYER_DELTANET) {
                uint32_t d_inner = eng->arch.ssm_d_inner;
                uint32_t conv_k  = eng->arch.ssm_conv_kernel;
                uint32_t num_v_heads = eng->arch.ssm_dt_rank;
                uint32_t head_v_dim  = d_inner / num_v_heads;
                uint32_t head_k_dim  = eng->arch.ssm_d_state;
                
                conv_state_size = (size_t)(conv_k - 1) * d_inner * sizeof(float);
                ssm_state_size  = (size_t)num_v_heads * head_v_dim * head_k_dim * sizeof(float);
                
            } else if (layer->type == LAYER_MAMBA) {
                uint32_t d_inner = eng->arch.ssm_d_inner;
                uint32_t d_state = eng->arch.ssm_d_state;
                uint32_t conv_k  = eng->arch.ssm_conv_kernel;
                
                /* Mamba1 conv_state: (d_conv - 1) * d_inner */
                conv_state_size = (size_t)(conv_k - 1) * d_inner * sizeof(float);
                /* Mamba1 ssm_state: d_state * d_inner */
                ssm_state_size  = (size_t)d_state * d_inner * sizeof(float);
                
            } else if (layer->type == LAYER_MAMBA2) {
                uint32_t d_inner  = eng->arch.ssm_d_inner;
                uint32_t d_state  = eng->arch.ssm_d_state;
                uint32_t conv_k   = eng->arch.ssm_conv_kernel;
                uint32_t n_group  = eng->arch.ssm_n_group;
                uint32_t n_head   = eng->arch.ssm_n_head;
                uint32_t dim_head = (n_head > 0) ? (d_inner / n_head) : 0;
                
                /* Mamba2 conv_state: (d_conv-1) * (d_inner + 2*n_group*d_state) */
                uint32_t conv_channels = d_inner + 2 * n_group * d_state;
                conv_state_size = (size_t)(conv_k - 1) * conv_channels * sizeof(float);
                /* Mamba2 ssm_state: n_head * dim_per_head * d_state */
                ssm_state_size  = (size_t)n_head * dim_head * d_state * sizeof(float);
            }
            
            if (!vk_alloc_device(&eng->vk, &eng->recurrent.conv_state[ci], conv_state_size)) return false;
            if (!vk_alloc_device(&eng->vk, &eng->recurrent.ssm_state[ci],  ssm_state_size))  return false;
        }
        
        printf("engine: Recurrent state: %u layers (DeltaNet=%u, Mamba=%u, Mamba2=%u)\n",
               n_recurrent_layers, n_delta_layers, n_mamba_layers, n_mamba2_layers);
    }
    
    /* ── Scratch buffers (sized for single-token generation) ── */
    vk_alloc_device(&eng->vk, &eng->scratch.hidden,       H * sizeof(float));
    vk_alloc_device(&eng->vk, &eng->scratch.residual,     H * sizeof(float));
    vk_alloc_device(&eng->vk, &eng->scratch.norm_out,     H * sizeof(float));
    
    if (n_attn_layers > 0) {
        vk_alloc_device(&eng->vk, &eng->scratch.q,            eng->arch.n_heads * eng->arch.head_dim * sizeof(float));
        vk_alloc_device(&eng->vk, &eng->scratch.k,            eng->arch.n_kv_heads * eng->arch.head_dim * sizeof(float));
        vk_alloc_device(&eng->vk, &eng->scratch.v,            eng->arch.n_kv_heads * eng->arch.head_dim * sizeof(float));
        vk_alloc_device(&eng->vk, &eng->scratch.attn_out,     H * sizeof(float));
        vk_alloc_device(&eng->vk, &eng->scratch.attn_scores,  eng->arch.n_heads * max_seq_len * sizeof(float));
    }
    
    if (I > 0) {
        vk_alloc_device(&eng->vk, &eng->scratch.ffn_gate_out, I * sizeof(float));
        vk_alloc_device(&eng->vk, &eng->scratch.ffn_up_out,   I * sizeof(float));
        vk_alloc_device(&eng->vk, &eng->scratch.ffn_down_out, H * sizeof(float));
    }
    
    vk_alloc_device(&eng->vk, &eng->scratch.logits,       V * sizeof(float));
    
    /* DeltaNet scratch buffers */
    if (n_delta_layers > 0) {
        uint32_t d_inner = eng->arch.ssm_d_inner;
        uint32_t head_k_dim = eng->arch.ssm_d_state;
        uint32_t num_k_heads = eng->arch.ssm_n_group;
        uint32_t head_v_dim = d_inner / eng->arch.ssm_dt_rank;
        uint32_t num_v_heads = eng->arch.ssm_dt_rank;
        uint32_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
        
        vk_alloc_device(&eng->vk, &eng->scratch.delta_qkv,  qkv_dim * sizeof(float));
        vk_alloc_device(&eng->vk, &eng->scratch.delta_gate,  d_inner * sizeof(float));
        vk_alloc_device(&eng->vk, &eng->scratch.delta_conv,  d_inner * sizeof(float));
        vk_alloc_device(&eng->vk, &eng->scratch.delta_out,   d_inner * sizeof(float));
    }
    
    /* Mamba scratch buffers */
    if (n_mamba_layers > 0 || n_mamba2_layers > 0) {
        uint32_t d_inner   = eng->arch.ssm_d_inner;
        uint32_t d_state   = eng->arch.ssm_d_state;
        uint32_t dt_rank   = eng->arch.ssm_dt_rank;
        uint32_t n_group   = eng->arch.ssm_n_group;
        uint32_t n_head    = eng->arch.ssm_n_head;
        
        /* mamba_in: max of Mamba1's 2*d_inner and Mamba2's d_in_proj */
        uint32_t mamba_in_size = 2 * d_inner;
        if (eng->arch.ssm_d_in_proj > mamba_in_size)
            mamba_in_size = eng->arch.ssm_d_in_proj;
        vk_alloc_device(&eng->vk, &eng->scratch.mamba_in, mamba_in_size * sizeof(float));
        
        /* mamba_x: Mamba1 only: dt_rank + 2*d_state */
        if (n_mamba_layers > 0) {
            vk_alloc_device(&eng->vk, &eng->scratch.mamba_x, (dt_rank + 2 * d_state) * sizeof(float));
        }
        
        /* mamba_dt: max of d_inner (Mamba1) and n_head (Mamba2) */
        uint32_t dt_size = d_inner;
        if (n_head > dt_size) dt_size = n_head;
        vk_alloc_device(&eng->vk, &eng->scratch.mamba_dt, dt_size * sizeof(float));
        
        /* mamba_B, mamba_C: max of d_state (Mamba1) and n_group*d_state (Mamba2) */
        uint32_t bc_size = d_state;
        if (n_group * d_state > bc_size) bc_size = n_group * d_state;
        vk_alloc_device(&eng->vk, &eng->scratch.mamba_B, bc_size * sizeof(float));
        vk_alloc_device(&eng->vk, &eng->scratch.mamba_C, bc_size * sizeof(float));
        
        /* mamba_y: d_inner (output of SSM scan) */
        vk_alloc_device(&eng->vk, &eng->scratch.mamba_y, d_inner * sizeof(float));
    }
    
    printf("engine: caches allocated (ctx=%u) — total VRAM: %.2f GB / %.2f GB\n",
           max_seq_len,
           (double)vk_memory_used(&eng->vk) / (1024.0*1024.0*1024.0),
           (double)vk_memory_total(&eng->vk) / (1024.0*1024.0*1024.0));
    
    return true;
}

/* ───── Attention Layer Forward Pass ───── */

static void forward_layer_attention(engine* eng, uint32_t layer, uint32_t pos) {
    uint32_t H  = eng->arch.hidden_size;
    uint32_t I  = eng->arch.intermediate_size;
    uint32_t n_heads = eng->arch.n_heads;
    uint32_t n_kv = eng->arch.n_kv_heads;
    uint32_t hd = eng->arch.head_dim;
    hybrid_layer* lw = &eng->weights.layers[layer];
    uint32_t ci = lw->cache_index;
    
    /* 1. Attention norm */
    vk_rmsnorm(&eng->vk, &eng->scratch.hidden, &lw->attn_norm,
               &eng->scratch.norm_out, H, eng->arch.rms_norm_eps);
    vk_barrier(&eng->vk);
    
    /* 2. QKV projections */
    vk_matmul_auto(&eng->vk, &eng->scratch.norm_out, &lw->wq,
              &eng->scratch.q, 1, n_heads * hd, H);
    vk_matmul_auto(&eng->vk, &eng->scratch.norm_out, &lw->wk,
              &eng->scratch.k, 1, n_kv * hd, H);
    vk_matmul_auto(&eng->vk, &eng->scratch.norm_out, &lw->wv,
              &eng->scratch.v, 1, n_kv * hd, H);
    vk_barrier(&eng->vk);
    
    /* 2b. Per-head Q/K norms (qwen35 attention layers) */
    if (lw->attn_q_norm.mapped || lw->attn_q_norm.buffer) {
        vk_rmsnorm_head(&eng->vk, &eng->scratch.q, &lw->attn_q_norm,
                        &eng->scratch.q, n_heads, hd, eng->arch.rms_norm_eps);
        vk_barrier(&eng->vk);
    }
    if (lw->attn_k_norm.mapped || lw->attn_k_norm.buffer) {
        vk_rmsnorm_head(&eng->vk, &eng->scratch.k, &lw->attn_k_norm,
                        &eng->scratch.k, n_kv, hd, eng->arch.rms_norm_eps);
        vk_barrier(&eng->vk);
    }
    
    /* 3. RoPE */
    vk_rope(&eng->vk, &eng->scratch.q, &eng->scratch.k,
            hd, n_heads, n_kv, pos, eng->arch.rope_freq_base);
    vk_barrier(&eng->vk);
    
    /* 4. Store K, V into cache at position `pos` */
    uint32_t kv_dim = n_kv * hd;
    vk_kv_cache_store(&eng->vk, &eng->scratch.k, &eng->cache.k[ci],
                      kv_dim, pos, eng->cache.max_seq);
    vk_kv_cache_store(&eng->vk, &eng->scratch.v, &eng->cache.v[ci],
                      kv_dim, pos, eng->cache.max_seq);
    vk_barrier(&eng->vk);
    
    /* 5-7. Multi-head attention with GQA and causal masking */
    vk_gqa_attention(&eng->vk,
                     &eng->scratch.q,
                     &eng->cache.k[ci],
                     &eng->cache.v[ci],
                     &eng->scratch.attn_scores,
                     &eng->scratch.attn_out,
                     hd, n_heads, n_kv, pos + 1, eng->cache.max_seq, pos);
    vk_barrier(&eng->vk);
    
    /* 8. Output projection */
    vk_matmul_auto(&eng->vk, &eng->scratch.attn_out, &lw->wo,
              &eng->scratch.norm_out, 1, H, n_heads * hd);
    vk_barrier(&eng->vk);
    
    /* 9. Residual connection (hidden += attn_output) */
    vk_add(&eng->vk, &eng->scratch.hidden, &eng->scratch.norm_out,
            &eng->scratch.hidden, H);
    vk_barrier(&eng->vk);
    
    /* 10. FFN norm + FFN (if layer has FFN) */
    if (lw->ffn_gate.mapped || lw->ffn_gate.buffer) {
        gpu_buffer* ffn_norm_w = (lw->post_attn_norm.mapped || lw->post_attn_norm.buffer)
                               ? &lw->post_attn_norm : &lw->ffn_norm;
        vk_rmsnorm(&eng->vk, &eng->scratch.hidden, ffn_norm_w,
                   &eng->scratch.norm_out, H, eng->arch.rms_norm_eps);
        vk_barrier(&eng->vk);
        
        vk_matmul_auto(&eng->vk, &eng->scratch.norm_out, &lw->ffn_gate,
                  &eng->scratch.ffn_gate_out, 1, I, H);
        vk_matmul_auto(&eng->vk, &eng->scratch.norm_out, &lw->ffn_up,
                  &eng->scratch.ffn_up_out, 1, I, H);
        vk_barrier(&eng->vk);
        
        vk_silu(&eng->vk, &eng->scratch.ffn_gate_out, I);
        vk_barrier(&eng->vk);
        
        vk_mul(&eng->vk, &eng->scratch.ffn_gate_out, &eng->scratch.ffn_up_out,
                &eng->scratch.ffn_gate_out, I);
        vk_barrier(&eng->vk);
        
        vk_matmul_auto(&eng->vk, &eng->scratch.ffn_gate_out, &lw->ffn_down,
                  &eng->scratch.ffn_down_out, 1, H, I);
        vk_barrier(&eng->vk);
        
        vk_add(&eng->vk, &eng->scratch.hidden, &eng->scratch.ffn_down_out,
                &eng->scratch.hidden, H);
    }
}

/* ───── DeltaNet Layer Forward Pass ───── */

static void forward_layer_deltanet(engine* eng, uint32_t layer, uint32_t pos) {
    (void)pos;
    
    uint32_t H = eng->arch.hidden_size;
    uint32_t I = eng->arch.intermediate_size;
    uint32_t d_inner     = eng->arch.ssm_d_inner;
    uint32_t head_k_dim  = eng->arch.ssm_d_state;
    uint32_t num_k_heads = eng->arch.ssm_n_group;
    uint32_t num_v_heads = eng->arch.ssm_dt_rank;
    uint32_t head_v_dim  = d_inner / num_v_heads;
    uint32_t conv_k      = eng->arch.ssm_conv_kernel;
    
    hybrid_layer* lw = &eng->weights.layers[layer];
    uint32_t ci = lw->cache_index;
    
    uint32_t q_dim = num_k_heads * head_k_dim;
    uint32_t v_dim = num_v_heads * head_v_dim;
    uint32_t qkv_total = q_dim + q_dim + v_dim;
    
    /* 1. Attention norm */
    vk_rmsnorm(&eng->vk, &eng->scratch.hidden, &lw->attn_norm,
               &eng->scratch.norm_out, H, eng->arch.rms_norm_eps);
    vk_barrier(&eng->vk);
    
    /* 2. Fused QKV projection */
    vk_matmul_auto(&eng->vk, &eng->scratch.norm_out, &lw->wqkv,
              &eng->scratch.delta_qkv, 1, qkv_total, H);
    vk_barrier(&eng->vk);
    
    /* 3. Gate Z projection */
    vk_matmul_auto(&eng->vk, &eng->scratch.norm_out, &lw->wqkv_gate,
              &eng->scratch.delta_gate, 1, d_inner, H);
    vk_barrier(&eng->vk);
    
    /* 4. Sigmoid gate */
    vk_sigmoid(&eng->vk, &eng->scratch.delta_gate, d_inner);
    vk_barrier(&eng->vk);
    
    /* 5. Conv1d on V portion */
    vk_conv1d(&eng->vk,
              &eng->recurrent.conv_state[ci],
              &eng->scratch.delta_qkv,
              &lw->ssm_conv1d,
              &eng->scratch.delta_conv,
              d_inner, conv_k);
    vk_barrier(&eng->vk);
    
    /* 6. SiLU on convolved V */
    vk_silu(&eng->vk, &eng->scratch.delta_conv, d_inner);
    vk_barrier(&eng->vk);
    
    /* 7. L2 normalize Q and K */
    vk_l2_norm(&eng->vk, &eng->scratch.delta_qkv, &eng->scratch.delta_qkv,
               q_dim, head_k_dim);
    vk_barrier(&eng->vk);
    
    /* 9. DeltaNet recurrent step */
    vk_deltanet_step(&eng->vk,
                     &eng->scratch.delta_qkv,
                     &eng->scratch.delta_qkv,
                     &eng->scratch.delta_conv,
                     &lw->ssm_a,
                     &eng->recurrent.ssm_state[ci],
                     &eng->scratch.delta_out,
                     num_v_heads, head_k_dim, head_v_dim);
    vk_barrier(&eng->vk);
    
    /* 10. Per-head RMS norm */
    vk_rmsnorm_head(&eng->vk, &eng->scratch.delta_out, &lw->ssm_norm,
                    &eng->scratch.delta_out,
                    num_v_heads, head_v_dim, eng->arch.rms_norm_eps);
    vk_barrier(&eng->vk);
    
    /* 11. Gate */
    vk_mul(&eng->vk, &eng->scratch.delta_out, &eng->scratch.delta_gate,
            &eng->scratch.delta_out, d_inner);
    vk_barrier(&eng->vk);
    
    /* 12. Output projection */
    vk_matmul_auto(&eng->vk, &eng->scratch.delta_out, &lw->ssm_out,
              &eng->scratch.norm_out, 1, H, d_inner);
    vk_barrier(&eng->vk);
    
    /* 13. Residual */
    vk_add(&eng->vk, &eng->scratch.hidden, &eng->scratch.norm_out,
            &eng->scratch.hidden, H);
    vk_barrier(&eng->vk);
    
    /* 14-15. FFN */
    if (lw->ffn_gate.mapped || lw->ffn_gate.buffer) {
        gpu_buffer* ffn_norm_w = (lw->post_attn_norm.mapped || lw->post_attn_norm.buffer)
                               ? &lw->post_attn_norm : &lw->ffn_norm;
        vk_rmsnorm(&eng->vk, &eng->scratch.hidden, ffn_norm_w,
                   &eng->scratch.norm_out, H, eng->arch.rms_norm_eps);
        vk_barrier(&eng->vk);
        
        vk_matmul_auto(&eng->vk, &eng->scratch.norm_out, &lw->ffn_gate,
                  &eng->scratch.ffn_gate_out, 1, I, H);
        vk_matmul_auto(&eng->vk, &eng->scratch.norm_out, &lw->ffn_up,
                  &eng->scratch.ffn_up_out, 1, I, H);
        vk_barrier(&eng->vk);
        
        vk_silu(&eng->vk, &eng->scratch.ffn_gate_out, I);
        vk_barrier(&eng->vk);
        
        vk_mul(&eng->vk, &eng->scratch.ffn_gate_out, &eng->scratch.ffn_up_out,
                &eng->scratch.ffn_gate_out, I);
        vk_barrier(&eng->vk);
        
        vk_matmul_auto(&eng->vk, &eng->scratch.ffn_gate_out, &lw->ffn_down,
                  &eng->scratch.ffn_down_out, 1, H, I);
        vk_barrier(&eng->vk);
        
        vk_add(&eng->vk, &eng->scratch.hidden, &eng->scratch.ffn_down_out,
                &eng->scratch.hidden, H);
    }
}

/* ───── Mamba-1 (S6) Layer Forward Pass ───── */

static void forward_layer_mamba1(engine* eng, uint32_t layer, uint32_t pos) {
    (void)pos;  /* Mamba is recurrent, position is implicit in state */
    
    uint32_t H       = eng->arch.hidden_size;
    uint32_t I       = eng->arch.intermediate_size;
    uint32_t d_inner = eng->arch.ssm_d_inner;
    uint32_t d_state = eng->arch.ssm_d_state;
    uint32_t dt_rank = eng->arch.ssm_dt_rank;
    uint32_t conv_k  = eng->arch.ssm_conv_kernel;
    
    hybrid_layer* lw = &eng->weights.layers[layer];
    uint32_t ci = lw->cache_index;
    
    /* 1. RMS norm (attn_norm) */
    vk_rmsnorm(&eng->vk, &eng->scratch.hidden, &lw->attn_norm,
               &eng->scratch.norm_out, H, eng->arch.rms_norm_eps);
    vk_barrier(&eng->vk);
    
    /* 2. ssm_in projection: hidden → [2*d_inner] (x and z) */
    vk_matmul_auto(&eng->vk, &eng->scratch.norm_out, &lw->ssm_in,
              &eng->scratch.mamba_in, 1, 2 * d_inner, H);
    vk_barrier(&eng->vk);
    
    /* Split mamba_in into x [d_inner] and z [d_inner]:
     * x = mamba_in[0..d_inner-1], z = mamba_in[d_inner..2*d_inner-1]
     * For conv1d we pass x portion. z is used later for gating.
     * We use mamba_y as temp for x, and keep z in-place in mamba_in[d_inner:]. */
    
    /* 3. conv1d on x (using conv_state ring buffer) + bias */
    /* The conv1d function reads from the start of the input buffer.
     * We need to use the first d_inner elements of mamba_in.
     * Since conv1d reads from x->mapped, and mamba_in starts with x, this works directly. */
    vk_conv1d(&eng->vk,
              &eng->recurrent.conv_state[ci],
              &eng->scratch.mamba_in,      /* x is at start of mamba_in */
              &lw->ssm_conv1d,
              &eng->scratch.mamba_y,        /* conv output → mamba_y (reuse as temp) */
              d_inner, conv_k);
    vk_barrier(&eng->vk);
    
    /* Add conv1d bias */
    vk_add_bias(&eng->vk, &eng->scratch.mamba_y, &lw->ssm_conv1d_b, d_inner);
    vk_barrier(&eng->vk);
    
    /* 4. SiLU on x (now in mamba_y) */
    vk_silu(&eng->vk, &eng->scratch.mamba_y, d_inner);
    vk_barrier(&eng->vk);
    
    /* 5. ssm_x projection: x → [dt_rank + 2*d_state] to get dt_raw, B, C */
    vk_matmul_auto(&eng->vk, &eng->scratch.mamba_y, &lw->ssm_x,
              &eng->scratch.mamba_x, 1, dt_rank + 2 * d_state, d_inner);
    vk_barrier(&eng->vk);
    
    /* Split mamba_x into dt_raw [dt_rank], B [d_state], C [d_state] via pointer arithmetic */
    /* For CPU: the data is contiguous in mamba_x.mapped */
    {
        float* mx = (float*)eng->scratch.mamba_x.mapped;
        float* B_dst = (float*)eng->scratch.mamba_B.mapped;
        float* C_dst = (float*)eng->scratch.mamba_C.mapped;
        
        /* Copy B and C from mamba_x to their scratch buffers */
        memcpy(B_dst, mx + dt_rank, d_state * sizeof(float));
        memcpy(C_dst, mx + dt_rank + d_state, d_state * sizeof(float));
    }
    
    /* 6. dt projection: dt_raw [dt_rank] @ ssm_dt_w [dt_rank, d_inner] + ssm_dt_b → dt [d_inner] */
    /* We need a temp buffer for dt_raw. Reuse mamba_x (first dt_rank elements) as input.
     * mamba_x already has dt_raw at the start. */
    vk_matmul_auto(&eng->vk, &eng->scratch.mamba_x, &lw->ssm_dt_w,
              &eng->scratch.mamba_dt, 1, d_inner, dt_rank);
    vk_barrier(&eng->vk);
    
    /* Add dt bias */
    vk_add_bias(&eng->vk, &eng->scratch.mamba_dt, &lw->ssm_dt_b, d_inner);
    vk_barrier(&eng->vk);
    
    /* 7. softplus on dt */
    vk_softplus(&eng->vk, &eng->scratch.mamba_dt, d_inner);
    vk_barrier(&eng->vk);
    
    /* 8. SSM scan: mamba1_ssm_step(x, dt, A, B, C, D, state, y)
     * x is in mamba_y (post-conv, post-silu) */
    vk_mamba1_ssm_step(&eng->vk,
                       &eng->scratch.mamba_y,        /* x [d_inner] */
                       &eng->scratch.mamba_dt,        /* dt [d_inner] */
                       &lw->ssm_a,                    /* A [d_state, d_inner] */
                       &eng->scratch.mamba_B,         /* B [d_state] */
                       &eng->scratch.mamba_C,         /* C [d_state] */
                       &lw->ssm_d,                    /* D [d_inner] */
                       &eng->recurrent.ssm_state[ci], /* state [d_state * d_inner] */
                       &eng->scratch.delta_out,       /* y [d_inner] — reuse delta_out as temp */
                       d_inner, d_state);
    vk_barrier(&eng->vk);
    
    /* 9. y = y * silu(z) — output gating
     * z is in mamba_in at offset d_inner */
    {
        float* z = (float*)eng->scratch.mamba_in.mapped + d_inner;
        float* y_out = (float*)eng->scratch.delta_out.mapped;
        for (uint32_t i = 0; i < d_inner; i++) {
            float silu_z = z[i] / (1.0f + expf(-z[i]));
            y_out[i] *= silu_z;
        }
    }
    vk_barrier(&eng->vk);
    
    /* 10. ssm_out projection: [d_inner] → [hidden_size] */
    vk_matmul_auto(&eng->vk, &eng->scratch.delta_out, &lw->ssm_out,
              &eng->scratch.norm_out, 1, H, d_inner);
    vk_barrier(&eng->vk);
    
    /* 11. Residual add */
    vk_add(&eng->vk, &eng->scratch.hidden, &eng->scratch.norm_out,
            &eng->scratch.hidden, H);
    vk_barrier(&eng->vk);
    
    /* 12. FFN (if layer has it — pure Mamba may not) */
    if (lw->ffn_gate.mapped || lw->ffn_gate.buffer) {
        gpu_buffer* ffn_norm_w = (lw->post_attn_norm.mapped || lw->post_attn_norm.buffer)
                               ? &lw->post_attn_norm : &lw->ffn_norm;
        vk_rmsnorm(&eng->vk, &eng->scratch.hidden, ffn_norm_w,
                   &eng->scratch.norm_out, H, eng->arch.rms_norm_eps);
        vk_barrier(&eng->vk);
        
        vk_matmul_auto(&eng->vk, &eng->scratch.norm_out, &lw->ffn_gate,
                  &eng->scratch.ffn_gate_out, 1, I, H);
        vk_matmul_auto(&eng->vk, &eng->scratch.norm_out, &lw->ffn_up,
                  &eng->scratch.ffn_up_out, 1, I, H);
        vk_barrier(&eng->vk);
        
        vk_silu(&eng->vk, &eng->scratch.ffn_gate_out, I);
        vk_barrier(&eng->vk);
        
        vk_mul(&eng->vk, &eng->scratch.ffn_gate_out, &eng->scratch.ffn_up_out,
                &eng->scratch.ffn_gate_out, I);
        vk_barrier(&eng->vk);
        
        vk_matmul_auto(&eng->vk, &eng->scratch.ffn_gate_out, &lw->ffn_down,
                  &eng->scratch.ffn_down_out, 1, H, I);
        vk_barrier(&eng->vk);
        
        vk_add(&eng->vk, &eng->scratch.hidden, &eng->scratch.ffn_down_out,
                &eng->scratch.hidden, H);
    }
}

/* ───── Mamba-2 (SSD) Layer Forward Pass ───── */

static void forward_layer_mamba2(engine* eng, uint32_t layer, uint32_t pos) {
    (void)pos;
    
    uint32_t H       = eng->arch.hidden_size;
    uint32_t I       = eng->arch.intermediate_size;
    uint32_t d_inner = eng->arch.ssm_d_inner;
    uint32_t d_state = eng->arch.ssm_d_state;
    uint32_t conv_k  = eng->arch.ssm_conv_kernel;
    uint32_t n_head  = eng->arch.ssm_n_head;
    uint32_t n_group = eng->arch.ssm_n_group;
    uint32_t dim_per_head = (n_head > 0) ? (d_inner / n_head) : 0;
    
    /* Mamba-2 ssm_in projects to: x[d_inner] + z[d_inner] + B[n_group*d_state] + C[n_group*d_state] + dt[n_head]
     * Total d_in_proj = 2*d_inner + 2*n_group*d_state + n_head */
    uint32_t d_in_proj = 2 * d_inner + 2 * n_group * d_state + n_head;
    
    /* Conv1d channels for Mamba2: d_inner + 2*n_group*d_state (x + B + C get convolved) */
    uint32_t conv_channels = d_inner + 2 * n_group * d_state;
    
    hybrid_layer* lw = &eng->weights.layers[layer];
    uint32_t ci = lw->cache_index;
    
    /* 1. RMS norm */
    vk_rmsnorm(&eng->vk, &eng->scratch.hidden, &lw->attn_norm,
               &eng->scratch.norm_out, H, eng->arch.rms_norm_eps);
    vk_barrier(&eng->vk);
    
    /* 2. ssm_in projection: hidden → [d_in_proj] */
    vk_matmul_auto(&eng->vk, &eng->scratch.norm_out, &lw->ssm_in,
              &eng->scratch.mamba_in, 1, d_in_proj, H);
    vk_barrier(&eng->vk);
    
    /* Split mamba_in into: x[d_inner], z[d_inner], B[n_group*d_state], C[n_group*d_state], dt_raw[n_head]
     * Layout: [x | z | B | C | dt] */
    float* proj_data = (float*)eng->scratch.mamba_in.mapped;
    /* x  = proj_data[0 .. d_inner-1]
     * z  = proj_data[d_inner .. 2*d_inner-1]
     * B  = proj_data[2*d_inner .. 2*d_inner + n_group*d_state - 1]
     * C  = proj_data[2*d_inner + n_group*d_state .. 2*d_inner + 2*n_group*d_state - 1]
     * dt = proj_data[2*d_inner + 2*n_group*d_state .. d_in_proj-1] */
    
    /* 3. Prepare conv input: concatenate [x, B, C] → conv_channels elements
     * x is already at the start of mamba_in. We need to move B and C to follow x.
     * For conv1d, we need a contiguous buffer of [x[d_inner], B[ng*ds], C[ng*ds]].
     * Since z sits between x and B, we need to assemble this. Use mamba_y as temp. */
    {
        float* conv_in = (float*)eng->scratch.mamba_y.mapped;
        /* Copy x */
        memcpy(conv_in, proj_data, d_inner * sizeof(float));
        /* Copy B */
        memcpy(conv_in + d_inner, proj_data + 2 * d_inner, n_group * d_state * sizeof(float));
        /* Copy C */
        memcpy(conv_in + d_inner + n_group * d_state,
               proj_data + 2 * d_inner + n_group * d_state, n_group * d_state * sizeof(float));
    }
    
    /* Conv1d on [x, B, C] concatenated */
    /* We use a temp approach: upload conv_in via mamba_y, conv1d into delta_out (reuse) */
    vk_conv1d(&eng->vk,
              &eng->recurrent.conv_state[ci],
              &eng->scratch.mamba_y,          /* [x, B, C] input */
              &lw->ssm_conv1d,
              &eng->scratch.delta_out,         /* conv output → reuse delta_out [conv_channels] */
              conv_channels, conv_k);
    vk_barrier(&eng->vk);
    
    /* Add conv1d bias */
    vk_add_bias(&eng->vk, &eng->scratch.delta_out, &lw->ssm_conv1d_b, conv_channels);
    vk_barrier(&eng->vk);
    
    /* 4. Split conv output back to x[d_inner], B[n_group*d_state], C[n_group*d_state] */
    {
        float* conv_out = (float*)eng->scratch.delta_out.mapped;
        /* x is conv_out[0..d_inner-1] — stays in delta_out */
        /* B → mamba_B */
        memcpy((float*)eng->scratch.mamba_B.mapped, conv_out + d_inner, n_group * d_state * sizeof(float));
        /* C → mamba_C */
        memcpy((float*)eng->scratch.mamba_C.mapped, conv_out + d_inner + n_group * d_state, n_group * d_state * sizeof(float));
        /* Copy x to mamba_y for SiLU + SSM */
        memcpy((float*)eng->scratch.mamba_y.mapped, conv_out, d_inner * sizeof(float));
    }
    
    /* 5. SiLU on x (now in mamba_y) */
    vk_silu(&eng->vk, &eng->scratch.mamba_y, d_inner);
    vk_barrier(&eng->vk);
    
    /* 6. softplus(dt_raw + dt_bias) → mamba_dt */
    {
        float* dt_raw = proj_data + 2 * d_inner + 2 * n_group * d_state;
        float* dt_out = (float*)eng->scratch.mamba_dt.mapped;
        memcpy(dt_out, dt_raw, n_head * sizeof(float));
    }
    vk_add_bias(&eng->vk, &eng->scratch.mamba_dt, &lw->ssm_dt_b, n_head);
    vk_barrier(&eng->vk);
    vk_softplus(&eng->vk, &eng->scratch.mamba_dt, n_head);
    vk_barrier(&eng->vk);
    
    /* 7. SSM scan: mamba2_ssm_step */
    vk_mamba2_ssm_step(&eng->vk,
                       &eng->scratch.mamba_y,         /* x [d_inner] */
                       &eng->scratch.mamba_dt,         /* dt [n_head] */
                       &lw->ssm_a,                     /* A [n_head] */
                       &eng->scratch.mamba_B,          /* B [n_group*d_state] */
                       &eng->scratch.mamba_C,          /* C [n_group*d_state] */
                       &lw->ssm_d,                     /* D [n_head] */
                       &eng->recurrent.ssm_state[ci],  /* state */
                       &eng->scratch.delta_out,        /* y [d_inner] — reuse */
                       n_head, dim_per_head, d_state, n_group);
    vk_barrier(&eng->vk);
    
    /* 8. Per-group RMS norm on y */
    if (lw->ssm_norm.mapped || lw->ssm_norm.buffer) {
        uint32_t group_dim = d_inner / n_group;
        vk_rmsnorm_head(&eng->vk, &eng->scratch.delta_out, &lw->ssm_norm,
                        &eng->scratch.delta_out,
                        n_group, group_dim, eng->arch.rms_norm_eps);
        vk_barrier(&eng->vk);
    }
    
    /* 9. y = y * silu(z) — output gating
     * z is at proj_data + d_inner */
    {
        float* z = proj_data + d_inner;
        float* y_out = (float*)eng->scratch.delta_out.mapped;
        for (uint32_t i = 0; i < d_inner; i++) {
            float silu_z = z[i] / (1.0f + expf(-z[i]));
            y_out[i] *= silu_z;
        }
    }
    vk_barrier(&eng->vk);
    
    /* 10. ssm_out projection: [d_inner] → [hidden_size] */
    vk_matmul_auto(&eng->vk, &eng->scratch.delta_out, &lw->ssm_out,
              &eng->scratch.norm_out, 1, H, d_inner);
    vk_barrier(&eng->vk);
    
    /* 11. Residual add */
    vk_add(&eng->vk, &eng->scratch.hidden, &eng->scratch.norm_out,
            &eng->scratch.hidden, H);
    vk_barrier(&eng->vk);
    
    /* 12. FFN (if layer has it) */
    if (lw->ffn_gate.mapped || lw->ffn_gate.buffer) {
        gpu_buffer* ffn_norm_w = (lw->post_attn_norm.mapped || lw->post_attn_norm.buffer)
                               ? &lw->post_attn_norm : &lw->ffn_norm;
        vk_rmsnorm(&eng->vk, &eng->scratch.hidden, ffn_norm_w,
                   &eng->scratch.norm_out, H, eng->arch.rms_norm_eps);
        vk_barrier(&eng->vk);
        
        vk_matmul_auto(&eng->vk, &eng->scratch.norm_out, &lw->ffn_gate,
                  &eng->scratch.ffn_gate_out, 1, I, H);
        vk_matmul_auto(&eng->vk, &eng->scratch.norm_out, &lw->ffn_up,
                  &eng->scratch.ffn_up_out, 1, I, H);
        vk_barrier(&eng->vk);
        
        vk_silu(&eng->vk, &eng->scratch.ffn_gate_out, I);
        vk_barrier(&eng->vk);
        
        vk_mul(&eng->vk, &eng->scratch.ffn_gate_out, &eng->scratch.ffn_up_out,
                &eng->scratch.ffn_gate_out, I);
        vk_barrier(&eng->vk);
        
        vk_matmul_auto(&eng->vk, &eng->scratch.ffn_gate_out, &lw->ffn_down,
                  &eng->scratch.ffn_down_out, 1, H, I);
        vk_barrier(&eng->vk);
        
        vk_add(&eng->vk, &eng->scratch.hidden, &eng->scratch.ffn_down_out,
                &eng->scratch.hidden, H);
    }
}

/* ───── Layer Dispatch ───── */

static void forward_layer(engine* eng, uint32_t layer, uint32_t pos) {
    switch (eng->weights.layers[layer].type) {
        case LAYER_ATTENTION: forward_layer_attention(eng, layer, pos); break;
        case LAYER_DELTANET:  forward_layer_deltanet(eng, layer, pos); break;
        case LAYER_MAMBA:     forward_layer_mamba1(eng, layer, pos); break;
        case LAYER_MAMBA2:    forward_layer_mamba2(eng, layer, pos); break;
    }
}

/* ───── Full Forward Pass ───── */

bool engine_forward(engine* eng, const uint32_t* tokens, uint32_t n_tokens) {
    uint32_t pos = eng->cache.seq_len;
    
    for (uint32_t t = 0; t < n_tokens; t++) {
        /* Embed token (handles quantized embeddings) */
        vk_begin_compute(&eng->vk);
        vk_embedding_auto(&eng->vk, &eng->weights.token_embd, &eng->scratch.hidden,
                          tokens[t], eng->arch.hidden_size);
        vk_barrier(&eng->vk);
        
        /* Run through all layers (dispatch based on type) */
        for (uint32_t l = 0; l < eng->arch.n_layers; l++) {
            forward_layer(eng, l, pos + t);
        }
        
        /* Final RMS norm */
        vk_rmsnorm(&eng->vk, &eng->scratch.hidden, &eng->weights.output_norm,
                   &eng->scratch.norm_out, eng->arch.hidden_size, eng->arch.rms_norm_eps);
        vk_barrier(&eng->vk);
        
        /* LM head: logits = norm_out @ output_weight^T (quantization-aware) */
        vk_matmul_auto(&eng->vk, &eng->scratch.norm_out, &eng->weights.output,
                       &eng->scratch.logits, 1, eng->vocab_size, eng->arch.hidden_size);
        
        vk_submit_and_wait(&eng->vk);
    }
    
    eng->cache.seq_len += n_tokens;
    return true;
}

/* ───── Token Sampling (CPU-side) ───── */

static uint32_t sample_token(float* logits, uint32_t vocab_size, const sample_params* params) {
    if (params->temperature > 0.0f && params->temperature != 1.0f) {
        float inv_temp = 1.0f / params->temperature;
        for (uint32_t i = 0; i < vocab_size; i++) {
            logits[i] *= inv_temp;
        }
    }
    
    if (params->temperature == 0.0f) {
        uint32_t best = 0;
        float best_val = logits[0];
        for (uint32_t i = 1; i < vocab_size; i++) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best = i;
            }
        }
        return best;
    }
    
    float max_val = logits[0];
    for (uint32_t i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    
    float sum = 0.0f;
    for (uint32_t i = 0; i < vocab_size; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }
    for (uint32_t i = 0; i < vocab_size; i++) {
        logits[i] /= sum;
    }
    
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    for (uint32_t i = 0; i < vocab_size; i++) {
        cumsum += logits[i];
        if (cumsum >= r) return i;
    }
    
    return vocab_size - 1;
}

/* ───── Generate Tokens ───── */

uint32_t engine_generate(engine* eng, const uint32_t* prompt, uint32_t prompt_len,
                         const sample_params* params, token_callback cb, void* user_data) {
    if (params->seed != 0) srand(params->seed);
    else srand((unsigned)time(NULL));
    
    printf("engine: prefilling %u tokens...\n", prompt_len);
    engine_forward(eng, prompt, prompt_len);
    
    float* logits_cpu = malloc(eng->vocab_size * sizeof(float));
    if (!logits_cpu) return 0;
    
    uint32_t generated = 0;
    uint32_t token = 0;
    
#ifdef _WIN32
    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);
#else
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    
    for (uint32_t i = 0; i < params->max_tokens; i++) {
        vk_download(&eng->vk, logits_cpu, &eng->scratch.logits,
                    eng->vocab_size * sizeof(float));
        
        token = sample_token(logits_cpu, eng->vocab_size, params);
        
        if (token == eng->arch.eos_token_id) break;
        
        generated++;
        
        if (cb) {
            const char* token_str = engine_detokenize(eng, token);
            cb(token, token_str ? token_str : "?", user_data);
        }
        
        engine_forward(eng, &token, 1);
    }
    
#ifdef _WIN32
    QueryPerformanceCounter(&end);
    double elapsed = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart;
#else
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
#endif
    
    printf("engine: generated %u tokens in %.2fs (%.1f tok/s)\n",
           generated, elapsed, generated / elapsed);
    
    free(logits_cpu);
    return generated;
}

/* ───── Tokenization (BPE) ───── */

uint32_t* engine_tokenize(const engine* eng, const char* text, uint32_t* n_tokens) {
    if (eng->tok.loaded) {
        return tokenizer_encode(&eng->tok, text, n_tokens, true);
    }
    
    size_t len = strlen(text);
    uint32_t* tokens = malloc((len + 2) * sizeof(uint32_t));
    tokens[0] = eng->arch.bos_token_id;
    uint32_t count = 1;
    for (size_t i = 0; i < len; i++) {
        tokens[count++] = (uint32_t)(unsigned char)text[i];
    }
    *n_tokens = count;
    return tokens;
}

const char* engine_detokenize(const engine* eng, uint32_t token_id) {
    if (eng->tok.loaded) {
        return tokenizer_decode(&eng->tok, token_id);
    }
    
    static char buf[8];
    if (token_id < 128 && token_id >= 32) {
        buf[0] = (char)token_id;
        buf[1] = '\0';
        return buf;
    }
    return "?";
}

/* ───── Reset ───── */

void engine_reset(engine* eng) {
    eng->cache.seq_len = 0;
    
    /* Zero out all recurrent states (DeltaNet + Mamba + Mamba2) */
    for (uint32_t i = 0; i < eng->recurrent.n_recurrent_layers; i++) {
        if (eng->recurrent.conv_state[i].mapped)
            memset(eng->recurrent.conv_state[i].mapped, 0, eng->recurrent.conv_state[i].size);
        if (eng->recurrent.ssm_state[i].mapped)
            memset(eng->recurrent.ssm_state[i].mapped, 0, eng->recurrent.ssm_state[i].size);
    }
}

/* ───── Status ───── */

void engine_print_status(const engine* eng) {
    printf("\n═══ Artifact Engine Status ═══\n");
    printf("Model:      %s\n", eng->arch.arch);
    printf("Hybrid:     %s\n", eng->weights.is_hybrid ? "yes" : "no");
    printf("Layers:     %u\n", eng->arch.n_layers);
    
    uint32_t n_attn = 0, n_delta = 0, n_mamba = 0, n_mamba2 = 0;
    for (uint32_t l = 0; l < eng->arch.n_layers; l++) {
        switch (eng->weights.layers[l].type) {
            case LAYER_ATTENTION: n_attn++; break;
            case LAYER_DELTANET:  n_delta++; break;
            case LAYER_MAMBA:     n_mamba++; break;
            case LAYER_MAMBA2:    n_mamba2++; break;
        }
    }
    
    if (n_attn > 0)   printf("  Attention: %u\n", n_attn);
    if (n_delta > 0)  printf("  DeltaNet:  %u\n", n_delta);
    if (n_mamba > 0)  printf("  Mamba:     %u\n", n_mamba);
    if (n_mamba2 > 0) printf("  Mamba-2:   %u\n", n_mamba2);
    
    if (eng->arch.ssm_d_inner > 0) {
        printf("  SSM inner: %u | state: %u | groups: %u | dt_rank: %u\n",
               eng->arch.ssm_d_inner, eng->arch.ssm_d_state,
               eng->arch.ssm_n_group, eng->arch.ssm_dt_rank);
        if (eng->arch.ssm_n_head > 0) {
            printf("  SSM heads: %u\n", eng->arch.ssm_n_head);
        }
    }
    
    printf("Hidden:     %u\n", eng->arch.hidden_size);
    if (eng->arch.n_heads > 0) {
        printf("Heads:      %u (KV: %u)\n", eng->arch.n_heads, eng->arch.n_kv_heads);
    }
    if (eng->arch.intermediate_size > 0) {
        printf("FFN:        %u\n", eng->arch.intermediate_size);
    }
    printf("Vocab:      %u\n", eng->vocab_size);
    printf("Context:    %u\n", eng->cache.max_seq);
    printf("VRAM:       %.2f / %.2f GB\n",
           (double)vk_memory_used(&eng->vk) / (1024.0*1024.0*1024.0),
           (double)vk_memory_total(&eng->vk) / (1024.0*1024.0*1024.0));
    vk_print_info(&eng->vk);
}

/* ───── Cleanup ───── */

void engine_destroy(engine* eng) {
    if (!eng->loaded) return;
    
    /* Free scratch buffers */
    vk_free_buffer(&eng->vk, &eng->scratch.hidden);
    vk_free_buffer(&eng->vk, &eng->scratch.residual);
    vk_free_buffer(&eng->vk, &eng->scratch.norm_out);
    vk_free_buffer(&eng->vk, &eng->scratch.q);
    vk_free_buffer(&eng->vk, &eng->scratch.k);
    vk_free_buffer(&eng->vk, &eng->scratch.v);
    vk_free_buffer(&eng->vk, &eng->scratch.attn_out);
    vk_free_buffer(&eng->vk, &eng->scratch.attn_scores);
    vk_free_buffer(&eng->vk, &eng->scratch.ffn_gate_out);
    vk_free_buffer(&eng->vk, &eng->scratch.ffn_up_out);
    vk_free_buffer(&eng->vk, &eng->scratch.ffn_down_out);
    vk_free_buffer(&eng->vk, &eng->scratch.logits);
    
    /* Free DeltaNet scratch */
    vk_free_buffer(&eng->vk, &eng->scratch.delta_qkv);
    vk_free_buffer(&eng->vk, &eng->scratch.delta_gate);
    vk_free_buffer(&eng->vk, &eng->scratch.delta_conv);
    vk_free_buffer(&eng->vk, &eng->scratch.delta_out);
    
    /* Free Mamba scratch */
    vk_free_buffer(&eng->vk, &eng->scratch.mamba_in);
    vk_free_buffer(&eng->vk, &eng->scratch.mamba_x);
    vk_free_buffer(&eng->vk, &eng->scratch.mamba_dt);
    vk_free_buffer(&eng->vk, &eng->scratch.mamba_B);
    vk_free_buffer(&eng->vk, &eng->scratch.mamba_C);
    vk_free_buffer(&eng->vk, &eng->scratch.mamba_y);
    
    /* Free KV cache (attention layers) */
    for (uint32_t l = 0; l < eng->cache.n_layers; l++) {
        vk_free_buffer(&eng->vk, &eng->cache.k[l]);
        vk_free_buffer(&eng->vk, &eng->cache.v[l]);
    }
    free(eng->cache.k);
    free(eng->cache.v);
    
    /* Free recurrent state (DeltaNet + Mamba layers) */
    for (uint32_t i = 0; i < eng->recurrent.n_recurrent_layers; i++) {
        vk_free_buffer(&eng->vk, &eng->recurrent.conv_state[i]);
        vk_free_buffer(&eng->vk, &eng->recurrent.ssm_state[i]);
    }
    free(eng->recurrent.conv_state);
    free(eng->recurrent.ssm_state);
    
    /* Free model weights */
    vk_free_buffer(&eng->vk, &eng->weights.token_embd);
    for (uint32_t l = 0; l < eng->weights.n_layers; l++) {
        hybrid_layer* lw = &eng->weights.layers[l];
        
        /* Common */
        vk_free_buffer(&eng->vk, &lw->attn_norm);
        vk_free_buffer(&eng->vk, &lw->post_attn_norm);
        vk_free_buffer(&eng->vk, &lw->ffn_norm);
        vk_free_buffer(&eng->vk, &lw->ffn_gate);
        vk_free_buffer(&eng->vk, &lw->ffn_up);
        vk_free_buffer(&eng->vk, &lw->ffn_down);
        
        if (lw->type == LAYER_ATTENTION) {
            vk_free_buffer(&eng->vk, &lw->wq);
            vk_free_buffer(&eng->vk, &lw->wk);
            vk_free_buffer(&eng->vk, &lw->wv);
            vk_free_buffer(&eng->vk, &lw->wo);
            vk_free_buffer(&eng->vk, &lw->attn_q_norm);
            vk_free_buffer(&eng->vk, &lw->attn_k_norm);
        } else if (lw->type == LAYER_DELTANET) {
            vk_free_buffer(&eng->vk, &lw->wqkv);
            vk_free_buffer(&eng->vk, &lw->wqkv_gate);
            vk_free_buffer(&eng->vk, &lw->ssm_alpha);
            vk_free_buffer(&eng->vk, &lw->ssm_beta);
            vk_free_buffer(&eng->vk, &lw->ssm_a);
            vk_free_buffer(&eng->vk, &lw->ssm_dt);
            vk_free_buffer(&eng->vk, &lw->ssm_conv1d);
            vk_free_buffer(&eng->vk, &lw->ssm_norm);
            vk_free_buffer(&eng->vk, &lw->ssm_out);
        } else if (lw->type == LAYER_MAMBA) {
            vk_free_buffer(&eng->vk, &lw->ssm_in);
            vk_free_buffer(&eng->vk, &lw->ssm_conv1d);
            vk_free_buffer(&eng->vk, &lw->ssm_conv1d_b);
            vk_free_buffer(&eng->vk, &lw->ssm_x);
            vk_free_buffer(&eng->vk, &lw->ssm_dt_w);
            vk_free_buffer(&eng->vk, &lw->ssm_dt_b);
            vk_free_buffer(&eng->vk, &lw->ssm_a);
            vk_free_buffer(&eng->vk, &lw->ssm_d);
            vk_free_buffer(&eng->vk, &lw->ssm_out);
        } else if (lw->type == LAYER_MAMBA2) {
            vk_free_buffer(&eng->vk, &lw->ssm_in);
            vk_free_buffer(&eng->vk, &lw->ssm_conv1d);
            vk_free_buffer(&eng->vk, &lw->ssm_conv1d_b);
            vk_free_buffer(&eng->vk, &lw->ssm_dt_b);
            vk_free_buffer(&eng->vk, &lw->ssm_a);
            vk_free_buffer(&eng->vk, &lw->ssm_d);
            vk_free_buffer(&eng->vk, &lw->ssm_norm);
            vk_free_buffer(&eng->vk, &lw->ssm_out);
        }
    }
    free(eng->weights.layers);
    vk_free_buffer(&eng->vk, &eng->weights.output_norm);
    if (eng->weights.output.buffer != eng->weights.token_embd.buffer &&
        eng->weights.output.mapped != eng->weights.token_embd.mapped) {
        vk_free_buffer(&eng->vk, &eng->weights.output);
    }
    
    /* Free tokenizer */
    tokenizer_free(&eng->tok);
    
    /* Free GGUF */
    gguf_free(eng->gguf);
    
    /* Destroy Vulkan */
    vk_destroy(&eng->vk);
    
    eng->loaded = false;
}
