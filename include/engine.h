/*
 * Artifact Engine — Model & Inference Interface
 * 
 * The transformer forward pass and token generation.
 * Supports hybrid architectures: pure attention, pure DeltaNet,
 * pure Mamba (S6), pure Mamba-2 (SSD), or mixed (Qwen3.5-style,
 * Jamba-style, Falcon-H1-style hybrids).
 */

#ifndef ENGINE_H
#define ENGINE_H

#include "gguf.h"
#include "tokenizer.h"
#include "vulkan_compute.h"
#include <stdbool.h>
#include <stdint.h>

/* ───── Layer Type (hybrid architecture) ───── */
typedef enum {
    LAYER_ATTENTION = 0,
    LAYER_DELTANET  = 1,
    LAYER_MAMBA     = 2,   /* Mamba-1 (S6) */
    LAYER_MAMBA2    = 3    /* Mamba-2 (SSD) */
} layer_type;

/* ───── KV Cache (for attention layers) ───── */
typedef struct {
    gpu_buffer* k;  /* [n_attn_layers][max_seq][n_kv_heads * head_dim] */
    gpu_buffer* v;  /* [n_attn_layers][max_seq][n_kv_heads * head_dim] */
    uint32_t    seq_len;     /* current sequence length */
    uint32_t    max_seq;     /* maximum sequence length */
    uint32_t    n_layers;    /* total number of layers with KV cache */
} kv_cache;

/* ───── Recurrent State Cache (for DeltaNet + Mamba layers) ───── */
typedef struct {
    gpu_buffer* conv_state;  /* [n_recurrent_layers]: conv ring buffer */
    gpu_buffer* ssm_state;   /* [n_recurrent_layers]: SSM state matrix */
    uint32_t    n_recurrent_layers;  /* total DeltaNet + Mamba + Mamba2 layers */
} recurrent_cache;

/* ───── Per-Layer Weights (hybrid — supports attention, DeltaNet, Mamba, Mamba2) ───── */
typedef struct {
    layer_type type;
    
    /* Common weights (all layers) */
    gpu_buffer attn_norm;        /* pre-attention RMS norm [hidden_size] */
    gpu_buffer post_attn_norm;   /* post-attention norm (replaces ffn_norm for qwen35) [hidden_size] */
    gpu_buffer ffn_norm;         /* classic ffn_norm for non-qwen35 models [hidden_size] */
    gpu_buffer ffn_gate;         /* [hidden_size, intermediate_size] */
    gpu_buffer ffn_up;           /* [hidden_size, intermediate_size] */
    gpu_buffer ffn_down;         /* [intermediate_size, hidden_size] */
    
    /* ── Attention layers only ── */
    gpu_buffer wq;               /* [hidden_size, n_heads * head_dim] */
    gpu_buffer wk;               /* [hidden_size, n_kv_heads * head_dim] */
    gpu_buffer wv;               /* [hidden_size, n_kv_heads * head_dim] */
    gpu_buffer wo;               /* [n_heads * head_dim, hidden_size] */
    gpu_buffer attn_q_norm;      /* per-head Q norm [head_dim] (qwen35 attention) */
    gpu_buffer attn_k_norm;      /* per-head K norm [head_dim] (qwen35 attention) */
    
    /* ── DeltaNet layers only ── */
    gpu_buffer wqkv;             /* fused QKV: [hidden, head_k*nk*2 + head_v*nv] */
    gpu_buffer wqkv_gate;        /* gate Z: [hidden, d_inner] */
    gpu_buffer ssm_alpha;        /* [num_v_heads] */
    gpu_buffer ssm_beta;         /* [num_v_heads] */
    gpu_buffer ssm_a;            /* DeltaNet: [num_v_heads], Mamba1: [d_state, d_inner], Mamba2: [1, n_head] */
    gpu_buffer ssm_dt;           /* DeltaNet: [num_v_heads], Mamba1/2: used for bias storage */
    gpu_buffer ssm_conv1d;       /* DeltaNet: [conv_kernel * d_inner], Mamba1: [d_conv, d_inner], Mamba2: [d_conv, d_inner+2*ng*d_state] */
    gpu_buffer ssm_norm;         /* DeltaNet: [head_v_dim], Mamba2: [d_inner/n_group, n_group] */
    gpu_buffer ssm_out;          /* [d_inner, hidden_size] */
    
    /* ── Mamba layers (S6/Mamba-1 and Mamba-2/SSD) ── */
    gpu_buffer ssm_in;           /* [hidden, 2*d_inner] (Mamba1) or [hidden, d_in_proj] (Mamba2) */
    gpu_buffer ssm_conv1d_b;     /* conv1d bias [d_inner] or [d_inner+2*ng*d_state] */
    gpu_buffer ssm_x;            /* [d_inner, dt_rank + 2*d_state] (Mamba1 only: projects to dt,B,C) */
    gpu_buffer ssm_dt_w;         /* dt projection weight [dt_rank, d_inner] (Mamba1) */
    gpu_buffer ssm_dt_b;         /* dt bias [d_inner] (Mamba1) or [n_head] (Mamba2) */
    gpu_buffer ssm_d;            /* skip connection [d_inner] (Mamba1) or [1, n_head] (Mamba2) */
    /* Note: ssm_a, ssm_conv1d, ssm_norm, ssm_out are reused from DeltaNet fields above */
    
    /* Indices for cache lookup (which slot in KV/recurrent cache) */
    uint32_t cache_index;        /* index into kv_cache or recurrent_cache arrays */
} hybrid_layer;

/* ───── Model Weights (on GPU) ───── */
typedef struct {
    gpu_buffer  token_embd;       /* [vocab_size, hidden_size] */
    
    /* Per-layer weights (hybrid: each layer knows its type) */
    hybrid_layer* layers;
    
    gpu_buffer  output_norm;      /* [hidden_size] */
    gpu_buffer  output;           /* [hidden_size, vocab_size] (LM head) */
    
    uint32_t    n_layers;
    bool        is_hybrid;        /* true if model has mixed layer types */
} model_weights;

/* ───── Scratch Buffers (reused each forward pass) ───── */
typedef struct {
    gpu_buffer  hidden;      /* [seq_len, hidden_size] */
    gpu_buffer  residual;    /* [seq_len, hidden_size] */
    gpu_buffer  norm_out;    /* [seq_len, hidden_size] */
    gpu_buffer  q;           /* [seq_len, n_heads * head_dim] */
    gpu_buffer  k;           /* [seq_len, n_kv_heads * head_dim] */
    gpu_buffer  v;           /* [seq_len, n_kv_heads * head_dim] */
    gpu_buffer  attn_out;    /* [seq_len, n_heads * head_dim] */
    gpu_buffer  attn_scores; /* [n_heads, seq_len, full_seq_len] */
    gpu_buffer  ffn_gate_out;/* [seq_len, intermediate_size] */
    gpu_buffer  ffn_up_out;  /* [seq_len, intermediate_size] */
    gpu_buffer  ffn_down_out;/* [seq_len, hidden_size] */
    gpu_buffer  logits;      /* [vocab_size] (last token only for generation) */
    
    /* DeltaNet scratch */
    gpu_buffer  delta_qkv;   /* fused QKV output [head_k*nk*2 + head_v*nv] */
    gpu_buffer  delta_gate;  /* gate Z output [d_inner] */
    gpu_buffer  delta_conv;  /* conv1d output [d_inner] */
    gpu_buffer  delta_out;   /* DeltaNet layer output [d_inner] */
    
    /* Mamba scratch */
    gpu_buffer  mamba_in;    /* ssm_in projection output [2*d_inner] (Mamba1) or [d_in_proj] (Mamba2) */
    gpu_buffer  mamba_x;     /* x projection output [dt_rank + 2*d_state] (Mamba1) */
    gpu_buffer  mamba_dt;    /* dt after projection [d_inner] (Mamba1) or [n_head] (Mamba2) */
    gpu_buffer  mamba_B;     /* B vector [d_state] (Mamba1) or [n_group*d_state] (Mamba2) */
    gpu_buffer  mamba_C;     /* C vector [d_state] (Mamba1) or [n_group*d_state] (Mamba2) */
    gpu_buffer  mamba_y;     /* scan output [d_inner] */
} scratch_buffers;

/* ───── Sampling Parameters ───── */
typedef struct {
    float    temperature;   /* 0.0 = greedy, default 0.7 */
    float    top_p;         /* nucleus sampling, default 0.9 */
    uint32_t top_k;         /* top-k sampling, default 40 */
    float    repeat_penalty;/* repetition penalty, default 1.1 */
    uint32_t max_tokens;    /* max tokens to generate */
    uint32_t seed;          /* RNG seed, 0 = random */
} sample_params;

/* ───── Engine ───── */
typedef struct {
    vk_context      vk;
    gguf_file*      gguf;
    model_arch      arch;
    model_weights   weights;
    kv_cache        cache;           /* KV cache for attention layers */
    recurrent_cache recurrent;       /* recurrent state for DeltaNet + Mamba layers */
    scratch_buffers scratch;
    
    /* Tokenizer (BPE) */
    tokenizer   tok;
    char**          vocab;
    float*          vocab_scores;
    uint32_t        vocab_size;
    
    bool            loaded;
} engine;

/* ───── API ───── */

/* Initialize engine (Vulkan + empty state) */
bool engine_init(engine* eng, const char* shader_dir);

/* Load a GGUF model file */
bool engine_load_model(engine* eng, const char* model_path);

/* Allocate KV cache for given max sequence length */
bool engine_alloc_cache(engine* eng, uint32_t max_seq_len);

/* Run one forward pass: tokens → logits */
bool engine_forward(engine* eng, const uint32_t* tokens, uint32_t n_tokens);

/* Generate tokens autoregressively */
typedef void (*token_callback)(uint32_t token_id, const char* token_str, void* user_data);
uint32_t engine_generate(engine* eng, const uint32_t* prompt, uint32_t prompt_len,
                         const sample_params* params, token_callback cb, void* user_data);

/* Tokenize a string (BPE) */
uint32_t* engine_tokenize(const engine* eng, const char* text, uint32_t* n_tokens);

/* Detokenize a token ID */
const char* engine_detokenize(const engine* eng, uint32_t token_id);

/* Reset KV cache (for new conversation) */
void engine_reset(engine* eng);

/* Free everything */
void engine_destroy(engine* eng);

/* Print engine status */
void engine_print_status(const engine* eng);

#endif /* ENGINE_H */
