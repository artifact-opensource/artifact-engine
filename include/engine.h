/*
 * Artifact Engine — Model & Inference Interface
 * 
 * The transformer forward pass and token generation.
 */

#ifndef ENGINE_H
#define ENGINE_H

#include "gguf.h"
#include "tokenizer.h"
#include "vulkan_compute.h"
#include <stdbool.h>
#include <stdint.h>

/* ───── KV Cache ───── */
typedef struct {
    gpu_buffer* k;  /* [n_layers][max_seq][n_kv_heads * head_dim] */
    gpu_buffer* v;  /* [n_layers][max_seq][n_kv_heads * head_dim] */
    uint32_t    seq_len;     /* current sequence length */
    uint32_t    max_seq;     /* maximum sequence length */
    uint32_t    n_layers;
} kv_cache;

/* ───── Model Weights (on GPU) ───── */
typedef struct {
    gpu_buffer  token_embd;       /* [vocab_size, hidden_size] */
    
    /* Per-layer weights */
    struct {
        gpu_buffer attn_norm;     /* [hidden_size] */
        gpu_buffer wq;            /* [hidden_size, n_heads * head_dim] */
        gpu_buffer wk;            /* [hidden_size, n_kv_heads * head_dim] */
        gpu_buffer wv;            /* [hidden_size, n_kv_heads * head_dim] */
        gpu_buffer wo;            /* [n_heads * head_dim, hidden_size] */
        gpu_buffer ffn_norm;      /* [hidden_size] */
        gpu_buffer ffn_gate;      /* [hidden_size, intermediate_size] */
        gpu_buffer ffn_up;        /* [hidden_size, intermediate_size] */
        gpu_buffer ffn_down;      /* [intermediate_size, hidden_size] */
    } *layers;
    
    gpu_buffer  output_norm;      /* [hidden_size] */
    gpu_buffer  output;           /* [hidden_size, vocab_size] (LM head) */
    
    uint32_t    n_layers;
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
    kv_cache        cache;
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
