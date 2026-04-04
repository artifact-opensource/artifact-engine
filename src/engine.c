/*
 * Artifact Engine — Transformer Forward Pass
 *
 * Loads model weights to GPU, runs the transformer layer by layer,
 * generates tokens autoregressively.
 */

#define _GNU_SOURCE
#include "../include/engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

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
    
    if (!vk_alloc_device(&eng->vk, dst, ti->n_bytes)) {
        fprintf(stderr, "engine: failed to allocate GPU buffer for '%s' (%zu bytes)\n",
                name, (size_t)ti->n_bytes);
        return false;
    }
    
    if (!vk_upload(&eng->vk, dst, data, ti->n_bytes)) {
        fprintf(stderr, "engine: failed to upload '%s' to GPU\n", name);
        return false;
    }
    
    return true;
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
    
    printf("engine: uploading %u layers to GPU...\n", eng->arch.n_layers);
    
    /* Allocate layer array */
    eng->weights.n_layers = eng->arch.n_layers;
    eng->weights.layers = calloc(eng->arch.n_layers, sizeof(*eng->weights.layers));
    if (!eng->weights.layers) return false;
    
    /* Upload token embeddings */
    if (!upload_tensor(eng, &eng->weights.token_embd, "token_embd.weight", true))
        return false;
    
    /* Upload per-layer weights */
    for (uint32_t l = 0; l < eng->arch.n_layers; l++) {
        char name[128];
        
        #define UPLOAD_LAYER(field, tensor_name) \
            snprintf(name, sizeof(name), "blk.%u.%s", l, tensor_name); \
            if (!upload_tensor(eng, &eng->weights.layers[l].field, name, true)) return false;
        
        UPLOAD_LAYER(attn_norm, "attn_norm.weight");
        UPLOAD_LAYER(wq,        "attn_q.weight");
        UPLOAD_LAYER(wk,        "attn_k.weight");
        UPLOAD_LAYER(wv,        "attn_v.weight");
        UPLOAD_LAYER(wo,        "attn_output.weight");
        UPLOAD_LAYER(ffn_norm,  "ffn_norm.weight");
        UPLOAD_LAYER(ffn_gate,  "ffn_gate.weight");
        UPLOAD_LAYER(ffn_up,    "ffn_up.weight");
        UPLOAD_LAYER(ffn_down,  "ffn_down.weight");
        
        #undef UPLOAD_LAYER
        
        if ((l + 1) % 4 == 0 || l == eng->arch.n_layers - 1) {
            printf("  layer %u/%u — VRAM: %.2f GB / %.2f GB\n",
                   l + 1, eng->arch.n_layers,
                   (double)vk_memory_used(&eng->vk) / (1024.0*1024.0*1024.0),
                   (double)vk_memory_total(&eng->vk) / (1024.0*1024.0*1024.0));
        }
    }
    
    /* Upload output norm + LM head */
    if (!upload_tensor(eng, &eng->weights.output_norm, "output_norm.weight", true))
        return false;
    
    /* output.weight might be tied to token_embd.weight */
    if (!upload_tensor(eng, &eng->weights.output, "output.weight", false)) {
        /* Weight tying: reuse token embeddings */
        printf("engine: output.weight not found — using weight tying\n");
        eng->weights.output = eng->weights.token_embd;
    }
    
    /* Load tokenizer vocabulary */
    const gguf_kv* tokens_kv = gguf_find_kv(eng->gguf, "tokenizer.ggml.tokens");
    if (tokens_kv && tokens_kv->type == GGUF_TYPE_ARRAY) {
        eng->vocab_size = (uint32_t)tokens_kv->value.arr.count;
        /* Token strings are in the GGUF mmap — we'll access them lazily */
        /* For MVP, we use token IDs directly */
        printf("engine: vocabulary size: %u tokens\n", eng->vocab_size);
    } else {
        eng->vocab_size = eng->arch.vocab_size;
    }
    
    eng->loaded = true;
    printf("engine: model loaded — %.2f GB VRAM used\n",
           (double)vk_memory_used(&eng->vk) / (1024.0*1024.0*1024.0));
    
    return true;
}

/* ───── KV Cache Allocation ───── */

bool engine_alloc_cache(engine* eng, uint32_t max_seq_len) {
    eng->cache.n_layers = eng->arch.n_layers;
    eng->cache.max_seq = max_seq_len;
    eng->cache.seq_len = 0;
    
    /* Per layer: K and V caches */
    size_t kv_size = (size_t)max_seq_len * eng->arch.n_kv_heads * eng->arch.head_dim * sizeof(float);
    
    eng->cache.k = calloc(eng->arch.n_layers, sizeof(gpu_buffer));
    eng->cache.v = calloc(eng->arch.n_layers, sizeof(gpu_buffer));
    
    for (uint32_t l = 0; l < eng->arch.n_layers; l++) {
        if (!vk_alloc_device(&eng->vk, &eng->cache.k[l], kv_size)) return false;
        if (!vk_alloc_device(&eng->vk, &eng->cache.v[l], kv_size)) return false;
    }
    
    /* Allocate scratch buffers (sized for single-token generation) */
    uint32_t H = eng->arch.hidden_size;
    uint32_t I = eng->arch.intermediate_size;
    uint32_t V = eng->vocab_size;
    
    vk_alloc_device(&eng->vk, &eng->scratch.hidden,       H * sizeof(float));
    vk_alloc_device(&eng->vk, &eng->scratch.residual,     H * sizeof(float));
    vk_alloc_device(&eng->vk, &eng->scratch.norm_out,     H * sizeof(float));
    vk_alloc_device(&eng->vk, &eng->scratch.q,            eng->arch.n_heads * eng->arch.head_dim * sizeof(float));
    vk_alloc_device(&eng->vk, &eng->scratch.k,            eng->arch.n_kv_heads * eng->arch.head_dim * sizeof(float));
    vk_alloc_device(&eng->vk, &eng->scratch.v,            eng->arch.n_kv_heads * eng->arch.head_dim * sizeof(float));
    vk_alloc_device(&eng->vk, &eng->scratch.attn_out,     H * sizeof(float));
    vk_alloc_device(&eng->vk, &eng->scratch.attn_scores,  eng->arch.n_heads * max_seq_len * sizeof(float));
    vk_alloc_device(&eng->vk, &eng->scratch.ffn_gate_out, I * sizeof(float));
    vk_alloc_device(&eng->vk, &eng->scratch.ffn_up_out,   I * sizeof(float));
    vk_alloc_device(&eng->vk, &eng->scratch.ffn_down_out, H * sizeof(float));
    vk_alloc_device(&eng->vk, &eng->scratch.logits,       V * sizeof(float));
    
    printf("engine: KV cache allocated (ctx=%u) — total VRAM: %.2f GB / %.2f GB\n",
           max_seq_len,
           (double)vk_memory_used(&eng->vk) / (1024.0*1024.0*1024.0),
           (double)vk_memory_total(&eng->vk) / (1024.0*1024.0*1024.0));
    
    return true;
}

/* ───── Single Layer Forward Pass ───── */

static void forward_layer(engine* eng, uint32_t layer, uint32_t pos) {
    uint32_t H = eng->arch.hidden_size;
    uint32_t I = eng->arch.intermediate_size;
    uint32_t n_heads = eng->arch.n_heads;
    uint32_t n_kv = eng->arch.n_kv_heads;
    uint32_t hd = eng->arch.head_dim;
    
    /* 1. Attention norm */
    vk_rmsnorm(&eng->vk, &eng->scratch.hidden, &eng->weights.layers[layer].attn_norm,
               &eng->scratch.norm_out, H, eng->arch.rms_norm_eps);
    vk_barrier(&eng->vk);
    
    /* 2. QKV projections */
    vk_matmul(&eng->vk, &eng->scratch.norm_out, &eng->weights.layers[layer].wq,
              &eng->scratch.q, 1, n_heads * hd, H);
    vk_matmul(&eng->vk, &eng->scratch.norm_out, &eng->weights.layers[layer].wk,
              &eng->scratch.k, 1, n_kv * hd, H);
    vk_matmul(&eng->vk, &eng->scratch.norm_out, &eng->weights.layers[layer].wv,
              &eng->scratch.v, 1, n_kv * hd, H);
    vk_barrier(&eng->vk);
    
    /* 3. RoPE */
    vk_rope(&eng->vk, &eng->scratch.q, &eng->scratch.k,
            hd, n_heads, n_kv, pos, eng->arch.rope_freq_base);
    vk_barrier(&eng->vk);
    
    /* 4. Store K, V into cache at position `pos` */
    /* TODO: vk_copy into cache.k[layer] and cache.v[layer] at offset pos */
    
    /* 5. Attention scores: for each head, Q @ K^T / sqrt(d_k) */
    /* TODO: multi-head attention with GQA support */
    /* For MVP: simplified single-head-at-a-time attention */
    
    /* 6. Softmax attention weights */
    /* TODO: causal mask */
    
    /* 7. Attention output = weights @ V */
    /* TODO */
    
    /* 8. Output projection */
    vk_matmul(&eng->vk, &eng->scratch.attn_out, &eng->weights.layers[layer].wo,
              &eng->scratch.norm_out, 1, H, n_heads * hd);
    vk_barrier(&eng->vk);
    
    /* 9. Residual connection (hidden += attn_output) */
    vk_add(&eng->vk, &eng->scratch.hidden, &eng->scratch.norm_out,
            &eng->scratch.hidden, H);
    vk_barrier(&eng->vk);
    
    /* 10. FFN norm */
    vk_rmsnorm(&eng->vk, &eng->scratch.hidden, &eng->weights.layers[layer].ffn_norm,
               &eng->scratch.norm_out, H, eng->arch.rms_norm_eps);
    vk_barrier(&eng->vk);
    
    /* 11. FFN: gate = SiLU(x @ W_gate), up = x @ W_up, down = (gate * up) @ W_down */
    vk_matmul(&eng->vk, &eng->scratch.norm_out, &eng->weights.layers[layer].ffn_gate,
              &eng->scratch.ffn_gate_out, 1, I, H);
    vk_matmul(&eng->vk, &eng->scratch.norm_out, &eng->weights.layers[layer].ffn_up,
              &eng->scratch.ffn_up_out, 1, I, H);
    vk_barrier(&eng->vk);
    
    /* SiLU on gate */
    vk_silu(&eng->vk, &eng->scratch.ffn_gate_out, I);
    vk_barrier(&eng->vk);
    
    /* gate * up */
    vk_mul(&eng->vk, &eng->scratch.ffn_gate_out, &eng->scratch.ffn_up_out,
            &eng->scratch.ffn_gate_out, I);
    vk_barrier(&eng->vk);
    
    /* down projection */
    vk_matmul(&eng->vk, &eng->scratch.ffn_gate_out, &eng->weights.layers[layer].ffn_down,
              &eng->scratch.ffn_down_out, 1, H, I);
    vk_barrier(&eng->vk);
    
    /* Residual connection */
    vk_add(&eng->vk, &eng->scratch.hidden, &eng->scratch.ffn_down_out,
            &eng->scratch.hidden, H);
}

/* ───── Full Forward Pass ───── */

bool engine_forward(engine* eng, const uint32_t* tokens, uint32_t n_tokens) {
    /* For generation, we process one token at a time (with KV cache) */
    /* For prompt encoding, we'd process all tokens (TODO: batch mode) */
    
    uint32_t pos = eng->cache.seq_len;
    
    for (uint32_t t = 0; t < n_tokens; t++) {
        /* Embed token */
        vk_begin_compute(&eng->vk);
        vk_embedding(&eng->vk, &eng->weights.token_embd, &eng->scratch.hidden,
                     tokens[t], eng->arch.hidden_size);
        vk_barrier(&eng->vk);
        
        /* Run through all layers */
        for (uint32_t l = 0; l < eng->arch.n_layers; l++) {
            forward_layer(eng, l, pos + t);
        }
        
        /* Final RMS norm */
        vk_rmsnorm(&eng->vk, &eng->scratch.hidden, &eng->weights.output_norm,
                   &eng->scratch.norm_out, eng->arch.hidden_size, eng->arch.rms_norm_eps);
        vk_barrier(&eng->vk);
        
        /* LM head: logits = norm_out @ output_weight^T */
        vk_matmul(&eng->vk, &eng->scratch.norm_out, &eng->weights.output,
                  &eng->scratch.logits, 1, eng->vocab_size, eng->arch.hidden_size);
        
        vk_submit_and_wait(&eng->vk);
    }
    
    eng->cache.seq_len += n_tokens;
    return true;
}

/* ───── Token Sampling (CPU-side) ───── */

static uint32_t sample_token(float* logits, uint32_t vocab_size, const sample_params* params) {
    /* Temperature */
    if (params->temperature > 0.0f && params->temperature != 1.0f) {
        float inv_temp = 1.0f / params->temperature;
        for (uint32_t i = 0; i < vocab_size; i++) {
            logits[i] *= inv_temp;
        }
    }
    
    /* Greedy (temperature = 0) */
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
    
    /* Softmax */
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
    
    /* Top-k filtering */
    if (params->top_k > 0 && params->top_k < vocab_size) {
        /* Simple approach: zero out everything below top-k threshold */
        /* TODO: proper top-k with partial sort */
    }
    
    /* Top-p (nucleus) sampling */
    /* TODO: sort by probability, cumsum, cutoff at top_p */
    
    /* Sample from distribution */
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
    
    /* Process prompt (prefill) */
    printf("engine: prefilling %u tokens...\n", prompt_len);
    engine_forward(eng, prompt, prompt_len);
    
    /* Allocate CPU buffer for logits download */
    float* logits_cpu = malloc(eng->vocab_size * sizeof(float));
    if (!logits_cpu) return 0;
    
    uint32_t generated = 0;
    uint32_t token = 0;
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (uint32_t i = 0; i < params->max_tokens; i++) {
        /* Download logits from GPU */
        vk_download(&eng->vk, logits_cpu, &eng->scratch.logits,
                    eng->vocab_size * sizeof(float));
        
        /* Sample next token */
        token = sample_token(logits_cpu, eng->vocab_size, params);
        
        /* Check for EOS */
        if (token == eng->arch.eos_token_id) break;
        
        generated++;
        
        /* Callback */
        if (cb) {
            const char* token_str = engine_detokenize(eng, token);
            cb(token, token_str ? token_str : "?", user_data);
        }
        
        /* Forward pass for next token */
        engine_forward(eng, &token, 1);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("engine: generated %u tokens in %.2fs (%.1f tok/s)\n",
           generated, elapsed, generated / elapsed);
    
    free(logits_cpu);
    return generated;
}

/* ───── Tokenization (BPE) ───── */

/* MVP tokenizer: byte-level encoding. 
 * Full BPE requires loading merge rules from GGUF — TODO */
uint32_t* engine_tokenize(const engine* eng, const char* text, uint32_t* n_tokens) {
    /* For MVP: simple byte-level tokenization 
     * Real tokenizer needs BPE merge table from tokenizer.ggml.merges */
    size_t len = strlen(text);
    uint32_t* tokens = malloc((len + 2) * sizeof(uint32_t));
    
    /* Add BOS token */
    tokens[0] = eng->arch.bos_token_id;
    uint32_t count = 1;
    
    /* Simple: each byte maps to its vocab index (works for byte-level BPE models) */
    for (size_t i = 0; i < len; i++) {
        tokens[count++] = (uint32_t)(unsigned char)text[i];
    }
    
    *n_tokens = count;
    return tokens;
}

const char* engine_detokenize(const engine* eng, uint32_t token_id) {
    /* MVP: return "?" for unknown tokens */
    /* Real implementation reads from tokenizer.ggml.tokens array in GGUF */
    (void)eng;
    (void)token_id;
    
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
    /* KV cache buffers stay allocated, just reset position */
}

/* ───── Status ───── */

void engine_print_status(const engine* eng) {
    printf("\n═══ Artifact Engine Status ═══\n");
    printf("Model:      %s\n", eng->arch.arch);
    printf("Layers:     %u\n", eng->arch.n_layers);
    printf("Hidden:     %u\n", eng->arch.hidden_size);
    printf("Heads:      %u (KV: %u)\n", eng->arch.n_heads, eng->arch.n_kv_heads);
    printf("FFN:        %u\n", eng->arch.intermediate_size);
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
    
    /* Free KV cache */
    for (uint32_t l = 0; l < eng->cache.n_layers; l++) {
        vk_free_buffer(&eng->vk, &eng->cache.k[l]);
        vk_free_buffer(&eng->vk, &eng->cache.v[l]);
    }
    free(eng->cache.k);
    free(eng->cache.v);
    
    /* Free model weights */
    vk_free_buffer(&eng->vk, &eng->weights.token_embd);
    for (uint32_t l = 0; l < eng->weights.n_layers; l++) {
        vk_free_buffer(&eng->vk, &eng->weights.layers[l].attn_norm);
        vk_free_buffer(&eng->vk, &eng->weights.layers[l].wq);
        vk_free_buffer(&eng->vk, &eng->weights.layers[l].wk);
        vk_free_buffer(&eng->vk, &eng->weights.layers[l].wv);
        vk_free_buffer(&eng->vk, &eng->weights.layers[l].wo);
        vk_free_buffer(&eng->vk, &eng->weights.layers[l].ffn_norm);
        vk_free_buffer(&eng->vk, &eng->weights.layers[l].ffn_gate);
        vk_free_buffer(&eng->vk, &eng->weights.layers[l].ffn_up);
        vk_free_buffer(&eng->vk, &eng->weights.layers[l].ffn_down);
    }
    free(eng->weights.layers);
    vk_free_buffer(&eng->vk, &eng->weights.output_norm);
    /* Only free output if not tied to token_embd */
    if (eng->weights.output.buffer != eng->weights.token_embd.buffer) {
        vk_free_buffer(&eng->vk, &eng->weights.output);
    }
    
    /* Free GGUF */
    gguf_free(eng->gguf);
    
    /* Destroy Vulkan */
    vk_destroy(&eng->vk);
    
    eng->loaded = false;
}
