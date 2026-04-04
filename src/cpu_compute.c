/*
 * Artifact Engine — CPU Compute Backend
 *
 * Drop-in replacement for vulkan_compute.c.
 * Implements the same API using CPU-only operations.
 * Used for platforms without Vulkan (Xbox UWP, embedded).
 *
 * Compile with -DCPU_ONLY to exclude Vulkan entirely.
 */

#ifdef CPU_ONLY

#include "../include/vulkan_compute.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ───── CPU "GPU" Buffer — just malloc'd memory ───── */

bool vk_init(vk_context* ctx, const char* shader_dir) {
    (void)shader_dir;
    memset(ctx, 0, sizeof(*ctx));
    ctx->device_memory_size = 0;
    ctx->used_memory = 0;
    ctx->max_buffer_size = (size_t)16 * 1024 * 1024 * 1024;  /* 16 GB */
    snprintf(ctx->device_name, sizeof(ctx->device_name), "CPU (Artifact Engine)");
    ctx->initialized = true;
    printf("cpu: initialized (no GPU, using CPU fallback)\n");
    printf("cpu: all tensor operations will run on CPU\n");
    return true;
}

void vk_destroy(vk_context* ctx) {
    if (ctx) ctx->initialized = false;
}

bool vk_alloc_device(vk_context* ctx, gpu_buffer* buf, size_t size) {
    buf->mapped = malloc(size);
    if (!buf->mapped) {
        fprintf(stderr, "cpu: failed to allocate %zu bytes\n", size);
        return false;
    }
    memset(buf->mapped, 0, size);
    buf->size = size;
    buf->buffer = NULL;
    buf->memory = NULL;
    ctx->used_memory += size;
    return true;
}

bool vk_alloc_staging(vk_context* ctx, gpu_buffer* buf, size_t size) {
    return vk_alloc_device(ctx, buf, size);
}

bool vk_upload(vk_context* ctx, gpu_buffer* dst, const void* src, size_t size) {
    (void)ctx;
    if (!dst->mapped || size > dst->size) return false;
    memcpy(dst->mapped, src, size);
    return true;
}

bool vk_download(vk_context* ctx, void* dst, const gpu_buffer* src, size_t size) {
    (void)ctx;
    if (!src->mapped || size > src->size) return false;
    memcpy(dst, src->mapped, size);
    return true;
}

void vk_free_buffer(vk_context* ctx, gpu_buffer* buf) {
    if (buf->mapped) {
        ctx->used_memory -= buf->size;
        free(buf->mapped);
        buf->mapped = NULL;
    }
    buf->size = 0;
}

/* ───── Compute Dispatch (no-ops on CPU) ───── */

void vk_begin_compute(vk_context* ctx) { (void)ctx; }
void vk_barrier(vk_context* ctx) { (void)ctx; }

void vk_dispatch(vk_context* ctx, shader_id shader,
                 const gpu_buffer* buffers[], uint32_t n_buffers,
                 const push_constants* pc,
                 uint32_t gx, uint32_t gy, uint32_t gz) {
    (void)ctx; (void)shader; (void)buffers; (void)n_buffers;
    (void)pc; (void)gx; (void)gy; (void)gz;
}

bool vk_submit_and_wait(vk_context* ctx) { (void)ctx; return true; }

/* ───── fp16 helper ───── */

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        float f = (float)mant / 1024.0f * (1.0f / 16384.0f);
        return sign ? -f : f;
    }
    if (exp == 31) return sign ? -INFINITY : INFINITY;
    float f = (1.0f + (float)mant / 1024.0f) * powf(2.0f, (float)exp - 15.0f);
    return sign ? -f : f;
}

/* ───── Q4_K Dequantization ───── */

static void dequant_q4_k_block(const uint8_t* block, float* out, int n) {
    uint16_t d_h, dmin_h;
    memcpy(&d_h, block, 2);
    memcpy(&dmin_h, block + 2, 2);
    float d = fp16_to_fp32(d_h);
    float dmin = fp16_to_fp32(dmin_h);
    const uint8_t* scales = block + 4;
    const uint8_t* qs = block + 16;
    
    for (int sb = 0; sb < 8 && sb * 32 < n; sb++) {
        uint8_t sc, m;
        if (sb < 4) {
            sc = scales[sb] & 0x3f;
            m  = scales[sb + 4] & 0x3f;
        } else {
            sc = ((scales[sb + 4] & 0xF0) >> 4) | ((scales[sb - 4] >> 6) << 4);
            m  = ((scales[sb + 4] & 0x0F))       | ((scales[sb]     >> 6) << 4);
        }
        float block_d = d * sc;
        float block_m = dmin * m;
        for (int j = 0; j < 32 && sb * 32 + j < n; j++) {
            int idx = sb * 32 + j;
            int byte_idx = idx / 2;
            uint8_t q = (idx % 2 == 0) ? (qs[byte_idx] & 0x0F) : ((qs[byte_idx] >> 4) & 0x0F);
            out[idx] = block_d * q - block_m;
        }
    }
}

/* ───── High-Level Operations (matching vulkan_compute.h signatures exactly) ───── */

void vk_matmul(vk_context* ctx,
               const gpu_buffer* A, const gpu_buffer* B, gpu_buffer* C,
               uint32_t M, uint32_t N, uint32_t K) {
    (void)ctx;
    const float* a = (const float*)A->mapped;
    const float* b = (const float*)B->mapped;
    float* c = (float*)C->mapped;
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; k++) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

void vk_matmul_q4k(vk_context* ctx,
                    const gpu_buffer* A_quant, const gpu_buffer* B, gpu_buffer* C,
                    uint32_t M, uint32_t N, uint32_t K) {
    (void)ctx;
    float* a_row = (float*)malloc(K * sizeof(float));
    if (!a_row) return;
    const uint8_t* a_data = (const uint8_t*)A_quant->mapped;
    const float* b = (const float*)B->mapped;
    float* c = (float*)C->mapped;
    size_t block_size = 144;
    size_t elems_per_block = 256;
    size_t blocks_per_row = (K + elems_per_block - 1) / elems_per_block;
    size_t row_bytes = blocks_per_row * block_size;
    for (uint32_t i = 0; i < M; i++) {
        const uint8_t* row_data = a_data + i * row_bytes;
        for (size_t blk = 0; blk < blocks_per_row; blk++) {
            int remaining = (int)(K - blk * elems_per_block);
            if (remaining > (int)elems_per_block) remaining = (int)elems_per_block;
            dequant_q4_k_block(row_data + blk * block_size,
                               a_row + blk * elems_per_block, remaining);
        }
        for (uint32_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; k++) sum += a_row[k] * b[k * N + j];
            c[i * N + j] = sum;
        }
    }
    free(a_row);
}

/* RMS normalization: out = x * rsqrt(mean(x^2) + eps) * weight */
void vk_rmsnorm(vk_context* ctx,
                const gpu_buffer* x, const gpu_buffer* weight, gpu_buffer* out,
                uint32_t n_elements, float eps) {
    (void)ctx;
    const float* xd = (const float*)x->mapped;
    const float* wd = (const float*)weight->mapped;
    float* od = (float*)out->mapped;
    float ss = 0.0f;
    for (uint32_t i = 0; i < n_elements; i++) ss += xd[i] * xd[i];
    float rms = 1.0f / sqrtf(ss / n_elements + eps);
    for (uint32_t i = 0; i < n_elements; i++) od[i] = xd[i] * rms * wd[i];
}

/* Rotary position embedding — applies to both Q and K in-place */
void vk_rope(vk_context* ctx,
             gpu_buffer* q, gpu_buffer* k,
             uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads,
             uint32_t position, float freq_base) {
    (void)ctx;
    /* Apply RoPE to Q */
    float* qd = (float*)q->mapped;
    for (uint32_t h = 0; h < n_heads; h++) {
        for (uint32_t i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(freq_base, 2.0f * i / head_dim);
            float theta = position * freq;
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);
            uint32_t idx = h * head_dim + i * 2;
            float x0 = qd[idx], x1 = qd[idx + 1];
            qd[idx]     = x0 * cos_t - x1 * sin_t;
            qd[idx + 1] = x0 * sin_t + x1 * cos_t;
        }
    }
    /* Apply RoPE to K */
    float* kd = (float*)k->mapped;
    for (uint32_t h = 0; h < n_kv_heads; h++) {
        for (uint32_t i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(freq_base, 2.0f * i / head_dim);
            float theta = position * freq;
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);
            uint32_t idx = h * head_dim + i * 2;
            float x0 = kd[idx], x1 = kd[idx + 1];
            kd[idx]     = x0 * cos_t - x1 * sin_t;
            kd[idx + 1] = x0 * sin_t + x1 * cos_t;
        }
    }
}

/* Softmax along last dimension */
void vk_softmax(vk_context* ctx, gpu_buffer* x, uint32_t rows, uint32_t cols) {
    (void)ctx;
    float* xd = (float*)x->mapped;
    for (uint32_t r = 0; r < rows; r++) {
        float* row = xd + r * cols;
        float max_val = row[0];
        for (uint32_t i = 1; i < cols; i++)
            if (row[i] > max_val) max_val = row[i];
        float sum = 0.0f;
        for (uint32_t i = 0; i < cols; i++) {
            row[i] = expf(row[i] - max_val);
            sum += row[i];
        }
        for (uint32_t i = 0; i < cols; i++) row[i] /= sum;
    }
}

/* SiLU activation: x = x * sigmoid(x) — in-place */
void vk_silu(vk_context* ctx, gpu_buffer* x, uint32_t n_elements) {
    (void)ctx;
    float* xd = (float*)x->mapped;
    for (uint32_t i = 0; i < n_elements; i++)
        xd[i] = xd[i] / (1.0f + expf(-xd[i]));
}

/* Element-wise add: out = a + b */
void vk_add(vk_context* ctx,
            const gpu_buffer* a, const gpu_buffer* b, gpu_buffer* out,
            uint32_t n_elements) {
    (void)ctx;
    const float* ad = (const float*)a->mapped;
    const float* bd = (const float*)b->mapped;
    float* od = (float*)out->mapped;
    for (uint32_t i = 0; i < n_elements; i++) od[i] = ad[i] + bd[i];
}

/* Element-wise multiply: out = a * b */
void vk_mul(vk_context* ctx,
            const gpu_buffer* a, const gpu_buffer* b, gpu_buffer* out,
            uint32_t n_elements) {
    (void)ctx;
    const float* ad = (const float*)a->mapped;
    const float* bd = (const float*)b->mapped;
    float* od = (float*)out->mapped;
    for (uint32_t i = 0; i < n_elements; i++) od[i] = ad[i] * bd[i];
}

/* Token embedding lookup */
void vk_embedding(vk_context* ctx,
                  const gpu_buffer* table, gpu_buffer* out,
                  uint32_t token_id, uint32_t dim) {
    (void)ctx;
    const float* td = (const float*)table->mapped;
    float* od = (float*)out->mapped;
    memcpy(od, td + (size_t)token_id * dim, dim * sizeof(float));
}

/* Store K/V vectors into KV cache at given position */
void vk_kv_cache_store(vk_context* ctx,
                       const gpu_buffer* kv_current, gpu_buffer* kv_cache,
                       uint32_t kv_dim, uint32_t pos, uint32_t max_seq) {
    (void)ctx;
    (void)max_seq;
    float* cd = (float*)kv_cache->mapped;
    const float* kvd = (const float*)kv_current->mapped;
    memcpy(cd + (size_t)pos * kv_dim, kvd, kv_dim * sizeof(float));
}

/* Grouped-Query Attention with KV cache */
void vk_gqa_attention(vk_context* ctx,
                      const gpu_buffer* q,
                      const gpu_buffer* k_cache,
                      const gpu_buffer* v_cache,
                      gpu_buffer* attn_scores,
                      gpu_buffer* attn_out,
                      uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads,
                      uint32_t seq_len, uint32_t max_seq, uint32_t current_pos) {
    (void)ctx;
    (void)attn_scores;  /* we use our own local buffer */
    (void)max_seq;
    
    const float* qd = (const float*)q->mapped;
    const float* kd = (const float*)k_cache->mapped;
    const float* vd = (const float*)v_cache->mapped;
    float* od = (float*)attn_out->mapped;
    
    uint32_t kv_group = n_heads / n_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    uint32_t pos_len = current_pos + 1;
    if (pos_len > seq_len) pos_len = seq_len;
    
    float* scores = (float*)malloc(pos_len * sizeof(float));
    if (!scores) return;
    
    for (uint32_t h = 0; h < n_heads; h++) {
        uint32_t kv_h = h / kv_group;
        const float* q_head = qd + h * head_dim;
        
        float max_score = -INFINITY;
        for (uint32_t s = 0; s < pos_len; s++) {
            const float* k_head = kd + (size_t)s * n_kv_heads * head_dim + kv_h * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += q_head[d] * k_head[d];
            scores[s] = dot * scale;
            if (scores[s] > max_score) max_score = scores[s];
        }
        
        float sum = 0.0f;
        for (uint32_t s = 0; s < pos_len; s++) {
            scores[s] = expf(scores[s] - max_score);
            sum += scores[s];
        }
        for (uint32_t s = 0; s < pos_len; s++) scores[s] /= sum;
        
        float* o_head = od + h * head_dim;
        memset(o_head, 0, head_dim * sizeof(float));
        for (uint32_t s = 0; s < pos_len; s++) {
            const float* v_head = vd + (size_t)s * n_kv_heads * head_dim + kv_h * head_dim;
            for (uint32_t d = 0; d < head_dim; d++) o_head[d] += scores[s] * v_head[d];
        }
    }
    
    free(scores);
}

/* ─── Diagnostics ─── */

void vk_print_info(const vk_context* ctx) {
    printf("vk: Backend: CPU-only (no GPU)\n");
    printf("vk: Memory used: %.2f MB\n", (double)ctx->used_memory / (1024 * 1024));
}

size_t vk_memory_used(const vk_context* ctx) { return ctx->used_memory; }
size_t vk_memory_total(const vk_context* ctx) { return ctx->device_memory_size; }

#endif /* CPU_ONLY */
