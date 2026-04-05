/*
 * Artifact Engine — CPU Compute Backend
 *
 * Drop-in replacement for vulkan_compute.c.
 * Implements the same API using CPU-only operations.
 * Used for platforms without Vulkan (Xbox UWP, embedded).
 *
 * Compile with -DCPU_ONLY to exclude Vulkan entirely.
 *
 * Includes DeltaNet and Mamba operations for hybrid architecture support.
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
    buf->buffer = (void*)0x1;  /* sentinel: "owned by us, free on cleanup" */
    buf->memory = NULL;
    buf->dtype = 0;   /* default F32 */
    buf->n_rows = 0;
    buf->n_cols = 0;
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
    if (buf->mapped && buf->buffer == (void*)0x1) {
        /* Only free buffers we allocated (not mmap'd pointers) */
        ctx->used_memory -= buf->size;
        free(buf->mapped);
    }
    buf->mapped = NULL;
    buf->buffer = NULL;
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

/* ════════════════════════════════════════════════════════════════════
 * HIGH-LEVEL OPERATIONS — matching vulkan_compute.h signatures
 * ════════════════════════════════════════════════════════════════════ */

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

/* ───── Q3_K Dequantization ───── */
/* Q3_K block: 256 elements, 110 bytes
 * Layout: [2B hmask] [32B qs (2-bit)] [64B signs/hmask] [12B scales] [2B d_fp16]
 * Actually: struct { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; uint16_t d; }
 * hmask = high bit mask, qs = low 2 bits, scales = 4-bit sub-block scales
 * Total per block = 32 + 64 + 12 + 2 = 110 bytes */

static void dequant_q3_k_block(const uint8_t* block, float* out, int n) {
    /* Q3_K layout (110 bytes for 256 elements):
     * Bytes 0..31:   hmask[32]  — high bit for each element
     * Bytes 32..95:  qs[64]     — packed 2-bit quantized values (2 per byte = 128 pairs → 256)
     * Bytes 96..107: scales[12] — sub-block scales (packed 6-bit)
     * Bytes 108..109: d (fp16)  — overall scale factor
     */
    const uint8_t* hmask  = block;           /* 32 bytes */
    const uint8_t* qs     = block + 32;      /* 64 bytes */
    const uint8_t* sc_raw = block + 96;      /* 12 bytes */
    uint16_t d_h;
    memcpy(&d_h, block + 108, 2);
    float d = fp16_to_fp32(d_h);

    /* Decode 6-bit scales for 8 sub-blocks */
    int8_t scales[8];
    for (int i = 0; i < 8; i++) {
        uint8_t s;
        if (i < 4) {
            /* Lower 4 sub-blocks: 6 bits from scales[i] (low) + scales[i+8] bits */
            s = sc_raw[i] & 0x3f;
        } else {
            /* Upper 4 sub-blocks: rearranged from remaining bytes */
            s = ((sc_raw[i + 4] & 0xF0) >> 4) | ((sc_raw[i - 4] >> 6) << 4);
        }
        scales[i] = (int8_t)(s - 32);  /* scales are stored offset by 32 */
    }

    for (int sb = 0; sb < 8 && sb * 32 < n; sb++) {
        float block_d = d * scales[sb];
        for (int j = 0; j < 32 && sb * 32 + j < n; j++) {
            int idx = sb * 32 + j;
            /* Extract 2-bit value from qs */
            int byte_idx = idx / 4;
            int bit_shift = (idx % 4) * 2;
            uint8_t q2 = (qs[byte_idx] >> bit_shift) & 0x03;
            /* Extract high bit from hmask */
            int hm_byte = idx / 8;
            int hm_bit  = idx % 8;
            uint8_t hbit = (hmask[hm_byte] >> hm_bit) & 1;
            /* Combine: 3-bit value = q2 | (hbit << 2), range 0..7, centered at 4 */
            int q3 = (int)q2 | ((int)hbit << 2);
            out[idx] = block_d * (q3 - 4);
        }
    }
}

/* ───── Q6_K Dequantization ───── */
/* Q6_K: 256 elements, 210 bytes
 * Layout: uint8_t ql[128] (low 4 bits), uint8_t qh[64] (high 2 bits),
 *         int8_t scales[16], uint16_t d */

static void dequant_q6_k_block(const uint8_t* block, float* out, int n) {
    const uint8_t* ql = block;          /* 128 bytes — low 4 bits */
    const uint8_t* qh = block + 128;    /* 64 bytes — high 2 bits */
    const int8_t*  sc = (const int8_t*)(block + 192);  /* 16 bytes — scales */
    uint16_t d_h;
    memcpy(&d_h, block + 208, 2);
    float d = fp16_to_fp32(d_h);

    for (int sb = 0; sb < 16 && sb * 16 < n; sb++) {
        float block_d = d * sc[sb];
        for (int j = 0; j < 16 && sb * 16 + j < n; j++) {
            int idx = sb * 16 + j;
            /* Low 4 bits from ql */
            uint8_t lo;
            if (idx < 128) {
                lo = (idx % 2 == 0) ? (ql[idx / 2] & 0x0F) : ((ql[idx / 2] >> 4) & 0x0F);
            } else {
                int qi = idx - 128;
                lo = (qi % 2 == 0) ? (ql[64 + qi / 2] & 0x0F) : ((ql[64 + qi / 2] >> 4) & 0x0F);
            }
            /* High 2 bits from qh */
            int qh_idx = idx / 4;
            int qh_shift = (idx % 4) * 2;
            uint8_t hi = (qh[qh_idx] >> qh_shift) & 0x03;
            /* 6-bit value = lo | (hi << 4), range 0..63, centered at 32 */
            int q6 = (int)lo | ((int)hi << 4);
            out[idx] = block_d * (q6 - 32);
        }
    }
}

/* ───── Q5_K Dequantization ───── */
/* Q5_K: 256 elements, 176 bytes
 * Layout: fp16 d, fp16 dmin, uint8_t scales[12], uint8_t qh[32], uint8_t qs[128]
 * Similar to Q4_K but with an extra high bit per element */

static void dequant_q5_k_block(const uint8_t* block, float* out, int n) {
    uint16_t d_h, dmin_h;
    memcpy(&d_h, block, 2);
    memcpy(&dmin_h, block + 2, 2);
    float d = fp16_to_fp32(d_h);
    float dmin = fp16_to_fp32(dmin_h);
    const uint8_t* scales = block + 4;  /* 12 bytes */
    const uint8_t* qh = block + 16;     /* 32 bytes — high bits */
    const uint8_t* qs = block + 48;     /* 128 bytes — low 4 bits */

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
            /* Low 4 bits from qs */
            int byte_idx = idx / 2;
            uint8_t q4 = (idx % 2 == 0) ? (qs[byte_idx] & 0x0F) : ((qs[byte_idx] >> 4) & 0x0F);
            /* High bit from qh */
            int qh_byte = idx / 8;
            int qh_bit  = idx % 8;
            uint8_t hbit = (qh[qh_byte] >> qh_bit) & 1;
            /* 5-bit value = q4 | (hbit << 4) */
            int q5 = (int)q4 | ((int)hbit << 4);
            out[idx] = block_d * q5 - block_m;
        }
    }
}

/* ───── Q8_0 Dequantization ───── */
/* Q8_0: 32 elements, 34 bytes: fp16 d + 32 x int8 */

static void dequant_q8_0_block(const uint8_t* block, float* out, int n) {
    uint16_t d_h;
    memcpy(&d_h, block, 2);
    float d = fp16_to_fp32(d_h);
    const int8_t* qs = (const int8_t*)(block + 2);
    for (int i = 0; i < n && i < 32; i++) {
        out[i] = d * qs[i];
    }
}

/* ───── Generic quantized matmul helper ───── */
/* W[M rows × K cols] is quantized, input[1 × K] is fp32, out[1 × M] is fp32 
 * Note: for weight matrices, the GGUF stores them as [out_features, in_features]
 * So M = out_features (rows of weight = N in matmul notation), K = in_features */

typedef void (*dequant_fn)(const uint8_t* block, float* out, int n);

static void matmul_quantized(const uint8_t* w_data, const float* input, float* output,
                             uint32_t M, uint32_t N, uint32_t K,
                             size_t bytes_per_block, size_t elems_per_block,
                             dequant_fn dequant) {
    float* w_row = (float*)malloc(K * sizeof(float));
    if (!w_row) return;
    
    size_t blocks_per_row = (K + elems_per_block - 1) / elems_per_block;
    size_t row_bytes = blocks_per_row * bytes_per_block;
    
    for (uint32_t i = 0; i < N; i++) {  /* N = out_features = rows of weight matrix */
        const uint8_t* row_data = w_data + i * row_bytes;
        for (size_t blk = 0; blk < blocks_per_row; blk++) {
            int remaining = (int)(K - blk * elems_per_block);
            if (remaining > (int)elems_per_block) remaining = (int)elems_per_block;
            dequant(row_data + blk * bytes_per_block,
                    w_row + blk * elems_per_block, remaining);
        }
        for (uint32_t j = 0; j < M; j++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; k++) sum += input[j * K + k] * w_row[k];
            output[j * N + i] = sum;
        }
    }
    free(w_row);
}

void vk_matmul_q3k(vk_context* ctx,
                    const gpu_buffer* W, const gpu_buffer* B, gpu_buffer* C,
                    uint32_t M, uint32_t N, uint32_t K) {
    (void)ctx;
    matmul_quantized((const uint8_t*)W->mapped, (const float*)B->mapped,
                     (float*)C->mapped, M, N, K, 110, 256, dequant_q3_k_block);
}

void vk_matmul_q6k(vk_context* ctx,
                    const gpu_buffer* W, const gpu_buffer* B, gpu_buffer* C,
                    uint32_t M, uint32_t N, uint32_t K) {
    (void)ctx;
    matmul_quantized((const uint8_t*)W->mapped, (const float*)B->mapped,
                     (float*)C->mapped, M, N, K, 210, 256, dequant_q6_k_block);
}

void vk_matmul_q5k(vk_context* ctx,
                    const gpu_buffer* W, const gpu_buffer* B, gpu_buffer* C,
                    uint32_t M, uint32_t N, uint32_t K) {
    (void)ctx;
    matmul_quantized((const uint8_t*)W->mapped, (const float*)B->mapped,
                     (float*)C->mapped, M, N, K, 176, 256, dequant_q5_k_block);
}

void vk_matmul_q8_0(vk_context* ctx,
                     const gpu_buffer* W, const gpu_buffer* B, gpu_buffer* C,
                     uint32_t M, uint32_t N, uint32_t K) {
    (void)ctx;
    matmul_quantized((const uint8_t*)W->mapped, (const float*)B->mapped,
                     (float*)C->mapped, M, N, K, 34, 32, dequant_q8_0_block);
}

void vk_matmul_f16(vk_context* ctx,
                    const gpu_buffer* W, const gpu_buffer* B, gpu_buffer* C,
                    uint32_t M, uint32_t N, uint32_t K) {
    (void)ctx;
    /* Dequantize entire F16 weight row-by-row */
    const uint16_t* w_data = (const uint16_t*)W->mapped;
    const float* input = (const float*)B->mapped;
    float* output = (float*)C->mapped;
    float* w_row = (float*)malloc(K * sizeof(float));
    if (!w_row) return;
    
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t k = 0; k < K; k++) {
            w_row[k] = fp16_to_fp32(w_data[i * K + k]);
        }
        for (uint32_t j = 0; j < M; j++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; k++) sum += input[j * K + k] * w_row[k];
            output[j * N + i] = sum;
        }
    }
    free(w_row);
}

/* ───── Auto-dispatch matmul based on weight buffer dtype ───── */

void vk_matmul_auto(vk_context* ctx,
                    const gpu_buffer* input, const gpu_buffer* W, gpu_buffer* C,
                    uint32_t M, uint32_t N, uint32_t K) {
    switch (W->dtype) {
        case 0:  /* GGML_TYPE_F32 */
            vk_matmul(ctx, input, W, C, M, N, K);
            break;
        case 1:  /* GGML_TYPE_F16 */
            vk_matmul_f16(ctx, W, input, C, M, N, K);
            break;
        case 11: /* GGML_TYPE_Q3_K */
            vk_matmul_q3k(ctx, W, input, C, M, N, K);
            break;
        case 12: /* GGML_TYPE_Q4_K */
            vk_matmul_q4k(ctx, W, input, C, M, N, K);
            break;
        case 13: /* GGML_TYPE_Q5_K */
            vk_matmul_q5k(ctx, W, input, C, M, N, K);
            break;
        case 14: /* GGML_TYPE_Q6_K */
            vk_matmul_q6k(ctx, W, input, C, M, N, K);
            break;
        case 8:  /* GGML_TYPE_Q8_0 */
            vk_matmul_q8_0(ctx, W, input, C, M, N, K);
            break;
        default:
            fprintf(stderr, "cpu: unsupported dtype %u for matmul — falling back to fp32\n", W->dtype);
            vk_matmul(ctx, input, W, C, M, N, K);
            break;
    }
}

/* ───── Auto-dispatch embedding lookup ───── */

void vk_embedding_auto(vk_context* ctx,
                       const gpu_buffer* table, gpu_buffer* out,
                       uint32_t token_id, uint32_t dim) {
    (void)ctx;
    float* od = (float*)out->mapped;
    
    switch (table->dtype) {
        case 0: { /* F32 */
            const float* td = (const float*)table->mapped;
            memcpy(od, td + (size_t)token_id * dim, dim * sizeof(float));
            break;
        }
        case 1: { /* F16 */
            const uint16_t* td = (const uint16_t*)table->mapped;
            const uint16_t* row = td + (size_t)token_id * dim;
            for (uint32_t i = 0; i < dim; i++) od[i] = fp16_to_fp32(row[i]);
            break;
        }
        case 11: { /* Q3_K */
            size_t blocks_per_row = (dim + 255) / 256;
            size_t row_bytes = blocks_per_row * 110;
            const uint8_t* row = (const uint8_t*)table->mapped + (size_t)token_id * row_bytes;
            for (size_t blk = 0; blk < blocks_per_row; blk++) {
                int rem = (int)(dim - blk * 256);
                if (rem > 256) rem = 256;
                dequant_q3_k_block(row + blk * 110, od + blk * 256, rem);
            }
            break;
        }
        case 12: { /* Q4_K */
            size_t blocks_per_row = (dim + 255) / 256;
            size_t row_bytes = blocks_per_row * 144;
            const uint8_t* row = (const uint8_t*)table->mapped + (size_t)token_id * row_bytes;
            for (size_t blk = 0; blk < blocks_per_row; blk++) {
                int rem = (int)(dim - blk * 256);
                if (rem > 256) rem = 256;
                dequant_q4_k_block(row + blk * 144, od + blk * 256, rem);
            }
            break;
        }
        case 14: { /* Q6_K */
            size_t blocks_per_row = (dim + 255) / 256;
            size_t row_bytes = blocks_per_row * 210;
            const uint8_t* row = (const uint8_t*)table->mapped + (size_t)token_id * row_bytes;
            for (size_t blk = 0; blk < blocks_per_row; blk++) {
                int rem = (int)(dim - blk * 256);
                if (rem > 256) rem = 256;
                dequant_q6_k_block(row + blk * 210, od + blk * 256, rem);
            }
            break;
        }
        case 13: { /* Q5_K */
            size_t blocks_per_row = (dim + 255) / 256;
            size_t row_bytes = blocks_per_row * 176;
            const uint8_t* row = (const uint8_t*)table->mapped + (size_t)token_id * row_bytes;
            for (size_t blk = 0; blk < blocks_per_row; blk++) {
                int rem = (int)(dim - blk * 256);
                if (rem > 256) rem = 256;
                dequant_q5_k_block(row + blk * 176, od + blk * 256, rem);
            }
            break;
        }
        case 8: { /* Q8_0 */
            size_t blocks_per_row = (dim + 31) / 32;
            size_t row_bytes = blocks_per_row * 34;
            const uint8_t* row = (const uint8_t*)table->mapped + (size_t)token_id * row_bytes;
            for (size_t blk = 0; blk < blocks_per_row; blk++) {
                int rem = (int)(dim - blk * 32);
                if (rem > 32) rem = 32;
                dequant_q8_0_block(row + blk * 34, od + blk * 32, rem);
            }
            break;
        }
        default:
            fprintf(stderr, "cpu: unsupported embedding dtype %u — zero fill\n", table->dtype);
            memset(od, 0, dim * sizeof(float));
            break;
    }
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
    (void)attn_scores;
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

/* ════════════════════════════════════════════════════════════════════
 * DELTANET OPERATIONS — for hybrid architecture
 * ════════════════════════════════════════════════════════════════════ */

/* 1D causal convolution with ring buffer state (depthwise, single step). */
void vk_conv1d(vk_context* ctx,
               gpu_buffer* conv_state, const gpu_buffer* x,
               const gpu_buffer* weight, gpu_buffer* out,
               uint32_t channels, uint32_t kernel_size) {
    (void)ctx;
    float* state = (float*)conv_state->mapped;
    const float* xd = (const float*)x->mapped;
    const float* wd = (const float*)weight->mapped;
    float* od = (float*)out->mapped;
    
    uint32_t hist = kernel_size - 1;
    
    for (uint32_t c = 0; c < channels; c++) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < hist; j++) {
            sum += wd[j * channels + c] * state[j * channels + c];
        }
        sum += wd[hist * channels + c] * xd[c];
        od[c] = sum;
    }
    
    if (hist > 1) {
        memmove(state, state + channels, (size_t)(hist - 1) * channels * sizeof(float));
    }
    if (hist > 0) {
        memcpy(state + (hist - 1) * channels, xd, channels * sizeof(float));
    }
}

/* L2 normalization per group */
void vk_l2_norm(vk_context* ctx,
                const gpu_buffer* x, gpu_buffer* out,
                uint32_t total_dim, uint32_t group_size) {
    (void)ctx;
    const float* xd = (const float*)x->mapped;
    float* od = (float*)out->mapped;
    uint32_t num_groups = total_dim / group_size;
    
    for (uint32_t g = 0; g < num_groups; g++) {
        uint32_t base = g * group_size;
        float norm_sq = 0.0f;
        for (uint32_t i = 0; i < group_size; i++) {
            norm_sq += xd[base + i] * xd[base + i];
        }
        float inv_norm = 1.0f / (sqrtf(norm_sq) + 1e-12f);
        for (uint32_t i = 0; i < group_size; i++) {
            od[base + i] = xd[base + i] * inv_norm;
        }
    }
}

/* Sigmoid: x = 1 / (1 + exp(-x)), in-place */
void vk_sigmoid(vk_context* ctx, gpu_buffer* x, uint32_t n_elements) {
    (void)ctx;
    float* xd = (float*)x->mapped;
    for (uint32_t i = 0; i < n_elements; i++) {
        xd[i] = 1.0f / (1.0f + expf(-xd[i]));
    }
}

/* Softplus: x = log(1 + exp(x)), in-place */
void vk_softplus(vk_context* ctx, gpu_buffer* x, uint32_t n_elements) {
    (void)ctx;
    float* xd = (float*)x->mapped;
    for (uint32_t i = 0; i < n_elements; i++) {
        if (xd[i] > 20.0f) {
            /* xd[i] stays as is */
        } else if (xd[i] < -20.0f) {
            xd[i] = 0.0f;
        } else {
            xd[i] = logf(1.0f + expf(xd[i]));
        }
    }
}

/* DeltaNet recurrent step (single token, all heads). */
void vk_deltanet_step(vk_context* ctx,
                      const gpu_buffer* q, const gpu_buffer* k,
                      const gpu_buffer* v, const gpu_buffer* beta,
                      gpu_buffer* state, gpu_buffer* out,
                      uint32_t num_v_heads, uint32_t head_k_dim,
                      uint32_t head_v_dim) {
    (void)ctx;
    
    const float* q_data = (const float*)q->mapped;
    
    uint32_t qkv_total = (uint32_t)(q->size / sizeof(float));
    uint32_t v_in_qkv = num_v_heads * head_v_dim;
    uint32_t qk_floats = qkv_total - v_in_qkv;
    uint32_t num_k_heads = qk_floats / (head_k_dim * 2);
    uint32_t k_offset = num_k_heads * head_k_dim;
    
    uint32_t q_offset = 0;
    const float* k_data = q_data + k_offset;
    const float* v_data = (const float*)v->mapped;
    const float* ssm_a_data = (const float*)beta->mapped;
    float* S = (float*)state->mapped;
    float* od = (float*)out->mapped;
    
    uint32_t kv_group = (num_k_heads > 0) ? (num_v_heads / num_k_heads) : 1;
    if (kv_group == 0) kv_group = 1;
    
    for (uint32_t vh = 0; vh < num_v_heads; vh++) {
        uint32_t kh = vh / kv_group;
        
        const float* q_h = q_data + q_offset + kh * head_k_dim;
        const float* k_h = k_data + kh * head_k_dim;
        const float* v_h = v_data + vh * head_v_dim;
        
        float a_val = ssm_a_data[vh];
        float sp = (a_val > 20.0f) ? a_val : ((a_val < -20.0f) ? 0.0f : logf(1.0f + expf(a_val)));
        float decay = 1.0f / (1.0f + expf(sp));
        
        float* S_h = S + (size_t)vh * head_v_dim * head_k_dim;
        
        for (uint32_t vi = 0; vi < head_v_dim; vi++) {
            for (uint32_t ki = 0; ki < head_k_dim; ki++) {
                S_h[vi * head_k_dim + ki] = decay * S_h[vi * head_k_dim + ki]
                                           + v_h[vi] * k_h[ki];
            }
        }
        
        float* o_h = od + vh * head_v_dim;
        for (uint32_t vi = 0; vi < head_v_dim; vi++) {
            float sum = 0.0f;
            for (uint32_t ki = 0; ki < head_k_dim; ki++) {
                sum += S_h[vi * head_k_dim + ki] * q_h[ki];
            }
            o_h[vi] = sum;
        }
    }
}

/* RMS normalization per-head */
void vk_rmsnorm_head(vk_context* ctx,
                     const gpu_buffer* x, const gpu_buffer* weight,
                     gpu_buffer* out,
                     uint32_t num_heads, uint32_t head_dim, float eps) {
    (void)ctx;
    const float* xd = (const float*)x->mapped;
    const float* wd = (const float*)weight->mapped;
    float* od = (float*)out->mapped;
    
    for (uint32_t h = 0; h < num_heads; h++) {
        uint32_t base = h * head_dim;
        float ss = 0.0f;
        for (uint32_t i = 0; i < head_dim; i++) {
            ss += xd[base + i] * xd[base + i];
        }
        float rms = 1.0f / sqrtf(ss / head_dim + eps);
        for (uint32_t i = 0; i < head_dim; i++) {
            od[base + i] = xd[base + i] * rms * wd[i];
        }
    }
}

/* ════════════════════════════════════════════════════════════════════
 * MAMBA OPERATIONS — S6 (Mamba-1) and SSD (Mamba-2)
 * ════════════════════════════════════════════════════════════════════ */

/* Mamba-1 selective scan step (single token, autoregressive).
 * Implements: s[j,i] = s[j,i] * exp(dt[i] * A[j,i]) + B[j] * (x[i] * dt[i])
 *             y[i]   = sum_j(s[j,i] * C[j]) + D[i] * x[i]
 *
 * x:     [d_inner] — input (after conv + silu)
 * dt:    [d_inner] — time step (after softplus)
 * A:     [d_state, d_inner] — diagonal decay (negative, stored as log)
 * B:     [d_state] — input projection
 * C:     [d_state] — output projection
 * D:     [d_inner] — skip connection
 * state: [d_state * d_inner] — recurrent state (updated in-place)
 * y:     [d_inner] — output
 */
void vk_mamba1_ssm_step(vk_context* ctx,
                        const gpu_buffer* x, const gpu_buffer* dt_buf,
                        const gpu_buffer* A, const gpu_buffer* B_buf,
                        const gpu_buffer* C_buf, const gpu_buffer* D_buf,
                        gpu_buffer* state, gpu_buffer* y,
                        uint32_t d_inner, uint32_t d_state) {
    (void)ctx;
    const float* xd = (const float*)x->mapped;
    const float* dt = (const float*)dt_buf->mapped;
    const float* Ad = (const float*)A->mapped;
    const float* Bd = (const float*)B_buf->mapped;
    const float* Cd = (const float*)C_buf->mapped;
    const float* Dd = (const float*)D_buf->mapped;
    float* sd = (float*)state->mapped;
    float* yd = (float*)y->mapped;
    
    for (uint32_t i = 0; i < d_inner; i++) {
        float yi = 0.0f;
        float dt_i = dt[i];
        float x_dt = xd[i] * dt_i;
        
        for (uint32_t j = 0; j < d_state; j++) {
            uint32_t si = j * d_inner + i;  /* state[d_state, d_inner] */
            /* dA = exp(dt * A), where A is stored as negative values */
            float dA = expf(dt_i * Ad[j * d_inner + i]);
            /* state update */
            sd[si] = sd[si] * dA + Bd[j] * x_dt;
            /* output accumulation */
            yi += sd[si] * Cd[j];
        }
        
        /* Skip connection: y = y + D * x */
        yd[i] = yi + Dd[i] * xd[i];
    }
}

/* Mamba-2 selective scan step (single token, autoregressive).
 * Mamba-2 uses scalar decay per head (not per element like Mamba-1).
 *
 * x:     [n_head * dim_per_head] — input (d_inner = n_head * dim)
 * dt:    [n_head] — time step per head (after softplus)
 * A:     [n_head] — scalar decay per head (stored as -exp(A_log))
 * B:     [n_group * d_state] — input projection (groups shared across heads)
 * C:     [n_group * d_state] — output projection
 * D:     [n_head] — skip connection per head
 * state: [n_head * dim_per_head * d_state] — recurrent state (updated in-place)
 * y:     [n_head * dim_per_head] — output (d_inner)
 */
void vk_mamba2_ssm_step(vk_context* ctx,
                        const gpu_buffer* x, const gpu_buffer* dt_buf,
                        const gpu_buffer* A, const gpu_buffer* B_buf,
                        const gpu_buffer* C_buf, const gpu_buffer* D_buf,
                        gpu_buffer* state, gpu_buffer* y,
                        uint32_t n_head, uint32_t dim_per_head,
                        uint32_t d_state, uint32_t n_group) {
    (void)ctx;
    const float* xd = (const float*)x->mapped;
    const float* dt = (const float*)dt_buf->mapped;
    const float* Ad = (const float*)A->mapped;
    const float* Bd = (const float*)B_buf->mapped;
    const float* Cd = (const float*)C_buf->mapped;
    const float* Dd = (const float*)D_buf->mapped;
    float* sd = (float*)state->mapped;
    float* yd = (float*)y->mapped;
    
    uint32_t heads_per_group = (n_group > 0) ? (n_head / n_group) : n_head;
    
    for (uint32_t h = 0; h < n_head; h++) {
        float dt_sp = dt[h];  /* already softplus'd */
        float dA = expf(dt_sp * Ad[h]);
        uint32_t g = h / heads_per_group;  /* which group this head belongs to */
        
        for (uint32_t d = 0; d < dim_per_head; d++) {
            uint32_t xidx = h * dim_per_head + d;
            float x_dt = xd[xidx] * dt_sp;
            float yi = 0.0f;
            
            for (uint32_t s = 0; s < d_state; s++) {
                uint32_t si = (h * dim_per_head + d) * d_state + s;
                /* state update: s = dA * s + B * x_dt */
                sd[si] = sd[si] * dA + Bd[g * d_state + s] * x_dt;
                /* output: y += s * C */
                yi += sd[si] * Cd[g * d_state + s];
            }
            
            /* Skip connection: y += D * x */
            yd[xidx] = yi + Dd[h] * xd[xidx];
        }
    }
}

/* Add bias to buffer, in-place: x[i] += bias[i] */
void vk_add_bias(vk_context* ctx, gpu_buffer* x, const gpu_buffer* bias, uint32_t n) {
    (void)ctx;
    float* xd = (float*)x->mapped;
    const float* bd = (const float*)bias->mapped;
    for (uint32_t i = 0; i < n; i++)
        xd[i] += bd[i];
}

/* ─── Diagnostics ─── */

void vk_print_info(const vk_context* ctx) {
    printf("vk: Backend: CPU-only (no GPU)\n");
    printf("vk: Memory used: %.2f MB\n", (double)ctx->used_memory / (1024 * 1024));
}

size_t vk_memory_used(const vk_context* ctx) { return ctx->used_memory; }
size_t vk_memory_total(const vk_context* ctx) { return ctx->device_memory_size; }

#endif /* CPU_ONLY */
