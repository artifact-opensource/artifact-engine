/*
 * Artifact Engine — GGUF Format Definitions
 * Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
 *
 * GGUF is the standard format for quantized LLM weights.
 * We implement just enough to load Q4_K_M and Q8_0 models:
 * Qwen 3.5, Mamba, Mamba-2, Jamba, Falcon-H1.
 */

#ifndef GGUF_H
#define GGUF_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* GGUF magic number: "GGUF" in little-endian */
#define GGUF_MAGIC 0x46554747  /* "GGUF" */
#define GGUF_VERSION 3

/* ───── Value Types ───── */
typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_type;

/* ───── Tensor Types (quantization formats) ───── */
typedef enum {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
} ggml_type;

/* Block sizes for quantized types */
static inline size_t ggml_type_block_size(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return 1;
        case GGML_TYPE_F16:  return 1;
        case GGML_TYPE_Q4_0: return 32;
        case GGML_TYPE_Q4_1: return 32;
        case GGML_TYPE_Q5_0: return 32;
        case GGML_TYPE_Q5_1: return 32;
        case GGML_TYPE_Q8_0: return 32;
        case GGML_TYPE_Q8_1: return 32;
        case GGML_TYPE_Q2_K: return 256;
        case GGML_TYPE_Q3_K: return 256;
        case GGML_TYPE_Q4_K: return 256;
        case GGML_TYPE_Q5_K: return 256;
        case GGML_TYPE_Q6_K: return 256;
        case GGML_TYPE_Q8_K: return 256;
        case GGML_TYPE_BF16: return 1;
        default: return 0;
    }
}

/* Bytes per block for quantized types */
static inline size_t ggml_type_bytes_per_block(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return 4;
        case GGML_TYPE_F16:  return 2;
        case GGML_TYPE_Q4_0: return 18;
        case GGML_TYPE_Q4_1: return 20;
        case GGML_TYPE_Q5_0: return 22;
        case GGML_TYPE_Q5_1: return 24;
        case GGML_TYPE_Q8_0: return 34;
        case GGML_TYPE_Q8_1: return 36;
        case GGML_TYPE_Q2_K: return 84;
        case GGML_TYPE_Q3_K: return 110;
        case GGML_TYPE_Q4_K: return 144;
        case GGML_TYPE_Q5_K: return 176;
        case GGML_TYPE_Q6_K: return 210;
        case GGML_TYPE_Q8_K: return 292;
        case GGML_TYPE_BF16: return 2;
        default: return 0;
    }
}

/* ───── GGUF String ───── */
typedef struct {
    uint64_t len;
    char*    data;
} gguf_string;

/* ───── GGUF Key-Value pair ───── */
typedef struct {
    gguf_string key;
    gguf_type   type;
    union {
        uint8_t   u8;
        int8_t    i8;
        uint16_t  u16;
        int16_t   i16;
        uint32_t  u32;
        int32_t   i32;
        float     f32;
        uint64_t  u64;
        int64_t   i64;
        double    f64;
        bool      b;
        gguf_string str;
        struct {
            gguf_type elem_type;
            uint64_t  count;
            void*     data;
        } arr;
    } value;
} gguf_kv;

/* ───── GGUF Tensor Info ───── */
#define GGML_MAX_DIMS 4

typedef struct {
    gguf_string name;
    uint32_t    n_dims;
    uint64_t    dims[GGML_MAX_DIMS];
    ggml_type   type;
    uint64_t    offset;
    
    uint64_t    n_elements;
    uint64_t    n_bytes;
} gguf_tensor_info;

/* ───── GGUF File ───── */
typedef struct {
    uint32_t    version;
    uint64_t    n_tensors;
    uint64_t    n_kv;
    
    gguf_kv*         kv;
    gguf_tensor_info* tensors;
    
    void*    data;
    size_t   data_size;
    
    void*    mmap_addr;
    size_t   mmap_size;
    int      fd;
} gguf_file;

/* ───── Model Architecture (extracted from GGUF metadata) ───── */
typedef struct {
    char     arch[64];          /* e.g. "qwen2", "qwen35", "mamba", "mamba2", "jamba", "falcon_h1" */
    uint32_t vocab_size;
    uint32_t hidden_size;       /* embedding dim */
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;        /* GQA heads */
    uint32_t intermediate_size; /* FFN dim */
    uint32_t max_position;      /* max context length */
    float    rope_freq_base;
    float    rms_norm_eps;
    uint32_t head_dim;          /* hidden_size / n_heads */
    
    /* Hybrid architecture (DeltaNet + Attention, or Mamba + Attention) */
    uint32_t full_attn_interval;  /* every Nth layer is full attention (0 = all attention) */
    
    /* SSM parameters (DeltaNet / Mamba / Mamba-2) */
    uint32_t ssm_d_inner;        /* SSM inner dimension (d_inner) */
    uint32_t ssm_d_state;        /* SSM state dimension (d_state, e.g. 16 for Mamba) */
    uint32_t ssm_n_group;        /* DeltaNet: K heads; Mamba2: number of groups */
    uint32_t ssm_dt_rank;        /* DeltaNet: V heads; Mamba1: dt_rank (e.g. 16) */
    uint32_t ssm_conv_kernel;    /* conv kernel size (e.g. 4) */
    uint32_t ssm_n_head;         /* Mamba2: number of SSM heads */
    uint32_t ssm_d_in_proj;      /* Mamba2: input projection size (x+z+B+C+dt fused) */
    bool     ssm_dt_b_c_rms;     /* Mamba1 variant: RMS norm on dt, B, C */
    
    /* Tokenizer */
    uint32_t bos_token_id;
    uint32_t eos_token_id;
    uint32_t pad_token_id;
    char**   vocab_tokens;
    float*   vocab_scores;
} model_arch;

/* ───── API ───── */

gguf_file* gguf_load(const char* path);
void gguf_free(gguf_file* gf);
bool gguf_extract_arch(const gguf_file* gf, model_arch* arch);
const gguf_tensor_info* gguf_find_tensor(const gguf_file* gf, const char* name);
const void* gguf_tensor_data(const gguf_file* gf, const gguf_tensor_info* ti);
const gguf_kv* gguf_find_kv(const gguf_file* gf, const char* key);
const char* gguf_get_string(const gguf_file* gf, const char* key);
uint32_t gguf_get_u32(const gguf_file* gf, const char* key, uint32_t default_val);
float gguf_get_f32(const gguf_file* gf, const char* key, float default_val);
void gguf_print_info(const gguf_file* gf);

#endif /* GGUF_H */
