/*
 * Artifact Engine — GGUF File Loader
 * 
 * Memory-maps a GGUF file and parses its header, metadata, and tensor info.
 * Tensor data stays in the mmap — zero-copy until we upload to GPU.
 *
 * Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
 */

#define _GNU_SOURCE  /* for madvise */

#include "../include/gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>

#ifdef _WIN32
  #include <windows.h>
  #include <io.h>
#else
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

/* ───── Internal: Binary Reader ───── */

typedef struct {
    const uint8_t* data;
    size_t         size;
    size_t         pos;
} reader;

static inline bool reader_has(const reader* r, size_t n) {
    return r->pos + n <= r->size;
}

static inline uint8_t read_u8(reader* r) {
    uint8_t v = r->data[r->pos];
    r->pos += 1;
    return v;
}

static inline uint16_t read_u16(reader* r) {
    uint16_t v;
    memcpy(&v, r->data + r->pos, 2);
    r->pos += 2;
    return v;
}

static inline uint32_t read_u32(reader* r) {
    uint32_t v;
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return v;
}

static inline uint64_t read_u64(reader* r) {
    uint64_t v;
    memcpy(&v, r->data + r->pos, 8);
    r->pos += 8;
    return v;
}

static inline int32_t read_i32(reader* r) {
    int32_t v;
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return v;
}

static inline int64_t read_i64(reader* r) {
    int64_t v;
    memcpy(&v, r->data + r->pos, 8);
    r->pos += 8;
    return v;
}

static inline float read_f32(reader* r) {
    float v;
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return v;
}

static inline double read_f64(reader* r) {
    double v;
    memcpy(&v, r->data + r->pos, 8);
    r->pos += 8;
    return v;
}

static gguf_string read_string(reader* r) {
    gguf_string s = {0};
    if (!reader_has(r, 8)) return s;
    s.len = read_u64(r);
    if (!reader_has(r, s.len)) { s.len = 0; return s; }
    s.data = malloc(s.len + 1);
    if (!s.data) { s.len = 0; return s; }
    memcpy(s.data, r->data + r->pos, s.len);
    s.data[s.len] = '\0';
    r->pos += s.len;
    return s;
}

/* Read a GGUF value of given type */
static bool read_value(reader* r, gguf_type type, gguf_kv* kv) {
    kv->type = type;
    switch (type) {
        case GGUF_TYPE_UINT8:   kv->value.u8  = read_u8(r);  break;
        case GGUF_TYPE_INT8:    kv->value.i8  = (int8_t)read_u8(r); break;
        case GGUF_TYPE_UINT16:  kv->value.u16 = read_u16(r); break;
        case GGUF_TYPE_INT16:   kv->value.i16 = (int16_t)read_u16(r); break;
        case GGUF_TYPE_UINT32:  kv->value.u32 = read_u32(r); break;
        case GGUF_TYPE_INT32:   kv->value.i32 = read_i32(r); break;
        case GGUF_TYPE_FLOAT32: kv->value.f32 = read_f32(r); break;
        case GGUF_TYPE_BOOL:    kv->value.b   = read_u8(r) != 0; break;
        case GGUF_TYPE_STRING:  kv->value.str = read_string(r); break;
        case GGUF_TYPE_UINT64:  kv->value.u64 = read_u64(r); break;
        case GGUF_TYPE_INT64:   kv->value.i64 = read_i64(r); break;
        case GGUF_TYPE_FLOAT64: kv->value.f64 = read_f64(r); break;
        case GGUF_TYPE_ARRAY: {
            kv->value.arr.elem_type = (gguf_type)read_u32(r);
            kv->value.arr.count = read_u64(r);
            kv->value.arr.data = (void*)(uintptr_t)r->pos;
            for (uint64_t i = 0; i < kv->value.arr.count; i++) {
                switch (kv->value.arr.elem_type) {
                    case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL:
                        r->pos += 1; break;
                    case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16:
                        r->pos += 2; break;
                    case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32:
                        r->pos += 4; break;
                    case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64:
                        r->pos += 8; break;
                    case GGUF_TYPE_STRING: {
                        gguf_string s = read_string(r);
                        free(s.data);
                        break;
                    }
                    default:
                        fprintf(stderr, "gguf: unsupported array element type %d\n", kv->value.arr.elem_type);
                        return false;
                }
            }
            break;
        }
        default:
            fprintf(stderr, "gguf: unknown value type %d\n", type);
            return false;
    }
    return true;
}

/* ───── Platform-specific mmap ───── */

#ifdef _WIN32

static void* mmap_file(const char* path, size_t* out_size, void** handle_out) {
    HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE) return NULL;
    
    LARGE_INTEGER size;
    if (!GetFileSizeEx(file, &size)) { CloseHandle(file); return NULL; }
    
    HANDLE mapping = CreateFileMappingA(file, NULL, PAGE_READONLY, 
                                         size.HighPart, size.LowPart, NULL);
    if (!mapping) { CloseHandle(file); return NULL; }
    
    void* addr = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(mapping);
    CloseHandle(file);
    
    if (!addr) return NULL;
    *out_size = (size_t)size.QuadPart;
    *handle_out = addr;
    return addr;
}

static void munmap_file(void* addr, size_t size) {
    (void)size;
    UnmapViewOfFile(addr);
}

#else

static void* mmap_file(const char* path, size_t* out_size, void** handle_out) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return NULL; }
    
    void* addr = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    
    if (addr == MAP_FAILED) return NULL;
    
    madvise(addr, st.st_size, MADV_SEQUENTIAL);
    
    *out_size = st.st_size;
    *handle_out = addr;
    return addr;
}

static void munmap_file(void* addr, size_t size) {
    munmap(addr, size);
}

#endif

/* ───── Public API ───── */

gguf_file* gguf_load(const char* path) {
    gguf_file* gf = calloc(1, sizeof(gguf_file));
    if (!gf) return NULL;
    
    gf->mmap_addr = mmap_file(path, &gf->mmap_size, &gf->mmap_addr);
    if (!gf->mmap_addr) {
        fprintf(stderr, "gguf: failed to mmap '%s': %s\n", path, strerror(errno));
        free(gf);
        return NULL;
    }
    
    reader r = { .data = (const uint8_t*)gf->mmap_addr, .size = gf->mmap_size, .pos = 0 };
    
    if (!reader_has(&r, 24)) goto fail;
    
    uint32_t magic = read_u32(&r);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "gguf: bad magic 0x%08x (expected 0x%08x)\n", magic, GGUF_MAGIC);
        goto fail;
    }
    
    gf->version   = read_u32(&r);
    gf->n_tensors = read_u64(&r);
    gf->n_kv      = read_u64(&r);
    
    if (gf->version < 2 || gf->version > 3) {
        fprintf(stderr, "gguf: unsupported version %u (expected 2 or 3)\n", gf->version);
        goto fail;
    }
    
    printf("gguf: version=%u, tensors=%lu, kv_pairs=%lu\n",
           gf->version, (unsigned long)gf->n_tensors, (unsigned long)gf->n_kv);
    
    gf->kv = calloc(gf->n_kv, sizeof(gguf_kv));
    if (!gf->kv) goto fail;
    
    for (uint64_t i = 0; i < gf->n_kv; i++) {
        gf->kv[i].key = read_string(&r);
        if (!gf->kv[i].key.data) goto fail;
        
        gguf_type vtype = (gguf_type)read_u32(&r);
        if (!read_value(&r, vtype, &gf->kv[i])) goto fail;
    }
    
    gf->tensors = calloc(gf->n_tensors, sizeof(gguf_tensor_info));
    if (!gf->tensors) goto fail;
    
    for (uint64_t i = 0; i < gf->n_tensors; i++) {
        gguf_tensor_info* ti = &gf->tensors[i];
        ti->name   = read_string(&r);
        ti->n_dims = read_u32(&r);
        
        if (ti->n_dims > GGML_MAX_DIMS) {
            fprintf(stderr, "gguf: tensor '%s' has %u dims (max %d)\n",
                    ti->name.data, ti->n_dims, GGML_MAX_DIMS);
            goto fail;
        }
        
        ti->n_elements = 1;
        for (uint32_t d = 0; d < ti->n_dims; d++) {
            ti->dims[d] = read_u64(&r);
            ti->n_elements *= ti->dims[d];
        }
        for (uint32_t d = ti->n_dims; d < GGML_MAX_DIMS; d++) {
            ti->dims[d] = 1;
        }
        
        ti->type   = (ggml_type)read_u32(&r);
        ti->offset = read_u64(&r);
        
        size_t block_size = ggml_type_block_size(ti->type);
        size_t bytes_per_block = ggml_type_bytes_per_block(ti->type);
        if (block_size == 0 || bytes_per_block == 0) {
            fprintf(stderr, "gguf: tensor '%s' has unsupported type %d\n",
                    ti->name.data, ti->type);
            goto fail;
        }
        ti->n_bytes = (ti->n_elements / block_size) * bytes_per_block;
    }
    
    size_t alignment = 32;
    const gguf_kv* align_kv = gguf_find_kv(gf, "general.alignment");
    if (align_kv && align_kv->type == GGUF_TYPE_UINT32) {
        alignment = align_kv->value.u32;
    }
    
    size_t data_offset = r.pos;
    data_offset = (data_offset + alignment - 1) & ~(alignment - 1);
    
    gf->data = (uint8_t*)gf->mmap_addr + data_offset;
    gf->data_size = (data_offset <= gf->mmap_size) ? (gf->mmap_size - data_offset) : 0;
    
    printf("gguf: data section at offset %zu, size %.2f GB\n",
           data_offset, (double)gf->data_size / (1024.0*1024.0*1024.0));
    
    return gf;
    
fail:
    gguf_free(gf);
    return NULL;
}

void gguf_free(gguf_file* gf) {
    if (!gf) return;
    
    if (gf->kv) {
        for (uint64_t i = 0; i < gf->n_kv; i++) {
            free(gf->kv[i].key.data);
            if (gf->kv[i].type == GGUF_TYPE_STRING) {
                free(gf->kv[i].value.str.data);
            } else if (gf->kv[i].type == GGUF_TYPE_ARRAY &&
                       gf->kv[i].value.arr.elem_type == GGUF_TYPE_STRING &&
                       gf->kv[i].value.arr.data) {
                gguf_string* arr = (gguf_string*)gf->kv[i].value.arr.data;
                for (uint64_t j = 0; j < gf->kv[i].value.arr.count; j++) {
                    free(arr[j].data);
                }
                free(arr);
            }
        }
        free(gf->kv);
    }
    
    if (gf->tensors) {
        for (uint64_t i = 0; i < gf->n_tensors; i++) {
            free(gf->tensors[i].name.data);
        }
        free(gf->tensors);
    }
    
    if (gf->mmap_addr) {
        munmap_file(gf->mmap_addr, gf->mmap_size);
    }
    
    free(gf);
}

const gguf_kv* gguf_find_kv(const gguf_file* gf, const char* key) {
    for (uint64_t i = 0; i < gf->n_kv; i++) {
        if (gf->kv[i].key.data && strcmp(gf->kv[i].key.data, key) == 0) {
            return &gf->kv[i];
        }
    }
    return NULL;
}

const char* gguf_get_string(const gguf_file* gf, const char* key) {
    const gguf_kv* kv = gguf_find_kv(gf, key);
    if (kv && kv->type == GGUF_TYPE_STRING) return kv->value.str.data;
    return NULL;
}

uint32_t gguf_get_u32(const gguf_file* gf, const char* key, uint32_t def) {
    const gguf_kv* kv = gguf_find_kv(gf, key);
    if (!kv) return def;
    switch (kv->type) {
        case GGUF_TYPE_UINT32:  return kv->value.u32;
        case GGUF_TYPE_INT32:   return (uint32_t)kv->value.i32;
        case GGUF_TYPE_UINT64:  return (uint32_t)kv->value.u64;
        case GGUF_TYPE_FLOAT32: return (uint32_t)kv->value.f32;
        default: return def;
    }
}

float gguf_get_f32(const gguf_file* gf, const char* key, float def) {
    const gguf_kv* kv = gguf_find_kv(gf, key);
    if (!kv) return def;
    switch (kv->type) {
        case GGUF_TYPE_FLOAT32: return kv->value.f32;
        case GGUF_TYPE_FLOAT64: return (float)kv->value.f64;
        case GGUF_TYPE_UINT32:  return (float)kv->value.u32;
        case GGUF_TYPE_INT32:   return (float)kv->value.i32;
        default: return def;
    }
}

static bool gguf_get_bool(const gguf_file* gf, const char* key, bool def) {
    const gguf_kv* kv = gguf_find_kv(gf, key);
    if (!kv) return def;
    switch (kv->type) {
        case GGUF_TYPE_BOOL:    return kv->value.b;
        case GGUF_TYPE_UINT8:   return kv->value.u8 != 0;
        case GGUF_TYPE_UINT32:  return kv->value.u32 != 0;
        default: return def;
    }
}

const gguf_tensor_info* gguf_find_tensor(const gguf_file* gf, const char* name) {
    for (uint64_t i = 0; i < gf->n_tensors; i++) {
        if (gf->tensors[i].name.data && strcmp(gf->tensors[i].name.data, name) == 0) {
            return &gf->tensors[i];
        }
    }
    return NULL;
}

const void* gguf_tensor_data(const gguf_file* gf, const gguf_tensor_info* ti) {
    return (const uint8_t*)gf->data + ti->offset;
}

bool gguf_extract_arch(const gguf_file* gf, model_arch* arch) {
    memset(arch, 0, sizeof(*arch));
    
    const char* arch_str = gguf_get_string(gf, "general.architecture");
    if (arch_str) {
        strncpy(arch->arch, arch_str, sizeof(arch->arch) - 1);
    } else {
        strcpy(arch->arch, "unknown");
    }
    
    char prefix[128];
    snprintf(prefix, sizeof(prefix), "%s.", arch->arch);
    
    char key[256];
    
    #define GET_ARCH_U32(field, name, def) \
        snprintf(key, sizeof(key), "%s%s", prefix, name); \
        arch->field = gguf_get_u32(gf, key, def)
    
    #define GET_ARCH_F32(field, name, def) \
        snprintf(key, sizeof(key), "%s%s", prefix, name); \
        arch->field = gguf_get_f32(gf, key, def)
    
    #define GET_ARCH_BOOL(field, name, def) \
        snprintf(key, sizeof(key), "%s%s", prefix, name); \
        arch->field = gguf_get_bool(gf, key, def)
    
    GET_ARCH_U32(vocab_size,        "vocab_size",            0);
    GET_ARCH_U32(hidden_size,       "embedding_length",      0);
    GET_ARCH_U32(n_layers,          "block_count",           0);
    GET_ARCH_U32(n_heads,           "attention.head_count",  0);
    GET_ARCH_U32(n_kv_heads,        "attention.head_count_kv", 0);
    GET_ARCH_U32(intermediate_size, "feed_forward_length",   0);
    GET_ARCH_U32(max_position,      "context_length",        4096);
    GET_ARCH_F32(rope_freq_base,    "rope.freq_base",        10000.0f);
    GET_ARCH_F32(rms_norm_eps,      "attention.layer_norm_rms_epsilon", 1e-6f);
    
    /* Hybrid architecture: DeltaNet + Attention or Mamba + Attention */
    GET_ARCH_U32(full_attn_interval, "full_attention_interval", 0);
    
    /* SSM parameters (DeltaNet / Mamba / Mamba-2) */
    GET_ARCH_U32(ssm_d_inner,       "ssm.inner_size",         0);
    GET_ARCH_U32(ssm_d_state,       "ssm.state_size",         0);
    GET_ARCH_U32(ssm_n_group,       "ssm.group_count",        0);
    GET_ARCH_U32(ssm_dt_rank,       "ssm.time_step_rank",     0);
    GET_ARCH_U32(ssm_conv_kernel,   "ssm.conv_kernel",        4);
    
    /* Mamba-2 specific */
    GET_ARCH_U32(ssm_n_head,        "ssm.head_count",         0);
    GET_ARCH_U32(ssm_d_in_proj,     "ssm.in_proj_size",       0);
    
    /* Mamba-1 variant flag: some models RMS-norm dt, B, C */
    GET_ARCH_BOOL(ssm_dt_b_c_rms,   "ssm.dt_b_c_rms",        false);
    
    #undef GET_ARCH_U32
    #undef GET_ARCH_F32
    #undef GET_ARCH_BOOL
    
    /* Derived */
    if (arch->n_heads > 0) {
        arch->head_dim = arch->hidden_size / arch->n_heads;
    }
    if (arch->n_kv_heads == 0) {
        arch->n_kv_heads = arch->n_heads; /* MHA if not specified */
    }
    
    /* For pure Mamba models, attention heads may be 0.
     * Derive Mamba-2 head count from ssm_d_inner if ssm_n_head is set. */
    if (arch->ssm_n_head > 0 && arch->ssm_d_inner > 0) {
        /* Mamba-2: dim_per_head = d_inner / n_head */
    }
    
    /* For Mamba-1: if d_inner is 0 but hidden_size is set, d_inner defaults to 2*hidden */
    if (strcmp(arch->arch, "mamba") == 0 && arch->ssm_d_inner == 0 && arch->hidden_size > 0) {
        arch->ssm_d_inner = arch->hidden_size * 2;  /* Mamba convention */
    }
    
    /* For Mamba-1: dt_rank defaults to hidden_size / 16 if not specified */
    if (strcmp(arch->arch, "mamba") == 0 && arch->ssm_dt_rank == 0 && arch->hidden_size > 0) {
        arch->ssm_dt_rank = arch->hidden_size / 16;
    }
    
    /* For Mamba-2: derive n_group from ssm_d_inner if not set */
    if (strcmp(arch->arch, "mamba2") == 0 && arch->ssm_n_group == 0) {
        arch->ssm_n_group = 1;  /* default 1 group */
    }
    
    /* Tokenizer */
    arch->bos_token_id = gguf_get_u32(gf, "tokenizer.ggml.bos_token_id", 1);
    arch->eos_token_id = gguf_get_u32(gf, "tokenizer.ggml.eos_token_id", 2);
    arch->pad_token_id = gguf_get_u32(gf, "tokenizer.ggml.padding_token_id", 0);
    
    /* Validate minimum requirements:
     * For attention-based models, need hidden + layers + heads.
     * For pure Mamba models, heads may be 0 — validate differently. */
    bool is_mamba_pure = (strcmp(arch->arch, "mamba") == 0 || 
                          strcmp(arch->arch, "mamba2") == 0);
    
    if (arch->hidden_size == 0 || arch->n_layers == 0) {
        fprintf(stderr, "gguf: incomplete architecture: hidden=%u layers=%u\n",
                arch->hidden_size, arch->n_layers);
        return false;
    }
    
    if (!is_mamba_pure && arch->n_heads == 0) {
        fprintf(stderr, "gguf: incomplete architecture: heads=%u (not a pure Mamba model)\n",
                arch->n_heads);
        return false;
    }
    
    return true;
}

void gguf_print_info(const gguf_file* gf) {
    printf("\n═══ GGUF File Info ═══\n");
    printf("Version:    %u\n", gf->version);
    printf("Tensors:    %lu\n", (unsigned long)gf->n_tensors);
    printf("KV pairs:   %lu\n", (unsigned long)gf->n_kv);
    printf("Data size:  %.2f GB\n", (double)gf->data_size / (1024.0*1024.0*1024.0));
    
    const char* keys[] = {
        "general.architecture", "general.name",
        NULL
    };
    for (int i = 0; keys[i]; i++) {
        const char* v = gguf_get_string(gf, keys[i]);
        if (v) printf("%-30s = %s\n", keys[i], v);
    }
    
    model_arch arch;
    if (gguf_extract_arch(gf, &arch)) {
        printf("\n─── Architecture: %s ───\n", arch.arch);
        printf("Vocab size:       %u\n", arch.vocab_size);
        printf("Hidden size:      %u\n", arch.hidden_size);
        printf("Layers:           %u\n", arch.n_layers);
        if (arch.n_heads > 0) {
            printf("Attention heads:  %u (KV: %u)\n", arch.n_heads, arch.n_kv_heads);
            printf("Head dim:         %u\n", arch.head_dim);
        }
        if (arch.intermediate_size > 0) {
            printf("FFN size:         %u\n", arch.intermediate_size);
        }
        printf("Max context:      %u\n", arch.max_position);
        if (arch.n_heads > 0) {
            printf("RoPE freq base:   %.1f\n", arch.rope_freq_base);
        }
        printf("RMS norm eps:     %g\n", arch.rms_norm_eps);
        
        /* DeltaNet hybrid info */
        if (arch.full_attn_interval > 0 && arch.ssm_d_inner > 0) {
            printf("─── Hybrid Architecture (DeltaNet) ───\n");
            printf("Attn interval:    every %u layers\n", arch.full_attn_interval);
            printf("SSM inner size:   %u\n", arch.ssm_d_inner);
            printf("SSM state size:   %u\n", arch.ssm_d_state);
            printf("SSM group count:  %u (K heads)\n", arch.ssm_n_group);
            printf("SSM dt rank:      %u (V heads)\n", arch.ssm_dt_rank);
            printf("SSM conv kernel:  %u\n", arch.ssm_conv_kernel);
            uint32_t n_attn = arch.n_layers / arch.full_attn_interval;
            uint32_t n_delta = arch.n_layers - n_attn;
            printf("Attention layers: %u | DeltaNet layers: %u\n", n_attn, n_delta);
        }
        
        /* Pure Mamba / Mamba-2 info */
        bool is_mamba = (strcmp(arch.arch, "mamba") == 0);
        bool is_mamba2 = (strcmp(arch.arch, "mamba2") == 0);
        bool is_jamba = (strcmp(arch.arch, "jamba") == 0);
        bool is_falcon_h1 = (strcmp(arch.arch, "falcon_h1") == 0);
        
        if (is_mamba || is_mamba2) {
            printf("─── %s Architecture ───\n", is_mamba ? "Mamba (S6)" : "Mamba-2 (SSD)");
            printf("SSM inner size:   %u\n", arch.ssm_d_inner);
            printf("SSM state size:   %u\n", arch.ssm_d_state);
            printf("SSM conv kernel:  %u\n", arch.ssm_conv_kernel);
            if (is_mamba) {
                printf("SSM dt rank:      %u\n", arch.ssm_dt_rank);
            }
            if (is_mamba2) {
                printf("SSM heads:        %u\n", arch.ssm_n_head);
                printf("SSM groups:       %u\n", arch.ssm_n_group);
            }
            printf("All %u layers: %s\n", arch.n_layers, is_mamba ? "MAMBA" : "MAMBA2");
        }
        
        if (is_jamba || is_falcon_h1) {
            printf("─── Hybrid Architecture (%s) ───\n", arch.arch);
            printf("SSM inner size:   %u\n", arch.ssm_d_inner);
            printf("SSM state size:   %u\n", arch.ssm_d_state);
            printf("SSM conv kernel:  %u\n", arch.ssm_conv_kernel);
            if (arch.ssm_n_head > 0) {
                printf("SSM heads:        %u\n", arch.ssm_n_head);
            }
            printf("SSM groups:       %u\n", arch.ssm_n_group);
            if (arch.full_attn_interval > 0) {
                printf("Attn interval:    every %u layers\n", arch.full_attn_interval);
            }
        }
    }
    
    /* Tensor type histogram */
    printf("\n─── Tensor Types ───\n");
    int type_counts[32] = {0};
    uint64_t type_bytes[32] = {0};
    for (uint64_t i = 0; i < gf->n_tensors; i++) {
        int t = gf->tensors[i].type;
        if (t < 32) {
            type_counts[t]++;
            type_bytes[t] += gf->tensors[i].n_bytes;
        }
    }
    const char* type_names[] = {
        "F32","F16","Q4_0","Q4_1","","","Q5_0","Q5_1",
        "Q8_0","Q8_1","Q2_K","Q3_K","Q4_K","Q5_K","Q6_K","Q8_K"
    };
    for (int t = 0; t < 16; t++) {
        if (type_counts[t] > 0) {
            printf("  %-6s: %4d tensors, %.2f GB\n",
                   type_names[t], type_counts[t],
                   (double)type_bytes[t] / (1024.0*1024.0*1024.0));
        }
    }
    
    printf("═══════════════════\n\n");
    
    printf("─── First 30 Tensor Names ───\n");
    for (uint64_t i = 0; i < gf->n_tensors && i < 30; i++) {
        printf("  [%lu] %s\n", (unsigned long)i, gf->tensors[i].name.data);
    }
    if (gf->n_tensors > 30) printf("  ... (%lu more)\n", (unsigned long)(gf->n_tensors - 30));
    printf("═══════════════════\n\n");
}
