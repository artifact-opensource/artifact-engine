/*
 * Artifact Engine — Vulkan Compute Interface
 * 
 * Manages Vulkan device, compute pipelines, and shader dispatch.
 * This is the GPU brain — all tensor operations run through here.
 */

#ifndef VULKAN_COMPUTE_H
#define VULKAN_COMPUTE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* Forward declare Vulkan types to avoid requiring vulkan.h in every file */
typedef struct VkInstance_T*       VkInstance;
typedef struct VkPhysicalDevice_T* VkPhysicalDevice;
typedef struct VkDevice_T*         VkDevice;
typedef struct VkQueue_T*          VkQueue;
typedef struct VkCommandPool_T*    VkCommandPool;
typedef struct VkCommandBuffer_T*  VkCommandBuffer;
typedef struct VkBuffer_T*         VkBuffer;
typedef struct VkDeviceMemory_T*   VkDeviceMemory;
typedef struct VkShaderModule_T*   VkShaderModule;
typedef struct VkPipeline_T*       VkPipeline;
typedef struct VkPipelineLayout_T* VkPipelineLayout;
typedef struct VkDescriptorSetLayout_T* VkDescriptorSetLayout;
typedef struct VkDescriptorPool_T* VkDescriptorPool;
typedef struct VkDescriptorSet_T*  VkDescriptorSet;
typedef struct VkFence_T*          VkFence;

/* ───── GPU Buffer ───── */
typedef struct {
    VkBuffer       buffer;
    VkDeviceMemory memory;
    size_t         size;
    void*          mapped;  /* persistently mapped pointer (if host-visible) */
} gpu_buffer;

/* ───── Compute Shader ───── */
typedef enum {
    SHADER_MATMUL = 0,
    SHADER_MATMUL_Q4K,     /* quantized matmul for Q4_K */
    SHADER_MATMUL_Q8,      /* quantized matmul for Q8_0 */
    SHADER_ATTENTION,
    SHADER_RMSNORM,
    SHADER_ROPE,
    SHADER_SILU,
    SHADER_SOFTMAX,
    SHADER_EMBEDDING,
    SHADER_DEQUANT_Q4K,
    SHADER_DEQUANT_Q8,
    SHADER_ADD,
    SHADER_MUL,
    SHADER_COPY,
    SHADER_SAMPLE,
    SHADER_KV_CACHE_STORE,
    SHADER_GQA_ATTENTION,
    SHADER_COUNT
} shader_id;

/* ───── Push Constants (per-dispatch parameters) ───── */
typedef struct {
    uint32_t M;       /* rows of output */
    uint32_t N;       /* cols of output */
    uint32_t K;       /* inner dimension */
    uint32_t stride;  /* for various uses */
    float    scale;   /* attention scale, norm eps, temperature */
    uint32_t offset;  /* position offset for RoPE, buffer offset */
    uint32_t head_dim;
    uint32_t n_heads;
} push_constants;

/* ───── Vulkan Context ───── */
typedef struct {
    VkInstance       instance;
    VkPhysicalDevice physical_device;
    VkDevice         device;
    VkQueue          compute_queue;
    uint32_t         compute_family;
    VkCommandPool    cmd_pool;
    VkCommandBuffer  cmd_buf;
    VkFence          fence;
    
    /* Shader pipelines */
    VkShaderModule        shader_modules[SHADER_COUNT];
    VkPipeline            pipelines[SHADER_COUNT];
    VkPipelineLayout      pipeline_layouts[SHADER_COUNT];
    VkDescriptorSetLayout desc_layouts[SHADER_COUNT];
    VkDescriptorPool      desc_pool;
    
    /* Device properties */
    uint32_t max_workgroup_size;
    uint32_t max_workgroup_count[3];
    size_t   max_buffer_size;
    size_t   device_memory_size;  /* total VRAM */
    size_t   used_memory;         /* tracked allocations */
    char     device_name[256];
    
    bool     initialized;
} vk_context;

/* ───── API ───── */

/* Initialize Vulkan context — finds GPU, creates device, compiles shaders */
bool vk_init(vk_context* ctx, const char* shader_dir);

/* Shutdown and free all Vulkan resources */
void vk_destroy(vk_context* ctx);

/* ─── Buffer Management ─── */

/* Allocate GPU buffer (device-local, not host-visible) */
bool vk_alloc_device(vk_context* ctx, gpu_buffer* buf, size_t size);

/* Allocate staging buffer (host-visible, for upload/download) */
bool vk_alloc_staging(vk_context* ctx, gpu_buffer* buf, size_t size);

/* Upload data from CPU to GPU buffer */
bool vk_upload(vk_context* ctx, gpu_buffer* dst, const void* src, size_t size);

/* Download data from GPU to CPU */
bool vk_download(vk_context* ctx, void* dst, const gpu_buffer* src, size_t size);

/* Free a GPU buffer */
void vk_free_buffer(vk_context* ctx, gpu_buffer* buf);

/* ─── Compute Dispatch ─── */

/* Begin recording compute commands */
void vk_begin_compute(vk_context* ctx);

/* Dispatch a compute shader */
void vk_dispatch(vk_context* ctx, shader_id shader, 
                 const gpu_buffer* buffers[], uint32_t n_buffers,
                 const push_constants* pc,
                 uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);

/* Memory barrier between dispatches */
void vk_barrier(vk_context* ctx);

/* Submit recorded commands and wait for completion */
bool vk_submit_and_wait(vk_context* ctx);

/* ─── High-Level Operations ─── */

/* C = A @ B (matrix multiply, fp32 or dequantized) */
void vk_matmul(vk_context* ctx, 
               const gpu_buffer* A, const gpu_buffer* B, gpu_buffer* C,
               uint32_t M, uint32_t N, uint32_t K);

/* Quantized matmul: C = dequant(A_q4k) @ B */
void vk_matmul_q4k(vk_context* ctx,
                    const gpu_buffer* A_quant, const gpu_buffer* B, gpu_buffer* C,
                    uint32_t M, uint32_t N, uint32_t K);

/* RMS normalization: out = x * rsqrt(mean(x^2) + eps) * weight */
void vk_rmsnorm(vk_context* ctx,
                const gpu_buffer* x, const gpu_buffer* weight, gpu_buffer* out,
                uint32_t n_elements, float eps);

/* Rotary position embedding */
void vk_rope(vk_context* ctx,
             gpu_buffer* q, gpu_buffer* k,
             uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads,
             uint32_t position, float freq_base);

/* Softmax along last dimension */
void vk_softmax(vk_context* ctx, gpu_buffer* x, uint32_t rows, uint32_t cols);

/* SiLU activation: x * sigmoid(x) */
void vk_silu(vk_context* ctx, gpu_buffer* x, uint32_t n_elements);

/* Element-wise add: out = a + b */
void vk_add(vk_context* ctx, 
            const gpu_buffer* a, const gpu_buffer* b, gpu_buffer* out,
            uint32_t n_elements);

/* Element-wise multiply: out = a * b */
void vk_mul(vk_context* ctx,
            const gpu_buffer* a, const gpu_buffer* b, gpu_buffer* out,
            uint32_t n_elements);

/* Token embedding lookup */
void vk_embedding(vk_context* ctx,
                  const gpu_buffer* table, gpu_buffer* out,
                  uint32_t token_id, uint32_t dim);

/* Store K/V vectors into KV cache at given position */
void vk_kv_cache_store(vk_context* ctx,
                       const gpu_buffer* kv_current, gpu_buffer* kv_cache,
                       uint32_t kv_dim, uint32_t pos, uint32_t max_seq);

/* Grouped-Query Attention with KV cache */
void vk_gqa_attention(vk_context* ctx,
                      const gpu_buffer* q,
                      const gpu_buffer* k_cache,
                      const gpu_buffer* v_cache,
                      gpu_buffer* attn_scores,
                      gpu_buffer* attn_out,
                      uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads,
                      uint32_t seq_len, uint32_t max_seq, uint32_t current_pos);

/* ─── DeltaNet Operations ─── */

/* 1D causal convolution: out[i] = sum(w[j] * x[i-j]) for j=0..kernel-1
 * conv_state: ring buffer of (kernel_size-1) previous inputs [channels * (kernel-1)]
 * x: current input [channels]
 * weight: conv kernel [kernel_size * channels] (depthwise)
 * out: result [channels]
 * Updates conv_state in-place. */
void vk_conv1d(vk_context* ctx,
               gpu_buffer* conv_state, const gpu_buffer* x,
               const gpu_buffer* weight, gpu_buffer* out,
               uint32_t channels, uint32_t kernel_size);

/* L2 normalization: out = x / ||x||_2 (per group)
 * x: [total_dim], group_size elements per group, total_dim/group_size groups */
void vk_l2_norm(vk_context* ctx,
                const gpu_buffer* x, gpu_buffer* out,
                uint32_t total_dim, uint32_t group_size);

/* Sigmoid: out = 1 / (1 + exp(-x)), in-place */
void vk_sigmoid(vk_context* ctx, gpu_buffer* x, uint32_t n_elements);

/* Softplus: out = log(1 + exp(x)), in-place */
void vk_softplus(vk_context* ctx, gpu_buffer* x, uint32_t n_elements);

/* DeltaNet recurrent step (single token):
 * Given Q [num_v_heads, head_k_dim], K [num_v_heads, head_k_dim], V [num_v_heads, head_v_dim],
 * beta [num_v_heads] (decay from sigmoid(ssm_a)),
 * S_prev [num_v_heads, head_v_dim, head_k_dim] (state matrix),
 * computes:
 *   S_new = beta * S_prev + V^T @ K  (per head)
 *   out = S_new @ Q  (per head), shaped [num_v_heads * head_v_dim]
 * Updates S in-place. */
void vk_deltanet_step(vk_context* ctx,
                      const gpu_buffer* q, const gpu_buffer* k,
                      const gpu_buffer* v, const gpu_buffer* beta,
                      gpu_buffer* state, gpu_buffer* out,
                      uint32_t num_v_heads, uint32_t head_k_dim,
                      uint32_t head_v_dim);

/* RMS normalization per-head (for DeltaNet ssm_norm)
 * x: [num_heads * head_dim], applies norm within each head_dim chunk
 * weight: [head_dim] (shared across heads) */
void vk_rmsnorm_head(vk_context* ctx,
                     const gpu_buffer* x, const gpu_buffer* weight,
                     gpu_buffer* out,
                     uint32_t num_heads, uint32_t head_dim, float eps);

/* ─── Mamba Operations ─── */

/* Mamba-1 selective scan step (single token, autoregressive).
 * Implements: s[i] = s[i] * exp(dt * A[i]) + B * (x * dt)
 *             y = sum_i(s[i] * C[i]) + D * x
 *
 * x:     [d_inner] — input (after conv + silu)
 * dt:    [d_inner] — time step (after softplus)
 * A:     [d_state, d_inner] — diagonal decay (negative, stored as log)
 * B:     [d_state] — input projection
 * C:     [d_state] — output projection
 * D:     [d_inner] — skip connection
 * state: [d_inner * d_state] — recurrent state (updated in-place)
 * y:     [d_inner] — output */
void vk_mamba1_ssm_step(vk_context* ctx,
                        const gpu_buffer* x, const gpu_buffer* dt,
                        const gpu_buffer* A, const gpu_buffer* B,
                        const gpu_buffer* C, const gpu_buffer* D,
                        gpu_buffer* state, gpu_buffer* y,
                        uint32_t d_inner, uint32_t d_state);

/* Mamba-2 selective scan step (single token, autoregressive).
 * Mamba-2 uses scalar decay per head (not per element like Mamba-1).
 *
 * x:     [n_head * dim_per_head] — input (d_inner = n_head * dim)
 * dt:    [n_head] — time step per head (after softplus)
 * A:     [n_head] — scalar decay per head (stored as -exp(A_log))
 * B:     [n_group * d_state] — input projection (groups shared across heads)
 * C:     [n_group * d_state] — output projection
 * D:     [n_head] — skip connection per head
 * state: [n_head * dim * d_state] — recurrent state (updated in-place)
 * y:     [n_head * dim] — output (d_inner) */
void vk_mamba2_ssm_step(vk_context* ctx,
                        const gpu_buffer* x, const gpu_buffer* dt,
                        const gpu_buffer* A, const gpu_buffer* B,
                        const gpu_buffer* C, const gpu_buffer* D,
                        gpu_buffer* state, gpu_buffer* y,
                        uint32_t n_head, uint32_t dim_per_head,
                        uint32_t d_state, uint32_t n_group);

/* Bias addition: x[i] += bias[i], in-place */
void vk_add_bias(vk_context* ctx, gpu_buffer* x, const gpu_buffer* bias, uint32_t n);

/* ─── Diagnostics ─── */

/* Print GPU info */
void vk_print_info(const vk_context* ctx);

/* Get memory usage */
size_t vk_memory_used(const vk_context* ctx);
size_t vk_memory_total(const vk_context* ctx);

#endif /* VULKAN_COMPUTE_H */
