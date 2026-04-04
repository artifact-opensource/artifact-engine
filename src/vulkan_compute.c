/*
 * Artifact Engine — Vulkan Compute Backend
 *
 * Initializes Vulkan, loads SPIR-V shaders, manages GPU buffers,
 * and dispatches compute workloads.
 *
 * No rendering — pure compute pipeline.
 */

#define _GNU_SOURCE
#include "../include/vulkan_compute.h"

/* Now we need the real Vulkan headers */
#include <vulkan/vulkan.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* ───── Shader filenames (match shader_id enum order) ───── */
static const char* shader_names[SHADER_COUNT] = {
    [SHADER_MATMUL]      = "matmul.spv",
    [SHADER_MATMUL_Q4K]  = "dequant_q4k.spv",  /* fused dequant+matmul */
    [SHADER_MATMUL_Q8]   = "matmul.spv",        /* TODO: dedicated Q8 shader */
    [SHADER_ATTENTION]   = "matmul.spv",         /* attention uses matmul */
    [SHADER_RMSNORM]     = "rmsnorm.spv",
    [SHADER_ROPE]        = "rope.spv",
    [SHADER_SILU]        = "silu.spv",
    [SHADER_SOFTMAX]     = "softmax.spv",
    [SHADER_EMBEDDING]   = "embedding.spv",
    [SHADER_DEQUANT_Q4K] = "dequant_q4k.spv",
    [SHADER_DEQUANT_Q8]  = "matmul.spv",        /* placeholder */
    [SHADER_ADD]         = "add.spv",
    [SHADER_MUL]         = "mul.spv",
    [SHADER_COPY]        = "matmul.spv",         /* placeholder */
    [SHADER_SAMPLE]      = "matmul.spv",         /* placeholder */
};

/* Max bindings per shader (we use up to 3: A, B, C) */
#define MAX_BINDINGS 4

/* Number of bindings each shader uses */
static const uint32_t shader_binding_count[SHADER_COUNT] = {
    [SHADER_MATMUL]      = 3,
    [SHADER_MATMUL_Q4K]  = 3,
    [SHADER_MATMUL_Q8]   = 3,
    [SHADER_ATTENTION]   = 3,
    [SHADER_RMSNORM]     = 3,
    [SHADER_ROPE]        = 2,
    [SHADER_SILU]        = 1,
    [SHADER_SOFTMAX]     = 1,
    [SHADER_EMBEDDING]   = 2,
    [SHADER_DEQUANT_Q4K] = 3,
    [SHADER_DEQUANT_Q8]  = 3,
    [SHADER_ADD]         = 3,
    [SHADER_MUL]         = 3,
    [SHADER_COPY]        = 2,
    [SHADER_SAMPLE]      = 2,
};

/* ───── Helper: Check Vulkan result ───── */
#define VK_CHECK(call) do { \
    VkResult _r = (call); \
    if (_r != VK_SUCCESS) { \
        fprintf(stderr, "Vulkan error %d at %s:%d\n", _r, __FILE__, __LINE__); \
        return false; \
    } \
} while(0)

/* ───── Load SPIR-V file ───── */
static uint32_t* load_spirv(const char* path, size_t* size) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "vk: cannot open shader: %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    uint32_t* code = malloc(*size);
    if (!code) { fclose(f); return NULL; }
    fread(code, 1, *size, f);
    fclose(f);
    return code;
}

/* ───── Find memory type ───── */
static uint32_t find_memory_type(VkPhysicalDevice pdev, uint32_t type_filter,
                                  VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(pdev, &mem_props);
    
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return UINT32_MAX;
}

/* ───── Initialize Vulkan ───── */
bool vk_init(vk_context* ctx, const char* shader_dir) {
    memset(ctx, 0, sizeof(*ctx));
    
    /* 1. Create instance */
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Artifact Engine",
        .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
        .pEngineName = "Artifact",
        .engineVersion = VK_MAKE_VERSION(0, 1, 0),
        .apiVersion = VK_API_VERSION_1_2,
    };
    
    VkInstanceCreateInfo inst_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
    };
    
    VK_CHECK(vkCreateInstance(&inst_info, NULL, &ctx->instance));
    
    /* 2. Pick physical device (prefer discrete GPU) */
    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(ctx->instance, &dev_count, NULL);
    if (dev_count == 0) {
        fprintf(stderr, "vk: no Vulkan-capable GPU found\n");
        return false;
    }
    
    VkPhysicalDevice* devices = malloc(dev_count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(ctx->instance, &dev_count, devices);
    
    /* Pick best device: prefer discrete > integrated > other */
    int best_score = -1;
    for (uint32_t i = 0; i < dev_count; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);
        
        int score = 0;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score = 100;
        else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score = 50;
        else score = 10;
        
        if (score > best_score) {
            best_score = score;
            ctx->physical_device = devices[i];
        }
    }
    free(devices);
    
    /* Get device properties */
    VkPhysicalDeviceProperties dev_props;
    vkGetPhysicalDeviceProperties(ctx->physical_device, &dev_props);
    strncpy(ctx->device_name, dev_props.deviceName, sizeof(ctx->device_name) - 1);
    ctx->max_workgroup_size = dev_props.limits.maxComputeWorkGroupSize[0];
    ctx->max_workgroup_count[0] = dev_props.limits.maxComputeWorkGroupCount[0];
    ctx->max_workgroup_count[1] = dev_props.limits.maxComputeWorkGroupCount[1];
    ctx->max_workgroup_count[2] = dev_props.limits.maxComputeWorkGroupCount[2];
    ctx->max_buffer_size = dev_props.limits.maxStorageBufferRange;
    
    /* Get VRAM size */
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(ctx->physical_device, &mem_props);
    ctx->device_memory_size = 0;
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
        if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            ctx->device_memory_size += mem_props.memoryHeaps[i].size;
        }
    }
    
    printf("vk: GPU: %s\n", ctx->device_name);
    printf("vk: VRAM: %.2f GB\n", (double)ctx->device_memory_size / (1024.0*1024.0*1024.0));
    printf("vk: Max workgroup: %u, Max buffer: %.2f GB\n",
           ctx->max_workgroup_size,
           (double)ctx->max_buffer_size / (1024.0*1024.0*1024.0));
    
    /* 3. Find compute queue family */
    uint32_t queue_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical_device, &queue_count, NULL);
    VkQueueFamilyProperties* queue_props = malloc(queue_count * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical_device, &queue_count, queue_props);
    
    ctx->compute_family = UINT32_MAX;
    for (uint32_t i = 0; i < queue_count; i++) {
        if (queue_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            ctx->compute_family = i;
            /* Prefer a dedicated compute queue (no graphics) */
            if (!(queue_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) break;
        }
    }
    free(queue_props);
    
    if (ctx->compute_family == UINT32_MAX) {
        fprintf(stderr, "vk: no compute queue found\n");
        return false;
    }
    
    /* 4. Create logical device */
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = ctx->compute_family,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };
    
    VkDeviceCreateInfo dev_create = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
    };
    
    VK_CHECK(vkCreateDevice(ctx->physical_device, &dev_create, NULL, &ctx->device));
    vkGetDeviceQueue(ctx->device, ctx->compute_family, 0, &ctx->compute_queue);
    
    /* 5. Create command pool + buffer */
    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = ctx->compute_family,
    };
    VK_CHECK(vkCreateCommandPool(ctx->device, &pool_info, NULL, &ctx->cmd_pool));
    
    VkCommandBufferAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VK_CHECK(vkAllocateCommandBuffers(ctx->device, &alloc_info, &ctx->cmd_buf));
    
    /* 6. Create fence for synchronization */
    VkFenceCreateInfo fence_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    };
    VK_CHECK(vkCreateFence(ctx->device, &fence_info, NULL, &ctx->fence));
    
    /* 7. Create descriptor pool (large enough for all our dispatches) */
    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1024 },
    };
    VkDescriptorPoolCreateInfo desc_pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 256,
        .poolSizeCount = 1,
        .pPoolSizes = pool_sizes,
    };
    VK_CHECK(vkCreateDescriptorPool(ctx->device, &desc_pool_info, NULL, &ctx->desc_pool));
    
    /* 8. Load and compile shaders, create pipelines */
    for (int s = 0; s < SHADER_COUNT; s++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/%s", shader_dir, shader_names[s]);
        
        size_t code_size = 0;
        uint32_t* code = load_spirv(path, &code_size);
        if (!code) {
            fprintf(stderr, "vk: warning: shader %s not found, skipping\n", shader_names[s]);
            continue;
        }
        
        /* Create shader module */
        VkShaderModuleCreateInfo module_info = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = code_size,
            .pCode = code,
        };
        VK_CHECK(vkCreateShaderModule(ctx->device, &module_info, NULL, &ctx->shader_modules[s]));
        free(code);
        
        /* Create descriptor set layout */
        uint32_t n_bindings = shader_binding_count[s];
        VkDescriptorSetLayoutBinding bindings[MAX_BINDINGS];
        for (uint32_t b = 0; b < n_bindings; b++) {
            bindings[b] = (VkDescriptorSetLayoutBinding){
                .binding = b,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            };
        }
        
        VkDescriptorSetLayoutCreateInfo layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = n_bindings,
            .pBindings = bindings,
        };
        VK_CHECK(vkCreateDescriptorSetLayout(ctx->device, &layout_info, NULL, &ctx->desc_layouts[s]));
        
        /* Create pipeline layout with push constants */
        VkPushConstantRange pc_range = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(push_constants),
        };
        
        VkPipelineLayoutCreateInfo pipe_layout_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &ctx->desc_layouts[s],
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pc_range,
        };
        VK_CHECK(vkCreatePipelineLayout(ctx->device, &pipe_layout_info, NULL, &ctx->pipeline_layouts[s]));
        
        /* Create compute pipeline */
        VkComputePipelineCreateInfo pipe_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = ctx->shader_modules[s],
                .pName = "main",
            },
            .layout = ctx->pipeline_layouts[s],
        };
        VK_CHECK(vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &pipe_info, NULL, &ctx->pipelines[s]));
    }
    
    ctx->initialized = true;
    printf("vk: initialized (%d shaders loaded)\n", SHADER_COUNT);
    return true;
}

/* ───── Buffer Management ───── */

static bool vk_alloc_buffer(vk_context* ctx, gpu_buffer* buf, size_t size,
                             VkBufferUsageFlags usage, VkMemoryPropertyFlags mem_flags) {
    buf->size = size;
    buf->mapped = NULL;
    
    VkBufferCreateInfo buf_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VK_CHECK(vkCreateBuffer(ctx->device, &buf_info, NULL, &buf->buffer));
    
    VkMemoryRequirements mem_req;
    vkGetBufferMemoryRequirements(ctx->device, buf->buffer, &mem_req);
    
    uint32_t mem_type = find_memory_type(ctx->physical_device, mem_req.memoryTypeBits, mem_flags);
    if (mem_type == UINT32_MAX) {
        fprintf(stderr, "vk: no suitable memory type found\n");
        return false;
    }
    
    VkMemoryAllocateInfo alloc = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_req.size,
        .memoryTypeIndex = mem_type,
    };
    VK_CHECK(vkAllocateMemory(ctx->device, &alloc, NULL, &buf->memory));
    VK_CHECK(vkBindBufferMemory(ctx->device, buf->buffer, buf->memory, 0));
    
    /* Persistently map host-visible buffers */
    if (mem_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        VK_CHECK(vkMapMemory(ctx->device, buf->memory, 0, size, 0, &buf->mapped));
    }
    
    ctx->used_memory += size;
    return true;
}

bool vk_alloc_device(vk_context* ctx, gpu_buffer* buf, size_t size) {
    return vk_alloc_buffer(ctx, buf, size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

bool vk_alloc_staging(vk_context* ctx, gpu_buffer* buf, size_t size) {
    return vk_alloc_buffer(ctx, buf, size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

bool vk_upload(vk_context* ctx, gpu_buffer* dst, const void* src, size_t size) {
    /* Use a staging buffer for device-local uploads */
    gpu_buffer staging;
    if (!vk_alloc_staging(ctx, &staging, size)) return false;
    
    memcpy(staging.mapped, src, size);
    
    /* Record copy command */
    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkResetCommandBuffer(ctx->cmd_buf, 0);
    vkBeginCommandBuffer(ctx->cmd_buf, &begin);
    
    VkBufferCopy region = { .size = size };
    vkCmdCopyBuffer(ctx->cmd_buf, staging.buffer, dst->buffer, 1, &region);
    
    vkEndCommandBuffer(ctx->cmd_buf);
    
    VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &ctx->cmd_buf,
    };
    vkResetFences(ctx->device, 1, &ctx->fence);
    vkQueueSubmit(ctx->compute_queue, 1, &submit, ctx->fence);
    vkWaitForFences(ctx->device, 1, &ctx->fence, VK_TRUE, UINT64_MAX);
    
    vk_free_buffer(ctx, &staging);
    return true;
}

bool vk_download(vk_context* ctx, void* dst, const gpu_buffer* src, size_t size) {
    gpu_buffer staging;
    if (!vk_alloc_staging(ctx, &staging, size)) return false;
    
    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkResetCommandBuffer(ctx->cmd_buf, 0);
    vkBeginCommandBuffer(ctx->cmd_buf, &begin);
    
    VkBufferCopy region = { .size = size };
    vkCmdCopyBuffer(ctx->cmd_buf, src->buffer, staging.buffer, 1, &region);
    
    vkEndCommandBuffer(ctx->cmd_buf);
    
    VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &ctx->cmd_buf,
    };
    vkResetFences(ctx->device, 1, &ctx->fence);
    vkQueueSubmit(ctx->compute_queue, 1, &submit, ctx->fence);
    vkWaitForFences(ctx->device, 1, &ctx->fence, VK_TRUE, UINT64_MAX);
    
    memcpy(dst, staging.mapped, size);
    vk_free_buffer(ctx, &staging);
    return true;
}

void vk_free_buffer(vk_context* ctx, gpu_buffer* buf) {
    if (buf->mapped) {
        vkUnmapMemory(ctx->device, buf->memory);
        buf->mapped = NULL;
    }
    if (buf->buffer) {
        vkDestroyBuffer(ctx->device, buf->buffer, NULL);
        buf->buffer = VK_NULL_HANDLE;
    }
    if (buf->memory) {
        vkFreeMemory(ctx->device, buf->memory, NULL);
        buf->memory = VK_NULL_HANDLE;
    }
    if (ctx->used_memory >= buf->size) ctx->used_memory -= buf->size;
    buf->size = 0;
}

/* ───── Compute Dispatch ───── */

void vk_begin_compute(vk_context* ctx) {
    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkResetCommandBuffer(ctx->cmd_buf, 0);
    vkBeginCommandBuffer(ctx->cmd_buf, &begin);
}

void vk_dispatch(vk_context* ctx, shader_id shader,
                 const gpu_buffer* buffers[], uint32_t n_buffers,
                 const push_constants* pc,
                 uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
    
    /* Allocate descriptor set */
    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = ctx->desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &ctx->desc_layouts[shader],
    };
    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(ctx->device, &alloc_info, &desc_set);
    
    /* Update descriptor set with buffer bindings */
    VkDescriptorBufferInfo buf_infos[MAX_BINDINGS];
    VkWriteDescriptorSet writes[MAX_BINDINGS];
    
    for (uint32_t i = 0; i < n_buffers && i < MAX_BINDINGS; i++) {
        buf_infos[i] = (VkDescriptorBufferInfo){
            .buffer = buffers[i]->buffer,
            .offset = 0,
            .range = buffers[i]->size,
        };
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = desc_set,
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buf_infos[i],
        };
    }
    vkUpdateDescriptorSets(ctx->device, n_buffers, writes, 0, NULL);
    
    /* Bind pipeline and descriptors */
    vkCmdBindPipeline(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[shader]);
    vkCmdBindDescriptorSets(ctx->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layouts[shader], 0, 1, &desc_set, 0, NULL);
    
    /* Push constants */
    if (pc) {
        vkCmdPushConstants(ctx->cmd_buf, ctx->pipeline_layouts[shader],
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), pc);
    }
    
    /* Dispatch! */
    vkCmdDispatch(ctx->cmd_buf, groups_x, groups_y, groups_z);
}

void vk_barrier(vk_context* ctx) {
    VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };
    vkCmdPipelineBarrier(ctx->cmd_buf,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, NULL, 0, NULL);
}

bool vk_submit_and_wait(vk_context* ctx) {
    vkEndCommandBuffer(ctx->cmd_buf);
    
    VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &ctx->cmd_buf,
    };
    vkResetFences(ctx->device, 1, &ctx->fence);
    VkResult r = vkQueueSubmit(ctx->compute_queue, 1, &submit, ctx->fence);
    if (r != VK_SUCCESS) return false;
    
    r = vkWaitForFences(ctx->device, 1, &ctx->fence, VK_TRUE, UINT64_MAX);
    
    /* Free descriptor sets for reuse */
    vkResetDescriptorPool(ctx->device, ctx->desc_pool, 0);
    
    return r == VK_SUCCESS;
}

/* ───── High-Level Operations ───── */

void vk_matmul(vk_context* ctx,
               const gpu_buffer* A, const gpu_buffer* B, gpu_buffer* C,
               uint32_t M, uint32_t N, uint32_t K) {
    const gpu_buffer* bufs[] = { A, B, C };
    push_constants pc = { .M = M, .N = N, .K = K, .scale = 1.0f };
    uint32_t gx = (N + 15) / 16;
    uint32_t gy = (M + 15) / 16;
    vk_dispatch(ctx, SHADER_MATMUL, bufs, 3, &pc, gx, gy, 1);
}

void vk_matmul_q4k(vk_context* ctx,
                    const gpu_buffer* A_quant, const gpu_buffer* B, gpu_buffer* C,
                    uint32_t M, uint32_t N, uint32_t K) {
    const gpu_buffer* bufs[] = { A_quant, B, C };
    push_constants pc = { .M = M, .N = N, .K = K };
    uint32_t total = M * N;
    vk_dispatch(ctx, SHADER_MATMUL_Q4K, bufs, 3, &pc, (total + 255) / 256, 1, 1);
}

void vk_rmsnorm(vk_context* ctx,
                const gpu_buffer* x, const gpu_buffer* weight, gpu_buffer* out,
                uint32_t n_elements, float eps) {
    /* n_elements is hidden_size, one workgroup per row */
    const gpu_buffer* bufs[] = { x, weight, out };
    push_constants pc = { .M = 1, .N = n_elements, .scale = eps };
    vk_dispatch(ctx, SHADER_RMSNORM, bufs, 3, &pc, 1, 1, 1);
}

void vk_rope(vk_context* ctx,
             gpu_buffer* q, gpu_buffer* k,
             uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads,
             uint32_t position, float freq_base) {
    const gpu_buffer* bufs[] = { q, k };
    push_constants pc = {
        .M = 1,
        .N = n_heads * head_dim,
        .K = n_kv_heads * head_dim,
        .scale = freq_base,
        .offset = position,
        .head_dim = head_dim,
        .n_heads = n_heads,
    };
    vk_dispatch(ctx, SHADER_ROPE, bufs, 2, &pc, (head_dim / 2 + 255) / 256, 1, 1);
}

void vk_softmax(vk_context* ctx, gpu_buffer* x, uint32_t rows, uint32_t cols) {
    const gpu_buffer* bufs[] = { x };
    push_constants pc = { .M = rows, .N = cols, .scale = 1.0f };
    vk_dispatch(ctx, SHADER_SOFTMAX, bufs, 1, &pc, rows, 1, 1);
}

void vk_silu(vk_context* ctx, gpu_buffer* x, uint32_t n_elements) {
    const gpu_buffer* bufs[] = { x };
    push_constants pc = { .M = n_elements };
    vk_dispatch(ctx, SHADER_SILU, bufs, 1, &pc, (n_elements + 255) / 256, 1, 1);
}

void vk_add(vk_context* ctx,
            const gpu_buffer* a, const gpu_buffer* b, gpu_buffer* out,
            uint32_t n_elements) {
    const gpu_buffer* bufs[] = { a, b, out };
    push_constants pc = { .M = n_elements };
    vk_dispatch(ctx, SHADER_ADD, bufs, 3, &pc, (n_elements + 255) / 256, 1, 1);
}

void vk_mul(vk_context* ctx,
            const gpu_buffer* a, const gpu_buffer* b, gpu_buffer* out,
            uint32_t n_elements) {
    const gpu_buffer* bufs[] = { a, b, out };
    push_constants pc = { .M = n_elements };
    vk_dispatch(ctx, SHADER_MUL, bufs, 3, &pc, (n_elements + 255) / 256, 1, 1);
}

void vk_embedding(vk_context* ctx,
                  const gpu_buffer* table, gpu_buffer* out,
                  uint32_t token_id, uint32_t dim) {
    const gpu_buffer* bufs[] = { table, out };
    push_constants pc = { .M = token_id, .N = dim, .K = 0 };
    vk_dispatch(ctx, SHADER_EMBEDDING, bufs, 2, &pc, (dim + 255) / 256, 1, 1);
}

/* ───── Diagnostics ───── */

void vk_print_info(const vk_context* ctx) {
    printf("\n═══ Vulkan GPU Info ═══\n");
    printf("Device:     %s\n", ctx->device_name);
    printf("VRAM:       %.2f GB (%.2f GB used)\n",
           (double)ctx->device_memory_size / (1024.0*1024.0*1024.0),
           (double)ctx->used_memory / (1024.0*1024.0*1024.0));
    printf("Max WG:     %u\n", ctx->max_workgroup_size);
    printf("Max Buffer: %.2f GB\n", (double)ctx->max_buffer_size / (1024.0*1024.0*1024.0));
    printf("═══════════════════════\n\n");
}

size_t vk_memory_used(const vk_context* ctx) { return ctx->used_memory; }
size_t vk_memory_total(const vk_context* ctx) { return ctx->device_memory_size; }

/* ───── Cleanup ───── */

void vk_destroy(vk_context* ctx) {
    if (!ctx->initialized) return;
    
    vkDeviceWaitIdle(ctx->device);
    
    for (int s = 0; s < SHADER_COUNT; s++) {
        if (ctx->pipelines[s]) vkDestroyPipeline(ctx->device, ctx->pipelines[s], NULL);
        if (ctx->pipeline_layouts[s]) vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layouts[s], NULL);
        if (ctx->desc_layouts[s]) vkDestroyDescriptorSetLayout(ctx->device, ctx->desc_layouts[s], NULL);
        if (ctx->shader_modules[s]) vkDestroyShaderModule(ctx->device, ctx->shader_modules[s], NULL);
    }
    
    if (ctx->desc_pool) vkDestroyDescriptorPool(ctx->device, ctx->desc_pool, NULL);
    if (ctx->fence) vkDestroyFence(ctx->device, ctx->fence, NULL);
    if (ctx->cmd_pool) vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
    if (ctx->device) vkDestroyDevice(ctx->device, NULL);
    if (ctx->instance) vkDestroyInstance(ctx->instance, NULL);
    
    memset(ctx, 0, sizeof(*ctx));
}
