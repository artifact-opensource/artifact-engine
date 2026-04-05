/*
 * Artifact Engine — DirectML Compute Backend
 *
 * Drop-in replacement for vulkan_compute.c / cpu_compute.c.
 * Implements the same vk_* API using DirectML on Direct3D 12.
 *
 * This is the key backend for Xbox Series X — shares the RDNA 2 GPU
 * with game rendering. DirectML is native to Xbox OS.
 *
 * Compile with -DDIRECTML_BACKEND to use instead of Vulkan/CPU.
 *
 * Architecture:
 *   D3D12 Device → DML Device → Operator Graph → Compiled → Dispatch
 *   All tensor ops (matmul, attention, norms) run as DML operators.
 *   GPU memory managed via D3D12 heaps (same allocator as game titles).
 *
 * Why DirectML over Vulkan on Xbox:
 *   - Vulkan is NOT available on Xbox (DirectX only)
 *   - DirectML is the ML inference API built into Windows/Xbox
 *   - Shares GPU compute with game rendering (async compute queue)
 *   - Xbox ML Super Resolution already proves this works
 *   - Native quantized tensor support (INT4, INT8, FP16)
 */

#ifdef DIRECTML_BACKEND

/* Must define WIN32_LEAN_AND_MEAN before windows.h */
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include "../include/vulkan_compute.h"  /* Same API surface */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ─────────────────────────────────────────────────────────
 * D3D12 + DirectML Headers
 *
 * On Xbox, these come from the GDK (Game Development Kit).
 * On desktop Windows, from the Windows SDK + DirectML NuGet.
 * ───────────────────────────────────────────────────────── */

#include <d3d12.h>
#include <dxgi1_6.h>
#include <DirectML.h>

/* For MSVC, link against these */
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "directml.lib")

/* ───── Internal State ───── */

typedef struct {
    /* D3D12 core */
    ID3D12Device*              d3d_device;
    ID3D12CommandQueue*        cmd_queue;
    ID3D12CommandAllocator*    cmd_alloc;
    ID3D12GraphicsCommandList* cmd_list;
    ID3D12Fence*               fence;
    HANDLE                     fence_event;
    UINT64                     fence_value;

    /* DirectML */
    IDMLDevice*                dml_device;
    IDMLCommandRecorder*       dml_recorder;

    /* Memory heaps */
    ID3D12Heap*                default_heap;     /* GPU-local (device) */
    ID3D12Heap*                upload_heap;       /* CPU→GPU staging */
    ID3D12Heap*                readback_heap;     /* GPU→CPU readback */

    /* Compiled operator cache */
    /* TODO: hash map of (op_type, dims) → IDMLCompiledOperator* */
} dml_state;

static dml_state g_dml = {0};

/* ───── Error Helpers ───── */

#define DML_CHECK(hr, msg) \
    if (FAILED(hr)) { \
        fprintf(stderr, "directml: %s (HRESULT: 0x%08lX)\n", msg, (unsigned long)hr); \
        return false; \
    }

#define DML_CHECK_VOID(hr, msg) \
    if (FAILED(hr)) { \
        fprintf(stderr, "directml: %s (HRESULT: 0x%08lX)\n", msg, (unsigned long)hr); \
        return; \
    }

/* ───── Fence sync helper ───── */

static bool dml_sync(void) {
    HRESULT hr;
    g_dml.fence_value++;
    hr = g_dml.cmd_queue->lpVtbl->Signal(g_dml.cmd_queue,
                                          g_dml.fence,
                                          g_dml.fence_value);
    if (FAILED(hr)) return false;

    if (g_dml.fence->lpVtbl->GetCompletedValue(g_dml.fence) < g_dml.fence_value) {
        g_dml.fence->lpVtbl->SetEventOnCompletion(g_dml.fence,
                                                    g_dml.fence_value,
                                                    g_dml.fence_event);
        WaitForSingleObject(g_dml.fence_event, INFINITE);
    }
    return true;
}

/* ───── Command list helpers ───── */

static bool dml_begin_commands(void) {
    HRESULT hr;
    hr = g_dml.cmd_alloc->lpVtbl->Reset(g_dml.cmd_alloc);
    if (FAILED(hr)) return false;
    hr = g_dml.cmd_list->lpVtbl->Reset(g_dml.cmd_list, g_dml.cmd_alloc, NULL);
    return SUCCEEDED(hr);
}

static bool dml_submit_commands(void) {
    HRESULT hr;
    hr = g_dml.cmd_list->lpVtbl->Close(g_dml.cmd_list);
    if (FAILED(hr)) return false;

    ID3D12CommandList* lists[] = { (ID3D12CommandList*)g_dml.cmd_list };
    g_dml.cmd_queue->lpVtbl->ExecuteCommandLists(g_dml.cmd_queue, 1, lists);
    return dml_sync();
}

/* ═══════════════════════════════════════════════════════════
 * vk_* API Implementation (same interface as Vulkan/CPU)
 * ═══════════════════════════════════════════════════════════ */

bool vk_init(vk_context* ctx, const char* shader_dir) {
    (void)shader_dir;  /* DirectML doesn't use SPIR-V shaders */
    memset(ctx, 0, sizeof(*ctx));
    HRESULT hr;

    /* ── 1. Create D3D12 Device ── */
    /* On Xbox, we skip adapter enumeration — there's only one GPU */
#ifdef _GAMING_XBOX
    D3D12XBOX_CREATE_DEVICE_PARAMETERS params = {0};
    params.Version = D3D12_SDK_VERSION;
    hr = D3D12XboxCreateDevice(NULL, &params,
                                IID_PPV_ARGS(&g_dml.d3d_device));
#else
    /* Desktop: create with default adapter (RDNA 2 / whatever's primary) */
    hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_12_0,
                           &IID_ID3D12Device, (void**)&g_dml.d3d_device);
#endif
    DML_CHECK(hr, "failed to create D3D12 device");

    /* ── 2. Create Compute Command Queue ── */
    /* COMPUTE queue (not DIRECT) — doesn't block game rendering */
    D3D12_COMMAND_QUEUE_DESC queue_desc = {0};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    queue_desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    hr = g_dml.d3d_device->lpVtbl->CreateCommandQueue(
        g_dml.d3d_device, &queue_desc,
        &IID_ID3D12CommandQueue, (void**)&g_dml.cmd_queue);
    DML_CHECK(hr, "failed to create compute command queue");

    /* ── 3. Command Allocator + List ── */
    hr = g_dml.d3d_device->lpVtbl->CreateCommandAllocator(
        g_dml.d3d_device, D3D12_COMMAND_LIST_TYPE_COMPUTE,
        &IID_ID3D12CommandAllocator, (void**)&g_dml.cmd_alloc);
    DML_CHECK(hr, "failed to create command allocator");

    hr = g_dml.d3d_device->lpVtbl->CreateCommandList(
        g_dml.d3d_device, 0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
        g_dml.cmd_alloc, NULL,
        &IID_ID3D12GraphicsCommandList, (void**)&g_dml.cmd_list);
    DML_CHECK(hr, "failed to create command list");
    g_dml.cmd_list->lpVtbl->Close(g_dml.cmd_list);

    /* ── 4. Fence for GPU sync ── */
    g_dml.fence_value = 0;
    hr = g_dml.d3d_device->lpVtbl->CreateFence(
        g_dml.d3d_device, 0, D3D12_FENCE_FLAG_NONE,
        &IID_ID3D12Fence, (void**)&g_dml.fence);
    DML_CHECK(hr, "failed to create fence");
    g_dml.fence_event = CreateEvent(NULL, FALSE, FALSE, NULL);

    /* ── 5. Create DirectML Device ── */
    hr = DMLCreateDevice(g_dml.d3d_device,
                         DML_CREATE_DEVICE_FLAG_NONE,
                         &IID_IDMLDevice, (void**)&g_dml.dml_device);
    DML_CHECK(hr, "failed to create DirectML device");

    /* ── 6. Create DML Command Recorder ── */
    hr = g_dml.dml_device->lpVtbl->CreateCommandRecorder(
        g_dml.dml_device,
        &IID_IDMLCommandRecorder, (void**)&g_dml.dml_recorder);
    DML_CHECK(hr, "failed to create DML command recorder");

    /* ── 7. Query device info ── */
    /* Get adapter description for name + VRAM */
    IDXGIFactory4* factory = NULL;
    IDXGIAdapter1* adapter = NULL;
    hr = CreateDXGIFactory1(&IID_IDXGIFactory4, (void**)&factory);
    if (SUCCEEDED(hr)) {
        hr = factory->lpVtbl->EnumAdapters1(factory, 0, &adapter);
        if (SUCCEEDED(hr)) {
            DXGI_ADAPTER_DESC1 desc;
            adapter->lpVtbl->GetDesc1(adapter, &desc);

            /* Convert wide string device name to char */
            WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1,
                                ctx->device_name, sizeof(ctx->device_name),
                                NULL, NULL);
            ctx->device_memory_size = desc.DedicatedVideoMemory;

            adapter->lpVtbl->Release(adapter);
        }
        factory->lpVtbl->Release(factory);
    }

    ctx->used_memory = 0;
    ctx->max_buffer_size = ctx->device_memory_size;
    ctx->initialized = true;

    printf("directml: initialized on %s\n", ctx->device_name);
    printf("directml: VRAM: %zu MB\n", ctx->device_memory_size / (1024 * 1024));
    printf("directml: using COMPUTE queue (async, non-blocking)\n");

    return true;
}

void vk_destroy(vk_context* ctx) {
    if (!ctx || !ctx->initialized) return;

    if (g_dml.dml_recorder) g_dml.dml_recorder->lpVtbl->Release(g_dml.dml_recorder);
    if (g_dml.dml_device) g_dml.dml_device->lpVtbl->Release(g_dml.dml_device);
    if (g_dml.fence_event) CloseHandle(g_dml.fence_event);
    if (g_dml.fence) g_dml.fence->lpVtbl->Release(g_dml.fence);
    if (g_dml.cmd_list) g_dml.cmd_list->lpVtbl->Release(g_dml.cmd_list);
    if (g_dml.cmd_alloc) g_dml.cmd_alloc->lpVtbl->Release(g_dml.cmd_alloc);
    if (g_dml.cmd_queue) g_dml.cmd_queue->lpVtbl->Release(g_dml.cmd_queue);
    if (g_dml.d3d_device) g_dml.d3d_device->lpVtbl->Release(g_dml.d3d_device);

    memset(&g_dml, 0, sizeof(g_dml));
    ctx->initialized = false;
    printf("directml: destroyed\n");
}

/* ───── Buffer Management ─────
 *
 * D3D12 buffers map to our gpu_buffer abstraction:
 *   - Device-local: D3D12_HEAP_TYPE_DEFAULT (GPU VRAM)
 *   - Staging:      D3D12_HEAP_TYPE_UPLOAD  (CPU→GPU)
 *   - Readback:     D3D12_HEAP_TYPE_READBACK (GPU→CPU)
 *
 * We store the ID3D12Resource* in gpu_buffer.buffer (cast).
 * gpu_buffer.memory is unused (D3D12 manages heaps internally).
 * gpu_buffer.mapped is set for upload/readback buffers.
 */

static bool create_d3d12_buffer(gpu_buffer* buf, size_t size,
                                 D3D12_HEAP_TYPE heap_type,
                                 D3D12_RESOURCE_STATES initial_state) {
    D3D12_HEAP_PROPERTIES heap_props = {0};
    heap_props.Type = heap_type;

    D3D12_RESOURCE_DESC desc = {0};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = (UINT64)size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    /* UAV flag needed for DirectML to write into the buffer */
    if (heap_type == D3D12_HEAP_TYPE_DEFAULT) {
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }

    ID3D12Resource* resource = NULL;
    HRESULT hr = g_dml.d3d_device->lpVtbl->CreateCommittedResource(
        g_dml.d3d_device,
        &heap_props, D3D12_HEAP_FLAG_NONE,
        &desc, initial_state,
        NULL,
        &IID_ID3D12Resource, (void**)&resource);

    if (FAILED(hr)) {
        fprintf(stderr, "directml: failed to allocate %zu bytes (heap type %d)\n",
                size, heap_type);
        return false;
    }

    buf->buffer = (VkBuffer)resource;  /* Cast — same pointer storage */
    buf->memory = NULL;
    buf->size = size;
    buf->mapped = NULL;

    /* Map upload/readback buffers persistently */
    if (heap_type == D3D12_HEAP_TYPE_UPLOAD ||
        heap_type == D3D12_HEAP_TYPE_READBACK) {
        D3D12_RANGE range = {0, 0};
        hr = resource->lpVtbl->Map(resource, 0, &range, &buf->mapped);
        if (FAILED(hr)) buf->mapped = NULL;
    }

    return true;
}

bool vk_alloc_device(vk_context* ctx, gpu_buffer* buf, size_t size) {
    bool ok = create_d3d12_buffer(buf, size,
                                   D3D12_HEAP_TYPE_DEFAULT,
                                   D3D12_RESOURCE_STATE_COMMON);
    if (ok) ctx->used_memory += size;
    return ok;
}

bool vk_alloc_staging(vk_context* ctx, gpu_buffer* buf, size_t size) {
    (void)ctx;
    return create_d3d12_buffer(buf, size,
                                D3D12_HEAP_TYPE_UPLOAD,
                                D3D12_RESOURCE_STATE_GENERIC_READ);
}

bool vk_upload(vk_context* ctx, gpu_buffer* dst, const void* src, size_t size) {
    /* Create temporary upload buffer, copy data, then GPU copy to device buffer */
    gpu_buffer staging;
    if (!vk_alloc_staging(ctx, &staging, size)) return false;

    memcpy(staging.mapped, src, size);

    if (!dml_begin_commands()) goto fail;

    /* Transition destination to COPY_DEST */
    D3D12_RESOURCE_BARRIER barrier = {0};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = (ID3D12Resource*)dst->buffer;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    g_dml.cmd_list->lpVtbl->ResourceBarrier(g_dml.cmd_list, 1, &barrier);

    g_dml.cmd_list->lpVtbl->CopyBufferRegion(
        g_dml.cmd_list,
        (ID3D12Resource*)dst->buffer, 0,
        (ID3D12Resource*)staging.buffer, 0,
        size);

    /* Transition back to UAV for compute */
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    g_dml.cmd_list->lpVtbl->ResourceBarrier(g_dml.cmd_list, 1, &barrier);

    if (!dml_submit_commands()) goto fail;

    vk_free_buffer(ctx, &staging);
    return true;

fail:
    vk_free_buffer(ctx, &staging);
    return false;
}

bool vk_download(vk_context* ctx, void* dst, const gpu_buffer* src, size_t size) {
    gpu_buffer readback;
    if (!create_d3d12_buffer(&readback, size,
                              D3D12_HEAP_TYPE_READBACK,
                              D3D12_RESOURCE_STATE_COPY_DEST))
        return false;

    if (!dml_begin_commands()) goto fail;

    D3D12_RESOURCE_BARRIER barrier = {0};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = (ID3D12Resource*)src->buffer;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    g_dml.cmd_list->lpVtbl->ResourceBarrier(g_dml.cmd_list, 1, &barrier);

    g_dml.cmd_list->lpVtbl->CopyBufferRegion(
        g_dml.cmd_list,
        (ID3D12Resource*)readback.buffer, 0,
        (ID3D12Resource*)src->buffer, 0,
        size);

    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    g_dml.cmd_list->lpVtbl->ResourceBarrier(g_dml.cmd_list, 1, &barrier);

    if (!dml_submit_commands()) goto fail;

    memcpy(dst, readback.mapped, size);
    vk_free_buffer(ctx, &readback);
    return true;

fail:
    vk_free_buffer(ctx, &readback);
    return false;
}

void vk_free_buffer(vk_context* ctx, gpu_buffer* buf) {
    if (buf && buf->buffer) {
        ID3D12Resource* res = (ID3D12Resource*)buf->buffer;
        if (buf->mapped) {
            res->lpVtbl->Unmap(res, 0, NULL);
        }
        res->lpVtbl->Release(res);
        if (ctx) ctx->used_memory -= buf->size;
        memset(buf, 0, sizeof(*buf));
    }
}

/* ───── Compute Dispatch Stubs ─────
 *
 * DirectML doesn't use shaders — it uses operator descriptors.
 * These are the low-level recording functions.
 */

void vk_begin_compute(vk_context* ctx) {
    (void)ctx;
    dml_begin_commands();
}

void vk_dispatch(vk_context* ctx, shader_id shader,
                 const gpu_buffer* buffers[], uint32_t n_buffers,
                 const push_constants* pc,
                 uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
    /* DirectML replaces shader dispatch with operator execution.
     * This function exists for API compatibility but should not be called
     * directly. Use the high-level ops (vk_matmul, vk_rmsnorm, etc.) */
    (void)ctx; (void)shader; (void)buffers; (void)n_buffers;
    (void)pc; (void)groups_x; (void)groups_y; (void)groups_z;
    fprintf(stderr, "directml: raw dispatch not supported, use high-level ops\n");
}

void vk_barrier(vk_context* ctx) {
    (void)ctx;
    D3D12_RESOURCE_BARRIER barrier = {0};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = NULL;  /* Global UAV barrier */
    g_dml.cmd_list->lpVtbl->ResourceBarrier(g_dml.cmd_list, 1, &barrier);
}

bool vk_submit_and_wait(vk_context* ctx) {
    (void)ctx;
    return dml_submit_commands();
}

/* ═══════════════════════════════════════════════════════════
 * DirectML Operator Helpers
 *
 * Each high-level op:
 *   1. Creates a DML_OPERATOR_DESC
 *   2. Compiles it (or fetches from cache)
 *   3. Creates a binding table
 *   4. Records execution via IDMLCommandRecorder
 *   5. Submits and waits
 *
 * TODO: Operator caching + batched execution (record many ops,
 *       submit once). Currently one-shot per op for correctness.
 * ═══════════════════════════════════════════════════════════ */

/* Helper: Create a DML tensor desc for a 2D matrix */
static DML_BUFFER_TENSOR_DESC make_tensor_desc_2d(
    const gpu_buffer* buf, uint32_t rows, uint32_t cols,
    DML_TENSOR_DATA_TYPE dtype) {

    DML_BUFFER_TENSOR_DESC desc = {0};
    desc.DataType = dtype;
    desc.DimensionCount = 4;  /* DML requires 4D: {1, 1, rows, cols} */

    static UINT sizes[4];  /* WARNING: not thread-safe, single-threaded engine */
    sizes[0] = 1;
    sizes[1] = 1;
    sizes[2] = rows;
    sizes[3] = cols;
    desc.Sizes = sizes;

    /* Row-major strides */
    static UINT strides[4];
    strides[0] = rows * cols;
    strides[1] = rows * cols;
    strides[2] = cols;
    strides[3] = 1;
    desc.Strides = strides;

    desc.TotalTensorSizeInBytes = buf->size;
    desc.GuaranteedBaseOffsetAlignment = 0;

    return desc;
}

/* Helper: Compile + initialize + execute a DML operator (single-shot) */
static bool dml_execute_op(const DML_OPERATOR_DESC* op_desc,
                           const DML_BUFFER_BINDING* input_bindings,
                           uint32_t n_inputs,
                           const DML_BUFFER_BINDING* output_bindings,
                           uint32_t n_outputs) {
    HRESULT hr;

    /* 1. Create operator */
    IDMLOperator* op = NULL;
    hr = g_dml.dml_device->lpVtbl->CreateOperator(
        g_dml.dml_device, op_desc, &IID_IDMLOperator, (void**)&op);
    if (FAILED(hr)) {
        fprintf(stderr, "directml: CreateOperator failed 0x%08lX\n", (unsigned long)hr);
        return false;
    }

    /* 2. Compile */
    IDMLCompiledOperator* compiled = NULL;
    hr = g_dml.dml_device->lpVtbl->CompileOperator(
        g_dml.dml_device, op,
        DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION,
        &IID_IDMLCompiledOperator, (void**)&compiled);
    op->lpVtbl->Release(op);
    if (FAILED(hr)) {
        fprintf(stderr, "directml: CompileOperator failed 0x%08lX\n", (unsigned long)hr);
        return false;
    }

    /* 3. Query binding requirements */
    DML_BINDING_PROPERTIES bind_props;
    compiled->lpVtbl->GetBindingProperties(compiled, &bind_props);

    /* 4. Create temporary buffer if operator needs persistent/temp memory */
    gpu_buffer temp_buf = {0};
    gpu_buffer persist_buf = {0};

    if (bind_props.TemporaryResourceSize > 0) {
        create_d3d12_buffer(&temp_buf, bind_props.TemporaryResourceSize,
                             D3D12_HEAP_TYPE_DEFAULT,
                             D3D12_RESOURCE_STATE_COMMON);
    }
    if (bind_props.PersistentResourceSize > 0) {
        create_d3d12_buffer(&persist_buf, bind_props.PersistentResourceSize,
                             D3D12_HEAP_TYPE_DEFAULT,
                             D3D12_RESOURCE_STATE_COMMON);
    }

    /* 5. Initialize operator (uploads internal weights/tables) */
    {
        IDMLBindingTable* init_table = NULL;
        DML_BINDING_TABLE_DESC table_desc = {0};
        table_desc.Dispatchable = (IDMLDispatchable*)compiled;
        table_desc.CPUDescriptorHandle.ptr = 0;  /* TODO: descriptor heap */
        table_desc.GPUDescriptorHandle.ptr = 0;
        table_desc.SizeInDescriptors = bind_props.RequiredDescriptorCount;

        /* TODO: proper descriptor heap management
         * For now, skip initialization if no persistent resource needed.
         * Many basic ops (matmul, add, etc.) don't need initialization. */
    }

    /* 6. Record execution */
    dml_begin_commands();

    DML_BINDING_TABLE_DESC table_desc = {0};
    table_desc.Dispatchable = (IDMLDispatchable*)compiled;
    /* TODO: descriptor heap — for now this is a placeholder */

    /* Record the dispatch */
    g_dml.dml_recorder->lpVtbl->RecordDispatch(
        g_dml.dml_recorder,
        (ID3D12CommandList*)g_dml.cmd_list,
        (IDMLDispatchable*)compiled,
        NULL  /* binding table — TODO */
    );

    bool ok = dml_submit_commands();

    /* Cleanup */
    if (temp_buf.buffer) vk_free_buffer(NULL, &temp_buf);
    if (persist_buf.buffer) vk_free_buffer(NULL, &persist_buf);
    compiled->lpVtbl->Release(compiled);

    return ok;
}

/* ═══════════════════════════════════════════════════════════
 * High-Level Tensor Operations
 * ═══════════════════════════════════════════════════════════ */

void vk_matmul(vk_context* ctx,
               const gpu_buffer* A, const gpu_buffer* B, gpu_buffer* C,
               uint32_t M, uint32_t N, uint32_t K) {
    (void)ctx;

    /* DML_GEMM: C = A × B (no transpose, alpha=1, beta=0) */
    DML_BUFFER_TENSOR_DESC a_desc = make_tensor_desc_2d(A, M, K, DML_TENSOR_DATA_TYPE_FLOAT32);
    DML_BUFFER_TENSOR_DESC b_desc = make_tensor_desc_2d(B, K, N, DML_TENSOR_DATA_TYPE_FLOAT32);
    DML_BUFFER_TENSOR_DESC c_desc = make_tensor_desc_2d(C, M, N, DML_TENSOR_DATA_TYPE_FLOAT32);

    DML_TENSOR_DESC a_tensor = { DML_TENSOR_TYPE_BUFFER, &a_desc };
    DML_TENSOR_DESC b_tensor = { DML_TENSOR_TYPE_BUFFER, &b_desc };
    DML_TENSOR_DESC c_tensor = { DML_TENSOR_TYPE_BUFFER, &c_desc };

    DML_GEMM_OPERATOR_DESC gemm = {0};
    gemm.ATensor = &a_tensor;
    gemm.BTensor = &b_tensor;
    gemm.CTensor = NULL;  /* No bias */
    gemm.OutputTensor = &c_tensor;
    gemm.TransA = DML_MATRIX_TRANSFORM_NONE;
    gemm.TransB = DML_MATRIX_TRANSFORM_NONE;
    gemm.Alpha = 1.0f;
    gemm.Beta = 0.0f;
    gemm.FusedActivation = NULL;

    DML_OPERATOR_DESC op_desc = { DML_OPERATOR_GEMM, &gemm };

    /* TODO: proper binding table */
    dml_execute_op(&op_desc, NULL, 0, NULL, 0);
}

void vk_matmul_q4k(vk_context* ctx,
                    const gpu_buffer* A_quant, const gpu_buffer* B, gpu_buffer* C,
                    uint32_t M, uint32_t N, uint32_t K) {
    (void)ctx;
    /*
     * Q4_K dequantization + matmul.
     *
     * DirectML doesn't natively understand GGUF Q4_K format.
     * Options:
     *   1. CPU dequant → upload → DML matmul (slow)
     *   2. Custom DML graph with INT4 support (DML 1.8+)
     *   3. HLSL custom shader via D3D12 compute (most control)
     *
     * For v0.5.0: Option 1 (correctness first, optimize later)
     * For v0.6.0: Option 3 (custom HLSL dequant kernel)
     */

    /* TODO: Dequantize Q4_K blocks to fp32, then matmul */
    /* Temporary: fall back to CPU for quantized ops */
    fprintf(stderr, "directml: Q4_K matmul not yet implemented (use CPU fallback)\n");
}

void vk_rmsnorm(vk_context* ctx,
                const gpu_buffer* x, const gpu_buffer* weight, gpu_buffer* out,
                uint32_t n_elements, float eps) {
    (void)ctx;

    /*
     * RMSNorm: out = x * rsqrt(mean(x²) + eps) * weight
     *
     * DirectML has DML_MEAN_VARIANCE_NORMALIZATION (can do RMSNorm).
     * Or build as graph: x² → mean → +eps → rsqrt → mul_x → mul_weight
     */

    /* TODO: implement via DML operator graph */
    fprintf(stderr, "directml: rmsnorm stub\n");
}

void vk_rope(vk_context* ctx,
             gpu_buffer* q, gpu_buffer* k,
             uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads,
             uint32_t position, float freq_base) {
    (void)ctx;
    /* RoPE must be custom — no native DML operator.
     * Options: HLSL compute shader or CPU */
    fprintf(stderr, "directml: rope stub\n");
}

void vk_softmax(vk_context* ctx, gpu_buffer* x, uint32_t rows, uint32_t cols) {
    (void)ctx;
    /* DML has native DML_OPERATOR_ACTIVATION_SOFTMAX */
    fprintf(stderr, "directml: softmax stub\n");
}

void vk_silu(vk_context* ctx, gpu_buffer* x, uint32_t n_elements) {
    (void)ctx;
    /* SiLU = x * sigmoid(x). DML doesn't have SiLU natively.
     * Build as: sigmoid(x) → mul(x, sigmoid_result)
     * Or use DML_OPERATOR_ACTIVATION_SIGMOID + element-wise mul */
    fprintf(stderr, "directml: silu stub\n");
}

void vk_add(vk_context* ctx,
            const gpu_buffer* a, const gpu_buffer* b, gpu_buffer* out,
            uint32_t n_elements) {
    (void)ctx;
    /* DML_OPERATOR_ELEMENT_WISE_ADD — straightforward */
    fprintf(stderr, "directml: add stub\n");
}

void vk_mul(vk_context* ctx,
            const gpu_buffer* a, const gpu_buffer* b, gpu_buffer* out,
            uint32_t n_elements) {
    (void)ctx;
    /* DML_OPERATOR_ELEMENT_WISE_MULTIPLY — straightforward */
    fprintf(stderr, "directml: mul stub\n");
}

void vk_embedding(vk_context* ctx,
                  const gpu_buffer* table, gpu_buffer* out,
                  uint32_t token_id, uint32_t dim) {
    (void)ctx;
    /* Simple buffer offset copy — barely needs GPU */
    fprintf(stderr, "directml: embedding stub\n");
}

void vk_kv_cache_store(vk_context* ctx,
                       const gpu_buffer* kv_current, gpu_buffer* kv_cache,
                       uint32_t kv_dim, uint32_t pos, uint32_t max_seq) {
    (void)ctx;
    /* Buffer copy at offset — D3D12 CopyBufferRegion */
    fprintf(stderr, "directml: kv_cache_store stub\n");
}

void vk_gqa_attention(vk_context* ctx,
                      const gpu_buffer* q,
                      const gpu_buffer* k_cache,
                      const gpu_buffer* v_cache,
                      gpu_buffer* attn_scores,
                      gpu_buffer* attn_out,
                      uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads,
                      uint32_t seq_len, uint32_t max_seq, uint32_t current_pos) {
    (void)ctx;
    /*
     * GQA Attention: the big one.
     * Q × K^T → scale → softmax → × V
     *
     * Can be built as a DML operator graph or as chained ops.
     * For maximum performance on RDNA 2, custom HLSL might be better
     * since DML attention operators don't support GQA natively.
     */
    fprintf(stderr, "directml: gqa_attention stub\n");
}

/* ───── Diagnostics ───── */

void vk_print_info(const vk_context* ctx) {
    printf("═══ DirectML Backend ═══\n");
    printf("  Device: %s\n", ctx->device_name);
    printf("  VRAM:   %zu MB\n", ctx->device_memory_size / (1024*1024));
    printf("  Used:   %zu MB\n", ctx->used_memory / (1024*1024));
    printf("  API:    Direct3D 12 + DirectML\n");
    printf("  Queue:  COMPUTE (async, game-safe)\n");
    printf("════════════════════════\n");
}

size_t vk_memory_used(const vk_context* ctx) { return ctx->used_memory; }
size_t vk_memory_total(const vk_context* ctx) { return ctx->device_memory_size; }

#endif /* DIRECTML_BACKEND */
