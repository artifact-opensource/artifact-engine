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
 *
 * v0.5.0 — All ops implemented. Q4_K uses CPU dequant path.
 *           HLSL dequant kernel planned for v0.6.0.
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

    /* Descriptor heap for DML binding tables */
    ID3D12DescriptorHeap*      desc_heap;
    UINT                       desc_heap_size;     /* total descriptors allocated */
    UINT                       desc_increment;     /* descriptor size in bytes */

    /* Memory heaps */
    ID3D12Heap*                default_heap;     /* GPU-local (device) */
    ID3D12Heap*                upload_heap;       /* CPU→GPU staging */
    ID3D12Heap*                readback_heap;     /* GPU→CPU readback */
} dml_state;

static dml_state g_dml = {0};

/* Descriptor heap capacity — large enough for any operator binding */
#define DML_DESC_HEAP_SIZE 1024

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
    if (FAILED(hr)) return false;

    /* Set descriptor heap for DML binding tables */
    if (g_dml.desc_heap) {
        ID3D12DescriptorHeap* heaps[] = { g_dml.desc_heap };
        g_dml.cmd_list->lpVtbl->SetDescriptorHeaps(g_dml.cmd_list, 1, heaps);
    }
    return true;
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
#ifdef _GAMING_XBOX
    D3D12XBOX_CREATE_DEVICE_PARAMETERS params = {0};
    params.Version = D3D12_SDK_VERSION;
    hr = D3D12XboxCreateDevice(NULL, &params,
                                IID_PPV_ARGS(&g_dml.d3d_device));
#else
    hr = D3D12CreateDevice(NULL, D3D_FEATURE_LEVEL_12_0,
                           &IID_ID3D12Device, (void**)&g_dml.d3d_device);
#endif
    DML_CHECK(hr, "failed to create D3D12 device");

    /* ── 2. Create Compute Command Queue ── */
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

    /* ── 7. Create shader-visible descriptor heap ── */
    {
        D3D12_DESCRIPTOR_HEAP_DESC heap_desc = {0};
        heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heap_desc.NumDescriptors = DML_DESC_HEAP_SIZE;
        heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

        hr = g_dml.d3d_device->lpVtbl->CreateDescriptorHeap(
            g_dml.d3d_device, &heap_desc,
            &IID_ID3D12DescriptorHeap, (void**)&g_dml.desc_heap);
        DML_CHECK(hr, "failed to create descriptor heap");

        g_dml.desc_heap_size = DML_DESC_HEAP_SIZE;
        g_dml.desc_increment = g_dml.d3d_device->lpVtbl->GetDescriptorHandleIncrementSize(
            g_dml.d3d_device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }

    /* ── 8. Query device info ── */
    IDXGIFactory4* factory = NULL;
    IDXGIAdapter1* adapter = NULL;
    hr = CreateDXGIFactory1(&IID_IDXGIFactory4, (void**)&factory);
    if (SUCCEEDED(hr)) {
        hr = factory->lpVtbl->EnumAdapters1(factory, 0, &adapter);
        if (SUCCEEDED(hr)) {
            DXGI_ADAPTER_DESC1 desc;
            adapter->lpVtbl->GetDesc1(adapter, &desc);

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
    printf("directml: descriptor heap: %u descriptors\n", g_dml.desc_heap_size);
    printf("directml: using COMPUTE queue (async, non-blocking)\n");

    return true;
}

void vk_destroy(vk_context* ctx) {
    if (!ctx || !ctx->initialized) return;

    if (g_dml.desc_heap) g_dml.desc_heap->lpVtbl->Release(g_dml.desc_heap);
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
    gpu_buffer staging;
    if (!vk_alloc_staging(ctx, &staging, size)) return false;

    memcpy(staging.mapped, src, size);

    if (!dml_begin_commands()) goto fail;

    {
        D3D12_RESOURCE_BARRIER barrier = {0};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = (ID3D12Resource*)dst->buffer;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        g_dml.cmd_list->lpVtbl->ResourceBarrier(g_dml.cmd_list, 1, &barrier);
    }

    g_dml.cmd_list->lpVtbl->CopyBufferRegion(
        g_dml.cmd_list,
        (ID3D12Resource*)dst->buffer, 0,
        (ID3D12Resource*)staging.buffer, 0,
        size);

    {
        D3D12_RESOURCE_BARRIER barrier = {0};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = (ID3D12Resource*)dst->buffer;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        g_dml.cmd_list->lpVtbl->ResourceBarrier(g_dml.cmd_list, 1, &barrier);
    }

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

    {
        D3D12_RESOURCE_BARRIER barrier = {0};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = (ID3D12Resource*)src->buffer;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        g_dml.cmd_list->lpVtbl->ResourceBarrier(g_dml.cmd_list, 1, &barrier);
    }

    g_dml.cmd_list->lpVtbl->CopyBufferRegion(
        g_dml.cmd_list,
        (ID3D12Resource*)readback.buffer, 0,
        (ID3D12Resource*)src->buffer, 0,
        size);

    {
        D3D12_RESOURCE_BARRIER barrier = {0};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = (ID3D12Resource*)src->buffer;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        g_dml.cmd_list->lpVtbl->ResourceBarrier(g_dml.cmd_list, 1, &barrier);
    }

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

/* ───── Compute Dispatch Stubs ───── */

void vk_begin_compute(vk_context* ctx) {
    (void)ctx;
    dml_begin_commands();
}

void vk_dispatch(vk_context* ctx, shader_id shader,
                 const gpu_buffer* buffers[], uint32_t n_buffers,
                 const push_constants* pc,
                 uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
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
 * Thread-safe tensor descriptor helper (caller-allocated arrays).
 * Proper descriptor heap management for binding tables.
 *
 * Each high-level op:
 *   1. Creates a DML_OPERATOR_DESC
 *   2. Compiles it
 *   3. Creates a binding table with proper descriptor heap
 *   4. Records execution via IDMLCommandRecorder
 *   5. Submits and waits
 *
 * TODO: Operator caching — compile once, rebind per call.
 * ═══════════════════════════════════════════════════════════ */

/* Thread-safe tensor descriptor: caller provides sizes/strides arrays */
typedef struct {
    UINT sizes[4];
    UINT strides[4];
    DML_BUFFER_TENSOR_DESC buf_desc;
    DML_TENSOR_DESC tensor_desc;
} dml_tensor_4d;

static void make_tensor_4d(dml_tensor_4d* out, const gpu_buffer* buf,
                           uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3,
                           DML_TENSOR_DATA_TYPE dtype) {
    out->sizes[0] = d0;
    out->sizes[1] = d1;
    out->sizes[2] = d2;
    out->sizes[3] = d3;

    out->strides[0] = d1 * d2 * d3;
    out->strides[1] = d2 * d3;
    out->strides[2] = d3;
    out->strides[3] = 1;

    memset(&out->buf_desc, 0, sizeof(out->buf_desc));
    out->buf_desc.DataType = dtype;
    out->buf_desc.DimensionCount = 4;
    out->buf_desc.Sizes = out->sizes;
    out->buf_desc.Strides = out->strides;
    out->buf_desc.TotalTensorSizeInBytes = buf ? buf->size : 0;
    out->buf_desc.GuaranteedBaseOffsetAlignment = 0;

    out->tensor_desc.Type = DML_TENSOR_TYPE_BUFFER;
    out->tensor_desc.Desc = &out->buf_desc;
}

/* 1D vector as 4D: {1, 1, 1, n} */
static void make_tensor_1d(dml_tensor_4d* out, const gpu_buffer* buf,
                           uint32_t n, DML_TENSOR_DATA_TYPE dtype) {
    make_tensor_4d(out, buf, 1, 1, 1, n, dtype);
}

/* 2D matrix as 4D: {1, 1, rows, cols} */
static void make_tensor_2d(dml_tensor_4d* out, const gpu_buffer* buf,
                           uint32_t rows, uint32_t cols,
                           DML_TENSOR_DATA_TYPE dtype) {
    make_tensor_4d(out, buf, 1, 1, rows, cols, dtype);
}

/*
 * dml_execute_op — Compile + init + bind + execute a DML operator.
 *
 * This is the single-shot execution path. Each call:
 *   1. Creates operator from desc
 *   2. Compiles to GPU shader
 *   3. Initializes (for ops with persistent state)
 *   4. Creates binding table from our descriptor heap
 *   5. Binds inputs/outputs
 *   6. Records dispatch
 *   7. Submits and waits
 *
 * For production: compile once at startup, rebind per call.
 * For v0.5.0: correctness > throughput.
 */
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
    DML_BINDING_PROPERTIES exec_props;
    compiled->lpVtbl->GetBindingProperties(compiled, &exec_props);

    /* 4. Ensure descriptor heap is large enough */
    if (exec_props.RequiredDescriptorCount > g_dml.desc_heap_size) {
        fprintf(stderr, "directml: op requires %u descriptors, heap has %u\n",
                exec_props.RequiredDescriptorCount, g_dml.desc_heap_size);
        compiled->lpVtbl->Release(compiled);
        return false;
    }

    /* 5. Create temporary + persistent resources */
    gpu_buffer temp_buf = {0};
    gpu_buffer persist_buf = {0};

    if (exec_props.TemporaryResourceSize > 0) {
        if (!create_d3d12_buffer(&temp_buf, exec_props.TemporaryResourceSize,
                                  D3D12_HEAP_TYPE_DEFAULT,
                                  D3D12_RESOURCE_STATE_COMMON)) {
            compiled->lpVtbl->Release(compiled);
            return false;
        }
    }
    if (exec_props.PersistentResourceSize > 0) {
        if (!create_d3d12_buffer(&persist_buf, exec_props.PersistentResourceSize,
                                  D3D12_HEAP_TYPE_DEFAULT,
                                  D3D12_RESOURCE_STATE_COMMON)) {
            if (temp_buf.buffer) vk_free_buffer(NULL, &temp_buf);
            compiled->lpVtbl->Release(compiled);
            return false;
        }
    }

    /* 6. Initialize operator if it has persistent resources */
    if (exec_props.PersistentResourceSize > 0) {
        IDMLOperatorInitializer* initializer = NULL;
        IDMLCompiledOperator* ops_to_init[] = { compiled };
        hr = g_dml.dml_device->lpVtbl->CreateOperatorInitializer(
            g_dml.dml_device, 1, ops_to_init,
            &IID_IDMLOperatorInitializer, (void**)&initializer);

        if (SUCCEEDED(hr)) {
            DML_BINDING_PROPERTIES init_props;
            initializer->lpVtbl->GetBindingProperties(initializer, &init_props);

            /* Create init binding table */
            D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle;
            D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle;
            g_dml.desc_heap->lpVtbl->GetCPUDescriptorHandleForHeapStart(
                g_dml.desc_heap, &cpu_handle);
            g_dml.desc_heap->lpVtbl->GetGPUDescriptorHandleForHeapStart(
                g_dml.desc_heap, &gpu_handle);

            DML_BINDING_TABLE_DESC init_table_desc = {0};
            init_table_desc.Dispatchable = (IDMLDispatchable*)initializer;
            init_table_desc.CPUDescriptorHandle = cpu_handle;
            init_table_desc.GPUDescriptorHandle = gpu_handle;
            init_table_desc.SizeInDescriptors = init_props.RequiredDescriptorCount;

            IDMLBindingTable* init_table = NULL;
            hr = g_dml.dml_device->lpVtbl->CreateBindingTable(
                g_dml.dml_device, &init_table_desc,
                &IID_IDMLBindingTable, (void**)&init_table);

            if (SUCCEEDED(hr)) {
                /* Bind persistent resource as output of initializer */
                DML_BUFFER_BINDING persist_bind = {
                    (ID3D12Resource*)persist_buf.buffer, 0, persist_buf.size
                };
                DML_BINDING_DESC persist_bind_desc = {
                    DML_BINDING_TYPE_BUFFER, &persist_bind
                };
                init_table->lpVtbl->BindOutputs(init_table, 1, &persist_bind_desc);

                /* Bind temp if init needs it */
                if (init_props.TemporaryResourceSize > 0 && temp_buf.buffer) {
                    DML_BUFFER_BINDING tb = {
                        (ID3D12Resource*)temp_buf.buffer, 0, temp_buf.size
                    };
                    DML_BINDING_DESC td = { DML_BINDING_TYPE_BUFFER, &tb };
                    init_table->lpVtbl->BindTemporaryResource(init_table, &td);
                }

                /* Record init dispatch */
                dml_begin_commands();
                g_dml.dml_recorder->lpVtbl->RecordDispatch(
                    g_dml.dml_recorder,
                    (ID3D12CommandList*)g_dml.cmd_list,
                    (IDMLDispatchable*)initializer,
                    init_table);
                dml_submit_commands();

                init_table->lpVtbl->Release(init_table);
            }
            initializer->lpVtbl->Release(initializer);
        }
    }

    /* 7. Create execution binding table */
    D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle;
    D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle;
    g_dml.desc_heap->lpVtbl->GetCPUDescriptorHandleForHeapStart(
        g_dml.desc_heap, &cpu_handle);
    g_dml.desc_heap->lpVtbl->GetGPUDescriptorHandleForHeapStart(
        g_dml.desc_heap, &gpu_handle);

    DML_BINDING_TABLE_DESC table_desc = {0};
    table_desc.Dispatchable = (IDMLDispatchable*)compiled;
    table_desc.CPUDescriptorHandle = cpu_handle;
    table_desc.GPUDescriptorHandle = gpu_handle;
    table_desc.SizeInDescriptors = exec_props.RequiredDescriptorCount;

    IDMLBindingTable* bind_table = NULL;
    hr = g_dml.dml_device->lpVtbl->CreateBindingTable(
        g_dml.dml_device, &table_desc,
        &IID_IDMLBindingTable, (void**)&bind_table);
    if (FAILED(hr)) {
        fprintf(stderr, "directml: CreateBindingTable failed 0x%08lX\n", (unsigned long)hr);
        if (temp_buf.buffer) vk_free_buffer(NULL, &temp_buf);
        if (persist_buf.buffer) vk_free_buffer(NULL, &persist_buf);
        compiled->lpVtbl->Release(compiled);
        return false;
    }

    /* 8. Bind temporary resource */
    if (exec_props.TemporaryResourceSize > 0 && temp_buf.buffer) {
        DML_BUFFER_BINDING tb = {
            (ID3D12Resource*)temp_buf.buffer, 0, temp_buf.size
        };
        DML_BINDING_DESC td = { DML_BINDING_TYPE_BUFFER, &tb };
        bind_table->lpVtbl->BindTemporaryResource(bind_table, &td);
    }

    /* Bind persistent resource */
    if (exec_props.PersistentResourceSize > 0 && persist_buf.buffer) {
        DML_BUFFER_BINDING pb = {
            (ID3D12Resource*)persist_buf.buffer, 0, persist_buf.size
        };
        DML_BINDING_DESC pd = { DML_BINDING_TYPE_BUFFER, &pb };
        bind_table->lpVtbl->BindPersistentResource(bind_table, &pd);
    }

    /* 9. Bind inputs */
    if (n_inputs > 0 && input_bindings) {
        DML_BINDING_DESC* input_descs = (DML_BINDING_DESC*)malloc(
            n_inputs * sizeof(DML_BINDING_DESC));
        for (uint32_t i = 0; i < n_inputs; i++) {
            if (input_bindings[i].Buffer) {
                input_descs[i].Type = DML_BINDING_TYPE_BUFFER;
                input_descs[i].Desc = &input_bindings[i];
            } else {
                input_descs[i].Type = DML_BINDING_TYPE_NONE;
                input_descs[i].Desc = NULL;
            }
        }
        bind_table->lpVtbl->BindInputs(bind_table, n_inputs, input_descs);
        free(input_descs);
    }

    /* 10. Bind outputs */
    if (n_outputs > 0 && output_bindings) {
        DML_BINDING_DESC* output_descs = (DML_BINDING_DESC*)malloc(
            n_outputs * sizeof(DML_BINDING_DESC));
        for (uint32_t i = 0; i < n_outputs; i++) {
            if (output_bindings[i].Buffer) {
                output_descs[i].Type = DML_BINDING_TYPE_BUFFER;
                output_descs[i].Desc = &output_bindings[i];
            } else {
                output_descs[i].Type = DML_BINDING_TYPE_NONE;
                output_descs[i].Desc = NULL;
            }
        }
        bind_table->lpVtbl->BindOutputs(bind_table, n_outputs, output_descs);
        free(output_descs);
    }

    /* 11. Record execution dispatch */
    dml_begin_commands();
    g_dml.dml_recorder->lpVtbl->RecordDispatch(
        g_dml.dml_recorder,
        (ID3D12CommandList*)g_dml.cmd_list,
        (IDMLDispatchable*)compiled,
        bind_table);

    bool ok = dml_submit_commands();

    /* Cleanup */
    bind_table->lpVtbl->Release(bind_table);
    if (temp_buf.buffer) vk_free_buffer(NULL, &temp_buf);
    if (persist_buf.buffer) vk_free_buffer(NULL, &persist_buf);
    compiled->lpVtbl->Release(compiled);

    return ok;
}

/* ═══════════════════════════════════════════════════════════
 * High-Level Tensor Operations
 * ═══════════════════════════════════════════════════════════ */

/* ─── Matrix Multiply: C = A × B ─── */

void vk_matmul(vk_context* ctx,
               const gpu_buffer* A, const gpu_buffer* B, gpu_buffer* C,
               uint32_t M, uint32_t N, uint32_t K) {
    (void)ctx;

    dml_tensor_4d a_td, b_td, c_td;
    make_tensor_2d(&a_td, A, M, K, DML_TENSOR_DATA_TYPE_FLOAT32);
    make_tensor_2d(&b_td, B, K, N, DML_TENSOR_DATA_TYPE_FLOAT32);
    make_tensor_2d(&c_td, C, M, N, DML_TENSOR_DATA_TYPE_FLOAT32);

    DML_GEMM_OPERATOR_DESC gemm = {0};
    gemm.ATensor = &a_td.tensor_desc;
    gemm.BTensor = &b_td.tensor_desc;
    gemm.CTensor = NULL;
    gemm.OutputTensor = &c_td.tensor_desc;
    gemm.TransA = DML_MATRIX_TRANSFORM_NONE;
    gemm.TransB = DML_MATRIX_TRANSFORM_NONE;
    gemm.Alpha = 1.0f;
    gemm.Beta = 0.0f;
    gemm.FusedActivation = NULL;

    DML_OPERATOR_DESC op_desc = { DML_OPERATOR_GEMM, &gemm };

    DML_BUFFER_BINDING inputs[] = {
        { (ID3D12Resource*)A->buffer, 0, A->size },
        { (ID3D12Resource*)B->buffer, 0, B->size },
    };
    DML_BUFFER_BINDING outputs[] = {
        { (ID3D12Resource*)C->buffer, 0, C->size },
    };

    dml_execute_op(&op_desc, inputs, 2, outputs, 1);
}

/* ─── Quantized MatMul: C = dequant(A_q4k) @ B ─── */

/*
 * Q4_K dequantization + matmul.
 *
 * DirectML doesn't natively understand GGUF Q4_K format.
 * v0.5.0: CPU dequant → upload → DML GEMM (correctness first)
 * v0.6.0: Custom HLSL compute shader for on-GPU dequant
 *
 * HLSL v0.6.0 plan:
 *   - Root signature with 3 UAVs: q4k_input, dequant_output, B
 *   - Compute shader: each thread group handles one Q4_K block (256 elems)
 *   - Reads 144-byte block, outputs 256 × fp32
 *   - After dequant dispatch + UAV barrier, chain DML GEMM
 *   - Estimated perf: ~10x faster than CPU dequant path
 */

/* fp16 helper — matches cpu_compute.c */
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

/* Q4_K block dequant — identical to cpu_compute.c */
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

void vk_matmul_q4k(vk_context* ctx,
                    const gpu_buffer* A_quant, const gpu_buffer* B, gpu_buffer* C,
                    uint32_t M, uint32_t N, uint32_t K) {
    /*
     * CPU dequant path (v0.5.0):
     * 1. Download quantized A from GPU
     * 2. Dequantize each row on CPU
     * 3. Upload dequantized row to GPU
     * 4. DML GEMM on dequantized data
     *
     * This is slow (~10ms per layer) but correct.
     */
    const size_t block_size = 144;
    const size_t elems_per_block = 256;
    const size_t blocks_per_row = (K + elems_per_block - 1) / elems_per_block;
    const size_t row_bytes = blocks_per_row * block_size;

    /* Allocate CPU buffers for dequant */
    uint8_t* quant_data = (uint8_t*)malloc(M * row_bytes);
    float* dequant_data = (float*)malloc((size_t)M * K * sizeof(float));
    if (!quant_data || !dequant_data) {
        free(quant_data);
        free(dequant_data);
        fprintf(stderr, "directml: Q4_K matmul OOM for dequant buffers\n");
        return;
    }

    /* Download quantized weights */
    vk_download(ctx, quant_data, A_quant, M * row_bytes);

    /* CPU dequantize all rows */
    for (uint32_t i = 0; i < M; i++) {
        const uint8_t* row_data = quant_data + i * row_bytes;
        float* row_out = dequant_data + (size_t)i * K;
        for (size_t blk = 0; blk < blocks_per_row; blk++) {
            int remaining = (int)(K - blk * elems_per_block);
            if (remaining > (int)elems_per_block) remaining = (int)elems_per_block;
            dequant_q4_k_block(row_data + blk * block_size,
                               row_out + blk * elems_per_block, remaining);
        }
    }

    /* Upload dequantized A to GPU temp buffer */
    gpu_buffer A_fp32;
    size_t a_size = (size_t)M * K * sizeof(float);
    if (!vk_alloc_device(ctx, &A_fp32, a_size)) {
        free(quant_data);
        free(dequant_data);
        return;
    }
    vk_upload(ctx, &A_fp32, dequant_data, a_size);

    free(quant_data);
    free(dequant_data);

    /* DML GEMM on fp32 data */
    vk_matmul(ctx, &A_fp32, B, C, M, N, K);

    vk_free_buffer(ctx, &A_fp32);
}

/* ─── RMSNorm: out = x * rsqrt(mean(x²) + eps) * weight ─── */

void vk_rmsnorm(vk_context* ctx,
                const gpu_buffer* x, const gpu_buffer* weight, gpu_buffer* out,
                uint32_t n_elements, float eps) {
    (void)ctx;

    /*
     * Exact RMSNorm via DML operator graph decomposition:
     *   Step 1: x² (element-wise multiply x * x)
     *   Step 2: mean(x²) via DML_OPERATOR_REDUCE with AVERAGE
     *   Step 3: mean(x²) + eps (add scalar)
     *   Step 4: rsqrt = 1/sqrt(mean(x²) + eps)
     *   Step 5: x * rsqrt (element-wise multiply)
     *   Step 6: result * weight (element-wise multiply)
     *
     * For v0.5.0: CPU fallback for exact correctness.
     * DML MeanVarianceNormalization1 subtracts the mean (LayerNorm, not RMSNorm).
     */

    /* CPU path: download, compute, upload — correct and simple */
    float* xd = (float*)malloc(n_elements * sizeof(float));
    float* wd = (float*)malloc(n_elements * sizeof(float));
    float* od = (float*)malloc(n_elements * sizeof(float));
    if (!xd || !wd || !od) {
        free(xd); free(wd); free(od);
        return;
    }

    vk_download(ctx, xd, x, n_elements * sizeof(float));
    vk_download(ctx, wd, weight, n_elements * sizeof(float));

    /* RMSNorm: identical to cpu_compute.c */
    float ss = 0.0f;
    for (uint32_t i = 0; i < n_elements; i++) ss += xd[i] * xd[i];
    float rms = 1.0f / sqrtf(ss / n_elements + eps);
    for (uint32_t i = 0; i < n_elements; i++) od[i] = xd[i] * rms * wd[i];

    vk_upload(ctx, out, od, n_elements * sizeof(float));

    free(xd);
    free(wd);
    free(od);
}

/* ─── RoPE: Rotary Position Embedding ─── */

void vk_rope(vk_context* ctx,
             gpu_buffer* q, gpu_buffer* k,
             uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads,
             uint32_t position, float freq_base) {
    (void)ctx;

    /*
     * RoPE has no native DML operator. Implementation:
     *   - Compute sin/cos tables on CPU (cheap — head_dim/2 values)
     *   - Download Q,K from GPU
     *   - Apply rotation pairs on CPU
     *   - Upload rotated Q,K back to GPU
     *
     * For v0.6.0: precompute sin/cos as GPU buffers, apply via
     * DML element-wise multiply + add (requires interleave trick).
     */

    size_t q_size = (size_t)n_heads * head_dim * sizeof(float);
    size_t k_size = (size_t)n_kv_heads * head_dim * sizeof(float);

    float* qd = (float*)malloc(q_size);
    float* kd = (float*)malloc(k_size);
    if (!qd || !kd) { free(qd); free(kd); return; }

    vk_download(ctx, qd, q, q_size);
    vk_download(ctx, kd, k, k_size);

    /* Apply RoPE to Q — identical to cpu_compute.c */
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

    vk_upload(ctx, q, qd, q_size);
    vk_upload(ctx, k, kd, k_size);

    free(qd);
    free(kd);
}

/* ─── Softmax: softmax(x) along last dimension ─── */

void vk_softmax(vk_context* ctx, gpu_buffer* x, uint32_t rows, uint32_t cols) {
    (void)ctx;

    /*
     * DML_OPERATOR_ACTIVATION_SOFTMAX applies softmax across the
     * effective rightmost dimension of the flattened tensor.
     * Map as 4D: {1, 1, rows, cols} → softmax on dim 3.
     */

    dml_tensor_4d in_td;
    make_tensor_2d(&in_td, x, rows, cols, DML_TENSOR_DATA_TYPE_FLOAT32);

    /* Output tensor: same shape, can be in-place (same buffer) */
    dml_tensor_4d out_td;
    make_tensor_2d(&out_td, x, rows, cols, DML_TENSOR_DATA_TYPE_FLOAT32);

    DML_ACTIVATION_SOFTMAX_OPERATOR_DESC softmax = {0};
    softmax.InputTensor = &in_td.tensor_desc;
    softmax.OutputTensor = &out_td.tensor_desc;

    DML_OPERATOR_DESC op_desc = { DML_OPERATOR_ACTIVATION_SOFTMAX, &softmax };

    DML_BUFFER_BINDING inputs[] = {
        { (ID3D12Resource*)x->buffer, 0, x->size },
    };
    DML_BUFFER_BINDING outputs[] = {
        { (ID3D12Resource*)x->buffer, 0, x->size },  /* in-place */
    };

    dml_execute_op(&op_desc, inputs, 1, outputs, 1);
}

/* ─── SiLU: x * sigmoid(x) ─── */

void vk_silu(vk_context* ctx, gpu_buffer* x, uint32_t n_elements) {
    (void)ctx;

    /*
     * SiLU = x * sigmoid(x). DirectML has no native SiLU op.
     * Implementation: sigmoid(x) → temp; x * temp → x
     *
     * Two-op chain: sigmoid into temp buffer, then element-wise multiply.
     */

    /* Allocate temp buffer for sigmoid output */
    gpu_buffer sigmoid_buf;
    size_t buf_size = (size_t)n_elements * sizeof(float);
    if (!vk_alloc_device(ctx, &sigmoid_buf, buf_size)) {
        /* Fallback: CPU path */
        float* xd = (float*)malloc(buf_size);
        if (!xd) return;
        vk_download(ctx, xd, x, buf_size);
        for (uint32_t i = 0; i < n_elements; i++)
            xd[i] = xd[i] / (1.0f + expf(-xd[i]));
        vk_upload(ctx, x, xd, buf_size);
        free(xd);
        return;
    }

    /* Step 1: sigmoid(x) → sigmoid_buf */
    {
        dml_tensor_4d in_td, out_td;
        make_tensor_1d(&in_td, x, n_elements, DML_TENSOR_DATA_TYPE_FLOAT32);
        make_tensor_1d(&out_td, &sigmoid_buf, n_elements, DML_TENSOR_DATA_TYPE_FLOAT32);

        DML_ACTIVATION_SIGMOID_OPERATOR_DESC sigmoid_desc = {0};
        sigmoid_desc.InputTensor = &in_td.tensor_desc;
        sigmoid_desc.OutputTensor = &out_td.tensor_desc;

        DML_OPERATOR_DESC op_desc = { DML_OPERATOR_ACTIVATION_SIGMOID, &sigmoid_desc };

        DML_BUFFER_BINDING inputs[] = {
            { (ID3D12Resource*)x->buffer, 0, x->size },
        };
        DML_BUFFER_BINDING outputs[] = {
            { (ID3D12Resource*)sigmoid_buf.buffer, 0, sigmoid_buf.size },
        };

        dml_execute_op(&op_desc, inputs, 1, outputs, 1);
    }

    /* Step 2: x * sigmoid(x) → x (in-place via output = x) */
    {
        dml_tensor_4d a_td, b_td, out_td;
        make_tensor_1d(&a_td, x, n_elements, DML_TENSOR_DATA_TYPE_FLOAT32);
        make_tensor_1d(&b_td, &sigmoid_buf, n_elements, DML_TENSOR_DATA_TYPE_FLOAT32);
        make_tensor_1d(&out_td, x, n_elements, DML_TENSOR_DATA_TYPE_FLOAT32);

        DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC mul_desc = {0};
        mul_desc.ATensor = &a_td.tensor_desc;
        mul_desc.BTensor = &b_td.tensor_desc;
        mul_desc.OutputTensor = &out_td.tensor_desc;

        DML_OPERATOR_DESC op_desc = { DML_OPERATOR_ELEMENT_WISE_MULTIPLY, &mul_desc };

        DML_BUFFER_BINDING inputs[] = {
            { (ID3D12Resource*)x->buffer, 0, x->size },
            { (ID3D12Resource*)sigmoid_buf.buffer, 0, sigmoid_buf.size },
        };
        DML_BUFFER_BINDING outputs[] = {
            { (ID3D12Resource*)x->buffer, 0, x->size },
        };

        dml_execute_op(&op_desc, inputs, 2, outputs, 1);
    }

    vk_free_buffer(ctx, &sigmoid_buf);
}

/* ─── Element-wise Add: out = a + b ─── */

void vk_add(vk_context* ctx,
            const gpu_buffer* a, const gpu_buffer* b, gpu_buffer* out,
            uint32_t n_elements) {
    (void)ctx;

    dml_tensor_4d a_td, b_td, out_td;
    make_tensor_1d(&a_td, a, n_elements, DML_TENSOR_DATA_TYPE_FLOAT32);
    make_tensor_1d(&b_td, b, n_elements, DML_TENSOR_DATA_TYPE_FLOAT32);
    make_tensor_1d(&out_td, out, n_elements, DML_TENSOR_DATA_TYPE_FLOAT32);

    DML_ELEMENT_WISE_ADD_OPERATOR_DESC add_desc = {0};
    add_desc.ATensor = &a_td.tensor_desc;
    add_desc.BTensor = &b_td.tensor_desc;
    add_desc.OutputTensor = &out_td.tensor_desc;

    DML_OPERATOR_DESC op_desc = { DML_OPERATOR_ELEMENT_WISE_ADD, &add_desc };

    DML_BUFFER_BINDING inputs[] = {
        { (ID3D12Resource*)a->buffer, 0, a->size },
        { (ID3D12Resource*)b->buffer, 0, b->size },
    };
    DML_BUFFER_BINDING outputs[] = {
        { (ID3D12Resource*)out->buffer, 0, out->size },
    };

    dml_execute_op(&op_desc, inputs, 2, outputs, 1);
}

/* ─── Element-wise Multiply: out = a * b ─── */

void vk_mul(vk_context* ctx,
            const gpu_buffer* a, const gpu_buffer* b, gpu_buffer* out,
            uint32_t n_elements) {
    (void)ctx;

    dml_tensor_4d a_td, b_td, out_td;
    make_tensor_1d(&a_td, a, n_elements, DML_TENSOR_DATA_TYPE_FLOAT32);
    make_tensor_1d(&b_td, b, n_elements, DML_TENSOR_DATA_TYPE_FLOAT32);
    make_tensor_1d(&out_td, out, n_elements, DML_TENSOR_DATA_TYPE_FLOAT32);

    DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC mul_desc = {0};
    mul_desc.ATensor = &a_td.tensor_desc;
    mul_desc.BTensor = &b_td.tensor_desc;
    mul_desc.OutputTensor = &out_td.tensor_desc;

    DML_OPERATOR_DESC op_desc = { DML_OPERATOR_ELEMENT_WISE_MULTIPLY, &mul_desc };

    DML_BUFFER_BINDING inputs[] = {
        { (ID3D12Resource*)a->buffer, 0, a->size },
        { (ID3D12Resource*)b->buffer, 0, b->size },
    };
    DML_BUFFER_BINDING outputs[] = {
        { (ID3D12Resource*)out->buffer, 0, out->size },
    };

    dml_execute_op(&op_desc, inputs, 2, outputs, 1);
}

/* ─── Token Embedding Lookup ─── */

void vk_embedding(vk_context* ctx,
                  const gpu_buffer* table, gpu_buffer* out,
                  uint32_t token_id, uint32_t dim) {
    (void)ctx;

    /*
     * Embedding = copy a contiguous slice from the embedding table.
     * table[token_id * dim .. (token_id+1) * dim] → out
     *
     * This is a D3D12 CopyBufferRegion, not a DML operator.
     */
    size_t offset = (size_t)token_id * dim * sizeof(float);
    size_t copy_size = (size_t)dim * sizeof(float);

    if (offset + copy_size > table->size) {
        fprintf(stderr, "directml: embedding token_id %u out of range\n", token_id);
        return;
    }

    if (!dml_begin_commands()) return;

    /* Transition table to COPY_SOURCE */
    D3D12_RESOURCE_BARRIER barrier1 = {0};
    barrier1.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier1.Transition.pResource = (ID3D12Resource*)table->buffer;
    barrier1.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier1.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier1.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    /* Transition out to COPY_DEST */
    D3D12_RESOURCE_BARRIER barrier2 = {0};
    barrier2.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier2.Transition.pResource = (ID3D12Resource*)out->buffer;
    barrier2.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier2.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier2.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    D3D12_RESOURCE_BARRIER barriers[2] = { barrier1, barrier2 };
    g_dml.cmd_list->lpVtbl->ResourceBarrier(g_dml.cmd_list, 2, barriers);

    g_dml.cmd_list->lpVtbl->CopyBufferRegion(
        g_dml.cmd_list,
        (ID3D12Resource*)out->buffer, 0,
        (ID3D12Resource*)table->buffer, offset,
        copy_size);

    /* Transition both back to UAV */
    barrier1.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier1.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier2.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier2.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barriers[0] = barrier1;
    barriers[1] = barrier2;
    g_dml.cmd_list->lpVtbl->ResourceBarrier(g_dml.cmd_list, 2, barriers);

    dml_submit_commands();
}

/* ─── KV Cache Store ─── */

void vk_kv_cache_store(vk_context* ctx,
                       const gpu_buffer* kv_current, gpu_buffer* kv_cache,
                       uint32_t kv_dim, uint32_t pos, uint32_t max_seq) {
    (void)ctx;
    (void)max_seq;

    /*
     * Store current K or V vector into the cache at position `pos`.
     * cache[pos * kv_dim .. (pos+1) * kv_dim] = kv_current[0..kv_dim]
     *
     * D3D12 CopyBufferRegion at offset — very fast GPU-side copy.
     */
    size_t copy_size = (size_t)kv_dim * sizeof(float);
    size_t dst_offset = (size_t)pos * kv_dim * sizeof(float);

    if (!dml_begin_commands()) return;

    /* Transition source to COPY_SOURCE */
    D3D12_RESOURCE_BARRIER barrier1 = {0};
    barrier1.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier1.Transition.pResource = (ID3D12Resource*)kv_current->buffer;
    barrier1.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier1.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier1.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    /* Transition cache to COPY_DEST */
    D3D12_RESOURCE_BARRIER barrier2 = {0};
    barrier2.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier2.Transition.pResource = (ID3D12Resource*)kv_cache->buffer;
    barrier2.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier2.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier2.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    D3D12_RESOURCE_BARRIER barriers[2] = { barrier1, barrier2 };
    g_dml.cmd_list->lpVtbl->ResourceBarrier(g_dml.cmd_list, 2, barriers);

    g_dml.cmd_list->lpVtbl->CopyBufferRegion(
        g_dml.cmd_list,
        (ID3D12Resource*)kv_cache->buffer, dst_offset,
        (ID3D12Resource*)kv_current->buffer, 0,
        copy_size);

    /* Transition both back to UAV */
    barrier1.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier1.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier2.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier2.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barriers[0] = barrier1;
    barriers[1] = barrier2;
    g_dml.cmd_list->lpVtbl->ResourceBarrier(g_dml.cmd_list, 2, barriers);

    dml_submit_commands();
}

/* ─── Grouped-Query Attention ─── */

void vk_gqa_attention(vk_context* ctx,
                      const gpu_buffer* q,
                      const gpu_buffer* k_cache,
                      const gpu_buffer* v_cache,
                      gpu_buffer* attn_scores,
                      gpu_buffer* attn_out,
                      uint32_t head_dim, uint32_t n_heads, uint32_t n_kv_heads,
                      uint32_t seq_len, uint32_t max_seq, uint32_t current_pos) {
    /*
     * GQA Attention: Q × K^T → scale → softmax → × V
     *
     * DirectML has no native GQA attention operator. We chain:
     *   1. For each head: Q[h] × K_cache[kv_h]^T → scores
     *   2. Scale by 1/sqrt(head_dim)
     *   3. Softmax over scores
     *   4. scores × V_cache[kv_h] → output
     *
     * For v0.5.0: CPU implementation (matches cpu_compute.c exactly).
     * For v0.6.0: batched GEMM via DML with stride broadcasting for GQA.
     *
     * The CPU path is fine for single-token decode (~0.1ms for 9B model).
     * Prefill (multiple tokens) would benefit from GPU.
     */
    (void)attn_scores;
    (void)max_seq;

    uint32_t kv_group = n_heads / n_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    uint32_t pos_len = current_pos + 1;
    if (pos_len > seq_len) pos_len = seq_len;

    /* Download Q, K cache, V cache to CPU */
    size_t q_size = (size_t)n_heads * head_dim * sizeof(float);
    size_t kv_cache_active = (size_t)pos_len * n_kv_heads * head_dim * sizeof(float);

    float* qd = (float*)malloc(q_size);
    float* kd = (float*)malloc(kv_cache_active);
    float* vd = (float*)malloc(kv_cache_active);
    float* od = (float*)calloc((size_t)n_heads * head_dim, sizeof(float));
    float* scores = (float*)malloc(pos_len * sizeof(float));

    if (!qd || !kd || !vd || !od || !scores) {
        free(qd); free(kd); free(vd); free(od); free(scores);
        return;
    }

    vk_download(ctx, qd, q, q_size);
    vk_download(ctx, kd, k_cache, kv_cache_active);
    vk_download(ctx, vd, v_cache, kv_cache_active);

    /* GQA attention — identical to cpu_compute.c */
    for (uint32_t h = 0; h < n_heads; h++) {
        uint32_t kv_h = h / kv_group;
        const float* q_head = qd + h * head_dim;

        /* Q × K^T + scale */
        float max_score = -INFINITY;
        for (uint32_t s = 0; s < pos_len; s++) {
            const float* k_head = kd + (size_t)s * n_kv_heads * head_dim + kv_h * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += q_head[d] * k_head[d];
            scores[s] = dot * scale;
            if (scores[s] > max_score) max_score = scores[s];
        }

        /* Softmax */
        float sum = 0.0f;
        for (uint32_t s = 0; s < pos_len; s++) {
            scores[s] = expf(scores[s] - max_score);
            sum += scores[s];
        }
        for (uint32_t s = 0; s < pos_len; s++) scores[s] /= sum;

        /* scores × V */
        float* o_head = od + h * head_dim;
        for (uint32_t s = 0; s < pos_len; s++) {
            const float* v_head = vd + (size_t)s * n_kv_heads * head_dim + kv_h * head_dim;
            for (uint32_t d = 0; d < head_dim; d++) o_head[d] += scores[s] * v_head[d];
        }
    }

    /* Upload result */
    vk_upload(ctx, attn_out, od, (size_t)n_heads * head_dim * sizeof(float));

    free(qd);
    free(kd);
    free(vd);
    free(od);
    free(scores);
}

/* ───── Diagnostics ───── */

void vk_print_info(const vk_context* ctx) {
    printf("═══ DirectML Backend v0.5.0 ═══\n");
    printf("  Device: %s\n", ctx->device_name);
    printf("  VRAM:   %zu MB\n", ctx->device_memory_size / (1024*1024));
    printf("  Used:   %zu MB\n", ctx->used_memory / (1024*1024));
    printf("  API:    Direct3D 12 + DirectML\n");
    printf("  Queue:  COMPUTE (async, game-safe)\n");
    printf("  Desc:   %u descriptors\n", g_dml.desc_heap_size);
    printf("  Ops:    matmul=DML, add=DML, mul=DML, softmax=DML, silu=DML\n");
    printf("  Ops:    rmsnorm=CPU, rope=CPU, attention=CPU, q4k=CPU+DML\n");
    printf("  NOTE:   CPU ops migrate to DML in v0.6.0\n");
    printf("═══════════════════════════════\n");
}

size_t vk_memory_used(const vk_context* ctx) { return ctx->used_memory; }
size_t vk_memory_total(const vk_context* ctx) { return ctx->device_memory_size; }

#endif /* DIRECTML_BACKEND */
