/*
 * Artifact Engine — Frame Capture via DXGI Desktop Duplication
 *
 * Captures screen frames for the AI companion's vision system.
 * Uses IDXGIOutputDuplication (Desktop Duplication API) which
 * works on both Desktop Windows and Xbox UWP Dev Mode.
 *
 * Pipeline:
 *   Screen → DXGI Duplicate → GPU Texture → Downscale → Tensor
 *   All stays on GPU when gpu_only=true. DirectML processes in-place.
 *
 * Compile: Always included. Only functional on Windows.
 * Requires: d3d12.lib, dxgi.lib, d3d11.lib
 */

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include "../include/frame_capture.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <d3d12.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3d12.lib")

/* ───── Internal State ───── */

struct frame_capture {
    /* DXGI Desktop Duplication (D3D11 device required) */
    ID3D11Device*            d3d11_device;
    ID3D11DeviceContext*     d3d11_ctx;
    IDXGIOutputDuplication*  duplication;

    /* Staging texture for CPU readback */
    ID3D11Texture2D*         staging_tex;
    uint32_t                 desktop_width;
    uint32_t                 desktop_height;

    /* Downscale staging (CPU-side) */
    uint8_t*                 downscale_buf;

    /* Configuration */
    uint32_t                 target_fps;
    bool                     gpu_only;
    uint64_t                 frame_interval_us;  /* microseconds between captures */

    /* Stats */
    frame_capture_stats      stats;
    uint64_t                 last_capture_us;
    LARGE_INTEGER            perf_freq;
};

/* ───── Timing helpers ───── */

static uint64_t get_time_us(const frame_capture* fc) {
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (uint64_t)(now.QuadPart * 1000000 / fc->perf_freq.QuadPart);
}

/* ═══════════════════════════════════════════════════════════
 * Init / Destroy
 * ═══════════════════════════════════════════════════════════ */

frame_capture* frame_capture_init(uint32_t target_fps, bool gpu_only) {
    HRESULT hr;

    frame_capture* fc = (frame_capture*)calloc(1, sizeof(frame_capture));
    if (!fc) return NULL;

    QueryPerformanceFrequency(&fc->perf_freq);
    fc->target_fps = target_fps;
    fc->gpu_only = gpu_only;
    fc->frame_interval_us = target_fps > 0 ? (1000000 / target_fps) : 0;

    /* ── 1. Create D3D11 device (required for Desktop Duplication) ── */
    D3D_FEATURE_LEVEL feature_levels[] = { D3D_FEATURE_LEVEL_11_0 };
    D3D_FEATURE_LEVEL feature_level;

    hr = D3D11CreateDevice(
        NULL,
        D3D_DRIVER_TYPE_HARDWARE,
        NULL,
        0,  /* no debug */
        feature_levels, 1,
        D3D11_SDK_VERSION,
        &fc->d3d11_device,
        &feature_level,
        &fc->d3d11_ctx);

    if (FAILED(hr)) {
        fprintf(stderr, "frame_capture: D3D11CreateDevice failed 0x%08lX\n", (unsigned long)hr);
        free(fc);
        return NULL;
    }

    /* ── 2. Get DXGI adapter and output ── */
    IDXGIDevice* dxgi_device = NULL;
    hr = fc->d3d11_device->lpVtbl->QueryInterface(
        fc->d3d11_device, &IID_IDXGIDevice, (void**)&dxgi_device);
    if (FAILED(hr)) goto fail;

    IDXGIAdapter* adapter = NULL;
    hr = dxgi_device->lpVtbl->GetAdapter(dxgi_device, &adapter);
    dxgi_device->lpVtbl->Release(dxgi_device);
    if (FAILED(hr)) goto fail;

    IDXGIOutput* output = NULL;
    hr = adapter->lpVtbl->EnumOutputs(adapter, 0, &output);
    adapter->lpVtbl->Release(adapter);
    if (FAILED(hr)) {
        fprintf(stderr, "frame_capture: no display output found\n");
        goto fail;
    }

    /* ── 3. Get output1 for DuplicateOutput ── */
    IDXGIOutput1* output1 = NULL;
    hr = output->lpVtbl->QueryInterface(output, &IID_IDXGIOutput1, (void**)&output1);

    /* Grab desktop dimensions before releasing output */
    DXGI_OUTPUT_DESC out_desc;
    output->lpVtbl->GetDesc(output, &out_desc);
    fc->desktop_width = out_desc.DesktopCoordinates.right - out_desc.DesktopCoordinates.left;
    fc->desktop_height = out_desc.DesktopCoordinates.bottom - out_desc.DesktopCoordinates.top;

    output->lpVtbl->Release(output);

    if (FAILED(hr)) {
        fprintf(stderr, "frame_capture: IDXGIOutput1 not available\n");
        goto fail;
    }

    /* ── 4. Create desktop duplication ── */
    hr = output1->lpVtbl->DuplicateOutput(
        output1, (IUnknown*)fc->d3d11_device, &fc->duplication);
    output1->lpVtbl->Release(output1);

    if (FAILED(hr)) {
        fprintf(stderr, "frame_capture: DuplicateOutput failed 0x%08lX\n", (unsigned long)hr);
        fprintf(stderr, "  (may need to run as admin, or no desktop to duplicate on Xbox)\n");
        goto fail;
    }

    /* ── 5. Create CPU staging texture ── */
    if (!gpu_only) {
        D3D11_TEXTURE2D_DESC tex_desc = {0};
        tex_desc.Width = fc->desktop_width;
        tex_desc.Height = fc->desktop_height;
        tex_desc.MipLevels = 1;
        tex_desc.ArraySize = 1;
        tex_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        tex_desc.SampleDesc.Count = 1;
        tex_desc.Usage = D3D11_USAGE_STAGING;
        tex_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

        hr = fc->d3d11_device->lpVtbl->CreateTexture2D(
            fc->d3d11_device, &tex_desc, NULL, &fc->staging_tex);
        if (FAILED(hr)) {
            fprintf(stderr, "frame_capture: CreateTexture2D staging failed\n");
            goto fail;
        }
    }

    printf("frame_capture: initialized %ux%u @ %u fps (gpu_only=%d)\n",
           fc->desktop_width, fc->desktop_height, target_fps, gpu_only);

    return fc;

fail:
    if (fc->duplication) fc->duplication->lpVtbl->Release(fc->duplication);
    if (fc->staging_tex) fc->staging_tex->lpVtbl->Release(fc->staging_tex);
    if (fc->d3d11_ctx) fc->d3d11_ctx->lpVtbl->Release(fc->d3d11_ctx);
    if (fc->d3d11_device) fc->d3d11_device->lpVtbl->Release(fc->d3d11_device);
    free(fc);
    return NULL;
}

void frame_capture_destroy(frame_capture* fc) {
    if (!fc) return;

    if (fc->duplication) {
        fc->duplication->lpVtbl->ReleaseFrame(fc->duplication);
        fc->duplication->lpVtbl->Release(fc->duplication);
    }
    if (fc->staging_tex) fc->staging_tex->lpVtbl->Release(fc->staging_tex);
    if (fc->d3d11_ctx) fc->d3d11_ctx->lpVtbl->Release(fc->d3d11_ctx);
    if (fc->d3d11_device) fc->d3d11_device->lpVtbl->Release(fc->d3d11_device);
    if (fc->downscale_buf) free(fc->downscale_buf);

    printf("frame_capture: destroyed (captured %llu frames, dropped %llu)\n",
           (unsigned long long)fc->stats.frames_captured,
           (unsigned long long)fc->stats.frames_dropped);

    free(fc);
}

/* ═══════════════════════════════════════════════════════════
 * Capture
 * ═══════════════════════════════════════════════════════════ */

bool frame_capture_next(frame_capture* fc, captured_frame* out) {
    if (!fc || !fc->duplication || !out) return false;

    memset(out, 0, sizeof(*out));

    /* Rate limiting */
    uint64_t now_us = get_time_us(fc);
    if (fc->frame_interval_us > 0 &&
        (now_us - fc->last_capture_us) < fc->frame_interval_us) {
        return false;  /* too soon */
    }

    uint64_t start_us = now_us;

    /* Acquire next frame (non-blocking: 0ms timeout) */
    DXGI_OUTDUPL_FRAME_INFO frame_info;
    IDXGIResource* desktop_resource = NULL;

    HRESULT hr = fc->duplication->lpVtbl->AcquireNextFrame(
        fc->duplication, 0, &frame_info, &desktop_resource);

    if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
        return false;  /* no new frame */
    }

    if (FAILED(hr)) {
        if (hr == DXGI_ERROR_ACCESS_LOST) {
            fprintf(stderr, "frame_capture: access lost (display mode changed?)\n");
            /* TODO: re-init duplication */
        } else {
            fprintf(stderr, "frame_capture: AcquireNextFrame 0x%08lX\n", (unsigned long)hr);
        }
        fc->stats.frames_dropped++;
        return false;
    }

    /* Get the desktop texture */
    ID3D11Texture2D* desktop_tex = NULL;
    hr = desktop_resource->lpVtbl->QueryInterface(
        desktop_resource, &IID_ID3D11Texture2D, (void**)&desktop_tex);
    desktop_resource->lpVtbl->Release(desktop_resource);

    if (FAILED(hr)) {
        fc->duplication->lpVtbl->ReleaseFrame(fc->duplication);
        fc->stats.frames_dropped++;
        return false;
    }

    out->width = fc->desktop_width;
    out->height = fc->desktop_height;
    out->frame_number = fc->stats.frames_captured;
    out->timestamp_us = now_us;

    if (fc->gpu_only) {
        /* GPU-only: keep texture reference, don't copy to CPU */
        out->gpu_texture = (void*)desktop_tex;  /* caller must NOT release */
        out->on_gpu = true;
        out->pixels = NULL;
        out->stride = 0;
    } else {
        /* Copy desktop texture to staging (CPU-readable) */
        fc->d3d11_ctx->lpVtbl->CopyResource(
            fc->d3d11_ctx,
            (ID3D11Resource*)fc->staging_tex,
            (ID3D11Resource*)desktop_tex);

        desktop_tex->lpVtbl->Release(desktop_tex);

        /* Map staging texture for CPU read */
        D3D11_MAPPED_SUBRESOURCE mapped;
        hr = fc->d3d11_ctx->lpVtbl->Map(
            fc->d3d11_ctx,
            (ID3D11Resource*)fc->staging_tex,
            0, D3D11_MAP_READ, 0, &mapped);

        if (FAILED(hr)) {
            fc->duplication->lpVtbl->ReleaseFrame(fc->duplication);
            fc->stats.frames_dropped++;
            return false;
        }

        out->pixels = (uint8_t*)mapped.pData;
        out->stride = mapped.RowPitch;
        out->on_gpu = false;
        out->gpu_texture = NULL;

        /* Note: pixels pointer is valid until next call to frame_capture_next()
         * (we unmap + release frame at the start of next capture) */
    }

    /* Release the frame (we've copied what we need) */
    if (!fc->gpu_only) {
        /* Unmap will happen implicitly on next Map, or we leave it mapped */
        /* Actually unmap now for safety */
        fc->d3d11_ctx->lpVtbl->Unmap(
            fc->d3d11_ctx, (ID3D11Resource*)fc->staging_tex, 0);

        /* For persistent access, copy the pixel data */
        size_t data_size = (size_t)out->height * out->stride;
        uint8_t* copy = (uint8_t*)malloc(data_size);
        if (copy) {
            /* Need to re-map to copy */
            D3D11_MAPPED_SUBRESOURCE mapped2;
            hr = fc->d3d11_ctx->lpVtbl->Map(
                fc->d3d11_ctx,
                (ID3D11Resource*)fc->staging_tex,
                0, D3D11_MAP_READ, 0, &mapped2);
            if (SUCCEEDED(hr)) {
                memcpy(copy, mapped2.pData, data_size);
                fc->d3d11_ctx->lpVtbl->Unmap(
                    fc->d3d11_ctx, (ID3D11Resource*)fc->staging_tex, 0);
                out->pixels = copy;
                out->stride = mapped2.RowPitch;
            } else {
                free(copy);
                out->pixels = NULL;
            }
        }
    }

    fc->duplication->lpVtbl->ReleaseFrame(fc->duplication);

    /* Update stats */
    uint64_t end_us = get_time_us(fc);
    double capture_ms = (double)(end_us - start_us) / 1000.0;

    fc->stats.frames_captured++;
    fc->stats.avg_capture_ms =
        (fc->stats.avg_capture_ms * (fc->stats.frames_captured - 1) + capture_ms)
        / fc->stats.frames_captured;

    if (fc->last_capture_us > 0) {
        double elapsed_s = (double)(now_us - fc->last_capture_us) / 1000000.0;
        if (elapsed_s > 0) {
            double instant_fps = 1.0 / elapsed_s;
            fc->stats.avg_fps = fc->stats.avg_fps * 0.9 + instant_fps * 0.1;
        }
    }

    fc->last_capture_us = now_us;
    return true;
}

void frame_capture_get_stats(const frame_capture* fc, frame_capture_stats* stats) {
    if (fc && stats) *stats = fc->stats;
}

/* ═══════════════════════════════════════════════════════════
 * Frame Processing: Downscale + Tensor Conversion
 * ═══════════════════════════════════════════════════════════ */

/*
 * Bilinear downscale on CPU.
 *
 * For v0.6.0: use DML_OPERATOR_RESAMPLE2 on GPU.
 * CPU is fine for 1-5 FPS vision pipeline.
 */
bool frame_downscale(const captured_frame* src, captured_frame* dst,
                     uint32_t target_width, uint32_t target_height) {
    if (!src || !dst || !src->pixels) return false;
    if (src->on_gpu) {
        fprintf(stderr, "frame_downscale: GPU-only frames not yet supported (need DML resize)\n");
        return false;
    }

    uint32_t sw = src->width;
    uint32_t sh = src->height;
    uint32_t tw = target_width;
    uint32_t th = target_height;

    size_t out_size = (size_t)tw * th * 4;  /* RGBA8 */
    uint8_t* out_pixels = (uint8_t*)malloc(out_size);
    if (!out_pixels) return false;

    float x_ratio = (float)sw / tw;
    float y_ratio = (float)sh / th;

    for (uint32_t ty = 0; ty < th; ty++) {
        float src_y = ty * y_ratio;
        uint32_t sy0 = (uint32_t)src_y;
        uint32_t sy1 = sy0 + 1 < sh ? sy0 + 1 : sy0;
        float fy = src_y - sy0;

        for (uint32_t tx = 0; tx < tw; tx++) {
            float src_x = tx * x_ratio;
            uint32_t sx0 = (uint32_t)src_x;
            uint32_t sx1 = sx0 + 1 < sw ? sx0 + 1 : sx0;
            float fx = src_x - sx0;

            /* Bilinear interpolation for each BGRA channel */
            /* Source format is BGRA (DXGI_FORMAT_B8G8R8A8_UNORM) */
            for (int c = 0; c < 4; c++) {
                float p00 = src->pixels[sy0 * src->stride + sx0 * 4 + c];
                float p10 = src->pixels[sy0 * src->stride + sx1 * 4 + c];
                float p01 = src->pixels[sy1 * src->stride + sx0 * 4 + c];
                float p11 = src->pixels[sy1 * src->stride + sx1 * 4 + c];

                float val = p00 * (1 - fx) * (1 - fy)
                          + p10 * fx * (1 - fy)
                          + p01 * (1 - fx) * fy
                          + p11 * fx * fy;

                out_pixels[(ty * tw + tx) * 4 + c] = (uint8_t)(val + 0.5f);
            }
        }
    }

    dst->pixels = out_pixels;
    dst->width = tw;
    dst->height = th;
    dst->stride = tw * 4;
    dst->timestamp_us = src->timestamp_us;
    dst->frame_number = src->frame_number;
    dst->on_gpu = false;
    dst->gpu_texture = NULL;

    return true;
}

/*
 * Convert BGRA8 frame to normalized float tensor in CHW format.
 *
 * Input:  [H, W, 4] uint8 BGRA (0-255)
 * Output: [C, H, W] float32 (0.0-1.0), where C = channels (3 for RGB)
 *
 * Reorders BGRA→RGB and normalizes in one pass.
 * For vision models like ViT, MobileNet, etc.
 */
bool frame_to_tensor(const captured_frame* frame, void* output_buffer,
                     uint32_t channels) {
    if (!frame || !frame->pixels || !output_buffer) return false;
    if (frame->on_gpu) {
        fprintf(stderr, "frame_to_tensor: GPU-only frames not yet supported\n");
        return false;
    }
    if (channels != 3 && channels != 4) {
        fprintf(stderr, "frame_to_tensor: channels must be 3 (RGB) or 4 (RGBA)\n");
        return false;
    }

    uint32_t w = frame->width;
    uint32_t h = frame->height;
    float* out = (float*)output_buffer;

    /* CHW layout: [channel][row][col] */
    /* Source is BGRA, we want RGB(A) */
    for (uint32_t y = 0; y < h; y++) {
        const uint8_t* row = frame->pixels + y * frame->stride;
        for (uint32_t x = 0; x < w; x++) {
            uint8_t b = row[x * 4 + 0];
            uint8_t g = row[x * 4 + 1];
            uint8_t r = row[x * 4 + 2];
            uint8_t a = row[x * 4 + 3];

            /* Channel 0 = R, Channel 1 = G, Channel 2 = B */
            out[0 * h * w + y * w + x] = r / 255.0f;
            out[1 * h * w + y * w + x] = g / 255.0f;
            out[2 * h * w + y * w + x] = b / 255.0f;
            if (channels == 4) {
                out[3 * h * w + y * w + x] = a / 255.0f;
            }
        }
    }

    return true;
}

#else /* !_WIN32 */

/* Stub implementation for non-Windows platforms */
#include "../include/frame_capture.h"
#include <stdio.h>

frame_capture* frame_capture_init(uint32_t target_fps, bool gpu_only) {
    (void)target_fps; (void)gpu_only;
    fprintf(stderr, "frame_capture: not available on this platform (Windows/Xbox only)\n");
    return NULL;
}

bool frame_capture_next(frame_capture* fc, captured_frame* out) {
    (void)fc; (void)out;
    return false;
}

void frame_capture_get_stats(const frame_capture* fc, frame_capture_stats* stats) {
    (void)fc; (void)stats;
}

void frame_capture_destroy(frame_capture* fc) {
    (void)fc;
}

bool frame_downscale(const captured_frame* src, captured_frame* dst,
                     uint32_t target_width, uint32_t target_height) {
    (void)src; (void)dst; (void)target_width; (void)target_height;
    return false;
}

bool frame_to_tensor(const captured_frame* frame, void* output_buffer,
                     uint32_t channels) {
    (void)frame; (void)output_buffer; (void)channels;
    return false;
}

#endif /* _WIN32 */
