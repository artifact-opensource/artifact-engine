/*
 * Artifact Engine — Frame Capture (Xbox Game DVR / DXGI Desktop Duplication)
 *
 * Captures the frame buffer from whatever's rendering on screen.
 * This is how the AI companion SEES the game world.
 *
 * On Xbox:
 *   - Game DVR API provides frame access
 *   - Or: IDXGIOutputDuplication (Desktop Duplication API)
 *   - Frames land in GPU memory — no CPU round-trip needed
 *   - DirectML can process them in-place
 *
 * On Desktop (dev/test):
 *   - DXGI Desktop Duplication API
 *   - Works with any D3D11/12 application
 *
 * Pipeline:
 *   Frame Buffer → Capture → Downscale → Vision Encoder → Embedding
 *   All on GPU. The embedding feeds into the companion's context.
 */

#ifndef FRAME_CAPTURE_H
#define FRAME_CAPTURE_H

#include <stdint.h>
#include <stdbool.h>

/* ───── Captured Frame ───── */
typedef struct {
    uint8_t*  pixels;         /* RGBA8 pixel data (NULL if GPU-only) */
    uint32_t  width;
    uint32_t  height;
    uint32_t  stride;         /* bytes per row */
    uint64_t  timestamp_us;   /* capture timestamp (microseconds) */
    uint64_t  frame_number;
    void*     gpu_texture;    /* ID3D12Resource* if captured to GPU */
    bool      on_gpu;         /* true if data lives on GPU (no CPU copy) */
} captured_frame;

/* ───── Frame Capture Context ───── */
typedef struct frame_capture frame_capture;

/* Initialize frame capture system
 * Returns NULL on failure (no output to duplicate, permissions, etc.)
 * target_fps: how often to capture (0 = every frame, 1 = 1fps, etc.)
 * gpu_only: if true, frames stay on GPU (faster, but can't inspect on CPU)
 */
frame_capture* frame_capture_init(uint32_t target_fps, bool gpu_only);

/* Capture the next frame
 * Returns true if a new frame was captured.
 * Returns false if no new frame available (game hasn't rendered).
 * The returned frame is valid until the next call to frame_capture_next().
 */
bool frame_capture_next(frame_capture* fc, captured_frame* out);

/* Get capture statistics */
typedef struct {
    uint64_t frames_captured;
    uint64_t frames_dropped;   /* couldn't keep up */
    double   avg_capture_ms;   /* average capture time */
    double   avg_fps;          /* actual capture rate */
} frame_capture_stats;

void frame_capture_get_stats(const frame_capture* fc, frame_capture_stats* stats);

/* Shutdown and free resources */
void frame_capture_destroy(frame_capture* fc);

/* ───── Frame Processing ───── */

/* Downscale frame on GPU for vision model input
 * Most vision encoders want 224×224 or 384×384.
 * This runs a bilinear resize on the compute queue.
 */
bool frame_downscale(const captured_frame* src, captured_frame* dst,
                     uint32_t target_width, uint32_t target_height);

/* Convert RGBA8 frame to normalized float tensor (CHW format)
 * Output: [3, height, width] float32 tensor, values in [0,1]
 * Ready for vision encoder input.
 */
bool frame_to_tensor(const captured_frame* frame, void* output_buffer,
                     uint32_t channels);

#endif /* FRAME_CAPTURE_H */
