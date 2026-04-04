/*
 * Artifact Engine — HTTP Model Downloader
 *
 * Downloads GGUF model files from HTTP URLs.
 * Shows progress bar. Resumes partial downloads.
 * No external dependencies — raw sockets.
 */

#ifndef MODEL_FETCH_H
#define MODEL_FETCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Download a model from HTTP URL to local path.
 * Shows progress during download.
 * Returns true on success. */
bool model_fetch(const char* url, const char* output_path);

/* Get filename from URL (last path component) */
const char* model_fetch_filename(const char* url);

#endif /* MODEL_FETCH_H */
