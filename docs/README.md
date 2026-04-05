# Documentation

Artifact Engine — LLM inference from scratch. Vulkan compute. Pure C. Zero dependencies.

## Contents

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | System design, data flow, memory layout, inference pipeline |
| [Building](building.md) | Build instructions for Linux, Windows, and Xbox |
| [Configuration](configuration.md) | CLI flags, runtime options, backend selection |
| [GGUF Format](gguf-format.md) | Model file parsing, tensor extraction, metadata handling |
| [Tokenizer](tokenizer.md) | BPE implementation, merge pairs, special tokens, Unicode |
| [Compute Backends](compute-backends.md) | Vulkan, CPU, and DirectML backend internals |
| [HTTP API](http-api.md) | OpenAI-compatible API reference, all endpoints |
| [Model Fetching](model-fetching.md) | HTTP download, LAN auto-discovery, Xbox model paths |
| [Xbox Deployment](xbox-deployment.md) | Complete guide to running on Xbox Series X/S |
| [Troubleshooting](troubleshooting.md) | Common issues, debugging, GPU selection |
| [Changelog](changelog.md) | Version history and release notes |

## Quick Navigation

**I want to build and run it** → [Building](building.md) → [Configuration](configuration.md)

**I want to understand the internals** → [Architecture](architecture.md) → [Compute Backends](compute-backends.md)

**I want to deploy on Xbox** → [Xbox Deployment](xbox-deployment.md)

**I want to hit the API** → [HTTP API](http-api.md)

**Something isn't working** → [Troubleshooting](troubleshooting.md)
