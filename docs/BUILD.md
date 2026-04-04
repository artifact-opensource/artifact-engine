# Artifact Engine — Build Instructions

> Version 0.1.0 · CONFIDENTIAL

## Prerequisites

All platforms require:
- **Vulkan SDK** (1.2 or later) — provides `vulkan.h`, `libvulkan`, and `glslc`
- **C11 compiler** (GCC, Clang, or MSVC)
- **CMake** 3.16+ (optional — manual build also documented)

---

## Linux Native (GCC + Vulkan SDK)

### Install Dependencies

```bash
# Debian/Ubuntu/Kali
sudo apt install vulkan-sdk libvulkan-dev vulkan-tools glslang-tools

# Fedora
sudo dnf install vulkan-headers vulkan-loader-devel vulkan-validation-layers glslang

# Arch
sudo pacman -S vulkan-headers vulkan-icd-loader vulkan-tools glslang
```

Verify:
```bash
vulkaninfo --summary   # Should show your GPU
glslc --version        # Should show shaderc/glslc version
```

### Build with CMake

```bash
cd /path/to/artifact-engine
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Binary: build/artifact-engine (68 KB)
# Shaders: build/shaders/*.spv
```

### Build Manually (no CMake)

```bash
cd /path/to/artifact-engine

# 1. Compile shaders to SPIR-V
mkdir -p build/shaders
for shader in matmul rmsnorm rope softmax silu add mul embedding dequant_q4k; do
    glslc -O shaders/${shader}.comp -o build/shaders/${shader}.spv
done

# 2. Compile C sources
gcc -O2 -std=c11 -Wall -Wextra -I include \
    src/main.c src/gguf.c src/vulkan_compute.c src/engine.c src/http_server.c \
    -o build/artifact-engine \
    -lvulkan -lm -lpthread

# Binary: build/artifact-engine
```

### Verify Build

```bash
# Check it runs (will fail without a model but proves the binary works)
./build/artifact-engine --help

# If you have a model:
./build/artifact-engine --model path/to/model.gguf --info
```

---

## Windows Cross-Compile (MinGW from Linux)

Cross-compile a Windows executable from Linux using MinGW. This produces a standalone `.exe` that only needs `vulkan-1.dll` at runtime.

### Install MinGW

```bash
sudo apt install gcc-mingw-w64-x86-64
```

### Compile Shaders (same as Linux)

```bash
mkdir -p build/shaders
for shader in matmul rmsnorm rope softmax silu add mul embedding dequant_q4k; do
    glslc -O shaders/${shader}.comp -o build/shaders/${shader}.spv
done
```

### Cross-Compile

```bash
x86_64-w64-mingw32-gcc -O2 -std=c11 -D_WIN32 -I include \
    -c src/gguf.c -o build/gguf_win.o
x86_64-w64-mingw32-gcc -O2 -std=c11 -D_WIN32 -I include \
    -c src/vulkan_compute.c -o build/vulkan_compute_win.o
x86_64-w64-mingw32-gcc -O2 -std=c11 -D_WIN32 -I include \
    -c src/engine.c -o build/engine_win.o
x86_64-w64-mingw32-gcc -O2 -std=c11 -D_WIN32 -I include \
    -c src/http_server.c -o build/http_server_win.o
x86_64-w64-mingw32-gcc -O2 -std=c11 -D_WIN32 -I include \
    -c src/main.c -o build/main_win.o

x86_64-w64-mingw32-gcc \
    build/main_win.o build/gguf_win.o build/vulkan_compute_win.o \
    build/engine_win.o build/http_server_win.o \
    -o build/artifact-engine.exe \
    -lvulkan-1 -lws2_32 -lm
```

**Note:** Linking requires `vulkan-1.lib` or `libvulkan-1.a`. If not available in the MinGW sysroot, the `.o` files compile cleanly and can be linked on a Windows machine. The current build produces all 5 `.o` files (verified) and a 427 KB `.exe` via direct linking.

### Runtime Dependencies (Windows)

- `vulkan-1.dll` — from the Vulkan Runtime (bundled with GPU drivers)
- Shader `.spv` files in a `shaders/` directory relative to the executable (or specified via `--shaders`)

---

## Windows Native (MSVC)

### Prerequisites

1. Install [Vulkan SDK](https://vulkan.lunarg.com/) — adds `VULKAN_SDK` environment variable
2. Visual Studio 2019+ or Build Tools with C/C++ workload

### Build via Developer Command Prompt

```cmd
cd artifact-engine

:: Compile shaders
mkdir build\shaders
for %%s in (matmul rmsnorm rope softmax silu add mul embedding dequant_q4k) do (
    "%VULKAN_SDK%\Bin\glslc.exe" -O shaders\%%s.comp -o build\shaders\%%s.spv
)

:: Compile C sources
cl /O2 /std:c11 /I include /I "%VULKAN_SDK%\Include" ^
    src\main.c src\gguf.c src\vulkan_compute.c src\engine.c src\http_server.c ^
    /Fe:build\artifact-engine.exe ^
    /link /LIBPATH:"%VULKAN_SDK%\Lib" vulkan-1.lib ws2_32.lib
```

### Build via CMake (MSVC)

```cmd
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

---

## Shader Compilation Details

### Using glslc (recommended)

`glslc` is part of the Vulkan SDK (via Google's shaderc). It produces optimized SPIR-V:

```bash
glslc -O shaders/matmul.comp -o build/shaders/matmul.spv
```

Flags:
- `-O` — optimize SPIR-V output (performance-oriented)
- `-O0` — disable optimizations (for debugging)
- `--target-env=vulkan1.2` — explicitly target Vulkan 1.2 (default)
- `-g` — include debug info (shader names in validation layers)

### Using glslangValidator (alternative)

```bash
glslangValidator -V shaders/matmul.comp -o build/shaders/matmul.spv
```

Flags:
- `-V` — compile for Vulkan (GLSL → SPIR-V)
- `-Os` — optimize for size
- `-g` — include debug info

### Compile All Shaders (script)

```bash
#!/bin/bash
SHADER_DIR=shaders
OUT_DIR=build/shaders
mkdir -p "$OUT_DIR"

SHADERS=(matmul rmsnorm rope softmax silu add mul embedding dequant_q4k)

for s in "${SHADERS[@]}"; do
    echo "Compiling $s.comp → $s.spv"
    glslc -O "$SHADER_DIR/$s.comp" -o "$OUT_DIR/$s.spv"
done

echo "Done: $(ls -1 $OUT_DIR/*.spv | wc -l) shaders compiled"
```

---

## Build Artifacts

After a successful build:

```
build/
├── artifact-engine           68 KB  (Linux ELF x86_64)
├── artifact-engine.exe      427 KB  (Windows PE x86_64, MinGW)
├── shaders/
│   ├── add.spv
│   ├── dequant_q4k.spv
│   ├── embedding.spv
│   ├── matmul.spv
│   ├── mul.spv
│   ├── rmsnorm.spv
│   ├── rope.spv
│   ├── silu.spv
│   └── softmax.spv
└── *.o                       Object files (intermediate)
```

### Deployment Package

For deployment, you need:
1. The binary (`artifact-engine` or `artifact-engine.exe`)
2. The `shaders/` directory with all `.spv` files
3. A GGUF model file

```bash
# Create deployment package
mkdir -p artifact-engine-v0.1.0/shaders
cp build/artifact-engine artifact-engine-v0.1.0/
cp build/shaders/*.spv artifact-engine-v0.1.0/shaders/
tar czf artifact-engine-v0.1.0-linux-x64.tar.gz artifact-engine-v0.1.0/
```

---

## CMakeLists.txt Notes

The current CMakeLists.txt has a known issue: `src/vulkan_compute.c`, `src/engine.c`, and `src/http_server.c` are commented out from the `SOURCES` list (marked as TODO). This was from early development — the manual build commands above compile all source files correctly. To use CMake for a full build, uncomment those lines:

```cmake
set(SOURCES
    src/main.c
    src/gguf.c
    src/vulkan_compute.c
    src/engine.c
    src/http_server.c
)
```

---

## Troubleshooting

### "glslc not found"

Install the Vulkan SDK. On Linux: `sudo apt install vulkan-sdk` or download from https://vulkan.lunarg.com/.

### "vulkan/vulkan.h not found"

The project vendors Vulkan headers in `include/vulkan/`. Ensure your include path has `-I include`.

### "libvulkan.so: cannot open shared object"

Vulkan runtime not installed. Install your GPU driver (NVIDIA/AMD/Intel) which includes `libvulkan.so`.

### "vk: no Vulkan-capable GPU found"

No GPU driver with Vulkan support detected. On headless servers, you may need `mesa-vulkan-drivers` (software rendering — very slow, for testing only).

### Windows: "vulkan-1.dll not found"

Install GPU drivers (NVIDIA GeForce Experience / AMD Adrenalin / Intel Arc). The Vulkan runtime DLL is bundled with the driver.

### MinGW: "cannot find -lvulkan-1"

The MinGW sysroot may not have the Vulkan import library. Options:
1. Compile all `.o` files on Linux, link on Windows with MSVC
2. Create a `.def` file from `vulkan-1.dll` and generate an import library: `x86_64-w64-mingw32-dlltool -d vulkan-1.def -l libvulkan-1.a`
3. Use the full MSVC build path instead
