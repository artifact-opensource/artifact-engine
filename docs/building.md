# Building

Build instructions for all supported platforms.

## Prerequisites

| Dependency | Required | Notes |
|-----------|----------|-------|
| C compiler | Yes | GCC 11+, Clang 14+, or MSVC 2022 |
| CMake | Yes | 3.16 or later |
| Vulkan SDK | No | Required for Vulkan backend. CPU-only builds skip this. |
| glslangValidator | No | For compiling GLSL shaders to SPIR-V. Included in Vulkan SDK. |

## Linux

### Vulkan Build (GPU acceleration)

```bash
# Install Vulkan SDK (Ubuntu/Debian)
wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan.list \
  https://packages.lunarg.com/vulkan/lunarg-vulkan-noble.list
sudo apt update
sudo apt install vulkan-sdk

# Build
git clone https://github.com/artifact-opensource/artifact-engine.git
cd artifact-engine
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### CPU-Only Build (no GPU required)

```bash
git clone https://github.com/artifact-opensource/artifact-engine.git
cd artifact-engine
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCPU_ONLY=ON
make -j$(nproc)
```

The CPU-only binary has zero external dependencies. It statically links everything and produces a binary under 250KB.

### Debug Build

```bash
mkdir build-debug && cd build-debug
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

Debug builds include:
- Full symbol information for GDB/LLDB
- Address sanitizer integration (if available)
- Verbose logging to stderr
- Vulkan validation layer activation

### Cross-Compilation (aarch64)

```bash
mkdir build-arm64 && cd build-arm64
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
  -DCPU_ONLY=ON
make -j$(nproc)
```

## Windows

### Visual Studio 2022

```cmd
git clone https://github.com/artifact-opensource/artifact-engine.git
cd artifact-engine
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

The binary is output to `build\Release\artifact-engine.exe`.

### MSVC Command Line

```cmd
:: Open "x64 Native Tools Command Prompt for VS 2022"
git clone https://github.com/artifact-opensource/artifact-engine.git
cd artifact-engine
mkdir build && cd build
cmake .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
nmake
```

### CPU-Only on Windows

```cmd
cmake .. -G "Visual Studio 17 2022" -A x64 -DCPU_ONLY=ON
cmake --build . --config Release
```

## Xbox Series X|S

Xbox builds use the DirectML backend and produce a UWP .appx package.

### Prerequisites

- Windows 10/11 host
- Visual Studio 2022 with UWP workload
- Xbox GDK (Game Development Kit)
- Xbox in Developer Mode

### Build and Package

```cmd
:: Build
build_xbox.bat

:: Package as .appx
package_xbox.bat
```

### Deploy

```cmd
:: Deploy via Xbox Device Portal
curl -sk -u USERNAME:PASSWORD -X POST ^
  -F "file=@ArtifactEngine.appx" ^
  https://XBOX_IP:11443/api/app/packagemanager/package
```

See [Xbox Deployment](xbox-deployment.md) for the complete deployment guide.

## Compiling Shaders

If you modify the GLSL compute shaders in `shaders/`, recompile them to SPIR-V:

```bash
cd shaders

# Compile all shaders
for f in *.comp; do
  glslangValidator -V "$f" -o "${f%.comp}.spv"
  echo "Compiled: $f → ${f%.comp}.spv"
done
```

Or compile individually:

```bash
glslangValidator -V shaders/matmul.comp -o shaders/matmul.spv
```

The SPIR-V binaries (`.spv` files) are loaded at runtime by the Vulkan backend. They must be present in the `shaders/` directory relative to the executable, or in the working directory.

## CMake Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `CPU_ONLY` | BOOL | `OFF` | Disable Vulkan. Build with CPU backend only. |
| `CMAKE_BUILD_TYPE` | STRING | `Release` | `Debug`, `Release`, `RelWithDebInfo`, `MinSizeRel` |
| `CMAKE_INSTALL_PREFIX` | PATH | `/usr/local` | Installation prefix |

## Build Outputs

| File | Description |
|------|-------------|
| `artifact-engine` (Linux) / `artifact-engine.exe` (Windows) | Main executable |
| `shaders/*.spv` | Compiled SPIR-V shaders (Vulkan backend) |
| `ArtifactEngine.appx` | Xbox UWP package (Xbox build only) |

## Verifying the Build

```bash
# Check binary
./build/artifact-engine --help

# Verify Vulkan is available (optional)
vulkaninfo --summary 2>/dev/null && echo "Vulkan: OK" || echo "Vulkan: Not available"

# Verify shaders compile
cd shaders && for f in *.comp; do
  glslangValidator -V "$f" -o /dev/null && echo "OK: $f" || echo "FAIL: $f"
done
```

## Troubleshooting Build Issues

**"vulkan/vulkan.h: No such file or directory"**
- Vulkan SDK not installed. Either install it or build with `-DCPU_ONLY=ON`.

**"glslangValidator: command not found"**
- Install the Vulkan SDK, or install `glslang-tools` (`apt install glslang-tools`).

**MSVC link errors on Windows**
- Ensure you're using the x64 configuration. 32-bit builds are not supported.

**CMake version too old**
- Artifact Engine requires CMake 3.16+. Update via `pip install cmake` or download from cmake.org.
