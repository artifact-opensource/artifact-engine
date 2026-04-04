@echo off
setlocal

:: MSVC Build Tools 2022
set "MSVC=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207"
set "WINSDK=C:\Program Files (x86)\Windows Kits\10"
set "SDKVER=10.0.26100.0"

set "PATH=%MSVC%\bin\Hostx64\x64;%WINSDK%\bin\%SDKVER%\x64;%PATH%"
set "INCLUDE=%MSVC%\include;%WINSDK%\Include\%SDKVER%\ucrt;%WINSDK%\Include\%SDKVER%\um;%WINSDK%\Include\%SDKVER%\shared"
set "LIB=%MSVC%\lib\x64;%WINSDK%\Lib\%SDKVER%\ucrt\x64;%WINSDK%\Lib\%SDKVER%\um\x64"

cd /d C:\Users\ali\artifact-engine

echo ========================================
echo  Artifact Engine — CPU Build for Xbox
echo ========================================
echo.

:: CPU-only build — no Vulkan dependency
echo [1/2] Compiling...
cl.exe /nologo /O2 /DCPU_ONLY /DWIN32 /D_CRT_SECURE_NO_WARNINGS ^
  /Fe:artifact-engine.exe ^
  src\gguf.c src\tokenizer.c src\cpu_compute.c src\engine.c src\http_server.c src\model_fetch.c src\main.c ^
  /I include ^
  /link ws2_32.lib advapi32.lib

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo BUILD FAILED
    exit /b 1
)

echo.
echo [2/2] Build complete!
echo.
dir artifact-engine.exe
echo.

:: Quick test — model info
if exist "C:\Users\ali\models\Qwen3.5-9B-Q4_K_M.gguf" (
    echo === Model Info Test ===
    artifact-engine.exe --info --model "C:\Users\ali\models\Qwen3.5-9B-Q4_K_M.gguf"
)
