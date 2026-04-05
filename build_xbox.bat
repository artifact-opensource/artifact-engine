@echo off
setlocal enabledelayedexpansion

REM ═══════════════════════════════════════════════════════════
REM  Artifact Engine — CPU Build for Xbox/Windows
REM
REM  Builds v0.5.0 with all modules (CPU compute backend)
REM  Usage: build_xbox.bat [--package] [--deploy]
REM ═══════════════════════════════════════════════════════════

set VERSION=0.5.0

REM ── MSVC Build Tools 2022 ──
set "MSVC=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207"
set "WINSDK=C:\Program Files (x86)\Windows Kits\10"
set "SDKVER=10.0.26100.0"

set "PATH=%MSVC%\bin\Hostx64\x64;%WINSDK%\bin\%SDKVER%\x64;%PATH%"
set "INCLUDE=%MSVC%\include;%WINSDK%\Include\%SDKVER%\ucrt;%WINSDK%\Include\%SDKVER%\um;%WINSDK%\Include\%SDKVER%\shared"
set "LIB=%MSVC%\lib\x64;%WINSDK%\Lib\%SDKVER%\ucrt\x64;%WINSDK%\Lib\%SDKVER%\um\x64"

cd /d C:\Users\ali\artifact-engine

REM ── Parse args ──
set DO_PACKAGE=0
set DO_DEPLOY=0
for %%a in (%*) do (
    if /I "%%a"=="--package" set DO_PACKAGE=1
    if /I "%%a"=="--deploy"  set DO_DEPLOY=1
)

echo.
echo  ╔══════════════════════════════════════╗
echo  ║  Artifact Engine v%VERSION% — CPU     ║
echo  ║  Xbox Series X Build                 ║
echo  ╚══════════════════════════════════════╝
echo.

REM ── Source files (full v0.5.0) ──
set SOURCES=^
    src\main.c ^
    src\gguf.c ^
    src\tokenizer.c ^
    src\cpu_compute.c ^
    src\engine.c ^
    src\http_server.c ^
    src\model_fetch.c ^
    src\companion.c ^
    src\frame_capture.c

REM ── Compile ──
echo  [1/4] Compiling (CPU_ONLY, all modules)...
cl.exe /nologo /O2 /GL ^
    /DCPU_ONLY /DWIN32 /D_CRT_SECURE_NO_WARNINGS /DWIN32_LEAN_AND_MEAN ^
    /DVERSION=\"%VERSION%\" ^
    /Fe:artifact-engine.exe ^
    %SOURCES% ^
    /I include ^
    /link /LTCG /OPT:REF /OPT:ICF ws2_32.lib advapi32.lib user32.lib gdi32.lib winhttp.lib d3d11.lib dxgi.lib dxguid.lib ole32.lib kernel32.lib

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  BUILD FAILED
    exit /b 1
)

for %%I in (artifact-engine.exe) do (
    set /a SIZE_KB=%%~zI/1024
    echo  OK — artifact-engine.exe (!SIZE_KB! KB^)
)

REM ── Quick sanity check ──
echo  [2/4] Version check...
artifact-engine.exe --version 2>nul || echo    (no --version flag, checking binary)
echo.

if %DO_PACKAGE%==0 if %DO_DEPLOY%==0 (
    echo  Done. Use --package to create APPX, --deploy to push to Xbox.
    goto :done
)

REM ── Package APPX ──
echo  [3/4] Packaging APPX...
if exist package rmdir /s /q package
mkdir package
mkdir package\Assets
copy artifact-engine.exe package\ >nul
copy AppxManifest.xml package\ >nul
copy Assets\*.png package\Assets\ >nul 2>nul

makeappx.exe pack /d package /p ArtifactEngine.appx /o >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo  APPX packaging failed!
    makeappx.exe pack /d package /p ArtifactEngine.appx /o
    exit /b 1
)

REM ── Sign ──
if exist ArtifactVirtual.pfx (
    signtool.exe sign /fd SHA256 /a /f ArtifactVirtual.pfx /p artifact2026 ArtifactEngine.appx >nul 2>&1
    if %ERRORLEVEL%==0 (
        echo  OK — ArtifactEngine.appx signed
    ) else (
        echo  WARNING — signing failed (non-critical for dev sideloading)
    )
) else (
    echo  WARNING — No ArtifactVirtual.pfx found, APPX unsigned
)

for %%I in (ArtifactEngine.appx) do (
    set /a ASIZE_KB=%%~zI/1024
    echo  Package: ArtifactEngine.appx (!ASIZE_KB! KB^)
)

if %DO_DEPLOY%==0 (
    echo.
    echo  Package ready. Use --deploy to push to Xbox.
    goto :done
)

REM ── Deploy to Xbox ──
echo  [4/4] Deploying to Xbox (192.168.1.11)...

REM Uninstall old version first
curl -sk -u artifact:sirius -X DELETE -H "Content-Length: 0" ^
    "https://192.168.1.11:11443/api/app/packagemanager/package?package=ArtifactVirtual.ArtifactEngine_0.3.0.0_x64__40mv32fwdcjmj" >nul 2>&1
curl -sk -u artifact:sirius -X DELETE -H "Content-Length: 0" ^
    "https://192.168.1.11:11443/api/app/packagemanager/package?package=ArtifactVirtual.ArtifactEngine_0.4.0.0_x64__40mv32fwdcjmj" >nul 2>&1

timeout /t 2 >nul

REM Deploy new version
curl -sk -u artifact:sirius -X POST ^
    -F "file=@ArtifactEngine.appx;filename=ArtifactEngine.appx" ^
    "https://192.168.1.11:11443/api/app/packagemanager/package?package=ArtifactEngine.appx"

if %ERRORLEVEL%==0 (
    echo.
    echo  Deploy request sent. Check Device Portal for status.
) else (
    echo.
    echo  Deploy failed — is Xbox in Dev Mode?
)

:done
echo.
echo  ╔══════════════════════════════════════╗
echo  ║  Artifact Engine v%VERSION% — DONE    ║
echo  ╚══════════════════════════════════════╝
echo.

endlocal
