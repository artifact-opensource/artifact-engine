@echo off
REM ═══════════════════════════════════════════════════════════
REM  Artifact Engine — DirectML Build Script (MSVC)
REM
REM  Builds the engine with DirectML backend for Xbox Series X.
REM  Requires: Visual Studio 2022 Build Tools, Windows SDK 10.0.26100+
REM
REM  Usage: build_directml.bat [Release|Debug] [--package]
REM
REM  Produces:
REM    build\ArtifactEngine.exe   — 64-bit DirectML binary
REM    build\ArtifactEngine.appx  — Xbox UWP package (with --package)
REM ═══════════════════════════════════════════════════════════

setlocal enabledelayedexpansion

REM ── Configuration ──
set VERSION=0.5.0
set BUILD_DIR=build
set OUTPUT=ArtifactEngine.exe

REM ── Parse args ──
set CONFIG=Release
set DO_PACKAGE=0
for %%a in (%*) do (
    if /I "%%a"=="Debug" set CONFIG=Debug
    if /I "%%a"=="Release" set CONFIG=Release
    if /I "%%a"=="--package" set DO_PACKAGE=1
)

echo.
echo  ╔══════════════════════════════════════╗
echo  ║  Artifact Engine v%VERSION% — DirectML  ║
echo  ║  MSVC Build for Xbox / Windows       ║
echo  ╚══════════════════════════════════════╝
echo.
echo  Config: %CONFIG%

REM ── Find MSVC ──
if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
) else (
    echo ERROR: Visual Studio 2022 Build Tools not found!
    echo Install: winget install Microsoft.VisualStudio.2022.BuildTools
    exit /b 1
)

REM ── Verify tools ──
where cl >nul 2>&1
if errorlevel 1 (
    echo ERROR: cl.exe not found in PATH after vcvars64.bat
    exit /b 1
)

echo  Compiler: & cl 2>&1 | findstr /C:"Microsoft" | head -1

REM ── Create build dir ──
if not exist %BUILD_DIR% mkdir %BUILD_DIR%

REM ── Source files ──
set SOURCES=^
    src\main.c ^
    src\directml_compute.c ^
    src\frame_capture.c ^
    src\companion.c ^
    src\engine.c ^
    src\gguf.c ^
    src\http_server.c ^
    src\tokenizer.c ^
    src\model_fetch.c

REM ── Compiler flags ──
set COMMON_FLAGS=/nologo /W3 /utf-8 /DDIRECTML_BACKEND /D_CRT_SECURE_NO_WARNINGS /DWIN32_LEAN_AND_MEAN /DVERSION=\"%VERSION%\"
set INCLUDE_FLAGS=/Iinclude

if "%CONFIG%"=="Release" (
    set CFLAGS=%COMMON_FLAGS% /O2 /GL /DNDEBUG
    set LFLAGS=/link /LTCG /OPT:REF /OPT:ICF
) else (
    set CFLAGS=%COMMON_FLAGS% /Od /Zi /DDEBUG /D_DEBUG
    set LFLAGS=/link /DEBUG
)

REM ── Libraries ──
REM DirectML: from Windows SDK or bundled NuGet package
REM D3D12/DXGI: Windows SDK
REM D3D11: For DXGI Desktop Duplication (frame capture)
REM ws2_32: HTTP server sockets
REM winhttp: Model fetcher
set LIBS=d3d12.lib dxgi.lib directml.lib d3d11.lib ws2_32.lib winhttp.lib ole32.lib user32.lib kernel32.lib

echo.
echo  [1/3] Compiling...

cl %CFLAGS% %INCLUDE_FLAGS% %SOURCES% /Fe:%BUILD_DIR%\%OUTPUT% %LFLAGS% %LIBS%

if errorlevel 1 (
    echo.
    echo  ╔═══════════════════════════════╗
    echo  ║  BUILD FAILED                 ║
    echo  ╚═══════════════════════════════╝
    exit /b 1
)

echo.
echo  [2/3] Build successful!

REM ── File size ──
for %%I in (%BUILD_DIR%\%OUTPUT%) do (
    set SIZE=%%~zI
    set /a SIZE_KB=!SIZE!/1024
    echo  Binary: %BUILD_DIR%\%OUTPUT% ^(!SIZE_KB! KB^)
)

REM ── Package as APPX if requested ──
if %DO_PACKAGE%==1 (
    echo.
    echo  [3/3] Packaging APPX...

    REM Create staging directory
    set APPX_DIR=%BUILD_DIR%\appx_staging
    if exist !APPX_DIR! rmdir /s /q !APPX_DIR!
    mkdir !APPX_DIR!

    REM Copy binary
    copy /Y %BUILD_DIR%\%OUTPUT% !APPX_DIR!\

    REM Copy DirectML.dll if present locally
    if exist DirectML.dll copy /Y DirectML.dll !APPX_DIR!\

    REM Copy manifest
    copy /Y AppxManifest.xml !APPX_DIR!\

    REM Copy assets
    if exist Assets (
        mkdir !APPX_DIR!\Assets
        copy /Y Assets\* !APPX_DIR!\Assets\
    )

    REM Create APPX package
    where makeappx >nul 2>&1
    if errorlevel 1 (
        echo  WARNING: makeappx.exe not found. Looking in Windows Kits...
        set MAKEAPPX=
        for /f "delims=" %%p in ('dir /s /b "C:\Program Files (x86)\Windows Kits\10\bin\*\x64\makeappx.exe" 2^>nul') do set MAKEAPPX=%%p
        if not defined MAKEAPPX (
            echo  ERROR: makeappx.exe not found anywhere
            goto skip_package
        )
    ) else (
        set MAKEAPPX=makeappx
    )

    !MAKEAPPX! pack /d !APPX_DIR! /p %BUILD_DIR%\ArtifactEngine.appx /o
    if errorlevel 1 (
        echo  APPX packaging failed
        goto skip_package
    )

    REM Sign with self-signed cert
    where signtool >nul 2>&1
    if errorlevel 1 (
        set SIGNTOOL=
        for /f "delims=" %%p in ('dir /s /b "C:\Program Files (x86)\Windows Kits\10\bin\*\x64\signtool.exe" 2^>nul') do set SIGNTOOL=%%p
    ) else (
        set SIGNTOOL=signtool
    )

    if defined SIGNTOOL (
        REM Check for existing cert
        if exist ArtifactVirtual.pfx (
            !SIGNTOOL! sign /fd SHA256 /a /f ArtifactVirtual.pfx %BUILD_DIR%\ArtifactEngine.appx
            echo  APPX signed with ArtifactVirtual.pfx
        ) else (
            echo  WARNING: No ArtifactVirtual.pfx found. APPX unsigned.
            echo  Create with: New-SelfSignedCertificate -Subject "CN=ArtifactVirtual" -Type CodeSigningCert
        )
    )

    for %%I in (%BUILD_DIR%\ArtifactEngine.appx) do (
        set ASIZE=%%~zI
        set /a ASIZE_KB=!ASIZE!/1024
        echo  Package: %BUILD_DIR%\ArtifactEngine.appx ^(!ASIZE_KB! KB^)
    )

    :skip_package
    REM Cleanup staging
    if exist !APPX_DIR! rmdir /s /q !APPX_DIR!
) else (
    echo  [3/3] Skipping APPX packaging (use --package to enable)
)

echo.
echo  ╔══════════════════════════════════════╗
echo  ║  BUILD COMPLETE                      ║
echo  ║  %BUILD_DIR%\%OUTPUT%           ║
echo  ╚══════════════════════════════════════╝
echo.
echo  Run: %BUILD_DIR%\%OUTPUT% --model path\to\model.gguf
echo  Xbox: %BUILD_DIR%\%OUTPUT% --companion --profile player.json
echo  Test: %BUILD_DIR%\%OUTPUT% --capture --vision-fps 5
echo.

endlocal
