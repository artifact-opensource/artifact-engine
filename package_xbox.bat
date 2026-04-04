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

echo =============================================
echo  Artifact Engine — Xbox APPX Package Builder
echo =============================================
echo.

:: Step 1: Compile
echo [1/4] Compiling CPU-only build...
cl.exe /nologo /O2 /DCPU_ONLY /DWIN32 /D_CRT_SECURE_NO_WARNINGS ^
  /Fe:artifact-engine.exe ^
  src\gguf.c src\tokenizer.c src\cpu_compute.c src\engine.c src\http_server.c src\model_fetch.c src\main.c ^
  /I include ^
  /link ws2_32.lib advapi32.lib user32.lib gdi32.lib

if %ERRORLEVEL% NEQ 0 (
    echo BUILD FAILED
    exit /b 1
)
echo    OK — artifact-engine.exe built

:: Step 2: Prepare package layout
echo [2/4] Preparing package layout...
if exist package rmdir /s /q package
mkdir package
mkdir package\Assets
copy artifact-engine.exe package\ >nul
copy AppxManifest.xml package\ >nul
copy Assets\*.png package\Assets\ >nul
echo    OK — package layout ready

:: Step 3: Create self-signed cert for dev sideloading
echo [3/4] Creating test certificate...
if not exist ArtifactVirtual.pfx (
    powershell -Command "New-SelfSignedCertificate -Type Custom -Subject 'CN=ArtifactVirtual' -KeyUsage DigitalSignature -FriendlyName 'Artifact Virtual Dev' -CertStoreLocation 'Cert:\CurrentUser\My' -TextExtension @('2.5.29.37={text}1.3.6.1.5.5.7.3.3', '2.5.29.19={text}') | ForEach-Object { $pwd = ConvertTo-SecureString -String 'artifact2026' -Force -AsPlainText; Export-PfxCertificate -Cert $_ -FilePath ArtifactVirtual.pfx -Password $pwd }"
    echo    OK — certificate created
) else (
    echo    OK — certificate exists
)

:: Step 4: Package and sign
echo [4/4] Creating APPX package...
makeappx.exe pack /d package /p ArtifactEngine.appx /o >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo APPX packaging failed
    makeappx.exe pack /d package /p ArtifactEngine.appx /o
    exit /b 1
)
echo    OK — ArtifactEngine.appx created

signtool.exe sign /fd SHA256 /a /f ArtifactVirtual.pfx /p artifact2026 ArtifactEngine.appx >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Signing failed (non-critical for dev sideloading)
) else (
    echo    OK — package signed
)

echo.
echo =============================================
echo  Build complete!
echo =============================================
dir ArtifactEngine.appx
echo.
echo Deploy to Xbox:
echo   curl -sk -u artifact:sirius -X POST -F "file=@ArtifactEngine.appx" https://192.168.1.11:11443/api/app/packagemanager/package
