@echo off
REM ═══════════════════════════════════════════════════════════
REM  Artifact Engine — Package + Deploy (assumes already built)
REM  Usage: package_xbox.bat [--deploy]
REM
REM  Calls build_xbox.bat --package [--deploy]
REM ═══════════════════════════════════════════════════════════

set ARGS=--package
for %%a in (%*) do (
    if /I "%%a"=="--deploy" set ARGS=--package --deploy
)

call build_xbox.bat %ARGS%
