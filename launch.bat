@echo off
title Artifex Assistant
cd /d "%~dp0"

:: Starts the terminal version with the Python that setup.cmd installed
:: (venv\Scripts\python.exe). Same as "Artifex CLI.cmd".
if not exist "%~dp0venv\Scripts\python.exe" goto :novenv

:: Reduce CUDA memory fragmentation: prevents OOM when free VRAM is available
:: but split into chunks too small for the requested allocation.
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

"%~dp0venv\Scripts\python.exe" "%~dp0main.py"

if errorlevel 1 (
    echo.
    echo Artifex stopped with an error. Scroll up for the reason;
    echo logs\artifex.log has the details.
    echo.
    pause
    exit /b 1
)
exit /b 0

:novenv
echo.
echo Artifex is not installed in this folder yet:
echo   "%~dp0venv\Scripts\python.exe" was not found.
echo.
echo Double-click setup.cmd in this folder first, then start Artifex again.
echo.
pause
exit /b 1
