@echo off
title Artifex Assistant
cd /d "%~dp0"

:: Reduce CUDA memory fragmentation — prevents OOM when free VRAM is available
:: but split into chunks too small for the requested allocation.
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

set "PYTHONHOME=%~dp0runtime"
set "PYTHONPATH=%~dp0venv\Lib\site-packages;%~dp0"
set "PATH=%~dp0runtime;%~dp0runtime\DLLs;%PATH%"
set "TCL_LIBRARY=%~dp0runtime\tcl\tcl8.6"
set "TK_LIBRARY=%~dp0runtime\tcl\tk8.6"

"%~dp0runtime\python.exe" "%~dp0main.py"

if errorlevel 1 (
    echo.
    echo Something went wrong. Make sure you have an NVIDIA GPU with drivers installed.
    echo.
    pause
)
