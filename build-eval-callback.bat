@echo off
REM Build llama-eval-callback (per-tensor activation dump tool) using the
REM same toolchain as build-llama-server.bat.
setlocal
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_PATH_V12_8=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64
if errorlevel 1 exit /b 1
set "PATH=%CUDA_PATH%\bin;%PATH%"
cd /d "%~dp0vendor\llama.cpp"
cmake --build build --config Release --target llama-eval-callback -j 12
