@echo off
REM Build llama-cpp-python from source against current llama.cpp with CUDA 12.8 + Blackwell (sm_120)
REM Logs to build-llama-cpp-python.log next to this script.

setlocal

set "LOG=%~dp0build-llama-cpp-python.log"
set "PY=%~dp0venv\Scripts\python.exe"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_PATH_V12_8=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDAToolkitDir=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

REM 1. Initialize MSVC x64 environment (cl.exe, link.exe, libs)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64
if errorlevel 1 goto :err

REM 2. Put CUDA on PATH so cmake finds nvcc
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"

REM 3. CMake args: CUDA backend, Blackwell consumer arch only (no fat binary)
set "CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120"
set "CMAKE_BUILD_PARALLEL_LEVEL=8"
set "FORCE_CMAKE=1"

echo === Build started %DATE% %TIME% ===                       >  "%LOG%"
echo CUDA_PATH=%CUDA_PATH%                                     >> "%LOG%"
echo CMAKE_ARGS=%CMAKE_ARGS%                                   >> "%LOG%"
where cl                                                       >> "%LOG%" 2>&1
where nvcc                                                     >> "%LOG%" 2>&1
nvcc --version                                                 >> "%LOG%" 2>&1
"%PY%" --version                                               >> "%LOG%" 2>&1

REM 4. Uninstall old wheel, force a fresh source build of latest llama-cpp-python
"%PY%" -m pip uninstall -y llama-cpp-python                    >> "%LOG%" 2>&1
"%PY%" -m pip install --no-cache-dir --no-binary :all: --upgrade llama-cpp-python >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%

echo.                                                          >> "%LOG%"
echo === Build finished %DATE% %TIME% (exit %RC%) ===          >> "%LOG%"
"%PY%" -m pip show llama-cpp-python                            >> "%LOG%" 2>&1

exit /b %RC%

:err
echo vcvarsall.bat failed >> "%LOG%"
exit /b 1
