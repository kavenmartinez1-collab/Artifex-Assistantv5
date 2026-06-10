@echo off
REM Clone (if needed) and build llama-server with CUDA 12.8 + Blackwell sm_120.
REM Logs to build-llama-server.log next to this script.
REM Binary ends up at: vendor\llama.cpp\build\bin\Release\llama-server.exe

setlocal

set "ROOT=%~dp0"
set "LOG=%ROOT%build-llama-server.log"
set "SRC=%ROOT%vendor\llama.cpp"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_PATH_V12_8=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

REM 1. MSVC x64 env
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64
if errorlevel 1 goto :err_vcvars

REM 2. CUDA + cmake on PATH
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"

echo === Build started %DATE% %TIME% ===                       >  "%LOG%"
echo CUDA_PATH=%CUDA_PATH%                                     >> "%LOG%"
where cl                                                        >> "%LOG%" 2>&1
where nvcc                                                      >> "%LOG%" 2>&1
where cmake                                                     >> "%LOG%" 2>&1
where git                                                       >> "%LOG%" 2>&1
nvcc --version                                                  >> "%LOG%" 2>&1
cmake --version                                                 >> "%LOG%" 2>&1

REM 3. Clone if not present (shallow, master branch)
if not exist "%SRC%" (
    if not exist "%ROOT%vendor" mkdir "%ROOT%vendor"
    echo --- cloning llama.cpp into %SRC% ---                  >> "%LOG%"
    git clone --depth=1 https://github.com/ggerganov/llama.cpp.git "%SRC%" >> "%LOG%" 2>&1
    if errorlevel 1 goto :err_clone
) else (
    echo --- existing clone found at %SRC% ---                 >> "%LOG%"
)

REM 4. Configure (CUDA on, Blackwell consumer arch, Release)
cd /d "%SRC%"
echo --- cmake configure ---                                   >> "%LOG%"
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release >> "%LOG%" 2>&1
if errorlevel 1 goto :err_configure

REM 5. Build only the llama-server target (much faster than building everything)
echo --- cmake build (target: llama-server) ---                >> "%LOG%"
cmake --build build --config Release --target llama-server -j 8 >> "%LOG%" 2>&1
if errorlevel 1 goto :err_build

REM 6. Verify
if exist "%SRC%\build\bin\Release\llama-server.exe" (
    echo === BUILD OK %DATE% %TIME% ===                        >> "%LOG%"
    echo Binary: %SRC%\build\bin\Release\llama-server.exe       >> "%LOG%"
    exit /b 0
)
echo === BINARY NOT FOUND %DATE% %TIME% ===                    >> "%LOG%"
exit /b 1

:err_vcvars
echo vcvarsall.bat failed >> "%LOG%"
exit /b 1

:err_clone
echo git clone failed >> "%LOG%"
exit /b 2

:err_configure
echo cmake configure failed >> "%LOG%"
exit /b 3

:err_build
echo cmake build failed >> "%LOG%"
exit /b 4
