@echo off
rem Artifex setup.
rem   Double-click:        install Artifex (shows the plan, asks first)
rem   setup packs          list the optional extras
rem   setup pack voice     add one (voice, vision, create, webgpu, ...)
rem   setup remote check   is anything reachable that shouldn't be?
rem
rem Uses a Python 3.11-3.13 you already have; if there is none, fetches
rem python.org's portable Python into .\python (pinned size, SHA-512 and
rem code signature; no admin, no PATH or registry changes).
setlocal
cd /d "%~dp0"
rem The installer then shows its hints as "setup ..." commands.
set "ARTIFEX_SETUP=1"
chcp 65001 >nul

set "PY="
for %%V in (3.13 3.12 3.11) do (
  if not defined PY (
    py -%%V -c "pass" >nul 2>&1 && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY if exist "python\tools\python.exe" set "PY=python\tools\python.exe"
if not defined PY (
  echo No Python 3.11-3.13 found. Getting the portable one ^(about 15 MB^)...
  powershell -NoProfile -ExecutionPolicy Bypass -File "installer\get-python.ps1"
  if errorlevel 1 goto :failed
  set "PY=python\tools\python.exe"
)

if not "%~1"=="" (
  %PY% -m installer %*
  exit /b %errorlevel%
)

title Artifex setup
echo Using: %PY%
echo.
%PY% -m installer install
if errorlevel 1 goto :failed
echo.
echo Done. Start Artifex by double-clicking "Artifex GUI.cmd" in this folder
echo ^(or "Artifex CLI.cmd" for the terminal version^).
echo Extras any time: in this folder's window type   setup packs
echo.
pause
exit /b 0

:failed
echo.
echo Setup did not finish. Nothing outside this folder was changed.
echo Scroll up for the reason, or see the README's "Install" section.
echo.
pause
exit /b 1
