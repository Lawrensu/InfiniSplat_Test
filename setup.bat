@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem One click setup of InfiniSplat v3 for the lab PC (RTX 5070, no admin).
rem Safe to run again: it only does what is missing. Step 1 is here, steps 2 to 7
rem (git, clone, torch, requirements, weights, verify) are in setup_env.py.
rem The pinned commit, torch version and weight hashes are constants in setup_env.py.

set "ENV_NAME=infinisplat3"
set "PYVER=3.11"
rem Windows defaults to cp1252. UTF-8 keeps reading and printing paths safe.
set "PYTHONUTF8=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

echo ==========================================
echo  InfiniSplat v3 setup, conda env %ENV_NAME%
echo ==========================================

rem ---- 0. Checks that assume nothing about this PC ----
where curl.exe >nul 2>&1
if errorlevel 1 goto err_curl

rem ---- 1. Conda and the env ----
echo.
echo [1/7] Conda env %ENV_NAME% (python %PYVER%)
set "CONDA_ROOT="
for %%D in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\miniforge3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniconda3") do if not defined CONDA_ROOT if exist "%%~D\Scripts\activate.bat" set "CONDA_ROOT=%%~D"
if not defined CONDA_ROOT goto err_conda
echo Using conda at: %CONDA_ROOT%
call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ROOT%"
if errorlevel 1 goto err_activate_base
if exist "%CONDA_ROOT%\envs\%ENV_NAME%\python.exe" goto have_env
call conda create -y -n %ENV_NAME% -c conda-forge --override-channels python=%PYVER%
if errorlevel 1 goto err_create
:have_env
call conda activate %ENV_NAME%
if errorlevel 1 goto err_activate_env

rem ---- 2. Steps 2 to 7 ----
python "%~dp0setup_env.py"
if errorlevel 1 goto err_setup

echo.
echo ==========================================
echo  Setup complete. Environment: %ENV_NAME%
echo  Next:
echo    1. Run  run_scene.bat object rgb offload  to check the install.
echo    2. Read the README for the rest.
echo ==========================================
pause
exit /b 0

rem ---- Errors ----

:err_curl
echo.
echo ERROR: curl.exe not found (it ships with Windows 10 version 1803 and newer).
pause
exit /b 1

:err_conda
echo.
echo ERROR: no conda found in %%USERPROFILE%%\miniconda3, miniforge3, anaconda3 or %%LOCALAPPDATA%%\miniconda3.
echo Install Miniconda for this user only (no admin needed) from https://www.anaconda.com/docs/getting-started/miniconda/install and run setup.bat again.
pause
exit /b 1

:err_activate_base
echo.
echo ERROR: activating conda base failed.
pause
exit /b 1

:err_create
echo.
echo ERROR: creating conda env %ENV_NAME% failed (see messages above).
pause
exit /b 1

:err_activate_env
echo.
echo ERROR: activating conda env %ENV_NAME% failed.
pause
exit /b 1

:err_setup
echo.
echo ERROR: setup failed (see messages above)
pause
exit /b 1
