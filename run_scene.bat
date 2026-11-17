@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Runs InfiniSplat v3 on one scene through run_scene.py.
rem Usage: run_scene.bat <scene> <rgb|pose> [views] [offload|cpu] [sky]
rem Run it without arguments to see the options. Needs setup.bat to have been run once.

set "ENV=infinisplat3"
set "INFERENCE=%~dp0InfiniSplat\InfiniSplat_v3\inference.py"
set "PYTHONUTF8=1"

if not exist "%INFERENCE%" echo ERROR: %INFERENCE% not found. Run setup.bat first. & goto fail_quiet

rem ---- Find conda (same places as setup.bat) ----
set "CONDA_ROOT="
for %%D in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\miniforge3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%~dp0tools\miniforge3") do if not defined CONDA_ROOT if exist "%%~D\Scripts\activate.bat" set "CONDA_ROOT=%%~D"
if not defined CONDA_ROOT echo ERROR: No conda found. Run setup.bat first. & goto fail_quiet
echo Using conda at: %CONDA_ROOT%
if not exist "%CONDA_ROOT%\envs\%ENV%\python.exe" echo ERROR: conda env %ENV% not found. Run setup.bat first. & goto fail_quiet
call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ROOT%"
if errorlevel 1 set "FAILED=Activating conda base" & goto fail
call conda activate %ENV%
if errorlevel 1 set "FAILED=Activating conda env %ENV%" & goto fail

python "%~dp0run_scene.py" %*
set "RC=%ERRORLEVEL%"

echo.
echo ==== Done %DATE% %TIME% ====
pause
exit /b %RC%

:fail
echo.
echo ERROR: %FAILED% failed (see messages above).
:fail_quiet
pause
exit /b 1
