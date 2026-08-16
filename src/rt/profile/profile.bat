@echo off
setlocal EnableExtensions

REM ============================================================================
REM Denoiser Research - Nsight Compute profiling
REM
REM Script location:
REM   src\rt\profile\profile.bat
REM
REM Repository root:
REM   three levels above this script
REM
REM Usage:
REM   profile.bat FP16
REM   profile.bat FP32
REM   profile.bat INT8
REM   profile.bat ALL
REM
REM Output:
REM   data\nsight\ncu-files\
REM ============================================================================

set "REPO_ROOT=%~dp0..\..\.."
for %%I in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fI"

set "TRTEXEC=%REPO_ROOT%\src\rt\external\TensorRT-11.2.1.2\bin\trtexec.exe"
set "NCU=C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\target\windows-desktop-win7-x64\ncu.exe"
set "OUTPUT_DIR=%REPO_ROOT%\data\nsight\ncu-files"

REM ---------------------------------------------------------------------------
REM Validate argument
REM ---------------------------------------------------------------------------

if "%~1"=="" (
    echo Usage: %~nx0 ^<FP16^|FP32^|INT8^|ALL^>
    exit /b 1
)

set "MODEL=%~1"

if /I "%MODEL%"=="FP16" goto RUN_FP16
if /I "%MODEL%"=="FP32" goto RUN_FP32
if /I "%MODEL%"=="INT8" goto RUN_INT8
if /I "%MODEL%"=="ALL" goto RUN_ALL

echo Invalid model: %MODEL%
echo Usage: %~nx0 ^<FP16^|FP32^|INT8^|ALL^>
exit /b 1

REM ---------------------------------------------------------------------------
REM Run all three profiles
REM ---------------------------------------------------------------------------

:RUN_ALL
call :PROFILE FP16 "%REPO_ROOT%\src\models\fp16\rt_hdr_alb_nrm_fp16.engine"
if errorlevel 1 exit /b 1

call :PROFILE FP32 "%REPO_ROOT%\src\models\fp32\rt_hdr_alb_nrm_fp32.engine"
if errorlevel 1 exit /b 1

call :PROFILE INT8 "%REPO_ROOT%\src\models\int8\rt_hdr_alb_nrm_int8.engine"
if errorlevel 1 exit /b 1

echo.
echo ============================================================
echo  All Nsight Compute profiles completed successfully.
echo ============================================================
exit /b 0

REM ---------------------------------------------------------------------------
REM Individual profiles
REM ---------------------------------------------------------------------------

:RUN_FP16
call :PROFILE FP16 "%REPO_ROOT%\src\models\fp16\rt_hdr_alb_nrm_fp16.engine"
exit /b %errorlevel%

:RUN_FP32
call :PROFILE FP32 "%REPO_ROOT%\src\models\fp32\rt_hdr_alb_nrm_fp32.engine"
exit /b %errorlevel%

:RUN_INT8
call :PROFILE INT8 "%REPO_ROOT%\src\models\int8\rt_hdr_alb_nrm_int8.engine"
exit /b %errorlevel%

REM ---------------------------------------------------------------------------
REM Profile function
REM ---------------------------------------------------------------------------

:PROFILE
set "CURRENT_MODEL=%~1"
set "ENGINE=%~2"
set "MODEL_NAME=%~1"
set "REPORT=%OUTPUT_DIR%\%MODEL_NAME%_detailed.ncu-rep"
set "LOG=%OUTPUT_DIR%\%MODEL_NAME%_profile.log"

if not exist "%TRTEXEC%" (
    echo ERROR: TensorRT trtexec not found:
    echo        %TRTEXEC%
    exit /b 1
)

if not exist "%NCU%" (
    echo ERROR: Nsight Compute CLI not found:
    echo        %NCU%
    exit /b 1
)

if not exist "%ENGINE%" (
    echo ERROR: %CURRENT_MODEL% TensorRT engine not found:
    echo        %ENGINE%
    exit /b 1
)

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if errorlevel 1 (
    echo ERROR: Could not create output directory:
    echo        %OUTPUT_DIR%
    exit /b 1
)

echo.
echo ============================================================
echo  Nsight Compute profile: %CURRENT_MODEL%
echo ============================================================
echo Engine: %ENGINE%
echo Report: %REPORT%
echo.

REM ---------------------------------------------------------------------------
REM NCU configuration:
REM   detailed
REM   CUDA graph node profiling
REM   boost clock control
REM   all cache control
REM   stable Tensor Core boost
REM
REM trtexec configuration:
REM   warmUp=0
REM   duration=0
REM   iterations=1
REM   avgRuns=1
REM
REM TensorRT 11.2.1.2 enables CUDA graphs and disables DMA transfers by
REM default, so the deprecated --useCudaGraph and --noDataTransfers options
REM are intentionally omitted.
REM ---------------------------------------------------------------------------

"%NCU%" ^
    --set detailed ^
    --graph-profiling node ^
    --clock-control boost ^
    --cache-control all ^
    --pipeline-boost-state stable ^
    --export "%REPORT%" ^
    --force-overwrite ^
    -- ^
    "%TRTEXEC%" ^
    --loadEngine="%ENGINE%" ^
    --warmUp=0 ^
    --duration=0 ^
    --iterations=1 ^
    --avgRuns=1 ^
    > "%LOG%" 2>&1

if errorlevel 1 (
    echo.
    echo ERROR: Nsight Compute failed for %CURRENT_MODEL%.
    echo Exit code: %errorlevel%
    echo See log:
    echo %LOG%
    exit /b 1
)

if not exist "%REPORT%" (
    echo.
    echo ERROR: Nsight Compute returned success, but no report was created.
    echo Expected:
    echo %REPORT%
    exit /b 1
)

echo %CURRENT_MODEL% profiling completed successfully.
echo Report: %REPORT%
echo Log:    %LOG%

exit /b 0
