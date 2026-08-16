param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("FP16", "FP32", "INT8", "All")]
    [string]$Model
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$TensorRTBin = Join-Path $RepoRoot "src\rt\external\TensorRT-11.2.1.2\bin"
$TrtExec    = Join-Path $TensorRTBin "trtexec.exe"
$Ncu        = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\target\windows-desktop-win7-x64\ncu.exe"

$OutputDir = Join-Path $RepoRoot "data\nsight\ncu-files"

$Engines = @{
    FP16 = Join-Path $RepoRoot "src\models\fp16\rt_hdr_alb_nrm_fp16.engine"
    FP32 = Join-Path $RepoRoot "src\models\fp32\rt_hdr_alb_nrm_fp32.engine"
    INT8 = Join-Path $RepoRoot "src\models\int8\rt_hdr_alb_nrm_int8.engine"
}

function Assert-FileExists {
    param(
        [string]$Path,
        [string]$Description
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description not found: $Path"
    }
}

Assert-FileExists $TrtExec "TensorRT trtexec"
Assert-FileExists $Ncu "Nsight Compute CLI"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$ModelsToRun = if ($Model -eq "All") {
    @("FP16", "FP32", "INT8")
} else {
    @($Model)
}

foreach ($CurrentModel in $ModelsToRun) {

    $Engine = $Engines[$CurrentModel]
    Assert-FileExists $Engine "$CurrentModel TensorRT engine"

    $Report = Join-Path $OutputDir "${CurrentModel.ToLowerInvariant()}_detailed.ncu-rep"
    $Log    = Join-Path $OutputDir "${CurrentModel.ToLowerInvariant()}_profile.log"

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Nsight Compute profile: $CurrentModel"
    Write-Host "============================================================"
    Write-Host "Engine: $Engine"
    Write-Host "Report: $Report"
    Write-Host ""

    # Record the command/configuration in a human-readable log.
    $StartTime = Get-Date
    @(
        "Denoiser Research - Nsight Compute profile"
        "Model: $CurrentModel"
        "Start time: $StartTime"
        "Engine: $Engine"
        "TensorRT: 11.2.1.2"
        "Nsight Compute: 2026.2.1"
        "NCU set: detailed"
        "NCU graph profiling: node"
        "NCU clock control: boost"
        "NCU cache control: all"
        "NCU pipeline boost state: stable"
        "trtexec warmUp: 0"
        "trtexec duration: 0"
        "trtexec iterations: 1"
        "trtexec avgRuns: 1"
        ""
    ) | Set-Content -LiteralPath $Log -Encoding UTF8

    $NcuArgs = @(
        "--set", "detailed",
        "--graph-profiling", "node",
        "--clock-control", "boost",
        "--cache-control", "all",
        "--pipeline-boost-state", "stable",
        "--export", $Report,
        "--force-overwrite",
        "--",
        $TrtExec,
        "--loadEngine=$Engine",
        "--warmUp=0",
        "--duration=0",
        "--iterations=1",
        "--avgRuns=1"
    )

    # Tee the profiler's console output into the per-model log while also
    # displaying it in the terminal.
    & $Ncu @NcuArgs 2>&1 | Tee-Object -FilePath $Log -Append

    if ($LASTEXITCODE -ne 0) {
        throw "Nsight Compute failed for $CurrentModel with exit code $LASTEXITCODE."
    }

    if (-not (Test-Path -LiteralPath $Report -PathType Leaf)) {
        throw "Nsight Compute reported success, but the expected report was not created: $Report"
    }

    $EndTime = Get-Date
    Add-Content -LiteralPath $Log -Value ""
    Add-Content -LiteralPath $Log -Value "End time: $EndTime"
    Add-Content -LiteralPath $Log -Value "Result: SUCCESS"
    Add-Content -LiteralPath $Log -Value "Report: $Report"

    Write-Host ""
    Write-Host "$CurrentModel profiling completed successfully."
    Write-Host "Report: $Report"
}

Write-Host ""
Write-Host "All requested Nsight Compute profiles completed successfully."
