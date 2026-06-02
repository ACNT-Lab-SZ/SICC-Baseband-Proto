param(
    [string]$OutDir = "deliverables",
    [string]$Tag = ""
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
if ([string]::IsNullOrWhiteSpace($Tag)) {
    $Tag = Get-Date -Format "yyyyMMdd_HHmmss"
}

$PackageName = "rs_ai_gpu_direct_two_scheme_handoff_$Tag"
$PackageRoot = Join-Path $Root (Join-Path $OutDir $PackageName)
$ZipPath = "$PackageRoot.zip"

if (Test-Path $PackageRoot) {
    Remove-Item -LiteralPath $PackageRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $PackageRoot | Out-Null

$Dirs = @(
    "rs_ai_link",
    "scripts",
    "docs",
    "src",
    "config",
    "models",
    "sample_data",
    "tests"
)

foreach ($Dir in $Dirs) {
    $Src = Join-Path $Root $Dir
    if (Test-Path $Src) {
        Copy-Item -Path $Src -Destination (Join-Path $PackageRoot $Dir) -Recurse
    }
}

Get-ChildItem -Path $PackageRoot -Recurse -Directory -Filter "__pycache__" | ForEach-Object {
    Remove-Item -LiteralPath $_.FullName -Recurse -Force
}

$Files = @(
    "requirements-rs-ai.txt",
    "README_PORTABLE.md",
    "PROJECT_MANIFEST.txt"
)

foreach ($File in $Files) {
    $Src = Join-Path $Root $File
    if (Test-Path $Src) {
        Copy-Item -Path $Src -Destination (Join-Path $PackageRoot $File)
    }
}

$SummarySrc = Join-Path $Root "logs\gpu_direct_two_scheme\summary.json"
if (Test-Path $SummarySrc) {
    New-Item -ItemType Directory -Path (Join-Path $PackageRoot "validation") -Force | Out-Null
    Copy-Item -Path $SummarySrc -Destination (Join-Path $PackageRoot "validation\gpu_direct_two_scheme_summary.json")
}

$Sen1ValidationSrc = Join-Path $Root "logs\sen1floods11_validation"
if (Test-Path $Sen1ValidationSrc) {
    New-Item -ItemType Directory -Path (Join-Path $PackageRoot "validation\sen1floods11") -Force | Out-Null
    Copy-Item -Path (Join-Path $Sen1ValidationSrc "*") -Destination (Join-Path $PackageRoot "validation\sen1floods11") -Recurse
}

$Sen1VideoSrc = Join-Path $Root "logs\sen1floods11_ui_videos"
if (Test-Path $Sen1VideoSrc) {
    $Sen1VideoDest = Join-Path $PackageRoot "validation\sen1floods11\ui_videos"
    New-Item -ItemType Directory -Path $Sen1VideoDest -Force | Out-Null
    Get-ChildItem -Path $Sen1VideoSrc | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $Sen1VideoDest -Recurse -Force
    }
}

$SatBurnValidationSrc = Join-Path $Root "logs\satburn_validation"
if (Test-Path $SatBurnValidationSrc) {
    New-Item -ItemType Directory -Path (Join-Path $PackageRoot "validation\satburn") -Force | Out-Null
    Copy-Item -Path (Join-Path $SatBurnValidationSrc "*") -Destination (Join-Path $PackageRoot "validation\satburn") -Recurse
}

$SatBurnVideoSrc = Join-Path $Root "logs\satburn_ui_videos"
if (Test-Path $SatBurnVideoSrc) {
    $SatBurnVideoDest = Join-Path $PackageRoot "validation\satburn\ui_videos"
    New-Item -ItemType Directory -Path $SatBurnVideoDest -Force | Out-Null
    Get-ChildItem -Path $SatBurnVideoSrc | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $SatBurnVideoDest -Recurse -Force
    }
}

$MaritimeValidationSrc = Join-Path $Root "logs\maritime_ship_validation"
if (Test-Path $MaritimeValidationSrc) {
    New-Item -ItemType Directory -Path (Join-Path $PackageRoot "validation\maritime_ship") -Force | Out-Null
    Copy-Item -Path (Join-Path $MaritimeValidationSrc "*") -Destination (Join-Path $PackageRoot "validation\maritime_ship") -Recurse
}

$MaritimeVideoSrc = Join-Path $Root "logs\maritime_ship_ui_videos"
if (Test-Path $MaritimeVideoSrc) {
    $MaritimeVideoDest = Join-Path $PackageRoot "validation\maritime_ship\ui_videos"
    New-Item -ItemType Directory -Path $MaritimeVideoDest -Force | Out-Null
    Get-ChildItem -Path $MaritimeVideoSrc | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $MaritimeVideoDest -Recurse -Force
    }
}

$Readme = @"
# RS-AI GPU Direct Two-Scheme Handoff

Main document:

docs/rs_ai_gpu_direct_handoff.md

Interfaces:

src/rs_ai_gpu_direct_api.h
src/rs_ai_gpu_direct_protocol.hpp

4090 validation:

scripts/rs_ai_gpu_direct_two_scheme_test.sh

Emergency task replacement:

docs/rs_ai_xbd_emergency_task.md
docs/rs_ai_sen1floods11_emergency_task.md
docs/rs_ai_sen1floods11_validation_report.md
docs/rs_ai_quickquake_emergency_task.md
docs/rs_ai_satburn_wildfire_emergency_task.md
docs/rs_ai_maritime_ship_security_task.md
scripts/rs_ai_convert_xbd_to_yolo.py
scripts/rs_ai_train_xbd_damage_yolo.sh
scripts/rs_ai_xbd_emergency_task_switch_demo.sh
scripts/rs_ai_download_sen1floods11.sh
scripts/rs_ai_start_sen1floods11_download.sh
scripts/rs_ai_extract_sen1floods11.py
scripts/rs_ai_convert_sen1floods11_to_yolo.py
scripts/rs_ai_sen1floods11_smoke_test.py
scripts/rs_ai_train_sen1floods11_yolo.sh
scripts/rs_ai_start_sen1floods11_training.sh
scripts/rs_ai_sen1floods11_emergency_task_switch_demo.sh
scripts/rs_ai_convert_quickquakebuildings_to_yolo.py
scripts/rs_ai_quickquake_smoke_test.py
scripts/rs_ai_train_quickquakebuildings_yolo.sh
scripts/rs_ai_quickquake_emergency_task_switch_demo.sh
scripts/rs_ai_quickquake_full_validation.sh
scripts/rs_ai_start_quickquake_full_validation.sh
scripts/rs_ai_download_satburn.py
scripts/rs_ai_download_satburn.sh
scripts/rs_ai_start_satburn_download.sh
scripts/rs_ai_convert_satburn_to_yolo.py
scripts/rs_ai_train_satburn_yolo.sh
scripts/rs_ai_start_satburn_training.sh
scripts/rs_ai_satburn_emergency_task_switch_demo.sh
scripts/rs_ai_satburn_full_validation.sh
scripts/rs_ai_start_satburn_full_validation.sh
scripts/rs_ai_convert_maritime_ship_to_yolo.py
scripts/rs_ai_train_maritime_ship_yolo.sh
scripts/rs_ai_maritime_ship_security_task_switch_demo.sh
scripts/rs_ai_maritime_ship_full_validation.sh
scripts/rs_ai_start_maritime_ship_full_validation.sh

Sample VISO sequence:

sample_data/viso_002/002.avi
sample_data/viso_002/gt2.txt

File-based fallback:

scripts/rs_ai_viso_sequence_demo.sh
scripts/rs_ai_make_baseband_payload.sh
scripts/rs_ai_make_baseband_payload_roi_layered.sh
"@

$Readme | Set-Content -Path (Join-Path $PackageRoot "HANDOFF_README.md") -Encoding UTF8

if (Test-Path $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}
Compress-Archive -Path (Join-Path $PackageRoot "*") -DestinationPath $ZipPath

Write-Host "[RS-AI] package_root=$PackageRoot"
Write-Host "[RS-AI] package_zip=$ZipPath"
