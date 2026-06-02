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

$Readme = @"
# RS-AI GPU Direct Two-Scheme Handoff

Main document:

docs/rs_ai_gpu_direct_handoff.md

Interfaces:

src/rs_ai_gpu_direct_api.h
src/rs_ai_gpu_direct_protocol.hpp

4090 validation:

scripts/rs_ai_gpu_direct_two_scheme_test.sh

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
