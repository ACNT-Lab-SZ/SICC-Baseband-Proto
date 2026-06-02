param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [int]$Codewords = 100000,
    [int]$SnrStart = 3,
    [int]$SnrStop = 10,
    [string]$Decoder = 'osd-only'
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'setup_env.ps1') -ProjectRoot $ProjectRoot

$exe = Join-Path $ProjectRoot 'bin\gpu_bp_osd_awgn_bench.exe'
$outDir = Join-Path $ProjectRoot 'logs\decoder_only_awgn'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$csv = Join-Path $outDir ("{0}_{1}_{2}dB.csv" -f $Decoder, $SnrStart, $SnrStop)

& $exe `
    --decoder $Decoder `
    --snr-start $SnrStart `
    --snr-stop $SnrStop `
    --snr-step 1 `
    --codewords $Codewords `
    --warmup 4096 `
    --batch 4096 `
    --seed 20260513 `
    --csv $csv

Write-Host "[CSV] $csv"
