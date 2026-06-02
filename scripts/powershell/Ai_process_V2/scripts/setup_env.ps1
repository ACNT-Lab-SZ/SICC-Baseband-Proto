param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [string]$BasebandRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
    [string]$UhdRoot = $(if ($env:UHD_ROOT) { $env:UHD_ROOT } else { $(if ($env:UHD_ROOT) { $env:UHD_ROOT } elseif ($env:UHD_PKG_PATH) { $env:UHD_PKG_PATH } else { "" }) }),
    [string]$Python = $(if ($env:RS_AI_PYTHON) { $env:RS_AI_PYTHON } else { 'python' })
)

$ErrorActionPreference = 'Stop'

$bin = Join-Path $ProjectRoot 'bin'
$rootGpuExe = Join-Path $BasebandRoot 'build\uhd_cpp_gpu_pipeline\Release\uhd_ldpc_ofdm_link.exe'
$rootReleaseExe = Join-Path $BasebandRoot 'build\uhd_cpp\Release\uhd_ldpc_ofdm_link.exe'
$portableExe = Join-Path $BasebandRoot 'portable\usrp_video_link_portable_20260513\bin\uhd_ldpc_ofdm_link.exe'
$localExe = Join-Path $bin 'uhd_ldpc_ofdm_link.exe'

$alistLocal = Join-Path $ProjectRoot 'matrices\LDPC\CCSDS_ldpc_n128_k64.alist'
$alistRoot = Join-Path $BasebandRoot 'Code_Matrices_Lib\LDPC\CCSDS_ldpc_n128_k64.alist'

$candidateExe = @($localExe, $rootGpuExe, $rootReleaseExe, $portableExe) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
$candidateAlist = @($alistLocal, $alistRoot) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1

$env:PATH = "$bin;$UhdRoot\bin;$env:PATH"
$env:UHD_ROOT = $UhdRoot
$env:UHD_PKG_PATH = $UhdRoot
$env:UHD_IMAGES_DIR = Join-Path $UhdRoot 'share\uhd\images'
$env:UHD_RFNOC_DIR = Join-Path $UhdRoot 'share\uhd\rfnoc'
$env:RS_AI_PROJECT_ROOT = $ProjectRoot
$env:RS_AI_BASEBAND_ROOT = $BasebandRoot
$env:RS_AI_PYTHON = $Python
if ($candidateExe) { $env:RS_AI_BASEBAND_EXE = $candidateExe }
if ($candidateAlist) { $env:RS_AI_ALIST_N128 = $candidateAlist }

Write-Host "[ENV] ProjectRoot=$ProjectRoot"
Write-Host "[ENV] BasebandRoot=$BasebandRoot"
Write-Host "[ENV] Python=$Python"
Write-Host "[ENV] BasebandExe=$env:RS_AI_BASEBAND_EXE"
Write-Host "[ENV] AlistN128=$env:RS_AI_ALIST_N128"
Write-Host "[ENV] UHD_ROOT=$env:UHD_ROOT"
Write-Host "[ENV] UHD_IMAGES_DIR=$env:UHD_IMAGES_DIR"

