param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [string]$UhdRoot = $(if ($env:UHD_ROOT) { $env:UHD_ROOT } else { $(if ($env:UHD_ROOT) { $env:UHD_ROOT } elseif ($env:UHD_PKG_PATH) { $env:UHD_PKG_PATH } else { "" }) })
)

$ErrorActionPreference = 'Stop'

$bin = Join-Path $ProjectRoot 'bin'
$env:PATH = "$bin;$UhdRoot\bin;$env:PATH"
$env:UHD_ROOT = $UhdRoot
$env:UHD_PKG_PATH = $UhdRoot
$env:UHD_IMAGES_DIR = Join-Path $UhdRoot 'share\uhd\images'
$env:UHD_RFNOC_DIR = Join-Path $UhdRoot 'share\uhd\rfnoc'

Write-Host "[ENV] ProjectRoot=$ProjectRoot"
Write-Host "[ENV] UHD_ROOT=$env:UHD_ROOT"
Write-Host "[ENV] UHD_IMAGES_DIR=$env:UHD_IMAGES_DIR"

