param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [Parameter(Mandatory = $true)][string]$VideoFile,
    [Parameter(Mandatory = $true)][string]$ModelPath,
    [string]$OutputBitstream = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'logs\rs_ai_tx_features.rsbf'),
    [int]$SplitLayer = 22,
    [int]$ImageSize = 640,
    [string]$Device = 'cuda:0',
    [int]$MaxFrames = 0,
    [int]$Stride = 1,
    [ValidateSet('int8', 'float16', 'float32')][string]$TensorCodec = 'int8',
    [string]$Python = 'python',
    [switch]$Half
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$env:PYTHONPATH = "$ProjectRoot;$env:PYTHONPATH"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputBitstream) | Out-Null

$argsList = @(
    '-m', 'rs_ai_link', 'encode-video',
    '--video', $VideoFile,
    '--model', $ModelPath,
    '--output', $OutputBitstream,
    '--split-layer', $SplitLayer,
    '--device', $Device,
    '--imgsz', $ImageSize,
    '--max-frames', $MaxFrames,
    '--stride', $Stride,
    '--tensor-codec', $TensorCodec
)
if ($Half) {
    $argsList += '--half'
}

& $Python @argsList
