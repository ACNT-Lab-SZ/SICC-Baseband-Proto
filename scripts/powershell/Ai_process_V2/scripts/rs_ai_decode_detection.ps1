param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [Parameter(Mandatory = $true)][string]$BitstreamFile,
    [Parameter(Mandatory = $true)][string]$ModelPath,
    [string]$OutputJson = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'logs\rs_ai_detections.json'),
    [int]$SplitLayer = 0,
    [int]$ImageSize = 640,
    [string]$Device = 'cuda:0',
    [double]$Confidence = 0.25,
    [double]$Iou = 0.45,
    [int]$MaxFrames = 0,
    [ValidateSet('int8', 'float16', 'float32')][string]$TensorCodec = 'int8',
    [string]$Python = 'python',
    [switch]$Half
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$env:PYTHONPATH = "$ProjectRoot;$env:PYTHONPATH"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputJson) | Out-Null

$argsList = @(
    '-m', 'rs_ai_link', 'decode-bitstream',
    '--input', $BitstreamFile,
    '--model', $ModelPath,
    '--output-json', $OutputJson,
    '--split-layer', $SplitLayer,
    '--device', $Device,
    '--imgsz', $ImageSize,
    '--conf', $Confidence,
    '--iou', $Iou,
    '--max-frames', $MaxFrames,
    '--tensor-codec', $TensorCodec
)
if ($Half) {
    $argsList += '--half'
}

& $Python @argsList
