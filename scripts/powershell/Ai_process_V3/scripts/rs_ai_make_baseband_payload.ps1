param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [ValidateSet('video', 'images', 'viso')][string]$Mode = 'video',
    [string]$VideoFile = '',
    [string]$ImageDir = '',
    [string]$FramesDir = '',
    [string]$SequenceRoot = '',
    [string]$AnnotationFile = '',
    [Parameter(Mandatory = $true)][string]$ModelPath,
    [string]$OutputBitstream = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'logs\baseband\rs_ai_tx_features.rsbf'),
    [string]$InspectJson = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'logs\baseband\rs_ai_tx_features.inspect.json'),
    [string]$ManifestJson = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'logs\baseband\rs_ai_source_manifest.json'),
    [string]$Pattern = '*.jpg',
    [int]$SplitLayer = 22,
    [int]$ImageSize = 640,
    [string]$Device = 'cuda:0',
    [int]$MaxFrames = 0,
    [int]$Stride = 1,
    [ValidateSet('int8', 'float16', 'float32')][string]$TensorCodec = 'int8',
    [int]$ZlibLevel = 1,
    [string]$Python = 'python',
    [switch]$Half
)

$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = "$ProjectRoot;$env:PYTHONPATH"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputBitstream) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $InspectJson) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ManifestJson) | Out-Null

$argsList = @()
switch ($Mode) {
    'video' {
        if (-not $VideoFile) {
            throw 'Mode=video requires -VideoFile.'
        }
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
            '--tensor-codec', $TensorCodec,
            '--zlib-level', $ZlibLevel
        )
    }
    'images' {
        if (-not $ImageDir) {
            throw 'Mode=images requires -ImageDir.'
        }
        $argsList = @(
            '-m', 'rs_ai_link', 'encode-images',
            '--image-dir', $ImageDir,
            '--pattern', $Pattern,
            '--model', $ModelPath,
            '--output', $OutputBitstream,
            '--split-layer', $SplitLayer,
            '--device', $Device,
            '--imgsz', $ImageSize,
            '--max-images', $MaxFrames,
            '--tensor-codec', $TensorCodec,
            '--zlib-level', $ZlibLevel
        )
    }
    'viso' {
        if (-not $VideoFile -and -not $FramesDir -and -not $SequenceRoot) {
            throw 'Mode=viso requires -VideoFile, -FramesDir, or -SequenceRoot.'
        }
        $argsList = @('-m', 'rs_ai_link', 'encode-viso')
        if ($SequenceRoot) {
            $argsList += @('--sequence', $SequenceRoot)
        }
        if ($VideoFile) {
            $argsList += @('--video', $VideoFile)
        }
        if ($FramesDir) {
            $argsList += @('--frames-dir', $FramesDir, '--pattern', $Pattern)
        }
        if ($AnnotationFile) {
            $argsList += @('--annotation', $AnnotationFile)
        }
        $argsList += @(
            '--model', $ModelPath,
            '--output', $OutputBitstream,
            '--manifest-json', $ManifestJson,
            '--split-layer', $SplitLayer,
            '--device', $Device,
            '--imgsz', $ImageSize,
            '--max-frames', $MaxFrames,
            '--tensor-codec', $TensorCodec,
            '--zlib-level', $ZlibLevel
        )
    }
}

if ($Half) {
    $argsList += '--half'
}

& $Python @argsList

$inspectArgs = @('-m', 'rs_ai_link', 'inspect-bitstream', '--input', $OutputBitstream, '--max-frames', 1)
& $Python @inspectArgs | Set-Content -Encoding utf8 $InspectJson

Write-Host "[RS-AI] baseband_payload=$OutputBitstream"
Write-Host "[RS-AI] inspect_json=$InspectJson"
