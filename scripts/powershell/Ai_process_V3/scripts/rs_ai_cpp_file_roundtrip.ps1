param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [ValidateSet('video', 'images', 'viso')][string]$Mode = 'viso',
    [string]$VideoFile = '',
    [string]$ImageDir = '',
    [string]$FramesDir = '',
    [string]$SequenceRoot = '',
    [string]$AnnotationFile = '',
    [Parameter(Mandatory = $true)][string]$ModelPath,
    [ValidateSet('sim', 'copy')][string]$LinkMode = 'sim',
    [string]$BasebandExe = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'bin\uhd_ldpc_ofdm_link.exe'),
    [string]$TxBitstream = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'logs\baseband\rs_ai_tx_features.rsbf'),
    [string]$RxBitstream = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'logs\baseband\rs_ai_rx_features.rsbf'),
    [string]$OutputJson = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'logs\baseband\rs_ai_detections.json'),
    [string]$ManifestJson = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'logs\baseband\rs_ai_source_manifest.json'),
    [string]$InspectJson = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'logs\baseband\rs_ai_tx_features.inspect.json'),
    [string]$UiDir = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'logs\baseband\ui_assets'),
    [string]$OutputVideo = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'logs\baseband\ui_assets\annotated.mp4'),
    [string]$MetricsJson = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'logs\baseband\ui_assets\ui_metrics.json'),
    [string]$Pattern = '*.jpg',
    [int]$SplitLayer = 22,
    [int]$ImageSize = 640,
    [string]$Device = 'cuda:0',
    [int]$MaxFrames = 120,
    [int]$Stride = 1,
    [double]$SnrDb = 8.0,
    [string]$Decoder = 'cuda-osd',
    [string]$Alist = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'matrices\LDPC\CCSDS_ldpc_n128_k64.alist'),
    [double]$Rate = 12.5e6,
    [double]$Freq = 5e9,
    [int]$ActiveSc = 768,
    [int]$NumSymbols = 96,
    [int]$PilotPeriod = 4,
    [string]$Modulation = 'qpsk',
    [ValidateSet('int8', 'float16', 'float32')][string]$TensorCodec = 'int8',
    [string]$Python = 'python',
    [switch]$Half,
    [switch]$Render
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'setup_env.ps1') -ProjectRoot $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot;$env:PYTHONPATH"

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $TxBitstream) | Out-Null
New-Item -ItemType Directory -Force -Path $UiDir | Out-Null

$payloadArgs = @{
    ProjectRoot = $ProjectRoot
    Mode = $Mode
    ModelPath = $ModelPath
    OutputBitstream = $TxBitstream
    InspectJson = $InspectJson
    ManifestJson = $ManifestJson
    Pattern = $Pattern
    SplitLayer = $SplitLayer
    ImageSize = $ImageSize
    Device = $Device
    MaxFrames = $MaxFrames
    Stride = $Stride
    TensorCodec = $TensorCodec
    Python = $Python
}
if ($VideoFile) { $payloadArgs.VideoFile = $VideoFile }
if ($ImageDir) { $payloadArgs.ImageDir = $ImageDir }
if ($FramesDir) { $payloadArgs.FramesDir = $FramesDir }
if ($SequenceRoot) { $payloadArgs.SequenceRoot = $SequenceRoot }
if ($AnnotationFile) { $payloadArgs.AnnotationFile = $AnnotationFile }
if ($Half) { $payloadArgs.Half = $true }

& (Join-Path $PSScriptRoot 'rs_ai_make_baseband_payload.ps1') @payloadArgs

if ($LinkMode -eq 'sim') {
    if (-not (Test-Path $BasebandExe)) {
        throw "Baseband executable not found: $BasebandExe"
    }
    & $BasebandExe `
        --mode sim `
        --traffic file `
        --input $TxBitstream `
        --output $RxBitstream `
        --alist $Alist `
        --systematic-front-info `
        --rate $Rate `
        --freq $Freq `
        --active-sc $ActiveSc `
        --num-symbols $NumSymbols `
        --pilot-period $PilotPeriod `
        --modulation $Modulation `
        --decoder $Decoder `
        --sim-snr-db $SnrDb
}
else {
    Copy-Item -LiteralPath $TxBitstream -Destination $RxBitstream -Force
}

& (Join-Path $PSScriptRoot 'rs_ai_decode_detection.ps1') `
    -ProjectRoot $ProjectRoot `
    -BitstreamFile $RxBitstream `
    -ModelPath $ModelPath `
    -OutputJson $OutputJson `
    -SplitLayer $SplitLayer `
    -ImageSize $ImageSize `
    -Device $Device `
    -TensorCodec $TensorCodec `
    -Python $Python

if ($Render) {
    $renderArgs = @(
        '-m', 'rs_ai_link', 'render-detections',
        '--detections-json', $OutputJson,
        '--output-dir', (Join-Path $UiDir 'annotated'),
        '--output-video', $OutputVideo,
        '--metrics-json', $MetricsJson,
        '--tx-bitstream', $TxBitstream,
        '--rx-bitstream', $RxBitstream,
        '--track'
    )
    switch ($Mode) {
        'video' {
            if (-not $VideoFile) { throw 'Render requires -VideoFile for Mode=video.' }
            $renderArgs += @('--video', $VideoFile)
        }
        'images' {
            if (-not $ImageDir) { throw 'Render requires -ImageDir for Mode=images.' }
            $renderArgs += @('--image-dir', $ImageDir, '--pattern', $Pattern)
        }
        'viso' {
            if ($VideoFile) {
                $renderArgs += @('--video', $VideoFile)
            }
            elseif ($FramesDir) {
                $renderArgs += @('--image-dir', $FramesDir, '--pattern', $Pattern)
            }
            else {
                throw 'Render for Mode=viso requires -VideoFile or -FramesDir.'
            }
            if ($AnnotationFile) {
                $renderArgs += @('--gt-annotation', $AnnotationFile)
            }
            $renderArgs += @('--fps', 8)
        }
    }
    & $Python @renderArgs
}

Write-Host "[RS-AI] tx_bitstream=$TxBitstream"
Write-Host "[RS-AI] rx_bitstream=$RxBitstream"
Write-Host "[RS-AI] detections_json=$OutputJson"
if ($Render) {
    Write-Host "[RS-AI] annotated_video=$OutputVideo"
    Write-Host "[RS-AI] ui_metrics=$MetricsJson"
}
