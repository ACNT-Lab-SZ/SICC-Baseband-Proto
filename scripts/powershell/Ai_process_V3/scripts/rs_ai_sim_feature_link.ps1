param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [Parameter(Mandatory = $true)][string]$VideoFile,
    [Parameter(Mandatory = $true)][string]$ModelPath,
    [string]$TxBitstream = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'logs\rs_ai_tx_features.rsbf'),
    [string]$RxBitstream = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'logs\rs_ai_rx_features.rsbf'),
    [string]$OutputJson = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'logs\rs_ai_detections.json'),
    [double]$SnrDb = 8.0,
    [int]$SplitLayer = 22,
    [int]$ImageSize = 640,
    [string]$Device = 'cuda:0',
    [int]$MaxFrames = 30,
    [ValidateSet('int8', 'float16', 'float32')][string]$TensorCodec = 'int8',
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'setup_env.ps1') -ProjectRoot $ProjectRoot

& (Join-Path $PSScriptRoot 'rs_ai_encode_video.ps1') `
    -ProjectRoot $ProjectRoot `
    -VideoFile $VideoFile `
    -ModelPath $ModelPath `
    -OutputBitstream $TxBitstream `
    -SplitLayer $SplitLayer `
    -ImageSize $ImageSize `
    -Device $Device `
    -MaxFrames $MaxFrames `
    -TensorCodec $TensorCodec `
    -Python $Python

$exe = Join-Path $ProjectRoot 'bin\uhd_ldpc_ofdm_link.exe'
$alist = Join-Path $ProjectRoot 'matrices\LDPC\CCSDS_ldpc_n128_k64.alist'
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $RxBitstream) | Out-Null

& $exe `
    --mode sim `
    --traffic file `
    --input $TxBitstream `
    --output $RxBitstream `
    --alist $alist `
    --systematic-front-info `
    --rate 12.5e6 `
    --freq 5e9 `
    --active-sc 768 `
    --num-symbols 96 `
    --pilot-period 4 `
    --modulation qpsk `
    --decoder cuda-osd `
    --sim-snr-db $SnrDb

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

