param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [string]$InputFile = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'media\test_video1.mp4'),
    [string]$OutputFile = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'logs\rx_test_video1.mp4'),
    [int]$DurationSec = 60,
    [int]$TxRepeat = 3,
    [switch]$LoopFile = $true
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'setup_env.ps1') -ProjectRoot $ProjectRoot

$exe = Join-Path $ProjectRoot 'bin\uhd_ldpc_ofdm_link.exe'
$alist = Join-Path $ProjectRoot 'matrices\LDPC\CCSDS_ldpc_n128_k64.alist'
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputFile) | Out-Null

$loopArg = @()
if ($LoopFile) {
    $loopArg = @('--loop-file')
}

& $exe `
    --mode trx `
    --traffic file `
    --input $InputFile `
    --output $OutputFile `
    @loopArg `
    --duration $DurationSec `
    --args 'type=x300,resource=RIO1' `
    --tx-subdev 'A:0' `
    --rx-subdev 'B:0' `
    --tx-channel 0 `
    --rx-channel 1 `
    --tx-ant 'TX/RX' `
    --rx-ant 'RX2' `
    --freq 5e9 `
    --mcr 200e6 `
    --rate 12.5e6 `
    --active-sc 768 `
    --num-symbols 96 `
    --pilot-period 4 `
    --modulation qpsk `
    --alist $alist `
    --systematic-front-info `
    --decoder cuda-osd `
    --gpu-pipeline `
    --rx-buffered `
    --rx-buffer-samples 40000000 `
    --tx-gain 25 `
    --rx-gain 30 `
    --tx-repeat $TxRepeat `
    --profile-pipeline `
    --report-every 50
