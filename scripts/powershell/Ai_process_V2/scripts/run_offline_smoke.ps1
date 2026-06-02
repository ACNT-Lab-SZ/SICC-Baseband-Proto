param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [int]$Frames = 20,
    [double]$SnrDb = 7.0
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'setup_env.ps1') -ProjectRoot $ProjectRoot

$exe = Join-Path $ProjectRoot 'bin\uhd_ldpc_ofdm_link.exe'
$alist = Join-Path $ProjectRoot 'matrices\LDPC\CCSDS_ldpc_n128_k64.alist'

& $exe `
    --mode sim `
    --frames $Frames `
    --alist $alist `
    --systematic-front-info `
    --rate 12.5e6 `
    --freq 5e9 `
    --active-sc 768 `
    --num-symbols 96 `
    --pilot-period 4 `
    --modulation qpsk `
    --decoder cuda-osd `
    --cuda-min-batch 30 `
    --cuda-max-batch 4096 `
    --cuda-latency-us 2000 `
    --sim-snr-db $SnrDb
