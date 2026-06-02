param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [int]$Frames = 200,
    [int]$SnrStart = 3,
    [int]$SnrStop = 10,
    [string]$Decoder = 'cuda-osd'
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'setup_env.ps1') -ProjectRoot $ProjectRoot

$exe = Join-Path $ProjectRoot 'bin\uhd_ldpc_ofdm_link.exe'
$alist = Join-Path $ProjectRoot 'matrices\LDPC\CCSDS_ldpc_n128_k64.alist'
$logDir = Join-Path $ProjectRoot 'logs\awgn_sweep'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$rows = @()
foreach ($snr in $SnrStart..$SnrStop) {
    $log = Join-Path $logDir ("snr_{0}dB_{1}.log" -f $snr, $Decoder)
    Write-Host "[RUN] SNR=$snr dB decoder=$Decoder"
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
        --decoder $Decoder `
        --cuda-min-batch 30 `
        --cuda-max-batch 4096 `
        --cuda-latency-us 2000 `
        --sim-snr-db $snr `
        --suppress-error-frames *> $log

    $summary = Select-String -Path $log -Pattern 'frames=.*FER=' | Select-Object -Last 1
    if ($summary) {
        $line = $summary.Line
        $rows += [pscustomobject]@{
            SNR_dB = $snr
            Frames = ([regex]::Match($line, 'frames=(\d+)').Groups[1].Value)
            OK = ([regex]::Match($line, 'ok=(\d+)').Groups[1].Value)
            Err = ([regex]::Match($line, 'err=(\d+)').Groups[1].Value)
            FER = ([regex]::Match($line, 'FER=([0-9.eE+-]+)').Groups[1].Value)
            BER = ([regex]::Match($line, 'BER=([0-9.eE+-]+)').Groups[1].Value)
            PreBER = ([regex]::Match($line, 'preBER=([0-9.eE+-]+)').Groups[1].Value)
            FPS = ([regex]::Match($line, 'fps=([0-9.]+)').Groups[1].Value)
            Goodput_Mbps = ([regex]::Match($line, 'goodput=([0-9.]+)').Groups[1].Value)
            Log = $log
        }
    }
}

$csv = Join-Path $logDir 'summary.csv'
$rows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $csv
$rows | Format-Table -AutoSize
Write-Host "[CSV] $csv"
