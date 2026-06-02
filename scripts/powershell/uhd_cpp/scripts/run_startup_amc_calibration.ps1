param(
    [ValidateSet("Offline", "USRP", "Both")]
    [string]$Suite = "Both",
    [string]$RepoRoot = $(if ($env:PROJECT_ROOT) { $env:PROJECT_ROOT } else { (Get-Location).Path }),
    [string]$LogRoot = "",
    [int]$OfflineFrames = 10000,
    [int]$OfflineMaxFrameErrors = 100,
    [int]$OfflineZeroErrorStopFrames = 10000,
    [string[]]$EbN0DbList = @("0:2:16"),
    [string[]]$Modulations = @("bpsk", "qpsk", "16qam", "64qam"),
    [int]$UsrpDurationSec = 10,
    [string[]]$UsrpModulations = @("bpsk", "qpsk", "16qam", "64qam"),
    [ValidateSet("Baseline", "GpuRxOnly", "FullGpu", "QualityAndFullGpu", "Compare")]
    [string]$UsrpPipelineMode = "QualityAndFullGpu",
    [double]$TxGain = 23,
    [double]$RxGain = 24,
    [double]$Freq = 5.0e9,
    [double]$Rate = 25e6,
    [string]$UsrpArgs = "type=x300,resource=RIO1",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$sweep = Join-Path $RepoRoot "uhd_cpp\scripts\run_codec_matrix_sweep.ps1"
if (-not (Test-Path -LiteralPath $sweep)) {
    throw "Matrix sweep script not found: $sweep"
}
if ($LogRoot -eq "") {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogRoot = Join-Path $RepoRoot "build\uhd_cpp_gpu_pipeline\logs\startup_amc_calibration_$stamp"
}
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

# CCSDS covers latency-sensitive short packets; DVB-S2 short frames cover sustained payload streams.
$candidatePattern = '^(CCSDS_ldpc_n128_k64|CCSDS_ldpc_n256_k128|CCSDS_ldpc_n512_k256|DVB_S2_short_N16200_rate_1_4|DVB_S2_short_N16200_rate_1_2|DVB_S2_short_N16200_rate_5_6)$'
$offlineRoot = Join-Path $LogRoot "offline"
$usrpRoot = Join-Path $LogRoot "usrp"

if ($Suite -eq "Offline" -or $Suite -eq "Both") {
    Write-Host "[AMC-CAL] Running offline GPU SNR/FER calibration, including CCSDS short-packet candidates."
    $offlineArgs = @{
        Suite = "Offline"
        RepoRoot = $RepoRoot
        LogRoot = $offlineRoot
        OfflineCodePattern = $candidatePattern
        EbN0DbList = $EbN0DbList
        Modulations = $Modulations
        Frames = $OfflineFrames
        MaxFrameErrors = $OfflineMaxFrameErrors
        ZeroErrorStopFrames = $OfflineZeroErrorStopFrames
        LdpcIter = 50
        LdpcNormalization = 0.95
        LdpcOffset = 0
        LdpcDamping = 0
        LdpcSchedule = 2
        OfdmProfile = "Auto"
        ChannelProfile = "None"
        Rate = $Rate
        Cp = 128
        DryRun = $DryRun
    }
    & $sweep @offlineArgs
}

if ($Suite -eq "USRP" -or $Suite -eq "Both") {
    Write-Host "[AMC-CAL] Running USRP calibration. Baseline rows provide SNR/EVM/CFO; GPU rows provide final throughput/drop behavior."
    $usrpSweepArgs = @{
        Suite = "USRP"
        RepoRoot = $RepoRoot
        LogRoot = $usrpRoot
        UsrpCodePattern = $candidatePattern
        UsrpMaxCodeN = 16200
        Modulations = $UsrpModulations
        UsrpDurationSec = $UsrpDurationSec
        UsrpPipelineMode = $UsrpPipelineMode
        LdpcIter = 50
        LdpcNormalization = 0.95
        LdpcOffset = 0
        LdpcDamping = 0
        LdpcSchedule = 2
        OfdmProfile = "Auto"
        ChannelProfile = "None"
        Rate = $Rate
        Mcr = 200e6
        Cp = 128
        UsrpArgs = $UsrpArgs
        Freq = $Freq
        TxChannel = 0
        RxChannel = 1
        TxAntenna = "TX/RX"
        RxAntenna = "RX2"
        TxGain = $TxGain
        RxGain = $RxGain
        DryRun = $DryRun
    }
    & $sweep @usrpSweepArgs
}

$combined = New-Object System.Collections.Generic.List[object]
$offlineCsv = Join-Path $offlineRoot "offline_metrics.csv"
if (Test-Path -LiteralPath $offlineCsv) {
    foreach ($row in (Import-Csv -LiteralPath $offlineCsv)) {
        $combined.Add([pscustomobject]@{
            environment = "offline_gpu"
            code = $row.code
            N = $row.N
            K = $row.K
            rate = $row.rate
            modulation = $row.modulation
            decoder = $row.decoder
            ebn0_db = $row.ebn0_db
            injected_snr_db = $row.injected_snr_db
            frames = $row.frames
            err = $row.err
            FER = $row.FER
            BER = $row.BER
            goodput_mbps = $row.goodput_mbps
            avg_snr_db = ""
            avg_evm_rms = ""
            avg_abs_cfo_hz = ""
            drops = ""
            log = $row.log
        })
    }
}
$usrpCsv = Join-Path $usrpRoot "usrp_metrics.csv"
if (Test-Path -LiteralPath $usrpCsv) {
    foreach ($row in (Import-Csv -LiteralPath $usrpCsv)) {
        $combined.Add([pscustomobject]@{
            environment = "usrp_$($row.usrp_pipeline)"
            code = $row.code
            N = $row.N
            K = $row.K
            rate = $row.rate
            modulation = $row.modulation
            decoder = $row.decoder
            ebn0_db = ""
            injected_snr_db = ""
            frames = $row.frames
            err = $row.err
            FER = $row.FER
            BER = ""
            goodput_mbps = $row.goodput_mbps
            avg_snr_db = $row.avg_snr_db
            avg_evm_rms = $row.avg_evm_rms
            avg_abs_cfo_hz = $row.avg_abs_cfo_hz
            drops = "$($row.iq_buffer_dropped)/$($row.gpu_iq_buffer_dropped)/$($row.device_frame_queue_dropped)"
            log = $row.log
        })
    }
}
if ($combined.Count -gt 0) {
    $combinedPath = Join-Path $LogRoot "amc_calibration_metrics.csv"
    $combined | Export-Csv -LiteralPath $combinedPath -NoTypeInformation -Encoding UTF8
    Write-Host "[AMC-CAL] Combined metrics: $combinedPath"
}

@(
    "Startup AMC calibration",
    "Candidates: CCSDS n128/n256/n512 for short packets; DVB-S2 short 1/4, 1/2, 5/6 for sustained streams.",
    "Offline metrics: $offlineCsv",
    "USRP metrics: $usrpCsv",
    "Interpretation: baseline_cpu_sync_gpu_fec rows expose measured SNR/EVM/CFO; full_gpu_pipeline rows expose final GPU throughput/drop/FER.",
    "Acceptance target: select only profiles whose observed FER is below 1e-3; extend observation for final zero-error claims."
) | Out-File -LiteralPath (Join-Path $LogRoot "README.txt") -Encoding UTF8

Write-Host "[AMC-CAL] Complete: $LogRoot"

