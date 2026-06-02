param(
    [string]$Repo = $(if ($env:PROJECT_ROOT) { $env:PROJECT_ROOT } else { (Get-Location).Path }),
    [string]$Config = "Release",
    [string]$UhdRoot = $(if ($env:UHD_ROOT) { $env:UHD_ROOT } elseif ($env:UHD_PKG_PATH) { $env:UHD_PKG_PATH } else { "" }),
    [string]$TorchLib = $(if ($env:TORCH_LIB_DIR) { $env:TORCH_LIB_DIR } else { "" }),
    [string]$LogRoot = "",
    [switch]$CleanExisting = $true,

    [string]$Traffic = "test",
    [string]$InputFile = "",
    [string]$OutputFile = "",
    [double]$DurationSec = 60,
    [int]$RxTailSec = 12,

    [string]$TxDeviceArgs = "resource=RIO1",
    [string]$RxDeviceArgs = "resource=RIO0",
    [string]$TxSubdev = "A:0",
    [string]$RxSubdev = "B:0",
    [string]$TxAntenna = "TX/RX",
    [string]$RxAntenna = "RX2",
    [int]$TxChannel = 0,
    [int]$RxChannel = 0,

    [double]$FreqHz = 5.0e9,
    [double]$RateHz = 6.25e6,
    [double]$McrHz = 200e6,
    [double]$TxGain = 25,
    [double]$RxGain = 28,
    [double]$Amplitude = 0.55,
    [double]$SyncThreshold = 0.35,
    [ValidateSet("random-qpsk", "zc-ofdm")]
    [string]$SyncPreamble = "random-qpsk",
    [int]$SyncZcRoot = 25,

    [string]$Modulation = "qpsk",
    [string]$Alist = "",
    [string]$Decoder = "cuda-bp-osd",
    [int]$Nfft = 1024,
    [int]$Cp = 72,
    [int]$NumSymbols = 48,
    [int]$ActiveSc = 256,
    [int]$PilotPeriod = 4,
    [int]$LdpcIter = 20,
    [int]$ReportEvery = 100,
    [int]$WarmupFrames = 200,
    [int]$RxQueueBlocks = 2048,
    [int]$RxFrameQueue = 2048,
    [int]$RxBlockSamps = 32768,
    [switch]$RxBuffered,
    [UInt64]$RxBufferSamples = 0,
    [int]$StartTimeoutSec = 180,
    [switch]$ParallelInit,
    [switch]$GpuPipeline,
    [switch]$GpuPhyBackend,
    [switch]$ProfilePipeline,
    [switch]$UiDashboard
)

$ErrorActionPreference = "Stop"

if ($LogRoot -eq "") {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogRoot = Join-Path $Repo "build\uhd_cpp_gpu_pipeline\logs\usrp_gated_pair_$stamp"
}
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

$exe = Join-Path $Repo "build\uhd_cpp_gpu_pipeline\$Config\uhd_ldpc_ofdm_link.exe"
if (-not (Test-Path $exe)) {
    $exe = Join-Path $Repo "build\uhd_cpp\$Config\uhd_ldpc_ofdm_link.exe"
}
if (-not (Test-Path $exe)) {
    throw "Executable not found: $exe"
}

if ($Alist -eq "") {
    $Alist = Join-Path $Repo "Code_Matrices_Lib\LDPC\CCSDS_ldpc_n128_k64.alist"
}
if (-not (Test-Path $Alist)) {
    throw "LDPC alist not found: $Alist"
}

if ($Traffic -eq "file") {
    if ($InputFile -eq "" -or -not (Test-Path $InputFile)) {
        throw "Traffic=file requires an existing -InputFile"
    }
    if ($OutputFile -eq "") {
        $OutputFile = Join-Path $LogRoot "rx_payload.bin"
    }
}

if ($CleanExisting) {
    Get-Process -Name "uhd_ldpc_ofdm_link" -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 2
}

if (-not (Test-Path $UhdRoot)) {
    throw "UHD root not found: $UhdRoot"
}
$env:UHD_PKG_PATH = $UhdRoot
$env:UHD_IMAGES_DIR = Join-Path $UhdRoot "share\uhd\images"
$env:UHD_RFNOC_DIR = Join-Path $UhdRoot "share\uhd\rfnoc"
$env:PATH = "$(Join-Path $UhdRoot 'bin');$TorchLib;$env:PATH"

$gateFile = Join-Path $LogRoot "start_gate.go"
Remove-Item -LiteralPath $gateFile -Force -ErrorAction SilentlyContinue

$commonArgs = @(
    "--traffic", $Traffic,
    "--alist", $Alist,
    "--modulation", $Modulation,
    "--decoder", $Decoder,
    "--systematic-front-info",
    "--freq", "$FreqHz",
    "--rate", "$RateHz",
    "--mcr", "$McrHz",
    "--nfft", "$Nfft",
    "--cp", "$Cp",
    "--num-symbols", "$NumSymbols",
    "--active-sc", "$ActiveSc",
    "--pilot-period", "$PilotPeriod",
    "--ldpc-iter", "$LdpcIter",
    "--warmup-frames", "$WarmupFrames",
    "--sync-preamble", $SyncPreamble,
    "--sync-zc-root", "$SyncZcRoot",
    "--sync-threshold", "$SyncThreshold",
    "--amplitude", "$Amplitude",
    "--rx-queue-blocks", "$RxQueueBlocks",
    "--rx-frame-queue", "$RxFrameQueue",
    "--rx-block-samps", "$RxBlockSamps",
    "--start-gate-file", $gateFile,
    "--start-gate-timeout-sec", "$StartTimeoutSec",
    "--suppress-error-frames"
)
if ($GpuPhyBackend) {
    $commonArgs += "--gpu-phy-backend"
} elseif ($GpuPipeline) {
    $commonArgs += "--gpu-pipeline"
}
if ($ProfilePipeline) {
    $commonArgs += "--profile-pipeline"
}
if ($UiDashboard) {
    $commonArgs += "--ui-dashboard"
}
if ($RxBuffered) {
    $commonArgs += "--rx-buffered"
    if ($RxBufferSamples -gt 0) {
        $commonArgs += @("--rx-buffer-samples", "$RxBufferSamples")
    }
}

$rxDuration = [Math]::Max(0, $DurationSec + $RxTailSec)
$rxArgs = @(
    "--mode", "rx",
    "--args", $RxDeviceArgs,
    "--rx-subdev", $RxSubdev,
    "--rx-channel", "$RxChannel",
    "--antenna", $RxAntenna,
    "--rx-gain", "$RxGain",
    "--duration", "$rxDuration",
    "--report-every", "$ReportEvery"
) + $commonArgs
if ($Traffic -eq "file") {
    $rxArgs += @("--output", $OutputFile)
}

$txArgs = @(
    "--mode", "tx",
    "--args", $TxDeviceArgs,
    "--tx-subdev", $TxSubdev,
    "--tx-channel", "$TxChannel",
    "--antenna", $TxAntenna,
    "--tx-gain", "$TxGain",
    "--duration", "$DurationSec",
    "--report-every", "$ReportEvery"
) + $commonArgs
if ($Traffic -eq "file") {
    $txArgs += @("--input", $InputFile, "--loop-file")
}

$rxLog = Join-Path $LogRoot "rx.log"
$rxErr = Join-Path $LogRoot "rx.err.log"
$txLog = Join-Path $LogRoot "tx.log"
$txErr = Join-Path $LogRoot "tx.err.log"
Remove-Item -LiteralPath $rxLog,$rxErr,$txLog,$txErr -Force -ErrorAction SilentlyContinue

function Test-LogReady {
    param([string]$Path, [string]$Pattern)
    if (-not (Test-Path $Path)) {
        return $false
    }
    return [bool](Select-String -Path $Path -Pattern $Pattern -Quiet)
}

Write-Host "[START] log root: $LogRoot"
Write-Host "[START] RX: $RxDeviceArgs $RxSubdev ch$RxChannel ant=$RxAntenna gain=$RxGain"
Write-Host "[START] TX: $TxDeviceArgs $TxSubdev ch$TxChannel ant=$TxAntenna gain=$TxGain"
Write-Host "[START] PHY: rate=$RateHz freq=$FreqHz activeSC=$ActiveSc symbols=$NumSymbols cp=$Cp mod=$Modulation decoder=$Decoder"

function Wait-ProcessReady {
    param(
        [System.Diagnostics.Process]$Proc,
        [string]$Role,
        [string]$LogPath,
        [string]$ReadyPattern,
        [int]$TimeoutSec
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $Proc.Refresh()
        if ($Proc.HasExited) {
            throw "$Role exited before ready. See $LogPath"
        }
        $ready = Test-LogReady $LogPath $ReadyPattern
        Write-Progress -Activity "Waiting for $Role readiness" -Status "$Role ready=$ready" -PercentComplete 50
        if ($ready) {
            Write-Progress -Activity "Waiting for $Role readiness" -Completed
            return
        }
        Start-Sleep -Milliseconds 300
    }
    Write-Progress -Activity "Waiting for $Role readiness" -Completed
    throw "Timed out waiting for $Role readiness. See $LogPath"
}

if ($ParallelInit) {
    $rxProc = Start-Process -FilePath $exe -ArgumentList $rxArgs -PassThru -WindowStyle Hidden -RedirectStandardOutput $rxLog -RedirectStandardError $rxErr
    $txProc = Start-Process -FilePath $exe -ArgumentList $txArgs -PassThru -WindowStyle Hidden -RedirectStandardOutput $txLog -RedirectStandardError $txErr
    Wait-ProcessReady $rxProc "RX" $rxLog "RX ready:" $StartTimeoutSec
    Wait-ProcessReady $txProc "TX" $txLog "TX ready:" $StartTimeoutSec
} else {
    Write-Host "[START] sequential NI-RIO initialization: RX first, then TX."
    $rxProc = Start-Process -FilePath $exe -ArgumentList $rxArgs -PassThru -WindowStyle Hidden -RedirectStandardOutput $rxLog -RedirectStandardError $rxErr
    Wait-ProcessReady $rxProc "RX" $rxLog "RX ready:" $StartTimeoutSec
    Write-Host "[START] RX ready and parked at start gate. Starting TX..."
    $txProc = Start-Process -FilePath $exe -ArgumentList $txArgs -PassThru -WindowStyle Hidden -RedirectStandardOutput $txLog -RedirectStandardError $txErr
    Wait-ProcessReady $txProc "TX" $txLog "TX ready:" $StartTimeoutSec
}

Set-Content -LiteralPath $gateFile -Value "go $(Get-Date -Format o)"
Write-Host "[GATE] released: $gateFile"

$txProc.WaitForExit()
$rxProc.WaitForExit()
$txProc.Refresh()
$rxProc.Refresh()

$rxExitCode = $rxProc.ExitCode
$txExitCode = $txProc.ExitCode

$exitInfo = @(
    "RX exit code: $rxExitCode"
    "TX exit code: $txExitCode"
    "Log root: $LogRoot"
    "Gate file: $gateFile"
)
$exitInfo | Set-Content -LiteralPath (Join-Path $LogRoot "exit.info.txt")
$exitInfo | ForEach-Object { Write-Host $_ }

if ($null -eq $rxExitCode -or $null -eq $txExitCode) {
    Write-Warning "Process exit code was not reported by PowerShell. Check logs for application errors: $LogRoot"
} elseif ($rxExitCode -ne 0 -or $txExitCode -ne 0) {
    throw "Gated USRP pair failed. RX=$rxExitCode, TX=$txExitCode. See $LogRoot"
}

Write-Host "[DONE] Gated USRP pair completed."

