param(
    [string]$InputFile = "",
    [string]$OutputFile = "",
    [ValidateSet("Realtime", "Bringup", "LowRate", "StableGpu")]
    [string]$Profile = "Realtime",
    [switch]$LoopFile,
    [double]$TxGain = 5,
    [double]$RxGain = 20,
    [string]$TxAntenna = "TX/RX",
    [string]$RxAntenna = "RX2",
    [int]$TxChannel = -1,
    [int]$RxChannel = -1,
    [ValidateSet("cpu", "cuda-bp-osd", "cuda-osd")]
    [string]$Decoder = "cpu",
    [int]$RxLeadSeconds = 3,
    [double]$DurationSec = 0,
    [string]$TxAddr = "192.168.40.2",
    [string]$RxAddr = "192.168.50.2",
    [string]$TxDeviceArgs = "",
    [string]$RxDeviceArgs = "",
    [string]$UhdRoot = $(if ($env:UHD_ROOT) { $env:UHD_ROOT } elseif ($env:UHD_PKG_PATH) { $env:UHD_PKG_PATH } else { "" }),
    [string]$UhdImagesDir = "",
    [string]$Config = "Release",
    [double]$SampleRateHz = 0,
    [double]$FreqHz = 0,
    [double]$SyncThreshold = -1,
    [int]$ActiveSc = 0,
    [int]$PilotPeriod = 0,
    [int]$RadioOversample = 1,
    [int]$RxQueueBlocks = 256,
    [int]$RxFrameQueue = 512,
    [int]$RxBlockSamps = 32768,
    [double]$RxSnrGateDb = -120,
    [int]$RxReportEvery = 100,
    [int]$TxReportEvery = 100,
    [switch]$VerboseErrorFrames,
    [switch]$Adaptive,
    [int]$AdaptiveWindowFrames = 20,
    [int]$TxRepeatMin = 1,
    [int]$TxRepeatMax = 3,
    [int]$AdaptiveFeedbackPort = 65435,
    [switch]$UiDashboard,
    [int]$UiMetricsPort = 0,
    [switch]$ProfilePipeline,
    [string]$LogDir = ""
)

$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$exe = Join-Path $repo "build\uhd_cpp\$Config\uhd_ldpc_ofdm_link.exe"

if ($InputFile -eq "") {
    $InputFile = Join-Path $repo "uhd_cpp\media\test_pattern.ppm"
}
if ($OutputFile -eq "") {
    $OutputFile = Join-Path $repo "build\uhd_cpp\rx_media_payload.bin"
}

$requestedActiveSc = $ActiveSc
$requestedPilotPeriod = $PilotPeriod

if ($Profile -eq "Bringup") {
    $activeSc = 64
    $pilotPeriod = 2
    $alistName = "CCSDS_ldpc_n128_k64.alist"
    $sampleRate = "10e6"
} elseif ($Profile -eq "LowRate") {
    $activeSc = 512
    $pilotPeriod = 4
    $alistName = if ($Decoder -eq "cuda-bp-osd" -or $Decoder -eq "cuda-osd") { "CCSDS_ldpc_n128_k64.alist" } else { "CCSDS_ldpc_n512_k256.alist" }
    $sampleRate = "2e6"
    if (-not $PSBoundParameters.ContainsKey("TxGain")) {
        $TxGain = 20
    }
    if (-not $PSBoundParameters.ContainsKey("RxGain")) {
        $RxGain = 25
    }
} elseif ($Profile -eq "StableGpu") {
    $activeSc = 256
    $pilotPeriod = 4
    $alistName = "CCSDS_ldpc_n128_k64.alist"
    $sampleRate = "8e6"
    if (-not $PSBoundParameters.ContainsKey("Decoder")) {
        $Decoder = "cuda-bp-osd"
    }
    if (-not $PSBoundParameters.ContainsKey("TxGain")) {
        $TxGain = 20
    }
    if (-not $PSBoundParameters.ContainsKey("RxGain")) {
        $RxGain = 25
    }
} else {
    $activeSc = 512
    $pilotPeriod = 4
    $alistName = if ($Decoder -eq "cuda-bp-osd" -or $Decoder -eq "cuda-osd") { "CCSDS_ldpc_n128_k64.alist" } else { "CCSDS_ldpc_n512_k256.alist" }
    $sampleRate = "10e6"
}
if (($Decoder -eq "cuda-bp-osd" -or $Decoder -eq "cuda-osd")) {
    if (-not $PSBoundParameters.ContainsKey("RxQueueBlocks")) {
        $RxQueueBlocks = 1024
    }
    if (-not $PSBoundParameters.ContainsKey("RxFrameQueue")) {
        $RxFrameQueue = 1024
    }
}
if ($SampleRateHz -gt 0) {
    $sampleRate = "$SampleRateHz"
}
if ($requestedActiveSc -gt 0) {
    $activeSc = $requestedActiveSc
}
if ($requestedPilotPeriod -gt 0) {
    $pilotPeriod = $requestedPilotPeriod
}

$alist = Join-Path $repo "Code_Matrices_Lib\LDPC\$alistName"

if (-not (Test-Path $exe)) {
    throw "Executable not found: $exe"
}
if (-not (Test-Path $alist)) {
    throw "LDPC alist not found: $alist"
}
if (-not (Test-Path $InputFile)) {
    throw "Input media file not found: $InputFile"
}

$commonArgs = @(
    "--traffic", "file",
    "--alist", $alist,
    "--active-sc", "$activeSc",
    "--pilot-period", "$pilotPeriod",
    "--mcr", "200e6",
    "--rate", "$sampleRate",
    "--radio-oversample", "$RadioOversample",
    "--rx-queue-blocks", "$RxQueueBlocks",
    "--rx-frame-queue", "$RxFrameQueue",
    "--rx-block-samps", "$RxBlockSamps",
    "--rx-snr-gate-db", "$RxSnrGateDb",
    "--ldpc-iter", "20",
    "--decoder", $Decoder
)
if ($FreqHz -gt 0) {
    $commonArgs += @("--freq", "$FreqHz")
}
if ($SyncThreshold -ge 0) {
    $commonArgs += @("--sync-threshold", "$SyncThreshold")
}
if ($UiDashboard) {
    $commonArgs += "--ui-dashboard"
}
if (-not $VerboseErrorFrames) {
    $commonArgs += "--suppress-error-frames"
}
if ($UiMetricsPort -gt 0) {
    $commonArgs += @("--ui-metrics-port", "$UiMetricsPort")
}
if ($ProfilePipeline) {
    $commonArgs += "--profile-pipeline"
}
if ($Adaptive) {
    $commonArgs += @(
        "--adaptive",
        "--adaptive-window-frames", "$AdaptiveWindowFrames",
        "--tx-repeat-min", "$TxRepeatMin",
        "--tx-repeat-max", "$TxRepeatMax",
        "--adaptive-feedback-port", "$AdaptiveFeedbackPort"
    )
}
if ($TxChannel -ge 0) {
    $txChannelArgs = @("--tx-channel", "$TxChannel")
} else {
    $txChannelArgs = @()
}
if ($RxChannel -ge 0) {
    $rxChannelArgs = @("--rx-channel", "$RxChannel")
} else {
    $rxChannelArgs = @()
}

$rxDurationSec = $DurationSec
if ($DurationSec -gt 0) {
    $rxDurationSec = $DurationSec + $RxLeadSeconds + 2
}
if ($LoopFile) {
    $commonArgs += "--loop-file"
}
if ($UhdRoot -ne "") {
    if (-not (Test-Path $UhdRoot)) {
        throw "UHD root not found: $UhdRoot"
    }
    $env:UHD_PKG_PATH = $UhdRoot
    $uhdBin = Join-Path $UhdRoot "bin"
    $env:PATH = "$uhdBin;$env:PATH"
    if ($UhdImagesDir -eq "") {
        $UhdImagesDir = Join-Path $UhdRoot "share\uhd\images"
    }
    $rfnocDir = Join-Path $UhdRoot "share\uhd\rfnoc"
    if (Test-Path $rfnocDir) {
        $env:UHD_RFNOC_DIR = $rfnocDir
    }
    Write-Host "UHD_ROOT: $UhdRoot"
}
if ($UhdImagesDir -ne "") {
    if (-not (Test-Path $UhdImagesDir)) {
        throw "UHD images directory not found: $UhdImagesDir"
    }
    $env:UHD_IMAGES_DIR = $UhdImagesDir
    Write-Host "UHD_IMAGES_DIR: $env:UHD_IMAGES_DIR"
}

$rxArgs = @(
    "--mode", "rx",
    "--rx-addr", $RxAddr,
    "--rx-gain", "$RxGain",
    "--antenna", "$RxAntenna",
    "--output", $OutputFile,
    "--report-every", "$RxReportEvery"
) + $rxChannelArgs + $commonArgs
if ($RxDeviceArgs -ne "") {
    $rxArgs += @("--args", $RxDeviceArgs)
}
if ($DurationSec -gt 0) {
    $rxArgs += @("--duration", "$rxDurationSec")
}

$txArgs = @(
    "--mode", "tx",
    "--tx-addr", $TxAddr,
    "--tx-gain", "$TxGain",
    "--antenna", "$TxAntenna",
    "--input", $InputFile,
    "--report-every", "$TxReportEvery"
) + $txChannelArgs + $commonArgs
if ($TxDeviceArgs -ne "") {
    $txArgs += @("--args", $TxDeviceArgs)
}
if ($DurationSec -gt 0) {
    $txArgs += @("--duration", "$DurationSec")
}

Write-Host "Starting UHD media RX first..."
Write-Host "Profile: $Profile, input: $InputFile"
Write-Host "Output: $OutputFile"
Write-Host "PHY args: activeSC=$activeSc pilotPeriod=$pilotPeriod sampleRate=$sampleRate radioOversample=$RadioOversample syncThreshold=$SyncThreshold"
if ($LogDir -ne "") {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
}

$rxStart = @{
    FilePath = $exe
    ArgumentList = $rxArgs
    PassThru = $true
    WindowStyle = "Hidden"
}
if ($LogDir -ne "") {
    $rxStart.RedirectStandardOutput = Join-Path $LogDir "rx.log"
    $rxStart.RedirectStandardError = Join-Path $LogDir "rx.err.log"
}
$rxProc = Start-Process @rxStart

Start-Sleep -Seconds $RxLeadSeconds

Write-Host "Starting UHD media TX..."
$txStart = @{
    FilePath = $exe
    ArgumentList = $txArgs
    PassThru = $true
    WindowStyle = "Hidden"
}
if ($LogDir -ne "") {
    $txStart.RedirectStandardOutput = Join-Path $LogDir "tx.log"
    $txStart.RedirectStandardError = Join-Path $LogDir "tx.err.log"
}
$txProc = Start-Process @txStart

Write-Host "RX PID: $($rxProc.Id)"
Write-Host "TX PID: $($txProc.Id)"

Wait-Process -Id $rxProc.Id, $txProc.Id

$rxProc.Refresh()
$txProc.Refresh()
Write-Host "RX exit code: $($rxProc.ExitCode)"
Write-Host "TX exit code: $($txProc.ExitCode)"
if ($rxProc.ExitCode -ne 0 -or $txProc.ExitCode -ne 0) {
    throw "UHD media pair failed. RX=$($rxProc.ExitCode), TX=$($txProc.ExitCode)"
}

