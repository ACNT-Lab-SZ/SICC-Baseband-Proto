param(
    [double]$TxGain = 10,
    [double]$RxGain = 20,
    [int]$RxLeadSeconds = 3,
    [double]$DurationSec = 60,
    [int]$ActiveSc = 64,
    [int]$PilotPeriod = 2,
    [double]$MasterClockRate = 200e6,
    [double]$SampleRate = 10e6,
    [string]$AlistName = "CCSDS_ldpc_n128_k64.alist",
    [string]$TxAddr = "192.168.40.2",
    [string]$RxAddr = "192.168.50.2",
    [string]$UhdRoot = $(if ($env:UHD_ROOT) { $env:UHD_ROOT } elseif ($env:UHD_PKG_PATH) { $env:UHD_PKG_PATH } else { "" }),
    [string]$Config = "Release"
)

$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$exe = Join-Path $repo "build\uhd_cpp\$Config\uhd_ldpc_ofdm_link.exe"
$alist = Join-Path $repo "Code_Matrices_Lib\LDPC\$AlistName"
$RxDurationSec = $DurationSec + $RxLeadSeconds + 2

if (-not (Test-Path $UhdRoot)) {
    throw "UHD root not found: $UhdRoot"
}
$env:UHD_PKG_PATH = $UhdRoot
$env:UHD_IMAGES_DIR = Join-Path $UhdRoot "share\uhd\images"
$env:UHD_RFNOC_DIR = Join-Path $UhdRoot "share\uhd\rfnoc"
$env:PATH = "$(Join-Path $UhdRoot "bin");$env:PATH"

if (-not (Test-Path $exe)) {
    throw "Executable not found: $exe"
}

if (-not (Test-Path $alist)) {
    throw "LDPC alist not found: $alist"
}

$rxArgs = @(
    "--mode", "rx",
    "--rx-addr", $RxAddr,
    "--alist", $alist,
    "--rx-gain", "$RxGain",
    "--duration", "$RxDurationSec",
    "--active-sc", "$ActiveSc",
    "--pilot-period", "$PilotPeriod",
    "--mcr", "$MasterClockRate",
    "--rate", "$SampleRate",
    "--report-every", "20",
    "--ldpc-iter", "20"
)

$txArgs = @(
    "--mode", "tx",
    "--tx-addr", $TxAddr,
    "--alist", $alist,
    "--tx-gain", "$TxGain",
    "--duration", "$DurationSec",
    "--active-sc", "$ActiveSc",
    "--pilot-period", "$PilotPeriod",
    "--mcr", "$MasterClockRate",
    "--rate", "$SampleRate",
    "--report-every", "50"
)

Write-Host "Starting UHD C++ RX first..."
$rxProc = Start-Process -FilePath $exe -ArgumentList $rxArgs -PassThru -WindowStyle Hidden

Start-Sleep -Seconds $RxLeadSeconds

Write-Host "Starting UHD C++ TX..."
$txProc = Start-Process -FilePath $exe -ArgumentList $txArgs -PassThru -WindowStyle Hidden

Write-Host "RX PID: $($rxProc.Id)"
Write-Host "TX PID: $($txProc.Id)"

Wait-Process -Id $rxProc.Id, $txProc.Id

