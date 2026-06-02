param(
    [string]$Exe = $(Join-Path $(if ($env:PROJECT_ROOT) { $env:PROJECT_ROOT } else { (Get-Location).Path }) "build\compile_check\uhd_ldpc_ofdm_link_refactor.exe"),
    [string]$InputFile = $(Join-Path $(if ($env:PROJECT_ROOT) { $env:PROJECT_ROOT } else { (Get-Location).Path }) "data\media\portable\usrp_video_link_portable_20260513\media\test_video1.mp4"),
    [string]$OutputFile = $(Join-Path $(if ($env:PROJECT_ROOT) { $env:PROJECT_ROOT } else { (Get-Location).Path }) "generated\rx_test_video1_pxie_refactor_cpu_180s.mp4"),
    [string]$LogDir = $(Join-Path $(if ($env:PROJECT_ROOT) { $env:PROJECT_ROOT } else { (Get-Location).Path }) "generated\logs_pxie_refactor_cpu_180s"),
    [string]$TxDeviceArgs = "resource=RIO1",
    [string]$RxDeviceArgs = "resource=RIO0",
    [string]$UhdRoot = $(if ($env:UHD_ROOT) { $env:UHD_ROOT } elseif ($env:UHD_PKG_PATH) { $env:UHD_PKG_PATH } else { "" }),
    [string]$AlistName = "CCSDS_ldpc_n512_k256.alist",
    [int]$ActiveSc = 512,
    [int]$PilotPeriod = 4,
    [double]$SampleRateHz = 2e6,
    [ValidateSet("cpu", "cuda-bp-osd")]
    [string]$Decoder = "cpu",
    [int]$RxQueueBlocks = 256,
    [int]$RxFrameQueue = 512,
    [int]$RxBlockSamps = 32768,
    [int]$CudaMinBatch = 30,
    [int]$CudaMaxBatch = 4096,
    [int]$CudaLatencyUs = 2000,
    [int]$TxDurationSec = 180,
    [int]$RxDurationSec = 230,
    [int]$TxStartDelaySec = 45
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $UhdRoot)) {
    throw "UHD root not found: $UhdRoot"
}
$env:UHD_PKG_PATH = $UhdRoot
$env:UHD_IMAGES_DIR = Join-Path $UhdRoot "share\uhd\images"
$env:UHD_RFNOC_DIR = Join-Path $UhdRoot "share\uhd\rfnoc"
$env:PATH = "$(Join-Path $UhdRoot "bin");$env:PATH"

$alist = $(Join-Path $(if ($env:PROJECT_ROOT) { $env:PROJECT_ROOT } else { (Get-Location).Path }) "data\code_matrices\Code_Matrices_Lib\LDPC\$AlistName")
if (-not (Test-Path $alist)) {
    throw "LDPC alist not found: $alist"
}
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
if (Test-Path $OutputFile) {
    Remove-Item -LiteralPath $OutputFile -Force
}

$common = @(
    "--traffic", "file",
    "--alist", $alist,
    "--active-sc", "$ActiveSc",
    "--pilot-period", "$PilotPeriod",
    "--mcr", "200e6",
    "--rate", "$SampleRateHz",
    "--ldpc-iter", "20",
    "--decoder", $Decoder,
    "--rx-queue-blocks", "$RxQueueBlocks",
    "--rx-frame-queue", "$RxFrameQueue",
    "--rx-block-samps", "$RxBlockSamps",
    "--cuda-min-batch", "$CudaMinBatch",
    "--cuda-max-batch", "$CudaMaxBatch",
    "--cuda-latency-us", "$CudaLatencyUs",
    "--loop-file"
)

$rxArgs = @(
    "--mode", "rx",
    "--args", $RxDeviceArgs,
    "--rx-gain", "25",
    "--antenna", "RX2",
    "--output", $OutputFile,
    "--duration", "$RxDurationSec",
    "--report-every", "20"
) + $common

$txArgs = @(
    "--mode", "tx",
    "--args", $TxDeviceArgs,
    "--tx-gain", "20",
    "--antenna", "TX/RX",
    "--input", $InputFile,
    "--duration", "$TxDurationSec",
    "--report-every", "50"
) + $common

$rx = Start-Process -FilePath $Exe -ArgumentList $rxArgs -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $LogDir "rx.log") `
    -RedirectStandardError (Join-Path $LogDir "rx.err.log")

Start-Sleep -Seconds $TxStartDelaySec

$tx = Start-Process -FilePath $Exe -ArgumentList $txArgs -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $LogDir "tx.log") `
    -RedirectStandardError (Join-Path $LogDir "tx.err.log")

$rx.WaitForExit()
$tx.WaitForExit()
$rx.Refresh()
$tx.Refresh()

"RX_EXIT=$($rx.ExitCode) TX_EXIT=$($tx.ExitCode) OUTPUT_EXISTS=$(Test-Path $OutputFile)"
if (Test-Path $OutputFile) {
    Get-Item $OutputFile | Select-Object FullName, Length
}

