param(
    [string]$RepoRoot = $(if ($env:PROJECT_ROOT) { $env:PROJECT_ROOT } else { (Get-Location).Path }),
    [string]$Exe = "",
    [string]$MatrixFile = "",
    [string]$LogRoot = "",
    [ValidateSet("SingleTrx", "DualGated")]
    [string]$RadioMode = "SingleTrx",
    [ValidateSet("Auto", "Stable", "HighThroughput")]
    [string]$LinkProfile = "Auto",
    [string]$UsrpArgs = "type=x300,resource=RIO1",
    [string]$TxDeviceArgs = "type=x300,resource=RIO1",
    [string]$RxDeviceArgs = "type=x300,resource=RIO0",
    [string]$TxSubdev = "A:0",
    [string]$RxSubdev = "B:0",
    [string]$TxAntenna = "TX/RX",
    [string]$RxAntenna = "RX2",
    [int]$TxChannel = 0,
    [int]$RxChannel = -1,
    [ValidateSet("test", "file")]
    [string]$Traffic = "test",
    [string]$InputFile = "",
    [string]$OutputFile = "",
    [double]$Freq = 5.0e9,
    [double]$Rate = 12.5e6,
    [double]$Mcr = 200e6,
    [double[]]$TxGains = @(20, 25, 30),
    [double[]]$RxGains = @(25, 30, 35),
    [int[]]$NumSymbols = @(60, 120, 180),
    [ValidateSet("OneFactor", "FullFactorial")]
    [string]$Design = "OneFactor",
    [double]$BaselineTxGain = 25,
    [double]$BaselineRxGain = 30,
    [int]$BaselineNumSymbols = 120,
    [int]$DurationSec = 20,
    [int]$RxTailSec = 12,
    [int]$ReportEvery = 200,
    [int]$WarmupFrames = 200,
    [int]$TxQueueFrames = 128,
    [int]$RxQueueBlocks = 1024,
    [int]$RxFrameQueue = 1024,
    [int]$RxBlockSamps = 32768,
    [int]$StartTimeoutSec = 180,
    [int]$ActiveSc = 720,
    [int]$PilotPeriod = 4,
    [int]$Nfft = 1024,
    [int]$Cp = 72,
    [ValidateSet("bpsk", "qpsk", "16qam", "64qam")]
    [string]$Modulation = "qpsk",
    [ValidateSet("fc32", "sc16")]
    [string]$TxHostFormat = "sc16",
    [ValidateSet("random-qpsk", "zc-ofdm")]
    [string]$SyncPreamble = "zc-ofdm",
    [int]$SyncZcRoot = 29,
    [switch]$ParallelInit,
    [switch]$DisableGpuPipeline,
    [switch]$DisableRxGpuBuffered,
    [switch]$UiDashboard,
    [switch]$Resume,
    [switch]$DryRun,
    [string]$UhdRoot = $(if ($env:UHD_ROOT) { $env:UHD_ROOT } elseif ($env:UHD_PKG_PATH) { $env:UHD_PKG_PATH } else { "" }),
    [string]$TorchLib = $(if ($env:TORCH_LIB_DIR) { $env:TORCH_LIB_DIR } else { "" }),
    [string]$CudaOsdDllDir = $(if ($env:CUDA_OSD_DLL_DIR) { $env:CUDA_OSD_DLL_DIR } elseif ($env:CUDA_OSD_ROOT) { Join-Path $env:CUDA_OSD_ROOT "build\vs2022\Release" } else { "" }),
    [string]$CudaBin = $(if ($env:CUDA_PATH) { Join-Path $env:CUDA_PATH "bin" } else { "" })
)

$ErrorActionPreference = "Stop"

if ($Exe -eq "") {
    $Exe = Join-Path $RepoRoot "build\uhd_cpp_gpu_pipeline\Release\uhd_ldpc_ofdm_link.exe"
}
if ($MatrixFile -eq "") {
    $MatrixFile = Join-Path $RepoRoot "Code_Matrices_Lib\LDPC\DVB_S2_short_N16200_rate_1_4.alist"
}
if ($LogRoot -eq "") {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogRoot = Join-Path $RepoRoot "build\uhd_cpp_gpu_pipeline\logs\usrp_gain_frame_sweep_$stamp"
}
if (-not (Test-Path -LiteralPath $Exe)) {
    throw "Executable not found: $Exe"
}
if (-not (Test-Path -LiteralPath $MatrixFile)) {
    throw "LDPC matrix not found: $MatrixFile"
}
if ($Traffic -eq "file") {
    if ($InputFile -eq "" -or -not (Test-Path -LiteralPath $InputFile)) {
        throw "Traffic=file requires an existing -InputFile"
    }
    if ($OutputFile -eq "") {
        $OutputFile = Join-Path $LogRoot "rx_payload.bin"
    }
}

$matrixDims = ((Get-Content -LiteralPath $MatrixFile -TotalCount 1) -split '\s+' | Where-Object { $_ -ne "" })
$codeN = [int]$matrixDims[0]
$codeK = $codeN - [int]$matrixDims[1]
if ($codeN -le 0 -or $codeK -le 0) {
    throw "Invalid LDPC matrix dimensions: $MatrixFile"
}

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
$rawDir = Join-Path $LogRoot "usrp"
New-Item -ItemType Directory -Force -Path $rawDir | Out-Null

$env:UHD_IMAGES_DIR = Join-Path $UhdRoot "share\uhd\images"
$env:UHD_PKG_PATH = $UhdRoot
$env:UHD_RFNOC_DIR = Join-Path $UhdRoot "share\uhd\rfnoc"
$env:PATH = "$TorchLib;$(Join-Path $UhdRoot 'bin');$CudaOsdDllDir;$CudaBin;$env:PATH"

$EffectiveRxChannel = if ($RxChannel -ge 0) {
    $RxChannel
} elseif ($RadioMode -eq "SingleTrx") {
    1
} else {
    0
}

if ($RadioMode -eq "DualGated" -and ($LinkProfile -eq "Auto" -or $LinkProfile -eq "Stable")) {
    # Dual-device RX currently uses the standalone RX path, so choose a profile
    # that keeps CPU sync/demod ahead of the UHD sample stream.
    if (-not $PSBoundParameters.ContainsKey("Rate")) { $Rate = 6.25e6 }
    if (-not $PSBoundParameters.ContainsKey("ActiveSc")) { $ActiveSc = 450 }
    if (-not $PSBoundParameters.ContainsKey("Cp")) { $Cp = 128 }
    if (-not $PSBoundParameters.ContainsKey("NumSymbols")) { $NumSymbols = @(96) }
    if (-not $PSBoundParameters.ContainsKey("BaselineNumSymbols")) { $BaselineNumSymbols = 96 }
    if (-not $PSBoundParameters.ContainsKey("RxQueueBlocks")) { $RxQueueBlocks = 2048 }
    if (-not $PSBoundParameters.ContainsKey("RxFrameQueue")) { $RxFrameQueue = 2048 }
    if (-not $PSBoundParameters.ContainsKey("SyncZcRoot")) { $SyncZcRoot = 17 }
}
if ($RadioMode -eq "DualGated" -and $LinkProfile -eq "HighThroughput") {
    # Verified two-device profile: higher rate while keeping standalone RX below
    # the overrun point. Longer 120-symbol frames were less reliable in testing.
    if (-not $PSBoundParameters.ContainsKey("Rate")) { $Rate = 12.5e6 }
    if (-not $PSBoundParameters.ContainsKey("ActiveSc")) { $ActiveSc = 450 }
    if (-not $PSBoundParameters.ContainsKey("Cp")) { $Cp = 128 }
    if (-not $PSBoundParameters.ContainsKey("NumSymbols")) { $NumSymbols = @(96) }
    if (-not $PSBoundParameters.ContainsKey("BaselineNumSymbols")) { $BaselineNumSymbols = 96 }
    if (-not $PSBoundParameters.ContainsKey("RxQueueBlocks")) { $RxQueueBlocks = 4096 }
    if (-not $PSBoundParameters.ContainsKey("RxFrameQueue")) { $RxFrameQueue = 4096 }
    if (-not $PSBoundParameters.ContainsKey("SyncZcRoot")) { $SyncZcRoot = 17 }
}
if ($PSBoundParameters.ContainsKey("NumSymbols") -and
    -not $PSBoundParameters.ContainsKey("BaselineNumSymbols") -and
    $NumSymbols.Count -gt 0) {
    $BaselineNumSymbols = [int]$NumSymbols[0]
}
$GpuPipelineEnabled = (($RadioMode -eq "SingleTrx" -or $RadioMode -eq "DualGated") -and -not $DisableGpuPipeline)
$RxGpuBufferedEnabled = (($RadioMode -eq "SingleTrx" -or $RadioMode -eq "DualGated") -and -not $DisableRxGpuBuffered)
if ($RadioMode -eq "DualGated" -and $Rate -ge 10e6 -and $ActiveSc -ge 700) {
    Write-Warning (("DualGated is using a wide waveform at rate={0} activeSC={1}. " +
        "Keep RX full-GPU pipeline enabled, or use -LinkProfile Stable for conservative RF/host headroom.") -f $Rate,$ActiveSc)
}

function Get-DataSymbols {
    param([int]$Symbols, [int]$Period)
    $pilotCount = 0
    for ($s = 0; $s -lt $Symbols; $s += $Period) {
        $pilotCount++
    }
    return $Symbols - $pilotCount
}

function Stop-StaleLinkProcesses {
    $stale = @(Get-Process -Name uhd_ldpc_ofdm_link -ErrorAction SilentlyContinue)
    if ($stale.Count -gt 0) {
        Write-Host "[CLEANUP] stopping stale link process(es): $($stale.Id -join ', ')"
        $stale | Stop-Process -Force
        Start-Sleep -Seconds 2
    }
    if (Get-Process -Name uhd_ldpc_ofdm_link -ErrorAction SilentlyContinue) {
        throw "A prior uhd_ldpc_ofdm_link process is still running."
    }
}

function Format-NativeCommand {
    param([string]$File, [string[]]$ArgsList)
    return "& `"$File`" " + (($ArgsList | ForEach-Object {
        if ($_ -match '\s') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }) -join ' ')
}

function Invoke-LinkRun {
    param([string[]]$ArgsList, [string]$LogFile)
    $commandText = Format-NativeCommand -File $Exe -ArgsList $ArgsList
    $commandText | Out-File -LiteralPath $LogFile -Encoding Unicode
    if ($DryRun) {
        return 0
    }
    "" | Out-File -LiteralPath $LogFile -Append -Encoding Unicode
    $cmdArgs = (($ArgsList | ForEach-Object {
        if ($_ -match '[\s"&|<>^]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }) -join ' ')
    $native = "`"$Exe`" $cmdArgs 2>&1"
    & $env:ComSpec /D /S /C $native | ForEach-Object {
        $_ | Out-File -LiteralPath $LogFile -Append -Encoding Unicode
        Write-Host $_
    }
    if ($null -eq $LASTEXITCODE) { return 0 }
    return [int]$LASTEXITCODE
}

function Test-LogReady {
    param([string]$Path, [string]$Pattern)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    return [bool](Select-String -LiteralPath $Path -Pattern $Pattern -Quiet)
}

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
        $ready = Test-LogReady -Path $LogPath -Pattern $ReadyPattern
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

function Join-GatedLogs {
    param(
        [string]$CombinedLog,
        [string]$RxLog,
        [string]$RxErr,
        [string]$TxLog,
        [string]$TxErr
    )
    "`r`n===== RX STDOUT =====" | Out-File -LiteralPath $CombinedLog -Append -Encoding Unicode
    if (Test-Path -LiteralPath $RxLog) { Get-Content -LiteralPath $RxLog -Raw | Out-File -LiteralPath $CombinedLog -Append -Encoding Unicode }
    "`r`n===== RX STDERR =====" | Out-File -LiteralPath $CombinedLog -Append -Encoding Unicode
    if (Test-Path -LiteralPath $RxErr) { Get-Content -LiteralPath $RxErr -Raw | Out-File -LiteralPath $CombinedLog -Append -Encoding Unicode }
    "`r`n===== TX STDOUT =====" | Out-File -LiteralPath $CombinedLog -Append -Encoding Unicode
    if (Test-Path -LiteralPath $TxLog) { Get-Content -LiteralPath $TxLog -Raw | Out-File -LiteralPath $CombinedLog -Append -Encoding Unicode }
    "`r`n===== TX STDERR =====" | Out-File -LiteralPath $CombinedLog -Append -Encoding Unicode
    if (Test-Path -LiteralPath $TxErr) { Get-Content -LiteralPath $TxErr -Raw | Out-File -LiteralPath $CombinedLog -Append -Encoding Unicode }
}

function Invoke-GatedPairRun {
    param(
        [string[]]$CommonArgs,
        [string]$CaseDir,
        [string]$CombinedLogFile,
        [double]$TxGain,
        [double]$RxGain
    )
    New-Item -ItemType Directory -Force -Path $CaseDir | Out-Null
    $gateFile = Join-Path $CaseDir "start_gate.go"
    $rxLog = Join-Path $CaseDir "rx.log"
    $rxErr = Join-Path $CaseDir "rx.err.log"
    $txLog = Join-Path $CaseDir "tx.log"
    $txErr = Join-Path $CaseDir "tx.err.log"
    Remove-Item -LiteralPath $gateFile,$rxLog,$rxErr,$txLog,$txErr,$CombinedLogFile -Force -ErrorAction SilentlyContinue

    $rxDuration = if ($Traffic -eq "test") {
        $DurationSec
    } else {
        [Math]::Max(0, $DurationSec + $RxTailSec)
    }
    $rxArgs = @(
        "--mode", "rx",
        "--args", $RxDeviceArgs,
        "--rx-subdev", $RxSubdev,
        "--rx-channel", "$EffectiveRxChannel",
        "--antenna", $RxAntenna,
        "--rx-gain", "$RxGain",
        "--duration", "$rxDuration",
        "--report-every", "$ReportEvery",
        "--start-gate-file", $gateFile,
        "--start-gate-timeout-sec", "$StartTimeoutSec"
    ) + $CommonArgs
    if ($GpuPipelineEnabled) {
        $rxArgs += "--gpu-pipeline"
    }
    if ($RxGpuBufferedEnabled) {
        $rxArgs += "--rx-gpu-buffered"
    }
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
        "--report-every", "$ReportEvery",
        "--start-gate-file", $gateFile,
        "--start-gate-timeout-sec", "$StartTimeoutSec"
    ) + $CommonArgs
    if ($Traffic -eq "file") {
        $txArgs += @("--input", $InputFile)
    }

    @(
        "[DUAL-GATED] RX: $RxDeviceArgs $RxSubdev ch$EffectiveRxChannel ant=$RxAntenna gain=$RxGain duration=${rxDuration}s",
        "[DUAL-GATED] TX: $TxDeviceArgs $TxSubdev ch$TxChannel ant=$TxAntenna gain=$TxGain duration=${DurationSec}s",
        "[DUAL-GATED] gate: $gateFile",
        "RX command:",
        (Format-NativeCommand -File $Exe -ArgsList $rxArgs),
        "TX command:",
        (Format-NativeCommand -File $Exe -ArgsList $txArgs)
    ) | Out-File -LiteralPath $CombinedLogFile -Encoding Unicode

    if ($DryRun) {
        return 0
    }

    $rxProc = $null
    $txProc = $null
    try {
        if ($ParallelInit) {
            $rxProc = Start-Process -FilePath $Exe -ArgumentList $rxArgs -PassThru -WindowStyle Hidden -RedirectStandardOutput $rxLog -RedirectStandardError $rxErr
            $txProc = Start-Process -FilePath $Exe -ArgumentList $txArgs -PassThru -WindowStyle Hidden -RedirectStandardOutput $txLog -RedirectStandardError $txErr
            Wait-ProcessReady -Proc $rxProc -Role "RX" -LogPath $rxLog -ReadyPattern "RX ready:" -TimeoutSec $StartTimeoutSec
            Wait-ProcessReady -Proc $txProc -Role "TX" -LogPath $txLog -ReadyPattern "TX ready:" -TimeoutSec $StartTimeoutSec
        } else {
            $rxProc = Start-Process -FilePath $Exe -ArgumentList $rxArgs -PassThru -WindowStyle Hidden -RedirectStandardOutput $rxLog -RedirectStandardError $rxErr
            Wait-ProcessReady -Proc $rxProc -Role "RX" -LogPath $rxLog -ReadyPattern "RX ready:" -TimeoutSec $StartTimeoutSec
            $txProc = Start-Process -FilePath $Exe -ArgumentList $txArgs -PassThru -WindowStyle Hidden -RedirectStandardOutput $txLog -RedirectStandardError $txErr
            Wait-ProcessReady -Proc $txProc -Role "TX" -LogPath $txLog -ReadyPattern "TX ready:" -TimeoutSec $StartTimeoutSec
        }
        Set-Content -LiteralPath $gateFile -Value "go $(Get-Date -Format o)"
        Write-Host "[GATE] released: $gateFile"

        $txProc.WaitForExit()
        $rxProc.WaitForExit()
        $txProc.Refresh()
        $rxProc.Refresh()
        Join-GatedLogs -CombinedLog $CombinedLogFile -RxLog $rxLog -RxErr $rxErr -TxLog $txLog -TxErr $txErr
        "RX exit code: $($rxProc.ExitCode)`r`nTX exit code: $($txProc.ExitCode)" |
            Out-File -LiteralPath (Join-Path $CaseDir "exit.info.txt") -Encoding UTF8
        if ($rxProc.ExitCode -ne 0 -or $txProc.ExitCode -ne 0) {
            return 1
        }
        return 0
    } catch {
        Join-GatedLogs -CombinedLog $CombinedLogFile -RxLog $rxLog -RxErr $rxErr -TxLog $txLog -TxErr $txErr
        throw
    } finally {
        foreach ($proc in @($rxProc, $txProc)) {
            if ($null -ne $proc) {
                $proc.Refresh()
                if (-not $proc.HasExited) {
                    $proc | Stop-Process -Force -ErrorAction SilentlyContinue
                }
            }
        }
    }
}

function Read-CaseMetric {
    param(
        [string]$LogFile,
        [double]$TxGain,
        [double]$RxGain,
        [int]$Symbols,
        [int]$DataSymbols,
        [int]$CodedBits,
        [string]$Status
    )
    if (-not (Test-Path -LiteralPath $LogFile)) {
        return $null
    }
    $raw = (Get-Content -LiteralPath $LogFile -Raw) -replace "`0", ""
    $summary = [regex]::Matches($raw,
        'frames=(\d+)\s+ok=(\d+)\s+err=(\d+)\s+FER=([0-9.eE+\-]+)\s+BER=([^ ]+).*?fps=([0-9.eE+\-]+)\s+goodput=([0-9.eE+\-]+)\s+Mbps') |
        Select-Object -Last 1
    if ($null -eq $summary) {
        return $null
    }
    $syncLine = ([regex]::Matches($raw, '\[SYNC\].*') | Select-Object -Last 1).Value
    $profile = [regex]::Match($raw,
        'decode_consume: frames=(\d+)\s+fps=([0-9.eE+\-]+)\s+info=([0-9.eE+\-]+)\s+Mbps\s+total_ms/frame=([0-9.eE+\-]+)')
    $decode = [regex]::Match($raw,
        'decode_breakdown_ms/frame:\s+cfo=([0-9.eE+\-]+)\s+fft=([0-9.eE+\-]+)\s+channel=([0-9.eE+\-]+)\s+llr=([0-9.eE+\-]+)\s+fec=([0-9.eE+\-]+)\s+ref_ber=([0-9.eE+\-]+)')
    $phy = [regex]::Match($raw, 'frameSamples=(\d+)\s+rate=([0-9.eE+\-]+)\s+Msps')
    $txUnderflowMarkers = 0
    $rxOverrunMarkers = 0
    foreach ($line in ($raw -split "`r?`n")) {
        if ($line -match '^([UO]+)') {
            foreach ($marker in $matches[1].ToCharArray()) {
                if ($marker -eq 'U') { $txUnderflowMarkers++ }
                if ($marker -eq 'O') { $rxOverrunMarkers++ }
            }
        }
    }
    $rxOverflowEvents = ([regex]::Matches($raw, '\[(?:TRX|RX)\] overflow; dropping buffered samples')).Count
    $syncFound = $null
    $syncMiss = $null
    $trackingMiss = $null
    $syncIncomplete = $null
    $gpuIqDropped = $null
    $deviceFrameDropped = $null
    if ($syncLine -ne "") {
        if ($syncLine -match 'found=(\d+)') { $syncFound = [int]$matches[1] }
        if ($syncLine -match 'miss=(\d+)') { $syncMiss = [int]$matches[1] }
        if ($syncLine -match 'trackingMiss=(\d+)') { $trackingMiss = [int]$matches[1] }
        if ($syncLine -match 'incomplete=(\d+)') { $syncIncomplete = [int]$matches[1] }
        if ($syncLine -match 'gpu_iq_buffer_dropped=(\d+)') { $gpuIqDropped = [int]$matches[1] }
        if ($syncLine -match 'device_frame_queue_dropped=(\d+)') { $deviceFrameDropped = [int]$matches[1] }
    }
    [pscustomobject]@{
        tx_gain_db = $TxGain
        rx_gain_db = $RxGain
        num_symbols = $Symbols
        data_symbols = $DataSymbols
        active_sc = $ActiveSc
        coded_bits_per_frame = $CodedBits
        frame_samples = if ($phy.Success) { [int]$phy.Groups[1].Value } else { $null }
        rate_sps = $Rate
        code = [IO.Path]::GetFileNameWithoutExtension($MatrixFile)
        N = $codeN
        K = $codeK
        modulation = "qpsk"
        pipeline = if ($RadioMode -eq "SingleTrx") { "single_usrp_trx_full_gpu_pipeline" } else { "dual_usrp_gated_gpu_pipeline" }
        frames = [int]$summary.Groups[1].Value
        ok = [int]$summary.Groups[2].Value
        err = [int]$summary.Groups[3].Value
        FER = [double]$summary.Groups[4].Value
        BER = $summary.Groups[5].Value
        fps = [double]$summary.Groups[6].Value
        goodput_mbps = [double]$summary.Groups[7].Value
        decode_total_ms_per_frame = if ($profile.Success) { [double]$profile.Groups[4].Value } else { $null }
        fec_ms_per_frame = if ($decode.Success) { [double]$decode.Groups[5].Value } else { $null }
        sync_found = $syncFound
        sync_miss = $syncMiss
        tracking_miss = $trackingMiss
        sync_incomplete = $syncIncomplete
        gpu_iq_buffer_dropped = $gpuIqDropped
        device_frame_queue_dropped = $deviceFrameDropped
        tx_underflow_markers = $txUnderflowMarkers
        rx_overrun_markers = $rxOverrunMarkers
        rx_overflow_events = $rxOverflowEvents
        uhd_overflow_markers = $txUnderflowMarkers
        status = $Status
        log = $LogFile
    }
}

$metricsPath = Join-Path $LogRoot "usrp_gain_frame_metrics.csv"
$manifestPath = Join-Path $LogRoot "manifest.csv"
$readmePath = Join-Path $LogRoot "README.txt"
$rows = New-Object System.Collections.Generic.List[object]
$manifest = New-Object System.Collections.Generic.List[object]
$cases = New-Object System.Collections.Generic.List[object]
$caseKeys = @{}
function Add-SweepCase {
    param([double]$TxGain, [double]$RxGain, [int]$Symbols)
    $key = "$Symbols|$TxGain|$RxGain"
    if (-not $caseKeys.ContainsKey($key)) {
        $caseKeys[$key] = $true
        $cases.Add([pscustomobject]@{
            TxGain = $TxGain
            RxGain = $RxGain
            Symbols = $Symbols
        })
    }
}
if ($Design -eq "FullFactorial") {
    foreach ($symbols in $NumSymbols) {
        foreach ($txGain in $TxGains) {
            foreach ($rxGain in $RxGains) {
                Add-SweepCase -TxGain $txGain -RxGain $rxGain -Symbols $symbols
            }
        }
    }
} else {
    foreach ($txGain in $TxGains) {
        Add-SweepCase -TxGain $txGain -RxGain $BaselineRxGain -Symbols $BaselineNumSymbols
    }
    foreach ($rxGain in $RxGains) {
        Add-SweepCase -TxGain $BaselineTxGain -RxGain $rxGain -Symbols $BaselineNumSymbols
    }
    foreach ($symbols in $NumSymbols) {
        Add-SweepCase -TxGain $BaselineTxGain -RxGain $BaselineRxGain -Symbols $symbols
    }
}
$total = $cases.Count
$index = 0

foreach ($case in $cases) {
    $symbols = [int]$case.Symbols
    $txGain = [double]$case.TxGain
    $rxGain = [double]$case.RxGain
    $dataSymbols = Get-DataSymbols -Symbols $symbols -Period $PilotPeriod
    $codedBits = $ActiveSc * $dataSymbols * 2
    if (($codedBits % $codeN) -ne 0) {
        Write-Host "[SKIP] symbols=$symbols gives codedBits=$codedBits, not divisible by N=$codeN."
        continue
    }
    $index++
    $safeTx = ("{0:g}" -f $txGain).Replace(".", "p")
    $safeRx = ("{0:g}" -f $rxGain).Replace(".", "p")
    $caseName = "sym${symbols}_tx${safeTx}_rx${safeRx}"
    $caseDir = Join-Path $rawDir $caseName
    $logFile = if ($RadioMode -eq "SingleTrx") {
        Join-Path $rawDir ($caseName + ".log")
    } else {
        Join-Path $caseDir "combined.log"
    }
    Write-Progress -Activity "USRP gain/frame sweep" -Status "$index/$total $caseName" -PercentComplete ([Math]::Round(100 * $index / $total, 1))
    if ($Resume -and (Test-Path -LiteralPath $logFile) -and
        (Select-String -LiteralPath $logFile -Pattern '^frames=\d+\s+ok=\d+\s+err=\d+\s+FER=' -Quiet)) {
        Write-Host "[RESUME] retaining completed case $caseName"
        $status = "RESUME_EXISTING"
    } else {
        if (-not $DryRun) {
            Stop-StaleLinkProcesses
            Start-Sleep -Seconds 1
        }
        $commonArgs = @(
                    "--alist", $MatrixFile,
                    "--decoder", "cuda-bp",
                    "--modulation", $Modulation,
                    "--ldpc-iter", "50",
                    "--ldpc-normalization", "0.95",
                    "--ldpc-offset", "0",
                    "--ldpc-damping", "0",
                    "--ldpc-schedule", "2",
                    "--freq", "$Freq",
                    "--traffic", $Traffic,
                    "--systematic-front-info",
                    "--nfft", "$Nfft",
                    "--cp", "$Cp",
                    "--num-symbols", "$symbols",
                    "--active-sc", "$ActiveSc",
                    "--pilot-period", "$PilotPeriod",
                    "--sync-preamble", $SyncPreamble,
                    "--sync-zc-root", "$SyncZcRoot",
                    "--rx-channel-est", "pilot",
                    "--rate", "$Rate",
                    "--mcr", "$Mcr",
                    "--tx-host-format", "$TxHostFormat",
                    "--tx-queue-frames", "$TxQueueFrames",
                    "--rx-queue-blocks", "$RxQueueBlocks",
                    "--rx-frame-queue", "$RxFrameQueue",
                    "--rx-block-samps", "$RxBlockSamps",
                    "--warmup-frames", "$WarmupFrames",
                    "--profile-pipeline",
                    "--suppress-error-frames"
        )
        if ($UiDashboard) {
            $commonArgs += "--ui-dashboard"
        }
        if ($GpuPipelineEnabled -and $RadioMode -eq "SingleTrx") {
            $commonArgs += "--gpu-pipeline"
        }
        if ($RxGpuBufferedEnabled -and $RadioMode -eq "SingleTrx") {
            $commonArgs += "--rx-gpu-buffered"
        }
        Write-Host "[RUN $index/$total] mode=$RadioMode symbols=$symbols tx=$txGain rx=$rxGain duration=${DurationSec}s"
        if ($RadioMode -eq "SingleTrx") {
            $args = @(
                "--mode", "trx",
                "--args", $UsrpArgs,
                "--duration", "$DurationSec",
                "--report-every", "$ReportEvery",
                "--tx-channel", "$TxChannel",
                "--rx-channel", "$EffectiveRxChannel",
                "--tx-antenna", $TxAntenna,
                "--rx-antenna", $RxAntenna,
                "--tx-gain", "$txGain",
                "--rx-gain", "$rxGain"
            ) + $commonArgs
            if ($Traffic -eq "file") {
                $args += @("--input", $InputFile, "--output", $OutputFile)
            }
            $exitCode = Invoke-LinkRun -ArgsList $args -LogFile $logFile
        } else {
            $exitCode = Invoke-GatedPairRun -CommonArgs $commonArgs -CaseDir $caseDir -CombinedLogFile $logFile -TxGain $txGain -RxGain $rxGain
        }
        $status = if ($DryRun) { "DRYRUN" } elseif ($exitCode -eq 0) { "OK" } elseif ($exitCode -eq 2) { "DONE_WITH_ERRORS" } else { "FAILED" }
    }
    $manifest.Add([pscustomobject]@{
        case = $caseName
        radio_mode = $RadioMode
        tx_gain_db = $txGain
        rx_gain_db = $rxGain
        num_symbols = $symbols
        data_symbols = $dataSymbols
        coded_bits_per_frame = $codedBits
        status = $status
        log = $logFile
    })
    $metric = Read-CaseMetric -LogFile $logFile -TxGain $txGain -RxGain $rxGain -Symbols $symbols -DataSymbols $dataSymbols -CodedBits $codedBits -Status $status
    if ($null -ne $metric) {
        $rows.Add($metric)
        $rows | Export-Csv -LiteralPath $metricsPath -NoTypeInformation -Encoding UTF8
    }
    $manifest | Export-Csv -LiteralPath $manifestPath -NoTypeInformation -Encoding UTF8
}
if (-not $DryRun) {
    Stop-StaleLinkProcesses
}
@(
    "USRP gain/frame-length sweep",
    "Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
    "Radio mode: $RadioMode",
    "Link profile: $LinkProfile",
    "SingleTrx device: $UsrpArgs; TX ch$TxChannel $TxAntenna; RX ch$EffectiveRxChannel $RxAntenna",
    "DualGated TX: $TxDeviceArgs $TxSubdev ch$TxChannel $TxAntenna; RX: $RxDeviceArgs $RxSubdev ch$EffectiveRxChannel $RxAntenna; rxTail=${RxTailSec}s",
    "PHY fixed: freq=$Freq Hz rate=$Rate Sps nfft=$Nfft cp=$Cp activeSC=$ActiveSc pilotPeriod=$PilotPeriod QPSK txHostFormat=$TxHostFormat txQueueFrames=$TxQueueFrames",
    "Buffers: rxQueueBlocks=$RxQueueBlocks rxFrameQueue=$RxFrameQueue rxBlockSamps=$RxBlockSamps gpuPipeline=$GpuPipelineEnabled rxGpuBuffered=$RxGpuBufferedEnabled warmupFrames=$WarmupFrames",
    "Sync: preamble=$SyncPreamble zcRoot=$SyncZcRoot channelEst=pilot",
    "FEC fixed: $([IO.Path]::GetFileName($MatrixFile)) N=$codeN K=$codeK decoder=cuda-bp",
    "Duration per case: $DurationSec s",
    "Report interval: $ReportEvery decoded/transmitted frames",
    "Design: $Design; baseline tx=$BaselineTxGain rx=$BaselineRxGain symbols=$BaselineNumSymbols",
    "TX gains: $($TxGains -join ', ')",
    "RX gains: $($RxGains -join ', ')",
    "Frame lengths (OFDM data symbols parameter): $($NumSymbols -join ', ')",
    "Events: tx_underflow_markers counts UHD U markers; rx_overrun_markers counts UHD O markers; rx_overflow_events counts explicit application RX overflow. uhd_overflow_markers is retained as a legacy alias for TX underrun.",
    "Metrics: $metricsPath",
    "Manifest: $manifestPath"
) | Out-File -LiteralPath $readmePath -Encoding UTF8
Write-Progress -Activity "USRP gain/frame sweep" -Completed
Write-Host "Sweep complete. Metrics: $metricsPath"

