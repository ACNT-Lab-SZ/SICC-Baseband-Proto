param(
    [ValidateSet("Offline", "USRP", "Both")]
    [string]$Suite = "Offline",

    [switch]$DryRun,
    [switch]$ThroughputOnly,
    [switch]$Resume,

    [string]$RepoRoot = $(if ($env:PROJECT_ROOT) { $env:PROJECT_ROOT } else { (Get-Location).Path }),
    [string]$Exe = "",
    [string]$MatrixRoot = "",
    [string]$LogRoot = "",
    [string]$OfflineCodePattern = "",

    [string[]]$EbN0DbList = @("0", "2", "4", "6", "8", "10", "12", "14", "16"),
    [string[]]$Modulations = @("bpsk", "qpsk", "16qam", "64qam"),

    [int]$Frames = 300,
    [int]$MaxFrameErrors = 0,
    [int]$ZeroErrorStopFrames = 0,
    [int]$LdpcIter = 8,
    [double]$LdpcNormalization = 0.80,
    [double]$LdpcOffset = 0.15,
    [double]$LdpcDamping = 0.15,
    [int]$LdpcSchedule = -1,
    [switch]$GeneratePlots,
    [string]$MatlabExe = $(if ($env:MATLAB_EXE) { $env:MATLAB_EXE } else { "matlab" }),
    [int]$UsrpDurationSec = 20,
    [int]$UsrpMaxCodeN = 16200,
    [string]$UsrpCodePattern = "",
    [ValidateSet("Baseline", "GpuRxOnly", "FullGpu", "QualityAndFullGpu", "Compare")]
    [string]$UsrpPipelineMode = "Baseline",

    [int]$Nfft = 1024,
    [int]$Cp = 72,
    [int]$NumSymbols = 96,
    [int]$ActiveSc = 750,
    [int]$PilotPeriod = 4,
    [ValidateSet("Auto", "Custom", "DvbS2Universal", "SmallLdpc")]
    [string]$OfdmProfile = "Auto",
    [ValidateSet("None", "Perfect", "Offset1e2", "Custom")]
    [string]$ChannelProfile = "None",
    [string]$PrecompChannelFiles = "",
    [string]$ActualChannelFiles = "",
    [double]$Rate = 25e6,
    [double]$Mcr = 200e6,

    [string]$UsrpArgs = "type=x300,resource=RIO1",
    [double]$Freq = 5.0e9,
    [int]$TxChannel = 0,
    [int]$RxChannel = 1,
    [string]$TxAntenna = "TX/RX",
    [string]$RxAntenna = "RX2",
    [double]$TxGain = 25,
    [double]$RxGain = 30,

    [string]$UhdRoot = $(if ($env:UHD_ROOT) { $env:UHD_ROOT } elseif ($env:UHD_PKG_PATH) { $env:UHD_PKG_PATH } else { "" }),
    [string]$TorchLib = $(if ($env:TORCH_LIB_DIR) { $env:TORCH_LIB_DIR } else { "" }),
    [string]$CudaOsdDllDir = $(if ($env:CUDA_OSD_DLL_DIR) { $env:CUDA_OSD_DLL_DIR } elseif ($env:CUDA_OSD_ROOT) { Join-Path $env:CUDA_OSD_ROOT "build\vs2022\Release" } else { "" }),
    [string]$CudaBin = $(if ($env:CUDA_PATH) { Join-Path $env:CUDA_PATH "bin" } else { "" })
)

$ErrorActionPreference = "Stop"

function ConvertTo-EbN0List {
    param([string[]]$Values)

    $tokens = @($Values | ForEach-Object {
        "$_" -split '[,;\s]+'
    } | Where-Object {
        $_ -ne ""
    })
    $parsed = New-Object System.Collections.Generic.List[double]
    foreach ($token in $tokens) {
        if ($token -match '^([+-]?[0-9]+(?:\.[0-9]+)?):([+-]?[0-9]+(?:\.[0-9]+)?):([+-]?[0-9]+(?:\.[0-9]+)?)$') {
            $start = [Convert]::ToDouble($matches[1], [Globalization.CultureInfo]::InvariantCulture)
            $step = [Convert]::ToDouble($matches[2], [Globalization.CultureInfo]::InvariantCulture)
            $stop = [Convert]::ToDouble($matches[3], [Globalization.CultureInfo]::InvariantCulture)
            if ($step -eq 0 -or (($stop - $start) * $step -lt 0)) {
                throw "Invalid Eb/N0 range '$token'. Expected start:nonzeroStep:stop."
            }
            for ($value = $start; ($step -gt 0 -and $value -le $stop + 1e-9) -or ($step -lt 0 -and $value -ge $stop - 1e-9); $value += $step) {
                $parsed.Add([double]$value)
            }
        } else {
            try {
                $parsed.Add([Convert]::ToDouble($token, [Globalization.CultureInfo]::InvariantCulture))
            } catch {
                throw "Invalid Eb/N0 value '$token'. Use comma-separated points such as '0,2,4' or a range such as '0:2:16'."
            }
        }
    }
    if ($parsed.Count -eq 0) {
        throw "EbN0DbList cannot be empty."
    }
    return @($parsed)
}

$EbN0DbList = @(ConvertTo-EbN0List -Values $EbN0DbList)

$Modulations = @($Modulations | ForEach-Object {
    "$_" -split ','
} | Where-Object {
    $_ -ne ""
} | ForEach-Object {
    $_.Trim().ToLowerInvariant()
})

if ($Exe -eq "") {
    $Exe = Join-Path $RepoRoot "build\uhd_cpp_gpu_pipeline\Release\uhd_ldpc_ofdm_link.exe"
}
if ($MatrixRoot -eq "") {
    $MatrixRoot = Join-Path $RepoRoot "Code_Matrices_Lib"
}
if ($LogRoot -eq "") {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogRoot = Join-Path $RepoRoot "build\uhd_cpp_gpu_pipeline\logs\codec_matrix_sweep_$stamp"
}

if (-not (Test-Path -LiteralPath $Exe)) {
    throw "Executable not found: $Exe"
}
if (-not (Test-Path -LiteralPath $MatrixRoot)) {
    throw "Matrix root not found: $MatrixRoot"
}

$defaultChannelDir = Join-Path $RepoRoot "uhd_cpp\test_channels"
if ($ChannelProfile -eq "Perfect") {
    $PrecompChannelFiles = Join-Path $defaultChannelDir "perfect_equal_multipath.channel"
    $ActualChannelFiles = $PrecompChannelFiles
} elseif ($ChannelProfile -eq "Offset1e2") {
    $PrecompChannelFiles = Join-Path $defaultChannelDir "predicted_base_3path.channel"
    $ActualChannelFiles = Join-Path $defaultChannelDir "actual_offset_1e2_3path.channel"
} elseif ($ChannelProfile -eq "Custom" -and ($PrecompChannelFiles -eq "" -or $ActualChannelFiles -eq "")) {
    throw "ChannelProfile Custom requires both -PrecompChannelFiles and -ActualChannelFiles."
}
if ($PrecompChannelFiles -ne "" -and -not (Test-Path -LiteralPath $PrecompChannelFiles)) {
    throw "Precomp channel file not found: $PrecompChannelFiles"
}
if ($ActualChannelFiles -ne "" -and -not (Test-Path -LiteralPath $ActualChannelFiles)) {
    throw "Actual channel file not found: $ActualChannelFiles"
}

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

$env:UHD_IMAGES_DIR = Join-Path $UhdRoot "share\uhd\images"
$env:UHD_PKG_PATH = $UhdRoot
$env:UHD_RFNOC_DIR = Join-Path $UhdRoot "share\uhd\rfnoc"
$env:PATH = "$TorchLib;$(Join-Path $UhdRoot 'bin');$CudaOsdDllDir;$CudaBin;$env:PATH"

function Get-BitsPerSymbol {
    param([string]$Modulation)
    switch ($Modulation.ToLowerInvariant()) {
        "bpsk" { return 1 }
        "qpsk" { return 2 }
        "16qam" { return 4 }
        "64qam" { return 6 }
        default { throw "Unsupported modulation: $Modulation" }
    }
}

function Get-DataSymbols {
    param([int]$Symbols, [int]$Period)
    if ($Period -gt 0) {
        $pilot = 0
        for ($s = 0; $s -lt $Symbols; $s += $Period) {
            $pilot++
        }
        return $Symbols - $pilot
    }
    return $Symbols - 3
}

function Get-OfdmCaseConfig {
    param([object]$Matrix)

    $profile = $OfdmProfile
    if ($profile -eq "Auto") {
        if ($Matrix.Family -eq "LDPC" -and [int]$Matrix.N -ge 16200) {
            $profile = "DvbS2Universal"
        } else {
            $profile = "SmallLdpc"
        }
    }

    if ($profile -eq "DvbS2Universal") {
        $caseSymbols = 120
        $caseActiveSc = 720
        $casePilotPeriod = 4
    } elseif ($profile -eq "SmallLdpc") {
        $caseSymbols = 20
        $caseActiveSc = 512
        $casePilotPeriod = 4
    } else {
        $caseSymbols = $NumSymbols
        $caseActiveSc = $ActiveSc
        $casePilotPeriod = $PilotPeriod
    }

    $caseDataSymbols = Get-DataSymbols -Symbols $caseSymbols -Period $casePilotPeriod
    [pscustomobject]@{
        Profile = $profile
        Nfft = $Nfft
        Cp = $Cp
        NumSymbols = $caseSymbols
        ActiveSc = $caseActiveSc
        PilotPeriod = $casePilotPeriod
        DataSymbols = $caseDataSymbols
        Rate = $Rate
        Mcr = $Mcr
    }
}

function Get-LdpcDims {
    param([string]$Path)
    $line = Get-Content -LiteralPath $Path -TotalCount 1
    $tok = $line -split '\s+' | Where-Object { $_ -ne "" }
    if ($tok.Count -lt 2) {
        throw "Invalid alist first line: $Path"
    }
    $n = [int]$tok[0]
    $m = [int]$tok[1]
    $k = $n - $m
    if ($n -le 0 -or $k -le 0) {
        throw "Invalid LDPC dimensions in $Path : n=$n m=$m"
    }
    [pscustomobject]@{ N = $n; K = $k; M = $m }
}

function Get-PolarDims {
    param([string]$Path)
    $name = [IO.Path]::GetFileNameWithoutExtension($Path)
    if ($name -match 'N(\d+)_K(\d+)') {
        return [pscustomobject]@{ N = [int]$matches[1]; K = [int]$matches[2]; M = $null }
    }
    [pscustomobject]@{ N = $null; K = $null; M = $null }
}

function Select-Decoder {
    param([string]$Family, [int]$N)
    if ($Family -eq "Polar") {
        return "polar-scl"
    }
    if ($N -le 256) {
        return "cuda-bp-osd"
    }
    return "cuda-bp"
}

function Convert-EbN0ToSimSnr {
    param([double]$EbN0Db, [double]$CodeRate, [int]$BitsPerSymbol)
    return $EbN0Db + 10.0 * [Math]::Log10([Math]::Max($CodeRate * $BitsPerSymbol, 1.0e-12))
}

function Get-MatrixNOrZero {
    param([object]$Matrix)
    if ($null -eq $Matrix.N -or $Matrix.N -eq "") {
        return 0
    }
    return [int]$Matrix.N
}

function New-SafeName {
    param([string]$Text)
    return ($Text -replace '[\\/:*?"<>|\s\[\]\(\)]+', '_').Trim('_')
}

function Invoke-LinkRun {
    param(
        [string[]]$ArgsList,
        [string]$LogFile,
        [switch]$NoRun
    )
    $commandText = "& `"$Exe`" " + (($ArgsList | ForEach-Object {
        if ($_ -match '\s') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }) -join ' ')
    if ($NoRun) {
        $commandText | Out-File -LiteralPath $LogFile -Encoding UTF8
        return 0
    }
    # UHD sends normal INFO/WARNING diagnostics to stderr. Merge the native streams
    # inside cmd.exe so Windows PowerShell receives ordinary output rather than ErrorRecord objects.
    $cmdArguments = (($ArgsList | ForEach-Object {
        if ($_ -match '[\s"&|<>^]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }) -join ' ')
    $nativeCommandLine = "`"$Exe`" $cmdArguments 2>&1"
    # Write both the command and streamed native output in one encoding so later
    # metric extraction can parse logs produced from either PowerShell edition.
    $commandText | Out-File -LiteralPath $LogFile -Encoding Unicode
    "" | Out-File -LiteralPath $LogFile -Append -Encoding Unicode
    & $env:ComSpec /D /S /C $nativeCommandLine | ForEach-Object {
        $_ | Out-File -LiteralPath $LogFile -Append -Encoding Unicode
        Write-Host $_
    }
    $exit = $LASTEXITCODE
    if ($null -eq $exit) {
        return 0
    }
    return [int]$exit
}

$manifest = New-Object System.Collections.Generic.List[object]
$matrices = @()

$ldpcDir = Join-Path $MatrixRoot "LDPC"
if (Test-Path -LiteralPath $ldpcDir) {
    Get-ChildItem -LiteralPath $ldpcDir -Filter *.alist | Sort-Object FullName | ForEach-Object {
        $dims = Get-LdpcDims -Path $_.FullName
        $matrices += [pscustomobject]@{
            Family = "LDPC"
            Path = $_.FullName
            Name = $_.BaseName
            N = $dims.N
            K = $dims.K
            M = $dims.M
            Supported = $true
            UnsupportedReason = ""
        }
    }
}

$polarDir = Join-Path $MatrixRoot "Polar"
if (Test-Path -LiteralPath $polarDir) {
    Get-ChildItem -LiteralPath $polarDir -Filter *.txt | Sort-Object FullName | ForEach-Object {
        $dims = Get-PolarDims -Path $_.FullName
        $matrices += [pscustomobject]@{
            Family = "Polar"
            Path = $_.FullName
            Name = $_.BaseName
            N = $dims.N
            K = $dims.K
            M = $dims.M
            Supported = $false
            UnsupportedReason = "Current uhd_ldpc_ofdm_link CLI wires --alist to LDPC only; Polar txt is loaded by MatrixLoader but not yet connected to PHY TX/RX."
        }
    }
}

if ($matrices.Count -eq 0) {
    throw "No .alist or .txt coding files found under $MatrixRoot"
}

function Add-ManifestRow {
    param(
        [string]$SuiteName,
        [object]$Matrix,
        [string]$Modulation,
        [object]$EbN0Db,
        [object]$SimSnrDb,
        [int]$CodedBits,
        [string]$Decoder,
        [string]$Status,
        [string]$Reason,
        [string]$LogFile,
        [object]$ExitCode,
        [object]$Ofdm = $null,
        [string]$Pipeline = ""
    )
    if ($null -eq $Ofdm) {
        $Ofdm = Get-OfdmCaseConfig -Matrix $Matrix
    }
    $manifest.Add([pscustomobject]@{
        suite = $SuiteName
        family = $Matrix.Family
        code = $Matrix.Name
        path = $Matrix.Path
        N = $Matrix.N
        K = $Matrix.K
        rate = if ($Matrix.N -and $Matrix.K) { [double]$Matrix.K / [double]$Matrix.N } else { $null }
        modulation = $Modulation
        ebn0_db = $EbN0Db
        sim_snr_db = $SimSnrDb
        coded_bits_per_frame = $CodedBits
        ofdm_profile = $Ofdm.Profile
        nfft = $Ofdm.Nfft
        cp = $Ofdm.Cp
        num_symbols = $Ofdm.NumSymbols
        active_sc = $Ofdm.ActiveSc
        pilot_period = $Ofdm.PilotPeriod
        data_symbols = $Ofdm.DataSymbols
        sample_rate = $Ofdm.Rate
        channel_profile = $ChannelProfile
        precomp_channel = $PrecompChannelFiles
        actual_channel = $ActualChannelFiles
        decoder = $Decoder
        usrp_pipeline = $Pipeline
        status = $Status
        reason = $Reason
        log = $LogFile
        exit_code = $ExitCode
    })
}

function New-PhyArgs {
    param([object]$Ofdm)
    $args = @(
        "--traffic", "test",
        "--systematic-front-info",
        "--nfft", "$($Ofdm.Nfft)",
        "--cp", "$($Ofdm.Cp)",
        "--num-symbols", "$($Ofdm.NumSymbols)",
        "--active-sc", "$($Ofdm.ActiveSc)",
        "--pilot-period", "$($Ofdm.PilotPeriod)",
        "--sync-preamble", "zc-ofdm",
        "--sync-zc-root", "29",
        "--rx-channel-est", "pilot",
        "--rate", "$($Ofdm.Rate)",
        "--mcr", "$($Ofdm.Mcr)",
        "--profile-pipeline",
        "--suppress-error-frames"
    )
    if ($ThroughputOnly) {
        $args += "--skip-reference-ber"
    }
    if ($PrecompChannelFiles -ne "") {
        $args += @("--precomp-channel", $PrecompChannelFiles)
    }
    if ($ActualChannelFiles -ne "") {
        $args += @("--actual-channel", $ActualChannelFiles)
    }
    return $args
}

function Get-UsrpPipelineVariants {
    $baseline = [pscustomobject]@{
        Name = "baseline_cpu_sync_gpu_fec"
        Args = @()
    }
    $gpuRxOnly = [pscustomobject]@{
        Name = "gpu_rx_only"
        Args = @("--gpu-rx-sync", "--gpu-rx-demod", "--rx-gpu-buffered")
    }
    $fullGpu = [pscustomobject]@{
        Name = "full_gpu_pipeline"
        Args = @("--gpu-pipeline", "--rx-gpu-buffered")
    }
    switch ($UsrpPipelineMode) {
        "Baseline" { return @($baseline) }
        "GpuRxOnly" { return @($gpuRxOnly) }
        "FullGpu" { return @($fullGpu) }
        "QualityAndFullGpu" { return @($baseline, $fullGpu) }
        "Compare" { return @($baseline, $gpuRxOnly, $fullGpu) }
    }
}

function Test-RunnableCombination {
    param([object]$Matrix, [string]$Modulation, [string]$SuiteName = "")
    if (-not $Matrix.Supported) {
        return $false
    }
    if ($SuiteName -eq "usrp" -and $UsrpMaxCodeN -gt 0 -and [int]$Matrix.N -gt $UsrpMaxCodeN) {
        return $false
    }
    if ($SuiteName -eq "usrp" -and $UsrpCodePattern -ne "" -and $Matrix.Name -notmatch $UsrpCodePattern) {
        return $false
    }
    $ofdm = Get-OfdmCaseConfig -Matrix $Matrix
    $bps = Get-BitsPerSymbol -Modulation $Modulation
    $codedBits = $ofdm.ActiveSc * $ofdm.DataSymbols * $bps
    return (($codedBits % [int]$Matrix.N) -eq 0)
}

$usrpPipelineVariants = @(Get-UsrpPipelineVariants)
$script:totalRuns = 0
foreach ($matrix in $matrices) {
    foreach ($mod in $Modulations) {
        if ($Suite -eq "Offline" -or $Suite -eq "Both") {
            if (($OfflineCodePattern -eq "" -or $matrix.Name -match $OfflineCodePattern) -and
                (Test-RunnableCombination -Matrix $matrix -Modulation $mod -SuiteName "offline_gpu")) {
                $script:totalRuns += $EbN0DbList.Count
            }
        }
        if ($Suite -eq "USRP" -or $Suite -eq "Both") {
            if (Test-RunnableCombination -Matrix $matrix -Modulation $mod -SuiteName "usrp") {
                $script:totalRuns += $usrpPipelineVariants.Count
            }
        }
    }
}
$script:runIndex = 0

function Show-CaseProgress {
    param(
        [string]$SuiteName,
        [object]$Matrix,
        [string]$Modulation,
        [string]$Decoder,
        [object]$EbN0Db,
        [string]$LogFile,
        [string]$PipelineName = ""
    )
    $script:runIndex += 1
    $pct = if ($script:totalRuns -gt 0) {
        [Math]::Min(100, [Math]::Round(100.0 * $script:runIndex / $script:totalRuns, 1))
    } else {
        100
    }
    $ebText = if ($null -ne $EbN0Db -and "$EbN0Db" -ne "") { " Eb/N0=$('{0:g}' -f [double]$EbN0Db)dB" } else { "" }
    $pipelineText = if ($PipelineName -ne "") { " | $PipelineName" } else { "" }
    $status = "$SuiteName | $($Matrix.Name) | $Modulation | $Decoder$pipelineText$ebText"
    Write-Progress -Activity "Codec matrix sweep" -Status $status -PercentComplete $pct
    Write-Host ("[Sweep {0}/{1}] {2} -> {3}" -f $script:runIndex, $script:totalRuns, $status, $LogFile)
}

function Test-CompletedOfflineLog {
    param([string]$Path)
    if (-not $Resume -or -not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    return $null -ne (Select-String -LiteralPath $Path -Pattern '^frames=\d+\s+ok=\d+\s+err=\d+\s+FER=' | Select-Object -Last 1)
}

if ($Suite -eq "Offline" -or $Suite -eq "Both") {
    $offlineDir = Join-Path $LogRoot "offline_gpu"
    New-Item -ItemType Directory -Force -Path $offlineDir | Out-Null
    foreach ($matrix in $matrices) {
        foreach ($mod in $Modulations) {
            if ($OfflineCodePattern -ne "" -and $matrix.Name -notmatch $OfflineCodePattern) {
                continue
            }
            $ofdm = Get-OfdmCaseConfig -Matrix $matrix
            $phyArgs = New-PhyArgs -Ofdm $ofdm
            $bps = Get-BitsPerSymbol -Modulation $mod
            $codedBits = $ofdm.ActiveSc * $ofdm.DataSymbols * $bps
            $decoder = Select-Decoder -Family $matrix.Family -N (Get-MatrixNOrZero $matrix)
            if (-not $matrix.Supported) {
                Add-ManifestRow "offline_gpu" $matrix $mod $null $null $codedBits $decoder "SKIP_UNSUPPORTED" $matrix.UnsupportedReason "" $null $ofdm
                continue
            }
            if (($codedBits % [int]$matrix.N) -ne 0) {
                Add-ManifestRow "offline_gpu" $matrix $mod $null $null $codedBits $decoder "SKIP_INCOMPATIBLE_OFDM" "coded_bits_per_frame is not divisible by code N with OFDM profile $($ofdm.Profile)." "" $null $ofdm
                continue
            }
            $codeRate = [double]$matrix.K / [double]$matrix.N
            foreach ($ebn0 in $EbN0DbList) {
                $simSnr = Convert-EbN0ToSimSnr -EbN0Db $ebn0 -CodeRate $codeRate -BitsPerSymbol $bps
                $caseName = New-SafeName ("{0}_{1}_{2}_EbN0_{3:0.0}dB" -f $matrix.Name, $mod, $decoder, $ebn0)
                $logFile = Join-Path $offlineDir ($caseName + ".log")
                if (Test-CompletedOfflineLog -Path $logFile) {
                    Show-CaseProgress -SuiteName "offline_gpu" -Matrix $matrix -Modulation $mod -Decoder $decoder -EbN0Db $ebn0 -LogFile $logFile
                    $summary = (Select-String -LiteralPath $logFile -Pattern '^frames=\d+\s+ok=\d+\s+err=\d+\s+FER=' | Select-Object -Last 1).Line
                    $existingExit = if ($summary -match '\serr=0\s') { 0 } else { 2 }
                    Add-ManifestRow "offline_gpu" $matrix $mod $ebn0 $simSnr $codedBits $decoder "RESUME_EXISTING" "Existing completed log retained by -Resume." $logFile $existingExit $ofdm
                    continue
                }
                $args = @(
                    "--mode", "gpu-sim",
                    "--alist", $matrix.Path,
                    "--decoder", $decoder,
                    "--modulation", $mod,
                    "--frames", "$Frames",
                    "--sim-snr-db", ("{0:R}" -f $simSnr),
                    "--ldpc-iter", "$LdpcIter",
                    "--ldpc-normalization", ("{0:R}" -f $LdpcNormalization),
                    "--ldpc-offset", ("{0:R}" -f $LdpcOffset),
                    "--ldpc-damping", ("{0:R}" -f $LdpcDamping),
                    "--ldpc-schedule", "$LdpcSchedule"
                ) + $phyArgs
                if ($MaxFrameErrors -gt 0) {
                    $args += @("--max-frame-errors", "$MaxFrameErrors")
                }
                if ($ZeroErrorStopFrames -gt 0) {
                    $args += @("--zero-error-stop-frames", "$ZeroErrorStopFrames")
                }
                Show-CaseProgress -SuiteName "offline_gpu" -Matrix $matrix -Modulation $mod -Decoder $decoder -EbN0Db $ebn0 -LogFile $logFile
                $exit = Invoke-LinkRun -ArgsList $args -LogFile $logFile -NoRun:$DryRun
                $status = if ($DryRun) { "DRYRUN" } elseif ($exit -eq 0) { "OK" } elseif ($exit -eq 2) { "DONE_WITH_FRAME_ERRORS" } else { "FAILED" }
                $reason = if ($exit -eq 2) { "Link executable returned 2 because FER/err was non-zero; this is retained as a measured low-SNR result." } else { "" }
                Add-ManifestRow "offline_gpu" $matrix $mod $ebn0 $simSnr $codedBits $decoder $status $reason $logFile $exit $ofdm
            }
        }
    }
}

if ($Suite -eq "USRP" -or $Suite -eq "Both") {
    $usrpDir = Join-Path $LogRoot "usrp"
    New-Item -ItemType Directory -Force -Path $usrpDir | Out-Null
    foreach ($matrix in $matrices) {
        foreach ($mod in $Modulations) {
            $ofdm = Get-OfdmCaseConfig -Matrix $matrix
            $phyArgs = New-PhyArgs -Ofdm $ofdm
            $bps = Get-BitsPerSymbol -Modulation $mod
            $codedBits = $ofdm.ActiveSc * $ofdm.DataSymbols * $bps
            $decoder = Select-Decoder -Family $matrix.Family -N (Get-MatrixNOrZero $matrix)
            if (-not $matrix.Supported) {
                Add-ManifestRow "usrp" $matrix $mod $null $null $codedBits $decoder "SKIP_UNSUPPORTED" $matrix.UnsupportedReason "" $null $ofdm
                continue
            }
            if ($UsrpMaxCodeN -gt 0 -and [int]$matrix.N -gt $UsrpMaxCodeN) {
                Add-ManifestRow "usrp" $matrix $mod $null $null $codedBits $decoder "SKIP_USRP_MAX_CODE_N" "USRP sweep skips code N>$UsrpMaxCodeN." "" $null $ofdm
                continue
            }
            if ($UsrpCodePattern -ne "" -and $matrix.Name -notmatch $UsrpCodePattern) {
                Add-ManifestRow "usrp" $matrix $mod $null $null $codedBits $decoder "SKIP_USRP_CODE_PATTERN" "USRP sweep code does not match pattern '$UsrpCodePattern'." "" $null $ofdm
                continue
            }
            if (($codedBits % [int]$matrix.N) -ne 0) {
                Add-ManifestRow "usrp" $matrix $mod $null $null $codedBits $decoder "SKIP_INCOMPATIBLE_OFDM" "coded_bits_per_frame is not divisible by code N with OFDM profile $($ofdm.Profile)." "" $null $ofdm
                continue
            }
            foreach ($variant in $usrpPipelineVariants) {
                if (-not $DryRun) {
                    Get-Process uhd_ldpc_ofdm_link -ErrorAction SilentlyContinue | Stop-Process -Force
                    Start-Sleep -Seconds 2
                }
                $caseName = New-SafeName ("{0}_{1}_{2}_{3}_usrp" -f $matrix.Name, $mod, $decoder, $variant.Name)
                $logFile = Join-Path $usrpDir ($caseName + ".log")
                $args = @(
                    "--mode", "trx",
                    "--args", $UsrpArgs,
                    "--alist", $matrix.Path,
                    "--decoder", $decoder,
                    "--modulation", $mod,
                    "--ldpc-iter", "$LdpcIter",
                    "--ldpc-normalization", ("{0:R}" -f $LdpcNormalization),
                    "--ldpc-offset", ("{0:R}" -f $LdpcOffset),
                    "--ldpc-damping", ("{0:R}" -f $LdpcDamping),
                    "--ldpc-schedule", "$LdpcSchedule",
                    "--duration", "$UsrpDurationSec",
                    "--freq", "$Freq",
                    "--tx-channel", "$TxChannel",
                    "--rx-channel", "$RxChannel",
                    "--tx-antenna", $TxAntenna,
                    "--rx-antenna", $RxAntenna,
                    "--tx-gain", "$TxGain",
                    "--rx-gain", "$RxGain",
                    "--rx-queue-blocks", "1024",
                    "--rx-frame-queue", "1024",
                    "--rx-block-samps", "32768"
                ) + $phyArgs + $variant.Args
                Show-CaseProgress -SuiteName "usrp" -Matrix $matrix -Modulation $mod -Decoder $decoder -EbN0Db $null -LogFile $logFile -PipelineName $variant.Name
                $exit = Invoke-LinkRun -ArgsList $args -LogFile $logFile -NoRun:$DryRun
                $status = if ($DryRun) { "DRYRUN" } elseif ($exit -eq 0) { "OK" } elseif ($exit -eq 2) { "DONE_WITH_FRAME_ERRORS" } else { "FAILED" }
                $reason = if ($exit -eq 2) { "Link executable returned 2 because FER/err was non-zero; this is retained as a measured USRP result." } else { "" }
                Add-ManifestRow "usrp" $matrix $mod $null $null $codedBits $decoder $status $reason $logFile $exit $ofdm $variant.Name
            }
        }
    }
}

$manifestPath = Join-Path $LogRoot "manifest.csv"
$manifest | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $manifestPath

function Get-OfflineLogMetric {
    param([object]$Row)
    if ($Row.suite -ne "offline_gpu" -or $Row.log -eq "" -or -not (Test-Path -LiteralPath $Row.log)) {
        return $null
    }
    $rawLog = (Get-Content -LiteralPath $Row.log -Raw) -replace "`0", ""
    $summaryMatches = [regex]::Matches(
        $rawLog,
        'frames=(\d+)\s+ok=(\d+)\s+err=(\d+)\s+FER=([0-9.eE+\-]+)\s+BER=([^ ]+).*?fps=([0-9.eE+\-]+)\s+goodput=([0-9.eE+\-]+)\s+Mbps')
    if ($summaryMatches.Count -eq 0) {
        return $null
    }
    $summary = $summaryMatches[$summaryMatches.Count - 1]
    $noise = [regex]::Match(
        $rawLog,
        '\[GPU-SIM\] CUDA frequency-domain AWGN enabled: snr=([0-9.eE+\-]+) dB noiseVar=([0-9.eE+\-]+)')
    $demod = [regex]::Match(
        $rawLog,
        '\[GPU-PIPE\] RX demod enabled:.*?demodNoiseVar=([0-9.eE+\-]+)')
    $fec = [regex]::Match(
        $rawLog,
        'decode\.fec_decode: ms/frame=([0-9.eE+\-]+) cap_fps=([0-9.eE+\-]+) cap_info=([0-9.eE+\-]+) Mbps')
    $sync = [regex]::Match(
        $rawLog,
        '\[PROFILE\] gpu_sync: found=(\d+) miss=(\d+) trackingMiss=(\d+) incomplete=(\d+)')

    [pscustomobject]@{
        code = $Row.code
        N = $Row.N
        K = $Row.K
        rate = $Row.rate
        modulation = $Row.modulation
        decoder = $Row.decoder
        ebn0_db = $Row.ebn0_db
        injected_snr_db = if ($noise.Success) { [double]$noise.Groups[1].Value } else { $Row.sim_snr_db }
        injected_noise_var = if ($noise.Success) { [double]$noise.Groups[2].Value } else { $null }
        llr_demod_noise_var = if ($demod.Success) { [double]$demod.Groups[1].Value } else { $null }
        frames = [int]$summary.Groups[1].Value
        ok = [int]$summary.Groups[2].Value
        err = [int]$summary.Groups[3].Value
        FER = [double]$summary.Groups[4].Value
        BER = $summary.Groups[5].Value
        fps = [double]$summary.Groups[6].Value
        goodput_mbps = [double]$summary.Groups[7].Value
        fec_ms_per_frame = if ($fec.Success) { [double]$fec.Groups[1].Value } else { $null }
        fec_cap_info_mbps = if ($fec.Success) { [double]$fec.Groups[3].Value } else { $null }
        sync_miss_total = if ($sync.Success) {
            [int]$sync.Groups[2].Value + [int]$sync.Groups[3].Value + [int]$sync.Groups[4].Value
        } else {
            $null
        }
        log = $Row.log
    }
}

$offlineMetrics = @($manifest | ForEach-Object { Get-OfflineLogMetric -Row $_ } | Where-Object { $null -ne $_ })
$offlineMetricsPath = Join-Path $LogRoot "offline_metrics.csv"
if ($offlineMetrics.Count -gt 0) {
    $offlineMetrics | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $offlineMetricsPath
}

function Get-UsrpLogMetric {
    param([object]$Row)
    if ($Row.suite -ne "usrp" -or $Row.log -eq "" -or -not (Test-Path -LiteralPath $Row.log)) {
        return $null
    }
    $logLines = @(Get-Content -LiteralPath $Row.log -Encoding Unicode)
    $summaryLines = @($logLines | Select-String -Pattern '^frames=\d+\s+ok=\d+\s+err=\d+\s+FER=')
    if ($summaryLines.Count -eq 0) {
        # Compatibility for logs written before the native output encoding was unified.
        $rawLog = (Get-Content -LiteralPath $Row.log -Raw) -replace "`0", ""
        $logLines = @($rawLog -split "`r?`n")
        $summaryLines = @($logLines | Select-String -Pattern '^frames=\d+\s+ok=\d+\s+err=\d+\s+FER=')
    }
    if ($summaryLines.Count -eq 0) {
        return $null
    }
    $summary = $summaryLines[$summaryLines.Count - 1].Line
    if ($summary -notmatch 'frames=(\d+)\s+ok=(\d+)\s+err=(\d+)\s+FER=([0-9.eE+\-]+).*?fps=([0-9.eE+\-]+)\s+goodput=([0-9.eE+\-]+)\s+Mbps') {
        return $null
    }
    $metricFrames = [int]$matches[1]
    $metricOk = [int]$matches[2]
    $metricErr = [int]$matches[3]
    $metricFer = [double]$matches[4]
    $metricFps = [double]$matches[5]
    $metricGoodput = [double]$matches[6]
    $quality = [regex]::Match(
        $summary,
        'avgSNR=([0-9.eE+\-]+)\s+dB\s+minSNR=([0-9.eE+\-]+)\s+dB(?:\s+avgAbsCFO=([0-9.eE+\-]+)\s+Hz\s+avgEVM=([^ ]+))?')
    $overflowWords = @($logLines | Select-String -Pattern 'overflow|overrun').Count
    $uhdOverflowMarkers = 0
    foreach ($line in $logLines) {
        if ($line -match '^U+') {
            $uhdOverflowMarkers += $matches[0].Length
        }
    }
    $syncLine = @($logLines | Select-String -Pattern '\[SYNC\].*dropped=') | Select-Object -Last 1
    $iqDropped = 0
    $gpuIqDropped = 0
    $frameDropped = 0
    $deviceFrameDropped = 0
    if ($null -ne $syncLine) {
        if ($syncLine.Line -match 'iq_buffer_dropped=(\d+)') { $iqDropped = [int]$matches[1] }
        if ($syncLine.Line -match 'gpu_iq_buffer_dropped=(\d+)') { $gpuIqDropped = [int]$matches[1] }
        if ($syncLine.Line -match 'frame_queue_dropped=(\d+)') { $frameDropped = [int]$matches[1] }
        if ($syncLine.Line -match 'device_frame_queue_dropped=(\d+)') { $deviceFrameDropped = [int]$matches[1] }
    }
    [pscustomobject]@{
        code = $Row.code
        N = $Row.N
        K = $Row.K
        rate = $Row.rate
        modulation = $Row.modulation
        decoder = $Row.decoder
        usrp_pipeline = $Row.usrp_pipeline
        freq_hz = $Freq
        sample_rate_sps = $Rate
        tx_gain_db = $TxGain
        rx_gain_db = $RxGain
        frames = $metricFrames
        ok = $metricOk
        err = $metricErr
        FER = $metricFer
        fps = $metricFps
        goodput_mbps = $metricGoodput
        avg_snr_db = if ($quality.Success) { [double]$quality.Groups[1].Value } else { $null }
        min_snr_db = if ($quality.Success) { [double]$quality.Groups[2].Value } else { $null }
        avg_abs_cfo_hz = if ($quality.Success -and $quality.Groups[3].Success) { [double]$quality.Groups[3].Value } else { $null }
        avg_evm_rms = if ($quality.Success -and $quality.Groups[4].Success -and $quality.Groups[4].Value -ne "n/a") {
            [double]$quality.Groups[4].Value
        } else {
            $null
        }
        uhd_overflow_markers = $uhdOverflowMarkers
        overflow_word_events = $overflowWords
        iq_buffer_dropped = $iqDropped
        gpu_iq_buffer_dropped = $gpuIqDropped
        frame_queue_dropped = $frameDropped
        device_frame_queue_dropped = $deviceFrameDropped
        log = $Row.log
    }
}

$usrpMetrics = @($manifest | ForEach-Object { Get-UsrpLogMetric -Row $_ } | Where-Object { $null -ne $_ })
$usrpMetricsPath = Join-Path $LogRoot "usrp_metrics.csv"
if ($usrpMetrics.Count -gt 0) {
    $usrpMetrics | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $usrpMetricsPath
}

$summaryPath = Join-Path $LogRoot "README.txt"
@(
    "Codec matrix sweep",
    "LogRoot: $LogRoot",
    "Suite: $Suite",
    "DryRun: $DryRun",
    "ThroughputOnly: $ThroughputOnly",
    "Resume completed offline logs: $Resume",
    "Max frame errors per offline case: $MaxFrameErrors",
    "Zero-error observation frames per offline case: $ZeroErrorStopFrames",
    "USRP max code N: $UsrpMaxCodeN",
    "USRP code pattern: $UsrpCodePattern",
    "USRP pipeline mode: $UsrpPipelineMode",
    "USRP pipeline variants: $($usrpPipelineVariants.Name -join ', ')",
    "OFDM profile: $OfdmProfile",
    "Custom OFDM: nfft=$Nfft cp=$Cp symbols=$NumSymbols activeSC=$ActiveSc pilotPeriod=$PilotPeriod rate=$Rate",
    "Auto profiles: SmallLdpc => activeSC=512 symbols=20 dataSymbols=15; DvbS2Universal => activeSC=720 symbols=120 dataSymbols=90.",
    "Channel profile: $ChannelProfile",
    "Precomp channel: $PrecompChannelFiles",
    "Actual channel: $ActualChannelFiles",
    "Offline Eb/N0 dB: $($EbN0DbList -join ', ')",
    "Modulations: $($Modulations -join ', ')",
    "Offline code pattern: $OfflineCodePattern",
    "GPU BP parameters: iter=$LdpcIter normalization=$LdpcNormalization offset=$LdpcOffset damping=$LdpcDamping schedule=$LdpcSchedule",
    "Manifest: $manifestPath",
    "Offline metrics: $offlineMetricsPath",
    "USRP metrics: $usrpMetricsPath",
    "",
    "Notes:",
    "- Offline --sim-snr-db is set from Eb/N0 using Es/N0 = Eb/N0 + 10log10(code_rate * bits_per_symbol).",
    "- GPU-SIM now passes its injected frequency-domain AWGN variance into GPU demod/LLR; compare injected_noise_var and llr_demod_noise_var in offline_metrics.csv.",
    "- offline_metrics.csv is the plotting input for both FER versus Eb/N0 and FER versus actual injected SNR.",
    "- USRP sweep skips LDPC/Polar entries whose N is larger than UsrpMaxCodeN. Set -UsrpMaxCodeN 0 to disable this limit.",
    "- UsrpPipelineMode Compare measures baseline CPU sync/demod plus GPU FEC, isolated GPU RX sync/demod plus buffered RX, and full GPU TX/RX pipeline plus buffered RX.",
    "- usrp_metrics.csv extracts final FER, goodput, fps, UHD U overflow markers, and host/device queue dropped counters for each completed USRP case.",
    "- ChannelProfile Perfect uses the same channel file for TX precompensation and actual injection.",
    "- ChannelProfile Offset1e2 uses predicted_base_3path.channel for precompensation and actual_offset_1e2_3path.channel for actual injection.",
    "- Polar .txt files are enumerated but skipped because the current link executable only connects LDPC --alist to PHY TX/RX.",
    "- Exit code 2 from the link executable is recorded as DONE_WITH_FRAME_ERRORS, not a script failure, because low Eb/N0 can legitimately produce FER > 0.",
    "- OFDM combinations whose coded bits per frame are not divisible by code N are skipped."
) | Out-File -LiteralPath $summaryPath -Encoding UTF8

Write-Host "Sweep complete."
Write-Host "Logs: $LogRoot"
Write-Host "Manifest: $manifestPath"
if ($offlineMetrics.Count -gt 0) {
    Write-Host "Offline metrics: $offlineMetricsPath"
}
if ($usrpMetrics.Count -gt 0) {
    Write-Host "USRP metrics: $usrpMetricsPath"
}
if ($GeneratePlots -and $offlineMetrics.Count -gt 0 -and -not $DryRun) {
    if (-not (Test-Path -LiteralPath $MatlabExe)) {
        throw "MATLAB executable not found for -GeneratePlots: $MatlabExe"
    }
    $matlabDir = Join-Path $RepoRoot "uhd_cpp\matlab"
    $figureDir = Join-Path $LogRoot "figures"
    $matlabDirArg = ($matlabDir -replace '\\', '/').Replace("'", "''")
    $metricsArg = ($offlineMetricsPath -replace '\\', '/').Replace("'", "''")
    $figureDirArg = ($figureDir -replace '\\', '/').Replace("'", "''")
    $matlabCommand = "addpath('$matlabDirArg'); plot_codec_sweep_fer_axes('$metricsArg', 'SaveDir', '$figureDirArg', 'Visible', 'off');"
    Write-Host "Generating FER plots with MATLAB -> $figureDir"
    & $MatlabExe -batch $matlabCommand
    if ($LASTEXITCODE -ne 0) {
        throw "MATLAB plot generation failed with exit code $LASTEXITCODE."
    }
}
Write-Progress -Activity "Codec matrix sweep" -Completed

