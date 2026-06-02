param(
    [string]$RepoRoot = $(if ($env:PROJECT_ROOT) { $env:PROJECT_ROOT } else { (Get-Location).Path }),
    [string]$Config = "Release",
    [int]$Frames = 6000,
    [double]$BaseSnrDb = 24.0,
    [double]$FadeSnrDb = 4.0,
    [int]$FadeStartFrame = 352,
    [int]$FadeEndFrame = 641,
    [double]$JitterDb = 2.0,
    [int]$Seed = 20260531,
    [string]$LogRoot = "",
    [switch]$NoPlot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($LogRoot)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogRoot = Join-Path $RepoRoot "build\uhd_cpp_gpu_pipeline\logs\predictive_power_amc_$stamp"
}
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

$exe = Join-Path $RepoRoot "build\uhd_cpp_gpu_pipeline\$Config\uhd_ldpc_ofdm_link.exe"
if (-not (Test-Path $exe)) {
    throw "Executable not found: $exe"
}

$UhdRoot = $(if ($env:UHD_ROOT) { $env:UHD_ROOT } elseif ($env:UHD_PKG_PATH) { $env:UHD_PKG_PATH } else { "" })
$TorchLib = $(if ($env:TORCH_LIB_DIR) { $env:TORCH_LIB_DIR } else { "" })
$CudaBin = $(if ($env:CUDA_PATH) { Join-Path $env:CUDA_PATH "bin" } else { "" })
$CudaOsdDllDir = $(if ($env:CUDA_OSD_DLL_DIR) { $env:CUDA_OSD_DLL_DIR } elseif ($env:CUDA_OSD_ROOT) { Join-Path $env:CUDA_OSD_ROOT "build\vs2022\Release" } else { "" })
$env:UHD_IMAGES_DIR = Join-Path $UhdRoot "share\uhd\images"
$env:UHD_PKG_PATH = $UhdRoot
$env:UHD_RFNOC_DIR = Join-Path $UhdRoot "share\uhd\rfnoc"
$env:PATH = "$TorchLib;$(Split-Path $exe -Parent);$(Join-Path $UhdRoot 'bin');$CudaBin;$CudaOsdDllDir;$env:PATH"

$rng = [System.Random]::new($Seed)
$snrTrace = New-Object double[] $Frames
$powerTrace = New-Object double[] $Frames
for ($i = 0; $i -lt $Frames; $i++) {
    $frame = $i + 1
    if ($frame -ge $FadeStartFrame -and $frame -le $FadeEndFrame) {
        $snrTrace[$i] = $FadeSnrDb
    } else {
        $segLen = [Math]::Max(1, [Math]::Ceiling($Frames / 5.0))
        $seg = [Math]::Min(4, [int][Math]::Floor($i / $segLen))
        $baseTrace = @(8.0, 14.5, 18.0, 10.0, 15.0)[$seg]
        $snrTrace[$i] = $baseTrace + (($rng.NextDouble() * 2.0 - 1.0) * $JitterDb)
    }
    $powerTrace[$i] = $snrTrace[$i] - $BaseSnrDb
}
$powerTraceFile = Join-Path $LogRoot "simulated_channel_gain_trace_db.csv"
$snrTraceFile = Join-Path $LogRoot "actual_snr_trace_db.csv"
"frame,relativePowerDb" | Set-Content -Encoding UTF8 $powerTraceFile
"frame,snrDb" | Set-Content -Encoding UTF8 $snrTraceFile
for ($i = 0; $i -lt $Frames; $i++) {
    "{0},{1:F6}" -f ($i + 1), $powerTrace[$i] | Add-Content -Encoding UTF8 $powerTraceFile
    "{0},{1:F6}" -f ($i + 1), $snrTrace[$i] | Add-Content -Encoding UTF8 $snrTraceFile
}

$common = @(
    "--mode", "gpu-sim",
    "--traffic", "test",
    "--frames", "$Frames",
    "--nfft", "1024",
    "--cp", "128",
    "--num-symbols", "120",
    "--active-sc", "720",
    "--pilot-period", "4",
    "--rate", "25e6",
    "--alist", (Join-Path $RepoRoot "Code_Matrices_Lib\LDPC\DVB_S2_short_N16200_rate_5_6.alist"),
    "--decoder", "cuda-bp",
    "--ldpc-iter", "50",
    "--ldpc-normalization", "0.95",
    "--ldpc-offset", "0",
    "--ldpc-damping", "0",
    "--ldpc-schedule", "2",
    "--skip-reference-ber"
)

$cases = @(
    @{ Name = "fixed_qpsk"; Args = @("--modulation", "qpsk", "--sim-snr-trace", $snrTraceFile) },
    @{ Name = "fixed_16qam"; Args = @("--modulation", "16qam", "--sim-snr-trace", $snrTraceFile) },
    @{ Name = "predictive_amc"; Args = @(
        "--modulation", "qpsk",
        "--adaptive-power-trace", $powerTraceFile,
        "--adaptive-power-base-snr-db", "$BaseSnrDb",
        "--adaptive-enable-16qam",
        "--tx-repeat-min", "1",
        "--tx-repeat-max", "1",
        "--adaptive-power-hold-frames", "1"
    ) }
)

foreach ($case in $cases) {
    $log = Join-Path $LogRoot ($case.Name + ".log")
    Write-Host "[AMC-DEMO] Running $($case.Name) -> $log"
    $argsList = @() + $common + $case.Args
    & $exe @argsList 2>&1 | Tee-Object -FilePath $log | Out-Host
    $exit = $LASTEXITCODE
    if ($exit -ne 0 -and $exit -ne 2) {
        throw "Case $($case.Name) failed with exit code $exit"
    }
}

function Parse-CaseLog {
    param([string]$CaseName, [string]$LogPath)
    $lines = Get-Content -LiteralPath $LogPath
    $phyLine = ($lines | Select-String -Pattern '^PHY:' | Select-Object -First 1).Line
    $frameSamples = 1
    $rate = 1.0
    if ($phyLine -match 'frameSamples=(\d+)\s+rate=([0-9.]+)\s+Msps') {
        $frameSamples = [int]$matches[1]
        $rate = [double]$matches[2] * 1e6
    }
    $infoByMod = @{}
    $profileLine = ($lines | Select-String -Pattern 'profiles=' | Select-Object -First 1).Line
    if ($profileLine) {
        foreach ($m in [regex]::Matches($profileLine, '([A-Za-z0-9]+)\((\d+)b\)')) {
            $infoByMod[$m.Groups[1].Value.ToLower()] = [double]$m.Groups[2].Value
        }
    }
    if ($infoByMod.Count -eq 0) {
        $ldpcLine = ($lines | Select-String -Pattern 'infoBits/frame=(\d+)' | Select-Object -First 1).Line
        if ($ldpcLine -match 'infoBits/frame=(\d+)') {
            $infoByMod["qpsk"] = [double]$matches[1]
            $infoByMod["16qam"] = [double]$matches[1]
            $infoByMod["bpsk"] = [double]$matches[1]
        }
    }

    $rows = New-Object System.Collections.Generic.List[object]
    $logical = 0
    $ok = 0
    $err = 0
    $okBits = 0.0
    $physical = 0
    foreach ($line in $lines) {
        if ($line -match '\[GPU-SIM\]\s+modulation=([A-Za-z0-9]+)\s+repeat=(\d+).*?snr=([0-9.\-]+)\s+dB.*?ok=(yes|no)') {
            $logical++
            $mod = $matches[1].ToLower()
            $rep = [int]$matches[2]
            $snr = [double]$matches[3]
            $isOk = $matches[4] -eq "yes"
            if ($isOk) { $ok++ } else { $err++ }
            $physical += [Math]::Max($rep, 1)
            $bits = if ($infoByMod.ContainsKey($mod)) { [double]$infoByMod[$mod] } else { 0.0 }
            if ($isOk) { $okBits += $bits }
            $linkSec = [Math]::Max(1e-9, $physical * $frameSamples / $rate)
            $rows.Add([pscustomobject]@{
                case = $CaseName
                logical_frame = $logical
                physical_frames = $physical
                modulation = $mod
                repeat = $rep
                snr_db = $snr
                ok = [int]$isOk
                cum_fer = $err / [Math]::Max($logical, 1)
                cum_goodput_mbps = $okBits / $linkSec / 1e6
            })
        }
    }
    return $rows
}

$curveRows = New-Object System.Collections.Generic.List[object]
foreach ($case in $cases) {
    $caseRows = Parse-CaseLog -CaseName $case.Name -LogPath (Join-Path $LogRoot ($case.Name + ".log"))
    foreach ($row in $caseRows) { $curveRows.Add($row) }
}
$curvesCsv = Join-Path $LogRoot "predictive_power_amc_curves.csv"
$curveRows | Export-Csv -NoTypeInformation -Encoding UTF8 $curvesCsv

$summaryRows = $curveRows | Group-Object case | ForEach-Object {
    $last = $_.Group | Select-Object -Last 1
    [pscustomobject]@{
        case = $_.Name
        frames = $_.Group.Count
        final_fer = $last.cum_fer
        final_goodput_mbps = $last.cum_goodput_mbps
        physical_frames = $last.physical_frames
        avg_snr_db = ($_.Group | Measure-Object snr_db -Average).Average
    }
}
$summaryCsv = Join-Path $LogRoot "predictive_power_amc_summary.csv"
$summaryRows | Export-Csv -NoTypeInformation -Encoding UTF8 $summaryCsv

if (-not $NoPlot) {
    $plotScript = Join-Path $RepoRoot "uhd_cpp\scripts\plot_predictive_power_amc_demo.py"
    $py = Join-Path $RepoRoot "UI_NEW\.venv312\Scripts\python.exe"
    if (-not (Test-Path $py)) { $py = "python" }
    & $py $plotScript --curves $curvesCsv --summary $summaryCsv --out-dir $LogRoot
}

Write-Host "[AMC-DEMO] Done."
Write-Host "  LogRoot : $LogRoot"
Write-Host "  Curves  : $curvesCsv"
Write-Host "  Summary : $summaryCsv"
Write-Host "  Power   : $powerTraceFile"
Write-Host "  SNR     : $snrTraceFile"

