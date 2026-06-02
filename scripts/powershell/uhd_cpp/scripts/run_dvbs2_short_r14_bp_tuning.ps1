param(
    [string]$RepoRoot = $(if ($env:PROJECT_ROOT) { $env:PROJECT_ROOT } else { (Get-Location).Path }),
    [string]$Exe = "",
    [string]$Alist = "",
    [string]$LogRoot = "",
    [double[]]$EbN0DbList = @(4),
    [int[]]$Iterations = @(8, 20, 50),
    [double[]]$Normalizations = @(0.60, 0.70, 0.80, 0.90),
    [double[]]$Offsets = @(0.15),
    [double[]]$Dampings = @(0.15),
    [int[]]$Schedules = @(2),
    [int]$Frames = 1000,
    [int]$TestSeed = 12345,
    [string]$TorchLib = $(if ($env:TORCH_LIB_DIR) { $env:TORCH_LIB_DIR } else { "" }),
    [string]$CudaOsdDllDir = $(if ($env:CUDA_OSD_DLL_DIR) { $env:CUDA_OSD_DLL_DIR } elseif ($env:CUDA_OSD_ROOT) { Join-Path $env:CUDA_OSD_ROOT "build\vs2022\Release" } else { "" }),
    [string]$CudaBin = $(if ($env:CUDA_PATH) { Join-Path $env:CUDA_PATH "bin" } else { "" })
)

$ErrorActionPreference = "Stop"

if ($Exe -eq "") {
    $Exe = Join-Path $RepoRoot "build\uhd_cpp_gpu_pipeline\Release\uhd_ldpc_ofdm_link.exe"
}
if ($Alist -eq "") {
    $Alist = Join-Path $RepoRoot "Code_Matrices_Lib\LDPC\DVB_S2_short_N16200_rate_1_4.alist"
}
if ($LogRoot -eq "") {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogRoot = Join-Path $RepoRoot "build\uhd_cpp_gpu_pipeline\logs\dvbs2_r14_bp_tuning_$stamp"
}
if (-not (Test-Path -LiteralPath $Exe)) {
    throw "Executable not found: $Exe"
}
if (-not (Test-Path -LiteralPath $Alist)) {
    throw "DVB-S2 short 1/4 alist not found: $Alist"
}

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
$env:PATH = "$TorchLib;$CudaOsdDllDir;$CudaBin;$env:PATH"

$codeRate = 3240.0 / 16200.0
$bitsPerSymbol = 2.0
$total = $EbN0DbList.Count * $Iterations.Count * $Normalizations.Count * $Offsets.Count * $Dampings.Count * $Schedules.Count
$index = 0
$rows = New-Object System.Collections.Generic.List[object]

foreach ($ebn0 in $EbN0DbList) {
    $simSnr = $ebn0 + 10.0 * [Math]::Log10($codeRate * $bitsPerSymbol)
    foreach ($iter in $Iterations) {
        foreach ($normalization in $Normalizations) {
          foreach ($offset in $Offsets) {
           foreach ($damping in $Dampings) {
            foreach ($schedule in $Schedules) {
            $index += 1
            $status = "Eb/N0=$('{0:g}' -f $ebn0)dB iter=$iter norm=$('{0:0.00}' -f $normalization) offset=$('{0:0.00}' -f $offset) damping=$('{0:0.00}' -f $damping) schedule=$schedule"
            Write-Progress -Activity "DVB-S2 short 1/4 GPU BP tuning" -Status $status -PercentComplete ([Math]::Round(100.0 * $index / $total, 1))
            Write-Host ("[Tune {0}/{1}] {2}" -f $index, $total, $status)

            $name = "r14_qpsk_EbN0_{0:0.0}_iter_{1}_nms_{2:0.00}_off_{3:0.00}_damp_{4:0.00}_sched_{5}" -f $ebn0, $iter, $normalization, $offset, $damping, $schedule
            $name = $name -replace '\.', 'p'
            $logFile = Join-Path $LogRoot ($name + ".log")
            $args = @(
                "--mode", "gpu-sim",
                "--alist", $Alist,
                "--decoder", "cuda-bp",
                "--modulation", "qpsk",
                "--frames", "$Frames",
                "--sim-snr-db", ("{0:R}" -f $simSnr),
                "--ldpc-iter", "$iter",
                "--ldpc-normalization", ("{0:R}" -f $normalization),
                "--ldpc-offset", ("{0:R}" -f $offset),
                "--ldpc-damping", ("{0:R}" -f $damping),
                "--ldpc-schedule", "$schedule",
                "--traffic", "test",
                "--systematic-front-info",
                "--nfft", "1024",
                "--cp", "72",
                "--num-symbols", "120",
                "--active-sc", "720",
                "--pilot-period", "4",
                "--sync-preamble", "zc-ofdm",
                "--sync-zc-root", "29",
                "--rx-channel-est", "pilot",
                "--rate", "25000000",
                "--mcr", "200000000",
                "--profile-pipeline",
                "--suppress-error-frames",
                "--test-seed", "$TestSeed"
            )
            $command = "& `"$Exe`" " + (($args | ForEach-Object {
                if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
            }) -join " ")
            $command | Out-File -LiteralPath $logFile -Encoding Unicode
            "" | Out-File -LiteralPath $logFile -Append -Encoding Unicode
            $output = & $Exe @args 2>&1 | ForEach-Object {
                "$_" | Out-File -LiteralPath $logFile -Append -Encoding Unicode
                "$_"
            }
            $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
            $text = ($output | Out-String)

            $summary = [regex]::Matches(
                $text,
                'frames=(\d+) ok=(\d+) err=(\d+) FER=([0-9.eE+\-]+) BER=([^ ]+).*?fps=([0-9.]+) goodput=([0-9.]+) Mbps.*?avgIter=([0-9.]+)',
                [System.Text.RegularExpressions.RegexOptions]::Singleline)
            $bench = [regex]::Matches(
                $text,
                'decode\.fec_decode: ms/frame=([0-9.]+) cap_fps=([0-9.]+) cap_info=([0-9.]+) Mbps')
            if ($summary.Count -eq 0) {
                throw "Could not parse RX summary from $logFile"
            }
            $s = $summary[$summary.Count - 1].Groups
            $decodeMs = $null
            $capacity = $null
            if ($bench.Count -gt 0) {
                $b = $bench[$bench.Count - 1].Groups
                $decodeMs = [double]$b[1].Value
                $capacity = [double]$b[3].Value
            }
            $rows.Add([pscustomobject]@{
                ebn0_db = $ebn0
                sim_snr_db = $simSnr
                iterations = $iter
                normalization = $normalization
                offset = $offset
                damping = $damping
                schedule = $schedule
                frames = [int]$s[1].Value
                ok = [int]$s[2].Value
                err = [int]$s[3].Value
                FER = [double]$s[4].Value
                BER = $s[5].Value
                fps = [double]$s[6].Value
                goodput_mbps = [double]$s[7].Value
                avg_iterations = [double]$s[8].Value
                decode_ms_per_frame = $decodeMs
                fec_capacity_mbps = $capacity
                exit_code = $exitCode
                log = $logFile
            })
            }
           }
          }
        }
    }
}

Write-Progress -Activity "DVB-S2 short 1/4 GPU BP tuning" -Completed
$metricsPath = Join-Path $LogRoot "tuning_metrics.csv"
$rows | Sort-Object FER, @{Expression = "goodput_mbps"; Descending = $true} |
    Export-Csv -LiteralPath $metricsPath -NoTypeInformation -Encoding UTF8
$rows | Sort-Object FER, @{Expression = "goodput_mbps"; Descending = $true} |
    Format-Table ebn0_db, iterations, normalization, offset, damping, schedule, FER, BER, goodput_mbps, decode_ms_per_frame, fec_capacity_mbps -AutoSize
Write-Host "Metrics: $metricsPath"

