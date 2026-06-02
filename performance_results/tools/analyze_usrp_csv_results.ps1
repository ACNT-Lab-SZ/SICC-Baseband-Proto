param(
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [string]$OutDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"

$tablesDir = Join-Path $OutDir "tables"
$figuresDir = Join-Path $OutDir "figures"
foreach ($dir in @($tablesDir, $figuresDir)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

function To-Double($Value) {
    if ($null -eq $Value -or $Value -eq "" -or $Value -eq "n/a") { return $null }
    return [double]$Value
}

function Stats($Values) {
    $vals = @($Values | Where-Object { $null -ne $_ })
    if ($vals.Count -eq 0) {
        return [pscustomobject]@{ Count = 0; Mean = $null; Min = $null; Max = $null; Std = $null }
    }
    $avg = ($vals | Measure-Object -Average).Average
    $min = ($vals | Measure-Object -Minimum).Minimum
    $max = ($vals | Measure-Object -Maximum).Maximum
    $var = 0.0
    foreach ($v in $vals) { $var += [math]::Pow($v - $avg, 2) }
    $std = [math]::Sqrt($var / [math]::Max($vals.Count, 1))
    return [pscustomobject]@{ Count = $vals.Count; Mean = $avg; Min = $min; Max = $max; Std = $std }
}

$rxRoot = Join-Path $Root "data\datasets\transmission_file\Rx"
$offlineRows = New-Object System.Collections.Generic.List[object]
$traceRows = New-Object System.Collections.Generic.List[object]

$offlineDirs = Get-ChildItem -LiteralPath $rxRoot -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "ui_offline_gpu_*" } |
    Sort-Object Name

$runIndex = 0
foreach ($dir in $offlineDirs) {
    $snrFile = Join-Path $dir.FullName "simulated_snr_trace_db.csv"
    $gainFile = Join-Path $dir.FullName "simulated_channel_gain_trace_db.csv"
    $predFile = Join-Path $dir.FullName "predicted_relative_power_trace_db.csv"
    if ((Test-Path -LiteralPath $snrFile) -and (Test-Path -LiteralPath $gainFile)) {
        $runIndex++
        $snrRows = Import-Csv -LiteralPath $snrFile
        $gainRows = Import-Csv -LiteralPath $gainFile
        $snrVals = @($snrRows | ForEach-Object { To-Double $_.snrDb })
        $gainVals = @($gainRows | ForEach-Object { To-Double $_.relativePowerDb })
        $nominalVals = @()
        $n = [math]::Min($snrVals.Count, $gainVals.Count)
        for ($i = 0; $i -lt $n; $i++) {
            if ($null -ne $snrVals[$i] -and $null -ne $gainVals[$i]) {
                $nominalVals += ($snrVals[$i] - $gainVals[$i])
                $traceRows.Add([pscustomobject]@{
                    run_index = $runIndex
                    run_id = $dir.Name
                    frame = [int]$snrRows[$i].frame
                    snr_db = $snrVals[$i]
                    channel_gain_db = $gainVals[$i]
                    nominal_snr_db = $snrVals[$i] - $gainVals[$i]
                })
            }
        }
        $snrStats = Stats $snrVals
        $gainStats = Stats $gainVals
        $nominalStats = Stats $nominalVals
        $offlineRows.Add([pscustomobject]@{
            run_index = $runIndex
            run_id = $dir.Name
            mode = "offline_gpu_simulated_channel"
            samples = $snrStats.Count
            nominal_snr_db = $nominalStats.Mean
            snr_mean_db = $snrStats.Mean
            snr_min_db = $snrStats.Min
            snr_max_db = $snrStats.Max
            snr_std_db = $snrStats.Std
            channel_gain_mean_db = $gainStats.Mean
            channel_gain_min_db = $gainStats.Min
            channel_gain_max_db = $gainStats.Max
            channel_gain_std_db = $gainStats.Std
            source_snr = $snrFile.Replace($Root + "\", "")
            source_gain = $gainFile.Replace($Root + "\", "")
        })
    } elseif (Test-Path -LiteralPath $predFile) {
        $runIndex++
        $predRows = Import-Csv -LiteralPath $predFile
        $predVals = @($predRows | ForEach-Object { To-Double $_.relativePowerDb })
        $predStats = Stats $predVals
        $offlineRows.Add([pscustomobject]@{
            run_index = $runIndex
            run_id = $dir.Name
            mode = "offline_gpu_predicted_power"
            samples = $predStats.Count
            nominal_snr_db = $null
            snr_mean_db = $null
            snr_min_db = $null
            snr_max_db = $null
            snr_std_db = $null
            channel_gain_mean_db = $predStats.Mean
            channel_gain_min_db = $predStats.Min
            channel_gain_max_db = $predStats.Max
            channel_gain_std_db = $predStats.Std
            source_snr = ""
            source_gain = $predFile.Replace($Root + "\", "")
        })
    }
}

$usrpMetricsPath = Join-Path $rxRoot "ui_usrp_link_20260601_175341\usrp_gain_frame_metrics.csv"
$usrpRows = @()
if (Test-Path -LiteralPath $usrpMetricsPath) {
    $usrpRows = Import-Csv -LiteralPath $usrpMetricsPath | ForEach-Object {
        [pscustomobject]@{
            run = "ui_usrp_link_20260601_175341"
            radio_mode = "DualGated"
            tx_gain_db = To-Double $_.tx_gain_db
            rx_gain_db = To-Double $_.rx_gain_db
            total_gain_db = (To-Double $_.tx_gain_db) + (To-Double $_.rx_gain_db)
            sample_rate_msps = (To-Double $_.rate_sps) / 1e6
            num_symbols = [int]$_.num_symbols
            data_symbols = [int]$_.data_symbols
            active_sc = [int]$_.active_sc
            coded_bits_per_frame = [int]$_.coded_bits_per_frame
            frame_samples = [int]$_.frame_samples
            code = $_.code
            code_n = [int]$_.N
            code_k = [int]$_.K
            code_rate = (To-Double $_.K) / (To-Double $_.N)
            modulation = $_.modulation
            pipeline = $_.pipeline
            frames = [int]$_.frames
            ok = [int]$_.ok
            err = [int]$_.err
            fer = To-Double $_.FER
            ber = $_.BER
            fps = To-Double $_.fps
            goodput_mbps = To-Double $_.goodput_mbps
            decode_ms_per_frame = To-Double $_.decode_total_ms_per_frame
            fec_ms_per_frame = To-Double $_.fec_ms_per_frame
            sync_found = [int]$_.sync_found
            sync_miss = [int]$_.sync_miss
            tracking_miss = [int]$_.tracking_miss
            sync_incomplete = [int]$_.sync_incomplete
            tx_underflow_markers = [int]$_.tx_underflow_markers
            uhd_overflow_markers = [int]$_.uhd_overflow_markers
            status = $_.status
            source = $usrpMetricsPath.Replace($Root + "\", "")
        }
    }
}

$decoderPath = Join-Path $Root "data\datasets\portable\usrp_video_link_portable_20260513\logs\decoder_only_awgn\osd-only_3_3dB.csv"
$decoderRows = @()
if (Test-Path -LiteralPath $decoderPath) {
    $decoderRows = Import-Csv -LiteralPath $decoderPath | ForEach-Object {
        [pscustomobject]@{
            decoder = "osd-only"
            snr_db = To-Double $_.snr_db
            codewords = [int]$_.codewords
            bit_errors = [int]$_.bit_errors
            frame_errors = [int]$_.frame_errors
            ber = To-Double $_.BER
            fer = To-Double $_.FER
            info_mbps = To-Double $_.info_Mbps
            coded_mbps = To-Double $_.coded_Mbps
            cw_per_s = To-Double $_.cw_per_s
            used_osd = [int]$_.used_osd
            decode_seconds = To-Double $_.decode_seconds
            source = $decoderPath.Replace($Root + "\", "")
        }
    }
}

$configurationRows = @()
$configurationRows += [pscustomobject]@{
    scenario = "Offline GPU simulated channel"
    configuration_method = "Each run stores a per-frame channel gain trace and SNR trace. nominal_snr_db is inferred as snrDb - relativePowerDb."
    csv_files = 118
    runs = ($offlineRows | Where-Object { $_.mode -eq "offline_gpu_simulated_channel" }).Count
    key_parameters = "frame, relativePowerDb, snrDb, inferred nominal_snr_db"
}
$configurationRows += [pscustomobject]@{
    scenario = "Offline GPU predicted power"
    configuration_method = "Prediction-only relative power traces for early UI/offline GPU runs."
    csv_files = 4
    runs = ($offlineRows | Where-Object { $_.mode -eq "offline_gpu_predicted_power" }).Count
    key_parameters = "frame, relativePowerDb"
}
$configurationRows += [pscustomobject]@{
    scenario = "Real USRP gated GPU link"
    configuration_method = "Single measured radio profile uses TX/RX gains plus OFDM/FEC/modulation parameters in usrp_gain_frame_metrics.csv."
    csv_files = 2
    runs = $usrpRows.Count
    key_parameters = "tx_gain_db, rx_gain_db, sample_rate, num_symbols, active_sc, DVB-S2 rate 5/6, QPSK"
}
$configurationRows += [pscustomobject]@{
    scenario = "Decoder AWGN"
    configuration_method = "Decoder-only AWGN point uses fixed SNR and reports BER/FER/throughput."
    csv_files = 1
    runs = $decoderRows.Count
    key_parameters = "snr_db, codewords, BER, FER, info_Mbps, cw_per_s"
}

$offlineRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $tablesDir "usrp_offline_trace_summary.csv")
$traceRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $tablesDir "usrp_offline_trace_long.csv")
$usrpRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $tablesDir "usrp_radio_configuration_performance.csv")
$decoderRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $tablesDir "decoder_awgn_configuration_performance.csv")
$configurationRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $tablesDir "usrp_transmission_configuration_methods.csv")

$simRows = @($offlineRows | Where-Object { $_.mode -eq "offline_gpu_simulated_channel" })
$predRows = @($offlineRows | Where-Object { $_.mode -eq "offline_gpu_predicted_power" })
$simStats = Stats (@($simRows | ForEach-Object { $_.snr_mean_db }))
$gainStats = Stats (@($simRows | ForEach-Object { $_.channel_gain_mean_db }))
$nominalStats = Stats (@($simRows | ForEach-Object { $_.nominal_snr_db }))

$md = New-Object System.Collections.Generic.List[string]
$md.Add("# USRP Transfer CSV Analysis")
$md.Add("")
$md.Add("## CSV inventory")
$md.Add("")
$md.Add("| Group | Runs | CSV files | Meaning |")
$md.Add("|---|---:|---:|---|")
$md.Add("| Offline simulated SNR/channel traces | $($simRows.Count) | 118 | 每组 run 包含 simulated_snr_trace_db.csv 与 simulated_channel_gain_trace_db.csv |")
$md.Add("| Offline predicted power traces | $($predRows.Count) | 4 | 早期预测相对功率 trace |")
$md.Add("| Real USRP profile/performance | $($usrpRows.Count) | 2 | manifest.csv + usrp_gain_frame_metrics.csv |")
$md.Add("| Decoder AWGN | $($decoderRows.Count) | 1 | decoder-only BER/FER/throughput |")
$md.Add("")
$md.Add("## Offline channel/SNR configuration summary")
$md.Add("")
$md.Add("| Metric | Value |")
$md.Add("|---|---:|")
$md.Add("| Simulated runs | $($simRows.Count) |")
$md.Add("| Mean nominal SNR (dB) | $([math]::Round($nominalStats.Mean, 3)) |")
$md.Add("| Mean run SNR (dB) | $([math]::Round($simStats.Mean, 3)) |")
$md.Add("| Mean channel gain (dB) | $([math]::Round($gainStats.Mean, 3)) |")
$md.Add("| Prediction-only runs | $($predRows.Count) |")
$md.Add("")
if ($usrpRows.Count -gt 0) {
    $u = $usrpRows[0]
    $md.Add("## Real USRP transfer profile")
    $md.Add("")
    $md.Add("| Parameter | Value |")
    $md.Add("|---|---:|")
    $md.Add("| TX gain (dB) | $($u.tx_gain_db) |")
    $md.Add("| RX gain (dB) | $($u.rx_gain_db) |")
    $md.Add("| Total gain setting (dB) | $($u.total_gain_db) |")
    $md.Add("| Sample rate (Msps) | $($u.sample_rate_msps) |")
    $md.Add("| OFDM symbols / data symbols | $($u.num_symbols) / $($u.data_symbols) |")
    $md.Add("| Active subcarriers | $($u.active_sc) |")
    $md.Add("| FEC | $($u.code), N=$($u.code_n), K=$($u.code_k), rate=$([math]::Round($u.code_rate, 4)) |")
    $md.Add("| Modulation | $($u.modulation) |")
    $md.Add("| Frames / OK / ERR | $($u.frames) / $($u.ok) / $($u.err) |")
    $md.Add("| FER | $($u.fer) |")
    $md.Add("| Goodput (Mbps) | $($u.goodput_mbps) |")
    $md.Add("| FPS | $($u.fps) |")
    $md.Add("| Decode ms/frame | $($u.decode_ms_per_frame) |")
    $md.Add("| Status | $($u.status) |")
    $md.Add("")
}
if ($decoderRows.Count -gt 0) {
    $d = $decoderRows[0]
    $md.Add("## Decoder AWGN reference point")
    $md.Add("")
    $md.Add("| SNR (dB) | BER | FER | Info Mbps | Codeword/s |")
    $md.Add("|---:|---:|---:|---:|---:|")
    $md.Add("| $($d.snr_db) | $($d.ber) | $($d.fer) | $($d.info_mbps) | $($d.cw_per_s) |")
}
$md | Set-Content -Encoding UTF8 -Path (Join-Path $tablesDir "usrp_transfer_csv_analysis.md")
