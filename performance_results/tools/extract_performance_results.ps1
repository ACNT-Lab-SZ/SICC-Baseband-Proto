param(
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [string]$OutDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"

$tablesDir = Join-Path $OutDir "tables"
$rawDir = Join-Path $OutDir "raw"
$figuresDir = Join-Path $OutDir "figures"
foreach ($dir in @($tablesDir, $rawDir, $figuresDir)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

function Copy-ResultFile([string]$Source, [string]$Group) {
    if (-not (Test-Path -LiteralPath $Source)) { return $null }
    $destDir = Join-Path $rawDir $Group
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null
    $relative = $Source.Replace($Root + "\", "")
    $safeName = $relative -replace "[:\\\/]+", "__"
    $dest = Join-Path $destDir $safeName
    Copy-Item -LiteralPath $Source -Destination $dest -Force
    return $dest
}

function Format-Num($Value, [int]$Digits = 4) {
    if ($null -eq $Value -or $Value -eq "") { return "n/a" }
    $num = 0.0
    if ([double]::TryParse([string]$Value, [ref]$num)) {
        return ([math]::Round($num, $Digits)).ToString()
    }
    return [string]$Value
}

function Add-YoloSummary([string]$Path, [string]$Task) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    Copy-ResultFile $Path "yolo_training" | Out-Null
    $rows = Import-Csv -LiteralPath $Path
    if ($rows.Count -eq 0) { return $null }
    $bestMap50 = $rows | Sort-Object {[double]$_.'metrics/mAP50(B)'} -Descending | Select-Object -First 1
    $bestMap5095 = $rows | Sort-Object {[double]$_.'metrics/mAP50-95(B)'} -Descending | Select-Object -First 1
    $last = $rows[-1]
    return [pscustomobject]@{
        task = $Task
        epochs = $rows.Count
        best_map50 = [double]$bestMap50.'metrics/mAP50(B)'
        best_map50_epoch = [int]$bestMap50.epoch
        best_map50_95 = [double]$bestMap5095.'metrics/mAP50-95(B)'
        best_map50_95_epoch = [int]$bestMap5095.epoch
        final_precision = [double]$last.'metrics/precision(B)'
        final_recall = [double]$last.'metrics/recall(B)'
        final_map50 = [double]$last.'metrics/mAP50(B)'
        final_map50_95 = [double]$last.'metrics/mAP50-95(B)'
        final_train_box_loss = [double]$last.'train/box_loss'
        final_val_box_loss = [double]$last.'val/box_loss'
        source = $Path.Replace($Root + "\", "")
    }
}

function Add-RoundtripSummary([string]$Path, [string]$Task) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    Copy-ResultFile $Path "gpu_direct_roundtrip" | Out-Null
    $json = Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
    $summary = $json.summary
    if ($null -eq $summary) { return $null }
    $detections = 0
    $payloadMin = $null
    $payloadMax = $null
    if ($json.frames) {
        foreach ($frame in $json.frames) {
            if ($frame.detections) { $detections += @($frame.detections).Count }
            $payload = [double]$frame.payload_nbytes
            if ($null -eq $payloadMin -or $payload -lt $payloadMin) { $payloadMin = $payload }
            if ($null -eq $payloadMax -or $payload -gt $payloadMax) { $payloadMax = $payload }
        }
    }
    return [pscustomobject]@{
        task = $Task
        scheme = $summary.scheme
        frames = [int]$summary.frames
        fps = [double]$summary.fps
        avg_payload_kb = [double]$summary.avg_payload_nbytes / 1024.0
        min_payload_kb = if ($null -eq $payloadMin) { $null } else { $payloadMin / 1024.0 }
        max_payload_kb = if ($null -eq $payloadMax) { $null } else { $payloadMax / 1024.0 }
        detections = $detections
        source = $Path.Replace($Root + "\", "")
    }
}

function Add-V2TwoSchemeSummary([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return @() }
    Copy-ResultFile $Path "gpu_direct_roundtrip" | Out-Null
    $json = Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
    $rows = @()
    foreach ($scheme in @("uniform", "roi-layered")) {
        $s = $json.$scheme
        $rows += [pscustomobject]@{
            task = "VISO vehicle split inference"
            scheme = $scheme
            frames = [int]$s.frames
            fps = [double]$s.fps
            avg_payload_kb = [double]$s.avg_payload_nbytes / 1024.0
            min_payload_kb = $null
            max_payload_kb = $null
            detections = [int]$s.detections
            source = $Path.Replace($Root + "\", "")
        }
    }
    return $rows
}

function Add-UiMetricSummary([string]$Path, [string]$Name) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    Copy-ResultFile $Path "ui_metrics" | Out-Null
    $json = Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
    $eval = $json.evaluation
    $link = $json.link_payload
    return [pscustomobject]@{
        run = $Name
        frames = [int]$json.frames
        detections = [int]$json.detections
        avg_confidence = [double]$json.confidence.avg
        max_confidence = [double]$json.confidence.max
        tx_bytes = [double]$link.tx_bytes
        rx_bytes = [double]$link.rx_bytes
        byte_delta = [double]$link.byte_delta
        precision = if ($eval) { [double]$eval.precision } else { $null }
        recall = if ($eval) { [double]$eval.recall } else { $null }
        f1 = if ($eval) { [double]$eval.f1 } else { $null }
        source = $Path.Replace($Root + "\", "")
    }
}

function Add-LinkMetricSummary([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    Copy-ResultFile $Path "usrp_link" | Out-Null
    $rows = Import-Csv -LiteralPath $Path
    if ($rows.Count -eq 0) { return $null }
    $r = $rows[0]
    return [pscustomobject]@{
        run = "dual_usrp_gated_gpu_pipeline"
        tx_gain_db = [double]$r.tx_gain_db
        rx_gain_db = [double]$r.rx_gain_db
        frames = [int]$r.frames
        ok = [int]$r.ok
        err = [int]$r.err
        fer = [double]$r.FER
        ber = $r.BER
        fps = [double]$r.fps
        goodput_mbps = [double]$r.goodput_mbps
        decode_total_ms_per_frame = [double]$r.decode_total_ms_per_frame
        fec_ms_per_frame = [double]$r.fec_ms_per_frame
        status = $r.status
        source = $Path.Replace($Root + "\", "")
    }
}

function Add-AwgnSummary([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    Copy-ResultFile $Path "decoder_awgn" | Out-Null
    $rows = Import-Csv -LiteralPath $Path
    if ($rows.Count -eq 0) { return $null }
    $r = $rows[0]
    return [pscustomobject]@{
        decoder = "osd-only"
        snr_db = [double]$r.snr_db
        codewords = [int]$r.codewords
        bit_errors = [int]$r.bit_errors
        frame_errors = [int]$r.frame_errors
        ber = [double]$r.BER
        fer = [double]$r.FER
        info_mbps = [double]$r.info_Mbps
        coded_mbps = [double]$r.coded_Mbps
        cw_per_s = [double]$r.cw_per_s
        decode_seconds = [double]$r.decode_seconds
        source = $Path.Replace($Root + "\", "")
    }
}

function Add-PowerTraceSummary([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    Copy-ResultFile $Path "power_trace" | Out-Null
    $rows = Import-Csv -LiteralPath $Path
    if ($rows.Count -eq 0) { return $null }
    $numericColumn = ($rows[0].PSObject.Properties.Name | Select-Object -Last 1)
    $values = @($rows | ForEach-Object { [double]$_.$numericColumn })
    return [pscustomobject]@{
        run = Split-Path (Split-Path $Path -Parent) -Leaf
        samples = $values.Count
        mean_db = ($values | Measure-Object -Average).Average
        min_db = ($values | Measure-Object -Minimum).Minimum
        max_db = ($values | Measure-Object -Maximum).Maximum
        source = $Path.Replace($Root + "\", "")
    }
}

$yoloSummaries = @(
    Add-YoloSummary (Join-Path $Root "data\datasets\Ai_process_V3\tasks\earthquake\runs\results.csv") "earthquake damage detection"
    Add-YoloSummary (Join-Path $Root "data\datasets\Ai_process_V3\tasks\maritime_ship\runs\results.csv") "maritime ship detection"
) | Where-Object { $_ }

$roundtripSummaries = @()
$roundtripSummaries += Add-V2TwoSchemeSummary (Join-Path $Root "configs\Ai_process_V2\validation\gpu_direct_two_scheme_summary.json")
$roundtripSummaries += Add-RoundtripSummary (Join-Path $Root "configs\Ai_process_V3\tasks\earthquake\validation\gpu_direct_roundtrip.json") "earthquake"
$roundtripSummaries += Add-RoundtripSummary (Join-Path $Root "configs\Ai_process_V3\tasks\maritime_ship\validation\gpu_direct_roundtrip.json") "maritime_ship"
$roundtripSummaries += Add-RoundtripSummary (Join-Path $Root "configs\Ai_process_V3\tasks\flood\validation\roi_layered_roundtrip.json") "flood"
$roundtripSummaries += Add-RoundtripSummary (Join-Path $Root "configs\Ai_process_V3\tasks\flood\validation\uniform_roundtrip.json") "flood"
$roundtripSummaries = $roundtripSummaries | Where-Object { $_ }

$uiMetricSummaries = @()
$uiMetricFiles = Get-ChildItem -Recurse -Force -File (Join-Path $Root "configs\Ai_process_V2\logs\baseband") -Filter "ui_metrics.json" -ErrorAction SilentlyContinue
foreach ($file in $uiMetricFiles) {
    $uiMetricSummaries += Add-UiMetricSummary $file.FullName (Split-Path $file.DirectoryName -Leaf)
}

$linkSummaries = @(
    Add-LinkMetricSummary (Join-Path $Root "data\datasets\transmission_file\Rx\ui_usrp_link_20260601_175341\usrp_gain_frame_metrics.csv")
) | Where-Object { $_ }

$awgnSummaries = @(
    Add-AwgnSummary (Join-Path $Root "data\datasets\portable\usrp_video_link_portable_20260513\logs\decoder_only_awgn\osd-only_3_3dB.csv")
) | Where-Object { $_ }

$powerTraceSummaries = @()
$powerTraceFiles = Get-ChildItem -Recurse -Force -File (Join-Path $Root "data\datasets\transmission_file\Rx") -Filter "predicted_relative_power_trace_db.csv" -ErrorAction SilentlyContinue
foreach ($file in $powerTraceFiles) {
    $powerTraceSummaries += Add-PowerTraceSummary $file.FullName
}

$resultInventory = Get-ChildItem -Recurse -Force -File $rawDir | ForEach-Object {
    [pscustomobject]@{
        group = Split-Path $_.DirectoryName -Leaf
        file = $_.Name
        bytes = $_.Length
        path = $_.FullName.Replace($OutDir + "\", "")
    }
}

$yoloSummaries | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $tablesDir "yolo_training_summary.csv")
$roundtripSummaries | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $tablesDir "gpu_direct_roundtrip_summary.csv")
$uiMetricSummaries | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $tablesDir "ui_metric_summary.csv")
$linkSummaries | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $tablesDir "usrp_link_summary.csv")
$awgnSummaries | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $tablesDir "decoder_awgn_summary.csv")
$powerTraceSummaries | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $tablesDir "power_trace_summary.csv")
$resultInventory | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $tablesDir "result_file_inventory.csv")

$md = New-Object System.Collections.Generic.List[string]
$md.Add("# Performance Results Three-Line Tables")
$md.Add("")
$md.Add("Generated from organized result files under ``organized_workspace_20260601``.")
$md.Add("")
$md.Add("## YOLO Training")
$md.Add("")
$md.Add("| Task | Epochs | Best mAP50 | Best mAP50-95 | Final Precision | Final Recall |")
$md.Add("|---|---:|---:|---:|---:|---:|")
foreach ($r in $yoloSummaries) {
    $md.Add("| $($r.task) | $($r.epochs) | $(Format-Num $r.best_map50) | $(Format-Num $r.best_map50_95) | $(Format-Num $r.final_precision) | $(Format-Num $r.final_recall) |")
}
$md.Add("")
$md.Add("## GPU Direct Roundtrip")
$md.Add("")
$md.Add("| Task | Scheme | Frames | FPS | Avg Payload (KB) | Detections |")
$md.Add("|---|---|---:|---:|---:|---:|")
foreach ($r in $roundtripSummaries) {
    $md.Add("| $($r.task) | $($r.scheme) | $($r.frames) | $(Format-Num $r.fps) | $(Format-Num $r.avg_payload_kb 2) | $($r.detections) |")
}
$md.Add("")
$md.Add("## USRP Link")
$md.Add("")
$md.Add("| Run | Frames | OK | ERR | FER | FPS | Goodput (Mbps) | Decode ms/frame | Status |")
$md.Add("|---|---:|---:|---:|---:|---:|---:|---:|---|")
foreach ($r in $linkSummaries) {
    $md.Add("| $($r.run) | $($r.frames) | $($r.ok) | $($r.err) | $(Format-Num $r.fer 6) | $(Format-Num $r.fps 2) | $(Format-Num $r.goodput_mbps 3) | $(Format-Num $r.decode_total_ms_per_frame 3) | $($r.status) |")
}
$md.Add("")
$md.Add("## Decoder AWGN")
$md.Add("")
$md.Add("| Decoder | SNR (dB) | Codewords | BER | FER | Info Mbps | Codeword/s | Decode Seconds |")
$md.Add("|---|---:|---:|---:|---:|---:|---:|---:|")
foreach ($r in $awgnSummaries) {
    $md.Add("| $($r.decoder) | $(Format-Num $r.snr_db 1) | $($r.codewords) | $(Format-Num $r.ber 8) | $(Format-Num $r.fer 6) | $(Format-Num $r.info_mbps 3) | $(Format-Num $r.cw_per_s 1) | $(Format-Num $r.decode_seconds 4) |")
}
$md | Set-Content -Encoding UTF8 -Path (Join-Path $tablesDir "performance_three_line_tables.md")

$tex = New-Object System.Collections.Generic.List[string]
$tex.Add("% Requires \usepackage{booktabs}")
$tex.Add("\begin{table}[htbp]")
$tex.Add("\centering")
$tex.Add("\caption{YOLO training performance}")
$tex.Add("\begin{tabular}{lrrrrr}")
$tex.Add("\toprule")
$tex.Add("Task & Epochs & Best mAP50 & Best mAP50-95 & Precision & Recall \\")
$tex.Add("\midrule")
foreach ($r in $yoloSummaries) {
    $tex.Add("$($r.task) & $($r.epochs) & $(Format-Num $r.best_map50) & $(Format-Num $r.best_map50_95) & $(Format-Num $r.final_precision) & $(Format-Num $r.final_recall) \\")
}
$tex.Add("\bottomrule")
$tex.Add("\end{tabular}")
$tex.Add("\end{table}")
$tex.Add("")
$tex.Add("\begin{table}[htbp]")
$tex.Add("\centering")
$tex.Add("\caption{GPU direct roundtrip performance}")
$tex.Add("\begin{tabular}{llrrrr}")
$tex.Add("\toprule")
$tex.Add("Task & Scheme & Frames & FPS & Avg payload KB & Detections \\")
$tex.Add("\midrule")
foreach ($r in $roundtripSummaries) {
    $tex.Add("$($r.task) & $($r.scheme) & $($r.frames) & $(Format-Num $r.fps) & $(Format-Num $r.avg_payload_kb 2) & $($r.detections) \\")
}
$tex.Add("\bottomrule")
$tex.Add("\end{tabular}")
$tex.Add("\end{table}")
$tex | Set-Content -Encoding UTF8 -Path (Join-Path $tablesDir "performance_three_line_tables.tex")
