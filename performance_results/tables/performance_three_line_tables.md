# Performance Results Three-Line Tables

Generated from organized result files under `organized_workspace_20260601`.

## YOLO Training

| Task | Epochs | Best mAP50 | Best mAP50-95 | Final Precision | Final Recall |
|---|---:|---:|---:|---:|---:|
| earthquake damage detection | 50 | 0.8562 | 0.8562 | 0.8845 | 0.7942 |
| maritime ship detection | 50 | 0.9861 | 0.7024 | 0.9627 | 0.9453 |

## GPU Direct Roundtrip

| Task | Scheme | Frames | FPS | Avg Payload (KB) | Detections |
|---|---|---:|---:|---:|---:|
| VISO vehicle split inference | uniform | 120 | 13.0696 | 1700 | 5277 |
| VISO vehicle split inference | roi-layered | 120 | 10.3915 | 907.87 | 5277 |
| earthquake | roi-layered | 40 | 8.7896 | 1824.61 | 42 |
| maritime_ship | roi-layered | 40 | 8.7775 | 442.99 | 88 |
| flood | roi-layered | 20 | 4.6626 | 1423.1 | 112 |
| flood | uniform | 20 | 5.4982 | 1700 | 112 |

## USRP Link

| Run | Frames | OK | ERR | FER | FPS | Goodput (Mbps) | Decode ms/frame | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| dual_usrp_gated_gpu_pipeline | 14531 | 14398 | 133 | 0.009153 | 26 | 0.69 | 1.97 | FAILED |

## Decoder AWGN

| Decoder | SNR (dB) | Codewords | BER | FER | Info Mbps | Codeword/s | Decode Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| osd-only | 3 | 1000 | 0.00021875 | 0.001 | 24.254 | 378974.5 | 0.003 |
