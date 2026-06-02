# USRP Transfer CSV Analysis

## CSV inventory

| Group | Runs | CSV files | Meaning |
|---|---:|---:|---|
| Offline simulated SNR/channel traces | 59 | 118 | 每组 run 包含 simulated_snr_trace_db.csv 与 simulated_channel_gain_trace_db.csv |
| Offline predicted power traces | 4 | 4 | 早期预测相对功率 trace |
| Real USRP profile/performance | 1 | 2 | manifest.csv + usrp_gain_frame_metrics.csv |
| Decoder AWGN | 1 | 1 | decoder-only BER/FER/throughput |

## Offline channel/SNR configuration summary

| Metric | Value |
|---|---:|
| Simulated runs | 59 |
| Mean nominal SNR (dB) | 24 |
| Mean run SNR (dB) | 12.604 |
| Mean channel gain (dB) | -11.396 |
| Prediction-only runs | 4 |

## Real USRP transfer profile

| Parameter | Value |
|---|---:|
| TX gain (dB) | 31 |
| RX gain (dB) | 22 |
| Total gain setting (dB) | 53 |
| Sample rate (Msps) | 12.5 |
| OFDM symbols / data symbols | 96 / 72 |
| Active subcarriers | 450 |
| FEC | DVB_S2_short_N16200_rate_5_6, N=16200, K=13320, rate=0.8222 |
| Modulation | qpsk |
| Frames / OK / ERR | 14531 / 14398 / 133 |
| FER | 0.009153 |
| Goodput (Mbps) | 0.69 |
| FPS | 26 |
| Decode ms/frame | 1.97 |
| Status | FAILED |

## Decoder AWGN reference point

| SNR (dB) | BER | FER | Info Mbps | Codeword/s |
|---:|---:|---:|---:|---:|
| 3 | 0.00021875 | 0.001 | 24.254 | 378974.495 |
