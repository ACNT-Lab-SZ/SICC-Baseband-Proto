# USRP Realtime Performance Table

The table below is a curated subset of measured USRP runs. `FER = 0` means no frame errors were observed within the listed sample, not a proof of zero error probability.

| Profile | PHY path | Modulation | FEC | Decoder | Rate (Msps) | Frames | FER | BER | FPS | Goodput (Mbps) | Avg frame latency (ms) |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Short-packet reliability | USRP realtime full-GPU RX | QPSK | CCSDS LDPC n128/k64 | CUDA BP-OSD | 25 | 59137 | 0 | 0 | 985.51 | 7.57 | 0.11 |
| Stream reliability | USRP realtime full-GPU RX | QPSK | DVB-S2 short N16200 rate 1/2 | CUDA BP | 25 | 10656 | 0 | 0 | 177.52 | 10.22 | 1.47 |
| Stream QPSK throughput | USRP realtime full-GPU RX | QPSK | DVB-S2 short N16200 rate 5/6 | CUDA BP | 25 | 10653 | 0 | 0 | 177.44 | 18.91 | n/a |
| Stream 16QAM throughput | USRP realtime full-GPU RX | 16QAM | DVB-S2 short N16200 rate 5/6 | CUDA BP | 25 | 10622 | 9.414e-05 | 1.184e-07 | 176.94 | 37.71 | 2.02 |
| 64QAM visible sample | USRP realtime full-GPU RX | 64QAM | DVB-S2 short N16200 rate 1/2 | CUDA BP | 25 | 870 | 0 | 0 | 172.95 | 29.89 | 2.06 |
| Dual-USRP gated check | Two-USRP gated GPU pipeline | QPSK | DVB-S2 short N16200 rate 1/4 | CUDA BP | 12.5 | 3114 | 3.211e-04 | 0 | 110.72 | 1.43 | n/a |

Source details and log paths are preserved in `performance_results/tables/usrp_realtime_performance.csv`.

