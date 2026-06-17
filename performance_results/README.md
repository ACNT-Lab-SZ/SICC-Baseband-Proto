# Performance Artifacts

This directory keeps a compact set of measured baseband performance artifacts for the repository README. Large raw logs are intentionally not the primary display target here; source CSV/log paths are preserved in the data tables when available.

## Layout

```text
performance_results/
  figures/   GPU-Pipeline performance figures and selected vector exports
  tables/    GitHub-readable summary tables and CSV metadata
  data/      Compact CSV data used by the displayed figures
  raw/       Archived historical result snapshots, when included
  tools/     Historical extraction and plotting scripts, when included
```

## Displayed Figures

- `figures/gpu_pipeline_fer_qpsk_all_codes.svg`: pure GPU-Pipeline QPSK FER curves across available code families.
- `figures/gpu_pipeline_16qam_goodput_latency.svg`: pure GPU-Pipeline 16QAM effective throughput and frame latency.

PNG/PDF companions are included where available for local viewing or document export:

- `figures/gpu_pipeline_16qam_goodput_latency.png`
- `figures/gpu_pipeline_fer_qpsk_all_codes.pdf`
- `figures/gpu_pipeline_16qam_goodput_latency.pdf`

## Displayed Tables

- `tables/usrp_realtime_performance.md`: curated USRP realtime performance table.
- `tables/usrp_realtime_performance.csv`: same rows with source paths and overflow notes.

## Compact Data Sources

- `data/gpu_pipeline_fer_qpsk_all_codes.csv`
- `data/gpu_pipeline_goodput_all_codes.csv`
- `data/gpu_pipeline_latency_all_codes.csv`
- `data/usrp_realtime_reliable_source.csv`

## Interpretation Notes

- `FER = 0` in the USRP table means no frame errors were observed within the listed sample.
- The 64QAM row is a visible calibration sample and should not be treated as a long-run stability claim.
- Pure GPU-Pipeline figures exclude USRP hardware effects such as RF impairment, synchronization misses and UHD overflow.
- USRP rows include real hardware scheduling and RF behavior; compare them against offline GPU-Pipeline rows only with that distinction in mind.

