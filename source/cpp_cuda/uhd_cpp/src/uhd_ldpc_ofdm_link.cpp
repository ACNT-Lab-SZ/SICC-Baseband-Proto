#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include "uhd_c_radio.h"
#include "uhd_radio_session.hpp"
#include "matrix_loader.hpp"
#include "predictive_amc.hpp"
#include "rx_decoder_router.hpp"
#ifdef HAVE_GPU_FULL_PIPELINE
#include "gpu_ofdm_pipeline.hpp"
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <atomic>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <complex>
#include <cstdlib>
#include <cstdint>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <cstring>
#include <mutex>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#ifdef HAVE_CUDA_OSD
#include "matlab_decoder_c_api.h"
#endif

using cf32 = std::complex<float>;
using cd = std::complex<double>;

constexpr double kPi = 3.141592653589793238462643383279502884;

static void fft_inplace(std::vector<cd>& a, bool inverse);
static std::vector<cf32> fft_to_shifted_grid(const cf32* time, int nfft);

struct Options {
    std::string mode = "sim";
    std::string tx_addr = "192.168.40.2";
    std::string rx_addr = "192.168.50.2";
    std::string device_args;
    std::string alist_path = "../Code_Matrices_Lib/LDPC/CCSDS_ldpc_n512_k256.alist";
    std::string clock_source = "internal";
    std::string time_source = "internal";
    std::string antenna;
    std::string subdev;
    std::string tx_antenna;
    std::string rx_antenna;
    std::string tx_subdev;
    std::string rx_subdev;
    std::string traffic_mode = "test";
    std::string decoder = "cpu";
    std::string modulation = "qpsk";
    std::string input_file;
    std::string output_file;
    std::string iq_input_file;
    std::string iq_output_file;
    std::string precomp_channel_files;
    std::string actual_channel_files;
    std::string sync_preamble = "random-qpsk";
    std::string rx_channel_estimation = "pilot";
    std::string tx_awgn_reference = "frame";
    std::string tx_awgn_scope = "frame";
    std::string tx_awgn_snr_trace;
    std::string tx_host_format = "fc32";
    std::string ui_constellation_host = "127.0.0.1";
    std::string ui_spectrum_host = "127.0.0.1";
    std::string ui_metrics_host = "127.0.0.1";
    std::string ui_ntn_host = "127.0.0.1";
    std::string adaptive_feedback_host = "127.0.0.1";
    std::string startup_amc_mode = "throughput";
    std::string startup_amc_matrix_dir;
    std::string sim_snr_trace;
    std::string adaptive_power_trace;
    std::string start_gate_file;

    double freq = 2.45e9;
    double rate = 15.36e6;
    double master_clock_rate = 184.32e6;
    double tx_gain = 10.0;
    double rx_gain = 15.0;
    double amplitude = 0.70;
    double sync_threshold = 0.45;
    double rx_snr_gate_db = -120.0;
    double sim_snr_db = std::numeric_limits<double>::infinity();
    double tx_awgn_snr_db = std::numeric_limits<double>::infinity();
    double sim_dynamic_upgrade_snr_db = 13.5;
    double sim_dynamic_downgrade_snr_db = 12.0;
    double ldpc_normalization = 0.80;
    double ldpc_offset = 0.15;
    double ldpc_damping = 0.15;
    double duration_sec = 0.0;
    double adaptive_predicted_snr_db = 10.0;
    double adaptive_prediction_confidence = 0.80;
    double adaptive_prediction_age_ms = 0.0;
    double adaptive_prediction_evm = 0.05;
    double adaptive_power_base_snr_db = 24.0;
    double sync_late_peak_threshold = 0.0;
    double start_gate_timeout_sec = 0.0;

    int channel = 0;
    int tx_channel = -1;
    int rx_channel = -1;
    int nfft = 1024;
    int cp = 72;
    int num_symbols = 48;
    int active_sc = 512;
    int pre_half_len = 256;
    int sync_zc_root = 25;
    int sync_tracking_window = 0;
    int pilot_period = 0;
    int ldpc_max_iter = 8;
    int ldpc_schedule = -1;
    int frames = 0;
    int max_frame_errors = 0;
    int zero_error_stop_frames = 0;
    int warmup_frames = 0;
    int report_every = 20;
    int max_buffered_frames = 4;
    int test_seed = 20260418;
    int cuda_min_batch = 30;
    int cuda_max_batch = 4096;
    int cuda_latency_us = 2000;
    int rx_queue_blocks = 256;
    int rx_frame_queue_frames = 512;
    int rx_block_samps = 32768;
    int tx_queue_frames = 128;
    size_t rx_buffer_samples = 0;
    size_t rx_gpu_buffer_bytes = 0;
    int radio_oversample = 1;
    int adaptive_feedback_port = 65435;
    int adaptive_window_frames = 20;
    int tx_repeat_min = 1;
    int tx_repeat_max = 3;
    int adaptive_power_live_wait_ms = 2000;
    int adaptive_power_hold_frames = 1;
    int ui_constellation_port = 65432;
    int ui_constellation_points = 1000;
    int ui_constellation_interval_ms = 50;
    int ui_spectrum_port = 65433;
    int ui_spectrum_interval_ms = 100;
    int ui_metrics_port = 65434;
    int ui_metrics_interval_ms = 200;
    int ui_ntn_port = 65436;
    int ui_ntn_interval_ms = 200;
    std::uint64_t startup_amc_file_bytes = 0;
    bool verbose = false;
    bool profile_pipeline = false;
    bool loop_file = false;
    bool systematic_front_info = false;
    bool ui_constellation = false;
    bool ui_spectrum = false;
    bool ui_metrics = false;
    bool ui_ntn = false;
    bool suppress_error_frames = false;
    bool adaptive = false;
    bool adaptive_predictive = false;
    bool adaptive_power_trace_live = false;
    bool adaptive_enable_16qam = false;
    bool adaptive_dynamic_modulation = false;
    bool startup_amc = false;
    bool rx_buffered = false;
    bool rx_gpu_buffered = false;
    bool gpu_pipeline = false;
    bool gpu_tx_baseband = false;
    bool gpu_rx_sync = false;
    bool gpu_rx_demod = false;
    bool skip_reference_ber = false;
    bool gpu_sim_wall_clock_throughput = false;
};

struct PhyConfig {
    int nfft = 1024;
    int cp = 72;
    int num_symbols = 48;
    int active_sc = 512;
    int pre_half_len = 256;
    int sync_tracking_window = 0;
    std::string sync_preamble = "random-qpsk";
    std::string rx_channel_estimation = "pilot";
    std::string modulation = "qpsk";
    int bits_per_symbol = 2;
    int max_buffered_frames = 4;
    double rate = 15.36e6;
    double amplitude = 0.70;
    double sync_threshold = 0.45;
    double sync_late_peak_threshold = 0.0;
    int test_seed = 20260418;
    int radio_oversample = 1;

    std::vector<int> pilot_symbols;
    std::vector<int> data_symbols;
    std::vector<int> used_indices;
    std::vector<cf32> constellation;
    std::vector<std::vector<uint8_t>> constellation_bits;
    std::vector<cf32> preamble;
    std::vector<std::vector<cf32>> pilot_grid; // [pilot][carrier]

    int sym_len() const { return nfft + cp; }
    int frame_len() const { return static_cast<int>(preamble.size()) + num_symbols * sym_len(); }
    int coded_bits_per_frame() const { return active_sc * static_cast<int>(data_symbols.size()) * bits_per_symbol; }
};

struct Alist {
    int n = 0;
    int m = 0;
    std::vector<std::vector<int>> col_rows;
    std::vector<std::vector<int>> row_cols;
};

struct LdpcCode {
    enum class EncoderKind {
        Rref,
        DvbS2Accumulator,
    };

    int n = 0;
    int m = 0;
    int k = 0;
    int rank = 0;
    int words = 0;
    EncoderKind encoder_kind = EncoderKind::Rref;

    std::vector<std::vector<int>> row_cols;
    std::vector<std::vector<int>> col_rows;
    std::vector<std::vector<int>> row_edges;
    std::vector<std::vector<int>> col_edges;
    std::vector<int> edge_col;

    std::vector<int> pivot_cols;
    std::vector<int> info_cols;
    std::vector<std::vector<uint64_t>> rref_rows;
};

struct DecodeResult {
    std::vector<uint8_t> info_bits;
    std::vector<uint8_t> code_bits;
    bool parity_ok = false;
    int iterations = 0;
};

struct FrameDecodeResult {
    bool detected = false;
    bool parity_ok = false;
    bool frame_ok = false;
    uint32_t frame_id = 0;
    int bit_errors = 0;
    int pre_fec_bit_errors = -1;
    int iterations = 0;
    double peak = 0.0;
    double cfo_hz = 0.0;
    double snr_db = 0.0;
    double evm_rms = std::numeric_limits<double>::quiet_NaN();
    std::vector<uint8_t> info_bits;
};

struct MediaHeader {
    uint32_t magic = 0;
    uint16_t version = 0;
    uint16_t header_len = 0;
    uint32_t flags = 0;
    uint32_t stream_id = 0;
    uint32_t frame_id = 0;
    uint64_t total_size = 0;
    uint64_t offset = 0;
    uint32_t payload_len = 0;
    uint32_t payload_crc32 = 0;
    uint32_t header_crc32 = 0;
};

struct RxStats {
    uint64_t detected = 0;
    uint64_t ok = 0;
    uint64_t err = 0;
    uint64_t ok_info_bits = 0;
    uint64_t bit_errors = 0;
    uint64_t pre_fec_bit_errors = 0;
    uint64_t pre_fec_bits = 0;
    uint64_t info_bits = 0;
    uint64_t sum_iter = 0;
    double sum_snr_db = 0.0;
    double min_snr_db = std::numeric_limits<double>::infinity();
    double sum_evm_rms = 0.0;
    uint64_t evm_samples = 0;
    double sum_abs_cfo_hz = 0.0;
};

struct DecodeProfile {
    uint64_t frames = 0;
    uint64_t cfo_ns = 0;
    uint64_t fft_ns = 0;
    uint64_t channel_ns = 0;
    uint64_t llr_ns = 0;
    uint64_t fec_ns = 0;
    uint64_t reference_ns = 0;
};

struct PipelineProfile {
    std::atomic<uint64_t> rx_blocks{0};
    std::atomic<uint64_t> rx_samples{0};
    std::atomic<uint64_t> rx_recv_ns{0};
    std::atomic<uint64_t> rx_enqueue_ns{0};
    std::atomic<uint64_t> sync_blocks{0};
    std::atomic<uint64_t> sync_input_samples{0};
    std::atomic<uint64_t> sync_frames{0};
    std::atomic<uint64_t> sync_ns{0};
    std::atomic<uint64_t> tx_frames{0};
    std::atomic<uint64_t> media_frames{0};
    std::atomic<uint64_t> media_accepts{0};
    std::atomic<uint64_t> media_ns{0};
    DecodeProfile decode;
};

enum class SyncState {
    Acquisition,
    Tracking,
};

struct SyncResult {
    bool found = false;
    bool incomplete = false;
    size_t start = 0;
    double coarse_metric = 0.0;
    double fine_peak = 0.0;
    double cfo_hz = 0.0;
};

struct SyncStats {
    uint64_t found = 0;
    uint64_t miss = 0;
    uint64_t tracking_miss = 0;
    uint64_t incomplete = 0;
    uint64_t skipped_samples = 0;
    uint64_t max_start = 0;
};

static uint64_t elapsed_ns(std::chrono::steady_clock::time_point t0, std::chrono::steady_clock::time_point t1)
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
}

static void print_usage()
{
    std::cout
        << "UHD CCSDS-LDPC OFDM link\n\n"
        << "Modes:\n"
        << "  --mode sim                 Build/decode frames in memory, no radio.\n"
        << "  --mode gpu-sim             Local no-radio loopback through GPU TX/sync/demod/decoder stages.\n"
        << "  --mode bench               Offline subsystem throughput benchmark, no radio.\n"
        << "  --mode tx                  Transmit continuous frames from one USRP.\n"
        << "  --mode rx                  Receive/decode continuous frames on one USRP.\n\n"
        << "  --mode trx                 Full-duplex single-USRP TX/RX with RX three-stage pipeline.\n\n"
        << "  --mode trx-capture         Full-duplex TX/RX, write captured RX IQ to disk without decoding.\n"
        << "  --mode replay              Decode a captured RX IQ file offline.\n\n"
        << "Common options:\n"
        << "  --alist <path>             LDPC alist file. Default: CCSDS n512 k256.\n"
        << "  --modulation <name>        bpsk|qpsk|16qam|64qam. Default: qpsk.\n"
        << "  --active-sc <N>            Active OFDM subcarriers. Default: 512.\n"
        << "  --pilot-period <N>         Insert a full-band pilot every N OFDM symbols. 0 keeps MATLAB-video pilots.\n"
        << "  --sync-preamble <name>     random-qpsk|zc-ofdm. zc-ofdm repeats two full OFDM/ZC blocks.\n"
        << "  --sync-zc-root <N>         Zadoff-Chu root for --sync-preamble zc-ofdm. Default: 25.\n"
        << "  --sync-tracking-window <N> Samples searched after the predicted next frame. 0 auto-tunes by preamble.\n"
        << "  --sync-late-peak-threshold <x>  Extra peak gate for late tracking candidates. 0 auto-tunes.\n"
        << "  --rx-channel-est <name>    pilot|none. none assumes TX precompensation removes channel distortion.\n"
        << "  --freq <Hz>                RF center frequency. Default: 2.45e9.\n"
        << "  --rate <Sps>               Baseband sample rate. Default: 15.36e6.\n"
        << "  --mcr <Hz>                 Master clock rate. Default: 184.32e6.\n"
        << "  --frames <N>               Stop after N frames. 0 means unlimited for tx/rx.\n"
        << "  --max-frame-errors <N>     Stop test-traffic FER runs after N erroneous frames. 0 disables.\n"
        << "  --zero-error-stop-frames <N> Stop test-traffic FER runs when the first N frames contain no errors. 0 disables.\n"
        << "  --warmup-frames <N>        Decode but do not count the first N RX test frames in FER/BER. Default: 0.\n"
        << "  --duration <sec>           Stop after this many seconds. 0 means unlimited.\n"
        << "  --start-gate-file <path>   Wait for this file before TX sends or RX starts streaming.\n"
        << "  --start-gate-timeout-sec <s> Timeout for --start-gate-file. 0 waits forever.\n"
        << "  --ldpc-iter <N>            Normalized min-sum iterations. Default: 8.\n"
        << "  --ldpc-normalization <x>   NMS message scale in (0,1]. Default: 0.80.\n"
        << "  --ldpc-offset <x>          Offset min-sum subtraction, >= 0. Default: 0.15.\n"
        << "  --ldpc-damping <x>         BP message damping in [0,1). Default: 0.15.\n"
        << "  --ldpc-schedule <N>        GPU BP schedule: 0=flooding, 1=node parallel, 2=colored layered, 3=flat layered; default auto.\n\n"
        << "  --suppress-error-frames    Do not print every failed frame; useful for long tests.\n\n"
        << "Decoder options:\n"
        << "  --decoder cpu|cuda-bp|cuda-bp-osd|cuda-osd  RX decoder. cuda-bp-osd supports dynamic short systematic LDPC up to n256/k128.\n"
        << "  --systematic-front-info    Put LDPC information bits in the first k columns, matching CUDA BP-OSD.\n"
        << "  --cuda-min-batch <N>       CUDA stream decoder minimum batch. Default: 30.\n"
        << "  --cuda-max-batch <N>       CUDA stream decoder maximum batch. Default: 4096.\n"
        << "  --cuda-latency-us <N>      CUDA stream decoder latency budget. Default: 2000.\n\n"
        << "RX threading options:\n"
        << "  --rx-queue-blocks <N>      Sample blocks buffered between UHD recv and demod. Default: 256.\n\n"
        << "  --rx-frame-queue <N>       OFDM frames buffered between sync and decoder. Default: 512.\n\n"
        << "  --rx-block-samps <N>       Samples requested per UHD recv call. Default: 32768.\n\n"
        << "  --tx-queue-frames <N>      Frames prefetched between TX baseband and UHD send. Default: 128.\n\n"
        << "  --rx-buffered              Use a continuous local RX IQ buffer before online sync.\n"
        << "  --rx-buffer-samples <N>    Capacity of --rx-buffered in complex samples. Default: about 2 s.\n\n"
        << "  --rx-gpu-buffered          Stage online RX IQ into a CUDA ring buffer before GPU PHY.\n"
        << "  --rx-gpu-buffer-size <S>   CUDA ring capacity: 500M, 1G, 2G, or 4G. Default: 1G.\n\n"
        << "  --radio-oversample <N>     Integer TX upsample/RX downsample factor before UHD. Default: 1.\n\n"
        << "  --rx-snr-gate-db <dB>      Skip FEC decode below this estimated SNR. Default: disabled.\n\n"
        << "  --sim-snr-db <dB>          Add AWGN in --mode sim at the requested SNR. Default: disabled.\n\n"
        << "  --tx-awgn-snr-db <dB>      Add complex AWGN to TX baseband immediately before UHD send.\n\n"
        << "  --tx-awgn-snr-trace <spec> Per-frame TX AWGN SNR trace, e.g. 8:200,14.5:200,18:200.\n"
        << "                             In predictive TRX, the same trace is treated as known predicted SNR.\n\n"
        << "  --tx-awgn-scope <name>     frame|data-subcarriers. data-subcarriers preserves preamble/pilots.\n\n"
        << "  --tx-awgn-reference <name> frame|preamble. Power reference for --tx-awgn-snr-db. Default: frame.\n\n"
        << "  --tx-host-format <name>    UHD TX CPU IQ format: fc32|sc16. Default: fc32.\n\n"
        << "GPU simulation dynamic SNR options:\n"
        << "  --sim-snr-trace <spec|file> Per-frame SNR trace for --mode gpu-sim. Format: snr:frames,..., demo, or CSV/TXT.\n"
        << "                             Example: 8:200,14.5:200,18:200,10:200,15:200,7:200.\n"
        << "  --sim-dynamic-upgrade-snr-db <dB>  Dynamic modulation switches QPSK->16QAM above this SNR. Default: 13.5.\n"
        << "  --sim-dynamic-downgrade-snr-db <dB> Dynamic modulation switches 16QAM->QPSK below this SNR. Default: 12.0.\n\n"
        << "Channel injection options:\n"
        << "  --precomp-channel <paths>  Comma/semicolon separated predictor channel file(s) for TX precompensation.\n"
        << "  --actual-channel <paths>   Comma/semicolon separated actual channel file(s) injected after TX precompensation.\n"
        << "                             File format: P, then P rows of h k delay or h_real h_imag k delay.\n\n"
        << "  --profile-pipeline        Print RX subsystem throughput and per-frame timing.\n\n"
        << "  --skip-reference-ber      Throughput-only test mode: skip deterministic BER comparison and count parity-ok frames.\n\n"
        << "Experimental GPU pipeline options (--mode gpu-sim/trx/replay, build with ENABLE_GPU_FULL_PIPELINE=ON):\n"
        << "  --gpu-pipeline            Enable GPU TX baseband, GPU RX sync, and GPU RX demod/LLR.\n"
        << "  --gpu-phy-backend         Alias for GPU TX/RX PHY, CUDA BP-OSD, and --rx-gpu-buffered.\n"
        << "  --gpu-tx-baseband         Move LDPC encode, modulation, precompensation, and OFDM IFFT to CUDA.\n"
        << "  --gpu-rx-sync             Use CUDA Schmidl-Cox frame acquisition in the sync thread.\n"
        << "  --gpu-rx-demod            Move RX CFO/FFT/channel/LLR to CUDA; FEC decode still uses selected decoder.\n\n"
        << "Adaptive reliability options:\n"
        << "  --adaptive                Enable RX feedback and TX frame repetition control.\n"
        << "  --adaptive-predictive     Use predicted channel quality for AMC/ARQ feed-forward decisions.\n"
        << "  --adaptive-predicted-snr-db <dB>  Predicted post-equalization SNR used by AMC. Default: 10.\n"
        << "  --adaptive-confidence <0..1>      Predictor confidence used by AMC. Default: 0.80.\n"
        << "  --adaptive-prediction-age-ms <ms> Predictor age penalty input. Default: 0.\n"
        << "  --adaptive-predicted-evm <x>      Predicted residual EVM. Default: 0.05.\n"
        << "  --adaptive-power-trace <spec|file> Predicted relative channel power in dB. Values are mapped to\n"
        << "                             SNR by --adaptive-power-base-snr-db + relativePowerDb, and then drive AMC.\n"
        << "                             Format matches --sim-snr-trace, or use a CSV/TXT file with one power value per line.\n"
        << "  --adaptive-power-base-snr-db <dB>  SNR at 0 dB predicted relative power. Default: 24.\n"
        << "  --adaptive-power-trace-live  Treat --adaptive-power-trace as a live CSV/TXT file refreshed during runtime.\n"
        << "  --adaptive-power-live-wait-ms <N> Wait up to N ms for each new live trace sample before reusing last value. Default: 2000.\n"
        << "  --adaptive-power-hold-frames <N> Apply each predicted power sample to N PHY frames. Default: 1.\n"
        << "  --adaptive-enable-16qam   Allow predictive AMC to recommend 16QAM.\n"
        << "  --adaptive-dynamic-modulation  Enable runtime QPSK/16QAM profile-bank hot switching.\n"
        << "  --gpu-sim-wall-clock-throughput  Report gpu-sim throughput from wall-clock processing time instead of sample-rate-limited link time.\n"
        << "  --adaptive-feedback-host <ip>  Feedback UDP target/listen host. Default: 127.0.0.1.\n"
        << "  --adaptive-feedback-port <N>   Feedback UDP port. Default: 65435.\n"
        << "  --adaptive-window-frames <N>   RX decision window. Default: 20.\n"
        << "  --tx-repeat-min <N>       Minimum media frame repetitions. Default: 1.\n"
        << "  --tx-repeat-max <N>       Maximum media frame repetitions. Default: 3.\n\n"
        << "Startup-only adaptive coding and modulation:\n"
        << "  --startup-amc             Select one fixed CCSDS/DVB-S2 modulation/code profile before transfer starts.\n"
        << "  --startup-amc-mode <name> throughput|reliability. Default: throughput.\n"
        << "  --startup-amc-matrix-dir <path> Directory containing CCSDS and DVB-S2 .alist profile matrices.\n"
        << "  --startup-amc-file-bytes <N> File length shared by separate TX/RX startup selection.\n"
        << "                             gpu-sim selects from --sim-snr-db; USRP modes select from tx/rx gain budget.\n"
        << "                             This mode is intentionally incompatible with runtime --adaptive.\n\n"
        << "UI telemetry options:\n"
        << "  --ui-constellation        Send MATLAB-compatible constellation UDP packets to the UI.\n"
        << "  --ui-host <ip>            UI UDP target. Default: 127.0.0.1.\n"
        << "  --ui-port <N>             UI UDP port. Default: 65432.\n"
        << "  --ui-points <N>           Max constellation points per packet. Default: 1000.\n"
        << "  --ui-interval-ms <N>      Minimum UDP interval. Default: 50 ms.\n\n"
        << "  --ui-spectrum             Send RF/baseband Welch PSD UDP packets to the UI.\n"
        << "  --ui-spectrum-port <N>    Spectrum UDP port. Default: 65433.\n"
        << "  --ui-spectrum-interval-ms <N>  Minimum spectrum interval. Default: 100 ms.\n"
        << "  --ui-metrics              Send BER/FER/SNR UDP packets to the UI.\n"
        << "  --ui-metrics-port <N>     Metrics UDP port. Default: 65434.\n"
        << "  --ui-metrics-interval-ms <N>   Minimum metrics interval. Default: 200 ms.\n\n"
        << "  --ui-ntn                  Send Satellite_UI JSON throughput/MCS packets to UDP 65436.\n"
        << "  --ui-ntn-host <ip>        Satellite_UI JSON target. Default: 127.0.0.1.\n"
        << "  --ui-ntn-port <N>         Satellite_UI JSON port. Default: 65436.\n"
        << "  --ui-ntn-interval-ms <N>  Minimum JSON metric interval. Default: 200 ms.\n\n"
        << "Traffic options:\n"
        << "  --traffic test|file        test sends deterministic BER payloads; file sends a local image/video file.\n"
        << "  --input <path>             Local file to transmit in file mode.\n"
        << "  --output <path>            RX reassembled output file. Default: rx_media_payload.bin.\n"
        << "  --iq-output <path>         Captured complex-float IQ file for --mode trx-capture.\n"
        << "  --iq-input <path>          Captured complex-float IQ file for --mode replay.\n"
        << "  --loop-file                Repeat input file continuously for video-loop demonstrations.\n\n"
        << "Radio options:\n"
        << "  --tx-addr <ip>             Default: 192.168.40.2.\n"
        << "  --rx-addr <ip>             Default: 192.168.50.2.\n"
        << "  --args <uhd args>          Overrides addr=... if provided.\n"
        << "  --tx-gain <dB>             Default: 10.\n"
        << "  --rx-gain <dB>             Default: 15.\n"
        << "  --clock-source <name>      internal/external/gpsdo. Default: internal.\n"
        << "  --time-source <name>       internal/external/gpsdo. Default: internal.\n"
        << "  --antenna <name>           Optional UHD antenna selection.\n"
        << "  --subdev <spec>            Optional UHD subdevice spec.\n"
        << "  --tx-antenna <name>        TX antenna override for --mode trx or tx.\n"
        << "  --rx-antenna <name>        RX antenna override for --mode trx or rx.\n"
        << "  --tx-subdev <spec>         TX subdevice override, e.g. A:0.\n"
        << "  --rx-subdev <spec>         RX subdevice override, e.g. B:0.\n"
        << "  --tx-channel <N>           TX channel override. Default: --channel.\n"
        << "  --rx-channel <N>           RX channel override. Default: --channel.\n";
}

static bool is_option(const std::string& s)
{
    return s.rfind("--", 0) == 0;
}

static std::string require_value(int& i, int argc, char** argv)
{
    if (i + 1 >= argc || is_option(argv[i + 1])) {
        throw std::runtime_error("missing value for " + std::string(argv[i]));
    }
    return argv[++i];
}

static size_t parse_size_bytes(const std::string& text)
{
    std::string s;
    s.reserve(text.size());
    for (unsigned char c : text) {
        if (!std::isspace(c) && c != '_') {
            s.push_back(static_cast<char>(std::tolower(c)));
        }
    }
    if (s.empty()) {
        throw std::runtime_error("empty size string");
    }

    size_t pos = 0;
    while (pos < s.size() && (std::isdigit(static_cast<unsigned char>(s[pos])) || s[pos] == '.')) {
        ++pos;
    }
    if (pos == 0) {
        throw std::runtime_error("invalid size string: " + text);
    }

    const double value = std::stod(s.substr(0, pos));
    const std::string suffix = s.substr(pos);
    double scale = 1.0;
    if (suffix.empty() || suffix == "b") {
        scale = 1.0;
    } else if (suffix == "k" || suffix == "kb" || suffix == "kib") {
        scale = 1024.0;
    } else if (suffix == "m" || suffix == "mb" || suffix == "mib") {
        scale = 1024.0 * 1024.0;
    } else if (suffix == "g" || suffix == "gb" || suffix == "gib") {
        scale = 1024.0 * 1024.0 * 1024.0;
    } else {
        throw std::runtime_error("invalid size suffix in: " + text);
    }
    if (value <= 0.0) {
        throw std::runtime_error("size must be positive: " + text);
    }
    return static_cast<size_t>(value * scale);
}

static Options parse_options(int argc, char** argv)
{
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--sim-snr-trace" || a == "--dynamic-snr-trace") {
            opt.sim_snr_trace = require_value(i, argc, argv);
            continue;
        }
        if (a == "--sim-dynamic-upgrade-snr-db") {
            opt.sim_dynamic_upgrade_snr_db = std::stod(require_value(i, argc, argv));
            continue;
        }
        if (a == "--sim-dynamic-downgrade-snr-db") {
            opt.sim_dynamic_downgrade_snr_db = std::stod(require_value(i, argc, argv));
            continue;
        }
        if (a == "--adaptive-power-trace" || a == "--predicted-power-trace" ||
            a == "--channel-power-trace") {
            opt.adaptive = true;
            opt.adaptive_predictive = true;
            opt.adaptive_dynamic_modulation = true;
            opt.adaptive_power_trace = require_value(i, argc, argv);
            continue;
        }
        if (a == "--adaptive-power-base-snr-db" || a == "--predicted-power-base-snr-db") {
            opt.adaptive = true;
            opt.adaptive_predictive = true;
            opt.adaptive_dynamic_modulation = true;
            opt.adaptive_power_base_snr_db = std::stod(require_value(i, argc, argv));
            continue;
        }
        if (a == "--adaptive-power-trace-live" || a == "--predicted-power-trace-live") {
            opt.adaptive = true;
            opt.adaptive_predictive = true;
            opt.adaptive_dynamic_modulation = true;
            opt.adaptive_power_trace_live = true;
            continue;
        }
        if (a == "--adaptive-power-live-wait-ms") {
            opt.adaptive_power_live_wait_ms = std::max(0, std::stoi(require_value(i, argc, argv)));
            continue;
        }
        if (a == "--adaptive-power-hold-frames" || a == "--predicted-power-hold-frames") {
            opt.adaptive_power_hold_frames = std::max(1, std::stoi(require_value(i, argc, argv)));
            continue;
        }
        if (a == "--gpu-sim-wall-clock-throughput" || a == "--gpu-sim-unlimited-throughput") {
            opt.gpu_sim_wall_clock_throughput = true;
            continue;
        }
        if (a == "--start-gate-file") {
            opt.start_gate_file = require_value(i, argc, argv);
            continue;
        }
        if (a == "--start-gate-timeout-sec") {
            opt.start_gate_timeout_sec = std::stod(require_value(i, argc, argv));
            continue;
        }
        if (a == "--warmup-frames" || a == "--rx-warmup-frames") {
            opt.warmup_frames = std::stoi(require_value(i, argc, argv));
            continue;
        }
        if (a == "--help" || a == "-h") {
            print_usage();
            std::exit(0);
        } else if (a == "--mode") {
            opt.mode = require_value(i, argc, argv);
        } else if (a == "--tx-addr") {
            opt.tx_addr = require_value(i, argc, argv);
        } else if (a == "--rx-addr") {
            opt.rx_addr = require_value(i, argc, argv);
        } else if (a == "--args") {
            opt.device_args = require_value(i, argc, argv);
        } else if (a == "--alist") {
            opt.alist_path = require_value(i, argc, argv);
        } else if (a == "--clock-source") {
            opt.clock_source = require_value(i, argc, argv);
        } else if (a == "--time-source") {
            opt.time_source = require_value(i, argc, argv);
        } else if (a == "--antenna") {
            opt.antenna = require_value(i, argc, argv);
        } else if (a == "--subdev") {
            opt.subdev = require_value(i, argc, argv);
        } else if (a == "--tx-antenna") {
            opt.tx_antenna = require_value(i, argc, argv);
        } else if (a == "--rx-antenna") {
            opt.rx_antenna = require_value(i, argc, argv);
        } else if (a == "--tx-subdev") {
            opt.tx_subdev = require_value(i, argc, argv);
        } else if (a == "--rx-subdev") {
            opt.rx_subdev = require_value(i, argc, argv);
        } else if (a == "--traffic") {
            opt.traffic_mode = require_value(i, argc, argv);
        } else if (a == "--decoder") {
            opt.decoder = require_value(i, argc, argv);
        } else if (a == "--modulation") {
            opt.modulation = require_value(i, argc, argv);
            std::transform(opt.modulation.begin(), opt.modulation.end(), opt.modulation.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        } else if (a == "--input" || a == "--payload-file") {
            opt.input_file = require_value(i, argc, argv);
        } else if (a == "--output") {
            opt.output_file = require_value(i, argc, argv);
        } else if (a == "--iq-output" || a == "--capture-output") {
            opt.iq_output_file = require_value(i, argc, argv);
        } else if (a == "--iq-input" || a == "--capture-input") {
            opt.iq_input_file = require_value(i, argc, argv);
        } else if (a == "--ui-constellation") {
            opt.ui_constellation = true;
        } else if (a == "--ui-dashboard") {
            opt.ui_constellation = true;
            opt.ui_spectrum = true;
            opt.ui_metrics = true;
            opt.ui_ntn = true;
        } else if (a == "--ui-host") {
            opt.ui_constellation_host = require_value(i, argc, argv);
            opt.ui_spectrum_host = opt.ui_constellation_host;
            opt.ui_metrics_host = opt.ui_constellation_host;
            opt.ui_ntn_host = opt.ui_constellation_host;
        } else if (a == "--ui-port") {
            opt.ui_constellation_port = std::stoi(require_value(i, argc, argv));
        } else if (a == "--ui-points") {
            opt.ui_constellation_points = std::stoi(require_value(i, argc, argv));
        } else if (a == "--ui-interval-ms") {
            opt.ui_constellation_interval_ms = std::stoi(require_value(i, argc, argv));
        } else if (a == "--ui-spectrum") {
            opt.ui_spectrum = true;
        } else if (a == "--ui-spectrum-host") {
            opt.ui_spectrum_host = require_value(i, argc, argv);
        } else if (a == "--ui-spectrum-port") {
            opt.ui_spectrum_port = std::stoi(require_value(i, argc, argv));
        } else if (a == "--ui-spectrum-interval-ms") {
            opt.ui_spectrum_interval_ms = std::stoi(require_value(i, argc, argv));
        } else if (a == "--ui-metrics") {
            opt.ui_metrics = true;
        } else if (a == "--ui-metrics-host") {
            opt.ui_metrics_host = require_value(i, argc, argv);
        } else if (a == "--ui-metrics-port") {
            opt.ui_metrics_port = std::stoi(require_value(i, argc, argv));
        } else if (a == "--ui-metrics-interval-ms") {
            opt.ui_metrics_interval_ms = std::stoi(require_value(i, argc, argv));
        } else if (a == "--ui-ntn" || a == "--ui-satellite") {
            opt.ui_ntn = true;
        } else if (a == "--ui-ntn-host") {
            opt.ui_ntn_host = require_value(i, argc, argv);
        } else if (a == "--ui-ntn-port") {
            opt.ui_ntn_port = std::stoi(require_value(i, argc, argv));
        } else if (a == "--ui-ntn-interval-ms") {
            opt.ui_ntn_interval_ms = std::stoi(require_value(i, argc, argv));
        } else if (a == "--loop-file") {
            opt.loop_file = true;
        } else if (a == "--systematic-front-info") {
            opt.systematic_front_info = true;
        } else if (a == "--freq") {
            opt.freq = std::stod(require_value(i, argc, argv));
        } else if (a == "--rate") {
            opt.rate = std::stod(require_value(i, argc, argv));
        } else if (a == "--mcr") {
            opt.master_clock_rate = std::stod(require_value(i, argc, argv));
        } else if (a == "--tx-gain") {
            opt.tx_gain = std::stod(require_value(i, argc, argv));
        } else if (a == "--rx-gain") {
            opt.rx_gain = std::stod(require_value(i, argc, argv));
        } else if (a == "--amplitude") {
            opt.amplitude = std::stod(require_value(i, argc, argv));
        } else if (a == "--sync-threshold") {
            opt.sync_threshold = std::stod(require_value(i, argc, argv));
        } else if (a == "--rx-snr-gate-db") {
            opt.rx_snr_gate_db = std::stod(require_value(i, argc, argv));
        } else if (a == "--sim-snr-db") {
            opt.sim_snr_db = std::stod(require_value(i, argc, argv));
        } else if (a == "--tx-awgn-snr-db" || a == "--tx-noise-snr-db") {
            opt.tx_awgn_snr_db = std::stod(require_value(i, argc, argv));
        } else if (a == "--tx-awgn-snr-trace" || a == "--tx-noise-snr-trace") {
            opt.tx_awgn_snr_trace = require_value(i, argc, argv);
        } else if (a == "--tx-awgn-scope") {
            opt.tx_awgn_scope = require_value(i, argc, argv);
            std::transform(opt.tx_awgn_scope.begin(), opt.tx_awgn_scope.end(), opt.tx_awgn_scope.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        } else if (a == "--tx-awgn-reference") {
            opt.tx_awgn_reference = require_value(i, argc, argv);
            std::transform(opt.tx_awgn_reference.begin(), opt.tx_awgn_reference.end(), opt.tx_awgn_reference.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        } else if (a == "--tx-host-format" || a == "--tx-cpu-format") {
            opt.tx_host_format = require_value(i, argc, argv);
            std::transform(opt.tx_host_format.begin(), opt.tx_host_format.end(), opt.tx_host_format.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        } else if (a == "--precomp-channel" || a == "--precomp-channel-files" ||
                   a == "--predicted-channel" || a == "--tx-precomp-channel") {
            opt.precomp_channel_files = require_value(i, argc, argv);
        } else if (a == "--actual-channel" || a == "--actual-channel-files" ||
                   a == "--inject-channel" || a == "--channel-model") {
            opt.actual_channel_files = require_value(i, argc, argv);
        } else if (a == "--ldpc-normalization") {
            opt.ldpc_normalization = std::stod(require_value(i, argc, argv));
        } else if (a == "--ldpc-offset") {
            opt.ldpc_offset = std::stod(require_value(i, argc, argv));
        } else if (a == "--ldpc-damping") {
            opt.ldpc_damping = std::stod(require_value(i, argc, argv));
        } else if (a == "--ldpc-schedule") {
            opt.ldpc_schedule = std::stoi(require_value(i, argc, argv));
        } else if (a == "--duration") {
            opt.duration_sec = std::stod(require_value(i, argc, argv));
        } else if (a == "--channel") {
            opt.channel = std::stoi(require_value(i, argc, argv));
        } else if (a == "--tx-channel") {
            opt.tx_channel = std::stoi(require_value(i, argc, argv));
        } else if (a == "--rx-channel") {
            opt.rx_channel = std::stoi(require_value(i, argc, argv));
        } else if (a == "--nfft") {
            opt.nfft = std::stoi(require_value(i, argc, argv));
        } else if (a == "--cp") {
            opt.cp = std::stoi(require_value(i, argc, argv));
        } else if (a == "--num-symbols") {
            opt.num_symbols = std::stoi(require_value(i, argc, argv));
        } else if (a == "--active-sc") {
            opt.active_sc = std::stoi(require_value(i, argc, argv));
        } else if (a == "--pre-half-len") {
            opt.pre_half_len = std::stoi(require_value(i, argc, argv));
        } else if (a == "--pilot-period") {
            opt.pilot_period = std::stoi(require_value(i, argc, argv));
        } else if (a == "--sync-preamble") {
            opt.sync_preamble = require_value(i, argc, argv);
            std::transform(opt.sync_preamble.begin(), opt.sync_preamble.end(), opt.sync_preamble.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        } else if (a == "--sync-zc-root") {
            opt.sync_zc_root = std::stoi(require_value(i, argc, argv));
        } else if (a == "--sync-tracking-window") {
            opt.sync_tracking_window = std::stoi(require_value(i, argc, argv));
        } else if (a == "--sync-late-peak-threshold") {
            opt.sync_late_peak_threshold = std::stod(require_value(i, argc, argv));
        } else if (a == "--rx-channel-est" || a == "--rx-channel-estimation") {
            opt.rx_channel_estimation = require_value(i, argc, argv);
            std::transform(opt.rx_channel_estimation.begin(), opt.rx_channel_estimation.end(), opt.rx_channel_estimation.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        } else if (a == "--rx-no-channel-est") {
            opt.rx_channel_estimation = "none";
        } else if (a == "--ldpc-iter") {
            opt.ldpc_max_iter = std::stoi(require_value(i, argc, argv));
        } else if (a == "--frames") {
            opt.frames = std::stoi(require_value(i, argc, argv));
        } else if (a == "--max-frame-errors") {
            opt.max_frame_errors = std::stoi(require_value(i, argc, argv));
        } else if (a == "--zero-error-stop-frames") {
            opt.zero_error_stop_frames = std::stoi(require_value(i, argc, argv));
        } else if (a == "--report-every") {
            opt.report_every = std::stoi(require_value(i, argc, argv));
        } else if (a == "--test-seed") {
            opt.test_seed = std::stoi(require_value(i, argc, argv));
        } else if (a == "--cuda-min-batch") {
            opt.cuda_min_batch = std::stoi(require_value(i, argc, argv));
        } else if (a == "--cuda-max-batch") {
            opt.cuda_max_batch = std::stoi(require_value(i, argc, argv));
        } else if (a == "--cuda-latency-us") {
            opt.cuda_latency_us = std::stoi(require_value(i, argc, argv));
        } else if (a == "--rx-queue-blocks") {
            opt.rx_queue_blocks = std::stoi(require_value(i, argc, argv));
        } else if (a == "--rx-frame-queue") {
            opt.rx_frame_queue_frames = std::stoi(require_value(i, argc, argv));
        } else if (a == "--rx-block-samps") {
            opt.rx_block_samps = std::stoi(require_value(i, argc, argv));
        } else if (a == "--tx-queue-frames") {
            opt.tx_queue_frames = std::stoi(require_value(i, argc, argv));
        } else if (a == "--rx-buffered" || a == "--online-buffered") {
            opt.rx_buffered = true;
        } else if (a == "--rx-buffer-samples") {
            opt.rx_buffer_samples = static_cast<size_t>(std::stoull(require_value(i, argc, argv)));
        } else if (a == "--rx-gpu-buffered") {
            opt.rx_gpu_buffered = true;
        } else if (a == "--rx-gpu-buffer-size" || a == "--rx-gpu-buffer-bytes") {
            opt.rx_gpu_buffered = true;
            opt.rx_gpu_buffer_bytes = parse_size_bytes(require_value(i, argc, argv));
        } else if (a == "--radio-oversample" || a == "--oversample") {
            opt.radio_oversample = std::stoi(require_value(i, argc, argv));
        } else if (a == "--adaptive") {
            opt.adaptive = true;
        } else if (a == "--adaptive-predictive" || a == "--predictive-amc") {
            opt.adaptive = true;
            opt.adaptive_predictive = true;
        } else if (a == "--adaptive-predicted-snr-db") {
            opt.adaptive = true;
            opt.adaptive_predictive = true;
            opt.adaptive_predicted_snr_db = std::stod(require_value(i, argc, argv));
        } else if (a == "--adaptive-confidence" || a == "--adaptive-prediction-confidence") {
            opt.adaptive = true;
            opt.adaptive_predictive = true;
            opt.adaptive_prediction_confidence = std::stod(require_value(i, argc, argv));
        } else if (a == "--adaptive-prediction-age-ms") {
            opt.adaptive = true;
            opt.adaptive_predictive = true;
            opt.adaptive_prediction_age_ms = std::stod(require_value(i, argc, argv));
        } else if (a == "--adaptive-predicted-evm") {
            opt.adaptive = true;
            opt.adaptive_predictive = true;
            opt.adaptive_prediction_evm = std::stod(require_value(i, argc, argv));
        } else if (a == "--adaptive-enable-16qam") {
            opt.adaptive = true;
            opt.adaptive_predictive = true;
            opt.adaptive_enable_16qam = true;
            opt.adaptive_dynamic_modulation = true;
        } else if (a == "--adaptive-dynamic-modulation" || a == "--adaptive-modulation") {
            opt.adaptive = true;
            opt.adaptive_predictive = true;
            opt.adaptive_enable_16qam = true;
            opt.adaptive_dynamic_modulation = true;
        } else if (a == "--adaptive-feedback-host") {
            opt.adaptive_feedback_host = require_value(i, argc, argv);
        } else if (a == "--adaptive-feedback-port") {
            opt.adaptive_feedback_port = std::stoi(require_value(i, argc, argv));
        } else if (a == "--adaptive-window-frames") {
            opt.adaptive_window_frames = std::stoi(require_value(i, argc, argv));
        } else if (a == "--tx-repeat-min") {
            opt.tx_repeat_min = std::stoi(require_value(i, argc, argv));
        } else if (a == "--tx-repeat-max") {
            opt.tx_repeat_max = std::stoi(require_value(i, argc, argv));
        } else if (a == "--startup-amc") {
            opt.startup_amc = true;
        } else if (a == "--startup-amc-mode" || a == "--startup-amc-objective") {
            opt.startup_amc = true;
            opt.startup_amc_mode = require_value(i, argc, argv);
            std::transform(opt.startup_amc_mode.begin(), opt.startup_amc_mode.end(), opt.startup_amc_mode.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        } else if (a == "--startup-amc-matrix-dir") {
            opt.startup_amc = true;
            opt.startup_amc_matrix_dir = require_value(i, argc, argv);
        } else if (a == "--startup-amc-file-bytes") {
            opt.startup_amc = true;
            opt.startup_amc_file_bytes = static_cast<std::uint64_t>(
                std::stoull(require_value(i, argc, argv)));
        } else if (a == "--verbose") {
            opt.verbose = true;
        } else if (a == "--profile-pipeline") {
            opt.profile_pipeline = true;
        } else if (a == "--skip-reference-ber" || a == "--throughput-only") {
            opt.skip_reference_ber = true;
        } else if (a == "--suppress-error-frames") {
            opt.suppress_error_frames = true;
        } else if (a == "--gpu-pipeline" || a == "--gpu-full-pipeline") {
            opt.gpu_pipeline = true;
            opt.gpu_tx_baseband = true;
            opt.gpu_rx_sync = true;
            opt.gpu_rx_demod = true;
        } else if (a == "--gpu-phy-backend") {
            opt.gpu_pipeline = true;
            opt.gpu_tx_baseband = true;
            opt.gpu_rx_sync = true;
            opt.gpu_rx_demod = true;
            opt.rx_gpu_buffered = true;
            opt.decoder = "cuda-bp-osd";
            opt.systematic_front_info = true;
        } else if (a == "--gpu-tx-baseband") {
            opt.gpu_tx_baseband = true;
        } else if (a == "--gpu-rx-sync") {
            opt.gpu_rx_sync = true;
        } else if (a == "--gpu-rx-demod") {
            opt.gpu_rx_demod = true;
        } else {
            throw std::runtime_error("unknown option: " + a);
        }
    }
    std::transform(opt.mode.begin(), opt.mode.end(), opt.mode.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    std::transform(opt.traffic_mode.begin(), opt.traffic_mode.end(), opt.traffic_mode.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    std::transform(opt.decoder.begin(), opt.decoder.end(), opt.decoder.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (opt.traffic_mode == "media" || opt.traffic_mode == "video") {
        opt.traffic_mode = "file";
    }
    if (opt.decoder == "cuda" || opt.decoder == "gpu" || opt.decoder == "bp-osd") {
        opt.decoder = "cuda-bp-osd";
    } else if (opt.decoder == "osd" || opt.decoder == "gpu-osd" || opt.decoder == "osd-only") {
        opt.decoder = "cuda-osd";
    } else if (opt.decoder == "bp" || opt.decoder == "gpu-bp") {
        opt.decoder = "cuda-bp";
    }
    opt.rx_queue_blocks = std::max(opt.rx_queue_blocks, 4);
    opt.max_frame_errors = std::max(opt.max_frame_errors, 0);
    opt.zero_error_stop_frames = std::max(opt.zero_error_stop_frames, 0);
    opt.warmup_frames = std::max(opt.warmup_frames, 0);
    if (opt.sync_preamble == "zc" || opt.sync_preamble == "zc_ofdm") {
        opt.sync_preamble = "zc-ofdm";
    }
    if (opt.sync_preamble != "random-qpsk" && opt.sync_preamble != "zc-ofdm") {
        throw std::runtime_error("unsupported --sync-preamble: " + opt.sync_preamble);
    }
    if (opt.rx_channel_estimation == "off" || opt.rx_channel_estimation == "identity") {
        opt.rx_channel_estimation = "none";
    }
    if (opt.rx_channel_estimation != "pilot" && opt.rx_channel_estimation != "none") {
        throw std::runtime_error("unsupported --rx-channel-est: " + opt.rx_channel_estimation);
    }
    if (opt.tx_awgn_reference != "frame" && opt.tx_awgn_reference != "preamble") {
        throw std::runtime_error("unsupported --tx-awgn-reference: " + opt.tx_awgn_reference);
    }
    if (opt.tx_awgn_scope == "data" || opt.tx_awgn_scope == "data-subcarrier") {
        opt.tx_awgn_scope = "data-subcarriers";
    }
    if (opt.tx_awgn_scope != "frame" && opt.tx_awgn_scope != "data-subcarriers") {
        throw std::runtime_error("unsupported --tx-awgn-scope: " + opt.tx_awgn_scope);
    }
    if (opt.tx_host_format != "fc32" && opt.tx_host_format != "sc16") {
        throw std::runtime_error("unsupported --tx-host-format: " + opt.tx_host_format);
    }
    opt.sync_zc_root = std::max(opt.sync_zc_root, 1);
    opt.sync_tracking_window = std::max(opt.sync_tracking_window, 0);
    opt.sync_late_peak_threshold = std::max(opt.sync_late_peak_threshold, 0.0);
    opt.start_gate_timeout_sec = std::max(opt.start_gate_timeout_sec, 0.0);
    opt.rx_frame_queue_frames = std::max(opt.rx_frame_queue_frames, 4);
    opt.rx_block_samps = std::max(opt.rx_block_samps, 1024);
    opt.tx_queue_frames = std::max(opt.tx_queue_frames, 1);
    if (opt.rx_buffer_samples == 0) {
        opt.rx_buffer_samples = static_cast<size_t>(std::max(4.0 * opt.rate, 8.0 * 1024.0 * 1024.0));
    }
    if (opt.rx_gpu_buffered && opt.rx_gpu_buffer_bytes == 0) {
        opt.rx_gpu_buffer_bytes = static_cast<size_t>(1024ull * 1024ull * 1024ull);
    }
    opt.rx_snr_gate_db = std::max(opt.rx_snr_gate_db, -120.0);
    opt.radio_oversample = std::max(opt.radio_oversample, 1);
    if (!(opt.ldpc_normalization > 0.0 && opt.ldpc_normalization <= 1.0)) {
        throw std::runtime_error("--ldpc-normalization must be in (0,1]");
    }
    if (opt.ldpc_offset < 0.0) {
        throw std::runtime_error("--ldpc-offset must be >= 0");
    }
    if (!(opt.ldpc_damping >= 0.0 && opt.ldpc_damping < 1.0)) {
        throw std::runtime_error("--ldpc-damping must be in [0,1)");
    }
    if (opt.ldpc_schedule < -1 || opt.ldpc_schedule > 3) {
        throw std::runtime_error("--ldpc-schedule must be -1 (auto), 0, 1, 2, or 3");
    }
    opt.adaptive_feedback_port = std::max(opt.adaptive_feedback_port, 1);
    opt.adaptive_window_frames = std::max(opt.adaptive_window_frames, 1);
    opt.tx_repeat_min = std::max(opt.tx_repeat_min, 1);
    opt.tx_repeat_max = std::max(opt.tx_repeat_max, opt.tx_repeat_min);
    if (opt.startup_amc_mode == "high-throughput" || opt.startup_amc_mode == "high_throughput") {
        opt.startup_amc_mode = "throughput";
    } else if (opt.startup_amc_mode == "high-reliability" || opt.startup_amc_mode == "high_reliability") {
        opt.startup_amc_mode = "reliability";
    }
    if (opt.startup_amc_mode != "throughput" && opt.startup_amc_mode != "reliability") {
        throw std::runtime_error("--startup-amc-mode must be throughput or reliability");
    }
    opt.adaptive_prediction_confidence = std::max(0.0, std::min(1.0, opt.adaptive_prediction_confidence));
    opt.adaptive_prediction_age_ms = std::max(0.0, opt.adaptive_prediction_age_ms);
    opt.adaptive_prediction_evm = std::max(0.0, opt.adaptive_prediction_evm);
    opt.ui_constellation_port = std::max(opt.ui_constellation_port, 1);
    opt.ui_constellation_points = std::max(opt.ui_constellation_points, 1);
    opt.ui_constellation_interval_ms = std::max(opt.ui_constellation_interval_ms, 1);
    opt.ui_spectrum_port = std::max(opt.ui_spectrum_port, 1);
    opt.ui_spectrum_interval_ms = std::max(opt.ui_spectrum_interval_ms, 1);
    opt.ui_metrics_port = std::max(opt.ui_metrics_port, 1);
    opt.ui_metrics_interval_ms = std::max(opt.ui_metrics_interval_ms, 1);
    opt.ui_ntn_port = std::max(opt.ui_ntn_port, 1);
    opt.ui_ntn_interval_ms = std::max(opt.ui_ntn_interval_ms, 1);
#ifndef HAVE_GPU_FULL_PIPELINE
    if (opt.gpu_pipeline || opt.gpu_tx_baseband || opt.gpu_rx_sync || opt.gpu_rx_demod || opt.rx_gpu_buffered) {
        throw std::runtime_error("GPU full-pipeline options require configuring with -DENABLE_GPU_FULL_PIPELINE=ON");
    }
#endif
    const bool gpu_mode_ok =
        opt.mode == "gpu-sim" ||
        opt.mode == "gpu-local" ||
        opt.mode == "rx" ||
        opt.mode == "trx" ||
        opt.mode == "trx-capture" ||
        opt.mode == "replay";
    if ((opt.gpu_pipeline || opt.gpu_tx_baseband || opt.gpu_rx_sync || opt.gpu_rx_demod) &&
        !gpu_mode_ok) {
        throw std::runtime_error("GPU full-pipeline stages are wired into --mode gpu-sim, rx, trx, trx-capture, and replay");
    }
    return opt;
}

static void apply_startup_amc(Options& opt)
{
    if (!opt.startup_amc) {
        return;
    }
    if (opt.adaptive) {
        throw std::runtime_error(
            "--startup-amc performs a fixed pre-transfer selection and cannot be combined with runtime --adaptive");
    }

    usrp_link::StartupAmcInput input;
    input.mode = opt.startup_amc_mode == "reliability"
        ? usrp_link::StartupAmcMode::HighReliability
        : usrp_link::StartupAmcMode::HighThroughput;
    if (opt.mode == "gpu-sim" || opt.mode == "gpu-local" || opt.mode == "sim") {
        if (!std::isfinite(opt.sim_snr_db)) {
            throw std::runtime_error(
                "--startup-amc in an offline mode requires --sim-snr-db <dB>");
        }
        input.link_source = usrp_link::StartupLinkSource::OfflineGpuSnr;
        input.snr_db = opt.sim_snr_db;
    } else if (opt.mode == "tx" || opt.mode == "rx" || opt.mode == "trx" ||
               opt.mode == "trx-capture" || opt.mode == "capture") {
        input.link_source = usrp_link::StartupLinkSource::UsrpGainBudget;
        input.tx_gain_db = opt.tx_gain;
        input.rx_gain_db = opt.rx_gain;
    } else {
        throw std::runtime_error(
            "--startup-amc is supported in sim/gpu-sim and active USRP tx/rx/trx modes only");
    }

    input.file_size_bytes = opt.startup_amc_file_bytes;
    if (input.file_size_bytes == 0 && opt.traffic_mode == "file" && !opt.input_file.empty()) {
        std::error_code ec;
        const auto bytes = std::filesystem::file_size(std::filesystem::path(opt.input_file), ec);
        if (!ec) {
            input.file_size_bytes = static_cast<std::uint64_t>(bytes);
        }
    }
    if (opt.mode == "rx" && opt.traffic_mode == "file" && input.file_size_bytes == 0) {
        throw std::runtime_error(
            "--startup-amc with standalone RX file traffic requires --startup-amc-file-bytes <TX input size> "
            "so TX and RX choose the same fixed MCS");
    }

    const usrp_link::StartupAmcDecision selected = usrp_link::StartupAmcPolicy().select(input);
    const std::filesystem::path matrix_dir = opt.startup_amc_matrix_dir.empty()
        ? std::filesystem::path(opt.alist_path).parent_path()
        : std::filesystem::path(opt.startup_amc_matrix_dir);
    const std::filesystem::path selected_matrix = matrix_dir / selected.fec_matrix_file;
    if (!std::filesystem::exists(selected_matrix)) {
        throw std::runtime_error(
            "startup AMC selected " + selected.fec_profile + " but the matrix was not found: " +
            selected_matrix.string() + ". Pass --startup-amc-matrix-dir <Code_Matrices_Lib/LDPC>.");
    }

    opt.modulation = selected.modulation;
    opt.alist_path = selected_matrix.string();
    if (selected.fec_profile == "ccsds-ldpc-128-64") {
        opt.active_sc = 512;
        opt.num_symbols = 20;
        opt.pilot_period = 4;
        std::cout << "[STARTUP-AMC] shortPacketPhy activeSC=" << opt.active_sc
                  << " symbols=" << opt.num_symbols
                  << " pilotPeriod=" << opt.pilot_period
                  << " (CCSDS divisibility profile)\n";
    }
    std::cout << "[STARTUP-AMC] mode=" << usrp_link::StartupAmcPolicy::to_string(input.mode)
              << " source=" << selected.metric_name
              << " metric=" << std::fixed << std::setprecision(2) << selected.selection_metric << " dB"
              << " fileBytes=" << input.file_size_bytes
              << " fileClass=" << selected.size_class << "\n"
              << "[STARTUP-AMC] fixedSelection modulation=" << selected.modulation
              << " fec=" << selected.fec_profile
              << " nominalRate=" << selected.fec_rate
              << " matrix=" << opt.alist_path << "\n"
              << "[STARTUP-AMC] reason=\"" << selected.reason
              << "\" runtimeSwitch=disabled\n";
}

static bool uses_cuda_bp_osd(const Options& opt)
{
    return opt.decoder == "cuda-bp-osd";
}

static bool uses_cuda_short_decoder(const Options& opt)
{
    return opt.decoder == "cuda-bp-osd" || opt.decoder == "cuda-osd";
}

static bool uses_front_info_layout(const Options& opt)
{
    return opt.systematic_front_info || uses_cuda_short_decoder(opt);
}

static std::optional<double> add_awgn_to_time_samples(
    std::vector<cf32>& samples,
    double snr_db,
    std::mt19937& rng,
    size_t reference_samples = 0)
{
    if (!std::isfinite(snr_db) || samples.empty()) {
        return std::nullopt;
    }
    const size_t nref = reference_samples == 0
        ? samples.size()
        : std::min(reference_samples, samples.size());
    double signal_power = 0.0;
    for (size_t i = 0; i < nref; ++i) {
        signal_power += std::norm(samples[i]);
    }
    signal_power = std::max(signal_power / static_cast<double>(nref), 1.0e-12);
    const double noise_var = signal_power / std::pow(10.0, snr_db / 10.0);
    const float sigma = static_cast<float>(std::sqrt(std::max(noise_var, 0.0) * 0.5));
    std::normal_distribution<float> noise_dist(0.0f, 1.0f);
    for (auto& s : samples) {
        s += cf32(sigma * noise_dist(rng), sigma * noise_dist(rng));
    }
    return noise_var;
}

static int64_t ui_now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

class UdpConstellationSink {
public:
    UdpConstellationSink(const std::string& host, int port, int max_points, int interval_ms)
        : max_points_(std::max(max_points, 1)),
          interval_(std::chrono::milliseconds(std::max(interval_ms, 1)))
    {
#if defined(_WIN32)
        WSADATA wsa{};
        const int wsa_status = WSAStartup(MAKEWORD(2, 2), &wsa);
        if (wsa_status != 0) {
            throw std::runtime_error("WSAStartup failed for UI constellation UDP sender");
        }
        wsa_started_ = true;
        sock_ = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (sock_ == INVALID_SOCKET) {
            throw std::runtime_error("failed to create UI constellation UDP socket");
        }
#else
        sock_ = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (sock_ < 0) {
            throw std::runtime_error("failed to create UI constellation UDP socket");
        }
#endif
        std::memset(&addr_, 0, sizeof(addr_));
        addr_.sin_family = AF_INET;
        addr_.sin_port = htons(static_cast<uint16_t>(port));
        if (::inet_pton(AF_INET, host.c_str(), &addr_.sin_addr) != 1) {
            throw std::runtime_error("UI constellation host must be an IPv4 address: " + host);
        }

        std::cout << "[UI] constellation UDP enabled: " << host << ":" << port
                  << " points=" << max_points_
                  << " intervalMs=" << interval_.count()
                  << " format=float32[tx_i,tx_q,rx_i,rx_q]\n";
    }

    ~UdpConstellationSink()
    {
#if defined(_WIN32)
        if (sock_ != INVALID_SOCKET) {
            closesocket(sock_);
            sock_ = INVALID_SOCKET;
        }
        if (wsa_started_) {
            WSACleanup();
        }
#else
        if (sock_ >= 0) {
            close(sock_);
            sock_ = -1;
        }
#endif
    }

    int max_points() const { return max_points_; }

    bool due() const
    {
        return std::chrono::steady_clock::now() >= next_send_;
    }

    void send(const std::vector<cf32>& tx, const std::vector<cf32>& rx)
    {
        const size_t n = std::min(tx.size(), rx.size());
        if (n == 0) {
            return;
        }

        std::vector<float> payload(4 * n);
        for (size_t i = 0; i < n; ++i) {
            payload[i] = tx[i].real();
            payload[n + i] = tx[i].imag();
            payload[2 * n + i] = rx[i].real();
            payload[3 * n + i] = rx[i].imag();
        }

        const char* bytes = reinterpret_cast<const char*>(payload.data());
        const int byte_count = static_cast<int>(payload.size() * sizeof(float));
#if defined(_WIN32)
        const int sent = ::sendto(sock_, bytes, byte_count, 0, reinterpret_cast<const sockaddr*>(&addr_), sizeof(addr_));
        if (sent == SOCKET_ERROR && ++send_errors_ <= 3) {
            std::cerr << "[UI] constellation UDP send failed, WSA error=" << WSAGetLastError() << "\n";
        }
#else
        const ssize_t sent = ::sendto(sock_, bytes, static_cast<size_t>(byte_count), 0, reinterpret_cast<const sockaddr*>(&addr_), sizeof(addr_));
        if (sent < 0 && ++send_errors_ <= 3) {
            std::cerr << "[UI] constellation UDP send failed\n";
        }
#endif
        next_send_ = std::chrono::steady_clock::now() + interval_;
    }

private:
    int max_points_ = 1000;
    std::chrono::milliseconds interval_{50};
    std::chrono::steady_clock::time_point next_send_{};
    int send_errors_ = 0;
    sockaddr_in addr_{};
#if defined(_WIN32)
    SOCKET sock_ = INVALID_SOCKET;
    bool wsa_started_ = false;
#else
    int sock_ = -1;
#endif
};

class UdpSpectrumSink {
public:
    UdpSpectrumSink(const std::string& host, int port, int interval_ms)
        : interval_(std::chrono::milliseconds(std::max(interval_ms, 1)))
    {
#if defined(_WIN32)
        WSADATA wsa{};
        const int wsa_status = WSAStartup(MAKEWORD(2, 2), &wsa);
        if (wsa_status != 0) {
            throw std::runtime_error("WSAStartup failed for UI spectrum UDP sender");
        }
        wsa_started_ = true;
        sock_ = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (sock_ == INVALID_SOCKET) {
            throw std::runtime_error("failed to create UI spectrum UDP socket");
        }
#else
        sock_ = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (sock_ < 0) {
            throw std::runtime_error("failed to create UI spectrum UDP socket");
        }
#endif
        std::memset(&addr_, 0, sizeof(addr_));
        addr_.sin_family = AF_INET;
        addr_.sin_port = htons(static_cast<uint16_t>(port));
        if (::inet_pton(AF_INET, host.c_str(), &addr_.sin_addr) != 1) {
            throw std::runtime_error("UI spectrum host must be an IPv4 address: " + host);
        }
        std::cout << "[UI] spectrum UDP enabled: " << host << ":" << port
                  << " intervalMs=" << interval_.count()
                  << " format=float32[freq_hz...,psd_db...]\n";
    }

    ~UdpSpectrumSink()
    {
#if defined(_WIN32)
        if (sock_ != INVALID_SOCKET) {
            closesocket(sock_);
            sock_ = INVALID_SOCKET;
        }
        if (wsa_started_) {
            WSACleanup();
        }
#else
        if (sock_ >= 0) {
            close(sock_);
            sock_ = -1;
        }
#endif
    }

    bool due() const
    {
        return std::chrono::steady_clock::now() >= next_send_;
    }

    void send(const std::vector<float>& freq_hz, const std::vector<float>& psd_db)
    {
        const size_t n = std::min(freq_hz.size(), psd_db.size());
        if (n == 0) {
            return;
        }
        std::vector<float> payload(2 * n);
        std::copy(freq_hz.begin(), freq_hz.begin() + static_cast<std::ptrdiff_t>(n), payload.begin());
        std::copy(psd_db.begin(), psd_db.begin() + static_cast<std::ptrdiff_t>(n), payload.begin() + static_cast<std::ptrdiff_t>(n));
        send_payload(payload);
        next_send_ = std::chrono::steady_clock::now() + interval_;
    }

private:
    void send_payload(const std::vector<float>& payload)
    {
        const char* bytes = reinterpret_cast<const char*>(payload.data());
        const int byte_count = static_cast<int>(payload.size() * sizeof(float));
#if defined(_WIN32)
        const int sent = ::sendto(sock_, bytes, byte_count, 0, reinterpret_cast<const sockaddr*>(&addr_), sizeof(addr_));
        if (sent == SOCKET_ERROR && ++send_errors_ <= 3) {
            std::cerr << "[UI] spectrum UDP send failed, WSA error=" << WSAGetLastError() << "\n";
        }
#else
        const ssize_t sent = ::sendto(sock_, bytes, static_cast<size_t>(byte_count), 0, reinterpret_cast<const sockaddr*>(&addr_), sizeof(addr_));
        if (sent < 0 && ++send_errors_ <= 3) {
            std::cerr << "[UI] spectrum UDP send failed\n";
        }
#endif
    }

    std::chrono::milliseconds interval_{100};
    std::chrono::steady_clock::time_point next_send_{};
    int send_errors_ = 0;
    sockaddr_in addr_{};
#if defined(_WIN32)
    SOCKET sock_ = INVALID_SOCKET;
    bool wsa_started_ = false;
#else
    int sock_ = -1;
#endif
};

class UdpMetricsSink {
public:
    UdpMetricsSink(const std::string& host, int port, int interval_ms)
        : interval_(std::chrono::milliseconds(std::max(interval_ms, 1)))
    {
#if defined(_WIN32)
        WSADATA wsa{};
        const int wsa_status = WSAStartup(MAKEWORD(2, 2), &wsa);
        if (wsa_status != 0) {
            throw std::runtime_error("WSAStartup failed for UI metrics UDP sender");
        }
        wsa_started_ = true;
        sock_ = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (sock_ == INVALID_SOCKET) {
            throw std::runtime_error("failed to create UI metrics UDP socket");
        }
#else
        sock_ = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (sock_ < 0) {
            throw std::runtime_error("failed to create UI metrics UDP socket");
        }
#endif
        std::memset(&addr_, 0, sizeof(addr_));
        addr_.sin_family = AF_INET;
        addr_.sin_port = htons(static_cast<uint16_t>(port));
        if (::inet_pton(AF_INET, host.c_str(), &addr_.sin_addr) != 1) {
            throw std::runtime_error("UI metrics host must be an IPv4 address: " + host);
        }
        std::cout << "[UI] metrics UDP enabled: " << host << ":" << port
                  << " intervalMs=" << interval_.count()
                  << " format=float64[frames,ber,fer,preber,snr_db,goodput_mbps]\n";
    }

    ~UdpMetricsSink()
    {
#if defined(_WIN32)
        if (sock_ != INVALID_SOCKET) {
            closesocket(sock_);
            sock_ = INVALID_SOCKET;
        }
        if (wsa_started_) {
            WSACleanup();
        }
#else
        if (sock_ >= 0) {
            close(sock_);
            sock_ = -1;
        }
#endif
    }

    bool due() const
    {
        return std::chrono::steady_clock::now() >= next_send_;
    }

    void send(double frames, double ber, double fer, double preber, double snr_db, double goodput_mbps)
    {
        std::array<double, 6> payload{frames, ber, fer, preber, snr_db, goodput_mbps};
        const char* bytes = reinterpret_cast<const char*>(payload.data());
        const int byte_count = static_cast<int>(payload.size() * sizeof(double));
#if defined(_WIN32)
        const int sent = ::sendto(sock_, bytes, byte_count, 0, reinterpret_cast<const sockaddr*>(&addr_), sizeof(addr_));
        if (sent == SOCKET_ERROR && ++send_errors_ <= 3) {
            std::cerr << "[UI] metrics UDP send failed, WSA error=" << WSAGetLastError() << "\n";
        }
#else
        const ssize_t sent = ::sendto(sock_, bytes, static_cast<size_t>(byte_count), 0, reinterpret_cast<const sockaddr*>(&addr_), sizeof(addr_));
        if (sent < 0 && ++send_errors_ <= 3) {
            std::cerr << "[UI] metrics UDP send failed\n";
        }
#endif
        next_send_ = std::chrono::steady_clock::now() + interval_;
    }

private:
    std::chrono::milliseconds interval_{200};
    std::chrono::steady_clock::time_point next_send_{};
    int send_errors_ = 0;
    sockaddr_in addr_{};
#if defined(_WIN32)
    SOCKET sock_ = INVALID_SOCKET;
    bool wsa_started_ = false;
#else
    int sock_ = -1;
#endif
};

class UdpNtnMetricsSink {
public:
    UdpNtnMetricsSink(const std::string& host, int port, int interval_ms)
        : interval_(std::chrono::milliseconds(std::max(interval_ms, 1)))
    {
#if defined(_WIN32)
        WSADATA wsa{};
        const int wsa_status = WSAStartup(MAKEWORD(2, 2), &wsa);
        if (wsa_status != 0) {
            throw std::runtime_error("WSAStartup failed for Satellite_UI JSON sender");
        }
        wsa_started_ = true;
        sock_ = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (sock_ == INVALID_SOCKET) {
            throw std::runtime_error("failed to create Satellite_UI JSON socket");
        }
#else
        sock_ = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (sock_ < 0) {
            throw std::runtime_error("failed to create Satellite_UI JSON socket");
        }
#endif
        std::memset(&addr_, 0, sizeof(addr_));
        addr_.sin_family = AF_INET;
        addr_.sin_port = htons(static_cast<uint16_t>(port));
        if (::inet_pton(AF_INET, host.c_str(), &addr_.sin_addr) != 1) {
            throw std::runtime_error("Satellite_UI host must be an IPv4 address: " + host);
        }
        std::cout << "[UI] Satellite_UI JSON enabled: " << host << ":" << port
                  << " intervalMs=" << interval_.count()
                  << " type=ntn_channel_subplots fields=ber,fer,snrDb,relativePowerDb,throughputMbps,mcsIndex\n";
    }

    ~UdpNtnMetricsSink()
    {
#if defined(_WIN32)
        if (sock_ != INVALID_SOCKET) {
            closesocket(sock_);
            sock_ = INVALID_SOCKET;
        }
        if (wsa_started_) {
            WSACleanup();
        }
#else
        if (sock_ >= 0) {
            close(sock_);
            sock_ = -1;
        }
#endif
    }

    void send(
        double frame,
        double total_frames,
        double ber,
        bool ber_valid,
        double fer,
        double snr_db,
        double relative_power_db,
        double goodput_mbps,
        int mcs_index)
    {
        std::ostringstream oss;
        oss << "{\"type\":\"ntn_channel_subplots\",\"profile\":\"GPU-PHY/USRP\","
            << "\"frame\":" << static_cast<uint64_t>(std::max(frame, 0.0))
            << ",\"totalFrames\":" << static_cast<uint64_t>(std::max(total_frames, 0.0))
            << ",\"ber\":" << std::scientific << std::setprecision(6) << std::max(ber, 0.0)
            << ",\"berValid\":" << (ber_valid ? "true" : "false")
            << ",\"fer\":" << std::scientific << std::setprecision(6) << std::max(fer, 0.0)
            << ",\"snrDb\":" << std::fixed << std::setprecision(3) << snr_db
            << ",\"relativePowerDb\":";
        if (std::isfinite(relative_power_db)) {
            oss << std::fixed << std::setprecision(3) << relative_power_db;
        } else {
            oss << "null";
        }
        oss
            << ",\"channelPowerTimeStart\":" << static_cast<uint64_t>(std::max(frame, 0.0))
            << ",\"channelPowerTimeEnd\":" << static_cast<uint64_t>(std::max(frame, 0.0))
            << ",\"throughputMbps\":" << std::fixed << std::setprecision(6) << goodput_mbps
            << ",\"mcsIndex\":" << mcs_index << "}";
        const std::string payload = oss.str();
#if defined(_WIN32)
        const int sent = ::sendto(sock_, payload.data(), static_cast<int>(payload.size()), 0,
                                  reinterpret_cast<const sockaddr*>(&addr_), sizeof(addr_));
        if (sent == SOCKET_ERROR && ++send_errors_ <= 3) {
            std::cerr << "[UI] Satellite_UI JSON send failed, WSA error=" << WSAGetLastError() << "\n";
        }
#else
        const ssize_t sent = ::sendto(sock_, payload.data(), payload.size(), 0,
                                      reinterpret_cast<const sockaddr*>(&addr_), sizeof(addr_));
        if (sent < 0 && ++send_errors_ <= 3) {
            std::cerr << "[UI] Satellite_UI JSON send failed\n";
        }
#endif
        next_send_ = std::chrono::steady_clock::now() + interval_;
    }

private:
    std::chrono::milliseconds interval_{200};
    std::chrono::steady_clock::time_point next_send_{};
    int send_errors_ = 0;
    sockaddr_in addr_{};
#if defined(_WIN32)
    SOCKET sock_ = INVALID_SOCKET;
    bool wsa_started_ = false;
#else
    int sock_ = -1;
#endif
};

class AdaptiveFeedbackSender {
public:
    AdaptiveFeedbackSender(const std::string& host, int port)
    {
#if defined(_WIN32)
        WSADATA wsa{};
        const int wsa_status = WSAStartup(MAKEWORD(2, 2), &wsa);
        if (wsa_status != 0) {
            throw std::runtime_error("WSAStartup failed for adaptive feedback sender");
        }
        wsa_started_ = true;
        sock_ = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (sock_ == INVALID_SOCKET) {
            throw std::runtime_error("failed to create adaptive feedback socket");
        }
#else
        sock_ = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (sock_ < 0) {
            throw std::runtime_error("failed to create adaptive feedback socket");
        }
#endif
        std::memset(&addr_, 0, sizeof(addr_));
        addr_.sin_family = AF_INET;
        addr_.sin_port = htons(static_cast<uint16_t>(port));
        if (::inet_pton(AF_INET, host.c_str(), &addr_.sin_addr) != 1) {
            throw std::runtime_error("adaptive feedback host must be an IPv4 address: " + host);
        }
        std::cout << "[ADAPT-RX] feedback UDP enabled: " << host << ":" << port << "\n";
    }

    ~AdaptiveFeedbackSender()
    {
#if defined(_WIN32)
        if (sock_ != INVALID_SOCKET) {
            closesocket(sock_);
            sock_ = INVALID_SOCKET;
        }
        if (wsa_started_) {
            WSACleanup();
        }
#else
        if (sock_ >= 0) {
            close(sock_);
            sock_ = -1;
        }
#endif
    }

    void send(int repeat, double avg_snr_db, double fer, int frames)
    {
        std::ostringstream oss;
        oss << "ADAPT repeat=" << repeat
            << " snr=" << std::fixed << std::setprecision(2) << avg_snr_db
            << " fer=" << std::scientific << fer
            << " frames=" << frames << "\n";
        const std::string msg = oss.str();
#if defined(_WIN32)
        const int sent = ::sendto(sock_, msg.data(), static_cast<int>(msg.size()), 0, reinterpret_cast<const sockaddr*>(&addr_), sizeof(addr_));
        if (sent == SOCKET_ERROR && ++send_errors_ <= 3) {
            std::cerr << "[ADAPT-RX] UDP feedback send failed, WSA error=" << WSAGetLastError() << "\n";
        }
#else
        const ssize_t sent = ::sendto(sock_, msg.data(), msg.size(), 0, reinterpret_cast<const sockaddr*>(&addr_), sizeof(addr_));
        if (sent < 0 && ++send_errors_ <= 3) {
            std::cerr << "[ADAPT-RX] UDP feedback send failed\n";
        }
#endif
    }

    void send(const usrp_link::AmcDecision& decision, double avg_snr_db, double fer, int frames)
    {
        std::ostringstream oss;
        oss << "ADAPT repeat=" << decision.repeat_count
            << " mcs=" << static_cast<int>(decision.mcs)
            << " modulation=" << decision.modulation
            << " fec=" << decision.fec_profile
            << " effRate=" << std::fixed << std::setprecision(4) << decision.effective_rate
            << " snr=" << std::fixed << std::setprecision(2) << avg_snr_db
            << " fer=" << std::scientific << fer
            << " frames=" << frames << "\n";
        const std::string msg = oss.str();
#if defined(_WIN32)
        const int sent = ::sendto(sock_, msg.data(), static_cast<int>(msg.size()), 0, reinterpret_cast<const sockaddr*>(&addr_), sizeof(addr_));
        if (sent == SOCKET_ERROR && ++send_errors_ <= 3) {
            std::cerr << "[ADAPT-RX] UDP feedback send failed, WSA error=" << WSAGetLastError() << "\n";
        }
#else
        const ssize_t sent = ::sendto(sock_, msg.data(), msg.size(), 0, reinterpret_cast<const sockaddr*>(&addr_), sizeof(addr_));
        if (sent < 0 && ++send_errors_ <= 3) {
            std::cerr << "[ADAPT-RX] UDP feedback send failed\n";
        }
#endif
    }

private:
    int send_errors_ = 0;
    sockaddr_in addr_{};
#if defined(_WIN32)
    SOCKET sock_ = INVALID_SOCKET;
    bool wsa_started_ = false;
#else
    int sock_ = -1;
#endif
};

class AdaptiveTxController {
public:
    explicit AdaptiveTxController(const Options& opt)
        : min_repeat_(std::max(opt.tx_repeat_min, 1)),
          max_repeat_(std::max(opt.tx_repeat_max, std::max(opt.tx_repeat_min, 1))),
          port_(opt.adaptive_feedback_port)
    {
        repeat_.store(min_repeat_, std::memory_order_release);
        decision_.repeat_count = min_repeat_;
        decision_.effective_rate = 0.5 / static_cast<double>(std::max(min_repeat_, 1));
#if defined(_WIN32)
        WSADATA wsa{};
        const int wsa_status = WSAStartup(MAKEWORD(2, 2), &wsa);
        if (wsa_status != 0) {
            throw std::runtime_error("WSAStartup failed for adaptive TX controller");
        }
        wsa_started_ = true;
        sock_ = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (sock_ == INVALID_SOCKET) {
            throw std::runtime_error("failed to create adaptive TX feedback socket");
        }
#else
        sock_ = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (sock_ < 0) {
            throw std::runtime_error("failed to create adaptive TX feedback socket");
        }
#endif
        sockaddr_in bind_addr{};
        bind_addr.sin_family = AF_INET;
        bind_addr.sin_port = htons(static_cast<uint16_t>(port_));
        bind_addr.sin_addr.s_addr = htonl(INADDR_ANY);
        if (::bind(sock_, reinterpret_cast<const sockaddr*>(&bind_addr), sizeof(bind_addr)) != 0) {
            throw std::runtime_error("failed to bind adaptive TX feedback UDP port");
        }
        std::cout << "[ADAPT-TX] enabled: listenPort=" << port_
                  << " repeat=" << min_repeat_ << ".." << max_repeat_ << "\n";
        worker_ = std::thread([this]() { run(); });
    }

    ~AdaptiveTxController()
    {
        stop_.store(true, std::memory_order_release);
        if (worker_.joinable()) {
            worker_.join();
        }
#if defined(_WIN32)
        if (sock_ != INVALID_SOCKET) {
            closesocket(sock_);
            sock_ = INVALID_SOCKET;
        }
        if (wsa_started_) {
            WSACleanup();
        }
#else
        if (sock_ >= 0) {
            close(sock_);
            sock_ = -1;
        }
#endif
    }

    int repeat_count() const
    {
        return repeat_.load(std::memory_order_acquire);
    }

    uint64_t decision_generation() const
    {
        return decision_generation_.load(std::memory_order_acquire);
    }

    usrp_link::AmcDecision current_decision() const
    {
        std::lock_guard<std::mutex> lock(decision_mutex_);
        return decision_;
    }

private:
    static bool parse_int_key(const std::string& msg, const std::string& key, int& out)
    {
        const auto pos = msg.find(key);
        if (pos == std::string::npos) {
            return false;
        }
        try {
            out = std::stoi(msg.substr(pos + key.size()));
            return true;
        } catch (...) {
            return false;
        }
    }

    static bool parse_double_key(const std::string& msg, const std::string& key, double& out)
    {
        const auto pos = msg.find(key);
        if (pos == std::string::npos) {
            return false;
        }
        try {
            out = std::stod(msg.substr(pos + key.size()));
            return true;
        } catch (...) {
            return false;
        }
    }

    static bool parse_string_key(const std::string& msg, const std::string& key, std::string& out)
    {
        const auto pos = msg.find(key);
        if (pos == std::string::npos) {
            return false;
        }
        size_t first = pos + key.size();
        while (first < msg.size() && std::isspace(static_cast<unsigned char>(msg[first]))) {
            ++first;
        }
        size_t last = first;
        while (last < msg.size() && !std::isspace(static_cast<unsigned char>(msg[last]))) {
            ++last;
        }
        out = msg.substr(first, last - first);
        return !out.empty();
    }

    void run()
    {
        while (!stop_.load(std::memory_order_acquire)) {
            fd_set readfds;
            FD_ZERO(&readfds);
            FD_SET(sock_, &readfds);
            timeval tv{};
            tv.tv_sec = 0;
            tv.tv_usec = 200000;
            const int ready = ::select(static_cast<int>(sock_ + 1), &readfds, nullptr, nullptr, &tv);
            if (ready <= 0 || !FD_ISSET(sock_, &readfds)) {
                continue;
            }
            char buf[256] = {};
#if defined(_WIN32)
            const int n = ::recvfrom(sock_, buf, static_cast<int>(sizeof(buf) - 1), 0, nullptr, nullptr);
            if (n == SOCKET_ERROR) {
                continue;
            }
#else
            const ssize_t n = ::recvfrom(sock_, buf, sizeof(buf) - 1, 0, nullptr, nullptr);
            if (n <= 0) {
                continue;
            }
#endif
            buf[n] = '\0';
            parse_feedback(std::string(buf));
        }
    }

    void parse_feedback(const std::string& msg)
    {
        int requested = 0;
        if (!parse_int_key(msg, "repeat=", requested)) {
            return;
        }
        requested = std::max(min_repeat_, std::min(max_repeat_, requested));
        const int old = repeat_.exchange(requested, std::memory_order_acq_rel);
        {
            std::lock_guard<std::mutex> lock(decision_mutex_);
            decision_.repeat_count = requested;
            decision_.effective_rate = 0.5 / static_cast<double>(std::max(requested, 1));
            int mcs = static_cast<int>(decision_.mcs);
            if (parse_int_key(msg, "mcs=", mcs)) {
                decision_.mcs = static_cast<usrp_link::McsId>(std::max(0, std::min(5, mcs)));
            }
            parse_string_key(msg, "modulation=", decision_.modulation);
            parse_string_key(msg, "fec=", decision_.fec_profile);
            parse_double_key(msg, "effRate=", decision_.effective_rate);
        }
        decision_generation_.fetch_add(1, std::memory_order_acq_rel);
        if (requested != old) {
            std::cout << "[ADAPT-TX] repeat " << old << " -> " << requested
                      << " feedback=\"" << msg.substr(0, msg.find('\n')) << "\"\n";
        }
    }

    int min_repeat_ = 1;
    int max_repeat_ = 3;
    int port_ = 65435;
    std::atomic<int> repeat_{1};
    mutable std::mutex decision_mutex_;
    usrp_link::AmcDecision decision_;
    std::atomic<uint64_t> decision_generation_{0};
    std::atomic<bool> stop_{false};
    std::thread worker_;
#if defined(_WIN32)
    SOCKET sock_ = INVALID_SOCKET;
    bool wsa_started_ = false;
#else
    int sock_ = -1;
#endif
};

class AdaptiveRxDecision {
public:
    explicit AdaptiveRxDecision(const Options& opt)
        : window_frames_(std::max(opt.adaptive_window_frames, 1)),
          min_repeat_(std::max(opt.tx_repeat_min, 1)),
          max_repeat_(std::max(opt.tx_repeat_max, std::max(opt.tx_repeat_min, 1)))
    {}

    bool observe(const FrameDecodeResult& r, int& repeat, double& avg_snr, double& fer, int& frames)
    {
        ++frames_;
        if (r.frame_ok) {
            ++ok_;
        }
        snr_sum_ += r.snr_db;
        if (frames_ < window_frames_) {
            return false;
        }
        avg_snr = snr_sum_ / static_cast<double>(frames_);
        fer = static_cast<double>(frames_ - ok_) / static_cast<double>(frames_);
        repeat = choose_repeat(avg_snr, fer);
        frames = frames_;
        frames_ = 0;
        ok_ = 0;
        snr_sum_ = 0.0;
        return true;
    }

private:
    int choose_repeat(double avg_snr, double fer) const
    {
        if (fer >= 0.05 || avg_snr < 12.5) {
            return max_repeat_;
        }
        if (fer >= 0.01 || avg_snr < 14.0) {
            return std::min(max_repeat_, std::max(min_repeat_, 2));
        }
        return min_repeat_;
    }

    int window_frames_ = 20;
    int min_repeat_ = 1;
    int max_repeat_ = 3;
    int frames_ = 0;
    int ok_ = 0;
    double snr_sum_ = 0.0;
};

struct UiMetricsSample {
    double frames = 0.0;
    double total_frames = 0.0;
    double ber = 0.0;
    bool ber_valid = false;
    double fer = 0.0;
    double preber = 0.0;
    double snr_db = 0.0;
    double relative_power_db = std::numeric_limits<double>::quiet_NaN();
    double goodput_mbps = 0.0;
    int mcs_index = -1;
};

static int static_ui_mcs_index(const std::string& modulation)
{
    if (modulation == "bpsk") {
        return 0;
    }
    if (modulation == "qpsk") {
        return 3;
    }
    if (modulation == "16qam") {
        return 5;
    }
    if (modulation == "64qam") {
        return 6;
    }
    return 0;
}

class UiTelemetryWorker {
public:
    UiTelemetryWorker(const Options& opt, const PhyConfig& phy)
    {
        constellation_interval_ns_ = interval_to_ns(std::chrono::milliseconds(std::max(opt.ui_constellation_interval_ms, 1)));
        spectrum_interval_ns_ = interval_to_ns(std::chrono::milliseconds(std::max(opt.ui_spectrum_interval_ms, 1)));
        const int metric_interval_ms = opt.ui_metrics && opt.ui_ntn
            ? std::min(opt.ui_metrics_interval_ms, opt.ui_ntn_interval_ms)
            : (opt.ui_ntn ? opt.ui_ntn_interval_ms : opt.ui_metrics_interval_ms);
        metrics_interval_ns_ = interval_to_ns(std::chrono::milliseconds(std::max(metric_interval_ms, 1)));
        default_mcs_index_ = static_ui_mcs_index(opt.modulation);
        if (opt.ui_constellation) {
            constellation_sink_ = std::make_unique<UdpConstellationSink>(
                opt.ui_constellation_host,
                opt.ui_constellation_port,
                opt.ui_constellation_points,
                opt.ui_constellation_interval_ms);
            constellation_points_ = std::max(opt.ui_constellation_points, 1);
        }
        if (opt.ui_spectrum) {
            spectrum_sink_ = std::make_unique<UdpSpectrumSink>(
                opt.ui_spectrum_host,
                opt.ui_spectrum_port,
                opt.ui_spectrum_interval_ms);
            spectrum_fft_size_ = phy.nfft;
            const double df = phy.rate / static_cast<double>(phy.nfft);
            spectrum_freq_hz_.resize(static_cast<size_t>(spectrum_fft_size_), 0.0f);
            for (int bin = 0; bin < spectrum_fft_size_; ++bin) {
                spectrum_freq_hz_[static_cast<size_t>(bin)] =
                    static_cast<float>((bin - spectrum_fft_size_ / 2) * df);
            }
        }
        if (opt.ui_metrics) {
            metrics_sink_ = std::make_unique<UdpMetricsSink>(
                opt.ui_metrics_host,
                opt.ui_metrics_port,
                opt.ui_metrics_interval_ms);
        }
        if (opt.ui_ntn) {
            ntn_metrics_sink_ = std::make_unique<UdpNtnMetricsSink>(
                opt.ui_ntn_host,
                opt.ui_ntn_port,
                opt.ui_ntn_interval_ms);
        }
        worker_ = std::thread([this]() { run(); });
    }

    ~UiTelemetryWorker()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    int max_constellation_points() const
    {
        return constellation_sink_ ? constellation_points_ : 0;
    }

    bool try_begin_constellation_capture()
    {
        return try_begin_capture(
            constellation_sink_ != nullptr,
            next_constellation_capture_ns_,
            constellation_capture_pending_);
    }

    bool try_begin_spectrum_capture()
    {
        return try_begin_capture(
            spectrum_sink_ != nullptr,
            next_spectrum_capture_ns_,
            spectrum_capture_pending_);
    }

    bool try_begin_metrics_sample()
    {
        return try_begin_capture(
            metrics_sink_ != nullptr || ntn_metrics_sink_ != nullptr,
            next_metrics_capture_ns_,
            metrics_capture_pending_);
    }

    void submit_constellation(std::vector<cf32> tx, std::vector<cf32> rx)
    {
        if (!constellation_sink_ || tx.empty() || rx.empty()) {
            finish_capture(next_constellation_capture_ns_, constellation_capture_pending_, constellation_interval_ns_);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            latest_tx_points_ = std::move(tx);
            latest_rx_points_ = std::move(rx);
            has_constellation_ = true;
        }
        cv_.notify_one();
    }

    void submit_spectrum_iq(std::vector<cf32> iq_samples)
    {
        if (!spectrum_sink_ || iq_samples.empty()) {
            finish_capture(next_spectrum_capture_ns_, spectrum_capture_pending_, spectrum_interval_ns_);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            latest_spectrum_iq_ = std::move(iq_samples);
            has_spectrum_ = true;
        }
        cv_.notify_one();
    }

    void submit_metrics(const UiMetricsSample& sample)
    {
        if (!metrics_sink_ && !ntn_metrics_sink_) {
            finish_capture(next_metrics_capture_ns_, metrics_capture_pending_, metrics_interval_ns_);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            latest_metrics_ = sample;
            has_metrics_ = true;
        }
        cv_.notify_one();
    }

private:
    static int64_t interval_to_ns(std::chrono::milliseconds interval)
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(interval).count();
    }

    static bool try_begin_capture(
        bool enabled,
        std::atomic<int64_t>& next_capture_ns,
        std::atomic<bool>& pending)
    {
        if (!enabled) {
            return false;
        }
        if (ui_now_ns() < next_capture_ns.load(std::memory_order_relaxed)) {
            return false;
        }
        bool expected = false;
        return pending.compare_exchange_strong(expected, true, std::memory_order_acq_rel);
    }

    static void finish_capture(
        std::atomic<int64_t>& next_capture_ns,
        std::atomic<bool>& pending,
        int64_t interval_ns)
    {
        next_capture_ns.store(ui_now_ns() + interval_ns, std::memory_order_release);
        pending.store(false, std::memory_order_release);
    }

    static std::vector<float> compute_welch_psd_db(const std::vector<cf32>& samples, int fft_size)
    {
        if (samples.empty() || fft_size <= 0) {
            return {};
        }

        std::vector<float> window(static_cast<size_t>(fft_size), 1.0f);
        double window_power = 0.0;
        if (fft_size > 1) {
            for (int i = 0; i < fft_size; ++i) {
                const float w = static_cast<float>(
                    0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(i) / static_cast<double>(fft_size - 1)));
                window[static_cast<size_t>(i)] = w;
                window_power += static_cast<double>(w) * static_cast<double>(w);
            }
        } else {
            window_power = 1.0;
        }
        window_power = std::max(window_power, 1e-12);

        std::vector<double> accum(static_cast<size_t>(fft_size), 0.0);
        std::vector<cf32> segment(static_cast<size_t>(fft_size), cf32(0.0f, 0.0f));
        const size_t hop = static_cast<size_t>(std::max(1, fft_size / 2));
        size_t segments = 0;

        auto add_segment = [&](size_t start, size_t valid) {
            std::fill(segment.begin(), segment.end(), cf32(0.0f, 0.0f));
            for (size_t i = 0; i < valid; ++i) {
                segment[i] = samples[start + i] * window[i];
            }
            const auto spectrum = fft_to_shifted_grid(segment.data(), fft_size);
            for (int i = 0; i < fft_size; ++i) {
                accum[static_cast<size_t>(i)] += static_cast<double>(std::norm(spectrum[static_cast<size_t>(i)]));
            }
            ++segments;
        };

        if (samples.size() <= static_cast<size_t>(fft_size)) {
            add_segment(0, samples.size());
        } else {
            for (size_t start = 0; start + static_cast<size_t>(fft_size) <= samples.size(); start += hop) {
                add_segment(start, static_cast<size_t>(fft_size));
            }
        }

        std::vector<double> power(accum.size(), 0.0);
        double max_power = 1e-20;
        const double norm = window_power * static_cast<double>(std::max<size_t>(segments, 1));
        for (size_t i = 0; i < accum.size(); ++i) {
            power[i] = std::max(accum[i] / norm, 1e-20);
            max_power = std::max(max_power, power[i]);
        }

        std::vector<float> psd_db(power.size(), -200.0f);
        for (size_t i = 0; i < power.size(); ++i) {
            psd_db[i] = static_cast<float>(10.0 * std::log10(power[i] / max_power));
        }
        return psd_db;
    }

    void run()
    {
        while (true) {
            std::vector<cf32> tx_points;
            std::vector<cf32> rx_points;
            std::vector<cf32> spectrum_iq;
            UiMetricsSample metrics;
            bool send_constellation = false;
            bool send_spectrum = false;
            bool send_metrics = false;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait_for(lock, std::chrono::milliseconds(5), [this]() {
                    return stop_ || has_constellation_ || has_spectrum_ || has_metrics_;
                });
                if (stop_ && !has_constellation_ && !has_spectrum_ && !has_metrics_) {
                    break;
                }
                if (constellation_sink_ && has_constellation_) {
                    tx_points = latest_tx_points_;
                    rx_points = latest_rx_points_;
                    has_constellation_ = false;
                    send_constellation = true;
                }
                if (spectrum_sink_ && has_spectrum_) {
                    spectrum_iq = latest_spectrum_iq_;
                    has_spectrum_ = false;
                    send_spectrum = true;
                }
                if ((metrics_sink_ || ntn_metrics_sink_) && has_metrics_) {
                    metrics = latest_metrics_;
                    has_metrics_ = false;
                    send_metrics = true;
                }
            }

            if (send_constellation) {
                constellation_sink_->send(tx_points, rx_points);
                finish_capture(next_constellation_capture_ns_, constellation_capture_pending_, constellation_interval_ns_);
            }
            if (send_spectrum) {
                const auto psd_db = compute_welch_psd_db(spectrum_iq, spectrum_fft_size_);
                spectrum_sink_->send(spectrum_freq_hz_, psd_db);
                finish_capture(next_spectrum_capture_ns_, spectrum_capture_pending_, spectrum_interval_ns_);
            }
            if (send_metrics) {
                if (metrics_sink_) {
                    metrics_sink_->send(
                        metrics.frames,
                        metrics.ber,
                        metrics.fer,
                        metrics.preber,
                        metrics.snr_db,
                        metrics.goodput_mbps);
                }
                if (ntn_metrics_sink_) {
                    const int mcs_index = metrics.mcs_index >= 0
                        ? metrics.mcs_index
                        : default_mcs_index_;
                    ntn_metrics_sink_->send(
                        metrics.frames,
                        metrics.total_frames,
                        metrics.ber,
                        metrics.ber_valid,
                        metrics.fer,
                        metrics.snr_db,
                        metrics.relative_power_db,
                        metrics.goodput_mbps,
                        mcs_index);
                }
                finish_capture(next_metrics_capture_ns_, metrics_capture_pending_, metrics_interval_ns_);
            }
        }
    }

    std::unique_ptr<UdpConstellationSink> constellation_sink_;
    std::unique_ptr<UdpSpectrumSink> spectrum_sink_;
    std::unique_ptr<UdpMetricsSink> metrics_sink_;
    std::unique_ptr<UdpNtnMetricsSink> ntn_metrics_sink_;
    int constellation_points_ = 0;
    int default_mcs_index_ = 0;
    int64_t constellation_interval_ns_ = 50'000'000;
    int64_t spectrum_interval_ns_ = 100'000'000;
    int64_t metrics_interval_ns_ = 200'000'000;
    std::vector<float> spectrum_freq_hz_;
    int spectrum_fft_size_ = 0;
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
    bool has_constellation_ = false;
    bool has_spectrum_ = false;
    bool has_metrics_ = false;
    std::vector<cf32> latest_tx_points_;
    std::vector<cf32> latest_rx_points_;
    std::vector<cf32> latest_spectrum_iq_;
    UiMetricsSample latest_metrics_;
    std::atomic<int64_t> next_constellation_capture_ns_{0};
    std::atomic<int64_t> next_spectrum_capture_ns_{0};
    std::atomic<int64_t> next_metrics_capture_ns_{0};
    std::atomic<bool> constellation_capture_pending_{false};
    std::atomic<bool> spectrum_capture_pending_{false};
    std::atomic<bool> metrics_capture_pending_{false};
};

static UiTelemetryWorker* g_ui_telemetry = nullptr;

static void submit_ui_spectrum_iq_if_due(const std::vector<cf32>& samples)
{
    if (g_ui_telemetry != nullptr &&
        g_ui_telemetry->try_begin_spectrum_capture()) {
        g_ui_telemetry->submit_spectrum_iq(std::vector<cf32>(samples.begin(), samples.end()));
    }
}

static std::vector<uint8_t> random_bits(size_t count, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> bit(0, 1);
    std::vector<uint8_t> out(count);
    for (auto& b : out) {
        b = static_cast<uint8_t>(bit(rng));
    }
    return out;
}

static cf32 bits_to_qpsk(uint8_t b0, uint8_t b1)
{
    const float s = static_cast<float>(1.0 / std::sqrt(2.0));
    const float i = b0 ? -s : s;
    const float q = b1 ? -s : s;
    return {i, q};
}

static std::vector<cf32> bits_to_qpsk(const std::vector<uint8_t>& bits)
{
    if (bits.size() % 2 != 0) {
        throw std::runtime_error("QPSK mapper needs an even number of bits");
    }
    std::vector<cf32> sym(bits.size() / 2);
    for (size_t i = 0, k = 0; i < bits.size(); i += 2, ++k) {
        sym[k] = bits_to_qpsk(bits[i], bits[i + 1]);
    }
    return sym;
}

static int bits_to_int_msb(const uint8_t* bits, int count)
{
    int v = 0;
    for (int i = 0; i < count; ++i) {
        v = (v << 1) | (bits[i] & 1u);
    }
    return v;
}

static int gray_to_binary(int g)
{
    int b = 0;
    for (; g != 0; g >>= 1) {
        b ^= g;
    }
    return b;
}

static std::vector<uint8_t> int_to_bits_msb(int v, int count)
{
    std::vector<uint8_t> bits(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        bits[static_cast<size_t>(i)] = static_cast<uint8_t>((v >> (count - 1 - i)) & 1);
    }
    return bits;
}

static void configure_modulation(PhyConfig& cfg, const std::string& modulation)
{
    cfg.modulation = modulation;
    cfg.constellation.clear();
    cfg.constellation_bits.clear();

    if (modulation == "bpsk") {
        cfg.bits_per_symbol = 1;
        cfg.constellation = {cf32(1.0f, 0.0f), cf32(-1.0f, 0.0f)};
    } else if (modulation == "qpsk" || modulation == "4qam") {
        cfg.modulation = "qpsk";
        cfg.bits_per_symbol = 2;
        cfg.constellation = {
            bits_to_qpsk(0, 0),
            bits_to_qpsk(0, 1),
            bits_to_qpsk(1, 0),
            bits_to_qpsk(1, 1),
        };
    } else if (modulation == "16qam" || modulation == "qam16" ||
               modulation == "64qam" || modulation == "qam64") {
        const bool is16 = (modulation == "16qam" || modulation == "qam16");
        cfg.modulation = is16 ? "16qam" : "64qam";
        cfg.bits_per_symbol = is16 ? 4 : 6;
        const int axis_bits = cfg.bits_per_symbol / 2;
        const int levels = 1 << axis_bits;
        const int m = 1 << cfg.bits_per_symbol;
        cfg.constellation.reserve(static_cast<size_t>(m));
        double avg_power = 0.0;
        for (int idx = 0; idx < m; ++idx) {
            const auto bits = int_to_bits_msb(idx, cfg.bits_per_symbol);
            const int gi = bits_to_int_msb(bits.data(), axis_bits);
            const int gq = bits_to_int_msb(bits.data() + axis_bits, axis_bits);
            const int bi = gray_to_binary(gi);
            const int bq = gray_to_binary(gq);
            const float i = static_cast<float>((levels - 1) - 2 * bi);
            const float q = static_cast<float>((levels - 1) - 2 * bq);
            cfg.constellation.emplace_back(i, q);
            avg_power += static_cast<double>(i * i + q * q);
        }
        const float scale = static_cast<float>(1.0 / std::sqrt(avg_power / static_cast<double>(m)));
        for (auto& s : cfg.constellation) {
            s *= scale;
        }
    } else {
        throw std::runtime_error("unsupported modulation: " + modulation + " (use bpsk|qpsk|16qam|64qam)");
    }

    const int m = 1 << cfg.bits_per_symbol;
    cfg.constellation_bits.reserve(static_cast<size_t>(m));
    for (int idx = 0; idx < m; ++idx) {
        cfg.constellation_bits.push_back(int_to_bits_msb(idx, cfg.bits_per_symbol));
    }
}

static std::vector<cf32> bits_to_symbols(const PhyConfig& phy, const std::vector<uint8_t>& bits)
{
    if (bits.size() % static_cast<size_t>(phy.bits_per_symbol) != 0) {
        throw std::runtime_error("mapper bit count is not divisible by bits/symbol");
    }
    std::vector<cf32> sym(bits.size() / static_cast<size_t>(phy.bits_per_symbol));
    for (size_t i = 0, k = 0; i < bits.size(); i += static_cast<size_t>(phy.bits_per_symbol), ++k) {
        const int idx = bits_to_int_msb(bits.data() + i, phy.bits_per_symbol);
        sym[k] = phy.constellation[static_cast<size_t>(idx)];
    }
    return sym;
}

static cf32 nearest_constellation_symbol(const PhyConfig& phy, cf32 x)
{
    int best = 0;
    float best_d2 = std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < phy.constellation.size(); ++i) {
        const float d2 = std::norm(x - phy.constellation[i]);
        if (d2 < best_d2) {
            best_d2 = d2;
            best = static_cast<int>(i);
        }
    }
    return phy.constellation[static_cast<size_t>(best)];
}

static void submit_ui_constellation_from_host_frame(
    const std::vector<cf32>& rx_frame,
    const PhyConfig& phy,
    double cfo_hz)
{
    if (g_ui_telemetry == nullptr) {
        return;
    }
    const int max_points = g_ui_telemetry->max_constellation_points();
    if (max_points <= 0 || static_cast<int>(rx_frame.size()) != phy.frame_len()) {
        g_ui_telemetry->submit_constellation({}, {});
        return;
    }

    std::vector<cf32> frame = rx_frame;
    for (size_t n = 0; n < frame.size(); ++n) {
        const double a = -2.0 * kPi * cfo_hz * static_cast<double>(n) / phy.rate;
        frame[n] *= cf32(static_cast<float>(std::cos(a)), static_cast<float>(std::sin(a)));
    }

    std::vector<std::vector<cf32>> y(
        static_cast<size_t>(phy.num_symbols),
        std::vector<cf32>(static_cast<size_t>(phy.active_sc)));
    size_t off = phy.preamble.size();
    for (int sym = 0; sym < phy.num_symbols; ++sym) {
        const cf32* td = frame.data() + off + static_cast<size_t>(phy.cp);
        const auto shifted = fft_to_shifted_grid(td, phy.nfft);
        for (int sc = 0; sc < phy.active_sc; ++sc) {
            y[sym][sc] = shifted[phy.used_indices[sc]];
        }
        off += static_cast<size_t>(phy.sym_len());
    }

    std::vector<cf32> yd;
    std::vector<cf32> hd;
    yd.reserve(static_cast<size_t>(phy.active_sc * phy.data_symbols.size()));
    hd.reserve(static_cast<size_t>(phy.active_sc * phy.data_symbols.size()));
    if (phy.rx_channel_estimation == "none") {
        for (int sym : phy.data_symbols) {
            for (int sc = 0; sc < phy.active_sc; ++sc) {
                yd.push_back(y[sym][sc]);
                hd.push_back(cf32(1.0f, 0.0f));
            }
        }
    } else {
        std::vector<std::vector<cf32>> hp(
            phy.pilot_symbols.size(),
            std::vector<cf32>(static_cast<size_t>(phy.active_sc)));
        for (size_t p = 0; p < phy.pilot_symbols.size(); ++p) {
            const int sym = phy.pilot_symbols[p];
            for (int sc = 0; sc < phy.active_sc; ++sc) {
                const cf32 pilot = phy.pilot_grid[p][sc];
                hp[p][sc] = y[sym][sc] * std::conj(pilot) / std::max(std::norm(pilot), 1e-9f);
            }
        }
        for (int sym : phy.data_symbols) {
            size_t hi = 0;
            while (hi + 1 < phy.pilot_symbols.size() && sym > phy.pilot_symbols[hi + 1]) {
                ++hi;
            }
            size_t p0 = hi;
            size_t p1 = std::min(hi + 1, phy.pilot_symbols.size() - 1);
            if (sym <= phy.pilot_symbols.front()) {
                p0 = p1 = 0;
            } else if (sym >= phy.pilot_symbols.back()) {
                p0 = p1 = phy.pilot_symbols.size() - 1;
            }
            double alpha = 0.0;
            if (p0 != p1) {
                alpha = static_cast<double>(sym - phy.pilot_symbols[p0]) /
                        static_cast<double>(phy.pilot_symbols[p1] - phy.pilot_symbols[p0]);
            }
            for (int sc = 0; sc < phy.active_sc; ++sc) {
                const cf32 h =
                    hp[p0][sc] * static_cast<float>(1.0 - alpha) +
                    hp[p1][sc] * static_cast<float>(alpha);
                yd.push_back(y[sym][sc]);
                hd.push_back(h);
            }
        }
    }

    const size_t points = std::min(static_cast<size_t>(max_points), yd.size());
    std::vector<cf32> tx_points;
    std::vector<cf32> rx_points;
    tx_points.reserve(points);
    rx_points.reserve(points);
    for (size_t j = 0; j < points; ++j) {
        const size_t i = (points <= 1 || yd.size() <= 1)
            ? 0
            : (j * (yd.size() - 1)) / (points - 1);
        const cf32 eq = yd[i] * std::conj(hd[i]) / std::max(std::norm(hd[i]), 1e-9f);
        tx_points.push_back(nearest_constellation_symbol(phy, eq));
        rx_points.push_back(eq);
    }
    g_ui_telemetry->submit_constellation(std::move(tx_points), std::move(rx_points));
}

static void submit_ui_constellation_from_host_frame_if_due(
    const std::vector<cf32>& rx_frame,
    const PhyConfig& phy,
    double cfo_hz)
{
    if (g_ui_telemetry != nullptr &&
        g_ui_telemetry->try_begin_constellation_capture()) {
        submit_ui_constellation_from_host_frame(rx_frame, phy, cfo_hz);
    }
}

static void symbol_to_llr_maxlog(const PhyConfig& phy, cf32 x, double sigma2, float* out)
{
    sigma2 = std::max(sigma2, 1e-8);
    for (int bit = 0; bit < phy.bits_per_symbol; ++bit) {
        float d0 = std::numeric_limits<float>::infinity();
        float d1 = std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < phy.constellation.size(); ++i) {
            const float d2 = std::norm(x - phy.constellation[i]);
            if (phy.constellation_bits[i][static_cast<size_t>(bit)] == 0) {
                d0 = std::min(d0, d2);
            } else {
                d1 = std::min(d1, d2);
            }
        }
        out[bit] = static_cast<float>((d1 - d0) / sigma2);
    }
}

static uint32_t bits_to_u32_msb(const std::vector<uint8_t>& bits, size_t off)
{
    uint32_t v = 0;
    for (int i = 0; i < 32; ++i) {
        v = (v << 1) | (bits[off + i] & 1u);
    }
    return v;
}

static void put_u32_msb(std::vector<uint8_t>& bits, size_t off, uint32_t v)
{
    for (int i = 0; i < 32; ++i) {
        bits[off + i] = static_cast<uint8_t>((v >> (31 - i)) & 1u);
    }
}

static uint16_t read_u16_le(const uint8_t* p)
{
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

static uint32_t read_u32_le(const uint8_t* p)
{
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

static uint64_t read_u64_le(const uint8_t* p)
{
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) {
        v = (v << 8) | p[i];
    }
    return v;
}

static void put_u16_le(std::vector<uint8_t>& b, size_t off, uint16_t v)
{
    b[off + 0] = static_cast<uint8_t>(v & 0xffu);
    b[off + 1] = static_cast<uint8_t>((v >> 8) & 0xffu);
}

static void put_u32_le(std::vector<uint8_t>& b, size_t off, uint32_t v)
{
    b[off + 0] = static_cast<uint8_t>(v & 0xffu);
    b[off + 1] = static_cast<uint8_t>((v >> 8) & 0xffu);
    b[off + 2] = static_cast<uint8_t>((v >> 16) & 0xffu);
    b[off + 3] = static_cast<uint8_t>((v >> 24) & 0xffu);
}

static void put_u64_le(std::vector<uint8_t>& b, size_t off, uint64_t v)
{
    for (int i = 0; i < 8; ++i) {
        b[off + static_cast<size_t>(i)] = static_cast<uint8_t>((v >> (8 * i)) & 0xffu);
    }
}

static uint32_t crc32_bytes(const uint8_t* data, size_t len)
{
    static std::array<uint32_t, 256> table = {};
    static bool ready = false;
    if (!ready) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int j = 0; j < 8; ++j) {
                c = (c & 1u) ? (0xedb88320u ^ (c >> 1)) : (c >> 1);
            }
            table[i] = c;
        }
        ready = true;
    }

    uint32_t c = 0xffffffffu;
    for (size_t i = 0; i < len; ++i) {
        c = table[(c ^ data[i]) & 0xffu] ^ (c >> 8);
    }
    return c ^ 0xffffffffu;
}

static std::vector<uint8_t> bits_to_bytes_msb(const std::vector<uint8_t>& bits)
{
    std::vector<uint8_t> bytes((bits.size() + 7) / 8, 0);
    for (size_t i = 0; i < bits.size(); ++i) {
        bytes[i / 8] |= static_cast<uint8_t>((bits[i] & 1u) << (7 - (i % 8)));
    }
    return bytes;
}

static std::vector<uint8_t> bytes_to_bits_msb(const std::vector<uint8_t>& bytes, size_t bit_count)
{
    std::vector<uint8_t> bits(bit_count, 0);
    for (size_t i = 0; i < bit_count; ++i) {
        bits[i] = static_cast<uint8_t>((bytes[i / 8] >> (7 - (i % 8))) & 1u);
    }
    return bits;
}

static std::vector<uint8_t> read_binary_file(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open input file: " + path);
    }
    in.seekg(0, std::ios::end);
    const auto end = in.tellg();
    if (end < 0) {
        throw std::runtime_error("cannot determine file size: " + path);
    }
    std::vector<uint8_t> data(static_cast<size_t>(end));
    in.seekg(0, std::ios::beg);
    if (!data.empty()) {
        in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
        if (!in) {
            throw std::runtime_error("failed to read input file: " + path);
        }
    }
    return data;
}

static std::string default_output_file(const Options& opt)
{
    return opt.output_file.empty() ? std::string("rx_media_payload.bin") : opt.output_file;
}

static int popcount64(uint64_t x)
{
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt64(x));
#else
    return __builtin_popcountll(x);
#endif
}

static bool get_bit(const std::vector<uint64_t>& row, int col)
{
    return ((row[static_cast<size_t>(col) >> 6] >> (col & 63)) & 1ull) != 0;
}

static void flip_bit(std::vector<uint64_t>& row, int col)
{
    row[static_cast<size_t>(col) >> 6] ^= (1ull << (col & 63));
}

static void xor_row(std::vector<uint64_t>& dst, const std::vector<uint64_t>& src)
{
    for (size_t i = 0; i < dst.size(); ++i) {
        dst[i] ^= src[i];
    }
}

static Alist read_alist(const std::string& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open alist: " + path);
    }

    Alist a;
    int max_col_w = 0;
    int max_row_w = 0;
    in >> a.n >> a.m;
    in >> max_col_w >> max_row_w;
    if (a.n <= 0 || a.m <= 0) {
        throw std::runtime_error("invalid alist header");
    }

    std::vector<int> col_w(a.n), row_w(a.m);
    for (int i = 0; i < a.n; ++i) in >> col_w[i];
    for (int i = 0; i < a.m; ++i) in >> row_w[i];

    a.col_rows.assign(a.n, {});
    a.row_cols.assign(a.m, {});

    for (int c = 0; c < a.n; ++c) {
        for (int j = 0; j < max_col_w; ++j) {
            int r = 0;
            in >> r;
            if (j < col_w[c] && r > 0) {
                a.col_rows[c].push_back(r - 1);
            }
        }
    }

    for (int r = 0; r < a.m; ++r) {
        for (int j = 0; j < max_row_w; ++j) {
            int c = 0;
            in >> c;
            if (j < row_w[r] && c > 0) {
                a.row_cols[r].push_back(c - 1);
            }
        }
    }

    if (!in.good() && !in.eof()) {
        throw std::runtime_error("failed while reading alist: " + path);
    }

    return a;
}

static Alist alist_from_dvb_s2_csr(const fec::FecConfig& cfg)
{
    const auto* params = std::get_if<fec::DvbS2LdpcParams>(&cfg.params);
    if (params == nullptr) {
        throw std::runtime_error("current LDPC TX/RX bridge requires a DVB-S2 CSR FEC config");
    }

    const auto& h = params->h;
    if (h.rows <= 0 || h.cols <= 0 ||
        static_cast<int>(h.row_ptr.size()) != h.rows + 1 ||
        h.col_ind.size() != h.val.size()) {
        throw std::runtime_error("invalid DVB-S2 CSR matrix for legacy LDPC bridge");
    }

    Alist a;
    a.n = h.cols;
    a.m = h.rows;
    a.row_cols.resize(static_cast<size_t>(a.m));
    a.col_rows.resize(static_cast<size_t>(a.n));
    for (int r = 0; r < h.rows; ++r) {
        const int32_t begin = h.row_ptr[static_cast<size_t>(r)];
        const int32_t end = h.row_ptr[static_cast<size_t>(r + 1)];
        if (begin < 0 || end < begin || end > static_cast<int32_t>(h.col_ind.size())) {
            throw std::runtime_error("invalid CSR row_ptr while building legacy LDPC bridge");
        }
        auto& row = a.row_cols[static_cast<size_t>(r)];
        row.reserve(static_cast<size_t>(end - begin));
        for (int32_t idx = begin; idx < end; ++idx) {
            const int c = h.col_ind[static_cast<size_t>(idx)];
            if (c < 0 || c >= h.cols || h.val[static_cast<size_t>(idx)] == 0) {
                continue;
            }
            row.push_back(c);
            a.col_rows[static_cast<size_t>(c)].push_back(r);
        }
    }
    return a;
}

static LdpcCode build_ldpc(const Alist& a, bool systematic_front_info)
{
    LdpcCode code;
    code.n = a.n;
    code.m = a.m;
    code.words = (a.n + 63) / 64;
    code.row_cols = a.row_cols;
    code.col_rows = a.col_rows;

    auto finish_edge_graph = [&]() {
        int edge_id = 0;
        code.row_edges.assign(a.m, {});
        code.col_edges.assign(a.n, {});
        code.edge_col.clear();
        for (int r = 0; r < a.m; ++r) {
            for (int c : a.row_cols[r]) {
                code.edge_col.push_back(c);
                code.row_edges[r].push_back(edge_id);
                code.col_edges[c].push_back(edge_id);
                ++edge_id;
            }
        }
    };

    auto has_dvb_s2_accumulator_parity = [&]() {
        if (!systematic_front_info || a.n <= a.m) {
            return false;
        }
        const int k = a.n - a.m;
        for (int r = 0; r < a.m; ++r) {
            std::vector<int> parity_cols;
            for (int c : a.row_cols[static_cast<size_t>(r)]) {
                if (c >= k) {
                    parity_cols.push_back(c - k);
                }
            }
            std::sort(parity_cols.begin(), parity_cols.end());
            const std::vector<int> expected =
                (r == 0) ? std::vector<int>{0} : std::vector<int>{r - 1, r};
            if (parity_cols != expected) {
                return false;
            }
        }
        return true;
    };

    if (has_dvb_s2_accumulator_parity()) {
        code.encoder_kind = LdpcCode::EncoderKind::DvbS2Accumulator;
        code.rank = a.m;
        code.k = a.n - a.m;
        code.info_cols.resize(static_cast<size_t>(code.k));
        std::iota(code.info_cols.begin(), code.info_cols.end(), 0);
        code.pivot_cols.resize(static_cast<size_t>(code.m));
        std::iota(code.pivot_cols.begin(), code.pivot_cols.end(), code.k);
        finish_edge_graph();
        return code;
    }

    std::vector<std::vector<uint64_t>> rows(static_cast<size_t>(a.m), std::vector<uint64_t>(code.words, 0));
    for (int r = 0; r < a.m; ++r) {
        for (int c : a.row_cols[r]) {
            flip_bit(rows[r], c);
        }
    }

    int rank = 0;
    std::vector<int> pivot_cols;
    std::vector<int> col_order;
    col_order.reserve(static_cast<size_t>(a.n));
    if (systematic_front_info) {
        if (a.n <= a.m) {
            throw std::runtime_error("--systematic-front-info requires a code with n > m");
        }
        for (int col = a.n - a.m; col < a.n; ++col) {
            col_order.push_back(col);
        }
        for (int col = 0; col < a.n - a.m; ++col) {
            col_order.push_back(col);
        }
    } else {
        for (int col = 0; col < a.n; ++col) {
            col_order.push_back(col);
        }
    }

    for (int col : col_order) {
        if (rank >= a.m) {
            break;
        }
        int pivot = -1;
        for (int r = rank; r < a.m; ++r) {
            if (get_bit(rows[r], col)) {
                pivot = r;
                break;
            }
        }
        if (pivot < 0) {
            continue;
        }
        if (pivot != rank) {
            std::swap(rows[pivot], rows[rank]);
        }
        for (int r = 0; r < a.m; ++r) {
            if (r != rank && get_bit(rows[r], col)) {
                xor_row(rows[r], rows[rank]);
            }
        }
        pivot_cols.push_back(col);
        ++rank;
    }

    if (rank != a.m) {
        throw std::runtime_error("LDPC parity-check matrix is rank deficient for this encoder");
    }

    std::vector<uint8_t> is_pivot(static_cast<size_t>(a.n), 0);
    for (int c : pivot_cols) {
        is_pivot[c] = 1;
    }

    code.pivot_cols = std::move(pivot_cols);
    for (int c = 0; c < a.n; ++c) {
        if (!is_pivot[c]) {
            code.info_cols.push_back(c);
        }
    }

    code.rank = rank;
    code.k = static_cast<int>(code.info_cols.size());
    code.rref_rows = std::move(rows);

    finish_edge_graph();

    return code;
}

static std::vector<uint8_t> ldpc_encode(const LdpcCode& code, const uint8_t* info)
{
    std::vector<uint8_t> cw(static_cast<size_t>(code.n), 0);

    if (code.encoder_kind == LdpcCode::EncoderKind::DvbS2Accumulator) {
        if (code.k <= 0 || code.m <= 0 || code.k + code.m != code.n) {
            throw std::runtime_error("invalid DVB-S2 accumulator LDPC dimensions");
        }
        for (int i = 0; i < code.k; ++i) {
            cw[static_cast<size_t>(i)] = static_cast<uint8_t>(info[i] & 1u);
        }

        uint8_t previous_parity = 0;
        for (int r = 0; r < code.m; ++r) {
            uint8_t row_sum = 0;
            for (int c : code.row_cols[static_cast<size_t>(r)]) {
                if (c < code.k) {
                    row_sum ^= cw[static_cast<size_t>(c)];
                }
            }
            const uint8_t parity = static_cast<uint8_t>(row_sum ^ ((r == 0) ? 0u : previous_parity));
            cw[static_cast<size_t>(code.k + r)] = parity;
            previous_parity = parity;
        }
        return cw;
    }

    std::vector<uint64_t> cw_words(static_cast<size_t>(code.words), 0);

    for (int i = 0; i < code.k; ++i) {
        if (info[i] & 1u) {
            const int c = code.info_cols[i];
            cw[c] = 1;
            flip_bit(cw_words, c);
        }
    }

    for (int r = 0; r < code.rank; ++r) {
        int parity = 0;
        for (int w = 0; w < code.words; ++w) {
            parity ^= (popcount64(code.rref_rows[r][w] & cw_words[w]) & 1);
        }
        if (parity) {
            const int c = code.pivot_cols[r];
            cw[c] = 1;
            flip_bit(cw_words, c);
        }
    }

    return cw;
}

static bool check_syndrome(const LdpcCode& code, const std::vector<uint8_t>& bits)
{
    for (const auto& row : code.row_cols) {
        int s = 0;
        for (int c : row) {
            s ^= (bits[c] & 1u);
        }
        if (s) {
            return false;
        }
    }
    return true;
}

static DecodeResult ldpc_decode_min_sum(
    const LdpcCode& code,
    const float* llr,
    int max_iter,
    double normalization)
{
    const int edges = static_cast<int>(code.edge_col.size());
    std::vector<float> q(static_cast<size_t>(edges), 0.0f);
    std::vector<float> rmsg(static_cast<size_t>(edges), 0.0f);
    std::vector<float> app(static_cast<size_t>(code.n), 0.0f);
    std::vector<uint8_t> hard(static_cast<size_t>(code.n), 0);

    for (int e = 0; e < edges; ++e) {
        q[e] = llr[code.edge_col[e]];
    }

    int used_iter = 0;
    bool ok = false;

    for (int it = 0; it < max_iter; ++it) {
        used_iter = it + 1;

        for (int row = 0; row < code.m; ++row) {
            const auto& erow = code.row_edges[row];
            float min1 = std::numeric_limits<float>::max();
            float min2 = std::numeric_limits<float>::max();
            int min1_edge = -1;
            int sign_prod = 1;

            for (int e : erow) {
                const float v = q[e];
                if (v < 0.0f) {
                    sign_prod = -sign_prod;
                }
                const float av = std::abs(v);
                if (av < min1) {
                    min2 = min1;
                    min1 = av;
                    min1_edge = e;
                } else if (av < min2) {
                    min2 = av;
                }
            }

            for (int e : erow) {
                const int sign_without = (q[e] < 0.0f) ? -sign_prod : sign_prod;
                const float mag = (e == min1_edge) ? min2 : min1;
                rmsg[e] = static_cast<float>(normalization) * static_cast<float>(sign_without) * mag;
            }
        }

        for (int col = 0; col < code.n; ++col) {
            float v = llr[col];
            for (int e : code.col_edges[col]) {
                v += rmsg[e];
            }
            app[col] = v;
            hard[col] = (v < 0.0f) ? 1 : 0;
        }

        ok = check_syndrome(code, hard);
        if (ok) {
            break;
        }

        for (int col = 0; col < code.n; ++col) {
            for (int e : code.col_edges[col]) {
                q[e] = app[col] - rmsg[e];
            }
        }
    }

    DecodeResult out;
    out.code_bits = std::move(hard);
    out.info_bits.resize(static_cast<size_t>(code.k));
    for (int i = 0; i < code.k; ++i) {
        out.info_bits[i] = out.code_bits[code.info_cols[i]];
    }
    out.parity_ok = ok;
    out.iterations = used_iter;
    return out;
}

static bool ldpc_has_front_systematic_info(const LdpcCode& code)
{
    if (code.k <= 0 || code.m <= 0 || code.k + code.m != code.n) {
        return false;
    }
    if (static_cast<int>(code.info_cols.size()) != code.k ||
        static_cast<int>(code.pivot_cols.size()) != code.m) {
        return false;
    }
    for (int i = 0; i < code.k; ++i) {
        if (code.info_cols[static_cast<size_t>(i)] != i) {
            return false;
        }
    }
    std::vector<uint8_t> seen(static_cast<size_t>(code.m), 0);
    for (int c : code.pivot_cols) {
        if (c < code.k || c >= code.n) {
            return false;
        }
        seen[static_cast<size_t>(c - code.k)] = 1;
    }
    return std::all_of(seen.begin(), seen.end(), [](uint8_t v) { return v != 0; });
}

static std::vector<unsigned char> build_dense_h_for_cuda_osd(const LdpcCode& code)
{
    std::vector<unsigned char> h(static_cast<size_t>(code.m) * static_cast<size_t>(code.n), 0);
    for (int r = 0; r < code.m; ++r) {
        for (int c : code.row_cols[static_cast<size_t>(r)]) {
            if (c >= 0 && c < code.n) {
                h[static_cast<size_t>(r) * static_cast<size_t>(code.n) + static_cast<size_t>(c)] = 1;
            }
        }
    }
    return h;
}

static std::vector<unsigned long long> build_systematic_parity_for_cuda_osd(
    const LdpcCode& code,
    int& parity_words_per_row)
{
    if (!ldpc_has_front_systematic_info(code)) {
        throw std::runtime_error(
            "CUDA BP-OSD dynamic matrices require systematic-front LDPC layout: information columns first, parity columns last. "
            "Pass --systematic-front-info or use a compatible alist.");
    }
    parity_words_per_row = (code.n - code.k + 63) / 64;
    std::vector<unsigned long long> parity(
        static_cast<size_t>(code.k) * static_cast<size_t>(parity_words_per_row),
        0ULL);
    std::vector<uint8_t> unit(static_cast<size_t>(code.k), 0);
    for (int row = 0; row < code.k; ++row) {
        std::fill(unit.begin(), unit.end(), 0);
        unit[static_cast<size_t>(row)] = 1;
        const auto cw = ldpc_encode(code, unit.data());
        for (int p = 0; p < code.n - code.k; ++p) {
            if (cw[static_cast<size_t>(code.k + p)] != 0) {
                parity[static_cast<size_t>(row) * static_cast<size_t>(parity_words_per_row) +
                       static_cast<size_t>(p / 64)] |=
                    (1ULL << (p % 64));
            }
        }
    }
    return parity;
}

struct CudaBpOsdDecoder {
    struct ReadyCodeword {
        uint64_t frame_id = 0;
        uint32_t block_index = 0;
        bool used_osd = false;
        std::vector<uint8_t> info_bits;
    };

    unsigned long long handle = 0;
    unsigned long long osd_handle = 0;
    int n = 0;
    int k = 0;
    int min_batch_codewords = 1;
    int max_batch_codewords = 1;
    uint64_t frame_sequence = 0;
    bool osd_only = false;
    float* d_async_llrs = nullptr;
    int async_codewords = 0;
    std::vector<unsigned long long> async_timestamps;
    std::vector<unsigned long long> async_frame_ids;
    std::vector<unsigned int> async_block_indices;

    CudaBpOsdDecoder(const Options& opt, const LdpcCode& code)
        : n(code.n), k(code.k)
    {
#ifdef HAVE_CUDA_OSD
        osd_only = (opt.decoder == "cuda-osd");
        constexpr int kCudaOsdMaxN = 256;
        constexpr int kCudaOsdMaxK = 128;
        if (code.n <= 0 || code.k <= 0 || code.m <= 0 || code.k + code.m != code.n) {
            throw std::runtime_error("CUDA BP-OSD requires a full-rank LDPC code with n = k + m");
        }
        if (code.n > kCudaOsdMaxN || code.k > kCudaOsdMaxK || (code.n - code.k) > 128) {
            std::ostringstream oss;
            oss << "CUDA BP-OSD dynamic backend currently supports short LDPC only: n <= "
                << kCudaOsdMaxN << ", k <= " << kCudaOsdMaxK
                << ", n-k <= 128; got n=" << code.n << " k=" << code.k
                << ". Use --decoder cuda-bp for longer LDPC codes.";
            throw std::runtime_error(oss.str());
        }
        if (!ldpc_has_front_systematic_info(code)) {
            throw std::runtime_error(
                "CUDA BP-OSD requires systematic-front LDPC layout. Add --systematic-front-info.");
        }

        if (osd_only) {
            if (code.n != 128 || code.k != 64 || code.m != 64) {
                throw std::runtime_error("CUDA OSD-only path is still specialized for CCSDS n128/k64; use cuda-bp-osd for dynamic short LDPC.");
            }
            cuda_osd_osd_config_t cfg{};
            cfg.n = code.n;
            cfg.k = code.k;
            cfg.max_batch_size = opt.cuda_max_batch;
            cfg.osd_threads = 256;

            const int status = cuda_osd_create_osd_decoder(&cfg, &osd_handle);
            if (status != CUDA_OSD_STATUS_OK || osd_handle == 0) {
                throw std::runtime_error(std::string("cuda_osd_create_osd_decoder failed: ") +
                                         cuda_osd_get_last_error());
            }
            std::cout << "[CUDA] OSD-only decoder enabled: n=" << code.n
                      << " k=" << code.k
                      << " maxBatch=" << cfg.max_batch_size << "\n";
            return;
        }

        cuda_osd_stream_config_t cfg{};
        cfg.n = code.n;
        cfg.k = code.k;
        cfg.m = code.m;
        cfg.max_batch_size = opt.cuda_max_batch;
        cfg.min_batch_size = opt.cuda_min_batch;
        cfg.max_latency_us = static_cast<unsigned long long>(std::max(opt.cuda_latency_us, 1));
        cfg.arrival_ewma_alpha = 0.15;
        cfg.default_interarrival_us = 150;
        cfg.bp_max_iterations = opt.ldpc_max_iter;
        cfg.bp_normalization = static_cast<float>(opt.ldpc_normalization);
        cfg.bp_offset = static_cast<float>(opt.ldpc_offset);
        cfg.bp_damping = static_cast<float>(opt.ldpc_damping);
        cfg.bp_min_abs_llr_accept = 1.5f;
        cfg.osd_threads = 256;
        min_batch_codewords = std::max(cfg.min_batch_size, 1);
        max_batch_codewords = std::max(cfg.max_batch_size, 1);

        int parity_words_per_row = 0;
        const auto dense_h = build_dense_h_for_cuda_osd(code);
        const auto systematic_parity =
            build_systematic_parity_for_cuda_osd(code, parity_words_per_row);

        const int status = cuda_osd_create_stream_decoder_with_matrices(
            &cfg,
            dense_h.data(),
            systematic_parity.data(),
            parity_words_per_row,
            &handle);
        if (status != CUDA_OSD_STATUS_OK || handle == 0) {
            throw std::runtime_error(std::string("cuda_osd_create_stream_decoder_with_matrices failed: ") +
                                     cuda_osd_get_last_error());
        }
        const cudaError_t alloc_status = cudaMalloc(
            &d_async_llrs,
            static_cast<size_t>(max_batch_codewords) * static_cast<size_t>(n) * sizeof(float));
        if (alloc_status != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc BP-OSD async LLR staging failed: ") +
                                     cudaGetErrorString(alloc_status));
        }
        std::cout << "[CUDA] BP-OSD decoder enabled: n=" << code.n
                  << " k=" << code.k
                  << " minBatch=" << cfg.min_batch_size
                  << " maxBatch=" << cfg.max_batch_size
                  << " latencyUs=" << cfg.max_latency_us
                  << " matrix=dynamic-systematic\n";
#else
        (void)opt;
        (void)code;
        throw std::runtime_error("this executable was built without CUDA BP-OSD support");
#endif
    }

    ~CudaBpOsdDecoder()
    {
#ifdef HAVE_CUDA_OSD
        if (handle != 0) {
            cuda_osd_destroy_stream_decoder(handle);
            handle = 0;
        }
        if (osd_handle != 0) {
            cuda_osd_destroy_osd_decoder(osd_handle);
            osd_handle = 0;
        }
        if (d_async_llrs != nullptr) {
            cudaFree(d_async_llrs);
            d_async_llrs = nullptr;
        }
#endif
    }

    int async_pending_codewords() const
    {
        return async_codewords;
    }

    int async_max_codewords() const
    {
        return max_batch_codewords;
    }

    uint64_t enqueue_device_frame_async(const float* device_llrs, int blocks)
    {
#ifdef HAVE_CUDA_OSD
        if (osd_only) {
            throw std::runtime_error("async device batching is implemented for stream BP-OSD, not OSD-only");
        }
        if (device_llrs == nullptr || blocks <= 0) {
            throw std::runtime_error("CUDA BP-OSD async enqueue received an empty frame");
        }
        if (blocks > max_batch_codewords || async_codewords + blocks > max_batch_codewords) {
            throw std::runtime_error("CUDA BP-OSD async staging batch is full");
        }
        if (d_async_llrs == nullptr) {
            throw std::runtime_error("CUDA BP-OSD async staging buffer was not allocated");
        }

        const uint64_t frame_id = frame_sequence++;
        const auto now = std::chrono::steady_clock::now().time_since_epoch();
        const auto now_us = static_cast<unsigned long long>(
            std::chrono::duration_cast<std::chrono::microseconds>(now).count());

        const size_t dst_offset = static_cast<size_t>(async_codewords) * static_cast<size_t>(n);
        const size_t copy_floats = static_cast<size_t>(blocks) * static_cast<size_t>(n);
        const cudaError_t copy_status = cudaMemcpy(
            d_async_llrs + dst_offset,
            device_llrs,
            copy_floats * sizeof(float),
            cudaMemcpyDeviceToDevice);
        if (copy_status != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpy BP-OSD async LLR staging failed: ") +
                                     cudaGetErrorString(copy_status));
        }

        for (int b = 0; b < blocks; ++b) {
            async_timestamps.push_back(now_us);
            async_frame_ids.push_back(static_cast<unsigned long long>(frame_id));
            async_block_indices.push_back(static_cast<unsigned int>(b));
        }
        async_codewords += blocks;
        return frame_id;
#else
        (void)device_llrs;
        (void)blocks;
        throw std::runtime_error("this executable was built without CUDA BP-OSD support");
#endif
    }

    uint64_t submit_async_batch()
    {
#ifdef HAVE_CUDA_OSD
        if (async_codewords <= 0) {
            return 0;
        }
        if (handle == 0 || d_async_llrs == nullptr) {
            throw std::runtime_error("CUDA BP-OSD async submit called before decoder initialization");
        }
        const auto t0 = std::chrono::steady_clock::now();
        const int status = cuda_osd_submit_device_codewords(
            handle,
            d_async_llrs,
            async_codewords,
            async_timestamps.data(),
            async_frame_ids.data(),
            async_block_indices.data());
        if (status != CUDA_OSD_STATUS_OK) {
            throw std::runtime_error(std::string("cuda_osd_submit_device_codewords async batch failed: ") +
                                     cuda_osd_get_last_error());
        }
        const auto t1 = std::chrono::steady_clock::now();
        async_codewords = 0;
        async_timestamps.clear();
        async_frame_ids.clear();
        async_block_indices.clear();
        return elapsed_ns(t0, t1);
#else
        throw std::runtime_error("this executable was built without CUDA BP-OSD support");
#endif
    }

    std::vector<ReadyCodeword> poll_ready_codewords(int max_codewords)
    {
#ifdef HAVE_CUDA_OSD
        if (max_codewords <= 0) {
            return {};
        }
        std::vector<uint8_t> tmp_info(static_cast<size_t>(max_codewords) * static_cast<size_t>(k), 0);
        std::vector<unsigned long long> out_timestamps(static_cast<size_t>(max_codewords), 0);
        std::vector<unsigned long long> out_frame_ids(static_cast<size_t>(max_codewords), 0);
        std::vector<unsigned int> out_block_indices(static_cast<size_t>(max_codewords), 0);
        std::vector<unsigned char> out_used_osd(static_cast<size_t>(max_codewords), 0);

        int ready = 0;
        const int status = cuda_osd_poll_ready(
            handle,
            max_codewords,
            tmp_info.data(),
            out_timestamps.data(),
            out_frame_ids.data(),
            out_block_indices.data(),
            out_used_osd.data(),
            &ready);
        if (status != CUDA_OSD_STATUS_OK) {
            throw std::runtime_error(std::string("cuda_osd_poll_ready async batch failed: ") +
                                     cuda_osd_get_last_error());
        }
        std::vector<ReadyCodeword> out;
        out.reserve(static_cast<size_t>(std::max(ready, 0)));
        for (int i = 0; i < ready; ++i) {
            ReadyCodeword item;
            item.frame_id = static_cast<uint64_t>(out_frame_ids[static_cast<size_t>(i)]);
            item.block_index = static_cast<uint32_t>(out_block_indices[static_cast<size_t>(i)]);
            item.used_osd = out_used_osd[static_cast<size_t>(i)] != 0;
            item.info_bits.assign(
                tmp_info.begin() + static_cast<std::ptrdiff_t>(i * k),
                tmp_info.begin() + static_cast<std::ptrdiff_t>((i + 1) * k));
            out.push_back(std::move(item));
        }
        (void)out_timestamps;
        return out;
#else
        (void)max_codewords;
        throw std::runtime_error("this executable was built without CUDA BP-OSD support");
#endif
    }

    std::vector<uint8_t> decode(const float* llrs, int blocks, int& used_osd_count)
    {
#ifdef HAVE_CUDA_OSD
        if (llrs == nullptr || blocks <= 0) {
            throw std::runtime_error("CUDA BP-OSD decoder received an empty frame");
        }

        if (osd_only) {
            std::vector<uint8_t> info(static_cast<size_t>(blocks * k), 0);
            std::vector<float> distances(static_cast<size_t>(blocks), 0.0f);
            double total_decode_ms = 0.0;
            double throughput_mbps = 0.0;
            const int status = cuda_osd_decode_osd_batch(
                osd_handle,
                llrs,
                blocks,
                info.data(),
                distances.data(),
                &total_decode_ms,
                &throughput_mbps);
            if (status != CUDA_OSD_STATUS_OK) {
                throw std::runtime_error(std::string("cuda_osd_decode_osd_batch failed: ") +
                                         cuda_osd_get_last_error());
            }
            used_osd_count = blocks;
            return info;
        }

        const auto now = std::chrono::steady_clock::now().time_since_epoch();
        const auto now_us = static_cast<unsigned long long>(
            std::chrono::duration_cast<std::chrono::microseconds>(now).count());
        std::vector<unsigned long long> timestamps(static_cast<size_t>(blocks), now_us);
        std::vector<unsigned long long> frame_ids(static_cast<size_t>(blocks), frame_sequence++);
        std::vector<unsigned int> block_indices(static_cast<size_t>(blocks), 0);
        for (int b = 0; b < blocks; ++b) {
            block_indices[static_cast<size_t>(b)] = static_cast<unsigned int>(b);
        }

        int status = cuda_osd_submit_codewords(
            handle,
            llrs,
            blocks,
            timestamps.data(),
            frame_ids.data(),
            block_indices.data());
        if (status != CUDA_OSD_STATUS_OK) {
            throw std::runtime_error(std::string("cuda_osd_submit_codewords failed: ") +
                                     cuda_osd_get_last_error());
        }

        status = cuda_osd_flush(handle);
        if (status != CUDA_OSD_STATUS_OK) {
            throw std::runtime_error(std::string("cuda_osd_flush failed: ") +
                                     cuda_osd_get_last_error());
        }

        std::vector<uint8_t> info(static_cast<size_t>(blocks * k), 0);
        std::vector<uint8_t> tmp_info(static_cast<size_t>(blocks * k), 0);
        std::vector<unsigned long long> out_timestamps(static_cast<size_t>(blocks), 0);
        std::vector<unsigned long long> out_frame_ids(static_cast<size_t>(blocks), 0);
        std::vector<unsigned int> out_block_indices(static_cast<size_t>(blocks), 0);
        std::vector<unsigned char> out_used_osd(static_cast<size_t>(blocks), 0);

        int decoded = 0;
        used_osd_count = 0;
        while (decoded < blocks) {
            int ready = 0;
            status = cuda_osd_poll_ready(
                handle,
                blocks,
                tmp_info.data(),
                out_timestamps.data(),
                out_frame_ids.data(),
                out_block_indices.data(),
                out_used_osd.data(),
                &ready);
            if (status != CUDA_OSD_STATUS_OK) {
                throw std::runtime_error(std::string("cuda_osd_poll_ready failed: ") +
                                         cuda_osd_get_last_error());
            }
            if (ready <= 0) {
                throw std::runtime_error("CUDA BP-OSD flush returned no ready codewords");
            }

            for (int i = 0; i < ready; ++i) {
                const unsigned int b = out_block_indices[static_cast<size_t>(i)];
                if (b >= static_cast<unsigned int>(blocks)) {
                    throw std::runtime_error("CUDA BP-OSD returned an invalid block index");
                }
                std::copy(
                    tmp_info.begin() + static_cast<std::ptrdiff_t>(i * k),
                    tmp_info.begin() + static_cast<std::ptrdiff_t>((i + 1) * k),
                    info.begin() + static_cast<std::ptrdiff_t>(b * k));
                used_osd_count += out_used_osd[static_cast<size_t>(i)] ? 1 : 0;
            }
            decoded += ready;
        }

        return info;
#else
        (void)llrs;
        (void)blocks;
        (void)used_osd_count;
        throw std::runtime_error("this executable was built without CUDA BP-OSD support");
#endif
    }

    std::vector<uint8_t> decode_device(const float* device_llrs, int blocks, int& used_osd_count)
    {
#ifdef HAVE_CUDA_OSD
        if (osd_only) {
            throw std::runtime_error("device-pointer decode is currently implemented for stream BP-OSD, not OSD-only");
        }
        if (device_llrs == nullptr || blocks <= 0) {
            throw std::runtime_error("CUDA BP-OSD device decoder received an empty frame");
        }

        const auto now = std::chrono::steady_clock::now().time_since_epoch();
        const auto now_us = static_cast<unsigned long long>(
            std::chrono::duration_cast<std::chrono::microseconds>(now).count());
        std::vector<unsigned long long> timestamps(static_cast<size_t>(blocks), now_us);
        std::vector<unsigned long long> frame_ids(static_cast<size_t>(blocks), frame_sequence++);
        std::vector<unsigned int> block_indices(static_cast<size_t>(blocks), 0);
        for (int b = 0; b < blocks; ++b) {
            block_indices[static_cast<size_t>(b)] = static_cast<unsigned int>(b);
        }

        int status = cuda_osd_submit_device_codewords(
            handle,
            device_llrs,
            blocks,
            timestamps.data(),
            frame_ids.data(),
            block_indices.data());
        if (status != CUDA_OSD_STATUS_OK) {
            throw std::runtime_error(std::string("cuda_osd_submit_device_codewords failed: ") +
                                     cuda_osd_get_last_error());
        }

        status = cuda_osd_flush(handle);
        if (status != CUDA_OSD_STATUS_OK) {
            throw std::runtime_error(std::string("cuda_osd_flush failed: ") +
                                     cuda_osd_get_last_error());
        }

        std::vector<uint8_t> info(static_cast<size_t>(blocks * k), 0);
        std::vector<uint8_t> tmp_info(static_cast<size_t>(blocks * k), 0);
        std::vector<unsigned long long> out_timestamps(static_cast<size_t>(blocks), 0);
        std::vector<unsigned long long> out_frame_ids(static_cast<size_t>(blocks), 0);
        std::vector<unsigned int> out_block_indices(static_cast<size_t>(blocks), 0);
        std::vector<unsigned char> out_used_osd(static_cast<size_t>(blocks), 0);

        int decoded = 0;
        used_osd_count = 0;
        while (decoded < blocks) {
            int ready = 0;
            status = cuda_osd_poll_ready(
                handle,
                blocks,
                tmp_info.data(),
                out_timestamps.data(),
                out_frame_ids.data(),
                out_block_indices.data(),
                out_used_osd.data(),
                &ready);
            if (status != CUDA_OSD_STATUS_OK) {
                throw std::runtime_error(std::string("cuda_osd_poll_ready failed: ") +
                                         cuda_osd_get_last_error());
            }
            if (ready <= 0) {
                throw std::runtime_error("CUDA BP-OSD device decode returned no ready codewords");
            }

            for (int i = 0; i < ready; ++i) {
                const unsigned int b = out_block_indices[static_cast<size_t>(i)];
                if (b >= static_cast<unsigned int>(blocks)) {
                    throw std::runtime_error("CUDA BP-OSD device decode returned an invalid block index");
                }
                std::copy(
                    tmp_info.begin() + static_cast<std::ptrdiff_t>(i * k),
                    tmp_info.begin() + static_cast<std::ptrdiff_t>((i + 1) * k),
                    info.begin() + static_cast<std::ptrdiff_t>(b * k));
                used_osd_count += out_used_osd[static_cast<size_t>(i)] ? 1 : 0;
            }
            decoded += ready;
        }

        return info;
#else
        (void)device_llrs;
        (void)blocks;
        (void)used_osd_count;
        throw std::runtime_error("this executable was built without CUDA BP-OSD support");
#endif
    }
};

struct CudaBpDecoder {
    unsigned long long handle = 0;
    int n = 0;
    int k = 0;

    CudaBpDecoder(const Options& opt, const LdpcCode& code, const fec::FecConfig& fec_config)
        : n(code.n), k(code.k)
    {
#ifdef HAVE_CUDA_OSD
        const auto* dvb = std::get_if<fec::DvbS2LdpcParams>(&fec_config.params);
        if (dvb == nullptr) {
            throw std::runtime_error("generic CUDA BP currently requires a Type-A CSR LDPC configuration");
        }
        if (dvb->h.rows != code.m || dvb->h.cols != code.n || dvb->h.nnz <= 0) {
            throw std::runtime_error("generic CUDA BP CSR dimensions do not match the active LDPC code");
        }

        cuda_osd_bp_csr_config_t cfg{};
        cfg.n = code.n;
        cfg.m = code.m;
        cfg.max_batch_size = opt.cuda_max_batch;
        cfg.max_iterations = opt.ldpc_max_iter;
        cfg.normalization = static_cast<float>(opt.ldpc_normalization);
        cfg.offset = static_cast<float>(opt.ldpc_offset);
        cfg.damping = static_cast<float>(opt.ldpc_damping);
        cfg.min_abs_llr_accept = 0.0f;
        cfg.nnz = dvb->h.nnz;
        cfg.schedule = (opt.ldpc_schedule >= 0) ? opt.ldpc_schedule : ((code.n >= 1000) ? 2 : 0);

        const int status = cuda_osd_create_bp_csr_decoder(
            &cfg,
            dvb->h.row_ptr.data(),
            dvb->h.col_ind.data(),
            &handle);
        if (status != CUDA_OSD_STATUS_OK || handle == 0) {
            throw std::runtime_error(std::string("cuda_osd_create_bp_csr_decoder failed: ") +
                                     cuda_osd_get_last_error());
        }
        std::cout << "[CUDA] "
                  << (cfg.schedule == 2 ? "DVB-S2 colored-layered NMS BP decoder enabled: n="
                      : (cfg.schedule == 1 ? "DVB-S2 node-parallel NMS BP decoder enabled: n="
                                           : "generic BP decoder enabled: n="))
                  << code.n
                  << " k=" << code.k
                  << " m=" << code.m
                  << " nnz=" << dvb->h.nnz
                  << " maxBatch=" << cfg.max_batch_size
                  << " maxIter=" << cfg.max_iterations
                  << " norm=" << cfg.normalization
                  << " offset=" << cfg.offset
                  << " damping=" << cfg.damping
                  << " schedule=" << cfg.schedule << "\n";
#else
        (void)opt;
        (void)code;
        (void)fec_config;
        throw std::runtime_error("this executable was built without generic CUDA BP support");
#endif
    }

    ~CudaBpDecoder()
    {
#ifdef HAVE_CUDA_OSD
        if (handle != 0) {
            cuda_osd_destroy_bp_csr_decoder(handle);
            handle = 0;
        }
#endif
    }

    std::vector<uint8_t> decode(const float* llrs, int blocks, int& success_count, double& total_decode_ms)
    {
#ifdef HAVE_CUDA_OSD
        if (llrs == nullptr || blocks <= 0) {
            throw std::runtime_error("generic CUDA BP decoder received an empty frame");
        }
        std::vector<unsigned char> code_bits(static_cast<size_t>(blocks * n), 0);
        std::vector<unsigned char> success(static_cast<size_t>(blocks), 0);
        total_decode_ms = 0.0;
        const int status = cuda_osd_decode_bp_csr_batch(
            handle,
            llrs,
            blocks,
            code_bits.data(),
            success.data(),
            &total_decode_ms);
        if (status != CUDA_OSD_STATUS_OK) {
            throw std::runtime_error(std::string("cuda_osd_decode_bp_csr_batch failed: ") +
                                     cuda_osd_get_last_error());
        }

        success_count = 0;
        for (unsigned char ok : success) {
            success_count += ok ? 1 : 0;
        }

        std::vector<uint8_t> info(static_cast<size_t>(blocks * k), 0);
        for (int b = 0; b < blocks; ++b) {
            for (int i = 0; i < k; ++i) {
                info[static_cast<size_t>(b * k + i)] =
                    code_bits[static_cast<size_t>(b * n + i)];
            }
        }
        return info;
#else
        (void)llrs;
        (void)blocks;
        (void)success_count;
        (void)total_decode_ms;
        throw std::runtime_error("this executable was built without generic CUDA BP support");
#endif
    }

    std::vector<uint8_t> decode_device(const float* device_llrs, int blocks, int& success_count, double& total_decode_ms)
    {
#ifdef HAVE_CUDA_OSD
        if (device_llrs == nullptr || blocks <= 0) {
            throw std::runtime_error("generic CUDA BP device decoder received an empty frame");
        }
        std::vector<unsigned char> code_bits(static_cast<size_t>(blocks * n), 0);
        std::vector<unsigned char> success(static_cast<size_t>(blocks), 0);
        total_decode_ms = 0.0;
        const int status = cuda_osd_decode_bp_csr_device_batch(
            handle,
            device_llrs,
            blocks,
            code_bits.data(),
            success.data(),
            &total_decode_ms);
        if (status != CUDA_OSD_STATUS_OK) {
            throw std::runtime_error(std::string("cuda_osd_decode_bp_csr_device_batch failed: ") +
                                     cuda_osd_get_last_error());
        }

        success_count = 0;
        for (unsigned char ok : success) {
            success_count += ok ? 1 : 0;
        }

        std::vector<uint8_t> info(static_cast<size_t>(blocks * k), 0);
        for (int b = 0; b < blocks; ++b) {
            for (int i = 0; i < k; ++i) {
                info[static_cast<size_t>(b * k + i)] =
                    code_bits[static_cast<size_t>(b * n + i)];
            }
        }
        return info;
#else
        (void)device_llrs;
        (void)blocks;
        (void)success_count;
        (void)total_decode_ms;
        throw std::runtime_error("this executable was built without generic CUDA BP support");
#endif
    }
};

static void fft_inplace(std::vector<cd>& a, bool inverse)
{
    const size_t n = a.size();
    if (n == 0 || (n & (n - 1)) != 0) {
        throw std::runtime_error("FFT size must be a power of two");
    }

    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }

    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = 2.0 * kPi / static_cast<double>(len) * (inverse ? 1.0 : -1.0);
        const cd wlen(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            cd w(1.0, 0.0);
            for (size_t j = 0; j < len / 2; ++j) {
                const cd u = a[i + j];
                const cd v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (inverse) {
        const double inv_n = 1.0 / static_cast<double>(n);
        for (auto& x : a) {
            x *= inv_n;
        }
    }
}

static std::vector<cf32> ifft_shifted_grid(const std::vector<cf32>& shifted_grid, int nfft)
{
    std::vector<cd> natural(static_cast<size_t>(nfft));
    for (int k = 0; k < nfft; ++k) {
        const int shifted_idx = (k + nfft / 2) % nfft;
        natural[k] = shifted_grid[shifted_idx];
    }
    fft_inplace(natural, true);

    std::vector<cf32> out(static_cast<size_t>(nfft));
    for (int i = 0; i < nfft; ++i) {
        out[i] = cf32(static_cast<float>(natural[i].real()), static_cast<float>(natural[i].imag()));
    }
    return out;
}

static std::vector<cf32> fft_to_shifted_grid(const cf32* time, int nfft)
{
    std::vector<cd> natural(static_cast<size_t>(nfft));
    for (int i = 0; i < nfft; ++i) {
        natural[i] = time[i];
    }
    fft_inplace(natural, false);

    std::vector<cf32> shifted(static_cast<size_t>(nfft));
    for (int i = 0; i < nfft; ++i) {
        const int natural_idx = (i + nfft / 2) % nfft;
        shifted[i] = cf32(static_cast<float>(natural[natural_idx].real()), static_cast<float>(natural[natural_idx].imag()));
    }
    return shifted;
}

static std::vector<cf32> build_preamble(int half_len)
{
    const auto bits = random_bits(static_cast<size_t>(2 * half_len), 11);
    const auto half = bits_to_qpsk(bits);

    std::vector<cf32> pre;
    pre.reserve(static_cast<size_t>(2 * half_len));
    pre.insert(pre.end(), half.begin(), half.end());
    pre.insert(pre.end(), half.begin(), half.end());

    double p = 0.0;
    for (const auto& x : pre) {
        p += std::norm(x);
    }
    p = std::sqrt(p / std::max<size_t>(pre.size(), 1));
    for (auto& x : pre) {
        x /= static_cast<float>(p);
    }
    return pre;
}

static void normalize_unit_rms(std::vector<cf32>& samples)
{
    double p = 0.0;
    for (const auto& x : samples) {
        p += std::norm(x);
    }
    p = std::sqrt(p / std::max<size_t>(samples.size(), 1));
    if (p <= 0.0 || !std::isfinite(p)) {
        throw std::runtime_error("cannot normalize an empty or zero-energy sequence");
    }
    for (auto& x : samples) {
        x /= static_cast<float>(p);
    }
}

static std::vector<cf32> build_zc_sequence(int length, int root)
{
    if (length <= 0) {
        throw std::runtime_error("ZC sequence length must be positive");
    }
    root = ((root % length) + length) % length;
    if (root == 0) {
        root = 1;
    }
    if (std::gcd(root, length) != 1) {
        std::ostringstream oss;
        oss << "--sync-zc-root (" << root
            << ") must be coprime with active subcarriers (" << length << ")";
        throw std::runtime_error(oss.str());
    }

    std::vector<cf32> zc(static_cast<size_t>(length));
    for (int n = 0; n < length; ++n) {
        const double phase = -kPi * static_cast<double>(root) *
                             static_cast<double>(n) * static_cast<double>(n + 1) /
                             static_cast<double>(length);
        zc[static_cast<size_t>(n)] =
            cf32(static_cast<float>(std::cos(phase)), static_cast<float>(std::sin(phase)));
    }
    return zc;
}

static std::vector<cf32> build_zc_ofdm_preamble(const PhyConfig& phy, int zc_root)
{
    if (phy.cp <= 0 || phy.cp >= phy.nfft) {
        throw std::runtime_error("--sync-preamble zc-ofdm requires 0 < --cp < --nfft");
    }
    if (static_cast<int>(phy.used_indices.size()) != phy.active_sc) {
        throw std::runtime_error("ZC OFDM preamble requires active carrier indices");
    }

    const auto zc = build_zc_sequence(phy.active_sc, zc_root);
    std::vector<cf32> grid(static_cast<size_t>(phy.nfft), cf32(0.0f, 0.0f));
    for (int sc = 0; sc < phy.active_sc; ++sc) {
        grid[static_cast<size_t>(phy.used_indices[sc])] = zc[static_cast<size_t>(sc)];
    }

    const auto td = ifft_shifted_grid(grid, phy.nfft);
    std::vector<cf32> block;
    block.reserve(static_cast<size_t>(phy.sym_len()));
    block.insert(block.end(), td.end() - phy.cp, td.end());
    block.insert(block.end(), td.begin(), td.end());
    normalize_unit_rms(block);

    std::vector<cf32> pre;
    pre.reserve(block.size() * 2);
    pre.insert(pre.end(), block.begin(), block.end());
    pre.insert(pre.end(), block.begin(), block.end());
    normalize_unit_rms(pre);
    return pre;
}

static std::vector<std::vector<cf32>> build_pilots(int active_sc, int n_pilots)
{
    const auto bits = random_bits(static_cast<size_t>(2 * active_sc * n_pilots), 73);
    const auto syms = bits_to_qpsk(bits);

    std::vector<std::vector<cf32>> pilots(
        static_cast<size_t>(n_pilots),
        std::vector<cf32>(static_cast<size_t>(active_sc)));

    for (int p = 0; p < n_pilots; ++p) {
        for (int sc = 0; sc < active_sc; ++sc) {
            pilots[p][sc] = syms[static_cast<size_t>(p * active_sc + sc)];
        }
    }
    return pilots;
}

static PhyConfig build_phy_config(const Options& opt)
{
    if ((opt.nfft & (opt.nfft - 1)) != 0) {
        throw std::runtime_error("--nfft must be a power of two");
    }
    if (opt.active_sc <= 0 || opt.active_sc >= opt.nfft || (opt.active_sc % 2) != 0) {
        throw std::runtime_error("--active-sc must be a positive even number below --nfft");
    }
    if (opt.num_symbols < 4) {
        throw std::runtime_error("--num-symbols is too small");
    }

    PhyConfig cfg;
    cfg.nfft = opt.nfft;
    cfg.cp = opt.cp;
    cfg.num_symbols = opt.num_symbols;
    cfg.active_sc = opt.active_sc;
    cfg.pre_half_len = opt.pre_half_len;
    cfg.sync_tracking_window = opt.sync_tracking_window;
    cfg.sync_preamble = opt.sync_preamble;
    cfg.rx_channel_estimation = opt.rx_channel_estimation;
    cfg.max_buffered_frames = opt.max_buffered_frames;
    cfg.rate = opt.rate;
    cfg.amplitude = opt.amplitude;
    cfg.sync_threshold = opt.sync_threshold;
    cfg.sync_late_peak_threshold = opt.sync_late_peak_threshold;
    cfg.test_seed = opt.test_seed;
    cfg.radio_oversample = opt.radio_oversample;
    configure_modulation(cfg, opt.modulation);

    if (opt.pilot_period > 0) {
        for (int s = 0; s < opt.num_symbols; s += opt.pilot_period) {
            cfg.pilot_symbols.push_back(s);
        }
    } else {
        cfg.pilot_symbols = {0, opt.num_symbols / 3, 2 * opt.num_symbols / 3};
    }
    std::sort(cfg.pilot_symbols.begin(), cfg.pilot_symbols.end());
    cfg.pilot_symbols.erase(std::unique(cfg.pilot_symbols.begin(), cfg.pilot_symbols.end()), cfg.pilot_symbols.end());

    for (int s = 0; s < opt.num_symbols; ++s) {
        if (std::find(cfg.pilot_symbols.begin(), cfg.pilot_symbols.end(), s) == cfg.pilot_symbols.end()) {
            cfg.data_symbols.push_back(s);
        }
    }
    if (cfg.data_symbols.empty()) {
        throw std::runtime_error("pilot pattern leaves no data OFDM symbols");
    }

    const int n2 = cfg.nfft / 2;
    const int a2 = cfg.active_sc / 2;
    for (int i = n2 - a2; i <= n2 - 1; ++i) {
        cfg.used_indices.push_back(i);
    }
    for (int i = n2 + 1; i <= n2 + a2; ++i) {
        cfg.used_indices.push_back(i);
    }

    if (cfg.sync_preamble == "zc-ofdm") {
        cfg.pre_half_len = cfg.sym_len();
        cfg.preamble = build_zc_ofdm_preamble(cfg, opt.sync_zc_root);
    } else {
        cfg.preamble = build_preamble(cfg.pre_half_len);
    }
    if (cfg.sync_tracking_window <= 0) {
        const int legacy_window = std::max(256, 4 * cfg.cp);
        cfg.sync_tracking_window = legacy_window;
    }
    if (cfg.sync_late_peak_threshold <= 0.0 &&
        cfg.sync_preamble == "zc-ofdm" &&
        cfg.sync_tracking_window > std::max(256, 4 * cfg.cp)) {
        cfg.sync_late_peak_threshold = std::max(cfg.sync_threshold + 0.20, 0.85);
    }
    cfg.pilot_grid = build_pilots(cfg.active_sc, static_cast<int>(cfg.pilot_symbols.size()));
    return cfg;
}

static std::vector<uint8_t> build_info_bits(const PhyConfig& phy, const LdpcCode& code, uint32_t frame_id)
{
    const int blocks = phy.coded_bits_per_frame() / code.n;
    std::vector<uint8_t> bits(static_cast<size_t>(blocks * code.k), 0);
    put_u32_msb(bits, 0, frame_id);

    auto payload = random_bits(bits.size() - 32, static_cast<uint32_t>(phy.test_seed) + frame_id);
    std::copy(payload.begin(), payload.end(), bits.begin() + 32);
    return bits;
}

static void apply_test_reference_result(
    FrameDecodeResult& r,
    const PhyConfig& phy,
    const LdpcCode& code,
    uint32_t expected_frame_id,
    bool skip_reference_ber,
    DecodeProfile* profile = nullptr)
{
    const auto t0 = std::chrono::steady_clock::now();
    r.frame_id = expected_frame_id;
    r.pre_fec_bit_errors = -1;
    if (skip_reference_ber) {
        r.bit_errors = -1;
        r.frame_ok = r.parity_ok;
    } else {
        const auto ref_info = build_info_bits(phy, code, expected_frame_id);
        int bit_errors = 0;
        for (size_t b = 0; b < std::min(r.info_bits.size(), ref_info.size()); ++b) {
            bit_errors += (r.info_bits[b] != ref_info[b]) ? 1 : 0;
        }
        r.bit_errors = bit_errors;
        r.frame_ok = r.parity_ok && r.bit_errors == 0;
    }
    const auto t1 = std::chrono::steady_clock::now();
    if (profile != nullptr) {
        profile->reference_ns += elapsed_ns(t0, t1);
    }
}

static void apply_parity_only_test_result(FrameDecodeResult& r)
{
    r.frame_id = r.info_bits.size() >= 32 ? bits_to_u32_msb(r.info_bits, 0) : 0;
    r.bit_errors = -1;
    r.pre_fec_bit_errors = -1;
    r.frame_ok = r.parity_ok;
}

constexpr uint32_t kMediaMagic = 0x31444956u; // "VID1" in little-endian byte order.
constexpr uint16_t kMediaVersion = 1;
constexpr uint16_t kMediaHeaderBytes = 48;
constexpr uint32_t kMediaFlagFirst = 1u << 0;
constexpr uint32_t kMediaFlagEof = 1u << 1;
constexpr uint32_t kMediaFlagControl = 1u << 8;
constexpr uint32_t kControlMagic = 0x314c5443u; // "CTL1" in little-endian byte order.
constexpr uint16_t kControlVersion = 1;
constexpr uint16_t kControlBytes = 80;

struct LinkControlMessage {
    uint32_t effective_frame_id = 0;
    uint32_t mcs = 1;
    uint32_t repeat = 3;
    uint32_t flags = 0;
    float effective_rate = 1.0f / 6.0f;
    float snr_for_mcs_db = 0.0f;
    char modulation[16] = {};
    char fec_profile[24] = {};
};

struct LinkControlState {
    uint32_t active_effective_frame_id = 0;
    usrp_link::AmcDecision active_decision;
    uint64_t accepted_controls = 0;

    void accept(const LinkControlMessage& msg)
    {
        active_effective_frame_id = msg.effective_frame_id;
        active_decision.mcs = static_cast<usrp_link::McsId>(std::max<uint32_t>(0, std::min<uint32_t>(5, msg.mcs)));
        active_decision.modulation = msg.modulation[0] ? std::string(msg.modulation) : "qpsk";
        active_decision.fec_profile = msg.fec_profile[0] ? std::string(msg.fec_profile) : "ccsds-ldpc-128-64";
        active_decision.repeat_count = static_cast<int>(std::max<uint32_t>(1, msg.repeat));
        active_decision.effective_rate = msg.effective_rate;
        ++accepted_controls;
        std::cout << "[CTRL-RX] accepted control effectiveFrame=" << msg.effective_frame_id
                  << " mcs=" << msg.mcs
                  << " modulation=" << active_decision.modulation
                  << " repeat=" << active_decision.repeat_count
                  << " effRate=" << std::fixed << std::setprecision(3) << active_decision.effective_rate
                  << " snrForMcs=" << std::setprecision(2) << msg.snr_for_mcs_db
                  << " note=profile-bank hot-switch applied after control decode\n";
    }
};

struct MediaInfoBits {
    std::vector<uint8_t> bits;
    size_t payload_len = 0;
    uint64_t next_offset = 0;
    bool eof = false;
};

static std::vector<uint8_t> build_media_header(
    uint32_t stream_id,
    uint32_t frame_id,
    uint64_t total_size,
    uint64_t offset,
    uint32_t payload_len,
    uint32_t payload_crc,
    uint32_t flags)
{
    std::vector<uint8_t> h(kMediaHeaderBytes, 0);
    put_u32_le(h, 0, kMediaMagic);
    put_u16_le(h, 4, kMediaVersion);
    put_u16_le(h, 6, kMediaHeaderBytes);
    put_u32_le(h, 8, flags);
    put_u32_le(h, 12, stream_id);
    put_u32_le(h, 16, frame_id);
    put_u64_le(h, 20, total_size);
    put_u64_le(h, 28, offset);
    put_u32_le(h, 36, payload_len);
    put_u32_le(h, 40, payload_crc);
    put_u32_le(h, 44, crc32_bytes(h.data(), 44));
    return h;
}

static void copy_fixed_string(std::vector<uint8_t>& out, size_t off, size_t max_len, const std::string& s)
{
    const size_t n = std::min(max_len - 1, s.size());
    std::copy_n(s.data(), n, reinterpret_cast<char*>(out.data() + static_cast<std::ptrdiff_t>(off)));
    out[off + n] = 0;
}

static std::vector<uint8_t> build_control_payload(const usrp_link::AmcDecision& decision,
                                                  uint32_t effective_frame_id,
                                                  float snr_for_mcs_db)
{
    std::vector<uint8_t> p(kControlBytes, 0);
    put_u32_le(p, 0, kControlMagic);
    put_u16_le(p, 4, kControlVersion);
    put_u16_le(p, 6, kControlBytes);
    put_u32_le(p, 8, effective_frame_id);
    put_u32_le(p, 12, static_cast<uint32_t>(decision.mcs));
    put_u32_le(p, 16, static_cast<uint32_t>(std::max(decision.repeat_count, 1)));
    put_u32_le(p, 20, decision.allow_runtime_switch ? 1u : 0u);
    static_assert(sizeof(float) == 4, "float must be 32-bit");
    const float effective_rate = static_cast<float>(decision.effective_rate);
    std::memcpy(p.data() + 24, &effective_rate, sizeof(float));
    std::memcpy(p.data() + 28, &snr_for_mcs_db, sizeof(float));
    copy_fixed_string(p, 32, 16, decision.modulation);
    copy_fixed_string(p, 48, 24, decision.fec_profile);
    put_u32_le(p, 76, crc32_bytes(p.data(), 76));
    return p;
}

static bool parse_control_payload(const std::vector<uint8_t>& payload, LinkControlMessage& msg)
{
    if (payload.size() < kControlBytes) {
        return false;
    }
    if (read_u32_le(payload.data() + 0) != kControlMagic ||
        read_u16_le(payload.data() + 4) != kControlVersion ||
        read_u16_le(payload.data() + 6) != kControlBytes) {
        return false;
    }
    if (read_u32_le(payload.data() + 76) != crc32_bytes(payload.data(), 76)) {
        return false;
    }
    msg.effective_frame_id = read_u32_le(payload.data() + 8);
    msg.mcs = read_u32_le(payload.data() + 12);
    msg.repeat = read_u32_le(payload.data() + 16);
    msg.flags = read_u32_le(payload.data() + 20);
    std::memcpy(&msg.effective_rate, payload.data() + 24, sizeof(float));
    std::memcpy(&msg.snr_for_mcs_db, payload.data() + 28, sizeof(float));
    std::memcpy(msg.modulation, payload.data() + 32, 16);
    msg.modulation[15] = '\0';
    std::memcpy(msg.fec_profile, payload.data() + 48, 24);
    msg.fec_profile[23] = '\0';
    return true;
}

static bool parse_media_header(const std::vector<uint8_t>& bytes, MediaHeader& h)
{
    if (bytes.size() < kMediaHeaderBytes) {
        return false;
    }

    h.magic = read_u32_le(bytes.data() + 0);
    h.version = read_u16_le(bytes.data() + 4);
    h.header_len = read_u16_le(bytes.data() + 6);
    h.flags = read_u32_le(bytes.data() + 8);
    h.stream_id = read_u32_le(bytes.data() + 12);
    h.frame_id = read_u32_le(bytes.data() + 16);
    h.total_size = read_u64_le(bytes.data() + 20);
    h.offset = read_u64_le(bytes.data() + 28);
    h.payload_len = read_u32_le(bytes.data() + 36);
    h.payload_crc32 = read_u32_le(bytes.data() + 40);
    h.header_crc32 = read_u32_le(bytes.data() + 44);

    if (h.magic != kMediaMagic || h.version != kMediaVersion || h.header_len != kMediaHeaderBytes) {
        return false;
    }
    if (h.header_crc32 != crc32_bytes(bytes.data(), 44)) {
        return false;
    }
    if (h.payload_len > bytes.size() - kMediaHeaderBytes) {
        return false;
    }
    if (h.offset > h.total_size || static_cast<uint64_t>(h.payload_len) > h.total_size - h.offset) {
        return false;
    }
    return true;
}

struct MediaPacketizer {
    std::vector<uint8_t> file_bytes;
    uint32_t stream_id = 0;
    size_t info_bytes_per_frame = 0;
    size_t payload_bytes_per_frame = 0;
    uint64_t total_chunks = 0;
    bool loop = false;

    MediaPacketizer(const std::string& path, size_t info_bytes, bool loop_file)
        : file_bytes(read_binary_file(path)),
          info_bytes_per_frame(info_bytes),
          loop(loop_file)
    {
        if (info_bytes_per_frame <= kMediaHeaderBytes) {
            throw std::runtime_error("LDPC information payload is too small for media header");
        }
        if (file_bytes.empty()) {
            throw std::runtime_error("input file is empty: " + path);
        }

        payload_bytes_per_frame = info_bytes_per_frame - kMediaHeaderBytes;
        total_chunks = static_cast<uint64_t>((file_bytes.size() + payload_bytes_per_frame - 1) / payload_bytes_per_frame);

        std::vector<uint8_t> id_material(path.begin(), path.end());
        for (int i = 0; i < 8; ++i) {
            id_material.push_back(static_cast<uint8_t>((file_bytes.size() >> (8 * i)) & 0xffu));
        }
        const size_t prefix = std::min<size_t>(file_bytes.size(), 4096);
        id_material.insert(id_material.end(), file_bytes.begin(), file_bytes.begin() + static_cast<std::ptrdiff_t>(prefix));
        stream_id = crc32_bytes(id_material.data(), id_material.size());
        if (stream_id == 0) {
            stream_id = 1;
        }
    }

    bool done(uint64_t tx_frame_count) const
    {
        return !loop && tx_frame_count >= total_chunks;
    }

    bool done_with_payload(uint64_t tx_chunk_count, size_t payload_bytes) const
    {
        if (loop) {
            return false;
        }
        const uint64_t chunks = static_cast<uint64_t>((file_bytes.size() + payload_bytes - 1) / payload_bytes);
        return tx_chunk_count >= chunks;
    }

    bool done_offset(uint64_t byte_offset) const
    {
        return !loop && byte_offset >= file_bytes.size();
    }

    std::vector<uint8_t> build_info_bits(uint32_t frame_id) const
    {
        if (done(frame_id)) {
            throw std::runtime_error("media packetizer asked for a frame past EOF");
        }

        return build_media_info_bits(frame_id, static_cast<uint64_t>(frame_id));
    }

    std::vector<uint8_t> build_media_info_bits(uint32_t frame_id, uint64_t media_chunk_count) const
    {
        return build_media_info_bits(frame_id, media_chunk_count, info_bytes_per_frame, payload_bytes_per_frame);
    }

    std::vector<uint8_t> build_media_info_bits(uint32_t frame_id,
                                               uint64_t media_chunk_count,
                                               size_t info_bytes,
                                               size_t payload_bytes) const
    {
        if (!loop && media_chunk_count * payload_bytes >= file_bytes.size()) {
            throw std::runtime_error("media packetizer asked for a chunk past EOF");
        }
        if (info_bytes <= kMediaHeaderBytes || payload_bytes > info_bytes - kMediaHeaderBytes) {
            throw std::runtime_error("invalid media payload capacity for active profile");
        }

        const uint64_t offset = media_chunk_count * payload_bytes;
        return build_media_info_bits_from_offset(frame_id, offset, info_bytes, payload_bytes).bits;
    }

    MediaInfoBits build_media_info_bits_from_offset(uint32_t frame_id,
                                                    uint64_t byte_offset,
                                                    size_t info_bytes,
                                                    size_t payload_bytes) const
    {
        if (done_offset(byte_offset)) {
            throw std::runtime_error("media packetizer asked for a byte offset past EOF");
        }
        if (info_bytes <= kMediaHeaderBytes || payload_bytes > info_bytes - kMediaHeaderBytes) {
            throw std::runtime_error("invalid media payload capacity for active profile");
        }

        const uint64_t offset = loop
            ? (byte_offset % static_cast<uint64_t>(file_bytes.size()))
            : byte_offset;
        const size_t payload_len = static_cast<size_t>(
            std::min<uint64_t>(payload_bytes, static_cast<uint64_t>(file_bytes.size()) - offset));

        std::vector<uint8_t> info(info_bytes, 0);
        std::vector<uint8_t> payload(payload_len, 0);
        if (payload_len > 0) {
            std::copy(
                file_bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                file_bytes.begin() + static_cast<std::ptrdiff_t>(offset + payload_len),
                payload.begin());
        }

        uint32_t flags = 0;
        if (offset == 0) {
            flags |= kMediaFlagFirst;
        }
        if (offset + payload_len >= file_bytes.size()) {
            flags |= kMediaFlagEof;
        }

        const auto header = build_media_header(
            stream_id,
            frame_id,
            static_cast<uint64_t>(file_bytes.size()),
            offset,
            static_cast<uint32_t>(payload_len),
            payload.empty() ? 0u : crc32_bytes(payload.data(), payload.size()),
            flags);
        std::copy(header.begin(), header.end(), info.begin());
        std::copy(payload.begin(), payload.end(), info.begin() + kMediaHeaderBytes);
        MediaInfoBits out;
        out.bits = bytes_to_bits_msb(info, info.size() * 8);
        out.payload_len = payload_len;
        out.eof = (flags & kMediaFlagEof) != 0;
        out.next_offset = loop && offset + payload_len >= file_bytes.size()
            ? 0
            : offset + payload_len;
        return out;
    }

    std::vector<uint8_t> build_control_info_bits(uint32_t frame_id,
                                                 const usrp_link::AmcDecision& decision,
                                                 uint32_t effective_frame_id,
                                                 float snr_for_mcs_db) const
    {
        return build_control_info_bits(
            frame_id,
            decision,
            effective_frame_id,
            snr_for_mcs_db,
            info_bytes_per_frame);
    }

    std::vector<uint8_t> build_control_info_bits(uint32_t frame_id,
                                                 const usrp_link::AmcDecision& decision,
                                                 uint32_t effective_frame_id,
                                                 float snr_for_mcs_db,
                                                 size_t info_bytes) const
    {
        const auto payload = build_control_payload(decision, effective_frame_id, snr_for_mcs_db);
        if (kMediaHeaderBytes + payload.size() > info_bytes) {
            throw std::runtime_error("LDPC information payload is too small for link control frame");
        }

        std::vector<uint8_t> info(info_bytes, 0);
        const auto header = build_media_header(
            stream_id,
            frame_id,
            static_cast<uint64_t>(payload.size()),
            0,
            static_cast<uint32_t>(payload.size()),
            crc32_bytes(payload.data(), payload.size()),
            kMediaFlagControl);
        std::copy(header.begin(), header.end(), info.begin());
        std::copy(payload.begin(), payload.end(), info.begin() + kMediaHeaderBytes);
        return bytes_to_bits_msb(info, info.size() * 8);
    }
};

struct MediaReassembler {
    std::string output_path;
    std::fstream out;
    uint32_t stream_id = 0;
    uint64_t total_size = 0;
    size_t chunk_bytes = 0;
    std::vector<uint8_t> received;
    std::vector<std::pair<uint64_t, uint64_t>> received_ranges;
    uint64_t received_bytes = 0;
    uint64_t received_chunks = 0;
    uint64_t accepted_packets = 0;
    uint64_t rejected_packets = 0;
    bool complete = false;

    explicit MediaReassembler(std::string path)
        : output_path(std::move(path))
    {}

    void reset(const MediaHeader& h, size_t payload_capacity)
    {
        if (out.is_open()) {
            out.close();
        }

        stream_id = h.stream_id;
        total_size = h.total_size;
        chunk_bytes = payload_capacity;
        if (chunk_bytes == 0) {
            chunk_bytes = 1;
        }
        received_chunks = 0;
        received_bytes = 0;
        received_ranges.clear();
        accepted_packets = 0;
        rejected_packets = 0;
        complete = false;
        const size_t chunks = static_cast<size_t>((total_size + chunk_bytes - 1) / chunk_bytes);
        received.assign(chunks, 0);

        {
            std::ofstream trunc(output_path, std::ios::binary | std::ios::trunc);
            if (!trunc) {
                throw std::runtime_error("cannot create RX output file: " + output_path);
            }
        }

        out.open(output_path, std::ios::binary | std::ios::in | std::ios::out);
        if (!out) {
            throw std::runtime_error("cannot reopen RX output file: " + output_path);
        }

        std::cout << "[MEDIA] new stream id=" << stream_id
                  << " size=" << total_size
                  << " bytes chunks=" << chunks
                  << " output=" << output_path << "\n";
    }

    void mark_received_range(uint64_t begin, uint64_t end)
    {
        end = std::min(end, total_size);
        if (begin >= end) {
            return;
        }

        uint64_t new_begin = begin;
        uint64_t new_end = end;
        uint64_t overlapped = 0;
        auto it = received_ranges.begin();
        while (it != received_ranges.end()) {
            if (it->second < new_begin) {
                ++it;
                continue;
            }
            if (it->first > new_end) {
                break;
            }
            const uint64_t ov_begin = std::max(new_begin, it->first);
            const uint64_t ov_end = std::min(new_end, it->second);
            if (ov_begin < ov_end) {
                overlapped += ov_end - ov_begin;
            }
            new_begin = std::min(new_begin, it->first);
            new_end = std::max(new_end, it->second);
            it = received_ranges.erase(it);
        }
        received_ranges.insert(it, {new_begin, new_end});
        received_bytes += (end - begin) - overlapped;
    }

    bool accept(const MediaHeader& h, const std::vector<uint8_t>& payload, size_t payload_capacity)
    {
        if (payload.size() != h.payload_len) {
            ++rejected_packets;
            return false;
        }
        if (!payload.empty() && crc32_bytes(payload.data(), payload.size()) != h.payload_crc32) {
            ++rejected_packets;
            return false;
        }
        if (payload.empty() && h.payload_crc32 != 0) {
            ++rejected_packets;
            return false;
        }
        if (!out.is_open() || h.stream_id != stream_id || h.total_size != total_size) {
            reset(h, payload_capacity);
        }
        if (payload.empty()) {
            ++accepted_packets;
            return true;
        }

        out.seekp(static_cast<std::streamoff>(h.offset));
        out.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
        if (!out) {
            throw std::runtime_error("failed to write RX output file: " + output_path);
        }

        mark_received_range(h.offset, h.offset + static_cast<uint64_t>(payload.size()));
        received_chunks = received_bytes / chunk_bytes;
        ++accepted_packets;
        const bool was_complete = complete;
        complete = received_bytes >= total_size;
        if ((!was_complete && complete) || (h.flags & kMediaFlagEof)) {
            out.flush();
        }
        if ((!was_complete && complete) || (accepted_packets % 50 == 0) || (h.flags & kMediaFlagEof)) {
            const double pct = total_size == 0 ? 100.0 : 100.0 * static_cast<double>(received_bytes) / static_cast<double>(total_size);
            std::cout << "[MEDIA] accepted=" << accepted_packets
                      << " receivedBytes=" << received_bytes << "/" << total_size
                      << " complete=" << (complete ? "yes" : "no")
                      << " progress=" << std::fixed << std::setprecision(1) << pct << "%\n";
        }
        return true;
    }
};

static std::vector<uint8_t> encode_frame_bits(const PhyConfig& phy, const LdpcCode& code, const std::vector<uint8_t>& info_bits)
{
    const int blocks = phy.coded_bits_per_frame() / code.n;
    if (static_cast<int>(info_bits.size()) != blocks * code.k) {
        throw std::runtime_error("info bit count does not match the configured frame");
    }

    std::vector<uint8_t> coded;
    coded.reserve(static_cast<size_t>(phy.coded_bits_per_frame()));
    for (int b = 0; b < blocks; ++b) {
        auto cw = ldpc_encode(code, info_bits.data() + static_cast<size_t>(b * code.k));
        coded.insert(coded.end(), cw.begin(), cw.end());
    }
    return coded;
}

static std::vector<cf32> build_tx_frame_from_info(const PhyConfig& phy, const LdpcCode& code, const std::vector<uint8_t>& info_bits)
{
    const auto coded_bits = encode_frame_bits(phy, code, info_bits);
    const auto mapped_symbols = bits_to_symbols(phy, coded_bits);

    std::vector<cf32> out;
    out.reserve(static_cast<size_t>(phy.frame_len()));
    out.insert(out.end(), phy.preamble.begin(), phy.preamble.end());

    int data_col = 0;
    for (int sym = 0; sym < phy.num_symbols; ++sym) {
        std::vector<cf32> grid(static_cast<size_t>(phy.nfft), cf32(0.0f, 0.0f));
        const auto pit = std::find(phy.pilot_symbols.begin(), phy.pilot_symbols.end(), sym);
        if (pit != phy.pilot_symbols.end()) {
            const int pidx = static_cast<int>(std::distance(phy.pilot_symbols.begin(), pit));
            for (int sc = 0; sc < phy.active_sc; ++sc) {
                grid[phy.used_indices[sc]] = phy.pilot_grid[pidx][sc];
            }
        } else {
            for (int sc = 0; sc < phy.active_sc; ++sc) {
                grid[phy.used_indices[sc]] = mapped_symbols[static_cast<size_t>(data_col * phy.active_sc + sc)];
            }
            ++data_col;
        }

        auto td = ifft_shifted_grid(grid, phy.nfft);
        out.insert(out.end(), td.end() - phy.cp, td.end());
        out.insert(out.end(), td.begin(), td.end());
    }

    float max_abs = 1e-12f;
    for (const auto& x : out) {
        max_abs = std::max(max_abs, std::abs(x));
    }
    const float scale = static_cast<float>(phy.amplitude) / max_abs;
    for (auto& x : out) {
        x *= scale;
    }
    return out;
}

static std::vector<cf32> build_tx_frame(const PhyConfig& phy, const LdpcCode& code, uint32_t frame_id)
{
    const auto info_bits = build_info_bits(phy, code, frame_id);
    return build_tx_frame_from_info(phy, code, info_bits);
}

static std::vector<cf32> upsample_for_radio(const std::vector<cf32>& baseband, int factor)
{
    if (factor <= 1 || baseband.empty()) {
        return baseband;
    }
    std::vector<cf32> out;
    out.reserve(baseband.size() * static_cast<size_t>(factor));
    for (const auto& sample : baseband) {
        for (int i = 0; i < factor; ++i) {
            out.push_back(sample);
        }
    }
    return out;
}

static std::vector<cf32> downsample_from_radio(const cf32* radio_samples, size_t count, int factor, size_t& phase)
{
    if (factor <= 1) {
        return std::vector<cf32>(radio_samples, radio_samples + static_cast<std::ptrdiff_t>(count));
    }
    std::vector<cf32> out;
    out.reserve(count / static_cast<size_t>(factor) + 1);
    for (size_t i = 0; i < count; ++i) {
        if (phase == 0) {
            out.push_back(radio_samples[i]);
        }
        phase = (phase + 1) % static_cast<size_t>(factor);
    }
    return out;
}

struct ChannelPath {
    cf32 gain = cf32(1.0f, 0.0f);
    double doppler_hz = 0.0;
    double delay_samples = 0.0;
};

struct InjectionChannel {
    std::vector<std::string> source_files;
    std::vector<ChannelPath> paths;

    bool empty() const { return paths.empty(); }
};

static std::vector<std::string> split_path_list(const std::string& list)
{
    std::vector<std::string> out;
    std::string item;
    for (char c : list) {
        if (c == ',' || c == ';') {
            if (!item.empty()) {
                out.push_back(item);
                item.clear();
            }
        } else {
            item.push_back(c);
        }
    }
    if (!item.empty()) {
        out.push_back(item);
    }
    for (auto& path : out) {
        const auto first = path.find_first_not_of(" \t\r\n\"'");
        const auto last = path.find_last_not_of(" \t\r\n\"'");
        path = (first == std::string::npos) ? std::string{} : path.substr(first, last - first + 1);
    }
    out.erase(std::remove_if(out.begin(), out.end(), [](const std::string& s) { return s.empty(); }), out.end());
    return out;
}

static std::vector<double> read_numeric_tokens(const std::string& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open channel file: " + path);
    }

    std::vector<double> values;
    std::string line;
    while (std::getline(in, line)) {
        const auto hash = line.find('#');
        if (hash != std::string::npos) {
            line.resize(hash);
        }
        const auto pct = line.find('%');
        if (pct != std::string::npos) {
            line.resize(pct);
        }
        const auto slash = line.find("//");
        if (slash != std::string::npos) {
            line.resize(slash);
        }

        const char* p = line.c_str();
        while (*p != '\0') {
            char* end = nullptr;
            const double v = std::strtod(p, &end);
            if (end != p) {
                values.push_back(v);
                p = end;
            } else {
                ++p;
            }
        }
    }
    return values;
}

static InjectionChannel load_injection_channel(const std::string& path_list)
{
    InjectionChannel channel;
    for (const auto& path : split_path_list(path_list)) {
        const auto values = read_numeric_tokens(path);
        if (values.empty()) {
            throw std::runtime_error("channel file has no numeric content: " + path);
        }
        const int path_count = static_cast<int>(std::lround(values[0]));
        if (path_count <= 0 || std::abs(values[0] - static_cast<double>(path_count)) > 1e-6) {
            throw std::runtime_error("channel file first numeric token must be a positive integer P: " + path);
        }

        const size_t remaining = values.size() - 1;
        const bool complex_gain = (remaining == static_cast<size_t>(4 * path_count));
        const bool real_gain = (remaining == static_cast<size_t>(3 * path_count));
        if (!complex_gain && !real_gain) {
            std::ostringstream oss;
            oss << "channel file " << path
                << " must contain P followed by P*(h,k,delay) or P*(h_real,h_imag,k,delay) values";
            throw std::runtime_error(oss.str());
        }

        size_t off = 1;
        for (int i = 0; i < path_count; ++i) {
            ChannelPath pth;
            if (complex_gain) {
                pth.gain = cf32(static_cast<float>(values[off + 0]), static_cast<float>(values[off + 1]));
                pth.doppler_hz = values[off + 2];
                pth.delay_samples = values[off + 3];
                off += 4;
            } else {
                pth.gain = cf32(static_cast<float>(values[off + 0]), 0.0f);
                pth.doppler_hz = values[off + 1];
                pth.delay_samples = values[off + 2];
                off += 3;
            }
            channel.paths.push_back(pth);
        }
        channel.source_files.push_back(path);
    }
    return channel;
}

static cf32 ofdm_channel_response(const InjectionChannel& channel, const PhyConfig& phy, int sym, int sc)
{
    const int shifted_index = phy.used_indices[static_cast<size_t>(sc)];
    const int signed_bin = shifted_index - phy.nfft / 2;
    const double symbol_time =
        (static_cast<double>(phy.preamble.size()) + static_cast<double>(sym) * static_cast<double>(phy.sym_len())) /
        std::max(phy.rate, 1.0);

    cd h(0.0, 0.0);
    for (const auto& path : channel.paths) {
        const double delay_phase = -2.0 * kPi * static_cast<double>(signed_bin) * path.delay_samples /
                                   static_cast<double>(phy.nfft);
        const double doppler_phase = 2.0 * kPi * path.doppler_hz * symbol_time;
        const double phase = delay_phase + doppler_phase;
        h += cd(path.gain.real(), path.gain.imag()) * cd(std::cos(phase), std::sin(phase));
    }
    return cf32(static_cast<float>(h.real()), static_cast<float>(h.imag()));
}

static void apply_ofdm_channel_transform(std::vector<cf32>& frame,
                                         const PhyConfig& phy,
                                         const InjectionChannel& channel,
                                         bool inverse)
{
    if (channel.empty()) {
        return;
    }
    if (frame.size() < static_cast<size_t>(phy.frame_len())) {
        throw std::runtime_error("channel transform received a partial OFDM frame");
    }

    size_t off = phy.preamble.size();
    for (int sym = 0; sym < phy.num_symbols; ++sym) {
        const size_t body = off + static_cast<size_t>(phy.cp);
        auto grid = fft_to_shifted_grid(frame.data() + body, phy.nfft);
        for (int sc = 0; sc < phy.active_sc; ++sc) {
            const cf32 h = ofdm_channel_response(channel, phy, sym, sc);
            const float h2 = std::norm(h);
            if (inverse) {
                if (h2 < 1e-10f) {
                    throw std::runtime_error("precompensation channel response is too close to zero");
                }
                grid[phy.used_indices[static_cast<size_t>(sc)]] *= std::conj(h) / h2;
            } else {
                grid[phy.used_indices[static_cast<size_t>(sc)]] *= h;
            }
        }
        auto td = ifft_shifted_grid(grid, phy.nfft);
        std::copy(td.end() - phy.cp, td.end(), frame.begin() + static_cast<std::ptrdiff_t>(off));
        std::copy(td.begin(), td.end(), frame.begin() + static_cast<std::ptrdiff_t>(body));
        off += static_cast<size_t>(phy.sym_len());
    }
}

static double active_subcarrier_power(const std::vector<cf32>& frame, const PhyConfig& phy)
{
    if (frame.size() < static_cast<size_t>(phy.frame_len())) {
        return 0.0;
    }

    double power = 0.0;
    uint64_t count = 0;
    size_t off = phy.preamble.size();
    for (int sym = 0; sym < phy.num_symbols; ++sym) {
        const size_t body = off + static_cast<size_t>(phy.cp);
        const auto grid = fft_to_shifted_grid(frame.data() + body, phy.nfft);
        for (int sc = 0; sc < phy.active_sc; ++sc) {
            power += std::norm(grid[phy.used_indices[static_cast<size_t>(sc)]]);
            ++count;
        }
        off += static_cast<size_t>(phy.sym_len());
    }
    return count > 0 ? power / static_cast<double>(count) : 0.0;
}

static std::optional<double> add_awgn_to_data_subcarriers(
    std::vector<cf32>& frame,
    const PhyConfig& phy,
    double snr_db,
    std::mt19937& rng)
{
    if (!std::isfinite(snr_db) || frame.empty()) {
        return std::nullopt;
    }
    if (frame.size() < static_cast<size_t>(phy.frame_len())) {
        throw std::runtime_error("data-subcarrier AWGN received a partial OFDM frame");
    }

    double signal_power = 0.0;
    uint64_t count = 0;
    for (const int sym : phy.data_symbols) {
        const size_t body =
            phy.preamble.size() + static_cast<size_t>(sym) * static_cast<size_t>(phy.sym_len()) +
            static_cast<size_t>(phy.cp);
        const auto grid = fft_to_shifted_grid(frame.data() + body, phy.nfft);
        for (int sc = 0; sc < phy.active_sc; ++sc) {
            signal_power += std::norm(grid[phy.used_indices[static_cast<size_t>(sc)]]);
            ++count;
        }
    }
    if (count == 0) {
        return std::nullopt;
    }

    signal_power = std::max(signal_power / static_cast<double>(count), 1.0e-12);
    const double noise_var = signal_power / std::pow(10.0, snr_db / 10.0);
    const float sigma = static_cast<float>(std::sqrt(std::max(noise_var, 0.0) * 0.5));
    std::normal_distribution<float> noise_dist(0.0f, 1.0f);

    for (const int sym : phy.data_symbols) {
        const size_t off = phy.preamble.size() + static_cast<size_t>(sym) * static_cast<size_t>(phy.sym_len());
        const size_t body = off + static_cast<size_t>(phy.cp);
        auto grid = fft_to_shifted_grid(frame.data() + body, phy.nfft);
        for (int sc = 0; sc < phy.active_sc; ++sc) {
            const size_t idx = static_cast<size_t>(phy.used_indices[static_cast<size_t>(sc)]);
            grid[idx] += cf32(sigma * noise_dist(rng), sigma * noise_dist(rng));
        }
        const auto td = ifft_shifted_grid(grid, phy.nfft);
        std::copy(td.end() - phy.cp, td.end(), frame.begin() + static_cast<std::ptrdiff_t>(off));
        std::copy(td.begin(), td.end(), frame.begin() + static_cast<std::ptrdiff_t>(body));
    }
    return noise_var;
}

struct SnrTraceSegment {
    double snr_db = std::numeric_limits<double>::infinity();
    uint64_t frames = 0;
};

static std::string trim_ascii_copy(std::string s)
{
    auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [&](char c) {
        return !is_space(static_cast<unsigned char>(c));
    }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [&](char c) {
        return !is_space(static_cast<unsigned char>(c));
    }).base(), s.end());
    return s;
}

static std::vector<SnrTraceSegment> parse_snr_trace_segments(const std::string& spec, const char* option_name)
{
    std::string text = trim_ascii_copy(spec);
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (text.empty()) {
        return {};
    }
    if (lower == "demo" || lower == "amc-demo" || lower == "dynamic-demo") {
        return {
            {8.0, 200},
            {14.5, 200},
            {18.0, 200},
            {10.0, 200},
            {15.0, 200},
            {7.0, 200},
        };
    }

    std::vector<SnrTraceSegment> segments;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = trim_ascii_copy(token);
        if (token.empty()) {
            continue;
        }
        const size_t colon = token.find(':');
        if (colon == std::string::npos) {
            segments.push_back({std::stod(token), 1});
        } else {
            const double snr_db = std::stod(trim_ascii_copy(token.substr(0, colon)));
            const uint64_t frames = static_cast<uint64_t>(std::stoull(trim_ascii_copy(token.substr(colon + 1))));
            if (frames == 0) {
                throw std::runtime_error(std::string(option_name) + " segment frame count must be positive: " + token);
            }
            segments.push_back({snr_db, frames});
        }
    }
    if (segments.empty()) {
        throw std::runtime_error(std::string(option_name) + " did not contain any valid SNR segments");
    }
    return segments;
}

static double snr_trace_value_for_frame(
    const std::vector<SnrTraceSegment>& segments,
    uint64_t frame_id,
    double fallback_snr_db)
{
    if (segments.empty()) {
        return fallback_snr_db;
    }
    uint64_t period = 0;
    for (const auto& seg : segments) {
        period += seg.frames;
    }
    if (period == 0) {
        return fallback_snr_db;
    }
    uint64_t pos = frame_id % period;
    for (const auto& seg : segments) {
        if (pos < seg.frames) {
            return seg.snr_db;
        }
        pos -= seg.frames;
    }
    return segments.back().snr_db;
}

static bool tx_awgn_enabled(const Options& opt)
{
    return std::isfinite(opt.tx_awgn_snr_db) || !opt.tx_awgn_snr_trace.empty();
}

static std::optional<double> add_tx_awgn_if_requested(
    std::vector<cf32>& frame,
    const PhyConfig& phy,
    const Options& opt,
    std::mt19937& rng,
    double snr_db)
{
    if (!std::isfinite(snr_db)) {
        return std::nullopt;
    }
    if (opt.tx_awgn_scope == "data-subcarriers") {
        return add_awgn_to_data_subcarriers(frame, phy, snr_db, rng);
    }
    const size_t ref_samples = (opt.tx_awgn_reference == "preamble")
        ? phy.preamble.size()
        : 0;
    return add_awgn_to_time_samples(frame, snr_db, rng, ref_samples);
}

static void print_injection_channel(const char* label, const InjectionChannel& channel)
{
    if (channel.empty()) {
        return;
    }
    std::cout << "[CHANNEL] " << label
              << " files=" << channel.source_files.size()
              << " paths=" << channel.paths.size() << "\n";
}

static usrp_link::PredictiveAmcConfig make_predictive_amc_config(const Options& opt)
{
    usrp_link::PredictiveAmcConfig cfg;
    cfg.min_repeat = opt.tx_repeat_min;
    cfg.max_repeat = opt.tx_repeat_max;
    cfg.enable_16qam = opt.adaptive_enable_16qam;
    return cfg;
}

static usrp_link::ChannelPrediction make_channel_prediction_frame(const InjectionChannel& channel,
                                                                  const Options& opt,
                                                                  uint64_t target_frame_id)
{
    usrp_link::ChannelPrediction pred;
    pred.prediction_id = target_frame_id;
    pred.target_phy_frame_id = target_frame_id;
    pred.age_ms = opt.adaptive_prediction_age_ms;
    pred.confidence = static_cast<float>(opt.adaptive_prediction_confidence);

    if (channel.empty()) {
        pred.paths.push_back(usrp_link::PredictedPath{1.0f, 0.0f, 0.0f, 0.0f});
        return pred;
    }

    pred.paths.reserve(channel.paths.size());
    for (const auto& p : channel.paths) {
        pred.paths.push_back(usrp_link::PredictedPath{
            p.gain.real(),
            p.gain.imag(),
            static_cast<float>(p.doppler_hz),
            static_cast<float>(p.delay_samples),
        });
    }
    return pred;
}

static double predicted_fade_depth_db(const InjectionChannel& channel, const PhyConfig& phy)
{
    if (channel.empty()) {
        return 0.0;
    }

    double min_abs = std::numeric_limits<double>::infinity();
    double sum_abs = 0.0;
    uint64_t count = 0;
    for (int sym = 0; sym < phy.num_symbols; ++sym) {
        for (int sc = 0; sc < phy.active_sc; ++sc) {
            const double a = std::abs(ofdm_channel_response(channel, phy, sym, sc));
            min_abs = std::min(min_abs, a);
            sum_abs += a;
            ++count;
        }
    }
    if (count == 0 || !std::isfinite(min_abs)) {
        return 0.0;
    }
    const double mean_abs = sum_abs / static_cast<double>(count);
    return 20.0 * std::log10(std::max(min_abs, 1.0e-12) / std::max(mean_abs, 1.0e-12));
}

static usrp_link::ChannelQuality evaluate_predictive_quality(usrp_link::PredictiveAmcPolicy& policy,
                                                             const InjectionChannel& prediction_channel,
                                                             const Options& opt,
                                                             const PhyConfig& phy,
                                                             const usrp_link::ChannelPrediction& pred,
                                                             double predicted_snr_db)
{
    return policy.evaluate_prediction(
        pred,
        predicted_snr_db,
        predicted_fade_depth_db(prediction_channel, phy),
        opt.adaptive_prediction_evm);
}

static void print_predictive_decision(const char* tag,
                                      const usrp_link::AmcDecision& decision,
                                      const usrp_link::ChannelQuality& quality)
{
    std::cout << tag
              << " mcs=" << static_cast<int>(decision.mcs)
              << " modulation=" << decision.modulation
              << " repeat=" << decision.repeat_count
              << " effRate=" << std::fixed << std::setprecision(3) << decision.effective_rate
              << " snrForMcs=" << std::setprecision(2) << quality.snr_for_mcs_db
              << " fadeDepth=" << quality.fade_depth_db
              << " reason=\"" << decision.reason << "\"\n";
}

static bool find_frame(
    const std::deque<cf32>& buf,
    const PhyConfig& phy,
    size_t& start_idx,
    double& peak,
    double& cfo_hz)
{
    const size_t L = static_cast<size_t>(phy.pre_half_len);
    const size_t pre_len = 2 * L;
    if (buf.size() < static_cast<size_t>(phy.frame_len())) {
        return false;
    }

    const size_t search_limit = buf.size() - pre_len + 1;
    cd p(0.0, 0.0);
    double e1 = 0.0;
    double e2 = 0.0;
    for (size_t i = 0; i < L; ++i) {
        p += std::conj(cd(buf[i])) * cd(buf[i + L]);
        e1 += std::norm(buf[i]);
        e2 += std::norm(buf[i + L]);
    }

    size_t best = 0;
    double best_metric = 0.0;
    cd best_p = p;

    for (size_t d = 0; d < search_limit; ++d) {
        const double metric = std::norm(p) / std::max(e1 * e2, 1e-18);
        if (metric > best_metric) {
            best_metric = metric;
            best = d;
            best_p = p;
        }

        if (d + pre_len >= buf.size()) {
            break;
        }

        const cf32 x0 = buf[d];
        const cf32 x1 = buf[d + L];
        const cf32 x2 = buf[d + pre_len];
        p += std::conj(cd(x1)) * cd(x2) - std::conj(cd(x0)) * cd(x1);
        e1 += std::norm(x1) - std::norm(x0);
        e2 += std::norm(x2) - std::norm(x1);
    }

    if (best_metric < phy.sync_threshold) {
        return false;
    }

    const int refine_radius = std::min(16, phy.cp);
    const size_t lo = (best > static_cast<size_t>(refine_radius)) ? best - refine_radius : 0;
    const size_t hi = std::min(search_limit - 1, best + static_cast<size_t>(refine_radius));
    double pre_energy = 0.0;
    for (const auto& x : phy.preamble) {
        pre_energy += std::norm(x);
    }

    double best_corr_metric = 0.0;
    size_t best_corr = best;
    for (size_t d = lo; d <= hi; ++d) {
        cd corr(0.0, 0.0);
        double eng = 0.0;
        for (size_t i = 0; i < pre_len; ++i) {
            corr += cd(buf[d + i]) * std::conj(cd(phy.preamble[i]));
            eng += std::norm(buf[d + i]);
        }
        const double m = std::norm(corr) / std::max(eng * pre_energy, 1e-18);
        if (m > best_corr_metric) {
            best_corr_metric = m;
            best_corr = d;
        }
    }

    peak = best_corr_metric;
    if (peak < phy.sync_threshold) {
        return false;
    }

    cd cfo_acc(0.0, 0.0);
    for (size_t i = 0; i < L; ++i) {
        cfo_acc += std::conj(cd(buf[best_corr + i])) * cd(buf[best_corr + i + L]);
    }

    start_idx = best_corr;
    cfo_hz = std::arg(cfo_acc) * phy.rate / (2.0 * kPi * static_cast<double>(L));
    (void)best_p;
    return true;
}

static bool refine_frame_candidate_vector(
    const std::vector<cf32>& buf,
    size_t base,
    size_t coarse_start,
    const PhyConfig& phy,
    double coarse_metric,
    size_t available,
    SyncResult& out)
{
    const size_t L = static_cast<size_t>(phy.pre_half_len);
    const size_t pre_len = 2 * L;
    const size_t frame_len = static_cast<size_t>(phy.frame_len());
    const int refine_radius = std::min(16, phy.cp);
    const size_t search_limit = available - pre_len + 1;
    const size_t lo = (coarse_start > static_cast<size_t>(refine_radius)) ? coarse_start - refine_radius : 0;
    const size_t hi = std::min(search_limit - 1, coarse_start + static_cast<size_t>(refine_radius));

    double pre_energy = 0.0;
    for (const auto& x : phy.preamble) {
        pre_energy += std::norm(x);
    }

    double best_corr_metric = 0.0;
    size_t best_corr = coarse_start;
    for (size_t d = lo; d <= hi; ++d) {
        cd corr(0.0, 0.0);
        double eng = 0.0;
        for (size_t i = 0; i < pre_len; ++i) {
            corr += cd(buf[base + d + i]) * std::conj(cd(phy.preamble[i]));
            eng += std::norm(buf[base + d + i]);
        }
        const double m = std::norm(corr) / std::max(eng * pre_energy, 1e-18);
        if (m > best_corr_metric) {
            best_corr_metric = m;
            best_corr = d;
        }
    }

    if (best_corr_metric < phy.sync_threshold) {
        return false;
    }
    if (best_corr + frame_len > available) {
        out.incomplete = true;
        out.start = best_corr;
        out.coarse_metric = coarse_metric;
        out.fine_peak = best_corr_metric;
        return false;
    }

    cd cfo_acc(0.0, 0.0);
    for (size_t i = 0; i < L; ++i) {
        cfo_acc += std::conj(cd(buf[base + best_corr + i])) * cd(buf[base + best_corr + i + L]);
    }

    out.found = true;
    out.start = best_corr;
    out.coarse_metric = coarse_metric;
    out.fine_peak = best_corr_metric;
    out.cfo_hz = std::arg(cfo_acc) * phy.rate / (2.0 * kPi * static_cast<double>(L));
    return true;
}

static bool find_frame_vector_earliest(
    const std::vector<cf32>& buf,
    size_t base,
    const PhyConfig& phy,
    size_t max_search_samples,
    SyncResult& out)
{
    out = SyncResult{};
    const size_t L = static_cast<size_t>(phy.pre_half_len);
    const size_t pre_len = 2 * L;
    if (buf.size() < base || buf.size() - base < pre_len) {
        return false;
    }

    const size_t available = buf.size() - base;
    size_t search_limit = available - pre_len + 1;
    if (max_search_samples > 0) {
        search_limit = std::min(search_limit, max_search_samples + 1);
    }

    cd p(0.0, 0.0);
    double e1 = 0.0;
    double e2 = 0.0;
    for (size_t i = 0; i < L; ++i) {
        p += std::conj(cd(buf[base + i])) * cd(buf[base + i + L]);
        e1 += std::norm(buf[base + i]);
        e2 += std::norm(buf[base + i + L]);
    }

    for (size_t d = 0; d < search_limit; ++d) {
        const double metric = std::norm(p) / std::max(e1 * e2, 1e-18);
        if (metric >= phy.sync_threshold) {
            if (refine_frame_candidate_vector(buf, base, d, phy, metric, available, out)) {
                return true;
            }
            if (out.incomplete) {
                return false;
            }
        }

        if (d + pre_len >= available) {
            break;
        }

        const cf32 x0 = buf[base + d];
        const cf32 x1 = buf[base + d + L];
        const cf32 x2 = buf[base + d + pre_len];
        p += std::conj(cd(x1)) * cd(x2) - std::conj(cd(x0)) * cd(x1);
        e1 += std::norm(x1) - std::norm(x0);
        e2 += std::norm(x2) - std::norm(x1);
    }

    return false;
}

static FrameDecodeResult decode_frame(
    const std::vector<cf32>& rx_frame,
    const PhyConfig& phy,
    const LdpcCode& code,
    CudaBpDecoder* cuda_bp_decoder,
    CudaBpOsdDecoder* cuda_decoder,
    double cfo_hz,
    double peak,
    int ldpc_max_iter,
    double ldpc_normalization,
    double rx_snr_gate_db,
    bool reference_test,
    DecodeProfile* profile = nullptr,
    std::optional<uint32_t> expected_reference_frame_id = std::nullopt,
    std::optional<double> known_frequency_noise_var = std::nullopt)
{
    if (static_cast<int>(rx_frame.size()) != phy.frame_len()) {
        throw std::runtime_error("decode_frame received a partial frame");
    }

    submit_ui_spectrum_iq_if_due(rx_frame);

    const auto t_decode0 = std::chrono::steady_clock::now();
    std::vector<cf32> frame = rx_frame;
    for (size_t n = 0; n < frame.size(); ++n) {
        const double a = -2.0 * kPi * cfo_hz * static_cast<double>(n) / phy.rate;
        frame[n] *= cf32(static_cast<float>(std::cos(a)), static_cast<float>(std::sin(a)));
    }
    const auto t_cfo1 = std::chrono::steady_clock::now();

    std::vector<std::vector<cf32>> y(
        static_cast<size_t>(phy.num_symbols),
        std::vector<cf32>(static_cast<size_t>(phy.active_sc)));

    size_t off = phy.preamble.size();
    for (int sym = 0; sym < phy.num_symbols; ++sym) {
        const cf32* td = frame.data() + off + static_cast<size_t>(phy.cp);
        auto shifted = fft_to_shifted_grid(td, phy.nfft);
        for (int sc = 0; sc < phy.active_sc; ++sc) {
            y[sym][sc] = shifted[phy.used_indices[sc]];
        }
        off += static_cast<size_t>(phy.sym_len());
    }
    const auto t_fft1 = std::chrono::steady_clock::now();

    std::vector<cf32> yd;
    std::vector<cf32> hd;
    yd.reserve(static_cast<size_t>(phy.active_sc * phy.data_symbols.size()));
    hd.reserve(static_cast<size_t>(phy.active_sc * phy.data_symbols.size()));

    if (phy.rx_channel_estimation == "none") {
        for (int sym : phy.data_symbols) {
            for (int sc = 0; sc < phy.active_sc; ++sc) {
                yd.push_back(y[sym][sc]);
                hd.push_back(cf32(1.0f, 0.0f));
            }
        }
    } else {
        std::vector<std::vector<cf32>> hp(
            phy.pilot_symbols.size(),
            std::vector<cf32>(static_cast<size_t>(phy.active_sc)));

        for (size_t p = 0; p < phy.pilot_symbols.size(); ++p) {
            const int sym = phy.pilot_symbols[p];
            for (int sc = 0; sc < phy.active_sc; ++sc) {
                const cf32 pilot = phy.pilot_grid[p][sc];
                const float den = std::max(std::norm(pilot), 1e-9f);
                hp[p][sc] = y[sym][sc] * std::conj(pilot) / den;
            }
        }

        for (int sym : phy.data_symbols) {
            size_t hi = 0;
            while (hi + 1 < phy.pilot_symbols.size() && sym > phy.pilot_symbols[hi + 1]) {
                ++hi;
            }

            size_t p0 = hi;
            size_t p1 = std::min(hi + 1, phy.pilot_symbols.size() - 1);
            if (sym <= phy.pilot_symbols.front()) {
                p0 = p1 = 0;
            } else if (sym >= phy.pilot_symbols.back()) {
                p0 = p1 = phy.pilot_symbols.size() - 1;
            }

            double alpha = 0.0;
            if (p0 != p1) {
                alpha = static_cast<double>(sym - phy.pilot_symbols[p0]) /
                        static_cast<double>(phy.pilot_symbols[p1] - phy.pilot_symbols[p0]);
            }

            for (int sc = 0; sc < phy.active_sc; ++sc) {
                const cf32 h = hp[p0][sc] * static_cast<float>(1.0 - alpha) + hp[p1][sc] * static_cast<float>(alpha);
                yd.push_back(y[sym][sc]);
                hd.push_back(h);
            }
        }
    }
    const auto t_channel1 = std::chrono::steady_clock::now();

    double measured_noise_var = 0.0;
    double equalized_error_power = 0.0;
    double equalized_reference_power = 0.0;
    const int ui_max_points =
        (g_ui_telemetry != nullptr && g_ui_telemetry->try_begin_constellation_capture())
            ? g_ui_telemetry->max_constellation_points()
            : 0;
    const bool capture_ui_constellation = ui_max_points > 0;

    for (size_t i = 0; i < yd.size(); ++i) {
        const float h2 = std::max(std::norm(hd[i]), 1e-9f);
        const cf32 eq0 = yd[i] * std::conj(hd[i]) / h2;
        const cf32 hard = nearest_constellation_symbol(phy, eq0);
        const cf32 resid = yd[i] - hd[i] * hard;
        measured_noise_var += std::norm(resid);
        equalized_error_power += std::norm(eq0 - hard);
        equalized_reference_power += std::norm(hard);
    }
    measured_noise_var = std::max(measured_noise_var / std::max<size_t>(yd.size(), 1), 1e-8);
    const bool use_known_noise =
        known_frequency_noise_var.has_value() &&
        std::isfinite(*known_frequency_noise_var) &&
        *known_frequency_noise_var > 0.0;
    const double noise_var = use_known_noise
        ? std::max(*known_frequency_noise_var, 1e-8)
        : measured_noise_var;
    double hpow = 0.0;
    for (const auto& h : hd) {
        hpow += std::norm(h);
    }
    hpow /= std::max<size_t>(hd.size(), 1);
    const double estimated_snr_db = 10.0 * std::log10(std::max(hpow / noise_var, 1e-12));
    const double estimated_evm_rms = std::sqrt(
        std::max(equalized_error_power, 0.0) /
        std::max(equalized_reference_power, 1.0e-12));
    if (capture_ui_constellation) {
        const size_t ui_points = std::min(static_cast<size_t>(ui_max_points), yd.size());
        std::vector<cf32> ui_tx_points;
        std::vector<cf32> ui_rx_points;
        ui_tx_points.reserve(ui_points);
        ui_rx_points.reserve(ui_points);
        for (size_t j = 0; j < ui_points; ++j) {
            const size_t i = (ui_points <= 1 || yd.size() <= 1)
                ? 0
                : (j * (yd.size() - 1)) / (ui_points - 1);
            const float h2 = std::max(std::norm(hd[i]), 1e-9f);
            const cf32 eq0 = yd[i] * std::conj(hd[i]) / h2;
            ui_tx_points.push_back(nearest_constellation_symbol(phy, eq0));
            ui_rx_points.push_back(eq0);
        }
        g_ui_telemetry->submit_constellation(std::move(ui_tx_points), std::move(ui_rx_points));
    }

    if (estimated_snr_db < rx_snr_gate_db) {
        FrameDecodeResult out;
        out.detected = true;
        out.parity_ok = false;
        out.iterations = 0;
        out.frame_id = 0;
        out.peak = peak;
        out.cfo_hz = cfo_hz;
        out.snr_db = estimated_snr_db;
        out.evm_rms = estimated_evm_rms;
        out.bit_errors = -1;
        out.pre_fec_bit_errors = -1;
        out.frame_ok = false;
        if (profile != nullptr) {
            const auto t_ref1 = std::chrono::steady_clock::now();
            ++profile->frames;
            profile->cfo_ns += elapsed_ns(t_decode0, t_cfo1);
            profile->fft_ns += elapsed_ns(t_cfo1, t_fft1);
            profile->channel_ns += elapsed_ns(t_fft1, t_channel1);
            profile->reference_ns += elapsed_ns(t_channel1, t_ref1);
        }
        return out;
    }

    std::vector<float> llr(static_cast<size_t>(phy.coded_bits_per_frame()), 0.0f);
    std::vector<uint8_t> coded_hard(static_cast<size_t>(phy.coded_bits_per_frame()), 0);

    for (size_t i = 0; i < yd.size(); ++i) {
        const float h2 = std::max(std::norm(hd[i]), 1e-9f);
        const cf32 eq = yd[i] * std::conj(hd[i]) / (h2 + static_cast<float>(noise_var));
        const double sigma2_eq = noise_var / h2;
        float* dst = llr.data() + i * static_cast<size_t>(phy.bits_per_symbol);
        symbol_to_llr_maxlog(phy, eq, sigma2_eq, dst);
        for (int bit = 0; bit < phy.bits_per_symbol; ++bit) {
            const size_t bi = i * static_cast<size_t>(phy.bits_per_symbol) + static_cast<size_t>(bit);
            coded_hard[bi] = (llr[bi] < 0.0f) ? 1 : 0;
        }
    }
    const auto t_llr1 = std::chrono::steady_clock::now();

    std::vector<uint8_t> info_bits;
    std::vector<uint8_t> decoded_coded_bits;
    info_bits.reserve(static_cast<size_t>((phy.coded_bits_per_frame() / code.n) * code.k));
    decoded_coded_bits.reserve(static_cast<size_t>(phy.coded_bits_per_frame()));

    int iter_sum = 0;
    bool parity_ok = true;
    const int blocks = phy.coded_bits_per_frame() / code.n;
    if (cuda_bp_decoder != nullptr) {
        int success_count = 0;
        double total_decode_ms = 0.0;
        info_bits = cuda_bp_decoder->decode(llr.data(), blocks, success_count, total_decode_ms);
        parity_ok = (success_count == blocks);
        iter_sum = success_count;
    } else if (cuda_decoder != nullptr) {
        int used_osd_count = 0;
        info_bits = cuda_decoder->decode(llr.data(), blocks, used_osd_count);
        iter_sum = used_osd_count;
    } else {
        for (int b = 0; b < blocks; ++b) {
            auto dr = ldpc_decode_min_sum(
                code,
                llr.data() + static_cast<size_t>(b * code.n),
                ldpc_max_iter,
                ldpc_normalization);
            parity_ok = parity_ok && dr.parity_ok;
            iter_sum += dr.iterations;
            info_bits.insert(info_bits.end(), dr.info_bits.begin(), dr.info_bits.end());
            decoded_coded_bits.insert(decoded_coded_bits.end(), dr.code_bits.begin(), dr.code_bits.end());
        }
    }
    const auto t_fec1 = std::chrono::steady_clock::now();

    FrameDecodeResult out;
    out.detected = true;
    out.parity_ok = parity_ok;
    out.iterations = blocks > 0 ? static_cast<int>(std::lround(static_cast<double>(iter_sum) / blocks)) : 0;
    out.frame_id = reference_test ? bits_to_u32_msb(info_bits, 0) : 0;
    out.peak = peak;
    out.cfo_hz = cfo_hz;

    out.snr_db = estimated_snr_db;
    out.evm_rms = estimated_evm_rms;

    if (reference_test) {
        const uint32_t reference_frame_id = expected_reference_frame_id.value_or(out.frame_id);
        const auto ref_info = build_info_bits(phy, code, reference_frame_id);
        int bit_errors = 0;
        for (size_t i = 0; i < std::min(info_bits.size(), ref_info.size()); ++i) {
            bit_errors += (info_bits[i] != ref_info[i]) ? 1 : 0;
        }
        out.bit_errors = bit_errors;

        const auto ref_coded = encode_frame_bits(phy, code, ref_info);
        int pre_errors = 0;
        for (size_t i = 0; i < std::min(coded_hard.size(), ref_coded.size()); ++i) {
            pre_errors += (coded_hard[i] != ref_coded[i]) ? 1 : 0;
        }
        out.pre_fec_bit_errors = pre_errors;
        out.frame_ok = out.parity_ok && out.bit_errors == 0;
    } else {
        out.bit_errors = -1;
        out.pre_fec_bit_errors = -1;
        out.frame_ok = false;
    }
    out.info_bits = std::move(info_bits);
    const auto t_ref1 = std::chrono::steady_clock::now();

    if (profile != nullptr) {
        ++profile->frames;
        profile->cfo_ns += elapsed_ns(t_decode0, t_cfo1);
        profile->fft_ns += elapsed_ns(t_cfo1, t_fft1);
        profile->channel_ns += elapsed_ns(t_fft1, t_channel1);
        profile->llr_ns += elapsed_ns(t_channel1, t_llr1);
        profile->fec_ns += elapsed_ns(t_llr1, t_fec1);
        profile->reference_ns += elapsed_ns(t_fec1, t_ref1);
    }

    return out;
}

static bool process_rx_buffer(
    std::deque<cf32>& rxbuf,
    const PhyConfig& phy,
    const LdpcCode& code,
    CudaBpDecoder* cuda_bp_decoder,
    CudaBpOsdDecoder* cuda_decoder,
    int ldpc_max_iter,
    double ldpc_normalization,
    bool reference_test,
    FrameDecodeResult& result,
    std::optional<uint32_t> expected_reference_frame_id = std::nullopt,
    std::optional<double> known_frequency_noise_var = std::nullopt)
{
    size_t start = 0;
    double peak = 0.0;
    double cfo = 0.0;
    if (!find_frame(rxbuf, phy, start, peak, cfo)) {
        const size_t keep = static_cast<size_t>(phy.max_buffered_frames * phy.frame_len());
        if (rxbuf.size() > keep) {
            rxbuf.erase(rxbuf.begin(), rxbuf.end() - static_cast<std::ptrdiff_t>(phy.frame_len()));
        }
        return false;
    }

    if (start + static_cast<size_t>(phy.frame_len()) > rxbuf.size()) {
        return false;
    }

    std::vector<cf32> one(static_cast<size_t>(phy.frame_len()));
    for (size_t i = 0; i < one.size(); ++i) {
        one[i] = rxbuf[start + i];
    }

    result = decode_frame(
        one,
        phy,
        code,
        cuda_bp_decoder,
        cuda_decoder,
        cfo,
        peak,
        ldpc_max_iter,
        ldpc_normalization,
        -120.0,
        reference_test,
        nullptr,
        expected_reference_frame_id,
        known_frequency_noise_var);
    rxbuf.erase(rxbuf.begin(), rxbuf.begin() + static_cast<std::ptrdiff_t>(start + phy.frame_len()));
    return true;
}

struct ExtractedFrame {
    std::vector<cf32> samples;
    double peak = 0.0;
    double cfo_hz = 0.0;
};

#ifdef HAVE_GPU_FULL_PIPELINE
static void check_cuda_runtime(cudaError_t status, const char* what);

class DeviceFramePool {
public:
    struct Lease {
        DeviceFramePool* pool = nullptr;
        size_t index = 0;
        cf32* ptr = nullptr;

        Lease(DeviceFramePool* owner, size_t slot_index, cf32* device_ptr)
            : pool(owner), index(slot_index), ptr(device_ptr)
        {}

        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;

        ~Lease()
        {
            if (pool != nullptr) {
                pool->release(index);
            }
        }
    };

    DeviceFramePool(size_t capacity_frames, size_t samples_per_frame)
        : capacity_frames_(std::max<size_t>(capacity_frames, 4)),
          samples_per_frame_(samples_per_frame)
    {
        check_cuda_runtime(
            cudaMalloc(&storage_, capacity_frames_ * samples_per_frame_ * sizeof(cf32)),
            "cudaMalloc RX device frame pool");
        free_indices_.clear();
        for (size_t i = 0; i < capacity_frames_; ++i) {
            free_indices_.push_back(i);
        }
    }

    ~DeviceFramePool()
    {
        if (storage_ != nullptr) {
            cudaFree(storage_);
        }
    }

    DeviceFramePool(const DeviceFramePool&) = delete;
    DeviceFramePool& operator=(const DeviceFramePool&) = delete;

    std::shared_ptr<Lease> try_acquire()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (free_indices_.empty()) {
            ++exhausted_count_;
            return {};
        }
        const size_t index = free_indices_.front();
        free_indices_.pop_front();
        return std::make_shared<Lease>(
            this,
            index,
            storage_ + index * samples_per_frame_);
    }

    uint64_t exhausted_count() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return exhausted_count_;
    }

private:
    void release(size_t index)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        free_indices_.push_back(index);
    }

    friend struct Lease;

    mutable std::mutex mutex_;
    std::deque<size_t> free_indices_;
    cf32* storage_ = nullptr;
    size_t capacity_frames_ = 0;
    size_t samples_per_frame_ = 0;
    uint64_t exhausted_count_ = 0;
};

struct ExtractedDeviceFrame {
    cf32* device_samples = nullptr;
    size_t sample_count = 0;
    double peak = 0.0;
    double cfo_hz = 0.0;
    std::shared_ptr<DeviceFramePool::Lease> lease;

    ExtractedDeviceFrame() = default;
    ~ExtractedDeviceFrame()
    {
        release();
    }

    ExtractedDeviceFrame(const ExtractedDeviceFrame&) = delete;
    ExtractedDeviceFrame& operator=(const ExtractedDeviceFrame&) = delete;

    ExtractedDeviceFrame(ExtractedDeviceFrame&& other) noexcept
    {
        move_from(other);
    }

    ExtractedDeviceFrame& operator=(ExtractedDeviceFrame&& other) noexcept
    {
        if (this != &other) {
            release();
            move_from(other);
        }
        return *this;
    }

    void release()
    {
        lease.reset();
        device_samples = nullptr;
        sample_count = 0;
        peak = 0.0;
        cfo_hz = 0.0;
    }

private:
    void move_from(ExtractedDeviceFrame& other) noexcept
    {
        device_samples = other.device_samples;
        sample_count = other.sample_count;
        peak = other.peak;
        cfo_hz = other.cfo_hz;
        lease = std::move(other.lease);
        other.device_samples = nullptr;
        other.sample_count = 0;
        other.peak = 0.0;
        other.cfo_hz = 0.0;
    }
};
#endif

static bool extract_rx_frame(std::deque<cf32>& rxbuf, const PhyConfig& phy, ExtractedFrame& frame)
{
    size_t start = 0;
    double peak = 0.0;
    double cfo = 0.0;
    if (!find_frame(rxbuf, phy, start, peak, cfo)) {
        const size_t keep = static_cast<size_t>(phy.max_buffered_frames * phy.frame_len());
        if (rxbuf.size() > keep) {
            rxbuf.erase(rxbuf.begin(), rxbuf.end() - static_cast<std::ptrdiff_t>(phy.frame_len()));
        }
        return false;
    }

    if (start + static_cast<size_t>(phy.frame_len()) > rxbuf.size()) {
        return false;
    }

    frame.samples.resize(static_cast<size_t>(phy.frame_len()));
    for (size_t i = 0; i < frame.samples.size(); ++i) {
        frame.samples[i] = rxbuf[start + i];
    }
    frame.peak = peak;
    frame.cfo_hz = cfo;
    rxbuf.erase(rxbuf.begin(), rxbuf.begin() + static_cast<std::ptrdiff_t>(start + phy.frame_len()));
    return true;
}

static void compact_rx_vector(std::vector<cf32>& rxbuf, size_t& base)
{
    if (base == 0) {
        return;
    }
    if (base > 4 * 1024 * 1024 || base > rxbuf.size() / 2) {
        rxbuf.erase(rxbuf.begin(), rxbuf.begin() + static_cast<std::ptrdiff_t>(base));
        base = 0;
    }
}

static bool extract_rx_frame_vector(
    std::vector<cf32>& rxbuf,
    size_t& base,
    const PhyConfig& phy,
    ExtractedFrame& frame,
    SyncState& sync_state,
    SyncStats& sync_stats,
    size_t acquisition_max_search_samples = 0)
{
    if (rxbuf.size() < base || rxbuf.size() - base < phy.preamble.size()) {
        return false;
    }

    SyncResult sr;
    bool found = false;
    if (sync_state == SyncState::Tracking) {
        const size_t tracking_window = static_cast<size_t>(std::max(256, phy.sync_tracking_window));
        found = find_frame_vector_earliest(rxbuf, base, phy, tracking_window, sr);
        const size_t strict_window = static_cast<size_t>(std::max(256, 4 * phy.cp));
        if (found &&
            phy.sync_late_peak_threshold > 0.0 &&
            sr.start > strict_window &&
            sr.fine_peak < phy.sync_late_peak_threshold) {
            found = false;
        }
        if (!found && !sr.incomplete) {
            ++sync_stats.tracking_miss;
            sync_state = SyncState::Acquisition;
        }
    }

    if (sync_state == SyncState::Acquisition) {
        found = find_frame_vector_earliest(rxbuf, base, phy, acquisition_max_search_samples, sr);
        if (found) {
            sync_state = SyncState::Tracking;
        }
    }

    if (!found) {
        ++sync_stats.miss;
        if (sr.incomplete) {
            ++sync_stats.incomplete;
            return false;
        }
        const size_t keep = static_cast<size_t>(phy.max_buffered_frames * phy.frame_len());
        if (rxbuf.size() >= base && rxbuf.size() - base > keep) {
            base = rxbuf.size() - static_cast<size_t>(phy.frame_len());
            compact_rx_vector(rxbuf, base);
        }
        return false;
    }

    if (base + sr.start + static_cast<size_t>(phy.frame_len()) > rxbuf.size()) {
        ++sync_stats.incomplete;
        return false;
    }

    const auto frame_begin = rxbuf.begin() + static_cast<std::ptrdiff_t>(base + sr.start);
    const auto frame_end = frame_begin + static_cast<std::ptrdiff_t>(phy.frame_len());
    frame.samples.assign(frame_begin, frame_end);
    frame.peak = sr.fine_peak;
    frame.cfo_hz = sr.cfo_hz;
    ++sync_stats.found;
    sync_stats.skipped_samples += static_cast<uint64_t>(sr.start);
    sync_stats.max_start = std::max<uint64_t>(sync_stats.max_start, static_cast<uint64_t>(sr.start));
    base += sr.start + static_cast<size_t>(phy.frame_len());
    compact_rx_vector(rxbuf, base);
    return true;
}

static void update_stats(RxStats& stats, const FrameDecodeResult& r, int info_bits_per_frame, int coded_bits_per_frame)
{
    ++stats.detected;
    stats.sum_iter += static_cast<uint64_t>(std::max(r.iterations, 0));
    stats.sum_snr_db += r.snr_db;
    stats.min_snr_db = std::min(stats.min_snr_db, r.snr_db);
    stats.sum_abs_cfo_hz += std::abs(r.cfo_hz);
    if (std::isfinite(r.evm_rms)) {
        stats.sum_evm_rms += r.evm_rms;
        ++stats.evm_samples;
    }
    if (r.pre_fec_bit_errors >= 0) {
        stats.pre_fec_bit_errors += static_cast<uint64_t>(r.pre_fec_bit_errors);
        stats.pre_fec_bits += static_cast<uint64_t>(coded_bits_per_frame);
    }
    if (r.bit_errors >= 0) {
        stats.info_bits += static_cast<uint64_t>(info_bits_per_frame);
        stats.bit_errors += static_cast<uint64_t>(r.bit_errors);
    }

    if (r.frame_ok) {
        ++stats.ok;
        stats.ok_info_bits += static_cast<uint64_t>(std::max(info_bits_per_frame, 0));
    } else {
        ++stats.err;
    }
}

static void print_frame_result(const char* prefix, const FrameDecodeResult& r)
{
    std::cout << prefix
              << " frame=" << r.frame_id
              << " peak=" << std::fixed << std::setprecision(3) << r.peak
              << " cfo=" << std::setprecision(1) << r.cfo_hz << " Hz"
              << " snr=" << std::setprecision(1) << r.snr_db << " dB";
    if (r.pre_fec_bit_errors >= 0) {
        std::cout << " preErr=" << r.pre_fec_bit_errors;
    } else {
        std::cout << " preErr=n/a";
    }
    if (r.bit_errors >= 0) {
        std::cout << " postErr=" << r.bit_errors;
    } else {
        std::cout << " postErr=n/a";
    }
    std::cout << " iter=" << r.iterations
              << " parity=" << (r.parity_ok ? "ok" : "fail")
              << " ok=" << (r.frame_ok ? "yes" : "no")
              << "\n";
}

static void print_stats(const RxStats& stats, double elapsed, int info_bits_per_frame)
{
    const double fer = static_cast<double>(stats.err) / std::max<uint64_t>(stats.detected, 1);
    const double goodput = static_cast<double>(stats.ok_info_bits) /
                           std::max(elapsed, 1e-9) / 1e6;
    const double fps = static_cast<double>(stats.detected) / std::max(elapsed, 1e-9);
    const double avg_iter = static_cast<double>(stats.sum_iter) / std::max<uint64_t>(stats.detected, 1);
    const double avg_snr = stats.sum_snr_db / static_cast<double>(std::max<uint64_t>(stats.detected, 1));
    const double min_snr = std::isfinite(stats.min_snr_db) ? stats.min_snr_db : 0.0;
    const double avg_abs_cfo = stats.sum_abs_cfo_hz / static_cast<double>(std::max<uint64_t>(stats.detected, 1));

    std::cout << "---- RX summary ----\n"
              << "frames=" << stats.detected
              << " ok=" << stats.ok
              << " err=" << stats.err
              << " FER=" << std::scientific << std::setprecision(3) << fer
              << " BER=";
    if (stats.info_bits > 0) {
        std::cout << static_cast<double>(stats.bit_errors) / static_cast<double>(stats.info_bits);
    } else {
        std::cout << "n/a";
    }
    std::cout << " preBER=";
    if (stats.pre_fec_bits > 0) {
        std::cout << static_cast<double>(stats.pre_fec_bit_errors) / static_cast<double>(stats.pre_fec_bits);
    } else {
        std::cout << "n/a";
    }
    std::cout << std::fixed << std::setprecision(2)
              << " fps=" << fps
              << " goodput=" << goodput << " Mbps"
              << " avgSNR=" << avg_snr << " dB"
              << " minSNR=" << min_snr << " dB"
              << " avgAbsCFO=" << avg_abs_cfo << " Hz"
              << " avgEVM=";
    if (stats.evm_samples > 0) {
        std::cout << (stats.sum_evm_rms / static_cast<double>(stats.evm_samples));
    } else {
        std::cout << "n/a";
    }
    std::cout << " avgIter=" << avg_iter << "\n";
}

static void send_ui_metrics_if_due(
    const RxStats& stats,
    double elapsed,
    int info_bits_per_frame,
    const FrameDecodeResult& last_frame,
    int mcs_index = -1,
    int total_frames = 0,
    double relative_power_db = std::numeric_limits<double>::quiet_NaN())
{
    if (g_ui_telemetry == nullptr || stats.detected == 0) {
        return;
    }
    if (!g_ui_telemetry->try_begin_metrics_sample()) {
        return;
    }

    const double fer = static_cast<double>(stats.err) / std::max<uint64_t>(stats.detected, 1);
    const double ber = stats.info_bits > 0
        ? static_cast<double>(stats.bit_errors) / static_cast<double>(stats.info_bits)
        : 0.0;
    const double preber = stats.pre_fec_bits > 0
        ? static_cast<double>(stats.pre_fec_bit_errors) / static_cast<double>(stats.pre_fec_bits)
        : 0.0;
    const double goodput = static_cast<double>(stats.ok_info_bits) /
                           std::max(elapsed, 1e-9) / 1e6;
    UiMetricsSample sample;
    sample.frames = static_cast<double>(stats.detected);
    sample.total_frames = static_cast<double>(std::max(total_frames, 0));
    sample.ber = ber;
    sample.ber_valid = stats.info_bits > 0;
    sample.fer = fer;
    sample.preber = preber;
    sample.snr_db = last_frame.snr_db;
    sample.relative_power_db = relative_power_db;
    sample.goodput_mbps = goodput;
    sample.mcs_index = mcs_index;
    g_ui_telemetry->submit_metrics(sample);
}

static void print_pipeline_profile(
    const PipelineProfile& p,
    double elapsed,
    const PhyConfig& phy,
    int info_bits_per_frame)
{
    const uint64_t rx_samples = p.rx_samples.load(std::memory_order_relaxed);
    const uint64_t rx_blocks = p.rx_blocks.load(std::memory_order_relaxed);
    const uint64_t sync_frames = p.sync_frames.load(std::memory_order_relaxed);
    const uint64_t sync_blocks = p.sync_blocks.load(std::memory_order_relaxed);
    const uint64_t sync_samples = p.sync_input_samples.load(std::memory_order_relaxed);
    const uint64_t tx_frames = p.tx_frames.load(std::memory_order_relaxed);
    const uint64_t media_frames = p.media_frames.load(std::memory_order_relaxed);
    const uint64_t media_accepts = p.media_accepts.load(std::memory_order_relaxed);
    const uint64_t dec_frames = p.decode.frames;
    const double sec = std::max(elapsed, 1e-9);

    auto ms_per = [](uint64_t ns, uint64_t n) {
        return n > 0 ? static_cast<double>(ns) / static_cast<double>(n) / 1.0e6 : 0.0;
    };
    auto fps_cap = [](double ms) {
        return ms > 0.0 ? 1000.0 / ms : 0.0;
    };

    const double cfo_ms = ms_per(p.decode.cfo_ns, dec_frames);
    const double fft_ms = ms_per(p.decode.fft_ns, dec_frames);
    const double channel_ms = ms_per(p.decode.channel_ns, dec_frames);
    const double llr_ms = ms_per(p.decode.llr_ns, dec_frames);
    const double fec_ms = ms_per(p.decode.fec_ns, dec_frames);
    const double ref_ms = ms_per(p.decode.reference_ns, dec_frames);
    const double decode_ms = cfo_ms + fft_ms + channel_ms + llr_ms + fec_ms + ref_ms;
    const double media_ms = ms_per(p.media_ns.load(std::memory_order_relaxed), media_frames);

    std::cout << "---- RX pipeline profile ----\n"
              << "tx: frames=" << tx_frames
              << " fps=" << (static_cast<double>(tx_frames) / sec)
              << " info=" << (static_cast<double>(tx_frames) * static_cast<double>(info_bits_per_frame) / sec / 1e6)
              << " Mbps\n"
              << "uhd_rx: samples=" << rx_samples
              << " blocks=" << rx_blocks
              << " Msps=" << (static_cast<double>(rx_samples) / sec / 1e6)
              << " recv_ms/block=" << ms_per(p.rx_recv_ns.load(std::memory_order_relaxed), rx_blocks)
              << " enqueue_ms/block=" << ms_per(p.rx_enqueue_ns.load(std::memory_order_relaxed), rx_blocks)
              << "\n"
              << "sync_extract: inputMsps=" << (static_cast<double>(sync_samples) / sec / 1e6)
              << " blocks=" << sync_blocks
              << " frames=" << sync_frames
              << " fps=" << (static_cast<double>(sync_frames) / sec)
              << " ms/block=" << ms_per(p.sync_ns.load(std::memory_order_relaxed), sync_blocks)
              << "\n"
              << "decode_consume: frames=" << dec_frames
              << " fps=" << (static_cast<double>(dec_frames) / sec)
              << " info=" << (static_cast<double>(dec_frames) * static_cast<double>(info_bits_per_frame) / sec / 1e6)
              << " Mbps"
              << " total_ms/frame=" << decode_ms
              << " cap_fps=" << fps_cap(decode_ms)
              << "\n"
              << "decode_breakdown_ms/frame:"
              << " cfo=" << cfo_ms
              << " fft=" << fft_ms
              << " channel=" << channel_ms
              << " llr=" << llr_ms
              << " fec=" << fec_ms
              << " ref_ber=" << ref_ms
              << "\n"
              << "media_write: frames=" << media_frames
              << " accepted=" << media_accepts
              << " fps=" << (static_cast<double>(media_frames) / sec)
              << " ms/frame=" << media_ms
              << " cap_fps=" << fps_cap(media_ms)
              << "\n"
              << "nominal: frameSamples=" << phy.frame_len()
              << " txFrameRate=" << (phy.rate / static_cast<double>(phy.frame_len()))
              << " info=" << (phy.rate / static_cast<double>(phy.frame_len()) * static_cast<double>(info_bits_per_frame) / 1e6)
              << " Mbps\n";
}

static int link_info_bits_per_frame(const PhyConfig& phy, const LdpcCode& code)
{
    return (phy.coded_bits_per_frame() / code.n) * code.k;
}

static size_t link_info_bytes_per_frame(const PhyConfig& phy, const LdpcCode& code)
{
    const int bits = link_info_bits_per_frame(phy, code);
    if (bits % 8 != 0) {
        throw std::runtime_error("media mode requires an integer number of information bytes per frame");
    }
    return static_cast<size_t>(bits / 8);
}

static size_t media_payload_capacity_bytes(const PhyConfig& phy, const LdpcCode& code)
{
    const size_t info_bytes = link_info_bytes_per_frame(phy, code);
    if (info_bytes <= kMediaHeaderBytes) {
        throw std::runtime_error("LDPC information payload is too small for media mode");
    }
    return info_bytes - kMediaHeaderBytes;
}

static bool accept_media_decode(FrameDecodeResult& r,
                                MediaReassembler& media,
                                size_t payload_capacity,
                                LinkControlState* control_state = nullptr)
{
    r.bit_errors = -1;
    r.pre_fec_bit_errors = -1;

    const auto bytes = bits_to_bytes_msb(r.info_bits);
    MediaHeader h;
    if (!parse_media_header(bytes, h)) {
        r.frame_ok = false;
        return false;
    }

    r.frame_id = h.frame_id;
    const auto payload_begin = bytes.begin() + static_cast<std::ptrdiff_t>(kMediaHeaderBytes);
    const auto payload_end = payload_begin + static_cast<std::ptrdiff_t>(h.payload_len);
    const std::vector<uint8_t> payload(payload_begin, payload_end);
    if ((h.flags & kMediaFlagControl) != 0) {
        LinkControlMessage msg;
        const bool ok = parse_control_payload(payload, msg);
        if (ok && control_state != nullptr) {
            control_state->accept(msg);
        }
        r.frame_ok = ok;
        return ok;
    }
    const bool ok = media.accept(h, payload, payload_capacity);
    r.frame_ok = ok;
    return ok;
}

static void print_link_config(const PhyConfig& phy, const LdpcCode& code)
{
    const int blocks = phy.coded_bits_per_frame() / code.n;
    const int info_bits = blocks * code.k;
    const double frame_rate = phy.rate / static_cast<double>(phy.frame_len());
    const double info_rate = frame_rate * static_cast<double>(info_bits) / 1e6;
    const double coded_rate = frame_rate * static_cast<double>(phy.coded_bits_per_frame()) / 1e6;

    std::cout << "PHY: Nfft=" << phy.nfft
              << " CP=" << phy.cp
              << " symbols=" << phy.num_symbols
              << " activeSC=" << phy.active_sc
              << " modulation=" << phy.modulation
              << " bits/sym=" << phy.bits_per_symbol
              << " sync=" << phy.sync_preamble
              << " preHalf=" << phy.pre_half_len
              << " preLen=" << phy.preamble.size()
              << " trackWin=" << phy.sync_tracking_window
              << " latePeak=" << phy.sync_late_peak_threshold
              << " rxChanEst=" << phy.rx_channel_estimation
              << " frameSamples=" << phy.frame_len()
              << " rate=" << phy.rate / 1e6 << " Msps"
              << " radioOversample=" << phy.radio_oversample
              << " radioRate=" << (phy.rate * static_cast<double>(phy.radio_oversample)) / 1e6 << " Msps\n"
              << "LDPC: n=" << code.n
              << " k=" << code.k
              << " blocks/frame=" << blocks
              << " infoBits/frame=" << info_bits
              << " codedBits/frame=" << phy.coded_bits_per_frame() << "\n"
              << "Nominal: frameRate=" << std::fixed << std::setprecision(2) << frame_rate
              << " fps, info=" << info_rate
              << " Mbps, coded=" << coded_rate << " Mbps\n";
}

static std::string radio_args_for_mode(const Options& opt, const std::string& mode)
{
    if (!opt.device_args.empty()) {
        return opt.device_args;
    }
    if (mode == "tx") {
        return "addr=" + opt.tx_addr;
    }
    return "addr=" + opt.rx_addr;
}

static usrp_link::RadioConfig radio_config_for_mode(const Options& opt, const std::string& mode)
{
    usrp_link::RadioConfig cfg;
    cfg.args = radio_args_for_mode(opt, mode);
    cfg.master_clock_rate = opt.master_clock_rate;
    cfg.sample_rate = opt.rate * static_cast<double>(opt.radio_oversample);
    cfg.center_freq = opt.freq;
    cfg.gain = (mode == "tx") ? opt.tx_gain : opt.rx_gain;
    cfg.clock_source = opt.clock_source;
    cfg.time_source = opt.time_source;
    cfg.antenna = (mode == "tx" && !opt.tx_antenna.empty()) ? opt.tx_antenna :
                  (mode == "rx" && !opt.rx_antenna.empty()) ? opt.rx_antenna :
                  opt.antenna;
    cfg.subdev = (mode == "tx" && !opt.tx_subdev.empty()) ? opt.tx_subdev :
                 (mode == "rx" && !opt.rx_subdev.empty()) ? opt.rx_subdev :
                 opt.subdev;
    cfg.tx_cpu_format = opt.tx_host_format;
    const int channel = (mode == "tx" && opt.tx_channel >= 0) ? opt.tx_channel :
                        (mode == "rx" && opt.rx_channel >= 0) ? opt.rx_channel :
                        opt.channel;
    cfg.channel = static_cast<size_t>(std::max(channel, 0));
    return cfg;
}

static usrp_link::DuplexRadioConfig duplex_radio_config(const Options& opt)
{
    usrp_link::DuplexRadioConfig cfg;
    cfg.args = radio_args_for_mode(opt, "tx");
    cfg.master_clock_rate = opt.master_clock_rate;
    cfg.sample_rate = opt.rate * static_cast<double>(opt.radio_oversample);
    cfg.center_freq = opt.freq;
    cfg.tx_gain = opt.tx_gain;
    cfg.rx_gain = opt.rx_gain;
    cfg.clock_source = opt.clock_source;
    cfg.time_source = opt.time_source;
    cfg.tx_antenna = !opt.tx_antenna.empty() ? opt.tx_antenna : opt.antenna;
    cfg.rx_antenna = !opt.rx_antenna.empty() ? opt.rx_antenna : opt.antenna;
    cfg.tx_subdev = !opt.tx_subdev.empty() ? opt.tx_subdev : opt.subdev;
    cfg.rx_subdev = !opt.rx_subdev.empty() ? opt.rx_subdev : opt.subdev;
    cfg.tx_cpu_format = opt.tx_host_format;
    cfg.tx_channel = static_cast<size_t>(std::max(opt.tx_channel >= 0 ? opt.tx_channel : opt.channel, 0));
    cfg.rx_channel = static_cast<size_t>(std::max(opt.rx_channel >= 0 ? opt.rx_channel : opt.channel, 0));
    return cfg;
}

static int16_t quantize_tx_sc16_component(float x)
{
    const float limited = std::max(-1.0f, std::min(1.0f, x));
    return static_cast<int16_t>(limited * 32767.0f);
}

static std::vector<usrp_link::ci16> quantize_tx_sc16(const std::vector<cf32>& frame)
{
    std::vector<usrp_link::ci16> packed(frame.size());
    std::transform(frame.begin(), frame.end(), packed.begin(), [](const cf32& sample) {
        return usrp_link::ci16{
            quantize_tx_sc16_component(sample.real()),
            quantize_tx_sc16_component(sample.imag())};
    });
    return packed;
}

static void send_tx_frame(
    usrp_link::RadioEndpoint& radio,
    const std::vector<cf32>& frame,
    const std::string& tx_host_format,
    bool& first_packet,
    const char* log_prefix)
{
    const size_t send_chunk = tx_host_format == "sc16"
        ? static_cast<size_t>(32768)
        : std::max<size_t>(radio.tx_max_samps(), 1024);
    size_t off = 0;
    if (tx_host_format == "sc16") {
        const auto packed = quantize_tx_sc16(frame);
        while (off < packed.size()) {
            const size_t count = std::min(send_chunk, packed.size() - off);
            size_t sent = 0;
            radio.send(packed.data() + off, count, first_packet, false, 1.0, sent);
            if (sent != count) {
                std::cerr << log_prefix << " short send: " << sent << "/" << count << "\n";
            }
            first_packet = false;
            off += sent;
            if (sent == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        return;
    }

    while (off < frame.size()) {
        const size_t count = std::min(send_chunk, frame.size() - off);
        size_t sent = 0;
        radio.send(frame.data() + off, count, first_packet, false, 1.0, sent);
        if (sent != count) {
            std::cerr << log_prefix << " short send: " << sent << "/" << count << "\n";
        }
        first_packet = false;
        off += sent;
        if (sent == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

static void send_tx_packed_sc16_frame(
    usrp_link::RadioEndpoint& radio,
    const std::vector<usrp_link::ci16>& packed,
    bool& first_packet,
    const char* log_prefix)
{
    constexpr size_t kSendChunk = 32768;
    size_t off = 0;
    while (off < packed.size()) {
        const size_t count = std::min(kSendChunk, packed.size() - off);
        size_t sent = 0;
        radio.send(packed.data() + off, count, first_packet, false, 1.0, sent);
        if (sent != count) {
            std::cerr << log_prefix << " short send: " << sent << "/" << count << "\n";
        }
        first_packet = false;
        off += sent;
        if (sent == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

class TxBasebandChain {
public:
    TxBasebandChain(
        const PhyConfig& phy,
        const LdpcCode& code,
        const InjectionChannel* precomp_channel = nullptr,
        const InjectionChannel* actual_channel = nullptr)
        : phy_(phy),
          code_(code),
          precomp_channel_(precomp_channel),
          actual_channel_(actual_channel)
    {}

    std::vector<cf32> build_test_frame(uint32_t frame_id) const
    {
        auto frame = build_tx_frame(phy_, code_, frame_id);
        apply_channel_chain(frame);
        return upsample_for_radio(frame, phy_.radio_oversample);
    }

    std::vector<cf32> build_media_frame(const std::vector<uint8_t>& info_bits) const
    {
        auto frame = build_tx_frame_from_info(phy_, code_, info_bits);
        apply_channel_chain(frame);
        return upsample_for_radio(frame, phy_.radio_oversample);
    }

private:
    void apply_channel_chain(std::vector<cf32>& frame) const
    {
        if (precomp_channel_ != nullptr && !precomp_channel_->empty()) {
            apply_ofdm_channel_transform(frame, phy_, *precomp_channel_, true);
        }
        if (actual_channel_ != nullptr && !actual_channel_->empty()) {
            apply_ofdm_channel_transform(frame, phy_, *actual_channel_, false);
        }
    }

    const PhyConfig& phy_;
    const LdpcCode& code_;
    const InjectionChannel* precomp_channel_ = nullptr;
    const InjectionChannel* actual_channel_ = nullptr;
};

class RxBasebandChain {
public:
    RxBasebandChain(
        const PhyConfig& phy,
        const LdpcCode& code,
        CudaBpDecoder* cuda_bp_decoder,
        CudaBpOsdDecoder* cuda_decoder,
        int ldpc_max_iter,
        double ldpc_normalization,
        bool reference_test)
        : phy_(phy),
          code_(code),
          cuda_bp_decoder_(cuda_bp_decoder),
          cuda_decoder_(cuda_decoder),
          ldpc_max_iter_(ldpc_max_iter),
          ldpc_normalization_(ldpc_normalization),
          reference_test_(reference_test)
    {}

    bool process(std::deque<cf32>& rxbuf, FrameDecodeResult& result) const
    {
        return process_rx_buffer(
            rxbuf,
            phy_,
            code_,
            cuda_bp_decoder_,
            cuda_decoder_,
            ldpc_max_iter_,
            ldpc_normalization_,
            reference_test_,
            result);
    }

private:
    const PhyConfig& phy_;
    const LdpcCode& code_;
    CudaBpDecoder* cuda_bp_decoder_ = nullptr;
    CudaBpOsdDecoder* cuda_decoder_ = nullptr;
    int ldpc_max_iter_ = 0;
    double ldpc_normalization_ = 0.0;
    bool reference_test_ = false;
};

#ifdef HAVE_GPU_FULL_PIPELINE
namespace gpu_pipe = usrp_gpu_pipeline_bridge;

static void check_cuda_runtime(cudaError_t status, const char* what);

static gpu_pipe::Config make_gpu_pipeline_config(
    const PhyConfig& phy,
    const LdpcCode& code,
    float demod_noise_var = 1.0e-3f)
{
    gpu_pipe::Config cfg;
    cfg.nfft = phy.nfft;
    cfg.cp = phy.cp;
    cfg.preamble_len = static_cast<int>(phy.preamble.size());
    cfg.pre_half_len = phy.pre_half_len;
    cfg.num_symbols = phy.num_symbols;
    cfg.active_sc = phy.active_sc;
    cfg.bits_per_symbol = phy.bits_per_symbol;
    cfg.code_n = code.n;
    cfg.code_k = code.k;
    cfg.blocks_per_frame = phy.coded_bits_per_frame() / code.n;
    cfg.max_rx_search_samples = std::max(phy.frame_len(), phy.max_buffered_frames * phy.frame_len());
    cfg.sample_rate = static_cast<float>(phy.rate);
    cfg.sync_threshold = static_cast<float>(phy.sync_threshold);
    cfg.tx_amplitude = static_cast<float>(phy.amplitude);
    cfg.demod_noise_var = std::max(demod_noise_var, 1.0e-8f);
    cfg.use_pilot_channel_est = (phy.rx_channel_estimation != "none");
    return cfg;
}

static std::vector<uint8_t> build_dense_generator_kn_for_gpu(const LdpcCode& code)
{
    const uint64_t entries = static_cast<uint64_t>(std::max(code.k, 0)) * static_cast<uint64_t>(std::max(code.n, 0));
    constexpr uint64_t kDenseGeneratorLimit = 512ull * 1024ull * 1024ull;
    if (entries == 0 || entries > kDenseGeneratorLimit) {
        throw std::runtime_error(
            "GPU TX bridge currently uses a dense generator matrix; this code is too large. "
            "Use compact DVB-S2/QC encoder kernels for long LDPC codes.");
    }

    std::vector<uint8_t> generator(static_cast<size_t>(entries), 0);
    std::vector<uint8_t> unit(static_cast<size_t>(code.k), 0);
    for (int row = 0; row < code.k; ++row) {
        std::fill(unit.begin(), unit.end(), 0);
        unit[static_cast<size_t>(row)] = 1;
        const auto cw = ldpc_encode(code, unit.data());
        std::copy(cw.begin(), cw.end(), generator.begin() + static_cast<std::ptrdiff_t>(row * code.n));
    }
    return generator;
}

static gpu_pipe::StaticTables build_gpu_static_tables(
    const PhyConfig& phy,
    const LdpcCode& code,
    const InjectionChannel* precomp_channel,
    bool include_tx_generator,
    const InjectionChannel* actual_channel = nullptr)
{
    gpu_pipe::StaticTables tables;
    if (include_tx_generator) {
        if (code.encoder_kind == LdpcCode::EncoderKind::DvbS2Accumulator) {
            tables.dvb_s2_accumulator = true;
            tables.row_info_offsets.reserve(static_cast<size_t>(code.m + 1));
            tables.row_info_offsets.push_back(0);
            for (int r = 0; r < code.m; ++r) {
                for (int c : code.row_cols[static_cast<size_t>(r)]) {
                    if (c < code.k) {
                        tables.row_info_cols.push_back(c);
                    }
                }
                tables.row_info_offsets.push_back(static_cast<int>(tables.row_info_cols.size()));
            }
        } else {
            tables.generator_kn = build_dense_generator_kn_for_gpu(code);
        }
    }
    tables.used_indices = phy.used_indices;
    tables.pilot_symbols = phy.pilot_symbols;
    tables.data_symbols = phy.data_symbols;
    tables.pilot_symbol_index.assign(static_cast<size_t>(phy.num_symbols), -1);
    tables.data_symbol_index.assign(static_cast<size_t>(phy.num_symbols), -1);

    for (size_t i = 0; i < phy.pilot_symbols.size(); ++i) {
        tables.pilot_symbol_index[static_cast<size_t>(phy.pilot_symbols[i])] = static_cast<int>(i);
    }
    for (size_t i = 0; i < phy.data_symbols.size(); ++i) {
        tables.data_symbol_index[static_cast<size_t>(phy.data_symbols[i])] = static_cast<int>(i);
    }

    tables.interp_p0.reserve(phy.data_symbols.size());
    tables.interp_p1.reserve(phy.data_symbols.size());
    tables.interp_alpha.reserve(phy.data_symbols.size());
    for (int sym : phy.data_symbols) {
        size_t hi = 0;
        while (hi + 1 < phy.pilot_symbols.size() && sym > phy.pilot_symbols[hi + 1]) {
            ++hi;
        }

        size_t p0 = hi;
        size_t p1 = std::min(hi + 1, phy.pilot_symbols.size() - 1);
        if (sym <= phy.pilot_symbols.front()) {
            p0 = p1 = 0;
        } else if (sym >= phy.pilot_symbols.back()) {
            p0 = p1 = phy.pilot_symbols.size() - 1;
        }

        float alpha = 0.0f;
        if (p0 != p1) {
            alpha = static_cast<float>(
                static_cast<double>(sym - phy.pilot_symbols[p0]) /
                static_cast<double>(phy.pilot_symbols[p1] - phy.pilot_symbols[p0]));
        }
        tables.interp_p0.push_back(static_cast<int>(p0));
        tables.interp_p1.push_back(static_cast<int>(p1));
        tables.interp_alpha.push_back(alpha);
    }

    tables.preamble.assign(phy.preamble.begin(), phy.preamble.end());
    tables.pilot_grid.reserve(phy.pilot_symbols.size() * static_cast<size_t>(phy.active_sc));
    for (const auto& row : phy.pilot_grid) {
        tables.pilot_grid.insert(tables.pilot_grid.end(), row.begin(), row.end());
    }

    if (precomp_channel != nullptr && !precomp_channel->empty()) {
        tables.precomp_h.reserve(static_cast<size_t>(phy.num_symbols * phy.active_sc));
        for (int sym = 0; sym < phy.num_symbols; ++sym) {
            for (int sc = 0; sc < phy.active_sc; ++sc) {
                tables.precomp_h.push_back(ofdm_channel_response(*precomp_channel, phy, sym, sc));
            }
        }
    }
    if (actual_channel != nullptr && !actual_channel->empty()) {
        tables.actual_h.reserve(static_cast<size_t>(phy.num_symbols * phy.active_sc));
        for (int sym = 0; sym < phy.num_symbols; ++sym) {
            for (int sc = 0; sc < phy.active_sc; ++sc) {
                tables.actual_h.push_back(ofdm_channel_response(*actual_channel, phy, sym, sc));
            }
        }
    }

    return tables;
}

static float gpu_sim_frequency_noise_var(
    const PhyConfig& phy,
    const InjectionChannel* precomp_channel,
    const InjectionChannel* actual_channel,
    double snr_db)
{
    if (!std::isfinite(snr_db)) {
        return 0.0f;
    }
    double power = 0.0;
    uint64_t count = 0;
    for (int sym : phy.data_symbols) {
        for (int sc = 0; sc < phy.active_sc; ++sc) {
            cf32 h(1.0f, 0.0f);
            if (precomp_channel != nullptr && !precomp_channel->empty()) {
                const cf32 p = ofdm_channel_response(*precomp_channel, phy, sym, sc);
                const float p2 = std::norm(p);
                if (p2 > 1.0e-10f) {
                    h *= std::conj(p) / p2;
                }
            }
            if (actual_channel != nullptr && !actual_channel->empty()) {
                h *= ofdm_channel_response(*actual_channel, phy, sym, sc);
            }
            power += std::norm(h);
            ++count;
        }
    }
    const double avg_symbol_power = count > 0 ? power / static_cast<double>(count) : 1.0;
    const double noise_var = std::max(avg_symbol_power, 1.0e-12) / std::pow(10.0, snr_db / 10.0);
    return static_cast<float>(noise_var);
}

static FrameDecodeResult finish_decode_from_llr(
    const std::vector<float>& llr,
    const PhyConfig& phy,
    const LdpcCode& code,
    CudaBpDecoder* cuda_bp_decoder,
    CudaBpOsdDecoder* cuda_decoder,
    int ldpc_max_iter,
    double ldpc_normalization,
    bool reference_test,
    double peak,
    double cfo_hz,
    double snr_db,
    uint64_t demod_ns,
    DecodeProfile* profile)
{
    const auto t_fec0 = std::chrono::steady_clock::now();
    std::vector<uint8_t> coded_hard(llr.size(), 0);
    for (size_t i = 0; i < llr.size(); ++i) {
        coded_hard[i] = (llr[i] < 0.0f) ? 1 : 0;
    }

    std::vector<uint8_t> info_bits;
    info_bits.reserve(static_cast<size_t>((phy.coded_bits_per_frame() / code.n) * code.k));

    int iter_sum = 0;
    bool parity_ok = true;
    const int blocks = phy.coded_bits_per_frame() / code.n;
    if (cuda_bp_decoder != nullptr) {
        int success_count = 0;
        double total_decode_ms = 0.0;
        info_bits = cuda_bp_decoder->decode(llr.data(), blocks, success_count, total_decode_ms);
        parity_ok = (success_count == blocks);
        iter_sum = success_count;
    } else if (cuda_decoder != nullptr) {
        int used_osd_count = 0;
        info_bits = cuda_decoder->decode(llr.data(), blocks, used_osd_count);
        iter_sum = used_osd_count;
    } else {
        for (int b = 0; b < blocks; ++b) {
            auto dr = ldpc_decode_min_sum(
                code,
                llr.data() + static_cast<size_t>(b * code.n),
                ldpc_max_iter,
                ldpc_normalization);
            parity_ok = parity_ok && dr.parity_ok;
            iter_sum += dr.iterations;
            info_bits.insert(info_bits.end(), dr.info_bits.begin(), dr.info_bits.end());
        }
    }
    const auto t_fec1 = std::chrono::steady_clock::now();

    FrameDecodeResult out;
    out.detected = true;
    out.parity_ok = parity_ok;
    out.iterations = blocks > 0 ? static_cast<int>(std::lround(static_cast<double>(iter_sum) / blocks)) : 0;
    out.frame_id = reference_test ? bits_to_u32_msb(info_bits, 0) : 0;
    out.peak = peak;
    out.cfo_hz = cfo_hz;
    out.snr_db = snr_db;

    if (reference_test) {
        const auto ref_info = build_info_bits(phy, code, out.frame_id);
        int bit_errors = 0;
        for (size_t i = 0; i < std::min(info_bits.size(), ref_info.size()); ++i) {
            bit_errors += (info_bits[i] != ref_info[i]) ? 1 : 0;
        }
        out.bit_errors = bit_errors;

        const auto ref_coded = encode_frame_bits(phy, code, ref_info);
        int pre_errors = 0;
        for (size_t i = 0; i < std::min(coded_hard.size(), ref_coded.size()); ++i) {
            pre_errors += (coded_hard[i] != ref_coded[i]) ? 1 : 0;
        }
        out.pre_fec_bit_errors = pre_errors;
        out.frame_ok = out.parity_ok && out.bit_errors == 0;
    } else {
        out.bit_errors = -1;
        out.pre_fec_bit_errors = -1;
        out.frame_ok = false;
    }
    out.info_bits = std::move(info_bits);
    const auto t_ref1 = std::chrono::steady_clock::now();

    if (profile != nullptr) {
        ++profile->frames;
        profile->llr_ns += demod_ns;
        profile->fec_ns += elapsed_ns(t_fec0, t_fec1);
        profile->reference_ns += elapsed_ns(t_fec1, t_ref1);
    }
    return out;
}

static FrameDecodeResult finish_decode_from_info_bits(
    std::vector<uint8_t> info_bits,
    const PhyConfig& phy,
    const LdpcCode& code,
    int iter_sum,
    bool parity_ok,
    bool reference_test,
    double peak,
    double cfo_hz,
    double snr_db,
    uint64_t demod_ns,
    uint64_t fec_ns,
    DecodeProfile* profile)
{
    const auto t_ref0 = std::chrono::steady_clock::now();
    const int blocks = phy.coded_bits_per_frame() / code.n;

    FrameDecodeResult out;
    out.detected = true;
    out.parity_ok = parity_ok;
    out.iterations = blocks > 0 ? static_cast<int>(std::lround(static_cast<double>(iter_sum) / blocks)) : 0;
    out.frame_id = reference_test ? bits_to_u32_msb(info_bits, 0) : 0;
    out.peak = peak;
    out.cfo_hz = cfo_hz;
    out.snr_db = snr_db;

    if (reference_test) {
        const auto ref_info = build_info_bits(phy, code, out.frame_id);
        int bit_errors = 0;
        for (size_t i = 0; i < std::min(info_bits.size(), ref_info.size()); ++i) {
            bit_errors += (info_bits[i] != ref_info[i]) ? 1 : 0;
        }
        out.bit_errors = bit_errors;
        out.pre_fec_bit_errors = -1;
        out.frame_ok = out.parity_ok && out.bit_errors == 0;
    } else {
        out.bit_errors = -1;
        out.pre_fec_bit_errors = -1;
        out.frame_ok = false;
    }

    out.info_bits = std::move(info_bits);
    const auto t_ref1 = std::chrono::steady_clock::now();
    if (profile != nullptr) {
        ++profile->frames;
        profile->llr_ns += demod_ns;
        profile->fec_ns += fec_ns;
        profile->reference_ns += elapsed_ns(t_ref0, t_ref1);
    }
    return out;
}

class GpuTxBasebandChain {
public:
    GpuTxBasebandChain(
        const PhyConfig& phy,
        const LdpcCode& code,
        const InjectionChannel* precomp_channel,
        const InjectionChannel* actual_channel = nullptr)
        : phy_(phy),
          code_(code),
          actual_channel_(actual_channel),
          pipeline_(make_gpu_pipeline_config(phy, code), 3)
    {
        pipeline_.upload_static_tables(build_gpu_static_tables(phy_, code_, precomp_channel, true, actual_channel_));
        std::cout << "[GPU-PIPE] TX baseband enabled: encode+map+precomp+OFDM on CUDA";
        if (actual_channel_ != nullptr && !actual_channel_->empty()) {
            std::cout << " + CUDA actual-channel injection";
        }
        std::cout << "\n";
    }

    std::vector<cf32> build_test_frame(uint32_t frame_id)
    {
        const auto info = build_info_bits(phy_, code_, frame_id);
        return build_media_frame(info);
    }

    std::vector<cf32> build_media_frame(const std::vector<uint8_t>& info_bits)
    {
        const int slot = next_slot_++ % 3;
        std::vector<cf32> frame(static_cast<size_t>(phy_.frame_len()));
        pipeline_.submit_tx_frame_async(slot, info_bits.data(), frame.data());
        pipeline_.synchronize_tx(slot);
        return upsample_for_radio(frame, phy_.radio_oversample);
    }

    struct DeviceFrame {
        const cf32* samples = nullptr;
        int slot = 0;
        size_t sample_count = 0;
    };

    DeviceFrame build_test_frame_device(uint32_t frame_id, float frequency_noise_var, uint64_t noise_seed)
    {
        const auto info = build_info_bits(phy_, code_, frame_id);
        return build_media_frame_device(info, frequency_noise_var, noise_seed);
    }

    DeviceFrame build_media_frame_device(
        const std::vector<uint8_t>& info_bits,
        float frequency_noise_var,
        uint64_t noise_seed)
    {
        const int slot = next_slot_++ % 3;
        const bool add_noise = frequency_noise_var > 0.0f;
        const cf32* d_frame = pipeline_.submit_tx_frame_device_async(
            slot,
            info_bits.data(),
            frequency_noise_var,
            noise_seed,
            add_noise);
        pipeline_.synchronize_tx(slot);
        return DeviceFrame{d_frame, slot, static_cast<size_t>(phy_.frame_len())};
    }

private:
    const PhyConfig& phy_;
    const LdpcCode& code_;
    const InjectionChannel* actual_channel_ = nullptr;
    gpu_pipe::Pipeline pipeline_;
    int next_slot_ = 0;
};

class GpuRxSyncExtractor {
public:
    GpuRxSyncExtractor(const PhyConfig& phy, const LdpcCode& code)
        : phy_(phy),
          pipeline_(make_gpu_pipeline_config(phy, code), 2)
    {
        pipeline_.upload_static_tables(build_gpu_static_tables(phy_, code, nullptr, false));
        std::cout << "[GPU-PIPE] RX sync enabled: Schmidl-Cox acquisition on CUDA\n";
    }

    bool extract(std::vector<cf32>& rxbuf, size_t& base, ExtractedFrame& frame, SyncState& sync_state, SyncStats& sync_stats)
    {
        if (rxbuf.size() < base || rxbuf.size() - base < static_cast<size_t>(phy_.preamble.size())) {
            return false;
        }

        const size_t available = rxbuf.size() - base;
        const int max_samples = std::max(phy_.frame_len(), phy_.max_buffered_frames * phy_.frame_len());
        const size_t search_samples =
            (sync_state == SyncState::Tracking)
                ? static_cast<size_t>(std::max(256, phy_.sync_tracking_window))
                : static_cast<size_t>(max_samples);
        const size_t sample_limit =
            (sync_state == SyncState::Tracking)
                ? static_cast<size_t>(phy_.preamble.size()) + search_samples
                : search_samples;
        const int sample_count = static_cast<int>(std::min<size_t>(available, sample_limit));
        const int slot = next_slot_++ % 2;
        const bool tracking_attempt = (sync_state == SyncState::Tracking);
        const auto result = pipeline_.find_frame(slot, rxbuf.data() + static_cast<std::ptrdiff_t>(base), sample_count);
        const size_t strict_window = static_cast<size_t>(std::max(256, 4 * phy_.cp));
        const bool reject_late_tracking =
            tracking_attempt &&
            phy_.sync_late_peak_threshold > 0.0 &&
            result.start > static_cast<int>(strict_window) &&
            result.metric < static_cast<float>(phy_.sync_late_peak_threshold);
        if (result.metric < static_cast<float>(phy_.sync_threshold) || reject_late_tracking) {
            if (sync_state == SyncState::Tracking) {
                ++sync_stats.tracking_miss;
                sync_state = SyncState::Acquisition;
            } else {
                ++sync_stats.miss;
            }
            const size_t keep = static_cast<size_t>(phy_.max_buffered_frames * phy_.frame_len());
            if (rxbuf.size() >= base && rxbuf.size() - base > keep) {
                base = rxbuf.size() - static_cast<size_t>(phy_.frame_len());
                compact_rx_vector(rxbuf, base);
            }
            return false;
        }

        if (static_cast<size_t>(result.start) + static_cast<size_t>(phy_.frame_len()) > available) {
            ++sync_stats.incomplete;
            return false;
        }

        const auto frame_begin = rxbuf.begin() + static_cast<std::ptrdiff_t>(base + static_cast<size_t>(result.start));
        const auto frame_end = frame_begin + static_cast<std::ptrdiff_t>(phy_.frame_len());
        frame.samples.assign(frame_begin, frame_end);
        frame.peak = result.metric;
        frame.cfo_hz = result.cfo_hz;
        ++sync_stats.found;
        sync_stats.skipped_samples += static_cast<uint64_t>(std::max(result.start, 0));
        sync_stats.max_start = std::max<uint64_t>(sync_stats.max_start, static_cast<uint64_t>(std::max(result.start, 0)));
        sync_state = SyncState::Tracking;
        base += static_cast<size_t>(result.start) + static_cast<size_t>(phy_.frame_len());
        compact_rx_vector(rxbuf, base);
        return true;
    }

    bool extract_device(
        const cf32* d_samples,
        size_t sample_count,
        size_t& base,
        ExtractedDeviceFrame& frame,
        DeviceFramePool& frame_pool,
        SyncState& sync_state,
        SyncStats& sync_stats)
    {
        if (d_samples == nullptr || sample_count < base ||
            sample_count - base < static_cast<size_t>(phy_.preamble.size())) {
            return false;
        }

        const size_t available = sample_count - base;
        const int max_samples = std::max(phy_.frame_len(), phy_.max_buffered_frames * phy_.frame_len());
        const size_t search_samples =
            (sync_state == SyncState::Tracking)
                ? static_cast<size_t>(std::max(256, phy_.sync_tracking_window))
                : static_cast<size_t>(max_samples);
        const size_t sample_limit =
            (sync_state == SyncState::Tracking)
                ? static_cast<size_t>(phy_.preamble.size()) + search_samples
                : search_samples;
        const int search_count = static_cast<int>(std::min<size_t>(available, sample_limit));
        const int slot = next_slot_++ % 2;
        const bool tracking_attempt = (sync_state == SyncState::Tracking);
        const auto result = pipeline_.find_frame_device(
            slot,
            d_samples + static_cast<std::ptrdiff_t>(base),
            search_count);
        const size_t strict_window = static_cast<size_t>(std::max(256, 4 * phy_.cp));
        const bool reject_late_tracking =
            tracking_attempt &&
            phy_.sync_late_peak_threshold > 0.0 &&
            result.start > static_cast<int>(strict_window) &&
            result.metric < static_cast<float>(phy_.sync_late_peak_threshold);
        if (result.metric < static_cast<float>(phy_.sync_threshold) || reject_late_tracking) {
            if (sync_state == SyncState::Tracking) {
                ++sync_stats.tracking_miss;
                sync_state = SyncState::Acquisition;
            } else {
                ++sync_stats.miss;
            }
            return false;
        }

        if (static_cast<size_t>(result.start) + static_cast<size_t>(phy_.frame_len()) > available) {
            ++sync_stats.incomplete;
            return false;
        }

        frame.release();
        auto lease = frame_pool.try_acquire();
        if (!lease) {
            base += static_cast<size_t>(result.start) + static_cast<size_t>(phy_.frame_len());
            ++sync_stats.miss;
            return false;
        }
        frame.sample_count = static_cast<size_t>(phy_.frame_len());
        frame.device_samples = lease->ptr;
        frame.lease = std::move(lease);
        check_cuda_runtime(
            cudaMemcpy(
                frame.device_samples,
                d_samples + static_cast<std::ptrdiff_t>(base + static_cast<size_t>(result.start)),
                frame.sample_count * sizeof(cf32),
                cudaMemcpyDeviceToDevice),
            "copy detected RX frame to device queue frame");
        frame.peak = result.metric;
        frame.cfo_hz = result.cfo_hz;
        ++sync_stats.found;
        sync_stats.skipped_samples += static_cast<uint64_t>(std::max(result.start, 0));
        sync_stats.max_start = std::max<uint64_t>(sync_stats.max_start, static_cast<uint64_t>(std::max(result.start, 0)));
        sync_state = SyncState::Tracking;
        base += static_cast<size_t>(result.start) + static_cast<size_t>(phy_.frame_len());
        return true;
    }

private:
    const PhyConfig& phy_;
    gpu_pipe::Pipeline pipeline_;
    int next_slot_ = 0;
};

class GpuRxDemodChain {
public:
    struct DeviceLlrFrame {
        const float* device_llrs = nullptr;
        int blocks = 0;
        double peak = 0.0;
        double cfo_hz = 0.0;
        uint64_t demod_ns = 0;
    };

    GpuRxDemodChain(
        const PhyConfig& phy,
        const LdpcCode& code,
        const InjectionChannel* precomp_channel,
        float demod_noise_var = 1.0e-3f)
        : phy_(phy),
          code_(code),
          pipeline_(make_gpu_pipeline_config(phy, code, demod_noise_var), 3)
    {
        pipeline_.upload_static_tables(build_gpu_static_tables(phy_, code_, precomp_channel, false));
        std::cout << "[GPU-PIPE] RX demod enabled: GPU sync+CFO+FFT+channel+LLR, device FEC bridge when available"
                  << " demodNoiseVar=" << std::max(demod_noise_var, 1.0e-8f) << "\n";
    }

    FrameDecodeResult process(
        const ExtractedFrame& frame,
        CudaBpDecoder* cuda_bp_decoder,
        CudaBpOsdDecoder* cuda_decoder,
        int ldpc_max_iter,
        double ldpc_normalization,
        bool reference_test,
        DecodeProfile* profile)
    {
        submit_ui_spectrum_iq_if_due(frame.samples);
        submit_ui_constellation_from_host_frame_if_due(frame.samples, phy_, frame.cfo_hz);

        const int slot = next_slot_++ % 3;
        if (cuda_bp_decoder != nullptr ||
            (cuda_decoder != nullptr && !cuda_decoder->osd_only)) {
            const auto t0 = std::chrono::steady_clock::now();
            gpu_pipe::SyncResult sync;
            const float* device_llr = pipeline_.demod_frame_to_device_llr(slot, frame.samples.data(), &sync);
            const auto t1 = std::chrono::steady_clock::now();

            const int blocks = phy_.coded_bits_per_frame() / code_.n;
            const auto t_fec0 = std::chrono::steady_clock::now();
            int iter_sum = 0;
            bool parity_ok = true;
            std::vector<uint8_t> info_bits;
            if (cuda_bp_decoder != nullptr) {
                int success_count = 0;
                double total_decode_ms = 0.0;
                info_bits = cuda_bp_decoder->decode_device(device_llr, blocks, success_count, total_decode_ms);
                iter_sum = success_count;
                parity_ok = (success_count == blocks);
            } else {
                int used_osd_count = 0;
                info_bits = cuda_decoder->decode_device(device_llr, blocks, used_osd_count);
                iter_sum = used_osd_count;
            }
            const auto t_fec1 = std::chrono::steady_clock::now();

            return finish_decode_from_info_bits(
                std::move(info_bits),
                phy_,
                code_,
                iter_sum,
                parity_ok,
                reference_test,
                sync.metric,
                sync.cfo_hz,
                0.0,
                elapsed_ns(t0, t1),
                elapsed_ns(t_fec0, t_fec1),
                profile);
        }

        std::vector<float> llr(static_cast<size_t>(phy_.coded_bits_per_frame()), 0.0f);
        const auto t0 = std::chrono::steady_clock::now();
        const auto sync = pipeline_.demod_frame_to_host_llr(slot, frame.samples.data(), llr.data());
        const auto t1 = std::chrono::steady_clock::now();
        const uint64_t demod_ns = elapsed_ns(t0, t1);
        return finish_decode_from_llr(
            llr,
            phy_,
            code_,
            cuda_bp_decoder,
            cuda_decoder,
            ldpc_max_iter,
            ldpc_normalization,
            reference_test,
            sync.metric,
            sync.cfo_hz,
            0.0,
            demod_ns,
            profile);
    }

    DeviceLlrFrame demod_device_to_device_llr(const ExtractedDeviceFrame& frame)
    {
        if (frame.device_samples == nullptr || frame.sample_count < static_cast<size_t>(phy_.frame_len())) {
            throw std::runtime_error("device-frame RX path received an empty frame");
        }

        const bool capture_spectrum =
            g_ui_telemetry != nullptr &&
            g_ui_telemetry->try_begin_spectrum_capture();
        const bool capture_constellation =
            g_ui_telemetry != nullptr &&
            g_ui_telemetry->try_begin_constellation_capture();
        if (capture_spectrum || capture_constellation) {
            std::vector<cf32> host_samples(static_cast<size_t>(phy_.frame_len()));
            check_cuda_runtime(
                cudaMemcpy(
                    host_samples.data(),
                    frame.device_samples,
                    host_samples.size() * sizeof(cf32),
                    cudaMemcpyDeviceToHost),
                "copy device RX frame for UI telemetry");
            if (capture_spectrum) {
                g_ui_telemetry->submit_spectrum_iq(host_samples);
            }
            if (capture_constellation) {
                submit_ui_constellation_from_host_frame(host_samples, phy_, frame.cfo_hz);
            }
        }

        const int slot = next_slot_++ % 3;
        const auto t0 = std::chrono::steady_clock::now();
        gpu_pipe::SyncResult sync;
        sync.start = 0;
        sync.metric = static_cast<float>(frame.peak);
        sync.cfo_hz = static_cast<float>(frame.cfo_hz);
        const float* device_llr = pipeline_.demod_frame_device_to_device_llr(slot, frame.device_samples, sync);
        const auto t1 = std::chrono::steady_clock::now();

        DeviceLlrFrame out;
        out.device_llrs = device_llr;
        out.blocks = phy_.coded_bits_per_frame() / code_.n;
        out.peak = frame.peak;
        out.cfo_hz = frame.cfo_hz;
        out.demod_ns = elapsed_ns(t0, t1);
        return out;
    }

    FrameDecodeResult process_device(
        const ExtractedDeviceFrame& frame,
        CudaBpDecoder* cuda_bp_decoder,
        CudaBpOsdDecoder* cuda_decoder,
        int,
        double,
        bool reference_test,
        DecodeProfile* profile)
    {
        if (cuda_bp_decoder == nullptr && (cuda_decoder == nullptr || cuda_decoder->osd_only)) {
            throw std::runtime_error("device-frame RX path requires a CUDA BP device decoder");
        }
        const DeviceLlrFrame llr_frame = demod_device_to_device_llr(frame);

        const auto t_fec0 = std::chrono::steady_clock::now();
        int iter_sum = 0;
        bool parity_ok = true;
        std::vector<uint8_t> info_bits;
        if (cuda_bp_decoder != nullptr) {
            int success_count = 0;
            double total_decode_ms = 0.0;
            info_bits = cuda_bp_decoder->decode_device(llr_frame.device_llrs, llr_frame.blocks, success_count, total_decode_ms);
            iter_sum = success_count;
            parity_ok = (success_count == llr_frame.blocks);
        } else {
            int used_osd_count = 0;
            info_bits = cuda_decoder->decode_device(llr_frame.device_llrs, llr_frame.blocks, used_osd_count);
            iter_sum = used_osd_count;
        }
        const auto t_fec1 = std::chrono::steady_clock::now();

        return finish_decode_from_info_bits(
            std::move(info_bits),
            phy_,
            code_,
            iter_sum,
            parity_ok,
            reference_test,
            llr_frame.peak,
            llr_frame.cfo_hz,
            0.0,
            llr_frame.demod_ns,
            elapsed_ns(t_fec0, t_fec1),
            profile);
    }

private:
    const PhyConfig& phy_;
    const LdpcCode& code_;
    gpu_pipe::Pipeline pipeline_;
    int next_slot_ = 0;
};
#endif

struct LinkRuntimeProfile {
    usrp_link::McsId mcs = usrp_link::McsId::MCS1_QpskRepeat3;
    Options opt;
    PhyConfig phy;
    LdpcCode code;
    fec::RxDecoderPlan rx_plan;
    size_t info_bytes = 0;
    size_t media_payload_bytes = 0;
    std::unique_ptr<TxBasebandChain> tx_baseband;
#ifdef HAVE_GPU_FULL_PIPELINE
    std::unique_ptr<GpuTxBasebandChain> gpu_tx_baseband;
    std::unique_ptr<GpuRxDemodChain> gpu_rx_demod;
#endif
    std::unique_ptr<CudaBpDecoder> cuda_bp_decoder;
    std::unique_ptr<CudaBpOsdDecoder> cuda_decoder;

    CudaBpDecoder* bp_decoder() const { return cuda_bp_decoder.get(); }
    CudaBpOsdDecoder* bp_osd_decoder() const { return cuda_decoder.get(); }
};

class LinkRuntimeProfileBank {
public:
    LinkRuntimeProfileBank(const Options& base_opt,
                           const PhyConfig& base_phy,
                           const LdpcCode& base_code,
                           const fec::RxDecoderPlan& base_rx_plan,
                           const InjectionChannel* precomp_channel,
                           const InjectionChannel* actual_channel)
    {
        add_profile(
            usrp_link::McsId::MCS0_ReliabilityBpskRepeat3,
            "bpsk",
            base_opt,
            base_phy,
            base_code,
            base_rx_plan,
            precomp_channel,
            actual_channel);
        add_profile(
            usrp_link::McsId::MCS1_QpskRepeat3,
            "qpsk",
            base_opt,
            base_phy,
            base_code,
            base_rx_plan,
            precomp_channel,
            actual_channel);
        add_profile(
            usrp_link::McsId::MCS2_QpskRepeat2,
            "qpsk",
            base_opt,
            base_phy,
            base_code,
            base_rx_plan,
            precomp_channel,
            actual_channel);
        add_profile(
            usrp_link::McsId::MCS3_QpskRepeat1,
            "qpsk",
            base_opt,
            base_phy,
            base_code,
            base_rx_plan,
            precomp_channel,
            actual_channel);
        add_profile(
            usrp_link::McsId::MCS4_16qamRepeat2,
            "16qam",
            base_opt,
            base_phy,
            base_code,
            base_rx_plan,
            precomp_channel,
            actual_channel);
        add_profile(
            usrp_link::McsId::MCS5_16qamRepeat1,
            "16qam",
            base_opt,
            base_phy,
            base_code,
            base_rx_plan,
            precomp_channel,
            actual_channel);

        common_payload_bytes_ = std::numeric_limits<size_t>::max();
        for (const auto& p : profiles_) {
            common_payload_bytes_ = std::min(common_payload_bytes_, p->media_payload_bytes);
        }
        if (common_payload_bytes_ == std::numeric_limits<size_t>::max()) {
            common_payload_bytes_ = 0;
        }
    }

    LinkRuntimeProfile& get(usrp_link::McsId mcs)
    {
        return *profiles_.at(static_cast<size_t>(std::max(0, std::min(5, static_cast<int>(mcs)))));
    }

    const LinkRuntimeProfile& get(usrp_link::McsId mcs) const
    {
        return *profiles_.at(static_cast<size_t>(std::max(0, std::min(5, static_cast<int>(mcs)))));
    }

    size_t common_payload_bytes() const { return common_payload_bytes_; }

private:
    void add_profile(usrp_link::McsId mcs,
                     const std::string& modulation,
                     const Options& base_opt,
                     const PhyConfig&,
                     const LdpcCode& base_code,
                     const fec::RxDecoderPlan& base_rx_plan,
                     const InjectionChannel* precomp_channel,
                     const InjectionChannel* actual_channel)
    {
        auto p = std::make_unique<LinkRuntimeProfile>();
        p->mcs = mcs;
        p->opt = base_opt;
        p->opt.modulation = modulation;
        p->phy = build_phy_config(p->opt);
        if (p->phy.frame_len() != build_phy_config(base_opt).frame_len()) {
            throw std::runtime_error("runtime profile bank requires identical frame sample length across MCS profiles");
        }
        if (p->phy.coded_bits_per_frame() % base_code.n != 0) {
            std::ostringstream oss;
            oss << "runtime profile bank profile " << modulation
                << " has coded bits/frame=" << p->phy.coded_bits_per_frame()
                << ", which is not divisible by LDPC n=" << base_code.n;
            throw std::runtime_error(oss.str());
        }
        p->code = base_code;
        p->rx_plan = base_rx_plan;
        p->info_bytes = link_info_bytes_per_frame(p->phy, p->code);
        p->media_payload_bytes = media_payload_capacity_bytes(p->phy, p->code);
        p->tx_baseband = std::make_unique<TxBasebandChain>(
            p->phy,
            p->code,
            precomp_channel,
            actual_channel);
#ifdef HAVE_GPU_FULL_PIPELINE
        if (base_opt.gpu_tx_baseband) {
            p->gpu_tx_baseband = std::make_unique<GpuTxBasebandChain>(
                p->phy,
                p->code,
                precomp_channel,
                actual_channel);
        }
        if (base_opt.gpu_rx_demod) {
            p->gpu_rx_demod = std::make_unique<GpuRxDemodChain>(
                p->phy,
                p->code,
                precomp_channel);
        }
#endif
        if (p->rx_plan.kind == fec::RxDecoderKind::GpuBp) {
            p->cuda_bp_decoder = std::make_unique<CudaBpDecoder>(p->opt, p->code, *p->rx_plan.fec);
        } else if (p->rx_plan.kind == fec::RxDecoderKind::GpuBpOsd) {
            p->cuda_decoder = std::make_unique<CudaBpOsdDecoder>(p->opt, p->code);
        }
        profiles_.at(static_cast<size_t>(mcs)) = std::move(p);
    }

    std::array<std::unique_ptr<LinkRuntimeProfile>, 6> profiles_{};
    size_t common_payload_bytes_ = 0;
};

struct RxSampleBlock {
    std::vector<cf32> samples;
    int status = UHD_C_RX_ERROR;
};

static void print_bench_decode_substage(
    const std::string& name,
    uint64_t ns,
    uint64_t frames,
    int info_bits_per_frame);

class RxSampleQueue {
public:
    explicit RxSampleQueue(size_t capacity_blocks)
        : capacity_blocks_(std::max<size_t>(capacity_blocks, 4))
    {}

    void push(RxSampleBlock block)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return;
        }
        if (queue_.size() >= capacity_blocks_) {
            queue_.pop_front();
            ++dropped_blocks_;
        }
        queue_.push_back(std::move(block));
        cv_.notify_one();
    }

    bool pop_for(RxSampleBlock& block, std::chrono::milliseconds timeout)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock, timeout, [this] {
            return closed_ || !queue_.empty();
        });
        if (queue_.empty()) {
            return false;
        }
        block = std::move(queue_.front());
        queue_.pop_front();
        return true;
    }

    void close()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        cv_.notify_all();
    }

    uint64_t dropped_blocks() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_blocks_;
    }

    size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<RxSampleBlock> queue_;
    size_t capacity_blocks_ = 0;
    uint64_t dropped_blocks_ = 0;
    bool closed_ = false;
};

class OnlineIqBuffer {
public:
    explicit OnlineIqBuffer(size_t capacity_samples)
        : storage_(std::max<size_t>(capacity_samples, 4096))
    {}

    void push(const cf32* samples, size_t count)
    {
        if (samples == nullptr || count == 0) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return;
        }
        const size_t capacity = storage_.size();
        if (count > capacity) {
            const size_t skip = count - capacity;
            samples += skip;
            count = capacity;
            dropped_samples_ += static_cast<uint64_t>(skip);
        }

        if (size_ + count > capacity) {
            const size_t drop = size_ + count - capacity;
            read_pos_ = (read_pos_ + drop) % capacity;
            size_ -= drop;
            dropped_samples_ += static_cast<uint64_t>(drop);
        }

        size_t write_pos = (read_pos_ + size_) % capacity;
        size_t remaining = count;
        const cf32* src = samples;
        while (remaining > 0) {
            const size_t n = std::min(remaining, capacity - write_pos);
            std::copy(src, src + static_cast<std::ptrdiff_t>(n), storage_.begin() + static_cast<std::ptrdiff_t>(write_pos));
            src += n;
            remaining -= n;
            write_pos = (write_pos + n) % capacity;
        }
        size_ += count;
        cv_.notify_one();
    }

    bool pop_batch(std::vector<cf32>& out, size_t min_samples, size_t max_samples, std::chrono::milliseconds timeout)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock, timeout, [&] {
            return closed_ || size_ >= min_samples;
        });
        if (size_ == 0) {
            return false;
        }
        const size_t capacity = storage_.size();
        const size_t n = std::min(size_, std::max(min_samples, max_samples));
        out.resize(n);
        size_t copied = 0;
        while (copied < n) {
            const size_t chunk = std::min(n - copied, capacity - read_pos_);
            std::copy(
                storage_.begin() + static_cast<std::ptrdiff_t>(read_pos_),
                storage_.begin() + static_cast<std::ptrdiff_t>(read_pos_ + chunk),
                out.begin() + static_cast<std::ptrdiff_t>(copied));
            read_pos_ = (read_pos_ + chunk) % capacity;
            copied += chunk;
        }
        size_ -= n;
        return true;
    }

    bool pop_latest(std::vector<cf32>& out, size_t max_samples)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (size_ == 0) {
            return false;
        }
        const size_t capacity = storage_.size();
        if (size_ > max_samples) {
            const size_t drop = size_ - max_samples;
            read_pos_ = (read_pos_ + drop) % capacity;
            size_ -= drop;
            dropped_samples_ += static_cast<uint64_t>(drop);
        }
        const size_t n = size_;
        out.resize(n);
        size_t copied = 0;
        while (copied < n) {
            const size_t chunk = std::min(n - copied, capacity - read_pos_);
            std::copy(
                storage_.begin() + static_cast<std::ptrdiff_t>(read_pos_),
                storage_.begin() + static_cast<std::ptrdiff_t>(read_pos_ + chunk),
                out.begin() + static_cast<std::ptrdiff_t>(copied));
            read_pos_ = (read_pos_ + chunk) % capacity;
            copied += chunk;
        }
        size_ = 0;
        return true;
    }

    void close()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        cv_.notify_all();
    }

    size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_;
    }

    uint64_t dropped_samples() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_samples_;
    }

    size_t capacity() const
    {
        return storage_.size();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<cf32> storage_;
    size_t read_pos_ = 0;
    size_t size_ = 0;
    uint64_t dropped_samples_ = 0;
    bool closed_ = false;
};

#ifdef HAVE_GPU_FULL_PIPELINE
static void check_cuda_runtime(cudaError_t status, const char* what)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

class GpuOnlineIqBuffer {
public:
    explicit GpuOnlineIqBuffer(size_t capacity_bytes)
    {
        const size_t sample_bytes = sizeof(cf32);
        capacity_samples_ = std::max<size_t>(capacity_bytes / sample_bytes, 4096);
        const size_t alloc_bytes = capacity_samples_ * sample_bytes;
        check_cuda_runtime(cudaMalloc(&device_storage_, alloc_bytes), "cudaMalloc RX GPU IQ ring");
        check_cuda_runtime(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking), "cudaStreamCreate RX GPU IQ ring");
    }

    ~GpuOnlineIqBuffer()
    {
        if (copy_stream_ != nullptr) {
            cudaStreamDestroy(copy_stream_);
        }
        if (device_storage_ != nullptr) {
            cudaFree(device_storage_);
        }
        if (batch_device_ != nullptr) {
            cudaFree(batch_device_);
        }
    }

    GpuOnlineIqBuffer(const GpuOnlineIqBuffer&) = delete;
    GpuOnlineIqBuffer& operator=(const GpuOnlineIqBuffer&) = delete;

    void push(const cf32* samples, size_t count)
    {
        if (samples == nullptr || count == 0) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return;
        }
        if (count > capacity_samples_) {
            const size_t skip = count - capacity_samples_;
            samples += skip;
            count = capacity_samples_;
            dropped_samples_ += static_cast<uint64_t>(skip);
        }
        if (size_ + count > capacity_samples_) {
            const size_t drop = size_ + count - capacity_samples_;
            read_pos_ = (read_pos_ + drop) % capacity_samples_;
            size_ -= drop;
            dropped_samples_ += static_cast<uint64_t>(drop);
        }

        size_t write_pos = (read_pos_ + size_) % capacity_samples_;
        size_t remaining = count;
        const cf32* src = samples;
        while (remaining > 0) {
            const size_t n = std::min(remaining, capacity_samples_ - write_pos);
            check_cuda_runtime(
                cudaMemcpyAsync(
                    device_storage_ + write_pos,
                    src,
                    n * sizeof(cf32),
                    cudaMemcpyHostToDevice,
                    copy_stream_),
                "copy RX samples to GPU IQ ring");
            src += n;
            remaining -= n;
            write_pos = (write_pos + n) % capacity_samples_;
        }
        size_ += count;
        cv_.notify_one();
    }

    bool pop_batch_to_host(std::vector<cf32>& out, size_t min_samples, size_t max_samples, std::chrono::milliseconds timeout)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock, timeout, [&] {
            return closed_ || size_ >= min_samples;
        });
        if (size_ == 0) {
            return false;
        }
        check_cuda_runtime(cudaStreamSynchronize(copy_stream_), "sync RX GPU IQ ring before pop");
        const size_t n = std::min(size_, std::max(min_samples, max_samples));
        out.resize(n);
        size_t copied = 0;
        while (copied < n) {
            const size_t chunk = std::min(n - copied, capacity_samples_ - read_pos_);
            check_cuda_runtime(
                cudaMemcpyAsync(
                    out.data() + copied,
                    device_storage_ + read_pos_,
                    chunk * sizeof(cf32),
                    cudaMemcpyDeviceToHost,
                    copy_stream_),
                "copy RX samples from GPU IQ ring");
            read_pos_ = (read_pos_ + chunk) % capacity_samples_;
            copied += chunk;
        }
        check_cuda_runtime(cudaStreamSynchronize(copy_stream_), "sync RX GPU IQ ring pop");
        size_ -= n;
        return true;
    }

    bool pop_latest_to_host(std::vector<cf32>& out, size_t max_samples)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (size_ == 0) {
            return false;
        }
        if (size_ > max_samples) {
            const size_t drop = size_ - max_samples;
            read_pos_ = (read_pos_ + drop) % capacity_samples_;
            size_ -= drop;
            dropped_samples_ += static_cast<uint64_t>(drop);
        }
        check_cuda_runtime(cudaStreamSynchronize(copy_stream_), "sync RX GPU IQ ring before latest pop");
        const size_t n = size_;
        out.resize(n);
        size_t copied = 0;
        while (copied < n) {
            const size_t chunk = std::min(n - copied, capacity_samples_ - read_pos_);
            check_cuda_runtime(
                cudaMemcpyAsync(
                    out.data() + copied,
                    device_storage_ + read_pos_,
                    chunk * sizeof(cf32),
                    cudaMemcpyDeviceToHost,
                    copy_stream_),
                "copy latest RX samples from GPU IQ ring");
            read_pos_ = (read_pos_ + chunk) % capacity_samples_;
            copied += chunk;
        }
        check_cuda_runtime(cudaStreamSynchronize(copy_stream_), "sync latest RX GPU IQ ring pop");
        size_ = 0;
        return true;
    }

    bool pop_batch_device(const cf32*& d_out, size_t& out_count, size_t min_samples, size_t max_samples, std::chrono::milliseconds timeout)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock, timeout, [&] {
            return closed_ || size_ >= min_samples;
        });
        if (size_ == 0) {
            d_out = nullptr;
            out_count = 0;
            return false;
        }
        check_cuda_runtime(cudaStreamSynchronize(copy_stream_), "sync RX GPU IQ ring before device pop");
        const size_t n = std::min(size_, std::max(min_samples, max_samples));
        ensure_batch_capacity(n);
        copy_to_batch_locked(n);
        size_ -= n;
        d_out = batch_device_;
        out_count = n;
        return true;
    }

    bool peek_batch_device(const cf32*& d_out, size_t& out_count, size_t min_samples, size_t max_samples, std::chrono::milliseconds timeout)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock, timeout, [&] {
            return closed_ || size_ >= min_samples;
        });
        if (size_ == 0) {
            d_out = nullptr;
            out_count = 0;
            return false;
        }
        check_cuda_runtime(cudaStreamSynchronize(copy_stream_), "sync RX GPU IQ ring before device peek");
        const size_t n = std::min(size_, std::max(min_samples, max_samples));
        ensure_batch_capacity(n);
        copy_to_batch_locked(n, false);
        d_out = batch_device_;
        out_count = n;
        return true;
    }

    void consume_device_samples(size_t count)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t n = std::min(count, size_);
        read_pos_ = (read_pos_ + n) % capacity_samples_;
        size_ -= n;
    }

    bool pop_latest_device(const cf32*& d_out, size_t& out_count, size_t max_samples)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (size_ == 0) {
            d_out = nullptr;
            out_count = 0;
            return false;
        }
        if (size_ > max_samples) {
            const size_t drop = size_ - max_samples;
            read_pos_ = (read_pos_ + drop) % capacity_samples_;
            size_ -= drop;
            dropped_samples_ += static_cast<uint64_t>(drop);
        }
        check_cuda_runtime(cudaStreamSynchronize(copy_stream_), "sync RX GPU IQ ring before latest device pop");
        const size_t n = size_;
        ensure_batch_capacity(n);
        copy_to_batch_locked(n);
        size_ = 0;
        d_out = batch_device_;
        out_count = n;
        return true;
    }

    void close()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        cv_.notify_all();
    }

    size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_;
    }

    size_t capacity() const
    {
        return capacity_samples_;
    }

    uint64_t dropped_samples() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_samples_;
    }

private:
    void ensure_batch_capacity(size_t samples)
    {
        if (samples <= batch_capacity_samples_) {
            return;
        }
        if (batch_device_ != nullptr) {
            check_cuda_runtime(cudaFree(batch_device_), "cudaFree RX GPU IQ batch");
            batch_device_ = nullptr;
            batch_capacity_samples_ = 0;
        }
        check_cuda_runtime(cudaMalloc(&batch_device_, samples * sizeof(cf32)), "cudaMalloc RX GPU IQ batch");
        batch_capacity_samples_ = samples;
    }

    void copy_to_batch_locked(size_t n, bool advance = true)
    {
        size_t copied = 0;
        size_t pos = read_pos_;
        while (copied < n) {
            const size_t chunk = std::min(n - copied, capacity_samples_ - pos);
            check_cuda_runtime(
                cudaMemcpyAsync(
                    batch_device_ + copied,
                    device_storage_ + pos,
                    chunk * sizeof(cf32),
                    cudaMemcpyDeviceToDevice,
                    copy_stream_),
                "copy RX GPU IQ ring to device batch");
            pos = (pos + chunk) % capacity_samples_;
            copied += chunk;
        }
        check_cuda_runtime(cudaStreamSynchronize(copy_stream_), "sync RX GPU IQ device batch");
        if (advance) {
            read_pos_ = pos;
        }
    }

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    cf32* device_storage_ = nullptr;
    cf32* batch_device_ = nullptr;
    cudaStream_t copy_stream_ = nullptr;
    size_t capacity_samples_ = 0;
    size_t batch_capacity_samples_ = 0;
    size_t read_pos_ = 0;
    size_t size_ = 0;
    uint64_t dropped_samples_ = 0;
    bool closed_ = false;
};
#endif

class RxFrameQueue {
public:
    explicit RxFrameQueue(size_t capacity_frames)
        : capacity_frames_(std::max<size_t>(capacity_frames, 4))
    {}

    void push(ExtractedFrame frame)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return;
        }
        if (queue_.size() >= capacity_frames_) {
            queue_.pop_front();
            ++dropped_frames_;
        }
        queue_.push_back(std::move(frame));
        cv_.notify_one();
    }

    bool pop_for(ExtractedFrame& frame, std::chrono::milliseconds timeout)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock, timeout, [this] {
            return closed_ || !queue_.empty();
        });
        if (queue_.empty()) {
            return false;
        }
        frame = std::move(queue_.front());
        queue_.pop_front();
        return true;
    }

    void close()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        cv_.notify_all();
    }

    uint64_t dropped_frames() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_frames_;
    }

    size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<ExtractedFrame> queue_;
    size_t capacity_frames_ = 0;
    uint64_t dropped_frames_ = 0;
    bool closed_ = false;
};

struct TxQueuedFrame {
    std::vector<cf32> samples;
    std::vector<usrp_link::ci16> packed_sc16_samples;
    uint32_t frame_id = 0;
    uint64_t unique_frames = 0;
    int info_bits_per_frame = 0;
    int repeat = 1;
    int activate_mcs_after_send = -1;
    bool counts_unique_payload = false;
};

class TxFrameQueue {
public:
    explicit TxFrameQueue(size_t capacity_frames)
        : capacity_frames_(std::max<size_t>(capacity_frames, 1))
    {}

    bool push(TxQueuedFrame frame, const std::atomic<bool>& stop)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_space_.wait(lock, [&] {
            return closed_ || stop.load(std::memory_order_relaxed) || queue_.size() < capacity_frames_;
        });
        if (closed_ || stop.load(std::memory_order_relaxed)) {
            return false;
        }
        queue_.push_back(std::move(frame));
        cv_data_.notify_one();
        return true;
    }

    bool pop_for(TxQueuedFrame& frame, std::chrono::milliseconds timeout)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_data_.wait_for(lock, timeout, [&] {
            return closed_ || !queue_.empty();
        });
        if (queue_.empty()) {
            return false;
        }
        frame = std::move(queue_.front());
        queue_.pop_front();
        cv_space_.notify_one();
        return true;
    }

    void wait_for_prefill(size_t frames, const std::atomic<bool>& stop)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_data_.wait(lock, [&] {
            return closed_ || stop.load(std::memory_order_relaxed) || queue_.size() >= frames;
        });
    }

    void close()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        cv_data_.notify_all();
        cv_space_.notify_all();
    }

    size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    bool closed_and_empty() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_ && queue_.empty();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_data_;
    std::condition_variable cv_space_;
    std::deque<TxQueuedFrame> queue_;
    size_t capacity_frames_ = 0;
    bool closed_ = false;
};

#ifdef HAVE_GPU_FULL_PIPELINE
class RxDeviceFrameQueue {
public:
    explicit RxDeviceFrameQueue(size_t capacity_frames)
        : capacity_frames_(std::max<size_t>(capacity_frames, 4))
    {}

    void push(ExtractedDeviceFrame frame)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) {
            return;
        }
        if (queue_.size() >= capacity_frames_) {
            queue_.pop_front();
            ++dropped_frames_;
        }
        queue_.push_back(std::move(frame));
        cv_.notify_one();
    }

    bool pop_for(ExtractedDeviceFrame& frame, std::chrono::milliseconds timeout)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock, timeout, [this] {
            return closed_ || !queue_.empty();
        });
        if (queue_.empty()) {
            return false;
        }
        frame = std::move(queue_.front());
        queue_.pop_front();
        return true;
    }

    void close()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        cv_.notify_all();
    }

    uint64_t dropped_frames() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_frames_;
    }

    size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<ExtractedDeviceFrame> queue_;
    size_t capacity_frames_ = 0;
    uint64_t dropped_frames_ = 0;
    bool closed_ = false;
};
#endif

static void wait_for_start_gate(const Options& opt, const char* role)
{
    if (opt.start_gate_file.empty()) {
        return;
    }

    const std::filesystem::path gate_path(opt.start_gate_file);
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "[" << role << "] waiting start gate file=" << gate_path.string();
    if (opt.start_gate_timeout_sec > 0.0) {
        std::cout << " timeout=" << opt.start_gate_timeout_sec << " s";
    }
    std::cout << "\n";
    std::cout.flush();

    while (!std::filesystem::exists(gate_path)) {
        if (opt.start_gate_timeout_sec > 0.0) {
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            if (elapsed >= opt.start_gate_timeout_sec) {
                throw std::runtime_error(
                    std::string(role) + " start gate timed out waiting for " + gate_path.string());
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::cout << "[" << role << "] start gate released after "
              << std::fixed << std::setprecision(3) << elapsed << " s\n";
    std::cout.flush();
}

static int run_tx(const Options& opt, const PhyConfig& phy, const LdpcCode& code)
{
    usrp_link::set_realtime_priority();
    const bool file_mode = opt.traffic_mode == "file";
    const int info_bits = link_info_bits_per_frame(phy, code);
    const size_t info_bytes = link_info_bytes_per_frame(phy, code);
    InjectionChannel precomp_channel;
    if (!opt.precomp_channel_files.empty()) {
        precomp_channel = load_injection_channel(opt.precomp_channel_files);
        print_injection_channel("tx-precomp", precomp_channel);
    }
    InjectionChannel actual_channel;
    if (!opt.actual_channel_files.empty()) {
        actual_channel = load_injection_channel(opt.actual_channel_files);
        print_injection_channel("tx-actual", actual_channel);
    }
    TxBasebandChain baseband(
        phy,
        code,
        precomp_channel.empty() ? nullptr : &precomp_channel,
        actual_channel.empty() ? nullptr : &actual_channel);
    std::unique_ptr<MediaPacketizer> media;
    if (file_mode) {
        if (opt.input_file.empty()) {
            throw std::runtime_error("--traffic file requires --input <path> on TX");
        }
        media = std::make_unique<MediaPacketizer>(opt.input_file, info_bytes, opt.loop_file);
        std::cout << "[MEDIA] TX input=" << opt.input_file
                  << " bytes=" << media->file_bytes.size()
                  << " chunks=" << media->total_chunks
                  << " payload/frame=" << media->payload_bytes_per_frame
                  << " loop=" << (opt.loop_file ? "yes" : "no") << "\n";
    }

    auto radio = usrp_link::RadioEndpoint::open_tx(radio_config_for_mode(opt, "tx"));
    const auto pp = radio.pp_string();
    if (!pp.empty()) {
        std::cout << "TX ready: " << pp << "\n";
    }
    std::cout << "[TX-STREAM] hostFormat=" << opt.tx_host_format
              << " sendBatch=" << (opt.tx_host_format == "sc16" ? "32768" : "packet-limit") << "\n";
    if (std::isfinite(opt.tx_awgn_snr_db)) {
        std::cout << "[TX-AWGN] enabled snr=" << opt.tx_awgn_snr_db
                  << " dB scope=" << opt.tx_awgn_scope
                  << " reference=" << opt.tx_awgn_reference
                  << " injected before UHD send\n";
    }
    wait_for_start_gate(opt, "TX");

    uint64_t sent_frames = 0;
    uint64_t unique_frames_sent = 0;
    auto t0 = std::chrono::steady_clock::now();
    bool first_packet = true;
    std::unique_ptr<AdaptiveTxController> adaptive_tx;
    if (opt.adaptive && file_mode) {
        adaptive_tx = std::make_unique<AdaptiveTxController>(opt);
    }
    TxFrameQueue tx_frame_queue(static_cast<size_t>(opt.tx_queue_frames));
    std::atomic<bool> stop{false};
    std::exception_ptr producer_error;
    std::cout << "[TX-QUEUE] enabled capacity=" << opt.tx_queue_frames
              << " frames prefill=" << std::min<size_t>(static_cast<size_t>(opt.tx_queue_frames), 32)
              << " hostFormat=" << opt.tx_host_format << "\n";

    std::thread producer([&] {
        try {
            usrp_link::set_realtime_priority();
            uint32_t frame_id = 0;
            uint64_t unique_frames = 0;
            int last_repeat_report = adaptive_tx ? adaptive_tx->repeat_count() : 1;
            uint64_t last_control_generation = adaptive_tx ? adaptive_tx->decision_generation() : 0;
            std::mt19937 tx_noise_rng(static_cast<uint32_t>(opt.test_seed) ^ 0xa6175eedu);

            while (!stop.load(std::memory_order_relaxed)) {
                if (file_mode && media->done(unique_frames)) {
                    break;
                }
                const auto now = std::chrono::steady_clock::now();
                const double elapsed = std::chrono::duration<double>(now - t0).count();
                if (opt.frames > 0 && static_cast<int>(unique_frames) >= opt.frames) {
                    break;
                }
                if (opt.duration_sec > 0.0 && elapsed >= opt.duration_sec) {
                    break;
                }

                std::vector<cf32> frame;
                bool control_frame = false;
                if (file_mode) {
                    if (adaptive_tx && adaptive_tx->decision_generation() != last_control_generation) {
                        last_control_generation = adaptive_tx->decision_generation();
                        const auto decision = adaptive_tx->current_decision();
                        const auto info = media->build_control_info_bits(
                            frame_id,
                            decision,
                            frame_id + 1,
                            static_cast<float>(opt.adaptive_predicted_snr_db));
                        frame = baseband.build_media_frame(info);
                        control_frame = true;
                        std::cout << "[CTRL-TX] frame=" << frame_id
                                  << " effectiveFrame=" << (frame_id + 1)
                                  << " mcs=" << static_cast<int>(decision.mcs)
                                  << " modulation=" << decision.modulation
                                  << " repeat=" << decision.repeat_count
                                  << "\n";
                    } else {
                        const auto info = media->build_media_info_bits(frame_id, unique_frames);
                        frame = baseband.build_media_frame(info);
                    }
                } else {
                    frame = baseband.build_test_frame(frame_id);
                }

                const int repeat = control_frame ? std::min(opt.tx_repeat_max, std::max(1, 2))
                                                 : (adaptive_tx ? adaptive_tx->repeat_count() : 1);
                if (repeat != last_repeat_report) {
                    std::cout << "[TX] adaptive repeat=" << repeat << "\n";
                    last_repeat_report = repeat;
                }

                const uint64_t unique_after_frame = unique_frames + (control_frame ? 0 : 1);
                for (int rep = 0; rep < repeat; ++rep) {
                    std::vector<cf32> noisy_frame;
                    const std::vector<cf32>* frame_to_queue = &frame;
                    if (std::isfinite(opt.tx_awgn_snr_db)) {
                        noisy_frame = frame;
                        add_tx_awgn_if_requested(noisy_frame, phy, opt, tx_noise_rng, opt.tx_awgn_snr_db);
                        frame_to_queue = &noisy_frame;
                    }

                    TxQueuedFrame queued;
                    if (opt.tx_host_format == "sc16") {
                        queued.packed_sc16_samples = quantize_tx_sc16(*frame_to_queue);
                    } else {
                        queued.samples = *frame_to_queue;
                    }
                    queued.frame_id = frame_id;
                    queued.unique_frames = unique_after_frame;
                    queued.info_bits_per_frame = info_bits;
                    queued.repeat = repeat;
                    queued.counts_unique_payload = !control_frame && rep == repeat - 1;
                    if (!tx_frame_queue.push(std::move(queued), stop)) {
                        break;
                    }
                }

                ++frame_id;
                if (!control_frame) {
                    ++unique_frames;
                }
            }
        } catch (...) {
            producer_error = std::current_exception();
            stop.store(true, std::memory_order_relaxed);
        }
        tx_frame_queue.close();
    });

    const size_t prefill = std::min<size_t>(static_cast<size_t>(opt.tx_queue_frames), 32);
    tx_frame_queue.wait_for_prefill(prefill, stop);
    TxQueuedFrame queued;
    while (tx_frame_queue.pop_for(queued, std::chrono::milliseconds(20))) {
        if (opt.tx_host_format == "sc16") {
            send_tx_packed_sc16_frame(radio, queued.packed_sc16_samples, first_packet, "[TX]");
        } else {
            send_tx_frame(radio, queued.samples, opt.tx_host_format, first_packet, "[TX]");
        }
        ++sent_frames;
        if (queued.counts_unique_payload) {
            ++unique_frames_sent;
        }

        if (sent_frames % static_cast<uint64_t>(std::max(opt.report_every, 1)) == 0) {
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - t0).count();
            std::cout << "[TX] frame=" << queued.frame_id
                      << " sent=" << sent_frames
                      << " unique=" << unique_frames_sent
                      << " repeat=" << queued.repeat
                      << " queueDepth=" << tx_frame_queue.size()
                      << " info=" << (static_cast<double>(unique_frames_sent) * static_cast<double>(info_bits) / std::max(elapsed, 1e-9) / 1e6)
                      << " Mbps "
                      << " elapsed=" << std::fixed << std::setprecision(2) << elapsed << " s\n";
        }
    }
    stop.store(true, std::memory_order_relaxed);
    tx_frame_queue.close();
    if (producer.joinable()) {
        producer.join();
    }
    if (producer_error) {
        std::rethrow_exception(producer_error);
    }

    try {
        radio.send_end_of_burst(1.0);
    } catch (const std::exception& e) {
        std::cerr << "[TX] " << e.what() << "\n";
    }
    return 0;
}

static int run_rx(const Options& opt, const PhyConfig& phy, const LdpcCode& code, const fec::RxDecoderPlan& rx_plan)
{
    usrp_link::set_realtime_priority();
    const bool file_mode = opt.traffic_mode == "file";
    const int info_bits = link_info_bits_per_frame(phy, code);
    const size_t payload_capacity = file_mode ? media_payload_capacity_bytes(phy, code) : 0;
    std::unique_ptr<CudaBpDecoder> cuda_bp_decoder;
    std::unique_ptr<CudaBpOsdDecoder> cuda_decoder;
    if (rx_plan.kind == fec::RxDecoderKind::GpuBp) {
        cuda_bp_decoder = std::make_unique<CudaBpDecoder>(opt, code, *rx_plan.fec);
    } else if (rx_plan.kind == fec::RxDecoderKind::GpuBpOsd) {
        cuda_decoder = std::make_unique<CudaBpOsdDecoder>(opt, code);
    }
    std::unique_ptr<MediaReassembler> media;
    if (file_mode) {
        media = std::make_unique<MediaReassembler>(default_output_file(opt));
        std::cout << "[MEDIA] RX output=" << media->output_path
                  << " payload/frame=" << payload_capacity << "\n";
    }
    std::unique_ptr<AdaptiveFeedbackSender> adaptive_feedback;
    std::unique_ptr<AdaptiveRxDecision> adaptive_decision;
    InjectionChannel adaptive_prediction_channel;
    std::unique_ptr<usrp_link::PredictiveAmcPolicy> predictive_amc;
    if (opt.adaptive && file_mode) {
        adaptive_feedback = std::make_unique<AdaptiveFeedbackSender>(
            opt.adaptive_feedback_host,
            opt.adaptive_feedback_port);
        adaptive_decision = std::make_unique<AdaptiveRxDecision>(opt);
        const bool use_predictive = opt.adaptive_predictive || !opt.precomp_channel_files.empty();
        if (use_predictive) {
            if (!opt.precomp_channel_files.empty()) {
                adaptive_prediction_channel = load_injection_channel(opt.precomp_channel_files);
                print_injection_channel("adaptive-prediction", adaptive_prediction_channel);
            }
            predictive_amc = std::make_unique<usrp_link::PredictiveAmcPolicy>(
                make_predictive_amc_config(opt));
            std::cout << "[ADAPT-RX] predictive AMC enabled: predictedSNR="
                      << std::fixed << std::setprecision(1) << opt.adaptive_predicted_snr_db
                      << " confidence=" << opt.adaptive_prediction_confidence
                      << " ageMs=" << opt.adaptive_prediction_age_ms
                      << " evm=" << opt.adaptive_prediction_evm << "\n";
        }
    }

#ifdef HAVE_GPU_FULL_PIPELINE
    std::unique_ptr<GpuRxSyncExtractor> gpu_rx_sync;
    if (opt.gpu_rx_sync) {
        gpu_rx_sync = std::make_unique<GpuRxSyncExtractor>(phy, code);
    }
    std::unique_ptr<GpuRxDemodChain> gpu_rx_demod;
    if (opt.gpu_rx_demod) {
        gpu_rx_demod = std::make_unique<GpuRxDemodChain>(phy, code, nullptr);
    }
#endif

    auto radio = usrp_link::RadioEndpoint::open_rx(radio_config_for_mode(opt, "rx"));
    const auto pp = radio.pp_string();
    if (!pp.empty()) {
        std::cout << "RX ready: " << pp << "\n";
    }

    wait_for_start_gate(opt, "RX");
    radio.start_rx();

    const size_t chunk = std::max<size_t>(radio.rx_max_samps(), static_cast<size_t>(opt.rx_block_samps));
    RxSampleQueue sample_queue(static_cast<size_t>(opt.rx_queue_blocks));
    RxFrameQueue frame_queue(static_cast<size_t>(opt.rx_frame_queue_frames));
#ifdef HAVE_GPU_FULL_PIPELINE
    std::unique_ptr<GpuOnlineIqBuffer> gpu_iq_buffer;
    if (opt.rx_gpu_buffered) {
        gpu_iq_buffer = std::make_unique<GpuOnlineIqBuffer>(opt.rx_gpu_buffer_bytes);
        std::cout << "[RX-GPU-BUFFER] enabled capacity=" << opt.rx_gpu_buffer_bytes
                  << " bytes samples=" << gpu_iq_buffer->capacity()
                  << " approx=" << (static_cast<double>(gpu_iq_buffer->capacity()) / std::max(phy.rate, 1.0))
                  << " s\n";
    }
    const bool use_device_frame_queue = (gpu_iq_buffer != nullptr && gpu_rx_sync != nullptr && gpu_rx_demod != nullptr);
    std::unique_ptr<DeviceFramePool> device_frame_pool;
    if (use_device_frame_queue) {
        device_frame_pool = std::make_unique<DeviceFramePool>(
            static_cast<size_t>(opt.rx_frame_queue_frames) + 16,
            static_cast<size_t>(phy.frame_len()));
    }
    RxDeviceFrameQueue device_frame_queue(static_cast<size_t>(opt.rx_frame_queue_frames));
    if (use_device_frame_queue) {
        std::cout << "[RX-GPU-FRAME-QUEUE] enabled capacity=" << opt.rx_frame_queue_frames
                  << " frames; fixed device frame pool="
                  << (static_cast<size_t>(opt.rx_frame_queue_frames) + 16)
                  << " frames\n";
    }
#endif
    PipelineProfile profile;
    std::atomic<bool> rx_stop{false};
    std::atomic<bool> rx_thread_failed{false};
    std::atomic<bool> sync_thread_failed{false};
    std::exception_ptr rx_thread_exception;
    std::exception_ptr sync_thread_exception;
    std::thread rx_thread([&] {
        try {
            usrp_link::set_realtime_priority();
            std::vector<cf32> tmp(chunk);
            size_t decim_phase = 0;
            while (!rx_stop.load(std::memory_order_relaxed)) {
                size_t n = 0;
                int status = UHD_C_RX_ERROR;
                const auto t_recv0 = std::chrono::steady_clock::now();
                radio.recv(tmp.data(), tmp.size(), 0.1, n, status);
                const auto t_recv1 = std::chrono::steady_clock::now();
                if (opt.profile_pipeline) {
                    profile.rx_recv_ns.fetch_add(elapsed_ns(t_recv0, t_recv1), std::memory_order_relaxed);
                    if (status == UHD_C_RX_OK && n > 0) {
                        profile.rx_samples.fetch_add(static_cast<uint64_t>(n), std::memory_order_relaxed);
                        profile.rx_blocks.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                if (status == UHD_C_RX_TIMEOUT) {
                    continue;
                }

                RxSampleBlock block;
                block.status = status;
                if (status == UHD_C_RX_OK && n > 0) {
                    block.samples = downsample_from_radio(tmp.data(), n, phy.radio_oversample, decim_phase);
                } else if (status == UHD_C_RX_OVERFLOW) {
                    decim_phase = 0;
                }
                const auto t_push0 = std::chrono::steady_clock::now();
                if (
#ifdef HAVE_GPU_FULL_PIPELINE
                    gpu_iq_buffer &&
#else
                    false &&
#endif
                    block.status == UHD_C_RX_OK && !block.samples.empty()) {
#ifdef HAVE_GPU_FULL_PIPELINE
                    gpu_iq_buffer->push(block.samples.data(), block.samples.size());
#endif
                } else {
                    sample_queue.push(std::move(block));
                }
                const auto t_push1 = std::chrono::steady_clock::now();
                if (opt.profile_pipeline) {
                    profile.rx_enqueue_ns.fetch_add(elapsed_ns(t_push0, t_push1), std::memory_order_relaxed);
                }
            }
        } catch (...) {
            rx_thread_exception = std::current_exception();
            rx_thread_failed.store(true, std::memory_order_release);
#ifdef HAVE_GPU_FULL_PIPELINE
            if (gpu_iq_buffer) {
                gpu_iq_buffer->close();
            }
#endif
            sample_queue.close();
        }
    });

    std::thread sync_thread([&] {
        try {
            usrp_link::set_realtime_priority();
            std::vector<cf32> rxbuf;
            rxbuf.reserve(static_cast<size_t>(phy.max_buffered_frames * phy.frame_len() + 2 * chunk));
            size_t rxbase = 0;
            SyncState sync_state = SyncState::Acquisition;
            SyncStats sync_stats;
            uint64_t last_reported_queue_drops = 0;
            auto last_queue_drop_report = std::chrono::steady_clock::now();
            auto last_sync_report = std::chrono::steady_clock::now();
            uint64_t last_sync_found = 0;
            uint64_t last_sync_skipped = 0;
            uint64_t last_gpu_buffer_drops = 0;
            const size_t buffered_min = static_cast<size_t>(std::max(8 * phy.frame_len(), 2 * static_cast<int>(chunk)));
            const size_t buffered_max = static_cast<size_t>(std::max(64 * phy.frame_len(), 8 * static_cast<int>(chunk)));

            while (!rx_stop.load(std::memory_order_relaxed)) {
                RxSampleBlock block;
                const cf32* d_gpu_batch = nullptr;
                size_t gpu_batch_samples = 0;
                bool use_gpu_device_batch = false;
                if (
#ifdef HAVE_GPU_FULL_PIPELINE
                    gpu_iq_buffer && gpu_rx_sync
#else
                    false
#endif
                ) {
#ifdef HAVE_GPU_FULL_PIPELINE
                    if (gpu_iq_buffer->size() > gpu_iq_buffer->capacity() * 3 / 4) {
                        if (gpu_iq_buffer->pop_latest_device(d_gpu_batch, gpu_batch_samples, buffered_max)) {
                            rxbuf.clear();
                            rxbase = 0;
                            sync_state = SyncState::Acquisition;
                            use_gpu_device_batch = true;
                        } else {
                            continue;
                        }
                    } else {
                        if (!gpu_iq_buffer->peek_batch_device(
                                d_gpu_batch,
                                gpu_batch_samples,
                                buffered_min,
                                buffered_max,
                                std::chrono::milliseconds(100))) {
                            if (rx_stop.load(std::memory_order_relaxed) ||
                                rx_thread_failed.load(std::memory_order_acquire)) {
                                break;
                            }
                            continue;
                        }
                        use_gpu_device_batch = true;
                    }
                    block.status = UHD_C_RX_OK;
#endif
                } else {
                    if (!sample_queue.pop_for(block, std::chrono::milliseconds(100))) {
                        if (rx_stop.load(std::memory_order_relaxed) ||
                            rx_thread_failed.load(std::memory_order_acquire)) {
                            break;
                        }
                        continue;
                    }

                    const uint64_t queue_drops = sample_queue.dropped_blocks();
                    if (queue_drops != last_reported_queue_drops) {
                        const auto now = std::chrono::steady_clock::now();
                        if (queue_drops - last_reported_queue_drops >= 1024 ||
                            std::chrono::duration<double>(now - last_queue_drop_report).count() >= 1.0) {
                            std::cerr << "[RX] sample queue overrun; dropped "
                                      << (queue_drops - last_reported_queue_drops)
                                      << " block(s), total=" << queue_drops << "\n";
                            last_reported_queue_drops = queue_drops;
                            last_queue_drop_report = now;
                        }
                    }
                }

                if (block.status == UHD_C_RX_OVERFLOW) {
                    std::cerr << "[RX] overflow; dropping buffered samples\n";
                    rxbuf.clear();
                    rxbase = 0;
                    continue;
                }
                if (block.status != UHD_C_RX_OK) {
                    std::cerr << "[RX] metadata error: status=" << block.status << "\n";
                    continue;
                }

                const auto t_sync0 = std::chrono::steady_clock::now();
                size_t frames_this_block = 0;
                size_t sync_input_samples = block.samples.size();
                if (use_gpu_device_batch) {
#ifdef HAVE_GPU_FULL_PIPELINE
                    size_t gpu_base = 0;
                    sync_input_samples = gpu_batch_samples;
                    ExtractedDeviceFrame device_frame;
                    while (gpu_rx_sync->extract_device(
                        d_gpu_batch,
                        gpu_batch_samples,
                        gpu_base,
                        device_frame,
                        *device_frame_pool,
                        sync_state,
                        sync_stats)) {
                        device_frame_queue.push(std::move(device_frame));
                        ++frames_this_block;
                        device_frame = ExtractedDeviceFrame{};
                    }
                    if (gpu_base > 0) {
                        gpu_iq_buffer->consume_device_samples(gpu_base);
                    } else if (gpu_batch_samples > static_cast<size_t>(phy.frame_len())) {
                        gpu_iq_buffer->consume_device_samples(
                            gpu_batch_samples - static_cast<size_t>(phy.frame_len()));
                        sync_state = SyncState::Acquisition;
                    }
#endif
                } else {
                    rxbuf.insert(rxbuf.end(), block.samples.begin(), block.samples.end());

                    ExtractedFrame frame;
                    auto extract_one_frame = [&]() {
#ifdef HAVE_GPU_FULL_PIPELINE
                        if (gpu_rx_sync) {
                            return gpu_rx_sync->extract(rxbuf, rxbase, frame, sync_state, sync_stats);
                        }
#endif
                        return extract_rx_frame_vector(rxbuf, rxbase, phy, frame, sync_state, sync_stats);
                    };

                    while (extract_one_frame()) {
                        frame_queue.push(std::move(frame));
                        ++frames_this_block;
                        frame = ExtractedFrame{};
                    }
                }
                const auto t_sync1 = std::chrono::steady_clock::now();
                if (opt.profile_pipeline) {
                    profile.sync_blocks.fetch_add(1, std::memory_order_relaxed);
                    profile.sync_input_samples.fetch_add(static_cast<uint64_t>(sync_input_samples), std::memory_order_relaxed);
                    profile.sync_frames.fetch_add(static_cast<uint64_t>(frames_this_block), std::memory_order_relaxed);
                    profile.sync_ns.fetch_add(elapsed_ns(t_sync0, t_sync1), std::memory_order_relaxed);

                    const auto now = std::chrono::steady_clock::now();
                    const double report_elapsed = std::chrono::duration<double>(now - last_sync_report).count();
                    if (report_elapsed >= 1.0) {
                        const uint64_t found_delta = sync_stats.found - last_sync_found;
                        const uint64_t skipped_delta = sync_stats.skipped_samples - last_sync_skipped;
                        const double avg_start = sync_stats.found > 0
                            ? static_cast<double>(sync_stats.skipped_samples) / static_cast<double>(sync_stats.found)
                            : 0.0;
                        std::cout << "[SYNC] fps=" << (static_cast<double>(found_delta) / report_elapsed)
                                  << " state=" << (sync_state == SyncState::Tracking ? "tracking" : "acquisition")
                                  << " rxbuf_available=" << (rxbuf.size() >= rxbase ? rxbuf.size() - rxbase : 0)
                                  << " sample_queue_depth=" << sample_queue.size()
                                  << " sample_queue_dropped=" << sample_queue.dropped_blocks()
#ifdef HAVE_GPU_FULL_PIPELINE
                                  << " gpu_iq_buffer_samples=" << (gpu_iq_buffer ? gpu_iq_buffer->size() : 0)
                                  << " gpu_iq_buffer_dropped=" << (gpu_iq_buffer ? gpu_iq_buffer->dropped_samples() : 0)
                                  << " device_frame_queue_depth=" << (use_device_frame_queue ? device_frame_queue.size() : 0)
                                  << " device_frame_queue_dropped=" << (use_device_frame_queue ? device_frame_queue.dropped_frames() : 0)
#endif
                                  << " found=" << sync_stats.found
                                  << " miss=" << sync_stats.miss
                                  << " trackingMiss=" << sync_stats.tracking_miss
                                  << " incomplete=" << sync_stats.incomplete
                                  << " skippedDelta=" << skipped_delta
                                  << " avgStart=" << avg_start
                                  << " maxStart=" << sync_stats.max_start
                                  << "\n";
                        last_sync_report = now;
                        last_sync_found = sync_stats.found;
                        last_sync_skipped = sync_stats.skipped_samples;
#ifdef HAVE_GPU_FULL_PIPELINE
                        if (gpu_iq_buffer) {
                            const uint64_t drops = gpu_iq_buffer->dropped_samples();
                            if (drops != last_gpu_buffer_drops) {
                                std::cerr << "[RX] GPU IQ buffer dropped "
                                          << (drops - last_gpu_buffer_drops)
                                          << " sample(s), total=" << drops << "\n";
                                last_gpu_buffer_drops = drops;
                            }
                        }
#endif
                    }
                }
            }
            frame_queue.close();
#ifdef HAVE_GPU_FULL_PIPELINE
            device_frame_queue.close();
#endif
        } catch (...) {
            sync_thread_exception = std::current_exception();
            sync_thread_failed.store(true, std::memory_order_release);
            frame_queue.close();
#ifdef HAVE_GPU_FULL_PIPELINE
            device_frame_queue.close();
#endif
        }
    });

    RxStats stats;
    LinkControlState control_state;

    const auto run_t0 = std::chrono::steady_clock::now();
    auto stats_t0 = run_t0;
    bool stop_now = false;
    uint64_t last_reported_frame_drops = 0;
    uint64_t observed_frames = 0;
    bool warmup_reported = opt.warmup_frames == 0;
    while (true) {
        auto should_stop_after_empty_pop = [&]() {
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - run_t0).count();
            if (opt.duration_sec > 0.0 && elapsed >= opt.duration_sec) {
                return true;
            }
            return rx_thread_failed.load(std::memory_order_acquire) ||
                sync_thread_failed.load(std::memory_order_acquire);
        };

        FrameDecodeResult r;
#ifdef HAVE_GPU_FULL_PIPELINE
        if (use_device_frame_queue) {
            ExtractedDeviceFrame device_frame;
            if (!device_frame_queue.pop_for(device_frame, std::chrono::milliseconds(100))) {
                if (should_stop_after_empty_pop()) {
                    break;
                }
                continue;
            }
            const uint64_t frame_drops = device_frame_queue.dropped_frames();
            if (frame_drops != last_reported_frame_drops) {
                std::cerr << "[RX] device frame queue overrun; dropped "
                          << (frame_drops - last_reported_frame_drops)
                          << " frame(s), total=" << frame_drops << "\n";
                last_reported_frame_drops = frame_drops;
            }
            r = gpu_rx_demod->process_device(
                device_frame,
                cuda_bp_decoder.get(),
                cuda_decoder.get(),
                opt.ldpc_max_iter,
                opt.ldpc_normalization,
                !file_mode && !opt.skip_reference_ber,
                opt.profile_pipeline ? &profile.decode : nullptr);
        } else
#endif
        {
            ExtractedFrame frame;
            if (!frame_queue.pop_for(frame, std::chrono::milliseconds(100))) {
                if (should_stop_after_empty_pop()) {
                    break;
                }
                continue;
            }

            const uint64_t frame_drops = frame_queue.dropped_frames();
            if (frame_drops != last_reported_frame_drops) {
                std::cerr << "[RX] frame queue overrun; dropped "
                          << (frame_drops - last_reported_frame_drops)
                          << " frame(s), total=" << frame_drops << "\n";
                last_reported_frame_drops = frame_drops;
            }

#ifdef HAVE_GPU_FULL_PIPELINE
            if (gpu_rx_demod) {
                r = gpu_rx_demod->process(
                    frame,
                    cuda_bp_decoder.get(),
                    cuda_decoder.get(),
                    opt.ldpc_max_iter,
                    opt.ldpc_normalization,
                    !file_mode && !opt.skip_reference_ber,
                    opt.profile_pipeline ? &profile.decode : nullptr);
            } else
#endif
            {
                r = decode_frame(
                    frame.samples,
                    phy,
                    code,
                    cuda_bp_decoder.get(),
                    cuda_decoder.get(),
                    frame.cfo_hz,
                    frame.peak,
                    opt.ldpc_max_iter,
                    opt.ldpc_normalization,
                    opt.rx_snr_gate_db,
                    !file_mode && !opt.skip_reference_ber,
                    opt.profile_pipeline ? &profile.decode : nullptr);
            }
        }
        if (!file_mode && opt.skip_reference_ber) {
            apply_parity_only_test_result(r);
        }
        if (file_mode) {
            accept_media_decode(r, *media, payload_capacity, &control_state);
        }
        ++observed_frames;
        const bool warmup_frame =
            !file_mode &&
            opt.warmup_frames > 0 &&
            observed_frames <= static_cast<uint64_t>(opt.warmup_frames);
        if (warmup_frame) {
            if (opt.verbose ||
                observed_frames == 1 ||
                observed_frames == static_cast<uint64_t>(opt.warmup_frames) ||
                (observed_frames % static_cast<uint64_t>(std::max(opt.report_every, 1)) == 0)) {
                std::cout << "[RX-WARMUP] frame=" << observed_frames
                          << "/" << opt.warmup_frames
                          << " rxFrameId=" << r.frame_id
                          << " snr=" << std::fixed << std::setprecision(1) << r.snr_db
                          << " dB parity=" << (r.parity_ok ? "ok" : "fail")
                          << " counted=no\n";
            }
            if (observed_frames == static_cast<uint64_t>(opt.warmup_frames)) {
                stats_t0 = std::chrono::steady_clock::now();
                warmup_reported = true;
                std::cout << "[RX-WARMUP] complete; FER/BER counters start after "
                          << opt.warmup_frames << " decoded frame(s)\n";
            }
            continue;
        } else if (!warmup_reported && opt.warmup_frames > 0) {
            stats_t0 = std::chrono::steady_clock::now();
            warmup_reported = true;
        }
        update_stats(stats, r, info_bits, phy.coded_bits_per_frame());
        if (adaptive_feedback && adaptive_decision) {
            int repeat = 1;
            double avg_snr = 0.0;
            double fer = 0.0;
            int frames = 0;
            if (adaptive_decision->observe(r, repeat, avg_snr, fer, frames)) {
                if (predictive_amc) {
                    const auto pred = make_channel_prediction_frame(
                        adaptive_prediction_channel,
                        opt,
                        stats.detected);
                    const auto quality = evaluate_predictive_quality(
                        *predictive_amc,
                        adaptive_prediction_channel,
                        opt,
                        phy,
                        pred,
                        opt.adaptive_predicted_snr_db);
                    usrp_link::RxFeedbackWindow fb;
                    fb.frames = frames;
                    fb.ok_frames = frames - static_cast<int>(std::lround(fer * static_cast<double>(frames)));
                    fb.fer = fer;
                    fb.avg_snr_db = avg_snr;
                    const auto decision = predictive_amc->decide(pred, quality, &fb);
                    repeat = std::max(opt.tx_repeat_min, std::min(opt.tx_repeat_max, decision.repeat_count));
                    adaptive_feedback->send(decision, avg_snr, fer, frames);
                    print_predictive_decision("[ADAPT-RX-PRED]", decision, quality);
                } else {
                    adaptive_feedback->send(repeat, avg_snr, fer, frames);
                }
                std::cout << "[ADAPT-RX] window=" << frames
                          << " avgSNR=" << std::fixed << std::setprecision(1) << avg_snr
                          << " FER=" << std::scientific << fer
                          << " recommendRepeat=" << repeat << "\n";
            }
        }
        {
            const auto now = std::chrono::steady_clock::now();
            send_ui_metrics_if_due(stats, std::chrono::duration<double>(now - stats_t0).count(), info_bits, r);
        }
        if (opt.verbose ||
            stats.detected % static_cast<uint64_t>(std::max(opt.report_every, 1)) == 0 ||
            (!opt.suppress_error_frames && !r.frame_ok)) {
            print_frame_result("[RX]", r);
        }
        if (stats.detected % static_cast<uint64_t>(std::max(opt.report_every, 1)) == 0) {
            const auto now = std::chrono::steady_clock::now();
            print_stats(stats, std::chrono::duration<double>(now - stats_t0).count(), info_bits);
        }
        if (file_mode && media->complete && !opt.loop_file) {
            stop_now = true;
        }
        if (opt.frames > 0 && static_cast<int>(stats.detected) >= opt.frames) {
            stop_now = true;
        }

        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - run_t0).count();
        if (stop_now) {
            break;
        }
        if (opt.frames > 0 && static_cast<int>(stats.detected) >= opt.frames) {
            break;
        }
        if (opt.duration_sec > 0.0 && elapsed >= opt.duration_sec) {
            break;
        }
    }

    rx_stop.store(true, std::memory_order_relaxed);
#ifdef HAVE_GPU_FULL_PIPELINE
    if (gpu_iq_buffer) {
        gpu_iq_buffer->close();
    }
    device_frame_queue.close();
#endif
    sample_queue.close();
    frame_queue.close();
    try {
        radio.stop_rx();
    } catch (const std::exception& e) {
        std::cerr << "[RX] " << e.what() << "\n";
    }
    if (rx_thread.joinable()) {
        rx_thread.join();
    }
    if (sync_thread.joinable()) {
        sync_thread.join();
    }
    if (rx_thread_exception) {
        std::rethrow_exception(rx_thread_exception);
    }
    if (sync_thread_exception) {
        std::rethrow_exception(sync_thread_exception);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - stats_t0).count();
    print_stats(stats, elapsed, info_bits);
    if (opt.profile_pipeline) {
        print_pipeline_profile(profile, elapsed, phy, info_bits);
    }
    return 0;
}

static int run_trx_capture(const Options& opt, const PhyConfig& phy, const LdpcCode& code)
{
    usrp_link::set_realtime_priority();
    if (opt.iq_output_file.empty()) {
        throw std::runtime_error("--mode trx-capture requires --iq-output <path>");
    }
    const bool file_mode = opt.traffic_mode == "file";
    const int info_bits = link_info_bits_per_frame(phy, code);
    const size_t info_bytes = link_info_bytes_per_frame(phy, code);

    InjectionChannel precomp_channel;
    if (!opt.precomp_channel_files.empty()) {
        precomp_channel = load_injection_channel(opt.precomp_channel_files);
        print_injection_channel("tx-precomp", precomp_channel);
    }
    InjectionChannel actual_channel;
    if (!opt.actual_channel_files.empty()) {
        actual_channel = load_injection_channel(opt.actual_channel_files);
        print_injection_channel("tx-actual", actual_channel);
    }
    TxBasebandChain tx_baseband(
        phy,
        code,
        precomp_channel.empty() ? nullptr : &precomp_channel,
        actual_channel.empty() ? nullptr : &actual_channel);
#ifdef HAVE_GPU_FULL_PIPELINE
    std::unique_ptr<GpuTxBasebandChain> gpu_tx_baseband;
    if (opt.gpu_tx_baseband) {
        gpu_tx_baseband = std::make_unique<GpuTxBasebandChain>(
            phy,
            code,
            precomp_channel.empty() ? nullptr : &precomp_channel,
            actual_channel.empty() ? nullptr : &actual_channel);
    }
#endif

    std::unique_ptr<MediaPacketizer> tx_media;
    if (file_mode) {
        if (opt.input_file.empty()) {
            throw std::runtime_error("--traffic file requires --input <path> in trx-capture mode");
        }
        tx_media = std::make_unique<MediaPacketizer>(opt.input_file, info_bytes, opt.loop_file);
        std::cout << "[MEDIA] CAPTURE input=" << opt.input_file
                  << " bytes=" << tx_media->file_bytes.size()
                  << " chunks=" << tx_media->total_chunks
                  << " payload/frame=" << tx_media->payload_bytes_per_frame
                  << " loop=" << (opt.loop_file ? "yes" : "no") << "\n";
    }

    std::ofstream iq_out(opt.iq_output_file, std::ios::binary | std::ios::trunc);
    if (!iq_out) {
        throw std::runtime_error("failed to open IQ capture output: " + opt.iq_output_file);
    }

    auto radio = usrp_link::RadioEndpoint::open_trx(duplex_radio_config(opt));
    const auto pp = radio.pp_string();
    if (!pp.empty()) {
        std::cout << "TRX capture ready: " << pp << "\n";
    }
    std::cout << "[TX-STREAM] hostFormat=" << opt.tx_host_format
              << " sendBatch=" << (opt.tx_host_format == "sc16" ? "32768" : "packet-limit") << "\n";
    radio.start_rx();
    if (std::isfinite(opt.tx_awgn_snr_db)) {
        std::cout << "[TX-AWGN] enabled snr=" << opt.tx_awgn_snr_db
                  << " dB scope=" << opt.tx_awgn_scope
                  << " reference=" << opt.tx_awgn_reference
                  << " injected before UHD send\n";
    }

    const size_t rx_chunk = std::max<size_t>(radio.rx_max_samps(), static_cast<size_t>(opt.rx_block_samps));
    std::atomic<bool> stop{false};
    std::atomic<bool> tx_done{false};
    std::atomic<uint64_t> captured_samples{0};
    std::atomic<uint64_t> captured_blocks{0};
    std::atomic<uint64_t> rx_overflows{0};
    std::atomic<uint64_t> tx_frames{0};
    std::exception_ptr rx_thread_exception;
    std::exception_ptr tx_thread_exception;
    const auto t0 = std::chrono::steady_clock::now();

    std::thread rx_thread([&] {
        try {
            usrp_link::set_realtime_priority();
            std::vector<cf32> tmp(rx_chunk);
            std::vector<cf32> baseband;
            size_t decim_phase = 0;
            auto last_report = std::chrono::steady_clock::now();
            uint64_t last_samples = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                size_t n = 0;
                int status = UHD_C_RX_ERROR;
                radio.recv(tmp.data(), tmp.size(), 0.1, n, status);
                if (status == UHD_C_RX_TIMEOUT) {
                    continue;
                }
                if (status == UHD_C_RX_OVERFLOW) {
                    ++rx_overflows;
                    decim_phase = 0;
                    continue;
                }
                if (status != UHD_C_RX_OK) {
                    std::cerr << "[CAPTURE-RX] metadata error: status=" << status << "\n";
                    continue;
                }
                if (n == 0) {
                    continue;
                }

                baseband = downsample_from_radio(tmp.data(), n, phy.radio_oversample, decim_phase);
                if (!baseband.empty()) {
                    iq_out.write(
                        reinterpret_cast<const char*>(baseband.data()),
                        static_cast<std::streamsize>(baseband.size() * sizeof(cf32)));
                    if (!iq_out) {
                        throw std::runtime_error("failed while writing IQ capture file");
                    }
                    captured_samples.fetch_add(static_cast<uint64_t>(baseband.size()), std::memory_order_relaxed);
                    captured_blocks.fetch_add(1, std::memory_order_relaxed);
                }

                const auto now = std::chrono::steady_clock::now();
                const double elapsed = std::chrono::duration<double>(now - last_report).count();
                if (elapsed >= 1.0) {
                    const uint64_t samples_now = captured_samples.load(std::memory_order_relaxed);
                    const double msps = static_cast<double>(samples_now - last_samples) / elapsed / 1.0e6;
                    std::cout << "[CAPTURE-RX] captured=" << samples_now
                              << " samples rate=" << std::fixed << std::setprecision(2) << msps
                              << " Msps blocks=" << captured_blocks.load(std::memory_order_relaxed)
                              << " overflows=" << rx_overflows.load(std::memory_order_relaxed)
                              << "\n";
                    last_report = now;
                    last_samples = samples_now;
                }
            }
        } catch (...) {
            rx_thread_exception = std::current_exception();
            stop.store(true, std::memory_order_release);
        }
    });

    std::thread tx_thread([&] {
        try {
            usrp_link::set_realtime_priority();
            uint32_t frame_id = 0;
            uint64_t unique_frames = 0;
            bool first_packet = true;
            while (!stop.load(std::memory_order_relaxed)) {
                if (file_mode && tx_media->done(unique_frames)) {
                    break;
                }

                std::vector<cf32> frame;
                if (file_mode) {
                    const auto info = tx_media->build_info_bits(frame_id);
#ifdef HAVE_GPU_FULL_PIPELINE
                    if (gpu_tx_baseband) {
                        frame = gpu_tx_baseband->build_media_frame(info);
                    } else
#endif
                    {
                        frame = tx_baseband.build_media_frame(info);
                    }
                } else {
#ifdef HAVE_GPU_FULL_PIPELINE
                    if (gpu_tx_baseband) {
                        frame = gpu_tx_baseband->build_test_frame(frame_id);
                    } else
#endif
                    {
                        frame = tx_baseband.build_test_frame(frame_id);
                    }
                }

                const int repeat = (opt.adaptive && file_mode) ? std::max(opt.tx_repeat_min, 1) : 1;
                for (int rep = 0; rep < repeat && !stop.load(std::memory_order_relaxed); ++rep) {
                    send_tx_frame(radio, frame, opt.tx_host_format, first_packet, "[CAPTURE-TX]");
                    tx_frames.fetch_add(1, std::memory_order_relaxed);
                }

                const uint64_t sent_frames = tx_frames.load(std::memory_order_relaxed);
                if (sent_frames % static_cast<uint64_t>(std::max(opt.report_every, 1)) == 0) {
                    const auto now = std::chrono::steady_clock::now();
                    const double elapsed = std::chrono::duration<double>(now - t0).count();
                    std::cout << "[CAPTURE-TX] frame=" << frame_id
                              << " sent=" << sent_frames
                              << " repeat=" << repeat
                              << " info=" << (static_cast<double>(sent_frames) * static_cast<double>(info_bits) / std::max(elapsed, 1e-9) / 1e6)
                              << " Mbps elapsed=" << std::fixed << std::setprecision(2) << elapsed << " s\n";
                }

                ++frame_id;
                ++unique_frames;
                const auto now = std::chrono::steady_clock::now();
                const double elapsed = std::chrono::duration<double>(now - t0).count();
                if (opt.frames > 0 && static_cast<int>(unique_frames) >= opt.frames) {
                    break;
                }
                if (opt.duration_sec > 0.0 && elapsed >= opt.duration_sec) {
                    break;
                }
            }
            try {
                radio.send_end_of_burst(1.0);
            } catch (const std::exception& e) {
                std::cerr << "[CAPTURE-TX] " << e.what() << "\n";
            }
            tx_done.store(true, std::memory_order_release);
            if (opt.duration_sec <= 0.0) {
                stop.store(true, std::memory_order_release);
            }
        } catch (...) {
            tx_thread_exception = std::current_exception();
            tx_done.store(true, std::memory_order_release);
            stop.store(true, std::memory_order_release);
        }
    });

    while (!stop.load(std::memory_order_acquire)) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - t0).count();
        if (opt.duration_sec > 0.0 && elapsed >= opt.duration_sec) {
            stop.store(true, std::memory_order_release);
            break;
        }
        if (tx_done.load(std::memory_order_acquire) && opt.duration_sec <= 0.0) {
            stop.store(true, std::memory_order_release);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    stop.store(true, std::memory_order_release);
    try {
        radio.stop_rx();
    } catch (const std::exception& e) {
        std::cerr << "[CAPTURE-RX] " << e.what() << "\n";
    }
    if (tx_thread.joinable()) {
        tx_thread.join();
    }
    if (rx_thread.joinable()) {
        rx_thread.join();
    }
    iq_out.flush();
    if (tx_thread_exception) {
        std::rethrow_exception(tx_thread_exception);
    }
    if (rx_thread_exception) {
        std::rethrow_exception(rx_thread_exception);
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[CAPTURE] output=" << opt.iq_output_file
              << " samples=" << captured_samples.load(std::memory_order_relaxed)
              << " blocks=" << captured_blocks.load(std::memory_order_relaxed)
              << " overflows=" << rx_overflows.load(std::memory_order_relaxed)
              << " avgMsps=" << (static_cast<double>(captured_samples.load(std::memory_order_relaxed)) / std::max(elapsed, 1e-9) / 1.0e6)
              << " txFrames=" << tx_frames.load(std::memory_order_relaxed)
              << "\n";
    return rx_overflows.load(std::memory_order_relaxed) == 0 ? 0 : 2;
}

static int run_trx(const Options& opt, const PhyConfig& phy, const LdpcCode& code, const fec::RxDecoderPlan& rx_plan)
{
    usrp_link::set_realtime_priority();
    const bool file_mode = opt.traffic_mode == "file";
    const int info_bits = link_info_bits_per_frame(phy, code);
    const size_t info_bytes = link_info_bytes_per_frame(phy, code);
    size_t payload_capacity = file_mode ? media_payload_capacity_bytes(phy, code) : 0;

    InjectionChannel precomp_channel;
    if (!opt.precomp_channel_files.empty()) {
        precomp_channel = load_injection_channel(opt.precomp_channel_files);
        print_injection_channel("tx-precomp", precomp_channel);
    }
    InjectionChannel actual_channel;
    if (!opt.actual_channel_files.empty()) {
        actual_channel = load_injection_channel(opt.actual_channel_files);
        print_injection_channel("tx-actual", actual_channel);
    }
    TxBasebandChain tx_baseband(
        phy,
        code,
        precomp_channel.empty() ? nullptr : &precomp_channel,
        actual_channel.empty() ? nullptr : &actual_channel);
#ifdef HAVE_GPU_FULL_PIPELINE
    std::unique_ptr<GpuTxBasebandChain> gpu_tx_baseband;
    if (opt.gpu_tx_baseband) {
        gpu_tx_baseband = std::make_unique<GpuTxBasebandChain>(
            phy,
            code,
            precomp_channel.empty() ? nullptr : &precomp_channel,
            actual_channel.empty() ? nullptr : &actual_channel);
    }
#endif
    std::unique_ptr<MediaPacketizer> tx_media;
    if (file_mode) {
        if (opt.input_file.empty()) {
            throw std::runtime_error("--traffic file requires --input <path> in trx mode");
        }
        tx_media = std::make_unique<MediaPacketizer>(opt.input_file, info_bytes, opt.loop_file);
        std::cout << "[MEDIA] TRX input=" << opt.input_file
                  << " bytes=" << tx_media->file_bytes.size()
                  << " chunks=" << tx_media->total_chunks
                  << " payload/frame=" << tx_media->payload_bytes_per_frame
                  << " loop=" << (opt.loop_file ? "yes" : "no") << "\n";
    }

    std::unique_ptr<CudaBpDecoder> cuda_bp_decoder;
    std::unique_ptr<CudaBpOsdDecoder> cuda_decoder;
    if (rx_plan.kind == fec::RxDecoderKind::GpuBp) {
        cuda_bp_decoder = std::make_unique<CudaBpDecoder>(opt, code, *rx_plan.fec);
    } else if (rx_plan.kind == fec::RxDecoderKind::GpuBpOsd) {
        cuda_decoder = std::make_unique<CudaBpOsdDecoder>(opt, code);
    }

    std::unique_ptr<LinkRuntimeProfileBank> profile_bank;
    if (opt.adaptive_dynamic_modulation && file_mode) {
        profile_bank = std::make_unique<LinkRuntimeProfileBank>(
            opt,
            phy,
            code,
            rx_plan,
            precomp_channel.empty() ? nullptr : &precomp_channel,
            actual_channel.empty() ? nullptr : &actual_channel);
        payload_capacity = profile_bank->common_payload_bytes();
        std::cout << "[PROFILE-BANK] enabled profiles=6 commonPayload/frame="
                  << payload_capacity
                  << " bytes; hot switch by control effective_frame_id\n";
    } else if (opt.adaptive_predictive && file_mode) {
        std::cout << "[PROFILE-BANK] disabled; pass --adaptive-dynamic-modulation "
                     "to hot-switch QPSK/16QAM profiles\n";
    }

    std::unique_ptr<MediaReassembler> rx_media;
    if (file_mode) {
        rx_media = std::make_unique<MediaReassembler>(default_output_file(opt));
        std::cout << "[MEDIA] TRX output=" << rx_media->output_path
                  << " payload/frame=" << payload_capacity << "\n";
    }

#ifdef HAVE_GPU_FULL_PIPELINE
    std::unique_ptr<GpuRxSyncExtractor> gpu_rx_sync;
    if (opt.gpu_rx_sync) {
        gpu_rx_sync = std::make_unique<GpuRxSyncExtractor>(phy, code);
    }
    std::unique_ptr<GpuRxDemodChain> gpu_rx_demod;
    if (opt.gpu_rx_demod) {
        gpu_rx_demod = std::make_unique<GpuRxDemodChain>(
            phy,
            code,
            precomp_channel.empty() ? nullptr : &precomp_channel);
    }
#endif

    auto radio = usrp_link::RadioEndpoint::open_trx(duplex_radio_config(opt));
    const auto pp = radio.pp_string();
    if (!pp.empty()) {
        std::cout << "TRX ready: " << pp << "\n";
    }
    std::cout << "[TX-STREAM] hostFormat=" << opt.tx_host_format
              << " sendBatch=" << (opt.tx_host_format == "sc16" ? "32768" : "packet-limit") << "\n";

    radio.start_rx();
    if (std::isfinite(opt.tx_awgn_snr_db)) {
        std::cout << "[TX-AWGN] enabled snr=" << opt.tx_awgn_snr_db
                  << " dB scope=" << opt.tx_awgn_scope
                  << " reference=" << opt.tx_awgn_reference
                  << " injected before UHD send\n";
    }

    const size_t rx_chunk = std::max<size_t>(radio.rx_max_samps(), static_cast<size_t>(opt.rx_block_samps));
    RxSampleQueue sample_queue(static_cast<size_t>(opt.rx_queue_blocks));
    std::unique_ptr<OnlineIqBuffer> iq_buffer;
#ifdef HAVE_GPU_FULL_PIPELINE
    std::unique_ptr<GpuOnlineIqBuffer> gpu_iq_buffer;
    if (opt.rx_gpu_buffered) {
        gpu_iq_buffer = std::make_unique<GpuOnlineIqBuffer>(opt.rx_gpu_buffer_bytes);
        std::cout << "[RX-GPU-BUFFER] enabled capacity=" << opt.rx_gpu_buffer_bytes
                  << " bytes samples=" << gpu_iq_buffer->capacity()
                  << " approx=" << (static_cast<double>(gpu_iq_buffer->capacity()) / std::max(phy.rate, 1.0))
                  << " s\n";
    }
#endif
    if (opt.rx_buffered) {
        iq_buffer = std::make_unique<OnlineIqBuffer>(opt.rx_buffer_samples);
        std::cout << "[RX-BUFFER] enabled capacity=" << opt.rx_buffer_samples
                  << " samples approx=" << (static_cast<double>(opt.rx_buffer_samples) / std::max(phy.rate, 1.0))
                  << " s\n";
    }
    RxFrameQueue frame_queue(static_cast<size_t>(opt.rx_frame_queue_frames));
#ifdef HAVE_GPU_FULL_PIPELINE
    const bool use_device_frame_queue = (gpu_iq_buffer != nullptr && gpu_rx_sync != nullptr && gpu_rx_demod != nullptr);
    std::unique_ptr<DeviceFramePool> device_frame_pool;
    if (use_device_frame_queue) {
        device_frame_pool = std::make_unique<DeviceFramePool>(
            static_cast<size_t>(opt.rx_frame_queue_frames) + 16,
            static_cast<size_t>(phy.frame_len()));
    }
    RxDeviceFrameQueue device_frame_queue(static_cast<size_t>(opt.rx_frame_queue_frames));
    if (use_device_frame_queue) {
        std::cout << "[RX-GPU-FRAME-QUEUE] enabled capacity=" << opt.rx_frame_queue_frames
                  << " frames; fixed device frame pool="
                  << (static_cast<size_t>(opt.rx_frame_queue_frames) + 16)
                  << " frames\n";
    }
#endif
    PipelineProfile profile;
    std::atomic<bool> stop{false};
    std::atomic<bool> rx_thread_failed{false};
    std::atomic<bool> sync_thread_failed{false};
    std::atomic<bool> tx_thread_failed{false};
    std::atomic<bool> tx_done{false};
    const size_t tx_queue_capacity = (opt.adaptive && file_mode)
        ? static_cast<size_t>(1)
        : static_cast<size_t>(opt.tx_queue_frames);
    TxFrameQueue tx_frame_queue(tx_queue_capacity);
    std::atomic<int> adaptive_repeat{std::max(opt.tx_repeat_min, 1)};
    std::atomic<int> tx_active_mcs{static_cast<int>(usrp_link::McsId::MCS1_QpskRepeat3)};
    std::atomic<int> rx_active_mcs{static_cast<int>(usrp_link::McsId::MCS1_QpskRepeat3)};
    std::mutex adaptive_control_mutex;
    usrp_link::AmcDecision adaptive_control_decision;
    std::atomic<uint64_t> adaptive_control_generation{0};
    std::atomic<float> adaptive_control_snr_for_mcs{0.0f};
    std::exception_ptr rx_thread_exception;
    std::exception_ptr sync_thread_exception;
    std::exception_ptr tx_thread_exception;
    std::exception_ptr tx_producer_exception;

    std::thread rx_thread([&] {
        try {
            usrp_link::set_realtime_priority();
            std::vector<cf32> tmp(rx_chunk);
            size_t decim_phase = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                size_t n = 0;
                int status = UHD_C_RX_ERROR;
                const auto t_recv0 = std::chrono::steady_clock::now();
                radio.recv(tmp.data(), tmp.size(), 0.1, n, status);
                const auto t_recv1 = std::chrono::steady_clock::now();
                if (opt.profile_pipeline) {
                    profile.rx_recv_ns.fetch_add(elapsed_ns(t_recv0, t_recv1), std::memory_order_relaxed);
                    if (status == UHD_C_RX_OK && n > 0) {
                        profile.rx_samples.fetch_add(static_cast<uint64_t>(n), std::memory_order_relaxed);
                        profile.rx_blocks.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                if (status == UHD_C_RX_TIMEOUT) {
                    continue;
                }

                RxSampleBlock block;
                block.status = status;
                if (status == UHD_C_RX_OK && n > 0) {
                    block.samples = downsample_from_radio(tmp.data(), n, phy.radio_oversample, decim_phase);
                } else if (status == UHD_C_RX_OVERFLOW) {
                    decim_phase = 0;
                }
                const auto t_push0 = std::chrono::steady_clock::now();
                if (
#ifdef HAVE_GPU_FULL_PIPELINE
                    gpu_iq_buffer &&
#else
                    false &&
#endif
                    block.status == UHD_C_RX_OK && !block.samples.empty()) {
#ifdef HAVE_GPU_FULL_PIPELINE
                    gpu_iq_buffer->push(block.samples.data(), block.samples.size());
#endif
                } else if (iq_buffer && block.status == UHD_C_RX_OK && !block.samples.empty()) {
                    iq_buffer->push(block.samples.data(), block.samples.size());
                } else {
                    sample_queue.push(std::move(block));
                }
                const auto t_push1 = std::chrono::steady_clock::now();
                if (opt.profile_pipeline) {
                    profile.rx_enqueue_ns.fetch_add(elapsed_ns(t_push0, t_push1), std::memory_order_relaxed);
                }
            }
        } catch (...) {
            rx_thread_exception = std::current_exception();
            rx_thread_failed.store(true, std::memory_order_release);
            if (iq_buffer) {
                iq_buffer->close();
            }
#ifdef HAVE_GPU_FULL_PIPELINE
            if (gpu_iq_buffer) {
                gpu_iq_buffer->close();
            }
#endif
            sample_queue.close();
        }
    });

    std::thread sync_thread([&] {
        try {
            usrp_link::set_realtime_priority();
            std::vector<cf32> rxbuf;
            rxbuf.reserve(static_cast<size_t>(phy.max_buffered_frames * phy.frame_len() + 2 * rx_chunk));
            size_t rxbase = 0;
            SyncState sync_state = SyncState::Acquisition;
            SyncStats sync_stats;
            uint64_t last_reported_queue_drops = 0;
            auto last_queue_drop_report = std::chrono::steady_clock::now();
            auto last_sync_report = std::chrono::steady_clock::now();
            uint64_t last_sync_found = 0;
            uint64_t last_sync_skipped = 0;
            uint64_t last_buffer_drops = 0;
            const size_t buffered_min = static_cast<size_t>(std::max(8 * phy.frame_len(), 2 * static_cast<int>(rx_chunk)));
            const size_t buffered_max = static_cast<size_t>(std::max(64 * phy.frame_len(), 8 * static_cast<int>(rx_chunk)));
            std::vector<cf32> buffered_batch;

            while (!stop.load(std::memory_order_relaxed)) {
                RxSampleBlock block;
                const cf32* d_gpu_batch = nullptr;
                size_t gpu_batch_samples = 0;
                bool use_gpu_device_batch = false;
                if (
#ifdef HAVE_GPU_FULL_PIPELINE
                    gpu_iq_buffer
#else
                    false
#endif
                ) {
#ifdef HAVE_GPU_FULL_PIPELINE
                    if (gpu_rx_sync) {
                        if (gpu_iq_buffer->size() > gpu_iq_buffer->capacity() * 3 / 4) {
                            if (gpu_iq_buffer->pop_latest_device(d_gpu_batch, gpu_batch_samples, buffered_max)) {
                                rxbuf.clear();
                                rxbase = 0;
                                sync_state = SyncState::Acquisition;
                                use_gpu_device_batch = true;
                            } else {
                                continue;
                            }
                        } else {
                            if (!gpu_iq_buffer->peek_batch_device(
                                    d_gpu_batch,
                                    gpu_batch_samples,
                                    buffered_min,
                                    buffered_max,
                                    std::chrono::milliseconds(100))) {
                                if (stop.load(std::memory_order_relaxed) ||
                                    rx_thread_failed.load(std::memory_order_acquire)) {
                                    break;
                                }
                                continue;
                            }
                            use_gpu_device_batch = true;
                        }
                    } else {
                        if (gpu_iq_buffer->size() > gpu_iq_buffer->capacity() * 3 / 4) {
                            if (gpu_iq_buffer->pop_latest_to_host(buffered_batch, buffered_max)) {
                                rxbuf.clear();
                                rxbase = 0;
                                sync_state = SyncState::Acquisition;
                            } else {
                                continue;
                            }
                        } else {
                            if (!gpu_iq_buffer->pop_batch_to_host(
                                    buffered_batch,
                                    buffered_min,
                                    buffered_max,
                                    std::chrono::milliseconds(100))) {
                                if (stop.load(std::memory_order_relaxed) ||
                                    rx_thread_failed.load(std::memory_order_acquire)) {
                                    break;
                                }
                                continue;
                            }
                        }
                        block.samples.swap(buffered_batch);
                    }
                    block.status = UHD_C_RX_OK;
#endif
                } else if (iq_buffer) {
                    if (iq_buffer->size() > iq_buffer->capacity() * 3 / 4) {
                        if (iq_buffer->pop_latest(buffered_batch, buffered_max)) {
                            rxbuf.clear();
                            rxbase = 0;
                            sync_state = SyncState::Acquisition;
                        } else {
                            continue;
                        }
                    } else {
                        if (!iq_buffer->pop_batch(
                                buffered_batch,
                                buffered_min,
                                buffered_max,
                                std::chrono::milliseconds(100))) {
                            if (stop.load(std::memory_order_relaxed) ||
                                rx_thread_failed.load(std::memory_order_acquire)) {
                                break;
                            }
                            continue;
                        }
                    }
                    block.status = UHD_C_RX_OK;
                    block.samples.swap(buffered_batch);
                } else {
                    if (!sample_queue.pop_for(block, std::chrono::milliseconds(100))) {
                        if (stop.load(std::memory_order_relaxed) ||
                            rx_thread_failed.load(std::memory_order_acquire)) {
                            break;
                        }
                        continue;
                    }

                    const uint64_t queue_drops = sample_queue.dropped_blocks();
                    if (queue_drops != last_reported_queue_drops) {
                        const auto now = std::chrono::steady_clock::now();
                        if (queue_drops - last_reported_queue_drops >= 1024 ||
                            std::chrono::duration<double>(now - last_queue_drop_report).count() >= 1.0) {
                            std::cerr << "[TRX] sample queue overrun; dropped "
                                      << (queue_drops - last_reported_queue_drops)
                                      << " block(s), total=" << queue_drops << "\n";
                            last_reported_queue_drops = queue_drops;
                            last_queue_drop_report = now;
                        }
                    }
                }

                if (block.status == UHD_C_RX_OVERFLOW) {
                    std::cerr << "[TRX] overflow; dropping buffered samples\n";
                    rxbuf.clear();
                    rxbase = 0;
                    continue;
                }
                if (block.status != UHD_C_RX_OK) {
                    std::cerr << "[TRX] metadata error: status=" << block.status << "\n";
                    continue;
                }

                const auto t_sync0 = std::chrono::steady_clock::now();
                size_t frames_this_block = 0;
                ExtractedFrame frame;
#ifdef HAVE_GPU_FULL_PIPELINE
                ExtractedDeviceFrame device_frame;
#endif
                size_t sync_input_samples = block.samples.size();
                if (use_gpu_device_batch) {
#ifdef HAVE_GPU_FULL_PIPELINE
                    size_t gpu_base = 0;
                    sync_input_samples = gpu_batch_samples;
                    while (gpu_rx_sync->extract_device(
                        d_gpu_batch,
                        gpu_batch_samples,
                        gpu_base,
                        device_frame,
                        *device_frame_pool,
                        sync_state,
                        sync_stats)) {
                        device_frame_queue.push(std::move(device_frame));
                        ++frames_this_block;
                        device_frame = ExtractedDeviceFrame{};
                    }
                    if (gpu_base > 0) {
                        gpu_iq_buffer->consume_device_samples(gpu_base);
                    } else if (gpu_batch_samples > static_cast<size_t>(phy.frame_len())) {
                        gpu_iq_buffer->consume_device_samples(
                            gpu_batch_samples - static_cast<size_t>(phy.frame_len()));
                        sync_state = SyncState::Acquisition;
                    }
#endif
                } else {
                    rxbuf.insert(rxbuf.end(), block.samples.begin(), block.samples.end());

                    auto extract_one_frame = [&]() {
#ifdef HAVE_GPU_FULL_PIPELINE
                        if (gpu_rx_sync) {
                            return gpu_rx_sync->extract(rxbuf, rxbase, frame, sync_state, sync_stats);
                        }
#endif
                        const size_t acq_search = iq_buffer ? static_cast<size_t>(2 * phy.frame_len()) : 0;
                        return extract_rx_frame_vector(rxbuf, rxbase, phy, frame, sync_state, sync_stats, acq_search);
                    };

                    while (extract_one_frame()) {
                        frame_queue.push(std::move(frame));
                        ++frames_this_block;
                        frame = ExtractedFrame{};
                    }
                }
                const auto t_sync1 = std::chrono::steady_clock::now();
                if (opt.profile_pipeline) {
                    profile.sync_blocks.fetch_add(1, std::memory_order_relaxed);
                    profile.sync_input_samples.fetch_add(static_cast<uint64_t>(sync_input_samples), std::memory_order_relaxed);
                    profile.sync_frames.fetch_add(static_cast<uint64_t>(frames_this_block), std::memory_order_relaxed);
                    profile.sync_ns.fetch_add(elapsed_ns(t_sync0, t_sync1), std::memory_order_relaxed);
                }
                if (opt.profile_pipeline) {
                    const auto now = std::chrono::steady_clock::now();
                    const double report_elapsed = std::chrono::duration<double>(now - last_sync_report).count();
                    if (report_elapsed >= 1.0) {
                        const uint64_t found_delta = sync_stats.found - last_sync_found;
                        const uint64_t skipped_delta = sync_stats.skipped_samples - last_sync_skipped;
                        const double avg_start = sync_stats.found > 0
                            ? static_cast<double>(sync_stats.skipped_samples) / static_cast<double>(sync_stats.found)
                            : 0.0;
                        std::cout << "[SYNC] fps=" << (static_cast<double>(found_delta) / report_elapsed)
                                  << " state=" << (sync_state == SyncState::Tracking ? "tracking" : "acquisition")
                                  << " rxbuf_available=" << (rxbuf.size() >= rxbase ? rxbuf.size() - rxbase : 0)
                                  << " sample_queue_depth=" << ((iq_buffer
#ifdef HAVE_GPU_FULL_PIPELINE
                                      || gpu_iq_buffer
#endif
                                      ) ? 0 : sample_queue.size())
                                  << " iq_buffer_samples=" << (iq_buffer ? iq_buffer->size() : 0)
                                  << " iq_buffer_dropped=" << (iq_buffer ? iq_buffer->dropped_samples() : 0)
#ifdef HAVE_GPU_FULL_PIPELINE
                                  << " gpu_iq_buffer_samples=" << (gpu_iq_buffer ? gpu_iq_buffer->size() : 0)
                                  << " gpu_iq_buffer_dropped=" << (gpu_iq_buffer ? gpu_iq_buffer->dropped_samples() : 0)
#endif
                                  << " frame_queue_depth=" << frame_queue.size()
                                  << " frame_queue_dropped=" << frame_queue.dropped_frames()
#ifdef HAVE_GPU_FULL_PIPELINE
                                  << " device_frame_queue_depth=" << (use_device_frame_queue ? device_frame_queue.size() : 0)
                                  << " device_frame_queue_dropped=" << (use_device_frame_queue ? device_frame_queue.dropped_frames() : 0)
#endif
                                  << " found=" << sync_stats.found
                                  << " miss=" << sync_stats.miss
                                  << " trackingMiss=" << sync_stats.tracking_miss
                                  << " incomplete=" << sync_stats.incomplete
                                  << " skippedDelta=" << skipped_delta
                                  << " avgStart=" << avg_start
                                  << " maxStart=" << sync_stats.max_start
                                  << "\n";
                        last_sync_report = now;
                        last_sync_found = sync_stats.found;
                        last_sync_skipped = sync_stats.skipped_samples;
                        if (iq_buffer) {
                            const uint64_t drops = iq_buffer->dropped_samples();
                            if (drops != last_buffer_drops) {
                                std::cerr << "[TRX] IQ buffer dropped "
                                          << (drops - last_buffer_drops)
                                          << " sample(s), total=" << drops << "\n";
                                last_buffer_drops = drops;
                            }
                        }
#ifdef HAVE_GPU_FULL_PIPELINE
                        if (gpu_iq_buffer) {
                            const uint64_t drops = gpu_iq_buffer->dropped_samples();
                            if (drops != last_buffer_drops) {
                                std::cerr << "[TRX] GPU IQ buffer dropped "
                                          << (drops - last_buffer_drops)
                                          << " sample(s), total=" << drops << "\n";
                                last_buffer_drops = drops;
                            }
                        }
#endif
                    }
                }
            }
            frame_queue.close();
        } catch (...) {
            sync_thread_exception = std::current_exception();
            sync_thread_failed.store(true, std::memory_order_release);
            frame_queue.close();
        }
    });

    auto t0 = std::chrono::steady_clock::now();
    std::unique_ptr<AdaptiveRxDecision> trx_adaptive_decision;
    std::unique_ptr<usrp_link::PredictiveAmcPolicy> trx_predictive_amc;
    if (opt.adaptive && file_mode) {
        trx_adaptive_decision = std::make_unique<AdaptiveRxDecision>(opt);
        std::cout << "[ADAPT-TRX] enabled: repeat=" << opt.tx_repeat_min
                  << ".." << opt.tx_repeat_max << "\n";
        const bool use_predictive = opt.adaptive_predictive || !precomp_channel.empty();
        if (use_predictive) {
            trx_predictive_amc = std::make_unique<usrp_link::PredictiveAmcPolicy>(
                make_predictive_amc_config(opt));
            const auto pred = make_channel_prediction_frame(precomp_channel, opt, 0);
            const auto quality = evaluate_predictive_quality(
                *trx_predictive_amc,
                precomp_channel,
                opt,
                phy,
                pred,
                opt.adaptive_predicted_snr_db);
            const auto decision = trx_predictive_amc->decide(pred, quality, nullptr);
            adaptive_repeat.store(
                std::max(opt.tx_repeat_min, std::min(opt.tx_repeat_max, decision.repeat_count)),
                std::memory_order_release);
            {
                std::lock_guard<std::mutex> lock(adaptive_control_mutex);
                adaptive_control_decision = decision;
            }
            adaptive_control_snr_for_mcs.store(static_cast<float>(quality.snr_for_mcs_db), std::memory_order_release);
            adaptive_control_generation.fetch_add(1, std::memory_order_acq_rel);
            std::cout << "[ADAPT-TRX] predictive AMC enabled: predictedSNR="
                      << std::fixed << std::setprecision(1) << opt.adaptive_predicted_snr_db
                      << " confidence=" << opt.adaptive_prediction_confidence
                      << " ageMs=" << opt.adaptive_prediction_age_ms
                      << " evm=" << opt.adaptive_prediction_evm << "\n";
            print_predictive_decision("[ADAPT-TRX-PRED:init]", decision, quality);
        }
    }
    std::cout << "[TX-QUEUE] enabled capacity=" << tx_queue_capacity
              << " frames prefill=" << std::min<size_t>(tx_queue_capacity, 32)
              << " adaptiveSerialized=" << ((opt.adaptive && file_mode) ? "yes" : "no") << "\n";
    std::thread tx_producer_thread([&] {
        try {
            uint32_t frame_id = 0;
            uint64_t unique_frames = 0;
            uint64_t media_byte_offset = 0;
            int last_repeat_report = adaptive_repeat.load(std::memory_order_acquire);
            uint64_t last_control_generation = 0;
            std::mt19937 tx_noise_rng(static_cast<uint32_t>(opt.test_seed) ^ 0x71a6175eu);
            while (!stop.load(std::memory_order_relaxed)) {
                if (file_mode && tx_media->done_offset(media_byte_offset)) {
                    break;
                }

                std::vector<cf32> frame;
                bool control_frame = false;
                int control_repeat = 1;
                int queued_info_bits = info_bits;
                uint64_t next_media_byte_offset = media_byte_offset;
                int switch_mcs_after_control = tx_active_mcs.load(std::memory_order_acquire);
                if (file_mode) {
                    LinkRuntimeProfile* tx_profile = profile_bank
                        ? &profile_bank->get(static_cast<usrp_link::McsId>(
                              std::max(0, std::min(5, tx_active_mcs.load(std::memory_order_acquire)))))
                        : nullptr;
                    if (trx_predictive_amc &&
                        adaptive_control_generation.load(std::memory_order_acquire) != last_control_generation) {
                        last_control_generation = adaptive_control_generation.load(std::memory_order_acquire);
                        usrp_link::AmcDecision decision;
                        {
                            std::lock_guard<std::mutex> lock(adaptive_control_mutex);
                            decision = adaptive_control_decision;
                        }
                        const auto info = tx_media->build_control_info_bits(
                            frame_id,
                            decision,
                            frame_id + 1,
                            adaptive_control_snr_for_mcs.load(std::memory_order_acquire),
                            tx_profile ? tx_profile->info_bytes : info_bytes);
                        queued_info_bits = tx_profile
                            ? link_info_bits_per_frame(tx_profile->phy, tx_profile->code)
                            : info_bits;
#ifdef HAVE_GPU_FULL_PIPELINE
                        if (tx_profile && tx_profile->gpu_tx_baseband) {
                            frame = tx_profile->gpu_tx_baseband->build_media_frame(info);
                        } else if (gpu_tx_baseband) {
                            frame = gpu_tx_baseband->build_media_frame(info);
                        } else
#endif
                        {
                            frame = tx_profile ? tx_profile->tx_baseband->build_media_frame(info)
                                               : tx_baseband.build_media_frame(info);
                        }
                        control_frame = true;
                        switch_mcs_after_control = static_cast<int>(decision.mcs);
                        if (profile_bank) {
                            const int current_mcs = tx_active_mcs.load(std::memory_order_acquire);
                            control_repeat = switch_mcs_after_control > current_mcs
                                ? 1
                                : std::min(opt.tx_repeat_max, std::max(2, decision.repeat_count));
                        } else {
                            control_repeat = std::min(opt.tx_repeat_max, std::max(1, 2));
                        }
                        std::cout << "[CTRL-TRX-TX] frame=" << frame_id
                                  << " effectiveFrame=" << (frame_id + 1)
                                  << " mcs=" << static_cast<int>(decision.mcs)
                                  << " modulation=" << decision.modulation
                                  << " repeat=" << decision.repeat_count
                                  << "\n";
                    } else {
                        const size_t info_bytes_active = tx_profile ? tx_profile->info_bytes : info_bytes;
                        const size_t payload_bytes_active = tx_profile ? tx_profile->media_payload_bytes : payload_capacity;
                        const auto packet = tx_media->build_media_info_bits_from_offset(
                            frame_id,
                            media_byte_offset,
                            info_bytes_active,
                            payload_bytes_active);
                        const auto& info = packet.bits;
                        next_media_byte_offset = packet.next_offset;
                        queued_info_bits = tx_profile
                            ? link_info_bits_per_frame(tx_profile->phy, tx_profile->code)
                            : info_bits;
#ifdef HAVE_GPU_FULL_PIPELINE
                        if (tx_profile && tx_profile->gpu_tx_baseband) {
                            frame = tx_profile->gpu_tx_baseband->build_media_frame(info);
                        } else if (gpu_tx_baseband) {
                            frame = gpu_tx_baseband->build_media_frame(info);
                        } else
#endif
                        {
                            frame = tx_profile ? tx_profile->tx_baseband->build_media_frame(info)
                                               : tx_baseband.build_media_frame(info);
                        }
                    }
                } else {
#ifdef HAVE_GPU_FULL_PIPELINE
                    if (gpu_tx_baseband) {
                        frame = gpu_tx_baseband->build_test_frame(frame_id);
                    } else
#endif
                    {
                        frame = tx_baseband.build_test_frame(frame_id);
                    }
                }

                const int repeat = control_frame
                    ? control_repeat
                    : opt.adaptive && file_mode
                    ? adaptive_repeat.load(std::memory_order_acquire)
                    : 1;
                if (repeat != last_repeat_report) {
                    std::cout << "[TRX-TX] adaptive repeat=" << repeat << "\n";
                    last_repeat_report = repeat;
                }
                const uint64_t unique_after_frame = unique_frames + (control_frame ? 0 : 1);
                for (int rep = 0; rep < repeat && !stop.load(std::memory_order_relaxed); ++rep) {
                    std::vector<cf32> noisy_frame;
                    if (std::isfinite(opt.tx_awgn_snr_db)) {
                        noisy_frame = frame;
                        add_tx_awgn_if_requested(noisy_frame, phy, opt, tx_noise_rng, opt.tx_awgn_snr_db);
                    }
                    TxQueuedFrame queued;
                    if (opt.tx_host_format == "sc16") {
                        queued.packed_sc16_samples = quantize_tx_sc16(
                            std::isfinite(opt.tx_awgn_snr_db) ? noisy_frame : frame);
                    } else if (std::isfinite(opt.tx_awgn_snr_db)) {
                        queued.samples = std::move(noisy_frame);
                    } else if (rep == repeat - 1) {
                        queued.samples = std::move(frame);
                    } else {
                        queued.samples = frame;
                    }
                    queued.frame_id = frame_id;
                    queued.unique_frames = unique_after_frame;
                    queued.info_bits_per_frame = queued_info_bits;
                    queued.repeat = repeat;
                    queued.activate_mcs_after_send =
                        (control_frame && rep == repeat - 1) ? switch_mcs_after_control : -1;
                    queued.counts_unique_payload = !control_frame && rep == repeat - 1;
                    if (!tx_frame_queue.push(std::move(queued), stop)) {
                        break;
                    }
                }
                if (control_frame && profile_bank && switch_mcs_after_control >= 0) {
                    tx_active_mcs.store(switch_mcs_after_control, std::memory_order_release);
                    std::cout << "[PROFILE-BANK-TX-PREP] activeMcs=" << switch_mcs_after_control
                              << " queuedAfterControlFrame=" << (frame_id + 1) << "\n";
                }

                ++frame_id;
                if (!control_frame) {
                    ++unique_frames;
                    media_byte_offset = next_media_byte_offset;
                }
                const auto now = std::chrono::steady_clock::now();
                const double elapsed = std::chrono::duration<double>(now - t0).count();
                if (opt.frames > 0 && static_cast<int>(unique_frames) >= opt.frames) {
                    break;
                }
                if (opt.duration_sec > 0.0 && elapsed >= opt.duration_sec) {
                    break;
                }
            }
            tx_frame_queue.close();
        } catch (...) {
            tx_producer_exception = std::current_exception();
            tx_thread_failed.store(true, std::memory_order_release);
            tx_frame_queue.close();
        }
    });

    std::thread tx_thread([&] {
        try {
            usrp_link::set_realtime_priority();
            bool first_packet = true;
            uint64_t sent_frames = 0;
            uint64_t unique_info_bits_sent = 0;
            const size_t prefill = std::min<size_t>(tx_queue_capacity, 32);
            tx_frame_queue.wait_for_prefill(prefill, stop);
            TxQueuedFrame queued;
            while (!stop.load(std::memory_order_relaxed)) {
                if (!tx_frame_queue.pop_for(queued, std::chrono::milliseconds(20))) {
                    if (tx_frame_queue.closed_and_empty()) {
                        break;
                    }
                    continue;
                }
                if (opt.tx_host_format == "sc16") {
                    send_tx_packed_sc16_frame(radio, queued.packed_sc16_samples, first_packet, "[TRX-TX]");
                } else {
                    send_tx_frame(radio, queued.samples, opt.tx_host_format, first_packet, "[TRX-TX]");
                }
                ++sent_frames;
                if (queued.counts_unique_payload) {
                    unique_info_bits_sent += static_cast<uint64_t>(std::max(queued.info_bits_per_frame, 0));
                }
                if (queued.activate_mcs_after_send >= 0) {
                    tx_active_mcs.store(queued.activate_mcs_after_send, std::memory_order_release);
                    std::cout << "[PROFILE-BANK-TX] activeMcs=" << queued.activate_mcs_after_send
                              << " fromFrame=" << (queued.frame_id + 1) << "\n";
                }
                if (opt.profile_pipeline) {
                    profile.tx_frames.store(sent_frames, std::memory_order_relaxed);
                }
                if (sent_frames % static_cast<uint64_t>(std::max(opt.report_every, 1)) == 0) {
                    const auto now = std::chrono::steady_clock::now();
                    const double elapsed = std::chrono::duration<double>(now - t0).count();
                    std::cout << "[TRX-TX] frame=" << queued.frame_id
                              << " sent=" << sent_frames
                              << " unique=" << queued.unique_frames
                              << " repeat=" << queued.repeat
                              << " queueDepth=" << tx_frame_queue.size()
                              << " info=" << (static_cast<double>(unique_info_bits_sent) / std::max(elapsed, 1e-9) / 1e6)
                              << " Mbps elapsed=" << std::fixed << std::setprecision(2) << elapsed << " s\n";
                }
            }
            try {
                radio.send_end_of_burst(1.0);
            } catch (const std::exception& e) {
                std::cerr << "[TRX-TX] " << e.what() << "\n";
            }
            tx_done.store(true, std::memory_order_release);
        } catch (...) {
            tx_thread_exception = std::current_exception();
            tx_thread_failed.store(true, std::memory_order_release);
            tx_done.store(true, std::memory_order_release);
            tx_frame_queue.close();
        }
    });

    RxStats stats;
    LinkControlState control_state;
    bool stop_now = false;
    uint64_t last_reported_frame_drops = 0;
    uint64_t last_reported_device_frame_drops = 0;
#ifdef HAVE_GPU_FULL_PIPELINE
    const bool trx_async_bp_osd =
        use_device_frame_queue &&
        !profile_bank &&
        cuda_decoder != nullptr &&
        !cuda_decoder->osd_only &&
        cuda_bp_decoder == nullptr;
    const int trx_blocks_per_frame = phy.coded_bits_per_frame() / code.n;
    const int trx_async_max_frames = trx_async_bp_osd
        ? std::max(1, cuda_decoder->async_max_codewords() / std::max(trx_blocks_per_frame, 1))
        : 1;
    const int trx_async_window_frames = trx_async_bp_osd
        ? std::max(1, std::min(16, trx_async_max_frames))
        : 1;
    struct TrxAsyncGpuFrame {
        double peak = 0.0;
        double cfo_hz = 0.0;
        uint64_t demod_ns = 0;
        uint64_t fec_ns = 0;
        int used_osd_count = 0;
        int ready_blocks = 0;
        std::vector<uint8_t> info_bits;
        std::vector<uint8_t> block_ready;
        std::chrono::steady_clock::time_point submit_time;
    };
    std::map<uint64_t, TrxAsyncGpuFrame> trx_async_pending;
    std::deque<FrameDecodeResult> trx_async_ready_frames;
    if (trx_async_bp_osd) {
        std::cout << "[TRX-RX] BP-OSD cross-frame batching enabled: blocks/frame="
                  << trx_blocks_per_frame
                  << " windowFrames=" << trx_async_window_frames
                  << " maxBatchCodewords=" << cuda_decoder->async_max_codewords()
                  << "\n";
    }
    auto finish_trx_async_frame = [&](TrxAsyncGpuFrame&& pending) {
        const uint64_t fec_ns = pending.fec_ns != 0
            ? pending.fec_ns
            : elapsed_ns(pending.submit_time, std::chrono::steady_clock::now());
        trx_async_ready_frames.push_back(finish_decode_from_info_bits(
            std::move(pending.info_bits),
            phy,
            code,
            pending.used_osd_count,
            true,
            !file_mode && !opt.skip_reference_ber,
            pending.peak,
            pending.cfo_hz,
            0.0,
            pending.demod_ns,
            fec_ns,
            opt.profile_pipeline ? &profile.decode : nullptr));
    };
    auto drain_trx_async_decoder = [&](bool submit_staged) {
        if (!trx_async_bp_osd) {
            return;
        }
        if (submit_staged) {
            const int submitted_codewords = cuda_decoder->async_pending_codewords();
            const int submitted_frames = submitted_codewords / std::max(trx_blocks_per_frame, 1);
            const uint64_t submitted_ns = cuda_decoder->submit_async_batch();
            if (submitted_frames > 0 && submitted_ns > 0) {
                const uint64_t per_frame_ns = submitted_ns / static_cast<uint64_t>(submitted_frames);
                int tagged = 0;
                for (auto& kv : trx_async_pending) {
                    if (kv.second.fec_ns == 0) {
                        kv.second.fec_ns = per_frame_ns;
                        if (++tagged >= submitted_frames) {
                            break;
                        }
                    }
                }
            }
        }
        while (true) {
            auto ready = cuda_decoder->poll_ready_codewords(cuda_decoder->async_max_codewords());
            if (ready.empty()) {
                break;
            }
            for (const auto& cw : ready) {
                auto it = trx_async_pending.find(cw.frame_id);
                if (it == trx_async_pending.end()) {
                    throw std::runtime_error("TRX CUDA BP-OSD async decoder returned an unknown frame id");
                }
                TrxAsyncGpuFrame& pending = it->second;
                if (cw.block_index >= static_cast<uint32_t>(trx_blocks_per_frame)) {
                    throw std::runtime_error("TRX CUDA BP-OSD async decoder returned an invalid block index");
                }
                if (pending.block_ready[static_cast<size_t>(cw.block_index)] == 0) {
                    std::copy(
                        cw.info_bits.begin(),
                        cw.info_bits.end(),
                        pending.info_bits.begin() + static_cast<std::ptrdiff_t>(cw.block_index * code.k));
                    pending.block_ready[static_cast<size_t>(cw.block_index)] = 1;
                    pending.used_osd_count += cw.used_osd ? 1 : 0;
                    ++pending.ready_blocks;
                }
                if (pending.ready_blocks == trx_blocks_per_frame) {
                    TrxAsyncGpuFrame complete = std::move(pending);
                    trx_async_pending.erase(it);
                    finish_trx_async_frame(std::move(complete));
                }
            }
        }
    };
#endif
    while (true) {
        auto should_stop_after_empty_pop = [&]() {
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - t0).count();
            if (opt.duration_sec > 0.0 && elapsed >= opt.duration_sec) {
                return true;
            }
            if ((tx_done.load(std::memory_order_acquire) && opt.duration_sec <= 0.0) ||
                rx_thread_failed.load(std::memory_order_acquire) ||
                sync_thread_failed.load(std::memory_order_acquire) ||
                tx_thread_failed.load(std::memory_order_acquire)) {
                return true;
            }
            return false;
        };

        FrameDecodeResult r;
        LinkRuntimeProfile* rx_profile = profile_bank
            ? &profile_bank->get(static_cast<usrp_link::McsId>(
                  std::max(0, std::min(5, rx_active_mcs.load(std::memory_order_acquire)))))
            : nullptr;
        const PhyConfig& active_phy = rx_profile ? rx_profile->phy : phy;
        const LdpcCode& active_code = rx_profile ? rx_profile->code : code;
        const int active_info_bits = link_info_bits_per_frame(active_phy, active_code);
        bool have_prefetched_async_frame = false;
#ifdef HAVE_GPU_FULL_PIPELINE
        if (!trx_async_ready_frames.empty()) {
            r = std::move(trx_async_ready_frames.front());
            trx_async_ready_frames.pop_front();
            have_prefetched_async_frame = true;
        }
        if (!have_prefetched_async_frame)
#endif
        {
#ifdef HAVE_GPU_FULL_PIPELINE
        if (use_device_frame_queue) {
            ExtractedDeviceFrame device_frame;
            if (!device_frame_queue.pop_for(device_frame, std::chrono::milliseconds(100))) {
#ifdef HAVE_GPU_FULL_PIPELINE
                if (trx_async_bp_osd && !trx_async_pending.empty()) {
                    drain_trx_async_decoder(true);
                    if (!trx_async_ready_frames.empty()) {
                        continue;
                    }
                }
#endif
                if (should_stop_after_empty_pop()) {
                    break;
                }
                continue;
            }
            const uint64_t frame_drops = device_frame_queue.dropped_frames();
            if (frame_drops != last_reported_device_frame_drops) {
                std::cerr << "[TRX] device frame queue overrun; dropped "
                          << (frame_drops - last_reported_device_frame_drops)
                          << " frame(s), total=" << frame_drops << "\n";
                last_reported_device_frame_drops = frame_drops;
            }
            GpuRxDemodChain* active_gpu_rx_demod = rx_profile && rx_profile->gpu_rx_demod
                ? rx_profile->gpu_rx_demod.get()
                : gpu_rx_demod.get();
            if (trx_async_bp_osd) {
                if (cuda_decoder->async_pending_codewords() + trx_blocks_per_frame >
                    cuda_decoder->async_max_codewords()) {
                    drain_trx_async_decoder(true);
                }
                const auto llr_frame = active_gpu_rx_demod->demod_device_to_device_llr(device_frame);
                const uint64_t decoder_frame_id =
                    cuda_decoder->enqueue_device_frame_async(llr_frame.device_llrs, llr_frame.blocks);
                TrxAsyncGpuFrame pending;
                pending.peak = llr_frame.peak;
                pending.cfo_hz = llr_frame.cfo_hz;
                pending.demod_ns = llr_frame.demod_ns;
                pending.info_bits.assign(static_cast<size_t>(trx_blocks_per_frame * code.k), 0);
                pending.block_ready.assign(static_cast<size_t>(trx_blocks_per_frame), 0);
                pending.submit_time = std::chrono::steady_clock::now();
                trx_async_pending.emplace(decoder_frame_id, std::move(pending));
                if (static_cast<int>(trx_async_pending.size()) >= trx_async_window_frames) {
                    drain_trx_async_decoder(true);
                }
                if (trx_async_ready_frames.empty()) {
                    continue;
                }
                r = std::move(trx_async_ready_frames.front());
                trx_async_ready_frames.pop_front();
            } else {
                r = active_gpu_rx_demod->process_device(
                    device_frame,
                    rx_profile ? rx_profile->bp_decoder() : cuda_bp_decoder.get(),
                    rx_profile ? rx_profile->bp_osd_decoder() : cuda_decoder.get(),
                    opt.ldpc_max_iter,
                    opt.ldpc_normalization,
                    !file_mode && !opt.skip_reference_ber,
                    opt.profile_pipeline ? &profile.decode : nullptr);
            }
        } else
#endif
        {
            ExtractedFrame frame;
            if (!frame_queue.pop_for(frame, std::chrono::milliseconds(100))) {
                if (should_stop_after_empty_pop()) {
                    break;
                }
                continue;
            }

            const uint64_t frame_drops = frame_queue.dropped_frames();
            if (frame_drops != last_reported_frame_drops) {
                std::cerr << "[TRX] frame queue overrun; dropped "
                          << (frame_drops - last_reported_frame_drops)
                          << " frame(s), total=" << frame_drops << "\n";
                last_reported_frame_drops = frame_drops;
            }

#ifdef HAVE_GPU_FULL_PIPELINE
            GpuRxDemodChain* active_gpu_rx_demod = rx_profile && rx_profile->gpu_rx_demod
                ? rx_profile->gpu_rx_demod.get()
                : gpu_rx_demod.get();
            if (active_gpu_rx_demod) {
                r = active_gpu_rx_demod->process(
                    frame,
                    rx_profile ? rx_profile->bp_decoder() : cuda_bp_decoder.get(),
                    rx_profile ? rx_profile->bp_osd_decoder() : cuda_decoder.get(),
                    opt.ldpc_max_iter,
                    opt.ldpc_normalization,
                    !file_mode && !opt.skip_reference_ber,
                    opt.profile_pipeline ? &profile.decode : nullptr);
            } else
#endif
            {
                r = decode_frame(
                    frame.samples,
                    active_phy,
                    active_code,
                    rx_profile ? rx_profile->bp_decoder() : cuda_bp_decoder.get(),
                    rx_profile ? rx_profile->bp_osd_decoder() : cuda_decoder.get(),
                    frame.cfo_hz,
                    frame.peak,
                    opt.ldpc_max_iter,
                    opt.ldpc_normalization,
                    opt.rx_snr_gate_db,
                    !file_mode && !opt.skip_reference_ber,
                    opt.profile_pipeline ? &profile.decode : nullptr);
            }
        }
        }
        if (!file_mode && opt.skip_reference_ber) {
            apply_parity_only_test_result(r);
        }
        if (file_mode) {
            const auto media_t0 = std::chrono::steady_clock::now();
            const uint64_t controls_before = control_state.accepted_controls;
            const size_t active_payload_capacity = rx_profile ? rx_profile->media_payload_bytes : payload_capacity;
            const bool media_ok = accept_media_decode(r, *rx_media, active_payload_capacity, &control_state);
            if (control_state.accepted_controls != controls_before) {
                rx_active_mcs.store(
                    static_cast<int>(control_state.active_decision.mcs),
                    std::memory_order_release);
                std::cout << "[PROFILE-BANK-RX] activeMcs="
                          << static_cast<int>(control_state.active_decision.mcs)
                          << " effectiveFrame=" << control_state.active_effective_frame_id
                          << "\n";
            }
            const auto media_t1 = std::chrono::steady_clock::now();
            if (opt.profile_pipeline) {
                profile.media_frames.fetch_add(1, std::memory_order_relaxed);
                if (media_ok) {
                    profile.media_accepts.fetch_add(1, std::memory_order_relaxed);
                }
                profile.media_ns.fetch_add(
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(media_t1 - media_t0).count()),
                    std::memory_order_relaxed);
            }
        }
        update_stats(stats, r, active_info_bits, active_phy.coded_bits_per_frame());
        if (trx_adaptive_decision) {
            int repeat = 1;
            double avg_snr = 0.0;
            double fer = 0.0;
            int frames = 0;
            if (trx_adaptive_decision->observe(r, repeat, avg_snr, fer, frames)) {
                if (trx_predictive_amc) {
                    const auto pred = make_channel_prediction_frame(
                        precomp_channel,
                        opt,
                        stats.detected);
                    const auto quality = evaluate_predictive_quality(
                        *trx_predictive_amc,
                        precomp_channel,
                        opt,
                        phy,
                        pred,
                        opt.adaptive_predicted_snr_db);
                    usrp_link::RxFeedbackWindow fb;
                    fb.frames = frames;
                    fb.ok_frames = frames - static_cast<int>(std::lround(fer * static_cast<double>(frames)));
                    fb.fer = fer;
                    fb.avg_snr_db = avg_snr;
                    fb.overflow = false;
                    const auto decision = trx_predictive_amc->decide(pred, quality, &fb);
                    repeat = decision.repeat_count;
                    {
                        std::lock_guard<std::mutex> lock(adaptive_control_mutex);
                        if (static_cast<int>(adaptive_control_decision.mcs) != static_cast<int>(decision.mcs) ||
                            adaptive_control_decision.repeat_count != decision.repeat_count ||
                            adaptive_control_decision.modulation != decision.modulation ||
                            adaptive_control_decision.fec_profile != decision.fec_profile) {
                            adaptive_control_decision = decision;
                            adaptive_control_snr_for_mcs.store(static_cast<float>(quality.snr_for_mcs_db), std::memory_order_release);
                            adaptive_control_generation.fetch_add(1, std::memory_order_acq_rel);
                        }
                    }
                    print_predictive_decision("[ADAPT-TRX-PRED]", decision, quality);
                }
                repeat = std::max(opt.tx_repeat_min, std::min(opt.tx_repeat_max, repeat));
                const int old = adaptive_repeat.exchange(repeat, std::memory_order_acq_rel);
                if (repeat != old) {
                    std::cout << "[ADAPT-TRX] repeat " << old << " -> " << repeat
                              << " window=" << frames
                              << " avgSNR=" << std::fixed << std::setprecision(1) << avg_snr
                              << " FER=" << std::scientific << fer << "\n";
                }
            }
        }
        {
            const auto now = std::chrono::steady_clock::now();
            send_ui_metrics_if_due(
                stats,
                std::chrono::duration<double>(now - t0).count(),
                active_info_bits,
                r,
                rx_active_mcs.load(std::memory_order_acquire),
                opt.frames);
        }
        if (opt.verbose ||
            stats.detected % static_cast<uint64_t>(std::max(opt.report_every, 1)) == 0 ||
            (!opt.suppress_error_frames && !r.frame_ok)) {
            print_frame_result("[TRX-RX]", r);
        }
        if (stats.detected % static_cast<uint64_t>(std::max(opt.report_every, 1)) == 0) {
            const auto now = std::chrono::steady_clock::now();
            print_stats(stats, std::chrono::duration<double>(now - t0).count(), active_info_bits);
        }
        if (file_mode && rx_media->complete && !opt.loop_file) {
            stop_now = true;
        }
        if (opt.frames > 0 && static_cast<int>(stats.detected) >= opt.frames) {
            stop_now = true;
        }

        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - t0).count();
        if (stop_now) {
            break;
        }
        if (opt.duration_sec > 0.0 && elapsed >= opt.duration_sec) {
            break;
        }
    }

    stop.store(true, std::memory_order_relaxed);
    tx_frame_queue.close();
    if (iq_buffer) {
        iq_buffer->close();
    }
#ifdef HAVE_GPU_FULL_PIPELINE
    if (gpu_iq_buffer) {
        gpu_iq_buffer->close();
    }
#endif
    sample_queue.close();
    frame_queue.close();
    try {
        radio.stop_rx();
    } catch (const std::exception& e) {
        std::cerr << "[TRX-RX] " << e.what() << "\n";
    }
    if (tx_thread.joinable()) {
        tx_thread.join();
    }
    if (tx_producer_thread.joinable()) {
        tx_producer_thread.join();
    }
    if (rx_thread.joinable()) {
        rx_thread.join();
    }
    if (sync_thread.joinable()) {
        sync_thread.join();
    }
    if (tx_thread_exception) {
        std::rethrow_exception(tx_thread_exception);
    }
    if (tx_producer_exception) {
        std::rethrow_exception(tx_producer_exception);
    }
    if (rx_thread_exception) {
        std::rethrow_exception(rx_thread_exception);
    }
    if (sync_thread_exception) {
        std::rethrow_exception(sync_thread_exception);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    print_stats(stats, elapsed, info_bits);
    if (opt.profile_pipeline) {
        print_pipeline_profile(profile, elapsed, phy, info_bits);
    }
    return 0;
}

static int run_replay(const Options& opt, const PhyConfig& phy, const LdpcCode& code, const fec::RxDecoderPlan& rx_plan)
{
    if (opt.iq_input_file.empty()) {
        throw std::runtime_error("--mode replay requires --iq-input <path>");
    }
    const bool file_mode = opt.traffic_mode == "file";
    const int info_bits = link_info_bits_per_frame(phy, code);
    const size_t payload_capacity = file_mode ? media_payload_capacity_bytes(phy, code) : 0;

    std::ifstream iq_in(opt.iq_input_file, std::ios::binary | std::ios::ate);
    if (!iq_in) {
        throw std::runtime_error("failed to open IQ replay input: " + opt.iq_input_file);
    }
    const auto bytes = iq_in.tellg();
    if (bytes < 0 || (static_cast<uint64_t>(bytes) % sizeof(cf32)) != 0) {
        throw std::runtime_error("IQ replay input size is not a whole number of cf32 samples");
    }
    const size_t samples = static_cast<size_t>(static_cast<uint64_t>(bytes) / sizeof(cf32));
    iq_in.seekg(0, std::ios::beg);
    std::vector<cf32> rxbuf(samples);
    if (!rxbuf.empty()) {
        iq_in.read(reinterpret_cast<char*>(rxbuf.data()), static_cast<std::streamsize>(rxbuf.size() * sizeof(cf32)));
        if (!iq_in) {
            throw std::runtime_error("failed while reading IQ replay input");
        }
    }
    std::cout << "[REPLAY] input=" << opt.iq_input_file
              << " samples=" << rxbuf.size()
              << " seconds=" << (static_cast<double>(rxbuf.size()) / std::max(phy.rate, 1.0))
              << "\n";

    std::unique_ptr<CudaBpDecoder> cuda_bp_decoder;
    std::unique_ptr<CudaBpOsdDecoder> cuda_decoder;
    if (rx_plan.kind == fec::RxDecoderKind::GpuBp) {
        cuda_bp_decoder = std::make_unique<CudaBpDecoder>(opt, code, *rx_plan.fec);
    } else if (rx_plan.kind == fec::RxDecoderKind::GpuBpOsd) {
        cuda_decoder = std::make_unique<CudaBpOsdDecoder>(opt, code);
    }

    std::unique_ptr<MediaReassembler> media;
    if (file_mode) {
        media = std::make_unique<MediaReassembler>(default_output_file(opt));
        std::cout << "[MEDIA] REPLAY output=" << media->output_path
                  << " payload/frame=" << payload_capacity << "\n";
    }

#ifdef HAVE_GPU_FULL_PIPELINE
    std::unique_ptr<GpuRxDemodChain> gpu_rx_demod;
    if (opt.gpu_rx_demod) {
        gpu_rx_demod = std::make_unique<GpuRxDemodChain>(phy, code, nullptr);
    }
#endif

    RxStats stats;
    DecodeProfile decode_profile;
    size_t rxbase = 0;
    SyncState sync_state = SyncState::Acquisition;
    SyncStats sync_stats;
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t extracted = 0;

    while (true) {
        ExtractedFrame frame;
        if (!extract_rx_frame_vector(rxbuf, rxbase, phy, frame, sync_state, sync_stats)) {
            break;
        }
        ++extracted;

        FrameDecodeResult r;
#ifdef HAVE_GPU_FULL_PIPELINE
        if (gpu_rx_demod) {
            r = gpu_rx_demod->process(
                frame,
                cuda_bp_decoder.get(),
                cuda_decoder.get(),
                opt.ldpc_max_iter,
                opt.ldpc_normalization,
                !file_mode,
                opt.profile_pipeline ? &decode_profile : nullptr);
        } else
#endif
        {
            r = decode_frame(
                frame.samples,
                phy,
                code,
                cuda_bp_decoder.get(),
                cuda_decoder.get(),
                frame.cfo_hz,
                frame.peak,
                opt.ldpc_max_iter,
                opt.ldpc_normalization,
                opt.rx_snr_gate_db,
                !file_mode,
                opt.profile_pipeline ? &decode_profile : nullptr);
        }

        if (file_mode) {
            accept_media_decode(r, *media, payload_capacity);
        }
        update_stats(stats, r, info_bits, phy.coded_bits_per_frame());
        if (opt.verbose ||
            stats.detected % static_cast<uint64_t>(std::max(opt.report_every, 1)) == 0 ||
            (!opt.suppress_error_frames && !r.frame_ok)) {
            print_frame_result("[REPLAY]", r);
        }
        if (stats.detected % static_cast<uint64_t>(std::max(opt.report_every, 1)) == 0) {
            const auto now = std::chrono::steady_clock::now();
            print_stats(stats, std::chrono::duration<double>(now - t0).count(), info_bits);
        }
        if (file_mode && media->complete && !opt.loop_file) {
            break;
        }
        if (opt.frames > 0 && static_cast<int>(stats.detected) >= opt.frames) {
            break;
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[REPLAY] sync_stats: found=" << sync_stats.found
              << " miss=" << sync_stats.miss
              << " trackingMiss=" << sync_stats.tracking_miss
              << " incomplete=" << sync_stats.incomplete
              << " skippedSamples=" << sync_stats.skipped_samples
              << " maxStart=" << sync_stats.max_start
              << " extracted=" << extracted
              << "\n";
    print_stats(stats, elapsed, info_bits);
    if (opt.profile_pipeline) {
        print_bench_decode_substage("llr", decode_profile.llr_ns, decode_profile.frames, info_bits);
        print_bench_decode_substage("fec", decode_profile.fec_ns, decode_profile.frames, info_bits);
        print_bench_decode_substage("reference_ber", decode_profile.reference_ns, decode_profile.frames, info_bits);
    }
    return (stats.detected > 0 && stats.err == 0) || (file_mode && media && media->complete) ? 0 : 2;
}

static int run_sim(const Options& opt, const PhyConfig& phy, const LdpcCode& code, const fec::RxDecoderPlan& rx_plan)
{
    const bool file_mode = opt.traffic_mode == "file";
    const int info_bits = link_info_bits_per_frame(phy, code);
    const size_t info_bytes = link_info_bytes_per_frame(phy, code);
    const size_t payload_capacity = file_mode ? media_payload_capacity_bytes(phy, code) : 0;
    std::unique_ptr<CudaBpDecoder> cuda_bp_decoder;
    std::unique_ptr<CudaBpOsdDecoder> cuda_decoder;
    if (rx_plan.kind == fec::RxDecoderKind::GpuBp) {
        cuda_bp_decoder = std::make_unique<CudaBpDecoder>(opt, code, *rx_plan.fec);
    } else if (rx_plan.kind == fec::RxDecoderKind::GpuBpOsd) {
        cuda_decoder = std::make_unique<CudaBpOsdDecoder>(opt, code);
    }
    std::unique_ptr<MediaPacketizer> tx_media;
    std::unique_ptr<MediaReassembler> rx_media;
    if (file_mode) {
        if (opt.input_file.empty()) {
            throw std::runtime_error("--traffic file requires --input <path> in sim mode");
        }
        tx_media = std::make_unique<MediaPacketizer>(opt.input_file, info_bytes, opt.loop_file);
        rx_media = std::make_unique<MediaReassembler>(default_output_file(opt));
        std::cout << "[MEDIA] SIM input=" << opt.input_file
                  << " output=" << rx_media->output_path
                  << " bytes=" << tx_media->file_bytes.size()
                  << " chunks=" << tx_media->total_chunks
                  << " payload/frame=" << tx_media->payload_bytes_per_frame << "\n";
    }

    InjectionChannel precomp_channel;
    InjectionChannel actual_channel;
    if (!opt.precomp_channel_files.empty()) {
        precomp_channel = load_injection_channel(opt.precomp_channel_files);
        print_injection_channel("sim-precomp", precomp_channel);
    }
    if (!opt.actual_channel_files.empty()) {
        actual_channel = load_injection_channel(opt.actual_channel_files);
        print_injection_channel("sim-actual", actual_channel);
    }

    const int frames = opt.frames > 0
        ? opt.frames
        : (file_mode ? static_cast<int>(tx_media->total_chunks) : 4);
    std::deque<cf32> rxbuf;
    RxStats stats;
    auto t0 = std::chrono::steady_clock::now();
    uint64_t sent_frames = 0;
    uint32_t expected_test_frame_id = 0;
    uint64_t gpu_tx_ns = 0;
    uint64_t gpu_tx_frames = 0;
    std::mt19937 noise_rng(static_cast<uint32_t>(opt.test_seed) ^ 0x5eed1234u);
    std::normal_distribution<float> noise_dist(0.0f, 1.0f);

    auto add_awgn_if_requested = [&](std::vector<cf32>& samples) -> std::optional<double> {
        if (!std::isfinite(opt.sim_snr_db) || samples.empty()) {
            return std::nullopt;
        }
        if (opt.tx_awgn_scope == "data-subcarriers") {
            return add_awgn_to_data_subcarriers(samples, phy, opt.sim_snr_db, noise_rng);
        }
        const double frequency_signal_power = std::max(active_subcarrier_power(samples, phy), 1e-12);
        const double frequency_noise_var = frequency_signal_power / std::pow(10.0, opt.sim_snr_db / 10.0);
        const double time_noise_var = frequency_noise_var / static_cast<double>(phy.nfft);
        const float sigma = static_cast<float>(std::sqrt(std::max(time_noise_var, 0.0) * 0.5));
        for (auto& s : samples) {
            s += cf32(sigma * noise_dist(noise_rng), sigma * noise_dist(noise_rng));
        }
        return frequency_noise_var;
    };

    for (int i = 0; i < frames; ++i) {
        if (file_mode && tx_media->done(sent_frames)) {
            break;
        }

        std::vector<cf32> tx;
        if (file_mode) {
            const auto info = tx_media->build_info_bits(static_cast<uint32_t>(i));
            tx = build_tx_frame_from_info(phy, code, info);
        } else {
            tx = build_tx_frame(phy, code, static_cast<uint32_t>(i));
        }
        if (!precomp_channel.empty()) {
            apply_ofdm_channel_transform(tx, phy, precomp_channel, true);
        }
        if (!actual_channel.empty()) {
            apply_ofdm_channel_transform(tx, phy, actual_channel, false);
        }
        ++sent_frames;
        const auto known_frequency_noise_var = add_awgn_if_requested(tx);
        for (const auto& x : tx) {
            rxbuf.push_back(x);
        }

        FrameDecodeResult r;
        while (true) {
            const std::optional<uint32_t> expected_ref =
                file_mode ? std::nullopt : std::optional<uint32_t>(expected_test_frame_id);
            if (!process_rx_buffer(
                    rxbuf,
                    phy,
                    code,
                    cuda_bp_decoder.get(),
                    cuda_decoder.get(),
                    opt.ldpc_max_iter,
                    opt.ldpc_normalization,
                    !file_mode,
                    r,
                    expected_ref,
                    known_frequency_noise_var)) {
                break;
            }
            if (!file_mode) {
                ++expected_test_frame_id;
            }
            if (file_mode) {
                accept_media_decode(r, *rx_media, payload_capacity);
            }
            update_stats(stats, r, info_bits, phy.coded_bits_per_frame());
            {
                const auto now = std::chrono::steady_clock::now();
                send_ui_metrics_if_due(stats, std::chrono::duration<double>(now - t0).count(), info_bits, r);
            }
            print_frame_result("[SIM]", r);
        }
        if (file_mode && rx_media->complete && !opt.loop_file) {
            break;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    print_stats(stats, std::chrono::duration<double>(t1 - t0).count(), info_bits);

    if (stats.detected != sent_frames || stats.err != 0) {
        return 2;
    }
    if (file_mode && opt.frames <= 0 && !opt.loop_file && !rx_media->complete) {
        return 2;
    }
    return 0;
}

#ifdef HAVE_GPU_FULL_PIPELINE
struct SimSnrSegment {
    double snr_db = std::numeric_limits<double>::infinity();
    int frames = 0;
};

static std::string trim_copy(std::string s)
{
    auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [&](char c) { return !is_space(static_cast<unsigned char>(c)); }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [&](char c) { return !is_space(static_cast<unsigned char>(c)); }).base(), s.end());
    return s;
}

static std::vector<SimSnrSegment> parse_sim_snr_segments(const Options& opt)
{
    std::string spec = trim_copy(opt.sim_snr_trace);
    std::string lower = spec;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (spec.empty()) {
        return {{opt.sim_snr_db, 0}};
    }
    if (lower == "demo" || lower == "amc-demo" || lower == "dynamic-demo") {
        return {
            {8.0, 200},
            {14.5, 200},
            {18.0, 200},
            {10.0, 200},
            {15.0, 200},
            {7.0, 200},
        };
    }

    std::vector<SimSnrSegment> segments;
    std::stringstream ss(spec);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = trim_copy(token);
        if (token.empty()) {
            continue;
        }
        const size_t colon = token.find(':');
        if (colon == std::string::npos) {
            segments.push_back({std::stod(token), 1});
        } else {
            const double snr_db = std::stod(trim_copy(token.substr(0, colon)));
            const int frames = std::stoi(trim_copy(token.substr(colon + 1)));
            if (frames <= 0) {
                throw std::runtime_error("--sim-snr-trace segment frame count must be positive: " + token);
            }
            segments.push_back({snr_db, frames});
        }
    }
    if (segments.empty()) {
        throw std::runtime_error("--sim-snr-trace did not contain any valid SNR segments");
    }
    return segments;
}

static std::vector<double> load_numeric_trace_file(const std::string& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open numeric trace file: " + path);
    }

    std::vector<double> values;
    std::string line;
    while (std::getline(in, line)) {
        const size_t hash = line.find('#');
        if (hash != std::string::npos) {
            line.resize(hash);
        }
        for (char& c : line) {
            if (c == ',' || c == ';' || c == '\t') {
                c = ' ';
            }
        }
        std::stringstream ss(line);
        std::string token;
        std::vector<double> nums;
        while (ss >> token) {
            try {
                size_t parsed = 0;
                const double v = std::stod(token, &parsed);
                if (parsed == token.size() && std::isfinite(v)) {
                    nums.push_back(v);
                }
            } catch (...) {
            }
        }
        if (nums.empty()) {
            continue;
        }
        // A two-column CSV usually stores frame,powerDb; use the last numeric column.
        values.push_back(nums.back());
    }
    if (values.empty()) {
        throw std::runtime_error("numeric trace file did not contain any values: " + path);
    }
    return values;
}

class LiveNumericTraceReader {
public:
    LiveNumericTraceReader(std::string path, int wait_ms)
        : path_(std::move(path)),
          wait_ms_(std::max(wait_ms, 0))
    {
    }

    double value_at(size_t index, double fallback)
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(wait_ms_);
        while (true) {
            refresh();
            if (index < values_.size()) {
                last_value_ = values_[index];
                have_last_ = true;
                return last_value_;
            }
            if (!values_.empty()) {
                last_value_ = values_.back();
                have_last_ = true;
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                return have_last_ ? last_value_ : fallback;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    size_t sample_count() const
    {
        return values_.size();
    }

private:
    void refresh()
    {
        if (!std::filesystem::exists(std::filesystem::path(path_))) {
            return;
        }
        try {
            values_ = load_numeric_trace_file(path_);
        } catch (...) {
            // The UI may be atomically refreshing the live file. Keep the
            // previous snapshot and try again on the next frame.
        }
    }

    std::string path_;
    int wait_ms_ = 0;
    std::vector<double> values_;
    double last_value_ = 0.0;
    bool have_last_ = false;
};

static std::vector<SimSnrSegment> parse_numeric_trace_segments(const std::string& spec,
                                                              const std::string& option_name)
{
    std::vector<SimSnrSegment> segments;
    std::stringstream ss(spec);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = trim_copy(token);
        if (token.empty()) {
            continue;
        }
        const size_t colon = token.find(':');
        if (colon == std::string::npos) {
            segments.push_back({std::stod(token), 1});
        } else {
            const double value = std::stod(trim_copy(token.substr(0, colon)));
            const int frames = std::stoi(trim_copy(token.substr(colon + 1)));
            if (frames <= 0) {
                throw std::runtime_error(option_name + " segment frame count must be positive: " + token);
            }
            segments.push_back({value, frames});
        }
    }
    if (segments.empty()) {
        throw std::runtime_error(option_name + " did not contain any valid segments");
    }
    return segments;
}

static std::vector<double> expand_numeric_trace_spec_or_file(const std::string& spec,
                                                             int frames,
                                                             const std::string& option_name)
{
    std::vector<double> source;
    if (!spec.empty() && std::filesystem::exists(std::filesystem::path(spec))) {
        source = load_numeric_trace_file(spec);
    } else {
        const auto segments = parse_numeric_trace_segments(spec, option_name);
        source.reserve(static_cast<size_t>(std::max(frames, 0)));
        while (static_cast<int>(source.size()) < frames) {
            for (const auto& seg : segments) {
                const int count = std::max(seg.frames, 1);
                for (int i = 0; i < count && static_cast<int>(source.size()) < frames; ++i) {
                    source.push_back(seg.snr_db);
                }
            }
        }
    }
    if (source.empty()) {
        throw std::runtime_error(option_name + " expanded to an empty trace");
    }
    if (static_cast<int>(source.size()) < frames) {
        const double last = source.back();
        source.resize(static_cast<size_t>(frames), last);
    } else if (static_cast<int>(source.size()) > frames) {
        source.resize(static_cast<size_t>(frames));
    }
    return source;
}

static std::vector<double> expand_sim_snr_trace(const Options& opt, int frames)
{
    if (!opt.sim_snr_trace.empty() &&
        std::filesystem::exists(std::filesystem::path(opt.sim_snr_trace))) {
        auto trace = load_numeric_trace_file(opt.sim_snr_trace);
        if (trace.empty()) {
            throw std::runtime_error("--sim-snr-trace file expanded to an empty trace");
        }
        if (static_cast<int>(trace.size()) < frames) {
            trace.resize(static_cast<size_t>(frames), trace.back());
        } else if (static_cast<int>(trace.size()) > frames) {
            trace.resize(static_cast<size_t>(frames));
        }
        return trace;
    }
    const auto segments = parse_sim_snr_segments(opt);
    std::vector<double> trace;
    trace.reserve(static_cast<size_t>(std::max(frames, 0)));
    if (segments.size() == 1 && segments.front().frames <= 0) {
        trace.assign(static_cast<size_t>(std::max(frames, 0)), segments.front().snr_db);
        return trace;
    }
    while (static_cast<int>(trace.size()) < frames) {
        for (const auto& seg : segments) {
            const int count = std::max(seg.frames, 1);
            for (int i = 0; i < count && static_cast<int>(trace.size()) < frames; ++i) {
                trace.push_back(seg.snr_db);
            }
        }
    }
    return trace;
}

static int run_gpu_sim_snr_trace(
    const Options& opt,
    const PhyConfig& base_phy,
    const LdpcCode& code,
    const fec::RxDecoderPlan& rx_plan)
{
    const bool file_mode = opt.traffic_mode == "file";
    if (rx_plan.kind != fec::RxDecoderKind::GpuBp &&
        rx_plan.kind != fec::RxDecoderKind::GpuBpOsd) {
        throw std::runtime_error("--mode gpu-sim with --sim-snr-trace requires --decoder cuda-bp or cuda-bp-osd");
    }

    std::unique_ptr<CudaBpDecoder> cuda_bp_decoder;
    std::unique_ptr<CudaBpOsdDecoder> cuda_decoder;
    if (rx_plan.kind == fec::RxDecoderKind::GpuBp) {
        cuda_bp_decoder = std::make_unique<CudaBpDecoder>(opt, code, *rx_plan.fec);
    } else if (rx_plan.kind == fec::RxDecoderKind::GpuBpOsd) {
        cuda_decoder = std::make_unique<CudaBpOsdDecoder>(opt, code);
    }

    InjectionChannel precomp_channel;
    InjectionChannel actual_channel;
    if (!opt.precomp_channel_files.empty()) {
        precomp_channel = load_injection_channel(opt.precomp_channel_files);
        print_injection_channel("gpu-sim-precomp", precomp_channel);
    }
    if (!opt.actual_channel_files.empty()) {
        actual_channel = load_injection_channel(opt.actual_channel_files);
        print_injection_channel("gpu-sim-actual", actual_channel);
    }

    struct Profile {
        std::string modulation;
        PhyConfig phy;
        int info_bits = 0;
        size_t info_bytes = 0;
        size_t payload_capacity = 0;
        std::unique_ptr<GpuTxBasebandChain> tx;
    };

    auto make_profile = [&](const std::string& modulation) {
        Options p_opt = opt;
        p_opt.modulation = modulation;
        Profile p;
        p.modulation = modulation;
        p.phy = build_phy_config(p_opt);
        if (p.phy.frame_len() != base_phy.frame_len()) {
            throw std::runtime_error("dynamic SNR profiles must keep identical OFDM frame length");
        }
        if (p.phy.coded_bits_per_frame() % code.n != 0) {
            throw std::runtime_error("coded bits per frame is not divisible by LDPC code length for modulation " + modulation);
        }
        p.info_bits = link_info_bits_per_frame(p.phy, code);
        p.info_bytes = link_info_bytes_per_frame(p.phy, code);
        p.payload_capacity = media_payload_capacity_bytes(p.phy, code);
        return p;
    };

    const bool predictive_power_amc = opt.adaptive_predictive && !opt.adaptive_power_trace.empty();

    std::vector<Profile> profiles;
    if (predictive_power_amc) {
        profiles.reserve(3);
        profiles.push_back(make_profile("bpsk"));
        profiles.push_back(make_profile("qpsk"));
        if (opt.adaptive_enable_16qam) {
            profiles.push_back(make_profile("16qam"));
        }
    } else if (opt.adaptive_dynamic_modulation) {
        profiles.reserve(2);
        profiles.push_back(make_profile("qpsk"));
        profiles.push_back(make_profile("16qam"));
    } else {
        profiles.reserve(1);
        profiles.push_back(make_profile(base_phy.modulation));
    }
    for (auto& p : profiles) {
        p.tx = std::make_unique<GpuTxBasebandChain>(p.phy, code, &precomp_channel, &actual_channel);
    }

    const size_t packetizer_info_bytes = profiles.front().info_bytes;
    std::unique_ptr<MediaPacketizer> tx_media;
    std::unique_ptr<MediaReassembler> rx_media;
    if (file_mode) {
        if (opt.input_file.empty()) {
            throw std::runtime_error("--traffic file requires --input <path> in gpu-sim mode");
        }
        tx_media = std::make_unique<MediaPacketizer>(opt.input_file, packetizer_info_bytes, opt.loop_file);
        rx_media = std::make_unique<MediaReassembler>(default_output_file(opt));
        std::cout << "[MEDIA] GPU-SIM dynamic input=" << opt.input_file
                  << " output=" << rx_media->output_path
                  << " bytes=" << tx_media->file_bytes.size()
                  << " basePayload/frame=" << profiles.front().payload_capacity << "\n";
    }

    int nominal_frames = opt.frames > 0
        ? opt.frames
        : (file_mode ? static_cast<int>((tx_media->file_bytes.size() + profiles.front().payload_capacity - 1) /
                                        std::max<size_t>(profiles.front().payload_capacity, 1))
                     : 1200);
    std::vector<double> predicted_power_trace_db;
    std::vector<double> snr_trace;
    std::unique_ptr<LiveNumericTraceReader> live_power_reader;
    if (predictive_power_amc) {
        if (opt.adaptive_power_trace_live) {
            live_power_reader = std::make_unique<LiveNumericTraceReader>(
                opt.adaptive_power_trace,
                opt.adaptive_power_live_wait_ms);
            predicted_power_trace_db.assign(static_cast<size_t>(nominal_frames), 0.0);
            snr_trace.assign(static_cast<size_t>(nominal_frames), opt.adaptive_power_base_snr_db);
        } else {
            if (!opt.sim_snr_trace.empty() &&
                std::filesystem::exists(std::filesystem::path(opt.sim_snr_trace))) {
                snr_trace = load_numeric_trace_file(opt.sim_snr_trace);
                if (!snr_trace.empty() && opt.frames <= 0) {
                    nominal_frames = static_cast<int>(snr_trace.size());
                }
            }
            predicted_power_trace_db = expand_numeric_trace_spec_or_file(
                opt.adaptive_power_trace,
                nominal_frames,
                "--adaptive-power-trace");
            if (snr_trace.empty()) {
                snr_trace.resize(predicted_power_trace_db.size());
                for (size_t i = 0; i < predicted_power_trace_db.size(); ++i) {
                    snr_trace[i] = opt.adaptive_power_base_snr_db + predicted_power_trace_db[i];
                }
            } else {
                if (static_cast<int>(snr_trace.size()) < nominal_frames) {
                    snr_trace.resize(static_cast<size_t>(nominal_frames), snr_trace.back());
                } else if (static_cast<int>(snr_trace.size()) > nominal_frames) {
                    snr_trace.resize(static_cast<size_t>(nominal_frames));
                }
            }
        }
    } else {
        snr_trace = expand_sim_snr_trace(opt, nominal_frames);
    }
    if (snr_trace.empty()) {
        throw std::runtime_error("dynamic SNR trace is empty");
    }

    std::map<std::pair<int, int>, std::unique_ptr<GpuRxDemodChain>> demod_cache;
    auto get_demod = [&](int profile_index, double snr_db) -> GpuRxDemodChain& {
        // Quantize the demod cache key. Per-frame jitter can otherwise create
        // thousands of CUDA demod objects and exhaust GPU/host memory.
        const double demod_snr_db = std::isfinite(snr_db)
            ? std::round(snr_db * 2.0) / 2.0
            : snr_db;
        const int snr_key = std::isfinite(demod_snr_db)
            ? static_cast<int>(std::llround(demod_snr_db * 2.0))
            : std::numeric_limits<int>::min();
        const auto key = std::make_pair(profile_index, snr_key);
        auto it = demod_cache.find(key);
        if (it != demod_cache.end()) {
            return *it->second;
        }
        const Profile& p = profiles[static_cast<size_t>(profile_index)];
        const float noise_var = gpu_sim_frequency_noise_var(
            p.phy,
            precomp_channel.empty() ? nullptr : &precomp_channel,
            actual_channel.empty() ? nullptr : &actual_channel,
            demod_snr_db);
        auto demod = std::make_unique<GpuRxDemodChain>(
            p.phy,
            code,
            precomp_channel.empty() ? nullptr : &precomp_channel,
            std::isfinite(demod_snr_db) ? noise_var : 1.0e-3f);
        auto inserted = demod_cache.emplace(key, std::move(demod));
        std::cout << "[GPU-SIM-SNR] demod profile modulation=" << p.modulation
                  << " snr=" << snr_db
                  << " cacheSnr=" << demod_snr_db
                  << " dB noiseVar=" << noise_var << "\n";
        return *inserted.first->second;
    };

    auto profile_index_for_modulation = [&](const std::string& modulation) {
        for (size_t i = 0; i < profiles.size(); ++i) {
            if (profiles[i].modulation == modulation) {
                return static_cast<int>(i);
            }
        }
        return 0;
    };

    usrp_link::PredictiveAmcConfig predictive_cfg;
    predictive_cfg.enable_16qam = opt.adaptive_enable_16qam;
    predictive_cfg.min_repeat = std::max(opt.tx_repeat_min, 1);
    predictive_cfg.max_repeat = std::max(opt.tx_repeat_max, predictive_cfg.min_repeat);
    if (predictive_power_amc) {
        // The per-frame power trace is a feed-forward predictor, so it can switch
        // immediately without the RX-feedback hold-down used by online ARQ.
        predictive_cfg.hold_down_windows = 0;
        predictive_cfg.upgrade_windows = 1;
    }
    usrp_link::PredictiveAmcPolicy predictive_policy(predictive_cfg);

    const double upgrade_snr = std::max(opt.sim_dynamic_upgrade_snr_db, opt.sim_dynamic_downgrade_snr_db);
    const double downgrade_snr = std::min(opt.sim_dynamic_upgrade_snr_db, opt.sim_dynamic_downgrade_snr_db);
    int active_profile = 0;
    uint64_t bpsk_frames = 0;
    uint64_t qpsk_frames = 0;
    uint64_t qam16_frames = 0;
    uint64_t repeat1_frames = 0;
    uint64_t repeat2_frames = 0;
    uint64_t repeat3_frames = 0;
    double sum_snr = 0.0;
    double sum_predicted_power_db = 0.0;
    double min_predicted_power_db = std::numeric_limits<double>::infinity();
    double max_predicted_power_db = -std::numeric_limits<double>::infinity();
    double min_snr = std::numeric_limits<double>::infinity();
    double max_snr = -std::numeric_limits<double>::infinity();
    uint64_t sent_frames = 0;
    uint64_t physical_tx_frames = 0;
    uint32_t expected_test_frame_id = 0;
    uint64_t media_offset = 0;
    uint64_t gpu_tx_ns = 0;
    uint64_t gpu_tx_frames = 0;
    bool stopped_on_frame_errors = false;
    bool stopped_on_zero_errors = false;
    SyncStats sync_stats;
    DecodeProfile decode_profile;
    RxStats stats;
    auto t0 = std::chrono::steady_clock::now();

    std::cout << "[GPU-SIM-SNR] traceFrames=" << snr_trace.size()
              << " dynamicModulation=" << (opt.adaptive_dynamic_modulation ? "yes" : "no")
              << " predictivePowerAmc=" << (predictive_power_amc ? "yes" : "no")
              << " downgrade=" << downgrade_snr
              << " upgrade=" << upgrade_snr
              << " profiles=";
    for (size_t i = 0; i < profiles.size(); ++i) {
        if (i) {
            std::cout << ",";
        }
        std::cout << profiles[i].modulation << "(" << profiles[i].info_bits << "b)";
    }
    std::cout << "\n";
    if (predictive_power_amc) {
        std::cout << "[ADAPT-POWER] baseSnr=" << opt.adaptive_power_base_snr_db
                  << " dB";
        if (opt.adaptive_power_trace_live) {
            std::cout << " liveTrace=" << opt.adaptive_power_trace
                      << " waitMs=" << opt.adaptive_power_live_wait_ms;
        } else {
            const auto [pmin_it, pmax_it] = std::minmax_element(
                predicted_power_trace_db.begin(), predicted_power_trace_db.end());
            std::cout << " relativePowerDb[min,max]=" << *pmin_it << "," << *pmax_it;
        }
        std::cout
                  << " confidence=" << opt.adaptive_prediction_confidence
                  << " evm=" << opt.adaptive_prediction_evm
                  << " holdFrames=" << opt.adaptive_power_hold_frames
                  << " repeat=" << predictive_cfg.min_repeat << ".." << predictive_cfg.max_repeat
                  << "\n";
    }

    auto test_fer_stop_conditions = [&]() {
        if (file_mode) {
            return;
        }
        if (opt.max_frame_errors > 0 &&
            stats.err >= static_cast<uint64_t>(opt.max_frame_errors)) {
            stopped_on_frame_errors = true;
        }
        if (opt.zero_error_stop_frames > 0 &&
            stats.detected >= static_cast<uint64_t>(opt.zero_error_stop_frames) &&
            stats.err == 0) {
            stopped_on_zero_errors = true;
        }
    };
    auto fer_stop_reached = [&]() {
        return stopped_on_frame_errors || stopped_on_zero_errors;
    };

    for (int i = 0; i < static_cast<int>(snr_trace.size()); ++i) {
        if (file_mode && tx_media->done_offset(media_offset)) {
            break;
        }
        double predicted_power_db = predictive_power_amc
            ? predicted_power_trace_db[static_cast<size_t>(i)]
            : 0.0;
        if (predictive_power_amc && live_power_reader) {
            predicted_power_db = live_power_reader->value_at(
                static_cast<size_t>(i / std::max(opt.adaptive_power_hold_frames, 1)),
                predicted_power_db);
            predicted_power_db = std::clamp(predicted_power_db, -30.0, 12.0);
            predicted_power_trace_db[static_cast<size_t>(i)] = predicted_power_db;
            snr_trace[static_cast<size_t>(i)] = opt.adaptive_power_base_snr_db + predicted_power_db;
        } else if (predictive_power_amc && opt.adaptive_power_hold_frames > 1) {
            const size_t held_index = std::min(
                static_cast<size_t>(i / opt.adaptive_power_hold_frames),
                predicted_power_trace_db.empty() ? size_t{0} : predicted_power_trace_db.size() - 1);
            predicted_power_db = predicted_power_trace_db[held_index];
            snr_trace[static_cast<size_t>(i)] = opt.adaptive_power_base_snr_db + predicted_power_db;
        }
        const double snr_db = snr_trace[static_cast<size_t>(i)];
        int repeat_count = 1;
        usrp_link::AmcDecision amc_decision;

        if (predictive_power_amc) {
            usrp_link::ChannelPrediction pred;
            pred.prediction_id = static_cast<uint64_t>(i);
            pred.target_phy_frame_id = static_cast<uint64_t>(i);
            pred.age_ms = opt.adaptive_prediction_age_ms;
            pred.confidence = static_cast<float>(opt.adaptive_prediction_confidence);
            const double path_amp = std::sqrt(std::pow(10.0, predicted_power_db / 10.0));
            pred.paths.push_back(usrp_link::PredictedPath{
                static_cast<float>(path_amp),
                0.0f,
                0.0f,
                0.0f,
            });
            const auto quality = predictive_policy.evaluate_prediction(
                pred,
                snr_db,
                0.0,
                opt.adaptive_prediction_evm);
            amc_decision = predictive_policy.decide(pred, quality, nullptr);
            active_profile = profile_index_for_modulation(amc_decision.modulation);
            repeat_count = std::max(1, amc_decision.repeat_count);
        } else if (opt.adaptive_dynamic_modulation && profiles.size() > 1) {
            const int old_profile = active_profile;
            if (active_profile == 0 && snr_db >= upgrade_snr) {
                active_profile = 1;
            } else if (active_profile == 1 && snr_db < downgrade_snr) {
                active_profile = 0;
            }
            if (old_profile != active_profile) {
                std::cout << "[GPU-SIM-DYN-SWITCH] frame=" << i
                          << " snr=" << snr_db
                          << " modulation=" << profiles[static_cast<size_t>(active_profile)].modulation << "\n";
            }
        }

        Profile& p = profiles[static_cast<size_t>(active_profile)];
        if (p.modulation == "bpsk") {
            ++bpsk_frames;
        } else if (p.modulation == "16qam") {
            ++qam16_frames;
        } else if (p.modulation == "qpsk") {
            ++qpsk_frames;
        }
        if (repeat_count <= 1) {
            ++repeat1_frames;
        } else if (repeat_count == 2) {
            ++repeat2_frames;
        } else {
            ++repeat3_frames;
        }
        if (std::isfinite(snr_db)) {
            sum_snr += snr_db;
            min_snr = std::min(min_snr, snr_db);
            max_snr = std::max(max_snr, snr_db);
        }
        if (predictive_power_amc && std::isfinite(predicted_power_db)) {
            sum_predicted_power_db += predicted_power_db;
            min_predicted_power_db = std::min(min_predicted_power_db, predicted_power_db);
            max_predicted_power_db = std::max(max_predicted_power_db, predicted_power_db);
        }

        MediaInfoBits media_info;
        bool have_media_info = false;
        if (file_mode) {
            media_info = tx_media->build_media_info_bits_from_offset(
                static_cast<uint32_t>(i),
                media_offset,
                p.info_bytes,
                p.payload_capacity);
            have_media_info = true;
            media_offset = media_info.next_offset;
        }

        FrameDecodeResult r;
        bool have_result = false;
        for (int rep = 0; rep < repeat_count; ++rep) {
            const float noise_var = gpu_sim_frequency_noise_var(
                p.phy,
                precomp_channel.empty() ? nullptr : &precomp_channel,
                actual_channel.empty() ? nullptr : &actual_channel,
                snr_db);
            const uint64_t noise_seed =
                (static_cast<uint64_t>(opt.test_seed) << 32) ^
                (0x679aa031ull + static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ull) ^
                (0x51ed2705ull + static_cast<uint64_t>(rep) * 0xbf58476d1ce4e5b9ull);

            const auto t_tx0 = std::chrono::steady_clock::now();
            GpuTxBasebandChain::DeviceFrame tx;
            if (file_mode) {
                tx = p.tx->build_media_frame_device(media_info.bits, noise_var, noise_seed);
            } else {
                tx = p.tx->build_test_frame_device(static_cast<uint32_t>(i), noise_var, noise_seed);
            }
            const auto t_tx1 = std::chrono::steady_clock::now();
            gpu_tx_ns += elapsed_ns(t_tx0, t_tx1);
            ++gpu_tx_frames;
            ++physical_tx_frames;
            ++sync_stats.found;

            ExtractedDeviceFrame extracted;
            extracted.device_samples = const_cast<cf32*>(tx.samples);
            extracted.sample_count = tx.sample_count;
            extracted.peak = 1.0;
            extracted.cfo_hz = 0.0;

            FrameDecodeResult candidate = get_demod(active_profile, snr_db).process_device(
                extracted,
                cuda_bp_decoder.get(),
                cuda_decoder.get(),
                opt.ldpc_max_iter,
                opt.ldpc_normalization,
                false,
                opt.profile_pipeline ? &decode_profile : nullptr);
            candidate.snr_db = snr_db;
            if (!file_mode) {
                apply_test_reference_result(
                    candidate,
                    p.phy,
                    code,
                    expected_test_frame_id,
                    opt.skip_reference_ber,
                    opt.profile_pipeline ? &decode_profile : nullptr);
                candidate.snr_db = snr_db;
            }
            if (!have_result || (!r.frame_ok && candidate.frame_ok)) {
                r = candidate;
                have_result = true;
            }
        }

        if (!have_result) {
            throw std::runtime_error("internal error: adaptive GPU-SIM produced no frame result");
        }
        ++sent_frames;
        if (!file_mode) {
            ++expected_test_frame_id;
        } else if (have_media_info) {
            accept_media_decode(r, *rx_media, p.payload_capacity);
        }
        update_stats(stats, r, p.info_bits, p.phy.coded_bits_per_frame());
        {
            const double ui_elapsed_link_sec =
                static_cast<double>(physical_tx_frames) *
                static_cast<double>(p.phy.frame_len()) /
                std::max(p.phy.rate, 1.0);
            const double ui_elapsed_sec = opt.gpu_sim_wall_clock_throughput
                ? std::max(std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - t0).count(),
                           1.0e-9)
                : ui_elapsed_link_sec;
            send_ui_metrics_if_due(
                stats,
                ui_elapsed_sec,
                p.info_bits,
                r,
                predictive_power_amc ? static_cast<int>(amc_decision.mcs) : static_ui_mcs_index(p.modulation),
                static_cast<int>(snr_trace.size()),
                predictive_power_amc ? predicted_power_db : (snr_db - opt.adaptive_power_base_snr_db));
        }
        if (!opt.suppress_error_frames ||
            i < 5 ||
            ((i + 1) % 500) == 0 ||
            (!r.frame_ok && stats.err <= 10)) {
            std::cout << "[GPU-SIM] modulation=" << p.modulation
                      << " repeat=" << repeat_count;
            if (predictive_power_amc) {
                std::cout << " predPower=" << std::fixed << std::setprecision(2) << predicted_power_db
                          << "dB mcs=" << static_cast<int>(amc_decision.mcs);
            }
            std::cout << " ";
            print_frame_result("", r);
        }
        test_fer_stop_conditions();
        if (fer_stop_reached()) {
            break;
        }
        if (file_mode && rx_media->complete && !opt.loop_file) {
            break;
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    if (stopped_on_frame_errors) {
        std::cout << "[GPU-SIM] early FER stop: frame_errors=" << stats.err
                  << " limit=" << opt.max_frame_errors
                  << " tested_frames=" << stats.detected << "\n";
    }
    if (stopped_on_zero_errors) {
        std::cout << "[GPU-SIM] early zero-error stop: frame_errors=0"
                  << " observation_frames=" << opt.zero_error_stop_frames
                  << " tested_frames=" << stats.detected << "\n";
    }
    const double link_sec =
        static_cast<double>(std::max<uint64_t>(physical_tx_frames, 1)) *
        static_cast<double>(profiles.front().phy.frame_len()) /
        std::max(profiles.front().phy.rate, 1.0);
    const double stats_sec = opt.gpu_sim_wall_clock_throughput ? std::max(sec, 1.0e-9) : link_sec;
    print_stats(stats, stats_sec, profiles.front().info_bits);
    const double finite_snr_frames = static_cast<double>(std::max<uint64_t>(stats.detected, 1));
    std::cout << "[GPU-SIM-SNR-SUMMARY]"
              << " frames=" << stats.detected
              << " physicalFrames=" << physical_tx_frames
              << " bpskFrames=" << bpsk_frames
              << " qpskFrames=" << qpsk_frames
              << " qam16Frames=" << qam16_frames
              << " repeat1Frames=" << repeat1_frames
              << " repeat2Frames=" << repeat2_frames
              << " repeat3Frames=" << repeat3_frames
              << " avgTraceSnr=" << (sum_snr / finite_snr_frames)
              << " minTraceSnr=" << (std::isfinite(min_snr) ? min_snr : 0.0)
              << " maxTraceSnr=" << (std::isfinite(max_snr) ? max_snr : 0.0)
              << " avgPredPowerDb=" << (predictive_power_amc ? (sum_predicted_power_db / finite_snr_frames) : 0.0)
              << " minPredPowerDb=" << (std::isfinite(min_predicted_power_db) ? min_predicted_power_db : 0.0)
              << " maxPredPowerDb=" << (std::isfinite(max_predicted_power_db) ? max_predicted_power_db : 0.0)
              << " okInfoBits=" << stats.ok_info_bits
              << " linkElapsedSec=" << link_sec
              << " wallElapsedSec=" << sec
              << " throughputClock=" << (opt.gpu_sim_wall_clock_throughput ? "wall" : "link")
              << "\n";
    if (opt.profile_pipeline) {
        print_bench_decode_substage("gpu_tx_encode_map_channel_awgn_ofdm", gpu_tx_ns, gpu_tx_frames, profiles.front().info_bits);
        print_bench_decode_substage("gpu_demod_llr", decode_profile.llr_ns, decode_profile.frames, profiles.front().info_bits);
        print_bench_decode_substage("fec_decode", decode_profile.fec_ns, decode_profile.frames, profiles.front().info_bits);
        print_bench_decode_substage("reference_ber", decode_profile.reference_ns, decode_profile.frames, profiles.front().info_bits);
    }

    if (stats.err != 0 || (!stopped_on_zero_errors && stats.detected != sent_frames)) {
        return 2;
    }
    if (file_mode && opt.frames <= 0 && !opt.loop_file && !rx_media->complete) {
        return 2;
    }
    return 0;
}

static int run_gpu_sim(const Options& opt, const PhyConfig& phy, const LdpcCode& code, const fec::RxDecoderPlan& rx_plan)
{
    if (!opt.sim_snr_trace.empty() || opt.adaptive_dynamic_modulation) {
        return run_gpu_sim_snr_trace(opt, phy, code, rx_plan);
    }

    const bool file_mode = opt.traffic_mode == "file";
    const int info_bits = link_info_bits_per_frame(phy, code);
    const size_t info_bytes = link_info_bytes_per_frame(phy, code);
    const size_t payload_capacity = file_mode ? media_payload_capacity_bytes(phy, code) : 0;

    std::unique_ptr<CudaBpDecoder> cuda_bp_decoder;
    std::unique_ptr<CudaBpOsdDecoder> cuda_decoder;
    if (rx_plan.kind == fec::RxDecoderKind::GpuBp) {
        cuda_bp_decoder = std::make_unique<CudaBpDecoder>(opt, code, *rx_plan.fec);
    } else if (rx_plan.kind == fec::RxDecoderKind::GpuBpOsd) {
        cuda_decoder = std::make_unique<CudaBpOsdDecoder>(opt, code);
    }

    std::unique_ptr<MediaPacketizer> tx_media;
    std::unique_ptr<MediaReassembler> rx_media;
    if (file_mode) {
        if (opt.input_file.empty()) {
            throw std::runtime_error("--traffic file requires --input <path> in gpu-sim mode");
        }
        tx_media = std::make_unique<MediaPacketizer>(opt.input_file, info_bytes, opt.loop_file);
        rx_media = std::make_unique<MediaReassembler>(default_output_file(opt));
        std::cout << "[MEDIA] GPU-SIM input=" << opt.input_file
                  << " output=" << rx_media->output_path
                  << " bytes=" << tx_media->file_bytes.size()
                  << " chunks=" << tx_media->total_chunks
                  << " payload/frame=" << tx_media->payload_bytes_per_frame << "\n";
    }

    InjectionChannel precomp_channel;
    InjectionChannel actual_channel;
    if (!opt.precomp_channel_files.empty()) {
        precomp_channel = load_injection_channel(opt.precomp_channel_files);
        print_injection_channel("gpu-sim-precomp", precomp_channel);
    }
    if (!opt.actual_channel_files.empty()) {
        actual_channel = load_injection_channel(opt.actual_channel_files);
        print_injection_channel("gpu-sim-actual", actual_channel);
    }
    const float gpu_frequency_noise_var = gpu_sim_frequency_noise_var(
        phy,
        precomp_channel.empty() ? nullptr : &precomp_channel,
        actual_channel.empty() ? nullptr : &actual_channel,
        opt.sim_snr_db);
    if (std::isfinite(opt.sim_snr_db)) {
        std::cout << "[GPU-SIM] CUDA frequency-domain AWGN enabled: snr=" << opt.sim_snr_db
                  << " dB noiseVar=" << gpu_frequency_noise_var
                  << " scope=data-subcarriers\n";
        if (opt.tx_awgn_scope != "data-subcarriers") {
            std::cout << "[GPU-SIM] note: GPU-sim uses data-subcarrier AWGN for all device-resident runs.\n";
        }
    }

    GpuTxBasebandChain gpu_tx(phy, code, &precomp_channel, &actual_channel);
    const float gpu_demod_noise_var = std::isfinite(opt.sim_snr_db)
        ? gpu_frequency_noise_var
        : 1.0e-3f;
    GpuRxDemodChain gpu_demod(phy, code, &precomp_channel, gpu_demod_noise_var);

    const int frames = opt.frames > 0
        ? opt.frames
        : (file_mode ? static_cast<int>(tx_media->total_chunks) : 4);
    SyncStats sync_stats;
    DecodeProfile decode_profile;
    RxStats stats;
    auto t0 = std::chrono::steady_clock::now();
    uint64_t sent_frames = 0;
    uint32_t expected_test_frame_id = 0;
    uint64_t gpu_tx_ns = 0;
    uint64_t gpu_tx_frames = 0;
    bool stopped_on_frame_errors = false;
    bool stopped_on_zero_errors = false;
    const int blocks_per_frame = phy.coded_bits_per_frame() / code.n;
    const bool skip_test_reference_ber = opt.skip_reference_ber && !file_mode;
    const bool async_bp_osd =
        cuda_decoder != nullptr && !cuda_decoder->osd_only && cuda_bp_decoder == nullptr;
    const int async_max_frames_per_batch = async_bp_osd
        ? std::max(1, cuda_decoder->async_max_codewords() / std::max(blocks_per_frame, 1))
        : 1;
    const int async_window_frames = async_bp_osd
        ? std::max(1, std::min(16, async_max_frames_per_batch))
        : 1;

    struct AsyncGpuFrame {
        uint32_t expected_frame_id = 0;
        double peak = 0.0;
        double cfo_hz = 0.0;
        uint64_t demod_ns = 0;
        uint64_t fec_ns = 0;
        int used_osd_count = 0;
        int ready_blocks = 0;
        std::vector<uint8_t> info_bits;
        std::vector<uint8_t> block_ready;
        std::chrono::steady_clock::time_point submit_time;
    };
    std::map<uint64_t, AsyncGpuFrame> async_pending;

    if (async_bp_osd) {
        std::cout << "[GPU-SIM] BP-OSD cross-frame batching enabled: blocks/frame="
                  << blocks_per_frame
                  << " windowFrames=" << async_window_frames
                  << " maxBatchCodewords=" << cuda_decoder->async_max_codewords()
                  << "\n";
    }

    auto test_fer_stop_conditions = [&]() {
        if (file_mode) {
            return;
        }
        if (opt.max_frame_errors > 0 &&
            stats.err >= static_cast<uint64_t>(opt.max_frame_errors)) {
            stopped_on_frame_errors = true;
        }
        if (opt.zero_error_stop_frames > 0 &&
            stats.detected >= static_cast<uint64_t>(opt.zero_error_stop_frames) &&
            stats.err == 0) {
            stopped_on_zero_errors = true;
        }
    };

    auto fer_stop_reached = [&]() {
        return stopped_on_frame_errors || stopped_on_zero_errors;
    };

    auto finish_async_frame = [&](AsyncGpuFrame&& pending) {
        if (fer_stop_reached()) {
            return;
        }
        const uint64_t fec_ns = pending.fec_ns != 0
            ? pending.fec_ns
            : elapsed_ns(pending.submit_time, std::chrono::steady_clock::now());
        FrameDecodeResult r = finish_decode_from_info_bits(
            std::move(pending.info_bits),
            phy,
            code,
            pending.used_osd_count,
            true,
            false,
            pending.peak,
            pending.cfo_hz,
            0.0,
            pending.demod_ns,
            fec_ns,
            opt.profile_pipeline ? &decode_profile : nullptr);
        if (!file_mode) {
            apply_test_reference_result(
                r,
                phy,
                code,
                pending.expected_frame_id,
                skip_test_reference_ber,
                opt.profile_pipeline ? &decode_profile : nullptr);
        } else {
            accept_media_decode(r, *rx_media, payload_capacity);
        }
        update_stats(stats, r, info_bits, phy.coded_bits_per_frame());
        {
            const auto now = std::chrono::steady_clock::now();
            send_ui_metrics_if_due(stats, std::chrono::duration<double>(now - t0).count(), info_bits, r);
        }
        if (!opt.suppress_error_frames ||
            stats.detected <= 5 ||
            (stats.detected % 500) == 0 ||
            (!r.frame_ok && stats.err <= 10)) {
            print_frame_result("[GPU-SIM]", r);
        }
        test_fer_stop_conditions();
    };

    auto drain_async_decoder = [&](bool submit_staged) {
        if (!async_bp_osd) {
            return false;
        }
        int submitted_frames = 0;
        uint64_t submitted_ns = 0;
        if (submit_staged) {
            const int submitted_codewords = cuda_decoder->async_pending_codewords();
            submitted_frames = submitted_codewords / std::max(blocks_per_frame, 1);
            submitted_ns = cuda_decoder->submit_async_batch();
            if (submitted_frames > 0 && submitted_ns > 0) {
                const uint64_t per_frame_ns = submitted_ns / static_cast<uint64_t>(submitted_frames);
                int tagged = 0;
                for (auto& kv : async_pending) {
                    if (kv.second.fec_ns == 0) {
                        kv.second.fec_ns = per_frame_ns;
                        if (++tagged >= submitted_frames) {
                            break;
                        }
                    }
                }
            }
        }
        bool media_complete = false;
        while (true) {
            auto ready = cuda_decoder->poll_ready_codewords(cuda_decoder->async_max_codewords());
            if (ready.empty()) {
                break;
            }
            for (const auto& cw : ready) {
                auto it = async_pending.find(cw.frame_id);
                if (it == async_pending.end()) {
                    throw std::runtime_error("CUDA BP-OSD async decoder returned an unknown frame id");
                }
                AsyncGpuFrame& pending = it->second;
                if (cw.block_index >= static_cast<uint32_t>(blocks_per_frame)) {
                    throw std::runtime_error("CUDA BP-OSD async decoder returned an invalid block index");
                }
                if (pending.block_ready[static_cast<size_t>(cw.block_index)] == 0) {
                    std::copy(
                        cw.info_bits.begin(),
                        cw.info_bits.end(),
                        pending.info_bits.begin() + static_cast<std::ptrdiff_t>(cw.block_index * code.k));
                    pending.block_ready[static_cast<size_t>(cw.block_index)] = 1;
                    pending.used_osd_count += cw.used_osd ? 1 : 0;
                    ++pending.ready_blocks;
                }
                if (pending.ready_blocks == blocks_per_frame) {
                    AsyncGpuFrame complete = std::move(pending);
                    async_pending.erase(it);
                    finish_async_frame(std::move(complete));
                    media_complete = media_complete || (file_mode && rx_media->complete && !opt.loop_file);
                }
            }
        }
        return media_complete;
    };

    for (int i = 0; i < frames; ++i) {
        if (file_mode && tx_media->done(sent_frames)) {
            break;
        }

        GpuTxBasebandChain::DeviceFrame tx;
        const uint64_t noise_seed =
            (static_cast<uint64_t>(opt.test_seed) << 32) ^
            (0x679aa031ull + static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ull);
        const auto t_tx0 = std::chrono::steady_clock::now();
        if (file_mode) {
            const auto info = tx_media->build_info_bits(static_cast<uint32_t>(i));
            tx = gpu_tx.build_media_frame_device(info, gpu_frequency_noise_var, noise_seed);
        } else {
            tx = gpu_tx.build_test_frame_device(static_cast<uint32_t>(i), gpu_frequency_noise_var, noise_seed);
        }
        const auto t_tx1 = std::chrono::steady_clock::now();
        gpu_tx_ns += elapsed_ns(t_tx0, t_tx1);
        ++gpu_tx_frames;
        ++sent_frames;
        ++sync_stats.found;

        ExtractedDeviceFrame extracted;
        extracted.device_samples = const_cast<cf32*>(tx.samples);
        extracted.sample_count = tx.sample_count;
        extracted.peak = 1.0;
        extracted.cfo_hz = 0.0;

        if (async_bp_osd) {
            if (cuda_decoder->async_pending_codewords() + blocks_per_frame > cuda_decoder->async_max_codewords()) {
                if (drain_async_decoder(true)) {
                    break;
                }
                if (fer_stop_reached()) {
                    break;
                }
            }
            const auto llr_frame = gpu_demod.demod_device_to_device_llr(extracted);
            const uint64_t decoder_frame_id =
                cuda_decoder->enqueue_device_frame_async(llr_frame.device_llrs, llr_frame.blocks);
            AsyncGpuFrame pending;
            pending.expected_frame_id = expected_test_frame_id;
            pending.peak = llr_frame.peak;
            pending.cfo_hz = llr_frame.cfo_hz;
            pending.demod_ns = llr_frame.demod_ns;
            pending.info_bits.assign(static_cast<size_t>(blocks_per_frame * code.k), 0);
            pending.block_ready.assign(static_cast<size_t>(blocks_per_frame), 0);
            pending.submit_time = std::chrono::steady_clock::now();
            async_pending.emplace(decoder_frame_id, std::move(pending));
            if (!file_mode) {
                ++expected_test_frame_id;
            }
            if (static_cast<int>(async_pending.size()) >= async_window_frames) {
                if (drain_async_decoder(true)) {
                    break;
                }
                if (fer_stop_reached()) {
                    break;
                }
            }
        } else {
            FrameDecodeResult r = gpu_demod.process_device(
                extracted,
                cuda_bp_decoder.get(),
                cuda_decoder.get(),
                opt.ldpc_max_iter,
                opt.ldpc_normalization,
                false,
                opt.profile_pipeline ? &decode_profile : nullptr);
                if (!file_mode) {
                    apply_test_reference_result(
                        r,
                        phy,
                        code,
                        expected_test_frame_id,
                        skip_test_reference_ber,
                        opt.profile_pipeline ? &decode_profile : nullptr);
                    ++expected_test_frame_id;
                } else {
                    accept_media_decode(r, *rx_media, payload_capacity);
                }
                update_stats(stats, r, info_bits, phy.coded_bits_per_frame());
                {
                    const auto now = std::chrono::steady_clock::now();
                    send_ui_metrics_if_due(stats, std::chrono::duration<double>(now - t0).count(), info_bits, r);
                }
                if (!opt.suppress_error_frames ||
                    i < 5 ||
                    ((i + 1) % 500) == 0 ||
                    (!r.frame_ok && stats.err <= 10)) {
                    print_frame_result("[GPU-SIM]", r);
                }
                test_fer_stop_conditions();
                if (fer_stop_reached()) {
                    break;
                }
        }
        if (file_mode && rx_media->complete && !opt.loop_file) {
            break;
        }
    }

    if (async_bp_osd && !async_pending.empty()) {
        drain_async_decoder(true);
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    if (stopped_on_frame_errors) {
        std::cout << "[GPU-SIM] early FER stop: frame_errors=" << stats.err
                  << " limit=" << opt.max_frame_errors
                  << " tested_frames=" << stats.detected << "\n";
    }
    if (stopped_on_zero_errors) {
        std::cout << "[GPU-SIM] early zero-error stop: frame_errors=0"
                  << " observation_frames=" << opt.zero_error_stop_frames
                  << " tested_frames=" << stats.detected << "\n";
    }
    print_stats(stats, sec, info_bits);
    if (opt.profile_pipeline) {
        std::cout << "[PROFILE] gpu_sync: found=" << sync_stats.found
                  << " miss=" << sync_stats.miss
                  << " trackingMiss=" << sync_stats.tracking_miss
                  << " incomplete=" << sync_stats.incomplete
                  << " skipped=" << sync_stats.skipped_samples
                  << " maxStart=" << sync_stats.max_start
                  << " equivSamples=" << (static_cast<double>(sync_stats.found) *
                                           static_cast<double>(phy.frame_len()) / std::max(sec, 1e-9) / 1e6)
                  << " Msps\n";
        print_bench_decode_substage("gpu_tx_encode_map_channel_awgn_ofdm", gpu_tx_ns, gpu_tx_frames, info_bits);
        print_bench_decode_substage("gpu_demod_llr", decode_profile.llr_ns, decode_profile.frames, info_bits);
        print_bench_decode_substage("fec_decode", decode_profile.fec_ns, decode_profile.frames, info_bits);
        print_bench_decode_substage("reference_ber", decode_profile.reference_ns, decode_profile.frames, info_bits);
    }

    if (stats.err != 0 || (!stopped_on_zero_errors && stats.detected != sent_frames)) {
        return 2;
    }
    if (file_mode && opt.frames <= 0 && !opt.loop_file && !rx_media->complete) {
        return 2;
    }
    return 0;
}
#endif

static double seconds_between(std::chrono::steady_clock::time_point t0, std::chrono::steady_clock::time_point t1)
{
    return std::max(std::chrono::duration<double>(t1 - t0).count(), 1e-9);
}

static void print_bench_stage(
    const std::string& name,
    uint64_t frames,
    double seconds,
    int info_bits_per_frame,
    int frame_samples)
{
    const double fps = static_cast<double>(frames) / std::max(seconds, 1e-9);
    const double info_mbps = fps * static_cast<double>(info_bits_per_frame) / 1.0e6;
    const double ms_frame = 1000.0 / std::max(fps, 1e-9);
    const double equiv_msps = fps * static_cast<double>(frame_samples) / 1.0e6;
    std::cout << "[BENCH] " << name
              << ": frames=" << frames
              << " sec=" << std::fixed << std::setprecision(3) << seconds
              << " fps=" << fps
              << " info=" << info_mbps << " Mbps"
              << " equivSamples=" << equiv_msps << " Msps"
              << " ms/frame=" << ms_frame
              << "\n";
}

static void print_bench_decode_substage(
    const std::string& name,
    uint64_t ns,
    uint64_t frames,
    int info_bits_per_frame)
{
    const double ms = frames > 0 ? static_cast<double>(ns) / static_cast<double>(frames) / 1.0e6 : 0.0;
    const double fps = ms > 0.0 ? 1000.0 / ms : 0.0;
    const double info_mbps = fps * static_cast<double>(info_bits_per_frame) / 1.0e6;
    std::cout << "[BENCH] decode." << name
              << ": ms/frame=" << std::fixed << std::setprecision(4) << ms
              << " cap_fps=" << std::setprecision(2) << fps
              << " cap_info=" << info_mbps << " Mbps\n";
}

static int run_bench(const Options& opt, const PhyConfig& phy, const LdpcCode& code, const fec::RxDecoderPlan& rx_plan)
{
    if (opt.traffic_mode != "test") {
        throw std::runtime_error("--mode bench currently benchmarks deterministic test traffic; use --traffic test");
    }

    const int frames = opt.frames > 0 ? opt.frames : 256;
    const int info_bits = link_info_bits_per_frame(phy, code);
    const uint64_t total_samples = static_cast<uint64_t>(frames) * static_cast<uint64_t>(phy.frame_len());
    InjectionChannel precomp_channel;
    if (!opt.precomp_channel_files.empty()) {
        precomp_channel = load_injection_channel(opt.precomp_channel_files);
        print_injection_channel("bench-precomp", precomp_channel);
    }
    InjectionChannel actual_channel;
    if (!opt.actual_channel_files.empty()) {
        actual_channel = load_injection_channel(opt.actual_channel_files);
        print_injection_channel("bench-actual", actual_channel);
    }
    TxBasebandChain tx_baseband(
        phy,
        code,
        precomp_channel.empty() ? nullptr : &precomp_channel,
        actual_channel.empty() ? nullptr : &actual_channel);

    std::unique_ptr<CudaBpDecoder> cuda_bp_decoder;
    std::unique_ptr<CudaBpOsdDecoder> cuda_decoder;
    if (rx_plan.kind == fec::RxDecoderKind::GpuBp) {
        cuda_bp_decoder = std::make_unique<CudaBpDecoder>(opt, code, *rx_plan.fec);
    } else if (rx_plan.kind == fec::RxDecoderKind::GpuBpOsd) {
        cuda_decoder = std::make_unique<CudaBpOsdDecoder>(opt, code);
    }

    std::cout << "[BENCH] frames=" << frames
              << " frameSamples=" << phy.frame_len()
              << " totalSamples=" << total_samples
              << " infoBits/frame=" << info_bits
              << " codedBits/frame=" << phy.coded_bits_per_frame()
              << " blocks/frame=" << (phy.coded_bits_per_frame() / code.n)
              << "\n";

    std::vector<std::vector<cf32>> tx_frames;
    tx_frames.reserve(static_cast<size_t>(frames));
    const auto t_tx0 = std::chrono::steady_clock::now();
    for (int i = 0; i < frames; ++i) {
        tx_frames.push_back(tx_baseband.build_test_frame(static_cast<uint32_t>(i)));
    }
    const auto t_tx1 = std::chrono::steady_clock::now();
    const double tx_sec = seconds_between(t_tx0, t_tx1);
    print_bench_stage("tx_fec_map_ofdm", static_cast<uint64_t>(tx_frames.size()), tx_sec, info_bits, phy.frame_len());

    std::vector<cf32> rxbuf;
    rxbuf.reserve(static_cast<size_t>(total_samples));
    for (const auto& f : tx_frames) {
        rxbuf.insert(rxbuf.end(), f.begin(), f.end());
    }
    tx_frames.clear();
    tx_frames.shrink_to_fit();

    std::vector<ExtractedFrame> extracted;
    extracted.reserve(static_cast<size_t>(frames));
    size_t rxbase = 0;
    SyncState sync_state = SyncState::Acquisition;
    SyncStats sync_stats;
    const auto t_sync0 = std::chrono::steady_clock::now();
    while (static_cast<int>(extracted.size()) < frames) {
        ExtractedFrame frame;
        if (!extract_rx_frame_vector(rxbuf, rxbase, phy, frame, sync_state, sync_stats)) {
            break;
        }
        extracted.push_back(std::move(frame));
    }
    const auto t_sync1 = std::chrono::steady_clock::now();
    const double sync_sec = seconds_between(t_sync0, t_sync1);
    print_bench_stage("rx_sync_extract", static_cast<uint64_t>(extracted.size()), sync_sec, info_bits, phy.frame_len());
    std::cout << "[BENCH] sync_stats: found=" << sync_stats.found
              << " miss=" << sync_stats.miss
              << " trackingMiss=" << sync_stats.tracking_miss
              << " incomplete=" << sync_stats.incomplete
              << " skippedSamples=" << sync_stats.skipped_samples
              << " maxStart=" << sync_stats.max_start
              << "\n";

    RxStats stats;
    DecodeProfile decode_profile;
    const auto t_dec0 = std::chrono::steady_clock::now();
    for (const auto& frame : extracted) {
        const auto r = decode_frame(
            frame.samples,
            phy,
            code,
            cuda_bp_decoder.get(),
            cuda_decoder.get(),
            frame.cfo_hz,
            frame.peak,
            opt.ldpc_max_iter,
            opt.ldpc_normalization,
            opt.rx_snr_gate_db,
            true,
            &decode_profile);
        update_stats(stats, r, info_bits, phy.coded_bits_per_frame());
    }
    const auto t_dec1 = std::chrono::steady_clock::now();
    const double dec_sec = seconds_between(t_dec0, t_dec1);
    print_bench_stage("rx_ofdm_demod_gpu_decode_total", stats.detected, dec_sec, info_bits, phy.frame_len());
    print_bench_decode_substage("cfo", decode_profile.cfo_ns, decode_profile.frames, info_bits);
    print_bench_decode_substage("fft_extract", decode_profile.fft_ns, decode_profile.frames, info_bits);
    print_bench_decode_substage("channel_est", decode_profile.channel_ns, decode_profile.frames, info_bits);
    print_bench_decode_substage("llr", decode_profile.llr_ns, decode_profile.frames, info_bits);
    print_bench_decode_substage("fec", decode_profile.fec_ns, decode_profile.frames, info_bits);
    print_bench_decode_substage("reference_ber", decode_profile.reference_ns, decode_profile.frames, info_bits);

    const double sync_fps = static_cast<double>(extracted.size()) / std::max(sync_sec, 1e-9);
    const double dec_fps = static_cast<double>(stats.detected) / std::max(dec_sec, 1e-9);
    const double tx_fps = static_cast<double>(frames) / std::max(tx_sec, 1e-9);
    const double limit_fps = std::min(tx_fps, std::min(sync_fps, dec_fps));
    std::string bottleneck = "tx_fec_map_ofdm";
    if (sync_fps <= tx_fps && sync_fps <= dec_fps) {
        bottleneck = "rx_sync_extract";
    } else if (dec_fps <= tx_fps && dec_fps <= sync_fps) {
        bottleneck = "rx_ofdm_demod_gpu_decode_total";
    }
    std::cout << "[BENCH] offline_pipeline_limit: bottleneck=" << bottleneck
              << " cap_fps=" << limit_fps
              << " cap_info=" << (limit_fps * static_cast<double>(info_bits) / 1.0e6) << " Mbps"
              << " cap_samples=" << (limit_fps * static_cast<double>(phy.frame_len()) / 1.0e6) << " Msps\n";

    print_stats(stats, dec_sec, info_bits);
    if (static_cast<int>(stats.detected) != frames || stats.err != 0) {
        return 2;
    }
    return 0;
}

int main(int argc, char** argv)
{
    try {
        Options opt = parse_options(argc, argv);
        apply_startup_amc(opt);
        if (opt.traffic_mode != "test" && opt.traffic_mode != "file") {
            throw std::runtime_error("unsupported traffic mode: " + opt.traffic_mode);
        }
        if (opt.decoder != "cpu" && opt.decoder != "cuda-bp-osd" && opt.decoder != "cuda-osd" && opt.decoder != "cuda-bp") {
            throw std::runtime_error("unsupported decoder: " + opt.decoder);
        }
        fec::MatrixLoader fec_loader;
        fec::FecConfigStore fec_store;
        const auto alist_file = std::filesystem::path(opt.alist_path).filename().string();
        const auto fec_request = alist_file.find("CCSDS") != std::string::npos
            ? fec::MatrixLoadRequest::ccsds_alist(opt.alist_path, 0, "cli-alist")
            : fec::MatrixLoadRequest::dvb_s2_alist(opt.alist_path, 0, "cli-alist");
        fec_store.swap(fec_loader.load(fec_request));
        const auto active_fec = fec_store.get();
        const auto alist = alist_from_dvb_s2_csr(*active_fec);
        const bool use_front_info =
            uses_front_info_layout(opt) ||
            (active_fec->standard == fec::FecStandard::DvbS2Ldpc && active_fec->block_length() > 1000);
        const auto code = build_ldpc(alist, use_front_info);
        const auto phy = build_phy_config(opt);

        fec::RxDecoderRuntimeOptions rx_route_options;
        rx_route_options.requested_decoder = opt.decoder;
        rx_route_options.bp_max_iterations = opt.ldpc_max_iter;
        rx_route_options.bp_normalization = static_cast<float>(opt.ldpc_normalization);
        rx_route_options.cuda_min_batch = opt.cuda_min_batch;
        rx_route_options.cuda_max_batch = opt.cuda_max_batch;
        rx_route_options.cuda_latency_us = opt.cuda_latency_us;
        const auto rx_decoder_plan = fec::RxDecoderRouter::make_plan(active_fec, rx_route_options);

        if (phy.coded_bits_per_frame() % code.n != 0) {
            std::ostringstream oss;
            oss << "coded bits per frame (" << phy.coded_bits_per_frame()
                << ") must be divisible by LDPC n (" << code.n << "). "
                << "Try --active-sc 512 or 768 with the default n512 code.";
            throw std::runtime_error(oss.str());
        }

        print_link_config(phy, code);
        std::cout << "FEC: standard=" << fec::to_string(rx_decoder_plan.fec->standard)
                  << " N=" << rx_decoder_plan.fec->block_length()
                  << " K=" << rx_decoder_plan.fec->information_length()
                  << " rxDecoder=" << fec::RxDecoderRouter::to_string(rx_decoder_plan.kind)
                  << " reason=\"" << rx_decoder_plan.reason << "\"\n";

        std::unique_ptr<UiTelemetryWorker> ui_telemetry;
        if (opt.ui_constellation || opt.ui_spectrum || opt.ui_metrics || opt.ui_ntn) {
            ui_telemetry = std::make_unique<UiTelemetryWorker>(opt, phy);
            g_ui_telemetry = ui_telemetry.get();
        }

        if (opt.mode == "sim") {
            return run_sim(opt, phy, code, rx_decoder_plan);
        }
#ifdef HAVE_GPU_FULL_PIPELINE
        if (opt.mode == "gpu-sim" || opt.mode == "gpu-local") {
            return run_gpu_sim(opt, phy, code, rx_decoder_plan);
        }
#endif
        if (opt.mode == "bench") {
            return run_bench(opt, phy, code, rx_decoder_plan);
        }
        if (opt.mode == "tx") {
            return run_tx(opt, phy, code);
        }
        if (opt.mode == "rx") {
            return run_rx(opt, phy, code, rx_decoder_plan);
        }
        if (opt.mode == "trx-capture" || opt.mode == "capture") {
            return run_trx_capture(opt, phy, code);
        }
        if (opt.mode == "trx") {
            return run_trx(opt, phy, code, rx_decoder_plan);
        }
        if (opt.mode == "replay") {
            return run_replay(opt, phy, code, rx_decoder_plan);
        }

        throw std::runtime_error("unsupported mode: " + opt.mode);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
