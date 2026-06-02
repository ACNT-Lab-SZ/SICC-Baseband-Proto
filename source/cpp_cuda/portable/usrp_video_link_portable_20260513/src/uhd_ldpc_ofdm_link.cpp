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
#include "rx_decoder_router.hpp"
#ifdef HAVE_GPU_FULL_PIPELINE
#include "gpu_ofdm_pipeline.hpp"
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
    std::string ui_constellation_host = "127.0.0.1";
    std::string ui_spectrum_host = "127.0.0.1";
    std::string ui_metrics_host = "127.0.0.1";
    std::string adaptive_feedback_host = "127.0.0.1";

    double freq = 2.45e9;
    double rate = 15.36e6;
    double master_clock_rate = 184.32e6;
    double tx_gain = 10.0;
    double rx_gain = 15.0;
    double amplitude = 0.70;
    double sync_threshold = 0.45;
    double rx_snr_gate_db = -120.0;
    double sim_snr_db = std::numeric_limits<double>::infinity();
    double ldpc_normalization = 0.80;
    double duration_sec = 0.0;

    int channel = 0;
    int tx_channel = -1;
    int rx_channel = -1;
    int nfft = 1024;
    int cp = 72;
    int num_symbols = 48;
    int active_sc = 512;
    int pre_half_len = 256;
    int pilot_period = 0;
    int ldpc_max_iter = 8;
    int frames = 0;
    int report_every = 20;
    int max_buffered_frames = 4;
    int test_seed = 20260418;
    int cuda_min_batch = 30;
    int cuda_max_batch = 4096;
    int cuda_latency_us = 2000;
    int rx_queue_blocks = 256;
    int rx_frame_queue_frames = 512;
    int rx_block_samps = 32768;
    size_t rx_buffer_samples = 0;
    int radio_oversample = 1;
    int adaptive_feedback_port = 65435;
    int adaptive_window_frames = 20;
    int tx_repeat_min = 1;
    int tx_repeat_max = 3;
    int ui_constellation_port = 65432;
    int ui_constellation_points = 1000;
    int ui_constellation_interval_ms = 50;
    int ui_spectrum_port = 65433;
    int ui_spectrum_interval_ms = 100;
    int ui_metrics_port = 65434;
    int ui_metrics_interval_ms = 200;
    bool verbose = false;
    bool profile_pipeline = false;
    bool loop_file = false;
    bool systematic_front_info = false;
    bool ui_constellation = false;
    bool ui_spectrum = false;
    bool ui_metrics = false;
    bool suppress_error_frames = false;
    bool adaptive = false;
    bool rx_buffered = false;
    bool gpu_pipeline = false;
    bool gpu_tx_baseband = false;
    bool gpu_rx_sync = false;
    bool gpu_rx_demod = false;
};

struct PhyConfig {
    int nfft = 1024;
    int cp = 72;
    int num_symbols = 48;
    int active_sc = 512;
    int pre_half_len = 256;
    std::string modulation = "qpsk";
    int bits_per_symbol = 2;
    int max_buffered_frames = 4;
    double rate = 15.36e6;
    double amplitude = 0.70;
    double sync_threshold = 0.45;
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
    uint64_t bit_errors = 0;
    uint64_t pre_fec_bit_errors = 0;
    uint64_t pre_fec_bits = 0;
    uint64_t info_bits = 0;
    uint64_t sum_iter = 0;
    double sum_snr_db = 0.0;
    double min_snr_db = std::numeric_limits<double>::infinity();
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
        << "  --freq <Hz>                RF center frequency. Default: 2.45e9.\n"
        << "  --rate <Sps>               Baseband sample rate. Default: 15.36e6.\n"
        << "  --mcr <Hz>                 Master clock rate. Default: 184.32e6.\n"
        << "  --frames <N>               Stop after N frames. 0 means unlimited for tx/rx.\n"
        << "  --duration <sec>           Stop after this many seconds. 0 means unlimited.\n"
        << "  --ldpc-iter <N>            Normalized min-sum iterations. Default: 8.\n\n"
        << "  --suppress-error-frames    Do not print every failed frame; useful for long tests.\n\n"
        << "Decoder options:\n"
        << "  --decoder cpu|cuda-bp-osd|cuda-osd  RX decoder. CUDA modes are specialized for CCSDS n128/k64.\n"
        << "  --systematic-front-info    Put LDPC information bits in the first k columns, matching CUDA BP-OSD.\n"
        << "  --cuda-min-batch <N>       CUDA stream decoder minimum batch. Default: 30.\n"
        << "  --cuda-max-batch <N>       CUDA stream decoder maximum batch. Default: 4096.\n"
        << "  --cuda-latency-us <N>      CUDA stream decoder latency budget. Default: 2000.\n\n"
        << "RX threading options:\n"
        << "  --rx-queue-blocks <N>      Sample blocks buffered between UHD recv and demod. Default: 256.\n\n"
        << "  --rx-frame-queue <N>       OFDM frames buffered between sync and decoder. Default: 512.\n\n"
        << "  --rx-block-samps <N>       Samples requested per UHD recv call. Default: 32768.\n\n"
        << "  --rx-buffered              Use a continuous local RX IQ buffer before online sync.\n"
        << "  --rx-buffer-samples <N>    Capacity of --rx-buffered in complex samples. Default: about 2 s.\n\n"
        << "  --radio-oversample <N>     Integer TX upsample/RX downsample factor before UHD. Default: 1.\n\n"
        << "  --rx-snr-gate-db <dB>      Skip FEC decode below this estimated SNR. Default: disabled.\n\n"
        << "  --sim-snr-db <dB>          Add AWGN in --mode sim at the requested SNR. Default: disabled.\n\n"
        << "Channel injection options:\n"
        << "  --precomp-channel <paths>  Comma/semicolon separated predictor channel file(s) for TX precompensation.\n"
        << "  --actual-channel <paths>   Comma/semicolon separated actual channel file(s) injected after TX precompensation.\n"
        << "                             File format: P, then P rows of h k delay or h_real h_imag k delay.\n\n"
        << "  --profile-pipeline        Print RX subsystem throughput and per-frame timing.\n\n"
        << "Experimental GPU pipeline options (--mode trx, build with ENABLE_GPU_FULL_PIPELINE=ON):\n"
        << "  --gpu-pipeline            Enable GPU TX baseband, GPU RX sync, and GPU RX demod/LLR.\n"
        << "  --gpu-tx-baseband         Move LDPC encode, modulation, precompensation, and OFDM IFFT to CUDA.\n"
        << "  --gpu-rx-sync             Use CUDA Schmidl-Cox frame acquisition in the sync thread.\n"
        << "  --gpu-rx-demod            Move RX CFO/FFT/channel/LLR to CUDA; FEC decode still uses selected decoder.\n\n"
        << "Adaptive reliability options:\n"
        << "  --adaptive                Enable RX feedback and TX frame repetition control.\n"
        << "  --adaptive-feedback-host <ip>  Feedback UDP target/listen host. Default: 127.0.0.1.\n"
        << "  --adaptive-feedback-port <N>   Feedback UDP port. Default: 65435.\n"
        << "  --adaptive-window-frames <N>   RX decision window. Default: 20.\n"
        << "  --tx-repeat-min <N>       Minimum media frame repetitions. Default: 1.\n"
        << "  --tx-repeat-max <N>       Maximum media frame repetitions. Default: 3.\n\n"
        << "UI telemetry options:\n"
        << "  --ui-constellation        Send MATLAB-compatible constellation UDP packets to the UI.\n"
        << "  --ui-host <ip>            UI UDP target. Default: 127.0.0.1.\n"
        << "  --ui-port <N>             UI UDP port. Default: 65432.\n"
        << "  --ui-points <N>           Max constellation points per packet. Default: 1000.\n"
        << "  --ui-interval-ms <N>      Minimum UDP interval. Default: 50 ms.\n\n"
        << "  --ui-spectrum             Send spectrum UDP packets to the UI.\n"
        << "  --ui-spectrum-port <N>    Spectrum UDP port. Default: 65433.\n"
        << "  --ui-spectrum-interval-ms <N>  Minimum spectrum interval. Default: 100 ms.\n"
        << "  --ui-metrics              Send BER/FER/SNR UDP packets to the UI.\n"
        << "  --ui-metrics-port <N>     Metrics UDP port. Default: 65434.\n"
        << "  --ui-metrics-interval-ms <N>   Minimum metrics interval. Default: 200 ms.\n\n"
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

static Options parse_options(int argc, char** argv)
{
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
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
        } else if (a == "--ui-host") {
            opt.ui_constellation_host = require_value(i, argc, argv);
            opt.ui_spectrum_host = opt.ui_constellation_host;
            opt.ui_metrics_host = opt.ui_constellation_host;
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
        } else if (a == "--precomp-channel" || a == "--precomp-channel-files" ||
                   a == "--predicted-channel" || a == "--tx-precomp-channel") {
            opt.precomp_channel_files = require_value(i, argc, argv);
        } else if (a == "--actual-channel" || a == "--actual-channel-files" ||
                   a == "--inject-channel" || a == "--channel-model") {
            opt.actual_channel_files = require_value(i, argc, argv);
        } else if (a == "--ldpc-normalization") {
            opt.ldpc_normalization = std::stod(require_value(i, argc, argv));
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
        } else if (a == "--ldpc-iter") {
            opt.ldpc_max_iter = std::stoi(require_value(i, argc, argv));
        } else if (a == "--frames") {
            opt.frames = std::stoi(require_value(i, argc, argv));
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
        } else if (a == "--rx-buffered" || a == "--online-buffered") {
            opt.rx_buffered = true;
        } else if (a == "--rx-buffer-samples") {
            opt.rx_buffer_samples = static_cast<size_t>(std::stoull(require_value(i, argc, argv)));
        } else if (a == "--radio-oversample" || a == "--oversample") {
            opt.radio_oversample = std::stoi(require_value(i, argc, argv));
        } else if (a == "--adaptive") {
            opt.adaptive = true;
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
        } else if (a == "--verbose") {
            opt.verbose = true;
        } else if (a == "--profile-pipeline") {
            opt.profile_pipeline = true;
        } else if (a == "--suppress-error-frames") {
            opt.suppress_error_frames = true;
        } else if (a == "--gpu-pipeline" || a == "--gpu-full-pipeline") {
            opt.gpu_pipeline = true;
            opt.gpu_tx_baseband = true;
            opt.gpu_rx_sync = true;
            opt.gpu_rx_demod = true;
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
    opt.rx_frame_queue_frames = std::max(opt.rx_frame_queue_frames, 4);
    opt.rx_block_samps = std::max(opt.rx_block_samps, 1024);
    if (opt.rx_buffer_samples == 0) {
        opt.rx_buffer_samples = static_cast<size_t>(std::max(4.0 * opt.rate, 8.0 * 1024.0 * 1024.0));
    }
    opt.rx_snr_gate_db = std::max(opt.rx_snr_gate_db, -120.0);
    opt.radio_oversample = std::max(opt.radio_oversample, 1);
    opt.adaptive_feedback_port = std::max(opt.adaptive_feedback_port, 1);
    opt.adaptive_window_frames = std::max(opt.adaptive_window_frames, 1);
    opt.tx_repeat_min = std::max(opt.tx_repeat_min, 1);
    opt.tx_repeat_max = std::max(opt.tx_repeat_max, opt.tx_repeat_min);
    opt.ui_constellation_port = std::max(opt.ui_constellation_port, 1);
    opt.ui_constellation_points = std::max(opt.ui_constellation_points, 1);
    opt.ui_constellation_interval_ms = std::max(opt.ui_constellation_interval_ms, 1);
    opt.ui_spectrum_port = std::max(opt.ui_spectrum_port, 1);
    opt.ui_spectrum_interval_ms = std::max(opt.ui_spectrum_interval_ms, 1);
    opt.ui_metrics_port = std::max(opt.ui_metrics_port, 1);
    opt.ui_metrics_interval_ms = std::max(opt.ui_metrics_interval_ms, 1);
#ifndef HAVE_GPU_FULL_PIPELINE
    if (opt.gpu_pipeline || opt.gpu_tx_baseband || opt.gpu_rx_sync || opt.gpu_rx_demod) {
        throw std::runtime_error("GPU full-pipeline options require configuring with -DENABLE_GPU_FULL_PIPELINE=ON");
    }
#endif
    const bool gpu_mode_ok =
        opt.mode == "trx" ||
        opt.mode == "trx-capture" ||
        opt.mode == "replay";
    if ((opt.gpu_pipeline || opt.gpu_tx_baseband || opt.gpu_rx_sync || opt.gpu_rx_demod) &&
        !gpu_mode_ok) {
        throw std::runtime_error("GPU full-pipeline stages are wired into --mode trx, trx-capture, and replay");
    }
    return opt;
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

private:
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
        const auto pos = msg.find("repeat=");
        if (pos == std::string::npos) {
            return;
        }
        int requested = 0;
        try {
            requested = std::stoi(msg.substr(pos + 7));
        } catch (...) {
            return;
        }
        requested = std::max(min_repeat_, std::min(max_repeat_, requested));
        const int old = repeat_.exchange(requested, std::memory_order_acq_rel);
        if (requested != old) {
            std::cout << "[ADAPT-TX] repeat " << old << " -> " << requested
                      << " feedback=\"" << msg.substr(0, msg.find('\n')) << "\"\n";
        }
    }

    int min_repeat_ = 1;
    int max_repeat_ = 3;
    int port_ = 65435;
    std::atomic<int> repeat_{1};
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
    double ber = 0.0;
    double fer = 0.0;
    double preber = 0.0;
    double snr_db = 0.0;
    double goodput_mbps = 0.0;
};

class UiTelemetryWorker {
public:
    UiTelemetryWorker(const Options& opt, const PhyConfig& phy)
    {
        constellation_interval_ns_ = interval_to_ns(std::chrono::milliseconds(std::max(opt.ui_constellation_interval_ms, 1)));
        spectrum_interval_ns_ = interval_to_ns(std::chrono::milliseconds(std::max(opt.ui_spectrum_interval_ms, 1)));
        metrics_interval_ns_ = interval_to_ns(std::chrono::milliseconds(std::max(opt.ui_metrics_interval_ms, 1)));
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
            const double df = phy.rate / static_cast<double>(phy.nfft);
            spectrum_freq_hz_.resize(static_cast<size_t>(phy.active_sc), 0.0f);
            for (int sc = 0; sc < phy.active_sc; ++sc) {
                spectrum_freq_hz_[static_cast<size_t>(sc)] =
                    static_cast<float>((phy.used_indices[sc] - phy.nfft / 2) * df);
            }
        }
        if (opt.ui_metrics) {
            metrics_sink_ = std::make_unique<UdpMetricsSink>(
                opt.ui_metrics_host,
                opt.ui_metrics_port,
                opt.ui_metrics_interval_ms);
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
            metrics_sink_ != nullptr,
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

    void submit_spectrum_symbol(std::vector<cf32> active_carriers)
    {
        if (!spectrum_sink_ || active_carriers.empty()) {
            finish_capture(next_spectrum_capture_ns_, spectrum_capture_pending_, spectrum_interval_ns_);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            latest_spectrum_symbol_ = std::move(active_carriers);
            has_spectrum_ = true;
        }
        cv_.notify_one();
    }

    void submit_metrics(const UiMetricsSample& sample)
    {
        if (!metrics_sink_) {
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

    void run()
    {
        while (true) {
            std::vector<cf32> tx_points;
            std::vector<cf32> rx_points;
            std::vector<cf32> spectrum_symbol;
            UiMetricsSample metrics;
            bool send_constellation = false;
            bool send_spectrum = false;
            bool send_metrics = false;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait_for(lock, std::chrono::milliseconds(5), [this]() {
                    return stop_ || has_constellation_ || has_spectrum_ || has_metrics_;
                });
                if (stop_) {
                    break;
                }
                if (constellation_sink_ && has_constellation_) {
                    tx_points = latest_tx_points_;
                    rx_points = latest_rx_points_;
                    has_constellation_ = false;
                    send_constellation = true;
                }
                if (spectrum_sink_ && has_spectrum_) {
                    spectrum_symbol = latest_spectrum_symbol_;
                    has_spectrum_ = false;
                    send_spectrum = true;
                }
                if (metrics_sink_ && has_metrics_) {
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
                std::vector<float> psd_db(spectrum_symbol.size(), 0.0f);
                for (size_t i = 0; i < spectrum_symbol.size(); ++i) {
                    psd_db[i] = static_cast<float>(
                        10.0 * std::log10(std::max(std::norm(spectrum_symbol[i]), 1e-12f)));
                }
                spectrum_sink_->send(spectrum_freq_hz_, psd_db);
                finish_capture(next_spectrum_capture_ns_, spectrum_capture_pending_, spectrum_interval_ns_);
            }
            if (send_metrics) {
                metrics_sink_->send(
                    metrics.frames,
                    metrics.ber,
                    metrics.fer,
                    metrics.preber,
                    metrics.snr_db,
                    metrics.goodput_mbps);
                finish_capture(next_metrics_capture_ns_, metrics_capture_pending_, metrics_interval_ns_);
            }
        }
    }

    std::unique_ptr<UdpConstellationSink> constellation_sink_;
    std::unique_ptr<UdpSpectrumSink> spectrum_sink_;
    std::unique_ptr<UdpMetricsSink> metrics_sink_;
    int constellation_points_ = 0;
    int64_t constellation_interval_ns_ = 50'000'000;
    int64_t spectrum_interval_ns_ = 100'000'000;
    int64_t metrics_interval_ns_ = 200'000'000;
    std::vector<float> spectrum_freq_hz_;
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
    bool has_constellation_ = false;
    bool has_spectrum_ = false;
    bool has_metrics_ = false;
    std::vector<cf32> latest_tx_points_;
    std::vector<cf32> latest_rx_points_;
    std::vector<cf32> latest_spectrum_symbol_;
    UiMetricsSample latest_metrics_;
    std::atomic<int64_t> next_constellation_capture_ns_{0};
    std::atomic<int64_t> next_spectrum_capture_ns_{0};
    std::atomic<int64_t> next_metrics_capture_ns_{0};
    std::atomic<bool> constellation_capture_pending_{false};
    std::atomic<bool> spectrum_capture_pending_{false};
    std::atomic<bool> metrics_capture_pending_{false};
};

static UiTelemetryWorker* g_ui_telemetry = nullptr;

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

struct CudaBpOsdDecoder {
    unsigned long long handle = 0;
    unsigned long long osd_handle = 0;
    int n = 0;
    int k = 0;
    uint64_t frame_sequence = 0;
    bool osd_only = false;

    CudaBpOsdDecoder(const Options& opt, const LdpcCode& code)
        : n(code.n), k(code.k)
    {
#ifdef HAVE_CUDA_OSD
        osd_only = (opt.decoder == "cuda-osd");
        if (code.n != 128 || code.k != 64 || code.m != 64) {
            throw std::runtime_error("CUDA short-code decoder is specialized for CCSDS n128/k64; use the n128 alist");
        }

        if (osd_only) {
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
        cfg.bp_offset = 0.15f;
        cfg.bp_damping = 0.15f;
        cfg.bp_min_abs_llr_accept = 1.5f;
        cfg.osd_threads = 256;

        const int status = cuda_osd_create_stream_decoder(&cfg, &handle);
        if (status != CUDA_OSD_STATUS_OK || handle == 0) {
            throw std::runtime_error(std::string("cuda_osd_create_stream_decoder failed: ") +
                                     cuda_osd_get_last_error());
        }
        std::cout << "[CUDA] BP-OSD decoder enabled: n=" << code.n
                  << " k=" << code.k
                  << " minBatch=" << cfg.min_batch_size
                  << " maxBatch=" << cfg.max_batch_size
                  << " latencyUs=" << cfg.max_latency_us << "\n";
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
        cfg.offset = 0.15f;
        cfg.damping = 0.15f;
        cfg.min_abs_llr_accept = 0.0f;
        cfg.nnz = dvb->h.nnz;

        const int status = cuda_osd_create_bp_csr_decoder(
            &cfg,
            dvb->h.row_ptr.data(),
            dvb->h.col_ind.data(),
            &handle);
        if (status != CUDA_OSD_STATUS_OK || handle == 0) {
            throw std::runtime_error(std::string("cuda_osd_create_bp_csr_decoder failed: ") +
                                     cuda_osd_get_last_error());
        }
        std::cout << "[CUDA] generic BP decoder enabled: n=" << code.n
                  << " k=" << code.k
                  << " m=" << code.m
                  << " nnz=" << dvb->h.nnz
                  << " maxBatch=" << cfg.max_batch_size
                  << " maxIter=" << cfg.max_iterations << "\n";
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
    cfg.max_buffered_frames = opt.max_buffered_frames;
    cfg.rate = opt.rate;
    cfg.amplitude = opt.amplitude;
    cfg.sync_threshold = opt.sync_threshold;
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

    cfg.preamble = build_preamble(cfg.pre_half_len);
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

constexpr uint32_t kMediaMagic = 0x31444956u; // "VID1" in little-endian byte order.
constexpr uint16_t kMediaVersion = 1;
constexpr uint16_t kMediaHeaderBytes = 48;
constexpr uint32_t kMediaFlagFirst = 1u << 0;
constexpr uint32_t kMediaFlagEof = 1u << 1;

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

    std::vector<uint8_t> build_info_bits(uint32_t frame_id) const
    {
        if (done(frame_id)) {
            throw std::runtime_error("media packetizer asked for a frame past EOF");
        }

        const uint64_t chunk_index = loop ? (static_cast<uint64_t>(frame_id) % total_chunks) : static_cast<uint64_t>(frame_id);
        const uint64_t offset = chunk_index * payload_bytes_per_frame;
        const size_t payload_len = static_cast<size_t>(std::min<uint64_t>(payload_bytes_per_frame, file_bytes.size() - offset));

        std::vector<uint8_t> info(info_bytes_per_frame, 0);
        std::vector<uint8_t> payload(payload_len, 0);
        if (payload_len > 0) {
            std::copy(
                file_bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                file_bytes.begin() + static_cast<std::ptrdiff_t>(offset + payload_len),
                payload.begin());
        }

        uint32_t flags = 0;
        if (chunk_index == 0) {
            flags |= kMediaFlagFirst;
        }
        if (chunk_index + 1 == total_chunks) {
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
        received_chunks = 0;
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

        const size_t chunk_index = static_cast<size_t>(h.offset / chunk_bytes);
        if (chunk_index < received.size() && !received[chunk_index]) {
            received[chunk_index] = 1;
            ++received_chunks;
        }
        ++accepted_packets;
        const bool was_complete = complete;
        complete = received_chunks == received.size();
        if ((!was_complete && complete) || (h.flags & kMediaFlagEof)) {
            out.flush();
        }
        if ((!was_complete && complete) || (accepted_packets % 50 == 0) || (h.flags & kMediaFlagEof)) {
            const double pct = received.empty() ? 100.0 : 100.0 * static_cast<double>(received_chunks) / static_cast<double>(received.size());
            std::cout << "[MEDIA] accepted=" << accepted_packets
                      << " chunks=" << received_chunks << "/" << received.size()
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

static void print_injection_channel(const char* label, const InjectionChannel& channel)
{
    if (channel.empty()) {
        return;
    }
    std::cout << "[CHANNEL] " << label
              << " files=" << channel.source_files.size()
              << " paths=" << channel.paths.size() << "\n";
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

    if (g_ui_telemetry != nullptr &&
        !phy.data_symbols.empty() &&
        g_ui_telemetry->try_begin_spectrum_capture()) {
        const int spectrum_sym = phy.data_symbols[phy.data_symbols.size() / 2];
        g_ui_telemetry->submit_spectrum_symbol(y[static_cast<size_t>(spectrum_sym)]);
    }

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

    std::vector<cf32> yd;
    std::vector<cf32> hd;
    yd.reserve(static_cast<size_t>(phy.active_sc * phy.data_symbols.size()));
    hd.reserve(static_cast<size_t>(phy.active_sc * phy.data_symbols.size()));

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
    const auto t_channel1 = std::chrono::steady_clock::now();

    double measured_noise_var = 0.0;
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
        const size_t tracking_window = static_cast<size_t>(std::max(256, 4 * phy.cp));
        found = find_frame_vector_earliest(rxbuf, base, phy, tracking_window, sr);
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
    const double goodput = static_cast<double>(stats.ok) * static_cast<double>(info_bits_per_frame) /
                           std::max(elapsed, 1e-9) / 1e6;
    const double fps = static_cast<double>(stats.detected) / std::max(elapsed, 1e-9);
    const double avg_iter = static_cast<double>(stats.sum_iter) / std::max<uint64_t>(stats.detected, 1);
    const double avg_snr = stats.sum_snr_db / static_cast<double>(std::max<uint64_t>(stats.detected, 1));
    const double min_snr = std::isfinite(stats.min_snr_db) ? stats.min_snr_db : 0.0;

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
              << " avgIter=" << avg_iter
              << "\n";
}

static void send_ui_metrics_if_due(
    const RxStats& stats,
    double elapsed,
    int info_bits_per_frame,
    const FrameDecodeResult& last_frame)
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
    const double goodput = static_cast<double>(stats.ok) * static_cast<double>(info_bits_per_frame) /
                           std::max(elapsed, 1e-9) / 1e6;
    UiMetricsSample sample;
    sample.frames = static_cast<double>(stats.detected);
    sample.ber = ber;
    sample.fer = fer;
    sample.preber = preber;
    sample.snr_db = last_frame.snr_db;
    sample.goodput_mbps = goodput;
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

static bool accept_media_decode(FrameDecodeResult& r, MediaReassembler& media, size_t payload_capacity)
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
    cfg.tx_channel = static_cast<size_t>(std::max(opt.tx_channel >= 0 ? opt.tx_channel : opt.channel, 0));
    cfg.rx_channel = static_cast<size_t>(std::max(opt.rx_channel >= 0 ? opt.rx_channel : opt.channel, 0));
    return cfg;
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

static gpu_pipe::Config make_gpu_pipeline_config(const PhyConfig& phy, const LdpcCode& code)
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
    cfg.tx_amplitude = static_cast<float>(phy.amplitude);
    cfg.demod_noise_var = 1.0e-3f;
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
    bool include_tx_generator)
{
    gpu_pipe::StaticTables tables;
    if (include_tx_generator) {
        tables.generator_kn = build_dense_generator_kn_for_gpu(code);
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

    return tables;
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
        pipeline_.upload_static_tables(build_gpu_static_tables(phy_, code_, precomp_channel, true));
        std::cout << "[GPU-PIPE] TX baseband enabled: encode+map+precomp+OFDM on CUDA";
        if (actual_channel_ != nullptr && !actual_channel_->empty()) {
            std::cout << " + actual-channel injection";
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
        if (actual_channel_ != nullptr && !actual_channel_->empty()) {
            apply_ofdm_channel_transform(frame, phy_, *actual_channel_, false);
        }
        return upsample_for_radio(frame, phy_.radio_oversample);
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
        std::cout << "[GPU-PIPE] RX sync enabled: Schmidl-Cox acquisition on CUDA\n";
    }

    bool extract(std::vector<cf32>& rxbuf, size_t& base, ExtractedFrame& frame, SyncState& sync_state, SyncStats& sync_stats)
    {
        if (rxbuf.size() < base || rxbuf.size() - base < static_cast<size_t>(phy_.preamble.size())) {
            return false;
        }

        const size_t available = rxbuf.size() - base;
        const int max_samples = std::max(phy_.frame_len(), phy_.max_buffered_frames * phy_.frame_len());
        const int sample_count = static_cast<int>(std::min<size_t>(available, static_cast<size_t>(max_samples)));
        const int slot = next_slot_++ % 2;
        const auto result = pipeline_.find_frame(slot, rxbuf.data() + static_cast<std::ptrdiff_t>(base), sample_count);
        if (result.metric < static_cast<float>(phy_.sync_threshold)) {
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

private:
    const PhyConfig& phy_;
    gpu_pipe::Pipeline pipeline_;
    int next_slot_ = 0;
};

class GpuRxDemodChain {
public:
    GpuRxDemodChain(const PhyConfig& phy, const LdpcCode& code, const InjectionChannel* precomp_channel)
        : phy_(phy),
          code_(code),
          pipeline_(make_gpu_pipeline_config(phy, code), 3)
    {
        pipeline_.upload_static_tables(build_gpu_static_tables(phy_, code_, precomp_channel, false));
        std::cout << "[GPU-PIPE] RX demod enabled: GPU sync+CFO+FFT+channel+LLR, host FEC API bridge\n";
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
        const int slot = next_slot_++ % 3;
        if (cuda_decoder != nullptr && !cuda_decoder->osd_only && cuda_bp_decoder == nullptr) {
            const auto t0 = std::chrono::steady_clock::now();
            gpu_pipe::SyncResult sync;
            const float* device_llr = pipeline_.demod_frame_to_device_llr(slot, frame.samples.data(), &sync);
            const auto t1 = std::chrono::steady_clock::now();

            int used_osd_count = 0;
            const int blocks = phy_.coded_bits_per_frame() / code_.n;
            const auto t_fec0 = std::chrono::steady_clock::now();
            std::vector<uint8_t> info_bits = cuda_decoder->decode_device(device_llr, blocks, used_osd_count);
            const auto t_fec1 = std::chrono::steady_clock::now();

            return finish_decode_from_info_bits(
                std::move(info_bits),
                phy_,
                code_,
                used_osd_count,
                true,
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

private:
    const PhyConfig& phy_;
    const LdpcCode& code_;
    gpu_pipe::Pipeline pipeline_;
    int next_slot_ = 0;
};
#endif

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

    uint32_t frame_id = 0;
    uint64_t sent_frames = 0;
    uint64_t unique_frames = 0;
    auto t0 = std::chrono::steady_clock::now();
    const size_t max_samps = std::max<size_t>(radio.tx_max_samps(), 1024);
    bool first_packet = true;
    std::unique_ptr<AdaptiveTxController> adaptive_tx;
    if (opt.adaptive && file_mode) {
        adaptive_tx = std::make_unique<AdaptiveTxController>(opt);
    }
    int last_repeat_report = adaptive_tx ? adaptive_tx->repeat_count() : 1;

    while (true) {
        if (file_mode && media->done(unique_frames)) {
            break;
        }

        std::vector<cf32> frame;
        if (file_mode) {
            const auto info = media->build_info_bits(frame_id);
            frame = baseband.build_media_frame(info);
        } else {
            frame = baseband.build_test_frame(frame_id);
        }
        const int repeat = adaptive_tx ? adaptive_tx->repeat_count() : 1;
        if (repeat != last_repeat_report) {
            std::cout << "[TX] adaptive repeat=" << repeat << "\n";
            last_repeat_report = repeat;
        }
        for (int rep = 0; rep < repeat; ++rep) {
            size_t off = 0;
            while (off < frame.size()) {
                const size_t n = std::min(max_samps, frame.size() - off);
                size_t sent = 0;
                radio.send(frame.data() + off, n, first_packet, false, 1.0, sent);
                if (sent != n) {
                    std::cerr << "[TX] short send: " << sent << "/" << n << "\n";
                }
                first_packet = false;
                off += sent;
                if (sent == 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
            ++sent_frames;
        }

        if (sent_frames % static_cast<uint64_t>(std::max(opt.report_every, 1)) == 0) {
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - t0).count();
            std::cout << "[TX] frame=" << frame_id
                      << " sent=" << sent_frames
                      << " unique=" << unique_frames
                      << " repeat=" << repeat
                      << " info=" << (static_cast<double>(unique_frames) * static_cast<double>(info_bits) / std::max(elapsed, 1e-9) / 1e6)
                      << " Mbps "
                      << " elapsed=" << std::fixed << std::setprecision(2) << elapsed << " s\n";
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
    if (opt.adaptive && file_mode) {
        adaptive_feedback = std::make_unique<AdaptiveFeedbackSender>(
            opt.adaptive_feedback_host,
            opt.adaptive_feedback_port);
        adaptive_decision = std::make_unique<AdaptiveRxDecision>(opt);
    }

    auto radio = usrp_link::RadioEndpoint::open_rx(radio_config_for_mode(opt, "rx"));
    const auto pp = radio.pp_string();
    if (!pp.empty()) {
        std::cout << "RX ready: " << pp << "\n";
    }

    radio.start_rx();

    const size_t chunk = std::max<size_t>(radio.rx_max_samps(), static_cast<size_t>(opt.rx_block_samps));
    RxSampleQueue sample_queue(static_cast<size_t>(opt.rx_queue_blocks));
    RxFrameQueue frame_queue(static_cast<size_t>(opt.rx_frame_queue_frames));
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
                radio.recv(tmp.data(), tmp.size(), 0.1, n, status);
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
                sample_queue.push(std::move(block));
            }
        } catch (...) {
            rx_thread_exception = std::current_exception();
            rx_thread_failed.store(true, std::memory_order_release);
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

            while (!rx_stop.load(std::memory_order_relaxed)) {
                RxSampleBlock block;
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

                rxbuf.insert(rxbuf.end(), block.samples.begin(), block.samples.end());

                ExtractedFrame frame;
                while (extract_rx_frame_vector(rxbuf, rxbase, phy, frame, sync_state, sync_stats)) {
                    frame_queue.push(std::move(frame));
                    frame = ExtractedFrame{};
                }
            }
            frame_queue.close();
        } catch (...) {
            sync_thread_exception = std::current_exception();
            sync_thread_failed.store(true, std::memory_order_release);
            frame_queue.close();
        }
    });

    RxStats stats;

    auto t0 = std::chrono::steady_clock::now();
    bool stop_now = false;
    uint64_t last_reported_frame_drops = 0;
    while (true) {
        ExtractedFrame frame;
        if (!frame_queue.pop_for(frame, std::chrono::milliseconds(100))) {
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - t0).count();
            if (opt.duration_sec > 0.0 && elapsed >= opt.duration_sec) {
                break;
            }
            if (rx_thread_failed.load(std::memory_order_acquire) ||
                sync_thread_failed.load(std::memory_order_acquire)) {
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

        FrameDecodeResult r = decode_frame(
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
            !file_mode);
        if (file_mode) {
            accept_media_decode(r, *media, payload_capacity);
        }
        update_stats(stats, r, info_bits, phy.coded_bits_per_frame());
        if (adaptive_feedback && adaptive_decision) {
            int repeat = 1;
            double avg_snr = 0.0;
            double fer = 0.0;
            int frames = 0;
            if (adaptive_decision->observe(r, repeat, avg_snr, fer, frames)) {
                adaptive_feedback->send(repeat, avg_snr, fer, frames);
                std::cout << "[ADAPT-RX] window=" << frames
                          << " avgSNR=" << std::fixed << std::setprecision(1) << avg_snr
                          << " FER=" << std::scientific << fer
                          << " recommendRepeat=" << repeat << "\n";
            }
        }
        {
            const auto now = std::chrono::steady_clock::now();
            send_ui_metrics_if_due(stats, std::chrono::duration<double>(now - t0).count(), info_bits, r);
        }
        if (opt.verbose ||
            stats.detected % static_cast<uint64_t>(std::max(opt.report_every, 1)) == 0 ||
            (!opt.suppress_error_frames && !r.frame_ok)) {
            print_frame_result("[RX]", r);
        }
        if (stats.detected % static_cast<uint64_t>(std::max(opt.report_every, 1)) == 0) {
            const auto now = std::chrono::steady_clock::now();
            print_stats(stats, std::chrono::duration<double>(now - t0).count(), info_bits);
        }
        if (file_mode && media->complete && !opt.loop_file) {
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
        if (opt.frames > 0 && static_cast<int>(stats.detected) >= opt.frames) {
            break;
        }
        if (opt.duration_sec > 0.0 && elapsed >= opt.duration_sec) {
            break;
        }
    }

    rx_stop.store(true, std::memory_order_relaxed);
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
    print_stats(stats, std::chrono::duration<double>(t1 - t0).count(), info_bits);
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
    radio.start_rx();

    const size_t rx_chunk = std::max<size_t>(radio.rx_max_samps(), static_cast<size_t>(opt.rx_block_samps));
    const size_t tx_chunk = std::max<size_t>(radio.tx_max_samps(), 1024);
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
                    size_t off = 0;
                    while (off < frame.size() && !stop.load(std::memory_order_relaxed)) {
                        const size_t n = std::min(tx_chunk, frame.size() - off);
                        size_t sent = 0;
                        radio.send(frame.data() + off, n, first_packet, false, 1.0, sent);
                        first_packet = false;
                        off += sent;
                        if (sent == 0) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        }
                    }
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
    const size_t payload_capacity = file_mode ? media_payload_capacity_bytes(phy, code) : 0;

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

    radio.start_rx();

    const size_t rx_chunk = std::max<size_t>(radio.rx_max_samps(), static_cast<size_t>(opt.rx_block_samps));
    const size_t tx_chunk = std::max<size_t>(radio.tx_max_samps(), 1024);
    RxSampleQueue sample_queue(static_cast<size_t>(opt.rx_queue_blocks));
    std::unique_ptr<OnlineIqBuffer> iq_buffer;
    if (opt.rx_buffered) {
        iq_buffer = std::make_unique<OnlineIqBuffer>(opt.rx_buffer_samples);
        std::cout << "[RX-BUFFER] enabled capacity=" << opt.rx_buffer_samples
                  << " samples approx=" << (static_cast<double>(opt.rx_buffer_samples) / std::max(phy.rate, 1.0))
                  << " s\n";
    }
    RxFrameQueue frame_queue(static_cast<size_t>(opt.rx_frame_queue_frames));
    PipelineProfile profile;
    std::atomic<bool> stop{false};
    std::atomic<bool> rx_thread_failed{false};
    std::atomic<bool> sync_thread_failed{false};
    std::atomic<bool> tx_thread_failed{false};
    std::atomic<bool> tx_done{false};
    std::atomic<int> adaptive_repeat{std::max(opt.tx_repeat_min, 1)};
    std::exception_ptr rx_thread_exception;
    std::exception_ptr sync_thread_exception;
    std::exception_ptr tx_thread_exception;

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
                if (iq_buffer && block.status == UHD_C_RX_OK && !block.samples.empty()) {
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
                if (iq_buffer) {
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
                rxbuf.insert(rxbuf.end(), block.samples.begin(), block.samples.end());

                ExtractedFrame frame;
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
                const auto t_sync1 = std::chrono::steady_clock::now();
                if (opt.profile_pipeline) {
                    profile.sync_blocks.fetch_add(1, std::memory_order_relaxed);
                    profile.sync_input_samples.fetch_add(static_cast<uint64_t>(block.samples.size()), std::memory_order_relaxed);
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
                                  << " sample_queue_depth=" << (iq_buffer ? 0 : sample_queue.size())
                                  << " iq_buffer_samples=" << (iq_buffer ? iq_buffer->size() : 0)
                                  << " iq_buffer_dropped=" << (iq_buffer ? iq_buffer->dropped_samples() : 0)
                                  << " frame_queue_depth=" << frame_queue.size()
                                  << " frame_queue_dropped=" << frame_queue.dropped_frames()
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
    if (opt.adaptive && file_mode) {
        trx_adaptive_decision = std::make_unique<AdaptiveRxDecision>(opt);
        std::cout << "[ADAPT-TRX] enabled: repeat=" << opt.tx_repeat_min
                  << ".." << opt.tx_repeat_max << "\n";
    }
    std::thread tx_thread([&] {
        try {
            usrp_link::set_realtime_priority();
            uint32_t frame_id = 0;
            uint64_t sent_frames = 0;
            uint64_t unique_frames = 0;
            bool first_packet = true;
            int last_repeat_report = adaptive_repeat.load(std::memory_order_acquire);
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

                const int repeat = opt.adaptive && file_mode
                    ? adaptive_repeat.load(std::memory_order_acquire)
                    : 1;
                if (repeat != last_repeat_report) {
                    std::cout << "[TRX-TX] adaptive repeat=" << repeat << "\n";
                    last_repeat_report = repeat;
                }
                for (int rep = 0; rep < repeat && !stop.load(std::memory_order_relaxed); ++rep) {
                    size_t off = 0;
                    while (off < frame.size() && !stop.load(std::memory_order_relaxed)) {
                        const size_t n = std::min(tx_chunk, frame.size() - off);
                        size_t sent = 0;
                        radio.send(frame.data() + off, n, first_packet, false, 1.0, sent);
                        if (sent != n) {
                            std::cerr << "[TRX-TX] short send: " << sent << "/" << n << "\n";
                        }
                        first_packet = false;
                        off += sent;
                        if (sent == 0) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        }
                    }
                    ++sent_frames;
                }

                if (opt.profile_pipeline) {
                    profile.tx_frames.store(sent_frames, std::memory_order_relaxed);
                }
                if (sent_frames % static_cast<uint64_t>(std::max(opt.report_every, 1)) == 0) {
                    const auto now = std::chrono::steady_clock::now();
                    const double elapsed = std::chrono::duration<double>(now - t0).count();
                    std::cout << "[TRX-TX] frame=" << frame_id
                              << " sent=" << sent_frames
                              << " unique=" << unique_frames
                              << " repeat=" << repeat
                              << " info=" << (static_cast<double>(unique_frames) * static_cast<double>(info_bits) / std::max(elapsed, 1e-9) / 1e6)
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
                std::cerr << "[TRX-TX] " << e.what() << "\n";
            }
            tx_done.store(true, std::memory_order_release);
        } catch (...) {
            tx_thread_exception = std::current_exception();
            tx_thread_failed.store(true, std::memory_order_release);
            tx_done.store(true, std::memory_order_release);
        }
    });

    RxStats stats;
    bool stop_now = false;
    uint64_t last_reported_frame_drops = 0;
    while (true) {
        ExtractedFrame frame;
        if (!frame_queue.pop_for(frame, std::chrono::milliseconds(100))) {
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - t0).count();
            if (opt.duration_sec > 0.0 && elapsed >= opt.duration_sec) {
                break;
            }
            if ((tx_done.load(std::memory_order_acquire) && opt.duration_sec <= 0.0) ||
                rx_thread_failed.load(std::memory_order_acquire) ||
                sync_thread_failed.load(std::memory_order_acquire) ||
                tx_thread_failed.load(std::memory_order_acquire)) {
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
                !file_mode,
                opt.profile_pipeline ? &profile.decode : nullptr);
        }
        if (file_mode) {
            const auto media_t0 = std::chrono::steady_clock::now();
            const bool media_ok = accept_media_decode(r, *rx_media, payload_capacity);
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
        update_stats(stats, r, info_bits, phy.coded_bits_per_frame());
        if (trx_adaptive_decision) {
            int repeat = 1;
            double avg_snr = 0.0;
            double fer = 0.0;
            int frames = 0;
            if (trx_adaptive_decision->observe(r, repeat, avg_snr, fer, frames)) {
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
            send_ui_metrics_if_due(stats, std::chrono::duration<double>(now - t0).count(), info_bits, r);
        }
        if (opt.verbose ||
            stats.detected % static_cast<uint64_t>(std::max(opt.report_every, 1)) == 0 ||
            (!opt.suppress_error_frames && !r.frame_ok)) {
            print_frame_result("[TRX-RX]", r);
        }
        if (stats.detected % static_cast<uint64_t>(std::max(opt.report_every, 1)) == 0) {
            const auto now = std::chrono::steady_clock::now();
            print_stats(stats, std::chrono::duration<double>(now - t0).count(), info_bits);
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
    if (iq_buffer) {
        iq_buffer->close();
    }
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
    if (rx_thread.joinable()) {
        rx_thread.join();
    }
    if (sync_thread.joinable()) {
        sync_thread.join();
    }
    if (tx_thread_exception) {
        std::rethrow_exception(tx_thread_exception);
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
    std::mt19937 noise_rng(static_cast<uint32_t>(opt.test_seed) ^ 0x5eed1234u);
    std::normal_distribution<float> noise_dist(0.0f, 1.0f);

    auto add_awgn_if_requested = [&](std::vector<cf32>& samples) -> std::optional<double> {
        if (!std::isfinite(opt.sim_snr_db) || samples.empty()) {
            return std::nullopt;
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
        const Options opt = parse_options(argc, argv);
        if (opt.traffic_mode != "test" && opt.traffic_mode != "file") {
            throw std::runtime_error("unsupported traffic mode: " + opt.traffic_mode);
        }
        if (opt.decoder != "cpu" && opt.decoder != "cuda-bp-osd" && opt.decoder != "cuda-osd" && opt.decoder != "cuda-bp") {
            throw std::runtime_error("unsupported decoder: " + opt.decoder);
        }
        fec::MatrixLoader fec_loader;
        fec::FecConfigStore fec_store;
        fec_store.swap(fec_loader.load(fec::MatrixLoadRequest::dvb_s2_alist(
            opt.alist_path,
            0,
            "cli-alist")));
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
        if (opt.ui_constellation || opt.ui_spectrum || opt.ui_metrics) {
            ui_telemetry = std::make_unique<UiTelemetryWorker>(opt, phy);
            g_ui_telemetry = ui_telemetry.get();
        }

        if (opt.mode == "sim") {
            return run_sim(opt, phy, code, rx_decoder_plan);
        }
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
