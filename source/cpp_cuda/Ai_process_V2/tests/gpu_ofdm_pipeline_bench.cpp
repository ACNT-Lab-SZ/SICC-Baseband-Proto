#include "gpu_ofdm_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using usrp_gpu_pipeline_bridge::Complex32;

constexpr float kInvSqrt2 = 0.7071067811865475f;

struct Options {
    int frames = 4000;
    int warmup = 64;
    int slots = 3;
    int nfft = 1024;
    int cp = 72;
    int num_symbols = 96;
    int active_sc = 768;
    int pre_half_len = 256;
    int pilot_period = 4;
    int bits_per_symbol = 2;
    int code_n = 128;
    int code_k = 64;
    int check_frames = 8;
    float rate = 12.5e6f;
    float amplitude = 0.20f;
    uint32_t seed = 20260513;
};

[[noreturn]] void usage(const char* exe)
{
    std::cerr
        << "Usage: " << exe << " [options]\n\n"
        << "Offline benchmark for the experimental GPU OFDM pipeline.\n\n"
        << "Options:\n"
        << "  --frames <N>          Measured frames. Default: 4000\n"
        << "  --warmup <N>          Warmup frames. Default: 64\n"
        << "  --slots <N>           Pipeline slots. Default: 3\n"
        << "  --nfft <N>            FFT size. Default: 1024\n"
        << "  --cp <N>              Cyclic prefix. Default: 72\n"
        << "  --symbols <N>         OFDM symbols per frame. Default: 96\n"
        << "  --active-sc <N>       Active subcarriers. Default: 768\n"
        << "  --pre-half-len <N>    Repeated-preamble half length. Default: 256\n"
        << "  --pilot-period <N>    Full-band pilot spacing. Default: 4\n"
        << "  --rate <Sps>          Sample rate. Default: 12.5e6\n"
        << "  --check-frames <N>    Frames checked for hard-decision errors. Default: 8\n";
    std::exit(2);
}

std::string require_value(int& i, int argc, char** argv)
{
    if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + argv[i]);
    }
    return argv[++i];
}

Options parse_options(int argc, char** argv)
{
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            usage(argv[0]);
        } else if (a == "--frames") {
            opt.frames = std::stoi(require_value(i, argc, argv));
        } else if (a == "--warmup") {
            opt.warmup = std::stoi(require_value(i, argc, argv));
        } else if (a == "--slots") {
            opt.slots = std::stoi(require_value(i, argc, argv));
        } else if (a == "--nfft") {
            opt.nfft = std::stoi(require_value(i, argc, argv));
        } else if (a == "--cp") {
            opt.cp = std::stoi(require_value(i, argc, argv));
        } else if (a == "--symbols" || a == "--num-symbols") {
            opt.num_symbols = std::stoi(require_value(i, argc, argv));
        } else if (a == "--active-sc") {
            opt.active_sc = std::stoi(require_value(i, argc, argv));
        } else if (a == "--pre-half-len") {
            opt.pre_half_len = std::stoi(require_value(i, argc, argv));
        } else if (a == "--pilot-period") {
            opt.pilot_period = std::stoi(require_value(i, argc, argv));
        } else if (a == "--rate") {
            opt.rate = std::stof(require_value(i, argc, argv));
        } else if (a == "--check-frames") {
            opt.check_frames = std::stoi(require_value(i, argc, argv));
        } else {
            throw std::runtime_error("unknown option: " + a);
        }
    }

    opt.frames = std::max(opt.frames, 1);
    opt.warmup = std::max(opt.warmup, 0);
    opt.slots = std::max(opt.slots, 1);
    opt.check_frames = std::max(opt.check_frames, 0);
    if ((opt.nfft & (opt.nfft - 1)) != 0 || opt.active_sc <= 0 || opt.active_sc >= opt.nfft ||
        (opt.active_sc % 2) != 0 || opt.num_symbols <= 0 || opt.pilot_period <= 0) {
        throw std::runtime_error("invalid OFDM dimensions");
    }
    if (opt.bits_per_symbol != 2 || opt.code_n != 128 || opt.code_k != 64) {
        throw std::runtime_error("this benchmark is currently QPSK + n128/k64 only");
    }
    return opt;
}

std::vector<uint8_t> random_bits(size_t n, std::mt19937& rng)
{
    std::vector<uint8_t> bits(n);
    for (auto& b : bits) {
        b = static_cast<uint8_t>(rng() & 1u);
    }
    return bits;
}

Complex32 qpsk(uint8_t b0, uint8_t b1)
{
    return Complex32(b0 ? -kInvSqrt2 : kInvSqrt2, b1 ? -kInvSqrt2 : kInvSqrt2);
}

std::vector<Complex32> build_preamble(int half_len, std::mt19937& rng)
{
    const auto bits = random_bits(static_cast<size_t>(2 * half_len), rng);
    std::vector<Complex32> half(static_cast<size_t>(half_len));
    for (int i = 0; i < half_len; ++i) {
        half[static_cast<size_t>(i)] = qpsk(bits[static_cast<size_t>(2 * i)], bits[static_cast<size_t>(2 * i + 1)]);
    }
    std::vector<Complex32> pre;
    pre.reserve(static_cast<size_t>(2 * half_len));
    pre.insert(pre.end(), half.begin(), half.end());
    pre.insert(pre.end(), half.begin(), half.end());
    return pre;
}

std::vector<int> build_used_indices(int nfft, int active_sc)
{
    std::vector<int> used;
    used.reserve(static_cast<size_t>(active_sc));
    const int n2 = nfft / 2;
    const int a2 = active_sc / 2;
    for (int i = n2 - a2; i <= n2 - 1; ++i) {
        used.push_back(i);
    }
    for (int i = n2 + 1; i <= n2 + a2; ++i) {
        used.push_back(i);
    }
    return used;
}

std::vector<int> build_pilot_symbols(int num_symbols, int pilot_period)
{
    std::vector<int> pilots;
    for (int s = 0; s < num_symbols; s += pilot_period) {
        pilots.push_back(s);
    }
    return pilots;
}

std::vector<int> build_data_symbols(int num_symbols, const std::vector<int>& pilots)
{
    std::vector<int> data;
    for (int s = 0; s < num_symbols; ++s) {
        if (std::find(pilots.begin(), pilots.end(), s) == pilots.end()) {
            data.push_back(s);
        }
    }
    return data;
}

std::vector<Complex32> build_pilots(int pilot_count, int active_sc, std::mt19937& rng)
{
    const auto bits = random_bits(static_cast<size_t>(2 * pilot_count * active_sc), rng);
    std::vector<Complex32> pilots(static_cast<size_t>(pilot_count * active_sc));
    for (int i = 0; i < pilot_count * active_sc; ++i) {
        pilots[static_cast<size_t>(i)] = qpsk(bits[static_cast<size_t>(2 * i)], bits[static_cast<size_t>(2 * i + 1)]);
    }
    return pilots;
}

std::vector<uint8_t> build_repetition_generator(int k, int n)
{
    std::vector<uint8_t> g(static_cast<size_t>(k * n), 0);
    for (int r = 0; r < k; ++r) {
        g[static_cast<size_t>(r * n + r)] = 1;
        if (r + k < n) {
            g[static_cast<size_t>(r * n + r + k)] = 1;
        }
    }
    return g;
}

usrp_gpu_pipeline_bridge::StaticTables build_tables(const Options& opt, const std::vector<int>& pilots, const std::vector<int>& data)
{
    std::mt19937 rng(opt.seed);
    usrp_gpu_pipeline_bridge::StaticTables t;
    t.generator_kn = build_repetition_generator(opt.code_k, opt.code_n);
    t.used_indices = build_used_indices(opt.nfft, opt.active_sc);
    t.pilot_symbols = pilots;
    t.data_symbols = data;
    t.pilot_symbol_index.assign(static_cast<size_t>(opt.num_symbols), -1);
    t.data_symbol_index.assign(static_cast<size_t>(opt.num_symbols), -1);
    for (size_t i = 0; i < pilots.size(); ++i) {
        t.pilot_symbol_index[static_cast<size_t>(pilots[i])] = static_cast<int>(i);
    }
    for (size_t i = 0; i < data.size(); ++i) {
        t.data_symbol_index[static_cast<size_t>(data[i])] = static_cast<int>(i);
    }

    for (int sym : data) {
        size_t hi = 0;
        while (hi + 1 < pilots.size() && sym > pilots[hi + 1]) {
            ++hi;
        }
        size_t p0 = hi;
        size_t p1 = std::min(hi + 1, pilots.size() - 1);
        if (sym <= pilots.front()) {
            p0 = p1 = 0;
        } else if (sym >= pilots.back()) {
            p0 = p1 = pilots.size() - 1;
        }
        float alpha = 0.0f;
        if (p0 != p1) {
            alpha = static_cast<float>(
                static_cast<double>(sym - pilots[p0]) / static_cast<double>(pilots[p1] - pilots[p0]));
        }
        t.interp_p0.push_back(static_cast<int>(p0));
        t.interp_p1.push_back(static_cast<int>(p1));
        t.interp_alpha.push_back(alpha);
    }

    t.preamble = build_preamble(opt.pre_half_len, rng);
    t.pilot_grid = build_pilots(static_cast<int>(pilots.size()), opt.active_sc, rng);
    return t;
}

usrp_gpu_pipeline_bridge::Config build_config(const Options& opt, int data_symbols)
{
    usrp_gpu_pipeline_bridge::Config cfg;
    cfg.nfft = opt.nfft;
    cfg.cp = opt.cp;
    cfg.preamble_len = 2 * opt.pre_half_len;
    cfg.pre_half_len = opt.pre_half_len;
    cfg.num_symbols = opt.num_symbols;
    cfg.active_sc = opt.active_sc;
    cfg.bits_per_symbol = opt.bits_per_symbol;
    cfg.code_n = opt.code_n;
    cfg.code_k = opt.code_k;
    cfg.blocks_per_frame = (data_symbols * opt.active_sc * opt.bits_per_symbol) / opt.code_n;
    cfg.max_rx_search_samples = cfg.frame_samples();
    cfg.sample_rate = opt.rate;
    cfg.tx_amplitude = opt.amplitude;
    cfg.demod_noise_var = 1.0e-5f;
    return cfg;
}

void fill_info(std::vector<uint8_t>& info, uint32_t frame_id)
{
    std::mt19937 rng(0x51f15eedu ^ frame_id);
    for (auto& b : info) {
        b = static_cast<uint8_t>(rng() & 1u);
    }
    for (int i = 0; i < 32 && i < static_cast<int>(info.size()); ++i) {
        info[static_cast<size_t>(i)] = static_cast<uint8_t>((frame_id >> (31 - i)) & 1u);
    }
}

uint8_t reference_coded_bit(const std::vector<uint8_t>& info, int global_bit, const usrp_gpu_pipeline_bridge::Config& cfg)
{
    const int block = global_bit / cfg.code_n;
    const int col = global_bit - block * cfg.code_n;
    const int src = col % cfg.code_k;
    return info[static_cast<size_t>(block * cfg.code_k + src)];
}

double seconds_between(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b)
{
    return std::max(std::chrono::duration<double>(b - a).count(), 1e-9);
}

void print_stage(const char* name,
                 int frames,
                 double sec,
                 const usrp_gpu_pipeline_bridge::Config& cfg)
{
    const double fps = static_cast<double>(frames) / sec;
    const double info_mbps = fps * static_cast<double>(cfg.info_bits_per_frame()) / 1.0e6;
    const double coded_mbps = fps * static_cast<double>(cfg.coded_bits_per_frame()) / 1.0e6;
    const double sample_msps = fps * static_cast<double>(cfg.frame_samples()) / 1.0e6;
    std::cout << "[GPU-BENCH] " << name
              << " frames=" << frames
              << " sec=" << std::fixed << std::setprecision(3) << sec
              << " fps=" << fps
              << " info=" << info_mbps << " Mbps"
              << " coded=" << coded_mbps << " Mbps"
              << " equivSamples=" << sample_msps << " Msps"
              << " ms/frame=" << (1000.0 / std::max(fps, 1e-9))
              << "\n";
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const Options opt = parse_options(argc, argv);
        const auto pilots = build_pilot_symbols(opt.num_symbols, opt.pilot_period);
        const auto data = build_data_symbols(opt.num_symbols, pilots);
        if (data.empty()) {
            throw std::runtime_error("pilot pattern leaves no data symbols");
        }
        const auto cfg = build_config(opt, static_cast<int>(data.size()));
        if (cfg.blocks_per_frame * cfg.code_n != static_cast<int>(data.size()) * opt.active_sc * opt.bits_per_symbol) {
            throw std::runtime_error("frame coded bits are not an integer number of codewords");
        }
        const auto tables = build_tables(opt, pilots, data);
        usrp_gpu_pipeline_bridge::Pipeline pipe(cfg, opt.slots);
        pipe.upload_static_tables(tables);

        std::vector<std::vector<uint8_t>> info_slots(static_cast<size_t>(opt.slots));
        std::vector<std::vector<Complex32>> frame_slots(static_cast<size_t>(opt.slots));
        std::vector<std::vector<float>> llr_slots(static_cast<size_t>(opt.slots));
        for (int s = 0; s < opt.slots; ++s) {
            info_slots[static_cast<size_t>(s)].resize(static_cast<size_t>(cfg.info_bits_per_frame()));
            frame_slots[static_cast<size_t>(s)].resize(static_cast<size_t>(cfg.frame_samples()));
            llr_slots[static_cast<size_t>(s)].resize(static_cast<size_t>(cfg.coded_bits_per_frame()));
        }

        std::cout << "[GPU-BENCH] config:"
                  << " nfft=" << cfg.nfft
                  << " cp=" << cfg.cp
                  << " symbols=" << cfg.num_symbols
                  << " pilots=" << pilots.size()
                  << " dataSymbols=" << data.size()
                  << " activeSC=" << cfg.active_sc
                  << " frameSamples=" << cfg.frame_samples()
                  << " blocks/frame=" << cfg.blocks_per_frame
                  << " infoBits/frame=" << cfg.info_bits_per_frame()
                  << " codedBits/frame=" << cfg.coded_bits_per_frame()
                  << "\n";

        for (int i = 0; i < opt.warmup; ++i) {
            const int slot = i % opt.slots;
            fill_info(info_slots[static_cast<size_t>(slot)], static_cast<uint32_t>(i));
            pipe.submit_tx_frame_async(slot, info_slots[static_cast<size_t>(slot)].data(), frame_slots[static_cast<size_t>(slot)].data());
            pipe.synchronize_tx(slot);
            pipe.demod_frame_to_host_llr(slot, frame_slots[static_cast<size_t>(slot)].data(), llr_slots[static_cast<size_t>(slot)].data());
        }

        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < opt.frames; ++i) {
            const int slot = i % opt.slots;
            fill_info(info_slots[static_cast<size_t>(slot)], static_cast<uint32_t>(i));
            pipe.submit_tx_frame_async(slot, info_slots[static_cast<size_t>(slot)].data(), frame_slots[static_cast<size_t>(slot)].data());
            pipe.synchronize_tx(slot);
        }
        auto t1 = std::chrono::steady_clock::now();
        const double tx_sec = seconds_between(t0, t1);
        print_stage("tx_encode_map_ifft", opt.frames, tx_sec, cfg);

        const int rx_source_slot = 0;
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < opt.frames; ++i) {
            const int slot = i % opt.slots;
            pipe.demod_frame_to_host_llr(slot, frame_slots[static_cast<size_t>(rx_source_slot)].data(), llr_slots[static_cast<size_t>(slot)].data());
        }
        t1 = std::chrono::steady_clock::now();
        const double rx_sec = seconds_between(t0, t1);
        print_stage("rx_sync_fft_channel_llr", opt.frames, rx_sec, cfg);

        uint64_t hard_errors = 0;
        uint64_t checked_bits = 0;
        const int checks = std::min(opt.check_frames, opt.frames);
        for (int i = 0; i < checks; ++i) {
            const int slot = i % opt.slots;
            fill_info(info_slots[static_cast<size_t>(slot)], static_cast<uint32_t>(i));
            pipe.submit_tx_frame_async(slot, info_slots[static_cast<size_t>(slot)].data(), frame_slots[static_cast<size_t>(slot)].data());
            pipe.synchronize_tx(slot);
            pipe.demod_frame_to_host_llr(slot, frame_slots[static_cast<size_t>(slot)].data(), llr_slots[static_cast<size_t>(slot)].data());
            for (int b = 0; b < cfg.coded_bits_per_frame(); ++b) {
                const uint8_t hard = llr_slots[static_cast<size_t>(slot)][static_cast<size_t>(b)] < 0.0f ? 1u : 0u;
                hard_errors += (hard != reference_coded_bit(info_slots[static_cast<size_t>(slot)], b, cfg)) ? 1u : 0u;
                ++checked_bits;
            }
        }

        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < opt.frames; ++i) {
            const int slot = i % opt.slots;
            fill_info(info_slots[static_cast<size_t>(slot)], static_cast<uint32_t>(i));
            pipe.submit_tx_frame_async(slot, info_slots[static_cast<size_t>(slot)].data(), frame_slots[static_cast<size_t>(slot)].data());
            pipe.synchronize_tx(slot);
            pipe.demod_frame_to_host_llr(slot, frame_slots[static_cast<size_t>(slot)].data(), llr_slots[static_cast<size_t>(slot)].data());
        }
        t1 = std::chrono::steady_clock::now();
        const double roundtrip_sec = seconds_between(t0, t1);
        print_stage("tx_plus_rx_roundtrip", opt.frames, roundtrip_sec, cfg);

        const double hard_ber = checked_bits > 0
            ? static_cast<double>(hard_errors) / static_cast<double>(checked_bits)
            : 0.0;
        std::cout << "[GPU-BENCH] hard_check frames=" << checks
                  << " bits=" << checked_bits
                  << " errors=" << hard_errors
                  << " BER=" << std::scientific << hard_ber << "\n";
        return hard_errors == 0 ? 0 : 3;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
