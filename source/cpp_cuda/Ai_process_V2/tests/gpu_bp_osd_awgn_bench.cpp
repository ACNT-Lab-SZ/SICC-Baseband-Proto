#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef HAVE_CUDA_OSD
#include "matlab_decoder_c_api.h"
#endif

namespace {

struct Options {
    int n = 128;
    int k = 64;
    int m = 64;
    double snr_start_db = 0.0;
    double snr_stop_db = 7.0;
    double snr_step_db = 1.0;
    int codewords = 100000;
    int warmup_codewords = 4096;
    int batch_size = 4096;
    int min_batch = 4096;
    int max_batch = 4096;
    int bp_max_iter = 20;
    float bp_normalization = 0.80f;
    float bp_offset = 0.15f;
    float bp_damping = 0.15f;
    float bp_min_abs_llr_accept = 1.5f;
    int osd_threads = 256;
    uint32_t seed = 1;
    std::string csv_path;
    std::string decoder = "osd-only";
};

[[noreturn]] void usage(const char* exe)
{
    std::cerr
        << "Usage: " << exe << " [options]\n\n"
        << "GPU BP-OSD AWGN benchmark for the CCSDS LDPC n=128/k=64 decoder.\n"
        << "The transmitted codeword is the all-zero linear-code codeword; BER/FER are\n"
        << "measured on the decoded information bits.\n\n"
        << "Options:\n"
        << "  --snr-start <dB>       Default: 0\n"
        << "  --snr-stop <dB>        Default: 7\n"
        << "  --snr-step <dB>        Default: 1\n"
        << "  --codewords <N>        Measured codewords per SNR. Default: 100000\n"
        << "  --warmup <N>           Warmup codewords before timing. Default: 4096\n"
        << "  --batch <N>            Submit/poll batch size. Default: 4096\n"
        << "  --min-batch <N>        Stream decoder min batch. Default: batch\n"
        << "  --max-batch <N>        Stream decoder max batch. Default: batch\n"
        << "  --bp-iter <N>          BP max iterations. Default: 20\n"
        << "  --bp-norm <x>          BP normalization. Default: 0.80\n"
        << "  --seed <N>             RNG seed. Default: 1\n"
        << "  --decoder <mode>       osd-only or stream-bp-osd. Default: osd-only\n"
        << "  --csv <path>           Also write CSV results.\n";
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
    bool min_batch_set = false;
    bool max_batch_set = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            usage(argv[0]);
        } else if (a == "--snr-start") {
            opt.snr_start_db = std::stod(require_value(i, argc, argv));
        } else if (a == "--snr-stop") {
            opt.snr_stop_db = std::stod(require_value(i, argc, argv));
        } else if (a == "--snr-step") {
            opt.snr_step_db = std::stod(require_value(i, argc, argv));
        } else if (a == "--codewords") {
            opt.codewords = std::stoi(require_value(i, argc, argv));
        } else if (a == "--warmup") {
            opt.warmup_codewords = std::stoi(require_value(i, argc, argv));
        } else if (a == "--batch") {
            opt.batch_size = std::stoi(require_value(i, argc, argv));
        } else if (a == "--min-batch") {
            opt.min_batch = std::stoi(require_value(i, argc, argv));
            min_batch_set = true;
        } else if (a == "--max-batch") {
            opt.max_batch = std::stoi(require_value(i, argc, argv));
            max_batch_set = true;
        } else if (a == "--bp-iter") {
            opt.bp_max_iter = std::stoi(require_value(i, argc, argv));
        } else if (a == "--bp-norm") {
            opt.bp_normalization = std::stof(require_value(i, argc, argv));
        } else if (a == "--seed") {
            opt.seed = static_cast<uint32_t>(std::stoul(require_value(i, argc, argv)));
        } else if (a == "--decoder") {
            opt.decoder = require_value(i, argc, argv);
        } else if (a == "--csv") {
            opt.csv_path = require_value(i, argc, argv);
        } else {
            throw std::runtime_error("unknown option: " + a);
        }
    }
    if (!min_batch_set) {
        opt.min_batch = opt.batch_size;
    }
    if (!max_batch_set) {
        opt.max_batch = opt.batch_size;
    }
    if (opt.snr_step_db <= 0.0 || opt.snr_stop_db < opt.snr_start_db) {
        throw std::runtime_error("invalid SNR range");
    }
    if (opt.codewords <= 0 || opt.warmup_codewords < 0 || opt.batch_size <= 0 ||
        opt.min_batch <= 0 || opt.max_batch <= 0 || opt.min_batch > opt.max_batch) {
        throw std::runtime_error("invalid batch/codeword settings");
    }
    if (opt.n != 128 || opt.k != 64 || opt.m != 64) {
        throw std::runtime_error("this benchmark is specialized for n=128/k=64/m=64");
    }
    if (opt.decoder != "osd-only" && opt.decoder != "stream-bp-osd") {
        throw std::runtime_error("--decoder must be osd-only or stream-bp-osd");
    }
    return opt;
}

double normal_sample(std::mt19937& rng)
{
    static thread_local bool has_spare = false;
    static thread_local double spare = 0.0;
    if (has_spare) {
        has_spare = false;
        return spare;
    }

    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double u1 = uni(rng);
    double u2 = uni(rng);
    u1 = std::max(u1, 1e-12);
    const double mag = std::sqrt(-2.0 * std::log(u1));
    const double a = 6.28318530717958647692 * u2;
    spare = mag * std::sin(a);
    has_spare = true;
    return mag * std::cos(a);
}

void fill_awgn_llrs(std::vector<float>& llrs, int codewords, const Options& opt, double ebn0_db, std::mt19937& rng)
{
    const double rate = static_cast<double>(opt.k) / static_cast<double>(opt.n);
    const double ebn0 = std::pow(10.0, ebn0_db / 10.0);
    const double sigma2 = 1.0 / (2.0 * rate * ebn0);
    const double sigma = std::sqrt(sigma2);
    const double llr_scale = 2.0 / sigma2;

    llrs.resize(static_cast<size_t>(codewords * opt.n));
    for (float& v : llrs) {
        const double y = 1.0 + sigma * normal_sample(rng);
        v = static_cast<float>(llr_scale * y);
    }
}

#ifdef HAVE_CUDA_OSD
class StreamDecoder {
public:
    explicit StreamDecoder(const Options& opt)
        : k_(opt.k)
    {
        cuda_osd_stream_config_t cfg{};
        cfg.n = opt.n;
        cfg.k = opt.k;
        cfg.m = opt.m;
        cfg.max_batch_size = opt.max_batch;
        cfg.min_batch_size = opt.min_batch;
        cfg.max_latency_us = 1;
        cfg.arrival_ewma_alpha = 0.15;
        cfg.default_interarrival_us = 1;
        cfg.bp_max_iterations = opt.bp_max_iter;
        cfg.bp_normalization = opt.bp_normalization;
        cfg.bp_offset = opt.bp_offset;
        cfg.bp_damping = opt.bp_damping;
        cfg.bp_min_abs_llr_accept = opt.bp_min_abs_llr_accept;
        cfg.osd_threads = opt.osd_threads;

        const int status = cuda_osd_create_stream_decoder(&cfg, &handle_);
        if (status != CUDA_OSD_STATUS_OK || handle_ == 0) {
            throw std::runtime_error(std::string("cuda_osd_create_stream_decoder failed: ") +
                                     cuda_osd_get_last_error());
        }
    }

    StreamDecoder(const StreamDecoder&) = delete;
    StreamDecoder& operator=(const StreamDecoder&) = delete;

    ~StreamDecoder()
    {
        if (handle_ != 0) {
            cuda_osd_destroy_stream_decoder(handle_);
        }
    }

    std::vector<unsigned char> decode(const float* llrs, int codewords, int& used_osd)
    {
        std::vector<unsigned long long> timestamps(static_cast<size_t>(codewords), 0);
        std::vector<unsigned long long> frame_ids(static_cast<size_t>(codewords), 0);
        std::vector<unsigned int> block_indices(static_cast<size_t>(codewords), 0);
        for (int i = 0; i < codewords; ++i) {
            frame_ids[static_cast<size_t>(i)] = frame_sequence_;
            block_indices[static_cast<size_t>(i)] = static_cast<unsigned int>(i);
        }
        ++frame_sequence_;

        int status = cuda_osd_submit_codewords(
            handle_,
            llrs,
            codewords,
            timestamps.data(),
            frame_ids.data(),
            block_indices.data());
        if (status != CUDA_OSD_STATUS_OK) {
            throw std::runtime_error(std::string("cuda_osd_submit_codewords failed: ") +
                                     cuda_osd_get_last_error());
        }
        status = cuda_osd_flush(handle_);
        if (status != CUDA_OSD_STATUS_OK) {
            throw std::runtime_error(std::string("cuda_osd_flush failed: ") +
                                     cuda_osd_get_last_error());
        }

        std::vector<unsigned char> info(static_cast<size_t>(codewords * k_), 0);
        std::vector<unsigned char> tmp_info(static_cast<size_t>(codewords * k_), 0);
        std::vector<unsigned long long> out_timestamps(static_cast<size_t>(codewords), 0);
        std::vector<unsigned long long> out_frame_ids(static_cast<size_t>(codewords), 0);
        std::vector<unsigned int> out_block_indices(static_cast<size_t>(codewords), 0);
        std::vector<unsigned char> out_used_osd(static_cast<size_t>(codewords), 0);

        int decoded = 0;
        used_osd = 0;
        while (decoded < codewords) {
            int ready = 0;
            status = cuda_osd_poll_ready(
                handle_,
                codewords,
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
                throw std::runtime_error("cuda_osd_poll_ready returned no codewords after flush");
            }
            for (int i = 0; i < ready; ++i) {
                const unsigned int b = out_block_indices[static_cast<size_t>(i)];
                if (b >= static_cast<unsigned int>(codewords)) {
                    throw std::runtime_error("decoder returned an invalid block index");
                }
                std::copy(
                    tmp_info.begin() + static_cast<std::ptrdiff_t>(i * k_),
                    tmp_info.begin() + static_cast<std::ptrdiff_t>((i + 1) * k_),
                    info.begin() + static_cast<std::ptrdiff_t>(b * k_));
                used_osd += out_used_osd[static_cast<size_t>(i)] ? 1 : 0;
            }
            decoded += ready;
        }
        return info;
    }

private:
    unsigned long long handle_ = 0;
    unsigned long long frame_sequence_ = 0;
    int k_ = 0;
};

class OsdOnlyDecoder {
public:
    explicit OsdOnlyDecoder(const Options& opt)
        : k_(opt.k)
    {
        cuda_osd_osd_config_t cfg{};
        cfg.n = opt.n;
        cfg.k = opt.k;
        cfg.max_batch_size = opt.max_batch;
        cfg.osd_threads = opt.osd_threads;

        const int status = cuda_osd_create_osd_decoder(&cfg, &handle_);
        if (status != CUDA_OSD_STATUS_OK || handle_ == 0) {
            throw std::runtime_error(std::string("cuda_osd_create_osd_decoder failed: ") +
                                     cuda_osd_get_last_error());
        }
    }

    OsdOnlyDecoder(const OsdOnlyDecoder&) = delete;
    OsdOnlyDecoder& operator=(const OsdOnlyDecoder&) = delete;

    ~OsdOnlyDecoder()
    {
        if (handle_ != 0) {
            cuda_osd_destroy_osd_decoder(handle_);
        }
    }

    std::vector<unsigned char> decode(const float* llrs, int codewords, int& used_osd)
    {
        std::vector<unsigned char> info(static_cast<size_t>(codewords * k_), 0);
        std::vector<float> distances(static_cast<size_t>(codewords), 0.0f);
        double total_decode_ms = 0.0;
        double throughput_mbps = 0.0;
        const int status = cuda_osd_decode_osd_batch(
            handle_,
            llrs,
            codewords,
            info.data(),
            distances.data(),
            &total_decode_ms,
            &throughput_mbps);
        if (status != CUDA_OSD_STATUS_OK) {
            throw std::runtime_error(std::string("cuda_osd_decode_osd_batch failed: ") +
                                     cuda_osd_get_last_error());
        }
        used_osd = codewords;
        return info;
    }

private:
    unsigned long long handle_ = 0;
    int k_ = 0;
};
#endif

struct Result {
    double snr_db = 0.0;
    int codewords = 0;
    uint64_t bit_errors = 0;
    int frame_errors = 0;
    int used_osd = 0;
    double decode_seconds = 0.0;
    double ber = 0.0;
    double fer = 0.0;
    double info_mbps = 0.0;
    double coded_mbps = 0.0;
    double codewords_per_sec = 0.0;
};

void print_header()
{
    std::cout << "snr_db,codewords,bit_errors,frame_errors,BER,FER,info_Mbps,coded_Mbps,cw_per_s,used_osd,decode_seconds\n";
}

void print_result(const Result& r)
{
    std::cout << std::fixed << std::setprecision(3)
              << r.snr_db << ","
              << r.codewords << ","
              << r.bit_errors << ","
              << r.frame_errors << ","
              << std::scientific << std::setprecision(6)
              << r.ber << ","
              << r.fer << ","
              << std::fixed << std::setprecision(3)
              << r.info_mbps << ","
              << r.coded_mbps << ","
              << r.codewords_per_sec << ","
              << r.used_osd << ","
              << r.decode_seconds << "\n";
}

void write_csv_header(std::ofstream& out)
{
    out << "snr_db,codewords,bit_errors,frame_errors,BER,FER,info_Mbps,coded_Mbps,cw_per_s,used_osd,decode_seconds\n";
}

void write_csv_result(std::ofstream& out, const Result& r)
{
    out << std::fixed << std::setprecision(3)
        << r.snr_db << ","
        << r.codewords << ","
        << r.bit_errors << ","
        << r.frame_errors << ","
        << std::scientific << std::setprecision(6)
        << r.ber << ","
        << r.fer << ","
        << std::fixed << std::setprecision(3)
        << r.info_mbps << ","
        << r.coded_mbps << ","
        << r.codewords_per_sec << ","
        << r.used_osd << ","
        << r.decode_seconds << "\n";
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const Options opt = parse_options(argc, argv);
#ifndef HAVE_CUDA_OSD
        (void)opt;
        throw std::runtime_error("gpu_bp_osd_awgn_bench was built without HAVE_CUDA_OSD");
#else
        std::mt19937 rng(opt.seed);
        StreamDecoder stream_decoder(opt);
        OsdOnlyDecoder osd_decoder(opt);

        std::ofstream csv;
        if (!opt.csv_path.empty()) {
            csv.open(opt.csv_path, std::ios::out | std::ios::trunc);
            if (!csv) {
                throw std::runtime_error("cannot open CSV output: " + opt.csv_path);
            }
            write_csv_header(csv);
        }

        std::vector<float> llrs;
        if (opt.warmup_codewords > 0) {
            int used_osd = 0;
            fill_awgn_llrs(llrs, opt.warmup_codewords, opt, opt.snr_stop_db, rng);
            if (opt.decoder == "osd-only") {
                (void)osd_decoder.decode(llrs.data(), opt.warmup_codewords, used_osd);
            } else {
                (void)stream_decoder.decode(llrs.data(), opt.warmup_codewords, used_osd);
            }
        }

        std::cout << "# GPU BP-OSD AWGN benchmark: n=" << opt.n
                  << " k=" << opt.k
                  << " decoder=" << opt.decoder
                  << " codewords/SNR=" << opt.codewords
                  << " batch=" << opt.batch_size
                  << " bpIter=" << opt.bp_max_iter
                  << " bpNorm=" << opt.bp_normalization
                  << "\n";
        print_header();

        for (double snr = opt.snr_start_db; snr <= opt.snr_stop_db + 1e-9; snr += opt.snr_step_db) {
            Result total;
            total.snr_db = snr;
            int remaining = opt.codewords;
            while (remaining > 0) {
                const int batch = std::min(opt.batch_size, remaining);
                fill_awgn_llrs(llrs, batch, opt, snr, rng);
                int used_osd = 0;
                const auto t0 = std::chrono::steady_clock::now();
                const auto info = (opt.decoder == "osd-only")
                                      ? osd_decoder.decode(llrs.data(), batch, used_osd)
                                      : stream_decoder.decode(llrs.data(), batch, used_osd);
                const auto t1 = std::chrono::steady_clock::now();
                total.decode_seconds += std::chrono::duration<double>(t1 - t0).count();

                total.used_osd += used_osd;
                total.codewords += batch;
                for (int b = 0; b < batch; ++b) {
                    int errs = 0;
                    for (int i = 0; i < opt.k; ++i) {
                        errs += info[static_cast<size_t>(b * opt.k + i)] ? 1 : 0;
                    }
                    total.bit_errors += static_cast<uint64_t>(errs);
                    total.frame_errors += (errs != 0) ? 1 : 0;
                }
                remaining -= batch;
            }
            total.ber = static_cast<double>(total.bit_errors) /
                        static_cast<double>(static_cast<uint64_t>(total.codewords) * static_cast<uint64_t>(opt.k));
            total.fer = static_cast<double>(total.frame_errors) / static_cast<double>(total.codewords);
            total.codewords_per_sec = static_cast<double>(total.codewords) / total.decode_seconds;
            total.info_mbps = total.codewords_per_sec * static_cast<double>(opt.k) / 1.0e6;
            total.coded_mbps = total.codewords_per_sec * static_cast<double>(opt.n) / 1.0e6;

            print_result(total);
            if (csv) {
                write_csv_result(csv, total);
            }
        }
        return 0;
#endif
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
