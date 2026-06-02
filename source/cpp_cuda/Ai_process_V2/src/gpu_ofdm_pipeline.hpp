#pragma once

#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

namespace usrp_gpu_pipeline_bridge {

using Complex32 = std::complex<float>;

struct Config {
    int nfft = 1024;
    int cp = 72;
    int preamble_len = 1024;
    int pre_half_len = 512;
    int num_symbols = 96;
    int active_sc = 768;
    int bits_per_symbol = 2;
    int code_n = 128;
    int code_k = 64;
    int blocks_per_frame = 0;
    int max_rx_search_samples = 0;
    float sample_rate = 12.5e6f;
    float tx_amplitude = 0.20f;
    float demod_noise_var = 1.0e-3f;

    int coded_bits_per_frame() const { return blocks_per_frame * code_n; }
    int info_bits_per_frame() const { return blocks_per_frame * code_k; }
    int data_symbols_per_frame() const { return coded_bits_per_frame() / (active_sc * bits_per_symbol); }
    int symbol_len() const { return nfft + cp; }
    int frame_samples() const { return preamble_len + num_symbols * symbol_len(); }
};

struct StaticTables {
    // Dense row-major generator matrix [code_k][code_n], binary uint8_t.
    // Required for GPU TX encode; optional for RX-only sync/demod.
    // Long DVB-S2/QC encoders should use their compact native kernels instead.
    std::vector<uint8_t> generator_kn;
    std::vector<int> used_indices;
    std::vector<int> pilot_symbol_index;
    std::vector<int> data_symbol_index;
    std::vector<int> pilot_symbols;
    std::vector<int> data_symbols;
    std::vector<int> interp_p0;
    std::vector<int> interp_p1;
    std::vector<float> interp_alpha;
    std::vector<Complex32> pilot_grid;
    std::vector<Complex32> preamble;
    std::vector<Complex32> precomp_h;
};

struct SyncResult {
    int start = 0;
    float metric = 0.0f;
    float cfo_hz = 0.0f;
};

class Pipeline {
public:
    explicit Pipeline(const Config& cfg, int slots = 3);
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    void upload_static_tables(const StaticTables& tables);

    void submit_tx_frame_async(int slot, const uint8_t* h_info_bits, Complex32* h_tx_frame);
    void synchronize_tx(int slot);

    SyncResult find_frame(int slot, const Complex32* h_samples, int sample_count);
    const float* demod_frame_to_device_llr(int slot, const Complex32* h_frame, SyncResult* sync_result = nullptr);
    SyncResult demod_frame_to_host_llr(int slot, const Complex32* h_frame, float* h_llr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace usrp_gpu_pipeline_bridge
