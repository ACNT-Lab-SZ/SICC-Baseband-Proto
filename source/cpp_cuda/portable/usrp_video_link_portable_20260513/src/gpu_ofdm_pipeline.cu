// Experimental GPU-resident OFDM/FEC pipeline.
//
// This file is intentionally kept independent from uhd_ldpc_ofdm_link.cpp so the
// current hardware path can stay stable while we move hot stages onto the GPU.
// UHD still requires host-visible buffers at the radio boundary; everything
// between "info bits in" and "baseband samples out", and between "samples in"
// and "decoder LLRs out", is designed to run on CUDA streams.

#include "gpu_ofdm_pipeline.hpp"

#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace usrp_gpu_pipeline {

namespace {

constexpr float kPiF = 3.14159265358979323846f;
constexpr int kSyncThreads = 256;

inline void check_cuda(cudaError_t status, const char* what)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

inline void check_cufft(cufftResult status, const char* what)
{
    if (status != CUFFT_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": cuFFT error " + std::to_string(static_cast<int>(status)));
    }
}

__device__ __forceinline__ cuFloatComplex cadd(cuFloatComplex a, cuFloatComplex b)
{
    return make_cuFloatComplex(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ cuFloatComplex csub(cuFloatComplex a, cuFloatComplex b)
{
    return make_cuFloatComplex(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__ cuFloatComplex cmul(cuFloatComplex a, cuFloatComplex b)
{
    return make_cuFloatComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ cuFloatComplex cconj(cuFloatComplex a)
{
    return make_cuFloatComplex(a.x, -a.y);
}

__device__ __forceinline__ cuFloatComplex cscale(cuFloatComplex a, float s)
{
    return make_cuFloatComplex(a.x * s, a.y * s);
}

__device__ __forceinline__ float cnorm(cuFloatComplex a)
{
    return a.x * a.x + a.y * a.y;
}

__device__ __forceinline__ int gray_to_binary_dev(int g)
{
    int b = 0;
    for (; g != 0; g >>= 1) {
        b ^= g;
    }
    return b;
}

__device__ __forceinline__ cuFloatComplex qam_from_bits_dev(const uint8_t* bits, int bits_per_symbol)
{
    if (bits_per_symbol == 1) {
        return make_cuFloatComplex(bits[0] ? -1.0f : 1.0f, 0.0f);
    }
    if (bits_per_symbol == 2) {
        constexpr float s = 0.7071067811865475f;
        return make_cuFloatComplex(bits[0] ? -s : s, bits[1] ? -s : s);
    }

    const int axis_bits = bits_per_symbol / 2;
    const int levels = 1 << axis_bits;
    int gi = 0;
    int gq = 0;
    for (int i = 0; i < axis_bits; ++i) {
        gi = (gi << 1) | static_cast<int>(bits[i] != 0);
        gq = (gq << 1) | static_cast<int>(bits[axis_bits + i] != 0);
    }
    const int bi = gray_to_binary_dev(gi);
    const int bq = gray_to_binary_dev(gq);
    const float i = static_cast<float>((levels - 1) - 2 * bi);
    const float q = static_cast<float>((levels - 1) - 2 * bq);
    const float scale = (bits_per_symbol == 4) ? 0.3162277660168379f : 0.1543033499620919f;
    return make_cuFloatComplex(i * scale, q * scale);
}

__device__ __forceinline__ uint8_t constellation_bit_dev(int symbol_idx, int bit, int bits_per_symbol)
{
    return static_cast<uint8_t>((symbol_idx >> (bits_per_symbol - 1 - bit)) & 1);
}

__device__ __forceinline__ cuFloatComplex qam_symbol_by_index_dev(int symbol_idx, int bits_per_symbol)
{
    uint8_t bits[6]{};
    for (int i = 0; i < bits_per_symbol; ++i) {
        bits[i] = static_cast<uint8_t>((symbol_idx >> (bits_per_symbol - 1 - i)) & 1);
    }
    return qam_from_bits_dev(bits, bits_per_symbol);
}

__device__ __forceinline__ int shifted_to_natural_bin_dev(int shifted_bin, int nfft)
{
    return (shifted_bin + nfft / 2) % nfft;
}

__global__ void ldpc_encode_generator_kernel(
    const uint8_t* __restrict__ info_bits,
    const uint8_t* __restrict__ generator_kn,
    uint8_t* __restrict__ coded_bits,
    int blocks,
    int k,
    int n)
{
    const int bit = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = blocks * n;
    if (bit >= total) {
        return;
    }
    const int cw = bit / n;
    const int out_col = bit - cw * n;
    uint8_t acc = 0;
    const uint8_t* info = info_bits + cw * k;
    for (int row = 0; row < k; ++row) {
        acc ^= static_cast<uint8_t>((info[row] & 1u) & (generator_kn[row * n + out_col] & 1u));
    }
    coded_bits[bit] = acc;
}

__global__ void fill_resource_grid_kernel(
    const uint8_t* __restrict__ coded_bits,
    const int* __restrict__ used_indices,
    const int* __restrict__ pilot_symbol_index,
    const int* __restrict__ data_symbol_index,
    const cuFloatComplex* __restrict__ pilot_grid,
    const cuFloatComplex* __restrict__ precomp_h,
    cuFloatComplex* __restrict__ grid,
    int nfft,
    int num_symbols,
    int active_sc,
    int bits_per_symbol)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_symbols * active_sc;
    if (idx >= total) {
        return;
    }

    const int sym = idx / active_sc;
    const int sc = idx - sym * active_sc;
    const int fft_bin = shifted_to_natural_bin_dev(used_indices[sc], nfft);
    const int pilot_idx = pilot_symbol_index[sym];

    cuFloatComplex value = make_cuFloatComplex(0.0f, 0.0f);
    if (pilot_idx >= 0) {
        value = pilot_grid[pilot_idx * active_sc + sc];
    } else {
        const int data_idx = data_symbol_index[sym];
        if (data_idx >= 0) {
            const int sym_idx = data_idx * active_sc + sc;
            value = qam_from_bits_dev(coded_bits + sym_idx * bits_per_symbol, bits_per_symbol);
        }
    }

    if (precomp_h != nullptr) {
        const cuFloatComplex h = precomp_h[sym * active_sc + sc];
        const float h2 = fmaxf(cnorm(h), 1.0e-10f);
        value = cmul(value, cscale(cconj(h), 1.0f / h2));
    }
    grid[sym * nfft + fft_bin] = value;
}

__global__ void compose_tx_frame_kernel(
    const cuFloatComplex* __restrict__ preamble,
    const cuFloatComplex* __restrict__ ifft_symbols,
    cuFloatComplex* __restrict__ frame,
    int preamble_len,
    int nfft,
    int cp,
    int num_symbols,
    float amplitude)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int sym_len = nfft + cp;
    const int frame_len = preamble_len + num_symbols * sym_len;
    if (idx >= frame_len) {
        return;
    }
    if (idx < preamble_len) {
        frame[idx] = cscale(preamble[idx], amplitude);
        return;
    }

    const int body = idx - preamble_len;
    const int sym = body / sym_len;
    const int pos = body - sym * sym_len;
    const int src = (pos < cp) ? (nfft - cp + pos) : (pos - cp);
    frame[idx] = cscale(ifft_symbols[sym * nfft + src], amplitude / static_cast<float>(nfft));
}

__global__ void schmidl_cox_metric_kernel(
    const cuFloatComplex* __restrict__ samples,
    float* __restrict__ metric,
    cuFloatComplex* __restrict__ corr,
    int candidates,
    int half_len)
{
    const int d = blockIdx.x;
    if (d >= candidates) {
        return;
    }

    __shared__ float p_re[kSyncThreads];
    __shared__ float p_im[kSyncThreads];
    __shared__ float e1[kSyncThreads];
    __shared__ float e2[kSyncThreads];

    float re = 0.0f;
    float im = 0.0f;
    float a = 0.0f;
    float b = 0.0f;
    for (int i = threadIdx.x; i < half_len; i += blockDim.x) {
        const cuFloatComplex x0 = samples[d + i];
        const cuFloatComplex x1 = samples[d + i + half_len];
        const cuFloatComplex p = cmul(cconj(x0), x1);
        re += p.x;
        im += p.y;
        a += cnorm(x0);
        b += cnorm(x1);
    }

    p_re[threadIdx.x] = re;
    p_im[threadIdx.x] = im;
    e1[threadIdx.x] = a;
    e2[threadIdx.x] = b;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            p_re[threadIdx.x] += p_re[threadIdx.x + stride];
            p_im[threadIdx.x] += p_im[threadIdx.x + stride];
            e1[threadIdx.x] += e1[threadIdx.x + stride];
            e2[threadIdx.x] += e2[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const cuFloatComplex p = make_cuFloatComplex(p_re[0], p_im[0]);
        corr[d] = p;
        metric[d] = cnorm(p) / fmaxf(e1[0] * e2[0], 1.0e-20f);
    }
}

struct DeviceSyncResult {
    int start;
    float metric;
    float cfo_hz;
};

__global__ void argmax_metric_kernel(
    const float* __restrict__ metric,
    const cuFloatComplex* __restrict__ corr,
    DeviceSyncResult* __restrict__ result,
    int candidates,
    int half_len,
    float sample_rate)
{
    __shared__ float best_metric[kSyncThreads];
    __shared__ int best_index[kSyncThreads];

    float local_metric = -1.0f;
    int local_index = 0;
    for (int i = threadIdx.x; i < candidates; i += blockDim.x) {
        const float m = metric[i];
        if (m > local_metric) {
            local_metric = m;
            local_index = i;
        }
    }

    best_metric[threadIdx.x] = local_metric;
    best_index[threadIdx.x] = local_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && best_metric[threadIdx.x + stride] > best_metric[threadIdx.x]) {
            best_metric[threadIdx.x] = best_metric[threadIdx.x + stride];
            best_index[threadIdx.x] = best_index[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const int start = best_index[0];
        const cuFloatComplex p = corr[start];
        const float phase = atan2f(p.y, p.x);
        result->start = start;
        result->metric = best_metric[0];
        result->cfo_hz = phase * sample_rate / (2.0f * kPiF * static_cast<float>(half_len));
    }
}

__global__ void cfo_strip_cp_kernel(
    const cuFloatComplex* __restrict__ samples,
    const DeviceSyncResult* __restrict__ sync,
    cuFloatComplex* __restrict__ fft_in,
    int preamble_len,
    int nfft,
    int cp,
    int num_symbols,
    float sample_rate)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_symbols * nfft;
    if (idx >= total) {
        return;
    }

    const int sym = idx / nfft;
    const int k = idx - sym * nfft;
    const int sym_len = nfft + cp;
    const int sample_index = sync->start + preamble_len + sym * sym_len + cp + k;
    const float phase = -2.0f * kPiF * sync->cfo_hz * static_cast<float>(sample_index) / sample_rate;
    const cuFloatComplex rot = make_cuFloatComplex(cosf(phase), sinf(phase));
    fft_in[idx] = cmul(samples[sample_index], rot);
}

__global__ void estimate_pilot_channel_kernel(
    const cuFloatComplex* __restrict__ fft_symbols,
    const int* __restrict__ used_indices,
    const int* __restrict__ pilot_symbols,
    const cuFloatComplex* __restrict__ pilot_grid,
    cuFloatComplex* __restrict__ pilot_h,
    int nfft,
    int pilot_count,
    int active_sc)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = pilot_count * active_sc;
    if (idx >= total) {
        return;
    }
    const int p = idx / active_sc;
    const int sc = idx - p * active_sc;
    const int sym = pilot_symbols[p];
    const int fft_bin = shifted_to_natural_bin_dev(used_indices[sc], nfft);
    const cuFloatComplex y = fft_symbols[sym * nfft + fft_bin];
    const cuFloatComplex pilot = pilot_grid[p * active_sc + sc];
    const float den = fmaxf(cnorm(pilot), 1.0e-9f);
    pilot_h[idx] = cscale(cmul(y, cconj(pilot)), 1.0f / den);
}

__global__ void equalize_llr_kernel(
    const cuFloatComplex* __restrict__ fft_symbols,
    const cuFloatComplex* __restrict__ pilot_h,
    const int* __restrict__ used_indices,
    const int* __restrict__ data_symbols,
    const int* __restrict__ interp_p0,
    const int* __restrict__ interp_p1,
    const float* __restrict__ interp_alpha,
    float* __restrict__ llr,
    int nfft,
    int data_symbol_count,
    int active_sc,
    int bits_per_symbol,
    float noise_var)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = data_symbol_count * active_sc;
    if (idx >= total) {
        return;
    }

    const int ds = idx / active_sc;
    const int sc = idx - ds * active_sc;
    const int sym = data_symbols[ds];
    const int p0 = interp_p0[ds];
    const int p1 = interp_p1[ds];
    const float alpha = interp_alpha[ds];

    const cuFloatComplex h0 = pilot_h[p0 * active_sc + sc];
    const cuFloatComplex h1 = pilot_h[p1 * active_sc + sc];
    const cuFloatComplex h = cadd(cscale(h0, 1.0f - alpha), cscale(h1, alpha));
    const float h2 = fmaxf(cnorm(h), 1.0e-9f);
    const int fft_bin = shifted_to_natural_bin_dev(used_indices[sc], nfft);
    const cuFloatComplex y = fft_symbols[sym * nfft + fft_bin];
    const cuFloatComplex eq = cscale(cmul(y, cconj(h)), 1.0f / (h2 + noise_var));
    const float sigma2_eq = fmaxf(noise_var / h2, 1.0e-8f);

    for (int bit = 0; bit < bits_per_symbol; ++bit) {
        float d0 = 1.0e30f;
        float d1 = 1.0e30f;
        const int m = 1 << bits_per_symbol;
        for (int s = 0; s < m; ++s) {
            const cuFloatComplex c = qam_symbol_by_index_dev(s, bits_per_symbol);
            const float d2 = cnorm(csub(eq, c));
            if (constellation_bit_dev(s, bit, bits_per_symbol) == 0) {
                d0 = fminf(d0, d2);
            } else {
                d1 = fminf(d1, d2);
            }
        }
        llr[idx * bits_per_symbol + bit] = (d1 - d0) / sigma2_eq;
    }
}

template <class T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { allocate(count); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
    {
        ptr_ = other.ptr_;
        count_ = other.count_;
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    ~DeviceBuffer() { release(); }

    void allocate(size_t count)
    {
        release();
        count_ = count;
        if (count_ > 0) {
            check_cuda(cudaMalloc(&ptr_, count_ * sizeof(T)), "cudaMalloc");
        }
    }

    void release()
    {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            count_ = 0;
        }
    }

    T* get() const { return ptr_; }
    size_t count() const { return count_; }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

} // namespace

enum class GpuModulation : int {
    Bpsk = 1,
    Qpsk = 2,
    Qam16 = 4,
    Qam64 = 6,
};

struct GpuPipelineConfig {
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

struct GpuStaticTables {
    // generator_kn is row-major [code_k][code_n], binary uint8_t.
    std::vector<uint8_t> generator_kn;
    std::vector<int> used_indices;
    std::vector<int> pilot_symbol_index; // size num_symbols; -1 for data symbols.
    std::vector<int> data_symbol_index;  // size num_symbols; -1 for pilot symbols.
    std::vector<int> pilot_symbols;
    std::vector<int> data_symbols;
    std::vector<int> interp_p0;          // size data_symbols.size()
    std::vector<int> interp_p1;          // size data_symbols.size()
    std::vector<float> interp_alpha;     // size data_symbols.size()
    std::vector<cuFloatComplex> pilot_grid; // [pilot_count][active_sc]
    std::vector<cuFloatComplex> preamble;   // preamble_len
    std::vector<cuFloatComplex> precomp_h;  // optional [num_symbols][active_sc]
};

struct GpuSyncResult {
    int start = 0;
    float metric = 0.0f;
    float cfo_hz = 0.0f;
};

class GpuOfdmPipeline {
public:
    explicit GpuOfdmPipeline(GpuPipelineConfig cfg, int slots = 3)
        : cfg_(cfg),
          slots_(std::max(slots, 1))
    {
        if (cfg_.blocks_per_frame <= 0) {
            throw std::runtime_error("blocks_per_frame must be set before constructing GpuOfdmPipeline");
        }
        if (cfg_.coded_bits_per_frame() !=
            cfg_.active_sc * cfg_.data_symbols_per_frame() * cfg_.bits_per_symbol) {
            throw std::runtime_error("coded bits per frame must match active data resource elements");
        }
        if (cfg_.bits_per_symbol != 1 && cfg_.bits_per_symbol != 2 &&
            cfg_.bits_per_symbol != 4 && cfg_.bits_per_symbol != 6) {
            throw std::runtime_error("bits_per_symbol must be 1, 2, 4, or 6");
        }

        check_cuda(cudaStreamCreateWithFlags(&tx_encode_stream_, cudaStreamNonBlocking), "cudaStreamCreate tx_encode");
        check_cuda(cudaStreamCreateWithFlags(&tx_ofdm_stream_, cudaStreamNonBlocking), "cudaStreamCreate tx_ofdm");
        check_cuda(cudaStreamCreateWithFlags(&rx_sync_stream_, cudaStreamNonBlocking), "cudaStreamCreate rx_sync");
        check_cuda(cudaStreamCreateWithFlags(&rx_demod_stream_, cudaStreamNonBlocking), "cudaStreamCreate rx_demod");
        check_cuda(cudaStreamCreateWithFlags(&fec_stream_, cudaStreamNonBlocking), "cudaStreamCreate fec");

        check_cufft(cufftPlan1d(&ifft_plan_, cfg_.nfft, CUFFT_C2C, cfg_.num_symbols), "cufftPlan1d IFFT");
        check_cufft(cufftPlan1d(&fft_plan_, cfg_.nfft, CUFFT_C2C, cfg_.num_symbols), "cufftPlan1d FFT");
        check_cufft(cufftSetStream(ifft_plan_, tx_ofdm_stream_), "cufftSetStream IFFT");
        check_cufft(cufftSetStream(fft_plan_, rx_demod_stream_), "cufftSetStream FFT");

        slot_state_.resize(static_cast<size_t>(slots_));
        for (auto& s : slot_state_) {
            allocate_slot(s);
        }
    }

    GpuOfdmPipeline(const GpuOfdmPipeline&) = delete;
    GpuOfdmPipeline& operator=(const GpuOfdmPipeline&) = delete;

    ~GpuOfdmPipeline()
    {
        if (ifft_plan_ != 0) {
            cufftDestroy(ifft_plan_);
        }
        if (fft_plan_ != 0) {
            cufftDestroy(fft_plan_);
        }
        if (tx_encode_stream_ != nullptr) {
            cudaStreamDestroy(tx_encode_stream_);
        }
        if (tx_ofdm_stream_ != nullptr) {
            cudaStreamDestroy(tx_ofdm_stream_);
        }
        if (rx_sync_stream_ != nullptr) {
            cudaStreamDestroy(rx_sync_stream_);
        }
        if (rx_demod_stream_ != nullptr) {
            cudaStreamDestroy(rx_demod_stream_);
        }
        if (fec_stream_ != nullptr) {
            cudaStreamDestroy(fec_stream_);
        }
    }

    void upload_static_tables(const GpuStaticTables& h)
    {
        if (!h.generator_kn.empty()) {
            require_size(h.generator_kn.size(), static_cast<size_t>(cfg_.code_k * cfg_.code_n), "generator_kn");
        }
        require_size(h.used_indices.size(), static_cast<size_t>(cfg_.active_sc), "used_indices");
        require_size(h.pilot_symbol_index.size(), static_cast<size_t>(cfg_.num_symbols), "pilot_symbol_index");
        require_size(h.data_symbol_index.size(), static_cast<size_t>(cfg_.num_symbols), "data_symbol_index");
        require_size(h.data_symbols.size(), static_cast<size_t>(cfg_.data_symbols_per_frame()), "data_symbols");
        require_size(h.interp_p0.size(), h.data_symbols.size(), "interp_p0");
        require_size(h.interp_p1.size(), h.data_symbols.size(), "interp_p1");
        require_size(h.interp_alpha.size(), h.data_symbols.size(), "interp_alpha");
        require_size(h.preamble.size(), static_cast<size_t>(cfg_.preamble_len), "preamble");
        require_size(h.pilot_grid.size(), h.pilot_symbols.size() * static_cast<size_t>(cfg_.active_sc), "pilot_grid");

        if (!h.generator_kn.empty()) {
            d_generator_.allocate(h.generator_kn.size());
        } else {
            d_generator_.release();
        }
        d_used_indices_.allocate(h.used_indices.size());
        d_pilot_symbol_index_.allocate(h.pilot_symbol_index.size());
        d_data_symbol_index_.allocate(h.data_symbol_index.size());
        d_pilot_symbols_.allocate(h.pilot_symbols.size());
        d_data_symbols_.allocate(h.data_symbols.size());
        d_interp_p0_.allocate(h.interp_p0.size());
        d_interp_p1_.allocate(h.interp_p1.size());
        d_interp_alpha_.allocate(h.interp_alpha.size());
        d_pilot_grid_.allocate(h.pilot_grid.size());
        d_preamble_.allocate(h.preamble.size());

        if (!h.generator_kn.empty()) {
            copy_to_device(d_generator_, h.generator_kn);
        }
        copy_to_device(d_used_indices_, h.used_indices);
        copy_to_device(d_pilot_symbol_index_, h.pilot_symbol_index);
        copy_to_device(d_data_symbol_index_, h.data_symbol_index);
        copy_to_device(d_pilot_symbols_, h.pilot_symbols);
        copy_to_device(d_data_symbols_, h.data_symbols);
        copy_to_device(d_interp_p0_, h.interp_p0);
        copy_to_device(d_interp_p1_, h.interp_p1);
        copy_to_device(d_interp_alpha_, h.interp_alpha);
        copy_to_device(d_pilot_grid_, h.pilot_grid);
        copy_to_device(d_preamble_, h.preamble);

        if (!h.precomp_h.empty()) {
            require_size(h.precomp_h.size(), static_cast<size_t>(cfg_.num_symbols * cfg_.active_sc), "precomp_h");
            d_precomp_h_.allocate(h.precomp_h.size());
            copy_to_device(d_precomp_h_, h.precomp_h);
        }
    }

    // TX stage: host info bits -> GPU LDPC encode -> GPU modulation/grid -> cuFFT
    // inverse -> host-visible complex frame for UHD send(). h_tx_frame should be
    // pinned memory for overlap with the next slot.
    void submit_tx_frame_async(int slot, const uint8_t* h_info_bits, cuFloatComplex* h_tx_frame)
    {
        Slot& s = checked_slot(slot);
        if (d_generator_.get() == nullptr) {
            throw std::runtime_error("GPU TX requested before uploading a generator matrix");
        }
        check_cuda(cudaMemcpyAsync(
                       s.info_bits.get(),
                       h_info_bits,
                       static_cast<size_t>(cfg_.info_bits_per_frame()),
                       cudaMemcpyHostToDevice,
                       tx_encode_stream_),
                   "copy TX info bits");

        const int enc_threads = 256;
        const int enc_blocks = div_up(cfg_.coded_bits_per_frame(), enc_threads);
        ldpc_encode_generator_kernel<<<enc_blocks, enc_threads, 0, tx_encode_stream_>>>(
            s.info_bits.get(),
            d_generator_.get(),
            s.coded_bits.get(),
            cfg_.blocks_per_frame,
            cfg_.code_k,
            cfg_.code_n);
        check_cuda(cudaGetLastError(), "ldpc_encode_generator_kernel");

        check_cuda(cudaEventRecord(s.encode_done, tx_encode_stream_), "record encode_done");
        check_cuda(cudaStreamWaitEvent(tx_ofdm_stream_, s.encode_done, 0), "wait encode_done");
        check_cuda(cudaMemsetAsync(
                       s.tx_grid.get(),
                       0,
                       static_cast<size_t>(cfg_.num_symbols * cfg_.nfft) * sizeof(cuFloatComplex),
                       tx_ofdm_stream_),
                   "clear TX grid");

        const int re_threads = 256;
        const int re_total = cfg_.num_symbols * cfg_.active_sc;
        fill_resource_grid_kernel<<<div_up(re_total, re_threads), re_threads, 0, tx_ofdm_stream_>>>(
            s.coded_bits.get(),
            d_used_indices_.get(),
            d_pilot_symbol_index_.get(),
            d_data_symbol_index_.get(),
            d_pilot_grid_.get(),
            d_precomp_h_.get(),
            s.tx_grid.get(),
            cfg_.nfft,
            cfg_.num_symbols,
            cfg_.active_sc,
            cfg_.bits_per_symbol);
        check_cuda(cudaGetLastError(), "fill_resource_grid_kernel");

        check_cufft(cufftExecC2C(
                        ifft_plan_,
                        reinterpret_cast<cufftComplex*>(s.tx_grid.get()),
                        reinterpret_cast<cufftComplex*>(s.tx_ifft.get()),
                        CUFFT_INVERSE),
                    "cufftExecC2C IFFT");

        const int frame_threads = 256;
        compose_tx_frame_kernel<<<div_up(cfg_.frame_samples(), frame_threads), frame_threads, 0, tx_ofdm_stream_>>>(
            d_preamble_.get(),
            s.tx_ifft.get(),
            s.tx_frame.get(),
            cfg_.preamble_len,
            cfg_.nfft,
            cfg_.cp,
            cfg_.num_symbols,
            cfg_.tx_amplitude);
        check_cuda(cudaGetLastError(), "compose_tx_frame_kernel");

        check_cuda(cudaMemcpyAsync(
                       h_tx_frame,
                       s.tx_frame.get(),
                       static_cast<size_t>(cfg_.frame_samples()) * sizeof(cuFloatComplex),
                       cudaMemcpyDeviceToHost,
                       tx_ofdm_stream_),
                   "copy TX frame to host");
        check_cuda(cudaEventRecord(s.tx_done, tx_ofdm_stream_), "record tx_done");
    }

    // RX sync stage: host samples -> GPU Schmidl-Cox search. The result copy is
    // asynchronous; synchronize rx_sync_stream() or the slot event before reading.
    void submit_rx_sync_async(int slot, const cuFloatComplex* h_rx_samples, int sample_count, GpuSyncResult* h_result)
    {
        Slot& s = checked_slot(slot);
        if (sample_count > s.max_rx_samples) {
            throw std::runtime_error("RX sample_count exceeds slot allocation");
        }
        const int candidates = sample_count - cfg_.preamble_len + 1;
        if (candidates <= 0) {
            throw std::runtime_error("RX sample_count is shorter than the preamble");
        }
        if (candidates > s.sync_metric.count()) {
            throw std::runtime_error("RX candidates exceed sync metric allocation");
        }

        check_cuda(cudaMemcpyAsync(
                       s.rx_samples.get(),
                       h_rx_samples,
                       static_cast<size_t>(sample_count) * sizeof(cuFloatComplex),
                       cudaMemcpyHostToDevice,
                       rx_sync_stream_),
                   "copy RX samples");
        schmidl_cox_metric_kernel<<<candidates, kSyncThreads, 0, rx_sync_stream_>>>(
            s.rx_samples.get(),
            s.sync_metric.get(),
            s.sync_corr.get(),
            candidates,
            cfg_.pre_half_len);
        check_cuda(cudaGetLastError(), "schmidl_cox_metric_kernel");

        argmax_metric_kernel<<<1, kSyncThreads, 0, rx_sync_stream_>>>(
            s.sync_metric.get(),
            s.sync_corr.get(),
            s.sync_result.get(),
            candidates,
            cfg_.pre_half_len,
            cfg_.sample_rate);
        check_cuda(cudaGetLastError(), "argmax_metric_kernel");

        check_cuda(cudaMemcpyAsync(
                       h_result,
                       s.sync_result.get(),
                       sizeof(GpuSyncResult),
                       cudaMemcpyDeviceToHost,
                       rx_sync_stream_),
                   "copy sync result");
        check_cuda(cudaEventRecord(s.sync_done, rx_sync_stream_), "record sync_done");
    }

    // RX demod stage: consumes the slot's sync result on device and produces
    // device LLRs. A device-pointer BP/OSD backend can wait on fec_stream().
    const float* submit_rx_demod_llr_async(int slot)
    {
        Slot& s = checked_slot(slot);
        check_cuda(cudaStreamWaitEvent(rx_demod_stream_, s.sync_done, 0), "wait sync_done");

        const int fft_threads = 256;
        const int fft_total = cfg_.num_symbols * cfg_.nfft;
        cfo_strip_cp_kernel<<<div_up(fft_total, fft_threads), fft_threads, 0, rx_demod_stream_>>>(
            s.rx_samples.get(),
            s.sync_result.get(),
            s.rx_fft_in.get(),
            cfg_.preamble_len,
            cfg_.nfft,
            cfg_.cp,
            cfg_.num_symbols,
            cfg_.sample_rate);
        check_cuda(cudaGetLastError(), "cfo_strip_cp_kernel");

        check_cufft(cufftExecC2C(
                        fft_plan_,
                        reinterpret_cast<cufftComplex*>(s.rx_fft_in.get()),
                        reinterpret_cast<cufftComplex*>(s.rx_fft_out.get()),
                        CUFFT_FORWARD),
                    "cufftExecC2C FFT");

        const int pilot_count = static_cast<int>(d_pilot_symbols_.count());
        const int pilot_total = pilot_count * cfg_.active_sc;
        estimate_pilot_channel_kernel<<<div_up(pilot_total, fft_threads), fft_threads, 0, rx_demod_stream_>>>(
            s.rx_fft_out.get(),
            d_used_indices_.get(),
            d_pilot_symbols_.get(),
            d_pilot_grid_.get(),
            s.pilot_h.get(),
            cfg_.nfft,
            pilot_count,
            cfg_.active_sc);
        check_cuda(cudaGetLastError(), "estimate_pilot_channel_kernel");

        const int data_total = cfg_.data_symbols_per_frame() * cfg_.active_sc;
        equalize_llr_kernel<<<div_up(data_total, fft_threads), fft_threads, 0, rx_demod_stream_>>>(
            s.rx_fft_out.get(),
            s.pilot_h.get(),
            d_used_indices_.get(),
            d_data_symbols_.get(),
            d_interp_p0_.get(),
            d_interp_p1_.get(),
            d_interp_alpha_.get(),
            s.llr.get(),
            cfg_.nfft,
            cfg_.data_symbols_per_frame(),
            cfg_.active_sc,
            cfg_.bits_per_symbol,
            cfg_.demod_noise_var);
        check_cuda(cudaGetLastError(), "equalize_llr_kernel");

        check_cuda(cudaEventRecord(s.llr_done, rx_demod_stream_), "record llr_done");
        check_cuda(cudaStreamWaitEvent(fec_stream_, s.llr_done, 0), "wait llr_done");
        return s.llr.get();
    }

    cudaStream_t tx_stream() const { return tx_ofdm_stream_; }
    cudaStream_t rx_sync_stream() const { return rx_sync_stream_; }
    cudaStream_t rx_demod_stream() const { return rx_demod_stream_; }
    cudaStream_t fec_stream() const { return fec_stream_; }

    void synchronize_tx(int slot)
    {
        Slot& s = checked_slot(slot);
        check_cuda(cudaEventSynchronize(s.tx_done), "synchronize tx_done");
    }

    void synchronize_rx_sync(int slot)
    {
        Slot& s = checked_slot(slot);
        check_cuda(cudaEventSynchronize(s.sync_done), "synchronize sync_done");
    }

    void synchronize_rx_llr(int slot)
    {
        Slot& s = checked_slot(slot);
        check_cuda(cudaEventSynchronize(s.llr_done), "synchronize llr_done");
    }

private:
    struct Slot {
        DeviceBuffer<uint8_t> info_bits;
        DeviceBuffer<uint8_t> coded_bits;
        DeviceBuffer<cuFloatComplex> tx_grid;
        DeviceBuffer<cuFloatComplex> tx_ifft;
        DeviceBuffer<cuFloatComplex> tx_frame;

        int max_rx_samples = 0;
        DeviceBuffer<cuFloatComplex> rx_samples;
        DeviceBuffer<float> sync_metric;
        DeviceBuffer<cuFloatComplex> sync_corr;
        DeviceBuffer<DeviceSyncResult> sync_result;
        DeviceBuffer<cuFloatComplex> rx_fft_in;
        DeviceBuffer<cuFloatComplex> rx_fft_out;
        DeviceBuffer<cuFloatComplex> pilot_h;
        DeviceBuffer<float> llr;

        cudaEvent_t encode_done = nullptr;
        cudaEvent_t tx_done = nullptr;
        cudaEvent_t sync_done = nullptr;
        cudaEvent_t llr_done = nullptr;

        ~Slot()
        {
            if (encode_done != nullptr) {
                cudaEventDestroy(encode_done);
            }
            if (tx_done != nullptr) {
                cudaEventDestroy(tx_done);
            }
            if (sync_done != nullptr) {
                cudaEventDestroy(sync_done);
            }
            if (llr_done != nullptr) {
                cudaEventDestroy(llr_done);
            }
        }

        Slot() = default;
        Slot(const Slot&) = delete;
        Slot& operator=(const Slot&) = delete;
        Slot(Slot&& other) noexcept
            : info_bits(std::move(other.info_bits)),
              coded_bits(std::move(other.coded_bits)),
              tx_grid(std::move(other.tx_grid)),
              tx_ifft(std::move(other.tx_ifft)),
              tx_frame(std::move(other.tx_frame)),
              max_rx_samples(other.max_rx_samples),
              rx_samples(std::move(other.rx_samples)),
              sync_metric(std::move(other.sync_metric)),
              sync_corr(std::move(other.sync_corr)),
              sync_result(std::move(other.sync_result)),
              rx_fft_in(std::move(other.rx_fft_in)),
              rx_fft_out(std::move(other.rx_fft_out)),
              pilot_h(std::move(other.pilot_h)),
              llr(std::move(other.llr)),
              encode_done(other.encode_done),
              tx_done(other.tx_done),
              sync_done(other.sync_done),
              llr_done(other.llr_done)
        {
            other.max_rx_samples = 0;
            other.encode_done = nullptr;
            other.tx_done = nullptr;
            other.sync_done = nullptr;
            other.llr_done = nullptr;
        }

        Slot& operator=(Slot&& other) noexcept
        {
            if (this != &other) {
                if (encode_done != nullptr) {
                    cudaEventDestroy(encode_done);
                }
                if (tx_done != nullptr) {
                    cudaEventDestroy(tx_done);
                }
                if (sync_done != nullptr) {
                    cudaEventDestroy(sync_done);
                }
                if (llr_done != nullptr) {
                    cudaEventDestroy(llr_done);
                }
                info_bits = std::move(other.info_bits);
                coded_bits = std::move(other.coded_bits);
                tx_grid = std::move(other.tx_grid);
                tx_ifft = std::move(other.tx_ifft);
                tx_frame = std::move(other.tx_frame);
                max_rx_samples = other.max_rx_samples;
                rx_samples = std::move(other.rx_samples);
                sync_metric = std::move(other.sync_metric);
                sync_corr = std::move(other.sync_corr);
                sync_result = std::move(other.sync_result);
                rx_fft_in = std::move(other.rx_fft_in);
                rx_fft_out = std::move(other.rx_fft_out);
                pilot_h = std::move(other.pilot_h);
                llr = std::move(other.llr);
                encode_done = other.encode_done;
                tx_done = other.tx_done;
                sync_done = other.sync_done;
                llr_done = other.llr_done;
                other.max_rx_samples = 0;
                other.encode_done = nullptr;
                other.tx_done = nullptr;
                other.sync_done = nullptr;
                other.llr_done = nullptr;
            }
            return *this;
        }
    };

    static int div_up(int a, int b) { return (a + b - 1) / b; }

    static void require_size(size_t actual, size_t expected, const char* name)
    {
        if (actual != expected) {
            throw std::runtime_error(std::string(name) + " has wrong size; expected " +
                                     std::to_string(expected) + ", got " + std::to_string(actual));
        }
    }

    template <class T>
    static void copy_to_device(DeviceBuffer<T>& dst, const std::vector<T>& src)
    {
        if (src.empty()) {
            return;
        }
        check_cuda(cudaMemcpy(dst.get(), src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice), "copy_to_device");
    }

    Slot& checked_slot(int slot)
    {
        if (slot < 0 || slot >= slots_) {
            throw std::runtime_error("invalid GPU pipeline slot");
        }
        return slot_state_[static_cast<size_t>(slot)];
    }

    void allocate_slot(Slot& s)
    {
        s.info_bits.allocate(static_cast<size_t>(cfg_.info_bits_per_frame()));
        s.coded_bits.allocate(static_cast<size_t>(cfg_.coded_bits_per_frame()));
        s.tx_grid.allocate(static_cast<size_t>(cfg_.num_symbols * cfg_.nfft));
        s.tx_ifft.allocate(static_cast<size_t>(cfg_.num_symbols * cfg_.nfft));
        s.tx_frame.allocate(static_cast<size_t>(cfg_.frame_samples()));

        s.max_rx_samples = cfg_.max_rx_search_samples > 0 ? cfg_.max_rx_search_samples : cfg_.frame_samples();
        s.rx_samples.allocate(static_cast<size_t>(s.max_rx_samples));
        const int max_candidates = s.max_rx_samples - cfg_.preamble_len + 1;
        s.sync_metric.allocate(static_cast<size_t>(std::max(max_candidates, 1)));
        s.sync_corr.allocate(static_cast<size_t>(std::max(max_candidates, 1)));
        s.sync_result.allocate(1);
        s.rx_fft_in.allocate(static_cast<size_t>(cfg_.num_symbols * cfg_.nfft));
        s.rx_fft_out.allocate(static_cast<size_t>(cfg_.num_symbols * cfg_.nfft));
        s.pilot_h.allocate(static_cast<size_t>(cfg_.num_symbols * cfg_.active_sc));
        s.llr.allocate(static_cast<size_t>(cfg_.coded_bits_per_frame()));

        check_cuda(cudaEventCreateWithFlags(&s.encode_done, cudaEventDisableTiming), "create encode_done");
        check_cuda(cudaEventCreateWithFlags(&s.tx_done, cudaEventDisableTiming), "create tx_done");
        check_cuda(cudaEventCreateWithFlags(&s.sync_done, cudaEventDisableTiming), "create sync_done");
        check_cuda(cudaEventCreateWithFlags(&s.llr_done, cudaEventDisableTiming), "create llr_done");
    }

    GpuPipelineConfig cfg_;
    int slots_ = 0;
    std::vector<Slot> slot_state_;

    DeviceBuffer<uint8_t> d_generator_;
    DeviceBuffer<int> d_used_indices_;
    DeviceBuffer<int> d_pilot_symbol_index_;
    DeviceBuffer<int> d_data_symbol_index_;
    DeviceBuffer<int> d_pilot_symbols_;
    DeviceBuffer<int> d_data_symbols_;
    DeviceBuffer<int> d_interp_p0_;
    DeviceBuffer<int> d_interp_p1_;
    DeviceBuffer<float> d_interp_alpha_;
    DeviceBuffer<cuFloatComplex> d_pilot_grid_;
    DeviceBuffer<cuFloatComplex> d_preamble_;
    DeviceBuffer<cuFloatComplex> d_precomp_h_;

    cudaStream_t tx_encode_stream_ = nullptr;
    cudaStream_t tx_ofdm_stream_ = nullptr;
    cudaStream_t rx_sync_stream_ = nullptr;
    cudaStream_t rx_demod_stream_ = nullptr;
    cudaStream_t fec_stream_ = nullptr;
    cufftHandle ifft_plan_ = 0;
    cufftHandle fft_plan_ = 0;
};

} // namespace usrp_gpu_pipeline

namespace usrp_gpu_pipeline_bridge {

namespace {

static_assert(sizeof(Complex32) == sizeof(cuFloatComplex), "Complex32 must match cuFloatComplex size");

inline void check_cuda_bridge(cudaError_t status, const char* what)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

usrp_gpu_pipeline::GpuPipelineConfig to_device_config(const Config& cfg)
{
    usrp_gpu_pipeline::GpuPipelineConfig out;
    out.nfft = cfg.nfft;
    out.cp = cfg.cp;
    out.preamble_len = cfg.preamble_len;
    out.pre_half_len = cfg.pre_half_len;
    out.num_symbols = cfg.num_symbols;
    out.active_sc = cfg.active_sc;
    out.bits_per_symbol = cfg.bits_per_symbol;
    out.code_n = cfg.code_n;
    out.code_k = cfg.code_k;
    out.blocks_per_frame = cfg.blocks_per_frame;
    out.max_rx_search_samples = cfg.max_rx_search_samples;
    out.sample_rate = cfg.sample_rate;
    out.tx_amplitude = cfg.tx_amplitude;
    out.demod_noise_var = cfg.demod_noise_var;
    return out;
}

std::vector<cuFloatComplex> to_cu_complex_vector(const std::vector<Complex32>& src)
{
    std::vector<cuFloatComplex> out(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        out[i] = make_cuFloatComplex(src[i].real(), src[i].imag());
    }
    return out;
}

usrp_gpu_pipeline::GpuStaticTables to_device_tables(const StaticTables& tables)
{
    usrp_gpu_pipeline::GpuStaticTables out;
    out.generator_kn = tables.generator_kn;
    out.used_indices = tables.used_indices;
    out.pilot_symbol_index = tables.pilot_symbol_index;
    out.data_symbol_index = tables.data_symbol_index;
    out.pilot_symbols = tables.pilot_symbols;
    out.data_symbols = tables.data_symbols;
    out.interp_p0 = tables.interp_p0;
    out.interp_p1 = tables.interp_p1;
    out.interp_alpha = tables.interp_alpha;
    out.pilot_grid = to_cu_complex_vector(tables.pilot_grid);
    out.preamble = to_cu_complex_vector(tables.preamble);
    out.precomp_h = to_cu_complex_vector(tables.precomp_h);
    return out;
}

SyncResult to_bridge_sync(const usrp_gpu_pipeline::GpuSyncResult& r)
{
    SyncResult out;
    out.start = r.start;
    out.metric = r.metric;
    out.cfo_hz = r.cfo_hz;
    return out;
}

} // namespace

struct Pipeline::Impl {
    explicit Impl(const Config& cfg, int slots)
        : cfg(cfg),
          device(to_device_config(cfg), slots)
    {}

    Config cfg;
    usrp_gpu_pipeline::GpuOfdmPipeline device;
};

Pipeline::Pipeline(const Config& cfg, int slots)
    : impl_(std::make_unique<Impl>(cfg, slots))
{}

Pipeline::~Pipeline() = default;

void Pipeline::upload_static_tables(const StaticTables& tables)
{
    impl_->device.upload_static_tables(to_device_tables(tables));
}

void Pipeline::submit_tx_frame_async(int slot, const uint8_t* h_info_bits, Complex32* h_tx_frame)
{
    impl_->device.submit_tx_frame_async(
        slot,
        h_info_bits,
        reinterpret_cast<cuFloatComplex*>(h_tx_frame));
}

void Pipeline::synchronize_tx(int slot)
{
    impl_->device.synchronize_tx(slot);
}

SyncResult Pipeline::find_frame(int slot, const Complex32* h_samples, int sample_count)
{
    usrp_gpu_pipeline::GpuSyncResult result;
    impl_->device.submit_rx_sync_async(
        slot,
        reinterpret_cast<const cuFloatComplex*>(h_samples),
        sample_count,
        &result);
    impl_->device.synchronize_rx_sync(slot);
    return to_bridge_sync(result);
}

const float* Pipeline::demod_frame_to_device_llr(int slot, const Complex32* h_frame, SyncResult* sync_result)
{
    usrp_gpu_pipeline::GpuSyncResult result;
    impl_->device.submit_rx_sync_async(
        slot,
        reinterpret_cast<const cuFloatComplex*>(h_frame),
        impl_->cfg.frame_samples(),
        &result);
    impl_->device.synchronize_rx_sync(slot);
    const float* d_llr = impl_->device.submit_rx_demod_llr_async(slot);
    check_cuda_bridge(cudaStreamSynchronize(impl_->device.rx_demod_stream()), "synchronize RX demod device LLR");
    if (sync_result != nullptr) {
        *sync_result = to_bridge_sync(result);
    }
    return d_llr;
}

SyncResult Pipeline::demod_frame_to_host_llr(int slot, const Complex32* h_frame, float* h_llr)
{
    SyncResult sync;
    const float* d_llr = demod_frame_to_device_llr(slot, h_frame, &sync);
    check_cuda_bridge(cudaMemcpyAsync(
                          h_llr,
                          d_llr,
                          static_cast<size_t>(impl_->cfg.coded_bits_per_frame()) * sizeof(float),
                          cudaMemcpyDeviceToHost,
                   impl_->device.rx_demod_stream()),
                      "copy RX LLRs to host");
    check_cuda_bridge(cudaStreamSynchronize(impl_->device.rx_demod_stream()), "synchronize RX demod LLR copy");
    return sync;
}

} // namespace usrp_gpu_pipeline_bridge
