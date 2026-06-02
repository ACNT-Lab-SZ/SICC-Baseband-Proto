#pragma once

#ifdef _WIN32
#if defined(CUDA_OSD_MATLAB_EXPORTS)
#define CUDA_OSD_MATLAB_API __declspec(dllexport)
#else
#define CUDA_OSD_MATLAB_API __declspec(dllimport)
#endif
#else
#define CUDA_OSD_MATLAB_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cuda_osd_stream_config_t {
    int n;
    int k;
    int m;
    int max_batch_size;
    int min_batch_size;
    unsigned long long max_latency_us;
    double arrival_ewma_alpha;
    unsigned long long default_interarrival_us;
    int bp_max_iterations;
    float bp_normalization;
    float bp_offset;
    float bp_damping;
    float bp_min_abs_llr_accept;
    int osd_threads;
} cuda_osd_stream_config_t;

typedef struct cuda_osd_osd_config_t {
    int n;
    int k;
    int max_batch_size;
    int osd_threads;
} cuda_osd_osd_config_t;

typedef struct cuda_osd_bp_csr_config_t {
    int n;
    int m;
    int max_batch_size;
    int max_iterations;
    float normalization;
    float offset;
    float damping;
    float min_abs_llr_accept;
    int nnz;
} cuda_osd_bp_csr_config_t;

typedef struct cuda_osd_stream_stats_t {
    unsigned long long submitted_codewords;
    unsigned long long decoded_codewords;
    unsigned long long bp_success_codewords;
    unsigned long long osd_fallback_codewords;
    unsigned long long decoded_batches;
    unsigned long long current_pending_codewords;
    unsigned long long max_pending_codewords;
    unsigned long long last_submit_timestamp_us;
    double ewma_interarrival_us;
    double average_batch_size;
    double total_decode_ms;
} cuda_osd_stream_stats_t;

enum {
    CUDA_OSD_STATUS_OK = 0,
    CUDA_OSD_STATUS_INVALID_ARGUMENT = 1,
    CUDA_OSD_STATUS_INTERNAL_ERROR = 2
};

CUDA_OSD_MATLAB_API int cuda_osd_create_stream_decoder(const cuda_osd_stream_config_t* config,
                                                       unsigned long long* out_handle);
CUDA_OSD_MATLAB_API int cuda_osd_destroy_stream_decoder(unsigned long long handle);
CUDA_OSD_MATLAB_API int cuda_osd_create_osd_decoder(const cuda_osd_osd_config_t* config,
                                                    unsigned long long* out_handle);
CUDA_OSD_MATLAB_API int cuda_osd_destroy_osd_decoder(unsigned long long handle);
CUDA_OSD_MATLAB_API int cuda_osd_create_bp_csr_decoder(const cuda_osd_bp_csr_config_t* config,
                                                       const int* row_ptr,
                                                       const int* col_ind,
                                                       unsigned long long* out_handle);
CUDA_OSD_MATLAB_API int cuda_osd_destroy_bp_csr_decoder(unsigned long long handle);
CUDA_OSD_MATLAB_API int cuda_osd_decode_bp_csr_batch(unsigned long long handle,
                                                     const float* llrs,
                                                     int codewords,
                                                     unsigned char* out_code_bits,
                                                     unsigned char* out_success,
                                                     double* out_total_decode_ms);
CUDA_OSD_MATLAB_API int cuda_osd_decode_osd_batch(unsigned long long handle,
                                                  const float* llrs,
                                                  int codewords,
                                                  unsigned char* out_info_bits,
                                                  float* out_distances,
                                                  double* out_total_decode_ms,
                                                  double* out_throughput_mbps);
CUDA_OSD_MATLAB_API int cuda_osd_submit_codewords(unsigned long long handle,
                                                  const float* llrs,
                                                  int codewords,
                                                  const unsigned long long* timestamps_us,
                                                  const unsigned long long* frame_ids,
                                                  const unsigned int* block_indices);
CUDA_OSD_MATLAB_API int cuda_osd_submit_device_codewords(unsigned long long handle,
                                                         const float* device_llrs,
                                                         int codewords,
                                                         const unsigned long long* timestamps_us,
                                                         const unsigned long long* frame_ids,
                                                         const unsigned int* block_indices);
CUDA_OSD_MATLAB_API int cuda_osd_poll_ready(unsigned long long handle,
                                            int max_codewords,
                                            unsigned char* out_info_bits,
                                            unsigned long long* out_timestamps_us,
                                            unsigned long long* out_frame_ids,
                                            unsigned int* out_block_indices,
                                            unsigned char* out_used_osd,
                                            int* out_codewords);
CUDA_OSD_MATLAB_API int cuda_osd_flush(unsigned long long handle);
CUDA_OSD_MATLAB_API int cuda_osd_get_stats(unsigned long long handle,
                                           cuda_osd_stream_stats_t* out_stats);
CUDA_OSD_MATLAB_API unsigned long long cuda_osd_get_decoded_codewords(unsigned long long handle);
CUDA_OSD_MATLAB_API double cuda_osd_get_total_decode_ms(unsigned long long handle);
CUDA_OSD_MATLAB_API int cuda_osd_find_frame(const double* signal_real,
                                            const double* signal_imag,
                                            int signal_length,
                                            const double* preamble_real,
                                            const double* preamble_imag,
                                            int preamble_length,
                                            int pre_half_len,
                                            double sync_peak_thresh,
                                            double fs,
                                            int* out_start_idx_1based,
                                            double* out_peak,
                                            double* out_cfo_hz);
CUDA_OSD_MATLAB_API const char* cuda_osd_get_last_error(void);

#ifdef __cplusplus
}
#endif
